"""
Phase 0 step 4: base-model baseline on the fixed validation set.

Same multi-turn loop as agent_eval.py (propose kernel.py -> bench -> feedback -> revise; every
turn's last ```python block is benchmarked), with:
  * prompt v2 (prompts_v2.py: consistent fenced-block protocol, otherwise v1 wording),
  * the held-out problems of results/v3/val_set/ (val_set_v1.py), not the 9 training prompts,
  * bench = autokernel-val-bench (val_bench_core.py, bench v2 conventions, no cache).
agent_eval.py and its results are untouched.

    modal deploy modal_app/val_bench_modal.py
    modal run modal_app/val_eval.py --n 2 --max-turns 4 --tag half1
    python modal_app/val_eval.py summarize half1,half2      # -> results/v3/phase0_baseline.json
    modal run modal_app/val_eval.py --n 2 --tag s1 --lora /vol/sft/sft_v1/step_378 --out-dir phase3_runs
    python modal_app/val_eval.py summarize phase3_runs/s1,phase3_runs/s2 phase3_sft_v1   # -> results/v3/phase3_sft_v1.json

Concurrency: 1 policy GPU (L40S by default) + <= 6 bench L4 containers.
"""
import ast
import hashlib
import json
import os
import pathlib
import re
import statistics
import sys
import time

import modal

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prompts_v2 import OBS_LABEL_V2, SYSTEM_PROMPT_V2, user_prompt_v2  # noqa: E402

V3 = ROOT / "results" / "v3"
VAL = V3 / "val_set"
app = modal.App("autokernel-val-eval")

MAX_CTX = 32768
MAX_NEW = 4096
_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text):
    blocks = _CODE.findall(text)
    return blocks[-1] if blocks else None


def ast_key(code):
    """Normalised AST (comments/docstrings/formatting stripped) for starter-resubmit detection."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()


def load_tasks(max_turns, ids=None):
    meta = json.loads((V3 / "val_set.json").read_text())
    tasks = []
    for p in meta["problems"]:
        if ids and p["id"] not in ids:
            continue
        d = VAL / p["dir"]
        spec = json.loads((d / "problem.json").read_text())
        ref, starter = (d / "reference.py").read_text(), (d / "starter.py").read_text()
        tl_, size, tdt = spec["timing"][0]
        kt = f"{spec['family']} (variant: {spec['variant']} -- {spec['desc']})"
        dtype = f"{tdt} (timed); correctness is checked on {', '.join(spec['dtypes'])}"
        tasks.append({"task_id": spec["id"], "family": spec["family"], "spec": spec, "reference_code": ref,
                      "starter_code": starter, "starter_ast": ast_key(starter), "messages": [
                          {"role": "system", "content": SYSTEM_PROMPT_V2},
                          {"role": "user", "content": user_prompt_v2(kt, size, dtype, ref, starter, max_turns)}]})
    return tasks


def load_baselines(path=V3 / "val_set_starters.jsonl"):
    """problem_id -> {compile_over_starter, starter_vs_eager, starter_us, eager_us, compile_us} (medians)."""
    by = {}
    if not path.exists():
        return by
    for line in open(path):
        r = json.loads(line)
        for lab, t in (r.get("timing") or {}).items():
            if isinstance(t, dict) and "compile_over_starter" in t:
                by.setdefault(r["problem_id"], []).append(t)
    out = {}
    for pid, ts in by.items():
        out[pid] = {"compile_over_starter": statistics.median(t["compile_over_starter"]["median"] for t in ts),
                    "starter_vs_eager": statistics.median(t["starter_vs_eager"]["median"] for t in ts),
                    "starter_us": statistics.median(t["starter"]["median_us"] for t in ts),
                    "eager_us": statistics.median(t["eager"]["median_us"] for t in ts),
                    "compile_us": statistics.median(t["compile"]["median_us"] for t in ts)}
    return out


def timing_of(res):
    return next((v for v in (res.get("timing") or {}).values() if isinstance(v, dict) and "kernel" in v), None)


def observation(res, base):
    v = res.get("verdict") or "CRASH"
    keep = {"correctness": v}
    t = timing_of(res) if v == "PASS" else None
    if t:
        vs = t.get("vs_starter", {}).get("median")
        keep.update({"speedup_vs_starter": round(vs, 3) if vs else None,
                     "speedup_vs_pytorch": round(t["vs_eager"]["median"], 3),
                     "speedup_vs_torch_compile": round(vs * base["compile_over_starter"], 3) if (vs and base) else None,
                     "latency_us": round(t["kernel"]["median_us"], 1),
                     "starter_latency_us": round(t["starter"]["median_us"], 1) if "starter" in t else None,
                     "pytorch_latency_us": round(t["eager"]["median_us"], 1)})
    msg = OBS_LABEL_V2 + "\n" + json.dumps({k: x for k, x in keep.items() if x is not None})
    if v != "PASS":
        lines = list(res.get("fail_reasons") or [])
        if res.get("reason") and res["reason"] not in " ".join(lines):
            lines.insert(0, res["reason"])
        if lines:
            msg += "\n\nfailures:\n" + "\n".join(l[:1000] for l in lines[:6])
    return msg


def category(turn):
    """format | pass | python | compile | runtime_resources | runtime | numerics | no_triton | timeout | infra."""
    if not (turn["code"] or "").strip():   # no fenced block, or an empty one
        return "format"
    v, r = turn["verdict"], turn.get("reason") or ""
    fr = " ".join(turn.get("fail_reasons") or [])
    if v == "PASS":
        return "pass"
    if v == "TIMEOUT" or "TIMEOUT" in fr:
        return "timeout"
    if v in ("INFRA_ERROR",):
        return "infra"
    et, where = turn.get("exc_type") or "", turn.get("exc_where") or ""
    if v == "CRASH":
        if r.startswith("import:"):
            return "python"
        if et == "OutOfResources" or "OutOfResources" in r:
            return "runtime_resources"
        if "fatal CUDA" in r or "illegal memory" in r or "CUDA error" in r:
            return "runtime"
        if et == "CompilationError" or where == "triton_compile":
            return "compile"
        return "python"
    if "no Triton kernel launched" in fr:
        return "no_triton"
    if "EXC " in fr:  # exception on a later case
        if "OutOfResources" in fr:
            return "runtime_resources"
        if "CompilationError" in fr:
            return "compile"
        if "CUDA error" in fr or "illegal memory" in fr:
            return "runtime"
        return "python"
    return "numerics"


def norm_msg(turn):
    c = category(turn)
    if c in ("numerics", "pass"):
        return c
    m = (turn.get("reason") or " ".join(turn.get("fail_reasons") or [])).split(" [...] ")[-1]
    m = re.sub(r"\d+", "N", m)[-200:]
    return c + "|" + m


def run(tasks, n, max_turns, gen_fn, bench, base, log=print):
    trajs = [{"task_id": t["task_id"], "family": t["family"], "sample": s, "turns": [], "done": False,
              "messages": [dict(m) for m in t["messages"]], "_t": t} for t in tasks for s in range(n)]
    for turn in range(max_turns):
        active = [t for t in trajs if not t["done"]]
        if not active:
            break
        t0 = time.time()
        gens = gen_fn([t["messages"] for t in active])
        t_gen = time.time() - t0
        codes = []
        for t, g in zip(active, gens):
            t["messages"].append({"role": "assistant", "content": g["text"]})
            t["_gen"] = g
            codes.append(extract_code(g["text"]))
        idx = [i for i, c in enumerate(codes) if c]
        jobs = [{"id": f"{active[i]['task_id']}|s{active[i]['sample']}|t{turn}", "spec": active[i]["_t"]["spec"],
                 "reference_code": active[i]["_t"]["reference_code"], "starter_code": active[i]["_t"]["starter_code"],
                 "code": codes[i], "timing": True, "compile": False, "reps": 40, "early_abort": True} for i in idx]
        t0 = time.time()
        res = dict(zip(idx, list(bench.run_one.map(jobs, return_exceptions=True, wrap_returned_exceptions=False))))
        t_bench = time.time() - t0
        for i, t in enumerate(active):
            r = res.get(i)
            if r is None:
                r = {"verdict": "CRASH", "reason": "no ```python code block in reply"}
            elif isinstance(r, Exception):
                r = {"verdict": "INFRA_ERROR", "reason": repr(r)[:300]}
            g = t.pop("_gen")
            b = base.get(t["task_id"])
            tm = timing_of(r) if r.get("verdict") == "PASS" else None
            vs = (tm or {}).get("vs_starter", {})
            rec = {"turn": turn, "verdict": r.get("verdict"), "reason": (r.get("reason") or "")[:1500],
                   "fail_reasons": [x[:600] for x in (r.get("fail_reasons") or [])],
                   "exc_type": r.get("exc_type"), "exc_where": r.get("exc_where"),
                   "triton_launches": (r.get("correctness") or {}).get("triton_launches"),
                   "n_cases_run": (r.get("correctness") or {}).get("n_run"),
                   "vs_starter": vs.get("median"), "vs_starter_ci": [vs.get("ci_lo"), vs.get("ci_hi")] if vs else None,
                   "vs_eager": (tm or {}).get("vs_eager", {}).get("median"),
                   "vs_compile": (vs.get("median") * b["compile_over_starter"]) if (vs and b) else None,
                   "latency_us": (tm or {}).get("kernel", {}).get("median_us"),
                   "bench_wall_s": r.get("wall_s"), "prompt_tokens": g["prompt_tokens"],
                   "completion_tokens": g["completion_tokens"], "finish": g["finish"],
                   "code": codes[i], "code_sha": hashlib.sha256((codes[i] or "").encode()).hexdigest()[:12],
                   "is_starter": bool(codes[i]) and ast_key(codes[i]) == t["_t"]["starter_ast"]}
            rec["category"] = category(rec)
            t["turns"].append(rec)
            t["messages"].append({"role": "user", "content": observation(r, b)})
            if g["prompt_tokens"] + 2 * MAX_NEW > MAX_CTX:
                t["done"] = True
        cats = {}
        for t in active:
            c = t["turns"][-1]["category"]
            cats[c] = cats.get(c, 0) + 1
        log(f"turn {turn}: {len(active)} active, gen {t_gen:.0f}s, bench {t_bench:.0f}s "
            f"(sum job wall {sum((t['turns'][-1]['bench_wall_s'] or 0) for t in active):.0f}s), {cats}", flush=True)
    for t in trajs:
        t.pop("done")
        t.pop("_t")
    return trajs


@app.local_entrypoint()
def main(n: int = 2, max_turns: int = 4, tag: str = "half1", model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
         policy_gpu: str = "L40S", temperature: float = 0.8, ids: str = "", keep_messages: bool = True,
         lora: str = "", out_dir: str = "phase0_runs"):
    """lora: adapter dir on the autokernel-rlvr Volume (e.g. /vol/sft/sft_v1/step_378); out_dir under results/v3/."""
    tasks = load_tasks(max_turns, set(ids.split(",")) if ids else None)
    base = load_baselines()
    missing = [t["task_id"] for t in tasks if t["task_id"] not in base]
    print(len(tasks), "tasks; baselines missing for", missing, flush=True)
    Policy = modal.Cls.from_name("autokernel-policy", "Policy").with_options(gpu=policy_gpu)
    pol = Policy(model=model, enable_lora=bool(lora))
    bench = modal.Cls.from_name("autokernel-val-bench", "ValBench")()

    def gen_fn(convs):
        outs = pol.generate.remote(convs, n=1, temperature=temperature, max_tokens=MAX_NEW,
                                   lora_path=lora, lora_id=1 if lora else 0)
        return [o[0] for o in outs]

    t0 = time.time()
    trajs = run(tasks, n, max_turns, gen_fn, bench, base)
    wall = time.time() - t0
    out = V3 / out_dir / f"{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in trajs:
            row = dict(t)
            if not keep_messages:
                row.pop("messages")
            f.write(json.dumps(row) + "\n")
    meta = {"tag": tag, "model": model, "lora": lora or None, "policy_gpu": policy_gpu, "bench_gpu": "L4", "n": n,
            "max_turns": max_turns, "temperature": temperature, "top_p": 0.95, "max_new_tokens": MAX_NEW,
            "tasks": len(tasks), "wall_s": wall, "prompt": "v2 (modal_app/prompts_v2.py)",
            "bench": "val_bench_v1 (bench v2 conventions)",
            "bench_job_wall_s": sum((x["bench_wall_s"] or 0) for t in trajs for x in t["turns"]),
            "completion_tokens": sum(x["completion_tokens"] for t in trajs for x in t["turns"])}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", out, json.dumps(meta), flush=True)


# ---------------------------------------------------------------------------------------------
# summary (local, no GPU)
# ---------------------------------------------------------------------------------------------
def _geo(xs):
    import math
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def summarize(tags, out_name="phase0_baseline"):
    """tags: runs; "a+b" = run a with every problem that run b covers replaced by b's episodes
    (used when a problem's starter was fixed after run a)."""
    trajs, metas = [], []
    for tg in tags:
        parts = tg.split("+")
        merged = {}
        for part in parts:
            p = V3 / (f"{part}.jsonl" if "/" in part else f"phase0_runs/{part}.jsonl")
            rows = [json.loads(line) for line in open(p)]
            for pid in {r["task_id"] for r in rows}:
                merged[pid] = [r for r in rows if r["task_id"] == pid]
            metas.append(json.loads(p.with_suffix(".meta.json").read_text()))
        for pid, rows in merged.items():
            for r in rows:
                r["run"] = tg
                for x in r["turns"]:
                    x["category"] = category(x)   # recompute with the current classifier
                trajs.append(r)
    turns = [x for t in trajs for x in t["turns"]]
    cats = {}
    for x in turns:
        cats[x["category"]] = cats.get(x["category"], 0) + 1
    fails = [x for x in turns if x["category"] != "pass"]
    # solve = PASS with code that is not the (AST-identical) starter
    solved = lambda x: x["category"] == "pass" and not x["is_starter"]  # noqa: E731
    beat = lambda x: solved(x) and x["vs_starter"] and x["vs_starter"] > 1.05 and (x["vs_starter_ci"] or [0])[0] > 1.0  # noqa
    by_prob = {}
    for t in trajs:
        by_prob.setdefault(t["task_id"], []).append(t)
    n_per = {pid: len(ts) for pid, ts in by_prob.items()}

    def pass_at(pred, first_turn_only=False):
        p1, pn = [], []
        for pid, ts in by_prob.items():
            c = sum(any(pred(x) for x in (t["turns"][:1] if first_turn_only else t["turns"])) for t in ts)
            p1.append(c / len(ts))
            pn.append(1.0 if c > 0 else 0.0)
        return sum(p1) / len(p1), sum(pn) / len(pn)

    rep_pairs = rep_same = 0
    for t in trajs:
        ts = t["turns"]
        for a, b in zip(ts, ts[1:]):
            if a["category"] != "pass" and b["category"] != "pass":
                rep_pairs += 1
                rep_same += norm_msg(a) == norm_msg(b)
    best_vs_starter, best_vs_compile, best_vs_eager = [], [], []
    for t in trajs:
        ok = [x for x in t["turns"] if solved(x) and x["vs_starter"]]
        if ok:
            bx = max(ok, key=lambda x: x["vs_starter"])
            best_vs_starter.append(bx["vs_starter"])
            best_vs_eager.append(bx["vs_eager"])
            if bx["vs_compile"]:
                best_vs_compile.append(bx["vs_compile"])
    fam = {}
    for pid, ts in by_prob.items():
        f = ts[0]["family"]
        e = fam.setdefault(f, {"problems": 0, "episodes": 0, "solved_episodes": 0, "problems_solved": 0,
                               "turns": 0, "compile_turns": 0, "pass_turns_nonstarter": 0})
        e["problems"] += 1
        e["episodes"] += len(ts)
        c = sum(any(solved(x) for x in t["turns"]) for t in ts)
        e["solved_episodes"] += c
        e["problems_solved"] += c > 0
        for t in ts:
            e["turns"] += len(t["turns"])
            e["compile_turns"] += sum(x["category"] == "compile" for x in t["turns"])
            e["pass_turns_nonstarter"] += sum(solved(x) for x in t["turns"])
    for e in fam.values():
        e["pass@1"] = round(e["solved_episodes"] / e["episodes"], 4)
        e["compile_error_rate"] = round(e["compile_turns"] / max(1, e["turns"]), 4)
    p1, pn = pass_at(solved)
    p1_t0, pn_t0 = pass_at(solved, first_turn_only=True)
    b1, bn = pass_at(beat)
    anyp1, anypn = pass_at(lambda x: x["category"] == "pass")
    # per-run pass@1 (seed-to-seed spread for the SFT gate)
    per_run = {}
    for tg in tags:
        rs = [t for t in trajs if t["run"] == tg]
        bp = {}
        for t in rs:
            bp.setdefault(t["task_id"], []).append(any(solved(x) for x in t["turns"]))
        per_run[tg] = round(sum(sum(v) / len(v) for v in bp.values()) / len(bp), 4)
    summ = {
        "model": metas[0]["model"], "lora": None, "runs": tags, "n_per_problem": sorted(set(n_per.values())),
        "max_turns": metas[0]["max_turns"], "temperature": metas[0]["temperature"], "problems": len(by_prob),
        "run_files": [m["tag"] for m in metas],
        "episodes": len(trajs), "turns": len(turns),
        "definitions": {
            "solve": "a turn whose verdict is PASS (val_bench_v1: full case list, fp32/fp64 golden, launch check, "
                     "determinism) and whose code is not AST-identical to the starter",
            "pass@1": "mean over problems of (episodes with >= 1 solve) / n", "pass@n": "fraction of problems "
                      "with >= 1 solving episode", "compile_error_rate": "turns failing with a Triton compile "
                      "error / all benchmarked turns (share of failing turns also given)",
            "repeat_error_rate": "consecutive fail->fail turn pairs with the same category + normalised message "
                                 "(numerics compared by category only) / all fail->fail pairs",
            "speedup": "best solving turn per episode (by vs_starter); geomean over solving episodes; "
                       "vs torch.compile = vs_starter x per-problem compile/starter baseline",
            "beat_starter": "solve with paired vs_starter median > 1.05 and 95% CI lower bound > 1.0"},
        "pass@1": round(p1, 4), "pass@n": round(pn, 4), "pass@1_turn0": round(p1_t0, 4), "pass@n_turn0": round(pn_t0, 4),
        "pass@1_incl_starter_resubmits": round(anyp1, 4), "pass@n_incl_starter_resubmits": round(anypn, 4),
        "beat_starter@1": round(b1, 4), "beat_starter@n": round(bn, 4), "pass@1_per_run": per_run,
        "turn_categories": cats, "turn_category_share": {k: round(v / len(turns), 4) for k, v in cats.items()},
        "compile_error_rate": round(cats.get("compile", 0) / len(turns), 4),
        "compile_share_of_failures": round(cats.get("compile", 0) / max(1, len(fails)), 4),
        "starter_resubmit_turns": sum(x["is_starter"] for x in turns),
        "repeat_error_rate": round(rep_same / max(1, rep_pairs), 4), "fail_fail_pairs": rep_pairs,
        "p_pass_next_given_fail": None,
        "solving_episodes": len(best_vs_starter),
        "speedup_vs_starter_geomean": _geo(best_vs_starter), "speedup_vs_starter_median":
            statistics.median(best_vs_starter) if best_vs_starter else None,
        "speedup_vs_compile_geomean": _geo(best_vs_compile), "speedup_vs_compile_median":
            statistics.median(best_vs_compile) if best_vs_compile else None,
        "speedup_vs_eager_geomean": _geo(best_vs_eager),
        "by_family": fam,
        "wall_s": sum(m["wall_s"] for m in metas), "bench_job_wall_s": sum(m["bench_job_wall_s"] for m in metas),
        "completion_tokens": sum(m["completion_tokens"] for m in metas),
    }
    import difflib
    starters = {t["task_id"]: t["starter_code"] for t in load_tasks(summ["max_turns"])}
    sims = [difflib.SequenceMatcher(None, starters[t["task_id"]], x["code"]).ratio()
            for t in trajs for x in t["turns"] if solved(x)]
    summ["solving_turns"] = len(sims)
    summ["solving_turns_near_starter_share"] = round(sum(r >= 0.9 for r in sims) / max(1, len(sims)), 4)
    summ["definitions"]["near_starter"] = "difflib ratio >= 0.9 between the solving code and the starter"
    nf = sum(1 for t in trajs for a, b in zip(t["turns"], t["turns"][1:]) if a["category"] != "pass")
    npf = sum(1 for t in trajs for a, b in zip(t["turns"], t["turns"][1:]) if a["category"] != "pass" and solved(b))
    summ["p_pass_next_given_fail"] = round(npf / max(1, nf), 4)
    with (V3 / f"{out_name}.jsonl").open("w") as f:
        for t in trajs:
            row = {k: v for k, v in t.items() if k != "messages"}
            f.write(json.dumps(row) + "\n")
    (V3 / f"{out_name}.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("by_family", "definitions")}, indent=1))
    return summ


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "summarize":
    summarize(sys.argv[2].split(","), *(sys.argv[3:4]))
