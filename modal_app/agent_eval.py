"""
Multi-turn agent eval (no training): propose kernel -> bench -> read result -> revise.

Same episode structure as src/autokernel-rlvr/agent_loop/autokernel_loop.py,
but the "tool call" is a fenced ```python block (every turn is benchmarked),
so it works with any chat model without verl's tool parser.

Requires the bench/policy apps to be deployed:
    modal deploy modal_app/bench_modal.py
    modal deploy modal_app/kernelbench_modal.py
    modal deploy modal_app/policy.py
Then:
    modal run modal_app/agent_eval.py --suite autokernel --n 8 --max-turns 8
    modal run modal_app/agent_eval.py --suite kb --n 1 --max-turns 4

Concurrency: 1 policy GPU + <=8 bench GPUs = 9 (account cap is 10).
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import sys
import time

import modal

ROOT = pathlib.Path(os.environ.get("AK_ROOT") or pathlib.Path(__file__).resolve().parents[1])
SRC = ROOT / "src" / "autokernel-rlvr"
RESULTS = ROOT / "results"
sys.path.insert(0, str(pathlib.Path(__file__).parent))

app = modal.App("autokernel-agent-eval")

MAX_CTX = 32768
MAX_NEW = 4096


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


reward_mod = _load(SRC / "agent_loop" / "reward.py", "ak_reward")

PROTOCOL = """

Tool protocol for this session: instead of a function call, end every reply with the
FULL kernel.py in one ```python fenced block. It is benchmarked automatically and the
autokernel_bench result comes back as the next message."""

KB_SYSTEM = """You are an expert GPU kernel engineer. You write Triton kernels for NVIDIA GPUs.

You are given a KernelBench problem: a PyTorch `Model` class plus get_inputs()/get_init_inputs().
Write a `ModelNew` class with the same constructor signature and outputs (atol=rtol=1e-2)
that runs faster, using custom Triton kernels (@triton.jit) for the heavy ops.

Each reply must end with the FULL kernel.py in one ```python fenced block containing
imports, your Triton kernels, and class ModelNew. It is benchmarked automatically and you
get back correctness (PASS/FAIL/CRASH/TIMEOUT) and speedup vs the PyTorch Model. Revise
using that feedback. A fast but wrong kernel scores zero."""


def autokernel_tasks(max_turns: int) -> list:
    bd = _load(SRC / "data" / "build_dataset.py", "ak_build")
    src = RESULTS / "autokernel_src"
    ref = (src / "reference.py").read_text()
    tasks = []
    # bench.py ignores the dataset's shape/dtype (it sweeps its own sizes), so one
    # prompt per kernel_type is the honest task count.
    for kt in bd.KERNELS:
        shape, dtype = bd.SHAPE_SWEEP[kt][-1], "float16"
        starter = (src / "kernels" / f"{kt}.py").read_text()
        tasks.append({"task_id": kt, "suite": "autokernel", "kernel_type": kt, "messages": [
            {"role": "system", "content": bd.SYSTEM_PROMPT + PROTOCOL},
            {"role": "user", "content": bd.user_prompt(kt, shape, dtype, ref, starter, max_turns)},
        ]})
    return tasks


def kb_tasks(limit: int, stride: int = 1) -> list:
    rows = [json.loads(l) for l in (RESULTS / "kernelbench_l1_problems.jsonl").read_text().splitlines() if l]
    # only harness-valid problems: the identity ModelNew PASSes (kernelbench_modal.py::sanity)
    ok = set(json.loads((RESULTS / "kb_valid_problems.json").read_text())["valid"])
    rows = [p for p in rows if p["problem_id"] in ok]
    tasks = []
    for p in rows[::stride][:limit]:
        tasks.append({"task_id": f"L1_{p['problem_id']}", "suite": "kb",
                      "problem_id": p["problem_id"], "messages": [
            {"role": "system", "content": KB_SYSTEM},
            {"role": "user", "content": f"KernelBench Level 1, problem {p['problem_id']}.\n\n"
                                        f"```python\n{p['code']}\n```\n\nWrite ModelNew."},
        ]})
    return tasks


_CODE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str):
    blocks = _CODE.findall(text)
    return blocks[-1] if blocks else None


FEEDBACK = os.environ.get("AK_FEEDBACK", "v2")   # v1 = log tail only (original worker)


def observation(res: dict) -> str:
    keep = {k: res.get(k) for k in ("correctness", "speedup_vs_pytorch", "pct_peak",
                                    "latency_us", "pytorch_latency_us", "bottleneck", "stages")
            if res.get(k) not in (None, "", {})}
    msg = "autokernel_bench result:\n" + json.dumps(keep)
    if res.get("correctness") == "PASS":
        return msg
    if FEEDBACK == "v2" and res.get("fail_lines"):
        msg += "\n\nfailures:\n" + "\n".join(res["fail_lines"])
        if res.get("correctness") == "CRASH" and res.get("raw"):
            msg += "\n\nlog tail:\n" + res["raw"][-800:]
    elif res.get("raw"):
        msg += "\n\nlog tail:\n" + res["raw"][-1500:]
    return msg


def run_episodes(tasks: list, n: int, max_turns: int, gen_fn, bench, suite: str,
                 max_new: int = MAX_NEW, log=print) -> list:
    """Roll out n episodes per task. gen_fn(list_of_conversations) -> list of gen dicts.

    Every turn's code is benchmarked with the full bench (not --quick): quick mode skips
    the numerical-stability stage, and the rmsnorm/flash_attention starters PASS quick
    but FAIL full (results/stage1_baselines.jsonl).
    """
    trajs = [{"task": {k: v for k, v in t.items() if k != "messages"}, "sample": s,
              "messages": [dict(m) for m in t["messages"]], "turns": [], "done": False}
             for t in tasks for s in range(n)]
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
        key = "problem_id" if suite == "kb" else "kernel_type"
        timeout = 300 if suite == "kb" else 180
        args = [(active[i]["task"][key], codes[i], False, timeout, True) for i in idx]
        t0 = time.time()
        # list() drains the generator fully; zip() alone leaves Modal's async iterator open
        results = dict(zip(idx, list(bench.run.starmap(args, return_exceptions=True,
                                                                wrap_returned_exceptions=False))))
        t_bench = time.time() - t0
        for i, t in enumerate(active):
            if i in results:
                r = results[i]
                if isinstance(r, Exception):
                    r = {"correctness": "INFRA_ERROR", "raw": repr(r)}
                if "speedup" in r and "speedup_vs_pytorch" not in r:
                    r["speedup_vs_pytorch"] = r["speedup"] if r.get("correctness") == "PASS" else 0.0
            else:
                r = {"correctness": "CRASH", "raw": "no ```python code block in reply"}
            g = t.pop("_gen")
            t["turns"].append({
                "turn": turn, "correctness": r.get("correctness"),
                "speedup": float(r.get("speedup_vs_pytorch") or 0.0),
                "pct_peak": r.get("pct_peak"), "cached": r.get("cached"),
                "bench_wall_s": r.get("wall_s"), "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"], "finish": g["finish"],
                "code_sha": hashlib.sha256((codes[i] or "").encode()).hexdigest()[:12],
                "code": codes[i], "error_tail": (r.get("raw") or "")[-600:],
                "fail_lines": r.get("fail_lines") or [], "stages": r.get("stages"),
            })
            t["messages"].append({"role": "user", "content": observation(r)})
            if g["prompt_tokens"] + 2 * max_new > MAX_CTX:
                t["done"] = True
        log(f"turn {turn}: {len(active)} active, gen {t_gen:.0f}s, bench {t_bench:.0f}s, "
            f"PASS {sum(1 for t in active if t['turns'][-1]['correctness'] == 'PASS')}")

    for t in trajs:
        ts = t["turns"]
        passes = [x for x in ts if x["correctness"] == "PASS"]
        info = {"any_pass": bool(passes),
                "best_speedup": max((x["speedup"] for x in passes), default=0.0),
                "pass_count": len(passes),
                "crash_count": sum(x["correctness"] == "CRASH" for x in ts)}
        t["reward"] = reward_mod.autokernel_reward("autokernel", "", "", info)
        t.update(info)
        t.pop("done")
    return trajs


def write_trajs(trajs: list, out: pathlib.Path, keep_messages: bool = False):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in trajs:
            row = {k: v for k, v in t.items() if k != "messages"}
            row["final_messages_chars"] = sum(len(m["content"]) for m in t["messages"])
            if keep_messages:
                row["messages"] = t["messages"]
            f.write(json.dumps(row) + "\n")


@app.local_entrypoint()
def main(suite: str = "autokernel", model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
         lora: str = "", n: int = 8, max_turns: int = 8, limit: int = 100,
         temperature: float = 0.8, bench_gpu: str = "L4", stride: int = 1, tag: str = ""):
    Policy = modal.Cls.from_name("autokernel-policy", "Policy")
    if suite == "kb":
        Bench = modal.Cls.from_name("autokernel-kernelbench", "KBBencher")
        tasks = kb_tasks(limit, stride)
    else:
        Bench = modal.Cls.from_name("autokernel-bench", "Bencher")
        tasks = autokernel_tasks(max_turns)
    bench = Bench.with_options(gpu=bench_gpu)()
    pol = Policy(model=model, enable_lora=bool(lora))

    def gen_fn(convs):
        outs = pol.generate.remote(convs, n=1, temperature=temperature, max_tokens=MAX_NEW,
                                   lora_path=lora, lora_id=1 if lora else 0)
        return [o[0] for o in outs]

    t_start = time.time()
    trajs = run_episodes(tasks, n, max_turns, gen_fn, bench, suite)
    name = tag or f"{suite}_{model.split('/')[-1]}_n{n}_t{max_turns}_{bench_gpu}" + ("_lora" if lora else "")
    out = RESULTS / "agent_eval" / f"{name}.jsonl"
    write_trajs(trajs, out)
    meta = {"feedback": FEEDBACK, "suite": suite, "model": model, "lora": lora, "n": n, "max_turns": max_turns,
            "temperature": temperature, "bench_gpu": bench_gpu, "stride": stride, "tasks": len(tasks),
            "wall_s": time.time() - t_start}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out, json.dumps(meta))
