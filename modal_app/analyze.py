"""
Offline analysis -> results/numbers.json (single source of truth for deck + Q&A).

No GPU. Reads whatever stages exist in results/ and skips the rest.
    python modal_app/analyze.py
"""
import collections
import importlib.util
import itertools
import json
import math
import pathlib
import random
import statistics as st

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"

H100_PER_HR, L4_PER_HR, A100_PER_HR = 3.95, 0.80, 2.50   # Modal list prices, Sept 2026


def jl(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]


def _reward_mod():
    spec = importlib.util.spec_from_file_location(
        "rw", ROOT / "src" / "autokernel-rlvr" / "agent_loop" / "reward.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RW = _reward_mod()


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def stage1(N):
    p = R / "stage1_baselines.jsonl"
    if not p.exists():
        return
    rows = jl(p)
    out = {}
    for gpu in ("H100", "L4"):
        d = {}
        for kt in sorted({r["kernel_type"] for r in rows}):
            q = [r for r in rows if r["gpu_req"] == gpu and r["kernel_type"] == kt and r["quick"]]
            f = [r for r in rows if r["gpu_req"] == gpu and r["kernel_type"] == kt and not r["quick"]][0]
            s = [r["speedup_vs_pytorch"] for r in q]
            d[kt] = {"quick": q[0]["correctness"], "full": f["correctness"],
                     "speedup_quick": round(st.mean(s), 3),
                     "cv_pct": round(100 * st.pstdev(s) / st.mean(s), 2) if st.mean(s) else 0.0,
                     "wall_s_full": f["wall_s"], "stages_full": f.get("stages")}
        out[gpu] = d
    N["stage1"] = {
        "per_gpu": out,
        "n_runs": len(rows),
        "starters_pass_full": {g: sum(v["full"] == "PASS" for v in out[g].values()) for g in out},
        "starters_pass_quick": {g: sum(v["quick"] == "PASS" for v in out[g].values()) for g in out},
        "quick_pass_full_fail": {g: [k for k, v in out[g].items() if v["quick"] == "PASS" and v["full"] != "PASS"] for g in out},
        "max_cv_pct": max(v["cv_pct"] for g in out for v in out[g].values()),
        "bench_wall_s": {g: {"p50": pct([r["wall_s"] for r in rows if r["gpu_req"] == g], .5),
                             "p95": pct([r["wall_s"] for r in rows if r["gpu_req"] == g], .95)} for g in out},
    }


def kb_sanity(N):
    p = R / "kb_sanity.jsonl"
    if not p.exists():
        return
    rows = jl(p)                       # one identity-solution run per Level 1 problem
    sp = [r["speedup"] for r in rows if r["correctness"] == "PASS"]
    d = {"n": len(rows), "unique_problems": len({r["problem_id"] for r in rows}),
         "counts": dict(collections.Counter(r["correctness"] for r in rows)),
         "identity_speedup_median": round(st.median(sp), 3) if sp else None,
         "identity_speedup_p5_p95": [round(pct(sp, .05), 3), round(pct(sp, .95), 3)] if sp else None,
         "identity_within_1pct": sum(0.99 <= s <= 1.01 for s in sp),
         "wall_s_p50": pct([r["wall_s"] for r in rows], .5),
         "wall_s_p95": pct([r["wall_s"] for r in rows], .95)}
    vp = R / "kb_valid_problems.json"
    if vp.exists():
        v = json.loads(vp.read_text())
        d.update({"valid": len(v["valid"]), "flaky_timeouts": len(v["flaky_1_50"]),
                  "conv_unseeded": len(v["conv_unseeded"])})
    r1, r2 = R / "kb_sanity_1_50_run1.jsonl", R / "kb_sanity_1_50_run2.jsonl"
    if r1.exists() and r2.exists():   # repeatability: same identity solution benched twice
        a, b = {r["problem_id"]: r for r in jl(r1)}, {r["problem_id"]: r for r in jl(r2)}
        both = [k for k in a if a[k]["correctness"] == b[k]["correctness"] == "PASS"]
        d["repeat"] = {"problems": len(a),
                       "same_verdict": sum(a[k]["correctness"] == b[k]["correctness"] for k in a),
                       "speedup_abs_diff_median": round(st.median(abs(a[k]["speedup"] - b[k]["speedup"]) for k in both), 4) if both else None}
    old = R / "kb_sanity_defaultcpu.jsonl"
    if old.exists():   # also problems 1-50 x2 (listing bug); report per-run share
        oc = collections.Counter(r["correctness"] for r in jl(old))
        d["default_cpu_pass_frac"] = round(oc["PASS"] / sum(oc.values()), 3)
    N["kb_sanity"] = d


def fast_p(trajs, ps=(0.0, 1.0, 1.5, 2.0)):
    """KernelBench fast_p over tasks, using each trajectory's best PASS."""
    by = collections.defaultdict(list)
    for t in trajs:
        by[t["task"]["task_id"]].append(t)
    out = {}
    for p in ps:
        hits = [st.mean(float(t["any_pass"] and t["best_speedup"] > p if p else t["any_pass"]) for t in g)
                for g in by.values()]
        out[f"fast_{p}"] = round(st.mean(hits), 4)
    return out


def agent_eval(N):
    d = R / "agent_eval"
    if not d.exists():
        return
    N["agent_eval"] = {}
    for p in sorted(d.glob("*.jsonl")):
        trajs = jl(p)
        meta = json.loads(p.with_suffix(".meta.json").read_text()) if p.with_suffix(".meta.json").exists() else {}
        T = max(len(t["turns"]) for t in trajs)
        turns = [x for t in trajs for x in t["turns"]]
        e = {"meta": meta, "n_traj": len(trajs),
             "pass_rate": round(st.mean(t["any_pass"] for t in trajs), 4),
             "reward_mean": round(st.mean(t["reward"] for t in trajs), 4),
             "best_speedup_max": round(max(t["best_speedup"] for t in trajs), 3),
             "fast_p": fast_p(trajs),
             "turn_outcomes": dict(collections.Counter(x["correctness"] for x in turns)),
             "no_code_block": sum("no ```python" in (x["error_tail"] or "") for x in turns),
             "pass_at_turn": [round(st.mean(any(x["correctness"] == "PASS" for x in t["turns"][:k]) for t in trajs), 4)
                              for k in range(1, T + 1)],
             "prompt_tokens_by_turn": [round(st.mean(t["turns"][k]["prompt_tokens"] for t in trajs if len(t["turns"]) > k))
                                       for k in range(T)],
             "completion_tokens_mean": round(st.mean(x["completion_tokens"] for x in turns)),
             "truncated_frac": round(st.mean(x["finish"] == "length" for x in turns), 4),
             "bench_wall_s_p50": pct([x["bench_wall_s"] or 0 for x in turns if not x["cached"]], .5),
             "cache_hit_rate": round(st.mean(bool(x["cached"]) for x in turns), 4),
             "dup_code_frac": round(1 - len({(t["task"]["task_id"], x["code_sha"]) for t in trajs for x in t["turns"]}) / len(turns), 4),
             }
        if trajs[0]["task"]["suite"] == "autokernel":
            per = collections.defaultdict(list)
            for t in trajs:
                per[t["task"]["kernel_type"]].append(t)
            e["per_kernel"] = {k: {"pass_rate": round(st.mean(t["any_pass"] for t in v), 3),
                                   "best_speedup": round(max(t["best_speedup"] for t in v), 3),
                                   "reward_mean": round(st.mean(t["reward"] for t in v), 3)}
                               for k, v in sorted(per.items())}
            e["ablations"] = ablations(trajs)
        e["failure_taxonomy"] = failure_taxonomy(turns)
        N["agent_eval"][p.stem] = e


def failure_taxonomy(turns):
    cats = collections.Counter()
    for x in turns:
        if x["correctness"] == "PASS":
            continue
        e = "\n".join(x.get("fail_lines") or []) + "\n" + (x["error_tail"] or "")
        stages = x.get("stages") or {}
        if "no ```python" in e:
            c = "no code block"
        elif x["correctness"] == "TIMEOUT":
            c = "timeout"
        elif "instantiation failed" in e:
            c = "wrong ModelNew signature"
        elif "OutOfResources" in e or "shared memory" in e:
            c = "out of shared memory"
        elif "CompilationError" in e or "triton" in e.lower() and "Error" in e:
            c = "Triton API / compile error"
        elif "exceeds tol" in e or "max_abs_error" in e:
            c = "numerical mismatch"
        elif "numerical_stability" in e or "nan" in e.lower() or str(stages.get("numerical_stability", "")).startswith("FAIL"):
            c = "numerical stability"
        elif str(stages.get("smoke_test", "")).startswith("FAIL"):
            c = "smoke test mismatch"
        elif "SyntaxError" in e or "IndentationError" in e:
            c = "syntax error"
        elif "AttributeError" in e or "NameError" in e or "ImportError" in e:
            c = "API/name error"
        elif "illegal memory" in e.lower() or "CUDA error" in e:
            c = "illegal memory access"
        else:
            c = "other FAIL/CRASH"
        cats[c] += 1
    return dict(cats.most_common())


def ablations(trajs):
    """Offline what-ifs on logged trajectories (no extra GPU)."""
    A = {}
    by = collections.defaultdict(list)
    for t in trajs:
        by[t["task"]["task_id"]].append(t)

    def rew(t, turns=None, clip_high=3.0, shaping=False):
        ts = t["turns"][:turns] if turns else t["turns"]
        ps = [x for x in ts if x["correctness"] == "PASS"]
        if not ps:
            r = 0.0
        else:
            r = math.log2(max(max(x["speedup"] for x in ps), 1e-6))
            r = max(-1.0, min(clip_high, r))
        if shaping:
            r += 0.02 * len(ps) - 0.02 * sum(x["correctness"] == "CRASH" for x in ts)
        return r

    # 1. clip: how many trajectories hit the 3.0 ceiling / the -1 floor
    raw = [math.log2(t["best_speedup"]) for t in trajs if t["any_pass"] and t["best_speedup"] > 0]
    A["clip"] = {"n_pass_traj": len(raw), "hit_ceiling_3": sum(r > 3 for r in raw),
                 "hit_floor_-1": sum(r < -1 for r in raw),
                 "slower_than_pytorch_pass": sum(r < 0 for r in raw)}
    # 2. zero-variance groups vs group size (subsample from n)
    n = len(next(iter(by.values())))
    rng = random.Random(0)
    zg = {}
    for g in sorted({2, 4, n}):
        if g > n:
            continue
        fr = []
        for _ in range(200):
            fr.append(st.mean(float(len({round(t["reward"], 6) for t in rng.sample(v, g)}) == 1) for v in by.values()))
        zg[g] = round(st.mean(fr), 4)
    A["zero_adv_group_frac_by_group_size"] = zg
    # 3. turn budget: reward if max_turns were k
    T = max(len(t["turns"]) for t in trajs)
    A["reward_by_turn_budget"] = {k: round(st.mean(rew(t, k) for t in trajs), 4) for k in range(1, T + 1)}
    # 4. step shaping: does it reorder trajectories within groups?
    flips, pairs = 0, 0
    for v in by.values():
        for a, b in itertools.combinations(v, 2):
            d0 = rew(a) - rew(b)
            d1 = rew(a, shaping=True) - rew(b, shaping=True)
            if d0 != 0 or d1 != 0:
                pairs += 1
                flips += (d0 > 0) != (d1 > 0) and d0 != 0 or (d0 == 0 and d1 != 0)
    A["step_shaping"] = {"pairs_compared": pairs, "pairs_reordered_or_split": flips,
                         "mean_reward_with_shaping": round(st.mean(rew(t, shaping=True) for t in trajs), 4)}
    # 5. early stop (README sharp edge #7): stop after turn>=5 once speedup>2
    saved, lost, total = 0, 0.0, 0
    for t in trajs:
        total += len(t["turns"])
        for i, x in enumerate(t["turns"]):
            if i + 1 >= 5 and x["correctness"] == "PASS" and x["speedup"] > 2.0:
                saved += len(t["turns"]) - (i + 1)
                lost += rew(t) - rew(t, i + 1)
                break
    A["early_stop"] = {"turns_saved_frac": round(saved / total, 4), "reward_lost_total": round(lost, 4)}
    # 6. reward floor asymmetry: pass-but-slower gets -1..0, never-pass gets 0
    A["reward_floor"] = {"traj_negative_reward": sum(t["reward"] < 0 for t in trajs),
                         "traj_zero_no_pass": sum(not t["any_pass"] for t in trajs)}
    return A


def grpo(N):
    import sys
    sys.path.insert(0, str(ROOT / "modal_app"))
    from reward_v2 import reward_v2
    for run in sorted((R / "grpo").glob("*/metrics.jsonl")) if (R / "grpo").exists() else []:
        ms = jl(run)
        cfg = json.loads((run.parent / "config.json").read_text())
        roll = run.parent / "rollouts"
        for m in ms:   # v1 runs didn't log reward_v2; recompute from the logged turns
            if "reward_v2_mean" not in m and (roll / f"step_{m['step']:03d}.jsonl").exists():
                tr = jl(roll / f"step_{m['step']:03d}.jsonl")
                m["reward_v2_mean"] = st.mean(reward_v2(t["turns"]) for t in tr)
            m.setdefault("reward_v2_mean", None)
            rp = roll / f"step_{m['step']:03d}.jsonl"
            if rp.exists():   # correctness progress, from bench.py's stage results
                xs = [x for t in jl(rp) for x in t["turns"]]
                m["smoke_pass_rate"] = st.mean(str((x.get("stages") or {}).get("smoke_test") or "").startswith("PASS") for x in xs)
        k = min(3, len(ms))
        v2s = [m["reward_v2_mean"] for m in ms if m["reward_v2_mean"] is not None]
        N.setdefault("grpo", {})[run.parent.name] = {
            "config": cfg, "steps": len(ms),
            "reward_first3": round(st.mean(m["reward_mean"] for m in ms[:k]), 4),
            "reward_last3": round(st.mean(m["reward_mean"] for m in ms[-k:]), 4),
            "reward_v2_first3": round(st.mean(v2s[:k]), 4) if v2s else None,
            "reward_v2_last3": round(st.mean(v2s[-k:]), 4) if v2s else None,
            "pass_first3": round(st.mean(m["pass_rate"] for m in ms[:k]), 4),
            "pass_last3": round(st.mean(m["pass_rate"] for m in ms[-k:]), 4),
            "best_speedup_max": max(m["best_speedup_max"] for m in ms),
            "rollout_s_mean": round(st.mean(m["rollout_s"] for m in ms)),
            "train_s_mean": round(st.mean(m["train_s"] for m in ms)),
            "zero_adv_mean": round(st.mean(m["zero_adv_group_frac"] for m in ms), 4),
            "cache_hit_last3": round(st.mean(m["cache_hit_rate"] for m in ms[-k:]), 4),
            "smoke_first3": round(st.mean(m.get("smoke_pass_rate", 0) for m in ms[:k]), 4),
            "smoke_last3": round(st.mean(m.get("smoke_pass_rate", 0) for m in ms[-k:]), 4),
            "crash_first3": round(st.mean(m["crash_rate"] for m in ms[:k]), 4),
            "crash_last3": round(st.mean(m["crash_rate"] for m in ms[-k:]), 4),
            "kernels_ever_passed": sorted({kk for m in ms for kk, v in m["per_kernel_best"].items() if v > 0}),
            "wall_h": round(sum(m["rollout_s"] + m["train_s"] for m in ms) / 3600, 2),
            "est_cost_usd": round(sum(m["rollout_s"] + m["train_s"] for m in ms) / 3600 * 2 * H100_PER_HR, 2),
            "curve": [{k2: m.get(k2) for k2 in ("step", "reward_mean", "reward_v2_mean", "pass_rate",
                                                 "best_speedup_max", "zero_adv_group_frac", "cache_hit_rate",
                                                 "grad_norm", "mean_token_logp", "crash_rate", "turn_pass_rate",
                                                 "smoke_pass_rate")}
                      for m in ms],
            "per_kernel_reward_first_last": {kk: [round(ms[0]["per_kernel_reward"][kk], 3), round(ms[-1]["per_kernel_reward"][kk], 3)]
                                             for kk in ms[0]["per_kernel_reward"]},
        }


def main():
    N = {"label": "Modal re-run, Sept 2026",
         "prices_usd_per_gpu_hr": {"H100": H100_PER_HR, "L4": L4_PER_HR, "A100": A100_PER_HR}}
    for f in (stage1, kb_sanity, agent_eval, grpo):
        f(N)
    extra = R / "extra_numbers.json"     # GPU ablations that write their own summaries
    if extra.exists():
        N["extra"] = json.loads(extra.read_text())
    (R / "numbers.json").write_text(json.dumps(N, indent=1))
    print("wrote results/numbers.json with sections:", [k for k in N])


if __name__ == "__main__":
    main()
