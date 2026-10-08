"""Summarise kb v2 sanity + ablations -> results/v3/kb_v2_valid_problems.json (+ stats on stdout)."""
import json
import pathlib
import statistics
from collections import Counter

R = pathlib.Path(__file__).resolve().parents[1] / "results"
V3 = R / "v3"

old = json.loads((R / "kb_valid_problems.json").read_text())
old_valid = set(old["valid"])
old_cat = {**{p: "conv_unseeded" for p in old["conv_unseeded"]},
           **{p: "flaky_timeout" for p in old["flaky_1_50"]}}
for p in range(1, 101):
    if p not in old_valid and p not in old_cat:
        old_cat[p] = "timeout_both_runs"          # #38: never passed in 1-50 runs

rows = {r["problem_id"]: r for r in map(json.loads, (V3 / "kb_v2_sanity.jsonl").read_text().splitlines())}
st = lambda r, run, c: r.get(run, {}).get("cands", {}).get(c, {}).get("status")
valid, failing, speed, strict_fail, weights = [], {}, [], [], Counter()
starter = Counter()
for p in range(1, 101):
    r = rows.get(p, {})
    s1, s2 = st(r, "run1", "identity"), st(r, "run2", "identity")
    starter[("bridge", st(r, "run1", "starter_bridge"))] += 1
    starter[("fixed", st(r, "run1", "starter_fixed"))] += 1
    if s1 == s2 == "PASS":
        valid.append(p)
        i1 = r["run1"]["cands"]["identity"]
        weights[i1.get("weights")] += 1
        if not i1.get("strict_1e-4"):
            strict_fail.append((p, i1.get("max_abs")))
        for run in ("run1", "run2"):
            if (sp := r[run]["cands"]["identity"].get("speedup")):
                speed.append(sp)
    else:
        why = [r.get(run, {}).get("cands", {}).get("identity", {}).get("reason", r.get("error", "missing"))
               for run in ("run1", "run2")]
        failing[p] = {"run1": s1, "run2": s2, "reason": why}

recovered = Counter(old_cat[p] for p in valid if p not in old_valid)
regressed = [p for p in old_valid if p not in valid]

abl = {}
for f in V3.glob("kb_v2_ablate_*.jsonl"):
    rs = [json.loads(l) for l in f.read_text().splitlines()]
    abl[f.stem] = {"n": len(rs), "status": dict(Counter(r.get("cands", {}).get("identity", {}).get("status") for r in rs)),
                   "fail_ids": sorted(r["problem_id"] for r in rs
                                      if r.get("cands", {}).get("identity", {}).get("status") != "PASS")}

out = {
    "valid": valid,
    "n_valid": len(valid),
    "failing": failing,
    "before_n_valid": len(old_valid),
    "recovered_by_old_category": dict(recovered),
    "regressed_vs_old": regressed,
    "starters_run1": {f"{k[0]}:{k[1]}": v for k, v in sorted(starter.items(), key=str)},
    "weights_mode_valid": dict(weights),
    "identity_speedup": {"n": len(speed), "median": statistics.median(speed) if speed else None,
                         "min": min(speed, default=None), "max": max(speed, default=None),
                         "outside_0.95_1.05": sum(1 for s in speed if not 0.95 <= s <= 1.05)},
    "identity_strict_1e-4_fail": strict_fail,
    "ablations": abl,
    "note": "valid = identity ModelNew(Model) PASSes kb v2 in both independent runs (5 seeded + 1 hidden fresh-seed trial each).",
}
(V3 / "kb_v2_valid_problems.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "valid"}, indent=1))
