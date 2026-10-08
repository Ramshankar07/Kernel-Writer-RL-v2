"""
SFT Phase 1: in-repo positives + repair pairs in the shared candidate schema (results/v3/sft/README.md).
Local CPU only, no GPU, re-runnable.

    python modal_app/sft_ingest_inrepo.py
      -> results/v3/sft/candidates_inrepo.jsonl         (format autokernel, bench-v2 PASS, non-starter, AST-deduped)
      -> results/v3/sft/candidates_inrepo_repair.jsonl  (failing turn + its bench feedback -> later v2-PASS turn)
      -> results/v3/sft/ingest_inrepo_summary.json

Inputs
  * results/v3/rescore_bench_v2.jsonl: one row per distinct old-PASS kernel, keyed by `sha` =
    sha256(turn["code"]) (bench_v2_modal.load_pass_kernels); `verdict` is the bench v2 verdict.
  * trajectories: results/grpo/*/rollouts/*.jsonl + results/agent_eval/autokernel_*.jsonl (suite autokernel),
    the same files the rescore read. Code bodies are recovered from the turns by that sha256.
  * starters: results/autokernel_src/kernels/<kt>.py (the starter the policy saw and the rescore timed against).

Positives: v2 PASS -> drop rows whose normalised AST (sft_static.ast_key: comments, docstrings, formatting
stripped) equals the starter's -> one row per distinct AST (representative = best vs-starter geomean, then sha).

Repair pairs: in each trajectory, every turn k whose old-bench verdict is not PASS and that has code is paired
with the first later turn j>k whose code sha256 is in the bench-v2 PASS set. Dropped: target AST == starter
(a "resubmit the starter" fix), failed AST == target AST, failed code itself v2-PASS. Deduplicated on
(failed AST, target AST). `repair.feedback` re-renders the turn's real bench output (fail_lines, error_tail,
stages, as stored in the rollout) in the prompt-v2 observation layout (prompts_v2.OBS_LABEL_V2); the stored
fields are kept verbatim in meta.feedback_fields. These verdicts come from the original bench (v1).
"""
import collections
import difflib
import glob
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sft_static import ast_key, static_check  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
V3 = ROOT / "results" / "v3"
OUT = V3 / "sft"
OBS_LABEL_V2 = "bench result:"  # prompts_v2.OBS_LABEL_V2 (not imported: keep this script import-light)
FAMILY = {"reduce": ["reduce"], "softmax": ["softmax"], "cross_entropy": ["cross_entropy"],
          "rmsnorm": ["rmsnorm"], "layernorm": ["layernorm"], "matmul": ["matmul"],
          "flash_attention": ["attention"], "fused_mlp": ["mlp", "matmul", "activation"],
          "rotary_embedding": ["rotary"]}


def sha256(s):
    return hashlib.sha256(s.encode()).hexdigest()


def traj_files():
    files = sorted(glob.glob(str(ROOT / "results/grpo/*/rollouts/*.jsonl")))
    files += sorted(glob.glob(str(ROOT / "results/agent_eval/autokernel_*.jsonl")))
    return files


def iter_trajs():
    """Yields (source, step, line_no, record) for autokernel trajectories (same sources as the rescore)."""
    for fp in traj_files():
        p = pathlib.Path(fp)
        src = p.parts[-3] if "grpo" in p.parts else "agent_eval_" + p.stem.split("_")[-1]
        step = p.stem if "grpo" in p.parts else ""
        for i, line in enumerate(open(fp)):
            r = json.loads(line)
            if r.get("task", {}).get("suite") != "autokernel":
                continue
            yield src, step, i, r


def starter(kt):
    return (ROOT / "results" / "autokernel_src" / "kernels" / f"{kt}.py").read_text()


def feedback(t):
    head = {"correctness": t.get("correctness")}
    if isinstance(t.get("stages"), dict) and any(v is not None for v in t["stages"].values()):
        head["stages"] = t["stages"]
    msg = OBS_LABEL_V2 + "\n" + json.dumps(head)
    fl = t.get("fail_lines") or []
    tail = (t.get("error_tail") or "").strip()
    if fl:
        msg += "\n\nfailures:\n" + "\n".join(fl[:12])
    if tail and (t.get("correctness") == "CRASH" or not fl):
        msg += "\n\nlog tail:\n" + tail
    return msg


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    resc = [json.loads(l) for l in open(V3 / "rescore_bench_v2.jsonl")]
    summ_rows = {r["code_sha12"]: r for r in json.loads((V3 / "rescore_summary.json").read_text())["rows"]}
    v2 = {r["sha"]: r for r in resc}
    v2_pass = {s for s, r in v2.items() if r["verdict"] == "PASS"}

    # recover code bodies + occurrences for every rescored sha
    code, occ = {}, collections.defaultdict(list)
    for src, step, i, r in iter_trajs():
        for t in r.get("turns", []):
            if t.get("code"):
                h = sha256(t["code"])
                if h in v2:
                    code[h] = t["code"]
                    if t.get("correctness") == "PASS":
                        occ[h].append(f"{src}/{step}/s{r.get('sample')}/t{t.get('turn')}")
    missing = sorted(s for s in v2 if s not in code)

    st_key = {kt: ast_key(starter(kt)) for kt in FAMILY}
    steps = collections.OrderedDict()
    steps["rescored_old_pass"] = len(v2)
    steps["v2_verdicts"] = dict(collections.Counter(r["verdict"] for r in resc))
    steps["v2_pass"] = len(v2_pass)
    steps["v2_pass_code_recovered"] = sum(s in code for s in v2_pass)
    groups = collections.defaultdict(list)
    n_starter_rows = 0
    starter_keys_hit = set()
    for s in v2_pass:
        if s not in code:
            continue
        kt = v2[s]["kernel_type"]
        k = ast_key(code[s])
        if k == st_key[kt]:
            n_starter_rows += 1
            starter_keys_hit.add(kt)
            continue
        groups[(kt, k)].append(s)
    steps["v2_pass_rows_equal_starter_ast"] = n_starter_rows
    steps["starter_programs"] = len(starter_keys_hit)
    steps["v2_pass_distinct_ast_incl_starter"] = len(groups) + len(starter_keys_hit)
    steps["distinct_non_starter_ast"] = len(groups)

    def score(s):
        g = summ_rows.get(s[:12], {}).get("vs_starter_geomean")
        return (-(g or 0.0), s)

    pos = []
    for (kt, k), shas in sorted(groups.items(), key=lambda kv: (kv[0][0], min(kv[1]))):
        shas = sorted(shas, key=score)
        rep = shas[0]
        c = code[rep]
        sr = summ_rows.get(rep[:12], {})
        st = static_check(c, "autokernel")
        st["target_ast_sha"] = k
        pos.append({
            "id": f"inrepo:{kt}:{rep[:12]}",
            "source": "inrepo", "format": "autokernel", "reference": kt, "kernel_type": kt,
            "target": c, "family": FAMILY[kt], "repair": None,
            "meta": {"code_sha256": rep, "member_sha256": shas, "n_members": len(shas),
                     "sources": sorted({x for s in shas for x in v2[s]["sources"]}),
                     "n_pass_occurrences": sum(len(occ[s]) for s in shas),
                     "first_seen": v2[rep].get("first_seen"),
                     "bench_v2_verdict": "PASS", "bench_v2_harness": v2[rep].get("harness"),
                     "gpu": v2[rep].get("gpu"),
                     "vs_starter_geomean": sr.get("vs_starter_geomean"),
                     "vs_eager_geomean": sr.get("vs_eager_geomean"),
                     "vs_compile_geomean": sr.get("vs_compile_geomean"),
                     "real_speedup_vs_starter": sr.get("real_speedup_vs_starter"),
                     "starter_similarity": round(difflib.SequenceMatcher(None, starter(kt), c, autojunk=False).ratio(), 4),
                     "starter_file": f"results/autokernel_src/kernels/{kt}.py",
                     "policy": "Qwen2.5-Coder-7B-Instruct (GRPO / agent_eval rollouts)"},
            "static": st,
        })

    # ---------------- repair pairs ----------------
    rs = collections.Counter()
    pairs, seen = [], set()
    for src, step, li, r in iter_trajs():
        kt = r["task"]["kernel_type"]
        turns = r.get("turns", [])
        tsha = [sha256(t["code"]) if t.get("code") else None for t in turns]
        for k, t in enumerate(turns):
            if t.get("correctness") == "PASS":
                continue
            rs["failing_turns"] += 1
            j = next((j for j in range(k + 1, len(turns)) if tsha[j] in v2_pass), None)
            if j is None:
                continue
            rs["failing_turns_with_later_v2_pass"] += 1
            if not t.get("code"):
                rs["drop_failed_no_code"] += 1
                continue
            if tsha[k] in v2_pass:
                rs["drop_failed_code_is_v2_pass"] += 1
                continue
            tgt = code.get(tsha[j]) or turns[j]["code"]
            tk, fk = ast_key(tgt), ast_key(t["code"])
            if tk == st_key[kt]:
                rs["drop_target_is_starter"] += 1
                continue
            if fk is not None and fk == tk:
                rs["drop_failed_same_ast_as_target"] += 1
                continue
            key = (fk or sha256(t["code"]), tk)
            if key in seen:
                rs["drop_duplicate_pair"] += 1
                continue
            seen.add(key)
            st = static_check(tgt, "autokernel")
            st["target_ast_sha"] = tk
            st["failed_parses"] = fk is not None
            fb = feedback(t)
            pairs.append({
                "id": f"inrepo:repair:{src}:{step or 'eval'}:{li}:s{r.get('sample')}:t{k}-t{j}",
                "source": "inrepo", "format": "autokernel", "reference": kt, "kernel_type": kt,
                "target": tgt, "family": FAMILY[kt],
                "repair": {"failed_code": t["code"], "feedback": fb},
                "meta": {"traj_source": src, "traj_step": step, "traj_line": li, "sample": r.get("sample"),
                         "failed_turn": k, "target_turn": j, "gap": j - k,
                         "failed_code_sha256": tsha[k], "target_code_sha256": tsha[j],
                         "failed_old_verdict": t.get("correctness"),
                         "failed_is_starter": fk == st_key[kt],
                         "feedback_fields": {"correctness": t.get("correctness"), "stages": t.get("stages"),
                                             "fail_lines": t.get("fail_lines") or [],
                                             "error_tail": t.get("error_tail") or ""},
                         "feedback_mode_at_rollout": "v1_log_tail_only" if src == "agent_eval_fbv1" else "v2",
                         "feedback_note": "re-rendered from the stored rollout fields; bench v1 verdict for the "
                                          "failing turn, bench v2 PASS for the target",
                         "target_bench_v2_verdict": "PASS"},
                "static": st,
            })
    rs["pairs"] = len(pairs)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("candidates_inrepo.jsonl", pos), ("candidates_inrepo_repair.jsonl", pairs)):
        with (OUT / name).open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    toks = lambda s: len(s) / 3.5  # noqa: E731
    summary = {
        "inputs": {"rescore": "results/v3/rescore_bench_v2.jsonl", "trajectory_files": len(traj_files()),
                   "starters": "results/autokernel_src/kernels/<kt>.py"},
        "positives": {
            "steps": steps, "missing_code_for_rescored_sha": missing,
            "n_candidates": len(pos),
            "by_kernel_type": dict(collections.Counter(p["kernel_type"] for p in pos)),
            "static_ok": sum(p["static"]["ok"] for p in pos),
            "static_fail_reasons": dict(collections.Counter(f for p in pos for f in p["static"]["forbidden"])),
            "starter_similarity_lt_0_9": sum(p["meta"]["starter_similarity"] < 0.9 for p in pos),
            "real_speedup_vs_starter": sum(bool(p["meta"]["real_speedup_vs_starter"]) for p in pos),
            "target_tokens_chars_div_3_5": {"p50": pct([toks(p["target"]) for p in pos], 0.5),
                                            "p90": pct([toks(p["target"]) for p in pos], 0.9)},
        },
        "repair": {
            "steps": dict(rs),
            "by_kernel_type": dict(collections.Counter(p["kernel_type"] for p in pairs)),
            "by_traj_source": dict(collections.Counter(p["meta"]["traj_source"] for p in pairs)),
            "by_failed_old_verdict": dict(collections.Counter(p["meta"]["failed_old_verdict"] for p in pairs)),
            "gap_histogram": dict(sorted(collections.Counter(p["meta"]["gap"] for p in pairs).items())),
            "failed_is_starter": sum(p["meta"]["failed_is_starter"] for p in pairs),
            "distinct_targets_ast": len({p["static"]["target_ast_sha"] for p in pairs}),
            "targets_static_ok": sum(p["static"]["ok"] for p in pairs),
            "feedback_has_fail_lines": sum(bool(p["meta"]["feedback_fields"]["fail_lines"]) for p in pairs),
            "tokens_chars_div_3_5": {
                "failed_code_plus_feedback_plus_target_p50": pct(
                    [toks(p["repair"]["failed_code"] + p["repair"]["feedback"] + p["target"]) for p in pairs], 0.5),
                "p90": pct([toks(p["repair"]["failed_code"] + p["repair"]["feedback"] + p["target"])
                            for p in pairs], 0.9)},
        },
    }
    (OUT / "ingest_inrepo_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
