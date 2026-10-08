"""
SFT final assembly (Phase 1 steps 6-8): verified + decontaminated candidates -> Qwen2.5 chat
`messages`, repair data, rebalanced mix, token lengths, train/dev split, report. Local CPU only.

    python modal_app/sft_assemble.py                      # -> results/v3/sft/sft_{train,dev}.jsonl,
                                                          #    mix_report.{md,json}
    python modal_app/test_sft_assemble.py                 # render 5 examples, assert format
    python modal_app/sft_assemble.py --assume-pass --out-dir /tmp/x   # projection only (see below)

Fully re-runnable: it rebuilds every output from whatever results/v3/sft/verified_<subset>.jsonl
and decontam_<subset>.jsonl exist at the time (partial files are fine; a truncated last line is
skipped). Subsets: inrepo, inrepo_repair, drkernel, kernelbook_sel, drkernel_repair_sel.

Pipeline
  1. keep verify.verdict == "PASS"; drop decontam flag == true (KB-L1 / val-set near-dup).
     Decontam comes from decontam_<subset>.jsonl (by id); ids it does not cover are checked
     inline with sft_decontam.Index (same code; a None-returning visitor is filtered so rows
     that crash sft_decontam.py are still checked; noted per row as decontam_src).
     op_overlap (KB-L1 op Jaccard) never drops a row; it is recorded on the example.
  2. target must parse and contain no ``` (else it cannot be one fenced block).
  3. dedupe: single-turn by normalized target AST; repair by (failed AST, target AST).
     Priority inrepo > drkernel > kernelbook (the first one seen is kept).
  4. messages: autokernel -> SYSTEM_PROMPT_V2 + user_prompt_v2(...max_turns=4); modelnew ->
     prompts_sft.SYSTEM_PROMPT_MODELNEW + module + "Write ModelNew."; repair rows add
     assistant(failed) + user("bench result:\n" + feedback) + assistant(fix). The assistant turn
     is one deterministic sentence derived from the family and the number of @triton.jit kernels
     (no generated reasoning) + the full code in one ```python block. loss_mask: 1 only on the
     final assistant message (train_on = its index).
  5. tokens: real Qwen2.5-Coder-7B-Instruct tokenizer (tokenizer files only), apply_chat_template;
     drop > MAX_TOKENS.
  6. mix (seed SEED): all 9-family single-turn rows other than reduce are kept; scarce families
     (rotary, mlp, attention) are upsampled by min(3, ceil(SCARCE_FLOOR / n)); the generic
     families (reduce, activation, other, conv) share a budget so that they are <= GENERIC_MAX_FRAC
     of the single-turn examples (water-filled across those 4, autokernel-format rows first);
     single-turn total <= TARGET * (1 - REPAIR_FRAC). Repair rows (water-filled across families,
     in-repo first) target REPAIR_FRAC of the total.
  7. dev = 2% stratified (format, repair, primary family) holdout of the unique selected
     examples, taken before upsampling (no copy of a dev example is in train). Not the val set.

--assume-pass: treat every static.ok candidate that has no verify row yet as PASS (verified
non-PASS rows still drop). For capacity planning only; requires --out-dir outside results/.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import pathlib
import random
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from prompts_sft import OBS_LABEL, SYSTEM_PROMPT_MODELNEW, user_prompt_modelnew  # noqa: E402
from prompts_v2 import SYSTEM_PROMPT_V2, user_prompt_v2  # noqa: E402

SFT = ROOT / "results" / "v3" / "sft"
AK_SRC = ROOT / "results" / "autokernel_src"
STARTERS = HERE / "starters_v2"
BUILD_DATASET = ROOT / "src" / "autokernel-rlvr" / "data" / "build_dataset.py"
TOKENIZER = "Qwen/Qwen2.5-Coder-7B-Instruct"

SUBSETS = ["inrepo", "inrepo_repair", "drkernel", "kernelbook_sel", "drkernel_repair_sel"]
SOURCE_PRIORITY = {"inrepo": 0, "drkernel": 1, "kernelbook": 2}
SEED = 0
MAX_TOKENS = 4096
MAX_TURNS = 4
TARGET = 5000
REPAIR_FRAC = 0.15
GENERIC = ["reduce", "activation", "other", "conv"]
FEEDBACK_MAX_CHARS = 800   # lead 2026-10-06: same budget agent_eval gives crash tails; 31-41% of repair rows exceeded 4k tokens
GENERIC_MAX_FRAC = 0.80   # lead 2026-10-06: verified data has only ~500 non-generic single-turn rows, so 0.5-0.6 shrank the mix to ~1.5k; generic ops still teach Triton API use (31% of failures are compile errors)
SCARCE = ["rotary", "mlp", "attention"]
SCARCE_FLOOR = 150
MAX_UPSAMPLE = 3
DEV_FRAC = 0.02
AK9 = ["softmax", "layernorm", "rmsnorm", "reduce", "cross_entropy", "rotary", "matmul", "attention", "mlp"]
# primary family: scarce first, then the rest of the 9, then generic tags
FAMILY_ORDER = ["rotary", "attention", "mlp", "matmul", "cross_entropy", "rmsnorm", "layernorm",
                "softmax", "reduce", "activation", "conv", "other"]
FAMILY_LABEL = {"softmax": "softmax", "layernorm": "layer norm", "rmsnorm": "RMS norm",
                "reduce": "reduction", "cross_entropy": "cross-entropy loss",
                "rotary": "rotary embedding", "matmul": "matmul", "attention": "attention",
                "mlp": "fused MLP", "activation": "elementwise op", "conv": "convolution",
                "other": "op"}
LICENSE = {
    "inrepo": "Ours: kernels written by our Qwen2.5-Coder-7B-Instruct policy during GRPO v1/v2 and "
              "agent_eval rollouts in this repo, re-verified with bench v2 (Apache-2.0 base model; "
              "no third-party data).",
    "kernelbook": "GPUMODE/KernelBook @ 1576375bc92745b490e1cdf2fce01eba76d9f847 (MIT at that "
                  "revision; dataset_permissive.parquet). Targets are TorchInductor output; "
                  "per-row upstream repo licenses are kept in meta.",
    "drkernel": "hkust-nlp/drkernel-coldstart-8k @ cba0ef06a5b1e3c307b7acfa8b6acb7a46578105: MIT tag "
                "on HF, but the paper says the trajectories are distilled from GPT-5, so OpenAI's "
                "output terms (no use to develop competing models) may apply. Research use only "
                "until cleared.",
}


# ----------------------------------------------------------------------------- io helpers
def read_jsonl(path: pathlib.Path):
    """Rows of a (possibly still being written) jsonl; a truncated/bad line is counted, not fatal."""
    rows, bad = [], 0
    if not path.exists():
        return rows, bad, False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad, True


def ast_key(code: str):
    """sha256 of the AST with docstrings removed (comments/formatting already gone)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree, annotate_fields=False).encode()).hexdigest()


def stable_int(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


# ----------------------------------------------------------------------------- decontam
class InlineDecontam:
    """sft_decontam.Index, with the visitor-returns-None crash filtered out (see module doc)."""

    def __init__(self):
        import sft_decontam as d
        orig_cls, orig_fn = d._Norm.visit_ClassDef, d._Norm.visit_FunctionDef

        def _clean(fn):
            def wrapped(self, node):
                out = fn(self, node)
                out.body = [s for s in out.body if s is not None] or [d.ast.Pass()]
                return out
            return wrapped

        orig_strip = d._Norm._strip_doc

        def strip_doc(self, body):  # drop statements that will visit to None *before* visiting
            body = orig_strip(self, body)
            keep = [s for s in body if not (isinstance(s, d.ast.AnnAssign) and s.value is None)
                    and not (isinstance(s, d.ast.Expr) and isinstance(s.value, d.ast.Constant)
                             and isinstance(s.value.value, str))]
            return keep or [d.ast.Pass()]

        d._Norm._strip_doc = strip_doc
        d._Norm.visit_ClassDef = _clean(orig_cls)
        d._Norm.visit_FunctionDef = _clean(orig_fn)
        d._Norm.visit_AsyncFunctionDef = d._Norm.visit_FunctionDef
        self.idx = d.Index()

    def check(self, row):
        return self.idx.check(row)


# ----------------------------------------------------------------------------- prompt building
def load_shape_sweep():
    """SHAPE_SWEEP / DTYPES from build_dataset.py without importing it (it imports pandas)."""
    tree = ast.parse(BUILD_DATASET.read_text())
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) in ("SHAPE_SWEEP", "DTYPES")
                                                 for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(BUILD_DATASET), "exec"), {}, ns)
    return ns["SHAPE_SWEEP"], ns["DTYPES"]


SHAPE_SWEEP, DTYPES = load_shape_sweep()
_REF_SRC = None
_STARTER = {}


def reference_src():
    global _REF_SRC
    if _REF_SRC is None:
        _REF_SRC = (AK_SRC / "reference.py").read_text()
    return _REF_SRC


def starter_src(kt):
    if kt not in _STARTER:
        _STARTER[kt] = (STARTERS / f"{kt}.py").read_text()
    return _STARTER[kt]


def primary_family(fams):
    fams = fams or []
    for f in FAMILY_ORDER:
        if f in fams:
            return f
    return "other"


def n_jit(code: str) -> int:
    return len(re.findall(r"^\s*@triton\.jit", code, re.M))


def lead_in(fam: str, code: str, fmt: str, fix: bool) -> str:
    n = n_jit(code)
    kern = f" ({n} @triton.jit kernel{'s' if n != 1 else ''})" if n else ""
    what = "kernel.py" if fmt == "autokernel" else "file with ModelNew"
    if fix:
        return f"Fixed Triton {FAMILY_LABEL.get(fam, 'op')}{kern} for the bench failure above; full {what} below."
    return f"Triton {FAMILY_LABEL.get(fam, 'op')}{kern}; full {what} below."


def assistant_msg(fam, code, fmt, fix=False):
    return f"{lead_in(fam, code, fmt, fix)}\n\n```python\n{code.rstrip()}\n```"


def user_task(row):
    if row["format"] == "autokernel":
        kt = row["kernel_type"]
        shapes = SHAPE_SWEEP[kt]
        h = stable_int(row["id"])
        shape, dtype = shapes[h % len(shapes)], DTYPES[(h // len(shapes)) % len(DTYPES)]
        return SYSTEM_PROMPT_V2, user_prompt_v2(kt, shape, dtype, reference_src(), starter_src(kt), MAX_TURNS)
    return SYSTEM_PROMPT_MODELNEW, user_prompt_modelnew(row["reference"])


def build_messages(row):
    fam = primary_family(row.get("family"))
    system, user = user_task(row)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    rep = row.get("repair")
    if rep:
        fb = rep["feedback"].strip()
        if len(fb) > FEEDBACK_MAX_CHARS:   # keep the head (verdict) and the tail (traceback end)
            fb = fb[:300] + "\n...\n" + fb[-(FEEDBACK_MAX_CHARS - 305):]
        if not fb.startswith(OBS_LABEL):
            fb = f"{OBS_LABEL}\n{fb}"
        msgs += [{"role": "assistant", "content": assistant_msg(fam, rep["failed_code"], row["format"])},
                 {"role": "user", "content": fb},
                 {"role": "assistant", "content": assistant_msg(fam, row["target"], row["format"], fix=True)}]
    else:
        msgs.append({"role": "assistant", "content": assistant_msg(fam, row["target"], row["format"])})
    return msgs


# ----------------------------------------------------------------------------- mixing
def water_fill(groups: dict, budget: int):
    """groups: name -> ordered list. Take a common per-group cap so the total <= budget."""
    total = sum(len(v) for v in groups.values())
    if budget >= total:
        return {k: list(v) for k, v in groups.items()}
    lo, hi = 0, max((len(v) for v in groups.values()), default=0)
    while lo < hi:  # largest cap with sum(min(n, cap)) <= budget
        mid = (lo + hi + 1) // 2
        if sum(min(len(v), mid) for v in groups.values()) <= budget:
            lo = mid
        else:
            hi = mid - 1
    out = {k: list(v[:lo]) for k, v in groups.items()}
    left = budget - sum(len(v) for v in out.values())
    for k in sorted(groups, key=lambda k: -len(groups[k])):  # hand out the remainder
        if left <= 0:
            break
        if len(groups[k]) > lo:
            out[k].append(groups[k][lo])
            left -= 1
    return out


def prio_order(rows, rng):
    """autokernel format first (in-repo, the policy's own format), then a seeded shuffle."""
    rows = list(rows)
    rng.shuffle(rows)
    return sorted(rows, key=lambda r: 0 if r["format"] == "autokernel" else 1)


def stratified_dev(examples, frac, rng):
    """Systematic sample every 1/frac within a stratum-sorted, shuffled list; only rows whose
    target key is unique in the selection (so no train row shares a dev target)."""
    tkey_count = collections.Counter(e["_tkey"] for e in examples)
    strata = collections.defaultdict(list)
    for e in examples:
        strata[(e["format"], e["is_repair"], e["primary_family"])].append(e)
    seq = []
    for k in sorted(strata, key=str):
        v = strata[k]
        rng.shuffle(v)
        seq += v
    step = round(1 / frac)
    off = rng.randrange(step)
    dev_ids = set()
    for i, e in enumerate(seq):
        if i % step == off:
            if tkey_count[e["_tkey"]] == 1:
                dev_ids.add(e["id"])
            else:  # take the next unique one in the same stratum instead
                for e2 in seq[i + 1:i + step]:
                    if tkey_count[e2["_tkey"]] == 1 and (e2["format"], e2["is_repair"], e2["primary_family"]) == \
                            (e["format"], e["is_repair"], e["primary_family"]):
                        dev_ids.add(e2["id"])
                        break
    return dev_ids


# ----------------------------------------------------------------------------- tokens
def load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install 'transformers<5' tokenizers  (tokenizer files only; no weights are downloaded)")
    return AutoTokenizer.from_pretrained(TOKENIZER)


def token_lengths(tok, examples, batch=256):
    full = [tok.apply_chat_template(e["messages"], tokenize=False) for e in examples]
    pref = [tok.apply_chat_template(e["messages"][:-1], tokenize=False, add_generation_prompt=True)
            for e in examples]
    for i in range(0, len(examples), batch):
        a = tok(full[i:i + batch], add_special_tokens=False)["input_ids"]
        b = tok(pref[i:i + batch], add_special_tokens=False)["input_ids"]
        for e, x, y in zip(examples[i:i + batch], a, b):
            e["n_tokens"], e["n_train_tokens"] = len(x), len(x) - len(y)


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)]


def tok_stats(xs):
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": max(xs),
            "mean": round(statistics.mean(xs), 1), "total": sum(xs)}


# ----------------------------------------------------------------------------- main
def collect(assume_pass: bool):
    """-> candidate examples (passing filters, before token drop), per-subset accounting."""
    acct = {}
    inline = None
    pool = []
    for sub in SUBSETS:
        cands, cbad, cexists = read_jsonl(SFT / f"candidates_{sub}.jsonl")
        ver, vbad, vexists = read_jsonl(SFT / f"verified_{sub}.jsonl")
        dec, dbad, dexists = read_jsonl(SFT / f"decontam_{sub}.jsonl")
        dec_by = {d["id"]: d for d in dec}
        ver_by = {}
        for r in ver:
            ver_by[r["id"]] = r  # last write wins
        static_ok = [c for c in cands if (c.get("static") or {}).get("ok")]
        a = {"candidates": len(cands), "candidates_static_ok": len(static_ok),
             "verified_file": vexists, "verified_rows": len(ver_by), "verified_bad_lines": vbad,
             "decontam_file": dexists, "decontam_rows": len(dec_by), "decontam_bad_lines": dbad,
             "verdicts": dict(collections.Counter((r.get("verify") or {}).get("verdict") for r in ver_by.values())),
             "pending_verification": sum(1 for c in static_ok if c["id"] not in ver_by),
             "drops": collections.Counter(), "decontam_inline": 0, "kept": 0}
        rows = list(ver_by.values())
        if assume_pass:
            for c in static_ok:
                if c["id"] not in ver_by:
                    rows.append(dict(c, verify={"verdict": "PASS", "reason": "assumed (projection)"}))
        for r in rows:
            v = (r.get("verify") or {}).get("verdict")
            if v != "PASS":
                a["drops"][f"verify_{v}"] += 1
                continue
            d = dec_by.get(r["id"])
            src = "file"
            if d is None:
                if inline is None:
                    inline = InlineDecontam()
                try:
                    d = inline.check(r)
                    src = "inline"
                except Exception as e:  # noqa: BLE001 -- unknown parse failure: be conservative
                    a["drops"]["decontam_error"] += 1
                    a.setdefault("decontam_errors", []).append(f"{r['id']}: {type(e).__name__}: {e}"[:200])
                    continue
                a["decontam_inline"] += 1
            if d.get("flag"):
                a["drops"]["decontam_" + ("kb_l1" if d.get("near_dup_kb") else "val" if d.get("near_dup_val") else "flag")] += 1
                continue
            tgt = r.get("target") or ""
            if "```" in tgt or (r.get("repair") and "```" in r["repair"].get("failed_code", "")):
                a["drops"]["contains_fence"] += 1
                continue
            tkey = ast_key(tgt)
            if tkey is None:
                a["drops"]["target_does_not_parse"] += 1
                continue
            if r["format"] == "autokernel" and not (STARTERS / f"{r.get('kernel_type')}.py").exists():
                a["drops"]["no_starter_for_kernel_type"] += 1
                continue
            rep = r.get("repair")
            fkey = None
            if rep:
                fkey = ast_key(rep.get("failed_code", "")) or hashlib.sha256(
                    rep.get("failed_code", "").encode()).hexdigest()
            op = d.get("op_overlap") or {}
            pool.append({"id": r["id"], "subset": sub, "source": r["source"], "format": r["format"],
                         "kernel_type": r.get("kernel_type"), "family": r.get("family") or [],
                         "primary_family": primary_family(r.get("family")), "is_repair": bool(rep),
                         "_row": r, "_tkey": tkey, "_dkey": (fkey, tkey) if rep else tkey,
                         "decontam": {"src": src, "op_overlap": op or None,
                                      "op_overlap_flag": bool(d.get("op_overlap_flag")),
                                      "families_in_val": d.get("families_in_val", [])},
                         "verify": {k: (r.get("verify") or {}).get(k) for k in ("verdict", "gpu", "harness", "reason")}})
        acct[sub] = a
    return pool, acct


def dedupe(pool, acct):
    pool.sort(key=lambda e: (SOURCE_PRIORITY.get(e["source"], 9), SUBSETS.index(e["subset"]), e["id"]))
    seen, out = {}, []
    for e in pool:
        k = (e["is_repair"], e["_dkey"])
        if k in seen:
            acct[e["subset"]]["drops"]["duplicate_of_" + seen[k]] += 1
            continue
        seen[k] = e["subset"]
        out.append(e)
    return out


def mix(pool, rng, target):
    single = [e for e in pool if not e["is_repair"]]
    repair = [e for e in pool if e["is_repair"]]
    by = collections.defaultdict(list)
    for e in single:
        by[e["primary_family"]].append(e)
    for k in by:
        by[k] = prio_order(by[k], rng)
    factors = {}
    for f in SCARCE:
        n = len(by.get(f, []))
        factors[f] = min(MAX_UPSAMPLE, max(1, math.ceil(SCARCE_FLOOR / n))) if n else 0
    core = {k: v for k, v in by.items() if k not in GENERIC}
    gen = {k: v for k, v in by.items() if k in GENERIC}
    single_target = round(target * (1 - REPAIR_FRAC))
    eff = lambda k, n: n * factors.get(k, 1)  # noqa: E731
    n_core_eff = sum(eff(k, len(v)) for k, v in core.items())
    notes = []
    if n_core_eff > single_target:  # only if data grows a lot: water-fill the core too
        core = water_fill(core, single_target)
        n_core_eff = sum(eff(k, len(v)) for k, v in core.items())
        notes.append("core (non-generic 9-family) rows exceeded the single-turn target and were water-filled")
    # generic share g/(core+g) <= GENERIC_MAX_FRAC  ->  g <= core * f/(1-f)
    g_cap = int(n_core_eff * GENERIC_MAX_FRAC / (1 - GENERIC_MAX_FRAC))
    g_budget = max(0, min(sum(len(v) for v in gen.values()), g_cap, single_target - n_core_eff))
    gen_sel = water_fill(gen, g_budget)
    single_sel = {**core, **gen_sel}
    n_single_eff = sum(eff(k, len(v)) for k, v in single_sel.items())
    r_target = round(n_single_eff * REPAIR_FRAC / (1 - REPAIR_FRAC))
    rby = collections.defaultdict(list)
    for e in repair:
        rby[e["primary_family"]].append(e)
    for k in rby:
        rby[k] = prio_order(rby[k], rng)
    rep_sel = water_fill(rby, min(r_target, len(repair)))
    selected = [e for v in single_sel.values() for e in v] + [e for v in rep_sel.values() for e in v]
    info = {"single_target": single_target, "core_effective": n_core_eff, "generic_cap_from_frac": g_cap,
            "generic_budget": g_budget, "generic_available": {k: len(v) for k, v in gen.items()},
            "repair_target": r_target, "repair_available": len(repair),
            "upsample_factors": {k: v for k, v in factors.items()}, "notes": notes}
    return selected, factors, info


def counts(examples, key):
    c = collections.Counter(key(e) for e in examples)
    return dict(sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def public(e, dup=0):
    r = e["_row"]
    n = len(e["messages"])
    meta = {k: r.get("meta", {}).get(k) for k in ("revision", "final_speedup", "fixed_speedup", "repair_kind",
                                                     "vs_starter_geomean", "repo_licenses", "license")
            if r.get("meta", {}).get(k) is not None}
    return {"id": e["id"] + (f"#dup{dup}" if dup else ""), "orig_id": e["id"], "subset": e["subset"],
            "source": e["source"], "format": e["format"], "kernel_type": e["kernel_type"],
            "family": e["family"], "primary_family": e["primary_family"], "is_repair": e["is_repair"],
            "messages": e["messages"], "loss_mask": [0] * (n - 1) + [1], "train_on": n - 1,
            "n_tokens": e["n_tokens"], "n_train_tokens": e["n_train_tokens"], "dup": dup,
            "decontam": e["decontam"], "verify": e["verify"], "meta": meta}


def trunc(s, head=700, tail=300):
    if len(s) <= head + tail + 40:
        return s
    return s[:head] + f"\n... [{len(s) - head - tail} chars omitted] ...\n" + s[-tail:]


def write_report(out_dir, rep, samples):
    j = out_dir / "mix_report.json"
    j.write_text(json.dumps(rep, indent=1, default=str))
    L = ["# SFT mix report", "", f"Generated by `modal_app/sft_assemble.py` (seed {SEED}). "
         f"{'**PROJECTION (--assume-pass): unverified candidates treated as PASS.**' if rep['assume_pass'] else ''}",
         "", "## Totals", "",
         f"| split | examples | unique | tokens (total) | trained tokens |", "|---|---|---|---|---|"]
    for s in ("train", "dev"):
        t = rep["splits"][s]
        L.append(f"| {s} | {t['examples']} | {t['unique']} | {t['tokens']['total'] if t['tokens'].get('n') else 0} | "
                 f"{t['train_tokens']['total'] if t['train_tokens'].get('n') else 0} |")
    L += ["", f"Target {rep['config']['target']} examples (4-6k); repair target {rep['config']['repair_frac']:.0%}; "
          f"generic families {GENERIC} capped at {GENERIC_MAX_FRAC:.0%} of single-turn.", ""]
    for n in rep["mix"]["notes"]:
        L.append(f"- {n}")
    L += ["", "## Per subset (input accounting)", "",
          "| subset | candidates | static ok | verified rows | PASS | pending verify | decontam file | inline decontam | kept after filters | in train+dev |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sub, a in rep["subsets"].items():
        L.append(f"| {sub} | {a['candidates']} | {a['candidates_static_ok']} | {a['verified_rows']} | "
                 f"{a['verdicts'].get('PASS', 0)} | {a['pending_verification']} | "
                 f"{'yes' if a['decontam_file'] else 'no'} ({a['decontam_rows']}) | {a['decontam_inline']} | "
                 f"{a['kept']} | {rep['selected_by_subset'].get(sub, 0)} |")
    L += ["", "## Drops by reason", "", "| subset | reason | n |", "|---|---|---|"]
    for sub, a in rep["subsets"].items():
        for k, v in sorted(a["drops"].items(), key=lambda kv: -kv[1]):
            L.append(f"| {sub} | {k} | {v} |")
    L += ["", "## Train composition", ""]
    for title, key in (("source", "by_source"), ("format", "by_format"), ("repair", "by_repair"),
                       ("primary family", "by_family")):
        L += [f"**By {title}** (train, incl. upsampled copies): " +
              ", ".join(f"{k}: {v}" for k, v in rep["splits"]["train"][key].items()), ""]
    t = rep["splits"]["train"]
    L += [f"9 AutoKernel families: {t['ak9_share']:.1%} of train; generic (reduce/activation/other/conv): "
          f"{t['generic_share']:.1%}; repair: {t['repair_share']:.1%}; autokernel format: {t['autokernel_share']:.1%}.", "",
          "**Upsampling factors** (scarce families, duplication, <=3x): " +
          ", ".join(f"{k}: {v}x (n={rep['mix']['available_single_by_family'].get(k, 0)})"
                    for k, v in rep["mix"]["upsample_factors"].items()), "",
          "**Available single-turn by primary family** (after filters): " +
          ", ".join(f"{k}: {v}" for k, v in rep["mix"]["available_single_by_family"].items()), "",
          "**Available repair by primary family**: " +
          ", ".join(f"{k}: {v}" for k, v in rep["mix"]["available_repair_by_family"].items()), "",
          f"op_overlap_flag (KB-L1 op Jaccard >= 0.5, kept): {t['op_overlap_flag']} train examples.", "",
          "## Token lengths (Qwen2.5-Coder-7B-Instruct tokenizer, chat template)", "",
          "| set | n | p50 | p90 | max | mean | total |", "|---|---|---|---|---|---|---|"]
    for name, s in rep["tokens"].items():
        if s.get("n"):
            L.append(f"| {name} | {s['n']} | {s['p50']} | {s['p90']} | {s['max']} | {s['mean']} | {s['total']} |")
    L += ["", f"Examples over {MAX_TOKENS} tokens are dropped (`too_long` above).", "", "## Licenses", ""]
    for k, v in LICENSE.items():
        L.append(f"- **{k}**: {v}")
    L += ["", "## Sample conversations (truncated)", ""]
    for s in samples:
        L += [f"### {s['id']} ({s['format']}, {s['primary_family']}, repair={s['is_repair']}, {s['n_tokens']} tokens)", ""]
        for i, m in enumerate(s["messages"]):
            L += [f"**{m['role']}** (loss_mask={s['loss_mask'][i]}):", "", "````", trunc(m["content"]), "````", ""]
    (out_dir / "mix_report.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(SFT))
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--assume-pass", action="store_true")
    a = ap.parse_args(argv)
    out_dir = pathlib.Path(a.out_dir).resolve()
    if a.assume_pass and out_dir == SFT.resolve():
        sys.exit("--assume-pass is a projection; pass --out-dir outside results/v3/sft")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    pool, acct = collect(a.assume_pass)
    pool = dedupe(pool, acct)
    for e in pool:
        e["messages"] = build_messages(e["_row"])
    tok = load_tokenizer()
    token_lengths(tok, pool)
    all_tokens = [e["n_tokens"] for e in pool]
    kept = []
    for e in pool:
        if e["n_tokens"] > MAX_TOKENS:
            acct[e["subset"]]["drops"]["too_long"] += 1
        else:
            kept.append(e)
    for e in kept:
        acct[e["subset"]]["kept"] += 1

    selected, factors, info = mix(kept, rng, a.target)
    dev_ids = stratified_dev(selected, DEV_FRAC, rng)
    dev = [e for e in selected if e["id"] in dev_ids]
    train = []
    for e in selected:
        if e["id"] in dev_ids:
            continue
        f = 1 if e["is_repair"] else factors.get(e["primary_family"], 1)
        train += [public(e, d) for d in range(f)]
    rng.shuffle(train)
    dev = [public(e) for e in dev]
    dev.sort(key=lambda e: e["id"])
    for name, rows in (("sft_train.jsonl", train), ("sft_dev.jsonl", dev)):
        with open(out_dir / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def split_stats(rows):
        n = len(rows) or 1
        return {"examples": len(rows), "unique": len({r["orig_id"] for r in rows}),
                "by_source": counts(rows, lambda r: r["source"]), "by_subset": counts(rows, lambda r: r["subset"]),
                "by_format": counts(rows, lambda r: r["format"]), "by_repair": counts(rows, lambda r: r["is_repair"]),
                "by_family": counts(rows, lambda r: r["primary_family"]),
                "by_family_x_format": counts(rows, lambda r: f"{r['primary_family']}/{r['format']}"),
                "ak9_share": sum(r["primary_family"] in AK9 for r in rows) / n,
                "generic_share": sum(r["primary_family"] in GENERIC for r in rows) / n,
                "repair_share": sum(r["is_repair"] for r in rows) / n,
                "autokernel_share": sum(r["format"] == "autokernel" for r in rows) / n,
                "op_overlap_flag": sum(r["decontam"]["op_overlap_flag"] for r in rows),
                "tokens": tok_stats([r["n_tokens"] for r in rows]),
                "train_tokens": tok_stats([r["n_train_tokens"] for r in rows])}

    for a_ in acct.values():
        a_["drops"] = dict(a_["drops"])
    rep = {"assume_pass": a.assume_pass, "seed": SEED, "tokenizer": TOKENIZER,
           "config": {"target": a.target, "repair_frac": REPAIR_FRAC, "generic": GENERIC,
                      "generic_max_frac": GENERIC_MAX_FRAC, "scarce": SCARCE, "scarce_floor": SCARCE_FLOOR,
                      "max_upsample": MAX_UPSAMPLE, "max_tokens": MAX_TOKENS, "dev_frac": DEV_FRAC,
                      "max_turns_in_prompt": MAX_TURNS},
           "subsets": acct,
           "selected_by_subset": counts(selected, lambda e: e["subset"]),
           "mix": {**info,
                   "available_single_by_family": counts([e for e in kept if not e["is_repair"]], lambda e: e["primary_family"]),
                   "available_repair_by_family": counts([e for e in kept if e["is_repair"]], lambda e: e["primary_family"])},
           "splits": {"train": split_stats(train), "dev": split_stats(dev)},
           "tokens": {"all_after_filters_before_length_drop": tok_stats(all_tokens),
                      "train": tok_stats([r["n_tokens"] for r in train]),
                      "train_trained_tokens": tok_stats([r["n_train_tokens"] for r in train]),
                      "dev": tok_stats([r["n_tokens"] for r in dev])},
           "license": LICENSE}
    samples, seen = [], set()
    for want in (lambda r: r["format"] == "autokernel" and not r["is_repair"],
                 lambda r: r["format"] == "modelnew" and not r["is_repair"],
                 lambda r: r["is_repair"]):
        for r in train:
            if want(r) and r["orig_id"] not in seen:
                samples.append(r)
                seen.add(r["orig_id"])
                break
    write_report(out_dir, rep, samples)
    t = rep["splits"]["train"]
    print(json.dumps({"out_dir": str(out_dir), "train": t["examples"], "train_unique": t["unique"],
                      "dev": len(dev), "by_format": t["by_format"], "by_repair": t["by_repair"],
                      "by_family": t["by_family"], "tokens": rep["tokens"]["train"],
                      "pending_verification": {k: v["pending_verification"] for k, v in acct.items()}}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
