"""SFT Phase 1: ingest hkust-nlp/drkernel-coldstart-8k (pinned) into candidate rows.

Local CPU only (no Modal). Re-runnable: downloads the pinned parquet via huggingface_hub
(cached in ~/.cache/huggingface) and rewrites all outputs.

    python3 modal_app/sft_ingest_drkernel.py

Outputs (schema: results/v3/sft/README.md):
  results/v3/sft/candidates_drkernel.jsonl         single-turn, best round, final_speedup >= 1.0
  results/v3/sft/candidates_drkernel_repair.jsonl  failed turn + server feedback -> better next turn
  results/v3/sft/ingest_drkernel_summary.json      counts per step, family histogram, token lengths

Dataset facts (inspected at the pinned revision):
  columns: messages, uuid, entry_point, repo_name, module_name, final_speedup, num_rounds,
           original_python_code, best_round, timestamp, conversion_mode, enable_thinking
  every row: 5 rounds = [user, assistant] x 5. user[0] = task prompt with original_python_code;
  user[k] (k=1..4) = KernelGym "Server feedback" for assistant round k. Round 5 has NO feedback
  in the trajectory; its result is known only via final_speedup when best_round == 5.
  best_round (1-based) is a dataset column; final_speedup == speedup of best_round (checked
  against the feedback for all best_round < 5 rows).
"""
from __future__ import annotations

import ast
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path

REPO_ID = "hkust-nlp/drkernel-coldstart-8k"
REVISION = "cba0ef06a5b1e3c307b7acfa8b6acb7a46578105"
PARQUET = "drkernel-coldstart-8k.parquet"
SOURCE = "drkernel"
MIN_SPEEDUP = 1.0
FEEDBACK_MAX_CHARS = 1500

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "v3" / "sft"
OUT_MAIN = OUT_DIR / "candidates_drkernel.jsonl"
OUT_REPAIR = OUT_DIR / "candidates_drkernel_repair.jsonl"
OUT_SUMMARY = OUT_DIR / "ingest_drkernel_summary.json"

LICENSE_NOTE = "MIT (HF tag); distilled from GPT-5 per Dr. Kernel paper section 4.1, OpenAI output terms may apply"


# ----------------------------------------------------------------------------- loading
def load_rows() -> list[dict]:
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    path = hf_hub_download(REPO_ID, PARQUET, repo_type="dataset", revision=REVISION)
    return pq.read_table(path).to_pylist()


# ----------------------------------------------------------------------------- code extraction
CODE_BLOCK_RE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract_code(answer: str) -> tuple[str | None, int]:
    """Final full code of an assistant answer: the last fenced block defining ModelNew,
    else the last fenced block. Returns (code, n_blocks)."""
    blocks = [b for b in CODE_BLOCK_RE.findall(answer) if b.strip()]
    if not blocks:
        return None, 0
    with_model = [b for b in blocks if re.search(r"class\s+ModelNew\b", b)]
    code = (with_model or blocks)[-1]
    return code.strip("\n") + "\n", len(blocks)


# ----------------------------------------------------------------------------- feedback parsing
FB_START = "Server feedback (status/metrics/errors):\n"
FB_END_RE = re.compile(r"\n\s*\n\s*\nReturn an improved Triton implementation", re.S)


def feedback_block(user_msg: str) -> str:
    """The server-feedback JSON-ish block, verbatim (without the instruction boilerplate)."""
    i = user_msg.find(FB_START)
    body = user_msg[i + len(FB_START):] if i >= 0 else user_msg
    m = FB_END_RE.search(body)
    if m:
        body = body[: m.start()]
    return body.strip("\n")


def _field(txt: str, key: str) -> str | None:
    # top-level fields are one per line: `  "key": value` (no trailing commas)
    m = re.search(r'^\s*"%s": (.*?)\s*$' % re.escape(key), txt, re.M)
    if m:
        return m.group(1)
    # failed tasks nest the result in "result_payload": {..., "key": value, ...}
    m = re.search(r'"result_payload": \{.*?"%s": ([^,}]+)' % re.escape(key), txt)
    return m.group(1).strip() if m else None


def parse_feedback(txt: str) -> dict:
    def b(v):
        return {"true": True, "false": False}.get(v) if v is not None else None

    sp = _field(txt, "speedup")
    try:
        spf = float(sp) if sp not in (None, "null") else None
    except ValueError:
        spf = None
    status = _field(txt, "status")
    return {
        "status": status.strip('"') if status else None,
        "compiled": b(_field(txt, "compiled")),
        "correctness": b(_field(txt, "correctness")),
        "decoy_kernel": b(_field(txt, "decoy_kernel")),
        "speedup": spf,
    }


def truncate_feedback(txt: str, limit: int = FEEDBACK_MAX_CHARS) -> tuple[str, bool]:
    """Keep the text verbatim; if too long, cut the middle of the longest lines (keeping each
    line's head and tail, where tracebacks end) until it fits ~limit chars."""
    if len(txt) <= limit:
        return txt, False
    lines = txt.split("\n")
    head_keep, tail_keep = 150, 450
    min_line = head_keep + tail_keep + 40
    while sum(len(l) for l in lines) + len(lines) - 1 > limit:
        excess = sum(len(l) for l in lines) + len(lines) - 1 - limit
        j = max(range(len(lines)), key=lambda k: len(lines[k]))
        line = lines[j]
        if len(line) <= min_line:
            break
        new_len = max(min_line, len(line) - excess)
        tail = tail_keep + max(0, new_len - min_line)
        cut = len(line) - head_keep - tail
        lines[j] = line[:head_keep] + f" ...[{cut} chars truncated]... " + line[len(line) - tail:]
    out = "\n".join(lines)
    if len(out) > limit + 300:  # many long lines: hard head/tail cut as last resort
        out = out[: limit // 2] + "\n...[truncated]...\n" + out[-limit // 2:]
    return out, True


# ----------------------------------------------------------------------------- static filter
FORBIDDEN_F = {"linear", "softmax", "layer_norm"}


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _is_triton_jit(dec: ast.AST, jit_aliases: set[str]) -> bool:
    d = dec.func if isinstance(dec, ast.Call) else dec
    name = _dotted(d)
    return name is not None and (name in ("triton.jit",) or name in jit_aliases)


def static_check(code: str) -> dict:
    res = {
        "ok": False, "parses": False, "jit_kernels": 0, "jit_names": [], "launched": False,
        "launched_kernels": [], "has_modelnew": False, "forbidden": [], "tl_symbols": [],
        "reasons": [],
    }
    if "extern_kernels" in code:
        res["forbidden"].append("extern_kernels")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as e:
        res["reasons"].append(f"parse_error: {type(e).__name__}: {e}"[:200])
        return res
    res["parses"] = True

    # import aliases
    tl_aliases, F_aliases, torch_aliases, jit_aliases = set(), set(), {"torch"}, set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "triton.language":
                    tl_aliases.add(a.asname or "triton.language")
                elif a.name == "torch.nn.functional" and a.asname:
                    F_aliases.add(a.asname)
                elif a.name == "torch":
                    torch_aliases.add(a.asname or "torch")
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if n.module == "triton" and a.name == "language":
                    tl_aliases.add(a.asname or "language")
                if n.module == "triton" and a.name == "jit":
                    jit_aliases.add(a.asname or "jit")
                if n.module == "torch.nn" and a.name == "functional":
                    F_aliases.add(a.asname or "functional")
    if not tl_aliases:
        tl_aliases.add("tl")

    # jit kernels
    jit = {}
    funcs = {}
    classes = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[n.name] = n
            if any(_is_triton_jit(d, jit_aliases) for d in n.decorator_list):
                jit[n.name] = n
        elif isinstance(n, ast.ClassDef):
            classes[n.name] = n
    # also nested jit defs (rare)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name not in jit:
            if any(_is_triton_jit(d, jit_aliases) for d in n.decorator_list):
                jit[n.name] = n
    res["jit_kernels"] = len(jit)
    res["jit_names"] = sorted(jit)

    # launches: name[grid](...)
    launched = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript):
            v = _dotted(n.func.value)
            if v and v.split(".")[-1] in jit:
                launched.add(v.split(".")[-1])
    res["launched_kernels"] = sorted(launched)
    res["launched"] = bool(launched)

    # tl symbols (outermost attribute chain rooted at a tl alias)
    syms = set()
    seen_inner = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and id(n) not in seen_inner:
            d = _dotted(n)
            if d is None:
                continue
            # mark inner attribute nodes so only the full chain is recorded
            inner = n.value
            while isinstance(inner, ast.Attribute):
                seen_inner.add(id(inner))
                inner = inner.value
            for al in tl_aliases:
                if d.startswith(al + "."):
                    syms.add("tl." + d[len(al) + 1:])
    res["tl_symbols"] = sorted(syms)

    # try / inheritance from Model
    tries = [n for n in ast.walk(tree) if isinstance(n, (ast.Try, getattr(ast, "TryStar", ast.Try)))]
    if tries:
        res["forbidden"].append("try")
    # informational: every try only guards imports
    # (`try: import triton; import triton.language as tl; _HAS_TRITON = True / except: _HAS_TRITON = False`)
    def _guard_stmt(s):
        return isinstance(s, (ast.Import, ast.ImportFrom)) or (
            isinstance(s, ast.Assign) and isinstance(s.value, ast.Constant))
    res["try_import_guard_only"] = bool(tries) and all(
        any(isinstance(s, (ast.Import, ast.ImportFrom)) for s in t.body) and all(_guard_stmt(s) for s in t.body)
        for t in tries)
    for c in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for b in c.bases:
            bd = _dotted(b)
            if bd and bd.split(".")[-1] == "Model":
                res["forbidden"].append(f"inherits_Model:{c.name}")
    mn = classes.get("ModelNew")
    res["has_modelnew"] = mn is not None
    if mn is None:
        res["reasons"].append("no_class_ModelNew")

    # forward path: ModelNew.forward + transitively reachable in-file helpers
    # (self.<method>, module-level functions, <AutogradFn>.apply -> its forward); jit kernels excluded
    if mn is not None:
        methods = {m.name: m for m in mn.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))}
        todo = [("ModelNew.forward", methods["forward"])] if "forward" in methods else []
        if not todo:
            res["reasons"].append("no_ModelNew.forward")
        visited = set()
        hits = []
        while todo:
            label, fn = todo.pop()
            if label in visited:
                continue
            visited.add(label)
            for n in ast.walk(fn):
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult):
                    hits.append(f"@ in {label}")
                if not isinstance(n, ast.Call):
                    continue
                d = _dotted(n.func)
                if not d:
                    continue
                parts = d.split(".")
                if parts[0] in torch_aliases and parts[1:] == ["matmul"]:
                    hits.append(f"torch.matmul in {label}")
                if d.startswith("torch.nn.functional.") or (parts[0] in F_aliases and len(parts) == 2):
                    hits.append(f"F.{parts[-1]} in {label}")
                elif parts[0] == "F" and len(parts) == 2 and parts[1] in FORBIDDEN_F:
                    hits.append(f"F.{parts[-1]} in {label}")
                # follow helpers
                if len(parts) == 2 and parts[0] == "self" and parts[1] in methods:
                    todo.append((f"ModelNew.{parts[1]}", methods[parts[1]]))
                elif len(parts) == 1 and parts[0] in funcs and parts[0] not in jit:
                    todo.append((parts[0], funcs[parts[0]]))
                elif len(parts) == 2 and parts[1] == "apply" and parts[0] in classes:
                    for m in classes[parts[0]].body:
                        if isinstance(m, ast.FunctionDef) and m.name == "forward":
                            todo.append((f"{parts[0]}.forward", m))
        res["forbidden"].extend(sorted(set(hits)))

    if res["jit_kernels"] < 1:
        res["reasons"].append("no_triton_jit")
    elif not res["launched"]:
        res["reasons"].append("jit_not_launched")
    if res["forbidden"]:
        res["reasons"].append("forbidden")
    res["ok"] = res["parses"] and res["jit_kernels"] >= 1 and res["launched"] and res["has_modelnew"] \
        and not res["forbidden"] and "no_ModelNew.forward" not in res["reasons"]
    return res


# ----------------------------------------------------------------------------- family tags
FAMILY_PATTERNS = [
    ("softmax", r"softmax|Softmax"),
    ("layernorm", r"layer_norm|LayerNorm"),
    ("rmsnorm", r"rms_norm|RMSNorm|RmsNorm|\brms\b|rsqrt\([^\n]*pow\(2\)[^\n]*mean|pow\(2\)\.mean\([^\n]*rsqrt"),
    ("cross_entropy", r"cross_entropy|CrossEntropyLoss|nll_loss|NLLLoss"),
    ("rotary", r"rotary|Rotary|\brope\b|RoPE|rotate_half|freqs_cis|apply_rotary"),
    ("attention", r"scaled_dot_product_attention|MultiheadAttention|[Aa]ttention|\battn\b"),
    ("matmul", r"matmul|\bmm\(|\bbmm\(|einsum|nn\.Linear|\bLinear\(|F\.linear|addmm|baddbmm|\S\s*@\s*\S"),
    ("activation", r"relu|ReLU|gelu|GELU|silu|SiLU|sigmoid|Sigmoid|tanh|Tanh|swish|mish|Mish|\belu\b|ELU|hardswish|Hardswish|softplus|Softplus|hardtanh|Hardtanh|selu|SELU|\bglu\b|GLU|hardsigmoid|Hardsigmoid"),
    ("conv", r"conv[123]d|Conv[123]d|conv_transpose|ConvTranspose"),
    ("reduce", r"\.(sum|mean|amax|amin|max|min|prod|argmax|argmin|norm|var|std|logsumexp|cumsum|cumprod|all|any)\(|torch\.(sum|mean|amax|amin|max|min|prod|argmax|argmin|norm|var|std|logsumexp|cumsum|cumprod)\("),
]
MLP_NAME_RE = re.compile(r"\bmlp\b|MLP|[Ff]eed[_]?[Ff]orward|\bffn\b|FFN|gate_proj|up_proj|down_proj")
LINEAR_RE = re.compile(r"nn\.Linear\(|\bLinear\(|F\.linear\(")


def _strip_comments(src: str) -> str:
    src = re.sub(r'("""|\'\'\')(.*?)\1', "", src, flags=re.S)
    return re.sub(r"#[^\n]*", "", src)


def family_tags(reference: str) -> list[str]:
    s = _strip_comments(reference)
    tags = [name for name, pat in FAMILY_PATTERNS if re.search(pat, s)]
    if MLP_NAME_RE.search(s) or (len(LINEAR_RE.findall(s)) >= 2 and "activation" in tags):
        tags.append("mlp")
    return tags or ["other"]


# ----------------------------------------------------------------------------- helpers
def toks(*texts: str) -> float:
    return sum(len(t) for t in texts) / 3.5


def pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 1)


def _only_import_try(st: dict) -> bool:
    """Row fails static only because of a `try: import ...` guard."""
    return (not st["ok"]) and st.get("try_import_guard_only", False) and st["forbidden"] == ["try"] \
        and set(st["reasons"]) == {"forbidden"}


ALLOW_IMPORT_GUARD = True   # lead decision 2026-10-06: a try/except around `import triton` is not a
                            # PyTorch fallback; the GPU launch check rejects any real fallback.


def allow_guard(st: dict) -> dict:
    if ALLOW_IMPORT_GUARD and _only_import_try(st):
        st["ok"] = True
        st["ok_via_import_guard"] = True
    return st


def is_good(r: dict | None) -> bool:
    return bool(r) and r.get("correctness") is True and r.get("decoy_kernel") is not True \
        and r.get("speedup") is not None and r["speedup"] > 0


def round_results(row: dict) -> list[dict | None]:
    """Per-round (1..5) parsed result. Round 5 is only known via final_speedup if best_round==5."""
    msgs = row["messages"]
    out = []
    for k in range(1, 6):
        if 2 * k < len(msgs):
            fbt = feedback_block(msgs[2 * k]["content"])
            r = parse_feedback(fbt)
            r["source"] = f"feedback_after_round_{k}"
            r["feedback"] = fbt
            out.append(r)
        elif row["best_round"] == k:
            fs = row["final_speedup"]
            out.append({"status": None, "compiled": None, "correctness": fs is not None and fs > 0,
                        "decoy_kernel": None, "speedup": fs, "source": "final_speedup (no feedback in trajectory)",
                        "feedback": None})
        else:
            out.append(None)
    return out


# ----------------------------------------------------------------------------- main
def main() -> None:
    rows = load_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cnt = Counter()
    cnt["rows_total"] = len(rows)
    main_rows, repair_rows = [], []
    fam_hist_main, fam_hist_repair = Counter(), Counter()
    static_reasons_main, static_reasons_repair = Counter(), Counter()
    best_round_hist = Counter()
    repair_kind = Counter()

    for row in rows:
        uuid = row["uuid"]
        assert row["entry_point"] == "Model" and row["num_rounds"] == 5
        msgs = row["messages"]
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant"] * 5, roles
        reference = row["original_python_code"]
        fam = family_tags(reference)
        results = round_results(row)
        codes = [extract_code(msgs[2 * k - 1]["content"]) for k in range(1, 6)]
        meta_common = {"uuid": uuid, "revision": REVISION, "dataset": REPO_ID, "license": LICENSE_NOTE,
                       "num_rounds": row["num_rounds"],
                       "round_speedups": [r["speedup"] if r else None for r in results],
                       "round_correct": [(is_good(r) if r else None) for r in results]}

        # ---------------- single-turn best round
        fs = row["final_speedup"]
        b = row["best_round"]
        if fs is None or not math.isfinite(fs):
            cnt["main_drop_nonfinite_speedup"] += 1
        elif fs < MIN_SPEEDUP:
            cnt["main_drop_speedup_lt_1"] += 1
        else:
            cnt["main_after_speedup_filter"] += 1
            code, nblk = codes[b - 1]
            br = results[b - 1]
            if code is None:
                cnt["main_drop_no_code_block"] += 1
            elif br is not None and br.get("decoy_kernel") is True:
                cnt["main_drop_best_round_decoy"] += 1
            else:
                cnt["main_written"] += 1
                best_round_hist[b] += 1
                st = allow_guard(static_check(code))
                for rsn in st["reasons"]:
                    static_reasons_main[rsn] += 1
                for f in st["forbidden"]:
                    static_reasons_main["forbidden:" + f.split(" in ")[0].split(":")[0]] += 1
                cnt["main_static_ok" if st["ok"] else "main_static_fail"] += 1
                for t in fam:
                    fam_hist_main[t] += 1
                main_rows.append({
                    "id": f"{SOURCE}:{uuid}", "source": SOURCE, "format": "modelnew",
                    "reference": reference, "kernel_type": None, "target": code, "family": fam,
                    "repair": None,
                    "meta": {**meta_common, "best_round": b, "final_speedup": fs,
                             "best_round_result_source": br["source"] if br else None,
                             "n_code_blocks_in_answer": nblk,
                             "tokens_est": round(toks(reference, code), 1)},
                    "static": st,
                })

        # ---------------- repair pairs: round k (failed) + feedback k -> round k+1 (better)
        for k in range(1, 5):
            cnt["repair_pairs_considered"] += 1
            rk, rn = results[k - 1], results[k]
            code_k, _ = codes[k - 1]
            code_n, _ = codes[k]
            good_k = is_good(rk)
            if good_k and rk["speedup"] >= MIN_SPEEDUP:
                cnt["repair_skip_turn_not_failed"] += 1
                continue
            if rn is None:
                cnt["repair_skip_next_result_unknown"] += 1  # round 5 without feedback, not best
                continue
            if not is_good(rn):
                cnt["repair_skip_next_not_better"] += 1
                continue
            if good_k:
                if not (rn["speedup"] > rk["speedup"]):
                    cnt["repair_skip_next_not_faster"] += 1
                    continue
                kind = "slow_to_faster"
            else:
                kind = "incorrect_to_correct"
            if code_k is None:
                cnt["repair_skip_failed_turn_no_code"] += 1
                continue
            if code_n is None:
                cnt["repair_skip_next_no_code"] += 1
                continue
            if code_k.strip() == code_n.strip():
                cnt["repair_skip_identical_code"] += 1
                continue
            fb_full = rk["feedback"]
            fb, truncated = truncate_feedback(fb_full)
            st = allow_guard(static_check(code_n))
            for rsn in st["reasons"]:
                static_reasons_repair[rsn] += 1
            for f in st["forbidden"]:
                static_reasons_repair["forbidden:" + f.split(" in ")[0].split(":")[0]] += 1
            cnt["repair_written"] += 1
            cnt["repair_static_ok" if st["ok"] else "repair_static_fail"] += 1
            repair_kind[kind] += 1
            for t in fam:
                fam_hist_repair[t] += 1
            repair_rows.append({
                "id": f"{SOURCE}:{uuid}:r{k}to{k + 1}", "source": SOURCE, "format": "modelnew",
                "reference": reference, "kernel_type": None, "target": code_n, "family": fam,
                "repair": {"failed_code": code_k, "feedback": fb},
                "meta": {**meta_common, "failed_round": k, "fixed_round": k + 1, "repair_kind": kind,
                         "failed_status": rk.get("status"), "failed_compiled": rk.get("compiled"),
                         "failed_correctness": rk.get("correctness"),
                         "failed_decoy_kernel": rk.get("decoy_kernel"), "failed_speedup": rk.get("speedup"),
                         "fixed_speedup": rn["speedup"], "fixed_result_source": rn["source"],
                         "feedback_chars_full": len(fb_full), "feedback_truncated": truncated,
                         "tokens_est": round(toks(reference, code_k, fb, code_n), 1)},
                "static": st,
            })

    for path, data in ((OUT_MAIN, main_rows), (OUT_REPAIR, repair_rows)):
        with open(path, "w") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def tokstats(data, key_ok=None):
        xs = [r["meta"]["tokens_est"] for r in data if key_ok is None or r["static"]["ok"] == key_ok]
        return {"n": len(xs), "p50": pct(xs, 0.5), "p90": pct(xs, 0.9), "max": round(max(xs), 1) if xs else None}

    fs_kept = [r["meta"]["final_speedup"] for r in main_rows]
    summary = {
        "dataset": REPO_ID, "revision": REVISION, "script": "modal_app/sft_ingest_drkernel.py",
        "best_round_identification": "dataset column `best_round` (1-based); final_speedup == that round's "
                                     "feedback speedup for all best_round<5 rows. Round 5 has no feedback "
                                     "message, so its result is known only through final_speedup.",
        "counts": dict(cnt),
        "main": {
            "written": len(main_rows),
            "static_ok": sum(r["static"]["ok"] for r in main_rows),
            "static_fail": sum(not r["static"]["ok"] for r in main_rows),
            "static_fail_reasons": dict(static_reasons_main.most_common()),
            "static_fail_only_import_guard_try": sum(_only_import_try(r["static"]) for r in main_rows),
            "static_ok_via_import_guard": sum(bool(r["static"].get("ok_via_import_guard")) for r in main_rows),
            "best_round_hist": {str(k): v for k, v in sorted(best_round_hist.items())},
            "final_speedup_p10_p50_p90": [pct(fs_kept, 0.1), pct(fs_kept, 0.5), pct(fs_kept, 0.9)],
            "final_speedup_gt_100x": sum(x > 100 for x in fs_kept),
            "family_hist": dict(fam_hist_main.most_common()),
            "family_hist_static_ok": dict(Counter(t for r in main_rows if r["static"]["ok"] for t in r["family"]).most_common()),
            "tokens_prompt_plus_target_chars_div_3.5": {"all": tokstats(main_rows), "static_ok": tokstats(main_rows, True)},
        },
        "repair": {
            "written": len(repair_rows),
            "static_ok": sum(r["static"]["ok"] for r in repair_rows),
            "static_fail": sum(not r["static"]["ok"] for r in repair_rows),
            "static_fail_reasons": dict(static_reasons_repair.most_common()),
            "static_fail_only_import_guard_try": sum(_only_import_try(r["static"]) for r in repair_rows),
            "static_ok_via_import_guard": sum(bool(r["static"].get("ok_via_import_guard")) for r in repair_rows),
            "repair_kind": dict(repair_kind),
            "feedback_truncated": sum(r["meta"]["feedback_truncated"] for r in repair_rows),
            "family_hist": dict(fam_hist_repair.most_common()),
            "tokens_reference_failedcode_feedback_target_chars_div_3.5": {"all": tokstats(repair_rows), "static_ok": tokstats(repair_rows, True)},
        },
        "notes": [
            "static.ok = parses & >=1 @triton.jit & a jit kernel launched as name[grid](...) & class ModelNew "
            "with forward & no forbidden (extern_kernels, try:, inherits Model, torch.matmul/@/torch.nn.functional "
            "calls in ModelNew.forward or in-file helpers it reaches).",
            "static.try_import_guard_only marks rows whose only try blocks guard imports; they still fail "
            "(rule: no try:), see static_fail_only_import_guard_try for how many would pass otherwise.",
            "Forward-path check is static and includes CPU/fallback branches, so it is conservative.",
            "tl_symbols are recorded but not yet checked against the Triton 3.2 allowlist.",
            "Rows whose best round reported decoy_kernel=true are dropped (counted); round-5 decoy status is unknown.",
            "No KernelBench/val-set decontamination is done here (later stage).",
        ],
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"counts": summary["counts"], "main_static_ok": summary["main"]["static_ok"],
                      "repair_static_ok": summary["repair"]["static_ok"]}, indent=2))


if __name__ == "__main__":
    main()
