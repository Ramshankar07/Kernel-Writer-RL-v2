"""
SFT Phase 1: ingest GPUMODE/KernelBook @ 1576375b (MIT) into the shared candidate schema
(results/v3/sft/README.md). Local CPU only, re-runnable (HF cache is reused).

    python modal_app/sft_ingest_kernelbook.py
      -> results/v3/sft/candidates_kernelbook.jsonl
      -> results/v3/sft/ingest_kernelbook_summary.json

Columns used (dataset_permissive.parquet): uuid, entry_point, python_code (PyTorch module +
get_inputs/get_init_inputs), triton_code (TorchInductor output defining `<entry_point>New`),
repo_name, module_name, licenses, stars, sha, repo_link, synthetic.

Steps (counts in the summary):
  1. load                     all rows
  2. pure_triton              no `extern_kernels` in triton_code (cuBLAS/cuDNN fallbacks dropped)
  3. rename                   class <entry_point> -> Model in the prompt, <entry_point>New -> ModelNew in
                              the target (word-boundary rename; dropped if `Model`/`ModelNew` already
                              names something else, or if the renamed source no longer parses)
  4. static                   sft_static.static_check(target, "modelnew") ok: parses, >=1 @triton.jit
                              launched as name[grid](...), no extern_kernels / try / inheriting the
                              reference / matmul/@/F.*/torch.ops.aten on the ModelNew.forward path
  5. dedup                    drop rows whose (prompt AST, target AST) pair duplicates an earlier uuid
static.inductor_imports lists every `torch._inductor...` import of the target; these rows were
generated with torch 2.5 and GPU verification under torch 2.6 drops the ones that break.
"""
import ast
import collections
import io
import json
import pathlib
import re
import statistics
import sys
import tokenize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sft_static import ast_key, static_check  # noqa: E402

REPO = "GPUMODE/KernelBook"
REVISION = "1576375bc92745b490e1cdf2fce01eba76d9f847"
FILE = "dataset_permissive.parquet"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v3" / "sft"

FAMILIES = [  # (tag, regex over the PyTorch prompt); multi-label, "other" if none match
    ("softmax", r"softmax|Softmax"),
    ("layernorm", r"layer_norm|LayerNorm|layernorm|Layernorm"),
    ("rmsnorm", r"rms_norm|RMSNorm|RmsNorm|rmsnorm|\brms\b"),
    ("cross_entropy", r"cross_entropy|CrossEntropy|nll_loss|NLLLoss|log_softmax.*gather"),
    ("rotary", r"rotary|Rotary|\brope\b|RoPE|apply_rotary|rotate_half"),
    ("attention", r"[Aa]ttention|scaled_dot_product|MultiheadAttention|\battn\b"),
    ("mlp", r"\bMLP\b|Mlp|FeedForward|FeedFoward|FFN|\bmlp\b|SwiGLU|GEGLU|GLU\b"),
    ("matmul", r"torch\.matmul|torch\.mm\b|torch\.bmm|\.matmul\(|\.mm\(|\.bmm\(|\s@\s|nn\.Linear|F\.linear|einsum"),
    ("conv", r"[Cc]onv[123]d|conv_transpose|ConvTranspose|F\.conv"),
    ("reduce", r"torch\.(sum|mean|amax|amin|prod|logsumexp|var|std|norm|max|min)\(|"
               r"\.(sum|mean|amax|amin|prod|logsumexp|var|std|norm)\("),
    ("activation", r"relu|ReLU|gelu|GELU|silu|SiLU|sigmoid|Sigmoid|tanh|Tanh|\belu\b|ELU|leaky|Leaky|softplus|"
                   r"Softplus|hardswish|Hardswish|mish|Mish|swish|Swish|hardtanh|Hardtanh|selu|SELU|celu|CELU"),
]
FAM_RE = [(t, re.compile(p)) for t, p in FAMILIES]


def families(src):
    tags = [t for t, r in FAM_RE if r.search(src)]
    return tags or ["other"]


def inductor_imports(code):
    out = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("torch._inductor"):
            for a in n.names:
                out.add(f"{n.module}.{a.name}")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("torch._inductor"):
                    out.add(a.name)
    return sorted(out)


def torch_internal_refs(code):
    return sorted(set(re.findall(r"torch\._(?:C|dynamo|inductor)[\w\.]*", code)))


def rename(src, old, new):
    """Rename identifier `old` -> `new` at the token level: NAME tokens only, never an attribute
    (`nn.MSELoss` stays when the user class is also called MSELoss), never inside strings/comments."""
    lines = io.StringIO(src).readlines()  # same line split as tokenize
    offs = [0]
    for ln in lines:
        offs.append(offs[-1] + len(ln))
    hits, prev = [], None
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.NAME and tok.string == old and not (prev is not None and prev.string == "."):
            hits.append(offs[tok.start[0] - 1] + tok.start[1])
        if tok.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT):
            prev = tok
    for h in reversed(hits):
        src = src[:h] + new + src[h + len(old):]
    return src


def toks(s):
    return len(s) / 3.5


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REVISION)
    df = pd.read_parquet(path)
    steps = collections.OrderedDict()
    steps["load"] = len(df)
    steps["synthetic_true"] = int(df["synthetic"].sum())

    df = df[~df["triton_code"].str.contains("extern_kernels", regex=False)]
    steps["pure_triton"] = len(df)

    drop = collections.Counter()
    forb = collections.Counter()
    seen_pairs = set()
    rows = []
    for r in df.sort_values("uuid").itertuples(index=False):
        ep = r.entry_point
        prompt, target = r.python_code, r.triton_code
        if ep != "Model":
            if re.search(r"\bModel\b", prompt) or re.search(r"\bModelNew\b", target):
                drop["rename_collision"] += 1
                continue
            if not re.search(rf"\b{re.escape(ep)}New\b", target):
                drop["no_entry_new_class"] += 1
                continue
            try:
                prompt = rename(prompt, ep, "Model")
                target = rename(target, ep + "New", "ModelNew")
                ast.parse(prompt)
            except (SyntaxError, tokenize.TokenError, IndentationError):
                drop["prompt_syntax_error"] += 1
                continue
        try:
            ast.parse(prompt)
        except SyntaxError:
            drop["prompt_syntax_error"] += 1
            continue
        st = static_check(target, "modelnew", ref_names=("Model", ep))
        if not st["ok"]:
            drop["static_fail"] += 1
            for f in st["forbidden"]:
                forb[f.split(".")[0] + "." + f.split(".")[1] if f.startswith("torch.ops.") else f] += 1
            continue
        pk, tk = ast_key(prompt), ast_key(target)
        if (pk, tk) in seen_pairs:
            drop["duplicate_pair"] += 1
            continue
        seen_pairs.add((pk, tk))
        st["inductor_imports"] = inductor_imports(target)
        st["torch_internal_refs"] = torch_internal_refs(target)
        st["has_get_inputs"] = "def get_inputs" in prompt
        st["has_get_init_inputs"] = "def get_init_inputs" in prompt
        st["prompt_ast_sha"], st["target_ast_sha"] = pk, tk
        rows.append({
            "id": f"kernelbook:{r.uuid}",
            "source": "kernelbook",
            "format": "modelnew",
            "reference": prompt,
            "kernel_type": None,
            "target": target,
            "family": families(r.python_code),  # original names (pre-rename)
            "repair": None,
            "meta": {"revision": REVISION, "dataset_license": "MIT", "file": FILE, "uuid": int(r.uuid),
                     "entry_point": ep, "module_name": r.module_name, "renamed_to_model": ep != "Model",
                     "repo_name": r.repo_name, "repo_link": r.repo_link, "repo_sha": r.sha,
                     "repo_licenses": list(r.licenses) if r.licenses is not None else [],
                     "stars": int(r.stars), "synthetic": bool(r.synthetic),
                     "generator": "TorchInductor (torch 2.5.0 per dataset card)",
                     "get_init_inputs_convention": "[args_list, kwargs_dict]"},
            "static": st,
        })
    steps["after_rename"] = steps["pure_triton"] - drop["rename_collision"] - drop["no_entry_new_class"] \
        - drop["prompt_syntax_error"]
    steps["static_ok"] = steps["after_rename"] - drop["static_fail"]
    steps["after_dedup"] = len(rows)

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidates_kernelbook.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    fam = collections.Counter(t for row in rows for t in row["family"])
    fam_primary = collections.Counter(row["family"][0] for row in rows)
    pt = [toks(r["reference"]) for r in rows]
    tt = [toks(r["target"]) for r in rows]
    tot = [a + b for a, b in zip(pt, tt)]
    ind = collections.Counter(i for r in rows for i in r["static"]["inductor_imports"])
    tls = collections.Counter(s for r in rows for s in r["static"]["tl_symbols"])
    summary = {
        "dataset": REPO, "revision": REVISION, "file": FILE, "license_at_revision": "MIT",
        "steps": steps, "drops": dict(drop), "static_fail_reasons": dict(forb.most_common()),
        "n_candidates": len(rows),
        "family_histogram": dict(fam.most_common()),
        "family_primary_histogram": dict(fam_primary.most_common()),
        "n_multi_family": sum(len(r["family"]) > 1 for r in rows),
        "tokens_chars_div_3_5": {
            "prompt": {"p50": pct(pt, 0.5), "p90": pct(pt, 0.9), "mean": statistics.mean(pt) if pt else None},
            "target": {"p50": pct(tt, 0.5), "p90": pct(tt, 0.9), "mean": statistics.mean(tt) if tt else None},
            "prompt_plus_target": {"p50": pct(tot, 0.5), "p90": pct(tot, 0.9),
                                   "n_gt_4096": sum(x > 4096 for x in tot)},
        },
        "inductor_imports": {"rows_with_any": sum(bool(r["static"]["inductor_imports"]) for r in rows),
                             "by_import": dict(ind.most_common())},
        "jit_kernels_per_row": {"p50": pct([r["static"]["jit_kernels"] for r in rows], 0.5),
                                "p90": pct([r["static"]["jit_kernels"] for r in rows], 0.9)},
        "tl_symbols_top": dict(tls.most_common(60)),
        "n_distinct_tl_symbols": len(tls),
        "distinct_prompts_ast": len({r["static"]["prompt_ast_sha"] for r in rows}),
        "distinct_targets_ast": len({r["static"]["target_ast_sha"] for r in rows}),
    }
    (OUT / "ingest_kernelbook_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("steps", "drops", "static_fail_reasons", "family_histogram",
                                              "tokens_chars_div_3_5")}, indent=1))


if __name__ == "__main__":
    main()
