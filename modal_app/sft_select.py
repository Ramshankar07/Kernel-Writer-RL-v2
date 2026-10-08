"""
Budgeted selection of SFT candidates for GPU verification (Phase 1 budget ~$11; verifying
everything was estimated at $13-20).

  candidates_kernelbook.jsonl      -> candidates_kernelbook_sel.jsonl
      only families that Dr. Kernel lacks (norms, softmax, cross_entropy, rotary, attention,
      mlp, matmul); KernelBook is Inductor-style, so it fills gaps rather than forming the bulk.
  candidates_drkernel_repair.jsonl -> candidates_drkernel_repair_sel.jsonl
      a family-stratified random sample: repair should be ~15% of a 4-6k mix, so ~1.5k
      candidates leaves room for verification losses.

    python3 modal_app/sft_select.py
"""
import collections
import json
import pathlib
import random

SFT = pathlib.Path(__file__).resolve().parents[1] / "results" / "v3" / "sft"
KB_FAMILIES = {"softmax", "layernorm", "rmsnorm", "cross_entropy", "rotary", "attention", "mlp", "matmul"}
REPAIR_N = 1500
SEED = 0


def rows(name):
    with (SFT / f"candidates_{name}.jsonl").open() as f:
        return [json.loads(l) for l in f if l.strip()]


def write(name, rs):
    with (SFT / f"candidates_{name}.jsonl").open("w") as f:
        for r in rs:
            f.write(json.dumps(r) + "\n")


def main():
    kb = [r for r in rows("kernelbook")
          if r["static"].get("ok") and KB_FAMILIES & set(r["family"])]
    write("kernelbook_sel", kb)

    rep = [r for r in rows("drkernel_repair") if r["static"].get("ok")]
    by = collections.defaultdict(list)
    for r in rep:
        by[r["family"][0] if r["family"] else "other"].append(r)
    rng = random.Random(SEED)
    for v in by.values():
        rng.shuffle(v)
    # round-robin across families so scarce families are kept whole before common ones are cut
    sel, i = [], 0
    while len(sel) < REPAIR_N and any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v) and len(sel) < REPAIR_N:
                sel.append(v[i])
        i += 1
    write("drkernel_repair_sel", sel)

    summary = {
        "kernelbook_sel": len(kb),
        "kernelbook_sel_families": dict(collections.Counter(t for r in kb for t in r["family"]).most_common()),
        "drkernel_repair_sel": len(sel),
        "drkernel_repair_sel_families": dict(collections.Counter(r["family"][0] if r["family"] else "other" for r in sel).most_common()),
        "seed": SEED,
    }
    (SFT / "select_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
