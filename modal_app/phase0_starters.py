"""
Phase 0 step 1: verify the fixed starters (modal_app/starters_v2/) under bench v2.

Reuses the bench v2 app definition unchanged (bench_v2_modal.py / bench_v2_core.py); the
starters are passed in the job payload, so no harness file is edited.

    cd modal_app && modal run phase0_starters.py::verify    # -> results/v3/phase0_starters.jsonl
"""
import json
import pathlib

from bench_v2_modal import BenchV2, _batches, app  # noqa: F401  (app hosts the entrypoint)
from common import KERNELS

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
STARTERS_V2 = HERE / "starters_v2"
OUT = ROOT / "results" / "v3" / "phase0_starters.jsonl"


def starter_v2(kt: str) -> str:
    return (STARTERS_V2 / f"{kt}.py").read_text()


@app.local_entrypoint()
def verify(kernels: str = "", repeats: int = 1):
    kts = [k for k in (kernels.split(",") if kernels else KERNELS) if k]
    jobs = [{"id": f"starter_v2:{kt}:{i}", "kernel_type": kt, "code": starter_v2(kt),
             "starter_code": starter_v2(kt), "compile": True, "timing": True, "reps": 60,
             "early_abort": False}
            for kt in kts for i in range(repeats)]
    res = []
    for r in BenchV2().run_batch.map(_batches(jobs, 1), return_exceptions=True):
        if isinstance(r, Exception):
            print("batch error:", repr(r)[:300], flush=True)
            continue
        for x in r:
            print(x["id"], x.get("verdict"), x.get("wall_s"), (x.get("fail_reasons") or [])[:2],
                  x.get("reason", ""), flush=True)
        res += r
    mode = "a" if kernels else "w"
    with OUT.open(mode) as f:
        for r in res:
            f.write(json.dumps(r, default=str) + "\n")
    print("wrote", OUT)


@app.local_entrypoint()
def probe(kernel_type: str, files: str, out: str = "", timing: bool = False):
    """Bench v2 on ad-hoc kernel files (comma-separated paths), full case list (timing optional)."""
    jobs = []
    for p in files.split(","):
        code = pathlib.Path(p).read_text()
        jobs.append({"id": f"probe:{pathlib.Path(p).stem}", "kernel_type": kernel_type, "code": code,
                     "starter_code": starter_v2(kernel_type) if timing else None, "timing": timing,
                     "early_abort": False})
    res = []
    for r in BenchV2().run_batch.map([[j] for j in jobs], return_exceptions=True):
        if isinstance(r, Exception):
            print("batch error:", repr(r)[:300], flush=True)
            continue
        res += r
        for x in r:
            print(x["id"], x.get("verdict"), x.get("wall_s"), x.get("reason", ""), flush=True)
            for lab, t in (x.get("timing") or {}).items():
                if isinstance(t, dict) and "kernel" in t:
                    print("   T", lab, "kernel_us %.1f eager_us %.1f vs_starter_v2 %.3f" % (
                        t["kernel"]["median_us"], t["eager"]["median_us"], t["vs_starter"]["median"]), flush=True)
            for c in (x.get("correctness") or {}).get("cases", []):
                if c["ok"] is not True:
                    print("   ", c["stage"], c["label"], c["dtype"], c["ok"], c.get("reason", "")[:160], flush=True)
    if out:
        with open(out, "w") as f:
            for r in res:
                f.write(json.dumps(r, default=str) + "\n")
