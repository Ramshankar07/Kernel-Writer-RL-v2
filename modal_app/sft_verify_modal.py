"""
SFT candidate GPU verification (Phase 1 steps 2-3), Modal app `autokernel-sft-verify`.

Evaluation logic: modal_app/sft_verify_core.py (one forked process per candidate, 60 s default
timeout; modelnew via kb_v2_eval helpers, autokernel via bench_v2_core, both imported unchanged).
GPU L40S, falling back to L4; max_containers=4 (README rule: <=4 for verification).

    cd modal_app
    modal deploy sft_verify_modal.py
    modal run sft_verify_modal.py::allowlist                 # -> results/v3/sft/tl_allowlist_triton32.json
    modal run sft_verify_modal.py::smoke                     # 10 GPU controls + 1 static control
    modal run sft_verify_modal.py::run --source drkernel --limit 0      # 0 = all
    modal run sft_verify_modal.py::run --source kernelbook --limit 200

`run` reads results/v3/sft/candidates_<source>.jsonl, skips static.ok == false, rejects rows whose
tl.* symbols are outside the Triton 3.2 allowlist (no GPU), verifies the rest in batches of ~50
and appends results/v3/sft/verified_<source>.jsonl (row + `verify`) as batches finish. Re-running
resumes: ids already in verified_<source>.jsonl are skipped. Run summary (verdict counts, seconds
per candidate, $ estimate) -> verified_<source>.summary.json.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import sys
import time

import modal

from common import bench_base_image

APP = "autokernel-sft-verify"
MAX_CONTAINERS = 4
GPU = ["L40S", "L4"]          # Modal falls back to L4 when no L40S is available
CPU, MEM_MB = 4.0, 16384
# Modal list prices (modal.com/pricing, checked 2026-09-30): GPU per hour, + CPU core-h, + GiB-h.
GPU_PER_H = {"L40S": 1.95, "L4": 0.80, "A10G": 1.10, "H100": 3.95}
CPU_CORE_H, MEM_GIB_H = 0.0472, 0.008
DEFAULT_TIMEOUT = 60

ROOT = pathlib.Path(__file__).resolve().parents[1]
SFT = ROOT / "results" / "v3" / "sft"
ALLOW = SFT / "tl_allowlist_triton32.json"

app = modal.App(APP)
image = bench_base_image.add_local_python_source("common", "bench_v2_core", "kb_v2_eval", "sft_verify_core")


def _gpu_name():
    import subprocess
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"


@app.cls(image=image, gpu=GPU, cpu=CPU, memory=MEM_MB, timeout=4 * 3600,
         max_containers=MAX_CONTAINERS, scaledown_window=30)
class SFTVerify:
    @modal.method()
    def allowlist(self) -> dict:
        import importlib.util
        import subprocess
        import tempfile
        core = importlib.util.find_spec("sft_verify_core").origin
        out = os.path.join(tempfile.mkdtemp(), "allow.json")
        subprocess.run([sys.executable, core, "allowlist", out], check=True)
        return json.load(open(out))

    @modal.method()
    def verify_batch(self, jobs: list) -> dict:
        """Runs the zygote over `jobs`; if the zygote itself dies, restarts it on what's left."""
        import importlib.util
        import subprocess
        import tempfile
        t_start = time.time()
        core = importlib.util.find_spec("sft_verify_core").origin
        td = tempfile.mkdtemp()
        results, remaining, attempt = {}, list(jobs), 0
        while remaining and attempt < 4:
            attempt += 1
            jp, op, pp, lp = (os.path.join(td, f"{n}{attempt}") for n in ("jobs", "out", "prog", "log"))
            json.dump(remaining, open(jp, "w"))
            open(pp, "w").close()
            budget = sum(2.0 * float(j.get("timeout") or DEFAULT_TIMEOUT) + 10 for j in remaining) + 120
            with open(lp, "w") as log:
                try:
                    subprocess.run([sys.executable, core, "batch", jp, op, pp], stdout=log,
                                   stderr=subprocess.STDOUT, timeout=budget, cwd=td,
                                   env={**os.environ, "AUTOKERNEL_DIR": "/autokernel"})
                except subprocess.TimeoutExpired:
                    pass
            if os.path.exists(op):
                for line in open(op):
                    r = json.loads(line)
                    results[r["id"]] = r
            tail = open(lp).read()[-800:]
            before = len(remaining)
            remaining = [j for j in remaining if j["id"] not in results]
            if remaining and len(remaining) == before:      # zygote can't even start one
                for j in remaining:
                    results[j["id"]] = {"id": j["id"], "verdict": "INFRA_ERROR", "reason": "runner: " + tail}
                remaining = []
            elif remaining:                                  # zygote died mid-candidate: blame it
                bad = remaining[0]
                results[bad["id"]] = {"id": bad["id"], "verdict": "CRASH", "reason": "runner died: " + tail[-400:]}
                remaining = remaining[1:]
        for j in remaining:
            results[j["id"]] = {"id": j["id"], "verdict": "INFRA_ERROR", "reason": "runner restarted too often"}
        return {"gpu": _gpu_name(), "container_s": round(time.time() - t_start, 2),
                "results": [results[j["id"]] for j in jobs]}


# =============================================================================================
# local side
# =============================================================================================
def tl_symbols(src: str):
    """tl.* attribute chains used in `src` (normalized: 'load', 'math.exp', 'extra.cuda.libdevice.tanh')."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    roots = {"tl": "", "triton.language": ""}       # alias -> prefix inside triton.language
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "triton.language" or a.name.startswith("triton.language."):
                    if a.asname:
                        roots[a.asname] = a.name[len("triton.language"):].lstrip(".")
        elif isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "triton.language" or node.module.startswith("triton.language.")
                or node.module == "triton"):
            base = "" if node.module in ("triton", "triton.language") else node.module[len("triton.language."):]
            for a in node.names:
                if node.module == "triton":
                    if a.name == "language":
                        roots[a.asname or "language"] = ""
                    continue
                full = (base + "." + a.name).lstrip(".")
                if a.name in ("libdevice", "math", "extra", "cuda"):   # namespaces, not symbols
                    roots[a.asname or a.name] = full
    out = set()

    def chain(n):
        parts = []
        while isinstance(n, ast.Attribute):
            parts.append(n.attr)
            n = n.value
        if isinstance(n, ast.Name):
            parts.append(n.id)
            return list(reversed(parts))
        return None

    seen_inner = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and id(node) not in seen_inner:
            c = chain(node)
            if not c:
                continue
            # mark inner attributes so a.b.c is counted once
            v = node.value
            while isinstance(v, ast.Attribute):
                seen_inner.add(id(v))
                v = v.value
            for k in range(len(c) - 1, 0, -1):
                root = ".".join(c[:k])
                if root in roots:
                    pre = roots[root]
                    sym = ".".join(([pre] if pre else []) + c[k:])
                    out.add(sym)
                    break
    return sorted(out)


def load_allow():
    a = json.load(open(ALLOW))
    return {"tl": set(a["tl"]), "math": set(a["tl.math"]),
            "libdevice": set(a["tl.extra.cuda.libdevice"]) | set(a["tl.extra.libdevice"])}


def symbol_ok(sym: str, allow) -> bool:
    s = sym
    for p in ("triton.language.", "tl."):
        if s.startswith(p):
            s = s[len(p):]
    parts = s.split(".")
    if len(parts) == 1:
        return parts[0] in allow["tl"]
    if parts[0] == "math" and len(parts) == 2:
        return parts[1] in allow["math"]
    if "libdevice" in parts and parts[-1] != "libdevice":
        return parts[-1] in allow["libdevice"]
    if parts[0] in allow["tl"]:          # e.g. tl.float32.something / tl.core.x: top-level must exist
        return True
    return False


def bad_symbols(row, allow):
    syms = (row.get("static") or {}).get("tl_symbols")
    if syms is None:
        syms = tl_symbols(row.get("target", "")) or []
    return sorted({s for s in syms if not symbol_ok(s, allow)})


def ensure_allowlist():
    if ALLOW.exists():
        return
    a = SFTVerify().allowlist.remote()
    SFT.mkdir(parents=True, exist_ok=True)
    ALLOW.write_text(json.dumps(a, indent=1))
    print("wrote", ALLOW, "triton", a["triton"], len(a["tl"]), "tl names", flush=True)


def to_job(row, timeout):
    fmt = row.get("format", "modelnew")
    j = {"id": row["id"], "format": fmt, "target": row["target"],
         "timeout": timeout * (2 if fmt == "autokernel" else 1)}
    if fmt == "autokernel":
        j["kernel_type"] = row.get("kernel_type") or row.get("reference")
    else:
        j["reference"] = row["reference"]
    return j


def cost(container_s_by_gpu: dict) -> float:
    tot = 0.0
    for g, s in container_s_by_gpu.items():
        rate = next((v for k, v in GPU_PER_H.items() if k.lower() in g.lower()), GPU_PER_H["L40S"])
        tot += s / 3600.0 * (rate + CPU * CPU_CORE_H + MEM_MB / 1024 * MEM_GIB_H)
    return round(tot, 4)


def verify_rows(rows, timeout, batch, sink):
    """Static symbol check locally, then GPU batches. `sink(row, verify)` per result."""
    allow = load_allow()
    jobs, by_id = [], {}
    for r in rows:
        by_id[r["id"]] = r
        bad = bad_symbols(r, allow)
        if bad:
            sink(r, {"verdict": "REJECT_SYMBOL", "reason": f"tl symbols not in Triton 3.2: {bad}",
                     "bad_symbols": bad, "launch_seen": False, "cases": [], "wall_s": 0.0})
        else:
            jobs.append(to_job(r, timeout))
    batches = [jobs[i:i + batch] for i in range(0, len(jobs), batch)]
    stats = {"container_s": {}, "batches": len(batches), "n_gpu": len(jobs)}
    if not batches:
        return stats
    for out in SFTVerify().verify_batch.map(batches, order_outputs=False, return_exceptions=True):
        if isinstance(out, Exception):
            print("batch error:", repr(out)[:300], flush=True)
            continue
        stats["container_s"][out["gpu"]] = stats["container_s"].get(out["gpu"], 0.0) + out["container_s"]
        for v in out["results"]:
            v.setdefault("gpu", out["gpu"])
            sink(by_id[v["id"]], {k: x for k, x in v.items() if k != "id"})
    return stats


def summarize(vs, stats):
    from collections import Counter
    walls = sorted(v["wall_s"] for v in vs if v.get("wall_s"))
    s = {"n": len(vs), "verdicts": dict(Counter(v["verdict"] for v in vs)),
         "per_candidate_s_mean": round(sum(walls) / len(walls), 2) if walls else None,
         "per_candidate_s_median": walls[len(walls) // 2] if walls else None,
         "container_s": stats["container_s"], "batches": stats["batches"],
         "est_usd": cost(stats["container_s"])}
    n = stats.get("n_gpu") or 0
    if n:
        s["container_s_per_candidate"] = round(sum(stats["container_s"].values()) / n, 2)
    return s


@app.local_entrypoint()
def allowlist():
    if ALLOW.exists():
        ALLOW.unlink()
    ensure_allowlist()


@app.local_entrypoint()
def smoke(timeout: int = DEFAULT_TIMEOUT):
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from sft_verify_controls import controls, static_controls
    ensure_allowlist()
    ctl = controls() + static_controls()
    meta = {row["id"]: (name, cat, exp) for name, cat, exp, row in ctl}
    got = {}
    t0 = time.time()
    stats = verify_rows([row for *_, row in ctl], timeout, 50, lambda r, v: got.__setitem__(r["id"], v))
    wall = round(time.time() - t0, 1)
    out = SFT / "smoke_verify.jsonl"
    rows = []
    with out.open("w") as f:
        for name, cat, exp, row in ctl:
            v = got.get(row["id"], {"verdict": "MISSING"})
            ok = v["verdict"] == exp
            rows.append((name, cat, exp, v["verdict"], ok, v.get("wall_s"), v.get("launch_seen"),
                         v.get("second_shape", "-"), (v.get("reason") or "")[:90]))
            f.write(json.dumps({"control": name, "category": cat, "expected": exp, "match": ok,
                                "verify": v}, default=str) + "\n")
    print(f"\n{'control':32} {'category':20} {'expected':13} {'got':13} ok   wall_s launch  shape2 | reason")
    for r in rows:
        print(f"{r[0]:32} {r[1]:20} {r[2]:13} {r[3]:13} {'Y' if r[4] else 'N':4} {r[5]!s:6} {r[6]!s:6} "
              f"{str(r[7])[:28]:28} | {r[8]}")
    s = summarize([got[k] for k in got if not k.endswith("static_bad_tl_symbol")], stats)
    s.update(local_wall_s=wall, all_match=all(r[4] for r in rows))
    (SFT / "smoke_verify.summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))


@app.local_entrypoint()
def run(source: str, limit: int = 0, batch: int = 50, timeout: int = DEFAULT_TIMEOUT):
    ensure_allowlist()
    src = SFT / f"candidates_{source}.jsonl"
    dst = SFT / f"verified_{source}.jsonl"
    done = set()
    if dst.exists():
        for line in open(dst):
            try:
                done.add(json.loads(line)["id"])
            except Exception:  # noqa: BLE001
                pass
    rows, n_static_skip, n_done = [], 0, 0
    for line in open(src):
        r = json.loads(line)
        if (r.get("static") or {}).get("ok") is False:
            n_static_skip += 1
            continue
        if r["id"] in done:
            n_done += 1
            continue
        rows.append(r)
    if limit:
        rows = rows[:limit]
    print(f"{source}: {len(rows)} to verify ({n_static_skip} static.ok=false skipped, {n_done} already verified)",
          flush=True)
    vs = []
    t0 = time.time()
    with dst.open("a") as f:
        def sink(row, v):
            vs.append(v)
            f.write(json.dumps({**row, "verify": v}, default=str) + "\n")
            f.flush()
            if len(vs) % 50 == 0:
                print(f"  {len(vs)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
        stats = verify_rows(rows, timeout, batch, sink)
    s = summarize(vs, stats)
    s.update(source=source, local_wall_s=round(time.time() - t0, 1), skipped_static=n_static_skip,
             skipped_done=n_done)
    sp = SFT / f"verified_{source}.summary.json"
    hist = json.loads(sp.read_text()) if sp.exists() else []
    hist = hist if isinstance(hist, list) else [hist]
    sp.write_text(json.dumps(hist + [s], indent=1))
    print(json.dumps(s, indent=1))
