"""
SFT candidate verifier, in-container core (run by sft_verify_modal.py; never imported locally).

    python sft_verify_core.py batch jobs.json out.jsonl progress.txt
    python sft_verify_core.py allowlist out.json

`batch` is a single-threaded "zygote": it imports torch + triton once (no CUDA init), installs
the Triton launch counter, then os.fork()s one child per candidate. Each child creates its own
CUDA context, evaluates one candidate, writes one JSON result and _exits. The parent kills a
child that exceeds its timeout (default 60 s), so a CUDA fault, segfault or GPU hang costs only
that candidate. Wall time is measured per child (fork -> exit).

format `modelnew` (reference Model vs candidate ModelNew; helpers from kb_v2_eval, unchanged):
  * weights: torch/numpy/python RNGs seeded identically before each constructor
    (kb_v2_eval.build), then Model's state_dict copied into ModelNew when the keys/shapes match
    (or, if only the names differ, when the ordered list of shapes matches: "copied_positional");
  * golden = reference Model in fp32 (params + floating inputs upcast), TF32 off globally;
  * 3 seeds x 2 shapes: shape 1 = the reference's get_inputs(); shape 2 = batch/row dim rescaled
    (2x, or an odd size when 2x would exceed 2 GiB of inputs), via a batch-like module global used
    only by get_inputs, else by redrawing the inputs with a new dim 0 when every tensor input is
    floating and shares dim 0. Shape 2 is `skipped` (with a reason) when neither is safe or the
    reference itself fails at the new shape;
  * allclose |out-gold| <= atol + rtol*|gold|: fp32 inputs 1e-4/1e-4, fp16/bf16 1e-2/1e-2, non-finite
    output where the golden is finite = fail. The 1e-2 verdict is recorded alongside (`loose_ok`);
  * reductions (sum/mean/norm/softmax/logsumexp/var/std in the reference source): inputs x1e3 and
    x1e-3 (shape 1, seed 0); outputs must be finite wherever the golden is finite;
  * launch check: every candidate forward must make >= 1 @triton.jit launch (JITFunction.run,
    as val_bench_core counts them); calls to torch compute ops (matmul/conv/softmax/norm/...)
    inside the candidate forward = FALLBACK.
format `autokernel`: bench_v2_core.correctness (golden reference, adversarial magnitudes,
determinism; imported unchanged) on kernel_fn, wrapped with the same launch / fallback checks.

Verdicts: PASS | FAIL (numerics, shape 1) | FAIL_SHAPE2 | FAIL_EXTREME | NO_LAUNCH | FALLBACK |
CRASH | TIMEOUT | REF_ERROR (reference itself broken -> row unusable, not the candidate's fault).
"""
from __future__ import annotations

import copy
import functools
import inspect
import json
import math
import os
import re
import signal
import sys
import tempfile
import time
import traceback

import torch
import triton  # noqa: F401
import triton.language as tl  # noqa: F401

HARNESS_VERSION = "sft_verify_v1"
TOL = {torch.float32: 1e-4, torch.float64: 1e-4, torch.float16: 1e-2, torch.bfloat16: 1e-2}
LOOSE = 1e-2
MAX_SHAPE2_BYTES = 2 * 2**30
BATCH_NAMES = ("batch_size", "batch", "bsz", "bs", "B", "M", "n_rows", "rows", "num_rows", "BATCH_SIZE")
REDUCTION_RE = re.compile(r"\b(sum|mean|norm|softmax|log_softmax|logsumexp|var|std|layer_norm|"
                          r"group_norm|batch_norm|instance_norm|rms_norm|cross_entropy|nll_loss|"
                          r"amax|amin|max|min|prod|cumsum|cumprod|LayerNorm|GroupNorm|BatchNorm\w*|"
                          r"InstanceNorm\w*|RMSNorm|Softmax|LogSoftmax|CrossEntropyLoss)\b")
# torch-level compute ops that, inside the candidate forward, mean "PyTorch did the real work"
FALLBACK_OPS = {
    "matmul", "mm", "bmm", "addmm", "addbmm", "baddbmm", "addmv", "mv", "dot", "einsum", "tensordot",
    "linear", "bilinear", "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
    "conv_transpose3d", "softmax", "log_softmax", "layer_norm", "group_norm", "batch_norm",
    "instance_norm", "rms_norm", "scaled_dot_product_attention", "cross_entropy", "nll_loss",
    "__matmul__", "__rmatmul__", "logsumexp", "embedding", "max_pool1d", "max_pool2d", "max_pool3d",
    "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool2d", "adaptive_max_pool2d",
    "multi_head_attention_forward", "lstm", "gru", "rnn_tanh", "rnn_relu",
}

# ---------------------------------------------------------------------------------------------
# Triton launch counter (same hook as val_bench_core.py): every kernel[grid](...) call, also via
# autotune/heuristics, ends in JITFunction.run.
# ---------------------------------------------------------------------------------------------
LAUNCHES = [0]
try:
    from triton.runtime.jit import JITFunction as _JF
    _orig_run = _JF.run

    def _counting_run(self, *a, **kw):
        if not kw.get("warmup", False):
            LAUNCHES[0] += 1
        return _orig_run(self, *a, **kw)

    _JF.run = _counting_run
    HOOK_OK = True
except Exception:  # noqa: BLE001
    HOOK_OK = False


class OpRecorder(torch.overrides.TorchFunctionMode):
    """Records torch-level function names called while active (torch.*, F.*, Tensor methods)."""

    def __init__(self):
        super().__init__()
        self.hits = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        n = getattr(func, "__name__", "") or ""
        if n in FALLBACK_OPS:
            self.hits[n] = self.hits.get(n, 0) + 1
        return func(*args, **(kwargs or {}))


def kb():
    import kb_v2_eval  # unchanged evaluator helpers (seed_all, load_module, build, alarm)
    return kb_v2_eval


# ---------------------------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------------------------
def _leaves(x):
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out += _leaves(v)
        return out
    return [x]


def compare(out, gold, tol, finite_only=False):
    """-> {ok, loose_ok, max_abs, max_rel, n_viol, n_nonfinite, reason}. gold computed in fp32."""
    os_, gs = _leaves(out), _leaves(gold)
    if len(os_) != len(gs):
        return {"ok": False, "loose_ok": False, "reason": f"output structure: {len(os_)} vs {len(gs)} leaves"}
    agg = {"ok": True, "loose_ok": True, "max_abs": 0.0, "max_rel": 0.0, "n_viol": 0, "n_nonfinite": 0, "reason": ""}
    for o, g in zip(os_, gs):
        if not torch.is_tensor(g):
            same = o == g
            if not same:
                agg.update(ok=False, loose_ok=False, reason=f"non-tensor mismatch {o!r} vs {g!r}"[:200])
            continue
        if not torch.is_tensor(o):
            agg.update(ok=False, loose_ok=False, reason=f"expected tensor, got {type(o).__name__}")
            continue
        if tuple(o.shape) != tuple(g.shape):
            agg.update(ok=False, loose_ok=False, reason=f"shape {tuple(o.shape)} vs {tuple(g.shape)}")
            continue
        if not (g.is_floating_point() or g.is_complex()):
            eq = torch.equal(o.to(g.dtype), g)
            if not eq:
                agg.update(ok=False, loose_ok=False, reason=f"integer/bool output differs")
            continue
        of, gf = o.detach().double(), g.detach().double()
        g_fin = torch.isfinite(gf)
        o_fin = torch.isfinite(of)
        bad_nf = g_fin & ~o_fin                       # non-finite where golden finite
        n_nf = int(bad_nf.sum())
        agg["n_nonfinite"] += n_nf
        if finite_only:
            if n_nf:
                agg.update(ok=False, loose_ok=False,
                           reason=f"{n_nf}/{g.numel()} non-finite outputs where the golden is finite")
            both = g_fin & o_fin
            if both.any():
                err = (of - gf).abs()[both]
                agg["max_rel"] = max(agg["max_rel"], float((err / gf.abs()[both].clamp_min(1e-30)).max()))
            continue
        # non-finite golden positions must be matched exactly (same inf / both nan)
        nf_mismatch = int((~g_fin & ~((of == gf) | (torch.isnan(of) & torch.isnan(gf)))).sum())
        err = (of - gf).abs()
        err = torch.where(g_fin & o_fin, err, torch.zeros_like(err))
        lim = tol + tol * gf.abs()
        llim = LOOSE + LOOSE * gf.abs()
        n_viol = int(((err > lim) & g_fin).sum()) + n_nf + nf_mismatch
        n_lviol = int(((err > llim) & g_fin).sum()) + n_nf + nf_mismatch
        agg["n_viol"] += n_viol
        agg["max_abs"] = max(agg["max_abs"], float(err.max()) if err.numel() else 0.0)
        if err.numel():
            agg["max_rel"] = max(agg["max_rel"], float((err / gf.abs().clamp_min(tol)).max()))
        if n_lviol:
            agg["loose_ok"] = False
        if n_viol:
            agg["ok"] = False
            agg["reason"] = (f"{n_viol}/{g.numel()} elems exceed atol=rtol={tol:g} "
                             f"(max_abs={agg['max_abs']:.3e}, nonfinite={n_nf})")
    if not agg["ok"] and not agg["reason"]:
        agg["reason"] = "mismatch"
    return agg


def exc_msg(e, head=300, tail=500):
    m = f"{type(e).__name__}: {e}"
    return m if len(m) <= head + tail + 5 else m[:head] + " [...] " + m[-tail:]


def _fatal(msg):
    m = msg.lower()
    return any(s in m for s in ("illegal memory", "unspecified launch", "device-side assert",
                                "cuda error", "misaligned address", "illegal instruction"))


# ---------------------------------------------------------------------------------------------
# modelnew
# ---------------------------------------------------------------------------------------------
def _float_inputs(inp, dtype=torch.float32, scale=None):
    out = []
    for v in inp:
        if torch.is_tensor(v) and v.is_floating_point():
            v = v.to(dtype) if dtype is not None else v
            if scale is not None:
                v = v * scale
        out.append(v)
    return out


def _low_dtype(inp):
    dts = [v.dtype for v in inp if torch.is_tensor(v) and v.is_floating_point()]
    for d in (torch.float16, torch.bfloat16):
        if d in dts:
            return d
    return torch.float32


def _nbytes(inp):
    return sum(v.numel() * v.element_size() for v in inp if torch.is_tensor(v))


def _new_dim(b, nbytes):
    if b <= 1:
        return 3
    if nbytes * 2 <= MAX_SHAPE2_BYTES:
        return 2 * b
    n = (b // 2) | 1
    return n if n != b else max(1, b - 2)


class Shape2:
    """Builds second-shape inputs for a seed, or says why it can't."""

    def __init__(self, ref_mod, inp1, model_src):
        self.mod, self.kind, self.reason = ref_mod, None, ""
        gi = ref_mod.get_inputs
        gi_names = set(getattr(gi, "__code__", None).co_names) if hasattr(gi, "__code__") else set()
        ginit = getattr(ref_mod, "get_init_inputs", None)
        init_names = set(ginit.__code__.co_names) if ginit is not None and hasattr(ginit, "__code__") else set()
        nb = _nbytes(inp1)
        for n in BATCH_NAMES:
            v = getattr(ref_mod, n, None)
            if isinstance(v, int) and not isinstance(v, bool) and n in gi_names and n not in init_names \
                    and not re.search(r"\b%s\b" % re.escape(n), model_src):
                self.kind, self.name, self.old = "global", n, v
                self.new = _new_dim(v, nb)
                self.desc = f"{n}: {v} -> {self.new}"
                return
        tens = [v for v in inp1 if torch.is_tensor(v)]
        others = [v for v in inp1 if not torch.is_tensor(v)]
        if not tens:
            self.reason = "no tensor inputs"
            return
        if not all(t.is_floating_point() and t.dim() >= 1 for t in tens):
            self.reason = "non-floating or 0-d tensor input"
            return
        d0 = {t.shape[0] for t in tens}
        if len(d0) != 1:
            self.reason = f"tensor inputs disagree on dim 0 {sorted(d0)}"
            return
        b = d0.pop()
        if any(isinstance(o, int) and o == b for o in others):
            self.reason = "a scalar input equals dim 0"
            return
        self.kind, self.new = "redraw", _new_dim(b, nb)
        self.stats = []
        for v in inp1:
            if torch.is_tensor(v):
                vf = v.float()
                self.stats.append((bool((vf >= 0).all()), float(vf.abs().max()), float(vf.mean()),
                                   float(vf.std()) if vf.numel() > 1 else 1.0))
            else:
                self.stats.append(None)
        self.desc = f"dim0: {b} -> {self.new} (redrawn)"

    def inputs(self, seed):
        K = kb()
        if self.kind == "global":
            setattr(self.mod, self.name, self.new)
            try:
                K.seed_all(seed)
                with torch.device("cuda"):
                    inp = list(self.mod.get_inputs())
            finally:
                setattr(self.mod, self.name, self.old)
            inp = K.to_cuda(inp)
            torch.cuda.synchronize()
            return inp
        K.seed_all(seed)
        base = K.gen_inputs(self.mod.get_inputs, seed, "gpu")
        out = []
        for v, st in zip(base, self.stats):
            if st is None:
                out.append(v)
                continue
            nonneg, amax, mean, std = st
            shp = (self.new,) + tuple(v.shape[1:])
            if nonneg:
                t = torch.rand(shp, device="cuda") * max(amax, 1e-6)
            else:
                t = torch.randn(shp, device="cuda") * (std if std > 0 else 1.0) + mean
            out.append(t.to(v.dtype))
        return out


def _copy_weights(ref, m):
    rs, cs = ref.state_dict(), m.state_dict()
    if not rs:
        return "none"
    if rs.keys() == cs.keys() and all(rs[k].shape == cs[k].shape for k in rs):
        m.load_state_dict(rs)
        return "copied"
    rv, cv = list(rs.values()), list(cs.values())
    if len(rv) == len(cv) and all(a.shape == b.shape for a, b in zip(rv, cv)):
        with torch.no_grad():
            for a, b in zip(rv, cv):
                b.copy_(a)
        return "copied_positional"
    return "seeded"


def _split_init(init):
    """KernelBook/paritybench convention: get_init_inputs() -> [args_list, kwargs_dict]."""
    init = list(init)
    if len(init) == 2 and isinstance(init[0], (list, tuple)) and isinstance(init[1], dict):
        return list(init[0]), dict(init[1])
    return init, {}


def _install_paritybench_stub():
    """KernelBook rows import `_mock_config` from paritybench's helper module (absent here):
    it builds a config object whose attributes are the keyword arguments."""
    import types
    if "_paritybench_helpers" not in sys.modules:
        m = types.ModuleType("_paritybench_helpers")
        m._mock_config = lambda **kw: types.SimpleNamespace(**kw)
        sys.modules["_paritybench_helpers"] = m


def _cand_forward(m, inp):
    rec = OpRecorder()
    LAUNCHES[0] = 0
    with torch.no_grad(), rec:
        out = m(*inp)
    torch.cuda.synchronize()
    return out, LAUNCHES[0], rec.hits


def eval_modelnew(job, d):
    K = kb()
    res = {"format": "modelnew", "cases": [], "launch_seen": False, "launches_min": None}
    t0 = time.time()
    ref_p, cand_p = os.path.join(d, "reference.py"), os.path.join(d, "candidate.py")
    open(ref_p, "w").write(job["reference"])
    open(cand_p, "w").write(job["target"])
    # --- reference ---
    try:
        _install_paritybench_stub()
        ref_mod = K.load_module(ref_p, "sft_reference")
        Model, get_inputs = ref_mod.Model, ref_mod.get_inputs
        get_init = getattr(ref_mod, "get_init_inputs", lambda: [])
        seed0 = int(job.get("seed") or int.from_bytes(os.urandom(3), "little"))
        K.seed_all(seed0)
        init_args, init_kwargs = _split_init(get_init())
        ref = K.build(functools.partial(Model, **init_kwargs), init_args, seed0)
        gold_model = ref if all(p.dtype == torch.float32 for p in ref.parameters()) else copy.deepcopy(ref).float()
        try:
            model_src = inspect.getsource(Model)
        except Exception:  # noqa: BLE001
            model_src = job["reference"]
    except Exception as e:  # noqa: BLE001
        res.update(verdict="REF_ERROR", reason="reference setup: " + exc_msg(e), tb=traceback.format_exc()[-800:])
        return res
    res["seeds"] = [seed0, seed0 + 1, seed0 + 2]
    # --- candidate ---
    try:
        cmod = K.load_module(cand_p, "sft_candidate")
        ModelNew = cmod.ModelNew
        m = K.build(functools.partial(ModelNew, **init_kwargs), init_args, seed0)
        res["weights"] = _copy_weights(ref, m)
    except Exception as e:  # noqa: BLE001
        res.update(verdict="CRASH", stage="import/instantiate", reason=exc_msg(e), tb=traceback.format_exc()[-800:])
        return res
    res["t_setup_s"] = round(time.time() - t0, 2)

    def golden(inp, scale=None):
        with torch.no_grad():
            g = gold_model(*_float_inputs(inp, torch.float32, scale))
        torch.cuda.synchronize()
        return g

    launches, fallback = [], {}

    def run(label, inp, tol, g, finite_only=False):
        try:
            out, nl, hits = _cand_forward(m, inp)
        except Exception as e:  # noqa: BLE001
            msg = exc_msg(e)
            return {"label": label, "ok": False, "crash": True, "fatal": _fatal(msg), "reason": "EXC " + msg}
        launches.append(nl)
        for k, v in hits.items():
            fallback[k] = fallback.get(k, 0) + v
        c = compare(out, g, tol, finite_only=finite_only)
        return {"label": label, "launches": nl, **c}

    # --- shape 1, 3 seeds ---
    inp1 = None
    first_fail = None
    for i, s in enumerate(res["seeds"]):
        try:
            inp = K.gen_inputs(get_inputs, s, "gpu")
        except Exception as e:  # noqa: BLE001
            res.update(verdict="REF_ERROR", reason="get_inputs: " + exc_msg(e))
            return res
        if inp1 is None:
            inp1 = inp
        try:
            g = golden(inp)
        except Exception as e:  # noqa: BLE001
            res.update(verdict="REF_ERROR", reason="reference forward: " + exc_msg(e))
            return res
        r = run(f"shape1/seed{i}", inp, TOL.get(_low_dtype(inp), 1e-4), g)
        r["dtype"] = str(_low_dtype(inp)).replace("torch.", "")
        res["cases"].append(r)
        del g
        if r.get("crash"):
            res.update(verdict="CRASH", stage=r["label"], reason=r["reason"])
            return res
        if not r["ok"]:
            first_fail = r
            break   # numerics failed on the dataset shape: no need for more cases
    res["shape1"] = {"shapes": [list(v.shape) if torch.is_tensor(v) else v for v in inp1][:6]}

    # --- shape 2 ---
    s2_fail = None
    if first_fail is None:
        try:
            sh = Shape2(ref_mod, inp1, model_src)
        except Exception as e:  # noqa: BLE001
            sh = None
            res["second_shape"] = "skipped"
            res["second_shape_reason"] = "probe: " + exc_msg(e, 150, 100)
        if sh is not None and sh.kind is None:
            res["second_shape"] = "skipped"
            res["second_shape_reason"] = sh.reason
        elif sh is not None:
            res["second_shape"] = sh.desc
            for i, s in enumerate(res["seeds"]):
                try:
                    inp = sh.inputs(s)
                    g = golden(inp)
                except Exception as e:  # noqa: BLE001
                    res["second_shape"] = "skipped"
                    res["second_shape_reason"] = "reference fails at new shape: " + exc_msg(e, 150, 100)
                    break
                r = run(f"shape2/seed{i}", inp, TOL.get(_low_dtype(inp), 1e-4), g)
                res["cases"].append(r)
                del inp, g
                if r.get("crash"):
                    if r.get("fatal"):
                        res.update(verdict="CRASH", stage=r["label"], reason=r["reason"])
                        return res
                    s2_fail = r
                    break
                if not r["ok"]:
                    s2_fail = r
                    break
            torch.cuda.empty_cache()

    # --- extreme magnitudes (reductions) ---
    ext_fail = None
    has_red = bool(REDUCTION_RE.search(model_src))
    res["reduction"] = has_red
    if first_fail is None and s2_fail is None and has_red:
        for sc in (1e3, 1e-3):
            scaled = _float_inputs(inp1, None, sc)
            try:
                g = golden(scaled)
            except Exception as e:  # noqa: BLE001
                res.setdefault("extreme_skipped", []).append(f"x{sc:g}: {exc_msg(e, 100, 50)}")
                continue
            r = run(f"extreme/x{sc:g}", scaled, None, g, finite_only=True)
            del g
            res["cases"].append(r)
            if r.get("crash"):
                if r.get("fatal"):
                    res.update(verdict="CRASH", stage=r["label"], reason=r["reason"])
                    return res
                ext_fail = r
                break
            if not r["ok"]:
                ext_fail = r
                break

    res["launches_per_call"] = launches
    res["launches_min"] = min(launches) if launches else 0
    res["launch_seen"] = bool(launches) and max(launches) > 0
    res["fallback_ops"] = fallback
    res["loose_ok"] = all(c.get("loose_ok", c.get("ok")) for c in res["cases"] if c["label"].startswith("shape"))
    # verdict precedence: launch/fallback problems first (a correct PyTorch fallback is not a kernel)
    if not res["launch_seen"] or res["launches_min"] == 0:
        res.update(verdict="NO_LAUNCH", reason="a candidate forward made no @triton.jit launch "
                   f"(launches per call {launches})")
    elif fallback:
        res.update(verdict="FALLBACK", reason=f"torch compute ops inside the candidate forward: {fallback}")
    elif first_fail is not None:
        res.update(verdict="FAIL", stage=first_fail["label"], reason=first_fail["reason"])
    elif s2_fail is not None:
        res.update(verdict="FAIL_SHAPE2", stage=s2_fail["label"], reason=s2_fail["reason"])
    elif ext_fail is not None:
        res.update(verdict="FAIL_EXTREME", stage=ext_fail["label"], reason=ext_fail["reason"])
    else:
        res.update(verdict="PASS", reason="")
    return res


# ---------------------------------------------------------------------------------------------
# autokernel (bench v2 checker, unchanged)
# ---------------------------------------------------------------------------------------------
def eval_autokernel(job, d):
    os.environ.setdefault("AUTOKERNEL_DIR", "/autokernel")
    import bench_v2_core as b2
    kt = job["kernel_type"]
    res = {"format": "autokernel", "kernel_type": kt, "harness": b2.HARNESS_VERSION}
    seed = int(job.get("seed") or int.from_bytes(os.urandom(4), "little"))
    res["seed"] = seed
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        kfn = b2.load_kernel(job["target"], "cand", d)
    except Exception as e:  # noqa: BLE001
        res.update(verdict="CRASH", stage="import", reason=exc_msg(e))
        return res
    launches, fallback = [], {}

    def wrapped(**kw):
        rec = OpRecorder()
        LAUNCHES[0] = 0
        with rec:
            out = kfn(**kw)
        launches.append(LAUNCHES[0])
        for k, v in rec.hits.items():
            fallback[k] = fallback.get(k, 0) + v
        return out

    c = b2.correctness(wrapped, kt, seed, early_abort=True)
    cases = c.pop("cases")
    res["bench_v2"] = c
    res["cases"] = [{k: r.get(k) for k in ("stage", "label", "dtype", "case", "ok", "max_abs_err", "reason")}
                    for r in cases if r.get("ok") is not True][:8]
    res["n_cases"], res["n_run"] = c["n_cases"], c["n_run"]
    res["launches_per_call"] = launches[:64]
    res["launches_min"] = min(launches) if launches else 0
    res["launch_seen"] = bool(launches) and max(launches) > 0
    res["fallback_ops"] = fallback
    if c["fatal"]:
        res.update(verdict="CRASH", reason="fatal CUDA error: " + next((r.get("reason", "") for r in cases if r.get("fatal")), ""))
    elif cases and cases[0]["ok"] is False and str(cases[0].get("reason", "")).startswith("EXC"):
        res.update(verdict="CRASH", reason=cases[0]["reason"])
    elif not res["launch_seen"] or res["launches_min"] == 0:
        res.update(verdict="NO_LAUNCH", reason=f"kernel_fn made no @triton.jit launch (launches per call {launches[:8]})")
    elif fallback:
        res.update(verdict="FALLBACK", reason=f"torch compute ops inside kernel_fn: {fallback}")
    elif c["ok"]:
        res.update(verdict="PASS", reason="")
    else:
        fails = [f"{r['stage']}/{r['label']}/{r['dtype']}: {r.get('reason', '')}" for r in cases if r["ok"] is False]
        if c["determinism"] is False:
            fails.append("determinism: two runs on identical inputs differ")
        res.update(verdict="FAIL", reason="; ".join(fails[:3])[:600])
    return res


# ---------------------------------------------------------------------------------------------
# child / zygote
# ---------------------------------------------------------------------------------------------
def child(job, res_path, log_path):
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    d = tempfile.mkdtemp()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        if job["format"] == "autokernel":
            r = eval_autokernel(job, d)
        else:
            r = eval_modelnew(job, d)
    except Exception as e:  # noqa: BLE001
        msg = exc_msg(e)
        r = {"verdict": "CRASH", "reason": "harness: " + msg, "tb": traceback.format_exc()[-1200:]}
    try:
        r["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    with open(res_path + ".tmp", "w") as f:
        json.dump(r, f, default=str)
    os.replace(res_path + ".tmp", res_path)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def run_one(job, d, timeout):
    res_path = os.path.join(d, f"res_{os.getpid()}_{time.time_ns()}.json")
    log_path = res_path[:-5] + ".log"
    t0 = time.time()
    pid = os.fork()
    if pid == 0:
        try:
            child(job, res_path, log_path)
        finally:
            os._exit(1)
    status, killed = None, False
    while True:
        wp, st = os.waitpid(pid, os.WNOHANG)
        if wp == pid:
            status = st
            break
        if time.time() - t0 > timeout:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            killed = True
            break
        time.sleep(0.02)
    wall = round(time.time() - t0, 2)
    tail = ""
    try:
        tail = open(log_path).read()[-600:]
    except Exception:  # noqa: BLE001
        pass
    if killed:
        r = {"verdict": "TIMEOUT", "reason": f"candidate exceeded {timeout:g}s (killed)", "log_tail": tail}
    elif os.path.exists(res_path):
        r = json.load(open(res_path))
    else:
        how = (f"signal {os.WTERMSIG(status)}" if os.WIFSIGNALED(status)
               else f"exit {os.WEXITSTATUS(status)}") if status is not None else "?"
        r = {"verdict": "CRASH", "reason": f"process died ({how})", "log_tail": tail}
    for p in (res_path, log_path):
        try:
            os.remove(p)
        except Exception:  # noqa: BLE001
            pass
    r["wall_s"] = wall
    r["harness"] = HARNESS_VERSION
    r["hook_ok"] = HOOK_OK
    return r


def batch(jobs_path, out_path, prog_path):
    jobs = json.load(open(jobs_path))
    d = tempfile.mkdtemp()
    for job in jobs:
        with open(prog_path, "a") as f:
            f.write(f"START {job['id']}\n")
        tmo = float(job.get("timeout") or 60)
        r = run_one(job, d, tmo)
        r["id"] = job["id"]
        with open(out_path, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        with open(prog_path, "a") as f:
            f.write(f"DONE {job['id']}\n")


def allowlist(out_path):
    import triton.language as tl_
    top = sorted(n for n in dir(tl_) if not n.startswith("_"))
    math_ = sorted(n for n in dir(tl_.math) if not n.startswith("_"))
    try:
        from triton.language.extra.cuda import libdevice
        libd = sorted(n for n in dir(libdevice) if not n.startswith("_"))
    except Exception:  # noqa: BLE001
        libd = []
    try:
        from triton.language.extra import libdevice as libd2
        libd_generic = sorted(n for n in dir(libd2) if not n.startswith("_"))
    except Exception:  # noqa: BLE001
        libd_generic = []
    out = {"triton": triton.__version__, "torch": torch.__version__, "tl": top, "tl.math": math_,
           "tl.extra.cuda.libdevice": libd, "tl.extra.libdevice": libd_generic,
           "note": "dir() of each namespace inside the autokernel bench image (names starting with _ dropped)"}
    json.dump(out, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "batch":
        batch(*sys.argv[2:5])
    elif sys.argv[1] == "allowlist":
        allowlist(sys.argv[2])
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]}")
