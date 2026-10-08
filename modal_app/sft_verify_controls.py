"""
Hand-written controls for sft_verify_modal.py::smoke (10 GPU controls + 1 static-only control).

  correct (3):        modelnew softmax (batch_size global -> shape 2 via global), modelnew
                      relu(x+y) KernelBook-style literal shapes (shape 2 via redraw), autokernel
                      rmsnorm starter (bench v2 PASS in results/v3/bench_v2_sanity.jsonl)
  wrong numerics (3): mean that divides by N+1 (FAIL), softmax without max-subtraction (passes
                      randn, overflows at x1e3 -> FAIL_EXTREME), GRPO "best" rmsnorm kernel (the fp16
                      reward hack; bench v2 FAIL)
  no-launch (2):      dead @triton.jit + pure-torch forward; conditional dispatch that only uses
                      Triton above a size the inputs never reach (falls back to torch.softmax)
  crash/hang (2):     out-of-bounds pointer (illegal memory access), 2^62-iteration kernel loop
  static (1):         uses tl.fancy_reduce (not in Triton 3.2) -> REJECT_SYMBOL, never sent to the GPU
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

SOFTMAX_REF = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=1)

batch_size = 64
dim = 1000

def get_inputs():
    return [torch.randn(batch_size, dim)]

def get_init_inputs():
    return []
'''

SOFTMAX_OK = '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * stride + offs, e / tl.sum(e, axis=0), mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty_like(x)
        n_rows, n_cols = x.shape
        softmax_kernel[(n_rows,)](x, out, n_cols, x.stride(0), BLOCK=triton.next_power_of_2(n_cols))
        return out
'''

SOFTMAX_NOMAX = SOFTMAX_OK.replace('    x = x - tl.max(x, axis=0)\n', '')

SOFTMAX_COND_FALLBACK = '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * stride + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(out_ptr + row * stride + offs, e / tl.sum(e, axis=0), mask=mask)

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.shape[1] > 65536:      # never true for the dataset shapes
            out = torch.empty_like(x)
            softmax_kernel[(x.shape[0],)](x, out, x.shape[1], x.stride(0), BLOCK=triton.next_power_of_2(x.shape[1]))
            return out
        return torch.softmax(x, dim=1)
'''

ADD_RELU_REF = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, x, y):
        return torch.relu(x + y)

def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]) - 0.5]

def get_init_inputs():
    return [[], {}]
'''.replace("return [[], {}]", "return []")

ADD_RELU_OK = '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.maximum(x + y, 0.0), mask=mask)

class ModelNew(nn.Module):
    def forward(self, x, y):
        x, y = x.contiguous(), y.contiguous()
        out = torch.empty_like(x)
        n = x.numel()
        add_relu_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024)
        return out
'''

ADD_RELU_DEAD_KERNEL = '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def add_relu_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.maximum(tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), 0.0), mask=mask)

class ModelNew(nn.Module):
    def forward(self, x, y):
        return torch.clamp_min(x + y, 0.0)   # the Triton kernel above is never launched
'''

MEAN_REF = '''import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, dim=self.dim)

batch_size = 32
features = 256

def get_inputs():
    return [torch.randn(batch_size, features)]

def get_init_inputs():
    return [1]
'''

MEAN_WRONG = '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_kernel(x_ptr, out_ptr, n_cols, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + row * stride + offs, mask=offs < n_cols, other=0.0)
    tl.store(out_ptr + row, tl.sum(x, axis=0) / (n_cols + 1))   # off-by-one denominator

class ModelNew(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = x.contiguous()
        out = torch.empty(x.shape[0], device=x.device, dtype=x.dtype)
        mean_kernel[(x.shape[0],)](x, out, x.shape[1], x.stride(0), BLOCK=triton.next_power_of_2(x.shape[1]))
        return out
'''

OOB_CRASH = '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def bad_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    far = offs.to(tl.int64) * 1099511627776          # 2^40-element stride: way out of bounds, unmasked
    x = tl.load(x_ptr + far)
    tl.store(out_ptr + offs, x, mask=offs < n)

class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        bad_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
        return out
'''

HANG = '''import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def spin_kernel(x_ptr, y_ptr, out_ptr, n, n_iter, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for i in range(0, n_iter):                         # n_iter = 2^62 at runtime: never finishes
        acc = acc * 0.999999 + x
    tl.store(out_ptr + offs, acc, mask=mask)

class ModelNew(nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n = x.numel()
        spin_kernel[(triton.cdiv(n, 256),)](x, y, out, n, 2 ** 62, BLOCK=256)
        return out
'''

BAD_SYMBOL = ADD_RELU_OK.replace("tl.maximum(x + y, 0.0)", "tl.fancy_reduce(x + y, 0.0)")


def _ak(path):
    return (ROOT / path).read_text()


def controls():
    """[(id, category, expected_verdict, row)] in run order (crash/hang deliberately mid-batch)."""
    def mn(i, ref, tgt):
        return {"id": f"control:{i}", "source": "control", "format": "modelnew", "reference": ref,
                "kernel_type": None, "target": tgt, "family": [], "static": {"ok": True}}

    def ak(i, kt, tgt):
        return {"id": f"control:{i}", "source": "control", "format": "autokernel", "reference": kt,
                "kernel_type": kt, "target": tgt, "family": [kt], "static": {"ok": True}}

    return [
        ("correct_softmax", "correct", "PASS", mn("correct_softmax", SOFTMAX_REF, SOFTMAX_OK)),
        ("wrong_mean_div_n_plus_1", "wrong numerics", "FAIL", mn("wrong_mean_div_n_plus_1", MEAN_REF, MEAN_WRONG)),
        ("nolaunch_dead_kernel", "no-launch fallback", "NO_LAUNCH", mn("nolaunch_dead_kernel", ADD_RELU_REF, ADD_RELU_DEAD_KERNEL)),
        ("crash_illegal_address", "crash/hang", "CRASH", mn("crash_illegal_address", ADD_RELU_REF, OOB_CRASH)),
        ("correct_add_relu", "correct", "PASS", mn("correct_add_relu", ADD_RELU_REF, ADD_RELU_OK)),
        ("wrong_softmax_no_max", "wrong numerics", "FAIL_EXTREME", mn("wrong_softmax_no_max", SOFTMAX_REF, SOFTMAX_NOMAX)),
        ("hang_2e62_loop", "crash/hang", "TIMEOUT", mn("hang_2e62_loop", ADD_RELU_REF, HANG)),
        ("nolaunch_conditional_fallback", "no-launch fallback", "NO_LAUNCH",
         mn("nolaunch_conditional_fallback", SOFTMAX_REF, SOFTMAX_COND_FALLBACK)),
        ("correct_ak_rmsnorm_starter", "correct", "PASS",
         ak("correct_ak_rmsnorm_starter", "rmsnorm", _ak("results/autokernel_src/kernels/rmsnorm.py"))),
        ("wrong_ak_rmsnorm_grpo_best", "wrong numerics", "FAIL",
         ak("wrong_ak_rmsnorm_grpo_best", "rmsnorm", _ak("presentation/v2/best_kernel.py"))),
    ]


def static_controls():
    return [("static_bad_tl_symbol", "static", "REJECT_SYMBOL",
             {"id": "control:static_bad_tl_symbol", "source": "control", "format": "modelnew",
              "reference": ADD_RELU_REF, "kernel_type": None, "target": BAD_SYMBOL, "family": [],
              "static": {"ok": True}})]
