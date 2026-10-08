"""
Fixed validation set v1 (Phase 0 / sft_plan Phase 1 step 5): held-out AutoKernel-family problems
with shapes and variants the training prompts never used. NOT KernelBench.

    python modal_app/val_set_v1.py      # -> results/v3/val_set/<family>__<variant>/{reference.py,
                                        #    starter.py, problem.json} + results/v3/val_set.json

Each problem:
  reference.py  `reference(**inputs)`: PyTorch ground truth, shown to the model. The harness
                (val_bench_core.py) evaluates it on inputs upcast to fp64 (fp32, TF32 off, for the
                matmul-type families; fp64 for their fp32 inputs) = the golden.
  starter.py    a Triton `kernel_fn(**inputs)` that PASSes (verified: results/v3/val_set_starters.jsonl)
  problem.json  family, variant, input generator, correctness shapes x dtypes, adversarial cases,
                timing shape, per-dtype tolerances (upstream bench.py values for the family),
                tags for decontaminating SFT data.

Tolerances: the family's upstream per-dtype atol/rtol (bench.py @78435821), rtol floored at 2u by
the harness; rmsnorm/reduce have no upstream fp32 tolerance, so fp32 uses 1e-5 (softmax's).
"""
import json
import pathlib
import re
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "v3" / "val_set"
META = ROOT / "results" / "v3" / "val_set.json"

H, B, F = "float16", "bfloat16", "float32"
ALL = [H, B, F]
TOL = {
    "matmul": {H: (1e-2, 1e-2), B: (2e-2, 2e-2), F: (1e-4, 1e-4)},
    "softmax": {H: (1e-3, 1e-3), B: (2e-3, 2e-3), F: (1e-5, 1e-5)},
    "layernorm": {H: (1e-3, 1e-3), B: (2e-3, 2e-3), F: (1e-5, 1e-5)},
    "rmsnorm": {H: (1e-2, 1e-2), B: (1e-1, 5e-2), F: (1e-5, 1e-5)},
    "flash_attention": {H: (1e-2, 1e-2), B: (2e-2, 2e-2), F: (1e-4, 1e-4)},
    "fused_mlp": {H: (1e-2, 1e-2), B: (2e-2, 2e-2), F: (1e-4, 1e-4)},
    "cross_entropy": {H: (1e-2, 1e-2), B: (2e-2, 2e-2), F: (1e-5, 1e-5)},
    "rotary_embedding": {H: (1e-3, 1e-3), B: (2e-3, 2e-3), F: (1e-5, 1e-5)},
    "reduce": {H: (1e-2, 1e-2), B: (1e-1, 5e-2), F: (1e-5, 1e-5)},
}
HEAD = "import torch\nimport triton\nimport triton.language as tl\n"
REFHEAD = "import torch\nimport torch.nn.functional as F\n\n\n"


def d(s):
    return textwrap.dedent(s).strip("\n") + "\n"


def starter_doc(family, variant, note):
    return f'"""\nValidation starter: {family} / {variant}.\n{note}\n"""\n\nKERNEL_TYPE = "{family}"\n\n' + HEAD


PROBLEMS = []


def add(family, variant, desc, ref_body, starter, gen, sizes, timing, adversarial=(), gen_args=None,
        tags=()):
    PROBLEMS.append({"family": family, "variant": variant, "desc": desc, "ref": REFHEAD + d(ref_body),
                     "starter": starter, "gen": gen, "sizes": sizes, "timing": timing,
                     "adversarial": list(adversarial), "gen_args": gen_args or {}, "tags": list(tags)})


# =============================================================================================
# softmax
# =============================================================================================
def softmax_row_kernel(variant, note, pre="", log=False):
    body = "result = row - tl.log(denominator)" if log else "result = numerator / denominator"
    return starter_doc("softmax", variant, note) + d(f'''

        @triton.jit
        def softmax_kernel(
            input_ptr, output_ptr, n_cols, stride_input_row, stride_output_row,
            BLOCK_SIZE: tl.constexpr,
        ):
            """One program per row; the whole row is one block (BLOCK_SIZE >= n_cols)."""
            row_idx = tl.program_id(0)
            col_offsets = tl.arange(0, BLOCK_SIZE)
            mask = col_offsets < n_cols
            row = tl.load(input_ptr + row_idx * stride_input_row + col_offsets, mask=mask,
                          other=float("-inf")).to(tl.float32)
            {pre or "# (no pre-scaling)"}
            row_max = tl.max(row, axis=0)
            row = row - row_max
            numerator = tl.exp(row)
            denominator = tl.sum(numerator, axis=0)
            {body}
            tl.store(output_ptr + row_idx * stride_output_row + col_offsets, result, mask=mask)
        ''') + "\n\n"


def softmax_row(variant, note, **kw):
    dim0 = kw.pop("dim0", False)
    head = softmax_row_kernel(variant, note, **kw)
    fn = ("def kernel_fn(x: torch.Tensor) -> torch.Tensor:\n"
          + ("    xt = x.t().contiguous()  # softmax over dim 0 = row softmax of x^T\n" if dim0 else
             "    xt = x.reshape(-1, x.shape[-1])\n")
          + "    n_rows, n_cols = xt.shape\n"
            "    out = torch.empty_like(xt)\n"
            "    BLOCK_SIZE = triton.next_power_of_2(n_cols)\n"
            "    softmax_kernel[(n_rows,)](xt, out, n_cols, xt.stride(0), out.stride(0), BLOCK_SIZE=BLOCK_SIZE)\n"
          + ("    return out.t()\n" if dim0 else "    return out.view(x.shape)\n"))
    return head + fn


SOFTMAX_ONLINE = starter_doc("softmax", "online_large_N",
                             "Rows are too long for one block: single-pass online max/sum over\n"
                             "BLOCK_SIZE-wide lanes, then a second pass writes exp(x - m) / s.") + d('''

    @triton.jit
    def softmax_online_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o, BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        x_row = x_ptr + row * stride_x
        o_row = out_ptr + row * stride_o
        offs = tl.arange(0, BLOCK_SIZE)
        m_vec = tl.full((BLOCK_SIZE,), -1e30, dtype=tl.float32)   # per-lane running max
        s_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)          # per-lane running sum
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            x = tl.load(x_row + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
            m_new = tl.maximum(m_vec, x)
            s_vec = s_vec * tl.exp(m_vec - m_new) + tl.exp(x - m_new)
            m_vec = m_new
        m = tl.max(m_vec, axis=0)
        s = tl.sum(s_vec * tl.exp(m_vec - m), axis=0)
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            mask = cols < n_cols
            x = tl.load(x_row + cols, mask=mask, other=float("-inf")).to(tl.float32)
            tl.store(o_row + cols, tl.exp(x - m) / s, mask=mask)


    def kernel_fn(x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)
        softmax_online_kernel[(n_rows,)](x2, out, n_cols, x2.stride(0), out.stride(0), BLOCK_SIZE=4096)
        return out.view(x.shape)
''')

SM_REF = '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Softmax over the last dim."""
        return F.softmax(x, dim=-1)
'''
add("softmax", "nonpow2_cols", "row softmax, non-power-of-2 column counts",
    SM_REF, softmax_row("nonpow2_cols", "Row softmax; column counts are not powers of 2."), "softmax",
    [["r1537_c3000", {"shape": [1537, 3000]}, ALL], ["r333_c777", {"shape": [333, 777]}, ALL],
     ["r1_c50000", {"shape": [1, 50000]}, [H]]],
    [["r4096_c3000", {"shape": [4096, 3000]}, H]],
    [["big_logits", "big_logits_x100", {"shape": [1024, 3000]}, [H, B]],
     ["constant", "constant", {"shape": [256, 999]}, [H]]], tags=["nonpow2"])
add("softmax", "bf16_wide", "row softmax, bf16 only",
    SM_REF, softmax_row("bf16_wide", "Row softmax, bfloat16 inputs."), "softmax",
    [["r2048_c8192", {"shape": [2048, 8192]}, [B]], ["r100_c1000", {"shape": [100, 1000]}, [B]]],
    [["r8192_c8192", {"shape": [8192, 8192]}, B]],
    [["big_logits", "big_logits_x100", {"shape": [1024, 8192]}, [B]]], tags=["bf16"])
add("softmax", "online_large_N", "row softmax with rows of 300k-1M elements (needs a looped/online kernel)",
    SM_REF, SOFTMAX_ONLINE, "softmax",
    [["r8_c1048576", {"shape": [8, 1048576]}, [H, F]], ["r13_c300007", {"shape": [13, 300007]}, [H, B]]],
    [["r32_c262144", {"shape": [32, 262144]}, H]],
    [["big_logits", "big_logits_x100", {"shape": [4, 500000]}, [H]]], tags=["large_N", "online"])
add("softmax", "dim0", "softmax over dim 0 (columns)",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Softmax over dim 0 (each column sums to 1)."""
        return F.softmax(x, dim=0)
    ''', softmax_row("dim0", "Softmax over dim 0, done as a row softmax of the transposed copy.", dim0=True),
    "softmax",
    [["r4096_c1024", {"shape": [4096, 1024]}, ALL], ["r1000_c777", {"shape": [1000, 777]}, ALL]],
    [["r8192_c2048", {"shape": [8192, 2048]}, H]],
    [["big_logits", "big_logits_x100", {"shape": [2048, 512]}, [H]]], tags=["dim0"])
add("softmax", "scaled", "softmax(x * 1/sqrt(128))",
    '''
    SCALE = 0.08838834764831845  # 1 / sqrt(128)


    def reference(x: torch.Tensor) -> torch.Tensor:
        """Scaled softmax over the last dim: softmax(x * SCALE)."""
        return F.softmax(x * SCALE, dim=-1)
    ''', softmax_row("scaled", "Scaled row softmax, softmax(x * 1/sqrt(128)).",
                     pre="row = row * 0.08838834764831845"), "softmax",
    [["r4096_c2048", {"shape": [4096, 2048]}, ALL], ["r257_c1000", {"shape": [257, 1000]}, ALL]],
    [["r4096_c4096", {"shape": [4096, 4096]}, H]],
    [["big_logits", "big_logits_x100", {"shape": [1024, 2048]}, [H, B]]], tags=["fused_scale"])
add("softmax", "log_softmax", "log_softmax over the last dim",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Log-softmax over the last dim."""
        return F.log_softmax(x, dim=-1)
    ''', softmax_row("log_softmax", "Row log-softmax: (x - max) - log(sum(exp(x - max))).", log=True),
    "softmax",
    [["r2048_c4096", {"shape": [2048, 4096]}, ALL], ["r100_c30000", {"shape": [100, 30000]}, ALL]],
    [["r4096_c8192", {"shape": [4096, 8192]}, H]],
    [["big_logits", "big_logits_x100", {"shape": [512, 4096]}, [H, B]]], tags=["log_softmax"])
add("softmax", "4d_scores", "softmax over the last dim of a 4-D attention-score tensor (197x197, ViT)",
    SM_REF, softmax_row("4d_scores", "Row softmax over the last dim of a [B, H, Sq, Sk] tensor."), "softmax",
    [["b2_h12_197", {"shape": [2, 12, 197, 197]}, ALL], ["b1_h4_1x77", {"shape": [1, 4, 1, 77]}, [H]]],
    [["b8_h16_577", {"shape": [8, 16, 577, 577]}, H]],
    [["big_logits", "big_logits_x100", {"shape": [1, 12, 197, 197]}, [H]]], tags=["4d", "nonpow2"])


# =============================================================================================
# layernorm
# =============================================================================================
def ln_row(variant, note, weight=True, bias=True, residual=False, eps="1e-5"):
    args = ["x: torch.Tensor"] + (["residual: torch.Tensor"] if residual else []) + \
           (["weight: torch.Tensor"] if weight else []) + (["bias: torch.Tensor"] if bias else [])
    load_r = ("    x += tl.load(R_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)\n"
              if residual else "")
    wb = ""
    if weight:
        wb += "    y = y * tl.load(W_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)\n"
    if bias:
        wb += "    y = y + tl.load(B_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)\n"
    call = ["x2", "r2" if residual else "x2", "y", "weight" if weight else "x2", "bias" if bias else "x2"]
    return starter_doc("layernorm", variant, note) + (
        "\n\n@triton.jit\n"
        "def layernorm_kernel(X_ptr, R_ptr, Y_ptr, W_ptr, B_ptr, stride_x_row, stride_y_row, N, eps,\n"
        "                     BLOCK_SIZE: tl.constexpr):\n"
        '    """One program per row, two-pass mean/variance in fp32 on one block."""\n'
        "    row_idx = tl.program_id(0)\n"
        "    col_offsets = tl.arange(0, BLOCK_SIZE)\n"
        "    mask = col_offsets < N\n"
        "    x = tl.load(X_ptr + row_idx * stride_x_row + col_offsets, mask=mask, other=0.0).to(tl.float32)\n"
        + load_r +
        "    mean = tl.sum(x, axis=0) / N\n"
        "    xc = tl.where(mask, x - mean, 0.0)\n"
        "    var = tl.sum(xc * xc, axis=0) / N\n"
        "    y = xc / tl.sqrt(var + eps)\n"
        + wb +
        "    tl.store(Y_ptr + row_idx * stride_y_row + col_offsets, y, mask=mask)\n\n\n"
        f"def kernel_fn({', '.join(args)}) -> torch.Tensor:\n"
        "    N = x.shape[-1]\n"
        "    x2 = x.reshape(-1, N)\n"
        + ("    r2 = residual.reshape(-1, N)\n" if residual else "") +
        "    y = torch.empty_like(x2)\n"
        "    BLOCK_SIZE = triton.next_power_of_2(N)\n"
        f"    layernorm_kernel[(x2.shape[0],)]({', '.join(call)}, x2.stride(0), y.stride(0), N, {eps},\n"
        "                                     BLOCK_SIZE=BLOCK_SIZE)\n"
        "    return y.view(x.shape)\n")


LN_LOOPED = starter_doc("layernorm", "large_D", "Rows of 100k-300k elements: three looped passes\n"
                        "(mean, variance of x - mean, normalise) over BLOCK_SIZE chunks, fp32.") + d('''

    @triton.jit
    def layernorm_looped_kernel(X_ptr, Y_ptr, W_ptr, B_ptr, stride_x_row, stride_y_row, N, eps,
                                BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        x_row = X_ptr + row * stride_x_row
        y_row = Y_ptr + row * stride_y_row
        offs = tl.arange(0, BLOCK_SIZE)
        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in range(0, N, BLOCK_SIZE):
            cols = start + offs
            acc += tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        mean = tl.sum(acc, axis=0) / N
        acc2 = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in range(0, N, BLOCK_SIZE):
            cols = start + offs
            mask = cols < N
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            xc = tl.where(mask, x - mean, 0.0)
            acc2 += xc * xc
        rstd = 1.0 / tl.sqrt(tl.sum(acc2, axis=0) / N + eps)
        for start in range(0, N, BLOCK_SIZE):
            cols = start + offs
            mask = cols < N
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(y_row + cols, (x - mean) * rstd * w + b, mask=mask)


    def kernel_fn(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        N = x.shape[-1]
        x2 = x.reshape(-1, N)
        y = torch.empty_like(x2)
        layernorm_looped_kernel[(x2.shape[0],)](x2, y, weight, bias, x2.stride(0), y.stride(0), N, 1e-5,
                                                BLOCK_SIZE=4096)
        return y.view(x.shape)
''')

LN_ADV = lambda shp, dts=(H, B): [["big_scale", "big_scale_x300", {"shape": shp}, list(dts)],
                                  ["large_mean", "mean_1000", {"shape": shp}, [H]],
                                  ["constant", "constant", {"shape": [256, shp[-1]]}, [H]]]
add("layernorm", "no_bias", "layer_norm with weight but no bias, D=768/1536/12288",
    '''
    def reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """LayerNorm over the last dim with a weight and NO bias (eps=1e-5)."""
        return F.layer_norm(x, x.shape[-1:], weight, None, 1e-5)
    ''', ln_row("no_bias", "LayerNorm, weight only (no bias).", bias=False), "layernorm",
    [["r1000_d1536", {"shape": [1000, 1536]}, ALL], ["r4096_d768", {"shape": [4096, 768]}, [H, B]],
     ["r7_d12288", {"shape": [7, 12288]}, [H]]],
    [["r8192_d768", {"shape": [8192, 768]}, H]], LN_ADV([1024, 1536]),
    gen_args={"weight": True, "bias": False}, tags=["no_bias", "nonpow2"])
add("layernorm", "no_affine", "layer_norm without weight or bias",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """LayerNorm over the last dim, no elementwise affine (eps=1e-5)."""
        return F.layer_norm(x, x.shape[-1:], None, None, 1e-5)
    ''', ln_row("no_affine", "LayerNorm without weight/bias.", weight=False, bias=False), "layernorm",
    [["r2048_d1600", {"shape": [2048, 1600]}, ALL], ["r37_d5000", {"shape": [37, 5000]}, ALL]],
    [["r8192_d1600", {"shape": [8192, 1600]}, H]], LN_ADV([1024, 1600]),
    gen_args={"weight": False, "bias": False}, tags=["no_affine", "nonpow2"])
LN_REF = '''
    def reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """LayerNorm over the last dim (eps=1e-5)."""
        return F.layer_norm(x, x.shape[-1:], weight, bias, 1e-5)
'''
add("layernorm", "bf16", "layer_norm, bf16 only, D=2560/3000",
    LN_REF, ln_row("bf16", "LayerNorm, bfloat16 inputs."), "layernorm",
    [["r8192_d2560", {"shape": [8192, 2560]}, [B]], ["r100_d3000", {"shape": [100, 3000]}, [B]]],
    [["r8192_d2560", {"shape": [8192, 2560]}, B]],
    [["big_scale", "big_scale_x300", {"shape": [1024, 2560]}, [B]]],
    gen_args={"weight": True, "bias": True}, tags=["bf16"])
add("layernorm", "3d_eps1e-6", "layer_norm on [B, S, D] with eps=1e-6",
    '''
    def reference(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """LayerNorm over the last dim of a [batch, seq, dim] tensor, eps=1e-6."""
        return F.layer_norm(x, x.shape[-1:], weight, bias, 1e-6)
    ''', ln_row("3d_eps1e-6", "LayerNorm on [B, S, D], eps=1e-6.", eps="1e-6"), "layernorm",
    [["b8_s512_d1024", {"shape": [8, 512, 1024]}, ALL], ["b3_s5_d640", {"shape": [3, 5, 640]}, ALL]],
    [["b16_s1024_d1024", {"shape": [16, 1024, 1024]}, H]], LN_ADV([4, 256, 1024]),
    gen_args={"weight": True, "bias": True}, tags=["3d", "eps"])
add("layernorm", "large_D", "layer_norm with D = 100k-300k (needs a looped kernel)",
    LN_REF, LN_LOOPED, "layernorm",
    [["r32_d100000", {"shape": [32, 100000]}, [H, F]], ["r8_d300000", {"shape": [8, 300000]}, [H, B]]],
    [["r64_d131072", {"shape": [64, 131072]}, H]],
    [["big_scale", "big_scale_x300", {"shape": [16, 100000]}, [H]]],
    gen_args={"weight": True, "bias": True}, tags=["large_N"])
add("layernorm", "residual_add", "layer_norm(x + residual) fused",
    '''
    def reference(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                  bias: torch.Tensor) -> torch.Tensor:
        """Fused residual add + LayerNorm: layer_norm(x + residual) over the last dim (eps=1e-5)."""
        return F.layer_norm(x + residual, x.shape[-1:], weight, bias, 1e-5)
    ''', ln_row("residual_add", "Fused residual add + LayerNorm.", residual=True), "layernorm",
    [["r4096_d4096", {"shape": [4096, 4096]}, [H, B]], ["r1000_d1000", {"shape": [1000, 1000]}, ALL]],
    [["r8192_d4096", {"shape": [8192, 4096]}, H]],
    [["big_scale", "big_scale_x300", {"shape": [1024, 2048]}, [H, B]]],
    gen_args={"weight": True, "bias": True, "residual": True}, tags=["residual", "fusion"])
add("layernorm", "fp32", "layer_norm, fp32 only (tol 1e-5)",
    LN_REF, ln_row("fp32", "LayerNorm, float32 inputs."), "layernorm",
    [["r2048_d4096", {"shape": [2048, 4096]}, [F]], ["r333_d1000", {"shape": [333, 1000]}, [F]]],
    [["r4096_d4096", {"shape": [4096, 4096]}, F]],
    [["big_scale", "big_scale_x300", {"shape": [512, 4096]}, [F]]],
    gen_args={"weight": True, "bias": True}, tags=["fp32"])


# =============================================================================================
# rmsnorm
# =============================================================================================
def rms_row(variant, note, gemma=False, residual=False, eps="1e-6"):
    args = ["x: torch.Tensor"] + (["residual: torch.Tensor"] if residual else []) + ["weight: torch.Tensor"]
    load_r = ("    x += tl.load(R_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)\n"
              if residual else "")
    wexpr = "(1.0 + w)" if gemma else "w"
    return starter_doc("rmsnorm", variant, note) + (
        "\n\n@triton.jit\n"
        "def rmsnorm_kernel(X_ptr, R_ptr, W_ptr, OUT_ptr, N, stride_row, stride_out, eps,\n"
        "                   BLOCK_SIZE: tl.constexpr):\n"
        '    """One program per row; fp32 sum of squares over one block."""\n'
        "    row = tl.program_id(0)\n"
        "    offs = tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offs < N\n"
        "    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)\n"
        + load_r +
        "    rms = tl.sqrt(tl.sum(x * x, axis=0) / N + eps)\n"
        "    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)\n"
        f"    tl.store(OUT_ptr + row * stride_out + offs, (x / rms) * {wexpr}, mask=mask)\n\n\n"
        f"def kernel_fn({', '.join(args)}) -> torch.Tensor:\n"
        "    N = x.shape[-1]\n"
        "    x2 = x.reshape(-1, N)\n"
        + ("    r2 = residual.reshape(-1, N)\n" if residual else "") +
        "    out = torch.empty_like(x2)\n"
        f"    rmsnorm_kernel[(x2.shape[0],)](x2, {'r2' if residual else 'x2'}, weight, out, N, x2.stride(0), out.stride(0),\n"
        f"                                   {eps}, BLOCK_SIZE=triton.next_power_of_2(N))\n"
        "    return out.view(x.shape)\n")


RMS_REF = '''
    def reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """RMS normalization over the last dim."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return (x / rms) * weight
'''
RMS_ADV = lambda shp, dts=(H, B): [["rms_300", "rms_300", {"shape": shp}, list(dts)],
                                   ["row_mixed", "row_mixed_scale", {"shape": shp}, [H]],
                                   ["tiny", "tiny_1e-3", {"shape": shp}, [H]]]
add("rmsnorm", "N16384", "rmsnorm with N=16384",
    RMS_REF, rms_row("N16384", "RMSNorm, N=16384 (one block per row)."), "rmsnorm",
    [["m512_n16384", {"shape": [512, 16384]}, [H, B]], ["m64_n16384", {"shape": [64, 16384]}, [H]]],
    [["m2048_n16384", {"shape": [2048, 16384]}, H]], RMS_ADV([256, 16384]), tags=["large_N"])
add("rmsnorm", "nonpow2", "rmsnorm with N=5120/3000/1000",
    RMS_REF, rms_row("nonpow2", "RMSNorm, non-power-of-2 N."), "rmsnorm",
    [["m2048_n5120", {"shape": [2048, 5120]}, [H, B]], ["m1000_n3000", {"shape": [1000, 3000]}, [H, B]],
     ["m3_n1000", {"shape": [3, 1000]}, [H]]],
    [["m4096_n5120", {"shape": [4096, 5120]}, H]], RMS_ADV([1024, 5120]), tags=["nonpow2"])
add("rmsnorm", "bf16", "rmsnorm, bf16 only",
    RMS_REF, rms_row("bf16", "RMSNorm, bfloat16 inputs."), "rmsnorm",
    [["m4096_n8192", {"shape": [4096, 8192]}, [B]], ["m123_n2048", {"shape": [123, 2048]}, [B]]],
    [["m8192_n4096", {"shape": [8192, 4096]}, B]],
    [["rms_300", "rms_300", {"shape": [1024, 4096]}, [B]], ["row_mixed", "row_mixed_scale", {"shape": [1024, 4096]}, [B]]],
    tags=["bf16"])
add("rmsnorm", "3d", "rmsnorm on [B, S, D]",
    RMS_REF, rms_row("3d", "RMSNorm on [batch, seq, dim]."), "rmsnorm",
    [["b4_s2048_d4096", {"shape": [4, 2048, 4096]}, [H, B]], ["b2_s3_d2048", {"shape": [2, 3, 2048]}, [H]]],
    [["b8_s2048_d4096", {"shape": [8, 2048, 4096]}, H]], RMS_ADV([2, 512, 4096]), tags=["3d"])
add("rmsnorm", "gemma_offset", "Gemma-style rmsnorm: (x / rms) * (1 + weight)",
    '''
    def reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Gemma-style RMSNorm: the learned scale is (1 + weight)."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        return (x / rms) * (1.0 + weight)
    ''', rms_row("gemma_offset", "RMSNorm with a (1 + weight) scale (Gemma).", gemma=True), "rmsnorm",
    [["m2048_n3072", {"shape": [2048, 3072]}, [H, B]], ["m100_n2304", {"shape": [100, 2304]}, [H, B]]],
    [["m4096_n3072", {"shape": [4096, 3072]}, H]], RMS_ADV([1024, 3072]), tags=["offset_weight", "nonpow2"])
add("rmsnorm", "residual_add", "rmsnorm(x + residual) * weight fused",
    '''
    def reference(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
        """Fused residual add + RMSNorm over the last dim."""
        h = x + residual
        rms = torch.sqrt(torch.mean(h ** 2, dim=-1, keepdim=True) + eps)
        return (h / rms) * weight
    ''', rms_row("residual_add", "Fused residual add + RMSNorm.", residual=True), "rmsnorm",
    [["m4096_n4096", {"shape": [4096, 4096]}, [H, B]], ["m1000_n1000", {"shape": [1000, 1000]}, [H, B]]],
    [["m8192_n4096", {"shape": [8192, 4096]}, H]], RMS_ADV([1024, 4096]),
    gen_args={"residual": True}, tags=["residual", "fusion"])
add("rmsnorm", "fp32", "rmsnorm, fp32 only (tol 1e-5)",
    RMS_REF, rms_row("fp32", "RMSNorm, float32 inputs."), "rmsnorm",
    [["m2048_n4096", {"shape": [2048, 4096]}, [F]], ["m333_n1000", {"shape": [333, 1000]}, [F]]],
    [["m4096_n4096", {"shape": [4096, 4096]}, F]],
    [["row_mixed", "row_mixed_scale", {"shape": [1024, 4096]}, [F]]], tags=["fp32"])


# =============================================================================================
# reduce
# =============================================================================================
def reduce_row(variant, note, op="sum"):
    init = {"sum": "tl.zeros((BLOCK_SIZE,), dtype=tl.float32)", "mean": "tl.zeros((BLOCK_SIZE,), dtype=tl.float32)",
            "sumsq": "tl.zeros((BLOCK_SIZE,), dtype=tl.float32)",
            "max": "tl.full((BLOCK_SIZE,), float(\"-inf\"), dtype=tl.float32)"}[op]
    other = 'float("-inf")' if op == "max" else "0.0"
    upd = {"sum": "acc += x", "mean": "acc += x", "sumsq": "acc += x * x", "max": "acc = tl.maximum(acc, x)"}[op]
    fin = {"sum": "tl.sum(acc, axis=0)", "mean": "tl.sum(acc, axis=0) / N", "sumsq": "tl.sum(acc, axis=0)",
           "max": "tl.max(acc, axis=0)"}[op]
    return starter_doc("reduce", variant, note) + (
        "\n\n@triton.jit\n"
        "def reduce_rows_kernel(X_ptr, OUT_ptr, N, stride_row, BLOCK_SIZE: tl.constexpr):\n"
        '    """One program per row; loops over the row in BLOCK_SIZE chunks, fp32 accumulator."""\n'
        "    row = tl.program_id(0).to(tl.int64)\n"
        "    offs = tl.arange(0, BLOCK_SIZE)\n"
        f"    acc = {init}\n"
        "    for start in range(0, N, BLOCK_SIZE):\n"
        "        cols = start + offs\n"
        f"        x = tl.load(X_ptr + row * stride_row + cols, mask=cols < N, other={other}).to(tl.float32)\n"
        f"        {upd}\n"
        f"    tl.store(OUT_ptr + row, {fin})\n\n\n"
        "def kernel_fn(x: torch.Tensor) -> torch.Tensor:\n"
        "    N = x.shape[-1]\n"
        "    x2 = x.reshape(-1, N)\n"
        "    out = torch.empty(x2.shape[0], device=x.device, dtype=torch.float32)\n"
        "    BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)\n"
        "    reduce_rows_kernel[(x2.shape[0],)](x2, out, N, x2.stride(0), BLOCK_SIZE=BLOCK_SIZE)\n"
        "    return out.to(x.dtype).view(x.shape[:-1])\n")


REDUCE_COLS = lambda variant, note, three_d: starter_doc("reduce", variant, note) + (
    "\n\n@triton.jit\n"
    "def colsum_kernel(X_ptr, OUT_ptr, M, N, stride_b, stride_m, stride_n, stride_ob,\n"
    "                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):\n"
    '    """Sum over the M axis of a [batch, M, N] view. Program (column block, batch)."""\n'
    "    pid_n = tl.program_id(0)\n"
    "    pid_b = tl.program_id(1).to(tl.int64)\n"
    "    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n"
    "    rows = tl.arange(0, BLOCK_M)\n"
    "    base = X_ptr + pid_b * stride_b\n"
    "    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
    "    for start in range(0, M, BLOCK_M):\n"
    "        r = start + rows\n"
    "        m = (r[:, None] < M) & (cols[None, :] < N)\n"
    "        acc += tl.load(base + r[:, None] * stride_m + cols[None, :] * stride_n, mask=m, other=0.0).to(tl.float32)\n"
    "    tl.store(OUT_ptr + pid_b * stride_ob + cols, tl.sum(acc, axis=0), mask=cols < N)\n\n\n"
    "def kernel_fn(x: torch.Tensor) -> torch.Tensor:\n"
    + ("    x3 = x.contiguous()\n    Bt, M, N = x3.shape\n" if three_d else
       "    x3 = x.contiguous().unsqueeze(0)\n    Bt, M, N = x3.shape\n") +
    "    out = torch.empty((Bt, N), device=x.device, dtype=torch.float32)\n"
    "    BLOCK_M, BLOCK_N = 32, 128\n"
    "    grid = (triton.cdiv(N, BLOCK_N), Bt)\n"
    "    colsum_kernel[grid](x3, out, M, N, x3.stride(0), x3.stride(1), x3.stride(2), out.stride(0),\n"
    "                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)\n"
    + ("    return out.to(x.dtype)\n" if three_d else "    return out[0].to(x.dtype)\n"))

RED_REF = '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Sum over the last dim."""
        return x.sum(dim=-1)
'''
POS = lambda shp, dts=(H, B): [["pos_uniform", "pos_uniform", {"shape": shp}, list(dts)]]
add("reduce", "sum_nonpow2", "row sum, non-power-of-2 widths",
    RED_REF, reduce_row("sum_nonpow2", "Row sum over the last dim."), "reduce",
    [["m3001_n1537", {"shape": [3001, 1537]}, [H, B]], ["m100_n100003", {"shape": [100, 100003]}, [H, B]]],
    [["m8192_n6000", {"shape": [8192, 6000]}, H]], POS([256, 30000]), tags=["nonpow2"])
add("reduce", "max", "row max (values)",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Max over the last dim (values only)."""
        return x.max(dim=-1).values
    ''', reduce_row("max", "Row max over the last dim.", op="max"), "reduce",
    [["m4096_n4096", {"shape": [4096, 4096]}, ALL], ["m1000_n777", {"shape": [1000, 777]}, ALL]],
    [["m8192_n8192", {"shape": [8192, 8192]}, H]], [], tags=["max"])
add("reduce", "mean", "row mean",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Mean over the last dim."""
        return x.mean(dim=-1)
    ''', reduce_row("mean", "Row mean over the last dim.", op="mean"), "reduce",
    [["m4096_n8192", {"shape": [4096, 8192]}, ALL], ["m999_n1000", {"shape": [999, 1000]}, ALL]],
    [["m8192_n8192", {"shape": [8192, 8192]}, H]], POS([256, 32768]), tags=["mean"])
add("reduce", "sum_dim0", "column sum (reduce over dim 0)",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Sum over dim 0 (column sums)."""
        return x.sum(dim=0)
    ''', REDUCE_COLS("sum_dim0", "Column sum: each program owns BLOCK_N columns and loops over rows.", False),
    "reduce",
    [["m8192_n1024", {"shape": [8192, 1024]}, [H, B]], ["m1000_n777", {"shape": [1000, 777]}, [H, B]]],
    [["m16384_n4096", {"shape": [16384, 4096]}, H]], POS([32768, 256]), tags=["dim0"])
add("reduce", "sum_dim1_3d", "sum over the middle dim of [B, N, D]",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Sum over dim 1 of a [batch, n, d] tensor -> [batch, d]."""
        return x.sum(dim=1)
    ''', REDUCE_COLS("sum_dim1_3d", "Sum over dim 1 of [B, N, D]: program (column block, batch).", True),
    "reduce",
    [["b16_n4096_d256", {"shape": [16, 4096, 256]}, [H, B]], ["b3_n777_d100", {"shape": [3, 777, 100]}, [H, B]]],
    [["b32_n4096_d512", {"shape": [32, 4096, 512]}, H]], POS([4, 32768, 64]), tags=["3d", "middle_dim"])
add("reduce", "sum_bf16_long", "row sum of 1M-4M-element bf16 rows",
    RED_REF, reduce_row("sum_bf16_long", "Row sum over very long bf16 rows (one program per row)."), "reduce",
    [["m8_n4000000", {"shape": [8, 4000000]}, [B]], ["m4_n1000003", {"shape": [4, 1000003]}, [B]]],
    [["m16_n4194304", {"shape": [16, 4194304]}, B]], POS([4, 262144], [B]), tags=["bf16", "large_N"])
add("reduce", "sum_of_squares", "row sum of squares",
    '''
    def reference(x: torch.Tensor) -> torch.Tensor:
        """Sum of squares over the last dim."""
        return torch.sum(x * x, dim=-1)
    ''', reduce_row("sum_of_squares", "Row sum of squares over the last dim.", op="sumsq"), "reduce",
    [["m4096_n8192", {"shape": [4096, 8192]}, [H, B]], ["m100_n3000", {"shape": [100, 3000]}, [H, B]]],
    [["m8192_n8192", {"shape": [8192, 8192]}, H]], [], tags=["sumsq"])


# =============================================================================================
# cross_entropy
# =============================================================================================
def ce_row(variant, note, mode="mean"):
    """mode: mean | none | ignore | smooth."""
    tgt = ("    target = tl.load(targets_ptr + row_idx)\n" if mode != "ignore" else
           "    target = tl.load(targets_ptr + row_idx)\n"
           "    valid = target != -100\n"
           "    target = tl.where(valid, target, 0)\n")
    extra = "    row_mean = tl.sum(tl.where(mask, logits, 0.0), axis=0) / n_cols\n" if mode == "smooth" else ""
    loss = {"mean": "lse - target_logit", "none": "lse - target_logit",
            "ignore": "tl.where(valid, lse - target_logit, 0.0)",
            "smooth": "0.9 * (lse - target_logit) + 0.1 * (lse - row_mean)"}[mode]
    ret = {"mean": "    return losses.mean().to(logits.dtype)\n",
           "none": "    return losses.to(logits.dtype).view(targets.shape)\n",
           "ignore": "    n_valid = (t2 != -100).sum()\n    return (losses.sum() / n_valid).to(logits.dtype)\n",
           "smooth": "    return losses.mean().to(logits.dtype)\n"}[mode]
    return starter_doc("cross_entropy", variant, note) + (
        "\n\n@triton.jit\n"
        "def cross_entropy_kernel(logits_ptr, targets_ptr, losses_ptr, n_cols, stride_row,\n"
        "                         BLOCK_SIZE: tl.constexpr):\n"
        '    """One program per row; whole row in one block; fp32 log-sum-exp."""\n'
        "    row_idx = tl.program_id(0)\n"
        "    row_start = logits_ptr + row_idx.to(tl.int64) * stride_row\n"
        "    cols = tl.arange(0, BLOCK_SIZE)\n"
        "    mask = cols < n_cols\n"
        '    logits = tl.load(row_start + cols, mask=mask, other=float("-inf")).to(tl.float32)\n'
        "    row_max = tl.max(logits, axis=0)\n"
        "    lse = row_max + tl.log(tl.sum(tl.exp(logits - row_max), axis=0))\n"
        + extra + tgt +
        "    target_logit = tl.load(row_start + target).to(tl.float32)\n"
        f"    tl.store(losses_ptr + row_idx, {loss})\n\n\n"
        "def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:\n"
        "    V = logits.shape[-1]\n"
        "    l2 = logits.reshape(-1, V)\n"
        "    t2 = targets.reshape(-1)\n"
        "    losses = torch.empty(l2.shape[0], device=logits.device, dtype=torch.float32)\n"
        "    cross_entropy_kernel[(l2.shape[0],)](l2, t2, losses, V, l2.stride(0),\n"
        "                                         BLOCK_SIZE=triton.next_power_of_2(V))\n"
        + ret)


CE_LOOPED = starter_doc("cross_entropy", "vocab_151936", "Qwen-size vocab: one program per row with an\n"
                        "online (running max / rescaled sum) log-sum-exp over BLOCK_SIZE chunks.") + d('''

    @triton.jit
    def cross_entropy_online_kernel(logits_ptr, targets_ptr, losses_ptr, n_cols, stride_row,
                                    BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        row_start = logits_ptr + row * stride_row
        offs = tl.arange(0, BLOCK_SIZE)
        m_vec = tl.full((BLOCK_SIZE,), -1e30, dtype=tl.float32)
        s_vec = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start in range(0, n_cols, BLOCK_SIZE):
            cols = start + offs
            x = tl.load(row_start + cols, mask=cols < n_cols, other=float("-inf")).to(tl.float32)
            m_new = tl.maximum(m_vec, x)
            s_vec = s_vec * tl.exp(m_vec - m_new) + tl.exp(x - m_new)
            m_vec = m_new
        m = tl.max(m_vec, axis=0)
        lse = m + tl.log(tl.sum(s_vec * tl.exp(m_vec - m), axis=0))
        target = tl.load(targets_ptr + row)
        target_logit = tl.load(row_start + target).to(tl.float32)
        tl.store(losses_ptr + row, lse - target_logit)


    def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        V = logits.shape[-1]
        l2 = logits.reshape(-1, V)
        t2 = targets.reshape(-1)
        losses = torch.empty(l2.shape[0], device=logits.device, dtype=torch.float32)
        cross_entropy_online_kernel[(l2.shape[0],)](l2, t2, losses, V, l2.stride(0), BLOCK_SIZE=4096)
        return losses.mean().to(logits.dtype)
''')
CE_REF = '''
    def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy loss over all rows."""
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
'''
CE_ADV = lambda lead, V, dts=(H, B): [["big_logits", "big_logits_x100", {"lead": lead, "vocab": V}, list(dts)]]
add("cross_entropy", "nonpow2_vocab", "mean CE, vocab 3001 / 32003",
    CE_REF, ce_row("nonpow2_vocab", "Mean cross-entropy; vocab sizes are not powers of 2."), "cross_entropy",
    [["b1000_v3001", {"lead": [1000], "vocab": 3001}, ALL], ["b4096_v32003", {"lead": [4096], "vocab": 32003}, [H, B]]],
    [["b4096_v32003", {"lead": [4096], "vocab": 32003}, H]], CE_ADV([1024, ], 32003), tags=["nonpow2"])
add("cross_entropy", "vocab_151936", "mean CE with Qwen vocab 151936 (looped/online LSE)",
    CE_REF, CE_LOOPED, "cross_entropy",
    [["b512_v151936", {"lead": [512], "vocab": 151936}, [H, B]], ["b64_v151936", {"lead": [64], "vocab": 151936}, [F]]],
    [["b1024_v151936", {"lead": [1024], "vocab": 151936}, H]], CE_ADV([256], 151936, [H]),
    tags=["large_vocab", "online"])
add("cross_entropy", "reduction_none", "per-row CE losses (reduction='none')",
    '''
    def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Per-row cross-entropy losses (no reduction)."""
        return F.cross_entropy(logits, targets, reduction="none")
    ''', ce_row("reduction_none", "Per-row cross-entropy (reduction='none').", mode="none"), "cross_entropy",
    [["b4096_v32000", {"lead": [4096], "vocab": 32000}, [H, B]], ["b100_v5000", {"lead": [100], "vocab": 5000}, ALL]],
    [["b4096_v32000", {"lead": [4096], "vocab": 32000}, H]], CE_ADV([1024], 32000), tags=["no_reduction"])
add("cross_entropy", "ignore_index", "mean CE with ~10% targets = -100 (ignored)",
    '''
    def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy over rows whose target != -100 (ignore_index)."""
        return F.cross_entropy(logits, targets, ignore_index=-100)
    ''', ce_row("ignore_index", "Mean cross-entropy that skips targets == -100.", mode="ignore"), "cross_entropy",
    [["b4096_v32000", {"lead": [4096], "vocab": 32000}, [H, B]], ["b333_v1000", {"lead": [333], "vocab": 1000}, ALL]],
    [["b4096_v32000", {"lead": [4096], "vocab": 32000}, H]], CE_ADV([1024], 32000),
    gen_args={"ignore_frac": 0.1}, tags=["ignore_index"])
add("cross_entropy", "label_smoothing", "mean CE with label_smoothing=0.1",
    '''
    def reference(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Mean cross-entropy with label smoothing 0.1."""
        return F.cross_entropy(logits, targets, label_smoothing=0.1)
    ''', ce_row("label_smoothing", "Mean cross-entropy with label smoothing 0.1:\n"
                "loss = 0.9 * (lse - x[target]) + 0.1 * (lse - mean(x)).", mode="smooth"), "cross_entropy",
    [["b4096_v32000", {"lead": [4096], "vocab": 32000}, [H, B]], ["b100_v5000", {"lead": [100], "vocab": 5000}, ALL]],
    [["b4096_v32000", {"lead": [4096], "vocab": 32000}, H]], CE_ADV([1024], 32000), tags=["label_smoothing"])
add("cross_entropy", "3d_logits", "mean CE on [B, S, V] logits",
    CE_REF, ce_row("3d_logits", "Mean cross-entropy on [batch, seq, vocab] logits."), "cross_entropy",
    [["b4_s512_v32000", {"lead": [4, 512], "vocab": 32000}, [H, B]], ["b2_s7_v1000", {"lead": [2, 7], "vocab": 1000}, ALL]],
    [["b8_s512_v32000", {"lead": [8, 512], "vocab": 32000}, H]], CE_ADV([2, 256], 32000), tags=["3d"])
add("cross_entropy", "bf16_vocab_100277", "mean CE, bf16, vocab 100277",
    CE_REF, ce_row("bf16_vocab_100277", "Mean cross-entropy, bfloat16 logits, vocab 100277."), "cross_entropy",
    [["b2048_v100277", {"lead": [2048], "vocab": 100277}, [B]], ["b100_v4096", {"lead": [100], "vocab": 4096}, [B]]],
    [["b2048_v100277", {"lead": [2048], "vocab": 100277}, B]], CE_ADV([256], 100277, [B]), tags=["bf16", "large_vocab"])


# =============================================================================================
# rotary_embedding
# =============================================================================================
ROPE_REF = '''
    def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Interleaved RoPE. x: [batch, heads, seq, head_dim]; cos, sin: [seq, head_dim // 2]."""
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.stack([rx1, rx2], dim=-1).flatten(-2)
'''


def rope_starter(variant, note, style="interleaved", layout="bhsd"):
    if style == "interleaved":
        ld = ("    x1 = tl.load(x_row + offs * 2, mask=mask, other=0.0).to(tl.float32)\n"
              "    x2 = tl.load(x_row + offs * 2 + 1, mask=mask, other=0.0).to(tl.float32)\n")
        st = ("    tl.store(o_row + offs * 2, x1 * c - x2 * s, mask=mask)\n"
              "    tl.store(o_row + offs * 2 + 1, x1 * s + x2 * c, mask=mask)\n")
    else:
        ld = ("    x1 = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)\n"
              "    x2 = tl.load(x_row + half + offs, mask=mask, other=0.0).to(tl.float32)\n")
        st = ("    tl.store(o_row + offs, x1 * c - x2 * s, mask=mask)\n"
              "    tl.store(o_row + half + offs, x2 * c + x1 * s, mask=mask)\n")
    pos = "(row // H) % S" if layout == "bshd" else "row % S"
    shape = ("    Bt, S, Hh, D = x.shape\n" if layout == "bshd" else "    Bt, Hh, S, D = x.shape\n")
    return starter_doc("rotary_embedding", variant, note) + (
        "\n\n@triton.jit\n"
        "def rope_kernel(X_ptr, COS_ptr, SIN_ptr, OUT_ptr, S, H, half, BLOCK_SIZE: tl.constexpr):\n"
        f'    """One program per (row of head_dim elements); position = {pos}."""\n'
        "    row = tl.program_id(0).to(tl.int64)\n"
        f"    p = {pos}\n"
        "    offs = tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offs < half\n"
        "    x_row = X_ptr + row * (2 * half)\n"
        "    o_row = OUT_ptr + row * (2 * half)\n"
        + ld +
        "    c = tl.load(COS_ptr + p * half + offs, mask=mask, other=1.0).to(tl.float32)\n"
        "    s = tl.load(SIN_ptr + p * half + offs, mask=mask, other=0.0).to(tl.float32)\n"
        + st + "\n\n"
        "def kernel_fn(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:\n"
        + shape +
        "    x = x.contiguous()\n"
        "    cos, sin = cos.contiguous(), sin.contiguous()\n"
        "    out = torch.empty_like(x)\n"
        "    n_rows = x.numel() // D\n"
        "    rope_kernel[(n_rows,)](x, cos, sin, out, S, Hh, D // 2, BLOCK_SIZE=triton.next_power_of_2(D // 2))\n"
        "    return out\n")


RB = lambda b, h, s, d_: {"b": b, "h": h, "s": s, "d": d_}
ROPE_ADV = lambda sz, dts=(H, B): [["big_x", "big_x_x1000", sz, list(dts)]]
add("rotary_embedding", "head_dim_96", "interleaved RoPE, head_dim 96",
    ROPE_REF, rope_starter("head_dim_96", "Interleaved RoPE; head_dim 96 (half = 48, masked block)."), "rotary",
    [["b1_h32_s2048_d96", RB(1, 32, 2048, 96), ALL], ["b2_h4_s100_d96", RB(2, 4, 100, 96), ALL]],
    [["b2_h32_s2048_d96", RB(2, 32, 2048, 96), H]], ROPE_ADV(RB(1, 8, 1024, 96)), tags=["head_dim_96"])
add("rotary_embedding", "head_dim_80", "interleaved RoPE, head_dim 80",
    ROPE_REF, rope_starter("head_dim_80", "Interleaved RoPE; head_dim 80."), "rotary",
    [["b2_h20_s1024_d80", RB(2, 20, 1024, 80), ALL], ["b1_h3_s77_d80", RB(1, 3, 77, 80), ALL]],
    [["b4_h20_s2048_d80", RB(4, 20, 2048, 80), H]], ROPE_ADV(RB(1, 8, 1024, 80)), tags=["head_dim_80"])
add("rotary_embedding", "head_dim_256", "interleaved RoPE, head_dim 256",
    ROPE_REF, rope_starter("head_dim_256", "Interleaved RoPE; head_dim 256."), "rotary",
    [["b1_h8_s4096_d256", RB(1, 8, 4096, 256), [H, B]], ["b2_h2_s33_d256", RB(2, 2, 33, 256), ALL]],
    [["b2_h8_s4096_d256", RB(2, 8, 4096, 256), H]], ROPE_ADV(RB(1, 4, 1024, 256)), tags=["head_dim_256"])
add("rotary_embedding", "head_dim_32", "interleaved RoPE, head_dim 32",
    ROPE_REF, rope_starter("head_dim_32", "Interleaved RoPE; head_dim 32."), "rotary",
    [["b4_h32_s2048_d32", RB(4, 32, 2048, 32), [H, B]], ["b1_h2_s5_d32", RB(1, 2, 5, 32), ALL]],
    [["b8_h32_s2048_d32", RB(8, 32, 2048, 32), H]], ROPE_ADV(RB(1, 8, 1024, 32)), tags=["head_dim_32"])
add("rotary_embedding", "neox_rotate_half", "NeoX/Llama-HF rotate-half RoPE (first half / second half)",
    '''
    def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Rotate-half (NeoX) RoPE. x: [batch, heads, seq, head_dim]; cos, sin: [seq, head_dim // 2]."""
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    ''', rope_starter("neox_rotate_half", "Rotate-half RoPE: pairs (i, i + head_dim/2).", style="half"), "rotary",
    [["b2_h32_s1024_d128", RB(2, 32, 1024, 128), ALL], ["b1_h4_s63_d128", RB(1, 4, 63, 128), ALL]],
    [["b2_h32_s2048_d128", RB(2, 32, 2048, 128), H]], ROPE_ADV(RB(1, 8, 1024, 128)), tags=["rotate_half"])
add("rotary_embedding", "bshd_layout", "interleaved RoPE on [B, S, H, D] layout",
    '''
    def reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Interleaved RoPE on x: [batch, seq, heads, head_dim]; cos, sin: [seq, head_dim // 2]."""
        c = cos[:, None, :]
        s = sin[:, None, :]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)
    ''', rope_starter("bshd_layout", "Interleaved RoPE, [B, S, H, D] layout: position = (row // H) % S.",
                      layout="bshd"), "rotary",
    [["b2_s1024_h32_d128", RB(2, 32, 1024, 128), ALL], ["b1_s50_h3_d64", RB(1, 3, 50, 64), ALL]],
    [["b2_s2048_h32_d128", RB(2, 32, 2048, 128), H]], ROPE_ADV(RB(1, 8, 1024, 128)),
    gen_args={"layout": "bshd"}, tags=["bshd"])
add("rotary_embedding", "bf16", "interleaved RoPE, bf16 only",
    ROPE_REF, rope_starter("bf16", "Interleaved RoPE, bfloat16 inputs."), "rotary",
    [["b2_h32_s2048_d128", RB(2, 32, 2048, 128), [B]], ["b1_h8_s127_d64", RB(1, 8, 127, 64), [B]]],
    [["b2_h32_s2048_d128", RB(2, 32, 2048, 128), B]], ROPE_ADV(RB(1, 8, 1024, 128), [B]), tags=["bf16"])


# =============================================================================================
# matmul
# =============================================================================================
def mm_starter(variant, note, mode="mm", bm=64, bn=64, bk=32):
    """mode: mm (A[M,K] @ B[K,N]) | linear (x @ w.T) | bias_relu | bmm."""
    batch = mode == "bmm"
    bias = mode == "bias_relu"
    epi64 = ("        acc64 += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float64)[None, :]\n"
             "        acc64 = tl.maximum(acc64, 0.0)\n" if bias else "")
    epi = ("        acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]\n"
           "        acc = tl.maximum(acc, 0.0)\n" if bias else "")
    boff = ("    pid_b = tl.program_id(2).to(tl.int64)\n"
            "    A_ptr += pid_b * stride_ab\n    B_ptr += pid_b * stride_bb\n    C_ptr += pid_b * stride_cb\n"
            if batch else "")
    sig_extra = "stride_ab, stride_bb, stride_cb, " if batch else ""
    kern = (
        "\n\n@triton.jit\n"
        "def matmul_kernel(\n"
        f"    A_ptr, B_ptr, C_ptr, {'bias_ptr, ' if bias else ''}M, N, K,\n"
        f"    {sig_extra}stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,\n"
        "    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,\n"
        "    GROUP_K: tl.constexpr, IEEE_FP32: tl.constexpr,\n"
        "):\n"
        '    """Tiled matmul (same accumulation scheme as the fixed AutoKernel matmul starter):\n'
        "    fp16/bf16: fp32 tensor-core partials per GROUP_K tiles, summed with fp32 adds;\n"
        '    fp32: input_precision="ieee" tiles summed in fp64."""\n'
        "    pid_m = tl.program_id(0)\n"
        "    pid_n = tl.program_id(1)\n"
        + boff +
        "    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)\n"
        "    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)\n"
        "    offs_k = tl.arange(0, BLOCK_SIZE_K)\n"
        "    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak\n"
        "    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn\n"
        "    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn\n"
        "    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)\n"
        "    if IEEE_FP32:\n"
        "        acc64 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float64)\n"
        "        for k in range(0, K, BLOCK_SIZE_K):\n"
        "            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)\n"
        "            b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)\n"
        '            acc64 += tl.dot(a, b, input_precision="ieee").to(tl.float64)\n'
        "            a_ptrs += BLOCK_SIZE_K * stride_ak\n"
        "            b_ptrs += BLOCK_SIZE_K * stride_bk\n"
        "            offs_k += BLOCK_SIZE_K\n"
        + epi64 +
        "        tl.store(c_ptrs, acc64.to(tl.float32), mask=c_mask)\n"
        "    else:\n"
        "        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)\n"
        "        for k0 in range(0, K, BLOCK_SIZE_K * GROUP_K):\n"
        "            blk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)\n"
        "            for kk in tl.static_range(GROUP_K):\n"
        "                a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)\n"
        "                b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)\n"
        "                blk += tl.dot(a, b)\n"
        "                a_ptrs += BLOCK_SIZE_K * stride_ak\n"
        "                b_ptrs += BLOCK_SIZE_K * stride_bk\n"
        "                offs_k += BLOCK_SIZE_K\n"
        "            acc += blk\n"
        + epi +
        "        tl.store(c_ptrs, acc.to(C_ptr.dtype.element_ty), mask=c_mask)\n\n\n")
    if mode == "linear":
        fn = ("def kernel_fn(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:\n"
              "    M, K = x.shape\n    N = w.shape[0]\n"
              "    A, Bm = x, w\n"
              "    C = torch.empty((M, N), device=x.device, dtype=x.dtype)\n")
        strides = ("A.stride(0), A.stride(1), Bm.stride(1), Bm.stride(0), C.stride(0), C.stride(1)")  # w^T
    elif batch:
        fn = ("def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:\n"
              "    Bt, M, K = A.shape\n    N = B.shape[2]\n    Bm = B\n"
              "    C = torch.empty((Bt, M, N), device=A.device, dtype=A.dtype)\n")
        strides = ("A.stride(0), Bm.stride(0), C.stride(0), A.stride(1), A.stride(2), Bm.stride(1), Bm.stride(2), "
                   "C.stride(1), C.stride(2)")
    else:
        fn = (f"def kernel_fn(A: torch.Tensor, B: torch.Tensor{', bias: torch.Tensor' if bias else ''}) -> torch.Tensor:\n"
              "    M, K = A.shape\n    N = B.shape[1]\n    Bm = B\n"
              "    C = torch.empty((M, N), device=A.device, dtype=A.dtype)\n")
        strides = "A.stride(0), A.stride(1), Bm.stride(0), Bm.stride(1), C.stride(0), C.stride(1)"
    grid = "(triton.cdiv(M, BM), triton.cdiv(N, BN)" + (", Bt)" if batch else ")")
    a0 = "x" if mode == "linear" else "A"
    fn += (f"    BM, BN, BK = {bm}, {bn}, {bk}\n"
           f"    matmul_kernel[{grid}](\n"
           f"        A, Bm, C, {'bias, ' if bias else ''}M, N, K, {strides},\n"
           "        BLOCK_SIZE_M=BM, BLOCK_SIZE_N=BN, BLOCK_SIZE_K=BK, GROUP_K=2,\n"
           f"        IEEE_FP32=({a0}.dtype == torch.float32))\n"
           "    return C\n")
    return starter_doc("matmul", variant, note) + kern + fn


MM = lambda m, n, k, **kw: {"M": m, "N": n, "K": k, **kw}
MM_REF = '''
    def reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """C = A @ B, A: [M, K], B: [K, N]."""
        return torch.matmul(A, B)
'''
MM_ADV = lambda dts=(H, B): [["big_scale", "big_scale_K4096", MM(1024, 1024, 4096), list(dts)],
                             ["pos_uniform", "pos_uniform_K8192", MM(256, 256, 8192), list(dts)]]
add("matmul", "nonpow2", "A @ B with non-power-of-2 M, N, K",
    MM_REF, mm_starter("nonpow2", "Tiled matmul; M, N, K are not multiples of the tile sizes."), "matmul",
    [["m777_n1537_k1025", MM(777, 1537, 1025), ALL], ["m4000_n3000_k1000", MM(4000, 3000, 1000), ALL]],
    [["m4000_n3000_k1000", MM(4000, 3000, 1000), H]], MM_ADV(), tags=["nonpow2"])
add("matmul", "bf16", "A @ B, bf16 only",
    MM_REF, mm_starter("bf16", "Tiled matmul, bfloat16 inputs."), "matmul",
    [["m2048_n3072_k1024", MM(2048, 3072, 1024), [B]], ["m100_n200_k300", MM(100, 200, 300), [B]]],
    [["m4096_n4096_k2048", MM(4096, 4096, 2048), B]], MM_ADV([B]), tags=["bf16"])
add("matmul", "fp32_exact", "A @ B, fp32 only (tol 1e-4 vs an fp64 golden; TF32 is not accurate enough)",
    MM_REF, mm_starter("fp32_exact", "Tiled matmul, float32 inputs (IEEE fp32 tiles, fp64 sum)."), "matmul",
    [["m1024_n1024_k1024", MM(1024, 1024, 1024), [F]], ["m500_n700_k300", MM(500, 700, 300), [F]],
     ["m1024_n1024_k4096", MM(1024, 1024, 4096), [F]]],
    [["m2048_n2048_k2048", MM(2048, 2048, 2048), F]], [["pos_uniform", "pos_uniform_K8192", MM(256, 256, 8192), [F]]],
    tags=["fp32"])
add("matmul", "linear_xwT", "x @ w.T with w stored [N, K] (nn.Linear, no bias)",
    '''
    def reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Linear layer without bias: x @ w.T, x: [M, K], w: [N, K]."""
        return x @ w.T
    ''', mm_starter("linear_xwT", "x @ w.T: w is read with swapped strides (no transpose copy).", mode="linear"),
    "matmul",
    [["m1000_n3000_k777", MM(1000, 3000, 777), ALL], ["m4096_n4096_k1024", MM(4096, 4096, 1024), [H]]],
    [["m4096_n4096_k1024", MM(4096, 4096, 1024), H]],
    [["big_scale", "big_scale_K4096", MM(1024, 1024, 4096), [H, B]]], gen_args={"mode": "linear"},
    tags=["transposed_B", "linear"])
add("matmul", "bias_relu", "relu(A @ B + bias) fused epilogue",
    '''
    def reference(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """relu(A @ B + bias), bias: [N]."""
        return torch.relu(torch.matmul(A, B) + bias)
    ''', mm_starter("bias_relu", "Matmul with a fused bias + ReLU epilogue.", mode="bias_relu"), "matmul",
    [["m2048_n2048_k2048", MM(2048, 2048, 2048), [H, B]], ["m1000_n999_k512", MM(1000, 999, 512), ALL]],
    [["m2048_n2048_k2048", MM(2048, 2048, 2048), H]],
    [["big_scale", "big_scale_K4096", MM(1024, 1024, 4096), [H]]], gen_args={"mode": "bias_relu"},
    tags=["epilogue", "fusion"])
add("matmul", "batched_bmm", "batched matmul [b, M, K] @ [b, K, N]",
    '''
    def reference(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Batched matmul: A [b, M, K] @ B [b, K, N]."""
        return torch.bmm(A, B)
    ''', mm_starter("batched_bmm", "Batched matmul; grid axis 2 = batch.", mode="bmm"), "matmul",
    [["b32_m256_n256_k64", MM(256, 256, 64, batch=32), ALL], ["b8_m1000_n777_k500", MM(1000, 777, 500, batch=8), ALL]],
    [["b64_m512_n512_k128", MM(512, 512, 128, batch=64), H]],
    [["big_scale", "big_scale_K4096", MM(512, 512, 4096, batch=4), [H]]], gen_args={"mode": "bmm"},
    tags=["batched"])
add("matmul", "skinny_M", "decode-style skinny matmul, M = 1-16",
    MM_REF, mm_starter("skinny_M", "Skinny matmul (M = 1..16): BLOCK_SIZE_M = 16.", bm=16, bn=64, bk=64), "matmul",
    [["m8_n4096_k4096", MM(8, 4096, 4096), [H, B]], ["m1_n11008_k4096", MM(1, 11008, 4096), [H]],
     ["m16_n1000_k777", MM(16, 1000, 777), [F]]],
    [["m8_n11008_k4096", MM(8, 11008, 4096), H]],
    [["big_scale", "big_scale_K4096", MM(16, 1024, 4096), [H]]], tags=["skinny", "gemv"])


# =============================================================================================
# flash_attention
# =============================================================================================
def attn_ref(causal=True, gqa=False, doc=""):
    rep = ("    K = K.repeat_interleave(Q.shape[1] // K.shape[1], dim=1)\n"
           "    V = V.repeat_interleave(Q.shape[1] // V.shape[1], dim=1)\n" if gqa else "")
    mask = ("    s_q, s_k = Q.shape[-2], K.shape[-2]\n"
            "    mask = torch.triu(torch.ones(s_q, s_k, device=Q.device, dtype=torch.bool), diagonal=1)\n"
            "    attn = attn.masked_fill(mask, float('-inf'))\n" if causal else "")
    return ("def reference(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:\n"
            f'    """{doc}"""\n' + rep +
            "    attn = torch.matmul(Q, K.transpose(-2, -1)) * (Q.shape[-1] ** -0.5)\n" + mask +
            "    return torch.matmul(F.softmax(attn, dim=-1), V)\n")


def attn_starter(variant, note, causal=True):
    return starter_doc("flash_attention", variant, note) + d('''
        import math


        @triton.jit
        def flash_attention_kernel(
            Q_ptr, K_ptr, V_ptr, O_ptr,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vn, stride_vk,
            stride_oz, stride_oh, stride_om, stride_ok,
            M_size, N_size, GROUP, sm_scale,
            D: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        ):
            """Online-softmax attention (fixed AutoKernel starter). Program = (query block, head, batch).
            K/V head = head // GROUP (GQA); keys may be longer than queries (N_size != M_size)."""
            pid_m = tl.program_id(0)
            pid_h = tl.program_id(1)
            pid_z = tl.program_id(2)
            kv_h = pid_h // GROUP
            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_d = tl.arange(0, D)
            q = tl.load(Q_ptr + pid_z * stride_qz + pid_h * stride_qh + offs_m[:, None] * stride_qm
                        + offs_d[None, :] * stride_qk, mask=offs_m[:, None] < M_size, other=0.0)
            m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
            l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
            acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
            if IS_CAUSAL:
                kv_end = tl.minimum(N_size, (pid_m + 1) * BLOCK_M)
            else:
                kv_end = N_size
            k_base = K_ptr + pid_z * stride_kz + kv_h * stride_kh
            v_base = V_ptr + pid_z * stride_vz + kv_h * stride_vh
            for start_n in range(0, kv_end, BLOCK_N):
                offs_n = start_n + tl.arange(0, BLOCK_N)
                k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk,
                            mask=offs_n[:, None] < N_size, other=0.0)
                qk = tl.dot(q, tl.trans(k)) * sm_scale
                if IS_CAUSAL:
                    qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float("-inf"))
                qk = tl.where(offs_n[None, :] < N_size, qk, float("-inf"))
                m_new = tl.maximum(m_i, tl.max(qk, axis=1))
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(qk - m_new[:, None])
                l_i = l_i * alpha + tl.sum(p, axis=1)
                v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk,
                            mask=offs_n[:, None] < N_size, other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
                m_i = m_new
            acc = acc / l_i[:, None]
            tl.store(O_ptr + pid_z * stride_oz + pid_h * stride_oh + offs_m[:, None] * stride_om
                     + offs_d[None, :] * stride_ok, acc.to(O_ptr.dtype.element_ty), mask=offs_m[:, None] < M_size)


        def kernel_fn(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
            Z, Hq, M_size, D = Q.shape
            Hkv, N_size = K.shape[1], K.shape[2]
            O = torch.empty_like(Q)
            BLOCK_M = 64
            BLOCK_N, extra = (32, {"num_stages": 2}) if D >= 128 else (64, {})  # L4 shared memory
            grid = (triton.cdiv(M_size, BLOCK_M), Hq, Z)
            flash_attention_kernel[grid](
                Q, K, V, O,
                Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                O.stride(0), O.stride(1), O.stride(2), O.stride(3),
                M_size, N_size, Hq // Hkv, 1.0 / math.sqrt(D),
                D=D, IS_CAUSAL=CAUSAL, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, **extra,
            )
            return O
    ''').replace("CAUSAL, BLOCK_M=BLOCK_M", f"{causal}, BLOCK_M=BLOCK_M")


AT = lambda b, h, s, d_, **kw: {"b": b, "h": h, "s": s, "d": d_, **kw}
QK40 = lambda sz, dts=(H, B): [["qk_40", "qk_x40", sz, list(dts)]]
CAUSAL_DOC = "Causal scaled dot-product attention, Q/K/V: [batch, heads, seq, head_dim]."
add("flash_attention", "non_causal", "non-causal attention, head_dim 64",
    attn_ref(False, doc="Non-causal scaled dot-product attention, Q/K/V: [batch, heads, seq, head_dim]."),
    attn_starter("non_causal", "Non-causal attention (encoder).", causal=False), "attention",
    [["b2_h16_s1024_d64", AT(2, 16, 1024, 64), [H, B]], ["b1_h4_s100_d64", AT(1, 4, 100, 64), [H, B]]],
    [["b2_h16_s2048_d64", AT(2, 16, 2048, 64), H]], QK40(AT(2, 8, 512, 64)), tags=["non_causal"])
add("flash_attention", "head_dim_128", "causal attention, head_dim 128 (L4 shared-memory limit)",
    attn_ref(True, doc=CAUSAL_DOC), attn_starter("head_dim_128", "Causal attention, head_dim 128."), "attention",
    [["b2_h4_s257_d128", AT(2, 4, 257, 128), [H, B]], ["b1_h16_s1024_d128", AT(1, 16, 1024, 128), [H]]],
    [["b1_h16_s2048_d128", AT(1, 16, 2048, 128), H]], [["qk_4", "qk_x4", AT(1, 8, 512, 128), [H]]],
    tags=["head_dim_128"])
add("flash_attention", "head_dim_32", "causal attention, head_dim 32",
    attn_ref(True, doc=CAUSAL_DOC), attn_starter("head_dim_32", "Causal attention, head_dim 32."), "attention",
    [["b4_h16_s1024_d32", AT(4, 16, 1024, 32), [H]], ["b1_h2_s70_d32", AT(1, 2, 70, 32), [H, B]]],
    [["b4_h16_s2048_d32", AT(4, 16, 2048, 32), H]], QK40(AT(2, 8, 512, 32), [H]), tags=["head_dim_32"])
add("flash_attention", "nonpow2_seq", "causal attention, seq 1000 / 777 / 1",
    attn_ref(True, doc=CAUSAL_DOC), attn_starter("nonpow2_seq", "Causal attention; seq_len not a multiple of the block."),
    "attention",
    [["b2_h8_s1000_d64", AT(2, 8, 1000, 64), [H, B]], ["b1_h8_s777_d64", AT(1, 8, 777, 64), [H]],
     ["b1_h2_s1_d64", AT(1, 2, 1, 64), [H]]],
    [["b2_h16_s1500_d64", AT(2, 16, 1500, 64), H]], QK40(AT(1, 8, 999, 64), [H]), tags=["nonpow2"])
add("flash_attention", "bf16", "causal attention, bf16 only",
    attn_ref(True, doc=CAUSAL_DOC), attn_starter("bf16", "Causal attention, bfloat16 inputs."), "attention",
    [["b2_h32_s1024_d64", AT(2, 32, 1024, 64), [B]], ["b1_h8_s127_d64", AT(1, 8, 127, 64), [B]]],
    [["b2_h32_s2048_d64", AT(2, 32, 2048, 64), B]], QK40(AT(2, 8, 512, 64), [B]), tags=["bf16"])
add("flash_attention", "gqa", "causal grouped-query attention (32 query heads, 8 KV heads)",
    attn_ref(True, gqa=True, doc="Causal grouped-query attention. Q: [b, h, s, d]; K, V: [b, h_kv, s, d], h % h_kv == 0."),
    attn_starter("gqa", "Causal GQA: query head h reads KV head h // (h / h_kv)."), "attention",
    [["b2_h32_kv8_s1024_d64", AT(2, 32, 1024, 64, hkv=8), [H, B]], ["b1_h8_kv2_s300_d64", AT(1, 8, 300, 64, hkv=2), [H]]],
    [["b2_h32_kv8_s2048_d64", AT(2, 32, 2048, 64, hkv=8), H]], QK40(AT(1, 16, 512, 64, hkv=4), [H]), tags=["gqa"])
add("flash_attention", "cross_attention", "non-causal cross-attention, 256 queries x 2048 keys",
    attn_ref(False, doc="Non-causal cross-attention. Q: [b, h, s_q, d]; K, V: [b, h, s_k, d]."),
    attn_starter("cross_attention", "Non-causal attention with s_k != s_q.", causal=False), "attention",
    [["b2_h16_q256_k2048_d64", AT(2, 16, 256, 64, sk=2048), [H, B]], ["b1_h4_q77_k1000_d64", AT(1, 4, 77, 64, sk=1000), [H]]],
    [["b4_h16_q512_k4096_d64", AT(4, 16, 512, 64, sk=4096), H]], QK40(AT(1, 8, 128, 64, sk=1024), [H]),
    tags=["cross", "non_causal"])


# =============================================================================================
# fused_mlp
# =============================================================================================
def mlp_starter(variant, note, act="silu", down=True):
    act_code = {"silu": "g * tl.sigmoid(g)",
                "gelu_tanh": "0.5 * g * (1.0 + (2.0 * tl.sigmoid(2.0 * 0.7978845608028654 * (g + 0.044715 * g * g * g)) - 1.0))",
                "relu2": "tl.maximum(g, 0.0) * tl.maximum(g, 0.0)"}[act]
    store = ("    tl.store(out_ptrs, h, mask=out_mask)  # float32 hidden\n" if down else
             "    tl.store(out_ptrs, h.to(Out_ptr.dtype.element_ty), mask=out_mask)\n")
    args = "x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor" + (", w_down: torch.Tensor" if down else "")
    tail = ("    out = (hidden @ w_down.to(torch.float32).t()).to(x.dtype)  # down projection in fp32\n"
            "    return out.view(*lead, out.shape[-1])\n" if down else
            "    return hidden.view(*lead, N)\n")
    return starter_doc("fused_mlp", variant, note) + (
        "\n\n@triton.jit\n"
        "def gate_up_kernel(X_ptr, Wg_ptr, Wu_ptr, Out_ptr, M, N, K,\n"
        "                   stride_xm, stride_xk, stride_wk, stride_wn, stride_om, stride_on,\n"
        "                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,\n"
        "                   IEEE_FP32: tl.constexpr):\n"
        f'    """h = act(x @ w_gate.T) * (x @ w_up.T), act = {act}. W is [N, K], read transposed."""\n'
        "    pid_m = tl.program_id(0)\n"
        "    pid_n = tl.program_id(1)\n"
        "    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n"
        "    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n"
        "    offs_k = tl.arange(0, BLOCK_K)\n"
        "    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
        "    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n"
        "    for k0 in range(0, K, BLOCK_K):\n"
        "        kk = k0 + offs_k\n"
        "        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + kk[None, :] * stride_xk,\n"
        "                    mask=(offs_m[:, None] < M) & (kk[None, :] < K), other=0.0)\n"
        "        w_off = kk[:, None] * stride_wk + offs_n[None, :] * stride_wn\n"
        "        w_mask = (kk[:, None] < K) & (offs_n[None, :] < N)\n"
        "        wg = tl.load(Wg_ptr + w_off, mask=w_mask, other=0.0)\n"
        "        wu = tl.load(Wu_ptr + w_off, mask=w_mask, other=0.0)\n"
        "        if IEEE_FP32:\n"
        '            acc_g += tl.dot(x, wg, input_precision="ieee")\n'
        '            acc_u += tl.dot(x, wu, input_precision="ieee")\n'
        "        else:\n"
        "            acc_g += tl.dot(x, wg)\n"
        "            acc_u += tl.dot(x, wu)\n"
        "    g = acc_g\n"
        f"    h = ({act_code}) * acc_u\n"
        "    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on\n"
        "    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)\n"
        + store + "\n\n"
        f"def kernel_fn({args}) -> torch.Tensor:\n"
        "    lead = x.shape[:-1]\n"
        "    x2 = x.reshape(-1, x.shape[-1])\n"
        "    M, K = x2.shape\n"
        "    N = w_gate.shape[0]\n"
        f"    hidden = torch.empty((M, N), device=x.device, dtype={'torch.float32' if down else 'x.dtype'})\n"
        "    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32\n"
        "    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n"
        "    gate_up_kernel[grid](x2, w_gate, w_up, hidden, M, N, K,\n"
        "                         x2.stride(0), x2.stride(1), w_gate.stride(1), w_gate.stride(0),\n"
        "                         hidden.stride(0), hidden.stride(1),\n"
        "                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,\n"
        "                         IEEE_FP32=(x.dtype == torch.float32))\n"
        + tail)


def mlp_ref(act="silu", down=True, doc=""):
    a = {"silu": "F.silu(x @ w_gate.T)", "gelu_tanh": 'F.gelu(x @ w_gate.T, approximate="tanh")',
         "relu2": "F.relu(x @ w_gate.T) ** 2"}[act]
    if down:
        return ("def reference(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor,\n"
                "              w_down: torch.Tensor) -> torch.Tensor:\n"
                f'    """{doc}"""\n'
                f"    gate = {a}\n"
                "    up = x @ w_up.T\n"
                "    return (gate * up) @ w_down.T\n")
    return ("def reference(x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:\n"
            f'    """{doc}"""\n'
            f"    return {a} * (x @ w_up.T)\n")


ML = lambda lead, dim, hidden: {"lead": lead, "dim": dim, "hidden": hidden}
BIG = lambda sz, dts=(H, B): [["big_act", "big_act", sz, list(dts)]]
add("fused_mlp", "gelu_tanh", "GeGLU-style MLP with tanh-approximate GELU gate",
    mlp_ref("gelu_tanh", doc="down(gelu_tanh(x @ w_gate.T) * (x @ w_up.T)); weights [out, in]."),
    mlp_starter("gelu_tanh", "Fused gate/up with a tanh-GELU gate (tanh via 2*sigmoid(2y)-1), fp32 hidden,\n"
                "down projection in fp32.", act="gelu_tanh"), "mlp",
    [["b1024_d1024_h2816", ML([1024], 1024, 2816), [H, B]], ["b100_d512_h1000", ML([100], 512, 1000), ALL]],
    [["b2048_d2048_h5632", ML([2048], 2048, 5632), H]], BIG(ML([512], 1024, 2048)), tags=["gelu_tanh"])
add("fused_mlp", "nonpow2", "SwiGLU MLP with non-multiple-of-64 dims",
    mlp_ref("silu", doc="SwiGLU MLP: down(silu(x @ w_gate.T) * (x @ w_up.T)); weights [out, in]."),
    mlp_starter("nonpow2", "Fused SwiGLU gate/up, fp32 hidden, down projection in fp32."), "mlp",
    [["b777_d1024_h2730", ML([777], 1024, 2730), [H, B]], ["b100_d1000_h3001", ML([100], 1000, 3001), ALL]],
    [["b2048_d2048_h5461", ML([2048], 2048, 5461), H]], BIG(ML([512], 1024, 2048)), tags=["nonpow2"])
add("fused_mlp", "bf16", "SwiGLU MLP, bf16 only",
    mlp_ref("silu", doc="SwiGLU MLP: down(silu(x @ w_gate.T) * (x @ w_up.T)); weights [out, in]."),
    mlp_starter("bf16", "Fused SwiGLU gate/up, fp32 hidden, bfloat16 inputs."), "mlp",
    [["b2048_d2048_h5632", ML([2048], 2048, 5632), [B]], ["b64_d1024_h2816", ML([64], 1024, 2816), [B]]],
    [["b2048_d4096_h11008", ML([2048], 4096, 11008), B]], BIG(ML([512], 1024, 2048), [B]), tags=["bf16"])
add("fused_mlp", "relu2", "MLP with squared-ReLU gate",
    mlp_ref("relu2", doc="down(relu(x @ w_gate.T)**2 * (x @ w_up.T)); weights [out, in]."),
    mlp_starter("relu2", "Fused gate/up with a squared-ReLU gate, fp32 hidden.", act="relu2"), "mlp",
    [["b1024_d1024_h4096", ML([1024], 1024, 4096), [H, B]], ["b100_d512_h1000", ML([100], 512, 1000), ALL]],
    [["b2048_d2048_h5632", ML([2048], 2048, 5632), H]], [], tags=["relu2"])
add("fused_mlp", "gate_up_only", "silu(x @ w_gate.T) * (x @ w_up.T) without the down projection",
    mlp_ref("silu", down=False, doc="Fused SwiGLU gate/up without the down projection -> [M, hidden]."),
    mlp_starter("gate_up_only", "Fused SwiGLU gate/up only (output in the input dtype).", down=False), "mlp",
    [["b2048_d2048_h5632", ML([2048], 2048, 5632), [H, B]], ["b100_d1024_h2000", ML([100], 1024, 2000), ALL]],
    [["b2048_d4096_h11008", ML([2048], 4096, 11008), H]], [], gen_args={"down": False}, tags=["no_down_proj"])
add("fused_mlp", "3d_input", "SwiGLU MLP on [B, S, D] input",
    mlp_ref("silu", doc="SwiGLU MLP on x: [batch, seq, dim]; weights [out, in]."),
    mlp_starter("3d_input", "Fused SwiGLU on [batch, seq, dim] input (flattened to rows)."), "mlp",
    [["b4_s256_d1024_h2816", ML([4, 256], 1024, 2816), [H, B]], ["b2_s7_d512_h1024", ML([2, 7], 512, 1024), ALL]],
    [["b8_s512_d2048_h5632", ML([8, 512], 2048, 5632), H]], BIG(ML([2, 256], 1024, 2048)), tags=["3d"])


# =============================================================================================
def spec_of(p):
    pid = f"{p['family']}/{p['variant']}"
    dtypes = sorted({dt for _, _, dts in p["sizes"] for dt in dts})
    return {"id": pid, "family": p["family"], "variant": p["variant"], "desc": p["desc"], "gen": p["gen"],
            "gen_args": p["gen_args"], "sizes": p["sizes"], "adversarial": p["adversarial"],
            "timing": p["timing"], "dtypes": dtypes,
            "tol": {dt: {"atol": a, "rtol": r} for dt, (a, r) in TOL[p["family"]].items()},
            "tags": [p["family"]] + p["tags"], "val_set_version": "v1"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    seen = set()
    for p in PROBLEMS:
        spec = spec_of(p)
        assert spec["id"] not in seen, spec["id"]
        seen.add(spec["id"])
        dd = OUT / spec["id"].replace("/", "__")
        dd.mkdir(exist_ok=True)
        (dd / "reference.py").write_text(p["ref"])
        p["starter"] = re.sub(r"\n{3,}", "\n\n\n", re.sub(r"(import triton\.language as tl\n)(?=\S)", r"\1\n\n", p["starter"]))
        (dd / "starter.py").write_text(p["starter"])
        (dd / "problem.json").write_text(json.dumps(spec, indent=1))
        compile(p["ref"], str(dd / "reference.py"), "exec")
        compile(p["starter"], str(dd / "starter.py"), "exec")
        rows.append({"id": spec["id"], "dir": dd.name, "family": spec["family"], "variant": spec["variant"],
                     "desc": spec["desc"], "tags": spec["tags"], "dtypes": spec["dtypes"],
                     "timing": spec["timing"][0], "n_correctness_cases":
                         sum(len(x[2]) for x in spec["sizes"]) + sum(len(x[3]) for x in spec["adversarial"])})
    fams = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    META.write_text(json.dumps({
        "version": "v1", "created": "2026-09-30", "n_problems": len(rows), "by_family": fams,
        "harness": "modal_app/val_bench_core.py (bench v2 conventions)", "generator": "modal_app/val_set_v1.py",
        "not_kernelbench": True,
        "decontamination": "SFT data must drop examples whose (family, variant tags) match a problem here at the "
                           "op level (same reference semantics), and report seen/unseen families separately.",
        "problems": rows}, indent=1))
    print(len(rows), "problems", fams)


if __name__ == "__main__":
    main()
