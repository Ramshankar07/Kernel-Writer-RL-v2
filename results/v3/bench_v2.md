# bench v2: fixed AutoKernel harness + rescore of every old PASS kernel

Code: `modal_app/bench_v2_core.py` (harness), `modal_app/bench_v2_modal.py` (Modal app `autokernel-bench-v2`, L4, <=4 containers, no cache), `modal_app/bench_v2_report.py` (this file). v1 (`bench_modal.py`, app `autokernel-bench`, Dict `autokernel-bench-cache`) is untouched.

Data: `results/v3/bench_v2_sanity.jsonl`, `results/v3/rescore_bench_v2.jsonl`, `results/v3/bench_v2_baselines.jsonl`, `results/v3/rescore_summary.json`.

## Headline

- 362 distinct (by code sha256) kernels were PASS under the v1 full bench in GRPO v1/v2 rollouts and agent_eval (kernel types: cross_entropy, reduce, rmsnorm, softmax).
- Under bench v2: 346 survive (95.6%); verdicts {'PASS': 346, 'FAIL': 14, 'TIMEOUT': 2}.
- Kernels with a real speedup vs the starter (>1.05x geomean, CI-significant at every timing shape, v2 PASS): **0**. Median geomean speedup vs starter over survivors: 1.000x; vs eager PyTorch: 0.990x (v1 reported median 0.987x vs eager).
- Not counted (significant >1.05x at one timing shape only): 857345b45745 (cross_entropy, grpo_v1): large 1.00x [CI 1.00-1.00] vs starter = 1.01x vs torch.compile, gpt2 1.05x [CI 1.05-1.06] vs starter = 0.31x vs torch.compile; a3b8f062d840 (cross_entropy, grpo_v1,grpo_v2): large 1.00x [CI 1.00-1.00] vs starter = 1.01x vs torch.compile, gpt2 1.05x [CI 1.05-1.06] vs starter = 0.31x vs torch.compile; a6affa408962 (cross_entropy, grpo_v1): large 1.00x [CI 1.00-1.00] vs starter = 1.01x vs torch.compile, gpt2 1.05x [CI 1.05-1.06] vs starter = 0.31x vs torch.compile; aae7840e2865 (softmax, grpo_v2): large 1.00x [CI 1.00-1.00] vs starter = 1.01x vs torch.compile, vocab 2.13x [CI 2.13-2.13] vs starter = 0.20x vs torch.compile; b5a26e50441d (softmax, grpo_v1): large 1.00x [CI 1.00-1.01] vs starter = 1.01x vs torch.compile, vocab 2.47x [CI 2.47-2.47] vs starter = 0.23x vs torch.compile.
- 2 kernels TIMEOUT (a single evaluation hung > 420 s; counted as not surviving).

| kernel type | old PASS | v2 PASS | v2 FAIL (adversarial only) | real speedup vs starter | median vs starter (survivors) | max vs starter | median vs eager | median vs torch.compile | no @triton.jit |
|---|---|---|---|---|---|---|---|---|---|
| cross_entropy | 71 | 70 | 1 (0) | 0 | 0.998 | 1.026 | 1.073 | 0.548 | 0 |
| reduce | 184 | 182 | 0 (0) | 0 | 0.999 | 1.008 | 0.990 | 0.871 | 0 |
| rmsnorm | 12 | 0 | 12 (12) | 0 | - | - | - | - | 0 |
| softmax | 95 | 94 | 1 (1) | 0 | 1.000 | 1.574 | 0.288 | 0.306 | 0 |

| source | old PASS | v2 PASS | real speedup |
|---|---|---|---|
| agent_eval_fbv1 | 16 | 15 | 0 |
| agent_eval_fbv2 | 47 | 46 | 0 |
| grpo_v1 | 205 | 199 | 0 |
| grpo_v2 | 120 | 112 | 0 |

(A kernel seen in several sources is counted in each.)

## What changed (v1 -> v2)

| | v1 (upstream bench.py @78435821) | v2 |
|---|---|---|
| reference | reference.py in the input dtype (fp16/bf16) | same reference.py on inputs upcast to fp64 (norms, softmax, CE, RoPE, reduce) or fp32 with TF32 off (matmul, fused_mlp, attention) |
| tolerance | per-dtype atol/rtol; x10 relaxed for stability cases | same per-dtype atol/rtol, never relaxed, applied vs the golden; rtol floored at 2u of the dtype (bf16: 7.8e-3), because upstream bf16 rtol 2e-3 (layernorm/softmax/RoPE) is below bf16 unit roundoff 3.9e-3 and fails a correctly rounded output (v2.0 sanity run: golden_cast FAILed layernorm/RoPE bf16) |
| non-finite | PASS if kernel and reference are both NaN/Inf | FAIL on any non-finite where the golden is finite; golden must be finite and representable in the dtype |
| stability inputs | first dtype only, 'small' size, every float input x60000 / x1e-6 / mixed 1e3,1e-3 (weights too) | per-type magnitude cases whose true output is representable but whose fp16 intermediates overflow/underflow/stagnate, fp16 and bf16, plus large-N shapes (below) |
| inputs | seed 42 every run; layernorm weight=1, bias=0; RoPE cos/sin ~ randn | fresh os.urandom seed per evaluation; weight, bias ~ randn; cos/sin of real angles |
| cache | modal.Dict keyed on code (201/271 best-PASS timings were replays) | none |
| timing | triton do_bench, warm L2, kernel vs eager only | CUDA events, 256 MB L2 flush before each rep, 10 warmup, 60 reps, implementations interleaved rep-by-rep; median, CV, paired starter/kernel ratio with bootstrap 95% CI; baselines eager, starter, torch.compile(reference) |
| timing shapes | 'large', first dtype | 'large' (= v1) + one LLM-sized shape, first dtype |

Adversarial cases (see `ADVERSARIAL` in bench_v2_core.py): matmul uniform[0,1) K=8192 and randn x8 K=4096; softmax / CE logits x100, softmax N=131072, CE vocab=128256; layernorm x300, mean 1000 (+randn), N=65536; rmsnorm row RMS 8 and 300 at N=4096 (threshold sqrt(65504/4096)=4), x1e-3, per-row 10^U(-2,2.5), RMS 2 at N=65536; attention Q,K x4 and x40 (raw q.k up to ~1e5), seq 8192; fused_mlp gate/up std ~150 (|silu(g)*u| > 65504 for a few % of elements, output std ~1e3); RoPE x x1000, seq 16384; reduce uniform[0,1) N=32768 (sum ~16k), N=2^20. Plus zeros/constant rows.

## Bug audit: does the upstream reference itself fail against the fp32/fp64 golden?

Measured by evaluating reference.py in the input dtype on every v2 case (from the `golden_cast` sanity jobs).

| kernel type | low-precision intermediate bug | cases where upstream ref fails / total | failing cases | mechanism |
|---|---|---|---|---|
| matmul | YES (mild) | 10/41 | big_scale_K4096/bfloat16, big_scale_K4096/float16, edge/edge_1023/bfloat16, edge/edge_1023/float16, sweep/deep_k/bfloat16, sweep/deep_k/float16, sweep/large/bfloat16, sweep/large/float16, sweep/medium/bfloat16, sweep/medium/float16 | mild: torch.matmul with PyTorch's default allow_fp16/bf16_reduced_precision_reduction=True lets cuBLAS reduce split-K partials in fp16/bf16; errors of several output ulps at K>=1024 exceed upstream tol, so v1 FAILs a correctly rounded matmul (golden_cast) |
| softmax | no | 0/34 | - | none: F.softmax upcasts to fp32 internally (max-subtracted) |
| layernorm | no | 0/33 | - | none in the reference (fp32 Welford); but upstream bf16 rtol 2e-3 < bf16 unit roundoff, so v1 FAILs the correctly rounded output (and the starter); weight=1/bias=0 in sweep |
| rmsnorm | YES | 2/20 | rms_300_N4096/float16, row_mixed_scale/float16 | `x ** 2` materialised in fp16: overflows elementwise at |x|>256 -> mean=inf -> output rows = 0 (the mean itself accumulates in fp32, so RMS 8 is fine for the reference). fp16-accumulating kernels (GRPO) overflow the row SUM already at RMS > sqrt(65504/N) = 4 at N=4096 |
| flash_attention | YES | 4/25 | qk_x4/bfloat16, qk_x4/float16, qk_x40_overflow/bfloat16, qk_x40_overflow/float16 | `Q @ K^T` materialised in fp16/bf16 BEFORE `* sm_scale`: raw scores are rounded at their unscaled magnitude (and overflow when |q.k| > 65504), P is rounded before P@V; errors up to 1.9 (fp16) at Q,K x40 |
| fused_mlp | YES | 5/27 | big_act/bfloat16, big_act/float16, sweep/llm_13b/bfloat16, sweep/llm_7b/bfloat16, sweep/xlarge/bfloat16 | gate, up and silu(gate)*up materialised in fp16/bf16 (|silu(g)*u| > 65504 -> inf); bf16 intermediates already exceed tol on the xlarge/llm sweep shapes |
| cross_entropy | no | 0/28 | - | none: F.cross_entropy log-softmax + NLL accumulate in fp32 |
| rotary_embedding | YES | 16/28 | big_x_x1000/bfloat16, big_x_x1000/float16, long_16384/float16, edge/edge_1023/bfloat16, edge/edge_127/bfloat16, sweep/large/bfloat16, sweep/large/float16, sweep/llm_13b/bfloat16, sweep/llm_13b/float16, sweep/llm_7b/bfloat16, sweep/llm_7b/float16, sweep/medium/bfloat16, sweep/small/bfloat16, sweep/tiny/bfloat16, sweep/xlarge/bfloat16, sweep/xlarge/float16 | x1*cos, x2*sin and their difference each rounded to fp16/bf16 (3 roundings + cancellation): the reference's own error exceeds upstream tol (fp16 atol 1e-3) on normal inputs, which is why v1 FAILs the fp32-computing starter at smoke |
| reduce | no | 0/15 | - | none: x.sum accumulates fp16/bf16 in fp32 |

Other v1 harness defects (by reading bench.py): (1) stability stage PASSes when kernel and reference are both NaN/Inf, and near_max multiplies every float input (weights included) by 60000, which overflows fp16 inputs themselves; (2) stability/determinism/edge stages use only the first dtype; (3) `--quick` skips stages 3-5; (4) layernorm weight=ones/bias=zeros in smoke/sweep/edge, so a kernel that ignores weight/bias passes `--quick` (the full bench only catches it because the stability transforms also rescale the weights); (5) fixed seed 42 + result cache; (6) speedup only vs eager PyTorch, so any fusion looks like a win; (7) atol 1e-3 on softmax is larger than every probability at 4096+ columns (a zero output passes at those shapes; v2 keeps upstream tolerances, so this is only caught by the small shapes and the zeros control).

## Sanity table (L4)

Controls: `golden_cast` = reference on fp32/fp64-upcast inputs, rounded once to the input dtype (the best any kernel can do); `upstream_ref_native` = reference.py itself in the input dtype; `zeros`; `perturbed_5pct` = golden x1.05. v1 column: upstream bench.py full mode run in the v2 container (no cache), starters from `results/stage1_baselines.jsonl`, fp32acc/grpo_best from `results/v3/rmsnorm_remeasure.md`. Starter failures are genuine: matmul uses TF32 for fp32 inputs (err ~4e-2 vs tol 1e-4), flash_attention needs 128 KB smem at head_dim=128 (L4 limit 99 KB), fused_mlp calls `tl.math.tanh` (absent in Triton 3.2). v1 fails the same three.

All controls with a definite expectation behaved as expected: **True**.

| kernel | expected | v2 | v1 full bench | as expected | v2 failure (first) |
|---|---|---|---|---|---|
| matmul:starter | PASS if starter is correct | FAIL | FAIL | (info) | sweep/tiny/float32: 15672/16384 elems out of tol (atol=0.0001, rtol=0.0001), max_abs_err=4.190e-02, nonfinite=0 |
| matmul:golden_cast | PASS (tolerance calibration) | PASS | FAIL | yes |  |
| matmul:upstream_ref_native | FAIL iff reference has a low-precision bug | FAIL | PASS (is the reference) | (info) | sweep/medium/float16: 288/1048576 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=9.568e-02, nonfinite=0 |
| matmul:zeros | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 16371/16384 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=4.664e+01, nonfinite=0, zero_rows=128 |
| matmul:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 16093/16384 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=2.329e+00, nonfinite=0 |
| softmax:starter | PASS if starter is correct | PASS | PASS | (info) |  |
| softmax:golden_cast | PASS (tolerance calibration) | PASS | PASS | yes |  |
| softmax:upstream_ref_native | FAIL iff reference has a low-precision bug | PASS | PASS (is the reference) | (info) |  |
| softmax:zeros | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 3838/4096 elems out of tol (atol=0.001, rtol=0.001), max_abs_err=1.155e-01, nonfinite=0, zero_rows=32 |
| softmax:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 287/4096 elems out of tol (atol=0.001, rtol=0.001), max_abs_err=1.260e-02, nonfinite=0 |
| layernorm:starter | PASS if starter is correct | PASS | FAIL | (info) |  |
| layernorm:golden_cast | PASS (tolerance calibration) | PASS | FAIL | yes |  |
| layernorm:upstream_ref_native | FAIL iff reference has a low-precision bug | PASS | PASS (is the reference) | (info) |  |
| layernorm:zeros | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 4094/4096 elems out of tol (atol=0.001, rtol=0.001), max_abs_err=7.772e+00, nonfinite=0, zero_rows=32 |
| layernorm:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 4050/4096 elems out of tol (atol=0.001, rtol=0.001), max_abs_err=3.684e-01, nonfinite=0 |
| layernorm:ignores_weight_bias | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 4092/4096 elems out of tol (atol=0.001, rtol=0.001), max_abs_err=9.079e+00, nonfinite=0 |
| rmsnorm:starter | PASS if starter is correct | PASS | FAIL | (info) |  |
| rmsnorm:golden_cast | PASS (tolerance calibration) | PASS | FAIL | yes |  |
| rmsnorm:upstream_ref_native | FAIL iff reference has a low-precision bug | FAIL | PASS (is the reference) | (info) | adversarial/rms_300_N4096/float16: 4037493/4194304 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=1.379e+01, nonfinite=0, zero_rows=1024 |
| rmsnorm:zeros | FAIL | FAIL | FAIL | yes | sweep/small/float16: 759989/786432 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=1.183e+01, nonfinite=0, zero_rows=1024 |
| rmsnorm:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/small/float16: 492273/786432 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=6.681e-01, nonfinite=0 |
| rmsnorm:fp32acc | PASS | PASS | FAIL | yes |  |
| rmsnorm:grpo_best | FAIL | FAIL | PASS | yes | adversarial/rms_8_N4096/float16: 4038663/4194304 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=1.342e+01, nonfinite=0, zero_rows=1024 |
| flash_attention:starter | PASS if starter is correct | FAIL | FAIL | (info) | sweep/gqa/float16: EXC OutOfResources: out of resource: shared memory, Required: 131072, Hardware limit: 101376. Reducing block sizes or `num_stages` may help. |
| flash_attention:golden_cast | PASS (tolerance calibration) | PASS | FAIL | yes |  |
| flash_attention:upstream_ref_native | FAIL iff reference has a low-precision bug | FAIL | PASS (is the reference) | (info) | adversarial/qk_x4/float16: 681/524288 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=3.052e-02, nonfinite=0 |
| flash_attention:zeros | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 15930/16384 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=2.543e+00, nonfinite=0 |
| flash_attention:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 6158/16384 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=1.367e-01, nonfinite=0 |
| fused_mlp:starter | PASS if starter is correct | CRASH | FAIL | (info) | EXC AttributeError: module 'triton.language.math' has no attribute 'tanh' |
| fused_mlp:golden_cast | PASS (tolerance calibration) | PASS | FAIL | yes |  |
| fused_mlp:upstream_ref_native | FAIL iff reference has a low-precision bug | FAIL | PASS (is the reference) | (info) | sweep/xlarge/bfloat16: 5926/16777216 elems out of tol (atol=0.02, rtol=0.02), max_abs_err=5.020e-02, nonfinite=0 |
| fused_mlp:zeros | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 812/4096 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=3.312e-02, nonfinite=0, zero_rows=32 |
| fused_mlp:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/tiny/float32: 3319/4096 elems out of tol (atol=0.0001, rtol=0.0001), max_abs_err=2.287e-03, nonfinite=0 |
| cross_entropy:starter | PASS if starter is correct | PASS | PASS | (info) |  |
| cross_entropy:golden_cast | PASS (tolerance calibration) | PASS | PASS | yes |  |
| cross_entropy:upstream_ref_native | FAIL iff reference has a low-precision bug | PASS | PASS (is the reference) | (info) |  |
| cross_entropy:zeros | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 1/1 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=5.854e+00, nonfinite=0 |
| cross_entropy:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 1/1 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=3.009e-01, nonfinite=0 |
| rotary_embedding:starter | PASS if starter is correct | PASS | FAIL | (info) |  |
| rotary_embedding:golden_cast | PASS (tolerance calibration) | PASS | FAIL | yes |  |
| rotary_embedding:upstream_ref_native | FAIL iff reference has a low-precision bug | FAIL | PASS (is the reference) | (info) | sweep/tiny/bfloat16: 52/16384 elems out of tol (atol=0.002, rtol=0.0078125), max_abs_err=2.161e-02, nonfinite=0 |
| rotary_embedding:zeros | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 16373/16384 elems out of tol (atol=0.001, rtol=0.001), max_abs_err=4.242e+00, nonfinite=0 |
| rotary_embedding:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/tiny/float16: 16099/16384 elems out of tol (atol=0.001, rtol=0.001), max_abs_err=2.175e-01, nonfinite=0 |
| reduce:starter | PASS if starter is correct | PASS | PASS | (info) |  |
| reduce:golden_cast | PASS (tolerance calibration) | PASS | PASS | yes |  |
| reduce:upstream_ref_native | FAIL iff reference has a low-precision bug | PASS | PASS (is the reference) | (info) |  |
| reduce:zeros | FAIL | FAIL | FAIL | yes | sweep/small/float16: 1024/1024 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=1.279e+02, nonfinite=0 |
| reduce:perturbed_5pct | FAIL | FAIL | FAIL | yes | sweep/small/float16: 1017/1024 elems out of tol (atol=0.01, rtol=0.01), max_abs_err=5.523e+00, nonfinite=0 |

## Baselines (L4, fp16, median of 3 fresh-seed repeats)

| kernel type | shape | eager us | starter us (CV) | torch.compile us (CV) | starter vs eager | compile/starter latency |
|---|---|---|---|---|---|---|
| matmul | large | 280.6 | 327.7 (0.007) | 280.6 (0.008) | 0.857 | 0.857 |
| matmul | llm_qkv | 287.7 | 353.3 (0.008) | 288.8 (0.010) | 0.816 | 0.816 |
| softmax | large | 291.8 | 287.7 (0.029) | 288.8 (0.028) | 1.014 | 1.007 |
| softmax | vocab | 3524.1 | 42735.1 (0.005) | 3969.0 (0.003) | 0.083 | 0.093 |
| layernorm | large | 158.7 | 150.5 (0.018) | 152.6 (0.013) | 1.055 | 1.014 |
| layernorm | llm_7b | 297.0 | 290.8 (0.010) | 289.8 (0.007) | 1.021 | 0.996 |
| rmsnorm | large | 847.9 | 293.9 (0.010) | 294.9 (0.009) | 2.885 | 1.000 |
| rmsnorm | medium | 144.4 | 84.0 (0.028) | 85.0 (0.024) | 1.750 | 1.006 |
| flash_attention | large | 5898.2 | 399.4 (0.025) | 2475.0 (0.004) | 14.793 | 6.198 |
| flash_attention | xlarge | 26706.9 | 1414.7 (0.033) | 9133.1 (0.002) | 18.832 | 6.454 |
| cross_entropy | large | 2368.5 | 1166.3 (0.003) | 1175.6 (0.001) | 2.031 | 1.008 |
| cross_entropy | gpt2 | 3651.6 | 6507.0 (0.006) | 1943.6 (0.004) | 0.561 | 0.299 |
| rotary_embedding | large | 462.8 | 223.7 (0.007) | 153.6 (0.013) | 2.064 | 0.686 |
| rotary_embedding | llm_7b | 472.1 | 225.3 (0.007) | 154.6 (0.015) | 2.094 | 0.687 |
| reduce | large | 654.3 | 663.6 (0.002) | 571.4 (0.002) | 0.986 | 0.860 |
| reduce | wide | 361.5 | 363.5 (0.002) | 321.5 (0.004) | 0.994 | 0.885 |

vs torch.compile for a candidate = paired (starter/kernel) ratio x baseline (compile/starter) ratio, so it inherits cross-container drift (a few %).

## Cost (estimate)

Summed per-job wall time on L4: 1.49 h (rescore + baselines + final sanity). Adding the v2.0 sanity pass, 27 v1 full-bench control runs, 2 hung jobs (420 s each) and container start-up, roughly 2x that in billed container time: ~3.0 L4-h x $1.25/h (GPU+4 CPU+32 GB list price) = ~$3.7.

