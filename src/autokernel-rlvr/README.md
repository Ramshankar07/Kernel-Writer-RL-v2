# AutoKernel RLVR

Train Qwen2.5-Coder-7B to drive AutoKernel's edit loop using verl GRPO
on SkyPilot. The agent proposes a full `kernel.py`, calls a verifier
(AutoKernel's `bench.py` running on a spot GPU), reads the metrics,
revises. Reward = log2(best_speedup) if any PASS in the trajectory, else 0.

The project now supports two hardware targets: the original **NVIDIA path**
(H100 trainer + A10/L4/4090 bench workers, Triton kernels) and a new
**AMD MI300X path** (MI300X trainer + MI300X bench workers, HIP/Triton-ROCm
kernels from the AMD competition dataset).

## Architecture

```
┌──────────────────────┐           ┌──────────────────────┐
│ Trainer cluster (A)  │  HTTP    │ Bench queue service   │  HTTP
│ 8× H100 on-demand    │  ───────►│ FastAPI + Redis (CPU) │ ◄─────┐
│  — or —              │ ◄─────── │  (hardware-agnostic)  │       │
│ 8× MI300X on-demand  │  result   └──────────────────────┘       │
└──────────────────────┘                                           │ pull job,
                                              ┌────────────────────┴───────────┐
                                              │ Bench workers (C)              │
                                              │ N× A10/L4/4090 spot  (NVIDIA)  │
                                              │  — or —                        │
                                              │ N× MI300X spot       (AMD)     │
                                              │ run bench.py on kernel.py      │
                                              └────────────────────────────────┘
```

Three SkyPilot clusters, loosely coupled via HTTP. Only the trainer is
on-demand; everything else is spot. The bench queue service is the same
for both hardware targets — it doesn't care what GPU the workers run on.

## Files

    hardware/
      mi300x_specs.py           MI300X peak compute, memory, topology constants

    skypilot/
      trainer.yaml              8×H100 verl GRPO cluster  (NVIDIA)
      trainer_amd.yaml          8×MI300X verl GRPO cluster  (AMD)
      bench_server.yaml         CPU cluster running FastAPI+Redis  (shared)
      bench_worker.yaml         spot GPU pool — NVIDIA
      bench_worker_amd.yaml     spot GPU pool — AMD MI300X

    bench_server/app.py         queue with content-hash reward cache
    bench_worker/worker.py      NVIDIA bench worker (CUDA_VISIBLE_DEVICES)
    bench_worker/worker_amd.py  AMD bench worker (ROCR_VISIBLE_DEVICES, 300s timeout)

    agent_loop/tools.yaml          verl tool registration
    agent_loop/autokernel_loop.py  BenchHTTPTool + custom AgentLoop  (shared)
    agent_loop/reward.py           NVIDIA reward: log2(speedup), clip at 3.0
    agent_loop/reward_amd.py       AMD reward: same structure, higher clip for FP8/MXFP4

    data/build_dataset.py       NVIDIA dataset: matmul, softmax, layernorm, etc.
    data/build_amd_dataset.py   AMD dataset: fp8-gemm, moe, mla-decode, all2all, …

---

## AMD MI300X

### Why AMD?

The AMD competition dataset (GPUMODE/kernelbot-data on HuggingFace) has over
1.1 million submissions for MI300X-specific kernels, including three large
problem sets: `amd-mxfp4-mm` (problem 763), `amd-moe-mxfp4` (764), and
`amd-mixed-mla` (765). The MI300X is a meaningfully different chip from the
H100 — 192 GB of HBM3, 5.3 TB/s memory bandwidth, 2614.9 TFLOPs of FP8
compute, and a 64-thread wavefront instead of NVIDIA's 32-thread warp. Kernels
that are fast on CUDA aren't necessarily fast here, and the model needs to learn
that difference.

### Target kernels

The nine AMD competition problem types covered by `build_amd_dataset.py`:

| Kernel | Type | Key challenge |
|---|---|---|
| `fp8-gemm` | Compute-bound | MFMA intrinsics, FP8 tile layout |
| `moe` | Mixed | Expert routing + compute overlap |
| `mla-decode` | Memory-bound | KV cache with low-rank compression |
| `all2all` | Communication | Infinity Fabric alignment |
| `gemm+reducescatter` | Compute + comms | Pipeline GEMM with collective |
| `allgather+gemm` | Comms + compute | Overlap gather with local GEMM |
| `mxfp4-mm` | Compute-bound | MX FP4 scale-factor handling |
| `moe-mxfp4` | Mixed | MoE routing with FP4 weights |
| `mixed-mla` | Mixed | Mixed-precision latent attention |

### MI300X hardware at a glance

These numbers are baked into `hardware/mi300x_specs.py` and imported
directly into the system prompt so the model always has them in context.

```
Compute (dense / with sparsity):
  FP8    :  2614.9 /  5229.8 TFLOPs
  BF16   :  1307.4 /  2614.9 TFLOPs
  FP16   :  1307.4 /  2614.9 TFLOPs
  TF32   :   653.7 /  1307.4 TFLOPs
  INT8   :  2614.9 /  5229.8 TOPS

Memory:
  Capacity  : 192 GB HBM3
  Bandwidth : 5.3 TB/s theoretical, ~4.7 TB/s sustained
  Interface : 8192-bit, up to 5.2 GT/s

Topology:
  Compute units : 304  (19,456 stream processors)
  Matrix cores  : 1216
  Infinity Cache: 256 MB last-level
  Scale-up links: 7× 128 GB/s Infinity Fabric
  Host I/O      : PCIe Gen5 ×16 (128 GB/s)
  TBP           : 750 W

Programming model:
  Wavefront size : 64 threads  (not 32 — remember this)
  LDS per CU     : 64 KB  (AMD's equivalent of CUDA shared memory)
```

### Reward differences (reward_amd.py)

The NVIDIA reward clips at log2(8×) = 3.0. On AMD, FP8 kernels benchmarked
against a BF16 PyTorch baseline can realistically exceed 8× just from the
precision difference — so we clip higher for the low-precision kernels:

- `fp8-gemm`, `mxfp4-mm`, `moe-mxfp4`, `mixed-mla` → clip at 4.0 (16×)
- `all2all`, `gemm+reducescatter`, `allgather+gemm` → clip at 3.5 (~11×)
- `moe`, `mla-decode` → clip at 3.0 (8×, same as NVIDIA default)

This stops one lucky FP8 kernel from drowning out the rest of the batch in
the GRPO advantage computation.

### What's different in the AMD worker (worker_amd.py)

Three things changed from `worker.py`:

1. `ROCR_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` are set instead of
   `CUDA_VISIBLE_DEVICES`. We also explicitly unset `CUDA_VISIBLE_DEVICES`
   so any stray CUDA calls fail loudly rather than silently routing to the
   wrong device.

2. Default timeout is 300 s instead of 180 s. ROCm's HIP-Clang JIT compiler
   is slower on first run than CUDA's nvrtc. On subsequent runs with the same
   binary cached, it's comparable, but the first compile of a new kernel
   architecture can take a while.

3. `bench.py` is invoked with `--backend rocm` so the AutoKernel AMD fork
   knows to target HIP instead of CUDA. Once the AMD branch is merged
   upstream this flag may become the default.

### Using competition data as starters

`build_amd_dataset.py` accepts an optional `--submissions-dir` pointing to
a local mirror of the `GPUMODE/kernelbot-data` parquets from HuggingFace.
When provided, it picks the highest-throughput passing submission for each
kernel type and uses that as the starter in the prompt, instead of the
AutoKernel repo's stub. This gives the model a genuinely strong baseline to
improve from rather than starting cold.

```bash
# Download the parquets from HuggingFace first (requires huggingface-cli)
huggingface-cli download GPUMODE/kernelbot-data --repo-type dataset \
    --local-dir ~/data/kernelbot-data

python3 data/build_amd_dataset.py \
    --autokernel-dir ./autokernel \
    --out ~/data/autokernel_amd \
    --max-turns 20 \
    --submissions-dir ~/data/kernelbot-data
```

---

## NVIDIA run order

### 1. Deploy the bench queue (5 min)

```bash
sky launch -c bench-queue skypilot/bench_server.yaml -y
BENCH_URL=http://$(sky status --ip bench-queue):8000
echo $BENCH_URL   # save this — both trainer and workers need it
curl $BENCH_URL/healthz   # expect "ok"
```

### 2. Launch bench workers (10 min to come up)

```bash
sky launch -c bench-workers skypilot/bench_worker.yaml \
    --env BENCH_URL=$BENCH_URL \
    --num-nodes 4 -y
sky logs bench-workers --tail 50
# expect one "worker online" log per node
```

### 3. Smoke-test the queue end-to-end

Before burning trainer $$, verify the queue actually works:

```bash
python3 - <<'PY'
import httpx, os, time, json
BENCH = os.environ["BENCH_URL"]
code = open("path/to/autokernel/kernels/matmul.py").read()
r = httpx.post(f"{BENCH}/bench", json={"kernel_type":"matmul","code":code}).json()
print("enqueued:", r)
if not r.get("cached"):
    job = r["job_id"]
    while True:
        p = httpx.get(f"{BENCH}/bench/{job}").json()
        if p["status"] == "done":
            print(json.dumps(p, indent=2)); break
        time.sleep(3)
PY
```

You should see a `PASS` with speedup ≈ 1.0 (starter kernel vs itself).

### 4. Launch the trainer

```bash
sky launch -c trainer skypilot/trainer.yaml \
    --secret WANDB_API_KEY --secret HF_TOKEN \
    --env BENCH_URL=$BENCH_URL -y

sky logs trainer                         # stream training logs
sky status --endpoint 8265 trainer       # Ray dashboard
```

Expect ~20 minutes before the first weight update (dataset build, model
download, first rollout batch). Watch `$BENCH_URL/stats` — queue depth
should rise as rollouts start, cache hit rate should climb past ~30%
within an hour as the policy revisits ideas.

### 5. Scale

If the queue is saturating (workers busy, depth > 50), add workers:

```bash
sky launch -c bench-workers skypilot/bench_worker.yaml \
    --env BENCH_URL=$BENCH_URL --num-nodes 8 -y
```

Spot preemption is fine — the queue TTL and idempotent jobs handle it.

### 6. Teardown

```bash
sky down trainer
sky down bench-workers
sky down bench-queue
```

Checkpoints persist in the mounted bucket (`$CHECKPOINT_BUCKET`).

---

## AMD MI300X run order

The bench queue is the same service — deploy it once and reuse it for both
NVIDIA and AMD workers simultaneously if you want to run both.

### 1. Deploy the bench queue (same as NVIDIA)

```bash
sky launch -c bench-queue skypilot/bench_server.yaml -y
BENCH_URL=http://$(sky status --ip bench-queue):8000
```

### 2. Build the ROCm verl image

The standard `verlai/verl` image is CUDA-only. You need to build the ROCm
variant from the verl repo before launching the AMD trainer:

```bash
git clone https://github.com/volcengine/verl.git
cd verl
docker build -f docker/rocm/Dockerfile -t verlai/verl:rocm6.1-verl0.6 .
docker push your-registry/verl:rocm6.1-verl0.6
```

Then update the `image_id` line in `skypilot/trainer_amd.yaml` to point to
your built image. This is the main setup step that has no shortcut.

### 3. Launch AMD bench workers

```bash
sky launch -c amd-workers skypilot/bench_worker_amd.yaml \
    --env BENCH_URL=$BENCH_URL \
    --num-nodes 4 -y
sky logs amd-workers --tail 50
```

Workers automatically set `ROCR_VISIBLE_DEVICES=0` and `PYTORCH_ROCM_ARCH=gfx942`
(MI300X ISA). The setup step checks out the `amd-hip` branch of AutoKernel
and runs `prepare.py --backend rocm` to pre-build baselines.

### 4. Smoke-test with an AMD kernel

```bash
python3 - <<'PY'
import httpx, os, time, json
BENCH = os.environ["BENCH_URL"]
code = open("path/to/autokernel/kernels/fp8-gemm.py").read()
r = httpx.post(f"{BENCH}/bench", json={"kernel_type":"fp8-gemm","code":code}).json()
print("enqueued:", r)
if not r.get("cached"):
    job = r["job_id"]
    while True:
        p = httpx.get(f"{BENCH}/bench/{job}").json()
        if p["status"] == "done":
            print(json.dumps(p, indent=2)); break
        time.sleep(5)   # ROCm JIT is slower — poll less aggressively
PY
```

### 5. Launch the AMD trainer

```bash
sky launch -c amd-trainer skypilot/trainer_amd.yaml \
    --secret WANDB_API_KEY --secret HF_TOKEN \
    --env BENCH_URL=$BENCH_URL -y

sky logs amd-trainer
sky status --endpoint 8265 amd-trainer
```

The dataset is built from `data/build_amd_dataset.py` during setup. Training
uses `agent_loop/reward_amd.py` for the per-kernel-class clipping.

### 6. Teardown

```bash
sky down amd-trainer
sky down amd-workers
sky down bench-queue
```

---

## Cost model

**NVIDIA path** — rough numbers at current spot pricing per hour of training:

| Component        | Hardware          | $/hr       | Notes                  |
|------------------|-------------------|------------|------------------------|
| Trainer          | 8× H100 on-demand | $25        | Biggest line item      |
| Bench queue      | 8-cpu CPU node    | $0.20      | Negligible             |
| 4× bench workers | 4× A10 spot       | $1.50      | Scale with queue depth |
| **Total**        |                   | **~$27/hr**|                        |

**AMD path** — MI300X pricing varies more by cloud provider. Lambda Labs
and Azure NDv5 are the main options at time of writing. Rough estimates:

| Component        | Hardware              | $/hr        | Notes                  |
|------------------|-----------------------|-------------|------------------------|
| Trainer          | 8× MI300X on-demand   | ~$30–40     | Provider-dependent     |
| Bench queue      | 8-cpu CPU node        | $0.20       | Same service           |
| 4× bench workers | 4× MI300X spot        | ~$10–15     | Spot availability thin |
| **Total**        |                       | **~$40–55/hr** |                     |

MI300X spot instances are less available than NVIDIA spot — if you're getting
preempted constantly, consider using on-demand workers for AMD or running
fewer nodes.

---

## Tuning knobs worth actually using

**`GROUP_SIZE` (rollouts per prompt, default 8).** GRPO's core knob.
Smaller = cheaper but noisier advantage estimates. 8 is a reasonable
default; drop to 4 if you're GPU-poor, push to 16 if you can afford it.

**`MAX_TURNS` (edit budget, default 20).** Longer episodes = richer
credit assignment but massive context bloat. If you're seeing OOM on
long rollouts, drop to 12–16. AMD rollouts tend to run longer because the
ROCm JIT means the model needs more turns before getting a PASS — 20 is
probably the right floor for AMD.

**Reward clip.** NVIDIA uses `SPEEDUP_CLIP_HIGH = 3.0` in `reward.py`.
AMD uses per-kernel clips in `reward_amd.py` — 4.0 for FP8/MXFP4 kernels,
3.5 for collectives, 3.0 for the rest. Raise a specific value if you see
the policy consistently butting against the ceiling for that kernel class.

**Cache TTL.** `bench_server/app.py:CACHE_TTL` — 7 days by default. Kernel
code is deterministic (same code → same result), so this is safe to leave
high. If cache hit rate is stuck below 20%, the policy is either producing
very diverse code (good) or shuffling whitespace (bad — check the diff).

---

## Known sharp edges

1. **verl API drift.** The `ToolAgentLoop` import path may differ between
   verl 0.5/0.6/0.7. If you get an ImportError, grep the installed verl for
   `class ToolAgentLoop` and adjust. The agent-loop docs at
   https://verl.readthedocs.io/en/latest/advance/agent_loop.html are
   authoritative.

2. **bench.py kernel-type argument.** AutoKernel's `bench.py` takes its
   target from the `kernel.py` in the repo. The `--kernel-type` flag in
   `worker.py` and `worker_amd.py` is defensive — if `bench.py` doesn't
   accept it, strip that arg. Always test step 3 above before training.

3. **AMD worker: `--backend rocm` flag.** This assumes the AutoKernel AMD
   fork exists and accepts that flag. If you're on a fork that doesn't have
   it yet, remove it from `worker_amd.py:cmd`. The `amd-hip` branch checkout
   in the worker setup YAML may also need to be changed to whatever the
   actual branch is called in the forked repo.

4. **ROCm verl image.** There's no published ROCm-enabled verl image right
   now. You need to build one from the verl repo's `docker/rocm/Dockerfile`.
   This is the single biggest setup hurdle for the AMD path — block a few
   hours for the first build.

5. **vLLM memory on MI300X.** `gpu_memory_utilization=0.6` is a safe
   starting point given 192 GB of HBM3. You can push it higher (0.7–0.75)
   on AMD since there's substantially more memory headroom than on H100.

6. **Reward credit assignment.** Terminal reward means the policy can't
   directly tell which edit helped. GRPO's group-relative advantage partially
   covers this. If you want finer signal, flip `USE_STEP_SHAPING=True` in
   either reward file — but be careful, the policy will learn to generate
   many trivially-correct slow kernels to farm the per-PASS bonus.

7. **Episode termination.** With `max_turns=20` and no early-stop, every
   episode runs to full budget even after the policy finds a great kernel.
   Post-v1 improvement: add an early-stop when
   `speedup_vs_pytorch > 2.0 && turns >= 5`.

---

## What's next

- **Software pattern extraction.** The competition submissions contain
  implicit signals about which compiler intrinsics and software patterns
  led to top scores (MFMA tile sizes, async pipeline depths, LDS layout
  choices). Extracting those patterns as structured features for data
  preprocessing is the next planned step — currently held.

- **Separate rollout cluster.** Cheaper GPUs for generation via verl's
  server-based async rollout. The current YAMLs keep rollout on the trainer
  node for simplicity.

- **Kernel fusion tasks.** Extend beyond single-kernel benchmarks to
  fused operations (e.g. layernorm + matmul, attention + softmax) which
  are where the real headroom lives.

- **Distillation.** Once you have strong trajectories, distill the agent
  into a smaller model (3B) using the trace data.

- **Elastic autoscaling.** Wrap the worker pool in `sky serve` with a
  queue-depth policy so it scales automatically rather than requiring
  manual `--num-nodes` adjustments.
