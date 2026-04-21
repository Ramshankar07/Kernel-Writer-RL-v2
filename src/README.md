# AutoKernel RLVR

Train Qwen2.5-Coder-7B to drive AutoKernel's edit loop using verl GRPO
on SkyPilot. The agent proposes a full `kernel.py`, calls a verifier
(AutoKernel's `bench.py` running on a spot GPU), reads the metrics,
revises. Reward = log2(best_speedup) if any PASS in the trajectory, else 0.

The project supports two hardware targets: **NVIDIA** (H100 trainer, Triton
kernels) and **AMD MI300X** (MI300X trainer, HIP/Triton-ROCm kernels).
See `autokernel-rlvr/README.md` for the full runbook and AMD details.

## Architecture

```
┌──────────────────────┐           ┌──────────────────────┐
│ Trainer cluster (A)  │  HTTP    │ Bench queue service   │  HTTP
│ 8× H100  — or —      │  ───────►│ FastAPI + Redis (CPU) │ ◄─────┐
│ 8× MI300X on-demand  │ ◄─────── │  (hardware-agnostic)  │       │
└──────────────────────┘  result   └──────────────────────┘       │
                                                                   │ pull job,
                                              ┌────────────────────┴───────────┐
                                              │ Bench workers (C)              │
                                              │ N× A10/L4/4090 spot  (NVIDIA)  │
                                              │  — or —                        │
                                              │ N× MI300X spot       (AMD)     │
                                              └────────────────────────────────┘
```

Three SkyPilot clusters, loosely coupled via HTTP. Only the trainer is
on-demand; everything else is spot. The bench queue is shared between both
hardware paths.

## Files

    autokernel-rlvr/hardware/
      mi300x_specs.py             MI300X peak compute, memory, topology constants

    autokernel-rlvr/skypilot/
      trainer.yaml                8×H100 verl GRPO cluster  (NVIDIA)
      trainer_amd.yaml            8×MI300X verl GRPO cluster  (AMD)
      bench_server.yaml           CPU cluster running FastAPI+Redis  (shared)
      bench_worker.yaml           spot GPU pool — NVIDIA
      bench_worker_amd.yaml       spot GPU pool — AMD MI300X

    autokernel-rlvr/bench_server/app.py     queue with content-hash reward cache
    autokernel-rlvr/bench_worker/worker.py  NVIDIA bench worker
    autokernel-rlvr/bench_worker/worker_amd.py  AMD bench worker (ROCm env, 300s timeout)

    autokernel-rlvr/agent_loop/tools.yaml          verl tool registration  (shared)
    autokernel-rlvr/agent_loop/autokernel_loop.py  BenchHTTPTool + custom AgentLoop  (shared)
    autokernel-rlvr/agent_loop/reward.py           NVIDIA reward: log2(speedup), clip 3.0
    autokernel-rlvr/agent_loop/reward_amd.py       AMD reward: per-kernel clip (3.0–4.0)

    autokernel-rlvr/data/build_dataset.py      NVIDIA dataset builder
    autokernel-rlvr/data/build_amd_dataset.py  AMD dataset builder (6 single-GPU MI300X kernels)

## Run order

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
# watch them come online:
sky logs bench-workers --tail 50
# expect one "worker online" log per node
```

### 3. Smoke-test the queue end-to-end

Before burning trainer $$, verify the queue actually works:

```bash
# From your laptop, post a known-good kernel (the starter) and confirm
# you get a PASS reward back.
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
    --env BENCH_URL=$BENCH_URL --num-nodes 8 -y   # was 4
```

Spot preemption is fine — the queue TTL and idempotent jobs handle it.

### 6. Teardown

```bash
sky down trainer         # stop trainer first (saves $$)
sky down bench-workers
sky down bench-queue
```

Checkpoints persist in the mounted bucket (`$CHECKPOINT_BUCKET`).

## Cost model

Rough numbers at current spot pricing, per hour of wall-clock training:

| Component          | Hardware               | $/hr  | Notes                     |
|--------------------|------------------------|-------|---------------------------|
| Trainer            | 8× H100 on-demand      | $25   | Biggest line item         |
| Bench queue        | 8-cpu CPU node         | $0.20 | Negligible                |
| 4× bench workers   | 4× A10 spot            | $1.50 | Scale with queue depth    |
| **Total**          |                        | **~$27/hr** |                     |

If queue depth is consistently low, you're overprovisioned on trainer
rollouts — bump `rollout.n` or add more prompts. If queue depth is
consistently high, trainer is waiting on rewards — add workers or cache
harder (see tuning below).

## Tuning knobs worth actually using

**`GROUP_SIZE` (rollouts per prompt, default 8).** GRPO's core knob.
Smaller = cheaper but noisier advantage estimates. 8 is a reasonable
default; drop to 4 if you're GPU-poor, push to 16 if you can afford it.

**`MAX_TURNS` (edit budget, default 20).** Longer episodes = richer
credit assignment but massive context bloat. If you're seeing OOM on
long rollouts, drop to 12–16.

**Reward clip (`SPEEDUP_CLIP_HIGH = 3` in reward.py).** Anything above
log2(8x) = 3 gets clipped. This prevents a lucky 15× on one kernel from
dominating the advantage computation. Raise only if you observe the
policy consistently bumping the ceiling.

**Cache TTL.** `bench_server/app.py:CACHE_TTL` — 7 days. Aggressive, but
kernel code is deterministic, so same code → same reward. If you're
paying for storage, lower it. If the cache hit rate is low (<20%),
check whether the policy is producing meaningfully different code each
step or just shuffling whitespace.

## Known sharp edges

1. **verl API drift.** The `ToolAgentLoop` import path may differ
   between verl 0.5/0.6/0.7. If you get an ImportError, grep the
   installed verl for `class ToolAgentLoop` and adjust. The agent-loop
   doc at https://verl.readthedocs.io/en/latest/advance/agent_loop.html
   is authoritative.

2. **bench.py kernel-type argument.** AutoKernel's `bench.py` takes its
   target from the `kernel.py` currently in the repo, not a CLI flag.
   The `--kernel-type` in `worker.py` is defensive — if `bench.py`
   doesn't accept it, strip that arg. Test step 3 above before training.

3. **vLLM memory sharing with training.** With `gpu_memory_utilization=0.6`
   on 8×H100, you should have room. If you OOM during rollout, drop to
   0.5 or enable verl's `enable_prefix_caching=False`.

4. **Reward credit assignment.** Terminal reward means the policy can't
   distinguish "which edit helped". GRPO's group-relative advantage does
   some of this work across trajectories. If you need finer credit,
   flip `USE_STEP_SHAPING=True` in `reward.py` — but be wary of the
   policy gaming the shaping by producing many trivially-correct kernels.

5. **Episode termination.** With `max_turns=20` but no early-stop, every
   episode runs to full budget even after the policy finds a great
   kernel. That's wasted rollout compute. Post-v1, add an early-stop
   when `speedup_vs_pytorch > 2.0 && turns >= 5`.

## What v2 should look like

- Separate rollout cluster (cheaper GPUs for generation) via verl's
  server-based async rollout. Current YAML has rollout on the trainer.
- Kernel fusion tasks (e.g. "fuse layernorm + matmul"), not just single kernels.
- Distill the agent into a smaller model (3B) using trajectory traces.
- Elastic worker autoscaling via `sky serve` with a queue-depth policy.
