# SFT Phase 1 data pipeline (results/v3/sft/)

Plan: `results/v3/sft_plan.md` Phase 1. Research: `results/v3/sft_data_research.md`.
Stack: torch 2.6.0, Triton 3.2.0, L4/L40S. Policy: Qwen2.5-Coder-7B-Instruct (chat template).

## Candidate schema (one JSON object per line, `candidates_<source>.jsonl`)
| field | type | meaning |
|---|---|---|
| `id` | str | `<source>:<original row id or index>` (stable, unique) |
| `source` | str | `drkernel` \| `kernelbook` \| `inrepo` |
| `format` | str | `modelnew` (prompt = PyTorch `Model` module + `get_inputs`/`get_init_inputs`; target = full file defining `ModelNew` with Triton kernels) or `autokernel` (prompt = AutoKernel reference.py + starter; target = full kernel.py with `KERNEL_TYPE` + `kernel_fn`) |
| `reference` | str | PyTorch reference source (modelnew: full module file incl. get_inputs/get_init_inputs; autokernel: kernel_type name; the reference/starter files live in `results/autokernel_src` / `modal_app/starters_v2`) |
| `kernel_type` | str \| null | autokernel only: one of the 9 AutoKernel types |
| `target` | str | candidate Triton source (what the assistant will output) |
| `family` | list[str] | op-family tags (softmax, layernorm, rmsnorm, reduce, cross_entropy, rotary, matmul, attention, mlp, activation, conv, other, …) |
| `repair` | null \| object | for repair data: `{"failed_code": str, "feedback": str}` (the failing turn + real bench feedback that `target` fixes) |
| `meta` | object | anything source-specific (speedup claims, round index, license, revision sha) |
| `static` | object | static-filter verdicts, e.g. `{"parses": true, "jit_kernels": 2, "launched": true, "forbidden": []}` |

Verification adds `verify` = `{"verdict": "PASS"\|"FAIL"\|..., "cases": ..., "launch_seen": bool, "reason": str}` in `verified_<source>.jsonl`.

## Rules (all agents)
- Pinned revisions only: drkernel-coldstart-8k @ `cba0ef06a5b1e3c307b7acfa8b6acb7a46578105`, KernelBook @ `1576375bc92745b490e1cdf2fce01eba76d9f847` (MIT).
- Excluded: `ppbhatt500/kernelbook-triton-reasoning-traces` (KernelBench-derived), `siro1/*` (user decision 2026-09-30).
- Never tune on or train on KernelBench L1 or the val set (`results/v3/val_set/`, `results/v3/val_set.json`).
- Modal: ≤4 concurrent containers for verification; account cap 10 total. Phase 1 GPU budget ≈ $11.
- Branch `modal-replication`; do not commit, push, or switch branches. Don't edit existing harness files; add new files.
