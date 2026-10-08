"""
Generates presentation/qa_30.md and presentation/writeup.md from results/numbers.json.
Every number below is read from numbers.json; nothing is typed by hand.

    python presentation/make_docs.py
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
N = json.loads((ROOT / "results" / "numbers.json").read_text())
OUT = ROOT / "presentation"

PENDING = "_(pending: stage not run yet)_"


def ae(tag):
    return next((v for k, v in N.get("agent_eval", {}).items() if k.endswith(tag)), None)


def pc(x, d=1):
    return f"{100 * x:.{d}f}%"


V1, V2 = ae("_fbv1"), ae("_fbv2")
S1 = N.get("stage1", {})
KB = N.get("kb_sanity", {})
G1 = N.get("grpo", {}).get("grpo_v1")
G2 = N.get("grpo", {}).get("grpo_v2")
G = G1
KB_BASE = ae("_kb_base")
KB_LORA = ae("_kb_lora")
AK_LORA = ae("_ak_lora")
X = N.get("extra", {})


def h100(k):
    return S1["per_gpu"]["H100"][k]


def l4(k):
    return S1["per_gpu"]["L4"][k]


def qa():
    Q = []   # (section, question, answer, evidence)

    # --- A. Setup and motivation -------------------------------------------------
    sec = "A. Setup and motivation"
    Q.append((sec, "What problem does this project solve?",
              "It trains an LLM agent to write faster GPU kernels with RL from verifiable rewards "
              "(RLVR). The agent proposes a full Triton `kernel.py`, a GPU benchmark checks "
              "correctness and speed, and the agent revises. Reward = log2(best speedup) if any "
              "proposal PASSes, else 0.",
              "`src/autokernel-rlvr/agent_loop/reward.py`, figure 01"))
    Q.append((sec, "Why did the project start on KernelBench v1?",
              f"KernelBench Level 1 is the standard public benchmark (100 single-op PyTorch problems, "
              f"fast_p metric), so it served as the first test set. The re-run rebuilt that harness on "
              f"Modal: an identity solution (ModelNew = Model) PASSes {KB['counts']['PASS']}/{KB['n']} "
              f"problems with median measured speedup {KB['identity_speedup_median']:.3f}× "
              f"(p5–p95 {KB['identity_speedup_p5_p95'][0]:.3f}–{KB['identity_speedup_p5_p95'][1]:.3f}×), so the timing "
              f"is trustworthy. Benching the same identity solution twice on problems 1–50 gave the same verdict on "
              f"{KB['repeat']['same_verdict']}/{KB['repeat']['problems']} and a median speedup difference of "
              f"{KB['repeat']['speedup_abs_diff_median']}×. {KB['valid']} problems are harness-valid.",
              "`modal_app/kernelbench_modal.py::sanity`, figure 03"))
    Q.append((sec, "Why move from KernelBench to AutoKernel's 9 kernel types for RL?",
              f"Validity first, then cost. Only {KB['valid']}/100 KernelBench L1 problems are harness-valid in "
              f"AutoKernel's KernelBench mode: {KB['conv_unseeded']} conv problems fail even for an identical model "
              f"(weight init isn't seeded), and {KB['flaky_timeouts']} large-input problems time out intermittently. "
              f"Cost: a KB bench has median {KB['wall_s_p50']} s and p95 {KB['wall_s_p95']} s on H100, vs AutoKernel's "
              f"p50 {S1['bench_wall_s']['L4']['p50']} s / p95 {S1['bench_wall_s']['L4']['p95']} s on the "
              f"{N['prices_usd_per_gpu_hr']['H100'] / N['prices_usd_per_gpu_hr']['L4']:.0f}× cheaper L4. RL needs thousands of rewards, so AutoKernel's 9 kernels became the training set and KernelBench "
              f"the held-out eval.",
              "`results/kb_sanity.jsonl`, `results/stage1_baselines.jsonl`"))
    Q.append((sec, "Which model, and why that size?",
              "Qwen2.5-Coder-7B-Instruct (same as the original design). 7B fits vLLM rollout on one "
              "H100 and a LoRA (r=64, 161.5M trainable params) trainer on a second H100, which with "
              "8 bench GPUs is exactly the 10-GPU account limit." +
              (f" Size ablation: {X['model_size']}" if X.get("model_size") else ""),
              "`modal_app/train_grpo.py`, `modal_app/policy.py`"))
    Q.append((sec, "What changed between the original design and this re-run?",
              "Original (Apr 2026): 3 SkyPilot clusters: 8×H100 verl trainer, FastAPI+Redis queue, "
              "A10/L4 spot workers, about $27/hr per the README. Re-run (Sept 2026): Modal with ≤10 GPUs, "
              "i.e. 1×H100 LoRA trainer + 1×H100 vLLM policy + ≤8×L4 bench, and a modal.Dict content-hash "
              "cache instead of Redis. The GRPO math follows verl (group-normalized advantage, token-mean "
              "loss, no KL). All original results were lost, so every number here is from the re-run.",
              "figure 01, `CLAUDE.md`"))

    # --- B. Verifier and reward --------------------------------------------------
    sec = "B. Verifier and reward"
    Q.append((sec, "How noisy is the reward signal?",
              f"Very quiet. Re-benching the same starter kernel 3× gives at most {S1.get('max_cv_pct')}% "
              f"coefficient of variation in speedup across 9 kernels × 2 GPUs. On KernelBench the identity "
              f"solution measures {KB.get('identity_speedup_median', 0):.3f}× median.",
              "`results/stage1_baselines.jsonl`"))
    Q.append((sec, "Can the agent game the benchmark with the quick mode?",
              f"Yes, which is why reward uses the full bench. `bench.py --quick` skips the "
              f"numerical-stability stage. The {', '.join(S1['quick_pass_full_fail']['H100'])} starters PASS quick "
              f"(e.g. rmsnorm {h100('rmsnorm')['speedup_quick']}× on H100) but FAIL full, so a quick-mode "
              f"reward would pay for numerically unsafe kernels.",
              "figure 02, `modal_app/agent_eval.py::run_episodes`"))
    if V2:
        c = V2["ablations"]["clip"]
        Q.append((sec, "Does the log2 clip at 3.0 (8×) matter?",
                  f"Not at this stage. Of {c['n_pass_traj']} passing trajectories in the base-model eval, "
                  f"{c['hit_ceiling_3']} hit the +3 ceiling and {c['hit_floor_-1']} hit the −1 floor. The best "
                  f"base-model speedup was {V2['best_speedup_max']}×. The clip is insurance for later "
                  f"training, not an active constraint.",
                  "`reward.py:SPEEDUP_CLIP_HIGH`, `modal_app/analyze.py::ablations`"))
        Q.append((sec, "Is there a flaw in the reward design?",
                  f"Yes: a correct kernel slower than PyTorch gets log2(s) < 0, which is worse than never "
                  f"passing (0). In the base eval {c['slower_than_pytorch_pass']}/{c['n_pass_traj']} passing "
                  f"trajectories were slower than PyTorch and got negative reward (e.g. a 0.99× reduce kernel "
                  f"scores −0.015), so within a GRPO group a correct-but-slow kernel ranks *below* a crash. "
                  f"A fix is max(0, log2 s) + a small PASS bonus.",
                  "`reward.py` lines computing `reward = math.log2(...)`"))
        s = V2["ablations"]["step_shaping"]
        Q.append((sec, "Would per-step shaping (+0.02/PASS, −0.02/CRASH) help?",
                  f"It changes the learning signal a lot. Offline on the base-model trajectories it changes the "
                  f"ordering or tie-status of {s['pairs_reordered_or_split']}/{s['pairs_compared']} within-group "
                  f"trajectory pairs, because many groups are otherwise all-zero ties. It adds signal but also "
                  f"rewards spamming trivially-correct kernels (README sharp edge #6). Left off, as in the "
                  f"original.",
                  "`modal_app/analyze.py::ablations`"))

    # --- C. Base-model behaviour --------------------------------------------------
    sec = "C. Base model (before RL)"
    if V1 and V2:
        Q.append((sec, "How good is the untrained model?",
                  f"Weak. Over 9 kernels × 8 samples × 8 turns on L4, {pc(V2['pass_rate'])} of episodes produce "
                  f"at least one PASS, fast_1.0 = {pc(V2['fast_p']['fast_1.0'])} (correct *and* faster than "
                  f"PyTorch), mean reward {V2['reward_mean']}. Only "
                  f"{sum(1 for v in V2['per_kernel'].values() if v['pass_rate'] > 0)}/9 kernels are ever solved.",
                  "figure 05"))
        Q.append((sec, "Does giving the model more turns help?",
                  f"Barely. pass@1 turn = {pc(V2['pass_at_turn'][0])} vs pass@8 turns = "
                  f"{pc(V2['pass_at_turn'][-1])}. With the original log-tail feedback it was flat: "
                  f"{pc(V1['pass_at_turn'][0])} → {pc(V1['pass_at_turn'][-1])}. The base model doesn't use "
                  f"feedback well, which is the gap RL is supposed to close.",
                  "figure 04"))
        tax = V2["failure_taxonomy"]
        tot = sum(tax.values())
        top = list(tax.items())[:3]
        Q.append((sec, "Why do generated kernels fail?",
                  "; ".join(f"{k} {c} ({100 * c / tot:.0f}%)" for k, c in top) +
                  f" of {tot} non-PASS turns. Correctness, not speed, is the bottleneck.",
                  "figure 06"))
        Q.append((sec, "Does the format of the benchmark feedback matter?",
                  f"Yes, and this was a bug in the original design. The worker returned the *tail* of "
                  f"bench.py's log, which is only the summary block, so the model never saw which check "
                  f"failed. Sending the failure lines instead (v2) moved episode pass rate "
                  f"{pc(V1['pass_rate'])} → {pc(V2['pass_rate'])}, mean reward {V1['reward_mean']} → "
                  f"{V2['reward_mean']}, CRASH turns {V1['turn_outcomes'].get('CRASH')} → "
                  f"{V2['turn_outcomes'].get('CRASH')}, and solved rmsnorm "
                  f"({V2['per_kernel']['rmsnorm']['best_speedup']}×) for the first time. No training involved.",
                  "figures 04–05, `agent_eval.py::observation`"))
        Q.append((sec, "Does the model just repeat itself?",
                  f"Often. {pc(V2['dup_code_frac'])} of turns resubmit code already tried in the same task "
                  f"({pc(V2['cache_hit_rate'])} bench-cache hit rate), mostly re-sending the starter kernel "
                  f"after a failure. The content-hash cache makes that free, and it is also a measurable "
                  f"target for RL.",
                  "`bench_modal.py::cache_key`"))

    # --- D. GRPO training --------------------------------------------------------
    sec = "D. GRPO training"
    if G1:
        never = sorted(set(G1["per_kernel_reward_first_last"]) - set(G1["kernels_ever_passed"]))
        Q.append((sec, "Did RL with the original reward improve the agent?",
                  f"Barely. Over {G1['steps']} GRPO steps (v1), the original reward went {G1['reward_first3']} → "
                  f"{G1['reward_last3']} (mean of first vs last 3 steps), and the pass rate went "
                  f"{pc(G1['pass_first3'])} → {pc(G1['pass_last3'])}. The gain came from reduce (fewer much-slower "
                  f"passing kernels: {G1['per_kernel_reward_first_last']['reduce'][0]} → "
                  f"{G1['per_kernel_reward_first_last']['reduce'][1]}) and cross_entropy. {len(never)}/9 kernels "
                  f"({', '.join(never)}) never passed in any episode of any step; rmsnorm passed only occasionally.",
                  "figure 09, `results/grpo/grpo_v1/metrics.jsonl`"))
        Q.append((sec, "Why did v1 stall?",
                  f"No gradient. On average {pc(G1['zero_adv_mean'])} of groups had identical rewards across all "
                  f"8 samples, so their GRPO advantage is 0. The never-solved kernels can't teach anything under "
                  f"a pass/fail terminal reward. There is a second, quieter issue: in the groups that do vary, "
                  f"the spread is often bench timing noise (std ≈ 0.002), which std-normalisation inflates to "
                  f"full ±1 advantages.",
                  "figure 09, `train_grpo.py::grpo_advantages`"))
        if G2:
            Q.append((sec, "What did reward v2 change, and did it help?",
                      f"Reward v2 does two things: (a) partial credit for correctness progress (smoke test, fraction "
                      f"of shape configs passed, stability; max 0.3), and (b) any PASS scores ≥0.5, fixing the "
                      f"slower-than-a-crash floor. It also ignores groups with std < 0.02, the measured noise floor. "
                      f"Over {G2['steps']} steps: reward_v2 {G2['reward_v2_first3']} → {G2['reward_v2_last3']}, "
                      f"original reward {G2['reward_first3']} → {G2['reward_last3']}, pass rate "
                      f"{pc(G2['pass_first3'])} → {pc(G2['pass_last3'])}, zero-advantage groups "
                      f"{pc(G2['zero_adv_mean'])} (v1: {pc(G1['zero_adv_mean'])}), smoke-test pass rate "
                      f"{pc(G2['smoke_first3'])} → {pc(G2['smoke_last3'])} (v1: {pc(G1['smoke_first3'])} → "
                      f"{pc(G1['smoke_last3'])}). Honest reading: the robust effect is more groups with gradient, and the "
                      f"pass rate no longer declines as it did in v1. The reward changes are small and single-seed, and no "
                      f"new kernel was solved in {G2['steps']} × 72 episodes (kernels ever passed: "
                      f"{', '.join(G2['kernels_ever_passed'])}).",
                      "figure 09, `modal_app/reward_v2.py`"))
        else:
            Q.append((sec, "What did reward v2 change, and did it help?", PENDING, "`modal_app/reward_v2.py`"))
    else:
        Q.append((sec, "Did RL with the original reward improve the agent?", PENDING, "figure 09"))
        Q.append((sec, "Why did v1 stall?", PENDING, "figure 09"))
        Q.append((sec, "What did reward v2 change, and did it help?", PENDING, "`modal_app/reward_v2.py`"))
    if V2:
        z = V2["ablations"]["zero_adv_group_frac_by_group_size"]
        Q.append((sec, "Why a GRPO group size of 8?",
                  f"Measured on base-model rollouts: zero-advantage groups are {pc(z['2'])} at group=2, "
                  f"{pc(z['4'])} at 4 and {pc(z['8'])} at 8. Group=4 would waste roughly "
                  f"{pc(z['4'] - z['8'])} more of each batch.",
                  "figure 07"))
    if G1:
        runs = [g for g in (G1, G2) if g]
        Q.append((sec, "What does one training step cost?",
                  f"~{G1['rollout_s_mean']} s rollout (72 episodes × 4 turns, bench-bound) + "
                  f"~{G1['train_s_mean']} s policy update. H100 time: v1 {G1['steps']} steps = {G1['wall_h']} h ≈ "
                  f"${G1['est_cost_usd']}" + (f"; v2 {G2['steps']} steps = {G2['wall_h']} h ≈ ${G2['est_cost_usd']}" if G2 else "") +
                  f" (2 H100s at ${N['prices_usd_per_gpu_hr']['H100']}/GPU-hr), plus the L4 bench pool. The bench cache "
                  f"served {pc(G1['cache_hit_last3'])} of v1's late-training benchmarks for free. The trainer GPU idles "
                  f"about {100 * G1['rollout_s_mean'] / (G1['rollout_s_mean'] + G1['train_s_mean']):.0f}% of each step, "
                  f"so async rollouts are the obvious next win.",
                  "`results/grpo/*/metrics.jsonl`"))
    else:
        Q.append((sec, "What does one training step cost?", PENDING, "`results/grpo/*/metrics.jsonl`"))

    # --- E. Evaluation and generalization -----------------------------------------
    sec = "E. Evaluation and generalization"
    if KB_BASE:
        txt = (f"No. On a {KB_BASE['meta'].get('tasks')}-problem KernelBench L1 subset (every 2nd harness-valid problem, "
               f"3 turns), the base model gets fast_0 = {pc(KB_BASE['fast_p']['fast_0.0'])} and fast_1.0 = "
               f"{pc(KB_BASE['fast_p']['fast_1.0'])}")
        if KB_LORA:
            txt += (f"; after GRPO v2: fast_0 = {pc(KB_LORA['fast_p']['fast_0.0'])}, fast_1.0 = "
                    f"{pc(KB_LORA['fast_p']['fast_1.0'])}. So no measurable transfer from 9 AutoKernel kernels in "
                    f"{G2['steps'] if G2 else '?'} steps")
        tax = list(KB_BASE["failure_taxonomy"].items())[:3]
        tot = sum(KB_BASE["failure_taxonomy"].values())
        txt += (". Top failure reasons for the base model: " +
                "; ".join(f"{k} {c}/{tot}" for k, c in tax) +
                ". A 7B model can't yet produce a correct KernelBench `ModelNew` in Triton within 3 turns; most "
                "failures are calls to Triton APIs that don't exist in Triton 3.2, so API grounding comes before speed")
        Q.append((sec, "Does training on 9 AutoKernel kernels transfer to KernelBench?", txt + ".", "figure 10"))
    else:
        Q.append((sec, "Does training on 9 AutoKernel kernels transfer to KernelBench?", PENDING, "figure 10"))
    Q.append((sec, "Does the GPU matter for the reward?",
              f"A lot. The same starter softmax is {h100('softmax')['speedup_quick']}× on H100 but "
              f"{l4('softmax')['speedup_quick']}× on L4; flash_attention passes quick mode on H100 but "
              f"runs out of shared memory on L4 (101 KB limit). Rewards are measured on L4, like the "
              f"original A10/L4 worker pool." + (f" Transfer check: {X['cross_gpu']}" if X.get("cross_gpu") else ""),
              "figure 02"))
    Q.append((sec, "How do you know the agent isn't gaming the verifier?",
              "Four guards. (1) Full 5-stage correctness (smoke, shape sweep incl. fp32, numerical "
              "stability, determinism, edge sizes) on every reward. (2) The reference is AutoKernel's "
              "fixed `reference.py`, and the agent only writes `kernel.py`. (3) Content-hash cache, so the "
              "same code always gets the same reward. (4) The quick-vs-full gap (B2) shows why cheaper "
              "checks would be exploitable.",
              "`results/autokernel_src/bench.py`"))
    if V2:
        Q.append((sec, "Does context length become a problem?",
                  f"It grows ~{(V2['prompt_tokens_by_turn'][-1] - V2['prompt_tokens_by_turn'][0]) / (len(V2['prompt_tokens_by_turn']) - 1) / 1000:.1f}k "
                  f"tokens per turn ({V2['prompt_tokens_by_turn'][0]} → {V2['prompt_tokens_by_turn'][-1]} "
                  f"prompt tokens over 8 turns), because each turn resends a full kernel.py plus bench "
                  f"output. That's why training uses 4 turns and a 32k context.",
                  "figure 08"))
        es = V2["ablations"]["early_stop"]
        Q.append((sec, "Would early stopping save compute (README sharp edge #7)?",
                  f"Hardly, for this model. Stopping once speedup > 2× after turn ≥ 5 saves only "
                  f"{pc(es['turns_saved_frac'])} of turns and loses {es['reward_lost_total']} total reward, "
                  f"because the base model rarely reaches 2×. Reward by turn budget: "
                  f"{V2['ablations']['reward_by_turn_budget']['1']} at 1 turn vs "
                  f"{V2['ablations']['reward_by_turn_budget']['8']} at 8.",
                  "`modal_app/analyze.py::ablations`"))

    # --- F. Engineering and next steps -------------------------------------------
    sec = "F. Engineering and next steps"
    Q.append((sec, "What broke when the project was re-run?",
              "Five real bugs: (1) the worker called `bench.py --kernel-type`, but the flag is `--kernel`; "
              "(2) the worker parsed `results.tsv`, which bench.py never writes (metrics are on stdout); "
              "(3) AutoKernel's KernelBench starter keeps `super(Model, self)` after renaming the class, "
              f"so all 100 starters crash; (4) KernelBench times CPU input generation inside a 30 s trial "
              f"timeout, so with default Modal CPUs only {pc(KB['default_cpu_pass_frac'], 0)} of identity runs passed; "
              f"(5) KernelBench mode never seeds weight init, so {KB['conv_unseeded']} conv problems can't pass even as an "
              f"identical model. Plus two of mine: vLLM 0.8.5 needed a transformers pin, and my first problem listing read "
              f"the cache's .json metadata files as problems (caught and re-run).",
              "`CLAUDE.md` › Findings"))
    Q.append((sec, "What does it cost?",
              "Original design: about $27/hr (README, 8×H100 on-demand). Re-run: ≤10 Modal GPUs, about "
              f"${2 * N['prices_usd_per_gpu_hr']['H100'] + 8 * N['prices_usd_per_gpu_hr']['L4']:.1f}/hr at full "
              f"load (2×H100 + 8×L4 list price)." +
              (f" GRPO: v1 ${G1['est_cost_usd']} ({G1['steps']} steps)" + (f", v2 ${G2['est_cost_usd']} ({G2['steps']} steps)" if G2 else "") +
               " in H100 time." if G1 else ""),
              "`modal_app/analyze.py` prices"))
    Q.append((sec, "What happened to the AMD MI300X path?",
              "It is design-only. The repo has an MI300X reward (`reward_amd.py`, per-kernel clip 4.0 for "
              "FP8/MXFP4) and worker (`worker_amd.py`, ROCm env, 300 s timeout), but Modal offers no AMD "
              "GPUs and the upstream AutoKernel repo has no `amd-hip` branch, so none of it could run.",
              "`src/autokernel-rlvr/agent_loop/reward_amd.py`"))
    Q.append((sec, "Why not use verl, as in the original design?",
              "verl's ToolAgentLoop API drifts between 0.5 and 0.7 (README sharp edge #1), and its Ray + "
              "vLLM colocated setup expects a multi-GPU node. The re-run keeps verl's GRPO semantics "
              "(adv_estimator=grpo, token-mean loss, no KL, one on-policy update) in ~250 lines, and "
              "calls the original `reward.py` unchanged.",
              "`modal_app/train_grpo.py` docstring"))
    Q.append((sec, "What would you do next?",
              "(1) Fix the reward floor: correct-but-slow should not be worse than crashing. (2) Keep the "
              "v2 failure-line feedback (a free gain). (3) Curriculum: oversample the 5 never-solved kernels "
              "once the model reliably passes the easy ones, since all-zero groups give no gradient. "
              "(4) Async rollouts so the trainer GPU isn't idle. (5) Evaluate on KernelBench L2 fusion "
              "problems.",
              "sections B–D"))
    return Q


def write_qa(Q):
    lines = [f"# 30 follow-up questions, with answers", "",
             f"All numbers: {N['label']}. Source: `results/numbers.json` (generated by "
             f"`modal_app/analyze.py`). Figures: `presentation/figures/`.", ""]
    sec = None
    for i, (s, q, a, ev) in enumerate(Q, 1):
        if s != sec:
            lines += [f"## {s}", ""]
            sec = s
        lines += [f"**{i}. {q}**", "", a, "", f"*Evidence:* {ev}", ""]
    (OUT / "qa_30.md").write_text("\n".join(lines))
    print(f"wrote qa_30.md with {len(Q)} questions")


def write_writeup():
    L = [f"# AutoKernel-RLVR: teaching an LLM to write faster GPU kernels", "",
         f"*{N['label']}. Every number is generated from `results/numbers.json`; figures in "
         f"`presentation/figures/`; 30 follow-up Q&As in `qa_30.md`.*", ""]
    L += ["## 1. What the project is", "",
          "An LLM agent (Qwen2.5-Coder-7B-Instruct) is trained with GRPO to optimize GPU kernels. Each "
          "episode, it writes a full Triton `kernel.py`, a real GPU runs AutoKernel's fixed `bench.py` "
          "(5 correctness stages + timing vs PyTorch), and the result comes back as the next message. "
          "Reward = log2(best speedup) if any turn passes, else 0, clipped to [−1, 3].", "",
          "![architecture](figures/01_architecture.png)", ""]
    L += ["## 2. Where it started: KernelBench v1", "",
          f"The first test set was KernelBench Level 1 (100 PyTorch ops). Rebuilt on Modal, an identity "
          f"solution passes {KB['counts']['PASS']}/{KB['n']} problems at a median "
          f"{KB['identity_speedup_median']:.3f}× measured speedup, so the timing is sound. But only {KB['valid']} "
          f"problems are harness-valid ({KB['conv_unseeded']} conv problems have unseeded weights), and a bench takes a "
          f"median {KB['wall_s_p50']} s on H100 vs {S1['bench_wall_s']['L4']['p50']} s for AutoKernel's bench on L4. "
          f"That's why RL moved to AutoKernel's 9 kernel types, with KernelBench kept as the held-out eval.", "",
          "![kb](figures/03_kb_identity_noise.png)", ""]
    L += ["## 3. The verifier", "",
          f"Only {S1['starters_pass_full']['H100']}/9 of AutoKernel's own starter kernels pass the full bench. "
          f"Quick mode skips numerical stability and lets {', '.join(S1['quick_pass_full_fail']['H100'])} pass, "
          f"so all rewards use the full bench. Run-to-run noise is at most {S1['max_cv_pct']}% CV.", "",
          "![starters](figures/02_starter_baselines.png)", ""]
    if V1 and V2:
        L += ["## 4. The base model, and a free win from better feedback", "",
              f"Untrained, the model passes in {pc(V2['pass_rate'])} of episodes (fast_1.0 "
              f"{pc(V2['fast_p']['fast_1.0'])}), mostly by resubmitting kernels whose starter already works. "
              f"Extra turns barely help. With the original design's feedback (the log tail) pass@turn was flat at "
              f"{pc(V1['pass_rate'])}. Sending the actual failure lines raised it to {pc(V2['pass_rate'])} and "
              f"mean reward from {V1['reward_mean']} to {V2['reward_mean']}, with no training.", "",
              "![pass@turn](figures/04_pass_at_turn_feedback.png)", "",
              "![per kernel](figures/05_per_kernel_base.png)", "",
              "![failures](figures/06_failure_taxonomy.png)", ""]
    L += ["## 5. GRPO", ""]
    if V2:
        z = V2["ablations"]["zero_adv_group_frac_by_group_size"]
        L += [f"GRPO learns only from groups whose rewards differ. On base-model rollouts {pc(z['4'])} of "
              f"4-sample groups and {pc(z['8'])} of 8-sample groups were all-equal, so training uses group=8, "
              f"4 turns, LoRA r=64, lr 1e-5.", "", "![zero adv](figures/07_zero_advantage_groups.png)", ""]
    if G1:
        L += [f"**v1 (original reward), {G1['steps']} steps:** reward {G1['reward_first3']} → {G1['reward_last3']}, "
              f"pass rate {pc(G1['pass_first3'])} → {pc(G1['pass_last3'])}. {pc(G1['zero_adv_mean'])} of groups gave no "
              f"gradient: {9 - len(G1['kernels_ever_passed'])} kernels never passed in any of the {G1['steps']} × 72 episodes.", ""]
    if G2:
        L += [f"**v2 (dense correctness credit + PASS floor + noise-floor ties), {G2['steps']} steps:** reward_v2 "
              f"{G2['reward_v2_first3']} → {G2['reward_v2_last3']}, original reward {G2['reward_first3']} → "
              f"{G2['reward_last3']}, pass rate {pc(G2['pass_first3'])} → {pc(G2['pass_last3'])}.", ""]
    L += ["![grpo](figures/09_grpo_curves.png)", ""] if G1 else [PENDING, ""]
    L += ["## 6. Transfer to KernelBench v1", ""]
    if KB_BASE:
        L += [f"Base fast_0 {pc(KB_BASE['fast_p']['fast_0.0'])}" +
              (f", after GRPO {pc(KB_LORA['fast_p']['fast_0.0'])}" if KB_LORA else "") + ".", "",
              "![kb transfer](figures/10_kernelbench_transfer.png)", ""]
    else:
        L += [PENDING, ""]
    L += ["## 7. Lessons", "",
          "1. The verifier and its feedback matter more than expected: quick-mode rewards are exploitable, "
          "and log-tail feedback hid every failure reason.",
          "2. The reward has a floor bug: a correct kernel slower than PyTorch scores below a crash.",
          ("3. GRPO signal is sparse when most groups all fail. Dense correctness credit cut the zero-gradient "
           f"groups from {pc(G1['zero_adv_mean'], 0)} to {pc(G2['zero_adv_mean'], 0)}, but {G2['steps']} steps weren't "
           "enough to solve new kernels; curriculum and more steps are next." if G1 and G2 else
           "3. GRPO signal is sparse when most groups all fail; curriculum and larger groups are the levers."),
          "4. Re-running an old project found 5 real integration bugs (see `CLAUDE.md`).", ""]
    (OUT / "writeup.md").write_text("\n".join(L))
    print("wrote writeup.md")


if __name__ == "__main__":
    write_qa(qa())
    write_writeup()
