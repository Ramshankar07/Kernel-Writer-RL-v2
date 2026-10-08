# Reference blog: "Training Search Agents with GRPO" (Jasper Lu)
URL: https://jasperlu.com/blog/training-search-agents-grpo/

Style and structure to borrow:
- Hands-on, radically open: ships viewers for every artifact (dataset explorer, rollout explorer,
  group explorer). "If any conclusions aren't backed up clearly... let me know."
- Opens with a thesis: "a well-designed reward function can influence how a model searches more
  effectively than system prompt changes or harness engineering."
- Order: dataset → harness (tool table) → reward (derives F1 → F4 from the user's cost of a miss)
  → quick GRPO recap → ablations (learning-rate sweep, batch/groups-per-step) → scaled run
  (train reward, eval F1 0.193 → 0.332, entropy 1.1 → 0.8 nats, "no signal" groups per update)
  → reward shaping (trajectory-recall bonus; −0.1 format penalty for off-form tool calls)
  → closing thoughts (curriculum by empirical pass rate, SFT warmup, more HP combos).
- Uses Dr. GRPO: drops the std division because "a query where the eight rollouts nearly agree has
  a small deviation... the update can end up dominated by queries the policy is already consistent on."
- Group explorer: "Seven of the eight rollouts score zero F4 and are indistinguishable under the
  older reward, but separable under either [new reward]."
- Names reward hacking explicitly (8 groups/step learned to "curate anything" to escape the −0.2 floor).
- Watches entropy: "entropy collapse ... a policy that always takes the same actions produces
  groups with no variance, and groups with no variance produce no gradient."
- One update per batch, synchronous, no ratio clipping (same simplification as our loop).

Direct parallels to our project (use these as the spine of deck v2):
| Blog | Ours (numbers in results/numbers.json) |
|---|---|
| "no signal" groups per update | zero-advantage groups: v1 65% → v2 38% |
| Dr. GRPO drops std normalisation | we found std≈0.002 timing-noise groups inflated to ±1 advantages; added a 0.02 noise floor |
| group explorer: 7/8 tied at 0 separable under new reward | reward v2 partial credit separates all-fail groups |
| batch size / LR ablations | group-size ablation (offline), feedback-format ablation, turn-budget ablation |
| format penalty for off-form tool calls | our "no code block" / invented-Triton-API failures |
| curriculum + SFT warmup as next steps | same conclusions: 5 kernels never pass; KernelBench fails on Triton API grounding |
| eval F1 went up 0.19 → 0.33 | ours did NOT improve: be as honest as the blog is open |
