# Deck v3 storyline: GPU kernel verification

Build: `python3 presentation/v3/make_v3_numbers.py && node presentation/v3/build_deck_v3.js`
Render: `modal run modal_app/render_deck.py --deck presentation/v3/deck_v3.pptx --out presentation/v3/_render`

The subject is the harness and the RMSNorm kernel. The GRPO study is the use case that motivated the harness.
This file holds no numbers of its own. Every figure on the slides comes from `results/numbers.json`,
`results/extra_numbers.json`, `results/v3/v3_numbers.json` or `results/v3/rmsnorm_remeasure.json`,
and the speaker notes in the pptx quote them.

## Main (15)
| # | Slide | One-liner | Speaker-note gist |
|---|---|---|---|
| 1 | Title: GPU kernel verification | Why correctness and timing are harder than they look; the RL agent is the use case | Built and debugged a Triton eval harness, reward = harness; torch/Triton versions pinned |
| 2 | The verifier paid 2.9× for 0% speed and broken numerics | Two fp32→fp16 casts flipped FAIL→PASS. Remeasured, model ≈ starter latency; model zeroes rows at RMS≈4 | Speed is the starter's fusion; reference overflows the same way; reward scored vs PyTorch, not the starter |
| 3 | The harness: model output → reward | Parse → correctness → timing → cache → reward, each with the defect found; KB bridge strip | Every box gave a wrong answer once; the KB bridge left 55/100 usable |
| 4 | RMSNorm, line by line | Verbatim kernel, one program per row, masks/tails, compiled regs/loads, bench contract | The casts moved arithmetic to fp16 but left memory ops alone (PTX counts) |
| 5 | What the two casts change numerically | Dtype-flow table (ref / starter / model); overflow thresholds computed from fp16 max; stability-flip mechanism | This is a prediction read from the code; the next slide tests it |
| 6 | Measured vs fp64 truth: the oracle is the bug | Stage-3 replay: starter, fp32-acc and torch.compile FAIL; the fp16 kernel PASSes with the reference's error. First failing scale matches sqrt(65504/N) | Fix the oracle, not the kernel |
| 7 | Memory traffic: same bytes, same speed | Per-shape bytes, floor, cold µs, GB/s (~78% of L4 peak); eager = 6 kernels, 3.5× bytes; headroom ≤1.28× | The 2.9× is fusion; torch.compile ties |
| 8 | How the 2.9× was measured | Logged starter (quick) vs model (full) vs remeasure (200 reps, warm/cold, 3 baselines); do_bench semantics | Timer was fine, attribution wasn't; model/starter ≈1.00× across all shapes, L4 and H100 |
| 9 | Harness findings I: AutoKernel bench.py | FIXED/AVOIDED/REMAINS/IDENTIFIED: flag+stdout, --quick, both-NaN rule, fp16 reference, narrow stability, H100 spec lookup | The three REMAINS items are exactly what let the fp16 kernel pass |
| 10 | Harness findings II: KernelBench bridge | Identity test; super(Model) crash, unseeded init, CPU get_inputs timeouts, listing bug | Timing is trustworthy where valid; coverage isn't |
| 11 | Timing: what is noise, what is cache | Low-variance groups: 27/42 mix PASS/FAIL, 201/271 best-PASS timings cached | Remeasure fixed code uncached before calling anything noise |
| 12 | The use case: an RL kernel agent | Architecture + setup table; benchmarking is most of step wall time | Harness is both bottleneck and signal |
| 13 | Verification quality set training-signal quality | 65% → 38% zero-advantage groups; floor bug; 78.9% |A| (with caveat); 0/28 → 0/28 | The whole RL result on one slide; changes were bundled, one run each |
| 14 | Roadmap: verification before scale | Hidden-test verification; deterministic timing; uncached, attributed timing (starter/prev-turn + torch.compile baselines) | Decision rule: scale only after independent evaluation shows a reproducible gain |
| 15 | Takeaways | 2 casts; 55/100; 201/271; 65% | Verifier sets the ceiling on signal |

## Appendix (5)
A1 reward v1 vs v2 (formulas + worked cases) · A2 ties and advantage mass (offline regrouping, no-std replay, DAPO) ·
A3 KernelBench transfer and time breakdown (+ cost estimates) · A4 remeasure for every shape (fp16/bf16 model÷starter) · A5 related work.

## Changes from v2
- The headline changed from GRPO to verification. v2 slides 9–13 (feedback, group size, group explorer, curves, noise mass) are now slide 13 plus A2.
- v2 slide 17 (diffusion-policy question, "No diffusion model was tested here") is replaced by the roadmap on slide 14.
- RMSNorm is now the hero across slides 2 and 4–8, with remeasure data. If `results/v3/rmsnorm_remeasure.json` is missing, those slides render dashed "PENDING remeasure" boxes.
- The corrected-deck caveats stay in: low variance ≠ noise, |A| mass ≠ gradient mass, bundled v2 changes, one run each.
