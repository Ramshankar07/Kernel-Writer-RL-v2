"""
Figures for the deck. Reads results/numbers.json (+ raw jsonl where needed).
Every figure is skipped if its data doesn't exist yet.

    python presentation/make_figures.py
"""
import json
import pathlib

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "presentation" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Reference palette (dataviz skill, light mode), fixed slot order.
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, CRIT = "#0ca30c", "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 15, "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

N = json.loads((R / "numbers.json").read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_architecture():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax in axes:
        ax.set_xlim(0, 10), ax.set_ylim(0, 10), ax.axis("off")

    def box(ax, x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                    fc=SURFACE, ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=12.5, weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=10, color=INK2)

    def arrow(ax, a, b, label=""):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.5))
        if label:
            vertical = abs(a[0] - b[0]) < 0.01
            ax.text((a[0] + b[0]) / 2 + (0.25 if vertical else 0), (a[1] + b[1]) / 2 + (0 if vertical else 0.35),
                    label, ha="left" if vertical else "center", va="center", fontsize=9.5, color=MUTED)

    ax = axes[0]
    ax.set_title("Original design (Apr 2026): SkyPilot + verl")
    box(ax, 0.2, 5.8, 4, 2.6, "Trainer", "8×H100 on-demand\nverl GRPO + vLLM rollout", S1)
    box(ax, 5.8, 5.8, 4, 2.6, "Bench queue", "FastAPI + Redis (CPU)\ncontent-hash cache", S2)
    box(ax, 5.8, 1.0, 4, 2.6, "Bench workers", "N× A10/L4 spot\nAutoKernel bench.py", S3)
    arrow(ax, (4.2, 7.6), (5.8, 7.6), "POST")
    arrow(ax, (5.8, 6.6), (4.2, 6.6))
    arrow(ax, (7.8, 3.6), (7.8, 5.8), "pull job")
    ax.text(0.2, 2.2, "~$27/hr (README cost model)\n3 clusters, HTTP-coupled", fontsize=10.5, color=INK2)

    ax = axes[1]
    ax.set_title("Re-run (Sept 2026): Modal, ≤10 GPUs")
    box(ax, 0.2, 5.8, 4, 2.6, "Trainer", "1×H100, LoRA r=64\nGRPO loop (verl semantics)", S1)
    box(ax, 5.8, 5.8, 4, 2.6, "Policy", "1×H100 vLLM 0.8.5\nLoRA hot-swap per step", S4)
    box(ax, 5.8, 1.0, 4, 2.6, "Bencher", "≤8× L4 autoscaled\nfull bench, Dict cache", S3)
    box(ax, 0.2, 1.0, 4, 2.6, "KernelBench v1", "Level 1, 100 problems\nH100 bench_kb.py", S5)
    arrow(ax, (4.2, 7.4), (5.8, 7.4), "generate")
    arrow(ax, (4.2, 5.9), (5.8, 3.4))
    ax.text(5.3, 4.9, "bench\n(starmap)", fontsize=9.5, color=MUTED)
    save(fig, "01_architecture")


def fig_stage1():
    s = N.get("stage1")
    if not s:
        return
    kts = sorted(s["per_gpu"]["H100"], key=lambda k: -s["per_gpu"]["H100"][k]["speedup_quick"])
    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = range(len(kts))
    h = 0.3
    for off, gpu, col in ((-h / 2 - 0.02, "H100", S1), (h / 2 + 0.02, "L4", S2)):
        d = s["per_gpu"][gpu]
        vals = [d[k]["speedup_quick"] for k in kts]
        ax.barh([i + off for i in y], vals, height=h, color=col, label=gpu)
        for i, k in enumerate(kts):
            full = d[k]["full"]
            txt = f"{vals[i]:.2f}×" if vals[i] else "FAIL"
            if vals[i] and full != "PASS":
                txt += "  (fails full bench)"
            ax.text(vals[i] + 0.06, i + off, txt, va="center", fontsize=9.5,
                    color=INK2 if vals[i] else CRIT)
    ax.set_yticks(list(y), kts)
    ax.invert_yaxis()
    ax.axvline(1.0, color=AXIS, lw=1)
    ax.set_xlabel("speedup vs PyTorch (quick bench, mean of 3)")
    ax.set_title("AutoKernel starter kernels: only 3/9 pass the full bench")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v["speedup_quick"] for g in s["per_gpu"].values() for v in g.values()) * 1.45)
    save(fig, "02_starter_baselines")


def fig_kb_sanity():
    p = R / "kb_sanity.jsonl"
    if not p.exists() or "kb_sanity" not in N:
        return
    rows = [json.loads(l) for l in p.read_text().splitlines() if l]
    sp = [r["speedup"] for r in rows if r["correctness"] == "PASS"]
    if not sp:
        return
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(sp, bins=30, color=S1, edgecolor=SURFACE, linewidth=2)
    ax.axvline(1.0, color=INK2, lw=1)
    k = N["kb_sanity"]
    ax.set_title(f"KernelBench L1 harness test: identity ModelNew, {len(sp)}/{k['n']} PASS")
    ax.set_xlabel("measured speedup of an identical model (true value 1.0×)")
    ax.set_ylabel("problems")
    ax.grid(axis="x", visible=False)
    ax.text(0.98, 0.92, f"median {k['identity_speedup_median']:.3f}×\n"
            f"p5–p95 {k['identity_speedup_p5_p95'][0]:.3f}–{k['identity_speedup_p5_p95'][1]:.3f}×",
            transform=ax.transAxes, ha="right", va="top", color=INK2)
    save(fig, "03_kb_identity_noise")


def _ae(tag):
    return next((v for k, v in N.get("agent_eval", {}).items() if k.endswith(tag)), None)


def fig_pass_at_turn():
    v1, v2 = _ae("_fbv1"), _ae("_fbv2")
    if not (v1 and v2):
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for e, col, lab in ((v1, S2, "v1: log tail (original worker)"), (v2, S1, "v2: failure lines")):
        xs = range(1, len(e["pass_at_turn"]) + 1)
        ys = [100 * p for p in e["pass_at_turn"]]
        ax.plot(xs, ys, color=col, marker="o", ms=6, mec=SURFACE, mew=2, label=lab)
        ax.text(len(ys) + 0.15, ys[-1], f"{ys[-1]:.1f}%", va="center", color=INK2)
    ax.set_xlabel("turn budget (edits allowed)")
    ax.set_ylabel("episodes with ≥1 PASS (%)")
    ax.set_ylim(0, 50)
    ax.set_xlim(0.7, len(v1["pass_at_turn"]) + 0.9)
    ax.set_title("Base Qwen2.5-Coder-7B barely improves with more turns")
    ax.legend(loc="lower right")
    save(fig, "04_pass_at_turn_feedback")


def fig_per_kernel():
    v1, v2 = _ae("_fbv1"), _ae("_fbv2")
    if not (v1 and v2):
        return
    kts = sorted(v2["per_kernel"], key=lambda k: -v2["per_kernel"][k]["pass_rate"])
    fig, ax = plt.subplots(figsize=(10, 5))
    h = 0.3
    for off, e, col, lab in ((-h / 2 - 0.02, v1, S2, "v1 feedback"), (h / 2 + 0.02, v2, S1, "v2 feedback")):
        vals = [100 * e["per_kernel"][k]["pass_rate"] for k in kts]
        ax.barh([i + off for i in range(len(kts))], vals, height=h, color=col, label=lab)
        for i, k in enumerate(kts):
            b = e["per_kernel"][k]["best_speedup"]
            if vals[i]:
                ax.text(vals[i] + 1, i + off, f"{vals[i]:.0f}%  best {b:.2f}×", va="center", fontsize=9.5, color=INK2)
    ax.set_yticks(range(len(kts)), kts)
    ax.invert_yaxis()
    ax.set_xlim(0, 135)
    ax.set_xlabel("episodes with ≥1 PASS (%), 8 samples × 8 turns, L4")
    ax.set_title("Base model only solves kernels whose starter already works")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "05_per_kernel_base")


def fig_failures():
    v2 = _ae("_fbv2")
    if not v2:
        return
    items = list(v2["failure_taxonomy"].items())[:7]
    total = sum(v2["failure_taxonomy"].values())
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.barh(range(len(items)), [c for _, c in items], height=0.55, color=S1)
    for i, (k, c) in enumerate(items):
        ax.text(c + 3, i, f"{c}  ({100 * c / total:.0f}%)", va="center", color=INK2, fontsize=10)
    ax.set_yticks(range(len(items)), [k for k, _ in items])
    ax.invert_yaxis()
    ax.set_xlim(0, items[0][1] * 1.25)
    ax.set_xlabel("failed turns")
    ax.set_title(f"Why kernels fail: {total} non-PASS turns (base model, v2 feedback)")
    ax.grid(axis="y", visible=False)
    save(fig, "06_failure_taxonomy")


def fig_zero_adv():
    v1, v2 = _ae("_fbv1"), _ae("_fbv2")
    if not (v1 and v2):
        return
    fig, ax = plt.subplots(figsize=(8, 4.4))
    gs = sorted(int(g) for g in v2["ablations"]["zero_adv_group_frac_by_group_size"])
    w = 0.34
    for off, e, col, lab in ((-w / 2 - 0.02, v1, S2, "v1 feedback"), (w / 2 + 0.02, v2, S1, "v2 feedback")):
        z = e["ablations"]["zero_adv_group_frac_by_group_size"]
        vals = [100 * z[str(g)] for g in gs]
        ax.bar([i + off for i in range(len(gs))], vals, width=w, color=col, label=lab)
        for i, v in enumerate(vals):
            ax.text(i + off, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10, color=INK2)
    ax.set_xticks(range(len(gs)), [f"group = {g}" for g in gs])
    ax.set_ylim(0, 100)
    ax.set_ylabel("GRPO groups with zero advantage (%)")
    ax.set_title("Most GRPO groups give no gradient; bigger groups help")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save(fig, "07_zero_advantage_groups")


def fig_context():
    v2 = _ae("_fbv2")
    if not v2:
        return
    ys = v2["prompt_tokens_by_turn"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(range(1, len(ys) + 1), [y / 1000 for y in ys], color=S1, marker="o", ms=6, mec=SURFACE, mew=2)
    ax.fill_between(range(1, len(ys) + 1), [y / 1000 for y in ys], color=S1, alpha=0.1)
    ax.text(len(ys), ys[-1] / 1000 + 0.8, f"{ys[-1] / 1000:.1f}k", ha="center", color=INK2)
    ax.text(1, ys[0] / 1000 + 0.8, f"{ys[0] / 1000:.1f}k", ha="center", color=INK2)
    ax.set_ylim(0, max(ys) / 1000 * 1.25)
    ax.set_xlabel("turn")
    ax.set_ylabel("prompt tokens (thousands)")
    ax.set_title("Context grows ~1.8k tokens per turn")
    save(fig, "08_context_growth")


def fig_grpo():
    runs = N.get("grpo", {})
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = (("reward_v2_mean", "reward v2 (dense, per episode)", 1),
              ("reward_mean", "original reward (log2 speedup)", 1),
              ("zero_adv_group_frac", "zero-advantage groups (%)", 100))
    style = {"grpo_v1": (S2, "v1: original reward"), "grpo_v2": (S1, "v2: dense reward + noise floor")}
    for ax, (k, lab, mul) in zip(axes, panels):
        for name, g in runs.items():
            if name not in style:
                continue
            col, rl = style[name]
            pts = [(m["step"], m[k] * mul) for m in g["curve"] if m.get(k) is not None]
            if not pts:
                continue
            ax.plot(*zip(*pts), color=col, marker="o", ms=5, mec=SURFACE, mew=1.5, label=rl)
        ax.set_title(lab, fontsize=13)
        ax.set_xlabel("GRPO step")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        top = max((m.get(k) or 0) * mul for g in runs.values() for m in g["curve"])
        ax.set_ylim(0, 100 if mul == 100 else top * 1.3)
    axes[0].legend(loc="lower right", fontsize=10)
    fig.suptitle("GRPO on Qwen2.5-Coder-7B: 72 episodes per step (both runs scored on both rewards)",
                 x=0.01, ha="left", fontweight="bold", fontsize=15)
    fig.tight_layout()
    save(fig, "09_grpo_curves")


def fig_kb_transfer():
    b, l = _ae("_kb_base"), _ae("_kb_lora")
    if not b:
        return
    cats = list(b["failure_taxonomy"])[:5]
    if l:
        cats += [c for c in list(l["failure_taxonomy"])[:5] if c not in cats]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    h = 0.34
    series = [(b, S2, "base")] + ([(l, S1, "after GRPO v2")] if l else [])
    for j, (e, col, lab) in enumerate(series):
        off = (j - (len(series) - 1) / 2) * (h + 0.04)
        vals = [e["failure_taxonomy"].get(c, 0) for c in cats]
        ax.barh([i + off for i in range(len(cats))], vals, height=h, color=col, label=lab)
        for i, v in enumerate(vals):
            if v:
                ax.text(v + 0.6, i + off, str(v), va="center", fontsize=10, color=INK2)
    ax.set_yticks(range(len(cats)), cats)
    ax.invert_yaxis()
    ax.set_xlabel("failed turns")
    npass = [sum(e["turn_outcomes"].get("PASS", 0) for e in [x]) for x, _, _ in series]
    ax.set_title(f"KernelBench v1 L1 ({b['meta'].get('tasks')} valid problems, 3 turns): "
                 f"PASS turns base {npass[0]}" + (f", after GRPO {npass[1]}" if l else ""))
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, "10_kernelbench_transfer")

if __name__ == "__main__":
    fig_architecture()
    fig_stage1()
    fig_kb_sanity()
    fig_pass_at_turn()
    fig_per_kernel()
    fig_failures()
    fig_zero_adv()
    fig_context()
    fig_grpo()
    fig_kb_transfer()
