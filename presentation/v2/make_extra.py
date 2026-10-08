"""
Extra analyses + figures for the researcher-oriented deck (v2). Offline only: reads
existing files in results/, never calls Modal.

    python3 presentation/v2/make_extra.py

Writes results/extra_numbers.json, presentation/v2/figures/x*.png and
presentation/v2/best_kernel.py. Re-runnable (overwrites its outputs).
"""
import collections
import difflib
import glob
import json
import math
import pathlib
import re
import statistics as st
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = pathlib.Path(__file__).resolve().parents[2]
R = ROOT / "results"
V2 = ROOT / "presentation" / "v2"
OUT = V2 / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "modal_app"))
from reward_v2 import progress, reward_v2  # noqa: E402

# Palette / rcParams: identical to presentation/make_figures.py
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
GOOD, CRIT = "#0ca30c", "#d03b3b"
# Sequential blue ramp (dataviz reference palette, steps 100 -> 700) for the heatmap.
BLUE_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "font.size": 12,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 15, "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round",
})

NOISE_STD = 0.02          # v2 adv_min_std: bench timing noise in log2 units
GROUP = 8
RUNS = {"grpo_v1": "reward", "grpo_v2": "reward_v2"}   # training reward per run
AE = R / "agent_eval"
BASE = "autokernel_Qwen2.5-Coder-7B-Instruct_n8_t8_L4_fb{}.jsonl"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def jl(p):
    return [json.loads(l) for l in pathlib.Path(p).read_text().splitlines() if l.strip()]


def r4(x):
    return None if x is None else round(float(x), 4)


def load_rollouts(run):
    steps = {}
    for p in sorted((R / "grpo" / run / "rollouts").glob("step_*.jsonl")):
        s = int(p.stem.split("_")[1])
        trajs = jl(p)
        for t in trajs:
            t["reward_v2_re"] = reward_v2(t["turns"])
        steps[s] = trajs
    return steps


def groups_of(trajs):
    g = collections.OrderedDict()
    for t in trajs:
        g.setdefault(t["task"]["task_id"], []).append(t)
    return g


def advantages(rs, scheme, min_std=0.0):
    """scheme: 'std' (GRPO: (r-mu)/(sd+1e-6), ties -> 0) or 'dr' (Dr. GRPO: r-mu)."""
    mu, sd = st.mean(rs), st.pstdev(rs)
    if scheme == "dr":
        return [r - mu for r in rs]
    if sd <= min_std or sd == 0:
        return [0.0] * len(rs)
    return [(r - mu) / (sd + 1e-6) for r in rs]


# --------------------------------------------------------------------------- a. time
TURN_RE = re.compile(r"\[(\d\d:\d\d:\d\d)\] turn (\d+): (\d+) active, gen (\d+)s, bench (\d+)s, PASS (\d+)")
STEP_RE = re.compile(r"\[(\d\d:\d\d:\d\d)\] step (\d+)/\d+ .* roll (\d+)s train (\d+)s")


def parse_train_log(path):
    """Assign per-turn gen/bench lines to the step line that follows them. A 'resumed' line
    discards the buffered turns of an interrupted step (they were re-done after restart)."""
    steps, buf, abandoned = {}, [], []
    for line in pathlib.Path(path).read_text().splitlines():
        if "resumed from" in line:
            abandoned.extend(buf)
            buf = []
        elif m := TURN_RE.search(line):
            buf.append(dict(turn=int(m[2]), gen=int(m[4]), bench=int(m[5]), n_pass=int(m[6])))
        elif m := STEP_RE.search(line):
            steps[int(m[2])] = dict(turns=buf, roll=int(m[3]), train=int(m[4]), ts=m[1])
            buf = []
    return steps, abandoned, buf   # buf = turns of a step that never finished


def parse_stage2(path):
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        m = re.search(r"turn (\d+): (\d+) active, gen (\d+)s, bench (\d+)s", line)
        if m:
            rows.append(dict(turn=int(m[1]), gen=int(m[3]), bench=int(m[4])))
    return rows


def time_breakdown(E):
    out = {}
    logp = R / "grpo" / "grpo_v1" / "train.log"
    lsteps, abandoned, unfinished = parse_train_log(logp)
    ms = {r: jl(R / "grpo" / r / "metrics.jsonl") for r in RUNS}
    # v1: per-turn split where the log has all 4 turns for that step
    split = []
    for m in ms["grpo_v1"]:
        s = lsteps.get(m["step"])
        if s and len(s["turns"]) == 4:
            g, b = sum(x["gen"] for x in s["turns"]), sum(x["bench"] for x in s["turns"])
            split.append(dict(step=m["step"], gen=g, bench=b, rollout_s=m["rollout_s"],
                              train_s=m["train_s"], other_roll=m["rollout_s"] - g - b))
    gen, bench = sum(x["gen"] for x in split), sum(x["bench"] for x in split)
    oth, upd = sum(x["other_roll"] for x in split), sum(x["train_s"] for x in split)
    tot = gen + bench + oth + upd
    out["grpo_v1"] = {
        "steps_in_metrics": len(ms["grpo_v1"]),
        "steps_with_per_turn_split": len(split),
        "source": "results/grpo/grpo_v1/train.log per-turn 'gen Xs, bench Ys' lines + metrics.jsonl train_s",
        "mean_step_s": {"generation": r4(gen / len(split)), "benchmarking": r4(bench / len(split)),
                        "rollout_other": r4(oth / len(split)), "policy_update": r4(upd / len(split))},
        "share_pct": {"generation": round(100 * gen / tot, 1), "benchmarking": round(100 * bench / tot, 1),
                      "rollout_other": round(100 * oth / tot, 1), "policy_update": round(100 * upd / tot, 1)},
        "bench_share_of_rollout_pct": round(100 * bench / (gen + bench + oth), 1),
        "bench_turn_s_range": [min(x["bench"] for s in split for x in lsteps[s["step"]]["turns"]),
                               max(x["bench"] for s in split for x in lsteps[s["step"]]["turns"])],
        "gen_turn_s_range": [min(x["gen"] for s in split for x in lsteps[s["step"]]["turns"]),
                             max(x["gen"] for s in split for x in lsteps[s["step"]]["turns"])],
        "abandoned_turn_lines": len(abandoned), "unfinished_turn_lines_after_last_step": len(unfinished),
    }
    for run in RUNS:
        roll = sum(m["rollout_s"] for m in ms[run])
        tr = sum(m["train_s"] for m in ms[run])
        d = out.setdefault(run, {})
        d["rollout_vs_update_pct"] = {"rollout": round(100 * roll / (roll + tr), 1),
                                      "policy_update": round(100 * tr / (roll + tr), 1)}
        d["mean_rollout_s"], d["mean_train_s"] = r4(roll / len(ms[run])), r4(tr / len(ms[run]))
        d["trainer_gpu_idle_pct"] = round(100 * roll / (roll + tr), 1)
    out["grpo_v2"]["per_turn_split"] = None
    out["grpo_v2"]["note"] = ("no per-turn gen/bench log was saved for grpo_v2 (no train.log in results/), "
                              "so only rollout_s vs train_s from metrics.jsonl")
    for tag, f in (("fbv1", "stage2.log"), ("fbv2", "stage2_fbv2.log")):
        rows = parse_stage2(R / f)
        g, b = sum(x["gen"] for x in rows), sum(x["bench"] for x in rows)
        out[f"base_eval_{tag}"] = {"turns": len(rows), "gen_s": g, "bench_s": b,
                                   "bench_share_pct": round(100 * b / (g + b), 1)}
    out["definition"] = ("trainer_gpu_idle_pct = rollout_s / (rollout_s + train_s): the trainer H100 holds "
                         "the model but does nothing while vLLM generates and L4s benchmark (synchronous loop). "
                         "Checkpoint save time is outside both timers and not counted.")
    E["time_breakdown"] = out

    # figure: stacked horizontal bars (mean seconds per step)
    v1 = out["grpo_v1"]["mean_step_s"]
    rows = [
        ("GRPO v1\n(mean of %d steps)" % len(split),
         [("generation (vLLM, H100)", v1["generation"], S1), ("benchmarking (≤8× L4)", v1["benchmarking"], S2),
          ("policy update (trainer H100)", v1["policy_update"], S3)]),
        ("GRPO v2\n(mean of %d steps)" % len(ms["grpo_v2"]),
         [("rollout: gen + bench (split not logged)", out["grpo_v2"]["mean_rollout_s"], MUTED),
          ("policy update (trainer H100)", out["grpo_v2"]["mean_train_s"], S3)]),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.9))
    seen = set()
    for i, (lab, segs) in enumerate(rows):
        left, tot = 0, sum(v for _, v, _ in segs)
        for name, v, c in segs:
            ax.barh(i, v, left=left, height=0.55, color=c, edgecolor=SURFACE, linewidth=2,
                    label=None if name in seen else name)
            seen.add(name)
            pct = 100 * v / tot
            txt = f"{v:.0f}s · {pct:.0f}%"
            if v / tot > 0.12:
                ax.text(left + v / 2, i, txt, ha="center", va="center", color="white", fontsize=10.5,
                        weight="bold")
            else:
                ax.text(left + v + 6, i, txt, ha="left", va="center", color=INK2, fontsize=10.5)
            left += v
    ax.set_axisbelow(True)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("seconds per GRPO step (72 episodes × 4 turns)")
    ax.set_xlim(0, max(sum(v for _, v, _ in s) for _, s in rows) * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Where a GRPO step goes: the trainer GPU is idle "
                 f"{out['grpo_v1']['trainer_gpu_idle_pct']:.0f}% (v1) / "
                 f"{out['grpo_v2']['trainer_gpu_idle_pct']:.0f}% (v2) of the time")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=10.5)
    save(fig, "x1_time_breakdown")


# --------------------------------------------------------------------------- b. Dr. GRPO
def dr_grpo(E, ROLL):
    out = {}
    for run, key in RUNS.items():
        min_std_trained = NOISE_STD if run == "grpo_v2" else 0.0
        schemes = {"grpo_std_as_trained": ("std", min_std_trained),
                   "grpo_std_no_floor": ("std", 0.0), "grpo_std_floor_0.02": ("std", NOISE_STD),
                   "dr_grpo_no_std": ("dr", 0.0)}
        mass = {k: 0.0 for k in schemes}
        noise_mass = {k: 0.0 for k in schemes}
        n_groups = n_zero = n_noise = 0
        per_step, noise_kernels, max_diff = [], collections.Counter(), 0.0
        for s, trajs in sorted(ROLL[run].items()):
            gs = groups_of(trajs)
            z = nz = 0
            for tid, g in gs.items():
                rs = [t["reward"] if key == "reward" else t["reward_v2_re"] for t in g]
                sd = st.pstdev(rs)
                n_groups += 1
                zero, noise = sd == 0, 0 < sd < NOISE_STD
                z += zero
                nz += noise
                if noise:
                    noise_kernels[g[0]["task"]["kernel_type"]] += 1
                for name, (sch, ms_) in schemes.items():
                    a = advantages(rs, sch, ms_)
                    m = sum(abs(x) for x in a)
                    mass[name] += m
                    if noise:
                        noise_mass[name] += m
                    if name == "grpo_std_as_trained":
                        max_diff = max(max_diff, max(abs(x - t["advantage"]) for x, t in zip(a, g)))
            n_zero += z
            n_noise += nz
            k = len(gs)
            per_step.append(dict(step=s, groups=k, zero_var=z, noise_level=nz,
                                 kept_groups_dapo=k - z, eff_batch_dapo=(k - z) * GROUP,
                                 kept_groups_floor=k - z - nz, eff_batch_floor=(k - z - nz) * GROUP))
        eb = [p["eff_batch_dapo"] for p in per_step]
        ebf = [p["eff_batch_floor"] for p in per_step]
        out[run] = {
            "training_reward": "original reward.py" if key == "reward" else "reward_v2",
            "steps": len(per_step), "groups_total": n_groups,
            "zero_variance_groups": n_zero, "noise_level_groups_0_lt_std_lt_0.02": n_noise,
            "noise_level_group_kernels": dict(noise_kernels),
            "abs_adv_mass_share_from_noise_groups_pct": {
                k: round(100 * noise_mass[k] / mass[k], 1) if mass[k] else None for k in schemes},
            "abs_adv_mass_total": {k: r4(mass[k]) for k in schemes},
            "dapo_filter_removed_groups_pct": round(100 * n_zero / n_groups, 1),
            "dapo_eff_batch_trajs": {"mean": r4(st.mean(eb)), "min": min(eb), "max": max(eb),
                                     "nominal": 9 * GROUP},
            "dapo_plus_noise_floor_removed_pct": round(100 * (n_zero + n_noise) / n_groups, 1),
            "dapo_plus_noise_floor_eff_batch_trajs": {"mean": r4(st.mean(ebf)), "min": min(ebf),
                                                     "max": max(ebf)},
            "check_max_abs_diff_vs_logged_advantage": r4(max_diff),
            "per_step": per_step,
        }
    out["definitions"] = {
        "grpo_std": "A = (r - mean) / (pstdev + 1e-6), groups with pstdev <= min_std -> 0 (train_grpo.grpo_advantages)",
        "dr_grpo_no_std": "A = r - mean (no std division)",
        "noise_level_group": "0 < pstdev(group reward) < 0.02 (bench timing noise, log2 units)",
        "dapo_filter": "drop groups whose 8 rewards are identical (pstdev == 0)",
    }
    E["dr_grpo_whatif"] = out

    # figure: share of |A| mass from noise-level groups under each scheme
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    names = [("grpo_std_no_floor", "GRPO, std-normalised (v1 as trained)", S2),
             ("grpo_std_floor_0.02", "GRPO + 0.02 noise floor (v2 as trained)", S1),
             ("dr_grpo_no_std", "Dr. GRPO (no std division)", S3)]
    w = 0.26
    for j, (k, lab, c) in enumerate(names):
        vals = [out[r]["abs_adv_mass_share_from_noise_groups_pct"][k] for r in RUNS]
        xs = [i + (j - 1) * (w + 0.02) for i in range(len(RUNS))]
        ax.bar(xs, vals, width=w, color=c, label=lab, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=10.5, color=INK2)
    ax.set_xticks(range(len(RUNS)), [f"{r} rollouts\n({out[r]['training_reward']}, "
                                     f"{out[r]['noise_level_groups_0_lt_std_lt_0.02']}/{out[r]['groups_total']}"
                                     f" noise groups)" for r in RUNS])
    ax.set_ylim(0, 100)
    ax.set_axisbelow(True)
    ax.set_ylabel("% of total |advantage| from\ngroups with reward std < 0.02")
    ax.grid(axis="x", visible=False)
    v1s = out["grpo_v1"]["abs_adv_mass_share_from_noise_groups_pct"]["grpo_std_no_floor"]
    ax.set_title(f"Std-normalised GRPO: timing-noise groups (std < 0.02) carry {v1s:.0f}% of v1's |advantage|",
                 fontsize=14)
    ax.legend(loc="upper right", fontsize=10.5)
    save(fig, "x2_dr_grpo_noise_mass")


# --------------------------------------------------------------------------- c. group explorer
def stage_summary(turns):
    best = max(turns, key=lambda x: progress(x.get("stages")))
    s = best.get("stages") or {}
    if not str(s.get("smoke_test") or "").startswith("PASS"):
        return "smoke fail" if s else "no bench"
    sw = str(s.get("shape_sweep") or "")
    m = re.search(r"\((\d+)/(\d+) failed\)", sw)
    txt = "sweep ok" if sw.startswith("PASS") else (f"sweep {int(m[2]) - int(m[1])}/{m[2]}" if m else "sweep ?")
    ns = "stab ok" if str(s.get("numerical_stability") or "").startswith("PASS") else "stab fail"
    return f"smoke ok\n{txt}\n{ns}"


def group_explorer(E, ROLL):
    cands = []
    for s, trajs in sorted(ROLL["grpo_v1"].items()):
        for tid, g in groups_of(trajs).items():
            kt = g[0]["task"]["kernel_type"]
            if kt in ("layernorm", "matmul") and all(t["reward"] == 0 for t in g):
                v2 = [t["reward_v2_re"] for t in g]
                cands.append((len(set(round(x, 4) for x in v2)), st.pstdev(v2), -s, tid, s, g))
    cands.sort(key=lambda c: (c[0], c[1], c[2]), reverse=True)
    _, _, _, tid, step, g = cands[0]
    g = sorted(g, key=lambda t: t["sample"])
    v1 = [t["reward"] for t in g]
    v2 = [t["reward_v2_re"] for t in g]
    a_v1 = advantages(v1, "std", 0.0)
    a_v2 = advantages(v2, "std", NOISE_STD)
    a_dr = advantages(v2, "dr")
    # tied under v1 but separable under v2, per step, for both runs
    tied = {}
    for run in RUNS:
        per = []
        for s, trajs in sorted(ROLL[run].items()):
            n = 0
            for _, gg in groups_of(trajs).items():
                if st.pstdev([t["reward"] for t in gg]) == 0 and \
                        st.pstdev([t["reward_v2_re"] for t in gg]) > NOISE_STD:
                    n += 1
            per.append(n)
        tied[run] = {"per_step": per, "mean": r4(st.mean(per)), "min": min(per), "max": max(per),
                     "of_groups_per_step": 9}
    E["group_explorer"] = {
        "run": "grpo_v1", "step": step, "task_id": tid, "kernel_type": g[0]["task"]["kernel_type"],
        "selection": "grpo_v1 layernorm/matmul group with all 8 original rewards 0; most distinct reward_v2 "
                     "values, then highest reward_v2 std, then earliest step",
        "candidates_all_zero_layernorm_matmul": len(cands),
        "rollouts": [{"sample": t["sample"], "reward_v1": r4(a), "reward_v2": r4(b), "adv_v1": r4(c),
                      "adv_v2_std_floor": r4(d), "adv_v2_dr_grpo": r4(e), "best_stage": stage_summary(t["turns"]).replace("\n", ", ")}
                     for t, a, b, c, d, e in zip(g, v1, v2, a_v1, a_v2, a_dr)],
        "reward_v2_std": r4(st.pstdev(v2)),
        "tied_under_v1_separable_under_v2": tied,
        "tied_definition": "pstdev(original reward) == 0 and pstdev(reward_v2) > 0.02 within a 9-kernel step",
        "stored_reward_v2_matches_recompute_grpo_v2": all(
            abs(t["reward_v2"] - t["reward_v2_re"]) < 1e-9 for tr in ROLL["grpo_v2"].values() for t in tr),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.12))
    xs = list(range(len(g)))
    w = 0.36
    ax1.bar([x - w / 2 - 0.01 for x in xs], v1, width=w, color=S2, label="original reward (v1)")
    ax1.bar([x + w / 2 + 0.01 for x in xs], v2, width=w, color=S1, label="reward v2 (partial credit)")
    for x, a in zip(xs, v1):
        ax1.plot([x - w / 2 - 0.01 - w / 2, x - w / 2 - 0.01 + w / 2], [0, 0], color=S2, lw=3,
                 solid_capstyle="butt")
        ax1.text(x - w / 2 - 0.01, 0.006, "0", ha="center", va="bottom", fontsize=9.5, color=INK2)
    for x, b, t in zip(xs, v2, g):
        ax1.text(x + w / 2 + 0.01, b + 0.006, f"{b:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax1.set_ylim(0, max(0.3, max(v2) * 1.35))
    ax1.set_ylabel("episode reward")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left", ncol=2, fontsize=10.5)
    ax1.set_title(f"Group explorer: {g[0]['task']['kernel_type']}, GRPO v1 step {step}. "
                  f"All 8 tied at 0 under v1, separable under v2")
    ax2.axhline(0, color=AXIS, lw=1)
    ax2.bar([x - w / 2 - 0.01 for x in xs], a_v1, width=w, color=S2, label="advantage under v1 (all 0: no gradient)")
    ax2.bar([x + w / 2 + 0.01 for x in xs], a_v2, width=w, color=S1, label="advantage under v2 (std-normalised)")
    for x, a in zip(xs, a_v1):
        ax2.plot([x - w / 2 - 0.01 - w / 2, x - w / 2 - 0.01 + w / 2], [0, 0], color=S2, lw=3,
                 solid_capstyle="butt")
    lim = max(abs(a) for a in a_v2) * 1.3 or 1
    ax2.set_ylim(-lim, lim)
    ax2.set_ylabel("advantage")
    ax2.grid(axis="x", visible=False)
    ax2.legend(loc="lower left", ncol=2, fontsize=10.5)
    ax2.set_xticks(xs, [f"rollout {t['sample']}\n{stage_summary(t['turns'])}" for t in g], fontsize=9.5)
    save(fig, "x3_group_explorer")


# --------------------------------------------------------------------------- d. entropy proxy
def entropy_proxy(E):
    out = {"definition": "-mean_token_logp from metrics.jsonl: mean negative log-prob (nats) of the sampled "
                         "assistant tokens of the trajectories that were trained on (|advantage| > 0), "
                         "under the current policy before the update. A proxy, NOT true entropy "
                         "(sampling at T=1.0; the token population changes with which groups are trained)."}
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for run, c in (("grpo_v1", S2), ("grpo_v2", S1)):
        ms = jl(R / "grpo" / run / "metrics.jsonl")
        xs, ys = [m["step"] for m in ms], [-m["mean_token_logp"] for m in ms]
        out[run] = {"per_step": [r4(y) for y in ys], "first": r4(ys[0]), "last": r4(ys[-1]),
                    "mean": r4(st.mean(ys)), "min": r4(min(ys)), "max": r4(max(ys)),
                    "n_trained_trajs_per_step": [m["n_trained_trajs"] for m in ms]}
        ax.plot(xs, ys, color=c, marker="o", ms=5, label=f"{run} ({'original' if run == 'grpo_v1' else 'v2'} reward)")
        ax.text(xs[-1] + 0.3, ys[-1], run, color=INK2, va="center", fontsize=10.5)
    ax.set_ylim(0, max(max(out[r]["per_step"]) for r in RUNS) * 1.3)
    ax.set_xlim(0.5, 20)
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("−mean log p(sampled token)  [nats]")
    ax.set_title("Entropy proxy: −mean token log-prob of trained samples")
    ax.legend(loc="lower left", fontsize=10.5)
    save(fig, "x4_entropy_proxy")
    E["entropy_proxy"] = out


# --------------------------------------------------------------------------- e. transitions
OUTC = ["PASS", "FAIL", "CRASH"]


def outcome(x):
    c = x["correctness"]
    return c if c in ("PASS", "FAIL") else "CRASH"   # TIMEOUT / INFRA_ERROR folded into CRASH


def transitions(E):
    out = {}
    for tag in ("v2", "v1"):
        trajs = jl(AE / BASE.format(tag))
        cnt = {a: collections.Counter() for a in OUTC}
        other = collections.Counter()
        same_code_pp = pp = 0
        for t in trajs:
            ts = t["turns"]
            for x in ts:
                if x["correctness"] not in ("PASS", "FAIL", "CRASH"):
                    other[x["correctness"]] += 1
            for a, b in zip(ts, ts[1:]):
                cnt[outcome(a)][outcome(b)] += 1
                if outcome(a) == outcome(b) == "PASS":
                    pp += 1
                    same_code_pp += a.get("code_sha") == b.get("code_sha")
        P = {a: {b: r4(cnt[a][b] / sum(cnt[a].values())) if sum(cnt[a].values()) else None for b in OUTC}
             for a in OUTC}
        out[f"base_fb{tag}"] = {
            "file": f"results/agent_eval/{BASE.format(tag)}", "trajectories": len(trajs),
            "transitions": sum(sum(c.values()) for c in cnt.values()),
            "counts": {a: dict(cnt[a]) for a in OUTC}, "P_next_given_current": P,
            "P_pass_broken_next_turn": r4(1 - P["PASS"]["PASS"]),
            "pass_to_pass_same_code_sha_frac": r4(same_code_pp / pp) if pp else None,
            "folded_into_CRASH": dict(other),
        }
        if tag == "v2":
            M, C = [[P[a][b] for b in OUTC] for a in OUTC], cnt
    E["turn_transitions"] = out

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_SEQ)
    fig, ax = plt.subplots(figsize=(7.2, 6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for i, a in enumerate(OUTC):
        for j, b in enumerate(OUTC):
            v = M[i][j]
            ax.text(j, i, f"{v:.2f}\n(n={C[a][b]})", ha="center", va="center", fontsize=12,
                    color="white" if v > 0.45 else INK, weight="bold" if i == j else "normal")
    ax.set_xticks(range(3), OUTC)
    ax.set_yticks(range(3), [f"{a}\n(n={sum(C[a].values())})" for a in OUTC])
    ax.set_xlabel("outcome at turn t+1")
    ax.set_ylabel("outcome at turn t")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, labelsize=12, labelcolor=INK2)
    ax.set_title("Base model, v2 feedback: P(next turn | this turn)", fontsize=13.5)
    save(fig, "x5_turn_transitions")


# --------------------------------------------------------------------------- f/g. speedups
def all_sources():
    src = []
    for tag in ("v1", "v2"):
        for t in jl(AE / BASE.format(tag)):
            src.append((f"base_eval_fb{tag}", None, t))
    for p in sorted(AE.glob("*.jsonl")):
        if p.name.startswith("kb_") or p.name == "smoke.jsonl":
            for t in jl(p):
                src.append((f"agent_eval/{p.stem}", None, t))
    for run in RUNS:
        for p in sorted((R / "grpo" / run / "rollouts").glob("step_*.jsonl")):
            for t in jl(p):
                src.append((run, int(p.stem.split("_")[1]), t))
    return src


def speedups(E):
    src = all_sources()
    passes = []
    for name, step, t in src:
        for x in t["turns"]:
            if x["correctness"] == "PASS":
                passes.append((x.get("speedup") or 0.0, name, step, t, x))
    best = max(passes, key=lambda p: p[0])
    sp, name, step, t, x = best
    kt = t["task"].get("kernel_type", t["task"]["task_id"])
    code = x["code"]
    same = [p[0] for p in passes if p[4].get("code_sha") == x.get("code_sha")]
    top = sorted(passes, key=lambda p: -p[0])[:10]
    starter_p = R / "autokernel_src" / "kernels" / f"{kt}.py"
    sim = r4(difflib.SequenceMatcher(None, starter_p.read_text(), code).ratio()) if starter_p.exists() else None
    stage1 = json.loads((R / "numbers.json").read_text())["stage1"]["per_gpu"]["L4"].get(kt, {})
    lines = code.splitlines()
    k0 = next((i for i, l in enumerate(lines) if "@triton.jit" in l), 0)
    header = (f"# Best PASS kernel found (extracted by presentation/v2/make_extra.py; code below is verbatim).\n"
              f"# source={name} step={step} kernel={kt} sample={t.get('sample')} turn={x['turn']} "
              f"speedup={sp:.3f}x vs PyTorch (L4 full bench) code_sha={x.get('code_sha')}\n\n")
    (V2 / "best_kernel.py").write_text(header + code + ("\n" if not code.endswith("\n") else ""))
    E["best_kernel"] = {
        "kernel_type": kt, "speedup": r4(sp), "source": name, "step": step, "sample": t.get("sample"),
        "turn": x["turn"], "code_sha": x.get("code_sha"), "cached": x.get("cached"),
        "same_code_sha_pass_count": len(same), "same_code_sha_speedup_range": [r4(min(same)), r4(max(same))],
        "similarity_to_starter_difflib": sim,
        "diff_vs_starter": [l for l in difflib.unified_diff(
            starter_p.read_text().splitlines(), lines, lineterm="", n=0)
            if l[:1] in "+-" and not l.startswith(("+++", "---"))] if starter_p.exists() else None,
        "starter_L4_stage1": {"speedup_quick": stage1.get("speedup_quick"), "full": stage1.get("full")},
        "saved_to": "presentation/v2/best_kernel.py",
        "excerpt_25_lines": "\n".join(lines[k0:k0 + 25]),
        "top10": [{"speedup": r4(p[0]), "kernel": p[3]["task"].get("kernel_type"), "source": p[1],
                   "step": p[2], "code_sha": p[4].get("code_sha")} for p in top],
    }

    # g. histogram
    groups = [("base model evals", ("base_eval", "agent_eval/"), S1), ("GRPO v1 rollouts", ("grpo_v1",), S2),
              ("GRPO v2 rollouts", ("grpo_v2",), S3)]
    pos = [p for p in passes if p[0] > 0]
    vals = [[math.log2(p[0]) for p in pos if p[1].startswith(prs)] for _, prs, _ in groups]
    allv = [math.log2(p[0]) for p in pos]
    by_k = collections.Counter(p[3]["task"].get("kernel_type") for p in pos)
    uniq = len({p[4].get("code_sha") for p in pos})
    E["pass_speedups"] = {
        "pass_turns": len(passes), "pass_turns_speedup_gt0": len(pos),
        "pass_turns_speedup_0": len(passes) - len(pos), "unique_code_sha": uniq,
        "by_source": {g: len(v) for (g, _, _), v in zip(groups, vals)},
        "by_source_file": dict(collections.Counter(p[1] for p in pos)),
        "by_kernel": dict(by_k),
        "median_speedup": r4(2 ** st.median(allv)),
        "frac_ge_1x": r4(sum(v >= 0 for v in allv) / len(allv)),
        "frac_lt_1x": r4(sum(v < 0 for v in allv) / len(allv)),
        "min": r4(2 ** min(allv)), "max": r4(2 ** max(allv)),
        "note": "every PASS turn counted (identical resubmitted code counts again; see unique_code_sha)",
    }
    # per-kernel clusters + how close the unique PASS codes are to the starter
    kstats = {}
    for k in sorted(by_k, key=lambda k: -by_k[k]):
        v = sorted(p[0] for p in pos if p[3]["task"].get("kernel_type") == k)
        uc = {p[4].get("code_sha"): p[4]["code"] for p in pos if p[3]["task"].get("kernel_type") == k}
        stp = R / "autokernel_src" / "kernels" / f"{k}.py"
        sims = [difflib.SequenceMatcher(None, stp.read_text(), c).ratio() for c in uc.values()] if stp.exists() else []
        kstats[k] = {"n": len(v), "min": r4(v[0]), "median": r4(st.median(v)), "max": r4(v[-1]),
                     "unique_codes": len(uc),
                     "median_similarity_to_starter": r4(st.median(sims)) if sims else None}
    E["pass_speedups"]["per_kernel"] = kstats
    lin = [p[0] for p in pos]
    E["pass_speedups"]["bands"] = {"lt_0.95x": sum(x < 0.95 for x in lin),
                                  "0.95_to_1.05x": sum(0.95 <= x < 1.05 for x in lin),
                                  "ge_1.05x": sum(x >= 1.05 for x in lin)}
    band = E["pass_speedups"]["bands"]
    XL, XH = -2.0, 2.0   # plotted range 0.25x .. 4x
    below = sum(v < XL for v in allv)
    kcol = [S1, S2, S3, S4, S5]
    kvals = [[math.log2(p[0]) for p in pos if p[3]["task"].get("kernel_type") == k] for k in kstats]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    bw = 1 / 16
    bins = [XL + i * bw for i in range(int((XH - XL) / bw) + 1)]
    ax.hist([[max(XL, x) for x in v] for v in kvals], bins=bins, stacked=True, color=kcol[:len(kvals)],
            label=[f"{k} (n={kstats[k]['n']}, median {kstats[k]['median']:.2f}×)" for k in kstats],
            edgecolor=SURFACE, linewidth=1)
    ax.set_axisbelow(True)
    ax.axvline(0, color=INK2, lw=1.2)
    ymax = ax.get_ylim()[1]
    ax.text(0.04, ymax * 0.97, "1.0× (PyTorch)", color=INK2, va="top", fontsize=10.5)
    ax.text(XL + 0.02, ymax * 0.62, f"{below} turns < 0.25×\n(min {E['pass_speedups']['min']:.3f}×)\nbinned at left edge",
            color=MUTED, va="top", fontsize=9.5)
    ticks = list(range(int(XL), int(XH) + 1))
    ax.set_xticks(ticks, [f"{2 ** t:g}×" for t in ticks])
    ax.set_xlim(XL, XH)
    ax.set_xlabel("speedup vs PyTorch (log2 scale, bins of 1/16 octave)")
    ax.set_ylabel("PASS turns")
    ax.grid(axis="x", visible=False)
    ax.set_title(f"All {len(pos)} PASS turns ({uniq} unique codes): "
                 f"{100 * band['0.95_to_1.05x'] / len(pos):.0f}% within ±5% of PyTorch, "
                 f"{100 * band['ge_1.05x'] / len(pos):.0f}% faster", fontsize=14)
    leg = ax.legend(loc="upper left", fontsize=10, title="kernel", title_fontsize=10)
    leg._legend_box.align = "left"
    save(fig, "x6_speedup_hist")


def main():
    E = {"generated_by": "presentation/v2/make_extra.py"}
    ROLL = {run: load_rollouts(run) for run in RUNS}
    time_breakdown(E)
    dr_grpo(E, ROLL)
    group_explorer(E, ROLL)
    entropy_proxy(E)
    transitions(E)
    speedups(E)
    (R / "extra_numbers.json").write_text(json.dumps(E, indent=1))
    print("wrote results/extra_numbers.json")


if __name__ == "__main__":
    main()
