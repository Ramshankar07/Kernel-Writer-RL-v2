#!/usr/bin/env python3
"""Copy the project website's data and figures into docs/assets/.

Re-runnable and idempotent: a file is only written when its bytes would
change, so a second run reports no changes.

    python3 scripts/build_site_assets.py

Outputs
  docs/assets/data/{numbers,extra_numbers,phase3,mix_report}.json
  docs/assets/figures/<original filename>.png   (presentation/figures, presentation/v2/figures)
  docs/assets/figures/deck_v3_cover.png         (presentation/v3/_render/slide-01.png)
  docs/assets/figures/index.json                [{file, source_path, width, height, title}]

PNGs wider than MAX_WIDTH are downscaled (aspect ratio kept) when Pillow is
installed; otherwise they are copied as-is.
"""
from __future__ import annotations

import io
import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_ASSETS = ROOT / "docs" / "assets"
DATA_OUT = DOCS_ASSETS / "data"
FIG_OUT = DOCS_ASSETS / "figures"
MAX_WIDTH = 1600
SIZE_BUDGET = 15 * 1024 * 1024

DATA_SOURCES = {
    "numbers.json": "results/numbers.json",
    "extra_numbers.json": "results/extra_numbers.json",
    "phase3.json": "results/v3/phase3.json",
    "mix_report.json": "results/v3/sft/mix_report.json",
}
FIGURE_DIRS = ["presentation/figures", "presentation/v2/figures"]
EXTRA_FIGURES = {"deck_v3_cover.png": "presentation/v3/_render/slide-01.png"}

# Short human titles, taken from the chart titles in presentation/make_figures.py
# and presentation/v2/make_extra.py (data-dependent numbers dropped on purpose:
# numbers on the site must come from docs/assets/data, not from this table).
TITLES = {
    "01_architecture.png": "Original design (SkyPilot + verl) vs. Modal re-run",
    "02_starter_baselines.png": "AutoKernel starter kernels on the full bench",
    "03_kb_identity_noise.png": "KernelBench L1 harness test: identity ModelNew",
    "04_pass_at_turn_feedback.png": "Base Qwen2.5-Coder-7B barely improves with more turns",
    "05_per_kernel_base.png": "Base model only solves kernels whose starter already works",
    "06_failure_taxonomy.png": "Why kernels fail (base model, v2 feedback)",
    "07_zero_advantage_groups.png": "Most GRPO groups give no gradient; bigger groups help",
    "08_context_growth.png": "Context growth per agent turn",
    "09_grpo_curves.png": "GRPO training curves (v1 and v2 reward)",
    "10_kernelbench_transfer.png": "KernelBench v1 L1 transfer",
    "x1_time_breakdown.png": "Where a GRPO step goes: time breakdown",
    "x2_dr_grpo_noise_mass.png": "Advantage mass from timing-noise groups: GRPO vs. Dr. GRPO",
    "x3_group_explorer.png": "Group explorer: one GRPO v1 group",
    "x4_entropy_proxy.png": "Entropy proxy: mean token log-prob of trained samples",
    "x5_turn_transitions.png": "Base model, v2 feedback: P(next turn | this turn)",
    "x6_speedup_hist.png": "Speedup distribution of PASS turns",
    "deck_v3_cover.png": "Deck v3 cover slide",
}

try:
    from PIL import Image  # type: ignore
except ImportError:  # pragma: no cover - depends on environment
    Image = None


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def title_from_filename(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"^(\d+|x\d+)_", "", stem)
    words = stem.replace("-", " ").replace("_", " ").split()
    return " ".join(words).capitalize() if words else name


def png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def write_if_changed(dest: Path, data: bytes) -> str:
    if dest.exists():
        if dest.read_bytes() == data:
            return "unchanged"
        status = "updated"
    else:
        status = "created"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return status


def render_png(src: Path) -> bytes:
    raw = src.read_bytes()
    w, _ = png_size(raw)
    if Image is None or w <= MAX_WIDTH:
        return raw
    with Image.open(io.BytesIO(raw)) as im:
        im.load()
        h = round(im.height * MAX_WIDTH / im.width)
        out = im.resize((MAX_WIDTH, h), Image.LANCZOS)
        buf = io.BytesIO()
        out.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def collect_figures() -> dict[str, Path]:
    figures: dict[str, Path] = {}
    for d in FIGURE_DIRS:
        src_dir = ROOT / d
        if not src_dir.is_dir():
            die(f"missing figure directory {d}")
        for p in sorted(src_dir.glob("*.png")):
            if p.name in figures:
                die(f"filename collision: {p.relative_to(ROOT)} vs "
                    f"{figures[p.name].relative_to(ROOT)}; refusing to overwrite")
            figures[p.name] = p
    for name, rel in EXTRA_FIGURES.items():
        p = ROOT / rel
        if not p.is_file():
            die(f"missing figure source {rel}")
        if name in figures:
            die(f"filename collision: {rel} vs {figures[name].relative_to(ROOT)}")
        figures[name] = p
    return figures


def main() -> int:
    rows: list[tuple[str, str, str, str]] = []  # kind, file, size, status
    changes = 0

    # --- data ---------------------------------------------------------------
    for out_name, rel in DATA_SOURCES.items():
        src = ROOT / rel
        if not src.is_file():
            die(f"missing data source {rel}")
        try:
            obj = json.loads(src.read_text())
        except json.JSONDecodeError as e:
            die(f"{rel} is not valid JSON: {e}")
        data = (json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        status = write_if_changed(DATA_OUT / out_name, data)
        changes += status != "unchanged"
        rows.append(("data", f"data/{out_name}", f"{len(data) / 1024:.1f} KB", status))

    # --- figures ------------------------------------------------------------
    figures = collect_figures()
    index = []
    for name, src in figures.items():
        data = render_png(src)
        w, h = png_size(data)
        status = write_if_changed(FIG_OUT / name, data)
        changes += status != "unchanged"
        rows.append(("figure", f"figures/{name}", f"{len(data) / 1024:.1f} KB {w}x{h}", status))
        index.append({
            "file": name,
            "source_path": src.relative_to(ROOT).as_posix(),
            "width": w,
            "height": h,
            "title": TITLES.get(name) or title_from_filename(name),
        })

    expected = set(figures) | {"index.json"}
    stale = sorted(p.name for p in FIG_OUT.iterdir() if p.name not in expected)
    if stale:
        print(f"warning: files in docs/assets/figures not produced by this script: {stale}",
              file=sys.stderr)

    idx_bytes = (json.dumps(index, indent=2, ensure_ascii=False) + "\n").encode()
    status = write_if_changed(FIG_OUT / "index.json", idx_bytes)
    changes += status != "unchanged"
    rows.append(("index", "figures/index.json", f"{len(idx_bytes) / 1024:.1f} KB", status))

    # --- summary ------------------------------------------------------------
    widths = [max(len(r[i]) for r in rows + [("kind", "file", "size", "status")]) for i in range(4)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format("kind", "file", "size", "status"))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    total = sum(p.stat().st_size for p in DOCS_ASSETS.rglob("*") if p.is_file())
    print(f"\n{len(rows)} files, total docs/assets size {total / 1024 / 1024:.2f} MB"
          f"{'' if Image else ' (Pillow not installed: PNGs copied as-is)'}")
    print("no changes" if changes == 0 else f"{changes} file(s) written")
    if total > SIZE_BUDGET:
        print(f"warning: docs/assets exceeds {SIZE_BUDGET // 1024 // 1024} MB budget", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
