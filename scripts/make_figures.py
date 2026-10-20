#!/usr/bin/env python3
"""Regenerate every figure in figures/ from the CSVs in results/.

    python3 scripts/make_figures.py

Needs matplotlib. The eager-vs-compiled figure is drawn only when
results/closing/pilot_rerun.csv exists.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

# Validated categorical pair (blue, aqua) on a light surface. Aqua sits below
# 3:1 contrast, so every bar carries a visible value label.
BLUE, AQUA = "#2a78d6", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
})


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def title(fig, text, sub=None):
    fig.text(0.02, 0.97, text, ha="left", va="top", fontsize=13, weight="semibold", color=INK)
    if sub:
        fig.text(0.02, 0.905, sub, ha="left", va="top", fontsize=9.5, color=INK2)


def label_bars(ax, bars, fmt):
    for b in bars:
        ax.annotate(fmt(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=INK)


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def interconnect():
    r = rows(RES / "pilot" / "interconnect.csv")
    get = {(x["configuration"], x["path_mtu_bytes"]): float(x["aggregate_gbps"]) for x in r}
    cats = ["single path", "dual path"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(top=0.78, left=0.09, right=0.98, bottom=0.12)
    w, gap = 0.36, 0.02
    xs = range(len(cats))
    b1 = ax.bar([x - (w + gap) / 2 for x in xs], [get[(c, "1024")] for c in cats], w,
                color=AQUA, label="Interface MTU 1500 (RoCE path MTU 1024)")
    b2 = ax.bar([x + (w + gap) / 2 for x in xs], [get[(c, "4096")] for c in cats], w,
                color=BLUE, label="Interface MTU 9000 (RoCE path MTU 4096)")
    label_bars(ax, list(b1) + list(b2), lambda v: f"{v:.1f}")
    ax.axhline(200, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.5, 202, "200 Gb/s line rate", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(list(xs), ["One PCIe path", "Both PCIe paths"])
    ax.set_ylim(0, 260)
    ax.set_yticks(range(0, 201, 50))
    ax.set_ylabel("RDMA write bandwidth (Gb/s)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    title(fig, "Line rate needs both PCIe paths",
          "One path caps at 112 Gb/s at any MTU. Jumbo frames only help once both paths share the wire.")
    save(fig, "interconnect.png")


def paired_panels(name, run_ids, labels, colors, head, sub):
    r = {x["run_id"]: x for x in rows(RES / "pilot" / "runs.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.0))
    fig.subplots_adjust(top=0.76, left=0.09, right=0.98, bottom=0.1, wspace=0.3)
    for ax, conc in zip(axes, ("1", "32")):
        vals = [float(r[f"{rid}-{conc}"]["output_tok_s"]) for rid in run_ids]
        bars = ax.bar(range(len(vals)), vals, 0.6, color=colors)
        label_bars(ax, bars, lambda v: f"{v:.2f}" if v < 10 else f"{v:.1f}")
        ax.set_xticks(range(len(vals)), labels)
        ax.set_ylim(0, max(vals) * 1.22)
        ratio = vals[1] / vals[0] if run_ids[0].endswith("1N") else vals[0] / vals[1]
        ax.set_title(f"Concurrency {conc}:  {ratio:.2f}x", fontsize=10.5, color=INK, loc="left")
    axes[0].set_ylabel("Output tokens per second")
    title(fig, head, sub)
    save(fig, name)


def weight_streaming():
    r = rows(RES / "pilot" / "weight_streaming.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    fig.subplots_adjust(top=0.74, left=0.22, right=0.97, bottom=0.14)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ys = list(range(len(r)))[::-1]
    for y, x in zip(ys, r):
        mem, res = float(x["predicted_memory_ms"]), float(x["residual_ms"])
        ax.barh(y, mem, 0.55, color=BLUE)
        ax.barh(y, res, 0.55, left=mem + 2, color=AQUA)
        ax.text(mem + res + 8, y, f"+{res:.0f} ms", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [x["config"] for x in r])
    ax.set_xlim(0, 680)
    ax.set_xlabel("Median time per output token at concurrency 1 (ms)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label="Weight streaming at 273 GB/s (predicted)"),
                       Patch(color=AQUA, label="Everything else (measured minus predicted)")],
              loc="lower left", bbox_to_anchor=(-0.27, 1.0), ncol=2, frameon=False, fontsize=9,
              handlelength=1.0, columnspacing=1.5)
    title(fig, "Eager mode adds about 30 ms per forward pass",
          "Same residual across models and node counts; pipeline parallelism pays it twice.")
    save(fig, "weight_streaming.png")


def eager_vs_compiled():
    path = RES / "closing" / "pilot_rerun.csv"
    if not path.exists():
        print("skip eager_vs_compiled.png (no results/closing/pilot_rerun.csv yet)")
        return
    pilot = {(x["model"], x["config"], x["concurrency"]): x for x in rows(RES / "pilot" / "runs.csv")}
    topo = {"one node": "single node", "single node": "single node", "TP2": "TP2"}
    pairs = []
    for x in rows(path):
        if x["concurrency"] != "1":
            continue
        key = (x["model"], topo.get(x["topology"], x["topology"]), "1")
        if key in pilot:
            short = "Qwen 27B" if "Qwen" in x["model"] else "Llama 70B"
            pairs.append((f"{short} {key[1]}", float(pilot[key]["median_tpot_ms"]), float(x["median_tpot_ms"])))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.subplots_adjust(top=0.76, left=0.09, right=0.98, bottom=0.12)
    w, gap = 0.36, 0.02
    xs = range(len(pairs))
    b1 = ax.bar([x - (w + gap) / 2 for x in xs], [p[1] for p in pairs], w, color=AQUA, label="Pilot, eager")
    b2 = ax.bar([x + (w + gap) / 2 for x in xs], [p[2] for p in pairs], w, color=BLUE, label="Compiled, CUDA graphs")
    label_bars(ax, list(b1) + list(b2), lambda v: f"{v:.0f}")
    ax.set_xticks(list(xs), [p[0] for p in pairs])
    ax.set_ylabel("Median time per output token (ms)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    title(fig, "Removing the eager-mode constant", "Concurrency 1, 1024 input / 256 output tokens.")
    save(fig, "eager_vs_compiled.png")


if __name__ == "__main__":
    interconnect()
    paired_panels("tp_vs_pp.png", ["L-TP", "L-PP"], ["TP2", "PP2"], [BLUE, AQUA],
                  "Tensor parallelism beats pipeline parallelism across the link",
                  "Llama 3.3 70B split over two nodes, 1024 input / 256 output tokens, eager mode.")
    paired_panels("one_vs_two.png", ["Q-1N", "Q-2N"], ["One node", "Two nodes (TP2)"], [AQUA, BLUE],
                  "A second node helps even when the model fits on one",
                  "Qwen3.8-27B, 1024 input / 256 output tokens, eager mode.")
    weight_streaming()
    eager_vs_compiled()
