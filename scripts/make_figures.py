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
    topo = {"1N": "single node", "one node": "single node", "single node": "single node", "TP2": "TP2"}
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
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=9)
    title(fig, "CUDA graphs helped Qwen, not Llama",
          "Concurrency 1, 1024 input / 256 output tokens. Pilot in eager mode; rerun on the native compiled stack.")
    save(fig, "eager_vs_compiled.png")


def _box(ax, x, y, w, h, text, fc="white", ec=GRID, size=10, weight="normal", color=INK, lw=1.2):
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            weight=weight, color=color, linespacing=1.4)


def _canvas(figsize, xlim, ylim):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def _place(ax, name, cx, y0, width, z=3):
    """Draw figures/assets/<name> centred at cx with its base at y0, keeping its aspect ratio."""
    img = plt.imread(OUT / "assets" / name)
    h = width * img.shape[0] / img.shape[1]
    ax.imshow(img, extent=(cx - width / 2, cx + width / 2, y0, y0 + h), zorder=z, interpolation="lanczos")
    return cx - width / 2, cx + width / 2, y0, y0 + h


def topology():
    from matplotlib.patches import Ellipse, FancyBboxPatch
    W, H = 9.0, 4.8
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    line = dict(color=INK2, lw=1.3, solid_capstyle="round", zorder=1)

    # devices
    lap = _place(ax, "laptop.png", 2.6, 2.86, 1.75)
    sw = _place(ax, "switch.png", 2.6, 2.06, 1.0)
    s1 = _place(ax, "dgx-spark.png", 1.0, 0.74, 1.5)
    s2 = _place(ax, "dgx-spark.png", 4.2, 0.74, 1.5)
    for s in (s1, s2):
        ax.add_patch(Ellipse(((s[0] + s[1]) / 2, s[2] + 0.01), 1.45, 0.07, color=INK, alpha=0.10, lw=0, zorder=2))

    # management network: laptop to switch, switch to both nodes
    bus = 1.78
    ax.plot([2.6, 2.6], [lap[2] + 0.02, sw[3] - 0.02], **line)
    ax.plot([2.6, 2.6], [sw[2] + 0.02, bus], **line)
    ax.plot([1.0, 4.2], [bus, bus], **line)
    for cx, top in ((1.0, s1[3]), (4.2, s2[3])):
        ax.plot([cx, cx], [bus, top - 0.04], **line)

    # cluster link: one cable, two PCIe paths
    cy = 1.02
    for dy in (0.045, -0.045):
        ax.plot([s1[1] - 0.04, s2[0] + 0.04], [cy + dy, cy + dy], color=BLUE, lw=2.4,
                solid_capstyle="butt", zorder=2)
    ax.text(2.6, cy + 0.17, "ConnectX-7, 200 GbE", ha="center", va="bottom", fontsize=10,
            weight="semibold", color=BLUE)
    ax.text(2.6, cy - 0.15, "two PCIe paths\n196 Gb/s combined", ha="center", va="top", fontsize=9,
            color=INK2, linespacing=1.3)

    # device labels
    lx = 3.62
    ax.text(lx, (lap[2] + lap[3]) / 2 + 0.1, "Laptop", fontsize=10.5, weight="semibold", color=INK, va="bottom")
    ax.text(lx, (lap[2] + lap[3]) / 2 + 0.06, "control, SSH, benchmarks", fontsize=9.5, color=INK2, va="top")
    ax.text(lx, (sw[2] + sw[3]) / 2 + 0.06, "Management switch", fontsize=10.5, weight="semibold", color=INK,
            va="bottom")
    ax.text(lx, (sw[2] + sw[3]) / 2 + 0.02, "10 GbE, admin and files", fontsize=9.5, color=INK2, va="top")
    for s, name, role in ((s1, "spark1", "Ray head, serves the API"), (s2, "spark2", "Ray worker")):
        cx = (s[0] + s[1]) / 2
        ax.text(cx, s[2] - 0.1, name, ha="center", va="top", fontsize=10.5, weight="semibold", color=INK)
        ax.text(cx, s[2] - 0.34, role, ha="center", va="top", fontsize=9.5, color=INK2)

    # hardware panel
    px, pw = 5.5, 3.3
    ax.add_patch(FancyBboxPatch((px, 0.16), pw, 3.78, boxstyle="round,pad=0,rounding_size=0.1",
                                fc="#f3f2ee", ec="none", zorder=0))
    ax.text(px + 0.25, 3.68, "The hardware", fontsize=11.5, weight="semibold", color=INK, va="top")
    spec = [
        ("Nodes", "2 x NVIDIA DGX Spark"),
        ("Chip", "GB10 Grace Blackwell"),
        ("Memory", "128 GB unified, 273 GB/s"),
        ("Usable", "about 121 GiB per node"),
        ("Cluster link", "ConnectX-7, 200 GbE cable"),
        ("Management", "10 GbE through a switch"),
        ("Software", "DGX OS 7.6, vLLM 0.29.0"),
    ]
    for i, (k, v) in enumerate(spec):
        y = 3.2 - i * 0.33
        ax.text(px + 0.25, y, k, fontsize=9.5, weight="semibold", color=INK, va="center")
        ax.text(px + 1.28, y, v, fontsize=9.5, color=INK2, va="center")
        if i:
            ax.plot([px + 0.25, px + pw - 0.25], [y + 0.165, y + 0.165], color=GRID, lw=0.8, zorder=1)
    ax.plot([px + 0.27, px + 0.27], [0.4, 0.92], color=BLUE, lw=2.4, solid_capstyle="butt", zorder=2)
    ax.text(px + 0.42, 0.66, "CPU and GPU share one memory pool, so\n"
            "running out stalls the whole node. Every\nlaunch runs under a memory watchdog.",
            fontsize=8.8, color=INK2, va="center", linespacing=1.4, zorder=3)

    ax.text(0.25, H - 0.22, "Cluster topology", fontsize=16, weight="semibold", color=INK, va="top")
    ax.text(0.25, H - 0.6, "Management network through a switch; inference traffic on a direct "
            "ConnectX-7 link.", fontsize=10, color=INK2, va="top")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    save(fig, "topology.png")


def parallelism():
    r = {x["run_id"]: x for x in rows(RES / "pilot" / "runs.csv")}
    pp, tp = float(r["L-PP-1"]["output_tok_s"]), float(r["L-TP-1"]["output_tok_s"])
    fig, ax = _canvas((7, 4.4), (0, 10), (0, 6.4))
    fig.subplots_adjust(top=0.86, left=0.01, right=0.99, bottom=0.01)
    t1, t2 = (BLUE + "2e"), (AQUA + "2e")
    # pipeline parallelism
    ax.text(0.3, 5.75, "Pipeline parallelism: split by depth", fontsize=10, weight="semibold", color=INK)
    _box(ax, 1.2, 4.1, 2.6, 1.05, "Layers 1 to 40", fc=t1, ec=BLUE, size=9)
    _box(ax, 1.2, 2.95, 2.6, 1.05, "Layers 41 to 80", fc=t2, ec=AQUA, size=9)
    ax.text(1.05, 4.62, "spark1", ha="right", va="center", fontsize=9, color=INK2)
    ax.text(1.05, 3.47, "spark2", ha="right", va="center", fontsize=9, color=INK2)
    ax.annotate("", xy=(4.2, 2.95), xytext=(4.2, 5.15), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    ax.text(4.32, 4.05, "then", fontsize=8.5, color=INK2, va="center", style="italic")
    ax.text(2.5, 2.55, "Half the weights per node,\nread one stage after the other", ha="center", va="top",
            fontsize=8.5, color=INK2, linespacing=1.35)
    ax.text(2.5, 1.2, f"{pp:.2f} tok/s", ha="center", fontsize=17, weight="semibold", color=INK)
    ax.text(2.5, 0.75, "Llama 3.3 70B, one stream", ha="center", fontsize=8.5, color=INK2)
    # tensor parallelism
    ax.text(5.3, 5.75, "Tensor parallelism: split every layer", fontsize=10, weight="semibold", color=INK)
    ax.text(6.95, 5.3, "spark1", ha="center", fontsize=9, color=INK2)
    ax.text(8.45, 5.3, "spark2", ha="center", fontsize=9, color=INK2)
    for i, lab in enumerate(["Layer 1", "Layer 2", "Layer 3", "Layer 80"]):
        y = 4.6 - i * 0.55 - (0.25 if i == 3 else 0)
        _box(ax, 6.2, y, 1.48, 0.45, "", fc=t1, ec=BLUE, lw=1)
        _box(ax, 7.72, y, 1.48, 0.45, "", fc=t2, ec=AQUA, lw=1)
        ax.annotate("", xy=(7.45, y + 0.225), xytext=(7.95, y + 0.225),
                    arrowprops=dict(arrowstyle="<|-|>", color=INK2, lw=0.9, mutation_scale=8))
        ax.text(6.05, y + 0.225, lab, ha="right", va="center", fontsize=8.5, color=INK2)
    ax.text(7.7, 3.34, "· · ·", ha="center", va="center", fontsize=10, color=INK2)
    ax.text(7.7, 2.55, "Half of every layer per node,\nboth read at the same time", ha="center", va="top",
            fontsize=8.5, color=INK2, linespacing=1.35)
    ax.text(7.7, 1.2, f"{tp:.2f} tok/s", ha="center", fontsize=17, weight="semibold", color=INK)
    ax.text(7.7, 0.75, "same model, same hardware", ha="center", fontsize=8.5, color=INK2)
    title(fig, "Two ways to split a model across two nodes",
          "Decode speed is set by how many weight bytes each node streams per token, and when.")
    save(fig, "parallelism.png")


def frontier():
    path = RES / "sweep" / "sweep.csv"
    if not path.exists():
        print("skip frontier.png (no results/sweep/sweep.csv yet)")
        return
    r = [x for x in rows(path) if x["valid"] == "1"]
    style = {"1N": ("One node", AQUA, "o"), "DP2": ("Two servers (DP2)", INK2, "s"),
             "TP2": ("One server on both (TP2)", BLUE, "D")}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(top=0.78, left=0.11, right=0.97, bottom=0.13)
    ax.grid(axis="x")
    for cfg in ["1N", "DP2", "TP2"]:
        pts = sorted((int(x["concurrency"]), float(x["output_tok_s"]), float(x["median_tpot_ms"]))
                     for x in r if x["config"] == cfg)
        name, color, marker = style[cfg]
        ax.plot([p[1] for p in pts], [p[2] for p in pts], color=color, marker=marker, ms=5, lw=1.8, label=name)
        for c, x, y in pts:
            if c in (1, 4, 64, 256) or (cfg == "1N" and c == 128):
                ax.annotate(f"{c}", (x, y), xytext=(5, -3), textcoords="offset points", fontsize=8, color=color)
    ax.axhline(150, color=GRID, lw=1.2, ls="--", zorder=0)
    ax.text(330, 118, "150 ms per token", fontsize=8, color=INK2, ha="right", va="top")
    ax.set_xlabel("Total output tokens per second")
    ax.set_ylabel("Median time per output token (ms)")
    ax.set_ylim(0, 1350)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=9)
    title(fig, "One server across both nodes matches or beats two",
          "Qwen3.8-27B BF16, 1024 input / 256 output tokens. Labels are concurrent requests.")
    save(fig, "frontier.png")


def govreport():
    path = RES / "sweep" / "govreport.csv"
    if not path.exists():
        print("skip govreport.png (no results/sweep/govreport.csv yet)")
        return
    r = {x["topology"]: x for x in rows(path)}
    keys = [("1N", "One node", AQUA), ("TP2", "TP2, both nodes", BLUE)]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.4))
    fig.subplots_adjust(top=0.74, left=0.06, right=0.98, bottom=0.12, wspace=0.45)
    for ax, (field, label, fmt) in zip(axes, [("reports_per_hour", "Reports per hour", "{:.0f}"),
                                               ("median_ttft_s", "Median time to first token (s)", "{:.1f}"),
                                               ("rougeL", "ROUGE-L", "{:.1f}")]):
        vals = [float(r[k][field]) for k, _, _ in keys]
        bars = ax.bar([n for _, n, _ in keys], vals, 0.6, color=[c for _, _, c in keys])
        label_bars(ax, bars, lambda v, f=fmt: f.format(v))
        ax.set_title(label, fontsize=9.5, color=INK2, loc="left")
        ax.set_ylim(0, max(vals) * 1.2)
        ax.tick_params(axis="x", labelsize=8.5)
    title(fig, "Long reports: the second node adds 1.78x throughput at equal quality",
          "100 GovReport documents (median 9,033 tokens), Qwen3.8-27B BF16, 16 at a time.")
    save(fig, "govreport.png")


if __name__ == "__main__":
    topology()
    parallelism()
    interconnect()
    paired_panels("tp_vs_pp.png", ["L-TP", "L-PP"], ["TP2", "PP2"], [BLUE, AQUA],
                  "Tensor parallelism beats pipeline parallelism across the link",
                  "Llama 3.3 70B split over two nodes, 1024 input / 256 output tokens, eager mode.")
    paired_panels("one_vs_two.png", ["Q-1N", "Q-2N"], ["One node", "Two nodes (TP2)"], [AQUA, BLUE],
                  "A second node helps even when the model fits on one",
                  "Qwen3.8-27B, 1024 input / 256 output tokens, eager mode.")
    weight_streaming()
    eager_vs_compiled()
    frontier()
    govreport()
