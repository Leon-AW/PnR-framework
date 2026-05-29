#!/usr/bin/env python3
"""Generate the joint edit-success / forgetting Pareto figure for the thesis.

Three panels (CounterFact, SituatedQA, AIT QM). Each plots the domain's primary
edit-success rate (y) against the system-level D_control forgetting rate (x). The
desired region is the top-left: high edit success AND no forgetting. Only the two
routing systems (PnR, Parallel Orchestrator) occupy it across all three domains.

Numbers are the values reported in the thesis main results table (tab:r1_main):
CF ESR (exact match), SQA ESR (exact match), QM strict containment ESR, and the
D_control forgetting rate FR. Source: scripts/summarize_results.py.

Output: docs/files/images/pareto_frontier.png
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# system -> (FR, CF ESR, SQA ESR, QM strict ESR), all in %
DATA = {
    "Frozen Base":           (0.4,  0.0,  0.0,  1.2),
    "X-LoRA":                (74.8, 0.0,  0.4,  0.0),
    "Monolithic LoRA":       (100.0, 0.0, 20.1, 23.4),
    "LoRA + RAG":            (99.5, 7.7,  29.2, 21.4),
    "RECIPE":                (47.8, 0.3,  19.8, 50.0),
    "Parallel Orchestrator": (0.6,  33.5, 86.6, 57.0),
    "PnR":                   (0.6,  30.4, 86.4, 62.4),
}

# marker / colour styling; the two routing systems are emphasised
STYLE = {
    "Frozen Base":           dict(c="#9e9e9e", marker="o", s=70,  z=3),
    "X-LoRA":                dict(c="#8c564b", marker="o", s=70,  z=3),
    "Monolithic LoRA":       dict(c="#7f7f7f", marker="o", s=70,  z=3),
    "LoRA + RAG":            dict(c="#bcbd22", marker="o", s=70,  z=3),
    "RECIPE":                dict(c="#1f77b4", marker="o", s=70,  z=3),
    "Parallel Orchestrator": dict(c="#ff7f0e", marker="D", s=130, z=5),
    "PnR":                   dict(c="#d62728", marker="*", s=320, z=6),
}

PANELS = [("CounterFact", 1), ("SituatedQA", 2), ("AIT QM", 3)]  # idx into DATA tuple

fig, axes = plt.subplots(1, 3, figsize=(11, 3.7), sharex=True, sharey=True)

for ax, (title, idx) in zip(axes, PANELS):
    # "no forgetting" band (low FR) shaded lightly
    ax.axvspan(-3, 10, color="#2ca02c", alpha=0.07, zorder=0)
    for name, vals in DATA.items():
        fr, esr = vals[0], vals[idx]
        st = STYLE[name]
        ax.scatter(fr, esr, c=st["c"], marker=st["marker"], s=st["s"],
                   edgecolors="black", linewidths=0.6, zorder=st["z"])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 100)
    ax.set_xlabel("Forgetting rate FR (%)  $\\rightarrow$ worse", fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=8)

axes[0].set_ylabel("Edit success rate (%)  $\\rightarrow$ better", fontsize=9)
# annotate the desired corner once
axes[0].annotate("desired:\nhigh edit success,\nno forgetting",
                 xy=(3, 92), xytext=(22, 70), fontsize=7.5, color="#2ca02c",
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))

# shared legend below
handles = [Line2D([0], [0], marker=STYLE[n]["marker"], color="none",
                  markerfacecolor=STYLE[n]["c"], markeredgecolor="black",
                  markersize=(13 if n == "PnR" else 9), label=n)
           for n in DATA]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.06))

fig.tight_layout(rect=(0, 0.06, 1, 1))

out = Path(__file__).resolve().parents[1] / "docs/files/images/pareto_frontier.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
