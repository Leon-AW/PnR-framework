#!/usr/bin/env python3
"""Generate the cumulative update-cost scaling figure for the thesis.

Visualises the structural R2 efficiency claim (Section 6.3.2, tab:eff_update):
adding successive knowledge updates, PnR's per-update cost stays FLAT (it trains
one patch on the new increment only), so its cumulative cost grows LINEARLY in
the number of updates K. A monolithic adapter must retrain on the whole grown
corpus at every update, so its cumulative cost grows QUADRATICALLY (~K^2/2).

Two panels:
  (a) cumulative training steps  -- abstract, deterministic, reproduces Table 6;
  (b) cumulative GPU wall-clock  -- the same curves rescaled by the *measured*
      seconds/step from the A100 cost benchmark, so the y-axis reads in real
      minutes for the "enterprise cost" argument.

Single source of truth
----------------------
The step curves are computed from `benchmark_update_cost.coverage_steps`, the
exact function behind Table 6, using the same 0.25-epoch budget (eff_batch=16)
that produced the measured anchors in Table 5 (53 steps for one 3,340-record
increment; 309 steps for the 19,728-record full corpus = ceil(records/64)).
This reproduces the published endpoints exactly:
    PnR  cumulative @ K=6 = 318    (linear, 53 steps/update)
    Mono cumulative @ K=6 = 1,099  (quadratic), ratio 3.46x = (K+1)/2.

The GPU-time panel rescales each curve by its measured seconds/step, logged in
MLflow as `train_runtime` for the two matched A100 cost-benchmark runs (Table 5):
    PnR patch (1 increment)  : run `clean-boar-159` -> 419 s / 53 steps
    Monolithic (full corpus) : run `capable-auk-759` -> 2,463 s / 309 steps

Output: docs/files/images/update_cost_scaling.png
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from scripts.benchmark_update_cost import coverage_steps  # noqa: E402

# ── geometry matching tab:eff_update (Table 6) ──────────────────────────────
N_INCREMENT = 3340   # records per increment (counterfact_relfam_5.jsonl)
EFF_BATCH = 16       # effective batch size
EPOCHS = 0.25        # per-example exposure budget -> ceil(records/64) steps
K_MAX = 6            # number of successive updates

updates = list(range(1, K_MAX + 1))
inc_steps = coverage_steps(N_INCREMENT, EFF_BATCH, EPOCHS)  # == 53

# PnR: each update trains ONE increment -> flat per-update cost, linear cumulative.
pnr_cum, running = [], 0
for _ in updates:
    running += inc_steps
    pnr_cum.append(running)

# Monolithic: update k retrains the CUMULATIVE corpus (k increments) -> the
# per-update cost grows with k, so the cumulative cost grows quadratically.
mono_cum, running = [], 0
for k in updates:
    running += coverage_steps(k * N_INCREMENT, EFF_BATCH, EPOCHS)
    mono_cum.append(running)

ratio = mono_cum[-1] / pnr_cum[-1]
assert pnr_cum[-1] == 318 and mono_cum[-1] == 1099, (pnr_cum, mono_cum)

# Measured seconds/step from the matched A100 cost benchmark (MLflow train_runtime,
# Table 5). Each strategy is rescaled by its own measured rate -> real GPU minutes.
PNR_S_PER_STEP = 419 / 53     # run clean-boar-159: 419 s over 53 steps
MONO_S_PER_STEP = 2463 / 309  # run capable-auk-759: 2,463 s over 309 steps
pnr_min = [s * PNR_S_PER_STEP / 60 for s in pnr_cum]
mono_min = [s * MONO_S_PER_STEP / 60 for s in mono_cum]
ratio_t = mono_min[-1] / pnr_min[-1]

# ── figure (styling consistent with plot_pareto.py) ─────────────────────────
PNR_C, MONO_C = "#d62728", "#7f7f7f"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)

panels = [
    (axes[0], pnr_cum, mono_cum, ratio, "Cumulative training steps",
     "{:,}".format, "(a) Training steps"),
    (axes[1], pnr_min, mono_min, ratio_t, "Cumulative GPU time (minutes)",
     lambda v: f"{v:.0f} min", "(b) Measured GPU wall-clock (A100)"),
]

for ax, pnr_y, mono_y, rr, ylabel, fmt, title in panels:
    ax.fill_between(updates, pnr_y, mono_y, color=MONO_C, alpha=0.10, zorder=1)
    ax.plot(updates, mono_y, color=MONO_C, marker="o", markersize=7,
            markeredgecolor="black", markeredgewidth=0.6, linewidth=2.2,
            label="Monolithic (retrain full corpus)  $O(K^2)$", zorder=3)
    ax.plot(updates, pnr_y, color=PNR_C, marker="*", markersize=15,
            markeredgecolor="black", markeredgewidth=0.6, linewidth=2.2,
            label="PnR (one patch per update)  $O(K)$", zorder=4)

    ax.annotate(fmt(mono_y[-1]), xy=(K_MAX, mono_y[-1]), xytext=(-2, 9),
                textcoords="offset points", ha="right", fontsize=10,
                fontweight="bold", color=MONO_C)
    ax.annotate(fmt(pnr_y[-1]), xy=(K_MAX, pnr_y[-1]), xytext=(-2, -16),
                textcoords="offset points", ha="right", fontsize=10,
                fontweight="bold", color=PNR_C)
    ax.annotate(f"${rr:.2f}\\times$\nat $K=6$",
                xy=(K_MAX, (pnr_y[-1] + mono_y[-1]) / 2),
                xytext=(K_MAX - 1.55, (pnr_y[-1] + mono_y[-1]) / 2),
                fontsize=10, ha="center", va="center", color="#333333",
                arrowprops=dict(arrowstyle="-[, widthB=2.4", color="#666666", lw=1.1))

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of successive knowledge updates $K$", fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_xlim(0.7, 6.3)
    ax.set_ylim(0, mono_y[-1] * 1.12)
    ax.set_xticks(updates)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.tick_params(labelsize=9)

axes[0].legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.9)

fig.tight_layout()

out = REPO_ROOT / "docs/files/images/update_cost_scaling.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
print(f"PnR cumulative steps : {pnr_cum}")
print(f"Mono cumulative steps: {mono_cum}")
print(f"PnR GPU-min : {[round(v,1) for v in pnr_min]}")
print(f"Mono GPU-min: {[round(v,1) for v in mono_min]}")
print(f"ratio steps @ K={K_MAX}: {ratio:.2f}x | ratio time: {ratio_t:.2f}x")
