"""035b -- Figures for 035 (single joint change point, effect-size floor delta, circular-shift null).
  035_coverage_sweep.{png,pdf}: learners (circular-shift p < 0.05) per delta, stacked by learning category, per cohort;
     the within-type shuffle (stationary null) at delta 0 for comparison; median LT per delta.
  035_review_<cohort>.png: every mouse -- whisker (cohort colour) and FA (grey) HMM curves (sigma = 1, display only); LT for
     each delta (colour = delta; solid = learner, dotted = not significant); title: circular-shift p per delta.
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/035b_delta_figures.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE))
import lt_lib as L  # noqa: E402

COL = {"R+": "#00B400", "R-": "#C800C8"}
CAT_COL = {"good": "#1a9850", "moderate": "#fee08b", "bad": "#d73027", "NA": "0.75"}
DCOL = {0.0: "#9ecae1", 0.1: "#4292c6", 0.2: "#08519c", 0.3: "#08306b"}
FS = 6.5


def main():
    plt.rcParams.update({"font.family": "Arial", "axes.spines.top": False, "axes.spines.right": False, "font.size": FS})
    R = pd.read_csv(ART / "035_single_cp_delta.csv")
    R["cat"] = R.learning_category.fillna("NA")
    inp = pickle.load(open(ART / "028_chain_all" / "001_inputs.pkl", "rb"))
    deltas = sorted(R.delta.unique())
    fig, axes = plt.subplots(1, 3, figsize=(8.27, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.2, wspace=0.4)
    for ax, rg in zip(axes[:2], ("R+", "R-")):
        g = R[R.reward_group == rg]
        ntot = g[g.delta == deltas[0]].cat.value_counts()
        labels = ["shuffle\nδ=0"] + [f"δ={d:g}" for d in deltas]
        for x, lab in enumerate(labels):
            sel = (g[(g.delta == 0) & (g.p_shuffle < 0.05)] if x == 0 else g[(g.delta == deltas[x - 1]) & g.learner])
            bottom = 0
            for c in ("good", "moderate", "bad", "NA"):
                v = int((sel.cat == c).sum())
                if v:
                    ax.bar(x, v, bottom=bottom, color=CAT_COL[c], width=0.65)
                    ax.text(x, bottom + v / 2, str(v), ha="center", va="center", fontsize=FS - 1)
                    bottom += v
            ax.text(x, bottom + 0.6, str(bottom), ha="center", fontsize=FS - 0.5, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=FS - 0.5)
        ax.set_ylabel("mice with an LT")
        ax.set_title(f"{'R+' if rg == 'R+' else 'R−'} (n = {int(ntot.sum())}: " +
                     ", ".join(f"{c} {int(ntot.get(c, 0))}" for c in ("good", "moderate", "bad", "NA")) + ")",
                     color=COL[rg], fontsize=FS)
    ax = axes[2]
    for rg in ("R+", "R-"):
        med = [R[(R.reward_group == rg) & (R.delta == d) & R.learner].LT.median() for d in deltas]
        ax.plot(deltas, med, "o-", color=COL[rg], ms=3, label=f"{'R+' if rg == 'R+' else 'R−'}")
    ax.set_xlabel("effect-size floor δ")
    ax.set_ylabel("median LT of learners (whisker trial)")
    ax.legend(frameon=False)
    fig.suptitle("Single learning-trial definition: joint whisker + FA change point (optional lapse, no start gates), "
                 "change ≥ δ in discrimination, circular-shift null (whisker vs FA alignment), p < 0.05.\nColours: learning "
                 "category (green good, yellow moderate, red bad, grey NA); first bar: within-type shuffle (stationary null).",
                 fontsize=FS)
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"035_coverage_sweep.{ext}", dpi=250)
    plt.close(fig)
    for rg in ("R+", "R-"):
        W = R[R.reward_group == rg].pivot_table(index="session_id", columns="delta", values="p_circ")
        order = W[0.2].sort_values().index if 0.2 in W.columns else W.index
        ncol = 6
        nrow = int(np.ceil(len(order) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 1.75 * nrow + 0.5), squeeze=False)
        fig.subplots_adjust(left=0.03, right=0.995, top=1 - 0.45 / (1.75 * nrow + 0.5), bottom=0.02, hspace=0.8, wspace=0.15)
        for ax, sid in zip(axes.flat, order):
            d = inp[sid]
            tw = np.asarray(d["w_start"], float)
            w = L.forward_backward(np.asarray(d["w_outcomes"], int), 1.0)[0] @ L.P_GRID
            f = L.interp_marginals(L.forward_backward(np.asarray(d["n_outcomes"], int), 1.0)[0],
                                   np.asarray(d["n_start"], float), tw) @ L.P_GRID
            x = np.arange(len(w))
            ax.plot(x, f, color="0.45", lw=0.9)
            ax.plot(x, w, color=COL[rg], lw=1.1)
            ps = []
            for k, dl in enumerate(deltas):
                r = R[(R.session_id == sid) & (R.delta == dl)].iloc[0]
                ps.append(f"{r.p_circ:.2f}")
                if np.isfinite(r.LT):
                    yy = 1.06 + 0.06 * k
                    ax.plot([r.LT_ci05, r.LT_ci95], [yy, yy], color=DCOL[dl], lw=1.0 if r.learner else 0.5)
                    ax.plot([r.LT, r.LT], [-0.05, yy], color=DCOL[dl], lw=1.1 if r.learner else 0.6,
                            ls="-" if r.learner else ":")
            ax.set_ylim(-0.08, 1.3)
            ax.set_title(f"{d['mouse_id']} {d['learning_category'] or 'NA'} p " + "/".join(ps), fontsize=5.8)
            ax.tick_params(labelsize=5)
        for ax in axes.flat[len(order):]:
            ax.set_axis_off()
        fig.text(0.03, 0.998, f"{'R+' if rg == 'R+' else 'R−'}: whisker / FA (grey) curves; LT ± 90% CI per effect-size floor "
                 f"δ = {', '.join(f'{d:g}' for d in deltas)} (light → dark blue; solid = circular-shift p < 0.05, dotted = "
                 "not); title: p per δ. Sorted by p at δ = 0.2.", fontsize=7, va="top")
        fig.savefig(HERE / f"035_review_{'Rplus' if rg == 'R+' else 'Rminus'}.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
