"""015 -- Every learning-trial method on every session (user request 2026-09-25: "For all these
methods, show more examples"). One figure per cohort; each panel = one session: smoothed whisker
curve (80% band), FA at real times, D = whisker - FA, raw whisker licks, and a vertical line per
method (values from artifacts/013_learning_trials_all_methods.csv). Sessions sorted by how much the
methods disagree (spread of defined LTs), sessions where no method finds a learning event last.
A strip under each panel marks each method's LT on its own row, so coinciding lines stay readable.
Outputs: exploratory-analyses/015_all_methods_Rplus.png, 015_all_methods_Rminus.png
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
METHODS = [("L0 stored", "#d62728"), ("L1 stored rule, exact", "#ff9896"), ("L2 stored rule, smooth", "#e377c2"),
           ("L3 sustained prob.", "#bcbd22"), ("L5 whisker CP", "#7f7f7f"), ("L6 joint CP", "#1f77b4"),
           ("L6 lenient", "#9ecae1"), ("L5w lenient (R+)", "#8c564b"), ("L7 half-way", "#ff7f0e"),
           ("L8 fixed margin", "#17becf"), ("lenient cascade + clean gate", "#000000")]
SHORT = {"L0 stored": "L0", "L1 stored rule, exact": "L1", "L2 stored rule, smooth": "L2", "L3 sustained prob.": "L3",
         "L5 whisker CP": "L5", "L6 joint CP": "L6", "L6 lenient": "L6len", "L5w lenient (R+)": "L5w",
         "L7 half-way": "L7", "L8 fixed margin": "L8", "lenient cascade + clean gate": "gate"}


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    tab = pd.read_csv(ART / "013_learning_trials_all_methods.csv")
    names = [m for m, _ in METHODS]
    vals = tab[names]
    tab["n_def"] = vals.notna().sum(1)
    tab["spread"] = vals.max(1) - vals.min(1)
    for rg in ("R+", "R-"):
        g = tab[tab.reward_group == rg].copy()
        meths = [(m, c) for m, c in METHODS if not (rg == "R-" and m == "L5w lenient (R+)")]
        g["order"] = np.where(g.n_def <= 1, 1e6, g.spread.fillna(0))
        g = g.sort_values(["order", "session_id"], ascending=[False, True])
        g = pd.concat([g[g.order < 1e6], g[g.order >= 1e6]])
        ncol = 5
        nrow = int(np.ceil(len(g) / ncol))
        fig = plt.figure(figsize=(4.4 * ncol, 3.25 * nrow + 1.0))
        gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[3, 1.15] * nrow, hspace=0.55, wspace=0.18)
        for i, row in enumerate(g.itertuples(index=False)):
            r, c = divmod(i, ncol)
            ax = fig.add_subplot(gs[2 * r, c])
            axm = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
            sid = row.session_id
            d, e = inputs[sid], curves[sid]["eb"]
            o = d["w_outcomes"]
            x = np.arange(len(o))
            col = COHORT_COLOR[rg]
            ax.fill_between(x, e["p_low80"], e["p_high80"], color=col, alpha=0.18, lw=0)
            ax.plot(x, e["p_mean"], color=col, lw=1.3)
            ax.plot(x, d["stored"]["p_mean"], color=col, lw=0.4, alpha=0.5)
            ax.plot(x, e["fa_time"], color="#555555", lw=1, ls="--")
            ax.plot(x, e["p_mean"] - e["fa_time"], color="#8c564b", lw=1, ls="-.")
            ax.scatter(x, np.where(o == 1, 1.08, -0.28), s=1.5, color="k", marker="|")
            rowd = dict(zip(tab.columns, row))
            for k, (m, cc) in enumerate(meths):
                v = rowd[m]
                if not pd.isna(v):
                    ax.axvline(v, color=cc, lw=1.0, alpha=0.85)
                    axm.plot([v, v], [k - 0.4, k + 0.4], color=cc, lw=2.2)
                    axm.plot(v, k, marker="o", ms=2.5, color=cc)
            ax.set_ylim(-0.35, 1.15)
            ax.set_xlim(-1, len(o))
            ax.tick_params(labelsize=6, labelbottom=False)
            nd = sum(not pd.isna(rowd[m]) for m, _ in meths)
            sp = rowd["spread"]
            ax.set_title(f"{sid[:5]} {rowd['learning_category']} | {rowd['L6 category']}\n{nd}/{len(meths)} methods define an LT"
                         + (f", spread {sp:.0f}" if nd > 1 else ""), fontsize=7)
            axm.set_yticks(range(len(meths)), [SHORT[m] for m, _ in meths], fontsize=5)
            axm.set_ylim(-0.7, len(meths) - 0.3)
            axm.invert_yaxis()
            axm.tick_params(axis="x", labelsize=6)
            axm.grid(axis="y", color="#eeeeee", lw=0.5)
        handles = [plt.Line2D([], [], color=cc, lw=2.5, label=m) for m, cc in meths]
        handles += [plt.Line2D([], [], color=COHORT_COLOR[rg], lw=1.5, label="whisker (smoothed, 80%)"),
                    plt.Line2D([], [], color="#555555", ls="--", label="FA at real times"),
                    plt.Line2D([], [], color="#8c564b", ls="-.", label="D = whisker - FA")]
        fig.legend(handles=handles, loc="upper center", ncol=7, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"{rg}: every learning-trial method on every session (n = {len(g)}). Sorted by disagreement between "
                     f"methods (largest first); sessions where at most one method finds a learning event at the end. "
                     f"Strip below each panel: one row per method.", fontsize=11, y=1.0 - 0.9 / (3.25 * nrow + 1.0))
        fig.savefig(HERE / f"015_all_methods_{rg.replace('+', 'plus').replace('-', 'minus')}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(rg, len(g), "sessions")


if __name__ == "__main__":
    main()
