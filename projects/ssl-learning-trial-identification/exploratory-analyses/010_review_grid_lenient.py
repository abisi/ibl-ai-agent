"""010 -- Per-session review grid for the less conservative learning trials
(007): every session, re-estimated whisker curve (80% CI) + FA at real
times + stored curve (thin), stored LT (red), lenient LT (orange: solid =
passes the clean-separation gate, dashed = fails), 20-trial windows used by
the gate shaded. Title: rule that produced the LT, gate P, whisker hit rate
before/after, category. Sorted: clean learners, gated-out, no LT.
Outputs: exploratory-analyses/010_review_grid_lenient_<Rplus|Rminus>.png
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


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    df = pd.read_csv(ART / "007_learning_trials_v2.csv")
    df["status"] = np.where(df.lt_lenient_clean.notna(), "1 clean learner",
                            np.where(df.lt_lenient.notna(), "2 LT fails clean gate", "3 no LT"))
    for rg in ("R+", "R-"):
        g = df[df.reward_group == rg].sort_values(["status", "lt_lenient_source", "session_id"])
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.35 * nrow), constrained_layout=True)
        for ax, row in zip(axes.flat, g.itertuples()):
            d, e = inputs[row.session_id], curves[row.session_id]["eb"]
            o = d["w_outcomes"]
            x = np.arange(len(o))
            col = COHORT_COLOR[rg]
            ax.fill_between(x, e["p_low80"], e["p_high80"], color=col, alpha=0.2, lw=0)
            ax.plot(x, e["p_mean"], color=col, lw=1.4)
            ax.plot(x, d["stored"]["p_mean"], color=col, lw=0.5, alpha=0.45)
            ax.plot(x, e["fa_time"], color="#444444", lw=1, ls="--")
            ax.scatter(x, np.where(o == 1, 1.07, -0.07), s=2, color="k", marker="|")
            if not pd.isna(row.L0_stored):
                ax.axvline(row.L0_stored, color="#d62728", lw=1.2)
            if not pd.isna(row.lt_lenient):
                clean = not pd.isna(row.lt_lenient_clean)
                ax.axvspan(max(0, row.lt_lenient - 20), row.lt_lenient, color="#ffd27f", alpha=0.25, lw=0)
                ax.axvspan(row.lt_lenient, min(len(o), row.lt_lenient + 20), color="#ff7f0e", alpha=0.15, lw=0)
                ax.axvline(row.lt_lenient, color="#ff7f0e", lw=1.8, ls="-" if clean else "--")
            ax.set_ylim(-0.12, 1.12)
            ax.set_title(f"{row.session_id[:5]} {row.learning_category} | {row.status[2:]}\n"
                         f"stored {row.L0_stored:.0f} -> new {row.lt_lenient:.0f} [{row.lt_lenient_source}] "
                         f"P={row.lt_lenient_sep_p:.2f} hit {row.lt_lenient_hit_pre20:.2f}->{row.lt_lenient_hit_post20:.2f}"
                         .replace("nan", "-"), fontsize=6.2)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(g):]:
            ax.axis("off")
        fig.suptitle(f"{rg}: less conservative learning trials. Thick = re-estimated whisker curve (80% CI), thin = stored, "
                     f"dashed grey = FA at real times. Red = stored LT; orange = new lenient LT (solid = passes clean-separation "
                     f"gate, dashed = fails); shaded = 20-trial windows the gate compares", fontsize=9)
        fig.savefig(HERE / f"010_review_grid_lenient_{rg.replace('+', 'plus').replace('-', 'minus')}.png", dpi=150)
    print(pd.crosstab(df.reward_group, df.status))


if __name__ == "__main__":
    main()
