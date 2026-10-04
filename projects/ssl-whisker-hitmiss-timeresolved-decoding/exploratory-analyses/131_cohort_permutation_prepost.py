"""131 -- Cohort-label permutation test on the pre/post decoding change (user 2026-10-02: "run a permutation test (many shuffles)
on group labels to compare the pre/post effect change, plotting the null distribution and the real effect -- interaction
between direction of change and cohort, i.e. difference of difference").
Per mouse (one learning session each), from 126 (every-whisker-trial placebo run, perf != 6, A1-trimmed, separate
size-matched decoders before / after the split):
  delta   = acc_post - acc_pre at the split (direction and size of the change);
  excess  = delta - mean delta over the session's placebo splits >= 10 whisker trials away.
Statistic = mean(R+) - mean(R-) of delta (difference of differences: cohort x pre/post interaction), and the same for
excess. Null: R+ / R- labels shuffled across mice N_PERM times (cohort sizes kept); two-sided p = P(|null| >= |real|).
Splits (LT variants): session half, L6x forced (every mouse), L6x relaxed (learners), L5 whisker CP, L6 lenient, L0 stored.
Decodings: hit vs miss 5-50 / 5-100 ms (stimulus), whisker vs auditory -100..0 ms pre-lick.
Outputs: figures/131_cohort_permutation_{delta,excess}.{pdf,png,svg}, 131_cohort_permutation.csv
Run (haas): python 131_cohort_permutation_prepost.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
COL = {"R+": "#00B400", "R-": "#C800C8"}
VARIANTS = ["half", "L6x forced", "L6x relaxed", "L5 whisker CP", "L6 lenient", "L0 stored"]
MEASURES = [("hitmiss", "5-50ms", "hit/miss 5-50 ms"), ("hitmiss", "5-100ms", "hit/miss 5-100 ms"),
            ("modality_lick", "-100-0ms", "whisker vs auditory pre-lick")]
N_PERM = 20000
FS = 6.5


def main():
    P = pd.read_csv(OUT / f"126_lt_variants_per_session{os.environ.get('SSL_PLACEBO_TAG', '_step1')}.csv")
    rng = np.random.default_rng(0)
    rows, nulls = [], {}
    for stat in ("delta", "excess"):
        for dec, w, _ in MEASURES:
            for v in VARIANTS:
                g = P[(P.decoding == dec) & (P.window == w) & (P.variant == v)].dropna(subset=[stat])
                x, lab = g[stat].to_numpy(), (g.reward_group == "R+").to_numpy()
                if lab.sum() < 3 or (~lab).sum() < 3:
                    continue
                real = x[lab].mean() - x[~lab].mean()
                null = np.empty(N_PERM)
                for i in range(N_PERM):
                    p = rng.permutation(lab)
                    null[i] = x[p].mean() - x[~p].mean()
                pv = float((np.sum(np.abs(null) >= abs(real)) + 1) / (N_PERM + 1))
                nulls[(stat, dec, w, v)] = (null, real, pv)
                rows.append(dict(stat=stat, decoding=dec, window=w, variant=v, n_rplus=int(lab.sum()), n_rminus=int((~lab).sum()),
                                 mean_rplus=x[lab].mean(), mean_rminus=x[~lab].mean(), diff_of_diff=real, p_perm=pv,
                                 null_sd=null.std()))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "131_cohort_permutation.csv", index=False)
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    for stat in ("delta", "excess"):
        fig, axes = plt.subplots(len(MEASURES), len(VARIANTS), figsize=(11.7, 6.2), squeeze=False)
        fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.08, wspace=0.35, hspace=0.75)
        for r, (dec, w, lab) in enumerate(MEASURES):
            for c, v in enumerate(VARIANTS):
                ax = axes[r, c]
                k = (stat, dec, w, v)
                if k not in nulls:
                    ax.set_axis_off()
                    continue
                null, real, pv = nulls[k]
                row = R[(R.stat == stat) & (R.decoding == dec) & (R.window == w) & (R.variant == v)].iloc[0]
                ax.hist(null, bins=60, color="0.75", lw=0)
                ax.axvline(real, color="k", lw=1.4)
                ax.axvline(0, color="0.5", lw=0.5, ls=":")
                ax.set_title(f"{v}\nR+ {row.mean_rplus:+.3f} (n={row.n_rplus}), R− {row.mean_rminus:+.3f} (n={row.n_rminus})"
                             f"\nΔΔ {real:+.3f}, p = {pv:.4f}", fontsize=FS - 0.5,
                             color="k" if pv >= 0.05 else "#d62728")
                ax.set_yticks([])
                ax.tick_params(labelsize=FS - 1)
                if c == 0:
                    ax.set_ylabel(lab, fontsize=FS)
                if r == len(MEASURES) - 1:
                    ax.set_xlabel("mean(R+) − mean(R−)", fontsize=FS - 0.5)
        what = ("post − pre accuracy at the split" if stat == "delta"
                else "post − pre at the split minus the session's placebo-split mean")
        fig.suptitle(f"Cohort × pre/post interaction (difference of differences) of {what}: real value (black) vs null from "
                     f"{N_PERM} shuffles of R+/R− labels across mice (grey); red title = two-sided p < 0.05 (uncorrected).",
                     fontsize=FS + 0.5)
        (OUT / "figures").mkdir(exist_ok=True)
        for ext in ("pdf", "png", "svg"):
            fig.savefig(OUT / "figures" / f"131_cohort_permutation_{stat}.{ext}", dpi=250)
        plt.close(fig)
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
