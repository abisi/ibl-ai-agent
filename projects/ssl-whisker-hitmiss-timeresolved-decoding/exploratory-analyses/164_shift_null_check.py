"""164 -- Passive pre -> post whisker change on the hit / miss axis vs axes rebuilt from linearly shifted hit / miss labels (user
2026-10-08: "compare my effects with those of CD obtained from linear shifted versions ... before looking at effects relative to another
condition"; native scale, unit-norm axis, with and without the per-unit scaling). Same layout as ssl-prelick-convergence within_day 026.
Data: 135 (shared tracked stable units, whole brain, learners): shift_real_d{W,A}P = real passive post - pre change of the projection on
the unit lick axis / sqrt(n units); shift_null_d{W,A}P_mean = mean of the same change on axes rebuilt from labels shifted by 10-50 % of
the time-ordered active whisker trials (non-wrapping); excess = real - null mean. 'unit' = per-unit value x sqrt(n units).
Statistics (unit = session): real vs null (Wilcoxon, paired), excess vs 0 (Wilcoxon), R+ vs R- (Mann-Whitney).
Outputs: figures/publication/164_shift_null_check_learners.{png,pdf,svg}; 164_stats_learners.csv
Run (haas, repo root): python .../164_shift_null_check.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
COL, COH, FIGDIR = H.COL, H.COH, H.FIGDIR
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def main():
    H.setup()
    A = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    A = A[(A.area == "All units") & A.skipped_reason.isna()]; A = A[H.mouse_of(A).isin(H.learners())].dropna(subset=["shift_real_dWP", "shift_null_dWP_mean", "shift_null_dAP_mean"]).copy()
    for s in ("W", "A"):
        A[f"real_pu_{s}"], A[f"null_pu_{s}"] = A[f"shift_real_d{s}P"], A[f"shift_null_d{s}P_mean"]
        A[f"excess_pu_{s}"] = A[f"real_pu_{s}"] - A[f"null_pu_{s}"]
        for k in ("real", "null", "excess"):
            A[f"{k}_unit_{s}"] = A[f"{k}_pu_{s}"] * np.sqrt(A.n_units)
    rows = []
    for s in ("W", "A"):
        for sc in ("pu", "unit"):
            for c in COH:
                d = A[A.reward_group == c]
                rows.append(dict(stim=s, scale=sc, cohort=c, n=len(d), real=d[f"real_{sc}_{s}"].mean(), null=d[f"null_{sc}_{s}"].mean(),
                                 excess=d[f"excess_{sc}_{s}"].mean(), p_real_vs0=stats.wilcoxon(d[f"real_{sc}_{s}"]).pvalue,
                                 p_real_vs_null=stats.wilcoxon(d[f"real_{sc}_{s}"], d[f"null_{sc}_{s}"]).pvalue,
                                 p_excess_vs0=stats.wilcoxon(d[f"excess_{sc}_{s}"]).pvalue))
            a, b = A[A.reward_group == "R+"], A[A.reward_group == "R-"]
            rows.append(dict(stim=s, scale=sc, cohort="R+ vs R-", p_real=stats.mannwhitneyu(a[f"real_{sc}_{s}"], b[f"real_{sc}_{s}"]).pvalue,
                             p_null=stats.mannwhitneyu(a[f"null_{sc}_{s}"], b[f"null_{sc}_{s}"]).pvalue,
                             p_excess=stats.mannwhitneyu(a[f"excess_{sc}_{s}"], b[f"excess_{sc}_{s}"]).pvalue))
    T = pd.DataFrame(rows); T.to_csv(EA / "164_stats_learners.csv", index=False)
    print(T.round(4).to_string())
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.9), gridspec_kw=dict(wspace=0.7))
    for j, sc in enumerate(("pu", "unit")):
        ax = axs[2 * j]
        for k, c in enumerate(COH):
            d = A[A.reward_group == c]; x0 = k * 2.2
            for xx, col in ((x0, f"null_{sc}_W"), (x0 + 1, f"real_{sc}_W")):
                ax.errorbar(xx, d[col].mean(), H.sem(d[col]), fmt="o", ms=3.5, color=COL[c], mfc="white" if "null" in col else COL[c], lw=1, capsize=0)
            ax.plot([x0, x0 + 1], [d[f"null_{sc}_W"].mean(), d[f"real_{sc}_W"].mean()], color=COL[c], lw=0.8)
            r = T[(T.stim == "W") & (T.scale == sc) & (T.cohort == c)].iloc[0]
            ax.text(x0 + 0.5, 1.0, f"p = {r.p_real_vs_null:.3g}", color=COL[c], transform=ax.get_xaxis_transform(), ha="center", fontsize=5)
        ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels(["shifted", "real"] * 2, fontsize=5); ax.set_xlim(-0.5, 3.7)
        ax.set_ylabel("passive whisker post − pre\non the hit − miss axis " + ("(SD per neuron)" if sc == "pu" else "(z units)"))
        ax.set_title("Real vs shifted-label axes\n" + ("per neuron (÷ √N)" if sc == "pu" else "unit-norm axis, no ÷ √N"), pad=12)
        ax = axs[2 * j + 1]
        for k, c in enumerate(COH):
            d = A[A.reward_group == c][f"excess_{sc}_W"]
            r = T[(T.stim == "W") & (T.scale == sc) & (T.cohort == c)].iloc[0]
            ax.errorbar(k, d.mean(), H.sem(d), fmt="o", ms=3.8, color=COL[c], mfc=COL[c] if r.p_excess_vs0 < 0.05 else "white", lw=1.1, capsize=0)
        rr = T[(T.stim == "W") & (T.scale == sc) & (T.cohort == "R+ vs R-")].iloc[0]
        ax.text(0.5, 1.0, f"R+ vs R− p = {rr.p_excess:.3g}", transform=ax.get_xaxis_transform(), ha="center", fontsize=5)
        ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2))); ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R−"]); ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("excess (real − shifted)"); ax.set_title("Beyond session time\n(filled: ≠ 0)", pad=12)
    fig.suptitle("Axis 2 (stimulus onset, learners): passive whisker pre → post change on the hit − miss axis vs axes from linearly shifted "
                 "hit / miss labels", fontsize=6.5, y=1.22)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"164_shift_null_check_learners.{ext}", dpi=300, bbox_inches="tight")
    print("ALL DONE", FIGDIR / "164_shift_null_check_learners.png")


if __name__ == "__main__":
    main()
