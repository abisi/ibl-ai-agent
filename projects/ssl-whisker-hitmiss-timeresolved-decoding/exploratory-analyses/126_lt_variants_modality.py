"""126 -- Modality decoding (whisker vs auditory, -100..0 ms before the corrected first lick) at the learning trial, compared
across LT variants, against each session's placebo splits (user 2026-10-01: "How do the modality decoding compare for the
different variants of the LT"; placebo = every whisker trial as a candidate split, 122 run with SSL_PLACEBO_STEP=1).
Hit vs miss (5-50 / 5-100 ms) computed the same way and saved in the csv for reference.
LT variants (whisker-trial index of the curve-aligned list):
  the 12 definitions of the LT chain (028 table, all mice incl. behaviour-only; only ephys sessions are decoded);
  L6x relaxed  joint whisker + FA CP with lapse, log10 BF > 0.3, P(whisker > FA) > 0.9, no start gates (036);
  L6x forced   the same model's CP for EVERY session (learner or not);
  half         the session-half split of the decoded trials.
Per session x variant: delta = post - pre size-matched balanced accuracy at the LT (122 split at the LT; with the 4-trial
placebo run the nearest split within 2 trials), placebo = all splits >= EXCLUDE_NEAR whisker trials from the LT;
excess = delta - mean placebo delta; percentile = rank of delta among placebo deltas.
Per variant x cohort: n, mean delta, mean excess (+- SEM), mean percentile; excess vs 0 (Wilcoxon AND one-sample t);
R+ vs R- on the excess (Mann-Whitney AND Welch). Uncorrected.
Input tag: env SSL_PLACEBO_TAG ("" = 4-trial placebo run, "_step1" = every whisker trial).
Outputs: figures/126_lt_variants_modality{tag}.{pdf,png,svg}, 126_lt_variants_stats{tag}.csv, 126_lt_variants_per_session{tag}.csv
Run (haas): python 126_lt_variants_modality.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts"
TAG = os.environ.get("SSL_PLACEBO_TAG", "")
COL = {"R+": "#00B400", "R-": "#C800C8"}
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate"]
VARIANTS = DEFS + ["L6x relaxed", "L6x forced", "half"]
EXCLUDE_NEAR = 10
MAXDIST = 0 if TAG.startswith("_step1") else 2
FS = 6.5


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def lt_table():
    t = pd.read_csv(LTP / "028_learning_trials_all_methods_all_mice.csv").set_index("session_id")[DEFS]
    r = pd.read_csv(LTP / "036_joint_cp_relaxed.csv")
    r = r[(r.version == "L6x") & (r.bf_th == 0.3) & (r.p_th == 0.9)].set_index("session_id")
    t["L6x relaxed"] = r.LT.where(r.learner)
    t["L6x forced"] = r.LT
    return t


def main():
    d = pd.read_parquet(OUT / f"122_lt_placebo_whole_brain{TAG}.parquet")
    d = d[d.skipped_reason.isna()]
    LT = lt_table()
    rows = []
    for (sid, rg, dec, w), g in d.groupby(["session_id", "reward_group", "decoding", "window"]):
        pl_all = g[g.split_k >= 0]
        for v in VARIANTS:
            if v == "half":
                h = g[g.real_for == "half"]
                if not len(h):
                    continue
                real = h.delta_matched.iloc[0]
                far = np.abs(pl_all.n_pre.to_numpy() - h.n_pre.iloc[0]) >= EXCLUDE_NEAR
                k = np.nan
            else:
                k = LT.loc[sid, v] if sid in LT.index else np.nan
                if not np.isfinite(k) or not len(pl_all):
                    continue
                dist = np.abs(pl_all.split_k.to_numpy() - k)
                i = int(np.argmin(dist))
                if dist[i] > MAXDIST:
                    continue
                real = pl_all.delta_matched.to_numpy()[i]
                far = dist >= EXCLUDE_NEAR
            pl = pl_all.delta_matched.to_numpy()[far]
            pl = pl[np.isfinite(pl)]
            if len(pl) < 5 or not np.isfinite(real):
                continue
            rows.append(dict(session_id=sid, reward_group=rg, decoding=dec, window=w, variant=v, lt=k, delta=real,
                             placebo_mean=pl.mean(), excess=real - pl.mean(),
                             percentile=float(np.mean(pl < real) + 0.5 * np.mean(pl == real)), n_placebo=len(pl)))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / f"126_lt_variants_per_session{TAG}.csv", index=False)
    srows = []
    for (dec, w, v), g in P.groupby(["decoding", "window", "variant"]):
        ex = {}
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            ex[rg] = x.excess.to_numpy()
            r = dict(decoding=dec, window=w, variant=v, cohort=rg, n=len(x), mean_delta=x.delta.mean(),
                     mean_excess=x.excess.mean(), sem_excess=x.excess.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
                     mean_percentile=x.percentile.mean())
            if len(x) >= 5:
                r.update(p_wilcoxon=stats.wilcoxon(x.excess).pvalue, p_t=stats.ttest_1samp(x.excess, 0).pvalue)
            srows.append(r)
        if min(len(ex["R+"]), len(ex["R-"])) >= 3:
            pm = stats.mannwhitneyu(ex["R+"], ex["R-"]).pvalue
            pw = stats.ttest_ind(ex["R+"], ex["R-"], equal_var=False).pvalue
        else:
            pm = pw = np.nan
        for r in srows[-2:]:
            r.update(p_mw_cohorts=pm, p_welch_cohorts=pw)
    S = pd.DataFrame(srows)
    S.to_csv(OUT / f"126_lt_variants_stats{TAG}.csv", index=False)
    figure(S)
    pd.set_option("display.width", 220)
    print(S[S.decoding == "modality_lick"][["variant", "cohort", "n", "mean_delta", "mean_excess", "sem_excess",
                                            "mean_percentile", "p_wilcoxon", "p_t", "p_mw_cohorts", "p_welch_cohorts"]]
          .round(3).to_string(index=False))


def figure(S):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    rows = [("modality_lick", "-100-0ms", "whisker vs auditory, −100-0 ms pre-lick"), ("hitmiss", "5-50ms", "hit/miss 5-50 ms"),
            ("hitmiss", "5-100ms", "hit/miss 5-100 ms")]
    fig, axes = plt.subplots(1, 3, figsize=(8.27, 4.6), sharey=True)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.86, bottom=0.12, wspace=0.55)
    y = np.arange(len(VARIANTS))[::-1]
    for ax, (dec, w, lab) in zip(axes, rows):
        s = S[(S.decoding == dec) & (S.window == w)].set_index(["variant", "cohort"])
        for k, v in enumerate(VARIANTS):
            for rg, dy in (("R+", 0.17), ("R-", -0.17)):
                if (v, rg) not in s.index:
                    continue
                r = s.loc[(v, rg)]
                sig = np.isfinite(r.get("p_wilcoxon", np.nan)) and (r.p_wilcoxon < 0.05 or r.p_t < 0.05)
                ax.errorbar(r.mean_excess, y[k] + dy, xerr=r.sem_excess, fmt="o", color=COL[rg],
                            mfc=COL[rg] if sig else "white", ms=3.2, elinewidth=0.7, capsize=0)
                ax.text(1.02, y[k] + dy, f"{int(r.n)} {pf(r.get('p_wilcoxon', np.nan))}", transform=ax.get_yaxis_transform(),
                        fontsize=FS - 1.5, color=COL[rg], va="center")
            pm = s.loc[(v, "R+"), "p_mw_cohorts"] if (v, "R+") in s.index else np.nan
            if np.isfinite(pm) and pm < 0.05:
                ax.text(ax.get_xlim()[0], y[k], "*", fontsize=FS + 2, va="center")
        ax.axvline(0, color="0.6", lw=0.6, ls=":")
        for yy in (y[len(DEFS)] + 0.5, y[len(DEFS) + 2] + 0.5):
            ax.axhline(yy, color="0.85", lw=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(VARIANTS, fontsize=FS - 0.5)
        ax.set_xlabel("Δacc at LT − placebo mean (mean ± SEM)")
        ax.set_title(lab, fontsize=FS + 0.5, pad=14)
        ax.text(1.02, y[0] + 0.75, "n  p(W)", transform=ax.get_yaxis_transform(), fontsize=FS - 1.5)
    fig.suptitle(f"Pre/post decoding change at the learning trial beyond the session's placebo splits, per LT variant "
                 f"(placebo: {'every whisker trial' if TAG == '_step1' else 'every 4th whisker trial'}, ≥ {EXCLUDE_NEAR} "
                 "trials from the LT). Filled = p < 0.05 vs 0 (Wilcoxon or t); uncorrected.", fontsize=FS + 0.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"126_lt_variants_modality{TAG}.{ext}", dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    main()
