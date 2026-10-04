"""122b -- Analysis of the placebo-split test (122) for every learning-trial definition (user 2026-10-01).
Per session x decoding x window x definition: the real pre/post change (delta = post - pre size-matched balanced
accuracy at the definition's LT) is ranked among the session's placebo splits (every 4th whisker trial) that lie at
least EXCLUDE_NEAR whisker trials away from the LT: percentile = P(placebo < real) + 0.5 P(placebo = real). 0.5 = the
LT is not special; > 0.5 = the change is larger at the LT than at arbitrary splits. The session-half split ("half") is
ranked the same way (reference).
Group level per definition x cohort x window: mean percentile (one-sample Wilcoxon AND t-test vs 0.5), fraction of
sessions >= 0.95, and real delta vs the session's mean placebo delta (paired Wilcoxon AND paired t). R+ vs R- on the
percentile: Mann-Whitney AND Welch. All uncorrected (exploratory).
Also: mean placebo delta as a function of the split position (fraction of the session), per cohort -- how much any
early/late split changes decodability (drift), with the real LTs' positions.
Outputs: figures/122b_lt_placebo.{pdf,png,svg}, 122b_lt_placebo_stats.csv, 122b_lt_placebo_per_session.csv
Run: python 122b_lt_placebo_analysis.py
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
FIG = OUT / "figures"
COL = {"R+": "#00B400", "R-": "#C800C8"}
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate", "half"]
ROWS = [("hitmiss", "5-50ms", "Hit vs miss 5-50 ms"), ("hitmiss", "5-100ms", "Hit vs miss 5-100 ms"),
        ("modality_lick", "-100-0ms", "Whisker vs auditory −100-0 ms pre-lick")]
EXCLUDE_NEAR = 10
MIN_PLACEBO = 5
FS_L, FS_M, FS_S = 8, 7, 6


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def per_session(d):
    out = []
    for (sid, rg, dec, w), g in d.groupby(["session_id", "reward_group", "decoding", "window"]):
        plac = g[g.split_k >= 0]
        for df_ in DEFS:
            if df_ == "half":
                r = g[g.real_for == "half"]
                if not len(r):
                    continue
                pos = float((r.n_pre.iloc[0]) / max(r.n_trials.iloc[0], 1))
                k_lt = None
            else:
                r = g[g.real_for.fillna("").str.split("|").map(lambda xs: df_ in xs) & (g.split_k >= 0)]
                if not len(r):
                    continue
                k_lt = int(r.split_k.iloc[0])
                pos = k_lt / max(int(r.n_whisker_curve.iloc[0]) - 1, 1)
            real = float(r.delta_matched.iloc[0])
            if k_lt is None:
                # half split: placebo = splits away from the median split by trial count
                kk = plac.n_pre.to_numpy()
                pl = plac.delta_matched.to_numpy()[np.abs(kk - r.n_pre.iloc[0]) >= EXCLUDE_NEAR]
            else:
                pl = plac.delta_matched.to_numpy()[np.abs(plac.split_k.to_numpy() - k_lt) >= EXCLUDE_NEAR]
            pl = pl[np.isfinite(pl)]
            if len(pl) < MIN_PLACEBO or not np.isfinite(real):
                continue
            pct = float(np.mean(pl < real) + 0.5 * np.mean(pl == real))
            out.append(dict(session_id=sid, reward_group=rg, decoding=dec, window=w, definition=df_, lt_pos=pos,
                            real_delta=real, placebo_mean=float(pl.mean()), placebo_sd=float(pl.std()),
                            n_placebo=len(pl), percentile=pct))
    return pd.DataFrame(out)


def group_stats(P):
    rows = []
    for (dec, w, df_), g in P.groupby(["decoding", "window", "definition"]):
        pc = {}
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            pc[rg] = x.percentile.to_numpy()
            n = len(x)
            r = dict(decoding=dec, window=w, definition=df_, cohort=rg, n=n,
                     mean_percentile=x.percentile.mean() if n else np.nan,
                     sem_percentile=x.percentile.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan,
                     frac_ge95=float((x.percentile >= 0.95).mean()) if n else np.nan,
                     mean_real_delta=x.real_delta.mean() if n else np.nan,
                     mean_placebo_delta=x.placebo_mean.mean() if n else np.nan)
            if n >= 5:
                r["p_wilcoxon_pct"] = stats.wilcoxon(x.percentile - 0.5).pvalue if np.any(x.percentile != 0.5) else np.nan
                r["p_t_pct"] = stats.ttest_1samp(x.percentile, 0.5).pvalue
                r["p_wilcoxon_real_vs_placebo"] = stats.wilcoxon(x.real_delta, x.placebo_mean).pvalue
                r["p_t_real_vs_placebo"] = stats.ttest_rel(x.real_delta, x.placebo_mean).pvalue
            rows.append(r)
        if min(len(pc["R+"]), len(pc["R-"])) >= 3:
            pm = stats.mannwhitneyu(pc["R+"], pc["R-"]).pvalue
            pw = stats.ttest_ind(pc["R+"], pc["R-"], equal_var=False).pvalue
        else:
            pm = pw = np.nan
        for r in rows[-2:]:
            r.update(p_mw_cohorts=pm, p_welch_cohorts=pw)
    return pd.DataFrame(rows)


def figure(P, S, d):
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(3, 3, figsize=(8.27, 8.6), gridspec_kw=dict(width_ratios=[1.3, 0.55, 1.0]))
    fig.subplots_adjust(left=0.2, right=0.98, top=0.93, bottom=0.06, wspace=0.35, hspace=0.45)
    y = np.arange(len(DEFS))[::-1]
    for r, (dec, w, lab) in enumerate(ROWS):
        s = S[(S.decoding == dec) & (S.window == w)].set_index(["definition", "cohort"])
        ax = axes[r, 0]
        for k, df_ in enumerate(DEFS):
            for rg, dy in (("R+", 0.15), ("R-", -0.15)):
                if (df_, rg) not in s.index or s.loc[(df_, rg), "n"] == 0:
                    continue
                v = s.loc[(df_, rg)]
                sig = np.isfinite(v.get("p_wilcoxon_pct", np.nan)) and v.p_wilcoxon_pct < 0.05
                ax.errorbar(v.mean_percentile, y[k] + dy, xerr=v.sem_percentile, fmt="o", color=COL[rg],
                            mfc=COL[rg] if sig else "white", ms=3.4, elinewidth=0.7, capsize=0)
                ax.text(1.03, y[k] + dy, f"{int(v.n)} {pf(v.get('p_wilcoxon_pct', np.nan))}", fontsize=FS_S - 0.5,
                        color=COL[rg], va="center", transform=ax.get_yaxis_transform())
        ax.axvline(0.5, color="0.6", lw=0.6, ls="--")
        ax.axhline(y[-1] + 0.5, color="0.85", lw=0.5)
        ax.set_xlim(0.15, 0.95)
        ax.set_yticks(y)
        ax.set_yticklabels(DEFS, fontsize=FS_S)
        ax.tick_params(axis="x", labelsize=FS_S)
        ax.set_xlabel("percentile of the real change among placebo splits", fontsize=FS_M)
        ax.set_title(lab, fontsize=FS_M, loc="left", fontweight="bold")
        ax.text(1.03, y[0] + 0.7, "n  p(W)", fontsize=FS_S - 0.5, transform=ax.get_yaxis_transform())
        ax = axes[r, 1]
        for k, df_ in enumerate(DEFS):
            for rg, dy in (("R+", 0.15), ("R-", -0.15)):
                if (df_, rg) in s.index and s.loc[(df_, rg), "n"] > 0:
                    ax.barh(y[k] + dy, s.loc[(df_, rg), "frac_ge95"], height=0.28, color=COL[rg])
        ax.axvline(0.05, color="0.6", lw=0.6, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.set_xlim(0, 0.6)
        ax.tick_params(labelsize=FS_S)
        ax.set_xlabel("fraction of sessions\npercentile ≥ 0.95", fontsize=FS_M)
        ax = axes[r, 2]
        g = d[(d.decoding == dec) & (d.window == w) & (d.split_k >= 0)].copy()
        g["pos"] = g.split_k / (g.n_whisker_curve - 1).clip(lower=1)
        bins = np.linspace(0, 1, 11)
        for rg in ("R+", "R-"):
            x = g[g.reward_group == rg]
            sess_curves = []
            for _, gs in x.groupby("session_id"):
                idx = np.digitize(gs.pos, bins) - 1
                sess_curves.append([gs.delta_matched[idx == b].mean() if (idx == b).any() else np.nan
                                    for b in range(10)])
            M = np.array(sess_curves, float)
            m = np.nanmean(M, 0)
            se = np.nanstd(M, 0, ddof=1) / np.sqrt(np.isfinite(M).sum(0))
            xc = (bins[:-1] + bins[1:]) / 2
            ax.fill_between(xc, m - se, m + se, color=COL[rg], alpha=0.2, lw=0)
            ax.plot(xc, m, color=COL[rg], lw=1.4, label=f"{'R+' if rg == 'R+' else 'R−'} placebo splits")
            lt = P[(P.decoding == dec) & (P.window == w) & (P.reward_group == rg) & (P.definition == "lenient cascade")]
            ax.scatter(lt.lt_pos, lt.real_delta, s=7, color=COL[rg], alpha=0.6, lw=0, zorder=3)
        ax.axhline(0, color="0.6", lw=0.6, ls=":")
        ax.tick_params(labelsize=FS_S)
        ax.set_xlabel("split position (fraction of whisker trials)", fontsize=FS_M)
        ax.set_ylabel("Δ acc. (post − pre)", fontsize=FS_M)
        if r == 0:
            ax.legend(fontsize=FS_S, frameon=False)
            ax.set_title("mean placebo change vs split position\n(dots: real LT, lenient cascade)", fontsize=FS_S)
    for ax, L in zip(axes[0], "abc"):
        ax.text(-0.1 if L != "a" else -0.62, 1.12, L, transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    fig.suptitle("Placebo-split test of the learning trial, per definition (whole brain, A1-trimmed; open = n.s., filled = "
                 f"Wilcoxon p < 0.05 vs 0.5; placebo splits >= {EXCLUDE_NEAR} trials from the LT; uncorrected)",
                 fontsize=FS_M, y=0.985)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"122b_lt_placebo.{ext}", dpi=300)
    plt.close(fig)


def main():
    d = pd.read_parquet(OUT / f"122_lt_placebo_whole_brain{os.environ.get('SSL_PLACEBO_TAG', '')}.parquet")
    d = d[d.skipped_reason.isna()].copy()
    P = per_session(d)
    P.to_csv(OUT / "122b_lt_placebo_per_session.csv", index=False)
    S = group_stats(P)
    S.to_csv(OUT / "122b_lt_placebo_stats.csv", index=False)
    figure(P, S, d)
    pd.set_option("display.width", 220)
    print(S[["decoding", "window", "definition", "cohort", "n", "mean_percentile", "frac_ge95", "p_wilcoxon_pct",
             "p_t_pct", "p_mw_cohorts"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
