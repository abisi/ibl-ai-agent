"""133c -- Figure for the latest 133 results (user 2026-10-02: "plot these latest results"): the SINGLE decoder (one decoder
trained on passive_pre + active + passive_post, evaluated on each epoch), its cost relative to epoch-specific decoders, and the
correlation of the passive -> active coding-axis change with active-session behaviour (whole brain).
  a  single decoder per epoch (acc - trial-shuffle null), per cohort: mean +- SEM, faint lines = mice; Friedman AND RM-ANOVA
     within cohort; R+ vs R- on each change (Mann-Whitney AND Welch);
  b  cost of sharing = single - within (epoch-specific decoder) per epoch; vs 0 Wilcoxon AND t per cohort;
  c-f  per-mouse scatter, OLS line + 95% CI band (solid if Pearson p < 0.05), Spearman AND Pearson per cohort:
     pre -> active angle vs max slope of whisker - FA; pre -> active normalised cos vs max slope; pre -> active angle vs mean
     whisker P(lick); single-decoder active - passive change vs mean whisker - FA.
Behaviour (whisker day 0, full session): HMM curves (sigma = 1, 024); max slope = steepest rise (R+) / fall (R-) of whisker - FA
over 10 whisker trials (124). Units: env SSL_UNITS (all = good + mua; good = good-only tracked units, 133 *_good).
Outputs: figures/133c_single_decoder_behaviour{tag}.{pdf,png,svg}, 133c_correlations{tag}.csv
Run (haas): python 133c_single_decoder_behaviour_figure.py
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification" / "artifacts"
TAG = "" if os.environ.get("SSL_UNITS", "all") == "all" else "_good"
COL = {"R+": "#00B400", "R-": "#C800C8"}
EP = ["passive_pre", "active", "passive_post"]
FS = 6.5


def pf(p):
    return "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def load():
    d = pd.read_parquet(OUT / f"133_modality_stim_epochs{TAG}.parquet")
    d = d[d.skipped_reason.isna() & (d.area == "All units")].copy()
    d["single_act_vs_passive"] = d.single_active_corr - (d.single_passive_pre_corr + d.single_passive_post_corr) / 2
    P = pickle.load(open(LTP / "024_avg_curves_per_mouse.pkl", "rb"))["sessions"]
    beh = {p["session_id"]: p["curves"] for p in P}
    d["mean_wfa"] = d.session_id.map(lambda s: float(np.mean(beh[s]["w-fa"])) if s in beh else np.nan)
    d["mean_w"] = d.session_id.map(lambda s: float(np.mean(beh[s]["whisker"])) if s in beh else np.nan)
    sc = pd.read_csv(OUT / "124_per_session.csv").set_index("session_id")
    d["max_slope"] = d.session_id.map(sc.max_slope)
    return d


def scatter(ax, d, xcol, ycol, xlabel, ylabel, rows):
    for rg in ("R+", "R-"):
        g = d[d.reward_group == rg][[xcol, ycol]].dropna()
        if len(g) < 5:
            continue
        x, y = g[xcol].to_numpy(), g[ycol].to_numpy()
        ax.scatter(x, y, s=9, color=COL[rg], lw=0, alpha=0.8)
        rs, ps = stats.spearmanr(x, y)
        rp, pp = stats.pearsonr(x, y)
        sl, ic = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        X = np.c_[np.ones_like(x), x]
        res = y - X @ np.array([ic, sl])
        cov = (res @ res / max(len(x) - 2, 1)) * np.linalg.inv(X.T @ X)
        Xs = np.c_[np.ones_like(xs), xs]
        se = np.sqrt(np.einsum("ij,jk,ik->i", Xs, cov, Xs))
        tq = stats.t.ppf(0.975, max(len(x) - 2, 1))
        ax.fill_between(xs, ic + sl * xs - tq * se, ic + sl * xs + tq * se, color=COL[rg], alpha=0.15, lw=0)
        ax.plot(xs, ic + sl * xs, color=COL[rg], lw=1.2, ls="-" if pp < 0.05 else "--")
        ax.text(0.02 if rg == "R+" else 0.52, 1.02, f"{'R+' if rg == 'R+' else 'R−'} ρ {rs:.2f} ({pf(ps)})\nr {rp:.2f} ({pf(pp)})",
                transform=ax.transAxes, fontsize=FS - 1, color=COL[rg], va="bottom")
        rows.append(dict(cohort=rg, x=xcol, y=ycol, n=len(x), spearman=rs, p_spearman=ps, pearson=rp, p_pearson=pp))
    ax.set_xlabel(xlabel, fontsize=FS)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.tick_params(labelsize=FS - 0.5)


def main():
    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": FS})
    d = load()
    fig, axes = plt.subplots(2, 3, figsize=(8.27, 5.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.09, wspace=0.42, hspace=0.75)
    rng = np.random.default_rng(0)
    # a single decoder per epoch
    ax = axes[0, 0]
    cols = [f"single_{e}_corr" for e in EP]
    for k, rg in enumerate(("R+", "R-")):
        x = d[d.reward_group == rg][cols].dropna()
        off = (k - 0.5) * 0.12
        for _, r in x.iterrows():
            ax.plot(np.arange(3) + off, r.to_numpy(), color=COL[rg], lw=0.3, alpha=0.25)
        ax.errorbar(np.arange(3) + off, x.mean(), yerr=x.std(ddof=1) / np.sqrt(len(x)), color=COL[rg], lw=1.6, marker="o",
                    ms=3, capsize=0, label=f"{'R+' if rg == 'R+' else 'R−'} n={len(x)}")
    txt = []
    for rg in ("R+", "R-"):
        x = d[d.reward_group == rg][cols].dropna()
        fr = stats.friedmanchisquare(*[x[c] for c in cols]).pvalue
        from statsmodels.stats.anova import AnovaRM
        long = x.reset_index().melt(id_vars="index", value_vars=cols, var_name="ep", value_name="v")
        rm = float(AnovaRM(long, "v", "index", within=["ep"]).fit().anova_table["Pr > F"].iloc[0])
        txt.append(f"{'R+' if rg == 'R+' else 'R−'}: Friedman {pf(fr)}, RM-ANOVA {pf(rm)}")
    for name, (i, j) in (("active−pre", (1, 0)), ("post−active", (2, 1))):
        a = d[d.reward_group == "R+"][cols[i]] - d[d.reward_group == "R+"][cols[j]]
        b = d[d.reward_group == "R-"][cols[i]] - d[d.reward_group == "R-"][cols[j]]
        txt.append(f"Δ{name} R+ vs R−: MW {pf(stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue)}, "
                   f"Welch {pf(stats.ttest_ind(a.dropna(), b.dropna(), equal_var=False).pvalue)}")
    ax.set_title("\n".join(txt), fontsize=FS - 1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["pre", "active", "post"])
    ax.set_ylabel("single decoder, acc − shuffle")
    ax.legend(fontsize=FS - 1, frameon=False, loc="lower left")
    # b cost of sharing
    ax = axes[0, 1]
    for k, rg in enumerate(("R+", "R-")):
        g = d[d.reward_group == rg]
        for i, e in enumerate(EP):
            v = (g[f"single_{e}_corr"] - g[f"within_{e}_corr"]).dropna().to_numpy()
            x0 = i + (k - 0.5) * 0.3
            ax.scatter(x0 + rng.uniform(-0.06, 0.06, len(v)), v, s=5, color=COL[rg], lw=0, alpha=0.6)
            ax.errorbar(x0, v.mean(), yerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="_", color="k", ms=8, elinewidth=1)
            ax.text(x0, 1.0, f"W {pf(stats.wilcoxon(v).pvalue)}\nt {pf(stats.ttest_1samp(v, 0).pvalue)}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=FS - 2, color=COL[rg])
    ax.axhline(0, color="0.6", lw=0.6, ls=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["pre", "active", "post"])
    ax.set_ylabel("single − epoch-specific decoder")
    ax.set_title("cost of one shared decoder", fontsize=FS, pad=18)
    rows = []
    scatter(axes[0, 2], d, "max_slope", "angle_passive_pre__active", "max slope of whisker − FA\n(R+ rise, R− fall; per trial)",
            "pre → active coding angle (°)", rows)
    scatter(axes[1, 0], d, "max_slope", "cosnorm_passive_pre__active", "max slope of whisker − FA",
            "pre → active alignment (norm. cos)", rows)
    scatter(axes[1, 1], d, "mean_w", "angle_passive_pre__active", "mean whisker P(lick), session",
            "pre → active coding angle (°)", rows)
    scatter(axes[1, 2], d, "mean_wfa", "single_act_vs_passive", "mean whisker − FA, session",
            "single decoder: active − passive", rows)
    for ax, L in zip(axes.flat, "abcdef"):
        ax.text(-0.22, 1.12, L, transform=ax.transAxes, fontweight="bold", fontsize=FS + 1.5)
    units = "good + mua" if TAG == "" else "good units tracked in every epoch"
    n = d.groupby("reward_group").size().to_dict()
    fig.suptitle(f"Whisker vs auditory, 5-35 ms, whole brain ({units}; R+ n={n.get('R+', 0)}, R− n={n.get('R-', 0)}): one "
                 "decoder for all epochs, and the passive → active coding change vs active-session behaviour. Uncorrected.",
                 fontsize=FS + 0.5)
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(OUT / "figures" / f"133c_single_decoder_behaviour{TAG}.{ext}", dpi=250)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / f"133c_correlations{TAG}.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
