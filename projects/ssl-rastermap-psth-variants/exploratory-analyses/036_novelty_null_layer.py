"""Step 2 of the novelty-model plan: the null layer (non-specific drift) and the R- learning caveat.

A. Drift per region: for every session and region, OLS slope of the trial response vs clock time (per 10 min) on
   AUDITORY trials (familiar, rewarded in both cohorts) and on CATCH trials (no stimulus), for the pre-stimulus
   baseline (z_base) and the stimulus windows (z_early 5-50 ms, z_late 50-150 ms). Auditory warm-up trials included;
   first 5 auditory trials excluded (as in the RPE-v2 analysis). Unit = session (= mouse on day 0).
   Per region x cohort: mean slope, Wilcoxon AND one-sample t vs 0; R+ vs R-: Mann-Whitney AND Welch.
   Regions: area_group (+ 'all'); regions need >= MIN_SESS sessions per cohort with >= MIN_UNITS good units.
B. R- caveat (does R- learn "whisker = no reward"?): P(lick) on whisker trials vs catch trials along the session.
   Per session: logistic-free linear slope of lick probability vs trial position (per 100 trials of that type) for
   whisker and catch trials; whisker - catch slope difference, tested vs 0 within cohort and R+ vs R-.
Output: combined_results_ks4/_novelty_model/null_layer/ (csv + null_layer.png/.pdf)
"""
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import importlib                                                          # noqa: E402
m = importlib.import_module("029_roc_rpe_v2")
BASE = m.RES / "_novelty_model"
OUT = BASE / "null_layer"
COH = {"R+": "#00B400", "R-": "#C800C8"}
MIN_UNITS, MIN_SESS = 5, 4
WINS = ["z_base", "z_early", "z_late"]


def slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8 or np.ptp(x[ok]) == 0:
        return np.nan
    return np.polyfit(x[ok], y[ok], 1)[0]


def tests(v_plus, v_minus):
    out = {}
    for coh, v in [("Rplus", v_plus), ("Rminus", v_minus)]:
        v = pd.Series(v).dropna()
        out[f"mean_{coh}"], out[f"n_{coh}"] = v.mean(), len(v)
        out[f"p_wilcoxon_{coh}"] = stats.wilcoxon(v).pvalue if len(v) >= MIN_SESS else np.nan
        out[f"p_ttest_{coh}"] = stats.ttest_1samp(v, 0).pvalue if len(v) >= MIN_SESS else np.nan
    a, b = pd.Series(v_plus).dropna(), pd.Series(v_minus).dropna()
    out["p_MWU_RplusRminus"] = stats.mannwhitneyu(a, b).pvalue if min(len(a), len(b)) >= MIN_SESS else np.nan
    out["p_Welch_RplusRminus"] = stats.ttest_ind(a, b, equal_var=False).pvalue if min(len(a), len(b)) >= MIN_SESS else np.nan
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    R = pd.read_parquet(BASE / "trial_region_table.parquet")
    C = pd.read_parquet(BASE / "trial_covariates.parquet")
    R = R[R.level.isin(["area_group", "all"]) & (R.n_units >= MIN_UNITS)]
    D = R.merge(C, on=["session_id", "trial"])
    D = D[~((D.ttype == "auditory") & (D.aud_rank < 5))]
    # ---- A: per-session slopes (per 10 min)
    rows = []
    for (sid, reg, tt), g in D[D.ttype.isin(["auditory", "catch"])].groupby(["session_id", "region", "ttype"]):
        r = dict(session_id=sid, cohort=g.cohort.iloc[0], region=reg, ttype=tt, n_trials=len(g), n_units=g.n_units.iloc[0])
        for w in WINS:
            r[w] = slope(g.clock_min.to_numpy() / 10, g[w].to_numpy())
        rows.append(r)
    S = pd.DataFrame(rows); S.to_csv(OUT / "drift_slopes_per_session.csv", index=False)
    T = []
    for (reg, tt), g in S.groupby(["region", "ttype"]):
        for w in WINS:
            T.append(dict(region=reg, ttype=tt, window=w,
                          **tests(g.loc[g.cohort == "R+", w], g.loc[g.cohort == "R-", w])))
    T = pd.DataFrame(T); T.to_csv(OUT / "drift_slopes_region_stats.csv", index=False)
    # ---- B: R- caveat, lick probability along the session
    lk = []
    for sid, g in C.groupby("session_id"):
        r = dict(session_id=sid, cohort=g.cohort.iloc[0])
        for tt in ["whisker", "catch", "auditory"]:
            h = g[g.ttype == tt]
            r[f"slope_{tt}"] = slope(np.arange(len(h)) / 100, h.lick.to_numpy(float))
            r[f"plick_{tt}"] = h.lick.mean()
        r["whisker_minus_catch"] = r["slope_whisker"] - r["slope_catch"]
        # gap P(lick | whisker) - P(lick | catch) early (trials 1-30 of each type) vs later (31-100)
        wv, cv_ = g[g.ttype == "whisker"].lick.to_numpy(float), g[g.ttype == "catch"].lick.to_numpy(float)
        r["gap_early"] = wv[:30].mean() - cv_[:30].mean()
        r["gap_late"] = wv[30:100].mean() - cv_[30:100].mean() if min(len(wv), len(cv_)) > 40 else np.nan
        r["gap_change"] = r["gap_late"] - r["gap_early"]
        lk.append(r)
    L = pd.DataFrame(lk); L.to_csv(OUT / "lick_slopes_per_session.csv", index=False)
    LB = pd.DataFrame([dict(metric=k, **tests(L.loc[L.cohort == "R+", k], L.loc[L.cohort == "R-", k]))
                       for k in ["slope_whisker", "slope_catch", "slope_auditory", "whisker_minus_catch",
                                 "gap_early", "gap_late", "gap_change"]])
    LB.to_csv(OUT / "lick_slopes_stats.csv", index=False)
    # ---- figure
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.45, left=0.2, right=0.98, top=0.95, bottom=0.07)
    regs = (S[S.region != "all"].groupby("region").session_id.nunique().sort_values(ascending=False))
    regs = ["all"] + [r for r in regs.index if (S[(S.region == r)].groupby("cohort").session_id.nunique() >= MIN_SESS).sum() == 2]
    for j, (tt, w) in enumerate([("auditory", "z_late"), ("catch", "z_late"), ("auditory", "z_base")]):
        ax = fig.add_subplot(gs[0:2, j])
        for k, coh in enumerate(COH):
            mu, se = [], []
            for r in regs:
                v = S[(S.region == r) & (S.ttype == tt) & (S.cohort == coh)][w].dropna()
                mu.append(v.mean()); se.append(v.std() / np.sqrt(max(len(v), 1)))
            y = np.arange(len(regs)) + (k - 0.5) * 0.3
            ax.errorbar(mu, y, xerr=se, fmt="o", ms=3, color=COH[coh], label=coh.replace("-", "−"), elinewidth=0.8)
        for yy, r in enumerate(regs):
            t = T[(T.region == r) & (T.ttype == tt) & (T.window == w)]
            if len(t) and (t.p_wilcoxon_Rplus.iloc[0] < .05 and t.p_ttest_Rplus.iloc[0] < .05 or
                           t.p_wilcoxon_Rminus.iloc[0] < .05 and t.p_ttest_Rminus.iloc[0] < .05):
                ax.text(ax.get_xlim()[1] if False else 0, yy + 0.35, "*", fontsize=7, ha="center")
        ax.axvline(0, color="k", lw=0.5); ax.set_yticks(range(len(regs)), regs if j == 0 else [""] * len(regs), fontsize=6)
        ax.invert_yaxis(); ax.set_xlabel("slope (z per 10 min)")
        ax.set_title(f"{'abc'[j]}  {tt} trials, {w.replace('z_', '')} window", loc="left", fontsize=7.5)
        if j == 0:
            ax.legend(frameon=False, fontsize=6)
    ax = fig.add_subplot(gs[2, 0:2]); binsz = 10
    for coh, c in COH.items():
        for tt, ls in [("whisker", "-"), ("catch", ":")]:
            curves = []
            for sid, g in C[(C.cohort == coh) & (C.ttype == tt)].groupby("session_id"):
                v = g.lick.to_numpy(float)
                nb = len(v) // binsz
                curves.append(np.pad(v[:nb * binsz].reshape(nb, binsz).mean(1), (0, 30 - nb) if nb < 30 else (0, 0),
                                     constant_values=np.nan)[:30])
            A = np.vstack(curves); n = np.sum(np.isfinite(A), 0); mu = np.nanmean(A, 0); se = np.nanstd(A, 0) / np.sqrt(np.maximum(n, 1))
            ok = n >= 5; x = (np.arange(30) + 0.5) * binsz
            ax.plot(x[ok], mu[ok], color=c, ls=ls, lw=1, label=f"{coh.replace('-', chr(8722))} {tt}")
            ax.fill_between(x[ok], (mu - se)[ok], (mu + se)[ok], color=c, alpha=0.15, lw=0)
    ax.set_xlabel("trial of that type (session order)"); ax.set_ylabel("P(lick)"); ax.legend(frameon=False, fontsize=6, ncol=2)
    ax.set_title("d  Licking on whisker vs catch trials (bins of 10 trials)", loc="left", fontsize=7.5)
    ax = fig.add_subplot(gs[2, 2]); rng = np.random.default_rng(0)
    for k, coh in enumerate(COH):
        v = L[L.cohort == coh].whisker_minus_catch.dropna()
        ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, s=5, color=COH[coh], alpha=0.5, lw=0)
        ax.errorbar(k + 0.25, v.mean(), v.std() / np.sqrt(len(v)), fmt="o", ms=3.5, mfc=COH[coh], color="k", capsize=2)
    t = LB.set_index("metric").loc["whisker_minus_catch"]
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks([0, 1], ["R+", "R−"])
    ax.set_ylabel("lick-prob. slope, whisker − catch\n(per 100 trials)")
    ax.set_title(f"e  R− vs 0: Wilcoxon p={t.p_wilcoxon_Rminus:.2g}, t p={t.p_ttest_Rminus:.2g}\n"
                 f"   R+ vs R−: MWU p={t.p_MWU_RplusRminus:.2g}, Welch p={t.p_Welch_RplusRminus:.2g}", loc="left", fontsize=6)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"null_layer.{ext}", dpi=250)
    pd.set_option("display.width", 250)
    print(T[T.region == "all"].round(4).to_string())
    print(LB.round(4).to_string())
    print("saved", OUT / "null_layer.png")


if __name__ == "__main__":
    main()
