"""016 -- learning analyses of the full run (haas), from the per-trial measures of 015.

Unit of analysis: session (= mouse; one whisker day-0 session per mouse). Population scopes: all mice, learners only
(sheet learning_category good / moderate), each split R+ / R- (cohort per mouse from the reference sheet).
A. Learning-curve correlation: whisker trials; learning curve = Gaussian-smoothed (sigma 5 whisker trials) whisker
   P(lick); the measure is smoothed the same way; r = Pearson(measure, curve). Null: linear, non-wrapping shifts of the
   measure against the curve (lags 10-50 % of the trials, both directions; shift_null convention); excess = r - mean
   null. Group: sign-flip permutation of the per-mouse excess (10000 flips), two-sided.
B. Change at the learning trial (ssl-learning-trial-identification 017 lt_final, whisker-trial index; learners with an
   LT): delta = mean of the 20 whisker trials after the LT - mean of the 20 before; null = the same delta at every
   pseudo-LT with >= 20 whisker trials on each side and >= 10 trials from the real LT; excess = delta - mean null; group
   sign-flip.
C. Cohort x trial interaction: per session and trial type (whisker, auditory, no-stim), OLS slope of the measure over
   the normalised trial position (0 = first, 1 = last kept trial); whisker slope R+ vs R- and the difference of slopes
   (whisker - auditory, whisker - no-stim; within-session controls) R+ vs R-: Mann-Whitney + Welch, uncorrected,
   exploratory; R- serves as the time control (same session time, whisker not rewarded).
D. Pre-stim actions predicting whisker hits (lick on whisker trials): per session, AUC of the pre-stim measure for hit
   vs miss in the early and late half of the whisker trials (>= 5 hits and >= 5 misses per half); group: AUC - 0.5
   sign-flip per half; late - early: Wilcoxon signed-rank.
E. R+ vs R- per measure (session mean per epoch, whisker trials): Mann-Whitney + Welch, uncorrected, exploratory.
Output: tables/016_{A,B,C,D,E}_*.csv, figures/016_*.png|pdf|svg, provenance/016.json.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402

SIG = 5
LO, HI = 0.1, 0.5
NFLIP = 10000
LT_TABLE = L.ROOT / "combined_results_ks4" / "ssl-learning-trial-identification" / "artifacts" / "017_learning_trials_final.csv"
rng = np.random.default_rng(0)
plt.rcParams.update({"font.size": 6, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})


def signflip(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    obs = abs(x.mean())
    s = rng.choice([-1, 1], size=(NFLIP, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= obs) + 1) / (NFLIP + 1))


def two_tests(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def shift_null_r(m, p):
    n = len(m)
    out = []
    for k in range(max(1, int(LO * n)), max(1, int(HI * n)) + 1):
        for dr in (0, 1):
            a, b = (m[k:], p[:n - k]) if dr else (m[:n - k], p[k:])
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() > 10 and np.std(a[ok]) > 0:
                out.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return np.array(out)


def smooth_nan(x, sig):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.full(len(x), np.nan)
    num = gaussian_filter1d(np.where(ok, x, 0), sig, mode="nearest")
    den = gaussian_filter1d(ok.astype(float), sig, mode="nearest")
    return np.where(den > 0.2, num / den, np.nan)


def main():
    D = pd.read_parquet(L.TABLES / "full_trial_measures.parquet")
    S = pd.read_csv(L.TABLES / "full_sessions.csv")
    coh, lrn = dict(zip(S.session_id, S.reward_group)), dict(zip(S.session_id, S.learner))
    meas = [c for c in D.columns if c.split("_")[0] in ("act", "meta", "trans", "lat")]
    lt = pd.read_csv(LT_TABLE) if LT_TABLE.exists() else pd.DataFrame(columns=["session_id", "lt_final"])
    ltd = dict(zip(lt.session_id, lt.lt_final))
    A, B, C, Dd, E = [], [], [], [], []
    for sid, g in D.groupby("session_id"):
        w = g[g.stim == "whisker"]
        for ep, ge in w.groupby("epoch"):
            ge = ge.sort_values("trial")
            curve = smooth_nan(ge.lick_flag.to_numpy(float), SIG)
            for m in meas:
                x = ge[m].to_numpy(float)
                if np.isfinite(x).sum() < 20:
                    continue
                ms = smooth_nan(x, SIG)
                ok = np.isfinite(ms) & np.isfinite(curve)
                if ok.sum() < 20 or np.std(ms[ok]) == 0:
                    continue
                r = float(np.corrcoef(ms[ok], curve[ok])[0, 1])
                nl = shift_null_r(ms, curve)
                A.append(dict(session_id=sid, epoch=ep, measure=m, r=r, null_mean=float(np.nanmean(nl)) if len(nl) else np.nan,
                              excess=r - np.nanmean(nl) if len(nl) else np.nan, n_trials=int(ok.sum())))
                # B: learning trial
                ltv = ltd.get(sid, np.nan)
                if np.isfinite(ltv):
                    idx = ge.whisker_trial_index.to_numpy()
                    pos = int(np.searchsorted(idx, int(ltv)))
                    if 20 <= pos <= len(x) - 20:
                        dlt = lambda p: np.nanmean(x[p:p + 20]) - np.nanmean(x[p - 20:p])  # noqa: E731
                        pseudo = [dlt(p) for p in range(20, len(x) - 20 + 1) if abs(p - pos) >= 10]
                        if len(pseudo) >= 5:
                            B.append(dict(session_id=sid, epoch=ep, measure=m, delta=dlt(pos), null_mean=float(np.nanmean(pseudo)),
                                          excess=dlt(pos) - float(np.nanmean(pseudo)),
                                          pct=float(np.mean(np.array(pseudo) <= dlt(pos)))))
                # D: pre-stim prediction of hits
                if ep == "pre" and not m.startswith("lat"):
                    y = ge.lick_flag.to_numpy()
                    half = len(y) // 2
                    for name, sl in (("early", slice(0, half)), ("late", slice(half, None))):
                        yy, xx = y[sl], x[sl]
                        ok2 = np.isfinite(xx)
                        if (yy[ok2] == 1).sum() >= 5 and (yy[ok2] == 0).sum() >= 5 and np.std(xx[ok2]) > 0:
                            Dd.append(dict(session_id=sid, measure=m, half=name, auc=roc_auc_score(yy[ok2], xx[ok2])))
        # C: slopes per trial type
        tt = g.trial.to_numpy()
        tpos = (tt - tt.min()) / max(tt.max() - tt.min(), 1)
        g = g.assign(tpos=tpos)
        for (ep, st), gs in g.groupby(["epoch", "stim"]):
            for m in meas:
                x, tp = gs[m].to_numpy(float), gs.tpos.to_numpy()
                ok = np.isfinite(x)
                if ok.sum() >= 10 and np.std(x[ok]) > 0:
                    C.append(dict(session_id=sid, epoch=ep, stim=st, measure=m, slope=float(np.polyfit(tp[ok], x[ok], 1)[0])))
        for ep, ge in w.groupby("epoch"):
            for m in meas:
                E.append(dict(session_id=sid, epoch=ep, measure=m, mean=float(ge[m].mean())))
    A, B, C, Dd, E = map(pd.DataFrame, (A, B, C, Dd, E))
    for X in (A, B, C, Dd, E):
        if len(X):
            X["cohort"] = X.session_id.map(coh)
            X["learner"] = X.session_id.map(lrn)
    scopes = {"all": lambda X: X, "learners": lambda X: X[X.learner == "learner"]}
    # group statistics
    GA, GB, GC, GD, GE = [], [], [], [], []
    for sc, f in scopes.items():
        for (ep, m, c), g in f(A).groupby(["epoch", "measure", "cohort"]):
            GA.append(dict(scope=sc, epoch=ep, measure=m, cohort=c, n=len(g), r_mean=g.r.mean(), excess_mean=g.excess.mean(),
                           p_signflip=signflip(g.excess)))
        if len(B):
            for (ep, m, c), g in f(B).groupby(["epoch", "measure", "cohort"]):
                GB.append(dict(scope=sc, epoch=ep, measure=m, cohort=c, n=len(g), excess_mean=g.excess.mean(), p_signflip=signflip(g.excess)))
        Cw = f(C)
        for (ep, m), g in Cw.groupby(["epoch", "measure"]):
            p = g.pivot_table(index=["session_id", "cohort"], columns="stim", values="slope").reset_index()
            if "whisker_trial" in p.columns:
                p = p.rename(columns={"whisker_trial": "whisker", "auditory_trial": "auditory", "no_stim_trial": "no_stim"})
            for contrast in ("whisker", "whisker-auditory", "whisker-no_stim"):
                if contrast == "whisker":
                    v = p.get("whisker")
                else:
                    a_, b_ = contrast.split("-")
                    v = p.get(a_) - p.get(b_) if a_ in p and b_ in p else None
                if v is None:
                    continue
                vp, vm = v[p.cohort == "R+"], v[p.cohort == "R-"]
                pmw, pw = two_tests(vp, vm)
                GC.append(dict(scope=sc, epoch=ep, measure=m, contrast=contrast, mean_Rplus=vp.mean(), mean_Rminus=vm.mean(),
                               n_Rplus=int(vp.notna().sum()), n_Rminus=int(vm.notna().sum()), p_mannwhitney=pmw, p_welch=pw))
        if len(Dd):
            for (m, c), g in f(Dd).groupby(["measure", "cohort"]):
                pv = g.pivot_table(index="session_id", columns="half", values="auc")
                row = dict(scope=sc, measure=m, cohort=c)
                for h in ("early", "late"):
                    if h in pv:
                        row[f"auc_{h}"] = pv[h].mean()
                        row[f"n_{h}"] = int(pv[h].notna().sum())
                        row[f"p_{h}_signflip"] = signflip(pv[h] - 0.5)
                if {"early", "late"} <= set(pv.columns):
                    q = pv.dropna()
                    row["p_late_vs_early_wilcoxon"] = float(stats.wilcoxon(q.late, q.early).pvalue) if len(q) >= 5 else np.nan
                GD.append(row)
        for (ep, m), g in f(E).groupby(["epoch", "measure"]):
            pmw, pw = two_tests(g[g.cohort == "R+"]["mean"], g[g.cohort == "R-"]["mean"])
            GE.append(dict(scope=sc, epoch=ep, measure=m, mean_Rplus=g[g.cohort == "R+"]["mean"].mean(),
                           mean_Rminus=g[g.cohort == "R-"]["mean"].mean(), p_mannwhitney=pmw, p_welch=pw))
    out = {"A_curve_correlation": (A, GA), "B_learning_trial": (B, GB), "C_cohort_trial": (C, GC), "D_prestim_auc": (Dd, GD),
           "E_rplus_rminus": (E, GE)}
    for k, (per, grp) in out.items():
        per.to_csv(L.TABLES / f"016_{k}_per_session.csv", index=False)
        pd.DataFrame(grp).to_csv(L.TABLES / f"016_{k}_group.csv", index=False)
    figures(pd.DataFrame(GA), pd.DataFrame(GB), pd.DataFrame(GC), pd.DataFrame(GD))
    json.dump(dict(script="016_learning_stats.py", sigma_trials=SIG, shift_lags=[LO, HI], n_flips=NFLIP,
                   n_sessions=int(D.session_id.nunique()), lt_table=str(LT_TABLE), versions=L.versions()),
              open(L.RES / "provenance" / "016.json", "w"), indent=1)


KEY = ["act_still", "act_whisking", "act_jaw", "act_lick", "act_still_nl", "act_whisking_nl", "act_jaw_nl",
       "trans_side", "trans_top", "trans_side_nl", "trans_top_nl"]


def figures(GA, GB, GC, GD):
    L.FIGS.mkdir(exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 7.0))
    for ax, (G, val, title) in zip(axs.ravel(), ((GA, "excess_mean", "A  Correlation with the whisker P(lick) curve\n(r - linear-shift null mean)"),
                                                 (GB, "excess_mean", "B  Change at the learning trial\n(20 after - 20 before, minus pseudo-LT mean)"))):
        if not len(G):
            ax.axis("off")
            continue
        g = G[(G.scope == "all") & G.measure.isin(KEY + [m for m in G.measure.unique() if m.startswith("meta_") and not m.endswith("_nl")])]
        ms = list(dict.fromkeys(g.measure))
        cols = [(ep, c) for ep in ("pre", "resp", "post", "iti") for c in ("R+", "R-")]
        M = np.full((len(ms), len(cols)), np.nan)
        P = np.full_like(M, np.nan)
        for i, m in enumerate(ms):
            for j, (ep, c) in enumerate(cols):
                q = g[(g.measure == m) & (g.epoch == ep) & (g.cohort == c)]
                if len(q):
                    M[i, j], P[i, j] = q[val].iloc[0], q.p_signflip.iloc[0]
        v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
        im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        for i in range(len(ms)):
            for j in range(len(cols)):
                if P[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f"{ep}\n{c}" for ep, c in cols], fontsize=5)
        for lab_, c in zip(ax.get_xticklabels(), cols):
            lab_.set_color(L.COH_COL[c[1]])
        ax.set_yticks(range(len(ms)))
        ax.set_yticklabels(ms, fontsize=4.5)
        ax.set_title(title + "; * p < 0.05 sign-flip (uncorrected)", fontsize=6, loc="left")
        plt.colorbar(im, ax=ax, fraction=0.04)
    ax = axs[1, 0]
    if len(GC):
        g = GC[(GC.scope == "all") & (GC.contrast == "whisker-auditory") & GC.measure.isin(KEY)]
        g = g.assign(p=g[["p_mannwhitney", "p_welch"]].min(1)).sort_values("p").head(12)
        ax.barh(range(len(g)), g.mean_Rplus - g.mean_Rminus, color=["k" if p < 0.05 else "#999999" for p in g.p], lw=0)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f"{m} ({e})" for m, e in zip(g.measure, g.epoch)], fontsize=4.5)
        ax.set_xlabel("R+ minus R- difference of slopes (whisker - auditory)")
        ax.axvline(0, color="k", lw=0.5)
    ax.set_title("C  Cohort x trial interaction (12 smallest p; black p < 0.05,\nexploratory, uncorrected)", fontsize=6, loc="left")
    ax = axs[1, 1]
    if len(GD):
        g = GD[(GD.scope == "all") & GD.measure.isin(KEY)]
        for c, off in (("R+", -0.15), ("R-", 0.15)):
            q = g[g.cohort == c]
            x = np.arange(len(q))
            ax.scatter(x + off, q.auc_early, marker="o", facecolor="none", edgecolor=L.COH_COL[c], s=10, label=f"{c} early")
            ax.scatter(x + off, q.auc_late, marker="o", color=L.COH_COL[c], s=10, label=f"{c} late")
        ax.set_xticks(range(len(g[g.cohort == "R+"])))
        ax.set_xticklabels(list(g[g.cohort == "R+"].measure), rotation=60, ha="right", fontsize=4.5)
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.set_ylabel("AUC (pre-stim measure -> whisker hit)")
        ax.legend(frameon=False, fontsize=4.5, ncol=2)
    ax.set_title("D  Pre-stim actions predicting whisker hits\n(open = early half, filled = late half)", fontsize=6, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"016_learning_summary.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
