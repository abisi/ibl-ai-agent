"""Trial-level time course of whisker-hit activity within a session, on a fixed session axis, compared with experts.

Question: within the learning day, does pre-lick activity on whisker hits (WH) drift along the axis separating
spontaneous licks (SL, unrewarded) from auditory hits (AH, rewarded), in opposite directions in R+ and R- mice, and does
the end of day 0 reach the expert level? No split point: every trial contributes, with its time in the session.

Events (051, PRELICK_REF=sl): WH, AH, SL ("FA"), 100 ms before the corrected first lick; trials active, perf != 6,
warm-up cut, A1 end trim. Units: good + mua, mean raw pre-lick rate >= 0.1 Hz; whole brain; sessions with >= 4 WH (MIN_WH;
>= 8 before 2026-10-05), >= 8 AH and >= 8 SL events and >= 5 units.

Fixed session axis (5-fold stratified CV over the AH and SL events of the whole session; units z-scored on training
folds):
  md   unit-norm mean-difference axis AH - SL of the training folds (as lambda): held-out AH / SL events and all WH
       events (averaged over folds) are projected; one normalisation for the session (pooled held-out SL mean = 0, AH
       mean = 1; score ~ lambda per trial: 0 = like SL, 1 = like AH); kept only if the held-out AH-vs-SL d' >= 0.3
  dec  L2 logistic-regression decoder (C = 0.05, balanced): P(AH) of held-out AH / SL events and fold-averaged P(AH) of
       WH events
Time: normalised position in the session, tau = (t - t_first) / (t_last - t_first) over all kept events (0 = start,
1 = end).
Per session and class: OLS slope of the trial scores against tau (change over the whole session, in score units) and
the fitted value at tau = 0 and tau = 1. Relative slopes: WH - SL (WH moving relative to the unrewarded reference
measured at the same times) and WH - AH. Binned trajectories: 5 equal tau bins, per-session class means, then mean +-
s.e.m. over sessions; the expert level (session mean of all expert trials) is shown as a band.
Comparison with experts: day-0 fitted start (tau = 0) and end (tau = 1) WH scores vs the expert session-mean WH score
(MWU, Welch), per cohort.
Statistics (unit = session): slope vs 0 per group (Wilcoxon, one-sample t); day 0 R+ vs R- (MWU, Welch, mouse-level
cohort permutation); cohort x stage on the slopes (mouse-level permutation). Populations: all mice and learners.
Output: combined_results_ks4/ssl-prelick-convergence/within_day/<ref>/slopes/ (incl. trial_scores.parquet: one row per event, CD projection
`cd` and decoder P(AH) `p_ah`, used by the mixed model 009)
"""
import argparse
import importlib
import json
import multiprocessing as mp
import pathlib
import sys
import time
import warnings

import os

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent                                         # across-day scripts (projects merged 2026-10-07)
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m001 = importlib.import_module("001_within_session_halves")
UNIT_SET = ("good", "mua")
MIN_UNITS, MIN_EV, C_REG, N_BINS, MIN_DPRIME = 5, 8, 0.05, 5, 0.3
# 2026-10-05: separate minimum for whisker hits (env SSL_MIN_WH, default MIN_EV). WH never enter the axis (built from AH vs SL),
# so a lower WH minimum keeps the axis quality and admits sessions with few whisker licks (e.g. R- experts); outputs then go to
# slopes_wh<N>/ so the default results stay untouched
# default 4 since 2026-10-05 (user: "switch that threshold on the within-day project too"); the earlier >= 8 results are in
# slopes_wh8/
MIN_WH = int(os.environ.get("SSL_MIN_WH", 4))
OUT = m51.WITHIN / ("slopes" if MIN_WH == 4 else f"slopes_wh{MIN_WH}")
AXES = ["md", "dec"]
CLS = ["WH", "AH", "FA"]


def trial_scores(X, lab, seed=0):
    """per event: normalised mean-difference score and decoder P(AH) (held-out for AH / SL, fold-averaged for WH)"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    m = np.isin(lab, ["AH", "FA"]); w = np.where(lab == "WH")[0]; ia = np.where(m)[0]
    y = (lab[m] == "AH").astype(int)
    S = {a: np.full(len(lab), np.nan) for a in AXES}
    Sw = {a: [] for a in AXES}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(ia, y):
        A = X[ia[tr]]; mu, sd = A.mean(0), A.std(0); sd[sd == 0] = 1
        z = lambda q: (q - mu) / sd
        ax_ = z(A[y[tr] == 1]).mean(0) - z(A[y[tr] == 0]).mean(0)
        ax_ = ax_ / max(np.linalg.norm(ax_), 1e-12)                 # unit axis: projections comparable across folds
        S["md"][ia[te]] = z(X[ia[te]]) @ ax_; Sw["md"].append(z(X[w]) @ ax_)
        c = LogisticRegression(C=C_REG, class_weight="balanced", max_iter=2000).fit(z(A), y[tr])
        S["dec"][ia[te]] = c.predict_proba(z(X[ia[te]]))[:, 1]; Sw["dec"].append(c.predict_proba(z(X[w]))[:, 1])
    for a in AXES:
        if Sw[a]:
            S[a][w] = np.mean(Sw[a], 0)
    # md: one normalisation for the whole session (pooled held-out SL mean = 0, AH mean = 1); axis kept only if the
    # held-out separation is reliable (d' >= MIN_DPRIME), otherwise the per-trial scores are undefined
    pa, pf = S["md"][lab == "AH"], S["md"][lab == "FA"]
    sp = np.sqrt(0.5 * (pa.var(ddof=1) + pf.var(ddof=1)))
    dprime = (pa.mean() - pf.mean()) / sp if sp > 0 else np.nan
    if not np.isfinite(dprime) or dprime < MIN_DPRIME:
        S["md"][:] = np.nan
    else:
        S["md"] = (S["md"] - pf.mean()) / (pa.mean() - pf.mean())
    S["dprime"] = dprime
    return S


def session(args):
    sid, f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["trial_start"].astype(float)[o]
    if (lab == "WH").sum() < MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < MIN_EV:
        return None
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(UNIT_SET).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    if ok.sum() < MIN_UNITS:
        return None
    S = trial_scores(X[ok].T, lab, seed)
    tau = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    row = dict(meta, n_units=int(ok.sum()), dprime=S.pop("dprime"), **{f"n_{c}": int((lab == c).sum()) for c in CLS})
    traj = []
    for a in AXES:
        for c in CLS:
            k = (lab == c) & np.isfinite(S[a])
            if k.sum() < (MIN_WH if c == "WH" else MIN_EV):
                continue
            sl, ic = np.polyfit(tau[k], S[a][k], 1)
            row[f"{a}_{c}_slope"], row[f"{a}_{c}_start"], row[f"{a}_{c}_end"] = sl, ic, ic + sl
            row[f"{a}_{c}_mean"] = S[a][k].mean()
            row[f"{a}_{c}_r"] = stats.spearmanr(tau[k], S[a][k])[0]
            b = np.minimum((tau[k] * N_BINS).astype(int), N_BINS - 1)
            for i in range(N_BINS):
                if (b == i).sum():
                    traj.append(dict(meta, axis=a, cls=c, bin=i, score=S[a][k][b == i].mean(), n=int((b == i).sum())))
        for c in ("FA", "AH"):
            if f"{a}_WH_slope" in row and f"{a}_{c}_slope" in row:
                row[f"{a}_WH-{c}_slope"] = row[f"{a}_WH_slope"] - row[f"{a}_{c}_slope"]
    # whisker-hit and spontaneous-lick rates per tau bin (behavioural context)
    dur = (t.max() - t.min()) / 60 / N_BINS
    b = np.minimum((tau * N_BINS).astype(int), N_BINS - 1)
    for c in CLS:
        for i in range(N_BINS):
            traj.append(dict(meta, axis="rate", cls=c, bin=i, score=((lab == c) & (b == i)).sum() / dur if dur > 0 else np.nan, n=0))
    trials = pd.DataFrame(dict(session_id=meta["session_id"], mouse_id=meta["mouse_id"], cohort=meta["cohort"],
                               stage=meta["stage"], cls=lab, tau=tau, t=t, cd=S["md"], p_ah=S["dec"]))
    trials = trials[trials.cls.isin(CLS)]
    return row, traj, trials


def tests(D):
    rows = []
    cols = [c for c in D.columns if c.endswith(("_slope", "_start", "_end", "_mean"))]
    for col in cols:
        r = dict(measure=col)
        for c in ("R+", "R-"):
            for s_ in ("learning", "expert"):
                v = D[(D.cohort == c) & (D.stage == s_)][col].dropna()
                g = f"{c} {s_}"
                r[f"n {g}"], r[f"mean {g}"], r[f"sem {g}"] = len(v), v.mean(), v.sem()
                if len(v) >= 3 and col.endswith("_slope"):
                    r[f"p_wilcoxon {g}"] = stats.wilcoxon(v).pvalue
                    r[f"p_t {g}"] = stats.ttest_1samp(v, 0).pvalue
        a = D[(D.cohort == "R+") & (D.stage == "learning")][col].dropna()
        b = D[(D.cohort == "R-") & (D.stage == "learning")][col].dropna()
        if len(a) >= 3 and len(b) >= 3:
            r["day0 R+ vs R- p_MWU"] = stats.mannwhitneyu(a, b).pvalue
            r["day0 R+ vs R- p_Welch"] = stats.ttest_ind(a, b, equal_var=False).pvalue
            r["day0 R+ - R-"], r["day0 p_perm"] = m001.perm_interaction(D, col, kind="learning")
        r["cohort x stage"], r["cxs p_perm"] = m001.perm_interaction(D, col, kind="stage")
        rows.append(r)
    # day-0 start / end vs expert level (WH), per cohort and axis
    for a in AXES:
        for c in ("R+", "R-"):
            L = D[(D.cohort == c) & (D.stage == "learning")]; E = D[(D.cohort == c) & (D.stage == "expert")][f"{a}_WH_mean"].dropna()
            for when in ("start", "end"):
                v = L[f"{a}_WH_{when}"].dropna()
                if len(v) >= 3 and len(E) >= 3:
                    rows.append(dict(measure=f"{a}_WH day0 {when} vs expert {c}", **{f"mean day0 {when}": v.mean(), "mean expert": E.mean(),
                                     "p_MWU": stats.mannwhitneyu(v, E).pvalue,
                                     "p_Welch": stats.ttest_ind(v, E, equal_var=False).pvalue, "n_day0": len(v), "n_expert": len(E)}))
    return pd.DataFrame(rows)


def figures(D, TR, T, out, pop):
    m62 = importlib.import_module("062_pub_convergence_figures")
    plt = m62.setup()
    COH, CL, GROUPS, GLAB = m62.COH, m62.CL, m62.GROUPS, m62.GLAB
    xb = (np.arange(N_BINS) + 0.5) / N_BINS
    fig = plt.figure(figsize=(m62.W_IN, 7.2))
    gs = fig.add_gridspec(3, 4, hspace=0.95, wspace=0.6, left=0.08, right=0.98, top=0.9, bottom=0.07)
    # row 1: binned trajectories (md axis) per group, WH / AH / SL, expert WH level as band in the learning panels
    axs1 = []
    for j, k in enumerate(GROUPS):
        ax = fig.add_subplot(gs[0, j]); axs1.append(ax)
        g = TR[(TR.axis == "md") & (TR.cohort == k[0]) & (TR.stage == k[1])]
        for c in CLS:
            q = g[g.cls == c].groupby("bin").score.agg(["mean", "sem"]).reindex(range(N_BINS))
            ax.fill_between(xb, q["mean"] - q["sem"], q["mean"] + q["sem"], color=CL[c], alpha=0.2, lw=0)
            ax.plot(xb, q["mean"], color=CL[c], lw=1.0, marker="o", ms=2, label=m62.CLAB[c])
        if k[1] == "learning":
            e = D[(D.cohort == k[0]) & (D.stage == "expert")]["md_WH_mean"].dropna()
            if len(e):
                ax.axhspan(e.mean() - e.sem(), e.mean() + e.sem(), color=CL["WH"], alpha=0.15, lw=0)
                ax.axhline(e.mean(), color=CL["WH"], lw=0.7, ls=(0, (3, 2)))
                ax.text(1.0, e.mean(), "expert WH", fontsize=4.3, color="#b07e00", ha="right", va="bottom")
        ax.set_xlabel("Time in session (normalised)"); ax.set_xlim(0, 1)
        ax.set_title(f"{GLAB[k]} ({g.session_id.nunique()} sessions)", color=COH[k[0]], fontsize=5.6)
        if j == 0:
            ax.set_ylabel("Score on SL → AH axis\n(0 = SL, 1 = AH)"); ax.legend(frameon=False, fontsize=4.5, loc="upper left")
    yl = (min(a.get_ylim()[0] for a in axs1), max(a.get_ylim()[1] for a in axs1))
    for a in axs1:
        a.set_ylim(*yl)
    # row 2: slopes (md): WH, WH - SL; decoder WH; WH - SL decoder
    panels = [("md_WH_slope", "WH slope (score / session)", "WH on the SL → AH axis"),
              ("md_WH-FA_slope", "WH − SL slope", "WH relative to SL"),
              ("dec_WH_slope", "Slope of P(AH | WH)", "Decoder: WH"),
              ("dec_WH-FA_slope", "WH − SL slope (P(AH))", "Decoder: WH relative to SL")]
    axs2 = []
    for j, (col, yl_, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j]); axs2.append(ax)
        m62.dots_panel(ax, D, col, yl_, np.random.default_rng(0), f"slope {col}", ttl, ref=[(0, "0.6")])
        r = T[T.measure == col]
        if len(r):
            r = r.iloc[0]
            ax.set_title(f"{ttl}\nday 0 R+ vs R− {m62.fmt_p(r.get('day0 p_perm', np.nan))}; × stage {m62.fmt_p(r['cxs p_perm'])}",
                         fontsize=5.0)
    # row 3: day-0 start / end vs expert (md WH), behavioural rates
    axs3 = []
    for j, c in enumerate(["R+", "R-"]):
        ax = fig.add_subplot(gs[2, j]); axs3.append(ax)
        L = D[(D.cohort == c) & (D.stage == "learning")]; E = D[(D.cohort == c) & (D.stage == "expert")]
        vals = [L["md_WH_start"].dropna(), L["md_WH_end"].dropna(), E["md_WH_mean"].dropna()]
        for i, v in enumerate(vals):
            ax.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), v, s=5, color=COH[c], alpha=0.5, lw=0)
            ax.errorbar(i + 0.25, v.mean(), v.sem(), fmt="o", ms=4, color=COH[c], mfc="white" if i < 2 else COH[c], capsize=0)
        ax.set_xticks([0, 1, 2], ["Day 0\nstart", "Day 0\nend", "Expert"])
        ax.set_ylabel("WH score (0 = SL, 1 = AH)")
        ps = [T[T.measure == f"md_WH day0 {w} vs expert {c}"] for w in ("start", "end")]
        txt = ", ".join(f"{w} vs expert {m62.fmt_p(p.iloc[0]['p_MWU'])}" for w, p in zip(("start", "end"), ps) if len(p))
        ax.set_title(f"{c.replace('-', '−')}: day-0 fit vs expert\n{txt}", color=COH[c], fontsize=5.0)
    for j, c in enumerate(["WH", "FA"]):
        ax = fig.add_subplot(gs[2, 2 + j]); axs3.append(ax)
        for k in GROUPS:
            q = TR[(TR.axis == "rate") & (TR.cls == c) & (TR.cohort == k[0]) & (TR.stage == k[1])].groupby("bin").score.agg(
                ["mean", "sem"]).reindex(range(N_BINS))
            if q["mean"].isna().all():
                continue
            ax.errorbar(xb, q["mean"], q["sem"], color=COH[k[0]], ls="-" if k[1] == "expert" else (0, (2, 1.5)), lw=0.9,
                        marker="o", ms=2, capsize=0, label=GLAB[k])
        ax.set_xlabel("Time in session (normalised)"); ax.set_ylabel("Events / min")
        ax.set_title(f"{m62.CLAB[c]} rate", fontsize=5.4)
        if j == 0:
            ax.legend(frameon=False, fontsize=4.3)
    m62.letter_row(fig, axs1, "abcd"); m62.letter_row(fig, axs2, "efgh"); m62.letter_row(fig, axs3, "ijkl")
    fig.suptitle(f"Within-session time course on a fixed session axis (trial level; reference: spontaneous licks; {pop})",
                 x=0.08, y=0.985, ha="left", fontsize=7.2, weight="bold")
    m62.save(fig, out, "trial_slopes"); plt.close(fig)


def main(a):
    t0 = time.time()
    m61 = importlib.import_module("061_roc_prelick_learners")
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for i, r in enumerate(ss.itertuples()):
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            jobs.append((r.session_id, f, dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0],
                                               stage=Ws.stage.iloc[0]), Ws[["electrode_group", "cluster_id", "quality_label"]], i))
    if a.test:
        jobs = jobs[:a.test]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.replot:
        D, TR = pd.read_csv(OUT / "trial_slopes_sessions.csv"), pd.read_csv(OUT / "trial_slopes_trajectories.csv")
    else:
        rows, traj, trials = [], [], []
        with mp.get_context("fork").Pool(a.n_proc) as pool:
            for res in pool.imap_unordered(session, jobs):
                if res:
                    rows.append(res[0]); traj += res[1]; trials.append(res[2])
        D, TR = pd.DataFrame(rows), pd.DataFrame(traj)
        D.to_csv(OUT / "trial_slopes_sessions.csv", index=False); TR.to_csv(OUT / "trial_slopes_trajectories.csv", index=False)
        pd.concat(trials, ignore_index=True).to_parquet(OUT / "trial_scores.parquet", index=False)   # single-trial CD projections
    for pop in ["all", "learners"]:
        Dp = m61.learner_filter(D) if pop == "learners" else D
        TRp = TR[TR.session_id.isin(Dp.session_id)]
        out = OUT / pop; out.mkdir(exist_ok=True)
        T = tests(Dp); T.to_csv(out / "trial_slopes_tests.csv", index=False)
        figures(Dp, TRp, T, out, pop)
    json.dump(dict(script="002_trial_slopes.py", ref=m51.REF, min_events=MIN_EV, min_wh=MIN_WH, n_bins=N_BINS, C=C_REG, unit_set=UNIT_SET,
                   min_fr=m51.MIN_FR, n_sessions=int(D.session_id.nunique()), runtime_min=round((time.time() - t0) / 60, 1)),
              open(OUT / "provenance.json", "w"), indent=1)
    print("ALL DONE", OUT, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    ap.add_argument("--test", type=int, default=0)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
