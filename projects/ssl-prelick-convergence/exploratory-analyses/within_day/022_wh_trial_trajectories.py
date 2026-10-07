"""022 -- Reward-lick CD scores along the first N whisker-hit trials instead of session time (user 2026-10-07: "same types of
figures as e-h, but trial-by-trial projections, up to 10 and 20 whisker hit trials").

Data: 002 trial_scores.parquet (5-fold CD score c of every analysed event; SL = 0, AH = 1; held-out d' >= 0.3).
Per session: the references are time-matched, i.e. OLS lines of the SL and AH scores against normalised session time tau
(011 / Eq. 16), evaluated at the time of each whisker hit -- R- mice make most whisker hits early, so the first N whisker
hits cover different parts of the session in the two cohorts. For the first N whisker hits k = 1..N (time order):
  WH - SL(k) = c_WH,k - SLfit(tau_k)      AH - WH(k) = AHfit(tau_k) - c_WH,k
Sessions with fewer than N whisker hits are dropped. Per session: OLS slope of WH - SL and of WH - AH against x = (k - 1)
/ (N - 1), i.e. the change over the N trials (> 0: WH move toward AH relative to the reference; same convention as the
drift D of 011). Matched set: N = 10 restricted to the sessions that also enter N = 20 (separates N from dropout).
Statistics (unit = session): Wilcoxon vs 0 per group; day 0 vs expert per cohort (MWU, Welch); R+ vs R- per stage (MWU,
Welch, mouse-level cohort permutation); cohort x stage interaction (mouse-level permutation, 10,000).
Output: within_day/<ref>/wh_trials/<pop>/wh_trials_N{10,20}.{png,pdf,svg}, sessions_N{N}.csv, trials_N{N}.csv, tests.csv.
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
OUT = m51.WITHIN / "wh_trials"
COH = m62.COH
STG = [("learning", "Day 0"), ("expert", "Expert")]
NS = (10, 20)


def per_trial(T, N):
    """first N whisker hits per session with time-matched SL / AH references; per-session slopes over the N trials"""
    rows, sess = [], []
    for sid, g in T.dropna(subset=["cd"]).groupby("session_id"):
        w = g[g.cls == "WH"].sort_values("t")
        s, a = g[g.cls == "FA"], g[g.cls == "AH"]
        if len(w) < N or len(s) < 3 or len(a) < 3:
            continue
        fs, fa = np.polyfit(s.tau, s.cd, 1), np.polyfit(a.tau, a.cd, 1)
        w = w.iloc[:N]
        sl, ah = np.polyval(fs, w.tau.to_numpy()), np.polyval(fa, w.tau.to_numpy())
        meta = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0])
        x = np.arange(N) / (N - 1)
        d_sl, d_ah = w.cd.to_numpy() - sl, w.cd.to_numpy() - ah
        for k in range(N):
            rows.append(dict(meta, k=k + 1, wh=w.cd.iloc[k], WH_SL=d_sl[k], AH_WH=-d_ah[k], tau=w.tau.iloc[k]))
        sess.append(dict(meta, n_WH=int((g.cls == "WH").sum()), tau_last=w.tau.iloc[-1],
                         slope_WH_SL=np.polyfit(x, d_sl, 1)[0], slope_WH_AH=np.polyfit(x, d_ah, 1)[0],
                         slope_WH=np.polyfit(x, w.cd.to_numpy(), 1)[0]))
    return pd.DataFrame(rows), pd.DataFrame(sess)


def tests(S, label):
    out = []
    for col in ("slope_WH_SL", "slope_WH_AH", "slope_WH", "tau_last"):
        r = dict(set=label, measure=col)
        for c in ("R+", "R-"):
            for st, _ in STG:
                v = S[(S.cohort == c) & (S.stage == st)][col].dropna()
                g = f"{c} {st}"
                r[f"n {g}"], r[f"mean {g}"], r[f"sem {g}"] = len(v), v.mean(), v.sem()
                r[f"p_wilcoxon {g}"] = stats.wilcoxon(v).pvalue if len(v) >= 3 and (v != 0).any() else np.nan
            a_, e_ = (S[(S.cohort == c) & (S.stage == st)][col].dropna() for st in ("learning", "expert"))
            if len(a_) >= 3 and len(e_) >= 3:
                r[f"{c} day0 vs expert p_MWU"] = stats.mannwhitneyu(a_, e_).pvalue
                r[f"{c} day0 vs expert p_Welch"] = stats.ttest_ind(a_, e_, equal_var=False).pvalue
        for st, _ in STG:
            d = S[S.stage == st].dropna(subset=[col])
            x, y = d[d.cohort == "R+"][col], d[d.cohort == "R-"][col]
            if len(x) >= 3 and len(y) >= 3:
                r[f"{st} R+ vs R- p_MWU"] = stats.mannwhitneyu(x, y).pvalue
                r[f"{st} R+ vs R- p_Welch"] = stats.ttest_ind(x, y, equal_var=False).pvalue
                r[f"{st} R+ - R-"], r[f"{st} p_perm"] = m001.perm_interaction(d.assign(stage="learning"), col, kind="learning")
        r["cohort x stage"], r["cxs p_perm"] = m001.perm_interaction(S.dropna(subset=[col]), col, kind="stage")
        out.append(r)
    return out


def traj(ax, D, coh, col, N):
    k = np.arange(1, N + 1)
    for st, sl in STG:
        d = D[(D.cohort == coh) & (D.stage == st)]
        if d.empty:
            continue
        g = d.groupby("k")[col]
        mu, se = g.mean().reindex(k), g.sem().reindex(k)
        ls, mfc = ("--", "white") if st == "learning" else ("-", COH[coh])
        ax.fill_between(k, mu - se, mu + se, color=COH[coh], alpha=0.13, lw=0)
        ax.plot(k, mu, ls=ls, color=COH[coh], marker="o", ms=2.6, lw=1.0, mfc=mfc,
                label=f"{sl} ({d.session_id.nunique()})")
    ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_xlim(0.5, N + 0.5); ax.set_xlabel("Whisker-hit trial")
    ax.legend(frameon=False, fontsize=4.5, loc="upper left")


def figure(plt, D, S, Sm, N, out, pop):
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(m62.W_IN, 4.1 if Sm is None else 5.9))
    nrow = 2 if Sm is None else 3
    gs = fig.add_gridspec(nrow, 4, hspace=0.85, wspace=0.6, left=0.08, right=0.98, top=0.86 if nrow == 2 else 0.9, bottom=0.08)
    axs1 = []
    for j, (col, coh) in enumerate([(c, h) for c in ("WH_SL", "AH_WH") for h in ("R+", "R-")]):
        ax = fig.add_subplot(gs[0, j]); axs1.append(ax); ax.set_box_aspect(1)
        traj(ax, D, coh, col, N)
        ax.set_ylabel("WH − SL (score)" if col == "WH_SL" else "AH − WH (distance to AH)", fontsize=5)
        ax.set_title(f"{coh.replace('-', '−')}: " + ("WH relative to SL" if col == "WH_SL" else "WH distance to AH"),
                     color=COH[coh], fontsize=5.6)
    for a_, b_ in ((0, 1), (2, 3)):
        yl = (min(axs1[a_].get_ylim()[0], axs1[b_].get_ylim()[0]), max(axs1[a_].get_ylim()[1], axs1[b_].get_ylim()[1]))
        axs1[a_].set_ylim(*yl); axs1[b_].set_ylim(*yl)
    rows_ax = [axs1]
    for i, (SS, tag) in enumerate([(S, f"all sessions with ≥ {N} WH")] + ([] if Sm is None else [(Sm, "sessions with ≥ 20 WH")])):
        axs = []
        for j, (col, ttl) in enumerate([("slope_WH_SL", f"Change of WH − SL over {N} WH"), ("slope_WH_AH", f"Change of WH − AH over {N} WH"),
                                        ("slope_WH", f"Change of WH score over {N} WH"), ("tau_last", f"Session time of WH #{N}")]):
            ax = fig.add_subplot(gs[1 + i, j]); axs.append(ax); ax.set_box_aspect(1)
            ref = [(0, "0.6")] if col != "tau_last" else []
            m62.dots_panel(ax, SS, col, (tag if j == 0 else ""), rng, f"WHtr N{N} {i} {col}", ttl, ref=ref)
            ax.set_title(ax.get_title(), fontsize=5.0)
        rows_ax.append(axs)
    for r_, letters in zip(rows_ax, ("abcd", "efgh", "ijkl")):
        m62.letter_row(fig, r_, letters)
    fig.suptitle(f"Reward-lick CD along the first {N} whisker hits (time-matched SL / AH references; {pop})",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, f"wh_trials_N{N}"); plt.close(fig)


def main(a):
    plt = m62.setup()
    T0 = pd.read_parquet(m51.WITHIN / "slopes" / "trial_scores.parquet")
    for pop in ("learners", "all"):
        T = m61.learner_filter(T0) if pop == "learners" else T0
        out = OUT / pop; out.mkdir(parents=True, exist_ok=True)
        R, res = [], {}
        for N in NS:
            D, S = per_trial(T, N)
            D.to_csv(out / f"trials_N{N}.csv", index=False); S.to_csv(out / f"sessions_N{N}.csv", index=False)
            res[N] = (D, S)
            R += tests(S, f"N{N}")
        D10, S10 = res[10]
        keep = set(res[20][1].session_id)
        Sm = S10[S10.session_id.isin(keep)]
        R += tests(Sm, "N10 matched to N20 sessions")
        figure(plt, D10, S10, Sm, 10, out, pop)
        figure(plt, *res[20], None, 20, out, pop)
        TT = pd.DataFrame(R); TT.to_csv(out / "tests.csv", index=False)
        pd.DataFrame(m62.STATS).to_csv(out / "stats_dots.csv", index=False); m62.STATS.clear()
        cols = ["set", "measure", "mean R+ learning", "mean R- learning", "learning p_perm", "mean R+ expert", "mean R- expert",
                "expert p_perm", "cxs p_perm", "n R+ learning", "n R- learning", "n R+ expert", "n R- expert"]
        with pd.option_context("display.width", 250):
            print(pop); print(TT[[c for c in cols if c in TT]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
