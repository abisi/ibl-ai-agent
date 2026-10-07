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
Figure (user 2026-10-07): every single trial plotted (dots) with the mean +- s.e.m. per trial number, R+ and R-
overlaid within each stage; per-session changes contrasted R+ vs R- within stage.
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


def trials_panel(ax, D, st, col, N, rng):
    """every single whisker-hit trial of every session (dots, R+ left / R- right of each trial number) and the mean +- s.e.m.
    over sessions per trial number, the two cohorts overlaid within one stage"""
    k = np.arange(1, N + 1)
    for coh, dx in (("R+", -0.17), ("R-", 0.17)):
        d = D[(D.cohort == coh) & (D.stage == st)]
        if d.empty:
            continue
        ax.scatter(d.k + dx + rng.uniform(-0.07, 0.07, len(d)), d[col], s=2.5, color=COH[coh], alpha=0.28, lw=0, zorder=1)
        g = d.groupby("k")[col]
        mu, se = g.mean().reindex(k), g.sem().reindex(k)
        ax.errorbar(k + dx, mu, se, color=COH[coh], lw=1.1, marker="o", ms=2.8, capsize=0, zorder=3, mfc="white",
                    label=f"{coh.replace('-', '−')} ({d.session_id.nunique()})")
    ax.axhline(0, color="0.5", lw=0.5, ls=":")
    ax.set_xlim(0.4, N + 0.6); ax.set_xticks(k if N <= 10 else k[::2])
    ax.set_xlabel("Whisker-hit trial")
    ax.legend(frameon=False, fontsize=4.6, loc="upper left")


def cohort_dots(ax, S, st, col, rng, label):
    """per-session change over the N trials, R+ vs R- within one stage; p: mouse-level cohort permutation (MWU, Welch)"""
    d = S[S.stage == st].dropna(subset=[col])
    for i, coh in enumerate(("R+", "R-")):
        v = d[d.cohort == coh][col]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=4, color=COH[coh], alpha=0.45, lw=0)
        ax.errorbar(i + 0.3, v.mean(), v.sem(), fmt="o", ms=3.8, color=COH[coh], lw=1.1, capsize=0, mfc="white")
    x, y = d[d.cohort == "R+"][col], d[d.cohort == "R-"][col]
    _, pp = m001.perm_interaction(d.assign(stage="learning"), col, kind="learning")
    pm, pw = stats.mannwhitneyu(x, y).pvalue, stats.ttest_ind(x, y, equal_var=False).pvalue
    top = np.nanmax(d[col]); rng_ = top - np.nanmin(d[col])
    ax.plot([0.1, 1.1], [top + 0.08 * rng_] * 2, color="0.3", lw=0.6)
    ax.text(0.6, top + 0.1 * rng_, f"perm {m62.fmt_p(pp)[4:] if m62.fmt_p(pp).startswith('p = ') else m62.fmt_p(pp)}",
            ha="center", va="bottom", fontsize=4.5)
    ax.axhline(0, color="0.6", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0.15, 1.15], ["R+", "R−"]); ax.set_xlim(-0.4, 1.6)
    for t, c in zip(ax.get_xticklabels(), ("R+", "R-")):
        t.set_color(COH[c])
    ax.set_ylabel(label, fontsize=5)
    m62.STATS.append(dict(panel="022 cohort dots", measure=col, stage=st, n_Rplus=len(x), n_Rminus=len(y),
                          mean_Rplus=x.mean(), mean_Rminus=y.mean(), p_perm=pp, p_MWU=pm, p_Welch=pw))


def figure(plt, D, S, Sm, N, out, pop):
    rng = np.random.default_rng(0)
    nrow = 2 if Sm is None else 3
    fig = plt.figure(figsize=(m62.W_IN, 4.3 if Sm is None else 6.1))
    gs = fig.add_gridspec(nrow, 4, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.87 if nrow == 2 else 0.9, bottom=0.08)
    panels = [(c, st, sl) for c in ("WH_SL", "AH_WH") for st, sl in STG]
    axs1 = []
    for j, (col, st, sl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j]); axs1.append(ax); ax.set_box_aspect(1)
        trials_panel(ax, D, st, col, N, rng)
        ax.set_ylabel("WH − SL (score)" if col == "WH_SL" else "AH − WH (distance to AH)", fontsize=5)
        ax.set_title(f"{sl}: " + ("WH relative to SL" if col == "WH_SL" else "WH distance to AH"), fontsize=5.6)
    for a_, b_, col in ((0, 1, "WH_SL"), (2, 3, "AH_WH")):        # same y range per reference; robust to single outlying trials
        lo, hi = np.nanpercentile(D[col], [1, 99])
        for ax in (axs1[a_], axs1[b_]):
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.25 * (hi - lo))
    rows_ax = [axs1]
    sets = [(S, f"sessions with ≥ {N} WH")] + ([] if Sm is None else [(Sm, "sessions with ≥ 20 WH")])
    for i, (SS, tag) in enumerate(sets):
        axs = []
        for j, (col, st, sl) in enumerate(panels):
            ax = fig.add_subplot(gs[1 + i, j]); axs.append(ax); ax.set_box_aspect(1)
            m = "slope_WH_SL" if col == "WH_SL" else "slope_WH_AH"
            cohort_dots(ax, SS, st, m, rng, (f"Change over {N} WH\n({tag})" if j == 0 else ""))
            ax.set_title(f"{sl}: " + ("WH − SL" if col == "WH_SL" else "WH − AH") + f" over {N} WH", fontsize=5.4)
        rows_ax.append(axs)
    for r_, letters in zip(rows_ax, ("abcd", "efgh", "ijkl")):
        m62.letter_row(fig, r_, letters)
    fig.suptitle(f"First {N} whisker hits, trial by trial: R+ vs R− within each stage (time-matched SL / AH references; {pop})",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=6.8, weight="bold")
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
