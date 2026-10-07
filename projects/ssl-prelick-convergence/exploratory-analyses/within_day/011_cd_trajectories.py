"""011 -- Reward-lick coding direction (CD) within sessions: learning day and expert sessions, R+ and R- (user review
2026-10-07: one figure per analysis, expert equivalents of the day-0 timeline panels; "may be my main result").

Data: 002 trial_scores.parquet -- every analysed event (WH, AH, SL) of every session with its 5-fold CD score
c_i (held-out for AH / SL, fold-averaged for WH; SL = 0, AH = 1 per session; sessions with held-out d' >= 0.3 and >= 4
WH, >= 8 AH and >= 8 SL) and its normalised time in the session tau_i (0 = first, 1 = last analysed event).
Per session and class k: OLS c_i = alpha_k + beta_k tau_i (beta_k = change of the class's mean CD score over the whole
session); drift of whisker hits relative to spontaneous licks beta_WH - beta_SL (relative to auditory hits:
beta_WH - beta_AH). Binned trajectories: 5 equal tau bins, per-session class means, mean +- s.e.m. over sessions.
Statistics (unit = session): beta vs 0 per group (Wilcoxon, one-sample t); learning day vs expert per cohort and R+ vs R-
per stage (Mann-Whitney U, Welch; mouse-level cohort permutation for R+ vs R-); learning x cohort interaction of the drift
(mouse-level permutation, 001 perm_interaction). Mixed-model estimates (009) shown for both stages and references.
Figure: within_day/<ref>/cd_trajectories/<pop>/cd_trajectories.{png,pdf,svg}; tables: sessions.csv, tests.csv.
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
sys.path.insert(0, str(CONV)); sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
OUT = m51.WITHIN / "cd_trajectories"
NB = 5
CLS = ["WH", "AH", "FA"]
STG = [("learning", "Day 0"), ("expert", "Expert")]


def session_table(T):
    rows, bins = [], []
    for sid, g in T.groupby("session_id"):
        g = g.dropna(subset=["cd"])
        if g.empty:                                     # axis failed the d' criterion: no scores
            continue
        r = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0])
        for k in CLS:
            q = g[g.cls == k]
            r[f"n_{k}"] = len(q)
            if len(q) >= 3 and q.tau.std() > 0:
                r[f"beta_{k}"] = np.polyfit(q.tau, q.cd, 1)[0]
                r[f"mean_{k}"] = q.cd.mean()
        if "beta_WH" in r and "beta_FA" in r:
            r["drift_WH_SL"] = r["beta_WH"] - r["beta_FA"]
        if "beta_WH" in r and "beta_AH" in r:
            r["drift_WH_AH"] = r["beta_WH"] - r["beta_AH"]
        rows.append(r)
        b = np.minimum((g.tau * NB).astype(int), NB - 1)
        for (k, bi), q in g.groupby([g.cls, b]):
            bins.append(dict(session_id=sid, cohort=r["cohort"], stage=r["stage"], cls=k, bin=bi, score=q.cd.mean()))
    S = pd.DataFrame(rows)
    B = pd.DataFrame(bins)
    D = B.pivot_table(index=["session_id", "cohort", "stage", "bin"], columns="cls", values="score").reset_index()
    D["WH_SL"] = D.WH - D.FA
    return S, B, D


def tests(S):
    rows = []
    for col in ["drift_WH_SL", "drift_WH_AH", "beta_WH", "beta_AH", "beta_FA"]:
        r = dict(measure=col)
        for c in ("R+", "R-"):
            for st, _ in STG:
                v = S[(S.cohort == c) & (S.stage == st)][col].dropna()
                g = f"{c} {st}"
                r[f"n {g}"], r[f"mean {g}"], r[f"sem {g}"] = len(v), v.mean(), v.sem()
                if len(v) >= 3:
                    r[f"p_wilcoxon {g}"] = stats.wilcoxon(v).pvalue if (v != 0).any() else np.nan
                    r[f"p_t {g}"] = stats.ttest_1samp(v, 0).pvalue
            a = S[(S.cohort == c) & (S.stage == "learning")][col].dropna(); e = S[(S.cohort == c) & (S.stage == "expert")][col].dropna()
            if len(a) >= 3 and len(e) >= 3:
                r[f"{c} day0 vs expert p_MWU"] = stats.mannwhitneyu(a, e).pvalue
                r[f"{c} day0 vs expert p_Welch"] = stats.ttest_ind(a, e, equal_var=False).pvalue
        for st, _ in STG:
            a = S[(S.cohort == "R+") & (S.stage == st)][col].dropna(); b = S[(S.cohort == "R-") & (S.stage == st)][col].dropna()
            if len(a) >= 3 and len(b) >= 3:
                r[f"{st} R+ vs R- p_MWU"] = stats.mannwhitneyu(a, b).pvalue
                r[f"{st} R+ vs R- p_Welch"] = stats.ttest_ind(a, b, equal_var=False).pvalue
                d = S[S.stage == st].copy()
                r[f"{st} R+ - R-"], r[f"{st} p_perm"] = m001.perm_interaction(d.assign(stage="learning"), col, kind="learning")
        r["cohort x stage"], r["cxs p_perm"] = m001.perm_interaction(S, col, kind="stage")
        rows.append(r)
    return pd.DataFrame(rows)


def traj(ax, D, coh, st, cols, colors, labels, ls="-", mfc=None):
    d = D[(D.cohort == coh) & (D.stage == st)]
    x = (np.arange(NB) + 0.5) / NB
    for col, c, lab in zip(cols, colors, labels):
        g = d.groupby("bin")[col]
        mu, se = g.mean().reindex(range(NB)), g.sem().reindex(range(NB))
        ax.fill_between(x, mu - se, mu + se, color=c, alpha=0.15, lw=0)
        ax.plot(x, mu, ls=ls, color=c, marker="o", ms=2.6, lw=1.0, mfc=mfc or c, label=lab)
    return d.session_id.nunique()


def main(a):
    plt = m62.setup()
    out = OUT / a.population
    out.mkdir(parents=True, exist_ok=True)
    T = pd.read_parquet(m51.WITHIN / "slopes" / "trial_scores.parquet")
    if a.population == "learners":
        T = m61.learner_filter(T)
    S, B, D = session_table(T)
    S.to_csv(out / "sessions.csv", index=False)
    TT = tests(S)
    TT.to_csv(out / "tests.csv", index=False)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(m62.W_IN, 7.6))
    gs = fig.add_gridspec(3, 4, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.93, bottom=0.06)
    # row 1: class trajectories per cohort x stage
    axs1 = []
    for j, (coh, (st, sl)) in enumerate([(c, s) for c in ("R+", "R-") for s in STG]):
        ax = fig.add_subplot(gs[0, j]); axs1.append(ax)
        n = traj(ax, D, coh, st, ["WH", "AH", "FA"], [m62.CL["WH"], m62.CL["AH"], m62.CL["FA"]], ["WH", "AH", "SL"])
        ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(1, color=m62.CL["AH"], lw=0.5, ls=":")
        ax.set_title(f"{coh.replace('-', '−')} {sl.lower()} ({n} sessions)", color=m62.COH[coh], fontsize=5.8)
        ax.set_xlabel("Time in session τ"); ax.set_xlim(0, 1); ax.set_ylim(-0.4, 1.5)
        if j == 0:
            ax.set_ylabel("Reward-lick CD score\n(SL = 0, AH = 1)"); ax.legend(frameon=False, fontsize=4.6, loc="upper left")
    # row 2: WH - SL trajectories, day 0 vs expert; drift per session
    axs2 = []
    for j, coh in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[1, j]); axs2.append(ax)
        for st, sl in STG:
            n = traj(ax, D, coh, st, ["WH_SL"], [m62.COH[coh]], [f"{sl}"], ls="--" if st == "learning" else "-",
                     mfc="white" if st == "learning" else None)
        ax.axhline(0, color="0.6", lw=0.5, ls=":")
        ax.set_xlabel("Time in session τ"); ax.set_ylabel("CD score, WH − SL"); ax.set_xlim(0, 1)
        ax.legend(frameon=False, fontsize=4.6, loc="upper left")
        ax.set_title(f"{coh.replace('-', '−')}: WH relative to SL", color=m62.COH[coh], fontsize=5.8)
    for j, (col, ttl) in enumerate([("drift_WH_SL", "Drift of WH relative to SL\nβ_WH − β_SL per session"),
                                    ("drift_WH_AH", "Drift of WH relative to AH\nβ_WH − β_AH per session")]):
        ax = fig.add_subplot(gs[1, 2 + j]); axs2.append(ax)
        m62.dots_panel(ax, S, col, "Change over one session" if j == 0 else "", rng, f"W-cd {col}", ttl, ref=[(0, "0.6")])
        ax.set_title(ax.get_title(), fontsize=5.4)
    # row 3: per-class slopes (what moves), and the mixed model
    axs3 = []
    for j, (col, ttl) in enumerate([("beta_WH", "Slope of WH score β_WH"), ("beta_FA", "Slope of SL score β_SL"),
                                    ("beta_AH", "Slope of AH score β_AH")]):
        ax = fig.add_subplot(gs[2, j]); axs3.append(ax)
        m62.dots_panel(ax, S, col, "Change over one session" if j == 0 else "", rng, f"W-cd {col}", ttl, ref=[(0, "0.6")])
        ax.set_title(ax.get_title(), fontsize=5.4)
    ax = fig.add_subplot(gs[2, 3]); axs3.append(ax)
    MM = pd.read_csv(m51.WITHIN / "mixed_model" / "mixed_model_terms.csv")
    for k_, (ref_, mk) in enumerate((("spontaneous licks", "o"), ("auditory hits", "s"))):
        for i, (st, sl) in enumerate(STG):
            r = MM[(MM.stage == st) & (MM.reference == ref_)].iloc[0]
            for coh, key, dx in (("R+", "Rplus", -0.08), ("R-", "Rminus", 0.08)):
                x = i * 2 + k_ + dx
                ax.errorbar(x, r[f"drift_{key}"], 1.96 * r[f"se_{key}"], fmt=mk, ms=3, color=m62.COH[coh],
                            mfc="white" if st == "learning" else m62.COH[coh], lw=0.8)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_xticks([0, 1, 2, 3], ["D0\nvs SL", "D0\nvs AH", "Exp.\nvs SL", "Exp.\nvs AH"], fontsize=4.8)
    ax.set_ylabel("WH drift (mixed model, ± 95 % CI)"); ax.set_title("Mixed model (all mice)", fontsize=5.6)
    m62.letter_row(fig, axs1, "abcd"); m62.letter_row(fig, axs2, "efgh"); m62.letter_row(fig, axs3, "ijkl")
    fig.suptitle(f"Reward-lick coding direction within sessions: learning day and expert sessions ({a.population})",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, "cd_trajectories"); plt.close(fig)
    pd.DataFrame(m62.STATS).to_csv(out / "stats_dots.csv", index=False)
    print(TT.round(4).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
