"""Shared paths and statistics of ssl-within-day-two-axes.
Inputs come from ssl-prelick-convergence (reward-lick axis, 051 / 105 events, DLC loader 113, statistics 062) and ssl-hitmiss-axis-passive
(hit - miss axis, 135 / 175). Outputs: combined_results_ks4/ssl-within-day-two-axes/{tables,figures}.
Run everything on haas with the ssl-prelick-convergence environment:
  cd ~/code/unit_spikes_analysis && PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities ./.venv/bin/python ...
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parents[1]
PL = PROJECTS / "ssl-prelick-convergence" / "exploratory-analyses"
HM = PROJECTS / "ssl-hitmiss-axis-passive" / "exploratory-analyses"
sys.path[:0] = [str(PL), str(PL / "within_day"), str(PROJECTS.parent / "scripts")]
from axel_bisi_paths import axel_bisi_root  # noqa: E402

m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
ROOT = axel_bisi_root() / "combined_results_ks4" / "ssl-within-day-two-axes"
TAB, FIG = ROOT / "tables", ROOT / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
GLAB = {("R+", "learning"): "R+ day 0", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− day 0", ("R-", "expert"): "R− expert"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
ALPHA = 0.05


def population(D):
    """learners rule on day 0 (definition under review), all expert sessions"""
    return m61.learner_filter(D)


def interaction_perm(D, col, n_perm=10000, seed=0):
    """[E - D0](R+) - [E - D0](R-), cohort labels permuted across mice (a mouse keeps its sessions); two-sided"""
    D = D.dropna(subset=[col])
    if D.groupby(["cohort", "stage"]).size().reindex(GROUPS).fillna(0).min() < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    mice = D.drop_duplicates("mouse_id").set_index("mouse_id").cohort
    v, st, mid = D[col].to_numpy(float), D.stage.to_numpy(), D.mouse_id.to_numpy()

    def stat(lab):
        c = pd.Series(mid).map(lab).to_numpy()
        m = lambda cc, s: v[(c == cc) & (st == s)].mean()
        return (m("R+", "expert") - m("R+", "learning")) - (m("R-", "expert") - m("R-", "learning"))
    obs = stat(mice)
    null = np.array([stat(pd.Series(rng.permutation(mice.to_numpy()), index=mice.index)) for _ in range(n_perm)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)


def classify(p0, pe, ps, m0, me):
    """within-session change of one cohort: day 0 (p0, m0), expert (pe, me), day 0 vs expert (ps)"""
    if not np.isfinite(p0):
        return "n/a"
    s0, se, sd = p0 < ALPHA, (np.isfinite(pe) and pe < ALPHA), (np.isfinite(ps) and ps < ALPHA)
    if s0 and not se:
        return "learning-specific" if sd else "day 0 only (stage n.s.)"
    if s0 and se:
        if np.sign(m0) == np.sign(me) and abs(m0) > abs(me) and sd:
            return "stronger with learning"
        return "both stages" if not sd else "both stages, differ"
    if se and not s0:
        return "expert only"
    return "none"


def measure_stats(D, col, level, measure, n_perm=10000):
    """one row per cohort: mean / s.e.m. / n / Wilcoxon vs 0 per stage, day 0 vs expert (MWU, Welch), class; and a cohort row
    (R+ vs R- per stage, cohort x stage permutation)"""
    rows = []
    g = {k: D[(D.cohort == k[0]) & (D.stage == k[1])][col].dropna().to_numpy(float) for k in GROUPS}
    w0 = lambda x: stats.wilcoxon(x).pvalue if len(x) >= 5 else (stats.ttest_1samp(x, 0).pvalue if len(x) >= 3 else np.nan)
    two = lambda a, b: (stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue) if len(a) >= 3 and len(b) >= 3 else (np.nan, np.nan)
    for c in ("R+", "R-"):
        a, b = g[(c, "learning")], g[(c, "expert")]
        p0, pe = w0(a) if len(a) else np.nan, w0(b) if len(b) else np.nan
        ps, psw = two(a, b)
        m0, me = (a.mean() if len(a) else np.nan), (b.mean() if len(b) else np.nan)
        rows.append(dict(level=level, measure=measure, cohort=c, n_d0=len(a), n_e=len(b), mean_d0=m0, sem_d0=stats.sem(a) if len(a) > 1 else np.nan,
                         mean_e=me, sem_e=stats.sem(b) if len(b) > 1 else np.nan, p_d0=p0, p_e=pe, p_stage_mwu=ps, p_stage_welch=psw,
                         cls=classify(p0, pe, ps, m0, me)))
    c0, c0w = two(g[("R+", "learning")], g[("R-", "learning")]); ce, cew = two(g[("R+", "expert")], g[("R-", "expert")])
    obs, pint = interaction_perm(D, col, n_perm)
    rows.append(dict(level=level, measure=measure, cohort="R+ vs R-", p_d0=c0, p_d0_welch=c0w, p_e=ce, p_e_welch=cew,
                     interaction=obs, p_interaction=pint,
                     cls=classify(c0, ce, pint, np.nanmean(g[("R+", "learning")]) - np.nanmean(g[("R-", "learning")]),
                                  np.nanmean(g[("R+", "expert")]) - np.nanmean(g[("R-", "expert")]))))
    return rows


def fp(p):
    return "n/a" if not np.isfinite(p) else ("p<.001" if p < 0.001 else f"p={p:.2f}".replace("0.", "."))


def dot_panel(ax, D, col, R, title, letter=None, rc=None):
    """day 0 (open) vs expert (filled) per cohort, sessions as dots, mean +- s.e.m.; per cohort class and p values stacked above the
    axes (left R+, right R-); rc: the cohort row of measure_stats for an x-label with the cohort tests"""
    for j, c in enumerate(("R+", "R-")):
        for i, s_ in enumerate(("learning", "expert")):
            v = D[(D.cohort == c) & (D.stage == s_)][col].dropna(); x = j * 2.4 + i
            ax.scatter(x + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=3, color=COH[c], alpha=0.35, lw=0)
            ax.errorbar(x, v.mean(), v.sem(), fmt="o", ms=3.5, color=COH[c], mfc="white" if s_ == "learning" else COH[c], capsize=0)
            ax.text(x, 0.01, f"{len(v)}", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4, color="0.45")
        r = R[R.cohort == c].iloc[0]
        ax.text(0.0 if j == 0 else 1.0, 1.03, f"{c.replace('-', '−')}: {r.cls}\nD0 {fp(r.p_d0)}, E {fp(r.p_e)}\nD0 vs E {fp(r.p_stage_mwu)}",
                transform=ax.transAxes, ha="left" if j == 0 else "right", va="bottom", fontsize=4.0, color=COH[c], linespacing=1.15)
    ax.axhline(0, color="0.6", lw=0.5, ls=":"); ax.set_xticks([0, 1, 2.4, 3.4], ["D0", "E", "D0", "E"])
    ax.set_title(title, fontsize=5.6, pad=24)
    if rc is not None:
        ax.set_xlabel(f"R+ vs R−: D0 {fp(rc.p_d0)}, E {fp(rc.p_e)}\ncohort × stage {fp(rc.p_interaction)}", fontsize=4.4)
    if letter:
        ax.text(-0.32, 1.4, letter, transform=ax.transAxes, fontsize=8, weight="bold")


def bh(p):
    p = np.asarray(p, float); q = np.full_like(p, np.nan); ok = np.isfinite(p)
    if ok.sum():
        x = p[ok]; o = np.argsort(x); r = x[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; y = np.empty_like(x); y[o] = np.minimum(r, 1); q[ok] = y
    return q
