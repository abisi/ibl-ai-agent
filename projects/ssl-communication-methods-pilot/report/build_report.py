"""Report of ssl-communication-methods-pilot (skills/project-report, report_lib), first version 2026-11-10.
Inter-area communication on the first whisker day, R+ vs R- learners: five communication measures computed on the same draws (RRR,
CCA, partial CCA, partial RRR, unique ΔR²; 016 + 013), Chen et al. 2024-style coupling of task-axis projections (014 + 015), seven
area groups (main) and all area groups (supplement). Main figure: 017_publication_figure.py.
Run (haas): cd ~/code/ibl-ai-agent/projects/ssl-communication-methods-pilot && \
  PYTHONPATH=$HOME/code/NWB_reader:$HOME/code/unit_spikes_analysis:$HOME/code/ephys_utilities ~/code/unit_spikes_analysis/.venv/bin/python report/build_report.py
Render locally: bash projects/ssl-communication-methods-pilot/report/render.sh
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(PROJ)]
from report_lib import Report  # noqa: E402
import comm_lib as cl  # noqa: E402

RES = cl.OUT
OUT = RES / "report"
FIG = RES / "report_figures"
R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
_num = num
num = lambda *a, **k: _num(*a, **k).replace("-", "−")
R.register([
    ("eq-rrr", "Eq. (1)"), ("eq-ceil", "Eq. (2)"), ("eq-prrr", "Eq. (3)"), ("eq-dr2", "Eq. (4)"), ("eq-cd", "Eq. (5)"),
    ("eq-rise", "Eq. (6)"), ("tbl-n", "Table S1"), ("fig-main", "Figure 1"), ("tbl-methods", "Table 1"),
    ("fig-all", "Figure S1"), ("fig-prep", "Figure S2"), ("tbl-registry", "Table S2"),
])
RUNS = {"cs": "comm_v3_seven", "ca": "comm_v3_all", "ds": "cd_coupling_seven", "da": "cd_coupling"}
SRC = {f"{k}_{f}": RES / run / "learners" / f"{f}.csv" for k, run in RUNS.items()
       for f in (("stats_pooled", "method_counts", "method_correlations") if k[0] == "c" else ("stats_pooled", "stats_clusters", "dprime", "compare_rrr"))}
SRC.update({f"{k}_clusters": RES / RUNS[k] / "learners" / "stats_clusters.csv" for k in ("cs", "ca")})
SRC.update({f"{k}_pairs": RES / RUNS[k] / "learners" / "stats_pairs.csv" for k in ("cs",)})
SRC.update({f"{k}_allmice": RES / RUNS[k] / "all_mice" / "stats_pooled.csv" for k in ("cs", "ds")})
SRC.update({f"prep_{a}": FIG / f"preprocessing_{a}_learners.csv" for a in ("seven", "all")})  # 018 (2026-11-10)
FROZEN = dict(zip(SRC, Report.snapshot(list(SRC.values()), OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}
for _a in ("seven", "all"):  # one pooled-statistics table per preprocessing: T["prep_seven_z"], ...
    for _prep, _d in T.pop(f"prep_{_a}").groupby("prep"):
        T[f"prep_{_a}_{_prep}"] = _d
NAME = {"r2": "RRR", "cca": "CCA", "pcca": "pCCA", "prrr": "pRRR", "dr2": "ΔR²"}
SEVEN = ["Somatosensory cortex", "Auditory areas", "Midbrain", "Thalamus", "Hippocampus", "Motor and frontal areas", "Striatum and pallidum"]


EQ_RRR = r'R^2_{X\to Y} = 1 - \frac{\sum_{\text{held-out}} \lVert Y - \hat Y(X) \rVert^2}{\sum_{\text{held-out}} \lVert Y - \bar Y \rVert^2}'
EQ_CEIL = r'\tilde R^2_{X\to Y} = \frac{R^2_{X\to Y} - \langle R^2_{\text{shuffle}} \rangle}{\langle R^2_{Y^{\prime}\to Y} \rangle_{\text{session}}}'
EQ_PRRR = r'R^2_{X\to Y \mid Z} = R^2_{\mathrm{RRR}}\!\left(X - Z\hat B_X,\; Y - Z\hat B_Y\right)'
EQ_DR2 = r'\Delta R^2 = R^2(Y \mid [X, Z]) - R^2(Y \mid Z)'
EQ_CD = r'w = \frac{\bar r_A - \bar r_B}{\lVert \bar r_A - \bar r_B \rVert}'
EQ_RISE = r'\Delta_{\text{early}} = \bar m_{\text{early}} - \bar m_{\text{pre}}, \qquad \Delta_{\text{late}} = \bar m_{\text{late}} - \bar m_{\text{pre}}'


def P(k, q, cond, var, src="all", tgt="all"):
    d = T[k]
    d = d[(d.quantity == q) & (d.condition == cond) & (d.variable == var) & (d.source == src) & (d.target == tgt)]
    assert len(d) == 1, (k, q, cond, var, len(d))
    return d.iloc[0]


def cmp(key, k, q, cond, var, fmt="{:.3f}"):
    """R+ vs R- pooled: means, Mann-Whitney + Welch"""
    r = P(k, q, cond, var)
    return (f"R+ {num(key + '_p', r.mean_rplus, fmt)} vs R− {num(key + '_m', r.mean_rminus, fmt)} "
            f"(n = {num(key + '_np', int(r.n_rplus))} / {num(key + '_nm', int(r.n_rminus))} sessions; Mann-Whitney p {pv(key + '_mw', r.mannwhitney_p)}, "
            f"Welch p {pv(key + '_w', r.welch_p)})")


def within(key, k, q, cond, var, fmt="{:.4f}"):
    """per-cohort mean and Wilcoxon against 0"""
    r = P(k, q, cond, var)
    return (f"R+ {num(key + '_p', r.mean_rplus, fmt)} (Wilcoxon p {pv(key + '_pw', r.rplus_wilcoxon_p)}), "
            f"R− {num(key + '_m', r.mean_rminus, fmt)} (p {pv(key + '_mw', r.rminus_wilcoxon_p)})")


def counts(k, q, cond="W_lick", var="rise_early"):
    d = T[k]
    return d[(d.quantity == q) & (d.condition == cond) & (d.variable == var)].iloc[0]


def corr(k, a, b, level="early rise per session x pair"):
    d = T[k]
    return d[(d.comparison == f"{a} vs {b}") & (d.level == level)].iloc[0].spearman


def clusters(k, q=None):
    d = T[k]
    d = d[(d.source == "all") & (d.target == "all") & (d.p < 0.05)]
    return d if q is None else d[d.quantity == q]


def sample_table():
    rows = []
    for run, lab in ((RUNS["cs"], "seven"), (RUNS["ca"], "all")):
        M = [json.loads(f.read_text()) for f in sorted((RES / run / "sessions").glob("*_meta.json"))]
        M = [m for m in M if m.get("learner")]
        for m in M:
            for g in m.get("groups", []):
                rows.append(dict(set=lab, area=g, cohort=m["cohort"], session=m["session_id"], units=m["n_units"].get(g, np.nan)))
    D = pd.DataFrame(rows)
    D = D[D.set == "seven"]
    out = []
    for g in SEVEN:
        x = D[D.area == g]
        out.append({"Area group": g, **{f"sessions {c}": int((x.cohort == c).sum()) for c in ("R+", "R-")},
                    "median units": int(np.nanmedian(x.units)) if len(x) else 0})
    return pd.DataFrame(out)


def dprime(k, cd):
    d = T[k]
    s = d[d.cd == cd].groupby(["session_id", "cohort"]).cd_dprime.mean().reset_index()
    a, b = s[s.cohort == "R+"].cd_dprime, s[s.cohort == "R-"].cd_dprime
    return a.mean(), b.mean(), stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue, len(a), len(b)


def cd_vs_rrr(k):
    d = T[k]
    return {m: stats.spearmanr(d.cd_r, d[m], nan_policy="omit")[0] for m in ("r2", "cca", "pcca", "prrr", "dr2")}


def method_table(k):
    rows = []
    for q in ("r2", "cca", "pcca", "prrr", "dr2"):
        c, r = counts(k + "_method_counts", q), P(k + "_stats_pooled", q, "W_lick", "rise_early")
        rows.append({"Method": NAME[q], "Removes rest of brain": "yes" if q in ("pcca", "prrr", "dr2") else "no",
                     "Pairs with early rise, R+": f"{int(c.rplus_rise)} / {int(c.n_pairs)}",
                     "Pairs with early rise, R−": f"{int(c.rminus_rise)} / {int(c.n_pairs)}",
                     "Pooled early rise R+": f"{r.mean_rplus:.4f}", "Pooled early rise R−": f"{r.mean_rminus:.4f}",
                     "R+ vs R− (MW / Welch p)": f"{r.mannwhitney_p:.2f} / {r.welch_p:.2f}",
                     "ρ with RRR": "1" if q == "r2" else f"{corr(k + '_method_correlations', 'r2', q):.2f}"})
    return pd.DataFrame(rows)


def build():
    n_all = sum(1 for _ in (RES / RUNS["cs"] / "sessions").glob("*_meta.json"))
    r0 = P("cs_stats_pooled", "r2n", "W_lick", "rise_early")
    r1 = P("cs_stats_pooled", "r2n", "A_hit", "pre")
    dp_hm, dp_pl = dprime("ds_dprime", "hitmiss"), dprime("ds_dprime", "prelick")
    cdr = cd_vs_rrr("ds_compare_rrr")
    cmc = counts("cs_method_counts", "r2")
    pcl = clusters("cs_clusters")
    pr_cl = pcl[(pcl.quantity == "prrrn") & (pcl.condition == "W_nolick")].iloc[0]
    ca_cl = pcl[(pcl.quantity == "ccan") & (pcl.condition == "A_hit")].iloc[0]
    dcl = clusters("ds_stats_clusters")
    pl_cl = dcl[(dcl.quantity == "r") & (dcl.condition == "prelick:ahsl_pooled")].iloc[0]
    g_cl = dcl[(dcl.quantity == "r_vs_gain") & (dcl.condition == "prelick:ahsl_pooled")].iloc[0]
    k_pr, k_pa = counts("cs_method_counts", "prrr"), counts("cs_method_counts", "pcca")
    k_dr = counts("cs_method_counts", "dr2")
    SH = {"Somatosensory cortex": "SSp", "Auditory areas": "AUD", "Midbrain": "MB", "Thalamus": "TH", "Hippocampus": "HPC",
          "Motor and frontal areas": "Frontal", "Striatum and pallidum": "Str"}
    QP = T["cs_pairs"]
    QP = QP[QP.quantity.isin(["r2", "r2n", "cca", "ccan", "pcca", "prrr", "dr2"]) & QP.variable.isin(["rise_early", "rise_late"])]
    n_tests = len(QP)
    sig = QP[QP.mw_q < 0.05]
    q_pairs = len(sig)
    q_list = "; ".join(f"{NAME.get(r.quantity.rstrip('n'), r.quantity)}{' / ceiling' if r.quantity.endswith('n') and r.quantity != 'cca' else ''}, "
                       f"{r.condition}, {SH[r.source]} → {SH[r.target]}, {r.variable.replace('rise_', '')} rise "
                       f"(R+ {r.mean_rplus:.3f} vs R− {r.mean_rminus:.3f}, q = {r.mw_q:.3f})" for r in sig.itertuples()).replace("-0", "−0")
    mf = QP[(QP.quantity == "r2n") & (QP.condition == "WL-WN") & (QP.variable == "rise_late") & (QP.source == "Midbrain")
            & (QP.target == "Motor and frontal areas")].iloc[0]

    md = rf"""---
title: "Communication between brain areas on the first whisker day: R+ vs R− learners, five methods, seven area groups"
author: "Axel Bisi (analysis with Claude Code)"
date: "{date.today().isoformat()}"
---

**Data.** Kilosort 4, stable good + MUA units. First whisker day, one session per mouse: {num('n_sessions_run', n_all)} sessions analysed. Main population: learners, {num('n_r1p', int(r1.n_rplus))} R+ and {num('n_r1m', int(r1.n_rminus))} R− sessions with at least two of the seven area groups. Expert days are not included.

# Summary

We asked whether trial-to-trial communication between brain areas rises after the stimulus in the same way in R+ and R− mice, and how much of it remains when the rest of the recorded brain is removed. On the first whisker day, the shared variability between area groups (ridge reduced-rank regression, RRR, normalised by a within-area ceiling) rises after whisker hits and auditory hits in both cohorts and does not rise after whisker misses ({ref('fig-main')}a–c). The size of this rise does not differ between cohorts.

The rise is mostly shared with the rest of the brain. After removing the top five principal components of all other recorded units, by partial RRR, partial CCA or the unique ΔR² of the source, no area pair keeps a significant rise ({ref('fig-main')}d–h, {ref('tbl-methods')}). The five methods agree only moderately with each other ({ref('fig-main')}g), and pair-level cohort differences are rare: the only one found by several methods is a larger late rise between midbrain and motor/frontal areas on whisker hits relative to misses in R+ mice ({ref('fig-main')}i).

On task axes (Chen et al. 2024), the coupling of hit − miss axis projections across areas rises more in R+ than in R− mice, but this difference disappears when the rest of the brain is regressed out. The coupling of pre-lick axis projections is larger in R− mice relative to a population-gain axis ({ref('fig-main')}j–k). The same conclusions hold with all ten area groups ({ref('fig-all')}) and with all mice.

**Key results** (learners; seven area groups; session = unit of analysis; R+ vs R− tested with Mann-Whitney and Welch; every claim is listed in {ref('tbl-registry')}):

1. {claim('rise', 'Communication rises after whisker hits in both cohorts, with the same size.', 'Fig. 1a', 'r2n W_lick early rise', '33/32')} Early rise of RRR R² / ceiling: {cmp('rise', 'cs_stats_pooled', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}. Pairs with a significant rise (BH q < 0.05): R+ {num('cnt_p', int(cmc.rplus_rise))}, R− {num('cnt_m', int(cmc.rminus_rise))} of {num('cnt_n', int(cmc.n_pairs))}.
2. {claim('pre', 'Before the stimulus, whisker-hit communication is higher in R− sessions.', 'Fig. 1a', 'r2n W_lick pre', '33/32', 'fragile')} {cmp('pre', 'cs_stats_pooled', 'r2n', 'W_lick', 'pre', '{:.2f}')}. {claim('prep', 'The pre-stimulus difference disappears when each trial’s baseline is removed; the evoked rise is unchanged.', 'Fig. S2', 'r2n W_lick pre / rise_early, zbase', '33/32')} With each trial's baseline removed: pre-stimulus {cmp('preb', 'prep_seven_zbase', 'r2n', 'W_lick', 'pre', '{:.2f}')}.
3. {claim('partial', 'Removing the rest of the brain removes the pair-level rise for all three partial methods.', 'Fig. 1d-h', 'pairs with rise', '40 pairs')} Pairs with an early rise on whisker hits: partial RRR {num('kpr_p', int(k_pr.rplus_rise))} / {num('kpr_m', int(k_pr.rminus_rise))}, partial CCA {num('kpa_p', int(k_pa.rplus_rise))} / {num('kpa_m', int(k_pa.rminus_rise))}, ΔR² {num('kdr_p', int(k_dr.rplus_rise))} / {num('kdr_m', int(k_dr.rminus_rise))} (R+ / R−).
4. {claim('agree', 'Methods that remove the rest of the brain agree only weakly with RRR.', 'Fig. 1g', 'Spearman of early rise', '3448 session x pair')} Spearman ρ of the early rise across sessions and pairs: RRR vs CCA {num('rho_cca', corr('cs_method_correlations', 'r2', 'cca'), '{:.2f}')}, vs partial RRR {num('rho_prrr', corr('cs_method_correlations', 'r2', 'prrr'), '{:.2f}')}, vs partial CCA {num('rho_pcca', corr('cs_method_correlations', 'r2', 'pcca'), '{:.2f}')}, vs ΔR² {num('rho_dr2', corr('cs_method_correlations', 'r2', 'dr2'), '{:.2f}')}.
5. {claim('hm', 'Hit − miss axis coupling rises more in R+ than R−, but not after regressing out the rest of the brain.', 'Fig. 1j', 'r hm_pooled early rise; r_partial', '33/32')} Raw: {cmp('hm', 'ds_stats_pooled', 'r', 'hitmiss:hm_pooled', 'rise_early', '{:.2f}')}. Partial: {cmp('hmp', 'ds_stats_pooled', 'r_partial', 'hitmiss:hm_pooled', 'rise_early', '{:.2f}')}.
6. {claim('pl', 'Pre-lick axis coupling, relative to a population-gain axis, is larger in R− mice.', 'Fig. 1k', 'r_vs_gain ahsl_pooled early', '33/34')} {cmp('pl', 'ds_stats_pooled', 'r_vs_gain', 'prelick:ahsl_pooled', 'rise_early', '{:.2f}')}; R+ vs R− cluster on raw r {num('plc_s', pl_cl.start, '{:+.2f}')} to {num('plc_e', pl_cl.end, '{:+.2f}')} s, p {pv('plc_p', pl_cl.p)}.

# Introduction

R+ mice are rewarded for licking after a whisker stimulus; R− mice are not. Both cohorts are rewarded for licking after an auditory stimulus. On the first whisker day, both cohorts lick after many whisker stimuli. The question is whether the trial-to-trial co-fluctuations between brain areas, a proxy for communication, differ between cohorts, and which part of them is specific to a pair of areas rather than shared with the rest of the brain.

Pairwise measures such as RRR and CCA cannot tell direct communication from a common input that reaches both areas. Movement, arousal and lick-related signals reach most of the brain on these trials. We therefore compare RRR and CCA with three methods that condition on the rest of the recording, all computed on the same units and trials. We also compare these population measures with the approach of Chen et al. (2024, Cell 187:676), which correlates single-trial projections onto task coding directions across areas.

The analysis has seven area groups, as requested: somatosensory cortex, auditory areas, midbrain, thalamus, hippocampus, motor and frontal areas, and striatum and pallidum. A supplementary analysis keeps all ten area groups of the v3 grouping.

# Methods

## Data

{R.methods('task', 'learners', 'trials', 'units')}

**Sessions.** Whisker day 0, one session per mouse, all sessions not excluded in the reference sheet ({num('n_sessions_run2', n_all)}). The learners population is the main one; all mice are a robustness check.

**Area groups.** Somatosensory cortex merges SSp-whisker and SSp-orofacial; the other six are v3 area groups. Units outside these seven groups (posterior parietal, retrosplenial, brainstem, unassigned) are not analysed as sources or targets, but they are part of the rest of the brain used by the partial methods. A group enters a session if it has at least 15 stable units with a firing rate ≥ 1 Hz. {ref('tbl-n')} lists sessions and units per group.

**Conditions.** Whisker hits (W_lick), whisker misses (W_nolick) and auditory hits (A_hit), with at least 5 trials per condition. The within-session contrasts WL − AH and WL − WN are computed per session.

## Communication measures

Spikes are counted in 50-ms bins from −0.6 to +1.0 s around the stimulus, and each unit's condition mean in each bin is subtracted, so only trial-to-trial fluctuations remain. A 300-ms window (6 bins) slides in 100-ms steps, centred from −0.4 to +0.8 s. In each window, 15 source and 15 target units are drawn at random, with up to 20 trials per condition. The draw is repeated 5 times and averaged.

**RRR.** The target activity $Y$ is predicted from the source activity $X$ with ridge reduced-rank regression (Wu & Pillow 2025). The rank is chosen by 10-fold cross-validation over trials with the one-s.e.m. rule. The held-out variance explained is
{eq('eq-rrr', EQ_RRR)}
where $\hat Y(X)$ is the rank-$k$ ridge prediction (ridge α in {{0.01, 0.1, 1, 10}}, $k \le 10$). Every measure is reported as an excess over a null in which the source trials are shuffled (10 shuffles): value minus the null mean.

**Within-area ceiling.** The same 15 target units are predicted from 15 other units of the target group (groups with ≥ 30 units). The normalised measure is
{eq('eq-ceil', EQ_CEIL)}
where $Y'$ are the other target-group units and the denominator is the session-mean ceiling (sessions with a ceiling below 0.01 are dropped). The same normalisation is applied to every method.

**CCA.** The first held-out canonical correlation after reducing source and target to 10 principal components each.

**Rest of the brain.** $Z$ is the top 5 principal components of all units outside the source and target groups, in the same window and trials.

**Partial CCA.** The first canonical correlation of $X$ and $Y$ conditioned on $Z$ (vendored `partial_CCA` 0.1.0, cross-validated).

**Partial RRR.** Source and target are first residualised on $Z$ by least squares, then RRR (Eq. 1) is fit on the residuals:
{eq('eq-prrr', EQ_PRRR)}
where $\hat B_X$ and $\hat B_Y$ are the least-squares coefficients of $X$ and $Y$ on $Z$, fit on the training folds.

**Unique ΔR².** The variance of the target explained by the source beyond the rest of the brain:
{eq('eq-dr2', EQ_DR2)}
with cross-validated ridge regressions and the ridge α chosen for the full model.

## Task-axis coupling (Chen et al. 2024)

For each area, a coding direction (CD) is the normalised difference of the mean population vectors of two trial types:
{eq('eq-cd', EQ_CD)}
computed leave-one-trial-out, so a trial's projection never uses that trial. Two axes are used. The **hit − miss axis** contrasts whisker hits and misses 5–35 ms after the stimulus (minus a −55 to −20 ms baseline). The **pre-lick axis** contrasts auditory hits (AH) and spontaneous licks (SL) in the 100 ms before the first lick (lick time corrected for the NWB offset). Single-trial projections are taken in 200-ms windows (50-ms steps), and for each pair of areas the Spearman correlation $r$ of projections across trials is computed, within one trial type or pooled over the two trial types used to define the axis (hits + misses; AH + SL). Areas need at least 10 units.

Three controls: $r$ minus a trial-permutation null (100 permutations); $r_{{\text{{partial}}}}$, after regressing the top 5 principal components of the rest of the brain out of both projections; and $r - r_{{\text{{gain}}}}$, the excess over the correlation along each area's population-mean axis. Axis strength is the held-out d′ along each area's CD. Values are Fisher-z transformed before statistics.

## Epochs and statistics

For each session and pair, the value is averaged in three epochs: pre (window centres −0.4 to −0.2 s), early (+0.2 to +0.3 s) and late (+0.5 to +0.7 s). For task-axis coupling the epochs are −0.4..−0.2 / +0.1..+0.2 / +0.4..+0.6 s after the stimulus (hit − miss) and −0.5..−0.35 / −0.1..0 / +0.15..+0.3 s around the first lick (pre-lick). The rise is
{eq('eq-rise', EQ_RISE)}
where $\bar m$ is the epoch mean of any of the measures (Eqs. 1–4).

{R.methods('statistics')}

Pooled tests average over all pairs of a session first. Pair-level tests keep pairs with at least 3 sessions per cohort, and are corrected with Benjamini-Hochberg across pairs. Rises within a cohort use a Wilcoxon signed-rank test against 0. Time courses are compared between cohorts with cluster permutation tests (1000 permutations of the cohort labels). Rises in {ref('eq-rise')} use the excess measures of {ref('eq-rrr')}–{ref('eq-dr2')}, normalised by {ref('eq-ceil')} where stated; the task-axis coupling uses the projections of {ref('eq-cd')}.

**Scripts.** 016_comm_methods_v3.py (all communication measures, `--areas seven|all`), 013_rrr_timeresolved_v2_stats.py (statistics), 014_cd_projection_coupling.py and 015_cd_coupling_stats.py (task-axis coupling), 017_publication_figure.py (figures). Code: projects/ssl-communication-methods-pilot; results: combined_results_ks4/ssl-communication-methods-pilot.

# Results

## Communication rises after hits in both cohorts, by the same amount

Across all pairs, the RRR R² normalised by the within-area ceiling rises after whisker hits and auditory hits, and stays flat after whisker misses ({ref('fig-main')}a–c). On whisker hits, the early rise is {cmp('rise2', 'cs_stats_pooled', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}. Each cohort's rise differs from zero: {within('risew', 'cs_stats_pooled', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}. On auditory hits the rise is also similar: {cmp('ah', 'cs_stats_pooled', 'r2n', 'A_hit', 'rise_early', '{:.2f}')}.

In absolute terms the shared variance is small: the excess R² rise on whisker hits is {within('abs', 'cs_stats_pooled', 'r2', 'W_lick', 'rise_early', '{:.3f}')}. The rise is larger on whisker hits than misses in both cohorts (WL − WN: {within('wlwn', 'cs_stats_pooled', 'r2n', 'WL-WN', 'rise_early', '{:.2f}')}), but not larger than on auditory hits (WL − AH: {within('wlah', 'cs_stats_pooled', 'r2n', 'WL-AH', 'rise_early', '{:.2f}')}). The extra communication on whisker hits therefore follows the lick, not the whisker stimulus.

Before the stimulus, R− sessions have higher whisker-hit communication than R+ sessions ({cmp('pre2', 'cs_stats_pooled', 'r2n', 'W_lick', 'pre', '{:.2f}')}). This difference is not seen on auditory hits ({cmp('preah', 'cs_stats_pooled', 'r2n', 'A_hit', 'pre', '{:.2f}')}) and is not corrected across conditions; we treat it as fragile.

## Most of the rise is shared with the rest of the brain

The three methods that remove the rest of the brain give rises about ten times smaller than RRR ({ref('fig-main')}d–f, {ref('tbl-methods')}). Pooled over pairs, the partial-RRR ({ref('eq-prrr')}) early rise on whisker hits is still above zero in both cohorts ({within('prrr', 'cs_stats_pooled', 'prrr', 'W_lick', 'rise_early', '{:.4f}')}), but no single pair has a significant rise after correction ({ref('fig-main')}h). The normalised partial-RRR time course is noisier because the ceiling is not partialled ({ref('fig-main')}d–f); its only cohort cluster is on whisker misses, R− > R+ at {num('prc_s', pr_cl.start, '{:+.1f}')} to {num('prc_e', pr_cl.end, '{:+.1f}')} s (p {pv('prc_p', pr_cl.p)}, {ref('fig-main')}f).

The cohorts do not differ on any partial method: early rise on whisker hits, partial RRR {cmp('prrr2', 'cs_stats_pooled', 'prrr', 'W_lick', 'rise_early', '{:.4f}')}; partial CCA {cmp('pcca', 'cs_stats_pooled', 'pcca', 'W_lick', 'rise_early', '{:.4f}')}; ΔR² {cmp('dr2', 'cs_stats_pooled', 'dr2', 'W_lick', 'rise_early', '{:.4f}')}.

## The five methods agree only moderately

RRR and CCA, which both ignore the rest of the brain, agree best ({ref('fig-main')}g: ρ = {num('rho_cca2', corr('cs_method_correlations', 'r2', 'cca'), '{:.2f}')}). The partial methods agree with each other and with RRR at ρ ≈ 0.2–0.4. CCA also gives fewer pairs with a rise in R− than in R+ ({ref('fig-main')}h). The normalised CCA has one pooled cohort cluster on auditory hits, R− > R+ from {num('cac_s', ca_cl.start, '{:+.1f}')} to {num('cac_e', ca_cl.end, '{:+.1f}')} s (p {pv('cac_p', ca_cl.p)}), not seen with RRR.

{table('tbl-methods', method_table('cs'), 'Whisker-hit early rise per method, learners, seven area groups. Pairs: ordered pairs with a rise ≠ 0 within the cohort (Wilcoxon, BH q < 0.05). Pooled: session mean over pairs of the excess measure (Eqs. 1, 3, 4; CCA and pCCA: first canonical correlation). ρ: Spearman correlation with RRR of the early rise across session × pair.')}

## Pair-level cohort differences are rare and method-specific

At the pair level, the R+ − R− difference of the normalised RRR rise has a few pairs with uncorrected p < 0.05 ({ref('fig-main')}i, asterisks), about the number expected by chance among 40 pairs. Across methods (RRR, CCA and their normalised versions, partial CCA, partial RRR, ΔR²), conditions and both rises, {num('q_pairs', q_pairs)} of {num('n_tests', n_tests)} pair-level cohort tests survive Benjamini-Hochberg correction (within each method, condition and epoch): {q_list}. {claim('mbfr', 'The only pair-level cohort difference found by several methods is midbrain ↔ motor and frontal areas, whisker hits minus misses, late rise, R+ > R−.', 'Table S2 / summary.md', 'CCA / ceiling and pCCA q < 0.05; RRR / ceiling uncorrected', '18-20 / 17', 'fragile')} It is significant for normalised CCA in both directions and for partial CCA, and RRR / ceiling points the same way without surviving correction (midbrain → frontal: R+ {num('mf_p', mf.mean_rplus, '{:.2f}')} vs R− {num('mf_m', mf.mean_rminus, '{:.2f}')}, n = {num('mf_np', int(mf.n_rplus))} / {num('mf_nm', int(mf.n_rminus))}, Mann-Whitney p {pv('mf_mw', mf.mannwhitney_p)}, q {pv('mf_q', mf.mw_q)}). The two ΔR² results are isolated. Given the number of tests, these are leads for a targeted test, not findings.

## Task-axis coupling separates the cohorts, but through shared signals

The hit − miss axis is stronger in R+ mice (held-out d′, session mean over areas: R+ {num('dphm_p', dp_hm[0], '{:.2f}')} vs R− {num('dphm_m', dp_hm[1], '{:.2f}')}, Mann-Whitney p {pv('dphm_mw', dp_hm[2])}, Welch p {pv('dphm_w', dp_hm[3])}; {ref('fig-main')}l). The coupling of hit − miss projections across areas rises after the stimulus more in R+ mice ({ref('fig-main')}j; {cmp('hm2', 'ds_stats_pooled', 'r', 'hitmiss:hm_pooled', 'rise_early', '{:.2f}')}). Once the rest of the brain is regressed out of both projections, the cohort difference disappears ({cmp('hmp2', 'ds_stats_pooled', 'r_partial', 'hitmiss:hm_pooled', 'rise_early', '{:.2f}')}; dashed lines in {ref('fig-main')}j).

The pre-lick axis has similar strength in both cohorts (d′: R+ {num('dppl_p', dp_pl[0], '{:.2f}')} vs R− {num('dppl_m', dp_pl[1], '{:.2f}')}, Mann-Whitney p {pv('dppl_mw', dp_pl[2])}). Its coupling across areas is higher in R− mice around the first lick ({ref('fig-main')}k; cluster {num('plc_s2', pl_cl.start, '{:+.2f}')} to {num('plc_e2', pl_cl.end, '{:+.2f}')} s, p {pv('plc_p2', pl_cl.p)}). The difference is clearest relative to the population-gain axis: {cmp('pl2', 'ds_stats_pooled', 'r_vs_gain', 'prelick:ahsl_pooled', 'rise_early', '{:.2f}')}; cluster {num('glc_s', g_cl.start, '{:+.2f}')} to {num('glc_e', g_cl.end, '{:+.2f}')} s, p {pv('glc_p', g_cl.p)}. After regressing out the rest of the brain it is no longer significant ({cmp('plp', 'ds_stats_pooled', 'r_partial', 'prelick:ahsl_pooled', 'rise_early', '{:.2f}')}).

Task-axis coupling and population communication measure different things: across session × pair, the hit − miss coupling correlates weakly with the RRR rise (Spearman ρ = {num('cd_r2', cdr['r2'], '{:.2f}')}) and less with the partial methods (partial RRR {num('cd_prrr', cdr['prrr'], '{:.2f}')}, ΔR² {num('cd_dr2', cdr['dr2'], '{:.2f}')}).

## The pre-stimulus cohort difference is a shared baseline state; the evoked rise does not depend on preprocessing

The main analysis uses condition-mean residuals of spike counts. We repeated it with each unit soft-normalised (residuals divided by the unit's SD plus a constant, so that every unit weighs about equally; "normalised") and, in addition, with each trial's own baseline (−1.1 to −0.6 s before the stimulus) subtracted per unit before normalising ("normalised, trial baseline removed"), which removes slow states shared across areas ({ref('fig-prep')}).

The early rise on whisker hits does not differ between cohorts with any preprocessing: RRR / ceiling, normalised {cmp('pz', 'prep_seven_z', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}; trial baseline removed {cmp('pzb', 'prep_seven_zbase', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}. Each cohort's rise stays above zero ({within('pzbw', 'prep_seven_zbase', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}), and the fraction of pairs with a rise is similar across preprocessings ({ref('fig-prep')}i).

The pre-stimulus difference, by contrast, depends on the baseline. It is present in counts and after normalisation ({cmp('ppz', 'prep_seven_z', 'r2n', 'W_lick', 'pre', '{:.2f}')}; the only cohort cluster of {ref('fig-prep')}b, before the stimulus) but disappears once each trial's baseline is removed ({cmp('ppzb', 'prep_seven_zbase', 'r2n', 'W_lick', 'pre', '{:.2f}')}; {ref('fig-prep')}c). The higher pre-stimulus communication of R− sessions is therefore a slow state shared across areas that varies from trial to trial, not a difference in how areas interact around the stimulus. With all ten area groups the pattern is the same (pre-stimulus: normalised {cmp('ppza', 'prep_all_z', 'r2n', 'W_lick', 'pre', '{:.2f}')}; trial baseline removed {cmp('ppzba', 'prep_all_zbase', 'r2n', 'W_lick', 'pre', '{:.2f}')}).

Partial methods remain small and do not separate the cohorts consistently: partial RRR / ceiling early rise on whisker hits, normalised {cmp('pprz', 'prep_seven_z', 'prrrn', 'W_lick', 'rise_early', '{:.2f}')}; trial baseline removed {cmp('pprzb', 'prep_seven_zbase', 'prrrn', 'W_lick', 'rise_early', '{:.2f}')}. The R− mean is driven by a few sessions (wide band in {ref('fig-prep')}e–f). Isolated uncorrected p < 0.05 for partial CCA appear with one preprocessing but not the others ({ref('fig-prep')}g–h); we do not interpret them.

## All area groups and all mice

With all ten area groups ({ref('fig-all')}), the conclusions are the same: whisker-hit early rise of RRR / ceiling {cmp('all', 'ca_stats_pooled', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}; partial RRR {cmp('allp', 'ca_stats_pooled', 'prrr', 'W_lick', 'rise_early', '{:.4f}')}; hit − miss coupling {cmp('allhm', 'da_stats_pooled', 'r', 'hitmiss:hm_pooled', 'rise_early', '{:.2f}')}. Including non-learners (seven groups, all mice): RRR / ceiling {cmp('am', 'cs_allmice', 'r2n', 'W_lick', 'rise_early', '{:.2f}')}; hit − miss coupling {cmp('amhm', 'ds_allmice', 'r', 'hitmiss:hm_pooled', 'rise_early', '{:.2f}')}.

# Discussion

On the first whisker day, the post-stimulus increase of shared variability between area groups follows the lick: it is present on whisker and auditory hits, absent on misses, and the same in both cohorts. Most of it is shared with the rest of the recorded brain, as expected from lick- and movement-related signals that reach most areas. What remains after removing the rest of the brain is about a tenth of the RRR signal, is not resolved at the level of single pairs with these sample sizes, and does not differ between cohorts.

The task-axis approach gives cohort differences that the population measures do not: a larger hit − miss coupling rise in R+ mice and larger pre-lick coupling in R− mice. Both disappear when the rest of the brain is regressed out, so they reflect brain-wide signals expressed along each area's task axis, consistent with the earlier finding that the hit − miss axis differs between cohorts by rotation rather than gain. They are not evidence of pair-specific communication.

# Caveats and limitations

- The partial methods remove only what the top 5 principal components of the other recorded units capture; unrecorded common inputs are not removed, and genuine communication relayed through recorded areas is partly removed.
- Units and trials are subsampled (15 units, ≤ 20 trials, 5 draws); with 5–20 sessions per pair and cohort, small pair-level effects cannot be detected.
- Whisker hits and auditory hits include licking from ~200 ms; the early epoch overlaps jaw movement. Movement was not regressed out in this run.
- The ceiling of Eq. 2 is not partialled, so the normalised partial measures are smaller than a partialled ceiling would give.
- Pair-level results are not corrected across methods and conditions.

# Figures

{figure('fig-main', FIG / 'fig_communication_comm_v3_seven_learners.png', f"""Communication between seven area groups on the first whisker day, learners (R+ green, {int(r0.n_rplus)} sessions; R− magenta, {int(r0.n_rminus)} sessions). **a–c**, RRR held-out R² minus the shuffle null, normalised by the within-area ceiling (Eqs. 1–2), session mean over all ordered pairs, for whisker hits (a), auditory hits (b) and whisker misses (c); mean ± s.e.m. across sessions; 300-ms windows, 100-ms steps; time 0: stimulus. **d–f**, Same for partial RRR (Eq. 3), removing the top 5 principal components of all other units. Black bars: R+ vs R− cluster permutation p < 0.05. **g**, Spearman correlation between methods of the early rise (Eq. 6) across session × pair. **h**, Fraction of the 40 ordered pairs with an early rise ≠ 0 on whisker hits (Wilcoxon per pair, BH q < 0.05), per method and cohort. **i**, R+ − R− mean early rise of RRR / ceiling on whisker hits per source → target pair; asterisks: Mann-Whitney p < 0.05, uncorrected; colour limit at the 98th percentile. **j**, Spearman correlation across areas of single-trial projections on the hit − miss axis (Eq. 5), hits + misses pooled; solid: raw $r$; dashed: after regressing out the rest of the brain; mean ± s.e.m. **k**, Same for the pre-lick axis, auditory hits + spontaneous licks, aligned to the first lick; bar: cluster p < 0.05. **l**, Held-out d′ along each area's axis (squares: hit − miss; dots: pre-lick), mean ± s.e.m. across sessions. Area colours: lab area-group palette. Script: 017_publication_figure.py (comm_v3_seven, cd_coupling_seven).""", script="017_publication_figure.py")}

# Supplementary material

{figure('fig-all', FIG / 'fig_communication_comm_v3_all_learners.png', "Same as Figure 1 for all ten v3 area groups (somatosensory-whisker and somatosensory-orofacial separate, plus posterior parietal and retrosplenial areas), learners (R+ 34, R− 36 sessions); panel h out of 70 ordered pairs; task-axis coupling from cd_coupling. Script: 017_publication_figure.py (comm_v3_all, cd_coupling).", script="017_publication_figure.py", kind="supplementary")}

{figure('fig-prep', FIG / 'fig_preprocessing_seven_learners.png', "Communication by preprocessing, seven area groups, learners (R+ green, R− magenta). **a–c**, RRR / ceiling over time on whisker hits for spike-count residuals (a), soft-normalised residuals (b) and soft-normalised residuals with each trial's −1.1 to −0.6 s baseline removed (c); session mean over pairs, mean ± s.e.m. across sessions; black bar: R+ vs R− cluster permutation p < 0.05. **d–f**, Same for partial RRR. **g**, Early rise on whisker hits per method and preprocessing (filled circles: counts; squares: normalised; triangles: trial baseline removed), mean ± s.e.m. across sessions. **h**, Pre-stimulus level, same layout. **i**, Fraction of pairs with an early rise ≠ 0 on whisker hits (Wilcoxon per pair, BH q < 0.05), bars per preprocessing. All measures divided by the within-area ceiling of the same preprocessing. Script: 018_preprocessing_comparison.py (comm_v3_seven, _z, _zbase).", script="018_preprocessing_comparison.py", kind="supplementary")}

{table('tbl-n', sample_table(), 'Sample sizes, learners, seven area groups: sessions in which each group has at least 15 stable units with a firing rate ≥ 1 Hz, per cohort, and the median number of such units. Pairs need both groups in the same session.')}

# Appendix

{R.registry_table('tbl-registry')}

### Result folders

combined_results_ks4/ssl-communication-methods-pilot/: comm_v3_seven/, comm_v3_all/ (016; per-session parquet and meta; learners/ and all_mice/ statistics, summary.md and detailed figures: per-pair forest plots, matrices, time courses), comm_v3_<areas>_z/, comm_v3_<areas>_zbase/ (016 --prep z / zbase, 2026-11-10), cd_coupling_seven/, cd_coupling/ (014, 015), report_figures/ (017, 018, with source-data CSVs per panel). Earlier stages of the pilot (002–013: hypothesis test, benchmark, time-resolved RRR v1/v2) are superseded by v3 and stay in their folders. Statistics read by this report are frozen in report/_snapshots/.
"""
    return md


def summary():
    r = P("cs_stats_pooled", "r2n", "W_lick", "rise_early")
    h = P("ds_stats_pooled", "r", "hitmiss:hm_pooled", "rise_early")
    return f"""# ssl-communication-methods-pilot: summary, {date.today().isoformat()}

1. Communication (RRR / ceiling) rises after whisker and auditory hits in both cohorts, equally (whisker hits: R+ {r.mean_rplus:.2f} vs R− {r.mean_rminus:.2f}, MW p = {r.mannwhitney_p:.2f}).
2. Removing the rest of the brain (partial RRR, partial CCA, ΔR²) leaves ~10 % of the rise and no pair with a significant rise.
3. Higher pre-stimulus communication in R− is a shared slow state: it disappears when each trial's baseline is removed; the evoked rise is unchanged by preprocessing (018).
4. Hit − miss axis coupling rises more in R+ (p = {h.mannwhitney_p:.3f}), but not after partialling the rest of the brain; pre-lick coupling vs gain is larger in R−.

![Main figure](figures/fig-main.png)
"""


def main():
    R.write(OUT, build(), summary=summary())


if __name__ == "__main__":
    main()
