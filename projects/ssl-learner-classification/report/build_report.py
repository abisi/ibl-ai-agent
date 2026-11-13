"""Report of ssl-learner-classification (skills/project-report, report_lib), first version 2026-11-13.

Agnostic learner / non-learner classification of whisker day 0 from the exact learning curves, the learning-trial
methods of ssl-learning-trial-identification and DTW clustering; comparison with the reference sheet and the earlier
behaviour clustering; performance of learners vs non-learners. Every number is read from the 001 result tables.
Run (haas): ~/code/ibl-ai-agent/.venv/bin/python projects/ssl-learner-classification/report/build_report.py
Render (PC, Git Bash): bash projects/ssl-learner-classification/report/render.sh
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(REPO / "scripts")]
from report_lib import Report  # noqa: E402
from axel_bisi_paths import axel_bisi_path  # noqa: E402

RES = axel_bisi_path("combined_results_ks4") / "ssl-learner-classification"
OUT = RES / "report"
TAB = RES / "tables"
SCRIPTS = "projects/ssl-learner-classification/exploratory-analyses"
CRITERIA = ["exact_rule", "high_state", "half2", "late30", "sustained", "L6", "cascade", "gmm", "dtw"]
CRIT_NAME = {"exact_rule": "exact-curve rule", "high_state": "high-performance state", "half2": "second-half test",
             "late30": "last-30-trial test", "sustained": "sustained run", "L6": "joint change point (L6)",
             "cascade": "lenient cascade", "gmm": "Gaussian mixture", "dtw": "DTW clustering"}

R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
R.register([
    ("eq-rw", "Eq. (1)"), ("eq-dprime", "Eq. (2)"), ("eq-frac", "Eq. (3)"), ("tbl-criteria", "Table 1"),
    ("eq-consensus", "Eq. (4)"), ("eq-auc", "Eq. (5)"),
    ("fig-main", "Figure 1"), ("tbl-disagree", "Table 2"),
    ("tbl-n", "Table S1"), ("tbl-perf", "Table S2"), ("tbl-stab", "Table S3"), ("tbl-rule", "Table S4"),
    ("tbl-clust", "Table S5"), ("tbl-registry", "Table S6"), ("tbl-scripts", "Table S7"),
])

SRC = {k: TAB / f"001_{k}.csv" for k in ("per_mouse", "agreement", "clustering_agreement", "validation_day1",
                                         "comparisons", "stability", "simple_rule")}
FROZEN = dict(zip(SRC, Report.snapshot(list(SRC.values()), OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}
PROV = json.loads((RES / "provenance_001.json").read_text())
d, ag, cl, val, comp, stab, rule = (T[k] for k in ("per_mouse", "agreement", "clustering_agreement", "validation_day1",
                                                    "comparisons", "stability", "simple_rule"))

# ------------------------------------------------------------------------------------------------ numbers
n = {c: int((d.cohort == c).sum()) for c in ("R+", "R-")}
nl = {c: int(d[d.cohort == c].learner.sum()) for c in n}
ns = {c: int(d[d.cohort == c].sheet_learner.fillna(False).astype(bool).sum()) for c in n}
dis = d[d.learner != d.sheet_learner.fillna(False).astype(bool)]
ndis = {c: int((dis.cohort == c).sum()) for c in n}
sheet_to_non = {c: int(((dis.cohort == c) & ~dis.learner).sum()) for c in n}
non_to_sheet = {c: int(((dis.cohort == c) & dis.learner).sum()) for c in n}
fixed_over = int((d.fixed_bad & d.learner_consensus).sum())
flex_learn = d[d.flexible & d.learner].mouse_id.tolist()
kap = ag[ag.reference == "final label"].set_index(["cohort", "labelling"]).kappa
crit_learn = {c: {k: int(d[(d.cohort == c)][f"c_{k}"].sum()) for k in CRITERIA} for c in n}
v = val.set_index(["cohort", "outcome", "labelling"])
st0 = stab[(stab.dropped != "none") & (stab.threshold == 0.5)]
stmin = st0.groupby("cohort").same_as_final.min()
th = stab[(stab.dropped == "none")].set_index(["cohort", "threshold"]).n_learner
ru = rule.set_index(["cohort", "feature"])
cp = comp[comp.labelling == "learner"].set_index(["cohort", "measure"])
cs = comp[comp.labelling == "sheet_learner"].set_index(["cohort", "measure"])
clu = cl.set_index(["cohort", "clustering", "reference"])
OLD = "DTW k=3, old curves (behavior_clustering.py)"
cons_R = {c: d[d.cohort == c].consensus for c in n}

# ------------------------------------------------------------------------------------------------ text
md = f"""---
title: "Who learns on whisker day 0? An agnostic classification of learners and non-learners"
date: "{date.today().isoformat()}"
---

**Data.** Whisker day-0 sessions of {num('n_mice', len(d))} mice ({num('n_rp', n['R+'])} R+, {num('n_rm', n['R-'])} R−) from the
NWB_ks4 files (shared behaviour table of the paper figures), exact learning curves (grid HMM, σ = 1), learning-trial methods of
ssl-learning-trial-identification (table 028), the earlier behaviour clustering of `behaviour_analysis/behavior_clustering.py`.
Code: `{SCRIPTS}` ({ref('tbl-scripts')}); results: `combined_results_ks4/ssl-learner-classification`; every claim with its
statistic is listed in the claims registry ({ref('tbl-registry')}).

# Abstract

The reference sheet labels mice as good, moderate or bad learners, but the label is not tied to an explicit rule. We
classified every whisker day-0 session with nine criteria computed from the exact learning curves, the session's lick
rates, change-point learning-trial methods and two unsupervised models, and called a mouse a learner when at least half of
the criteria were met. In R+, {num('nl_rp', nl['R+'])} of {n['R+']} mice are learners by this consensus, against
{num('ns_rp', ns['R+'])} by the sheet; {num('dis_rp', ndis['R+'])} mice change label, almost all sheet good or moderate mice
that never discriminate the whisker stimulus from the no-stimulus rate. The user's fixed non-learners were never called
learners ({num('fixed_over', fixed_over)} overridden). The agnostic label predicts whisker performance on the next day better
than the sheet (AUC {num('auc_rp_final', v.loc[('R+', 'hit_w_day1', 'final label'), 'auc'], '{:.2f}')} vs
{num('auc_rp_sheet', v.loc[('R+', 'hit_w_day1', 'sheet'), 'auc'], '{:.2f}')}), and one number summarises it: an R+ learner has its
whisker curve credibly above the no-stimulus curve on at least {num('th_rp', ru.loc[('R+', 'frac_above'), 'threshold'], '{:.2f}')}
of its whisker trials (balanced accuracy {num('ba_rp', ru.loc[('R+', 'frac_above'), 'balanced_acc'], '{:.2f}')}). In R−, the criteria
agree that almost every mouse learns ({num('nl_rm', nl['R-'])} of {n['R-']}); the {num('nn_rm', n['R-'] - nl['R-'])} non-learners keep
licking the non-rewarded whisker stimulus above chance, and day +1 does not separate the groups.

**Key results**

1. {claim('c1', f"R+: {nl['R+']} learners and {n['R+'] - nl['R+']} non-learners by consensus of nine criteria; the label is stable to dropping any one criterion (at least {stmin['R+']:.0%} of mice unchanged).", 'Figure 1A, B; Table S3', f"leave-one-out min {stmin['R+']:.2f}", f"{n['R+']} mice")}
2. {claim('c2', f"The agnostic label and the sheet agree in R+ with κ = {kap[('R+', 'sheet')]:.2f}; {sheet_to_non['R+']} sheet learners are non-learners, {non_to_sheet['R+']} sheet non-learner is a learner (flexible mouse {', '.join(flex_learn) or 'none'}).", 'Figure 1E; Table 2', f"κ {kap[('R+', 'sheet')]:.2f}", f"{n['R+']} mice")}
3. {claim('c3', f"Day +1 validation, R+: agnostic label AUC {v.loc[('R+', 'hit_w_day1', 'final label'), 'auc']:.2f} (whisker P(lick)) and {v.loc[('R+', 'dprime_day1', 'final label'), 'auc']:.2f} (d'), sheet {v.loc[('R+', 'hit_w_day1', 'sheet'), 'auc']:.2f} and {v.loc[('R+', 'dprime_day1', 'sheet'), 'auc']:.2f}.", 'Figure 1C', 'Mann-Whitney AUC', f"{int(v.loc[('R+', 'hit_w_day1', 'final label'), 'n'])} mice")}
4. {claim('c4', f"What a learner is (R+): fraction of whisker trials with the whisker curve credibly above chance ≥ {ru.loc[('R+', 'frac_above'), 'threshold']:.2f} (leave-one-out accuracy {ru.loc[('R+', 'frac_above'), 'loo_accuracy']:.2f}); or d' ≥ {ru.loc[('R+', 'dprime'), 'threshold']:.2f}.", 'Figure 1D; Table S4', 'balanced accuracy', f"{n['R+']} mice")}
5. {claim('c5', f"R+ learners lick the whisker stimulus more (P(lick) {cp.loc[('R+', 'hit_w'), 'learner_mean']:.2f} vs {cp.loc[('R+', 'hit_w'), 'non_mean']:.2f}) with similar false-alarm and auditory rates, react faster and hit earlier; the difference persists on day +1.", 'Figure 1F, G; Table S2', 'Mann-Whitney + Welch', f"{nl['R+']} vs {n['R+'] - nl['R+']} mice")}
6. {claim('c6', f"R−: {nl['R-']} of {n['R-']} mice are learners; non-learners are defined by persistent whisker licking above chance, but the split is small and does not predict day +1.", 'Figure 1A, C, D', f"AUC {v.loc[('R-', 'hit_w_day1', 'final label'), 'auc']:.2f}", f"{n['R-']} mice", status='fragile')}

# Introduction

The paper restricts day-0 neural analyses to learners, and the sheet's `learning_category` (good, moderate, bad) defines
them. The category was assigned by inspection of the learning curves, so it is not reproducible from a rule and mixes two
things: whether the mouse acquires the contingency, and how cleanly. Several explicit definitions exist in the lab's code
(the stored learning-trial rule, performance states, change-point models) and a clustering of the curves, and they do not
agree with each other or with the sheet.

The question is therefore which mice learn on day 0 by a definition that does not depend on the sheet, and how the sheet
and the clustering compare with it. We combine many definitions instead of choosing one, because each criterion has a
known failure mode (lenient rules accept short excursions above chance; rate tests need enough trials; unsupervised
models split the data whether or not there are two groups), and a majority over criteria with different failure modes is
more robust than any single one. Next-day performance, which no criterion uses, serves as the validation.

# Methods

## Task, trials and cohorts

{R.methods('task')}Trials: performance code 6 and passive trials removed, the day-0 auditory warm-up block cut except its last
trial, and early-lick trials dropped (as in Figure 1 of the paper). One day-0 session per mouse (the longest if several).
No disengagement trimming is applied, since learning is assessed over the whole session.

## Learning curves

The probability of licking on trial $t$ of one trial type follows a random walk on the logit scale,
{eq('eq-rw', r"x_t = x_{t-1} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2),\quad p_t = \frac{1}{1 + e^{-x_t}},\quad y_t \sim \mathrm{Bernoulli}(p_t)")}
where $y_t$ is the lick outcome and $\\sigma = 1$. The posterior of $p_t$ given all trials is computed exactly on a grid
(forward–backward, `learning_curve_exact.py`); $\\bar p_t$ is its mean and $p^{{\\mathrm{{low}}}}_t$, $p^{{\\mathrm{{high}}}}_t$ the
10th and 90th percentiles. The whisker curve is fitted on whisker trials; the chance curve $p^{{\\mathrm{{ch}}}}_t$ is the
no-stimulus curve fitted on no-stimulus trials with the same model ({ref('eq-rw')}) and interpolated in time onto the whisker trials.

## Features

Rates and discrimination per session: whisker, no-stimulus (false alarm, FA) and auditory P(lick), and the sensitivity
{eq('eq-dprime', r"d' = \Phi^{-1}\!\left(\frac{h + 0.5}{n_W + 1}\right) - \Phi^{-1}\!\left(\frac{f + 0.5}{n_N + 1}\right)")}
where $h$ and $f$ are the numbers of licked whisker and no-stimulus trials, $n_W$ and $n_N$ the numbers of trials and
$\\Phi^{{-1}}$ the standard normal quantile (log-linear correction). From the curves, the fraction of whisker trials on which
the whisker curve is credibly above chance,
{eq('eq-frac', r"F_{\mathrm{above}} = \frac{1}{n_W} \sum_{t=1}^{n_W} \mathbb{1}\left[p^{\mathrm{low}}_t > p^{\mathrm{ch}}_t\right]")}
its longest run, the longest run not above chance after the first whisker hit, and the gap $\\bar p_t - p^{{\\mathrm{{ch}}}}_t$
(mean, maximum, last 20 trials). Second-half and last-30-trial whisker and FA rates use the no-stimulus trials of the same
period.

## Criteria

R+ mice learn to lick the rewarded whisker stimulus above the no-stimulus rate; R− mice learn to stop licking the
non-rewarded whisker stimulus, i.e. to bring it back to the no-stimulus rate after the initial generalisation from the
auditory stimulus. Each criterion is applied in the direction of its cohort ({ref('tbl-criteria')}).

"""
crit_tbl = pd.DataFrame([
    ("Exact-curve rule", "learning_utils rule on the exact curves (5 consecutive trials credibly above chance; expert if high from the start)", "same rule for the non-rewarded case", crit_learn["R+"]["exact_rule"], crit_learn["R-"]["exact_rule"]),
    ("High-performance state", "any state with p_low > p_ch for 5 consecutive trials (assign_expertise_blocks)", "any state with p_low < p_ch for 5 trials", crit_learn["R+"]["high_state"], crit_learn["R-"]["high_state"]),
    ("Second-half test", "Fisher exact, whisker > FA P(lick), p < 0.05, second half of the whisker trials", "not significant (p ≥ 0.05)", crit_learn["R+"]["half2"], crit_learn["R-"]["half2"]),
    ("Last-30-trial test", "same on the last 30 whisker trials", "not significant", crit_learn["R+"]["late30"], crit_learn["R-"]["late30"]),
    ("Sustained run", f"longest run credibly above chance ≥ {PROV['run_min']} trials", f"longest run not above chance after the first hit ≥ {PROV['run_min']}", crit_learn["R+"]["sustained"], crit_learn["R-"]["sustained"]),
    ("Joint change point (L6)", "whisker + FA segment model: learner or high from start", "learner or never licked", crit_learn["R+"]["L6"], crit_learn["R-"]["L6"]),
    ("Lenient cascade", "learning trial by L6 lenient, else whisker change point, else half-way rule; or high from start", "same; or never licked", crit_learn["R+"]["cascade"], crit_learn["R-"]["cascade"]),
    ("Gaussian mixture", "2 components on 9 standardised features; component with the higher last-30 gap", "component with the lower gap", crit_learn["R+"]["gmm"], crit_learn["R-"]["gmm"]),
    ("DTW clustering", "k = 2 time-series k-means (DTW) on whisker + chance curves, 100 points; cluster with the higher gap", "cluster with the lower gap", crit_learn["R+"]["dtw"], crit_learn["R-"]["dtw"]),
], columns=["Criterion", "R+ learner if", "R− learner if", "R+ learners", "R− learners"])
md += table("tbl-criteria", crit_tbl, "The nine learner criteria, their rule per cohort and the number of mice meeting each "
            f"(of {n['R+']} R+ and {n['R-']} R− mice). p_low, lower bound of the 80 % posterior band of the whisker curve; p_ch, chance "
            "(no-stimulus) curve; gap, whisker − FA P(lick).")
md += f"""
The consensus of mouse $i$ is the fraction of its criteria that are met,
{eq('eq-consensus', r"c_i = \frac{1}{|K_i|} \sum_{k \in K_i} \mathbb{1}\left[\text{criterion } k \text{ met by mouse } i\right]")}
where $K_i$ are the criteria available for the mouse (all nine, except the two change-point criteria for the
{num('n_missing_lt', int(d.c_L6.isna().sum()))} mice outside table 028). A mouse is a learner if $c_i \\ge 0.5$ in
{ref('eq-consensus')}. Constraint (user): the R+ mice {', '.join(PROV['fixed_bad'])} are non-learners whatever their
consensus; the flexible R+ mice ({', '.join(PROV['flexible'])}) take the consensus label. The sheet labels (good,
moderate = learner; bad = non-learner) are used only for the comparison.

## Comparisons, validation and robustness

Agreement of binary labels is Cohen's κ; agreement of clusterings with the sheet categories and with the agnostic label is
the adjusted Rand index (ARI). The earlier clustering is `behavior_clustering.py` (3 clusters per cohort, whisker and FA
channels, DTW, on the 2025 PyMC curves), read from its saved assignments; the same clustering was rerun on the exact curves
with k = 3. Validation: for each labelling, the area under the ROC curve of next-day performance,
{eq('eq-auc', r"\mathrm{AUC} = P\!\left(s\,y_{\mathrm{learner}} > s\,y_{\mathrm{non\text{-}learner}}\right)")}
where $y$ is whisker P(lick) or d′ on day +1 ({ref('eq-dprime')}) and $s = +1$ for R+, $-1$ for R−; p from the Mann–Whitney test.
Robustness: the label recomputed without each criterion in turn and with thresholds 0.4 and 0.6 on $c_i$. A single-feature
rule: for each feature, the threshold with the best balanced accuracy for the agnostic label, and its leave-one-out
accuracy (threshold re-chosen without the held-out mouse). Learners and non-learners are compared per cohort with the
Mann–Whitney and Welch tests (mice), without correction across measures. Sample sizes are in {ref('tbl-n')}.

# Results

## Criteria differ in leniency, and their consensus is stable

The criteria range from lenient to strict ({ref('tbl-criteria')}, {ref('fig-main')}A). In R+, the exact-curve rule and the
high-state criterion accept {num('lenient_rp', crit_learn['R+']['exact_rule'])} of {n['R+']} mice, since five consecutive
trials above chance occur by chance in long sessions, while the rate tests on the second half and the last 30 trials
accept {num('half2_rp', crit_learn['R+']['half2'])} and {num('late30_rp', crit_learn['R+']['late30'])}. The change-point,
mixture and DTW criteria lie in between and agree best with the consensus (κ of the joint change point
{num('k_L6_rp', kap[('R+', 'L6')], '{:.2f}')}, Gaussian mixture {num('k_gmm_rp', kap[('R+', 'gmm')], '{:.2f}')}, DTW
{num('k_dtw_rp', kap[('R+', 'dtw')], '{:.2f}')}; {ref('fig-main')}B). The consensus splits R+ into
{num('nl_rp2', nl['R+'])} learners and {num('nn_rp', n['R+'] - nl['R+'])} non-learners. Dropping any single criterion changes
the label of at most {num('drop_max_rp', int(round((1 - stmin['R+']) * n['R+'])))} mice; thresholds 0.4 and 0.6 give
{num('th04_rp', int(th[('R+', 0.4)]))} and {num('th06_rp', int(th[('R+', 0.6)]))} learners ({ref('tbl-stab')}).

None of the user's fixed non-learners reaches the consensus threshold, so the constraint changes no label
({num('fixed_over2', fixed_over)} mice overridden). Of the flexible mice, {', '.join(flex_learn) or 'none'} becomes a learner;
the others stay non-learners.

## The sheet calls more R+ mice learners than the criteria do

The agnostic label and the sheet agree on {num('agree_rp', int(n['R+'] - ndis['R+']))} of {n['R+']} R+ mice (κ =
{num('k_sheet_rp', kap[('R+', 'sheet')], '{:.2f}')}; {ref('fig-main')}E, {ref('tbl-disagree')}). The disagreements are one-sided:
{num('s2n_rp', sheet_to_non['R+'])} sheet good or moderate mice are non-learners by the consensus, and
{num('n2s_rp', non_to_sheet['R+'])} sheet bad mouse is a learner. The mice that move lick the whisker stimulus at a rate close
to their false-alarm rate for most of the session ({ref('fig-main')}D): their whisker curve is credibly above chance on
less than a quarter of the whisker trials, typically in one early excursion, which the lenient rules accept and the rate
tests and models do not.

The earlier DTW clustering agrees with the sheet categories better than with the agnostic label (R+ ARI
{num('ari_old_sheet', clu.loc[('R+', OLD, 'sheet category'), 'ari'], '{:.2f}')} vs
{num('ari_old_final', clu.loc[('R+', OLD, 'final label'), 'ari'], '{:.2f}')}; {ref('tbl-clust')}). Rerun on the exact curves with
k = 3, it agrees less with both (ARI {num('ari_new_sheet', clu.loc[('R+', 'DTW k=3, exact curves', 'sheet category'), 'ari'], '{:.2f}')}
and {num('ari_new_final', clu.loc[('R+', 'DTW k=3, exact curves', 'final label'), 'ari'], '{:.2f}')}); with two clusters, used as a
criterion, it agrees with the agnostic label at ARI {num('ari_k2_final', clu.loc[('R+', 'DTW k=2, exact curves', 'final label'), 'ari'], '{:.2f}')}.
Curve clustering alone is therefore a weak learner definition: the clusters follow the shape of the whole curve, including
the overall licking level and the late decline, not only the whisker–FA separation.

## The agnostic label predicts next-day performance

No criterion uses day +1. In R+, the agnostic label separates next-day whisker P(lick) with AUC
{num('auc_rp_final2', v.loc[('R+', 'hit_w_day1', 'final label'), 'auc'], '{:.2f}')} (p {pv('p_rp_final', v.loc[('R+', 'hit_w_day1', 'final label'), 'p'])})
and next-day d′ with AUC {num('auc_rp_final_d', v.loc[('R+', 'dprime_day1', 'final label'), 'auc'], '{:.2f}')}, against
{num('auc_rp_sheet2', v.loc[('R+', 'hit_w_day1', 'sheet'), 'auc'], '{:.2f}')} and
{num('auc_rp_sheet_d', v.loc[('R+', 'dprime_day1', 'sheet'), 'auc'], '{:.2f}')} for the sheet ({ref('fig-main')}C). The lenient
criteria are the worst predictors (exact-curve rule AUC {num('auc_rp_exact', v.loc[('R+', 'hit_w_day1', 'exact_rule'), 'auc'], '{:.2f}')}),
the strict ones the best (Gaussian mixture {num('auc_rp_gmm', v.loc[('R+', 'hit_w_day1', 'gmm'), 'auc'], '{:.2f}')}, second-half
test {num('auc_rp_half2', v.loc[('R+', 'hit_w_day1', 'half2'), 'auc'], '{:.2f}')}). The consensus score itself correlates with
next-day whisker P(lick) (Spearman ρ = {num('rho_rp', v.loc[('R+', 'hit_w_day1', 'consensus score (Spearman ρ)'), 'auc'], '{:.2f}')}).

In R−, no labelling predicts day +1 (agnostic label AUC
{num('auc_rm_final', v.loc[('R-', 'hit_w_day1', 'final label'), 'auc'], '{:.2f}')}, sheet
{num('auc_rm_sheet', v.loc[('R-', 'hit_w_day1', 'sheet'), 'auc'], '{:.2f}')}): on day +1 almost every R− mouse licks the whisker
stimulus at the no-stimulus rate, so there is little to predict.

## What a learner is

One feature reproduces the R+ consensus: the fraction of whisker trials with the whisker curve credibly above chance
({ref('eq-frac')}). Learners have $F_{{\\mathrm{{above}}}} \\ge$ {num('th_rp2', ru.loc[('R+', 'frac_above'), 'threshold'], '{:.2f}')}
(balanced accuracy {num('ba_rp2', ru.loc[('R+', 'frac_above'), 'balanced_acc'], '{:.2f}')}, leave-one-out accuracy
{num('loo_rp', ru.loc[('R+', 'frac_above'), 'loo_accuracy'], '{:.2f}')}; {ref('fig-main')}D, {ref('tbl-rule')}). Session d′
({ref('eq-dprime')}) works almost as well (d′ ≥ {num('th_d_rp', ru.loc[('R+', 'dprime'), 'threshold'], '{:.2f}')}, leave-one-out
accuracy {num('loo_d_rp', ru.loc[('R+', 'dprime'), 'loo_accuracy'], '{:.2f}')}). In words: an R+ learner discriminates the
whisker stimulus from no stimulus for a sustained part of the session, not only in a brief excursion. An R− non-learner is
the mirror image, a mouse whose whisker curve stays credibly above chance on more than
{num('th_rm', ru.loc[('R-', 'frac_above'), 'threshold'], '{:.2f}')} of the whisker trials, i.e. that keeps generalising from the
auditory stimulus.

## Learners and non-learners differ in whisker licking, not in general engagement

R+ learners lick on {num('hw_l', cp.loc[('R+', 'hit_w'), 'learner_mean'], '{:.2f}')} of whisker trials against
{num('hw_n', cp.loc[('R+', 'hit_w'), 'non_mean'], '{:.2f}')} for non-learners (Mann–Whitney p
{pv('p_hw', cp.loc[('R+', 'hit_w'), 'mannwhitney_p'])}), with similar false-alarm (p {pv('p_fa', cp.loc[('R+', 'fa'), 'mannwhitney_p'])})
and auditory P(lick) (p {pv('p_ha', cp.loc[('R+', 'hit_a'), 'mannwhitney_p'])}) and the same number of pretraining trials
(p {pv('p_pre', cp.loc[('R+', 'n_pretraining_trials'), 'mannwhitney_p'])}) ({ref('fig-main')}F, G; {ref('tbl-perf')}). Their
whisker reaction times are shorter ({num('rt_l', cp.loc[('R+', 'rt_w'), 'learner_mean'], '{:.2f}')} vs
{num('rt_n', cp.loc[('R+', 'rt_w'), 'non_mean'], '{:.2f}')} s, p {pv('p_rt', cp.loc[('R+', 'rt_w'), 'mannwhitney_p'])}) and they hit
earlier ({num('fh_l', cp.loc[('R+', 'first_hit'), 'learner_mean'], '{:.1f}')} vs
{num('fh_n', cp.loc[('R+', 'first_hit'), 'non_mean'], '{:.1f}')} whisker trials before the first hit, p
{pv('p_fh', cp.loc[('R+', 'first_hit'), 'mannwhitney_p'])}). On day +1 the gap remains (whisker P(lick)
{num('hw1_l', cp.loc[('R+', 'hit_w_day1'), 'learner_mean'], '{:.2f}')} vs {num('hw1_n', cp.loc[('R+', 'hit_w_day1'), 'non_mean'], '{:.2f}')},
p {pv('p_hw1', cp.loc[('R+', 'hit_w_day1'), 'mannwhitney_p'])}), although the non-learners improve. With the sheet split the
same comparisons are weaker (day +1 whisker P(lick) p {pv('p_hw1_sheet', cs.loc[('R+', 'hit_w_day1'), 'mannwhitney_p'])}).

The mean curves show the difference directly ({ref('fig-main')}F): learners' whisker curve rises above the false-alarm
curve early in the session and stays above it until the late decline, whereas non-learners' whisker curve tracks their
false-alarm curve throughout. In R−, the {n['R-'] - nl['R-']} non-learners lick the whisker stimulus more than learners
(P(lick) {num('hw_rm_n', cp.loc[('R-', 'hit_w'), 'non_mean'], '{:.2f}')} vs {num('hw_rm_l', cp.loc[('R-', 'hit_w'), 'learner_mean'], '{:.2f}')}),
which is the definition, and do not differ on day +1.

# Discussion

A majority over nine criteria with different failure modes gives a learner label for R+ that is stable, reproducible from
one sentence ("whisker curve credibly above chance on at least a quarter of the whisker trials") and better at predicting
next-day performance than the sheet. The sheet errs on the generous side: it calls learners several mice whose curves
cross chance only briefly. For the day-0 neural analyses restricted to learners, the agnostic label would remove these
mice; whether that changes the paper's learner-restricted results can be checked with the sensitivity analysis already
used for the flexible mice (ssl-within-day-two-axes 020).

The lenient criteria (stored rule, high state) are poor learner definitions on their own: five consecutive trials above
chance is a low bar when sessions last 100–200 whisker trials. They remain useful to time learning in mice already known to
learn, which is how the learning trial is used.

For R−, learning on day 0 means suppressing the whisker response to the no-stimulus level. Almost all R− mice do this, so
the classification has little to separate, and no labelling predicts day +1; the three R− non-learners are best described
as mice that keep generalising from the auditory stimulus.

# Caveats and limitations

- The criteria are not independent (several use the same curves), so the consensus is not a vote of independent judges;
  the robustness checks and the day +1 validation address this only in part.
- The R+ criteria and the sheet were both built from day-0 behaviour; the day +1 validation is the only external check and
  covers {int(v.loc[('R+', 'hit_w_day1', 'final label'), 'n'])} of {n['R+']} R+ mice.
- R− has {n['R-'] - nl['R-']} non-learners; statistics for R− learners vs non-learners are not interpretable.
- The two change-point criteria are missing for the mice outside table 028; their consensus uses seven criteria.
- The DTW criterion depends on the random initialisation (one seed, 5 initialisations); k = 2 was fixed, not selected.
- Comparisons across measures are not corrected for multiple testing.

# Open questions and next steps

- Rerun the learner-restricted day-0 analyses of the paper with the agnostic label (sensitivity analysis).
- Decide whether Figure 1 and Supplementary Figure 1 use the agnostic label; if so, add the supplementary figure to the
  figure planner.
- Define day-0 learning in R− more finely (e.g. time of suppression), if it is needed beyond learner vs non-learner.

# Figures

"""
md += figure("fig-main", RES / "figures" / "SuppFigure_learners_tight.png",
             "Agnostic classification of learners on whisker day 0. "
             f"**A**, the nine learner criteria ({ref('tbl-criteria')}) per mouse (columns; R+ left, n = {n['R+']}, R− right, n = {n['R-']}), "
             "sorted within cohort by consensus; cohort colour, criterion met; light grey, not met; white, not available. Below: consensus "
             f"(fraction of criteria met, {ref('eq-consensus')}; darker = more), agnostic label (cohort colour, learner; dark grey, non-learner), sheet "
             "category (cohort colour, good; light, moderate; dark grey, bad), the earlier DTW clusters (behavior_clustering.py, 3 clusters, "
             "grey levels) and the user's fixed (black) and flexible (grey) R+ mice. "
             "**B**, agreement (Cohen's κ) of each criterion and of the sheet with the agnostic label, per cohort; 'all mice met', criterion "
             "constant in that cohort. "
             f"**C**, validation: AUC ({ref('eq-auc')}) of each labelling for whisker P(lick) (circles) and d′ (squares) on day +1, R+ "
             "(green) and R− (magenta; learners expected lower); filled, Mann–Whitney p < 0.05; dotted line, chance. "
             "**D**, what a learner is: fraction of whisker trials with the whisker curve credibly above chance per mouse "
             f"({ref('eq-frac')}), by agnostic label (L learner, NL non-learner; marker = sheet category); dashed line, best single threshold "
             "(balanced accuracy). "
             "**E**, sheet category against the agnostic label (numbers of mice; bold, agreement). "
             "**F**, mean ± s.e.m. over mice of the exact whisker (cohort colour) and false-alarm (black) curves against session progress, "
             "learners (solid) and non-learners (dashed). "
             "**G**, performance of learners (L) and non-learners (NL): whisker, false-alarm and auditory P(lick) and d′ on day 0, whisker "
             "P(lick) and d′ on day +1; dots, mice; black-edged dot, mean ± s.e.m.; stars, Mann–Whitney (* p < 0.05, ** < 0.01, *** < 0.001; "
             f"n.s., not significant). Mice: {n['R+']} R+, {n['R-']} R− (day +1: those with a day +1 session).",
             script=f"{SCRIPTS}/002_supp_figure.py")
dtab = dis.assign(**{"Sheet": dis.learning_category, "Consensus": dis.consensus.round(2),
                     "Criteria met": (dis.consensus * dis.n_criteria).round().astype(int).astype(str) + " / " + dis.n_criteria.astype(int).astype(str),
                     "Agnostic label": np.where(dis.learner, "learner", "non-learner"),
                     "F_above": dis.frac_above.round(2), "d' day 0": dis.dprime.round(2),
                     "W P(lick) day +1": dis.hit_w_day1.round(2),
                     "Note": np.where(dis.flexible, "flexible (user)", "")})
md += table("tbl-disagree", dtab[["mouse_id", "cohort", "Sheet", "Agnostic label", "Consensus", "Criteria met", "F_above",
                                  "d' day 0", "W P(lick) day +1", "Note"]].rename(columns={"mouse_id": "Mouse", "cohort": "Cohort"}),
            "Mice whose agnostic label differs from the sheet. Consensus, fraction of criteria met "
            f"({ref('eq-consensus')}); F_above, fraction of whisker trials with the whisker curve credibly above chance ({ref('eq-frac')}).")
md += "\n# Supplementary tables\n"
ntab = pd.DataFrame([dict(Cohort=c, Mice=n[c], **{"Agnostic learners": nl[c], "Sheet learners": ns[c],
                                                  "With day +1": int(d[(d.cohort == c)].hit_w_day1.notna().sum()),
                                                  "In table 028 (change points)": int(d[(d.cohort == c)].c_L6.notna().sum()),
                                                  "Whisker trials, median": int(d[d.cohort == c].n_whisker.median())}) for c in n])
md += table("tbl-n", ntab, "Sample sizes (mice; one whisker day-0 session per mouse).")
pt = comp[comp.labelling == "learner"].copy()
pt = pt.assign(**{"Learners": pt.learner_mean.round(3).astype(str) + " ± " + pt.learner_sem.round(3).astype(str),
                  "Non-learners": pt.non_mean.round(3).astype(str) + " ± " + pt.non_sem.round(3).astype(str),
                  "n": pt.n_learner.astype(str) + " / " + pt.n_non.astype(str),
                  "MW p": pt.mannwhitney_p.map(lambda x: "" if pd.isna(x) else f"{x:.2g}"),
                  "Welch p": pt.welch_p.map(lambda x: "" if pd.isna(x) else f"{x:.2g}")})
md += table("tbl-perf", pt[["cohort", "label", "n", "Learners", "Non-learners", "MW p", "Welch p"]].rename(
    columns={"cohort": "Cohort", "label": "Measure"}),
    "Learners vs non-learners (agnostic label), mean ± s.e.m. over mice; Mann–Whitney (MW) and Welch tests, uncorrected.")
sb = stab[(stab.threshold == 0.5) | (stab.dropped == "none")].copy()
sb["Variant"] = np.where(sb.dropped == "none", "threshold " + sb.threshold.astype(str), "without " + sb.dropped.map(CRIT_NAME))
md += table("tbl-stab", sb[["cohort", "Variant", "n_learner", "same_as_final", "changed"]].assign(
    same_as_final=sb.same_as_final.round(3)).rename(columns={"cohort": "Cohort", "n_learner": "Learners",
                                                             "same_as_final": "Same label", "changed": "Mice that change"}),
    f"Robustness of the agnostic label: each criterion left out (threshold 0.5 on {ref('eq-consensus')}) and thresholds 0.4 / 0.6 "
    "with all criteria; fraction of mice with the same label as the final one.")
md += table("tbl-rule", rule.assign(threshold=rule.threshold.round(3), balanced_acc=rule.balanced_acc.round(3),
                                    loo_accuracy=rule.loo_accuracy.round(3), auc=rule.auc.round(3))[
    ["cohort", "label", "side", "threshold", "balanced_acc", "loo_accuracy", "auc"]].rename(
    columns={"cohort": "Cohort", "label": "Feature", "side": "Learner if", "threshold": "Threshold",
             "balanced_acc": "Balanced acc.", "loo_accuracy": "Leave-one-out acc.", "auc": "AUC"}),
    "Single-feature rules for the agnostic label: best threshold (balanced accuracy), leave-one-out accuracy with the threshold "
    "re-chosen without the held-out mouse, and AUC of the feature (learners vs non-learners).")
md += table("tbl-clust", cl.assign(ari=cl.ari.round(3), nmi=cl.nmi.round(3)).rename(
    columns={"cohort": "Cohort", "clustering": "Clustering", "reference": "Compared with", "n": "Mice", "ari": "ARI", "nmi": "NMI"}),
    "Agreement of the DTW clusterings with the sheet categories (good / moderate / bad) and with the agnostic label: adjusted "
    "Rand index (ARI) and normalised mutual information (NMI).")
md += "\n# Appendix\n"
md += R.registry_table("tbl-registry")
md += table("tbl-scripts", pd.DataFrame([
    (f"{SCRIPTS}/001_learner_classification.py", "features, criteria, consensus, agreement, validation, stability, rule, comparisons",
     "tables/001_*.csv, cache/001_*"),
    (f"{SCRIPTS}/002_supp_figure.py", "supplementary figure", "figures/SuppFigure_learners.*, manuscript Figures/SuppFigure_learners_<date>"),
    ("projects/ssl-paper-figures/learning_curve_exact.py", "exact curves and the stored learning-trial rule", "-"),
    ("projects/ssl-learning-trial-identification (028)", "change-point learning-trial methods", "artifacts/028_learning_trials_all_methods_all_mice.csv"),
    ("behaviour_analysis/behavior_clustering.py", "earlier DTW clustering (3 clusters)", "combined_results/behavior_clustering/3_both_dtw_interpolated_input"),
], columns=["Script", "What", "Outputs"]), "Scripts and outputs (results under combined_results_ks4/ssl-learner-classification).")

summary = f"""# ssl-learner-classification: summary ({date.today().isoformat()})

- R+: {nl['R+']} learners / {n['R+']} by the consensus of nine criteria (sheet: {ns['R+']}); {ndis['R+']} mice change label.
- Agnostic label predicts day +1 whisker P(lick) better than the sheet (AUC {v.loc[('R+', 'hit_w_day1', 'final label'), 'auc']:.2f} vs {v.loc[('R+', 'hit_w_day1', 'sheet'), 'auc']:.2f}).
- Rule: whisker curve credibly above chance on ≥ {ru.loc[('R+', 'frac_above'), 'threshold']:.2f} of whisker trials.
- R−: {nl['R-']} / {n['R-']} learners; split not validated by day +1.

![Supplementary figure](figures/fig-main.png)
"""
R.write(OUT, md, summary=summary)
