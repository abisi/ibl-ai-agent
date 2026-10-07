"""Merged article-style report of ssl-prelick-convergence (skills/project-report, report_lib): Part I across days,
Part II within days (learning day and expert sessions). Run on haas after 062 (full, both populations), 074-078 and the
within_day scripts; render locally with render.sh.

Main population: learners (061 rule); reference: spontaneous licks (SL). Every quoted number is read from the result
tables (062 stats, 074/075/078 supplements, within_day halves / slopes / epochs / mixed model) through report_lib.
Settings are those of the current results (SL definition of 051, pooled-SD unit scaling, good + mua units); the decided
changes (C8 reference, within-class SD, stable units, 4 events per class) are listed as pending in the caveats.
Run (haas):  cd ~/code/unit_spikes_analysis && PRELICK_REF=sl PYTHONPATH=~/code/NWB_reader:. ./.venv/bin/python \
             ~/code/ibl-ai-agent/projects/ssl-prelick-convergence/report/build_report.py
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EA = PROJ / "exploratory-analyses"
sys.path.insert(0, str(EA))
sys.path.insert(0, str(REPO / "skills" / "project-report"))
from report_lib import Report  # noqa: E402

m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
POP = "learners"
PUB = m51.OUTROOT / "publication"
P1 = PUB / POP
WD = m51.WITHIN
NORM = m51.OUTROOT / "normalisation"
OUT = m51.HOME / "report"

R = Report(banned=())
num, pv, ref, figure, table = R.num, R.pv, R.ref, R.figure, R.table
G4 = ["R+ learning", "R+ expert", "R− learning", "R− expert"]


# ---------------------------------------------------------------------------------------------------------- tables
def rd(f):
    return pd.read_csv(f) if Path(f).exists() else pd.DataFrame()


ST = {POP: rd(P1 / f"stats_{POP}.csv"), "all": rd(PUB / "all" / "stats_all.csv")}
SS = rd(P1 / f"stats_supp_selectivity_angles_{POP}.csv")
S78 = rd(P1 / f"stats_supp_078_{POP}.csv")
NS = rd(NORM / "stats.csv")
HV = rd(WD / "halves" / POP / "within_session_tests.csv")
SL = rd(WD / "slopes" / POP / "trial_slopes_tests.csv")
EP = rd(WD / "epochs" / POP / "epoch_contrasts.csv")
MM = rd(WD / "mixed_model" / "mixed_model_terms.csv")
TLS = rd(WD / "cosyne" / "COSYNE_convergence_timeline_stats.csv")


def row(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    r = df[m]
    if len(r) == 0:
        raise KeyError(f"no row for {kw}")
    return r.iloc[0]


def P(v):
    v = float(v)
    return "n/a" if not np.isfinite(v) else ("< 0.001" if v < 0.001 else f"= {v:.3f}" if v < 0.01 else f"= {v:.2f}")


def grp(panel, key, df=None, fmt="{:.3f}", unit=""):
    """R+ / R- learning -> expert means, MWU per cohort, expert cohort contrast, interaction (dots_panel stats row)"""
    r = row(ST[POP] if df is None else df, panel=panel)
    k = f"{key}"
    vals = [num(f"{k}|mean {g}", r[f"mean_{g}"], fmt) for g in G4]
    ns = [int(r[f"n_{g}"]) for g in G4]
    pr, pm, pe = (num(f"{k}|p R+ LvE", r["R+ L vs E p_MWU"], "{:.4g}"), num(f"{k}|p R- LvE", r["R- L vs E p_MWU"], "{:.4g}"),
                  num(f"{k}|p exp R+ vs R-", r["expert R+ vs R- p_MWU"], "{:.4g}"))
    pi = num(f"{k}|p interaction", r["interaction_p_perm"], "{:.4g}")
    return (f"R+ {vals[0]}{unit} → {vals[1]}{unit} (MWU p {P(r['R+ L vs E p_MWU'])}, n = {ns[0]} / {ns[1]} sessions), "
            f"R− {vals[2]}{unit} → {vals[3]}{unit} (p {P(r['R- L vs E p_MWU'])}, n = {ns[2]} / {ns[3]}); experts R+ vs R− "
            f"p {P(r['expert R+ vs R- p_MWU'])}; learning × cohort interaction p {P(r['interaction_p_perm'])}")


def gi(panel, df=None):
    r = row(ST[POP] if df is None else df, panel=panel)
    return f"p {P(r['interaction_p_perm'])}"


def sizes():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"]) & W.quality_label.isin(["good", "mua"])]
    fr = W["fr_window:whisker_hit_vs_fa_prelick@all"]
    W = W[W["sig:whisker_hit_vs_fa_prelick@all"].notna() & (fr >= m51.MIN_FR)]
    W = m61.learner_filter(W)
    T = W.groupby(["cohort", "stage"]).agg(Sessions=("session_id", "nunique"), Mice=("mouse_id", "nunique"),
                                           Units=("session_id", "size")).reset_index()
    T["cohort"] = T.cohort.str.replace("-", "−")
    A = W.groupby(["area_group", "cohort", "stage"]).agg(s=("session_id", "nunique"), u=("session_id", "size")).reset_index()
    A["cell"] = A.s.astype(str) + " / " + A.u.astype(str)
    A["col"] = A.cohort.str.replace("-", "−") + " " + A.stage
    TA = A.pivot(index="area_group", columns="col", values="cell").fillna("–").reset_index()
    TA = TA.rename(columns={"area_group": "Area group"})[["Area group"] + [c for c in G4 if c in TA]]
    return T.rename(columns={"cohort": "Cohort", "stage": "Stage"}), TA, W


# ---------------------------------------------------------------------------------------------------------- anchors
R.register([
    ("fig-task", "Figure 1"), ("fig-single", "Figure 2"), ("fig-distance", "Figure 3"), ("fig-areas", "Figure 4"),
    ("fig-decoders", "Figure 5"), ("fig-timeline", "Figure 6"), ("fig-halves", "Figure 7"), ("fig-slopes", "Figure 8"),
    ("fig-epochs", "Figure 9"),
    ("fig-s-converging", "Figure S1"), ("fig-s-abssel", "Figure S2"), ("fig-s-lambda", "Figure S3"), ("fig-s-angles", "Figure S4"),
    ("fig-s-norm", "Figure S5"), ("fig-s-shift", "Figure S6"), ("fig-s-cd", "Figure S7"), ("fig-s-quiet", "Figure S8"),
    ("fig-s-perf", "Figure S9"), ("fig-s-single-all", "Figure S10"), ("fig-s-distance-all", "Figure S11"),
    ("fig-s-decoders-all", "Figure S12"), ("fig-s-oddeven", "Figure S13"), ("fig-s-controls", "Figure S14"),
    ("fig-s-halves-all", "Figure S15"), ("fig-s-slopes-all", "Figure S16"), ("fig-s-epochs-all", "Figure S17"),
    ("fig-s-timeline-exp", "Figure S18"), ("fig-s-decschemes", "Figure S19"),
    ("tbl-mixed", "Table 1"), ("tbl-halves", "Table 2"), ("tbl-epochs", "Table 3"),
    ("tbl-s-sizes", "Table S1"), ("tbl-s-sizes-areas", "Table S2"), ("tbl-s-slopes", "Table S3"),
    ("tbl-a-params", "Table A1"), ("tbl-a-files", "Table A2")])


# ---------------------------------------------------------------------------------------------------------- sections
def front(T):
    tot = T[["Sessions", "Mice", "Units"]].sum()
    mm = MM.set_index(["stage", "reference"])
    a = mm.loc[("learning", "spontaneous licks")]
    return f"""---
title: "Pre-lick convergence: whisker-triggered licks come to resemble rewarded auditory licks, across and within days"
author: "Axel Bisi (analysis with Claude Code)"
date: "{date.today().isoformat()}"
---

**Data version.** Kilosort 4 (NWB_ks4), v2 unit table, good + mua units with mean pre-lick rate ≥ {m51.MIN_FR} Hz;
SSL whisker-training sessions, R+ and R− cohorts (cohort per mouse from the reference sheet), learning day (day 0) and
expert days; learners population (non-learner mice lose only their day-0 session). Part I: {num('n_sessions', int(tot.Sessions))}
sessions, {num('n_units', int(tot.Units))} tested units ({ref('tbl-s-sizes')}, {ref('tbl-s-sizes-areas')}). Reference for
unrewarded licks: spontaneous licks (SL). Results produced with the current analysis settings; pending changes are listed in
the caveats.

# Abstract

In the SSL task, licks after a whisker stimulus are rewarded in R+ mice and unrewarded in R− mice, while licks after an
auditory tone are always rewarded. We asked whether the neural activity that precedes a whisker-triggered lick (whisker
hit, WH) comes to resemble the activity preceding a rewarded auditory lick (auditory hit, AH) rather than an unrewarded
spontaneous lick (SL), and whether this depends on the reward contingency.

Across days (Part I), in R+ but not R− mice, more neurons distinguished whisker hits from spontaneous licks, their
selectivity aligned with that of auditory hits, and the population vector of whisker hits moved away from spontaneous
licks toward auditory hits. Decoders trained on auditory hits vs spontaneous licks read whisker hits as more auditory-like
in R+ experts.

Within days (Part II), the change was already visible during the learning session: on a single-trial coding direction,
R+ whisker hits drifted toward auditory hits by {num('mm_rplus_d0', a.drift_Rplus, '{:+.2f}')} of the SL → AH distance over
one session, R− whisker hits did not ({num('mm_rminus_d0', a.drift_Rminus, '{:+.2f}')}), and the drift continued within
expert sessions.

# Key results

1. **Single neurons.** WH vs SL selective units: {grp('2b', 'k1')} ({ref('fig-single')}b).
2. **Population distance.** Δd = d(WH, SL) − d(WH, AH): {grp('3c', 'k2')} ({ref('fig-distance')}c).
3. **Where the change lies.** The whisker-hit vector grew rather than rotated: length ratio interaction
   {gi('S-ang ratio', SS)}, cos θ interaction {gi('S-ang cos', SS)} ({ref('fig-s-angles')}).
4. **Decoders.** P(AH | WH) − P(AH | SL), chance-corrected: {grp('5e', 'k4')} ({ref('fig-decoders')}e).
5. **Within days.** Mixed model of single-trial coding-direction scores ({ref('tbl-mixed')}, {ref('fig-timeline')}):
   day-0 drift R+ {a.drift_Rplus:+.2f} ± {a.se_Rplus:.2f} (p {pv('mm_p_rplus_d0', a.p_Rplus)}), R− {a.drift_Rminus:+.2f} ± {a.se_Rminus:.2f}
   (p {pv('mm_p_rminus_d0', a.p_Rminus)}); R+ − R− mouse-permutation p {pv('mm_pperm_d0', a.p_perm_diff)}.
"""


def intro():
    return """
# Introduction

Mice in the SSL task lick to report a brief whisker deflection or an auditory tone. The two cohorts differ only in the
outcome of whisker-triggered licks: rewarded in R+ mice, unrewarded in R− mice. Auditory licks are rewarded in both and
spontaneous licks between trials are never rewarded. Because the motor act is the same lick, comparing the activity in the
100 ms before the first lick across these event types isolates what the brain represents about the upcoming lick beyond
the movement itself: the sensory trigger, the expected outcome, or both.

The hypothesis is that reward expectation is part of the pre-lick representation. If so, whisker-triggered licks of R+
mice, which become rewarded, should come to resemble auditory licks (rewarded), while those of R− mice should stay like
spontaneous licks (unrewarded). The report tests this at the level of single neurons, population geometry and decoders
(Part I, learning day vs expert sessions), and then asks when the change happens: already within the learning session or
only between days, and whether it continues within expert sessions (Part II).
"""


def methods():
    return f"""
# Methods

## Data, events and units

**Sessions and cohorts.** All whisker-training sessions with KS4 NWB files; learning = day 0, expert = later days. Main
population: learners (a mouse whose learning-day category is not "learner" contributes only its expert sessions); the
all-mice population is shown as a control ({ref('fig-s-single-all')}, {ref('fig-s-distance-all')},
{ref('fig-s-decoders-all')}; Part II: {ref('fig-s-halves-all')}, {ref('fig-s-slopes-all')}, {ref('fig-s-epochs-all')}). Sample sizes per
cohort, stage and area group: {ref('tbl-s-sizes')}, {ref('tbl-s-sizes-areas')}.

**Trials.** Active context (per-trial rule: a trial in a fixed ~3 s inter-trial sequence is passive), perf ≠ 6, auditory
warm-up cut (one trial before the first whisker trial kept), disengagement tail trimmed (rule A1: trials after the last lick
dropped when that tail holds ≥ 5 whisker and ≥ 1 auditory trials).

**Events.** Whisker hit (WH) and auditory hit (AH): first lick of a licked whisker / auditory trial; the NWB lick time is
late by the artefact window, so the corrected first lick is $t_{{\\mathrm{{lick}}}} = t_{{\\mathrm{{start}}}} +
(t_{{\\mathrm{{lick,NWB}}}} - t_{{\\mathrm{{rw,start}}}})$. Spontaneous lick (SL): a piezo lick ≥ 1 s after the previous piezo
lick (bout onset), outside every trial window $[t_{{\\mathrm{{start}}}} - 0.2\\,\\mathrm{{s}},\\ t_{{\\mathrm{{rw,stop}}}} +
0.5\\,\\mathrm{{s}}]$, within the analysed epoch.

**Rates.** Pre-lick rate of unit $u$ for event $i$: spikes in $[t_i - 100\\,\\mathrm{{ms}}, t_i)$ / 0.1 s, minus a baseline
(hits: [−1.0, −0.015] s before trial start; SL: [−1.0, −0.5] s before the lick). Spike trains are whisker-artefact corrected.
Units: quality good or mua, mean raw pre-lick rate ≥ {m51.MIN_FR} Hz over the analysed events.

## Single-neuron selectivity

For two event classes A and B, the ROC area under the curve of a unit's pre-lick rates is
$\\mathrm{{AUC}} = P(r_B > r_A) + \\tfrac12 P(r_B = r_A)$ and its selectivity $s = 2\\,\\mathrm{{AUC}} - 1 \\in [-1, 1]$
(positive: B higher). Significance: 1,000 label permutations, one-tailed on the side of the observed $s$, p < 0.05.
Comparisons: WH vs SL, AH vs SL, WH vs AH.

- **Fraction selective**: significant units / tested units per session (sessions with ≥ 10 tested units).
- **Converging neurons**: among reward-lick neurons (AH vs SL significant), the fraction whose WH vs SL selectivity is also
  significant with the same sign.
- **AH-likeness** of a reward-lick neuron with class means $m_W, m_A, m_S$:
  $c = (|m_W - m_S| - |m_W - m_A|) / |m_A - m_S| \\in [-1, 1]$ by the triangle inequality ($-1$: WH like SL, $+1$: like AH).
- **Shared hit code**: Spearman correlation across a session's units of $s_{{AH\\,vs\\,SL}}$ and $s_{{WH\\,vs\\,SL}}$. Pooled
  over neurons, Pearson $r$ per stage with a session-bootstrap test of $\\Delta r$ ({ref('fig-single')}h–i).
- **Absolute selectivity** ({ref('fig-s-abssel')}): mean $|s|$ over all tested units, the fraction significant with $s > 0$
  and with $s < 0$, and mean $|s|$ over significant units.

## Population geometry

**Unit scaling.** Per session, each unit's event rates are z-scored over the analysed events (mean and SD over all events
pooled). {ref('fig-s-norm')} compares this with raw spikes/s and with the within-class SD
$\\sigma_u = \\sqrt{{\\tfrac13 \\sum_{{c}} \\mathrm{{Var}}_c(r_u)}}$ (diagonal noise normalisation).

**Cross-validated squared distance per unit.** For classes X and Y, the events of each class are split at random into
halves a and b ($n_s = 50$ splits):
$$d(X, Y) = \\frac{{1}}{{N}} \\,\\big\\langle (\\bar x_a - \\bar y_a) \\cdot (\\bar x_b - \\bar y_b) \\big\\rangle_{{\\mathrm{{splits}}}},$$
with $N$ the number of units. Because the noise of halves a and b is independent, $E[d] = \\|\\mu_X - \\mu_Y\\|^2 / N$: the
squared distance between the true class means per unit, without the positive noise bias of $\\|\\bar x - \\bar y\\|^2$ (which
adds $\\mathrm{{tr}}(\\Sigma)(1/n_X + 1/n_Y)$). Single estimates can be negative. With z-scored units, $d$ is the mean squared
effect size per unit. Distance difference: $\\Delta d = d(WH, SL) - d(WH, AH)$ (> 0: WH nearer AH).

**Along and orthogonal components.** With the reference $S$ at the origin, the polarisation identity gives the
cross-validated dot product
$$a = (\\mu_W - \\mu_S)\\cdot(\\mu_A - \\mu_S) / N = \\tfrac12\\,[d(W,S) + d(A,S) - d(W,A)]$$
({ref('fig-distance')}i), and the squared orthogonal part $o^2 = d(W,S) - a^2/d(A,S)$ ({ref('fig-distance')}j). The group
triangles ({ref('fig-distance')}b, real group-mean distances) place SL at (0, 0), AH at $(\\sqrt{{d(A,S)}}, 0)$ and WH at
$(x, y)$ with $x = [d(W,S) - d(W,A) + d(A,S)] / (2\\sqrt{{d(A,S)}})$, $y = \\sqrt{{d(W,S) - x^2}}$.

**λ.** The position of the whisker-hit mean on the SL → AH axis:
$$\\lambda = \\frac{{(\\mu_W - \\mu_S)\\cdot(\\mu_A - \\mu_S)}}{{\\|\\mu_A - \\mu_S\\|^2}} = \\frac{{a}}{{d(A,S)}},$$
estimated with the same split halves (numerator and denominator each cross-validated), so $\\lambda = 0$: WH like SL,
$\\lambda = 1$: like AH; WH ≈ $(1-\\lambda)\\,S + \\lambda\\,A$ along the axis. Sessions need $d(A,S) \\ge 0.01$; λ clipped to [−1, 2].

**Angle and length** ({ref('fig-s-angles')}). $\\lambda = \\rho \\cos\\theta$ with
$\\cos\\theta = a / \\sqrt{{d(W,S)\\,d(A,S)}}$ (angle between WH − SL and AH − SL) and the length ratio
$\\rho = \\sqrt{{d(W,S)/d(A,S)}} = \\|\\mu_W - \\mu_S\\| / \\|\\mu_A - \\mu_S\\|$. The same λ can arise from a long vector at a wide
angle or a short one at a narrow angle, and λ > 1 is possible when $\\rho \\cos\\theta > 1$; the two factors are reported
separately.

**λ_LDA and axis quality** ({ref('fig-s-lambda')}). A shrinkage LDA (Ledoit–Wolf) separates AH from SL on training folds
(5-fold, units z-scored on training folds); held-out AH and SL and all WH events are projected and normalised so that the
training SL mean is 0 and the AH mean 1; λ_LDA is the mean WH projection. LDA weights each unit by the inverse within-class
covariance, so it is the noise-whitened counterpart of λ. Axis quality: held-out $d' = (\\bar p_{{AH}} - \\bar p_{{SL}})/\\sigma_p$
(pooled SD of the held-out projections), sessions with $d' \\ge 0.3$. The second axis of the example planes is the WH off-axis
direction, defined on half the WH events and shown for the other half.

**Coding-direction estimators** ({ref('fig-s-cd')}). Part II needs a score per event. (i) Unified split-half score: in each
split, the axis $w = \\bar a_A - \\bar a_S$ comes from one half and every event of the other half is scored
$(x_i - \\bar b_S)\\cdot w$; scores are averaged over the splits in which the event was held out and divided by
$D = \\langle (\\bar b_A - \\bar b_S)\\cdot w \\rangle$. The mean WH score equals λ (same numerator and denominator).
(ii) Part II 5-fold coding direction: unit axis from the training folds, held-out projections, one normalisation per session
(held-out SL = 0, AH = 1), sessions with held-out $d' \\ge 0.3$.

**Linear-shift chance** ({ref('fig-s-shift')}). Event labels in time order are shifted against the activity by $k$ events
($|k| = 5 \\dots n/3$, 40 shifts, no wrap-around), which keeps the temporal autocorrelation of event types and of neural
drift but breaks the event–activity pairing. Corrected value = observed − median over shifts (for ratios, numerator and
denominator corrected separately).

**Quiet windows** ({ref('fig-s-quiet')}). Raw rates in [−200, −100] ms before every trial start of the analysed epoch,
lick-free (no lick in the preceding 1 s), scaled with the event parameters and placed on the same readouts.

## Decoders

L2 logistic regression ($C = 0.05$, balanced classes, units z-scored on training folds), trained on AH vs SL with stratified
$k$-fold cross-validation ($k = \\min(5, n_{{\\min}})$); WH events are never used for training and are scored by every fold's
decoder. Readouts: balanced accuracy on held-out AH / SL; transfer (yes/no) $= (\\mathrm{{FR}}_{{WH}} - \\mathrm{{FPR}}) /
(\\mathrm{{TPR}} - \\mathrm{{FPR}})$ with $\\mathrm{{FR}}_{{WH}}$ the fraction of WH decoded as AH; transfer (probability), the same with
mean $P(AH)$; and $P(AH|WH) - P(AH|SL)$. Chance: linear shifts (above). Pseudo-populations: hierarchical bootstrap (mice →
sessions → neurons → events) of $M$ neurons, same decoder and readouts ({ref('fig-decoders')}f–m).

## Statistics

Unit of analysis: session. Learning vs expert per cohort and R+ vs R− per stage: Mann-Whitney U (Welch t-test in the stats
tables). Learning × cohort interaction $[\\bar x_{{E}} - \\bar x_{{L}}]_{{R+}} - [\\bar x_{{E}} - \\bar x_{{L}}]_{{R-}}$ by 10,000
permutations of cohort labels across mice (sessions of a mouse move together). Correlations with behaviour: Spearman, OLS
with 95 % CI band ({ref('fig-s-perf')}).

## Part II: within-session analyses

**Halves.** Each session is split at the midpoint between its two middle auditory hits (equal AH per half); per half the
events are count-matched by subsampling (20 subsamples) and Δd, λ and decoder readouts recomputed; change = late − early.
Null split: odd vs even events ({ref('fig-s-oddeven')}).

**Trial-level trajectories.** Every event gets the session's 5-fold coding-direction score and decoder $P(AH)$; per class,
OLS of the scores on normalised time $\\tau = (t - t_{{\\mathrm{{first}}}})/(t_{{\\mathrm{{last}}}} - t_{{\\mathrm{{first}}}})$; the
WH − SL slope removes drift shared by all events.

**Epochs on a common footing.** Day-0 and expert halves each estimated from exactly 6 events per class (20 subsamples) and
150 units; one decoder per session read out per half; hierarchical bootstrap over mice then sessions; contrasts: within
day 0 (late − early), across days (expert − day 0, same phase) and carry-over (expert early − day-0 late).

**Mixed model.** Single events: $c_i \\sim \\mathrm{{WH}}_i \\times \\tau_i \\times \\mathrm{{R+}}$ with random intercept, time
slope and WH offset per session (experts: plus a mouse intercept); the WH × τ coefficient is the drift of WH relative to the
reference over one session in units of the SL → AH distance; R+ vs R− also by 500 cohort-label permutations across mice.
"""


def part1(W):
    r3S = row(ST[POP], panel="3S i")
    nsn = NS[(NS.population == POP) & (NS.metric == "dd")].set_index("norm")
    sp = lambda k: num(f"norm_rho_{k}", nsn.loc[f"spearman pooled vs {k}", "interaction"], "{:.2f}")
    return f"""
# Part I: across days

The task, the alignment and the data are summarised in {ref('fig-task')}.

{figure('fig-task', P1 / 'Fig1_task_data.png', f"Task and data. (a) Task and reward contingencies: auditory licks rewarded in both cohorts, whisker licks only in R+; spontaneous licks never rewarded. (b) Alignment: the 100 ms before the corrected first lick (shaded); hit baseline 1 s before trial start. (c) Reaction-time distributions of whisker and auditory hits (pooled trials; legend: medians). (d) Median reaction time per session, mean ± s.e.m. over sessions, by cohort and stage. (e) Sessions per cohort and stage, with mice and tested units. (f) All recorded units in Allen CCF coordinates (sagittal and coronal projections; random subset; R+ green, R− magenta). Population: {POP}.")}

## Single neurons

More neurons separated whisker hits from spontaneous licks in R+ experts than on the learning day, and not in R−:
{grp('2b', 'fracWH')} ({ref('fig-single')}b).

The fraction of neurons separating whisker hits from auditory hits changed little: {grp('2c', 'fracWHAH')} ({ref('fig-single')}c).

Among reward-lick neurons, the fraction that also separated whisker hits from spontaneous licks with the same sign
(converging neurons): {grp('2d', 'conv')} ({ref('fig-single')}d). Their AH-likeness: {grp('2e', 'like')} ({ref('fig-single')}e).

The shared hit code, the correlation across units of AH vs SL and WH vs SL selectivities, rose in R+:
{grp('2g', 'shared')} ({ref('fig-single')}g); pooled over neurons the correlation increased in R+ and not in R−
({ref('fig-single')}h–i). Converging neurons were found across areas ({ref('fig-single')}j); their examples, functional
types and anatomical distribution are in {ref('fig-s-converging')}.

The increase in WH vs SL selective units in R+ is carried by positively selective units (WH > SL; interaction
{gi('S-sel pos whisker_hit_vs_fa_prelick', SS)}); mean absolute selectivity over all units changed less
({ref('fig-s-abssel')}).

{figure('fig-single', P1 / 'Fig2_single_neurons.png', f"Single neurons. (a) Example neurons (R+ expert top, R− expert bottom): converging (AH vs SL and WH vs SL significant, same sign), reward-lick-only (AH vs SL significant, |WH vs SL| < 0.1) and whisker-hit-only; PSTH (10-ms bins, Gaussian σ = 20 ms, mean ± s.e.m. over events) above rasters (≤ 25 events per type); titles: area and selectivity s = 2·AUC − 1. (b–e) Schematics (top) and per-session values (bottom; open: learning, filled: expert; large symbol: mean ± s.e.m.): (b) fraction of tested units with significant WH vs SL ROC (1,000 label permutations, p < 0.05); (c) fraction with significant WH vs AH ROC; (d) converging neurons as a fraction of reward-lick neurons; (e) AH-likeness c of reward-lick neurons. (f) Schematic of the shared hit code; (g) Spearman r across units per session; (h, i) selectivities of all tested units (log density) with OLS fits per stage (learning dashed, expert solid), Pearson r and the session-bootstrap Δr. (j) Converging neurons per area, expert sessions, mean ± s.e.m. over sessions; ANOVA with cohort labels permuted across mice. Brackets: MWU p; titles: learning × cohort interaction, mouse-level permutation. Sessions with ≥ 10 tested units; {POP}.")}

## Population geometry

In R+, whisker hits moved away from spontaneous licks and toward auditory hits: Δd {grp('3c', 'dd')}
({ref('fig-distance')}c). The change came from d(WH, SL), which increased in R+ ({grp('3d', 'dws')}), while d(WH, AH)
({gi('3e')}) and the axis length d(AH, SL) ({gi('3f')}) did not change differently between cohorts ({ref('fig-distance')}d–f).

The whisker-hit displacement grew both along the SL → AH direction ({grp('3i', 'along')}) and orthogonal to it
({grp('3j', 'ortho')}) ({ref('fig-distance')}i–j). The along component divided by the axis length is λ, which increased in
R+ only ({grp('3S d', 'lam')}; {ref('fig-s-lambda')}); λ and λ_LDA agreed (Spearman r = {num('lam_vs_lda', r3S.r, '{:.2f}')},
n = {int(r3S.n)} sessions).

Decomposing λ = ρ cos θ ({ref('fig-s-angles')}): the angle between the whisker-hit and auditory-hit displacements did not
change differently between cohorts (cos θ interaction {gi('S-ang cos', SS)}), while the length ratio did ({gi('S-ang ratio', SS)}):
in R+ the whisker-hit vector grew, at a roughly constant angle.

These results did not depend on the unit scaling (session values of Δd vs the current scaling: within-class SD ρ =
{sp('within')}, raw rates ρ = {sp('raw')}; {ref('fig-s-norm')}), survived the linear-shift correction (Δd interaction
observed {gi('S-shift dd', S78)}, corrected {gi('S-shift dd_corrected', S78)}; {ref('fig-s-shift')}), and the two
coding-direction estimators agreed with λ ({ref('fig-s-cd')}). Lick-free quiet windows sat near the spontaneous-lick end of the
axis and did not show the whisker-hit change ({ref('fig-s-quiet')}).

{figure('fig-distance', P1 / 'Fig3_population_distance.png', f"Population distance. (a) Schematic: each event type is a mean pre-lick population vector; d = cross-validated squared Euclidean distance per unit (50 random split halves); Δd = d(WH, SL) − d(WH, AH). (b) Group-mean geometry, real data: SL at the origin, AH on the x-axis at √d(AH, SL), WH placed from its two distances; axes in √(squared distance per unit); learning dashed / open, expert solid / filled. (c–f) Per session: Δd, d(WH, SL), d(WH, AH), d(AH, SL). (g) RT matching is not defined for spontaneous licks. (h) Cumulative distributions of Δd. (i) Cross-validated dot product (WH − SL)·(AH − SL) per unit = λ·d(AH, SL). (j) Orthogonal squared distance per unit. Statistics as {R.label('fig-single')}; sessions with ≥ 5 units and ≥ 4 events per class; {POP}.")}

## Brain areas

Δd computed per session × area group changed in R+ mainly in motor, frontal, retrosplenial, insular and hippocampal groups,
but no area group survived family-wise correction ({ref('fig-areas')}a–c); lick-aligned PSTHs of two groups illustrate the
three event types per neuron and per session ({ref('fig-areas')}d–e).

{figure('fig-areas', P1 / 'Fig4_areas.png', "Brain areas. (a) Change of Δd from learning to expert per area group (mean ± s.e.m.; filled: MWU p < 0.05), area groups with ≥ 3 sessions at both stages in a cohort; right: interaction p (mouse permutation) and family-wise p (max-|z|). Inset: ANOVA Δd ~ cohort × stage × area, F test and cohort-label permutation across mice. (b) Mean Δd per area group, cohort and stage. (c) Δd per session for four area groups. (d) Lick-aligned PSTHs, rate − baseline (spikes/s), mean ± s.e.m. over units (expert sessions). (e) The same, averaged over units within session then mean ± s.e.m. over sessions. Baselines: hits 1 s before trial start, SL [−1, −0.5] s before the lick; shaded: pre-lick window.")}

## Decoders

Decoders separated auditory hits from spontaneous licks well above chance in every group ({ref('fig-decoders')}a–b; raw
accuracy {grp('5a', 'bacc_raw')}). Applied to whisker hits, they read R+ expert whisker hits as more auditory-like:
yes/no transfer {grp('5c', 'tb')}; P(AH|WH) − P(AH|SL) {grp('5e', 'num')} ({ref('fig-decoders')}c–e). Pseudo-populations
of up to 2,000 neurons gave the same direction ({ref('fig-decoders')}f–m).

{figure('fig-decoders', P1 / 'Fig5_decoders.png', "Decoders. (a) Raw balanced accuracy of the single-session AH vs SL decoder (dots: sessions) with the median (line) and 2.5–97.5 % range (band) of each group's linear-shift chance level. (b) Accuracy above chance; (c) WH decoded as AH (yes/no) and (d) (probability), as transfer (0 = like SL, 1 = like AH); (e) P(AH|WH) − P(AH|SL); all chance-corrected by 40 linear shifts. (f–i) Hierarchical pseudo-populations (mice → sessions → neurons → events) at M = 2,000 neurons: bootstrap distributions and 95 % CI. (j–l) Readouts vs M. (m) Learning → expert change at M = 2,000 with bootstrap CI; top: interaction p.")}

## Behaviour

Across sessions, most single-neuron and geometry measures tracked whisker hit rate and d′, strongly in R+ ({ref('fig-s-perf')}).
"""


def mm_table():
    T = MM.copy()
    out = pd.DataFrame({
        "Stage": T.stage, "Reference": T.reference, "Sessions / mice": T.n_sessions.astype(str) + " / " + T.n_mice.astype(str),
        "WH events": T.n_wh,
        "R+ drift": [f"{a:+.2f} ± {b:.2f}" for a, b in zip(T.drift_Rplus, T.se_Rplus)], "p R+": T.p_Rplus.map(lambda v: P(v)[2:] if P(v).startswith("=") else P(v)),
        "R− drift": [f"{a:+.2f} ± {b:.2f}" for a, b in zip(T.drift_Rminus, T.se_Rminus)], "p R−": T.p_Rminus.map(lambda v: P(v)[2:] if P(v).startswith("=") else P(v)),
        "R+ − R−": [f"{a:+.2f} ± {b:.2f}" for a, b in zip(T.diff_Rplus_minus_Rminus, T.se_diff)],
        "p perm": T.p_perm_diff.map(lambda v: P(v)[2:] if P(v).startswith("=") else P(v))})
    return out


def pp(v):
    s = P(v)
    return s[2:] if s.startswith("=") else s


def halves_table():
    lab = {"dd": "Δd", "lam": "λ", "ws_num_c": "Decoder P(AH|WH) − P(AH|SL), whole session", "xh_num_c": "Decoder, cross-half",
           "ws_bacc_c": "Accuracy − chance, whole session"}
    rows = []
    for m, l in lab.items():
        r = row(HV, split="time", measure=m)
        d = {"Measure": l}
        for g, gl in (("R+ learning", "R+ day 0"), ("R+ expert", "R+ expert"), ("R- learning", "R− day 0"), ("R- expert", "R− expert")):
            d[gl] = f"{r[f'mean Δ {g}']:+.3f} (p {pp(r[f'p_wilcoxon {g}'])})"
        d["Day 0 R+ vs R− (perm)"] = pp(r["day0 p_perm"]); d["Cohort × stage (perm)"] = pp(r["cxsxh p_perm"])
        rows.append(d)
    return pd.DataFrame(rows)


def slopes_table():
    lab = {"md_WH_slope": "WH score slope", "md_WH-FA_slope": "WH − SL slope", "dec_WH_slope": "Decoder P(AH|WH) slope",
           "dec_WH-FA_slope": "Decoder WH − SL slope"}
    rows = []
    for m, l in lab.items():
        r = row(SL, measure=m)
        d = {"Measure": l}
        for g, gl in (("R+ learning", "R+ day 0"), ("R+ expert", "R+ expert"), ("R- learning", "R− day 0"), ("R- expert", "R− expert")):
            d[gl] = f"{r[f'mean {g}']:+.3f} (p {pp(r[f'p_wilcoxon {g}'])})"
        d["Day 0 R+ vs R− (perm)"] = pp(r["day0 p_perm"]); d["Cohort × stage (perm)"] = pp(r["cxs p_perm"])
        rows.append(d)
    return pd.DataFrame(rows)


def epochs_table():
    lab = {"pos": "WH − SL position on the CD", "dec": "Decoder P(AH|WH) − P(AH|SL) − chance", "ddn": "Δd / d(AH, SL)"}
    rows = []
    for m, l in lab.items():
        for c in ("within-day", "across-day (early)", "carry-over"):
            d = {"Measure": l, "Contrast": c}
            for coh in ("R+", "R-"):
                r = row(EP, measure=m, cohort=coh, kind="contrast", name=c)
                d[coh.replace("-", "−")] = f"{r.value:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]"
            r = row(EP, measure=m, cohort="R+ - R-", kind="contrast", name=c)
            d["R+ vs R− (perm)"] = pp(r.p_perm)
            rows.append(d)
    return pd.DataFrame(rows)


def part2():
    mm = MM.set_index(["stage", "reference"])
    a, e = mm.loc[("learning", "spontaneous licks")], mm.loc[("expert", "spontaneous licks")]
    hv = row(HV, split="time", measure="dd")
    s1 = row(SL, measure="md_WH-FA_slope")
    w = lambda coh, name, m="pos": row(EP, measure=m, cohort=coh, kind="contrast", name=name)
    return f"""
# Part II: within days

The across-day change raises the question of when it happens. Each session was analysed along its own time axis, on the
learning day and in expert sessions, with three complementary approaches: early vs late halves, trial-level trajectories
on a fixed session axis, and a single-trial mixed model; a common-footing comparison puts within-day and across-day
changes on the same scale ({ref('fig-timeline')}; methods with examples and controls in {ref('fig-s-timeline-exp')};
decoder results across unit-sampling schemes in {ref('fig-s-decschemes')}).

{figure('fig-timeline', WD / 'cosyne' / 'COSYNE_convergence_timeline.png', "Convergence timeline (all mice). (a) First-lick-aligned population PSTHs of R+ (top) and R− (bottom) mice: day-0 early half, day-0 late half and expert sessions; rate − baseline, mean over units within session, then mean ± s.e.m. over sessions (10-ms bins, 30-ms boxcar); shaded: pre-lick window. (b) Day 0: mean WH − SL score on the session's cross-validated SL → AH coding direction (SL = 0, AH = 1) in five bins of normalised session time, mean ± s.e.m. over sessions; right: expert session mean. (c) Per-session WH − SL slope on day 0 (OLS); title: mixed-model R+ vs R− p. (d) Common footing: decoder readout P(AH|WH) − P(AH|SL) − chance per half from exactly 4 events per class and 150 units, mean over mice and 95 % hierarchical-bootstrap CI. (e) Within day 0, across days and carry-over changes of (d), 95 % CI (filled: p < 0.05); top: R+ vs R− by cohort-label permutation across mice.")}

## Learning day and expert sessions: mixed model

On single events, whisker hits drifted toward auditory hits relative to spontaneous licks over the learning session in R+
({a.drift_Rplus:+.2f} ± {a.se_Rplus:.2f} of the SL → AH distance, p {P(a.p_Rplus)}) but not in R− ({a.drift_Rminus:+.2f} ±
{a.se_Rminus:.2f}, p {P(a.p_Rminus)}); R+ − R− mouse-permutation p {P(a.p_perm_diff)}. Within expert sessions the drift
continued in R+ ({num('mm_rplus_e', e.drift_Rplus, '{:+.2f}')}, p {pv('mm_p_rplus_e', e.p_Rplus)}) and reversed in R−
({num('mm_rminus_e', e.drift_Rminus, '{:+.2f}')}, p {pv('mm_p_rminus_e', e.p_Rminus)}); R+ − R− p {pv('mm_pperm_e', e.p_perm_diff)}
({ref('tbl-mixed')}). With auditory hits as the reference, R+ and R− diverged on both the learning day and in experts.

{table('tbl-mixed', mm_table(), "Single-trial mixed model of coding-direction scores (5-fold CD; SL = 0, AH = 1): WH × time coefficient = drift of whisker hits relative to the reference over one session (normalised time 0 → 1), estimate ± s.e. and Wald p per cohort; R+ − R− with cohort-label permutation across mice (500 permutations). Learning day and expert sessions; reference: spontaneous licks or auditory hits.")}

## Session halves

Splitting each session at its middle auditory hit, Δd increased from the early to the late half on the learning day in R+
(mean Δ {num('hv_dd_rplus', hv['mean Δ R+ learning'], '{:+.3f}')}, Wilcoxon p {pv('hv_dd_p_rplus', hv['p_wilcoxon R+ learning'])})
and not in R− (day-0 R+ vs R− permutation p {pv('hv_dd_perm', hv['day0 p_perm'])}); the same held for λ and the
whole-session decoder readout ({ref('tbl-halves')}, {ref('fig-halves')}). In expert sessions the halves changed less. An
odd/even split, which removes time, shows no change ({ref('fig-s-oddeven')}); event rates, reaction times and firing rates
per half are in {ref('fig-s-controls')}.

{figure('fig-halves', WD / 'halves' / POP / 'within_session_time.png', f"Session halves, learning day and expert sessions ({POP}). Each session split at the midpoint between its two middle auditory hits; events count-matched per half (20 subsamples). (a–f) Early (open) and late (filled) half per group, lines join the halves of a session: Δd, λ, decoder P(AH|WH) − P(AH|SL) − chance (whole-session decoder; cross-half decoder: trained on one half, read out on the other), accuracy − chance. (g–l) Late − early per session; titles: day-0 R+ vs R− (mouse permutation) and cohort × stage × half interaction.")}

{table('tbl-halves', halves_table(), f"Late − early change per group (mean; Wilcoxon p vs 0), time split, {POP}; day 0 R+ vs R− and cohort × stage by mouse-level permutation.")}

## Trial-level trajectories

On a fixed session axis, the WH − SL slope was positive on day 0 in R+ and differed from R− (day-0 permutation p
{pv('sl_perm', s1['day0 p_perm'])}; {ref('tbl-s-slopes')}, {ref('fig-slopes')}e–h). Fitted at the start of the learning
session, R+ whisker hits were far from their expert position; by the end they approached it ({ref('fig-slopes')}i). Spontaneous
lick rates fell during sessions while whisker hit rates rose in R+ ({ref('fig-slopes')}k–l), which is why the slopes are taken
relative to spontaneous licks measured at the same times.

{figure('fig-slopes', WD / 'slopes' / POP / 'trial_slopes.png', f"Trial-level trajectories ({POP}). (a–d) Scores on the session's SL → AH axis (SL = 0, AH = 1) in five bins of normalised session time per class, mean ± s.e.m. over sessions; dashed band: expert WH level. (e–h) Per-session OLS slopes against normalised time: WH, WH − SL, decoder P(AH|WH) and decoder WH − SL. (i, j) Day-0 fitted WH score at the start and end of the session vs the expert session mean, per cohort. (k, l) Whisker hit and spontaneous lick rates (events / min) over the session.")}

## Within days vs across days

On a common footing (same events per class and units in every half), the across-day change in R+ was larger than the
within-day change (WH − SL position: within day 0 {num('ep_w_rp', w('R+', 'within-day').value, '{:+.2f}')}, across days
{num('ep_a_rp', w('R+', 'across-day (early)').value, '{:+.2f}')}), and part of it was already present at the start of expert
sessions (carry-over {num('ep_c_rp', w('R+', 'carry-over').value, '{:+.2f}')}) ({ref('tbl-epochs')}, {ref('fig-epochs')}).

{figure('fig-epochs', WD / 'epochs' / POP / 'epoch_comparison.png', f"Within-day vs across-day change on a common footing ({POP}): phase-matched halves, 6 events per class, 150 units. (a–c) Epoch means with 95 % hierarchical-bootstrap CI (mice, then sessions): WH − SL position on the session's coding direction, decoder P(AH|WH) − P(AH|SL) − chance, and Δd / d(AH, SL). (d–f) Contrasts (filled: bootstrap p < 0.05): within day 0, across days (early and late phase) and carry-over; top: R+ vs R− by mouse-level permutation.")}

{table('tbl-epochs', epochs_table(), f"Epoch contrasts on a common footing ({POP}): estimate and 95 % hierarchical-bootstrap CI per cohort; R+ vs R− by cohort-label permutation across mice.")}
"""


def discussion():
    return """
# Discussion

Across days, the pre-lick activity of whisker-triggered licks in R+ mice moved away from that of unrewarded spontaneous
licks, partly toward that of rewarded auditory licks, at the level of single neurons, population distances and decoders. R−
mice, whose whisker licks are not rewarded, did not show this change. The shared lick movement does not explain the
difference: all three event types are first licks, compared in the same 100-ms window.

The geometry says more about how the change happens. The angle between the whisker-hit and auditory-hit displacements from
the spontaneous-lick state did not change; the whisker-hit displacement grew. In R+ mice, learning amplifies an existing
direction of whisker-hit activity, which is partly shared with auditory hits, rather than rotating it onto the auditory-hit
direction. The orthogonal component also grew, so whisker hits do not become auditory hits: they become more distinct from
spontaneous licks, partly along the reward-lick axis.

Within days, the change starts during the learning session, in the direction of the contingency, and continues in expert
sessions; a large part of the across-day change is already present at the start of expert sessions, consistent with
consolidation between sessions.

The measures track behaviour: sessions with higher whisker hit rates show stronger convergence. This is expected if the
representation underlies the behaviour, but it also means that the difference between cohorts and stages is partly a
difference in performance.
"""


def caveats():
    return """
# Caveats and limitations

- **Spontaneous-lick reference.** The current definition excludes licks inside trial windows (no-stim trials included) but
  keeps licks 1.5–8 s after stimuli (reward collection); the decided definition (C8: false-alarm licks counted, exclusion
  [stim − 0.5 s, stim + 8 s] after every stimulus trial) is pending a rerun. In two example sessions it changed neither the
  pre-lick activity nor the per-neuron selectivity.
- **Pending settings.** Rerun pending with stable units (coverage, presence and drift tests) as the main set, within-class
  SD unit scaling and a minimum of 4 events per class everywhere; the false-alarm reference variants are not shown.
- **Performance.** Convergence measures correlate with whisker hit rate; cohort and stage differences cannot be fully
  separated from performance differences.
- **Time within a session** is confounded with satiety and engagement; the reference events measured at the same times
  control for shared drifts, not for event-specific ones.
- **Areas.** No area group survives family-wise correction; area results are descriptive.
- **Multiple measures.** No correction across panels; the measures are strongly correlated (e.g. λ and Δd).

# Open questions and next steps

1. Rerun with the pending settings (C8 reference, stable units, within-class SD, 4 events per class), then the false-alarm
   reference as a control.
2. Separate convergence from performance (e.g. performance-matched sessions or performance as a covariate).
3. Within-day analyses on the redefined reference, and the axis inclusion rule (axis length, d′ or above chance).
"""


def supp(T, TA):
    o = ["\n# Supplementary figures\n"]
    F = [("fig-s-converging", P1 / "Fig2S_converging_neurons.png", "Converging neurons. (a) Example reward-lick neurons (PSTH only) that do and do not also separate WH from SL. (b) Functional types of converging neurons, reward-lick-only neurons and all tested units in the stimulus-aligned ROC analyses; right: enrichment of converging neurons. (c, d) Density of converging neurons on coronal and sagittal slabs (expert sessions)."),
         ("fig-s-abssel", P1 / "FigS_abs_selectivity.png", f"Absolute selectivity and its sign ({POP}; one dot per session; sessions with ≥ 10 tested units). Columns: WH vs SL (+: WH > SL), WH vs AH (+: AH > WH), AH vs SL (+: AH > SL). Rows: (a–c) mean |s| over all tested units; (d–f) fraction significant with s > 0; (g–i) fraction significant with s < 0; (j–l) mean |s| over significant units. Statistics as {R.label('fig-single')}."),
         ("fig-s-lambda", P1 / "Fig3S_lambda_lambdaLDA.png", "λ and λ_LDA. (a) Schematic of λ. (b) Single-event projections on the cross-validated SL → AH axis for an example R+ and R− expert session (SL = 0, AH = 1; bars: means). (c) Pooled single-event projections per group. (d) λ per session. (e) λ vs Δd. (f) RT matching not defined for SL. (g) Cumulative distributions of λ. (h) Example sessions in the plane of the shrinkage-LDA axis (x; SL = 0, AH = 1) and the WH off-axis direction (y; defined on half of the WH events, the other half shown); arrow: mean-difference axis. (i) λ vs λ_LDA. (j) λ_LDA per session. (k) Held-out d′ of the LDA axis (AH vs SL). (m) Cumulative distributions of λ_LDA."),
         ("fig-s-angles", P1 / "FigS_angles.png", "Angle and length behind λ. (a) cos θ between WH − SL and AH − SL; (b) length ratio |WH − SL| / |AH − SL|; (c) λ = ratio × cos θ (all from the cross-validated distances; sessions with a valid axis); (d) sessions in the (cos θ, ratio) plane with iso-λ lines."),
         ("fig-s-norm", NORM / f"geometry_normalisation_{POP}.png", "Unit scaling. Δd, λ and the three distances under raw spikes/s (left), pooled-SD z-scoring (current) and within-class SD (diagonal noise normalisation), same events and splits; right: session values against the current scaling (z-scored across sessions), Spearman ρ."),
         ("fig-s-shift", P1 / "FigS_shift_geometry.png", "Linear-shift control. Δd, along component and λ observed (a–c) and corrected by the median of 40 linear shifts of the event labels against the activity (d–f; for λ, numerator and denominator corrected separately)."),
         ("fig-s-cd", P1 / "FigS_cd_estimators.png", "Coding-direction estimators. (a) λ vs the mean WH score of the unified split-half trial scores (equal by construction). (b) λ vs the mean WH score of the Part II 5-fold coding direction (sessions with held-out d′ ≥ 0.3). (c) 5-fold CD per cohort and stage."),
         ("fig-s-quiet", P1 / "FigS_quiet_windows.png", "Quiet windows. Raw rates in [−200, −100] ms before every trial start (lick-free: no lick in the preceding 1 s) placed on the readouts: (a–c) cross-validated squared distance per unit to SL, AH and WH; (d) position on the SL → AH mean-difference axis (λ_Q); (e) WH computed the same way; (f) position on the shrinkage-LDA axis. Events are baseline-subtracted pre-lick rates, quiet windows raw rates."),
         ("fig-s-perf", P1 / "FigS_performance.png", "Session quantities vs behaviour: rows, single-neuron, geometry and decoder measures; columns, whisker hit rate, false-alarm rate (no-stim trials) and d′ (whisker hit vs no-stim lick rate) on the analysed trials. Open: learning, filled: expert; one OLS fit per cohort with 95 % CI band (solid if Spearman p < 0.05); titles: Spearman ρ, p, n."),
         ("fig-s-single-all", PUB / "all" / "Fig2_single_neurons.png", f"As {R.label('fig-single')}, all mice."),
         ("fig-s-distance-all", PUB / "all" / "Fig3_population_distance.png", f"As {R.label('fig-distance')}, all mice."),
         ("fig-s-decoders-all", PUB / "all" / "Fig5_decoders.png", f"As {R.label('fig-decoders')}, all mice."),
         ("fig-s-oddeven", WD / "halves" / POP / "within_session_oddeven.png", f"Session halves, odd/even null split ({POP}): events alternately assigned to the two halves, which removes time; changes should vanish."),
         ("fig-s-controls", WD / "halves" / POP / "within_session_controls.png", f"Session-half controls ({POP}): d(AH, SL), event rates, reaction times, firing rates and half durations per half."),
         ("fig-s-halves-all", WD / "halves" / "all" / "within_session_time.png", f"As {R.label('fig-halves')}, all mice."),
         ("fig-s-slopes-all", WD / "slopes" / "all" / "trial_slopes.png", f"As {R.label('fig-slopes')}, all mice."),
         ("fig-s-epochs-all", WD / "epochs" / "all" / "epoch_comparison.png", f"As {R.label('fig-epochs')}, all mice."),
         ("fig-s-timeline-exp", WD / "cosyne" / "COSYNE_convergence_timeline_expanded.png", f"Convergence timeline, expanded: methods with real examples, results and controls (all mice); conventions as {R.label('fig-timeline')}."),
         ("fig-s-decschemes", WD / "cosyne" / "decoder_schemes.png", "Decoder readouts across unit-sampling schemes (number of units per single-session decoder).")]
    for a, f, c in F:
        o.append(figure(a, f, c))
    o.append("\n# Supplementary tables\n")
    o.append(table("tbl-s-sizes", T, f"Sample sizes, Part I ({POP}): sessions, mice and tested units (good + mua, mean pre-lick rate ≥ {m51.MIN_FR} Hz, WH vs SL ROC tested) per cohort and stage."))
    o.append(table("tbl-s-sizes-areas", TA, f"Sessions / tested units per area group, cohort and stage ({POP})."))
    o.append(table("tbl-s-slopes", slopes_table(), f"Trial-level slopes against normalised session time (mean per group; Wilcoxon p vs 0), {POP}; day 0 R+ vs R− and cohort × stage by mouse-level permutation."))
    return "\n".join(o)


def appendix():
    par = pd.DataFrame([
        ("Pre-lick window", "[−100, 0) ms before the corrected first lick"), ("Hit baseline", "[−1.0, −0.015] s before trial start"),
        ("SL baseline", "[−1.0, −0.5] s before the lick"), ("SL definition", "piezo bout onset (≥ 1 s gap), outside [start − 0.2, rw stop + 0.5] s"),
        ("Min firing rate", f"{m51.MIN_FR} Hz"), ("ROC permutations", "1,000"), ("Split halves (d, λ)", "50"),
        ("Min units / events per class", "5 / 4"), ("Decoder", "L2 logistic, C = 0.05, balanced, k-fold"),
        ("Linear shifts", "40, |k| = 5 … n/3"), ("Interaction permutations", "10,000 (mouse level)"),
        ("Halves subsamples", "20"), ("Epoch footing", "6 events per class, 150 units")], columns=["Parameter", "Value"])
    files = pd.DataFrame([
        ("across_days/sl/publication/<pop>/", "Part I figures and stats (062, 075, 078)"), ("across_days/sl/{lambda,lambda_lda,transfer,decoder_*,pseudopop}/", "Part I session tables"),
        ("across_days/sl/normalisation/, extras/", "074 scaling comparison; 076 / 077 per-session extras"),
        ("within_day/sl/{halves,slopes,epochs,mixed_model,cosyne}/", "Part II (within_day 001–010)")], columns=["Results (combined_results_ks4/ssl-prelick-convergence/)", "Content"])
    return f"""
# Appendix

{table('tbl-a-params', par, "Parameters.")}

{table('tbl-a-files', files, "Result files.")}

**Code.** `projects/ssl-prelick-convergence/exploratory-analyses/` (Part I: 051–078, `coding_direction.py`; Part II:
`within_day/` 001–010); report: `report/build_report.py`. Parameters and files: {ref('tbl-a-params')}, {ref('tbl-a-files')}.
**History.** 2026-10-07: projects ssl-prelick-convergence and ssl-within-day-remapping merged into this report (Parts I
and II); figure revisions and new supplements (absolute selectivity, angles, normalisation, linear-shift geometry,
coding-direction estimators, quiet windows, performance).
"""


def main():
    T, TA, W = sizes()
    md = front(T) + intro() + methods() + part1(W) + part2() + discussion() + caveats() + supp(T, TA) + appendix()
    R.write(OUT, md)


if __name__ == "__main__":
    main()
