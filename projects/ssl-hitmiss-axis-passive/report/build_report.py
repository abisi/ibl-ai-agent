"""Report of ssl-hitmiss-axis-passive (first version, 2026-10-15; skills/project-report, report_lib).
Question: does learning the whisker-reward contingency change the early (5-35 ms) passive whisker response relative to the
population direction that separates whisker hits from misses? Levels: the axis and within-day projections (158), changes
beyond session time with the angle / projection split and areas (165 row 1), across days (177); whisker vs auditory, R+ vs R-,
real vs shifted axes and area groups shown inside each level's figure. Controls (decoder readout 160, exposure 153, expert
sessions 156) are supplementary. Part III of the ssl-whisker-hitmiss-timeresolved-decoding report (135b, 145, 147-151) is
superseded by this report.
Run (haas, after 135 HARM / min2, 158, 160, 153, 156, 165 SSL_165_ANGLE=1 SSL_165_ROW1=1, 177):
  cd ~/code/unit_spikes_analysis && PRELICK_REF=sl PRELICK_UNITS=stable PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities \
    ./.venv/bin/python ~/code/ibl-ai-agent/projects/ssl-hitmiss-axis-passive/report/build_report.py
Render locally: bash projects/ssl-hitmiss-axis-passive/report/render.sh
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EA = PROJ / "exploratory-analyses"
PUB = EA / "figures" / "publication"
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(REPO / "scripts")]
from report_lib import Report  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402

OUT = axel_bisi_root() / "combined_results_ks4" / PROJ.name / "report"
R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
_num = num
num = lambda *a, **k: _num(*a, **k).replace("-", "−")
R.register([
    ("fig-within", "Figure 1"), ("fig-time", "Figure 2"), ("fig-days", "Figure 3"),
    ("eq-axis", "Eq. (1)"), ("eq-proj", "Eq. (2)"), ("eq-excess", "Eq. (3)"), ("eq-angle", "Eq. (4)"),
    ("fig-s-shift", "Figure S1"), ("fig-s-dec", "Figure S2"), ("fig-s-expo", "Figure S3"), ("fig-s-expert", "Figure S4"),
    ("tbl-sizes", "Table S1"), ("tbl-registry", "Table S2"),
])

SRC = {"158": EA / "158_stats_learners.csv", "165": EA / "165_two_axes_compact_angle_row1_stats.csv",
       "177": EA / "177_report_across_days_stats.csv", "160": EA / "160_stats_learners.csv", "153": EA / "153_stats.csv",
       "156": EA / "156_stats.csv", "164": EA / "164_stats_learners.csv"}
FROZEN = dict(zip(SRC, Report.snapshot([SRC[k] for k in SRC], OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}


def r158(measure, cohort, k="158"):
    d = T[k][(T[k].measure == measure) & (T[k].cohort == cohort)]
    assert len(d) == 1, (measure, cohort)
    return d.iloc[0]


def r165(row, measure, cohort):
    d = T["165"][(T["165"].row == row) & (T["165"].measure == measure) & (T["165"].cohort == cohort)]
    assert len(d) == 1, (row, measure, cohort)
    return d.iloc[0]


def r177(panel):
    d = T["177"][T["177"].panel == panel]
    assert len(d) == 1, panel
    return d.iloc[0]


def wd(key, measure, fmt="{:+.3f}", k="158"):
    """within day: per cohort mean (Wilcoxon vs 0) and R+ vs R- (Mann-Whitney)"""
    p, m, c = r158(measure, "R+", k), r158(measure, "R-", k), r158(measure, "R+ vs R-", k)
    return dict(rp=f"R+ {num(key + '_rp', p.mean_a, fmt)} (p {pv(key + '_rp_p', p.p_nonparam)})",
                rm=f"R− {num(key + '_rm', m.mean_a, fmt)} (p {pv(key + '_rm_p', m.p_nonparam)})",
                c=f"R+ vs R− p {pv(key + '_c', c.p_nonparam)}", n=f"{int(p.n)} / {int(m.n)}")


def ad(key, panel, fmt="{:.3f}"):
    r = r177(panel)
    f = lambda c, s: num(f"{key}_{c}_{s}", r[f"mean_{c} {s}"], fmt)
    return dict(rp=f"R+ {f('R+', 'learning')} → {f('R+', 'expert')} (p {pv(key + '_rp_p', r['R+ L vs E p_MWU'])})",
                rm=f"R− {f('R−', 'learning')} → {f('R−', 'expert')} (p {pv(key + '_rm_p', r['R- L vs E p_MWU'])})",
                int=f"cohort × stage p {pv(key + '_int', r['interaction_p_perm'])}",
                ex=f"expert R+ vs R− p {pv(key + '_ex', r['expert R+ vs R- p_MWU'])}",
                d0=f"day 0 R+ vs R− p {pv(key + '_d0', r['learning R+ vs R- p_MWU'])}",
                n=f"{int(r['n_R+ learning'])} / {int(r['n_R+ expert'])} / {int(r['n_R− learning'])} / {int(r['n_R− expert'])}")


# ------------------------------------------------------------------------------------------------ numbers
X0 = r158("passive-pre whisker x", "both")
WP, AP = wd("w_proj", "W projection post - pre"), wd("a_proj", "A projection post - pre")
EW, EA_, EWA = wd("ex_w", "shift_excess_dWP"), wd("ex_a", "shift_excess_dAP"), wd("ex_wa", "shift_excess_dWAP")
XD = wd("x_disp", "whisker x displacement", "{:+.2f}")
TH = {c: r165("hit-miss angle", "q_th_W", c) for c in ("R+", "R-")}
THc, THA = r165("hit-miss angle", "q_th_W", "R+ vs R-"), r165("hit-miss angle", "q_th_A", "R+ vs R-")
THi = r165("hit-miss angle", "interaction q_th_W - q_th_A", "R+ vs R-")
TOST = r165("hit-miss angle", "R- post angle within 90 +- 10 deg (TOST)", "R-")
TPP = {c: r165("hit-miss angle", "theta pre / post (deg)", c) for c in ("R+", "R-")}
QW = {c: r165("hit-miss", "q_W", c) for c in ("R+", "R-")}
QWc, QWi = r165("hit-miss", "q_W", "R+ vs R-"), r165("hit-miss", "interaction q_W - q_A", "R+ vs R-")
QAN = T["165"][(T["165"].row == "hit-miss") & (T["165"].area == "ANOVA")].iloc[0]
QAR = T["165"][(T["165"].row == "hit-miss") & T["165"].area.notna() & (T["165"].area != "ANOVA")]
DPRE, DPOST, DPA = ad("d_pre", "177 pre_W_ex"), ad("d_post", "177 post_W_ex"), ad("d_posta", "177 post_A_ex")
DDW, DDA = ad("d_dw", "177 dW_ex"), ad("d_da", "177 dA_ex")
AN7 = r177("177h ANOVA"); Q7 = T["177"][T["177"].panel == "177h q"]
DEC = {k: wd("dec_" + k, m, "{:+.2f}", "160") for k, m in (("w", "shift_excess_dW"), ("wa", "shift_excess_dWA"))}
E3 = T["153"]
cov = lambda m, x: E3[(E3.panel == "covariate model") & (E3.measure == m) & (E3.x == x)].iloc[0]
cnt = lambda m: E3[(E3.panel == "counts") & (E3.measure == m)].iloc[0]
S6 = T["156"]
s6 = lambda m, test, grp: S6[(S6.measure == m) & (S6.test == test) & (S6.group == grp)].iloc[0]


def sizes():
    rows = [{"analysis": "Within day 0 (Fig. 1b, d)", "R+ day 0": WP["n"].split(" / ")[0], "R− day 0": WP["n"].split(" / ")[1],
             "R+ expert": "", "R− expert": ""},
            {"analysis": "Beyond session time, angle (Fig. 2)", "R+ day 0": int(QW["R+"].n), "R− day 0": int(QW["R-"].n), "R+ expert": "", "R− expert": ""}]
    n = DPRE["n"].split(" / ")
    rows.append({"analysis": "Across days (Fig. 3)", "R+ day 0": n[0], "R− day 0": n[2], "R+ expert": n[1], "R− expert": n[3]})
    rows.append({"analysis": "Expert control, lick axis (Fig. S4)", "R+ day 0": int(s6("shift_excess_dWR", "vs 0 (Wilcoxon | t)", "R+ learning").n),
                 "R− day 0": int(s6("shift_excess_dWR", "vs 0 (Wilcoxon | t)", "R- learning").n),
                 "R+ expert": int(s6("shift_excess_dWR", "vs 0 (Wilcoxon | t)", "R+ expert").n),
                 "R− expert": int(s6("shift_excess_dWR", "vs 0 (Wilcoxon | t)", "R- expert").n)})
    return pd.DataFrame(rows)


def build():
    md = rf"""---
title: "Passive whisker responses and the hit − miss axis: within day 0 and across days"
subtitle: "ssl-hitmiss-axis-passive · learners · tracked stable units · Kilosort 4"
date: "{date.today().isoformat()}"
---

## Abstract

In every session the population response 5–35 ms after the whisker stimulus, before any movement, separates whisker trials that
will be licked (hits) from those that will not (misses). We asked how passive whisker responses, recorded before and after the
task, lie along this hit − miss axis and how this changes with learning in mice rewarded (R+) or not rewarded (R−) for licking
to the whisker stimulus. Before the task the passive whisker response lies between the hit and miss levels in both cohorts. Over
day 0, the passive whisker response of R− mice moves away from the hit side beyond what session time explains, while that of R+
mice barely changes; the auditory response, a control stimulus rewarded in both cohorts, does not separate the cohorts. The R−
change is a rotation: the angle between the whisker response and the axis approaches 90° after the task. Across days, the
passive whisker response of R+ experts lies further toward the hit side than that of R− experts, but the cohort × stage
interaction is not significant. Area groups change in the same direction without an area-specific effect after correction.

**Key results** (sessions are the unit; real − shifted-axis values; Wilcoxon vs 0 within cohort, Mann–Whitney between cohorts).

1. Day 0, beyond session time: passive whisker response moves away from the hit side in R− ({EW['rm']}), much less in R+ ({EW['rp']}); {EW['c']} ({ref('fig-within')}d).
2. The R− change is a rotation toward orthogonality: angle change {num('th_rm', TH['R-']['mean'], '{:+.1f}')}° (p {pv('th_rm_p', TH['R-'].p_vs0)}) vs R+ {num('th_rp', TH['R+']['mean'], '{:+.1f}')}°; R+ vs R− p {pv('th_c', THc.p_nonparam)}; auditory unchanged ({ref('fig-time')}c).
3. Whisker vs auditory: cohort × modality interaction of the projection change p {pv('qwi', QWi.p_perm)} ({ref('fig-time')}d).
4. Across days, passive post (whisker): {DPOST['rp']}; {DPOST['rm']}; {DPOST['ex']}; {DPOST['int']} ({ref('fig-days')}d).
5. Areas: cohort effect on the day-0 change p {pv('qan_c', QAN.p_cohort_perm)}, cohort × area p {pv('qan_ca', QAN.p_cohort_x_area_perm)} ({ref('fig-time')}e); across days cohort × stage × area p {pv('an7_csa', AN7['cohort x stage x area p_perm'])} ({ref('fig-days')}h).

## Introduction

The whisker stimulus evokes a fast response across the brain within 35 ms, before any lick. On active trials, this early
response already differs between hits and misses, defining a hit − miss axis in each session. Passive whisker responses
recorded before (pre) and after (post) the active block, with no lick and no reward, show whether the sensory response itself
moves along this axis. If learning that the whisker stimulus predicts reward (R+) or does not (R−) shapes the sensory response,
the passive whisker response should move toward the hit side in R+ and toward the miss side, or away from the axis, in R−. The
auditory stimulus is rewarded in both cohorts and is the control at every level. Hit and miss trials drift over the session,
so every comparison is made against axes rebuilt from time-shifted labels ({ref('fig-s-shift')}).

The analysis is organised in three levels: the axis and the within-day change ({ref('fig-within')}), its separation from session time
and its geometry ({ref('fig-time')}), and the comparison of day 0 with expert sessions ({ref('fig-days')}). Every claim, with its figure,
statistic and status, is listed in {ref('tbl-registry')}.

## Methods

{R.methods('task', 'learners', 'trials', 'units')}
Sessions need passive trials before and after the active block (sessions per analysis in {ref('tbl-sizes')}). Responses are the spike count 5–35 ms after stimulus onset
minus the unit's mean baseline (−55 to −20 ms) in the same epoch; active whisker trials with a lick before 35 ms are excluded.
Units are scaled with their mean and SD over the active trials (harmonised scaling), so passive projections are in units of
the active response variability.

The hit − miss axis is computed per session from a random half of the active whisker trials and averaged over 50 splits:
{eq('eq-axis', r"\mathbf{u} = \frac{\bar{\mathbf{z}}_{\mathrm{hit}} - \bar{\mathbf{z}}_{\mathrm{miss}}}{\lVert \bar{\mathbf{z}}_{\mathrm{hit}} - \bar{\mathbf{z}}_{\mathrm{miss}} \rVert}")}
where $\bar{{\mathbf z}}$ are class means of the scaled population responses. Passive whisker (W) and auditory (A) responses, pre
and post, are projected per neuron:
{eq('eq-proj', r"\pi_{s}^{e} = \frac{1}{\sqrt{N}}\, \mathbf{u}^\top \bar{\mathbf{z}}_{s}^{e}, \qquad s \in \{W, A\},\ e \in \{\mathrm{pre}, \mathrm{post}\}")}
with $N$ the number of units. Within day 0 the axis requires at least 10 hits and 10 misses; across days at least 4 of each, so
that expert sessions (R+ rarely miss, R− rarely hit) can enter.

{R.methods('shift_null')}
Here the hit / miss labels of the time-ordered active whisker trials are shifted (50 shifts), the axis rebuilt, and the passive
projections recomputed with the passive data unchanged. Every reported value is the excess over this null:
{eq('eq-excess', r"\Delta\pi_s = \left(\pi_s^{\mathrm{post}} - \pi_s^{\mathrm{pre}}\right) - \frac{1}{K}\sum_{k=1}^{K}\left(\pi_s^{\mathrm{post},(k)} - \pi_s^{\mathrm{pre},(k)}\right)")}
where $k$ indexes the $K$ shifted axes. The projection of a passive response is its cross-validated size times the cosine of its
angle to the axis. The angle is:
{eq('eq-angle', r"\theta_s^{e} = \arccos\frac{\mathbf{u}^\top \bar{\mathbf{z}}_{s}^{e}}{\lVert \bar{\mathbf{z}}_{s}^{e} \rVert}")}
with the norm cross-validated from random halves of the passive trials. It separates a rotation of the response relative to
the axis from a change in its size (gain).

{R.methods('statistics', 'areas')}
Within day 0, the session values are tested against 0 per cohort (Wilcoxon) and between cohorts (Mann–Whitney); the cohort ×
modality interaction (whisker vs auditory change) uses 10,000 permutations of the cohort labels across mice. Area groups need at
least 20 tracked units.

## Results

### The axis and the within-day change

Before the task, the passive whisker response projects between the active miss and hit levels in both cohorts (mean passive-pre
position {num('x0_pre', X0.mean_a, '{:.1f}')} vs hits {num('x0_hit', X0.mean_b, '{:.1f}')} on the axis, z units; {ref('fig-within')}a). Over
the active block the projection of the passive whisker response fell in both cohorts, by much more in R−: {WP['rp']}; {WP['rm']};
{WP['c']} ({ref('fig-within')}b). The auditory response fell in R+ ({AP['rp']}) but not in R− ({AP['rm']}); {AP['c']} ({ref('fig-within')}c).

Beyond session time ({ref('eq-excess')}), the R− whisker response still moved away from the hit side ({EW['rm']}), the R+ response
moved much less ({EW['rp']}), and the cohorts differed ({EW['c']}). The auditory excess did not differ between cohorts ({EA_['c']}), and the
whisker − auditory contrast, in which any drift shared by the two stimuli cancels, did ({EWA['rm']}; {EWA['c']}) ({ref('fig-within')}d).
In the state space spanned by the axis and the passive-pre whisker pattern, the R− whisker response moved to the level of the active
misses ({XD['rm']}), the R+ response less ({XD['rp']}) ({ref('fig-within')}e).

### Beyond session time: a rotation away from the axis in R−

The angle between the passive whisker response and the axis ({ref('eq-angle')}) increased from passive pre to post beyond session time
in R− ({num('th_rm2', TH['R-']['mean'], '{:+.1f}')}°, p {pv('th_rm_p2', TH['R-'].p_vs0)}) and not significantly in R+
({num('th_rp2', TH['R+']['mean'], '{:+.1f}')}°, p {pv('th_rp_p', TH['R+'].p_vs0)}); R+ vs R− p {pv('th_c2', THc.p_nonparam)} ({ref('fig-time')}b–c). After the
task the R− whisker response was close to orthogonal to the axis ({num('tpp_rm_post', TPP['R-'].mean_post, '{:.0f}')}°, within 90 ± 10°:
TOST p {pv('tost', TOST.p_tost)}; R+ {num('tpp_rp_post', TPP['R+'].mean_post, '{:.0f}')}°). The auditory angle did not change differently between
cohorts (p {pv('tha_c', THA.p_nonparam)}); the cohort × modality interaction of the angle was p {pv('thi', THi.p_perm)}. The projection change
beyond session time, on the same sessions, gave R+ {num('qw_rp', QW['R+']['mean'], '{:+.3f}')} and R− {num('qw_rm', QW['R-']['mean'], '{:+.3f}')}
(R+ vs R− p {pv('qw_c', QWc.p_nonparam)}; cohort × modality p {pv('qwi2', QWi.p_perm)}) ({ref('fig-time')}d). The R− change is therefore a rotation of the
whisker response away from the hit − miss direction rather than a loss of gain.

Per area, the change was negative in R− for most groups and near zero in R+ ({ref('fig-time')}e). The two-way model gave a cohort effect
(p {pv('qan_c2', QAN.p_cohort_perm)}) without cohort × area dependence (p {pv('qan_ca2', QAN.p_cohort_x_area_perm)}); no single area survived the
correction (smallest q = {num('qar_min', QAR.q_BH.min(), '{:.2f}')}).

### Across days

At expert sessions the axis is estimated from fewer trials of the rare class, and the comparison is between sessions. Before the
session, the passive whisker response of R+ experts lay further on the hit side than that of R− experts ({DPRE['ex']}); this was
already a trend on day 0 ({DPRE['d0']}). After the session the difference was clear at both stages ({DPOST['d0']}; {DPOST['ex']}),
with R+ rising from day 0 to expert ({DPOST['rp']}) and R− not changing ({DPOST['rm']}); the cohort × stage interaction was not
significant ({DPOST['int']}) ({ref('fig-days')}b–d). The auditory response did not differ between cohorts or stages ({DPA['ex']}; {DPA['int']};
{ref('fig-days')}e). The within-session change of expert sessions was small in both cohorts (whisker {DDW['rp']}; {DDW['rm']}; {DDW['int']};
{ref('fig-days')}f–g), so the R− rotation of day 0 is not repeated in expert sessions.

Per area, the expert − day 0 change of the passive-post whisker projection did not depend on the area (cohort × stage × area p
{pv('an7_csa2', AN7['cohort x stage x area p_perm'])}; cohort × stage p {pv('an7_cs', AN7['cohort x stage p_perm'])}; {num('an7_n', int(AN7['n_areas']))}
groups with ≥ 3 sessions in all four groups; smallest post hoc q = {num('q7_min', Q7.q_BH.min(), '{:.2f}')}; {ref('fig-days')}h).

### Controls

*Decoder readout.* A hit vs miss decoder (L2 logistic regression) scoring passive trials gives the same direction in R− but is
more sensitive to session time: beyond its linear-shift null, the whisker readout fell in both cohorts ({DEC['w']['rp']};
{DEC['w']['rm']}; {DEC['w']['c']}) and the whisker − auditory contrast did not separate them ({DEC['wa']['c']}) ({ref('fig-s-dec')}). The mean-difference axis is
the main measure; the decoder is reported as a variant whose readout can use any time-correlated direction.

*Stimulus exposure.* R− sessions are longer, so R− mice received more active whisker trials (median
{num('nw_rp', cnt('nW').median_rp, '{:.0f}')} vs {num('nw_rm', cnt('nW').median_rm, '{:.0f}')}) and auditory trials, in the same proportion. Trial
counts predicted the shrinkage of each passive response along its own pattern but not the choice-axis effect: with trial counts as
covariates, the R− coefficient of the whisker − auditory displacement along the axis stayed {num('cov_wa', cov('dx_WA', '+ nW, nA').spearman, '{:+.2f}')}
(p {pv('cov_wa_p', cov('dx_WA', '+ nW, nA').p_param)}) ({ref('fig-s-expo')}).

*Expert sessions as a within-session control.* The same within-session measures in expert sessions show no cohort difference
(lick-axis whisker − auditory projection, expert R+ vs R− p {pv('s6_ex', s6('shift_excess_dWAP', 'R+ vs R- (Mann-Whitney | Welch)', 'expert').p_nonparam)};
day 0 p {pv('s6_d0', s6('shift_excess_dWAP', 'R+ vs R- (Mann-Whitney | Welch)', 'learning').p_nonparam)}; cohort × stage p
{pv('s6_int', s6('shift_excess_dWAP', 'cohort x stage (permutation | OLS)', 'interaction').p_nonparam)}) ({ref('fig-s-expert')}). The R− rotation is
specific to the learning day, with the caveat that only 5 R− expert sessions enter this test.

## Discussion

On the day R− mice learn that the whisker stimulus is not rewarded, their passive whisker response rotates away from the population
direction that separates licked from unlicked whisker trials, beyond what session time produces, while the auditory response does not.
R+ mice show little change within day 0, but across days their passive whisker response sits further toward the hit side than in R−
experts. The two cohorts therefore diverge in opposite ways: R− decouples the sensory response from the lick axis within the learning
session, and R+ shows a hit-side bias across days, although neither the across-days interaction nor any single area reaches
significance.

The sensory responses of both stimuli also shrink after the task in both cohorts, scaled by the number of trials; this adaptation-like
change does not separate the cohorts and lies mostly off the hit − miss axis. The rotation result depends on the linear-shift null,
which is conservative because true learning is also time-correlated.

## Caveats and limitations

- The learner definition is under review (Methods); day-0 results may change when it is settled.
- Expert sessions are few ({ref('tbl-sizes')}) and the across-days axis is estimated from 4 trials of the rare class at minimum.
- The decoder variant does not separate the cohorts beyond session time; the result rests on the mean-difference axis.
- The relation of the axis change to behaviour (hit-rate change, learning curves) is confounded with session time and is not claimed.
- Five sessions miss drift-test files and are absent from the tracked unit list (rerun pending).

## Open questions and next steps

- Settle the learner definition; rerun the drift tests of the five missing sessions; rebuild.
- Combined within-day report with the reward-lick coding direction (ssl-prelick-convergence), expert sessions as control.

## Figures

{figure('fig-within', PUB / '158_cosyne_within_day_learners.png', rf'''Within day 0 (learners, whole brain, tracked stable units, 5–35 ms).
**a**, State space before the task: x = hit − miss axis ({ref('eq-axis')}), y = passive-pre whisker pattern orthogonal to it; active hits
and misses (triangles), passive-pre whisker (orange) and auditory (blue) responses (mean over sessions). **b**, Projection of the
whisker response on the axis per epoch (passive pre, active halves, passive post; {ref('eq-proj')}); mean ± s.e.m. over sessions; title:
post − pre within cohort (Wilcoxon) and between cohorts (Mann–Whitney). **c**, Same for the auditory response. **d**, Passive pre →
post change beyond session time ({ref('eq-excess')}) for whisker, auditory and whisker − auditory; filled: ≠ 0. **e**, State space per
cohort with the pre → post displacement of the passive responses.''', "158_cosyne_within_day.py")}
{figure('fig-time', PUB / '165_two_axes_compact_angle_row1.png', rf'''Changes beyond session time: rotation vs projection, and areas.
**a**, Time course of held-out active hits / misses and passive pre / post whisker trials on the axis (grey band 5–35 ms; schematic slot
left blank). **b**, Angle between the passive whisker response and the axis ({ref('eq-angle')}), pre and post per session (thin lines),
real axis (colour) vs shifted axes (grey). **c**, Change of angle beyond session time for whisker (yellow) and auditory (blue)
responses per cohort; legend: R+ vs R− per modality; title: cohort × modality permutation. **d**, Change of projection beyond session
time, same layout. **e**, Per area group, change of projection (R+ green, R− magenta; allen_utils order); title: two-way permutation
ANOVA (cohort, cohort × area).''', "165_two_axes_figure.py (SSL_165_ANGLE=1 SSL_165_ROW1=1)")}
{figure('fig-days', PUB / '177_report_across_days.png', rf'''Across days (day 0 learners vs expert sessions; axis with ≥ 4 hits and misses).
**a**, Held-out active hits (orange), misses (brown) and passive-post whisker trials (black) on the axis over time, per cohort and stage
(n sessions in titles). **b**, Passive pre / post projections, real (colour) vs shifted axes (grey), whisker and auditory on the same
scale (open day 0, filled expert). **c–e**, Passive pre whisker, passive post whisker and passive post auditory projections, real −
shifted, day 0 vs expert (n under the axis: R+ D0 / E / R− D0 / E = {DPOST['n']}); brackets Mann–Whitney; title: cohort × stage by mouse-level
permutation. **f–g**, Within-session change post − pre, whisker and auditory. **h**, Per area group, change of the passive-post whisker
projection (expert − day 0; areas with ≥ 3 sessions in all four groups); title: omnibus cohort × stage × area; * post hoc BH q < 0.05 (none).''',
                 "177_report_across_days.py")}

## Supplementary figures and tables

{figure('fig-s-shift', PUB / '174_linear_shift_schematic.png', '''Linear-shift control (schematic): the hit / miss labels of the time-ordered
active trials are shifted against the neural data, the axis is rebuilt from the shifted labels, and the passive projections are recomputed;
slow drifts shared by labels and activity are kept in the null.''', "174_linear_shift_schematic.py", kind="supplementary")}
{figure('fig-s-dec', PUB / '160_cosyne_decoder_learners.png', '''Decoder variant of Figure 1: choice-decoder readout (L2 logistic, hit vs
miss on active whisker trials; passive and auditory trials scored by models not trained on them; SD units, + hit side) per epoch for
whisker (b) and auditory (c) trials, and the pre → post change beyond the decoder's linear-shift null (d).''', "160_cosyne_decoder.py",
                 kind="supplementary")}
{figure('fig-s-expo', PUB / '153_exposure_control.png', '''Stimulus exposure: a, active whisker and auditory trials per session; b, shrinkage of
each passive response along its own pre-task pattern vs the number of active whisker trials; c, choice-axis changes vs the same counts
(OLS line, 95 % CI, solid if Pearson p < 0.05); d, R− − R+ effect without (grey) and with (black) trial counts as covariates.''',
                 "153_exposure_control.py", kind="supplementary")}
{figure('fig-s-expert', PUB / '156_expert_control.png', '''Expert sessions as a within-session control: the day-0 within-session measures
(lick-axis cosines and projections beyond session time, decoder readout, state space, active hit / miss decoding) for learning (filled) and
expert (open) sessions; p values per group vs 0 (Wilcoxon | t), R+ vs R− (Mann–Whitney | Welch), cohort × stage (permutation | OLS).''',
                 "156_expert_control.py", kind="supplementary")}
{table('tbl-sizes', sizes(), '''Sessions per analysis (learners on day 0; all mice at expert).''')}
"""
    claims()
    md += "\n## Appendix\n\n" + R.registry_table("tbl-registry") + APPENDIX
    return md


def claims():
    claim("C1", "Before the task the passive whisker response lies between miss and hit levels", "Fig. 1a", "descriptive")
    claim("C2", "Day 0: R− passive whisker response moves away from the hit side beyond session time; R+ little", "Fig. 1d",
          f"{EW['rm']}; {EW['rp']}; {EW['c']}", EW["n"])
    claim("C3", "Auditory control does not separate the cohorts; whisker − auditory does", "Fig. 1d", f"{EA_['c']}; {EWA['c']}", EWA["n"])
    claim("C4", "The R− change is a rotation toward orthogonality (angle), not a gain loss", "Fig. 2b–c",
          f"R− {TH['R-']['mean']:+.1f}°; R+ vs R− p = {THc.p_nonparam:.3f}; TOST p = {TOST.p_tost:.3f}")
    claim("C5", "Cohort × modality interaction of the projection change", "Fig. 2d", f"p = {QWi.p_perm:.4f}")
    claim("C6", "Day-0 change distributed across areas (cohort effect, no area dependence)", "Fig. 2e",
          f"cohort p = {QAN.p_cohort_perm:.3f}; cohort × area p = {QAN.p_cohort_x_area_perm:.2f}")
    claim("C7", "Across days R+ passive whisker sits further on the hit side than R− at expert", "Fig. 3d", f"{DPOST['ex']}", DPOST["n"])
    claim("C8", "Across-days cohort × stage interaction", "Fig. 3c–d", f"pre {DPRE['int']}; post {DPOST['int']}", DPOST["n"], status="n.s.")
    claim("C9", "Decoder readout separates cohorts beyond session time", "Fig. S2", DEC["wa"]["c"], status="n.s.",
          note="decoder uses time-correlated directions")
    claim("C10", "Exposure explains the shared shrinkage, not the cohort difference on the axis", "Fig. S3",
          f"dx_WA R− coef with counts {cov('dx_WA', '+ nW, nA').spearman:+.2f}, p = {cov('dx_WA', '+ nW, nA').p_param:.3f}")
    claim("C11", "R− rotation is specific to the learning day (no cohort difference in expert sessions)", "Fig. S4",
          f"expert p = {s6('shift_excess_dWAP', 'R+ vs R- (Mann-Whitney | Welch)', 'expert').p_nonparam:.2f}", status="fragile",
          note="5 R− expert sessions")
    claim("C12", "Axis change relates to behaviour (hit-rate change, learning curves)", "not shown", "", status="fragile",
          note="confounded by session time (170, 176)")


APPENDIX = """
### Figure versions and provenance

New in this report: 165_two_axes_compact_angle_row1 (hit − miss row of 165 compact angle) and 177_report_across_days (panels of
ssl-prelick-convergence 119 row 1 and 175 v2, with the auditory projection and the area panel). Superseded: Part III of the
ssl-whisker-hitmiss-timeresolved-decoding report (135b alignment, 145 coding direction, 147 choice readout, 149 / 150 shift nulls,
151 state-space variants, 153 v1), 175 v1 / v2 (replaced by 177), 162 (per-area breakdown in the old order; same quantity as Figure 2e),
164 (real vs shifted axes on two scales; the per-neuron scale is Figure 1d). Conference composites (154, 158 is used as Figure 1, 160,
165 two-row, 173) are not included as such. Source files, hashes and dates: figures.json.

### Result folders

combined_results_ks4/ssl-hitmiss-axis-passive/: tables (135 HARM / min2 parquet files, 1xx_stats*.csv), figures/publication
(all figures), cache. Statistics read by this report are frozen in report/_snapshots/.
"""


def summary():
    return f"""# ssl-hitmiss-axis-passive: summary, {date.today().isoformat()}

Question: how do passive whisker responses (5–35 ms) lie along each session's hit − miss axis, and how does this change with learning
in R+ vs R− mice (within day 0 and across days; learners; tracked stable units)?

1. Day 0, beyond session time: R− passive whisker response moves away from the hit side ({EW['rm']}); R+ much less ({EW['rp']}); {EW['c']}.
2. It is a rotation: the R− whisker response becomes nearly orthogonal to the axis after the task (angle +{TH['R-']['mean']:.1f}°); auditory unchanged; cohort × modality interaction of the projection change p {pv('s_qwi', QWi.p_perm)}.
3. Across days: R+ passive whisker sits further on the hit side at expert ({DPOST['ex']}), but {DPOST['int']} (n.s.).
4. Areas: distributed, no area survives correction.
5. Controls: exposure explains the shared shrinkage, not the axis effect; the decoder variant is time-confounded; expert sessions show no within-session cohort difference (few R− experts).

Open: learner definition; drift files of five sessions.

![Within day 0](figures/fig-within.png)

Full report: report.pdf / report.html; claims registry: registry.csv.
"""


def main():
    md = build()
    R.write(OUT, md, summary=summary())


if __name__ == "__main__":
    main()
