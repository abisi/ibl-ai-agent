"""Report of ssl-whisker-hitmiss-timeresolved-decoding, rebuilt 2026-10-15 on report_lib (skills/project-report).
Scope (user 2026-10-15): decodability (session-wide, time-resolved), single-session vs pseudo-population decoding, within-session
change (session splits), learning-trial definitions, single-trial readout vs behaviour. The passive hit / miss axis analyses (old
Part III) are in ssl-hitmiss-axis-passive. Cohort, area groups and the control decoding (stimulus modality) are shown inside each
level's figure. The previous generator (Parts I-III, hand-numbered figures) is in git history.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-whisker-hitmiss-timeresolved-decoding/report/build_report.py
Render locally: bash projects/ssl-whisker-hitmiss-timeresolved-decoding/report/render.sh
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
EA = PROJ / "exploratory-analyses"
FIG, PUB = EA / "figures", EA / "figures" / "publication"
PP = PROJ.parent / "ssl-pseudopopulation-area-decoding"
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(REPO / "scripts")]
from report_lib import Report  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402

OUT = axel_bisi_root() / "combined_results_ks4" / PROJ.name / "report"
R = Report(banned=())
num, pv, ref, figure, table, eq, claim = R.num, R.pv, R.ref, R.figure, R.table, R.eq, R.claim
_num = num
num = lambda *a, **k: _num(*a, **k).replace("-", "−")
R.register([
    ("fig-choice", "Figure 1"), ("fig-premod", "Figure 2"), ("fig-halves", "Figure 3"), ("fig-hitmedian", "Figure 4"),
    ("fig-placebo", "Figure 5"), ("fig-margin", "Figure 6"),
    ("eq-change", "Eq. (1)"), ("eq-excess", "Eq. (2)"), ("eq-margin", "Eq. (3)"),
    ("fig-s-stim", "Figure S1"), ("fig-s-perf", "Figure S2"), ("fig-s-matched", "Figure S3"), ("fig-s-pp", "Figure S4"),
    ("fig-s-ncount", "Figure S5"), ("fig-s-l5", "Figure S6"), ("fig-s-step", "Figure S7"), ("fig-s-perm", "Figure S8"),
    ("fig-s-xcorr", "Figure S9"), ("fig-s-cosyne", "Figure S10"),
    ("tbl-wb", "Table S1"), ("tbl-registry", "Table S2"),
])

SRC = {"113": EA / "113_stats.csv", "110": EA / "110_publication_stats.csv", "117": EA / "117b_halves_matched_full_stats.csv",
       "016": PP / "artifacts" / "016_pseudopop_halves_stats.csv", "144": EA / "144_stats_all.csv",
       "122b": EA / "122b_lt_placebo_stats.csv", "126": EA / "126_lt_variants_stats_step1_pl100.csv", "131": EA / "131_cohort_permutation.csv",
       "143": EA / "143_stats_L5_all.csv", "143sg": EA / "143_step_vs_gradual_stats_all.csv", "127": EA / "127_stats.csv",
       "176": EA / "176_xcorr_stats.csv", "129": EA / "129_stats.csv", "130": EA / "130_stats.csv"}
FROZEN = dict(zip(SRC, Report.snapshot([SRC[k] for k in SRC], OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    d = df[m]
    if not len(d):
        raise KeyError(kw)
    return d.iloc[0]


def pick(df, **kw):
    """group-level row (no session_id), whole brain, learning stage"""
    d = df[df.session_id.isna()] if "session_id" in df else df
    return sel(d, level="whole_brain", stage="learning", **kw)


def clusters(df, analysis, cohort, test):
    m = ((df.level == "whole_brain") & (df.stage == "learning") & (df.window == "time-resolved") & (df.analysis == analysis)
         & (df.cohort == cohort) & (df.test == test) & (df.condition == "whole") & (df.p < 0.05))
    return df[m & df.session_id.isna()].sort_values("t_start_ms") if "session_id" in df else df[m].sort_values("t_start_ms")


VS0, MWW = "accuracy - null vs 0 (Wilcoxon | one-sample t)", "Mann-Whitney | Welch"
COHT = "two-sided cluster vs mouse-level cohort-label permutation"

# ------------------------------------------------------------------------------------------------ numbers
HS = {c: pick(T["113"], analysis="hitmiss", window="sensory", condition="whole", cohort=c, test=VS0) for c in ("R+", "R-")}
HB = {c: pick(T["113"], analysis="hitmiss", window="baseline", condition="whole", cohort=c, test=VS0) for c in ("R+", "R-")}
HC = pick(T["113"], analysis="hitmiss", window="sensory", condition="whole", cohort="R+ vs R-", test=MWW)
HCL = clusters(T["113"], "hitmiss", "R+ - R-", COHT)
HCL = HCL[HCL.t_start_ms > 0]
HD = pick(T["113"], analysis="hitmiss", window="sensory", condition="whole", cohort="R-", test="correlation with behavioural d' (Pearson | Spearman)")
MS = {c: pick(T["113"], analysis="modality_stim", window="sensory", condition="whole", cohort=c, test=VS0) for c in ("R+", "R-")}
ML = {c: pick(T["110"], analysis="modality_lick", window="pre-lick", condition="whole", cohort=c, test=VS0) for c in ("R+", "R-")}
MLC = pick(T["110"], analysis="modality_lick", window="pre-lick", condition="whole", cohort="R+ vs R-", test=MWW)
MH = {c: pick(T["110"], analysis="modality_lick", window="pre-lick", condition="half first vs second", cohort=c,
              test="paired Wilcoxon | paired t") for c in ("R+", "R-")}
LAT = T["110"][(T["110"].analysis == "modality_lick") & (T["110"].stage == "learning") & (T["110"].level == "area_group")
               & (T["110"].test == "50%-of-peak latency, mouse bootstrap 95% CI")].pivot_table(index="area", columns="cohort", values="latency_ms")
PS = {c: pick(T["113"], analysis="perfstate", window="sensory", condition="whole", cohort=c, test=VS0) for c in ("R+", "R-")}
H117 = {c: sel(T["117"], cohort=c, mode="matched") for c in ("R+", "R-")}
P016 = sel(T["016"], decoding="hitmiss", window="5-100ms", area="All units")
L5 = {c: sel(T["126"], decoding="hitmiss", window="5-100ms", variant="L5 whisker CP", cohort=c) for c in ("R+", "R-")}
HALF = sel(T["126"], decoding="hitmiss", window="5-100ms", variant="half", cohort="R+")
L8 = sel(T["126"], decoding="hitmiss", window="5-100ms", variant="L8 fixed margin", cohort="R-")
PCT = sel(T["122b"], decoding="hitmiss", window="5-100ms", definition="L5 whisker CP", cohort="R+")
C131 = sel(T["131"], stat="excess", decoding="hitmiss", window="5-100ms", variant="L5 whisker CP")
B129 = sel(T["129"], decoding="hitmiss", window="5-100ms", cohort="R+")
D144 = T["144"]
HM144 = D144[D144.window.str.startswith("hitmiss") & D144.row.isin(["sep", "sgl"]) & (D144.test != "description") & (D144.split == "hitmedian")]
M144 = {c: sel(D144, row="sep", window="modality_lick -100-0ms", split="hitmedian", cohort=c) for c in ("R+", "R-", "R+ vs R-")}
X143 = T["143"][T["143"].window == "5-100ms"]
R143 = sel(X143, panel="real vs placebo", group="R+")
E143 = sel(X143, panel="excess", test="R+ vs R- (Mann-Whitney | Welch)")
P143 = X143[X143.test.str.contains("permutation")].iloc[0]
SG = sel(T["143sg"], window="5-50ms", measure="exc", group="step vs gradual", cohort="R+")
W127 = sel(T["127"], decoding="whisker_nostim", window="5-100ms", curve="whisker", cohort="R+")
H127 = {c: sel(T["127"], decoding="hitmiss", window="5-100ms", curve="whisker", cohort=c) for c in ("R+", "R-")}
X176 = T["176"][(T["176"].window == "5-50ms") & (T["176"].curve == "w-fa") & (T["176"].signal == "level") & (T["176"].pop == "learners")]
S176 = {c: X176[(X176.cohort == c) & (X176.q < 0.05)] for c in ("R+", "R-")}
N130 = {c: sel(T["130"], decoding="hitmiss", window="5-50ms", cohort=c) for c in ("R+", "R-")}


def pp_(k, r, c):
    return f"{num(k, r['mean'], '{:+.3f}')} (n = {int(r.n)}, p {pv(k + '_p', r.p_nonparam)})"


def wb_table():
    rows = []
    for an, df, w, lab in (("hitmiss", T["113"], "sensory", "hit vs miss, 5–50 ms"), ("hitmiss", T["113"], "baseline", "hit vs miss, −200 to −10 ms"),
                           ("modality_stim", T["113"], "sensory", "whisker vs auditory, 5–50 ms"),
                           ("modality_lick", T["110"], "pre-lick", "whisker vs auditory, −100 to 0 ms before lick"),
                           ("perfstate", T["113"], "sensory", "performance state, 5–50 ms")):
        a = {c: pick(df, analysis=an, window=w, condition="whole", cohort=c, test=VS0) for c in ("R+", "R-")}
        b = pick(df, analysis=an, window=w, condition="whole", cohort="R+ vs R-", test=MWW)
        f = lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}"
        rows.append({"decoding": lab, "R+ (n)": f"{a['R+']['mean']:+.3f} ({int(a['R+'].n)})", "p R+": f(a["R+"].p_nonparam),
                     "R− (n)": f"{a['R-']['mean']:+.3f} ({int(a['R-'].n)})", "p R−": f(a["R-"].p_nonparam),
                     "R+ vs R− p (MW | Welch)": f"{f(b.p_nonparam)} | {f(b.p_param)}"})
    return pd.DataFrame(rows)


def build():
    lat = {c: float(LAT[c].median()) for c in ("R+", "R-")}
    hmin = np.nanmin(HM144[["p_nonparam", "p_param"]].min(axis=1))
    md = rf"""---
title: "Choice and modality information during single-session whisker learning: decodability and within-session change"
subtitle: "ssl-whisker-hitmiss-timeresolved-decoding · day 0 · Kilosort 4"
date: "{date.today().isoformat()}"
---

## Abstract

Mice of two cohorts learn the same whisker Go/NoGo task within one session; licking after the whisker stimulus is rewarded in R+
and not in R−. From whole-brain and area-level Neuropixels populations we decoded the upcoming choice (lick vs no lick on whisker
trials), the stimulus modality at onset and before the first lick, and the performance state, and asked whether this information
differs between cohorts and changes within the learning session. Over the whole session, choice was decodable in both cohorts
before and after the whisker stimulus, without a cohort difference in the early window; pre-lick modality information was equal
in the two cohorts. Splitting each session in halves, with or without equal numbers of hits, showed no change of choice decoding.
Splitting at the behavioural change point of the whisker hit sequence, and comparing with placebo splits of the same session,
showed a gain of choice decoding in R+ step learners, larger than in R−. Single-trial decoder margins followed the learning curve
with cohort-specific signs. All analyses are exploratory and p values uncorrected across analyses.

**Key results** (day 0; session = unit; accuracy minus the session's linear-shift null; Wilcoxon / Mann–Whitney unless stated).

1. Choice 5–50 ms after the whisker stimulus: R+ {pp_('k_hs_rp', HS['R+'], 'R+')}, R− {pp_('k_hs_rm', HS['R-'], 'R-')}; R+ vs R− p {pv('k_hc', HC.p_nonparam)} ({ref('fig-choice')}).
2. Pre-lick modality: R+ {pp_('k_ml_rp', ML['R+'], 'R+')}, R− {pp_('k_ml_rm', ML['R-'], 'R-')}; R+ vs R− p {pv('k_mlc', MLC.p_nonparam)} ({ref('fig-premod')}).
3. Halves with equal hits: change R+ {num('k_h_rp', H117['R+'].change, '{:+.3f}')} (p {pv('k_h_rp_p', H117['R+'].p_wilcoxon)}), R− {num('k_h_rm', H117['R-'].change, '{:+.3f}')} (p {pv('k_h_rm_p', H117['R-'].p_wilcoxon)}) ({ref('fig-halves')}, {ref('fig-hitmedian')}).
4. Whisker change point (L5) vs placebo splits, R+: excess {num('k_l5', L5['R+'].mean_excess, '{:+.3f}')} (n = {int(L5['R+'].n)}, p {pv('k_l5_p', L5['R+'].p_wilcoxon)}); cohort-label permutation p {pv('k_c131', C131.p_perm)} ({ref('fig-placebo')}).
5. Single-trial margins vs learning curve: R+ whisker vs no-stimulus margin rises with the whisker curve (excess r {num('k_w127', W127.mean_excess, '{:+.3f}')}, p {pv('k_w127_p', W127.p_wilcoxon)}); R− hit / miss margin falls ({num('k_h127', H127['R-'].mean_excess, '{:+.3f}')}, p {pv('k_h127_p', H127['R-'].p_wilcoxon)}) ({ref('fig-margin')}).

## Introduction

In the single-session-learning task, head-fixed mice that already lick for an auditory tone learn, within one session, whether to
lick after a whisker deflection. The cohorts receive identical stimuli and differ only in whether a whisker lick is rewarded (R+) or
not (R−). The same overt action is therefore the trained response in R+ and an error in R−. We ask (i) how much information about the
upcoming choice and the stimulus modality whole-brain and area-level populations carry over the session, (ii) whether this
information changes within the session, and when, and (iii) whether single-trial decoder outputs follow the behavioural learning
curve. The stimulus-modality decoding is the non-choice control at each level, and every level is shown for both cohorts and for
area groups. Every claim, with its figure, statistic and status, is listed in {ref('tbl-registry')}.

## Methods

{R.methods('task', 'learners', 'trials', 'lick_alignment')}
Day-0 sessions of all mice (one per mouse) unless stated; units are good and multi-unit clusters of the whole brain or of an area
group (≥ 5 units).

*Decoders.* Standardisation then L2 logistic regression, one $C$ per session and decoding by pooled cross-validation, stratified
$K$-fold cross-validation ($K = \min(5, n_{{\min}})$), balanced accuracy. Time-resolved decoding uses causal 50-ms bins with a 5-ms
stride (stimulus-aligned −200 to 600 ms; lick-aligned −600 to 200 ms). Hit vs miss uses the lick label of whisker trials (the same
label in both cohorts). Window values: baseline −200 to −10 ms, sensory 5–50 ms, pre-lick −100 to 0 ms, post-lick 5–200 ms.

{R.methods('shift_null', 'statistics')}
Group significance of time courses: cluster-mass tests against a group null of mouse-block sign flips between each session's real
and surrogate curves (1,000 draws); cohort differences by two-sided cluster tests with mouse-level cohort-label permutation.

*Splits.* For a split at whisker trial $s$, separate decoders are trained within the pre and post epochs, each subsampled to the
smaller epoch's count per class (50 subsamples). The change is:
{eq('eq-change', r"\Delta_s = \left(a_{\mathrm{post}} - \bar a^{\mathrm{null}}_{\mathrm{post}}\right) - \left(a_{\mathrm{pre}} - \bar a^{\mathrm{null}}_{\mathrm{pre}}\right)")}
where $a$ is the balanced accuracy and $\bar a^{{\mathrm{{null}}}}$ the mean linear-shift null of that epoch. Split definitions: session
half, hit-median split (equal numbers of hits before and after), and learning trials (stored, rule-based, L5 Bayesian change point
of the whisker hit sequence, L6 joint change points, L8 fixed margin; skill ssl-learning-trial-identification). Every other whisker
trial is a placebo split, and the excess of a real split is:
{eq('eq-excess', r"\mathrm{excess}_s = \Delta_s - \frac{1}{|P|}\sum_{k \in P} \Delta_k, \qquad P = \{k : |k - s| \geq 10\}")}
together with the percentile of $\Delta_s$ among the placebo splits (0.5 = a typical split). Cohort differences of the excess are
also tested by permuting cohort labels across mice (20,000 shuffles).

*Single-trial margins.* Nested repeated cross-validation gives each trial a held-out signed margin $m_t$, scaled per session and
class-residualised:
{eq('eq-margin', r"z_t = \frac{m_t}{\sigma_m} - \overline{\left(\frac{m}{\sigma_m}\right)}_{y_t}")}
where $y_t$ is the trial's class, so that a changing class composition cannot create a trend. The correlation of $z$ with the
behavioural learning curve (whisker $P(\mathrm{{lick}})$ or whisker − false alarm, HMM, $\sigma = 1$) is reported as its excess over
200 linear shifts; lagged cross-correlations use the same null with BH correction over lags.

## Results

### Choice and modality information over the whole session

Whole-brain activity 5–50 ms after the whisker stimulus predicted the upcoming lick above the session's null in both cohorts
(R+ {pp_('hs_rp', HS['R+'], 'R+')}; R− {pp_('hs_rm', HS['R-'], 'R-')}), without a cohort difference in that window (Mann–Whitney p
{pv('hc_mw', HC.p_nonparam)}, Welch p {pv('hc_w', HC.p_param)}) ({ref('fig-choice')}d–f). Choice was also decodable before the stimulus
(R+ {num('hb_rp', HB['R+']['mean'], '{:+.3f}')}, R− {num('hb_rm', HB['R-']['mean'], '{:+.3f}')}), so part of it reflects the pre-stimulus
engagement state. The cohort curves diverged only late, from {num('hcl_t0', HCL.t_start_ms.iloc[0] if len(HCL) else np.nan, '{:.0f}')} ms after
the stimulus, when licks are under way ({ref('fig-choice')}e). In R−, choice decoding increased with behavioural d′ across mice (Pearson r
{num('hd_r', HD.r_pearson, '{:+.2f}')}, p {pv('hd_p', HD.p_nonparam)}; {ref('fig-choice')}g). Area groups carried choice information with
similar latencies in the two cohorts and no cohort × area effect ({ref('fig-choice')}h–m).

Whisker and auditory trials were separable from the first bins after stimulus onset (5–50 ms: R+ {num('ms_rp', MS['R+']['mean'], '{:+.3f}')},
R− {num('ms_rm', MS['R-']['mean'], '{:+.3f}')}; {ref('fig-s-stim')}). In the 100 ms before the first lick, licked whisker and auditory trials
were equally decodable in the two cohorts (R+ {pp_('ml_rp', ML['R+'], 'R+')}; R− {pp_('ml_rm', ML['R-'], 'R-')}; Mann–Whitney p
{pv('mlc_mw', MLC.p_nonparam)}). Across area groups, modality information reached half its peak earlier before the lick in R+ (median
{num('lat_rp', lat['R+'], '{:.0f}')} ms) than in R− ({num('lat_rm', lat['R-'], '{:.0f}')} ms). In R+ it was lower in the second half of the session
({num('mh_a', MH['R+'].mean_a, '{:.3f}')} → {num('mh_b', MH['R+'].mean_b, '{:.3f}')}, paired Wilcoxon p {pv('mh_p', MH['R+'].p_nonparam)}), not in R−
(p {pv('mh_rm', MH['R-'].p_nonparam)}) ({ref('fig-premod')}). The high vs low performance state of 5-trial blocks was decodable at a low level
(R+ {num('ps_rp', PS['R+']['mean'], '{:+.3f}')}, R− {num('ps_rm', PS['R-']['mean'], '{:+.3f}')}; {ref('fig-s-perf')}). Window values are collected in
{ref('tbl-wb')}.

### Single-session vs pseudo-population decoding

Single-session decoding is the main estimator: it respects each session's trial structure and has its own linear-shift null.
Pseudo-populations equalise neuron and trial numbers across cohorts; on the session halves they gave the same answer as single
sessions (R+ change {num('pp_rp', P016['delta_R+_med'], '{:+.3f}')}, p {pv('pp_rp_p', P016['p_delta_R+'])}; R− {num('pp_rm', P016['delta_R-_med'], '{:+.3f}')},
p {pv('pp_rm_p', P016['p_delta_R-'])}; 100 iterations, pilot; {ref('fig-s-pp')}). Accuracy grows with the number of units without reaching a
plateau at the session sizes used, which makes raw accuracies of sessions with different unit counts not comparable; all session
values are therefore relative to each session's own null ({ref('fig-s-ncount')}). The time-resolved whole-session pseudo-population
pilot (033: 15 draws, 20-ms stride) saturates at an accuracy of 1 at every time bin, including before the stimulus, and is not
interpretable; it is not used here and needs to be rerun (Open questions).

### Within-session change: session halves

The chronological halves differ in hit rate, so a decoder of the second half of an R+ session is trained on more hits. With the
halves matched in the numbers of hits and misses, choice decoding did not change in either cohort (R+ {num('h_rp_a', H117['R+']['first'], '{:.3f}')}
→ {num('h_rp_b', H117['R+'].second, '{:.3f}')}, p {pv('h_rp_p', H117['R+'].p_wilcoxon)}, n = {int(H117['R+'].n)}; R− {num('h_rm_a', H117['R-']['first'], '{:.3f}')}
→ {num('h_rm_b', H117['R-'].second, '{:.3f}')}, p {pv('h_rm_p', H117['R-'].p_wilcoxon)}, n = {int(H117['R-'].n)}; cohort p
{pv('h_c', H117['R+'].p_cohort_mannwhitney)}; {ref('fig-s-matched')}). Without matching, the time courses, the cross-half generalisation and the
per-area changes showed no cohort difference either ({ref('fig-halves')}).

With the hit-median split (equal numbers of hits before and after), choice decoding did not change in any window or with either
decoder type (smallest p {pv('hm_pmin', hmin)}, uncorrected), while pre-lick modality decoding fell in R+ ({num('m144_a', M144['R+'].mean_a, '{:.3f}')}
→ {num('m144_b', M144['R+'].mean_b, '{:.3f}')}, p {pv('m144_p', M144['R+'].p_nonparam)}), not in R− (p {pv('m144_rm', M144['R-'].p_nonparam)}); cohort
difference p {pv('m144_c', M144['R+ vs R-'].p_nonparam)}. Decoders trained on one half generalised worse to the other half in both cohorts, so
the hit / miss direction changes within the session even where the accuracy does not ({ref('fig-hitmedian')}).

### Within-session change at learning trials

Each learning-trial definition selects its own sessions, so its split is compared with the placebo splits of the same session
({ref('eq-excess')}). In R+, the change of choice decoding at the whisker change point (L5) was larger than at typical splits (percentile
{num('pct_l5', PCT.mean_percentile, '{:.2f}')}, excess {num('l5_rp', L5['R+'].mean_excess, '{:+.3f}')}, Wilcoxon p {pv('l5_rp_p', L5['R+'].p_wilcoxon)},
n = {int(L5['R+'].n)}) and larger than in R− (Mann–Whitney p {pv('l5_mw', L5['R+'].p_mw_cohorts)}; cohort-label permutation p
{pv('c131', C131.p_perm)}). The half split was not special (R+ excess {num('half', HALF.mean_excess, '{:+.3f}')}, p {pv('half_p', HALF.p_wilcoxon)}).
In R−, choice decoding dropped after the fixed-margin learning trial (L8: {num('l8', L8.mean_excess, '{:+.3f}')}, p {pv('l8_p', L8.p_wilcoxon)},
n = {int(L8.n)}) ({ref('fig-placebo')}, {ref('fig-s-perm')}; the abstract version for the stored learning trial is {ref('fig-s-cosyne')}).

The L5 result held in the 5–100 ms window with the R+ change {num('r143_b', R143.mean_b, '{:+.3f}')} vs placebo {num('r143_a', R143.mean_a, '{:+.3f}')}
(p {pv('r143_p', R143.p_nonparam)}; R+ vs R− p {pv('e143', E143.p_nonparam)}, permutation p {pv('p143', P143.p_nonparam)}; {ref('fig-s-l5')}), and it
is carried by step learners: R+ learners without a change point show no change at their midpoint (step vs gradual, 5–50 ms, p
{pv('sg', SG.p_nonparam)}; {ref('fig-s-step')}). The split with the largest change did not sit at the behavioural change point (Spearman
{num('b129', B129.spearman_best_cp, '{:+.2f}')}, permutation p {pv('b129_p', B129.p_perm_best_cp)}), and a step-like neural transition beyond
surrogates was found in few sessions ({int(N130['R+'].n_sig)}/{int(N130['R+'].n)} R+, {int(N130['R-'].n_sig)}/{int(N130['R-'].n)} R−): the change is
a property of the behavioural split, not a sharp neural step.

### Single-trial readout and behaviour

Class-residualised margins ({ref('eq-margin')}) correlated with the learning curve beyond the shift null with cohort-specific signs: the R+
whisker vs no-stimulus margin rose with the whisker curve (excess r {num('w127', W127.mean_excess, '{:+.3f}')}, p {pv('w127_p', W127.p_wilcoxon)},
n = {int(W127.n)}); the R− hit / miss margin fell with it ({num('h127_rm', H127['R-'].mean_excess, '{:+.3f}')}, p {pv('h127_rm_p', H127['R-'].p_wilcoxon)};
R+ {num('h127_rp', H127['R+'].mean_excess, '{:+.3f}')}, p {pv('h127_rp_p', H127['R+'].p_wilcoxon)}; cohort p {pv('h127_c', H127['R-'].p_mw_cohorts)})
({ref('fig-margin')}). Lagged cross-correlations of the hit / miss margin (5–50 ms) with the whisker − false-alarm curve (learners) were
negative in R− at {num('x176_n', len(S176['R-']))} lags after BH correction (lags {num('x176_lo', S176['R-'].lag.min() if len(S176['R-']) else np.nan, '{:.0f}')}
to {num('x176_hi', S176['R-'].lag.max() if len(S176['R-']) else np.nan, '{:.0f}')} trials) and at {num('x176_np', len(S176['R+']))} lags in R+
({ref('fig-s-xcorr')}); because both series drift with time, this coupling is not taken as evidence that the readout leads or follows behaviour.

## Discussion

Choice and modality information is present throughout the learning session in both cohorts and does not differ between them in the
early window after the whisker stimulus, so the contingency does not set how much choice information the early whisker response
carries over the session as a whole. Within the session, coarse splits show no change once the numbers of hits are equalised. A
change appears only when the split is placed at the behavioural change point of R+ step learners and compared with placebo splits of
the same session: choice decoding increases after the point where the mouse starts licking to the whisker stimulus. Pre-lick modality
information declines over the R+ session, consistent with whisker and auditory pre-lick activity becoming more similar as the whisker
lick becomes rewarded (ssl-prelick-convergence). Single-trial margins track the learning curve, but with time-correlated nulls the
direction of influence cannot be resolved.

## Caveats and limitations

- p values are uncorrected across the many decodings, windows and split definitions; the placebo and cohort-permutation tests are the
  protection against split-specific artefacts.
- The learner definition and the learning-trial definition are under review (Methods); L5 sessions are a subset (n in {ref('fig-placebo')}).
- Pseudo-population values are pilots (100 iterations; 033 time-resolved pilot not interpretable).
- 117 (count-matched halves) was run before the exclusion of perf = 6 trials.

## Open questions and next steps

- Rerun the time-resolved pseudo-population (033) with settings that avoid saturation (fewer units per pseudo-population, more draws,
  5-ms stride) and decide the final repetition numbers.
- Settle the learning-trial definition with ssl-learning-trial-identification, then rebuild.

## Figures

{figure('fig-choice', PUB / '113_hitmiss_set1_session_wide.png', '''Choice (hit vs miss) decoding over the whole day-0 session (113 set 1).
**a**, Schematic; **b–c**, example sessions (real vs own null); **d**, whole brain per cohort (mean ± s.e.m.; bars: clusters vs the group
mouse-block null); **e**, R+ − R− (two-sided cluster, mouse-level cohort permutation); **f**, window values (Mann–Whitney | Welch);
**g**, decoding vs behavioural d′ (OLS, 95 % CI; solid if p < 0.05); **h–i**, area overlays; **j**, onset latency per area; **k–l**, area ×
time maps; **m**, per-area R+ vs R− with the two-way ANOVA cohort × area. Values: accuracy minus each session's linear-shift null.''',
                 "113_publication_merged.py")}
{figure('fig-premod', PUB / '110_modality_lick.png', '''Whisker vs auditory decoding aligned to the corrected first lick (110; pre-lick window
−100 to 0 ms): whole-brain curves per cohort with clusters vs the group null, R+ − R− cluster, window values, session halves, area overlays,
onset latencies (50 % of peak, mouse-bootstrap 95 % CI) and area × time maps.''', "110_publication_figures.py")}
{figure('fig-halves', PUB / '113_hitmiss_set2_within_session.png', '''Choice decoding in the first vs second half of the session (113 set 2).
**a**, Schematic; **b–c**, whole-brain time courses per half; **d**, sensory-window values per half (paired Wilcoxon | t; R+ vs R− on the
change); **e**, change over time (paired clusters); **f**, cross-half generalisation; **g–h**, area × time maps of the change; **i–j**, per-area
halves with two-way ANOVA; **k**, per-area change R+ vs R− (ANOVA cohort × area, BH post hoc).''', "113_publication_merged.py")}
{figure('fig-hitmedian', PUB / '144_hitmedian_split_all.png', '''Hit-median split (first half of the session's whisker hits before the split) vs
midpoint split (139 / 144). **a**, Split position; **b**, whisker hit rate before (open) and after (filled); **c**, separate count-matched
decoders and **d**, one whole-session decoder, accuracy minus the within-half shift null per half (circles hit-median, squares midpoint) for
hit vs miss (baseline, 5–35, 5–50, 5–100 ms) and pre-lick modality; **e**, cross-half generalisation. Brackets: paired Wilcoxon | t per
cohort; R+ vs R− on the change (Mann–Whitney | Welch).''', "144_hitmedian_split_figure.py")}
{figure('fig-placebo', FIG / '122b_lt_placebo.png', rf'''Learning-trial splits vs placebo splits (122b). **a**, Percentile of the real change
({ref('eq-change')}) among placebo splits ≥ 10 whisker trials away, per definition and cohort (mean ± s.e.m.; filled: Wilcoxon p < 0.05 vs 0.5;
n and p beside each row), for hit vs miss 5–50 and 5–100 ms and pre-lick modality; **b**, fraction of sessions with percentile ≥ 0.95;
**c**, mean placebo change vs split position, dots: real learning trials.''', "122b_lt_placebo_analysis.py")}
{figure('fig-margin', FIG / '127_margin_vs_learning_curve.png', rf'''Single-trial margin vs learning curve (127): per session, excess correlation
between the class-residualised held-out margin ({ref('eq-margin')}) and the whisker or whisker − false-alarm learning curve, for hit / miss,
whisker vs no-stimulus and whisker vs auditory decoders; dots = mice, bars = cohort mean ± s.e.m.''', "127_margin_vs_learning_curve.py")}

## Supplementary figures and tables

{figure('fig-s-stim', PUB / '113_modality_stim_set1_session_wide.png', 'Whisker vs auditory decoding aligned to stimulus onset, whole session (113 set 1; layout as Figure 1).', "113_publication_merged.py", kind="supplementary")}
{figure('fig-s-perf', PUB / '113_perfstate_set1_session_wide.png', 'Performance-state decoding (high vs low hit rate of 5-trial blocks), whole session (113 set 1; layout as Figure 1).', "113_publication_merged.py", kind="supplementary")}
{figure('fig-s-matched', FIG / '117b_halves_matched_full.png', 'Session halves per session (117b): choice decoding (5–50 ms, accuracy minus shift null) per half, unmatched (left) and matched in hits and misses (right); lines = mice.', "117b_halves_matched_full_figure.py", kind="supplementary")}
{figure('fig-s-pp', PP / 'exploratory-analyses' / 'figures' / '016_pseudopop_halves.png', 'Pseudo-population halves (016; 20 sessions × 10 units per iteration, 100 iterations, pilot): trial-shuffle-corrected balanced accuracy per half, whole brain and area groups.', "ssl-pseudopopulation-area-decoding 016", kind="supplementary")}
{figure('fig-s-ncount', FIG / '073_hitmiss_timeresolved_nsubsample_peakaccuracy.png', 'Choice decoding accuracy vs the number of subsampled units (073): peak accuracy per session and cohort as units are added.', "073_hitmiss_timeresolved_nsubsample_control.py", kind="supplementary")}
{figure('fig-s-l5', PUB / '143_lt_split_L5_windows_all.png', 'Split at the L5 change point per window (5–35, 5–50, 5–100 ms): before vs after above the shift null, real vs placebo change, change beyond placebo (R+ vs R− and cohort-label permutation), and change at every split position relative to the change point.', "143_lt_split_windows_figures.py", kind="supplementary")}
{figure('fig-s-step', PUB / '143_step_vs_gradual_all.png', 'Step learners (L5 change point, split there) vs gradual learners (no change point, split at the midpoint) and non-learners: change above the shift null and beyond placebo per window; Mann–Whitney | Welch.', "143_lt_split_windows_figures.py", kind="supplementary")}
{figure('fig-s-perm', FIG / '131_cohort_permutation_excess.png', 'Cohort-label permutation of the excess over placebo (131): real difference of means R+ − R− (black) and null from 20,000 shuffles across mice (grey), per decoding and definition.', "131_cohort_permutation_prepost.py", kind="supplementary")}
{figure('fig-s-xcorr', FIG / '176_margin_learning_xcorr.png', 'Lagged cross-correlation of the class-residualised hit / miss margin (5–50 ms) with the learning curves (whisker − FA, whisker P(lick); level and first difference), minus the linear-shift null; squares: BH q < 0.05 over lags; right: margin around the learning trial vs random pseudo-learning trials (learners).', "176_margin_learning_xcorr.py", kind="supplementary")}
{figure('fig-s-cosyne', PUB / '138_cosyne_lt_placebo_all.png', 'Conference composite (COSYNE): choice decoding at the stored learning trial vs the shift null and placebo splits (138); its panels re-plot the placebo comparison of Figure 5 for the stored learning trial.', "138_cosyne_lt_placebo_figure.py", kind="conference", panels_from=["fig-placebo"])}
{table('tbl-wb', wb_table(), 'Session-wide whole-brain decoding on day 0: accuracy minus the shift null per cohort (Wilcoxon p) and R+ vs R− (Mann–Whitney | Welch).')}
"""
    claims()
    md += "\n## Appendix\n\n" + R.registry_table("tbl-registry") + APPENDIX
    return md


def claims():
    claim("C1", "Choice decodable 5–50 ms after the whisker stimulus in both cohorts, no cohort difference", "Fig. 1d, f",
          f"R+ {HS['R+']['mean']:+.3f}, R− {HS['R-']['mean']:+.3f}; MW p = {HC.p_nonparam:.2f}")
    claim("C2", "Pre-lick modality equal across cohorts; earlier in R+ areas; declines over the R+ session", "Fig. 2",
          f"MW p = {MLC.p_nonparam:.2f}; R+ half p = {MH['R+'].p_nonparam:.3f}", status="fragile", note="latency from area medians")
    claim("C3", "No change of choice decoding between count-matched halves", "Fig. 3, S3, Fig. 4",
          f"R+ p = {H117['R+'].p_wilcoxon:.2f}, R− p = {H117['R-'].p_wilcoxon:.2f}", status="n.s.")
    claim("C4", "R+ choice decoding increases at the whisker change point beyond placebo, more than R−", "Fig. 5, S6, S8",
          f"excess {L5['R+'].mean_excess:+.3f}, p = {L5['R+'].p_wilcoxon:.3f}; permutation p = {C131.p_perm:.3f}", n=str(int(L5["R+"].n)),
          status="fragile", note="subset of sessions with a change point; uncorrected over definitions")
    claim("C5", "The L5 effect is carried by step learners", "Fig. S7", f"step vs gradual p = {SG.p_nonparam:.3f}", status="fragile")
    claim("C6", "Single-trial margins follow the learning curve with cohort-specific signs", "Fig. 6",
          f"R+ whisker vs no-stim r {W127.mean_excess:+.3f} (p = {W127.p_wilcoxon:.3f}); R− hit / miss {H127['R-'].mean_excess:+.3f} (p = {H127['R-'].p_wilcoxon:.3f})")
    claim("C7", "Readout leads or follows behaviour (lagged cross-correlation)", "Fig. S9", "R− negative coupling at several lags",
          status="fragile", note="time-confounded")
    claim("C8", "Time-resolved pseudo-population pilot (033)", "not shown", "saturates at 1", status="superseded", note="rerun needed")


APPENDIX = """
### Figure versions and provenance

Superseded in this version: Part III (moved to ssl-hitmiss-axis-passive), the hand-numbered tables of the previous report, 119 / 126
definition comparisons (Figure 5 = 122b percentiles; 126 numbers quoted in the text), 120, 123b, 124 / 125, 129, 130 (quoted in the text),
115 halves composite and the deck figures (archived). Pilot scripts 000–023 and their figures are in exploratory-analyses/_archive.
Source files, hashes and dates: figures.json.

### Result folders

combined_results_ks4/ssl-whisker-hitmiss-timeresolved-decoding/: tables (024 sweep parquet files, 1xx stats), figures, figures/publication.
Statistics read by this report are frozen in report/_snapshots/.
"""


def summary():
    return f"""# ssl-whisker-hitmiss-timeresolved-decoding: summary, {date.today().isoformat()}

Question: how much choice and modality information do whole-brain and area populations carry during single-session whisker learning,
does it differ between R+ and R−, and does it change within the session?

1. Choice is decodable 5–50 ms after the whisker stimulus in both cohorts (R+ {HS['R+']['mean']:+.3f}, R− {HS['R-']['mean']:+.3f} above the shift null), no cohort difference early.
2. Pre-lick modality is equal across cohorts; it declines over the R+ session.
3. No change of choice decoding between session halves once hits are equalised.
4. R+ choice decoding increases at the whisker change point beyond placebo splits (fragile: subset of sessions, uncorrected), carried by step learners.
5. Single-trial margins track the learning curve with cohort-specific signs; direction of influence unresolved.

Open: rerun the time-resolved pseudo-population (pilot saturates); learning-trial definition.

![Choice decoding over the session](figures/fig-choice.png)

Full report: report.pdf / report.html; claims registry: registry.csv.
"""


def main():
    md = build()
    R.write(OUT, md, summary=summary())


if __name__ == "__main__":
    main()
