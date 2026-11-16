"""Report of ssl-evoked-kinematics (skills/project-report, report_lib): stimulus-evoked orofacial kinematics in R+ and R-
mice (day 0 and expert), tongue-reach and whisker-sweep trajectories, day-0 whisker movement after the stimulus, and the
day-0 pre-stimulus whisking state that precedes whisker hits. Reads the result tables of 000-006 (no recomputation).
Run (haas, after 000-006):
  cd ~/code/ibl-ai-agent/projects/ssl-evoked-kinematics/report && nice -n 10 ~/code/ibl-ai-agent/.venv/bin/python build_report.py
Render locally (Git Bash, Quarto + TinyTeX): bash projects/ssl-evoked-kinematics/report/render.sh
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(REPO / "scripts")]
from report_lib import Report  # noqa: E402
from axel_bisi_paths import axel_bisi_path  # noqa: E402

ROOT = axel_bisi_path("combined_results_ks4") / "ssl-evoked-kinematics"
TAB, FIG, OUT = ROOT / "tables", ROOT / "figures", ROOT / "report"

R = Report(banned=())
pv, ref, figure, table, eq, claim = R.pv, R.ref, R.figure, R.table, R.eq, R.claim
_num = R.num


def num(*a, **k):
    return _num(*a, **k).replace("-", "−")


R.register([
    ("eq-angle", "Eq. (1)"), ("eq-speed", "Eq. (2)"), ("eq-ext", "Eq. (3)"), ("eq-onset", "Eq. (4)"), ("eq-detrend", "Eq. (5)"),
    ("tbl-n", "Table 1"),
    ("fig-tc", "Figure 1"), ("fig-metrics", "Figure 2"), ("tbl-metrics", "Table 2"), ("fig-tongue", "Figure 3"),
    ("fig-whisker", "Figure 4"), ("fig-sensing", "Figure 5"), ("fig-prestim", "Figure 6"),
    ("fig-s-tc0", "Figure S1"), ("fig-s-controls", "Figure S2"), ("fig-s-spec", "Figure S3"),
    ("tbl-s-params", "Table S1"), ("tbl-s-scripts", "Table S2"), ("tbl-s-registry", "Table S3"),
])

SRC = {
    "n0": TAB / "000_day0_n.csv", "ne": TAB / "000_expert_n.csv",
    "m0": TAB / "000_day0_metrics_mouse.csv", "me": TAB / "000_expert_metrics_mouse.csv",
    "s0": TAB / "000_day0_stats.csv", "se": TAB / "000_expert_stats.csv",
    "002": TAB / "002_day0_whisker_sensing_stats.csv",
    "003g": TAB / "003_day0_glm_stats.csv",
    "004": TAB / "004_day0_time_controls.csv", "004s": TAB / "004_day0_specificity.csv",
    "005w": TAB / "005_day0_prestim_whisking_stats.csv", "005p": TAB / "005_day0_prestim_whisker_posture_stats.csv",
    "005j": TAB / "005_day0_prestim_jaw_posture_stats.csv",
}
FROZEN = dict(zip(SRC, Report.snapshot([SRC[k] for k in SRC], OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}
TR3 = pd.read_parquet(TAB / "003_day0_trials.parquet", columns=["mouse", "session", "cohort", "cls", "lick"])
STAGES = {"day0": "day 0", "expert": "expert"}
COH = ("R+", "R-")
MODS = ("whisker", "auditory")


# ------------------------------------------------------------------------------------------------ lookups
def one(df, **kw):
    d = df
    for k, v in kw.items():
        d = d[d[k] == v]
    assert len(d) == 1, (kw, len(d))
    return d.iloc[0]


def mstat(stage, col, c, mod):
    v = T["m0" if stage == "day0" else "me"]
    v = v[(v.cohort == c) & (v["mod"] == mod)][col].dropna()
    return v.mean(), v.sem(), len(v)


def ms(key, stage, col, c, mod, fmt="{:.3f}"):
    m, s, _ = mstat(stage, col, c, mod)
    return f"{num(key, m, fmt)} ± {num(key + '_sem', s, fmt)}"


def p000(key, stage, col, test, subset):
    r = one(T["s0" if stage == "day0" else "se"], metric=col, test=test, subset=subset)
    return f"p {pv(key + '_np', r.p_mwu)} / {pv(key + '_t', r.p_welch)}"


def p002(key, metric, test, subset):
    r = one(T["002"], metric=metric, test=test, subset=subset)
    return f"p {pv(key + '_np', r.p_nonparam)} / {pv(key + '_t', r.p_param)}"


def v002(key, metric, test, subset, which="mean1", fmt="{:.1f}"):
    return num(key, one(T["002"], metric=metric, test=test, subset=subset)[which], fmt)


def r004(key, cls, var, c, meth, fmt="{:.1f}"):
    r = one(T["004"], cls=cls, var=var, cohort=c, method=meth)
    return (f"{num(key, r['mean'], fmt)} (n = {num(key + '_n', int(r.n_mice))} mice, p {pv(key + '_np', r.p_wilcoxon)} / "
            f"{pv(key + '_t', r.p_t)})")


def r005(key, tab, c, value, comp, fmt="{:.1f}"):
    r = one(T[tab], cohort=c, value=value, comparison=comp)
    return (f"{num(key + '_a', r.mean_a, fmt)} vs {num(key + '_b', r.mean_b, fmt)} (n = {num(key + '_n', int(r.n_mice))} "
            f"mice, p {pv(key + '_np', r.p_wilcoxon)} / {pv(key + '_t', r.p_paired_t)})")


def glm(key, cls, c, pred, fmt="{:.2f}"):
    r = one(T["003g"], cls=cls, model="full", cohort=c, predictor=pred)
    return f"{num(key, r.mean_coef, fmt)} (n = {num(key + '_n', int(r.n_mice))}, p {pv(key + '_np', r.p_wilcoxon)} / {pv(key + '_t', r.p_t)})"


def spec(key, c, term, fmt="{:.2f}"):
    d = T["004s"]
    v = d[(d.pupil == False) & (d.cohort == c)][term].dropna()  # noqa: E712
    return (f"{num(key, v.mean(), fmt)} (n = {num(key + '_n', len(v))}, p {pv(key + '_np', stats.wilcoxon(v).pvalue)} / "
            f"{pv(key + '_t', stats.ttest_1samp(v, 0).pvalue)})")


def nrow(stage, c, tt, col):
    d = T["n0" if stage == "day0" else "ne"]
    return int(one(d, cohort=c, trial_type=tt)[col])


# ------------------------------------------------------------------------------------------------ tables
def tbl_n():
    rows = []
    for st, lab in STAGES.items():
        for c in COH:
            r = dict(stage=lab, cohort=c, mice=nrow(st, c, "whisker hit", "mice"), sessions=nrow(st, c, "whisker miss", "sessions"))
            for tt in ("whisker hit", "whisker miss", "auditory hit"):
                r[f"{tt} trials"] = nrow(st, c, tt, "trials")
                r[f"{tt} first reaches"] = nrow(st, c, tt, "reaches")
            rows.append(r)
    for c in COH:
        d = TR3[TR3.cohort == c]
        rows.append({"stage": "day 0, pre-stimulus (003-005)", "cohort": c, "mice": d.mouse.nunique(), "sessions": d.session.nunique(),
                     "whisker hit trials": int(((d.cls == "whisker") & (d.lick == 1)).sum()),
                     "whisker miss trials": int(((d.cls == "whisker") & (d.lick == 0)).sum()),
                     "auditory hit trials": int(((d.cls == "auditory") & (d.lick == 1)).sum())})
    return pd.DataFrame(rows)


METRIC_LAB = [("reach_latency", "first reach latency (s)", "{:.3f}"), ("lat_jaw", "jaw onset latency (s)", "{:.3f}"),
              ("reach_duration", "reach duration (s)", "{:.3f}"), ("reach_max_ext", "max tongue extension (mm)", "{:.2f}"),
              ("reach_peak_speed", "peak tongue speed (mm/s)", "{:.0f}"), ("wh_amp", "whisker sweep amplitude (deg)", "{:.1f}")]


def fmt_p(a, b):
    f = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
    return f"{f(a)} / {f(b)}"


def tbl_metrics():
    rows = []
    for st, slab in STAGES.items():
        s = T["s0" if st == "day0" else "se"]
        for col, lab, fmt in METRIC_LAB:
            r = {"metric": lab, "stage": slab}
            for c in COH:
                for mod in MODS:
                    m, se, n = mstat(st, col, c, mod)
                    r[f"{c} {mod[0].upper()}"] = f"{fmt.format(m)} ± {fmt.format(se)}"
            for mod in MODS:
                x = one(s, metric=col, test="R+ vs R-", subset=mod)
                r[f"R+ vs R− {mod[0].upper()}"] = fmt_p(x.p_mwu, x.p_welch)
            for c in COH:
                x = one(s, metric=col, test="W vs A (paired)", subset=c)
                r[f"W vs A {c}"] = fmt_p(x.p_mwu, x.p_welch)
            rows.append(r)
    return pd.DataFrame(rows)


PARAMS = pd.DataFrame([
    ("stages", "day 0 = whisker_0; expert = whisker_+1 and whisker_+2 (session_description)", "000"),
    ("trial types (000-002)", "active trials; whisker hit (perf 2), whisker miss (perf 0), auditory hit (perf 3); passive and perf 6 excluded", "000"),
    ("trial classes (003-005)", "all active trials with early_lick = 0: whisker, auditory, no stimulus; choice = lick vs no lick", "003"),
    ("alignment", "stimulus time (whisker_stim_time / auditory_stim_time; no_stim_time for no-stimulus trials)", "000, 003"),
    ("head frame", "origin = median nose tip; x = median nose base → nose tip; frames with nose tip likelihood ≥ 0.1 and nose base ≥ 0.5", "000"),
    ("tongue out", "likelihood ≥ 0.6; single frames 0.3-0.6 inside a run bridged", "000"),
    ("first reach", "first tongue-out run starting 0-1 s after the stimulus, 3 frames to 0.3 s long", "000"),
    ("whisker frame validity", "base and particle likelihood ≥ 0.1, tip ≥ 0.5, both segment lengths 0.6-1.6 × session median; 3-frame running median", "000, 003"),
    ("time courses", "-0.5 to 1.5 s, 10-ms bins; postures baseline-subtracted (-0.5 to 0 s)", "000"),
    ("onset latency", "first of 3 consecutive 10-ms bins above 5 × session median baseline SD, 0-1 s", "000"),
    ("whisker sweep amplitude", "max |Δθ| 0-0.3 s", "000"),
    ("day-0 early window", "30-100 ms after the stimulus (0-30 ms whisker stimulus transient excluded)", "002"),
    ("pre-stimulus windows", "far -1 to -0.1 s; pre -100 to -30 ms; peri -30 to 0 ms; 5-ms bins", "003"),
    ("detrending", "trial minus mean of trials within ±5 trials (004: same class; 005: any active trial)", "004, 005"),
    ("local shuffle", "lick labels permuted within blocks of 8 same-class trials, 1,000 permutations", "004"),
    ("lick-onset exclusion", "tongue out or jaw opening above the session's no-lick 95th percentile in -100 to 0 ms", "004, 005"),
    ("logistic models", "per session, z-scored predictors, L2 (C = 1), sessions with ≥ 30 trials (003) / ≥ 60 trials (004)", "003, 004"),
    ("statistics", "per-mouse values; R+ vs R−: Mann-Whitney U and Welch t; within mouse: Wilcoxon signed-rank and paired t; uncorrected", "all"),
], columns=["parameter", "value", "script"])

SCRIPTS = pd.DataFrame([
    ("000_evoked_kinematics.py", "per-session cache (time courses, reaches, whisker skeletons, per-trial metrics); 000 figures; metrics and stats tables", "cache/000v1_*, tables/000_*"),
    ("001_trajectory_figures.py", "publication-style time courses, spout-centred reaches, whisker skeletons, example mice", "figures/001_*"),
    ("002_day0_whisker_sensing.py", "day-0 whisker movement after the stimulus by trial type; reach-aligned whisker angle", "figures/002_*, tables/002_*"),
    ("003_day0_prestim_state.py", "day-0 single-trial pre-stimulus signals (cache), lick vs no-lick windows, quintiles, per-session logistic models", "cache/003v1_*, tables/003_*, figures/003_*"),
    ("004_day0_prestim_time_controls.py", "time controls (stratified, detrended, matched, local shuffle) and trial-class specificity", "tables/004_*, figures/004_*"),
    ("005_day0_prestim_whisking_figure.py", "summary figures of pre-stimulus whisking, whisker posture and jaw posture", "tables/005_*, figures/005_*"),
    ("006_metrics_summary_figure.py", "per-mouse metric summary, day 0 and expert", "figures/006_*"),
    ("report/build_report.py", "this report", "report/"),
], columns=["script", "content", "outputs"])


# ------------------------------------------------------------------------------------------------ text
def main():
    n0R, n0M = nrow("day0", "R+", "whisker hit", "mice"), nrow("day0", "R-", "whisker hit", "mice")
    neR, neM = nrow("expert", "R+", "whisker hit", "mice"), nrow("expert", "R-", "whisker hit", "mice")
    seR, seM = nrow("expert", "R+", "whisker miss", "sessions"), nrow("expert", "R-", "whisker miss", "sessions")
    md = rf"""---
title: "Stimulus-evoked orofacial kinematics in R+ and R− mice"
subtitle: "ssl-evoked-kinematics -- exploratory report"
date: "{date.today().isoformat()}"
---

**Data version.** NWB_ks4 files (DeepLabCut series stored in the NWB, side and top cameras, mm via the NWB conversion
factor); cohort per mouse from `joint_mouse_reference_weight.xlsx` (R+ / R−, exclude = 1 dropped). Day 0:
{num('n0R', n0R)} R+ and {num('n0M', n0M)} R− mice; expert (whisker days +1, +2): {num('neR', neR)} R+ mice
({num('seR', seR)} sessions) and {num('neM', neM)} R− mice ({num('seM', seM)} sessions). All results are exploratory;
p values are uncorrected. Every claim is listed with its status in the claims registry ([Table S3](#tbl-s-registry)).

# Abstract

We describe the orofacial movements that follow whisker and auditory stimuli in mice whose whisker licks were rewarded
(R+) or not (R−), on the first whisker day (day 0) and in expert sessions, from the video tracking stored in the NWB
files: jaw opening, tongue reaches and nose movement from the side camera, and a three-point whisker skeleton from the
top camera, all in a head-centred frame in mm. On day 0, the evoked movements of the two cohorts were alike: whisker hits
were slower than auditory hits, but once started the tongue reach had the same duration and speed. In experts, R+ and R−
auditory hits remained similar, whereas R− whisker hits, which are not rewarded, came with a later and shorter tongue
reach, a smaller jaw opening and a smaller whisker protraction than R+ whisker hits. The whisker protraction in the first
100 ms after a whisker stimulus on day 0 was as large on auditory hits as on whisker hits, so it belongs to the lick
movement rather than to a whisker-specific sensing response. Before the stimulus on day 0, whisker hits were preceded by
less whisking and a slightly more retracted whisker than whisker misses, in both cohorts; this difference survived four
controls for slow drift within the session and a per-session regression on trial history, but it was also present before
licks on auditory and no-stimulus trials, so it marks a general readiness to lick rather than a whisker-specific state.

**Key results.**

1. Day 0, no cohort difference in the evoked movements; whisker hits are slower than auditory hits (first-reach latency
   R+ {ms('k1a', 'day0', 'reach_latency', 'R+', 'whisker')} vs {ms('k1b', 'day0', 'reach_latency', 'R+', 'auditory')} s,
   {p000('k1p', 'day0', 'reach_latency', 'W vs A (paired)', 'R+')}, n = {num('k1n', n0R)} mice) ([Figure 2](#fig-metrics)).
2. Expert, R− whisker hits have a later first reach ({ms('k2a', 'expert', 'reach_latency', 'R-', 'whisker')} vs R+
   {ms('k2b', 'expert', 'reach_latency', 'R+', 'whisker')} s, {p000('k2p', 'expert', 'reach_latency', 'R+ vs R-', 'whisker')}) and a
   shorter tongue extension ({ms('k2c', 'expert', 'reach_max_ext', 'R-', 'whisker', '{:.2f}')} vs
   {ms('k2d', 'expert', 'reach_max_ext', 'R+', 'whisker', '{:.2f}')} mm, {p000('k2q', 'expert', 'reach_max_ext', 'R+ vs R-', 'whisker')})
   ([Figure 1](#fig-tc), [Figure 3](#fig-tongue)).
3. Day 0, the whisker protracts by the same amount 30-100 ms after the stimulus on whisker and auditory hits (R+
   {v002('k3a', 'early_protraction_deg', 'whisker hit vs auditory hit (paired)', 'R+', 'mean1')} vs
   {v002('k3b', 'early_protraction_deg', 'whisker hit vs auditory hit (paired)', 'R+', 'mean2')} deg,
   {p002('k3p', 'early_protraction_deg', 'whisker hit vs auditory hit (paired)', 'R+')}) ([Figure 5](#fig-sensing)).
4. Day 0, whisking 1 s before the stimulus is lower before whisker hits than before misses, detrended difference
   R+ {r004('k4a', 'whisker', 'wh_speed_far', 'R+', 'detrended')} and R− {r004('k4b', 'whisker', 'wh_speed_far', 'R-', 'detrended')} deg/s
   ([Figure 6](#fig-prestim), [Figure S2](#fig-s-controls)).

# Introduction

{R.methods('task').strip()}

Licking is the reported choice in this task, and the movements that accompany it -- jaw opening, tongue protrusion, nose
and whisker movements -- are large signals in the video and in the brain. Two questions motivated this project. First,
do whisker- and auditory-evoked movements differ in their kinematics, and does this depend on whether the whisker lick is
rewarded (2 × 2 design: cohort × stimulus modality, at day 0 and in experts)? Beyond time courses of posture and speed,
we wanted to describe the stereotyped trajectories of the tongue reach and of the whisker sweep in a head-centred frame,
to know how much of a neural difference between conditions could be motor. Second, on day 0, when the whisker
association is being learned, is there a whisker movement specific to the whisker stimulus (an active-sensing response),
and does the whisking state just before the stimulus predict whether the mouse will lick?

# Methods

## Data and inclusion

Sessions are the NWB_ks4 files whose `session_description` is `whisker_0` (day 0) or `whisker_+1` / `whisker_+2`
(expert). Mice are R+ or R− in the reference sheet with exclude ≠ 1 (R+proba mice and the non-SSL mice AB091 and AB096
are not used). AB155_20250208 is dropped (its video spans the joined recording), and the top view of MH025_20250421 and
MH027_20250422 is dropped (failed whisker tracking). Analyses 000-002 use active trials of three types: whisker hits (perf
2), whisker misses (perf 0) and auditory hits (perf 3); auditory misses are too rare. Analyses 003-005 use every active
trial with early_lick = 0 (whisker, auditory and no-stimulus trials; choice = lick or no lick). Passive trials and perf 6
trials are never used. Sample sizes are in [Table 1](#tbl-n).

{table('tbl-n', tbl_n(), 'Sample sizes. Mice, sessions, trials and first reaches (first tongue reach 0-1 s after the stimulus) per stage, cohort and trial type for 000-002 (day 0 and expert), and mice, sessions and trials of the day-0 pre-stimulus analyses 003-005 (sessions with a usable whisker angle). Expert mice contribute one or two sessions.')}

## Signals

DLC series are read with `ephys_utilities.helpers.dlc_helpers` (`load_dlc`, physical units). For each session and camera, a
head frame is defined from the median nose-tip and nose-base positions (origin = nose tip, x axis = nose base → nose tip,
i.e. forward; y = up in the side view, lateral in the top view). The whisker skeleton is whisker_base → top_particle →
whisker_tip (top view); a frame is valid when the base and particle likelihoods are ≥ 0.1, the tip likelihood is ≥ 0.5 and
both segment lengths lie within 0.6-1.6 × their session medians (this rejects swapped or jumped points), followed by a
3-frame running median. The whisker angle at time t is the angle of the base-to-tip vector in the head frame, and its
change from baseline is
{eq('eq-angle', r'\Delta\theta(t) = \operatorname{atan2}\big(y_{tip}(t) - y_{base}(t),\, x_{tip}(t) - x_{base}(t)\big) - \langle \theta \rangle_{[-0.5,\,0)\,\mathrm{s}}')}
where x and y are head-frame coordinates (mm) and the angle brackets are the mean over the pre-stimulus baseline of the
trial. Negative changes are protraction. The angle is computed from the keypoints in the head frame, not taken from the
NWB `whisker_angle` series, and every comparison uses the baseline-subtracted change of [Eq. (1)](#eq-angle) (000-002) or
the angle minus the session median (003-005), so session-specific angle offsets cancel. The whisker angular speed is
{eq('eq-speed', r'\omega(t) = \left| \frac{d\theta}{dt} \right|')}
computed by finite differences on the valid frames (deg/s; [Eq. (2)](#eq-speed)). Jaw opening is the NWB `jaw_distance`
(mm, `dlc_helpers.clean_series`), baseline-subtracted; the jaw angle is not used. The tongue is out when its likelihood is
≥ 0.6; single frames between 0.3 and 0.6 inside a run are bridged. The tongue extension at time t is
{eq('eq-ext', r'e(t) = \left\| \mathbf{p}_{tongue}(t) - \bar{\mathbf{p}}_{jaw} \right\|')}
where $\mathbf{{p}}_{{tongue}}$ is the tongue keypoint (frames with tongue out) and $\bar{{\mathbf{{p}}}}_{{jaw}}$ the median
jaw keypoint over -0.5 to 0 s of the trial ([Eq. (3)](#eq-ext)). A reach is one tongue-out run of 3 frames to 0.3 s;
the first reach starting 0-1 s after the stimulus is analysed (latency, duration, maximal extension, peak tongue speed),
and its trajectory is sampled every 5 ms from reach onset. Reach trajectories are shown relative to the spout (median
spout keypoint, likelihood ≥ 0.5).

## Metrics and statistics

Time courses are computed from -0.5 to 1.5 s around the stimulus in 10-ms bins and averaged over trials, then over the
sessions of a mouse, then over mice (mean ± SEM). The onset latency of a movement (jaw opening, whisker angle, nose
displacement) is
{eq('eq-onset', r't_{on} = \min\left\{ t \in [0, 1]\,\mathrm{s} : x(t + k\Delta) > 5\,\sigma_b \;\; \text{for}\; k = 0, 1, 2 \right\}')}
where x is the baseline-subtracted signal in 10-ms bins (Δ = 10 ms) and σ_b the session median of the per-trial baseline
standard deviation ([Eq. (4)](#eq-onset)). The whisker sweep amplitude is the maximum of $|\Delta\theta|$ ([Eq. (1)](#eq-angle))
from 0 to 0.3 s. Per-trial metrics are reduced to the median over the hit trials of a mouse.

On day 0, the evoked whisker movement is measured 30-100 ms after the stimulus (after the 0-30 ms window in which the
whisker stimulus itself deflects the whisker, and before the jaw opens): mean $\Delta\theta$ and mean ω ([Eq. (2)](#eq-speed))
minus its baseline. The pre-stimulus state uses single-trial signals in 5-ms bins from -1 to +0.1 s, averaged in the far
window (-1 to -0.1 s; the stimulus marks trial start and falls in the no-lick quiet window). Slow drift within the session
is removed in four ways (004): stratification by session fifths, local detrending, matching each lick trial to the
nearest no-lick trial of the same class within 5 trials, and a local shuffle (labels permuted within blocks of 8
same-class trials, 1,000 permutations, z of the observed difference). The detrended value of trial i is
{eq('eq-detrend', r'\tilde{x}_i = x_i - \frac{1}{|N_i|} \sum_{j \in N_i} x_j')}
where $N_i$ are the trials within ± 5 trials of i (itself excluded; same class in 004, any active trial in 005;
[Eq. (5)](#eq-detrend)). Trials in which a lick was already starting before the stimulus (tongue out, or jaw opening above
the session's no-lick 95th percentile in -100 to 0 ms) are removed in a robustness check (004) and in 005. Per-session
logistic regressions (z-scored predictors, L2) relate the choice to pre-stimulus whisking, posture, other movements, trial
history (previous lick, previous reward, recent lick rate), session progression and pupil (003, 004).

Statistics are computed on per-mouse values (expert: mean over the mouse's sessions). R+ vs R− comparisons use the
Mann-Whitney U test and Welch's t test; within-mouse comparisons use the Wilcoxon signed-rank and paired t tests. Both p
values are given as non-parametric / parametric. No correction for multiple comparisons. Parameters are listed in
[Table S1](#tbl-s-params) and scripts in [Table S2](#tbl-s-scripts).

# Results

## Expert R− whisker hits come with smaller, later movements; auditory hits are alike across cohorts

In experts, the auditory-evoked movements were nearly identical in R+ and R− mice: jaw opening, tongue-out probability,
nose and whisker movements overlapped ([Figure 1](#fig-tc), right). On whisker hits the movements of R− mice were
smaller: the jaw opened less, the tongue was out less often after the first reach, and the whisker protracted less than
in R+ mice ([Figure 1](#fig-tc), left). The jaw onset was later on R− whisker hits
({ms('jl_m', 'expert', 'lat_jaw', 'R-', 'whisker')} vs R+ {ms('jl_p', 'expert', 'lat_jaw', 'R+', 'whisker')} s,
{p000('jl', 'expert', 'lat_jaw', 'R+ vs R-', 'whisker')}) and the whisker sweep amplitude smaller
({ms('wa_m', 'expert', 'wh_amp', 'R-', 'whisker', '{:.1f}')} vs {ms('wa_p', 'expert', 'wh_amp', 'R+', 'whisker', '{:.1f}')} deg,
{p000('wa', 'expert', 'wh_amp', 'R+ vs R-', 'whisker')}; [Figure 2](#fig-metrics), [Table 2](#tbl-metrics)). On day 0 the
time courses of the two cohorts were alike ([Figure S1](#fig-s-tc0)). Because R− whisker hits are not rewarded, the
movements after the first lick differ by design (no water to collect); the differences on the first reach itself (next
section) are the ones that are not a direct consequence of reward delivery.

{figure('fig-tc', FIG / '001_expert_timecourses.png', f'Evoked orofacial kinematics in expert sessions (whisker days +1, +2). Left, whisker hits; right, auditory hits. Rows: jaw opening (mm, baseline-subtracted), probability of tongue out, nose displacement (side view, mm), nose speed (side, mm/s), whisker angle change (deg, negative = protraction; [Eq. (1)](#eq-angle)), whisker angular speed (deg/s, [Eq. (2)](#eq-speed)), nose speed (top view). Mean ± SEM across mice (R+ #00B400, {neR} mice; R− #C800C8, {neM} mice; each mouse = mean over its sessions); 10-ms bins. On whisker trials the whisker transient at 0-30 ms is the stimulus deflecting the whisker.', script='001_trajectory_figures.py')}

## First-reach metrics: modality differences on day 0, cohort differences on whisker hits in experts

On day 0, in both cohorts, the first reach and the jaw opening started later after a whisker than after an auditory
stimulus (jaw onset R+ {ms('j0w', 'day0', 'lat_jaw', 'R+', 'whisker')} vs {ms('j0a', 'day0', 'lat_jaw', 'R+', 'auditory')} s,
{p000('j0', 'day0', 'lat_jaw', 'W vs A (paired)', 'R+')}; R− {p000('j0m', 'day0', 'lat_jaw', 'W vs A (paired)', 'R-')}), but the
reach itself had the same duration and peak speed ([Figure 2](#fig-metrics), [Table 2](#tbl-metrics)); the maximal
extension was slightly shorter on whisker hits (R+ {p000('x0', 'day0', 'reach_max_ext', 'W vs A (paired)', 'R+')}). No metric
differed between cohorts on day 0 with both tests (first-reach latency, whisker hits:
{p000('l0c', 'day0', 'reach_latency', 'R+ vs R-', 'whisker')}).

In experts, R+ latencies shortened while R− whisker-hit latencies stayed long, so the cohorts differed on whisker hits
(first-reach latency {p000('le', 'expert', 'reach_latency', 'R+ vs R-', 'whisker')}; maximal extension
{p000('xe', 'expert', 'reach_max_ext', 'R+ vs R-', 'whisker')}). Smaller differences in the same direction appeared on
auditory hits (latency {ms('lea_p', 'expert', 'reach_latency', 'R+', 'auditory')} vs {ms('lea_m', 'expert', 'reach_latency', 'R-', 'auditory')} s,
{p000('lea', 'expert', 'reach_latency', 'R+ vs R-', 'auditory')}; extension {p000('xea', 'expert', 'reach_max_ext', 'R+ vs R-', 'auditory')}).
Reach duration did not differ in any comparison.

{figure('fig-metrics', FIG / '006_metrics_summary.png', f'Per-mouse evoked-movement metrics, day 0 and expert. Each panel: day 0 then expert; within a stage R+ whisker hits, R+ auditory hits, R− whisker hits, R− auditory hits (R+ #00B400, R− #C800C8; filled circles = whisker hits, open squares = auditory hits). Dots = mice (median over the mouse\'s hit trials; expert = sessions pooled per mouse), black = mean ± SEM across mice. Metrics: first-reach latency, jaw onset latency ([Eq. (4)](#eq-onset)), reach duration, maximal tongue extension ([Eq. (3)](#eq-ext)), peak tongue speed, whisker sweep amplitude (max |Δθ| 0-0.3 s). Day 0: {n0R} R+, {n0M} R− mice; expert: {neR} R+, {neM} R− mice (fewer for whisker metrics, [Table 2](#tbl-metrics)). Tests in [Table 2](#tbl-metrics).', script='006_metrics_summary_figure.py')}

{table('tbl-metrics', tbl_metrics(), 'Evoked-movement metrics per stage, cohort and modality (W = whisker hits, A = auditory hits): mean ± SEM across mice of the per-mouse medians, and p values (non-parametric / parametric) for R+ vs R− (Mann-Whitney U / Welch t) and whisker vs auditory hits within cohort (Wilcoxon / paired t). Uncorrected. Source: tables/000_<stage>_stats.csv.')}

## The tongue reach: a stereotyped forward stroke that differs on R− whisker hits

The first reach was a forward stroke from behind the spout towards it, reaching its closest point to the spout within
5-15 ms of onset and then retracting ([Figure 3](#fig-tongue)). In experts its path was the same on R+ whisker and auditory
hits and on R− auditory hits. On R− whisker hits the mean path ended behind the spout, the maximal extension from the jaw
was shorter ([Table 2](#tbl-metrics)), and the tongue did not retract from the spout region as quickly as on the other trial
types ([Figure 3](#fig-tongue), right). Day-0 reaches have the same shape (figures/001_day0_tongue.png).

{figure('fig-tongue', FIG / '001_expert_tongue.png', 'First tongue reach after the stimulus in expert sessions, side view, spout-centred (origin = spout of the session; axes = head frame, forward and up, mm). Left and middle: thin lines = per-mouse mean reach (whisker hits, auditory hits; R+ top, R− bottom); dots = mean across mice every 5 ms from reach onset (colour = time, 0-80 ms) ± SEM (grey); steps shown where ≥ 50 % of reaches are still out. Right: distance from tongue tip to spout vs time from reach onset, mean ± SEM across mice. Numbers of mice and reaches in the panels and in [Table 1](#tbl-n).', script='001_trajectory_figures.py')}

## The whisker sweep: protraction of the whole shaft after the stimulus, larger on R+ whisker hits

The mean whisker skeleton moved forward as a whole within 50 ms of the stimulus and stayed protracted for at least
300 ms on hit trials ([Figure 4](#fig-whisker)). In experts the protraction was largest on R+ whisker hits, intermediate
on auditory hits, and smallest on R− whisker hits, matching the time courses ([Figure 1](#fig-tc)) and the sweep amplitude
([Figure 2](#fig-metrics)). Fewer mice contribute to the skeleton than to the time courses because a trial needs ≥ 80 %
valid frames of all three points.

{figure('fig-whisker', FIG / '001_expert_whisker.png', 'Whisker sweep in expert sessions, top view in the image orientation (forward = down, lateral = left, mm). Mean skeleton across mice (square = whisker base, circle = mid-shaft particle, triangle = tip) at -50, 0, 50, 100, 200 and 300 ms from the stimulus (colour); each mouse centred on its baseline whisker base. Black line = mean tip path 0-300 ms; thin lines = per-mouse tip paths. Trials with ≥ 80 % valid frames for all three points (validity rules in Methods); sessions with ≥ 3 such trials. Mice and trials in the panels.', script='001_trajectory_figures.py')}

## Day 0: no whisker-specific sensing movement after the stimulus

On day 0, the whisker protracted 30-100 ms after the stimulus on whisker hits (R+
{v002('s1', 'early_protraction_deg', 'vs 0', 'R+ whisker hit', 'mean1')} deg, R−
{v002('s2', 'early_protraction_deg', 'vs 0', 'R- whisker hit', 'mean1')} deg), but by the same amount on auditory hits (R+
{p002('s3', 'early_protraction_deg', 'whisker hit vs auditory hit (paired)', 'R+')}; R−
{p002('s4', 'early_protraction_deg', 'whisker hit vs auditory hit (paired)', 'R-')}), with no cohort difference (whisker hits
{p002('s5', 'early_protraction_deg', 'R+ vs R-', 'whisker hit')}) ([Figure 5](#fig-sensing)). Whisker misses showed a small
protraction (R+ {v002('s6', 'early_protraction_deg', 'vs 0', 'R+ whisker miss', 'mean1')} deg,
{p002('s6p', 'early_protraction_deg', 'vs 0', 'R+ whisker miss')}), smaller than on hits
({p002('s7', 'early_protraction_deg', 'whisker hit vs whisker miss (paired)', 'R+')}), and the same in both cohorts
({p002('s8', 'early_protraction_deg', 'R+ vs R-', 'whisker miss')}). Whisking speed followed the same pattern. Aligned to
the onset of the first reach, the whisker protracted over the 100 ms before the tongue came out on both whisker and
auditory hits ([Figure 5](#fig-sensing), bottom right; not tested). The early whisker movement on hits is therefore part of
the lick movement; we found no movement specific to the whisker stimulus or to the reward contingency.

{figure('fig-sensing', FIG / '002_day0_whisker_sensing.png', 'Day 0 whisker movement by trial type and cohort. Top: whisker angle change (deg, negative = protraction, [Eq. (1)](#eq-angle)); middle: angular speed (deg/s, [Eq. (2)](#eq-speed)); whisker hits (dark blue), whisker misses (light blue), auditory hits (red); mean ± SEM across mice (n in legends). Grey band = 0-30 ms whisker stimulus transient (excluded); yellow = early window 30-100 ms. Bottom left: per-mouse protraction and speed change (minus baseline) in the early window (lines = mice, dots = mean ± SEM); p values in the panel titles are Wilcoxon / Mann-Whitney (parametric tests in tables/002_day0_whisker_sensing_stats.csv). Bottom right: whisker angle change aligned to the onset of the first tongue reach (reaches 0.15-0.35 s after the stimulus; single-trial skeleton angles, mean ± SEM across mice); untested.', script='002_day0_whisker_sensing.py')}

## Day 0: less whisking before whisker hits than before misses, a general readiness-to-lick state

In the second before the stimulus on day 0, whisker hits were preceded by lower whisking than whisker misses in both
cohorts (per-mouse mean speed, whisker hits vs misses: R+ {r005('w1', '005w', 'R+', 'raw', 'whisker hit vs whisker miss')} deg/s;
R− {r005('w2', '005w', 'R-', 'raw', 'whisker hit vs whisker miss')} deg/s; [Figure 6](#fig-prestim)). The difference held after
local detrending ([Eq. (5)](#eq-detrend); R+ {r005('w3', '005w', 'R+', 'detrended', 'whisker hit vs whisker miss')}; R−
{r005('w4', '005w', 'R-', 'detrended', 'whisker hit vs whisker miss')}), and with every time control of 004: stratified by session
fifth (R+ {r004('c1', 'whisker', 'wh_speed_far', 'R+', 'stratified')}; R− {r004('c2', 'whisker', 'wh_speed_far', 'R-', 'stratified')}),
matched to neighbouring misses (R+ {r004('c3', 'whisker', 'wh_speed_far', 'R+', 'matched')}; R−
{r004('c4', 'whisker', 'wh_speed_far', 'R-', 'matched')}) and against a local shuffle (z, R+
{r004('c5', 'whisker', 'wh_speed_far', 'R+', 'local shuffle z', '{:.2f}')}; R− {r004('c6', 'whisker', 'wh_speed_far', 'R-', 'local shuffle z', '{:.2f}')})
([Figure S2](#fig-s-controls)). It also held after removing trials in which a lick was already starting (detrended, R+
{r004('c7', 'whisker (no pre-stim lick onset)', 'wh_speed_far', 'R+', 'detrended')}; R−
{r004('c8', 'whisker (no pre-stim lick onset)', 'wh_speed_far', 'R-', 'detrended')}). The whisker was also slightly more retracted
before hits (detrended angle difference, R+ {r004('c9', 'whisker', 'wh_angle_far', 'R+', 'detrended', '{:.2f}')} deg; R−
{r004('c10', 'whisker', 'wh_angle_far', 'R-', 'detrended', '{:.2f}')} deg), whereas jaw posture did not differ after detrending
(R+ {r005('j1', '005j', 'R+', 'detrended', 'whisker hit vs whisker miss', '{:.3f}')} mm). In the 100 ms just before the stimulus
whisking did not differ between hits and misses.

In a per-session logistic regression of the choice on whisker trials with whisking in three windows, posture, nose, jaw,
tongue, session progression, trial history and pupil, the far-window whisking coefficient was negative in both cohorts
(R+ {glm('g1', 'whisker', 'R+', 'wh_speed_far')}; R− {glm('g2', 'whisker', 'R-', 'wh_speed_far')}), and pupil size 1 s before trial
start was not predictive (R+ {glm('g3', 'whisker', 'R+', 'pupil_ts')}).

The effect was not specific to whisker trials. Low far-window whisking also preceded false alarms on no-stimulus trials in
R+ mice (detrended {r004('n1', 'no stim', 'wh_speed_far', 'R+', 'detrended')}) and auditory hits in R+ mice
({r004('n2', 'auditory', 'wh_speed_far', 'R+', 'detrended')}), and in the per-session regression on no-stimulus trials the
coefficient was negative in both cohorts (R− {glm('g4', 'no stim', 'R-', 'wh_speed_far')}). In one model over all trial classes
per session ([Figure S3](#fig-s-spec)), the whisking coefficient (whisker trials) was R+ {spec('sp1', 'R+', 'whisk')}, R−
{spec('sp2', 'R-', 'whisk')}, and the interactions with auditory and no-stimulus trials were positive (weaker effect;
R− × auditory {spec('sp3', 'R-', 'whisk_x_aud')}, R− × no stimulus {spec('sp4', 'R-', 'whisk_x_nostim')}; R+ × auditory
{spec('sp5', 'R+', 'whisk_x_aud')}), so the effect is somewhat stronger on whisker trials but present in all classes.

{figure('fig-prestim', FIG / '005_day0_prestim_whisking_sem.png', 'Day 0 whisking in the second before trial start (= stimulus). Left: whisker speed time course on whisker hits, whisker misses and auditory hits (per mouse: mean of log10(speed + 1) over trials, i.e. geometric; then mean ± SEM across mice; 5-ms bins, no smoothing; grey band = far window -1 to -0.1 s). Middle: per-mouse mean speed in the far window (grey lines = mice; coloured = geometric mean ± SEM across mice; open = whisker misses), log axis. Right: same after local detrending ([Eq. (5)](#eq-detrend), any active trial within ± 5 trials). R+ top (#00B400), R− bottom (#C800C8). Trials with a lick already starting before the stimulus removed. Brackets: Wilcoxon / paired t p values across mice, uncorrected.', script='005_day0_prestim_whisking_figure.py')}

# Discussion

Across the 2 × 2 design, the movements evoked by a stimulus are dominated by the lick: jaw opening, tongue reach, nose
movement and whisker protraction all start together, and on day 0 they do not depend on cohort. The modality difference on
day 0 lies in when the lick starts (later after a whisker stimulus), not in how the reach unfolds. In experts, the only
large cohort difference is on R− whisker hits, the trials whose licks are never rewarded: the first reach starts later and
falls short of the spout, the jaw opens less and the whisker protracts less. Together with the later and less frequent
licking after the first reach, this fits a hesitant, low-vigour lick when reward is not expected. The smaller auditory
differences between expert cohorts are in the same direction and may reflect a general reduction in licking vigour in
R− mice.

For neural analyses this means that, at day 0, comparisons between cohorts or between whisker and auditory hits are not
confounded by different reach kinematics once the latency is accounted for (for example by lick- or jaw-aligned windows),
whereas expert R+ vs R− comparisons on whisker hits are confounded by smaller and later movements in R−. The early
whisker protraction on day 0 is a component of the lick movement, so a whisker-trial-specific neural response after
30 ms is unlikely to be explained by a whisker-specific movement.

The pre-stimulus result shows that, on day 0, whisker hits occur when the mouse is whisking less, independently of slow
drifts over the session and of recent trial history. Because the same relation holds, more weakly, before false alarms
and auditory hits, it is better read as a quiet, ready-to-lick state than as a whisker-sensing strategy. It could be a
behavioural correlate of attention or engagement, and it is relevant for neural analyses of hits vs misses: pre-stimulus
activity differences between hits and misses may partly reflect whisking.

# Caveats and limitations

- **Old DLC models.** All kinematics come from the DLC models whose outputs are stored in the NWB files. A new, merged DLC
  model is being evaluated and may lead to re-inference; the results should then be rerun (000 → 006).
- **Failed whisker tracking.** Five top-view sessions have failed whisker tracking in the NWB (keypoints on the background,
  yet most frames pass the dlc_helpers whisker likelihood cutoff of 0.1): MH009_20250207, MH009_20250208 (expert),
  MH025_20250421 (day 0), MH025_20250422 (expert) and MH027_20250422 (day 0). None contributes whisker data here: the top
  views of MH025_20250421 and MH027_20250422 are dropped explicitly; for MH009_20250207 / 0208 the head frame could not be
  built from the top-view nose keypoints, so no whisker series was computed; MH025_20250422 had almost no frame passing the
  stricter skeleton rule of this project (tip likelihood ≥ 0.5 and segment-length check), so it yields no whisker time
  course, sweep amplitude or skeleton. Their side-view (jaw, tongue, nose) data are included. The exclusion of MH009 and
  MH025_20250422 is a consequence of the validity rules, not an explicit list; a future run should drop these top views
  explicitly.
- **Whisker angle convention.** The angle is recomputed from the keypoints in a per-session head frame (negative =
  protraction), not taken from the NWB `whisker_angle`, which increases with protraction and carries a per-session offset.
  All comparisons use baseline-subtracted changes (000-002) or the angle minus the session median (003-005); absolute
  angles are never compared across sessions.
- **Jaw.** Jaw opening is the NWB `jaw_distance` in mm; the jaw angle, whose scale depends on the position of the jaw_ref
  keypoint, is not used.
- **Whisker onset latency.** On whisker trials the whisker onset ([Eq. (4)](#eq-onset)) is dominated by the stimulus
  deflection itself; it is reported in the tables but not interpreted.
- **Unit of analysis.** Expert statistics use mice (sessions averaged per mouse), not sessions as in other expert-stage
  SSL analyses; tests are uncorrected for the many metrics and comparisons.
- **Trial and mouse selection.** The warm-up block and the disengaged tail (rule A1) are not trimmed, and no learners-only
  scope was run; day 0 includes all mice. Expert R− whisker hits are not rewarded, so post-lick movements differ by design.
- **Whisker tracking coverage.** The skeleton validity rule (tip likelihood ≥ 0.5) leaves few valid frames in some
  sessions, which then contribute no whisker data; whisker results rest on fewer mice than tongue and jaw results
  ([Table 2](#tbl-metrics)).

**Next steps.** Rerun with the new DLC model if adopted (with the five failed top views dropped explicitly); compute
expert statistics per session; add a learners-only scope; test whether the hit-vs-miss pre-stimulus whisking difference
is present in neural pre-stimulus activity.

# Supplementary figures

{figure('fig-s-tc0', FIG / '001_day0_timecourses.png', f'Evoked orofacial kinematics on day 0, layout as [Figure 1](#fig-tc) (whisker hits left, auditory hits right; R+ #00B400, {n0R} mice; R− #C800C8, {n0M} mice; mean ± SEM across mice).', script='001_trajectory_figures.py', kind='supplementary')}

{figure('fig-s-controls', FIG / '004_day0_time_controls.png', 'Day 0, whisker trials, far window (-1 to -0.1 s before the stimulus): hit minus miss per mouse for whisker speed (top, deg/s) and whisker angle (bottom, deg), raw and after four controls for slow session structure: stratified by session fifths, detrended ([Eq. (5)](#eq-detrend), same-class trials within ± 5), matched to the nearest miss within 5 trials, and z against a local shuffle (labels permuted within blocks of 8 trials, 1,000 permutations). Dots = mice (R+ #00B400, R− #C800C8), black = mean. p values: Wilcoxon (top line) and one-sample t (bottom line) across mice, uncorrected; n in tables/004_day0_time_controls.csv.', script='004_day0_prestim_time_controls.py', kind='supplementary')}

{figure('fig-s-spec', FIG / '004_day0_specificity.png', 'Day 0: specificity of the pre-stimulus whisking effect. (a) Detrended lick minus no-lick difference in far-window whisker speed per trial class (W = whisker, A = auditory, none = no stimulus) and cohort; dots = mice, black = mean ± SEM; Wilcoxon p across mice. (b, c) One logistic model over all trial classes per session (choice ~ class + z(whisking) + z(whisking) × class + session fraction + previous reward + recent lick rate, without and with pupil 1 s before trial start); coefficients mean ± SEM across mice (R+ #00B400, R− #C800C8); * = Wilcoxon and t both p < 0.05. Positive interactions = weaker whisking effect than on whisker trials.', script='004_day0_prestim_time_controls.py', kind='supplementary')}

# Appendix

{table('tbl-s-params', PARAMS, 'Parameters of the analyses (see Methods).')}

{table('tbl-s-scripts', SCRIPTS, 'Scripts (projects/ssl-evoked-kinematics/exploratory-analyses/) and their outputs (combined_results_ks4/ssl-evoked-kinematics/). Figures not shown in this report remain in figures/: 000_* (first versions of time courses, reaches, skeletons, examples, metrics per stage), 001_*_examples (example mice), 001_day0_tongue / 001_day0_whisker (day-0 trajectories), 003_day0_* (lick vs no-lick time courses, window effects, quintiles, confounds), 005_* posture and SD variants.')}

{R.registry_table('tbl-s-registry')}

**Version history.** 2026-11-05: analyses 000-005. 2026-11-10: project cleaned up (README), summary figure 006 added,
first report.
"""
    claim("C1", "Day 0: no cohort difference in evoked movements", "Figure 2, Table 2", "MWU / Welch per metric", f"{n0R} R+, {n0M} R− mice", "holds")
    claim("C2", "Day 0: whisker hits have later jaw and reach onsets than auditory hits; reach itself similar", "Figure 2", "Wilcoxon / paired t", "mice", "holds")
    claim("C3", "Expert: R− whisker hits have later, shorter reaches, smaller jaw opening and whisker sweep than R+", "Figures 1-4", "MWU / Welch", f"{neR} R+, {neM} R− mice", "holds", "sweep amplitude: fewer mice")
    claim("C4", "Expert: auditory hits differ little between cohorts (latency, extension)", "Figure 2", "MWU / Welch", "mice", "fragile", "small effects, uncorrected")
    claim("C5", "Day 0: early (30-100 ms) whisker protraction equal on whisker and auditory hits (no sensing movement)", "Figure 5", "Wilcoxon / paired t", "mice", "holds")
    claim("C6", "Day 0: less far-window whisking before whisker hits than misses, robust to time controls", "Figure 6, Figure S2", "Wilcoxon / t", "mice", "holds")
    claim("C7", "Day 0: slightly more retracted whisker before hits", "Figure S2", "Wilcoxon / t", "mice", "fragile", "≈1 deg; raw R+ t n.s.")
    claim("C8", "Pre-stimulus whisking effect not whisker-specific (also no-stim / auditory)", "Figure S3", "logistic coefficients, Wilcoxon / t", "mice", "holds")
    claim("C9", "Whisker protracts before reach onset on both modalities", "Figure 5", "none", "mice", "fragile", "not tested")
    summary = f"""# ssl-evoked-kinematics: summary ({date.today().isoformat()})

- Day 0: evoked movements alike in R+ and R−; whisker hits start later than auditory hits, reach itself similar.
- Expert: R− whisker hits (unrewarded) have later, shorter reaches, smaller jaw opening and whisker sweep; auditory hits alike.
- Day 0: the early whisker protraction after the stimulus is part of the lick movement (same on auditory hits).
- Day 0: less whisking in the second before the stimulus precedes whisker hits, robust to time controls, not whisker-specific.

![Figure 1](figures/fig-tc.png)
"""
    for tall in ("fig-tc", "fig-s-tc0"):  # tall time-course figures: keep them within one page
        md = md.replace(f"(figures/{tall}.png){{width=100%}}", f"(figures/{tall}.png){{width=70%}}")
    R.write(OUT, md, summary=summary)


if __name__ == "__main__":
    main()
