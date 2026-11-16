"""Report of ssl-dlc-utils-validation (skills/project-report, report_lib), first version 2026-11-10.

Validation of ephys_utilities dlc_helpers and of the DLC data in the NWB_ks4 files, organised by the six sections of
exploratory-analyses/ (README.md), plus the ssl-dlc-kuma 025 / 030 checks of 2026-11-10 (NWB rebuild from the DLC h5,
likelihood gating, whisker-angle direction, jaw-angle scale). Every number is read from the result tables.
Run (haas): ~/code/ibl-ai-agent/.venv/bin/python projects/ssl-dlc-utils-validation/report/build_report.py
Render (PC, Git Bash): bash projects/ssl-dlc-utils-validation/report/render.sh
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJ = HERE.parent
REPO = PROJ.parents[1]
sys.path[:0] = [str(REPO / "skills" / "project-report"), str(REPO / "scripts")]
from report_lib import Report  # noqa: E402
from axel_bisi_paths import axel_bisi_path  # noqa: E402

RES = axel_bisi_path("combined_results_ks4") / "ssl-dlc-utils-validation"
KUMA = axel_bisi_path("DLCProjects") / "merged_training_2026-11-09" / "pilot_inference"
OUT = RES / "report"
TAB, FIG = RES / "tables", RES / "figures"
SCRIPTS = "projects/ssl-dlc-utils-validation/exploratory-analyses"

R = Report(banned=())
num, ref, figure, table, eq, claim = R.num, R.ref, R.figure, R.table, R.eq, R.claim
R.register([
    ("tbl-defaults", "Table 1"),
    ("eq-ref", "Eq. (1)"), ("fig-scale", "Figure 1"), ("fig-ref", "Figure 2"),
    ("fig-jaw", "Figure 3"), ("fig-kp", "Figure 4"),
    ("eq-lick", "Eq. (2)"), ("fig-lick", "Figure 5"),
    ("fig-top", "Figure 6"), ("fig-whisk", "Figure S1"),
    ("fig-pupil", "Figure 7"), ("fig-overview", "Figure 8"), ("fig-jawscale", "Figure S2"),
    ("tbl-n", "Table S1"), ("tbl-registry", "Table S2"), ("tbl-scripts", "Table S3"),
])

SRC = {k: TAB / f"{k}.csv" for k in (
    "000_units", "000_rates", "001_reference", "002_cleaning", "003_dlc_lick_times", "004_rig_scale",
    "005_jaw_likelihood", "007_keypoint_likelihood", "008_top_orientation", "010_session_metrics", "011_n",
    "014_pupil_sessions", "015_pupil_tracking_metrics", "018_video_dlc_audit_whisker", "019_failed_top_dlc_likelihood",
    "dlc_reference_at_origin")}
SRC.update(rebuild=KUMA / "before_after" / "rebuild_check.csv", failed_top=KUMA / "failed_top" / "failed_top_summary.csv",
           whisk_dir=KUMA / "failed_top" / "whisker_angle_direction.csv")
FROZEN = dict(zip(SRC, Report.snapshot(list(SRC.values()), OUT, tag=date.today().isoformat())))
T = {k: pd.read_csv(v) for k, v in FROZEN.items()}
REFS = json.loads((KUMA / "before_after" / "references.json").read_text())

# ------------------------------------------------------------------------------------------------ numbers
u, rt = T["000_units"], T["000_rates"]
n_s000 = u.session_id.nunique()
slope_lo, slope_hi = rt.slope_vs_diff_x_nominal_rate.min(), rt.slope_vs_diff_x_nominal_rate.max()
tslope_lo, tslope_hi = rt.slope_vs_diff_over_true_dt.min(), rt.slope_vs_diff_over_true_dt.max()
rate_med = rt.true_rate.median()

ro = T["dlc_reference_at_origin"]
n_dlc = int(ro.has_dlc.sum())
n_jaw0, n_nose0, n_tnose0 = int(ro.jaw_ref0.sum()), int(ro.nose_ref0.sum()), int(ro.top_nose_ref0.sum())
r1 = T["001_reference"].set_index(["session_id", "series"])
r_ang = r1.loc[("MH038_20250524_105329", "jaw_angle"), "r_stored_recomputed"]
r_dist = r1.loc[("MH038_20250524_105329", "jaw_distance"), "r_stored_recomputed"]
f_below_mh038 = r1.loc[("MH038_20250524_105329", "jaw_angle"), "frac_below"]

rs = T["004_rig_scale"]
rig = rs.probe_sheet_rig.fillna("").astype(str)
n_s004 = len(rs)
top1 = rs.loc[rig.str.startswith("setup 1"), "top_nose_len_px"].median()
top2 = rs.loc[rig.str.startswith("setup 2"), "top_nose_len_px"].median()
n_rig2 = int(rig.str.startswith("setup 2").sum())
o8 = T["008_top_orientation"]
n_s008 = len(o8)
n_up, n_unclear = int((o8.dlc_orientation == "nose up").sum()), int((o8.dlc_orientation == "unclear").sum())

c2 = T["002_cleaning"]
wmed = c2[c2.series == "whisker_angle"].groupby("step").frac_nan.median()

j5 = T["005_jaw_likelihood"]
low5 = j5[j5.session_id != "MH009_20250206_104829"]
jlow_lo, jlow_hi = low5.frac_below_0p5.min(), low5.frac_below_0p5.max()

k7 = T["007_keypoint_likelihood"]
k7b = k7[k7.bin == "[0,0.1)"]
pup01 = k7b[k7b.group == "pupil"].frac.max()
mid7 = k7[k7.bin.isin(["[0.1,0.3)", "[0.3,0.5)"])].groupby(["session_id", "group"]).frac.sum()
mid7_max = mid7.max()

l3 = T["003_dlc_lick_times"]
fr = l3[l3.method == "frame"].groupby("part").median_ms.median()
of = l3[l3.method == "offset"].groupby("part").median_ms.median()
n_s003 = l3.session_id.nunique()

f19 = T["019_failed_top_dlc_likelihood"]
ref19 = f19[f19.session_id == "MH009_20250206_104829"].iloc[0]
bad19 = f19[f19.session_id != "MH009_20250206_104829"]
wb_bad_max = bad19.frac_whisker_base.max()
ft = T["failed_top"]
pv_old = ft[ft.model == "old"].pct_valid
pv_new = ft[ft.model == "new"].pct_valid
wd = T["whisk_dir"]
rb = T["rebuild"]
rb_max = rb.median_abs_diff_rebuilt_old_vs_nwb.max()
SIDE = [k for k in REFS if k.endswith("_side")]
jr_txt = "; ".join(f"{k[:5]} {k[6:14]}: {num('jr_old_' + k[:5], REFS[k]['old']['jaw_ref_x_median'], '{:.0f}')} vs "
                   f"{num('jr_new_' + k[:5], REFS[k]['new']['jaw_ref_x_median'], '{:.0f}')} px" for k in SIDE)

au = T["018_video_dlc_audit_whisker"]
flags = au["flags"].fillna("").str.split(";").explode()
flags = flags[flags != ""].value_counts()
n_s018 = au.session_id.nunique()

m10 = T["010_session_metrics"]
n11 = T["011_n"]
p14 = T["014_pupil_sessions"]
v14 = p14.verdict.value_counts()
n_rev = len(list(FIG.glob("015_pupil_review_*.png")))

# ------------------------------------------------------------------------------------------------ text
md = f"""---
title: "DLC utilities and NWB DLC data: validation"
subtitle: "ssl-dlc-utils-validation -- ephys_utilities dlc_helpers on NWB_ks4 (AB / MH mice)"
date: "{date.today().isoformat()}"
---

# Abstract

The SSL analyses read DeepLabCut (DLC) tracking from the NWB_ks4 files through one module, `ephys_utilities.helpers.dlc_helpers`.
Before dependent analyses switch to it, this project checked, by looking at the data through the module only, (1) the
units, reference points and camera scale of the stored series, (2) which frames and keypoints are trustworthy and how
the series should be cleaned, (3) how DLC lick times are put on the session clock, (4) the top-view orientation and the
completeness of videos and DLC per session, (5) the pupil, and (6) event-triggered overviews of all sessions.

Key results: the stored series are in physical units per second and the loader does not apply the frame rate twice; the
converter places the jaw reference at the image origin in a large share of sessions, which the loader recomputes; setup 2
has a different camera scale and an upside-down top camera, both now handled in the converter; the jaw is well tracked
even at low DLC likelihood and is not gated; pupil, whisker and nose points are usable from likelihood 0.1; DLC lick
times must be mapped through each frame's camera timestamp, not a single offset; the defaults of `clean_series` return
the stored values (no likelihood mask), except for the pupil, which is cleaned with the pupil QC. Rebuilding the NWB
series from the DLC h5 with the converter reproduces the NWB exactly; five upside-down top-view sessions failed in DLC
and remain unusable for top-view kinematics until they are re-inferred.

# Introduction

DLC data enter the NWB files through NWB_converter (`DLC_to_nwb.py`, `utils/dlc_utils.compute_kinematics_alt`): the
keypoints of the side camera (jaw, jaw reference, nose, tongue, four pupil points) and of the top camera (nose tip and
base, whisker base, whisker particle, spout) are smoothed lightly and turned into distances, angles, areas and their
frame-to-frame velocities, with the camera pulses as timestamps. `dlc_helpers` loads these series, converts units,
fixes known converter issues, cleans and aligns them. The question of this project is whether this chain is correct
and which defaults to adopt; the converter itself is taken as correct unless a check shows otherwise (user decision),
and dependent analyses are switched only after this validation.

The report follows the six sections of the code (`{SCRIPTS}/<section>/`, README.md). Each section states the question,
what was checked, the answer and the decision, with one figure. The adopted defaults are collected in {ref('tbl-defaults')}.

# Methods

Data: NWB_ks4 files of AB and MH mice with DLC (`processing/behavior/BehavioralTimeSeries`), {num('n_dlc', n_dlc)}
sessions with DLC at the time of the reference scan. Detailed checks use a sample of {num('n_s000', n_s000)} sessions
(both rigs, 2023 to 2026, including sessions with alignment clauses; `common.py`), the event overviews all sessions with
DLC and trials, the day-0 analyses one whisker day-0 session per mouse with the cohort from the reference sheet. Sample
sizes per analysis are in {ref('tbl-n')}. All loading goes through `dlc_helpers.load_dlc` (physical units: pixels times the
`conversion` attribute, mm per pixel of each camera; angles and likelihoods unconverted); each series is used with its own
camera's timestamps. Scripts and outputs are listed in {ref('tbl-scripts')}, the claims of the report in {ref('tbl-registry')}. No statistical tests are made; the checks are
descriptive (frame images, distributions, event-locked means).

[]{{#tbl-defaults}}

| Setting | Adopted value in `dlc_helpers` | Evidence |
|:--|:--|:--|
| Units | positions / distances mm, areas mm², angles deg, velocities per second (stored, not rate-multiplied again) | 000 |
| Reference point at the image origin | recomputed at loading over valid frames (`fix_zero_reference`) | 001 |
| Rig scale | setup 1: side 33.4, top 30.4 px/mm; setup 2: side 28.83, top 16.75 px/mm (converter clause `setup2_sessions`) | 004 |
| Default cleaning (`CLEAN_DEFAULTS`) | stored values returned: no likelihood mask, no jump / MAD step, no smoothing; gap interpolation ≤ 50 ms acts only on NaN | 002, 006 (archived) |
| Jaw | never gated by likelihood | 005 |
| Likelihood cutoffs when a mask is requested | 0.1 for pupil, whisker base / particle, side and top nose tip; 0.5 otherwise | 007 |
| Tongue presence | tongue_likelihood > 0.6 | 010, 011 |
| Pupil | pupil QC frame states (blink / squint / failed set to NaN), running median 0.1 s; sessions with QC verdict 'fail' excluded | 013-015 |
| DLC lick times | per-frame camera timestamp (`get_dlc_lick_times(method='frame')`) | 003 |
| Top-view orientation | upside-down top DLC rotated by the converter, decided from the DLC data | 008 |

: **Table 1.** Defaults and decisions adopted in `dlc_helpers` / NWB_converter from this validation, with the script that
supports each.

# Results

## 1. Units, reference point and camera scale

**Units and rates (000).** In the {num('n_s000_b', n_s000)} sample sessions the stored `*_velocity` series equal the
frame-to-frame difference of their base series times the nominal rate (regression slope
{num('slope_lo', slope_lo, '{:.4f}')} in every series and session), and the derivative from the true
timestamps gives slopes of {num('tslope_lo', tslope_lo, '{:.4f}')} to {num('tslope_hi', tslope_hi, '{:.4f}')} (true frame
rate median {num('rate_med', rate_med, '{:.2f}')} Hz). The stored velocities are therefore per second and the loader
must not multiply by the rate again, which it does not. Physical ranges are plausible for every series (`tables/000_units.csv`).
Velocity is the signed derivative of a scalar series, speed its absolute value or the 2-D keypoint speed; the module
keeps these apart (`derivative`, `keypoint_speed`).

**Camera scale (004).** The yaml 'setup' is set per experimenter, not per session, so the rig was identified from the
pixel size of body features that do not change between sessions ({ref('fig-scale')}). Top-view sessions fall into two
clusters: the median top nose length is {num('top1', top1, '{:.0f}')} px in probe-sheet setup-1 sessions and
{num('top2', top2, '{:.0f}')} px in setup-2 sessions ({num('n_rig2', n_rig2)} sessions on the probe sheet), as expected
from the two reference scales (top 30.4 vs 16.75 px/mm). The sessions of the small-scale cluster are also those with an
upside-down top camera (008). Decision: the converter applies the setup-2 scale to these sessions (clause
`setup2_sessions`, reconverted 2026-11-03); the table of {ref('fig-scale')} predates that reconversion, so it still shows
the setup-1 scale stored in every NWB.

{figure('fig-scale', FIG / '004_rig_scale.png', f"Camera scale per session (004). Each point is one NWB session with DLC (n = {n_s004}; blue AB mice, orange MH mice), median pixel distance between keypoints over frames with likelihood > 0.5: top nose tip to base vs whisker base to nose tip (left), side nose length vs pupil equivalent diameter (middle), top vs side nose length (right). Red circles: sessions listed as setup 2 in the ephys probe sheet (n = {n_rig2}); black crosses: top view nose up in 008 (n = {n_up}). Figure regenerated 2026-11-10 from the existing table (the earlier version circled every probe-sheet session).", script='1_units_reference_scale/004_rig_scale.py')}

**Reference point at the image origin (001).** `compute_kinematics_alt` takes the jaw / nose reference point as the 5th
percentile of the keypoint coordinates with low-likelihood frames set to zero, so the reference becomes the image
corner once enough frames are below the cutoff. With $x_t$ the keypoint position in frame $t$, $p_t$ its likelihood and
$c = 0.5$ the converter cutoff, the stored reference is
{eq('eq-ref', r"x_{\mathrm{ref}} = Q_{0.05}\left(\{x_t \,\mathbb{1}[p_t > c]\}_t\right), \qquad x_{\mathrm{ref}} = 0 \ \text{ if more than 5\% of frames have } p_t \le c")}
while `fix_zero_reference` takes the same percentile over the valid frames only. The scan of NWB_ks4 found the jaw
reference at the origin in {num('n_jaw0', n_jaw0)} of {num('n_dlc_b', n_dlc)} DLC sessions (side nose {num('n_nose0', n_nose0)},
top nose {num('n_tnose0', n_tnose0)}; `tables/dlc_reference_at_origin.csv`). Distances from the origin keep the time course
of the jaw (correlation with the recomputed series {num('r_dist', r_dist, '{:.2f}')} in MH038 0524, where
{num('f_below', 100 * f_below_mh038, '{:.0f}')}% of frames are below the cutoff), but angles do not
({num('r_ang', r_ang, '{:.2f}')}) and the lick-locked jaw angle is distorted ({ref('fig-ref')}). Decision: the loader recomputes
the reference ({ref('eq-ref')}, valid frames only) for affected series. The scale of the jaw angle still depends on where
the DLC model places `jaw_ref` (Section 6).

{figure('fig-ref', FIG / '001_reference_MH038_20250524_105329.png', "Jaw reference point at the image origin, MH038_20250524 (001). Top left: jaw keypoint positions (likelihood > 0.5) with the converter reference (red cross, image origin) and the reference over valid frames (green plus). Top middle and right: 20 s of jaw_distance and jaw_angle, stored (red, left axis) vs recomputed with the valid-frame reference (green, right axis). Bottom: mean z-scored jaw_distance and jaw_angle around piezo lick-bout onsets (n = 469 bouts), stored vs recomputed. Table: fraction of frames below likelihood 0.5, medians and frame-by-frame correlation.", script='1_units_reference_scale/001_reference_point.py')}

## 2. Likelihood and cleaning

**Cleaning steps (002).** Each step of `clean_trace` was applied cumulatively (likelihood mask 0.5, frame-to-frame jump,
8 × MAD from the median, interpolation of gaps ≤ 50 ms, Gaussian smoothing 10 ms) and compared on example traces and
event-locked means. The MAD step removed real whisks (median NaN fraction of whisker_angle after the likelihood mask
{num('w_lik', 100 * wmed['likelihood'], '{:.1f}')}%, after MAD {num('w_mad', 100 * wmed['mad'], '{:.1f}')}%) and the jump step removed
almost nothing; neither changed event-locked means for the better. Both are off by default.

**Jaw likelihood (005).** In {num('n_lowjaw', len(low5))} sessions {num('jlow_lo', 100 * jlow_lo, '{:.0f}')}-{num('jlow_hi', 100 * jlow_hi, '{:.0f}')}% of
jaw frames are below likelihood 0.5. Video frames sampled across likelihood bins show the jaw keypoint on the jaw edge at
every likelihood, and the jaw trace follows the piezo licks through low-likelihood stretches ({ref('fig-jaw')}). Low
likelihood here reflects an unusual appearance (fur, lighting), not a wrong placement. Decision: the jaw is never gated by
likelihood (`NO_LIKELIHOOD_KEYPOINTS`).

{figure('fig-jaw', FIG / '005_jaw_frames_AB158_20250415_105500.png', "Jaw keypoint at low likelihood, AB158_20250415 (005). Rows: side-video frames sampled in jaw-likelihood bins [0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 1], cropped around the jaw; red circle DLC jaw, cyan square jaw_ref, yellow triangle nose tip, green circle tongue (if likelihood > 0.5). Bottom: 10 s of jaw_y (black; red dots frames with likelihood < 0.5) and jaw likelihood (orange), piezo licks as blue lines. Other sessions: figures/005_jaw_frames_*.png.", script='2_likelihood_and_cleaning/005_jaw_likelihood_frames.py')}

**Keypoint likelihood cutoffs (007).** For pupil, whisker, top nose and side nose, frames were sampled per likelihood bin
of the gating likelihood (minimum over the group's keypoints) in sessions where the 0.5 cutoff rejected many frames. From
about 0.1 upward the points are on the right features; below 0.1 the eye is closed, the whisker is occluded by the spout
or the points are thrown off ({ref('fig-kp')}). Frames between 0.1 and 0.5, which 0.5 would reject, are up to
{num('mid7', 100 * mid7_max, '{:.0f}')}% of a session's frames for one keypoint group; frames below 0.1 up to
{num('pup01', 100 * pup01, '{:.0f}')}% for the pupil. Decision: `LIKELIHOOD_CUTOFF` = 0.1 for these keypoints, applied only when a
mask is requested; `clean_series` defaults do not gate by likelihood (user, 2026-11-04), since the NWB series come from
DLC filtered h5 files and are already smoothed.

{figure('fig-kp', FIG / '007_whisker_MH062_20260110_113351.png', "Whisker keypoints per likelihood bin, MH062_20260110 top video (007). Rows: frames sampled in bins of the minimum likelihood of whisker_base (red) and top_particle (cyan): [0, 0.1), [0.1, 0.3), [0.3, 0.5), [0.5, 1]; the likelihood of each point is written next to it. Same figures for pupil, top nose and side nose and other sessions: figures/007_*.png; fractions of frames per bin in tables/007_keypoint_likelihood.csv.", script='2_likelihood_and_cleaning/007_keypoint_likelihood_frames.py')}

## 3. DLC lick times

The NWB stores `tongue_dlc_licks` and `jaw_dlc_licks` on the video clock (side frame index divided by the nominal rate).
`get_dlc_lick_times` maps them to the session clock either per frame or with a single offset. With $t^{{v}}_i$ the video
time of lick $i$, $r$ the nominal rate and $T_k$ the camera timestamp of side frame $k$, the frame method is
{eq('eq-lick', r"t_i = T_{\mathrm{round}(r\, t^{v}_i)}, \qquad \text{offset method: } t_i = t^{v}_i + T_0")}
where the frame method absorbs the camera clock running off the nominal rate and missing pulses. Against the nearest piezo
lick ({num('n_s003', n_s003)} sessions, {ref('fig-lick')}), the frame method gives a stable lead over the session (median across sessions
{num('fr1', fr['third1'], '{:.0f}')}, {num('fr2', fr['third2'], '{:.0f}')}, {num('fr3', fr['third3'], '{:.0f}')} ms in the first, second and last third; DLC
detects the tongue before it touches the piezo), while the offset method drifts ({num('of1', of['third1'], '{:.0f}')} to
{num('of3', of['third3'], '{:.0f}')} ms) and fails in sessions whose video does not start on the first pulse. Decision:
`method='frame'` is the default ({ref('eq-lick')}).

{figure('fig-lick', FIG / '003_dlc_lick_times.png', f"DLC lick times on the session clock (003), {n_s003} sample sessions. Left: offset-method minus frame-method lick time over the session (one line per session). Right: median signed difference between each DLC tongue lick and the nearest piezo lick (within 150 ms) per session third; circles frame method, crosses offset method.", script='3_lick_times/003_dlc_lick_times.py')}

## 4. Top view: orientation, failed sessions and the whisker-angle convention

**Orientation (008).** The setup-2 top camera is mounted rotated by 180° (nose up, spout from the top). From the sign of
the median top nose tip minus base y and one frame per session ({num('n_s008', n_s008)} sessions reviewed), {num('n_up', n_up)} sessions are
nose up and {num('n_unclear', n_unclear)} unclear (`tables/008_top_orientation.csv`, figures/008_top_orientation_p*.png). Decision
(converter, since 2026-11-04): the rotation is decided from the DLC data (median tip minus base y below −20 px), so DLC
reruns on videos flipped before inference are not rotated twice. Distances and whisker angle are rotation invariant; top x / y
are rotated back and top_nose_angle changes sign.

**Failed top-view DLC (019).** Five upside-down sessions were tracked with an older model (shuffle 5) on the raw video:
the keypoints sit away from the animal ({ref('fig-top')}); whisker_base passes likelihood 0.6 in at most
{num('wb_bad', 100 * wb_bad_max, '{:.1f}')}% of frames, against {num('wb_ref', 100 * ref19.frac_whisker_base, '{:.0f}')}% in the
well-tracked reference MH009 0206. Because `clean_series` does not gate by likelihood and the likelihoods are low but not
below 0.1 ({num('pv_old_lo', pv_old.min(), '{:.0f}')}-{num('pv_old_hi', pv_old.max(), '{:.0f}')}% of frames pass the whisker cutoff 0.1 in the two sessions
re-inferred; merged model {num('pv_new_lo', pv_new.min(), '{:.1f}')}%), these sessions are not removed automatically: they must be
excluded explicitly or replaced by the re-inferred tracks (pending, TODO.md).

{figure('fig-top', FIG / '019_failed_top_dlc_frames.png', "Failed top-view DLC (019). Rows: MH009_20250206 (reference, shuffle 6, well tracked) and the five upside-down setup-2 sessions tracked with shuffle 5 (MH009 0207 / 0208, MH025 0421 / 0422, MH027 0422). Columns: raw top-video frames at 300, 1200, 2400, 3600 s with the top keypoints (filled: likelihood ≥ 0.6, open: below); right: fraction of frames with likelihood ≥ 0.6 per keypoint.", script='4_top_view_and_video_audit/019_failed_top_dlc_frames.py')}

**Whisker-angle convention (ssl-dlc-kuma 030).** The stored whisker angle increases with protraction: across four top
videos it is linear in the forward position of the particle relative to the whisker base (r = {num('wd_r_lo', wd.r.min(), '{:.3f}')} to
{num('wd_r_hi', wd.r.max(), '{:.3f}')}, {ref('fig-whisk')}). Its absolute value carries a per-session offset of twice the nose tilt in the
image. The user keeps this convention: analyses subtract a per-trial baseline, which removes the offset.

**Reproducibility of the NWB series (ssl-dlc-kuma 025).** Rebuilding the series from the old DLC h5 files with the
converter steps (smoothing, orientation check, `compute_kinematics_alt`, real timestamps and conversion) reproduces the
NWB series (largest median absolute difference {num('rb_max', rb_max, '{:.1e}')} in native units over {num('n_rb', len(rb))}
series of {num('n_rb_s', rb.session.nunique())} videos), so before / after comparisons of DLC models isolate the model.

**Video / DLC audit (018).** Over {num('n_s018', n_s018)} whisker-day sessions and both cameras, the audit lists missing DLC
({num('fl_nodlc', int(flags.get('no_dlc', 0)))} session-cameras), frame vs pulse mismatches in the NWB
({num('fl_fp', int(flags.get('nwb_frames_vs_pulses', 0)))}, all known alignment cases), incomplete DLC
({num('fl_inc', int(flags.get('dlc_incomplete', 0)))}), DLC present but absent from the NWB ({num('fl_nwb', int(flags.get('nwb_no_dlc', 0)))}) and
video files named after another mouse ({num('fl_wm', int(flags.get('wrong_mouse_in_name', 0)))}); per-session actions are recorded in
`skills/ssl-valid-data/references/nwb_session_exceptions.md` (`tables/018_video_dlc_issues.csv`).

## 5. Pupil

The stored pupil area has single-frame spikes at blinks, where DLC throws the four points off the closed eye, and some
sessions have long stretches with the eye closed or occluded (013, figures/013_pupil_frames_*.png). Smoothing alone
(archived 012) removes spikes but not these stretches. `clean_pupil` therefore uses the pupil QC of ssl-pupil-qc (blink,
squint and failed frames set to NaN, then a 0.1 s running median) and excludes sessions with QC verdict 'fail'
({num('v_fail', int(v14.get('fail', 0)))} of {num('n_p14', len(p14))} day-0 sessions; {num('v_pass', int(v14.get('pass', 0)))} pass,
{num('v_warn', int(v14.get('warn', 0)))} warn). Without the QC the day-0 stimulus-locked pupil means are dominated by artefacts; with it
they show clear dilations after hits and passive stimuli in both cohorts ({ref('fig-pupil')}). Candidate outlier mice
({num('n_rev', n_rev)} reviewed in 015) were judged on tracking quality only (jumps, no visible pupil), not on response size.

{figure('fig-pupil', FIG / '014_pupil_variants_cohorts.png', "Day-0 pupil area around stimulus onset by trial type (014), mean ± SEM across mice, R+ green (#00B400) and R− magenta (#C800C8); numbers per panel: mice (trials) for R+ / R−. Rows: (A) stored pupil area + 0.1 s running median, no QC, mm²; (B) pupil QC + 0.1 s running median, mm², QC 'fail' sessions excluded; (C) B as % of the session median. Each trial minus its mean in −1..0 s; trials kept if ≥ 80% of −1..4 s is finite; 20 ms bins.", script='5_pupil/014_pupil_qc_clean.py')}

## 6. Event-triggered overviews

Event-locked traces of every main series through the module (default cleaning) for all sessions with DLC and trials
({num('n_s010', m10.session_id.nunique())} sessions, {num('n_m010', m10.mouse.nunique())} mice; 010) show the expected structure in
nearly every session: the jaw opens at lick-bout onsets and after whisker and auditory stimuli in licking trials
({ref('fig-overview')}); rows without a whisker response are expected for R− mice and sessions with few whisker-evoked
licks. Isolated deviant rows (e.g. MH013) are sessions with alignment clauses in `nwb_session_exceptions.md`. The day-0 posture and movement
analysis (011; {num('n_m011p', int(n11[n11.cohort == 'R+'].n_mice.max()))} R+ and {num('n_m011m', int(n11[n11.cohort == 'R-'].n_mice.max()))} R− mice)
shows the same body parts per trial type and cohort (figures/011_*.png).

The jaw angle is not comparable across DLC models: it is measured around a point whose x is the median `jaw_ref`, and
the models place `jaw_ref` differently (median x, old vs merged model, in the side pilot videos: {jr_txt}). In the two AB
videos the angles of the two models differ by a scale factor (slopes in the titles of {ref('fig-jawscale')}); in the MH videos
the stored (NWB) jaw angles also use a reference at the image origin, which `fix_zero_reference` corrects at loading. Jaw
distance is less affected.

{figure('fig-overview', FIG / '010_heatmap_jaw_distance.png', f"Jaw distance around events, all sessions (010). Top: session means (rows, grouped by mouse; z-scored per session, baseline-subtracted) around piezo lick-bout onsets (left), whisker (middle) and auditory (right) stimulus onsets, active trials without perf 6 and passive trials; n = {m10.session_id.nunique()} sessions. Bottom: mouse means (green AB, purple MH) and the grand mean (black). Default cleaning, 10 ms bins. Other signals: figures/010_heatmap_*.png.", script='6_event_overviews/010_event_triggered_overview.py')}

# Discussion

The module reads the NWB DLC series correctly once three converter properties are handled: references at the image
origin (recomputed at loading), the setup-2 scale and orientation (fixed in the converter) and DLC lick times on the video
clock (mapped per frame at loading). The cleaning decisions were made by looking at frames rather than at likelihood
statistics: low likelihood is often a confident placement of an unusual-looking feature, so the defaults keep the stored
values and leave masks to the analyses that need them, with a cutoff of 0.1 where a mask is used.

The remaining problems are in the DLC itself, not in the module: the five failed top-view sessions, sessions with
incomplete or missing DLC (018), and the dependence of the jaw angle on the `jaw_ref` placement of the model. These are
handled by the DLC rerun decision of ssl-dlc-kuma, after which the affected sessions are reconverted.

# Caveats and next steps

- The failed top-view sessions are not removed by any default (no likelihood gating); top-view analyses must exclude them
  until they are re-inferred (pending user decision).
- Jaw angles are only comparable within one DLC model; compare jaw distance across models.
- The whisker angle has a per-session offset (twice the nose tilt); use baseline-subtracted angles only.
- 004's table predates the setup-2 reconversion (its NWB scale column shows setup 1 for all sessions).
- Sessions with unresolved alignment (`nwb_session_exceptions.md`, unresolved DLC sessions) are not corrected here.
- Next: user decision on the failed top-view sessions; switch dependent analyses to `dlc_helpers`.

# Supplementary figures

{figure('fig-whisk', KUMA / 'failed_top' / 'whisker_angle_direction.png', "Direction of the stored whisker angle (ssl-dlc-kuma 030). Top: whisker angle (NWB convention, compute_kinematics_alt, merged-model tracks) vs the forward position of the particle relative to the whisker base (projection on the session's median nose vector, mm), one panel per top video (MH015 0221, MH025 0421, MH027 0422, MH064 0114), hexbin density; r: Pearson correlation. Bottom: frames at the lowest (blue) and highest (red) angle within one whisking second, with the nose vector (white).", script='projects/ssl-dlc-kuma/030_failed_top_before_after.py', kind='supplementary')}

{figure('fig-jawscale', KUMA / 'before_after' / 'jaw_angle_scale.png', "Jaw-angle scale depends on the jaw_ref placement (ssl-dlc-kuma 025). Top: jaw angle, merged model (after) vs NWB (before), per side pilot video (AB153 0207, AB155 0208, MH062 0109, MH070 0121); dotted line identity, slope and r of the fit in the titles. Middle: the same for jaw distance. Bottom: converter references of each model on a frame (x: median jaw_ref, the angle x reference; +: jaw 5th percentile, the angle y reference). MH062 and MH070 have the old jaw reference at the image origin in the NWB, hence the off-diagonal clouds.", script='projects/ssl-dlc-kuma/025_pilot_before_after.py', kind='supplementary')}

# Appendix

"""

n_rows = pd.DataFrame([
    ("000 units, rates; 001 reference; 002 cleaning; 003 lick times", f"{n_s000} sample sessions (001: 3, 002: 3)", "common.SESSIONS"),
    ("Reference-at-origin scan", f"{n_dlc} sessions with DLC", "all NWB_ks4"),
    ("004 rig scale", f"{n_s004} sessions", "all NWB_ks4 with DLC"),
    ("005 jaw frames; 007 keypoint frames", f"{len(j5)}; {k7.session_id.nunique()} sessions", "low-likelihood sessions + control"),
    ("008 top orientation", f"{n_s008} sessions", "MH sessions with DLC + 6 AB references"),
    ("010 event overview", f"{m10.session_id.nunique()} sessions, {m10.mouse.nunique()} mice", "all with DLC and trials"),
    ("011 day-0 posture / movement", f"{int(n11[n11.cohort == 'R+'].n_mice.max())} R+, {int(n11[n11.cohort == 'R-'].n_mice.max())} R- mice", "one whisker day-0 session per mouse"),
    ("014 / 015 pupil", f"{len(p14)} day-0 sessions", "QC verdicts from ssl-pupil-qc"),
    ("018 audit", f"{n_s018} whisker-day sessions x 2 cameras", "AB082 onwards, MH"),
    ("019 failed top view", f"{len(f19)} sessions", "5 failed + 1 reference"),
], columns=["Analysis", "n", "Population"])
md += table("tbl-n", n_rows, "Sample sizes per analysis. Sessions are NWB_ks4 files; mice per the reference sheet.")

claims = [
    ("c1", "Stored velocities are per second; the loader does not re-apply the rate", "000", "slope vs diff x rate"),
    ("c2", "Jaw reference at the image origin in many sessions; angles distorted, recomputed at loading", "Figure 2", "scan counts, r"),
    ("c3", "Setup 2 has a smaller top camera scale; converter applies it", "Figure 1", "median feature size"),
    ("c4", "Jaw well placed at low likelihood; not gated", "Figure 3", "frames"),
    ("c5", "Pupil / whisker / nose usable from likelihood 0.1", "Figure 4", "frames"),
    ("c6", "Frame-based DLC lick times stable; offset method drifts", "Figure 5", "median lag per third"),
    ("c7", "Five upside-down top sessions failed; not removed by defaults", "Figure 6", "likelihood fractions"),
    ("c8", "Whisker angle increases with protraction; per-session offset", "Figure S1", "r"),
    ("c9", "NWB series reproduced exactly from the DLC h5", "text", "median abs diff"),
    ("c10", "QC-based pupil cleaning needed; fail sessions excluded", "Figure 7", "verdict counts"),
    ("c11", "Jaw angle scale depends on jaw_ref placement of the model", "Figure S2", "jaw_ref medians"),
]
for cid, text, fig, stat in claims:
    claim(cid, text, figure=fig, stat=stat, status="holds")
md += R.registry_table("tbl-registry")

md += table("tbl-scripts", pd.DataFrame([
    ("1_units_reference_scale/000, 001, 004", "tables/000_*, 001_*, 004_*, dlc_reference_at_origin.csv; figures/000_*, 001_*, 004_*"),
    ("2_likelihood_and_cleaning/002, 005, 007", "tables/002_*, 005_*, 007_*; figures/002_*, 005_*, 007_*"),
    ("3_lick_times/003", "tables/003_*; figures/003_*"),
    ("4_top_view_and_video_audit/008, 018, 019", "tables/008_*, top_orientation.csv, 018_*, 019_*; figures/008_*, 019_*"),
    ("5_pupil/013, 014, 015", "tables/014_*, 015_*; figures/013_*, 014_*, 015_*; cache/014_*"),
    ("6_event_overviews/010, 011", "tables/010_*, 011_*; figures/010_*, 011_*; cache/010_*, 011v3_*"),
    ("archive/006, 012", "tables/006_*, 012_*; figures/006_*, 012_* (superseded defaults)"),
    ("one-off checks without a script", "figures/009_top_rotation_*, 016_pupil_three_point_*, 017_*"),
    ("ssl-dlc-kuma 025 / 030", "DLCProjects/merged_training_2026-11-09/pilot_inference/{before_after,failed_top}/"),
], columns=["Script (exploratory-analyses/)", "Outputs (combined_results_ks4/ssl-dlc-utils-validation/)"]),
    f"Script to output index. Code: {SCRIPTS}/ (README.md); results: combined_results_ks4/ssl-dlc-utils-validation/. "
    "Version history: 2026-11-10 first report; scripts moved into sections the same day (numbers unchanged).")

summary = f"""# DLC utilities validation -- summary ({date.today().isoformat()})

- Stored DLC series are in physical units per second; dlc_helpers converts units only.
- Jaw / nose reference at the image origin in many sessions (jaw {n_jaw0} of {n_dlc}): recomputed at loading.
- Setup 2: smaller top scale and upside-down top camera, both handled by the converter.
- Jaw never gated by likelihood; pupil / whisker / nose cutoff 0.1 when a mask is requested; default cleaning returns stored values.
- DLC lick times: per-frame camera timestamps (offset method drifts).
- Pupil: QC-based cleaning, 'fail' sessions excluded.
- Five failed top-view sessions are not removed by defaults: exclude or re-infer.
- Whisker angle increases with protraction (per-session offset, use baseline-subtracted); jaw angle scale depends on the DLC model.
"""
R.write(OUT, md, summary=summary)
