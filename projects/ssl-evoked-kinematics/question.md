# ssl-evoked-kinematics

## Question
Do whisker- and auditory-evoked orofacial movements differ in their kinematics, and does this depend on the reward
cohort (R+: whisker hits rewarded; R-: whisker hits not rewarded)? 2 x 2 design: cohort (R+, R-) x stimulus modality
(whisker, auditory). Beyond the time courses of posture and speed, describe the stereotyped trajectories of the tongue
reach and of the whisker sweep, step by step in a head-centred frame in mm.

## Data
NWB_ks4 (KS4 dataset), DLC through `ephys_utilities.helpers.dlc_helpers` (side and top cameras, mm via the NWB
conversion factor). Cohort per mouse from joint_mouse_reference_weight.xlsx (R+ / R-, exclude == 1 dropped).
Stages: day 0 (`whisker_0`) and expert (`whisker_+1`, `whisker_+2`). Active trials only (perf 6 / passive dropped):
whisker hit (perf 2), whisker miss (perf 0), auditory hit (perf 3). Aligned to stimulus onset.
Session drops as in ssl-dlc-utils-validation 011 (AB155_20250208 video spans the joined recording; top view of
MH025_20250421 / MH027_20250422 failed tracking). The other failed top views (MH009_20250207 / 20250208,
MH025_20250422) give no whisker data through the validity rules (no top head frame / no valid skeleton frames).

## Methods (exploratory-analyses/000_evoked_kinematics.py)
- Head frame per camera and session: origin = median nose tip; x axis = median nose base -> nose tip direction
  (forward); y axis = up (side view; sign set so the pupil is above the nose line) or lateral (top view; sign set so the
  whisker base is on +y). Frames used for the frame: nose tip likelihood >= 0.1, nose base >= 0.5.
- Tongue: out where tongue_likelihood >= 0.6 (dh.TONGUE_PRESENCE_CUTOFF); single frames between 0.3 and 0.6 inside a
  run are bridged (counted in the reach duration, no position). A reach = one run of 3 frames .. 0.3 s. First reach
  starting 0 .. 1 s after the stimulus. Positions only from frames >= 0.6; relative to the jaw position at rest (median
  jaw keypoint, -0.5 .. 0 s, jaw not likelihood-gated, dh rule). Step-by-step trajectory: positions every 5 ms from the
  reach onset (nearest frame within half a frame).
- Whisker: skeleton whisker_base -> top_particle -> whisker_tip (top view). A frame is valid when base >= 0.1, particle
  >= 0.1 (dh cutoffs), tip >= 0.5 (not validated lower) and both segment lengths are within 0.6 .. 1.6 x their session
  medians (rejects swapped / jumped points); 3-frame running median on valid frames. Whisker angle = base -> tip angle
  to the head axis (deg; lower = more protracted); curvature = angle(particle -> tip) - angle(base -> particle).
- Time courses (-0.5 .. 1.5 s, 10 ms bins): jaw opening (dh.clean_series), tongue out (P), tongue extension from the
  jaw at rest (frames with tongue out), nose displacement (side, 2-D, from the trial baseline), nose speed (side),
  whisker angle, whisker angular speed, whisker curvature, top-view nose speed. Postures baseline-subtracted (-0.5 .. 0).
- Per-trial metrics: movement onset latency (jaw opening, whisker angle, nose displacement: first of 3 consecutive 10-ms
  bins above baseline mean + 5 x the session's median baseline SD, 0 .. 1 s), reach latency / duration / maximal
  extension / peak tongue speed / end point, whisker sweep amplitude (max |angle change| 0 .. 0.3 s).
- Averaging: trials -> session -> mouse (mean over the mouse's sessions in the stage) -> cohort mean +- SEM over mice.
  Stats on mouse values: R+ vs R- per modality (Mann-Whitney U and Welch t, both reported), whisker vs auditory hits
  within cohort (Wilcoxon signed-rank and paired t over mice with both). Uncorrected.

## Day-0 follow-up questions (002-005)
- 002: is there a whisker movement specific to the whisker stimulus (sensing) 30-100 ms after it, beyond the lick
  movement? Whisker hits vs misses vs auditory hits, R+ vs R-.
- 003-005: does whisker posture / whisking before the stimulus predict the choice (lick vs no lick), beyond session
  progression, engagement, trial history, pupil and licking already under way? Time controls (stratified, detrended,
  matched, local shuffle) and specificity across trial classes (whisker, auditory, no stimulus).

## Outputs
combined_results_ks4/ssl-evoked-kinematics/{figures,tables,cache,report}; script index in README.md

## Status
2026-11-05: first version (user request: evoked movements + tongue reach + whisker sweep trajectories, 2 x 2 grid).
2026-11-05: day-0 follow-ups 002 (whisker sensing), 003-005 (pre-stimulus state, time controls, summary figures).
2026-11-10: clean-up (README, docstrings), summary figure 006, first report in combined_results_ks4/ssl-evoked-kinematics/report/.
