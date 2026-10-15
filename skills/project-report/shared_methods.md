# Shared methods text (SSL projects)

Canonical wording of definitions used by several SSL reports. Reports include sections verbatim with
`Report.methods("<key>", ...)`, so a definition changes in one place only. No result numbers here. Sections start with
`## <key>`.

## task
Mice were first trained to lick for water after an auditory tone. A whisker stimulus was then introduced and rewarded
for one cohort (R+) but not for the other (R−), while auditory trials remained rewarded in both cohorts; no-stimulus
trials measured spontaneous licking (false alarms). Cohort membership is defined per mouse from the reference sheet
(`joint_mouse_reference_weight.xlsx`), not from session labels. Day 0 is the first whisker-training session; expert
sessions are later whisker sessions after the contingency was acquired.

## learners
Learning populations follow the learner rule: day-0 sessions of mice that did not learn the contingency on day 0 are
excluded, while their expert sessions are kept. (Definition under review, 2026-10-15: the learner list of the pre-lick
analyses and the learning-trial classification of ssl-learning-trial-identification 017 disagree for some mice; one of
the two will become canonical.)

## trials
Trials are restricted to the analysed active epoch: performance code 6 trials are always removed (except passive
trials), the warm-up block before the first whisker trial is cut, and a disengaged tail is trimmed when, after the last
lick, at least five whisker and one auditory trial remain unlicked (rule A1). Learning-trial identification uses the
untrimmed trials.

## units
Units are good and multi-unit clusters from Kilosort 4 that pass the stability criteria (coverage ≥ 0.9, presence ≥ 0.5
and no dependence of firing on probe drift). Analyses that compare epochs within a session use the shared list of tracked
stable units, which fire at ≥ 0.5 Hz in every epoch. Units are grouped into area groups of the custom Allen hierarchy;
somatosensory groups are written Ss-whisker, Ss-orofacial and Ss-body.

## lick_alignment
Lick-aligned windows use the corrected first lick, start_time + (lick_time − response_window_start_time), per trial;
the stored lick_time is late by the artifact window. Spontaneous licks are lick-bout onsets (≥ 1 s after the previous
lick) recorded by the piezo outside every trial window; they are on the piezo clock, about ± 25 ms from the trial clock.

## shift_null
Linear-shift null: the labels of the time-ordered events are shifted against the neural data by k events (10–50 % of the
events, both directions, without wrap-around), the analysis is rerun on the shifted labels, and the real value is
compared with the distribution over shifts. Slow drifts shared by labels and activity are therefore present in the null,
and a reported value is the real value minus the mean of the null.

## statistics
The session is the unit of analysis. Within-cohort stage comparisons (day 0 vs expert) and between-cohort comparisons
use the Mann–Whitney U test and Welch's t test; within-session paired comparisons use the Wilcoxon signed-rank test.
Cohort × stage interactions are tested by permuting cohort labels across mice (10,000 permutations; a mouse keeps all
its sessions). Across areas, an omnibus model value ~ cohort × stage × area (type II sums of squares, p from 1,000
mouse-level cohort permutations) precedes per-area post hoc tests, corrected with the Benjamini–Hochberg procedure.

## areas
An area group is analysed for a cohort when that cohort has at least three sessions with enough units at both stages;
cohort comparisons and interactions are tested only where both cohorts qualify. Areas are shown in the default order of
`ephys_utilities.allen_utils.get_area_group_custom_order()` with the colours of `get_custom_area_groups_colors()`.
