# ssl-behaviour-motifs: question

## Original request (2026-11-19, via orchestrating agent)
Pilot: find behavioural motifs from side- and top-view DLC keypoints of day-0 whisker sessions with three methods
(keypoint-MoSeq, sticky AR-HMM on interpretable features, supervised lightning-action on heuristic labels), one report
per session showing both views, plus a short cross-session summary.

## Explication
- Sessions: 8 whisker day-0 sessions, 4 R+ / 4 R- (cohort per mouse, reference sheet in share_internal), 2 learners +
  2 non-learners per cohort (sheet `learning_category`: good/moderate = learner, bad = non-learner), AB and MH mice.
- Views fitted separately (side: nose_tip, nose_base, jaw, jaw_ref, tongue; top: nose_tip, nose_base, whisker_base,
  whisker_tip); v2: native 200 Hz, keypoints / likelihoods / timestamps from NWB_ks4 (v1 used 50 Hz and the h5 files).
- Questions: (1) do the methods find interpretable, stable motifs? (2) do methods and views agree? (3) do motif usages
  follow task epochs, licks, jaw onsets and learning? (4) exploratory: R+ vs R- usage, also restricted to no-lick time.

## Definitions
- Whisker hit = lick on a whisker trial (both cohorts). First lick corrected per trial (skills/ssl-lick-alignment).
- Epochs: pre = [-2, 0) s before stimulus; resp = stimulus to response-window end; iti = rest of the active block.
- No-lick frames: no piezo lick within +-1 s.
- Heuristic labels (lightning-action): see 002 docstring (v2: still, whisking, nose, jaw, lick); class 0 = ambiguous /
  not scored. F1 against these labels is circular (labels and model inputs come from the same keypoints).

## Exploration / confirmation
Pilot = exploration only. R+ vs R- tests are exploratory, uncorrected (Mann-Whitney + Welch), mouse level, n = 4 vs 4.
