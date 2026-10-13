# ssl-jaw-lick-latency

## Question
Is the delay between DLC jaw-opening onset and the first (piezo) lick reliable enough, across trials, sessions and mice, to infer a
jaw-opening time for sessions without usable video, so that pre-movement windows can be defined from lick times alone?

## Data
- SSL sessions of `ssl-prelick-convergence` (stable units run, learners rule for day 0, all expert sessions).
- Events: whisker hits (WH) and auditory hits (AH) at the corrected first lick (`skills/ssl-lick-alignment`), spontaneous licks (SL,
  piezo bout onsets outside trials; piezo clock, about +-25 ms from the trial clock).
- Jaw onset: `ssl-prelick-convergence` 118 (first frame with jaw distance > 7 x baseline SD, rule of behaviour_analysis
  `find_dlc_jaw_onset_times`; WH / AH: baseline 1 s before trial start, search from 50 ms after the stimulus to the lick; SL:
  baseline [-1, -0.5] s, search 0.5 s before the lick; no crossing -> dropped). DLC handling: `skills/ssl-dlc-units`.

## Quantities
- lead = lick time - jaw onset (ms), per event.
- Variance components of lead: mouse, session within mouse, trial (mixed model), per event class.
- Dependence of lead on reaction time, event class, time in session, learning stage, setup.
- Inference test: leave-one-session-out prediction of jaw onset from the lick time (global, per-mouse, reaction-time model),
  compared with a within-session oracle; errors per trial and per session.

## Code / outputs
- Code: `projects/ssl-jaw-lick-latency/exploratory-analyses/`
- Results: `combined_results_ks4/ssl-jaw-lick-latency/{tables,figures}/`
