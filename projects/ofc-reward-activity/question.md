# Question

## Original question (from user)
What is the mean activity of orbitofrontal cortex (OFC) neurons as a function of
reward obtained? Visualize some example neurons. Mean activity means mean PSTHs,
10 ms bins, around event times.

## Term explication

- **Orbitofrontal cortex (OFC):** Beryl-atlas regions `ORBl` (lateral orbital),
  `ORBm` (medial orbital), `ORBvl` (ventrolateral orbital). Combined, these are
  the standard "OFC" grouping in mouse cortex. Local `bwm_ephys` (v1.2.0) has
  1181 good units across these 3 regions, from 20 insertions in 14 subjects
  (19 sessions).
- **Reward obtained (revised 2026-08-05 per user feedback):** not a binary
  rewarded-vs-unrewarded contrast. The user wants activity tracked **as a
  function of reward index / session progression** — i.e. how the neural
  response to reward evolves from the 1st reward to the last within a session
  (candidate mechanisms: satiety, adaptation, engagement decline). Operationalized
  as: take rewarded trials only (`rewardVolume > 0`) in chronological order within
  a session, split into consecutive groups of N trials (N=10 by default, per
  user's suggested 5-10 range), and compute one PSTH per group.
- **Event times:** trials are aligned to `feedback_times` (the moment reward
  delivery is triggered), since that is the event the reward-locked response
  is measured from.
- **Mean activity / PSTH:** mean firing rate across trials in 10 ms bins,
  per unit, computed separately per trial-order group. A companion scalar
  summary (mean evoked firing rate in a post-feedback response window minus
  a pre-feedback baseline) is plotted against group order to show activity
  as an explicit function of reward index/session progression.

## Data scope

- Dataset: local `bwm_ephys` v1.2.0 (good-unit spike shards) + `metadata/trials.parquet`.
- Trial mask: `bwm_include == True` (standard BWM trial-quality mask).
- Region filter: `beryl_acronym` in {ORBl, ORBm, ORBvl}.

## Exploration / confirmation split

**Approved 2026-08-05.** Replicate = subject (mouse), since units within a
session/subject are not independent. Split by subject, greedily balanced by
total OFC unit count:

- **Exploration set** (8 subjects, 589 units, 10 sessions): ZM_1897, ZM_2240,
  SWC_038, ZFM-01936, SWC_061, DY_014, KS094, KS052
- **Confirmation set** (6 subjects, 592 units, 9 sessions): CSHL060, KS046,
  ibl_witten_19, CSH_ZAD_024, NYU-30, ibl_witten_32

The confirmation set is held out untouched until a specific statistical test is
locked. All exploratory/population-level work below uses the exploration set only.

## Open questions for user feedback
1. Is feedback-time alignment (reward delivery moment) the right event?
2. Group size: using 10 rewarded trials per group as a default — okay, or
   prefer 5, or a different scheme (e.g. quintiles of the session instead of
   a fixed trial count)?
3. Should the response-window summary (evoked FR) use a specific window, or is
   0 to 0.3 s post-feedback (current default) reasonable?
4. Should unsmoothed 10 ms-bin means be shown, or a smoothed PSTH (e.g. Gaussian
   kernel) in addition?
5. Should trial groups pool across sessions/subjects eventually (e.g. "first N
   rewards of the session" per subject, pooled), or stay single-session for now?
