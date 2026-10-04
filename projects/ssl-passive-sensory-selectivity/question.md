# Question

## Original question (from user)
Analyze passive data only (excluding a 10ms dead-zone around stimulus time).
Sensory responsiveness and modality selectivity (whisker vs auditory), before
vs after the active task, at the single-neuron level and across the cohort.
For mice with passive data, quantify change in post-stimulus firing rate
(5-50ms and 5-30ms windows -- see explication below), after vs before the
task. Compare across brain area (large groups per allen_utils), cohort, and
modality. Make a time course. Plot the cohort x modality interaction. Build a
selectivity index: (whisker_diff - auditory_diff) / (whisker_diff +
auditory_diff). Run statistics across cohorts. Control for the number of
active whisker/auditory stimuli per mouse via a mixed-effects model: firing
rate change ~ modality * cohort + stim counts + (1|mouse). Use all mice.

## Term explication (resolved via user clarification, 2026-08-05)

- **Cohort** = reward-group, from session metadata's `wh_reward` field
  (whether whisker stimuli are rewarded for that mouse). Two levels expected
  (e.g. whisker-rewarded vs auditory-rewarded groups) -- exact levels to be
  confirmed empirically once loaded.
- **10ms exclusion** = a dead-zone in both baseline and response windows:
  never compute firing rate from spikes within 10ms of stimulus onset. This
  supersedes the user's original "5-50ms/5-30ms" example windows. Resolved
  windows:
  - Response windows (both computed, in parallel): **10-50ms** and **10-30ms**
    post-stim-onset.
  - Baseline window: **-60 to -10ms** pre-stim-onset (50ms, ending at the
    10ms dead-zone boundary).
- **Passive trial selection**: NOT based on the epochs.parquet-style epoch
  boundary timestamps (found unreliable for some sessions in the earlier QA
  report -- AB119/AB120 had overlapping active/passive_post windows). Instead,
  use the trial-level `context` column directly (values are exactly
  `'passive'`/`'active'` per trial, verified against AB116):
  - passive_pre trials = `context=='passive'` AND `start_time` before the
    session's first `context=='active'` trial.
  - passive_post trials = `context=='passive'` AND `start_time` after the
    session's last `context=='active'` trial.
- **response** (per unit, per modality, per period pre/post-task) =
  mean firing rate in the response window minus mean firing rate in the
  baseline window, averaged/computed across that period's trials of that
  modality.
- **diff** (per unit, per modality) = response(passive_post) - response(passive_pre).
- **selectivity index** = (whisker_diff - auditory_diff) / (whisker_diff + auditory_diff).
- **Day stratification**: run the entire analysis **separately** for
  `day==0` ("learning") sessions and `day>0` ("expert") sessions (per the
  existing `day_to_analyze` semantics in `combine_ephys_nwb`), rather than
  pooling across training stage.
- **Brain area grouping**: `allen_utils.get_custom_area_groups()` (Motor and
  frontal, Somatosensory, Auditory, Retrosplenial, Visual, Hippocampus,
  Striatum and pallidum, Thalamus, Midbrain, Pons and medulla, Olfactory,
  Amygdala and hypothalamus).
- **Active stimulus counts**: per mouse (per day-stage), count of active-epoch
  `whisker_trial` and `auditory_trial` rows (`context=='active'`), used as
  GLMM covariates to control for stimulus exposure.
- **GLMM structure**: `firing_rate_diff ~ modality * reward_group + whisker_active_count + auditory_active_count + (1|mouse_id)`,
  fit **pooled across all brain areas** (not a per-area or area-as-fixed-effect
  model -- area comparison happens only in the descriptive time-course/
  interaction figures, not inside this model). Fit separately per response
  window (10-50ms, 10-30ms) and per day-stage (learning/expert).

## Data scope

- **Loading**: `combine_ephys_nwb` from `M:\analysis\Axel_Bisi\unit_spikes_analysis\neural_utils.py`
  (run via that repo's own `.venv`, Python 3.14.3), NOT the `ssl_ephys`
  compressed dataset -- this is the user's own established, validated loading
  pipeline (`sys.path` needs both `unit_spikes_analysis` and
  `M:\analysis\Axel_Bisi\Github\allen_utils`).
- **Subjects**: all mice with valid passive data (both a `passive_pre` and a
  `passive_post` block, per the context-column definition above) -- the full
  NWB_ks4 tree (~204 candidate ephys sessions, ~98 subjects), not just the
  AB116-AB120 subset used for earlier QA.
- Given the scale (potentially tens of GB of raw spike data if loaded all at
  once), sessions are processed in batches; only compact per-unit response
  summaries are retained, not raw spike arrays.

## Exploration / confirmation split

Not applicable in the usual sense -- this is a fully specified, user-directed
analysis with a locked design (windows, cohort definition, model structure)
agreed before any computation, not an open-ended exploratory-then-confirmatory
cycle. No data split was requested or made; "run statistical tests across
cohorts" and the GLMM are treated as the (single) planned confirmatory
analysis on all available passive data.

## Open items still to resolve empirically once data is loaded
- Exact reward_group (cohort) levels and their sample sizes.
- Whether both day-stages (learning/expert) have enough passive-eligible
  sessions per cohort for the GLMM to be well-powered.
