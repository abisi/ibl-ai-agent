# Question

## Original question (from user)
Exploratory analysis of the ssl_ks2 dataset, contrasting cohorts on reward
history modulation indices. Define several reward-history modulation
indices, and test for enrichment across cohorts and brain areas. Remove bad
learners (per `learning_category` in the joint mouse reference weight
sheet). Indices defined for success, errors, and transitions from whisker to
auditory trials and vice-versa. Ignore auditory misses (too few). Compare
cohorts with PERMANOVA, then post-hoc across `area_custom_acronyms`.

Only analyze day 0 of whisker learning.

## Term explication (resolved via user clarification, 2026-08-11)

- **Cohort** = `reward_group` from `joint_mouse_reference_weight.xlsx`
  (`M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx`,
  sheet `Sheet1`). Levels: `R+`, `R-`, `R+proba`. **Only R+/R- used**;
  `R+proba` dropped (all 4 R+proba mice are unannotated for
  `learning_category`, i.e. outside this learning-cohort framework).
- **Bad-learner exclusion**: drop mice with `learning_category` in
  `{bad, NaN}`. Keep only `moderate`/`good`. (161 mice total; 74 remain after
  this + cohort filter + day-0-ephys availability, see below.)
- **`exclude_ephys` gate**: also require `exclude_ephys == 0` in the same
  sheet for the neural population.
- **Day 0 of whisker learning**: `sessions.session_description == 'whisker_0'`
  exactly, in `reports/datasets/ssl_ks2_ephys/1.0.0/metadata/sessions.parquet`.
  This matches the `<behavior_type>_<day>` encoding read by
  `NWB_reader_functions.get_bhv_type_and_training_day_index`
  (`M:\analysis\Axel_Bisi\NWB_reader`), day counted relative to whisker-training
  start. Excludes `whisker_on_N_*`/`whisker_off_N_*` opto variants (day 0 never
  appears in those) and all `auditory_*`/`free_licking_*` pre-whisker sessions.
- **Population after filters**: 74 mice with day-0 ephys passing
  cohort+learning+exclude filters (38 R+, 36 R-), computed 2026-08-11 from
  `sessions.parquet` x `joint_mouse_reference_weight.xlsx`.

### Trial outcome derivation (trials.parquet, `ssl_ks2_ephys` dataset)

Raw columns: `trial_type` (`whisker_trial`/`auditory_trial`/`no_stim_trial`),
`lick_flag`, `perf`, `context` (`active`/`passive`). `perf`/`TRIAL_MAP` (from
`M:\analysis\Axel_Bisi\Github\ephys_utilities\neural_utils\neural_utils.py`)
labels the raw stimulus-response pattern (0=whisker_miss, 1=auditory_miss,
2=whisker_hit, 3=auditory_hit, 4=correct_rejection, 5=false_alarm,
6=association) but is **not** cohort-corrected -- it does not know whether
that stimulus was actually rewarded for that mouse.

- Drop `context != 'active'` (passive trials) and `perf == 6` (association).
- **Cohort-corrected hit/non-hit** (only whisker outcome flips by cohort):
  - Whisker trials, R+: hit = `whisker_hit`, non-hit = `whisker_miss`.
  - Whisker trials, R-: hit = `whisker_miss` (withholding is the rewarded
    action), non-hit = `whisker_hit`.
  - Auditory trials (both cohorts): hit = `auditory_hit`. `auditory_miss` is
    excluded as a target trial *t* (too rare) but IS kept as a valid *t-1*
    conditioning event (non-hit) for the auditory-to-whisker transition index.
  - `no_stim_trial` (correct_rejection/false_alarm): not used in any index.

### Reward-history modulation indices (per unit, 4 total)

Common structure: pick which trials count as trial *t* (fixed outcome, fixed
modality), split by whether trial *t-1* was hit vs non-hit (cohort-corrected,
any modality), and contrast mean firing rate in a **5-50ms post-stimulus-onset
window**:

```
index = (FR | t-1 hit - FR | t-1 non-hit) / mean(FR across all trials satisfying the t-condition)
```

1. **Hit-stay** (whisker): *t* = whisker hit. Split by *t-1* hit/non-hit.
   Positive = stronger response repeating a just-rewarded choice (win-stay).
2. **Non-hit-stay** (whisker): *t* = whisker non-hit (error). Split by *t-1*
   hit/non-hit. Mirror of #1.
3. **Whisker-to-auditory transition**: *t* = `auditory_hit` with *t-1* a
   whisker trial. Split by that whisker *t-1*'s cohort-corrected outcome.
4. **Auditory-to-whisker transition**: *t* = cohort-corrected whisker-hit
   with *t-1* an auditory trial. Split by that auditory *t-1*'s outcome
   (hit vs `auditory_miss`-as-non-hit -- expect this bucket to be thin).

A unit's index for a given category is NaN if either t-1-history bucket has
fewer than 5 qualifying trials.

### Brain area labeling (`area_acronym_custom`)

1. Units table lacks `target_region` (dropped when the `ssl_ks2_ephys`
   builder flattened `electrode_group` to a name string). Re-extract it per
   (session_id, probe_name) directly from raw NWB files
   (`M:\analysis\Axel_Bisi\NWB_combined`) via the same logic as
   `neural_utils.convert_electrode_group_object_to_columns`
   (`electrode_group.location['area']`), for day-0 sessions of eligible mice
   only. Merge onto `units.parquet`.
2. Run `allen_utils.process_allen_labels(df)` (default args, i.e.
   `subdivide_areas=False`) from
   `M:\analysis\Axel_Bisi\Github\allen_utils\allen_utils.py` ->
   `area_acronym_custom` column. This also drops `get_excluded_areas()`
   internally.
3. Keep only areas shared across R+/R- cohorts with: >=2 mice per cohort,
   >=5 units per recording, >=30 total units (good+mua) overall.

### Statistics

- **Confirmatory**: PERMANOVA on the 4-dim (index-1..4) standardized vector
  per unit, factor = cohort (R+/R-), run on the **confirmation set** only.
- **Post-hoc**: pairwise/per-area test across `area_acronym_custom` to
  localize which areas carry the cohort effect ("enrichment"), confirmation
  set only.

## Exploration / confirmation split

Requested explicitly (unlike `ssl-passive-sensory-selectivity`, which skipped
it). Replicate unit = mouse. Stratified-by-cohort random split of the 74
eligible mice:
- **Exploration set**: 8 R+ / 7 R- (15 mice) -- used only to sanity-check
  index distributions/plausibility and finalize any remaining parameter
  choices before locking.
- **Confirmation set**: remaining 59 mice -- used only for the locked
  PERMANOVA + post-hoc.
- Fixed random seed for reproducibility; split saved to
  `artifacts/mouse_split.csv`.

## Data scope

- Loading: built `ssl_ks2_ephys` dataset at
  `reports/datasets/ssl_ks2_ephys/1.0.0/` (trials, units, sessions parquet +
  spike shards), NOT re-reading full NWB files except for the one-off
  `target_region` extraction (metadata only, no spike data).
- Spike shards decoded via `ssl_ks2_ephys.load_spike_shard`.

## Open items to watch during exploration
- `auditory_miss`-as-*t-1* bucket for index #4 may be too thin for many
  units -- check during exploration and consider dropping index #4 or
  loosening its trial floor if so.
- Whether re-extracted `target_region` is available/consistent for all day-0
  sessions of eligible mice (some probes may be missing this metadata).

## Results (2026-08-11)

**Index #4 (`a2w_transition`) dropped before locking**: exploration-set check
(with floor loosened down to 1 trial/bucket) showed 0 valid units in R+
(vs 302 in R- at floor=5). Full-population confirmed this is structural, not
a threshold artifact: only 2/38 R+ mice ever produce >=2 qualifying trials in
both t-1 buckets (vs 9/36 R-) -- R+ mice essentially never have a whisker-hit
trial immediately preceded by an `auditory_miss` on day 0 (auditory
performance is near-ceiling, ~88% hit rate, in R+). Locked confirmatory
PERMANOVA run on 3 indices (`hit_stay`, `nonhit_stay`, `w2a_transition`)
only.

**Bug found and fixed before reporting any result**: the first PERMANOVA run
permuted unit labels (7496 "samples") for the permutation null, which is
invalid -- `reward_group` is a mouse-level property (~680 non-independent
units/mouse), so the null must permute whole mice between cohorts. This
initial (invalid) run gave p=0.0004 with ORB/SSp-bfd/MOp "significant"
post-hoc; the corrected mouse-block-permutation version of the *same data*
gave p=0.57 with nothing significant post-hoc -- the original result was
pseudoreplication, not signal. All reported results below use the corrected
mouse-block permutation.

| Run | Indices | Population | n units | n mice | pseudo-F | p | Areas surviving BH-FDR 5% |
|---|---|---|---|---|---|---|---|
| Locked confirmatory | hit_stay, nonhit_stay, w2a_transition | confirmation set (59 mice) | 7496 | 11 | 5.73 | 0.573 | 0/10 tested |
| Full population | hit_stay, nonhit_stay, w2a_transition | all 74 mice | 9442 | 14 | 9.85 | 0.242 | 0/11 tested |
| Supplementary (not locked) | hit_stay, nonhit_stay | all 74 mice | 31153 | 41 | 2.08 | 0.809 | 0/32 tested (best: VIS, p=0.055 uncorrected) |

`w2a_transition`'s mouse-level bottleneck is itself structural, not a sample-
size problem: merging exploration+confirmation only recovered 14/74 mice
with any complete-case unit (barely more than the 11/59 in the confirmation
set alone) -- most day-0 sessions aren't modality-interleaved enough to ever
produce an auditory-hit trial immediately preceded by a whisker trial.
Dropping it (2-index supplementary run) recovered 41/74 mice, the
best-powered test run, and is also null.

**Conclusion**: no evidence of a cohort (R+/R-) difference in reward/choice-
history modulation of day-0 whisker-learning firing rates, across all three
index-set/population combinations tested, including the best-powered one.
No brain area survives multiple-comparisons correction in any version.
Artifacts: `artifacts/{main_permanova_result,main_permanova_result_fullpop,
main_permanova_result_fullpop_2index}.txt`,
`artifacts/posthoc_area_permanova*.csv`.
