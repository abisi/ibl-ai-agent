---
name: ssl-analyze
description: Use this skill for quantitative scientific analysis of the SSL whisker/auditory associative-learning task dataset (Go/No-Go for R- whisker trials only; Go for auditory and R+ whisker trials), after the always-on scientific workflow in AGENTS.md is active and ssl-load has resolved a data source.
---

# SSL Analyze

## Use this skill when
- The user asks a scientific question about the SSL cohort (task performance, passive sensory responses, reward-history modulation, learning trajectories, single-neuron/population dynamics).
- You need task-outcome semantics, cohort definitions, area-grouping conventions, or established analysis patterns for this dataset.

## Required first step
Apply the `AGENTS.md` scientific workflow and `ibl-analyze/references/scientific_context_and_metric_semantics.md`'s Semantic Match Gate and Shape-Before-Scalar rules before choosing metrics — those apply here unchanged; this skill adds SSL-specific facts, it does not replace that gate.

## References
- `references/ssl_behavioral_paradigm.md`: authoritative task design (free-licking → auditory pretraining → whisker learning stages, timing parameters, full outcome/reward-contingency tables), transcribed from Axel Bisi's own paper/thesis text — read this first for what the task actually is.
- `references/ssl_task_semantics.md`: how that design maps onto dataset columns — trial-outcome columns, cohort/reward-group semantics, epoch semantics — read this before interpreting any trial or outcome column.
- `references/ssl_artifact_dead_zone.md`: **mandatory** whisker-trial stimulus-artifact spike exclusion rule — read this before any neural computation (firing rates, PSTHs, tensors) that touches whisker-trial spikes, no exceptions.
- `references/ssl_auditory_warmup_block.md`: **mandatory** trim of the auditory-only warm-up block many whisker-training sessions begin with — read this before any analysis (especially modality decoding) that keeps both `whisker_trial` and `auditory_trial` rows from a session's active epoch together.
- `references/ssl_analysis_patterns.md`: established metric definitions and statistical designs from prior SSL projects (passive selectivity index, GLMM structure, reward-history indices, TCA pipeline, area-grouping schemes).
- `references/ssl_cohort_comparison_stats.md`: concrete test recipes for a mouse-level factor (cohort, `learning_category`, ...) on a unit-level metric — location (mouse-level Mann-Whitney+Welch, neuron-level LMM), proportion, and distribution (neuron-level KS vs. the mouse-block permutation KS that actually respects mouse-level replication) — read this before writing any R+/R- or similar cohort test beyond a single location test.
- `../ibl-analyze/references/reproducibility_qc.md`: general statistical hygiene (paired alignment, explicit exclusions, seeds) — applies unchanged.
- `../ssl-load/SKILL.md` and its references: resolve the data source before analyzing.

## Default analysis policy
0. Use the **KS4** source (`ssl_ephys`/`ssl_behavior`) unless the question, a prior project being reproduced, or the user explicitly calls for KS2 — see `ssl-load/references/ssl_loading_policy.md`'s Core policy.
1. Read `references/ssl_behavioral_paradigm.md` and `references/ssl_task_semantics.md` before interpreting `trial_type`, `lick_flag`, `perf`, or `context` — the "hit" outcome for whisker trials is **cohort-dependent** (flips meaning between reward groups); a naive `lick_flag`-only reading is wrong for cohort comparisons.
2. Before any computation involving whisker-trial spikes (firing rate, PSTH, tensor, model input), apply `references/ssl_artifact_dead_zone.md`'s -1ms/+4ms exclusion around `start_time` — this is a correctness requirement, not an optional convention, and applies regardless of which other pattern in this skill the analysis otherwise follows.
2b. Before any analysis of a whisker-training session's active trials that keeps both `whisker_trial` and `auditory_trial` rows together (e.g. modality decoding), drop every trial preceding that session's first active `whisker_trial` — see `references/ssl_auditory_warmup_block.md` for why (an auditory-only wake-up/engagement block precedes whisker-trial introduction in many sessions) and its exact mechanism for producing a spurious pre-stimulus "modality" signal if skipped.
3. State which cohort/reward_group source was used (see `ssl-load/references/ssl_loading_policy.md`) and which day-stage (`day==0` learning vs `day>0` expert) is in scope. Use `session_id` as the statistical unit for expert-arm analyses and `session_id`/`mouse_id` interchangeably for learning-arm analyses (they coincide) — see the Statistical unit of analysis section of `ssl_analysis_patterns.md`.
4. Apply the mandatory mouse-inclusion filters before any analysis, behavioral or neural: `exclude==0` (all analyses); additionally `exclude_ephys==0` for anything touching neural/ephys data; and drop `reward_group=='R+proba'` mice from R+/R- comparisons by default. Read from `joint_mouse_reference_weight.xlsx` — see `ssl_task_semantics.md`'s Mandatory mouse-inclusion filters section for the exact columns and file path.
5. Run every analysis under **both** population scopes — entire dataset (all mice passing item 4) and learners-only (`learning_category in {'good','moderate'}`) — with the R+/R- split applied separately within each scope. Report both scopes; do not silently pick one. See `ssl_task_semantics.md`'s Two population scopes section.
6. Check `references/ssl_analysis_patterns.md` for an existing established definition before inventing a new metric for a question type prior projects have already answered (passive responsiveness/selectivity, reward-history modulation, task-performance curves, trial-factor/TCA decomposition).
7. When grouping units by brain area, declare which scheme is used (Allen custom large-groups vs IBL Beryl) and do not mix them within one analysis — see the Area Grouping section of `ssl_analysis_patterns.md`.
7b. Whenever an analysis spans more than one day-stage (learning/expert) and/or more than one area-parcellation level (e.g. `whole_brain`/`area_group`/`area_acronym_custom`), organize saved results and figures into separate folders/filenames along each such axis in play — never flatten them into one directory — see `ssl_analysis_patterns.md`'s Output organization section for the validated `<area_level>/<day_stage>/` nesting pattern and its reference implementation.
8. For any group-level test where units are the row grain but the group factor (cohort, reward_group) is a **mouse-level** property, use mouse-block permutation or a mixed model with `(1|mouse)` — never permute or test at the unit level directly. This project's dataset has ~300-900 units per mouse; naive unit-level permutation is pseudoreplication (see the PERMANOVA caveat in `ssl_analysis_patterns.md`).
9. For any R+ vs R- comparison, run both a non-parametric unpaired test (e.g. Mann-Whitney U) and a parametric unpaired test (e.g. t-test), and report both — see the R+ vs R- cohort comparison section of `ssl_analysis_patterns.md`.
10. Treat this dataset as **private and unpublished** by default (Axel Bisi's own in-progress experiments, not a released public dataset). Apply `ibl-report/SKILL.md`'s privacy checklist before any figure, table, or report leaves the local environment, and do not suggest public GitHub Pages publishing without the user explicitly opting in per session.

## Quality gates
- Reject any whisker-trial neural computation that includes real spikes from the -1ms/+4ms stimulus-artifact dead zone around `start_time`, or that uses Poisson dead-zone-correction spikes (visualization-only) as real data in a statistic or model — see `ssl_artifact_dead_zone.md`.
- Reject any cohort comparison that uses `lick_flag`-derived hit/miss without applying the cohort-dependent correction from `ssl_task_semantics.md`.
- Reject any mixed-trial-type (whisker+auditory) active-trial analysis of a whisker-training session that does not first drop trials preceding that session's first active `whisker_trial` — see `ssl_auditory_warmup_block.md`.
- Reject unit-level statistical tests of a mouse-level factor (cohort, reward_group, learning_category) without mouse-block permutation or a mixed-effects `(1|mouse)` term.
- Reject an R+ vs R- comparison that runs only one of the non-parametric/parametric unpaired test pair, or that reports only the significant one — see `ssl_analysis_patterns.md`'s R+ vs R- cohort comparison section.
- Reject expert-arm (`day>0`) analyses that pool or average a mouse's multiple expert sessions into one `mouse_id`-level observation before testing — the statistical unit for expert-arm analyses is `session_id` (see `ssl_analysis_patterns.md`'s Statistical unit of analysis section), unless the tested factor is itself mouse-level (e.g. cohort), in which case mouse-block permutation still applies.
- Reject area-level comparisons that silently mix Allen-custom-group and Beryl-mapped labels.
- Reject saving results/figures from a multi-day-stage or multi-area-level analysis into one flat, undifferentiated location — see `ssl_analysis_patterns.md`'s Output organization section.
- Flag `auditory_miss` trials as too rare for most per-trial-type analyses (established across prior SSL projects); do not build a primary metric relying on neural data on them without checking trial counts first. The only exception for behavioural metric is the auditory hit rate that can be computed regardless of the number of auditory misses.
- Reject a "distribution differs" claim backed only by a neuron-level Kolmogorov-Smirnov test on pooled units — that pools units across mice with unequal counts, the same pseudoreplication risk as any other unit-level test of a mouse-level factor; require the mouse-block permutation version (see `ssl_cohort_comparison_stats.md`).
- Reject any analysis that does not state whether the mandatory mouse-inclusion filters (`exclude==0`; `exclude_ephys==0` for neural analyses; `R+proba` dropped from R+/R- comparisons) were applied — see `ssl_task_semantics.md`'s Mandatory mouse-inclusion filters section.
- Reject an R+/R- cohort comparison that includes `R+proba` mice without the question specifically asking about probabilistic reward.
- Reject an analysis that reports only one of {entire dataset, learners-only} scopes without at least noting the other was not computed, when both are feasible — see `ssl_task_semantics.md`'s Two population scopes section.
