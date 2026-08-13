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
- `references/ssl_analysis_patterns.md`: established metric definitions and statistical designs from prior SSL projects (passive selectivity index, GLMM structure, reward-history indices, TCA pipeline, area-grouping schemes).
- `../ibl-analyze/references/reproducibility_qc.md`: general statistical hygiene (paired alignment, explicit exclusions, seeds) — applies unchanged.
- `../ssl-load/SKILL.md` and its references: resolve the data source before analyzing.

## Default analysis policy
1. Read `references/ssl_behavioral_paradigm.md` and `references/ssl_task_semantics.md` before interpreting `trial_type`, `lick_flag`, `perf`, or `context` — the "hit" outcome for whisker trials is **cohort-dependent** (flips meaning between reward groups); a naive `lick_flag`-only reading is wrong for cohort comparisons.
2. State which cohort/reward_group source was used (see `ssl-load/references/ssl_loading_policy.md`) and which day-stage (`day==0` learning vs `day>0` expert) is in scope.
3. Check `references/ssl_analysis_patterns.md` for an existing established definition before inventing a new metric for a question type prior projects have already answered (passive responsiveness/selectivity, reward-history modulation, task-performance curves, trial-factor/TCA decomposition).
4. When grouping units by brain area, declare which scheme is used (Allen custom large-groups vs IBL Beryl) and do not mix them within one analysis — see the Area Grouping section of `ssl_analysis_patterns.md`.
5. For any group-level test where units are the row grain but the group factor (cohort, reward_group) is a **mouse-level** property, use mouse-block permutation or a mixed model with `(1|mouse)` — never permute or test at the unit level directly. This project's dataset has ~300-900 units per mouse; naive unit-level permutation is pseudoreplication (see the PERMANOVA caveat in `ssl_analysis_patterns.md`).
6. Treat this dataset as **private and unpublished** by default (Axel Bisi's own in-progress experiments, not a released public dataset). Apply `ibl-report/SKILL.md`'s privacy checklist before any figure, table, or report leaves the local environment, and do not suggest public GitHub Pages publishing without the user explicitly opting in per session.

## Quality gates
- Reject any cohort comparison that uses `lick_flag`-derived hit/miss without applying the cohort-dependent correction from `ssl_task_semantics.md`.
- Reject unit-level statistical tests of a mouse-level factor (cohort, reward_group, learning_category) without mouse-block permutation or a mixed-effects `(1|mouse)` term.
- Reject area-level comparisons that silently mix Allen-custom-group and Beryl-mapped labels.
- Flag `auditory_miss` trials as too rare for most per-trial-type analyses (established across prior SSL projects); do not build a primary metric on them without checking trial counts first.
