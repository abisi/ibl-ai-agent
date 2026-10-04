# TODO

- [X] Resolve all ambiguous terms with user (cohort, 10ms exclusion, diff/index
      definition, dataset scope, GLMM x area handling, response/baseline windows).
      See `question.md`.
- [X] Confirm `combine_ephys_nwb` loading path works (done in prior session:
      `unit_spikes_analysis` + `allen_utils` on sys.path, run via that repo's venv).
- [X] Confirm spike-time access pattern (`unit_table['spike_times']`,
      `unit_table['unit_id']` as safe per-unit key) and passive-trial selection
      logic (`context` column, verified 'passive'/'active' values on AB116).
- [ ] Enumerate full candidate ephys session list across NWB_ks4 (all subjects,
      not just AB116-120).
- [ ] Write batched per-unit response-summary extraction script: passive
      pre/post trial selection, active stim counts, response/baseline
      computation (both window pairs), diff, selectivity index. Validate on
      AB116-120 (known-good subset) before scaling.
- [ ] Run full-cohort batched extraction (background, likely long-running).
      Save compact per-unit summary table to `artifacts/`.
- [ ] Time-course figures (firing-rate-change time course), cohort x modality
      interaction plot, selectivity index by area group, split by
      learning/expert.
- [ ] Mixed-effects models (pooled across area, per window, per day-stage) and
      cohort statistics.
- [ ] Consult user with intermediate results before finalizing report.
