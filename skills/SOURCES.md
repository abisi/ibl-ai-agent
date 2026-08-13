# Skill Source Index

External sources used when drafting retained IBL skill references. Use this for maintenance/provenance, not normal runtime routing.

## ibl-access

`ibl-access/references/one_auth.md` and `session_search.md`:
- https://int-brain-lab.github.io/ONE/_autosummary/one.api.html
- https://docs.internationalbrainlab.org/notebooks_external/one_quickstart.html
- https://int-brain-lab.github.io/ONE/FAQ.html
- https://int-brain-lab.github.io/ONE/notebooks/one_advanced/one_advanced.html
- https://int-brain-lab.github.io/ONE/notebooks/one_search/one_search.html

## ibl-load

`ibl-load/references/data_loading.md`:
- https://int-brain-lab.github.io/ONE/_autosummary/one.api.html
- https://docs.internationalbrainlab.org/notebooks_external/one_quickstart.html
- https://int-brain-lab.github.io/ONE/notebooks/one_advanced/one_advanced.html

`ibl-load/references/bwm_runtime_policy.md`, `bwm_paper_repo.md`, and `bwm_release_scope.md`:
- https://github.com/int-brain-lab/paper-brain-wide-map
- https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/bwm_loading.py
- https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/meta/region_info.csv
- https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/meta/inclusion_info.csv
- https://www.nature.com/articles/s41586-025-09235-0
- https://www.nature.com/articles/s41586-025-09226-1
- https://www.internationalbrainlab.com/publications/

`ibl-load/references/ibl_behavior_task.md`:
- https://elifesciences.org/articles/63711
- https://www.nature.com/articles/s41592-022-01742-6
- https://www.nature.com/articles/s41593-021-01007-z
- https://www.internationalbrainlab.com/publications/

`ibl-load/references/brain_regions_qc.md`:
- https://docs.internationalbrainlab.org/_autosummary/brainbox.io.one.html
- https://docs.internationalbrainlab.org/notebooks_external/loading_trials_data.html
- https://docs.internationalbrainlab.org/atlas_examples.html
- https://docs.internationalbrainlab.org/atlas_working_with_ibllib_atlas.html
- https://docs.internationalbrainlab.org/atlas_plotting_scalar_on_slice.html
- https://docs.internationalbrainlab.org/_autosummary/iblatlas.plots.html

## ibl-analyze

`ibl-analyze/references/brainbox_routing.md`:
- https://docs.internationalbrainlab.org/_autosummary/brainbox.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.singlecell.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.population.decode.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.population.cca.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.task.closed_loop.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.task.trials.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.behavior.wheel.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.behavior.training.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.behavior.dlc.html
- https://docs.internationalbrainlab.org/_modules/brainbox/io/one.html
- https://docs.internationalbrainlab.org/_modules/ibllib/qc/dlc.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.metrics.single_units.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.metrics.electrode_drift.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.task.passive.html

`ibl-analyze/references/operators.md` and `visual_latency.md`:
- https://docs.internationalbrainlab.org/_autosummary/brainbox.population.decode.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
- https://docs.internationalbrainlab.org/notebooks_external/loading_spikesorting_data.html
- https://docs.internationalbrainlab.org/notebooks_external/loading_trials_data.html
- https://docs.internationalbrainlab.org/_autosummary/brainbox.singlecell.html

`ibl-analyze/references/scientific_context_and_metric_semantics.md`, `prior_and_block_semantics.md`, and `reproducibility_qc.md`:
- https://elifesciences.org/articles/63711
- https://elifesciences.org/articles/100840
- https://www.nature.com/articles/s41592-022-01742-6
- https://www.nature.com/articles/s41586-025-09235-0
- https://www.nature.com/articles/s41586-025-09226-1
- https://www.nature.com/articles/s41593-021-01007-z

`ibl-analyze/references/bwm_analysis_patterns.md`:
- https://github.com/int-brain-lab/paper-brain-wide-map
- https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/bwm_loading.py
- https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/decoding/README.md
- https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/decoding/decoding_example_script.py
- https://github.com/int-brain-lab/paper-brain-wide-map/blob/main/brainwidemap/meta/oscillations.py

## ibl-neuropixel

`ibl-neuropixel/SKILL.md`, `neuropixel_routing.md`, and `neuropixel_function_signatures.md`:
- https://github.com/int-brain-lab/ibl-neuropixel
- https://github.com/int-brain-lab/ibl-neuropixel/blob/main/README.md
- https://docs.internationalbrainlab.org/_autosummary/brainbox.io.spikeglx.html

## ssl-load, ssl-analyze

Not external URLs — SSL is a private, non-IBL dataset. Drafted directly from
this repo's own dataset builder code and from completed SSL project reports,
not from any published/external documentation:
- `ibl_ai_agent/datasets/ssl_ephys.py`, `ssl_behavior.py`, `ssl_ks2_ephys.py`, `ssl_ks2_behavior.py` (schema, shard contracts).
- `projects/ssl-ks2-dataset-description/report/report.qmd` (dataset-wide counts, KS4-vs-KS2 differences).
- `projects/ssl-passive-sensory-selectivity/question.md` (passive selectivity method).
- `projects/ssl-reward-history-modulation/question.md` (reward-history indices, PERMANOVA pseudoreplication fix).
- `projects/ssl-ks2-single-mouse-tca/question.md` (TCA pipeline, circular-shift null).
- `projects/ssl-task-performance/exploratory-analyses/000_active_task_performance.py` (naive outcome classification).
- Task-design confirmation from Axel Bisi (the experimenter) directly, 2026-08-11.
- `ssl_behavioral_paradigm.md` is transcribed directly from Axel Bisi's own
  paper/thesis draft text (provided verbatim in-conversation, 2026-08-13) —
  the primary/authoritative source for task design, superseding inference
  from code/project reports where the two differ.
