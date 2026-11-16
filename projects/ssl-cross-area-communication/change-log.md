# Change log

- 2026-08-31: Project created. `question.md`/`TODO.md` written per the plan
  approved the same day (`C:\Users\bisi\.claude\plans\cheerful-bouncing-cloud.md`).
  No prior state to diff against.
- 2026-08-31: Step 1 (recording-pair coverage) and the intermediate-checkpoint
  pipeline both completed. Two bugs found and fixed during this pass:
  1. `002_full_coverage_report.py` and (initially) `004_small_subset_pipeline.py`
     imported `ibl_ai_agent.data_locations.resolve_dataset_dir`, which isn't on
     the `bwa` conda env's path (only the repo's own `.venv` has it) -- crashed
     `002` right after its ~20min Path B load finished, losing that run.
     Fixed by reading `reports/datasets/ssl_ephys/1.0.0/metadata/*.parquet`
     directly instead; added a mid-script checkpoint parquet save as insurance
     against losing a future ~20min load to a downstream bug.
  2. `004_small_subset_pipeline.py`'s first "example correlated trials" figure
     fit CCA and evaluated it on the *same* trials (in-sample) -- inflated the
     apparent per-trial correlation (r~0.5-0.65) in a way inconsistent with the
     project's own CV requirement. Fixed to fit on a held-out half of trials
     and only show examples from the other half; the honest median dropped to
     r=0.09 (max 0.61) vs. the CV canonical correlation of 0.184+/-0.019 for
     the population-level dim-1 signal -- single trials are noisier than the
     aggregate measure, as expected.
  Known non-blocking issue: the noise-correlation illustration figure
  (`checkpoint_02`) picked its example unit by highest total spike count,
  which landed on what looks like an MUA cluster (rate snapping to round
  multiples of 100Hz, i.e. up to 4 spikes/10ms bin) rather than a clean
  single unit. Not fixed yet (would need a `quality_label=='good'`-only
  selection) -- flagged to Axel rather than blocking on a purely
  illustrative figure.
