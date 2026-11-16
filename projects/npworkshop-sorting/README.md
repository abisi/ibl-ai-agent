# npworkshop-sorting

Slides for a Neuropixels workshop spike-sorting discussion: how sorter, run-to-run variability and curation choices
change the units obtained from the same recording. Runs on Kuma (SLURM), next to the user's pipeline
(`~/code/ephys_preprocessing`, never edited), on three stable NP1.0 recordings (AB154 imec1, AB147 imec3, AB127 imec2;
random pick, criteria in `config.yaml`).

Figure 1 (motion correction) is not made here (the user's own drift figures). Responsiveness is not computed.

## Design
- **One shared input** per recording: the pipeline's own saved drift-corrected recording (`<probe>/preprocess` on the
  NAS: CatGT `_tcat_corrected` -> float32 -> DREDge interpolation, SpikeInterface 0.104.3), exactly what the existing
  KS4 sorted. CatGT applied tshift and global CAR, no band-pass. No sorter re-applies phase shift, CAR or drift
  correction; each keeps its own high-pass / whitening (SC2 and DARTsort: SpikeInterface filter instead, see config).
- **Sorters**: Kilosort 4 (pipeline params, `do_correction: false`), Kilosort 2.5 (SpikeInterface image), IBL sorter
  1.13.0, SpyKING Circus 2, MountainSort 5, DARTsort 0.5.31. Plus the existing pipeline KS4 output ("as run",
  drift-corrected twice: interpolation + KS4 correction) as a reference point.
- **KS4 variability**: identical rerun (determinism), seeds 11/23/37 (`cluster_init_seed`), `Th_universal` 7/8/10/11.
- **Curation**: IBL, the user's Bombcell (pipeline version and parameters, run on every sorting's Phy export), Allen,
  the user's own criteria (`ephys_utilities` `classify_units_quality`). Exact thresholds in `config.yaml`.

## Files
| step | script | runs in |
|---|---|---|
| envs | `00_setup_envs.sh` | login node (downloads only) |
| inputs | `sbatch/stage_inputs.sbatch` (rclone from the NAS) | SLURM |
| sorting | `02_sort.py` via `sbatch/sort_container.sbatch`, `sbatch/sort_env.sbatch` | SLURM arrays |
| metrics, Bombcell | `03_postprocess.py` via `sbatch/postprocess.sbatch` | SLURM array |
| agreement, variability | `04_compare.py` | `sbatch/figures.sbatch` |
| curation, sweeps | `05_curation.py` | `sbatch/figures.sbatch` |
| figures | `06_figures.py` | `sbatch/figures.sbatch` |

`npw.py`: config, paths, run table, provenance. `vendor/` (not versioned): copies of `ephys_utilities`
`unit_metrics_utils.py` (git 393fa7d) and the drift shift test functions (`unit_spikes_analysis`), synced to Kuma.

## Rerun
On Kuma, code in `/home/bisi/code/npworkshop_figures`, outputs in `/work/lsens/bisi/data/npworkshop`:
```
./submit.sh stage                       # once
./submit.sh sort-container --slice      # 5-min tests (also sort-host, sort-ibl, sort-dartsort, post, figures)
./submit.sh sort-container              # full: container (KS4 runs, SC2), then sort-host / sort-ibl / sort-dartsort
./submit.sh post                        # after all sortings
./submit.sh figures                     # tables, figures, copy to NAS presentations/NPWorkshop
```
Every submission freezes `config.yaml` and the run table in `config_snapshots/`. Every step skips finished outputs,
so rerunning `figures` never re-sorts. Each sorting / postprocessing folder has `run_info.json` / `post_info.json`
(versions, GPU, input, motion handling, extra preprocessing, code hashes).
