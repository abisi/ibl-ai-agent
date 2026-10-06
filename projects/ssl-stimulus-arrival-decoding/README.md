# SSL stimulus-arrival decoding

When whisker vs auditory stimulus information arrives in each brain area: pseudo-population decoding of stimulus
modality in the first tens of milliseconds, onset per area, information vs number of neurons.
Companion project: `ssl-sensory-spatial-maps`. Question and scope: `question.md`.
Results (not in git), all under `combined_results_ks4/ssl-stimulus-arrival-decoding/` on the NAS (moved 2026-10-06; the old
`_stimulus_arrival*` paths are symlinks; mapping in `results_home_manifest_20261006.tsv`):

| Folder | Content |
|---|---|
| `active/` | task (active) trials, N = 20-500, 100 iterations x 10 shuffles; matched accuracy |
| `passive/` | passive trials (pre + post), N = 20-500, 100 x 10 |
| `final_n200/` | task (active) trials, N = 200 only, 1000 x 20 (final onsets); reads `active/cache_all` |

## Method (short)

20 sessions with replacement x N/20 neurons of the area per session; pseudo-trials by balanced reuse; L2 logistic
regression per time bin (20-ms bins, 2-ms steps; 50-ms bins for the full time course), 3-fold CV, inner 2-fold for C;
corrected accuracy = real - mean of 10 within-session label shuffles; 100 iterations; N = 20-500. Bin above chance:
5th percentile > 0. Onset: first bin above chance with >= 80 % of the next 25 ms above chance. Whisker artefact
(-10 to +5 ms) replaced by Poisson spikes. Areas: all 18 area groups and the 40 best-sampled fine areas.

## Scripts (`exploratory-analyses/`, run on haas; `ARRIVAL_EPOCH=active|passive`)

| Script | What it does |
|---|---|
| _areas.py | area lists (groups, top-40 fine areas), output folder per epoch, colours (allen_utils palette) |
| _style.py | figure conventions |
| 001_arrival_pseudopop.py | `--cache` per-session rate cache; decoding sweep (area x N x iteration chunks, resumable); `--runs-file` for extra runs |
| 002_arrival_summary.py | summaries (onsets, early accuracy), time-course / accuracy-vs-N / onset-vs-accuracy figures |
| 003_matched_accuracy.py | `--plan` / `--plot`: neurons needed to match a reference area's early accuracy |
| 004_main_figures.py | main (N = 200) and summary figures per level, onset bootstrap |

## Decisions (user, 2026-10-04)

- Final onsets: N = 200 only, 1000 iterations x 20 shuffles, all 18 groups + 40 areas, after the passive sweep
  (`~/cArrFinal.sh` on haas -> `combined_results_ks4/ssl-stimulus-arrival-decoding/final_n200/`, `005_final_n200.py`). Other N stay at
  pilot sampling (100 x 10).
- No further controls for now (licked-only, cohort / stage, artefact variant): passive trials only.

## TODO

- [X] Overnight: active sweep -> 002 -> 003 -> 004 ("ACTIVE ALL DONE", 10-05 04:02), passive sweep ("PASSIVE DONE",
      10-06 01:29). Final N = 200 run (active only) running since 10-06 01:34 ("FINAL N200 DONE"). Compare with `combined_results_ks4/_snapshots/2026-10-04_2200/stimulus_arrival/`.
- [X] Results moved to `combined_results_ks4/ssl-stimulus-arrival-decoding/{active,passive}` (2026-10-06); `final_n200/` is
      moved automatically when the run ends (`~/mvArrFinal.sh` on haas, log `~/log/mvArrFinal.log`).
- [ ] Use the final N = 200 onsets in report and deck once available.
- [ ] Link single-neuron latency and population onset with all 18 groups (`build_deck_assets.py` link panel).
