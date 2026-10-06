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
| `active_final_n200/` | task trials, N = 200 only, 500 x 20 (final onsets; 21 areas first run to 1000, iterations 0-499 used); reads `active/cache_all` |
| `passive_final_n200/` | passive trials, N = 200 only, 500 x 20 (final onsets); reads `passive/cache_all` |
| `tables/`, `figures/` | cross-epoch outputs (006 task vs passive, 007 iteration check) |
| `report/` | article-style report (build_report.py on haas, render.sh locally) |

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
| 004_main_figures.py | main (N = 200) and summary figures per level, onset bootstrap; unreliable onsets (95 % range > 10 ms) flagged |
| 005_final_n200.py | final N = 200 onsets (500 x 20; only iterations 0-499 used), main figures, summaries |
| 006_active_vs_passive.py | task vs passive: onset, early accuracy, pre-stimulus baseline (paired Wilcoxon + t over areas), time courses |
| 007_iteration_check.py | 100 vs 1000 iterations at N = 200 (task trials) |
| report/build_report.py, render.sh | report sources (haas) and PDF / HTML render (local) |

## Decisions (user, 2026-10-04)

- Final onsets: N = 200 only, all 18 groups + 40 areas, task and passive trials. 2026-10-06: 500 iterations x 20 shuffles
  is the default (100 vs 1000 iterations agree, 007); `~/cArr500.sh` on haas (active remaining areas, then passive), then
  `~/cArr500_after.sh` reruns 006 and the report sources. Other N stay at pilot sampling (100 x 10).
- No further controls for now (licked-only, cohort / stage, artefact variant): passive trials only.

## TODO

- [X] Overnight 10-04/05: active sweep + 002-004, passive sweep; results moved to the project home (2026-10-06).
- [X] Iteration check (007, see report); default 500 x 20 (2026-10-06).
- [X] Task vs passive figure (006); unreliable onsets flagged (004); first report build (2026-10-06, provisional onsets).
- [ ] Final N = 200 runs at 500 x 20 (running: `~/log/cArr500.log`); then render the report locally (`report/render.sh`)
      and publish.
- [ ] Passive pre-stimulus decoding offset (corrected accuracy above 0 before onset; see report): rerun with an order-preserving null
      (circular / linear label shift within session) before interpreting task vs passive onset differences. ASK USER.
- [ ] Link single-neuron latency and population onset with all 18 groups (`build_deck_assets.py` link panel).
