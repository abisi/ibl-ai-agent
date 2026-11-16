# SSL stimulus-detection decoding

When and where the presence of a whisker or an auditory stimulus can be decoded (stimulus vs no stimulus), per brain area,
in task and passive trials: the stimulus-presence counterpart of `ssl-stimulus-arrival-decoding` (stimulus identity).
Question and decisions: `question.md`. Results (not in git): `combined_results_ks4/ssl-stimulus-detection-decoding/` on the NAS.

| Folder | Content |
|---|---|
| `cache_active/`, `cache_passive/` | per-session rates of every event class (task: W, A, N = CR; passive: W, A, SW / SA = shams before whisker / auditory stimuli) |
| `<target>_<epoch>/` | N sweep (N = 20-500, 100 iterations x 10 shifts), target = whisker / auditory, epoch = active / passive |
| `<target>_<epoch>_final_n200/` | final N = 200 onsets (500 x 20) |
| `<target>_task_vs_passive/` | cross-epoch outputs (006-009): figures, tables |

## Scripts (`exploratory-analyses/`, run on haas; `DETECT_TARGET=whisker|auditory`, `DETECT_EPOCH=active|passive`)

| Script | What it does |
|---|---|
| _areas.py | area lists (as arrival), condition switches, output folders, colours |
| _style.py | figure conventions (copy of arrival) |
| 001_detection_pseudopop.py | `--cache` per-epoch event cache; decoding sweep with the linear-shift null (resumable) |
| 002_detection_summary.py | summaries (onsets, early accuracy), curves / accuracy-vs-N / onset-vs-accuracy figures |
| 003_matched_accuracy.py | `--plan` / `--plot`: neurons needed to match a reference area's early accuracy |
| 004_main_figures.py | main (N = 200) and summary figures per level, onset bootstrap |
| 005_final_n200.py | final N = 200 onsets (500 x 20) and main figures |
| 006_active_vs_passive.py | task vs passive: onset, early accuracy, pre-stimulus baseline, time courses |
| 007_iteration_check.py | 100 (sweep) vs 500 (final) iterations at N = 200 |
| 008_sample_sizes.py | sessions / mice / units / events per area; event-sequence check |
| 009_area_ranking.py | areas ordered by peak accuracy and by accuracy at onset |

Run order: caches (`001 --cache`), then `~/cDet2.sh` on haas (N = 200 pilot: area groups for the 4 conditions, then the 40 fine areas; then the full N sweep) with the figure watcher `~/cDetFig.sh`, then 002-004 per
condition, 003 --plan/--plot, final N = 200 runs + 005, then 006-009 per target.

## TODO
- [x] Caches, N = 200 (both levels, 4 conditions) and N sweep (done 2026-10-20, `~/log/cDet.log`).
- [x] 002, 004, 006, 008, 009 per condition / target (figure watcher `~/cDetFig.sh`, done 2026-10-20).
- [x] Repetitions: 100 iterations x 10 shifts are final for this analysis (user 2026-10-23); no 500 x 20 run, 005 / 007 not used.
- [x] Anatomy sanity check (010): Allen densities + Liu et al. 2024 per-structure table; injection sites ringed, fine-level
      own-site leave-out for Liu, grouped injection sizes (2026-10-23).
- [x] 003 matched accuracy (--plan, the extra 001 runs, --plot). (done 2026-10-24, in the report)
- [x] Report (skills/project-report): first build 2026-10-23; rebuild when 003 lands.
- [ ] ON HOLD (user 2026-10-23), separate project: re-analyse Liu et al. 2024 voxel data (Zenodo 10.5281/zenodo.13377319,
      `Horizontal_Axon/{s1,s2}/npy/`, 37 brains, CCF 25 um, horizontal = annotation_25 moveaxis(1, 0), axon density per
      labelled cell, injection site not masked) into a 50-um map comparable to the Allen densities (mask source, 2x2x2 sum,
      mirror to the right hemisphere, per-brain shape normalisation, line-balanced mean, sigma 50 um; per-cell version kept),
      then use it in 010 as a third anatomy measure.
