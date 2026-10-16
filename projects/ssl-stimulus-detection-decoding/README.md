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

Run order: `~/cDet.sh` on haas (caches, N = 200 pilot for the 4 conditions, then the full N sweep), then 002-004 per
condition, 003 --plan/--plot, final N = 200 runs + 005, then 006-009 per target.

## TODO
- [ ] Caches, N = 200 pilot and N sweep (running 2026-10-16, `~/log/cDet.log`).
- [ ] 002-004, 003 matched accuracy, finals 500 x 20 + 005, 006-009 per target.
- [ ] Anatomy sanity check: decoding strength / onset per area vs projection strength from wS1 / wS2 (Liu et al. 2024 eLife;
      data source needed from the user) and Allen connectivity (ssl-sensory-spatial-maps 002 densities, SSp-bfd, SSs, AUDp,
      AUD secondary).
- [ ] Report (skills/project-report).
