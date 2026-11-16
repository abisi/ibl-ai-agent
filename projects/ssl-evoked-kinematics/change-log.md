# change-log

- 2026-11-05: project created; 000 (evoked kinematics, trajectories), 001 (publication-style figures).
- 2026-11-05: 002 (day-0 whisker sensing), 003 (day-0 pre-stimulus state), 004 (time controls, specificity),
  005 (pre-stimulus summary figures).
- 2026-11-10: clean-up. README.md added (scripts, run order, inputs / outputs); docstrings given Inputs and Run-order
  lines; question.md updated (day-0 follow-up questions, failed top views, status). No script archived: every script
  is either a cache producer used downstream (000, 003) or produces figures / tables used in the report.
  New 006_metrics_summary_figure.py (per-mouse metrics, day 0 and expert, from the 000 tables).
- 2026-11-10: report built (report/build_report.py, render.sh): 6 main figures (001 expert time courses, 006 metrics,
  001 expert tongue, 001 expert whisker, 002, 005 whisking SEM), 3 supplementary (001 day-0 time courses, 004 time
  controls, 004 specificity). Not included (kept in figures/): all 000_* first versions, 001 examples and day-0
  trajectories, 003_* (superseded by 004 / 005), 005 posture / jaw / SD variants.
