# SSL ZETA responses

Across-session summary of the ZETA / latenZy pipeline of unit_spikes_analysis (`zeta_analysis/`, per-session results in
`combined_results_ks4/<mouse>/whisker_<day>/zeta/`): responsive fractions vs the ROC, positive / negative neurons per
area group, latenZy onsets, early dips. Results and report (not in git): `combined_results_ks4/ssl-zeta-responses/`.

| Script | What it does |
|---|---|
| 001_assemble.py | pools all `<mouse>_zeta_results.csv` (good + mua, keep_passive_units), joins area group / cohort / day and the matching ROC result (045 choice-type sign flip undone) -> `zeta_long.parquet` |
| 002_figures.py | Figures 1-4 (ZETA vs ROC, area fractions, latencies, early dips) and `tables/` |
| report/build_report.py | article report (skills/project-report) in `report/`; PDF / HTML with `report/build.sh` |

Run (haas): `PYTHONPATH=~/code/ephys_utilities ~/code/unit_spikes_analysis/.venv/bin/python 001_assemble.py`, then
002, then the report builder; build the PDF locally (TinyTeX).

## Decisions
- Pooled over days, cohorts and stages (no stage comparison yet).
- Unit of analysis: session; fractions are session means.
