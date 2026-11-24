# ssl-anatomy-tracks

Probe-track and entry-point figures of the SSL dataset (atlas space, Allen CCF 25 um via brainglobe; area groups
via `ephys_utilities.allen_utils`).

The scripts were first written as exploratory trials (user 2026-11-16, "try all those in a temp folder") and run from
`combined_results_ks4/anatomy/_viz_trials_20261116/scripts_haas_20261119/`. They are versioned here unchanged
(2026-11-23) because `4d_entry_points_atlas_surface_plain.svg` is Supplementary Figure 2 (probe localization), panel B.

| Script | Output (`combined_results_ks4/anatomy/_viz_trials_20261116/`) |
|---|---|
| `probe_viz_trials.py` | shared helpers (`OUT`, target colours, outlines, `save`) and trials 1-6 (stripes, coverage, projections, small multiples) |
| `insertion_points.py` | `4b_insertion_points_centroids.*` (entry points, target centroids, off-target flags) |
| `entry_crossref.py` | `4c_entry_crossref.*` (sample vs atlas space entry points and labels; reads 4b) |
| `entry_surface_map.py` | `4d_entry_points_atlas_surface{,_allen_colors,_plain}{,_day0_vs_expert,_Rplus_vs_Rminus}.*` (reads 4b, 4c) |

Order: `insertion_points.py` -> `entry_crossref.py` -> `entry_surface_map.py`.

`anatomy/tracks/entry_points_surface/` holds copies of the 4d outputs (identical files, checked by md5 on 2026-11-23:
`4d_entry_points_atlas_surface_plain.svg`, 2026-11-19 16:01).

Run (haas, from this folder): `PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. ~/code/unit_spikes_analysis/.venv/bin/python <script>`
