# Fig 2G barrel flatmap, refined barrels

Writes `Figures/Figure2_sensory_maps_explore_2026-11-17/G_maps/barrel_flatmap_refined/` (manuscript folder).

1. `barrel_refined_units.py`: SSp-bfd units of the barrel row (`../barrel_flatmap_example_bfd_only/barrel_flatmap_units.csv`)
   -> refined barrel and flat position from the mouse's Anatomy `wS1_barrel_registration/` export (final user-picked D2
   fit, use_local mice; others keep the global registration of the NWB) -> `barrel_refined_units.csv`.
2. `barrel_activity_map.py` and `barrel_activity_map_cohorts.py` (R+ / R- rows): maps -> `barrel_final_units.csv`,
   `barrel_activity_map*.png/pdf`.

Run on haas from /tmp (as made, 2026-11-20; rerun 2026-11-22 after the final export):
`~/code/wS1_barrel_registration/.venv/bin/python barrel_refined_units.py`, then
`~/code/unit_spikes_analysis/.venv/bin/python -c "import sys; sys.path.insert(0, '/home/bisi/code/wS1_barrel_registration'); exec(open('barrel_activity_map_cohorts.py').read())"`
(same for `barrel_activity_map.py`).
