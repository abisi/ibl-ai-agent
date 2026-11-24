"""Run brain_wide_analysis plot_group_neural_data(plots=['data_summary']) -> plot_unit_quality_by_area (total / stable /
good units per area group, stable count annotated per group) and plot_summary_area_overview (mice / recordings / units
per area, R+ / R- mirrored; units as stable, quality_label breakdown or quality_label good) for three populations:

  day0_learners                 day-0 sessions of learners (sheet learning_category good / moderate, recording == 1)
  all_days_no_day0_nonlearners  learners: all sessions day >= 0; day-0 sessions only of learners (non-learners keep
                                expert days)
  all_mice                      all sessions day >= 0 of every included mouse

Mice: reference sheet (share_internal dataset_info) exclude == 0, exclude_ephys == 0, reward_group R+ / R-;
`recording` gates day 0 only. Outputs: combined_results_ks4/data_summary/unit_quality_area_group/
unit_quality_by_area_{vertical,horizontal}_{log,linear}_<population>.{pdf,png,svg},
unit_quality_by_area_counts_<population>.csv; data_summary/<area level>/summary_overview_{stable,quality_breakdown,good}_*.

Unit table cache (user 2026-11-16): every full run writes the processed unit table (scalar columns, as passed to the
plots) to data_summary/unit_table_cache/unit_table_<population>.parquet (+ _provenance.json). A grid-only run
(area_count_grid) replots from that cache in seconds; pass --reload to rebuild it from the NWBs (needed after NWB,
sheet, drift-test or area-label changes).

Usage (haas, brain_wide_analysis env, NAS repos first on PYTHONPATH):
  python 001_unit_quality_by_area.py <population> [plots, default data_summary; area_count_grid / area_acronym_count_grid = a grid alone]
                                     [--reload]
"""
import json
import os
import sys
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

for p in [axel_bisi_path('brain_wide_analysis'), axel_bisi_path('Github', 'ephys_utilities'),
          axel_bisi_path('NWB_reader')]:
    sys.path.insert(0, str(p))
sys.path.append(str(axel_bisi_path('Github', 'allen_utils')))  # standalone copy, imported by neural_utils
from plotting_functions import plot_area_group_count_grid, plot_group_neural_data  # noqa: E402

NWB_DIR = axel_bisi_path('NWB_ks4')
OUTPUT_PATH = axel_bisi_path('combined_results_ks4')
PROC_DATA_PATH = axel_bisi_path('combined_data_ks4')
SHEET = Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx') \
    if os.path.exists('/mnt/share_internal') else Path(r'M:\share_internal\Axel_Bisi_Share\dataset_info'
                                                       r'\joint_mouse_reference_weight.xlsx')
LEARNER = ('good', 'moderate')
CACHE_DIR = OUTPUT_PATH / 'data_summary' / 'unit_table_cache'


def main(population, plots=("data_summary",), reload=False):
    plots = list(plots)
    cache = CACHE_DIR / f'unit_table_{population}.parquet'
    grids = {'area_count_grid': 'area_group', 'area_acronym_count_grid': 'area_acronym_custom'}  # user 2026-11-17
    if plots and set(plots) <= set(grids) and cache.exists() and not reload:
        unit_table = pd.read_parquet(cache)
        print(f'{population}: {len(unit_table)} units from cache {cache} '
              f'(built {datetime.fromtimestamp(cache.stat().st_mtime):%Y-%m-%d %H:%M}; --reload to rebuild)')
        for plot in plots:
            level = grids[plot]
            out = OUTPUT_PATH / 'data_summary' / f'{level.replace("_custom", "")}_count_grid'
            out.mkdir(parents=True, exist_ok=True)
            plot_area_group_count_grid(unit_table, output_folder=str(out), area_col=level,
                                       fname_prefix=f'{level.replace("_custom", "")}_count_grid',
                                       fname_suffix=f'_{population}')
        return

    import NWB_reader_functions as nwb_reader  # full runs only: grid replots need no pynwb (local PC, user 2026-11-17)
    sheet = pd.read_excel(SHEET).rename(columns={'mouse_name': 'mouse_id'})
    sheet = sheet[(sheet['exclude'] == 0) & (sheet['exclude_ephys'] == 0) & sheet['reward_group'].isin(['R+', 'R-'])]
    day0_ok = set(sheet.loc[(sheet['recording'] == 1) & sheet['learning_category'].isin(LEARNER), 'mouse_id'])
    mice = set(sheet['mouse_id'])

    # session NWBs only (MOUSE_YYYYMMDD_HHMMSS.nwb)
    nwbs = sorted(str(NWB_DIR / f) for f in os.listdir(NWB_DIR)
                  if f.endswith('.nwb') and len(f[:-4].split('_')) == 3 and f.split('_')[0] in mice)
    with ThreadPoolExecutor(32) as ex:
        days = list(ex.map(lambda f: nwb_reader.get_bhv_type_and_training_day_index(f)[1], nwbs))
    sel = pd.DataFrame({'nwb': nwbs, 'day': days, 'mouse_id': [Path(f).name.split('_')[0] for f in nwbs]})
    sel = sel[sel['day'] >= 0]
    if population == 'day0_learners':
        sel = sel[(sel['day'] == 0) & sel['mouse_id'].isin(day0_ok)]
        day_to_analyze = 'learning'
    elif population == 'all_days_no_day0_nonlearners':
        sel = sel[(sel['day'] > 0) | sel['mouse_id'].isin(day0_ok)]
        day_to_analyze = 'all'
    elif population == 'all_mice':
        day_to_analyze = 'all'
    else:
        raise ValueError(population)
    print(f'{population}: {len(sel)} NWB files, {sel.mouse_id.nunique()} mice '
          f'(day 0: {(sel.day == 0).sum()}, day > 0: {(sel.day > 0).sum()}); NWBs without units are dropped later')

    plot_group_neural_data(sel['nwb'].tolist(), plots=plots, info_path=str(SHEET),
                           output_path=str(OUTPUT_PATH), proc_data_path=str(PROC_DATA_PATH),
                           day_to_analyze=day_to_analyze, fname_suffix=f'_{population}',
                           unit_table_cache=str(cache))
    prov = dict(population=population, built=datetime.now().isoformat(timespec='seconds'),
                day_to_analyze=day_to_analyze, sheet=str(SHEET), nwb_dir=str(NWB_DIR),
                nwb_files=[Path(f).name for f in sel['nwb']], n_mice=int(sel.mouse_id.nunique()),
                note='unit table as passed to plot_group_neural_data plots (after keep_shared_areas), scalar columns')
    (CACHE_DIR / f'unit_table_{population}_provenance.json').write_text(json.dumps(prov, indent=1))


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a != '--reload']
    main(args[0], *([args[1].split(",")] if len(args) > 1 else []), reload='--reload' in sys.argv)
