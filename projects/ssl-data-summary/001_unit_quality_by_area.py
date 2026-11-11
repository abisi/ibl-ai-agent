"""Run brain_wide_analysis plot_group_neural_data(plots=['data_summary']) -> plot_unit_quality_by_area (total / stable /
good units per area group, stable count annotated per group) for two populations:

  day0_learners                 day-0 sessions of learners (sheet learning_category good / moderate, recording == 1)
  all_days_no_day0_nonlearners  all sessions day >= 0; day-0 sessions only of learners (non-learners keep expert days)

Mice: reference sheet (share_internal dataset_info) exclude == 0, exclude_ephys == 0, reward_group R+ / R-;
`recording` gates day 0 only. Outputs: combined_results_ks4/data_summary/unit_quality_area_group/
unit_quality_by_area_{log,linear}_<population>.{pdf,png,svg} and unit_quality_by_area_counts_<population>.csv.

Usage (haas, brain_wide_analysis env, NAS repos first on PYTHONPATH): python 001_unit_quality_by_area.py <population>
"""
import os
import sys
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
import NWB_reader_functions as nwb_reader  # noqa: E402
from plotting_functions import plot_group_neural_data  # noqa: E402

NWB_DIR = axel_bisi_path('NWB_ks4')
OUTPUT_PATH = axel_bisi_path('combined_results_ks4')
PROC_DATA_PATH = axel_bisi_path('combined_data_ks4')
SHEET = Path('/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx') \
    if os.path.exists('/mnt/share_internal') else Path(r'M:\share_internal\Axel_Bisi_Share\dataset_info'
                                                       r'\joint_mouse_reference_weight.xlsx')
LEARNER = ('good', 'moderate')


def main(population):
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
    else:
        raise ValueError(population)
    print(f'{population}: {len(sel)} NWB files, {sel.mouse_id.nunique()} mice '
          f'(day 0: {(sel.day == 0).sum()}, day > 0: {(sel.day > 0).sum()}); NWBs without units are dropped later')

    plot_group_neural_data(sel['nwb'].tolist(), plots=['data_summary'], info_path=str(SHEET),
                           output_path=str(OUTPUT_PATH), proc_data_path=str(PROC_DATA_PATH),
                           day_to_analyze=day_to_analyze, fname_suffix=f'_{population}')


if __name__ == '__main__':
    main(sys.argv[1])
