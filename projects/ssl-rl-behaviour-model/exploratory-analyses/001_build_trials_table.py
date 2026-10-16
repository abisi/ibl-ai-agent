"""
001 -- Rebuild the per-trial behaviour table used by data_utils.load_data (trials_data_final_filtered.csv was not in the
repository) from the behaviour NWB files, then the interpolated targets (Lila/data_analysis_bis.ipynb, "without free licking").

Table (one row per trial):
  mouse_id, session_id, reward_group, day, bhv_type, trial_idx, trial_type (auditory / whisker / no_stim), lick_flag, perf, start_time

Selection (choices are explicit here because the filters behind the original "final_filtered" file are not documented):
  * sessions: behaviour type auditory, whisker or free_licking from the NWB session description (behaviour_<day>); context, psychometric,
    whisker_on / whisker_off sessions are skipped; day = training day relative to the first whisker day (auditory pretraining < 0)
  * trials: active only (context != passive when the trial table has a context column); perf == 6 dropped (aborted / invalid trials)
  * reward group per mouse from the mouse reference sheet (column reward_group; only R+ / R-; mice flagged exclude == 1 dropped), not from
    the session metadata, which is wrong for some sessions
  * mice kept if they have at least one auditory pretraining day (day < 0) and a whisker day 0
  trial_idx = position of the trial in its session (ordered by start time, before the trial filters), as data_utils expects.

Usage (haas):
  PYTHONPATH=~/code/NWB_reader python build_trials_table.py --nwb-dir /mnt/lsens-analysis/Axel_Bisi/NWB_combined \
      --sheet /mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx --out trials_data_rebuilt.csv --targets
"""

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

KEEP_BHV = {'auditory', 'whisker', 'free_licking'}
TT = {'auditory_trial': 'auditory', 'whisker_trial': 'whisker', 'no_stim_trial': 'no_stim'}


def bhv_and_day(description):
    """behaviour type and day from the NWB session description, as NWB_reader_functions.get_bhv_type_and_training_day_index"""
    d = description.split('_')
    if d[0] == 'free':
        return f'{d[0]}_{d[1]}', int(d[2])
    if d[1] in ('psy', 'context'):
        return f'{d[0]}_{d[1]}', int(d[2])
    if d[1] in ('on', 'off'):
        return '_'.join(d[:3]), int(d[-1])
    return d[0], int(d[1])


def read_session(path):
    from pynwb import NWBHDF5IO
    try:
        with NWBHDF5IO(str(path), 'r', load_namespaces=True) as io:
            nwb = io.read()
            bhv, day = bhv_and_day(nwb.session_description)
            if bhv not in KEEP_BHV or nwb.trials is None:
                return None, dict(file=path.name, skipped=bhv)
            t = nwb.trials.to_dataframe()
            mouse = nwb.subject.subject_id
    except Exception as e:  # noqa: BLE001
        return None, dict(file=path.name, error=repr(e)[:200])
    t = t.sort_values('start_time').reset_index(drop=True)
    t['trial_idx'] = np.arange(len(t))
    n0 = len(t)
    if 'context' in t:
        t = t[t['context'].astype(str) != 'passive']
    if 'perf' in t:
        t = t[t['perf'] != 6]
    t = t[t['trial_type'].isin(list(TT))]
    out = pd.DataFrame(dict(mouse_id=mouse, session_id=path.stem, day=day, bhv_type=bhv, trial_idx=t['trial_idx'].to_numpy(),
                            trial_type=t['trial_type'].map(TT).to_numpy(), lick_flag=t['lick_flag'].astype(int).to_numpy(),
                            perf=t['perf'].to_numpy() if 'perf' in t else np.nan, start_time=t['start_time'].to_numpy()))
    return out, dict(file=path.name, mouse_id=mouse, day=day, bhv_type=bhv, n_trials=n0, n_kept=len(out))


def build(a):
    files = sorted(Path(a.nwb_dir).glob('*.nwb'))
    with mp.get_context('spawn').Pool(a.n_jobs) as pool:
        res = pool.map(read_session, files)
    log = pd.DataFrame([r[1] for r in res])
    d = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
    ref = pd.read_excel(a.sheet)
    ref = ref[ref['reward_group'].isin(['R+', 'R-'])]
    if 'exclude' in ref:
        ref = ref[ref['exclude'].fillna(0) != 1]
    groups = ref.drop_duplicates('mouse_id').set_index('mouse_id')['reward_group']
    d['reward_group'] = d['mouse_id'].map(groups)
    no_group = sorted(d.loc[d.reward_group.isna(), 'mouse_id'].unique())
    d = d[d.reward_group.notna()]
    days = d.groupby('mouse_id')['day'].agg(lambda x: (x.min() < 0) and (0 in set(x)))
    d = d[d.mouse_id.isin(days[days].index)]
    dup = d.groupby(['mouse_id', 'day']).session_id.nunique()
    d = d.sort_values(['mouse_id', 'day', 'session_id', 'trial_idx']).reset_index(drop=True)
    d.to_csv(a.out, index=False)
    log.to_csv(Path(a.out).with_suffix('.log.csv'), index=False)
    print(f'mice {d.mouse_id.nunique()} (R+ {d[d.reward_group == "R+"].mouse_id.nunique()}, R- {d[d.reward_group == "R-"].mouse_id.nunique()}) | '
          f'trials {len(d):,} | sessions {d.session_id.nunique()}')
    print('trials per behaviour type:', d.bhv_type.value_counts().to_dict())
    print('mice without an R+/R- reward group in the sheet (dropped):', no_group)
    print('mouse-days with more than one session:', dup[dup > 1].to_dict())
    print('read errors:', int(log['error'].notna().sum()) if 'error' in log else 0)
    return d


def make_targets(csv, root):
    """as Lila/data_analysis_bis.ipynb: individual and group-mean targets, free licking removed (Lila/data_utils.load_data)"""
    import rl_data_utils as du   # removes free_licking
    d, groups = du.load_data(csv)
    ind, mean = Path(root) / 'final_individual_withoutfree', Path(root) / 'final_mean_withoutfree'
    ind.mkdir(parents=True, exist_ok=True); mean.mkdir(parents=True, exist_ok=True)
    for g in ['R+', 'R-']:
        for mid, c in du.get_all_interpolated_curves(d, groups, g).items():
            np.save(ind / f'target_{mid}.npy', dict(auditory=c['auditory'], whisker=c['whisker'], no_stim=c['no_stim'], mouse_id=mid, group=g))
        np.save(mean / f"target_{g.replace('+', 'plus').replace('-', 'minus')}.npy", du.get_mean_curves(d, groups, g))
    print('targets written to', ind, mean)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--nwb-dir', default=str(C.NWB_DIR))
    ap.add_argument('--sheet', default=None)
    ap.add_argument('--out', default=str(C.TRIALS))
    ap.add_argument('--targets-only', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=16)
    a = ap.parse_args()
    a.sheet = a.sheet or str(C.reference_sheet())
    if not a.targets_only:
        build(a)
    make_targets(a.out, C.TARGETS)
