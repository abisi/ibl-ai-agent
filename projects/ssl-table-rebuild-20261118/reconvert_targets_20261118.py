"""Reconvert NWB_ks4 sessions whose probe target changed in joint_probe_insertion_info.xlsx (user 2026-11-18: MH034 imec3 PPC -> wS1,
PPC-near-wS1 review, projects/ssl-ppc-target-review), then check
and swap them in. Converter: ~/code/NWB_converter_local (identical to the local NWB_converter), env
~/code/nwb_converter_env, AVX-512 disabled (memory: converter channel-sort ties). Configs from Axel_Bisi/data (as the
2026-11-11 reconversion).
Checks vs the current NWB_ks4 file: electrode-group target ('area' in location) changed to the expected value and the
other groups unchanged; trials (all columns), units (ids, cluster ids, spike counts, spike times, ccf_atlas labels and
coordinates), piezo licks identical. Only then: old file -> NWB_ks4_backup_20261118/, new file -> NWB_ks4.
Run (haas): NPY_DISABLE_CPU_FEATURES=... ~/code/nwb_converter_env/bin/python reconvert_targets_20261118.py <session>"""
import ast
import contextlib
import os
import shutil
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np

CONVERTER = os.path.expanduser('~/code/NWB_converter_local')
sys.path.insert(0, CONVERTER)
os.chdir(CONVERTER)
import NWB_conversion_parallel as ncp  # noqa: E402

AB = Path('/mnt/lsens-analysis/Axel_Bisi')
NWB = AB / 'NWB_ks4'
STAGE = AB / 'NWB_ks4_reconv_20261118'
BACKUP = AB / 'NWB_ks4_backup_20261118'   # earlier backups (NWB_ks4_backup_20261117*) must stay untouched
EXPECTED = {'MH034_20250514_104756': {'imec3': 'wS1'}, 'MH034_20250516_102648': {'imec3': 'wS1'}}  # user 2026-11-18


def groups(path):
    with h5py.File(path, 'r') as f:
        eg = f['general/extracellular_ephys']
        return {k.split('_')[0]: ast.literal_eval(eg[k].attrs['location'])['area'] for k in eg
                if 'location' in eg[k].attrs}


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind in 'fc' and b.dtype.kind in 'fc':
        return bool(np.allclose(a, b, equal_nan=True, rtol=0, atol=1e-9))
    return bool(np.array_equal(a, b))


def compare(old, new):
    out = {}
    with h5py.File(old, 'r') as fo, h5py.File(new, 'r') as fn:
        for k in fo['intervals/trials']:
            if k in ('id',) or not isinstance(fo[f'intervals/trials/{k}'], h5py.Dataset):
                continue
            out[f'trials/{k}'] = k in fn['intervals/trials'] and same(fo[f'intervals/trials/{k}'][()], fn[f'intervals/trials/{k}'][()])
        for k in ['id', 'cluster_id', 'spike_times', 'spike_times_index', 'ccf_atlas_acronym', 'ccf_atlas_ap',
                  'ccf_atlas_ml', 'ccf_atlas_dv', 'bc_label']:
            if k in fo['units']:
                out[f'units/{k}'] = k in fn['units'] and same(fo[f'units/{k}'][()], fn[f'units/{k}'][()])
        p = 'processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'
        if p in fo:
            out['piezo'] = p in fn and same(fo[p][()], fn[p][()])
    return out


def main(s):
    STAGE.mkdir(exist_ok=True)
    log = STAGE / f'{s}.log'
    new, old = STAGE / f'{s}.nwb', NWB / f'{s}.nwb'
    if not new.exists():
        with open(log, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                ncp.convert_data_to_nwb(config_file=str(AB / 'data' / s[:5] / s / f'config_{s}.yaml'),
                                        output_folder=str(STAGE), with_time_string=False, experimenter='Axel_Bisi')
                print('CONVERSION OK')
            except Exception:  # noqa: BLE001
                traceback.print_exc()
                print('CONVERSION FAILED')
    if not new.exists():
        print(s, 'CONVERSION FAILED, see', log)
        return
    g_old, g_new = groups(old), groups(new)
    exp = EXPECTED[s]
    ok_groups = all(g_new.get(p) == t for p, t in exp.items()) and all(g_new.get(p) == g_old.get(p) for p in g_old if p not in exp)
    checks = compare(old, new)
    bad = [k for k, v in checks.items() if not v]
    print(s, '| target old', {p: g_old.get(p) for p in exp}, '-> new', {p: g_new.get(p) for p in exp},
          '| other groups unchanged:', all(g_new.get(p) == g_old.get(p) for p in g_old if p not in exp),
          '|', len(checks), 'data checks,', 'all identical' if not bad else f'DIFFER: {bad}')
    if ok_groups and not bad:
        BACKUP.mkdir(exist_ok=True)
        shutil.move(str(old), str(BACKUP / old.name))
        shutil.move(str(new), str(old))
        print(s, 'SWAPPED IN (old ->', BACKUP / old.name, ')')
    else:
        print(s, 'NOT swapped, new file left in', new)


if __name__ == '__main__':
    main(sys.argv[1])
