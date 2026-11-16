"""015 -- Drift shift test (Harris 2021; the test behind drift_shift_test_pval / drift_abs_r in classify_units_quality)
on the joined AB155 session, run A (DREDge + Kilosort 4 drift correction), every unit.
Same recipe as unit_spikes_analysis/single_neuron_shift_test/unit_fr_motion_shift_test_harris.py
(time_binned_drift_test): firing rate in 1-s bins vs the DREDge displacement at the unit depth, first to last trial
start, Pearson, exhaustive shifts -N..N with N = 500, p_conservative = m / (N + 1) (minimum 1/501).
Joined clock (no gap; DREDge motion is in the time of the joined recording). Spike times from the joined units-only
test NWB (004), trial starts = the 010 test trial TTLs (310 trials).
Then: the junction losses (014_units_lost_post.csv: passive pre vs passive post rates) against the test result.
Output: concat_figure/015_drift_shift_test.csv (per unit, keyed probe / cluster_id)
Run (haas): /home/bisi/code/unit_spikes_analysis/.venv/bin/python 015_drift_shift_test_joined.py
"""
import sys

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, '/home/bisi/code/unit_spikes_analysis')
from single_neuron_shift_test.unit_fr_motion_shift_test_harris import (DEFAULT_CONFIG, load_motion,  # noqa: E402
                                                                       time_binned_drift_test)

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
UNITS = f'{J}/nwb_test/AB155_20250208_175103_joined_test.nwb'
TRIALS = f'{J}/nwb_test/inputs/sync_event_times/trial_start_times.txt'  # the 010 test trial TTLs
MOTION = J + '/Ephys/catgt_AB155_g0/AB155_g0_imec{p}/dredge/motion/motion'
LOST = f'{J}/concat_figure/014_units_lost_post.csv'
OUT = f'{J}/concat_figure/015_drift_shift_test.csv'
N = DEFAULT_CONFIG['drift_test_N']


def run_probe(p, rows, t0, t1):
    mo = load_motion(MOTION.format(p=p))
    out = []
    for cid, depth, spk in rows:
        r = time_binned_drift_test(spk, mo, depth, t0, t1, 1.0, N, DEFAULT_CONFIG)
        out.append(dict(probe=f'imec{p}', cluster_id=cid, depth=depth, drift_r=r['r'],
                        drift_shift_test_pval=r['p_conservative'], drift_m=r['m'], drift_error=r['error']))
    return out


def main():
    st = np.loadtxt(TRIALS)[:-1]  # the converter drops the last (session end) TTL
    t0, t1 = float(st.min()), float(st.max())
    with h5py.File(UNITS, 'r') as f:
        idx = f['units/spike_times_index'][()]
        sp = np.split(f['units/spike_times'][()], idx[:-1])
        cid = [int(x.decode() if isinstance(x, bytes) else x) for x in f['units/cluster_id'][()]]
        depth = f['units/depth'][()].astype(float)
        probe = [int(f[r].name.split('/')[-1].split('_')[0][4:]) for r in f['units/electrode_group'][()]]
    by = {}
    for p, c, d, s in zip(probe, cid, depth, sp):
        by.setdefault(p, []).append((c, d, s))
    print(f'window {t0:.1f}-{t1:.1f} s ({int(t1 - t0)} one-second bins, N = {N}), {len(cid)} units')
    res = Parallel(n_jobs=5)(delayed(run_probe)(p, rows, t0, t1) for p, rows in sorted(by.items()))
    d = pd.DataFrame([x for r in res for x in r])
    d['drift_abs_r'] = d.drift_r.abs()
    d['drift_fail'] = (d.drift_shift_test_pval < 0.01) & (d.drift_abs_r > 0.5)  # joint criterion
    lost = pd.read_csv(LOST)
    d = d.merge(lost[['probe', 'cluster_id', 'quality_label', 'bc_label', 'rate_pre', 'rate_active_last10min',
                      'rate_post']], on=['probe', 'cluster_id'], how='left', validate='one_to_one')
    d['post_pre'] = (d.rate_post + 0.05) / (d.rate_pre + 0.05)
    d.to_csv(OUT, index=False)
    print('errors:', d.drift_error.value_counts(dropna=False).to_dict())
    print('\nall units: p < 0.01 / |r| > 0.5 / joint fail, per probe')
    print(d.groupby('probe').agg(n=('cluster_id', 'size'), p01=('drift_shift_test_pval', lambda x: int((x < 0.01).sum())),
                                 r05=('drift_abs_r', lambda x: int((x > 0.5).sum())),
                                 fail=('drift_fail', 'sum')).to_string())
    g = d[d.quality_label == 'good']
    for lab, sel in (('good units, passive post < 20% of passive pre', g.post_pre < 0.2),
                     ('good units, post within x2 of pre', (g.post_pre >= 0.5) & (g.post_pre <= 2)),
                     ('silent in post (>= 1 Hz pre, < 0.05 Hz post), any label', None)):
        x = g[sel] if sel is not None else d[(d.rate_pre >= 1) & (d.rate_post < 0.05)]
        print(f'\n{lab}: n {len(x)}; p < 0.01 {int((x.drift_shift_test_pval < 0.01).sum())}, |r| > 0.5 '
              f'{int((x.drift_abs_r > 0.5).sum())}, joint fail {int(x.drift_fail.sum())}; median |r| '
              f'{x.drift_abs_r.median():.2f}, median p {x.drift_shift_test_pval.median():.3f}')
    pd.set_option('display.width', 250)
    x = g[(g.probe == 'imec4') & (g.post_pre < 0.2)].sort_values('depth')
    print('\nimec4 good units with passive post < 20% of passive pre:')
    print(x[['cluster_id', 'depth', 'rate_pre', 'rate_active_last10min', 'rate_post', 'drift_r', 'drift_shift_test_pval',
             'drift_fail']].round(3).to_string(index=False))
    x = d[(d.rate_pre >= 1) & (d.rate_post < 0.05)].sort_values(['probe', 'depth'])
    print('\nunits silent in passive post:')
    print(x[['probe', 'cluster_id', 'depth', 'quality_label', 'rate_pre', 'rate_post', 'drift_r', 'drift_shift_test_pval',
             'drift_fail']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
