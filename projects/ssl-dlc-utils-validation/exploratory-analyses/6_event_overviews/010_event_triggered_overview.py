"""010 -- Event-triggered DLC traces over trials, sessions and mice, through dlc_helpers with the default cleaning
(see the project question.md). Overall data quality check before switching dependent analyses.

All NWB_ks4 sessions with DLC and trials. Signals (clean_series defaults: jaw median 15 ms, likelihood cutoffs 0.1 /
tongue 0.5, gaps <= 50 ms interpolated): jaw_distance (mm), nose_distance (mm), top_nose_distance (mm),
whisker_angle (deg), whisker speed = |whisker_velocity| (deg/s), pupil_area (mm^2), tongue_likelihood (raw).
Events: lick-bout onsets (piezo, no lick in the previous 1 s), whisker and auditory stimulus onsets (active trials
without perf 6, and passive trials). Windows -1 .. 2 s (pupil -2 .. 5 s), 10 ms bins.
Per session and signal x event: mean trace (physical units), mean of the session-z-scored trace, pre-event baseline
SD, response (max |mean - baseline| in 0 .. 0.5 s, z), fraction of NaN, n events; trial matrices kept for examples.
Outputs: cache/010_<session>.npz; figures/010_heatmap_<signal>.png (sessions x time per event, mouse means),
figures/010_metrics.png, figures/010_trials_<session>.png; tables/010_session_metrics.csv
Run (haas): python 010_event_triggered_overview.py [--plot-only]
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pathlib  # noqa: E402
import sys  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1]  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))  # appended: the script folder and sibling folders below take precedence
from common import NWB_DIR, OUT, dh  # noqa: E402

SIGNALS = [('side', 'jaw_distance'), ('side', 'nose_distance'), ('top', 'top_nose_distance'), ('top', 'whisker_angle'),
           ('top', 'whisker_speed'), ('side', 'pupil_area'), ('side', 'tongue_likelihood')]
EVENTS = ['lick', 'whisker', 'auditory']
WIN = {'pupil_area': (-2.0, 5.0)}
DT = 0.01
CACHE = OUT / 'cache'
EXAMPLES = ['MH009_20250206_104829', 'AB151_20250130_131124', 'MH062_20260110_113351', 'AB158_20250415_105500']


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def events_of(f):
    licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
    t = f['intervals/trials']
    ctx = dec(t['context'][:]) if 'context' in t else np.array(['active'] * len(t['start_time']))
    perf = t['perf'][:]
    keep = (ctx == 'passive') | (perf != 6)
    st = t['start_time'][:]
    wst = t['whisker_stim_time'][:] if 'whisker_stim_time' in t else st
    ast = t['auditory_stim_time'][:] if 'auditory_stim_time' in t else st
    w = keep & (t['whisker_stim'][:] == 1)
    a = keep & (t['auditory_stim'][:] == 1)
    ev = {'lick': licks[np.r_[True, np.diff(licks) > 1.0]] if len(licks) else licks,
          'whisker': wst[w][np.isfinite(wst[w])], 'auditory': ast[a][np.isfinite(ast[a])]}
    return ev


def one(path):
    s = os.path.basename(path)[:-4]
    out = CACHE / f'010_{s}.npz'
    if out.exists():
        return s
    try:
        views = dh.load_dlc(path, mask_likelihood=False)
        if 'side' not in views:
            return None
        with h5py.File(path, 'r') as f:
            if 'intervals/trials' not in f:
                return None
            ev = events_of(f)
        store = {}
        for view, name in SIGNALS:
            v = views.get(view)
            if v is None:
                continue
            if name == 'whisker_speed':
                if 'whisker_velocity' not in v.data:
                    continue
                x, _ = dh.clean_series(v, 'whisker_velocity')
                x = np.abs(x)
            elif name == 'tongue_likelihood':
                if name not in v.likelihood:
                    continue
                x = v.likelihood[name].to_numpy()
            else:
                if name not in v.data:
                    continue
                x, _ = dh.clean_series(v, name)
            mu, sd = np.nanmean(x), np.nanstd(x)
            for e in EVENTS:
                if len(ev[e]) < 5:
                    continue
                g, m = dh.peth(x, v.timestamps, ev[e], WIN.get(name, (-1.0, 2.0)), dt=DT)
                store[f'{name}|{e}|trials'] = m.astype(np.float32)
                store[f'{name}|{e}|grid'] = g
                store[f'{name}|{e}|sd'] = np.array([sd])
                store[f'{name}|{e}|mu'] = np.array([mu])
        np.savez_compressed(out, **store)
        return s
    except Exception as e:
        print(s, 'ERROR', type(e).__name__, e, flush=True)
        return None


def metrics():
    rows, means = [], {}
    for p in sorted(CACHE.glob('010_*.npz')):
        s = p.stem[4:]
        d = np.load(p)
        for name in [n for _, n in SIGNALS]:
            for e in EVENTS:
                k = f'{name}|{e}|trials'
                if k not in d:
                    continue
                m, g, sd = d[k], d[f'{name}|{e}|grid'], float(d[f'{name}|{e}|sd'][0])
                mean = np.nanmean(m, 0)
                base = (g >= g[0]) & (g < 0)
                post = (g >= 0) & (g <= 0.5)
                b = np.nanmean(mean[base])
                resp = np.nanmax(np.abs(mean[post] - b)) / sd if sd else np.nan
                noise = np.nanmedian(np.nanstd(m[:, base], 1)) / sd if sd else np.nan
                rows.append(dict(session_id=s, mouse=s[:5], signal=name, event=e, n_events=len(m),
                                 frac_nan=float(np.mean(np.isnan(m))), response_z=float(resp), baseline_sd_z=float(noise),
                                 snr=float(resp / noise) if noise else np.nan, scale=float(np.nanmedian(mean)), session_sd=sd))
                means[(s, name, e)] = (g, (mean - b) / sd if sd else mean * np.nan, mean)
    return pd.DataFrame(rows), means


def plot(t, means):
    for name in [n for _, n in SIGNALS]:
        fig, axs = plt.subplots(2, len(EVENTS), figsize=(6 * len(EVENTS), 13), gridspec_kw=dict(height_ratios=[3, 1]))
        for c, e in enumerate(EVENTS):
            keys = sorted(k for k in means if k[1] == name and k[2] == e)
            if not keys:
                continue
            g = means[keys[0]][0]
            mat = np.array([means[k][1] for k in keys])
            ax = axs[0, c]
            lim = np.nanpercentile(np.abs(mat), 98)
            ax.imshow(mat, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, extent=[g[0], g[-1], len(keys), 0],
                      interpolation='nearest')
            mice = [k[0][:5] for k in keys]
            edges = [i for i in range(1, len(mice)) if mice[i] != mice[i - 1]]
            for y in edges:
                ax.axhline(y, color='k', lw=0.3)
            ax.set_yticks([(a + b) / 2 for a, b in zip([0] + edges, edges + [len(mice)])])
            ax.set_yticklabels([mice[a] for a in [0] + edges], fontsize=4)
            ax.axvline(0, color='k', lw=0.5)
            ax.set_title(f'{name} around {e}: session means (z, baseline-subtracted), n={len(keys)}', fontsize=8)
            ax = axs[1, c]
            per_mouse = {}
            for k, row in zip(keys, mat):
                per_mouse.setdefault(k[0][:5], []).append(row)
            for mm, rws in per_mouse.items():
                ax.plot(g, np.nanmean(rws, 0), lw=0.5, alpha=0.6, color='tab:green' if mm.startswith('AB') else 'tab:purple')
            ax.plot(g, np.nanmean(mat, 0), color='k', lw=2, label='all sessions')
            ax.axvline(0, color='k', lw=0.5)
            ax.set_xlabel('time from event (s)', fontsize=8)
            ax.set_title('mouse means (green AB, purple MH), black = all', fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'010_heatmap_{name}.png', dpi=110)
        plt.close(fig)
    fig, axs = plt.subplots(len(SIGNALS), len(EVENTS), figsize=(5 * len(EVENTS), 2.6 * len(SIGNALS)), squeeze=False)
    for r, (_, name) in enumerate(SIGNALS):
        for c, e in enumerate(EVENTS):
            d = t[(t.signal == name) & (t.event == e)]
            ax = axs[r, c]
            ax.scatter(d.baseline_sd_z, d.response_z, s=8, c=d.frac_nan, cmap='viridis', vmin=0, vmax=0.3)
            out = d[(d.snr < d.snr.quantile(0.05)) | (d.frac_nan > 0.2)]
            for _, x in out.iterrows():
                ax.text(x.baseline_sd_z, x.response_z, x.session_id[:14], fontsize=4)
            ax.set_title(f'{name} / {e}', fontsize=8)
            ax.set_xlabel('baseline SD (z)', fontsize=7)
            ax.set_ylabel('response (z)', fontsize=7)
    fig.suptitle('Per session: response vs pre-event noise (colour = fraction NaN, 0-0.3); labels = lowest 5% SNR or > 20% NaN',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '010_metrics.png', dpi=110)
    plt.close(fig)
    for s in EXAMPLES:
        p = CACHE / f'010_{s}.npz'
        if not p.exists():
            continue
        d = np.load(p)
        fig, axs = plt.subplots(len(SIGNALS), len(EVENTS), figsize=(5 * len(EVENTS), 2.4 * len(SIGNALS)), squeeze=False)
        for r, (_, name) in enumerate(SIGNALS):
            for c, e in enumerate(EVENTS):
                k = f'{name}|{e}|trials'
                ax = axs[r, c]
                if k not in d:
                    ax.axis('off')
                    continue
                m, g = d[k], d[f'{name}|{e}|grid']
                lo, hi = np.nanpercentile(m, [2, 98]) if np.isfinite(m).any() else (0, 1)
                ax.imshow(m, aspect='auto', cmap='magma', vmin=lo, vmax=hi, extent=[g[0], g[-1], len(m), 0],
                          interpolation='nearest')
                ax.axvline(0, color='c', lw=0.5)
                ax.set_title(f'{name} / {e} (n={len(m)})', fontsize=7)
        fig.suptitle(f'{s}: trial-level traces (cleaned, physical units)', fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'010_trials_{s}.png', dpi=100)
        plt.close(fig)


def main():
    CACHE.mkdir(exist_ok=True)
    if '--plot-only' not in sys.argv:
        skip = set(os.environ.get('SKIP_SESSIONS', '').split(','))  # e.g. NWBs being reconverted right now
        files = sorted(p for p in glob.glob(str(NWB_DIR / '*.nwb')) if '_backup' not in p and os.path.basename(p)[:-4] not in skip)
        with ProcessPoolExecutor(12) as ex:
            done = [s for s in ex.map(one, files) if s]
        print('sessions cached', len(done), flush=True)
    t, means = metrics()
    t.to_csv(OUT / 'tables' / '010_session_metrics.csv', index=False)
    plot(t, means)
    print(t.groupby(['signal', 'event'])[['response_z', 'baseline_sd_z', 'snr', 'frac_nan']].median().round(3).to_string())
    low = t[(t.frac_nan > 0.2)].groupby('session_id').signal.apply(lambda x: ','.join(sorted(set(x))))
    print('sessions with > 20% NaN in a signal:')
    print(low.to_string())


if __name__ == '__main__':
    main()
