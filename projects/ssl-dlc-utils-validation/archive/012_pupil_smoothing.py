"""012 -- Pupil area smoothing (dlc_helpers CLEAN_DEFAULTS['pupil']: running median 0.25 s, then Gaussian sigma 0.1 s)
against the stored trace and other settings, on day-0 example sessions.

Spike index per session: fraction of frames where |area - running median over 50 ms| > 20% of the session median area
(single-frame / few-frame spikes), before and after each setting. Examples: the 3 day-0 sessions with the highest
raw spike index and one at the median.
Per example: 60 s of trace (stored vs default), a 4 s zoom on the largest spike (every setting), and stimulus-locked
session means (whisker hit, passive whisker pre; baseline -1 .. 0 s subtracted) for every setting.
Outputs: figures/012_pupil_smoothing_examples.png, figures/012_pupil_spike_index.png, tables/012_pupil_spike_index.csv
Run (haas): python 012_pupil_smoothing.py
"""
import importlib
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pathlib  # noqa: E402
import sys  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1] / 'exploratory-analyses'  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))  # appended: the script folder and sibling folders below take precedence
sys.path.insert(0, str(_EA / '6_event_overviews'))  # 011_day0_posture_movement (imported below)
from common import OUT, dh  # noqa: E402

m011 = importlib.import_module('011_day0_posture_movement')

SETTINGS = {'stored': dict(median_s=None, smooth_sigma_s=None),
            'median 0.1 s': dict(median_s=0.1, smooth_sigma_s=None),
            'median 0.25 s': dict(median_s=0.25, smooth_sigma_s=None),
            'median 0.25 s + gauss 0.1 s (default)': dict(median_s=0.25, smooth_sigma_s=0.1),
            'median 0.5 s + gauss 0.1 s': dict(median_s=0.5, smooth_sigma_s=0.1)}
COLORS = ['0.6', 'tab:blue', 'tab:orange', 'k', 'tab:red']


def spike_index(x, ts):
    k = max(3, int(round(0.05 / np.median(np.diff(ts)))) | 1)
    fin = np.isfinite(x)
    ref = median_filter(np.where(fin, x, np.nanmedian(x)), size=k, mode='nearest')
    return float(np.mean(np.abs(x - ref)[fin] > 0.2 * np.nanmedian(x)))


def load(path):
    v = dh.load_dlc(path, views=('side',), mask_likelihood=False).get('side')
    if v is None or 'pupil_area' not in v.data:
        return None
    return v


def score(path):
    try:
        v = load(path)
        if v is None:
            return None
        row = dict(session_id=os.path.basename(path)[:-4])
        for name, kw in SETTINGS.items():
            x, _ = dh.clean_series(v, 'pupil_area', **kw)
            row[name] = spike_index(x, v.timestamps)
        return row
    except Exception as e:
        print(path, 'ERROR', type(e).__name__, e, flush=True)
        return None


def main():
    sess = m011.sessions_and_cohorts()
    with ProcessPoolExecutor(12) as ex:
        rows = [r for r in ex.map(score, list(sess)) if r]
    t = pd.DataFrame(rows).sort_values('stored', ascending=False)
    t.to_csv(OUT / 'tables' / '012_pupil_spike_index.csv', index=False)
    print(t.describe().round(5).to_string())
    print(t.head(8).round(5).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5, 5))
    for (name, _), c in zip(SETTINGS.items(), COLORS):
        ax.plot(np.sort(t[name].to_numpy())[::-1] * 100, color=c, label=name)
    ax.set_yscale('symlog', linthresh=0.01)
    ax.set_xlabel('day-0 sessions (sorted)')
    ax.set_ylabel('spike frames (%)')
    ax.set_title('Pupil spike index: |area - 50 ms median| > 20% of median area', fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '012_pupil_spike_index.png', dpi=130)
    plt.close(fig)

    examples = list(t.session_id[:3]) + [t.session_id.iloc[len(t) // 2]]
    fig, axs = plt.subplots(len(examples), 4, figsize=(16, 4 * len(examples)), squeeze=False)
    nwb_of = {os.path.basename(p)[:-4]: p for p in sess}
    for r, s in enumerate(examples):
        v = load(nwb_of[s])
        ts = v.timestamps
        tr = {name: dh.clean_series(v, 'pupil_area', **kw)[0] for name, kw in SETTINGS.items()}
        raw = tr['stored']
        # largest spike: biggest deviation from the 50 ms median
        k = max(3, int(round(0.05 / np.median(np.diff(ts)))) | 1)
        dev = np.abs(raw - median_filter(np.nan_to_num(raw, nan=np.nanmedian(raw)), size=k, mode='nearest'))
        i0 = int(np.nanargmax(dev))
        t0 = ts[i0]
        a = (ts > t0 - 30) & (ts < t0 + 30)
        ax = axs[r, 0]
        ax.plot(ts[a] - t0, raw[a], color='0.6', lw=0.5, label='stored')
        ax.plot(ts[a] - t0, tr['median 0.25 s + gauss 0.1 s (default)'][a], color='k', lw=1, label='default')
        ax.set_title(f'{s} ({sess[nwb_of[s]]}): 60 s, spike index {t.set_index("session_id").stored[s] * 100:.2f}%',
                     fontsize=8)
        ax.set_ylabel('pupil area (mm$^2$)')
        ax.legend(fontsize=6)
        ax = axs[r, 1]
        a = (ts > t0 - 2) & (ts < t0 + 2)
        for (name, x), c in zip(tr.items(), COLORS):
            ax.plot(ts[a] - t0, x[a], color=c, lw=0.8 if name != 'stored' else 0.6, label=name)
        ax.set_title('4 s around the largest spike', fontsize=8)
        ax.legend(fontsize=5)
        with h5py.File(nwb_of[s], 'r') as f:
            ev = m011.trial_events(f)
        for c_, tt in ((2, 'whisker hit'), (3, 'passive W pre')):
            ax = axs[r, c_]
            if len(ev[tt]) == 0:
                ax.axis('off')
                continue
            for (name, x), col in zip(tr.items(), COLORS):
                g, m = dh.peth(x, ts, ev[tt], (-1.0, 3.0), dt=0.01)
                m = m - np.nanmean(m[:, g < 0], 1, keepdims=True)
                ax.plot(g, np.nanmean(m, 0), color=col, lw=0.8)
            ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.set_title(f'{tt} (n={len(ev[tt])}), baseline-subtracted mean', fontsize=8)
            ax.set_xlabel('time from stimulus (s)')
        axs[r, 0].set_xlabel('time from largest spike (s)')
        axs[r, 1].set_xlabel('time from largest spike (s)')
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '012_pupil_smoothing_examples.png', dpi=110)
    plt.close(fig)
    print('examples', examples)


if __name__ == '__main__':
    main()
