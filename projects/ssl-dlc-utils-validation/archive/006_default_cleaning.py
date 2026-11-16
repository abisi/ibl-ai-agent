"""006 -- The default cleaning (dlc_helpers.clean_series / CLEAN_DEFAULTS) on the main series (see the project question.md).

Defaults: likelihood mask 0.5 (the series' own keypoints) + linear interpolation over gaps <= 50 ms; jaw: no
likelihood mask, running median 25 ms (5 frames) + the same gap interpolation. Per session and series: a 4 s example
(raw grey, cleaned black, frames rejected marked red), the event-locked mean raw vs cleaned (lick-bout onsets for
jaw / nose, whisker stimuli for whisker_angle and top_nose, trial starts for pupil), and for the jaw the distribution
of |raw - median-filtered| (size of the isolated wrong predictions removed).
Sessions: low jaw likelihood (AB158, MH062), low pupil likelihood (MH038 0524), setup 2 (MH009 0206), AB151.
Outputs: figures/006_default_cleaning_<session>.png, tables/006_default_cleaning.csv
Run (haas): python 006_default_cleaning.py
"""
import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pathlib  # noqa: E402
import sys  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1] / 'exploratory-analyses'  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))  # appended: the script folder and sibling folders below take precedence
from common import OUT, dh, nwb  # noqa: E402

CASES = ['AB158_20250415_105500', 'MH062_20260110_113351', 'MH038_20250524_105329', 'MH009_20250206_104829',
         'AB151_20250130_131124']
SERIES = [('side', 'jaw_distance', 'licks'), ('side', 'jaw_angle', 'licks'), ('side', 'nose_distance', 'licks'),
          ('side', 'pupil_area', 'trials'), ('top', 'whisker_angle', 'whisker'), ('top', 'top_nose_distance', 'whisker')]
WINDOW = {'licks': (-0.5, 1.0), 'whisker': (-0.5, 1.0), 'trials': (-2.0, 5.0)}


def main():
    rows = []
    for s in CASES:
        views = dh.load_dlc(nwb(s), mask_likelihood=False)
        with h5py.File(nwb(s), 'r') as f:
            licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
            t = f['intervals/trials']
            wst = t['whisker_stim_time'][:][t['whisker_stim'][:] == 1]
            starts = t['start_time'][:]
        events = {'licks': licks[np.r_[True, np.diff(licks) > 1.0]], 'whisker': wst[np.isfinite(wst)], 'trials': starts}
        fig, axs = plt.subplots(len(SERIES), 3, figsize=(18, 2.9 * len(SERIES)), squeeze=False,
                                gridspec_kw=dict(width_ratios=[2.2, 1.1, 0.8]))
        for r, (view, name, ev) in enumerate(SERIES):
            v = views.get(view)
            if v is None or name not in v.data:
                for a in axs[r]:
                    a.axis('off')
                continue
            raw = v.data[name].to_numpy()
            clean, bad = dh.clean_series(v, name)
            ts = v.timestamps
            p = dh.clean_defaults(name)
            ax = axs[r, 0]
            t0 = events['licks'][len(events['licks']) // 3] - 1
            w = (ts >= t0) & (ts < t0 + 4)
            ax.plot(ts[w] - t0, raw[w], color='0.65', lw=0.8, label='raw')
            ax.plot(ts[w] - t0, clean[w], color='k', lw=0.9, label='cleaned')
            ax.plot(ts[w][bad[w]] - t0, raw[w][bad[w]], '.', color='r', ms=3, label='rejected (likelihood)')
            for lk in licks[(licks >= t0) & (licks < t0 + 4)]:
                ax.axvline(lk - t0, color='b', lw=0.3)
            ax.set_title(f'{name} ({v.units[name]}); defaults {p}', fontsize=7)
            ax.legend(fontsize=6, loc='upper right')
            ax.set_xlabel('time (s); blue = piezo licks', fontsize=7)
            ax = axs[r, 1]
            evt = events[ev]
            for x, col, lab in ((raw, '0.6', 'raw'), (clean, 'k', 'cleaned')):
                g, m = dh.peth(x, ts, evt, WINDOW[ev])
                mean = np.nanmean(m, 0)
                ax.plot(g, mean, color=col, label=lab)
                ax.fill_between(g, mean - np.nanstd(m, 0) / np.sqrt(len(evt)), mean + np.nanstd(m, 0) / np.sqrt(len(evt)),
                                color=col, alpha=0.2)
            ax.axvline(0, color='k', lw=0.5)
            ax.set_title(f'mean +- SEM around {ev} (n={len(evt)})', fontsize=8)
            ax.legend(fontsize=6)
            ax = axs[r, 2]
            frac_nan = float(np.mean(~np.isfinite(clean)))
            if name.startswith('jaw'):
                d = np.abs(raw - clean)
                d = d[np.isfinite(d)]
                ax.hist(np.clip(d, 0, np.percentile(d, 99.9)), bins=60, color='k', log=True)
                ax.set_title(f'|raw - median filtered| ({v.units[name]})\n{100 * np.mean(d > 5 * np.median(d) + 1e-9):.2f}% frames changed > 5x median', fontsize=7)
            else:
                ax.bar(['rejected', 'NaN after\ninterpolation'], [100 * bad.mean(), 100 * frac_nan], color=['r', 'k'])
                ax.set_ylabel('% frames', fontsize=7)
            rows.append(dict(session_id=s, series=name, frac_rejected=float(bad.mean()), frac_nan_after=frac_nan,
                             median_abs_change=float(np.nanmedian(np.abs(raw - clean)))))
            for a in axs[r]:
                a.tick_params(labelsize=6)
        fig.suptitle(f'{s}: default DLC cleaning (dlc_helpers.clean_series)', fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'006_default_cleaning_{s}.png', dpi=110)
        plt.close(fig)
        print(s, 'done', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'tables' / '006_default_cleaning.csv', index=False)
    print(t.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
