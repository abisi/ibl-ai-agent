"""001 -- Jaw / nose reference point at the image origin: stored vs recomputed series (see the project question.md).

compute_kinematics_alt takes the reference point as the 5th percentile of np.where(likelihood > 0.5, x, 0): with more
than 5% of frames below the cutoff it is (0, 0), the image corner. Per session: the jaw keypoint cloud with both
references, a 20 s example of jaw_distance / jaw_angle stored vs recomputed (dlc_helpers.fix_zero_reference), their
frame-by-frame correlation, and the lick-bout-locked mean (z) of both. Control: an unaffected session.
Outputs: figures/001_reference_<session>.png, tables/001_reference.csv
Run (haas): python 001_reference_point.py
"""
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
from common import OUT, dh, nwb  # noqa: E402

CASES = ['MH038_20250524_105329', 'MH020_20250304_161732', 'MH009_20250206_104829']


def z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def main():
    rows = []
    for s in CASES:
        stored = dh.load_dlc(nwb(s), views=('side',), fix_reference=False, mask_likelihood=False)['side']
        fixed = dh.load_dlc(nwb(s), views=('side',), fix_reference=True, mask_likelihood=False)['side']
        ts = stored.timestamps
        lik = stored.likelihood['jaw_likelihood'].to_numpy()
        with h5py.File(nwb(s), 'r') as f:
            licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
        bouts = licks[np.r_[True, np.diff(licks) > 1.0]]
        fig, axs = plt.subplots(2, 3, figsize=(16, 7.5))
        x, y = stored.data['jaw_x'].to_numpy(), stored.data['jaw_y'].to_numpy()
        ok = lik > 0.5
        sub = np.random.default_rng(0).choice(np.flatnonzero(ok), min(20000, ok.sum()), replace=False)
        ax = axs[0, 0]
        ax.plot(x[sub], y[sub], ',', color='k', alpha=0.3)
        rx = np.percentile(np.where(ok, x, 0), 5)
        ry = np.percentile(np.where(ok, y, 0), 5)
        ax.plot(rx, ry, 'rx', ms=12, mew=2, label=f'converter reference ({rx:.2f}, {ry:.2f}) mm')
        ax.plot(np.percentile(x[ok], 5), np.percentile(y[ok], 5), 'g+', ms=14, mew=2, label='valid-frame reference')
        ax.set_title(f'jaw keypoint (likelihood > 0.5); {100 * (~ok).mean():.1f}% frames below', fontsize=8)
        ax.set_xlabel('x (mm)', fontsize=8)
        ax.set_ylabel('y (mm)', fontsize=8)
        ax.legend(fontsize=7)
        t0 = bouts[len(bouts) // 2] - 5
        w = (ts >= t0) & (ts < t0 + 20)
        for ax, name in zip((axs[0, 1], axs[0, 2]), ('jaw_distance', 'jaw_angle')):
            a, b = stored.data[name].to_numpy(), fixed.data[name].to_numpy()
            ax.plot(ts[w] - t0, a[w], color='tab:red', lw=0.7, label=f'stored ({stored.units[name]})')
            ax2 = ax.twinx()
            ax2.plot(ts[w] - t0, b[w], color='tab:green', lw=0.7, label='recomputed')
            ax.set_title(f'{name}: stored (red, left axis) vs recomputed (green, right axis)', fontsize=8)
            ax.set_xlabel('time (s)', fontsize=8)
            r = np.corrcoef(np.nan_to_num(a[ok]), np.nan_to_num(b[ok]))[0, 1]
            rows.append(dict(session_id=s, series=name, frac_below=float((~ok).mean()), fixed=bool(fixed.fixed_reference),
                             median_stored=float(np.nanmedian(a[ok])), median_recomputed=float(np.nanmedian(b[ok])),
                             r_stored_recomputed=float(r)))
        for ax, name in zip((axs[1, 0], axs[1, 1]), ('jaw_distance', 'jaw_angle')):
            for data, col, lab in ((stored, 'tab:red', 'stored'), (fixed, 'tab:green', 'recomputed')):
                g, m = dh.peth(z(data.data[name].to_numpy()), ts, bouts, (-0.5, 1.0))
                ax.plot(g * 1e3, np.nanmean(m, 0), color=col, label=lab)
            ax.axvline(0, color='k', lw=0.5)
            ax.set_title(f'{name} (z) around lick-bout onsets (n={len(bouts)})', fontsize=8)
            ax.set_xlabel('time from lick-bout onset (ms)', fontsize=8)
            ax.legend(fontsize=7)
        ax = axs[1, 2]
        d = pd.DataFrame([r for r in rows if r['session_id'] == s])
        ax.axis('off')
        ax.text(0, 1, d.round(3).to_string(index=False), va='top', family='monospace', fontsize=7)
        fig.suptitle(f'{s}: jaw reference point (fixed: {fixed.fixed_reference})', fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'001_reference_{s}.png', dpi=120)
        plt.close(fig)
        print(s, 'done', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'tables' / '001_reference.csv', index=False)
    print(t.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
