"""003 -- DLC lick times: video clock (NWB) -> session time at loading (dlc_helpers.get_dlc_lick_times; see
the project question.md).

Per session: tongue_dlc_licks with method='frame' (each lick -> its side frame's camera timestamp) and
method='offset' (video time + first side timestamp): their difference over the session, and the signed difference
to the nearest piezo lick (within 150 ms) for both methods, per session third.
Outputs: figures/003_dlc_lick_times.png, tables/003_dlc_lick_times.csv
Run (haas): python 003_dlc_lick_times.py
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
from common import OUT, SESSIONS, dh, nwb  # noqa: E402


def to_piezo(t, p, max_ms=150):
    i = np.clip(np.searchsorted(p, t), 1, len(p) - 1)
    j = np.where(np.abs(t - p[i - 1]) <= np.abs(p[i] - t), i - 1, i)
    d = (t - p[j]) * 1e3
    return np.where(np.abs(d) <= max_ms, d, np.nan)


def main():
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    for k, s in enumerate(SESSIONS):
        fr = dh.get_dlc_lick_times(nwb(s), 'frame').get('tongue', np.array([]))
        of = dh.get_dlc_lick_times(nwb(s), 'offset').get('tongue', np.array([]))
        with h5py.File(nwb(s), 'r') as f:
            p = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
        if len(fr) == 0:
            continue
        n = min(len(fr), len(of))
        axs[0].plot(fr[:n] - fr[0], (of[:n] - fr[:n]) * 1e3, lw=1, label=s)
        for method, t in (('frame', fr), ('offset', of)):
            d = to_piezo(t, p)
            for part, idx in zip(['all', 'third1', 'third2', 'third3'],
                                 [np.arange(len(t))] + np.array_split(np.arange(len(t)), 3)):
                dd = d[idx]
                rows.append(dict(session_id=s, method=method, part=part, n=int(np.isfinite(dd).sum()),
                                 median_ms=float(np.nanmedian(dd)) if np.isfinite(dd).any() else np.nan,
                                 iqr_ms=float(np.subtract(*np.nanpercentile(dd, [75, 25]))) if np.isfinite(dd).any() else np.nan))
    axs[0].set_xlabel('time in session (s)', fontsize=8)
    axs[0].set_ylabel("offset method - frame method (ms)", fontsize=8)
    axs[0].set_title('tongue_dlc_licks: single offset vs frame timestamps', fontsize=9)
    axs[0].legend(fontsize=6)
    t = pd.DataFrame(rows)
    piv = t[t.part != 'all'].pivot_table(index='session_id', columns=['method', 'part'], values='median_ms')
    for j, (method, mk) in enumerate((('frame', 'o'), ('offset', 'x'))):
        for i, s in enumerate(piv.index):
            vals = [piv.loc[s, (method, f'third{q}')] for q in (1, 2, 3)]
            axs[1].plot(np.array([1, 2, 3]) + (j - 0.5) * 0.15, vals, marker=mk, lw=0.6, color=f'C{i % 10}',
                        label=s if j == 0 else None)
    axs[1].set_xticks([1, 2, 3])
    axs[1].set_xticklabels(['third 1', 'third 2', 'third 3'])
    axs[1].set_ylabel('median DLC lick - nearest piezo lick (ms)', fontsize=8)
    axs[1].set_title('per session third: o = frame method, x = offset method', fontsize=9)
    axs[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '003_dlc_lick_times.png', dpi=130)
    plt.close(fig)
    t.to_csv(OUT / 'tables' / '003_dlc_lick_times.csv', index=False)
    print(t.pivot_table(index='session_id', columns=['method', 'part'], values='median_ms').round(1).to_string())


if __name__ == '__main__':
    main()
