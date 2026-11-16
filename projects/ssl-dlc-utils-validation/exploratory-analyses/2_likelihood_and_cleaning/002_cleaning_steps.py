"""002 -- What each cleaning step does to DLC traces (dlc_helpers.clean_trace_steps; see the project question.md).

Steps, cumulative: raw (as stored, physical units) -> likelihood mask (cutoff 0.5, the series' own keypoints) ->
jump (|frame-to-frame change| > JUMP) -> MAD (> 8 x MAD from the median) -> interpolation over gaps <= 50 ms ->
Gaussian smoothing (sigma 10 ms, NaN-aware). Per session and series: a 6 s example with every step, the fraction of
frames NaN after each step, the event-locked mean per step (lick-bout onsets; whisker stimuli for whisker_angle;
trial starts for pupil_area) and the pre-event noise (SD over -1 .. -0.1 s, median over events).
Outputs: figures/002_cleaning_<session>.png, tables/002_cleaning.csv
Run (haas): python 002_cleaning_steps.py
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

CASES = ['MH009_20250206_104829', 'MH038_20250524_105329', 'AB151_20250130_131124']
SERIES = [('side', 'jaw_distance', 'licks'), ('side', 'nose_distance', 'licks'), ('top', 'whisker_angle', 'whisker'),
          ('side', 'pupil_area', 'trials')]
JUMP = {'jaw_distance': 0.5, 'nose_distance': 0.5, 'whisker_angle': 20.0, 'pupil_area': 0.1}  # per frame
STEPS = ['raw', 'likelihood', 'jump', 'mad', 'interpolate', 'smooth']
COLORS = dict(zip(STEPS, ['0.6', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:blue', 'k']))
WINDOW = {'licks': (-0.5, 1.0), 'whisker': (-0.5, 1.0), 'trials': (-2.0, 5.0)}


def main():
    rows = []
    for s in CASES:
        views = dh.load_dlc(nwb(s), mask_likelihood=False)
        with h5py.File(nwb(s), 'r') as f:
            licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
            t = f['intervals/trials']
            ws = t['whisker_stim'][:] == 1
            wst = t['whisker_stim_time'][:][ws]
            starts = t['start_time'][:]
        events = {'licks': licks[np.r_[True, np.diff(licks) > 1.0]], 'whisker': wst[np.isfinite(wst)], 'trials': starts}
        fig, axs = plt.subplots(len(SERIES), 4, figsize=(20, 3.4 * len(SERIES)), squeeze=False,
                                gridspec_kw=dict(width_ratios=[2.2, 0.8, 1.2, 0.8]))
        for r, (view, name, ev) in enumerate(SERIES):
            v = views.get(view)
            if v is None or name not in v.data:
                continue
            gate = [l for l in dh.likelihood_series(name) if l in v.likelihood]
            lik = np.min(np.vstack([v.likelihood[l].to_numpy() for l in gate]), 0) if gate else None
            steps = dh.clean_trace_steps(v.data[name].to_numpy(), v.timestamps, likelihood=lik, cutoff=0.5,
                                         jump_thresh=JUMP[name], mad_thresh=8.0, max_gap_s=0.05, smooth_sigma_s=0.01)
            ts = v.timestamps
            ax = axs[r, 0]
            t0 = events['licks'][len(events['licks']) // 2] - 2
            w = (ts >= t0) & (ts < t0 + 6)
            span = np.nanpercentile(steps['raw'][0][w], 99) - np.nanpercentile(steps['raw'][0][w], 1)
            for k, st in enumerate(STEPS):
                ax.plot(ts[w] - t0, steps[st][0][w] - k * span, color=COLORS[st], lw=0.7, label=st)
            ax.set_title(f'{name} ({v.units[name]}): 6 s example, steps offset downwards', fontsize=8)
            ax.set_xlabel('time (s)', fontsize=8)
            ax.set_yticks([])
            ax.legend(fontsize=6, ncol=6, loc='upper right')
            ax = axs[r, 1]
            frac = [np.mean(~np.isfinite(steps[st][0])) for st in STEPS]
            ax.bar(range(len(STEPS)), np.array(frac) * 100, color=[COLORS[st] for st in STEPS])
            ax.set_xticks(range(len(STEPS)))
            ax.set_xticklabels(STEPS, rotation=40, fontsize=7)
            ax.set_ylabel('% frames NaN', fontsize=8)
            ax.set_title('frames NaN after each step', fontsize=8)
            ax = axs[r, 2]
            evt = events[ev]
            for st in STEPS:
                g, m = dh.peth(steps[st][0], ts, evt, WINDOW[ev])
                _, b = dh.peth(steps[st][0], ts, evt, (-1.0, -0.1))
                mean = np.nanmean(m, 0)
                ax.plot(g, mean - np.nanmean(mean[g < 0]), color=COLORS[st], lw=1, label=st)
                resp = np.nanmax(np.abs(mean - np.nanmean(mean[g < 0])))
                noise = np.nanmedian(np.nanstd(b, 1))
                rows.append(dict(session_id=s, series=name, step=st, frac_nan=float(np.mean(~np.isfinite(steps[st][0]))),
                                 frac_events_all_nan=float(np.mean(np.all(np.isnan(m), 1))), response=float(resp),
                                 baseline_sd=float(noise), snr=float(resp / noise) if noise else np.nan))
            ax.axvline(0, color='k', lw=0.5)
            ax.set_title(f'mean around {ev} (n={len(evt)}), baseline-subtracted', fontsize=8)
            ax.set_xlabel('time from event (s)', fontsize=8)
            ax = axs[r, 3]
            d = pd.DataFrame([x for x in rows if x['session_id'] == s and x['series'] == name])
            ax.plot(range(len(STEPS)), d.snr, 'o-', color='k')
            ax.set_xticks(range(len(STEPS)))
            ax.set_xticklabels(STEPS, rotation=40, fontsize=7)
            ax.set_title('response / pre-event noise SD', fontsize=8)
        fig.suptitle(f'{s}: DLC cleaning steps (cutoff 0.5, jump, MAD 8, gaps <= 50 ms, sigma 10 ms)', fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / f'002_cleaning_{s}.png', dpi=110)
        plt.close(fig)
        print(s, 'done', flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'tables' / '002_cleaning.csv', index=False)
    print(t.pivot_table(index=['session_id', 'series'], columns='step', values='frac_nan')[STEPS].round(4).to_string())
    print(t.pivot_table(index=['session_id', 'series'], columns='step', values='snr')[STEPS].round(2).to_string())


if __name__ == '__main__':
    main()
