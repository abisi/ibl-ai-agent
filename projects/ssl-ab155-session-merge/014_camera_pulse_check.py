"""014 -- Camera on / off pulses: single-trial DLC aligned to the whisker stimulus with and without their removal.
The whisker stimulus (coil) moves the whisker particle directly, so its deflection must start at or after the stimulus
TTL (t = 0): a frame-time assignment that shows the deflection before t = 0 is wrong.
  removed (as in the NWB, 010): per part, the pulse > 50 ms away from its neighbour (part 1 first = camera on, part 2
    last = camera off) is dropped -> frames = pulses of each part, one to one (595,292 + 77,288)
  not removed: the 672,580 DLC rows take the first 672,580 of the 672,582 synced pulses -> every frame gets the time
    of the previous pulse (part 1) or of the previous pulse with the part-1 last pulse as first part-2 frame
Signal: whisker particle displacement (px) from its mean position at -50..-5 ms, frames with likelihood >= 0.6.
Onset per trial: first frame after -30 ms with displacement > 5 px and > baseline mean + 5 SD.
Output: concat_figure/014_camera_pulse_check.png/.pdf
Run (local, conda env nwb): python 014_camera_pulse_check.py
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np

A = r'M:\analysis\Axel_Bisi'
J = rf'{A}\Misc\SessionToConcat\AB155_20250208_175103_joined'
NWB = rf'{J}\nwb_test\AB155_20250208_175103_joined_full_test_gap.nwb'
PULSES = rf'{J}\Ephys\catgt_AB155_g0\sync_event_times\cam0_frame_times.txt'  # side camera (cam2 in the NWB)
OUT = rf'{J}\concat_figure\014_camera_pulse_check'
WIN = (-0.05, 0.06)
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})


def aligned(ts, x, ev):
    """Per trial: frame times relative to the event and the displacement from the baseline position."""
    rows = []
    for e in ev:
        a, b = np.searchsorted(ts, [e + WIN[0], e + WIN[1]])
        t = ts[a:b] - e
        rows.append((t, x[a:b]))
    return rows


def displacement(t, xy):
    base = (t >= -0.05) & (t < -0.005)
    d = np.linalg.norm(xy - np.nanmean(xy[base], 0), axis=1)
    return d, d[base]


def onset(t, d, b):
    thr = max(5.0, np.nanmean(b) + 5 * np.nanstd(b))
    i = np.flatnonzero((t > -0.03) & (d > thr))
    return t[i[0]] if len(i) else np.nan


def main():
    f = h5py.File(NWB, 'r')
    tj, t2 = f['acquisition/recording_gap/timestamps'][()]
    bts = f['processing/behavior/BehavioralTimeSeries']
    ts_nwb = bts['particle_x/timestamps'][()]
    xy = np.c_[bts['particle_x/data'][()], bts['particle_y/data'][()]].astype(float)
    xy[bts['particle_likelihood/data'][()] < 0.6] = np.nan
    p = np.loadtxt(PULSES)
    p = np.where(p >= tj, p + (t2 - tj), p)
    ts_keep = p[:len(ts_nwb)]
    tr = f['intervals/trials']
    wst = tr['whisker_stim_time'][()]
    wst = wst[np.isfinite(wst) & (tr['whisker_stim'][()] == 1)]
    parts = {'part 1 (all whisker trials)': wst[wst < tj], 'part 2 (passive post)': wst[wst >= t2]}
    versions = {'on / off pulses removed (NWB)': ts_nwb, 'not removed': ts_keep}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), gridspec_kw={'width_ratios': [1, 1, 0.8]})
    for r, (pname, ev) in enumerate(parts.items()):
        ons = {}
        for c, (vname, ts) in enumerate(versions.items()):
            ax = axs[r, c]
            o, traces = [], []
            for t, x in aligned(ts, xy, ev):
                d, b = displacement(t, x)
                ax.plot(t * 1e3, d, color='0.6', lw=0.3, alpha=0.4)
                o.append(onset(t, d, b))
                traces.append(np.interp(np.arange(-50, 60, 5) / 1e3, t, d))
            ax.plot(np.arange(-50, 60, 5), np.nanmean(traces, 0), color='C0' if c == 0 else 'C3', lw=1.2)
            ax.axvline(0, color='k', lw=0.6, ls='--')
            ax.set_ylim(0, np.nanpercentile(np.r_[traces], 99) * 1.1)
            ax.set_title(f'{pname}\n{vname}', fontsize=7)
            ax.set_xlabel('time from whisker stim (ms)')
            ax.set_ylabel('particle displacement (px)')
            ons[vname] = np.array(o) * 1e3
        ax = axs[r, 2]
        bins = np.arange(-27.5, 40, 5)
        for c, (vname, o) in enumerate(ons.items()):
            ax.hist(o[np.isfinite(o)], bins=bins, histtype='step', lw=1.2, color='C0' if c == 0 else 'C3',
                    label=f'{vname}: {np.mean(o[np.isfinite(o)] < 0):.0%} before 0 (n={np.isfinite(o).sum()})')
            print(f'{pname} | {vname}: onset median {np.nanmedian(o):.1f} ms, before stim '
                  f'{np.mean(o[np.isfinite(o)] < 0):.0%} of {np.isfinite(o).sum()} trials')
        ax.axvline(0, color='k', lw=0.6, ls='--')
        ax.set_xlabel('deflection onset (ms)')
        ax.set_ylabel('trials')
        ax.legend(fontsize=5, frameon=False, loc='lower left', bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}.{ext}', dpi=300)
    print(f'saved {OUT}.png')


if __name__ == '__main__':
    main()
