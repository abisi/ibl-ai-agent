"""018 -- Top-camera DLC PSTHs before vs after the candidate top clauses of 017 (user 2026-11-16: "show how PSTHs of DLC
traces would look after correction"). Nothing is written to the converter or the NWB.

Per session: top camera under the current NWB pairing vs the 017 candidate (top frame f stamped at raw top pulse
f + D(f), 010a cache); side camera (current NWB, already corrected) as reference.
Rows: (1) top nose-tip speed aligned to the first piezo lick of each lick trial (session order), current | candidate |
side jaw speed; (2) top whisker angle (z) aligned to the auditory stimulus, current | candidate | side jaw distance;
(3) mean of row 1 per session third; (4) mean of row 2 per session third. Dashed lines = candidate top steps.
Outputs: drift_review/top_fit_psth_<date>.pdf (+ figures/018_<session>.png)
Run (haas, ibl-ai-agent .venv, after 017): python 018_top_fit_psth.py
"""
import datetime
import importlib.util
import pathlib

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('a014', HERE / '014_dataset_alignment_audit.py')
a014 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a014)
m006 = a014.m006
OUT = m006.OUT / 'drift_review'
CACHE = m006.OUT / 'cache'
SESSIONS = ['MH013_20250219_170437', 'AB092_20231205_140109', 'MH017_20250302_120918']
GRID = np.round(np.arange(-0.5, 1.0001, 0.005), 3)
THIRDS = ('#1b9e77', '#d95f02', '#7570b3')


def s(v):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def load(sid):
    with h5py.File(m006.NWB_DIR / f'{sid}.nwb', 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        n_t, n_s = len(b['whisker_angle']['data']), len(b['jaw_angle']['data'])
        d = dict(t_top=b['whisker_angle']['timestamps'][:n_t], t_side=b['jaw_angle']['timestamps'][:n_s],
                 nose=a014.zs(a014.speed(b, 'top_nose_tip', n_t)),
                 wa=a014.zs(b['whisker_angle']['data'][:n_t]),
                 jaw_speed=a014.zs(a014.speed(b, 'jaw', n_s)),
                 jaw=a014.zs(b['jaw_distance']['data'][:n_s]))
        licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
        tr = f['intervals/trials']
        starts = tr['start_time'][:]
        aud = tr['auditory_stim_time'][:][s(tr['trial_type'][:]) == 'auditory_trial']
    nxt = np.searchsorted(licks, starts)
    first = np.where(nxt < len(licks), licks[np.minimum(nxt, len(licks) - 1)], np.nan)
    first[(first - starts) > 1.5] = np.nan
    d['first'] = first[np.isfinite(first)]
    d['aud'] = np.sort(aud[np.isfinite(aud)])
    pulses = np.load(CACHE / f'raw_pulses_{sid}.npz')['top']
    segs = pd.read_csv(OUT / 'tables' / f'top_fit_{sid}.csv')
    D = np.repeat(segs.D.to_numpy(), (segs.frame_end - segs.frame_start).to_numpy())[:n_t]
    p = np.arange(n_t) + D
    d['t_top_new'] = np.where(p < len(pulses), pulses[np.minimum(p, len(pulses) - 1)], np.nan)
    d['step_t'] = [d['t_top_new'][fr] for fr in segs.frame_start.iloc[1:] if fr < n_t]
    return d


def heat(ax, m, ev, steps, title, color='k'):
    lo, hi = np.nanpercentile(m, [2, 98]) if np.isfinite(m).any() else (0, 1)
    ax.imshow(m, aspect='auto', cmap='magma', vmin=lo, vmax=hi, extent=[GRID[0] * 1e3, GRID[-1] * 1e3, len(ev), 0],
              interpolation='nearest')
    ax.axvline(0, color='c', lw=0.6)
    for t in steps:
        ax.axhline(np.searchsorted(ev, t), color='w', lw=0.8, ls='--')
    ax.set_title(title, fontsize=8, color=color)
    ax.set_xlabel('ms', fontsize=7)
    ax.set_ylabel('trial (session order)', fontsize=7)
    ax.tick_params(labelsize=6)


def thirds(ax, m, title):
    for k, idx in enumerate(np.array_split(np.arange(len(m)), 3)):
        ax.plot(GRID * 1e3, np.nanmean(m[idx], 0), color=THIRDS[k], lw=1.2, label=f'{["early", "mid", "late"][k]} ({len(idx)})')
    ax.axvline(0, color='k', lw=0.4)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel('ms', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)


def page(pdf, sid):
    d = load(sid)
    fig, axs = plt.subplots(4, 3, figsize=(15, 17), gridspec_kw=dict(height_ratios=[1.3, 1.3, 0.8, 0.8]))
    rows = [('first', 'first piezo lick', ('nose', 'top nose speed'), ('jaw_speed', 'side jaw speed (ref)')),
            ('aud', 'auditory stimulus', ('wa', 'top whisker angle'), ('jaw', 'side jaw distance (ref)'))]
    for r, (evk, evlab, (topk, toplab), (sidek, sidelab)) in enumerate(rows):
        ev = d[evk]
        mats = {'current': m006.triggered(d[topk], d['t_top'], ev, GRID),
                'candidate': m006.triggered(d[topk], d['t_top_new'], ev, GRID),
                'side': m006.triggered(d[sidek], d['t_side'], ev, GRID)}
        heat(axs[r, 0], mats['current'], ev, d['step_t'], f'{toplab}, CURRENT NWB, re {evlab}', '#d62728')
        heat(axs[r, 1], mats['candidate'], ev, d['step_t'], f'{toplab}, CANDIDATE, re {evlab}', '#1f77b4')
        heat(axs[r, 2], mats['side'], ev, d['step_t'], f'{sidelab}, re {evlab}')
        thirds(axs[r + 2, 0], mats['current'], f'{toplab}, current: mean per session third')
        thirds(axs[r + 2, 1], mats['candidate'], f'{toplab}, candidate: mean per session third')
        thirds(axs[r + 2, 2], mats['side'], f'{sidelab}: mean per session third')
    fig.suptitle(f'{sid}: top-camera DLC PSTHs, current NWB vs candidate clause (dashed = candidate top steps)',
                 fontsize=11, x=0.02, ha='left')
    fig.tight_layout()
    pdf.savefig(fig)
    fig.savefig(OUT / 'figures' / f'018_{sid}.png', dpi=60)
    plt.close(fig)
    print(sid, len(d['first']), 'lick trials,', len(d['aud']), 'auditory trials', flush=True)


def main():
    out = OUT / f'top_fit_psth_{datetime.date.today()}.pdf'
    with PdfPages(out) as pdf:
        for sid in SESSIONS:
            page(pdf, sid)
    print(out)


if __name__ == '__main__':
    main()
