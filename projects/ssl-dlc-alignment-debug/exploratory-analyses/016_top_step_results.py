"""016 -- Results of the top-camera step corrections of AB094_20231211 and AB087_20231017 (015), old NWB
(NWB_ks4_backup_20261116) vs corrected NWB (NWB_ks4) (user 2026-11-16).

Per session, old vs new: (a) top vs side nose-tip lag in 10 s windows over the whole session (015 lag_curve); (b) top
nose-tip speed and side jaw speed aligned to the first piezo lick of each lick trial (session order, dashed = the step);
(c) mean top nose-tip speed around the first lick for trials before vs after the step; (d) per-trial top nose-speed peak
time re first lick over the session.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/drift_review/top_step_results_<date>.pdf (+ figures/016_*.png)
Run (haas, ibl-ai-agent .venv): python 016_top_step_results.py
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
spec = importlib.util.spec_from_file_location('m015', HERE / '015_top_step_from_side.py')
m015 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m015)
a014, m006 = m015.a014, m015.m006
OUT = m015.OUT
BACKUP = m006.NWB_DIR.parent / 'NWB_ks4_backup_20261116'
SESSIONS = ['AB094_20231211_112445', 'AB087_20231017_141901']
GRID = np.round(np.arange(-0.5, 0.5001, 0.005), 3)
COL = {'old': '#d62728', 'new': '#1f77b4'}


def load(path):
    with h5py.File(path, 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        n_s, n_t = len(b['jaw_angle']['data']), len(b['whisker_angle']['data'])
        t_s, t_t = b['jaw_angle']['timestamps'][:n_s], b['whisker_angle']['timestamps'][:n_t]
        top = a014.zs(a014.speed(b, 'top_nose_tip', len(t_t)))
        jaw = a014.zs(a014.speed(b, 'jaw', len(t_s)))
        licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
        starts = f['intervals/trials/start_time'][:]
    return dict(t_s=t_s, t_t=t_t, top=top, jaw=jaw, licks=licks, starts=starts)


def first_licks(d):
    nxt = np.searchsorted(d['licks'], d['starts'])
    first = np.where(nxt < len(d['licks']), d['licks'][np.minimum(nxt, len(d['licks']) - 1)], np.nan)
    first[(first - d['starts']) > 1.5] = np.nan
    return first[np.isfinite(first)]


def page(pdf, sid):
    step = pd.read_csv(OUT / 'tables' / f'top_step_{sid}.csv')
    f_step, size = int(step.frame_start.iloc[1]), int(step.D.iloc[1] - step.D.iloc[0])
    data = {'old': load(BACKUP / f'{sid}.nwb'), 'new': load(m006.NWB_DIR / f'{sid}.nwb')}
    t_step = data['new']['t_t'][f_step]  # corrected time of the first frame after the drop
    ev = first_licks(data['new'])
    k_step = int(np.searchsorted(ev, t_step))
    fig = plt.figure(figsize=(16.5, 13))
    gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1.4, 1.0], hspace=0.45, wspace=0.3)
    # (a) top vs side lag over the session
    ax = fig.add_subplot(gs[0, :])
    for name in ('old', 'new'):
        g, ts, t0 = m015.grid_speeds(sid) if name == 'new' else (None, None, None)
        if name == 'old':
            d = data['old']
            t0 = min(np.nanmin(d['t_s']), np.nanmin(d['t_t']))
            ng = int((max(np.nanmax(d['t_s']), np.nanmax(d['t_t'])) - t0) * m015.FR) + 2
            g = {}
            with h5py.File(BACKUP / f'{sid}.nwb', 'r') as f:
                b = f['processing/behavior/BehavioralTimeSeries']
                v = {'side': a014.zs(a014.speed(b, 'nose_tip', len(d['t_s']))),
                     'top': a014.zs(a014.speed(b, 'top_nose_tip', len(d['t_t'])))}
            for view, t in (('side', d['t_s']), ('top', d['t_t'])):
                idx = np.round((t - t0) * m015.FR)
                ok = np.isfinite(idx)
                arr = np.zeros(ng)
                arr[idx[ok].astype(int)] = v[view][ok]
                g[view] = arr
        centres = np.arange(2000, len(g['top']) - 2000, 2000)
        lag = m015.lag_curve(g['top'], g['side'], centres)
        ax.plot(t0 + centres / m015.FR, lag, '.', ms=4, color=COL[name], label=f'{name} NWB')
    ax.axvline(t_step, color='k', ls='--', lw=0.8)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylim(-10, size + 10)
    ax.set_xlabel('session time (s)', fontsize=8)
    ax.set_ylabel('top vs side lag (frames)', fontsize=8)
    ax.set_title(f'(a) top vs side nose-tip lag, 10 s windows; dashed = step ({size} top frames dropped at frame '
                 f'{f_step:,}, {t_step:.1f} s)', fontsize=9)
    ax.legend(fontsize=8)
    # (b) heatmaps aligned to the first lick
    for c, (name, key, tkey, lab) in enumerate((('old', 'top', 't_t', 'top nose speed'), ('new', 'top', 't_t', 'top nose speed'),
                                                 ('old', 'jaw', 't_s', 'side jaw speed (ref)'),
                                                 ('new', 'jaw', 't_s', 'side jaw speed (ref)'))):
        d = data[name]
        m = m006.triggered(d[key], d[tkey], ev, GRID)
        ax = fig.add_subplot(gs[1, c])
        lo, hi = np.nanpercentile(m, [2, 98])
        ax.imshow(m, aspect='auto', cmap='magma', vmin=lo, vmax=hi, extent=[GRID[0] * 1e3, GRID[-1] * 1e3, len(ev), 0],
                  interpolation='nearest')
        ax.axhline(k_step, color='c', ls='--', lw=1)
        ax.axvline(0, color='w', lw=0.5)
        ax.set_title(f'(b) {lab}, {name} NWB', fontsize=9, color=COL[name])
        ax.set_xlabel('time from first piezo lick (ms)', fontsize=8)
        ax.set_ylabel('lick trial (session order); cyan = step', fontsize=8)
    # (c) mean before vs after; (d) per-trial peak time
    for c, name in enumerate(('old', 'new')):
        d = data[name]
        m = m006.triggered(d['top'], d['t_t'], ev, GRID)
        ax = fig.add_subplot(gs[2, c])
        for part, sl, ls in (('before step', slice(0, k_step), '-'), ('after step', slice(k_step, None), '--')):
            ax.plot(GRID * 1e3, np.nanmean(m[sl], 0), ls, color=COL[name], label=f'{part} ({len(ev[sl])})')
        ax.axvline(0, color='k', lw=0.4)
        ax.set_title(f'(c) mean top nose speed, {name} NWB', fontsize=9, color=COL[name])
        ax.set_xlabel('time from first piezo lick (ms)', fontsize=8)
        ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[2, 2:])
    for name in ('old', 'new'):
        d = data[name]
        m = m006.triggered(d['top'], d['t_t'], ev, GRID)
        win = (GRID >= -0.3) & (GRID <= 0.2)
        pk = np.full(len(ev), np.nan)
        ok = np.isfinite(m[:, win]).any(1)
        pk[ok] = GRID[win][np.nanargmax(np.where(np.isfinite(m[ok][:, win]), m[ok][:, win], -np.inf), 1)]
        ax.plot(ev, pk * 1e3, '.', ms=4, color=COL[name], label=f'{name}: median before / after step '
                f'{np.nanmedian(pk[:k_step]) * 1e3:.0f} / {np.nanmedian(pk[k_step:]) * 1e3:.0f} ms')
    ax.axvline(t_step, color='k', ls='--', lw=0.8)
    ax.set_xlabel('session time (s)', fontsize=8)
    ax.set_ylabel('top nose-speed peak re first lick (ms)', fontsize=8)
    ax.set_title('(d) per-trial top nose-speed peak time', fontsize=9)
    ax.legend(fontsize=7)
    fig.suptitle(f'{sid}: top-camera step correction ({size} frames = {size * 5} ms), old vs corrected NWB', fontsize=11,
                 x=0.02, ha='left')
    pdf.savefig(fig)
    fig.savefig(OUT / 'figures' / f'016_{sid}.png', dpi=70)
    plt.close(fig)


def main():
    out = OUT / f'top_step_results_{datetime.date.today()}.pdf'
    with PdfPages(out) as pdf:
        for sid in SESSIONS:
            page(pdf, sid)
    print(out)


if __name__ == '__main__':
    main()
