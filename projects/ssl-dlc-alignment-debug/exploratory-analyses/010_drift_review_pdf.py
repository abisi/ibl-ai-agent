"""010 -- One review PDF of the DLC frame -> camera pulse misalignment in the drift sessions (see ../question.md, 005,
008), with the proposed mapping and its check on trial-aligned jaw traces.

Three pairings of DLC frame f with camera pulse g(f), per camera:
  naive    : g(f) = f (start-aligned, what the converter does without a clause)
  current  : the pairing stored in NWB_ks4 now (stored timestamp -> nearest raw pulse)
  re-fit   : step function g(f) = f + D(f) (D non-decreasing: frames dropped, pulses kept), re-estimated here from the
             correlation of the DLC signal (side: tongue likelihood, top: whisker speed) with ALL raw piezo licks
             (010a; top: licks + whisker stimuli), with the 008 functions (windowed lags, isotonic fit, steps placed on
             events); D = lag - lead, side lead 5 frames, top lead 0 (own lag: the cameras can start differently). For the sessions fixed on 2026-11-03/09 it checks the
             converter clause against the new DLC; for the unfixed ones it is the proposal.
Pages, one per session: (a) offset D(f) over the video per camera (windowed lags, the three pairings); (b) pulse gaps;
(c) jaw distance aligned to trial start under each pairing (lick trials in session order, white dots = first piezo
lick); (d) jaw opening - first lick per trial over the session (flat at about -0.1 s when the pairing is right).
Last page: summary table (fraction of lick trials with jaw opening within [-0.25, +0.05] s of the first lick).
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/drift_review/drift_review_<date>.pdf,
tables/drift_review_summary.csv, tables/refit_<session>_<view>.csv (frame_start, frame_end, D on the raw pulse index)
Run (haas, ibl-ai-agent .venv, after 010a): python 010_drift_review_pdf.py
"""
import datetime
import importlib.util
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m008', HERE / '008_piecewise_mapping.py')
m008 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m008)
m006 = m008.m006
OUT = m006.OUT / 'drift_review'
CACHE = m006.OUT / 'cache'

SESSIONS = {  # session: state of the NWB_ks4 file now (NWB_converter converters/DLC_to_nwb.py clauses)
    'AB116_20240726_094206': 'fixed: end-aligned, DLC NaN before frame 434,963 (side + top)',
    'MH013_20250219_170437': 'fixed: piecewise clause (side 7 steps, top 6 steps)',
    'MH017_20250302_120918': 'fixed: piecewise clause (side 7 steps, top 2 steps)',
    'MH004_20241206_093158': 'fixed: piecewise clause (1 step)',
    'MH028_20250503_112316': 'fixed: end-aligned (camera test run before the video)',
    'AB092_20231205_140109': 'NOT fixed: start-aligned',
    'AB085_20231005_152636': 'NOT fixed: start-aligned',
    'AB095_20231212_141728': 'NOT fixed: start-aligned',
}
COL = {'naive': '0.55', 'current': '#1f77b4', 're-fit': '#ff7f0e'}
GRID = np.round(np.arange(-1.0, 2.0001, 0.01), 3)
OK_HALF = 0.1  # jaw opening - first lick within this of the mapping's median (s) = consistent
WGRID = np.round(np.arange(-0.5, 1.0001, 0.005), 3)
W_HALF = 0.05  # whisker-speed onset after the whisker stimulus within this of the mapping's median (s) = consistent
MIN_LICKS_PER_WIN = 25
WIN0, STEP0 = dict(m008.WIN), dict(m008.STEP)


def nwb_data(session_id):
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        side_sig = 'jaw_distance' if session_id in m008.JAW_SESSIONS else 'tongue_likelihood'
        d = dict(side=b[side_sig]['data'][:].astype(float), side_ts=b['jaw_angle']['timestamps'][:],
                 jaw=b['jaw_distance']['data'][:].astype(float))
        if 'whisker_angle' in b:
            w = b['whisker_angle']['data'][:].astype(float)
            d['top'] = np.abs(np.diff(w, prepend=w[0]))
            d['top_ts'] = b['whisker_angle']['timestamps'][:]
        d['starts'] = f['intervals/trials/start_time'][:]
        w = f['intervals/trials/whisker_stim_time'][:]
        d['wstim'] = np.sort(w[np.isfinite(w)])
    return d


def current_index(stored, pulses, n):
    """Pulse index paired with each frame in the NWB (nearest raw pulse to the stored timestamp)."""
    ts = stored[:n]
    i = np.clip(np.searchsorted(pulses, ts), 1, len(pulses) - 1)
    j = np.where(np.abs(ts - pulses[i - 1]) <= np.abs(pulses[i] - ts), i - 1, i)
    err = np.nanmax(np.abs(pulses[j] - ts)) * 1e3
    g = np.full(len(ts), -1)
    g[np.isfinite(ts)] = j[np.isfinite(ts)]
    return np.r_[g, np.full(n - len(g), -1)], err


def refit(x, pulses, licks, view, d0_side=None):
    """Step-function lag (pulse index - frame) from windowed lick correlation; returns segments and windows."""
    n = len(x)
    extra = len(pulses) - n
    lo, hi = -60, max(extra, 0) + 60  # frames after the last pulse (extra < 0) do not shift the start
    e = m008.train(pulses, licks, len(pulses))
    # 2023 AB sessions log only response-window licks (~0.12/s): widen the windows to >= 25 licks each
    rate = len(licks) / max(pulses[-1] - pulses[0], 1)
    win = int(np.clip(200 * MIN_LICKS_PER_WIN / max(rate, 1e-3), m008.WIN[view], 120000))
    m008.WIN.update(side=win if view == 'side' else m008.WIN['side'], top=win if view == 'top' else m008.WIN['top'])
    m008.STEP[view] = win // 2
    wl = m008.window_lags(x, e, lo, hi, win, win // 2)
    if not len(wl):
        return None, wl
    wl['D'] = wl.lag - (m008.SIDE_LEAD if view == 'side' else 0)
    wl['accepted'] = False
    ok = ((wl.z >= m008.MIN_Z) & (wl.n_events >= m008.MIN_EVENTS)).to_numpy()
    wl['accepted'] = m008.consistent(wl.lag.to_numpy(), ok)
    acc = wl[wl.accepted].reset_index(drop=True)
    if len(acc) < 2:
        return None, wl
    fit = np.round(m008.pava(acc.lag.to_numpy(float), acc.z.to_numpy()))
    bounds, lags = [0], [fit[0]]
    for i in range(1, len(acc)):
        if fit[i] - fit[i - 1] >= 3:
            fa, fb = int(acc.f0[i - 1] + m008.WIN[view] // 2), int(acc.f0[i] + m008.WIN[view] // 2)
            b, _, _ = m008.place_step(x, e, int(fit[i - 1]), int(fit[i]), fa, fb)
            bounds.append(b)
            lags.append(fit[i])
    bounds.append(n)
    bounds, lags = m008.segments_from_steps(x, e, bounds, lags)
    segs = pd.DataFrame(dict(frame_start=bounds[:-1], frame_end=bounds[1:], lag=lags))
    lead = m008.SIDE_LEAD if view == 'side' else 0  # top: own lag (cameras can start differently: MH028, AB116)
    segs['D'] = (segs.lag - lead).astype(int)
    wl['D'] = wl.lag - lead
    return segs, wl


def index_from_segs(segs, n, n_pulses):
    if segs is None:
        return None
    D = np.repeat(segs.D.to_numpy(), (segs.frame_end - segs.frame_start).to_numpy())
    g = np.arange(n) + D
    g[(g < 0) | (g >= n_pulses)] = -1
    return g


def times(g, pulses):
    t = np.full(len(g), np.nan)
    t[g >= 0] = pulses[g[g >= 0]]
    return t


def onsets(m, grid=GRID, t_max=1.5):
    """Half-maximum crossing of each trial's trace after the event (relative to its pre-event median)."""
    pre, post = grid < 0, (grid >= 0) & (grid <= t_max)
    base = np.nanmedian(m[:, pre], 1)
    peak = np.nanmax(np.where(post, m, -np.inf), 1)
    thr = base + 0.5 * (peak - base)
    above = (m >= thr[:, None]) & post[None, :]
    has = above.any(1) & np.isfinite(base) & (peak - base > 1.0)
    on = np.full(len(m), np.nan)
    on[has] = grid[np.argmax(above[has], 1)]
    return on


def session_page(pdf, session_id):
    raw = np.load(CACHE / f'raw_pulses_{session_id}.npz')
    licks = raw['licks']
    d = nwb_data(session_id)
    starts = d['starts']
    nxt = np.searchsorted(licks, starts)
    first = np.full(len(starts), np.nan)
    ok = nxt < len(licks)
    first[ok] = licks[nxt[ok]]
    first[(first - starts) > 1.5] = np.nan  # first piezo lick within 1.5 s of trial start (lick trials)
    lt = np.flatnonzero(np.isfinite(first))

    maps, rows, segs_out = {}, [], {}
    d0_side = None
    for view in ('side', 'top'):
        if view not in d or view not in raw.files:
            continue
        m008.WIN.update(WIN0)
        m008.STEP.update(STEP0)
        pulses = raw[view]
        x = np.nan_to_num(m006.zs(d[view]))
        n = len(x)
        g_cur, err = current_index(d[f'{view}_ts'], pulses, n)
        # top: whisker speed follows the whisker stimulus (and licks); start offset tied to the side camera (008)
        events = licks if view == 'side' else np.sort(np.r_[licks, d['wstim']])
        segs, wl = refit(x, pulses, events, view, d0_side)
        if view == 'side' and segs is not None:
            d0_side = int(segs.D.iloc[0])
        if segs is not None:
            segs.to_csv(OUT / 'tables' / f'refit_{session_id}_{view}.csv', index=False)
        maps[view] = dict(n=n, pulses=pulses, wl=wl, segs=segs, err=err,
                          g={'naive': np.where(np.arange(n) < len(pulses), np.arange(n), -1), 'current': g_cur,
                             're-fit': index_from_segs(segs, n, len(pulses))})
        segs_out[view] = segs

    fig = plt.figure(figsize=(16.5, 15.5))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1.3, 1.0, 0.9], hspace=0.45, wspace=0.42)
    # (a) offset curves
    for c, view in enumerate(('side', 'top')):
        ax = fig.add_subplot(gs[0, 3 * c:3 * c + 2])
        if view not in maps:
            ax.set_axis_off()
            ax.set_title(f'{view}: no DLC', fontsize=9)
            continue
        mp = maps[view]
        wl = mp['wl']
        if len(wl) and 'D' in wl:
            ax.scatter(wl.frame / 200, wl.D, s=5, color='0.85', label='window lag (all)', zorder=1)
            a = wl[wl.accepted]
            ax.scatter(a.frame / 200, a.D, s=9, color='k', label='window lag (accepted)', zorder=2)
        f = np.arange(mp['n'])
        for name, g in mp['g'].items():
            if g is None:
                continue
            v = np.where(g >= 0, g - f, np.nan).astype(float)
            ax.plot(f[::200] / 200, v[::200], color=COL[name], lw=1.6 if name != 'naive' else 1.0, label=name, zorder=3,
                    ls='--' if name == 're-fit' else '-')
        ax.set_xlabel('video time (DLC frame / 200 Hz, s)', fontsize=8)
        ax.set_ylabel('offset D = pulse index - frame', fontsize=8)
        segs = mp['segs']
        ax.set_title(f"{view}: {mp['n']:,} frames, {len(mp['pulses']):,} pulses ({len(mp['pulses']) - mp['n']:+,}); "
                     f"re-fit {0 if segs is None else len(segs) - 1} steps; NWB ts vs raw pulse max {mp['err']:.2f} ms",
                     fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        # (b) pulse gaps
        ax = fig.add_subplot(gs[0, 3 * c + 2])
        p = mp['pulses']
        ipi = np.diff(p) * 1e3
        ax.plot(p[1:] - p[0], ipi, '.', ms=2, color='0.3')
        big = ipi > 7.5
        ax.set_yscale('log')
        ax.set_xlabel('time from first pulse (s)', fontsize=8)
        ax.set_ylabel('inter-pulse interval (ms)', fontsize=8)
        ax.set_title(f'{view} pulses: {big.sum()} gaps > 7.5 ms', fontsize=8)
        ax.tick_params(labelsize=7)

    # (c) jaw heatmaps aligned to trial start; (d) jaw opening - first lick
    summary = dict(session_id=session_id, state=SESSIONS[session_id], n_lick_trials=len(lt))
    side = maps.get('side')
    jaw = m006.zs(d['jaw'])
    delta = {}
    for c, name in enumerate(('naive', 'current', 're-fit')):
        ax = fig.add_subplot(gs[1, 2 * c:2 * c + 2])
        g = None if side is None else side['g'][name]
        if g is None:
            ax.set_axis_off()
            ax.set_title(f'{name}: no mapping (too few accepted windows)', fontsize=9)
            continue
        t = times(g, side['pulses'])
        m = m006.triggered(jaw, t, starts[lt], GRID)
        lo_, hi_ = np.nanpercentile(m, [2, 98]) if np.isfinite(m).any() else (0, 1)
        ax.imshow(m, aspect='auto', cmap='magma', vmin=lo_, vmax=hi_, extent=[GRID[0], GRID[-1], len(lt), 0],
                  interpolation='nearest')
        ax.scatter(first[lt] - starts[lt], np.arange(len(lt)) + 0.5, s=1.2, color='w', lw=0)
        ax.axvline(0, color='c', lw=0.6)
        on = onsets(m)
        delta[name] = on - (first[lt] - starts[lt])
        med = np.nanmedian(delta[name])
        frac = np.mean(np.abs(delta[name] - med) <= OK_HALF)
        summary[f'side_frac_consistent_{name}'] = round(float(frac), 3)
        summary[f'side_median_jaw_minus_lick_ms_{name}'] = round(float(med * 1e3), 1)
        ax.set_title(f'SIDE jaw distance (z), {name}: {100 * frac:.0f}% trials jaw opening within +-{int(OK_HALF * 1e3)} ms '
                     f'of median ({med * 1e3:.0f} ms) re first lick', fontsize=8, color=COL[name])
        ax.set_xlabel('time from trial start (s); white = first piezo lick', fontsize=8)
        ax.set_ylabel('lick trial (session order)', fontsize=8)
        ax.tick_params(labelsize=7)
    top = maps.get('top')
    ws = d['wstim']
    for c, name in enumerate(('naive', 'current', 're-fit')):
        ax = fig.add_subplot(gs[2, 2 * c:2 * c + 2])
        g = None if top is None else top['g'][name]
        if g is None or not len(ws):
            ax.set_axis_off()
            ax.set_title(f'TOP {name}: no mapping' if top is not None else 'no top DLC', fontsize=9)
            continue
        m = m006.triggered(np.nan_to_num(m006.zs(d['top'])), times(g, top['pulses']), ws, WGRID)
        lo_, hi_ = np.nanpercentile(m, [2, 98]) if np.isfinite(m).any() else (0, 1)
        ax.imshow(m, aspect='auto', cmap='magma', vmin=lo_, vmax=hi_, extent=[WGRID[0], WGRID[-1], len(ws), 0],
                  interpolation='nearest')
        ax.axvline(0, color='c', lw=0.6)
        on = onsets(m, WGRID, 0.5)
        med = np.nanmedian(on)
        frac = np.mean(np.abs(on - med) <= W_HALF)
        summary[f'top_frac_consistent_{name}'] = round(float(frac), 3)
        summary[f'top_median_onset_ms_{name}'] = round(float(med * 1e3), 1)
        ax.scatter(on, np.arange(len(ws)) + 0.5, s=1.2, color='c', lw=0)
        ax.set_title(f'TOP whisker speed (z), {name}: {100 * frac:.0f}% whisker trials onset within +-{int(W_HALF * 1e3)} ms '
                     f'of median ({med * 1e3:.0f} ms); cyan = onset', fontsize=8, color=COL[name])
        ax.set_xlabel('time from whisker stimulus (s)', fontsize=8)
        ax.set_ylabel('whisker trial (session order)', fontsize=8)
        ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[3, :4])
    for name, dl in delta.items():
        ax.plot(starts[lt], np.clip(dl, -1, 1), '.', ms=3, color=COL[name], label=name, alpha=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel('session time (s)', fontsize=8)
    ax.set_ylabel('jaw opening - first lick (s, clipped)', fontsize=8)
    ax.legend(fontsize=7, markerscale=3)
    ax.tick_params(labelsize=7)
    ax = fig.add_subplot(gs[3, 4:])
    ax.set_axis_off()
    lines = [f'NWB now: {SESSIONS[session_id]}', f'{len(lt)} lick trials, {len(licks):,} raw piezo licks', '']
    for view, segs in segs_out.items():
        if segs is None:
            lines.append(f'{view} re-fit: none (too few accepted windows)')
            continue
        lines.append(f'{view} re-fit segments (start frame -> D):')
        lines += [f'   {int(r.frame_start):>9,} -> {int(r.D):+d}' for r in segs.head(12).itertuples()]
        lines.append(f'   (pulses - frames = {len(maps[view]["pulses"]) - maps[view]["n"]:+,})')
        summary[f'refit_steps_{view}'] = len(segs) - 1
    ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=7.5)
    fig.suptitle(f'{session_id}: DLC frame -> camera pulse pairing', fontsize=12, x=0.02, ha='left')
    pdf.savefig(fig)
    fig.savefig(OUT / 'figures' / f'{session_id}.png', dpi=75)
    plt.close(fig)
    print(summary)
    return summary


def main():
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    rows = []
    sessions = sys.argv[1:] or list(SESSIONS)
    out = OUT / f"drift_review_{datetime.date.today()}{'' if not sys.argv[1:] else '_test'}.pdf"
    with PdfPages(out) as pdf:
        for s in sessions:
            try:
                rows.append(session_page(pdf, s))
            except Exception as ex:
                print(s, 'ERROR', type(ex).__name__, ex)
        t = pd.DataFrame(rows)
        t.to_csv(OUT / 'tables' / 'drift_review_summary.csv', index=False)
        cols = ['session_id', 'state', 'n_lick_trials'] + [c for c in t.columns if c.startswith(('side_', 'top_', 'refit_'))]
        fig, ax = plt.subplots(figsize=(16.5, 11.7))
        ax.set_axis_off()
        tab = ax.table(cellText=t[cols].astype(str).values, colLabels=[c.replace('_', ' ') for c in cols], loc='upper left',
                       cellLoc='left')
        tab.auto_set_font_size(False)
        tab.set_fontsize(6.5)
        tab.scale(1, 1.6)
        ax.set_title('Summary per pairing. side: fraction of lick trials with jaw opening - first lick within 100 ms of the '
                     'median; top: fraction of whisker trials with whisker-speed onset within 50 ms of the median',
                     fontsize=10, loc='left')
        pdf.savefig(fig)
        plt.close(fig)
    print(out)


if __name__ == '__main__':
    main()
