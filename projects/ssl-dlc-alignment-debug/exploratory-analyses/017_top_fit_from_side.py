"""017 -- Candidate top-camera piecewise clauses from the corrected side camera (user 2026-11-16: "try and show me first")
for the tops excluded on 2026-11-16 (MH013_20250219, MH017_20250302, AB092_20231205).

Reference: side nose-tip speed (current NWB_ks4 side timestamps, re-fitted 2026-11-16) on a 200 Hz grid, sampled at every
raw top-camera pulse time (010a cache) -> S[p]. Top nose-tip speed per top DLC frame -> x[f]. For a top frame f paired with
pulse f + D, x[f] should match S[f + D]: D per window of WIN frames (step STEP) over [-60, pulses - frames + 60] (FFT,
014 window_lags incl. the neighbour-consistency filter), isotonic (non-decreasing, frames can only be dropped) fit of the
accepted window D, steps placed by the 008 two-segment score (place_step / segments_from_steps with S as the event train).
Checks: (1) final D vs top pulses - top frames (not used by the fit); (2) top steps vs side steps in time; (3) top vs side
lag (10 s windows) with the candidate applied to the timestamps, against the current NWB.
Outputs: drift_review/tables/top_fit_<session>.csv (frame_start, frame_end, D), drift_review/figures/017_<session>.png,
drift_review/top_fit_candidates_<date>.pdf. Nothing is written to the converter.
Run (haas, ibl-ai-agent .venv): python 017_top_fit_from_side.py
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m015 = _load('m015', '015_top_step_from_side.py')
m008 = _load('m008', '008_piecewise_mapping.py')
a014, m006 = m015.a014, m015.m006
OUT, CACHE, FR = m015.OUT, m015.CACHE, m015.FR
SESSIONS = ['MH017_20250302_120918', 'MH013_20250219_170437', 'AB092_20231205_140109']
WIN, STEP = 12000, 3000


def fit(session_id):
    raw = np.load(CACHE / f'raw_pulses_{session_id}.npz')
    pulses = raw['top']
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        n_t, n_s = len(b['whisker_angle']['data']), len(b['jaw_angle']['data'])
        t_top = b['whisker_angle']['timestamps'][:n_t]
        t_side = b['jaw_angle']['timestamps'][:n_s]
        x = a014.zs(a014.speed(b, 'top_nose_tip', n_t))
        vs = a014.zs(a014.speed(b, 'nose_tip', n_s))
    t0 = min(np.nanmin(t_side), pulses[0]) - 1
    ng = int((max(np.nanmax(t_side), pulses[-1]) - t0) * FR) + 2
    side = np.zeros(ng)
    idx = np.round((t_side - t0) * FR)
    ok = np.isfinite(idx)
    side[idx[ok].astype(int)] = vs[ok]
    S = side[np.round((pulses - t0) * FR).astype(int)]  # side reference at every top pulse
    n = len(x)
    surplus = len(pulses) - n
    # current top pairing (stored timestamps -> pulse index)
    j = np.clip(np.searchsorted(pulses, t_top), 1, len(pulses) - 1)
    j = np.where(np.abs(t_top - pulses[j - 1]) <= np.abs(pulses[j] - t_top), j - 1, j)
    D_cur = j - np.arange(n)
    # windows -> isotonic -> steps
    wl = a014.window_lags(x, S, -60, surplus + 60, WIN, STEP)
    acc = wl[wl.accepted].reset_index(drop=True)
    fitv = np.round(m008.pava(acc.lag.to_numpy(float), acc.z.to_numpy()))
    m008.WIN['side'] = WIN  # search range of segments_from_steps
    bounds, lags = [0], [fitv[0]]
    for i in range(1, len(acc)):
        if fitv[i] - fitv[i - 1] >= 2:
            fa, fb = int(acc.frame[i - 1]), int(acc.frame[i])
            bstep, _, _ = m008.place_step(x, S, int(fitv[i - 1]), int(fitv[i]), fa, fb)
            bounds.append(bstep)
            lags.append(fitv[i])
    bounds.append(n)
    bounds, lags = m008.segments_from_steps(x, S, bounds, lags)
    segs = pd.DataFrame(dict(frame_start=bounds[:-1], frame_end=bounds[1:], D=np.array(lags, int)))
    D_new = np.repeat(segs.D.to_numpy(), (segs.frame_end - segs.frame_start).to_numpy())
    return dict(x=x, S=S, pulses=pulses, side=side, t0=t0, n=n, surplus=surplus, D_cur=D_cur, D_new=D_new, wl=wl,
                segs=segs, t_top=t_top)


def lag_after(r, D):
    """Top vs side lag (10 s windows) with top frame f stamped at pulse f + D."""
    pidx = np.arange(r['n']) + D
    okp = (pidx >= 0) & (pidx < len(r['pulses']))
    t = np.where(okp, r['pulses'][np.clip(pidx, 0, len(r['pulses']) - 1)], np.nan)
    top = np.zeros(len(r['side']))
    idx = np.round((t - r['t0']) * FR)
    ok = np.isfinite(idx) & (idx >= 0) & (idx < len(top))
    top[idx[ok].astype(int)] = r['x'][ok]
    centres = np.arange(2000, len(top) - 2000, 2000)
    return r['t0'] + centres / FR, m015.lag_curve(top, r['side'], centres)


def page(pdf, sid):
    r = fit(sid)
    segs = r['segs']
    side_csv = OUT / 'tables' / f'refit_{sid}_side.csv'
    side_steps_t = []
    if side_csv.exists():
        sg = pd.read_csv(side_csv)
        side_steps_t = [(int(fr), int(dd)) for fr, dd in zip(sg.frame_start.iloc[1:], np.diff(sg.D))]
    t_of = lambda f: r['pulses'][min(int(f) + int(r['D_new'][min(int(f), r['n'] - 1)]), len(r['pulses']) - 1)]  # noqa: E731
    fig = plt.figure(figsize=(16.5, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 0.9], hspace=0.45)
    ax = fig.add_subplot(gs[0, :])
    wl = r['wl']
    ax.scatter(wl.frame / FR, wl.lag, s=5, color='0.85', label='window D (rejected)')
    a = wl[wl.accepted]
    ax.scatter(a.frame / FR, a.lag, s=10, c=np.clip(a.z, 0, 30), cmap='viridis', label='window D (accepted)')
    f = np.arange(r['n'])
    ax.plot(f[::200] / FR, r['D_cur'][::200], color='#d62728', lw=1.5, label='current NWB top')
    ax.plot(f[::200] / FR, r['D_new'][::200], color='#1f77b4', lw=1.5, ls='--', label='candidate fit')
    ax.axhline(r['surplus'], color='k', ls=':', lw=0.8, label=f'top pulses - frames ({r["surplus"]:+,})')
    ax.set_ylim(min(-20, wl.lag.min() if len(a) else -20), r['surplus'] + 60)
    ax.set_xlabel('top frame / 200 (s of video)', fontsize=8)
    ax.set_ylabel('D = pulse index - top frame', fontsize=8)
    ax.set_title(f'(a) {sid} top: window estimates and candidate staircase ({len(segs) - 1} steps)', fontsize=9)
    ax.legend(fontsize=7, ncol=3)
    # (b) lag after: current vs candidate
    ax = fig.add_subplot(gs[1, :2])
    for D, lab, col in ((r['D_cur'], 'current NWB top', '#d62728'), (r['D_new'], 'candidate applied', '#1f77b4')):
        tt, lag = lag_after(r, D)
        ax.plot(tt, lag, '.', ms=3, color=col, label=f'{lab}: |lag| <= 2 in {np.mean(np.abs(lag) <= 2) * 100:.0f}% of windows')
    for fr, dd in side_steps_t:
        ax.axvline(r['pulses'][0] + fr / FR, color='0.6', lw=0.6, ls=':')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('session time (s); dotted = side steps (approx.)', fontsize=8)
    ax.set_ylabel('top vs side lag (frames, clipped)', fontsize=8)
    ax.set_title('(b) top vs side nose lag, 10 s windows', fontsize=9)
    ax.legend(fontsize=7)
    # (c) table
    ax = fig.add_subplot(gs[1:, 2])
    ax.set_axis_off()
    lines = [f'candidate top clause ({len(segs)} segments):']
    for s_ in segs.itertuples():
        lines.append(f'  frame {s_.frame_start:>9,} -> D {s_.D:+5d}   ({s_.frame_start / FR:7.1f} s video)')
    lines += ['', f'final D {segs.D.iloc[-1]:+d} vs top pulses - frames {r["surplus"]:+d}',
              f'  -> {r["surplus"] - segs.D.iloc[-1]} pulses unaccounted (frames lost after',
              '     the last step or at the video end)']
    if side_steps_t:
        lines += ['', 'side steps (010 re-fit, side frame -> +size):']
        lines += [f'  {fr:>9,} -> +{dd}' for fr, dd in side_steps_t]
    ax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=8)
    # (d) step sizes vs side
    ax = fig.add_subplot(gs[2, :2])
    st = segs.iloc[1:]
    ax.bar(st.frame_start / FR, np.diff(segs.D), width=20, color='#1f77b4', label='top steps (candidate)')
    for fr, dd in side_steps_t:
        ax.bar(fr / FR, -dd, width=20, color='0.5')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('video time (s): top steps up, side steps down (grey)', fontsize=8)
    ax.set_ylabel('step size (frames)', fontsize=8)
    ax.set_title('(c) step times and sizes, top vs side', fontsize=9)
    fig.suptitle(f'{sid}: candidate top-camera clause from the corrected side camera (not applied)', fontsize=11, x=0.02,
                 ha='left')
    pdf.savefig(fig)
    fig.savefig(OUT / 'figures' / f'017_{sid}.png', dpi=70)
    plt.close(fig)
    segs.to_csv(OUT / 'tables' / f'top_fit_{sid}.csv', index=False)
    print(sid, 'segments', segs[['frame_start', 'D']].values.tolist(), 'surplus', r['surplus'], flush=True)


def main():
    out = OUT / f'top_fit_candidates_{datetime.date.today()}.pdf'
    with PdfPages(out) as pdf:
        for sid in SESSIONS:
            try:
                page(pdf, sid)
            except Exception as ex:
                print(sid, 'ERROR', type(ex).__name__, ex, flush=True)
    print(out)


if __name__ == '__main__':
    main()
