"""015 -- Review of candidate pupil outlier mice (day 0) on tracking quality only (user, 2026-11-04: outliers = bad
tracking or no visible pupil, i.e. jumps; strong responses, e.g. passive pre novelty, are not a criterion).

Per day-0 session (pupil through dh.clean_series = pupil QC states + running median 0.1 s): QC verdict, % valid
frames, % jump frames (QC), residual jumps after cleaning (% of finite frames > 20% away from a 1 s running median),
% trials kept (>= 80% of -1 .. 4 s finite), baseline pupil (median mm^2).
Per candidate mouse, one figure: (a) whole-session pupil (QC-cleaned, % of session median) with QC states, (b) trial
matrices (% change, -1 .. 4 s) for passive W pre, whisker hit, auditory hit, (c) side-video eye crops with the four DLC
points at a typical frame and at the 5 largest residual jumps among QC-valid frames.
Outputs: tables/015_pupil_tracking_metrics.csv, figures/015_pupil_tracking_metrics.png,
figures/015_pupil_review_<mouse>.png
Run (haas): python 015_pupil_review.py [mouse ...]
"""
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
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
sys.path.insert(0, str(_EA / '6_event_overviews'))  # 011_day0_posture_movement (imported below)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # 013_pupil_frames (this folder, ahead of old copies)
from common import OUT, dh  # noqa: E402

m011 = importlib.import_module('011_day0_posture_movement')
m013 = importlib.import_module('013_pupil_frames')
CANDIDATES = ['AB144', 'MH029', 'MH021', 'AB119', 'AB116', 'AB126', 'AB136', 'AB140', 'AB102', 'AB159']
REVIEW_TYPES = ['passive W pre', 'whisker hit', 'auditory hit']
WIN, DT = (-1.0, 4.0), 0.02
STATE_COL = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:red'}  # blink, squint, failed


def residual_jumps(x, ts, rel=0.2, win_s=1.0):
    w = max(3, int(round(win_s / np.median(np.diff(ts)))) | 1)
    ref = np.array(pd.Series(x).rolling(w, center=True, min_periods=w // 4).median(), dtype=float)
    fin = np.isfinite(x) & np.isfinite(ref)
    return (np.abs(x - ref) > rel * ref) & fin, ref


def metrics(item):
    path, cohort = item
    s = os.path.basename(path)[:-4]
    try:
        v = dh.load_dlc(path, views=('side',)).get('side')
        if v is None or 'pupil_area' not in v.data:
            return None
        ok, verdict = dh.pupil_session_ok(s)
        _, qc = dh.load_pupil_qc(s)
        x, _ = dh.clean_series(v, 'pupil_area')
        jumps, _ = residual_jumps(x, v.timestamps)
        with h5py.File(path, 'r') as f:
            ev = m011.trial_events(f, 'stim')
        n_all = n_kept = 0
        for tt in m011.TRIAL_TYPES:
            if len(ev[tt]) == 0:
                continue
            _, m = dh.peth(x, v.timestamps, ev[tt], WIN, dt=DT)
            n_all += len(m)
            n_kept += int((np.isfinite(m).mean(1) >= 0.8).sum())
        qc = qc or {}
        return dict(session_id=s, mouse=s[:5], cohort=cohort, verdict=verdict, pct_valid=qc.get('pct_valid'),
                    pct_jump_qc=qc.get('pct_jump'), pct_blink=qc.get('pct_blink'), pct_squint=qc.get('pct_squint'),
                    residual_jump_pct=100 * jumps.sum() / max(np.isfinite(x).sum(), 1),
                    pct_trials_kept=100 * n_kept / max(n_all, 1), median_area_mm2=float(np.nanmedian(x)))
    except Exception as e:
        print(s, 'ERROR', type(e).__name__, e, flush=True)
        return None


def review(path, cohort, row):
    s = os.path.basename(path)[:-4]
    v = dh.load_dlc(path, views=('side',)).get('side')
    vp = dh.load_dlc(path, views=('side',), physical_units=False)['side']
    frames, _ = dh.load_pupil_qc(s)
    x, _ = dh.clean_series(v, 'pupil_area')
    pct = dh.pupil_relative(x)
    ts = v.timestamps
    raw = v.data['pupil_area'].to_numpy()
    valid = dh.pupil_qc_valid(v, frames)
    # residual jumps among QC-valid frames of the stored trace (what the QC let through), before the median
    rj, ref = residual_jumps(np.where(valid, raw, np.nan), ts)
    dev = np.where(rj, np.abs(raw - ref) / ref, 0)
    picks = []
    for i in np.argsort(dev)[::-1]:
        if dev[i] <= 0:
            break
        if all(abs(ts[i] - ts[j]) > 5 for j in picks):
            picks.append(int(i))
        if len(picks) == 5:
            break
    med = np.nanmedian(raw[valid])
    cand = np.flatnonzero(valid & (np.abs(raw - med) < 0.05 * med))
    typical = int(cand[len(cand) // 2]) if len(cand) else len(raw) // 2

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1.3, 1.3])
    ax = fig.add_subplot(gs[0, :])
    step = max(1, len(ts) // 20000)
    ax.plot(ts[::step], pct[::step], color='k', lw=0.4)
    t_qc, st = frames['t'].to_numpy(), frames['state'].to_numpy()
    for code, col in STATE_COL.items():
        tt_ = t_qc[st == code][::5]
        ax.scatter(tt_, np.full(len(tt_), -45), s=1, color=col, marker='|')
    for i in picks:
        ax.axvline(ts[i], color='m', lw=0.6, ls=':')
    ax.set_ylim(-50, max(60, np.nanpercentile(pct, 99.9) + 5))
    ax.set_ylabel('pupil (% of session median)')
    ax.set_xlabel('session time (s)')
    ax.set_title(f'{s} ({cohort}), QC {row.verdict}: valid {row.pct_valid:.1f}%, QC jumps {row.pct_jump_qc:.2f}%, '
                 f'residual jumps {row.residual_jump_pct:.2f}%, trials kept {row.pct_trials_kept:.0f}%, median '
                 f'{row.median_area_mm2:.2f} mm$^2$. Ticks at -45: blink (blue), squint (orange), failed (red); '
                 f'magenta: frames shown below', fontsize=8)
    with h5py.File(path, 'r') as f:
        ev = m011.trial_events(f, 'stim')
    for c, tt in enumerate(REVIEW_TYPES):
        ax = fig.add_subplot(gs[1, 2 * c:2 * c + 2])
        if len(ev[tt]) == 0:
            ax.axis('off')
            continue
        g, m = dh.peth(pct, ts, ev[tt], WIN, dt=DT)
        m = m - np.nanmean(m[:, g < 0], 1, keepdims=True)
        lim = max(5, np.nanpercentile(np.abs(m), 98))
        ax.imshow(m, aspect='auto', cmap='RdBu_r', vmin=-lim, vmax=lim, extent=[g[0], g[-1], len(m), 0],
                  interpolation='nearest')
        ax.axvline(0, color='k', lw=0.5)
        ax2 = ax.twinx()
        ax2.plot(g, np.nanmean(m, 0), color='k', lw=1.2)
        ax2.set_ylabel('mean (%)', fontsize=7)
        ax.set_title(f'{tt}: trials (% change from baseline, colour +-{lim:.0f}%; white = NaN), n={len(m)}',
                     fontsize=8)
        ax.set_ylabel('trial')
        ax.set_xlabel('time from stimulus (s)')
    vid = m013.video_path(s)
    cap = cv2.VideoCapture(vid) if vid else None
    cx = np.nanmedian(np.concatenate([vp.data[f'{p}_x'].to_numpy() for p in m013.POINTS]))
    cy = np.nanmedian(np.concatenate([vp.data[f'{p}_y'].to_numpy() for p in m013.POINTS]))
    for c, (label, i) in enumerate([('typical', typical)] + [(f'jump {k + 1}', i) for k, i in enumerate(picks)]):
        ax = fig.add_subplot(gs[2, c])
        im = m013.frame(cap, i) if cap is not None else None
        if im is not None:
            ax.imshow(im, cmap='gray', vmin=0, vmax=255)
        xs = [vp.data[f'{p}_x'].iloc[i] for p in m013.POINTS]
        ys = [vp.data[f'{p}_y'].iloc[i] for p in m013.POINTS]
        ax.fill(xs, ys, fill=False, ec='w', lw=0.6, ls='--')
        for (p, col), x_, y_ in zip(m013.POINTS.items(), xs, ys):
            ax.plot(x_, y_, 'o', mfc='none', mec=col, ms=6, mew=1.2)
        ax.set_xlim(cx - m013.CROP, cx + m013.CROP)
        ax.set_ylim(cy + m013.CROP, cy - m013.CROP)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{label}: t={ts[i]:.0f} s\narea {raw[i] * 1:.2f} mm$^2$ (1 s median {ref[i]:.2f})',
                     fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f'015_pupil_review_{s[:5]}.png', dpi=105)
    plt.close(fig)


def main():
    sess = m011.sessions_and_cohorts()
    with ProcessPoolExecutor(12) as ex:
        rows = [r for r in ex.map(metrics, list(sess.items())) if r]
    t = pd.DataFrame(rows).sort_values('residual_jump_pct', ascending=False)
    t.to_csv(OUT / 'tables' / '015_pupil_tracking_metrics.csv', index=False)
    pd.set_option('display.width', 200)
    print(t.round(3).head(25).to_string(index=False))

    mice = sys.argv[1:] or CANDIDATES
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (xc, yc) in zip(axs, [('pct_valid', 'residual_jump_pct'), ('pct_jump_qc', 'residual_jump_pct'),
                                  ('pct_valid', 'pct_trials_kept')]):
        for _, r in t.iterrows():
            col = {'pass': 'tab:green', 'warn': 'tab:orange', 'fail': 'tab:red'}.get(r.verdict, 'grey')
            ax.scatter(r[xc], r[yc], s=14, color=col, edgecolor='k' if r.mouse in mice else 'none')
            if r.mouse in mice or r.verdict != 'pass':
                ax.text(r[xc], r[yc], r.mouse, fontsize=6)
        ax.set_xlabel(xc)
        ax.set_ylabel(yc)
        ax.set_box_aspect(1)
    fig.suptitle('Day-0 pupil tracking quality per session (colour = QC verdict; black edge = reviewed candidates)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '015_pupil_tracking_metrics.png', dpi=120)
    plt.close(fig)

    by_mouse = {os.path.basename(p)[:5]: (p, c) for p, c in sess.items()}
    for m in mice:
        p, c = by_mouse[m]
        r = t.set_index('session_id').loc[os.path.basename(p)[:-4]]
        review(p, c, r)
        print('reviewed', m, flush=True)


if __name__ == '__main__':
    main()
