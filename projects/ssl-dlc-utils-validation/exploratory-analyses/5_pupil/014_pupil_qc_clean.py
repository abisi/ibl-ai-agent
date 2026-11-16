"""014 -- Day-0 pupil area with the pupil QC (projects/ssl-pupil-qc) in dlc_helpers.clean_pupil, against the stored
trace, and outlier mice.

Variants: (A) stored pupil_area + running median 0.1 s (no QC, all sessions); (B) clean_pupil: QC blink / squint /
failed frames NaN, QC interpolation of failed gaps <= 0.1 s, running median 0.1 s (mm^2), sessions with QC verdict
'fail' excluded; (C) B as % change from the session median pupil area (dh.pupil_relative). Trials kept if >= 80% of
the window is finite (QC trial_min_valid); baseline = trial mean in -1 .. 0 s subtracted. Window -1 .. 4 s from
stimulus onset, 20 ms bins. Day-0 sessions, cohorts, trial types and exclusions as in 011.
Outliers: per mouse and variant, response = mean of the baseline-subtracted trace in 0.5 .. 3 s, for whisker hit and
passive W pre; flagged when |x - cohort median| > 4 x 1.4826 MAD.
Outputs: cache/014_<session>.npz; figures/014_pupil_variants_cohorts.png, figures/014_pupil_mice_{Rplus,Rminus}.png;
tables/014_pupil_sessions.csv (QC verdict, % valid frames, trials kept), tables/014_pupil_outliers.csv
Run (haas): python 014_pupil_qc_clean.py [--plot-only]
"""
import importlib
import os
import sys
from concurrent.futures import ProcessPoolExecutor

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
from common import OUT, dh  # noqa: E402

m011 = importlib.import_module('011_day0_posture_movement')
TRIAL_TYPES = m011.TRIAL_TYPES
COHORT_COLOR = m011.COHORT_COLOR
VARIANTS = {'A': 'stored + median 0.1 s, no QC (mm$^2$)', 'B': 'pupil QC + median 0.1 s (mm$^2$)',
            'C': 'pupil QC + median 0.1 s (% of session median)'}
WIN, DT = (-1.0, 4.0), 0.02
MIN_FINITE = 0.8
CACHE = OUT / 'cache'


def one(path):
    s = os.path.basename(path)[:-4]
    out = CACHE / f'014_{s}.npz'
    if out.exists():
        return s
    try:
        v = dh.load_dlc(path, views=('side',), mask_likelihood=False).get('side')
        if v is None or 'pupil_area' not in v.data:
            return None
        frames, metrics = dh.load_pupil_qc(s)
        with h5py.File(path, 'r') as f:
            ev = m011.trial_events(f, 'stim')
        traces = {'A': dh.clean_pupil(v, None)[0]}
        info = dict(verdict=np.array([metrics.get('verdict', 'no QC') if metrics is not None else 'no QC']),
                    pct_valid=np.array([metrics.get('pct_valid', np.nan) if metrics is not None else np.nan]))
        if frames is not None:
            b, _ = dh.clean_pupil(v, frames)
            traces['B'] = b
            traces['C'] = dh.pupil_relative(b, 'percent')
        store = dict(info)
        for var, x in traces.items():
            for tt, times in ev.items():
                if len(times) == 0:
                    continue
                g, m = dh.peth(x, v.timestamps, times, WIN, dt=DT)
                ok = np.isfinite(m).mean(1) >= MIN_FINITE
                store[f'{var}|{tt}|n_all'] = np.array([len(m)])
                if not ok.any():
                    continue
                m = m[ok] - np.nanmean(m[ok][:, g < 0], 1, keepdims=True)
                store[f'{var}|{tt}|mean'] = np.nanmean(m, 0)
                store[f'{var}|{tt}|n'] = np.array([ok.sum()])
                store['grid'] = g
        np.savez_compressed(out, **store)
        return s
    except Exception as e:
        print(s, 'ERROR', type(e).__name__, e, flush=True)
        return None


def collect(sess):
    acc, rows, grid = {}, [], None
    for p, c in sess.items():
        s = os.path.basename(p)[:-4]
        f = CACHE / f'014_{s}.npz'
        if not f.exists():
            continue
        d = np.load(f, allow_pickle=True)
        verdict = str(d['verdict'][0])
        row = dict(session_id=s, cohort=c, verdict=verdict, pct_valid=float(d['pct_valid'][0]))
        for tt in TRIAL_TYPES:
            row[f'trials_{tt}'] = int(d[f'B|{tt}|n'][0]) if f'B|{tt}|n' in d else 0
            row[f'trials_all_{tt}'] = int(d[f'A|{tt}|n_all'][0]) if f'A|{tt}|n_all' in d else 0
        rows.append(row)
        if 'grid' in d:
            grid = d['grid']
        for var in VARIANTS:
            if var != 'A' and verdict in ('fail', 'no QC'):
                continue
            for tt in TRIAL_TYPES:
                k = f'{var}|{tt}|mean'
                if k in d:
                    acc.setdefault((var, c, tt), {})[s[:5]] = (d[k], int(d[f'{var}|{tt}|n'][0]))
    res = {}
    for key, per_mouse in acc.items():
        mice = sorted(per_mouse)
        res[key] = (grid, np.array([per_mouse[m][0] for m in mice]), mice, sum(per_mouse[m][1] for m in mice))
    return res, pd.DataFrame(rows)


def outliers(res, grid):
    rows = []
    post = (grid >= 0.5) & (grid <= 3.0)
    for (var, c, tt), (_, mat, mice, _) in res.items():
        if tt not in ('whisker hit', 'passive W pre'):
            continue
        r = np.nanmean(mat[:, post], 1)
        med = np.nanmedian(r)
        mad = 1.4826 * np.nanmedian(np.abs(r - med))
        for m_, x in zip(mice, r):
            z = (x - med) / mad if mad > 0 else np.nan
            rows.append(dict(variant=var, cohort=c, trial_type=tt, mouse=m_, response=x, robust_z=z,
                             outlier=bool(abs(z) > 4)))
    return pd.DataFrame(rows)


def plot(res, out_tab):
    fig, axs = plt.subplots(len(VARIANTS), len(TRIAL_TYPES), figsize=(2.0 * len(TRIAL_TYPES), 2.0 * len(VARIANTS) + 0.6),
                            sharex=True, sharey='row', squeeze=False)
    for r, (var, label) in enumerate(VARIANTS.items()):
        axs[r, 0].set_ylabel(label, fontsize=6.5)
        lo, hi = [], []
        for k, tt in enumerate(TRIAL_TYPES):
            ax = axs[r, k]
            txt = []
            for c in ('R+', 'R-'):
                if (var, c, tt) not in res:
                    txt.append('0')
                    continue
                g, mat, mice, n_tr = res[(var, c, tt)]
                m, se = m011.mean_sem(mat)
                ax.fill_between(g, m - se, m + se, color=COHORT_COLOR[c], alpha=0.25, lw=0)
                ax.plot(g, m, color=COHORT_COLOR[c], lw=1.2)
                lo.append(np.nanmin(m - se))
                hi.append(np.nanmax(m + se))
                txt.append(f'{len(mice)} ({n_tr})')
            ax.text(0.98, 0.97, ' / '.join(txt), transform=ax.transAxes, ha='right', va='top', fontsize=5.5)
        if lo:
            pad = 0.08 * (max(hi) - min(lo))
            axs[r, 0].set_ylim(min(lo) - pad, max(hi) + pad)
    for k, tt in enumerate(TRIAL_TYPES):
        axs[0, k].set_title(tt, fontsize=8)
        axs[-1, k].set_xlabel('time from stimulus (s)', fontsize=7)
    for ax in axs.flat:
        ax.axvline(0, color='k', lw=0.5, ls=':')
        ax.axhline(0, color='0.7', lw=0.4)
        ax.tick_params(labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_box_aspect(1)
    fig.suptitle('Day 0 pupil area, baseline-subtracted, mean +- SEM across mice, R+ (green) / R- (magenta); '
                 'n = mice (trials), R+ / R-', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / 'figures' / '014_pupil_variants_cohorts.png', dpi=120)
    plt.close(fig)

    flagged = set(map(tuple, out_tab[out_tab.outlier][['variant', 'cohort', 'mouse']].values))
    for c, tag in (('R+', 'Rplus'), ('R-', 'Rminus')):
        fig, axs = plt.subplots(2, len(TRIAL_TYPES), figsize=(2.0 * len(TRIAL_TYPES), 4.6), sharex=True, sharey='row',
                                squeeze=False)
        for r, var in enumerate(('B', 'C')):
            axs[r, 0].set_ylabel(VARIANTS[var], fontsize=6.5)
            vals = []
            for k, tt in enumerate(TRIAL_TYPES):
                ax = axs[r, k]
                if (var, c, tt) not in res:
                    continue
                g, mat, mice, n_tr = res[(var, c, tt)]
                for m_, row in zip(mice, mat):
                    out = (var, c, m_) in flagged
                    ax.plot(g, row, color='tab:red' if out else COHORT_COLOR[c], lw=0.9 if out else 0.4,
                            alpha=0.9 if out else 0.4)
                    if out:
                        ax.text(g[-1], row[-1], m_, fontsize=5, color='tab:red', ha='right')
                m, se = m011.mean_sem(mat)
                ax.plot(g, m, color='k', lw=1.3)
                ax.text(0.98, 0.97, f'n={len(mice)} ({n_tr})', transform=ax.transAxes, ha='right', va='top',
                        fontsize=5.5)
                vals.append(mat)
            if vals:
                lim = m011.row_limits(vals)
                axs[r, 0].set_ylim(*lim)
        for k, tt in enumerate(TRIAL_TYPES):
            axs[0, k].set_title(tt, fontsize=8)
            axs[-1, k].set_xlabel('time from stimulus (s)', fontsize=7)
        for ax in axs.flat:
            ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.tick_params(labelsize=6)
            ax.spines[['top', 'right']].set_visible(False)
            ax.set_box_aspect(1)
        fig.suptitle(f'Day 0, {c}: pupil after QC, individual mice (red: outlier, |robust z| > 4 on whisker hit or '
                     f'passive W pre, 0.5-3 s), black = mean +- SEM', fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(OUT / 'figures' / f'014_pupil_mice_{tag}.png', dpi=120)
        plt.close(fig)


def main():
    CACHE.mkdir(exist_ok=True)
    sess = m011.sessions_and_cohorts()
    if '--plot-only' not in sys.argv:
        with ProcessPoolExecutor(12) as ex:
            print('sessions cached', len([s for s in ex.map(one, list(sess)) if s]), flush=True)
    res, tab = collect(sess)
    tab.to_csv(OUT / 'tables' / '014_pupil_sessions.csv', index=False)
    grid = next(iter(res.values()))[0]
    out_tab = outliers(res, grid)
    out_tab.to_csv(OUT / 'tables' / '014_pupil_outliers.csv', index=False)
    plot(res, out_tab)
    print(tab.groupby(['cohort', 'verdict']).size().unstack(fill_value=0).to_string())
    print(tab[tab.verdict.isin(['fail', 'warn', 'no QC'])][['session_id', 'cohort', 'verdict', 'pct_valid']]
          .round(1).to_string(index=False))
    print('outliers:')
    print(out_tab[out_tab.outlier].round(2).to_string(index=False))
    sem = []
    for (var, c, tt), (_, mat, mice, _) in res.items():
        sem.append(dict(variant=var, cohort=c, trial_type=tt, median_sem=np.nanmedian(m011.mean_sem(mat)[1]),
                        peak_abs_mean=np.nanmax(np.abs(m011.mean_sem(mat)[0]))))
    sem = pd.DataFrame(sem)
    sem['sem_over_peak'] = sem.median_sem / sem.peak_abs_mean
    print(sem.groupby('variant')[['median_sem', 'sem_over_peak']].median().round(4).to_string())


if __name__ == '__main__':
    main()
