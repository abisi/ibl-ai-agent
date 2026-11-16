"""011 -- Day-0 posture and movement of each body part around trial events, per trial type, R+ and R- separately,
through dlc_helpers with its defaults (no likelihood masking; pupil with the pupil QC; see the project question.md).

Sessions: NWB_ks4 whisker day 0 (session_description 'whisker_0'); cohort per mouse from the reference sheet
(joint_mouse_reference_weight.xlsx, R+ / R- only, exclude == 1 dropped; AB091 / AB096 not SSL).
Known bad DLC (nwb_session_exceptions.md, DLC reruns pending): AB155_20250208 dropped (video likely not this session),
top camera dropped for MH025_20250421 and MH027_20250422 (top DLC on the upside-down video failed).
Trial types: passive whisker / auditory, pre and post (passive trials before / after the main active block, from
trials.context), active whisker hit / miss, auditory hit, false alarm, correct rejection (perf 2 / 0 / 3 / 5 / 4; perf 6
dropped; auditory misses left out: too few trials, user 2026-11-04). Sessions without context (all 'nan') have no
passive trials.
Two windows: -1 .. 3 s from stimulus onset (10 ms bins) and -50 .. 100 ms from trial start_time (5 ms bins).
Posture: jaw_distance, nose_distance (side), top_nose_distance, whisker_angle, minus each trial's mean before 0;
pupil_area through dh.clean_series (pupil QC: blink / squint / failed frames NaN, running median 0.1 s; MH029 0502
3-point area) as % of the session median (dh.pupil_relative), sessions without QC excluded
(dh.pupil_session_ok), trials with a blink / squint in the window dropped (dh.pupil_trial_ok), trials kept if >= 80% of the window is finite (QC trial rule), minus each trial's mean before 0; tongue out = P(tongue_likelihood > 0.6)
(dh.tongue_presence), not baseline-subtracted. Movement: speed = |d/dt| of the posture series except pupil
(dh.derivative from the frame timestamps), not baseline-subtracted.
Averaging: trials -> session -> mouse (one day-0 session per mouse) -> cohort mean +- SEM across mice; thin lines =
individual mice. n per panel = mice (trials). Y range per row: 5th-95th percentile across mice at each bin (and the
mean +- SEM), so single outlier mice can leave the axes.
Outputs: cache/011v3_<session>.npz; figures/011_{posture,movement}_{Rplus,Rminus}_{stim,start}.png,
figures/011_cohorts_{posture,movement}_{stim,start}.png; tables/011_n.csv (mice, sessions, trials per window x cohort x
trial type x signal)
Run (haas): python 011_day0_posture_movement.py [--plot-only]
"""
import glob
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
from common import NWB_DIR, OUT, dh  # noqa: E402

SHEET = '/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx'
NOT_SSL = ('AB091', 'AB096')
DROP_SESSIONS = {'AB155_20250208_175103'}
DROP_VIEWS = {'MH025_20250421_115741': ('top',), 'MH027_20250422_111013': ('top',)}
COHORT_COLOR = {'R+': '#00B400', 'R-': '#C800C8'}

# (view, series, label, unit)
POSTURE = [('side', 'jaw_distance', 'jaw opening', 'mm'),
           ('side', 'tongue_presence', 'tongue out (P, likelihood > 0.6)', ''),
           ('side', 'nose_distance', 'nose, side view', 'mm'),
           ('top', 'top_nose_distance', 'nose, top view', 'mm'),
           ('top', 'whisker_angle', 'whisker angle', 'deg'),
           ('side', 'pupil_area', 'pupil area', '% of session median')]
MOVEMENT = [('side', 'jaw_distance', 'jaw speed', 'mm/s'),
            ('side', 'nose_distance', 'nose speed, side view', 'mm/s'),
            ('top', 'top_nose_distance', 'nose speed, top view', 'mm/s'),
            ('top', 'whisker_angle', 'whisker speed', 'deg/s')]
TRIAL_TYPES = ['passive W pre', 'passive W post', 'passive A pre', 'passive A post',
               'whisker hit', 'whisker miss', 'auditory hit', 'false alarm', 'correct rejection']
PERF = {'whisker hit': 2, 'whisker miss': 0, 'auditory hit': 3, 'false alarm': 5, 'correct rejection': 4}
# window name -> (window, bin, alignment)
WINDOWS = {'stim': ((-1.0, 3.0), 0.01, 'stimulus onset'), 'start': ((-0.05, 0.1), 0.005, 'trial start_time')}
CACHE = OUT / 'cache'


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def trial_events(f, align='stim'):
    """{trial type: stimulus onset times} (align='stim') or trial start times (align='start')."""
    t = f['intervals/trials']
    n = len(t['start_time'])
    ctx = dec(t['context'][:]) if 'context' in t else np.array(['nan'] * n)
    perf = t['perf'][:]
    passive = ctx == 'passive'
    act = np.flatnonzero(~passive)
    # main active block: the largest run of consecutive non-passive trials
    pre = post = np.zeros(n, bool)
    if passive.any() and len(act):
        runs = np.split(act, np.flatnonzero(np.diff(act) > 1) + 1)
        main = max(runs, key=len)
        idx = np.arange(n)
        pre, post = passive & (idx < main[0]), passive & (idx > main[-1])
    w, a = t['whisker_stim'][:] == 1, t['auditory_stim'][:] == 1
    if align == 'stim':
        nt = t['no_stim_time'][:] if 'no_stim_time' in t else t['start_time'][:]
        ev_t = np.where(w, t['whisker_stim_time'][:], np.where(a, t['auditory_stim_time'][:], nt))
    else:
        ev_t = t['start_time'][:]
    sel = {'passive W pre': pre & w, 'passive W post': post & w, 'passive A pre': pre & a, 'passive A post': post & a}
    for k, p in PERF.items():
        sel[k] = ~passive & (perf == p)
    return {k: ev_t[m][np.isfinite(ev_t[m])] for k, m in sel.items()}


def one(path):
    s = os.path.basename(path)[:-4]
    out = CACHE / f'011v3_{s}.npz'
    if out.exists():
        return s
    try:
        views = dh.load_dlc(path, mask_likelihood=False)
        for v in DROP_VIEWS.get(s, ()):
            views.pop(v, None)
        with h5py.File(path, 'r') as f:
            evs = {w: trial_events(f, w) for w in WINDOWS}
            t = f['intervals/trials']
            st, stim = t['start_time'][:], t['stim_onset'][:] if 'stim_onset' in t else t['start_time'][:]
        store = {'stim_minus_start': np.array([np.nanmedian(stim - st)])}
        for kind, spec in (('posture', POSTURE), ('movement', MOVEMENT)):
            for view, name, _, _ in spec:
                v = views.get(view)
                if v is None:
                    continue
                if name == 'tongue_presence':
                    if 'tongue_likelihood' not in v.likelihood:
                        continue
                    x = dh.tongue_presence(v)
                else:
                    if name not in v.data:
                        continue
                    if name == 'pupil_area' and not dh.pupil_session_ok(s)[0]:
                        continue
                    x, _ = dh.clean_series(v, name)
                    if name == 'pupil_area':
                        x = dh.pupil_relative(x)
                    if kind == 'movement':
                        x = dh.derivative(x, v.timestamps, kind='speed')
                for w, (win, dt, _) in WINDOWS.items():
                    for tt, times in evs[w].items():
                        if len(times) == 0:
                            continue
                        g, m = dh.peth(x, v.timestamps, times, win, dt=dt)
                        if kind == 'posture' and name != 'tongue_presence':
                            m = m - np.nanmean(m[:, g < 0], 1, keepdims=True)
                        # trials with DLC over most of the window (pupil: QC trial rule, >= 80%)
                        ok = np.isfinite(m).mean(1) >= (0.8 if name == 'pupil_area' else 0.5)
                        if name == 'pupil_area':  # trial-wise pupil QC (2026-11-12): no blink / squint in the window
                            ok &= dh.pupil_trial_ok(s, times, window=win)
                        if not ok.any():
                            continue
                        store[f'{w}|{kind}|{name}|{tt}|mean'] = np.nanmean(m[ok], 0)
                        store[f'{w}|{kind}|{name}|{tt}|n'] = np.array([ok.sum()])
                        store[f'{w}|grid'] = g
        np.savez_compressed(out, **store)
        return s
    except Exception as e:
        print(s, 'ERROR', type(e).__name__, e, flush=True)
        return None


def sessions_and_cohorts():
    sheet = pd.read_excel(SHEET)
    sheet = sheet[sheet.reward_group.isin(['R+', 'R-']) & (sheet.exclude != 1)]
    cohort = dict(zip(sheet.mouse_id, sheet.reward_group))
    out = {}
    for p in sorted(glob.glob(str(NWB_DIR / '*.nwb'))):
        s = os.path.basename(p)[:-4]
        if '_backup' in p or s[:5] in NOT_SSL or s in DROP_SESSIONS or s[:5] not in cohort:
            continue
        with h5py.File(p, 'r') as f:
            if f['session_description'][()].decode() != 'whisker_0':
                continue
        out[p] = cohort[s[:5]]
    return out


def collect(sess):
    """{(window, cohort, kind, name, trial type): (grid, mice x time, mouse ids, n sessions, n trials)}."""
    acc, grids, offsets = {}, {}, []
    for p, c in sess.items():
        s = os.path.basename(p)[:-4]
        f = CACHE / f'011v3_{s}.npz'
        if not f.exists():
            continue
        d = np.load(f)
        offsets.append(float(d['stim_minus_start'][0]))
        for w in WINDOWS:
            if f'{w}|grid' in d:
                grids[w] = d[f'{w}|grid']
        for k in d.files:
            if not k.endswith('|mean'):
                continue
            w, kind, name, tt, _ = k.split('|')
            acc.setdefault((w, c, kind, name, tt), {}).setdefault(s[:5], []).append(
                (d[k], int(d[k[:-5] + '|n'][0])))
    res, rows = {}, []
    for key, per_mouse in acc.items():
        mice = sorted(per_mouse)
        mat = np.array([np.nanmean([m for m, _ in per_mouse[mm]], 0) for mm in mice])
        n_sess = sum(len(per_mouse[mm]) for mm in mice)
        n_tr = sum(n for mm in mice for _, n in per_mouse[mm])
        res[key] = (grids[key[0]], mat, mice, n_sess, n_tr)
        rows.append(dict(window=key[0], cohort=key[1], kind=key[2], signal=key[3], trial_type=key[4],
                         n_mice=len(mice), n_sessions=n_sess, n_trials=n_tr))
    print('stim onset - start_time (s), median over sessions:', np.round(np.nanpercentile(offsets, [0, 50, 100]), 4))
    return res, pd.DataFrame(rows)


def mean_sem(mat):
    m = np.nanmean(mat, 0)
    if len(mat) < 2:
        return m, m * 0
    n = np.sum(np.isfinite(mat), 0)
    return m, np.nanstd(mat, 0, ddof=1) / np.sqrt(np.maximum(n, 1))


def row_limits(mats, envelope=True):
    """y limits of a row: 5th-95th percentile across mice per bin (envelope) and mean +- SEM, 8% padding."""
    lo, hi = [], []
    for mat in mats:
        m, se = mean_sem(mat)
        lo.append(np.nanmin(m - se))
        hi.append(np.nanmax(m + se))
        if envelope and len(mat) > 2:
            lo.append(np.nanmin(np.nanpercentile(mat, 5, axis=0)))
            hi.append(np.nanmax(np.nanpercentile(mat, 95, axis=0)))
    if not lo:
        return None
    lo, hi = np.nanmin(lo), np.nanmax(hi)
    pad = 0.08 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def panel_grid(spec, title, align):
    size = 2.0  # square panels
    fig, axs = plt.subplots(len(spec), len(TRIAL_TYPES), figsize=(size * len(TRIAL_TYPES), size * len(spec) + 0.6),
                            sharex=True, sharey='row', squeeze=False)
    for r, (_, name, label, unit) in enumerate(spec):
        axs[r, 0].set_ylabel(f'{label}\n({unit})' if unit else label, fontsize=7)
    for c, tt in enumerate(TRIAL_TYPES):
        axs[0, c].set_title(tt, fontsize=8)
        axs[-1, c].set_xlabel(f'time from {align} (s)', fontsize=7)
    for ax in axs.flat:
        ax.axvline(0, color='k', lw=0.5, ls=':')
        ax.tick_params(labelsize=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_box_aspect(1)
    fig.suptitle(title, fontsize=9)
    return fig, axs


def plot(res):
    for w, (win, _, align) in WINDOWS.items():
        for kind, spec in (('posture', POSTURE), ('movement', MOVEMENT)):
            for c, tag in (('R+', 'Rplus'), ('R-', 'Rminus')):
                fig, axs = panel_grid(spec, f'Day 0, {c}: {kind} around {align}. Thin: individual mice (session mean); '
                                            f'thick: mean +- SEM across mice. n = mice (trials)', align)
                for r, (_, name, _, _) in enumerate(spec):
                    mats = []
                    for k, tt in enumerate(TRIAL_TYPES):
                        ax = axs[r, k]
                        key = (w, c, kind, name, tt)
                        if key not in res:
                            ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center', fontsize=6, color='grey')
                            continue
                        g, mat, mice, _, n_tr = res[key]
                        mats.append(mat)
                        for row in mat:
                            ax.plot(g, row, color=COHORT_COLOR[c], lw=0.4, alpha=0.35)
                        m, se = mean_sem(mat)
                        ax.fill_between(g, m - se, m + se, color='k', alpha=0.2, lw=0)
                        ax.plot(g, m, color='k', lw=1.3)
                        ax.text(0.98, 0.97, f'n={len(mice)} ({n_tr})', transform=ax.transAxes, ha='right',
                                va='top', fontsize=5.5)
                    lim = row_limits(mats)
                    if lim:
                        axs[r, 0].set_ylim(*lim)
                axs[0, 0].set_xlim(*win)
                fig.tight_layout(rect=(0, 0, 1, 0.97))
                fig.savefig(OUT / 'figures' / f'011_{kind}_{tag}_{w}.png', dpi=120)
                plt.close(fig)
            fig, axs = panel_grid(spec, f'Day 0: {kind} around {align}, mean +- SEM across mice, R+ (green) vs R- '
                                        f'(magenta). n = mice (trials), R+ / R-', align)
            for r, (_, name, _, _) in enumerate(spec):
                mats = []
                for k, tt in enumerate(TRIAL_TYPES):
                    ax = axs[r, k]
                    txt = []
                    for c in ('R+', 'R-'):
                        key = (w, c, kind, name, tt)
                        if key not in res:
                            txt.append('0')
                            continue
                        g, mat, mice, _, n_tr = res[key]
                        mats.append(mat)
                        m, se = mean_sem(mat)
                        ax.fill_between(g, m - se, m + se, color=COHORT_COLOR[c], alpha=0.25, lw=0)
                        ax.plot(g, m, color=COHORT_COLOR[c], lw=1.2)
                        txt.append(f'{len(mice)} ({n_tr})')
                    ax.text(0.98, 0.97, ' / '.join(txt), transform=ax.transAxes, ha='right', va='top', fontsize=5.5)
                lim = row_limits(mats, envelope=False)
                if lim:
                    axs[r, 0].set_ylim(*lim)
            axs[0, 0].set_xlim(*win)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(OUT / 'figures' / f'011_cohorts_{kind}_{w}.png', dpi=120)
            plt.close(fig)


def main():
    CACHE.mkdir(exist_ok=True)
    sess = sessions_and_cohorts()
    print('day-0 sessions', len(sess), {c: sum(v == c for v in sess.values()) for c in ('R+', 'R-')}, flush=True)
    if '--plot-only' not in sys.argv:
        with ProcessPoolExecutor(12) as ex:
            done = [s for s in ex.map(one, list(sess)) if s]
        print('sessions cached', len(done), flush=True)
    res, n = collect(sess)
    n = n.sort_values(['window', 'kind', 'signal', 'cohort', 'trial_type'])
    n.to_csv(OUT / 'tables' / '011_n.csv', index=False)
    plot(res)
    sel = n[(n.window == 'stim') & (n.kind == 'posture') & (n.signal == 'jaw_distance')]
    print(sel.pivot(index='trial_type', columns='cohort', values=['n_mice', 'n_trials']).to_string())


if __name__ == '__main__':
    main()
