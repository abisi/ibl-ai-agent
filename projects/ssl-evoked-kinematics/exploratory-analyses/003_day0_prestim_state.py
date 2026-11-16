"""003 -- Day 0: does whisker posture or movement just before the stimulus predict the upcoming choice (lick / no lick),
and does that survive the obvious confounds (session progression, engagement, recent history, arousal, licking)?

Inputs: NWB_ks4 day-0 sessions (DLC via dlc_helpers; pupil with dlc_helpers pupil QC); whisker rules and session list of 000.
The cache and trial table written here are the inputs of 004 and 005; the 003 figures are superseded by 004 / 005.

Trials: every active trial of the day-0 session (passive and perf 6 dropped; early_lick == 1 dropped): whisker (hit /
miss), auditory (hit / miss), no stimulus (false alarm / correct rejection). Choice = lick (hit or FA) vs no lick.
Aligned to the stimulus (no-stim trials: no_stim_time).
Single-trial signals, 5-ms bins, -1.0 .. +0.1 s (dlc_helpers + the 000 whisker skeleton rules):
  wh_speed  whisker angular speed (deg/s; base -> tip angle, skeleton validity rules of 000)
  wh_angle  whisker angle minus the session median (deg; negative = more protracted posture)
  nose_speed side-view nose tip speed (mm/s), jaw (jaw opening, mm, minus session median),
  tongue (P tongue out, likelihood > 0.6), pupil (% of session median; pupil QC frames masked; trials with a blink / squint in the window NaN)
Windows: far -1.0 .. -0.1 s, pre -100 .. -30 ms, peri -30 .. 0 ms. The requested -30 .. +5 ms window is also
reported (peri5) but on whisker trials its last frame can already contain the stimulus deflection, so peri (-30 .. 0)
is the main one.
Analyses (per mouse = its day-0 session; both non-parametric and parametric tests; uncorrected):
  A time courses lick vs no lick per trial class and cohort, per-bin paired Wilcoxon across mice (bins p < 0.01 marked)
  B window effects: lick minus no-lick per mouse
  C P(lick) per within-session quintile of the window signal (peri whisker speed)
  D confounds: (1) signal and P(lick) across session fifths; (2) stratified lick - no-lick difference within session
    fifths (removes the progression); (3) per-session L2 logistic regression of choice on z-scored predictors (whisker
    speed peri / pre / far, whisker angle peri, nose speed peri, jaw peri, tongue out in the window, session fraction,
    previous-trial lick, previous-trial reward, recent lick rate over the last 10 trials of the same class, pupil),
    coefficients tested across mice; also whisker speed alone
Outputs (combined_results_ks4/ssl-evoked-kinematics/): cache/003v1_<session>.npz, figures/003_day0_*.{png,pdf},
tables/003_day0_*.csv
Run (haas; imports 000 for the shared rules, does not need its cache): python 003_day0_prestim_state.py [--plot-only]
"""
import importlib.util
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('k000', HERE / '000_evoked_kinematics.py')
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
spec = importlib.util.spec_from_file_location('f001', HERE / '001_trajectory_figures.py')
f1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1)
dh = k.dh

STAGE = 'day0'
WIN, DT = (-1.0, 0.1), 0.005
# ts = 1 s before trial start (stimulus = trial start in this task; inside the 2-5 s no-lick quiet window)
WINDOWS = {'far': (-1.0, -0.1), 'pre': (-0.1, -0.03), 'peri': (-0.03, 0.0), 'peri5': (-0.03, 0.005),
           'ts': (-1.0, 0.0)}
SIGNALS = ['wh_speed', 'wh_angle', 'nose_speed', 'jaw', 'tongue', 'pupil']
# one-line labels with units, same wording in every figure (Δ = lick minus no lick)
SIG_LABEL = {'wh_speed': 'whisker speed (deg/s)', 'wh_angle': 'whisker angle (deg)',
             'nose_speed': 'nose speed (mm/s)', 'jaw': 'jaw opening (mm)', 'tongue': 'tongue out (P)',
             'pupil': 'pupil area (%)'}
CLASSES = {'whisker': ('whisker hit', 'whisker miss'), 'auditory': ('auditory hit', 'auditory miss'),
           'no stim': ('false alarm', 'correct rejection')}
PERF = {0: ('whisker', 0), 2: ('whisker', 1), 1: ('auditory', 0), 3: ('auditory', 1), 4: ('no stim', 0),
        5: ('no stim', 1)}
CACHE = k.CACHE
TAB, FIG = k.OUT / 'tables', k.OUT / 'figures'
LICK_COL, NOLICK_COL = '#222222', '#9a9a9a'


def whisker_angle_series(top):
    hf = k.head_frame(top, 'top_nose_tip', 'top_nose_base', ['whisker_base'])
    if hf is None:
        return None
    pts = {kp: k.to_head(k.xy(top, kp), hf) for kp in k.CUT}
    ok = np.ones(len(top.timestamps), bool)
    for kp, c in k.CUT.items():
        ok &= k.lik(top, kp) >= c
    for a, b in (('top_particle', 'whisker_base'), ('whisker_tip', 'top_particle')):
        ln = np.linalg.norm(pts[a] - pts[b], axis=1)
        med = np.nanmedian(ln[ok]) if ok.any() else np.nan
        ok &= (ln > 0.6 * med) & (ln < 1.6 * med)
    d = pts['whisker_tip'] - pts['whisker_base']
    d[~ok] = np.nan
    return k.run_median(k.angle(d))


def one(row):
    s, path = row['session'], row['path']
    out = CACHE / f'003v1_{s}.npz'
    if out.exists():
        return s
    try:
        views = dh.load_dlc(path, mask_likelihood=False)
        for v in k.DROP_VIEWS.get(s, ()):
            views.pop(v, None)
        with h5py.File(path, 'r') as f:
            t = f['intervals/trials']
            n = len(t['start_time'])
            ctx = k.dec(t['context'][:]) if 'context' in t else np.array(['nan'] * n)
            tr = pd.DataFrame({'perf': t['perf'][:], 'start_time': t['start_time'][:],
                               'early_lick': t['early_lick'][:] if 'early_lick' in t else np.zeros(n),
                               'whisker_stim_time': t['whisker_stim_time'][:],
                               'auditory_stim_time': t['auditory_stim_time'][:],
                               'no_stim_time': t['no_stim_time'][:] if 'no_stim_time' in t else t['start_time'][:],
                               'reward_available': t['reward_available'][:] if 'reward_available' in t else np.nan,
                               'context': ctx})
        tr['trial_index'] = np.arange(n)
        tr = tr[(tr.context != 'passive') & tr.perf.isin(PERF) & (tr.early_lick != 1)].copy()
        tr['cls'] = tr.perf.map(lambda p: PERF[p][0])
        tr['lick'] = tr.perf.map(lambda p: PERF[p][1])
        tr['event'] = np.where(tr.cls == 'whisker', tr.whisker_stim_time,
                               np.where(tr.cls == 'auditory', tr.auditory_stim_time, tr.no_stim_time))
        tr = tr[np.isfinite(tr.event)].reset_index(drop=True)
        tr['rewarded'] = ((tr.lick == 1) & (tr.reward_available == 1)).astype(int)
        tr['session_frac'] = tr.trial_index / max(n - 1, 1)
        tr['prev_lick'] = tr.lick.shift(1).fillna(0).to_numpy()
        tr['prev_reward'] = tr.rewarded.shift(1).fillna(0).to_numpy()
        tr['recent_lick_rate'] = tr.groupby('cls').lick.transform(
            lambda x: x.shift(1).rolling(10, min_periods=3).mean()).fillna(tr.lick.mean())
        sig = {}
        side, top = views.get('side'), views.get('top')
        if top is not None and 'whisker_tip_x' in top.data:
            a = whisker_angle_series(top)
            if a is not None:
                sig['wh_angle'] = (a - np.nanmedian(a), top.timestamps)
                sig['wh_speed'] = (dh.derivative(a, top.timestamps, kind='speed'), top.timestamps)
        if side is not None:
            ts = side.timestamps
            jaw, _ = dh.clean_series(side, 'jaw_distance')
            sig['jaw'] = (jaw - np.nanmedian(jaw), ts)
            if 'tongue_likelihood' in side.likelihood:
                sig['tongue'] = (dh.tongue_presence(side), ts)
            if 'nose_tip_x' in side.data:
                nx, ny = side.data['nose_tip_x'].to_numpy().copy(), side.data['nose_tip_y'].to_numpy().copy()
                bad = side.likelihood['nose_tip_likelihood'].to_numpy() < 0.1
                nx[bad] = ny[bad] = np.nan
                sig['nose_speed'] = (dh.keypoint_speed(k.run_median(nx), k.run_median(ny), ts), ts)
            if 'pupil_area' in side.data and dh.pupil_session_ok(s)[0]:
                p, _ = dh.clean_series(side, 'pupil_area')
                sig['pupil'] = (dh.pupil_relative(p), ts)
        if 'wh_speed' not in sig:
            return None
        store = {c: tr[c].to_numpy() for c in ('perf', 'lick', 'event', 'trial_index', 'session_frac', 'prev_lick',
                                               'prev_reward', 'recent_lick_rate', 'rewarded')}
        store['cls'] = tr.cls.to_numpy().astype('U8')
        for name, (x, ts) in sig.items():
            g, m = dh.peth(x, ts, tr.event.to_numpy(), WIN, dt=DT)
            if name == 'pupil':  # trial-wise pupil QC (2026-11-12): no blink / squint in the window
                m[~dh.pupil_trial_ok(s, tr.event.to_numpy(), window=WIN)] = np.nan
            store[f'sig|{name}'] = m.astype(np.float32)
            store['grid'] = g
        np.savez_compressed(out, **store)
        return s
    except Exception as e:
        print(s, 'ERROR', type(e).__name__, e, flush=True)
        return None


# ---------------------------------------------------------------- analysis
def load_all():
    sess = k.sessions_and_cohorts()
    sess = sess[sess.stage == STAGE]
    data = {}
    for r in sess.itertuples():
        f = CACHE / f'003v1_{r.session}.npz'
        if f.exists():
            d = dict(np.load(f))
            d['cohort'], d['mouse'] = r.cohort, r.mouse
            data[r.session] = d
    return data


def wmean(m, g, w):
    sel = (g >= w[0] - 1e-9) & (g <= w[1] + 1e-9)
    with np.errstate(all='ignore'):
        return np.nanmean(m[:, sel], 1)


def trial_table(data):
    rows = []
    for s, d in data.items():
        g = d['grid']
        base = dict(session=s, mouse=d['mouse'], cohort=d['cohort'])
        n = len(d['lick'])
        tab = pd.DataFrame({'cls': d['cls'], 'lick': d['lick'], 'session_frac': d['session_frac'],
                            'prev_lick': d['prev_lick'], 'prev_reward': d['prev_reward'],
                            'recent_lick_rate': d['recent_lick_rate'], 'trial_index': d['trial_index']})
        for sname in SIGNALS:
            key = f'sig|{sname}'
            for wn, w in WINDOWS.items():
                tab[f'{sname}_{wn}'] = wmean(d[key], g, w) if key in d else np.full(n, np.nan)
        for kk, v in base.items():
            tab[kk] = v
        rows.append(tab)
    return pd.concat(rows, ignore_index=True)


def two_tests(a, b=None):
    if b is None:
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if len(a) < 4:
            return np.nan, np.nan, len(a)
        return stats.wilcoxon(a).pvalue, stats.ttest_1samp(a, 0).pvalue, len(a)
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 4 or len(b) < 4:
        return np.nan, np.nan, (len(a), len(b))
    return stats.mannwhitneyu(a, b).pvalue, stats.ttest_ind(a, b, equal_var=False).pvalue, (len(a), len(b))


def mouse_diff(tt, var, strat=False, min_n=3):
    """per mouse: mean(var | lick) - mean(var | no lick); strat: within session fifths, averaged over fifths."""
    out = []
    for (m, c), g in tt.groupby(['mouse', 'cohort']):
        if strat:
            g = g.assign(fifth=np.minimum((g.session_frac * 5).astype(int), 4))
            ds = []
            for _, h in g.groupby('fifth'):
                a, b = h.loc[h.lick == 1, var].dropna(), h.loc[h.lick == 0, var].dropna()
                if len(a) >= 2 and len(b) >= 2:
                    ds.append(a.mean() - b.mean())
            d = np.mean(ds) if len(ds) >= 2 else np.nan
        else:
            a, b = g.loc[g.lick == 1, var].dropna(), g.loc[g.lick == 0, var].dropna()
            d = a.mean() - b.mean() if len(a) >= min_n and len(b) >= min_n else np.nan
        out.append(dict(mouse=m, cohort=c, diff=d))
    return pd.DataFrame(out)


def squares(fig):
    for ax in fig.axes:
        ax.set_box_aspect(1)


def p_txt(p):
    return 'n.s.' if not np.isfinite(p) else ('p<0.001' if p < 0.001 else f'p={p:.3f}')


def fig_timecourses(data, rows_out):
    sigs = ['wh_speed', 'wh_angle', 'nose_speed', 'jaw', 'tongue', 'pupil']
    cols = [(c, cl) for cl in ('whisker', 'no stim') for c in k.COHORTS]
    fig, axs = plt.subplots(len(sigs), len(cols), figsize=(7.2, 1.75 * len(sigs) + 0.9), sharex=True, squeeze=False)
    g = next(iter(data.values()))['grid']
    for j, (c, cl) in enumerate(cols):
        for i, sname in enumerate(sigs):
            ax = axs[i, j]
            per = {1: [], 0: []}
            for d in data.values():
                key = f'sig|{sname}'
                if d['cohort'] != c or key not in d:
                    continue
                m = d[key]
                sel = d['cls'] == cl
                for lv in (1, 0):
                    s2 = sel & (d['lick'] == lv)
                    if s2.sum() >= 3:
                        with np.errstate(all='ignore'):
                            per[lv].append((d['mouse'], np.nanmean(m[s2], 0)))
            if not per[1] or not per[0]:
                continue
            mice = sorted(set(x for x, _ in per[1]) & set(x for x, _ in per[0]))
            A = np.array([v for x, v in per[1] if x in mice])
            B = np.array([v for x, v in per[0] if x in mice])
            for mat, col, lab in ((A, LICK_COL, 'lick'), (B, NOLICK_COL, 'no lick')):
                mm, se = k.mean_sem(mat)
                ax.fill_between(g, mm - se, mm + se, color=col, alpha=0.25, lw=0)
                ax.plot(g, mm, color=col, lw=0.9, label=f'{lab}')
            # per-bin paired Wilcoxon across mice
            ps = np.full(len(g), np.nan)
            for b in range(0, len(g), 2):
                x = A[:, b] - B[:, b]
                x = x[np.isfinite(x)]
                if len(x) >= 6 and np.any(x != 0):
                    ps[b] = stats.wilcoxon(x).pvalue
            sig = np.where(ps < 0.01)[0]
            if len(sig):
                y = ax.get_ylim()[1]
                ax.plot(g[sig], np.full(len(sig), y), 's', color=k.COHORT_COLOR[c], ms=1.5)
            for wn, (lo, hi), shade in (('pre', WINDOWS['pre'], '#fde8c8'), ('peri', WINDOWS['peri'], '#f6c6a1')):
                ax.axvspan(lo, hi, color=shade, lw=0, zorder=0)
            ax.axvline(0, color='k', lw=0.5, ls=':')
            if i == 0:
                ax.set_title(f'{c} · {cl} trials\n({len(mice)} mice)', color=k.COHORT_COLOR[c], fontsize=7)
            if j == 0:
                ax.set_ylabel(SIG_LABEL[sname], fontsize=6.5)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=5.5, loc='upper left')
            rows_out.append(dict(cohort=c, cls=cl, signal=sname, n_mice=len(mice),
                                 bins_p01=int(np.sum(ps < 0.01)), first_sig_bin_s=float(g[sig[0]]) if len(sig) else np.nan))
    for ax in axs[-1]:
        ax.set_xlabel('time from stimulus (s)')
        ax.set_xlim(-1.0, 0.1)
    fig.suptitle(f1.wrap('Day 0: signals before the stimulus on lick (black) vs no-lick (grey) trials. Whisker trials: '
                         'hit vs miss; no-stim trials: false alarm vs correct rejection. Mean +- SEM across mice; '
                         'squares = bins with paired Wilcoxon p < 0.01 across mice; shading = pre (-100..-30 ms) and '
                         'peri (-30..0 ms) windows', fig), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    squares(fig)
    f1.save(fig, '003_day0_timecourses')


def fig_windows(tt, stats_rows):
    sigs = ['wh_speed', 'wh_angle', 'nose_speed', 'pupil']
    fig, axs = plt.subplots(len(sigs), 3, figsize=(7.2, 2.45 * len(sigs) + 0.6), squeeze=False)
    for i, sname in enumerate(sigs):
        for j, cl in enumerate(('whisker', 'auditory', 'no stim')):
            ax = axs[i, j]
            x0 = 0
            ticks, labs = [], []
            for c in k.COHORTS:
                for q, wn in enumerate(('far', 'pre', 'peri')):
                    var = f'{sname}_{wn}'
                    sub = tt[tt.cls == cl]
                    d1 = mouse_diff(sub[sub.cohort == c], var)
                    d2 = mouse_diff(sub[sub.cohort == c], var, strat=True)
                    v = d1['diff'].dropna()
                    vs = d2['diff'].dropna()
                    x = x0 + q
                    ax.scatter(np.full(len(v), x) + np.random.default_rng(q).uniform(-0.15, 0.15, len(v)), v, s=4,
                               color=k.COHORT_COLOR[c], alpha=0.35, lw=0)
                    if len(v):
                        ax.errorbar(x - 0.22, v.mean(), v.sem(), fmt='o', color='k', ms=3, lw=0.8)
                    if len(vs):
                        ax.errorbar(x + 0.22, vs.mean(), vs.sem(), fmt='D', color='0.5', ms=2.5, lw=0.8)
                    pn, pp, n = two_tests(v)
                    pn2, pp2, n2 = two_tests(vs)
                    stats_rows.append(dict(signal=sname, cls=cl, cohort=c, window=wn, n_mice=n, mean_diff=v.mean(),
                                           p_wilcoxon=pn, p_t=pp, strat_n=n2, strat_mean_diff=vs.mean(),
                                           strat_p_wilcoxon=pn2, strat_p_t=pp2))
                    star = '*' if np.isfinite(pn) and pn < 0.05 and pp < 0.05 else ''
                    if star:
                        ax.text(x, 1.0, star, transform=ax.get_xaxis_transform(), ha='center', fontsize=8)
                    ticks.append(x)
                    labs.append(wn)
                x0 += 3.6
            ax.axhline(0, color='k', lw=0.4)
            ax.set_xticks(ticks, labs, fontsize=5.5)
            for ci, c in enumerate(k.COHORTS):
                ax.text(ci * 3.6 + 1, -0.32, c, transform=ax.get_xaxis_transform(), ha='center',
                        color=k.COHORT_COLOR[c], fontsize=6)
            if i == 0:
                ax.set_title(f'{cl} trials: lick - no lick', fontsize=7)
            if j == 0:
                ax.set_ylabel(f'Δ {SIG_LABEL[sname]}', fontsize=6.5)
    fig.suptitle(f1.wrap('Day 0: lick minus no-lick difference per mouse in each window (far -1..-0.1 s, pre '
                         '-100..-30 ms, peri -30..0 ms). Dots = mice; black circle = mean +- SEM; grey diamond = '
                         'within-session-fifth stratified difference (session progression removed); * = Wilcoxon and '
                         't both p < 0.05 (raw difference)', fig), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    squares(fig)
    f1.save(fig, '003_day0_windows')


def fig_quintiles(tt):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9))
    var = 'wh_speed_peri'
    for j, cl in enumerate(('whisker', 'auditory', 'no stim')):
        ax = axs[j]
        for c in k.COHORTS:
            curves = []
            for _, g in tt[(tt.cls == cl) & (tt.cohort == c)].groupby('mouse'):
                g = g.dropna(subset=[var])
                if len(g) < 15 or g.lick.nunique() < 2:
                    continue
                q = pd.qcut(g[var].rank(method='first'), 5, labels=False)
                curves.append(g.groupby(q).lick.mean().reindex(range(5)).to_numpy())
            if not curves:
                continue
            mat = np.array(curves)
            m, se = k.mean_sem(mat)
            ax.errorbar(np.arange(1, 6), m, se, fmt='o-', color=k.COHORT_COLOR[c], ms=3, lw=1,
                        label=f'{c} ({len(mat)} mice)')
            slopes = [np.polyfit(np.arange(5), r, 1)[0] for r in mat if np.isfinite(r).all()]
            pn, pp, _ = two_tests(slopes)
            ax.text(0.02, 0.05 + 0.08 * k.COHORTS.index(c), f'{c} slope {p_txt(pn)} / {p_txt(pp)}',
                    transform=ax.transAxes, fontsize=5.5, color=k.COHORT_COLOR[c])
        ax.set_xticks(range(1, 6), ['1\nstill', '2', '3', '4', '5\nwhisking'])
        ax.set_xlabel('whisker speed -30..0 ms (within-session quintile)')
        ax.set_ylabel('P(lick)')
        ax.set_title(f'{cl} trials')
        ax.legend(frameon=False, loc='upper right')
    fig.suptitle('Day 0: probability of licking by whisking just before the stimulus (slope tests: Wilcoxon / t '
                 'across mice)', fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    squares(fig)
    f1.save(fig, '003_day0_quintiles')


PRED_NOPUPIL = ['wh_speed_peri', 'wh_speed_pre', 'wh_speed_far', 'wh_angle_peri', 'nose_speed_peri', 'jaw_peri',
                'tongue_far', 'session_frac', 'prev_lick', 'prev_reward', 'recent_lick_rate']
PRED = PRED_NOPUPIL + ['pupil_ts']  # pupil area 1 s before trial start (sessions with pupil QC pass / warn)


def glm(tt, cl, preds, min_trials=30):
    rows = []
    for (s, m, c), g in tt[tt.cls == cl].groupby(['session', 'mouse', 'cohort']):
        g = g.dropna(subset=preds)
        if len(g) < min_trials or g.lick.nunique() < 2 or min(g.lick.mean(), 1 - g.lick.mean()) < 0.1:
            continue
        X = g[preds].to_numpy(float)
        sd = X.std(0)
        keep = sd > 0
        X = (X[:, keep] - X[:, keep].mean(0)) / sd[keep]
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(X, g.lick.to_numpy())
        coef = np.full(len(preds), np.nan)
        coef[keep] = lr.coef_[0]
        rows.append(dict(session=s, mouse=m, cohort=c, n=len(g), **dict(zip(preds, coef))))
    return pd.DataFrame(rows)


def fig_confounds(tt, glm_rows):
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45)
    # (a) whisker speed peri and P(lick) across session fifths, whisker trials
    tt = tt.assign(fifth=np.minimum((tt.session_frac * 5).astype(int), 4))
    for j, (var, lab) in enumerate((('wh_speed_peri', 'whisker speed -30..0 ms (deg/s)'), ('lick', 'P(lick)'))):
        ax = fig.add_subplot(gs[0, j])
        for c in k.COHORTS:
            for lv, ls in ((1, '-'), (0, '--')) if var != 'lick' else ((None, '-'),):
                sub = tt[(tt.cls == 'whisker') & (tt.cohort == c)]
                if lv is not None:
                    sub = sub[sub.lick == lv]
                mat = sub.groupby(['mouse', 'fifth'])[var].mean().unstack().reindex(columns=range(5)).to_numpy()
                m, se = k.mean_sem(mat)
                ax.errorbar(np.arange(1, 6) + (0.05 if c == 'R-' else 0), m, se, fmt='o' + ls, color=k.COHORT_COLOR[c],
                            ms=2.5, lw=0.9, label=f'{c}' + ('' if lv is None else (' hit' if lv else ' miss')))
        ax.set_xlabel('session fifth')
        ax.set_ylabel(lab)
        ax.set_title('whisker trials', fontsize=7)
        ax.legend(frameon=False, fontsize=5)
    # (b) raw vs stratified hit - miss difference, whisker speed peri / pre
    ax = fig.add_subplot(gs[0, 2])
    x = 0
    for c in k.COHORTS:
        for wn in ('pre', 'peri'):
            sub = tt[(tt.cls == 'whisker') & (tt.cohort == c)]
            for strat, mk, col in ((False, 'o', 'k'), (True, 'D', '0.5')):
                v = mouse_diff(sub, f'wh_speed_{wn}', strat=strat)['diff'].dropna()
                pn, pp, _ = two_tests(v)
                ax.errorbar(x + (0.2 if strat else -0.2), v.mean(), v.sem(), fmt=mk, color=col, ms=3, lw=0.8)
                ax.text(x + (0.2 if strat else -0.2), v.mean() - v.sem() - 4, p_txt(pn), rotation=90, fontsize=4.5,
                        ha='center', va='top')
            x += 1
        x += 0.5
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xticks([0, 1, 2.5, 3.5], ['R+ pre', 'R+ peri', 'R- pre', 'R- peri'], fontsize=6)
    ax.set_ylabel('Δ whisker speed (deg/s)')
    ax.set_title('raw (black) vs stratified\nby session fifth (grey)', fontsize=7)
    # (c) GLM coefficients, whisker trials (full model), cohorts
    for j, cl in enumerate(('whisker', 'no stim')):
        ax = fig.add_subplot(gs[1, j * 1:(j + 1) * 1 + (1 if j == 1 else 0)])
        d = glm_rows.get((cl, 'full'))
        if d is None or not len(d):
            continue
        for ci, c in enumerate(k.COHORTS):
            sub = d[d.cohort == c]
            for pi, p in enumerate(PRED):
                v = sub[p].dropna()
                y = pi + (0.18 if ci else -0.18)
                ax.errorbar(v.mean(), y, xerr=v.sem(), fmt='o', color=k.COHORT_COLOR[c], ms=3, lw=0.8)
                pn, pt, _ = two_tests(v)
                if np.isfinite(pn) and pn < 0.05 and pt < 0.05:
                    ax.text(v.mean() + v.sem() + 0.05, y, '*', va='center', fontsize=7, color=k.COHORT_COLOR[c])
        ax.axvline(0, color='k', lw=0.4)
        ax.set_yticks(range(len(PRED)), PRED, fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel('logistic coefficient (z-scored predictors)')
        nn = d.groupby('cohort').size().to_dict()
        ax.set_title(f'{cl} trials: lick ~ all predictors\n(R+ {nn.get("R+", 0)}, R- {nn.get("R-", 0)} mice)',
                     fontsize=7)
    fig.suptitle(f1.wrap('Day 0 confounds: session progression (a), progression-stratified hit-miss difference (b), '
                         'per-session logistic regression with history, engagement, licking and other movements '
                         '(c; * = Wilcoxon and t across mice both p < 0.05)', fig), fontsize=6.5)
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.17, right=0.98)
    squares(fig)
    f1.save(fig, '003_day0_confounds')


def main():
    sess = k.sessions_and_cohorts()
    sess = sess[sess.stage == STAGE]
    if '--plot-only' not in sys.argv:
        with ProcessPoolExecutor(12) as ex:
            done = [s for s in ex.map(one, sess.to_dict('records')) if s]
        print('sessions cached', len(done), 'of', len(sess), flush=True)
    data = load_all()
    tt = trial_table(data)
    tt.to_parquet(TAB / '003_day0_trials.parquet')
    print(tt.groupby(['cohort', 'cls']).agg(mice=('mouse', 'nunique'), trials=('lick', 'size'),
                                             p_lick=('lick', 'mean')).round(3).to_string(), flush=True)
    rows = []
    fig_timecourses(data, rows)
    pd.DataFrame(rows).to_csv(TAB / '003_day0_timecourse_bins.csv', index=False)
    srows = []
    fig_windows(tt, srows)
    st = pd.DataFrame(srows)
    st.to_csv(TAB / '003_day0_window_stats.csv', index=False)
    fig_quintiles(tt)
    glm_rows, gstat = {}, []
    for cl in ('whisker', 'no stim'):
        for name, preds in (('full', PRED), ('full_nopupil', PRED_NOPUPIL), ('whisker_only', ['wh_speed_peri'])):
            d = glm(tt, cl, preds)
            glm_rows[(cl, name)] = d
            d.to_csv(TAB / f'003_day0_glm_{cl.replace(" ", "")}_{name}.csv', index=False)
            for c in k.COHORTS:
                for p in preds:
                    v = d.loc[d.cohort == c, p].dropna() if len(d) else pd.Series(dtype=float)
                    pn, pt, n = two_tests(v)
                    gstat.append(dict(cls=cl, model=name, cohort=c, predictor=p, n_mice=n,
                                      mean_coef=v.mean() if len(v) else np.nan, p_wilcoxon=pn, p_t=pt))
    gs = pd.DataFrame(gstat)
    gs.to_csv(TAB / '003_day0_glm_stats.csv', index=False)
    fig_confounds(tt, glm_rows)
    pd.set_option('display.width', 220)
    print(st[st.signal.isin(['wh_speed', 'wh_angle'])].round(4).to_string(index=False))
    print(gs.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
