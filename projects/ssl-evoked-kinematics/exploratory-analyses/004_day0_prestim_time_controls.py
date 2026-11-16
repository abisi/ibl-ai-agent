"""004 -- Day 0: is the pre-stimulus whisking / posture difference between upcoming licks and no-licks more than a
slow drift over the session, and is it specific to whisker trials?

Inputs: the 003 cache (cache/003v1_<session>.npz).

Signal: whisker speed and whisker angle in the far window (-1 .. -0.1 s before the stimulus = trial start; inside the
2-5 s no-lick quiet window); also the pre (-100 .. -30 ms) and peri (-30 .. 0 ms) windows.
Time controls (per mouse = its day-0 session; tested across mice with Wilcoxon and t, both reported):
  raw         mean(lick trials) - mean(no-lick trials)
  stratified  same within session fifths, averaged over fifths
  detrended   each trial minus the mean of the same-class trials within +-5 trials of it (itself excluded), then
              lick - no lick: removes any drift slower than ~10 trials
  matched     each lick trial paired with the nearest no-lick trial of the same class within 5 trials (in trial
              order; ties: earlier); mean paired difference
  local shuffle  labels permuted within consecutive blocks of 8 same-class trials (1,000 permutations): per-mouse z of
              the raw difference against this null, which keeps the session's slow structure (time, engagement)
Specificity: per session, one logistic regression over all active trials: lick ~ class (auditory, no stim vs whisker)
  + z(whisker speed far) + z(whisker speed far) x class + session fraction + previous lick / reward + recent lick rate
  (+ pupil 1 s before trial start where available). The interaction coefficients test whether the whisking effect
  differs between whisker trials and the other classes.
Outputs (combined_results_ks4/ssl-evoked-kinematics/): figures/004_day0_{time_controls,specificity,example}.{png,pdf},
tables/004_day0_time_controls.csv, tables/004_day0_specificity.csv
Run (haas, after 003): python 004_day0_prestim_time_controls.py
"""
import importlib.util
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m003', HERE / '003_day0_prestim_state.py')
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
k, f1 = m3.k, m3.f1
TAB = m3.TAB
METHODS = ['raw', 'stratified', 'detrended', 'matched', 'local shuffle z']
RNG = np.random.default_rng(0)
CLS_ORDER = ('whisker', 'auditory', 'no stim')


def detrended(g, var, half=5):
    x = g[var].to_numpy(float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        lo, hi = max(0, i - half), min(len(x), i + half + 1)
        nb = np.r_[x[lo:i], x[i + 1:hi]]
        nb = nb[np.isfinite(nb)]
        if len(nb) >= 3 and np.isfinite(x[i]):
            out[i] = x[i] - nb.mean()
    return out


def matched_diff(g, var, max_gap=5):
    x, lick = g[var].to_numpy(float), g.lick.to_numpy()
    no = np.flatnonzero((lick == 0) & np.isfinite(x))
    d = []
    for i in np.flatnonzero((lick == 1) & np.isfinite(x)):
        if not len(no):
            break
        j = no[np.argmin(np.abs(no - i))]
        if abs(j - i) <= max_gap:
            d.append(x[i] - x[j])
    return np.mean(d) if len(d) >= 3 else np.nan


def shuffle_z(g, var, block=8, n=1000):
    x, lick = g[var].to_numpy(float), g.lick.to_numpy()
    ok = np.isfinite(x)
    x, lick = x[ok], lick[ok]
    if lick.sum() < 3 or (1 - lick).sum() < 3:
        return np.nan
    obs = x[lick == 1].mean() - x[lick == 0].mean()
    blocks = [np.arange(i, min(i + block, len(x))) for i in range(0, len(x), block)]
    null = np.empty(n)
    for p in range(n):
        perm = lick.copy()
        for b in blocks:
            perm[b] = RNG.permutation(perm[b])
        if perm.sum() == 0 or perm.sum() == len(perm):
            null[p] = np.nan
            continue
        null[p] = x[perm == 1].mean() - x[perm == 0].mean()
    sd = np.nanstd(null)
    return (obs - np.nanmean(null)) / sd if sd > 0 else np.nan


def per_mouse(tt, cl, var):
    rows = []
    for (m, c), g in tt[tt.cls == cl].groupby(['mouse', 'cohort']):
        g = g.sort_values('trial_index')
        if g.lick.sum() < 3 or (1 - g.lick).sum() < 3:
            continue
        raw = g.loc[g.lick == 1, var].mean() - g.loc[g.lick == 0, var].mean()
        strat = m3.mouse_diff(g, var, strat=True)['diff'].iat[0]
        det = detrended(g, var)
        dd = np.nanmean(det[g.lick.to_numpy() == 1]) - np.nanmean(det[g.lick.to_numpy() == 0])
        rows.append(dict(mouse=m, cohort=c, cls=cl, var=var, n_lick=int(g.lick.sum()), n_nolick=int((1 - g.lick).sum()),
                         raw=raw, stratified=strat, detrended=dd, matched=matched_diff(g, var),
                         **{'local shuffle z': shuffle_z(g, var)}))
    return pd.DataFrame(rows)


def p_txt(p):
    return 'n.s.' if not np.isfinite(p) or p >= 0.05 else ('p<0.001' if p < 0.001 else f'p={p:.3f}')


def fig_time_controls(res, st):
    vars_ = [('wh_speed_far', 'Δ whisker speed (deg/s)'), ('wh_angle_far', 'Δ whisker angle (deg)')]
    fig, axs = plt.subplots(len(vars_), len(METHODS), figsize=(7.2, 3.3), squeeze=False)
    for i, (var, lab) in enumerate(vars_):
        for j, meth in enumerate(METHODS):
            ax = axs[i, j]
            for ci, c in enumerate(k.COHORTS):
                d = res[(res['var'] == var) & (res.cls == 'whisker') & (res.cohort == c)][meth].dropna()
                x = ci
                ax.scatter(np.full(len(d), x) + RNG.uniform(-0.15, 0.15, len(d)), d, s=4, color=k.COHORT_COLOR[c],
                           alpha=0.4, lw=0)
                ax.errorbar(x + 0.3, d.mean(), d.sem(), fmt='o', color='k', ms=3, lw=0.8)
                r = st[(st['var'] == var) & (st.cls == 'whisker') & (st.cohort == c) & (st.method == meth)]
                if len(r):
                    ax.text(x, 1.01, f'{p_txt(r.p_wilcoxon.iat[0])}\n{p_txt(r.p_t.iat[0])}',
                            transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=4.5)
            ax.axhline(0, color='k', lw=0.4)
            ax.set_xticks([0, 1], k.COHORTS)
            ax.set_xlim(-0.6, 1.6)
            ax.set_box_aspect(1)
            if i == 0:
                ax.set_title(meth, fontsize=6.5, pad=14)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6.5)
            elif meth == 'local shuffle z':
                ax.set_ylabel('z vs local shuffle', fontsize=6)
    fig.suptitle(f1.wrap('Day 0, whisker trials, 1 s before the stimulus (-1..-0.1 s): hit minus miss per mouse, '
                         'after removing slow session structure in four ways. p values: Wilcoxon (top) and t (bottom) '
                         'across mice', fig), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    f1.save(fig, '004_day0_time_controls')


def specificity(tt, with_pupil):
    rows = []
    for (s, m, c), g in tt.groupby(['session', 'mouse', 'cohort']):
        cols = ['wh_speed_far', 'session_frac', 'prev_lick', 'prev_reward', 'recent_lick_rate'] + \
               (['pupil_ts'] if with_pupil else [])
        g = g.dropna(subset=cols)
        if len(g) < 60 or g.lick.nunique() < 2 or set(g.cls) != set(CLS_ORDER):
            continue
        if any(g[g.cls == cl].lick.nunique() < 2 for cl in ('whisker', 'no stim')):
            continue
        z = {cc: (g[cc] - g[cc].mean()) / (g[cc].std() or 1) for cc in cols}
        X = pd.DataFrame({'aud': (g.cls == 'auditory').astype(float), 'nostim': (g.cls == 'no stim').astype(float),
                          'whisk': z['wh_speed_far']})
        X['whisk_x_aud'] = X.whisk * X.aud
        X['whisk_x_nostim'] = X.whisk * X.nostim
        for cc in cols[1:]:
            X[cc] = z[cc]
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(X.to_numpy(), g.lick.to_numpy())
        rows.append(dict(session=s, mouse=m, cohort=c, n=len(g), pupil=with_pupil,
                         **dict(zip(X.columns, lr.coef_[0]))))
    return pd.DataFrame(rows)


def fig_specificity(res, spec_df, st):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    # (a) detrended lick - no-lick whisker speed (far) per class
    ax = axs[0]
    for ci, c in enumerate(k.COHORTS):
        for q, cl in enumerate(CLS_ORDER):
            d = res[(res['var'] == 'wh_speed_far') & (res.cls == cl) & (res.cohort == c)]['detrended'].dropna()
            x = ci * 3.6 + q
            ax.scatter(np.full(len(d), x) + RNG.uniform(-0.12, 0.12, len(d)), d, s=3, color=k.COHORT_COLOR[c],
                       alpha=0.35, lw=0)
            if len(d):
                ax.errorbar(x, d.mean(), d.sem(), fmt='o', color='k', ms=3, lw=0.8)
                r = st[(st['var'] == 'wh_speed_far') & (st.cls == cl) & (st.cohort == c) & (st.method == 'detrended')]
                if len(r):
                    ax.text(x, 1.01, p_txt(r.p_wilcoxon.iat[0]), transform=ax.get_xaxis_transform(), ha='center',
                            fontsize=4.5, rotation=90, va='bottom')
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6], ['W', 'A', 'none'] * 2)
    for ci, c in enumerate(k.COHORTS):
        ax.text(ci * 3.6 + 1, -0.2, c, transform=ax.get_xaxis_transform(), ha='center', color=k.COHORT_COLOR[c])
    ax.set_ylabel('Δ whisker speed (deg/s)')
    ax.set_title('lick - no lick, detrended\n(-1..-0.1 s)', fontsize=6.5, pad=16)
    # (b, c) logistic coefficients: whisking main effect (= whisker trials) and interactions
    for j, pupil in enumerate((False, True)):
        ax = axs[1 + j]
        d = spec_df[spec_df.pupil == pupil]
        terms = ['whisk', 'whisk_x_aud', 'whisk_x_nostim', 'session_frac', 'prev_reward', 'recent_lick_rate'] + \
                (['pupil_ts'] if pupil else [])
        for ci, c in enumerate(k.COHORTS):
            sub = d[d.cohort == c]
            for ti, t in enumerate(terms):
                v = sub[t].dropna()
                y = ti + (0.18 if ci else -0.18)
                ax.errorbar(v.mean(), y, xerr=v.sem(), fmt='o', color=k.COHORT_COLOR[c], ms=3, lw=0.8)
                if len(v) > 3:
                    pw, pt = stats.wilcoxon(v).pvalue, stats.ttest_1samp(v, 0).pvalue
                    if pw < 0.05 and pt < 0.05:
                        ax.text(v.mean() + v.sem() + 0.04, y, '*', va='center', fontsize=7, color=k.COHORT_COLOR[c])
        ax.axvline(0, color='k', lw=0.4)
        lab = {'whisk': 'whisking (whisker trials)', 'whisk_x_aud': 'whisking x auditory',
               'whisk_x_nostim': 'whisking x no stim', 'session_frac': 'session fraction',
               'prev_reward': 'previous reward', 'recent_lick_rate': 'recent lick rate', 'pupil_ts': 'pupil (1 s pre)'}
        ax.set_yticks(range(len(terms)), [lab[t] for t in terms], fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel('logistic coefficient (z)')
        nn = d.groupby('cohort').size().to_dict()
        ax.set_title(f'{"with" if pupil else "without"} pupil\nR+ {nn.get("R+", 0)}, R- {nn.get("R-", 0)} mice',
                     fontsize=6.5)
    for ax in axs:
        ax.set_box_aspect(1)
    fig.suptitle(f1.wrap('Day 0: is the pre-stimulus whisking effect specific to whisker trials? (a) detrended lick - '
                         'no-lick difference per trial class; (b, c) one logistic model over all trial classes per '
                         'session, interaction = difference from whisker trials (* Wilcoxon and t p < 0.05)', fig),
                 fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    f1.save(fig, '004_day0_specificity')


def fig_example(data, tt, res):
    d = res[(res['var'] == 'wh_speed_far') & (res.cls == 'whisker')].dropna(subset=['detrended'])
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 6.4))
    for ci, c in enumerate(k.COHORTS):
        comp = tt[tt.cls == 'whisker'].groupby('mouse').wh_speed_far.apply(lambda v: v.notna().mean())
        sub = d[(d.cohort == c) & d.mouse.map(comp).ge(0.95)].sort_values('detrended')
        mouse = sub.mouse.iloc[len(sub) // 4] if len(sub) else None  # complete tracking; first quartile of effect
        if mouse is None:
            continue
        s = tt[tt.mouse == mouse].session.iat[0]
        x = data[s]
        sel = x['cls'] == 'whisker'
        m = x['sig|wh_speed'][sel]
        lick = x['lick'][sel]
        g = x['grid']
        ax = axs[0, ci]
        im = ax.imshow(m, aspect='auto', cmap='magma', vmin=0, vmax=np.nanpercentile(m, 95),
                       extent=[g[0], g[-1], len(m), 0], interpolation='nearest')
        for r, lv in enumerate(lick):
            ax.plot(g[-1] + 0.02, r + 0.5, 's', color=m3.LICK_COL if lv else '#4fa3e0', ms=1.6, clip_on=False)
        ax.axvline(0, color='w', lw=0.6, ls=':')
        ax.set_xlabel('time from stimulus (s)')
        ax.set_ylabel('whisker trial (session order)')
        ax.set_title(f'{mouse} ({c}): whisker speed\nright: black = hit, blue = miss', color=k.COHORT_COLOR[c],
                     fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.08, label='whisker speed (deg/s)')
        ax = axs[1, ci]
        w = tt[(tt.mouse == mouse) & (tt.cls == 'whisker')].sort_values('trial_index')
        v = w.wh_speed_far.to_numpy()
        trend = pd.Series(v).rolling(11, center=True, min_periods=4).mean().to_numpy()
        idx = np.arange(len(v))
        ax.plot(idx, trend, color='0.4', lw=1, label='running mean (11 trials)')
        ax.scatter(idx[w.lick.to_numpy() == 1], v[w.lick.to_numpy() == 1], s=6, color=m3.LICK_COL, label='hit')
        ax.scatter(idx[w.lick.to_numpy() == 0], v[w.lick.to_numpy() == 0], s=6, color='#4fa3e0', label='miss')
        ax.set_xlabel('whisker trial (session order)')
        ax.set_ylabel('whisker speed (deg/s)')
        ax.set_title('-1..-0.1 s before each whisker stimulus', fontsize=6.5)
        ax.legend(frameon=False, fontsize=5.5)
    for ax in axs.flat:
        ax.set_box_aspect(1)
    fig.suptitle(f1.wrap('Day 0 example mice: whisking before each whisker stimulus across the session; hits fall on '
                         'the low side of the local running mean, not only in low-whisking periods of the session',
                         fig), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    f1.save(fig, '004_day0_example')


def main():
    data = m3.load_all()
    tt = m3.trial_table(data)
    tt = tt.rename(columns={})  # pupil_ts present from 003 windows
    res = []
    for cl in CLS_ORDER:
        for var in ('wh_speed_far', 'wh_angle_far', 'wh_speed_pre', 'wh_speed_peri', 'wh_angle_peri', 'pupil_ts'):
            res.append(per_mouse(tt, cl, var))
    res = pd.concat(res, ignore_index=True)
    # robustness: trials with the lick already starting (tongue out or jaw opening above the session's no-lick 95th
    # percentile in -100..0 ms) removed
    thr = tt[tt.lick == 0].groupby('session').jaw_pre.quantile(0.95)
    clean = tt[(tt.tongue_pre.fillna(0) == 0) & (tt.tongue_peri.fillna(0) == 0) &
               (tt.jaw_pre.fillna(-np.inf) <= tt.session.map(thr).fillna(np.inf)) &
               (tt.jaw_peri.fillna(-np.inf) <= tt.session.map(thr).fillna(np.inf))]
    print('clean trials kept', round(len(clean) / len(tt), 3), clean.groupby(['cls', 'lick']).size().to_dict())
    rc = pd.concat([per_mouse(clean, 'whisker', v) for v in ('wh_speed_far', 'wh_angle_far')], ignore_index=True)
    rc['cls'] = 'whisker (no pre-stim lick onset)'
    res = pd.concat([res, rc], ignore_index=True)
    rows = []
    for (cl, var, c), g in res.groupby(['cls', 'var', 'cohort']):
        for meth in METHODS:
            v = g[meth].dropna()
            if len(v) > 3:
                rows.append(dict(cls=cl, var=var, cohort=c, method=meth, n_mice=len(v), mean=v.mean(),
                                 p_wilcoxon=stats.wilcoxon(v).pvalue, p_t=stats.ttest_1samp(v, 0).pvalue))
    st = pd.DataFrame(rows)
    st.to_csv(TAB / '004_day0_time_controls.csv', index=False)
    res.to_csv(TAB / '004_day0_time_controls_mouse.csv', index=False)
    spec_df = pd.concat([specificity(tt, False), specificity(tt, True)], ignore_index=True)
    spec_df.to_csv(TAB / '004_day0_specificity.csv', index=False)
    fig_time_controls(res, st)
    fig_specificity(res, spec_df, st)
    fig_example(data, tt, res)
    pd.set_option('display.width', 200)
    print(st[st['var'].isin(['wh_speed_far', 'wh_angle_far', 'pupil_ts'])].round(4).to_string(index=False))
    sp = []
    for (pupil, c), g in spec_df.groupby(['pupil', 'cohort']):
        for t in ('whisk', 'whisk_x_aud', 'whisk_x_nostim', 'session_frac', 'pupil_ts'):
            if t in g and g[t].notna().sum() > 3:
                v = g[t].dropna()
                sp.append(dict(pupil=pupil, cohort=c, term=t, n=len(v), mean=v.mean(),
                               p_wilcoxon=stats.wilcoxon(v).pvalue, p_t=stats.ttest_1samp(v, 0).pvalue))
    print(pd.DataFrame(sp).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
