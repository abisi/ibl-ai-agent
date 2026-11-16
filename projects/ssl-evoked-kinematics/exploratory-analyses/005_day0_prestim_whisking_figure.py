"""005 -- Day 0 summary figures: whisker speed, whisker posture and jaw posture in the second before trial start (=
stimulus) on whisker hits, whisker misses and auditory hits, R+ and R-, with every mouse shown.

Inputs: the 003 cache (cache/003v1_<session>.npz).

Trials: active, early_lick == 0, and no lick onset before the stimulus (tongue out or jaw opening above the session's
no-lick 95th percentile in -100 .. 0 ms removed, as 004).
Variables (mean over -1 .. -0.1 s before trial start):
  whisking         whisker speed = |d angle / dt| of the base -> tip whisker angle (top view, 000 skeleton rules)
  whisker_posture  whisker angle minus the session median (deg; negative = more protracted)
  jaw_posture      jaw opening minus the session median (mm; dlc_helpers jaw_distance, not likelihood-gated)
Per mouse (day-0 session): mean per trial type (>= 3 trials). Detrended: each trial minus the mean of all active
trials within +-5 trials of it (any type, itself excluded), removing slow drift over the session.
Time courses: per mouse the mean over trials (whisker speed: mean of log10(speed + 1), i.e. geometric, which tames
single bursts), 5-ms bins without smoothing, then mean +- SEM or SD across mice (two versions of each figure).
Error bars: mean +- SEM or SD across mice (whisker speed, log axis: geometric mean x/ geometric SEM or SD).
Tests across mice, paired: whisker hit vs whisker miss, whisker hit vs auditory hit (Wilcoxon signed-rank and paired t,
both reported; uncorrected).
Outputs: figures/005_day0_prestim_<variable>_{sem,sd}.{png,pdf}, tables/005_day0_prestim_<variable>_{mouse,stats}.csv
Run (haas, after 003): python 005_day0_prestim_whisking_figure.py
"""
import importlib.util
import pathlib

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m003', HERE / '003_day0_prestim_state.py')
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)
k, f1 = m3.k, m3.f1
TT = ['whisker hit', 'whisker miss', 'auditory hit']
TT_SHORT = ['W hit', 'W miss', 'A hit']
FAR = (-1.0, -0.1)
BRACKETS = ((('whisker hit', 'whisker miss'), 0.90), (('whisker hit', 'auditory hit'), 1.00))
# name -> (cached signal, window column, y label, log axis, title phrase)
VARIABLES = {
    'whisking': ('wh_speed', 'wh_speed_far', 'whisker speed (deg/s)', True, 'whisker speed'),
    'whisker_posture': ('wh_angle', 'wh_angle_far', 'whisker angle (deg)', False, 'whisker angle (posture)'),
    'jaw_posture': ('jaw', 'jaw_far', 'jaw opening (mm)', False, 'jaw opening (posture)'),
}


def trial_type(cls, lick):
    if cls == 'whisker':
        return 'whisker hit' if lick else 'whisker miss'
    if cls == 'auditory':
        return 'auditory hit' if lick else 'auditory miss'
    return 'false alarm' if lick else 'correct rejection'


def clean_mask(tt):
    thr = tt[tt.lick == 0].groupby('session').jaw_pre.quantile(0.95)
    t = tt.session.map(thr).fillna(np.inf)
    return ((tt.tongue_pre.fillna(0) == 0) & (tt.tongue_peri.fillna(0) == 0) &
            (tt.jaw_pre.fillna(-np.inf) <= t) & (tt.jaw_peri.fillna(-np.inf) <= t))


def detrend_all(g, var, half=5):
    g = g.sort_values('trial_index')
    x = g[var].to_numpy(float)
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        nb = np.r_[x[max(0, i - half):i], x[i + 1:i + half + 1]]
        nb = nb[np.isfinite(nb)]
        if len(nb) >= 3 and np.isfinite(x[i]):
            out[i] = x[i] - nb.mean()
    return pd.Series(out, index=g.index)


def spread(x, err, axis=0):
    """SEM or SD across mice (NaN-aware)"""
    n = np.sum(np.isfinite(x), axis=axis)
    sd = np.nanstd(x, axis=axis, ddof=1)
    return sd if err == 'sd' else sd / np.sqrt(np.maximum(n, 1))


def centre_err(v, log, err):
    v = v.dropna()
    if log:
        lv = np.log10(v[v > 0]).to_numpy()
        mu, e = lv.mean(), spread(lv, err)
        return 10 ** mu, [[10 ** mu - 10 ** (mu - e)], [10 ** (mu + e) - 10 ** mu]]
    return v.mean(), spread(v.to_numpy(), err)


def p_fmt(v):
    return 'p<0.001' if v < 0.001 else f'p={v:.3f}'


def figure(name, data, tt_all, err='sem'):
    sig, col_far, ylab, log, phrase = VARIABLES[name]
    tt = tt_all.copy()
    tt['det'] = pd.concat([detrend_all(g, col_far) for _, g in tt.groupby('session')])
    tt = tt[clean_mask(tt)]
    rows = []
    for (m, c, t), g in tt[tt.type.isin(TT)].groupby(['mouse', 'cohort', 'type']):
        if g[col_far].notna().sum() >= 3:
            rows.append(dict(mouse=m, cohort=c, trial_type=t, n=int(g[col_far].notna().sum()),
                             raw=g[col_far].mean(), detrended=g['det'].mean()))
    mouse = pd.DataFrame(rows)
    mouse.to_csv(m3.TAB / f'005_day0_prestim_{name}_mouse.csv', index=False)
    st = []
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
    grid = next(iter(data.values()))['grid']
    tc_lims = []
    for i, c in enumerate(k.COHORTS):
        col = k.COHORT_COLOR[c]
        tc_style = {'whisker hit': dict(color=col, lw=1.3), 'whisker miss': dict(color=col, lw=1.3, alpha=0.4),
                    'auditory hit': dict(color='0.35', lw=1.3)}
        ax = axs[i, 0]
        for t in TT:
            per = []
            for s, d in data.items():
                if d['cohort'] != c or f'sig|{sig}' not in d:
                    continue
                sub = tt[(tt.session == s) & (tt.type == t)]
                if len(sub) < 3:
                    continue
                sel = np.isin(d['trial_index'], sub.trial_index.to_numpy())
                x = d[f'sig|{sig}'][sel].astype(float)
                with np.errstate(all='ignore'):
                    if log:  # speed: mean of log10(speed + 1) over trials (geometric, tames single bursts)
                        per.append(np.nanmean(np.log10(np.clip(x, 0, None) + 1), 0))
                    else:
                        per.append(np.nanmean(x, 0))
            if per:
                arr = np.array(per)  # mice x bins, no smoothing
                with np.errstate(all='ignore'):
                    mm, se = np.nanmean(arr, 0), spread(arr, err)
                if log:
                    mm, se_lo, se_hi = 10 ** mm - 1, 10 ** (mm - se) - 1, 10 ** (mm + se) - 1
                    se = None
                lo_b, hi_b = (se_lo, se_hi) if se is None else (mm - se, mm + se)
                ax.fill_between(grid, lo_b, hi_b, color=tc_style[t]['color'],
                                alpha=0.15 if t != 'whisker miss' else 0.08, lw=0)
                ax.plot(grid, mm, label=t, **tc_style[t])
                ok = (grid >= -1.0) & (grid <= -0.05)
                tc_lims += [np.nanmin(lo_b[ok]), np.nanmax(hi_b[ok])]
        ax.axvspan(*FAR, color='0.95', lw=0, zorder=0)
        ax.set_xlim(-1.0, 0.0)
        ax.set_ylabel(ylab)
        ax.set_title(f'{c} · time course', color=col, pad=16)
        ax.legend(frameon=False, loc='upper left', handlelength=1.6, fontsize=5.5)
        ax.set_xlabel('time from trial start (s)')
        for j, val in enumerate(('raw', 'detrended')):
            ax = axs[i, 1 + j]
            w = mouse[mouse.cohort == c].pivot(index='mouse', columns='trial_type', values=val).reindex(columns=TT)
            for _, r in w.iterrows():
                ax.plot(range(3), r.values, '-', color='0.72', lw=0.3, zorder=1)
                ax.plot(range(3), r.values, 'o', color='0.72', ms=1.2, mew=0, zorder=1)
            centre = []
            for q, t in enumerate(TT):
                cm, eb = centre_err(w[t], log and val == 'raw', err)
                centre.append(cm)
                ax.errorbar(q, cm, eb, fmt='o', color=col, mfc=col if t != 'whisker miss' else 'w', mec=col,
                            mew=1.0, ms=5.5, lw=1.4, capsize=0, zorder=4)
            ax.plot(range(3), centre, '-', color=col, lw=1.0, zorder=3)
            if val == 'raw' and log:
                ax.set_yscale('log')
                lo, hi = np.nanpercentile(w.to_numpy(), [1, 99])
                ax.set_ylim(max(lo * 0.6, 3), hi * 6)
            else:
                if val == 'detrended' or not log:
                    ax.axhline(0, color='k', lw=0.4, ls=':')
                lo, hi = np.nanpercentile(w.to_numpy(), [2, 98])
                span = hi - lo
                ax.set_ylim(lo - 0.1 * span, hi + 0.45 * span)
            tr = ax.get_xaxis_transform()
            for (a, b), y in BRACKETS:
                p = w[[a, b]].dropna()
                pw, pt = stats.wilcoxon(p[a], p[b]).pvalue, stats.ttest_rel(p[a], p[b]).pvalue
                st.append(dict(variable=name, cohort=c, value=val, comparison=f'{a} vs {b}', n_mice=len(p),
                               mean_a=p[a].mean(), mean_b=p[b].mean(), p_wilcoxon=pw, p_paired_t=pt))
                x0, x1 = TT.index(a), TT.index(b)
                ax.plot([x0, x0, x1, x1], [y - 0.025, y - 0.005, y - 0.005, y - 0.025], color='k', lw=0.5,
                        transform=tr, clip_on=False)
                ax.text((x0 + x1) / 2, y, f'{p_fmt(pw)} / {p_fmt(pt)}', ha='center', va='bottom', fontsize=5,
                        transform=tr)
            ax.set_xticks(range(3), TT_SHORT)
            ax.set_xlim(-0.4, 2.4)
            ax.set_ylabel(ylab if val == 'raw' else f'Δ {ylab}')
            n = int(w.notna().all(1).sum())
            ax.set_title(f'{"raw" if val == "raw" else "detrended (vs ±5 trials)"} · {n} mice', color=col, pad=16)
    if tc_lims:  # same y range for both cohorts' time courses
        lo, hi = np.nanmin(tc_lims), np.nanmax(tc_lims)
        pad = 0.1 * (hi - lo)
        for ax in axs[:, 0]:
            ax.set_ylim(lo - pad, hi + 2.5 * pad)
    for ax in axs.flat:
        ax.set_box_aspect(1)
    for j in range(3):
        fig.align_ylabels(axs[:, j])
    E = err.upper()
    fig.suptitle(f1.wrap(f'Day 0, {phrase} in the second before trial start (-1..-0.1 s, grey band). Time course: '
                         f'per-mouse mean over trials (speed: geometric), mean +- {E} across mice, 5-ms bins, no smoothing. Per mouse: '
                         f'grey lines = mice, coloured = mean +- {E} across mice' +
                         (' (raw: geometric)' if log else '') + '; open = whisker miss. Detrended = each trial minus '
                         'the mean of the active trials within +-5 trials. Trials with a lick already starting '
                         'removed. p = Wilcoxon / paired t, uncorrected', fig), fontsize=6.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.2)
    f1.save(fig, f'005_day0_prestim_{name}_{err}')
    st = pd.DataFrame(st)
    st.to_csv(m3.TAB / f'005_day0_prestim_{name}_stats.csv', index=False)
    return st


def main():
    data = m3.load_all()
    tt = m3.trial_table(data)
    tt['type'] = [trial_type(c, l) for c, l in zip(tt.cls, tt.lick)]
    pd.set_option('display.width', 200)
    for name in VARIABLES:
        for err in ('sem', 'sd'):
            st = figure(name, data, tt, err)
        print(st.round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
