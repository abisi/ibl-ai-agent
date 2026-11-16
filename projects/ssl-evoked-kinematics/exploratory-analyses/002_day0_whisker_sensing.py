"""002 -- Day 0: is there a stimulus- or cohort-specific whisker movement (sensing-like), beyond the lick-related
movements?

Inputs: the 000 cache (cache/000v1_<session>.npz / _trials.parquet), day-0 sessions only.

Trial types: whisker hit, whisker miss (no lick), auditory hit (auditory misses too rare). Whisker angle = base -> tip
angle to the head axis (deg; negative change = protraction), baseline-subtracted (-0.1 .. 0 s for single trials from
the skeleton cache; -0.5 .. 0 s for the session time courses). On whisker trials 0 .. 30 ms contains the stimulus
moving the whisker (magnetic deflection): excluded from every window.
Tests (per mouse = one day-0 session; R+ vs R-: Mann-Whitney U and Welch t; within mouse: Wilcoxon signed-rank and
paired t; uncorrected):
  A. evoked whisker movement 30-100 ms (before the jaw opens, jaw onset ~0.1-0.16 s): protraction (mean angle change)
     and whisking (mean angular speed minus baseline) per trial type; W miss vs 0; W hit vs W miss; W hit vs A hit;
     R+ vs R-.
  B. whisker angle around the first tongue reach (single trials from the skeleton, -150 .. +100 ms from reach onset;
     reaches 0.15 .. 0.35 s after the stimulus so that the window stays in the cached -0.1 .. 0.4 s): does the whisker
     protract before the tongue comes out, and more on whisker than auditory hits?
  C. pre-stimulus whisking (-0.5 .. 0 s angular speed) on whisker hits vs misses: whisking state at stimulus time.
Outputs (combined_results_ks4/ssl-evoked-kinematics/): figures/002_day0_whisker_sensing.{png,pdf},
tables/002_day0_whisker_sensing_mouse.csv, tables/002_day0_whisker_sensing_stats.csv
Run (haas, after 000): python 002_day0_whisker_sensing.py
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
spec = importlib.util.spec_from_file_location('k000', HERE / '000_evoked_kinematics.py')
k = importlib.util.module_from_spec(spec)
spec.loader.exec_module(k)
spec = importlib.util.spec_from_file_location('f001', HERE / '001_trajectory_figures.py')
f1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1)

STAGE = 'day0'
TT = ['whisker hit', 'whisker miss', 'auditory hit']
TT_COL = {'whisker hit': '#1f4e9c', 'whisker miss': '#8fb3e8', 'auditory hit': '#d1495b'}
EARLY = (0.03, 0.10)
BASE = (-0.5, 0.0)
REACH_WIN = (-0.15, 0.10)
REACH_LAT = (0.15, 0.35)


def angle_bt(sk):
    """skeleton trials x time x 3 x 2 -> base->tip angle (deg)"""
    d = sk[:, :, 2, :] - sk[:, :, 0, :]
    return np.degrees(np.arctan2(d[..., 1], d[..., 0]))


def collect():
    sess = k.sessions_and_cohorts()
    sess = sess[sess.stage == STAGE]
    rows, tcs, rl = [], {}, {}
    grid = sk_grid = None
    for r in sess.itertuples():
        f = k.CACHE / f'000v1_{r.session}.npz'
        if not f.exists():
            continue
        d = np.load(f)
        if 'grid' in d.files:
            grid = d['grid']
        t = pd.read_parquet(k.CACHE / f'000v1_{r.session}_trials.parquet')
        for tt in TT:
            row = dict(cohort=r.cohort, mouse=r.mouse, session=r.session, trial_type=tt)
            for var in ('wh_angle', 'wh_speed'):
                key = f'tc|{var}|{tt}'
                if key not in d.files:
                    continue
                x = d[key]
                tcs.setdefault((r.cohort, tt, var), {})[r.mouse] = x
                e = (grid >= EARLY[0]) & (grid <= EARLY[1])
                b = (grid >= BASE[0]) & (grid < BASE[1])
                if var == 'wh_angle':
                    row['early_protraction_deg'] = float(np.nanmean(x[e]))
                else:
                    row['early_speed_change'] = float(np.nanmean(x[e]) - np.nanmean(x[b]))
                    row['baseline_speed'] = float(np.nanmean(x[b]))
            row['n_trials'] = int(d[f'n|wh_angle|{tt}'][0]) if f'n|wh_angle|{tt}' in d.files else 0
            rows.append(row)
            # reach-aligned whisker angle from single-trial skeletons
            key = f'skel|{tt}'
            if tt != 'whisker miss' and key in d.files and 'reach_latency' in t:
                sk_grid = d['sk_grid']
                sk = d[key]
                lat = t.loc[t.trial_type == tt, 'reach_latency'].to_numpy()
                if len(lat) != len(sk):
                    continue
                ang = angle_bt(sk)
                ang = ang - np.nanmean(ang[:, sk_grid < 0], 1, keepdims=True)
                rel = np.arange(REACH_WIN[0], REACH_WIN[1] + 1e-9, 0.005)
                out = []
                for a, L in zip(ang, lat):
                    if not (np.isfinite(L) and REACH_LAT[0] <= L <= REACH_LAT[1]):
                        continue
                    q = L + rel
                    idx = np.round((q - sk_grid[0]) / 0.005).astype(int)
                    ok = (idx >= 0) & (idx < len(sk_grid))
                    v = np.full(len(rel), np.nan)
                    v[ok] = a[idx[ok]]
                    if np.isfinite(v).mean() >= 0.8:
                        out.append(v)
                if len(out) >= 5:
                    rl.setdefault((r.cohort, tt), {})[r.mouse] = (rel, np.nanmean(np.array(out), 0), len(out))
    return pd.DataFrame(rows), tcs, rl, grid


def tests(df):
    out = []
    for metric in ('early_protraction_deg', 'early_speed_change', 'baseline_speed'):
        for tt in TT:
            a = df[(df.cohort == 'R+') & (df.trial_type == tt)][metric].dropna()
            b = df[(df.cohort == 'R-') & (df.trial_type == tt)][metric].dropna()
            if len(a) > 2 and len(b) > 2:
                out.append(dict(metric=metric, test='R+ vs R-', subset=tt, n1=len(a), n2=len(b), mean1=a.mean(),
                                mean2=b.mean(), p_nonparam=stats.mannwhitneyu(a, b).pvalue,
                                p_param=stats.ttest_ind(a, b, equal_var=False).pvalue))
        for c in k.COHORTS:
            w = df[df.cohort == c].pivot_table(index='mouse', columns='trial_type', values=metric)
            for t1, t2 in (('whisker hit', 'whisker miss'), ('whisker hit', 'auditory hit'),
                           ('whisker miss', 'auditory hit')):
                if {t1, t2} <= set(w.columns):
                    p = w[[t1, t2]].dropna()
                    if len(p) > 2:
                        out.append(dict(metric=metric, test=f'{t1} vs {t2} (paired)', subset=c, n1=len(p), n2=len(p),
                                        mean1=p[t1].mean(), mean2=p[t2].mean(),
                                        p_nonparam=stats.wilcoxon(p[t1], p[t2]).pvalue,
                                        p_param=stats.ttest_rel(p[t1], p[t2]).pvalue))
            if metric != 'baseline_speed':
                for tt in TT:
                    v = df[(df.cohort == c) & (df.trial_type == tt)][metric].dropna()
                    if len(v) > 2:
                        out.append(dict(metric=metric, test='vs 0', subset=f'{c} {tt}', n1=len(v), n2=np.nan,
                                        mean1=v.mean(), mean2=0, p_nonparam=stats.wilcoxon(v).pvalue,
                                        p_param=stats.ttest_1samp(v, 0).pvalue))
    return pd.DataFrame(out)


def p_txt(p):
    return 'p<0.001' if p < 0.001 else f'p={p:.3f}'


def figure(df, tcs, rl, grid, st):
    fig = plt.figure(figsize=(7.2, 7.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], hspace=0.6, wspace=0.55)
    # A: time courses (angle, speed) per cohort, trial types overlaid
    for i, c in enumerate(k.COHORTS):
        for j, (var, lab) in enumerate((('wh_angle', 'whisker angle (deg)\n- = protraction'),
                                        ('wh_speed', 'whisker angular\nspeed (deg/s)'))):
            ax = fig.add_subplot(gs[j, 2 * i:2 * i + 2])
            for tt in TT:
                per = tcs.get((c, tt, var), {})
                if not per:
                    continue
                mat = np.array(list(per.values()))
                m, se = k.mean_sem(mat)
                ax.fill_between(grid, m - se, m + se, color=TT_COL[tt], alpha=0.25, lw=0)
                ax.plot(grid, m, color=TT_COL[tt], lw=1, label=f'{tt} ({len(mat)})')
            ax.axvspan(0, 0.03, color='0.85', lw=0, zorder=0)
            ax.axvspan(*EARLY, color='#fff3c4', lw=0, zorder=0)
            ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.set_xlim(-0.2, 0.6)
            ax.set_ylabel(lab)
            if j == 0:
                ax.set_title(f'{c}', color=k.COHORT_COLOR[c])
                ax.legend(frameon=False, loc='lower left' if var == 'wh_angle' else 'upper right')
            else:
                ax.set_xlabel('time from stimulus (s)')
    # B: per-mouse early window metrics
    for j, (metric, lab) in enumerate((('early_protraction_deg', 'protraction 30-100 ms (deg)'),
                                       ('early_speed_change', 'speed change 30-100 ms (deg/s)'))):
        ax = fig.add_subplot(gs[2, j])
        for i, c in enumerate(k.COHORTS):
            w = df[df.cohort == c].pivot_table(index='mouse', columns='trial_type', values=metric)[TT]
            x0 = i * 3.6
            for _, row in w.iterrows():
                ax.plot(x0 + np.arange(3), row.values, '-', color=k.COHORT_COLOR[c], lw=0.3, alpha=0.25)
            for q, tt in enumerate(TT):
                v = w[tt].dropna()
                ax.errorbar(x0 + q, v.mean(), v.sem(), fmt='o', color=TT_COL[tt], mec='k', mew=0.4, ms=4,
                            capsize=0, lw=1, zorder=3)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6], ['W hit', 'W miss', 'A hit'] * 2, rotation=60)
        ax.set_ylabel(lab)
        for i, c in enumerate(k.COHORTS):
            ax.text(i * 3.6 + 1, 1.02, c, transform=ax.get_xaxis_transform(), ha='center', color=k.COHORT_COLOR[c])
        s = st[(st.metric == metric)]
        lines = []
        for c in k.COHORTS:
            r1 = s[(s.test == 'vs 0') & (s.subset == f'{c} whisker miss')]
            r2 = s[(s.test == 'whisker hit vs auditory hit (paired)') & (s.subset == c)]
            if len(r1):
                lines.append(f'{c} W miss vs 0: {p_txt(r1.p_nonparam.iat[0])}')
            if len(r2):
                lines.append(f'{c} W hit vs A hit: {p_txt(r2.p_nonparam.iat[0])}')
        r3 = s[(s.test == 'R+ vs R-') & (s.subset == 'whisker miss')]
        if len(r3):
            lines.append(f'R+ vs R- W miss: {p_txt(r3.p_nonparam.iat[0])}')
        ax.set_title('\n'.join(lines), fontsize=5, loc='left')
    # C: reach-aligned whisker angle
    ax = fig.add_subplot(gs[2, 2:])
    for c in k.COHORTS:
        for tt, ls in (('whisker hit', '-'), ('auditory hit', '--')):
            per = rl.get((c, tt), {})
            if not per:
                continue
            rel = next(iter(per.values()))[0]
            mat = np.array([v[1] for v in per.values()])
            m, se = k.mean_sem(mat)
            ax.fill_between(rel * 1000, m - se, m + se, color=k.COHORT_COLOR[c], alpha=0.2, lw=0)
            ax.plot(rel * 1000, m, ls, color=k.COHORT_COLOR[c], lw=1,
                    label=f'{c} {tt} ({len(mat)} mice, {sum(v[2] for v in per.values())} trials)')
    ax.axvline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('time from tongue reach onset (ms)')
    ax.set_ylabel('whisker angle change (deg)')
    ax.set_title(f'whisker vs reach onset (reaches {REACH_LAT[0]:.2f}-{REACH_LAT[1]:.2f} s after stimulus)',
                 fontsize=6.5)
    ax.legend(frameon=False, fontsize=5)
    fig.suptitle(f1.wrap('Day 0: whisker movement by trial type and cohort. Grey band = whisker stimulus transient '
                         '(0-30 ms, excluded); yellow = early window 30-100 ms (before the jaw opens). Mean +- SEM '
                         'across mice; tests on mouse values (MWU / Wilcoxon shown; Welch / paired t in the table)',
                         fig), fontsize=7)
    fig.subplots_adjust(top=0.9, bottom=0.1, left=0.09, right=0.98)
    f1.save(fig, '002_day0_whisker_sensing')


def main():
    df, tcs, rl, grid = collect()
    df.to_csv(k.OUT / 'tables' / '002_day0_whisker_sensing_mouse.csv', index=False)
    st = tests(df)
    st.to_csv(k.OUT / 'tables' / '002_day0_whisker_sensing_stats.csv', index=False)
    pd.set_option('display.width', 200)
    print(st.round(4).to_string(index=False))
    figure(df, tcs, rl, grid, st)


if __name__ == '__main__':
    main()
