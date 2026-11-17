"""001 -- Striatal subdivisions (DMS / DLS / VS / TS; allen_utils.create_areas_subdivisions coordinate boxes) before vs
after the 2026-11-17 coordinate fix (allen_utils d6c32dc: bregma-centric ap / ml / dv from ccf_atlas_* instead of ccf_*),
for every NWB_ks4 unit (good + mua): counts (units, sessions, mice), old -> new transitions, positions, day-0 ROC response
prevalence per subdivision (session = unit of analysis: Kruskal-Wallis + one-way ANOVA across subdivisions), and
brain_wide_analysis plotting_functions.plot_area_proj_brain (the area_summary plot) per subdivision.
Run on haas (brain_wide_analysis env):  python 001_striatum_split.py [--reload] [--replot]
Output: combined_results_ks4/ssl-striatum-subdivisions/ (tables/, figures/, area_summary/, provenance.json)"""
import contextlib
import glob
import io
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd

A = '/mnt/lsens-analysis/Axel_Bisi'
OUT = f'{A}/combined_results_ks4/ssl-striatum-subdivisions'
SUBS = ['DMS', 'DLS', 'VS', 'TS']
COL = {'DMS': '#1b9e77', 'DLS': '#d95f02', 'VS': '#7570b3', 'TS': '#e7298a', 'other': '#bbbbbb'}
TYPES = [('whisker_passive_pre', 'Passive whisker'), ('auditory_passive_pre', 'Passive auditory'),
         ('whisker_active', 'Whisker (active)'), ('auditory_active', 'Auditory (active)'),
         ('whisker_choice', 'Whisker choice'), ('spontaneous_licks', 'Spontaneous licks')]
MIN_UNITS = 10                                    # units of a subdivision in a session for a session-level fraction
KEYS = ['session_id', 'electrode_group', 'cluster_id']


def one(nwb):
    for p in [f'{A}/Github/ephys_utilities', f'{A}/NWB_reader']:
        sys.path.insert(0, p)
    from pynwb import NWBHDF5IO
    from ephys_utilities.helpers import data_utils
    import ephys_utilities.allen_utils.allen_utils as au
    try:
        with NWBHDF5IO(nwb, 'r') as f:
            n = f.read()
            if n.units is None:
                return None
            u = n.units.to_dataframe(exclude={'spike_times', 'waveform_mean', 'waveform_mean_bc', 'maxChannels'})
        u['session_id'] = os.path.basename(nwb)[:-4]
        u['mouse_id'] = u['session_id'].str[:5]
        u = data_utils.convert_electrode_group_object_to_columns(u.reset_index(drop=True))
        u = u[u['bc_label'].isin(['good', 'mua'])]
        with contextlib.redirect_stdout(io.StringIO()):
            new = au.process_allen_labels(u.copy(), split_merge_areas=True)
            old_in = u.copy()
            for k in ('ap', 'ml', 'dv'):                      # previous behaviour: bregma coordinates from ccf_ap/ml/dv
                old_in[f'ccf_atlas_{k}'] = old_in[f'ccf_{k}']
            old = au.process_allen_labels(old_in, split_merge_areas=True)
        k = ['electrode_group', 'cluster_id']
        m = new[k + ['mouse_id', 'session_id', 'target_region', 'area_acronym_custom', 'area_group', 'ccf_atlas_ap',
                     'ccf_atlas_ml', 'ccf_atlas_dv', 'ap', 'ml', 'dv']].merge(
            old[k + ['area_acronym_custom']].rename(columns={'area_acronym_custom': 'old_label'}), on=k)
        m = m[m['area_acronym_custom'].isin(SUBS) | m['old_label'].isin(SUBS) | (m['area_group'] == 'Striatum')]
        m['electrode_group'] = m['electrode_group'].astype(str)
        return m
    except Exception as err:  # noqa: BLE001
        print(nwb, repr(err)[:150], flush=True)
        return None


def load_units(reload):
    f = f'{OUT}/tables/striatal_units.csv'
    if os.path.exists(f) and not reload:
        return pd.read_csv(f, dtype={'cluster_id': str, 'electrode_group': str})
    nwbs = sorted(p for p in glob.glob(f'{A}/NWB_ks4/*.nwb') if re.fullmatch(r'[A-Z]{2}\d{3}_\d{8}_\d{6}', os.path.basename(p)[:-4]))
    with ProcessPoolExecutor(64) as ex:
        s = pd.concat([r for r in ex.map(one, nwbs) if r is not None], ignore_index=True)
    s['new'] = np.where(s['area_acronym_custom'].isin(SUBS), s['area_acronym_custom'], 'other')
    s['old'] = np.where(s['old_label'].isin(SUBS), s['old_label'], 'other')
    s['cluster_id'] = s['cluster_id'].astype(str)
    s.to_csv(f, index=False)
    return s


def main():
    from scipy.stats import f_oneway, kruskal
    os.makedirs(f'{OUT}/tables', exist_ok=True); os.makedirs(f'{OUT}/figures', exist_ok=True)
    s = load_units('--reload' in sys.argv)

    # counts and transitions
    rows = []
    for lab in ('old', 'new'):
        for sub in SUBS:
            x = s[s[lab] == sub]
            rows.append(dict(labels=lab, subdivision=sub, units=len(x), sessions=x['session_id'].nunique(), mice=x['mouse_id'].nunique()))
    counts = pd.DataFrame(rows)
    counts.to_csv(f'{OUT}/tables/counts_old_vs_new.csv', index=False)
    trans = pd.crosstab(s['old'], s['new']).reindex(index=SUBS + ['other'], columns=SUBS + ['other'], fill_value=0)
    trans.to_csv(f'{OUT}/tables/transitions_old_to_new.csv')

    # day-0 ROC response prevalence, per session x subdivision (session = unit of analysis)
    R = f'{A}/combined_results_ks4/_roc_stage_analysis'
    roc = pd.read_parquet(f'{R}/roc_long.parquet', filters=[('analysis_type', 'in', [t[0] for t in TYPES])])
    info = pd.read_parquet(f'{R}/unit_info_all_days_v2.parquet', columns=KEYS + ['day'])
    for d in (roc, info):
        d['cluster_id'] = d['cluster_id'].astype(str); d['electrode_group'] = d['electrode_group'].astype(str)
    r = (roc.drop(columns=['mouse_id']).merge(info.drop_duplicates(KEYS), on=KEYS)
         .merge(s[KEYS + ['mouse_id', 'new', 'old']], on=KEYS))
    r = r[r['day'] == 0]
    per_s, tests = [], []
    for lab in ('old', 'new'):
        g = (r[r[lab].isin(SUBS)].groupby(['analysis_type', lab, 'session_id', 'mouse_id'])
             .agg(n=('sig', 'size'), frac_sig=('sig', 'mean')).reset_index().rename(columns={lab: 'subdivision'}))
        g = g[g['n'] >= MIN_UNITS].assign(labels=lab, frac_sig=lambda d: 100 * d['frac_sig'])
        per_s.append(g)
        for ty, _ in TYPES:
            grp = [g[(g.analysis_type == ty) & (g.subdivision == sub)]['frac_sig'].to_numpy() for sub in SUBS]
            ok = [x for x in grp if len(x) >= 3]
            kw = kruskal(*ok) if len(ok) >= 2 else (np.nan, np.nan)
            an = f_oneway(*ok) if len(ok) >= 2 else (np.nan, np.nan)
            row = dict(labels=lab, analysis_type=ty, kw_H=kw[0], kw_p=kw[1], anova_F=an[0], anova_p=an[1])
            for sub, x in zip(SUBS, grp):
                row[f'n_sessions_{sub}'] = len(x); row[f'mean_{sub}'] = np.mean(x) if len(x) else np.nan
                row[f'sem_{sub}'] = np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
            tests.append(row)
    per_s = pd.concat(per_s, ignore_index=True)
    per_s.to_csv(f'{OUT}/tables/roc_frac_sig_per_session.csv', index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(f'{OUT}/tables/roc_subdivision_tests.csv', index=False)

    # figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
          'font.size': 6, 'axes.titlesize': 6.5, 'axes.labelsize': 6, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
          'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5}
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(18 / 2.54, 15 / 2.54))
        gs = fig.add_gridspec(2, 4, hspace=0.6, wspace=0.55)
        a = fig.add_subplot(gs[0, 0])
        xx = np.arange(4)
        for j, lab in enumerate(('old', 'new')):
            c = counts[counts['labels'] == lab].set_index('subdivision').reindex(SUBS)
            a.bar(xx + (j - 0.5) * 0.38, c['units'], 0.38, color=[COL[x] for x in SUBS], alpha=0.4 if lab == 'old' else 1,
                  edgecolor='k', lw=0.3, label=f'{lab} labels')
        a.set_xticks(xx, SUBS); a.set_ylabel('Units (good + mua, all days)'); a.legend(frameon=False, fontsize=5)
        a.set_title('a  Units per subdivision', loc='left')
        b = fig.add_subplot(gs[0, 1])
        T = trans.to_numpy()
        b.imshow(np.log10(T + 1), cmap='Greys')
        for (i, j), v in np.ndenumerate(T):
            b.text(j, i, str(v), ha='center', va='center', fontsize=4.5, color='w' if v > T.max() / 8 else 'k')
        b.set_xticks(range(5), trans.columns); b.set_yticks(range(5), trans.index)
        b.set_xlabel('new (ccf_atlas coordinates)'); b.set_ylabel('old (ccf coordinates)')
        b.set_title('b  Old -> new label', loc='left'); b.spines[:].set_visible(False)
        for j, (xa, ya, title) in enumerate([('ml', 'ap', 'c  Top view'), ('ap', 'dv', 'd  Side view')]):
            a = fig.add_subplot(gs[0, 2 + j])
            for sub in SUBS:
                x = s[s['new'] == sub]
                X = np.abs(x[xa]) / 1000 if xa == 'ml' else x[xa] / 1000
                a.scatter(X, x[ya] / 1000, s=0.3, color=COL[sub], lw=0, alpha=0.5, rasterized=True, label=sub)
            a.set_xlabel('ML (mm)' if xa == 'ml' else 'AP from bregma (mm)')
            a.set_ylabel('AP from bregma (mm)' if ya == 'ap' else 'DV from bregma (mm)')
            if ya == 'dv':
                a.invert_yaxis()
            a.set_title(title + ' (new labels)', loc='left')
            if j == 0:
                a.legend(frameon=False, fontsize=5, markerscale=8, loc='lower left')
        a = fig.add_subplot(gs[1, :])
        g = per_s[per_s['labels'] == 'new']
        rng = np.random.default_rng(0)
        for k, (ty, _) in enumerate(TYPES):
            for j, sub in enumerate(SUBS):
                x = g[(g.analysis_type == ty) & (g.subdivision == sub)]['frac_sig'].to_numpy()
                pos = k + (j - 1.5) * 0.2
                a.scatter(pos + rng.uniform(-0.05, 0.05, len(x)), x, s=3, color=COL[sub], alpha=0.5, lw=0, zorder=2)
                if len(x) > 1:
                    a.errorbar(pos, x.mean(), yerr=x.std(ddof=1) / np.sqrt(len(x)), fmt='o', ms=3.5, mfc=COL[sub], mec='k',
                               mew=0.4, ecolor='k', elinewidth=0.6, capsize=0, zorder=3,
                               label=f'{sub} ({len(x)} sessions)' if k == 0 else None)
        t = tests[tests['labels'] == 'new'].set_index('analysis_type').reindex([t[0] for t in TYPES])
        fp = lambda p: 'p<0.001' if p < 0.001 else f'p={p:.3f}'
        a.set_xticks(range(len(TYPES)), [f'{lab}\nKW {fp(t.loc[ty, "kw_p"])}\nANOVA {fp(t.loc[ty, "anova_p"])}' for ty, lab in TYPES])
        a.set_ylabel('% significant units per session (day 0)')
        a.legend(frameon=False, fontsize=5, ncol=4, loc='upper left')
        a.set_title(f'e  ROC response prevalence per subdivision (sessions with >= {MIN_UNITS} units; mean +- s.e.m.)', loc='left')
        for e in ('png', 'svg', 'pdf'):
            fig.savefig(f'{OUT}/figures/striatum_split.{e}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    # area_summary (user's plot_area_proj_brain) per subdivision + one 2 x 2 composite
    for p_ in [f'{A}/brain_wide_analysis', f'{A}/NWB_reader', f'{A}/Github/ephys_utilities',
               f'{A}/Github/ephys_utilities/ephys_utilities/allen_utils']:
        sys.path.insert(0, p_)
    import plotting_functions as pf
    pngs = []
    for sub in SUBS:
        png = f'{OUT}/area_summary/striatal_subdivision/{sub}/neurons_proj_striatal_subdivision_{sub}_by-pooled.png'
        if '--replot' in sys.argv or not os.path.exists(png):
            pf.plot_area_proj_brain(s.rename(columns={'new': 'striatal_subdivision'}), area_name=sub,
                                    area_level='striatal_subdivision', color_by=None, output_folder=f'{OUT}/area_summary')
        pngs.append(png)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(4, 1, figsize=(18 / 2.54, 4 * 6.4 / 2.54))
        for ax, sub, png in zip(axs, SUBS, pngs):
            ax.imshow(plt.imread(png)); ax.axis('off'); ax.set_title(sub, loc='left', fontsize=8, color=COL[sub])
        fig.tight_layout()
        fig.savefig(f'{OUT}/figures/area_summary_subdivisions.png', dpi=250, bbox_inches='tight')
        plt.close(fig)

    json.dump(dict(script=os.path.abspath(__file__), run=f'{datetime.now():%Y-%m-%d %H:%M}', host=os.uname().nodename,
                   ephys_utilities=subprocess.run(['git', '-c', 'safe.directory=*', '-C', f'{A}/Github/ephys_utilities', 'log', '-1', '--format=%h %s'],
                                                  capture_output=True, text=True).stdout.strip(),
                   roc=f'{R}/roc_long.parquet', unit_info=f'{R}/unit_info_all_days_v2.parquet', min_units=MIN_UNITS,
                   n_units=len(s), n_sessions=int(s['session_id'].nunique()), n_mice=int(s['mouse_id'].nunique())),
              open(f'{OUT}/provenance.json', 'w'), indent=1)
    print('done', len(s), 'units')


if __name__ == '__main__':
    main()
