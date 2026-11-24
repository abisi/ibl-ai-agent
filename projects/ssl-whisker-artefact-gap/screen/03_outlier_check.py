"""Are the flagged probes outliers in the current day-0 results? (read-only, 2026-11-23)

Sources (existing tables, nothing recomputed):
  ROC   ssl-roc-learning-stages/roc_long.parquet (sensory ROC: pre -50..-5 ms vs post +5..+50 ms, i.e. mostly inside the
        5-35 ms dead time) -> whisker_active, auditory_active (control), whisker_passive_pre
  ZETA  ssl-zeta-responses/zeta_long.parquet -> significant, direction, latency_s for the same analysis types
  LAT   ssl-sensory-spatial-maps/unit_latency.parquet -> latency_whisker_ms, latency_auditory_ms
Units: day 0 from ssl-roc-learning-stages/units.parquet (area_group, reward_group). Probe = electrode_group prefix (imecN).
Per probe and per (session, area_group): fraction of units with a positive / negative significant response; each flagged
cell is ranked against all other day-0 cells (same area_group for the area table): percentile and robust z.
Outputs: screen/tables/03_outlier_probe.csv, 03_outlier_area.csv, screen/figures/03_outlier_check.png
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4'
OUT = f'{R}/ssl-whisker-artefact-gap/screen'
FL = {('MH037_20250524_143522', p) for p in ('imec0', 'imec1', 'imec2', 'imec3')} | {('MH020_20250304_161732', 'imec0')}
KEY = ['mouse_id', 'session_id', 'electrode_group', 'cluster_id']
TYPES = ['whisker_active', 'auditory_active', 'whisker_passive_pre']
MET = dict(roc_pos='mean', roc_neg='mean', zeta_up='mean', zeta_down='mean', zeta_lat_ms='median', lat_ms='median')


def rank(df, grp):
    out = []
    for _, g in df.groupby(grp):
        g = g.copy()
        for m in MET:
            ref = g.loc[~g.flag, m].dropna()
            mad = 1.4826 * np.median(np.abs(ref - ref.median())) if len(ref) else np.nan
            g[f'{m}_pct'] = [100 * (ref < v).mean() + 50 * (ref == v).mean() if len(ref) and pd.notna(v) else np.nan for v in g[m]]
            g[f'{m}_rz'] = (g[m] - ref.median()) / mad if mad and mad > 0 else np.nan
        out.append(g)
    return pd.concat(out)


def main():
    u = pd.read_parquet(f'{R}/ssl-roc-learning-stages/units.parquet', columns=KEY + ['day', 'reward_group', 'area_group'])
    u = u[u.day == 0].drop_duplicates(KEY)
    u['probe'] = u.electrode_group.str.split('_').str[0]
    roc = pq.read_table(f'{R}/ssl-roc-learning-stages/roc_long.parquet', columns=KEY + ['analysis_type', 'pos', 'neg'],
                        filters=[('analysis_type', 'in', TYPES)]).to_pandas().drop_duplicates(KEY + ['analysis_type'])
    z = pq.read_table(f'{R}/ssl-zeta-responses/zeta_long.parquet',
                      columns=KEY + ['analysis_type', 'significant', 'direction', 'latency_s'],
                      filters=[('analysis_type', 'in', TYPES)]).to_pandas().drop_duplicates(KEY + ['analysis_type'])
    lat = pd.read_parquet(f'{R}/ssl-sensory-spatial-maps/unit_latency.parquet',
                          columns=KEY + ['latency_whisker_ms', 'latency_auditory_ms']).drop_duplicates(KEY)
    print('zeta direction values', z.direction.unique()[:10], 'significant', z.significant.unique()[:5])
    rows = []
    for t in TYPES:
        a = roc[roc.analysis_type == t].merge(u, on=KEY)
        a = a.assign(roc_pos=a.pos.astype(float), roc_neg=a.neg.astype(float))[KEY + ['probe', 'area_group', 'reward_group', 'roc_pos', 'roc_neg']]
        b = z[z.analysis_type == t].copy()
        sig = b.significant.astype(bool)
        dirn = pd.to_numeric(b.direction, errors='coerce') if b.direction.dtype != object else b.direction.map({'positive': 1, 'negative': -1})
        b['zeta_up'] = (sig & (dirn > 0)).astype(float)
        b['zeta_down'] = (sig & (dirn < 0)).astype(float)
        b['zeta_lat_ms'] = np.where(sig, b.latency_s * 1e3, np.nan)
        a = a.merge(b[KEY + ['zeta_up', 'zeta_down', 'zeta_lat_ms']], on=KEY, how='left')
        if t != 'whisker_passive_pre':
            m = 'whisker' if t.startswith('whisker') else 'auditory'
            a = a.merge(lat[KEY + [f'latency_{m}_ms']].rename(columns={f'latency_{m}_ms': 'lat_ms'}), on=KEY, how='left')
        else:
            a['lat_ms'] = np.nan
        a['analysis_type'] = t
        rows.append(a)
    d = pd.concat(rows)
    d['flag'] = [(s, p) in FL for s, p in zip(d.session_id, d.probe)]
    agg = {k: (k, v) for k, v in MET.items()}
    pr = d.groupby(['analysis_type', 'session_id', 'probe', 'flag']).agg(n=('roc_pos', 'size'), **agg).reset_index()
    pr = rank(pr[pr.n >= 10], ['analysis_type'])
    ar = d.groupby(['analysis_type', 'area_group', 'session_id', 'flag']).agg(n=('roc_pos', 'size'), **agg).reset_index()
    ar = rank(ar[ar.n >= 5], ['analysis_type', 'area_group'])
    pr.to_csv(f'{OUT}/tables/03_outlier_probe.csv', index=False)
    ar.to_csv(f'{OUT}/tables/03_outlier_area.csv', index=False)
    pd.set_option('display.width', 250)
    c = ['analysis_type', 'session_id', 'probe', 'n', 'roc_pos', 'roc_pos_pct', 'roc_neg', 'roc_neg_pct', 'zeta_up', 'zeta_up_pct',
         'zeta_down_pct', 'lat_ms', 'lat_ms_pct', 'zeta_lat_ms', 'zeta_lat_ms_pct']
    print(pr[pr.flag][c].round(2).to_string(index=False))
    print('reference day-0 probes per type:', pr[~pr.flag].groupby('analysis_type').size().to_dict())
    print('reference medians:'); print(pr[~pr.flag].groupby('analysis_type')[list(MET)].median().round(3).to_string())
    fa = ar[ar.flag]
    print(fa[['analysis_type', 'area_group', 'session_id', 'n', 'roc_pos', 'roc_pos_pct', 'roc_neg', 'roc_neg_pct', 'zeta_up_pct',
              'lat_ms', 'lat_ms_pct']].round(2).to_string(index=False))
    print('flagged area cells, median percentile per type:')
    print(fa.groupby('analysis_type')[['roc_pos_pct', 'roc_neg_pct', 'zeta_up_pct', 'zeta_down_pct', 'lat_ms_pct']].median().round(0).to_string())
    fig, axs = plt.subplots(1, 5, figsize=(22, 4.5))
    for ax, (m, lab) in zip(axs, (('roc_pos', 'ROC +5..+50 ms: frac excited'), ('roc_neg', 'ROC: frac suppressed'),
                                  ('zeta_up', 'ZETA: frac up'), ('zeta_down', 'ZETA: frac down'), ('lat_ms', 'median latency (ms)'))):
        for i, t in enumerate(TYPES):
            g = pr[pr.analysis_type == t]
            x = i + np.random.default_rng(0).uniform(-0.2, 0.2, int((~g.flag).sum()))
            ax.scatter(x, g.loc[~g.flag, m], s=6, color='0.6')
            for _, r_ in g[g.flag].iterrows():
                ax.scatter(i, r_[m], s=40, color='#d62728' if r_.session_id.startswith('MH037') else '#ff7f0e', zorder=3,
                           edgecolor='k', lw=0.4)
        ax.set_xticks(range(len(TYPES))); ax.set_xticklabels(['whisker\nactive', 'auditory\nactive', 'whisker\npassive pre'], fontsize=8)
        ax.set_title(lab, fontsize=10)
    axs[0].scatter([], [], color='#d62728', label='MH037 imec0-3'); axs[0].scatter([], [], color='#ff7f0e', label='MH020 imec0')
    axs[0].scatter([], [], color='0.6', s=6, label='other day-0 probes'); axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle('Flagged probes vs all other day-0 probes (>= 10 units): existing ROC, ZETA and latency results', fontsize=11)
    fig.savefig(f'{OUT}/figures/03_outlier_check.png', dpi=130, bbox_inches='tight')


if __name__ == '__main__':
    main()
