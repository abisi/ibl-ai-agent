"""005 -- quality_label, spike rasters and stable-unit counts over the joined AB155 day-0 recording, read from the test NWB
(004). Labels: ephys_utilities unit_metrics_utils.compute_presence_coverage_metrics + classify_units_quality (default
thresholds and exclusions; no DREDge shift-test columns, so the joint drift check is skipped). presence_ratio /
coverage_ratio are computed over the joined run, so a unit lost or new at the junction fails coverage (>= 0.9).

Stable in a 60-s bin: unit rate in the bin >= 25% of the unit's own median bin rate over the run.

Outputs (<joined>/concat_figure/):
  005_raster_imec<p>.png        full-run raster of quality 'good' units sorted by depth (+ zoom +-60 s at the junction)
  005_stable_units.png          stable units per 60-s bin per probe and quality_label, junction marked
  005_quality_labels.csv        counts per probe x quality_label x bc_label
  005_units_quality.csv         per unit: probe, cluster_id, depth, bc_label, quality_label, presence/coverage, part rates
Run (haas): python 005_quality_units_nwb.py
"""
import importlib.util

import matplotlib
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
NWB = f'{J}/nwb_test/AB155_20250208_175103_joined_test.nwb'
OUT = f'{J}/concat_figure'
UMU = '/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities/ephys_utilities/neural_utils/unit_metrics_utils.py'
BIN = 60.0
STABLE_FRAC = 0.25
LABELS = ['good', 'mua', 'non-soma']
COLORS = {'good': 'C2', 'mua': 'C1', 'non-soma': 'C0'}
METRICS = ['nSpikes', 'percentageSpikesMissing_gaussian', 'fractionRPVs_estimatedTauR', 'maxDriftEstimate',
           'presenceRatio', 'isolationDistance', 'Lratio']

spec = importlib.util.spec_from_file_location('unit_metrics_utils', UMU)
umu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(umu)


def load():
    with NWBHDF5IO(NWB, 'r') as io:
        f = io.read()
        u = f.units.to_dataframe()
        ep = f.epochs.to_dataframe()
    u['probe'] = u['electrode_group'].map(lambda g: g.name.split('_')[0])
    u = u.drop(columns=['electrode_group', 'waveform_mean', 'waveform_mean_bc'], errors='ignore')
    for c in METRICS + ['depth', 'firing_rate']:
        u[c] = pd.to_numeric(u[c], errors='coerce')
    u['session_id'] = 'AB155_20250208_175103'
    tj = float(ep.loc[ep.tags.map(lambda t: 'part2' in t), 'start_time'].iloc[0])
    t_end = float(ep.stop_time.max())
    return u.reset_index(drop=True), tj, t_end


def stable_matrix(u, t_end):
    edges = np.arange(0, t_end + BIN, BIN)
    counts = np.stack([np.histogram(s, edges)[0] for s in u.spike_times]) / BIN
    med = np.median(counts, axis=1, keepdims=True)
    return edges, (counts >= STABLE_FRAC * med) & (med > 0)


def main():
    u, tj, t_end = load()
    u = umu.compute_presence_coverage_metrics(u)
    u = umu.classify_units_quality(u)
    u['rate_part1'] = u.spike_times.map(lambda s: np.sum(s < tj) / tj)
    u['rate_part2'] = u.spike_times.map(lambda s: np.sum(s >= tj) / (t_end - tj))
    probes = sorted(u.probe.unique())

    tab = u.groupby(['probe', 'quality_label', 'bc_label']).size().rename('n_units').reset_index()
    tab.to_csv(f'{OUT}/005_quality_labels.csv', index=False)
    u.drop(columns=['spike_times']).loc[:, ['probe', 'cluster_id', 'depth', 'bc_label', 'quality_label',
                                             'presence_ratio', 'coverage_ratio', 'firing_rate', 'rate_part1',
                                             'rate_part2'] + METRICS].to_csv(f'{OUT}/005_units_quality.csv',
                                                                            index=False)
    print(pd.crosstab(u.probe, u.quality_label, margins=True).to_string())
    print(pd.crosstab([u.probe, u.bc_label], u.quality_label).to_string())

    # stable units over time
    fig, axs = plt.subplots(len(probes), 2, figsize=(16, 3.0 * len(probes)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[3, 1]))
    rows = []
    for i, p in enumerate(probes):
        up = u[u.probe == p]
        edges, st = stable_matrix(up, t_end)
        mid = edges[:-1] + BIN / 2
        for lab in LABELS:
            m = (up.quality_label == lab).to_numpy()
            if not m.any():
                continue
            n = st[m].sum(0)
            axs[i, 0].step(mid, n, where='mid', color=COLORS[lab], label=f'{lab} (n = {m.sum()})')
            pre = n[(mid > tj - 600) & (mid < tj)].mean()
            post = n[(mid > tj) & (mid < t_end - BIN)].mean()
            rows.append(dict(probe=p, quality_label=lab, n_units=int(m.sum()), stable_mean_last10min_part1=pre,
                             stable_mean_part2=post))
        axs[i, 0].axvline(tj, color='k', lw=1)
        axs[i, 0].set_ylabel(f'{p}\nstable units')
        axs[i, 0].legend(fontsize=7, loc='lower left', ncol=3)
        g = up[up.quality_label == 'good']
        axs[i, 1].hist(g.coverage_ratio, bins=np.linspace(0, 1, 21), color='C2', alpha=0.7, label='good')
        axs[i, 1].hist(up[up.quality_label == 'mua'].coverage_ratio, bins=np.linspace(0, 1, 21), color='C1',
                       alpha=0.5, label='mua')
        axs[i, 1].axvline(0.9, color='k', lw=0.5)
        axs[i, 1].set_title('coverage_ratio (good >= 0.9)', fontsize=8)
        axs[i, 1].legend(fontsize=7)
    axs[-1, 0].set_xlabel('joined time (s); black = junction')
    fig.suptitle(f'AB155 day 0 joined test NWB: units stable per {BIN:.0f}-s bin (rate >= {STABLE_FRAC:.0%} of own '
                 f'median), by quality_label', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f'{OUT}/005_stable_units.png', dpi=110)
    plt.close(fig)
    s = pd.DataFrame(rows)
    s.to_csv(f'{OUT}/005_stable_units_summary.csv', index=False)
    print(s.round(1).to_string(index=False))

    # rasters
    for p in probes:
        g = u[(u.probe == p) & (u.quality_label == 'good')].sort_values('depth')
        fig, axs = plt.subplots(1, 2, figsize=(18, max(4, 0.06 * len(g) + 2)), gridspec_kw=dict(width_ratios=[3, 1]))
        for k, s in enumerate(g.spike_times):
            axs[0].plot(s, np.full(len(s), k), '|', ms=1.5, mew=0.15, color='k', rasterized=True)
            z = s[(s > tj - 60) & (s < tj + 60)] - tj
            axs[1].plot(z, np.full(len(z), k), '|', ms=1.5, mew=0.3, color='k', rasterized=True)
        for ax in axs:
            ax.set_ylim(-1, len(g))
            ax.set_ylabel(f'quality good units (n = {len(g)}), sorted by depth')
        axs[0].axvline(tj, color='r', lw=1)
        axs[0].set_xlim(0, t_end)
        axs[0].set_xlabel('joined time (s); red = junction')
        axs[1].axvline(0, color='r', lw=1)
        axs[1].set_xlim(-60, 60)
        axs[1].set_xlabel('time from junction (s)')
        fig.suptitle(f'AB155 day 0 joined test NWB, {p}: quality_label good units', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f'{OUT}/005_raster_{p}.png', dpi=110)
        plt.close(fig)
        print(f'{p}: raster of {len(g)} good units', flush=True)


if __name__ == '__main__':
    main()
