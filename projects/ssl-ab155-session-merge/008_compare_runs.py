"""008 -- Compare the two joined AB155 day-0 sortings per probe:
  A = run 4819418: DREDge interpolation before sorting + Kilosort 4 rigid drift correction (do_correction true)
  B = run 4825628: DREDge interpolation before sorting, Kilosort 4 drift correction off
Same CatGT supercat, same DREDge settings, same downstream steps (Bombcell, C_Waves, TPrime).

Per probe and run (non-noise units = Bombcell label != noise, as the NWB converter keeps them):
  - Bombcell labels and quality_label (ephys_utilities classify_units_quality, default thresholds; presence / coverage
    over the joined run, from TPrime-synced spike times)
  - Bombcell metrics (medians), Kilosort drift shift (ops dshift) range and junction step
  - junction: lost / new units (rate >= 1 Hz on one side, < 10% of it on the other, 300 s windows); quality good units
    kept across the junction (part-2 / part-1 rate 0.5-2), silent in part 2 (< 0.1)
  - unit matching A <-> B: spike coincidences within 0.5 ms for unit pairs within 100 um (depth); a unit is matched
    when >= 50% of its spikes coincide with one unit of the other run; a B unit holding >= 50% of the spikes of two or
    more A units is a merge of A units; it repairs a junction split when those include an A unit lost at the junction
    and an A unit new at the junction
Outputs (<joined>/concat_figure/): 008_compare_imec<p>.png, 008_compare_summary.csv, 008_matches_imec<p>.csv
Run (haas): python 008_compare_runs.py [probe ...]
"""
import glob
import importlib.util
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat'
RUNS = {'A: KS drift corr. on': f'{ROOT}/AB155_20250208_175103_joined/Ephys/catgt_AB155_g0',
        'B: KS drift corr. off': f'{ROOT}/AB155_20250208_175103_joined_noksdc/Ephys/catgt_AB155_g0'}
OUT = f'{ROOT}/AB155_20250208_175103_joined/concat_figure'
UMU = '/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities/ephys_utilities/neural_utils/unit_metrics_utils.py'
W = 300.0
TOL = 0.0005
BIN = 60.0
METRICS = ['nSpikes', 'percentageSpikesMissing_gaussian', 'fractionRPVs_estimatedTauR', 'maxDriftEstimate',
           'presenceRatio', 'isolationDistance', 'Lratio', 'rawAmplitude', 'signalToNoiseRatio']

spec = importlib.util.spec_from_file_location('unit_metrics_utils', UMU)
umu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(umu)


def meta(path):
    return {k.lstrip('~'): v for k, v in (l.rstrip('\n').split('=', 1) for l in open(path) if '=' in l)}


def load_run(run, p, tj):
    d = f'{run}/AB155_g0_imec{p}'
    k = f'{d}/kilosort4/sorter_output'
    fs = float(meta(glob.glob(f'{d}/*.ap.meta')[0])['imSampRate'])
    st = np.load(f'{d}/AB155_g0_imec{p}_kilosort4_spike_times_sec_sync.npy').ravel()
    sc = np.load(f'{k}/spike_clusters.npy').ravel()
    y = np.load(f'{k}/spike_positions.npy')[:, 1]
    lab = pd.read_csv(f'{k}/cluster_bc_unitType.tsv', sep='\t').rename(columns={'bc_unitType': 'bc_label'})
    lab['bc_label'] = lab.bc_label.str.lower()
    bc = pd.read_parquet(f'{k}/bombcell/templates._bc_qMetrics.parquet').rename(columns={'phy_clusterID': 'cluster_id'})
    u = lab.merge(bc[['cluster_id'] + [m for m in METRICS if m in bc]], on='cluster_id', how='left')
    u = u[u.bc_label != 'noise'].reset_index(drop=True)
    order = np.argsort(sc, kind='stable')
    bounds = np.searchsorted(sc[order], u.cluster_id.to_numpy())
    ends = np.searchsorted(sc[order], u.cluster_id.to_numpy(), side='right')
    u['spike_times'] = [np.sort(st[order[a:b]]) for a, b in zip(bounds, ends)]
    u['depth'] = [np.median(y[order[a:b]]) if b > a else np.nan for a, b in zip(bounds, ends)]
    u['session_id'] = 'AB155_20250208_175103'
    u = umu.compute_presence_coverage_metrics(u)
    u = umu.classify_units_quality(u)
    t_end = float(st.max())
    u['r_pre'] = u.spike_times.map(lambda s: np.sum((s >= tj - W) & (s < tj)) / W)
    u['r_post'] = u.spike_times.map(lambda s: np.sum((s >= tj) & (s < tj + W)) / W)
    u['rate_p1'] = u.spike_times.map(lambda s: np.sum(s < tj) / tj)
    u['rate_p2'] = u.spike_times.map(lambda s: np.sum(s >= tj) / (t_end - tj))
    u['lost'] = (u.r_pre >= 1) & (u.r_post < 0.1 * u.r_pre)
    u['new'] = (u.r_post >= 1) & (u.r_pre < 0.1 * u.r_post)
    u['ratio_p2_p1'] = (u.rate_p2 + 0.01) / (u.rate_p1 + 0.01)
    ops = np.load(f'{k}/ops.npy', allow_pickle=True).item()
    ds = np.asarray(ops.get('dshift') if ops.get('dshift') is not None else []).ravel()
    return u, ds, fs, t_end


def coincide(a, b):
    """number of spikes of a with a spike of b within TOL"""
    if not len(a) or not len(b):
        return 0
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return int(np.sum(np.minimum(np.abs(a - b[i - 1]), np.abs(a - b[i])) <= TOL))


def match(ua, ub):
    rows = []
    for i, ra in ua.iterrows():
        near = ub[np.abs(ub.depth - ra.depth) <= 100]
        for j, rb in near.iterrows():
            c = coincide(ra.spike_times, rb.spike_times)
            if c:
                rows.append(dict(a=ra.cluster_id, b=rb.cluster_id, n_a=len(ra.spike_times), n_b=len(rb.spike_times),
                                 coinc=c, frac_a=c / len(ra.spike_times), frac_b=c / len(rb.spike_times)))
    return pd.DataFrame(rows, columns=['a', 'b', 'n_a', 'n_b', 'coinc', 'frac_a', 'frac_b'])


def stable_counts(u, t_end):
    edges = np.arange(0, t_end + BIN, BIN)
    c = np.stack([np.histogram(s, edges)[0] for s in u.spike_times]) / BIN if len(u) else np.zeros((0, len(edges) - 1))
    med = np.median(c, axis=1, keepdims=True)
    return edges[:-1] + BIN / 2, ((c >= 0.25 * med) & (med > 0))


def main():
    off = {l.split()[0].rstrip(':'): l.split()[1:] for l in open(f'{RUNS["A: KS drift corr. on"]}/AB155_g0_sc_offsets.txt')}
    want = [int(a) for a in sys.argv[1:]] or range(5)
    summ = []
    for p in want:
        res = {}
        for name, run in RUNS.items():
            d = f'{run}/AB155_g0_imec{p}'
            if not os.path.exists(f'{d}/kilosort4/sorter_output/cluster_bc_unitType.tsv'):
                print(f'imec{p}: {name} missing, skipped')
                break
            fs = float(meta(glob.glob(f'{d}/*.ap.meta')[0])['imSampRate'])
            tj = float(off[f'smp_imap{p}'][1]) / fs
            res[name] = load_run(run, p, tj) + (tj,)
        if len(res) < 2:
            continue
        (na, (ua, dsa, _, t_end, tj)), (nb, (ub, dsb, _, _, _)) = res.items()
        m = match(ua, ub)
        best_a = m.sort_values('frac_a').groupby('a').tail(1)
        best_b = m.sort_values('frac_b').groupby('b').tail(1)
        matched_a = set(best_a[best_a.frac_a >= 0.5].a)
        matched_b = set(best_b[best_b.frac_b >= 0.5].b)
        merges = m[m.frac_a >= 0.5].groupby('b').a.apply(list)
        merges = merges[merges.map(len) >= 2]
        lost_a, new_a = set(ua[ua.lost].cluster_id), set(ua[ua.new].cluster_id)
        rep = {b: a for b, a in merges.items() if lost_a.intersection(a) and new_a.intersection(a)}
        m.to_csv(f'{OUT}/008_matches_imec{p}.csv', index=False)
        for name, u, ds in ((na, ua, dsa), (nb, ub, dsb)):
            g = u[u.quality_label == 'good']
            summ.append(dict(
                probe=p, run=name, n_nonnoise=len(u),
                **{f'bc_{k}': int((u.bc_label == k).sum()) for k in ('good', 'mua', 'non-soma')},
                **{f'q_{k}': int((u.quality_label == k).sum()) for k in ('good', 'mua', 'non-soma')},
                lost=int(u.lost.sum()), new=int(u.new.sum()),
                q_good_kept_0p5_2=int(((g.ratio_p2_p1 >= 0.5) & (g.ratio_p2_p1 <= 2)).sum()),
                q_good_silent_p2=int((g.ratio_p2_p1 < 0.1).sum()),
                matched_frac=len(matched_a) / len(ua) if name == na else len(matched_b) / len(ub),
                merges_of_A_units=len(merges) if name == nb else np.nan,
                merges_repairing_junction_loss=len(rep) if name == nb else np.nan,
                ks_dshift_range_um=float(np.ptp(ds)) if len(ds) else np.nan,
                **{f'med_{k}': float(np.nanmedian(u[k])) for k in METRICS if k in u}))
        print(f'imec{p}: A {len(ua)} units, B {len(ub)}; matched A {len(matched_a)}, B {len(matched_b)}; '
              f'B units merging >= 2 A units {len(merges)} ({len(rep)} join an A unit lost and an A unit new at the junction)',
              flush=True)
        for b, a in list(rep.items())[:15]:
            print(f'   B {b} <- A {a}')

        # figure
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 4)
        ax = fig.add_subplot(gs[0, 0])
        labs = ['good', 'mua', 'non-soma']
        x = np.arange(3)
        for k, (name, u) in enumerate(((na, ua), (nb, ub))):
            ax.bar(x + (k - 0.5) * 0.35, [(u.quality_label == l).sum() for l in labs], 0.35, label=name,
                   color=['C0', 'C1'][k])
            ax.bar(x + (k - 0.5) * 0.35, [(u.bc_label == l).sum() for l in labs], 0.35, fill=False, hatch='//',
                   edgecolor='k', lw=0.5)
        ax.set_xticks(x, labs)
        ax.set_title('units: quality_label (filled), Bombcell label (hatched)', fontsize=8)
        ax.legend(fontsize=7)
        for c, k in enumerate(['presenceRatio', 'maxDriftEstimate', 'fractionRPVs_estimatedTauR']):
            ax = fig.add_subplot(gs[0, 1 + c])
            data = [ua[k].dropna(), ub[k].dropna()]
            ax.boxplot(data, tick_labels=['A', 'B'], showfliers=False)
            ax.set_title(f'{k} (non-noise); median A {np.median(data[0]):.3g}, B {np.median(data[1]):.3g}',
                         fontsize=8)
        ax = fig.add_subplot(gs[1, 0])
        for name, ds, col in ((na, dsa, 'C0'), (nb, dsb, 'C1')):
            if len(ds):
                ax.plot((np.arange(len(ds)) + 0.5) * 2.0, ds, color=col, lw=0.8, label=name)
        ax.axvline(tj, color='c')
        ax.set_title('Kilosort drift shift (um); B has none when correction is off', fontsize=8)
        ax.legend(fontsize=7)
        ax = fig.add_subplot(gs[1, 1])
        for name, u, col in ((na, ua, 'C0'), (nb, ub, 'C1')):
            t, s = stable_counts(u[u.quality_label == 'good'], t_end)
            ax.step(t, s.sum(0), where='mid', color=col, label=f'{name}: quality good n = {(u.quality_label == "good").sum()}')
            t, s = stable_counts(u, t_end)
            ax.step(t, s.sum(0) / 5, where='mid', color=col, ls=':', label=f'{name}: all non-noise / 5')
        ax.axvline(tj, color='c')
        ax.set_title('units stable per 60-s bin (rate >= 25% of own median)', fontsize=8)
        ax.legend(fontsize=6)
        for c, (name, u) in enumerate(((na, ua), (nb, ub))):
            ax = fig.add_subplot(gs[1, 2 + c])
            ax.loglog(u.r_pre + 0.01, u.r_post + 0.01, '.', ms=3, color='0.6')
            g = u[u.quality_label == 'good']
            ax.loglog(g.r_pre + 0.01, g.r_post + 0.01, '.', ms=5, color='C3')
            ax.plot([0.01, 100], [0.01, 100], 'k', lw=0.5)
            ax.set_xlabel(f'rate {W:.0f} s before junction (Hz)')
            ax.set_ylabel('after (Hz)')
            ax.set_title(f'{name}: lost {int(u.lost.sum())}, new {int(u.new.sum())} (red = quality good)', fontsize=8)
        for c, (name, u) in enumerate(((na, ua), (nb, ub))):
            ax = fig.add_subplot(gs[2, 2 * c:2 * c + 2])
            g = u[u.quality_label == 'good'].sort_values('depth')
            for i, s in enumerate(g.spike_times):
                z = s[(s > tj - 120) & (s < tj + 120)] - tj
                ax.plot(z, np.full(len(z), i), '|', ms=2, mew=0.3, color='k', rasterized=True)
            ax.axvline(0, color='c')
            ax.set_xlim(-120, 120)
            ax.set_xlabel('time from junction (s)')
            ax.set_ylabel('quality good units by depth')
            ax.set_title(f'{name}: {len(g)} quality good units, +-120 s around the junction', fontsize=8)
        fig.suptitle(f'AB155 day 0 joined, imec{p}: run A (Kilosort drift correction on) vs run B (off); matched '
                     f'units A {len(matched_a)}/{len(ua)}, B {len(matched_b)}/{len(ub)}; B units merging A units: '
                     f'{len(merges)} ({len(rep)} repair a junction loss)', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f'{OUT}/008_compare_imec{p}.png', dpi=100)
        plt.close(fig)
    s = pd.DataFrame(summ)
    f = f'{OUT}/008_compare_summary.csv'
    if os.path.exists(f):
        old = pd.read_csv(f)
        s = pd.concat([old[~old.probe.isin(want)], s]).sort_values(['probe', 'run'])
    s.to_csv(f, index=False)
    pd.set_option('display.width', 250)
    print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
