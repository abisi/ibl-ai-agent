"""007 -- Drift at the junction per probe: raw detected peaks (DREDge peak_locations, before any correction), the DREDge
displacement estimate (applied to the recording before Kilosort via InterpolateMotionRecording), Kilosort 4's own rigid
shift (ops.npy dshift, nblocks = 1, on top), and the sorted spike depths (spike_positions, after both corrections).

Outputs (<joined>/concat_figure/): 007_drift_junction_imec<p>.png (full run + zoom +-300 s), 007_drift_summary.csv
(per probe: median raw peak depth shift of the dense bands, DREDge displacement step and KS dshift step at the junction).
Run (haas): python 007_drift_junction.py [probe ...]
"""
import glob
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
RUN = f'{J}/Ephys/catgt_AB155_g0'
OUT = f'{J}/concat_figure'
ZOOM = 300.0


def meta(path):
    return {k.lstrip('~'): v for k, v in (l.rstrip('\n').split('=', 1) for l in open(path) if '=' in l)}


def step(t, x, tj, w=60.0):
    return np.nanmedian(x[(t > tj) & (t < tj + w)], axis=0) - np.nanmedian(x[(t > tj - w) & (t < tj)], axis=0)


def main():
    off = {l.split()[0].rstrip(':'): l.split()[1:] for l in open(f'{RUN}/AB155_g0_sc_offsets.txt')}
    probes = [int(a) for a in sys.argv[1:]] or range(5)
    rows = []
    for p in probes:
        d = f'{RUN}/AB155_g0_imec{p}'
        fs = float(meta(glob.glob(f'{d}/*.ap.meta')[0])['imSampRate'])
        tj = float(off[f'smp_imap{p}'][1]) / fs
        m = f'{d}/dredge/motion'
        pk = np.load(f'{m}/peaks.npy')
        loc = np.load(f'{m}/peak_locations.npy')
        pt = pk['sample_index'] / fs
        py = loc['y']
        pa = np.abs(pk['amplitude'])
        disp = np.load(f'{m}/motion/displacement_seg0.npy')
        tb = np.load(f'{m}/motion/temporal_bins_s_seg0.npy')
        sb = np.load(f'{m}/motion/spatial_bins_um.npy')
        k = f'{d}/kilosort4/sorter_output'
        ops = np.load(f'{k}/ops.npy', allow_pickle=True).item()
        dshift = np.asarray(ops.get('dshift')).ravel()
        tbk = (np.arange(len(dshift)) + 0.5) * ops.get('batch_size', 60000) / fs
        st = np.load(f'{k}/spike_times.npy').ravel() / fs
        sy = np.load(f'{k}/spike_positions.npy')[:, 1]

        fig, axs = plt.subplots(3, 2, figsize=(18, 13), gridspec_kw=dict(width_ratios=[2, 1]), sharey='row')
        rng = np.random.default_rng(0)
        for c, (lo, hi) in enumerate(((0, pt.max()), (tj - ZOOM, tj + ZOOM))):
            sel = (pt > lo) & (pt < hi) & (pa > np.percentile(pa, 50))
            idx = np.flatnonzero(sel)
            idx = rng.choice(idx, min(len(idx), 400000), replace=False)
            ax = axs[0, c]
            ax.scatter(pt[idx], py[idx], s=0.3, c=np.log(pa[idx]), cmap='Greys', alpha=0.4, lw=0, rasterized=True)
            for j in range(0, len(sb), max(1, len(sb) // 8)):
                ax.plot(tb, sb[j] + disp[:, j], color='C3', lw=0.8)
            ax.set_title('raw detected peaks (before correction), amplitude > median; red = DREDge displacement '
                         'per spatial bin (applied before sorting)' if c == 0 else f'zoom +-{ZOOM:.0f} s', fontsize=8)
            ax.set_ylabel('depth (um)')
            ax = axs[1, c]
            for j in range(len(sb)):
                ax.plot(tb, disp[:, j], lw=0.6, color=plt.cm.viridis(j / max(1, len(sb) - 1)))
            ax.plot(tbk, dshift, 'k', lw=1, label='KS4 rigid dshift (after DREDge)')
            ax.set_ylabel('displacement (um)')
            ax.set_title('DREDge displacement per spatial bin (colour = depth) and KS4 dshift', fontsize=8)
            if c == 0:
                ax.legend(fontsize=7)
            ax = axs[2, c]
            sel = (st > lo) & (st < hi)
            idx = np.flatnonzero(sel)
            idx = rng.choice(idx, min(len(idx), 400000), replace=False)
            ax.scatter(st[idx], sy[idx], s=0.3, color='k', alpha=0.3, lw=0, rasterized=True)
            ax.set_ylabel('KS spike position y (um)')
            ax.set_title('sorted spike positions (after DREDge + KS4 correction)', fontsize=8)
            ax.set_xlabel('joined time (s)')
            for r in range(3):
                axs[r, c].axvline(tj, color='c', lw=1)
                axs[r, c].set_xlim(lo, hi)
        fig.suptitle(f'AB155 day 0 joined, imec{p}: drift around the junction ({tj:.1f} s, cyan)', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f'{OUT}/007_drift_junction_imec{p}.png', dpi=100)
        plt.close(fig)
        ds = step(tb, disp, tj)
        rows.append(dict(probe=p, junction_s=tj, dredge_step_median_um=float(np.median(ds)),
                         dredge_step_min_um=float(ds.min()), dredge_step_max_um=float(ds.max()),
                         ks_dshift_step_um=float(step(tbk, dshift, tj)),
                         dredge_range_partA_um=float(np.ptp(np.median(disp[tb < tj], 1))),
                         dredge_bin_s=float(np.median(np.diff(tb)))))
        print(rows[-1], flush=True)
    s = pd.DataFrame(rows)
    s.to_csv(f'{OUT}/007_drift_summary.csv', index=False)
    print(s.round(2).to_string(index=False))


if __name__ == '__main__':
    main()
