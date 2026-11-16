"""003 -- Per-unit junction check (concat_figure) of the joined AB155 day-0 sorting (Kilosort 4 on the supercat run):
are units tracked across the junction, or lost / interrupted? Read from the light Kilosort outputs copied to
SessionToConcat (spike_times, spike_clusters, amplitudes, spike_positions, cluster_KSLabel).

Junction per probe: run-1 start from AB155_g0_sc_offsets.txt (smp_imap<p>, samples). Windows W = 300 s before /
after (part 2 keeps ~408 s). Controls in part 1: 1,500 s (quiet) and 2,000 s (disturbance seen in the motion
estimates and multi-unit activity of every probe). Per unit and event: firing rate, median amplitude and median depth
(spike_positions y) before / after. Lost: rate before >= 1 Hz and after < 10% of before; new: the reverse.

Outputs (<joined>/concat_figure/): 003_units_imec<p>.png (rates / amplitudes / depth before vs after for the junction
and the controls; firing-rate map of every unit over the joined run, sorted by depth; raster of good units +-120 s),
003_units_summary.csv, 003_units_per_unit_imec<p>.csv.
Labels: Bombcell (cluster_bc_unitType.tsv, noise units left out) when present, else Kilosort KSLabel.
Run (haas): python 003_unit_junction.py [probe ...]
"""
import glob
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
RUN = f'{J}/Ephys/catgt_AB155_g0'
OUT = f'{J}/concat_figure'
W = 300.0
EVENTS = {'junction': None, 'control 1,500 s': 1500.0, 'control 2,000 s': 2000.0}


def offsets():
    d = {}
    for line in open(f'{RUN}/AB155_g0_sc_offsets.txt'):
        k, *v = line.split()
        d[k.rstrip(':')] = [float(x) for x in v]
    return d


def meta(path):
    return {k.lstrip('~'): v for k, v in (l.rstrip('\n').split('=', 1) for l in open(path) if '=' in l)}


def load(p):
    d = f'{RUN}/AB155_g0_imec{p}'
    k = f'{d}/kilosort4/sorter_output'
    fs = float(meta(glob.glob(f'{d}/*.ap.meta')[0])['imSampRate'])
    st = np.load(f'{k}/spike_times.npy').astype(np.int64).ravel() / fs
    sc = np.load(f'{k}/spike_clusters.npy').ravel()
    amp = np.load(f'{k}/amplitudes.npy').ravel()
    pos = np.load(f'{k}/spike_positions.npy')
    bc = f'{k}/cluster_bc_unitType.tsv'
    if os.path.exists(bc):  # Bombcell labels (as the NWB converter): GOOD / MUA / NOISE; noise units left out
        lab = pd.read_csv(bc, sep='\t').set_index('cluster_id')['bc_unitType'].str.lower()
        lab = lab[lab != 'noise']
    else:
        lab = pd.read_csv(f'{k}/cluster_KSLabel.tsv', sep='\t').set_index('cluster_id')['KSLabel']
    return st, sc, amp, pos[:, 1], lab, fs


def unit_table(st, sc, amp, y, lab, T):
    rows = []
    pre_m = (st >= T - W) & (st < T)
    post_m = (st >= T) & (st < T + W)
    for c in lab.index:
        m = sc == c
        a, b = m & pre_m, m & post_m
        na, nb = a.sum(), b.sum()
        rows.append(dict(cluster_id=c, label=lab[c], rate_before=na / W, rate_after=nb / W,
                         amp_before=np.median(amp[a]) if na > 20 else np.nan,
                         amp_after=np.median(amp[b]) if nb > 20 else np.nan,
                         depth_before=np.median(y[a]) if na > 20 else np.nan,
                         depth_after=np.median(y[b]) if nb > 20 else np.nan,
                         depth_all=np.median(y[m]) if m.any() else np.nan, n_spikes=int(m.sum())))
    t = pd.DataFrame(rows)
    t['lost'] = (t.rate_before >= 1) & (t.rate_after < 0.1 * t.rate_before)
    t['new'] = (t.rate_after >= 1) & (t.rate_before < 0.1 * t.rate_after)
    return t


def summary(t, p, name, T):
    both = t[(t.rate_before >= 0.5) & (t.rate_after >= 0.5)]
    out = {}
    for lbl, s in (('all', t), ('good', t[t.label == 'good'])):
        b = both[both.label == 'good'] if lbl == 'good' else both
        active = s[(s.rate_before >= 1) | (s.rate_after >= 1)]
        out.update({f'{lbl}_active': len(active), f'{lbl}_lost': int(s.lost.sum()), f'{lbl}_new': int(s.new.sum()),
                    f'{lbl}_median_abs_log2_rate': float(np.median(np.abs(np.log2(b.rate_after / b.rate_before))))
                    if len(b) else np.nan,
                    f'{lbl}_median_amp_ratio': float(np.nanmedian(b.amp_after / b.amp_before)) if len(b) else np.nan,
                    f'{lbl}_median_abs_depth_shift_um': float(np.nanmedian(np.abs(b.depth_after - b.depth_before)))
                    if len(b) else np.nan})
    return dict(probe=p, event=name, time_s=T, **out)


def figure(p, st, sc, amp, y, lab, tables, tj):
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.6])
    for c, (name, (T, t)) in enumerate(tables.items()):
        col = np.where(t.label == 'good', 'C3' if name == 'junction' else 'k', '0.65')
        ax = fig.add_subplot(gs[0, c])
        ax.loglog(t.rate_before + 0.01, t.rate_after + 0.01, '.', ms=3, color='0.6')
        g = t[t.label == 'good']
        ax.loglog(g.rate_before + 0.01, g.rate_after + 0.01, '.', ms=4, color='C3' if name == 'junction' else 'k')
        ax.plot([0.01, 100], [0.01, 100], 'k', lw=0.5)
        ax.set_title(f'{name} ({T:.0f} s): firing rate (Hz), {W:.0f} s before vs after\nlost {int(t.lost.sum())} '
                     f'(good {int(g.lost.sum())}), new {int(t.new.sum())} (good {int(g.new.sum())}); '
                     f'grey = all, colour = good', fontsize=8)
        ax.set_xlabel('before')
        ax.set_ylabel('after')
        ax = fig.add_subplot(gs[1, c])
        b = t[(t.rate_before >= 0.5) & (t.rate_after >= 0.5)]
        r = (b.amp_after / b.amp_before).dropna()
        ax.hist(np.clip(r, 0.5, 1.5), bins=np.linspace(0.5, 1.5, 41), color='C3' if name == 'junction' else '0.4')
        ax.axvline(1, color='k', lw=0.5)
        ax.set_title(f'amplitude after / before (units with >= 0.5 Hz both sides, n = {len(r)}), median '
                     f'{np.median(r):.3f}', fontsize=8)
        ax = fig.add_subplot(gs[2, c])
        dd = (b.depth_after - b.depth_before).dropna()
        ax.hist(np.clip(dd, -60, 60), bins=np.linspace(-60, 60, 49), color='C3' if name == 'junction' else '0.4')
        ax.axvline(0, color='k', lw=0.5)
        ax.set_title(f'depth after - before (um), median {np.median(dd):.1f}, median |shift| '
                     f'{np.median(np.abs(dd)):.1f}', fontsize=8)
    t = tables['junction'][1]
    order = t.sort_values('depth_all').cluster_id.to_numpy()
    edges = np.arange(0, st.max() + 10, 10)
    mat = np.zeros((len(order), len(edges) - 1))
    idx = {c: i for i, c in enumerate(order)}
    rows_ = np.array([idx.get(c, -1) for c in sc])
    ok = rows_ >= 0
    np.add.at(mat, (rows_[ok], np.clip(np.digitize(st[ok], edges) - 1, 0, len(edges) - 2)), 1)
    mat = mat / 10
    norm = mat / (np.percentile(mat, 95, axis=1, keepdims=True) + 1e-9)
    ax = fig.add_subplot(gs[3, :2])
    ax.imshow(np.clip(norm, 0, 1), aspect='auto', cmap='magma', interpolation='nearest',
              extent=[0, edges[-1], len(order), 0])
    ax.axvline(tj, color='c', lw=1)
    for T in (1500, 2000):
        ax.axvline(T, color='w', lw=0.5, ls='--')
    ax.set_xlabel('joined time (s); cyan = junction')
    ax.set_ylabel('units sorted by depth')
    ax.set_title(f'firing rate of every unit ({len(order)}) over the joined run, 10-s bins, each unit scaled to its '
                 f'95th percentile', fontsize=8)
    ax = fig.add_subplot(gs[3, 2])
    good = t[t.label == 'good'].sort_values('depth_all').cluster_id.to_numpy()
    for i, c in enumerate(good):
        s = st[(sc == c) & (st > tj - 120) & (st < tj + 120)] - tj
        ax.plot(s, np.full(len(s), i), '|', ms=2, color='k', mew=0.4)
    ax.axvline(0, color='c', lw=1)
    ax.set_xlim(-120, 120)
    ax.set_xlabel('time from junction (s)')
    ax.set_ylabel('good units sorted by depth')
    ax.set_title(f'good units ({len(good)}): spikes +-120 s around the junction', fontsize=8)
    fig.suptitle(f'AB155 day 0 joined sorting, imec{p}: units across the junction ({tj:.3f} s) vs part-1 controls',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f'{OUT}/003_units_imec{p}.png', dpi=105)
    plt.close(fig)


def main():
    off = offsets()
    want = [int(a) for a in sys.argv[1:]] or [p for p in range(5)
                                               if os.path.exists(f'{RUN}/AB155_g0_imec{p}/kilosort4/sorter_output/spike_times.npy')]
    rows = []
    for p in want:
        st, sc, amp, y, lab, fs = load(p)
        tj = off[f'smp_imap{p}'][1] / fs
        tables = {}
        for name, T in EVENTS.items():
            T = tj if T is None else T
            tables[name] = (T, unit_table(st, sc, amp, y, lab, T))
            rows.append(summary(tables[name][1], p, name, T))
        tables['junction'][1].to_csv(f'{OUT}/003_units_per_unit_imec{p}.csv', index=False)
        figure(p, st, sc, amp, y, lab, tables, tj)
        print(f'imec{p}: {len(lab)} units ({int((lab == "good").sum())} good), junction {tj:.3f} s', flush=True)
    s = pd.DataFrame(rows)
    f = f'{OUT}/003_units_summary.csv'
    if os.path.exists(f):
        old = pd.read_csv(f)
        s = pd.concat([old[~old.probe.isin(want)], s]).sort_values(['probe', 'time_s'])
    s.to_csv(f, index=False)
    pd.set_option('display.width', 250)
    print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
