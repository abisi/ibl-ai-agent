"""002 -- Junction check figures (concat_figure) before spike sorting finishes: TTL streams, DREDge motion and
multi-unit peaks (dredge peaks.npy: every detected spike peak, time, depth, amplitude) across the junction of the
joined AB155 day-0 run (supercat of g0 + g1; see question.md).

Junction (joined clock): run-1 start from AB155_g0_sc_offsets.txt (imec: per probe; NIDQ: 2,997.0869 s).
Controls: 2,000 s (a disturbance seen in the motion estimates of every probe, part 1) and 1,500 s (quiet part 1).
For each event time T, windows [T - W, T) and [T, T + W), W = 120 s.

Figures (written to <joined>/concat_figure/):
- 002_ttl_junction.png: per NIDQ channel, events +-60 s around the junction and inter-event intervals over the session;
  counts per part (joined vs the per-run CatGT g0 / g1 edge files).
- 002_motion_junction.png: DREDge displacement per probe (mean over depth bins and per depth bin), full session and
  +-300 s around the junction.
- 002_mua_junction_imec<p>.png: peak raster (depth vs time, +-300 s), peak rate per 40-um depth bin before vs after
  for the junction and the two controls, and median amplitude per bin before vs after.
Table: 002_mua_junction_summary.csv (per probe and event: correlation of the depth profiles of rate and amplitude
before vs after, median |log2 rate ratio|).
Run (haas): python 002_concat_figure_prelim.py
"""
import glob
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
RUN = f'{J}/Ephys/catgt_AB155_g0'
OUT = f'{J}/concat_figure'
A = '/mnt/lsens-analysis/Axel_Bisi/data/AB155'
G0 = f'{A}/AB155_20250208_175103/Ephys/catgt_AB155_g0'
G1 = f'{A}/AB155_20250208_185559/Ephys/catgt_AB155_g1'
CH = {0: 'sync', 1: 'trial start', 2: 'auditory', 3: 'whisker', 4: 'valve', 5: 'camera 0', 6: 'camera 1', 7: 'piezo'}
W = 120.0
CONTROLS = {'control 1,500 s': 1500.0, 'control 2,000 s (disturbance)': 2000.0}


def offsets():
    d = {}
    for line in open(f'{RUN}/AB155_g0_sc_offsets.txt'):
        k, *v = line.split()
        d[k.rstrip(':')] = [float(x) for x in v]
    return d


def meta(path):
    return {k.lstrip('~'): v for k, v in (l.rstrip('\n').split('=', 1) for l in open(path) if '=' in l)}


def edges(path):
    return np.loadtxt(path, ndmin=1) if os.path.exists(path) and os.path.getsize(path) else np.array([])


def ttl_figure(off):
    tj = off['sec_nidq'][1]
    fig, axs = plt.subplots(8, 2, figsize=(15, 17), gridspec_kw=dict(width_ratios=[1, 2]))
    rows = []
    for ch in range(8):
        e = edges(f'{RUN}/AB155_g0_tcat.nidq.xa_{ch}_0.txt')
        e0 = edges(f'{G0}/AB155_g0_tcat.nidq.xa_{ch}_0.txt')
        e1 = edges(f'{G1}/AB155_g1_tcat.nidq.xa_{ch}_0.txt')
        before, after = (e < tj).sum(), (e >= tj).sum()
        rows.append(dict(channel=ch, meaning=CH[ch], joined_part1=before, joined_part2=after, g0=len(e0), g1=len(e1),
                         lost_part1=len(e0) - before, lost_part2=len(e1) - after))
        ax = axs[ch, 0]
        w = e[(e > tj - 60) & (e < tj + 60)] - tj
        if ch in (5, 6, 0):
            ax.plot(w[1:], np.diff(w) * 1000, '.', ms=1.5)
            ax.set_ylabel('interval (ms)', fontsize=7)
        else:
            ax.eventplot(w, lineoffsets=0.5, linelengths=0.8, linewidths=0.6)
            ax.set_yticks([])
        ax.axvline(0, color='r', lw=0.8)
        ax.set_xlim(-60, 60)
        ax.set_title(f'ch{ch} {CH[ch]}: +-60 s around the junction', fontsize=8)
        ax = axs[ch, 1]
        if len(e) > 1:
            ax.plot(e[1:], np.diff(e), '.', ms=1, color='k')
            ax.set_yscale('log')
        ax.axvline(tj, color='r', lw=0.8)
        for c in CONTROLS.values():
            ax.axvline(c, color='0.6', lw=0.5, ls='--')
        ax.set_title(f'ch{ch} {CH[ch]}: inter-event interval (s) over the joined run; part 1 {before} + part 2 {after} '
                     f'edges (g0 {len(e0)}, g1 {len(e1)})', fontsize=8)
        ax.tick_params(labelsize=7)
    axs[-1, 0].set_xlabel('time from junction (s)')
    axs[-1, 1].set_xlabel('joined time (s); red = junction, grey = controls')
    fig.suptitle(f'AB155 day 0 joined run: NIDQ TTL streams across the junction ({tj:.3f} s)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f'{OUT}/002_ttl_junction.png', dpi=110)
    plt.close(fig)
    return pd.DataFrame(rows)


def probes():
    out = []
    for d in sorted(glob.glob(f'{RUN}/AB155_g0_imec*')):
        p = int(re.search(r'imec(\d+)$', d).group(1))
        if os.path.exists(f'{d}/dredge/motion/peaks.npy'):
            out.append((p, d))
    return out


def motion_figure(off, plist):
    fig, axs = plt.subplots(len(plist), 3, figsize=(17, 3.2 * len(plist)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[2, 1, 2]))
    for r, (p, d) in enumerate(plist):
        m = f'{d}/dredge/motion/motion'
        disp = np.load(f'{m}/displacement_seg0.npy')
        tb = np.load(f'{m}/temporal_bins_s_seg0.npy')
        sb = np.load(f'{m}/spatial_bins_um.npy')
        tj = off[f'sec_imap{p}'][1]
        ax = axs[r, 0]
        ax.plot(tb, disp, color='0.7', lw=0.4)
        ax.plot(tb, disp.mean(1), color='C0', lw=1)
        ax.axvline(tj, color='r', lw=0.8)
        for c in CONTROLS.values():
            ax.axvline(c, color='0.4', lw=0.5, ls='--')
        ax.set_title(f'imec{p}: DREDge displacement (um), grey = depth bins, blue = mean', fontsize=8)
        ax = axs[r, 1]
        k = (tb > tj - 300) & (tb < tj + 300)
        ax.plot(tb[k] - tj, disp[k], color='0.7', lw=0.4)
        ax.plot(tb[k] - tj, disp[k].mean(1), color='C0', lw=1)
        ax.axvline(0, color='r', lw=0.8)
        ax.set_title('+-300 s around the junction', fontsize=8)
        ax = axs[r, 2]
        lim = np.nanpercentile(np.abs(disp), 99)
        ax.imshow(disp.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                  extent=[tb[0], tb[-1], sb[0], sb[-1]])
        ax.axvline(tj, color='k', lw=0.8)
        ax.set_ylabel('depth (um)', fontsize=7)
        ax.set_title(f'displacement per depth (+-{lim:.0f} um)', fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel('time (s)', fontsize=8)
    fig.suptitle('AB155 day 0 joined run: probe motion (DREDge) across the junction (red)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f'{OUT}/002_motion_junction.png', dpi=110)
    plt.close(fig)


def mua_figure(off, p, d):
    ap = meta(glob.glob(f'{d}/*.ap.meta')[0])
    fs = float(ap['imSampRate'])
    pk = np.load(f'{d}/dredge/motion/peaks.npy')
    loc = np.load(f'{d}/dredge/motion/peak_locations.npy')
    t = pk['sample_index'] / fs
    y = loc['y']
    amp = np.abs(pk['amplitude'])
    tj = off[f'sec_imap{p}'][1]
    events = {'junction': tj, **CONTROLS}
    bins = np.arange(np.floor(y.min() / 40) * 40, y.max() + 40, 40)
    fig, axs = plt.subplots(2, 4, figsize=(19, 9), gridspec_kw=dict(width_ratios=[2, 1, 1, 1]))
    ax = axs[0, 0]
    k = np.flatnonzero((t > tj - 300) & (t < tj + 300))
    k = k[::max(1, len(k) // 150000)]
    ax.scatter(t[k] - tj, y[k], s=0.2, c=np.log10(amp[k] + 1), cmap='viridis', rasterized=True)
    ax.axvline(0, color='r', lw=0.8)
    ax.set_title(f'imec{p}: spike peaks +-300 s around the junction (colour = log amplitude)', fontsize=8)
    ax.set_xlabel('time from junction (s)')
    ax.set_ylabel('depth (um)')
    ax = axs[1, 0]
    tb = np.arange(0, t.max() + 10, 10)
    ax.plot(tb[:-1], np.histogram(t, tb)[0] / 10, color='k', lw=0.6)
    ax.axvline(tj, color='r', lw=0.8)
    for c in CONTROLS.values():
        ax.axvline(c, color='0.5', lw=0.5, ls='--')
    ax.set_title('total peak rate (Hz, 10-s bins) over the joined run', fontsize=8)
    ax.set_xlabel('joined time (s)')
    rows = []
    for c, (name, T) in enumerate(events.items()):
        pre = (t >= T - W) & (t < T)
        post = (t >= T) & (t < T + W)
        r_pre = np.histogram(y[pre], bins)[0] / W
        r_post = np.histogram(y[post], bins)[0] / W
        idx_pre = np.digitize(y[pre], bins) - 1
        idx_post = np.digitize(y[post], bins) - 1
        a_pre = np.array([np.median(amp[pre][idx_pre == i]) if (idx_pre == i).sum() > 20 else np.nan
                          for i in range(len(bins) - 1)])
        a_post = np.array([np.median(amp[post][idx_post == i]) if (idx_post == i).sum() > 20 else np.nan
                           for i in range(len(bins) - 1)])
        ok = (r_pre > 0.5) & (r_post > 0.5)
        lr = np.log2(r_post[ok] / r_pre[ok])
        oa = np.isfinite(a_pre) & np.isfinite(a_post)
        rows.append(dict(probe=p, event=name, time_s=T, n_bins=int(ok.sum()),
                         rate_profile_r=float(np.corrcoef(r_pre[ok], r_post[ok])[0, 1]) if ok.sum() > 2 else np.nan,
                         median_abs_log2_rate_ratio=float(np.median(np.abs(lr))) if len(lr) else np.nan,
                         amp_profile_r=float(np.corrcoef(a_pre[oa], a_post[oa])[0, 1]) if oa.sum() > 2 else np.nan,
                         median_amp_ratio=float(np.median(a_post[oa] / a_pre[oa])) if oa.sum() else np.nan))
        ax = axs[0, 1 + c]
        ax.loglog(r_pre[ok], r_post[ok], '.', ms=4, color='r' if name == 'junction' else '0.3')
        lo, hi = 0.3, max(r_pre.max(), r_post.max()) * 1.5
        ax.plot([lo, hi], [lo, hi], 'k', lw=0.5)
        ax.set_title(f'{name}: peak rate per 40-um bin (Hz)\nr = {rows[-1]["rate_profile_r"]:.3f}, '
                     f'median |log2 ratio| = {rows[-1]["median_abs_log2_rate_ratio"]:.2f}', fontsize=8)
        ax.set_xlabel(f'{W:.0f} s before')
        ax.set_ylabel(f'{W:.0f} s after')
        ax = axs[1, 1 + c]
        ax.plot(a_pre[oa], a_post[oa], '.', ms=4, color='r' if name == 'junction' else '0.3')
        lo2, hi2 = np.nanmin([a_pre[oa].min(), a_post[oa].min()]), np.nanmax([a_pre[oa].max(), a_post[oa].max()])
        ax.plot([lo2, hi2], [lo2, hi2], 'k', lw=0.5)
        ax.set_title(f'median peak amplitude per bin, r = {rows[-1]["amp_profile_r"]:.3f}, '
                     f'ratio {rows[-1]["median_amp_ratio"]:.3f}', fontsize=8)
        ax.set_xlabel('before')
        ax.set_ylabel('after')
    fig.suptitle(f'AB155 day 0 joined run, imec{p}: multi-unit activity across the junction ({tj:.3f} s) vs controls '
                 f'(windows {W:.0f} s)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f'{OUT}/002_mua_junction_imec{p}.png', dpi=110)
    plt.close(fig)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    off = offsets()
    ttl = ttl_figure(off)
    ttl.to_csv(f'{OUT}/002_ttl_counts.csv', index=False)
    print(ttl.to_string(index=False))
    plist = probes()
    print('probes with motion outputs:', [p for p, _ in plist])
    motion_figure(off, plist)
    rows = []
    for p, d in plist:
        rows += mua_figure(off, p, d)
    s = pd.DataFrame(rows)
    s.to_csv(f'{OUT}/002_mua_junction_summary.csv', index=False)
    print(s.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
