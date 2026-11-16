"""006 -- Trial-by-trial activity of stable units over the joined AB155 day-0 recording (test NWB from 004, labels from
005). Stable units = quality_label 'good' (classify_units_quality over the joined run: coverage >= 0.9, presence >= 0.5).

Trials: trial-start TTLs (joined NIDQ time, NWB acquisition 'trial_start_ttl') paired with the behaviour results:
part 1 = results_corrected_bis.csv (239 trials) -> first 239 part-1 TTLs (the 240th, 0.5 s before the junction, is the
part-1 end TTL, dropped); part 2 = results_2.csv (71 trials, passive post) -> first 71 part-2 TTLs (last = end TTL).
Pairing checked: TTL intervals vs results trial_time intervals (max abs difference printed, must be < 0.1 s).
Trial types: perf 0 W miss, 1 A miss, 2 W hit, 3 A hit, 4 CR, 5 FA, 6 passive (pre = part 1, post = part 2).
Whisker stim TTL coincides with the trial-start TTL, so all trials are aligned to the trial-start TTL.

Outputs (<joined>/concat_figure/):
  006_trials_imec<p>.png        per stable unit x trial: baseline rate (-1..0 s, / unit median) and evoked rate
                                (0..100 ms minus baseline, z per unit), trial-type strip, junction; population means
                                per trial; passive whisker response pre vs post per unit
  006_trial_rasters_imec<p>.png 12 most whisker-responsive stable units: trial rasters (-0.5..1 s, trials in
                                chronological order, coloured by type, junction line) + PSTH per trial type
  006_trials.csv                trial table (index, part, time, type)
  006_unit_pre_post.csv         per stable unit: passive whisker evoked rate pre vs post, baseline pre vs post
Run (haas): python 006_trial_activity_nwb.py
"""
import matplotlib
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
NWB = f'{J}/nwb_test/AB155_20250208_175103_joined_test.nwb'
OUT = f'{J}/concat_figure'
TRAIN = '/mnt/lsens-data/AB155/Training/AB155_20250208_175103'
BASE = (-1.0, 0.0)
EVOK = (0.0, 0.1)
RAST = (-0.5, 1.0)
TYPES = ['passive pre W', 'passive pre A', 'W hit', 'W miss', 'A hit', 'A miss', 'FA', 'CR',
         'passive post W', 'passive post A', 'passive no stim']
TCOL = {'passive pre W': '#1f77b4', 'passive pre A': '#9ecae1', 'W hit': '#2ca02c', 'W miss': '#98df8a',
        'A hit': '#d62728', 'A miss': '#ff9896', 'FA': '#ff7f0e', 'CR': '#7f7f7f',
        'passive post W': '#9467bd', 'passive post A': '#c5b0d5', 'passive no stim': '#000000'}


def trial_type(r, part):
    if r.perf == 6:
        mod = 'W' if r.is_whisker else ('A' if r.is_auditory else None)
        return f'passive {part} {mod}' if mod else 'passive no stim'
    return {0: 'W miss', 1: 'A miss', 2: 'W hit', 3: 'A hit', 4: 'CR', 5: 'FA'}.get(r.perf, 'passive no stim') \
        if part == 'pre' else ('passive no stim' if not (r.is_whisker or r.is_auditory) else
                               f'passive post {"W" if r.is_whisker else "A"}')


def trials(ttl, tj):
    r1 = pd.read_csv(f'{TRAIN}/results_corrected_bis.csv')
    r2 = pd.read_csv(f'{TRAIN}/results_2.csv')
    t1, t2 = ttl[ttl < tj], ttl[ttl >= tj]
    assert len(t1) == len(r1) + 1 and len(t2) == len(r2) + 1, (len(t1), len(r1), len(t2), len(r2))
    rows = []
    for part, r, t in (('pre', r1, t1[:len(r1)]), ('post', r2, t2[:len(r2)])):
        err = np.max(np.abs(np.diff(t) - np.diff(r.trial_time.to_numpy())))
        print(f'part {part}: {len(r)} trials, max |TTL interval - trial_time interval| = {err:.4f} s')
        assert err < 0.1
        for (_, x), tt in zip(r.iterrows(), t):
            rows.append(dict(part=1 if part == 'pre' else 2, time=tt, perf=x.perf, type=trial_type(x, part)))
    d = pd.DataFrame(rows)
    d.index.name = 'trial'
    return d


def counts(s, t0, win):
    i0 = np.searchsorted(s, t0 + win[0])
    i1 = np.searchsorted(s, t0 + win[1])
    return (i1 - i0) / (win[1] - win[0])


def main():
    with NWBHDF5IO(NWB, 'r') as io:
        f = io.read()
        u = f.units.to_dataframe()
        ttl = np.asarray(f.acquisition['trial_start_ttl'].timestamps[:])
        ep = f.epochs.to_dataframe()
    u['probe'] = u['electrode_group'].map(lambda g: g.name.split('_')[0])
    u['cluster_id'] = pd.to_numeric(u['cluster_id'])
    u['depth'] = pd.to_numeric(u['depth'], errors='coerce')
    q = pd.read_csv(f'{OUT}/005_units_quality.csv')[['probe', 'cluster_id', 'quality_label']]
    u = u.merge(q, on=['probe', 'cluster_id'], how='left')
    st = u[u.quality_label == 'good'].reset_index(drop=True)
    tj = float(ep.loc[ep.tags.map(lambda t: 'part2' in t), 'start_time'].iloc[0])
    tr = trials(ttl, tj)
    tr.to_csv(f'{OUT}/006_trials.csv')
    print(tr.groupby(['part', 'type']).size().to_string())
    nj = int((tr.part == 1).sum()) - 0.5   # junction between trial indices
    types_present = [t for t in TYPES if t in set(tr.type)]
    tcode = tr.type.map({t: i for i, t in enumerate(types_present)}).to_numpy()
    cmap_t = ListedColormap([TCOL[t] for t in types_present])

    pp_rows = []
    for p in sorted(st.probe.unique()):
        g = st[st.probe == p].sort_values('depth').reset_index(drop=True)
        if not len(g):
            continue
        sp = [np.sort(np.asarray(s)) for s in g.spike_times]
        base = np.stack([counts(s, tr.time.to_numpy(), BASE) for s in sp])
        evk = np.stack([counts(s, tr.time.to_numpy(), EVOK) for s in sp]) - base
        base_n = base / (np.median(base, axis=1, keepdims=True) + 1e-9)
        evk_z = (evk - evk.mean(1, keepdims=True)) / (evk.std(1, keepdims=True) + 1e-9)
        wpre = (tr.type == 'passive pre W').to_numpy()
        wpost = (tr.type == 'passive post W').to_numpy()
        for k, row in g.iterrows():
            pp_rows.append(dict(probe=p, cluster_id=row.cluster_id, depth=row.depth,
                                wh_evoked_pre=evk[k, wpre].mean(), wh_evoked_post=evk[k, wpost].mean(),
                                baseline_pre=base[k, tr.part.to_numpy() == 1].mean(),
                                baseline_post=base[k, tr.part.to_numpy() == 2].mean()))

        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(4, 2, height_ratios=[0.12, 1, 1, 0.8], width_ratios=[4, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(tcode[None], aspect='auto', cmap=cmap_t, interpolation='nearest',
                  vmin=-0.5, vmax=len(types_present) - 0.5)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_title(f'AB155 day 0 joined test NWB, {p}: {len(g)} stable (quality good) units x {len(tr)} trials; '
                     f'black line = junction', fontsize=10)
        for r_, (mat, ttl_, cm, lim) in enumerate(
                ((base_n, f'baseline rate {BASE[0]:.0f}..{BASE[1]:.0f} s / unit median', 'magma', (0, 2)),
                 (evk_z, f'evoked rate {EVOK[0]*1e3:.0f}..{EVOK[1]*1e3:.0f} ms minus baseline, z per unit',
                  'RdBu_r', (-3, 3)))):
            ax = fig.add_subplot(gs[1 + r_, 0])
            im = ax.imshow(mat, aspect='auto', cmap=cm, vmin=lim[0], vmax=lim[1], interpolation='nearest')
            ax.axvline(nj, color='k' if cm == 'RdBu_r' else 'c', lw=1.2)
            ax.set_ylabel('units (by depth)')
            ax.set_title(ttl_, fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.015, pad=0.005)
        ax = fig.add_subplot(gs[3, 0])
        x = np.arange(len(tr))
        for t in types_present:
            m = (tr.type == t).to_numpy()
            ax.plot(x[m], evk[:, m].mean(0), 'o', ms=3, color=TCOL[t], label=t)
        ax.plot(x, base.mean(0), '-', color='0.5', lw=0.8, label='baseline (mean rate)')
        ax.axvline(nj, color='k', lw=1)
        ax.set_xlim(-0.5, len(tr) - 0.5)
        ax.set_xlabel('trial (chronological)')
        ax.set_ylabel('population mean (Hz)')
        ax.legend(fontsize=7, ncol=6, loc='upper left')
        ax = fig.add_subplot(gs[1, 1])
        a, b = evk[:, wpre].mean(1), evk[:, wpost].mean(1)
        ax.plot(a, b, '.', color='k')
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        ax.plot([lo, hi], [lo, hi], 'k', lw=0.5)
        r = np.corrcoef(a, b)[0, 1] if len(a) > 2 else np.nan
        ax.set_xlabel(f'passive pre W evoked (Hz), n = {wpre.sum()} trials')
        ax.set_ylabel(f'passive post W evoked (Hz), n = {wpost.sum()} trials')
        ax.set_title(f'per unit, r = {r:.2f}', fontsize=9)
        ax = fig.add_subplot(gs[2, 1])
        a, b = base[:, tr.part.to_numpy() == 1].mean(1), base[:, tr.part.to_numpy() == 2].mean(1)
        ax.loglog(a + 0.01, b + 0.01, '.', color='k')
        ax.plot([0.01, 200], [0.01, 200], 'k', lw=0.5)
        ax.set_xlabel('baseline rate part 1 (Hz)')
        ax.set_ylabel('baseline rate part 2 (Hz)')
        ax.set_title('per unit, baseline before trials', fontsize=9)
        fig.tight_layout()
        fig.savefig(f'{OUT}/006_trials_{p}.png', dpi=105)
        plt.close(fig)

        # trial rasters of the most whisker-responsive stable units
        resp = evk[:, wpre | (tr.type == 'W hit').to_numpy() | (tr.type == 'W miss').to_numpy()].mean(1)
        top = np.argsort(resp)[::-1][:12]
        ncol = 4
        nrow = int(np.ceil(len(top) / ncol))
        fig = plt.figure(figsize=(18, 6.2 * nrow))
        gs = fig.add_gridspec(2 * nrow, ncol, height_ratios=[3, 1] * nrow)
        tt = tr.time.to_numpy()
        bins = np.arange(RAST[0], RAST[1] + 0.02, 0.02)
        for j, k in enumerate(top):
            s = sp[k]
            ax = fig.add_subplot(gs[2 * (j // ncol), j % ncol])
            axp = fig.add_subplot(gs[2 * (j // ncol) + 1, j % ncol], sharex=ax)
            psth = {}
            for i, t0 in enumerate(tt):
                z = s[np.searchsorted(s, t0 + RAST[0]):np.searchsorted(s, t0 + RAST[1])] - t0
                ax.plot(z, np.full(len(z), i), '|', ms=1.6, mew=0.5, color=TCOL[tr.type.iat[i]], rasterized=True)
                psth.setdefault(tr.type.iat[i], []).append(np.histogram(z, bins)[0])
            ax.axhline(nj, color='k', lw=0.8)
            ax.axvline(0, color='0.5', lw=0.5)
            ax.set_ylim(len(tt), -1)
            ax.set_title(f'cluster {g.cluster_id[k]}, depth {g.depth[k]:.0f} um', fontsize=8)
            if j % ncol == 0:
                ax.set_ylabel('trial (chronological)')
            for t in ('passive pre W', 'W hit', 'W miss', 'passive post W', 'passive pre A', 'passive post A'):
                if t in psth:
                    axp.plot(bins[:-1] + 0.01, np.mean(psth[t], 0) / 0.02, color=TCOL[t], lw=1, label=t)
            axp.axvline(0, color='0.5', lw=0.5)
            axp.set_xlim(*RAST)
            axp.set_xlabel('time from trial start (s)')
            if j == 0:
                axp.legend(fontsize=6)
                axp.set_ylabel('Hz')
        fig.suptitle(f'AB155 day 0 joined test NWB, {p}: trial rasters of the {len(top)} most whisker-responsive '
                     f'stable units (colour = trial type, black line = junction)', fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(f'{OUT}/006_trial_rasters_{p}.png', dpi=100)
        plt.close(fig)
        print(f'{p}: {len(g)} stable units; passive W evoked pre vs post r = {r:.2f}', flush=True)
    pd.DataFrame(pp_rows).to_csv(f'{OUT}/006_unit_pre_post.csv', index=False)


if __name__ == '__main__':
    main()
