"""Whisker-artefact gap: per-stage figures and quantification from the 01_extract_stages.py caches."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap'
CASES = [('MH037_20250524_143522_imec3', 'MH037 0524 imec3'), ('MH037_20250524_143522_imec1', 'MH037 0524 imec1'),
         ('MH020_20250304_161732_imec0', 'MH020 0304 imec0'), ('MH031_20250507_104425_imec3', 'MH031 0507 imec3 (normal)')]
STAGES = [('raw', 'raw .ap.bin (uV)'), ('catgt', 'CatGT tcat, -gblcar (uV)'), ('corr', 'tcat_corrected (uV)'),
          ('pre', 'KS4 input: preprocess/ (uV)'), ('ks4f', 'KS4 input + KS4 HP 300 Hz (uV)'),
          ('ks4w', 'KS4 HP + whitening (a.u.)'), ('tempwh', 'KS4 temp_wh.dat (a.u.)'), ('rawf', 'raw + KS4 HP (uV)')]
C_W, C_A = '#F5A700', '#1F1FD0'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 6, 'axes.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 0.6})


def smooth(x, n):
    k = np.ones(n) / n
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode='same'), 0, x)


def rms_ratio(z, st, fs):
    r = np.sqrt(z[f'{st}_sq'])                       # (t, ch) RMS across trials
    r = smooth(r, int(fs / 2000))                    # 0.5 ms
    t = z['t_ms']
    base = np.median(r[t < -5], 0)
    return r / np.maximum(base, 1e-9)


def duration(t, ratio_med, thr=2.0):
    """last time in 0..80 ms with channel-median RMS ratio > thr (0 if never)."""
    m = (t >= 0) & (ratio_med > thr)
    return float(t[m].max()) if m.any() else 0.0


rows = []
summ = {}
for tag, title in CASES:
    zz = {m: dict(np.load(f'{D}/cache/{tag}_{m}.npz')) for m in ('whisker', 'auditory')}
    for m, z in zz.items():
        fs = float(z['fs']); t = z['t_ms']
        post40 = (t >= 0) & (t < 40)
        satf = z['sat_all']                          # (trials, t) fraction of channels saturated
        sat_t = satf.mean(0)
        sat_end = float(t[(t >= 0) & (sat_t > 0.01)].max()) if ((t >= 0) & (sat_t > 0.01)).any() else 0.0
        # KS4 spikes (all clusters, all channels)
        ed = z['spk_edges_ms']; tc = (ed[:-1] + ed[1:]) / 2
        rate = z['spk_psth'].sum(1)                  # spikes / trial / ms over probe
        brate = rate[tc < -5].mean()
        rr = np.convolve(rate / brate, np.ones(3) / 3, mode='same')
        low = (tc > 5) & (rr < 0.5)
        gap_end = float(tc[low].max()) if low.any() else 0.0
        for st, _ in STAGES:
            if f'{st}_sq' not in z:
                continue
            R = rms_ratio(z, st, fs)
            med = np.median(R, 1)
            chdur = ((R > 2) & post40[:, None]).sum(0) / fs * 1e3   # ms above 2x per channel
            rows.append(dict(case=tag, modality=m, stage=st, art_dur_ms=duration(t, med),
                             peak_ratio=float(med[t >= 0].max()), n_ch_gt5ms=int((chdur > 5).sum()),
                             frac_sat_samples_0_40ms=float(satf[:, post40].mean()), sat_last_ms_gt1pct=sat_end,
                             frac_trials_with_sat=float((satf[:, post40].max(1) > 0.05).mean()),
                             ks4_rate_0_5=float(rate[(tc > 0) & (tc < 5)].mean() / brate),
                             ks4_rate_5_20=float(rate[(tc > 5) & (tc < 20)].mean() / brate),
                             ks4_rate_20_40=float(rate[(tc > 20) & (tc < 40)].mean() / brate),
                             ks4_gap_end_ms=gap_end, n_trials_avg=int(z['n_trials']), n_trials_all=int(z['n_trials_all'])))
            summ[(tag, m, st)] = med
q = pd.DataFrame(rows)
q.to_csv(f'{D}/quantification_per_stage.csv', index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 30, 'display.max_rows', 200):
    print(q.round(3).to_string())

# ---------------- summary figure: cases x [saturation, RMS ratio per stage, KS4 spike rate] ----------------
cols = {'raw': '0.1', 'catgt': '#8c6d31', 'corr': '#bd9e39', 'pre': '#3182bd', 'ks4f': '#6baed6', 'ks4w': '#9467bd',
        'tempwh': '#e377c2', 'rawf': '0.55'}
fig, axs = plt.subplots(3, len(CASES), figsize=(7.2, 4.6), sharex=True)
for j, (tag, title) in enumerate(CASES):
    zz = {m: dict(np.load(f'{D}/cache/{tag}_{m}.npz')) for m in ('whisker', 'auditory')}
    t = zz['whisker']['t_ms']
    ax = axs[0, j]
    for m, c in (('whisker', C_W), ('auditory', C_A)):
        ax.plot(t, 100 * zz[m]['sat_all'].mean(0), color=c, label=f'{m} (n={zz[m]["n_trials_all"]})')
    ax.set_title(title, fontsize=6.5); ax.set_ylim(0, 100)
    if j == 0:
        ax.set_ylabel('raw: % channels at ADC limit'); ax.legend(frameon=False, fontsize=5)
    ax = axs[1, j]
    for st, lab in STAGES:
        if (tag, 'whisker', st) in summ:
            ax.plot(t, summ[(tag, 'whisker', st)], color=cols[st], label=st, lw=0.6)
    ax.plot(t, summ[(tag, 'auditory', 'raw')], color=C_A, ls='--', lw=0.6, label='raw, auditory')
    ax.set_yscale('log'); ax.axhline(2, color='0.6', lw=0.4, ls=':')
    if j == 0:
        ax.set_ylabel('RMS / baseline RMS\n(median over channels)'); ax.legend(frameon=False, fontsize=4.5, ncol=2)
    ax = axs[2, j]
    for m, c in (('whisker', C_W), ('auditory', C_A)):
        z = zz[m]; ed = z['spk_edges_ms']; tc = (ed[:-1] + ed[1:]) / 2
        r = z['spk_psth'].sum(1); ax.plot(tc, r / r[tc < -5].mean(), color=c, drawstyle='steps-mid')
    ax.axhline(1, color='0.6', lw=0.4, ls=':')
    ax.set_xlabel('Time from stimulus (ms)')
    if j == 0:
        ax.set_ylabel('KS4 spikes (all clusters)\n/ baseline rate')
for ax in axs.ravel():
    ax.axvline(0, color='0.5', lw=0.4); ax.set_xlim(-20, 80)
fig.tight_layout()
fig.savefig(f'{D}/fig1_summary_saturation_rms_spikes.png', dpi=300); fig.savefig(f'{D}/fig1_summary_saturation_rms_spikes.svg')
plt.close(fig)

# ---------------- per-case stage figure ----------------
for tag, title in CASES:
    zz = {m: dict(np.load(f'{D}/cache/{tag}_{m}.npz')) for m in ('whisker', 'auditory')}
    z = zz['whisker']; t = z['t_ms']; fs = float(z['fs']); sel = z['sel_ch']
    sts = [(s, l) for s, l in STAGES if f'{s}_sq' in z]
    fig, axs = plt.subplots(len(sts) + 1, 4, figsize=(7.2, 1.25 * (len(sts) + 1)),
                            gridspec_kw=dict(width_ratios=[1, 1, 1.2, 1.2]))
    for i, (st, lab) in enumerate(sts):
        # trial mean, 8 channels offset
        ax = axs[i, 0]; x = z[f'{st}_mean'][:, sel]
        sc = np.percentile(np.abs(x[t < -5]), 99) * 6 + 1e-9
        for k in range(len(sel)):
            ax.plot(t, x[:, k] / sc + k, color='k', lw=0.4)
        ax.set_ylabel(lab, fontsize=5.5); ax.set_yticks([])
        # single trials of one mid channel (6 trials, offset)
        ax = axs[i, 1]; s1 = z[f'{st}_single'][:, :, len(sel) // 2]
        sc1 = np.percentile(np.abs(s1[:, t < -5]), 99) * 3 + 1e-9
        for k in range(s1.shape[0]):
            ax.plot(t, s1[k] / sc1 + k, color='k', lw=0.35)
        ax.set_yticks([])
        if st == 'raw':
            uvb = float(z['uv_per_bit']); lim = uvb * (2048 if uvb > 3 else 512)   # ADC limit: NP1 +-512, NP2 +-2048 counts
            ax.text(0.99, 0.98, f'ADC limit +-{lim:.0f} uV', transform=ax.transAxes, ha='right', va='top', fontsize=4.5)
        # heatmaps: RMS ratio whisker / auditory
        for jj, m in ((2, 'whisker'), (3, 'auditory')):
            R = rms_ratio(zz[m], st, fs)
            ax = axs[i, jj]
            im = ax.imshow(np.log10(np.clip(R.T, 0.1, 10)), aspect='auto', origin='lower', cmap='RdBu_r', vmin=-1, vmax=1,
                           extent=[t[0], t[-1], 0, 384], interpolation='nearest')
            if i == 0:
                ax.set_title(f'{m}: log10 RMS / baseline (-1 blue .. +1 red)', fontsize=5.5)
            ax.set_ylabel('channel', fontsize=5)
    # bottom row: saturation (raw) and KS4 spike raster/PSTH
    ax = axs[-1, 0]
    for m, c in (('whisker', C_W), ('auditory', C_A)):
        ax.plot(t, 100 * zz[m]['sat'].mean(1), color=c)
    ax.set_ylabel('% samples at\nADC limit (30 tr)')
    ax = axs[-1, 1]
    ax.scatter(z['rast_t'], z['rast_ch'], s=0.1, c='k', lw=0, rasterized=True); ax.set_ylabel('KS4 spike channel')
    ax.set_ylim(0, 384)
    for jj, m in ((2, 'whisker'), (3, 'auditory')):
        ax = axs[-1, jj]; zm = zz[m]; ed = zm['spk_edges_ms']
        p = zm['spk_psth'] / np.maximum(zm['spk_psth'][(ed[:-1] < -5)].mean(0), 1e-3)
        ax.imshow(np.clip(p.T, 0, 3), aspect='auto', origin='lower', cmap='Greys', vmin=0, vmax=3,
                  extent=[ed[0], ed[-1], 0, 384], interpolation='nearest')
        ax.set_ylabel('channel', fontsize=5); ax.set_title(f'{m}: KS4 spikes / baseline (0-3)', fontsize=5.5)
    for ax in axs.ravel():
        ax.axvline(0, color='c', lw=0.4); ax.set_xlim(-20, 80)
    for ax in axs[-1]:
        ax.set_xlabel('Time from stimulus (ms)')
    fig.suptitle(f'{title}: whisker trials, {int(z["n_trials"])} trials averaged; channels shown: {[int(c) for c in sel]}', fontsize=6.5)
    fig.tight_layout()
    fig.savefig(f'{D}/fig2_stages_{tag}.png', dpi=250)
    plt.close(fig)
print('done')
