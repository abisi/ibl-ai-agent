"""Does the post-whisker gap survive every preprocessing stage? Single trials, fast- vs slow-recovery trials (read-only, 2026-11-23).

Stages (same sample clock, see 01_extract_stages.py): raw SpikeGLX t0 -> CatGT tcat (-gblcar) -> tcat_corrected (lab artefact
replacement 3-4 ms) -> KS4 input (preprocess/traces_cached_seg0.raw, after DREDGE motion interpolation) -> + KS4 300 Hz
high-pass (KS4's own filter kernel) -> + KS4 whitening (KS4's Wrot).
Trials: all active whisker trials; per-trial recovery time from the raw spike band (pulse -0.5..4 ms bridged, Butterworth-3
300 Hz, median over in-brain channels < 330 of RMS / own pre-stimulus RMS, first t >= 5 ms with 3 bins >= 0.8). The 20 fastest
and the 20 slowest trials form the two groups; 20 auditory trials as reference.
Figure rows = stages; columns: fastest-trial example, slowest-trial example, auditory example (8 contiguous channels around
the most spiking in-brain channel, stage values, same scale per row, clipped), spike-band RMS / baseline over time (stage +
KS4 high-pass; median channels, mean trials) for fast / slow / auditory. Last row: KS4 spikes per 1-ms bin.
Output: <ROOT>/raw_after_5ms/stages_<session>_<probe>.png/.pdf
Usage: python 07_stages_by_recovery.py <case 0..2>
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, fftconvolve, sosfiltfilt
import h5py

ROOT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap'
NWB = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'
MH = '/mnt/lsens-analysis/Myriam_Hamon/data'
AB = '/mnt/lsens-analysis/Axel_Bisi/data'
RAW = '/mnt/lsens-data'
CASES = [('MH020_20250304_161732', 'MH020_g0', 0), ('MH037_20250524_143522', 'MH037_g1', 3),
         ('MH037_20250524_143522', 'MH037_g1', 1)]
PRE, POST, PAD = 30, 70, 150          # ms
NG = 20
INBRAIN = 330
CW, CA, CF, CS = '#f7b519', '#2c2cdb', '#1b9e77', '#d95f02'


def meta(p):
    d = {}
    for line in open(p):
        k, _, v = line.strip().partition('=')
        d[k.lstrip('~')] = v
    return d


def main(ci):
    sid, run, pr = CASES[ci]
    mouse = sid[:5]
    rawp = f'{RAW}/{mouse}/Recording/{sid}/Ephys/{run}/{run}_imec{pr}/{run}_t0.imec{pr}.ap'
    mhd = f'{MH}/{mouse}/{sid}/Ephys/catgt_{run}/{run}_imec{pr}'
    abd = f'{AB}/{mouse}/{sid}/Ephys/catgt_{run}/{run}_imec{pr}'
    so = f'{abd}/kilosort4/sorter_output'
    md = meta(rawp + '.meta')
    fs = float(md['imSampRate']); nch = int(md['nSavedChans']); maxint = int(md['imMaxInt'])
    ptype = md.get('imDatPrb_type', '0')
    gain = float(md.get('imChan0apGain', 80 if ptype != '0' else 500))
    uv = float(md['imAiRangeMax']) / maxint / gain * 1e6
    mm = {'raw': np.memmap(rawp + '.bin', dtype='int16', mode='r').reshape(-1, nch),
          'catgt': np.memmap(f'{mhd}/{run}_tcat.imec{pr}.ap.bin', dtype='int16', mode='r').reshape(-1, nch)}
    cp = f'{mhd}/{run}_tcat_corrected.imec{pr}.ap.bin'
    if os.path.exists(cp):
        mm['corr'] = np.memmap(cp, dtype='int16', mode='r').reshape(-1, nch)
    mm['pre'] = np.memmap(f'{abd}/preprocess/traces_cached_seg0.raw', dtype='float32', mode='r').reshape(-1, 384)
    ops = np.load(f'{so}/ops.npy', allow_pickle=True).item()
    hp = np.asarray(ops['preprocessing']['hp_filter'], float)
    W = np.asarray(ops['preprocessing']['whiten_mat'], float)
    c = len(hp) // 2; k = int(PAD / 1e3 * fs * 0.6); hp_s = hp[c - k:c + k + 1]

    # trials (NWB -> probe file clock, snapped to TPrime whisker times)
    with h5py.File(f'{NWB}/{sid}.nwb', 'r') as f:
        t_ = f['intervals/trials']
        tr = pd.DataFrame({q: t_[q][:] for q in ('start_time', 'whisker_stim', 'auditory_stim', 'perf')})
        ctx = t_['context'][:] if 'context' in t_ else np.array([b'nan'] * len(tr))
    tr['context'] = [x.decode() if isinstance(x, bytes) else str(x) for x in ctx]
    ft = np.loadtxt(f'{mhd}/whisker_stim_times_to_imec{pr}.txt').ravel()
    wall = tr.loc[tr.whisker_stim == 1, 'start_time'].to_numpy()
    d = ft[:, None] - wall[None, :]
    h, e = np.histogram(d.ravel(), bins=np.arange(-50, 50, 0.005)); off = e[np.argmax(h)] + 0.0025
    jj = np.abs(d - off).argmin(1); ok = np.abs(d[np.arange(len(ft)), jj] - off) < 0.02
    a, b = np.polyfit(wall[jj[ok]], ft[ok], 1)
    act = tr[(tr.context != 'passive') & (tr.perf != 6)]
    tw = a * act.loc[act.whisker_stim == 1, 'start_time'].to_numpy() + b
    q = np.abs(ft[None] - tw[:, None]).argmin(1); tw = np.where(np.abs(ft[q] - tw) < 0.005, ft[q], tw)
    ta = a * act.loc[act.auditory_stim == 1, 'start_time'].to_numpy() + b
    ta = ta[np.linspace(0, len(ta) - 1, min(NG, len(ta))).round().astype(int)]

    n0, n1, npd = (int(round(v / 1e3 * fs)) for v in (PRE, POST, PAD))
    tms = np.arange(-n0, n1) / fs * 1e3
    bm = (tms > -25) & (tms < -5)
    ed = np.arange(-PRE, POST + 1); bi = np.clip(np.searchsorted(ed, tms, side='right') - 1, 0, len(ed) - 2); ctr = ed[:-1] + .5
    cnt = np.bincount(bi, minlength=len(ctr))
    sos = butter(3, 300, 'highpass', fs=fs, output='sos')
    b0, b1 = n0 + int(round(-0.5e-3 * fs)), n0 + int(round(4e-3 * fs))

    def binrms(x):   # time x ch -> bins x ch
        return np.sqrt(np.stack([np.bincount(bi, weights=x[:, j] ** 2, minlength=len(ctr)) for j in range(x.shape[1])], 1) / cnt[:, None])

    # 1) per-trial recovery from raw
    rec = np.full(len(tw), np.nan)
    for i, t in enumerate(tw):
        s = int(round(t * fs))
        w = np.asarray(mm['raw'][s - n0:s + n1, :INBRAIN], np.float32) * uv
        w -= w[bm].mean(0)
        lam = np.linspace(0, 1, b1 - b0 + 1)[1:-1, None]; w[b0 + 1:b1] = w[b0] * (1 - lam) + w[b1] * lam
        r = binrms(sosfiltfilt(sos, w, axis=0)[:, ::4])
        rr = np.median(r / np.maximum(r[(ctr > -25) & (ctr < -5)].mean(0), 1e-6), 1)
        okb = rr >= 0.8
        for j in np.flatnonzero(ctr >= 5):
            if okb[j:j + 3].all():
                rec[i] = ctr[j] - .5; break
    recf = np.nan_to_num(rec, nan=POST)
    o = np.argsort(recf, kind='stable')
    groups = {'fast': tw[o[:NG]], 'slow': tw[o[-NG:]], 'auditory': ta}
    grec = {'fast': recf[o[:NG]], 'slow': recf[o[-NG:]]}
    print(sid, pr, 'recovery ms pct 0/25/50/75/100', np.percentile(recf, [0, 25, 50, 75, 100]), flush=True)

    # channels: 8 contiguous around the most-spiking in-brain channel
    st = np.load(f'{so}/spike_times.npy').astype(np.int64).ravel()
    sc = np.load(f'{so}/spike_clusters.npy').ravel()
    cch = np.load(f'{abd}/kilosort4/ibl_format/clusters.channels.npy'); sch = cch[sc]
    cn = np.bincount(sch, minlength=384)[:INBRAIN]
    c0 = int(np.clip(np.argmax(np.convolve(cn, np.ones(8), 'valid')), 0, INBRAIN - 8)); chs = np.arange(c0, c0 + 8)

    stages = [s for s in ('raw', 'catgt', 'corr', 'pre') if s in mm] + ['ks4f', 'ks4w']
    lab = dict(raw='raw .ap.bin (uV)', catgt='CatGT tcat -gblcar (uV)', corr='tcat_corrected (uV)',
               pre='KS4 input (motion-corr., uV)', ks4f='KS4 input + KS4 HP (uV)', ks4w='+ KS4 whitening (a.u.)')
    ex, rms = {}, {}
    for g, times in groups.items():
        acc = {s: [] for s in stages}
        for i, t in enumerate(times):
            s0 = int(round(t * fs))
            seg = {}
            for s in ('raw', 'catgt', 'corr'):
                if s in mm:
                    seg[s] = np.asarray(mm[s][s0 - n0 - npd:s0 + n1 + npd, :384], np.float64) * uv
            seg['pre'] = np.asarray(mm['pre'][s0 - n0 - npd:s0 + n1 + npd], np.float64) * uv
            seg['ks4f'] = fftconvolve(seg['pre'] - seg['pre'].mean(0), hp_s[:, None], mode='same', axes=0)
            seg['ks4w'] = seg['ks4f'] @ W.T / uv
            for s in stages:
                x = seg[s]
                xh = x if s in ('ks4f', 'ks4w') else fftconvolve(x - x.mean(0), hp_s[:, None], mode='same', axes=0)
                x = x[npd:npd + n0 + n1]; xh = xh[npd:npd + n0 + n1]
                x = x - x[bm].mean(0)
                r = binrms(xh[:, :INBRAIN])
                acc[s].append(np.median(r / np.maximum(r[(ctr > -25) & (ctr < -5)].mean(0), 1e-9), 1))
                if i == 0:
                    ex[g, s] = x[:, chs]
        for s in stages:
            rms[g, s] = np.mean(acc[s], 0)
    ks = {}
    for g, times in groups.items():
        c_ = np.zeros(len(ctr))
        for t in times:
            s0 = int(round(t * fs)); i0, i1 = np.searchsorted(st, [s0 - n0, s0 + n1])
            c_ += np.histogram((st[i0:i1] - s0) / fs * 1e3, ed)[0]
        ks[g] = c_ / len(times) / 1e-3

    fig, axs = plt.subplots(len(stages) + 1, 4, figsize=(17, 3.0 * (len(stages) + 1)),
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1.1], hspace=0.5, wspace=0.25))
    slow_rec = 'none' if grec['slow'][0] >= POST else f"{grec['slow'][0]:.0f} ms"
    titles = {'fast': f'fastest-recovery trial (rec. {grec["fast"][0]:.0f} ms)',
              'slow': f'slow-recovery trial (rec. {slow_rec})',
              'auditory': 'auditory trial'}
    for r_, s in enumerate(stages):
        step = 3 * np.median([np.std(ex[g, s][bm]) for g in groups]) * 2 + 1e-9
        for c_, (g, col) in enumerate((('fast', CW), ('slow', CW), ('auditory', CA))):
            ax = axs[r_, c_]
            for kk in range(8):
                ax.plot(tms, np.clip(ex[g, s][:, kk], -0.6 * step, 0.6 * step) - kk * step, color=col, lw=0.5)
            ax.axvspan(0, 5, color='0.88', zorder=0); ax.set_xlim(-PRE, POST); ax.set_yticks([])
            if r_ == 0:
                ax.set_title(titles[g], fontsize=10)
            if c_ == 0:
                ax.set_ylabel(lab[s] + f'\n{step:.0f} / ch', fontsize=9)
        ax = axs[r_, 3]
        ax.plot(ctr, rms['fast', s], color=CF, label=f'fast {NG} (rec. {grec["fast"].min():.0f}-{grec["fast"].max():.0f} ms)')
        ax.plot(ctr, rms['slow', s], color=CS, label=f'slow {NG} (rec. {grec["slow"].min():.0f}-{grec["slow"].max():.0f} ms)')
        ax.plot(ctr, rms['auditory', s], color=CA, label='auditory')
        ax.axhline(1, color='0.5', lw=0.5, ls='--'); ax.axvspan(0, 5, color='0.88', zorder=0)
        ax.set_yscale('log'); ax.set_ylim(0.05, 20); ax.set_xlim(-PRE, POST); ax.set_ylabel('spike-band RMS / pre', fontsize=9)
        if r_ == 0:
            ax.set_title('spike band (stage + KS4 HP), median ch < 330', fontsize=10); ax.legend(fontsize=7, frameon=False)
    for c_ in range(3):
        axs[-1, c_].axis('off')
    ax = axs[-1, 3]
    for g, col in (('fast', CF), ('slow', CS), ('auditory', CA)):
        ax.plot(ctr, ks[g], color=col)
    ax.axvspan(0, 5, color='0.88', zorder=0); ax.set_xlim(-PRE, POST); ax.set_xlabel('time from stimulus (ms)')
    ax.set_ylabel('KS4 spikes / s (probe)', fontsize=9); ax.set_title('KS4 output', fontsize=10)
    axs[-1, 0].text(0, 0.9, f'channels {chs[0]}-{chs[-1]} (8 contiguous, most KS4 spikes); traces clipped at +-0.6 spacing.\n'
                    f'Grey band 0-5 ms. Recovery: raw spike band >= 0.8 x pre (median channels < {INBRAIN}).\n'
                    f'Whisker trials {len(tw)}; recovery ms median {np.median(recf):.0f}, IQR {np.percentile(recf, 25):.0f}-'
                    f'{np.percentile(recf, 75):.0f}.', fontsize=9, va='top', transform=axs[-1, 0].transAxes)
    for ax in axs[-2, :3]:
        ax.set_xlabel('time from stimulus (ms)')
    nm = 'NP1.0' if ptype == '0' else f'NP2.0 ({ptype})'
    fig.suptitle(f'{sid} imec{pr} ({nm}): whisker artefact through the preprocessing stages, fast vs slow recovery trials', fontsize=12)
    os.makedirs(f'{ROOT}/raw_after_5ms', exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{ROOT}/raw_after_5ms/stages_{sid}_imec{pr}.{ext}', dpi=120, bbox_inches='tight')
    pd.DataFrame(dict(trial_time_file_s=tw, recovery_ms=rec)).assign(session_id=sid, probe=f'imec{pr}').to_csv(
        f'{ROOT}/raw_after_5ms/recovery_per_trial_{sid}_imec{pr}.csv', index=False)


if __name__ == '__main__':
    main(int(sys.argv[1]))
