"""Whisker-artefact gap: extract peri-stimulus AP-band data at every processing stage (read-only).

Stages (all in the probe file's own sample clock; raw t0 and CatGT tcat share firstSample, so sample i is the same):
  raw     SpikeGLX t0 .ap.bin (lsens-data), uV
  catgt   Myriam's CatGT tcat (-gblcar, no filter), uV
  corr    Myriam's tcat_corrected (pipeline run_artifact_correction output), uV
  pre     KS4 input: Axel_Bisi/data/.../preprocess/traces_cached_seg0.raw (float32, = int16 counts after astype
          [+ RemoveArtifacts for MH031] + DREDGE InterpolateMotion), uV
  ks4f    pre + KS4 high-pass (ops['preprocessing']['hp_filter'], i.e. filtfilt butter3 300 Hz), uV
  ks4w    ks4f whitened with KS4 Wrot (ops['preprocessing']['whiten_mat']), whitened units
  tempwh  KS4's own saved filtered+whitened copy temp_wh.dat (MH031 imec3 only)
  rawf    raw + same KS4 high-pass (to separate pipeline effects from the signal itself), uV
Spikes: KS4 sorter_output spike_times.npy / spike_clusters.npy (all clusters), channel = ibl clusters.channels.

Saves one npz per case and modality with trial-averaged summaries (no big arrays).
Usage: python 01_extract_stages.py <case_index>
"""
import sys, os
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from pynwb import NWBHDF5IO

OUT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap/cache'
NWB = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'
MH = '/mnt/lsens-analysis/Myriam_Hamon/data'
AB = '/mnt/lsens-analysis/Axel_Bisi/data'
RAW = '/mnt/lsens-data'
CASES = [('MH037_20250524_143522', 'MH037_g1', 3), ('MH037_20250524_143522', 'MH037_g1', 1),
         ('MH020_20250304_161732', 'MH020_g0', 0), ('MH031_20250507_104425', 'MH031_g0', 3)]
N_TRIALS = 30
PRE_MS, POST_MS = 50, 80
PAD_MS = 150                       # padding for the high-pass (KS4 filter impulse response decays in < 20 ms)
SEL_CH = 8                         # channels kept for single-trial traces
rng = np.random.default_rng(0)


def meta(p):
    d = {}
    for l in open(p):
        k, _, v = l.strip().partition('=')
        d[k.lstrip('~')] = v
    return d


def main(ci):
    sid, run, pr = CASES[ci]
    mouse = sid[:5]
    tag = f'{sid}_imec{pr}'
    os.makedirs(OUT, exist_ok=True)
    rawp = f'{RAW}/{mouse}/Recording/{sid}/Ephys/{run}/{run}_imec{pr}/{run}_t0.imec{pr}.ap'
    mhd = f'{MH}/{mouse}/{sid}/Ephys/catgt_{run}/{run}_imec{pr}'
    abd = f'{AB}/{mouse}/{sid}/Ephys/catgt_{run}/{run}_imec{pr}'
    so = f'{abd}/kilosort4/sorter_output'
    md = meta(rawp + '.meta')
    fs = float(md['imSampRate']); nch = int(md['nSavedChans'])
    uv = float(md['imAiRangeMax']) / int(md['imMaxInt']) / float(md.get('imChan0apGain', 500)) * 1e6
    maxint = int(md['imMaxInt'])
    mm = {'raw': np.memmap(rawp + '.bin', dtype='int16', mode='r').reshape(-1, nch),
          'catgt': np.memmap(f'{mhd}/{run}_tcat.imec{pr}.ap.bin', dtype='int16', mode='r').reshape(-1, nch)}
    if os.path.exists(f'{mhd}/{run}_tcat_corrected.imec{pr}.ap.bin'):
        mm['corr'] = np.memmap(f'{mhd}/{run}_tcat_corrected.imec{pr}.ap.bin', dtype='int16', mode='r').reshape(-1, nch)
    mm['pre'] = np.memmap(f'{abd}/preprocess/traces_cached_seg0.raw', dtype='float32', mode='r').reshape(-1, 384)
    if os.path.exists(f'{so}/temp_wh.dat'):
        mm['tempwh'] = np.memmap(f'{so}/temp_wh.dat', dtype='int16', mode='r').reshape(-1, 384)
    ops = np.load(f'{so}/ops.npy', allow_pickle=True).item()
    hp = np.asarray(ops['preprocessing']['hp_filter'], float)
    W = np.asarray(ops['preprocessing']['whiten_mat'], float)
    c = len(hp) // 2
    k = int(PAD_MS / 1e3 * fs * 0.6)
    hp_s = hp[c - k:c + k + 1]                      # central part of the zero-phase impulse response
    print(tag, 'fs', fs, 'uV/bit', uv, 'hp kernel centre', np.argmax(np.abs(hp)), len(hp), 'kept frac energy',
          (hp_s ** 2).sum() / (hp ** 2).sum(), 'stages', list(mm))

    # ---- trials --------------------------------------------------------------------------------------------
    with NWBHDF5IO(f'{NWB}/{sid}.nwb', 'r', load_namespaces=True) as io:
        tr = io.read().trials.to_dataframe()
    ft = np.loadtxt(f'{mhd}/whisker_stim_times_to_imec{pr}.txt')
    wall = tr.loc[tr.whisker_stim == 1, 'start_time'].to_numpy()
    d = ft[:, None] - wall[None, :]
    h, e = np.histogram(d.ravel(), bins=np.arange(-50, 50, 0.005))
    off = e[np.argmax(h)] + 0.0025
    jj = np.abs(d - off).argmin(1); ok = np.abs(d[np.arange(len(ft)), jj] - off) < 0.02
    a, b = np.polyfit(wall[jj[ok]], ft[ok], 1)       # NWB time -> probe-file time
    res = ft[ok] - (a * wall[jj[ok]] + b)
    print(f'  NWB->file fit on {ok.sum()}/{len(ft)} whisker stims: slope {a:.8f} icpt {b:.4f}, |resid| max {1e3 * np.abs(res).max():.3f} ms')
    act = tr[(tr.context.astype(str) != 'passive') & (tr['perf'] != 6)]
    on = {}
    for m in ('whisker', 'auditory'):
        t_nwb = act.loc[act[f'{m}_stim'] == 1, 'start_time'].to_numpy()
        t_file = a * t_nwb + b
        if m == 'whisker':                           # snap to the TPrime-aligned whisker artefact times
            q = np.abs(ft[None, :] - t_file[:, None]).argmin(1)
            t_file = np.where(np.abs(ft[q] - t_file) < 0.005, ft[q], t_file)
        on[m] = t_file
    # ---- spikes --------------------------------------------------------------------------------------------
    st = np.load(f'{so}/spike_times.npy').astype(np.int64).ravel()
    sc = np.load(f'{so}/spike_clusters.npy').ravel()
    cch = np.load(f'{abd}/kilosort4/ibl_format/clusters.channels.npy')
    sch = cch[sc]
    # channels for single-trial display: the channel with most spikes in each of SEL_CH depth blocks
    cnt = np.bincount(sch, minlength=384)
    sel = np.array([blk[np.argmax(cnt[blk])] for blk in np.array_split(np.arange(384), SEL_CH)])

    n0, n1 = int(round(PRE_MS / 1e3 * fs)), int(round(POST_MS / 1e3 * fs))
    npad = int(round(PAD_MS / 1e3 * fs))
    tms = np.arange(-n0, n1) / fs * 1e3
    bl = tms < -5
    for m in ('whisker', 'auditory'):
        allt = on[m]
        # saturation over ALL active trials of this modality (raw only)
        sat_all = np.zeros((len(allt), n0 + n1), np.float32)
        for i, t in enumerate(allt):
            s = int(round(t * fs))
            w = np.asarray(mm['raw'][s - n0:s + n1, :384])
            sat_all[i] = ((w >= maxint - 1) | (w <= -maxint)).mean(1)
        idx = np.linspace(0, len(allt) - 1, min(N_TRIALS, len(allt))).round().astype(int)
        trials = allt[idx]
        acc = {}
        single = {}
        for i, t in enumerate(trials):
            s = int(round(t * fs))
            seg = {}
            for name in ('raw', 'catgt', 'corr'):
                if name in mm:
                    seg[name] = np.asarray(mm[name][s - n0 - npad:s + n1 + npad, :384], float) * uv
            seg['pre'] = np.asarray(mm['pre'][s - n0 - npad:s + n1 + npad], float) * uv
            # KS4 high-pass (zero-phase kernel) on pre and on raw
            for src, dst in (('pre', 'ks4f'), ('raw', 'rawf')):
                x = seg[src] - seg[src].mean(0)
                seg[dst] = fftconvolve(x, hp_s[:, None], mode='same', axes=0)
            seg['ks4w'] = seg['ks4f'] @ W.T / uv          # KS4 whitens data in int16-count units
            if 'tempwh' in mm:
                seg['tempwh'] = np.asarray(mm['tempwh'][s - n0 - npad:s + n1 + npad], float)
            rw = np.asarray(mm['raw'][s - n0:s + n1, :384])
            seg_sat = ((rw >= maxint - 1) | (rw <= -maxint)).astype(np.float32)
            for name, x in seg.items():
                x = x[npad:npad + n0 + n1]
                x = x - x[bl].mean(0)                     # per-trial baseline offset
                for key, val in (('mean', x), ('abs', np.abs(x)), ('sq', x ** 2)):
                    acc.setdefault(f'{name}_{key}', np.zeros_like(x, dtype=np.float64))
                    acc[f'{name}_{key}'] += val
                if i < 6:
                    single.setdefault(name, []).append(x[:, sel].astype(np.float32))
            acc.setdefault('sat', np.zeros((n0 + n1, 384)))
            acc['sat'] += seg_sat
        nT = len(trials)
        outd = {kk: (v / nT).astype(np.float32) for kk, v in acc.items()}
        for name, lst in single.items():
            outd[f'{name}_single'] = np.stack(lst)
        # KS4 spikes: per-channel 1-ms PSTH and pooled raster
        ed = np.arange(-n0, n1 + 1, int(round(fs / 1000)))
        psth = np.zeros((len(ed) - 1, 384)); rast_t, rast_ch, rast_tr = [], [], []
        for i, t in enumerate(allt):
            s = int(round(t * fs))
            i0, i1 = np.searchsorted(st, [s - n0, s + n1])
            dt = st[i0:i1] - s; ch = sch[i0:i1]
            psth += np.histogram2d(dt, ch, bins=[ed, np.arange(385)])[0]
            if i in set(idx):
                rast_t.append(dt / fs * 1e3); rast_ch.append(ch); rast_tr.append(np.full(len(dt), i))
        outd['spk_psth'] = (psth / len(allt)).astype(np.float32)   # spikes per trial per 1-ms bin per channel
        outd['spk_edges_ms'] = ed / fs * 1e3
        outd['rast_t'] = np.concatenate(rast_t); outd['rast_ch'] = np.concatenate(rast_ch)
        outd['sat_all'] = sat_all
        np.savez_compressed(f'{OUT}/{tag}_{m}.npz', t_ms=tms, sel_ch=sel, n_trials=nT, n_trials_all=len(allt),
                            trial_times_file_s=trials, uv_per_bit=uv, fs=fs, fit=np.array([a, b]), **outd)
        print(f'  {m}: {nT} trials averaged, {len(allt)} for saturation/spikes; saved')


if __name__ == '__main__':
    main(int(sys.argv[1]))
