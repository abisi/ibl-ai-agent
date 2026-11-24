"""TENTATIVE pilot: recover spikes inside the post-whisker dead time on NP1.0 probes by undoing the front-end gain drop
(read-only; raw data, KS4 output and NWB untouched). 2026-11-23.

Idea: on NP1.0 (MH020_20250304 imec0, MH037_20250524 imec3) spikes and background noise shrink by the same factor after
the coil artefact (SNR preserved), so dividing the signal by a per-trial gain g(t) and re-detecting spikes by template
matching may recover them.

Per probe:
  prep     raw .ap.bin -> uV, per-channel baseline median removed, artefact bridged (-0.5 ms .. per-trial artefact end =
           last saturated sample in 0-8 ms + 0.5 ms, at least 4 ms; whisker only), common median reference over the
           affected channels (as CatGT -gblcar; removes the MH037 interference and release deflection), Butterworth-3
           300 Hz zero-phase high-pass. Channels = affected channels (whisker 6-18 ms RMS ratio < 0.7 on the slowest
           trials; MH020: channels >= ~335 are not attenuated and keep the KS4 output).
  gain     g(trial, t) = median over channels of [median |x| in 1-ms bins / that channel's median |x| at -55..-5 ms],
           running median over 3 bins, interpolated to samples (noise-based, robust to spikes; checked on auditory
           trials, where it should stay ~1 despite the evoked response).
  templates  per KS4 unit: mean high-passed waveform (-0.6..+1.4 ms) on its peak channel +-4 channels, from KS4 spikes in
           the pre-stimulus windows (-55..-5 ms) of all trials; units with >= 30 spikes.
  detect   matched-filter amplitude a(t) = <x, T> / <T, T> (1 = template-sized spike), peaks >= theta_u, >= 1 ms apart;
           theta_u per unit chosen on clean data (pre-stimulus windows + auditory trials) to maximise F1 vs KS4 spikes
           of that unit (+-0.5 ms). Units with clean F1 >= 0.6 are evaluated.
  synthetic test (ground truth known): auditory trials (clean, with evoked responses) are multiplied in raw int16
           counts by the gain profile g of a randomly paired whisker trial (from its artefact end on; earlier samples
           masked), re-quantised, then prep -> gain estimate -> correction -> detection. Compared with the detections of
           the same detector on the untouched auditory trial (D0) and with KS4 spikes: recall / precision per 5-ms bin
           and per true-gain level, for (a) no correction, (b) estimated gain, (c) true gain.
  real     whisker trials corrected with their estimated gain (masked before the artefact end and where g < G_MIN);
           recovered per-unit PSTHs vs KS4 whisker and auditory PSTHs.
Outputs: <ROOT>/gain_recovery_pilot/<tag>_{summary.json, synthetic.csv, psth.npz, units.csv, fig.png}
Usage: python 08_gain_recovery_pilot.py <0|1>
"""
import importlib.util
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, fftconvolve, find_peaks, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('scr', os.path.join(HERE, '..', 'screen', '01_screen_probe.py'))
scr = importlib.util.module_from_spec(spec); spec.loader.exec_module(scr)

ROOT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap'
OUT = f'{ROOT}/gain_recovery_pilot'
CASES = [('MH020_20250304_161732', 'imec0'), ('MH037_20250524_143522', 'imec3')]
PRE, POST, PAD = 60, 80, 40              # ms
BASE = (-55, -5)
G_MIN = 0.12
TOL_MS = 0.5
N_AUD = 120
SEED = 0
rng = np.random.default_rng(SEED)


def load(sid, probe):
    inv = pd.read_csv(f'{scr.OUT}/tables/00_inventory.csv')
    r = inv[(inv.session_id == sid) & (inv.probe == probe)].iloc[0]
    md = scr.meta(r.raw_bin[:-4] + '.meta')
    fs = float(md['imSampRate']); nch = int(md['nSavedChans']); maxint = int(md['imMaxInt'])
    uv = float(md['imAiRangeMax']) / maxint / float(md.get('imChan0apGain', 500)) * 1e6
    x = np.memmap(r.raw_bin, dtype='int16', mode='r').reshape(-1, nch)
    tr = scr.read_trials(sid)
    ft, a, b, *_ = scr.align(tr, r.whisker_times_file)
    act = tr[(tr.context != 'passive') & (tr.perf != 6)]
    tw = a * act.loc[act.whisker_stim == 1, 'start_time'].to_numpy() + b
    q = np.abs(ft[None] - tw[:, None]).argmin(1); tw = np.where(np.abs(ft[q] - tw) < 0.005, ft[q], tw)
    ta = a * act.loc[act.auditory_stim == 1, 'start_time'].to_numpy() + b
    ta = ta[np.linspace(0, len(ta) - 1, min(N_AUD, len(ta))).round().astype(int)]
    so = f'{r.ks4_dir}/sorter_output'
    st = np.load(f'{so}/spike_times.npy').astype(np.int64).ravel()
    sc = np.load(f'{so}/spike_clusters.npy').ravel()
    cch = np.load(f'{r.ks4_dir}/ibl_format/clusters.channels.npy')
    return r, x, fs, uv, maxint, tw, ta, st, sc, cch


class Prep:
    def __init__(self, x, fs, uv, maxint):
        self.x, self.fs, self.uv, self.maxint = x, fs, uv, maxint
        self.n0, self.n1, self.npd = (int(round(v / 1e3 * fs)) for v in (PRE, POST, PAD))
        self.tms = np.arange(-self.n0, self.n1) / fs * 1e3
        self.sos = butter(3, 300, 'highpass', fs=fs, output='sos')
        self.bm = (self.tms > BASE[0]) & (self.tms < BASE[1])
        self.ed = np.arange(-PRE, POST + 1, 1.0)
        self.bi = np.clip(np.searchsorted(self.ed, self.tms, side='right') - 1, 0, len(self.ed) - 2)
        self.ctr = self.ed[:-1] + 0.5

    def raw(self, t):
        s = int(round(t * self.fs))
        return np.asarray(self.x[s - self.n0 - self.npd:s + self.n1 + self.npd, :384]), s

    def art_end(self, w, chs):
        """sample index (in the cut window) where the artefact ends: last saturated sample in 0-8 ms + 0.5 ms, >= 4 ms"""
        i0 = self.npd + self.n0
        seg = w[i0:i0 + int(8e-3 * self.fs), chs]
        sat = (np.abs(seg) >= self.maxint - 1).any(1)
        e = (np.flatnonzero(sat).max() + 1) if sat.any() else 0
        return max(e + int(0.5e-3 * self.fs), int(4e-3 * self.fs)) + self.n0       # index in cut window

    def run(self, w, chs, bridge_end=None):
        """w: int16 (padded window, 384 ch) -> float32 high-passed CAR (cut window, chs)"""
        f = w[:, chs].astype(np.float32) * self.uv
        npd, n0 = self.npd, self.n0
        f -= np.median(f[npd:npd + n0 + self.n1][self.bm], 0)
        if bridge_end is not None:
            b0, b1 = npd + n0 - int(0.5e-3 * self.fs), npd + bridge_end
            lam = np.linspace(0, 1, b1 - b0 + 1)[1:-1, None]
            f[b0 + 1:b1] = f[b0] * (1 - lam) + f[b1] * lam
        f -= np.median(f, 1, keepdims=True)
        h = sosfiltfilt(self.sos, f, axis=0).astype(np.float32)
        return h[npd:npd + n0 + self.n1]

    def gain(self, h, start=None):
        """noise-based gain per sample; NaN before start"""
        nb = len(self.ctr)
        ma = np.zeros((nb, h.shape[1]), np.float32)
        for k in range(nb):
            ma[k] = np.median(np.abs(h[self.bi == k]), 0)
        base = np.median(ma[(self.ctr > BASE[0]) & (self.ctr < BASE[1])], 0)
        g = np.median(ma / np.maximum(base, 1e-6), 1)
        g = pd.Series(g).rolling(3, center=True, min_periods=1).median().to_numpy()
        gs = np.interp(self.tms, self.ctr, g)
        if start is not None:
            gs[:start] = np.nan
            first = np.searchsorted(self.ctr - 0.5, self.tms[start])          # first full bin after the artefact end
            if first < nb:
                gs[start:] = np.interp(self.tms[start:], self.ctr[first:], g[first:])
        return gs.astype(np.float32)

    def gain_hf(self, w, chs, start=None):
        """gain from 6-9.5 kHz noise of the raw signal (09_gain_estimators: least driven by responses); w = padded int16"""
        if not hasattr(self, 'sos_hf'):
            self.sos_hf = butter(4, [6000, 9500], 'bandpass', fs=self.fs, output='sos')
        f = w[:, chs].astype(np.float32) * self.uv
        npd, n0 = self.npd, self.n0
        f -= np.median(f[npd:npd + n0 + self.n1][self.bm], 0)
        if start is not None:
            b0, b1 = npd + n0 - int(0.5e-3 * self.fs), npd + start
            lam = np.linspace(0, 1, b1 - b0 + 1)[1:-1, None]
            f[b0 + 1:b1] = f[b0] * (1 - lam) + f[b1] * lam
        hf = sosfiltfilt(self.sos_hf, f, axis=0)[npd:npd + n0 + self.n1]
        return self.gain(hf, start)


GAIN = sys.argv[2] if len(sys.argv) > 2 else 'mad'


def templates(P, H, st, sc, cch, chs, onsets):
    """per unit: mean waveform on peak channel +-4 (indices into chs) from KS4 spikes in the pre-stimulus windows"""
    ch_index = {c: i for i, c in enumerate(chs)}
    L0, L1 = int(0.6e-3 * P.fs), int(1.4e-3 * P.fs)
    acc, cnt, chset = {}, {}, {}
    for k, t in enumerate(onsets):
        s = int(round(t * P.fs))
        i0, i1 = np.searchsorted(st, [s - P.n0, s + P.n1])
        for sp, cl in zip(st[i0:i1] - s + P.n0, sc[i0:i1]):
            tm = P.tms[min(sp, len(P.tms) - 1)]
            if not (BASE[0] < tm < BASE[1]) or sp - L0 < 0 or sp + L1 > len(P.tms):
                continue
            pc = cch[cl] if cl < len(cch) else -1
            if pc not in ch_index:
                continue
            j = ch_index[pc]
            cs = np.arange(max(0, j - 4), min(len(chs), j + 5))
            if len(cs) < 9:
                b = int(np.clip(j - 4, 0, len(chs) - 9)); cs = np.arange(b, b + 9)
            cl = int(cl)
            seg = H[k][sp - L0:sp + L1][:, cs]
            if cl not in acc:
                acc[cl] = np.zeros(seg.shape, np.float64); cnt[cl] = 0; chset[cl] = cs
            acc[cl] += seg; cnt[cl] += 1
    T = {cl: ((acc[cl] / cnt[cl]).astype(np.float32), chset[cl]) for cl in acc if cnt[cl] >= 30}
    return T, L0


def detect(h, tmpl, cs, theta, L0, valid=None, fs=30000.):
    """matched-filter amplitude, peaks >= theta; returns spike sample indices (trough-aligned) and amplitudes.
    valid: per-sample mask of the spike (trough) time"""
    tn = (tmpl ** 2).sum()
    L = len(tmpl)
    a = fftconvolve(h[:, cs], tmpl[::-1], mode='full', axes=0).sum(1)[L - 1:L - 1 + len(h)] / tn
    # a[i] = <h[i:i+L], T>/<T,T>; spike trough at i + L0; template must lie fully inside the data
    a[len(h) - L + 1:] = 0
    if valid is not None:
        vv = np.zeros(len(a), bool); vv[:len(h) - L0] = valid[L0:]
        a = np.where(vv, a, 0)
    pk, pr = find_peaks(a, height=theta, distance=int(1e-3 * fs))
    return pk + L0, pr['peak_heights']


def match(det, truth, tol):
    """greedy match of sorted sample arrays; returns n_matched"""
    if len(det) == 0 or len(truth) == 0:
        return 0, np.zeros(len(det), bool)
    j = np.searchsorted(truth, det)
    d = np.minimum(np.abs(det - truth[np.clip(j - 1, 0, len(truth) - 1)]), np.abs(det - truth[np.clip(j, 0, len(truth) - 1)]))
    ok = d <= tol
    return int(ok.sum()), ok


def main(ci):
    sid, probe = CASES[ci]
    tag = f'{sid}_{probe}' + ('' if GAIN == 'mad' else f'_{GAIN}')
    os.makedirs(OUT, exist_ok=True)
    r, x, fs, uv, maxint, tw, ta, st, sc, cch = load(sid, probe)
    P = Prep(x, fs, uv, maxint)
    tol = int(TOL_MS * 1e-3 * fs)
    print(tag, 'whisker', len(tw), 'auditory', len(ta), flush=True)

    # ---- affected channels: slow-recovery whisker trials, high-pass ratio 6-18 ms < 0.7 ----------------------------
    allch = np.arange(384)
    probe_tr = tw[np.linspace(0, len(tw) - 1, 60).round().astype(int)]
    rat = []
    for t in probe_tr:
        w, _ = P.raw(t)
        e = P.art_end(w, allch)
        h = P.run(w, allch, e)
        rr = np.sqrt((h[(P.tms > 6) & (P.tms < 18)] ** 2).mean(0) / np.maximum((h[P.bm] ** 2).mean(0), 1e-9))
        rat.append(rr)
    rat = np.array(rat)
    slow = np.median(rat, 1) < np.median(np.median(rat, 1))                      # the more attenuated half of trials
    chr_ = np.median(rat[slow], 0)
    chs = np.flatnonzero(chr_ < 0.7)
    print('affected channels', len(chs), 'of 384; median ratio', np.median(chr_[chs]).round(2), flush=True)

    # ---- preprocess all trials ---------------------------------------------------------------------------------------
    def prep_w(t):
        w, s = P.raw(t)
        e = P.art_end(w, chs)
        h = P.run(w, chs, e)
        return h, e, s
    Wd = Parallel(n_jobs=24, prefer='threads')(delayed(prep_w)(t) for t in tw)
    Hw = [d[0] for d in Wd]; Ew = np.array([d[1] for d in Wd]); Sw = np.array([d[2] for d in Wd])
    Ad = Parallel(n_jobs=24, prefer='threads')(delayed(lambda t: (P.run(P.raw(t)[0], chs), P.raw(t)[1]))(t) for t in ta)
    Ha = [d[0] for d in Ad]; Sa = np.array([d[1] for d in Ad])
    if GAIN == 'hf':
        Gw = np.array(Parallel(n_jobs=24, prefer='threads')(delayed(lambda t, e: P.gain_hf(P.raw(t)[0], chs, e))(t, e) for t, e in zip(tw, Ew)))
        Ga = np.array(Parallel(n_jobs=24, prefer='threads')(delayed(lambda t: P.gain_hf(P.raw(t)[0], chs))(t) for t in ta))
    else:
        Gw = np.array(Parallel(n_jobs=24, prefer='threads')(delayed(P.gain)(h, e) for h, e in zip(Hw, Ew)))
        Ga = np.array(Parallel(n_jobs=24, prefer='threads')(delayed(P.gain)(h) for h in Ha))
    ga = np.nanmedian(Ga, 0)[(P.tms > 5) & (P.tms < 50)]
    print('gain auditory (median trials) 5-50 ms: min %.2f max %.2f' % (ga.min(), ga.max()), flush=True)

    # ---- templates + per-unit thresholds on clean data -------------------------------------------------------------
    T, L0 = templates(P, Hw + Ha, st, sc, cch, chs, np.r_[tw, ta])
    print('units with templates', len(T), flush=True)

    def ks4_times(cl, s):
        i0, i1 = np.searchsorted(st, [s - P.n0, s + P.n1])
        return np.sort(st[i0:i1][sc[i0:i1] == cl] - s + P.n0)
    clean_valid_w = P.bm                                                           # pre-stimulus of whisker trials
    thetas = np.arange(0.25, 0.95, 0.05)

    def calib(cl):
        tmpl, cs = T[cl]
        tp = np.zeros(len(thetas)); nd = np.zeros(len(thetas)); nt = 0
        for H, S, valid in [(Hw, Sw, clean_valid_w), (Ha, Sa, None)]:
            for h, s in zip(H[:80], S[:80]):
                tru = ks4_times(cl, s)
                if valid is not None:
                    tru = tru[valid[np.clip(tru, 0, len(valid) - 1)]]
                nt += len(tru)
                a_pk, a_h = detect(h, tmpl, cs, thetas[0], L0, valid)
                for k, th in enumerate(thetas):
                    d = a_pk[a_h >= th]
                    tp[k] += match(d, tru, tol)[0]; nd[k] += len(d)
        f1 = 2 * tp / np.maximum(nd + nt, 1)
        k = int(np.argmax(f1))
        return cl, thetas[k], f1[k], tp[k] / max(nt, 1), tp[k] / max(nd[k], 1), nt
    cal = Parallel(n_jobs=48, prefer='threads')(delayed(calib)(cl) for cl in T)
    U = pd.DataFrame(cal, columns=['cluster_id', 'theta', 'f1_clean', 'recall_clean', 'precision_clean', 'n_ks4_clean'])
    U.insert(0, 'probe', probe); U.insert(0, 'session_id', sid)
    good = U[U.f1_clean >= 0.6].cluster_id.to_numpy()
    print('units clean F1 >= 0.6:', len(good), 'of', len(U), '; median F1', U.f1_clean.median().round(2), flush=True)
    th = dict(zip(U.cluster_id, U.theta))

    # ---- synthetic test --------------------------------------------------------------------------------------------
    pair = rng.integers(0, len(tw), len(ta))
    nS = len(P.tms)
    bins5 = np.arange(5, 55, 5)

    def synth(k):
        w, s = P.raw(ta[k])
        j = pair[k]; e = Ew[j]; gtrue = np.nan_to_num(Gw[j], nan=1.0)
        gfull = np.ones(len(w), np.float32)
        gfull[P.npd + e:P.npd + nS] = np.clip(gtrue[e:], 0, 1.5)
        ws = np.round(w.astype(np.float32) * gfull[:, None]).astype(np.int16)
        hs = P.run(ws, chs)
        ghat = P.gain_hf(ws, chs, e) if GAIN == 'hf' else P.gain(hs, e)
        valid = np.zeros(nS, bool); valid[e:] = True
        out = []
        for cond, hh in (('none', hs), ('est', hs / np.where(ghat >= G_MIN, ghat, np.inf)[:, None]),
                         ('true', hs / np.where(gtrue >= G_MIN, gtrue, np.inf)[:, None])):
            v = valid & (((ghat if cond != 'true' else gtrue) >= G_MIN) | (cond == 'none'))
            for cl in good:
                tmpl, cs = T[cl]
                d0, _ = detect(Ha[k], tmpl, cs, th[cl], L0)
                d, _ = detect(hh.astype(np.float32), tmpl, cs, th[cl], L0, v)
                tru = ks4_times(cl, s)
                for b0 in bins5:
                    lo, hi = np.searchsorted(P.tms, [b0, b0 + 5])
                    sel = lambda z: z[(z >= lo) & (z < hi)]
                    dd, d0b, tb = sel(d), sel(d0), sel(tru)
                    vfrac = v[lo:hi].mean()
                    out.append((k, cond, cl, b0, float(np.nanmean(gtrue[lo:hi])), vfrac, len(dd), len(d0b), len(tb),
                                match(dd, d0b, tol)[0], match(dd, tb, tol)[0]))
        return out
    rows = sum(Parallel(n_jobs=48, prefer='threads')(delayed(synth)(k) for k in range(len(ta))), [])
    S = pd.DataFrame(rows, columns=['trial', 'cond', 'cluster_id', 'bin_ms', 'g_true', 'valid_frac', 'n_det', 'n_det_clean',
                                    'n_ks4', 'tp_vs_clean', 'tp_vs_ks4'])
    S.to_csv(f'{OUT}/{tag}_synthetic.csv', index=False)

    def summ(df):
        return pd.Series(dict(recall_vs_clean=df.tp_vs_clean.sum() / max(df.n_det_clean.sum(), 1),
                              precision_vs_clean=df.tp_vs_clean.sum() / max(df.n_det.sum(), 1),
                              recall_vs_ks4=df.tp_vs_ks4.sum() / max(df.n_ks4.sum(), 1),
                              rate_ratio=df.n_det.sum() / max(df.n_det_clean.sum(), 1)))
    S['g_level'] = pd.cut(S.g_true, [0, 0.2, 0.4, 0.7, 2], labels=['<0.2', '0.2-0.4', '0.4-0.7', '>0.7'])
    by_bin = S.groupby(['cond', 'bin_ms']).apply(summ).reset_index()
    by_g = S.groupby(['cond', 'g_level'], observed=True).apply(summ).reset_index()
    pd.set_option('display.width', 200)
    print(by_g.round(3).to_string(index=False), flush=True)

    # ---- real whisker trials ----------------------------------------------------------------------------------------
    def real(k):
        h, e, s, g = Hw[k], Ew[k], Sw[k], Gw[k]
        gg = np.where(np.isnan(g), 1.0, g)
        idx = np.arange(nS)
        v = (idx < P.n0 - int(0.5e-3 * fs)) | ((idx >= e) & (gg >= G_MIN))     # pre-stimulus, or after artefact with g ok
        hc = h / np.maximum(gg, G_MIN)[:, None]
        out_r, out_k = np.zeros((len(good), len(P.ctr))), np.zeros((len(good), len(P.ctr)))
        msk = np.bincount(P.bi, weights=v.astype(float), minlength=len(P.ctr)) / np.bincount(P.bi, minlength=len(P.ctr))
        for u, cl in enumerate(good):
            tmpl, cs = T[cl]
            d, _ = detect(hc.astype(np.float32), tmpl, cs, th[cl], L0, v)
            out_r[u] = np.bincount(P.bi[d[d < nS]], minlength=len(P.ctr))
            tk = ks4_times(cl, s); tk = tk[(tk >= 0) & (tk < nS)]
            out_k[u] = np.bincount(P.bi[tk], minlength=len(P.ctr))
        return out_r, out_k, msk
    R = Parallel(n_jobs=48, prefer='threads')(delayed(real)(k) for k in range(len(tw)))
    rec = np.array([q[0] for q in R]); ksw = np.array([q[1] for q in R]); vm = np.array([q[2] for q in R])

    def audk(k):
        o = np.zeros((len(good), len(P.ctr)))
        for u, cl in enumerate(good):
            tk = ks4_times(cl, Sa[k]); tk = tk[(tk >= 0) & (tk < nS)]
            o[u] = np.bincount(P.bi[tk], minlength=len(P.ctr))
        return o
    ksa = np.array(Parallel(n_jobs=48, prefer='threads')(delayed(audk)(k) for k in range(len(ta))))
    # rates per unit (Hz), recovered: divide by valid coverage per bin
    cov = vm.sum(0)
    psth_rec = rec.sum(0) / np.maximum(cov, 1) / 1e-3
    psth_ksw = ksw.mean(0) / 1e-3
    psth_ksa = ksa.mean(0) / 1e-3
    np.savez_compressed(f'{OUT}/{tag}_psth.npz', t_ms=P.ctr, units=good, psth_recovered=psth_rec, psth_ks4_whisker=psth_ksw,
                        psth_ks4_auditory=psth_ksa, coverage=cov / len(tw), gain_whisker=Gw, gain_auditory=Ga, art_end=Ew,
                        trial_times_w=tw, trial_times_a=ta, channels=chs)
    U.to_csv(f'{OUT}/{tag}_units.csv', index=False)

    def win(p, lo, hi):
        m = (P.ctr > lo) & (P.ctr < hi)
        return float(np.nanmean(p[:, m]))
    summary = dict(session_id=sid, probe=probe, n_whisker=len(tw), n_auditory=len(ta), n_affected_channels=int(len(chs)),
                   n_units_templates=len(U), n_units_eval=int(len(good)),
                   gain_auditory_min_5_50=float(np.nanmin(np.nanmedian(Ga, 0)[(P.tms > 5) & (P.tms < 50)])),
                   gain_auditory_max_5_50=float(np.nanmax(np.nanmedian(Ga, 0)[(P.tms > 5) & (P.tms < 50)])),
                   rate_pre_recovered=win(psth_rec, -55, -5), rate_pre_ks4=win(psth_ksw, -55, -5),
                   **{f'rate_{lo}_{hi}_{nm}': win(p, lo, hi) for lo, hi in ((5, 15), (15, 30), (30, 50))
                      for nm, p in (('recovered', psth_rec), ('ks4_whisker', psth_ksw), ('ks4_auditory', psth_ksa))},
                   synthetic_by_gain=by_g.round(4).to_dict('records'))
    json.dump(summary, open(f'{OUT}/{tag}_summary.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != 'synthetic_by_gain'}, default=float, indent=0), flush=True)

    # ---- figure ----------------------------------------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(18, 9.5), gridspec_kw=dict(hspace=0.4, wspace=0.28))
    ax = axs[0, 0]
    im = ax.imshow(Gw, aspect='auto', cmap='magma', vmin=0, vmax=1.2, extent=[P.tms[0], P.tms[-1], len(tw), 0], interpolation='none')
    ax.set_xlim(-10, 60); ax.set_xlabel('time from whisker stimulus (ms)'); ax.set_ylabel('whisker trial (session order)')
    ax.set_title('A  estimated gain per trial (NaN = artefact)', loc='left', fontsize=10); plt.colorbar(im, ax=ax)
    ax = axs[0, 1]
    ax.plot(P.tms, np.nanmedian(Gw, 0), color='#f7b519', label='whisker (median trials)')
    ax.fill_between(P.tms, *np.nanpercentile(Gw, [10, 90], 0), color='#f7b519', alpha=0.2)
    ax.plot(P.tms, np.nanmedian(Ga, 0), color='#2c2cdb', label='auditory (should stay ~1)')
    ax.axhline(1, color='0.5', lw=0.5); ax.axhline(G_MIN, color='r', lw=0.5, ls='--')
    ax.set_xlim(-10, 60); ax.set_ylim(0, 1.6); ax.legend(fontsize=8, frameon=False); ax.set_xlabel('ms')
    ax.set_title('B  gain estimate: whisker vs auditory check', loc='left', fontsize=10)
    ax = axs[0, 2]
    cols = dict(none='0.5', est='#1b9e77', true='k')
    for cond in ('none', 'est', 'true'):
        d = by_bin[by_bin.cond == cond]
        ax.plot(d.bin_ms + 2.5, d.recall_vs_clean, '-o', color=cols[cond], ms=3, label=f'{cond}: recall')
        ax.plot(d.bin_ms + 2.5, d.precision_vs_clean, '--', color=cols[cond], lw=1, label=f'{cond}: precision')
    ax.set_ylim(0, 1.05); ax.set_xlabel('ms after (synthetic) stimulus'); ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.set_title('C  synthetic: attenuated auditory trials vs their clean detections', loc='left', fontsize=10)
    ax = axs[1, 0]
    for cond in ('none', 'est', 'true'):
        d = by_g[by_g.cond == cond]
        ax.plot(range(len(d)), d.recall_vs_clean, '-o', color=cols[cond], label=f'{cond} recall')
        ax.plot(range(len(d)), d.precision_vs_clean, '--', color=cols[cond], lw=1)
    ax.set_xticks(range(len(by_g[by_g.cond == 'est']))); ax.set_xticklabels(by_g[by_g.cond == 'est'].g_level.astype(str))
    ax.set_xlabel('true gain'); ax.set_ylim(0, 1.05); ax.legend(fontsize=8, frameon=False)
    ax.set_title('D  synthetic recall (solid) / precision (dashed) by gain', loc='left', fontsize=10)
    ax = axs[1, 1]
    ax.plot(P.ctr, np.nanmean(psth_ksa, 0), color='#2c2cdb', label='KS4 auditory')
    ax.plot(P.ctr, np.nanmean(psth_ksw, 0), color='0.5', label='KS4 whisker')
    ax.plot(P.ctr, np.nanmean(psth_rec, 0), color='#d95f02', label='recovered whisker')
    ax.axvspan(0, 5, color='0.88', zorder=0); ax.set_xlim(-30, 70); ax.set_xlabel('ms'); ax.set_ylabel('mean rate per unit (Hz)')
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(f'E  real data, {len(good)} units (clean F1 >= 0.6)', loc='left', fontsize=10)
    ax = axs[1, 2]
    ax.plot(P.ctr, cov / len(tw), color='k'); ax.set_ylim(0, 1.05); ax.set_xlim(-30, 70); ax.set_xlabel('ms')
    ax.set_title('F  fraction of whisker trials usable per bin (g >= %.2f)' % G_MIN, loc='left', fontsize=10)
    fig.suptitle(f'{tag} (NP1.0) - TENTATIVE gain-recovery pilot: {len(chs)} affected channels, {len(tw)} whisker / {len(ta)} auditory trials',
                 fontsize=11)
    fig.savefig(f'{OUT}/{tag}_fig.png', dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    main(int(sys.argv[1]))
