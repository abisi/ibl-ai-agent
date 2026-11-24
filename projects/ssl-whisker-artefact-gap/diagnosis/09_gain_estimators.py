"""TENTATIVE: compare gain estimators that should not be driven by neural activity (read-only, 2026-11-23).

Candidates (per trial, 1-ms bins, ratio to the trial's own -55..-5 ms, median over affected channels unless stated):
  mad      median |x| of the 300 Hz high-passed, common-median-referenced signal (pilot 08; inflated by responses)
  q20      20th percentile of |x| of the same signal (less sensitive to spikes / hash?)
  hf       median |x| after a 6-9.5 kHz band-pass of the raw signal per channel (no CAR): mostly amplifier noise, little
           spike power (NP1.0 AP hardware low-pass ~10 kHz)
  cm       common mode: median over channels of the raw signal, 300 Hz high-passed, RMS (MH037: ~100 uV interference)
Each estimate is smoothed with a 3-bin running median.
Checks:
  1. auditory trials (evoked response, true gain 1): median over trials should stay ~1 at 5-50 ms;
  2. real whisker trials, ground truth from spike size: at every KS4 spike of a template unit (pilot-08 templates, units
     with clean F1 >= 0.6, plus all units with >= 30 template spikes as a second set) the matched-filter amplitude
     a = <x, T>/<T, T> on the UNCORRECTED data estimates the true gain at that time (a ~ 1 at baseline). Per 2-ms bin:
     median a vs median estimate of each candidate at those spikes. KS4 only finds spikes where the gain is not too low,
     and a selects for larger spikes there, so this check covers the partial-gain range.
Outputs: <ROOT>/gain_recovery_pilot/<tag>_estimators.{png,csv,json}
Usage: python 09_gain_estimators.py <0|1>
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
from scipy.signal import butter, fftconvolve, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p08', os.path.join(HERE, '08_gain_recovery_pilot.py'))
p08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p08)
OUT = p08.OUT
EST = ['mad', 'q20', 'hf', 'cm']
COL = dict(mad='0.5', q20='#7570b3', hf='#1b9e77', cm='#d95f02')


def binned(P, v, stat):
    """v: time x ch (or time) -> bins x ch"""
    v = v if v.ndim == 2 else v[:, None]
    out = np.zeros((len(P.ctr), v.shape[1]), np.float32)
    for k in range(len(P.ctr)):
        out[k] = stat(v[P.bi == k])
    return out


def ratio_curve(P, b, start):
    base = np.median(b[(P.ctr > p08.BASE[0]) & (P.ctr < p08.BASE[1])], 0)
    g = np.median(b / np.maximum(base, 1e-9), 1)
    g = pd.Series(g).rolling(3, center=True, min_periods=1).median().to_numpy()
    gs = np.interp(P.tms, P.ctr, g).astype(np.float32)
    if start is not None:
        first = np.searchsorted(P.ctr - 0.5, P.tms[start])
        gs[:start] = np.nan
        if first < len(P.ctr):
            gs[start:] = np.interp(P.tms[start:], P.ctr[first:], g[first:])
    return gs


def estimators(P, w, chs, e, sos_hf):
    """w: padded int16 window; e: artefact end index (cut window) or None"""
    h = P.run(w, chs, e)                                               # HP + CAR (bridged if e)
    f = w[:, chs].astype(np.float32) * P.uv
    npd, n0, n1 = P.npd, P.n0, P.n1
    f -= np.median(f[npd:npd + n0 + n1][P.bm], 0)
    if e is not None:
        b0, b1 = npd + n0 - int(0.5e-3 * P.fs), npd + e
        lam = np.linspace(0, 1, b1 - b0 + 1)[1:-1, None]
        f[b0 + 1:b1] = f[b0] * (1 - lam) + f[b1] * lam
    hf = sosfiltfilt(sos_hf, f, axis=0)[npd:npd + n0 + n1]
    cm = sosfiltfilt(P.sos, np.median(f, 1))[npd:npd + n0 + n1]
    g = dict(mad=ratio_curve(P, binned(P, np.abs(h), lambda z: np.median(z, 0)), e),
             q20=ratio_curve(P, binned(P, np.abs(h), lambda z: np.percentile(z, 20, 0)), e),
             hf=ratio_curve(P, binned(P, np.abs(hf), lambda z: np.median(z, 0)), e),
             cm=ratio_curve(P, binned(P, cm ** 2, lambda z: np.sqrt(z.mean(0))), e))
    return h, g


def main(ci):
    sid, probe = p08.CASES[ci]
    tag = f'{sid}_{probe}'
    r, x, fs, uv, maxint, tw, ta, st, sc, cch = p08.load(sid, probe)
    P = p08.Prep(x, fs, uv, maxint)
    sos_hf = butter(4, [6000, 9500], 'bandpass', fs=fs, output='sos')
    psth = np.load(f'{OUT}/{tag}_psth.npz')
    chs = psth['channels']
    units = pd.read_csv(f'{OUT}/{tag}_units.csv')

    def one(t, whisker):
        w, s = P.raw(t)
        e = P.art_end(w, chs) if whisker else None
        h, g = estimators(P, w, chs, e, sos_hf)
        return h, g, e, s
    Rw = Parallel(n_jobs=32, prefer='threads')(delayed(one)(t, True) for t in tw)
    Ra = Parallel(n_jobs=32, prefer='threads')(delayed(one)(t, False) for t in ta)
    Gw = {k: np.array([q[1][k] for q in Rw]) for k in EST}
    Ga = {k: np.array([q[1][k] for q in Ra]) for k in EST}
    w550 = (P.tms > 5) & (P.tms < 50)
    res = dict(session_id=sid, probe=probe)
    for k in EST:
        ma = np.nanmedian(Ga[k], 0)[w550]
        res[f'aud_{k}_min'] = float(ma.min()); res[f'aud_{k}_max'] = float(ma.max())
        res[f'whisk_{k}_median_6_18'] = float(np.nanmedian(Gw[k][:, (P.tms > 6) & (P.tms < 18)]))
    print(json.dumps(res, indent=0), flush=True)

    # ---- spike-size ground truth on real whisker trials --------------------------------------------------------------
    Hw = [q[0] for q in Rw]; Ew = np.array([q[2] for q in Rw]); Sw = np.array([q[3] for q in Rw])
    T, L0 = p08.templates(P, Hw + [q[0] for q in Ra], st, sc, cch, chs, np.r_[tw, ta])
    good = set(units[units.f1_clean >= 0.6].cluster_id)
    rows = []
    L = None
    for k in range(len(tw)):
        s = Sw[k]
        i0, i1 = np.searchsorted(st, [s - P.n0, s + P.n1])
        for sp, cl in zip(st[i0:i1] - s + P.n0, sc[i0:i1]):
            cl = int(cl)
            if cl not in T:
                continue
            tmpl, cs = T[cl]
            L = len(tmpl)
            a0 = sp - L0
            if a0 < 0 or a0 + L > len(P.tms) or (P.tms[sp] > -1 and a0 < Ew[k]):
                continue                                            # skip spikes overlapping the artefact
            seg = Hw[k][a0:a0 + L][:, cs]
            amp = float((seg * tmpl).sum() / (tmpl ** 2).sum())
            rows.append((k, cl, cl in good, float(P.tms[sp]), amp, *[float(Gw[q][k, sp]) for q in EST]))
    A = pd.DataFrame(rows, columns=['trial', 'cluster_id', 'good', 't_ms', 'amp'] + [f'g_{q}' for q in EST])
    A.to_csv(f'{OUT}/{tag}_estimators_spikes.csv', index=False)
    A['tb'] = (A.t_ms // 2) * 2 + 1
    agg = A.groupby(['good', 'tb']).agg(n=('amp', 'size'), amp=('amp', 'median'),
                                        **{f'g_{q}': (f'g_{q}', 'median') for q in EST}).reset_index()
    agg.to_csv(f'{OUT}/{tag}_estimators.csv', index=False)
    # error of each estimator vs spike amplitude within 5-45 ms (bins with >= 30 spikes), all template units
    for gd in (True, False):
        m = (agg.good == gd) & (agg.tb > 5) & (agg.tb < 45) & (agg.n >= 30)
        for q in EST:
            res[f'{"good" if gd else "all"}_{q}_median_ratio_g_over_amp'] = float(np.median(agg.loc[m, f'g_{q}'] / agg.loc[m, 'amp']))
        res[f'{"good" if gd else "all"}_n_bins'] = int(m.sum())
    base = A[(A.t_ms > -55) & (A.t_ms < -5)]
    res['amp_baseline_median'] = float(base.amp.median())
    json.dump(res, open(f'{OUT}/{tag}_estimators.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if 'ratio' in k or 'n_bins' in k or 'baseline' in k}, indent=0), flush=True)

    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8))
    for q in EST:
        axs[0].plot(P.tms, np.nanmedian(Ga[q], 0), color=COL[q], label=q)
        axs[1].plot(P.tms, np.nanmedian(Gw[q], 0), color=COL[q], label=q)
    for ax, ttl in zip(axs[:2], ('auditory trials (true gain = 1)', 'whisker trials (median)')):
        ax.axhline(1, color='k', lw=0.5); ax.set_xlim(-10, 60); ax.set_ylim(0, 2); ax.set_xlabel('ms'); ax.legend(fontsize=8, frameon=False)
        ax.set_title(ttl, fontsize=10, loc='left')
    ax = axs[2]
    d = agg[(~agg.good | agg.good) & (agg.n >= 30)].groupby('tb').apply(
        lambda z: pd.Series({c: np.average(z[c], weights=z.n) for c in ['amp'] + [f'g_{q}' for q in EST]})).reset_index()
    ax.plot(d.tb, d.amp, 'k-', lw=2, label='KS4-spike amplitude (truth)')
    for q in EST:
        ax.plot(d.tb, d[f'g_{q}'], color=COL[q], label=f'{q} at those spikes')
    ax.axhline(1, color='0.5', lw=0.5); ax.set_xlim(-10, 60); ax.set_ylim(0, 2); ax.set_xlabel('ms'); ax.legend(fontsize=8, frameon=False)
    ax.set_title('real whisker trials: gain at KS4 spikes vs spike amplitude', fontsize=10, loc='left')
    fig.suptitle(f'{tag}: candidate gain estimators (TENTATIVE)', fontsize=11)
    fig.savefig(f'{OUT}/{tag}_estimators.png', dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    main(int(sys.argv[1]))
