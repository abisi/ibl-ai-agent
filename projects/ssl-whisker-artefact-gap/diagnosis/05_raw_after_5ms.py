"""Raw data after the whisker artefact on the flagged probes: is any neural signal left from +5 ms? (read-only, 2026-11-23)

Per probe (flagged MH037_20250524 imec0-3, MH020_20250304 imec0; control MH031_20250507 imec3, normal artefact), from
the raw SpikeGLX .ap.bin, same trials and alignment as screen/01_screen_probe.py (40 active whisker, 40 active auditory):
  A-D  one example trial, 12 most spike-active channels: raw (uV, per-channel baseline median removed) and 300 Hz
       high-pass (Butterworth-3 sosfiltfilt, pulse -0.5..+4 ms linearly bridged first), whisker vs auditory;
  E-F  best channel, high-pass, 12 whisker and 12 auditory trials stacked;
  G    negative threshold crossings (all good channels, 1-ms refractory) with a FIXED threshold, -4.5 x baseline sigma
       (sigma = MAD/0.6745 over -45..-5 ms): rate per channel, whisker vs auditory;
  H    same with a LOCAL threshold, -4.5 x sigma measured in the same 2-ms bin over all trials: if the front end only
       lost gain, attenuated spikes still cross a local threshold at the baseline rate; pure noise does not;
  I    KS4 spikes (all clusters) per 1-ms bin;
  J    mean waveform (local-sigma units) of local-threshold events at baseline vs at +6..+30 ms (whisker): spike shape
       or noise; median absolute amplitude ratio (post / baseline) in uV.
Outputs: <OUT>/raw_after_5ms/<session>_<probe>.png/.pdf + summary_raw_after_5ms.csv (one row per probe).
Usage: python 05_raw_after_5ms.py            (all probes, serial)   |   python 05_raw_after_5ms.py <i>   (one probe)
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
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('scr', os.path.join(HERE, '..', 'screen', '01_screen_probe.py'))
scr = importlib.util.module_from_spec(spec); spec.loader.exec_module(scr)

ROOT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap'
OUT = f'{ROOT}/raw_after_5ms'
PROBES = [('MH037_20250524_143522', 'imec0'), ('MH037_20250524_143522', 'imec1'), ('MH037_20250524_143522', 'imec2'),
          ('MH037_20250524_143522', 'imec3'), ('MH020_20250304_161732', 'imec0'), ('MH031_20250507_104425', 'imec3')]
PRE, POST = 60, 80                       # ms snippet
VIEW = (-20, 60)                         # ms shown
BASE = (-45, -5)
BRIDGE = (-0.5, 4.0)
POSTWIN = (6, 30)                        # window for post-artefact events
THR = 4.5
CW, CA = '#f7b519', '#2c2cdb'


def load_probe(sid, probe):
    inv = pd.read_csv(f'{scr.OUT}/tables/00_inventory.csv')
    r = inv[(inv.session_id == sid) & (inv.probe == probe)].iloc[0]
    md = scr.meta(r.raw_bin[:-4] + '.meta')
    fs = float(md['imSampRate']); nch = int(md['nSavedChans']); maxint = int(md['imMaxInt'])
    ptype = md.get('imDatPrb_type', '0')
    gain = float(md.get('imChan0apGain', 80 if ptype != '0' else 500))
    uv = float(md['imAiRangeMax']) / maxint / gain * 1e6
    x = np.memmap(r.raw_bin, dtype='int16', mode='r').reshape(-1, nch)
    tr = scr.read_trials(sid)
    ft, a, b, *_ = scr.align(tr, r.whisker_times_file)
    on = {}
    for m, col in (('whisker', 'whisker_stim'), ('auditory', 'auditory_stim')):
        t, _, _ = scr.pick(tr, col)
        tf = a * t + b
        if m == 'whisker':
            q = np.abs(ft[None, :] - tf[:, None]).argmin(1)
            tf = np.where(np.abs(ft[q] - tf) < 0.005, ft[q], tf)
        on[m] = tf
    return r, x, fs, uv, maxint, 'NP1.0' if ptype == '0' else f'NP2.0 ({ptype})', on


def snippets(x, fs, uv, onsets):
    n0, n1 = int(round(PRE / 1e3 * fs)), int(round(POST / 1e3 * fs))
    tms = np.arange(-n0, n1) / fs * 1e3
    b0, b1 = int(round(BRIDGE[0] / 1e3 * fs)) + n0, int(round(BRIDGE[1] / 1e3 * fs)) + n0
    sos = butter(3, 300, 'highpass', fs=fs, output='sos')
    bm = (tms > BASE[0]) & (tms < BASE[1])
    raw, hp = [], []
    for t in onsets:
        s = int(round(t * fs))
        if s - n0 < 0 or s + n1 > x.shape[0]:
            continue
        w = np.asarray(x[s - n0:s + n1, :384]).astype(np.float32) * uv
        w -= np.median(w[bm], 0)
        wb = w.copy()
        lam = np.linspace(0, 1, b1 - b0 + 1)[1:-1, None]
        wb[b0 + 1:b1] = wb[b0] * (1 - lam) + wb[b1] * lam
        raw.append(w); hp.append(sosfiltfilt(sos, wb - wb.mean(0), axis=0).astype(np.float32))
    return tms, np.stack(raw), np.stack(hp)


def crossings(hp, thr):
    """negative crossings of thr (broadcast to hp: trials x time x ch), 1-ms refractory handled by onset-only + gap"""
    below = hp < thr
    on = below[:, 1:] & ~below[:, :-1]
    return np.concatenate([np.zeros_like(on[:, :1]), on], 1)


def analyse(sid, probe):
    r, x, fs, uv, maxint, ptype, on = load_probe(sid, probe)
    tms, raw_w, hp_w = snippets(x, fs, uv, on['whisker'])
    _, raw_a, hp_a = snippets(x, fs, uv, on['auditory'])
    bm = (tms > BASE[0]) & (tms < BASE[1])
    sig_base = np.median(np.abs(np.concatenate([hp_w[:, bm], hp_a[:, bm]], 0)), axis=(0, 1)) / 0.6745    # per channel
    good = sig_base > 0.3 * np.median(sig_base)
    edges2 = np.arange(-PRE, POST + 1, 2.0)
    bi2 = np.clip(np.searchsorted(edges2, tms, side='right') - 1, 0, len(edges2) - 2)

    def local_sigma(hp):
        s = np.zeros((len(edges2) - 1, hp.shape[2]), np.float32)
        for k in range(len(edges2) - 1):
            s[k] = np.median(np.abs(hp[:, bi2 == k]), axis=(0, 1)) / 0.6745
        return s[bi2]                                                     # time x ch

    sl_w, sl_a = local_sigma(hp_w), local_sigma(hp_a)
    ev = {}
    for m, hp, sl in (('whisker', hp_w, sl_w), ('auditory', hp_a, sl_a)):
        cf = crossings(hp[:, :, good], -THR * sig_base[good])
        cl = crossings(hp[:, :, good], -THR * np.maximum(sl[:, good], 1e-3))
        ev[m] = (cf, cl)
    edges1 = np.arange(-PRE, POST + 1, 1.0)
    bi1 = np.clip(np.searchsorted(edges1, tms, side='right') - 1, 0, len(edges1) - 2)
    ctr = edges1[:-1] + 0.5
    rate = {}
    for m in ev:
        for j, nm in enumerate(('fixed', 'local')):
            c = ev[m][j].sum(axis=(0, 2)).astype(float)
            rate[m, nm] = np.bincount(bi1, weights=c, minlength=len(ctr)) / ev[m][j].shape[0] / good.sum() / 1e-3
    # KS4 spikes
    ks = {}
    st = np.load(f'{r.ks4_dir}/sorter_output/spike_times.npy', mmap_mode='r').ravel()
    for m in on:
        cnt = np.zeros(len(ctr))
        for t in on[m]:
            s = t * fs
            i0, i1 = np.searchsorted(st, [s - PRE * fs / 1e3, s + POST * fs / 1e3])
            cnt += np.histogram((np.asarray(st[i0:i1]) - s) / fs * 1e3, edges1)[0]
        ks[m] = cnt / len(on[m]) / 1e-3
    # waveforms of local-threshold events: baseline vs post window (whisker)
    gidx = np.flatnonzero(good)
    hpg = hp_w[:, :, good]; slg = sl_w[:, good]
    cl = ev['whisker'][1]
    h0, h1 = int(round(1e-3 * fs)), int(round(2e-3 * fs))
    wf = {}
    for nm, (lo, hi) in (('baseline', BASE), ('post', POSTWIN)):
        tt, ii, cc = np.nonzero(cl & ((tms >= lo) & (tms < hi))[None, :, None])
        keep = (ii - h0 >= 0) & (ii + h1 < len(tms))
        tt, ii, cc = tt[keep], ii[keep], cc[keep]
        if len(tt) > 5000:
            sel = np.random.default_rng(0).choice(len(tt), 5000, replace=False); tt, ii, cc = tt[sel], ii[sel], cc[sel]
        # align on the trough within 0.5 ms
        segs, amps = [], []
        for t_, i_, c_ in zip(tt, ii, cc):
            j = i_ + int(np.argmin(hpg[t_, i_:i_ + int(0.5e-3 * fs), c_]))
            if j - h0 < 0 or j + h1 >= len(tms):
                continue
            s_ = hpg[t_, j - h0:j + h1, c_]
            segs.append(s_ / slg[j, c_]); amps.append(-s_[h0])
        wf[nm] = (np.array(segs), np.array(amps), len(tt))
    wt = np.arange(-h0, h1) / fs * 1e3

    def ratio(v, lo, hi):
        return float(v[(ctr > lo) & (ctr < hi)].mean() / max(v[(ctr > BASE[0]) & (ctr < BASE[1])].mean(), 1e-9))
    row = dict(mouse_id=sid[:5], session_id=sid, probe=probe, probe_type=ptype, n_whisker=len(hp_w), n_auditory=len(hp_a),
               n_good_channels=int(good.sum()), base_sigma_uV=float(np.median(sig_base[good])),
               post_sigma_uV_whisker_6_18=float(np.median(sl_w[(tms > 6) & (tms < 18)][:, good])),
               post_sigma_uV_auditory_6_18=float(np.median(sl_a[(tms > 6) & (tms < 18)][:, good])))
    for lo, hi in ((5, 10), (10, 20), (20, 30), (30, 40)):
        for m in ('whisker', 'auditory'):
            row[f'fixed_thr_rate_ratio_{m}_{lo}_{hi}'] = ratio(rate[m, 'fixed'], lo, hi)
            row[f'local_thr_rate_ratio_{m}_{lo}_{hi}'] = ratio(rate[m, 'local'], lo, hi)
            row[f'ks4_rate_ratio_{m}_{lo}_{hi}'] = ratio(ks[m], lo, hi)
    if len(wf['post'][0]) > 10 and len(wf['baseline'][0]) > 10:
        mb, mp = wf['baseline'][0].mean(0), wf['post'][0].mean(0)
        row.update(n_events_baseline=int(len(wf['baseline'][0])), n_events_post=int(len(wf['post'][0])),
                   waveform_corr_post_vs_base=float(np.corrcoef(mb, mp)[0, 1]),
                   amp_uV_baseline=float(np.median(wf['baseline'][1])), amp_uV_post=float(np.median(wf['post'][1])))
    plot(sid, probe, ptype, tms, raw_w, raw_a, hp_w, hp_a, sig_base, good, ctr, rate, ks, wf, wt, row)
    return row


def plot(sid, probe, ptype, tms, raw_w, raw_a, hp_w, hp_a, sig_base, good, ctr, rate, ks, wf, wt, row):
    bm = (tms > BASE[0]) & (tms < BASE[1])
    act = np.zeros(384)
    act[good] = (np.concatenate([hp_w[:, bm], hp_a[:, bm]], 0) < -THR * sig_base).sum((0, 1))[good]
    top = np.sort(np.argsort(act)[::-1][:12])
    best = int(np.argmax(act))
    v = (tms >= VIEW[0]) & (tms <= VIEW[1])
    # example trial: the one whose 6-18 ms high-pass RMS on top channels is the median
    def ex(hp):
        rr = np.sqrt((hp[:, (tms > 6) & (tms < 18)][:, :, top] ** 2).mean((1, 2)))
        return int(np.argsort(rr)[len(rr) // 2])
    iw, ia = ex(hp_w), ex(hp_a)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3)

    def stack(ax, d, step, col, title, ylab):
        for k, c in enumerate(top):
            ax.plot(tms[v], d[v, c] - k * step, color=col, lw=0.6)
        ax.set_yticks(-np.arange(len(top)) * step); ax.set_yticklabels([str(c) for c in top], fontsize=7)
        ax.axvspan(0, 5, color='0.85', zorder=0); ax.axvline(0, color='k', lw=0.5)
        ax.set_title(title, fontsize=10, loc='left'); ax.set_xlabel('time from stimulus (ms)'); ax.set_ylabel(ylab)
        ax.set_xlim(VIEW)

    rs = np.percentile(np.abs(raw_a[ia][v][:, top]), 99) * 2 + 1
    hs = 10 * np.median(sig_base[top])
    stack(fig.add_subplot(gs[0, 0]), raw_w[iw], rs, CW, f'A  raw, whisker trial ({rs:.0f} uV / ch)', 'channel')
    axb = fig.add_subplot(gs[0, 1]); stack(axb, raw_a[ia], rs, CA, f'B  raw, auditory trial (same scale)', '')
    stack(fig.add_subplot(gs[0, 2]), hp_w[iw], hs, CW, f'C  high-pass 300 Hz, whisker ({hs:.0f} uV / ch)', '')
    stack(fig.add_subplot(gs[0, 3]), hp_a[ia], hs, CA, 'D  high-pass 300 Hz, auditory (same scale)', '')
    for j, (hp, col, lab) in enumerate(((hp_w, CW, 'E  whisker'), (hp_a, CA, 'F  auditory'))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        n = min(12, len(hp))
        for k in range(n):
            ax.plot(tms[v], hp[k, v, best] - k * hs * 0.6, color=col, lw=0.5)
        ax.axvspan(0, 5, color='0.85', zorder=0); ax.axhline(-THR * sig_base[best], color='0.6', lw=0, )
        ax.set_title(f'{lab}: channel {best} (most spikes), high-pass, {n} trials ({hs*0.6:.0f} uV / trial)', fontsize=10, loc='left')
        ax.set_yticks([]); ax.set_xlim(VIEW); ax.set_xlabel('time from stimulus (ms)')
    for j, (key, ttl) in enumerate((('fixed', f'G  threshold crossings, fixed -{THR} x baseline sigma'),
                                    ('local', f'H  threshold crossings, local -{THR} x sigma (2-ms bins)'))):
        ax = fig.add_subplot(gs[2, j])
        ax.plot(ctr, rate['whisker', key], color=CW, label='whisker'); ax.plot(ctr, rate['auditory', key], color=CA, label='auditory')
        ax.axvspan(0, 5, color='0.85', zorder=0); ax.set_xlim(VIEW); ax.set_title(ttl, fontsize=10, loc='left')
        ax.set_xlabel('time from stimulus (ms)'); ax.set_ylabel('events / s / channel'); ax.legend(fontsize=8, frameon=False)
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(ctr, ks['whisker'], color=CW); ax.plot(ctr, ks['auditory'], color=CA)
    ax.axvspan(0, 5, color='0.85', zorder=0); ax.set_xlim(VIEW); ax.set_title('I  KS4 spikes (all clusters)', fontsize=10, loc='left')
    ax.set_xlabel('time from stimulus (ms)'); ax.set_ylabel('spikes / s (probe)')
    ax = fig.add_subplot(gs[2, 3])
    for nm, col in (('baseline', '0.3'), ('post', CW)):
        s, a, n = wf[nm]
        if len(s):
            ax.plot(wt, s.mean(0), color=col, lw=1.5, label=f'{nm} ({POSTWIN[0]}-{POSTWIN[1]} ms)' if nm == 'post' else
                    f'baseline: n={len(s)}, median {np.median(a):.0f} uV')
            if nm == 'post':
                ax.lines[-1].set_label(f'whisker +{POSTWIN[0]}-{POSTWIN[1]} ms: n={len(s)}, median {np.median(a):.0f} uV')
    ax.set_title('J  mean event waveform (local sigma units)', fontsize=10, loc='left')
    ax.set_xlabel('ms from trough'); ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f'{sid} {probe} ({ptype}): raw data after the whisker artefact. Grey band = 0-5 ms (standard artefact window). '
                 f'{row["n_whisker"]} whisker / {row["n_auditory"]} auditory active trials; {row["n_good_channels"]} good channels; '
                 f'baseline sigma {row["base_sigma_uV"]:.1f} uV, whisker 6-18 ms {row["post_sigma_uV_whisker_6_18"]:.1f} uV',
                 fontsize=11)
    os.makedirs(OUT, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{sid}_{probe}.{ext}', dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    idx = [int(sys.argv[1])] if len(sys.argv) > 1 else range(len(PROBES))
    rows = []
    for i in idx:
        sid, probe = PROBES[i]
        row = analyse(sid, probe)
        print(json.dumps(row, default=float), flush=True)
        os.makedirs(OUT, exist_ok=True)
        json.dump(row, open(f'{OUT}/{sid}_{probe}.json', 'w'), default=float)
        rows.append(row)
    if len(sys.argv) == 1:
        pd.DataFrame(rows).to_csv(f'{OUT}/summary_raw_after_5ms.csv', index=False)
