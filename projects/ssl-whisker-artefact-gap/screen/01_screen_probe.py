"""Screen one probe for the post-whisker-artefact dead time (cheap version of diagnosis 01/03).

Per probe, from the raw SpikeGLX .ap.bin (AB155: merged CatGT tcat, no raw t0 file), on N_TRIALS active whisker and
N_TRIALS active auditory trials (perf != 6, not passive; passive added only if < 10 active):
  - spike band: per channel, -0.5..+4 ms linearly bridged (both modalities, removes the pulse before filtering),
    Butterworth-3 300 Hz high-pass (sosfiltfilt), squared and averaged over trials in 1-ms bins;
    ratio(t) = median over channels of RMS(t) / RMS(pre-stimulus -45..-5 ms) for each modality, and whisker / auditory;
  - saturation: fraction of channel x samples at the ADC limit in 0-4 ms (whisker), fraction of channels saturated;
  - flatness: fraction of zero sample-to-sample differences in 6-18 ms (NP2.0 blocked amplifier);
  - artefact peak latency (alignment check);
  - KS4 spike counts (all clusters, sorter_output/spike_times.npy) in 6-18 ms vs baseline, whisker and auditory.
Trial times: NWB trials -> probe-file seconds by a linear fit on the TPrime whisker times file (as in diagnosis 01).
Output: <OUT>/cache/<session>_<probe>.npz (time courses) and <OUT>/cache/<session>_<probe>.json (summary row).
Usage: python 01_screen_probe.py <row index in 00_inventory.csv>
"""
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

OUT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap/screen'
NWB = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'
MH = '/mnt/lsens-analysis/Myriam_Hamon/data'
N_TRIALS = 40
PRE_MS, POST_MS = 60, 80                 # snippet around onset
BASE = (-45, -5)                         # pre-stimulus baseline window (ms)
WIN = (6, 18)                            # dead-time test window (ms)
BRIDGE = (-0.5, 4.0)                     # pulse bridged before filtering (ms)
REC_THR, REC_RUN = 0.8, 3                # recovery: first t >= 5 ms with REC_RUN consecutive 1-ms bins >= REC_THR
SEED = 0


def meta(p):
    d = {}
    for line in open(p):
        k, _, v = line.strip().partition('=')
        d[k.lstrip('~')] = v
    return d


def read_trials(sid):
    with h5py.File(f'{NWB}/{sid}.nwb', 'r') as f:
        tr = f['intervals/trials']
        df = pd.DataFrame({k: tr[k][:] for k in ('start_time', 'whisker_stim', 'auditory_stim', 'perf')})
        ctx = tr['context'][:] if 'context' in tr else np.array([b'nan'] * len(df))
    df['context'] = [c.decode() if isinstance(c, bytes) else str(c) for c in ctx]
    return df


def align(tr, wfile):
    ft = np.loadtxt(wfile).ravel()
    wall = tr.loc[tr.whisker_stim == 1, 'start_time'].to_numpy()
    d = ft[:, None] - wall[None, :]
    h, e = np.histogram(d.ravel(), bins=np.arange(-50, 50, 0.005))
    off = e[np.argmax(h)] + 0.0025
    jj = np.abs(d - off).argmin(1)
    ok = np.abs(d[np.arange(len(ft)), jj] - off) < 0.02
    a, b = np.polyfit(wall[jj[ok]], ft[ok], 1)
    res = ft[ok] - (a * wall[jj[ok]] + b)
    return ft, a, b, int(ok.sum()), len(ft), len(wall), float(1e3 * np.abs(res).max())


def pick(tr, col):
    sel = tr[(tr[col] == 1) & (tr.perf != 6)]
    act = sel[sel.context != 'passive']
    use = act if len(act) >= 10 else sel
    t = use.start_time.to_numpy()
    idx = np.linspace(0, len(t) - 1, min(N_TRIALS, len(t))).round().astype(int) if len(t) else np.array([], int)
    return t[idx], len(act), len(sel)


def main(i):
    inv = pd.read_csv(f'{OUT}/tables/00_inventory.csv')
    r = inv.iloc[i]
    sid, probe, run = r.session_id, r.probe, r.run
    pn = probe[4:]
    tag = f'{sid}_{probe}'
    os.makedirs(f'{OUT}/cache', exist_ok=True)
    row = dict(mouse_id=r.mouse_id, session_id=sid, probe=probe, in_dataset_v2=bool(r.in_dataset_v2))
    if r.raw_exists:
        binp, source = r.raw_bin, 'raw_t0'
    else:                                                     # AB155: merged session, use CatGT tcat (-gblcar)
        binp = f'{os.path.dirname(r.ks4_dir)}/{run}_tcat.imec{pn}.ap.bin'
        if not os.path.exists(binp):
            binp = f'{MH}/{r.mouse_id}/{sid}/Ephys/catgt_{run}/{run}_imec{pn}/{run}_tcat.imec{pn}.ap.bin'
        source = 'catgt_tcat'
    md = meta(binp[:-4] + '.meta')
    fs = float(md['imSampRate']); nch = int(md['nSavedChans']); maxint = int(md['imMaxInt'])
    ptype = md.get('imDatPrb_type', '0')
    gain = float(md.get('imChan0apGain', 80 if ptype != '0' else 500))
    uv = float(md['imAiRangeMax']) / maxint / gain * 1e6
    row.update(source=source, probe_type_code=ptype, probe_type='NP1.0' if ptype == '0' else f'NP2.0 ({ptype})',
               uv_per_bit=uv, fs=fs)
    x = np.memmap(binp, dtype='int16', mode='r').reshape(-1, nch)
    nsamp = x.shape[0]

    tr = read_trials(sid)
    ft, a, b, n_ok, n_file, n_nwb, res_ms = align(tr, r.whisker_times_file)
    row.update(align_matched=n_ok, align_n_file=n_file, align_n_nwb=n_nwb, align_max_resid_ms=res_ms)
    on, ntr = {}, {}
    for m, col in (('whisker', 'whisker_stim'), ('auditory', 'auditory_stim')):
        t_nwb, n_act, n_all = pick(tr, col)
        tf = a * t_nwb + b
        if m == 'whisker':                                    # snap to TPrime whisker times
            q = np.abs(ft[None, :] - tf[:, None]).argmin(1) if len(tf) else np.array([], int)
            tf = np.where(np.abs(ft[q] - tf) < 0.005, ft[q], tf) if len(tf) else tf
        on[m] = tf; ntr[m] = (len(tf), n_act, n_all)
        row.update({f'n_{m}_used': len(tf), f'n_{m}_active': n_act})

    n0, n1 = int(round(PRE_MS / 1e3 * fs)), int(round(POST_MS / 1e3 * fs))
    tms = np.arange(-n0, n1) / fs * 1e3
    b0, b1 = int(round(BRIDGE[0] / 1e3 * fs)) + n0, int(round(BRIDGE[1] / 1e3 * fs)) + n0
    sos = butter(3, 300, 'highpass', fs=fs, output='sos')
    edges = np.arange(-PRE_MS, POST_MS + 1)                   # 1-ms bins
    bidx = np.clip(np.searchsorted(edges, tms, side='right') - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    bcnt = np.bincount(bidx, minlength=nb)[:, None]
    res = {}
    for m in ('whisker', 'auditory'):
        rms_tr = []; dev = np.zeros(nb); nt = 0
        sat_frac, sat_ch, flat_w, flat_b = [], [], [], []
        for t in on[m]:
            s = int(round(t * fs))
            if s - n0 < 0 or s + n1 > nsamp:
                continue
            w = np.asarray(x[s - n0:s + n1, :384])
            wf = w.astype(np.float64) * uv
            wf -= np.median(wf[(tms > BASE[0]) & (tms < BASE[1])], 0)
            dev += np.bincount(bidx, weights=np.abs(wf).mean(1), minlength=nb)
            sat = (w >= maxint - 1) | (w <= -maxint)
            z = (tms >= 0) & (tms < 4)
            sat_frac.append(sat[z].mean()); sat_ch.append(sat[z].any(0).mean())
            d0 = np.diff(w, axis=0) == 0
            flat_w.append(d0[(tms[1:] > WIN[0]) & (tms[1:] < WIN[1])].mean())
            flat_b.append(d0[(tms[1:] > BASE[0]) & (tms[1:] < BASE[1])].mean())
            wb = wf.copy()                                    # bridge the pulse, then high-pass
            lam = np.linspace(0, 1, b1 - b0 + 1)[1:-1, None]
            wb[b0 + 1:b1] = wb[b0] * (1 - lam) + wb[b1] * lam
            hp = sosfiltfilt(sos, wb - wb.mean(0), axis=0)
            rms_tr.append(np.sqrt(np.add.reduceat(hp ** 2, np.r_[0, np.flatnonzero(np.diff(bidx)) + 1], axis=0) / bcnt))
            nt += 1
        rms = np.median(np.stack(rms_tr), 0) if nt else np.full((nb, 384), np.nan)   # median over trials (robust to outlier trials)
        res[m] = dict(rms=rms, dev=dev / bcnt[:, 0] / max(nt, 1), nt=nt, sat_frac=np.mean(sat_frac) if nt else np.nan,
                      sat_ch=np.mean(sat_ch) if nt else np.nan, flat_w=np.mean(flat_w) if nt else np.nan,
                      flat_b=np.mean(flat_b) if nt else np.nan)
    ctr = edges[:-1] + 0.5
    bm = (ctr > BASE[0]) & (ctr < BASE[1]); wm = (ctr > WIN[0]) & (ctr < WIN[1])
    base_w = res['whisker']['rms'][bm].mean(0); base_a = res['auditory']['rms'][bm].mean(0)
    good = (base_w > 0.2 * np.median(base_w)) & (base_a > 0.2 * np.median(base_a))   # drop reference / dead channels
    rw_t = np.median(res['whisker']['rms'][:, good] / base_w[good], 1)
    ra_t = np.median(res['auditory']['rms'][:, good] / base_a[good], 1)
    rwa_t = np.median(res['whisker']['rms'][:, good] / np.maximum(res['auditory']['rms'][:, good], 1e-9), 1)
    ok = rw_t >= REC_THR
    rec = np.nan
    for k in np.flatnonzero(ctr >= 5):
        if k + REC_RUN <= len(ok) and ok[k:k + REC_RUN].all():
            rec = float(ctr[k] - 0.5); break
    sub = rw_t < REC_THR
    row.update(n_good_channels=int(good.sum()),
               base_rms_uV_whisker=float(np.median(base_w[good])), base_rms_uV_auditory=float(np.median(base_a[good])),
               ratio_w_6_18=float(np.median(res['whisker']['rms'][wm][:, good].mean(0) / base_w[good])),
               ratio_a_6_18=float(np.median(res['auditory']['rms'][wm][:, good].mean(0) / base_a[good])),
               ratio_wa_6_18=float(np.median(res['whisker']['rms'][wm][:, good].mean(0) /
                                             res['auditory']['rms'][wm][:, good].mean(0))),
               min_ratio_w_5_60=float(rw_t[(ctr >= 5) & (ctr < 60)].min()),
               recovery_ms=rec, n_bins_below_thr_5_60=int(sub[(ctr >= 5) & (ctr < 60)].sum()),
               sat_frac_0_4_whisker=res['whisker']['sat_frac'], sat_ch_frac_0_4_whisker=res['whisker']['sat_ch'],
               sat_frac_0_4_auditory=res['auditory']['sat_frac'],
               flat_frac_6_18_whisker=res['whisker']['flat_w'], flat_frac_base_whisker=res['whisker']['flat_b'],
               flat_frac_6_18_auditory=res['auditory']['flat_w'],
               artefact_peak_ms=float(ctr[np.argmax(res['whisker']['dev'])]),
               artefact_peak_over_base=float(res['whisker']['dev'].max() / np.median(res['whisker']['dev'][bm])))
    # KS4 spikes (all clusters)
    stp = f'{r.ks4_dir}/sorter_output/spike_times.npy'
    if os.path.exists(stp):
        st = np.load(stp, mmap_mode='r').ravel()
        for m in ('whisker', 'auditory'):
            cb = cw = 0
            for t in on[m]:
                s = int(round(t * fs))
                i0, i1, i2, i3 = np.searchsorted(st, [s + BASE[0] * fs / 1e3, s + BASE[1] * fs / 1e3,
                                                      s + WIN[0] * fs / 1e3, s + WIN[1] * fs / 1e3])
                cb += i1 - i0; cw += i3 - i2
            n = max(len(on[m]), 1)
            row[f'ks4_rate_base_{m}'] = cb / n / ((BASE[1] - BASE[0]) / 1e3)
            row[f'ks4_rate_6_18_{m}'] = cw / n / ((WIN[1] - WIN[0]) / 1e3)
        row['ks4_ratio_w_6_18'] = row['ks4_rate_6_18_whisker'] / max(row['ks4_rate_base_whisker'], 1e-9)
        row['ks4_ratio_a_6_18'] = row['ks4_rate_6_18_auditory'] / max(row['ks4_rate_base_auditory'], 1e-9)
    np.savez_compressed(f'{OUT}/cache/{tag}.npz', t_ms=ctr, ratio_w=rw_t, ratio_a=ra_t, ratio_wa=rwa_t,
                        rms_w=res['whisker']['rms'].astype(np.float32), rms_a=res['auditory']['rms'].astype(np.float32),
                        dev_w=res['whisker']['dev'], dev_a=res['auditory']['dev'], good=good,
                        trial_times_w=on['whisker'], trial_times_a=on['auditory'])
    json.dump(row, open(f'{OUT}/cache/{tag}.json', 'w'), default=float)
    print(json.dumps(row, default=float))


if __name__ == '__main__':
    main(int(sys.argv[1]))
