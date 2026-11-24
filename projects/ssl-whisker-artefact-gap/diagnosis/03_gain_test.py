"""Is the post-artefact gap a loss of signal (amplifier recovery / reduced gain) or silent neurons / sorter misses?
1) spike-band RMS (raw + KS4 high-pass, no CAR) in uV: baseline vs 6-18 ms, whisker vs auditory;
2) raw count statistics during the gap on single trials (flat / clipped values?);
3) KS4 spike amplitudes (amplitudes.npy) during 6-20 ms vs baseline, whisker vs auditory: a gain drop shrinks them.
"""
import numpy as np
import pandas as pd

D = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap'
AB = '/mnt/lsens-analysis/Axel_Bisi/data'
RAW = '/mnt/lsens-data'
CASES = [('MH037_20250524_143522', 'MH037_g1', 3), ('MH037_20250524_143522', 'MH037_g1', 1),
         ('MH020_20250304_161732', 'MH020_g0', 0), ('MH031_20250507_104425', 'MH031_g0', 3)]


def meta(p):
    d = {}
    for l in open(p):
        k, _, v = l.strip().partition('='); d[k.lstrip('~')] = v
    return d


rows = []
for sid, run, pr in CASES:
    tag = f'{sid}_imec{pr}'
    so = f'{AB}/{sid[:5]}/{sid}/Ephys/catgt_{run}/{run}_imec{pr}/kilosort4/sorter_output'
    st = np.load(f'{so}/spike_times.npy').astype(np.int64).ravel()
    amp = np.load(f'{so}/amplitudes.npy').ravel()
    rp = f'{RAW}/{sid[:5]}/Recording/{sid}/Ephys/{run}/{run}_imec{pr}/{run}_t0.imec{pr}.ap'
    md = meta(rp + '.meta'); nch = int(md['nSavedChans']); maxint = int(md['imMaxInt'])
    x = np.memmap(rp + '.bin', dtype='int16', mode='r').reshape(-1, nch)
    for m in ('whisker', 'auditory'):
        z = np.load(f'{D}/cache/{tag}_{m}.npz')
        t = z['t_ms']; fs = float(z['fs'])
        r = np.sqrt(z['rawf_sq'])                    # uV, (t, ch)
        rb = np.median(r[t < -5], 0); rg = np.median(r[(t > 6) & (t < 18)], 0)
        rw = np.sqrt(z['ks4w_sq']); wb = np.median(rw[t < -5], 0); wg = np.median(rw[(t > 6) & (t < 18)], 0)
        # all trials of this modality: KS4 amplitudes
        tt = z['trial_times_file_s']
        # use all trials stored for spikes: reconstruct from fit is not stored; use averaged trials (30) for amplitudes
        ab, ag, n_b, n_g = [], [], 0, 0
        for tr in tt:
            s = int(round(tr * fs))
            i0, i1 = np.searchsorted(st, [s - int(0.05 * fs), s - int(0.005 * fs)]); ab.append(amp[i0:i1])
            j0, j1 = np.searchsorted(st, [s + int(0.006 * fs), s + int(0.020 * fs)]); ag.append(amp[j0:j1])
        ab = np.concatenate(ab); ag = np.concatenate(ag)
        # raw counts on 5 single trials, 6-18 ms: per-channel SD and fraction of samples equal to their trial-mode value
        sd_b, sd_g, flat = [], [], []
        for tr in tt[:5]:
            s = int(round(tr * fs))
            w = np.asarray(x[s - int(0.05 * fs):s + int(0.03 * fs), :384], float)
            tw = (np.arange(len(w)) - int(0.05 * fs)) / fs * 1e3
            g = w[(tw > 6) & (tw < 18)]
            sd_b.append(np.median(np.diff(w[tw < -5], axis=0).std(0))); sd_g.append(np.median(np.diff(g, axis=0).std(0)))
            flat.append(np.mean(np.diff(g, axis=0) == 0))
        rows.append(dict(case=tag, modality=m, uV_per_bit=float(z['uv_per_bit']), ADC_max_counts=maxint,
                         spikeband_rms_base_uV=float(np.median(rb)), spikeband_rms_6_18ms_uV=float(np.median(rg)),
                         ratio_rawHP=float(np.median(rg / rb)), ratio_whitened=float(np.median(wg / wb)),
                         raw_diff_sd_base_counts=float(np.median(sd_b)), raw_diff_sd_6_18_counts=float(np.median(sd_g)),
                         frac_flat_samples_6_18=float(np.mean(flat)),
                         ks4_amp_median_base=float(np.median(ab)), ks4_amp_median_6_20=float(np.median(ag)) if len(ag) else np.nan,
                         ks4_spikes_per_ms_base=len(ab) / len(tt) / 45, ks4_spikes_per_ms_6_20=len(ag) / len(tt) / 14))
q = pd.DataFrame(rows)
q.to_csv(f'{D}/gain_test.csv', index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 30):
    print(q.round(3).to_string())
