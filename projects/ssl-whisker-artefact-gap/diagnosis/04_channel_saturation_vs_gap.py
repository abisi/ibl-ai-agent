"""Per channel: does the post-artefact loss of spike-band signal and of KS4 spikes follow ADC saturation of that channel?
x = fraction of 0-5 ms samples at the ADC limit (raw, whisker trials);
y1 = spike-band RMS 6-18 ms / baseline (KS4 input + KS4 HP, i.e. what KS4 sees before whitening);
y2 = KS4 spikes (all clusters, peak channel) 6-20 ms / baseline rate (all active whisker trials).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

D = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap'
CASES = [('MH037_20250524_143522_imec3', 'MH037 imec3'), ('MH037_20250524_143522_imec1', 'MH037 imec1 (NP2)'),
         ('MH020_20250304_161732_imec0', 'MH020 imec0'), ('MH031_20250507_104425_imec3', 'MH031 imec3 (normal)')]
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'font.size': 6, 'axes.linewidth': 0.5,
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.4))
rows = []
for j, (tag, title) in enumerate(CASES):
    out = {}
    for m in ('whisker', 'auditory'):
        z = np.load(f'{D}/cache/{tag}_{m}.npz'); t = z['t_ms']
        r = np.sqrt(z['ks4f_sq'])
        rr = np.median(r[(t > 6) & (t < 18)], 0) / np.median(r[t < -5], 0)
        ed = z['spk_edges_ms']; tc = (ed[:-1] + ed[1:]) / 2; p = z['spk_psth']
        b = p[tc < -5].mean(0); g = p[(tc > 6) & (tc < 20)].mean(0)
        sat = z['sat'][(t >= 0) & (t < 5)].mean(0)
        # raw flat (NP2 imec1: output held constant, not at ADC max): use raw RMS of HP signal instead
        rf = np.sqrt(z['rawf_sq']); rfr = np.median(rf[(t > 6) & (t < 18)], 0) / np.median(rf[t < -5], 0)
        out[m] = (sat, rr, np.where(b > 0.02, g / np.maximum(b, 1e-9), np.nan), rfr)
    sat, rr, sr, rfr = out['whisker']
    ok = np.isfinite(sr)
    satd = sat > 0.05
    rho1 = spearmanr(sat, rr).correlation; rho2 = spearmanr(sat[ok], sr[ok]).correlation
    rows.append(dict(case=tag, n_ch_saturated_gt5pct=int(satd.sum()),
                     rms_ratio_sat=np.median(rr[satd]) if satd.any() else np.nan, rms_ratio_unsat=np.median(rr[~satd]),
                     spk_ratio_sat=np.nanmedian(sr[satd]) if satd.any() else np.nan, spk_ratio_unsat=np.nanmedian(sr[~satd]),
                     rawHP_ratio_sat=np.median(rfr[satd]) if satd.any() else np.nan, rawHP_ratio_unsat=np.median(rfr[~satd]),
                     spearman_sat_vs_rms=rho1, spearman_sat_vs_spk=rho2,
                     aud_rms_ratio=np.median(out['auditory'][1]), aud_spk_ratio=np.nanmedian(out['auditory'][2])))
    ax = axs[0, j]
    ax.scatter(100 * sat, rr, s=2, c='#F5A700', lw=0); ax.axhline(1, color='0.6', lw=0.4, ls=':')
    ax.set_title(title, fontsize=6.5); ax.set_ylim(0, 2.5)
    ax = axs[1, j]
    ax.scatter(100 * sat[ok], sr[ok], s=2, c='#F5A700', lw=0); ax.axhline(1, color='0.6', lw=0.4, ls=':'); ax.set_ylim(0, 4)
    ax.set_xlabel('% samples at ADC limit, 0-5 ms')
axs[0, 0].set_ylabel('KS4-input spike-band RMS\n6-18 ms / baseline (per channel)')
axs[1, 0].set_ylabel('KS4 spikes 6-20 ms\n/ baseline (per channel)')
fig.tight_layout(); fig.savefig(f'{D}/fig3_channel_saturation_vs_gap.png', dpi=300); plt.close(fig)
q = pd.DataFrame(rows); q.to_csv(f'{D}/channel_saturation_vs_gap.csv', index=False)
with pd.option_context('display.width', 250, 'display.max_columns', 30):
    print(q.round(3).to_string())
