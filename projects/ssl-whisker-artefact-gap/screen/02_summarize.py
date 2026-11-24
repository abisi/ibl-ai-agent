"""Collect the per-probe screen rows, flag dead-time probes, write the table and the one-page summary figure.

Flag (dead time after the whisker artefact):
  spike-band RMS 6-18 ms after whisker onset < FLAG_RATIO x pre-stimulus baseline (median over channels and trials),
  OR raw signal flat: flat fraction in 6-18 ms exceeds the probe's own baseline flat fraction by > FLAG_FLAT
  (NP2.0 blocked amplifier; baseline-relative so probes with many dead channels, e.g. AB107 imec2, are not flagged).
Also writes ../flagged_probes.csv (one row per shank, column 'flagged') read by ssl-s1m2-communication pairs_lib.
Recovery: first time >= 5 ms after whisker onset with 3 consecutive 1-ms bins >= 0.8 x baseline (5 = no dead time
after the bridged -0.5..+4 ms pulse).
Outputs: <OUT>/tables/02_dead_time_screen.csv, <OUT>/figures/02_dead_time_screen.{png,pdf,svg}
"""
import glob
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap/screen'
FLAG_RATIO = 0.7
FLAG_FLAT = 0.3
REF = {('MH031_20250507_104425', 'imec3'): 'MH031 imec3 (normal)',
       ('MH037_20250524_143522', 'imec3'): 'MH037 imec3', ('MH037_20250524_143522', 'imec1'): 'MH037 imec1',
       ('MH020_20250304_161732', 'imec0'): 'MH020 imec0'}
C_NP1, C_NP2, C_FLAG = '#7f7f7f', '#1f77b4', '#d62728'
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})


def main():
    os.makedirs(f'{OUT}/figures', exist_ok=True)
    inv = pd.read_csv(f'{OUT}/tables/00_inventory.csv')
    rows = [json.load(open(f)) for f in sorted(glob.glob(f'{OUT}/cache/*.json'))]
    df = pd.DataFrame(rows)
    missing = inv.merge(df[['session_id', 'probe']], how='left', indicator=True).query('_merge == "left_only"')
    df['electrode_group'] = df.probe + '_shank0'
    df['flag_dead_time'] = (df.ratio_w_6_18 < FLAG_RATIO) | ((df.flat_frac_6_18_whisker - df.flat_frac_base_whisker) > FLAG_FLAT)
    df['dead_time_end_ms'] = np.where(df.flag_dead_time, df.recovery_ms, np.nan)
    df['align_ok'] = (df.align_matched >= 0.4 * df.align_n_file) & df.artefact_peak_ms.between(-1, 4)
    df['reference_case'] = [REF.get((s, p), '') for s, p in zip(df.session_id, df.probe)]
    first = ['mouse_id', 'session_id', 'probe', 'electrode_group', 'probe_type', 'in_dataset_v2', 'ratio_w_6_18',
             'ratio_a_6_18', 'ratio_wa_6_18', 'sat_frac_0_4_whisker', 'sat_ch_frac_0_4_whisker',
             'flat_frac_6_18_whisker', 'recovery_ms', 'flag_dead_time', 'dead_time_end_ms', 'ks4_ratio_w_6_18',
             'ks4_ratio_a_6_18', 'align_ok', 'reference_case']
    df = df[first + [c for c in df.columns if c not in first]].sort_values(['session_id', 'probe'])
    df.to_csv(f'{OUT}/tables/02_dead_time_screen.csv', index=False)
    fl = df[df.flag_dead_time]
    pd.DataFrame([dict(mouse_id=r.mouse_id, session_id=r.session_id, probe=r.probe, electrode_group=f'{r.probe}_shank{k}',
                       flagged=True, dead_time_end_ms=r.recovery_ms, ratio_w_6_18=r.ratio_w_6_18,
                       in_dataset_v2=r.in_dataset_v2) for r in fl.itertuples() for k in range(4)]
                 ).to_csv(f'{OUT}/../flagged_probes.csv', index=False)
    fl = df[df.flag_dead_time].sort_values('recovery_ms', ascending=False)
    print(f'{len(df)} probes screened ({df.session_id.nunique()} sessions, {df.mouse_id.nunique()} mice); '
          f'{len(missing)} inventory probes without result; {df.flag_dead_time.sum()} flagged; '
          f'align not ok: {(~df.align_ok).sum()}')
    with pd.option_context('display.width', 250, 'display.max_rows', 500):
        print(fl[['session_id', 'probe', 'probe_type', 'in_dataset_v2', 'ratio_w_6_18', 'ratio_a_6_18',
                  'sat_ch_frac_0_4_whisker', 'flat_frac_6_18_whisker', 'recovery_ms', 'ks4_ratio_w_6_18']].round(3).to_string())
        print(df[~df.align_ok][['session_id', 'probe', 'align_matched', 'align_n_file', 'artefact_peak_ms']])
        if len(missing):
            print(missing[['session_id', 'probe']])

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.8), constrained_layout=True)
    np1 = df.probe_type == 'NP1.0'
    a = ax[0, 0]
    bins = np.linspace(0, max(1.6, df.ratio_w_6_18.max() + 0.05), 41)
    a.hist([df.ratio_w_6_18[np1], df.ratio_w_6_18[~np1]], bins=bins, stacked=True, color=[C_NP1, C_NP2],
           label=['NP1.0', 'NP2.0'], lw=0, edgecolor='none')
    a.axvline(FLAG_RATIO, color=C_FLAG, ls='--', lw=0.8)
    ymax = a.get_ylim()[1]
    for k, (key, lab) in enumerate(REF.items()):
        v = df[(df.session_id == key[0]) & (df.probe == key[1])].ratio_w_6_18
        if len(v):
            a.annotate(lab, (v.iloc[0], 0), xytext=(v.iloc[0], ymax * (0.95 - 0.12 * k)), fontsize=6,
                       arrowprops=dict(arrowstyle='-', lw=0.5, color='k'), ha='left')
    a.set_xlabel('Spike-band RMS 6-18 ms after whisker onset\n/ pre-stimulus RMS')
    a.set_ylabel('Probes'); a.legend(frameon=False, loc='center right')
    a.set_title(f'A  {len(df)} probes, {df.session_id.nunique()} sessions; {df.flag_dead_time.sum()} flagged', loc='left')

    a = ax[0, 1]
    a.scatter(df.ratio_a_6_18, df.ratio_w_6_18, s=6, c=np.where(df.flag_dead_time, C_FLAG, np.where(np1, C_NP1, C_NP2)),
              lw=0)
    lim = [0, max(df.ratio_a_6_18.max(), df.ratio_w_6_18.max()) * 1.05]
    a.plot(lim, lim, ':', color='k', lw=0.5); a.axhline(FLAG_RATIO, color=C_FLAG, ls='--', lw=0.8)
    a.set_xlabel('Auditory: RMS 6-18 ms / pre-stimulus'); a.set_ylabel('Whisker: RMS 6-18 ms / pre-stimulus')
    a.set_title('B  Whisker vs auditory trials (same probe)', loc='left')

    a = ax[0, 2]
    a.scatter(df.sat_ch_frac_0_4_whisker, df.ratio_w_6_18, s=6,
              c=np.where(df.flag_dead_time, C_FLAG, np.where(np1, C_NP1, C_NP2)), lw=0)
    a.axhline(FLAG_RATIO, color=C_FLAG, ls='--', lw=0.8)
    a.set_xlabel('Fraction of channels at ADC limit 0-4 ms (whisker)'); a.set_ylabel('Whisker: RMS 6-18 ms / pre-stimulus')
    a.set_title('C  Saturation vs dead time', loc='left')

    a = ax[1, 0]
    for _, r in df[~df.flag_dead_time].iterrows():
        z = np.load(f'{OUT}/cache/{r.session_id}_{r.probe}.npz')
        a.plot(z['t_ms'], z['ratio_w'], color='0.8', lw=0.3)
    for _, r in fl.iterrows():
        z = np.load(f'{OUT}/cache/{r.session_id}_{r.probe}.npz')
        a.plot(z['t_ms'], z['ratio_w'], color=C_FLAG, lw=0.6, alpha=0.8)
    a.axvspan(-0.5, 4, color='0.9', lw=0, edgecolor='none'); a.axhline(1, color='k', lw=0.4, ls=':')
    a.axhline(0.8, color=C_FLAG, lw=0.4, ls=':')
    a.set_xlim(-20, 70); a.set_ylim(0, 2)
    a.set_xlabel('Time from whisker onset (ms)'); a.set_ylabel('Spike-band RMS / pre-stimulus')
    a.set_title('D  Time course (red: flagged; grey band: bridged pulse)', loc='left')

    a = ax[1, 1]
    if len(fl):
        lab = [f'{s[:5]} {s[6:14]} {p}' + (' *' if not v else '') for s, p, v in zip(fl.session_id, fl.probe, fl.in_dataset_v2)]
        y = np.arange(len(fl))
        a.barh(y, fl.recovery_ms.fillna(70), color=[C_NP1 if t == 'NP1.0' else C_NP2 for t in fl.probe_type], lw=0)
        a.set_yticks(y); a.set_yticklabels(lab, fontsize=max(3, min(6, 300 / len(fl))))
        a.invert_yaxis()
    a.set_xlabel('Recovery time after whisker onset (ms)\n(RMS back >= 0.8 x pre-stimulus)')
    a.set_title('E  Flagged probes (grey NP1.0, blue NP2.0; * not in v2)', loc='left')

    a = ax[1, 2]
    a.scatter(df.ratio_w_6_18, df.ks4_ratio_w_6_18, s=6,
              c=np.where(df.flag_dead_time, C_FLAG, np.where(np1, C_NP1, C_NP2)), lw=0)
    a.axvline(FLAG_RATIO, color=C_FLAG, ls='--', lw=0.8); a.axhline(1, color='k', lw=0.4, ls=':')
    a.set_xlabel('Whisker: spike-band RMS 6-18 ms / pre-stimulus'); a.set_ylabel('Whisker: KS4 spike rate 6-18 ms / pre-stimulus')
    a.set_title('F  Spike-band loss vs KS4 spike loss', loc='left')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(f'{OUT}/figures/02_dead_time_screen.{ext}', dpi=200)
    print('saved figure')


if __name__ == '__main__':
    main()
