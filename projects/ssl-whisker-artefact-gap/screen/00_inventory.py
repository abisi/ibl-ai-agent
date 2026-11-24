"""Inventory of every ephys session x probe to screen for the whisker-artefact dead time.

A probe is listed when it has a KS4 sorting (Axel_Bisi/data/<mouse>/<session>/Ephys/catgt_<run>/<run>_imecN/kilosort4)
and its session has a canonical NWB_ks4 file (MOUSE_YYYYMMDD_HHMMSS.nwb). For each probe the script records the raw
SpikeGLX .ap.bin (lsens-data), the TPrime whisker times file (Axel's catgt folder, else Myriam's), the probe type from the
raw meta, and whether the probe is in the current dataset unit table (unit_info_all_days_v2.parquet).
Output: <OUT>/tables/00_inventory.csv
"""
import glob
import os
import re

import pandas as pd

OUT = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-whisker-artefact-gap/screen'
NWB = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'
AB = '/mnt/lsens-analysis/Axel_Bisi/data'
MH = '/mnt/lsens-analysis/Myriam_Hamon/data'
RAW = '/mnt/lsens-data'
V2 = '/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/_roc_stage_analysis/unit_info_all_days_v2.parquet'
NWB_RE = re.compile(r'^([A-Z]{2}\d{3})_(\d{8})_(\d{6})\.nwb$')


def meta(p):
    d = {}
    for line in open(p):
        k, _, v = line.strip().partition('=')
        d[k.lstrip('~')] = v
    return d


def main():
    os.makedirs(f'{OUT}/tables', exist_ok=True)
    nwbs = sorted(f[:-4] for f in os.listdir(NWB) if NWB_RE.match(f))
    v2 = pd.read_parquet(V2, columns=['mouse_id', 'session_id', 'electrode_group', 'day'])
    v2['probe'] = v2.electrode_group.str.extract(r'(imec\d)')[0]
    v2 = v2.drop_duplicates(['session_id', 'probe'])
    v2key = set(zip(v2.session_id, v2.probe))
    rows = []
    for sid in nwbs:
        mouse = sid[:5]
        for kd in sorted(glob.glob(f'{AB}/{mouse}/{sid}/Ephys/catgt_*/*_imec[0-9]/kilosort4')):
            pdir = os.path.dirname(kd)
            run = os.path.basename(os.path.dirname(pdir))[len('catgt_'):]
            probe = re.search(r'imec\d', os.path.basename(pdir)).group(0)
            pn = probe[4:]
            raw = f'{RAW}/{mouse}/Recording/{sid}/Ephys/{run}/{run}_imec{pn}/{run}_t0.imec{pn}.ap.bin'
            wfile = None
            for base in (pdir, f'{MH}/{mouse}/{sid}/Ephys/catgt_{run}/{run}_imec{pn}'):
                p = f'{base}/whisker_stim_times_to_imec{pn}.txt'
                if os.path.exists(p):
                    wfile = p
                    break
            r = dict(mouse_id=mouse, session_id=sid, probe=probe, run=run, raw_bin=raw, raw_exists=os.path.exists(raw),
                     whisker_times_file=wfile, ks4_dir=kd,
                     ks4_spikes=os.path.exists(f'{kd}/sorter_output/spike_times.npy'),
                     in_dataset_v2=(sid, probe) in v2key)
            if r['raw_exists'] and os.path.exists(raw[:-4] + '.meta'):
                md = meta(raw[:-4] + '.meta')
                r.update(probe_type_code=md.get('imDatPrb_type', '0'), n_saved_chans=int(md['nSavedChans']),
                         fs=float(md['imSampRate']), raw_n_samples=int(md['fileSizeBytes']) // 2 // int(md['nSavedChans']))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT}/tables/00_inventory.csv', index=False)
    print(len(df), 'probes in', df.session_id.nunique(), 'sessions;', df.raw_exists.sum(), 'with raw;',
          df.whisker_times_file.notna().sum(), 'with whisker times file;', df.in_dataset_v2.sum(), 'in v2 dataset')
    print('v2 probes not found:', len(v2key - set(zip(df.session_id, df.probe))))
    print(df.probe_type_code.value_counts(dropna=False))


if __name__ == '__main__':
    main()
