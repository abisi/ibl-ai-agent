"""Exploration: map NWB whisker trial times to probe sample indices and look at raw artefact."""
import numpy as np, h5py
NWB = '/mnt/lsens-analysis/Axel_Bisi/NWB_ks4'
MH = '/mnt/lsens-analysis/Myriam_Hamon/data'
RAW = '/mnt/lsens-data'
CASES = [('MH037_20250524_143522', 'MH037_g1', 3), ('MH037_20250524_143522', 'MH037_g1', 1),
         ('MH020_20250304_161732', 'MH020_g0', 0), ('MH031_20250507_104425', 'MH031_g0', 3)]
def meta(p):
    d = {}
    for l in open(p):
        k, _, v = l.strip().partition('='); d[k.lstrip('~')] = v
    return d
for sid, run, pr in CASES:
    mouse = sid[:5]
    with h5py.File(f'{NWB}/{sid}.nwb', 'r') as f:
        tr = f['intervals/trials']
        st = tr['start_time'][:]; ws = tr['whisker_stim'][:]; au = tr['auditory_stim'][:]
        ctx = tr['context'][:] if 'context' in tr else None
    wn = st[ws == 1]
    ft = np.loadtxt(f'{MH}/{mouse}/{sid}/Ephys/catgt_{run}/{run}_imec{pr}/whisker_stim_times_to_imec{pr}.txt')
    # match: offsets
    d = ft[:, None] - wn[None, :]
    h, e = np.histogram(d.ravel(), bins=np.arange(-50, 50, 0.01))
    off = e[np.argmax(h)]
    m = np.abs(d - off).min(1) < 0.05
    print(sid, pr, 'n NWB whisker', len(wn), 'n file', len(ft), 'offset ~', round(off, 3), 'matched', m.sum())
    j = np.abs(d - off).argmin(1)
    a, b = np.polyfit(wn[j[m]], ft[m], 1)
    res = ft[m] - (a * wn[j[m]] + b)
    print('   fit slope', a, 'icpt', b, 'resid ms min/med/max', 1e3 * res.min(), 1e3 * np.median(res), 1e3 * res.max())
    rp = f'{RAW}/{mouse}/Recording/{sid}/Ephys/{run}/{run}_imec{pr}/{run}_t0.imec{pr}.ap'
    md = meta(rp + '.meta'); fs = float(md['imSampRate']); nch = int(md['nSavedChans'])
    x = np.memmap(rp + '.bin', dtype='int16', mode='r').reshape(-1, nch)
    uv = float(md['imAiRangeMax']) / int(md['imMaxInt']) / 500 * 1e6
    for t in ft[:3]:
        s = int(round(t * fs))
        w = np.asarray(x[s - 150:s + 1500, :384], float)
        mabs = np.abs(w - np.median(w[:120], 0)).mean(1) * uv
        sat = (np.abs(w) >= 511).mean(1)
        idx = np.arange(-150, 1500) / fs * 1e3
        print('   stim', round(t, 3), ' mean|dev| uV at ms:', {round(k, 1): round(v, 0) for k, v in zip(idx[::60], mabs[::60])})
        print('   sat frac:', {round(k, 1): round(v, 2) for k, v in zip(idx[::60], sat[::60]) if v > 0})
