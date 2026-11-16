"""001 -- Can the April 2025 supercat (lsens-data/AB155/Recording/AB155_20250208_175103/Ephys/AB155_g0, made with
-supercat_trim_edges from catgt_AB155_g0 + catgt_AB155_g1) be reused? Read-only.

(1) Segments: sc_offsets (start of run 1 in the joined file) vs the current per-run CatGT lengths; where g1 starts.
(2) Binary identity: int16 samples of the joined probe files vs the current per-run tcat files (all 385 channels,
    10,000 samples at several positions in each part; the g1 shift is found from the imec sync edges).
(3) TTL edges per NIDQ channel: joined edges before the junction vs g0 edges, after the junction (minus the offset)
    vs g1 edges (minus the g1 trim); lost / added edges and the edges inside the trimmed seconds.
Run (haas): python 001_supercat_reuse_check.py
"""
import glob
import os

import numpy as np

A = '/mnt/lsens-analysis/Axel_Bisi/data/AB155'
G0 = f'{A}/AB155_20250208_175103/Ephys/catgt_AB155_g0'
G1 = f'{A}/AB155_20250208_185559/Ephys/catgt_AB155_g1'
SC = '/mnt/lsens-data/AB155/Recording/AB155_20250208_175103/Ephys/AB155_g0'
NCH = 385
PROBES = range(5)


def offsets():
    d = {}
    for line in open(f'{SC}/AB155_g0_sc_offsets.txt'):
        k, *v = line.split()
        d[k.rstrip(':')] = [float(x) for x in v]
    return d


def edges(path):
    return np.loadtxt(path, ndmin=1) if os.path.getsize(path) else np.array([])


def read(path, start, n):
    mm = np.memmap(path, dtype=np.int16, mode='r')
    ns = mm.size // NCH
    start = int(start)
    if start < 0 or start + n > ns:
        return None
    return np.asarray(mm[start * NCH:(start + n) * NCH]).reshape(n, NCH)


def main():
    off = offsets()
    print('sc_offsets run-1 start (s):', {k: v[1] for k, v in off.items() if k.startswith('sec')})
    for p in PROBES:
        f0 = f'{G0}/AB155_g0_imec{p}/AB155_g0_tcat.imec{p}.ap.bin'
        f1 = f'{G1}/AB155_g1_imec{p}/AB155_g1_tcat.imec{p}.ap.bin'
        fs = f'{SC}/AB155_g0_imec{p}/AB155_g0_tcat.imec{p}.ap.bin'
        n0, n1, ns = (os.path.getsize(f) // (2 * NCH) for f in (f0, f1, fs))
        start1 = int(off[f'smp_imap{p}'][1])
        # g1 shift from the imec sync edges: joined edges after the junction vs g1 edges
        e_sc = edges(glob.glob(f'{SC}/AB155_g0_imec{p}/*ap.xd_384_6_500.txt')[0])
        e_1 = edges(glob.glob(f'{G1}/AB155_g1_imec{p}/*ap.xd_384_6_500.txt')[0])
        sec1 = off[f'sec_imap{p}'][1]
        after = e_sc[e_sc >= sec1] - sec1
        rate = start1 / sec1
        shift_s = np.median(e_1[:20] - after[:20]) if len(after) >= 20 else np.nan  # g1 time = joined time + shift
        shift = int(round(shift_s * rate))
        print(f'\nimec{p}: g0 {n0} samples, g1 {n1}, joined {ns}; run-1 start {start1} (g0 kept {start1 / n0:.4%}, '
              f'g0 samples dropped at the end {n0 - start1}); g1 starts at its sample {shift} ({shift_s:.3f} s), '
              f'g1 samples kept {ns - start1} of {n1} (dropped at the end {n1 - shift - (ns - start1)})')
        for label, pos in (('g0 begin', 1000), ('g0 middle', n0 // 2), ('g0 near junction', start1 - 20000)):
            a, b = read(fs, pos, 10000), read(f0, pos, 10000)
            print(f'   {label} @ {pos}: identical {np.array_equal(a, b) if a is not None and b is not None else "n/a"}')
        for label, k in (('g1 begin', 1000), ('g1 middle', (ns - start1) // 2), ('g1 end', ns - start1 - 20000)):
            a, b = read(fs, start1 + k, 10000), read(f1, shift + k, 10000)
            same = np.array_equal(a, b) if a is not None and b is not None else 'n/a'
            print(f'   {label} @ joined {start1 + k} / g1 {shift + k}: identical {same}')

    print('\nNIDQ edges')
    s1 = off['sec_nidq'][1]
    e_sc0 = edges(f'{SC}/AB155_g0_tcat.nidq.xa_0_500.txt')
    e_10 = edges(f'{G1}/AB155_g1_tcat.nidq.xa_0_500.txt')
    after0 = e_sc0[e_sc0 >= s1] - s1
    shift_ni = float(np.median(e_10[:20] - after0[:20]))
    print(f'junction at {s1:.4f} s (joined time); g1 time = joined time - {s1:.4f} + {shift_ni:.4f}')
    for ch in range(8):
        fsc = glob.glob(f'{SC}/AB155_g0_tcat.nidq.xa_{ch}_0.txt')
        f0 = glob.glob(f'{G0}/AB155_g0_tcat.nidq.xa_{ch}_0.txt')
        f1 = glob.glob(f'{G1}/AB155_g1_tcat.nidq.xa_{ch}_0.txt')
        if not (fsc and f0 and f1):
            print(f'ch{ch}: missing file(s) joined {bool(fsc)} g0 {bool(f0)} g1 {bool(f1)}')
            continue
        esc, e0, e1 = edges(fsc[0]), edges(f0[0]), edges(f1[0])
        a, b = esc[esc < s1], esc[esc >= s1] - s1 + shift_ni
        e0_kept, e0_cut = e0[e0 < s1], e0[e0 >= s1]
        e1_kept = e1[(e1 >= shift_ni) & (e1 < shift_ni + (off['sec_nidq'][1] if False else 1e9))]
        e1_cut_start = e1[e1 < shift_ni]
        m0 = len(a) == len(e0_kept) and np.allclose(a, e0_kept, atol=1e-4)
        n_b = min(len(b), len(e1_kept))
        m1 = len(b) == len(e1_kept) and np.allclose(b, e1_kept, atol=2e-3)
        print(f'ch{ch}: joined {len(esc)} = {len(a)} before + {len(b)} after junction | g0 {len(e0)} '
              f'(in trimmed end: {len(e0_cut)}, first {e0_cut[0] if len(e0_cut) else np.nan:.3f} s) match {m0} | g1 '
              f'{len(e1)} (before g1 start: {len(e1_cut_start)}) match {m1}'
              + ('' if m1 or not n_b else f' max|diff| {np.max(np.abs(b[:n_b] - e1_kept[:n_b])):.4f}'))


if __name__ == '__main__':
    main()
