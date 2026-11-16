"""013 -- Synchronisation of every stream in the gap-inserted joined test NWB (012), part 1 vs part 2.
  cameras: synced pulse counts per part (TPrime cam*_frame_times) vs DLC rows per part; part-1 frame times vs NWB_ks4
  stimulus TTLs: whisker / auditory stim event files vs trial stim times per part
  piezo: raw lick_trace (NIDQ, nominal 25 kHz clock, not TPrime-synced) vs piezo_lick_times (imec0) -> lag early/late
  DLC: whisker angle speed aligned to whisker stim in passive pre (part 1) vs passive post (part 2) -> response latency
  spikes: population PSTH of units aligned to whisker stim, passive pre vs passive post -> response latency
Run (local, conda env nwb): python 013_stream_sync_checks.py
"""
import h5py
import numpy as np

A = r'M:\analysis\Axel_Bisi'
J = rf'{A}\Misc\SessionToConcat\AB155_20250208_175103_joined'
NWB = rf'{J}\nwb_test\AB155_20250208_175103_joined_full_test_gap.nwb'
OLD = rf'{A}\NWB_ks4\AB155_20250208_175103.nwb'
SYNC = rf'{J}\Ephys\catgt_AB155_g0\sync_event_times'
N1, N2 = 595292, 77288  # DLC rows per part (= video frames)


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def peak_latency(t_ev, ts, x, win=(-0.05, 0.15), dt=0.005):
    """Mean of x around events (nearest sample per bin); latency of the max of the baseline-subtracted mean."""
    lags = np.arange(win[0], win[1] + 1e-9, dt)
    idx = np.searchsorted(ts, t_ev[:, None] + lags[None, :]).clip(0, len(ts) - 1)
    m = np.nanmean(x[idx], axis=0)
    m = m - np.nanmean(m[lags < 0])
    return lags[np.nanargmax(m)] * 1e3, m


def main():
    f = h5py.File(NWB, 'r')
    rg = f['acquisition/recording_gap/timestamps'][()]
    tj, t2 = rg
    shift = t2 - tj
    out = [f'part 1 ends {tj:.3f} s, part 2 starts {t2:.3f} s (shift {shift:.3f} s)']

    # cameras
    for cam in ('cam0', 'cam1'):
        p = np.loadtxt(rf'{SYNC}\{cam}_frame_times.txt')
        out.append(f'{cam} synced pulses: {len(p)} (part 1 {int((p < tj).sum())}, part 2 {int((p >= tj).sum())}); '
                   f'DLC rows {N1} + {N2}')
    acq = [k for k in f['acquisition'] if k.endswith('.avi')]
    with h5py.File(OLD, 'r') as fo:
        for k in acq:
            ts = f['acquisition'][k]['timestamps'][()]
            ko = [x for x in fo['acquisition'] if x.split('\\')[-1].split(' ')[0] == k.split('\\')[-1].split(' ')[0]]
            msg = ''
            if ko:
                to = fo['acquisition'][ko[0]]['timestamps'][()]
                msg = f'; vs NWB_ks4 part 1: {len(to)} frames, max |diff| first {min(len(to), N1)} ' \
                      f'{np.max(np.abs(ts[:min(len(to), N1)] - to[:min(len(to), N1)])) * 1e3:.3f} ms'
            out.append(f'{k.split(chr(92))[-1]}: {len(ts)} timestamps, part 1 {int((ts < tj).sum())}, part 2 '
                       f'{int((ts >= t2).sum())}, frame interval part 2 median {np.median(np.diff(ts[ts >= t2])) * 1e3:.3f} ms{msg}')

    # stimulus TTLs vs trials
    tr = f['intervals/trials']
    st = tr['start_time'][()]
    for name, col in (('whisker_stim_times', 'whisker_stim_time'), ('auditory_stim_times', 'auditory_stim_time')):
        ev = np.loadtxt(rf'{SYNC}\{name}.txt')
        ev = np.where(ev >= tj, ev + shift, ev)
        tt = tr[col][()]
        tt = tt[np.isfinite(tt)]
        for part, m_ev, m_tt in ((1, ev < tj, tt < tj), (2, ev >= t2, tt >= t2)):
            a, b = ev[m_ev], tt[m_tt]
            d = np.abs(b[:, None] - a[None, :]).min(1) if len(a) and len(b) else np.array([np.nan])
            out.append(f'{name} part {part}: {m_ev.sum()} TTLs vs {m_tt.sum()} trial times; max |trial - nearest TTL| '
                       f'{np.nanmax(d) * 1e3:.2f} ms')

    # piezo raw trace vs lick times (part 1; part 2 has no licks)
    lt = f['processing/behavior/BehavioralTimeSeries/lick_trace']
    lts, lx = lt['timestamps'][()], lt['data'][()]
    licks = f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][()]
    for lab, sel in (('first 200', licks[:200]), ('last 200', licks[-200:])):
        lag, _ = peak_latency(sel, lts, np.abs(lx), win=(-0.1, 0.1), dt=0.001)
        out.append(f'lick_trace peak relative to piezo_lick_times ({lab}, at ~{np.median(sel):.0f} s): {lag:+.0f} ms')
    p2 = lx[lts >= t2]
    out.append(f'lick_trace: {len(lts)} bins, {lts[0]:.3f}-{lts[-1]:.3f} s, part 2 bins {len(p2)}; max |trace| '
               f'part 1 {np.nanmax(np.abs(lx[lts < tj])):.0f}, part 2 {np.nanmax(np.abs(p2)):.0f}')

    # DLC and spikes: whisker-evoked responses, passive pre vs passive post
    ctx = dec(tr['context'][()])
    wh = tr['whisker_stim'][()] == 1
    wst = tr['whisker_stim_time'][()]
    pre = wst[wh & (ctx == 'passive') & (st < tj)]
    post = wst[wh & (st >= t2)]
    bts = f['processing/behavior/BehavioralTimeSeries']
    ts = bts['whisker_angle/timestamps'][()]
    ang = bts['whisker_angle/data'][()].astype(float)
    spd = np.abs(np.gradient(ang, ts))
    for lab, ev in (('passive pre', pre), ('passive post', post)):
        lag, m = peak_latency(ev, ts, spd)
        out.append(f'DLC whisker speed after whisker stim, {lab} ({len(ev)} trials): peak at {lag:.0f} ms, '
                   f'peak {np.nanmax(m):.0f} deg/s above baseline')
    idx = f['units/spike_times_index'][()]
    sp = f['units/spike_times']
    starts = np.r_[0, idx[:-1]]
    edges = np.arange(-0.05, 0.1 + 1e-9, 0.002)
    for lab, ev in (('passive pre', pre), ('passive post', post)):
        h = np.zeros(len(edges) - 1)
        for i in range(len(idx)):
            s = sp[starts[i]:idx[i]]
            for e in ev:
                a, b = np.searchsorted(s, [e - 0.05, e + 0.1])
                h += np.histogram(s[a:b] - e, edges)[0]
        c = (edges[:-1] + edges[1:]) / 2
        base = h[c < 0].mean()
        on = c[np.flatnonzero(h > base + 0.5 * (h.max() - base))[0]]
        out.append(f'population spikes after whisker stim, {lab}: half-max onset {on * 1e3:.0f} ms, peak '
                   f'{c[h.argmax()] * 1e3:.0f} ms, peak/baseline {h.max() / base:.2f}')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
