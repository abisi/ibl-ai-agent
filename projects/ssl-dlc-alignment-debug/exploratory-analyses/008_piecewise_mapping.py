"""008 -- Piecewise DLC frame -> camera pulse mapping for sessions with frames dropped during the video (see
../question.md, 005, 006).

Dropped frames: the camera pulse is recorded but the frame is not saved, so the offset D(f) = pulse index - DLC frame
index can only stay constant or increase along the video (a non-decreasing step function). Per session and camera:
1. offset in short windows (side 30 s, top 60 s): lag maximising the correlation of the DLC signal with the piezo lick
   train on the pulse grid (side: tongue_likelihood, top: whisker speed |d whisker_angle|), within the plausible range;
2. non-decreasing (isotonic, pool-adjacent-violators) fit of the accepted window lags -> plateaus;
3. each step placed at the frame that maximises the lick correlation of the two-plateau model over the frames between
   the plateaus (uncertainty = span between the closest licks on either side);
4. mapping D(f) = plateau lag - lead; side lead = 5 frames (tongue peak before the lick in in-register sessions; the
   session start is not assumed in register); top: same start offset as the side camera;
5. validation: lick-locked tongue / whisker peak latency per segment, NWB pairing vs piecewise mapping.
Outputs: tables/piecewise_<session>_<view>.csv (frame_start, frame_end, D), tables/piecewise_validation.csv,
figures/<session>_piecewise.png
Run (haas, ibl-ai-agent .venv): python 008_piecewise_mapping.py SESSION_ID [SESSION_ID ...]
"""
import importlib.util
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m006', HERE / '006_pairing_evidence.py')
m006 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m006)
OUT = m006.OUT
WIN = {'side': 12000, 'top': 24000}
STEP = {'side': 6000, 'top': 12000}
MIN_Z, MIN_EVENTS = 8.0, 10
# sessions where the tongue is rarely tracked: side camera from jaw_distance (lead ~+6 frames in in-register sessions)
JAW_SESSIONS = {'AB116_20240726_094206': 6, 'AB116_20240725_094541': 6, 'AB116_20240724_102941': 6}
SIDE_LEAD = 5  # tongue_likelihood peak precedes the piezo lick by ~5 frames in in-register sessions
POOL = 15  # frames searched around a plateau lag when re-estimating it from all frames of its segment


def signals(session_id):
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        licks = np.sort(f['processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'][:])
        out = {}
        if session_id in JAW_SESSIONS and 'jaw_distance' in b:
            out['side'] = (np.nan_to_num(m006.zs(b['jaw_distance']['data'][:])), b['jaw_angle']['timestamps'][:])
        elif 'tongue_likelihood' in b:
            out['side'] = (np.nan_to_num(m006.zs(b['tongue_likelihood']['data'][:])), b['jaw_angle']['timestamps'][:])
        if 'whisker_angle' in b:
            w = b['whisker_angle']['data'][:].astype(float)
            out['top'] = (np.nan_to_num(m006.zs(np.abs(np.diff(w, prepend=w[0])))), b['whisker_angle']['timestamps'][:])
    _, nidq, _, _ = m006.load(session_id)
    return out, nidq, licks


def train(pulses, events, n, smooth=2):
    e = np.zeros(n)
    i = np.searchsorted(pulses, events)
    i = i[(i > 0) & (i < n)]
    for k in range(-smooth, smooth + 1):
        j = i + k
        e[j[(j >= 0) & (j < n)]] += 1
    return e


def window_lags(x, e, lo, hi, win, step):
    rows = []
    for f0 in range(0, len(x) - win // 2, step):
        xw = x[f0:f0 + win]
        a, b = f0 + lo, f0 + len(xw) + hi
        if a < 0 or b > len(e):
            continue
        c = np.correlate(e[a:b], xw, 'valid')  # c[k] = sum_f xw[f] e[f0 + f + lo + k]
        k = int(np.argmax(c))
        mad = np.median(np.abs(c - np.median(c))) or 1
        rows.append(dict(frame=f0 + len(xw) // 2, f0=f0, f1=f0 + len(xw), lag=lo + k, z=(c[k] - np.median(c)) / mad,
                         n_events=int((e[f0 + lo:f0 + lo + len(xw)] > 0).sum() // 5)))
    return pd.DataFrame(rows)


def pava(y, w):
    """Non-decreasing least-squares fit (pool adjacent violators)."""
    blocks = [[yi, wi, 1] for yi, wi in zip(y, w)]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1 and out[-2][0] > out[-1][0]:
            v2, w2, n2 = out.pop()
            v1, w1, n1 = out.pop()
            out.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    fit = []
    for v, _, nn in out:
        fit += [v] * nn
    return np.array(fit)


def consistent(lags, ok, k=2, tol=4):
    """Keep an accepted window only if an accepted window within k windows has a lag within tol frames."""
    lags = np.asarray(lags)
    keep = ok.copy()
    idx = np.flatnonzero(ok)
    for i in idx:
        nb = idx[(np.abs(idx - i) <= k) & (idx != i)]
        keep[i] = len(nb) > 0 and np.any(np.abs(lags[nb] - lags[i]) <= tol)
    return keep


def pooled_lag(x, e, a, b, lag, half=POOL):
    """Lag in [lag - half, lag + half] maximising the lick correlation over all frames [a, b) of a segment."""
    lo = lag - half
    if a + lo < 0 or b + lag + half > len(e) or b - a < 10:
        return lag
    c = np.correlate(e[a + lo:b + lag + half], x[a:b], 'valid')
    return int(lo + np.argmax(c))


def segments_from_steps(x, e, bounds, lags):
    """Re-estimate each plateau lag from its whole segment, then re-place the steps (one iteration)."""
    lags = [pooled_lag(x, e, a, b, int(l)) for a, b, l in zip(bounds[:-1], bounds[1:], lags)]
    new_b = [bounds[0]]
    for i in range(1, len(lags)):
        if lags[i] - lags[i - 1] >= 2:
            fa = max(new_b[-1] + 1, bounds[i] - WIN['side'])
            fb = min(bounds[i + 1] - 1, bounds[i] + WIN['side'])
            b, _, _ = place_step(x, e, lags[i - 1], lags[i], fa, fb)
            new_b.append(b)
        else:
            lags[i] = lags[i - 1]
            new_b.append(bounds[i])
    new_b.append(bounds[-1])
    # merge plateaus with equal lags
    mb, ml = [new_b[0]], [lags[0]]
    for i in range(1, len(lags)):
        if lags[i] != ml[-1]:
            mb.append(new_b[i])
            ml.append(lags[i])
    mb.append(new_b[-1])
    return mb, ml


def place_step(x, e, la, lb, fa, fb):
    """Frame b in [fa, fb) maximising sum_{f<b} x e[f+la] + sum_{f>=b} x e[f+lb]."""
    f = np.arange(fa, fb)
    ok = (f + lb < len(e)) & (f + la >= 0)
    f = f[ok]
    ca = np.cumsum(x[f] * e[f + la])
    cb = np.cumsum((x[f] * e[f + lb])[::-1])[::-1]
    score = np.r_[0, ca] + np.r_[cb, 0]
    k = int(np.argmax(score))
    b = int(f[0] + k) if k < len(f) else int(f[-1] + 1)
    ev = np.flatnonzero(e[f + la] + e[f + lb] > 0)
    before = f[ev[f[ev] < b]].max() if (f[ev] < b).any() else fa
    after = f[ev[f[ev] >= b]].min() if (f[ev] >= b).any() else fb
    return b, int(before), int(after)


def latency(x, t, events, grid=np.round(np.arange(-0.3, 0.3001, 0.005), 3)):
    if len(events) < 5:
        return np.nan
    m = m006.triggered(x, t, events, grid)
    mean = np.nanmean(m, 0)
    return float(grid[np.nanargmax(mean)] * 1e3) if np.isfinite(mean).any() else np.nan


def main(session_id):
    sig, nidq, licks = signals(session_id)
    bouts = licks[np.r_[True, np.diff(licks) > 1.0]]
    fig, axs = plt.subplots(len(sig), 3, figsize=(20, 4.6 * len(sig)), squeeze=False, gridspec_kw=dict(width_ratios=[2.0, 1, 1]))
    vrows = []
    side_steps = {}
    for r, (view, (x, stored)) in enumerate(sorted(sig.items(), key=lambda kv: kv[0] != 'side')):
        n = len(x)
        pulses, d_nwb, src = m006.pulses_on_nwb_clock(stored, nidq.get(view))
        extra = len(pulses) - n
        lo, hi = min(-60, extra - 60), max(extra, 0) + 60  # more frames than pulses (AB085): negative lags too
        e = train(pulses, licks, len(pulses))
        wl = window_lags(x, e, lo, hi, WIN[view], STEP[view])
        ok = ((wl.z >= MIN_Z) & (wl.n_events >= MIN_EVENTS)).to_numpy()
        ok = consistent(wl.lag.to_numpy(), ok)
        acc = wl[ok].reset_index(drop=True)
        ax = axs[r, 0]
        ax.scatter(wl.frame / 200, wl.lag, s=5, color='0.8', label='all windows')
        steps = []
        if view == 'top' and 'side' in side_steps:
            # top signal is weak (whisker speed vs licks): side step times, top plateau lags pooled per segment
            sb = [0] + [st for st in side_steps['side'] if st < n] + [n]
            top_lead = pooled_lag(x, e, sb[0], sb[1], 0, half=60)
            seeds = [top_lead + (sl - side_steps['lags'][0]) for sl in side_steps['lags'][:len(sb) - 1]]
            bounds, lags = segments_from_steps(x, e, sb, seeds)
            source = 'side step times'
        else:
            if len(acc) < 2:
                ax.set_title(f'{view}: too few accepted windows ({len(acc)})', fontsize=8)
                continue
            fit = np.round(pava(acc.lag.to_numpy(float), acc.z.to_numpy()))
            acc['fit'] = fit
            bounds, lags = [0], [fit[0]]
            for i in range(1, len(acc)):
                if fit[i] - fit[i - 1] >= 3:
                    fa, fb = int(acc.f0[i - 1] + WIN[view] // 2), int(acc.f0[i] + WIN[view] // 2)
                    bstep, _, _ = place_step(x, e, int(fit[i - 1]), int(fit[i]), fa, fb)
                    bounds.append(bstep)
                    lags.append(fit[i])
            bounds.append(n)
            bounds, lags = segments_from_steps(x, e, bounds, lags)
            source = 'own windows'
        # segments with < 5 lick bouts carry no evidence: keep the previous segment's lag
        t_tmp = m006.frame_times(n, pulses, d_nwb)
        kb, kl = [bounds[0]], [lags[0]]
        for i in range(1, len(lags)):
            a_, b_ = bounds[i], bounds[i + 1] - 1
            nb = np.sum((bouts > t_tmp[a_]) & (bouts < t_tmp[min(b_, n - 1)])) if np.isfinite(t_tmp[a_]) else 0
            def seg_c(L, a_=a_, b_=b_):
                bb = min(b_, len(e) - L)  # last segment: stop at the end of the pulse train
                return float(np.dot(x[a_:bb], e[a_ + L:bb + L])) if bb > a_ and a_ + L >= 0 else -np.inf
            supported = seg_c(int(lags[i])) > 1.3 * max(seg_c(int(kl[-1])), 1e-9)
            if nb >= 5 and lags[i] != kl[-1] and supported:
                kb.append(bounds[i])
                kl.append(lags[i])
        bounds, lags = kb + [n], kl
        if view == 'side':
            side_steps['side'] = bounds[1:-1]
            side_steps['lags'] = list(lags)
        for i in range(1, len(lags)):
            steps.append(dict(frame=bounds[i], from_lag=lags[i - 1], to_lag=lags[i], uncertainty_frames=0))
        # lead of the signal in in-register sessions (pooled lag, AB151 / MH015 / MH028 / AB127 / MH064: tongue +3..+7
        # frames); the session start is not assumed in register (MH017 starts ~38 frames off). Top camera: whisker speed
        # lead is too variable across sessions, so the top start offset is taken equal to the side camera's.
        if view == 'side':
            side_lead = JAW_SESSIONS.get(session_id, SIDE_LEAD)
            lead = side_lead
            side_steps['D0'] = int(lags[0] - side_lead)
        else:
            lead = lags[0] - side_steps.get('D0', 0)
        seg_bounds, seg_lags = bounds, lags
        segs = pd.DataFrame(dict(frame_start=seg_bounds[:-1], frame_end=seg_bounds[1:], lag=seg_lags))
        segs['D'] = (segs.lag - lead).astype(int) + d_nwb
        segs.to_csv(OUT / 'tables' / f'piecewise_{session_id}_{view}.csv', index=False)
        D = np.repeat(segs.D.to_numpy(), (segs.frame_end - segs.frame_start).to_numpy())
        f = np.arange(n)
        t_new = np.full(n, np.nan)
        ok = (f + D >= 0) & (f + D < len(pulses))
        t_new[ok] = pulses[f[ok] + D[ok]]
        t_old = m006.frame_times(n, pulses, d_nwb)
        if len(acc):
            ax.scatter(acc.frame / 200, acc.lag, c=np.clip(acc.z, 0, 30), cmap='viridis', s=14, label='accepted windows')
        ax.step(np.r_[segs.frame_start, n] / 200, np.r_[segs.lag, segs.lag.iloc[-1]], where='post', color='r', lw=1.2,
                label='piecewise fit (steps placed on licks)')
        ax.set_ylim(min(seg_lags) - 30, max(seg_lags) + 30)
        ax.set_xlabel('DLC frame / 200 (s of video)', fontsize=8)
        ax.set_ylabel('pulse index - DLC frame (lag)', fontsize=8)
        ax.set_title(f'{view} ({source}): {n} frames, {len(pulses)} pulses; lead {lead:.0f}; {len(steps)} steps: ' +
                     ', '.join(f"{s_['frame'] / 200:.0f}s +{s_['to_lag'] - s_['from_lag']:.0f}" for s_ in steps[:8]),
                     fontsize=7)
        ax.legend(fontsize=7)
        ax = axs[r, 1]
        for k_, (a, b) in enumerate(zip(segs.frame_start, segs.frame_end)):
            tseg = (t_new[a], t_new[min(b, n) - 1]) if np.isfinite(t_new[a]) else (np.nan, np.nan)
            ev = bouts[(bouts > tseg[0]) & (bouts < tseg[1])]
            lat_old = latency(x, t_old, ev)
            lat_new = latency(x, t_new, ev)
            vrows.append(dict(session_id=session_id, view=view, segment=k_, frame_start=a, frame_end=b, D=int(segs.D[k_]),
                              n_bouts=len(ev), latency_nwb_ms=lat_old, latency_piecewise_ms=lat_new))
        grid = np.round(np.arange(-1.5, 1.5001, 0.01), 3)
        ev = bouts[(bouts > pulses[0] + 2) & (bouts < pulses[-1] - 2)]
        for c_, (tt, lab) in enumerate(((t_old, 'NWB pairing'), (t_new, 'piecewise mapping')), start=1):
            m = m006.triggered(x, tt, ev, grid)
            ax = axs[r, c_]
            lo_, hi_ = (np.nanpercentile(m, [1, 99]) if np.isfinite(m).any() else (0, 1))
            ax.imshow(m, aspect='auto', cmap='magma', vmin=lo_, vmax=hi_, extent=[grid[0], grid[-1], len(ev), 0],
                      interpolation='nearest')
            ax.axvline(0, color='c', lw=0.5)
            for st in steps:
                if np.isfinite(t_new[min(st['frame'], n - 1)]):
                    ax.axhline(np.searchsorted(ev, t_new[min(st['frame'], n - 1)]), color='w', lw=0.4, ls=':')
            ax.set_title(f'{view} {"tongue likelihood" if view == "side" else "whisker speed"}, {lab}', fontsize=8)
            ax.set_xlabel('time from lick-bout onset (s); dotted = steps', fontsize=7)
            ax.set_ylabel('lick bout (session order)', fontsize=7)
    fig.suptitle(f'{session_id}: piecewise DLC frame -> pulse mapping', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f'{session_id}_piecewise.png', dpi=120)
    plt.close(fig)
    t = pd.DataFrame(vrows)
    print(t.round(1).to_string(index=False))
    out = OUT / 'tables' / 'piecewise_validation.csv'
    old = pd.read_csv(out) if out.exists() else pd.DataFrame()
    if len(old):
        old = old[old.session_id != session_id]
    pd.concat([old, t]).to_csv(out, index=False)


if __name__ == '__main__':
    for s in sys.argv[1:]:
        try:
            main(s)
        except Exception as ex:
            print(s, 'ERROR', type(ex).__name__, ex)
