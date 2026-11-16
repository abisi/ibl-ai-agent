"""014 -- Dataset-wide audit of DLC alignment in NWB_ks4, on the timestamps stored in the NWB (what analyses read)
(user 2026-11-16: "review, for the entire dataset, whether DLC is correctly aligned and the NWB files are correct").

Per session NWB with DLC (jaw_angle or whisker_angle series):
  structure : frames vs timestamps per camera, timestamp gaps (> 7.5 ms) / repeats (< 2.5 ms), DLC time span vs trial
              span, NaN fraction, tracking quality (fraction of frames with likelihood >= 0.6 per keypoint), top
              orientation (median top nose_tip_y - nose_base_y over confident frames; upright > +20 px).
  side      : residual lag (frames) between the side signal (tongue likelihood; jaw distance when the tongue is rarely
              tracked) and the piezo licks, in windows of >= 25 licks over the session, on the NWB time grid; aligned =
              flat at about +3..+9 frames (tongue before the piezo contact). Plus 010's per-trial check: jaw opening -
              first lick within +-100 ms of the session median.
  top       : residual lag (frames) of the top camera against the side camera from the same body part: top nose-tip
              speed vs side nose-tip speed on a common 200 Hz grid built from each camera's NWB timestamps, 60 s
              windows (step 30 s), lags +-LAG_TOP; aligned = flat at about 0. No stimulus or behaviour assumption.
Window acceptance: correlation peak >= Z_MIN robust SDs above the median over lags (and >= 10 licks for the side), and
a neighbouring accepted window (+-2) within 4 frames (isolated peaks are spurious).
Flags (per camera): lag range over accepted windows (5th-95th pct) > RANGE_TOL frames, median lag outside the expected
band (side +-2..15, top +-3), accepted windows lost in part of the session, side jaw-lick consistency < 0.6, and
band, too few accepted windows, frames != timestamps, timestamp gaps, DLC not covering the trials.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/audit_014/ sessions.csv (one row per session), windows.csv
(accepted and rejected windows), audit_014_<date>.pdf (summary + offset curves of flagged sessions)
Run (haas, ibl-ai-agent .venv): python 014_dataset_alignment_audit.py [--jobs 60] [--sessions ...] [--plot-only]
"""
import argparse
import datetime
import importlib.util
import pathlib
import re

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import correlate

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m006', HERE / '006_pairing_evidence.py')
m006 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m006)
OUT = m006.OUT / 'audit_014'
SESSION_NWB = re.compile(r'^[A-Z]+\d+_\d{8}_\d{6}\.nwb$')
FR = 200.0
LIK = 0.6
Z_MIN = 8.0
LAG_SIDE = 4000  # +-20 s
LAG_TOP = 12000  # +-60 s (AB092 top drifts by 40 s)
WIN_TOP, STEP_TOP = 12000, 6000
MIN_LICKS = 25
RANGE_TOL = 10
SIDE_BAND = (-2, 15)
TOP_BAND = (-3, 3)
MIN_JAW_CONSISTENCY = 0.6
GRID = np.round(np.arange(-1.0, 2.0001, 0.01), 3)


def zs(x):
    x = np.asarray(x, float)
    s = np.nanstd(x)
    return np.nan_to_num((x - np.nanmean(x)) / (s if s > 0 else 1))


def window_lags(x, e, lo, hi, win, step, min_events=0, ev_idx=None):
    """Windows of x (signal on frames) against e (event train on the same frames) over lags [lo, hi] (FFT)."""
    rows = []
    n = len(x)
    for f0 in range(0, max(n - win // 2, 1), step):
        xw = x[f0:f0 + win]
        a, b = max(f0 + lo, 0), min(f0 + len(xw) + hi, len(e))
        if b - a < len(xw):
            continue
        c = correlate(e[a:b], xw, mode='valid', method='fft')  # c[k]: lag = a - f0 + k
        if not len(c) or not np.isfinite(c).all():
            continue
        k = int(np.argmax(c))
        mad = np.median(np.abs(c - np.median(c))) or 1e-9
        n_ev = int(np.sum((ev_idx >= f0) & (ev_idx < f0 + len(xw)))) if ev_idx is not None else 0
        rows.append(dict(frame=f0 + len(xw) // 2, lag=a - f0 + k, z=float((c[k] - np.median(c)) / mad), n_events=n_ev,
                         accepted=bool((c[k] - np.median(c)) / mad >= Z_MIN and n_ev >= min_events)))
    wl = pd.DataFrame(rows)
    if len(wl):  # isolated peaks are spurious: keep a window only if a neighbour (+-2 windows) agrees within 4 frames
        lag, ok = wl.lag.to_numpy(), wl.accepted.to_numpy()
        keep = ok.copy()
        idx = np.flatnonzero(ok)
        for i in idx:
            nb = idx[(np.abs(idx - i) <= 2) & (idx != i)]
            keep[i] = len(nb) > 0 and np.any(np.abs(lag[nb] - lag[i]) <= 4)
        wl['accepted'] = keep
    return wl


def lik_frac(b, k):
    return float(np.mean(b[f'{k}_likelihood']['data'][::10] >= LIK)) if f'{k}_likelihood' in b else np.nan


def speed(b, k, n):
    x, y = b[f'{k}_x']['data'][:n].astype(float), b[f'{k}_y']['data'][:n].astype(float)
    ok = b[f'{k}_likelihood']['data'][:n] >= LIK
    v = np.r_[0, np.hypot(np.diff(x), np.diff(y))]
    v[~(ok & np.r_[False, ok[:-1]])] = 0
    return np.clip(v, 0, np.nanpercentile(v, 99.9) if np.any(v > 0) else 1)


def summarize(wl, band, view, row):
    acc = wl[wl.accepted] if len(wl) else wl
    row[f'{view}_n_win'] = len(wl)
    row[f'{view}_n_acc'] = len(acc)
    if len(acc) >= 3:
        lag = acc.lag.to_numpy(float)
        row[f'{view}_lag_median'] = float(np.median(lag))
        row[f'{view}_lag_range'] = float(np.percentile(lag, 95) - np.percentile(lag, 5))
        third = np.array_split(acc.sort_values('frame').lag.to_numpy(float), 3)
        row[f'{view}_lag_first_third'] = float(np.median(third[0]))
        row[f'{view}_lag_last_third'] = float(np.median(third[-1]))
        flags = []
        if row[f'{view}_lag_range'] > RANGE_TOL:
            flags.append('lag not flat')
        if not band[0] <= row[f'{view}_lag_median'] <= band[1]:
            flags.append('lag offset')
        # accepted windows missing in part of the session (lag beyond the search range or pairing broken there)
        if len(wl) >= 6:
            fr = [float(np.mean(t)) for t in np.array_split(wl.sort_values('frame').accepted.to_numpy(float), 3)]
            row[f'{view}_acc_frac_thirds'] = '/'.join(f'{v:.2f}' for v in fr)
            if min(fr) < 0.2 and max(fr) > 0.5:
                flags.append('windows lost in part of the session')
        row[f'{view}_flag'] = '; '.join(flags)
    else:
        row[f'{view}_flag'] = 'too few accepted windows'
    return row


def audit(session_id):
    row = dict(session_id=session_id)
    wins = []
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        if 'processing/behavior/BehavioralTimeSeries' not in f:
            row['has_dlc'] = False
            return row, wins
        b = f['processing/behavior/BehavioralTimeSeries']
        has_side, has_top = 'jaw_angle' in b, 'whisker_angle' in b
        row.update(has_dlc=has_side or has_top, has_side=has_side, has_top=has_top)
        if not (has_side or has_top):
            return row, wins
        desc = f['session_description'][()]
        row['description'] = (desc.decode() if isinstance(desc, bytes) else str(desc))[:40]
        tr = f['intervals/trials']
        starts = tr['start_time'][:]
        row['trial_span'] = f'{starts.min():.0f}-{tr["stop_time"][:].max():.0f}' if len(starts) else ''
        lk_path = 'processing/behavior/BehavioralEvents/piezo_lick_times/timestamps'
        licks = np.sort(f[lk_path][:]) if lk_path in f else np.array([])
        row['n_licks'] = len(licks)
        ts = {}
        for view, ser in (('side', 'jaw_angle'), ('top', 'whisker_angle')):
            if ser not in b:
                continue
            t = b[ser]['timestamps'][:]
            n = len(b[ser]['data'])
            ts[view] = t[:n]
            d = np.diff(t[:n])
            row[f'{view}_n_frames'] = n
            row[f'{view}_n_ts'] = len(t)
            row[f'{view}_ts_gaps'] = int(np.sum(d > 0.0075))
            row[f'{view}_ts_repeats'] = int(np.sum(d < 0.0025))
            row[f'{view}_ts_nonmono'] = int(np.sum(d <= 0))
            row[f'{view}_t0'], row[f'{view}_t1'] = float(np.nanmin(t[:n])), float(np.nanmax(t[:n]))
            row[f'{view}_nan_frac'] = float(np.mean(~np.isfinite(b[ser]['data'][::10])))
            if len(starts):
                row[f'{view}_covers_trials'] = bool(row[f'{view}_t0'] <= starts.min() + 1 and
                                                    row[f'{view}_t1'] >= tr['stop_time'][:].max() - 1)
        for k in ('jaw', 'tongue', 'nose_tip', 'top_nose_tip', 'whisker_base', 'top_particle'):
            row[f'lik_{k}'] = lik_frac(b, k)
        if has_top and 'top_nose_tip_y' in b and 'top_nose_base_y' in b:
            n = len(ts['top'])
            ok = (b['top_nose_tip_likelihood']['data'][:n] >= LIK) & (b['top_nose_base_likelihood']['data'][:n] >= LIK)
            row['top_nose_conf_frac'] = float(ok.mean())
            if ok.sum() > 100:
                row['top_orientation_px'] = float(np.median(b['top_nose_tip_y']['data'][:n][ok] -
                                                            b['top_nose_base_y']['data'][:n][ok]))
        # side: signal vs piezo licks on the NWB time grid
        if has_side and len(licks) >= MIN_LICKS:
            t = ts['side']
            okt = np.isfinite(t)
            use_tongue = row['lik_tongue'] >= 0.01
            sig = b['tongue_likelihood']['data'][:len(t)] if use_tongue else b['jaw_distance']['data'][:len(t)]
            x = zs(np.where(okt, sig, np.nan))
            tt = np.where(okt, t, np.inf)
            i = np.searchsorted(tt, licks)
            i = i[(i > 0) & (i < len(t))]
            e = np.zeros(len(t))
            for k in range(-2, 3):
                j = i + k
                e[j[(j >= 0) & (j < len(t))]] += 1
            rate = len(licks) / max(np.nanmax(t) - np.nanmin(t), 1)
            win = int(np.clip(FR * MIN_LICKS / max(rate, 1e-3), 12000, 120000))
            wl = window_lags(x, e, -LAG_SIDE, LAG_SIDE, win, win // 2, min_events=10, ev_idx=i)
            wl['view'] = 'side'
            wins.append(wl)
            row['side_signal'] = 'tongue' if use_tongue else 'jaw'
            row = summarize(wl, SIDE_BAND, 'side', row)
            # per-trial jaw opening - first lick (010)
            nxt = np.searchsorted(licks, starts)
            first = np.where(nxt < len(licks), licks[np.minimum(nxt, len(licks) - 1)], np.nan)
            first[(first - starts) > 1.5] = np.nan
            lt = np.flatnonzero(np.isfinite(first))
            if len(lt) >= 10:
                jaw = zs(b['jaw_distance']['data'][:len(t)])
                m = m006.triggered(jaw, np.where(okt, t, np.nan), starts[lt], GRID)
                pre, post = GRID < 0, (GRID >= 0) & (GRID <= 1.5)
                base = np.nanmedian(m[:, pre], 1)
                peak = np.nanmax(np.where(post, m, -np.inf), 1)
                above = (m >= (base + 0.5 * (peak - base))[:, None]) & post[None, :]
                has = above.any(1) & np.isfinite(base) & (peak - base > 1.0)
                on = np.full(len(lt), np.nan)
                on[has] = GRID[np.argmax(above[has], 1)]
                dl = on - (first[lt] - starts[lt])
                med = np.nanmedian(dl)
                row['side_jaw_lick_median_ms'] = float(med * 1e3)
                row['side_jaw_consistency'] = float(np.mean(np.abs(dl - med) <= 0.1))
                row['n_lick_trials'] = len(lt)
                if row['side_jaw_consistency'] < MIN_JAW_CONSISTENCY:
                    row['side_flag'] = '; '.join(x for x in (row.get('side_flag', ''), 'jaw-lick inconsistent') if x)
        elif has_side:
            row['side_flag'] = 'too few licks'
        # top vs side: nose-tip speed on a common 200 Hz grid
        if has_side and has_top and 'nose_tip_x' in b and 'top_nose_tip_x' in b:
            t0 = np.nanmin([np.nanmin(ts['side']), np.nanmin(ts['top'])])
            t1 = np.nanmax([np.nanmax(ts['side']), np.nanmax(ts['top'])])
            ng = int((t1 - t0) * FR) + 2
            g = {}
            for view, k in (('side', 'nose_tip'), ('top', 'top_nose_tip')):
                t = ts[view]
                v = zs(speed(b, k, len(t)))
                idx = np.round((t - t0) * FR)
                ok = np.isfinite(idx)
                arr = np.zeros(ng)
                arr[idx[ok].astype(int)] = v[ok]
                g[view] = arr
            wl = window_lags(g['top'], g['side'], -LAG_TOP, LAG_TOP, WIN_TOP, STEP_TOP)
            # lag here: side bin = top bin + lag; a top camera stamped too early has the side event later -> lag > 0
            wl['view'] = 'top'
            wins.append(wl)
            row = summarize(wl, TOP_BAND, 'top', row)
    for w in wins:
        w['session_id'] = session_id
    return row, wins


def plot(rows, wins, out):
    s = pd.DataFrame(rows)
    w = pd.concat(wins) if wins else pd.DataFrame()
    with PdfPages(out) as pdf:
        fig, axs = plt.subplots(2, 3, figsize=(16.5, 9))
        for c, view in enumerate(('side', 'top')):
            if f'{view}_lag_median' not in s:
                continue
            ax = axs[c, 0]
            ax.hist(s[f'{view}_lag_median'].dropna().clip(-50, 50), 101, color='0.4')
            ax.set_title(f'{view}: median residual lag per session (frames, clipped +-50)', fontsize=9)
            ax = axs[c, 1]
            ax.hist(s[f'{view}_lag_range'].dropna().clip(0, 100), 101, color='0.4')
            ax.axvline(RANGE_TOL, color='r')
            ax.set_title(f'{view}: lag range 5-95 pct over the session (frames, clipped 100)', fontsize=9)
            ax = axs[c, 2]
            vc = s[f'{view}_flag'].fillna('ok').replace('', 'ok').value_counts()
            ax.barh(vc.index, vc.values, color='0.4')
            ax.set_title(f'{view}: flags (n sessions)', fontsize=9)
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        for col in ('side_flag', 'top_flag'):
            if col not in s:
                s[col] = ''
        sf, tf = s.side_flag.fillna(''), s.top_flag.fillna('')
        flagged = s[((sf.str.len() > 0) & (sf != 'too few licks')) | (tf.str.len() > 0)]
        for i in range(0, len(flagged), 6):
            fig, axs = plt.subplots(6, 2, figsize=(16.5, 22), squeeze=False)
            for r, (_, ss) in enumerate(flagged.iloc[i:i + 6].iterrows()):
                for c, view in enumerate(('side', 'top')):
                    ax = axs[r, c]
                    ww = w[(w.session_id == ss.session_id) & (w.view == view)] if len(w) else w
                    if not len(ww):
                        ax.set_title(f'{ss.session_id} {view}: no windows ({ss.get(f"{view}_flag", "")})', fontsize=8)
                        continue
                    ax.scatter(ww.frame / FR, ww.lag, s=4, color='0.8')
                    a = ww[ww.accepted]
                    ax.scatter(a.frame / FR, a.lag, s=8, c=np.clip(a.z, 0, 30), cmap='viridis')
                    lo, hi = (np.percentile(a.lag, [1, 99]) if len(a) else (-50, 50))
                    ax.set_ylim(min(lo, -20) - 10, max(hi, 20) + 10)
                    ax.axhspan(*(SIDE_BAND if view == 'side' else TOP_BAND), color='g', alpha=0.1)
                    ax.set_title(f'{ss.session_id} {view}: {ss.get(f"{view}_flag", "")}', fontsize=8)
                    ax.set_xlabel('frame / 200 (s)', fontsize=7)
                    ax.set_ylabel('residual lag (frames)', fontsize=7)
                    ax.tick_params(labelsize=6)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=60)
    ap.add_argument('--sessions', nargs='*')
    ap.add_argument('--plot-only', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        s = pd.read_csv(OUT / 'sessions.csv')
        w = pd.read_csv(OUT / 'windows.csv')
        plot(s.to_dict('records'), [w], OUT / f'audit_014_{datetime.date.today()}.pdf')
        return
    sessions = args.sessions or sorted(p.stem for p in m006.NWB_DIR.glob('*.nwb') if SESSION_NWB.match(p.name))
    from joblib import Parallel, delayed

    def safe(sid):
        try:
            return audit(sid)
        except Exception as ex:  # noqa: BLE001  (reported in the table)
            return dict(session_id=sid, error=f'{type(ex).__name__}: {ex}'), []
    res = Parallel(n_jobs=args.jobs, backend='loky', verbose=5)(delayed(safe)(s) for s in sessions)
    rows = [r for r, _ in res]
    wins = [w for _, ws in res for w in ws]
    tag = '' if not args.sessions else '_test'
    pd.DataFrame(rows).to_csv(OUT / f'sessions{tag}.csv', index=False)
    if wins:
        pd.concat(wins).to_csv(OUT / f'windows{tag}.csv', index=False)
    plot(rows, wins, OUT / f'audit_014_{datetime.date.today()}{tag}.pdf')
    print(OUT)


if __name__ == '__main__':
    main()
