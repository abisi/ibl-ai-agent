"""005 -- Which camera pulse does each DLC frame belong to? Offset curve over the session, per camera (see ../question.md).

For each camera, in sliding windows of DLC frames, cross-correlate a DLC movement signal with behavioural events placed
on the camera pulse train (event -> index of the nearest pulse), over all lags (FFT). The best lag L in a window means:
DLC frame f belongs to pulse f + L. A constant L over the session = frames and pulses in register (start-aligned: L ~ 0
plus the physiological lead / lag of the signal, a few frames); a step = frames dropped / added at that time; no peak =
no usable events in the window.
  side camera: tongue_likelihood (z) vs piezo lick times (NWB piezo_lick_times)
  top camera : summed |frame-to-frame change| (z) of top_particle_y, whisker_angle, whisker_tip_x vs whisker stimulus
               times (trials whisker_stim_time), and vs lick times as a second marker
Pulses: ephys sessions = raw NIDQ camera pulses (sync_event_times cam0_frame_times = top, cam1_frame_times = side, all
runs); behaviour-only sessions = the camera timestamps stored in the NWB (top: whisker_angle, side: jaw_angle).
DLC frames: the NWB DLC series, or the session's *filtered.h5 when the NWB has none (e.g. camera_flag 0 sessions).
Also plotted: the pairing the NWB uses (stored timestamp of frame f -> pulse index - f) and the counts (video frames from
the file header, DLC rows, pulses per run).
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/figures/<session>_offset_curve.png, tables/offset_curve.csv
Run (haas, ibl-ai-agent .venv): python 005_dlc_offset_curve.py SESSION_ID [SESSION_ID ...]
"""
import glob
import os
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

NWB_DIR = axel_bisi_path("NWB_ks4")
DATA = axel_bisi_path("data")
AN = pathlib.Path("/mnt/lsens-analysis")
RAW = "/mnt/lsens-data"
OUT = axel_bisi_path("combined_results_ks4") / "ssl-dlc-alignment-debug"
GAP = 0.05
WIN = {"side": 24000, "top": 60000}   # frames per window (2 min / 5 min at 200 Hz)
STEP = {"side": 12000, "top": 30000}
TOP_SERIES = ["top_particle_y", "whisker_angle", "whisker_tip_x"]


def zs(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x)


def speed_z(cols):
    out = None
    for x in cols:
        v = np.abs(np.diff(np.asarray(x, float), prepend=x[0]))
        v = zs(v)
        out = v if out is None else out + v
    return np.nan_to_num(zs(out))


def dlc_from_h5(session_id, view):
    exp = "Myriam_Hamon" if session_id.startswith("MH") else "Axel_Bisi"
    h = sorted(glob.glob(str(AN / exp / "data" / session_id[:5] / session_id / "Video" / f"*{view}*filtered.h5")))
    if not h:
        return None
    df = pd.read_hdf(h[0])
    df.columns = ["_".join(c[1:]) for c in df.columns]
    return df


def load(session_id):
    with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
        b = f.get("processing/behavior/BehavioralTimeSeries")
        g = lambda k: (b[k]["data"][:], b[k]["timestamps"][:] if "timestamps" in b[k] else np.array([])) if b is not None and k in b else (None, None)  # noqa: E731
        licks = f["processing/behavior/BehavioralEvents/piezo_lick_times/timestamps"][:]
        t = f["intervals/trials"]
        ws = t["whisker_stim"][:] == 1
        wst = t["whisker_stim_time"][:][ws] if "whisker_stim_time" in t else t["start_time"][:][ws]
        wst = wst[np.isfinite(wst)]
        ephys = "units" in f
        side_sig, side_ts = g("tongue_likelihood")
        top = {k: g(k) for k in TOP_SERIES}
        _, side_nwb_ts = g("jaw_angle")
        _, top_nwb_ts = g("whisker_angle")
    sig = {}
    if side_sig is not None:
        sig["side"] = (np.nan_to_num(zs(side_sig)), side_nwb_ts, "nwb")
    else:
        df = dlc_from_h5(session_id, "side")
        if df is not None and "tongue_likelihood" in df:
            sig["side"] = (np.nan_to_num(zs(df["tongue_likelihood"].to_numpy())), np.array([]), "h5")
    if all(top[k][0] is not None for k in TOP_SERIES):
        sig["top"] = (speed_z([top[k][0] for k in TOP_SERIES]), top_nwb_ts, "nwb")
    else:
        df = dlc_from_h5(session_id, "top")
        cols = [c for c in ("particle_y", "whisker_tip_x", "whisker_base_x", "nose_tip_x") if df is not None and c in df]
        if cols:
            sig["top"] = (speed_z([df[c].to_numpy() for c in cols]), np.array([]), "h5")
    pulses = {}
    sync = glob.glob(str(DATA / session_id[:5] / session_id / "Ephys" / "catgt_*" / "sync_event_times"))
    if ephys and sync:
        pulses = {"top": np.loadtxt(sync[0] + "/cam0_frame_times.txt"), "side": np.loadtxt(sync[0] + "/cam1_frame_times.txt")}
        src = "NIDQ"
    else:
        pulses = {"top": top_nwb_ts, "side": side_nwb_ts}
        src = "NWB stored"
    return sig, pulses, src, licks, wst


def video_frames(session_id, view):
    import cv2
    v = [p for p in glob.glob(f"{RAW}/{session_id[:5]}/Recording/{session_id}/Video/*")
         if p.lower().endswith((".avi", ".mp4")) and view in os.path.basename(p).lower()]
    return [(os.path.basename(p), int(cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT))) for p in sorted(v)]


SIDE_MIN_SCORE, SIDE_MIN_EVENTS = 40, 10   # side-camera window accepted (spurious peaks score < 30; checked on controls)
TOP_SEARCH = 3000                           # top camera: lags searched within +-3000 frames of the side-camera lag


def offset_curve(x, pulses, events, win, step, center=None, half_range=None):
    """Best lag L (DLC frame f <-> pulse f + L) per window of frames, by cross-correlation with the event train.
    center(frame) -> lag: restrict the search to center +- half_range (local search), else all lags."""
    P = len(pulses)
    e = np.zeros(P)
    idx = np.searchsorted(pulses, events)
    ok = (idx > 0) & (idx < P)
    e[idx[ok]] = 1
    rows = []
    for f0 in range(0, max(1, len(x) - win // 2), step):
        xw = x[f0:f0 + win]
        if len(xw) < win // 4:
            break
        # c[L'] = sum_f xw[f] * e[f0 + f + L]; full correlation via convolution with reversed window
        c = fftconvolve(e, xw[::-1], mode="full")  # c[k] = sum_f xw[f] e[k - (len(xw)-1) + f]
        lags = np.arange(len(c)) - (len(xw) - 1) - f0  # L such that pulse index = f0 + f + L
        n_ev = int(e[max(0, f0 - 1):min(P, f0 + win)].sum())
        if center is not None:
            c0 = center(f0 + len(xw) // 2)
            if c0 is None or not np.isfinite(c0):
                continue
            sel = np.abs(lags - c0) <= half_range
            c, lags = c[sel], lags[sel]
            if len(c) == 0:
                continue
        k = int(np.argmax(c))
        mad = np.median(np.abs(c - np.median(c))) or 1
        rows.append(dict(frame=f0 + len(xw) // 2, lag=int(lags[k]), score=(c[k] - np.median(c)) / mad, n_events_window=n_ev))
    return pd.DataFrame(rows)


def consistent(cur, ok, tol=100, max_dev=20000):
    """Keep an accepted window only if another accepted window has a lag within tol frames (drops isolated peaks) and
    its lag is within max_dev frames of the median accepted lag (drops agreeing pairs of far outliers)."""
    if len(cur) == 0:
        return pd.Series([], dtype=bool)
    lags = cur.lag.to_numpy()
    if ok.any():
        ok = ok & (np.abs(cur.lag - np.median(cur.lag[ok])) <= max_dev)
    ok = ok.to_numpy().copy()
    keep = ok.copy()
    for i in np.where(ok)[0]:
        others = np.where(ok)[0]
        others = others[others != i]
        keep[i] = len(others) > 0 and np.any(np.abs(lags[others] - lags[i]) <= tol)
    return pd.Series(keep, index=cur.index)


def nwb_pairing(stored_ts, pulses):
    if len(stored_ts) == 0 or len(pulses) == 0:
        return None
    pi = np.searchsorted(pulses, stored_ts - 0.0005)
    f = np.arange(len(stored_ts))
    return f, pi - f


def main(session_id):
    sig, pulses, src, licks, wst = load(session_id)
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw=dict(height_ratios=[1, 1, 0.8]))
    lines = [f"pulses from {src}"]
    tables = []
    side_center = None
    for ax, view, events, ev_name in ((axs[0], "side", licks, "licks"), (axs[1], "top", np.sort(np.r_[licks, wst]), "licks + whisker stim")):
        if view not in sig or pulses.get(view) is None or len(pulses.get(view)) == 0:
            ax.set_title(f"{view}: no DLC or no pulses", fontsize=9)
            continue
        x, stored, dsrc = sig[view]
        p = pulses[view]
        if p is None or len(p) < 2 or len(x) < WIN[view] // 4:
            ax.set_title(f"{view}: too few pulses ({0 if p is None else len(p)}) or DLC frames ({len(x)})", fontsize=9)
            lines.append(f"{view}: DLC frames {len(x)}; pulses {0 if p is None else len(p)}: not analysed")
            continue
        br = np.where(np.diff(p) > GAP)[0]
        runs = list(zip(np.r_[0, br + 1], np.r_[br, len(p) - 1]))
        if view == "side":
            cur = offset_curve(x, p, events, WIN[view], STEP[view])
            ok = (cur.score >= SIDE_MIN_SCORE) & (cur.n_events_window >= SIDE_MIN_EVENTS)
            ok = consistent(cur, ok)
            gs = cur[ok]
            if len(gs):
                fr, lg = gs.frame.to_numpy(), gs.lag.to_numpy()
                side_center = lambda f: float(lg[np.argmin(np.abs(fr - f))])  # noqa: E731
        else:
            if side_center is None:
                ax.set_title("top: no accepted side-camera offset to search around", fontsize=9)
                continue
            cur = offset_curve(x, p, events, WIN[view], STEP[view], center=side_center, half_range=TOP_SEARCH)
            ok = consistent(cur, (cur.score >= 8) & (cur.n_events_window >= SIDE_MIN_EVENTS))
        cur["view"], cur["marker"], cur["accepted"] = view, ev_name, ok
        tables.append(cur)
        good = cur[ok]
        ax.scatter(good.frame / 200, good.lag, c=np.clip(good.score, 0, 100), cmap="viridis", s=18,
                   label=f"vs {ev_name}: accepted windows")
        ax.scatter(cur[~ok].frame / 200, cur[~ok].lag, s=6, color="lightgrey", label="rejected (few events / weak peak)")
        pr = nwb_pairing(stored, p)
        if pr is not None:
            ax.plot(pr[0] / 200, pr[1], color="tab:red", lw=1, label="pairing stored in the NWB")
        for a, z in runs:
            ax.axhline(a, color="k", lw=0.4, ls=":")
        ax.set_ylabel("pulse index - DLC frame", fontsize=8)
        ax.set_title(f"{view} camera ({dsrc} DLC, {len(x)} frames): offset of DLC frames on the {src} pulse train "
                     f"({len(p)} pulses, {len(runs)} runs; dotted = run starts)", fontsize=8)
        ax.legend(fontsize=6.5, loc="best")
        if len(good):
            lo, hi = good.lag.min(), good.lag.max()
            pad = max(30, 0.1 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        vids = video_frames(session_id, view)
        lines.append(f"{view}: DLC frames {len(x)} ({dsrc}); pulses {len(p)}; runs " +
                     "; ".join(f"{z - a + 1}@{p[a]:.1f}s" for a, z in runs) +
                     f"; video {vids}; NWB stored timestamps {len(stored)}")
        if len(good):
            lines.append(f"   {view} offset (accepted): median {np.median(good.lag):.0f}, first {good.lag.iloc[0]}, "
                         f"last {good.lag.iloc[-1]}, range {good.lag.min()}..{good.lag.max()} frames, {len(good)}/{len(cur)} windows")
    axs[1].set_xlabel("DLC frame / 200 (s of video)", fontsize=8)
    axs[2].axis("off")
    axs[2].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7)
    fig.suptitle(f"{session_id}: DLC frame -> camera pulse offset over the session", fontsize=10)
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "figures" / f"{session_id}_offset_curve.png", dpi=130)
    plt.close(fig)
    if tables:
        t = pd.concat(tables)
        t["session_id"] = session_id
        out = OUT / "tables" / "offset_curve.csv"
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        if len(old):
            old = old[old.session_id != session_id]
        pd.concat([old, t]).to_csv(out, index=False)
    print("\n".join([session_id] + lines))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        try:
            main(s)
        except Exception as e:
            print(s, "ERROR", type(e).__name__, e)
