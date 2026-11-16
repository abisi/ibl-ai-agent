"""Pupil-tracking quality control for SSL sessions (NWB_ks4 side-camera DLC, 4 pupil points, no eyelid points).

Coarse pupil measure: area of the polygon through the 4 pupil points (px^2, as NWB pupil_area); no shape criteria
(the camera is not always frontal, so the pupil can look oval).

Frame states (user, 2026-11-03; dilation is measured on valid frames only):
    valid  -- all 4 points above the likelihood cutoff, no sudden jump of the area, eye fully open
    blink  -- >= 3 points lost (<= 1 tracked) for blink_min_s-blink_max_s, all 4 tracked within sync_s before the loss
              (the lid closes over a tracked pupil); partially tracked frames at the blink edges belong to the blink
    squint -- (a) eye closed longer than a blink (user, 2026-11-04): the same synchronous loss of >= 3 points as a
              blink, lasting blink_max_s-closure_max_s (blinks and squints can be fast, so the blink / squint boundary
              is 0.2 s, was 0.5 s); (b) eye partly closed: the pupil's height / width, normalised to the session's
              open-eye value, below squint_ratio for >= squint_min_s. Dilation scales the pupil isotropically (ratio ~1); a closing lid clips
              it vertically (ratio < 1). The session normalisation removes the camera angle.
    failed -- everything else (1-3 points lost, jumps, asynchronous losses, losses longer than closure_max_s)
Sudden jump: |area - rolling median over jump_window_s| > jump_rel x that median (25 %; frame-to-frame noise is
~0.4 %, so a robust-SD rule flagged several % of normal frames).
Single-trial label in trial_window_s around start_time: valid if >= trial_min_valid of the frames are valid (or
interpolated), else the dominant non-valid state (blink, squint or failed). The eye-ROI intensity from the raw video validates blinks on pilot sessions.
"""

import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal

NWB_DIR = Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
RESULTS = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
RAW = Path("/mnt/lsens-data")
DATA_DIRS = {"AB": Path("/mnt/lsens-analysis/Axel_Bisi/data"), "MH": Path("/mnt/lsens-analysis/Myriam_Hamon/data")}
SHEET = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
POINTS = ["top", "left", "right", "bottom"]

PARAMS = dict(
    p_cut=0.1,              # DLC likelihood cutoff (user 2026-11-12, = dlc_helpers; was 0.5, the converter pcutoff:
                            # the new DLC model is under-confident on small pupils, 007)
    sync_s=0.05,            # blink: all 4 points tracked within 50 ms before the loss
    blink_min_s=0.02,
    blink_max_s=0.2,        # longer synchronous losses are squints (eye closed), up to closure_max_s (user, 2026-11-04)
    closure_max_s=5.0,      # synchronous losses longer than this are failed (tracking lost, not a closure)
    edge_pre_s=0.05,        # partial-loss frames this close before / after a blink belong to it
    edge_post_s=0.1,
    jump_window_s=0.25,
    jump_rel=0.25,          # sudden jump: > 25 % away from the 250 ms rolling median
    interp_max_s=0.1,       # failed gaps up to this long between valid frames are interpolated (never blinks)
    snr_lowpass_hz=2.0,
    trial_window_s=(-1.0, 2.0),   # around start_time (user, 2026-11-03)
    baseline_s=(-1.0, 0.0),
    trial_min_valid=0.8,
    squint_ratio=0.7,       # normalised height / width below this = eye partly closed
    squint_min_s=0.1,
    squint_smooth_s=0.05,   # rolling median of the normalised ratio before thresholding
    open_ref_quantile=0.5,  # open-eye reference ratio: median over full-tracking frames with area above this quantile
)
VERDICT = dict(pct_valid=(90.0, 75.0), pct_trials_valid=(90.0, 70.0))  # (pass, warn)
# Sessions where one pupil point is systematically off the eye (with high likelihood): that point is replaced by the
# mirror image of the opposite point across the axis of the other two (user, 2026-11-04; MH029 0502: right point off
# in 64% of frames; the 3-point area matches the 4-point area at r = 0.988, -4%, on a clean session). Same list in
# ephys_utilities dlc_helpers.PUPIL_THREE_POINT.
THREE_POINT = {"MH029_20250502_110353": "right"}
OPPOSITE = {"right": ("left", "top", "bottom"), "left": ("right", "top", "bottom"),
            "top": ("bottom", "left", "right"), "bottom": ("top", "left", "right")}
STATE = {"valid": 0, "blink": 1, "squint": 2, "failed": 3}
STATE_NAMES = {v: k for k, v in STATE.items()}


# ----------------------------------------------------------------------------------------------- loading

def _series(f, name):
    g = f.get(f"processing/behavior/BehavioralTimeSeries/{name}")
    return (g["data"][:], g["timestamps"][:]) if g is not None else (None, None)


def _str(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load_session(session_id):
    """Pupil keypoints, motion, trials and metadata of one NWB_ks4 session."""
    with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
        desc = _str(f["session_description"][()])
        if _series(f, "pupil_top_x")[0] is None:
            raise ValueError(f"{session_id}: no pupil keypoints in the NWB")
        ts = _series(f, "pupil_top_x")[1]
        pts = {f"{p}_{c}": _series(f, f"pupil_{p}_{c}")[0] for p in POINTS for c in ("x", "y", "likelihood")}
        motion = {}
        for name in ("whisker_velocity", "jaw_velocity"):
            d, t = _series(f, name)
            if d is not None:
                m = min(len(d), len(t))  # data/timestamp lengths differ in some NWBs; motion is only auxiliary
                if m < 2 or abs(len(d) - len(t)) > 200:
                    continue
                motion[name] = np.interp(ts, t[:m], np.abs(d[:m]), left=np.nan, right=np.nan)
        tr = f["intervals/trials"]
        cols = ["start_time", "trial_type", "lick_flag", "perf", "context", "early_lick", "lick_time",
                "response_window_start_time", "reward_available"]
        trials = pd.DataFrame({c: [_str(x) if isinstance(x, bytes) else x for x in tr[c][:]] for c in cols if c in tr})
    n = min(len(ts), *(len(v) for v in pts.values()))
    df = pd.DataFrame({k: v[:n] for k, v in pts.items()})
    df["t"] = ts[:n]
    if session_id in THREE_POINT:
        mirror_point(df, THREE_POINT[session_id])
    for k, v in motion.items():
        df[k] = v[:n]
    sheet = pd.read_excel(SHEET).drop_duplicates("mouse_id").set_index("mouse_id")
    mouse = session_id.split("_")[0]
    meta = dict(session_id=session_id, mouse_id=mouse, description=desc, folder=desc.replace("+", ""),
                group=sheet.reward_group.get(mouse, "?"), fs=float(np.round(1 / np.median(np.diff(ts[:2000])))))
    return df, _trial_table(trials), meta


def mirror_point(df, point):
    """In place: replace `point` by the mirror image of its opposite point across the line through the other two
    (THREE_POINT); its likelihood becomes the minimum of the three points used."""
    opp, a, b = OPPOSITE[point]
    ax, ay, bx, by = df[f"{a}_x"], df[f"{a}_y"], df[f"{b}_x"], df[f"{b}_y"]
    ox, oy = df[f"{opp}_x"], df[f"{opp}_y"]
    dx, dy = bx - ax, by - ay
    nrm = np.hypot(dx, dy).replace(0, np.nan)
    ux, uy = dx / nrm, dy / nrm
    proj = (ox - ax) * ux + (oy - ay) * uy
    px, py = ax + proj * ux, ay + proj * uy
    df[f"{point}_x"], df[f"{point}_y"] = 2 * px - ox, 2 * py - oy
    df[f"{point}_likelihood"] = df[[f"{q}_likelihood" for q in (opp, a, b)]].min(axis=1)


def _trial_table(tr):
    """Active trials (no passive, perf 6, early lick) with lick and reward times and the outcome."""
    tr = tr.copy()
    # NWB lick_time is late by the artifact window (memory ssl-nwb-lick-time-offset)
    tr["lick"] = np.where(tr.lick_flag == 1, tr.start_time + (tr.lick_time - tr.response_window_start_time), np.nan)
    tr["reward"] = np.where((tr.lick_flag == 1) & (tr.reward_available == 1), tr.lick, np.nan)
    passive = tr.context.astype(str).eq("passive") if "context" in tr else False
    tr["active"] = ~passive & (tr.perf != 6) & (tr.get("early_lick", 0) == 0)
    names = {("whisker_trial", 1): "W hit", ("whisker_trial", 0): "W miss", ("auditory_trial", 1): "A hit",
             ("auditory_trial", 0): "A miss", ("no_stim_trial", 1): "FA", ("no_stim_trial", 0): "CR"}
    tr["outcome"] = [names.get((t, int(l)), "other") for t, l in zip(tr.trial_type, tr.lick_flag)]
    return tr[tr.active].reset_index(drop=True)


def find_video(meta, df):
    """Side video and the NWB-index -> video-frame offset, found by matching the NWB keypoints to each DLC filtered
    h5 of the session (handles top videos named '_side' and end-aligned sessions). The NWB keypoints are smoothed after
    DLC filtering, so the match uses the median absolute difference (side vs top differ by hundreds of px)."""
    mouse, sid = meta["mouse_id"], meta["session_id"]
    root = DATA_DIRS.get(mouse[:2])
    best = None
    ref = df.top_x.to_numpy()[:3000]
    for h5 in sorted(glob.glob(str(root / mouse / sid / "Video" / "*filtered.h5"))):
        if "_backup_" in h5:
            continue
        try:
            d = pd.read_hdf(h5)
        except Exception:
            continue
        cols = [c for c in d.columns if c[1] == "pupil_top" and c[2] == "x"]
        if not cols:
            continue
        x = d[cols[0]].to_numpy()
        for off in sorted({0, max(len(x) - len(df), 0)}):
            seg = x[off:off + len(ref)]
            if len(seg) == len(ref):
                err = np.nanmedian(np.abs(seg - ref))
                if best is None or err < best[0]:
                    best = (err, h5, off, len(x))
    if best is None or best[0] > 2.0:
        return None
    _, h5, off, n_frames = best
    stem = os.path.basename(h5).split("DLC_")[0]
    cands = []
    for r in ("Recording", "Training"):
        cands += glob.glob(str(RAW / mouse / r / sid / "Video" / f"{stem}.avi"))
    cands += glob.glob(str(root / mouse / sid / "Video" / f"{stem}*.mp4"))
    return dict(video=cands[0] if cands else None, h5=h5, offset=int(off), n_frames=int(n_frames))


# ----------------------------------------------------------------------------------------------- states

def _runs(mask):
    """(start, stop) index pairs of True runs (stop exclusive)."""
    m = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(np.diff(m.astype(int)))
    return edges[::2], edges[1::2]


def polygon_area(df):
    """Shoelace area of top -> right -> bottom -> left (same as NWB pupil_area)."""
    x = np.column_stack([df[f"{p}_x"] for p in ("top", "right", "bottom", "left")])
    y = np.column_stack([df[f"{p}_y"] for p in ("top", "right", "bottom", "left")])
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1, 1) - y * np.roll(x, -1, 1), 1))


def classify(df, fs, p=PARAMS):
    L = np.column_stack([df[f"{q}_likelihood"] for q in POINTS])
    n_good = (L >= p["p_cut"]).sum(1)
    area = polygon_area(df)
    full = (n_good == 4) & (area > 0)
    med = pd.Series(np.where(full, area, np.nan)).rolling(max(int(p["jump_window_s"] * fs), 3), center=True,
                                                         min_periods=3).median().to_numpy()
    jump = full & (np.abs(area - med) > p["jump_rel"] * med)
    state = np.full(len(df), STATE["failed"], np.int8)
    state[full & ~jump] = STATE["valid"]

    # squint: height / width normalised to the open-eye reference of the session
    dv = np.hypot(df.top_x - df.bottom_x, df.top_y - df.bottom_y).to_numpy()
    dh = np.hypot(df.left_x - df.right_x, df.left_y - df.right_y).to_numpy()
    ratio = np.divide(dv, dh, out=np.full(len(dv), np.nan), where=dh > 0)
    ok = full & ~jump
    open_ = ok & (area >= np.quantile(area[ok], p["open_ref_quantile"])) if ok.any() else ok
    ratio_ref = float(np.nanmedian(ratio[open_])) if open_.any() else np.nan
    ratio_norm = pd.Series(np.where(full, ratio / ratio_ref, np.nan)).rolling(
        max(int(p["squint_smooth_s"] * fs), 1), center=True, min_periods=1).median().to_numpy()
    starts, stops = _runs(ok & (ratio_norm < p["squint_ratio"]))
    for s_, e_ in zip(starts, stops):
        if (e_ - s_) / fs >= p["squint_min_s"]:
            state[s_:e_] = STATE["squint"]

    k = max(int(round(p["sync_s"] * fs)), 1)
    pre, post = int(round(p["edge_pre_s"] * fs)), int(round(p["edge_post_s"] * fs))
    starts, stops = _runs(n_good <= 1)
    for s, e in zip(starts, stops):
        dur = (e - s) / fs
        sync = s > 0 and np.any(n_good[max(s - k, 0):s] == 4)
        if sync and p["blink_min_s"] <= dur <= p["closure_max_s"]:
            a = s
            while a > 0 and s - a < pre and n_good[a - 1] < 4:
                a -= 1
            b = e
            while b < len(state) and b - e < post and n_good[b] < 4:
                b += 1
            state[a:b] = STATE["blink"] if dur <= p["blink_max_s"] else STATE["squint"]
    return dict(L=L, n_good=n_good, area=area, state=state, jump=jump, ratio_norm=ratio_norm, ratio_ref=ratio_ref)


def clean_area(area, state, fs, p=PARAMS):
    """Valid frames kept; failed gaps <= interp_max_s between valid frames interpolated; blinks stay NaN."""
    clean = np.where(state == STATE["valid"], area, np.nan)
    interp = np.zeros(len(area), bool)
    idx = np.arange(len(area))
    starts, stops = _runs(state == STATE["failed"])
    for s, e in zip(starts, stops):
        if (e - s) / fs <= p["interp_max_s"] and s > 0 and e < len(area) and \
                state[s - 1] == STATE["valid"] and state[e] == STATE["valid"]:
            clean[s:e] = np.interp(idx[s:e], [s - 1, e], [area[s - 1], area[e]])
            interp[s:e] = True
    return clean, interp


def failed_events(c, fs):
    """Failed runs with their kind: long loss (> closure_max_s), short loss, jump, partial loss (1-3 points)."""
    starts, stops = _runs(c["state"] == STATE["failed"])
    kinds = []
    for s, e in zip(starts, stops):
        ng = c["n_good"][s:e]
        if (ng <= 1).mean() > 0.5:
            kinds.append("long loss" if (e - s) / fs > PARAMS["closure_max_s"] else "short loss")
        elif c["jump"][s:e].any() and (ng == 4).all():
            kinds.append("jump")
        else:
            kinds.append("partial loss")
    return starts, stops, np.array(kinds)


# ----------------------------------------------------------------------------------------------- metrics

def snr(clean, fs, p=PARAMS):
    b, a = signal.butter(2, p["snr_lowpass_hz"] / (fs / 2))
    lo_var, hi_var = 0.0, 0.0
    starts, stops = _runs(np.isfinite(clean))
    for s, e in zip(starts, stops):
        if e - s >= int(2 * fs):
            seg = clean[s:e]
            low = signal.filtfilt(b, a, seg)
            lo_var += np.var(low - low.mean()) * (e - s)
            hi_var += np.var(seg - low) * (e - s)
    return lo_var / hi_var if hi_var > 0 else np.nan


def trial_qc(trials, ts, c, clean, interp, fs, p=PARAMS):
    st = c["state"]
    meanL = c["L"].mean(1)
    rows = []
    w0, w1 = p["trial_window_s"]
    for i, r in trials.iterrows():
        a, b = np.searchsorted(ts, [r.start_time + w0, r.start_time + w1])
        if b - a < 10:
            continue
        seg = st[a:b]
        frac = {k: float(((seg == v) & ~interp[a:b]).mean()) for k, v in STATE.items() if k != "valid"}
        valid_frac = 1 - sum(frac.values())
        label = "valid" if valid_frac >= p["trial_min_valid"] else max(frac, key=frac.get)
        ba, bb = np.searchsorted(ts, [r.start_time + p["baseline_s"][0], r.start_time + p["baseline_s"][1]])
        rows.append(dict(trial=i, start_time=r.start_time, outcome=r.outcome, label=label,
                         valid_frac=valid_frac, blink_frac=frac["blink"], squint_frac=frac["squint"],
                         failed_frac=frac["failed"],
                         baseline_area=np.nanmean(clean[ba:bb]) if np.isfinite(clean[ba:bb]).any() else np.nan,
                         mean_likelihood=float(meanL[a:b].mean())))
    return pd.DataFrame(rows)


def session_metrics(c, clean, interp, ts, fs, tq):
    st = c["state"]
    n = len(st)
    dur_min = (ts[-1] - ts[0]) / 60
    n_dropped = int(np.sum(np.clip(np.round(np.diff(ts) * fs) - 1, 0, None)))
    bs, be = _runs(st == STATE["blink"])
    nan_s, nan_e = _runs(~np.isfinite(clean))
    fs_, fe_, kinds = failed_events(c, fs)
    labels = tq.label.value_counts() if len(tq) else pd.Series(dtype=int)
    m = dict(
        n_frames=n, duration_min=dur_min, fs=fs,
        pct_valid=100 * (st == STATE["valid"]).mean(), pct_blink=100 * (st == STATE["blink"]).mean(),
        pct_squint=100 * (st == STATE["squint"]).mean(), squint_min=float((st == STATE["squint"]).sum() / fs / 60),
        squint_ratio_ref=c["ratio_ref"],
        pct_failed=100 * (st == STATE["failed"]).mean(), pct_jump=100 * c["jump"].mean(),
        long_loss_min=float(np.sum((fe_ - fs_)[kinds == "long loss"]) / fs / 60),
        blink_rate_per_min=len(bs) / dur_min, blink_median_ms=1000 * np.median((be - bs) / fs) if len(bs) else np.nan,
        n_blinks=len(bs), snr=snr(clean, fs),
        pct_dropped=100 * n_dropped / (n + n_dropped), pct_interpolated=100 * interp.mean(),
        longest_gap_s=float(((nan_e - nan_s).max() / fs) if len(nan_s) else 0.0),
        n_trials=int(len(tq)), median_area_px2=float(np.nanmedian(clean)),
        **{f"n_trials_{k}": int(labels.get(k, 0)) for k in STATE},
    )
    m["pct_trials_valid"] = 100 * m["n_trials_valid"] / max(m["n_trials"], 1)
    m["verdict"] = verdict(m)
    return m


def verdict(m):
    level = 0  # 0 pass, 1 warn, 2 fail
    for key, (ok, warn) in VERDICT.items():
        v = m.get(key, np.nan)
        if not np.isfinite(v) or v < warn:
            level = max(level, 2)
        elif v < ok:
            level = max(level, 1)
    return ["pass", "warn", "fail"][level]


def event_onsets(c, fs):
    """Blink onsets, and onsets of failed runs with low likelihood (losses not classified as blinks)."""
    st = c["state"]
    bs, _ = _runs(st == STATE["blink"])
    fs_, _ = _runs((c["L"].mean(1) < PARAMS["p_cut"]) & (st == STATE["failed"]))
    return bs, fs_


def run_session(session_id, out_root=RESULTS):
    df, trials, meta = load_session(session_id)
    fs, ts = meta["fs"], df.t.to_numpy()
    c = classify(df, fs)
    clean, interp = clean_area(c["area"], c["state"], fs)
    tq = trial_qc(trials, ts, c, clean, interp, fs)
    m = session_metrics(c, clean, interp, ts, fs, tq)
    vid = find_video(meta, df)
    out = Path(out_root) / meta["mouse_id"] / meta["folder"] / "pupil_qc"
    out.mkdir(parents=True, exist_ok=True)
    return dict(df=df, trials=trials, meta=meta, c=c, clean=clean, interp=interp, trials_qc=tq, metrics=m,
                video=vid, out=out)
