"""Pupil QC for a list of sessions: states, metrics, trial QC, pages -> combined_results_ks4/<mouse>/<day>/pupil_qc/.

Run on haas (needs OpenCV + ffmpeg: ibl-ai-agent venv):
    cd ~/code/ibl-ai-agent/projects/ssl-pupil-qc
    nice -n 10 ~/code/ibl-ai-agent/.venv/bin/python run.py AB131_20240905_123601 ... [--intensity] [--jobs 5]
--intensity reads the whole raw side video once for the eye-ROI intensity (pilot validation, ~12 min per session);
all video-derived data are cached in pupil_qc/video_cache/ and reused.
"""

import argparse
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import page
import pupil_qc as pq
import video


def lid_validation(r, z):
    """Fraction of blinks with an eye-ROI intensity excursion, and of low-likelihood failed runs with one (blinks the
    rule missed)."""
    st, fs = r["c"]["state"], r["meta"]["fs"]
    pad = int(0.02 * fs)

    def peaks(starts, stops):
        return np.array([np.nanmax(z[max(s - pad, 0):e + pad]) for s, e in zip(starts, stops)])

    bs, be = pq._runs(st == pq.STATE["blink"])
    fs_, fe_ = pq._runs((r["c"]["L"].mean(1) < pq.PARAMS["p_cut"]) & (st == pq.STATE["failed"]))
    pb, pf = peaks(bs, be), peaks(fs_, fe_)
    thr = page.Z_THRESHOLD
    return dict(video_blink_confirmed=float(np.mean(pb > thr)) if len(pb) else np.nan,
                video_failure_lid=float(np.mean(pf > thr)) if len(pf) else np.nan,
                video_n_blinks=int(len(pb)), video_n_low_failed=int(len(pf)))


def intensity_z(r, intensity):
    """Robust z of the eye-ROI intensity (aligned to the NWB index), oriented so that lid closure is positive."""
    off, n, fs = r["video"]["offset"], len(r["df"]), r["meta"]["fs"]
    x = np.full(n, np.nan, np.float32)
    seg = intensity[off:off + n]
    x[:len(seg)] = seg
    base = pd.Series(x).rolling(int(5 * fs), center=True, min_periods=int(fs)).median().to_numpy()
    res = x - base
    z = res / (np.nanmedian(np.abs(res)) * 1.4826 + 1e-6)
    bs, be = pq._runs(r["c"]["state"] == pq.STATE["blink"])
    if len(bs) and np.nanmedian(z[((bs + be) // 2).astype(int)]) < 0:
        z = -z
    return z


def process(session_id, with_intensity=False):
    r = pq.run_session(session_id)
    out, vid, sid = r["out"], r["video"], r["meta"]["session_id"]
    frames = None
    if vid and vid["video"]:
        cache = video.VideoCache(out, vid["video"])
        roi = cache.roi(r["df"], r["c"]["state"] == pq.STATE["valid"])
        r["roi"] = roi
        inten = cache.intensity(roi, compute=with_intensity)
        if inten is not None:
            r["intensity_z"] = intensity_z(r, inten)
            r["metrics"].update(lid_validation(r, r["intensity_z"]))
        frames = cache.frames([i + vid["offset"] for i in page.example_frame_indices(r)])
    page.make_pages(r, frames, out / f"{sid}_pupil_qc")
    r["trials_qc"].to_csv(out / f"{sid}_trials_qc.csv", index=False)
    pd.DataFrame({"t": r["df"].t.astype("float64"), "state": r["c"]["state"], "n_good": r["c"]["n_good"].astype("int8"),
                  "area_px2": r["c"]["area"].astype("float32"), "area_clean_px2": r["clean"].astype("float32"),
                  "hw_ratio_norm": r["c"]["ratio_norm"].astype("float32"),
                  "interpolated": r["interp"]}).to_parquet(out / f"{sid}_frames.parquet", index=False)
    meta = {**r["meta"], "video": vid, "roi": r.get("roi"), "params": pq.PARAMS, "verdict_thresholds": pq.VERDICT,
            "states": pq.STATE}
    (out / f"{sid}_metrics.json").write_text(json.dumps({**r["metrics"], "meta": meta}, indent=1, default=str))
    keys = ("pct_valid", "pct_blink", "pct_squint", "pct_failed", "pct_jump", "long_loss_min", "blink_rate_per_min",
            "pct_trials_valid", "n_trials_valid", "n_trials_blink", "n_trials_squint", "n_trials_failed",
            "video_blink_confirmed",
            "video_failure_lid", "video_n_blinks", "video_n_low_failed")
    return sid, r["metrics"]["verdict"], {k: r["metrics"].get(k) for k in keys}


def safe_process(session_id, with_intensity=False):
    """Batch wrapper: one failing session is reported, not fatal."""
    import traceback
    try:
        return process(session_id, with_intensity)
    except Exception as e:
        return session_id, "error", {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*")
    ap.add_argument("--list", help="text file with one session id per line")
    ap.add_argument("--intensity", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    sessions = list(a.sessions) + ([l.strip() for l in open(a.list) if l.strip()] if a.list else [])
    args = [(s, a.intensity) for s in sessions]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            res = pool.starmap(safe_process, args, chunksize=1)
    else:
        res = [safe_process(*x) for x in args]
    summary = pd.DataFrame([{"session_id": sid, "verdict": v, **m} for sid, v, m in res])
    summary.to_csv(pq.RESULTS / "ssl-pupil-qc" / "tables" / "batch_summary.csv", index=False)
    for sid, v, m in res:
        print(sid, v, {k: (round(float(x), 3) if isinstance(x, (float, np.floating)) else x) for k, x in m.items()})


if __name__ == "__main__":
    main()
