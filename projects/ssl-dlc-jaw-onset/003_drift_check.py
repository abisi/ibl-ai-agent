"""Is the drift between old (NWB_reader) and new (dlc_helpers) jaw onsets a timing artefact or behaviour? (user
2026-11-10, AB116 and the other 002 mice)

Two independent checks per session, on the trials of 002 (compare_dlc_helpers/trials.csv):
1. Predicted timing error of the old code. behaviour_analysis find_dlc_jaw_onset_times (NWB_reader version) reads the trial window from frame
   int((start_time - video_start + time_start) x freq), with video_start = first whisker_angle (TOP camera) timestamp
   and freq = NWB_reader.get_video_sampling_rate. The true time of that SIDE frame is its own camera timestamp; the
   error = true time - intended time (s). An onset found at window bin b is reported at b / freq + time_start but
   happened at ts_side[i0 + b] - start_time, so the old onset is late by -(error): predicted new - old = error
   (up to the 5 ms frame quantisation). If observed new - old follows the predicted error trial by trial, the drift
   is timing, not behaviour.
2. Lick - jaw onset gap vs time in session (piezo lick on the NIDQ clock, independent of the video): a behavioural
   drift moves the jaw onset and the lick together (gap flat); a timing artefact changes the gap with the old onsets
   only. Slope of (lick_time - start_time - onset) on start_time (ms per 1000 s), lick trials, old vs new.
Outputs: combined_results_ks4/ssl-dlc-jaw-onset/compare_dlc_helpers/drift_check.csv, drift_check.png
Run (haas): nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 003_drift_check.py
"""

import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl-paper-figures"))
from common import import_external_repos  # noqa: E402

import_external_repos()
AB = Path("/mnt/lsens-analysis/Axel_Bisi")
sys.path.insert(0, str(AB / "behaviour_analysis"))
import NWB_reader_functions as NWB_reader  # noqa: E402

NWB = AB / "NWB_ks4"
DIR = AB / "combined_results_ks4" / "ssl-dlc-jaw-onset" / "compare_dlc_helpers"
TIME_START = -1.0
TS = "processing/behavior/BehavioralTimeSeries"


def main():
    t = pd.read_csv(DIR / "trials.csv")
    rows, per_trial = [], []
    for sid, d in t.groupby("session_id"):
        nwb = NWB / f"{sid}.nwb"
        with h5py.File(nwb, "r") as f:
            g = f[TS]
            ts_side = g["jaw_distance"]["timestamps"][:]
            video_start = g["whisker_angle"]["timestamps"][0] if "whisker_angle" in g else g["pupil_area"]["timestamps"][0]
            tr = f["intervals/trials"]
            lick = pd.Series(tr["lick_time"][:], index=tr["start_time"][:])
        freq = NWB_reader.get_video_sampling_rate(str(nwb))
        d = d.copy()
        i0 = ((d.start_time - video_start + TIME_START) * freq).astype(int).to_numpy()
        ok = (i0 >= 0) & (i0 < len(ts_side))
        err = np.full(len(d), np.nan)
        err[ok] = ts_side[i0[ok]] - (d.start_time.to_numpy()[ok] + TIME_START)
        d["predicted_ms"] = 1000 * err
        d["lick_rel"] = d.start_time.map(lick).to_numpy() - d.start_time.to_numpy()
        both = d.diff_ms.notna() & d.predicted_ms.notna()
        resid = (d.diff_ms - d.predicted_ms)[both]
        row = {"session_id": sid, "mouse_id": d.mouse_id.iloc[0], "n": int(both.sum()), "freq_used": freq,
               "true_rate": (len(ts_side) - 1) / (ts_side[-1] - ts_side[0]),
               "video_start_minus_side_start_ms": 1000 * (video_start - ts_side[0]),
               "r_observed_predicted": float(np.corrcoef(d.diff_ms[both], d.predicted_ms[both])[0, 1])
               if both.sum() > 10 and d.predicted_ms[both].std() > 0 else np.nan,
               "median_abs_resid_ms": float(resid.abs().median()) if len(resid) else np.nan,
               "pct_resid_within_10ms": float(100 * (resid.abs() <= 10).mean()) if len(resid) else np.nan}
        lk = d[(d.lick_flag == 1) & np.isfinite(d.lick_rel)]
        for v in ("old", "new"):
            gap = 1000 * (lk.lick_rel - lk[f"jaw_dlc_onset_{v}"])
            m = np.isfinite(gap)
            row[f"gap_slope_ms_per_1000s_{v}"] = float(np.polyfit(lk.start_time[m], gap[m], 1)[0] * 1000) \
                if m.sum() > 10 else np.nan
            row[f"gap_median_ms_{v}"] = float(np.nanmedian(gap)) if m.any() else np.nan
        rows.append(row)
        per_trial.append(d)
    s = pd.DataFrame(rows)
    s.to_csv(DIR / "drift_check.csv", index=False)
    pt = pd.concat(per_trial)
    print(s.round(3).to_string())
    sessions = [x for x in ("AB116_20240724_102941", "AB116_20240725_094541", "AB116_20240726_094206",
                            "AB164_20250423_123413") if x in set(pt.session_id)]
    fig, axs = plt.subplots(2, len(sessions), figsize=(3.2 * len(sessions), 5.6), squeeze=False)
    for c, sid in enumerate(sessions):
        d = pt[pt.session_id == sid]
        ax = axs[0, c]
        ax.scatter(d.start_time / 60, d.diff_ms, s=3, color="0.6", label="observed new - old")
        ax.plot(d.start_time / 60, d.predicted_ms, color="r", lw=1, label="predicted timing error of old code")
        ax.set_title(sid, fontsize=8); ax.set_xlabel("Trial start (min)"); ax.set_ylim(-400, 300)
        if c == 0:
            ax.set_ylabel("ms"); ax.legend(fontsize=6, frameon=False)
        ax = axs[1, c]
        lk = d[(d.lick_flag == 1) & np.isfinite(d.lick_rel)]
        for v, col in (("old", "#d6604d"), ("new", "#2ca02c")):
            gap = 1000 * (lk.lick_rel - lk[f"jaw_dlc_onset_{v}"])
            ax.scatter(lk.start_time / 60, gap, s=3, color=col, alpha=0.5, label=f"{v} onset")
            m = np.isfinite(gap)
            if m.sum() > 10:
                p = np.polyfit(lk.start_time[m] / 60, gap[m], 1)
                xx = np.array([lk.start_time.min(), lk.start_time.max()]) / 60
                ax.plot(xx, np.polyval(p, xx), color=col, lw=1.2)
        ax.set_xlabel("Trial start (min)"); ax.set_ylim(-100, 600)
        if c == 0:
            ax.set_ylabel("Piezo lick - jaw onset (ms)"); ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(DIR / "drift_check.png", dpi=160)


if __name__ == "__main__":
    main()
