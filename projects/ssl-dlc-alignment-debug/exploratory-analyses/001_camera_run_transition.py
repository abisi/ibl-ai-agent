"""001 -- Camera pulse runs vs video start, per session (see ../question.md).

Camera pulses on the NIDQ (sync_event_times/cam0_frame_times.txt = top camera, cam1_frame_times.txt = side camera) are
split into runs at gaps > 50 ms. The video file name holds its recording start time (wall clock); SpikeGLX's
nidq.meta fileCreateTime gives the wall clock of NIDQ time 0, so the video start can be placed on the NIDQ time axis
(1 s resolution). Left: pulses around the transition to the last run, the video start, trial starts and the first DLC
timestamp stored in the NWB; right: the end of the last run, the pulse paired with the last video frame and the last stored
DLC timestamp; below: a
per-run table (pulses, span) against the video frame count.

Outputs: combined_results_ks4/ssl-dlc-alignment-debug/figures/<session>_camera_runs.png|pdf
Run (haas): python 001_camera_run_transition.py SESSION_ID [SESSION_ID ...]
"""
import datetime as dt
import glob
import pathlib
import re
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

RAW = pathlib.Path("/mnt/lsens-data")
OUT = axel_bisi_path("combined_results_ks4") / "ssl-dlc-alignment-debug" / "figures"
GAP = 0.05


def runs(t):
    br = np.where(np.diff(t) > GAP)[0]
    return list(zip(np.r_[0, br + 1], np.r_[br, len(t) - 1]))


def video_info(session_id, view):
    import cv2
    v = sorted(glob.glob(str(RAW / session_id[:5] / "Recording" / session_id / "Video" / f"*{view}*")))
    if not v:
        return None, None
    m = re.search(r"(\d\d)-(\d\d)-(\d\d) (\d\d)-(\d\d)-(\d\d)", pathlib.Path(v[0]).name)
    start = dt.datetime(2000 + int(m[1]), int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])) if m else None
    n = int(cv2.VideoCapture(v[0]).get(cv2.CAP_PROP_FRAME_COUNT))
    return start, n


def main(session_id):
    m = session_id[:5]
    sync = glob.glob(str(axel_bisi_path("data", m, session_id, "Ephys", "catgt_*", "sync_event_times")))[0]
    meta = glob.glob(str(RAW / m / "Recording" / session_id / "Ephys" / "*" / "*nidq.meta"))[0]
    t0 = dt.datetime.fromisoformat(re.search(r"fileCreateTime=(\S+)", open(meta).read())[1])
    trials = np.loadtxt(pathlib.Path(sync) / "trial_start_times.txt")
    cams = {"top (cam0_frame_times)": ("cam0_frame_times", "top", "whisker_angle"),
            "side (cam1_frame_times)": ("cam1_frame_times", "side", "jaw_angle")}
    nwb_ts = {}
    with h5py.File(axel_bisi_path("NWB_ks4") / f"{session_id}.nwb", "r") as f:
        b = f["processing/behavior/BehavioralTimeSeries"]
        for _, (_, view, series) in cams.items():
            if series in b and "timestamps" in b[series]:
                nwb_ts[view] = (b[series]["timestamps"][[0, -1]], b[series]["timestamps"].shape[0], b[series]["data"].shape[0])
    fig, axs = plt.subplots(len(cams) + 1, 2, figsize=(14, 7.5), gridspec_kw=dict(height_ratios=[1, 1, 1.2]))
    lines = []
    for r, (label, (fname, view, series)) in enumerate(cams.items()):
        t = np.loadtxt(pathlib.Path(sync) / f"{fname}.txt")
        # as the converter (format_ephys_timestamps): drop an isolated first / last pulse (> 50 ms from its neighbour)
        if t[1] - t[0] > GAP:
            t = t[1:]
        if t[-1] - t[-2] > GAP:
            t = t[:-1]
        rs = runs(t)
        vstart, nframes = video_info(session_id, view)
        vt = (vstart - t0).total_seconds() if vstart else np.nan
        a, z = rs[-1]
        last_frame_t = t[a + nframes - 1] if nframes and a + nframes - 1 < len(t) else np.nan
        # start: test run, stray pulse, video start, first pulse of the last run, first stored DLC timestamp
        ax = axs[r, 0]
        ax.vlines(t, 0, 1, color="k", lw=0.3)
        ax.axvline(vt, color="tab:red", lw=1.5, label=f"video start (file name {vstart:%H:%M:%S})")
        ax.axvline(t[a], color="tab:green", lw=1.5, ls="--", label=f"first pulse of last run ({t[a]:.2f} s)")
        if view in nwb_ts:
            ax.plot(nwb_ts[view][0][0], 0.5, "o", color="tab:orange", ms=7, mfc="none",
                    label=f"first NWB {series} timestamp ({nwb_ts[view][0][0]:.2f} s)")
        ax.set_xlim(t[a] - 120, t[a] + 20)
        # end: last pulse of the run, pulse paired with the last video frame, last stored DLC timestamp
        ax2 = axs[r, 1]
        ax2.vlines(t, 0, 1, color="k", lw=0.3)
        ax2.axvline(t[z], color="tab:green", lw=1.5, ls="--", label=f"last pulse of last run ({t[z]:.2f} s)")
        ax2.axvline(last_frame_t, color="tab:red", lw=1.0, label=f"pulse of last video frame ({last_frame_t:.2f} s)")
        if view in nwb_ts:
            ax2.plot(nwb_ts[view][0][1], 0.5, "o", color="tab:orange", ms=7, mfc="none",
                     label=f"last NWB {series} timestamp ({nwb_ts[view][0][1]:.2f} s)")
        ax2.set_xlim(t[z] - 0.2, t[z] + 0.1)
        for x in (ax, ax2):
            x.set_yticks([])
            x.legend(fontsize=6.5, loc="upper left", frameon=True, facecolor="white", framealpha=1)
        ax.set_ylabel(label, fontsize=8)
        for aa, bb in rs:
            lines.append(f"{view}: run pulses {aa}-{bb}  n={bb - aa + 1}  {t[aa]:.2f}-{t[bb]:.2f} s")
        lines.append(f"{view}: video frames {nframes}; last run {z - a + 1} pulses; difference {nframes - (z - a + 1)} frames")
        if view in nwb_ts:
            lines.append(f"{view}: NWB {series}: {nwb_ts[view][2]} DLC frames, {nwb_ts[view][1]} timestamps "
                         f"{nwb_ts[view][0][0]:.2f}-{nwb_ts[view][0][1]:.2f} s")
    axs[1, 0].vlines(trials, 0, 1, color="tab:blue", lw=0.8, alpha=0.6)
    axs[1, 0].set_xlabel(f"NIDQ time (s); 0 = SpikeGLX file start {t0:%H:%M:%S}; blue = trial starts", fontsize=8)
    axs[1, 1].set_xlabel("NIDQ time (s), end of the last run (5 ms pulses)", fontsize=8)
    axs[0, 0].set_title("start of the saved video", fontsize=9)
    axs[0, 1].set_title("end of the saved video", fontsize=9)
    for x in axs[-1]:
        x.axis("off")
    axs[-1, 0].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7.5)
    fig.suptitle(f"{session_id}: camera pulse runs (black ticks) vs video and NWB DLC timestamps", fontsize=10)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{session_id}_camera_runs.{ext}", dpi=150)
    print("\n".join(lines))


if __name__ == "__main__":
    for s in sys.argv[1:]:
        print("==", s)
        main(s)
