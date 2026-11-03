"""002 -- Scan every NWB_ks4 session with DLC: camera pulse runs vs video frame count (see ../question.md, 001).

Per camera, pulses = the camera timestamps the converter stored in the NWB (before any DLC pairing):
  top camera  pulses -> timestamps of the side-view DLC series (jaw_angle): DLC_to_nwb.py:65 gives side series 'cam1' = top
  side camera pulses -> timestamps of the top-view DLC series (whisker_angle): DLC_to_nwb.py:98 gives top series 'cam2' = side
(the swap in DLC_to_nwb is used here to recover each camera's own pulses). Pulses are split into runs at gaps > 50 ms and
compared with the frame count of the camera's video (avi/mp4 header, /mnt/lsens-data/<mouse>/Recording/<session>/Video).
Verdict per camera:
  all_pulses   frames == all pulses (+-2)           -> current NWB pairing correct
  last_run     frames == last run (+-2), > 1 run    -> video = last run (MH067 scenario)
  other_run    frames == another run / consecutive runs
  unmatched    nothing matches
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/tables/camera_run_scan.csv
Run (haas): python 002_camera_run_scan.py
"""
import glob
import os
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

NWB_DIR = axel_bisi_path("NWB_ks4")
RAW = "/mnt/lsens-data"
OUT = axel_bisi_path("combined_results_ks4") / "ssl-dlc-alignment-debug" / "tables"
GAP, TOL = 0.05, 2


def frames(session_id, view):
    import cv2
    vids = [v for v in glob.glob(f"{RAW}/{session_id[:5]}/Recording/{session_id}/Video/*")
            if v.lower().endswith((".avi", ".mp4")) and view in os.path.basename(v).lower() and "short" not in v]
    if len(vids) != 1:
        return len(vids), None
    return 1, int(cv2.VideoCapture(vids[0]).get(cv2.CAP_PROP_FRAME_COUNT))


def verdict(ts, n_frames):
    if len(ts) == 0 or n_frames is None or n_frames == 0:
        return "no_data", ""
    br = np.where(np.diff(ts) > GAP)[0]
    st, en = np.r_[0, br + 1], np.r_[br, len(ts) - 1]
    sizes = en - st + 1
    desc = ";".join(f"{s}@{ts[a]:.1f}s" for s, a in zip(sizes, st))
    if abs(len(ts) - n_frames) <= TOL:
        return "all_pulses", desc
    if len(sizes) > 1 and abs(sizes[-1] - n_frames) <= TOL:
        return "last_run", desc
    for i in range(len(sizes)):
        for j in range(i, len(sizes)):
            if abs(sizes[i:j + 1].sum() - n_frames) <= TOL:
                return f"runs_{i}-{j}", desc
    return "unmatched", desc


def one(fn):
    s = fn[:-4]
    row = dict(session_id=s)
    try:
        with h5py.File(NWB_DIR / fn, "r") as f:
            b = f.get("processing/behavior/BehavioralTimeSeries")
            if b is None or ("jaw_angle" not in b and "whisker_angle" not in b):
                return None
            for cam, series in (("top", "jaw_angle"), ("side", "whisker_angle")):
                g = b.get(series)
                ts = g["timestamps"][:] if g is not None and "timestamps" in g else np.array([])
                row[f"{cam}_n_dlc"] = g["data"].shape[0] if g is not None else 0
                row[f"{cam}_n_pulses"] = len(ts)
                nv, nf = frames(s, cam)
                row[f"{cam}_n_videos"], row[f"{cam}_n_frames"] = nv, nf
                row[f"{cam}_verdict"], row[f"{cam}_runs"] = verdict(ts, nf)
            row["units"] = f["units/id"].shape[0] if "units" in f else 0
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


if __name__ == "__main__":
    fs = sorted(f for f in os.listdir(NWB_DIR) if re.match(r"^[A-Z]{2}\d{3}_\d{8}_\d{6}\.nwb$", f))
    with ProcessPoolExecutor(24) as ex:
        rows = [r for r in ex.map(one, fs, chunksize=2) if r is not None]
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "camera_run_scan.csv", index=False)
    for cam in ("top", "side"):
        print(cam, df[f"{cam}_verdict"].value_counts().to_dict())
    print("wrote", OUT / "camera_run_scan.csv")
