"""Shared paths, constants and loaders for ssl-behaviour-motifs (runs on haas; resolves the NAS via axel_bisi_paths).

Conventions
- Keypoints: merged DLC model (CspnextM shuffle 6, snapshot 330), `*_filtered_new.h5` in the session's Video/ folder on the
  NAS (identical files to Kuma's dlc_merged/inference_cspnext_m_330). Frame i of the h5 = frame i of the NWB DLC series;
  the session time of each frame is the camera's own timestamps from NWB_ks4 via ephys_utilities dlc_helpers.load_dlc
  (timing rule of dlc_helpers: never start time x nominal rate).
- Trials from NWB_ks4 (canonical); first lick corrected per trial (skills/ssl-lick-alignment); active context, perf != 6,
  auditory warm-up block cut (skills/ssl-trial-exclusion); no disengagement trimming (behaviour analysis).
- Cohort and learner label per mouse from the reference sheet in share_internal (memory rules).
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root()
sys.path.insert(0, str(ROOT / "Github" / "ephys_utilities"))  # NAS checkout first (uncommitted dlc_helpers fixes)
NWB_DIR = ROOT / "NWB_ks4"
RES = ROOT / "combined_results_ks4" / "ssl-behaviour-motifs"
TABLES, CACHE, FIGS, REPORTS = RES / "tables", RES / "cache", RES / "figures", RES / "reports"
SHARE_INTERNAL = [Path(r"M:\share_internal"), Path("/mnt/share_internal")]
SLUG = "ssl-behaviour-motifs"

FS_DS = 200.0         # analysis frame rate (Hz): native camera rate, no downsampling (v2, user 2026-11-19; v1 used 50 Hz)
DS = 1
# NWB_ks4 DLC series names (dlc_helpers.load_dlc) of the modelled / static keypoints: top-camera points carry a top_ prefix
# except the whisker points
NWB_NAME = {"side": lambda kp: kp, "top": lambda kp: kp if kp.startswith("whisker") else f"top_{kp}"}
VIEWS = ("side", "top")
# v3 (user 2026-11-19): only moving keypoints are modelled (particle and jaw_ref move: kinematic points of the whisker /
# jaw angles); the spout tip is the single static reference (translation to a common frame, mm via the calibration)
KP_MODEL = {"side": ["nose_tip", "jaw", "jaw_ref", "tongue"], "top": ["nose_tip", "whisker_base", "whisker_tip", "particle"]}
KP_STATIC = {"side": ["spout", "nose_base"], "top": ["spout", "nose_base"]}  # spout = alignment reference; nose_base saved only
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
TASK_COL = {"whisker": "#f7b519", "auditory": "#2c2cdb", "no_stim": "#262626"}


def reference_sheet() -> pd.DataFrame:
    for p in SHARE_INTERNAL:
        f = p / "Axel_Bisi_Share" / "dataset_info" / "joint_mouse_reference_weight.xlsx"
        if f.exists():
            return pd.read_excel(f)
    raise FileNotFoundError("joint_mouse_reference_weight.xlsx not found in share_internal")


def video_dir(session_id: str) -> Path:
    mouse = session_id.split("_")[0]
    owner = "Axel_Bisi" if mouse.startswith("AB") else "Myriam_Hamon"
    return ROOT.parent / owner / "data" / mouse / session_id / "Video"


def dlc_h5(session_id: str, view: str) -> Path | None:
    fs = [f for f in glob.glob(str(video_dir(session_id) / f"*{view}*CspnextM*shuffle6*filtered_new.h5")) if "_backup" not in f]
    return Path(fs[0]) if len(fs) == 1 else None


def read_h5(path) -> pd.DataFrame:
    """DLC h5 -> DataFrame with columns (bodypart, coord)."""
    df = pd.read_hdf(path)
    df.columns = pd.MultiIndex.from_tuples([(c[-2], c[-1]) for c in df.columns])
    return df


def load_trials(session_id: str) -> pd.DataFrame:
    """NWB_ks4 trials with corrected first lick, outcome labels and the trial-selection flags."""
    with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
        g = f["intervals/trials"]
        cols = ["start_time", "stop_time", "trial_type", "lick_flag", "lick_time", "response_window_start_time",
                "response_window_stop_time", "perf", "context", "early_lick", "reward_available"]
        t = pd.DataFrame({c: g[c][:] for c in cols if c in g})
    for c in ("trial_type", "context"):
        t[c] = t[c].map(lambda x: x.decode() if isinstance(x, bytes) else str(x))
    t["reaction_time"] = t["lick_time"] - t["response_window_start_time"]
    t["first_lick_time"] = t["start_time"] + t["reaction_time"]  # physical first lick (lick_time is late by the artifact window)
    t["stim"] = t["trial_type"].str.replace("_trial", "", regex=False)
    t["active"] = (t["context"] != "passive") & (t["perf"] != 6)
    wh = np.flatnonzero((t["stim"] == "whisker") & t["active"])
    keep = t["active"].copy()
    if len(wh):  # auditory warm-up block: keep exactly one trial before the first whisker trial
        first = wh[0]
        keep.iloc[: max(first - 1, 0)] = False
    t["keep"] = keep
    t["whisker_trial_index"] = np.where(t.keep & (t.stim == "whisker"), np.cumsum(t.keep & (t.stim == "whisker")) - 1, -1)
    return t


POST_S = 5.0     # post-response epoch: response-window end -> +5 s (reward consumption / lick tail)
PRE_S = 2.0      # pre-stimulus epoch: the 2 s before the stimulus


def trial_windows(session_id: str, kept: pd.DataFrame) -> pd.DataFrame:
    """Epoch windows (session time) of the kept trials (user 2026-11-20):
    pre  = [stim - 2 s, stim); resp = [stim, response-window end];
    post = (response-window end, end + 5 s]  (reward consumption; capped at the next quiet window);
    iti  = (response-window end + 5 s, start of the NEXT trial's quiet window (abort_window_start_time)) -- empty (NaN)
           when that interval is < 0.1 s or there is no next trial.
    The next trial is the next trial of the NWB (kept or not)."""
    with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
        g = f["intervals/trials"]
        st = g["start_time"][:]
        qs = g["abort_window_start_time"][:] if "abort_window_start_time" in g else st.copy()
    qs = np.where(np.isfinite(qs) & (qs <= st), qs, st)  # no quiet window recorded -> next stimulus
    order = np.argsort(st)
    st, qs = st[order], qs[order]
    k = kept.reset_index(drop=True)
    j = np.searchsorted(st, k.start_time.to_numpy() + 1e-6)  # index of the next NWB trial
    nxt_q = np.where(j < len(st), qs[np.minimum(j, len(st) - 1)], np.nan)
    rw = k.response_window_stop_time.to_numpy()
    out = pd.DataFrame({"pre_start": k.start_time - PRE_S, "pre_stop": k.start_time, "resp_start": k.start_time,
                        "resp_stop": rw, "post_start": rw, "post_stop": np.fmin(rw + POST_S, nxt_q),
                        "iti_start": rw + POST_S, "iti_stop": nxt_q, "next_quiet_start": nxt_q})
    bad = ~(out.iti_stop - out.iti_start >= 0.1)
    out.loc[bad, ["iti_start", "iti_stop"]] = np.nan
    return out


def piezo_licks(session_id: str) -> np.ndarray:
    with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
        g = f["processing/behavior/BehavioralEvents/piezo_lick_times"]
        return np.asarray(g["timestamps"][:], float)


def jaw_onsets(session_id: str) -> pd.DataFrame:
    """per-trial jaw onsets: the user's per-mouse dlc_jaw_onset_times.pkl; if the mouse has none (MH026), the same
    function that builds it (dlc_helpers.jaw_onset_times, JAW_ONSET_PARAMS) on the NWB_ks4 trials. Columns: start_time,
    trial_type, jaw_dlc_onset (s from start), piezo_lick_time, threshold (mm), jaw_onset_abs, source."""
    mouse = session_id.split("_")[0]
    p = ROOT / "combined_results_ks4" / mouse / "dlc_jaw_onset_times.pkl"
    cols = ["start_time", "trial_type", "jaw_dlc_onset", "piezo_lick_time", "context", "perf", "threshold"]
    if p.exists():
        d = pd.read_pickle(p)
        d = d[d.session_id == session_id][cols].assign(source="pkl")
    else:
        from ephys_utilities.helpers import dlc_helpers
        with h5py.File(NWB_DIR / f"{session_id}.nwb", "r") as f:
            g = f["intervals/trials"]
            t = pd.DataFrame({c: g[c][:] for c in ["start_time", "trial_type", "response_window_start_time", "lick_time",
                                                   "context", "perf", "stop_time"]})
        for c in ("trial_type", "context"):
            t[c] = t[c].map(lambda x: x.decode() if isinstance(x, bytes) else str(x))
        d = dlc_helpers.jaw_onset_times(str(NWB_DIR / f"{session_id}.nwb"), t)[cols].assign(source="dlc_helpers.jaw_onset_times")
    return d.assign(jaw_onset_abs=d.start_time + d.jaw_dlc_onset)


def versions() -> dict:
    import importlib
    out = {}
    for m in ("numpy", "pandas", "scipy", "sklearn", "matplotlib", "h5py"):
        try:
            out[m] = importlib.import_module(m).__version__
        except Exception:
            pass
    return out
