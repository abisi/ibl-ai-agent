"""Shared input: pre-stimulus movements on every whisker trial of whisker day 0, for mice with dual-view video
(reference sheet with_dual_view == 1), from the DLC series of the NWB_ks4 files loaded with ephys_utilities
dlc_helpers (physical units, jaw / nose reference fix, likelihood-gated cleaning, pupil QC).

Metrics and windows as in behaviour_analysis/beh_correlates/beh_correlates_lick_decoding.py (and the GLM v3 movement
predictors): motion energy = mean |frame-to-frame difference| (dlc_helpers.window_metric, kind 'movement') of jaw
distance (side), nose distance (side and top combined, dlc_helpers.nose_norm; side only without the top camera) and
whisker angle (top) in [-0.2, 0) s before the trial start; pupil area = mean (kind 'posture') in [-2, 0) s. Trials: the
whisker trials of the shared behaviour table (early_lick == 0), so each row aligns with a point of the exact learning
curve. Cached in Figures/_cache/.
"""

import json
import sys
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

import shared_behaviour
from common import CACHE_DIR, NWB_DIR, REFERENCE_SHEET, axel_bisi_root

PARAMS = {"version": 3, "loader": "ephys_utilities.helpers.dlc_helpers (load_dlc, clean_series, nose_norm, window_metric)",
          "windows_s": {"jaw_motion": [-0.2, 0.0], "nose_motion": [-0.2, 0.0], "whisker_motion": [-0.2, 0.0],
                        "pupil_area": [-2.0, 0.0]},
          "kind": {"jaw_motion": "movement", "nose_motion": "movement", "whisker_motion": "movement",
                   "pupil_area": "posture"},
          "session_filter": "ephys_utilities data_utils.keep_video_data(views=('side', 'top'))"}
# v3 (2026-11-09, user: "use my utils only"): sessions selected with data_utils.keep_video_data (VIDEO_EXCLUDED_SESSIONS)
CACHE_FILE = CACHE_DIR / f"movements_day0_v{PARAMS['version']}.parquet"


def dlc_helpers():
    path = str(axel_bisi_root() / "Github" / "ephys_utilities")
    if path not in sys.path:
        sys.path.insert(0, path)
    from ephys_utilities.helpers import dlc_helpers as dh
    return dh


def data_utils():
    dlc_helpers()  # puts the NAS ephys_utilities checkout on sys.path
    from ephys_utilities.helpers import data_utils as du
    return du


def dlc_signals(session_id):
    """Cleaned DLC signals of one session in physical units: {name: (values, timestamps, unit)}; missing cameras or
    series are absent. jaw_distance (mm), nose (mm), whisker_angle (deg), whisker_speed (|whisker_velocity|, deg/s),
    pupil_area (mm^2)."""
    dh = dlc_helpers()
    path = NWB_DIR / f"{session_id}.nwb"
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        views = dh.load_dlc(str(path))
        side, top = views.get("side"), views.get("top")
        if side is not None:
            if "jaw_distance" in side.data:
                out["jaw_distance"] = (dh.clean_series(side, "jaw_distance")[0], side.timestamps, "mm")
            if "pupil_area" in side.data:
                out["pupil_area"] = (dh.clean_series(side, "pupil_area")[0], side.timestamps, "mm^2")
            if "nose_distance" in side.data:
                if top is not None and "top_nose_distance" in top.data:
                    out["nose"] = (*dh.nose_norm(views, "distance"), "mm")
                else:
                    out["nose"] = (dh.clean_series(side, "nose_distance")[0], side.timestamps, "mm")
        if top is not None and "whisker_angle" in top.data:
            out["whisker_angle"] = (dh.clean_series(top, "whisker_angle")[0], top.timestamps, "deg")
            out["whisker_speed"] = (np.abs(dh.clean_series(top, "whisker_velocity")[0]), top.timestamps, "deg/s")
    return out


SOURCE = {"jaw_motion": "jaw_distance", "nose_motion": "nose", "whisker_motion": "whisker_angle",
          "pupil_area": "pupil_area"}


def _one_session(args):
    session_id, trials = args
    if not (NWB_DIR / f"{session_id}.nwb").exists():
        return None
    try:
        sig = dlc_signals(session_id)
    except Exception as err:  # unreadable DLC: reported, skipped
        print("skip", session_id, type(err).__name__, err)
        return None
    if not sig:
        return None
    dh = dlc_helpers()
    out = trials[["mouse_id", "session_id", "cohort", "trial_id", "start_time"]].copy()
    for name, src in SOURCE.items():
        if src not in sig:
            out[name] = np.nan
            continue
        x, ts, _ = sig[src]
        out[name] = dh.window_metric(x, ts, out.start_time.to_numpy(), tuple(PARAMS["windows_s"][name]),
                                     kind=PARAMS["kind"][name])
    return out


def build(n_workers=16):
    table = shared_behaviour.load()
    ref = pd.read_excel(REFERENCE_SHEET)
    dual = set(ref[ref.with_dual_view == 1].mouse_id)
    d0 = table[(table.behavior == "whisker") & (table.day == 0) & (table.trial_type == "whisker_trial")
               & (table.early_lick == 0) & table.mouse_id.isin(dual)].sort_values("start_time")
    d0 = data_utils().keep_video_data(d0, views=("side", "top"))
    jobs = list(d0.groupby("session_id"))
    with Pool(n_workers) as pool:
        parts = [p for p in pool.map(_one_session, jobs) if p is not None]
    return pd.concat(parts, ignore_index=True), len(jobs)


def load(rebuild=False):
    if CACHE_FILE.exists() and not rebuild:
        return pd.read_parquet(CACHE_FILE)
    df, n_jobs = build()
    tmp = CACHE_FILE.with_suffix(".tmp")
    df.to_parquet(tmp)
    tmp.rename(CACHE_FILE)
    json.dump({**PARAMS, "n_sessions_requested": n_jobs, "n_sessions": int(df.session_id.nunique())},
              open(CACHE_FILE.with_suffix(".json"), "w"), indent=1)
    return df
