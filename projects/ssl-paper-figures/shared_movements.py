"""Shared input: pre-stimulus movements on every whisker trial of whisker day 0, for mice with dual-view video
(reference sheet with_dual_view == 1), from the DLC time series of the NWB_ks4 files (NWB_reader.get_dlc_data_dict).

Metrics and windows as in behaviour_analysis/beh_correlates/beh_correlates_lick_decoding.py (and the GLM v3 movement
predictors): motion energy = mean |diff| of jaw_distance, nose_norm_distance and whisker_angle in [-0.2, 0) s before
the trial start; pupil area = mean of pupil_area in [-2, 0) s. Trials: the whisker trials of the shared behaviour table
(early_lick == 0), so each row aligns with a point of the exact learning curve. Cached in Figures/_cache/.
"""

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import shared_behaviour
from common import CACHE_DIR, NWB_DIR, REFERENCE_SHEET, import_external_repos

PARAMS = {"version": 1, "windows_s": {"jaw_distance": [-0.2, 0.0], "nose_norm_distance": [-0.2, 0.0],
                                      "whisker_angle": [-0.2, 0.0], "pupil_area": [-2.0, 0.0]},
          "aggregate": {"jaw_distance": "motion energy", "nose_norm_distance": "motion energy",
                        "whisker_angle": "motion energy", "pupil_area": "mean"}}
METRICS = {"jaw_distance": "jaw_motion", "nose_norm_distance": "nose_motion", "whisker_angle": "whisker_motion",
           "pupil_area": "pupil_area"}
CACHE_FILE = CACHE_DIR / f"movements_day0_v{PARAMS['version']}.parquet"


def _one_session(args):
    session_id, trials = args
    import_external_repos()
    import NWB_reader_functions as nwb_reader
    path = NWB_DIR / f"{session_id}.nwb"
    if not path.exists():
        return None
    try:
        dlc = nwb_reader.get_dlc_data_dict(str(path))
    except Exception as err:  # unreadable DLC: reported, skipped
        print("skip", session_id, type(err).__name__, err)
        return None
    if dlc is None:
        return None
    out = trials[["mouse_id", "session_id", "cohort", "trial_id", "start_time"]].copy()
    for part, name in METRICS.items():
        if part not in dlc or np.isscalar(dlc[part]["data"]):
            out[name] = np.nan
            continue
        data, ts = np.asarray(dlc[part]["data"], float), np.asarray(dlc[part]["timestamps"], float)
        n = min(len(data), len(ts))  # some DLC series have one sample more than their timestamps
        data, ts = data[:n], ts[:n]
        t0, t1 = PARAMS["windows_s"][part]
        values = []
        for t in out.start_time.to_numpy():
            seg = data[(ts >= t + t0) & (ts < t + t1)]
            if PARAMS["aggregate"][part] == "mean":
                values.append(np.nanmean(seg) if len(seg) else np.nan)
            else:
                values.append(np.nanmean(np.abs(np.diff(seg))) if len(seg) > 1 else np.nan)
        out[name] = values
    return out


def build(n_workers=16):
    table = shared_behaviour.load()
    ref = pd.read_excel(REFERENCE_SHEET)
    dual = set(ref[ref.with_dual_view == 1].mouse_id)
    d0 = table[(table.behavior == "whisker") & (table.day == 0) & (table.trial_type == "whisker_trial")
               & (table.early_lick == 0) & table.mouse_id.isin(dual)].sort_values("start_time")
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
