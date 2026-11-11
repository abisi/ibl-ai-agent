"""Shared input: stimulus-aligned movement traces on whisker day 0, per session and trial type (whisker, auditory), for
mice with dual-view video (reference sheet with_dual_view == 1), in physical units.

Signals come from shared_movements.dlc_signals (ephys_utilities dlc_helpers: physical units, jaw / nose reference fix,
likelihood-gated cleaning, pupil QC); event-locked traces with dlc_helpers.peth. Signals and windows (planner
2026-11-05): whisker speed (|whisker_velocity|, deg/s), -50 to 100 ms; jaw opening (jaw_distance, mm), -100 to 300 ms;
pupil area (mm^2), -1 to 3 s. v4 (planner 2026-11-06): split by lick / no lick, and also aligned at the lick (lick
trials only): lick time = start_time + computed_reaction_time (= start_time + lick_time - response_window_start_time,
which removes the NWB lick_time artifact offset), same windows. v5 (user 2026-11-06): both alignments use each trial's
pre-stimulus mean as baseline (the pre-lick period already contains jaw opening). Rows: per session, trial type, lick
and alignment, the mean trace (value) and the mean of each trial's trace minus its pre-stimulus mean (value_delta).
Trials: whisker
and auditory trials of the shared behaviour table (early_lick == 0). Cached in Figures/_cache/.
"""

import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

import shared_behaviour
from common import CACHE_DIR, NWB_DIR, REFERENCE_SHEET
from shared_movements import data_utils, dlc_helpers, dlc_signals

PARAMS = {"version": 6, "loader": "ephys_utilities.helpers.dlc_helpers (via shared_movements.dlc_signals; peth)",
          "signals": {"whisker_speed": {"source": "whisker_speed", "window_s": [-0.05, 0.1], "dt_s": 0.005},
                      "jaw_opening": {"source": "jaw_distance", "window_s": [-0.1, 0.3], "dt_s": 0.005},
                      "pupil_area": {"source": "pupil_area", "window_s": [-1.0, 3.0], "dt_s": 0.05}},
          "trial_types": ["whisker_trial", "auditory_trial"], "alignments": ["stimulus", "lick"],
          "lick_time": "start_time + computed_reaction_time", "baseline": "each trial's pre-stimulus mean (both alignments)"}
CACHE_FILE = CACHE_DIR / f"movement_peth_day0_v{PARAMS['version']}.parquet"


def _one_session(args):
    session_id, trials = args
    if not (NWB_DIR / f"{session_id}.nwb").exists():
        return None
    try:
        sig = dlc_signals(session_id)
    except Exception as err:  # unreadable DLC: reported, skipped
        print("skip", session_id, type(err).__name__, err)
        return None
    dh = dlc_helpers()
    rows = []
    for name, spec in PARAMS["signals"].items():
        if spec["source"] not in sig:
            continue
        x, ts, unit = sig[spec["source"]]
        for (tt, lick), tr in trials.groupby(["trial_type", "lick_flag"]):
            win, dt = tuple(spec["window_s"]), spec["dt_s"]
            starts = tr.start_time.to_numpy()
            grid_s, m_s = dh.peth(x, ts, starts, window=win, dt=dt)
            with np.errstate(all="ignore"):
                base_s = np.nanmean(m_s[:, grid_s < 0], 1)  # each trial's pre-stimulus mean
            aligned = {"stimulus": (grid_s, m_s, base_s)}
            if lick == 1:
                rt = tr.computed_reaction_time.to_numpy()
                v = np.isfinite(rt) & (rt > 0)
                if v.any():
                    grid_l, m_l = dh.peth(x, ts, (starts + rt)[v], window=win, dt=dt)
                    aligned["lick"] = (grid_l, m_l, base_s[v])  # v5: pre-stimulus baseline, not pre-lick
            for align, (grid, m, base) in aligned.items():
                ok = (np.isfinite(m).mean(1) > 0.5) & np.isfinite(base)  # most of the window tracked
                if not ok.any():
                    continue
                m, base = m[ok], base[ok]
                with np.errstate(all="ignore"):
                    delta = m - base[:, None]
                    rows.append(pd.DataFrame({"session_id": session_id, "mouse_id": tr.mouse_id.iloc[0],
                                              "cohort": tr.cohort.iloc[0], "trial_type": tt, "lick": int(lick),
                                              "align": align, "signal": name, "unit": unit,
                                              "time_s": np.round(grid, 4), "value": np.nanmean(m, 0),
                                              "value_delta": np.nanmean(delta, 0), "n_trials": int(ok.sum())}))
    return pd.concat(rows, ignore_index=True) if rows else None


def build(n_workers=16):
    table = shared_behaviour.load()
    ref = pd.read_excel(REFERENCE_SHEET)
    dual = set(ref[ref.with_dual_view == 1].mouse_id)
    d0 = table[(table.behavior == "whisker") & (table.day == 0) & table.trial_type.isin(PARAMS["trial_types"])
               & (table.early_lick == 0) & table.mouse_id.isin(dual)].sort_values("start_time")
    d0 = data_utils().keep_video_data(d0, views=("side", "top"))  # v6 (2026-11-09): ephys_utilities video selection
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
