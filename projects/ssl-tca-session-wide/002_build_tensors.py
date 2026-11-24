"""Per-session spike-count tensors (trials x units x 5-ms bins, -1..+1 s around every trial start), all selected trials
(passive pre, active, passive post), artefact window removed. Trial table carries epoch, outcome, corrected first-lick
time and the exact learning curves. Reused by every later step.

Run: python 002_build_tensors.py [--pilot] [--n-jobs 30] [--skip-done]
     -> OUT/artifacts/tensors/<session_id>.npz (X uint8, t_centres), <session_id>_units.parquet, <session_id>_trials.parquet
"""
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import tca_lib as L


def flat(df):
    """Parquet-safe copy: drop array-valued columns, other object columns as str."""
    keep = [c for c in df.columns
            if not (df[c].dtype == object and df[c].map(lambda v: isinstance(v, (list, tuple, np.ndarray))).any())]
    df = df[keep].copy()
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].astype(str)
    return df


def one(sid, mouse, day, cohort, keys):
    try:
        t0 = time.time()
        units, trials = L.load_session(sid, keys)
        t = L.select_trials(trials)
        att = dict(t.attrs)
        # corrected first lick (ssl-lick-alignment): start_time + lick_time - response_window_start_time
        if {"lick_time", "response_window_start_time"} <= set(t.columns):
            t["lick_latency"] = t.lick_time - t.response_window_start_time
        t["cohort"] = cohort
        t["trial_index"] = np.arange(len(t))
        t, meta = L.learning_curves(mouse, day, t)
        t = t.sort_values("start_time").reset_index(drop=True)
        X = L.bin_tensor(units, t)
        L.TENSORS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(L.TENSORS / f"{sid}.npz", X=X, t_centres=L.bin_centres())
        flat(units.drop(columns=["spike_times"])).to_parquet(L.TENSORS / f"{sid}_units.parquet", index=False)
        flat(t).to_parquet(L.TENSORS / f"{sid}_trials.parquet", index=False)
        info = dict(session_id=sid, ok=True, n_trials=len(t), n_units=len(units), secs=round(time.time() - t0),
                    **att, **{k: (float(v) if v is not None and np.isscalar(v) else None) for k, v in meta.items()},
                    epochs=json.dumps(t.epoch.value_counts().to_dict()))
    except Exception:
        info = dict(session_id=sid, ok=False, error=traceback.format_exc()[-800:])
    print(info.get("session_id"), info.get("ok"), info.get("n_trials"), info.get("n_units"), info.get("secs"), flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=30)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    if a.pilot:
        s = s[s.pilot]
    if a.skip_done:
        s = s[[not (L.TENSORS / f"{x}_trials.parquet").exists() for x in s.session_id]]
    u = pd.read_parquet(L.OUT / "tables" / "units.parquet")
    keys = {sid: g[["electrode_group", "cluster_id", "area_acronym_custom", "target_region"]].reset_index(drop=True)
            for sid, g in u.groupby("session_id")}
    print(f"{len(s)} sessions", flush=True)
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(
        delayed(one)(r.session_id, r.mouse_id, r.day, r.cohort, keys[r.session_id]) for r in s.itertuples())
    out = pd.DataFrame(res)
    f = L.OUT / "tables" / "tensor_build_log.csv"
    if f.exists():
        out = pd.concat([pd.read_csv(f), out]).drop_duplicates("session_id", keep="last")
    out.to_csv(f, index=False)
    (L.OUT / "tables" / "config.json").write_text(json.dumps(L.CONFIG, indent=1))


if __name__ == "__main__":
    main()
