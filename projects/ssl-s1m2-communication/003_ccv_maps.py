"""Time-shifted CCA maps (paper recipe, s1m2_lib) for every (session, S1-M2 pair, condition), total and noise CCV,
with a trial-shuffle map. One npz per (session, pair) in OUT/maps/, holding the maps and all metadata.

Jobs are ordered learning first, expert last (scheduler rule). Run on haas:
  OMP_NUM_THREADS=1 nice -n 10 python 003_ccv_maps.py --n-jobs 60 [--sessions AB127_20240821_103757]
"""
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import s1m2_lib as L

MAPS = L.OUT / "maps"


def run_session(sid, rows, seed=0):
    c = L.CONFIG
    t0 = time.time()
    keys = pd.read_parquet(L.UNIT_TABLE, filters=[("session_id", "==", sid)])
    keys = keys[keys.quality_label.isin(c["unit_quality"]) & (keys.firing_rate >= c["min_rate_hz"])
                & keys.area_acronym_custom.isin([c["s1_area"], *c["m2_areas"]])]
    keys = keys[["electrode_group", "cluster_id", "area_acronym_custom", "target_region", "quality_label", "firing_rate"]]
    units, trials = L.load_session(sid, keys)
    edges = L.bin_edges()
    bs, wb = c["bin_s"], c["window_bins"]
    centres = np.round(np.arange(c["t_from_s"], c["t_to_s"] + 1e-9, c["step_s"]), 6)
    first_bin = np.array([int(round((ce - wb * bs / 2 - edges[0]) / bs)) for ce in centres])
    dsteps = np.arange(int(round(c["delays_s"][0] / bs)), int(round(c["delays_s"][1] / bs)) + 1)
    starts = trials.start_time.to_numpy(float)
    draw = np.random.default_rng(seed)

    def cap(df):
        return df.iloc[np.sort(draw.choice(len(df), c["max_units"], replace=False))] if len(df) > c["max_units"] else df

    s1_all = units[units.area_acronym_custom == c["s1_area"]].reset_index(drop=True)
    s1 = cap(s1_all).reset_index(drop=True)
    X_s1 = L.count_tensor(s1.spike_times.tolist(), starts, edges)
    masks = L.condition_masks(trials)
    groups = (trials.trial_type.astype(str) + "_" + trials.lick_flag.astype(int).astype(str)).to_numpy()
    out = []
    for _, r in rows.iterrows():
        m2_all = units[units.area_acronym_custom == r.m2_area].reset_index(drop=True)
        m2 = cap(m2_all).reset_index(drop=True)
        X_m2 = L.count_tensor(m2.spike_times.tolist(), starts, edges)
        res = {}
        for cond, m in masks.items():
            n = int(m.sum())
            res[f"{cond}__n_trials"] = n
            if n < c["min_trials"]:
                continue
            for kind in ("total", "noise"):
                S, M = X_s1[m], X_m2[m]
                if kind == "noise":
                    S, M = L.noise_residual(S, groups[m]), L.noise_residual(M, groups[m])
                rng = np.random.default_rng(seed)
                ccv, shf = L.ccv_map(S, M, first_bin, dsteps, wb, c["n_shuffles"], rng)
                res[f"{cond}__{kind}__ccv"] = ccv.astype(np.float32)
                res[f"{cond}__{kind}__shuffle"] = shf.astype(np.float32)
        meta = dict(session_id=sid, mouse_id=r.mouse_id, day=int(r.day), stage=r.stage, cohort=r.cohort,
                    learner=bool(r.learner), learning_category=str(r.learning_category), pair=r.pair,
                    s1_area=c["s1_area"], m2_area=r.m2_area, n_s1=len(s1), n_m2=len(m2),
                    n_s1_available=len(s1_all), n_m2_available=len(m2_all),
                    s1_targets=";".join(sorted(s1.target_region.dropna().unique())),
                    m2_targets=";".join(sorted(m2.target_region.dropna().unique())),
                    n_trials_active=len(trials), n_trimmed_a1=int(trials.n_trimmed_a1.iloc[0]) if len(trials) else 0,
                    hit_rate_W=float(trials[masks["W_all"]].lick_flag.mean()),
                    hit_rate_A=float(trials[trials.trial_type == "auditory_trial"].lick_flag.mean()),
                    fa_rate=float(trials[masks["NS_all"]].lick_flag.mean()),
                    elapsed_s=time.time() - t0)
        MAPS.mkdir(parents=True, exist_ok=True)
        f = MAPS / f"{sid}__{r.pair}.npz"
        tmp = f.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, times=centres, delays=dsteps * bs, meta=json.dumps(meta), config=json.dumps(c),
                            s1_units=s1[["electrode_group", "cluster_id"]].to_numpy(),
                            m2_units=m2[["electrode_group", "cluster_id"]].to_numpy(), **res)
        tmp.replace(f)
        out.append(meta)
    return out


def safe(sid, rows):
    try:
        r = run_session(sid, rows)
        print(f"done {sid} {[m['pair'] for m in r]} {r[0]['elapsed_s']:.0f}s", flush=True)
        return r
    except Exception:
        print(f"FAIL {sid}\n{traceback.format_exc()}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=60)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "sessions.csv")
    if a.sessions:
        s = s[s.session_id.isin(a.sessions)]
    if a.skip_done:
        s = s[[not (MAPS / f"{x.session_id}__{x.pair}.npz").exists() for x in s.itertuples()]]
    order = s.groupby("session_id").day.first().sort_values(kind="stable")  # day 0 first, expert last
    order = (order > 0).sort_values(kind="stable").index
    print(f"{len(order)} sessions, {len(s)} session-pairs, n_jobs {a.n_jobs}", flush=True)
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(safe)(sid, s[s.session_id == sid]) for sid in order)
    pd.DataFrame([m for r in res for m in r]).to_csv(L.OUT / f"run_meta_{time.strftime('%Y%m%d_%H%M')}.csv", index=False)


if __name__ == "__main__":
    main()
