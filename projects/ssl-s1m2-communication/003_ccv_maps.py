"""Time-shifted CCA maps (paper recipe, s1m2_lib CONFIG v2) for every (session, S1-M2 pair, condition): in-sample
total and noise CCV maps with a trial-shuffle map (paper main measure), and the 10-fold cross-validated CCV over delays
-15..+15 ms (paper S11 control). All stable units. One npz per (session, pair, condition) in OUT/maps_v<version>/, with metadata.

Jobs = (session, condition), learning first, expert last. Run on haas:
  OMP_NUM_THREADS=1 nice -n 10 python 003_ccv_maps.py --n-jobs 60 [--sessions ...] [--conditions ...] [--skip-done]
"""
import argparse
import importlib
import json
import time
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import s1m2_lib as L

def maps_dir():
    return L.OUT / f"maps_v{L.CONFIG['version']}"


def run_job(sid, cond, rows, seed=0):
    c = L.CONFIG
    t0 = time.time()
    su = importlib.import_module("002_sessions").stable_units()
    keys = su[(su.session_id == sid) & su.area_acronym_custom.isin(
        [c["s1_area"], *{a for v in c["m2_sets"].values() for a in v}])]
    keys = keys[["electrode_group", "cluster_id", "area_acronym_custom", "target_region", "quality_label", "firing_rate"]]
    units, trials = L.load_session(sid, keys)
    m = L.condition_masks(trials)[cond]
    n = int(m.sum())
    if n < c["min_trials"]:
        return [dict(session_id=sid, condition=cond, skipped=f"{n} trials")]
    tr = trials[m].reset_index(drop=True)
    edges = L.bin_edges()
    bs, wb = c["bin_s"], c["window_bins"]
    centres = np.round(np.arange(c["t_from_s"], c["t_to_s"] + 1e-9, c["step_s"]), 6)
    first_bin = np.array([int(round((ce - wb * bs / 2 - edges[0]) / bs)) for ce in centres])
    dsteps = np.arange(int(round(c["delays_s"][0] / bs)), int(round(c["delays_s"][1] / bs)) + 1)
    cvsteps = np.arange(int(round(c["cv_delays_s"][0] / bs)), int(round(c["cv_delays_s"][1] / bs)) + 1)
    starts = tr.start_time.to_numpy(float)
    groups = (tr.trial_type.astype(str) + "_" + tr.lick_flag.astype(int).astype(str)).to_numpy()
    s1 = units[units.area_acronym_custom == c["s1_area"]].reset_index(drop=True)
    X_s1 = L.count_tensor(s1.spike_times.tolist(), starts, edges).astype(np.float32)
    rng_f = np.random.default_rng(seed)
    folds = np.array_split(rng_f.permutation(n), c["cv_folds"])
    out, done = [], {}
    for _, r in rows.iterrows():
        m2 = units[units.area_acronym_custom.isin(c["m2_sets"][r.m2_set])].reset_index(drop=True)
        ukey = tuple(sorted(zip(m2.electrode_group.astype(str), m2.cluster_id.astype(int))))
        if ukey in done:  # identical M2 unit set (e.g. pooled wMO == MO-wM1 when no wM2 units): reuse
            res = done[ukey]
        else:
            res = None
        X_m2 = None if res is not None else L.count_tensor(m2.spike_times.tolist(), starts, edges).astype(np.float32)
        for kind in (() if res is not None else ("total", "noise")):
            res = res if res is not None else {}
            S, M = X_s1, X_m2
            if kind == "noise":
                S, M = L.noise_residual(S, groups), L.noise_residual(M, groups)
            k = c["pca_dims"]
            ccv, shf = L.ccv_map(L.pca_project(S, k), L.pca_project(M, k), first_bin, dsteps, wb, c["n_shuffles"],
                                 np.random.default_rng(seed))
            cv = L.cv_map(S, M, first_bin, cvsteps, wb, folds, k)
            res[f"{kind}__ccv"], res[f"{kind}__shuffle"], res[f"{kind}__cv"] = (
                ccv.astype(np.float32), shf.astype(np.float32), cv.astype(np.float32))
        done[ukey] = res
        meta = dict(session_id=sid, mouse_id=r.mouse_id, day=int(r.day), stage=r.stage, cohort=r.cohort,
                    learner=bool(r.learner), learning_category=str(r.learning_category), pair=r.pair, m2_set=r.m2_set,
                    condition=cond, n_trials=n, n_s1=len(s1), n_m2=len(m2),
                    s1_targets=";".join(sorted(s1.target_region.dropna().unique())),
                    m2_targets=";".join(sorted(m2.target_region.dropna().unique())),
                    n_trials_active=len(trials), n_trimmed_a1=int(trials.n_trimmed_a1.iloc[0]),
                    hit_rate_W=float(trials[trials.trial_type == "whisker_trial"].lick_flag.mean()),
                    hit_rate_A=float(trials[trials.trial_type == "auditory_trial"].lick_flag.mean()),
                    fa_rate=float(trials[trials.trial_type == "no_stim_trial"].lick_flag.mean()),
                    elapsed_s=time.time() - t0)
        maps_dir().mkdir(parents=True, exist_ok=True)
        f = maps_dir() / f"{sid}__{r.pair}__{cond}.npz"
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, times=centres, window_starts=centres - wb * bs / 2, delays=dsteps * bs, cv_delays=cvsteps * bs, meta=json.dumps(meta),
                            config=json.dumps(c), s1_units=s1[["electrode_group", "cluster_id"]].astype(str).to_numpy(),
                            m2_units=m2[["electrode_group", "cluster_id"]].astype(str).to_numpy(), **res)
        tmp.replace(f)
        out.append(meta)
    return out


def safe(sid, cond, rows, profile="v3"):
    L.set_profile(profile)  # loky workers import a fresh CONFIG
    try:
        r = run_job(sid, cond, rows)
        print(f"done {sid} {cond} {time.strftime('%H:%M')}", flush=True)
        return r
    except Exception:
        print(f"FAIL {sid} {cond}\n{traceback.format_exc()}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=60)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--conditions", nargs="*", default=list(L.CONFIG["conditions"]))
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--profile", default="v3", choices=list(L.PROFILES))
    a = ap.parse_args()
    L.set_profile(a.profile)
    s = pd.read_csv(L.OUT / "sessions_v2.csv")
    if a.sessions:
        s = s[s.session_id.isin(a.sessions)]
    days = s.groupby("session_id").day.first()
    order = list(days[days == 0].index) + list(days[days > 0].index)  # learning first, expert last
    jobs = [(sid, cond) for sid in order for cond in a.conditions
            if not (a.skip_done and all((maps_dir() / f"{sid}__{p}__{cond}.npz").exists()
                                        for p in s[s.session_id == sid].pair))]
    print(f"{len(order)} sessions, {len(jobs)} jobs, n_jobs {a.n_jobs}", flush=True)
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(safe)(sid, cond, s[s.session_id == sid], a.profile)
                                                    for sid, cond in jobs)
    pd.DataFrame([m for r in res for m in r]).to_csv(L.OUT / f"run_meta_v{L.CONFIG['version']}_{time.strftime('%Y%m%d_%H%M')}.csv",
                                                     index=False)


if __name__ == "__main__":
    main()
