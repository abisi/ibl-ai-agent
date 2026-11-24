"""Step 2 of the connected-pairs extension: time-shifted CCA maps (same recipe as 003, s1m2_lib functions) for the
selected area pairs, both profiles: "v3" (5-ms bins, 50-ms windows every 10 ms, delays +-100 ms, in-sample total / noise
CCV + one trial-shuffle map + 10-fold CV over delays +-15 ms; PCA top 10 per area) and "hires" (1-ms bins, 10-ms
windows every 1 ms, delays +-20 ms). The anatomical source (pairs_lib orientation: wS1 > wS2 > A1) plays the S1 role:
delay > 0 = source leads. Units: all stable good + mua units of each area (>= PAIR_CONFIG min_units = 5).

Selected pairs: every extracted pair of tables/connectivity_pairs.csv with pass_any (>= 3 sessions at both stages in >= 1
cohort), all classes (strong / weak / none: the weak / none pairs are the projection-strength controls).
Jobs = (profile, session, condition); all selected pairs of the session in one job (each area's counts computed once).
Order: learning sessions (v3 then hires) first, all expert jobs last. Output one npz per (session, pair, condition) in
OUT/pairs/maps_<profile>/ with full metadata (session, mouse, cohort, stage, pair, units, config, s1m2_lib md5).

Run (haas): OMP_NUM_THREADS=1 nice -n 10 ../../.venv/bin/python 009_pairs_ccv.py --n-jobs 100 [--pairs ...]
            [--sessions ...] [--profiles v3 hires] [--skip-done]
"""
import argparse
import copy
import json
import time
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import pairs_lib as P

L = P.L
BASE_CONFIG = copy.deepcopy(L.CONFIG)
CONDS = list(BASE_CONFIG["conditions"])


def use_profile(name):
    """reset to the base CONFIG before applying a profile (loky workers are reused across profiles)"""
    L.CONFIG.clear()
    L.CONFIG.update(copy.deepcopy(BASE_CONFIG))
    return L.set_profile(name)


def maps_dir(profile):
    return P.OUT / ("maps_v3" if profile == "v3" else "maps_v4_hires")


def run_job(profile, sid, cond, rows, keys, seed=0):
    c = use_profile(profile)
    t0 = time.time()
    units, trials = L.load_session(sid, keys)
    m = L.condition_masks(trials)[cond]
    n = int(m.sum())
    if n < c["min_trials"]:
        return [dict(session_id=sid, condition=cond, profile=profile, skipped=f"{n} trials")]
    tr = trials[m].reset_index(drop=True)
    edges = L.bin_edges()
    bs, wb = c["bin_s"], c["window_bins"]
    centres = np.round(np.arange(c["t_from_s"], c["t_to_s"] + 1e-9, c["step_s"]), 6)
    first_bin = np.array([int(round((ce - wb * bs / 2 - edges[0]) / bs)) for ce in centres])
    dsteps = np.arange(int(round(c["delays_s"][0] / bs)), int(round(c["delays_s"][1] / bs)) + 1)
    cvsteps = np.arange(int(round(c["cv_delays_s"][0] / bs)), int(round(c["cv_delays_s"][1] / bs)) + 1)
    starts = tr.start_time.to_numpy(float)
    groups = (tr.trial_type.astype(str) + "_" + tr.lick_flag.astype(int).astype(str)).to_numpy()
    folds = np.array_split(np.random.default_rng(seed).permutation(n), c["cv_folds"])
    X, out = {}, []

    def area_x(a):
        if a not in X:
            ua = units[units.area == a].reset_index(drop=True)
            X[a] = (ua, L.count_tensor(ua.spike_times.tolist(), starts, edges).astype(np.float32))
        return X[a]

    for _, r in rows.iterrows():
        (us, Xs), (ut, Xt) = area_x(r.source), area_x(r.target)
        res = {}
        for kind in ("total", "noise"):
            S, M = (Xs, Xt) if kind == "total" else (L.noise_residual(Xs, groups), L.noise_residual(Xt, groups))
            k = c["pca_dims"]
            ccv, shf = L.ccv_map(L.pca_project(S, k), L.pca_project(M, k), first_bin, dsteps, wb, c["n_shuffles"],
                                 np.random.default_rng(seed))
            cv = L.cv_map(S, M, first_bin, cvsteps, wb, folds, k)
            res[f"{kind}__ccv"], res[f"{kind}__shuffle"], res[f"{kind}__cv"] = (
                ccv.astype(np.float32), shf.astype(np.float32), cv.astype(np.float32))
        meta = dict(session_id=sid, mouse_id=r.mouse_id, day=int(r.day), stage=r.stage, cohort=r.cohort,
                    learner=bool(r.learner), learning_category=str(r.learning_category), pair=r.pair, source=r.source,
                    target=r.target, profile=profile, condition=cond, n_trials=n, n_source=len(us), n_target=len(ut),
                    source_probes=";".join(sorted(us.electrode_group.astype(str).unique())),
                    target_probes=";".join(sorted(ut.electrode_group.astype(str).unique())),
                    same_probe_frac=float(ut.electrode_group.isin(set(us.electrode_group)).mean()),
                    n_trials_active=len(trials), n_trimmed_a1=int(trials.n_trimmed_a1.iloc[0]),
                    hit_rate_W=float(trials[trials.trial_type == "whisker_trial"].lick_flag.mean()),
                    hit_rate_A=float(trials[trials.trial_type == "auditory_trial"].lick_flag.mean()),
                    fa_rate=float(trials[trials.trial_type == "no_stim_trial"].lick_flag.mean()),
                    s1m2_lib_md5=P.lib_md5(), elapsed_s=time.time() - t0)
        d = maps_dir(profile)
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{sid}__{r.pair}__{cond}.npz"
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez_compressed(tmp, times=centres, window_starts=centres - wb * bs / 2, delays=dsteps * bs,
                            cv_delays=cvsteps * bs, meta=json.dumps(meta), config=json.dumps(c),
                            pair_config=json.dumps(P.PAIR_CONFIG),
                            source_units=us[["electrode_group", "cluster_id"]].astype(str).to_numpy(),
                            target_units=ut[["electrode_group", "cluster_id"]].astype(str).to_numpy(), **res)
        tmp.replace(f)
        out.append(meta)
    return out


def safe(*a):
    try:
        r = run_job(*a)
        print(f"done {a[0]} {a[1]} {a[2]} {len(a[3])} pairs {time.strftime('%H:%M')}", flush=True)
        return r
    except Exception:
        print(f"FAIL {a[0]} {a[1]} {a[2]}\n{traceback.format_exc()}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=100)
    ap.add_argument("--pairs", nargs="*")
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--conditions", nargs="*", default=CONDS)
    ap.add_argument("--profiles", nargs="*", default=["v3", "hires"])
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    C = pd.read_csv(P.TABLES / "connectivity_pairs.csv")
    sel = C[C.extracted & C.pass_any].pair.tolist() if not a.pairs else a.pairs
    S = pd.read_csv(P.TABLES / "pair_sessions.csv")
    S = S[S.pair.isin(sel)]
    if a.sessions:
        S = S[S.session_id.isin(a.sessions)]
    u = P.stable_units()
    u = u[u.session_id.isin(S.session_id) & u.area.notna()]
    keys = {sid: g[["electrode_group", "cluster_id", "area", "area_acronym_custom", "target_region", "quality_label",
                    "firing_rate"]] for sid, g in u.groupby("session_id")}
    jobs = []
    for stage in ("learning", "expert"):           # expert jobs last (all profiles)
        for prof in a.profiles:
            for sid, rows in S[S.stage == stage].groupby("session_id"):
                for cond in a.conditions:
                    todo = rows
                    if a.skip_done:                # only the pairs of this session without a map yet
                        todo = rows[np.array([not (maps_dir(prof) / f"{sid}__{p}__{cond}.npz").exists()
                                              for p in rows.pair], bool)]
                        if todo.empty:
                            continue
                    jobs.append((prof, sid, cond, todo, keys[sid]))
    print(f"{len(sel)} pairs, {S.session_id.nunique()} sessions, {len(S)} session-pairs, {len(jobs)} jobs, "
          f"n_jobs {a.n_jobs}", flush=True)
    res = Parallel(n_jobs=a.n_jobs, backend="loky", batch_size=1)(delayed(safe)(*j) for j in jobs)
    P.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([m for r in res for m in r]).to_csv(P.OUT / f"run_meta_{time.strftime('%Y%m%d_%H%M')}.csv", index=False)


if __name__ == "__main__":
    main()
