"""123 -- Single-trial decoder outputs for ALL learning sessions (user 2026-10-01: "do the single-trial decoder margin run
and compare with LT definitions"); the full run of 121 (examples).
Per session (learning stage, R+ and R-, hit vs miss, whole brain, windows 5-50 and 5-100 ms after the stimulus, dead zone
-10..+5 ms): the SAME nested repeated CV as 121 (StandardScaler -> L2 logistic regression, class_weight balanced, C chosen
inside each training fold by inner 3-fold CV on log-loss, outer stratified K-fold x N_REP repeats), giving per trial the
held-out log2 P(true class) + 1 and the session-scaled signed margin, class-balanced smoothing (Gaussian, SMOOTH_SD
trials) and the linear-shift drift null (N_SHIFT shifts of 10-50%).
Difference from 121: terminal disengagement DROPPED with rule A1 (neural-analysis rule since 2026-10-01).
Also stored per session: every LT definition (013) converted to the decoded-trial index (number of decoded trials that
start before whisker trial LT of the curve-aligned list), the curve-aligned whisker start times.
Output: 123_singletrial_scores_all.pkl {res: {(sid, window): out}, meta: {sid: ...}, config}
Compare with LT definitions: 123b_singletrial_vs_lt.py.
Run (haas, repo root): python .../123_singletrial_scores_all.py
"""

from __future__ import annotations

import importlib
import os
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
SCRIPTS = str(OUT.parents[2] / "scripts")
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, str(OUT))
M121 = importlib.import_module("121_singletrial_scores_examples")
LTP = OUT.parents[1] / "ssl-learning-trial-identification"
DEFS = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
        "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
        "lenient cascade + clean gate"]
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
PKL = OUT / f"123_singletrial_scores_all{os.environ.get('SSL_ST_TAG', '')}.pkl"   # "_v2": perf==6 excluded (2026-10-01)


def main():
    os.chdir(OUT.parents[2])
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    sess = T.hitmiss_session_list(st)
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])].set_index("session_id")
    lt = pd.read_csv(LTP / "artifacts" / "013_learning_trials_all_methods.csv").set_index("session_id")
    prev = pickle.load(open(PKL, "rb")) if PKL.exists() else dict(res={}, meta={})
    jobs, meta = [], dict(prev["meta"])
    for sid in sess.index:
        if all((sid, w) in prev["res"] for w in M121.WINDOWS):
            continue
        tr = T.prep_hitmiss_trials(root, sid, st, tt)
        if tr is None or not len(tr):
            continue
        dis = T.detect_terminal_disengagement(sid, tt)
        n0 = len(tr)
        if dis["disengaged"]:
            tr = tr[tr.start_time < dis["t_cut"]].reset_index(drop=True)
        y = (tr.lick_flag == 1).to_numpy()
        if min(y.sum(), (~y).sum()) < 4:
            continue
        t = tr.start_time.to_numpy()
        units = T.area_units(sid, "whole_brain", "All units", labels)
        if len(units) < T.MIN_UNITS_PER_AREA:
            continue
        mats = T.sliding_bin_population_matrices(T.load_session_unit_spikes(root, sid), units, t, np.ones(len(y), bool),
                                                 list(M121.WINDOWS.values()), dead_zone=M121.DZ)
        cw = T._active_trials_from_whisker_onset_for_curve(sid, tt)
        cw_t = cw.loc[cw.trial_type == "whisker_trial", "start_time"].to_numpy()
        lts = {}
        for d in DEFS:
            v = lt[d].get(sid, np.nan) if d in lt.columns else np.nan
            lts[d] = dict(lt_whisker=float(v) if pd.notna(v) else np.nan,
                          lt_decoded=int(np.sum(t < cw_t[int(v)])) if pd.notna(v) and int(v) < len(cw_t) else np.nan)
        meta[sid] = dict(reward_group=sess.loc[sid, "reward_group"], subject_id=sess.loc[sid, "subject_id"], y=y, t=t,
                         cw_t=cw_t, n_units=len(units), lts=lts, disengaged=dis["disengaged"], n_dropped=n0 - len(tr),
                         category=lt["L6 category"].get(sid, "") if "L6 category" in lt.columns else "")
        for w, X in zip(M121.WINDOWS, mats):
            if np.isnan(X).any():
                continue
            jobs.append(((sid, w), X, y, zlib.crc32(f"{sid}|{w}|all".encode())))
    print(f"[123] {len(jobs)} jobs ({len(meta)} sessions), N_REP {M121.N_REP}, N_SHIFT {M121.N_SHIFT}, {N_WORKERS} workers",
          flush=True)
    res = dict(prev["res"])
    cfg = dict(n_rep=M121.N_REP, n_shift=M121.N_SHIFT, smooth_sd=M121.SMOOTH_SD, windows=M121.WINDOWS, dz=M121.DZ,
               disengagement_rule="A1 (>=5 whisker, >=1 auditory)", script=Path(__file__).name)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=M121._init) as ex:
        futs = {ex.submit(M121.process, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                k, out = f.result()
                res[k] = out
            except Exception as e:  # noqa: BLE001
                print("[123] error", futs[f], repr(e), flush=True)
            if i % 10 == 0 or i == len(jobs):
                pickle.dump(dict(res=res, meta=meta, config=cfg), open(PKL, "wb"))
                print(f"[123] {i}/{len(jobs)} -- {time.time() - t0:.0f}s", flush=True)
    pickle.dump(dict(res=res, meta=meta, config=cfg), open(PKL, "wb"))
    print("[123] DONE", flush=True)


if __name__ == "__main__":
    main()
