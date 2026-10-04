"""128 -- Single-trial decoder outputs for STIMULUS decodings whose labels are not behaviour (user 2026-10-01: "then with
whisker vs no-stim / whisker vs auditory stim overnight" -- the non-circular counterpart of the hit/miss margin, 123).
Same decoder and outputs as 121/123 (nested repeated CV, L2 logistic regression, per-trial held-out log-probability and
session-scaled signed margin, class-balanced smoothing, linear-shift drift null), whole brain, learning-stage sessions.
Decodings (stimulus-aligned on trial start_time; windows 5-50 and 5-100 ms; dead zone -10..+5 ms):
  whisker_nostim    whisker vs no-stim trials, ALL outcomes (licked or not), y = whisker;
  whisker_auditory  whisker vs auditory trials, all outcomes (prep_modality_trials), y = whisker.
Trials: prep_session (active context, perf != 6 -- user rule; warm-up block cut keeping 1 trial before the first whisker
trial); terminal disengagement dropped with rule A1.
Output: 128_singletrial_<decoding>.pkl {res: {(sid, window): out}, meta: {sid: y, t, trial_type, lick, cw_t, ...}, config}
Run (haas, repo root): python .../128_singletrial_stimulus_decoding.py whisker_nostim|whisker_auditory
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
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "40"))
TYPES = {"whisker_nostim": ("whisker_trial", "no_stim_trial"), "whisker_auditory": ("whisker_trial", "auditory_trial")}


def main():
    decoding = sys.argv[1]
    a_type, b_type = TYPES[decoding]
    pkl = OUT / f"128_singletrial_{decoding}.pkl"
    os.chdir(OUT.parents[2])
    import warnings
    warnings.filterwarnings("ignore")
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    from ssl_bwm_trial_prep import prep_session
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    sess = T.hitmiss_session_list(st)
    sess = sess[(sess.day_stage == "learning") & sess.reward_group.isin(["R+", "R-"])].set_index("session_id")
    prev = pickle.load(open(pkl, "rb")) if pkl.exists() else dict(res={}, meta={})
    jobs, meta = [], dict(prev["meta"])
    for sid in sess.index:
        if all((sid, w) in prev["res"] for w in M121.WINDOWS):
            continue
        p = prep_session(root, sid, st, tt)
        if p is None:
            continue
        tr = p["trials"]
        tr = tr[tr.trial_type.isin([a_type, b_type])].reset_index(drop=True)
        dis = T.detect_terminal_disengagement(sid, tt)
        n0 = len(tr)
        if dis["disengaged"]:
            tr = tr[tr.start_time < dis["t_cut"]].reset_index(drop=True)
        y = (tr.trial_type == a_type).to_numpy()
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
        meta[sid] = dict(reward_group=sess.loc[sid, "reward_group"], subject_id=sess.loc[sid, "subject_id"], y=y, t=t,
                         trial_type=tr.trial_type.to_numpy(), lick=tr.lick_flag.to_numpy(), cw_t=cw_t, n_units=len(units),
                         disengaged=dis["disengaged"], n_dropped=n0 - len(tr))
        for w, X in zip(M121.WINDOWS, mats):
            if np.isnan(X).any():
                continue
            jobs.append(((sid, w), X, y, zlib.crc32(f"{sid}|{w}|{decoding}".encode())))
    print(f"[128 {decoding}] {len(jobs)} jobs ({len(meta)} sessions), N_REP {M121.N_REP}, N_SHIFT {M121.N_SHIFT}", flush=True)
    res = dict(prev["res"])
    cfg = dict(decoding=decoding, n_rep=M121.N_REP, n_shift=M121.N_SHIFT, smooth_sd=M121.SMOOTH_SD, windows=M121.WINDOWS,
               dz=M121.DZ, disengagement_rule="A1", perf6_excluded=True, script=Path(__file__).name)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=M121._init) as ex:
        futs = {ex.submit(M121.process, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                k, out = f.result()
                res[k] = out
            except Exception as e:  # noqa: BLE001
                print("[128] error", futs[f], repr(e), flush=True)
            if i % 10 == 0 or i == len(jobs):
                pickle.dump(dict(res=res, meta=meta, config=cfg), open(pkl, "wb"))
                print(f"[128 {decoding}] {i}/{len(jobs)} -- {time.time() - t0:.0f}s", flush=True)
    pickle.dump(dict(res=res, meta=meta, config=cfg), open(pkl, "wb"))
    print(f"[128 {decoding}] DONE", flush=True)


if __name__ == "__main__":
    main()
