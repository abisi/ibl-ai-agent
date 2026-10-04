"""104 -- Definition-independent placebo splits for EVERY session (user request 2026-09-25: "Placebo shifts
should be the same right? As they don't rely on a LT definition").

For each learning-stage session (whole brain, 3 windows, disengaged trials KEPT), the size-matched pre/post
hit/miss decoding (same estimator as 097: matched balanced accuracy, N_SUBSAMPLE x N_REPEATS, one C per
session/window from all trials) is computed at:
  - every STEP-th whisker trial (curve-aligned index) whose split leaves >= 2 hits and >= 2 misses in both
    epochs (capped at MAX_POSITIONS, evenly thinned)                       -> kind = "placebo"
  - the session midpoint (median decoded whisker trial; as in the 095 halves split) -> kind = "midpoint"
  - the learning trial of the evaluation table, if the session has one        -> kind = "lt"
Placebo positions do NOT depend on any learning-trial definition, so the table can be reused for any
definition later (a new LT only needs its own extra row).
Output: 104_placebo_all_whole_brain.parquet (one row per session x window x split).
"""

from __future__ import annotations

import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

WINDOWS = {"sensory": (0.005, 0.050), "baseline": (-0.200, -0.010)}
WINDOW_NAMES = ["sensory", "baseline", "sensory_minus_base"]
WIDE_DEAD_ZONE = (-0.010, 0.005)
MIN_PER_CLASS = 2
MAX_FOLDS = 5
STEP = 3
MAX_POSITIONS = 60
N_SUBSAMPLE = 30
N_REPEATS = 3
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "24"))
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / "104_placebo_all_whole_brain.parquet"
LT_TABLE = OUT_DIR.parents[1] / "ssl-learning-trial-identification" / "artifacts" / "020_lt_eval.csv"


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def matched(X, y, is_pre, C, rng, decode_bin_pooled):
    idx = {"pre": np.where(is_pre)[0], "post": np.where(~is_pre)[0]}
    cnt = {ep: (int(y[i].sum()), int((~y[i]).sum())) for ep, i in idx.items()}
    th, tm = min(cnt["pre"][0], cnt["post"][0]), min(cnt["pre"][1], cnt["post"][1])
    nf = min(MAX_FOLDS, th, tm)
    acc = {}
    for ep, i_ep in idx.items():
        hi, mi = i_ep[y[i_ep]], i_ep[~y[i_ep]]
        vals = [decode_bin_pooled(X[s], y[s], C, rng, n_repeats=N_REPEATS, n_folds=nf)
                for s in (np.concatenate([rng.choice(hi, th, replace=False), rng.choice(mi, tm, replace=False)])
                          for _ in range(N_SUBSAMPLE))]
        acc[ep] = float(np.nanmean(vals))
    return dict(acc_pre=acc["pre"], acc_post=acc["post"], delta=acc["post"] - acc["pre"], n_pre=len(idx["pre"]),
                pre_hit=cnt["pre"][0], pre_miss=cnt["pre"][1], post_hit=cnt["post"][0], post_miss=cnt["post"][1],
                matched_n_hit=th, matched_n_miss=tm)


def process(args):
    sid, subject_id, rg, lcat = args
    sys.path.insert(0, SCRIPTS_DIR)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, _active_trials_from_whisker_onset_for_curve, add_whole_brain_column, area_units,
        areas_with_enough_units, decode_bin_pooled, load_session_unit_spikes, prep_hitmiss_trials,
        select_fixed_c_pooled, sliding_bin_population_matrices,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st = pd.read_parquet(root / "metadata" / "sessions.parquet")
    tt = pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    tab = pd.read_csv(LT_TABLE).set_index("session_id")
    base = dict(session_id=sid, mouse_id=subject_id, reward_group=rg, learning_category=lcat,
                group=tab["group"].get(sid, None), lt=tab["lt_cohort"].get(sid, np.nan))
    trials = prep_hitmiss_trials(root, sid, st, tt)
    if trials is None:
        return [dict(base, skipped_reason="no usable trials")]
    y = trials["lick_flag"].to_numpy().astype(bool)
    t_dec = trials["start_time"].to_numpy()
    cw = _active_trials_from_whisker_onset_for_curve(sid, tt)
    cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()

    def valid(is_pre):
        return min(y[is_pre].sum(), (~y[is_pre]).sum(), y[~is_pre].sum(), (~y[~is_pre]).sum()) >= MIN_PER_CLASS

    splits = []
    cand = [k for k in range(1, len(cw_t), STEP) if valid(t_dec < cw_t[k])]
    if len(cand) > MAX_POSITIONS:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, MAX_POSITIONS).round().astype(int)]
    splits += [("placebo", k, t_dec < cw_t[k]) for k in cand]
    mid_pre = np.arange(len(y)) < len(y) // 2
    k_mid = int(np.searchsorted(cw_t, t_dec[len(y) // 2]))
    if valid(mid_pre):
        splits.append(("midpoint", k_mid, mid_pre))
    if not pd.isna(base["lt"]) and int(base["lt"]) < len(cw_t) and valid(t_dec < cw_t[int(base["lt"])]):
        splits.append(("lt", int(base["lt"]), t_dec < cw_t[int(base["lt"])]))
    if not splits:
        return [dict(base, skipped_reason="no valid split")]
    spikes = load_session_unit_spikes(root, sid)
    rows = []
    for area_col in ("whole_brain",):
        for area in areas_with_enough_units(sid, area_col, labels):
            units = area_units(sid, area_col, area, labels)
            mats = sliding_bin_population_matrices(spikes, units, t_dec, np.ones(len(y), bool),
                                                   [WINDOWS["sensory"], WINDOWS["baseline"]], dead_zone=WIDE_DEAD_ZONE)
            feats = {"sensory": mats[0], "baseline": mats[1], "sensory_minus_base": mats[0] - mats[1]}
            for w in WINDOW_NAMES:
                X = feats[w]
                C = select_fixed_c_pooled(X, y, rng, n_folds=MAX_FOLDS)
                for kind, k, is_pre in splits:
                    rows.append(dict(base, window=w, kind=kind, split_k=k, rel_k_lt=(k - base["lt"]) if not pd.isna(base["lt"]) else np.nan,
                                     rel_k_mid=k - k_mid, n_units=len(units), C=C, n_whisker_curve=len(cw_t),
                                     **matched(X, y, is_pre, C, rng, decode_bin_pooled), skipped_reason=None))
    return rows


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    sess = hitmiss_session_list(pd.read_parquet(resolve_dataset_dir("ssl_ephys") / "metadata" / "sessions.parquet"))
    sess = sess[sess.day_stage == "learning"]
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    todo = sess[~sess.session_id.isin(done)]
    print(f"[104] {len(todo)} sessions ({len(done)} done), {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init) as ex:
        futs = {ex.submit(process, (r.session_id, r.subject_id, r.reward_group, r.learning_category)): r.session_id
                for r in todo.itertuples()}
        for i, f in enumerate(as_completed(futs), 1):
            rows = f.result()
            new = pd.DataFrame(rows)
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            ok = new[new.skipped_reason.isna()] if "skipped_reason" in new else new
            print(f"[{i}/{len(futs)}] {futs[f]} {len(ok)} rows ({rows[0].get('skipped_reason') or ''}) elapsed {time.time() - t0:.0f}s",
                  flush=True)


if __name__ == "__main__":
    main()
