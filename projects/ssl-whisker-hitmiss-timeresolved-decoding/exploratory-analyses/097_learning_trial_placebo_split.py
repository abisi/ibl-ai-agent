"""Placebo-split null for the learning_trial (user request 2026-09-24: "Do
the placebo-split null to test the learning trial"). Asks whether the
pre/post change in hit/miss decodability at the REAL learning_trial is
larger than at arbitrary split points in the same session -- i.e. whether
the learning trial is special, rather than any early/late split showing a
difference because of drift, engagement or hit-rate change.

Same pipeline as `095_learning_trial_split_shiftnull.py` (learning-stage
whisker trials, terminal disengagement dropped, three windows, size-matched
balanced accuracy with n_hit*/n_miss* = per-class min over epochs, one C per
(session, area, window) from all trials, min 2 trials/class/epoch), but the
split point varies:
  - candidate split k = every STEP-th whisker trial (curve-aligned index,
    the index learning_trial counts in) whose split leaves >= 2 of each
    class in both epochs, capped at MAX_POSITIONS (evenly thinned), plus the
    real learning_trial. Decoded trials split by start_time of whisker
    trial k (learning trial itself -> post), as in 095.
  - statistic per split: delta = acc_post_matched - acc_pre_matched
    (raw matched balanced accuracy -- no per-split shift null: the
    placebo distribution itself is the null, and every split shares the
    session's drift).
  - REAL and placebo splits use the SAME reduced estimator (N_SUBSAMPLE x
    N_REPEATS) so they are directly comparable (095's numbers use more
    subsamples; not mixed here).

Per-session test (in 098): percentile of the real delta among placebo
deltas with |k - learning_trial| >= EXCLUDE_NEAR (splits right next to the
real one share most of its partition).

Usage (repo root): python 097_learning_trial_placebo_split.py [area_schemes] [max_sessions]
Output: 097_lt_placebo_<schemes>.parquet, one row per (session, area, window, split).
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
MIN_TRIALS_PER_CLASS_EPOCH = 2
MAX_FOLDS = 5
STEP = 3
MAX_POSITIONS = 60
N_SUBSAMPLE = 30
N_REPEATS = 3
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "24"))

AREA_SCHEMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["whole_brain"]
# Variants (added 2026-09-25, user: "Rerun the placebo test with new learning
# trials" + "Keep disengagement trials"):
#   SSL_LT_COLUMN=<col>, SSL_LT_TABLE=<csv in ssl-learning-trial-identification/artifacts>
#   SSL_NO_DISENGAGE=1 -> keep the terminal disengaged block
LT_COLUMN = os.environ.get("SSL_LT_COLUMN", "")
LT_TABLE = Path(__file__).resolve().parents[2] / "ssl-learning-trial-identification" / "artifacts" / os.environ.get(
    "SSL_LT_TABLE", "007_learning_trials_v2.csv")
NO_DISENGAGE = os.environ.get("SSL_NO_DISENGAGE", "0") == "1"
MAX_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT_DIR = Path(__file__).resolve().parent
_variant = (f"_lt-{LT_COLUMN}" if LT_COLUMN else "") + ("_nodisengagedrop" if NO_DISENGAGE else "")
OUT_PATH = OUT_DIR / f"097_lt_placebo{_variant}_{'-'.join(AREA_SCHEMES)}{'_smoke' if MAX_SESSIONS else ''}.parquet"
CFG = (f"windows={WINDOWS} dz={WIDE_DEAD_ZONE} step={STEP} max_positions={MAX_POSITIONS} n_subsample={N_SUBSAMPLE} "
       f"n_repeats={N_REPEATS} min_per_class={MIN_TRIALS_PER_CLASS_EPOCH} estimator=decode_bin_pooled L2-logistic KS4 "
       f"disengagement=095 rule")


def _worker_init():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def matched_delta(X, y, is_pre, C, rng, decode_bin_pooled):
    idx = {"pre": np.where(is_pre)[0], "post": np.where(~is_pre)[0]}
    cnt = {ep: (int(y[i].sum()), int((~y[i]).sum())) for ep, i in idx.items()}
    tgt_hit, tgt_miss = min(cnt["pre"][0], cnt["post"][0]), min(cnt["pre"][1], cnt["post"][1])
    n_folds = min(MAX_FOLDS, tgt_hit, tgt_miss)
    acc = {}
    for ep, i_ep in idx.items():
        hit_i, miss_i = i_ep[y[i_ep]], i_ep[~y[i_ep]]
        vals = []
        for _ in range(N_SUBSAMPLE):
            sel = np.concatenate([rng.choice(hit_i, tgt_hit, replace=False), rng.choice(miss_i, tgt_miss, replace=False)])
            vals.append(decode_bin_pooled(X[sel], y[sel], C, rng, n_repeats=N_REPEATS, n_folds=n_folds))
        acc[ep] = float(np.nanmean(vals))
    return dict(acc_pre=acc["pre"], acc_post=acc["post"], delta=acc["post"] - acc["pre"], matched_n_hit=tgt_hit,
                matched_n_miss=tgt_miss, n_pre=len(idx["pre"]), pre_hit=cnt["pre"][0], pre_miss=cnt["pre"][1],
                post_hit=cnt["post"][0], post_miss=cnt["post"][1])


def process_one_session(args: tuple) -> list[dict]:
    session_id, subject_id, reward_group, learning_category = args
    sys.path.insert(0, SCRIPTS_DIR)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, _active_trials_from_whisker_onset_for_curve, add_whole_brain_column, area_units,
        areas_with_enough_units, decode_bin_pooled, detect_terminal_disengagement, load_session_unit_spikes,
        prep_hitmiss_trials_learning_split, select_fixed_c_pooled, sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    rng = np.random.default_rng(zlib.crc32(session_id.encode()))

    trials, info = prep_hitmiss_trials_learning_split(dataset_root, session_id, sessions_tbl, trials_tbl)
    base = dict(session_id=session_id, mouse_id=subject_id, reward_group=reward_group,
                learning_category=learning_category, day_stage="learning", **info)
    if LT_COLUMN:
        from ssl_timeresolved_decoding import prep_hitmiss_trials
        tab = pd.read_csv(LT_TABLE).set_index("session_id")
        new_lt = tab[LT_COLUMN].get(session_id, np.nan)
        base.update(lt_column=LT_COLUMN, stored_learning_trial=info["learning_trial"], learning_trial=new_lt,
                    lt_source_new=tab["lt_lenient_source"].get(session_id, None) if "lt_lenient_source" in tab else None)
        if pd.isna(new_lt):
            return [dict(base, skipped_reason=f"no learning trial under {LT_COLUMN}")]
        info["learning_trial"] = new_lt
        if trials is None:
            trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
            if trials is None:
                return [dict(base, skipped_reason="no usable hit/miss trials")]
    if trials is None:
        return [dict(base, skipped_reason=info["lt_skip_reason"])]
    dis = detect_terminal_disengagement(session_id, trials_tbl)
    if dis["disengaged"] and not NO_DISENGAGE:
        trials = trials[trials["start_time"] < dis["t_cut"]].reset_index(drop=True)
    base.update(disengaged=dis["disengaged"])
    y = trials["lick_flag"].to_numpy().astype(bool)
    start_time = trials["start_time"].to_numpy()
    lt = int(info["learning_trial"])

    cw = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
    cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()

    def split_at(k):
        return start_time < cw_t[k]

    def valid(is_pre):
        return min(y[is_pre].sum(), (~y[is_pre]).sum(), y[~is_pre].sum(), (~y[~is_pre]).sum()) >= MIN_TRIALS_PER_CLASS_EPOCH

    if lt >= len(cw_t) or not valid(split_at(lt)):
        return [dict(base, skipped_reason="real split fails class-count minimum")]
    cand = [k for k in range(1, len(cw_t), STEP) if k != lt and valid(split_at(k))]
    if len(cand) > MAX_POSITIONS:
        cand = [cand[i] for i in np.linspace(0, len(cand) - 1, MAX_POSITIONS).round().astype(int)]
    positions = [lt] + cand

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    rows = []
    for area_col in AREA_SCHEMES:
        for area_value in areas_with_enough_units(session_id, area_col, area_labels):
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            mats = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, np.ones(len(y), bool),
                                                   [WINDOWS["sensory"], WINDOWS["baseline"]], dead_zone=WIDE_DEAD_ZONE)
            feats = {"sensory": mats[0], "baseline": mats[1], "sensory_minus_base": mats[0] - mats[1]}
            for wname in WINDOW_NAMES:
                t0 = time.time()
                X = feats[wname]
                C = select_fixed_c_pooled(X, y, rng, n_folds=MAX_FOLDS)
                for k in positions:
                    r = matched_delta(X, y, split_at(k), C, rng, decode_bin_pooled)
                    rows.append(dict(base, area_col=area_col, area_value=area_value, window=wname, n_units=len(unit_ids),
                                     C=C, split_k=k, rel_k=k - lt, is_real=(k == lt), n_positions=len(positions),
                                     n_whisker_curve=len(cw_t), **r, skipped_reason=None, cfg=CFG))
                rows[-1]["compute_s_window"] = time.time() - t0
    return rows


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sess = hitmiss_session_list(pd.read_parquet(dataset_root / "metadata" / "sessions.parquet"))
    sess = sess[sess.day_stage == "learning"].reset_index(drop=True)
    if MAX_SESSIONS:
        sess = sess.head(MAX_SESSIONS)
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    todo = sess[~sess.session_id.isin(done)]
    print(f"[097 {AREA_SCHEMES}] {len(todo)} sessions ({len(done)} done), {N_WORKERS} workers", flush=True)
    t0 = time.time()
    args = [(r.session_id, r.subject_id, r.reward_group, r.learning_category) for r in todo.itertuples()]
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        futs = {ex.submit(process_one_session, a): a[0] for a in args}
        for i, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            new = pd.DataFrame(rows)
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            ok = new[new.skipped_reason.isna()]
            if len(ok):
                parts = []
                for w, g in ok[ok.area_col == ok.area_col.iloc[0]].groupby("window"):
                    real = g[g.is_real].delta.mean()
                    parts.append(f"{w}: real {real:+.3f} pct {(g[~g.is_real].delta < real).mean():.2f}")
                msg = f"{ok.n_positions.iloc[0]} splits | " + " | ".join(parts)
            else:
                msg = rows[0]["skipped_reason"]
            print(f"[{i}/{len(args)}] {futs[fut]} {msg}  elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
