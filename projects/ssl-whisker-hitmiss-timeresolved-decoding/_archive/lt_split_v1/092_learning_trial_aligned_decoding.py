"""Trial-resolved hit/miss decoding aligned to each session's learning_trial
(user request 2026-09-24: "Show decoding results on the learning trial" ->
chose "decoding aligned to LT"). Learning stage, sensory window 5-50ms.

Per (session, area):
  - features: mean rate per unit in SENSORY_WINDOW (same as 090).
  - per-trial held-out decoder output: `N_REPEATS` x stratified K-fold
    (K = min(5, minority count)) over ALL of the session's decoded whisker
    trials; `p_correct` = held-out predicted probability of the trial's TRUE
    class, averaged over repeats. Classifier = 090/024's StandardScaler +
    L2-logistic but with class_weight='balanced', so a trial-level output is
    not dominated by the (time-varying) hit/miss base rate -- chance is
    ~0.5 for both classes. C from `select_fixed_c_pooled` on all trials.
  - per-trial null: same procedure with labels shuffled, `N_SHUF` times ->
    `p_correct_null` (mean) per trial.
  - trial position: `curve_idx` (index in the curve-aligned whisker trials,
    the same index `learning_trial` counts in) and `rel_idx = curve_idx -
    learning_trial`; behavioral p_mean/p_chance attached from the curve.

Caveat (stated in figures' caption): one decoder per session is fit on all
trials, so the decision axis is dominated by whichever epoch has more
trials (usually post). A trial-level change at the learning trial therefore
means pre trials are less separable ALONG THAT AXIS -- complementary to
090's separately-fit, size-matched pre/post decoders.

Usage (repo root): python 092_learning_trial_aligned_decoding.py [area_schemes] [max_sessions]
Output: 092_lt_aligned_trials_<schemes>.parquet, one row per (session, area, trial).
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

SENSORY_WINDOW = (0.005, 0.050)
WIDE_DEAD_ZONE = (-0.010, 0.005)
MAX_FOLDS = 5
MIN_PER_CLASS = 3
N_REPEATS = 20
N_SHUF = 20
N_REPEATS_SHUF = 3
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "16"))

AREA_SCHEMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["whole_brain"]
MAX_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / f"092_lt_aligned_trials_{'-'.join(AREA_SCHEMES)}{'_smoke' if MAX_SESSIONS else ''}.parquet"


def _worker_init():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def heldout_p_correct(X, y, C, rng, n_repeats, n_folds):
    """Mean held-out P(true class) per trial over `n_repeats` stratified
    K-fold partitions (each trial held out exactly once per repeat)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out = np.zeros(len(y))
    for _ in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(rng.integers(0, 2**31 - 1)))
        for tr, te in skf.split(X, y):
            clf = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear", C=C,
                                                                     class_weight="balanced", max_iter=1000))
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[te])
            cls_col = {c: i for i, c in enumerate(clf.classes_)}
            out[te] += proba[np.arange(len(te)), [cls_col[v] for v in y[te]]]
    return out / n_repeats


def process_one_session(args: tuple) -> list[dict]:
    session_id, subject_id, reward_group, learning_category = args
    sys.path.insert(0, SCRIPTS_DIR)
    from axel_bisi_paths import axel_bisi_path
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import session_training_day
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, _active_trials_from_whisker_onset_for_curve, add_whole_brain_column, area_units,
        areas_with_enough_units, load_session_unit_spikes, load_whisker_curve_row,
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
    if trials is None:
        return [dict(base, skipped_reason=info["lt_skip_reason"])]
    y = trials["lick_flag"].to_numpy().astype(bool)
    n_hit, n_miss = int(y.sum()), int((~y).sum())
    if min(n_hit, n_miss) < MIN_PER_CLASS:
        return [dict(base, skipped_reason=f"session class counts {n_hit}h/{n_miss}m (< {MIN_PER_CLASS})")]
    n_folds = min(MAX_FOLDS, n_hit, n_miss)

    cw = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
    cw = cw[cw["trial_type"] == "whisker_trial"].reset_index(drop=True)
    idx_by_time = pd.Series(np.arange(len(cw)), index=cw["start_time"].to_numpy())
    curve_idx = idx_by_time.reindex(trials["start_time"].to_numpy()).to_numpy()
    desc = sessions_tbl.loc[sessions_tbl.session_id == session_id, "session_description"].iloc[0]
    curve = load_whisker_curve_row(axel_bisi_path("combined_results_ks4"), subject_id, session_training_day(desc))
    p_mean, p_chance = np.asarray(curve["p_mean"]), np.asarray(curve["p_chance"])
    ci = np.nan_to_num(curve_idx, nan=-1).astype(int)
    lt = int(info["learning_trial"])

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = trials["start_time"].to_numpy()
    rows = []
    for area_col in AREA_SCHEMES:
        for area_value in areas_with_enough_units(session_id, area_col, area_labels):
            t0 = time.time()
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, np.ones(len(y), bool), [SENSORY_WINDOW],
                                                dead_zone=WIDE_DEAD_ZONE)[0]
            C = select_fixed_c_pooled(X, y, rng, n_folds=n_folds)
            pc = heldout_p_correct(X, y, C, rng, N_REPEATS, n_folds)
            pc_null = np.mean([heldout_p_correct(X, rng.permutation(y), C, rng, N_REPEATS_SHUF, n_folds)
                               for _ in range(N_SHUF)], axis=0)
            dt = time.time() - t0
            for k in range(len(y)):
                rows.append(dict(base, area_col=area_col, area_value=area_value, n_units=len(unit_ids), C=C, n_folds=n_folds,
                                 trial_id=trials["trial_id"].iloc[k] if "trial_id" in trials else None,
                                 start_time=start_time[k], curve_idx=curve_idx[k],
                                 rel_idx=curve_idx[k] - lt if not np.isnan(curve_idx[k]) else np.nan,
                                 lt_epoch=trials["lt_epoch"].iloc[k], lick_flag=bool(y[k]),
                                 p_correct=pc[k], p_correct_null=pc_null[k],
                                 behav_p_mean=p_mean[ci[k]] if ci[k] >= 0 else np.nan,
                                 behav_p_chance=p_chance[ci[k]] if ci[k] >= 0 else np.nan,
                                 compute_s=dt, skipped_reason=None,
                                 cfg=f"window={SENSORY_WINDOW} dz={WIDE_DEAD_ZONE} repeats={N_REPEATS} shuf={N_SHUF}x{N_REPEATS_SHUF} "
                                     f"L2-logistic class_weight=balanced KS4"))
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
    print(f"[092 {AREA_SCHEMES}] {len(todo)} sessions ({len(done)} done), {N_WORKERS} workers", flush=True)
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
            msg = (f"{len(ok)} trial-rows, mean p_correct {ok.p_correct.mean():.3f} vs null {ok.p_correct_null.mean():.3f}"
                   if len(ok) else rows[0]["skipped_reason"])
            print(f"[{i}/{len(args)}] {futs[fut]} {msg}  elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
