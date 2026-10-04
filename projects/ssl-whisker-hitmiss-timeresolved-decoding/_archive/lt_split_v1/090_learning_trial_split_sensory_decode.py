"""Hit/miss (lick_flag) decoding in the sensory window, pre vs post each
session's `learning_trial` (user request 2026-09-24; replaces the
session-half split for this question). Learning-stage (day 0) whisker
trials only. Split + validation: `089_learning_trial_sanity.py`,
`ssl_timeresolved_decoding.prep_hitmiss_trials_learning_split`.

Design (locked with user 2026-09-24):
  - single window: SENSORY_WINDOW (5-50ms after start_time), mean rate per
    unit, wide dead zone (-10/+5ms, same as 024 -- does not overlap this
    window anyway). Not time-resolved.
  - learning_trial kept as stored for every learner, including the
    hardcoded 10s (floor10 / R- fallback10); `lt_source` saved per row and
    shown in every figure. Non-learners (NaN) and curve/trial length
    mismatches (MH038) skipped with a reason row.
  - min trials per class per epoch: 2 (generous; project default is 5).
  - imbalance: the pre epoch is usually far smaller than post (median R+
    13 vs 79 trials). Primary comparison is SIZE-MATCHED: target counts
    n_hit* = min(pre_hit, post_hit), n_miss* = min(pre_miss, post_miss);
    each epoch is subsampled N_SUBSAMPLE times to exactly (n_hit*, n_miss*)
    and its accuracies averaged. In the usual case (pre smaller in both
    classes) this is exactly "all pre trials vs post subsampled to pre's
    counts" (user's choice); the elementwise min generalizes it to late
    learning trials where post is the smaller epoch in a class. Both
    epochs use the same n_folds = min(5, n_hit*, n_miss*). Unmatched
    per-epoch scores (all trials of the epoch) kept as secondary readouts.
  - one C per (session, area), chosen by `select_fixed_c_pooled` on ALL of
    the session's trials (pre+post) and reused for both epochs --
    symmetric across epochs, and avoids selecting C on ~10-trial pre sets.
    Same "C selected on data that is also scored" compromise as 024.
  - estimator: `decode_bin_pooled` (pooled held-out predictions,
    resample-on-missing-class), balanced accuracy.
  - per-epoch label-shuffle null (not shift-null: a linear shift on a
    ~10-trial epoch is degenerate). Matched: one shuffled-label decode per
    subsample (same subsamples as the real score) -> N_SUBSAMPLE null
    values per epoch. Unmatched: N_NULL shuffles per epoch.

Usage: python 090_learning_trial_split_sensory_decode.py [area_schemes] [max_sessions]
  area_schemes comma-separated (default whole_brain); max_sessions for smoke tests.
Run from repo root. SSL_DECODE_N_WORKERS sets pool size (default 16).
Output: 090_lt_split_results_<schemes>.parquet (one row per session x area).
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
MIN_TRIALS_PER_CLASS_EPOCH = 2
MAX_FOLDS = 5
N_SUBSAMPLE = 100
N_REPEATS_PER_SUBSAMPLE = 5
N_REPEATS_FULL = 20
N_NULL = 100
N_REPEATS_NULL = 2
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "16"))

AREA_SCHEMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["whole_brain"]
MAX_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT_DIR = Path(__file__).resolve().parent
OUT_PATH = OUT_DIR / f"090_lt_split_results_{'-'.join(AREA_SCHEMES)}{'_smoke' if MAX_SESSIONS else ''}.parquet"
CONFIG = dict(sensory_window=list(SENSORY_WINDOW), dead_zone=list(WIDE_DEAD_ZONE),
              min_trials_per_class_epoch=MIN_TRIALS_PER_CLASS_EPOCH, max_folds=MAX_FOLDS,
              n_subsample=N_SUBSAMPLE, n_repeats_per_subsample=N_REPEATS_PER_SUBSAMPLE,
              n_repeats_full=N_REPEATS_FULL, n_null=N_NULL, n_repeats_null=N_REPEATS_NULL,
              estimator="decode_bin_pooled", classifier="StandardScaler+L2-logistic(liblinear)",
              c_selection="select_fixed_c_pooled on all session trials", spike_sorting="KS4 (ssl_ephys)",
              reward_group_source="mouse_reference.parquet (joint_mouse_reference_weight.xlsx)")


def _worker_init():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def class_counts(y: np.ndarray) -> tuple[int, int]:
    return int(y.sum()), int((~y).sum())


def process_one_session(args: tuple) -> list[dict]:
    session_id, subject_id, reward_group, learning_category = args
    sys.path.insert(0, SCRIPTS_DIR)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, areas_with_enough_units, decode_bin_pooled,
        load_session_unit_spikes, prep_hitmiss_trials_learning_split, select_fixed_c_pooled,
        sliding_bin_population_matrices,
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

    y_all = trials["lick_flag"].to_numpy().astype(bool)
    is_pre = (trials["lt_epoch"] == "pre").to_numpy()
    y_pre, y_post = y_all[is_pre], y_all[~is_pre]
    pre_hit, pre_miss = class_counts(y_pre)
    post_hit, post_miss = class_counts(y_post)
    base.update(n_trials=len(y_all), n_pre=int(is_pre.sum()), n_post=int((~is_pre).sum()),
                pre_hit=pre_hit, pre_miss=pre_miss, post_hit=post_hit, post_miss=post_miss,
                pre_trial_ids=trials.loc[is_pre, "trial_id"].tolist() if "trial_id" in trials else None)
    if min(pre_hit, pre_miss, post_hit, post_miss) < MIN_TRIALS_PER_CLASS_EPOCH:
        return [dict(base, skipped_reason=f"class counts pre {pre_hit}h/{pre_miss}m post {post_hit}h/{post_miss}m "
                                           f"(< {MIN_TRIALS_PER_CLASS_EPOCH} per class per epoch)")]

    tgt_hit, tgt_miss = min(pre_hit, post_hit), min(pre_miss, post_miss)
    n_folds = min(MAX_FOLDS, tgt_hit, tgt_miss)
    base.update(matched_n_hit=tgt_hit, matched_n_miss=tgt_miss, n_folds_matched=n_folds)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = trials["start_time"].to_numpy()
    is_whisker = np.ones(len(trials), dtype=bool)
    epoch_idx = {"pre": np.where(is_pre)[0], "post": np.where(~is_pre)[0]}

    def pval(real, null):
        null = np.asarray(null)
        return float((1 + np.sum(null >= real)) / (1 + np.sum(~np.isnan(null))))

    rows = []
    for area_col in AREA_SCHEMES:
        for area_value in areas_with_enough_units(session_id, area_col, area_labels):
            t0 = time.time()
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW],
                                                dead_zone=WIDE_DEAD_ZONE)[0]
            C = select_fixed_c_pooled(X, y_all, rng, n_folds=MAX_FOLDS)
            res = {}
            for ep, idx_ep in epoch_idx.items():
                hit_idx, miss_idx = idx_ep[y_all[idx_ep]], idx_ep[~y_all[idx_ep]]
                real, null = np.full(N_SUBSAMPLE, np.nan), np.full(N_SUBSAMPLE, np.nan)
                for k in range(N_SUBSAMPLE):
                    idx = np.concatenate([rng.choice(hit_idx, tgt_hit, replace=False),
                                          rng.choice(miss_idx, tgt_miss, replace=False)])
                    Xs, ys = X[idx], y_all[idx]
                    real[k] = decode_bin_pooled(Xs, ys, C, rng, n_repeats=N_REPEATS_PER_SUBSAMPLE, n_folds=n_folds)
                    null[k] = decode_bin_pooled(Xs, rng.permutation(ys), C, rng, n_repeats=N_REPEATS_NULL, n_folds=n_folds)
                acc_m = float(np.nanmean(real))
                ye = y_all[idx_ep]
                nf_full = min(MAX_FOLDS, int(ye.sum()), int((~ye).sum()))
                acc_f = decode_bin_pooled(X[idx_ep], ye, C, rng, n_repeats=N_REPEATS_FULL, n_folds=nf_full)
                null_f = np.array([decode_bin_pooled(X[idx_ep], rng.permutation(ye), C, rng, n_repeats=N_REPEATS_NULL,
                                                     n_folds=nf_full) for _ in range(N_NULL)])
                res.update({
                    f"acc_{ep}_matched": acc_m, f"{ep}_matched_subsample_acc": real.tolist(),
                    f"null_{ep}_matched": null.tolist(), f"nullmean_{ep}_matched": float(np.nanmean(null)),
                    f"p_{ep}_matched": pval(acc_m, null),
                    f"acc_{ep}_full": acc_f, f"null_{ep}_full": null_f.tolist(),
                    f"nullmean_{ep}_full": float(np.nanmean(null_f)), f"p_{ep}_full": pval(acc_f, null_f),
                    f"n_folds_{ep}_full": nf_full,
                })
            rows.append(dict(
                base, area_col=area_col, area_value=area_value, n_units=len(unit_ids),
                cluster_ids=[int(c) for c in unit_ids], C=C, **res,
                delta_matched=res["acc_post_matched"] - res["acc_pre_matched"],
                delta_matched_nullcorr=(res["acc_post_matched"] - res["nullmean_post_matched"])
                - (res["acc_pre_matched"] - res["nullmean_pre_matched"]),
                delta_full=res["acc_post_full"] - res["acc_pre_full"],
                compute_s=time.time() - t0, skipped_reason=None, **{f"cfg_{k}": v for k, v in CONFIG.items()},
            ))
    return rows


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    sess = hitmiss_session_list(sessions_tbl)
    sess = sess[sess.day_stage == "learning"].reset_index(drop=True)
    if MAX_SESSIONS:
        sess = sess.head(MAX_SESSIONS)
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    todo = sess[~sess.session_id.isin(done)]
    print(f"[090 {AREA_SCHEMES}] {len(todo)} sessions to run ({len(done)} done), {N_WORKERS} workers", flush=True)

    t0 = time.time()
    args = [(r.session_id, r.subject_id, r.reward_group, r.learning_category) for r in todo.itertuples()]
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        futs = {ex.submit(process_one_session, a): a[0] for a in args}
        for i, fut in enumerate(as_completed(futs), 1):
            rows = fut.result()
            new = pd.DataFrame(rows)
            out = pd.concat([pd.read_parquet(OUT_PATH), new], ignore_index=True) if OUT_PATH.exists() else new
            out.to_parquet(OUT_PATH, index=False)
            ok = [r for r in rows if r.get("skipped_reason") is None]
            msg = (f"pre_m {ok[0]['acc_pre_matched']:.3f} post_m {ok[0]['acc_post_matched']:.3f} "
                   f"({sum(r['compute_s'] for r in ok):.0f}s, {len(ok)} areas)") if ok else rows[0]["skipped_reason"]
            print(f"[{i}/{len(args)}] {futs[fut]} {msg}  elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
