"""Redo of `090_learning_trial_split_sensory_decode.py` (user request
2026-09-24: "Redo with drift linear shift null. Do both baseline-corrected
and non-corrected and baseline as windows for decoding. Drop last run of
misses at the end if it is clearly a stated state."). 090's outputs are kept
untouched for comparison.

Changes vs 090 (everything else identical -- learning-stage whisker trials,
pre/post split at learning_trial by start_time, min 2 trials/class/epoch,
size-matched primary comparison with n_hit*=min(pre_hit,post_hit),
n_miss*=min(pre_miss,post_miss), unmatched secondary, one C per
(session, area, window) from all trials, `decode_bin_pooled`):

1. NULL = linear shift null at SESSION level (this project's convention for
   hit/miss, `024_master_sweep.py` / `linear_shift_null_curves_pooled`:
   non-wrapping shift of 10-50% of the session, random direction). The
   whole label sequence is shifted against the neural trials; each
   (neural trial, shifted label) pair keeps the NEURAL trial's epoch; the
   exact matched/unmatched per-epoch decoding is then re-run on the shifted
   pairs (target counts recomputed per shift). Shifts are drawn until
   N_SHIFT VALID ones are found (both epochs keep >= 2 of each class under
   the shifted labels, the same condition the real split passes; up to
   MAX_SHIFT_ATTEMPTS draws) -- a first smoke test with a fixed 50 draws
   left only 5/50 valid for a 9-trial pre epoch. Preserves slow drift in both labels and neural
   activity, breaks their trial-by-trial correspondence. Replaces 090's
   per-epoch label shuffle, which does not control slow drift.
2. WINDOWS (features = mean rate per unit):
     sensory             5 to 50 ms (dead zone -10/+5 ms, as 024/090)
     baseline            -200 to -10 ms (pre-stimulus; same as 034's
                         BASELINE_WINDOW)
     sensory_minus_base  sensory rate - baseline rate, per unit per trial
3. DISENGAGEMENT: trials at/after `detect_terminal_disengagement`'s t_cut
   dropped (trailing block after the session's last lick on any trial type
   with >=5 whisker AND >=3 auditory trials) -- before the epoch split,
   so it only shortens 'post'.

Usage (repo root): python 095_learning_trial_split_shiftnull.py [area_schemes] [max_sessions]
Output: 095_lt_split_shiftnull_<schemes>.parquet, one row per (session, area, window).
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
N_SUBSAMPLE = 100
N_REPEATS_PER_SUBSAMPLE = 5
N_REPEATS_FULL = 20
N_SHIFT = 50
MAX_SHIFT_ATTEMPTS = 1000
N_SUBSAMPLE_SHIFT = 5
N_REPEATS_SHIFT = 2
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
N_WORKERS = int(os.environ.get("SSL_DECODE_N_WORKERS", "16"))

AREA_SCHEMES = sys.argv[1].split(",") if len(sys.argv) > 1 else ["whole_brain"]
MAX_SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else None
OUT_DIR = Path(__file__).resolve().parent
# Variants (added 2026-09-24, user: compare with the split-halves scenario;
# unsure whether to remove disengagement yet):
#   SSL_SPLIT_MODE=half   -> split at the median of the (post-drop) decoded
#                           whisker trials instead of learning_trial; same
#                           sessions (learners with a valid LT) for pairing.
#   SSL_NO_DISENGAGE=1    -> keep the terminal disengaged block.
#   SSL_SESSION_IDS=a,b   -> restrict to these sessions.
SPLIT_MODE = os.environ.get("SSL_SPLIT_MODE", "lt")
NO_DISENGAGE = os.environ.get("SSL_NO_DISENGAGE", "0") == "1"
SESSION_IDS = [s for s in os.environ.get("SSL_SESSION_IDS", "").split(",") if s]
#   SSL_LT_COLUMN=<col>  -> take the learning trial from column <col> of
#                           ssl-learning-trial-identification/artifacts/
#                           006_learning_trials_final.csv (same whisker-trial
#                           index as the stored LT; NaN -> session skipped)
#                           instead of the stored H5 value (added 2026-09-25).
LT_COLUMN = os.environ.get("SSL_LT_COLUMN", "")
LT_TABLE = Path(__file__).resolve().parents[2] / "ssl-learning-trial-identification" / "artifacts" / os.environ.get(
    "SSL_LT_TABLE", "006_learning_trials_final.csv")  # 007_learning_trials_v2.csv for lt_lenient_clean (2026-09-25)
assert SPLIT_MODE in ("lt", "half")
_variant = (("" if SPLIT_MODE == "lt" else "_halfsplit") + ("_nodisengagedrop" if NO_DISENGAGE else "")
            + (f"_lt-{LT_COLUMN}" if LT_COLUMN else ""))
OUT_PATH = OUT_DIR / (f"095_lt_split_shiftnull{_variant}_{'-'.join(AREA_SCHEMES)}"
                      f"{'_smoke' if MAX_SESSIONS else ''}{'_subset' if SESSION_IDS else ''}.parquet")
CONFIG = dict(windows=str(WINDOWS), dead_zone=list(WIDE_DEAD_ZONE), min_trials_per_class_epoch=MIN_TRIALS_PER_CLASS_EPOCH,
              n_subsample=N_SUBSAMPLE, n_repeats_per_subsample=N_REPEATS_PER_SUBSAMPLE, n_repeats_full=N_REPEATS_FULL,
              null="session-level linear shift (non-wrapping)", n_shift=N_SHIFT, max_shift_attempts=MAX_SHIFT_ATTEMPTS, n_subsample_shift=N_SUBSAMPLE_SHIFT,
              n_repeats_shift=N_REPEATS_SHIFT, shift_frac=[MIN_SHIFT_FRAC, MAX_SHIFT_FRAC],
              disengagement="trailing block after last lick with >=5 whisker & >=3 auditory trials dropped",
              estimator="decode_bin_pooled", classifier="StandardScaler+L2-logistic(liblinear)",
              c_selection="select_fixed_c_pooled on all session trials, per window", spike_sorting="KS4 (ssl_ephys)",
              split_mode=SPLIT_MODE, no_disengage_drop=NO_DISENGAGE, lt_column=LT_COLUMN or 'stored')


def _worker_init():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def epoch_scores(X, y, is_pre, C, rng, decode_bin_pooled, n_sub, n_rep_sub, n_rep_full):
    """Size-matched and unmatched balanced accuracy for both epochs, or None
    if an epoch has < MIN_TRIALS_PER_CLASS_EPOCH of a class."""
    idx = {"pre": np.where(is_pre)[0], "post": np.where(~is_pre)[0]}
    counts = {ep: (int(y[i].sum()), int((~y[i]).sum())) for ep, i in idx.items()}
    if min(min(c) for c in counts.values()) < MIN_TRIALS_PER_CLASS_EPOCH:
        return None
    tgt_hit = min(counts["pre"][0], counts["post"][0])
    tgt_miss = min(counts["pre"][1], counts["post"][1])
    n_folds = min(MAX_FOLDS, tgt_hit, tgt_miss)
    out = dict(matched_n_hit=tgt_hit, matched_n_miss=tgt_miss, n_folds_matched=n_folds)
    for ep, i_ep in idx.items():
        hit_i, miss_i = i_ep[y[i_ep]], i_ep[~y[i_ep]]
        accs = []
        for _ in range(n_sub):
            sel = np.concatenate([rng.choice(hit_i, tgt_hit, replace=False), rng.choice(miss_i, tgt_miss, replace=False)])
            accs.append(decode_bin_pooled(X[sel], y[sel], C, rng, n_repeats=n_rep_sub, n_folds=n_folds))
        ye = y[i_ep]
        nf = min(MAX_FOLDS, int(ye.sum()), int((~ye).sum()))
        out[f"acc_{ep}_matched"] = float(np.nanmean(accs))
        out[f"acc_{ep}_full"] = decode_bin_pooled(X[i_ep], ye, C, rng, n_repeats=n_rep_full, n_folds=nf)
    return out


def process_one_session(args: tuple) -> list[dict]:
    session_id, subject_id, reward_group, learning_category = args
    sys.path.insert(0, SCRIPTS_DIR)
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, areas_with_enough_units, decode_bin_pooled,
        detect_terminal_disengagement, load_session_unit_spikes, prep_hitmiss_trials_learning_split,
        select_fixed_c_pooled, sliding_bin_population_matrices,
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
        # Override the stored learning trial. The stored-LT prep still has to
        # succeed for the trial set itself; sessions it rejected only for lacking
        # a stored LT (non-learners) are re-prepped here via the no-LT path.
        from ssl_timeresolved_decoding import _active_trials_from_whisker_onset_for_curve, prep_hitmiss_trials
        tab = pd.read_csv(LT_TABLE).set_index("session_id")
        new_lt = tab[LT_COLUMN].get(session_id, np.nan)
        base.update(lt_column=LT_COLUMN, stored_learning_trial=info["learning_trial"], learning_trial=new_lt,
                    lt_source_new=tab.get("lt_recommended_broad_source", pd.Series(dtype=object)).get(session_id, None),
                    L6_category=tab["L6_category"].get(session_id, None))
        if pd.isna(new_lt):
            return [dict(base, skipped_reason=f"no learning trial under {LT_COLUMN}")]
        if trials is None:
            trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
            if trials is None:
                return [dict(base, skipped_reason="no usable hit/miss trials")]
        cw = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
        cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()
        trials["lt_epoch"] = np.where(trials["start_time"].to_numpy() >= cw_t[int(new_lt)], "post", "pre")
    if trials is None and SPLIT_MODE == "half":
        # The midpoint split needs no learning trial (fix 2026-09-25: sessions without a
        # stored LT -- stored non-learners, curve-length mismatch -- were skipped, but they
        # are exactly the non-learners the halves comparison needs).
        from ssl_timeresolved_decoding import prep_hitmiss_trials
        trials = prep_hitmiss_trials(dataset_root, session_id, sessions_tbl, trials_tbl)
    if trials is None:
        return [dict(base, skipped_reason=info["lt_skip_reason"])]
    dis = detect_terminal_disengagement(session_id, trials_tbl)
    n_before = len(trials)
    if dis["disengaged"] and not NO_DISENGAGE:
        trials = trials[trials["start_time"] < dis["t_cut"]].reset_index(drop=True)
    if SPLIT_MODE == "half":
        trials["lt_epoch"] = np.where(np.arange(len(trials)) < len(trials) // 2, "pre", "post")
    base.update(disengaged=dis["disengaged"], t_disengage=dis["t_cut"], n_whisker_dropped_disengaged=n_before - len(trials),
                n_auditory_in_disengaged_run=dis["n_auditory_in_run"])

    y = trials["lick_flag"].to_numpy().astype(bool)
    is_pre = (trials["lt_epoch"] == "pre").to_numpy()
    base.update(n_trials=len(y), n_pre=int(is_pre.sum()), n_post=int((~is_pre).sum()),
                pre_hit=int(y[is_pre].sum()), pre_miss=int((~y[is_pre]).sum()),
                post_hit=int(y[~is_pre].sum()), post_miss=int((~y[~is_pre]).sum()))
    if min(base["pre_hit"], base["pre_miss"], base["post_hit"], base["post_miss"]) < MIN_TRIALS_PER_CLASS_EPOCH:
        return [dict(base, skipped_reason=f"class counts pre {base['pre_hit']}h/{base['pre_miss']}m post "
                                           f"{base['post_hit']}h/{base['post_miss']}m (< {MIN_TRIALS_PER_CLASS_EPOCH})")]

    n = len(y)
    min_shift = max(1, int(MIN_SHIFT_FRAC * n))
    max_shift = max(min_shift, int(MAX_SHIFT_FRAC * n))
    # Candidate shifts drawn in a fixed random order; keep the first N_SHIFT
    # VALID ones (both epochs keep >= MIN_TRIALS_PER_CLASS_EPOCH of each class
    # under the shifted labels -- the same condition the real split had to
    # pass). Validity depends only on labels/epochs, so this one list is
    # reused for every area and window of the session.
    shifts, n_tried = [], 0
    while len(shifts) < N_SHIFT and n_tried < MAX_SHIFT_ATTEMPTS:
        k, fwd = int(rng.integers(min_shift, max_shift + 1)), bool(rng.random() < 0.5)
        n_tried += 1
        ys, ps = (y[k:], is_pre[: n - k]) if fwd else (y[: n - k], is_pre[k:])
        if min(ys[ps].sum(), (~ys[ps]).sum(), ys[~ps].sum(), (~ys[~ps]).sum()) >= MIN_TRIALS_PER_CLASS_EPOCH:
            shifts.append((k, fwd))
    base.update(n_valid_shifts=len(shifts), n_shift_attempts=n_tried)

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = trials["start_time"].to_numpy()
    is_whisker = np.ones(n, dtype=bool)
    rows = []
    for area_col in AREA_SCHEMES:
        for area_value in areas_with_enough_units(session_id, area_col, area_labels):
            unit_ids = area_units(session_id, area_col, area_value, area_labels)
            mats = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker,
                                                   [WINDOWS["sensory"], WINDOWS["baseline"]], dead_zone=WIDE_DEAD_ZONE)
            feats = {"sensory": mats[0], "baseline": mats[1], "sensory_minus_base": mats[0] - mats[1]}
            for wname in WINDOW_NAMES:
                t0 = time.time()
                X = feats[wname]
                C = select_fixed_c_pooled(X, y, rng, n_folds=MAX_FOLDS)
                real = epoch_scores(X, y, is_pre, C, rng, decode_bin_pooled, N_SUBSAMPLE, N_REPEATS_PER_SUBSAMPLE, N_REPEATS_FULL)
                null = {k: [] for k in ("pre_matched", "post_matched", "pre_full", "post_full")}
                for k, forward in shifts:
                    if forward:
                        Xs, ys, ps = X[: n - k], y[k:], is_pre[: n - k]
                    else:
                        Xs, ys, ps = X[k:], y[: n - k], is_pre[k:]
                    r = epoch_scores(Xs, ys, ps, C, rng, decode_bin_pooled, N_SUBSAMPLE_SHIFT, N_REPEATS_SHIFT, N_REPEATS_SHIFT)
                    for key in null:
                        null[key].append(np.nan if r is None else r[f"acc_{key}"])
                row = dict(base, area_col=area_col, area_value=area_value, window=wname, n_units=len(unit_ids),
                           cluster_ids=[int(c) for c in unit_ids], C=C, **real, skipped_reason=None)
                for key, vals in null.items():
                    vals = np.asarray(vals, dtype=float)
                    real_v = real[f"acc_{key}"]
                    row[f"shiftnull_{key}"] = vals.tolist()
                    row[f"nullmean_{key}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
                    row[f"n_valid_shifts_{key}"] = int(np.isfinite(vals).sum())
                    row[f"p_{key}"] = float((1 + np.nansum(vals >= real_v)) / (1 + np.isfinite(vals).sum()))
                row["delta_matched"] = row["acc_post_matched"] - row["acc_pre_matched"]
                row["delta_matched_nullcorr"] = ((row["acc_post_matched"] - row["nullmean_post_matched"])
                                                 - (row["acc_pre_matched"] - row["nullmean_pre_matched"]))
                row["compute_s"] = time.time() - t0
                row.update({f"cfg_{k}": v for k, v in CONFIG.items()})
                rows.append(row)
    return rows


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sess = hitmiss_session_list(pd.read_parquet(dataset_root / "metadata" / "sessions.parquet"))
    sess = sess[sess.day_stage == "learning"].reset_index(drop=True)
    if SESSION_IDS:
        sess = sess[sess.session_id.isin(SESSION_IDS)].reset_index(drop=True)
    if MAX_SESSIONS:
        sess = sess.head(MAX_SESSIONS)
    done = set(pd.read_parquet(OUT_PATH, columns=["session_id"]).session_id) if OUT_PATH.exists() else set()
    todo = sess[~sess.session_id.isin(done)]
    print(f"[095 {AREA_SCHEMES}] {len(todo)} sessions ({len(done)} done), {N_WORKERS} workers", flush=True)
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
                s = ok[ok.area_col == ok.area_col.iloc[0]].groupby("window")[["acc_pre_matched", "nullmean_pre_matched",
                                                                            "acc_post_matched", "nullmean_post_matched"]].mean()
                msg = " | ".join(f"{w}: pre {r.acc_pre_matched:.2f}/{r.nullmean_pre_matched:.2f} post {r.acc_post_matched:.2f}/"
                                 f"{r.nullmean_post_matched:.2f}" for w, r in s.iterrows())
                msg += f" (dropped {ok.n_whisker_dropped_disengaged.iloc[0]} disengaged, {ok.compute_s.sum():.0f}s)"
            else:
                msg = rows[0]["skipped_reason"]
            print(f"[{i}/{len(args)}] {futs[fut]} {msg}  elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
