"""Confirmatory run: session-level decoding (modality, response) across the
full ephys cohort, both day-stages, at the locked parameters from
question.md (n_runs=5, n_pseudo=50). Estimated ~5.5h. Imposter-session
sources for a given session are every other session in the same day-stage
bucket (learning vs expert), per question.md's decoding-null design.
Writes incrementally after each (session, target).
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

import numpy as np
import pandas as pd

from ssl_bwm_trial_prep import list_whisker_training_ephys_sessions, prep_session
from ssl_bwm_decoding import population_design_matrix, nested_cv_balanced_accuracy, imposter_pvalue
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATASET_ROOT = resolve_dataset_dir("ssl_ephys")
EVOKED_WINDOW = (0.005, 0.035)
BASELINE_WINDOW = (-0.200, -0.010)
N_RUNS = 5
N_PSEUDO = 50
RNG_SEED = 20260819
QC_VALUES = ("good", "mua")  # widened from good-only, user decision 2026-08-19

OUT_PATH = Path(__file__).resolve().parent / "decoding_results.parquet"
TARGETS = {"modality": EVOKED_WINDOW, "response": BASELINE_WINDOW}


def candidate_arrays(dataset_root, session_id, sessions_tbl, trials_tbl):
    prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
    if prepped is None:
        return None
    candidate = prepped["trials"][prepped["trials"]["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
    if len(candidate) < 8:
        return None
    is_whisker = (candidate["trial_type"] == "whisker_trial").to_numpy()
    response = candidate["lick_flag"].to_numpy().astype(bool)
    start_time = candidate["start_time"].to_numpy()
    return {"candidate": candidate, "is_whisker": is_whisker, "response": response, "start_time": start_time}


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")
    units_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "units.parquet")
    sessions = list_whisker_training_ephys_sessions(sessions_tbl)

    print(f"Preloading candidate arrays for {len(sessions)} sessions...", flush=True)
    prepped_by_session = {}
    for sid, day_stage in sessions:
        r = candidate_arrays(DATASET_ROOT, sid, sessions_tbl, trials_tbl)
        if r is not None:
            r["day_stage"] = day_stage
            prepped_by_session[sid] = r
    print(f"{len(prepped_by_session)} sessions usable.", flush=True)

    rows = []
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        rows = existing.to_dict("records")
        done = set(zip(existing["session_id"], existing["target"]))
        print(f"Resuming: {len(done)} (session,target) pairs already done.", flush=True)
    else:
        done = set()

    t_start = time.time()
    n_total = len(prepped_by_session) * len(TARGETS)
    n_done_this_run = 0
    for target, window in TARGETS.items():
        factor_key = "is_whisker" if target == "modality" else "response"
        for session_id, r in prepped_by_session.items():
            if (session_id, target) in done:
                continue
            t0 = time.time()
            day_stage = r["day_stage"]
            y = r[factor_key].astype(int)
            if len(np.unique(y)) < 2:
                print(f"SKIP {session_id}/{target}: only one class present", flush=True)
                continue

            sess_units = units_tbl[(units_tbl["session_id"] == session_id) & (units_tbl["bc_label"].isin(QC_VALUES))]
            unit_ids = sess_units["cluster_id"].to_numpy()
            if len(unit_ids) < 5:
                print(f"SKIP {session_id}/{target}: only {len(unit_ids)} good units", flush=True)
                continue

            X = population_design_matrix(DATASET_ROOT, session_id, unit_ids, r["start_time"], r["is_whisker"], window)
            X = np.nan_to_num(X, nan=0.0)
            observed = nested_cv_balanced_accuracy(X, y, rng, n_runs=N_RUNS)

            imposter_sources = [
                other[factor_key].astype(int)
                for sid2, other in prepped_by_session.items()
                if sid2 != session_id and other["day_stage"] == day_stage
            ]
            if len(imposter_sources) < 3:
                print(f"SKIP null for {session_id}/{target}: only {len(imposter_sources)} imposter sources", flush=True)
                p, null_scores = float("nan"), np.array([])
            else:
                p, null_scores = imposter_pvalue(observed, X, imposter_sources, rng, n_pseudo=N_PSEUDO, n_runs=N_RUNS)

            elapsed = time.time() - t0
            n_done_this_run += 1
            rows.append({
                "session_id": session_id, "day_stage": day_stage, "target": target,
                "n_trials": int(len(y)), "n_units": int(X.shape[1]),
                "observed_balanced_accuracy": observed,
                "null_mean": float(np.nanmean(null_scores)) if len(null_scores) else float("nan"),
                "null_std": float(np.nanstd(null_scores)) if len(null_scores) else float("nan"),
                "p_value": p,
                "null_scores": [float(v) for v in null_scores],
            })
            pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
            total_elapsed = (time.time() - t_start) / 60
            eta_min = (total_elapsed / max(n_done_this_run, 1)) * (n_total - len(done) - n_done_this_run)
            print(f"[{len(done)+n_done_this_run}/{n_total}] {session_id}/{target} ({day_stage}): "
                  f"acc={observed:.3f} null_mean={np.nanmean(null_scores):.3f} p={p:.4f} "
                  f"({elapsed:.1f}s, total {total_elapsed:.1f} min, ETA {eta_min:.1f} min)", flush=True)

    print(f"\nDone. Wrote {len(rows)} rows to {OUT_PATH}. Total elapsed {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
