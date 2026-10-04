"""Confirmatory per-area decoding: modality + response, decoded from only
the units in each (session, area) cell, at BOTH area parcellations
(`area_acronym_custom` fine, `area_group` coarse), min 5 units/area (same
threshold as the single-cell area aggregation, `003_*`). Reuses the same
nested-CV + imposter-null machinery as session-level decoding
(`002_run_all_decoding.py`) -- the imposter null still resamples target
sequences from other sessions of the same day-stage; only the design matrix
is restricted to that area's units.

This is a large job (thousands of (parcellation, session, area, target)
decodes) -- writes/resumes incrementally, checkpointing every 10 completed
decodes.
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
AREA_LABELS_PATH = Path("reports/ssl_analysis/derived/unit_area_labels.parquet")
EVOKED_WINDOW = (0.005, 0.035)
BASELINE_WINDOW = (-0.200, -0.010)
N_RUNS = 5
N_PSEUDO = 50
MIN_UNITS_PER_AREA = 5  # unchanged per user decision 2026-08-19
QC_VALUES = ("good", "mua")  # widened from good-only, user decision 2026-08-19
RNG_SEED = 20260819
CHECKPOINT_EVERY = 10

PARCELLATIONS = ["area_acronym_custom", "area_group"]
TARGETS = {"modality": EVOKED_WINDOW, "response": BASELINE_WINDOW}
OUT_PATH = Path(__file__).resolve().parent / "decoding_per_area_results.parquet"


def candidate_arrays(dataset_root, session_id, sessions_tbl, trials_tbl):
    prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
    if prepped is None:
        return None
    candidate = prepped["trials"][prepped["trials"]["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
    if len(candidate) < 8:
        return None
    return {
        "is_whisker": (candidate["trial_type"] == "whisker_trial").to_numpy(),
        "response": candidate["lick_flag"].to_numpy().astype(bool),
        "start_time": candidate["start_time"].to_numpy(),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)
    area_labels = area_labels[area_labels["bc_label"].isin(QC_VALUES)]

    sessions = list_whisker_training_ephys_sessions(sessions_tbl)
    print(f"Preloading candidate arrays for {len(sessions)} sessions...", flush=True)
    prepped_by_session = {}
    for sid, day_stage in sessions:
        r = candidate_arrays(DATASET_ROOT, sid, sessions_tbl, trials_tbl)
        if r is not None:
            r["day_stage"] = day_stage
            prepped_by_session[sid] = r
    print(f"{len(prepped_by_session)} sessions usable.", flush=True)

    jobs = []
    for parcellation in PARCELLATIONS:
        al = area_labels[area_labels["session_id"].isin(prepped_by_session.keys())]
        for (sid, area), sub in al.groupby(["session_id", parcellation]):
            unit_ids = sub["cluster_id"].to_numpy()
            if len(unit_ids) < MIN_UNITS_PER_AREA:
                continue
            jobs.append((parcellation, sid, area, unit_ids))
    n_total = len(jobs) * len(TARGETS)
    print(f"{len(jobs)} (parcellation, session, area) cells x {len(TARGETS)} targets = {n_total} decodes", flush=True)

    rows = []
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        rows = existing.to_dict("records")
        done = set(zip(existing["parcellation"], existing["session_id"], existing["area"], existing["target"]))
        print(f"Resuming: {len(done)} decodes already done.", flush=True)
    else:
        done = set()

    t_start = time.time()
    n_done_this_run = 0
    for target, window in TARGETS.items():
        factor_key = "is_whisker" if target == "modality" else "response"
        for parcellation, session_id, area, unit_ids in jobs:
            key = (parcellation, session_id, area, target)
            if key in done:
                continue
            t0 = time.time()
            r = prepped_by_session[session_id]
            day_stage = r["day_stage"]
            y = r[factor_key].astype(int)
            if len(np.unique(y)) < 2:
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
                p, null_scores = float("nan"), np.array([])
            else:
                p, null_scores = imposter_pvalue(observed, X, imposter_sources, rng, n_pseudo=N_PSEUDO, n_runs=N_RUNS)

            elapsed = time.time() - t0
            n_done_this_run += 1
            rows.append({
                "parcellation": parcellation, "session_id": session_id, "day_stage": day_stage,
                "area": area, "target": target, "n_units": int(len(unit_ids)),
                "n_trials": int(len(y)), "observed_balanced_accuracy": observed,
                "null_mean": float(np.nanmean(null_scores)) if len(null_scores) else float("nan"),
                "null_std": float(np.nanstd(null_scores)) if len(null_scores) else float("nan"),
                "p_value": p,
            })

            if n_done_this_run % CHECKPOINT_EVERY == 0:
                pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
                total_elapsed = (time.time() - t_start) / 60
                eta_min = (total_elapsed / n_done_this_run) * (n_total - len(done) - n_done_this_run)
                print(f"[{len(done)+n_done_this_run}/{n_total}] last: {parcellation}/{session_id}/{area}/{target} "
                      f"acc={observed:.3f} p={p:.4f} ({elapsed:.1f}s) "
                      f"total {total_elapsed:.1f} min, ETA {eta_min:.1f} min", flush=True)

    pd.DataFrame(rows).to_parquet(OUT_PATH, index=False)
    print(f"\nDone. Wrote {len(rows)} rows to {OUT_PATH}. Total elapsed {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
