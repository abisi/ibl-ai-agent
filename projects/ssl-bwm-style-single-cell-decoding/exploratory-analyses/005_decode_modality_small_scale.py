"""Small-scale checkpoint for the decoding pipeline (TODO.md): decode
`modality` (whisker vs auditory) from population activity in one session
(AB080_20230622_152205, learning), using all its good units and the locked
evoked window [5ms,35ms], L1-logistic nested CV
(`scripts/ssl_bwm_decoding.py`), with an imposter-session null built from
other learning-day sessions' modality sequences.

Reduced `n_runs`/`n_pseudo` vs the BWM defaults (10 / 200) purely for this
timing check -- see printed extrapolation for the full-setting cost.
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

from ssl_bwm_trial_prep import prep_session
from ssl_bwm_decoding import population_design_matrix, nested_cv_balanced_accuracy, imposter_pvalue
from ibl_ai_agent.data_locations import resolve_dataset_dir

DATASET_ROOT = resolve_dataset_dir("ssl_ephys")
EVOKED_WINDOW = (0.005, 0.035)
RNG_SEED = 20260819

TARGET_SESSION = "AB080_20230622_152205"
TARGET_DAY_STAGE = "learning"
N_RUNS_SMALL = 3       # BWM default: 10
N_PSEUDO_SMALL = 20    # BWM default: 200


def candidate_modality_array(dataset_root: Path, session_id: str, sessions_tbl: pd.DataFrame, trials_tbl: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray] | None:
    prepped = prep_session(dataset_root, session_id, sessions_tbl, trials_tbl)
    if prepped is None:
        return None
    candidate = prepped["trials"][prepped["trials"]["trial_type"].isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
    is_whisker = (candidate["trial_type"] == "whisker_trial").to_numpy()
    return candidate, is_whisker


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    sessions_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "trials.parquet")
    units_tbl = pd.read_parquet(DATASET_ROOT / "metadata" / "units.parquet")

    result = candidate_modality_array(DATASET_ROOT, TARGET_SESSION, sessions_tbl, trials_tbl)
    assert result is not None
    candidate, is_whisker = result
    start_time = candidate["start_time"].to_numpy()
    y = is_whisker.astype(int)
    print(f"Target session {TARGET_SESSION}: n_trials={len(y)}, n_whisker={y.sum()}, n_auditory={(1 - y).sum()}")

    sess_units = units_tbl[(units_tbl["session_id"] == TARGET_SESSION) & (units_tbl["bc_label"] == "good")]
    unit_ids = sess_units["cluster_id"].to_numpy()
    print(f"Good units in session: {len(unit_ids)}")

    t0 = time.time()
    X = population_design_matrix(DATASET_ROOT, TARGET_SESSION, unit_ids, start_time, is_whisker, EVOKED_WINDOW)
    X = np.nan_to_num(X, nan=0.0)  # a whisker trial fully inside the dead zone can't happen for [5,35]ms; safety net only
    print(f"Design matrix: {X.shape} (built in {time.time()-t0:.1f}s)")

    t0 = time.time()
    observed = nested_cv_balanced_accuracy(X, y, rng, n_runs=N_RUNS_SMALL)
    elapsed_real = time.time() - t0
    print(f"Observed nested-CV balanced accuracy (n_runs={N_RUNS_SMALL}): {observed:.4f} ({elapsed_real:.1f}s)")

    # Build imposter target sources: other learning-day sessions' modality sequences.
    day_sessions = sessions_tbl[sessions_tbl["has_ephys"]]["session_id"].tolist()
    imposter_sources = []
    t0 = time.time()
    for sid in day_sessions:
        if sid == TARGET_SESSION or "whisker_0" not in str(sessions_tbl.loc[sessions_tbl["session_id"] == sid, "session_description"].iloc[0]):
            continue
        r = candidate_modality_array(DATASET_ROOT, sid, sessions_tbl, trials_tbl)
        if r is None or len(r[1]) < 5:
            continue
        imposter_sources.append(r[1].astype(int))
        if len(imposter_sources) >= 10:  # small-scale: cap imposter source pool
            break
    print(f"Built {len(imposter_sources)} imposter source sequences in {time.time()-t0:.1f}s")

    t0 = time.time()
    p, null_scores = imposter_pvalue(observed, X, imposter_sources, rng, n_pseudo=N_PSEUDO_SMALL, n_runs=N_RUNS_SMALL)
    elapsed_null = time.time() - t0
    print(f"Imposter null (n_pseudo={N_PSEUDO_SMALL}): mean={np.nanmean(null_scores):.4f} "
          f"std={np.nanstd(null_scores):.4f} p={p:.4f} ({elapsed_null:.1f}s, {elapsed_null/N_PSEUDO_SMALL:.2f}s/pseudo)")

    full_estimate_min = (elapsed_real + elapsed_null / N_PSEUDO_SMALL * 200) * (10 / N_RUNS_SMALL) / 60
    print(f"\nExtrapolated to BWM defaults (n_runs=10, n_pseudo=200) for ONE session/target: ~{full_estimate_min:.1f} min")

    import json
    out = {
        "session_id": TARGET_SESSION, "day_stage": TARGET_DAY_STAGE, "target": "modality",
        "n_trials": int(len(y)), "n_units": int(X.shape[1]),
        "n_runs": N_RUNS_SMALL, "n_pseudo": N_PSEUDO_SMALL,
        "observed_balanced_accuracy": observed,
        "null_scores": [float(v) for v in null_scores],
        "p_value": p,
    }
    out_path = Path(__file__).resolve().parent / "005_decode_modality_small_scale_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
