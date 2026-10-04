"""Precompute stage for the full cross-mouse null sweep (`081`) -- speed
optimization requested by the user ("optimize for speed as it is a lot
of runs"). `079`'s pilot recomputed each session's whole_brain feature
matrix from raw spikes FRESH on every single (recipient, donor) pairing
that touched it -- fine at pilot scale (355 cells, each session touched a
handful of times), ruinous at full-sweep scale (~5600+ cells, where a
given session can be a DONOR for 70+ different recipients). This script
computes each session's (X_valid, Y_raw_valid) exactly ONCE and caches
it; `081` then does pure array slicing + PLS fitting against the cache,
with zero raw-data reloading in the pairwise step.

All 75 learning-stage good/moderate sessions (38 R+, 37 R-) from `065`'s
pool -- same scope the `079` pilot already used.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
MIN_TRIALS_FOR_REGRESSION = 25


def compute_one(args: tuple) -> dict:
    mouse, session_id, reward_group, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    out = dict(mouse=mouse, session_id=session_id, reward_group=reward_group, ok=False)
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return out
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    col_std = np.nanstd(X, axis=0)
    X = X[:, col_std > 1e-6]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        return out

    out.update(ok=True, X=X, Y_raw=Y_raw, n_trials=len(X), n_units=X.shape[1])
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with open(OUT_DIR / "065_perfquant_allmice_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    sessions = [(r["mouse"], r["session_id"], r["reward_group"]) for r in cache if r["day_stage"] == "learning"]
    print(f"{len(sessions)} learning-stage sessions to precompute "
          f"({sum(1 for _, _, g in sessions if g == 'R+')} R+, {sum(1 for _, _, g in sessions if g == 'R-')} R-)",
          flush=True)

    results = {}
    n_failed = 0
    with ProcessPoolExecutor(max_workers=min(len(sessions), 75)) as ex:
        futures = {ex.submit(compute_one, (m, s, g, SCRIPTS_DIR)): (m, s) for m, s, g in sessions}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                n_failed += 1
                print(f"  {futures[fut]} raised {type(e).__name__}: {e}", flush=True)
                continue
            if res.get("ok"):
                results[res["session_id"]] = res
                print(f"  {res['mouse']}/{res['session_id']}: n_trials={res['n_trials']}, n_units={res['n_units']}",
                      flush=True)
            else:
                print(f"  {res['mouse']}/{res['session_id']}: not usable (too few trials/units)", flush=True)

    print(f"{len(results)}/{len(sessions)} sessions cached ({n_failed} raised an exception)", flush=True)
    with open(OUT_DIR / "080_session_features_cache.pkl", "wb") as f:
        pickle.dump(results, f)
    print("saved 080_session_features_cache.pkl")
    print("DONE_080")


if __name__ == "__main__":
    main()
