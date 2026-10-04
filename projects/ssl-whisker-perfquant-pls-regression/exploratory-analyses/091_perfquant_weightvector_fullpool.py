"""Full-pool (88-session) PLS weight-vector extraction (user follow-up to
`089`/`090`'s 12-session pilot: "For the weight vector, show other
visualization and do on all data").

Scoped deliberately narrower than `089`: this computes ONLY the three
single-output PLS+1SE fits per session needed to get each target's
`coef_` weight vector (component selection + one final full-data fit) --
**no null shuffling, no cross-target prediction matrix**. Those are what
made `089` expensive (3 targets x 200 shuffles x 12 sessions); weight
vectors alone need no null at all, which is what makes a full 88-session
run tractable here (~123 PLS fits/session x 88 sessions, no shuffle
loop, versus `089`'s ~3000+ fits/session at pilot scale).

The cross-target SPECIFICITY test itself (the null-controlled part)
stays at `089`'s 12-session pilot scope -- not rescaled here, since the
user's request was specifically about the weight-vector comparison.

Same recipe as `058`/`089`: floored StandardScaler, sensory window
(5-50ms post-stimulus), PLS 1-SE component selection over `{2,5,10,15,
20,30}`, one independent fit per target (not `058`'s joint 3-output fit)
so each target finds its own best neuron-weighting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def select_pls_1se_components(X: np.ndarray, y: np.ndarray, rng: np.random.Generator,
                               n_folds: int = 5, grid: tuple = COMPONENT_GRID) -> int:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    Y = y.reshape(-1, 1)
    max_comp = max(1, min(30, X.shape[1] - 1, int(X.shape[0] * 0.6)))
    candidates = sorted(set(min(c, max_comp) for c in grid if c <= max_comp)) or [max_comp]
    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))

    mean_scores, se_scores = [], []
    for n_comp in candidates:
        fold_scores = []
        for tr, te in splits:
            scaler = _floored_scaler()
            scaler.fit(X[tr])
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(scaler.transform(X[tr]), Y[tr])
            Y_pred = pls.predict(scaler.transform(X[te]))
            fold_scores.append(r2_score(Y[te, 0], Y_pred[:, 0]))
        mean_scores.append(np.mean(fold_scores))
        se_scores.append(np.std(fold_scores) / np.sqrt(n_folds))

    best_idx = int(np.argmax(mean_scores))
    threshold = mean_scores[best_idx] - se_scores[best_idx]
    return next(c for i, c in enumerate(candidates) if mean_scores[i] >= threshold)


def _pls_fit_coef(X, y, n_components):
    from sklearn.cross_decomposition import PLSRegression
    scaler = _floored_scaler()
    scaler.fit(X)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(scaler.transform(X), y.reshape(-1, 1))
    return pls.coef_.ravel()


def process_one_session(args: tuple) -> dict:
    mouse, session_id, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import load_reward_group
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    out = dict(mouse=mouse, session_id=session_id, ok=False)
    reward_group = load_reward_group(mouse)
    if reward_group not in ("R+", "R-"):
        return out
    out["reward_group"] = reward_group

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return out
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    if len(unit_ids) < MIN_UNITS_PER_AREA:
        return out

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    col_std = np.nanstd(X, axis=0)
    keep_units = col_std > 1e-6
    X = X[:, keep_units]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        return out

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    n_components, coef_stack = {}, []
    for k, target in enumerate(TARGETS):
        y = Y[:, k]
        n_1se = select_pls_1se_components(X, y, rng)
        n_components[target] = n_1se
        coef_stack.append(_pls_fit_coef(X, y, n_1se))

    out.update(ok=True, n_trials=len(X), n_units=X.shape[1], n_components=n_components,
               coef_stack=np.stack(coef_stack, axis=0))
    print(f"{mouse}/{session_id} [{reward_group}]: n={len(X)}, p={X.shape[1]}, n_comp={n_components}", flush=True)
    return out


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import pickle

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    hm = hitmiss_session_list(sessions_tbl)
    learning = hm[hm["day_stage"] == "learning"]
    tasks = [(r["subject_id"], r["session_id"], SCRIPTS_DIR) for _, r in learning.iterrows()]
    print(f"{len(tasks)} learning-stage has_ephys sessions", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=48) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                print(f"  FAILED: {futures[fut]} -- {e!r}", flush=True)
                continue
            if res.get("ok"):
                results.append(res)
    print(f"{len(results)}/{len(tasks)} sessions usable", flush=True)

    with open(OUT_DIR / "091_perfquant_weightvector_fullpool_cache.pkl", "wb") as f:
        pickle.dump(results, f)
    print(f"cached {len(results)} sessions -> 091_perfquant_weightvector_fullpool_cache.pkl", flush=True)

    cos_records = []
    for r in results:
        coef = r["coef_stack"]
        for i, ti in enumerate(TARGETS):
            for j, tj in enumerate(TARGETS):
                if j <= i:
                    continue
                cos = float(np.dot(coef[i], coef[j]) / (np.linalg.norm(coef[i]) * np.linalg.norm(coef[j]) + 1e-12))
                cos_records.append(dict(mouse=r["mouse"], session_id=r["session_id"], reward_group=r["reward_group"],
                                         n_trials=r["n_trials"], n_units=r["n_units"],
                                         target_a=ti, target_b=tj, cosine=cos))
    cos_df = pd.DataFrame(cos_records)
    cos_df.to_csv(OUT_DIR / "091_perfquant_weightvector_fullpool_cosine.csv", index=False)
    print(f"saved 091_perfquant_weightvector_fullpool_cosine.csv ({len(cos_df)} rows)", flush=True)

    print("\n=== full-pool weight-vector cosine similarity summary ===")
    for (ta, tb), sub in cos_df.groupby(["target_a", "target_b"]):
        print(f"  {ta} vs {tb}: n={len(sub)}, mean cosine={sub.cosine.mean():+.3f}, median={sub.cosine.median():+.3f}, "
              f"range=[{sub.cosine.min():+.3f}, {sub.cosine.max():+.3f}]")
    print("DONE_091_COMPUTE")


if __name__ == "__main__":
    main()
