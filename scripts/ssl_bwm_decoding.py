"""SSL adaptation of the BWM decoding pipeline
(`brainwidemap/decoding/settings_template.py`, fetched 2026-08-18):
L1-regularized logistic regression (BWM's `ESTIMATOR` choice for a
categorical target; BWM's own alpha grid inverted to sklearn's `C = 1/alpha`
convention), nested cross-validation repeated `N_RUNS` times, significance
via an **imposter-session** null (BWM's own method, not a simpler
circular-shift null -- user decision, question.md) built by resampling
another real session's target-factor sequence as a same-length pseudo-target
against the real session's own neural design matrix.

Targets: `modality` (whisker vs auditory) and `response` (lick vs no-lick)
only -- `outcome` dropped from decoding for the same reason it was dropped
from the single-cell tests (question.md): it is a deterministic function of
(modality, response) in this task, so decoding it is not a distinct question
from decoding response.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from ssl_bwm_windows import unit_rates_for_trials
from ibl_ai_agent.datasets.ssl_ephys import load_spike_shard

# BWM's alpha grid [1e-5..10] inverted to sklearn LogisticRegression's C = 1/alpha.
C_GRID = 1.0 / np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
N_RUNS_DEFAULT = 10  # BWM's N_RUNS (full nested-xv repeats with different folds)
N_FOLDS_OUTER = 5
N_FOLDS_INNER = 5
N_PSEUDO_DEFAULT = 200  # BWM's N_PSEUDO


def population_design_matrix(
    dataset_root: Path,
    session_id: str,
    unit_cluster_ids: np.ndarray,
    start_time: np.ndarray,
    is_whisker: np.ndarray,
    window: tuple[float, float],
) -> np.ndarray:
    """Trials x units firing-rate design matrix, dead-zone-aware per
    `ssl_bwm_windows.unit_rates_for_trials`."""
    shard = load_spike_shard(dataset_root / "spikes" / session_id)
    spike_times_all = shard["spike_times_seconds"]
    spike_clusters_dense = shard["spike_clusters"]
    cluster_ids = shard["cluster_ids"]

    X = np.full((len(start_time), len(unit_cluster_ids)), np.nan)
    for j, cid in enumerate(unit_cluster_ids):
        dense_idx = np.where(cluster_ids == cid)[0]
        if len(dense_idx) == 0:
            continue
        unit_spikes = np.sort(spike_times_all[spike_clusters_dense == dense_idx[0]])
        X[:, j] = unit_rates_for_trials(unit_spikes, start_time, is_whisker, window)
    return X


def nested_cv_balanced_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_runs: int = N_RUNS_DEFAULT,
    n_folds_outer: int = N_FOLDS_OUTER,
    n_folds_inner: int = N_FOLDS_INNER,
    c_grid: np.ndarray = C_GRID,
) -> float:
    """Mean held-out balanced accuracy across `n_runs` repeats of nested CV
    (outer = evaluation, inner = C selection per outer-train fold) -- BWM's
    own 'full nested xv decoding' scheme."""
    scores = []
    for _ in range(n_runs):
        seed = int(rng.integers(0, 2**31 - 1))
        outer = StratifiedKFold(n_splits=n_folds_outer, shuffle=True, random_state=seed)
        for train_idx, test_idx in outer.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            best_c, best_inner_score = c_grid[0], -np.inf
            inner = StratifiedKFold(n_splits=n_folds_inner, shuffle=True, random_state=seed + 1)
            for c in c_grid:
                inner_scores = []
                for itr, ite in inner.split(X_train, y_train):
                    if len(np.unique(y_train[itr])) < 2:
                        continue
                    clf = LogisticRegression(penalty="l1", solver="liblinear", C=c, max_iter=1000)
                    clf.fit(X_train[itr], y_train[itr])
                    pred = clf.predict(X_train[ite])
                    inner_scores.append(balanced_accuracy_score(y_train[ite], pred))
                if inner_scores and np.mean(inner_scores) > best_inner_score:
                    best_inner_score, best_c = np.mean(inner_scores), c

            clf = LogisticRegression(penalty="l1", solver="liblinear", C=best_c, max_iter=1000)
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            scores.append(balanced_accuracy_score(y_test, pred))
    return float(np.mean(scores))


def build_imposter_target(real_len: int, source_factor: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same-length pseudo-target: a random contiguous slice of another real
    session's target-factor sequence (tiled first if that session is
    shorter than the real one). Preserves the source session's own
    autocorrelation/structure, unlike independent per-trial shuffling."""
    n_src = len(source_factor)
    if n_src >= real_len:
        start = int(rng.integers(0, n_src - real_len + 1))
        return source_factor[start:start + real_len]
    reps = int(np.ceil(real_len / n_src))
    tiled = np.tile(source_factor, reps)
    start = int(rng.integers(0, len(tiled) - real_len + 1))
    return tiled[start:start + real_len]


def imposter_pvalue(
    observed_score: float,
    X: np.ndarray,
    imposter_sources: list[np.ndarray],
    rng: np.random.Generator,
    n_pseudo: int,
    n_runs: int,
) -> tuple[float, np.ndarray]:
    """One-sided p-value: fraction of imposter-null decode scores >= the
    observed real-target decode score. imposter_sources: list of other
    sessions' target-factor arrays (same candidate-row semantics, e.g. all
    other same-day-stage-and-cohort sessions' `is_whisker` sequences)."""
    null_scores = np.empty(n_pseudo)
    for i in range(n_pseudo):
        src = imposter_sources[int(rng.integers(0, len(imposter_sources)))]
        y_pseudo = build_imposter_target(X.shape[0], src, rng).astype(int)
        if len(np.unique(y_pseudo)) < 2:
            null_scores[i] = np.nan
            continue
        null_scores[i] = nested_cv_balanced_accuracy(X, y_pseudo, rng, n_runs=n_runs)
    valid = null_scores[~np.isnan(null_scores)]
    p = float((valid >= observed_score).mean()) if len(valid) else float("nan")
    return p, null_scores
