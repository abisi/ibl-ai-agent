"""The "best recipe" for a cross-validated, well-calibrated, non-overfitting
single-animal perfquant model (user request 2026-09-20: "Implement all of
this, show results for example sessions" -- following the design
synthesis from `055`'s Ridge/Lasso/ElasticNet/PLS comparison and its
train-vs-test gap finding).

Two new methods, both built to directly address what `055` found wrong
with the straightforward versions:

1. **PLS + 1-SE-rule component selection** (`select_pls_components_1se`):
   `055`'s plain PLS picked "however many components maximize CV score",
   which turned out to be inconsistent -- 2 components for one session
   (real gap reduction) but 10 for another (train~0.999, no better than
   Ridge). The 1-SE rule instead picks the SMALLEST component count whose
   CV score is within one standard error of the best -- a standard
   statistical-learning heuristic that explicitly trades a little
   achievable test score for a much simpler, less overfit model, rather
   than hoping simplicity falls out of a pure score-maximizing search.

2. **Stability-selection Lasso** (`stability_selection_lasso`): plain
   Lasso (in `055`) catastrophically failed on one session (negative
   correlation on all 3 targets) -- a known Lasso weakness with
   correlated features and small n (single-fit coefficient selection is
   unstable). Instead: refit Lasso across many bootstrap resamples of the
   TRAINING fold only (never the test fold, to avoid leakage), keep only
   features selected (nonzero coefficient) in at least half the resamples,
   then fit a plain Ridge (fixed, light alpha -- no second hyperparameter
   search) on just that stable feature subset.

Both use the SAME floored scaler as `052`/`053`/`055` (never plain
`StandardScaler`) and both are scored via the SAME pooled-CV convention
used throughout this project, with a full retrain-under-linear-shift null
(matching `052`'s more rigorous within-session null, not `050`'s cheaper
post-hoc-only version -- affordable here since single-session fits are
cheap). Every result reports test AND train AND the gap together, per
the recipe's own recommendation.

NOT implemented here (a much bigger, separate lift, out of scope for "a
few test sessions"): pooling across sessions/pseudopopulation to actually
increase n rather than work around p>>n -- flagged in the design
discussion as the one lever that changes the fundamental ratio, not
worked around by any single-session method.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
N_REPEATS = 3
N_SHUF_NULL = 10  # reduced from 052's 20 -- reused fixed hyperparameter per shuffle, kept cheap for "a few sessions"
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
EXAMPLE_MICE = ["AB158", "AB154", "AB092", "MH011", "MH022"]  # same set 053/055 used
METHOD_COLORS = {"Ridge (baseline)": "#1f77b4", "PLS+1SE": "#d62728", "Stability-Lasso": "#9467bd", "PLS-Forest": "#2ca02c"}


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


# ---------------------------------------------------------------------------
# Method 1: PLS with 1-SE-rule component selection
# ---------------------------------------------------------------------------

def select_pls_components_1se(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                               n_folds: int = 5, grid: tuple = (2, 5, 10, 15, 20, 30)) -> int:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

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
            fold_scores.append(np.mean([r2_score(Y[te, k], Y_pred[:, k]) for k in range(Y.shape[1])]))
        mean_scores.append(np.mean(fold_scores))
        se_scores.append(np.std(fold_scores) / np.sqrt(n_folds))

    best_idx = int(np.argmax(mean_scores))
    threshold = mean_scores[best_idx] - se_scores[best_idx]
    # smallest n_components whose mean score clears (best - 1SE)
    for i, n_comp in enumerate(candidates):
        if mean_scores[i] >= threshold:
            return n_comp
    return candidates[best_idx]


def _pls_fit_predict(X_train, Y_train, X_test, n_components):
    from sklearn.cross_decomposition import PLSRegression
    scaler = _floored_scaler()
    scaler.fit(X_train)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(scaler.transform(X_train), Y_train)
    return pls.predict(scaler.transform(X_test))


def run_pls_1se(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n_repeats: int = N_REPEATS, n_folds: int = 5) -> dict:
    from sklearn.model_selection import KFold

    n_components = select_pls_components_1se(X, Y, rng)

    Y_pred_accum = np.zeros_like(Y)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            Y_pred_rep[te] = _pls_fit_predict(X[tr], Y[tr], X[te], n_components)
        Y_pred_accum += Y_pred_rep
    Y_test_pred = Y_pred_accum / n_repeats
    Y_train_pred = _pls_fit_predict(X, Y, X, n_components)

    null_r2 = np.full((N_SHUF_NULL, Y.shape[1]), np.nan)
    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    from sklearn.metrics import r2_score
    for s in range(N_SHUF_NULL):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_null = np.empty_like(Y_shift)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_shift):
            Y_pred_null[te] = _pls_fit_predict(X_shift[tr], Y_shift[tr], X_shift[te], n_components)
        null_r2[s] = [r2_score(Y_shift[:, k], Y_pred_null[:, k]) for k in range(Y.shape[1])]

    return _package_results(Y, Y_test_pred, Y_train_pred, null_r2, extra=dict(n_components=n_components))


# ---------------------------------------------------------------------------
# Method 3: PLS "forest" -- bagging + random feature subspace (user
# 2026-09-20: "Do PLS. Would it make sense to have a forest of models, do
# model bagging, ensembling?"). Random Forest's own recipe (bootstrap
# resample of samples + random subset of features per base learner,
# average predictions), applied to PLS instead of decision trees --
# motivated by the SAME p>>n instability this whole thread has been
# tracing (PLS's component count swinging from 2 to 10 session-to-session,
# Lasso's single-fit collapse on AB154): bagging trades a bit of bias for
# a real reduction in that fit-to-fit variance.
# ---------------------------------------------------------------------------

N_FOREST_ESTIMATORS = 30
FOREST_FEATURE_FRAC = 0.5
FOREST_N_COMPONENTS = 5  # fixed, not re-selected per tree (30 trees x 1SE-search would be too costly) --
                          # the modal choice `select_pls_components_1se` made across sessions in this thread


def _pls_forest_fit_predict(X_train, Y_train, X_test, rng, n_estimators=N_FOREST_ESTIMATORS,
                             feature_frac=FOREST_FEATURE_FRAC, n_components=FOREST_N_COMPONENTS):
    n_train = len(Y_train)
    n_feat_sub = max(n_components + 1, int(X_train.shape[1] * feature_frac))
    preds = np.zeros((X_test.shape[0], Y_train.shape[1]))
    for _ in range(n_estimators):
        row_idx = rng.integers(0, n_train, size=n_train)          # bootstrap resample of TRIALS
        col_idx = rng.choice(X_train.shape[1], size=n_feat_sub, replace=False)  # random subset of NEURONS
        preds += _pls_fit_predict(X_train[np.ix_(row_idx, col_idx)], Y_train[row_idx], X_test[:, col_idx], n_components)
    return preds / n_estimators


def run_pls_forest(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n_repeats: int = N_REPEATS, n_folds: int = 5) -> dict:
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    Y_pred_accum = np.zeros_like(Y)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            Y_pred_rep[te] = _pls_forest_fit_predict(X[tr], Y[tr], X[te], rng)
        Y_pred_accum += Y_pred_rep
    Y_test_pred = Y_pred_accum / n_repeats
    Y_train_pred = _pls_forest_fit_predict(X, Y, X, rng)

    null_r2 = np.full((N_SHUF_NULL, Y.shape[1]), np.nan)
    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    for s in range(N_SHUF_NULL):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_null = np.empty_like(Y_shift)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_shift):
            Y_pred_null[te] = _pls_forest_fit_predict(X_shift[tr], Y_shift[tr], X_shift[te], rng)
        null_r2[s] = [r2_score(Y_shift[:, k], Y_pred_null[:, k]) for k in range(Y.shape[1])]

    return _package_results(Y, Y_test_pred, Y_train_pred, null_r2,
                             extra=dict(n_estimators=N_FOREST_ESTIMATORS, feature_frac=FOREST_FEATURE_FRAC))


# ---------------------------------------------------------------------------
# Method 2: stability-selection Lasso
# ---------------------------------------------------------------------------

def stability_selection_mask(X_train: np.ndarray, Y_train: np.ndarray, rng: np.random.Generator,
                              n_boot: int = 30, alpha: float = 0.1, keep_frac: float = 0.5) -> np.ndarray:
    from sklearn.linear_model import Lasso
    n = len(Y_train)
    selected_counts = np.zeros(X_train.shape[1])
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)  # bootstrap resample, TRAINING fold only
        scaler = _floored_scaler()
        Xb = scaler.fit_transform(X_train[idx])
        lasso = Lasso(alpha=alpha, max_iter=5000)
        lasso.fit(Xb, Y_train[idx])
        # nonzero in ANY head counts as "selected" for that bootstrap draw
        selected_counts += (np.abs(lasso.coef_) > 1e-10).any(axis=0)
    mask = (selected_counts / n_boot) >= keep_frac
    if not mask.any():
        mask = np.ones(X_train.shape[1], dtype=bool)  # fallback: nothing stable enough, keep everything
    return mask


def _stability_fit_predict(X_train, Y_train, X_test, rng, ridge_alpha: float = 1.0):
    from sklearn.linear_model import Ridge
    mask = stability_selection_mask(X_train, Y_train, rng)
    scaler = _floored_scaler()
    scaler.fit(X_train[:, mask])
    reg = Ridge(alpha=ridge_alpha)
    reg.fit(scaler.transform(X_train[:, mask]), Y_train)
    return reg.predict(scaler.transform(X_test[:, mask])), mask.sum()


def run_stability_lasso(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n_repeats: int = N_REPEATS, n_folds: int = 5) -> dict:
    from sklearn.model_selection import KFold

    n_stable_features = []
    Y_pred_accum = np.zeros_like(Y)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            Y_pred_rep[te], n_feat = _stability_fit_predict(X[tr], Y[tr], X[te], rng)
            n_stable_features.append(n_feat)
        Y_pred_accum += Y_pred_rep
    Y_test_pred = Y_pred_accum / n_repeats
    Y_train_pred, n_feat_full = _stability_fit_predict(X, Y, X, rng)

    null_r2 = np.full((N_SHUF_NULL, Y.shape[1]), np.nan)
    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    from sklearn.metrics import r2_score
    for s in range(N_SHUF_NULL):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_null = np.empty_like(Y_shift)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_shift):
            Y_pred_null[te], _ = _stability_fit_predict(X_shift[tr], Y_shift[tr], X_shift[te], rng)
        null_r2[s] = [r2_score(Y_shift[:, k], Y_pred_null[:, k]) for k in range(Y.shape[1])]

    return _package_results(Y, Y_test_pred, Y_train_pred, null_r2, extra=dict(mean_n_stable_features=float(np.mean(n_stable_features))))


# ---------------------------------------------------------------------------
# Baseline for comparison: plain Ridge (055's version, unchanged)
# ---------------------------------------------------------------------------

def run_ridge_baseline(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n_repeats: int = N_REPEATS, n_folds: int = 5) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]

    def make_model(alpha):
        return make_pipeline(_floored_scaler(), Ridge(alpha=alpha))

    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    best_alpha, best_score = grid[0], -np.inf
    for a in grid:
        Y_pred = np.empty_like(Y)
        for tr, te in splits:
            m = make_model(a)
            m.fit(X[tr], Y[tr])
            Y_pred[te] = m.predict(X[te])
        score = np.mean([r2_score(Y[:, k], Y_pred[:, k]) for k in range(Y.shape[1])])
        if score > best_score:
            best_score, best_alpha = score, a

    Y_pred_accum = np.zeros_like(Y)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            m = make_model(best_alpha)
            m.fit(X[tr], Y[tr])
            Y_pred_rep[te] = m.predict(X[te])
        Y_pred_accum += Y_pred_rep
    Y_test_pred = Y_pred_accum / n_repeats
    m_train = make_model(best_alpha)
    m_train.fit(X, Y)
    Y_train_pred = m_train.predict(X)

    null_r2 = np.full((N_SHUF_NULL, Y.shape[1]), np.nan)
    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    for s in range(N_SHUF_NULL):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_null = np.empty_like(Y_shift)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_shift):
            m = make_model(best_alpha)
            m.fit(X_shift[tr], Y_shift[tr])
            Y_pred_null[te] = m.predict(X_shift[te])
        null_r2[s] = [r2_score(Y_shift[:, k], Y_pred_null[:, k]) for k in range(Y.shape[1])]

    return _package_results(Y, Y_test_pred, Y_train_pred, null_r2, extra=dict(alpha=best_alpha))


def _package_results(Y, Y_test_pred, Y_train_pred, null_r2, extra: dict) -> dict:
    from sklearn.metrics import r2_score
    test_pearson = [pearsonr(Y[:, k], Y_test_pred[:, k])[0] for k in range(Y.shape[1])]
    test_spearman = [spearmanr(Y[:, k], Y_test_pred[:, k])[0] for k in range(Y.shape[1])]
    test_r2 = [r2_score(Y[:, k], Y_test_pred[:, k]) for k in range(Y.shape[1])]
    train_pearson = [pearsonr(Y[:, k], Y_train_pred[:, k])[0] for k in range(Y.shape[1])]
    train_r2 = [r2_score(Y[:, k], Y_train_pred[:, k]) for k in range(Y.shape[1])]
    null_r2_mean = np.nanmean(null_r2, axis=0).tolist()
    null_r2_std = np.nanstd(null_r2, axis=0).tolist()
    return dict(Y_test_pred=Y_test_pred, Y_train_pred=Y_train_pred, test_pearson=test_pearson,
                test_spearman=test_spearman, test_r2=test_r2, train_pearson=train_pearson, train_r2=train_r2,
                null_r2_mean=null_r2_mean, null_r2_std=null_r2_std, **extra)


METHODS = {"Ridge (baseline)": run_ridge_baseline, "PLS+1SE": run_pls_1se, "Stability-Lasso": run_stability_lasso, "PLS-Forest": run_pls_forest}


def process_one(mouse: str) -> dict | None:
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_bwm_trial_prep import load_reward_group
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                         & (sessions_tbl["has_ephys"])]
    if len(sess) == 0:
        return None
    session_id = sess["session_id"].iloc[0]
    reward_group = load_reward_group(mouse)

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None:
        return None
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    if len(unit_ids) < MIN_UNITS_PER_AREA or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return None

    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    col_std = np.nanstd(X, axis=0)
    X = X[:, col_std > 1e-6]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    trial_index = np.arange(len(targets_df))[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION:
        return None

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    out = dict(mouse=mouse, reward_group=reward_group, session_id=session_id, trial_index=trial_index, Y_true=Y, methods={})
    for name, fn in METHODS.items():
        res = fn(X, Y, rng)
        out["methods"][name] = res
        print(f"  {mouse} [{name}]: n={len(X)}, p={X.shape[1]}, "
              f"test_pearson={[round(p, 3) for p in res['test_pearson']]}, "
              f"train_pearson={[round(p, 3) for p in res['train_pearson']]}, "
              f"null_r2_mean={[round(r, 3) for r in res['null_r2_mean']]}, "
              f"extra={ {k: v for k, v in res.items() if k not in ('Y_test_pred','Y_train_pred','test_pearson','test_spearman','test_r2','train_pearson','train_r2','null_r2_mean','null_r2_std')} }",
              flush=True)
    return out


def main():
    results = {}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=min(5, len(EXAMPLE_MICE))) as ex:
        futures = {ex.submit(process_one, m): m for m in EXAMPLE_MICE}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                results[res["mouse"]] = res

    # --- Figure 1: true vs predicted curves, all methods overlaid ---
    fig1, axes1 = plt.subplots(len(TARGETS), len(EXAMPLE_MICE), figsize=(4.2 * len(EXAMPLE_MICE), 3.4 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        for col_i, mouse in enumerate(EXAMPLE_MICE):
            ax = axes1[row_i][col_i]
            res = results.get(mouse)
            if res is None:
                continue
            ax.plot(res["trial_index"], res["Y_true"][:, row_i], color="#333333", lw=1.4, label="true")
            for name, m in res["methods"].items():
                r = m["test_pearson"][row_i]
                ax.plot(res["trial_index"], m["Y_test_pred"][:, row_i], color=METHOD_COLORS[name], lw=1.0, alpha=0.8,
                         label=f"{name} (r={r:.2f})")
            ax.set_title(f"{mouse}: {target}", fontsize=9)
            ax.set_xlabel("trial index", fontsize=8)
            ax.set_ylabel("z-scored value", fontsize=8)
            ax.legend(fontsize=6, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig1_path = OUT_DIR / "056_perfquant_best_recipe_curves.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- records / CSV ---
    records = []
    for mouse, res in results.items():
        for name, m in res["methods"].items():
            for k, target in enumerate(TARGETS):
                records.append(dict(mouse=mouse, method=name, target=target, test_pearson=m["test_pearson"][k],
                                     test_spearman=m["test_spearman"][k], test_r2=m["test_r2"][k],
                                     train_pearson=m["train_pearson"][k], train_r2=m["train_r2"][k],
                                     null_r2_mean=m["null_r2_mean"][k], null_r2_std=m["null_r2_std"][k]))
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "056_perfquant_best_recipe.csv", index=False)

    method_names = list(METHODS.keys())

    # --- Figure 2: test Pearson bar chart, grouped by session, per method, faceted by target ---
    fig2, axes2 = plt.subplots(len(TARGETS), 1, figsize=(10, 3.6 * len(TARGETS)), constrained_layout=True)
    mice_present = [m for m in EXAMPLE_MICE if m in results]
    bar_width = 0.8 / len(method_names)
    x_base = np.arange(len(mice_present))
    for row_i, target in enumerate(TARGETS):
        ax = axes2[row_i]
        for j, name in enumerate(method_names):
            vals = [df[(df.mouse == mouse) & (df.method == name) & (df.target == target)]["test_pearson"].iloc[0] for mouse in mice_present]
            ax.bar(x_base + j * bar_width, vals, width=bar_width, color=METHOD_COLORS[name], label=name)
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        ax.set_xticks(x_base + bar_width * (len(method_names) - 1) / 2)
        ax.set_xticklabels(mice_present)
        ax.set_ylabel("test Pearson r")
        ax.set_title(target, fontsize=10)
        if row_i == 0:
            ax.legend(fontsize=8, frameon=False, ncol=len(method_names))
        ax.spines[["top", "right"]].set_visible(False)
    fig2_path = OUT_DIR / "056_perfquant_best_recipe_bars.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    # --- Figure 3: train vs test gap, per method, faceted by target ---
    fig3, axes3 = plt.subplots(len(TARGETS), 1, figsize=(9, 3.6 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        ax = axes3[row_i]
        sub = df[df.target == target]
        train_means = [sub[sub.method == name]["train_pearson"].mean() for name in method_names]
        test_means = [sub[sub.method == name]["test_pearson"].mean() for name in method_names]
        x = np.arange(len(method_names))
        ax.bar(x - 0.2, train_means, width=0.4, color="#888888", label="train (in-sample)")
        ax.bar(x + 0.2, test_means, width=0.4, color=[METHOD_COLORS[n] for n in method_names], label="test (OOF)")
        for xi, (tr, te) in enumerate(zip(train_means, test_means)):
            ax.text(xi, max(tr, te) + 0.03, f"gap={tr - te:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.set_ylim(0, 1.15)
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        ax.set_ylabel("mean Pearson r")
        ax.set_title(f"{target}: train vs test (overfitting gap)", fontsize=10)
        if row_i == 0:
            ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig3_path = OUT_DIR / "056_perfquant_best_recipe_traintest_gap.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"saved {fig3_path.name}")

    print("\n=== summary: test pearson, train pearson, gap, real-vs-null (R2) per method ===")
    for target in TARGETS:
        sub = df[df.target == target]
        for name in method_names:
            s = sub[sub.method == name]
            above_null = (s["test_r2"] > s["null_r2_mean"] + s["null_r2_std"]).sum()
            print(f"  {target} / {name}: test={s['test_pearson'].mean():.3f}, train={s['train_pearson'].mean():.3f}, "
                  f"gap={s['train_pearson'].mean()-s['test_pearson'].mean():.3f}, "
                  f"R2 real={s['test_r2'].mean():.3f} vs null={s['null_r2_mean'].mean():.3f}, "
                  f"{above_null}/{len(s)} sessions above null+1sd")


if __name__ == "__main__":
    main()
