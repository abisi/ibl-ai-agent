"""PLS+1SE true-vs-predicted trial curves for MANY sessions (not just 6),
spanning the full-pool performance range best to worst, with the shift-null
predictions overlaid as a mean +/- 95% CI band on the SAME trial axis --
directly visualizing "what the real model predicted over null" per session
(user 2026-09-20: "Plot many sessions, both real and null predictions and
what the real models has predicted over null.").

For each selected session: the real PLS+1SE prediction comes from the same
pooled-CV used throughout (`058`/`059`). The null band comes from
N_SHUF_NULL full retrain-under-linear-shift nulls (same convention as
`052`/`056`/`058`), but here each shuffle's predictions are mapped back
onto their ORIGINAL trial indices (a trial covered by a given shuffle's
shifted+trimmed window gets that shuffle's prediction; trials outside a
given shuffle's window are NaN for that shuffle) so a mean+/-95%CI null
band can be drawn on the same trial axis as the true curve and the real
prediction, rather than reported only as a single pooled number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
N_SHUF_NULL = 10
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
N_EXAMPLES = 12


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def select_pls_1se_components(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                               n_folds: int = 5, grid: tuple = COMPONENT_GRID) -> int:
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
    return next(c for i, c in enumerate(candidates) if mean_scores[i] >= threshold)


def _pls_fit_predict(X_train, Y_train, X_test, n_components):
    from sklearn.cross_decomposition import PLSRegression
    scaler = _floored_scaler()
    scaler.fit(X_train)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(scaler.transform(X_train), Y_train)
    return pls.predict(scaler.transform(X_test))


def _pooled_cv_predict(X, Y, n_components, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS):
    from sklearn.model_selection import KFold
    Y_pred_accum = np.zeros_like(Y, dtype=float)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y, dtype=float)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            Y_pred_rep[te] = _pls_fit_predict(X[tr], Y[tr], X[te], n_components)
        Y_pred_accum += Y_pred_rep
    return Y_pred_accum / n_repeats


def null_predictions_per_trial(X: np.ndarray, Y: np.ndarray, n_components: int, rng: np.random.Generator,
                                n_shuf: int = N_SHUF_NULL, n_folds: int = N_FOLDS) -> np.ndarray:
    """Full retrain-under-shift null, predictions mapped back onto the
    ORIGINAL trial-index axis (NaN where a given shuffle's shifted window
    doesn't cover that trial) -- shape (n_shuf, n_trials, n_targets).
    """
    n = len(Y)
    out = np.full((n_shuf, n, Y.shape[1]), np.nan)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    for s in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
            orig_idx = np.arange(shift, n)
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
            orig_idx = np.arange(0, n - shift)
        Y_pred_shift = _pooled_cv_predict(X_shift, Y_shift, n_components, rng, n_repeats=1, n_folds=n_folds)
        out[s, orig_idx] = Y_pred_shift
    return out


def process_one_session(args: tuple) -> dict:
    mouse, session_id, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
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
    trial_index = np.arange(len(targets_df))[valid]

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    n_1se = select_pls_1se_components(X, Y, rng)
    Y_pred = _pooled_cv_predict(X, Y, n_1se, rng)
    test_pearson = [pearsonr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])]

    null_preds = null_predictions_per_trial(X, Y, n_1se, rng)
    null_mean = np.nanmean(null_preds, axis=0)
    null_std = np.nanstd(null_preds, axis=0)
    null_pearson = [pearsonr(Y[~np.isnan(null_mean[:, k]), k], null_mean[~np.isnan(null_mean[:, k]), k])[0]
                    for k in range(Y.shape[1])]

    return dict(mouse=mouse, session_id=session_id, n_components=n_1se, trial_index=trial_index,
                Y_true=Y, Y_pred=Y_pred, test_pearson=test_pearson,
                null_mean=null_mean, null_std=null_std, null_pearson=null_pearson)


def main():
    df = pd.read_csv(OUT_DIR / "058_perfquant_pls_nulldist_fullpool.csv")
    sub = df[df.method == "PLS+1SE"]
    mean_pearson = sub.groupby("session_id")["test_pearson"].mean().sort_values(ascending=False)
    mouse_lookup = sub.drop_duplicates("session_id").set_index("session_id")["mouse"]

    n = len(mean_pearson)
    idxs = np.unique(np.linspace(0, n - 1, N_EXAMPLES).round().astype(int))
    chosen_sessions = mean_pearson.index[idxs].tolist()
    rank_by_session = {sid: int(i) + 1 for i, sid in zip(idxs, chosen_sessions)}

    print(f"ranking {n} sessions by PLS+1SE mean test Pearson; selected {len(chosen_sessions)} (best -> worst):")
    for sid in chosen_sessions:
        print(f"  rank {rank_by_session[sid]}/{n}: {mouse_lookup[sid]}/{sid} mean_test_pearson={mean_pearson[sid]:.3f}")

    tasks = [(mouse_lookup[sid], sid, SCRIPTS_DIR) for sid in chosen_sessions]
    results = {}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=min(16, len(tasks))) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results[res["session_id"]] = res
            print(f"  done: {res['mouse']}/{res['session_id']} n_comp={res['n_components']} "
                  f"real_pearson={[round(v, 3) for v in res['test_pearson']]} "
                  f"null_pearson={[round(v, 3) for v in res['null_pearson']]}", flush=True)

    ordered_sessions = [sid for sid in chosen_sessions if sid in results]

    fig, axes = plt.subplots(len(ordered_sessions), len(TARGETS),
                              figsize=(4.2 * len(TARGETS), 2.4 * len(ordered_sessions)), constrained_layout=True)
    for row_i, sid in enumerate(ordered_sessions):
        res = results[sid]
        rank = rank_by_session[sid]
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i][col_i]
            t = res["trial_index"]
            ax.plot(t, res["Y_true"][:, col_i], color="#333333", lw=1.3, label="true")
            ax.plot(t, res["Y_pred"][:, col_i], color="#d62728", lw=1.1, alpha=0.9,
                    label=f"real (r={res['test_pearson'][col_i]:.2f})")
            null_mean = res["null_mean"][:, col_i]
            null_std = res["null_std"][:, col_i]
            ax.plot(t, null_mean, color="#888888", lw=1.0, alpha=0.9,
                    label=f"null mean (r={res['null_pearson'][col_i]:.2f})")
            ax.fill_between(t, null_mean - 1.96 * null_std, null_mean + 1.96 * null_std,
                             color="#888888", alpha=0.25, linewidth=0)
            title = f"rank {rank}/{n}: {res['mouse']} -- {target}" if col_i == 0 else target
            ax.set_title(title, fontsize=7.5)
            ax.set_xlabel("trial index", fontsize=6.5)
            ax.set_ylabel("z-scored", fontsize=6.5)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=5.5, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig_path = OUT_DIR / "060_perfquant_pls1se_real_vs_null_examples.png"
    fig.savefig(fig_path, dpi=140)
    print(f"saved {fig_path.name}")
    print("DONE_060")


if __name__ == "__main__":
    main()
