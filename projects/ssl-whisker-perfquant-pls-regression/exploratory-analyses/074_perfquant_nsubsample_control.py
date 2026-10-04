"""Neuron-count control for the PLS+1SE perfquant regression (user request
2026-09-23: same control as `073`, applied to the PLS analysis -- scoped
to the existing single sensory window, NOT extended to be time-resolved,
per user confirmation via AskUserQuestion: "Accuracy vs N only, same
single window").

Same 5 example sessions as `073`/`068` (`EXAMPLE_MICE`), learning-stage,
`whole_brain` -- subsamples the unit pool to N = 100, 200, ..., up to
`N_CAP` (or the session's own unit count if smaller), plus one extra
point at the session's true full unit count. `N_REPEATS_SUBSAMPLE`
independent draws per N, averaged.

Reuses `058`'s exact recipe (sensory window, floored scaler, PLS+1SE via
`select_pls_components`, pooled-CV test score) -- same estimator, so
scores are directly comparable to the production `058` numbers at
N=n_total.

**Real test score only, no null shuffles** -- same cost-saving rationale
as `073`: this control asks how the score scales with N, not whether it's
significant at every N (which would multiply cost by `N_SHUF_NULL`).
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
EXAMPLE_MICE = ["AB158", "AB154", "AB092", "MH011", "MH022"]
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_STEP = 100
N_CAP = 1000
N_REPEATS_SUBSAMPLE = 3
N_FOLDS = 5
N_REPEATS_CV = 2
MIN_TRIALS_FOR_REGRESSION = 25
TARGET_COLORS = {"whisker_curve": "#1f77b4", "falsealarm_curve": "#ff7f0e", "performance_curve": "#2ca02c"}


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def select_pls_components(X, Y, rng, n_folds=5, grid=COMPONENT_GRID):
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
    n_1se = next(c for i, c in enumerate(candidates) if mean_scores[i] >= threshold)
    return n_1se


def _pls_fit_predict(X_train, Y_train, X_test, n_components):
    from sklearn.cross_decomposition import PLSRegression
    scaler = _floored_scaler()
    scaler.fit(X_train)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(scaler.transform(X_train), Y_train)
    return pls.predict(scaler.transform(X_test))


def _pooled_cv_predict(X, Y, n_components, rng, n_repeats=N_REPEATS_CV, n_folds=N_FOLDS):
    from sklearn.model_selection import KFold
    Y_pred_accum = np.zeros_like(Y, dtype=float)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y, dtype=float)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            Y_pred_rep[te] = _pls_fit_predict(X[tr], Y[tr], X[te], n_components)
        Y_pred_accum += Y_pred_rep
    return Y_pred_accum / n_repeats


def process_one_session(args: tuple) -> dict:
    mouse, session_id, reward_group, scripts_dir = args
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
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return dict(mouse=mouse, session_id=session_id, ok=False)

    all_unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    n_total = len(all_unit_ids)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)

    Y_raw_full = targets_df[TARGETS].to_numpy()

    n_grid = list(range(N_STEP, min(n_total, N_CAP) + 1, N_STEP))
    if n_total not in n_grid:
        n_grid.append(n_total)

    scores_by_n = {}
    for n in n_grid:
        draw_scores = []
        for draw in range(N_REPEATS_SUBSAMPLE):
            rng = np.random.default_rng(abs(hash((session_id, n, draw))) % (2**31))
            unit_ids = rng.choice(all_unit_ids, size=n, replace=False) if n < n_total else all_unit_ids
            X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
            valid = ~np.isnan(X).any(axis=1)
            X_v = X[valid]
            col_std = np.nanstd(X_v, axis=0)
            X_v = X_v[:, col_std > 1e-6]
            Y_raw = Y_raw_full[valid]
            if len(X_v) < MIN_TRIALS_FOR_REGRESSION or X_v.shape[1] < 2:
                continue
            Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
            Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)
            n_1se = select_pls_components(X_v, Y, rng)
            Y_pred = _pooled_cv_predict(X_v, Y, n_1se, rng)
            scores = dict(
                r2=[np.nan if np.var(Y[:, k]) == 0 else
                    (1 - np.sum((Y[:, k] - Y_pred[:, k]) ** 2) / np.sum((Y[:, k] - Y[:, k].mean()) ** 2))
                    for k in range(Y.shape[1])],
                pearson=[pearsonr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])],
                spearman=[spearmanr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])],
            )
            draw_scores.append(scores)
        if not draw_scores:
            continue
        scores_by_n[n] = {
            metric: np.mean([d[metric] for d in draw_scores], axis=0) for metric in ("r2", "pearson", "spearman")
        }
        print(f"{mouse}/{session_id} N={n}: pearson={[round(v,3) for v in scores_by_n[n]['pearson']]}", flush=True)

    return dict(mouse=mouse, session_id=session_id, reward_group=reward_group, ok=True,
                n_total=n_total, scores_by_n=scores_by_n)


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    from concurrent.futures import ProcessPoolExecutor, as_completed

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    hm = hitmiss_session_list(sessions_tbl)
    learning = hm[hm["day_stage"] == "learning"]

    tasks = []
    for mouse in EXAMPLE_MICE:
        rows = learning[learning.subject_id == mouse]
        if len(rows) == 0:
            print(f"{mouse}: no learning-stage session, skipped")
            continue
        r = rows.iloc[0]
        tasks.append((mouse, r["session_id"], r.get("reward_group", "?"), SCRIPTS_DIR))
    print(f"{len(tasks)} example sessions", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), 10)) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("ok"):
                results.append(res)
    results.sort(key=lambda r: EXAMPLE_MICE.index(r["mouse"]))
    print(f"{len(results)}/{len(tasks)} sessions usable", flush=True)

    # --- Figure: pearson vs N, one panel per session, one line per target ---
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4.2), constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, res in zip(axes, results):
        ns = sorted(res["scores_by_n"].keys())
        for k, target in enumerate(TARGETS):
            vals = [res["scores_by_n"][n]["pearson"][k] for n in ns]
            ax.plot(ns, vals, marker="o", markersize=4, lw=1.3, color=TARGET_COLORS[target],
                     label=target.replace("_curve", ""))
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        ax.set_title(f"{res['mouse']} [{res['reward_group']}]\nn_total={res['n_total']}", fontsize=9.5)
        ax.set_xlabel("N units subsampled", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    axes[0].set_ylabel("test Pearson (pooled CV)", fontsize=9)
    fig.suptitle("PLS+1SE perfquant test Pearson vs N units (whole_brain, sensory window, real score only, no null)", fontsize=12)
    fig_path = OUT_DIR / "074_perfquant_nsubsample_pearson_vs_n.png"
    fig.savefig(fig_path, dpi=150)
    print(f"saved {fig_path.name}")
    print("DONE_074")


if __name__ == "__main__":
    main()
