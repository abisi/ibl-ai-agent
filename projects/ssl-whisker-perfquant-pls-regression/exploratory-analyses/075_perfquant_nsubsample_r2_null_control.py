"""Same neuron-count control as `074`, but with real null shuffles this
time (user request 2026-09-23: "Do the same perf quantification on the
PLS test data with R2 over null R2") -- `074` deliberately skipped null
at every N for cost reasons; this script adds it back, tracking
above-null R2 (and Pearson/Spearman, this project's standing
"always report all three" convention) vs N, not just raw test score.

Same 5 example sessions, same N grid (100,200,...,1000 + true full
count), same `N_REPEATS_SUBSAMPLE=3` per N. `058`'s exact recipe
otherwise: sensory window, floored scaler, PLS+1SE via
`select_pls_components`, full retrain-under-linear-shift null.

**`N_SHUF_NULL=30`, not `058`'s 1000** -- this sweep touches ~5 sessions x
~10 N-steps x 3 draws = ~150 (session, N, draw) cells, vs `058`'s single
cell per session; scaling the shuffle count down by roughly that same
factor (matching the reasoning `066`/`070` already used for their own
multiplicative designs) keeps this tractable while still giving a rough
above-null estimate at every N -- a scaling-shape diagnostic, not a
per-cell significance claim (nowhere near enough shuffles for a real
p-value at every N).

**Parallelized at (session, N) granularity, not per-session** -- unlike
`074` (5-way parallel, one session per worker), adding null shuffles
multiplies the per-cell cost enough that finer-grained parallelism
across haas's otherwise-idle 112 cores matters; each task redundantly
reloads its session's data (a few seconds), traded for much better wall-
clock via up to ~50-way concurrency instead of 5-way.
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
N_SHUF_NULL = 30  # see docstring -- reduced from 058's 1000 for this ~150-cell sweep
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
MIN_TRIALS_FOR_REGRESSION = 25
METRICS = ["r2", "pearson", "spearman"]
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


def _pooled_cv_predict(X, Y, n_components, rng, n_repeats, n_folds=N_FOLDS):
    from sklearn.model_selection import KFold
    Y_pred_accum = np.zeros_like(Y, dtype=float)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y, dtype=float)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            Y_pred_rep[te] = _pls_fit_predict(X[tr], Y[tr], X[te], n_components)
        Y_pred_accum += Y_pred_rep
    return Y_pred_accum / n_repeats


def _score(Y_true, Y_pred):
    from sklearn.metrics import r2_score
    n_targets = Y_true.shape[1]
    return dict(
        r2=[r2_score(Y_true[:, k], Y_pred[:, k]) for k in range(n_targets)],
        pearson=[pearsonr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
        spearman=[spearmanr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
    )


def evaluate_with_null(X, Y, n_components, rng):
    """058's evaluate_method, trimmed to test+null only (no train score --
    not needed for this control)."""
    Y_test_pred = _pooled_cv_predict(X, Y, n_components, rng, n_repeats=N_REPEATS_CV)
    test_scores = _score(Y, Y_test_pred)

    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    null = {m: np.full((N_SHUF_NULL, Y.shape[1]), np.nan) for m in METRICS}
    for s in range(N_SHUF_NULL):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        Y_pred_null = _pooled_cv_predict(X_shift, Y_shift, n_components, rng, n_repeats=1)
        null_scores = _score(Y_shift, Y_pred_null)
        for m in METRICS:
            null[m][s] = null_scores[m]
    return test_scores, null


def process_one_cell(args: tuple) -> dict:
    """One (session, N) cell -- averages N_REPEATS_SUBSAMPLE draws internally."""
    mouse, session_id, reward_group, n, scripts_dir, n_total = args
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
        return dict(mouse=mouse, session_id=session_id, n=n, ok=False)

    all_unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    Y_raw_full = targets_df[TARGETS].to_numpy()

    draw_test, draw_null = [], []
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
        test_scores, null = evaluate_with_null(X_v, Y, n_1se, rng)
        draw_test.append(test_scores)
        draw_null.append({m: np.nanmean(null[m], axis=0) for m in METRICS})  # mean over shuffles, per draw

    if not draw_test:
        return dict(mouse=mouse, session_id=session_id, n=n, ok=False)

    out = dict(mouse=mouse, session_id=session_id, reward_group=reward_group, n=n, n_total=n_total, ok=True)
    for m in METRICS:
        test_mean = np.mean([d[m] for d in draw_test], axis=0)
        null_mean = np.mean([d[m] for d in draw_null], axis=0)
        out[f"test_{m}"] = test_mean
        out[f"null_{m}"] = null_mean
        out[f"above_null_{m}"] = test_mean - null_mean
    print(f"{mouse}/{session_id} N={n}: test_r2={[round(v,3) for v in out['test_r2']]} "
          f"null_r2={[round(v,3) for v in out['null_r2']]} above_null_r2={[round(v,3) for v in out['above_null_r2']]}",
          flush=True)
    return out


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list, AREA_LABELS_PATH, add_whole_brain_column, area_units
    from concurrent.futures import ProcessPoolExecutor, as_completed

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    hm = hitmiss_session_list(sessions_tbl)
    learning = hm[hm["day_stage"] == "learning"]
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    sessions = []
    for mouse in EXAMPLE_MICE:
        rows = learning[learning.subject_id == mouse]
        if len(rows) == 0:
            print(f"{mouse}: no learning-stage session, skipped")
            continue
        r = rows.iloc[0]
        n_total = len(area_units(r["session_id"], "whole_brain", "All units", area_labels))
        sessions.append((mouse, r["session_id"], r.get("reward_group", "?"), n_total))
    print(f"{len(sessions)} example sessions", flush=True)

    tasks = []
    for mouse, session_id, reward_group, n_total in sessions:
        n_grid = list(range(N_STEP, min(n_total, N_CAP) + 1, N_STEP))
        if n_total not in n_grid:
            n_grid.append(n_total)
        for n in n_grid:
            tasks.append((mouse, session_id, reward_group, n, SCRIPTS_DIR, n_total))
    print(f"{len(tasks)} (session, N) cells, N_SHUF_NULL={N_SHUF_NULL}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=min(len(tasks), 50)) as ex:
        futures = {ex.submit(process_one_cell, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("ok"):
                results.append(res)
    print(f"{len(results)}/{len(tasks)} cells usable", flush=True)

    rows = []
    for res in results:
        for k, target in enumerate(TARGETS):
            row = dict(mouse=res["mouse"], session_id=res["session_id"], reward_group=res["reward_group"],
                       n=res["n"], n_total=res["n_total"], target=target)
            for m in METRICS:
                row[f"test_{m}"] = res[f"test_{m}"][k]
                row[f"null_{m}"] = res[f"null_{m}"][k]
                row[f"above_null_{m}"] = res[f"above_null_{m}"][k]
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["mouse", "n", "target"])
    df.to_csv(OUT_DIR / "075_perfquant_nsubsample_r2_null.csv", index=False)
    print(f"saved 075_perfquant_nsubsample_r2_null.csv")

    # --- Figure 1: test R2 (solid) vs null R2 (dashed), one row per target, one col per session ---
    mice_order = [m for m in EXAMPLE_MICE if m in df.mouse.unique()]
    fig1, axes1 = plt.subplots(len(TARGETS), len(mice_order), figsize=(3.8 * len(mice_order), 3.2 * len(TARGETS)),
                                constrained_layout=True, squeeze=False)
    for row_i, target in enumerate(TARGETS):
        for col_i, mouse in enumerate(mice_order):
            ax = axes1[row_i][col_i]
            sub = df[(df.mouse == mouse) & (df.target == target)].sort_values("n")
            color = TARGET_COLORS[target]
            ax.plot(sub.n, sub.test_r2, color=color, lw=1.6, marker="o", markersize=3.5, label="test R2")
            ax.plot(sub.n, sub.null_r2, color=color, lw=1.2, linestyle="--", alpha=0.6, marker="s", markersize=3, label="null R2")
            ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0:
                n_total = df[df.mouse == mouse].n_total.iloc[0]
                ax.set_title(f"{mouse}\nn_total={n_total}", fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"{target.replace('_curve','')}\nR2", fontsize=8.5)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=6.5, frameon=False)
            if row_i == len(TARGETS) - 1:
                ax.set_xlabel("N units", fontsize=8)
    fig1.suptitle("PLS+1SE: test R2 vs null R2 (mean of 30 shift-shuffles), by N units", fontsize=12)
    fig1_path = OUT_DIR / "075_perfquant_nsubsample_r2_vs_nullr2.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: above-null R2 vs N, one panel per session, lines per target (mirrors 074's layout) ---
    fig2, axes2 = plt.subplots(1, len(mice_order), figsize=(4.2 * len(mice_order), 4.2), constrained_layout=True, squeeze=False)
    axes2 = axes2[0]
    for ax, mouse in zip(axes2, mice_order):
        for target in TARGETS:
            sub = df[(df.mouse == mouse) & (df.target == target)].sort_values("n")
            ax.plot(sub.n, sub.above_null_r2, marker="o", markersize=4, lw=1.3, color=TARGET_COLORS[target],
                     label=target.replace("_curve", ""))
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        n_total = df[df.mouse == mouse].n_total.iloc[0]
        ax.set_title(f"{mouse}\nn_total={n_total}", fontsize=9.5)
        ax.set_xlabel("N units subsampled", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, frameon=False)
    axes2[0].set_ylabel("above-null R2\n(test - mean(null))", fontsize=9)
    fig2.suptitle(f"PLS+1SE above-null R2 vs N units (whole_brain, sensory window, N_SHUF_NULL={N_SHUF_NULL})", fontsize=12)
    fig2_path = OUT_DIR / "075_perfquant_nsubsample_abovenull_r2_vs_n.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")
    print("DONE_075")


if __name__ == "__main__":
    main()
