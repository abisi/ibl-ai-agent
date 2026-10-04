"""For each of the 5 example sessions, true-vs-predicted curves per
brain area (`area_group`) plus whole_brain as reference, with a
shift-null band overlay -- user 2026-09-21: "For single sessions, show
example predictions for the different areas."

Same PLS+1SE / sensory-window / floored-scaler recipe throughout. The
null band uses the same real-axis-remapping construction as `060`/`065`
(QUALITATIVE visual only, N_SHUF=20 -- not the properly-scored
significance test, which is `067`'s per-area Wilcoxon on `066`'s data).
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
AREA_COL = "area_group"
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
N_SHUF_BAND = 20
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
EXAMPLE_MICE = ["AB158", "AB154", "AB092", "MH011", "MH022"]


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def select_pls_1se_components(X, Y, rng, n_folds=5, grid=COMPONENT_GRID):
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


def null_predictions_per_trial(X, Y, n_components, rng, n_shuf=N_SHUF_BAND, n_folds=N_FOLDS):
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


def process_one(args: tuple) -> dict:
    mouse, session_id, area, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    out = dict(mouse=mouse, session_id=session_id, area=area, ok=False)
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return out
    if area == "whole_brain":
        area_labels = add_whole_brain_column(area_labels)
        unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    else:
        unit_ids = area_units(session_id, AREA_COL, area, area_labels)
    if len(unit_ids) < MIN_UNITS_PER_AREA:
        return out

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
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        return out

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(f"{session_id}_{area}")) % (2**31))
    try:
        n_1se = select_pls_1se_components(X, Y, rng)
        Y_pred = _pooled_cv_predict(X, Y, n_1se, rng)
    except (ValueError, np.linalg.LinAlgError) as e:
        out["log"] = f"PLS numerical failure: {e}"
        return out
    test_pearson = [pearsonr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])]

    try:
        null_preds = null_predictions_per_trial(X, Y, n_1se, rng)
    except (ValueError, np.linalg.LinAlgError):
        null_preds = np.full((N_SHUF_BAND, len(Y), Y.shape[1]), np.nan)
    null_mean = np.nanmean(null_preds, axis=0)
    null_std = np.nanstd(null_preds, axis=0)

    out.update(ok=True, n_trials=len(X), n_units=X.shape[1], n_components=n_1se, trial_index=trial_index,
               Y_true=Y, Y_pred=Y_pred, test_pearson=test_pearson, null_mean=null_mean, null_std=null_std)
    print(f"  {mouse}/{area}: n_units={X.shape[1]}, n_comp={n_1se}, test_pearson={[round(v, 3) for v in test_pearson]}", flush=True)
    return out


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import areas_with_enough_units, AREA_LABELS_PATH
    from concurrent.futures import ProcessPoolExecutor, as_completed

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

    session_by_mouse = {}
    tasks = []
    for mouse in EXAMPLE_MICE:
        sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                             & (sessions_tbl["has_ephys"])]
        if len(sess) == 0:
            continue
        session_id = sess["session_id"].iloc[0]
        session_by_mouse[mouse] = session_id
        areas = areas_with_enough_units(session_id, AREA_COL, area_labels, min_units=MIN_UNITS_PER_AREA)
        areas = [a for a in areas if pd.notna(a)] + ["whole_brain"]
        for area in areas:
            tasks.append((mouse, session_id, area, SCRIPTS_DIR))
        print(f"{mouse}/{session_id}: {len(areas)} areas (incl. whole_brain)", flush=True)

    results = {}
    with ProcessPoolExecutor(max_workers=40) as ex:
        futures = {ex.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("ok"):
                results.setdefault(res["mouse"], {})[res["area"]] = res
    print(f"done: {sum(len(v) for v in results.values())}/{len(tasks)} (mouse, area) tasks usable", flush=True)

    for mouse, area_results in results.items():
        areas_sorted = sorted(area_results.keys(), key=lambda a: (a != "whole_brain", -np.mean(area_results[a]["test_pearson"])))
        n_areas = len(areas_sorted)
        fig, axes = plt.subplots(n_areas, len(TARGETS), figsize=(4.2 * len(TARGETS), 2.0 * n_areas), constrained_layout=True)
        if n_areas == 1:
            axes = axes.reshape(1, -1)
        for row_i, area in enumerate(areas_sorted):
            res = area_results[area]
            for col_i, target in enumerate(TARGETS):
                ax = axes[row_i][col_i]
                t = res["trial_index"]
                ax.plot(t, res["Y_true"][:, col_i], color="#333333", lw=1.2, label="true")
                ax.plot(t, res["Y_pred"][:, col_i], color="#d62728", lw=1.0, alpha=0.9,
                        label=f"real (r={res['test_pearson'][col_i]:.2f})")
                null_mean, null_std = res["null_mean"][:, col_i], res["null_std"][:, col_i]
                ax.plot(t, null_mean, color="#888888", lw=0.8, alpha=0.8, label="null")
                ax.fill_between(t, null_mean - 1.96 * null_std, null_mean + 1.96 * null_std, color="#888888", alpha=0.2, linewidth=0)
                title = f"{area} (n_units={res['n_units']}) -- {target}" if col_i == 0 else target
                ax.set_title(title, fontsize=7.5)
                ax.tick_params(labelsize=6)
                ax.legend(fontsize=5.5, frameon=False)
                ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{mouse}: per-area predictions, PLS+1SE sensory window", fontsize=11)
        fig_path = OUT_DIR / f"068_perfquant_area_examples_{mouse}.png"
        fig.savefig(fig_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fig_path.name}")

    print("DONE_068")


if __name__ == "__main__":
    main()
