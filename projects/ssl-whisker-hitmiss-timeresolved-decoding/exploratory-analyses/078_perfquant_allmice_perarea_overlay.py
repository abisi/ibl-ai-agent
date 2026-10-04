"""Single-trial curves, different AREAS overlaid on one panel per session,
across the full session pool (user: "Plot single trial curves for the
different areas" -> "Do all of this: report in a single figure" --
combining the overlay-on-one-panel option with the all-sessions-not-just-
5-examples option, as one figure per target).

Areas: the 8 already vetted as significant in `066`/`067` (>=15 sessions/
cohort, BH-FDR q<0.05 on at least one target/cohort) -- Frontal areas,
Hippocampus, Midbrain, Motor areas, Retrosplenial areas, Somatosensory-
whisker, Striatum, Thalamus -- plus whole_brain as reference. Same 98
good/moderate sessions as `065`'s pool (learning + expert).

Real prediction only, no null -- 9 overlaid lines per panel already
pushes readability; adding a null band per area would make every panel
illegible. `066`'s own per-area significance testing (`067`) is the
authoritative statistical claim; this figure is a qualitative single-
trial view of the same fits, at the full session-pool scale `068` never
covered (068 was 5 example sessions only).

`select_pls_1se_components` + pooled-CV recipe copied verbatim from `066`
(same estimator, so scores are comparable to `066`'s own numbers at
N=full).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
AREA_COL = "area_group"
AREAS = ["whole_brain", "Frontal areas", "Hippocampus", "Midbrain", "Motor areas",
         "Retrosplenial areas", "Somatosensory-whisker", "Striatum", "Thalamus"]
AREA_COLOR = {
    "whole_brain": "#000000", "Frontal areas": "#1f77b4", "Hippocampus": "#ff7f0e",
    "Midbrain": "#2ca02c", "Motor areas": "#d62728", "Retrosplenial areas": "#9467bd",
    "Somatosensory-whisker": "#8c564b", "Striatum": "#e377c2", "Thalamus": "#17becf",
}
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
LEARNING_CATEGORIES_KEEP = {"good", "moderate"}
DAY_STAGES = ["learning", "expert"]
N_COLS = 8


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


def process_one_cell(args: tuple) -> dict:
    """One (session, area) cell."""
    mouse, session_id, area, reward_group, day_stage, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    out = dict(mouse=mouse, session_id=session_id, area=area, reward_group=reward_group,
               day_stage=day_stage, ok=False)
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        return out
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
    out.update(ok=True, trial_index=trial_index, Y_pred=Y_pred, test_pearson=test_pearson)
    # whole_brain also carries Y_true (same for every area of this session -- only needed once)
    if area == "whole_brain":
        out["Y_true_full"] = targets_df[TARGETS].to_numpy()
        out["n_full"] = len(targets_df)
    return out


def main():
    import pickle as pkl

    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache_path = OUT_DIR / "078_perfquant_allmice_perarea_cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            results = pkl.load(f)
        print(f"loaded {len(results)} cells from cache ({cache_path.name})", flush=True)
    else:
        dataset_root = resolve_dataset_dir("ssl_ephys")
        sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
        hm = hitmiss_session_list(sessions_tbl)

        tasks = []
        for day_stage in DAY_STAGES:
            sub = hm[(hm["day_stage"] == day_stage) & (hm["learning_category"].isin(LEARNING_CATEGORIES_KEEP))]
            for _, r in sub.iterrows():
                for area in AREAS:
                    tasks.append((r["subject_id"], r["session_id"], area, r["reward_group"], day_stage, SCRIPTS_DIR))
        print(f"{len(tasks)} (session, area) cells", flush=True)

        results = []
        with ProcessPoolExecutor(max_workers=100) as ex:
            futures = {ex.submit(process_one_cell, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures)):
                res = fut.result()
                if res.get("ok"):
                    results.append(res)
                if (i + 1) % 100 == 0:
                    print(f"  {i + 1}/{len(tasks)} cells done", flush=True)
        print(f"{len(results)}/{len(tasks)} cells usable", flush=True)
        with open(cache_path, "wb") as f:
            pkl.dump(results, f)
        print(f"cached -> {cache_path.name}", flush=True)

    # Organize by session
    by_session = {}
    for r in results:
        by_session.setdefault(r["session_id"], {}).update({
            "mouse": r["mouse"], "reward_group": r["reward_group"], "day_stage": r["day_stage"],
            "areas": by_session.get(r["session_id"], {}).get("areas", {}),
        })
        by_session[r["session_id"]]["areas"][r["area"]] = r
        if r["area"] == "whole_brain" and "Y_true_full" in r:
            by_session[r["session_id"]]["Y_true_full"] = r["Y_true_full"]

    sessions_with_wb = [sid for sid, d in by_session.items() if "Y_true_full" in d]
    print(f"{len(sessions_with_wb)} sessions have a usable whole_brain fit (reference curve)", flush=True)

    # Sort by whole_brain mean test_pearson, descending
    def sort_key(sid):
        return np.mean(by_session[sid]["areas"]["whole_brain"]["test_pearson"])
    sessions_with_wb.sort(key=sort_key, reverse=True)

    n_sessions = len(sessions_with_wb)
    n_rows = int(np.ceil(n_sessions / N_COLS))
    for target_idx, target in enumerate(TARGETS):
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(2.4 * N_COLS, 2.4 * n_rows), constrained_layout=True)
        axes_flat = axes.flatten() if n_sessions > 1 else [axes]
        for i, sid in enumerate(sessions_with_wb):
            ax = axes_flat[i]
            d = by_session[sid]
            Y_true_full = d["Y_true_full"][:, target_idx]
            ax.plot(np.arange(len(Y_true_full)), Y_true_full, color="#000000", lw=1.3, alpha=0.85, label="true", zorder=5)
            for area in AREAS:
                r = d["areas"].get(area)
                if r is None:
                    continue
                ax.plot(r["trial_index"], r["Y_pred"][:, target_idx], color=AREA_COLOR[area],
                        lw=0.8 if area != "whole_brain" else 1.1, alpha=0.85 if area != "whole_brain" else 1.0,
                        label=area)
            ax.set_title(f"{d['mouse']} ({d['day_stage'][:1]})", fontsize=6.5)
            ax.tick_params(labelsize=5)
            ax.spines[["top", "right"]].set_visible(False)
        for j in range(len(sessions_with_wb), len(axes_flat)):
            axes_flat[j].axis("off")
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 1.015))
        fig.suptitle(f"{target} -- areas overlaid, all good/moderate sessions (n={n_sessions}), PLS+1SE sensory window",
                     fontsize=11, y=1.03)
        fig_path = OUT_DIR / f"078_perfquant_allmice_perarea_overlay_{target}.png"
        fig.savefig(fig_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {fig_path.name}")

    print("DONE_078")


if __name__ == "__main__":
    main()
