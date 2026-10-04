"""True-vs-predicted trial curves (PLS+1SE, sensory window, the canonical
method -- user 2026-09-20: "Ok, just keep the PLS method for now") for
EVERY good/moderate-learner session in the full pool, with a shift-null
mean+/-95%CI band overlaid on the real trial axis (user: "Plot session
predictions for all mice with the null. Only plot statistics learners
(learning_category is good or moderate).").

The null band uses the SAME remapping-onto-real-axis construction as
`060` -- it is a QUALITATIVE visual only ("does the model track fine
structure the null misses"), not a significance test. The valid,
properly-scored null (each shuffle scored against its own shifted target)
is the one behind every p-value in `058`/`061`/`RESULTS_RECAP_perfquant.md`;
this figure does not change or duplicate that. N_SHUF=20 here (not
1000) since this is purely for visualization across ~76 sessions.

**2026-09-23 update ("On this figures, also show the mean performance of
the null")**: each panel's title now also reports the null's own mean
Pearson -- sourced from `058`/`070`'s CSV (`null_pearson_mean`, the
VALID properly-scored null, joined on session_id x target), NOT derived
from this script's own real-axis-remapped null band above (that band is
qualitative/inflated, per the docstring paragraph above -- deliberately
not used for any number). A session missing from those CSVs (shouldn't
happen -- same session lists) falls back to "n/a".

**2026-09-23 update ("Update session figures")**: extended to also cover
`day_stage='expert'` (070's session pool), not just learning -- 070
didn't exist when this was first written. `learning_category` is a
per-MOUSE trait, populated for expert-stage rows too (checked: 18
good/6 moderate/5 bad out of 29 expert-stage sessions), so the same
good/moderate filter and recipe apply unchanged; only the session list
and output filenames (`_expert` suffix, this project's standing day-stage
convention) differ per stage.
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
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
N_SHUF_BAND = 20
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
LEARNING_CATEGORIES_KEEP = {"good", "moderate"}
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


def process_one_session(args: tuple) -> dict:
    mouse, session_id, learning_category, reward_group, scripts_dir, day_stage = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    out = dict(mouse=mouse, session_id=session_id, learning_category=learning_category,
               reward_group=reward_group, day_stage=day_stage, ok=False)
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
    X = X[:, col_std > 1e-6]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    trial_index = np.arange(len(targets_df))[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        return out

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    n_1se = select_pls_1se_components(X, Y, rng)
    Y_pred = _pooled_cv_predict(X, Y, n_1se, rng)
    test_pearson = [pearsonr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])]

    null_preds = null_predictions_per_trial(X, Y, n_1se, rng)
    null_mean = np.nanmean(null_preds, axis=0)
    null_std = np.nanstd(null_preds, axis=0)

    out.update(ok=True, n_trials=len(X), n_components=n_1se, trial_index=trial_index,
               Y_true=Y, Y_pred=Y_pred, test_pearson=test_pearson,
               null_mean=null_mean, null_std=null_std)
    print(f"  {mouse}/{session_id} [{learning_category}/{reward_group}]: n={len(X)}, n_comp={n_1se}, "
          f"test_pearson={[round(v, 3) for v in test_pearson]}", flush=True)
    return out


DAY_STAGES = ["learning", "expert"]
CACHE_PATH_TEMPLATE = "065_perfquant_allmice_cache.pkl"


def main():
    import pickle

    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache_path = OUT_DIR / CACHE_PATH_TEMPLATE
    # Cache the full per-session (Y_true, Y_pred, null_mean/std, trial_index)
    # arrays, not just the CSV summary -- added 2026-09-23 ("make them
    # squares") after realizing a purely cosmetic figure change (panel
    # aspect ratio) still required a full ~100-session PLS+null recompute
    # because nothing but the pearson summary was ever persisted. Any
    # future cosmetic-only change reads this instead of recomputing.
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        print(f"loaded {len(results)} sessions from cache ({cache_path.name}) -- delete it to force a recompute",
              flush=True)
    else:
        dataset_root = resolve_dataset_dir("ssl_ephys")
        sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
        hm = hitmiss_session_list(sessions_tbl)

        tasks = []
        for day_stage in DAY_STAGES:
            sub = hm[(hm["day_stage"] == day_stage) & (hm["learning_category"].isin(LEARNING_CATEGORIES_KEEP))]
            for _, r in sub.iterrows():
                tasks.append((r["subject_id"], r["session_id"], r["learning_category"], r["reward_group"],
                              SCRIPTS_DIR, day_stage))
            print(f"{len(sub)} good/moderate {day_stage}-stage sessions selected "
                  f"(out of {len(hm[hm['day_stage'] == day_stage])} total {day_stage}-stage)", flush=True)

        results = []
        with ProcessPoolExecutor(max_workers=40) as ex:
            futures = {ex.submit(process_one_session, t): t for t in tasks}
            for fut in as_completed(futures):
                res = fut.result()
                if res.get("ok"):
                    results.append(res)
        print(f"{len(results)}/{len(tasks)} sessions usable", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(results, f)
        print(f"cached {len(results)} sessions -> {cache_path.name}", flush=True)

    # Valid null Pearson lookup, keyed (session_id, target) -- from 058/070's
    # properly-scored null (each shuffle scored against its OWN shifted
    # target), NOT this script's own qualitative real-axis-remapped band.
    null_lookup = {}
    for csv_name in ("058_perfquant_pls_nulldist_fullpool.csv", "070_perfquant_expert_pls_nulldist_fullpool.csv"):
        csv_path = OUT_DIR / csv_name
        if not csv_path.exists():
            print(f"WARNING: {csv_name} not found, null-performance annotation will show n/a for its sessions")
            continue
        d = pd.read_csv(csv_path)
        for _, r in d.iterrows():
            null_lookup[(r["session_id"], r["target"])] = r["null_pearson_mean"]

    for day_stage in DAY_STAGES:
        stage_results = [r for r in results if r["day_stage"] == day_stage]
        if not stage_results:
            print(f"{day_stage}: no usable sessions, skipping figures")
            continue
        stage_results.sort(key=lambda r: np.mean(r["test_pearson"]), reverse=True)
        suffix = "" if day_stage == "learning" else "_expert"

        n_sessions = len(stage_results)
        n_rows = int(np.ceil(n_sessions / N_COLS))
        for target_idx, target in enumerate(TARGETS):
            # Square panels (user 2026-09-23: "make them squares") -- square
            # figsize per panel (was 2.4x1.8, non-square) plus set_box_aspect(1)
            # per axis so the box stays square regardless of each session's
            # own x/y data range (same convention as 046's _paired_panel).
            fig, axes = plt.subplots(n_rows, N_COLS, figsize=(2.2 * N_COLS, 2.2 * n_rows), constrained_layout=True)
            axes_flat = axes.flatten() if n_sessions > 1 else [axes]
            for i, res in enumerate(stage_results):
                ax = axes_flat[i]
                t = res["trial_index"]
                ax.plot(t, res["Y_true"][:, target_idx], color="#333333", lw=1.0, label="true")
                ax.plot(t, res["Y_pred"][:, target_idx], color="#d62728", lw=0.9, alpha=0.9, label="real")
                null_mean = res["null_mean"][:, target_idx]
                null_std = res["null_std"][:, target_idx]
                ax.plot(t, null_mean, color="#888888", lw=0.8, alpha=0.8, label="null")
                ax.fill_between(t, null_mean - 1.96 * null_std, null_mean + 1.96 * null_std, color="#888888", alpha=0.2, linewidth=0)
                null_r = null_lookup.get((res["session_id"], target))
                null_str = f"{null_r:.2f}" if null_r is not None else "n/a"
                ax.set_title(f"{res['mouse']} ({res['learning_category']}) r={res['test_pearson'][target_idx]:.2f} "
                             f"(null={null_str})", fontsize=6)
                ax.tick_params(labelsize=5)
                ax.set_box_aspect(1)
                ax.spines[["top", "right"]].set_visible(False)
            for j in range(len(stage_results), len(axes_flat)):
                axes_flat[j].axis("off")
            handles, labels = axes_flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.01))
            fig.suptitle(f"{target} -- all good/moderate {day_stage} learners (n={n_sessions}), PLS+1SE sensory window",
                         fontsize=11, y=1.02)
            fig_path = OUT_DIR / f"065_perfquant_allmice_predictions_with_null_{target}{suffix}.png"
            fig.savefig(fig_path, dpi=110, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {fig_path.name}")

    records = []
    for res in results:
        for k, target in enumerate(TARGETS):
            records.append(dict(mouse=res["mouse"], session_id=res["session_id"], day_stage=res["day_stage"],
                                 learning_category=res["learning_category"], reward_group=res["reward_group"],
                                 n_trials=res["n_trials"], n_components=res["n_components"],
                                 target=target, test_pearson=res["test_pearson"][k]))
    pd.DataFrame(records).to_csv(OUT_DIR / "065_perfquant_allmice_predictions_with_null.csv", index=False)
    print("DONE_065")


if __name__ == "__main__":
    main()
