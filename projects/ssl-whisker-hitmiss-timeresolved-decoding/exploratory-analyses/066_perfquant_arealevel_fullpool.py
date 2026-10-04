"""Same canonical PLS+1SE recipe as `058`, but run PER `area_group`
instead of `whole_brain` (user 2026-09-21: "Then do the same per not
whole-brain but area_group, then for each cohort report which areas best
predict behaviour curves.").

Same conventions as `METHOD_perfquant_decoding.md`: sensory window,
floored scaler, PLS+1SE component selection (per session x area,
independently), pooled-CV test + in-sample train, full
retrain-under-linear-shift null scored against each shuffle's own shifted
target, R2/Pearson/Spearman. `N_SHUF_NULL=100` here (not 1000) -- with
~20 area_groups x up to 88 sessions, the full canonical 1000-shuffle
budget would be ~20x the cost of `058`'s single whole-brain run; 100
still gives ~1% p-value resolution, adequate for an exploratory
area-ranking screen.

Only (session, area) pairs with >= MIN_UNITS_PER_AREA units are run.
Areas are only included in the final "which area predicts best" report if
they have >= MIN_SESSIONS_FOR_REPORT sessions within a given cohort --
rare areas (e.g. Pons and medulla, Cortical subplate) do not have enough
coverage for a meaningful per-cohort ranking.
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
AREA_COL = "area_group"
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
MIN_SESSIONS_FOR_REPORT = 15
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 1
N_FOLDS = 5
N_SHUF_NULL = 100
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
METRICS = ["r2", "pearson", "spearman"]


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


def _score(Y_true, Y_pred):
    from sklearn.metrics import r2_score
    n_targets = Y_true.shape[1]
    return dict(
        r2=[r2_score(Y_true[:, k], Y_pred[:, k]) for k in range(n_targets)],
        pearson=[pearsonr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
        spearman=[spearmanr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
    )


def evaluate_method(X, Y, n_components, rng):
    Y_test_pred = _pooled_cv_predict(X, Y, n_components, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS)
    Y_train_pred = _pls_fit_predict(X, Y, X, n_components)
    test_scores, train_scores = _score(Y, Y_test_pred), _score(Y, Y_train_pred)

    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    null = {m: np.full((N_SHUF_NULL, Y.shape[1]), np.nan) for m in METRICS}
    for s in range(N_SHUF_NULL):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        Y_pred_null = _pooled_cv_predict(X_shift, Y_shift, n_components, rng, n_repeats=1, n_folds=N_FOLDS)
        sc = _score(Y_shift, Y_pred_null)
        for m in METRICS:
            null[m][s] = sc[m]
    return dict(test=test_scores, train=train_scores, null=null, n_components=n_components)


def process_one(args: tuple) -> dict:
    mouse, session_id, area, reward_group, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    import numpy as np  # noqa: F811
    import pandas as pd  # noqa: F811
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    out = dict(mouse=mouse, session_id=session_id, area=area, reward_group=reward_group, ok=False)
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

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
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        return out

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(f"{session_id}_{area}")) % (2**31))
    try:
        n_1se = select_pls_1se_components(X, Y, rng)
        ev = evaluate_method(X, Y, n_1se, rng)
    except (ValueError, np.linalg.LinAlgError) as e:
        # small-p areas (as low as MIN_UNITS_PER_AREA=5) can make PLS's internal
        # NIPALS deflation rank-deficient for some fold/shift combination -- a
        # degenerate case distinct from the SCALE_FLOOR scaler bug (052), since
        # it happens inside PLSRegression's own iteration, not the scaler. Treat
        # like any other "this (session, area) doesn't support a reliable fit"
        # exclusion rather than letting one bad combination kill the whole run.
        out["log"] = f"PLS numerical failure: {e}"
        return out

    out.update(ok=True, n_trials=len(X), n_units=X.shape[1], **ev)
    return out


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list, areas_with_enough_units, AREA_LABELS_PATH
    from concurrent.futures import ProcessPoolExecutor, as_completed

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    hm = hitmiss_session_list(sessions_tbl)
    learning = hm[hm["day_stage"] == "learning"]
    area_labels = pd.read_parquet(AREA_LABELS_PATH)

    tasks = []
    for _, r in learning.iterrows():
        areas = areas_with_enough_units(r["session_id"], AREA_COL, area_labels, min_units=MIN_UNITS_PER_AREA)
        for area in areas:
            if pd.isna(area):
                continue
            tasks.append((r["subject_id"], r["session_id"], area, r["reward_group"], SCRIPTS_DIR))
    print(f"{len(tasks)} (session, area) tasks across {learning['session_id'].nunique()} sessions "
          f"and {len(set(t[2] for t in tasks))} areas", flush=True)

    results = []
    n_numerical_failures = 0
    with ProcessPoolExecutor(max_workers=48) as ex:
        futures = {ex.submit(process_one, t): t for t in tasks}
        n_done = 0
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                print(f"  WORKER CRASH (unexpected, not the known PLS numerical case): {e}", flush=True)
                n_done += 1
                continue
            if res.get("log", "").startswith("PLS numerical failure"):
                n_numerical_failures += 1
            n_done += 1
            if res.get("ok"):
                results.append(res)
            if n_done % 100 == 0:
                print(f"  progress: {n_done}/{len(tasks)} tasks done, {len(results)} usable", flush=True)
    print(f"{len(results)}/{len(tasks)} (session, area) tasks usable "
          f"({n_numerical_failures} dropped for PLS numerical rank-deficiency)", flush=True)

    records = []
    for r in results:
        for k, target in enumerate(TARGETS):
            rec = dict(mouse=r["mouse"], session_id=r["session_id"], area=r["area"], reward_group=r["reward_group"],
                       target=target, n_components=r["n_components"], n_trials=r["n_trials"], n_units=r["n_units"])
            for metric in METRICS:
                test_val = r["test"][metric][k]
                train_val = r["train"][metric][k]
                null_vals = r["null"][metric][:, k]
                null_mean = float(np.nanmean(null_vals))
                rec[f"test_{metric}"] = test_val
                rec[f"train_{metric}"] = train_val
                rec[f"null_{metric}_mean"] = null_mean
                rec[f"above_null_{metric}"] = test_val - null_mean
            records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "066_perfquant_arealevel_fullpool.csv", index=False)

    # --- per-cohort, per-area ranking report ---
    print("\n=== which areas best predict behaviour curves, per cohort (mean above_null_pearson, >= "
          f"{MIN_SESSIONS_FOR_REPORT} sessions in that cohort) ===")
    summary_records = []
    for cohort in ("R+", "R-"):
        for target in TARGETS:
            sub = df[(df.reward_group == cohort) & (df.target == target)]
            grp = sub.groupby("area").agg(
                n_sessions=("session_id", "nunique"),
                mean_test_pearson=("test_pearson", "mean"),
                mean_null_pearson=("null_pearson_mean", "mean"),
                mean_above_null_pearson=("above_null_pearson", "mean"),
                mean_above_null_r2=("above_null_r2", "mean"),
            ).reset_index()
            grp = grp[grp["n_sessions"] >= MIN_SESSIONS_FOR_REPORT].sort_values("mean_above_null_pearson", ascending=False)
            grp["cohort"] = cohort
            grp["target"] = target
            summary_records.append(grp)
            print(f"\n  -- {cohort} / {target} --")
            for _, row in grp.iterrows():
                print(f"    {row['area']:<28} n={int(row['n_sessions']):>3}  test={row['mean_test_pearson']:.3f}  "
                      f"null={row['mean_null_pearson']:.3f}  above_null={row['mean_above_null_pearson']:.3f}")
    summary_df = pd.concat(summary_records, ignore_index=True)
    summary_df.to_csv(OUT_DIR / "066_perfquant_arealevel_area_ranking.csv", index=False)

    # --- heatmap figure: areas x targets, one panel per cohort ---
    areas_common = sorted(set(summary_df[summary_df.cohort == "R+"]["area"]) | set(summary_df[summary_df.cohort == "R-"]["area"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.42 * len(areas_common) + 2), constrained_layout=True)
    for ax, cohort in zip(axes, ("R+", "R-")):
        mat = np.full((len(areas_common), len(TARGETS)), np.nan)
        for i, area in enumerate(areas_common):
            for j, target in enumerate(TARGETS):
                row = summary_df[(summary_df.cohort == cohort) & (summary_df.area == area) & (summary_df.target == target)]
                if len(row):
                    mat[i, j] = row["mean_above_null_pearson"].iloc[0]
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.2)
        ax.set_yticks(range(len(areas_common)))
        ax.set_yticklabels(areas_common, fontsize=7)
        ax.set_xticks(range(len(TARGETS)))
        ax.set_xticklabels(TARGETS, fontsize=8, rotation=20, ha="right")
        ax.set_title(cohort, fontsize=10)
        fig.colorbar(im, ax=ax, label="mean above-null Pearson", shrink=0.7)
    fig_path = OUT_DIR / "066_perfquant_arealevel_cohort_heatmap.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nsaved {fig_path.name}")
    print("DONE_066")


if __name__ == "__main__":
    main()
