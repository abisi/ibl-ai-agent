"""Full-pool (88-session) real-vs-null validation for PLS (plain) and
PLS+1SE, at the true (lag=0) trial alignment -- complementing `057`'s
+/-20 trial lag sweep, which stays as-is (user 2026-09-20: "Keep the 20
trials for the lags. Implement the shift nulls distributions and compare
performance of real vs null, and report distributions of above-null
performances. Quantify R2, pearson and spearman both train and test.").

For each session and each of the two methods (component count fixed once
per session via `select_pls_components`, same as `057`):

- REAL test score: pooled-CV (n_repeats x KFold) out-of-fold predictions,
  scored with R2, Pearson AND Spearman (all three -- R2 is scale-sensitive
  and the one most likely to catch a numerical problem, Pearson/Spearman
  are scale/rank invariant and were the metrics `054` found unaffected by
  the SCALE_FLOOR bug).
- REAL train score: refit on all data, predict the same data (in-sample),
  same three metrics -- reported purely to quantify the train/test gap,
  never treated as a real number on its own.
- NULL distribution: N_SHUF_NULL=10 full retrain-under-linear-shift nulls
  (shift X relative to Y by a random 10-50% of session length, refit
  EVERYTHING from scratch with the SAME fixed n_components, matching the
  full-null convention used throughout this project, not the cheaper
  post-hoc-shift version) -- each shuffle scored the same way (R2,
  Pearson, Spearman) via its own pooled CV.
- above-null delta: real_test_metric - mean(null_metric) per session, per
  metric, per target, per method -- the actual quantity of interest ("how
  much does real performance clear the null, in absolute score units").
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

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
N_SHUF_NULL = 1000  # bumped from 10 -> 1000 (user 2026-09-20: "implement this version and 1000 null shuffles",
                     # following up on 061's per-session significance test on the 12 example sessions)
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
METHOD_COLORS = {"PLS+1SE": "#d62728"}  # plain PLS dropped -- user 2026-09-20: "Keep PLS SE"
METRICS = ["r2", "pearson", "spearman"]


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def select_pls_components(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator,
                           n_folds: int = 5, grid: tuple = COMPONENT_GRID) -> tuple[int, int]:
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
    n_plain = candidates[best_idx]
    threshold = mean_scores[best_idx] - se_scores[best_idx]
    n_1se = next(c for i, c in enumerate(candidates) if mean_scores[i] >= threshold)
    return n_plain, n_1se


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


def _score(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    from sklearn.metrics import r2_score
    n_targets = Y_true.shape[1]
    return dict(
        r2=[r2_score(Y_true[:, k], Y_pred[:, k]) for k in range(n_targets)],
        pearson=[pearsonr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
        spearman=[spearmanr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
    )


def evaluate_method(X: np.ndarray, Y: np.ndarray, n_components: int, rng: np.random.Generator) -> dict:
    # real test (pooled CV) + real train (in-sample)
    Y_test_pred = _pooled_cv_predict(X, Y, n_components, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS)
    Y_train_pred = _pls_fit_predict(X, Y, X, n_components)
    test_scores = _score(Y, Y_test_pred)
    train_scores = _score(Y, Y_train_pred)

    # full retrain-under-linear-shift null, same fixed n_components
    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    null_r2 = np.full((N_SHUF_NULL, Y.shape[1]), np.nan)
    null_pearson = np.full((N_SHUF_NULL, Y.shape[1]), np.nan)
    null_spearman = np.full((N_SHUF_NULL, Y.shape[1]), np.nan)
    for s in range(N_SHUF_NULL):
        shift = int(rng.integers(min_shift, max_shift + 1))
        if rng.random() < 0.5:
            X_shift, Y_shift = X[: n - shift], Y[shift:]
        else:
            X_shift, Y_shift = X[shift:], Y[: n - shift]
        Y_pred_null = _pooled_cv_predict(X_shift, Y_shift, n_components, rng, n_repeats=1, n_folds=N_FOLDS)
        null_scores = _score(Y_shift, Y_pred_null)
        null_r2[s] = null_scores["r2"]
        null_pearson[s] = null_scores["pearson"]
        null_spearman[s] = null_scores["spearman"]

    return dict(
        test=test_scores, train=train_scores, n_components=n_components,
        null=dict(r2=null_r2, pearson=null_pearson, spearman=null_spearman),
    )


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
    X = X[:, col_std > 1e-6]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        return out

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    _, n_1se = select_pls_components(X, Y, rng)
    res_1se = evaluate_method(X, Y, n_1se, rng)

    out.update(ok=True, n_trials=len(X), n_units=X.shape[1], methods={"PLS+1SE": res_1se})
    print(f"{mouse}/{session_id} [{reward_group}]: n={len(X)}, p={X.shape[1]}, "
          f"PLS+1SE(n_comp={n_1se}) test_pearson={[round(v, 3) for v in res_1se['test']['pearson']]} "
          f"null_pearson_mean={[round(v, 3) for v in np.nanmean(res_1se['null']['pearson'], axis=0)]}", flush=True)
    return out


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import hitmiss_session_list
    from concurrent.futures import ProcessPoolExecutor, as_completed

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
            res = fut.result()
            if res.get("ok"):
                results.append(res)
    print(f"{len(results)}/{len(tasks)} sessions usable", flush=True)

    method_names = ["PLS+1SE"]
    records = []
    for r in results:
        for method in method_names:
            m = r["methods"][method]
            for k, target in enumerate(TARGETS):
                rec = dict(mouse=r["mouse"], session_id=r["session_id"], reward_group=r["reward_group"],
                           method=method, target=target, n_components=m["n_components"],
                           n_trials=r["n_trials"], n_units=r["n_units"])
                for metric in METRICS:
                    test_val = m["test"][metric][k]
                    train_val = m["train"][metric][k]
                    null_vals = m["null"][metric][:, k]
                    null_mean = float(np.nanmean(null_vals))
                    null_std = float(np.nanstd(null_vals))
                    rec[f"test_{metric}"] = test_val
                    rec[f"train_{metric}"] = train_val
                    rec[f"null_{metric}_mean"] = null_mean
                    rec[f"null_{metric}_std"] = null_std
                    rec[f"above_null_{metric}"] = test_val - null_mean
                    rec[f"above_null1sd_{metric}"] = bool(test_val > null_mean + null_std)
                records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "058_perfquant_pls_nulldist_fullpool.csv", index=False)

    # --- summary print ---
    print("\n=== summary: mean test/train/null +/- above-null fraction, per method x target x metric ===")
    for method in method_names:
        for target in TARGETS:
            sub = df[(df.method == method) & (df.target == target)]
            for metric in METRICS:
                frac_above1sd = sub[f"above_null1sd_{metric}"].mean()
                print(f"  {method} / {target} / {metric}: test={sub[f'test_{metric}'].mean():.3f}, "
                      f"train={sub[f'train_{metric}'].mean():.3f}, null={sub[f'null_{metric}_mean'].mean():.3f}, "
                      f"above_null={sub[f'above_null_{metric}'].mean():.3f}, "
                      f"frac_above_null+1sd={frac_above1sd:.2f} (n={len(sub)})")

    # --- Figure 1: real (test) vs null distributions, per metric, faceted by target x method ---
    fig1, axes1 = plt.subplots(len(TARGETS), len(method_names), figsize=(5.2 * len(method_names), 3.6 * len(TARGETS)),
                                constrained_layout=True, squeeze=False)
    for row_i, target in enumerate(TARGETS):
        for col_i, method in enumerate(method_names):
            ax = axes1[row_i][col_i]
            sub = df[(df.method == method) & (df.target == target)]
            positions, labels, data, colors = [], [], [], []
            pos = 0
            for metric in METRICS:
                for kind, color in [("test", METHOD_COLORS[method]), ("null", "#999999")]:
                    col = f"{kind}_{metric}" if kind == "test" else f"null_{metric}_mean"
                    vals = sub[col].dropna().values
                    data.append(vals)
                    colors.append(color)
                    labels.append(f"{metric}\n{kind}")
                    positions.append(pos)
                    pos += 1
                pos += 0.6
            parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=False)
            for pc, color in zip(parts["bodies"], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=7)
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
            ax.set_title(f"{target} -- {method} (n={len(sub)})", fontsize=9)
            ax.set_ylabel("score", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    fig1_path = OUT_DIR / "058_perfquant_real_vs_null_distributions.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: distributions of ABOVE-NULL delta, per metric, faceted by target, methods overlaid ---
    fig2, axes2 = plt.subplots(1, len(TARGETS), figsize=(5.2 * len(TARGETS), 4.4), constrained_layout=True)
    for col_i, target in enumerate(TARGETS):
        ax = axes2[col_i]
        sub = df[df.target == target]
        positions, labels, data, colors = [], [], [], []
        pos = 0
        for metric in METRICS:
            for method in method_names:
                vals = sub[sub.method == method][f"above_null_{metric}"].dropna().values
                data.append(vals)
                colors.append(METHOD_COLORS[method])
                labels.append(f"{metric}\n{method}")
                positions.append(pos)
                pos += 1
            pos += 0.6
        parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.axhline(0, color="#888888", lw=0.8, linestyle="--")
        ax.set_title(target, fontsize=9)
        ax.set_ylabel("test metric - mean(null)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        for method in method_names:
            for metric in METRICS:
                vals = sub[sub.method == method][f"above_null_{metric}"].dropna().values
                if len(vals) > 5:
                    try:
                        wp = wilcoxon(vals).pvalue
                    except ValueError:
                        wp = float("nan")
                    print(f"  Wilcoxon (above-null != 0) {target}/{method}/{metric}: p={wp:.4g}, median={np.median(vals):.3f}")
    fig2_path = OUT_DIR / "058_perfquant_above_null_distributions.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    # --- Figure 3: train vs test, per method, faceted by target, all 3 metrics ---
    fig3, axes3 = plt.subplots(len(TARGETS), len(METRICS), figsize=(4.4 * len(METRICS), 3.4 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        for col_i, metric in enumerate(METRICS):
            ax = axes3[row_i][col_i]
            sub = df[df.target == target]
            x = np.arange(len(method_names))
            train_means = [sub[sub.method == m][f"train_{metric}"].mean() for m in method_names]
            test_means = [sub[sub.method == m][f"test_{metric}"].mean() for m in method_names]
            width = 0.35
            ax.bar(x - width / 2, train_means, width=width, color="#999999", label="train")
            ax.bar(x + width / 2, test_means, width=width, color=[METHOD_COLORS[m] for m in method_names], label="test")
            ax.set_xticks(x)
            ax.set_xticklabels(method_names, fontsize=8)
            ax.set_title(f"{target} -- {metric}", fontsize=9)
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=7, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig3_path = OUT_DIR / "058_perfquant_train_vs_test_allmetrics.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"saved {fig3_path.name}")

    print("DONE_058")


if __name__ == "__main__":
    main()
