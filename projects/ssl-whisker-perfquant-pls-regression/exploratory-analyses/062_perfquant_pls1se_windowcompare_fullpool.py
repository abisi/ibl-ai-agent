"""Canonical PLS+1SE full-pool validation (`058`) + +/-20 trial lag profile
(`057`), run separately for the SENSORY window and the (wide) BASELINE
window, to compare which window better decodes the learning curves (user
2026-09-20: "Use baseline window and sensory. Keep the rest as it is,
Keep the 20 lag profiles. Now, I want to see if we can decode/regression
behavioural states ie the learning curves from neural data. Does that
model allow me to answer that? Can we optimize overall performance?").

Everything else matches `METHOD_perfquant_decoding.md`: PLS+1SE only
(component count selected once per session via the 1-SE rule), pooled-CV
test + in-sample train, 1000-shuffle full retrain-under-shift null (each
shuffle scored against its OWN shifted target), R2/Pearson/Spearman
throughout, plus the +/-20 trial lag sweep as a diagnostic (kept
per the user's instruction, not removed).

Usage: python 062_perfquant_pls1se_windowcompare_fullpool.py [sensory|baseline]
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
WIDE_BASELINE_WINDOW = (-2.000, -0.010)
SHORT_BASELINE_WINDOW = (-0.055, -0.010)
DEAD_ZONE = (-0.001, 0.004)
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 2
N_FOLDS = 5
N_SHUF_NULL = 1000
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
LAGS = list(range(-20, 21))
N_REPEATS_LAG = 2
METRICS = ["r2", "pearson", "spearman"]
COHORT_COLORS = {"R+": "#00B400", "R-": "#C800C8"}

FEATURE_MODE = sys.argv[1] if len(sys.argv) > 1 else "sensory"
assert FEATURE_MODE in ("sensory", "baseline", "corrected"), f"unknown FEATURE_MODE {FEATURE_MODE!r}"
WINDOW = {"sensory": SENSORY_WINDOW, "baseline": WIDE_BASELINE_WINDOW, "corrected": SENSORY_WINDOW}[FEATURE_MODE]
TAG = FEATURE_MODE


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


def _score(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    from sklearn.metrics import r2_score
    n_targets = Y_true.shape[1]
    return dict(
        r2=[r2_score(Y_true[:, k], Y_pred[:, k]) for k in range(n_targets)],
        pearson=[pearsonr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
        spearman=[spearmanr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
    )


def evaluate_method(X: np.ndarray, Y: np.ndarray, n_components: int, rng: np.random.Generator) -> dict:
    Y_test_pred = _pooled_cv_predict(X, Y, n_components, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS)
    Y_train_pred = _pls_fit_predict(X, Y, X, n_components)
    test_scores = _score(Y, Y_test_pred)
    train_scores = _score(Y, Y_train_pred)

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
        s_scores = _score(Y_shift, Y_pred_null)
        null_r2[s] = s_scores["r2"]
        null_pearson[s] = s_scores["pearson"]
        null_spearman[s] = s_scores["spearman"]

    return dict(test=test_scores, train=train_scores, n_components=n_components,
                null=dict(r2=null_r2, pearson=null_pearson, spearman=null_spearman))


def lag_profile(X: np.ndarray, Y: np.ndarray, n_components: int, rng: np.random.Generator,
                 lags: list = LAGS, n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS_LAG) -> np.ndarray:
    n = len(Y)
    out = np.full((len(lags), Y.shape[1]), np.nan)
    for li, lag in enumerate(lags):
        t_min, t_max = max(0, -lag), min(n, n - lag)
        if t_max - t_min < MIN_TRIALS_FOR_REGRESSION:
            continue
        X_lag = X[t_min:t_max]
        Y_lag = Y[t_min + lag: t_max + lag]
        Y_pred_accum = np.zeros_like(Y_lag)
        for _ in range(n_repeats):
            Y_pred_accum += _pooled_cv_predict(X_lag, Y_lag, n_components, rng, n_repeats=1, n_folds=n_folds)
        Y_pred = Y_pred_accum / n_repeats
        out[li] = [pearsonr(Y_lag[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])]
    return out


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
    if FEATURE_MODE == "corrected":
        sensory_mat, short_baseline_mat = sliding_bin_population_matrices(
            unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW, SHORT_BASELINE_WINDOW], dead_zone=DEAD_ZONE,
        )
        X = sensory_mat - short_baseline_mat
    else:
        X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [WINDOW], dead_zone=DEAD_ZONE)[0]
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
    n_1se = select_pls_1se_components(X, Y, rng)
    ev = evaluate_method(X, Y, n_1se, rng)
    lp = lag_profile(X, Y, n_1se, rng)

    out.update(ok=True, n_trials=len(X), n_units=X.shape[1], n_components=n_1se,
               methods={"PLS+1SE": ev}, lag_profile=lp)
    print(f"[{TAG}] {mouse}/{session_id} [{reward_group}]: n={len(X)}, p={X.shape[1]}, n_comp={n_1se}, "
          f"test_pearson={[round(v, 3) for v in ev['test']['pearson']]} "
          f"null_pearson_mean={[round(v, 3) for v in np.nanmean(ev['null']['pearson'], axis=0)]}", flush=True)
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
    print(f"[{TAG}] {len(tasks)} learning-stage has_ephys sessions", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=40) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("ok"):
                results.append(res)
    print(f"[{TAG}] {len(results)}/{len(tasks)} sessions usable", flush=True)

    method = "PLS+1SE"
    records = []
    for r in results:
        m = r["methods"][method]
        for k, target in enumerate(TARGETS):
            rec = dict(mouse=r["mouse"], session_id=r["session_id"], reward_group=r["reward_group"],
                       window=TAG, target=target, n_components=r["n_components"],
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
    df.to_csv(OUT_DIR / f"062_perfquant_pls1se_{TAG}_fullpool.csv", index=False)

    print(f"\n=== [{TAG}] summary: mean test/train/null +/- above-null fraction, per target x metric ===")
    for target in TARGETS:
        sub = df[df.target == target]
        for metric in METRICS:
            frac_above1sd = sub[f"above_null1sd_{metric}"].mean()
            print(f"  {target} / {metric}: test={sub[f'test_{metric}'].mean():.3f}, "
                  f"train={sub[f'train_{metric}'].mean():.3f}, null={sub[f'null_{metric}_mean'].mean():.3f}, "
                  f"above_null={sub[f'above_null_{metric}'].mean():.3f}, "
                  f"frac_above_null+1sd={frac_above1sd:.2f} (n={len(sub)})")
            vals = sub[f"above_null_{metric}"].dropna().values
            if len(vals) > 5:
                try:
                    wp = wilcoxon(vals).pvalue
                except ValueError:
                    wp = float("nan")
                print(f"    Wilcoxon (above-null != 0): p={wp:.4g}, median={np.median(vals):.3f}")

    # --- lag profile records ---
    lag_records = []
    for r in results:
        for li, lag in enumerate(LAGS):
            for k, target in enumerate(TARGETS):
                lag_records.append(dict(mouse=r["mouse"], session_id=r["session_id"], reward_group=r["reward_group"],
                                         lag=lag, target=target, pearson=r["lag_profile"][li, k]))
    lag_df = pd.DataFrame(lag_records)
    lag_df.to_csv(OUT_DIR / f"062_perfquant_pls1se_{TAG}_lagprofile.csv", index=False)

    # --- Figure 1: real vs null distributions ---
    fig1, axes1 = plt.subplots(1, len(TARGETS), figsize=(5.2 * len(TARGETS), 4.2), constrained_layout=True)
    for col_i, target in enumerate(TARGETS):
        ax = axes1[col_i]
        sub = df[df.target == target]
        positions, labels, data, colors = [], [], [], []
        pos = 0
        for metric in METRICS:
            for kind, color in [("test", "#d62728"), ("null", "#999999")]:
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
        ax.set_title(f"{target} -- {TAG} window (n={len(sub)})", fontsize=9)
        ax.set_ylabel("score", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig1_path = OUT_DIR / f"062_perfquant_pls1se_{TAG}_real_vs_null.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: above-null distributions ---
    fig2, axes2 = plt.subplots(1, len(TARGETS), figsize=(4.4 * len(TARGETS), 4.2), constrained_layout=True)
    for col_i, target in enumerate(TARGETS):
        ax = axes2[col_i]
        sub = df[df.target == target]
        data = [sub[f"above_null_{metric}"].dropna().values for metric in METRICS]
        parts = ax.violinplot(data, positions=range(len(METRICS)), showmeans=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor("#d62728")
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(METRICS, fontsize=8)
        ax.axhline(0, color="#888888", lw=0.8, linestyle="--")
        ax.set_title(f"{target} -- {TAG}", fontsize=9)
        ax.set_ylabel("test metric - mean(null)", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig2_path = OUT_DIR / f"062_perfquant_pls1se_{TAG}_above_null.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    # --- Figure 3: train vs test, all metrics ---
    fig3, axes3 = plt.subplots(len(TARGETS), len(METRICS), figsize=(4.0 * len(METRICS), 3.2 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        for col_i, metric in enumerate(METRICS):
            ax = axes3[row_i][col_i]
            sub = df[df.target == target]
            train_mean = sub[f"train_{metric}"].mean()
            test_mean = sub[f"test_{metric}"].mean()
            ax.bar([0, 1], [train_mean, test_mean], color=["#999999", "#d62728"])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["train", "test"], fontsize=8)
            ax.set_title(f"{target} -- {metric}", fontsize=9)
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
            ax.spines[["top", "right"]].set_visible(False)
    fig3_path = OUT_DIR / f"062_perfquant_pls1se_{TAG}_train_vs_test.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"saved {fig3_path.name}")

    # --- Figure 4: lag profile, mean +/- 95% CI, per target, cohort split ---
    fig4, axes4 = plt.subplots(1, len(TARGETS), figsize=(5.2 * len(TARGETS), 4.2), constrained_layout=True)
    for col_i, target in enumerate(TARGETS):
        ax = axes4[col_i]
        for cohort, color in COHORT_COLORS.items():
            sub = lag_df[(lag_df.target == target) & (lag_df.reward_group == cohort)]
            grp = sub.groupby("lag")["pearson"]
            mean = grp.mean()
            sem = grp.sem()
            ax.plot(mean.index, mean.values, color=color, lw=1.6, label=cohort)
            ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.2)
        ax.axvline(0, color="#888888", lw=0.8, linestyle=":")
        ax.axhline(0, color="#cccccc", lw=0.6)
        ax.set_title(f"{target} -- {TAG} window", fontsize=9)
        ax.set_xlabel("lag (trials, X relative to Y)", fontsize=8)
        ax.set_ylabel("mean test Pearson r", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig4_path = OUT_DIR / f"062_perfquant_pls1se_{TAG}_lagprofile.png"
    fig4.savefig(fig4_path, dpi=150)
    print(f"saved {fig4_path.name}")

    print(f"DONE_062_{TAG}")


if __name__ == "__main__":
    main()
