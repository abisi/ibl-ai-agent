"""Regression-method comparison for the single-animal, per-neuron
perfquant regression (user request 2026-09-20: "On few test sessions, do
PLS and maybe compare L1, L2 and Elastic net. Plot results for these
sessions, a both curves and barplots.").

Four methods, all on the SAME whole_brain, SENSORY_WINDOW (5-50ms
post-stim) per-neuron features as `052`/`053`, same 5 example sessions
`053` used (AB158, AB154, AB092, MH011, MH022), same standardized-target
convention:
- Ridge (L2) -- the method used throughout `052`-`054`.
- Lasso (L1) -- implicit feature selection, may help given p>>n.
- Elastic Net (L1+L2) -- a compromise between the two.
- PLS (partial least squares) -- supervised dimensionality reduction,
  proposed as a better-motivated alternative to plain PCA (PCA keeps
  high-VARIANCE directions in X; PLS keeps directions that also covary
  with Y, so it shouldn't discard a real but low-variance signal the way
  PCA risks doing).

All four use the SAME `SCALE_FLOOR`-protected scaling as `052`/`053`
(manually applied before PLS too, with its own internal scaling turned
off, since sklearn's `PLSRegression` has no floor option and would be
vulnerable to the identical near-constant-within-fold-column blowup
`052`/`053` hit and fixed).

No null control here -- this is a method-vs-method comparison on the
same real data, not another significance test (kept the compute small
given "on few test sessions").

Two figures:
1. True-vs-predicted trial sequences, all 4 methods overlaid, per
   session x target (matches `053`'s example-figure style).
2. Bar chart of test Pearson r, grouped by session, one bar per method,
   faceted by target.
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
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
SCALE_FLOOR = 1e-3  # see 052/053 -- prevents the near-constant-within-fold-column StandardScaler blowup

EXAMPLE_MICE = ["AB158", "AB154", "AB092", "MH011", "MH022"]  # same set 053 used

METHOD_COLORS = {"Ridge (L2)": "#1f77b4", "Lasso (L1)": "#ff7f0e", "ElasticNet": "#2ca02c", "PLS": "#d62728"}


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def _fit_predict_oof(X: np.ndarray, Y: np.ndarray, make_model, param_grid: list, rng: np.random.Generator,
                      n_repeats: int = N_REPEATS, n_folds: int = 5):
    """Generic pooled-CV OOF prediction with inner hyperparameter search
    (mirrors 052/053's convention): for each candidate in `param_grid`,
    score mean R^2 across heads on one KFold partition; keep the best;
    then average OOF predictions across `n_repeats` independent
    partitions at that fixed hyperparameter."""
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    best_param, best_score = param_grid[0], -np.inf
    for param in param_grid:
        Y_pred = np.empty_like(Y)
        for tr, te in splits:
            model = make_model(param)
            model.fit(X[tr], Y[tr])
            Y_pred[te] = np.asarray(model.predict(X[te])).reshape(len(te), -1)
        score = np.mean([r2_score(Y[:, k], Y_pred[:, k]) for k in range(Y.shape[1])])
        if score > best_score:
            best_score, best_param = score, param

    Y_pred_accum = np.zeros_like(Y)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            model = make_model(best_param)
            model.fit(X[tr], Y[tr])
            Y_pred_rep[te] = np.asarray(model.predict(X[te])).reshape(len(te), -1)
        Y_pred_accum += Y_pred_rep
    Y_pred = Y_pred_accum / n_repeats
    pearsons = [pearsonr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])]
    r2s = [r2_score(Y[:, k], Y_pred[:, k]) for k in range(Y.shape[1])]
    spearmans = [spearmanr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])]

    # TRAIN (in-sample) score at the same chosen hyperparameter -- user
    # 2026-09-20 ("Why not PLS?"): PLS's whole point was fewer effective
    # parameters (as few as 2-30 components here, vs Ridge's implicit
    # ~600-2400), which should show up as a SMALLER train-test gap even
    # if test performance ties Ridge -- a real, meaningful difference
    # (a far less memorized model) this script's first pass never checked.
    model_train = make_model(best_param)
    model_train.fit(X, Y)
    Y_train_pred = np.asarray(model_train.predict(X)).reshape(len(Y), -1)
    train_pearsons = [pearsonr(Y[:, k], Y_train_pred[:, k])[0] for k in range(Y.shape[1])]
    train_r2s = [r2_score(Y[:, k], Y_train_pred[:, k]) for k in range(Y.shape[1])]

    return Y_pred, dict(pearson=pearsons, r2=r2s, spearman=spearmans, best_param=best_param,
                         train_pearson=train_pearsons, train_r2=train_r2s)


def run_ridge(X, Y, rng):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    return _fit_predict_oof(X, Y, lambda a: make_pipeline(_floored_scaler(), Ridge(alpha=a)), grid, rng)


def run_lasso(X, Y, rng):
    from sklearn.linear_model import Lasso
    from sklearn.pipeline import make_pipeline
    grid = [1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
    return _fit_predict_oof(X, Y, lambda a: make_pipeline(_floored_scaler(), Lasso(alpha=a, max_iter=5000)), grid, rng)


def run_elasticnet(X, Y, rng):
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import make_pipeline
    grid = [(a, l1) for a in [1e-3, 1e-2, 1e-1, 5e-1, 1] for l1 in [0.1, 0.5, 0.9]]
    return _fit_predict_oof(X, Y, lambda p: make_pipeline(_floored_scaler(), ElasticNet(alpha=p[0], l1_ratio=p[1], max_iter=5000)), grid, rng)


def run_pls(X, Y, rng):
    from sklearn.cross_decomposition import PLSRegression

    def make_model(n_components):
        scaler = _floored_scaler()

        class _PLSWrapped:
            def fit(self, Xtr, Ytr):
                scaler.fit(Xtr)
                Xs = scaler.transform(Xtr)
                self.pls = PLSRegression(n_components=n_components, scale=False)
                self.pls.fit(Xs, Ytr)
                return self

            def predict(self, Xte):
                return self.pls.predict(scaler.transform(Xte))
        return _PLSWrapped()

    max_comp = max(1, min(30, X.shape[1] - 1, int(X.shape[0] * 0.6)))
    grid = sorted(set(min(c, max_comp) for c in [2, 5, 10, 20, 30] if c <= max_comp)) or [max_comp]
    return _fit_predict_oof(X, Y, make_model, grid, rng)


METHODS = {"Ridge (L2)": run_ridge, "Lasso (L1)": run_lasso, "ElasticNet": run_elasticnet, "PLS": run_pls}


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
        Y_pred, metrics = fn(X, Y, rng)
        out["methods"][name] = dict(Y_pred=Y_pred, **metrics)
        print(f"  {mouse} [{name}]: n={len(X)}, p={X.shape[1]}, param={metrics['best_param']}, "
              f"test_pearson={[round(p, 3) for p in metrics['pearson']]}, "
              f"train_pearson={[round(p, 3) for p in metrics['train_pearson']]}", flush=True)
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
                r = m["pearson"][row_i]
                ax.plot(res["trial_index"], m["Y_pred"][:, row_i], color=METHOD_COLORS[name], lw=1.0, alpha=0.8,
                         label=f"{name} (r={r:.2f})")
            ax.set_title(f"{mouse}: {target}", fontsize=9)
            ax.set_xlabel("trial index", fontsize=8)
            ax.set_ylabel("z-scored value", fontsize=8)
            ax.legend(fontsize=6, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig1_path = OUT_DIR / "055_perfquant_method_comparison_curves.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: bar chart of test Pearson r, grouped by session, per method, faceted by target ---
    records = []
    for mouse, res in results.items():
        for name, m in res["methods"].items():
            for k, target in enumerate(TARGETS):
                records.append(dict(mouse=mouse, method=name, target=target, pearson=m["pearson"][k], r2=m["r2"][k],
                                     train_pearson=m["train_pearson"][k], train_r2=m["train_r2"][k],
                                     best_param=str(m["best_param"])))
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "055_perfquant_method_comparison.csv", index=False)

    fig2, axes2 = plt.subplots(len(TARGETS), 1, figsize=(10, 3.6 * len(TARGETS)), constrained_layout=True)
    mice_present = [m for m in EXAMPLE_MICE if m in results]
    method_names = list(METHODS.keys())
    bar_width = 0.8 / len(method_names)
    x_base = np.arange(len(mice_present))
    for row_i, target in enumerate(TARGETS):
        ax = axes2[row_i]
        for j, name in enumerate(method_names):
            vals = [df[(df.mouse == mouse) & (df.method == name) & (df.target == target)]["pearson"].iloc[0] for mouse in mice_present]
            ax.bar(x_base + j * bar_width, vals, width=bar_width, color=METHOD_COLORS[name], label=name)
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        ax.set_xticks(x_base + bar_width * (len(method_names) - 1) / 2)
        ax.set_xticklabels(mice_present)
        ax.set_ylabel("test Pearson r")
        ax.set_title(target, fontsize=10)
        if row_i == 0:
            ax.legend(fontsize=8, frameon=False, ncol=len(method_names))
        ax.spines[["top", "right"]].set_visible(False)
    fig2_path = OUT_DIR / "055_perfquant_method_comparison_bars.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    # --- Figure 3: TRAIN vs TEST gap per method -- user 2026-09-20 ("Why not
    # PLS?"): the whole point of PLS/sparser methods was fewer effective
    # parameters, which should show up as a SMALLER train-test gap even
    # at matched test performance -- this is what actually answers that
    # question, not the test-only bar chart above.
    fig3, axes3 = plt.subplots(len(TARGETS), 1, figsize=(10, 3.6 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        ax = axes3[row_i]
        sub = df[df.target == target]
        train_means = [sub[sub.method == name]["train_pearson"].mean() for name in method_names]
        test_means = [sub[sub.method == name]["pearson"].mean() for name in method_names]
        x = np.arange(len(method_names))
        ax.bar(x - 0.2, train_means, width=0.4, color="#888888", label="train (in-sample)")
        ax.bar(x + 0.2, test_means, width=0.4, color=[METHOD_COLORS[n] for n in method_names], label="test (OOF)")
        for xi, (tr, te) in enumerate(zip(train_means, test_means)):
            ax.text(xi, max(tr, te) + 0.03, f"gap={tr-te:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names)
        ax.set_ylim(0, 1.15)
        ax.axhline(0, color="#888888", lw=0.8, linestyle=":")
        ax.set_ylabel("mean Pearson r")
        ax.set_title(f"{target}: train vs test (overfitting gap)", fontsize=10)
        if row_i == 0:
            ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig3_path = OUT_DIR / "055_perfquant_method_comparison_traintest_gap.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"saved {fig3_path.name}")

    print("\n=== mean test Pearson r per method, across sessions ===")
    for target in TARGETS:
        sub = df[df.target == target]
        for name in method_names:
            tr = sub[sub.method == name]["train_pearson"].mean()
            te = sub[sub.method == name]["pearson"].mean()
            print(f"  {target} / {name}: mean test pearson={te:.3f}, mean train pearson={tr:.3f}, gap={tr - te:.3f}")


if __name__ == "__main__":
    main()
