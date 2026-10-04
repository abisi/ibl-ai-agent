"""dPCA and CEBRA compared against the canonical PLS+1SE recipe, on the
same 5 example sessions used throughout `053`/`055`/`056` (user
2026-09-20: "Do dPCA and DEVBRA [CEBRA] and compare").

Both are adapted, not used in their most natural setting -- worth stating
plainly:

- **dPCA** is built for trial-averaged PSTHs across discrete, repeated
  task conditions (stimulus x decision x time), not continuous single-trial
  regression. Adaptation here: each continuous target is discretized into
  3 tertiles (used ONLY to define trial groups for the marginalization,
  never leaked into scoring), and the sensory window is split into
  `DPCA_N_TIMEBINS` sub-bins to give it the "time" axis it expects. The
  top "condition" (`c`) marginalization axis is fit on training trials
  only, projected onto held-out trials, and calibrated back to the
  continuous scale via a 1D linear fit (train-only). dPCA's own
  auto-regularization path hit a real shape bug in this dPCA package
  version on our data shape (confirmed independently) -- `regularizer=0`
  is used instead (no shrinkage), a legitimate but weaker-than-ideal
  setting.
- **CEBRA** (`model_architecture="offset1-model"`, no temporal receptive
  field, matching our one-vector-per-trial data -- there is no fine
  within-trial time structure for a >1 model to exploit) is used in its
  standard behavior-contrastive mode: `.fit(X, Y)` with the 3 continuous
  targets as the auxiliary variable, giving an 8-D embedding; a
  `KNeighborsRegressor` (the standard CEBRA decoding choice) reads out
  the continuous targets from the embedding.

Both are scored with the SAME pooled-CV / train / shift-null convention
as the canonical recipe (`METHOD_perfquant_decoding.md`), but with
`N_SHUF_NULL=100` instead of 1000 -- CEBRA's neural-network fit cost
makes 1000 shuffles x 5 sessions expensive for a first exploratory pass;
100 still gives a p-value resolution of ~0.01, adequate to see whether
either method is even in the right ballpark before deciding whether to
scale up.
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
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
N_REPEATS = 1
N_FOLDS = 5
N_SHUF_NULL = 100
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
METRICS = ["r2", "pearson", "spearman"]
EXAMPLE_MICE = ["AB158", "AB154", "AB092", "MH011", "MH022"]
METHOD_COLORS = {"PLS+1SE": "#d62728", "dPCA": "#1f77b4", "CEBRA": "#2ca02c"}

DPCA_N_TIMEBINS = 4
DPCA_N_CONDITIONS = 3
CEBRA_OUTPUT_DIM = 8
CEBRA_MAX_ITER = 500
CEBRA_KNN_K = 10
CEBRA_RUN_NULL = False  # user 2026-09-20: "Do not do shuffles for CEBRA yet" -- null is the
                         # expensive part (100 shuffles x 5 folds x ~2.6s/fit); skip for now
                         # to see real test/train performance first before committing the compute


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


def _score(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    from sklearn.metrics import r2_score
    n_targets = Y_true.shape[1]
    return dict(
        r2=[r2_score(Y_true[:, k], Y_pred[:, k]) for k in range(n_targets)],
        pearson=[pearsonr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
        spearman=[spearmanr(Y_true[:, k], Y_pred[:, k])[0] for k in range(n_targets)],
    )


def _shift_pair(X, Y, rng):
    n = len(Y)
    min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
    shift = int(rng.integers(min_shift, max_shift + 1))
    if rng.random() < 0.5:
        return X[: n - shift], Y[shift:]
    return X[shift:], Y[: n - shift]


# ---------------------------------------------------------------------------
# Method 1: PLS+1SE (canonical, reused unchanged)
# ---------------------------------------------------------------------------

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


def run_pls1se(X, Y, rng):
    from sklearn.model_selection import KFold
    n_components = select_pls_1se_components(X, Y, rng)

    def pooled_predict(Xa, Ya, n_repeats=N_REPEATS, n_folds=N_FOLDS):
        Y_pred_accum = np.zeros_like(Ya, dtype=float)
        for _ in range(n_repeats):
            seed = int(rng.integers(0, 2**31 - 1))
            Y_pred_rep = np.empty_like(Ya, dtype=float)
            for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(Xa):
                Y_pred_rep[te] = _pls_fit_predict(Xa[tr], Ya[tr], Xa[te], n_components)
            Y_pred_accum += Y_pred_rep
        return Y_pred_accum / n_repeats

    Y_test_pred = pooled_predict(X, Y)
    Y_train_pred = _pls_fit_predict(X, Y, X, n_components)
    test_scores, train_scores = _score(Y, Y_test_pred), _score(Y, Y_train_pred)

    null = {m: np.full((N_SHUF_NULL, Y.shape[1]), np.nan) for m in METRICS}
    for s in range(N_SHUF_NULL):
        X_shift, Y_shift = _shift_pair(X, Y, rng)
        Y_pred_null = pooled_predict(X_shift, Y_shift, n_repeats=1)
        sc = _score(Y_shift, Y_pred_null)
        for m in METRICS:
            null[m][s] = sc[m]
    return dict(test=test_scores, train=train_scores, null=null, extra=dict(n_components=n_components))


# ---------------------------------------------------------------------------
# Method 2: dPCA (adapted -- see module docstring)
# ---------------------------------------------------------------------------

def _dpca_condition_labels(y: np.ndarray, n_bins=DPCA_N_CONDITIONS):
    ranks = pd.qcut(y, n_bins, labels=False, duplicates="drop")
    return np.nan_to_num(ranks, nan=0).astype(int)


def _dpca_fit_axis(X_bins_train: np.ndarray, cond_train: np.ndarray):
    from dPCA import dPCA
    groups = [np.where(cond_train == c)[0] for c in np.unique(cond_train)]
    min_n = min(len(g) for g in groups)
    if min_n < 3 or len(groups) < 2:
        return None
    n_neurons, n_timebins = X_bins_train.shape[1], X_bins_train.shape[2]
    trialX = np.zeros((n_neurons, len(groups), min_n, n_timebins))
    for ci, g in enumerate(groups):
        trialX[:, ci, :, :] = X_bins_train[g[:min_n]].transpose(1, 0, 2)
    X_mean = trialX.mean(axis=2)
    dpca = dPCA.dPCA(labels="ct", regularizer=0)
    dpca.protect = ["t"]
    dpca.fit(X_mean, trialX=trialX)
    return dpca.P["c"][:, 0]


def _dpca_project(X_bins: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.einsum("n,tnb->tb", axis, X_bins).mean(axis=1)


def _dpca_fit_predict_target(X_bins_train, y_train, X_bins_test, cond_train):
    from sklearn.linear_model import LinearRegression
    axis = _dpca_fit_axis(X_bins_train, cond_train)
    if axis is None:
        return np.full(len(X_bins_test), np.nan)
    score_train = _dpca_project(X_bins_train, axis)
    score_test = _dpca_project(X_bins_test, axis)
    lr = LinearRegression().fit(score_train.reshape(-1, 1), y_train)
    return lr.predict(score_test.reshape(-1, 1))


def run_dpca(X_bins: np.ndarray, Y: np.ndarray, rng):
    from sklearn.model_selection import KFold
    n_targets = Y.shape[1]

    def pooled_predict(Xb, Ya):
        Y_pred = np.full_like(Ya, np.nan, dtype=float)
        seed = int(rng.integers(0, 2**31 - 1))
        for tr, te in KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(Xb):
            for k in range(n_targets):
                cond_tr = _dpca_condition_labels(Ya[tr, k])
                Y_pred[te, k] = _dpca_fit_predict_target(Xb[tr], Ya[tr, k], Xb[te], cond_tr)
        return Y_pred

    Y_test_pred = pooled_predict(X_bins, Y)
    Y_train_pred = np.column_stack([
        _dpca_fit_predict_target(X_bins, Y[:, k], X_bins, _dpca_condition_labels(Y[:, k])) for k in range(n_targets)
    ])
    valid_test = ~np.isnan(Y_test_pred).any(axis=1)
    test_scores = _score(Y[valid_test], Y_test_pred[valid_test])
    train_scores = _score(Y, Y_train_pred)

    null = {m: np.full((N_SHUF_NULL, n_targets), np.nan) for m in METRICS}
    for s in range(N_SHUF_NULL):
        X_shift, Y_shift = _shift_pair(X_bins, Y, rng)
        Y_pred_null = pooled_predict(X_shift, Y_shift)
        valid = ~np.isnan(Y_pred_null).any(axis=1)
        if valid.sum() < 10:
            continue
        sc = _score(Y_shift[valid], Y_pred_null[valid])
        for m in METRICS:
            null[m][s] = sc[m]
    return dict(test=test_scores, train=train_scores, null=null, extra=dict(n_valid_test=int(valid_test.sum())))


# ---------------------------------------------------------------------------
# Method 3: CEBRA (behavior-contrastive, offset1-model + kNN readout)
# ---------------------------------------------------------------------------

def _cebra_fit_predict(X_train, Y_train, X_test):
    from cebra import CEBRA
    from sklearn.neighbors import KNeighborsRegressor
    scaler = _floored_scaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)
    model = CEBRA(model_architecture="offset1-model", batch_size=min(128, max(8, len(X_train) // 2)),
                   output_dimension=CEBRA_OUTPUT_DIM, max_iterations=CEBRA_MAX_ITER,
                   device="cpu", verbose=False)
    model.fit(X_train_s, Y_train.astype(np.float32))
    emb_train, emb_test = model.transform(X_train_s), model.transform(X_test_s)
    knn = KNeighborsRegressor(n_neighbors=min(CEBRA_KNN_K, max(1, len(X_train) // 3)))
    knn.fit(emb_train, Y_train)
    return knn.predict(emb_test)


def run_cebra(X, Y, rng):
    from sklearn.model_selection import KFold

    def pooled_predict(Xa, Ya):
        Y_pred = np.empty_like(Ya, dtype=float)
        seed = int(rng.integers(0, 2**31 - 1))
        for tr, te in KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(Xa):
            Y_pred[te] = _cebra_fit_predict(Xa[tr], Ya[tr], Xa[te])
        return Y_pred

    Y_test_pred = pooled_predict(X, Y)
    Y_train_pred = _cebra_fit_predict(X, Y, X)
    test_scores, train_scores = _score(Y, Y_test_pred), _score(Y, Y_train_pred)

    null = {m: np.full((N_SHUF_NULL, Y.shape[1]), np.nan) for m in METRICS}
    if CEBRA_RUN_NULL:
        for s in range(N_SHUF_NULL):
            X_shift, Y_shift = _shift_pair(X, Y, rng)
            Y_pred_null = pooled_predict(X_shift, Y_shift)
            sc = _score(Y_shift, Y_pred_null)
            for m in METRICS:
                null[m][s] = sc[m]
    return dict(test=test_scores, train=train_scores, null=null, extra={})


# ---------------------------------------------------------------------------

def process_one(mouse: str) -> dict | None:
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

    sess = sessions_tbl[(sessions_tbl["subject_id"] == mouse) & (sessions_tbl["session_description"] == "whisker_0")
                         & (sessions_tbl["has_ephys"])]
    if len(sess) == 0:
        return None
    session_id = sess["session_id"].iloc[0]

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)

    X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
    bin_edges = np.linspace(SENSORY_WINDOW[0], SENSORY_WINDOW[1], DPCA_N_TIMEBINS + 1)
    subwindows = [(bin_edges[i], bin_edges[i + 1]) for i in range(DPCA_N_TIMEBINS)]
    X_bins_list = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, subwindows, dead_zone=DEAD_ZONE)
    X_bins = np.stack(X_bins_list, axis=2)  # (n_trials, n_neurons, n_timebins)

    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(X_bins).any(axis=(1, 2))
    X, X_bins = X[valid], X_bins[valid]
    col_std = np.nanstd(X, axis=0)
    keep = col_std > 1e-6
    X, X_bins = X[:, keep], X_bins[:, keep, :]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    res_pls = run_pls1se(X, Y, rng)
    print(f"  {mouse} [PLS+1SE] done: test_pearson={[round(v,3) for v in res_pls['test']['pearson']]}", flush=True)
    res_dpca = run_dpca(X_bins, Y, rng)
    print(f"  {mouse} [dPCA] done: test_pearson={[round(v,3) for v in res_dpca['test']['pearson']]}", flush=True)
    res_cebra = run_cebra(X, Y, rng)
    print(f"  {mouse} [CEBRA] done: test_pearson={[round(v,3) for v in res_cebra['test']['pearson']]}", flush=True)

    return dict(mouse=mouse, session_id=session_id, n_trials=len(X), n_units=X.shape[1],
                methods={"PLS+1SE": res_pls, "dPCA": res_dpca, "CEBRA": res_cebra})


def main():
    results = {}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=len(EXAMPLE_MICE)) as ex:
        futures = {ex.submit(process_one, m): m for m in EXAMPLE_MICE}
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                results[res["mouse"]] = res
                print(f"=== {res['mouse']} ALL METHODS DONE ===", flush=True)

    method_names = ["PLS+1SE", "dPCA", "CEBRA"]
    records = []
    for mouse, res in results.items():
        for name in method_names:
            m = res["methods"][name]
            for k, target in enumerate(TARGETS):
                rec = dict(mouse=mouse, method=name, target=target)
                for metric in METRICS:
                    test_val = m["test"][metric][k]
                    train_val = m["train"][metric][k]
                    null_vals = m["null"][metric][:, k]
                    null_mean = float(np.nanmean(null_vals))
                    null_std = float(np.nanstd(null_vals))
                    rec[f"test_{metric}"] = test_val
                    rec[f"train_{metric}"] = train_val
                    rec[f"null_{metric}_mean"] = null_mean
                    rec[f"above_null_{metric}"] = test_val - null_mean
                records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "064_perfquant_dpca_cebra_comparison.csv", index=False)

    print("\n=== summary: mean test/train/null pearson, per method x target ===")
    for name in method_names:
        for target in TARGETS:
            sub = df[(df.method == name) & (df.target == target)]
            print(f"  {name:>8} / {target:<18}: test={sub['test_pearson'].mean():.3f}, "
                  f"train={sub['train_pearson'].mean():.3f}, null={sub['null_pearson_mean'].mean():.3f}, "
                  f"above_null={sub['above_null_pearson'].mean():.3f} (n={len(sub)})")

    fig, axes = plt.subplots(1, len(TARGETS), figsize=(5.0 * len(TARGETS), 4.2), constrained_layout=True)
    for col_i, target in enumerate(TARGETS):
        ax = axes[col_i]
        sub = df[df.target == target]
        x = np.arange(len(EXAMPLE_MICE))
        width = 0.8 / len(method_names)
        for j, name in enumerate(method_names):
            vals = [sub[(sub.mouse == mouse) & (sub.method == name)]["test_pearson"].iloc[0]
                    if len(sub[(sub.mouse == mouse) & (sub.method == name)]) else np.nan for mouse in EXAMPLE_MICE]
            ax.bar(x + j * width, vals, width=width, color=METHOD_COLORS[name], label=name)
        ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
        ax.set_xticks(x + width * (len(method_names) - 1) / 2)
        ax.set_xticklabels(EXAMPLE_MICE, fontsize=8)
        ax.set_title(target, fontsize=9)
        ax.set_ylabel("test Pearson r")
        if col_i == 0:
            ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig_path = OUT_DIR / "064_perfquant_dpca_cebra_comparison_bars.png"
    fig.savefig(fig_path, dpi=150)
    print(f"saved {fig_path.name}")
    print("DONE_064")


if __name__ == "__main__":
    main()
