"""Proper significance test for the same 12 example sessions from `060`
(best to worst, PLS+1SE) using N_SHUF_NULL=1000 shift-null shuffles per
session (user 2026-09-20: "For these sessions, do 1000 shuffles and check
for significance").

Unlike `060`'s null band (which remapped null predictions onto the REAL
trial axis and scored them against the real Y -- flagged there as an
invalid null because it inherited a boost from the behavioral curves'
long autocorrelation), each shuffle here is scored against ITS OWN shifted
target `Y_shift` -- the same valid convention `058` used for the full
88-session pool, just with 100x more shuffles (1000 vs 10) so a real
per-session empirical p-value can be computed instead of only a mean+/-std
comparison.

Empirical one-sided p-value per session x target x metric:
    p = (1 + #{null >= real}) / (1 + N_SHUF_NULL)
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
N_REPEATS = 2
N_FOLDS = 5
N_SHUF_NULL = 1000
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5
METRICS = ["r2", "pearson", "spearman"]

# same 12 sessions selected in `060` (best -> worst by PLS+1SE mean test Pearson)
EXAMPLE_SESSIONS = [
    ("MH070", "MH070_20260121_140848", 1),
    ("AB087", "AB087_20231017_141901", 9),
    ("MH028", "MH028_20250501_104058", 17),
    ("AB164", "AB164_20250422_115457", 25),
    ("MH037", "MH037_20250524_143522", 33),
    ("MH039", "MH039_20250525_112720", 41),
    ("AB119", "AB119_20240731_102619", 48),
    ("AB142", "AB142_20241128_113227", 56),
    ("AB104", "AB104_20240313_145433", 64),
    ("AB094", "AB094_20231211_112445", 72),
    ("MH027", "MH027_20250422_111013", 80),
    ("MH036", "MH036_20250515_111838", 88),
]


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


def process_one_session(args: tuple) -> dict:
    mouse, session_id, rank, scripts_dir = args
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
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)
    X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [SENSORY_WINDOW], dead_zone=DEAD_ZONE)[0]
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    col_std = np.nanstd(X, axis=0)
    X = X[:, col_std > 1e-6]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y = (Y_raw - Y_mean) / np.where(Y_std > 0, Y_std, 1.0)

    rng = np.random.default_rng(abs(hash(session_id)) % (2**31))
    n_1se = select_pls_1se_components(X, Y, rng)
    Y_test_pred = _pooled_cv_predict(X, Y, n_1se, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS)
    real_scores = _score(Y, Y_test_pred)

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
        Y_pred_null = _pooled_cv_predict(X_shift, Y_shift, n_1se, rng, n_repeats=1, n_folds=N_FOLDS)
        s_scores = _score(Y_shift, Y_pred_null)
        null_r2[s] = s_scores["r2"]
        null_pearson[s] = s_scores["pearson"]
        null_spearman[s] = s_scores["spearman"]

    null_by_metric = dict(r2=null_r2, pearson=null_pearson, spearman=null_spearman)
    p_values = {}
    for metric in METRICS:
        null_vals = null_by_metric[metric]
        p_values[metric] = [
            (1 + np.sum(null_vals[:, k] >= real_scores[metric][k])) / (1 + N_SHUF_NULL)
            for k in range(Y.shape[1])
        ]

    print(f"{mouse}/{session_id} (rank {rank}/88, n_comp={n_1se}, n={n}): " +
          " | ".join(f"{target}: " + ", ".join(
              f"{metric} real={real_scores[metric][k]:.3f} null={np.nanmean(null_by_metric[metric][:, k]):.3f} p={p_values[metric][k]:.4f}"
              for metric in METRICS) for k, target in enumerate(TARGETS)), flush=True)

    return dict(mouse=mouse, session_id=session_id, rank=rank, n_components=n_1se, n_trials=n,
                real=real_scores, null=null_by_metric, p_values=p_values)


def main():
    tasks = [(mouse, sid, rank, SCRIPTS_DIR) for mouse, sid, rank in EXAMPLE_SESSIONS]
    results = {}
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=len(tasks)) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            results[res["session_id"]] = res

    records = []
    for mouse, sid, rank in EXAMPLE_SESSIONS:
        res = results[sid]
        for k, target in enumerate(TARGETS):
            rec = dict(mouse=mouse, session_id=sid, rank=rank, n_components=res["n_components"], n_trials=res["n_trials"], target=target)
            for metric in METRICS:
                rec[f"real_{metric}"] = res["real"][metric][k]
                rec[f"null_{metric}_mean"] = float(np.nanmean(res["null"][metric][:, k]))
                rec[f"null_{metric}_std"] = float(np.nanstd(res["null"][metric][:, k]))
                rec[f"p_{metric}"] = res["p_values"][metric][k]
                rec[f"sig_{metric}_p05"] = res["p_values"][metric][k] < 0.05
            records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "061_perfquant_pls1se_shufflesignificance.csv", index=False)

    print("\n=== significance summary (1000 shuffles, p<0.05) ===")
    for metric in METRICS:
        n_sig = df[f"sig_{metric}_p05"].sum()
        print(f"  {metric}: {n_sig}/{len(df)} (session x target) pairs significant at p<0.05")
        for _, row in df.iterrows():
            flag = "*" if row[f"sig_{metric}_p05"] else " "
            print(f"    {flag} rank {row['rank']:>2}/88 {row['mouse']:>6} {row['target']:<18} "
                  f"real={row[f'real_{metric}']:.3f} null={row[f'null_{metric}_mean']:.3f} p={row[f'p_{metric}']:.4f}")

    # --- Figure: null histogram per session x target, real value marked, p annotated (Pearson) ---
    fig, axes = plt.subplots(len(EXAMPLE_SESSIONS), len(TARGETS),
                              figsize=(4.0 * len(TARGETS), 2.0 * len(EXAMPLE_SESSIONS)), constrained_layout=True)
    for row_i, (mouse, sid, rank) in enumerate(EXAMPLE_SESSIONS):
        res = results[sid]
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i][col_i]
            null_vals = res["null"]["pearson"][:, col_i]
            real_val = res["real"]["pearson"][col_i]
            p_val = res["p_values"]["pearson"][col_i]
            ax.hist(null_vals, bins=40, color="#999999", alpha=0.7)
            ax.axvline(real_val, color="#d62728", lw=1.8, label=f"real={real_val:.2f}")
            sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
            ax.set_title(f"rank {rank}/88 {mouse}: {target}\np={p_val:.4f} {sig_str}", fontsize=7.5)
            ax.set_xlabel("Pearson r (null distribution, n=1000)", fontsize=6.5)
            ax.tick_params(labelsize=6)
            ax.legend(fontsize=6, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig_path = OUT_DIR / "061_perfquant_pls1se_shufflesignificance_histograms.png"
    fig.savefig(fig_path, dpi=140)
    print(f"saved {fig_path.name}")
    print("DONE_061")


if __name__ == "__main__":
    main()
