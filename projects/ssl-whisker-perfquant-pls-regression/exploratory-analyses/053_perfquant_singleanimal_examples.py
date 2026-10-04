"""Example true-vs-predicted trial sequences for `052`'s single-animal,
per-neuron, multi-head regression (user request 2026-09-20: "Show example
fits"). `052` itself only ever saved aggregate per-session metrics, never
per-trial predictions -- this script redoes just the REAL (no null) decode
for a small, deliberately chosen set of sessions (best/worst by Pearson r,
picked from `052`'s already-saved summary CSVs for both window variants)
to capture out-of-fold per-trial predictions for plotting.

Two window variants shown side by side, matching `052`'s two runs:
SENSORY_WINDOW (5-50ms post-stim) and WIDE_BASELINE_WINDOW (-2000/-10ms
pre-stim).

**Extended 2026-09-20** (user: "Also plot the predictions from the null
distributions as mean+CI"): each example session/window/target now ALSO
gets a full retrain-under-shift null (`null_predictions_per_trial`,
`N_SHUF_EXAMPLE=10` shuffles -- reduced from `052`'s 20, since this is
for illustration, not another significance test), with each shuffle's
predictions mapped back onto the ORIGINAL trial-index axis so a mean +-
95% CI band can be drawn on the same plot as the true/real-predicted
lines. This roughly doubles this script's already-substantial per-session
cost (it was ~27 minutes for 10 session x window combos before this
addition)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
WIDE_BASELINE_WINDOW = (-2.000, -0.010)
WINDOWS = {"sensory": SENSORY_WINDOW, "baseline": WIDE_BASELINE_WINDOW}
DEAD_ZONE = (-0.001, 0.004)
N_REPEATS = 3
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]

# Best/worst-by-Pearson-r sessions, per target, per window (read off 052's
# saved summary CSVs before writing this script) -- deliberately a small,
# non-exhaustive set.
EXAMPLE_MICE = ["AB158", "AB154", "AB092", "MH011", "MH022"]


SCALE_FLOOR = 1e-3  # Hz -- see 052's _make_multihead_regressor for the full explanation: a
                     # column can be near-constant within one CV fold's training subset purely
                     # by chance even with fine whole-session variance, and vanilla
                     # StandardScaler's tiny-but-nonzero std for that fold then explodes 1/std
                     # for any held-out value that differs even slightly. Flooring the scale
                     # prevents that regardless of which fold happens to be near-degenerate.


def _make_regressor(alpha: float):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self

    return make_pipeline(_FlooredScaler(), Ridge(alpha=alpha))


def decode_with_oof(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, n_repeats: int = N_REPEATS, n_folds: int = 5):
    """Real-only decode (no null): selects alpha, then returns per-trial
    OOF predictions averaged across `n_repeats` independent partitions
    (reduces fold-assignment noise for the plotted line) plus per-head
    Pearson r for the title."""
    from sklearn.model_selection import KFold
    from scipy.stats import pearsonr
    RIDGE_ALPHA_GRID = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    best_alpha, best_score = RIDGE_ALPHA_GRID[0], -np.inf
    from sklearn.metrics import r2_score
    for a in RIDGE_ALPHA_GRID:
        Y_pred = np.empty_like(Y)
        for tr, te in splits:
            reg = _make_regressor(a)
            reg.fit(X[tr], Y[tr])
            Y_pred[te] = reg.predict(X[te])
        score = np.mean([r2_score(Y[:, k], Y_pred[:, k]) for k in range(Y.shape[1])])
        if score > best_score:
            best_score, best_alpha = score, a

    Y_pred_accum = np.zeros_like(Y)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_rep = np.empty_like(Y)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            reg = _make_regressor(best_alpha)
            reg.fit(X[tr], Y[tr])
            Y_pred_rep[te] = reg.predict(X[te])
        Y_pred_accum += Y_pred_rep
    Y_pred = Y_pred_accum / n_repeats

    pearsons = [pearsonr(Y[:, k], Y_pred[:, k])[0] for k in range(Y.shape[1])]

    # TRAIN performance (user 2026-09-20: "Plot both train and test
    # performance"): in-sample fit -- the chosen-alpha model fit on ALL
    # trials, predicting the SAME trials it was fit on (no held-out split
    # at all). This is the single-session analog of "train" here (there's
    # no separate pool of other subjects to train on, unlike 050/052's
    # cross-animal "train" meaning); TEST is the existing pooled-CV OOF
    # prediction above. The gap between the two is a direct visual of how
    # much the model overfits within one session.
    reg_train = _make_regressor(best_alpha)
    reg_train.fit(X, Y)
    Y_train_pred = reg_train.predict(X)
    train_pearsons = [pearsonr(Y[:, k], Y_train_pred[:, k])[0] for k in range(Y.shape[1])]

    return Y_pred, pearsons, Y_train_pred, train_pearsons, best_alpha


N_SHUF_EXAMPLE = 10  # reduced from 052's 20 -- this is for illustration, not another significance test
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5


def null_predictions_per_trial(X: np.ndarray, Y: np.ndarray, alpha: float, rng: np.random.Generator,
                                n_shuf: int = N_SHUF_EXAMPLE, n_folds: int = 5) -> np.ndarray:
    """User request 2026-09-20: "Also plot the predictions from the null
    distributions as mean+CI." Same full retrain-under-shift null as
    `052` (fixed alpha, reused from the real fit -- no re-search per
    shuffle), but here each shuffle's predictions are mapped BACK onto
    the ORIGINAL trial-index axis (rather than reduced to a single
    accuracy number) so they can be plotted on the same x-axis as the
    true/real-predicted lines. Returns `(n_shuf, n_trials, n_targets)`,
    NaN wherever a given shuffle's shift-truncation excluded that trial
    (shifts remove 10-50% of the session from one end)."""
    from sklearn.model_selection import KFold
    n = len(Y)
    min_shift = max(1, int(MIN_SHIFT_FRAC * n))
    max_shift = max(min_shift, int(MAX_SHIFT_FRAC * n))
    Y_null = np.full((n_shuf, n, Y.shape[1]), np.nan)
    for s in range(n_shuf):
        shift = int(rng.integers(min_shift, max_shift + 1)) if max_shift > min_shift else min_shift
        if rng.random() < 0.5:
            X_shift, Y_shift, orig_idx = X[: n - shift], Y[shift:], np.arange(shift, n)
        else:
            X_shift, Y_shift, orig_idx = X[shift:], Y[: n - shift], np.arange(0, n - shift)
        seed = int(rng.integers(0, 2**31 - 1))
        Y_pred_shift = np.empty_like(Y_shift)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_shift):
            reg = _make_regressor(alpha)
            reg.fit(X_shift[tr], Y_shift[tr])
            Y_pred_shift[te] = reg.predict(X_shift[te])
        Y_null[s, orig_idx, :] = Y_pred_shift
    return Y_null


def process_one(mouse: str, window_name: str, window: tuple[float, float]) -> dict | None:
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
    X = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, [window], dead_zone=DEAD_ZONE)[0]
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    Y_raw = targets_df[TARGETS].to_numpy()[valid]
    trial_index = np.arange(len(targets_df))[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION:
        return None

    Y_mean, Y_std = Y_raw.mean(axis=0), Y_raw.std(axis=0)
    Y_std_safe = np.where(Y_std > 0, Y_std, 1.0)
    Y = (Y_raw - Y_mean) / Y_std_safe

    rng = np.random.default_rng(abs(hash((session_id, window_name))) % (2**31))
    Y_pred, pearsons, Y_train_pred, train_pearsons, alpha = decode_with_oof(X, Y, rng)
    Y_null = null_predictions_per_trial(X, Y, alpha, rng)
    print(f"  {mouse} [{window_name}]: n={len(X)}, test_pearsons={[round(p, 3) for p in pearsons]}, "
          f"train_pearsons={[round(p, 3) for p in train_pearsons]}", flush=True)
    return dict(mouse=mouse, reward_group=reward_group, session_id=session_id, window_name=window_name,
                trial_index=trial_index, Y_true=Y, Y_pred=Y_pred, Y_train_pred=Y_train_pred, Y_null=Y_null,
                pearsons=pearsons, train_pearsons=train_pearsons)


def main():
    results = {}
    for mouse in EXAMPLE_MICE:
        for window_name, window in WINDOWS.items():
            res = process_one(mouse, window_name, window)
            if res is not None:
                results[(mouse, window_name)] = res

    fig, axes = plt.subplots(len(TARGETS), len(EXAMPLE_MICE), figsize=(4 * len(EXAMPLE_MICE), 3.4 * len(TARGETS)), constrained_layout=True)
    colors = {"sensory": "#1f77b4", "baseline": "#d62728"}
    for row_i, target in enumerate(TARGETS):
        for col_i, mouse in enumerate(EXAMPLE_MICE):
            ax = axes[row_i][col_i]
            plotted_true = False
            for window_name in WINDOWS:
                res = results.get((mouse, window_name))
                if res is None:
                    continue
                if not plotted_true:
                    ax.plot(res["trial_index"], res["Y_true"][:, row_i], color="#333333", lw=1.3, label="true (z-scored)")
                    plotted_true = True
                r = res["pearsons"][row_i]
                ax.plot(res["trial_index"], res["Y_pred"][:, row_i], color=colors[window_name], lw=1.2, alpha=0.85,
                         label=f"{window_name} (r={r:.2f})")
                # Null: mean +- 95% CI across shuffles, per trial index (NaN-safe --
                # a shift's truncation means not every shuffle covers every trial).
                null = res["Y_null"][:, :, row_i]  # (n_shuf, n_trials)
                null_n = np.sum(~np.isnan(null), axis=0)
                null_mean = np.nanmean(null, axis=0)
                null_sem = np.nanstd(null, axis=0) / np.sqrt(np.clip(null_n, 1, None))
                has_cov = null_n >= 3  # don't draw the band where too few shuffles cover a trial
                ti = res["trial_index"]
                ax.plot(ti[has_cov], null_mean[has_cov], color=colors[window_name], lw=1.0, linestyle="--", alpha=0.6,
                         label=f"{window_name} null (mean+-95%CI)")
                ax.fill_between(ti[has_cov], (null_mean - 1.96 * null_sem)[has_cov], (null_mean + 1.96 * null_sem)[has_cov],
                                 color=colors[window_name], alpha=0.15, linewidth=0)
            ax.set_title(f"{mouse}: {target}", fontsize=9)
            ax.set_xlabel("trial index", fontsize=8)
            ax.set_ylabel("z-scored value", fontsize=8)
            ax.legend(fontsize=6.5, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig_path = OUT_DIR / "053_perfquant_singleanimal_examples.png"
    fig.savefig(fig_path, dpi=150)
    print(f"saved {fig_path.name}")

    # --- companion figure: TRAIN (in-sample) vs TEST (pooled-CV OOF) ---
    # user 2026-09-20: "Plot both train and test performance." Kept
    # separate from the null-band figure above (which already has 7
    # plotted elements per panel) rather than overlaying everything into
    # one crowded axis. No null needed here -- train/test overfitting gap
    # is the thing being illustrated, not significance.
    fig2, axes2 = plt.subplots(len(TARGETS), len(EXAMPLE_MICE), figsize=(4 * len(EXAMPLE_MICE), 3.4 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        for col_i, mouse in enumerate(EXAMPLE_MICE):
            ax = axes2[row_i][col_i]
            plotted_true = False
            for window_name in WINDOWS:
                res = results.get((mouse, window_name))
                if res is None:
                    continue
                if not plotted_true:
                    ax.plot(res["trial_index"], res["Y_true"][:, row_i], color="#333333", lw=1.3, label="true (z-scored)")
                    plotted_true = True
                r_test = res["pearsons"][row_i]
                r_train = res["train_pearsons"][row_i]
                ax.plot(res["trial_index"], res["Y_train_pred"][:, row_i], color=colors[window_name], lw=1.0,
                         linestyle=":", alpha=0.8, label=f"{window_name} TRAIN (r={r_train:.2f})")
                ax.plot(res["trial_index"], res["Y_pred"][:, row_i], color=colors[window_name], lw=1.4,
                         alpha=0.85, label=f"{window_name} TEST (r={r_test:.2f})")
            ax.set_title(f"{mouse}: {target}", fontsize=9)
            ax.set_xlabel("trial index", fontsize=8)
            ax.set_ylabel("z-scored value", fontsize=8)
            ax.legend(fontsize=6.5, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig2_path = OUT_DIR / "053_perfquant_singleanimal_train_vs_test.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")


if __name__ == "__main__":
    main()
