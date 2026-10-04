"""Plain PLS vs PLS+1SE, run on the FULL 88-session pool (not just the 5
EXAMPLE_MICE), with a systematic +/-20 trial-lag sweep and an R+/R- cohort
comparison (user request 2026-09-20: "Do PLS normal, Pls+1SE. Rexplain the
1SE. Run on all data, add 20 plus or minus lags, compare cohorts.").

**The 1-SE rule, re-explained**: choosing "however many PLS components
maximize the mean CV score" is itself a noisy decision -- the CV score at
each candidate component count is an estimate with real fold-to-fold
sampling variability, and taking the argmax of several noisy estimates is
systematically biased toward whichever candidate got lucky, often the most
complex one on the grid (`055` saw this: 10 components on one session,
2 on another, with no real difference in achievable score). The 1-SE rule
instead: (1) compute the mean AND standard error of the CV score across
folds for every candidate; (2) find the best (highest-mean) candidate;
(3) pick the SIMPLEST candidate (fewest components) whose mean score is
still within one standard error of that best score. In words: "don't pay
for a more complex model unless it wins by more than noise." This directly
targets the train/test overfitting gap this whole thread has been chasing,
at the cost of a small amount of achievable CV score.

**The lag sweep, as the null**: rather than the random shift-null used
elsewhere in this project, here X is paired with Y at a *systematic* trial
offset from -20 to +20 (lag 0 = the real, correctly-aligned pairing). A
real same-trial neural-to-behavior coupling should show a SHARP peak at
lag 0 that decays within a few trials; a broad, flat profile across many
lags instead indicates the "prediction" is really just riding a shared
slow drift (e.g. satiety, arousal, electrode drift) that correlates with
both signals independently of trial-to-trial coupling. This is the same
diagnostic idea as the project's lag cross-correlation checks, applied
here directly to decoding performance instead of to raw signal
correlation.

Each session's PLS component count (both plain-argmax and 1SE-rule) is
selected ONCE at lag 0 and reused unchanged across all 41 lags -- letting
component count re-adapt to each lag would let the model chase whatever
noise exists at that specific (decorrelated) offset, defeating the point
of holding model complexity fixed while only the trial alignment changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu, ttest_ind

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
SENSORY_WINDOW = (0.005, 0.050)
DEAD_ZONE = (-0.001, 0.004)
MIN_UNITS_PER_AREA = 5
MIN_TRIALS_FOR_REGRESSION = 25
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
LAGS = list(range(-20, 21))
N_FOLDS_LAG = 5
N_REPEATS_LAG = 2
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
METHOD_COLORS = {"PLS": "#1f77b4", "PLS+1SE": "#d62728"}
COHORT_COLORS = {"R+": "#00B400", "R-": "#C800C8"}


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
    """One CV sweep over component counts -> (plain argmax pick, 1-SE-rule pick)."""
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


def lag_profile(X: np.ndarray, Y: np.ndarray, n_components: int, rng: np.random.Generator,
                 lags: list[int] = LAGS, n_folds: int = N_FOLDS_LAG, n_repeats: int = N_REPEATS_LAG) -> np.ndarray:
    """Pooled-CV test Pearson per target, at each systematic trial lag between X and Y.

    Pairs X[t] with Y[t + lag] over the valid overlap -- lag=0 is the real,
    correctly-aligned pairing.
    """
    from sklearn.model_selection import KFold

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
            seed = int(rng.integers(0, 2**31 - 1))
            Y_pred_rep = np.empty_like(Y_lag)
            for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X_lag):
                Y_pred_rep[te] = _pls_fit_predict(X_lag[tr], Y_lag[tr], X_lag[te], n_components)
            Y_pred_accum += Y_pred_rep
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
    n_plain, n_1se = select_pls_components(X, Y, rng)
    profile_plain = lag_profile(X, Y, n_plain, rng)
    profile_1se = lag_profile(X, Y, n_1se, rng)

    out.update(ok=True, n_trials=len(X), n_units=X.shape[1],
               n_components_plain=n_plain, n_components_1se=n_1se,
               profile_plain=profile_plain, profile_1se=profile_1se)
    lag0 = LAGS.index(0)
    print(f"{mouse}/{session_id} [{reward_group}]: n={len(X)}, p={X.shape[1]}, "
          f"n_comp(plain)={n_plain}, n_comp(1SE)={n_1se}, "
          f"lag0_pearson(plain)={[round(v, 3) for v in profile_plain[lag0]]}, "
          f"lag0_pearson(1SE)={[round(v, 3) for v in profile_1se[lag0]]}", flush=True)
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
    with ProcessPoolExecutor(max_workers=32) as ex:
        futures = {ex.submit(process_one_session, t): t for t in tasks}
        for fut in as_completed(futures):
            res = fut.result()
            if res.get("ok"):
                results.append(res)
    print(f"{len(results)}/{len(tasks)} sessions usable", flush=True)

    np.savez(OUT_DIR / "057_perfquant_pls_lagprofile_fullpool.npz",
              sessions=np.array([r["session_id"] for r in results]),
              mice=np.array([r["mouse"] for r in results]),
              reward_groups=np.array([r["reward_group"] for r in results]),
              lags=np.array(LAGS),
              profile_plain=np.stack([r["profile_plain"] for r in results]),
              profile_1se=np.stack([r["profile_1se"] for r in results]),
              n_components_plain=np.array([r["n_components_plain"] for r in results]),
              n_components_1se=np.array([r["n_components_1se"] for r in results]))

    records = []
    for r in results:
        for li, lag in enumerate(LAGS):
            for k, target in enumerate(TARGETS):
                records.append(dict(mouse=r["mouse"], session_id=r["session_id"], reward_group=r["reward_group"],
                                     lag=lag, target=target,
                                     pearson_plain=r["profile_plain"][li, k],
                                     pearson_1se=r["profile_1se"][li, k]))
    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "057_perfquant_pls_lagprofile_fullpool.csv", index=False)

    comp_df = pd.DataFrame([dict(mouse=r["mouse"], session_id=r["session_id"], reward_group=r["reward_group"],
                                  n_components_plain=r["n_components_plain"], n_components_1se=r["n_components_1se"])
                             for r in results])
    comp_df.to_csv(OUT_DIR / "057_perfquant_component_counts.csv", index=False)

    # --- Figure 1: mean +/- 95% CI lag profile, per target x method, split by cohort ---
    fig1, axes1 = plt.subplots(len(TARGETS), 2, figsize=(11, 3.4 * len(TARGETS)), constrained_layout=True)
    for row_i, target in enumerate(TARGETS):
        for col_i, (method_key, method_label) in enumerate([("pearson_plain", "PLS"), ("pearson_1se", "PLS+1SE")]):
            ax = axes1[row_i][col_i]
            for cohort, color in COHORT_COLORS.items():
                sub = df[(df.target == target) & (df.reward_group == cohort)]
                grp = sub.groupby("lag")[method_key]
                mean = grp.mean()
                sem = grp.sem()
                ax.plot(mean.index, mean.values, color=color, lw=1.6, label=cohort)
                ax.fill_between(mean.index, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.2)
            ax.axvline(0, color="#888888", lw=0.8, linestyle=":")
            ax.axhline(0, color="#cccccc", lw=0.6)
            ax.set_title(f"{target} -- {method_label}", fontsize=9)
            ax.set_xlabel("lag (trials, X relative to Y)", fontsize=8)
            ax.set_ylabel("mean test Pearson r", fontsize=8)
            ax.legend(fontsize=7, frameon=False)
            ax.spines[["top", "right"]].set_visible(False)
    fig1_path = OUT_DIR / "057_perfquant_lagprofile_cohorts.png"
    fig1.savefig(fig1_path, dpi=150)
    print(f"saved {fig1_path.name}")

    # --- Figure 2: lag=0 Pearson per cohort, both methods, with MW + Welch stats ---
    lag0 = df[df.lag == 0]
    fig2, axes2 = plt.subplots(1, len(TARGETS), figsize=(4.6 * len(TARGETS), 4.4), constrained_layout=True)
    for col_i, target in enumerate(TARGETS):
        ax = axes2[col_i]
        sub = lag0[lag0.target == target]
        positions, labels, data, colors = [], [], [], []
        pos = 0
        group_starts = {}
        for method_key, method_label in [("pearson_plain", "PLS"), ("pearson_1se", "PLS+1SE")]:
            group_starts[method_key] = pos
            for cohort in ("R+", "R-"):
                vals = sub[sub.reward_group == cohort][method_key].dropna().values
                data.append(vals)
                colors.append(COHORT_COLORS[cohort])
                labels.append(f"{method_label}\n{cohort}")
                positions.append(pos)
                pos += 1
            pos += 0.6
        parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=False)
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(target, fontsize=9)
        ax.set_ylabel("lag=0 test Pearson r", fontsize=8)
        ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
        ax.spines[["top", "right"]].set_visible(False)
        for method_key, method_label in [("pearson_plain", "PLS"), ("pearson_1se", "PLS+1SE")]:
            rplus = sub[sub.reward_group == "R+"][method_key].dropna().values
            rminus = sub[sub.reward_group == "R-"][method_key].dropna().values
            if len(rplus) > 2 and len(rminus) > 2:
                u_p = mannwhitneyu(rplus, rminus, alternative="two-sided").pvalue
                t_p = ttest_ind(rplus, rminus, equal_var=False).pvalue
                g0 = group_starts[method_key]
                y = max(np.nanmax(rplus), np.nanmax(rminus)) + 0.05
                ax.plot([g0, g0 + 1], [y, y], color="#333333", lw=1)
                ax.text(g0 + 0.5, y + 0.01, f"MW p={u_p:.3f}\nWelch p={t_p:.3f}", ha="center", fontsize=6)
    fig2_path = OUT_DIR / "057_perfquant_lagprofile_cohort_stats.png"
    fig2.savefig(fig2_path, dpi=150)
    print(f"saved {fig2_path.name}")

    # --- Figure 3: component-count comparison (plain vs 1SE) -- re-illustrates WHY 1SE matters ---
    fig3, ax3 = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    bins = np.arange(0, 32, 2)
    ax3[0].hist(comp_df["n_components_plain"], bins=bins, color=METHOD_COLORS["PLS"], alpha=0.7, label="PLS (plain)")
    ax3[0].hist(comp_df["n_components_1se"], bins=bins, color=METHOD_COLORS["PLS+1SE"], alpha=0.7, label="PLS+1SE")
    ax3[0].set_xlabel("selected n_components")
    ax3[0].set_ylabel("# sessions")
    ax3[0].legend(fontsize=8, frameon=False)
    ax3[0].spines[["top", "right"]].set_visible(False)
    ax3[1].scatter(comp_df["n_components_plain"], comp_df["n_components_1se"], s=14, alpha=0.6, color="#444444")
    lims = [0, 31]
    ax3[1].plot(lims, lims, color="#999999", lw=0.8, linestyle="--")
    ax3[1].set_xlabel("n_components (plain PLS)")
    ax3[1].set_ylabel("n_components (1SE rule)")
    ax3[1].spines[["top", "right"]].set_visible(False)
    fig3_path = OUT_DIR / "057_perfquant_component_selection_comparison.png"
    fig3.savefig(fig3_path, dpi=150)
    print(f"saved {fig3_path.name}")

    print("DONE_057")


if __name__ == "__main__":
    main()
