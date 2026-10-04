"""Same pipeline as 019 (PSTH -> residual -> PCA -> CCA, across peri-stimulus
time, averaged across all 63 qualifying Motor-frontal vs Striatum sessions),
extended per Axel's 2026-09-01 follow-up:

1. Curves are now tagged with `session_id` and `reward_group` (R+/R-) as they
   are collected, instead of only being pooled -- 019 never saved this, so a
   retroactive R+/R- split was not possible from its output; this reruns the
   (unchanged) fitting logic once, this time keeping enough structure to
   answer this and future re-slicing questions without recomputing again.
2. A canonical-correlation-vs-dimension ("scree") value is captured per
   (session, condition): `PartialCCA.transform()` already returns ALL
   min(n_units_a, n_units_b) canonical dimensions in one call (018/019 only
   ever read dimension 0 out of it) -- so this is free, no extra CCA fits,
   just reading more rows of an already-computed array. Held out the same
   way as dim 1 (train/test split), capped at MAX_DIMS=10.
3. Everything is pickled to
   `artifacts/pipeline_full_dataset_results.pkl` (per-session, per-condition,
   per-cohort curves + per-dimension correlations) so any future replot of
   this pair does not require re-running the ~5h fitting pass again.

Two output figures:
- `example_pair_04_rplus_rminus_comparison.png`: same 5-row structure as
  019's grid (PSTH / residual / PC1 / CCA-variate / CCA-correlation-across-
  time), but split R+ (#00B400) vs R- (#C800C8) with SEM-across-sessions
  shaded bands per cohort, area distinguished by linestyle (solid=Area A,
  dashed=Area B) within each cohort color. Sign-alignment (PC1/CCA-variate
  rows) is done SEPARATELY within each cohort so one cohort's arbitrary sign
  does not bias the other's average.
- `example_pair_05_correlation_vs_dimension.png`: held-out |canonical
  correlation| vs. dimension index (1..10), mean +/- SEM across sessions,
  R+ vs R-, one subplot per condition.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cov_lib = importlib.import_module("000_coverage_lib")
cca_lib = importlib.import_module("003_cca_lib")

import numpy as np
import pandas as pd
from partial_CCA import PartialCCA
from sklearn.decomposition import PCA as _PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
AREA_A, AREA_B = "Motor and frontal areas", "Striatum and pallidum"
CONDITIONS = [("whisker_trial", 0), ("whisker_trial", 1), ("auditory_trial", 1),
              ("no_stim_trial", 0), ("no_stim_trial", 1)]
MIN_UNITS = 20
MIN_TRIALS = 15
CANDIDATE_LAMBDAS = (1e-3, 1e-1, 1.0)
CV_FOLDS = 2
CV_TRIAL_COUNT_THRESHOLD = 100
FIXED_LAMBDA = 1e-3
TIER_FN = lambda ut: ut["quality_label"].isin(["good", "mua"])
MAX_DIMS = 10
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def spikes_of(units):
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def to_full_timeline(values_valid, valid_bins):
    full = np.full(len(valid_bins), np.nan)
    full[valid_bins] = values_valid
    return full


def sign_align(curves: list[np.ndarray]) -> list[np.ndarray]:
    """Flip each curve's sign so it positively correlates with a running
    mean reference (PC1/canonical-dim1 sign is arbitrary per session fit)."""
    if not curves:
        return curves
    aligned = [curves[0]]
    ref = curves[0].copy()
    for c in curves[1:]:
        m = ~(np.isnan(c) | np.isnan(ref))
        if m.sum() > 3 and np.corrcoef(c[m], ref[m])[0, 1] < 0:
            c = -c
        aligned.append(c)
        ref = np.nanmean(np.vstack(aligned), axis=0)
    return aligned


def sem_across(arr_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM across a list of same-length curves (NaN-aware)."""
    a = np.vstack(arr_list)
    mean = np.nanmean(a, axis=0)
    n = np.sum(~np.isnan(a), axis=0)
    sem = np.divide(np.nanstd(a, axis=0, ddof=1), np.sqrt(n), out=np.full_like(mean, np.nan), where=n > 1)
    return mean, sem


def main() -> None:
    session_ids = pd.read_csv(ARTIFACTS_DIR / "motor_striatum_session_list.csv")["session_id"].tolist()
    smoke_n = os.environ.get("SSL_SMOKE_N_SESSIONS")
    if smoke_n:
        # Ensure both cohorts appear in the smoke sample even though the
        # session list is not cohort-balanced in order.
        session_ids = session_ids[: int(smoke_n)]
        print(f"[SMOKE TEST] capped to first {len(session_ids)} sessions")
    files = [f"{s}.nwb" for s in session_ids]
    print(f"Loading {len(files)} sessions for {AREA_A} vs {AREA_B}...")

    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=16)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[TIER_FN(unit_table)]

    bin_edges = cca_lib.sliding_window_starts()
    rng = np.random.default_rng(1)

    curves = {cond: {"session_id": [], "reward_group": [], "mouse_id": [],
                      "n_units_a": [], "n_units_b": [], "n_trials": [],
                      "psth_a": [], "psth_b": [],
                      "resid_a": [], "resid_b": [], "pc1_a": [], "pc1_b": [],
                      "cca_a": [], "cca_b": [], "cca_r": [], "corr_t": [], "lambdas": [],
                      "cc_dims": []}
              for cond in CONDITIONS}
    n_sessions_used = 0

    for session_id in session_ids:
        units_a = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_A)]
        units_b = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_B)]
        if len(units_a) < MIN_UNITS or len(units_b) < MIN_UNITS:
            continue
        reward_group = units_a["reward_group"].iloc[0]
        mouse_id = units_a["mouse_id"].iloc[0]
        n_units_a, n_units_b = len(units_a), len(units_b)
        trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
        spikes_a, spikes_b = spikes_of(units_a), spikes_of(units_b)
        used_this_session = False

        for trial_type, lick_flag in CONDITIONS:
            is_whisker = trial_type == "whisker_trial"
            tt_df = trials_sess[(trials_sess["trial_type"] == trial_type)
                                 & (trials_sess["lick_flag"] == lick_flag)].sort_values("start_time")
            starts = tt_df["start_time"].to_numpy(dtype=float)
            if len(starts) < MIN_TRIALS:
                continue
            dz_mask = cca_lib.dead_zone_bin_mask_sliding(bin_edges, cca_lib.BIN_WIDTH, is_whisker)
            valid_bins = ~dz_mask

            tensor_a = cca_lib.population_tensor_sliding(spikes_a, starts, is_whisker, bin_edges)
            tensor_b = cca_lib.population_tensor_sliding(spikes_b, starts, is_whisker, bin_edges)
            resid_a = cca_lib.noise_correlation_residuals(tensor_a)
            resid_b = cca_lib.noise_correlation_residuals(tensor_b)

            cond = (trial_type, lick_flag)
            curves[cond]["session_id"].append(session_id)
            curves[cond]["reward_group"].append(reward_group)
            curves[cond]["mouse_id"].append(mouse_id)
            curves[cond]["n_units_a"].append(n_units_a)
            curves[cond]["n_units_b"].append(n_units_b)
            curves[cond]["n_trials"].append(len(starts))
            curves[cond]["psth_a"].append(np.nanmean(tensor_a, axis=(0, 2)))
            curves[cond]["psth_b"].append(np.nanmean(tensor_b, axis=(0, 2)))
            curves[cond]["resid_a"].append(np.nanmean(np.abs(resid_a), axis=(0, 2)))
            curves[cond]["resid_b"].append(np.nanmean(np.abs(resid_b), axis=(0, 2)))

            perm = rng.permutation(resid_a.shape[0])
            half = len(perm) // 2
            train_idx, test_idx = perm[:half], perm[half:]
            Xa_train, _ = cca_lib.flatten_trial_bins(resid_a[train_idx][:, valid_bins, :])
            Xb_train, _ = cca_lib.flatten_trial_bins(resid_b[train_idx][:, valid_bins, :])
            Xa_test = resid_a[test_idx][:, valid_bins, :].reshape(-1, len(spikes_a))
            Xb_test = resid_b[test_idx][:, valid_bins, :].reshape(-1, len(spikes_b))
            n_test, n_bins_v = len(test_idx), valid_bins.sum()

            pca_a = _PCA(n_components=1).fit(Xa_train)
            pca_b = _PCA(n_components=1).fit(Xb_train)
            pc1_a = pca_a.transform(Xa_test).reshape(n_test, n_bins_v)
            pc1_b = pca_b.transform(Xb_test).reshape(n_test, n_bins_v)
            curves[cond]["pc1_a"].append(to_full_timeline(np.nanmean(pc1_a, axis=0), valid_bins))
            curves[cond]["pc1_b"].append(to_full_timeline(np.nanmean(pc1_b, axis=0), valid_bins))

            dim_corrs = np.full(MAX_DIMS, np.nan)
            try:
                if len(starts) < CV_TRIAL_COUNT_THRESHOLD:
                    best_lambda, _, _ = cca_lib.select_regularization_trial_cv(
                        resid_a[train_idx], resid_b[train_idx], valid_bins,
                        candidate_lambdas=CANDIDATE_LAMBDAS, n_folds=CV_FOLDS, rng=rng)
                else:
                    best_lambda = FIXED_LAMBDA
                curves[cond]["lambdas"].append(best_lambda)
                model = PartialCCA(regularization=best_lambda).fit(Xa_train, Xb_train, None, verbose=False)
                ca, cb = model.transform(Xa_test, Xb_test)
                ca1 = ca[0].reshape(n_test, n_bins_v)
                cb1 = cb[0].reshape(n_test, n_bins_v)
                ca1_full = to_full_timeline(np.nanmean(ca1, axis=0), valid_bins)
                cb1_full = to_full_timeline(np.nanmean(cb1, axis=0), valid_bins)
                curves[cond]["cca_a"].append(ca1_full)
                curves[cond]["cca_b"].append(cb1_full)
                m = ~(np.isnan(ca1_full) | np.isnan(cb1_full))
                curves[cond]["cca_r"].append(np.corrcoef(ca1_full[m], cb1_full[m])[0, 1] if m.sum() > 3 else np.nan)

                ca1_full_2d = np.full((n_test, len(valid_bins)), np.nan)
                cb1_full_2d = np.full((n_test, len(valid_bins)), np.nan)
                ca1_full_2d[:, valid_bins] = ca1
                cb1_full_2d[:, valid_bins] = cb1
                curves[cond]["corr_t"].append(cca_lib.canonical_correlation_across_time(ca1_full_2d, cb1_full_2d))

                # Per-dimension held-out correlation -- free from the same
                # transform() call above, same recipe as dim 0's cca_r.
                n_dims_cap = min(ca.shape[0], MAX_DIMS)
                for d in range(n_dims_cap):
                    cad_mean = np.nanmean(ca[d].reshape(n_test, n_bins_v), axis=0)
                    cbd_mean = np.nanmean(cb[d].reshape(n_test, n_bins_v), axis=0)
                    if len(cad_mean) > 3:
                        dim_corrs[d] = np.corrcoef(cad_mean, cbd_mean)[0, 1]
            except Exception as e:
                print(f"  {session_id} {cond}: CCA failed ({e})")
            curves[cond]["cc_dims"].append(dim_corrs)
            used_this_session = True
        n_sessions_used += int(used_this_session)
        print(f"{session_id} ({reward_group}): done")

    print(f"\n{n_sessions_used}/{len(session_ids)} sessions contributed >=1 condition")
    for cond in CONDITIONS:
        lambdas = curves[cond]["lambdas"]
        if lambdas:
            print(f"  {cond}: n={len(curves[cond]['psth_a'])} sessions, CV-selected lambda "
                  f"median={np.median(lambdas):.4g} (range {min(lambdas):.4g}-{max(lambdas):.4g})")

    bin_centers = cca_lib.sliding_window_centers(bin_edges)
    config = {
        "area_a": AREA_A, "area_b": AREA_B, "conditions": CONDITIONS,
        "min_units": MIN_UNITS, "min_trials": MIN_TRIALS,
        "candidate_lambdas": CANDIDATE_LAMBDAS, "cv_folds": CV_FOLDS,
        "cv_trial_count_threshold": CV_TRIAL_COUNT_THRESHOLD, "fixed_lambda": FIXED_LAMBDA,
        "quality_tiers": ["good", "mua"], "max_dims": MAX_DIMS,
        "bin_width_s": cca_lib.BIN_WIDTH, "bin_stride_s": cca_lib.BIN_STRIDE,
        "dead_zone_start_s": cca_lib.DEAD_ZONE_START_S, "dead_zone_stop_s": cca_lib.DEAD_ZONE_STOP_S,
        "session_list_source": "motor_striatum_session_list.csv",
        "n_sessions_loaded": len(session_ids), "n_sessions_used": n_sessions_used,
        "day_to_analyze": "learning", "cohort_colors": COHORT_COLOR,
        "script": "020_pipeline_rplus_rminus_and_dims.py",
    }
    with open(ARTIFACTS_DIR / "pipeline_full_dataset_results.pkl", "wb") as f:
        pickle.dump({"curves": curves, "bin_centers": bin_centers, "config": config}, f)
    print("Wrote pipeline_full_dataset_results.pkl (curves + full config/provenance)")

    t_ms = bin_centers * 1000

    # ---- Figure A: R+/R- comparison, 5 rows x 5 conditions ----
    fig, axes = plt.subplots(5, 5, figsize=(32, 26))
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        c = curves[cond]
        rg = np.array(c["reward_group"])
        col_label = f"{trial_type} (lick={lick_flag})"

        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            n_c = len(idx)
            if n_c == 0:
                continue
            color = COHORT_COLOR[cohort]

            ax = axes[0, col]
            m_a, s_a = sem_across([c["psth_a"][i] for i in idx])
            m_b, s_b = sem_across([c["psth_b"][i] for i in idx])
            ax.plot(t_ms, m_a, color=color, linestyle="-", label=f"{cohort} {AREA_A} (n={n_c})")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--", label=f"{cohort} {AREA_B}")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)

            ax = axes[1, col]
            m_a, s_a = sem_across([c["resid_a"][i] for i in idx])
            m_b, s_b = sem_across([c["resid_b"][i] for i in idx])
            ax.plot(t_ms, m_a, color=color, linestyle="-")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)

            ax = axes[2, col]
            pc1_a_aligned = sign_align([c["pc1_a"][i] for i in idx])
            pc1_b_aligned = sign_align([c["pc1_b"][i] for i in idx])
            m_a, s_a = sem_across(pc1_a_aligned)
            m_b, s_b = sem_across(pc1_b_aligned)
            ax.plot(t_ms, m_a, color=color, linestyle="-")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)
            ax.axhline(0, color="black", linewidth=0.8)

            ax = axes[3, col]
            cca_a_aligned = sign_align([c["cca_a"][i] for i in idx])
            cca_b_aligned = sign_align([c["cca_b"][i] for i in idx])
            m_a, s_a = sem_across(cca_a_aligned)
            m_b, s_b = sem_across(cca_b_aligned)
            ax.plot(t_ms, m_a, color=color, linestyle="-")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)
            ax.axhline(0, color="black", linewidth=0.8)

            ax = axes[4, col]
            m_t, s_t = sem_across([c["corr_t"][i] for i in idx])
            ax.plot(t_ms, m_t, color=color, label=f"{cohort} (n={n_c})")
            ax.fill_between(t_ms, m_t - s_t, m_t + s_t, color=color, alpha=0.2)
            ax.axhline(0, color="black", linewidth=0.8)

        axes[0, col].set_title(f"{col_label}\n(solid=Area A, dashed=Area B)")
        axes[4, col].set_xlabel("time from start_time (ms)")
        if col == 0:
            axes[0, col].set_ylabel("PSTH\nmean rate (Hz)")
            axes[1, col].set_ylabel("residual\nmean |residual| (Hz)")
            axes[2, col].set_ylabel("PCA\nPC1 (held-out, sign-aligned/cohort)")
            axes[3, col].set_ylabel("CCA variate\n(dim 1, sign-aligned/cohort)")
            axes[4, col].set_ylabel("CCA correlation\nacross-trial r per time bin")
        axes[0, col].legend(fontsize=6)
        axes[4, col].legend(fontsize=7)

    fig.suptitle(f"R+ vs R- comparison, {AREA_A} vs {AREA_B} -- mean +/- SEM across sessions per cohort "
                 f"(colors: R+={COHORT_COLOR['R+']}, R-={COHORT_COLOR['R-']})", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ARTIFACTS_DIR / "example_pair_04_rplus_rminus_comparison.png", dpi=110)
    plt.close(fig)
    print("Wrote example_pair_04_rplus_rminus_comparison.png")

    # ---- Figure B: correlation vs. canonical dimension, R+ vs R- ----
    fig, axes = plt.subplots(1, 5, figsize=(30, 5.5), sharey=True)
    dims = np.arange(1, MAX_DIMS + 1)
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        c = curves[cond]
        rg = np.array(c["reward_group"])
        ax = axes[col]
        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            if len(idx) == 0:
                continue
            arr = np.abs(np.vstack([c["cc_dims"][i] for i in idx]))
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            sem = np.divide(np.nanstd(arr, axis=0, ddof=1), np.sqrt(n),
                             out=np.full_like(mean, np.nan), where=n > 1)
            color = COHORT_COLOR[cohort]
            ax.errorbar(dims, mean, yerr=sem, color=color, marker="o", markersize=4,
                        label=f"{cohort} (n={len(idx)} sessions)")
        ax.set_title(f"{trial_type} (lick={lick_flag})")
        ax.set_xlabel("canonical dimension")
        ax.set_xticks(dims)
        if col == 0:
            ax.set_ylabel("held-out |canonical correlation|\n(mean +/- SEM across sessions)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.6)

    fig.suptitle(f"Canonical correlation vs. dimension, {AREA_A} vs {AREA_B}, R+ vs R- "
                 f"(held-out, dims capped at {MAX_DIMS})")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "example_pair_05_correlation_vs_dimension.png", dpi=110)
    plt.close(fig)
    print("Wrote example_pair_05_correlation_vs_dimension.png")


if __name__ == "__main__":
    main()
