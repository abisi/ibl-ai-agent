"""Intermediate-checkpoint pipeline: one session (MH062_20260113_125836,
picked from the coverage smoke test -- 1184 "Motor and frontal areas" units,
1145 "Somatosensory areas" units, coarse level), whisker + auditory active
trials. Builds the noise-correlation tensors, runs variant-A CV-CCA + PCA
baseline, and produces the required checkpoint figures: input-data rasters,
noise-correlation illustration, piezo-lick regressor illustration (variant C
mechanics only, not yet its CCA result), and example correlated trials.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cov_lib = importlib.import_module("000_coverage_lib")
cca_lib = importlib.import_module("003_cca_lib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

SESSION_FILE = "MH062_20260113_125836.nwb"
AREA_A = "Motor and frontal areas"
AREA_B = "Somatosensory areas"
N_UNITS_SUBSAMPLE = 40  # per area, for a tractable checkpoint run
RNG = np.random.default_rng(0)


def main() -> None:
    unit_table, trial_table = cov_lib.load_units([SESSION_FILE], day_to_analyze="all", max_workers=1)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[unit_table["quality_label"] != "non-soma"].copy()
    session_id = unit_table["session_id"].iloc[0]
    print(f"session_id={session_id}, reward_group={unit_table['reward_group'].iloc[0]}, "
          f"{len(unit_table)} units after quality filter")

    trials = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
    print(f"active trials: {len(trials)}, by type:\n{trials['trial_type'].value_counts()}")

    units_a = unit_table[unit_table["area_group_coarse"] == AREA_A]
    units_b = unit_table[unit_table["area_group_coarse"] == AREA_B]
    print(f"{AREA_A}: {len(units_a)} units available; {AREA_B}: {len(units_b)} units available")
    units_a = units_a.sample(n=min(N_UNITS_SUBSAMPLE, len(units_a)), random_state=0)
    units_b = units_b.sample(n=min(N_UNITS_SUBSAMPLE, len(units_b)), random_state=0)
    print(f"subsampled to {len(units_a)} / {len(units_b)} units for this checkpoint run")

    bin_edges = cca_lib.time_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    results = {}
    for trial_type in ["whisker_trial", "auditory_trial"]:
        tt = trials[trials["trial_type"] == trial_type].sort_values("start_time")
        starts = tt["start_time"].to_numpy(dtype=float)
        is_whisker = trial_type == "whisker_trial"
        print(f"\n[{trial_type}] {len(starts)} trials")

        spikes_a = [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units_a.iterrows()]
        spikes_b = [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units_b.iterrows()]
        tensor_a = cca_lib.population_tensor(spikes_a, starts, is_whisker, bin_edges)
        tensor_b = cca_lib.population_tensor(spikes_b, starts, is_whisker, bin_edges)

        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)

        flat_a, valid_a = cca_lib.flatten_trial_bins(resid_a)
        flat_b, valid_b = cca_lib.flatten_trial_bins(resid_b)
        assert np.array_equal(valid_a, valid_b), "A/B dead-zone masks should match (same trials/bins)"
        n_dims = min(10, len(units_a), len(units_b))
        print(f"  {flat_a.shape[0]} valid (trial,bin) samples, n_dims={n_dims}")

        cv = cca_lib.cv_canonical_correlation(flat_a, flat_b, n_components=n_dims, n_folds=5)
        print(f"  variant A held-out canonical corr (dim1): {cv['cca_mean'][0]:.3f} +/- {cv['cca_sd'][0]:.3f}; "
              f"PCA-alignment baseline (dim1): {cv['pca_mean'][0]:.3f} +/- {cv['pca_sd'][0]:.3f}")
        results[trial_type] = dict(cv=cv, tensor_a=tensor_a, tensor_b=tensor_b, resid_a=resid_a, resid_b=resid_b,
                                    starts=starts, spikes_a=spikes_a, spikes_b=spikes_b)

    # --- Figure 1: input-data rasters (one example trial, both areas, raw spikes) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, area_name, spikes_list, starts, label in [
        (axes[0], AREA_A, results["whisker_trial"]["spikes_a"], results["whisker_trial"]["starts"], "A"),
        (axes[1], AREA_B, results["whisker_trial"]["spikes_b"], results["whisker_trial"]["starts"], "B"),
    ]:
        t0 = starts[len(starts) // 2]
        for i, st in enumerate(spikes_list):
            rel = st[(st >= t0 - 0.2) & (st <= t0 + 0.5)] - t0
            ax.vlines(rel * 1000, i, i + 0.8, color="k", linewidth=0.5)
        ax.axvspan(-1, 4, color="red", alpha=0.2, label="dead zone")
        ax.axvline(0, color="blue", linestyle="--", linewidth=1, label="start_time")
        ax.set_title(f"{area_name} (example whisker trial, n={len(spikes_list)} units)")
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel("unit #")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "checkpoint_01_input_rasters.png", dpi=130)
    plt.close(fig)

    # --- Figure 2: noise-correlation illustration (one example unit, raw vs PSTH vs residual) ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    tt = results["whisker_trial"]
    tensor_a, resid_a = tt["tensor_a"], tt["resid_a"]
    # Pick the most active unit (highest total spike count in-window) for a
    # legible illustration -- an arbitrary index risks landing on a near-silent
    # unit whose PSTH/residual is uninformative to look at.
    total_counts = np.nansum(tensor_a, axis=(0, 1))
    unit_idx = int(np.argmax(total_counts))
    for trial_i in range(min(15, tensor_a.shape[0])):
        axes[0].plot(bin_centers * 1000, tensor_a[trial_i, :, unit_idx], color="gray", alpha=0.4, linewidth=0.8)
    psth = np.nanmean(tensor_a[:, :, unit_idx], axis=0)
    axes[0].plot(bin_centers * 1000, psth, color="black", linewidth=2, label="trial-averaged PSTH")
    axes[0].set_ylabel("rate (Hz)")
    axes[0].set_title(f"{AREA_A} example unit: single trials + PSTH (whisker trials)")
    axes[0].legend()
    for trial_i in range(min(15, resid_a.shape[0])):
        axes[1].plot(bin_centers * 1000, resid_a[trial_i, :, unit_idx], color="tab:blue", alpha=0.4, linewidth=0.8)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_ylabel("residual rate (Hz)")
    axes[1].set_xlabel("time from start_time (ms)")
    axes[1].set_title("noise-correlation residual (PSTH subtracted, no baseline correction)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "checkpoint_02_noise_correlation_illustration.png", dpi=130)
    plt.close(fig)

    # --- Figure 3: piezo-lick regressor illustration (variant C mechanics) ---
    REPO_ROOT = Path(__file__).resolve().parents[3]
    SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")
    licks = events[(events["session_id"] == session_id) & (events["event_type"] == "piezo_lick_times")]
    lick_times = np.sort(licks["time"].to_numpy(dtype=float))
    print(f"\npiezo_lick_times events for this session: {len(lick_times)}")
    if len(lick_times):
        starts_w = results["whisker_trial"]["starts"]
        lick_reg = cca_lib.lick_rate_regressor(lick_times, starts_w, bin_edges)
        fig, ax = plt.subplots(figsize=(10, 4))
        example_trials = range(min(20, lick_reg.shape[0]))
        for i in example_trials:
            ax.plot(bin_centers * 1000, lick_reg[i], alpha=0.3, color="tab:green", linewidth=0.8)
        ax.plot(bin_centers * 1000, np.nanmean(lick_reg, axis=0), color="darkgreen", linewidth=2, label="mean lick rate")
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel("lick rate (Hz, 50ms bins resampled to 10ms)")
        ax.set_title("Variant C nuisance regressor: piezo-lick rate (50ms bins)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / "checkpoint_03_lick_regressor_illustration.png", dpi=130)
        plt.close(fig)
    else:
        print("  no piezo_lick_times events for this session -- variant C not illustrable here, will check other sessions at full scale")

    # --- Figure 4: example correlated trials (top canonical dimension, variant A) ---
    # Held-out only: fit CCA on a train half of trials, transform the *test*
    # half, and pick example trials from the test half -- an in-sample fit
    # is guaranteed to show inflated per-trial correlation (CCA actively
    # optimizes for exactly this), which would misrepresent the checkpoint's
    # own CV-canonical-correlation result above.
    from sklearn.cross_decomposition import CCA as _CCA
    tt = results["whisker_trial"]
    n_trials_w, n_bins_w, _ = tt["resid_a"].shape
    rng_split = np.random.default_rng(1)
    perm = rng_split.permutation(n_trials_w)
    train_trials, test_trials = perm[: n_trials_w // 2], perm[n_trials_w // 2:]

    def _flat_for_trials(resid, trial_idx):
        sub = resid[trial_idx]
        flat, valid = cca_lib.flatten_trial_bins(sub)
        return flat, valid

    flat_a_train, _ = _flat_for_trials(tt["resid_a"], train_trials)
    flat_b_train, _ = _flat_for_trials(tt["resid_b"], train_trials)
    n_dims = min(10, len(units_a), len(units_b))
    cca_fit = _CCA(n_components=n_dims, max_iter=2000).fit(flat_a_train, flat_b_train)

    flat_a_test, valid_test = _flat_for_trials(tt["resid_a"], test_trials)
    flat_b_test, _ = _flat_for_trials(tt["resid_b"], test_trials)
    Xc, Yc = cca_fit.transform(flat_a_test, flat_b_test)

    valid_mat_test = valid_test.reshape(len(test_trials), n_bins_w)
    Xc_full = np.full((len(test_trials), n_bins_w), np.nan)
    Yc_full = np.full((len(test_trials), n_bins_w), np.nan)
    Xc_full[valid_mat_test] = Xc[:, 0]
    Yc_full[valid_mat_test] = Yc[:, 0]
    per_trial_corr = np.array([
        np.corrcoef(Xc_full[i][~np.isnan(Xc_full[i])], Yc_full[i][~np.isnan(Yc_full[i])])[0, 1]
        if np.sum(~np.isnan(Xc_full[i])) > 3 else np.nan
        for i in range(len(test_trials))
    ])
    top_local = np.argsort(-np.nan_to_num(per_trial_corr, nan=-2))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, local_i in zip(axes.flat, top_local):
        orig_trial = test_trials[local_i]
        ax.plot(bin_centers * 1000, Xc_full[local_i], label=f"{AREA_A} canon. var 1", color="tab:blue")
        ax.plot(bin_centers * 1000, Yc_full[local_i], label=f"{AREA_B} canon. var 1", color="tab:orange")
        ax.set_title(f"trial {orig_trial} (held out), r={per_trial_corr[local_i]:.2f}")
        ax.set_xlabel("time from start_time (ms)")
        ax.legend(fontsize=7)
    fig.suptitle("Example correlated trials -- HELD-OUT (CCA fit on other half of trials), variant A, dim 1, whisker trials")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "checkpoint_04_example_correlated_trials.png", dpi=130)
    plt.close(fig)
    print(f"\n  held-out example-trial correlations: median={np.nanmedian(per_trial_corr):.2f}, "
          f"max={np.nanmax(per_trial_corr):.2f} (n_test_trials={len(test_trials)})")

    # --- Figure 5: CV canonical correlation vs PCA baseline, per dimension, both trial types ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, trial_type in zip(axes, ["whisker_trial", "auditory_trial"]):
        cv = results[trial_type]["cv"]
        dims = np.arange(1, cv["n_dims"] + 1)
        ax.errorbar(dims, cv["cca_mean"], yerr=cv["cca_sd"], label="variant A CV canonical corr", marker="o")
        ax.errorbar(dims, cv["pca_mean"], yerr=cv["pca_sd"], label="PCA-alignment baseline", marker="s")
        ax.set_xlabel("canonical dimension")
        ax.set_title(trial_type)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("held-out correlation")
    fig.suptitle(f"{AREA_A} vs {AREA_B}, session {session_id} (checkpoint subsample, n={N_UNITS_SUBSAMPLE}/area)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "checkpoint_05_cv_cca_vs_pca.png", dpi=130)
    plt.close(fig)

    print("\nWrote checkpoint_01..05 figures to", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
