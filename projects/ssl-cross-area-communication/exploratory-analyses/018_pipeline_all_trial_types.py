"""Didactic pipeline figure for one example pair, updated methodology
(no unit subsampling for the actual computation -- rasters subsample only
for legible display; -10ms/+5ms dead zone; whisker/auditory/no_stim trial
types, active only). Two figures:
1. Population raster, all three trial types, one figure.
2. PSTH -> residual -> PCA -> CCA, each stage plotted across peri-stimulus
   time (ms), one row per stage x one column per trial type.

Example pair: Motor and frontal areas vs Striatum and pallidum (the
richest coarse pair), session AB126_20240822_114405 (887/805 good+MUA
units -- picked as the best-covered session for this pair, no NWB
re-selection needed beyond this one session).
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
from partial_CCA import PartialCCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
R_PLUS_COLOR, R_MINUS_COLOR = "#00B400", "#C800C8"
AREA_A, AREA_B = "Motor and frontal areas", "Striatum and pallidum"
SESSION_FILE = "AB126_20240822_114405.nwb"
TRIAL_TYPES = [("whisker_trial", True), ("auditory_trial", False), ("no_stim_trial", False)]
N_DISPLAY_UNITS = 50  # raster legibility only -- computation below uses ALL units
REGULARIZATION = 1e-3


def flatten(tensor):
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


def spikes_of(units):
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def to_full_timeline(values_valid: np.ndarray, valid_bins: np.ndarray) -> np.ndarray:
    """Scatter a valid-bins-only 1D array back into a full-length array,
    NaN elsewhere -- so plotting against the FULL time axis leaves a
    genuine visual gap at excluded (dead-zone) bins instead of letting the
    line connect straight across them (matplotlib breaks lines at NaN)."""
    full = np.full(len(valid_bins), np.nan)
    full[valid_bins] = values_valid
    return full


def main() -> None:
    unit_table, trial_table = cov_lib.load_units([SESSION_FILE], day_to_analyze="learning", max_workers=1)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[unit_table["quality_label"].isin(["good", "mua"])]
    session_id = unit_table["session_id"].iloc[0]
    cohort = unit_table["reward_group"].iloc[0]

    units_a = unit_table[unit_table["area_group_coarse"] == AREA_A]
    units_b = unit_table[unit_table["area_group_coarse"] == AREA_B]
    print(f"{session_id} ({cohort}): {len(units_a)} {AREA_A} units, {len(units_b)} {AREA_B} units (ALL used, no cap)")

    trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
    bin_edges = cca_lib.sliding_window_starts()  # sliding-window starts, not tile edges -- see 003_cca_lib.py
    bin_centers = cca_lib.sliding_window_centers(bin_edges)
    rng = np.random.default_rng(0)

    spikes_a_all, spikes_b_all = spikes_of(units_a), spikes_of(units_b)

    data = {}
    for trial_type, is_whisker in TRIAL_TYPES:
        tt = trials_sess[trials_sess["trial_type"] == trial_type].sort_values("start_time")
        starts = tt["start_time"].to_numpy(dtype=float)
        print(f"  {trial_type}: {len(starts)} trials")
        tensor_a = cca_lib.population_tensor_sliding(spikes_a_all, starts, is_whisker, bin_edges)
        tensor_b = cca_lib.population_tensor_sliding(spikes_b_all, starts, is_whisker, bin_edges)
        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)
        dz_mask = cca_lib.dead_zone_bin_mask_sliding(bin_edges, cca_lib.BIN_WIDTH, is_whisker)
        valid_bins = ~dz_mask
        data[trial_type] = dict(starts=starts, tensor_a=tensor_a, tensor_b=tensor_b,
                                 resid_a=resid_a, resid_b=resid_b, valid_bins=valid_bins, is_whisker=is_whisker)

    # ================= Figure 1: population raster, all 3 trial types =================
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharex=True)
    disp_idx_a = rng.choice(len(spikes_a_all), size=min(N_DISPLAY_UNITS, len(spikes_a_all)), replace=False)
    disp_idx_b = rng.choice(len(spikes_b_all), size=min(N_DISPLAY_UNITS, len(spikes_b_all)), replace=False)
    for col, (trial_type, is_whisker) in enumerate(TRIAL_TYPES):
        starts = data[trial_type]["starts"]
        t0 = starts[len(starts) // 2]
        for row, (label, spikes_all, disp_idx) in enumerate([(AREA_A, spikes_a_all, disp_idx_a), (AREA_B, spikes_b_all, disp_idx_b)]):
            ax = axes[row, col]
            for i, ui in enumerate(disp_idx):
                st = spikes_all[ui]
                rel = st[(st >= t0 - 0.2) & (st <= t0 + 0.5)] - t0
                ax.vlines(rel * 1000, i, i + 0.8, color="k", linewidth=0.5)
            if is_whisker:
                ax.axvspan(cca_lib.DEAD_ZONE_START_S * 1000, cca_lib.DEAD_ZONE_STOP_S * 1000, color="red", alpha=0.15)
            ax.axvline(0, color="blue", linestyle="--", linewidth=1)
            ax.set_title(f"{trial_type}" if row == 0 else "", fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{label}\nunit # (n={N_DISPLAY_UNITS} of {len(spikes_all)} shown)")
            if row == 1:
                ax.set_xlabel("time from start_time (ms)")
    fig.suptitle(f"Population raster, example trial per type -- {AREA_A} vs {AREA_B}, session {session_id} ({cohort})\n"
                 f"dead zone (whisker only) = -10ms/+5ms, shaded red")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "example_pair_01_raster.png", dpi=120)
    plt.close(fig)
    print("Wrote example_pair_01_raster.png")

    # ================= Figure 2: PSTH -> residual -> PCA -> CCA, across time, per trial type =================
    fig, axes = plt.subplots(4, 3, figsize=(21, 22))
    unit_idx_a = int(np.argsort(np.nansum(data["whisker_trial"]["tensor_a"], axis=(0, 1)))[len(spikes_a_all) // 2])
    unit_idx_b = int(np.argsort(np.nansum(data["whisker_trial"]["tensor_b"], axis=(0, 1)))[len(spikes_b_all) // 2])

    for col, (trial_type, is_whisker) in enumerate(TRIAL_TYPES):
        d = data[trial_type]
        t_ms = bin_centers * 1000

        # Row 0: PSTH (trial-averaged), example unit per area
        ax = axes[0, col]
        psth_a = np.nanmean(d["tensor_a"][:, :, unit_idx_a], axis=0)
        psth_b = np.nanmean(d["tensor_b"][:, :, unit_idx_b], axis=0)
        ax.plot(t_ms, psth_a, color="tab:blue", label=f"{AREA_A} (example unit)")
        ax.plot(t_ms, psth_b, color="tab:orange", label=f"{AREA_B} (example unit)")
        ax.set_title(trial_type)
        if col == 0:
            ax.set_ylabel("PSTH\nrate (Hz)")
        ax.legend(fontsize=7)

        # Row 1: residual (single trials + zero line), same example units
        ax = axes[1, col]
        resid_a_u = d["resid_a"][:, :, unit_idx_a]
        for ti in range(min(15, resid_a_u.shape[0])):
            ax.plot(t_ms, resid_a_u[ti], color="tab:blue", alpha=0.3, linewidth=0.7)
        ax.axhline(0, color="black", linewidth=1)
        if col == 0:
            ax.set_ylabel("residual\nrate (Hz), area A")

        # Held-out split, shared by the PCA and CCA rows below. A residual is
        # PSTH-subtracted BY CONSTRUCTION (residual = raw - trial_mean), so its
        # trial-average over the SAME trials the mean/PCA/CCA were fit on is
        # exactly zero (found directly: an earlier version fit PCA on all
        # trials and averaged PC1 over those same trials -- values came out
        # at the 1e-13 floating-point-noise floor, not real signal). Fitting
        # on train trials and averaging the projection over held-out test
        # trials avoids this degeneracy for both rows.
        n_bins_v = d["valid_bins"].sum()
        perm = rng.permutation(d["resid_a"].shape[0])
        half = len(perm) // 2
        train_idx, test_idx = perm[:half], perm[half:]
        Xa_train, _ = cca_lib.flatten_trial_bins(d["resid_a"][train_idx][:, d["valid_bins"], :])
        Xb_train, _ = cca_lib.flatten_trial_bins(d["resid_b"][train_idx][:, d["valid_bins"], :])
        Xa_test = d["resid_a"][test_idx][:, d["valid_bins"], :].reshape(-1, len(spikes_a_all))
        Xb_test = d["resid_b"][test_idx][:, d["valid_bins"], :].reshape(-1, len(spikes_b_all))
        n_test = len(test_idx)

        # Row 2: PCA -- PC1 of the residual population, fit on train trials,
        # held-out projection averaged over test trials, across time.
        ax = axes[2, col]
        from sklearn.decomposition import PCA as _PCA
        pca_a = _PCA(n_components=1).fit(Xa_train)
        pca_b = _PCA(n_components=1).fit(Xb_train)
        pc1_a = pca_a.transform(Xa_test).reshape(n_test, n_bins_v)
        pc1_b = pca_b.transform(Xb_test).reshape(n_test, n_bins_v)
        ax.plot(t_ms, to_full_timeline(np.nanmean(pc1_a, axis=0), d["valid_bins"]), color="tab:blue", label=f"{AREA_A} PC1 (held-out)")
        ax.plot(t_ms, to_full_timeline(np.nanmean(pc1_b, axis=0), d["valid_bins"]), color="tab:orange", label=f"{AREA_B} PC1 (held-out)")
        ax.axhline(0, color="black", linewidth=0.8)
        if col == 0:
            ax.set_ylabel("PCA\nPC1 score (held-out trial-avg)")
        ax.legend(fontsize=7)

        # Row 3: CCA -- held-out canonical-variate (dim1) projected across time
        ax = axes[3, col]
        try:
            model = PartialCCA(regularization=REGULARIZATION).fit(Xa_train, Xb_train, None, verbose=False)
            ca, cb = model.transform(Xa_test, Xb_test)
            ca1 = ca[0].reshape(n_test, n_bins_v)
            cb1 = cb[0].reshape(n_test, n_bins_v)
            ax.plot(t_ms, to_full_timeline(np.nanmean(ca1, axis=0), d["valid_bins"]), color="tab:blue", label=f"{AREA_A} canon. var 1 (held-out)")
            ax.plot(t_ms, to_full_timeline(np.nanmean(cb1, axis=0), d["valid_bins"]), color="tab:orange", label=f"{AREA_B} canon. var 1 (held-out)")
            r = np.corrcoef(np.nanmean(ca1, axis=0), np.nanmean(cb1, axis=0))[0, 1]
            ax.set_title(f"{trial_type}  (r={r:.2f})", fontsize=9)
        except Exception as e:
            ax.text(0.5, 0.5, f"CCA failed:\n{e}", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        if col == 0:
            ax.set_ylabel("CCA\ncanon. variate (dim 1)")
        ax.set_xlabel("time from start_time (ms)")
        ax.legend(fontsize=7)

    fig.suptitle(f"Pipeline across peri-stimulus time -- {AREA_A} vs {AREA_B}, session {session_id} ({cohort}), "
                 f"ALL units used (no subsampling: {len(spikes_a_all)}/{len(spikes_b_all)})")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "example_pair_02_pipeline_across_time.png", dpi=120)
    plt.close(fig)
    print("Wrote example_pair_02_pipeline_across_time.png")


if __name__ == "__main__":
    main()
