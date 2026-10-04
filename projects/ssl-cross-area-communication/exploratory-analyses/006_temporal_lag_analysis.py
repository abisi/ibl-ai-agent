"""Last checkpoint piece before scaling: (1) project population activity
(both trial-averaged PSTH and single-trial residuals) onto the fitted
canonical axes to see area-A/area-B correlation across peri-stimulus time,
(2) lagged pCCA -- shift area B's canonical-variate signal relative to area
A in 10ms steps (+/-100ms) and recompute the across-trial correlation at
each (time bin, lag) pair, plotted as a matrix, plus the correlation-over-
time line at whichever lag is strongest overall (compared against lag=0).

Two example sessions (one per cohort, good+MUA tier, Motor-frontal vs.
Somatosensory, whisker trials) -- still checkpoint scale, not the full
dataset. CCA is fit on a train half of trials; everything shown here uses
only the held-out test half (same held-out discipline as checkpoint_04).
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
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
AREA_A, AREA_B = "Motor and frontal areas", "Somatosensory areas"
N_UNITS_SUBSAMPLE = 30
LAG_BINS = np.arange(-10, 11)  # +/-10 bins * 10ms/bin = +/-100ms, 10ms steps

EXAMPLES = {"R+": "AB127_20240821_103757", "R-": "AB126_20240822_114405"}


def project_tensor(tensor: np.ndarray, cca: CCA, side: str, dim: int = 0) -> np.ndarray:
    """(n_trials, n_bins, n_units) -> (n_trials, n_bins) canonical-variate
    scores for `dim`, using the fitted CCA's own centering (via .transform,
    not a raw weight dot-product, so train-set centering is applied
    correctly to held-out data)."""
    n_trials, n_bins, n_units = tensor.shape
    flat = tensor.reshape(n_trials * n_bins, n_units)
    valid = ~np.isnan(flat).any(axis=1)
    out = np.full((n_trials * n_bins,), np.nan)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return out.reshape(n_trials, n_bins)
    # sklearn's CCA.transform(X, Y) requires both sides' features to project
    # either one; feed zeros of the correct shape for the side not being
    # projected here (its output is discarded) -- centering/scaling for the
    # requested side is still exactly the train-fit transform.
    if side == "x":
        dummy_y = np.zeros((n_valid, cca.y_weights_.shape[0]))
        Xc, _ = cca.transform(flat[valid], dummy_y)
        out[valid] = Xc[:, dim]
    else:
        dummy_x = np.zeros((n_valid, cca.x_weights_.shape[0]))
        _, Yc = cca.transform(dummy_x, flat[valid])
        out[valid] = Yc[:, dim]
    return out.reshape(n_trials, n_bins)


def lagged_corr_matrix(Ca: np.ndarray, Cb: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """(n_time_bins, n_lags) matrix: corrcoef across trials of Ca[:, t] vs.
    Cb[:, t+lag], NaN where t+lag is out of range or too few valid trials."""
    n_trials, n_bins = Ca.shape
    mat = np.full((n_bins, len(lags)), np.nan)
    for li, lag in enumerate(lags):
        for t in range(n_bins):
            t2 = t + lag
            if t2 < 0 or t2 >= n_bins:
                continue
            a, b = Ca[:, t], Cb[:, t2]
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 10:
                continue
            if np.std(a[m]) > 0 and np.std(b[m]) > 0:
                mat[t, li] = np.corrcoef(a[m], b[m])[0, 1]
    return mat


def main() -> None:
    files = [f"{s}.nwb" for s in EXAMPLES.values()]
    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=2)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    tier_mask = lambda ut: ut["quality_label"].isin(["good", "mua"])

    bin_edges = cca_lib.time_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rng = np.random.default_rng(7)

    fig_proj, axes_proj = plt.subplots(2, len(EXAMPLES), figsize=(7 * len(EXAMPLES), 9))
    fig_lag, axes_lag = plt.subplots(1, len(EXAMPLES), figsize=(7.5 * len(EXAMPLES), 6))
    fig_line, axes_line = plt.subplots(1, len(EXAMPLES), figsize=(7.5 * len(EXAMPLES), 5), sharey=True)

    for i, (cohort, session_id) in enumerate(EXAMPLES.items()):
        sess = unit_table[(unit_table["session_id"] == session_id) & tier_mask(unit_table)]
        units_a = sess[sess["area_group_coarse"] == AREA_A].sample(
            n=min(N_UNITS_SUBSAMPLE, (sess["area_group_coarse"] == AREA_A).sum()), random_state=0)
        units_b = sess[sess["area_group_coarse"] == AREA_B].sample(
            n=min(N_UNITS_SUBSAMPLE, (sess["area_group_coarse"] == AREA_B).sum()), random_state=0)
        trials = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")
                              & (trial_table["trial_type"] == "whisker_trial")].sort_values("start_time")
        starts = trials["start_time"].to_numpy(dtype=float)
        print(f"{cohort} {session_id}: {len(units_a)}/{len(units_b)} units, {len(starts)} whisker trials")

        spikes_a = [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units_a.iterrows()]
        spikes_b = [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units_b.iterrows()]
        tensor_a = cca_lib.population_tensor(spikes_a, starts, True, bin_edges)
        tensor_b = cca_lib.population_tensor(spikes_b, starts, True, bin_edges)
        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)

        n_trials_tot = len(starts)
        perm = rng.permutation(n_trials_tot)
        train_idx, test_idx = perm[: n_trials_tot // 2], perm[n_trials_tot // 2:]

        flat_a_train, _ = cca_lib.flatten_trial_bins(resid_a[train_idx])
        flat_b_train, _ = cca_lib.flatten_trial_bins(resid_b[train_idx])
        n_dims = min(10, len(units_a), len(units_b))
        cca = CCA(n_components=n_dims, max_iter=2000).fit(flat_a_train, flat_b_train)

        Ca_test = project_tensor(resid_a[test_idx], cca, side="x", dim=0)
        Cb_test = project_tensor(resid_b[test_idx], cca, side="y", dim=0)

        # --- PSTH projection: mean (non-residual) tensor through the same weights ---
        Ca_psth = project_tensor(np.nanmean(tensor_a[test_idx], axis=0, keepdims=True), cca, side="x", dim=0)[0]
        Cb_psth = project_tensor(np.nanmean(tensor_b[test_idx], axis=0, keepdims=True), cca, side="y", dim=0)[0]

        ax = axes_proj[0, i]
        ax.plot(bin_centers * 1000, Ca_psth, label=f"{AREA_A} canon. PSTH", color="tab:blue")
        ax.plot(bin_centers * 1000, Cb_psth, label=f"{AREA_B} canon. PSTH", color="tab:orange")
        r_psth = np.corrcoef(Ca_psth[~np.isnan(Ca_psth)], Cb_psth[~np.isnan(Cb_psth)])[0, 1]
        ax.set_title(f"{cohort} {session_id}\nPSTH projected onto canonical axes (dim 1), r={r_psth:.2f}")
        ax.set_xlabel("time from start_time (ms)"); ax.legend(fontsize=8)

        lag0_corr = np.array([
            (np.corrcoef(Ca_test[:, t][m], Cb_test[:, t][m])[0, 1]
             if (m := ~(np.isnan(Ca_test[:, t]) | np.isnan(Cb_test[:, t]))).sum() >= 10
             and np.std(Ca_test[:, t][m]) > 0 and np.std(Cb_test[:, t][m]) > 0 else np.nan)
            for t in range(Ca_test.shape[1])
        ])
        ax2 = axes_proj[1, i]
        ax2.plot(bin_centers * 1000, lag0_corr, color="black")
        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.set_title("Trial-by-trial canonical-variate correlation\nacross time (held-out, lag=0)")
        ax2.set_xlabel("time from start_time (ms)"); ax2.set_ylabel("across-trial correlation")

        # --- lagged correlation matrix ---
        mat = lagged_corr_matrix(Ca_test, Cb_test, LAG_BINS)
        ax3 = axes_lag[i]
        im = ax3.imshow(mat.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.6, vmax=0.6,
                         extent=[bin_centers[0] * 1000, bin_centers[-1] * 1000, LAG_BINS[0] * 10, LAG_BINS[-1] * 10])
        ax3.set_xlabel("time from start_time (ms)")
        ax3.set_ylabel(f"lag (ms), {AREA_B} shifted rel. to {AREA_A}\n(positive = B lags A)")
        ax3.set_title(f"{cohort} {session_id}: lagged canonical correlation matrix")
        fig_lag.colorbar(im, ax=ax3, label="across-trial correlation")

        # --- strongest lag: highest mean (signed) correlation across time ---
        mean_by_lag = np.nanmean(mat, axis=0)
        best_li = int(np.nanargmax(mean_by_lag))
        best_lag_ms = LAG_BINS[best_li] * 10
        best_lag_corr = mat[:, best_li]
        axes_line[i].plot(bin_centers * 1000, lag0_corr, label="lag = 0ms", color="gray")
        axes_line[i].plot(bin_centers * 1000, best_lag_corr, label=f"best lag = {best_lag_ms:+.0f}ms "
                           f"(mean r={mean_by_lag[best_li]:.2f} vs {np.nanmean(lag0_corr):.2f} at lag 0)", color="tab:red")
        axes_line[i].axhline(0, color="gray", linewidth=0.8)
        axes_line[i].set_title(f"{cohort} {session_id}")
        axes_line[i].set_xlabel("time from start_time (ms)")
        axes_line[i].legend(fontsize=8)
        print(f"  best lag: {best_lag_ms:+.0f}ms (mean r={mean_by_lag[best_li]:.3f}) vs lag=0 mean r={np.nanmean(lag0_corr):.3f}")

    axes_proj[1, 0].set_ylabel("across-trial correlation")
    axes_line[0].set_ylabel("across-trial correlation")
    fig_proj.suptitle(f"Population activity projected onto canonical space -- {AREA_A} vs {AREA_B}, "
                       f"whisker trials, good+MUA, held-out test trials")
    fig_proj.tight_layout()
    fig_proj.savefig(ARTIFACTS_DIR / "checkpoint3_01_psth_projection_and_time_corr.png", dpi=130)

    fig_lag.suptitle(f"Lagged pCCA -- {AREA_A} vs {AREA_B}, whisker trials, good+MUA, held-out test trials "
                      f"(10ms lag steps, +/-100ms)")
    fig_lag.tight_layout()
    fig_lag.savefig(ARTIFACTS_DIR / "checkpoint3_02_lagged_correlation_matrix.png", dpi=130)

    fig_line.suptitle("Correlation over time at lag=0 vs. the strongest overall lag")
    fig_line.tight_layout()
    fig_line.savefig(ARTIFACTS_DIR / "checkpoint3_03_best_lag_line.png", dpi=130)

    print("\nWrote checkpoint3_01/02/03 figures to", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
