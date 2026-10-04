"""Distribution of peak lag across sessions, for all (pair, condition)
interactions, per Axel's 2026-08-31 follow-up. Variant A only (plain CCA,
no partialling) and good+MUA tier, to keep this bounded -- the lagged
method itself doesn't need a shuffle null here (peak-lag is a descriptive
summary, not a significance claim), so this is far cheaper than the
shuffle-based runs: one CCA fit per (session, pair, condition), then a
2ms-resolution lag matrix from that fit's dim-1 weights, same dead-zone
handling as 009/newrules_02.

Peak lag = the lag (of 101 candidates, +/-100ms in 2ms steps) with the
highest mean canonical correlation across all valid (non-dead-zone) time
bins -- same "best lag" definition used in checkpoint3/newrules.
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
AREA_PAIRS = [("Motor and frontal areas", "Somatosensory areas"),
              ("Motor and frontal areas", "Striatum and pallidum"),
              ("Somatosensory areas", "Striatum and pallidum")]
CONDITIONS = [("whisker_trial", 0), ("whisker_trial", 1), ("auditory_trial", 1)]
N_UNITS_CAP = 30
MIN_UNITS_GOOD_MUA = 30
MIN_TRIALS = 15
N_DIMS_FIT = 20
REGULARIZATION = 1e-3
BIN_WIDTH = 0.005
LAG_BIN_WIDTH = 0.002
LAG_RANGE_S = 0.100
WINDOW = cca_lib.WINDOW

SESSIONS = {
    "R+": ["AB127_20240821_103757", "AB130_20240902_123634", "AB125_20240817_123403"],
    "R-": ["MH034_20250514_104756", "AB126_20240822_114405", "AB085_20231005_152636"],
}
ALL_FILES = [f"{s}.nwb" for sessions in SESSIONS.values() for s in sessions]
TIER_FN = lambda ut: ut["quality_label"].isin(["good", "mua"])

REPO_ROOT = Path(__file__).resolve().parents[3]
SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"


def flatten(tensor: np.ndarray) -> np.ndarray:
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


def spikes_of(units: pd.DataFrame) -> list[np.ndarray]:
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def main() -> None:
    unit_table, trial_table = cov_lib.load_units(ALL_FILES, day_to_analyze="learning", max_workers=6)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)

    bin_edges = cca_lib.time_bin_edges(WINDOW, BIN_WIDTH)
    lag_bin_edges = cca_lib.time_bin_edges(WINDOW, LAG_BIN_WIDTH)
    lag_bin_centers = 0.5 * (lag_bin_edges[:-1] + lag_bin_edges[1:])
    lag_steps = np.arange(-int(LAG_RANGE_S / LAG_BIN_WIDTH), int(LAG_RANGE_S / LAG_BIN_WIDTH) + 1)
    rng_master = np.random.default_rng(7071)

    rows = []
    for cohort, session_ids in SESSIONS.items():
        for session_id in session_ids:
            trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
            mouse_id = session_id.split("_")[0]

            for area_a, area_b in AREA_PAIRS:
                sess_a = unit_table[(unit_table["session_id"] == session_id)
                                     & (unit_table["area_group_coarse"] == area_a) & TIER_FN(unit_table)]
                sess_b = unit_table[(unit_table["session_id"] == session_id)
                                     & (unit_table["area_group_coarse"] == area_b) & TIER_FN(unit_table)]
                if len(sess_a) < MIN_UNITS_GOOD_MUA or len(sess_b) < MIN_UNITS_GOOD_MUA:
                    continue
                units_a = sess_a.sample(n=min(N_UNITS_CAP, len(sess_a)), random_state=int(rng_master.integers(1e9)))
                units_b = sess_b.sample(n=min(N_UNITS_CAP, len(sess_b)), random_state=int(rng_master.integers(1e9)))
                spikes_a, spikes_b = spikes_of(units_a), spikes_of(units_b)

                for trial_type, lick_flag in CONDITIONS:
                    is_whisker = trial_type == "whisker_trial"
                    tt = trials_sess[(trials_sess["trial_type"] == trial_type)
                                      & (trials_sess["lick_flag"] == lick_flag)].sort_values("start_time")
                    starts = tt["start_time"].to_numpy(dtype=float)
                    if len(starts) < MIN_TRIALS:
                        continue

                    # Fit CCA once (5ms bins, dead zone excluded, no shuffle needed for a descriptive peak-lag summary).
                    dz_mask = cca_lib.dead_zone_bin_mask(bin_edges, is_whisker)
                    valid_bins = ~dz_mask
                    tensor_a = cca_lib.population_tensor(spikes_a, starts, is_whisker, bin_edges)
                    tensor_b = cca_lib.population_tensor(spikes_b, starts, is_whisker, bin_edges)
                    resid_a_v = cca_lib.noise_correlation_residuals(tensor_a)[:, valid_bins, :]
                    resid_b_v = cca_lib.noise_correlation_residuals(tensor_b)[:, valid_bins, :]
                    n_dims = min(N_DIMS_FIT, len(units_a), len(units_b))
                    model = PartialCCA(regularization=REGULARIZATION).fit(flatten(resid_a_v), flatten(resid_b_v), None, verbose=False)

                    # Project onto a finer (2ms) grid using the SAME fitted weights (neuron-indexed, bin-width-agnostic).
                    dz_mask_fine = cca_lib.dead_zone_bin_mask(lag_bin_edges, is_whisker)
                    tensor_a_fine = cca_lib.population_tensor(spikes_a, starts, is_whisker, lag_bin_edges)
                    tensor_b_fine = cca_lib.population_tensor(spikes_b, starts, is_whisker, lag_bin_edges)
                    resid_a_fine = cca_lib.noise_correlation_residuals(tensor_a_fine)
                    resid_b_fine = cca_lib.noise_correlation_residuals(tensor_b_fine)
                    n_trials_f, n_bins_f = resid_a_fine.shape[0], resid_a_fine.shape[1]
                    flat_a, flat_b = resid_a_fine.reshape(-1, resid_a_fine.shape[-1]), resid_b_fine.reshape(-1, resid_b_fine.shape[-1])
                    m = ~np.isnan(flat_a).any(axis=1)
                    proj_a_flat = np.full(n_trials_f * n_bins_f, np.nan)
                    proj_b_flat = np.full(n_trials_f * n_bins_f, np.nan)
                    pa, pb = model.transform(flat_a[m], flat_b[m])
                    proj_a_flat[m], proj_b_flat[m] = pa[0], pb[0]
                    Ca = proj_a_flat.reshape(n_trials_f, n_bins_f)
                    Cb = proj_b_flat.reshape(n_trials_f, n_bins_f)

                    mean_corr_by_lag = np.full(len(lag_steps), np.nan)
                    for li, lag in enumerate(lag_steps):
                        cell_corrs = []
                        for t in range(n_bins_f):
                            t2 = t + lag
                            if t2 < 0 or t2 >= n_bins_f or dz_mask_fine[t] or dz_mask_fine[t2]:
                                continue
                            a, b = Ca[:, t], Cb[:, t2]
                            mm = ~(np.isnan(a) | np.isnan(b))
                            if mm.sum() < 10 or np.std(a[mm]) == 0 or np.std(b[mm]) == 0:
                                continue
                            cell_corrs.append(np.corrcoef(a[mm], b[mm])[0, 1])
                        if cell_corrs:
                            mean_corr_by_lag[li] = np.mean(cell_corrs)

                    if np.all(np.isnan(mean_corr_by_lag)):
                        continue
                    best_li = int(np.nanargmax(mean_corr_by_lag))
                    peak_lag_ms = lag_steps[best_li] * (LAG_BIN_WIDTH * 1000)
                    rows.append({
                        "reward_group": cohort, "session_id": session_id, "mouse_id": mouse_id,
                        "area_a": area_a, "area_b": area_b, "trial_type": trial_type, "lick_flag": lick_flag,
                        "n_trials": len(starts), "peak_lag_ms": peak_lag_ms,
                        "peak_corr": mean_corr_by_lag[best_li], "lag0_corr": mean_corr_by_lag[len(lag_steps) // 2],
                    })
            print(f"{session_id} ({cohort}) done")

    df = pd.DataFrame(rows)
    df.to_csv(ARTIFACTS_DIR / "peak_lag_results.csv", index=False)
    print(f"\nWrote {len(df)} rows to peak_lag_results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bins = np.arange(-100, 101, 10) - 5
    axes[0].hist(df["peak_lag_ms"], bins=bins, color="tab:purple", edgecolor="black", alpha=0.8)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel(f"peak lag (ms), area_b shifted rel. to area_a\n(positive = area_b lags area_a)")
    axes[0].set_ylabel("count (sessions x pairs x conditions)")
    axes[0].set_title(f"Distribution of peak lag across all {len(df)} interactions\n(all pairs, all conditions, good+MUA, variant A)")
    axes[0].set_box_aspect(1)

    for area_a, area_b in AREA_PAIRS:
        sub = df[(df.area_a == area_a) & (df.area_b == area_b)]
        label = f"{area_a.replace(' areas','').replace(' and pallidum','')} vs {area_b.replace(' areas','').replace(' and pallidum','')}"
        axes[1].hist(sub["peak_lag_ms"], bins=bins, alpha=0.5, label=f"{label} (n={len(sub)})")
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("peak lag (ms)")
    axes[1].set_ylabel("count")
    axes[1].set_title("Same distribution, split by area pair")
    axes[1].legend(fontsize=8)
    axes[1].set_box_aspect(1)

    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "peak_lag_distribution.png", dpi=130)
    plt.close(fig)
    print(f"Wrote peak_lag_distribution.png")
    print(f"\nMedian peak lag: {df['peak_lag_ms'].median():.1f}ms, "
          f"fraction with |peak_lag|<=2ms (i.e. essentially lag-0): {(df['peak_lag_ms'].abs()<=2).mean():.2f}")


if __name__ == "__main__":
    main()
