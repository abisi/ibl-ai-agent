"""New rules per Axel's 2026-08-31 follow-up: 5ms analysis bins (dead zone
still excluded), 2ms-resolution lagged pCCA, tier-specific unit floors
(>=10 good OR >=30 good+MUA per area, replacing the flat >=20 floor), and a
canonical-correlation comparison across MULTIPLE area pairs (not just
Motor-frontal vs Somatosensory). Still the same 6-session checkpoint subset
(learning arm, 3 R+ + 3 R-) -- uses partial_CCA (see 008) and the corrected
refit-per-shuffle null (see 008's docstring for why).

Area pairs: of the 3 areas eligible (>=10 good or >=30 good+MUA) in ALL 6
sessions -- Motor and frontal areas, Somatosensory areas, Striatum and
pallidum (checked directly against full_unit_table_metadata.parquet before
writing this script) -- all 3 pairwise combinations are run.
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
AREAS = ["Motor and frontal areas", "Somatosensory areas", "Striatum and pallidum"]
AREA_PAIRS = [("Motor and frontal areas", "Somatosensory areas"),
              ("Motor and frontal areas", "Striatum and pallidum"),
              ("Somatosensory areas", "Striatum and pallidum")]
N_UNITS_CAP = 30                 # max units/area used for fitting (unchanged)
MIN_UNITS_GOOD = 10              # NEW tier-specific floor
MIN_UNITS_GOOD_MUA = 30          # NEW tier-specific floor
N_DIMS_MAX = 20
N_SHUFFLES = 500
REGULARIZATION = 1e-3

BIN_WIDTH = 0.005                # NEW: 5ms (was 10ms)
LAG_BIN_WIDTH = 0.002             # NEW: 2ms lag resolution
LAG_RANGE_S = 0.100                # +/-100ms, same window as before, finer steps
WINDOW = cca_lib.WINDOW

SESSIONS = {
    "R+": ["AB127_20240821_103757", "AB130_20240902_123634", "AB125_20240817_123403"],
    "R-": ["MH034_20250514_104756", "AB126_20240822_114405", "AB085_20231005_152636"],
}
ALL_FILES = [f"{s}.nwb" for sessions in SESSIONS.values() for s in sessions]
LAG_EXAMPLE_SESSIONS = {"R+": "AB127_20240821_103757", "R-": "AB126_20240822_114405"}
LAG_EXAMPLE_PAIR = ("Motor and frontal areas", "Somatosensory areas")  # keep lag-matrix scope bounded

QUALITY_TIERS = {
    "good": (lambda ut: ut["quality_label"] == "good", MIN_UNITS_GOOD),
    "good_mua": (lambda ut: ut["quality_label"].isin(["good", "mua"]), MIN_UNITS_GOOD_MUA),
}

REPO_ROOT = Path(__file__).resolve().parents[3]
SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"


def square(ax):
    ax.set_box_aspect(1)


def flatten(tensor: np.ndarray) -> np.ndarray:
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


def get_units_for_area(unit_table, session_id, area, tier_fn, min_units, rng):
    sess = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == area)]
    sess = sess[tier_fn(sess)]
    if len(sess) < min_units:
        return None
    n = min(N_UNITS_CAP, len(sess))
    return sess.sample(n=n, random_state=int(rng.integers(1e9)))


def spikes_of(units: pd.DataFrame) -> list[np.ndarray]:
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def main() -> None:
    unit_table, trial_table = cov_lib.load_units(ALL_FILES, day_to_analyze="learning", max_workers=6)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)

    bin_edges = cca_lib.time_bin_edges(WINDOW, BIN_WIDTH)
    rng_master = np.random.default_rng(31)

    rows = []
    lag_cache = {}

    for tier_name, (tier_fn, min_units) in QUALITY_TIERS.items():
        for cohort, session_ids in SESSIONS.items():
            for session_id in session_ids:
                trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
                mouse_id = session_id.split("_")[0]

                for area_a, area_b in AREA_PAIRS:
                    units_a = get_units_for_area(unit_table, session_id, area_a, tier_fn, min_units, rng_master)
                    units_b = get_units_for_area(unit_table, session_id, area_b, tier_fn, min_units, rng_master)
                    if units_a is None or units_b is None:
                        continue

                    for trial_type, is_whisker in [("whisker_trial", True), ("auditory_trial", False)]:
                        tt = trials_sess[trials_sess["trial_type"] == trial_type].sort_values("start_time")
                        starts = tt["start_time"].to_numpy(dtype=float)
                        if len(starts) < 20:
                            continue
                        dz_mask = cca_lib.dead_zone_bin_mask(bin_edges, is_whisker)
                        valid_bins = ~dz_mask

                        spikes_a, spikes_b = spikes_of(units_a), spikes_of(units_b)
                        tensor_a = cca_lib.population_tensor(spikes_a, starts, is_whisker, bin_edges)
                        tensor_b = cca_lib.population_tensor(spikes_b, starts, is_whisker, bin_edges)
                        resid_a_v = cca_lib.noise_correlation_residuals(tensor_a)[:, valid_bins, :]
                        resid_b_v = cca_lib.noise_correlation_residuals(tensor_b)[:, valid_bins, :]

                        n_dims = min(N_DIMS_MAX, len(units_a), len(units_b))
                        X, Y = flatten(resid_a_v), flatten(resid_b_v)
                        model = PartialCCA(regularization=REGULARIZATION).fit(X, Y, None, verbose=False)
                        observed = model.canonical_correlations_[:n_dims]

                        n_trials = resid_b_v.shape[0]
                        rng_shuf = np.random.default_rng(hash((tier_name, session_id, area_a, area_b, trial_type)) % (2**32))
                        shuf_corrs = np.full((N_SHUFFLES, n_dims), np.nan)
                        for s in range(N_SHUFFLES):
                            perm = rng_shuf.permutation(n_trials)
                            m_shuf = PartialCCA(regularization=REGULARIZATION).fit(X, flatten(resid_b_v[perm]), None, verbose=False)
                            shuf_corrs[s] = m_shuf.canonical_correlations_[:n_dims]
                        null_mean, null_sem = shuf_corrs.mean(axis=0), shuf_corrs.std(axis=0) / np.sqrt(N_SHUFFLES)
                        p_values = (1 + np.sum(shuf_corrs >= observed[None, :], axis=0)) / (N_SHUFFLES + 1)

                        for d in range(n_dims):
                            rows.append({
                                "quality_tier": tier_name, "reward_group": cohort, "session_id": session_id,
                                "mouse_id": mouse_id, "area_a": area_a, "area_b": area_b, "trial_type": trial_type,
                                "n_units_a": len(units_a), "n_units_b": len(units_b), "dimension": d + 1,
                                "observed_corr": observed[d], "shuffle_null_mean": null_mean[d],
                                "shuffle_null_sem": null_sem[d], "p_value": p_values[d],
                            })

                        if (tier_name == "good_mua" and trial_type == "whisker_trial"
                                and (area_a, area_b) == LAG_EXAMPLE_PAIR
                                and session_id in LAG_EXAMPLE_SESSIONS.values()):
                            # Cache the EXACT units_a/units_b (identity + order) the model
                            # was fit on -- model.transform() indexes features by fit-time
                            # order, so the lag section must reuse this same population,
                            # not a freshly reselected random subsample.
                            lag_cache[cohort] = dict(session_id=session_id, model=model, starts=starts,
                                                      is_whisker=is_whisker, units_a=units_a, units_b=units_b)
                print(f"[{tier_name}] {session_id} ({cohort}) done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(ARTIFACTS_DIR / "new_rules_results.csv", index=False)
    print(f"\nWrote {len(results_df)} rows (5ms bins, tier-specific floors, {len(AREA_PAIRS)} area pairs) "
          f"to new_rules_results.csv")

    # ================= Compare canonical correlation across areas AND dimensions =================
    for tier_name in QUALITY_TIERS:
        fig, ax = plt.subplots(figsize=(8, 8))
        sub = results_df[(results_df.quality_tier == tier_name) & (results_df.trial_type == "whisker_trial")]
        for area_a, area_b in AREA_PAIRS:
            pair_data = sub[(sub.area_a == area_a) & (sub.area_b == area_b)]
            per_dim = pair_data.groupby("dimension")["observed_corr"].agg(["mean", "sem"]).reindex(range(1, N_DIMS_MAX + 1))
            label = f"{area_a.replace(' areas', '').replace(' and pallidum', '')} vs {area_b.replace(' areas', '').replace(' and pallidum', '')}"
            ax.errorbar(per_dim.index, per_dim["mean"], yerr=per_dim["sem"], marker="o", capsize=3, label=label)
        ax.set_xlabel("canonical dimension")
        ax.set_ylabel("canonical correlation (mean +/- SEM across sessions x cohorts)")
        ax.set_title(f"Canonical correlation across area pairs and dimensions\n"
                     f"(variant A, whisker trials, {tier_name}, 5ms bins, dead zone excluded)")
        ax.legend(fontsize=8)
        square(ax)
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / f"newrules_01_areas_x_dims_{tier_name}.png", dpi=130)
        plt.close(fig)
        print(f"Wrote newrules_01_areas_x_dims_{tier_name}.png")

    # ================= 2ms-resolution lagged pCCA (Motor-Somatosensory, good+MUA, 2 example sessions) =================
    lag_bin_edges = cca_lib.time_bin_edges(WINDOW, LAG_BIN_WIDTH)
    lag_bin_centers = 0.5 * (lag_bin_edges[:-1] + lag_bin_edges[1:])
    lag_steps = np.arange(-int(LAG_RANGE_S / LAG_BIN_WIDTH), int(LAG_RANGE_S / LAG_BIN_WIDTH) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for i, (cohort, cache) in enumerate(lag_cache.items()):
        model, starts, is_whisker = cache["model"], cache["starts"], cache["is_whisker"]
        units_a, units_b = cache["units_a"], cache["units_b"]  # same population the model was fit on
        dz_mask_fine = cca_lib.dead_zone_bin_mask(lag_bin_edges, is_whisker)

        tensor_a_fine = cca_lib.population_tensor(spikes_of(units_a), starts, is_whisker, lag_bin_edges)
        tensor_b_fine = cca_lib.population_tensor(spikes_of(units_b), starts, is_whisker, lag_bin_edges)
        resid_a_fine = cca_lib.noise_correlation_residuals(tensor_a_fine)
        resid_b_fine = cca_lib.noise_correlation_residuals(tensor_b_fine)

        valid_fine = ~dz_mask_fine
        n_trials_f, n_bins_f = resid_a_fine.shape[0], resid_a_fine.shape[1]
        flat_a = resid_a_fine.reshape(n_trials_f * n_bins_f, -1)
        flat_b = resid_b_fine.reshape(n_trials_f * n_bins_f, -1)
        m = ~(np.isnan(flat_a).any(axis=1))
        proj_a_flat = np.full(n_trials_f * n_bins_f, np.nan)
        proj_b_flat = np.full(n_trials_f * n_bins_f, np.nan)
        pa, pb = model.transform(flat_a[m], flat_b[m])  # (n_dims, n_valid) -- dim 0 used
        proj_a_flat[m], proj_b_flat[m] = pa[0], pb[0]
        Ca = proj_a_flat.reshape(n_trials_f, n_bins_f)
        Cb = proj_b_flat.reshape(n_trials_f, n_bins_f)

        mat = np.full((n_bins_f, len(lag_steps)), np.nan)
        for li, lag in enumerate(lag_steps):
            for t in range(n_bins_f):
                t2 = t + lag
                if t2 < 0 or t2 >= n_bins_f or dz_mask_fine[t] or dz_mask_fine[t2]:
                    continue
                a, b = Ca[:, t], Cb[:, t2]
                mm = ~(np.isnan(a) | np.isnan(b))
                if mm.sum() < 10 or np.std(a[mm]) == 0 or np.std(b[mm]) == 0:
                    continue
                mat[t, li] = np.corrcoef(a[mm], b[mm])[0, 1]

        ax = axes[i]
        extent = [lag_bin_centers[0] * 1000, lag_bin_centers[-1] * 1000, lag_steps[0] * 2, lag_steps[-1] * 2]
        im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=extent)
        dz_t_ms = lag_bin_centers[dz_mask_fine] * 1000
        for dzt in dz_t_ms:
            ax.axvline(dzt, color="black", linewidth=1, alpha=0.5)
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel(f"lag (ms, 2ms steps), {LAG_EXAMPLE_PAIR[1]} shifted rel. to {LAG_EXAMPLE_PAIR[0]}")
        ax.set_title(f"{cohort} {cache['session_id']}")
        square(ax)
        fig.colorbar(im, ax=ax, label="across-trial correlation", fraction=0.046)

    fig.suptitle(f"2ms-resolution lagged canonical correlation -- {LAG_EXAMPLE_PAIR[0]} vs {LAG_EXAMPLE_PAIR[1]}, "
                 f"whisker trials, good+MUA, weights from the 5ms fit, evaluated on a 2ms-binned projection")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "newrules_02_lag_2ms_matrix.png", dpi=130)
    plt.close(fig)
    print("Wrote newrules_02_lag_2ms_matrix.png")


if __name__ == "__main__":
    main()
