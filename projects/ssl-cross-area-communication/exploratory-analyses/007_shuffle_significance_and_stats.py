"""Checkpoint extension per Axel's 2026-08-31 follow-up (still the same
small subset -- 6 learning-arm sessions, 3 R+ + 3 R-):
- Square plots, all axes labeled.
- Trial-identity shuffle control (lag=0 only) for variants A/B/C, on
  trial-by-trial canonical-variate correlation at each 10ms bin.
- Artifact dead zone ALWAYS excluded (as reference bin and as shifted
  target bin) in the lagged analysis, not just NaN-masked.
- Lagged matrix figure: canonical-level (existing) + a new neuron-level
  version (raw pairwise correlation, no CCA), both annotating the excluded
  dead-zone bin. No shuffle control for lags != 0.
- R+ vs R- significance testing (Mann-Whitney + Welch, the project's
  standard pair) per dimension, per variant, per tier, per trial type.
- Up to 20 canonical dimensions; variant-B nuisance PCA keeps as many
  components as reach 99% variance (was 90%/15 in the checkpoint).
- Summary: which canonical dimensions are significantly above the lag-0
  shuffle null, aggregated across every (session, tier, trial_type,
  variant) combination run here.
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
from scipy import stats as scipy_stats
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
AREA_A, AREA_B = "Motor and frontal areas", "Somatosensory areas"
N_UNITS_SUBSAMPLE = 30
N_DIMS_MAX = 20
N_SHUFFLES = 500
LAG_BINS = np.arange(-10, 11)  # +/-100ms, 10ms steps
EPS = 1e-12

SESSIONS = {
    "R+": ["AB127_20240821_103757", "AB130_20240902_123634", "AB125_20240817_123403"],
    "R-": ["MH034_20250514_104756", "AB126_20240822_114405", "AB085_20231005_152636"],
}
ALL_FILES = [f"{s}.nwb" for sessions in SESSIONS.values() for s in sessions]
EXAMPLE_SESSIONS = {"R+": "AB127_20240821_103757", "R-": "AB126_20240822_114405"}

QUALITY_TIERS = {
    "good": lambda ut: ut["quality_label"] == "good",
    "good_mua": lambda ut: ut["quality_label"].isin(["good", "mua"]),
}

REPO_ROOT = Path(__file__).resolve().parents[3]
SSL_EPHYS_DIR = REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0"


def square(ax):
    ax.set_box_aspect(1)


def flatten(tensor: np.ndarray) -> np.ndarray:
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


def unflatten(flat: np.ndarray, n_trials: int, n_bins: int) -> np.ndarray:
    return flat.reshape(n_trials, n_bins, -1)


def get_units(unit_table, session_id, tier_fn, rng):
    sess = unit_table[unit_table["session_id"] == session_id]
    sess = sess[tier_fn(sess)]
    units_a = sess[sess["area_group_coarse"] == AREA_A]
    units_b = sess[sess["area_group_coarse"] == AREA_B]
    others = sess[~sess["area_group_coarse"].isin([AREA_A, AREA_B])]
    n = min(N_UNITS_SUBSAMPLE, len(units_a), len(units_b))
    units_a = units_a.sample(n=n, random_state=int(rng.integers(1e9)))
    units_b = units_b.sample(n=n, random_state=int(rng.integers(1e9)))
    return units_a, units_b, others


def spikes_of(units: pd.DataFrame) -> list[np.ndarray]:
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def build_variant_train_test(resid_a, resid_b, resid_other, lick, variant, train_idx, test_idx, n_bins):
    """Returns (Xf_train, Yf_train, Xa_test_tensor, Xb_test_tensor), with the
    variant's nuisance regression fit on train only and applied to both."""
    if variant == "A":
        Xf_train = flatten(resid_a[train_idx])
        Yf_train = flatten(resid_b[train_idx])
        return Xf_train, Yf_train, resid_a[test_idx], resid_b[test_idx]

    if variant == "B":
        flat_other_train = flatten(resid_other[train_idx])
        pca_o, k = cca_lib.fit_nuisance_pca(flat_other_train)
        nuisance_train = pca_o.transform(flat_other_train)[:, :k]
        nuisance_test = pca_o.transform(flatten(resid_other[test_idx]))[:, :k]
    else:  # "C"
        nuisance_train = flatten(lick[train_idx])
        nuisance_test = flatten(lick[test_idx])

    flat_a_train, flat_b_train = flatten(resid_a[train_idx]), flatten(resid_b[train_idx])
    reg_a = LinearRegression().fit(nuisance_train, flat_a_train)
    reg_b = LinearRegression().fit(nuisance_train, flat_b_train)
    Xf_train = flat_a_train - reg_a.predict(nuisance_train)
    Yf_train = flat_b_train - reg_b.predict(nuisance_train)

    flat_a_test, flat_b_test = flatten(resid_a[test_idx]), flatten(resid_b[test_idx])
    Xa_test = unflatten(flat_a_test - reg_a.predict(nuisance_test), len(test_idx), n_bins)
    Xb_test = unflatten(flat_b_test - reg_b.predict(nuisance_test), len(test_idx), n_bins)
    return Xf_train, Yf_train, Xa_test, Xb_test


def lag0_shuffle_test(Ca_test, Cb_test, n_dims, rng):
    """(n_test, n_bins, n_dims) x2 -> observed mean-across-bins per-dim
    correlation, its shuffle null (trial identity shuffled, one permutation
    applied consistently across all bins/dims per shuffle draw), and a
    one-sided p-value (observed > null, since a true communication signal
    should be positive by construction of the CCA fit)."""
    n_test = Ca_test.shape[0]
    za = (Ca_test - Ca_test.mean(axis=0, keepdims=True)) / (Ca_test.std(axis=0, keepdims=True) + EPS)
    zb = (Cb_test - Cb_test.mean(axis=0, keepdims=True)) / (Cb_test.std(axis=0, keepdims=True) + EPS)
    observed = np.mean(za * zb, axis=(0, 1))  # (n_dims,) mean over trials then bins

    shuf_stats = np.empty((N_SHUFFLES, n_dims))
    for s in range(N_SHUFFLES):
        perm = rng.permutation(n_test)
        shuf_stats[s] = np.mean(za * zb[perm], axis=(0, 1))
    p_values = (1 + np.sum(shuf_stats >= observed[None, :], axis=0)) / (N_SHUFFLES + 1)
    return observed, shuf_stats.mean(axis=0), shuf_stats.std(axis=0), p_values


def main() -> None:
    unit_table, trial_table = cov_lib.load_units(ALL_FILES, day_to_analyze="learning", max_workers=6)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")

    bin_edges = cca_lib.time_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rng_master = np.random.default_rng(2026)

    rows = []
    example_cache = {}  # (cohort) -> dict for the matrix figures

    for tier_name, tier_fn in QUALITY_TIERS.items():
        for cohort, session_ids in SESSIONS.items():
            for session_id in session_ids:
                units_a, units_b, others = get_units(unit_table, session_id, tier_fn, rng_master)
                mouse_id = session_id.split("_")[0]
                trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
                lick_times = np.sort(events[(events["session_id"] == session_id)
                                             & (events["event_type"] == "piezo_lick_times")]["time"].to_numpy(dtype=float))

                for trial_type, is_whisker in [("whisker_trial", True), ("auditory_trial", False)]:
                    tt = trials_sess[trials_sess["trial_type"] == trial_type].sort_values("start_time")
                    starts = tt["start_time"].to_numpy(dtype=float)
                    if len(starts) < 20:
                        continue
                    dz_mask = cca_lib.dead_zone_bin_mask(bin_edges, is_whisker)

                    spikes_a, spikes_b = spikes_of(units_a), spikes_of(units_b)
                    tensor_a = cca_lib.population_tensor(spikes_a, starts, is_whisker, bin_edges)
                    tensor_b = cca_lib.population_tensor(spikes_b, starts, is_whisker, bin_edges)
                    resid_a = cca_lib.noise_correlation_residuals(tensor_a)
                    resid_b = cca_lib.noise_correlation_residuals(tensor_b)
                    n_bins = tensor_a.shape[1]

                    resid_other, lick_binned = None, None
                    if len(others) >= 5:
                        spikes_o = spikes_of(others)
                        tensor_o = cca_lib.population_tensor(spikes_o, starts, is_whisker, bin_edges)
                        resid_other = cca_lib.noise_correlation_residuals(tensor_o)
                    if len(lick_times):
                        lick_reg = cca_lib.lick_rate_regressor(lick_times, starts, bin_edges)
                        lick_binned = lick_reg[:, :, None]

                    n_trials_tot = len(starts)
                    perm = rng_master.permutation(n_trials_tot)
                    train_idx, test_idx = perm[: n_trials_tot // 2], perm[n_trials_tot // 2:]
                    n_dims = min(N_DIMS_MAX, len(units_a), len(units_b))

                    is_example = (tier_name == "good_mua" and trial_type == "whisker_trial"
                                  and session_id in EXAMPLE_SESSIONS.values())

                    # Dead zone ALWAYS excluded: sklearn's fit/predict/transform reject NaN
                    # outright (only project_tensor_multi's own internal masking is NaN-safe),
                    # so every fit/predict input here uses the bin axis pre-restricted to
                    # valid_bins -- never the full (dead-zone-including) tensors.
                    valid_bins = ~dz_mask
                    n_bins_v = int(valid_bins.sum())
                    resid_a_v, resid_b_v = resid_a[:, valid_bins, :], resid_b[:, valid_bins, :]
                    resid_other_v = resid_other[:, valid_bins, :] if resid_other is not None else None
                    lick_v = lick_binned[:, valid_bins, :] if lick_binned is not None else None

                    for variant in ["A", "B", "C"]:
                        if variant == "B" and resid_other_v is None:
                            continue
                        if variant == "C" and lick_v is None:
                            continue
                        Xf_train, Yf_train, Xa_test, Xb_test = build_variant_train_test(
                            resid_a_v, resid_b_v, resid_other_v, lick_v, variant, train_idx, test_idx, n_bins_v)

                        cca = CCA(n_components=n_dims, max_iter=2000).fit(Xf_train, Yf_train)
                        Ca_test = cca_lib.project_tensor_multi(Xa_test, cca, side="x", n_dims=n_dims)
                        Cb_test = cca_lib.project_tensor_multi(Xb_test, cca, side="y", n_dims=n_dims)

                        rng_shuf = np.random.default_rng(hash((tier_name, session_id, trial_type, variant)) % (2**32))
                        observed, shuf_mean, shuf_sd, p_values = lag0_shuffle_test(Ca_test, Cb_test, n_dims, rng_shuf)

                        for d in range(n_dims):
                            rows.append({
                                "quality_tier": tier_name, "reward_group": cohort, "session_id": session_id,
                                "mouse_id": mouse_id, "trial_type": trial_type, "variant": variant, "dimension": d + 1,
                                "observed_lag0_corr": observed[d], "shuffle_mean": shuf_mean[d], "shuffle_sd": shuf_sd[d],
                                "p_value": p_values[d],
                            })

                        if is_example and variant == "A":
                            example_cache[cohort] = dict(
                                session_id=session_id, resid_a=resid_a, resid_b=resid_b, dz_mask=dz_mask,
                                test_idx=test_idx, bin_centers=bin_centers,
                                Ca_test_full=cca_lib.project_tensor_multi(resid_a[test_idx], cca, side="x", n_dims=n_dims),
                                Cb_test_full=cca_lib.project_tensor_multi(resid_b[test_idx], cca, side="y", n_dims=n_dims),
                            )
                print(f"[{tier_name}] {session_id} ({cohort}) done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(ARTIFACTS_DIR / "shuffle_stats_results.csv", index=False)
    print(f"\nWrote {len(results_df)} rows to shuffle_stats_results.csv")

    # ================= R+ vs R- significance testing (Mann-Whitney + Welch, no other correlation control) =================
    cohort_rows = []
    for (tier, tt, variant, dim), grp in results_df.groupby(["quality_tier", "trial_type", "variant", "dimension"]):
        rplus = grp.loc[grp.reward_group == "R+", "observed_lag0_corr"].to_numpy()
        rminus = grp.loc[grp.reward_group == "R-", "observed_lag0_corr"].to_numpy()
        if len(rplus) < 2 or len(rminus) < 2:
            continue
        try:
            mw_stat, mw_p = scipy_stats.mannwhitneyu(rplus, rminus, alternative="two-sided")
        except ValueError:
            mw_p = np.nan
        t_stat, t_p = scipy_stats.ttest_ind(rplus, rminus, equal_var=False)
        cohort_rows.append({"quality_tier": tier, "trial_type": tt, "variant": variant, "dimension": dim,
                             "n_Rplus": len(rplus), "n_Rminus": len(rminus),
                             "mean_Rplus": rplus.mean(), "mean_Rminus": rminus.mean(),
                             "mannwhitney_p": mw_p, "welch_p": t_p})
    cohort_stats_df = pd.DataFrame(cohort_rows)
    cohort_stats_df.to_csv(ARTIFACTS_DIR / "shuffle_stats_Rplus_vs_Rminus.csv", index=False)
    print(f"Wrote {len(cohort_stats_df)} rows to shuffle_stats_Rplus_vs_Rminus.csv")

    # ================= Which dimensions are significant vs. shuffle, across all iterations =================
    fig, ax = plt.subplots(figsize=(7, 7))
    frac_sig = results_df.groupby("dimension")["p_value"].apply(lambda p: (p < 0.05).mean())
    mean_corr = results_df.groupby("dimension")["observed_lag0_corr"].mean()
    ax2 = ax.twinx()
    ax.bar(frac_sig.index, frac_sig.values, color="tab:blue", alpha=0.6, label="fraction of iterations p<0.05")
    ax2.plot(mean_corr.index, mean_corr.values, color="tab:red", marker="o", label="mean observed corr")
    ax.set_xlabel("canonical dimension"); ax.set_ylabel("fraction of iterations significant (p<0.05)", color="tab:blue")
    ax2.set_ylabel("mean observed lag-0 correlation (all iterations)", color="tab:red")
    ax.set_title(f"Which canonical dimensions are truly correlated?\n(lag-0 shuffle test, {N_SHUFFLES} shuffles/iteration, "
                 f"n={results_df.groupby(['quality_tier','session_id','trial_type','variant']).ngroups} iterations)")
    square(ax)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "checkpoint4_01_dims_significant_vs_shuffle.png", dpi=130)
    plt.close(fig)
    print("Wrote checkpoint4_01_dims_significant_vs_shuffle.png")
    print(pd.concat([frac_sig.rename("frac_sig"), mean_corr.rename("mean_corr")], axis=1).to_string())

    # ================= Canonical-level and neuron-level lagged matrices for the 2 example sessions =================
    fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 7))
    fig_n, axes_n = plt.subplots(1, 2, figsize=(14, 7))
    for i, (cohort, session_id) in enumerate(EXAMPLE_SESSIONS.items()):
        cache = example_cache[cohort]
        dz_mask = cache["dz_mask"]
        Ca_full, Cb_full = cache["Ca_test_full"][:, :, 0], cache["Cb_test_full"][:, :, 0]  # dim 1
        n_bins = Ca_full.shape[1]

        mat_c = np.full((n_bins, len(LAG_BINS)), np.nan)
        for li, lag in enumerate(LAG_BINS):
            for t in range(n_bins):
                t2 = t + lag
                if t2 < 0 or t2 >= n_bins or dz_mask[t] or dz_mask[t2]:
                    continue
                a, b = Ca_full[:, t], Cb_full[:, t2]
                m = ~(np.isnan(a) | np.isnan(b))
                if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
                    continue
                mat_c[t, li] = np.corrcoef(a[m], b[m])[0, 1]

        # neuron-level: raw pairwise correlation, no CCA, from the example session's own subsampled units.
        resid_a_test = cache["resid_a"][cache["test_idx"]]
        resid_b_test = cache["resid_b"][cache["test_idx"]]
        n_test = resid_a_test.shape[0]
        za_full = (resid_a_test - np.nanmean(resid_a_test, axis=0, keepdims=True)) / (np.nanstd(resid_a_test, axis=0, keepdims=True) + EPS)
        zb_full = (resid_b_test - np.nanmean(resid_b_test, axis=0, keepdims=True)) / (np.nanstd(resid_b_test, axis=0, keepdims=True) + EPS)
        mat_n = np.full((n_bins, len(LAG_BINS)), np.nan)
        for li, lag in enumerate(LAG_BINS):
            for t in range(n_bins):
                t2 = t + lag
                if t2 < 0 or t2 >= n_bins or dz_mask[t] or dz_mask[t2]:
                    continue
                pairwise = za_full[:, t, :].T @ zb_full[:, t2, :] / n_test
                mat_n[t, li] = np.nanmean(pairwise)

        for fig, axes, mat, label in [(fig_c, axes_c, mat_c, "canonical dim-1"), (fig_n, axes_n, mat_n, "neuron-level (mean pairwise)")]:
            ax = axes[i]
            extent = [bin_centers[0] * 1000, bin_centers[-1] * 1000, LAG_BINS[0] * 10, LAG_BINS[-1] * 10]
            im = ax.imshow(mat.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5, extent=extent)
            dz_time_ms = bin_centers[dz_mask] * 1000
            for dzt in dz_time_ms:
                ax.axvline(dzt, color="black", linewidth=1.5, alpha=0.5)
            if len(dz_time_ms):
                ax.text(dz_time_ms[0], LAG_BINS[-1] * 10 * 0.9, " dead zone\n excluded", fontsize=7, color="black")
            ax.set_xlabel("time from start_time (ms)")
            ax.set_ylabel(f"lag (ms), {AREA_B} shifted rel. to {AREA_A}")
            ax.set_title(f"{cohort} {session_id}\n{label}")
            square(ax)
            fig.colorbar(im, ax=ax, label="across-trial correlation", fraction=0.046)

    fig_c.suptitle(f"Lagged canonical correlation matrix -- {AREA_A} vs {AREA_B}, whisker, good+MUA, held-out")
    fig_c.tight_layout()
    fig_c.savefig(ARTIFACTS_DIR / "checkpoint4_02_lagged_matrix_canonical.png", dpi=130)
    fig_n.suptitle(f"Lagged NEURON-LEVEL correlation matrix (mean pairwise, no CCA) -- {AREA_A} vs {AREA_B}, "
                   f"whisker, good+MUA, held-out")
    fig_n.tight_layout()
    fig_n.savefig(ARTIFACTS_DIR / "checkpoint4_03_lagged_matrix_neuron_level.png", dpi=130)
    print("Wrote checkpoint4_02_lagged_matrix_canonical.png, checkpoint4_03_lagged_matrix_neuron_level.png")


if __name__ == "__main__":
    main()
