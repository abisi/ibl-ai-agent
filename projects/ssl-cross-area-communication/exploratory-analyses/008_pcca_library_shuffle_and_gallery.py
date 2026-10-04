"""Pipeline switched to the published `partial_CCA` library (Gonzalez,
Buzsaki/Chen labs, MIT license) per Axel's 2026-08-31 request, replacing the
sklearn-CCA-based engine in scripts 004-007. Still the same 6-session
checkpoint subset (3 R+ + 3 R-, learning arm).

Shuffle test, re-explained and corrected: for each of N_SHUFFLES draws,
permute area B's TRIAL identity (keeping each trial's own within-trial bin
sequence intact, and keeping area A at its real trial order), then REFIT
PartialCCA from scratch on the shuffled data. This gives a genuine null
distribution of canonical correlation *per dimension* (the k-th shuffle
draw's k-th eigenvalue), not just a re-projection through the real,
already-fitted weights (the earlier 007 script's method, which tests a
narrower question -- "does this specific real axis stay correlated" --
rather than "could a spectrum this strong arise from chance structure
alone"). The real per-dimension correlation is then compared against this
per-dimension null: "outside of that range" = the standard one-sided
permutation p-value, (1 + #shuffles >= observed) / (N_SHUFFLES + 1).

Variant A only gets the shuffle-null (24 iterations x 500 shuffles = fast,
no nuisance regression needed); variants B/C get their real (non-shuffled)
correlation via the same new engine, for the results table only.

Also: a much larger PSTH-on-canonical-axes gallery (all 6 sessions, each
panel labeled with session_id/cohort/areas), and all figures redone square
with fully labeled axes.
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
AREA_A, AREA_B = "Motor and frontal areas", "Somatosensory areas"
N_UNITS_SUBSAMPLE = 30
N_DIMS_MAX = 20
N_SHUFFLES = 500
REGULARIZATION = 1e-3

SESSIONS = {
    "R+": ["AB127_20240821_103757", "AB130_20240902_123634", "AB125_20240817_123403"],
    "R-": ["MH034_20250514_104756", "AB126_20240822_114405", "AB085_20231005_152636"],
}
ALL_FILES = [f"{s}.nwb" for sessions in SESSIONS.values() for s in sessions]

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


def build_Z(resid_other_v, lick_v, variant):
    if variant == "A":
        return None
    if variant == "B":
        if resid_other_v is None:
            return None
        flat_o = flatten(resid_other_v)
        pca_o, k = cca_lib.fit_nuisance_pca(flat_o)
        return pca_o.transform(flat_o)[:, :k]
    if lick_v is None:
        return None
    return flatten(lick_v)


def main() -> None:
    unit_table, trial_table = cov_lib.load_units(ALL_FILES, day_to_analyze="learning", max_workers=6)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")

    bin_edges = cca_lib.time_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rng_master = np.random.default_rng(2026)

    rows = []
    psth_gallery = []  # per-session PSTH-projection panels (variant A, whisker trials)

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
                    valid_bins = ~dz_mask

                    spikes_a, spikes_b = spikes_of(units_a), spikes_of(units_b)
                    tensor_a = cca_lib.population_tensor(spikes_a, starts, is_whisker, bin_edges)
                    tensor_b = cca_lib.population_tensor(spikes_b, starts, is_whisker, bin_edges)
                    resid_a = cca_lib.noise_correlation_residuals(tensor_a)
                    resid_b = cca_lib.noise_correlation_residuals(tensor_b)
                    resid_a_v, resid_b_v = resid_a[:, valid_bins, :], resid_b[:, valid_bins, :]

                    resid_other_v, lick_v = None, None
                    if len(others) >= 5:
                        spikes_o = spikes_of(others)
                        tensor_o = cca_lib.population_tensor(spikes_o, starts, is_whisker, bin_edges)
                        resid_other_v = cca_lib.noise_correlation_residuals(tensor_o)[:, valid_bins, :]
                    if len(lick_times):
                        lick_reg = cca_lib.lick_rate_regressor(lick_times, starts, bin_edges)
                        lick_v = lick_reg[:, valid_bins, None]

                    n_dims = min(N_DIMS_MAX, len(units_a), len(units_b))
                    X = flatten(resid_a_v)

                    for variant in ["A", "B", "C"]:
                        Z = build_Z(resid_other_v, lick_v, variant)
                        if variant != "A" and Z is None:
                            continue
                        Y = flatten(resid_b_v)
                        model = PartialCCA(regularization=REGULARIZATION).fit(X, Y, Z, verbose=False)
                        observed = model.canonical_correlations_[:n_dims]

                        if variant == "A":
                            n_trials = resid_b_v.shape[0]
                            rng_shuf = np.random.default_rng(hash((tier_name, session_id, trial_type)) % (2**32))
                            shuf_corrs = np.full((N_SHUFFLES, n_dims), np.nan)
                            for s in range(N_SHUFFLES):
                                perm = rng_shuf.permutation(n_trials)
                                Y_shuf = flatten(resid_b_v[perm])
                                m_shuf = PartialCCA(regularization=REGULARIZATION).fit(X, Y_shuf, None, verbose=False)
                                shuf_corrs[s] = m_shuf.canonical_correlations_[:n_dims]
                            null_mean = shuf_corrs.mean(axis=0)
                            null_sem = shuf_corrs.std(axis=0) / np.sqrt(N_SHUFFLES)
                            p_values = (1 + np.sum(shuf_corrs >= observed[None, :], axis=0)) / (N_SHUFFLES + 1)
                        else:
                            null_mean = np.full(n_dims, np.nan)
                            null_sem = np.full(n_dims, np.nan)
                            p_values = np.full(n_dims, np.nan)

                        for d in range(n_dims):
                            rows.append({
                                "quality_tier": tier_name, "reward_group": cohort, "session_id": session_id,
                                "mouse_id": mouse_id, "trial_type": trial_type, "variant": variant, "dimension": d + 1,
                                "observed_corr": observed[d], "shuffle_null_mean": null_mean[d],
                                "shuffle_null_sem": null_sem[d], "p_value": p_values[d],
                            })

                        if variant == "A" and trial_type == "whisker_trial" and tier_name == "good_mua":
                            psth_a = np.nanmean(tensor_a, axis=0)[valid_bins, :]  # (n_bins_v, n_a)
                            psth_b = np.nanmean(tensor_b, axis=0)[valid_bins, :]
                            proj_a, proj_b = model.transform(psth_a, psth_b)  # (n_dims, n_bins_v)
                            r = np.corrcoef(proj_a[0], proj_b[0])[0, 1]
                            psth_gallery.append(dict(session_id=session_id, cohort=cohort, mouse_id=mouse_id,
                                                      bin_centers_v=bin_centers[valid_bins],
                                                      proj_a=proj_a[0], proj_b=proj_b[0], r=r))
                print(f"[{tier_name}] {session_id} ({cohort}) done")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(ARTIFACTS_DIR / "pcca_lib_results.csv", index=False)
    print(f"\nWrote {len(results_df)} rows to pcca_lib_results.csv")

    # ================= Scree plots with shuffle mean+SEM band and significance highlighting =================
    for tier_name in QUALITY_TIERS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
        for ax, trial_type in zip(axes, ["whisker_trial", "auditory_trial"]):
            sub_a = results_df[(results_df.quality_tier == tier_name) & (results_df.trial_type == trial_type)
                                & (results_df.variant == "A")]
            for cohort, color in [("R+", "tab:red"), ("R-", "tab:blue")]:
                coh = sub_a[sub_a.reward_group == cohort]
                obs = coh.groupby("dimension")["observed_corr"].agg(["mean", "sem"]).reindex(range(1, N_DIMS_MAX + 1))
                null_m = coh.groupby("dimension")["shuffle_null_mean"].mean().reindex(range(1, N_DIMS_MAX + 1))
                null_s = coh.groupby("dimension")["shuffle_null_sem"].mean().reindex(range(1, N_DIMS_MAX + 1))
                frac_sig = coh.groupby("dimension")["p_value"].apply(lambda p: (p < 0.05).mean()).reindex(range(1, N_DIMS_MAX + 1))

                ax.errorbar(obs.index, obs["mean"], yerr=obs["sem"], color=color, marker="o", capsize=3,
                            label=f"{cohort} observed (mean+/-SEM across sessions)")
                ax.fill_between(null_m.index, null_m - null_s, null_m + null_s, color=color, alpha=0.15,
                                 label=f"{cohort} shuffle null (mean+/-SEM, {N_SHUFFLES} shuffles)")
                sig_dims = frac_sig.index[frac_sig >= (2 / 3)]  # majority (>=2/3 sessions) individually significant
                if len(sig_dims):
                    ax.scatter(sig_dims, obs.loc[sig_dims, "mean"], marker="*", s=180, color=color,
                                edgecolor="black", zorder=5,
                                label=f"{cohort} significant (>=2/3 sessions p<0.05)" if cohort == "R+" else None)
            ax.set_xlabel("canonical dimension")
            ax.set_title(f"{trial_type} ({tier_name})")
            ax.legend(fontsize=6.5, loc="upper right")
            square(ax)
        axes[0].set_ylabel("canonical correlation")
        fig.suptitle(f"Correlation vs. dimension, variant A, quality tier = {tier_name} -- stars mark dimensions where\n"
                     f">=2/3 of that cohort's sessions individually beat their own trial-shuffle null (p<0.05, {N_SHUFFLES} refits/session)")
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / f"pcca_01_scree_shuffle_{tier_name}.png", dpi=130)
        plt.close(fig)
        print(f"Wrote pcca_01_scree_shuffle_{tier_name}.png")

    # ================= PSTH-projected-onto-canonical-axes gallery: all 6 sessions =================
    n_panels = len(psth_gallery)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, panel in zip(axes_flat, psth_gallery):
        ax.plot(panel["bin_centers_v"] * 1000, panel["proj_a"], color="tab:blue", label=f"{AREA_A} (canon. PSTH)")
        ax.plot(panel["bin_centers_v"] * 1000, panel["proj_b"], color="tab:orange", label=f"{AREA_B} (canon. PSTH)")
        ax.set_title(f"{panel['cohort']} {panel['session_id']}\n(mouse {panel['mouse_id']}), r={panel['r']:.2f}", fontsize=10)
        ax.set_xlabel("time from start_time (ms)")
        ax.set_ylabel("canonical-variate amplitude")
        ax.legend(fontsize=7)
        square(ax)
    for ax in axes_flat[n_panels:]:
        ax.axis("off")
    fig.suptitle(f"PSTH projected onto canonical axes (dim 1) -- {AREA_A} vs {AREA_B}, whisker trials, good+MUA, "
                 f"variant A, all {n_panels} sessions")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "pcca_02_psth_canonical_gallery.png", dpi=130)
    plt.close(fig)
    print(f"Wrote pcca_02_psth_canonical_gallery.png ({n_panels} panels)")


if __name__ == "__main__":
    main()
