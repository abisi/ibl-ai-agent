"""Extended checkpoint per Axel's 2026-08-31 follow-up: (1) a multi-panel
pipeline figure (PSTH -> residual -> PCA -> CCA) showing variants A/B/C
together, (2) statistical test results per canonical dimension per cohort,
(3) a final cross-cohort/cross-analysis correlation-magnitude summary,
(4) good-only vs good+MUA quality tiers run separately, (5) restricted to
the **learning** day-stage only, KS4, across **a few mice per cohort**
(not yet the whole dataset) -- per the explicit follow-up request.

Area pair: coarse level, "Motor and frontal areas" vs "Somatosensory areas"
(the richest coarse pair in the Step-1 coverage report). 3 R+ + 3 R-
learning-arm sessions, picked from valid_area_pairs.parquet by unit count.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

AREA_A = "Motor and frontal areas"
AREA_B = "Somatosensory areas"
N_UNITS_SUBSAMPLE = 30
N_DIMS = 10
N_FOLDS = 5

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


def get_session_units(unit_table: pd.DataFrame, session_id: str, tier_mask_fn, rng: np.random.Generator):
    sess = unit_table[(unit_table["session_id"] == session_id)]
    sess = sess[tier_mask_fn(sess)]
    units_a = sess[sess["area_group_coarse"] == AREA_A]
    units_b = sess[sess["area_group_coarse"] == AREA_B]
    others = sess[~sess["area_group_coarse"].isin([AREA_A, AREA_B])]
    n = min(N_UNITS_SUBSAMPLE, len(units_a), len(units_b))
    units_a = units_a.sample(n=n, random_state=int(rng.integers(1e9)))
    units_b = units_b.sample(n=n, random_state=int(rng.integers(1e9)))
    return units_a, units_b, others


def spikes_of(units: pd.DataFrame) -> list[np.ndarray]:
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def run_one(units_a, units_b, others, starts, is_whisker, bin_edges, lick_times):
    spikes_a, spikes_b = spikes_of(units_a), spikes_of(units_b)
    tensor_a = cca_lib.population_tensor(spikes_a, starts, is_whisker, bin_edges)
    tensor_b = cca_lib.population_tensor(spikes_b, starts, is_whisker, bin_edges)
    resid_a = cca_lib.noise_correlation_residuals(tensor_a)
    resid_b = cca_lib.noise_correlation_residuals(tensor_b)
    flat_a, valid = cca_lib.flatten_trial_bins(resid_a)
    flat_b, _ = cca_lib.flatten_trial_bins(resid_b)

    results = {}
    results["A"] = cca_lib.cv_canonical_correlation(flat_a, flat_b, N_DIMS, N_FOLDS)

    if len(others) >= 5:
        from sklearn.decomposition import PCA as _PCA
        spikes_o = spikes_of(others)
        tensor_o = cca_lib.population_tensor(spikes_o, starts, is_whisker, bin_edges)
        resid_o = cca_lib.noise_correlation_residuals(tensor_o)
        # restrict to the same valid (trial,bin) samples as A/B (dead-zone NaN
        # pattern is per-trial-type/bin, not per-unit, so the same `valid` mask
        # applies), then reduce to leading PCs (fit on all valid samples -- a
        # documented simplification: only the final nuisance-removal regression
        # inside cv_canonical_correlation is fit train-only per CV fold, not
        # this basis) as the variant-B nuisance regressor.
        n_trials_o, n_bins_o, n_o = resid_o.shape
        flat_o = resid_o.reshape(n_trials_o * n_bins_o, n_o)[valid]
        pca_o = _PCA(n_components=min(15, flat_o.shape[1])).fit(flat_o)
        cum_var = np.cumsum(pca_o.explained_variance_ratio_)
        k = min(int(np.searchsorted(cum_var, 0.90) + 1), flat_o.shape[1])
        nuisance_b = pca_o.transform(flat_o)[:, :k]
        results["B"] = cca_lib.cv_canonical_correlation(flat_a, flat_b, N_DIMS, N_FOLDS, nuisance=nuisance_b)
    else:
        results["B"] = None

    if lick_times is not None and len(lick_times):
        lick_reg = cca_lib.lick_rate_regressor(lick_times, starts, bin_edges)
        lick_flat = lick_reg.reshape(-1, 1)[valid]
        results["C"] = cca_lib.cv_canonical_correlation(flat_a, flat_b, N_DIMS, N_FOLDS, nuisance=lick_flat)
    else:
        results["C"] = None

    return results, dict(tensor_a=tensor_a, tensor_b=tensor_b, resid_a=resid_a, resid_b=resid_b,
                          flat_a=flat_a, flat_b=flat_b, valid=valid, n_trials=len(starts))


def main() -> None:
    unit_table, trial_table = cov_lib.load_units(ALL_FILES, day_to_analyze="learning", max_workers=6)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    events = pd.read_parquet(SSL_EPHYS_DIR / "metadata" / "events.parquet")

    bin_edges = cca_lib.time_bin_edges()
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    rng = np.random.default_rng(42)

    rows = []
    example_bundle = None  # for the pipeline-illustration figure
    for tier_name, tier_fn in QUALITY_TIERS.items():
        for cohort, session_ids in SESSIONS.items():
            for session_id in session_ids:
                units_a, units_b, others = get_session_units(unit_table, session_id, tier_fn, rng)
                mouse_id = session_id.split("_")[0]
                trials = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
                lick_times = np.sort(events[(events["session_id"] == session_id)
                                             & (events["event_type"] == "piezo_lick_times")]["time"].to_numpy(dtype=float))
                print(f"[{tier_name}] {session_id} ({cohort}, mouse {mouse_id}): "
                      f"{len(units_a)}/{len(units_b)} units A/B, {len(others)} other units, {len(lick_times)} licks")

                for trial_type, is_whisker in [("whisker_trial", True), ("auditory_trial", False)]:
                    tt = trials[trials["trial_type"] == trial_type].sort_values("start_time")
                    starts = tt["start_time"].to_numpy(dtype=float)
                    if len(starts) < 10:
                        print(f"  skip {trial_type}: only {len(starts)} trials")
                        continue
                    results, bundle = run_one(units_a, units_b, others, starts, is_whisker, bin_edges, lick_times)

                    if (example_bundle is None and tier_name == "good_mua" and trial_type == "whisker_trial"
                            and cohort == "R-" and session_id == "AB126_20240822_114405"):
                        example_bundle = dict(session_id=session_id, cohort=cohort, bundle=bundle,
                                               results=results, bin_centers=bin_centers)

                    for variant, res in results.items():
                        if res is None:
                            continue
                        for d in range(res["n_dims"]):
                            rows.append({
                                "quality_tier": tier_name, "reward_group": cohort, "session_id": session_id,
                                "mouse_id": mouse_id, "trial_type": trial_type, "variant": variant,
                                "dimension": d + 1, "cca_mean": res["cca_mean"][d], "cca_sd": res["cca_sd"][d],
                                "pca_mean": res["pca_mean"][d], "pca_sd": res["pca_sd"][d],
                            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(ARTIFACTS_DIR / "multi_session_checkpoint_results.csv", index=False)
    print(f"\nWrote {len(results_df)} rows to multi_session_checkpoint_results.csv")

    # ================= Figure 1: pipeline illustration (PSTH -> residual -> PCA -> CCA), variants A/B/C together =================
    if example_bundle is not None:
        b = example_bundle["bundle"]
        res = example_bundle["results"]
        bc = example_bundle["bin_centers"]
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))

        total_counts = np.nansum(b["tensor_a"], axis=(0, 1))
        good_units = None  # placeholder, quality already applied upstream at 'good_mua' tier
        unit_idx = int(np.argsort(total_counts)[len(total_counts) // 2])  # median-activity unit, not the max (avoids MUA-like outliers)
        for ti in range(min(15, b["tensor_a"].shape[0])):
            axes[0].plot(bc * 1000, b["tensor_a"][ti, :, unit_idx], color="gray", alpha=0.35, linewidth=0.7)
        psth = np.nanmean(b["tensor_a"][:, :, unit_idx], axis=0)
        axes[0].plot(bc * 1000, psth, color="black", linewidth=2)
        axes[0].set_title("1. Single trials + PSTH\n(example unit, area A)")
        axes[0].set_xlabel("time from start_time (ms)"); axes[0].set_ylabel("rate (Hz)")

        for ti in range(min(15, b["resid_a"].shape[0])):
            axes[1].plot(bc * 1000, b["resid_a"][ti, :, unit_idx], color="tab:blue", alpha=0.35, linewidth=0.7)
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_title("2. Noise-correlation residual\n(PSTH subtracted, no baseline corr.)")
        axes[1].set_xlabel("time from start_time (ms)"); axes[1].set_ylabel("residual rate (Hz)")

        from sklearn.decomposition import PCA as _PCA
        pca_a = _PCA(n_components=5).fit(b["flat_a"])
        pca_b = _PCA(n_components=5).fit(b["flat_b"])
        axes[2].plot(np.arange(1, 6), np.cumsum(pca_a.explained_variance_ratio_), "o-", label="area A")
        axes[2].plot(np.arange(1, 6), np.cumsum(pca_b.explained_variance_ratio_), "s-", label="area B")
        axes[2].set_title("3. PCA on residuals\n(cumulative variance explained)")
        axes[2].set_xlabel("PC #"); axes[2].set_ylabel("cum. variance explained"); axes[2].legend(fontsize=8)

        variant_labels = {"A": "A: plain CCA", "B": "B: partial out\nother neurons", "C": "C: partial out\nlick rate"}
        xs, means, sds = [], [], []
        for i, (v, label) in enumerate(variant_labels.items()):
            if res.get(v) is None:
                continue
            xs.append(label); means.append(res[v]["cca_mean"][0]); sds.append(res[v]["cca_sd"][0])
        axes[3].bar(xs, means, yerr=sds, capsize=4, color=["tab:blue", "tab:orange", "tab:green"][:len(xs)])
        axes[3].axhline(res["A"]["pca_mean"][0], color="black", linestyle="--", label="PCA baseline (variant A)")
        axes[3].set_title("4. CCA/pCCA result, all variants\n(dim 1, held-out)")
        axes[3].set_ylabel("held-out canonical correlation"); axes[3].legend(fontsize=8)

        fig.suptitle(f"Pipeline illustration -- session {example_bundle['session_id']} ({example_bundle['cohort']}), "
                     f"{AREA_A} vs {AREA_B}, whisker trials, good+MUA tier. Error bars = SD across {N_FOLDS} CV folds.")
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / "checkpoint2_01_pipeline_illustration.png", dpi=130)
        plt.close(fig)
        print("Wrote checkpoint2_01_pipeline_illustration.png")

    # ================= Figure 2: scree plots per cohort, per tier, with per-dimension stats =================
    stat_rows = []
    for tier_name in QUALITY_TIERS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        for ax, trial_type in zip(axes, ["whisker_trial", "auditory_trial"]):
            for cohort, color in [("R+", "tab:red"), ("R-", "tab:blue")]:
                sub = results_df[(results_df.quality_tier == tier_name) & (results_df.trial_type == trial_type)
                                  & (results_df.reward_group == cohort) & (results_df.variant == "A")]
                per_dim = sub.groupby("dimension")["cca_mean"].agg(["mean", "sem"]).reindex(range(1, N_DIMS + 1))
                ax.errorbar(per_dim.index, per_dim["mean"], yerr=per_dim["sem"], label=f"{cohort} (n={sub.session_id.nunique()} sessions)",
                            color=color, marker="o", capsize=3)

                # per-dimension paired test: CCA vs PCA baseline, within this cohort,
                # one paired observation per session (session-level mean across its 5 CV folds).
                for d in range(1, N_DIMS + 1):
                    dd = sub[sub.dimension == d]
                    pca_dd = results_df[(results_df.quality_tier == tier_name) & (results_df.trial_type == trial_type)
                                         & (results_df.reward_group == cohort) & (results_df.variant == "A")
                                         & (results_df.dimension == d)]
                    if len(dd) >= 2:
                        cca_vals = dd.sort_values("session_id")["cca_mean"].to_numpy()
                        pca_vals = pca_dd.sort_values("session_id")["pca_mean"].to_numpy()
                        try:
                            w_stat, w_p = scipy_stats.wilcoxon(cca_vals, pca_vals)
                        except ValueError:
                            w_p = np.nan
                        t_stat, t_p = scipy_stats.ttest_rel(cca_vals, pca_vals)
                        stat_rows.append({"quality_tier": tier_name, "trial_type": trial_type, "reward_group": cohort,
                                           "dimension": d, "n_sessions": len(dd),
                                           "mean_cca_minus_pca": float(np.mean(cca_vals - pca_vals)),
                                           "wilcoxon_p": w_p, "paired_ttest_p": t_p})
            ax.set_xlabel("canonical dimension"); ax.set_title(f"{trial_type} ({tier_name})")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("held-out canonical correlation\n(mean +/- SEM across sessions)")
        fig.suptitle(f"Variant A scree plot by cohort, quality tier = {tier_name} "
                      f"({AREA_A} vs {AREA_B}, learning arm, {len(SESSIONS['R+'])}+{len(SESSIONS['R-'])} sessions)")
        fig.tight_layout()
        fig.savefig(ARTIFACTS_DIR / f"checkpoint2_02_scree_{tier_name}.png", dpi=130)
        plt.close(fig)
        print(f"Wrote checkpoint2_02_scree_{tier_name}.png")

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(ARTIFACTS_DIR / "multi_session_checkpoint_dim_stats.csv", index=False)
    print(f"\nPer-dimension CCA-vs-PCA-baseline stats (n_sessions={len(SESSIONS['R+'])}/cohort -- LOW POWER, "
          f"a Wilcoxon signed-rank test with n=3 cannot reach p<0.05; shown as descriptive at this checkpoint scale):")
    print(stat_df[(stat_df.dimension <= 3)].to_string(index=False))

    # ================= Final summary: correlation magnitude across cohort x variant x tier x trial_type =================
    summary = (results_df[results_df.dimension == 1]
               .groupby(["quality_tier", "reward_group", "variant", "trial_type"])
               .agg(mean_dim1_corr=("cca_mean", "mean"), sem_dim1_corr=("cca_mean", "sem"),
                    mean_pca_baseline=("pca_mean", "mean"), n_sessions=("session_id", "nunique"))
               .reset_index())
    summary.to_csv(ARTIFACTS_DIR / "multi_session_checkpoint_summary.csv", index=False)
    print("\n=== Final summary: dim-1 held-out canonical correlation (mean +/- SEM across sessions) ===")
    print("(error bars in the CSV/figure are SEM across the per-session means, each of which is itself")
    print(" a mean +/- SD across 5 CV folds -- see multi_session_checkpoint_results.csv for the fold-level SDs)")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(14, 6))
    summary["group"] = summary["quality_tier"] + " / " + summary["trial_type"].str.replace("_trial", "")
    groups = sorted(summary["group"].unique())
    variants = ["A", "B", "C"]
    cohorts = ["R+", "R-"]
    width = 0.12
    x = np.arange(len(groups))
    for i, (variant, cohort) in enumerate([(v, c) for v in variants for c in cohorts]):
        vals, errs = [], []
        for g in groups:
            row = summary[(summary.group == g) & (summary.variant == variant) & (summary.reward_group == cohort)]
            vals.append(row["mean_dim1_corr"].iloc[0] if len(row) else np.nan)
            errs.append(row["sem_dim1_corr"].iloc[0] if len(row) else np.nan)
        offset = (i - 2.5) * width
        color = {"A": "tab:blue", "B": "tab:orange", "C": "tab:green"}[variant]
        hatch = "" if cohort == "R+" else "//"
        ax.bar(x + offset, vals, width, yerr=errs, capsize=2, color=color, hatch=hatch,
               label=f"{variant} {cohort}", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("dim-1 held-out canonical correlation\n(mean +/- SEM across sessions)")
    ax.set_title(f"Summary: correlation magnitude across cohort x variant x quality tier x trial type\n"
                 f"({AREA_A} vs {AREA_B}, learning arm, {len(SESSIONS['R+'])} R+ / {len(SESSIONS['R-'])} R- sessions)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "checkpoint2_03_final_summary.png", dpi=130)
    plt.close(fig)
    print("\nWrote checkpoint2_03_final_summary.png")


if __name__ == "__main__":
    main()
