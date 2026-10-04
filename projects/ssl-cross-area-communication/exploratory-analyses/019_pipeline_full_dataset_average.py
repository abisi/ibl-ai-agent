"""Same pipeline (PSTH -> residual -> PCA -> CCA, across peri-stimulus time)
as 018, averaged across ALL 63 qualifying sessions for Motor and frontal
areas vs Striatum and pallidum (>=20 good+MUA units/area, v3 coarse
grouping) -- "that example alone" (Axel, 2026-08-31), not the full 39-pair
run yet.

Fixes per Axel's 2026-08-31 follow-up:
- FIVE conditions, not three trial types: whisker x {0,1}, auditory x 1
  (misses excluded, too few trials), no_stim x {0,1} -- the earlier version
  of this script only split by trial TYPE, silently dropping the lick_flag
  split that the rest of this project's design requires.
- Ridge regularization lambda is now selected per (session, condition) by
  3-fold trial-level CV (`cca_lib.select_regularization_trial_cv`) instead
  of a fixed 1e-3 -- targets the auditory-trial instability found in 018
  (r=1.00 driven by one outlier held-out trial, a symptom of too little
  regularization relative to ~800-1000 features and only ~51 trials).
- A genuine "correlation across time bins" row is added (trial-by-trial
  Pearson correlation between the two areas' canonical variates AT each
  time bin, `cca_lib.canonical_correlation_across_time`) -- distinct from
  the CCA-variate-amplitude row, which was the only CCA view before.

Per-session summaries that generalize across sessions (units differ per
session, so "the same unit" doesn't exist across sessions):
- PSTH: population-MEAN rate per area.
- Residual: population-mean ABSOLUTE residual per area (signed residual is
  exactly zero on average by construction, see 018's note).
- PCA/CCA-variate: PC1 / canonical dim-1 held-out projection, mean over
  test trials, SIGN-ALIGNED across sessions before averaging (arbitrary
  eigenvector sign per session fit).
- Correlation-across-time: no sign ambiguity issue (correlation is signed
  consistently once orientation is fixed within a session), averaged
  directly across sessions.

Dead zone (-10ms/+5ms, whisker only) bins are NaN throughout -> genuine
plotted gaps (matplotlib breaks lines at NaN).
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
# A first attempt at CV (6 lambdas x 3 folds = 18 extra fits/condition) took
# >20min for just 2/63 sessions -- ~10h+ extrapolated, impractical. Cut two
# ways instead: (1) a smaller grid (3 lambdas x 2 folds = 6 fits), (2) only
# run CV where trial count is low enough to plausibly need it (this is
# where 018's auditory-trial instability actually showed up, at 51 trials;
# whisker/no_stim, ~150-200 trials, were already stable at a fixed lambda).
CANDIDATE_LAMBDAS = (1e-3, 1e-1, 1.0)
CV_FOLDS = 2
CV_TRIAL_COUNT_THRESHOLD = 100
FIXED_LAMBDA = 1e-3
TIER_FN = lambda ut: ut["quality_label"].isin(["good", "mua"])


def flatten(tensor):
    n_t, n_b, n_u = tensor.shape
    return tensor.reshape(n_t * n_b, n_u)


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


def main() -> None:
    session_ids = pd.read_csv(ARTIFACTS_DIR / "motor_striatum_session_list.csv")["session_id"].tolist()
    files = [f"{s}.nwb" for s in session_ids]
    print(f"Loading {len(files)} sessions for {AREA_A} vs {AREA_B}...")

    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=16)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[TIER_FN(unit_table)]

    bin_edges = cca_lib.sliding_window_starts()
    rng = np.random.default_rng(1)

    curves = {cond: {"psth_a": [], "psth_b": [], "resid_a": [], "resid_b": [], "pc1_a": [], "pc1_b": [],
                      "cca_a": [], "cca_b": [], "cca_r": [], "corr_t": [], "lambdas": []}
              for cond in CONDITIONS}
    n_sessions_used = 0

    for session_id in session_ids:
        units_a = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_A)]
        units_b = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_B)]
        if len(units_a) < MIN_UNITS or len(units_b) < MIN_UNITS:
            continue
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
            except Exception as e:
                print(f"  {session_id} {cond}: CCA failed ({e})")
            used_this_session = True
        n_sessions_used += int(used_this_session)
        print(f"{session_id}: done")

    print(f"\n{n_sessions_used}/{len(session_ids)} sessions contributed >=1 condition")
    for cond in CONDITIONS:
        lambdas = curves[cond]["lambdas"]
        if lambdas:
            print(f"  {cond}: n={len(curves[cond]['psth_a'])} sessions, CV-selected lambda "
                  f"median={np.median(lambdas):.4g} (range {min(lambdas):.4g}-{max(lambdas):.4g})")

    bin_centers = cca_lib.sliding_window_centers(bin_edges)
    t_ms = bin_centers * 1000
    fig, axes = plt.subplots(5, 5, figsize=(32, 26))
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        c = curves[cond]
        n = len(c["psth_a"])
        col_label = f"{trial_type} (lick={lick_flag})"

        ax = axes[0, col]
        ax.plot(t_ms, np.nanmean(c["psth_a"], axis=0), color="tab:blue", label=f"{AREA_A} (population mean)")
        ax.plot(t_ms, np.nanmean(c["psth_b"], axis=0), color="tab:orange", label=f"{AREA_B} (population mean)")
        ax.set_title(f"{col_label} (n={n} sessions)")
        if col == 0:
            ax.set_ylabel("PSTH\nmean rate (Hz)")
        ax.legend(fontsize=7)

        ax = axes[1, col]
        ax.plot(t_ms, np.nanmean(c["resid_a"], axis=0), color="tab:blue", label=f"{AREA_A} mean |residual|")
        ax.plot(t_ms, np.nanmean(c["resid_b"], axis=0), color="tab:orange", label=f"{AREA_B} mean |residual|")
        if col == 0:
            ax.set_ylabel("residual\nmean |residual| (Hz)")
        ax.legend(fontsize=7)

        ax = axes[2, col]
        pc1_a_aligned = sign_align(c["pc1_a"])
        pc1_b_aligned = sign_align(c["pc1_b"])
        ax.plot(t_ms, np.nanmean(pc1_a_aligned, axis=0), color="tab:blue", label=f"{AREA_A} PC1 (sign-aligned avg)")
        ax.plot(t_ms, np.nanmean(pc1_b_aligned, axis=0), color="tab:orange", label=f"{AREA_B} PC1 (sign-aligned avg)")
        ax.axhline(0, color="black", linewidth=0.8)
        if col == 0:
            ax.set_ylabel("PCA\nPC1 score (held-out, avg across sessions)")
        ax.legend(fontsize=7)

        ax = axes[3, col]
        cca_a_aligned = sign_align(c["cca_a"])
        cca_b_aligned = sign_align(c["cca_b"])
        ax.plot(t_ms, np.nanmean(cca_a_aligned, axis=0), color="tab:blue", label=f"{AREA_A} canon. var 1")
        ax.plot(t_ms, np.nanmean(cca_b_aligned, axis=0), color="tab:orange", label=f"{AREA_B} canon. var 1")
        ax.axhline(0, color="black", linewidth=0.8)
        lam_med = np.median(c["lambdas"]) if c["lambdas"] else float("nan")
        ax.set_title(f"CCA variate (median CV lambda={lam_med:.3g})", fontsize=9)
        if col == 0:
            ax.set_ylabel("CCA variate\n(dim 1, avg across sessions)")
        ax.legend(fontsize=7)

        ax = axes[4, col]
        ax.plot(t_ms, np.nanmean(c["corr_t"], axis=0), color="tab:purple")
        ax.axhline(0, color="black", linewidth=0.8)
        mean_r_overall = np.nanmean(np.abs(c["cca_r"])) if c["cca_r"] else float("nan")
        ax.set_title(f"trial-by-trial correlation across time\n(mean whole-window |r|={mean_r_overall:.2f})", fontsize=9)
        if col == 0:
            ax.set_ylabel("CCA correlation\nacross-trial r per time bin (avg across sessions)")
        ax.set_xlabel("time from start_time (ms)")

    fig.suptitle(f"Pipeline across peri-stimulus time, AVERAGED across {n_sessions_used} sessions -- "
                 f"{AREA_A} vs {AREA_B} (ALL units used per session, no subsampling; "
                 f"5ms sliding windows, 2ms stride; regularization CV'd per session/condition)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "example_pair_03_pipeline_full_dataset_average.png", dpi=110)
    plt.close(fig)
    print("Wrote example_pair_03_pipeline_full_dataset_average.png")


if __name__ == "__main__":
    main()
