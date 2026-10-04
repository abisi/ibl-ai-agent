"""Compiles the FULL figure set (01-09, 11 from the menu given to Axel on
2026-09-01) for one area pair, at smoke-test scale, into a dedicated run
subfolder for review before scaling up ("add 7,8,9,11 ... compile all
figures in a run subfolder so I can check before doing any more").

Reuses 022's core fit/PCA-reduce/shuffle logic via import (not
duplicated). New in this script:
- 06: **standing figure design per Axel's 2026-09-01 follow-up** ("always
  have on the same figure the quantities for variants A and B, PCA
  baseline and shuffle distribution -- have a shuffle for A and a shuffle
  for B") -- ONE figure per pair showing, per condition per cohort: CCA
  variant A (plain), A's own shuffle-null, the PCA-alignment baseline, CCA
  variant B (partial out other units), and B's own shuffle-null. Variant B
  did not previously have a shuffle-null at all (only variant A did, in
  022) -- now computed identically to A's (same N_SHUFFLES, full refit per
  shuffle, just with Z passed to `PartialCCA.fit()`). This replaces what
  were two separate figures in the first version of this script (a
  CCA/PCA-baseline/shuffle-A-only figure, and a separate A-vs-B-true-only
  figure) -- apply this merged design to every future pair's figure set.
- 07: lagged cross-area correlation -- area B's residual tensor is
  shifted along the bin axis by a range of lags (+/-100ms at bin-stride
  resolution), the SAME PCA-reduce+fit+transform cycle is rerun at each
  lag (one example session, `LAG_CONDITION`), plus a peak-lag-per-session
  histogram across the smoke session set (same condition). `LAG_CONDITION`
  must be a NON-whisker condition -- shifting (`np.roll`) a tensor that
  contains the whisker dead-zone's NaN gap circularly moves that gap into
  bins the (unshifted) valid-bins mask treats as valid, crashing PCA on
  NaN input (hit and fixed during this run).
- 08: example correlated trials -- held-out TEST trials from the example
  session/condition ranked by their own ca-vs-cb time-course correlation;
  top few plotted overlaid.
- 11: unit-count confound check -- one example session/condition,
  subsampling area A/B unit counts to several levels (repeated), dim-1
  correlation vs n_units.

01/02 (raster, single-session pipeline-across-time) and 03 (pooled
dataset average, no cohort split) are also (re)built here so the whole
menu lands together in one folder.
"""
from __future__ import annotations

import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
cov_lib = importlib.import_module("000_coverage_lib")
cca_lib = importlib.import_module("003_cca_lib")
core = importlib.import_module("022_pipeline_pca_reduced_shuffle_baseline")

import numpy as np
import pandas as pd
from partial_CCA import PartialCCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RUN_DIR = ARTIFACTS_DIR / "runs" / "smoke_full_figure_set"
RUN_DIR.mkdir(parents=True, exist_ok=True)

AREA_A, AREA_B = core.AREA_A, core.AREA_B
CONDITIONS = core.CONDITIONS
FIXED_LAMBDA_BY_CONDITION = core.FIXED_LAMBDA_BY_CONDITION
MAX_DIMS = core.MAX_DIMS
COHORT_COLOR = core.COHORT_COLOR
MIN_UNITS = core.MIN_UNITS
MIN_TRIALS = core.MIN_TRIALS
TIER_FN = core.TIER_FN
N_SHUFFLES = core.N_SHUFFLES
pca_reduce = core.pca_reduce
fit_and_eval = core.fit_and_eval
spikes_of = core.spikes_of
to_full_timeline = core.to_full_timeline
sign_align = core.sign_align
sem_across = core.sem_across

N_SMOKE_SESSIONS = 8
LAG_CONDITION = ("whisker_trial", 0)  # dense/stable -- used for figs 07/08/11.
# Back to a whisker condition (2026-09-01, Axel: "generalize the mask logic"): `lag_sweep`/
# `shift_tensor_nan` now rebuild the valid-bins mask PER LAG from where the (non-circular)
# shifted tensor is actually NaN, so any lag/bin combination that would pull dead-zone-
# contaminated spikes into the comparison is excluded, for either area, at any lag -- no longer
# needs to avoid whisker conditions (that was a workaround for the earlier np.roll-based bug).
LAG_MS_RANGE = 50  # Axel, 2026-09-02: "next time, only check -50 and +50ms" (was 100) -- halves
                   # the lag-sweep step count (201 -> 101 at 1ms resolution) for future runs
UNIT_LEVELS_BASE = [20, 40, 80]
UNIT_REPEATS = 3
N_CORRELATED_TRIALS_TO_SHOW = 4


def fit_and_eval_z(resid_a, resid_b, resid_z, valid_bins, train_idx, test_idx, lam, n_trials,
                    max_dims: int = MAX_DIMS):
    """Like core.fit_and_eval but with an optional nuisance Z (variant B
    when given, variant A when None) -- Z is PCA-reduced the same way as
    X/Y (same K rule)."""
    Xa_train, _ = cca_lib.flatten_trial_bins(resid_a[train_idx][:, valid_bins, :])
    Xb_train, _ = cca_lib.flatten_trial_bins(resid_b[train_idx][:, valid_bins, :])
    Xa_test = resid_a[test_idx][:, valid_bins, :].reshape(-1, resid_a.shape[-1])
    Xb_test = resid_b[test_idx][:, valid_bins, :].reshape(-1, resid_b.shape[-1])
    n_test, n_bins_v = len(test_idx), int(valid_bins.sum())

    Xa_train_r, Xa_test_r, _ = pca_reduce(Xa_train, Xa_test, n_trials)
    Xb_train_r, Xb_test_r, _ = pca_reduce(Xb_train, Xb_test, n_trials)

    Z_train_r = None
    if resid_z is not None and resid_z.shape[-1] > 0:
        Z_train, _ = cca_lib.flatten_trial_bins(resid_z[train_idx][:, valid_bins, :])
        Z_test = resid_z[test_idx][:, valid_bins, :].reshape(-1, resid_z.shape[-1])
        Z_train_r, _, _ = pca_reduce(Z_train, Z_test, n_trials)

    model = PartialCCA(regularization=lam).fit(Xa_train_r, Xb_train_r, Z_train_r, verbose=False)
    n_keep = min(model.weights_x_.shape[1], max_dims)
    model.weights_x_ = model.weights_x_[:, :n_keep]
    model.weights_y_ = model.weights_y_[:, :n_keep]
    ca, cb = model.transform(Xa_test_r, Xb_test_r)
    ca1 = ca[0].reshape(n_test, n_bins_v)
    cb1 = cb[0].reshape(n_test, n_bins_v)
    ca1_full = to_full_timeline(np.nanmean(ca1, axis=0), valid_bins)
    cb1_full = to_full_timeline(np.nanmean(cb1, axis=0), valid_bins)
    m = ~(np.isnan(ca1_full) | np.isnan(cb1_full))
    return np.corrcoef(ca1_full[m], cb1_full[m])[0, 1] if m.sum() > 3 else np.nan


# shift_tensor_nan/lag_sweep moved to core (022_pipeline_pca_reduced_shuffle_baseline.py,
# 2026-09-02) so 024's process_session_full can run the lag sweep in the SAME per-session
# pass as the true fit, reusing already-computed residuals instead of a separate script
# recomputing tensors from scratch. Kept as aliases here so this script's own callers are
# unaffected.
shift_tensor_nan = core.shift_tensor_nan
lag_sweep = core.lag_sweep


def rank_correlated_trials(ca_full_2d, cb_full_2d):
    n_test = ca_full_2d.shape[0]
    per_trial_r = np.full(n_test, np.nan)
    for i in range(n_test):
        m = ~(np.isnan(ca_full_2d[i]) | np.isnan(cb_full_2d[i]))
        if m.sum() > 5:
            per_trial_r[i] = np.corrcoef(ca_full_2d[i, m], cb_full_2d[i, m])[0, 1]
    order = np.argsort(-np.nan_to_num(per_trial_r, nan=-2))
    return order, per_trial_r


def unit_count_confound(spikes_a, spikes_b, starts, is_whisker, bin_starts, valid_bins, lam, n_trials,
                         rng, levels, n_repeats):
    results: dict[int, list[float]] = {}
    for n in levels:
        if n > min(len(spikes_a), len(spikes_b)):
            continue
        rs = []
        for _ in range(n_repeats):
            idx_a = rng.choice(len(spikes_a), size=n, replace=False)
            idx_b = rng.choice(len(spikes_b), size=n, replace=False)
            sub_a = [spikes_a[i] for i in idx_a]
            sub_b = [spikes_b[i] for i in idx_b]
            tensor_a = cca_lib.population_tensor_sliding_smoothed(sub_a, starts, is_whisker, bin_starts)
            tensor_b = cca_lib.population_tensor_sliding_smoothed(sub_b, starts, is_whisker, bin_starts)
            resid_a = cca_lib.noise_correlation_residuals(tensor_a)
            resid_b = cca_lib.noise_correlation_residuals(tensor_b)
            perm = rng.permutation(resid_a.shape[0])
            half = len(perm) // 2
            train_idx, test_idx = perm[:half], perm[half:]
            try:
                r, *_ = fit_and_eval(resid_a, resid_b, valid_bins, train_idx, test_idx, lam, n_trials)
                rs.append(r)
            except Exception:
                continue
        if rs:
            results[n] = rs
    return results


def main() -> None:
    session_ids = pd.read_csv(ARTIFACTS_DIR / "motor_striatum_session_list.csv")["session_id"].tolist()
    session_ids = session_ids[:N_SMOKE_SESSIONS]
    files = [f"{s}.nwb" for s in session_ids]
    print(f"[full figure set] Loading {len(files)} sessions for {AREA_A} vs {AREA_B}...")

    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=16)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[TIER_FN(unit_table)]

    bin_starts = cca_lib.sliding_window_starts()
    smooth_kernel = cca_lib.causal_half_gaussian_kernel()
    valid_bins_by_cond = {
        cond: ~cca_lib.dead_zone_bin_mask_sliding_causal(bin_starts, cca_lib.BIN_WIDTH,
                                                          cond[0] == "whisker_trial", smooth_kernel)
        for cond in CONDITIONS
    }
    bin_centers = cca_lib.sliding_window_centers(bin_starts)
    t_ms = bin_centers * 1000

    sessions = []  # list of dicts with everything needed per session
    for session_id in session_ids:
        units_a = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_A)]
        units_b = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_B)]
        if len(units_a) < MIN_UNITS or len(units_b) < MIN_UNITS:
            continue
        units_other = unit_table[(unit_table["session_id"] == session_id)
                                  & (~unit_table.index.isin(units_a.index))
                                  & (~unit_table.index.isin(units_b.index))]
        trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
        sessions.append({
            "session_id": session_id, "reward_group": units_a["reward_group"].iloc[0],
            "mouse_id": units_a["mouse_id"].iloc[0],
            "n_units_a": len(units_a), "n_units_b": len(units_b), "n_units_other": len(units_other),
            "spikes_a": spikes_of(units_a), "spikes_b": spikes_of(units_b), "spikes_other": spikes_of(units_other),
            "trials_sess": trials_sess,
        })
    print(f"[full figure set] {len(sessions)} sessions usable")
    example = sessions[0]
    print(f"[full figure set] example session: {example['session_id']} "
          f"(A={example['n_units_a']}, B={example['n_units_b']}, other={example['n_units_other']})")

    # ---- core per-session/condition pass (variant A) -> figs 03/04/05/06 ----
    curves = {cond: {"session_id": [], "reward_group": [], "mouse_id": [], "n_units_a": [], "n_units_b": [],
                      "n_trials": [], "psth_a": [], "psth_b": [], "resid_a": [], "resid_b": [],
                      "pc1_a": [], "pc1_b": [], "pca_baseline_r": [], "cca_a": [], "cca_b": [], "cca_r": [],
                      "corr_t": [], "lambdas": [], "cc_dims": [], "shuffle_null": [], "k_a": [], "k_b": []}
              for cond in CONDITIONS}
    for s in sessions:
        task = (s["session_id"], s["reward_group"], s["mouse_id"], s["n_units_a"], s["n_units_b"],
                s["spikes_a"], s["spikes_b"], s["trials_sess"], bin_starts, valid_bins_by_cond)
        sid, reward_group, mouse_id, n_units_a, n_units_b, results = core.process_one_session(task)
        for cond, row in results.items():
            c = curves[cond]
            c["session_id"].append(sid); c["reward_group"].append(reward_group); c["mouse_id"].append(mouse_id)
            c["n_units_a"].append(n_units_a); c["n_units_b"].append(n_units_b)
            c["n_trials"].append(row.get("n_trials", np.nan))
            for key in ("psth_a", "psth_b", "resid_a", "resid_b", "pc1_a", "pc1_b", "pca_baseline_r",
                        "cca_a", "cca_b", "cca_r", "corr_t", "cc_dims", "shuffle_null", "k_a", "k_b"):
                c[key].append(row.get(key))
            c["lambdas"].append(row.get("lambda", np.nan))
        print(f"  [core pass] {sid} ({reward_group}): done")

    # ==== Fig 01: population raster, example session, all 5 conditions ====
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(6 * len(CONDITIONS), 8), sharex=True)
    N_TRIALS_SHOWN = 20
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        is_whisker = trial_type == "whisker_trial"
        tt_df = example["trials_sess"][(example["trials_sess"]["trial_type"] == trial_type)
                                        & (example["trials_sess"]["lick_flag"] == lick_flag)].sort_values("start_time")
        starts = tt_df["start_time"].to_numpy(dtype=float)[:N_TRIALS_SHOWN]
        for row_i, (area_label, spikes) in enumerate([(AREA_A, example["spikes_a"]), (AREA_B, example["spikes_b"])]):
            ax = axes[row_i, col]
            for ti, t0 in enumerate(starts):
                all_sp = np.concatenate([st[(st > t0 - 0.2) & (st < t0 + 0.5)] - t0 for st in spikes]) if spikes else np.array([])
                ax.scatter(all_sp * 1000, np.full(len(all_sp), ti), s=1, color="black", alpha=0.4)
            if is_whisker:
                ax.axvspan(cca_lib.DEAD_ZONE_START_S * 1000, cca_lib.DEAD_ZONE_STOP_S * 1000, color="red", alpha=0.15)
            ax.axvline(0, color="tab:blue", linewidth=0.8)
            if row_i == 0:
                ax.set_title(f"{trial_type} (lick={lick_flag})")
            if col == 0:
                ax.set_ylabel(f"{area_label}\ntrial #")
    axes[-1, 0].set_xlabel("time from start_time (ms)")
    fig.suptitle(f"Population raster (pooled units), example session {example['session_id']} ({example['reward_group']})")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "01_raster.png", dpi=110)
    plt.close(fig)
    print("Wrote 01_raster.png")

    # ==== Fig 02: pipeline-across-time, example session only ====
    example_task = (example["session_id"], example["reward_group"], example["mouse_id"],
                     example["n_units_a"], example["n_units_b"], example["spikes_a"], example["spikes_b"],
                     example["trials_sess"], bin_starts, valid_bins_by_cond)
    _, _, _, _, _, example_results = core.process_one_session(example_task)
    fig, axes = plt.subplots(4, len(CONDITIONS), figsize=(6 * len(CONDITIONS), 14))
    for col, cond in enumerate(CONDITIONS):
        if cond not in example_results:
            continue
        row = example_results[cond]
        axes[0, col].plot(t_ms, row["psth_a"], color="tab:blue", label=AREA_A)
        axes[0, col].plot(t_ms, row["psth_b"], color="tab:orange", label=AREA_B)
        axes[0, col].set_title(f"{cond[0]} (lick={cond[1]})")
        axes[1, col].plot(t_ms, row["resid_a"], color="tab:blue")
        axes[1, col].plot(t_ms, row["resid_b"], color="tab:orange")
        axes[2, col].plot(t_ms, row["pc1_a"], color="tab:blue")
        axes[2, col].plot(t_ms, row["pc1_b"], color="tab:orange")
        axes[2, col].axhline(0, color="black", linewidth=0.6)
        if "corr_t" in row:
            axes[3, col].plot(t_ms, row["corr_t"], color="tab:purple")
            axes[3, col].axhline(0, color="black", linewidth=0.6)
    axes[0, 0].legend(fontsize=8)
    for r_i, lbl in enumerate(["PSTH (Hz)", "mean |residual| (Hz)", "PC1 (held-out)", "CCA r across time"]):
        axes[r_i, 0].set_ylabel(lbl)
    fig.suptitle(f"Pipeline across time, example session {example['session_id']} ({example['reward_group']})")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "02_pipeline_across_time.png", dpi=110)
    plt.close(fig)
    print("Wrote 02_pipeline_across_time.png")

    # ==== Fig 03: pooled dataset average (no cohort split) ====
    fig, axes = plt.subplots(5, len(CONDITIONS), figsize=(6 * len(CONDITIONS), 22))
    for col, cond in enumerate(CONDITIONS):
        c = curves[cond]
        n = len(c["psth_a"])
        if n == 0:
            continue
        axes[0, col].plot(t_ms, np.nanmean(c["psth_a"], axis=0), color="tab:blue", label=AREA_A)
        axes[0, col].plot(t_ms, np.nanmean(c["psth_b"], axis=0), color="tab:orange", label=AREA_B)
        axes[0, col].set_title(f"{cond[0]} (lick={cond[1]}, n={n})")
        axes[1, col].plot(t_ms, np.nanmean(c["resid_a"], axis=0), color="tab:blue")
        axes[1, col].plot(t_ms, np.nanmean(c["resid_b"], axis=0), color="tab:orange")
        pc1_a_al = sign_align(c["pc1_a"]); pc1_b_al = sign_align(c["pc1_b"])
        axes[2, col].plot(t_ms, np.nanmean(pc1_a_al, axis=0), color="tab:blue")
        axes[2, col].plot(t_ms, np.nanmean(pc1_b_al, axis=0), color="tab:orange")
        axes[2, col].axhline(0, color="black", linewidth=0.6)
        cca_a_al = sign_align(c["cca_a"]); cca_b_al = sign_align(c["cca_b"])
        axes[3, col].plot(t_ms, np.nanmean(cca_a_al, axis=0), color="tab:blue")
        axes[3, col].plot(t_ms, np.nanmean(cca_b_al, axis=0), color="tab:orange")
        axes[3, col].axhline(0, color="black", linewidth=0.6)
        axes[4, col].plot(t_ms, np.nanmean(c["corr_t"], axis=0), color="tab:purple")
        axes[4, col].axhline(0, color="black", linewidth=0.6)
    axes[0, 0].legend(fontsize=8)
    for r_i, lbl in enumerate(["PSTH (Hz)", "mean |residual| (Hz)", "PC1", "CCA variate (dim1)", "CCA r across time"]):
        axes[r_i, 0].set_ylabel(lbl)
    fig.suptitle(f"Pooled dataset average (R+/R- pooled), {AREA_A} vs {AREA_B}, n={len(sessions)} smoke sessions")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "03_pooled_average.png", dpi=110)
    plt.close(fig)
    print("Wrote 03_pooled_average.png")

    # ==== Fig 04: R+/R- comparison (same layout as 022's) ====
    fig, axes = plt.subplots(5, len(CONDITIONS), figsize=(6 * len(CONDITIONS), 22))
    for col, cond in enumerate(CONDITIONS):
        c = curves[cond]
        rg = np.array(c["reward_group"])
        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            if len(idx) == 0:
                continue
            color = COHORT_COLOR[cohort]
            m_a, s_a = sem_across([c["psth_a"][i] for i in idx]); m_b, s_b = sem_across([c["psth_b"][i] for i in idx])
            axes[0, col].plot(t_ms, m_a, color=color, ls="-", label=f"{cohort} A (n={len(idx)})")
            axes[0, col].fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            axes[0, col].plot(t_ms, m_b, color=color, ls="--", label=f"{cohort} B")
            axes[0, col].fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)

            m_a, s_a = sem_across([c["resid_a"][i] for i in idx]); m_b, s_b = sem_across([c["resid_b"][i] for i in idx])
            axes[1, col].plot(t_ms, m_a, color=color, ls="-"); axes[1, col].fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            axes[1, col].plot(t_ms, m_b, color=color, ls="--"); axes[1, col].fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)

            pc1a = sign_align([c["pc1_a"][i] for i in idx]); pc1b = sign_align([c["pc1_b"][i] for i in idx])
            m_a, s_a = sem_across(pc1a); m_b, s_b = sem_across(pc1b)
            axes[2, col].plot(t_ms, m_a, color=color, ls="-"); axes[2, col].fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            axes[2, col].plot(t_ms, m_b, color=color, ls="--"); axes[2, col].fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)
            axes[2, col].axhline(0, color="black", linewidth=0.6)

            ccaa = sign_align([c["cca_a"][i] for i in idx]); ccab = sign_align([c["cca_b"][i] for i in idx])
            m_a, s_a = sem_across(ccaa); m_b, s_b = sem_across(ccab)
            axes[3, col].plot(t_ms, m_a, color=color, ls="-"); axes[3, col].fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            axes[3, col].plot(t_ms, m_b, color=color, ls="--"); axes[3, col].fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)
            axes[3, col].axhline(0, color="black", linewidth=0.6)

            m_t, s_t = sem_across([c["corr_t"][i] for i in idx])
            axes[4, col].plot(t_ms, m_t, color=color, label=f"{cohort} (n={len(idx)})")
            axes[4, col].fill_between(t_ms, m_t - s_t, m_t + s_t, color=color, alpha=0.2)
            axes[4, col].axhline(0, color="black", linewidth=0.6)
        axes[0, col].set_title(f"{cond[0]} (lick={cond[1]})")
    axes[0, 0].legend(fontsize=6); axes[4, 0].legend(fontsize=7)
    for r_i, lbl in enumerate(["PSTH (Hz)", "mean |residual| (Hz)", "PC1", "CCA variate (dim1)", "CCA r across time"]):
        axes[r_i, 0].set_ylabel(lbl)
    fig.suptitle(f"R+ vs R- comparison, {AREA_A} vs {AREA_B} (smoke, n={len(sessions)} sessions)")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "04_rplus_rminus_comparison.png", dpi=110)
    plt.close(fig)
    print("Wrote 04_rplus_rminus_comparison.png")

    # ==== Fig 05: correlation vs dimension ====
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(6 * len(CONDITIONS), 5), sharey=True)
    dims = np.arange(1, MAX_DIMS + 1)
    for col, cond in enumerate(CONDITIONS):
        c = curves[cond]; rg = np.array(c["reward_group"]); ax = axes[col]
        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            if len(idx) == 0:
                continue
            arr = np.abs(np.vstack([c["cc_dims"][i] for i in idx]))
            mean = np.nanmean(arr, axis=0); n = np.sum(~np.isnan(arr), axis=0)
            sem = np.divide(np.nanstd(arr, axis=0, ddof=1), np.sqrt(n), out=np.full_like(mean, np.nan), where=n > 1)
            ax.errorbar(dims, mean, yerr=sem, color=COHORT_COLOR[cohort], marker="o", markersize=4,
                        label=f"{cohort} (n={len(idx)})")
        ax.set_title(f"{cond[0]} (lick={cond[1]})"); ax.set_xticks(dims); ax.legend(fontsize=7)
        ax.axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("held-out |canonical correlation|")
    fig.suptitle("Correlation vs. canonical dimension (smoke)")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "05_correlation_vs_dimension.png", dpi=110)
    plt.close(fig)
    print("Wrote 05_correlation_vs_dimension.png")

    # ==== Fig 06: variant A vs B, each with its own shuffle-null, plus PCA baseline ====
    # (deferred until variant B + its shuffle-null are computed below, near
    # the old fig-09 spot -- see "Fig 06 (built here)" further down. Per
    # Axel's 2026-09-01 instruction: always show A, B, PCA-baseline, and a
    # SEPARATE shuffle-null for each of A/B on one figure, not split
    # across two.)

    # ==== Fig 07: lag sweep + peak-lag distribution (LAG_CONDITION only) ====
    # SUPERSEDED (2026-09-02): 024_updated_figures.py now computes this same figure inside
    # process_session_full, reusing residuals already built for the main pass instead of
    # this script's separate recompute -- same supersession pattern as 01/02/03/05/08.
    # Kept here (uses all N_SMOKE_SESSIONS, unlike 024's full session list) only as a
    # standalone smoke-test path; prefer 024's 07_lag_sweep.png for real results.
    trial_type, lick_flag = LAG_CONDITION
    is_whisker = trial_type == "whisker_trial"
    valid_bins = valid_bins_by_cond[LAG_CONDITION]
    lam = FIXED_LAMBDA_BY_CONDITION[LAG_CONDITION]
    bin_stride_ms = cca_lib.BIN_STRIDE * 1000
    max_lag_bins = int(round(LAG_MS_RANGE / bin_stride_ms))
    lag_bins_range = np.arange(-max_lag_bins, max_lag_bins + 1)
    lag_ms_axis = lag_bins_range * bin_stride_ms

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    peak_lags = []
    for s in sessions:
        tt_df = s["trials_sess"][(s["trials_sess"]["trial_type"] == trial_type)
                                  & (s["trials_sess"]["lick_flag"] == lick_flag)].sort_values("start_time")
        starts = tt_df["start_time"].to_numpy(dtype=float)
        if len(starts) < MIN_TRIALS:
            continue
        tensor_a = cca_lib.population_tensor_sliding_smoothed(s["spikes_a"], starts, is_whisker, bin_starts)
        tensor_b = cca_lib.population_tensor_sliding_smoothed(s["spikes_b"], starts, is_whisker, bin_starts)
        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)
        seed = zlib.crc32(f"{s['session_id']}_lag".encode()) % (2**32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(resid_a.shape[0])
        train_idx, test_idx = perm[:len(perm)//2], perm[len(perm)//2:]
        rs = lag_sweep(resid_a, resid_b, valid_bins, train_idx, test_idx, lam, len(starts), lag_bins_range)
        color = COHORT_COLOR[s["reward_group"]]
        axes[0].plot(lag_ms_axis, rs, color=color, alpha=0.5 if s is not example else 1.0,
                     linewidth=2.5 if s is example else 1.0,
                     label=(f"example ({s['session_id']})" if s is example else None))
        if np.any(~np.isnan(rs)):
            peak_lags.append(lag_ms_axis[np.nanargmax(np.abs(rs))])
    axes[0].axvline(0, color="black", linewidth=0.6)
    axes[0].set_xlabel("lag (ms): positive = area B activity shifted earlier relative to A")
    axes[0].set_ylabel("held-out dim-1 correlation")
    axes[0].set_title(f"Lag sweep, {trial_type} (lick={lick_flag}), all smoke sessions")
    axes[0].legend(fontsize=7)
    axes[1].hist(peak_lags, bins=15, color="tab:gray")
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set_xlabel("peak-|r| lag (ms)")
    axes[1].set_ylabel("# sessions")
    axes[1].set_title(f"Peak-lag distribution (n={len(peak_lags)} sessions)")
    fig.suptitle(f"Lagged cross-area correlation, {AREA_A} vs {AREA_B}")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "07_lag_sweep.png", dpi=110)
    plt.close(fig)
    print("Wrote 07_lag_sweep.png")

    # ==== Fig 08: example correlated trials (example session, LAG_CONDITION) ====
    tt_df = example["trials_sess"][(example["trials_sess"]["trial_type"] == trial_type)
                                    & (example["trials_sess"]["lick_flag"] == lick_flag)].sort_values("start_time")
    starts = tt_df["start_time"].to_numpy(dtype=float)
    tensor_a = cca_lib.population_tensor_sliding_smoothed(example["spikes_a"], starts, is_whisker, bin_starts)
    tensor_b = cca_lib.population_tensor_sliding_smoothed(example["spikes_b"], starts, is_whisker, bin_starts)
    resid_a = cca_lib.noise_correlation_residuals(tensor_a)
    resid_b = cca_lib.noise_correlation_residuals(tensor_b)
    seed = zlib.crc32(f"{example['session_id']}_{trial_type}_{lick_flag}".encode()) % (2**32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(resid_a.shape[0])
    train_idx, test_idx = perm[:len(perm)//2], perm[len(perm)//2:]
    r_dim1, ca_full_2d, cb_full_2d, *_ = fit_and_eval(resid_a, resid_b, valid_bins, train_idx, test_idx, lam, len(starts))
    order, per_trial_r = rank_correlated_trials(ca_full_2d, cb_full_2d)
    top = order[:N_CORRELATED_TRIALS_TO_SHOW]
    fig, axes = plt.subplots(1, len(top), figsize=(5 * len(top), 4), sharey=True)
    if len(top) == 1:
        axes = [axes]
    for k, ti in enumerate(top):
        axes[k].plot(t_ms, ca_full_2d[ti], color="tab:blue", label=AREA_A)
        axes[k].plot(t_ms, cb_full_2d[ti], color="tab:orange", label=AREA_B)
        axes[k].axhline(0, color="black", linewidth=0.6)
        axes[k].set_title(f"test trial #{ti}\nr={per_trial_r[ti]:.2f}")
        axes[k].set_xlabel("time from start_time (ms)")
    axes[0].legend(fontsize=8); axes[0].set_ylabel("canonical variate (dim 1)")
    fig.suptitle(f"Most-correlated held-out trials, example session {example['session_id']}, "
                 f"{trial_type} (lick={lick_flag}), dim1 mean r={r_dim1:.2f}")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "08_example_correlated_trials.png", dpi=110)
    plt.close(fig)
    print("Wrote 08_example_correlated_trials.png")

    # ==== Variant B (partial-out-other-units) + its OWN shuffle-null ====
    # Mirrors variant A's true-fit + N_SHUFFLES-shuffle-null exactly, just
    # with Z (PCA-reduced "other units") passed to PartialCCA.fit().
    varB_curves = {cond: {"reward_group": [], "r": [], "shuffle_null": []} for cond in CONDITIONS}
    for s in sessions:
        for cond in CONDITIONS:
            tt_df = s["trials_sess"][(s["trials_sess"]["trial_type"] == cond[0])
                                      & (s["trials_sess"]["lick_flag"] == cond[1])].sort_values("start_time")
            starts = tt_df["start_time"].to_numpy(dtype=float)
            if len(starts) < MIN_TRIALS or len(s["spikes_other"]) < 5:
                continue
            is_wh = cond[0] == "whisker_trial"
            vb = valid_bins_by_cond[cond]
            tensor_a = cca_lib.population_tensor_sliding_smoothed(s["spikes_a"], starts, is_wh, bin_starts)
            tensor_b = cca_lib.population_tensor_sliding_smoothed(s["spikes_b"], starts, is_wh, bin_starts)
            tensor_z = cca_lib.population_tensor_sliding_smoothed(s["spikes_other"], starts, is_wh, bin_starts)
            resid_a = cca_lib.noise_correlation_residuals(tensor_a)
            resid_b = cca_lib.noise_correlation_residuals(tensor_b)
            resid_z = cca_lib.noise_correlation_residuals(tensor_z)
            lam = FIXED_LAMBDA_BY_CONDITION[cond]
            seed = zlib.crc32(f"{s['session_id']}_{cond[0]}_{cond[1]}_varB".encode()) % (2**32)
            rng = np.random.default_rng(seed)
            perm = rng.permutation(resid_a.shape[0])
            train_idx, test_idx = perm[:len(perm)//2], perm[len(perm)//2:]
            try:
                r_b = fit_and_eval_z(resid_a, resid_b, resid_z, vb, train_idx, test_idx, lam, len(starts))
                varB_curves[cond]["reward_group"].append(s["reward_group"])
                varB_curves[cond]["r"].append(r_b)

                null_b = np.full(N_SHUFFLES, np.nan)
                for sh in range(N_SHUFFLES):
                    shuf_seed = zlib.crc32(f"{s['session_id']}_{cond[0]}_{cond[1]}_varB_shuf{sh}".encode()) % (2**32)
                    shuf_rng = np.random.default_rng(shuf_seed)
                    shuf_perm = shuf_rng.permutation(resid_b.shape[0])
                    try:
                        null_b[sh] = fit_and_eval_z(resid_a, resid_b[shuf_perm], resid_z, vb,
                                                     train_idx, test_idx, lam, len(starts))
                    except Exception:
                        continue
                varB_curves[cond]["shuffle_null"].append(null_b)
            except Exception as e:
                print(f"  varB {s['session_id']} {cond}: failed ({e})")
        print(f"  [variant B + shuffle-null] {s['session_id']}: done")

    # ==== Fig 06 (built here): A, A-shuffle, PCA baseline, B, B-shuffle, all together ====
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(6.5 * len(CONDITIONS), 5), sharey=True)
    x_labels = ["CCA A", "shuffle\n(A)", "PCA\nbaseline", "CCA B", "shuffle\n(B)"]
    for col, cond in enumerate(CONDITIONS):
        c_a = curves[cond]; rg_a = np.array(c_a["reward_group"])
        c_b = varB_curves[cond]; rg_b = np.array(c_b["reward_group"])
        ax = axes[col]
        offsets = {"R+": -0.12, "R-": 0.12}
        for cohort in ("R+", "R-"):
            idx_a = np.where(rg_a == cohort)[0]
            idx_b = np.where(rg_b == cohort)[0]
            if len(idx_a) == 0 and len(idx_b) == 0:
                continue
            true_a = np.abs(np.array([c_a["cca_r"][i] for i in idx_a], dtype=float)) if len(idx_a) else np.array([np.nan])
            null_a = (np.abs(np.concatenate([np.asarray(c_a["shuffle_null"][i]) for i in idx_a]))
                      if len(idx_a) else np.array([np.nan]))
            pca_v = np.abs(np.array([c_a["pca_baseline_r"][i] for i in idx_a], dtype=float)) if len(idx_a) else np.array([np.nan])
            true_b = np.abs(np.array([c_b["r"][i] for i in idx_b], dtype=float)) if len(idx_b) else np.array([np.nan])
            null_b = (np.abs(np.concatenate([np.asarray(c_b["shuffle_null"][i]) for i in idx_b]))
                      if len(idx_b) else np.array([np.nan]))
            groups = [true_a, null_a, pca_v, true_b, null_b]
            means = [np.nanmean(v) for v in groups]
            sems = [np.nanstd(v, ddof=1) / np.sqrt(max(1, np.sum(~np.isnan(v)))) for v in groups]
            x = np.arange(5) + offsets[cohort]
            ax.errorbar(x, means, yerr=sems, fmt="o", color=COHORT_COLOR[cohort], markersize=6, capsize=3,
                        label=f"{cohort} (nA={len(idx_a)}, nB={len(idx_b)})")
        ax.set_xticks(range(5)); ax.set_xticklabels(x_labels)
        ax.set_title(f"{cond[0]} (lick={cond[1]})"); ax.legend(fontsize=7); ax.axhline(0, color="black", linewidth=0.6)
    axes[0].set_ylabel("mean |correlation| (dim1)\n+/- SEM across sessions (shuffle: sessions x shuffles)")
    fig.suptitle(f"Variant A vs. B, each with its own {N_SHUFFLES}-shuffle null, plus PCA baseline "
                 f"-- {AREA_A} vs {AREA_B} (smoke)")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "06_variants_and_baselines.png", dpi=110)
    plt.close(fig)
    print("Wrote 06_variants_and_baselines.png")

    # ==== Fig 11: unit-count confound (example session, LAG_CONDITION) ====
    rng = np.random.default_rng(7)
    levels = sorted(set(UNIT_LEVELS_BASE + [min(example["n_units_a"], example["n_units_b"])]))
    confound = unit_count_confound(example["spikes_a"], example["spikes_b"], starts, is_whisker, bin_starts,
                                    valid_bins, lam, len(starts), rng, levels, UNIT_REPEATS)
    fig, ax = plt.subplots(figsize=(6, 5))
    ns = sorted(confound.keys())
    means = [np.nanmean(np.abs(confound[n])) for n in ns]
    sems = [np.nanstd(np.abs(confound[n]), ddof=1) / np.sqrt(max(1, len(confound[n]))) for n in ns]
    ax.errorbar(ns, means, yerr=sems, marker="o", color="tab:blue", capsize=3)
    ax.set_xlabel("n units subsampled per area"); ax.set_ylabel("mean |correlation| (dim1)")
    ax.set_title(f"Unit-count confound, example session {example['session_id']}, "
                 f"{trial_type} (lick={lick_flag}), {UNIT_REPEATS} repeats/level")
    fig.tight_layout()
    fig.savefig(RUN_DIR / "11_unit_count_confound.png", dpi=110)
    plt.close(fig)
    print("Wrote 11_unit_count_confound.png")

    print(f"\nAll figures written to {RUN_DIR}")


if __name__ == "__main__":
    main()
