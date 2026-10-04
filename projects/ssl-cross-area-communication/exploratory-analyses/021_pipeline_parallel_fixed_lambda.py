"""Same analysis as 020 (PSTH -> residual -> PCA -> CCA, across peri-
stimulus time, R+/R- split with SEM, correlation-vs-canonical-dimension),
sped up per Axel's 2026-09-01 follow-up ("parallelize and keep lambda
constant per conditions... other ways to vectorize?"):

1. **Fixed lambda per condition, no per-session CV.** Both binning smoke
   tests under 020 (5ms/2ms and 10ms/5ms) consistently found the same
   pattern: dense conditions (whisker x0, no_stim x0, >~150 trials) always
   selected lambda=0.001; sparse conditions (whisker x1, auditory x1,
   no_stim x1, ~50-100 trials) predominantly selected lambda=1.0. Using
   this as a fixed per-condition lookup removes ALL CV fit+transform
   cycles (was up to 3 lambdas x 2 folds = 6 extra full fit+transform
   round-trips per CV-triggered condition, on top of the final fit).
2. **Truncate `PartialCCA.weights_x_`/`weights_y_` to MAX_DIMS before
   `.transform()`.** The vendored library's `transform()` loops in pure
   Python over ALL `min(n_units_a, n_units_b)` canonical dimensions --
   for this pair, potentially 500-800+ -- even though only the first
   MAX_DIMS=10 are ever read downstream. `canonical_correlations_`/
   `weights_x_`/`weights_y_` are already sorted by descending canonical
   correlation, so slicing to the first MAX_DIMS columns right after
   `.fit()` gives IDENTICAL results for the dims actually used while
   cutting `.transform()` cost by roughly (n_units/MAX_DIMS)x -- this was
   likely the single largest hidden cost in 018/019/020, worse than the
   CV overhead itself.
3. **Vectorized spike-binning** (`003_cca_lib.py`): dead-zone bin clipping
   is now computed ONCE per (session, condition) and shared across all
   units instead of being redundantly recomputed per unit, and the
   per-bin Python loop over ~squeeze searchsorted calls is replaced by a
   single batched `searchsorted` call over the full (trial, bin) grid.
4. **Parallelized across sessions** via `ProcessPoolExecutor` -- this
   machine has 24 cores/32 threads and the serial 018-020 versions used
   exactly one. Sessions are independent (own unit sets, own trials), so
   this is an exact, safe speedup with no methodology change.

Note on "reduce number of shuffles": no shuffle/surrogate significance
test is currently wired into this script (018-020 dropped it in favor of
descriptive cross-session averaging + CV'd/fixed regularization) -- the
vendored `PartialCCA.surrogate_test(n_surrogates=100)` does a full refit
per surrogate and is NOT called here. If/when trial-identity-shuffle
significance testing is added back for the full 39-pair run, default to a
reduced count (e.g. 20-30) rather than the library's 100-default, per this
instruction -- flagging rather than guessing a number now since it isn't
wired up yet.
"""
from __future__ import annotations

import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
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
FIXED_LAMBDA_BY_CONDITION = {
    ("whisker_trial", 0): 1e-3,
    ("whisker_trial", 1): 1.0,
    ("auditory_trial", 1): 1.0,
    ("no_stim_trial", 0): 1e-3,
    ("no_stim_trial", 1): 1.0,
}
TIER_FN = lambda ut: ut["quality_label"].isin(["good", "mua"])
MAX_DIMS = 10
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
N_WORKERS = 10  # of 24 cores/32 threads -- leaves headroom for the OS/other work


def spikes_of(units):
    return [np.sort(np.asarray(r["spike_times"], dtype=float)) for _, r in units.iterrows()]


def to_full_timeline(values_valid, valid_bins):
    full = np.full(len(valid_bins), np.nan)
    full[valid_bins] = values_valid
    return full


def sign_align(curves: list[np.ndarray]) -> list[np.ndarray]:
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


def sem_across(arr_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    a = np.vstack(arr_list)
    mean = np.nanmean(a, axis=0)
    n = np.sum(~np.isnan(a), axis=0)
    sem = np.divide(np.nanstd(a, axis=0, ddof=1), np.sqrt(n), out=np.full_like(mean, np.nan), where=n > 1)
    return mean, sem


def process_one_session(task: tuple) -> tuple:
    """Runs in a worker process. `task` is fully picklable (numpy arrays,
    a plain DataFrame slice, python scalars) -- no shared state with the
    parent beyond what's passed in."""
    (session_id, reward_group, mouse_id, n_units_a, n_units_b,
     spikes_a, spikes_b, trials_sess, bin_starts, valid_bins_by_cond) = task

    out: dict = {}
    for trial_type, lick_flag in CONDITIONS:
        is_whisker = trial_type == "whisker_trial"
        tt_df = trials_sess[(trials_sess["trial_type"] == trial_type)
                             & (trials_sess["lick_flag"] == lick_flag)].sort_values("start_time")
        starts = tt_df["start_time"].to_numpy(dtype=float)
        if len(starts) < MIN_TRIALS:
            continue
        cond = (trial_type, lick_flag)
        valid_bins = valid_bins_by_cond[cond]

        tensor_a = cca_lib.population_tensor_sliding(spikes_a, starts, is_whisker, bin_starts)
        tensor_b = cca_lib.population_tensor_sliding(spikes_b, starts, is_whisker, bin_starts)
        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)

        row: dict = {
            "n_trials": len(starts),
            "psth_a": np.nanmean(tensor_a, axis=(0, 2)), "psth_b": np.nanmean(tensor_b, axis=(0, 2)),
            "resid_a": np.nanmean(np.abs(resid_a), axis=(0, 2)), "resid_b": np.nanmean(np.abs(resid_b), axis=(0, 2)),
        }

        seed = zlib.crc32(f"{session_id}_{trial_type}_{lick_flag}".encode()) % (2**32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(resid_a.shape[0])
        half = len(perm) // 2
        train_idx, test_idx = perm[:half], perm[half:]
        Xa_train, _ = cca_lib.flatten_trial_bins(resid_a[train_idx][:, valid_bins, :])
        Xb_train, _ = cca_lib.flatten_trial_bins(resid_b[train_idx][:, valid_bins, :])
        Xa_test = resid_a[test_idx][:, valid_bins, :].reshape(-1, len(spikes_a))
        Xb_test = resid_b[test_idx][:, valid_bins, :].reshape(-1, len(spikes_b))
        n_test, n_bins_v = len(test_idx), int(valid_bins.sum())

        pca_a = _PCA(n_components=1).fit(Xa_train)
        pca_b = _PCA(n_components=1).fit(Xb_train)
        pc1_a = pca_a.transform(Xa_test).reshape(n_test, n_bins_v)
        pc1_b = pca_b.transform(Xb_test).reshape(n_test, n_bins_v)
        row["pc1_a"] = to_full_timeline(np.nanmean(pc1_a, axis=0), valid_bins)
        row["pc1_b"] = to_full_timeline(np.nanmean(pc1_b, axis=0), valid_bins)

        dim_corrs = np.full(MAX_DIMS, np.nan)
        lam = FIXED_LAMBDA_BY_CONDITION[cond]
        try:
            model = PartialCCA(regularization=lam).fit(Xa_train, Xb_train, None, verbose=False)
            # Truncate to the dims we actually use -- transform() loops in
            # pure Python over weights_x_.shape[1] (up to min(n_x,n_y)
            # dims); weights are already sorted by descending correlation,
            # so this changes nothing about the first MAX_DIMS results.
            n_dims_avail = model.weights_x_.shape[1]
            n_keep = min(n_dims_avail, MAX_DIMS)
            model.weights_x_ = model.weights_x_[:, :n_keep]
            model.weights_y_ = model.weights_y_[:, :n_keep]
            ca, cb = model.transform(Xa_test, Xb_test)
            ca1 = ca[0].reshape(n_test, n_bins_v)
            cb1 = cb[0].reshape(n_test, n_bins_v)
            ca1_full = to_full_timeline(np.nanmean(ca1, axis=0), valid_bins)
            cb1_full = to_full_timeline(np.nanmean(cb1, axis=0), valid_bins)
            row["cca_a"] = ca1_full
            row["cca_b"] = cb1_full
            m = ~(np.isnan(ca1_full) | np.isnan(cb1_full))
            row["cca_r"] = np.corrcoef(ca1_full[m], cb1_full[m])[0, 1] if m.sum() > 3 else np.nan

            ca1_full_2d = np.full((n_test, len(valid_bins)), np.nan)
            cb1_full_2d = np.full((n_test, len(valid_bins)), np.nan)
            ca1_full_2d[:, valid_bins] = ca1
            cb1_full_2d[:, valid_bins] = cb1
            row["corr_t"] = cca_lib.canonical_correlation_across_time(ca1_full_2d, cb1_full_2d)

            for d in range(n_keep):
                cad_mean = np.nanmean(ca[d].reshape(n_test, n_bins_v), axis=0)
                cbd_mean = np.nanmean(cb[d].reshape(n_test, n_bins_v), axis=0)
                if len(cad_mean) > 3:
                    dim_corrs[d] = np.corrcoef(cad_mean, cbd_mean)[0, 1]
            row["lambda"] = lam
        except Exception as e:
            print(f"  {session_id} {cond}: CCA failed ({e})", flush=True)
        row["cc_dims"] = dim_corrs
        out[cond] = row
    return session_id, reward_group, mouse_id, n_units_a, n_units_b, out


def main() -> None:
    session_ids = pd.read_csv(ARTIFACTS_DIR / "motor_striatum_session_list.csv")["session_id"].tolist()
    smoke_n = os.environ.get("SSL_SMOKE_N_SESSIONS")
    if smoke_n:
        session_ids = session_ids[: int(smoke_n)]
        print(f"[SMOKE TEST] capped to first {len(session_ids)} sessions")
    files = [f"{s}.nwb" for s in session_ids]
    print(f"Loading {len(files)} sessions for {AREA_A} vs {AREA_B}...")

    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=16)
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[TIER_FN(unit_table)]

    bin_starts = cca_lib.sliding_window_starts()
    valid_bins_by_cond = {
        cond: ~cca_lib.dead_zone_bin_mask_sliding(bin_starts, cca_lib.BIN_WIDTH, cond[0] == "whisker_trial")
        for cond in CONDITIONS
    }

    tasks = []
    for session_id in session_ids:
        units_a = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_A)]
        units_b = unit_table[(unit_table["session_id"] == session_id) & (unit_table["area_group_coarse"] == AREA_B)]
        if len(units_a) < MIN_UNITS or len(units_b) < MIN_UNITS:
            continue
        reward_group = units_a["reward_group"].iloc[0]
        mouse_id = units_a["mouse_id"].iloc[0]
        trials_sess = trial_table[(trial_table["session_id"] == session_id) & (trial_table["context"] == "active")]
        tasks.append((session_id, reward_group, mouse_id, len(units_a), len(units_b),
                      spikes_of(units_a), spikes_of(units_b), trials_sess, bin_starts, valid_bins_by_cond))

    print(f"{len(tasks)} sessions queued, running with {N_WORKERS} parallel workers...")

    curves = {cond: {"session_id": [], "reward_group": [], "mouse_id": [],
                      "n_units_a": [], "n_units_b": [], "n_trials": [],
                      "psth_a": [], "psth_b": [], "resid_a": [], "resid_b": [],
                      "pc1_a": [], "pc1_b": [], "cca_a": [], "cca_b": [], "cca_r": [],
                      "corr_t": [], "lambdas": [], "cc_dims": []}
              for cond in CONDITIONS}
    n_sessions_used = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(process_one_session, t): t[0] for t in tasks}
        n_done = 0
        for fut in as_completed(futures):
            session_id = futures[fut]
            try:
                sid, reward_group, mouse_id, n_units_a, n_units_b, results = fut.result()
            except Exception as e:
                print(f"{session_id}: WORKER FAILED ({e})", flush=True)
                continue
            used = False
            for cond, row in results.items():
                c = curves[cond]
                c["session_id"].append(sid)
                c["reward_group"].append(reward_group)
                c["mouse_id"].append(mouse_id)
                c["n_units_a"].append(n_units_a)
                c["n_units_b"].append(n_units_b)
                c["n_trials"].append(row.get("n_trials", np.nan))
                for key in ("psth_a", "psth_b", "resid_a", "resid_b", "pc1_a", "pc1_b",
                            "cca_a", "cca_b", "cca_r", "corr_t", "cc_dims"):
                    c[key].append(row.get(key))
                c["lambdas"].append(row.get("lambda", np.nan))
                used = True
            n_sessions_used += int(used)
            n_done += 1
            print(f"[{n_done}/{len(tasks)}] {sid} ({reward_group}): done", flush=True)

    print(f"\n{n_sessions_used}/{len(session_ids)} sessions contributed >=1 condition")
    for cond in CONDITIONS:
        print(f"  {cond}: n={len(curves[cond]['psth_a'])} sessions, fixed lambda={FIXED_LAMBDA_BY_CONDITION[cond]:.4g}")

    bin_centers = cca_lib.sliding_window_centers(bin_starts)
    config = {
        "area_a": AREA_A, "area_b": AREA_B, "conditions": CONDITIONS,
        "min_units": MIN_UNITS, "min_trials": MIN_TRIALS,
        "fixed_lambda_by_condition": FIXED_LAMBDA_BY_CONDITION,
        "quality_tiers": ["good", "mua"], "max_dims": MAX_DIMS,
        "bin_width_s": cca_lib.BIN_WIDTH, "bin_stride_s": cca_lib.BIN_STRIDE,
        "dead_zone_start_s": cca_lib.DEAD_ZONE_START_S, "dead_zone_stop_s": cca_lib.DEAD_ZONE_STOP_S,
        "session_list_source": "motor_striatum_session_list.csv",
        "n_sessions_loaded": len(session_ids), "n_sessions_used": n_sessions_used,
        "day_to_analyze": "learning", "cohort_colors": COHORT_COLOR,
        "n_workers": N_WORKERS, "script": "021_pipeline_parallel_fixed_lambda.py",
    }
    with open(ARTIFACTS_DIR / "pipeline_full_dataset_results.pkl", "wb") as f:
        pickle.dump({"curves": curves, "bin_centers": bin_centers, "config": config}, f)
    print("Wrote pipeline_full_dataset_results.pkl (curves + full config/provenance)")

    t_ms = bin_centers * 1000

    fig, axes = plt.subplots(5, 5, figsize=(32, 26))
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        c = curves[cond]
        rg = np.array(c["reward_group"])
        col_label = f"{trial_type} (lick={lick_flag})"

        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            n_c = len(idx)
            if n_c == 0:
                continue
            color = COHORT_COLOR[cohort]

            ax = axes[0, col]
            m_a, s_a = sem_across([c["psth_a"][i] for i in idx])
            m_b, s_b = sem_across([c["psth_b"][i] for i in idx])
            ax.plot(t_ms, m_a, color=color, linestyle="-", label=f"{cohort} {AREA_A} (n={n_c})")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--", label=f"{cohort} {AREA_B}")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)

            ax = axes[1, col]
            m_a, s_a = sem_across([c["resid_a"][i] for i in idx])
            m_b, s_b = sem_across([c["resid_b"][i] for i in idx])
            ax.plot(t_ms, m_a, color=color, linestyle="-")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)

            ax = axes[2, col]
            pc1_a_aligned = sign_align([c["pc1_a"][i] for i in idx])
            pc1_b_aligned = sign_align([c["pc1_b"][i] for i in idx])
            m_a, s_a = sem_across(pc1_a_aligned)
            m_b, s_b = sem_across(pc1_b_aligned)
            ax.plot(t_ms, m_a, color=color, linestyle="-")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)
            ax.axhline(0, color="black", linewidth=0.8)

            ax = axes[3, col]
            cca_a_aligned = sign_align([c["cca_a"][i] for i in idx])
            cca_b_aligned = sign_align([c["cca_b"][i] for i in idx])
            m_a, s_a = sem_across(cca_a_aligned)
            m_b, s_b = sem_across(cca_b_aligned)
            ax.plot(t_ms, m_a, color=color, linestyle="-")
            ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
            ax.plot(t_ms, m_b, color=color, linestyle="--")
            ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)
            ax.axhline(0, color="black", linewidth=0.8)

            ax = axes[4, col]
            m_t, s_t = sem_across([c["corr_t"][i] for i in idx])
            ax.plot(t_ms, m_t, color=color, label=f"{cohort} (n={n_c})")
            ax.fill_between(t_ms, m_t - s_t, m_t + s_t, color=color, alpha=0.2)
            ax.axhline(0, color="black", linewidth=0.8)

        axes[0, col].set_title(f"{col_label}\n(solid=Area A, dashed=Area B)")
        axes[4, col].set_xlabel("time from start_time (ms)")
        if col == 0:
            axes[0, col].set_ylabel("PSTH\nmean rate (Hz)")
            axes[1, col].set_ylabel("residual\nmean |residual| (Hz)")
            axes[2, col].set_ylabel("PCA\nPC1 (held-out, sign-aligned/cohort)")
            axes[3, col].set_ylabel("CCA variate\n(dim 1, sign-aligned/cohort)")
            axes[4, col].set_ylabel("CCA correlation\nacross-trial r per time bin")
        axes[0, col].legend(fontsize=6)
        axes[4, col].legend(fontsize=7)

    fig.suptitle(f"R+ vs R- comparison, {AREA_A} vs {AREA_B} -- mean +/- SEM across sessions per cohort "
                 f"(colors: R+={COHORT_COLOR['R+']}, R-={COHORT_COLOR['R-']}; fixed lambda per condition, "
                 f"no per-session CV)", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ARTIFACTS_DIR / "example_pair_04_rplus_rminus_comparison.png", dpi=110)
    plt.close(fig)
    print("Wrote example_pair_04_rplus_rminus_comparison.png")

    fig, axes = plt.subplots(1, 5, figsize=(30, 5.5), sharey=True)
    dims = np.arange(1, MAX_DIMS + 1)
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        c = curves[cond]
        rg = np.array(c["reward_group"])
        ax = axes[col]
        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            if len(idx) == 0:
                continue
            arr = np.abs(np.vstack([c["cc_dims"][i] for i in idx]))
            mean = np.nanmean(arr, axis=0)
            n = np.sum(~np.isnan(arr), axis=0)
            sem = np.divide(np.nanstd(arr, axis=0, ddof=1), np.sqrt(n),
                             out=np.full_like(mean, np.nan), where=n > 1)
            color = COHORT_COLOR[cohort]
            ax.errorbar(dims, mean, yerr=sem, color=color, marker="o", markersize=4,
                        label=f"{cohort} (n={len(idx)} sessions)")
        ax.set_title(f"{trial_type} (lick={lick_flag})")
        ax.set_xlabel("canonical dimension")
        ax.set_xticks(dims)
        if col == 0:
            ax.set_ylabel("held-out |canonical correlation|\n(mean +/- SEM across sessions)")
        ax.legend(fontsize=8)
        ax.axhline(0, color="black", linewidth=0.6)

    fig.suptitle(f"Canonical correlation vs. dimension, {AREA_A} vs {AREA_B}, R+ vs R- "
                 f"(held-out, dims capped at {MAX_DIMS}, fixed lambda per condition)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "example_pair_05_correlation_vs_dimension.png", dpi=110)
    plt.close(fig)
    print("Wrote example_pair_05_correlation_vs_dimension.png")


if __name__ == "__main__":
    main()
