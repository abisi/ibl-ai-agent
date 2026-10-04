"""Same analysis as 021 (parallel, fixed-lambda-per-condition CCA pipeline),
extended per Axel's 2026-09-01 follow-up on well-posedness and baselines:

1. **PCA pre-reduction on the neuron axis, before CCA.** Confirmed with
   Axel: PCA reduces the FEATURE dimension of each area's
   (n_samples=trials x bins, n_units) residual matrix -- not trials, not
   bins. Without this, `PartialCCA` fits directly on ~500-900 raw units
   with only ridge regularization keeping Sigma_xx/Sigma_yy invertible --
   numerically fine, but statistically risky, since p (units) can vastly
   exceed the EFFECTIVE independent sample count (trials, not
   trials x bins -- overlapping sliding-window bins are not independent
   draws). Fix: reduce each area to K = min(K_90%-variance, n_trials // 5)
   PCs (fit on TRAIN only, applied to TEST via the same fitted transform,
   so held-out evaluation stays honest), independently per area. This also
   shrinks the CCA eigendecomposition itself (K x K instead of ~800 x 800),
   a further speed bonus.
2. **PCA baseline** (from the original plan, now computed as a scalar):
   correlation between each area's own PC1 (no cross-area optimization at
   all -- just "do the dominant variance directions happen to line up"),
   distinct from PCA-reduced CCA (which finds the OPTIMAL linear
   combination *within* the K-PC space maximizing cross-area correlation).
   Comparing the two shows whether CCA finds structure beyond naive
   variance alignment.
3. **Trial-identity shuffle null baseline**, 10 shuffles/condition (Axel:
   "keep shuffling trial ids relative to neural activity ... do 10
   shuffles"): area B's trial axis is globally permuted (decoupling which
   B trial pairs with which A trial) BEFORE the same
   PCA-reduce+fit+transform pipeline is rerun from scratch (a full refit
   per shuffle, not a reprojection through the true fitted weights, per
   the project's established shuffle-baseline convention) -- gives a null
   distribution of held-out dim-1 correlation to compare the true value
   against.
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
MIN_UNITS_AFTER_RATE_FILTER = 5  # floor AFTER cca_lib.rate_filter_mask (per condition, can be
                                 # much lower than MIN_UNITS since that's the session/area-level
                                 # inclusion floor computed on RAW unit counts) -- below this,
                                 # skip the (session, condition) rather than fit PCA/CCA on too
                                 # few surviving units.
FIXED_LAMBDA_BY_CONDITION = {
    ("whisker_trial", 0): 1e-3,
    ("whisker_trial", 1): 1.0,
    ("auditory_trial", 1): 1.0,
    ("no_stim_trial", 0): 1e-3,
    ("no_stim_trial", 1): 1.0,
}
TIER_FN = lambda ut: ut["quality_label"].isin(["good", "mua"])
MAX_DIMS = 10
N_SHUFFLES = 10
PCA_VAR_TARGET = 0.90  # as of 2026-09-02, Axel: smoke test at 99% showed a ~15-20x per-session
                       # slowdown (1ms bins + uncapped high-dim CCA fits) projecting to 2-4 days
                       # for the full coarse batch; Axel's call: "have a 90% variance threshold
                       # for PCA, optimize and vectorize as much as possible" -- back to 0.90
                       # (still uncapped by n_trials -- see pca_reduce docstring)
PCA_TRIALS_PER_PC = 5  # no longer used to CAP K (see pca_reduce docstring, 2026-09-02); kept
                       # only so main_pass_params_fingerprint()'s hash stays stable across the
                       # change (still logged/reported for provenance)
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
N_WORKERS = 10


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


def pca_reduce(Xtr: np.ndarray, Xte: np.ndarray, n_trials: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Z-score each neuron (column) on TRAIN statistics, then reduce the
    FEATURE (neuron) axis of Xtr/Xte to K components fit on TRAIN only.

    Z-scoring (2026-09-02, Axel: "implement z-scoring before PCA at the
    neuron level"): divides each neuron's already-mean-subtracted noise-
    correlation residual by that neuron's own train-set SD, matching the
    published pCCA paper's preprocessing (z-score before PCA, since PCA is
    scale-sensitive) -- applied ON TOP OF, not instead of, the existing
    per-trial PSTH-mean subtraction (`noise_correlation_residuals`), which
    stays the "noise correlation" definition. Constant/silent units (SD~0)
    are left un-rescaled (divide by 1) rather than blown up.

    K = #PCs needed for >=PCA_VAR_TARGET (99%) train variance, capped only
    by the mathematical ceiling min(n_units, n_train_samples-1) -- NOT by
    n_trials//PCA_TRIALS_PER_PC (that artificial trial-count cap was
    dropped 2026-09-02 so it can't silently prevent reaching 99%; the much
    larger n_train_samples at 1ms bins plus PartialCCA's ridge
    regularization are relied on for well-posedness instead, per Axel's
    explicit "make sure the number of PCs retain more than 99% of the
    variance"). PCA is fit adaptively (small n_components first, doubled/
    grown only if 99% isn't reached yet) so this doesn't default back to
    an expensive full-rank SVD on every call."""
    n_units = Xtr.shape[1]
    mean = Xtr.mean(axis=0, keepdims=True)
    std = Xtr.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    Xtr_z = (Xtr - mean) / std
    Xte_z = (Xte - mean) / std

    max_k = max(1, min(n_units, Xtr.shape[0] - 1))
    guess = min(max_k, 50)
    while True:
        pca = _PCA(n_components=guess).fit(Xtr_z)
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        if cum_var[-1] >= PCA_VAR_TARGET or guess >= max_k:
            break
        guess = min(max_k, guess * 4)
    k = int(np.searchsorted(cum_var, PCA_VAR_TARGET) + 1)
    k = max(1, min(k, guess))
    return pca.transform(Xtr_z)[:, :k], pca.transform(Xte_z)[:, :k], k


def fit_and_eval(resid_a: np.ndarray, resid_b: np.ndarray, valid_bins: np.ndarray,
                  train_idx: np.ndarray, test_idx: np.ndarray, lam: float, n_trials_for_k: int,
                  max_dims: int = MAX_DIMS):
    """One PCA-reduce + PartialCCA fit + held-out transform cycle. Returns
    (r_dim1, ca_full_2d_or_None, cb_full_2d_or_None, dim_corrs, k_a, k_b)."""
    n_bins_total = len(valid_bins)
    Xa_train, _ = cca_lib.flatten_trial_bins(resid_a[train_idx][:, valid_bins, :])
    Xb_train, _ = cca_lib.flatten_trial_bins(resid_b[train_idx][:, valid_bins, :])
    Xa_test = resid_a[test_idx][:, valid_bins, :].reshape(-1, resid_a.shape[-1])
    Xb_test = resid_b[test_idx][:, valid_bins, :].reshape(-1, resid_b.shape[-1])
    n_test, n_bins_v = len(test_idx), int(valid_bins.sum())

    Xa_train_r, Xa_test_r, k_a = pca_reduce(Xa_train, Xa_test, n_trials_for_k)
    Xb_train_r, Xb_test_r, k_b = pca_reduce(Xb_train, Xb_test, n_trials_for_k)

    dim_corrs = np.full(max_dims, np.nan)
    model = PartialCCA(regularization=lam).fit(Xa_train_r, Xb_train_r, None, verbose=False)
    n_keep = min(model.weights_x_.shape[1], max_dims)
    model.weights_x_ = model.weights_x_[:, :n_keep]
    model.weights_y_ = model.weights_y_[:, :n_keep]
    ca, cb = model.transform(Xa_test_r, Xb_test_r)
    ca1 = ca[0].reshape(n_test, n_bins_v)
    cb1 = cb[0].reshape(n_test, n_bins_v)
    ca1_full = to_full_timeline(np.nanmean(ca1, axis=0), valid_bins)
    cb1_full = to_full_timeline(np.nanmean(cb1, axis=0), valid_bins)
    m = ~(np.isnan(ca1_full) | np.isnan(cb1_full))
    r_dim1 = np.corrcoef(ca1_full[m], cb1_full[m])[0, 1] if m.sum() > 3 else np.nan

    for d in range(n_keep):
        cad_mean = np.nanmean(ca[d].reshape(n_test, n_bins_v), axis=0)
        cbd_mean = np.nanmean(cb[d].reshape(n_test, n_bins_v), axis=0)
        if len(cad_mean) > 3:
            dim_corrs[d] = np.corrcoef(cad_mean, cbd_mean)[0, 1]

    ca_full_2d = np.full((n_test, n_bins_total), np.nan)
    cb_full_2d = np.full((n_test, n_bins_total), np.nan)
    ca_full_2d[:, valid_bins] = ca1
    cb_full_2d[:, valid_bins] = cb1
    return r_dim1, ca_full_2d, cb_full_2d, ca1_full, cb1_full, dim_corrs, k_a, k_b


def shift_tensor_nan(tensor: np.ndarray, lag_bins: int) -> np.ndarray:
    """Shift `tensor` along the bin axis (axis=1) by `lag_bins`, WITHOUT
    wraparound: positions shifted in from outside the original range
    become NaN, exactly like a bin that was already NaN at its source
    position (e.g. inside the whisker dead zone) -- NaN propagates
    through the shift automatically since this is plain array indexing,
    not modular arithmetic. `np.roll`'s circular wraparound would instead
    splice in data from the opposite end of the -200/+500ms analysis
    window -- an unrelated time point -- which is wrong for any lag large
    enough to wrap, and can also rotate the dead zone's NaN gap into bins
    a fixed valid-bins mask still treated as valid. Moved here from
    023_full_figure_set.py (2026-09-02) so `process_session_full` (024)
    can run the lag sweep in the SAME pass as the true (lag=0) fit,
    instead of a separate script recomputing tensors/residuals from
    scratch -- see `lag_sweep` below."""
    n_bins = tensor.shape[1]
    shifted = np.full_like(tensor, np.nan)
    if lag_bins == 0:
        return tensor.copy()
    if lag_bins > 0 and lag_bins < n_bins:
        shifted[:, : n_bins - lag_bins, :] = tensor[:, lag_bins:, :]
    elif lag_bins < 0 and -lag_bins < n_bins:
        shifted[:, -lag_bins:, :] = tensor[:, : n_bins + lag_bins, :]
    return shifted


def lag_sweep(resid_a, resid_b, valid_bins, train_idx, test_idx, lam, n_trials, lag_bins_range):
    """For each lag, shift B (non-circular) and rebuild the valid-bins
    mask FOR THAT LAG from scratch: a bin only counts if BOTH (a) A's own
    bin is dead-zone-valid (the fixed, unshifted `valid_bins` -- already
    widened for the causal smoothing kernel's reach if smoothing is in
    use, since `valid_bins` is built from `dead_zone_bin_mask_sliding_causal`
    upstream) and (b) B's shifted bin is non-NaN -- false whenever the
    shift pulled in an out-of-window position OR a dead-zone/smoothing-
    contaminated source bin."""
    rs = []
    for lag in lag_bins_range:
        resid_b_shifted = shift_tensor_nan(resid_b, lag)
        # NaN pattern from the shift is identical across trials/units (the
        # shift and the underlying dead-zone mask are both bin-axis-only
        # operations), so any single (trial, unit) slice reveals it.
        b_ok = ~np.isnan(resid_b_shifted[0, :, 0])
        valid_lag = valid_bins & b_ok
        if valid_lag.sum() < 5:
            rs.append(np.nan)
            continue
        r, *_ = fit_and_eval(resid_a, resid_b_shifted, valid_lag, train_idx, test_idx, lam, n_trials)
        rs.append(r)
    return np.array(rs)


def process_one_session(task: tuple) -> tuple:
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
        n_trials = len(starts)

        tensor_a = cca_lib.population_tensor_sliding_smoothed(spikes_a, starts, is_whisker, bin_starts)
        tensor_b = cca_lib.population_tensor_sliding_smoothed(spikes_b, starts, is_whisker, bin_starts)
        # Drop units that don't clear MIN_UNIT_RATE_HZ in THIS condition's PSTH before they
        # ever reach z-scoring (in pca_reduce) -- see cca_lib.rate_filter_mask docstring for
        # why this is the fix, not a rescaling of sparse units.
        keep_a = cca_lib.rate_filter_mask(tensor_a)
        keep_b = cca_lib.rate_filter_mask(tensor_b)
        tensor_a, tensor_b = tensor_a[:, :, keep_a], tensor_b[:, :, keep_b]
        n_units_a_eff, n_units_b_eff = int(keep_a.sum()), int(keep_b.sum())
        if n_units_a_eff < MIN_UNITS_AFTER_RATE_FILTER or n_units_b_eff < MIN_UNITS_AFTER_RATE_FILTER:
            print(f"  {session_id} {cond}: SKIPPED (only {n_units_a_eff}/{n_units_b_eff} units "
                  f"clear {cca_lib.MIN_UNIT_RATE_HZ}Hz)", flush=True)
            continue
        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)

        row: dict = {
            "n_trials": n_trials,
            "psth_a": np.nanmean(tensor_a, axis=(0, 2)), "psth_b": np.nanmean(tensor_b, axis=(0, 2)),
            "resid_a": np.nanmean(np.abs(resid_a), axis=(0, 2)), "resid_b": np.nanmean(np.abs(resid_b), axis=(0, 2)),
        }

        seed = zlib.crc32(f"{session_id}_{trial_type}_{lick_flag}".encode()) % (2**32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(resid_a.shape[0])
        half = len(perm) // 2
        train_idx, test_idx = perm[:half], perm[half:]
        n_bins_v = int(valid_bins.sum())

        # Single-PC-per-area diagnostic (unreduced) -- also doubles as the
        # PCA baseline's inputs (naive PC1-vs-PC1 correlation, no CCA
        # optimization).
        Xa_train_full, _ = cca_lib.flatten_trial_bins(resid_a[train_idx][:, valid_bins, :])
        Xb_train_full, _ = cca_lib.flatten_trial_bins(resid_b[train_idx][:, valid_bins, :])
        Xa_test_full = resid_a[test_idx][:, valid_bins, :].reshape(-1, n_units_a_eff)
        Xb_test_full = resid_b[test_idx][:, valid_bins, :].reshape(-1, n_units_b_eff)
        n_test = len(test_idx)
        pca_a1 = _PCA(n_components=1).fit(Xa_train_full)
        pca_b1 = _PCA(n_components=1).fit(Xb_train_full)
        pc1_a = pca_a1.transform(Xa_test_full).reshape(n_test, n_bins_v)
        pc1_b = pca_b1.transform(Xb_test_full).reshape(n_test, n_bins_v)
        pc1_a_mean = np.nanmean(pc1_a, axis=0)
        pc1_b_mean = np.nanmean(pc1_b, axis=0)
        row["pc1_a"] = to_full_timeline(pc1_a_mean, valid_bins)
        row["pc1_b"] = to_full_timeline(pc1_b_mean, valid_bins)
        row["pca_baseline_r"] = (np.corrcoef(pc1_a_mean, pc1_b_mean)[0, 1]
                                  if len(pc1_a_mean) > 3 else np.nan)

        lam = FIXED_LAMBDA_BY_CONDITION[cond]
        try:
            r_dim1, ca_full_2d, cb_full_2d, ca1_full, cb1_full, dim_corrs, k_a, k_b = fit_and_eval(
                resid_a, resid_b, valid_bins, train_idx, test_idx, lam, n_trials)
            row["cca_a"] = ca1_full
            row["cca_b"] = cb1_full
            row["cca_r"] = r_dim1
            row["corr_t"] = cca_lib.canonical_correlation_across_time(ca_full_2d, cb_full_2d)
            row["cc_dims"] = dim_corrs
            row["lambda"] = lam
            row["k_a"] = k_a
            row["k_b"] = k_b

            null_rs = np.full(N_SHUFFLES, np.nan)
            for s in range(N_SHUFFLES):
                shuf_seed = zlib.crc32(f"{session_id}_{trial_type}_{lick_flag}_shuf{s}".encode()) % (2**32)
                shuf_rng = np.random.default_rng(shuf_seed)
                shuf_perm = shuf_rng.permutation(resid_b.shape[0])
                try:
                    r_null, *_ = fit_and_eval(resid_a, resid_b[shuf_perm], valid_bins,
                                               train_idx, test_idx, lam, n_trials)
                    null_rs[s] = r_null
                except Exception:
                    continue
            row["shuffle_null"] = null_rs
        except Exception as e:
            print(f"  {session_id} {cond}: CCA failed ({e})", flush=True)
            row["cc_dims"] = np.full(MAX_DIMS, np.nan)
            row["shuffle_null"] = np.full(N_SHUFFLES, np.nan)
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
    smooth_kernel = cca_lib.causal_half_gaussian_kernel()
    valid_bins_by_cond = {
        cond: ~cca_lib.dead_zone_bin_mask_sliding_causal(bin_starts, cca_lib.BIN_WIDTH,
                                                          cond[0] == "whisker_trial", smooth_kernel)
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

    print(f"{len(tasks)} sessions queued, running with {N_WORKERS} parallel workers "
          f"({N_SHUFFLES} shuffles/condition)...")

    curves = {cond: {"session_id": [], "reward_group": [], "mouse_id": [],
                      "n_units_a": [], "n_units_b": [], "n_trials": [],
                      "psth_a": [], "psth_b": [], "resid_a": [], "resid_b": [],
                      "pc1_a": [], "pc1_b": [], "pca_baseline_r": [],
                      "cca_a": [], "cca_b": [], "cca_r": [], "corr_t": [],
                      "lambdas": [], "cc_dims": [], "shuffle_null": [], "k_a": [], "k_b": []}
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
                for key in ("psth_a", "psth_b", "resid_a", "resid_b", "pc1_a", "pc1_b", "pca_baseline_r",
                            "cca_a", "cca_b", "cca_r", "corr_t", "cc_dims", "shuffle_null", "k_a", "k_b"):
                    c[key].append(row.get(key))
                c["lambdas"].append(row.get("lambda", np.nan))
                used = True
            n_sessions_used += int(used)
            n_done += 1
            print(f"[{n_done}/{len(tasks)}] {sid} ({reward_group}): done", flush=True)

    print(f"\n{n_sessions_used}/{len(session_ids)} sessions contributed >=1 condition")
    for cond in CONDITIONS:
        c = curves[cond]
        if not c["cca_r"]:
            continue
        true_r = np.nanmean(np.abs(c["cca_r"]))
        pca_r = np.nanmean(np.abs(c["pca_baseline_r"]))
        null_r = np.nanmean([np.nanmean(np.abs(x)) for x in c["shuffle_null"] if x is not None])
        k_a_med = np.nanmedian(c["k_a"]) if c["k_a"] else float("nan")
        k_b_med = np.nanmedian(c["k_b"]) if c["k_b"] else float("nan")
        print(f"  {cond}: n={len(c['psth_a'])} sessions, lambda={FIXED_LAMBDA_BY_CONDITION[cond]:.4g}, "
              f"median K_a={k_a_med:.0f} K_b={k_b_med:.0f} -- "
              f"mean|r|: CCA={true_r:.3f} PCA-baseline={pca_r:.3f} shuffle-null={null_r:.3f}")

    bin_centers = cca_lib.sliding_window_centers(bin_starts)
    config = {
        "area_a": AREA_A, "area_b": AREA_B, "conditions": CONDITIONS,
        "min_units": MIN_UNITS, "min_trials": MIN_TRIALS,
        "fixed_lambda_by_condition": FIXED_LAMBDA_BY_CONDITION,
        "quality_tiers": ["good", "mua"], "max_dims": MAX_DIMS,
        "n_shuffles": N_SHUFFLES, "pca_var_target": PCA_VAR_TARGET, "pca_trials_per_pc": PCA_TRIALS_PER_PC,
        "bin_width_s": cca_lib.BIN_WIDTH, "bin_stride_s": cca_lib.BIN_STRIDE,
        "dead_zone_start_s": cca_lib.DEAD_ZONE_START_S, "dead_zone_stop_s": cca_lib.DEAD_ZONE_STOP_S,
        "session_list_source": "motor_striatum_session_list.csv",
        "n_sessions_loaded": len(session_ids), "n_sessions_used": n_sessions_used,
        "day_to_analyze": "learning", "cohort_colors": COHORT_COLOR,
        "n_workers": N_WORKERS, "script": "022_pipeline_pca_reduced_shuffle_baseline.py",
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
            axes[3, col].set_ylabel("CCA variate\n(dim 1, PCA-reduced, sign-aligned/cohort)")
            axes[4, col].set_ylabel("CCA correlation\nacross-trial r per time bin")
        axes[0, col].legend(fontsize=6)
        axes[4, col].legend(fontsize=7)

    fig.suptitle(f"R+ vs R- comparison, {AREA_A} vs {AREA_B} -- mean +/- SEM across sessions per cohort "
                 f"(colors: R+={COHORT_COLOR['R+']}, R-={COHORT_COLOR['R-']}; fixed lambda/condition, "
                 f"PCA-reduced to K=min(90% var, n_trials/5) per area)", y=1.0)
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
                 f"(held-out, PCA-reduced input, dims capped at {MAX_DIMS})")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "example_pair_05_correlation_vs_dimension.png", dpi=110)
    plt.close(fig)
    print("Wrote example_pair_05_correlation_vs_dimension.png")

    # ---- Figure: true CCA vs PCA baseline vs shuffle-null, per condition/cohort ----
    fig, axes = plt.subplots(1, 5, figsize=(26, 5), sharey=True)
    x_labels = ["CCA\n(dim 1)", "PCA\nbaseline", "shuffle\nnull"]
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        c = curves[cond]
        rg = np.array(c["reward_group"])
        ax = axes[col]
        offsets = {"R+": -0.12, "R-": 0.12}
        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            if len(idx) == 0:
                continue
            true_vals = np.abs(np.array([c["cca_r"][i] for i in idx], dtype=float))
            pca_vals = np.abs(np.array([c["pca_baseline_r"][i] for i in idx], dtype=float))
            null_vals = np.abs(np.concatenate([np.asarray(c["shuffle_null"][i]) for i in idx]))
            means = [np.nanmean(true_vals), np.nanmean(pca_vals), np.nanmean(null_vals)]
            sems = [np.nanstd(true_vals, ddof=1) / np.sqrt(max(1, np.sum(~np.isnan(true_vals)))),
                    np.nanstd(pca_vals, ddof=1) / np.sqrt(max(1, np.sum(~np.isnan(pca_vals)))),
                    np.nanstd(null_vals, ddof=1) / np.sqrt(max(1, np.sum(~np.isnan(null_vals))))]
            x = np.arange(3) + offsets[cohort]
            color = COHORT_COLOR[cohort]
            ax.errorbar(x, means, yerr=sems, fmt="o", color=color, markersize=6, capsize=3,
                        label=f"{cohort} (n={len(idx)} sessions)")
        ax.set_xticks(range(3))
        ax.set_xticklabels(x_labels)
        ax.set_title(f"{trial_type} (lick={lick_flag})")
        if col == 0:
            ax.set_ylabel("mean |correlation| (dim 1)\n+/- SEM across sessions (null: across sessions x shuffles)")
        ax.legend(fontsize=7)
        ax.axhline(0, color="black", linewidth=0.6)

    fig.suptitle(f"True CCA vs. PCA-alignment baseline vs. trial-shuffle null, {AREA_A} vs {AREA_B} "
                 f"({N_SHUFFLES} shuffles/session/condition)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "example_pair_06_baselines_comparison.png", dpi=110)
    plt.close(fig)
    print("Wrote example_pair_06_baselines_comparison.png")


if __name__ == "__main__":
    main()
