"""Updated figure set per Axel's 2026-09-01 detailed spec message. Only
figures 1 (raster), 2 (single-session pipeline), 3 (dataset average), 5
(scree plot), 8 (example correlated trials -- merged into 1), and a NEW
window-quantification figure are (re)built here -- "ignore the other
figures for now" per Axel (06/07/11 untouched).

Global styling: bigger fonts, `constrained_layout=True` everywhere to
avoid any overlapping elements, explicit axis labels on every panel.

Interpretive choices made explicit (flag if wrong):
- "remove all smoothing" -> `003_cca_lib.py`'s BIN_WIDTH now equals
  BIN_STRIDE (5ms/5ms, non-overlapping) -- the previous 10ms-width/5ms-
  stride sliding window was itself an implicit smoothing filter across
  neighboring bins.
- Fig 1 "few example trials ... where correlation is found to be high":
  per condition, the SINGLE highest per-trial ca-vs-cb correlation held-
  out test trial (variant A, example session) is shown.
- Fig 8 folded into fig 1: same selected trial, population (canonical-
  variate, dim 1) activity plotted in a second row of subplots below the
  raster, x-axis (time) aligned with the raster above it.
- Fig 2 "reward group comparison directly here": TWO example sessions
  (one R+, one R-) shown as raw single-session traces side by side --
  keeps fig 2's single-session/didactic character while adding the
  cohort dimension (not the dataset-wide mean, that's fig 3).
- Fig 3 "reward group comparison directly here": the full smoke-set R+/R-
  mean+/-SEM comparison (this absorbs what was previously a separate
  fig 4 -- not rebuilding fig 4 separately, per "ignore other figures for
  now").
- Fig 5: MAX_DIMS bumped to 20 for this figure. Variant A, variant B, and
  a per-dimension PCA-alignment baseline (naive PC_i(A)-vs-PC_i(B)
  correlation, no CCA optimization) are all shown per cohort using shade/
  saturation variants of the cohort's base color; each variant's own
  shuffle-null is shown as a shaded SEM band only (no per-dim markers).
- New window-quantification figure: baseline window = [-200ms,
  dead_zone_start or 0ms); sensory window = (dead_zone_end or 0ms,
  +50ms]. Per session, the mean of the trial-by-trial ca-vs-cb
  correlation-across-time curve within that window (for variant A and B
  separately); R+ vs R- compared via the established test pair
  (Mann-Whitney U + Welch t-test), per condition x variant x window.

Caching (Axel, 2026-09-01: "save PCA intermediates and neuron loadings ...
presave so runs are faster if input data or parameters has not changed"):
- `load_sessions_cached`: the loaded spike/trial data (the NWB-read step)
  is cached to `artifacts/cache/sessions_<fp>.pkl`, fingerprinted on the
  session list, area names, quality tiers, day-stage, and the mouse-filter
  reference sheet's actual content (not just code) -- reused across ANY
  script/parameter change downstream of loading.
- The main compute pass and the fig-03 pass are each cached wholesale
  (`artifacts/cache/main_pass_<sessions_fp>_<params_fp>.pkl` /
  `fig03_curves_...pkl`), fingerprinted additionally on every parameter
  that affects their output (binning, dead zone, lambda table, PCA rule,
  dims, shuffle count). This is the practical equivalent of caching PCA
  intermediates: caching the deterministic OUTPUT of a PCA/CCA fit (keyed
  on everything that determines it) gives the same "skip recomputation if
  nothing changed" benefit as caching a fitted PCA object, without the
  fragility of serializing sklearn model objects that are only valid for
  their exact source data anyway.
- In-run (not just cross-run): within one (session, condition), area A's
  PCA reduction (and, for variant B, the nuisance Z's) is IDENTICAL across
  the true fit and all N_SHUFFLES shuffle draws, since shuffling only ever
  reorders area B. `fit_and_eval_prereduced` computes A's/Z's reduction
  ONCE per (session, condition) and reuses it, instead of recomputing it
  redundantly inside all 1+N_SHUFFLES calls (was ~10/11 wasted refits).
"""
from __future__ import annotations

import colorsys
import hashlib
import json
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
core = importlib.import_module("022_pipeline_pca_reduced_shuffle_baseline")

import numpy as np
import pandas as pd
from partial_CCA import PartialCCA
from sklearn.decomposition import PCA as _PCA
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.titlesize": 16, "lines.linewidth": 1.6,
})

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
CACHE_DIR = ARTIFACTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# AREA_A/AREA_B/RUN_DIR are mutated by main() when called with explicit
# arguments (e.g. from a multi-pair driver) -- safe because the only
# consumers of these globals run in the MAIN process (session-list
# building, plotting); the actual ProcessPoolExecutor worker functions
# (process_session_full, core.process_one_session) take spikes_a/spikes_b
# directly and never look up AREA_A/AREA_B themselves, so they're
# unaffected by spawn-based workers re-importing the module with its
# original module-level defaults.
AREA_A, AREA_B = core.AREA_A, core.AREA_B
RUN_DIR = ARTIFACTS_DIR / "runs" / "smoke_full_figure_set"
RUN_DIR.mkdir(parents=True, exist_ok=True)
# Which unit_table column defines "area" -- "area_group_coarse_v3" (10
# coarse areas) by default, or "area_acronym_custom" (69 fine areas) when
# main() is called with area_column="area_acronym_custom" (fine-level
# batch driver). Both columns always exist on any freshly-loaded
# unit_table (000_coverage_lib.load_units computes both), so the raw
# units cache is shared across coarse AND fine runs -- only which column
# build_sessions_for_pair filters on changes.
AREA_COLUMN = "area_group_coarse_v3"
CONDITIONS = core.CONDITIONS
FIXED_LAMBDA_BY_CONDITION = core.FIXED_LAMBDA_BY_CONDITION
COHORT_COLOR = core.COHORT_COLOR
MIN_UNITS = core.MIN_UNITS
MIN_TRIALS = core.MIN_TRIALS
TIER_FN = core.TIER_FN
N_SHUFFLES = core.N_SHUFFLES
PCA_TRIALS_PER_PC = core.PCA_TRIALS_PER_PC
pca_reduce = core.pca_reduce
spikes_of = core.spikes_of
to_full_timeline = core.to_full_timeline

MAX_DIMS_SCREE = 20
UNITS_SHOWN_PER_AREA_RASTER = 50
SENSORY_WINDOW_END_MS = 50.0
BASELINE_WINDOW_START_MS = -200.0
N_WORKERS = 10

# Lag sweep now folded into process_session_full (2026-09-02, Axel: "how to combine the
# main process with the lag sweep?") -- computed for LAG_CONDITION only, reusing the SAME
# resid_a/resid_b/train_idx/test_idx already built for that condition's true (lag=0) fit,
# instead of 023_full_figure_set.py's separate script recomputing tensors/residuals from
# scratch for every session. LAG_MS_RANGE=50 per Axel, 2026-09-02: "only check -50 and
# +50ms". 023 still exists for its OWN unique figures (04/06/08/11) but its fig07 should be
# rebuilt to read this cache instead of resweeping.
LAG_CONDITION = ("whisker_trial", 0)
LAG_MS_RANGE = 50
_lag_max_bins = int(round(LAG_MS_RANGE / (cca_lib.BIN_STRIDE * 1000)))
LAG_BINS_RANGE = np.arange(-_lag_max_bins, _lag_max_bins + 1)


def _stable_hash(obj) -> str:
    """Deterministic short hash of a JSON-serializable structure. Used to
    fingerprint the inputs/parameters behind each cache entry -- NOT a
    cryptographic hash, just a cheap way to detect "has anything that
    matters changed since the last run"."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_raw_units_cached(session_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Loads (or reuses a cached copy of) ALL quality-filtered units'
    spike times + area labels for these sessions -- the expensive NWB read
    + mouse-filter + quality-tier step. Deliberately independent of which
    area PAIR is being studied (no AREA_A/AREA_B/MIN_UNITS in the
    fingerprint): a first version of this cache was keyed per pair, and
    since `spikes_other` captures nearly every unit NOT in that pair's two
    areas, each pair's cache redundantly stored almost the whole brain's
    spike data again (1.4GB for just 6 sessions x 1 pair -- would have
    scaled to tens of GB across 39 pairs). This version caches the raw
    per-session unit table ONCE, shared across every pair; the cheap
    per-pair area-A/B/other split happens in-memory afterward via
    `build_sessions_for_pair`, not cached separately. Returns (unit_table,
    trial_table, fingerprint)."""
    ref_df = pd.read_excel(cov_lib.REF_XLSX, sheet_name="Sheet1")
    ref_fp = _stable_hash(ref_df[["mouse_id", "exclude", "exclude_ephys", "reward_group"]].astype(str).values.tolist())
    quality_tiers = ["good", "mua"]
    fp = _stable_hash({
        "session_ids": sorted(session_ids), "day_to_analyze": "learning",
        "quality_tiers": quality_tiers, "ref_fp": ref_fp,
    })
    cache_file = CACHE_DIR / f"units_{fp}.pkl"
    if cache_file.exists():
        print(f"[cache] units: HIT ({cache_file.name})")
        with open(cache_file, "rb") as f:
            unit_table, trial_table = pickle.load(f)
        return unit_table, trial_table, fp

    print(f"[cache] units: MISS ({cache_file.name}) -- loading {len(session_ids)} sessions from NWB...")
    files = [f"{s}.nwb" for s in session_ids]
    unit_table, trial_table = cov_lib.load_units(files, day_to_analyze="learning", max_workers=16)
    unit_table = cov_lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[TIER_FN(unit_table)]
    with open(cache_file, "wb") as f:
        pickle.dump((unit_table, trial_table), f)
    print(f"[cache] units: wrote {cache_file.name}")
    return unit_table, trial_table, fp


def build_sessions_for_pair(unit_table: pd.DataFrame, trial_table: pd.DataFrame,
                             session_ids: list[str]) -> list[dict]:
    """Cheap in-memory split of the (cached, pair-independent) unit table
    into this pair's area-A/B/other spike lists -- fast relative to
    NWB loading or any fitting, so not cached separately; runs fresh every
    call using whichever AREA_A/AREA_B/MIN_UNITS are currently set."""
    sessions = []
    for session_id in session_ids:
        units_a = unit_table[(unit_table["session_id"] == session_id) & (unit_table[AREA_COLUMN] == AREA_A)]
        units_b = unit_table[(unit_table["session_id"] == session_id) & (unit_table[AREA_COLUMN] == AREA_B)]
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
    return sessions


def load_sessions_cached(session_ids: list[str]) -> tuple[list[dict], str]:
    """Combines the shared raw-units cache with the per-pair split, and
    returns a per-pair fingerprint (units fingerprint + area names +
    MIN_UNITS) for use as the sessions_fp in the main-pass/fig-03 result
    caches -- those DO need to be pair-specific even though the raw
    loading cache isn't."""
    unit_table, trial_table, units_fp = load_raw_units_cached(session_ids)
    sessions = build_sessions_for_pair(unit_table, trial_table, session_ids)
    pair_fp = _stable_hash({"units_fp": units_fp, "area_a": AREA_A, "area_b": AREA_B, "min_units": MIN_UNITS,
                             "area_column": AREA_COLUMN})
    return sessions, pair_fp


def main_pass_params_fingerprint() -> str:
    """Fingerprint of every parameter that affects the main compute
    pass's output (binning, dead zone, PCA rule, lambda table, dims,
    shuffle count) -- if this and the sessions fingerprint both match a
    previous run, the cached result is byte-for-byte what a fresh run
    would produce, so recomputation is skipped entirely."""
    return _stable_hash({
        "bin_width": cca_lib.BIN_WIDTH, "bin_stride": cca_lib.BIN_STRIDE,
        "dead_zone_start": cca_lib.DEAD_ZONE_START_S, "dead_zone_stop": cca_lib.DEAD_ZONE_STOP_S,
        "min_trials": MIN_TRIALS, "lambdas": {f"{k[0]}|{k[1]}": v for k, v in FIXED_LAMBDA_BY_CONDITION.items()},
        "max_dims": MAX_DIMS_SCREE, "n_shuffles": N_SHUFFLES,
        "pca_var_target": core.PCA_VAR_TARGET, "pca_trials_per_pc": PCA_TRIALS_PER_PC,
        "conditions": [f"{c[0]}|{c[1]}" for c in CONDITIONS],
        # 2026-09-02: causal smoothing + folded-in lag sweep both now affect the main
        # pass's output -- must be fingerprinted or a settings change wouldn't invalidate
        # the cache.
        "causal_smooth_effective_window_s": cca_lib.CAUSAL_SMOOTH_EFFECTIVE_WINDOW_S,
        "causal_smooth_truncate_sd": cca_lib.CAUSAL_SMOOTH_TRUNCATE_SD,
        "min_unit_rate_hz": cca_lib.MIN_UNIT_RATE_HZ,
        "lag_condition": f"{LAG_CONDITION[0]}|{LAG_CONDITION[1]}", "lag_ms_range": LAG_MS_RANGE,
    })


def shade(hex_color: str, lighten: float) -> tuple:
    """Lighten a hex color toward white by `lighten` in [0, 1] (HLS lightness)."""
    r, g, b = mcolors.to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l + (1 - l) * lighten
    return colorsys.hls_to_rgb(h, l, s)


def fit_and_eval_general(resid_a, resid_b, resid_z, valid_bins, train_idx, test_idx, lam, n_trials, max_dims):
    """Variant A (resid_z=None) or B (resid_z given). Returns
    (dim_corrs, ca_full_2d, cb_full_2d, corr_t)."""
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

    return _fit_cca_and_score(Xa_train_r, Xa_test_r, Xb_train_r, Xb_test_r, Z_train_r, lam, max_dims,
                               n_test, n_bins_v, valid_bins)


def _fit_cca_and_score(Xa_train_r, Xa_test_r, Xb_train_r, Xb_test_r, Z_train_r, lam, max_dims,
                        n_test, n_bins_v, valid_bins):
    """Shared tail end of both fit_and_eval_general and
    fit_and_eval_prereduced: fit PartialCCA on already-PCA-reduced inputs,
    truncate to max_dims, score. Returns (dim_corrs, ca_full_2d,
    cb_full_2d, corr_t)."""
    model = PartialCCA(regularization=lam).fit(Xa_train_r, Xb_train_r, Z_train_r, verbose=False)
    n_keep = min(model.weights_x_.shape[1], max_dims)
    model.weights_x_ = model.weights_x_[:, :n_keep]
    model.weights_y_ = model.weights_y_[:, :n_keep]
    ca, cb = model.transform(Xa_test_r, Xb_test_r)

    dim_corrs = np.full(max_dims, np.nan)
    for d in range(n_keep):
        cad_mean = np.nanmean(ca[d].reshape(n_test, n_bins_v), axis=0)
        cbd_mean = np.nanmean(cb[d].reshape(n_test, n_bins_v), axis=0)
        if len(cad_mean) > 3:
            dim_corrs[d] = np.corrcoef(cad_mean, cbd_mean)[0, 1]

    ca1 = ca[0].reshape(n_test, n_bins_v)
    cb1 = cb[0].reshape(n_test, n_bins_v)
    ca_full_2d = np.full((n_test, len(valid_bins)), np.nan)
    cb_full_2d = np.full((n_test, len(valid_bins)), np.nan)
    ca_full_2d[:, valid_bins] = ca1
    cb_full_2d[:, valid_bins] = cb1
    corr_t = cca_lib.canonical_correlation_across_time(ca_full_2d, cb_full_2d)
    return dim_corrs, ca_full_2d, cb_full_2d, corr_t


def fit_and_eval_prereduced(Xa_train_r, Xa_test_r, Z_train_r, resid_b, valid_bins, train_idx, test_idx,
                             lam, n_trials, max_dims, n_test, n_bins_v):
    """Same result as fit_and_eval_general, but area A's (and, for variant
    B, the nuisance nuisance Z's) PCA reduction is passed in already
    computed, instead of being refit internally. Only area B is reduced
    here. This matters because within one (session, condition), A's and
    Z's PCA reduction are IDENTICAL across the true fit and every one of
    the N_SHUFFLES shuffle draws -- shuffling only ever reorders B's
    trials -- so computing them once per (session, condition) and reusing
    them across all 1+N_SHUFFLES calls removes what was 10/11 redundant
    refits of the same PCA (Axel, 2026-09-01: "implement the
    optimization")."""
    Xb_train, _ = cca_lib.flatten_trial_bins(resid_b[train_idx][:, valid_bins, :])
    Xb_test = resid_b[test_idx][:, valid_bins, :].reshape(-1, resid_b.shape[-1])
    Xb_train_r, Xb_test_r, _ = pca_reduce(Xb_train, Xb_test, n_trials)
    return _fit_cca_and_score(Xa_train_r, Xa_test_r, Xb_train_r, Xb_test_r, Z_train_r, lam, max_dims,
                               n_test, n_bins_v, valid_bins)


def pca_baseline_dims(Xa_train, Xa_test, Xb_train, Xb_test, n_trials, max_dims, n_test, n_bins_v):
    k_cap = max(1, min(n_trials // PCA_TRIALS_PER_PC, Xa_train.shape[1], Xb_train.shape[1],
                        Xa_train.shape[0] - 1, Xb_train.shape[0] - 1, max_dims))
    pca_a = _PCA(n_components=k_cap).fit(Xa_train)
    pca_b = _PCA(n_components=k_cap).fit(Xb_train)
    scores_a = pca_a.transform(Xa_test).reshape(n_test, n_bins_v, k_cap)
    scores_b = pca_b.transform(Xb_test).reshape(n_test, n_bins_v, k_cap)
    dim_corrs = np.full(max_dims, np.nan)
    for d in range(k_cap):
        m_a = np.nanmean(scores_a[:, :, d], axis=0)
        m_b = np.nanmean(scores_b[:, :, d], axis=0)
        if len(m_a) > 3:
            dim_corrs[d] = np.corrcoef(m_a, m_b)[0, 1]
    return dim_corrs


def rank_correlated_trials(ca_full_2d, cb_full_2d):
    n_test = ca_full_2d.shape[0]
    per_trial_r = np.full(n_test, np.nan)
    for i in range(n_test):
        m = ~(np.isnan(ca_full_2d[i]) | np.isnan(cb_full_2d[i]))
        if m.sum() > 5:
            per_trial_r[i] = np.corrcoef(ca_full_2d[i, m], cb_full_2d[i, m])[0, 1]
    return np.argsort(-np.nan_to_num(per_trial_r, nan=-2)), per_trial_r


def window_masks(t_ms: np.ndarray, is_whisker: bool) -> tuple[np.ndarray, np.ndarray]:
    dz_start_ms = cca_lib.DEAD_ZONE_START_S * 1000
    dz_end_ms = cca_lib.DEAD_ZONE_STOP_S * 1000
    baseline_end = dz_start_ms if is_whisker else 0.0
    sensory_start = dz_end_ms if is_whisker else 0.0
    baseline_mask = (t_ms >= BASELINE_WINDOW_START_MS) & (t_ms < baseline_end)
    sensory_mask = (t_ms > sensory_start) & (t_ms <= SENSORY_WINDOW_END_MS)
    return baseline_mask, sensory_mask


def process_session_full(task: tuple) -> tuple:
    """Runs in a worker process (main compute pass feeding figs 05/10):
    PCA baseline + variant A/B true fit + each variant's own N_SHUFFLES
    shuffle-null, per condition, for one session."""
    (session_id, reward_group, spikes_a, spikes_b, spikes_other, trials_sess,
     bin_starts, valid_bins_by_cond) = task
    out: dict = {}
    for cond in CONDITIONS:
        trial_type, lick_flag = cond
        is_whisker = trial_type == "whisker_trial"
        vb = valid_bins_by_cond[cond]
        lam = FIXED_LAMBDA_BY_CONDITION[cond]
        tt_df = trials_sess[(trials_sess["trial_type"] == trial_type)
                             & (trials_sess["lick_flag"] == lick_flag)].sort_values("start_time")
        starts = tt_df["start_time"].to_numpy(dtype=float)
        if len(starts) < MIN_TRIALS:
            continue
        tensor_a = cca_lib.population_tensor_sliding_smoothed(spikes_a, starts, is_whisker, bin_starts)
        tensor_b = cca_lib.population_tensor_sliding_smoothed(spikes_b, starts, is_whisker, bin_starts)
        # Drop units that don't clear MIN_UNIT_RATE_HZ in THIS condition's PSTH before they
        # ever reach z-scoring (in pca_reduce) -- see cca_lib.rate_filter_mask docstring.
        keep_a = cca_lib.rate_filter_mask(tensor_a)
        keep_b = cca_lib.rate_filter_mask(tensor_b)
        tensor_a, tensor_b = tensor_a[:, :, keep_a], tensor_b[:, :, keep_b]
        n_units_a_eff, n_units_b_eff = int(keep_a.sum()), int(keep_b.sum())
        if n_units_a_eff < core.MIN_UNITS_AFTER_RATE_FILTER or n_units_b_eff < core.MIN_UNITS_AFTER_RATE_FILTER:
            print(f"  {session_id} {cond}: SKIPPED (only {n_units_a_eff}/{n_units_b_eff} units "
                  f"clear {cca_lib.MIN_UNIT_RATE_HZ}Hz)", flush=True)
            continue
        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)
        resid_z = None
        if len(spikes_other) >= 5:
            tensor_z = cca_lib.population_tensor_sliding_smoothed(spikes_other, starts, is_whisker, bin_starts)
            keep_z = cca_lib.rate_filter_mask(tensor_z)
            if keep_z.sum() >= 1:
                resid_z = cca_lib.noise_correlation_residuals(tensor_z[:, :, keep_z])

        seed = zlib.crc32(f"{session_id}_{trial_type}_{lick_flag}_024".encode()) % (2**32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(resid_a.shape[0])
        train_idx, test_idx = perm[:len(perm)//2], perm[len(perm)//2:]

        Xa_train, _ = cca_lib.flatten_trial_bins(resid_a[train_idx][:, vb, :])
        Xb_train, _ = cca_lib.flatten_trial_bins(resid_b[train_idx][:, vb, :])
        Xa_test = resid_a[test_idx][:, vb, :].reshape(-1, n_units_a_eff)
        Xb_test = resid_b[test_idx][:, vb, :].reshape(-1, n_units_b_eff)
        n_test, n_bins_v = len(test_idx), int(vb.sum())
        pca_dims = pca_baseline_dims(Xa_train, Xa_test, Xb_train, Xb_test, len(starts), MAX_DIMS_SCREE,
                                      n_test, n_bins_v)

        # A's own PCA reduction is IDENTICAL across the true fit and every
        # shuffle draw (shuffling only ever reorders B) -- compute once
        # per (session, condition) and reuse via fit_and_eval_prereduced,
        # instead of refitting it inside every one of the 1+N_SHUFFLES
        # calls.
        Xa_train_r, Xa_test_r, _ = pca_reduce(Xa_train, Xa_test, len(starts))

        cond_result: dict = {"pca_baseline_dim_corrs": pca_dims}
        if cond == LAG_CONDITION:
            # Reuses resid_a/resid_b/vb/train_idx/test_idx/lam already built above for
            # this condition's true (lag=0) fit -- no separate tensor/residual rebuild.
            cond_result["lag_curve"] = core.lag_sweep(resid_a, resid_b, vb, train_idx, test_idx, lam,
                                                       len(starts), LAG_BINS_RANGE)
        for variant, z in (("A", None), ("B", resid_z)):
            if variant == "B" and z is None:
                continue
            Z_train_r = None
            if z is not None:
                # Likewise invariant across the shuffle loop: shuffling B
                # never touches the nuisance regressor either.
                Z_train, _ = cca_lib.flatten_trial_bins(z[train_idx][:, vb, :])
                Z_test = z[test_idx][:, vb, :].reshape(-1, z.shape[-1])
                Z_train_r, _, _ = pca_reduce(Z_train, Z_test, len(starts))
            try:
                dim_corrs, _, _, corr_t = fit_and_eval_prereduced(
                    Xa_train_r, Xa_test_r, Z_train_r, resid_b, vb, train_idx, test_idx, lam, len(starts),
                    MAX_DIMS_SCREE, n_test, n_bins_v)
            except Exception as e:
                print(f"  {session_id} {cond} {variant}: failed ({e})", flush=True)
                continue
            null_dims = np.full((N_SHUFFLES, MAX_DIMS_SCREE), np.nan)
            null_ct = []
            for sh in range(N_SHUFFLES):
                shuf_seed = zlib.crc32(f"{session_id}_{trial_type}_{lick_flag}_{variant}_shuf{sh}".encode()) % (2**32)
                shuf_rng = np.random.default_rng(shuf_seed)
                shuf_perm = shuf_rng.permutation(resid_b.shape[0])
                try:
                    nd, _, _, nct = fit_and_eval_prereduced(
                        Xa_train_r, Xa_test_r, Z_train_r, resid_b[shuf_perm], vb, train_idx, test_idx, lam,
                        len(starts), MAX_DIMS_SCREE, n_test, n_bins_v)
                    null_dims[sh] = nd
                    null_ct.append(nct)
                except Exception:
                    continue
            cond_result[variant] = {
                "dim_corrs": dim_corrs, "corr_t": corr_t, "null_dim_corrs": null_dims,
                "null_corr_t": np.array(null_ct) if null_ct else np.full((1, len(vb)), np.nan),
            }
        out[cond] = cond_result
    return session_id, reward_group, out


def main(area_a: str | None = None, area_b: str | None = None, session_ids: list[str] | None = None,
         run_dir: Path | None = None, area_column: str | None = None) -> dict:
    """Runs the full figure set for one area pair. With no arguments,
    behaves exactly as before (Motor-frontal vs Striatum, its dedicated
    session list, the smoke_full_figure_set run dir) -- for standalone
    `python 024_updated_figures.py` use. Called with explicit arguments by
    a multi-pair driver (e.g. `026_all_pairs_driver.py` for the 10 coarse
    areas, `032_all_fine_pairs_driver.py` for the 69 fine areas via
    `area_column="area_acronym_custom"`) to run any pair into its own
    run_dir. Returns a small status dict (pair, n_sessions, run_dir) for
    the driver to log."""
    global AREA_A, AREA_B, RUN_DIR, AREA_COLUMN
    if area_a is not None:
        AREA_A, AREA_B = area_a, area_b
    if area_column is not None:
        AREA_COLUMN = area_column
    if run_dir is not None:
        RUN_DIR = Path(run_dir)
        RUN_DIR.mkdir(parents=True, exist_ok=True)

    if session_ids is None:
        session_ids = pd.read_csv(ARTIFACTS_DIR / "motor_striatum_session_list.csv")["session_id"].tolist()
    smoke_n = os.environ.get("SSL_SMOKE_N_SESSIONS")
    if smoke_n:
        session_ids = session_ids[: int(smoke_n)]
        print(f"[024] [SMOKE TEST] capped to first {len(session_ids)} sessions")
    bin_starts = cca_lib.sliding_window_starts()
    smooth_kernel = cca_lib.causal_half_gaussian_kernel()
    valid_bins_by_cond = {
        cond: ~cca_lib.dead_zone_bin_mask_sliding_causal(bin_starts, cca_lib.BIN_WIDTH,
                                                          cond[0] == "whisker_trial", smooth_kernel)
        for cond in CONDITIONS
    }
    bin_centers = cca_lib.sliding_window_centers(bin_starts)
    t_ms = bin_centers * 1000

    sessions, sessions_fp = load_sessions_cached(session_ids)
    print(f"[024] {AREA_A} vs {AREA_B}: {len(sessions)} sessions usable")
    rplus_sessions = [s for s in sessions if s["reward_group"] == "R+"]
    rminus_sessions = [s for s in sessions if s["reward_group"] == "R-"]
    if not rplus_sessions or not rminus_sessions:
        print(f"[024] SKIPPING {AREA_A} vs {AREA_B}: missing a cohort entirely "
              f"(R+={len(rplus_sessions)}, R-={len(rminus_sessions)}) after session-level filtering "
              f"(pair-level mouse counts can look fine while no single SESSION clears MIN_TRIALS/MIN_UNITS "
              f"for one cohort)")
        return {"area_a": AREA_A, "area_b": AREA_B, "n_sessions": len(sessions), "status": "skipped_missing_cohort"}
    example_rplus, example_rminus = rplus_sessions[0], rminus_sessions[0]
    example = example_rplus
    print(f"[024] example (R+): {example_rplus['session_id']}, example (R-): {example_rminus['session_id']}")

    # ==== Full compute pass: variant A + B, true + shuffle-null, per session/condition ====
    # curves[cond][variant] holds per-session: dim_corrs, corr_t, reward_group
    # Cached as a whole: if the underlying session data (sessions_fp) AND
    # every parameter that affects this computation (main_pass_params_fp)
    # both match a previous run, that run's result is byte-for-byte what
    # recomputing now would produce, so it's reused outright and the
    # entire ProcessPoolExecutor pass is skipped (Axel, 2026-09-01: "save
    # PCA intermediates ... presave so runs are faster if input data or
    # parameters has not changed"). Caching the deterministic OUTPUT this
    # way is equivalent to caching the PCA intermediates themselves (which
    # are a pure function of the same inputs) without the fragility of
    # serializing fitted sklearn objects that are only valid for their
    # exact source data anyway.
    params_fp = main_pass_params_fingerprint()
    main_pass_cache_file = CACHE_DIR / f"main_pass_{sessions_fp}_{params_fp}.pkl"
    if main_pass_cache_file.exists():
        print(f"[cache] main compute pass: HIT ({main_pass_cache_file.name})")
        with open(main_pass_cache_file, "rb") as f:
            data = pickle.load(f)
    else:
        print(f"[cache] main compute pass: MISS ({main_pass_cache_file.name}) -- computing...")
        data = {cond: {"A": {"session_id": [], "reward_group": [], "dim_corrs": [], "corr_t": [],
                              "null_dim_corrs": [], "null_corr_t": []},
                       "B": {"session_id": [], "reward_group": [], "dim_corrs": [], "corr_t": [],
                             "null_dim_corrs": [], "null_corr_t": []},
                       "pca_baseline_dim_corrs": [], "reward_group_pca": [],
                       "lag_curves": [], "lag_session_id": [], "lag_reward_group": []}
                for cond in CONDITIONS}

        tasks = [(s["session_id"], s["reward_group"], s["spikes_a"], s["spikes_b"], s["spikes_other"],
                  s["trials_sess"], bin_starts, valid_bins_by_cond) for s in sessions]
        print(f"[024] main compute pass: {len(tasks)} sessions, {N_WORKERS} parallel workers...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(process_session_full, t): t[0] for t in tasks}
            n_done = 0
            for fut in as_completed(futures):
                session_id = futures[fut]
                try:
                    sid, reward_group, results = fut.result()
                except Exception as e:
                    print(f"{session_id}: WORKER FAILED ({e})", flush=True)
                    continue
                for cond, cond_result in results.items():
                    data[cond]["pca_baseline_dim_corrs"].append(cond_result["pca_baseline_dim_corrs"])
                    data[cond]["reward_group_pca"].append(reward_group)
                    if "lag_curve" in cond_result:
                        data[cond]["lag_curves"].append(cond_result["lag_curve"])
                        data[cond]["lag_session_id"].append(sid)
                        data[cond]["lag_reward_group"].append(reward_group)
                    for variant in ("A", "B"):
                        if variant not in cond_result:
                            continue
                        d = data[cond][variant]
                        r = cond_result[variant]
                        d["session_id"].append(sid); d["reward_group"].append(reward_group)
                        d["dim_corrs"].append(r["dim_corrs"]); d["corr_t"].append(r["corr_t"])
                        d["null_dim_corrs"].append(r["null_dim_corrs"]); d["null_corr_t"].append(r["null_corr_t"])
                n_done += 1
                print(f"  [024 compute {n_done}/{len(tasks)}] {sid} ({reward_group}): done", flush=True)
        with open(main_pass_cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"[cache] main compute pass: wrote {main_pass_cache_file.name}")

    dims_axis = np.arange(1, MAX_DIMS_SCREE + 1)

    # ==== Fig 05: extended 20-dim scree, variant A and B as SEPARATE ROWS, shared y-axis,
    # PCA baseline (variant-independent) repeated on both rows, per-variant shuffle-null band ====
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(7.5 * len(CONDITIONS), 12), sharey=True,
                              constrained_layout=True)
    for col, cond in enumerate(CONDITIONS):
        rg_pca = np.array(data[cond]["reward_group_pca"])
        for row_i, variant in enumerate(("A", "B")):
            ax = axes[row_i, col]
            for cohort in ("R+", "R-"):
                base = COHORT_COLOR[cohort]
                c_pca = shade(base, 0.55)
                d = data[cond][variant]
                rg = np.array(d["reward_group"])
                idx = np.where(rg == cohort)[0]
                if len(idx) > 0:
                    arr = np.abs(np.vstack([d["dim_corrs"][i] for i in idx]))
                    mean = np.nanmean(arr, axis=0)
                    n = np.sum(~np.isnan(arr), axis=0)
                    sem = np.divide(np.nanstd(arr, axis=0, ddof=1), np.sqrt(n), out=np.full_like(mean, np.nan), where=n > 1)
                    ax.errorbar(dims_axis, mean, yerr=sem, color=base, marker="o", markersize=4, linestyle="-",
                                label=f"{cohort} CCA {variant}", zorder=5)
                    # shuffle-null (this variant's own): SEM band only, no markers
                    null_arr = np.abs(np.vstack([d["null_dim_corrs"][i] for i in idx])).reshape(-1, MAX_DIMS_SCREE)
                    null_mean = np.nanmean(null_arr, axis=0)
                    n_null = np.sum(~np.isnan(null_arr), axis=0)
                    null_sem = np.divide(np.nanstd(null_arr, axis=0, ddof=1), np.sqrt(n_null),
                                          out=np.full_like(null_mean, np.nan), where=n_null > 1)
                    null_color = shade(base, 0.75)
                    ax.fill_between(dims_axis, null_mean - null_sem, null_mean + null_sem, color=null_color, alpha=0.5,
                                     label=f"{cohort} shuffle ({variant})", zorder=1)
                    ax.plot(dims_axis, null_mean, color=null_color, linewidth=0.8, zorder=2)

                # common PCA baseline, repeated on both rows for direct comparison
                idx_pca = np.where(rg_pca == cohort)[0]
                if len(idx_pca):
                    arr = np.abs(np.vstack([data[cond]["pca_baseline_dim_corrs"][i] for i in idx_pca]))
                    mean = np.nanmean(arr, axis=0)
                    n = np.sum(~np.isnan(arr), axis=0)
                    sem = np.divide(np.nanstd(arr, axis=0, ddof=1), np.sqrt(n), out=np.full_like(mean, np.nan), where=n > 1)
                    ax.errorbar(dims_axis, mean, yerr=sem, color=c_pca, marker="^", markersize=4, linestyle=":",
                                label=f"{cohort} PCA baseline", zorder=4)
            ax.set_xlabel("canonical dimension")
            ax.set_ylabel(f"variant {variant}\nheld-out |canonical correlation|\n(mean +/- SEM across sessions)")
            ax.set_title(f"{cond[0]} (lick={cond[1]}) -- variant {variant}")
            ax.set_xticks(dims_axis[::2])
            ax.axhline(0, color="black", linewidth=0.6)
            ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(f"Correlation vs. canonical dimension -- variant A (top row) vs. variant B (bottom row), "
                 f"same y-axis, common PCA baseline shown on both, {AREA_A} vs {AREA_B} ({MAX_DIMS_SCREE} dims, "
                 f"n={len(sessions)} sessions)")
    fig.savefig(RUN_DIR / "05_correlation_vs_dimension.png", dpi=130)
    plt.close(fig)
    print("Wrote 05_correlation_vs_dimension.png")

    # ==== Fig 1: raster + population activity for the highest-correlation trial, per condition ====
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(7 * len(CONDITIONS), 11), constrained_layout=True,
                              gridspec_kw={"height_ratios": [2.2, 1]})
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        is_whisker = trial_type == "whisker_trial"
        vb = valid_bins_by_cond[cond]
        lam = FIXED_LAMBDA_BY_CONDITION[cond]
        tt_df = example["trials_sess"][(example["trials_sess"]["trial_type"] == trial_type)
                                        & (example["trials_sess"]["lick_flag"] == lick_flag)].sort_values("start_time")
        starts = tt_df["start_time"].to_numpy(dtype=float)
        ax_r, ax_p = axes[0, col], axes[1, col]
        if len(starts) < MIN_TRIALS:
            ax_r.set_title(f"{trial_type} (lick={lick_flag})\n(insufficient trials)")
            continue
        tensor_a = cca_lib.population_tensor_sliding_smoothed(example["spikes_a"], starts, is_whisker, bin_starts)
        tensor_b = cca_lib.population_tensor_sliding_smoothed(example["spikes_b"], starts, is_whisker, bin_starts)
        keep_a = cca_lib.rate_filter_mask(tensor_a)
        keep_b = cca_lib.rate_filter_mask(tensor_b)
        tensor_a, tensor_b = tensor_a[:, :, keep_a], tensor_b[:, :, keep_b]
        if keep_a.sum() < core.MIN_UNITS_AFTER_RATE_FILTER or keep_b.sum() < core.MIN_UNITS_AFTER_RATE_FILTER:
            ax_r.set_title(f"{trial_type} (lick={lick_flag})\n(insufficient rate-qualifying units)")
            continue
        resid_a = cca_lib.noise_correlation_residuals(tensor_a)
        resid_b = cca_lib.noise_correlation_residuals(tensor_b)
        seed = zlib.crc32(f"{example['session_id']}_{trial_type}_{lick_flag}_024raster".encode()) % (2**32)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(resid_a.shape[0])
        train_idx, test_idx = perm[:len(perm)//2], perm[len(perm)//2:]
        _, ca2d, cb2d, _ = fit_and_eval_general(resid_a, resid_b, None, vb, train_idx, test_idx, lam,
                                                 len(starts), MAX_DIMS_SCREE)
        order, per_trial_r = rank_correlated_trials(ca2d, cb2d)
        best_row = order[0]
        best_trial_idx = test_idx[best_row]
        t0 = starts[best_trial_idx]
        best_r = per_trial_r[best_row]

        n_show_a = min(UNITS_SHOWN_PER_AREA_RASTER, len(example["spikes_a"]))
        n_show_b = min(UNITS_SHOWN_PER_AREA_RASTER, len(example["spikes_b"]))
        rates_a = [len(st[(st > t0 - 0.2) & (st < t0 + 0.5)]) for st in example["spikes_a"]]
        rates_b = [len(st[(st > t0 - 0.2) & (st < t0 + 0.5)]) for st in example["spikes_b"]]
        top_a = np.argsort(rates_a)[::-1][:n_show_a]
        top_b = np.argsort(rates_b)[::-1][:n_show_b]
        y = 0
        for ui in top_a:
            sp = example["spikes_a"][ui]
            sp = sp[(sp > t0 - 0.2) & (sp < t0 + 0.5)] - t0
            ax_r.scatter(sp * 1000, np.full(len(sp), y), s=4, color="tab:blue", marker="|")
            y += 1
        y_div = y
        for ui in top_b:
            sp = example["spikes_b"][ui]
            sp = sp[(sp > t0 - 0.2) & (sp < t0 + 0.5)] - t0
            ax_r.scatter(sp * 1000, np.full(len(sp), y), s=4, color="tab:orange", marker="|")
            y += 1
        ax_r.axhline(y_div - 0.5, color="black", linewidth=0.8)
        if is_whisker:
            for ax in (ax_r, ax_p):
                ax.axvspan(cca_lib.DEAD_ZONE_START_S * 1000, cca_lib.DEAD_ZONE_STOP_S * 1000, color="red", alpha=0.15,
                           label="dead zone (excluded)")
        ax_r.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_r.set_title(f"{trial_type} (lick={lick_flag}), trial r={best_r:.2f}")
        ax_r.set_xlabel("time from start_time (ms)")
        ax_r.set_ylabel(f"unit # ({AREA_A}: blue, top {n_show_a}\n{AREA_B}: orange, top {n_show_b})")
        ax_r.legend(fontsize=8, loc="upper right")

        ax_p.plot(t_ms, ca2d[best_row], color="tab:blue", label=AREA_A)
        ax_p.plot(t_ms, cb2d[best_row], color="tab:orange", label=AREA_B)
        ax_p.axhline(0, color="black", linewidth=0.6)
        ax_p.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax_p.set_xlabel("time from start_time (ms)")
        ax_p.set_ylabel("canonical variate\n(dim 1), this trial")
        ax_p.legend(fontsize=8)
    fig.suptitle(f"Raster (top) + population canonical-variate activity (bottom) for the highest-correlation "
                 f"held-out trial per condition, example session {example['session_id']} ({example['reward_group']})")
    fig.savefig(RUN_DIR / "01_raster.png", dpi=130)
    plt.close(fig)
    print("Wrote 01_raster.png (merged with former fig 8)")

    # ==== Fig 2: two example sessions (one per cohort), shared y-axis per row across conditions ====
    ex_by_cohort = {"R+": example_rplus, "R-": example_rminus}
    rows_data = {"R+": {}, "R-": {}}  # rows_data[cohort][cond] = {psth_a,...}
    for cohort, s in ex_by_cohort.items():
        task = (s["session_id"], s["reward_group"], s["mouse_id"], s["n_units_a"], s["n_units_b"],
                s["spikes_a"], s["spikes_b"], s["trials_sess"], bin_starts, valid_bins_by_cond)
        _, _, _, _, _, results = core.process_one_session(task)
        rows_data[cohort] = results

    fig, axes = plt.subplots(4, len(CONDITIONS), figsize=(7 * len(CONDITIONS), 15), constrained_layout=True)
    row_labels = ["PSTH (Hz)", "mean |residual| (Hz)", "PC1 (held-out)", "CCA r across time"]
    row_keys_ab = [("psth_a", "psth_b"), ("resid_a", "resid_b"), ("pc1_a", "pc1_b"), (None, None)]
    for col, cond in enumerate(CONDITIONS):
        for r_i in range(4):
            ax = axes[r_i, col]
            for cohort, ls in (("R+", "-"), ("R-", "--")):
                row = rows_data[cohort].get(cond)
                if row is None:
                    continue
                color = COHORT_COLOR[cohort]
                if r_i < 3:
                    ka, kb = row_keys_ab[r_i]
                    ax.plot(t_ms, row[ka], color=color, linestyle="-", label=f"{cohort} {AREA_A}")
                    ax.plot(t_ms, row[kb], color=color, linestyle="--", label=f"{cohort} {AREA_B}")
                else:
                    if "corr_t" in row:
                        ax.plot(t_ms, row["corr_t"], color=color, linestyle=ls, label=f"{cohort}")
            if r_i in (2, 3):
                ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xlabel("time from start_time (ms)")
            ax.set_ylabel(row_labels[r_i])
            if r_i == 0:
                ax.set_title(f"{cond[0]} (lick={cond[1]})")
            if col == 0:
                ax.legend(fontsize=7)
    for r_i in range(4):
        ylims = [axes[r_i, c].get_ylim() for c in range(len(CONDITIONS))]
        ymin, ymax = min(y[0] for y in ylims), max(y[1] for y in ylims)
        for c in range(len(CONDITIONS)):
            axes[r_i, c].set_ylim(ymin, ymax)
    fig.suptitle(f"Pipeline across time, two example sessions -- R+: {example_rplus['session_id']}, "
                 f"R-: {example_rminus['session_id']} (y-axis shared across conditions per row)")
    fig.savefig(RUN_DIR / "02_pipeline_across_time.png", dpi=130)
    plt.close(fig)
    print("Wrote 02_pipeline_across_time.png")

    # ==== Fig 3: full smoke-set R+/R- mean+/-SEM, shared y-axis per row across conditions ====
    # Cached the same way as the main pass -- core.process_one_session's
    # output is fully determined by the session data + core's own fixed
    # params (fixed-lambda table, MAX_DIMS=10, bin width/stride, dead
    # zone), so this fingerprint just needs to mark "which script/params
    # generated this" to stay correctly invalidated if core's constants
    # ever change.
    fig3_params_fp = _stable_hash({
        "script": "022_pipeline_pca_reduced_shuffle_baseline", "max_dims": core.MAX_DIMS,
        "bin_width": cca_lib.BIN_WIDTH, "bin_stride": cca_lib.BIN_STRIDE,
        "dead_zone_start": cca_lib.DEAD_ZONE_START_S, "dead_zone_stop": cca_lib.DEAD_ZONE_STOP_S,
        "lambdas": {f"{k[0]}|{k[1]}": v for k, v in FIXED_LAMBDA_BY_CONDITION.items()},
        # 2026-09-02 fix: process_one_session -> fit_and_eval -> pca_reduce depends on
        # core.PCA_VAR_TARGET (and, in principle, PCA_TRIALS_PER_PC), but this fingerprint
        # never included them -- a run that only changed PCA_VAR_TARGET (e.g. 99% -> 90%)
        # would silently keep serving fig-03 curves computed under the STALE target.
        "pca_var_target": core.PCA_VAR_TARGET, "pca_trials_per_pc": core.PCA_TRIALS_PER_PC,
        # 2026-09-02 fix: same gap for the causal-smoothing kernel -- process_one_session
        # now builds tensors via population_tensor_sliding_smoothed, which depends on these.
        "causal_smooth_effective_window_s": cca_lib.CAUSAL_SMOOTH_EFFECTIVE_WINDOW_S,
        "causal_smooth_truncate_sd": cca_lib.CAUSAL_SMOOTH_TRUNCATE_SD,
        "min_unit_rate_hz": cca_lib.MIN_UNIT_RATE_HZ,
    })
    fig3_cache_file = CACHE_DIR / f"fig03_curves_{sessions_fp}_{fig3_params_fp}.pkl"
    if fig3_cache_file.exists():
        print(f"[cache] fig-03 pass: HIT ({fig3_cache_file.name})")
        with open(fig3_cache_file, "rb") as f:
            curves = pickle.load(f)
    else:
        print(f"[cache] fig-03 pass: MISS ({fig3_cache_file.name}) -- computing...")
        curves = {cond: {"session_id": [], "reward_group": [], "psth_a": [], "psth_b": [], "resid_a": [],
                          "resid_b": [], "pc1_a": [], "pc1_b": [], "cca_a": [], "cca_b": [], "corr_t": []}
                  for cond in CONDITIONS}
        fig3_tasks = [(s["session_id"], s["reward_group"], s["mouse_id"], s["n_units_a"], s["n_units_b"],
                       s["spikes_a"], s["spikes_b"], s["trials_sess"], bin_starts, valid_bins_by_cond)
                      for s in sessions]
        print(f"[024] fig-03 pass: {len(fig3_tasks)} sessions, {N_WORKERS} parallel workers...")
        with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = {ex.submit(core.process_one_session, t): t[0] for t in fig3_tasks}
            n_done = 0
            for fut in as_completed(futures):
                session_id = futures[fut]
                try:
                    sid, reward_group, _, _, _, results = fut.result()
                except Exception as e:
                    print(f"{session_id}: WORKER FAILED ({e})", flush=True)
                    continue
                for cond, row in results.items():
                    c = curves[cond]
                    c["session_id"].append(sid); c["reward_group"].append(reward_group)
                    for key in ("psth_a", "psth_b", "resid_a", "resid_b", "pc1_a", "pc1_b", "cca_a", "cca_b", "corr_t"):
                        c[key].append(row.get(key))
                n_done += 1
                print(f"  [024 fig03 {n_done}/{len(fig3_tasks)}] {sid} ({reward_group}): done", flush=True)
        with open(fig3_cache_file, "wb") as f:
            pickle.dump(curves, f)
        print(f"[cache] fig-03 pass: wrote {fig3_cache_file.name}")

    fig, axes = plt.subplots(5, len(CONDITIONS), figsize=(7 * len(CONDITIONS), 19), constrained_layout=True)
    row_labels = ["PSTH (Hz)", "mean |residual| (Hz)", "PC1 (held-out, sign-aligned)",
                  "CCA variate (dim1, sign-aligned)", "CCA r across time"]
    for col, cond in enumerate(CONDITIONS):
        c = curves[cond]
        rg = np.array(c["reward_group"])
        for cohort in ("R+", "R-"):
            idx = np.where(rg == cohort)[0]
            if len(idx) == 0:
                continue
            color = COHORT_COLOR[cohort]
            for r_i, (ka, kb) in enumerate([("psth_a", "psth_b"), ("resid_a", "resid_b"),
                                             ("pc1_a", "pc1_b"), ("cca_a", "cca_b")]):
                ax = axes[r_i, col]
                va = [c[ka][i] for i in idx]; vb = [c[kb][i] for i in idx]
                if r_i >= 2:
                    va = core.sign_align(va); vb = core.sign_align(vb)
                m_a, s_a = core.sem_across(va); m_b, s_b = core.sem_across(vb)
                ax.plot(t_ms, m_a, color=color, linestyle="-", label=f"{cohort} {AREA_A} (n={len(idx)})")
                ax.fill_between(t_ms, m_a - s_a, m_a + s_a, color=color, alpha=0.15)
                ax.plot(t_ms, m_b, color=color, linestyle="--", label=f"{cohort} {AREA_B}")
                ax.fill_between(t_ms, m_b - s_b, m_b + s_b, color=color, alpha=0.15)
                if r_i >= 2:
                    ax.axhline(0, color="black", linewidth=0.6)
            ax = axes[4, col]
            m_t, s_t = core.sem_across([c["corr_t"][i] for i in idx])
            ax.plot(t_ms, m_t, color=color, label=f"{cohort} (n={len(idx)})")
            ax.fill_between(t_ms, m_t - s_t, m_t + s_t, color=color, alpha=0.2)
            ax.axhline(0, color="black", linewidth=0.6)
        for r_i in range(5):
            axes[r_i, col].set_xlabel("time from start_time (ms)")
            axes[r_i, col].set_ylabel(row_labels[r_i])
            if r_i == 0:
                axes[r_i, col].set_title(f"{cond[0]} (lick={cond[1]})")
            if col == 0:
                axes[r_i, col].legend(fontsize=7)
    for r_i in range(5):
        ylims = [axes[r_i, c].get_ylim() for c in range(len(CONDITIONS))]
        ymin, ymax = min(y[0] for y in ylims), max(y[1] for y in ylims)
        for c in range(len(CONDITIONS)):
            axes[r_i, c].set_ylim(ymin, ymax)
    fig.suptitle(f"R+ vs R- comparison, {AREA_A} vs {AREA_B} -- mean +/- SEM across {len(sessions)} sessions "
                 f"(y-axis shared across conditions per row)")
    fig.savefig(RUN_DIR / "03_pooled_average.png", dpi=130)
    plt.close(fig)
    print("Wrote 03_pooled_average.png (now cohort-split, absorbs former fig 4)")

    # ==== New figure: baseline/sensory window quantification, A vs B, R+ vs R-, with stats ====
    fig, axes = plt.subplots(2, len(CONDITIONS), figsize=(6.5 * len(CONDITIONS), 11), constrained_layout=True)
    stats_rows = []
    for col, cond in enumerate(CONDITIONS):
        trial_type, lick_flag = cond
        is_whisker = trial_type == "whisker_trial"
        baseline_mask, sensory_mask = window_masks(t_ms, is_whisker)
        for w_i, (win_name, mask) in enumerate([("baseline", baseline_mask), ("sensory", sensory_mask)]):
            ax = axes[w_i, col]
            offsets = {"R+": -0.15, "R-": 0.15}
            annotations = []  # (x_position, p_mw) collected first, drawn after ylim is final
            for variant_i, variant in enumerate(("A", "B")):
                d = data[cond][variant]
                rg = np.array(d["reward_group"])
                for cohort in ("R+", "R-"):
                    idx = np.where(rg == cohort)[0]
                    if len(idx) == 0:
                        continue
                    vals = np.array([np.nanmean(d["corr_t"][i][mask]) for i in idx])
                    x = variant_i + offsets[cohort]
                    ax.errorbar([x], [np.nanmean(vals)],
                                yerr=[np.nanstd(vals, ddof=1) / np.sqrt(max(1, np.sum(~np.isnan(vals))))],
                                fmt="o", color=COHORT_COLOR[cohort], markersize=7, capsize=4,
                                label=f"{cohort}" if variant_i == 0 else None)

                idx_p = np.where(rg == "R+")[0]; idx_m = np.where(rg == "R-")[0]
                if len(idx_p) > 1 and len(idx_m) > 1:
                    vp = np.array([np.nanmean(d["corr_t"][i][mask]) for i in idx_p])
                    vm = np.array([np.nanmean(d["corr_t"][i][mask]) for i in idx_m])
                    vp, vm = vp[~np.isnan(vp)], vm[~np.isnan(vm)]
                    if len(vp) > 1 and len(vm) > 1:
                        u_stat, p_mw = stats.mannwhitneyu(vp, vm, alternative="two-sided")
                        t_stat, p_welch = stats.ttest_ind(vp, vm, equal_var=False)
                        stats_rows.append({"condition": f"{trial_type}(lick={lick_flag})", "window": win_name,
                                            "variant": variant, "n_Rplus": len(vp), "n_Rminus": len(vm),
                                            "mean_Rplus": np.mean(vp), "mean_Rminus": np.mean(vm),
                                            "mannwhitney_p": p_mw, "welch_p": p_welch})
                        annotations.append((variant_i, p_mw))
            ax.set_xticks([0, 1]); ax.set_xticklabels(["variant A", "variant B"])
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_ylabel(f"mean CCA correlation-across-time\nin {win_name} window")
            ax.set_title(f"{trial_type} (lick={lick_flag}) -- {win_name} window")
            if col == 0:
                ax.legend(fontsize=8)
            y_top = ax.get_ylim()[1]
            for x_pos, p_mw in annotations:
                sig = "*" if p_mw < 0.05 else ""
                ax.text(x_pos, y_top, f"MW p={p_mw:.3f}{sig}", ha="center", va="bottom", fontsize=7)
            if annotations:
                ax.set_ylim(ax.get_ylim()[0], y_top + 0.12 * (y_top - ax.get_ylim()[0]))
    fig.suptitle(f"CCA-variate alignment strength, baseline [{BASELINE_WINDOW_START_MS:.0f}ms, dead-zone/0ms) vs. "
                 f"sensory (dead-zone/0ms, {SENSORY_WINDOW_END_MS:.0f}ms] windows, R+ vs R-, variant A/B "
                 f"({AREA_A} vs {AREA_B})")
    fig.savefig(RUN_DIR / "10_window_quantification.png", dpi=130)
    plt.close(fig)
    print("Wrote 10_window_quantification.png")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(RUN_DIR / "10_window_quantification_stats.csv", index=False)
    print("Wrote 10_window_quantification_stats.csv")
    print(stats_df.to_string(index=False))

    # ==== Fig 07: lag sweep, folded in from process_session_full (2026-09-02) -- see
    # "how to combine the main process with the lag sweep?" -- reuses the SAME per-session
    # residuals as the main pass instead of a separate script's own recompute. ====
    lag_data = data[LAG_CONDITION]
    lag_curves, lag_sids, lag_rgs = lag_data["lag_curves"], lag_data["lag_session_id"], lag_data["lag_reward_group"]
    if lag_curves:
        lag_ms_axis = LAG_BINS_RANGE * cca_lib.BIN_STRIDE * 1000
        peak_lags = [lag_ms_axis[np.nanargmax(np.abs(rs))] for rs in lag_curves if np.any(~np.isnan(rs))]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for rs, sid, rg in zip(lag_curves, lag_sids, lag_rgs):
            axes[0].plot(lag_ms_axis, np.abs(rs), color=COHORT_COLOR[rg], alpha=0.5, linewidth=1.0)
        axes[0].axvline(0, color="black", linewidth=0.6)
        axes[0].set_xlabel("lag (ms): positive = area B activity shifted earlier relative to A")
        axes[0].set_ylabel("held-out |dim-1 correlation|")
        axes[0].set_title(f"Lag sweep, {LAG_CONDITION[0]} (lick={LAG_CONDITION[1]}), n={len(lag_curves)} sessions")
        axes[1].hist(peak_lags, bins=15, color="tab:gray")
        axes[1].axvline(0, color="black", linewidth=0.6)
        axes[1].set_xlabel("peak-|r| lag (ms)")
        axes[1].set_ylabel("# sessions")
        axes[1].set_title(f"Peak-lag distribution (n={len(peak_lags)} sessions)")
        fig.suptitle(f"Lagged cross-area correlation, {AREA_A} vs {AREA_B}")
        fig.savefig(RUN_DIR / "07_lag_sweep.png", dpi=110)
        plt.close(fig)
        print("Wrote 07_lag_sweep.png")

    return {"area_a": AREA_A, "area_b": AREA_B, "n_sessions": len(sessions), "run_dir": str(RUN_DIR),
            "status": "ok"}


if __name__ == "__main__":
    main()
