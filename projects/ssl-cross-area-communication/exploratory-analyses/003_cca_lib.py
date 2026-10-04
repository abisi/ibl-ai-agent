"""Core library for the cross-area pCCA pipeline: dead-zone-aware time-binned
rate extraction, noise-correlation residualization, and CCA/partial-CCA
fitting.

Dead zone: as of 2026-08-31 this project uses **-10ms/+5ms** around
whisker-trial `start_time` (Axel Bisi, explicit instruction, confirmed after
flagging that `ssl_artifact_dead_zone.md` documents a 2026-08-15 correction
to the narrower -1ms/+4ms). This is NOT a regression: that same skill file
states a wider dead zone is always safe/conservative ("existing analyses
that used the previous, wider -10ms/+5ms window... are not biased... simply
more conservative than necessary") -- so -10/+5 is a valid, explicitly-
sanctioned superset of the -1/+4ms minimum, just scoped to this project
rather than the shared `scripts/ssl_bwm_windows.py` (which other SSL
projects still use at -1/+4ms and is left untouched here). See
`clip_window_for_whisker_wide` below, a local reimplementation of that
script's clipping logic at the wider bounds.

As of 2026-08-31, the CCA/pCCA engine itself is `partial_CCA.PartialCCA`
(Gonzalez, Buzsaki/Chen labs, MIT-licensed, pip-installable from TestPyPI --
installed into the `bwa` conda env from a locally downloaded wheel; see
change-log.md) -- a closed-form eigendecomposition CCA with explicit ridge
regularization, matching the classical (Hotelling) CCA solution, in place of
the earlier `sklearn.cross_decomposition.CCA` (an iterative NIPALS/PLS-family
approximation that is not guaranteed to match the closed-form solution,
especially past the first component). Partial-CCA (regressing X/Y on a
nuisance Z before CCA) is now handled internally by `PartialCCA.fit(X, Y, Z)`
rather than the earlier hand-rolled `regress_out` + separate CCA fit --
`regress_out`/`cv_canonical_correlation`/`project_tensor_multi` below are
kept only because earlier checkpoint scripts (004-007) still import them;
new code should use `fit_nuisance_pca` (still used, for variant B's "other
neurons" Z) plus `partial_CCA.PartialCCA` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from partial_CCA import PartialCCA

WINDOW = (-0.200, 0.500)   # seconds relative to trial start_time
BIN_WIDTH = 0.010          # seconds, bin width (as of 2026-09-02, Axel: "use a 10ms time bin,
                           # with 5ms stride" -- reverted from 1ms/1ms after 1ms bins + z-scoring
                           # proved to badly amplify single-spike outliers in sparse units;
                           # overlapping 10ms/5ms windows now supplement the explicit causal
                           # smoothing kernel rather than being the sole smoothing mechanism
                           # (the 2026-09-01 "remove all smoothing" non-overlap convention is
                           # superseded by this instruction)
BIN_STRIDE = 0.005         # seconds, stride between consecutive window starts
LICK_BIN_WIDTH = 0.050     # seconds, per instruction (variant C)

DEAD_ZONE_START_S = -0.010  # start_time - 10ms (project-local, see module docstring)
DEAD_ZONE_STOP_S = 0.005    # start_time + 5ms


def clip_window_for_whisker_wide(window: tuple[float, float], is_whisker: bool) -> tuple[float, float] | None:
    """Local reimplementation of `scripts/ssl_bwm_windows.py`'s clipping
    logic at this project's -10ms/+5ms dead zone (see module docstring)."""
    start, end = window
    if not is_whisker:
        return (start, end)
    dz0, dz1 = DEAD_ZONE_START_S, DEAD_ZONE_STOP_S
    if start >= dz0 and end <= dz1:
        return None
    if start < dz0 and end > dz1:
        raise ValueError(f"window {window} straddles the entire dead zone; split it explicitly")
    new_start, new_end = start, end
    if start < dz0 < end <= dz1:
        new_end = dz0
    if dz0 <= start < dz1 < end:
        new_start = dz1
    if new_start >= new_end:
        return None
    return (new_start, new_end)


clip_window_for_whisker = clip_window_for_whisker_wide  # internal alias, see below


def time_bin_edges(window: tuple[float, float] = WINDOW, bin_width: float = BIN_WIDTH) -> np.ndarray:
    """Non-overlapping tiling (scripts 004-017's convention -- kept as-is,
    untouched, for backward compat). New code (018+) uses the sliding-window
    functions below instead (5ms width / 2ms stride, per Axel's 2026-08-31
    instruction to update the binning before any further large run)."""
    n_bins = int(round((window[1] - window[0]) / bin_width))
    return window[0] + np.arange(n_bins + 1) * bin_width


def sliding_window_starts(window: tuple[float, float] = WINDOW, bin_width: float = BIN_WIDTH,
                           stride: float = BIN_STRIDE) -> np.ndarray:
    """Start times for a SLIDING window scheme: each window spans
    [start, start+bin_width), consecutive starts spaced by `stride`.
    Windows OVERLAP when stride < bin_width (the 2026-08-31 default: 5ms
    width, 2ms stride -- 3ms of overlap between neighbors). Note: because
    neighboring (trial, bin) samples now share most of their spikes, they
    are far from independent -- this does not affect the shuffle test's
    validity (it permutes whole trials, not bins, so trial-level
    exchangeability is untouched) but does mean the raw sample count fed to
    CCA/PCA overstates how much independent information it represents;
    treat within-trial adjacent-bin structure as smoothed, not as that many
    independent observations."""
    w0, w1 = window
    n_bins = int(round((w1 - bin_width - w0) / stride)) + 1
    return w0 + np.arange(n_bins) * stride


def sliding_window_centers(bin_starts: np.ndarray, bin_width: float = BIN_WIDTH) -> np.ndarray:
    return bin_starts + bin_width / 2


def _clipped_windows_for_bins(bin_starts: np.ndarray, bin_width: float, is_whisker: bool
                               ) -> list[tuple[float, float] | None]:
    """Dead-zone-clipped (w0, w1) window per bin (None if the whole bin
    falls inside the dead zone). Depends only on (bin_starts, bin_width,
    is_whisker) -- NOT on the unit -- so callers should compute this ONCE
    per (session, condition) and reuse it across all units, rather than
    recomputing per unit (the previous per-unit-loop version silently
    redid this identical clip-window logic once per unit, e.g. ~800x
    redundant for a large area)."""
    return [clip_window_for_whisker((s, s + bin_width), is_whisker=is_whisker) for s in bin_starts]


def binned_rates_for_trials_sliding(
    spike_times_sorted: np.ndarray,
    trial_start_times: np.ndarray,
    is_whisker: bool,
    bin_starts: np.ndarray,
    bin_width: float = BIN_WIDTH,
    clipped_windows: list[tuple[float, float] | None] | None = None,
) -> np.ndarray:
    """(n_trials, n_bins) firing-rate matrix for one unit, over the sliding
    windows in `bin_starts` (each [start, start+bin_width)), dead-zone-
    clipped per window for whisker trials. Vectorized across (trials, bins)
    via a single batched `searchsorted` call each for window starts/stops,
    instead of a Python loop over bins (each iteration previously issuing
    its own pair of searchsorted calls). Pass `clipped_windows` (from
    `_clipped_windows_for_bins`) to skip recomputing the per-bin clip logic
    when calling this once per unit in a population loop."""
    n_trials = len(trial_start_times)
    n_bins = len(bin_starts)
    if clipped_windows is None:
        clipped_windows = _clipped_windows_for_bins(bin_starts, bin_width, is_whisker)
    rates = np.full((n_trials, n_bins), np.nan)
    valid_b = [b for b, c in enumerate(clipped_windows) if c is not None]
    if not valid_b:
        return rates
    w0 = np.array([clipped_windows[b][0] for b in valid_b])
    w1 = np.array([clipped_windows[b][1] for b in valid_b])
    starts_2d = trial_start_times[:, None] + w0[None, :]  # (n_trials, n_valid_bins)
    stops_2d = trial_start_times[:, None] + w1[None, :]
    lo = np.searchsorted(spike_times_sorted, starts_2d, side="left")
    hi = np.searchsorted(spike_times_sorted, stops_2d, side="left")
    rates[:, valid_b] = (hi - lo) / (w1 - w0)[None, :]
    return rates


def population_tensor_sliding(unit_spike_times: list[np.ndarray], trial_start_times: np.ndarray,
                               is_whisker: bool, bin_starts: np.ndarray, bin_width: float = BIN_WIDTH) -> np.ndarray:
    """(n_trials, n_bins, n_units) rate tensor for a population, sliding
    windows. Dead-zone clipping is computed once and shared across units."""
    clipped_windows = _clipped_windows_for_bins(bin_starts, bin_width, is_whisker)
    mats = [binned_rates_for_trials_sliding(st, trial_start_times, is_whisker, bin_starts, bin_width, clipped_windows)
            for st in unit_spike_times]
    return np.stack(mats, axis=-1)


def dead_zone_bin_mask_sliding(bin_starts: np.ndarray, bin_width: float, is_whisker: bool) -> np.ndarray:
    """Boolean array, True for any sliding window touched by the dead zone."""
    n_bins = len(bin_starts)
    mask = np.zeros(n_bins, dtype=bool)
    if not is_whisker:
        return mask
    for b, s in enumerate(bin_starts):
        w = (s, s + bin_width)
        if clip_window_for_whisker(w, is_whisker=True) != w:
            mask[b] = True
    return mask


# ==== Causal half-Gaussian smoothing (2026-09-02, Axel: "I think in the next run I may want
# to use a causal half-gaussian kernel window of 10ms or so for smoothing" + "Implement the
# kernel so that the dead zone does not contaminate data") ====
# Not wired into the main pipeline yet -- these are the reusable primitives (kernel,
# convolution, widened dead-zone mask) for whenever the next run turns this on. NOT the
# default: BIN_WIDTH/BIN_STRIDE-equal non-overlapping bins (2026-09-01 "remove all
# smoothing") remain the current pipeline's behavior until a caller opts into this.
#
# "Effective window" of ~10ms is interpreted as the kernel's truncation point (3 SDs,
# ~99.7% of the half-Gaussian's mass) rather than literally the SD itself -- flagging this
# as the judgment call it is. SD = 10ms / 3 =~ 3.33ms.
CAUSAL_SMOOTH_EFFECTIVE_WINDOW_S = 0.010
CAUSAL_SMOOTH_TRUNCATE_SD = 3.0
CAUSAL_SMOOTH_SD_S = CAUSAL_SMOOTH_EFFECTIVE_WINDOW_S / CAUSAL_SMOOTH_TRUNCATE_SD


def causal_half_gaussian_kernel(bin_width: float = BIN_WIDTH, sd_s: float = CAUSAL_SMOOTH_SD_S,
                                 truncate_sd: float = CAUSAL_SMOOTH_TRUNCATE_SD) -> np.ndarray:
    """Causal (past-and-present-only) half-Gaussian smoothing kernel, in bin
    units. kernel[0] weights the CURRENT bin, kernel[k] the bin k steps in
    the past. Truncated at `truncate_sd` SDs and renormalized to sum to 1."""
    n_taps = int(np.ceil(truncate_sd * sd_s / bin_width)) + 1
    k = np.arange(n_taps)
    w = np.exp(-0.5 * (k * bin_width / sd_s) ** 2)
    return w / w.sum()


def apply_causal_smoothing(tensor: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal half-Gaussian smoothing along the bin axis (axis=1) of a
    (n_trials, n_bins, n_units) rate tensor (as returned by
    `population_tensor_sliding`, i.e. dead-zone bins are already NaN).

    Two edge cases, handled deliberately differently:
    - WINDOW edge (near the start of the -200..+500ms analysis window, where
      fewer than len(kernel) past bins exist yet): not missing/contaminated
      data, just structurally unavailable -- the kernel is truncated to the
      taps that exist and RENORMALIZED over them (standard causal-filter
      edge handling).
    - DEAD-ZONE bin (a real bin, but NaN because it's dead-zone-contaminated):
      NOT renormalized over -- the NaN propagates through the weighted sum via
      plain (non-nan-aware) arithmetic, poisoning every output bin whose
      window touches it. Pair with `dead_zone_bin_mask_sliding_causal` so
      those poisoned output bins are ALSO excluded from valid_bins downstream
      (belt-and-suspenders: NaN propagation alone already keeps them out of
      anything that consults valid_bins or drops NaN rows, e.g.
      `flatten_trial_bins`) -- this is what "does not contaminate data" means
      here: dead-zone information never leaks into a bin that looks clean."""
    n_trials, n_bins, n_units = tensor.shape
    n_taps = len(kernel)
    pad = n_taps - 1
    padded = np.concatenate([np.zeros((n_trials, pad, n_units)), tensor], axis=1)
    valid_pad = np.concatenate([np.zeros((n_trials, pad, n_units)),
                                 np.ones((n_trials, n_bins, n_units))], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=n_taps, axis=1)
    valid_windows = np.lib.stride_tricks.sliding_window_view(valid_pad, window_shape=n_taps, axis=1)
    w_rev = kernel[::-1]  # windows' last axis runs oldest->newest; align with that order
    numer = np.tensordot(windows, w_rev, axes=([-1], [0]))
    denom = np.tensordot(valid_windows, w_rev, axes=([-1], [0]))  # window-edge renorm only
    return numer / denom


def dead_zone_bin_mask_sliding_causal(bin_starts: np.ndarray, bin_width: float, is_whisker: bool,
                                       kernel: np.ndarray) -> np.ndarray:
    """Like `dead_zone_bin_mask_sliding`, but widened FORWARD by the causal
    kernel's reach: any bin whose smoothing window (itself plus up to
    len(kernel)-1 bins in the past) touches a dead-zone-invalid raw bin is
    ALSO marked invalid, since `apply_causal_smoothing` correctly propagates
    that contamination as NaN into every such output bin -- this mask keeps
    valid_bins consistent with what's actually NaN after smoothing."""
    base = dead_zone_bin_mask_sliding(bin_starts, bin_width, is_whisker)
    if not base.any():
        return base
    widened = base.copy()
    dead_idx = np.flatnonzero(base)
    for k in range(1, len(kernel)):
        shifted = dead_idx + k
        widened[shifted[shifted < len(base)]] = True
    return widened


def population_tensor_sliding_smoothed(unit_spike_times: list[np.ndarray], trial_start_times: np.ndarray,
                                        is_whisker: bool, bin_starts: np.ndarray, bin_width: float = BIN_WIDTH,
                                        kernel: np.ndarray | None = None) -> np.ndarray:
    """`population_tensor_sliding` followed by causal half-Gaussian
    smoothing. Use `dead_zone_bin_mask_sliding_causal` (not the plain,
    unwidened `dead_zone_bin_mask_sliding`) for valid_bins downstream of
    this -- see `apply_causal_smoothing`'s docstring for why."""
    if kernel is None:
        kernel = causal_half_gaussian_kernel(bin_width)
    tensor = population_tensor_sliding(unit_spike_times, trial_start_times, is_whisker, bin_starts, bin_width)
    return apply_causal_smoothing(tensor, kernel)


def binned_rates_for_trials(
    spike_times_sorted: np.ndarray,
    trial_start_times: np.ndarray,
    is_whisker: bool,
    bin_edges: np.ndarray,
) -> np.ndarray:
    """(n_trials, n_bins) firing-rate matrix for one unit, dead-zone-clipped
    per bin for whisker trials (only the bin straddling t=0 is affected,
    since the dead zone is 5ms and bins are BIN_WIDTH wide). NaN in any bin
    fully inside the dead zone."""
    n_trials = len(trial_start_times)
    n_bins = len(bin_edges) - 1
    rates = np.full((n_trials, n_bins), np.nan)
    for b in range(n_bins):
        w = (bin_edges[b], bin_edges[b + 1])
        clipped = clip_window_for_whisker(w, is_whisker=is_whisker)
        if clipped is None:
            continue
        w0, w1 = clipped
        lo = np.searchsorted(spike_times_sorted, trial_start_times + w0, side="left")
        hi = np.searchsorted(spike_times_sorted, trial_start_times + w1, side="left")
        rates[:, b] = (hi - lo) / (w1 - w0)
    return rates


def population_tensor(unit_spike_times: list[np.ndarray], trial_start_times: np.ndarray,
                       is_whisker: bool, bin_edges: np.ndarray) -> np.ndarray:
    """(n_trials, n_bins, n_units) rate tensor for a population."""
    mats = [binned_rates_for_trials(st, trial_start_times, is_whisker, bin_edges) for st in unit_spike_times]
    return np.stack(mats, axis=-1)


def noise_correlation_residuals(tensor: np.ndarray) -> np.ndarray:
    """Subtract the trial-averaged PSTH (mean over trials, per bin, per
    unit) -- no baseline correction on top of this, per explicit instruction.
    `tensor`: (n_trials, n_bins, n_units)."""
    psth = np.nanmean(tensor, axis=0, keepdims=True)  # (1, n_bins, n_units)
    return tensor - psth


MIN_UNIT_RATE_HZ = 0.5  # Axel, 2026-09-02: "Keep neurons that fire above 0.5 Hz in the
                        # PSTH computed data" -- fixes the root cause of the flat+spiky
                        # PCA/CCA traces: an isolated spike on an otherwise-near-silent unit
                        # gets a tiny per-unit std, so z-scoring (in pca_reduce) turned its
                        # smoothed echo into a ~100 SD outlier that dominated PCA. Dropping
                        # units that don't clear a minimum PSTH rate removes them before
                        # z-scoring ever sees them, rather than rescaling around them.


def rate_filter_mask(tensor: np.ndarray, min_hz: float = MIN_UNIT_RATE_HZ) -> np.ndarray:
    """Boolean mask (n_units,), True for units whose PSTH -- the trial-
    averaged rate, itself then averaged across time bins -- exceeds
    `min_hz`. Apply to the (SMOOTHED) rate tensor, per (session, area,
    condition), BEFORE `noise_correlation_residuals`/z-scoring/PCA, so
    sparse units never reach the step where they'd become z-score
    outliers. Condition-specific: a unit can pass for one trial type and
    fail for another."""
    psth = np.nanmean(tensor, axis=0)       # (n_bins, n_units)
    mean_rate = np.nanmean(psth, axis=0)    # (n_units,)
    return mean_rate > min_hz


def flatten_trial_bins(residual_tensor: np.ndarray) -> np.ndarray:
    """(n_trials, n_bins, n_units) -> (n_trials * n_bins, n_units), dropping
    (trial,bin) samples with any NaN unit (dead-zone-affected bins)."""
    n_trials, n_bins, n_units = residual_tensor.shape
    flat = residual_tensor.reshape(n_trials * n_bins, n_units)
    valid = ~np.isnan(flat).any(axis=1)
    return flat[valid], valid


def lick_rate_regressor(lick_times: np.ndarray, trial_start_times: np.ndarray,
                         bin_edges: np.ndarray, lick_bin_width: float = LICK_BIN_WIDTH) -> np.ndarray:
    """(n_trials, n_bins) piezo-lick-rate regressor: licks binned at
    `lick_bin_width` (50ms, per instruction), then each analysis bin
    inherits the rate of the coarse lick-bin it falls inside (no dead-zone
    clipping -- licks aren't a spike-artifact-affected signal)."""
    lo_edge = min(bin_edges[0], -lick_bin_width)
    hi_edge = max(bin_edges[-1], lick_bin_width) + lick_bin_width
    coarse_edges = np.arange(lo_edge, hi_edge + lick_bin_width, lick_bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    coarse_idx_for_bin = np.clip(np.searchsorted(coarse_edges, bin_centers, side="right") - 1, 0, len(coarse_edges) - 2)

    n_trials = len(trial_start_times)
    n_coarse = len(coarse_edges) - 1
    coarse_rate = np.zeros((n_trials, n_coarse))
    for i, t0 in enumerate(trial_start_times):
        edges = t0 + coarse_edges
        lo = np.searchsorted(lick_times, edges[:-1], side="left")
        hi = np.searchsorted(lick_times, edges[1:], side="left")
        coarse_rate[i] = (hi - lo) / lick_bin_width
    return coarse_rate[:, coarse_idx_for_bin]


def dead_zone_bin_mask(bin_edges: np.ndarray, is_whisker: bool) -> np.ndarray:
    """Boolean array, length n_bins, True for any bin touched by the mandatory
    whisker-trial artifact dead zone (-1/+4ms around start_time) -- always
    excluded from the lagged/shuffle trial-by-trial analyses, not just
    NaN-masked. Always all-False for auditory trials (dead zone is
    whisker-only)."""
    n_bins = len(bin_edges) - 1
    mask = np.zeros(n_bins, dtype=bool)
    if not is_whisker:
        return mask
    for b in range(n_bins):
        w = (bin_edges[b], bin_edges[b + 1])
        if clip_window_for_whisker(w, is_whisker=True) != w:
            mask[b] = True
    return mask


def project_tensor_multi(tensor: np.ndarray, cca: CCA, side: str, n_dims: int) -> np.ndarray:
    """(n_trials, n_bins, n_units) -> (n_trials, n_bins, n_dims) canonical-
    variate scores for dims 0..n_dims-1, via the fitted CCA's own transform
    (correct train-set centering/scaling applied to held-out data). The
    side not being projected is fed zeros (transform computes each side
    independently from stored fit parameters, so this doesn't affect the
    requested side's output)."""
    n_trials, n_bins, n_units = tensor.shape
    flat = tensor.reshape(n_trials * n_bins, n_units)
    valid = ~np.isnan(flat).any(axis=1)
    out = np.full((n_trials * n_bins, n_dims), np.nan)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return out.reshape(n_trials, n_bins, n_dims)
    if side == "x":
        dummy_y = np.zeros((n_valid, cca.y_weights_.shape[0]))
        Xc, _ = cca.transform(flat[valid], dummy_y)
        out[valid] = Xc[:, :n_dims]
    else:
        dummy_x = np.zeros((n_valid, cca.x_weights_.shape[0]))
        _, Yc = cca.transform(dummy_x, flat[valid])
        out[valid] = Yc[:, :n_dims]
    return out.reshape(n_trials, n_bins, n_dims)


def fit_nuisance_pca(flat_other_train: np.ndarray, variance_cutoff: float = 0.99,
                      max_components: int = 100) -> tuple[PCA, int]:
    """Fit PCA on a nuisance population's flattened TRAIN-only residuals
    (e.g. 'all other simultaneously recorded neurons', variant B), and
    return (fitted PCA, k) where k is the smallest number of components
    reaching `variance_cutoff` cumulative explained variance (capped at
    `max_components`). 'Retain as many PCs as possible' (Axel, 2026-08-31):
    variance_cutoff=0.99 rather than the checkpoint's earlier 0.90/15
    default -- max_components mainly keeps the nuisance regression
    well-posed relative to the ~5k-14k available (trial,bin) samples.
    Apply to both train and test data via `pca.transform(x)[:, :k]`."""
    n = min(max_components, flat_other_train.shape[1], flat_other_train.shape[0] - 1)
    pca_o = PCA(n_components=n).fit(flat_other_train)
    cum_var = np.cumsum(pca_o.explained_variance_ratio_)
    k = min(int(np.searchsorted(cum_var, variance_cutoff) + 1), flat_other_train.shape[1])
    return pca_o, k


def regress_out(signal: np.ndarray, nuisance: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Fit a linear regression of `signal` (n_samples, n_units) on
    `nuisance` (n_samples, n_regressors) using only `train_mask` rows,
    return residuals for all rows (train and test alike, using train-fit
    coefficients -- avoids test-set leakage)."""
    reg = LinearRegression().fit(nuisance[train_mask], signal[train_mask])
    return signal - reg.predict(nuisance)


def cv_canonical_correlation(X: np.ndarray, Y: np.ndarray, n_components: int, n_folds: int = 5,
                              random_state: int = 0, nuisance: np.ndarray | None = None) -> dict:
    """K-fold CV (folds over samples -- caller must pre-group by trial so
    within-trial bins never split across train/test) canonical correlation
    per dimension. Returns held-out correlation per dimension (mean +/- sd
    ACROSS THE n_folds CV FOLDS -- not across trials or units) and the
    corresponding PCA-alignment baseline (also mean +/- sd across folds).
    If `nuisance` (n_samples, n_regressors) is given, X and Y are each
    linearly regressed on it (fit on the fold's train rows only) before
    CCA/PCA -- this is the partial-CCA step for variants B/C."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    n_dims = min(n_components, X.shape[1], Y.shape[1])
    cca_corrs = np.full((n_folds, n_dims), np.nan)
    pca_corrs = np.full((n_folds, n_dims), np.nan)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[train_idx] = True
        Xf, Yf = X, Y
        if nuisance is not None:
            Xf = regress_out(X, nuisance, train_mask)
            Yf = regress_out(Y, nuisance, train_mask)

        cca = CCA(n_components=n_dims, max_iter=2000).fit(Xf[train_idx], Yf[train_idx])
        Xc, Yc = cca.transform(Xf[test_idx], Yf[test_idx])
        for d in range(n_dims):
            if np.std(Xc[:, d]) > 0 and np.std(Yc[:, d]) > 0:
                cca_corrs[fold, d] = np.corrcoef(Xc[:, d], Yc[:, d])[0, 1]

        pca_x = PCA(n_components=n_dims).fit(Xf[train_idx])
        pca_y = PCA(n_components=n_dims).fit(Yf[train_idx])
        Xp, Yp = pca_x.transform(Xf[test_idx]), pca_y.transform(Yf[test_idx])
        for d in range(n_dims):
            if np.std(Xp[:, d]) > 0 and np.std(Yp[:, d]) > 0:
                pca_corrs[fold, d] = abs(np.corrcoef(Xp[:, d], Yp[:, d])[0, 1])

    return {
        "cca_mean": np.nanmean(cca_corrs, axis=0), "cca_sd": np.nanstd(cca_corrs, axis=0),
        "pca_mean": np.nanmean(pca_corrs, axis=0), "pca_sd": np.nanstd(pca_corrs, axis=0),
        "cca_folds": cca_corrs, "pca_folds": pca_corrs,
        "n_dims": n_dims, "errorbar_definition": "SD across n_folds CV folds (train/test splits over trials)",
    }


def select_regularization_trial_cv(
    resid_a: np.ndarray, resid_b: np.ndarray, valid_bins: np.ndarray,
    candidate_lambdas: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    n_folds: int = 3, rng: np.random.Generator | None = None,
) -> tuple[float, float, dict]:
    """Pick PartialCCA's ridge `regularization` by k-fold CV, maximizing
    mean held-out dim-1 canonical correlation (variant A only -- no
    nuisance Z). Folds are over TRIALS, not flattened (trial,bin) rows:
    bins within a trial are highly correlated (especially with the
    2026-08-31 sliding-window overlap), so a naive row-level KFold would
    leak information between train/validation. `resid_a`/`resid_b`:
    (n_trials, n_bins, n_units), NOT yet restricted to valid_bins (done
    internally per fold). Returns (best_lambda, best_score, all_scores)."""
    rng = rng or np.random.default_rng(0)
    n_trials = resid_a.shape[0]
    perm = rng.permutation(n_trials)
    folds = np.array_split(perm, n_folds)

    all_scores: dict[float, float] = {}
    best_lambda, best_score = candidate_lambdas[0], -np.inf
    for lam in candidate_lambdas:
        fold_scores = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            if len(val_idx) < 2 or len(train_idx) < 2:
                continue
            Xtr, _ = flatten_trial_bins(resid_a[train_idx][:, valid_bins, :])
            Ytr, _ = flatten_trial_bins(resid_b[train_idx][:, valid_bins, :])
            Xval = resid_a[val_idx][:, valid_bins, :].reshape(-1, resid_a.shape[-1])
            Yval = resid_b[val_idx][:, valid_bins, :].reshape(-1, resid_b.shape[-1])
            try:
                m = PartialCCA(regularization=lam).fit(Xtr, Ytr, None, verbose=False)
                ca, cb = m.transform(Xval, Yval)
                r = np.corrcoef(ca[0], cb[0])[0, 1]
                if not np.isnan(r):
                    fold_scores.append(r)
            except Exception:
                continue
        if fold_scores:
            mean_score = float(np.mean(fold_scores))
            all_scores[lam] = mean_score
            if mean_score > best_score:
                best_score, best_lambda = mean_score, lam
    return best_lambda, best_score, all_scores


def canonical_correlation_across_time(ca: np.ndarray, cb: np.ndarray, min_trials: int = 5) -> np.ndarray:
    """Trial-by-trial Pearson correlation between two (n_test_trials,
    n_bins) canonical-variate arrays, computed independently at each time
    bin (not the variate amplitude itself). NaN where too few valid trials
    or zero variance at that bin."""
    n_bins = ca.shape[1]
    out = np.full(n_bins, np.nan)
    for t in range(n_bins):
        a, b = ca[:, t], cb[:, t]
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < min_trials or np.std(a[m]) == 0 or np.std(b[m]) == 0:
            continue
        out[t] = np.corrcoef(a[m], b[m])[0, 1]
    return out
