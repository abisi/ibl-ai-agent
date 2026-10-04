"""Circular-shift permutation test for correlating a TCA trial factor against
a behavioral trial factor, guarding against spurious correlation from the two
traces sharing a slow trend/drift over the session (see question.md, "Not
spurious").
"""
import numpy as np
from scipy.stats import pearsonr


def circular_shift_test(neural_trace, behavior_trace, n_shifts=2000, rng=None):
    """Test whether `neural_trace` and `behavior_trace` correlate more than
    expected when their alignment is randomized but each trace's own
    autocorrelation/trend is preserved.

    NaNs in `behavior_trace` (e.g. before enough trials exist for a rolling
    window) are dropped first, together with the matching entries of
    `neural_trace`, so the test always runs on a single contiguous block of
    real values -- required for circular shifting to make sense (shifting
    through NaNs would just delete data, not randomize alignment).

    Parameters
    ----------
    neural_trace : ndarray, shape (n_trials,), one TCA trial-mode loading
    behavior_trace : ndarray, shape (n_trials,), e.g. P(lick) or d-prime;
        same trial order as `neural_trace`, may contain NaN
    n_shifts : int, number of circular shifts for the null distribution
    rng : numpy.random.Generator

    Returns
    -------
    real_r : float, real Pearson correlation on the valid (non-NaN) block
    null_r : ndarray, shape (n_shifts,), null correlations from circular
        shifts of `behavior_trace` relative to `neural_trace`
    p_value : float, two-sided empirical p-value: fraction of |null_r| >=
        |real_r| (using (count+1)/(n_shifts+1), so p is never exactly 0)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    valid = ~np.isnan(behavior_trace)
    neural_valid = neural_trace[valid]
    behavior_valid = behavior_trace[valid]
    n = len(behavior_valid)

    real_r, _ = pearsonr(neural_valid, behavior_valid)

    # A circular shift by 0 (or by n, equivalently) reproduces the real
    # alignment, so shifts are drawn from 1..n-1 to always test a genuinely
    # different alignment.
    shifts = rng.integers(1, n, size=n_shifts)
    null_r = np.empty(n_shifts)
    for i, s in enumerate(shifts):
        shifted = np.roll(behavior_valid, s)
        null_r[i], _ = pearsonr(neural_valid, shifted)

    p_value = (np.sum(np.abs(null_r) >= np.abs(real_r)) + 1) / (n_shifts + 1)
    return real_r, null_r, p_value
