"""Faithful port of the BWM paper's single-cell condition-combined shuffle
test (`brainwidemap/single_cell_stats/single_cell_util.py` in
`int-brain-lab/paper-brain-wide-map`, fetched 2026-08-18), plus a generic
stratified-combination wrapper used by all four SSL-adapted single-cell
tests in `projects/ssl-bwm-style-single-cell-decoding/`.

`TwoNmannWhitneyUshuf`/`Time_TwoNmannWhitneyUshuf` are direct ports (same
algorithm, vectorized RNG via numpy Generator instead of the legacy global
`np.random.choice` calls upstream uses, for a fixed-seed reproducible run --
this is a mechanical change, not a change to the test statistic or shuffle
scheme).

The SSL adaptation has no `probabilityLeft`-style block; per
`question.md`'s locked design decision, block-aware tests key `bx`/`by` on a
consecutive-same-t-1-outcome run index (`t1_outcome_run_index`) instead of
BWM's consecutive-same-block run index.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def two_n_mannwhitneyu_shuf(x: np.ndarray, y: np.ndarray, n_shuf: int, rng: np.random.Generator) -> np.ndarray:
    """Port of `TwoNmannWhitneyUshuf`: unstratified shuffled Mann-Whitney U.
    Returns an array of length n_shuf+1: [0] is the observed combined
    statistic (min of the x>y and y>x rank-sum-based U), [1:] are the null
    draws from label-shuffling."""
    nx, ny = len(x), len(y)

    t2 = np.concatenate([x, y])
    t = rankdata(t2)
    t1 = np.empty((n_shuf + 1, nx))
    t1[0, :] = t[:nx]
    for i in range(n_shuf):
        z = rng.choice(nx + ny, size=nx, replace=False)
        t1[i + 1, :] = t[z]
    numer = t1.sum(axis=1) - nx * (nx + 1) / 2

    t4 = np.concatenate([y, x])
    t5 = rankdata(t4)
    t3 = np.empty((n_shuf + 1, ny))
    t3[0, :] = t5[:ny]
    for i in range(n_shuf):
        z = rng.choice(nx + ny, size=ny, replace=False)
        t3[i + 1, :] = t5[z]
    numer2 = t3.sum(axis=1) - ny * (ny + 1) / 2

    return np.minimum(numer, numer2)


def time_two_n_mannwhitneyu_shuf(
    x: np.ndarray, y: np.ndarray, bx: np.ndarray, by: np.ndarray, n_shuf: int, rng: np.random.Generator
) -> np.ndarray:
    """Port of `Time_TwoNmannWhitneyUshuf`: block-aware shuffled Mann-Whitney
    U -- shuffles are constrained within matched `bx`/`by` block labels
    (here: t-1-outcome run index) to control for slow drift. Same return
    convention as `two_n_mannwhitneyu_shuf`."""
    nx, ny = len(x), len(y)

    def _one_direction(a, b, ba, bb, na, nb):
        t2 = np.concatenate([a, b])
        t = rankdata(t2)
        t1 = np.empty((n_shuf + 1, na))
        t1[0, :] = t[:na]
        block_list = np.intersect1d(ba, bb)
        for i_shuf in range(n_shuf):
            final_index = np.arange(na + nb)
            for blk in block_list:
                a_idx = np.argwhere(ba == blk)[:, 0]
                b_idx = np.argwhere(bb == blk)[:, 0]
                temp_index = np.concatenate([a_idx, b_idx + na])
                z1 = rng.choice(len(a_idx) + len(b_idx), size=len(a_idx) + len(b_idx), replace=False)
                final_index[temp_index] = temp_index[z1]
            t1[i_shuf + 1, :] = t[final_index[:na]]
        return t1.sum(axis=1) - na * (na + 1) / 2

    numer = _one_direction(x, y, bx, by, nx, ny)
    numer3 = _one_direction(y, x, by, bx, ny, nx)
    return np.minimum(numer, numer3)


def t1_outcome_run_index(t1_outcome: np.ndarray) -> np.ndarray:
    """Consecutive-same-value run index over an ordered t-1-outcome array,
    the SSL block analog for `Time_TwoNmannWhitneyUshuf`'s `bx`/`by` --
    mirrors BWM's own `s_block` construction (increment on every value
    change) in `single_cell_Working_example_stimulus.py`."""
    run = np.zeros(len(t1_outcome), dtype=float)
    for i in range(1, len(t1_outcome)):
        run[i] = run[i - 1]
        if t1_outcome[i] != t1_outcome[i - 1]:
            run[i] += 1
    return run


def combined_stratified_pvalue(
    strata: list[dict],
    n_shuf: int,
    block_aware: bool,
    rng: np.random.Generator,
) -> float:
    """Generic version of BWM's per-neuron `get_stim_time_shuffle` /
    `get_feedback_time_shuffle` / `get_block` combination step: run the
    (block-aware or plain) shuffle test on each stratum, sum numerators and
    denominators across strata into one combined statistic, then rank the
    observed value against the null.

    Each stratum dict has keys 'x', 'y' (1-D firing-rate arrays for the two
    levels of the tested factor within that stratum) and, if
    `block_aware`, 'bx', 'by' (matching block/run-index arrays).
    Strata with an empty x or y are skipped (denominator contribution 0).
    """
    numer_total = np.zeros(n_shuf + 1)
    denom_total = 0
    for s in strata:
        x, y = np.asarray(s["x"], dtype=float), np.asarray(s["y"], dtype=float)
        if len(x) == 0 or len(y) == 0:
            continue
        if block_aware:
            bx, by = np.asarray(s["bx"]), np.asarray(s["by"])
            numer = time_two_n_mannwhitneyu_shuf(x, y, bx, by, n_shuf, rng)
        else:
            numer = two_n_mannwhitneyu_shuf(x, y, n_shuf, rng)
        numer_total = numer_total + numer
        denom_total += len(x) * len(y)

    if denom_total == 0:
        return float("nan")
    cp = numer_total / denom_total
    ranks = rankdata(cp)
    return float(ranks[0] / (1 + n_shuf))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. NaN p-values pass through as NaN
    and are excluded from the correction (not counted toward m)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    m = valid.sum()
    if m == 0:
        return out
    idx = np.argsort(p[valid])
    ranked = p[valid][idx]
    ranks = np.arange(1, m + 1)
    q = ranked * m / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    valid_idx = np.where(valid)[0]
    out[valid_idx[idx]] = q
    return out
