"""Fast rank-based AUC + permutation significance, equivalent to
sklearn.metrics.roc_auc_score / Mann-Whitney U, vectorized over the
permutation axis (not sklearn, which would be too slow to call per-unit
with an inner permutation loop at this scale -- see question.md's
responsiveness/selectivity AUC definitions for the two use cases this
serves).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc_and_perm_p(values_pos: np.ndarray, values_neg: np.ndarray, rng: np.random.Generator, n_perm: int = 1000) -> tuple[float, float]:
    """AUC for distinguishing values_pos (label 1) from values_neg (label 0),
    plus a two-tailed permutation p-value (shuffle which pooled positions
    are label 1, n_perm times, keeping the pooled value multiset fixed --
    equivalent to shuffling class labels).
    """
    n1, n2 = len(values_pos), len(values_neg)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    pooled = np.concatenate([values_pos, values_neg])
    ranks = rankdata(pooled)
    total = n1 + n2

    u_obs = ranks[:n1].sum() - n1 * (n1 + 1) / 2
    auc_obs = u_obs / (n1 * n2)

    perm_idx = np.argsort(rng.random((n_perm, total)), axis=1)
    perm_ranks = ranks[perm_idx]
    u_perm = perm_ranks[:, :n1].sum(axis=1) - n1 * (n1 + 1) / 2
    auc_perm = u_perm / (n1 * n2)

    p = (1 + (np.abs(auc_perm - 0.5) >= np.abs(auc_obs - 0.5)).sum()) / (n_perm + 1)
    return float(auc_obs), float(p)
