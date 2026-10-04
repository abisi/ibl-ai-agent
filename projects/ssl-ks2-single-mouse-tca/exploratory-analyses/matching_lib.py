"""Consensus cross-mouse component matching, elaborating the pairwise pilot
(004) to the full cohort.

TCA components are unordered per mouse; with no shared neuron axis (unlike
the megamouse pipeline), only the shared time axis ties mice together. This
finds, for each of the RANK "slots", a consensus time-factor shape and a
per-mouse assignment of which of that mouse's own components best matches
each slot -- an iterative Hungarian-assignment scheme, analogous to k-means
but with each mouse's assignment constrained to be a one-to-one permutation
(a mouse's own RANK components are meant to be distinct, so each should map
to a distinct consensus slot).

Non-negative CP (ncp_hals on a min-max-normalized, non-negative tensor)
makes every factor mode non-negative by construction, so there is no sign
ambiguity to resolve before matching -- only the permutation ambiguity.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def zscore_columns(x):
    """z-score each column of a (n, k) array."""
    return (x - x.mean(axis=0)) / x.std(axis=0)


def consensus_match(time_factors_by_mouse, rank, n_iter=8, reference_mouse=None):
    """Iterative Hungarian-assignment consensus clustering in time-factor
    shape space.

    Parameters
    ----------
    time_factors_by_mouse : dict {session_id: ndarray (n_time_bins, rank)},
        all mice must share the same n_time_bins (same time_window/bin_size)
    rank : int, number of components per mouse (= number of consensus slots)
    n_iter : int, number of assign/update rounds
    reference_mouse : str, session_id to initialize consensus prototypes
        from; if None, the first session_id in sorted order is used, for
        reproducibility

    Returns
    -------
    prototypes : ndarray, shape (n_time_bins, rank), consensus time-factor
        shapes (z-scored)
    assignment : dict {session_id: ndarray (rank,)}, assignment[sid][k] =
        which of that mouse's original components (0-indexed) was matched
        to consensus slot k
    similarity : dict {session_id: ndarray (rank,)}, similarity[sid][k] =
        correlation between that mouse's matched component and slot k's
        final prototype
    """
    session_ids = sorted(time_factors_by_mouse.keys())
    z_by_mouse = {sid: zscore_columns(time_factors_by_mouse[sid]) for sid in session_ids}

    ref = reference_mouse if reference_mouse is not None else session_ids[0]
    prototypes = z_by_mouse[ref].copy()

    assignment = {sid: None for sid in session_ids}
    for _ in range(n_iter):
        # Assignment step: for each mouse, match its RANK components to the
        # RANK current prototypes with a single Hungarian assignment
        # (maximize total correlation = minimize negative correlation).
        slot_members = [[] for _ in range(rank)]
        for sid in session_ids:
            z = z_by_mouse[sid]
            sim = (z.T @ prototypes) / z.shape[0]  # (rank, rank) Pearson r, comp x slot
            comp_idx, slot_idx = linear_sum_assignment(-sim)
            # comp_idx/slot_idx are already sorted 0..rank-1 by slot since
            # linear_sum_assignment returns rows in increasing order for a
            # square cost matrix's optimal columns -- reorder explicitly
            # anyway so assignment[sid][slot] is always well-defined.
            order = np.argsort(slot_idx)
            assignment[sid] = comp_idx[order]
            for slot in range(rank):
                slot_members[slot].append(z[:, assignment[sid][slot]])

        # Update step: each prototype becomes the mean shape of its
        # currently-assigned components, re-z-scored so scale stays fixed
        # across iterations.
        new_prototypes = np.stack([np.mean(members, axis=0) for members in slot_members], axis=1)
        prototypes = zscore_columns(new_prototypes)

    similarity = {}
    for sid in session_ids:
        z = z_by_mouse[sid]
        sim = (z.T @ prototypes) / z.shape[0]
        similarity[sid] = np.array([sim[assignment[sid][k], k] for k in range(rank)])

    return prototypes, assignment, similarity
