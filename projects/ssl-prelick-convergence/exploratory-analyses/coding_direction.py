"""Shared coding-direction estimators (user review 2026-10-07; used by Part I 076 and Part II within_day).

Both place events on the reference (R, label "FA" = false alarms or spontaneous licks) -> auditory-hit (AH) mean-
difference axis, normalised so that R = 0 and AH = 1. Z: (units, events) normalised rates; lab: event labels.

split_half_scores (unified estimator): the 50 random split-halves of lambda (057). For every split and both directions,
  the axis comes from one half (w_a = mean AH_a - mean R_a) and every event of the other half (all classes, WH included)
  gets the raw score (x - mean R_b) . w_a; each event's score is averaged over the splits / directions in which it was
  held out; one session normalisation D = mean over splits and directions of (mean AH_b - mean R_b) . w_a. Trial score
  t = raw / D. The WH mean of t equals lambda (same numerator, same denominator, same splits; up to the weighting of
  odd half sizes), so Part I session values and Part II trial-level time courses are one estimator.
kfold_scores: the Part II 5-fold coding direction (within_day 002 trial_scores, 'md' axis): unit axis from the training
  folds of AH and R, units z-scored on the training folds, held-out AH / R and fold-averaged WH projections, one session
  normalisation from the pooled held-out R and AH means; session undefined if held-out d' < MIN_DPRIME.
"""
import numpy as np

N_SPLIT = 50


def split_half_scores(Z, lab, n_split=N_SPLIT, seed=0):
    """returns (scores per event (nan if never held out), D, lambda_split = mean WH score)"""
    rng = np.random.default_rng(seed)
    idx = {c: np.where(lab == c)[0] for c in ("WH", "AH", "FA")}
    if min(len(v) for v in idx.values()) < 4:
        return None
    ssum, cnt, Dacc, nD = np.zeros(len(lab)), np.zeros(len(lab)), 0.0, 0
    for _ in range(n_split):
        h = {}
        for c, v in idx.items():
            p = rng.permutation(v); k = len(p) // 2
            h[c] = (p[:k], p[k:])
        for a, b in ((0, 1), (1, 0)):
            mA_a, mR_a = Z[:, h["AH"][a]].mean(1), Z[:, h["FA"][a]].mean(1)
            mA_b, mR_b = Z[:, h["AH"][b]].mean(1), Z[:, h["FA"][b]].mean(1)
            w = mA_a - mR_a
            Dacc += (mA_b - mR_b) @ w; nD += 1
            held = np.concatenate([h[c][b] for c in ("WH", "AH", "FA")])
            ssum[held] += (Z[:, held] - mR_b[:, None]).T @ w; cnt[held] += 1
    D = Dacc / nD
    if not np.isfinite(D) or D <= 0:
        return None
    t = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan) / D
    return t, D, float(np.nanmean(t[idx["WH"]]))


def kfold_scores(X_events_by_units, lab, seed=0):
    """Part II 5-fold coding direction (within_day 002 trial_scores); X: (events, units) rates. Returns the score dict
    of 002 ('md', 'dec', 'dprime')."""
    import importlib
    import pathlib
    import sys
    wd = pathlib.Path(__file__).resolve().parent / "within_day"
    sys.path.insert(0, str(wd))
    m002 = importlib.import_module("002_trial_slopes")
    return m002.trial_scores(X_events_by_units, lab, seed)
