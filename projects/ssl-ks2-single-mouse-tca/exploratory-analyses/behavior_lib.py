"""Behavioral trial-factor definitions: smoothed P(lick) and rolling d-prime,
computed directly from ssl_ks2_ephys trials.parquet (this dataset has no
stored d_prime_w/reward_group columns, unlike the M:-drive rates_df the
megamouse pipeline consumes -- see question.md).
"""
import numpy as np
from scipy.stats import norm

Z_EPS = 0.5  # standard log-linear correction: clip rate to [eps/n, 1-eps/n]-like bound


def smoothed_plick(aligned, window=10):
    """Rolling mean of lick_flag over the trial sequence in `aligned` (the
    same whisker-trial sequence, in the same order, as the tensor's trial
    axis -- so the output lines up index-for-index with a TCA trial factor).

    Parameters
    ----------
    aligned : DataFrame with a `lick_flag` column, one row per trial, in
        tensor trial order
    window : int, trailing rolling-window size in trials

    Returns
    -------
    ndarray, shape (n_trials,), rolling mean lick rate (NaN for the first
    few trials where the window is incomplete)
    """
    return aligned["lick_flag"].rolling(window, min_periods=3).mean().to_numpy()


def rolling_dprime(trials_session, aligned, window_trials=20):
    """Rolling d-prime (signal-detection sensitivity) at each whisker trial
    in `aligned`, using nearby active-context whisker trials as signal and
    nearby active-context no_stim trials as noise.

    d' = Z(hit rate) - Z(false-alarm rate), with hit/FA rates each computed
    over a trailing window of `window_trials` same-type active trials
    preceding (and including) the whisker trial's own time -- so this stays
    causal, like the smoothed P(lick) curve above, and is defined at every
    whisker trial in `aligned` (not just a session-wide scalar).

    Parameters
    ----------
    trials_session : DataFrame, this session's full trial table (all trial
        types), used to find nearby no_stim trials
    aligned : DataFrame, the whisker-trial sequence from
        `tca_lib.align_whisker_trials_first_hit` (tensor trial order)
    window_trials : int, number of preceding same-type active trials to
        average over for each rate

    Returns
    -------
    ndarray, shape (n_trials,), rolling d-prime (NaN before both a full
    hit-rate and FA-rate window exist)
    """
    active = trials_session[trials_session["context"] == "active"].sort_values("start_time")
    whisker_active = active[active["trial_type"] == "whisker_trial"]
    no_stim_active = active[active["trial_type"] == "no_stim_trial"]

    hit_rate = whisker_active["lick_flag"].rolling(window_trials, min_periods=window_trials).mean()
    fa_rate = no_stim_active["lick_flag"].rolling(window_trials, min_periods=window_trials).mean()
    hit_rate = hit_rate.reindex(whisker_active.index)
    fa_rate_at_time = fa_rate.reindex(no_stim_active.index)

    # For each whisker trial, look up the most recent no_stim rolling FA rate
    # at or before that trial's own time (asof join on start_time), so the
    # noise-rate estimate is causal too and always defined relative to the
    # whisker trial it's paired with.
    fa_series = fa_rate_at_time.copy()
    fa_series.index = no_stim_active["start_time"].to_numpy()
    fa_series = fa_series.sort_index()
    whisker_times = whisker_active["start_time"].to_numpy()
    fa_at_whisker = fa_series.reindex(fa_series.index.union(whisker_times)).ffill().reindex(whisker_times).to_numpy()

    hit_clipped = np.clip(hit_rate.to_numpy(), 1 / (2 * window_trials), 1 - 1 / (2 * window_trials))
    fa_clipped = np.clip(fa_at_whisker, 1 / (2 * window_trials), 1 - 1 / (2 * window_trials))
    dprime_by_whisker_trial = norm.ppf(hit_clipped) - norm.ppf(fa_clipped)

    dprime_series = whisker_active[["start_time"]].copy()
    dprime_series["dprime"] = dprime_by_whisker_trial
    merged = aligned[["start_time"]].merge(dprime_series, on="start_time", how="left")
    return merged["dprime"].to_numpy()
