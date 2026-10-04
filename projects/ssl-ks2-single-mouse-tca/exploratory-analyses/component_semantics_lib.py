"""Classify a TCA component's time-factor shape into a (sign, latency-stage)
label, using only that component's own shape -- no cross-mouse matching
needed, unlike the consensus slots in matching_lib.py (which this can be
cross-checked against, but doesn't depend on).

Sign and latency are two independent reads of the same shape:
- sign: does the component's largest deviation from its own pre-stimulus
  baseline rise above baseline ("activation") or dip below it
  ("suppression")? NCP factors are non-negative throughout, so a
  "suppression" here always means "lower than this component's own
  baseline", not a negative value.
- stage: when does that largest deviation happen, post-stimulus? Bucketed
  into four latency windows loosely mapped onto processing stages typical
  of a whisker/auditory detection task:
    0-40 ms   "sensory"   (S1/A1 first-spike latencies)
    40-100 ms "decision"  (sensory-to-motor integration)
    100-160 ms "motor"    (action execution / licking onset)
    160-200 ms "late"     (tail of the window -- could reflect motor
                           follow-through or reward-related activity, but
                           this dataset's tensor is stimulus-aligned and
                           stops at +200 ms, well before most trials'
                           reward delivery, so "reward" is NOT a validated
                           read here; labelled "late" rather than "reward"
                           for that reason -- see question.md)
"""
import numpy as np

STAGE_BINS_MS = [(0, 40, "sensory"), (40, 100, "decision"), (100, 160, "motor"), (160, 1000, "late")]
STAGE_ORDER = ["sensory", "decision", "motor", "late"]
SIGN_ORDER = ["activation", "suppression"]


def classify_component(time_factor, time_bins):
    """
    Parameters
    ----------
    time_factor : ndarray, shape (n_time_bins,), one component's time-mode
        loading (non-negative, NCP)
    time_bins : ndarray, shape (n_time_bins + 1,), bin edges in seconds

    Returns
    -------
    sign : str, "activation" or "suppression"
    stage : str, "sensory" | "decision" | "motor" | "late"
    latency_ms : float, latency of the largest same-signed deviation from
        baseline

    Sign is decided from the *whole* post-stimulus window's mean relative
    to baseline, not a single bin -- some components have both a brief
    early dip and a larger, more sustained rise (or vice versa); picking
    whichever single bin deviates most (an earlier version of this
    function) let a small transient dip outvote an unambiguously dominant
    sustained response just because one bin's noise happened to be larger.
    Given that whole-window sign, latency is the timing of the largest
    deviation *in that same direction* -- so a component correctly called
    "activation" always gets a latency from its rise, never its dip.
    """
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    pre_mask = time_centers < 0
    post_mask = ~pre_mask

    baseline = time_factor[pre_mask].mean()
    post_vals = time_factor[post_mask]
    post_centers_ms = time_centers[post_mask] * 1000
    post_dev = post_vals - baseline

    sign = "activation" if post_vals.mean() > baseline else "suppression"
    idx = np.argmax(post_dev) if sign == "activation" else np.argmin(post_dev)
    latency_ms = float(post_centers_ms[idx])

    stage = next(name for lo, hi, name in STAGE_BINS_MS if lo <= latency_ms < hi)
    return sign, stage, latency_ms
