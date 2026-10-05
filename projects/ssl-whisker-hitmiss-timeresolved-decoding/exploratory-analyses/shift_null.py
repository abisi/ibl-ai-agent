"""Linear-shift null for hit / miss axes (user 2026-10-05: "do the shifts for these analyses too"; same null as Part I and 146).
The lick labels of the time-ordered active whisker trials are shifted against the neural trials by k = 10-50 % of the trials,
non-wrapping (labels y[k:] with trials [0, n-k), or y[:n-k] with trials [k, n)); the axis (lick axis, coding direction) is
rebuilt from the shifted labels exactly as the real one and the passive pre -> post metric recomputed against it, with the
passive patterns unchanged. Shifts are drawn without replacement among the (lag, direction) pairs for which the axis can be
built (enough hits and misses), so every included session gets a null. Slow drift of labels and activity is kept, so an axis
that partly encodes session time is in the null; the null is conservative (real learning is time-correlated too).
"""

from __future__ import annotations

import numpy as np

LO, HI = 0.1, 0.5


def cut(y, k, dr):
    """shifted labels, positions (into the time-ordered trials) of the neural trials they are paired with, the trials' true labels"""
    n = len(y)
    if dr:
        return y[k:], np.arange(n - k), y[: n - k]
    return y[: n - k], np.arange(k, n), y[k:]


def valid_shifts(y, ok):
    """(lag, direction) pairs with lag in [LO n, HI n] whose shifted labels pass ok(y_shifted, positions)"""
    n = len(y)
    lo, hi = max(1, int(LO * n)), max(1, int(HI * n))
    out = []
    for k in range(lo, hi + 1):
        for dr in (0, 1):
            ys, pos, _ = cut(y, k, dr)
            if ok(ys, pos):
                out.append((k, dr))
    return out


def summarize(real, null, prefix="shift"):
    """real: dict metric -> real post - pre change; null: dict metric -> list of null changes.
    Columns: <prefix>_real_d<m>, _null_d<m>_mean / _sd, _excess_d<m> (real - null mean), _pct_d<m> (one-sided: fraction of null
    changes <= real, small = real more negative than time alone)."""
    out = {}
    for m, r in real.items():
        v = np.asarray([x for x in null.get(m, []) if np.isfinite(x)], float)
        out[f"{prefix}_real_d{m}"] = r
        out[f"{prefix}_null_d{m}_mean"] = float(v.mean()) if len(v) else np.nan
        out[f"{prefix}_null_d{m}_sd"] = float(v.std()) if len(v) else np.nan
        out[f"{prefix}_excess_d{m}"] = r - out[f"{prefix}_null_d{m}_mean"] if len(v) else np.nan
        out[f"{prefix}_pct_d{m}"] = float((np.sum(v <= r) + 1) / (len(v) + 1)) if len(v) and np.isfinite(r) else np.nan
    return out
