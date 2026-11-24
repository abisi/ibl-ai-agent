"""Quiet (no-lick) stretches of whisker day-0 sessions, for the v3 baseline analysis (user request 2026-11-18).

The NWB trial table does not hold the drawn quiet-window duration (abort_window_start_time == start_time), so the
quiet window of each trial is reconstructed from the piezo licks:
    quiet start = max(previous trial stop_time, last lick before the trial start + skip)
    quiet end   = trial start - 10 ms                       (whisker, auditory and false-alarm trials)
                = response_window_stop_time                 (correct rejections: quiet window + response window are
                                                             one contiguous lick-free stretch)
The task restarts the quiet timer on every lick, so no lick falls inside a stretch (checked: `n_licks_inside`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import spec_lib as sl

PRE_STIM_GAP = 0.01  # s kept free before the stimulus (artefact window)


def read_behaviour(session_id):
    """Full NWB trial table (all contexts, sorted) and piezo lick times (s)."""
    import pynwb
    with pynwb.NWBHDF5IO(str(sl.cl.NWB_DIR / f"{session_id}.nwb"), "r", load_namespaces=True) as io:
        n = io.read()
        t = n.trials.to_dataframe()
        licks = np.sort(np.asarray(n.processing["behavior"]["BehavioralEvents"].time_series["piezo_lick_times"]
                                   .timestamps[:], float))
    return t.sort_values("start_time").reset_index(drop=True), licks


def active_trials(t):
    """Active trials as the spectral run (sl.load_session): perf != 6, warm-up cut, A1 trim; early-lick trials kept
    here (flagged) because they count for quiescence."""
    a = t[(t.context != "passive") & (t.perf != 6)].sort_values("start_time").reset_index(drop=True)
    wi = np.flatnonzero((a.trial_type == "whisker_trial").to_numpy())
    if len(wi):
        a = a.iloc[max(0, wi[0] - 1):].reset_index(drop=True)
    # A1 trim on the trials without early licks (as the run), then cut the full table at the same time
    ok = a[a.early_lick == 0]
    lick = np.flatnonzero(ok.lick_flag.to_numpy() == 1)
    if len(lick):
        tail = ok.iloc[lick[-1] + 1:]
        if (tail.trial_type == "whisker_trial").sum() >= 5 and (tail.trial_type == "auditory_trial").sum() >= 1:
            a = a[a.start_time <= ok.start_time.iloc[lick[-1]] + 1e-6].reset_index(drop=True)
    return a


def stretches(t_full, licks, skip=0.0):
    """One row per active trial: quiet stretch bounds and quiescence measures.
    t_full: full trial table (all contexts) for the previous stop time; skip: s dropped after the last lick."""
    a = active_trials(t_full)
    stops = t_full.stop_time.to_numpy()
    starts_full = t_full.start_time.to_numpy()
    rows = []
    for r in a.itertuples():
        i = np.searchsorted(starts_full, r.start_time - 1e-6)  # index of this trial in the full table
        prev_stop = stops[i - 1] if i > 0 else np.nan
        prev_lick = int(t_full.lick_flag.iloc[i - 1]) if i > 0 else -1
        k = np.searchsorted(licks, r.start_time)
        last = licks[k - 1] if k > 0 else np.nan
        cr = (r.trial_type == "no_stim_trial") and r.lick_flag == 0 and r.early_lick == 0
        end = r.response_window_stop_time if cr else r.start_time - PRE_STIM_GAP
        q0_beh = np.nanmax([prev_stop, last]) if np.isfinite([prev_stop, last]).any() else np.nan
        q0 = np.nanmax([prev_stop, last + skip]) if np.isfinite([prev_stop, last]).any() else np.nan
        n_iti = int(((licks > prev_stop) & (licks < r.start_time)).sum()) if np.isfinite(prev_stop) else np.nan
        n_in = int(((licks > q0) & (licks < end)).sum()) if np.isfinite(q0) else np.nan
        kind = {"whisker_trial": "W", "auditory_trial": "A", "no_stim_trial": "NS"}[r.trial_type]
        rows.append(dict(start_time=r.start_time, trial_type=kind, lick_flag=int(r.lick_flag),
                         early_lick=int(r.early_lick), cr=cr, prev_lick=prev_lick, prev_stop=prev_stop, last_lick=last,
                         since_last_lick=r.start_time - last, quiet_beh=r.start_time - q0_beh,
                         iti=r.start_time - prev_stop, n_iti_licks=n_iti, q_start=q0, q_end=end,
                         q_len=end - q0 if np.isfinite(q0) else np.nan, n_licks_inside=n_in))
    return pd.DataFrame(rows)


def segments(st, seg_len=1.0):
    """Non-overlapping segments of seg_len s inside each usable stretch, laid back from the stretch end (so the last
    segment ends at the stimulus or at the CR response-window end). Early-lick trials are not used.
    Returns DataFrame(trial (row of st), t0, cr)."""
    out = []
    for j, r in st.iterrows():
        if r.early_lick or not np.isfinite(r.q_start) or r.q_len < seg_len or r.n_licks_inside:
            continue
        n = int(np.floor(r.q_len / seg_len + 1e-9))
        for s in range(n):
            out.append((j, r.q_end - (s + 1) * seg_len, bool(r.cr)))
    return pd.DataFrame(out, columns=["trial", "t0", "cr"])
