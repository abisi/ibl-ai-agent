"""009 -- Full dataset: decoders, continuous readouts and event counts saved threshold-free, so that every quantity can be
recomputed for any threshold; event rates at held-out-sensitivity thresholds across cohorts (user 2026-10-29, after 008).

Decoders (v5, user 2026-10-30): whisker and auditory, per area group and whole brain "All"; training = every target stimulus
vs no stimulus, in the active task (perf != 6, warm-up cut, rule A1; negatives = correct rejections) and in the passive pre and
post blocks (negatives = shams, 0.8 s before each passive stimulus of the other modality), no quiet filter; 30-ms window 5-35 ms
after onset; classes balanced within context (task / passive, sample weights), so the block cannot separate the classes;
nested CV (outer folds: out-of-fold decision values; balanced accuracy at 0 = decoder quality, overall and per context:
bacc_task, bacc_passive_pre / post; tpr_passive_pre / post = passive target stimuli detected).
Threshold (v6, user 2026-11-01): median over the N_SHUF shuffled templates of the shuffled readout's 99.9th percentile in
valid time (Renard, Foustoukos et al. rule; 99.9 at 200 windows / s ~ one null window above threshold every 5 s). The shuffled
readouts come from the same moments as the real one, so the threshold and the null counts are matched to the behavioural state;
an event then means the template's own pattern, beyond the population fluctuations that any rate-matched projection picks up.
Rows thr_source = shuf<q> (THR_MAIN = shuf99.9), q in NULL_Q. Good decoder: balanced accuracy >= GOOD_BACC.
v7 (user 2026-11-01): two readouts per target, each with its own 100 shuffles, threshold, events and rows:
  decoder      decision value of the logistic decoder (rows shuf<q>; shuffles = weights permuted within rate quintiles)
  correlation  Pearson correlation across units between the z-scored population vector of each 30-ms window and the template =
               mean z-scored training-window pattern of the target stimuli minus that of the no-stimulus training events
               (files / columns suffix _corr, rows corr_shuf<q>; shuffles = template permuted within rate quintiles)
  Valid time = every readout window not touching onset - 10 ms .. + 35 ms (artefact + training windows) of a trial onset or
  sham, passive blocks included, from START_EXCL s after the first spike to END_EXCL s before the last; no time shifts. Masks add evoked (window ends within
  EVOKED_S after a stimulus onset) and passive; still also excludes the evoked tail. Events carry readout, passive, evoked
  and dt_prev_stim (s from the peak to the previous stimulus onset).
Unit scaling (user 2026-10-30): each unit z-scored with its mean / s.d. over all valid continuous 30-ms windows (not the training
events, where sparse units have few or no spikes); s.d. 0 (unit silent in valid time) -> 1. Continuous readout: 30-ms window,
5-ms stride, decision value in median / MAD units of valid time (each null trace in its own).
Nulls: shuffled template = weights permuted within SHUF_STRATA quantiles of the continuous mean count (quintiles, user 2026-10-30;
a free permutation moves large weights onto sparse units and makes the shuffled readout spikier than the real one);
time shift = template kept, each unit's binned counts shifted linearly by its own lag (+-30-300 s, no wrap-around: every count
is evaluated only in the core of the recording, BORDER_FRAC (>= the largest lag) cut at each end; user 2026-10-30).
Rates in events / s.
Sessions: every session of the units table (quality_label good / mua) whose mouse is R+ or R- in the mouse reference sheet
(cohort from the sheet, mice flagged exclude / exclude_ephys dropped); stage = day 0 or expert (day >= 1).
Saved per session x group x target (threshold-free):
  counts[trace, mask, thr]: number of events (runs above threshold inside valid time, gaps < MERGE merged) for trace = real,
  N_SHUF shuffled templates, N_NULL time shifts; thr = THR_F (1-30 MAD, 0.1 steps); seconds per mask. Valid time (user
  2026-10-30, v4) = core of the recording (BORDER_FRAC cut at each end), outside every trial (onset - 10 ms .. onset + 5 s; 10 ms
  = the spike-artefact window) and outside the passive blocks (trial table context == passive, first start to last stop), so
  events are detected in all inter-trial time, not only in the task quiet window. Masks (MASKS, MASK_MODE):
    all       every valid-time event
    still     events whose every window is still: whisker tracked, no whisking / jaw movement (speed > median + MOVE_K robust s.d. per
              session and body part; jaw only where lick-jaw sync passes, 008) within MOVE_MARGIN of the window, no lick
              within LICK_MARGIN (main measure; the user filters events by movement wherever they fall)
    whisk / jaw / lick   events with at least one window in whisking / jaw movement / lick time (same definitions)
    prestart  events starting in the task quiet window (abort_window_start..stop_time of the kept active trials), as before
  Event table (events_<target>_<group>.parquet, SENS_MAIN): per event, whisk / jaw / lick (any window), still (all windows),
  movement ("none", "whisk", "jaw", "lick" or combinations, "unknown" without whisker tracking), whisk_max and jaw_max (raw
  maxima within MOVE_MARGIN of the event), lick_dt (s from the peak to the nearest lick), prestart, dt_prev_onset / dt_next_onset
  (s from the peak to the previous / next trial onset), identity p.
  the readout z (float32), weights, training mean / s.d., pAUC, out-of-fold values.
  peaks: per stimulus / sham event, the peak readout 10-100 ms after onset (held-out sensitivity at any threshold).
Identity p (007) is computed for the events at SENS_MAIN only (needs the spike counts).
Unit set (--unit-set): stable (default; quality_label good / mua AND stable in 137_stable_units: coverage >= 0.9,
presence >= 0.5, drift joint test; sessions without drift results have no stable unit) or all (good / mua).
Outputs: <HOME>/full/<unit_set>/<session>/: session.npz (grid, valid, quiet, epoch), behaviour.npz, params.json,
  g_<group>_<target>.npz, peaks_<group>.parquet, events_<target>_<group>.parquet (SENS_MAIN), decoders.csv, and the
  008-format continuous_<target>_<group>.npz for the 008 figures; session.npz also holds the still / whisk / jaw / lick grids. --summary: <HOME>/full_sessions.csv, full_decoders_<unit_set>.csv,
  full_cohort_stats_<unit_set>.csv, figures/full_<unit_set>/fig17_cohort_rates_<target>, fig18_full_real_vs_shuffled_<target>,
  fig24_passive_heldout_auc. Held-out passive decoding (decoders.csv auc_{sham,other}_{pre,post,prepost}): AUC of the readout in
  the training window (30-ms window ending 35 ms after onset), target stimuli vs shams / vs the other modality's stimuli.
"""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import pathlib
import sys
import time
import traceback
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m6 = importlib.import_module("006_fdr_threshold_heldout_controls")
m7 = importlib.import_module("007_content_threshold_participation")
m8 = importlib.import_module("008_sensitivity_threshold_cohorts")
m1.EXCL = (-0.01, 0.065)                        # v7 (user 2026-11-01): exclude only readout windows touching onset - 10 ms (spike-artefact
                                                 # window) .. onset + 35 ms (training window) of every trial onset and sham: window ends
                                                 # in [onset - 10 ms, onset + 35 ms + 30 ms); v4-v6: onset - 10 ms .. + 5 s
MOUSE_INFO = pathlib.Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
STABLE = m1.RES / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet"
UNIT_SETS = ("stable", "all")
ROOT = m1.HOME / "full"
SENS_LEVELS = (0.25, 0.5, 0.8)
NULL_Q = (99.0, 99.5, 99.9, 99.99)               # null-based thresholds: median over the null traces of this percentile of the
                                                 # null readout in valid time (as Renard, Foustoukos et al. 2026: median of the
                                                 # shuffles' 99th percentile); rows thr_source = shuf<q> (decoder readout) and
                                                 # corr_shuf<q> (correlation readout)
NULL_SRC = ("shuf",)                             # v7: shuffled templates only (time shifts dropped, user 2026-11-01)
READOUTS = ("", "_corr", "_pcorr")               # v7 (user 2026-11-01 / 11-02): decoder, correlation, partial correlation | whisker velocity
RNAME = {"": "decoder", "_corr": "corr", "_pcorr": "pcorr"}      # readout column of the tables
PREFIX = {"": "", "_corr": "corr_", "_pcorr": "pcorr_"}           # thr_source prefix per readout
AV_GAP, AV_LP, AV_MIN_RUN = 0.05, 30.0, 0.5      # whisker angular velocity (user 2026-11-02): DLC angle gaps <= AV_GAP s interpolated,
                                                 # AV_LP Hz 4th-order Butterworth low-pass (zero phase) on stretches >= AV_MIN_RUN s
VEL_LOG0 = 25.0                                  # deg/s: scale of the log-compressed velocity regressor (partial correlation)
CORR_PREFIX = "corr_"                            # thr_source prefix of the correlation-readout rows
EVOKED_S = 1.0                                   # s after a stimulus onset (any context): events there are flagged "evoked"
END_EXCL = 30.0                                  # s before the last spike excluded (v7: end-of-recording artefact; START_EXCL at the start)
NULL_Q_V5 = (99.0, 99.5, 99.9)                   # null_q columns of files written before v6 (no null_qs saved)
SOFT_C_FRAC = 1.0                                # correlation readout soft normalisation (user 2026-11-02, 016): z_u = (x_u - mu_u) /
                                                 # (sd_u + c), c = SOFT_C_FRAC x median unit s.d. over valid time; sparse units no
                                                 # longer dominate r (016 example: top-5-unit share at event peaks 0.96 -> 0.29)
SHUF_SCALE = "raw"                               # v7 (user 2026-11-01): every readout in raw units -- decision value (log-odds) of
                                                 # the real / shuffled decoder (permuted w, same b and unit scaling), Pearson r of the
                                                 # correlation readout; thresholds exact (no grid); MAD only as reference (thr_norm)
SENS_MAIN = 0.5
MIN_PER_BLOCK = 5
N_SHUF = 100                                     # v6: 100 shuffled templates (threshold and null counts; user 2026-11-01)
SHUF_STRATA = 5                                  # shuffle: permute w within this many firing-rate quantiles of the training mean
                                                 # count (1 = across all units); quintiles, user 2026-10-30 (tests 2026-10-30)
N_NULL = 0                                       # linear time shifts (v5-v6: 100); v7: dropped with the recording borders (user 2026-11-01)
BORDER_FRAC = 0.05                               # linear time shift: fraction of the recording cut at each end (>= max lag; only if N_NULL)
GOOD_PAUC = 0.8                                  # pAUC kept in the tables only
GOOD_BACC = 0.7                                  # good decoder: out-of-fold balanced accuracy (user 2026-10-30)
THR_MAIN = "shuf99.9"                            # main threshold rows (decoders.csv thr_source; v6, user 2026-11-01)
THR_MAIN_CORR = CORR_PREFIX + THR_MAIN           # same rule, correlation readout
THR_TEXT = (f"threshold = median over {N_SHUF} shuffled templates (weights / template permuted within {SHUF_STRATA} firing-rate "
            f"quantiles) of the shuffled readout's 99.9th percentile in valid time")
THR_SHORT = f"shuffle null: median over {N_SHUF} template shuffles of the 99.9th pct"
TRAIN = "target stimuli vs task correct rejections + passive pre / post shams (all trials, no quiet filter), classes balanced within context"
MIN_SESS = 3                                     # sessions per cohort at a stage for an R+ vs R- test (area inclusion rule)
THR_F = np.round(np.arange(1.0, 30.0 + 1e-9, 0.1), 2)
MOVE_K = 5.0                                     # movement of a body part: speed > median + MOVE_K robust s.d. (1.4826 MAD) of its
                                                 # speed over valid time, per session and body part (whisker, jaw; user 2026-11-01)
Z_EDGES = np.array([-np.inf, 1, 2, 3, 5, 10, 20, np.inf])     # speed bins, robust z (movement tuning curves)
LICK_EDGES = np.array([0, 0.25, 0.5, 1, 2, 5, np.inf])         # bins of the time to the nearest lick (s)
TUNE = {"wz": len(Z_EDGES) - 1, "jz": len(Z_EDGES) - 1, "lk": len(LICK_EDGES) - 1}
MASKS = ("all", "still", "whisk", "jaw", "lick", "prestart", "evoked", "passive", "nonevoked", "qw", "qw_still", "iti", "iti_still") + \
    tuple(f"{k}{i}" for k, n in TUNE.items() for i in range(n))
MASK_MODE = np.array(["any", "all", "any", "any", "any", "first", "any", "first", "first", "first", "all", "first", "all"]
                     + ["first"] * sum(TUNE.values()))   # event in mask: any / every / its first window in it
COND = ("qw", "qw_still", "iti", "iti_still")
# behavioural conditions (user 2026-11-01; overlapping):
#   qw          task quiet window (abort window before a kept active trial: no lick required) and no lick within LICK_MARGIN
#   qw_still    qw and no whisking / jaw movement (every window of the event)
#   iti         inter-trial time of the task epoch: outside every trial (start_time .. stop_time) and the evoked tail; includes qw;
#               movement allowed
#   iti_still   iti and no whisking / jaw movement / lick (009 still; every window of the event)
# movement tuning masks (outside the evoked tail; event = its first window): wz<i> / jz<i> = whisker / jaw speed in robust-z bin i
# (Z_EDGES; robust z of the session's speed over valid time), lk<i> = time to the nearest lick in LICK_EDGES bin i.
MOVE_MARGIN = 0.25                               # s: movement within this of a window's 30-ms span makes it moving
LICK_MARGIN = 1.0                                # s: lick within this of the window (= 001 QUIET_GAP)
METRICS = [("rate", "events / s,\nall valid time"), ("rate_still", "events / s,\nstill (no movement)"),
           ("excess", "all valid: excess vs shuffled\ntemplates (events / s)"), ("excess_still", "still: excess vs shuffled\ntemplates (events / s)")]
STILL_NOTE = ("valid = every readout window not touching onset - 10 ms .. + 35 ms of a trial onset or sham (artefact + training "
              f"windows; passive blocks included); still = no whisking or jaw movement (speed > median + "
              f"{MOVE_K:.0f} robust s.d., per session and body part) within {MOVE_MARGIN * 1000:.0f} ms, no lick within {LICK_MARGIN:.0f} s and not within {EVOKED_S:.0f} s after a stimulus, "
              "for every window of the event")
LINTHRESH = 1e-3                                 # symlog axes of rates (events / s)
GAP = max(1, int(round(m1.MERGE / m1.STEP)))
STAGES = m8.STAGES
COH = m8.COH


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        x = p[ok]
        o = np.argsort(x)
        r = x[o] * ok.sum() / np.arange(1, ok.sum() + 1)
        q[np.flatnonzero(ok)[o]] = np.minimum.accumulate(r[::-1])[::-1].clip(max=1)
    return q


def sessions_table():
    U = pd.read_parquet(m1.UNITS, columns=["session_id", "mouse_id", "day", "reward_group", "quality_label"])
    U = U[U.quality_label.isin(["good", "mua"])]
    n = U.groupby("session_id").size().rename("n_units_table")
    S = U.drop_duplicates("session_id")[["session_id", "mouse_id", "day", "reward_group"]].merge(n, on="session_id")
    M = pd.read_excel(MOUSE_INFO)[["mouse_id", "reward_group", "exclude", "exclude_ephys"]].drop_duplicates("mouse_id")
    S = S.merge(M.rename(columns=dict(reward_group="cohort")), on="mouse_id", how="left")
    S["label_mismatch"] = S.reward_group.notna() & (S.reward_group != S.cohort)
    S["include"] = S.cohort.isin(["R+", "R-"]) & (S.exclude.fillna(0) == 0) & (S.exclude_ephys.fillna(0) == 0)
    S["stage"] = np.where(S.day == 0, "day 0", "expert") + " " + S.cohort.fillna("?")
    return S.sort_values("n_units_table", ascending=False).reset_index(drop=True)


def stage_of(sid):
    S = sessions_table().set_index("session_id")
    return S.loc[sid, "stage"]


def unit_set_mask(sid, units, unit_set):
    """all: quality_label good / mua (units table); stable: also stable in 137_stable_units (coverage, presence, drift test;
    joined on electrode_group + NWB cluster id; a unit without a drift result is not stable)"""
    if unit_set == "all":
        return np.ones(len(units), bool)
    T = pd.read_parquet(STABLE, columns=["session_id", "electrode_group", "nwb_cluster_id", "stable"])
    T = T[T.session_id == sid].assign(cluster_id=lambda d: d.nwb_cluster_id.astype(str))
    x = units[["electrode_group", "cluster_id"]].merge(T[["electrode_group", "cluster_id", "stable"]], on=["electrode_group", "cluster_id"],
                                                       how="left", validate="one_to_one")
    return x.stable.fillna(False).to_numpy(bool)


def rate_strata(mu, n):
    """firing-rate quantile (0..n-1) of each unit; n = 1: one stratum"""
    return np.minimum(np.argsort(np.argsort(mu)) * n // len(mu), n - 1)


def count_curve(z, vidx, M, mode=MASK_MODE, thr=THR_F):
    """events per mask (rows of M, in MASKS order) at every threshold; an event = run of z > thr in valid time, gaps <= GAP bins
    merged; it counts in a mask if any / every / its first window above threshold lies in the mask (mode per mask)"""
    zv = z[vidx]
    Mv = M[:, vidx]
    out = np.zeros((len(M), len(thr)), np.int32)
    for j, t in enumerate(thr):
        k = np.flatnonzero(zv > t)
        if not len(k):
            break                                                   # thresholds increase: nothing above any more
        r0 = np.r_[0, np.flatnonzero(np.diff(vidx[k]) > GAP) + 1]
        cnt = np.add.reduceat(Mv[:, k].astype(np.int32), r0, axis=1)
        ln = np.diff(np.r_[r0, len(k)])
        res = np.where((mode == "first")[:, None], Mv[:, k[r0]], np.where((mode == "all")[:, None], cnt == ln, cnt > 0))
        out[:, j] = res.sum(1)
    return out


def dilate(b, before, after):
    """window i flagged if any b in [i - before, i + after]"""
    c = np.r_[0, np.cumsum(b, dtype=np.int64)]
    i = np.arange(len(b))
    return (c[np.clip(i + after + 1, 0, len(b))] - c[np.clip(i - before, 0, len(b))]) > 0


def to_grid(grid, t):
    """grid bin (window end) of each time: grid[i - 1] < t <= grid[i]"""
    i = np.searchsorted(grid, t)
    return i[(i >= 0) & (i < len(grid))]


def movement_grids(grid, licks, dlc, valid, jaw_ok):
    """per window: whisking / jaw movement within MOVE_MARGIN of its 30-ms span, lick within LICK_MARGIN; whisker tracking
    known (a tracked sample within 50 ms); jaw used only where lick-jaw sync passes (008)"""
    n = len(grid)
    pre, post = int(np.ceil((m1.WIDTH + MOVE_MARGIN) / m1.STEP)), int(np.ceil(MOVE_MARGIN / m1.STEP))
    out, info = {}, {}
    w_known = np.zeros(n, bool)
    whisk = np.zeros(n, bool)
    tc = grid - m1.WIDTH / 2                                        # window centres (continuous speeds)
    wz, jz = np.full(n, np.nan, np.float32), np.full(n, np.nan, np.float32)
    info["whisk_thr"] = info["jaw_thr"] = np.nan
    if "whisker_speed" in dlc:
        t, v = dlc["whisker_speed"]
        fin = np.isfinite(v)
        b = np.zeros(n, bool)
        b[to_grid(grid, t[fin])] = True
        w_known = dilate(b, 10, 10)
        ok = fin & valid[np.clip(np.searchsorted(grid, t), 0, n - 1)]
        if ok.sum() > 100:
            med, rsd = m1.robust(v[ok])
            info["whisk_thr"], info["whisk_med"], info["whisk_rsd"] = med + MOVE_K * rsd, med, rsd
            b = np.zeros(n, bool)
            b[to_grid(grid, t[fin & (v > info["whisk_thr"])])] = True
            whisk = dilate(b, pre, post) & w_known
            wz = np.where(w_known, np.interp(tc, t[fin], (v[fin] - med) / rsd), np.nan).astype(np.float32)
    jaw = np.zeros(n, bool)
    if jaw_ok and "jaw" in dlc:
        tj, jd = dlc["jaw"]
        sp = np.abs(np.gradient(jd, tj))
        gi = np.clip(np.searchsorted(grid, tj), 0, n - 1)
        ok = np.isfinite(sp) & valid[gi]
        if ok.sum() > 100:
            med, rsd = m1.robust(sp[ok])
            thr = med + MOVE_K * rsd
            b = np.zeros(n, bool)
            b[to_grid(grid, tj[np.isfinite(sp) & (sp > thr)])] = True
            jaw = dilate(b, pre, post)
            info["jaw_thr"], info["jaw_med"], info["jaw_rsd"] = thr, med, rsd
            f_ = np.isfinite(sp)
            jz = np.interp(tc, tj[f_], (sp[f_] - med) / rsd, left=np.nan, right=np.nan).astype(np.float32)
    b = np.zeros(n, bool)
    b[to_grid(grid, np.asarray(licks, float))] = True
    lp, lq = int(np.ceil((m1.WIDTH + LICK_MARGIN) / m1.STEP)), int(np.ceil(LICK_MARGIN / m1.STEP))
    lick = dilate(b, lp, lq)
    lk = np.sort(np.asarray(licks, float))
    lick_dt = np.full(n, np.inf, np.float32)
    if len(lk):
        j = np.searchsorted(lk, tc)
        lick_dt = np.minimum(np.abs(tc - lk[np.clip(j - 1, 0, len(lk) - 1)]), np.abs(lk[np.clip(j, 0, len(lk) - 1)] - tc)).astype(np.float32)
    out = dict(whisk=whisk, jaw=jaw, lick=lick, whisk_known=w_known, still=w_known & ~whisk & ~jaw & ~lick, wz=wz, jz=jz, lick_dt=lick_dt)
    info.update(jaw_used=bool(np.isfinite(info["jaw_thr"])), whisk_known_frac=float(w_known[valid].mean()) if valid.any() else np.nan,
                **{f"frac_{k}": float(out[k][valid].mean()) if valid.any() else np.nan for k in ("whisk", "jaw", "lick", "still")})
    info = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in info.items()}
    return out, info


def event_movement(E, grid, mv, dlc, licks, onsets):
    """movement flags / values of detected events (E: start, end, peak grid indices)"""
    if not len(E):
        return pd.DataFrame(columns=["whisk", "jaw", "lick", "still", "movement", "whisk_max", "jaw_max", "lick_dt", "dt_prev_onset",
                                     "dt_next_onset"])
    cs = {k: np.r_[0, np.cumsum(mv[k], dtype=np.int64)] for k in ("whisk", "jaw", "lick", "still", "whisk_known")}
    a, b = E[:, 0], E[:, 1] + 1
    f = {k: (cs[k][b] - cs[k][a]) for k in cs}
    ln = b - a
    d = pd.DataFrame(dict(whisk=f["whisk"] > 0, jaw=f["jaw"] > 0, lick=f["lick"] > 0, still=f["still"] == ln))
    known = f["whisk_known"] == ln
    d["movement"] = ["+".join(k for k in ("whisk", "jaw", "lick") if r[k]) or ("none" if kn else "unknown")
                     for r, kn in zip(d.to_dict("records"), known)]
    lo, hi = grid[a] - m1.WIDTH - MOVE_MARGIN, grid[E[:, 1]] + MOVE_MARGIN
    for nm, key in (("whisk_max", "whisker_speed"), ("jaw_max", "jaw")):
        v = np.full(len(E), np.nan)
        if key in dlc:
            t, x = dlc[key]
            if key == "jaw":
                x = np.abs(np.gradient(x, t))
            i0, i1 = np.searchsorted(t, lo), np.searchsorted(t, hi)
            v = np.array([np.nanmax(x[p:q]) if q > p and np.isfinite(x[p:q]).any() else np.nan for p, q in zip(i0, i1)])
        d[nm] = v
    tp = grid[E[:, 2]]
    lk = np.asarray(licks, float)
    if len(lk):
        j = np.clip(np.searchsorted(lk, tp), 1, len(lk) - 1) if len(lk) > 1 else np.zeros(len(tp), int)
        d["lick_dt"] = np.minimum(np.abs(tp - lk[np.maximum(j - 1, 0)]), np.abs(lk[j] - tp))
    else:
        d["lick_dt"] = np.nan
    on = np.sort(np.asarray(onsets, float))
    j = np.searchsorted(on, tp)
    d["dt_prev_onset"] = np.where(j > 0, tp - on[np.maximum(j - 1, 0)], np.nan)
    d["dt_next_onset"] = np.where(j < len(on), on[np.minimum(j, len(on) - 1)] - tp, np.nan)
    return d


def prestart_grid(grid, trials, licks):
    """windows inside the task quiet window of a trial ([abort_window_start_time, abort_window_stop_time]: the required no-lick
    period, Uniform(2, 5) s, ending at start_time); no lick or whisking criterion (user 2026-10-30); also returns the number of
    windows with a piezo lick inside (should be 0) and the number of trials"""
    T = trials[np.isin(trials.start_time.to_numpy(float), m1.task_events(trials).t.to_numpy(float))]   # kept active trials
    a0, a1 = T.abort_window_start_time.to_numpy(float), T.abort_window_stop_time.to_numpy(float)
    k = a1 > a0
    n_lick = int((np.searchsorted(licks, a0[k]) != np.searchsorted(licks, a1[k])).sum())
    pre = np.zeros(len(grid), bool)
    for lo, hi in zip(np.searchsorted(grid, a0[k], side="right"), np.searchsorted(grid, a1[k], side="right")):
        pre[lo:hi] = True                                           # window end (grid time) in (a0, a1]
    return pre, n_lick, int(k.sum())


def passive_grid(grid, trials):
    """windows inside a passive block: each run of consecutive passive trials (trial table context column, sorted by start time)
    from its first start_time + EXCL[0] to its last stop_time + EXCL[1] (user 2026-10-30)"""
    T = trials.sort_values("start_time")
    c = (T.context.astype(str).to_numpy() == "passive")
    out = np.zeros(len(grid), bool)
    b = np.flatnonzero(np.diff(np.r_[0, c.astype(int), 0]))
    for i, j in zip(b[::2], b[1::2]):
        lo, hi = T.start_time.iloc[i] + m1.EXCL[0], T.stop_time.iloc[j - 1] + m1.EXCL[1]
        out[np.searchsorted(grid, lo):np.searchsorted(grid, hi, side="right")] = True
    return out


def any_in(cs, E):
    """event (rows of E: start, end, peak) has a window in the mask whose cumulative sum (leading 0) is cs"""
    return (cs[E[:, 1] + 1] - cs[E[:, 0]]) > 0 if len(E) else np.zeros(0, bool)


def stim_onsets(trials):
    """onsets of every whisker / auditory stimulus (task, warm-up, perf 6 and passive trials)"""
    return np.sort(trials.loc[trials.trial_type.isin(["whisker_trial", "auditory_trial"]), "start_time"].to_numpy(float))


def evoked_grid(grid, onsets):
    """windows ending in (onset, onset + EVOKED_S] of a stimulus (evoked tail; v7)"""
    out = np.zeros(len(grid), bool)
    for lo, hi in zip(np.searchsorted(grid, onsets, side="right"), np.searchsorted(grid, onsets + EVOKED_S, side="right")):
        out[lo:hi] = True
    return out


def unit_moments(Xc, mu, sd):
    """per window: sum and sum of squares over units of the z-scored counts (correlation readout denominators)"""
    s1, s2 = np.zeros(len(Xc)), np.zeros(len(Xc))
    for i in range(Xc.shape[1]):
        zi = (Xc[:, i] - mu[i]) / sd[i]
        s1 += zi
        s2 += zi * zi
    return s1, s2


def corr_readout(Xc, TC, mu, sd, s1, s2):
    """Pearson correlation across units between each window's z-scored population vector and the centred template(s) TC
    (units x k, columns with mean 0): sum_u z_u tc_u / (||z - mean z|| ||tc||)"""
    TC = np.atleast_2d(TC.T).T
    num = Xc @ (TC / sd[:, None]).astype(np.float32) - ((mu / sd) @ TC).astype(np.float32)
    den = np.sqrt(np.maximum(s2 - s1 ** 2 / len(mu), 1e-12))[:, None] * np.linalg.norm(TC, axis=0)[None, :]
    return num / den.astype(np.float32)


def whisker_angvel(tw, a):
    """|angular velocity| (deg/s) of the DLC whisker angle (low-likelihood frames NaN): gaps <= AV_GAP s interpolated, AV_LP Hz
    low-pass (4th-order Butterworth, zero phase) on finite stretches >= AV_MIN_RUN s, |d angle / dt|; None without tracking"""
    from scipy.signal import butter, sosfiltfilt
    fs = 1 / np.median(np.diff(tw))
    fin = np.isfinite(a)
    if fin.sum() < fs * 10:
        return None
    ai = np.interp(tw, tw[fin], a[fin])
    d = np.diff(np.r_[0, (~fin).astype(np.int8), 0])
    for s0, s1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (s1 - s0) / fs > AV_GAP or s0 == 0 or s1 == len(a):
            ai[s0:s1] = np.nan
    sos = butter(4, AV_LP, "low", fs=fs, output="sos")
    ang = np.full(len(a), np.nan)
    d = np.diff(np.r_[0, np.isfinite(ai).astype(np.int8), 0])
    for s0, s1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if (s1 - s0) / fs >= AV_MIN_RUN:
            ang[s0:s1] = sosfiltfilt(sos, ai[s0:s1])
    return np.abs(np.gradient(ang, tw))


def window_mean(ts, v, ends, width=m1.WIDTH):
    """mean of the samples v(ts) in each readout window (ends - width, ends]; NaN without a finite sample"""
    ok = np.isfinite(v)
    cs, cn = np.r_[0, np.cumsum(np.where(ok, v, 0.0))], np.r_[0, np.cumsum(ok)]
    hi, lo = np.searchsorted(ts, ends, side="right"), np.searchsorted(ts, ends - width, side="right")
    n = cn[hi] - cn[lo]
    return np.where(n > 0, (cs[hi] - cs[lo]) / np.maximum(n, 1), np.nan)


def vel_design(V, fill):
    """regressors of the velocity partialling: intercept, |angular velocity|, log(1 + |angular velocity| / VEL_LOG0);
    windows without tracking get fill (the median velocity: no correction beyond the intercept)"""
    V = np.where(np.isfinite(V), V, fill)
    return np.c_[np.ones(len(V)), V, np.log1p(V / VEL_LOG0)]


def vel_fit(Xc, Dg, fit):
    """per-unit OLS of the counts on the velocity design over the fit windows: B (3 x units)"""
    idx = np.flatnonzero(fit)
    Df = Dg[idx]
    G = np.linalg.inv(Df.T @ Df)
    return np.stack([G @ (Df.T @ Xc[idx, i].astype(np.float64)) for i in range(Xc.shape[1])], 1)


def resid_moments(Xc, Dg, Bv, sd):
    """unit_moments of the velocity residuals (counts - Dg Bv, per unit) scaled by sd"""
    s1, s2 = np.zeros(len(Xc)), np.zeros(len(Xc))
    for i in range(Xc.shape[1]):
        zi = (Xc[:, i] - Dg @ Bv[:, i]) / sd[i]
        s1 += zi
        s2 += zi * zi
    return s1, s2


def corr_resid(Xc, TC, Dg, Bv, sd, s1, s2):
    """correlation readout of the velocity residuals: Pearson r across units between (counts - Dg Bv) / sd and TC"""
    TC = np.atleast_2d(TC.T).T
    W = (TC / sd[:, None])
    num = Xc @ W.astype(np.float32) - (Dg @ (Bv @ W)).astype(np.float32)
    den = np.sqrt(np.maximum(s2 - s1 ** 2 / Xc.shape[1], 1e-12))[:, None] * np.linalg.norm(TC, axis=0)[None, :]
    return num / den.astype(np.float32)


def corr_rows(Z, tc):
    """correlation of z-scored population vectors (... x units) with the centred template tc"""
    Zc = Z - Z.mean(-1, keepdims=True)
    return (Zc @ tc) / np.maximum(np.linalg.norm(Zc, axis=-1) * np.linalg.norm(tc), 1e-12)


def train_mask(ev, target):
    """v5 training events: target stimuli and no-stimulus events of the task (correct rejections) and of passive pre / post
    (shams)"""
    stim, sham = m1.TARGET_CLS[target]
    return (((ev.context == "task") & ev.cls.isin([stim, "N"])) | ((ev.context == "passive") & ev.cls.isin([stim, sham]))).to_numpy()


def bal_acc(y, oof):
    """balanced accuracy of the out-of-fold (outer nested-CV) decision values at 0 (classes balanced by sample weights)"""
    y, oof = np.asarray(y).astype(int), np.asarray(oof, float)
    if not ((y == 1).any() and (y == 0).any()):
        return np.nan
    return float((np.mean(oof[y == 1] > 0) + np.mean(oof[y == 0] <= 0)) / 2)


def snap(q):
    if not np.isfinite(q) or q > THR_F[-1]:
        return np.nan
    return float(THR_F[min(np.searchsorted(THR_F, q - 1e-9), len(THR_F) - 1)])


def has_passive(ev, target):
    stim = m1.TARGET_CLS[target][0]
    pas = (ev.context == "passive") & ev.cls.eq(stim)
    return all((pas & ev.epoch.eq(e)).sum() >= MIN_PER_BLOCK for e in ("passive_pre", "passive_post"))


def win_values(z, t, ev_t):
    """readout in the training window of each event: the 30-ms window ending TRAIN_WIN[1] after onset (first grid window
    ending at or after it, i.e. within one 5-ms step)"""
    return z[np.clip(np.searchsorted(t, np.asarray(ev_t, float) + m1.TRAIN_WIN[1] - 1e-6), 0, len(t) - 1)]


def auc(a, b):
    from sklearn.metrics import roc_auc_score
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


def thr_spec(src):
    """'shuf99.9' -> (99.9, 'shuf')"""
    s = src.rstrip("0123456789.")
    return float(src[len(s):]), s


def quantify(sid, g, target, G, pk, ev, sens, win=None, null_q=None, null_src="shuf"):
    """decoder row at one sensitivity level, from the saved counts and held-out peaks; win: training-window readout per event
    (held-out passive AUCs: target stimuli vs shams / vs the other modality's stimuli, pre, post, pre + post; user 2026-10-30);
    null_q / null_src: threshold = median over the shuffled templates ("shuf") or time shifts ("shift") of their null_q-th
    percentile"""
    ns = int(G["n_shuf"]) if "n_shuf" in G else 20                 # v5 files: 20 shuffles
    qs = [float(x) for x in G["null_qs"]] if "null_qs" in G else list(NULL_Q_V5)
    raw_u = "thr_q" in G                                            # v7 raw-unit files: exact thresholds and counts at them
    stim, sham = m1.TARGET_CLS[target]
    other = "A" if target == "whisker" else "W"
    pas = (ev.context == "passive").to_numpy()
    held = pas & ev.epoch.isin(["passive_pre", "passive_post"]).to_numpy()
    ok = has_passive(ev, target)
    zt = pk[held & ev.cls.eq(stim).to_numpy()]
    if null_q is None:                                              # held-out passive sensitivity
        thr, src = (snap(np.quantile(zt, 1 - sens)) if ok else np.nan), "passive"
    elif raw_u:                                                     # v7: exact threshold in raw units (saved per NULL_Q)
        thr = float(G["thr_q"][qs.index(null_q)]) if (null_q in qs and null_src == "shuf") else np.nan
        src, sens = f"{null_src}{null_q:g}", np.nan
    else:                                                           # null-based: median over the null traces of their percentile
        rows_ = slice(1, 1 + ns) if null_src == "shuf" else slice(1 + ns, None)
        ok_q = "null_q" in G and null_q in qs and len(G["null_q"][rows_])
        thr = snap(float(np.median(G["null_q"][rows_, qs.index(null_q)]))) if ok_q else np.nan
        src, sens = f"{null_src}{null_q:g}", np.nan
    md, sc_ = (float(G["med"]), float(G["sc"])) if "med" in G else (np.nan, np.nan)
    row = dict(session_id=sid, group=g, target=target, n_units=int(G["n_units"]), pauc=float(G["pauc"]),
               bacc=bal_acc(G["y"], G["oof"]) if "oof" in G else np.nan, n_train_stim=int(np.sum(G["y"] == 1)) if "y" in G else np.nan,
               n_train_N=int(np.sum(G["y"] == 0)) if "y" in G else np.nan, sens=sens, thr_source=src, has_passive=ok,
               thr_raw=thr if raw_u else md + thr * sc_,             # raw units: decision value (log-odds) / correlation r
               thr_norm=(thr - md) / sc_ if raw_u else thr,           # robust s.d. (1.4826 MAD) above the valid-time median
               units="raw" if raw_u else "MAD")
    if "train_idx" in G:                                            # out-of-fold quality per context (v5 training set)
        ti, y, oof = G["train_idx"], G["y"].astype(int), G["oof"]
        cx, epo = ev.context.to_numpy()[ti], ev.epoch.to_numpy()[ti]
        tk = cx == "task"
        row["bacc_task"] = bal_acc(y[tk], oof[tk])
        for es in ("pre", "post"):
            ep_ = epo == f"passive_{es}"
            row[f"bacc_passive_{es}"] = bal_acc(y[ep_], oof[ep_])
            mm = ep_ & (y == 1)
            row[f"tpr_passive_{es}"] = float(np.mean(oof[mm] > 0)) if mm.any() else np.nan
            row[f"n_train_passive_{es}"] = int(mm.sum())
    for es, eps in (("pre", ["passive_pre"]), ("post", ["passive_post"]), ("prepost", ["passive_pre", "passive_post"])):
        ins = pas & ev.epoch.isin(eps).to_numpy()
        for nm, cl in (("target", stim), ("sham", sham), ("other", other)):
            m = ins & ev.cls.eq(cl).to_numpy()
            row[f"cross_{nm}_{es}"] = float((pk[m] > thr).mean()) if (m.any() and np.isfinite(thr)) else np.nan
            if win is not None and nm != "target":
                row[f"auc_{nm}_{es}"] = auc(win[ins & ev.cls.eq(stim).to_numpy()], win[m])
            if win is not None:
                row[f"win_{nm}_{es}"] = float(np.median(win[m])) if m.any() else np.nan   # median training-window readout (MAD)
    if not np.isfinite(thr):
        return row
    if raw_u:
        C, secs = G["counts_q"][:, :, qs.index(null_q)], G["secs"]
    else:
        i = int(np.argmin(np.abs(THR_F - thr)))
        C, secs = G["counts"][:, :, i], G["secs"]
    real, shuf, nul = C[0], C[1:1 + ns], C[1 + ns:]
    for k, mk in enumerate(MASKS[:len(secs)]):                     # files written before a mask was added lack it
        if secs[k] < 30:
            continue
        sfx = "" if mk == "all" else f"_{mk}"
        r, s, n_ = real[k] / secs[k], shuf[:, k] / secs[k], nul[:, k] / secs[k]
        row[f"rate{sfx}"] = r
        row[f"rate_shuf{sfx}"] = float(s.mean())
        row[f"excess{sfx}"] = r - float(s.mean())
        row[f"ratio{sfx}"] = r / s.mean() if s.mean() > 0 else np.nan
        row[f"p_shuf{sfx}"] = float((1 + (s >= r).sum()) / (1 + ns))
        if len(n_):                                                 # time shifts (v5-v6 files only)
            row[f"null_shift{sfx}"] = float(n_.mean())
            row[f"excess_shift{sfx}"] = r - float(n_.mean())
            row[f"p_shift{sfx}"] = float((1 + (n_ >= r).sum()) / (1 + len(n_)))
        row[f"n_events{sfx}"] = int(real[k])
        row[f"secs{sfx}"] = float(secs[k])
        row[f"z_shuf{sfx}"] = (r - float(s.mean())) / float(s.std()) if s.std() > 0 else np.nan
    return row


def run(sid, groups, unit_set, force=False):
    t0 = time.time()
    out = ROOT / unit_set / sid
    out.mkdir(parents=True, exist_ok=True)
    done = lambda g: (out / f"g_{g.replace(' ', '_')}_auditory_corr.npz").exists()      # v7 files
    if "auto" not in groups and not force and all(done(g) for g in groups):
        print("SKIP", sid, "(done)", flush=True)
        return
    stage = stage_of(sid)
    D = m6.prepare(sid)
    ev, grid, valid, qg, spikes, units = D["ev"], D["grid"], D["valid"], D["quiet"], D["spikes"], D["units"]
    keep = unit_set_mask(sid, units, unit_set)
    spikes = [s for s, k in zip(spikes, keep) if k]
    units = units[keep].reset_index(drop=True)
    if "auto" in groups:                                                   # whole brain + every area group of the session
        groups = [m8.ALL] + sorted(units.area_group.dropna().astype(str).unique())
    todo = [g for g in groups if force or not done(g)]
    print(f"{sid}: {unit_set} units {keep.sum()} / {len(keep)}; groups to run: {todo}", flush=True)
    if not todo:
        print("SKIP", sid, "(done)", flush=True)
        return
    eg = m8.epoch_grid(grid, D["epochs"])
    ep = np.select([eg == "passive_pre", eg == "task", eg == "passive_post"], [1, 2, 3], 0).astype(np.int8)
    nb = max(int(np.ceil(BORDER_FRAC * len(grid))), int(np.ceil(m1.SHIFT_S[1] / m1.STEP))) if N_NULL else 0
    core = np.zeros(len(grid), bool)
    core[nb:len(grid) - nb] = True                                         # linear shift: lags <= nb never reach outside the data
    pas = passive_grid(grid, D["trials"])
    valid = valid & core & (grid <= D["t_hi"] - END_EXCL)                  # v7: passive blocks count; last END_EXCL s cut (end artefact)
    vidx = np.flatnonzero(valid)
    pre, n_pre_lick, n_pre = prestart_grid(grid, D["trials"], D["licks"])
    jaw_lag, jaw_r = m8.lick_jaw_sync(D["licks"], D["dlc"])
    jaw_ok = bool(np.isfinite(jaw_r) and abs(jaw_lag) <= m8.JAW_LAG_MAX and jaw_r >= m8.JAW_R_MIN)
    mv, mv_info = movement_grids(grid, D["licks"], D["dlc"], valid, jaw_ok)
    vel = av = None                                                        # whisker |angular velocity| per readout window
    if "whisker_angle" in D["dlc"]:
        tw_, wa_ = D["dlc"]["whisker_angle"]
        av = whisker_angvel(tw_, wa_)
        if av is not None:
            vel = window_mean(tw_, av, grid)
    s_on = stim_onsets(D["trials"])
    evk = evoked_grid(grid, s_on)
    cs_evk = np.r_[0, np.cumsum(evk, dtype=np.int64)]
    mv["still"] = mv["still"] & ~evk                                       # v7: still = no movement and outside the evoked tail
    mv_info["frac_still"] = float(mv["still"][valid].mean()) if valid.any() else np.nan
    mv_info["frac_evoked"] = float(evk[valid].mean()) if valid.any() else np.nan
    mv_info["frac_passive"] = float(pas[valid].mean()) if valid.any() else np.nan
    T_ = D["trials"]
    in_trial = np.zeros(len(grid), bool)                                   # within a trial: start_time .. stop_time
    for lo_, hi_ in zip(np.searchsorted(grid, T_.start_time.to_numpy(float)), np.searchsorted(grid, T_.stop_time.to_numpy(float), side="right")):
        in_trial[lo_:hi_] = True
    nomove = mv["whisk_known"] & ~mv["whisk"] & ~mv["jaw"]
    qw = pre & ~mv["lick"]
    iti = (eg == "task") & ~in_trial & ~evk
    cond_m = dict(qw=qw, qw_still=qw & nomove, iti=iti, iti_still=iti & mv["still"])
    tune = {}
    for key, x, edges in (("wz", mv["wz"], Z_EDGES), ("jz", mv["jz"], Z_EDGES), ("lk", mv["lick_dt"], LICK_EDGES)):
        fin = ~np.isnan(x)                                                # no tracking -> no bin (lick time: inf = no lick)
        bi = np.where(fin, np.digitize(np.nan_to_num(x, nan=0.0, posinf=1e9), edges[1:-1]), -1)
        for i in range(TUNE[key]):
            tune[f"{key}{i}"] = (bi == i) & ~evk
    M = np.stack([np.ones_like(valid), mv["still"], mv["whisk"], mv["jaw"], mv["lick"], pre, evk, pas, ~evk]
                 + [cond_m[c] for c in COND] + [tune[k] for k in MASKS[9 + len(COND):]]).astype(bool)
    secs = (M & valid).sum(1) * m1.STEP
    cond_g = np.full(len(grid), "none", dtype=object)                      # most specific condition of every window (first window)
    for c in ("iti", "iti_still", "qw", "qw_still"):
        cond_g[cond_m[c]] = c
    onsets = np.sort(D["trials"].start_time.to_numpy(float))
    np.savez_compressed(out / "session.npz", t=grid, valid=valid, quiet=qg, epoch=ep, prestart=pre, evoked=evk, passive=pas, **mv)
    np.savez_compressed(out / "behaviour.npz", licks=D["licks"], ev_t=ev.t.to_numpy(), ev_cls=ev.cls.to_numpy(str),
                        ev_context=ev.context.to_numpy(str), **{f"{k}_t": v[0] for k, v in D["dlc"].items()},
                        **{f"{k}_v": v[1] for k, v in D["dlc"].items()})
    (out / "params.json").write_text(json.dumps(dict(
        session_id=sid, stage=stage, unit_set=unit_set, n_units_set=int(keep.sum()), n_units_all=len(keep), sens_levels=SENS_LEVELS, sens_main=SENS_MAIN, min_per_block=MIN_PER_BLOCK, n_shuf=N_SHUF,
        n_null=N_NULL, n_strata=SHUF_STRATA, shuf_scale=SHUF_SCALE, null_q=NULL_Q, thr_main=THR_MAIN, readouts=READOUTS, soft_c_frac=SOFT_C_FRAC, evoked_s=EVOKED_S, n_perm=m7.N_PERM, thr_grid=[float(THR_F[0]), float(THR_F[-1]), 0.1], masks=MASKS,
        train=TRAIN, train_win=m1.TRAIN_WIN, step=m1.STEP, excl=m1.EXCL, start_excl=m6.START_EXCL,
        shift_s=m1.SHIFT_S, merge=m1.MERGE, det_win=m1.DET_WIN, quiet_gap=m1.QUIET_GAP, move_k=MOVE_K, z_edges=Z_EDGES[1:-1].tolist(), lick_edges=LICK_EDGES[1:-1].tolist(),
        scaling="continuous valid time", passive_s=float(pas.sum() * m1.STEP), border_frac=BORDER_FRAC, border_s=nb * m1.STEP, n_prestart=n_pre, n_prestart_lick=n_pre_lick, move_margin=MOVE_MARGIN, lick_margin=LICK_MARGIN, mask_mode=MASK_MODE.tolist(), **mv_info, secs=dict(zip(MASKS, secs.tolist())), jaw_lag=jaw_lag, jaw_r=jaw_r, jaw_ok=jaw_ok,
        has_passive={t: has_passive(ev, t) for t in m6.TARGETS}, epochs=D["epochs"],
        date=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=str))
    rng = np.random.default_rng(zlib.crc32(f"{sid}|full".encode()))
    ends_peri = ev.t.to_numpy()[:, None] + m1.PERI[None, :]
    i_tr = int(np.argmin(np.abs(m1.PERI - m1.TRAIN_WIN[1])))
    after = np.flatnonzero((m1.PERI >= m1.DET_WIN[0]) & (m1.PERI <= m1.DET_WIN[1]))
    for g in todo:
        sel = np.arange(len(units)) if g == m8.ALL else np.flatnonzero(units.area_group.to_numpy() == g)
        if len(sel) < 10:
            continue
        tg = time.time()
        S = [spikes[i] for i in sel]
        P = np.stack([m1.counts(s, ends_peri.ravel()).reshape(ends_peri.shape) for s in S], 2).astype(np.float32)
        Xc = np.empty((len(grid), len(S)), np.float32, order="F")             # column-major: per-unit shifts are contiguous
        for i, s in enumerate(S):
            Xc[:, i] = m1.counts(s, grid)
        mu_c = np.array([Xc[vidx, i].mean(dtype=np.float64) for i in range(len(S))])
        sd_c = np.array([Xc[vidx, i].std(dtype=np.float64) for i in range(len(S))])
        sd_c[sd_c == 0] = 1.0                                               # silent in valid time: no scaling
        dec = []
        for target in m6.TARGETS:
            tm = train_mask(ev, target)
            y = ev.cls.eq(m1.TARGET_CLS[target][0]).to_numpy()[tm].astype(int)
            ctx = ev.context.to_numpy()[tm].astype(str)
            wt = m1.weights(y, ctx)                                        # classes balanced within task and within passive
            _, oof, _, final = m1.nested(P[tm, i_tr].astype(float), y, wt, np.char.add(ctx, y.astype(str)), rng, scale=(mu_c, sd_c))
            a, c = m1.lin(final)
            dec.append(dict(target=target, a=a, c=c, model=final, oof=oof, y=y, n_train=int(tm.sum()), train_idx=np.flatnonzero(tm)))
        t_fit = time.time() - tg
        A = np.stack([d["a"] for d in dec], 1).astype(np.float32)
        Cv = np.array([d["c"] for d in dec], np.float32)
        raw = Xc @ A + Cv
        R_ev = np.einsum("etu,ud->etd", P, A) + Cv
        sd_s = sd_c + SOFT_C_FRAC * float(np.median(sd_c))                  # correlation readout: soft-normalised s.d.
        s1, s2 = unit_moments(Xc, mu_c, sd_s)                              # correlation readout denominators
        Bv = None                                                          # partial correlation | whisker velocity
        if vel is not None:
            fitv = valid & ~evk & np.isfinite(vel)
            if fitv.sum() > 10000:
                vfill = float(np.median(vel[fitv]))
                Dg = vel_design(vel, vfill)
                Bv = vel_fit(Xc, Dg, fitv)
                s1v, s2v = resid_moments(Xc, Dg, Bv, sd_s)
                Dp = vel_design(window_mean(tw_, av, ends_peri[:, after].ravel()), vfill).reshape(len(ev), len(after), 3)
        T = np.zeros((N_NULL, len(grid), len(dec)), np.float32)
        for j in range(N_NULL):
            lag = (rng.uniform(*m1.SHIFT_S, len(S)) / m1.STEP).astype(int) * rng.choice([-1, 1], len(S))
            for i in range(len(S)):
                r_ = np.roll(Xc[:, i], lag[i])                             # wrapped part lies outside core: linear shift
                for k in range(len(dec)):
                    T[j, :, k] += A[i, k] * r_
            T[j] += Cv
        t_null = time.time() - tg - t_fit
        peaks = ev[["t", "cls", "context", "epoch", "quiet"]].copy()
        gname = g.replace(" ", "_")

        def emit(target, sfx, z, null_trace, pk, extra, ident):
            """one readout (sfx "" = decoder, "_corr" = correlation, "_pcorr" = partial correlation), raw units (decision value / r):
            percentiles of the real and the N_SHUF null traces -> exact thresholds (median over the shuffles of their NULL_Q-th
            percentile); counts at the exact thresholds and on a rate-vs-threshold grid; saved files; events at the main
            threshold with their movement / stimulus flags; continuous trace"""
            med, sc = m1.robust(z[valid])                               # reference only (grid of the rate-vs-threshold curves)
            traces = lambda: itertools.chain([z], (null_trace(i) for i in range(N_SHUF)))
            null_q = np.stack([np.percentile(x[vidx], NULL_Q) for x in traces()])          # real, then the shuffles
            thr_q = np.maximum.accumulate(np.median(null_q[1:], 0))                      # exact thresholds, one per NULL_Q
            thr_grid = med + THR_F * sc                                                  # 1-30 robust s.d. above the median
            counts, counts_q = [], []
            for x in traces():
                counts.append(count_curve(x, vidx, M, thr=thr_grid))
                counts_q.append(count_curve(x, vidx, M, thr=thr_q))
            counts, counts_q = np.stack(counts), np.stack(counts_q)
            peaks[f"peak_{target}{sfx}"] = pk
            G = dict(counts=counts, counts_q=counts_q, thr_q=thr_q, secs=secs, n_units=len(S), pauc=extra["pauc"], y=extra["y"],
                     oof=extra["oof"], null_q=null_q, train_idx=extra["train_idx"], n_shuf=N_SHUF, null_qs=np.array(NULL_Q), med=med, sc=sc)
            np.savez_compressed(out / f"g_{gname}_{target}{sfx}.npz", counts=counts, counts_q=counts_q, thr_q=thr_q, secs=secs,
                                thr=thr_grid, masks=np.array(MASKS), n_shuf=N_SHUF, null_qs=np.array(NULL_Q), units="raw", z=z,
                                med=med, sc=sc, null_q=null_q, n_units=len(S), unit_index=sel,
                                cluster_id=units.cluster_id.to_numpy(str)[sel], electrode_group=units.electrode_group.to_numpy(str)[sel],
                                **extra)
            q = quantify(sid, g, target, G, pk, ev, np.nan, win_values(z, grid, ev.t), *thr_spec(THR_MAIN))
            thr = q["thr_raw"]
            E = m1.detect(z, valid, thr) if np.isfinite(thr) else np.zeros((0, 3), int)
            p_id = ident(E) if ident is not None else np.full(len(E), np.nan)
            tp = grid[E[:, 2]]
            j = np.searchsorted(s_on, tp)
            pd.concat([pd.DataFrame(dict(session_id=sid, group=g, target=target, readout=RNAME[sfx], t_peak=tp,
                                         peak=z[E[:, 2]], t_start=grid[E[:, 0]] - m1.WIDTH, t_end=grid[E[:, 1]],
                                         dur_ms=(E[:, 1] - E[:, 0] + 1) * m1.STEP * 1000, prestart=pre[E[:, 0]],
                                         epoch=eg[E[:, 0]].astype(str), passive=pas[E[:, 0]], evoked=any_in(cs_evk, E),
                                         condition=cond_g[E[:, 0]].astype(str),
                                         dt_prev_stim=np.where(j > 0, tp - s_on[np.maximum(j - 1, 0)], np.nan), p_identity=p_id)),
                       event_movement(E, grid, mv, D["dlc"], D["licks"], onsets)], axis=1).to_parquet(
                out / f"events_{target}_{gname}{sfx}.parquet")
            np.savez_compressed(out / f"continuous_{target}_{gname}{sfx}.npz", t=grid, z=z, valid=valid, quiet=qg, thr=thr, units="raw",
                                med=med, sc=sc, thr_source=PREFIX[sfx] + THR_MAIN, pauc=extra["pauc"],
                                unit_index=sel, **{k_: extra[k_] for k_ in ("w", "mu", "sd", "template") if k_ in extra})

        for k, d in enumerate(dec):
            fin, target = d["model"], d["target"]
            base = dict(pauc=m1.pauc(d["y"], d["oof"]), y=d["y"], oof=d["oof"], train_idx=d["train_idx"], n_train=d["n_train"])
            st = rate_strata(fin["mu"], SHUF_STRATA)
            # ---- decoder readout: raw decision value (log-odds); nulls = the same decoder with weights permuted within rate
            # quintiles (same intercept and unit scaling), on the same data
            z = raw[:, k].astype(np.float32)
            perm = np.stack([m7.perm_within(fin["w"], st, rng) for _ in range(N_SHUF)], 1)
            TW = Xc @ (perm / fin["sd"][:, None]).astype(np.float32)
            off = (fin["b"] - (perm / fin["sd"][:, None]).T @ fin["mu"]).astype(np.float32)   # intercepts (as m1.lin)
            jb = after[np.argmax(R_ev[:, after, k], 1)]
            pk = R_ev[np.arange(len(ev)), jb, k]
            Wp = np.stack([m7.perm_within(fin["w"], st, rng) for _ in range(m7.N_PERM)], 1)
            emit(target, "", z, lambda i: TW[:, i] + off[i], pk, dict(base, w=fin["w"], b=fin["b"], mu=fin["mu"], sd=fin["sd"], C=fin["C"]),
                 lambda E: m7.identity_p((Xc[E[:, 2]] - fin["mu"]) / fin["sd"], fin["w"], Wp))
            del TW
            # ---- correlation readout (v7): Pearson r across units between the soft-normalised population vector ((x - mu) /
            # (sd + c), SOFT_C_FRAC) and the template = mean soft-normalised training-window pattern of the target stimuli - of the
            # no-stimulus training events; nulls = the template permuted within the same rate quintiles; saved sd = sd + c
            Zt = (P[d["train_idx"], i_tr] - mu_c) / sd_s
            tmpl = Zt[d["y"] == 1].mean(0) - Zt[d["y"] == 0].mean(0)
            tc = tmpl - tmpl.mean()
            r = corr_readout(Xc, tc[:, None], mu_c, sd_s, s1, s2)[:, 0]
            permT = np.stack([m7.perm_within(tc, st, rng) for _ in range(N_SHUF)], 1)
            RS = corr_readout(Xc, permT, mu_c, sd_s, s1, s2)
            pkr = np.empty(len(ev))
            for c0 in range(0, len(ev), 200):                              # peak correlation 10-100 ms after each event
                pkr[c0:c0 + 200] = corr_rows((P[c0:c0 + 200][:, after] - mu_c) / sd_s, tc).max(1)
            cx = dict(base, template=tmpl, tc=tc, mu=mu_c, sd=sd_s, sd_raw=sd_c, soft_c=SOFT_C_FRAC * float(np.median(sd_c)))
            emit(target, "_corr", r.astype(np.float32), lambda i: RS[:, i], pkr, cx, None)
            # ---- partial correlation | whisker velocity (user 2026-11-02): each unit's counts minus its linear prediction from
            # the whisker |angular velocity| of the same window (vel_design, per-unit OLS over valid time outside the evoked
            # tail), soft-normalised as the correlation readout, Pearson r with the same template; nulls = the same shuffles
            if Bv is not None:
                PR = corr_resid(Xc, np.c_[tc, permT], Dg, Bv, sd_s, s1v, s2v)
                pkp = np.empty(len(ev))
                for c0 in range(0, len(ev), 200):                          # held-out peaks on the velocity residuals
                    Pr = P[c0:c0 + 200][:, after] - Dp[c0:c0 + 200] @ Bv
                    pkp[c0:c0 + 200] = corr_rows(Pr / sd_s, tc).max(1)
                emit(target, "_pcorr", PR[:, 0].copy(), lambda i: PR[:, 1 + i], pkp,
                     dict(cx, vel_beta=Bv, vel=vel.astype(np.float32), vel_fill=vfill, vel_log0=VEL_LOG0, partial="whisker angular velocity",
                          vel_r2=float(np.nanmean([np.corrcoef(Dg[fitv] @ Bv[:, i], Xc[fitv, i])[0, 1] ** 2 for i in range(0, Xc.shape[1], 10)]))), None)
                del PR
            del RS
        peaks.to_parquet(out / f"peaks_{g.replace(' ', '_')}.parquet")
        del Xc, P, raw, R_ev, T
        print(f"{sid} {g}: {len(S)} units | fit {t_fit:.0f} s, null {t_null:.0f} s, total {time.time() - tg:.0f} s", flush=True)
    decoders_table(sid, unit_set).to_csv(out / "decoders.csv", index=False)
    print("DONE", sid, f"{time.time() - t0:.0f} s", flush=True)


def decoders_table(sid, unit_set, sens_levels=SENS_LEVELS):
    """every decoder of a session at the given sensitivity levels, from the saved files (no refit)"""
    out = ROOT / unit_set / sid
    Pm = json.loads((out / "params.json").read_text())
    rows = []
    t = np.load(out / "session.npz")["t"]
    for f in sorted(out.glob("peaks_*.parquet")):
        g = f.stem[len("peaks_"):].replace("_", " ")
        ev = pd.read_parquet(f)
        for target, sfx in [(t_, s_) for t_ in m6.TARGETS for s_ in READOUTS]:
            gf = out / f"g_{g.replace(' ', '_')}_{target}{sfx}.npz"
            if not gf.exists() or f"peak_{target}{sfx}" not in ev:
                continue
            G = dict(np.load(gf))
            win = win_values(G["z"], t, ev.t)
            for src, q in [(s_, q_) for s_ in NULL_SRC for q_ in NULL_Q]:   # v5+: passive stimuli are training data, no passive threshold
                r = quantify(sid, g, target, G, ev[f"peak_{target}{sfx}"].to_numpy(), ev, np.nan, win, q, src)
                r["thr_source"] = PREFIX[sfx] + r["thr_source"]
                rows.append(dict(r, readout=RNAME[sfx], stage=Pm["stage"], mouse_id=sid[:5], jaw_ok=Pm["jaw_ok"],
                                 valid_s=Pm["secs"]["all"], jaw_used=Pm.get("jaw_used"), whisk_known_frac=Pm.get("whisk_known_frac"),
                                 **{f: Pm.get(f) for f in ("frac_still", "frac_whisk", "frac_jaw", "frac_lick", "frac_evoked", "frac_passive")}))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ summary
def summary(unit_set, groups=None):
    plt = m2.setup()
    col = m2.colors()
    S = sessions_table()
    S.to_csv(m1.HOME / "full_sessions.csv", index=False)
    done = [p.parent.name for p in (ROOT / unit_set).glob("*/params.json")]
    Dd = pd.concat([decoders_table(s, unit_set) for s in done], ignore_index=True)
    Dd = Dd.drop(columns=["stage"]).merge(S[["session_id", "stage", "cohort", "day"]], on="session_id", how="left")
    Dd.to_csv(m1.HOME / f"full_decoders_{unit_set}.csv", index=False)
    order = m8.GROUPS + sorted(set(Dd.group) - set(m8.GROUPS))
    G = [g for g in order if g in set(Dd.group) and (groups is None or "auto" in groups or g in groups)]
    fdir = m1.HOME / "figures" / f"full_{unit_set}"
    metrics = METRICS
    stats = []
    for tgt in m6.TARGETS:
        q0 = Dd[(Dd.target == tgt) & (Dd.thr_source == THR_MAIN) & (Dd.bacc >= GOOD_BACC)]
        Gt = [g for g in G if (q0.group == g).sum() >= 2 * MIN_SESS]                # groups with >= 6 usable sessions
        fig, axs = plt.subplots(len(Gt), len(metrics), figsize=(7.4, 1.0 * len(Gt) + 0.8), squeeze=False)
        for i, g in enumerate(Gt):
            cc = col.get(g, "k") if g != m8.ALL else "k"
            q = q0[q0.group == g]
            ns = [int((q.stage == s_).sum()) for s_ in STAGES]
            for j, (mt, lab) in enumerate(metrics):
                ax = axs[i, j]
                for x_, stg in enumerate(STAGES):
                    v = q[q.stage == stg][mt].to_numpy(float)
                    jit = np.random.default_rng(x_).uniform(-0.15, 0.15, len(v))
                    ax.scatter(x_ + jit, v, s=4, color=COH[stg[-2:]], lw=0, alpha=0.8, zorder=3)
                    if np.isfinite(v).any():
                        ax.plot([x_ - 0.28, x_ + 0.28], [np.nanmedian(v)] * 2, color="k", lw=0.9, zorder=4)
                for stg in ("day 0", "expert"):
                    a_, b_ = q[q.stage == f"{stg} R+"][mt].dropna(), q[q.stage == f"{stg} R-"][mt].dropna()
                    tested = len(a_) >= MIN_SESS and len(b_) >= MIN_SESS              # per-cohort inclusion (>= 3 sessions)
                    p_mw, p_t = m8.two_tests(a_, b_) if tested else (np.nan, np.nan)
                    stats.append(dict(target=tgt, group=g, metric=mt, stage=stg, n_rplus=len(a_), n_rminus=len(b_), tested=tested,
                                      median_rplus=a_.median(), median_rminus=b_.median(), p_mannwhitney=p_mw, p_welch=p_t))
                if mt.startswith("ratio"):
                    ax.axhline(1, color="0.6", lw=0.4, ls=":"); ax.set_yscale("log")
                elif mt.startswith("excess"):
                    ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.set_yscale("symlog", linthresh=LINTHRESH)
                elif mt.startswith("z_shuf"):
                    ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.set_yscale("symlog", linthresh=2)
                else:
                    ax.set_yscale("symlog", linthresh=LINTHRESH)
                ax.set_xticks(range(len(STAGES)), [s_.replace(" ", "\n", 2) for s_ in STAGES] if i == len(Gt) - 1 else [], fontsize=4)
                ax.set_xlim(-0.6, len(STAGES) - 0.4)
                ax.tick_params(labelsize=4.2)
                if i == 0:
                    ax.set_title(lab, fontsize=5)
            axs[i, 0].set_ylabel(f"{'whole brain' if g == m8.ALL else m2.short(g)}\nn {'/'.join(map(str, ns))}", fontsize=4.6, color=cc)
        St = pd.DataFrame([r for r in stats if r["target"] == tgt])
        St["q_mannwhitney"], St["q_welch"] = np.nan, np.nan
        for (mt, stg), k in St[St.tested].groupby(["metric", "stage"]).groups.items():      # BH across groups
            for c in ("mannwhitney", "welch"):
                St.loc[k, f"q_{c}"] = bh(St.loc[k, f"p_{c}"].to_numpy())
        for i, g in enumerate(Gt):
            for j, (mt, _) in enumerate(metrics):
                r = St[(St.group == g) & (St.metric == mt)].set_index("stage")
                txt = [f"{s_[:3]} {r.loc[s_, 'p_mannwhitney']:.2g}/{r.loc[s_, 'p_welch']:.2g}" + ("*" if r.loc[s_, "p_mannwhitney"] < 0.05
                       and r.loc[s_, "p_welch"] < 0.05 else "") for s_ in ("day 0", "expert") if r.loc[s_, "tested"]]
                axs[i, j].text(0.01, 0.99, "\n".join(txt), transform=axs[i, j].transAxes, va="top", fontsize=3.4)
        stats = [r for r in stats if r["target"] != tgt] + St.to_dict("records")
        fig.suptitle(f"{tgt} decoders, full dataset ({unit_set} units), {THR_TEXT}; one point per "
                     f"session (balanced accuracy >= {GOOD_BACC}); n = day 0 R+ / R- / expert R+ / R-\nbar: median; text: "
                     f"R+ vs R- p Mann-Whitney / Welch where both cohorts have >= {MIN_SESS} sessions; * both p < 0.05 (uncorrected); "
                     f"excess = real rate - mean of {N_SHUF} shuffled templates (weights permuted within {SHUF_STRATA} firing-rate quantiles); " + STILL_NOTE, x=0.01, ha="left", fontsize=5.4, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (1.0 * len(Gt) + 0.8)))
        m2.save(fig, fdir, f"fig17_cohort_rates_{tgt}")
        plt.close(fig)
    pd.DataFrame(stats).to_csv(m1.HOME / f"full_cohort_stats_{unit_set}.csv", index=False)
    # real vs shuffled, all sessions, one figure per target
    nc = 5
    nr = int(np.ceil(len(G) / nc))
    for tgt in m6.TARGETS:
        fig, axs = plt.subplots(nr, nc, figsize=(7.4, 1.55 * nr + 0.4), squeeze=False)
        for i, g in enumerate(G):
            ax = axs[i // nc, i % nc]
            q = Dd[(Dd.group == g) & (Dd.target == tgt) & (Dd.thr_source == THR_MAIN)]
            for coh, cl in COH.items():
                mm = q[q.cohort == coh]
                ax.scatter(mm.rate_shuf_still.clip(lower=1e-4), mm.rate_still.clip(lower=1e-4), s=5, lw=0.4, edgecolors=cl,
                           facecolors=[cl if p >= GOOD_BACC else "w" for p in mm.bacc], label=coh if i == 0 else None)
            lim = (1e-4, 5)
            ax.plot(lim, lim, color="0.6", lw=0.4, ls=":"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
            ax.set_title(f"{'whole brain' if g == m8.ALL else m2.short(g)} (n {len(q)})", fontsize=5,
                         color=col.get(g, "k") if g != m8.ALL else "k")
            ax.tick_params(labelsize=4.2)
            if i == 0:
                ax.legend(frameon=False, fontsize=4.5)
            if i % nc == 0:
                ax.set_ylabel("real (/ s)", fontsize=5)
            if i // nc == nr - 1 or i + nc >= len(G):
                ax.set_xlabel("shuffled (/ s)", fontsize=5)
        for k in range(len(G), nr * nc):
            axs[k // nc, k % nc].axis("off")
        fig.suptitle(f"{tgt} decoders: still-event rate, real vs shuffled template (weights permuted within {SHUF_STRATA} rate quantiles) at the main threshold ({THR_MAIN}), one point per session "
                     f"(filled: balanced accuracy >= {GOOD_BACC})", x=0.01, ha="left", fontsize=5.6, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        m2.save(fig, fdir, f"fig18_full_real_vs_shuffled_{tgt}")
        plt.close(fig)
    pd.set_option("display.width", 260)
    m = Dd[(Dd.thr_source == THR_MAIN)]
    print("sessions done", len(done), "| with passive (whisker)", m[(m.target == "whisker")].drop_duplicates("session_id").has_passive.sum())
    effects_vs_bacc(unit_set)
    print(m[(m.bacc >= GOOD_BACC)].groupby(["target", "group", "stage"])[["rate", "rate_still", "excess_still",
                                                                                         "excess", "rate_prestart"]].median().round(3).to_string())


TGT_COLS = [("win_target_pre", "pre: median readout\n(MAD, in-sample)"), ("win_target_post", "post: median readout\n(MAD, in-sample)"),
            ("win_target_delta", "post - pre\n(MAD units)"), ("cross_target_pre", "pre: fraction above\nthreshold (in-sample)"),
            ("cross_target_post", "post: fraction above\nthreshold (in-sample)")]
AUC_COLS = [("bacc_task", "task: balanced acc.\n(out-of-fold)"), ("bacc_passive_pre", "passive pre: stimulus vs sham\nbalanced acc. (out-of-fold)"),
            ("bacc_passive_post", "passive post: stimulus vs sham\nbalanced acc. (out-of-fold)"), ("auc_other_pre", "pre: vs other modality\n(AUC, 5-35 ms)"),
            ("auc_other_post", "post: vs other modality\n(AUC, 5-35 ms)")]
EFFECTS = [("rate", "events / s, all valid"), ("rate_still", "events / s, still"), ("excess", "all valid: excess vs\nshuffled templates (/ s)"),
           ("excess_still", "still: excess vs\nshuffled templates (/ s)")]


def symt(y):
    """symlog transform for fits and axes (linear below LINTHRESH)"""
    y = np.asarray(y, float)
    return np.sign(y) * np.log10(1 + np.abs(y) / LINTHRESH)


def ols_band(x, y, xs):
    """OLS fit, 95 % CI of the mean, slope p"""
    from scipy import stats
    n = len(x)
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    s2 = res @ res / (n - 2)
    sxx = ((x - x.mean()) ** 2).sum()
    se = np.sqrt(s2 * (1 / n + (xs - x.mean()) ** 2 / sxx))
    tq = stats.t.ppf(0.975, n - 2)
    p = 2 * stats.t.sf(abs(b / np.sqrt(s2 / sxx)), n - 2)
    return a + b * xs, tq * se, p


def ancova_p(d, mt):
    """OLS metric ~ 1 + cohort (R+) + bacc on the symlog scale: p of cohort and of bacc"""
    from scipy import stats
    X = np.c_[np.ones(len(d)), (d.cohort == "R+").to_numpy(float), d.bacc.to_numpy(float)]
    y = symt(d[mt])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = (res @ res / dof) * np.linalg.inv(X.T @ X)
    tt = beta / np.sqrt(np.diag(cov))
    return 2 * stats.t.sf(abs(tt[1]), dof), 2 * stats.t.sf(abs(tt[2]), dof)


def effects_vs_bacc(unit_set):
    """fig27_effects_vs_bacc_<stage>: per session, each effect vs the decoder's out-of-fold balanced accuracy (whole brain), per
    cohort: scatter, OLS line (symlog scale) + 95 % CI (solid if slope p < 0.05), Spearman rho; ANCOVA metric ~ cohort + bacc
    (p cohort | bacc, p bacc); table full_effects_vs_bacc_<unit_set>.csv"""
    from scipy import stats
    plt = m2.setup()
    Dd = pd.read_csv(m1.HOME / f"full_decoders_{unit_set}.csv")
    q0 = Dd[(Dd.group == m8.ALL) & (Dd.thr_source == THR_MAIN)]
    rows = []
    for stg in ("day 0", "expert"):
        fig, axs = plt.subplots(len(m6.TARGETS), len(EFFECTS), figsize=(7.4, 3.9), squeeze=False)
        for i, tg in enumerate(m6.TARGETS):
            q = q0[(q0.target == tg) & q0.stage.str.startswith(stg)]
            for j, (mt, lab) in enumerate(EFFECTS):
                ax = axs[i, j]
                txt = []
                for coh, cl in COH.items():
                    d = q[q.cohort == coh][["bacc", mt]].dropna()
                    x, y = d.bacc.to_numpy(float), symt(d[mt])
                    ax.scatter(x, y, s=5, color=cl, lw=0, alpha=0.8, zorder=3)
                    if len(d) >= MIN_SESS:
                        xs = np.linspace(x.min(), x.max(), 50)
                        yh, ci, p = ols_band(x, y, xs)
                        ax.plot(xs, yh, color=cl, lw=0.8, ls="-" if p < 0.05 else "--")
                        ax.fill_between(xs, yh - ci, yh + ci, color=cl, alpha=0.15, lw=0)
                        rho, prho = stats.spearmanr(x, d[mt])
                        txt.append(f"{coh} rho {rho:.2f} (p {prho:.2g}), OLS p {p:.2g}")
                        rows.append(dict(stage=stg, target=tg, metric=mt, cohort=coh, n=len(d), spearman_rho=rho, spearman_p=prho, ols_p=p))
                d = q[["bacc", mt, "cohort"]].dropna()
                if min((d.cohort == c).sum() for c in COH) >= MIN_SESS:
                    p_coh, p_b = ancova_p(d, mt)
                    txt.append(f"ANCOVA: cohort | bacc p {p_coh:.2g}, bacc p {p_b:.2g}")
                    rows.append(dict(stage=stg, target=tg, metric=mt, cohort="both", n=len(d), ancova_p_cohort=p_coh, ancova_p_bacc=p_b))
                ax.text(0.01, 0.99, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=3.4)
                ax.axvline(GOOD_BACC, color="0.6", lw=0.4, ls=":")
                tk = [v for v in (-1, -0.1, -0.01, 0, 0.01, 0.1, 1, 10) if not (mt.startswith("rate") and v < 0)]
                ax.set_yticks(symt(tk), [f"{v:g}" for v in tk])
                ax.tick_params(labelsize=4.2)
                if i == 0:
                    ax.set_title(lab, fontsize=5)
                if i == len(m6.TARGETS) - 1:
                    ax.set_xlabel("decoder balanced accuracy (out-of-fold)", fontsize=4.6)
            axs[i, 0].set_ylabel(f"{tg} decoder, whole brain\nn {len(q)}", fontsize=5)
        fig.suptitle(f"{stg}: effects vs decoder quality, whole-brain decoders ({unit_set} units), one point per session; {THR_TEXT}; "
                     f"line: OLS on the symlog scale + 95 % CI (solid: slope p < 0.05); dotted: good-decoder cut {GOOD_BACC}; "
                     f"ANCOVA metric ~ cohort + balanced accuracy, uncorrected", x=0.01, ha="left", fontsize=5.2, weight="bold", wrap=True)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        m2.save(fig, m1.HOME / "figures" / f"full_{unit_set}", f"fig27_effects_vs_bacc_{stg.replace(' ', '')}")
        plt.close(fig)
    R = pd.DataFrame(rows)
    R.to_csv(m1.HOME / f"full_effects_vs_bacc_{unit_set}.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False))


def passive_auc_fig(unit_set):
    """fig24: held-out passive decoding by the whole-brain decoders (training-window readout), one point per session;
    fig25: the decoder's readout on its own held-out passive target stimuli (whisker decoder on passive whisker stimuli, auditory
    on auditory), pre, post, post - pre, and the fraction above the event threshold (peak 10-100 ms; user 2026-10-30)"""
    plt = m2.setup()
    Dd = pd.read_csv(m1.HOME / f"full_decoders_{unit_set}.csv")
    q0 = Dd[(Dd.group == m8.ALL) & (Dd.thr_source == THR_MAIN)].copy()
    q0["win_target_delta"] = q0.win_target_post - q0.win_target_pre
    for cols, fname, ttl in ((AUC_COLS, "fig24_decoder_quality", "out-of-fold quality (balanced accuracy: task stimulus vs correct rejection, "
                              "passive stimulus vs sham) and specificity (AUC of the 5-35 ms readout, target vs the other modality's passive stimuli)"),
                             (TGT_COLS, "fig25_passive_target_readout", "readout on the decoder's own held-out passive target stimuli "
                              "(whisker decoder: passive whisker stimuli; auditory: auditory): median training-window readout (5-35 ms) "
                              f"and fraction whose peak (10-100 ms) crosses the event threshold ({THR_MAIN}); passive stimuli are training data")):
        passive_panels(plt, q0, cols, fname, ttl, unit_set)
    pd.set_option("display.width", 220)
    print(q0.groupby(["target", "stage"])[[c for c, _ in AUC_COLS + TGT_COLS]].median().round(3).to_string())


def passive_panels(plt, q0, cols, fname, ttl, unit_set):
    auc_fig = cols is AUC_COLS
    fig, axs = plt.subplots(len(m6.TARGETS), len(cols), figsize=(7.4 * len(cols) / 4, 3.6), squeeze=False)
    for i, tgt in enumerate(m6.TARGETS):
        q = q0[q0.target == tgt]
        for j, (mt, lab) in enumerate(cols):
            ax = axs[i, j]
            for x_, stg in enumerate(STAGES):
                s_ = q[q.stage == stg]
                jit = np.random.default_rng(x_).uniform(-0.15, 0.15, len(s_))
                ax.scatter(x_ + jit, s_[mt], s=5, lw=0.4, edgecolors=COH[stg[-2:]],
                           facecolors=[COH[stg[-2:]] if p >= GOOD_BACC else "w" for p in s_.bacc], zorder=3)
                if s_[mt].notna().any():
                    ax.plot([x_ - 0.28, x_ + 0.28], [s_[mt].median()] * 2, color="k", lw=0.9, zorder=4)
            txt = []
            for stg in ("day 0", "expert"):
                a_, b_ = q[q.stage == f"{stg} R+"][mt].dropna(), q[q.stage == f"{stg} R-"][mt].dropna()
                if len(a_) >= MIN_SESS and len(b_) >= MIN_SESS:
                    p_mw, p_t = m8.two_tests(a_, b_)
                    txt.append(f"{stg[:3]} {p_mw:.2g}/{p_t:.2g}")
            ax.text(0.01, 0.02, "\n".join(txt), transform=ax.transAxes, fontsize=3.8)
            if auc_fig or mt.startswith("cross"):
                ax.axhline(0.5, color="0.6", lw=0.4, ls=":")
                ax.set_ylim(-0.03, 1.03)
            else:
                ax.axhline(0, color="0.6", lw=0.4, ls=":")
            ax.set_xticks(range(len(STAGES)), [s_.replace(" ", "\n", 2) for s_ in STAGES] if i == len(m6.TARGETS) - 1 else [], fontsize=4)
            ax.set_xlim(-0.6, len(STAGES) - 0.4)
            ax.tick_params(labelsize=4.2)
            if i == 0:
                ax.set_title(lab, fontsize=5)
        axs[i, 0].set_ylabel(f"{tgt} decoder, whole brain", fontsize=5)
    fig.suptitle(f"passive stimuli, whole-brain decoders ({unit_set} units): {ttl}; one point per session (filled: balanced accuracy "
                 f">= {GOOD_BACC}), bar: median; text: R+ vs R- p Mann-Whitney / Welch, uncorrected", x=0.01, ha="left", fontsize=5.2,
                 weight="bold", wrap=True)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    m2.save(fig, m1.HOME / "figures" / f"full_{unit_set}", fname)
    plt.close(fig)


def day0_panel(ax, q_all, q_good, mt):
    """one panel: R+ vs R- for all and good decoders (one point per session); returns the stats rows and the p text"""
    rows, txt = [], []
    for k, (sn, q) in enumerate((("all decoders", q_all), (f"balanced accuracy >= {GOOD_BACC}", q_good))):
        a_, b_ = q[q.cohort == "R+"][mt].dropna(), q[q.cohort == "R-"][mt].dropna()
        for x_, (v, coh) in enumerate(((a_, "R+"), (b_, "R-"))):
            x = 2.6 * k + x_
            jit = np.random.default_rng(x_).uniform(-0.15, 0.15, len(v))
            ax.scatter(x + jit, v, s=4, color=COH[coh], lw=0, alpha=0.8, zorder=3)
            if len(v):
                ax.plot([x - 0.28, x + 0.28], [v.median()] * 2, color="k", lw=0.9, zorder=4)
        tested = len(a_) >= MIN_SESS and len(b_) >= MIN_SESS                       # per-cohort inclusion (>= 3 sessions)
        p_mw, p_t = m8.two_tests(a_, b_) if tested else (np.nan, np.nan)
        rows.append(dict(decoders=sn, metric=mt, n_rplus=len(a_), n_rminus=len(b_), tested=tested, median_rplus=a_.median(),
                         median_rminus=b_.median(), mean_rplus=a_.mean(), mean_rminus=b_.mean(), p_mannwhitney=p_mw, p_welch=p_t))
        if tested:
            txt.append(f"{'all' if k == 0 else 'good'} {p_mw:.2g}/{p_t:.2g}" + ("*" if p_mw < 0.05 and p_t < 0.05 else ""))
    if mt.startswith("excess"):
        ax.axhline(0, color="0.6", lw=0.4, ls=":")
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.text(0.01, 0.99, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=4)
    ax.set_xticks([0, 1, 2.6, 3.6], [f"{c}\n{s_}" for s_ in ("all", "good") for c in ("R+", "R-")], fontsize=4.2)
    ax.set_xlim(-0.6, 4.2)
    ax.tick_params(labelsize=4.2)
    return rows


def day0_cohorts(unit_set):
    """day 0 R+ vs R-, all decoders and good decoders (balanced accuracy >= GOOD_BACC), uncorrected p (user 2026-10-29): whole brain
    (fig19, rows = targets) and every area group with >= MIN_SESS sessions per cohort among all decoders (fig20, one per target)"""
    plt = m2.setup()
    col = m2.colors()
    Dd = pd.read_csv(m1.HOME / f"full_decoders_{unit_set}.csv")
    q0 = Dd[(Dd.thr_source == THR_MAIN) & Dd.stage.str.startswith("day 0")]
    note = (f"{THR_TEXT}; one point per session; all decoders vs balanced accuracy >= "
            f"{GOOD_BACC}\nbar: median; text: p Mann-Whitney / Welch, uncorrected (shown where both cohorts have >= {MIN_SESS} sessions); "
            f"* both p < 0.05; excess = real rate - mean of {N_SHUF} shuffled templates (weights permuted within {SHUF_STRATA} firing-rate quantiles); " + STILL_NOTE)
    fdir = m1.HOME / "figures" / f"full_{unit_set}"
    stats = []
    panels = [("whole brain", [(tgt, m8.ALL) for tgt in m6.TARGETS], "fig19_day0_wholebrain_cohorts")]
    order = [g for g in m8.GROUPS + sorted(set(Dd.group) - set(m8.GROUPS)) if g != m8.ALL]
    for tgt in m6.TARGETS:
        q = q0[q0.target == tgt]
        G = [g for g in order if min((q[q.group == g].cohort == c).sum() for c in COH) >= MIN_SESS]
        panels.append((f"{tgt}, per area group", [(tgt, g) for g in G], f"fig20_day0_areas_{tgt}"))
    for title, rows_, name in panels:
        fig, axs = plt.subplots(len(rows_), len(METRICS), figsize=(7.4, 1.25 * len(rows_) + 0.7), squeeze=False)
        for i, (tgt, g) in enumerate(rows_):
            q = q0[(q0.target == tgt) & (q0.group == g)]
            qg = q[q.bacc >= GOOD_BACC]
            for j, (mt, lab) in enumerate(METRICS):
                stats += [dict(r, target=tgt, group=g) for r in day0_panel(axs[i, j], q, qg, mt)]
                if i == 0:
                    axs[i, j].set_title(lab, fontsize=5)
            lab_g = "whole brain" if g == m8.ALL else m2.short(g)
            axs[i, 0].set_ylabel(f"{tgt if g == m8.ALL else lab_g}\nn all {len(q)} / good {len(qg)}", fontsize=4.8,
                                 color="k" if g == m8.ALL else col.get(g, "k"))
        fig.suptitle(f"day 0, R+ vs R-, {title} decoders ({unit_set} units); {note}", x=0.01, ha="left", fontsize=5.2, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 1 - 0.55 / (1.25 * len(rows_) + 0.7)))
        m2.save(fig, fdir, name)
        plt.close(fig)
    St = pd.DataFrame(stats).drop_duplicates(["target", "group", "decoders", "metric"])
    St = St[["target", "group"] + [c for c in St.columns if c not in ("target", "group")]]
    St.to_csv(m1.HOME / f"full_day0_cohort_stats_{unit_set}.csv", index=False)
    pd.set_option("display.width", 260)
    print(St[St.tested].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="*", default=[])
    ap.add_argument("--list", action="store_true", help="print the included sessions, largest first")
    ap.add_argument("--groups", nargs="*", default=["auto"], help="'auto': whole brain + every area group of the session")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--unit-set", choices=UNIT_SETS, default="stable")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--figures", nargs="*", default=[], help="sessions: 008 continuous + example figures from the saved files")
    ap.add_argument("--day0", "--day0-wholebrain", dest="day0", action="store_true", help="day 0 R+ vs R-, whole brain and per area group (all and good decoders)")
    a = ap.parse_args()
    if a.list:
        S = sessions_table()
        for s in S[S.include].session_id:
            print(s)
    for s in a.session:
        try:
            run(s, a.groups, a.unit_set, a.force)
        except Exception:
            print("FAILED", s, flush=True)
            traceback.print_exc()
    for s in a.figures:
        fd = m1.HOME / "figures" / f"full_{a.unit_set}"
        units, *_ = m1.load_session(s)
        units = units[unit_set_mask(s, units, a.unit_set)].reset_index(drop=True)      # unit_index refers to this set
        for tgt in m6.TARGETS:
            m8.fig_continuous(s, tgt, out=ROOT / a.unit_set / s, fig_dir=fd / "continuous")
            m8.fig_examples(s, tgt, out=ROOT / a.unit_set / s, fig_dir=fd / "examples", units=units)
    if a.summary:
        summary(a.unit_set, a.groups)
        passive_auc_fig(a.unit_set)
    if a.day0:
        day0_cohorts(a.unit_set)
