"""001 -- Per-session stimulus-presence template and continuous readout (pilot, user 2026-10-23).

One session, one target (whisker: decoder trained on whisker vs no stimulus; auditory: control, auditory vs no stimulus), one
decoder per key area group at a fixed number of neurons.

Training events (question.md decisions 2026-10-18 / 10-23):
  task trials (detection-project rules: active context, perf != 6, warm-up cut, rule A1): every stimulus trial of the target
    (hit or miss) vs correct rejections; passive trials (pre + post): target stimuli vs sham onsets SHAM s before the other
    modality's stimulus (as ssl-stimulus-detection-decoding 001). Quiet events only: no lick in QUIET_LICK before onset.
  Features: spike counts of N_UNITS neurons (random subsample of the group's good + mua units, seeded) in TRAIN_WIN after onset,
    z-scored with the training-event mean / s.d. Spikes: artefact window -10..+5 ms at every whisker onset and every event onset
    replaced by Poisson spikes (roc_utils correct_neuron_spike_train), as in the detection project.
  Classes balanced within context (task / passive) with sample weights, so the decoder cannot use task-vs-passive state.
  L2 logistic regression; C chosen by the standardised partial AUC up to FPR = MAX_FPR (low false positives), nested
  stratified cross-validation (outer OUTER folds for held-out scores, inner INNER folds for C); final model refit on all events.

Continuous readout: window of TRAIN_WIN's length ending at t, step STEP, over the whole recording; readout = decision value
(w . z + b), normalised per trace as (x - median) / (1.4826 MAD) over the valid time (data and every null
separately). Detection excludes EXCL around every trial onset and passive sham (stimulus responses; used as validation).

Session-own nulls (question.md): (1) weight shuffle: decoder weights permuted across neurons (N_NULL permutations);
(2) linear time shift: every neuron's spike train shifted independently by a random lag (SHIFT_S, both signs, non-wrapping;
evaluated on the time range covered by every shifted train), N_NULL repeats. Threshold = the larger of the two thresholds
at which a null's mean event rate falls to NULL_RATE (events / min) [first pilot]. Threshold (2026-10-24): one per area group,
in normalised readout units, the same for every session and template, chosen by 004 on the pooled pilot sessions
(thresholds_<target>.json; THR_DEFAULT until then). Event = run above threshold (gaps < MERGE merged), time = peak.

Validation: out-of-fold readout around held-out training events (-200..600 ms): fraction of target stimuli detected
(crossing the threshold with the window end within DET_WIN), latency; false detections on held-out no-stimulus events;
other events (other modality, licked / unlicked) read out with the final model. Each fold model's readout is normalised by
its own continuous median / MAD. Other-modality stimuli crossing are reported, not penalised.
--apply-only: recompute the threshold-dependent outputs (validation.csv, events.parquet, detected) from the saved readouts.

Outputs: <RES>/pilot/<session>/<target>/<model>/ (params.json, models.npz, peri_event.parquet, continuous_<group>.npz, events.parquet,
null_curves.parquet, validation.csv, figures/fig1..fig4).
Run (haas): cd ~/code/unit_spikes_analysis && PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities ./.venv/bin/python \
            .../001_session_reactivation.py --session MH030_20250503_154256 --target whisker
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE), str(REPO / "scripts"), str(pathlib.Path.home() / "code" / "unit_spikes_analysis")]
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
HOME = RES / "ssl-reactivation-decoding"
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
UNITS = RES / "_roc_stage_analysis" / "units.parquet"

GROUPS = ["Somatosensory-whisker", "Thalamus", "Midbrain", "Striatum", "Motor areas", "Auditory areas"]
N_UNITS = 0                                # 0 = every unit of the group (user 2026-10-23; the first pilot used 50)
TRAIN_WIN = (0.005, 0.035)                 # s after onset
WIDTH = TRAIN_WIN[1] - TRAIN_WIN[0]
STEP = 0.005
QUIET_LICK = 2.0                           # s without lick before a training event
SHAM = 0.8
OUTER, INNER = 5, 4
C_GRID = 1.0 / np.array([1e-3, 1e-2, 1e-1, 1, 10, 100])
MAX_FPR = 0.05
SD_FLOOR = 0.2                             # spikes per window: ~ a 1.3-Hz Poisson neuron in 30 ms
EXCL = (-0.1, 0.6)                         # s around every trial onset and sham: excluded from detection
N_NULL = 20
SHIFT_S = (30.0, 300.0)                    # |lag| range of the linear time shifts
NULL_RATE = 0.1                            # events / min (first pilot only)
THR_DEFAULT = 6.0                          # provisional per-group threshold until 004 writes thresholds_<target>.json
SPONT_PER_STIM = 3                         # model "spont": spontaneous negative windows per target stimulus
QUIET_GAP = 1.0                            # quiet continuous time: > 1 s from any lick ...
WHISK_MAX = 200.0                          # ... and whisker speed < 200 deg/s
MERGE = 0.05                               # s
PERI = np.round(np.arange(-0.2, 0.6 + 1e-9, STEP), 6)   # window ends relative to onset
DET_WIN = (0.010, 0.100)                   # detection of a held-out stimulus: a window ending in this range crosses
LIK = 0.5
TARGET_CLS = {"whisker": ("W", "SA"), "auditory": ("A", "SW")}   # (stimulus class, passive sham class)


# ------------------------------------------------------------------------------------------------ events
def resolve_context(t):
    st = t["start_time"].to_numpy()
    dp, dn = np.r_[np.inf, np.diff(st)], np.r_[np.diff(st), np.inf]
    fixed = (np.abs(dp - 3.0) < 0.3) | (np.abs(dn - 3.0) < 0.3)
    return pd.Series(np.where(fixed | (t["context"].astype(str).to_numpy() == "passive"), "passive", "active"), index=t.index)


def task_events(trials):
    """detection-project rules; class W / A / N (CR); lick flag kept (hit / miss)"""
    t = trials.sort_values("start_time").reset_index(drop=True)
    t = t[(resolve_context(t) == "active").to_numpy()].reset_index(drop=True)
    t = t[t["perf"] != 6].reset_index(drop=True)
    wi = np.where(t.trial_type.to_numpy() == "whisker_trial")[0]
    if len(wi):
        t = t.iloc[max(0, wi[0] - 1):].reset_index(drop=True)
    licked = np.where(t.lick_flag.to_numpy() == 1)[0]
    if len(licked):
        tail = t.iloc[licked[-1] + 1:]
        if (tail.trial_type == "whisker_trial").sum() >= 5 and (tail.trial_type == "auditory_trial").sum() >= 1:
            t = t.iloc[:licked[-1] + 1].reset_index(drop=True)
    tt, lf = t.trial_type.to_numpy(), t.lick_flag.to_numpy()
    cls = np.select([tt == "whisker_trial", tt == "auditory_trial", (tt == "no_stim_trial") & (lf == 0),
                     (tt == "no_stim_trial") & (lf == 1)], ["W", "A", "N", "FA"], "")
    k = cls != ""
    return pd.DataFrame(dict(t=t.start_time.to_numpy()[k], cls=cls[k], licked=lf[k].astype(bool), context="task"))


def passive_events(trials):
    t = trials.sort_values("start_time").reset_index(drop=True)
    t = t[(resolve_context(t) == "passive").to_numpy()]
    t = t[t.trial_type.isin(["whisker_trial", "auditory_trial"])]
    st, w = t.start_time.to_numpy(), (t.trial_type == "whisker_trial").to_numpy()
    ev = pd.DataFrame(dict(t=np.r_[st, st - SHAM], cls=np.r_[np.where(w, "W", "A"), np.where(w, "SW", "SA")],
                           licked=False, context="passive"))
    return ev.sort_values("t").reset_index(drop=True)


def quiet(times, licks, win=QUIET_LICK):
    return np.searchsorted(licks, times) == np.searchsorted(licks, times - win)


# ------------------------------------------------------------------------------------------------ data
def load_session(sid):
    from pynwb import NWBHDF5IO
    from roc_analysis import roc_utils_new as ru
    U = pd.read_parquet(UNITS, columns=["session_id", "electrode_group", "cluster_id", "quality_label", "area_group",
                                        "area_acronym_custom"])
    U = U[(U.session_id == sid) & U.quality_label.isin(["good", "mua"])].drop_duplicates(["electrode_group", "cluster_id"])
    with NWBHDF5IO(str(NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb, apply_artifact_correction=False)
        trials = nwb.trials.to_dataframe()
        beh = nwb.processing["behavior"].data_interfaces
        licks = np.sort(np.asarray(beh["BehavioralEvents"].time_series["piezo_lick_times"].data[:], float))
        ts = beh["BehavioralTimeSeries"].time_series

        def get(k):
            if k not in ts:
                return None, None
            t, d = np.asarray(ts[k].timestamps[:], float), np.asarray(ts[k].data[:], float)
            if len(t) != len(d):
                if len(t) and abs(len(t) - len(d)) <= 0.005 * len(d):
                    n = min(len(t), len(d)); t, d = t[:n], d[:n]
                else:
                    return None, None
            return t, d
        tw, wa = get("whisker_angle"); _, wbl = get("whisker_base_likelihood")
        tj, jd = get("jaw_distance")
        epochs = {}
        ep = beh.get("BehavioralEpochs")
        if ep is not None:
            for k, s in ep.interval_series.items():
                d, tt = np.asarray(s.data[:]), np.asarray(s.timestamps[:], float)
                epochs[k] = [(a, b) for a, b in zip(tt[d > 0], tt[d < 0])] if (d < 0).any() else [(tt[0], tt[-1])]
    units["cluster_id"] = units.cluster_id.astype(str)
    U["cluster_id"] = U.cluster_id.astype(str)
    units = units.merge(U[["electrode_group", "cluster_id", "area_group", "area_acronym_custom"]], on=["electrode_group", "cluster_id"],
                        how="inner", validate="one_to_one")
    dlc = {}
    if tw is not None:
        a = wa.copy()
        if wbl is not None and len(wbl) == len(a):
            a[wbl < LIK] = np.nan
        dlc["whisker_speed"] = (tw, np.abs(np.gradient(a, tw)))
        dlc["whisker_angle"] = (tw, a)                                   # deg, low-likelihood frames NaN (009 angular velocity)
    if tj is not None:
        dlc["jaw"] = (tj, jd - np.nanmedian(jd))
    return units, trials, licks, dlc, epochs, ru


def counts(st, ends, width=WIDTH):
    return np.searchsorted(st, ends) - np.searchsorted(st, ends - width)


# ------------------------------------------------------------------------------------------------ decoder
def folds(strata, k, rng):
    f = np.empty(len(strata), int)
    for s in np.unique(strata):
        i = rng.permutation(np.where(strata == s)[0])
        f[i] = np.arange(len(i)) % k
    return f


def weights(y, ctx):
    w = np.ones(len(y))
    for c in np.unique(ctx):
        m = ctx == c
        for v in (0, 1):
            mm = m & (y == v)
            if mm.any():
                w[mm] = m.sum() / (2 * mm.sum())
    return w / w.mean()


def fit(X, y, w, C, scale=None):
    from sklearn.linear_model import LogisticRegression
    # s.d. floor (2026-10-23): a neuron silent in the training windows would otherwise get sd ~ 0 and turn one continuous-data
    # spike into a huge z value (harmless for the fitted decoder, which gives it ~0 weight, but not for the weight-shuffle null)
    # scale = (mu, sd): fixed per-unit scaling (009: continuous valid time) instead of the training-event mean / s.d.
    mu, sd = scale if scale is not None else (X.mean(0), np.maximum(X.std(0), SD_FLOOR))
    m = LogisticRegression(C=C, penalty="l2", solver="liblinear", max_iter=2000).fit((X - mu) / sd, y, sample_weight=w)
    return dict(w=m.coef_[0], b=float(m.intercept_[0]), mu=mu, sd=sd, C=C)


def lin(model):
    """readout = X @ a + c"""
    a = model["w"] / model["sd"]
    return a.astype(np.float32), float(model["b"] - np.dot(model["mu"], a))


def score(model, X):
    return ((X - model["mu"]) / model["sd"]) @ model["w"] + model["b"]


def pauc(y, s):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, s, max_fpr=MAX_FPR) if len(np.unique(y)) == 2 else np.nan


def choose_C(X, y, w, strata, rng, scale=None):
    fo = folds(strata, INNER, rng)
    best, bv = C_GRID[0], -np.inf
    for C in C_GRID:
        s = np.empty(len(y))
        for k in range(INNER):
            tr = fo != k
            s[~tr] = score(fit(X[tr], y[tr], w[tr], C, scale), X[~tr])
        v = pauc(y, s)
        if v > bv:
            best, bv = C, v
    return best


def nested(X, y, w, strata, rng, scale=None):
    """outer folds: held-out scores and fold models (C chosen in the inner folds); final model on all events"""
    fo = folds(strata, OUTER, rng)
    oof, models = np.empty(len(y)), []
    for k in range(OUTER):
        tr = fo != k
        m = fit(X[tr], y[tr], w[tr], choose_C(X[tr], y[tr], w[tr], strata[tr], rng, scale), scale)
        oof[~tr] = score(m, X[~tr]); models.append(m)
    final = fit(X, y, w, choose_C(X, y, w, strata, rng, scale), scale)
    return fo, oof, models, final


# ------------------------------------------------------------------------------------------------ events / nulls
def detect(trace, valid, thr, step=STEP):
    """runs of trace > thr inside valid (gaps < MERGE merged); returns (start_idx, end_idx, peak_idx)"""
    above = (trace > thr) & valid
    if not above.any():
        return np.zeros((0, 3), int)
    i = np.where(above)[0]
    br = np.where(np.diff(i) > max(1, int(round(MERGE / step))))[0]
    st, en = np.r_[i[0], i[br + 1]], np.r_[i[br], i[-1]]
    pk = np.array([s + np.argmax(trace[s:e + 1]) for s, e in zip(st, en)])
    return np.c_[st, en, pk]


def rate_curve(trace, valid, thrs):
    mins = valid.sum() * STEP / 60
    return np.array([len(detect(trace, valid, t)) / mins for t in thrs])


THRS = np.r_[np.linspace(1, 10, 181), np.linspace(10.25, 40, 120)]    # normalised readout units


def robust(x):
    med = float(np.median(x))
    s = 1.4826 * float(np.median(np.abs(x - med)))
    return med, (s if s > 0 else float(np.std(x)) + 1e-9)


def norm(x, valid):
    med, s = robust(x[valid])
    return (x - med) / s


def null_threshold(curves, thrs):
    """smallest threshold with mean null rate <= NULL_RATE"""
    m = np.mean(curves, 0)
    ok = np.where(m <= NULL_RATE)[0]
    return thrs[ok[0]] if len(ok) else thrs[-1]


# ------------------------------------------------------------------------------------------------ main
def model_weights(y, ctx):
    """task / passive: classes balanced within context (weights()); spont negatives (model "spont"): together as heavy as the
    event negatives; positives then scaled so both classes weigh the same overall"""
    sp = ctx == "spont"
    w = np.ones(len(y))
    w[~sp] = weights(y[~sp], ctx[~sp])
    if sp.any():
        neg_ev = w[(y == 0) & ~sp].sum()
        w[sp] = neg_ev / sp.sum()
        w[y == 1] *= w[y == 0].sum() / w[y == 1].sum()
    return w / w.mean()


def quiet_grid(grid, licks, dlc):
    """quiet continuous time: > QUIET_GAP s from any lick and whisker speed < WHISK_MAX (DLC, when available)"""
    i = np.clip(np.searchsorted(licks, grid), 1, len(licks) - 1)
    d = np.minimum(np.abs(grid - licks[i - 1]), np.abs(licks[i] - grid))
    q = d > QUIET_GAP
    if "whisker_speed" in dlc:
        t, v = dlc["whisker_speed"]
        q &= np.interp(grid, t, np.nan_to_num(v, nan=np.inf)) < WHISK_MAX
    return q


def run(sid, target, model="clean", n_units=0):
    """model "clean": quiet target stimuli vs correct rejections + passive shams (question.md 2026-10-18); model "spont":
    the same plus random spontaneous windows of the continuous data (outside trial windows, any state) as negatives
    (user 2026-10-23). n_units 0 = every unit of the group (user 2026-10-23). Fitting and readouts are threshold-free;
    apply_thresholds() then uses one threshold per area group, the same for every session and template (user 2026-10-24)."""
    t0 = time.time()
    out = HOME / "pilot" / sid / target / model
    (out / "figures").mkdir(parents=True, exist_ok=True)
    units, trials, licks, dlc, epochs, ru = load_session(sid)
    ev = pd.concat([task_events(trials), passive_events(trials)], ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["quiet"] = quiet(ev.t.to_numpy(), licks)
    stim, sham = TARGET_CLS[target]
    ev["train"] = ev.quiet & (((ev.context == "task") & ev.cls.isin([stim, "N"])) |
                              ((ev.context == "passive") & ev.cls.isin([stim, sham])))
    ev["y"] = ev.cls.eq(stim).astype(int)
    # artefact replacement at every whisker onset and every event onset (as the detection project)
    wh = trials.loc[trials.whisker_stim == 1, "start_time"].to_numpy()
    onsets = np.unique(np.r_[wh, ev.t.to_numpy()])
    rng0 = np.random.default_rng(zlib.crc32(f"{sid}|react".encode()))
    spikes = [ru.correct_neuron_spike_train(np.asarray(st), onsets, rng0) for st in units.spike_times]
    t_lo = min(float(np.min(s)) for s in spikes if len(s)); t_hi = max(float(np.max(s)) for s in spikes if len(s))
    grid = np.round(np.arange(t_lo + WIDTH, t_hi, STEP), 6)
    excl_t = np.r_[trials.start_time.to_numpy(), ev.loc[ev.cls.isin(["SW", "SA"]), "t"].to_numpy()]
    valid = np.ones(len(grid), bool)
    for a in np.sort(excl_t):
        valid[np.searchsorted(grid, a + EXCL[0]):np.searchsorted(grid, a + EXCL[1])] = False
    qg = quiet_grid(grid, licks, dlc)
    vq = valid & qg
    rng = np.random.default_rng(zlib.crc32(f"{sid}|{target}|{model}|react".encode()))
    tr_ev = ev[ev.train].reset_index(drop=True)
    tr_t, y_ev, ctx_ev = tr_ev.t.to_numpy(), tr_ev.y.to_numpy(), tr_ev.context.to_numpy().astype(object)
    if model == "spont":                     # window ends drawn from the valid continuous time, any state
        n_sp = SPONT_PER_STIM * int(y_ev.sum())
        ends = np.sort(rng.choice(grid[valid], n_sp, replace=False))
        sp_onset = ends - TRAIN_WIN[1]
    else:
        sp_onset = np.zeros(0)
    t_all = np.r_[tr_t, sp_onset]
    y = np.r_[y_ev, np.zeros(len(sp_onset), int)]
    ctx = np.r_[ctx_ev, np.full(len(sp_onset), "spont", object)].astype(str)
    is_ev = np.r_[np.ones(len(tr_t), bool), np.zeros(len(sp_onset), bool)]
    params = dict(session_id=sid, target=target, model=model, n_units="all" if n_units == 0 else n_units, train_win=TRAIN_WIN,
                  step=STEP, quiet_lick=QUIET_LICK, sham=SHAM, outer=OUTER, inner=INNER, c_grid=C_GRID.tolist(), max_fpr=MAX_FPR,
                  sd_floor=SD_FLOOR, spont_per_stim=SPONT_PER_STIM if model == "spont" else 0, excl=EXCL,
                  quiet_gap=QUIET_GAP, whisk_max=WHISK_MAX, n_null=N_NULL, shift_s=SHIFT_S, merge=MERGE, det_win=DET_WIN,
                  n_train={**{f"{c}|{k}": int(v) for (c, k), v in tr_ev.groupby(["context", "cls"]).size().items()},
                           "spont": int(len(sp_onset))},
                  recording=(t_lo, t_hi), valid_min=float(valid.sum() * STEP / 60), quiet_min=float(vq.sum() * STEP / 60),
                  epochs=epochs, groups={})
    peri_rows, fit_rows, null_rows = [], [], []
    ends_peri = ev.t.to_numpy()[:, None] + PERI[None, :]
    tpos = {t: i for i, t in enumerate(ev.t.to_numpy())}
    after = (PERI >= DET_WIN[0]) & (PERI <= DET_WIN[1])
    for g in GROUPS:
        idx = np.where(units.area_group.to_numpy() == g)[0]
        n_use = len(idx) if n_units == 0 else n_units
        if len(idx) < max(n_use, 10):
            params["groups"][g] = f"skipped: {len(idx)} units"
            continue
        sel = np.sort(rng.choice(idx, n_use, replace=False))
        S = [spikes[i] for i in sel]
        X = np.stack([counts(s, t_all + TRAIN_WIN[1]) for s in S], 1).astype(float)
        w = model_weights(y, ctx)
        fo, oof, fmodels, final = nested(X, y, w, np.char.add(ctx, y.astype(str)), rng)
        # readouts as Xc @ a + c (a = w / sd, c = b - mu / sd . w): continuous, peri-event and nulls without a z-scored copy
        Xc = np.stack([counts(s, grid) for s in S], 1).astype(np.float32)
        P = np.stack([counts(s, ends_peri.ravel()).reshape(ends_peri.shape) for s in S], 2).astype(np.float32)   # ev x t x unit
        a_f, c_f = lin(final)
        raw = Xc @ a_f + c_f
        # continuous readout, normalised per trace: (x - median) / (1.4826 MAD) over the valid time (2026-10-23)
        med, scale = robust(raw[valid])
        trace = (raw - med) / scale
        del raw
        # peri-event readout: final model for every event, then held-out fold models for the training events, each fold model
        # normalised by the median / MAD of its own continuous readout (fix 2026-10-24: was the final model's)
        R = (P @ a_f + c_f - med) / scale
        for k, m in enumerate(fmodels):
            rows = [tpos[t] for t in tr_t[(fo[is_ev] == k)]]
            a_k, c_k = lin(m)
            med_k, sc_k = robust((Xc @ a_k + c_k)[valid])
            R[rows] = (P[rows] @ a_k + c_k - med_k) / sc_k
        del P
        pk = R[:, after].max(1)
        thrs = THRS
        perm = np.stack([rng.permutation(final["w"]) for _ in range(N_NULL)], 1)
        TW = Xc @ (perm / final["sd"][:, None]).astype(np.float32) + (final["b"] - (final["mu"] / final["sd"]) @ perm).astype(np.float32)
        del Xc
        c_ws = np.stack([rate_curve(norm(TW[:, k], valid), valid, thrs) for k in range(N_NULL)])
        del TW
        c_ts, c_tsq = [], []
        for k in range(N_NULL):
            lag = rng.uniform(*SHIFT_S, len(S)) * rng.choice([-1, 1], len(S))
            lo_ok, hi_ok = t_lo + np.clip(lag, 0, None).max(), t_hi + np.clip(lag, None, 0).min()
            Xs = np.stack([counts(s + l, grid) for s, l in zip(S, lag)], 1).astype(np.float32)
            ts_tr = Xs @ a_f + c_f
            del Xs
            vk = valid & (grid > lo_ok + WIDTH) & (grid < hi_ok)
            ts_n = norm(ts_tr, vk)
            c_ts.append(rate_curve(ts_n, vk, thrs)); c_tsq.append(rate_curve(ts_n, vk & qg, thrs))
        c_ts, c_tsq = np.stack(c_ts), np.stack(c_tsq)
        c_real, c_realq = rate_curve(trace, valid, thrs), rate_curve(trace, vq, thrs)
        for j, th in enumerate(thrs):
            null_rows.append(dict(group=g, thr=th, real=c_real[j], real_quiet=c_realq[j], null_weight=c_ws[:, j].mean(),
                                  null_weight_hi=np.quantile(c_ws[:, j], 0.95), null_shift=c_ts[:, j].mean(),
                                  null_shift_hi=np.quantile(c_ts[:, j], 0.95), null_shift_quiet=c_tsq[:, j].mean()))
        for i, r in ev.iterrows():
            peri_rows.append(dict(group=g, ev=i, t=r.t, cls=r.cls, context=r.context, licked=r.licked, quiet=r.quiet,
                                  train=r.train, y=r.y, peak=float(pk[i])))
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y[is_ev], oof[is_ev])
        fit_rows.append(dict(group=g, model=model, n_units=len(S), n_available=len(idx), C=final["C"],
                             pauc=pauc(y[is_ev], oof[is_ev]), auc=float(np.trapezoid(tpr, fpr)),
                             sens_fpr01=float(np.interp(0.01, fpr, tpr)), sens_fpr05=float(np.interp(0.05, fpr, tpr)),
                             pauc_spont=pauc(y, oof) if model == "spont" else np.nan, trace_scale=float(scale)))
        np.savez_compressed(out / f"continuous_{g.replace(' ', '_')}.npz", t=grid.astype(np.float64), trace=trace.astype(np.float32),
                            valid=valid, quiet=qg, norm_median=med, norm_scale=scale, peri=R.astype(np.float32), peri_t=PERI,
                            unit_index=sel, cluster_id=units.cluster_id.to_numpy(str)[sel],
                            electrode_group=units.electrode_group.to_numpy(str)[sel],
                            oof=oof[is_ev], y=y_ev, oof_all=oof, y_all=y, ctx_all=ctx, w=final["w"], b=final["b"], mu=final["mu"],
                            sd=final["sd"])
        params["groups"][g] = f"{len(idx)} units, C = {final['C']:g}"
        v = fit_rows[-1]
        print(f"{g}: {len(S)} units, pAUC {v['pauc']:.3f} | {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(peri_rows).to_parquet(out / "peri_event.parquet")
    pd.DataFrame(null_rows).to_parquet(out / "null_curves.parquet")
    pd.DataFrame(fit_rows).to_csv(out / "fit_summary.csv", index=False)
    ev.to_parquet(out / "task_events.parquet")
    np.savez_compressed(out / "behaviour.npz", licks=licks, **{f"{k}_t": v[0] for k, v in dlc.items()},
                        **{f"{k}_v": v[1] for k, v in dlc.items()})
    params["date"] = time.strftime("%Y-%m-%d %H:%M")
    (out / "params.json").write_text(json.dumps(params, indent=1, default=str))
    apply_thresholds(sid, target, model)
    print("DONE", out, f"{time.time() - t0:.0f} s")


def load_thresholds(target):
    """per-group thresholds (normalised readout units), the same for every session and both templates (user 2026-10-24);
    set by 004 from the pooled pilot sessions; THR_DEFAULT until then"""
    f = HOME / f"thresholds_{target}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return {g: float(d["thr"].get(g, THR_DEFAULT)) for g in GROUPS}, d.get("rule", str(f))
    return {g: THR_DEFAULT for g in GROUPS}, f"provisional default {THR_DEFAULT:g}"


def apply_thresholds(sid, target, model, thr=None):
    """threshold-dependent outputs from the saved readouts: validation.csv (fit summary + detection metrics), events.parquet,
    peri_event.parquet detected / latency. Other-modality stimuli crossing are reported, not penalised (user 2026-10-24)."""
    out = HOME / "pilot" / sid / target / model
    if thr is None:
        thr, rule = load_thresholds(target)
    else:
        rule = "given"
    P = json.loads((out / "params.json").read_text())
    F = pd.read_csv(out / "fit_summary.csv")
    PE = pd.read_parquet(out / "peri_event.parquet")
    N = pd.read_parquet(out / "null_curves.parquet")
    after = (PERI >= DET_WIN[0]) & (PERI <= DET_WIN[1])
    other_cls = "A" if target == "whisker" else "W"
    val, ev_rows, pe = [], [], []
    for _, f in F.iterrows():
        g, th = f.group, thr[f.group]
        c = np.load(out / f"continuous_{g.replace(' ', '_')}.npz")
        trace, valid, qg, R = c["trace"], c["valid"], c["quiet"], c["peri"]
        grid = c["t"]
        q = PE[PE.group == g].sort_values("ev").copy()
        hit = q.peak.to_numpy() > th
        q["detected"] = hit
        q["latency"] = np.where(hit, PERI[after][np.argmax(R[q.ev.to_numpy()][:, after] > th, 1)], np.nan)
        pe.append(q)
        D = detect(trace, valid, th)
        for s_, e_, p_ in D:
            ev_rows.append(dict(group=g, t_start=grid[s_] - WIDTH, t_end=grid[e_], t_peak=grid[p_], peak=float(trace[p_]),
                                dur_ms=float((e_ - s_ + 1) * STEP * 1000), quiet=bool(qg[p_])))
        n = N[N.group == g]
        stim, neg, oth = q.train & (q.y == 1), q.train & (q.y == 0), q.cls.eq(other_cls)
        val.append(dict(**f.to_dict(), thr=th, thr_rule=rule, sens=float(hit[stim.to_numpy()].mean()),
                        fp_events=float(hit[neg.to_numpy()].mean()),
                        fp_other_modality=float(hit[oth.to_numpy()].mean()) if oth.any() else np.nan,
                        rate_real=len(D) / P["valid_min"],
                        rate_real_quiet=sum(bool(qg[p_]) for *_, p_ in D) / P["quiet_min"],
                        rate_null_shift=float(np.interp(th, n.thr, n.null_shift)),
                        rate_null_shift_quiet=float(np.interp(th, n.thr, n.null_shift_quiet)),
                        rate_null_weight=float(np.interp(th, n.thr, n.null_weight))))
        v = val[-1]
        print(f"  {g}: thr {th:.2f} ({rule}) sens {v['sens']:.2f}, FP events {v['fp_events']:.2f}, other modality "
              f"{v['fp_other_modality']:.2f} | {v['rate_real']:.2f}/min vs time-shift null {v['rate_null_shift']:.3f}; quiet "
              f"{v['rate_real_quiet']:.2f} vs {v['rate_null_shift_quiet']:.3f}", flush=True)
    pd.concat(pe, ignore_index=True).to_parquet(out / "peri_event.parquet")
    pd.DataFrame(ev_rows, columns=["group", "t_start", "t_end", "t_peak", "peak", "dur_ms", "quiet"]).to_parquet(out / "events.parquet")
    pd.DataFrame(val).to_csv(out / "validation.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--target", default="whisker", choices=list(TARGET_CLS))
    ap.add_argument("--model", default="clean", choices=["clean", "spont"])
    ap.add_argument("--n-units", type=int, default=0, help="0 = every unit of the group")
    ap.add_argument("--apply-only", action="store_true")
    a = ap.parse_args()
    if a.apply_only:
        apply_thresholds(a.session, a.target, a.model)
    else:
        run(a.session, a.target, a.model, a.n_units)
