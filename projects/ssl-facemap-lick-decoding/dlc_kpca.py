"""DLC helpers for the face lick-decoding analyses: NWB trials + DLC loading, exact-curve trial table with history
predictors, DLC keypoint PCA (kpca) and 4-body-part (bp4) pre-stimulus features, blocked-CV AUC.

Moved verbatim (2026-11-10) from projects/ssl-face-lick-decoding/001_pilot_decoding.py (the DLC keypoint-PCA pilot, user
request 2026-11-03), so that project can be retired. Only the functions and constants used by other scripts are kept;
the pilot's own driver (run_session, main) is not. Code identical to the pilot.

Imported (importlib, by path) by:
  ssl-facemap-lick-decoding/003_analysis.py (and through it 006, 007)
  ssl-iti-signal-screen/001_features.py (trial_table)

Contents:
  load(path)                  NWB trials + DLC keypoint series (x, y, likelihood) + bp4 series, data/timestamp mismatch
  trial_table(trials)         whisker trials, context != passive, perf != 6, early_lick == 0 (= exact learning-curve
                              trials), with history predictors computed over all active trials
  keypoint_features(...)      keypoint PCA: 11 keypoints x/y (side + top), likelihood < P_CUT masked, resampled to
                              GRID_HZ, robust z-score; N_PCS posture PCs (positions) + N_PCS movement PCs (|velocity|);
                              per trial the window mean ("kpca_mean_<w>") or 250 ms bins ("kpca_bins_<w>")
  bp4_features(...)           jaw distance, top-nose distance, whisker angle, pupil area: posture (mean) and movement
                              (mean |diff|) per window (pupil -2..0 s for the 0.2 s window)
  blocked_auc(X, y)           blocked N_FOLDS-fold CV logistic regression AUC
"""

import warnings

import h5py
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
KEYPOINTS = ["jaw", "nose_base", "nose_tip", "pupil_top", "pupil_left", "pupil_right", "pupil_bottom",
             "top_nose_base", "top_nose_tip", "whisker_base", "whisker_tip"]
BP4 = ["jaw_distance", "top_nose_distance", "whisker_angle", "pupil_area"]
WINDOWS = {"0.2s": (-0.2, 0.0), "1s": (-1.0, 0.0), "2s": (-2.0, 0.0)}
GRID_HZ = 50
N_PCS = 10
BIN_S = 0.25
N_FOLDS = 5
P_CUT = 0.5
MAX_LEN_MISMATCH = 200  # frames (1 s at 200 Hz); larger data / timestamps differences = unresolved alignment, excluded


def _str(x):
    return x.decode() if isinstance(x, bytes) else x


def load(path):
    with h5py.File(path, "r") as f:
        desc = _str(f["session_description"][()])
        tr = f["intervals/trials"]
        trials = pd.DataFrame({c: [_str(v) for v in tr[c][:]] for c in
                               ("trial_type", "start_time", "lick_flag", "lick_time", "context", "perf", "early_lick",
                                "reward_available")})
        b = f.get("processing/behavior/BehavioralTimeSeries")
        series, bp4, mismatch = {}, {}, {}
        if b is not None:
            for k in KEYPOINTS:
                if f"{k}_x" in b:
                    arrs = [b[f"{k}_x"]["timestamps"][:], b[f"{k}_x"]["data"][:], b[f"{k}_y"]["data"][:],
                            b[f"{k}_likelihood"]["data"][:]]
                    n = min(len(v) for v in arrs)
                    mismatch[k] = max(len(v) for v in arrs) - n
                    series[k] = tuple(v[:n] for v in arrs)
            for k in BP4:
                if k in b:
                    ts, d = b[k]["timestamps"][:], b[k]["data"][:]
                    n = min(len(ts), len(d))
                    mismatch[k] = max(len(ts), len(d)) - n
                    bp4[k] = (ts[:n], d[:n])
    return desc, trials, series, bp4, mismatch


def trial_table(trials):
    t = trials.copy()
    for c in ("start_time", "lick_time"):
        t[c] = t[c].astype(float)
    for c in ("lick_flag", "perf", "early_lick", "reward_available"):
        t[c] = t[c].astype(float).astype(int)
    t = t[(t.context != "passive") & (t.perf != 6)].sort_values("start_time").reset_index(drop=True)
    # history over all active trials (any type)
    t["prev_lick"] = t.lick_flag.shift(1).fillna(0)
    t["prev_reward"] = (t.lick_flag * t.reward_available).shift(1).fillna(0)
    w = t.trial_type == "whisker_trial"
    t["prev_whisker_lick"] = t.lick_flag.where(w).ffill().shift(1).fillna(0)
    last = t.start_time.where(t.lick_flag == 1).ffill().shift(1)
    t["log_since_lick"] = np.log1p((t.start_time - last).fillna(600.0))
    t = t[w & (t.early_lick == 0)].reset_index(drop=True)
    t["trial_idx"] = np.arange(len(t)) / max(len(t) - 1, 1)
    return t


def keypoint_features(series, starts):
    """Session keypoint PCA on a 50 Hz grid; per-trial window means and 250 ms bin means of posture / movement PCs."""
    t0 = max(s[0][0] for s in series.values())
    t1 = min(s[0][-1] for s in series.values())
    grid = np.arange(t0, t1, 1 / GRID_HZ)
    cols = []
    for k, (ts, x, y, L) in series.items():
        for v in (x, y):
            ok = (L >= P_CUT) & np.isfinite(v)
            if ok.sum() < 100:
                continue
            cols.append(np.interp(grid, ts[ok], v[ok]))
    X = np.column_stack(cols)
    X = (X - np.median(X, 0)) / (np.median(np.abs(X - np.median(X, 0)), 0) * 1.4826 + 1e-6)
    V = np.abs(np.diff(X, axis=0, prepend=X[:1]))
    pos = PCA(N_PCS).fit(X)
    mov = PCA(N_PCS).fit(V)
    P, M = pos.transform(X), mov.transform(V)
    feats = {}
    for wname, (a, b) in WINDOWS.items():
        nb = int(round((b - a) / BIN_S))
        rows_mean, rows_bins = [], []
        for s in starts:
            i0, i1 = np.searchsorted(grid, [s + a, s + b])
            if i1 - i0 < 2:
                rows_mean.append(np.full(2 * N_PCS, np.nan))
                rows_bins.append(np.full(2 * N_PCS * nb, np.nan))
                continue
            seg = np.hstack([P[i0:i1], M[i0:i1]])
            rows_mean.append(seg.mean(0))
            edges = np.linspace(i0, i1, nb + 1).astype(int)
            rows_bins.append(np.concatenate([np.hstack([P[e0:e1], M[e0:e1]]).mean(0) if e1 > e0 else
                                             np.full(2 * N_PCS, np.nan) for e0, e1 in zip(edges[:-1], edges[1:])]))
        feats[f"kpca_mean_{wname}"] = np.array(rows_mean)
        feats[f"kpca_bins_{wname}"] = np.array(rows_bins)
    info = dict(n_coords=X.shape[1], var_pos=float(pos.explained_variance_ratio_.sum()),
                var_mov=float(mov.explained_variance_ratio_.sum()))
    return feats, info


def bp4_features(bp4, starts):
    feats = {}
    for wname, (a, b) in WINDOWS.items():
        cols = []
        for k in BP4:
            if k not in bp4:
                continue
            ts, d = bp4[k]
            pa = (-2.0, 0.0) if (k == "pupil_area" and wname == "0.2s") else (a, b)  # original script: pupil -2..0
            post, mov = [], []
            for s in starts:
                seg = d[(ts >= s + pa[0]) & (ts < s + pa[1])]
                post.append(np.nanmean(seg) if len(seg) else np.nan)
                mov.append(np.nanmean(np.abs(np.diff(seg))) if len(seg) > 1 else np.nan)
            cols += [post, mov]
        feats[f"bp4_{wname}"] = np.array(cols).T
    return feats


def blocked_auc(X, y):
    """Blocked K-fold AUC (contiguous blocks), standardisation and C chosen inside the training folds."""
    n = len(y)
    if n < 30 or y.sum() < 5 or (1 - y).sum() < 5:
        return np.nan
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    X = np.where(np.isfinite(X), X, 0.0)
    pred = np.full(n, np.nan)
    edges = np.linspace(0, n, N_FOLDS + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.zeros(n, bool)
        test[a:b] = True
        if len(np.unique(y[~test])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3, penalty="l2",
                                                                   class_weight="balanced", scoring="roc_auc",
                                                                   max_iter=2000))
        clf.fit(X[~test], y[~test])
        pred[test] = clf.predict_proba(X[test])[:, 1]
    ok = np.isfinite(pred)
    return roc_auc_score(y[ok], pred[ok]) if len(np.unique(y[ok])) == 2 else np.nan

