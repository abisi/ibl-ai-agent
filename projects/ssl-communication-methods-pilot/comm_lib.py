"""Library for the inter-areal communication methods pilot (user request 2026-11-05).

Signal: trial-to-trial co-fluctuations ("noise" activity) between two simultaneously recorded area groups: spike counts
per trial in time windows aligned to trial start, minus the condition-mean of each unit and window (removes the shared
stimulus/choice-locked PSTH, the classic confound of inter-areal measures).

Methods (established; time-resolved by sliding windows):
- RRR: reduced-rank ridge regression from source to target population (Semedo et al. 2019 Neuron, "communication
  subspace"): cross-validated predictive performance (1 - NSE) at the optimal rank, and the optimal rank.
- PCA -> CCA: each area reduced to its top PCs (fitted on training trials), then closed-form CCA; cross-validated
  (held-out trials) first canonical correlation (and mean of the top 3).
- Granger: ensemble (multi-trial) conditional Granger causality between the top PCs of the two areas in 10-ms bins
  (Ding et al. 2000 Biol Cybern; Geweke 1982), log ratio of restricted / full residual covariance determinants, in
  sliding windows; time-reversed Granger as a directionality control (Haufe et al. 2013; Winkler et al. 2016 IEEE TSP).
Nulls: trial-shuffle (source trials permuted within condition: keeps every PSTH and each area's own statistics, breaks
trial-matched coupling). Controls: within-area RRR ceiling, matched unit / trial counts with repeated draws, movement
covariates (dlc_helpers jaw opening and whisker speed per window) regressed out.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "projects" / "ssl-paper-figures"))
from common import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root()
NWB_DIR = ROOT / "NWB_ks4"
OUT = ROOT / "combined_results_ks4" / "ssl-communication-methods-pilot"
for p in (ROOT / "NWB_reader", ROOT / "Github" / "ephys_utilities", ROOT / "Github" / "allen_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "version": 1,
    "areas": ["Motor and frontal areas", "SSp-w", "Striatum and pallidum"],
    "area_level": "area_group_coarse_v3 (ssl-cross-area-communication 000_coverage_lib.COARSE_GROUP_V3_OF)",
    "quality": "quality_label == 'good' (unit_metrics_utils.classify_units_quality, project thresholds)",
    "min_rate_hz": 1.0,
    "conditions": {"W_hit": "whisker_trial & lick", "W_miss": "whisker_trial & no lick",
                   "A_hit": "auditory_trial & lick", "NS_cr": "no_stim_trial & no lick"},
    "trials": "context active, perf != 6, early_lick == 0; spikes with ephys_utilities whisker-artifact correction",
    "bin_s": 0.01, "bins_from_s": -0.5, "bins_to_s": 1.1,  # 10-ms bins; every method reads this tensor
    "window_s": 0.1, "step_s": 0.02, "t_min_s": -0.4, "t_max_s": 1.0,  # sliding windows of 10 bins
    "samples": "RRR / CCA samples = (trial, 10-ms bin) residual vectors inside the window; CV folds split by trial",
    "n_units_source": 20, "n_units_target": 20, "n_trials": 30, "n_draws": 2,
    "cv_folds": 10, "ridge_alphas": [0.01, 0.1, 1.0, 10.0], "max_rank": 10,
    "cca_pcs": 10, "cca_reg": 1e-3,
    "granger_bin_s": 0.01, "granger_pcs": 3, "granger_order": 3, "granger_window_s": 0.15,
    "n_shuffles_rrr_cca": 10, "n_shuffles_granger": 50,
    "movement_covariates": "dlc_helpers jaw_distance (side) and |whisker_velocity| (top), mean per window",
}


def _coverage_lib():
    path = REPO / "projects" / "ssl-cross-area-communication" / "exploratory-analyses" / "000_coverage_lib.py"
    spec = importlib.util.spec_from_file_location("coverage_lib", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- data

def load_session(session_id):
    """Units (good, area group, spike times) and trials (active, conditions) of one session, via ephys_utilities."""
    from ephys_utilities.helpers import data_utils
    from ephys_utilities.neural_utils import unit_metrics_utils
    import allen_utils
    cov = _coverage_lib()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = data_utils.process_single_nwb(str(NWB_DIR / f"{session_id}.nwb"), day_to_analyze="learning")
    if res is None:
        raise RuntimeError(f"{session_id}: no ephys or not a learning-day session")
    units, trials = res["unit_table"], res["trial_table"]
    units = unit_metrics_utils.classify_units_quality(units, thresholds=cov.THRESHOLDS, exclude=cov.EXCLUDE,
                                                      label_col="quality_label")
    units = allen_utils.process_allen_labels(units, split_merge_areas=True)
    units["area_group"] = units["area_acronym_custom"].map(cov.COARSE_GROUP_V3_OF)
    units = units[(units.quality_label == "good") & units.area_group.isin(CONFIG["areas"])].reset_index(drop=True)
    t = trials[(trials.context == "active") & (trials.perf != 6) & (trials.early_lick == 0)].copy()
    cond = pd.Series(pd.NA, index=t.index, dtype="object")
    cond[(t.trial_type == "whisker_trial") & (t.lick_flag == 1)] = "W_hit"
    cond[(t.trial_type == "whisker_trial") & (t.lick_flag == 0)] = "W_miss"
    cond[(t.trial_type == "auditory_trial") & (t.lick_flag == 1)] = "A_hit"
    cond[(t.trial_type == "no_stim_trial") & (t.lick_flag == 0)] = "NS_cr"
    t["condition"] = cond
    t = t[t.condition.notna()].sort_values("start_time").reset_index(drop=True)
    return units, t


def bin_edges():
    c = CONFIG
    n = int(round((c["bins_to_s"] - c["bins_from_s"]) / c["bin_s"]))
    return np.round(c["bins_from_s"] + c["bin_s"] * np.arange(n + 1), 4)


def window_grid():
    """Window centres and, per window, the indices of its 10-ms bins in bin_edges()."""
    c = CONFIG
    edges = bin_edges()
    centers = np.round(np.arange(c["t_min_s"], c["t_max_s"] + 1e-9, c["step_s"]), 4)
    half = c["window_s"] / 2
    idx = [np.flatnonzero((edges[:-1] >= ce - half - 1e-9) & (edges[1:] <= ce + half + 1e-9)) for ce in centers]
    return centers, idx


def fine_tensor(spike_times, starts):
    """Spike counts [trial, bin, unit] in the 10-ms bins of bin_edges() around each trial start."""
    edges = bin_edges()
    out = np.zeros((len(starts), len(edges) - 1, len(spike_times)), dtype=np.float32)
    q = starts[:, None] + edges[None, :]
    for u, st in enumerate(spike_times):
        c = np.searchsorted(np.asarray(st, float), q)
        out[:, :, u] = np.diff(c, axis=1)
    return out


def residualise(x, conditions):
    """Subtract each condition's mean (per unit and window): trial-to-trial fluctuations."""
    r = np.array(x, dtype=np.float64)
    for c in np.unique(conditions):
        m = conditions == c
        r[m] -= r[m].mean(0, keepdims=True)
    return r


def movement_covariates(session_id, starts, lo, hi):
    """[trial, 2, window]: mean jaw opening and whisker speed per window (dlc_helpers); None without DLC."""
    sys.path.insert(0, str(REPO / "projects" / "ssl-paper-figures"))
    try:
        from shared_movements import dlc_signals
        sig = dlc_signals(session_id)
    except Exception:
        return None
    out = np.full((len(starts), 2, len(lo)), np.nan)
    for k, name in enumerate(["jaw_distance", "whisker_speed"]):
        if name not in sig:
            continue
        x, ts, _ = sig[name]
        ok = np.isfinite(x)
        x, ts = x[ok], ts[ok]
        cs = np.r_[0, np.cumsum(x)]
        ia = np.searchsorted(ts, starts[:, None] + lo[None, :])
        ib = np.searchsorted(ts, starts[:, None] + hi[None, :])
        n = ib - ia
        with np.errstate(all="ignore"):
            out[:, k, :] = np.where(n > 0, (cs[ib] - cs[ia]) / n, np.nan)
    return out if np.isfinite(out).any() else None


# ---------------------------------------------------------------- methods (inputs: trials x units, residuals)

def _folds(n, k, rng):
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def _rows(T, trials):
    """[trial, bin, unit] -> (trial x bin, unit) rows for the given trials."""
    return T[trials].reshape(-1, T.shape[2])


def _ridge(X, Y, lam):
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ Y)


def rrr_cv(XT, YT, rng):
    """Reduced-rank ridge regression X -> Y (Semedo 2019) on [trial, bin, unit] window tensors (samples = trial-bin
    rows, folds split by trial): CV performance (1 - NSE, pooled over target units) of the full ridge model and at the
    optimal rank (smallest rank within 1 s.e.m. of the peak), the optimal rank and the chosen ridge alpha."""
    n, k, max_rank = len(XT), CONFIG["cv_folds"], min(CONFIG["max_rank"], XT.shape[2], YT.shape[2])
    folds = _folds(n, k, rng)
    X_all = _rows(XT, np.arange(n))
    scale = np.trace(X_all.T @ X_all) / X_all.shape[1] if X_all.shape[1] else 1.0  # ridge alphas relative to it
    res = {}
    for alpha in CONFIG["ridge_alphas"]:
        sse = np.zeros((k, max_rank + 1))  # last column: full ridge
        sst = np.zeros(k)
        for f, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            Xtr, Ytr, Xte, Yte = _rows(XT, train), _rows(YT, train), _rows(XT, test), _rows(YT, test)
            mx, my = Xtr.mean(0), Ytr.mean(0)
            Xtr, Ytr, Xte, Yte = Xtr - mx, Ytr - my, Xte - mx, Yte - my
            B = _ridge(Xtr, Ytr, alpha * scale)
            _, _, Vt = np.linalg.svd(Xtr @ B, full_matrices=False)
            for r in range(1, max_rank + 1):
                V = Vt[:r].T
                sse[f, r - 1] = ((Yte - Xte @ B @ V @ V.T) ** 2).sum()
            sse[f, max_rank] = ((Yte - Xte @ B) ** 2).sum()
            sst[f] = (Yte ** 2).sum()
        perf = 1 - sse / np.maximum(sst[:, None], 1e-12)
        res[alpha] = perf
    alpha = max(res, key=lambda a: res[a][:, -1].mean())
    perf = res[alpha]
    m, s = perf[:, :-1].mean(0), perf[:, :-1].std(0) / np.sqrt(k)
    peak = int(np.argmax(m))
    opt = int(np.argmax(m >= m[peak] - s[peak])) + 1
    return {"perf_full": float(perf[:, -1].mean()), "perf_rrr": float(m[opt - 1]), "rank": opt, "alpha": alpha}


def _pca_fit(X, k):
    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    return X.mean(0), Vt[:k].T


def _inv_sqrt(C, reg):
    w, V = np.linalg.eigh(C + reg * np.trace(C) / len(C) * np.eye(len(C)))
    w = np.maximum(w, 1e-12)
    return V @ np.diag(w ** -0.5) @ V.T


def pca_cca_cv(XT, YT, rng):
    """PCA (training trials) then closed-form regularised CCA on [trial, bin, unit] window tensors; held-out canonical
    correlations computed per test fold (canonical axes have an arbitrary sign per fold) and averaged."""
    n, k = len(XT), CONFIG["cv_folds"]
    kx, ky = min(CONFIG["cca_pcs"], XT.shape[2]), min(CONFIG["cca_pcs"], YT.shape[2])
    ccs = []
    for test in _folds(n, k, rng):
        train = np.setdiff1d(np.arange(n), test)
        Xtr, Ytr, Xte, Yte = _rows(XT, train), _rows(YT, train), _rows(XT, test), _rows(YT, test)
        mx, Px = _pca_fit(Xtr, kx)
        my, Py = _pca_fit(Ytr, ky)
        A, B = (Xtr - mx) @ Px, (Ytr - my) @ Py
        Cxx, Cyy, Cxy = A.T @ A / len(A), B.T @ B / len(B), A.T @ B / len(A)
        Wx, Wy = _inv_sqrt(Cxx, CONFIG["cca_reg"]), _inv_sqrt(Cyy, CONFIG["cca_reg"])
        U, S, Vt = np.linalg.svd(Wx @ Cxy @ Wy)
        a, b = Wx @ U[:, :3], Wy @ Vt[:3].T
        pa, pb = ((Xte - mx) @ Px) @ a, ((Yte - my) @ Py) @ b
        ccs.append([np.corrcoef(pa[:, d], pb[:, d])[0, 1] if pa[:, d].std() > 0 and pb[:, d].std() > 0 else np.nan
                    for d in range(pa.shape[1])])
    cc = np.nanmean(np.asarray(ccs, float), 0)
    return {"cc1": float(cc[0]), "cc_top3": float(np.nanmean(cc))}


def granger_ensemble(za, zb, order, t_index):
    """Conditional Granger A -> B and B -> A on [trial, time, pc] series, pooled over trials and the time points in
    t_index (each needs `order` past bins). Returns (gc_ab, gc_ba)."""
    def design(src_list, tgt):
        rows_y, rows_x = [], []
        for t in t_index:
            rows_y.append(tgt[:, t, :])
            rows_x.append(np.concatenate([s[:, t - lag, :] for s in src_list for lag in range(1, order + 1)], 1))
        return np.vstack(rows_x), np.vstack(rows_y)

    def resid_cov(Xd, Yd):
        Xd = np.c_[np.ones(len(Xd)), Xd]
        B = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
        E = Yd - Xd @ B
        return np.cov(E.T).reshape(Yd.shape[1], Yd.shape[1])

    def gc(src, tgt):
        Xr, Yr = design([tgt], tgt)
        Xf, Yf = design([tgt, src], tgt)
        return float(np.log(np.linalg.det(resid_cov(Xr, Yr)) / np.linalg.det(resid_cov(Xf, Yf))))

    return gc(za, zb), gc(zb, za)


def population_pcs(fine, conditions, k):
    """[trial, time, unit] fine-binned counts -> condition-residualised -> top-k PCs [trial, time, k] (fitted on all
    residual bins of the session's selected trials)."""
    r = residualise(fine, conditions)  # condition means per unit and bin
    flat = r.reshape(-1, r.shape[2])
    mu = flat.mean(0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    z = (flat - mu) @ Vt[:k].T
    z /= z.std(0, keepdims=True) + 1e-12
    return z.reshape(r.shape[0], r.shape[1], k)


# ---------------------------------------------------------------- methods v2 (benchmark 005, user request 2026-11-06)
# Directed / conditioned variants on the full [trial, time, unit] residual tensors; `idx` = the window's bins.
# pCCA: partial_CCA 0.1.0 (Gonzalez & Voroslakos et al. 2025, Buzsaki / Chen labs; PyPI partial-CCA), vendored in
# OUT/_vendor (no pip on haas). The package scores in-sample and its transform() ignores Z, so cross-validation and the
# held-out removal of Z (with the training-fold betas) are done here.
DLAG_NOTE = "DLAG / mDLAG (Gokcen et al.) exist only in MATLAB; no MATLAB on haas, so they are not benchmarked"


def _partial_cca_cls():
    vend = OUT / "_vendor" / "partial_cca-0.1.0"
    if str(vend) not in sys.path:
        sys.path.insert(0, str(vend))
    from partial_CCA import PartialCCA
    return PartialCCA


def lag_stack(full, idx, lags):
    """[trial, time, unit] -> [trial, len(idx), unit x len(lags)]: the window's bins, each with the given past lags."""
    return np.concatenate([full[:, np.asarray(idx) - lag] for lag in lags], axis=2)


def lagged_rrr_cv(S_full, T_full, idx, lag, rng):
    """RRR from the source `lag` bins earlier to the target in the window (Semedo et al. 2022 feedforward / feedback
    lagged communication). lag = 0 is the same-bin RRR."""
    return rrr_cv(S_full[:, np.asarray(idx) - lag], T_full[:, idx], rng)


def pcca_cv(XT, YT, ZT, rng, n_comp=3):
    """PCA (training trials) on X and Y, then partial CCA conditioned on Z (ZT: [trial, bin, k], e.g. a third area's
    PCs at lags 0..L via lag_stack) with partial_CCA; held-out canonical correlations per fold (test X, Y residualised on
    test Z with the training betas). ZT None: plain CCA through the same library."""
    PartialCCA = _partial_cca_cls()
    n, k = len(XT), CONFIG["cv_folds"]
    kx, ky = min(CONFIG["cca_pcs"], XT.shape[2]), min(CONFIG["cca_pcs"], YT.shape[2])
    ccs = []
    for test in _folds(n, k, rng):
        train = np.setdiff1d(np.arange(n), test)
        Xtr, Ytr, Xte, Yte = _rows(XT, train), _rows(YT, train), _rows(XT, test), _rows(YT, test)
        mx, Px = _pca_fit(Xtr, kx)
        my, Py = _pca_fit(Ytr, ky)
        A, B, At, Bt = (Xtr - mx) @ Px, (Ytr - my) @ Py, (Xte - mx) @ Px, (Yte - my) @ Py
        if ZT is None:
            m = PartialCCA(regularization=CONFIG["cca_reg"]).fit(A, B, None, verbose=False)
        else:
            Ztr, Zte = _rows(ZT, train), _rows(ZT, test)
            mz = Ztr.mean(0)
            Ztr, Zte = Ztr - mz, Zte - mz
            m = PartialCCA(regularization=CONFIG["cca_reg"]).fit(A, B, Ztr, verbose=False)
            At, Bt = At - Zte @ m.X_beta_, Bt - Zte @ m.Y_beta_
        pa, pb = At @ m.weights_x_[:, :n_comp], Bt @ m.weights_y_[:, :n_comp]
        ccs.append([np.corrcoef(pa[:, d], pb[:, d])[0, 1] if pa[:, d].std() > 0 and pb[:, d].std() > 0 else np.nan
                    for d in range(pa.shape[1])])
    cc = np.nanmean(np.asarray(ccs, float), 0)
    return {"cc1": float(cc[0]), "cc_top3": float(np.nanmean(cc))}


def lagged_cca_scan(A_full, B_full, idx, max_lag, rng):
    """Held-out PCA->CCA between A at t - lag and B at t for lag in -max_lag..max_lag (positive: A leads). Returns
    lags, cc1 per lag, the peak lag and the lead asymmetry (mean cc1 at positive minus negative lags)."""
    lags = np.arange(-max_lag, max_lag + 1)
    idx = np.asarray(idx)
    cc = np.array([pca_cca_cv(A_full[:, idx - L], B_full[:, idx], rng)["cc1"] if L >= 0 else
                   pca_cca_cv(A_full[:, idx], B_full[:, idx + L], rng)["cc1"] for L in lags])
    return {"lags": lags, "cc1": cc, "peak_lag": int(lags[np.nanargmax(cc)]),
            "asym": float(np.nanmean(cc[lags > 0]) - np.nanmean(cc[lags < 0]))}


def granger_conditional(za, zb, zc, order, t_index):
    """Conditional Granger A -> B | C and B -> A | C (C's past in both restricted and full models) on [trial, time, pc]
    series pooled over trials and t_index. zc None: the unconditional granger_ensemble."""
    if zc is None:
        return granger_ensemble(za, zb, order, t_index)

    def design(src_list, tgt):
        X = np.vstack([np.concatenate([s[:, t - lag, :] for s in src_list for lag in range(1, order + 1)], 1)
                       for t in t_index])
        return np.c_[np.ones(len(X)), X], np.vstack([tgt[:, t, :] for t in t_index])

    def logdet_resid(Xd, Yd):
        E = Yd - Xd @ np.linalg.lstsq(Xd, Yd, rcond=None)[0]
        return np.linalg.slogdet(np.atleast_2d(np.cov(E.T)))[1]

    def gc(src, tgt):
        return float(logdet_resid(*design([tgt, zc], tgt)) - logdet_resid(*design([tgt, zc, src], tgt)))

    return gc(za, zb), gc(zb, za)


def lagged_rrr_conditional(S_full, T_full, idx, n_lags, rng, Z=None, ridge=1e-2):
    """Granger-like lagged RRR: the target's window activity is first regressed (ridge, all trials) on its own past
    (lags 1..n_lags) and on Z (e.g. a third population's PCs at lags 0..n_lags, [trial, len(idx), k]); the residual is
    then predicted from the source's past (lags 1..n_lags) with rrr_cv. The conditioning set does not contain the
    source, so the trial-shuffle null of the source keeps it valid."""
    Tw = T_full[:, idx]
    P = lag_stack(T_full, idx, range(1, n_lags + 1))
    if Z is not None:
        P = np.concatenate([P, Z], axis=2)
    Pr, Tr = P.reshape(-1, P.shape[2]), Tw.reshape(-1, Tw.shape[2])
    Pr, Tr = Pr - Pr.mean(0), Tr - Tr.mean(0)
    B = _ridge(Pr, Tr, ridge * np.trace(Pr.T @ Pr) / max(Pr.shape[1], 1))
    resid = (Tr - Pr @ B).reshape(Tw.shape)
    return rrr_cv(lag_stack(S_full, idx, range(1, n_lags + 1)), resid, rng)


def residualise_gain(x, conditions):
    """Condition-mean residuals with a trial-wise gain of the evoked response removed per area (x: one area's
    [trial, time, unit]): per trial, the area's summed residual is regressed on its time-centred summed PSTH, and that
    gain times each unit's time-centred PSTH is subtracted (shared multiplicative arousal / gain fluctuations would
    otherwise appear as coupling around the evoked response)."""
    r = np.array(x, dtype=np.float64)
    for c in np.unique(conditions):
        m = conditions == c
        P = r[m].mean(0)  # [time, unit]
        Pc = P - P.mean(0, keepdims=True)
        y = (r[m] - P[None]).sum(2)  # [trial, time]
        p = Pc.sum(1)
        beta = y @ p / max(p @ p, 1e-12)  # [trial]
        r[m] = r[m] - P[None] - beta[:, None, None] * Pc[None]
    return r
