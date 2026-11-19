"""Time-shifted CCA between whisker S1 and whisker motor cortex, following Ranjbar-Slamloo & Kamigaki (2026,
PLoS Biol 24(10): e3004095, "Transient feedforward communication between somatosensory and premotor cortex supports
goal-directed behavior").

Their recipe (Methods + Zenodo scripts fig_3abcghi.m / fig_3defjkl.m; maps in paper_zenodo/data):
- spikes in 5-ms bins; CCA in 50-ms windows (10 bins) every 10 ms; samples = (trial, bin) rows of the window;
- S1 window at time t, M2 window at t + delay (delay > 0: S1 leads = feedforward), delays in 5-ms steps;
- CCV = first canonical correlation, in-sample (MATLAB canoncorr), "total" (raw counts) and "noise" (each unit's
  trial-type mean PSTH subtracted per bin); neurons without spikes in a window drop out (rank-deficient columns);
- control map: S1 trials shuffled (breaks trial correspondence, keeps both areas' PSTHs);
- per-session readout (fig_3defjkl.m): mean CCV over delays 0..+15 ms, minus its mean over -250..0 ms, averaged over
  0..100 ms after the stimulus.

Per-neuron z-scoring (as in the paper) does not change CCA within a window, so it is not applied.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root()
NWB_DIR = ROOT / "NWB_ks4"
OUT = ROOT / "combined_results_ks4" / "ssl-s1m2-communication"
UNIT_TABLE = ROOT / "combined_results_ks4" / "ssl-roc-learning-stages" / "unit_info_all_days_v2.parquet"
_SHARE = next((p for p in (Path(r"M:\share_internal"), Path("/mnt/share_internal")) if p.exists()), None)
REFERENCE_SHEET = _SHARE / "Axel_Bisi_Share" / "dataset_info" / "joint_mouse_reference_weight.xlsx" if _SHARE else None
for p in (ROOT / "NWB_reader", ROOT / "Github" / "ephys_utilities", ROOT / "Github" / "allen_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

STABLE_TABLE = ROOT / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet"

CONFIG = {
    "version": 3,
    "paper": "Ranjbar-Slamloo & Kamigaki 2026 PLoS Biol e3004095",
    "s1_area": "SSp-bfd",
    # M2: pooled MO-wM1 + MO-wM2 (main; the paper's M2 box, AP +1.3..2.2 / ML 0.5..1.0, lies between the two), and
    # each one separately
    "m2_sets": {"wMO": ["MO-wM1", "MO-wM2"], "MO-wM1": ["MO-wM1"], "MO-wM2": ["MO-wM2"]},
    "unit_quality": ["good", "mua"],
    "unit_stability": "137_stable_units.stable (coverage >= 0.9, presence >= 0.5, no probe-drift dependence)",
    "min_rate_hz": 0.0,          # paper: all clusters (neurons silent in a window drop out of that window's CCA)
    "min_units": 10,
    "max_units": None,           # all stable units (paper: all units)
    # v3: with all units (~150-270 per area) in-sample CCA saturates (CCV ~ 1 at baseline). Each area is reduced to its
    # top pca_dims principal components (one PCA per area and condition, fit on all trials x bins of the analysis
    # period; PCA -> CCA); in the CV control the PCA is refit on the training trials of each fold
    "pca_dims": 10,
    "min_trials": 20,
    "bin_s": 0.005,
    "window_bins": 10,           # 50 ms
    "step_s": 0.010,
    "t_from_s": -0.45, "t_to_s": 1.0,   # S1 window centres
    "delays_s": [-0.10, 0.10],   # 5-ms steps
    "n_shuffles": 1,             # paper: one trial-shuffle map
    "cv_folds": 10,              # paper S11: 10-fold cross-validated CCV (weights on 90 % of trials)
    "cv_delays_s": [-0.015, 0.015],
    "trials": "context active (resolve_context ITI rule), perf != 6, early_lick == 0, warm-up cut, rule-A1 tail trim",
    "artifact": "no Poisson fill; spikes in -10..+5 ms of EVERY trial start removed (all trial types, all windows "
                "analysed)",
    "artifact_window_s": [-0.010, 0.005],
    "conditions": {
        "All": "all active trials (whisker, auditory, no-stim)",
        "W_all": "whisker trials", "W_hit": "whisker & lick", "W_miss": "whisker & no lick",
        "A_hit": "auditory & lick", "NS_all": "no-stim trials",
    },
    "noise_groups": "trial type x lick (PSTH subtracted within each group)",
    "readout": {"ff_delays_s": [0.0, 0.015], "fb_delays_s": [-0.015, 0.0], "baseline_s": [-0.25, 0.0],
                "early_s": [0.0, 0.1]},
}


# ---------------------------------------------------------------- canoncorr (MATLAB port)

def _qr_basis(X, tol_scale=None):
    """Orthonormal basis of the centred columns of X (MATLAB canoncorr: QR with pivoting, rank by tolerance)."""
    import scipy.linalg as sla
    Xc = X - X.mean(0, keepdims=True)
    if Xc.shape[1] == 0:
        return np.zeros((X.shape[0], 0))
    Q, R, _ = sla.qr(Xc, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    if d.size == 0 or d[0] == 0:
        return np.zeros((X.shape[0], 0))
    tol = np.finfo(float).eps * max(Xc.shape) * d[0]
    r = int((d > tol).sum())
    return Q[:, :r]


def pca_project(X, k, rows=None):
    """Project [trial, bin, unit] onto the top-k principal components of the area, fit on all bins of the given trials
    (rows; default all). One fixed subspace per area and condition (PCA -> CCA)."""
    rows = np.arange(X.shape[0]) if rows is None else rows
    Xf = X[rows].reshape(-1, X.shape[2]).astype(np.float64)
    mu = Xf.mean(0)
    C = (Xf - mu).T @ (Xf - mu)
    w, V = np.linalg.eigh(C)
    V = V[:, np.argsort(w)[::-1][:k]]
    return ((X.astype(np.float64) - mu) @ V).astype(np.float32)


def first_cc(Q1, Q2):
    """First canonical correlation from two orthonormal bases."""
    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.nan
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(min(max(s[0], 0.0), 1.0))


def _qr_full(Xc):
    import scipy.linalg as sla
    Q, R, piv = sla.qr(Xc, mode="economic", pivoting=True)
    d = np.abs(np.diag(R))
    if d.size == 0 or d[0] == 0:
        return None
    r = int((d > np.finfo(float).eps * max(Xc.shape) * d[0]).sum())
    return Q[:, :r], R[:r, :r], piv[:r]


def cv_map(S, M, centres_idx, delay_steps, wb, folds, k):
    """Cross-validated first canonical correlation [delay, time] (paper S11 control): per fold, PCA (top k per area, all
    bins) and canoncorr weights are fit on the training trials; the held-out projections are correlated; mean over folds.
    S, M: raw [trial, bin, unit] tensors."""
    nT = S.shape[0]
    out = np.zeros((len(delay_steps), len(centres_idx)))
    cnt = np.zeros_like(out)
    for te in folds:
        tr = np.setdiff1d(np.arange(nT), te)
        Sk, Mk = pca_project(S, k, tr).astype(np.float64), pca_project(M, k, tr).astype(np.float64)
        cache = {}

        def dec(X, b0):
            Xw = X[tr, b0:b0 + wb].reshape(len(tr) * wb, -1)
            mu = Xw.mean(0)
            q = _qr_full(Xw - mu)
            return (mu, q) if q is not None else None

        for j, b0 in enumerate(centres_idx):
            ds = dec(Sk, b0)
            if ds is None:
                continue
            mus, (Qs, Rs, ps) = ds
            Ste = (Sk[te, b0:b0 + wb].reshape(len(te) * wb, -1) - mus)[:, ps]
            for i, d in enumerate(delay_steps):
                if b0 + d not in cache:
                    cache[b0 + d] = dec(Mk, b0 + d)
                dm = cache[b0 + d]
                if dm is None:
                    continue
                mum, (Qm, Rm, pm) = dm
                U, _, Vt = np.linalg.svd(Qs.T @ Qm)
                xs = Ste @ np.linalg.solve(Rs, U[:, 0])
                xm = (Mk[te, b0 + d:b0 + d + wb].reshape(len(te) * wb, -1) - mum)[:, pm] @ np.linalg.solve(Rm, Vt[0])
                if xs.std() > 0 and xm.std() > 0:
                    out[i, j] += np.corrcoef(xs, xm)[0, 1]
                    cnt[i, j] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, out / np.maximum(cnt, 1), np.nan)


# ---------------------------------------------------------------- data

def reference_sheet():
    ref = pd.read_excel(REFERENCE_SHEET)
    ref = ref.rename(columns={"mouse_id": "subject_id", "mouse_name": "subject_id"})
    return ref


def load_session(session_id, unit_keys):
    """Units (spike times, artefact-corrected) for the given (electrode_group, cluster_id) keys and the active trials.

    unit_keys: DataFrame with electrode_group, cluster_id, area_acronym_custom."""
    from ephys_utilities.helpers import data_utils
    sys.path.insert(0, str(REPO / "scripts"))
    from ssl_bwm_trial_prep import resolve_context
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = data_utils.process_single_nwb(str(NWB_DIR / f"{session_id}.nwb"), day_to_analyze="all",
                                             apply_artifact_correction=False)
    if res is None:
        raise RuntimeError(f"{session_id}: no ephys")
    units, trials = res["unit_table"], res["trial_table"]
    on = ["electrode_group", "cluster_id"]
    units = units.drop(columns=[x for x in unit_keys.columns if x in units.columns and x not in on])
    units["cluster_id"] = units["cluster_id"].astype(int)
    unit_keys = unit_keys.assign(cluster_id=unit_keys["cluster_id"].astype(int))
    units = units.merge(unit_keys, on=on, how="inner")
    if len(units) != len(unit_keys):
        raise RuntimeError(f"{session_id}: {len(units)} of {len(unit_keys)} units matched in the NWB")
    t = trials.sort_values("start_time").reset_index(drop=True)
    t = t[(resolve_context(t) == "active").to_numpy()]
    t = t[(t.perf != 6) & (t.early_lick == 0)].reset_index(drop=True)
    # auditory warm-up: keep exactly one trial before the first whisker trial
    w = np.flatnonzero(t.trial_type.to_numpy() == "whisker_trial")
    if len(w):
        t = t.iloc[max(w[0] - 1, 0):].reset_index(drop=True)
    # rule A1 end-of-session disengagement trim (as ssl_timeresolved_decoding.detect_terminal_disengagement): drop the
    # tail after the last lick when it holds >= 5 whisker and >= 1 auditory trials
    n0 = len(t)
    licked = np.flatnonzero(t.lick_flag.to_numpy() == 1)
    if len(licked):
        trail = t.iloc[licked[-1] + 1:]
        if (trail.trial_type == "whisker_trial").sum() >= 5 and (trail.trial_type == "auditory_trial").sum() >= 1:
            t = t.iloc[:licked[-1] + 1].reset_index(drop=True)
    t["n_trimmed_a1"] = n0 - len(t)
    # artefact: remove spikes in the artefact window around every trial start (all trial types, so whisker and other
    # trials are treated alike; no Poisson fill)
    lo, hi = CONFIG["artifact_window_s"]
    st_all = np.sort(trials.start_time.to_numpy(float))

    def blank(sp):
        sp = np.asarray(sp, float)
        i = np.searchsorted(st_all, sp, side="right") - 1
        prev = np.where(i >= 0, st_all[np.clip(i, 0, None)], -np.inf)
        nxt_i = np.clip(i + 1, 0, len(st_all) - 1)
        nxt = np.where(i + 1 < len(st_all), st_all[nxt_i], np.inf)
        bad = ((sp - prev) < hi) | ((nxt - sp) <= -lo)
        return sp[~bad]

    units["spike_times"] = units["spike_times"].apply(blank)
    return units, t


def condition_masks(t):
    wt, lk = t.trial_type.to_numpy(), t.lick_flag.to_numpy().astype(int)
    return {
        "All": np.ones(len(t), bool),
        "W_all": wt == "whisker_trial",
        "W_hit": (wt == "whisker_trial") & (lk == 1),
        "W_miss": (wt == "whisker_trial") & (lk == 0),
        "A_hit": (wt == "auditory_trial") & (lk == 1),
        "NS_all": wt == "no_stim_trial",
    }


def bin_edges():
    c = CONFIG
    lo = c["t_from_s"] + c["delays_s"][0] - c["window_bins"] * c["bin_s"]
    hi = c["t_to_s"] + c["delays_s"][1] + c["window_bins"] * c["bin_s"]
    n = int(round((hi - lo) / c["bin_s"]))
    return np.round(lo + c["bin_s"] * np.arange(n + 1), 6)


def count_tensor(spike_times, starts, edges):
    """Spike counts [trial, bin, unit]."""
    out = np.zeros((len(starts), len(edges) - 1, len(spike_times)), dtype=np.float64)
    q = starts[:, None] + edges[None, :]
    for u, st in enumerate(spike_times):
        c = np.searchsorted(np.asarray(st, float), q)
        out[:, :, u] = np.diff(c, axis=1)
    return out


def noise_residual(x, groups):
    r = x.copy()
    for g in np.unique(groups):
        m = groups == g
        r[m] -= r[m].mean(0, keepdims=True)
    return r


def ccv_map(S, M, centres_idx, delay_steps, wb, n_shuffles, rng):
    """CCV[delay, time] and the trial-shuffle mean.

    S, M: [trial, bin, unit]; centres_idx: first bin of the S1 window for each time; delay_steps in bins.
    Each window's QR basis is computed once (S1 per time, M2 per bin offset) and reused across delays / shuffles."""
    nT = S.shape[0]
    cache_m = {}

    def basis(X, b0):
        return _qr_basis(X[:, b0:b0 + wb, :].reshape(nT * wb, -1)).astype(np.float32)

    def mbasis(b0):
        if b0 not in cache_m:
            cache_m[b0] = basis(M, b0)
        return cache_m[b0]

    perms = [rng.permutation(nT) for _ in range(n_shuffles)]
    ccv = np.full((len(delay_steps), len(centres_idx)), np.nan)
    shf = np.full_like(ccv, np.nan)
    for j, b0 in enumerate(centres_idx):
        Qs = basis(S, b0)
        Qs3 = Qs.reshape(nT, wb, -1)
        Qs_perm = [Qs3[p].reshape(nT * wb, -1) for p in perms]  # permuting trials of S1 = permuting basis rows
        for i, d in enumerate(delay_steps):
            Qm = mbasis(b0 + d)
            ccv[i, j] = first_cc(Qs, Qm)
            if n_shuffles:
                shf[i, j] = np.mean([first_cc(Qp, Qm) for Qp in Qs_perm])
    return ccv, shf


def readout(ccv, delays_s, times_s, delay_range, cfg=CONFIG["readout"]):
    """Paper readout per session: mean over delays in delay_range, minus baseline mean, mean over early window."""
    dm = (delays_s >= delay_range[0] - 1e-9) & (delays_s <= delay_range[1] + 1e-9)
    trace = np.nanmean(ccv[dm], axis=0)
    b = (times_s >= cfg["baseline_s"][0] - 1e-9) & (times_s <= cfg["baseline_s"][1] + 1e-9)
    e = (times_s >= cfg["early_s"][0] - 1e-9) & (times_s <= cfg["early_s"][1] + 1e-9)
    trace_b = trace - np.nanmean(trace[b])
    return float(np.nanmean(trace_b[e])), trace_b, trace
