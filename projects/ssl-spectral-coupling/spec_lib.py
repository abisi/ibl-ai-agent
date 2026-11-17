"""Spectral (multitaper FFT) coupling between areas: library for ssl-spectral-coupling (user request 2026-11-17).

Frequency-domain counterparts of the ssl-communication-methods-pilot measures, computed on trial-to-trial fluctuations
(condition mean removed per unit and bin, soft-normalised per unit) of stable units in 5-ms bins:
  msc     magnitude-squared coherence of the two area "sum" signals (mean of the unit residuals)
  imc     imaginary part of their coherency (signed; > 0 when the source leads at small phase)
  cancoh  first canonical coherence of the top 3 PCs of each area (Brillinger 1981)
  pcoh    squared partial coherence of the sum signals given the top 3 PCs of up to 200 other units ("rest")
  psi     phase-slope index per band (Nolte et al. 2008; > 0 = source leads)
  gc_ab / gc_ba  spectral Granger causality source -> target / target -> source (Geweke 1982), bivariate VAR of order
          4 (20 ms) fitted on all trials of the draw (ensemble), evaluated on the FFT frequency grid
Null: source trials permuted (keeps every spectrum and PSTH, breaks trial-matched coupling); excess = value - null mean.
Cross-spectra use S_ab = <A conj(B)>, so a source leading the target has a phase that increases with frequency.
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal.windows import dpss

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "ssl-communication-methods-pilot"
sys.path.insert(0, str(PILOT))
import comm_lib as cl  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = cl.ROOT
OUT = ROOT / "combined_results_ks4" / "ssl-spectral-coupling"

BIN = 0.005
EDGES = np.round(np.arange(-1.2, 1.0 + 1e-9, BIN), 4)
EPOCHS = {"quiet": (-1.0, 0.0, 3.0), "early": (0.005, 0.155, 1.5), "execution": (0.2, 0.5, 3.0)}
TR = {"win": 0.2, "step": 0.05, "centers": (-0.9, 0.8), "NW": 2.0}
BANDS = {"theta": (3, 8), "alpha": (8, 13), "beta": (13, 30), "gamma_low": (30, 60), "gamma_high": (60, 100)}
FMAX = 100.0
SPEC_GRID = np.arange(2.0, 100.0 + 1e-9, 2.0)  # common grid (bin centres, 2-Hz bins) for saved spectra
N_PC, N_REST_PC, N_REST_UNITS, VAR_ORDER = 3, 3, 200, 4
SOFT_SD = float(np.sqrt(2.0 * BIN))  # Poisson SD of a 2-Hz unit in one bin (pilot "z" preprocessing)
MEASURES = ("msc", "imc", "cancoh", "pcoh", "gc_ab", "gc_ba")


# ---------------------------------------------------------------- data

def load_session(session_id):
    """Stable good + mua units (pilot v2 loader) and active trials with the warm-up cut; condition labels."""
    v2 = _load("rrr_v2", PILOT / "012_rrr_timeresolved_v2.py")
    cov3 = cl._coverage_lib()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        units, t, _, rule = v2.load(session_id)
    t = t.sort_values("start_time").reset_index(drop=True)
    wi = np.flatnonzero((t.trial_type == "whisker_trial").to_numpy())
    if len(wi):  # auditory warm-up block: keep exactly one trial before the first whisker trial
        t = t.iloc[max(0, wi[0] - 1):].reset_index(drop=True)
    assert not (t.loc[t.context != "passive", "perf"] == 6).any()
    kind = t.trial_type.map({"whisker_trial": "W", "auditory_trial": "A", "no_stim_trial": "NS"})
    lk = (t.lick_flag == 1).to_numpy()
    lab = np.where(kind == "W", np.where(lk, "W_lick", "W_nolick"),
                   np.where(kind == "A", np.where(lk, "A_hit", "A_miss"), np.where(lk, "NS_fa", "NS_cr")))
    t["condition"] = lab
    units = units.copy()
    units["area_group"] = units.area_acronym_custom.map(cov3.COARSE_GROUP_V3_OF)
    return units, t, rule


def counts(spike_times, starts):
    """[trial, bin, unit] spike counts in the 5-ms bins of EDGES around each trial start (float32)."""
    q = starts[:, None] + EDGES[None, :]
    out = np.zeros((len(starts), len(EDGES) - 1, len(spike_times)), dtype=np.float32)
    for u, st in enumerate(spike_times):
        out[:, :, u] = np.diff(np.searchsorted(np.sort(np.asarray(st, float)), q), axis=1)
    return out


def window_idx(t0, t1):
    return np.flatnonzero((EDGES[:-1] >= t0 - 1e-9) & (EDGES[1:] <= t1 + 1e-9))


def windows():
    """{name: (bin indices, NW)} for the epochs and the time-resolved 200-ms windows ("tr_<centre>")."""
    out = {k: (window_idx(a, b), nw) for k, (a, b, nw) in EPOCHS.items()}
    c0, c1 = TR["centers"]
    for c in np.round(np.arange(c0, c1 + 1e-9, TR["step"]), 3):
        out[f"tr_{c:+.2f}"] = (window_idx(c - TR["win"] / 2, c + TR["win"] / 2), TR["NW"])
    return out


def residuals(X, labels, soft=SOFT_SD):
    """Condition mean removed per unit and bin, then soft-normalised per unit (SD over trials and bins)."""
    R = np.array(X, dtype=np.float32)
    for c in np.unique(labels):
        m = labels == c
        R[m] -= R[m].mean(0, keepdims=True)
    R /= R.std((0, 1), keepdims=True) + soft
    return R


def pcs(R, k):
    """[trial, bin, k] top-k PC scores (unit SD) of R [trial, bin, unit]."""
    flat = R.reshape(-1, R.shape[2]).astype(np.float64)
    flat = flat - flat.mean(0)
    C = flat.T @ flat
    w, V = np.linalg.eigh(C)
    z = flat @ V[:, ::-1][:, :k]
    z /= z.std(0) + 1e-12
    return z.reshape(R.shape[0], R.shape[1], k).astype(np.float32)


def area_signals(R):
    """[trial, bin, 1 + N_PC]: mean of the unit residuals ("sum") and the top PCs."""
    return np.concatenate([R.mean(2, keepdims=True), pcs(R, N_PC)], axis=2)


# ---------------------------------------------------------------- spectra

_TAPERS = {}


def tapers(L, NW):
    key = (L, NW)
    if key not in _TAPERS:
        K = max(1, int(2 * NW) - 1)
        _TAPERS[key] = np.atleast_2d(dpss(L, NW, K)).astype(np.float64)
    return _TAPERS[key]


def nfft_for(L):
    return int(max(64, 2 ** np.ceil(np.log2(2 * L))))


def tfft(S, idx, NW):
    """Tapered FFT [trial, taper, freq, channel] of S[:, idx, :] (each trial's window mean removed), f in (0, FMAX]."""
    seg = S[:, idx, :].astype(np.float64)
    seg = seg - seg.mean(1, keepdims=True)
    tp = tapers(len(idx), NW)
    n = nfft_for(len(idx))
    F = np.fft.rfft(seg[:, None, :, :] * tp[None, :, :, None], n=n, axis=2)
    f = np.fft.rfftfreq(n, BIN)
    keep = (f > 0) & (f <= FMAX + 1e-9)
    return F[:, :, keep, :], f[keep]


def cross_matrix(V):
    """[freq, ch, ch] cross-spectral matrix averaged over trials and tapers; V [trial, taper, freq, ch]."""
    W = np.ascontiguousarray(np.transpose(V, (2, 0, 1, 3))).reshape(V.shape[2], -1, V.shape[3])  # [f, t*k, ch]
    return np.matmul(np.transpose(W, (0, 2, 1)), W.conj()) / W.shape[1]


def spectral_measures(S, n_a, n_b, n_z):
    """msc, imc, cancoh, pcoh [freq] and complex coherency from the cross matrix of [A (n_a), B (n_b), Z (n_z)];
    channel 0 of A and of B is the sum signal, the rest are PCs."""
    a, b = 0, n_a
    coh = S[:, a, b] / np.sqrt(S[:, a, a].real * S[:, b, b].real + 1e-30)
    out = {"msc": np.abs(coh) ** 2, "imc": coh.imag}
    ia, ib = np.arange(1, n_a), np.arange(n_a + 1, n_a + n_b)
    Saa, Sbb, Sab = S[:, ia][:, :, ia], S[:, ib][:, :, ib], S[:, ia][:, :, ib]
    eye = np.eye(len(ia))[None] * 1e-9
    M = np.linalg.solve(Saa + eye, Sab) @ np.linalg.solve(Sbb + eye, np.conj(np.transpose(Sab, (0, 2, 1))))
    out["cancoh"] = np.clip(np.linalg.eigvals(M).real.max(1), 0, 1)
    if n_z:
        sub = np.r_[a, b, np.arange(n_a + n_b, n_a + n_b + n_z)]
        G = np.linalg.inv(S[:, sub][:, :, sub] + np.eye(len(sub))[None] * 1e-9)
        out["pcoh"] = np.abs(G[:, 0, 1]) ** 2 / (G[:, 0, 0].real * G[:, 1, 1].real)
    else:
        out["pcoh"] = np.full(len(coh), np.nan)
    return out, coh


def psi_bands(coh, f, bands):
    """Phase-slope index per band: Im sum conj(C(f)) C(f + df) over consecutive frequencies inside the band."""
    prod = np.conj(coh[:-1]) * coh[1:]
    mid = (f[:-1] + f[1:]) / 2
    return {k: float(prod[(mid >= lo) & (mid < hi)].imag.sum()) for k, (lo, hi) in bands.items()}


def spectral_granger(xa, xb, f, p=VAR_ORDER):
    """Geweke spectral GC a->b and b->a [freq] from a bivariate VAR(p) fitted on all trials (rows within trials).
    xa, xb [trial, time], each trial's window mean already removed."""
    T = xa.shape[1]
    Y = np.stack([xa[:, p:], xb[:, p:]], -1).reshape(-1, 2)
    lags = [np.stack([xa[:, p - k:T - k], xb[:, p - k:T - k]], -1) for k in range(1, p + 1)]
    Xl = np.concatenate(lags, -1).reshape(-1, 2 * p)
    B, *_ = np.linalg.lstsq(Xl, Y, rcond=None)
    E = Y - Xl @ B
    Sig = E.T @ E / len(E)
    Ak = B.T.reshape(2, p, 2).transpose(1, 0, 2)  # [lag, out, in]
    z = np.exp(-2j * np.pi * f[:, None] * BIN * np.arange(1, p + 1)[None, :])  # [freq, lag]
    Af = np.eye(2)[None] - np.einsum("fk,koi->foi", z, Ak)
    H = np.linalg.inv(Af)
    Sf = H @ Sig @ np.conj(np.transpose(H, (0, 2, 1)))
    saa, sbb = Sf[:, 0, 0].real, Sf[:, 1, 1].real
    gc_ba = np.log(saa / np.maximum(saa - (Sig[1, 1] - Sig[0, 1] ** 2 / Sig[0, 0]) * np.abs(H[:, 0, 1]) ** 2, 1e-30))
    gc_ab = np.log(sbb / np.maximum(sbb - (Sig[0, 0] - Sig[0, 1] ** 2 / Sig[1, 1]) * np.abs(H[:, 1, 0]) ** 2, 1e-30))
    return gc_ab, gc_ba


def band_means(spec, f, bands):
    return {k: float(np.nanmean(spec[(f >= lo) & (f < hi)])) if np.any((f >= lo) & (f < hi)) else np.nan
            for k, (lo, hi) in bands.items()}


def to_grid(spec, f):
    """Average a spectrum into the 2-Hz bins of SPEC_GRID (NaN where no frequency falls in a bin)."""
    out = np.full(len(SPEC_GRID), np.nan)
    for i, c in enumerate(SPEC_GRID):
        m = (f >= c - 1) & (f < c + 1)
        if m.any():
            out[i] = np.nanmean(spec[m])
    return out


def bands_for(win_len_s):
    """Bands resolvable in a window: theta / alpha only for windows >= 0.5 s."""
    return {k: v for k, v in BANDS.items() if win_len_s >= 0.5 or v[0] >= 13}


def pair_all(FA, FB, FZ, xa, xb, f, perms, bands, gc=True):
    """Observed (perm 0) and shuffle values of every measure for one pair and window.
    FA, FB [trial, taper, freq, 1 + N_PC]; FZ [trial, taper, freq, N_REST_PC] or None; xa, xb [trial, time] sum
    signals of the window (mean removed). Returns {measure: [perm, freq]}, {psi_band: [perm]}."""
    n_a, n_b = FA.shape[3], FB.shape[3]
    n_z = 0 if FZ is None else FZ.shape[3]
    spec = {m: [] for m in MEASURES}
    psi = {k: [] for k in bands}
    for perm in perms:
        V = np.concatenate([FA[perm], FB] + ([FZ] if n_z else []), axis=3)
        m, coh = spectral_measures(cross_matrix(V), n_a, n_b, n_z)
        for k in ("msc", "imc", "cancoh", "pcoh"):
            spec[k].append(m[k])
        for k, v in psi_bands(coh, f, bands).items():
            psi[k].append(v)
        if gc:
            ab, ba = spectral_granger(xa[perm], xb, f)
        else:
            ab = ba = np.full(len(f), np.nan)
        spec["gc_ab"].append(ab)
        spec["gc_ba"].append(ba)
    return {k: np.array(v) for k, v in spec.items()}, {k: np.array(v) for k, v in psi.items()}
