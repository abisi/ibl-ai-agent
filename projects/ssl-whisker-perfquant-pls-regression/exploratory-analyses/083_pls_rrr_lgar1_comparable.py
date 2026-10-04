"""Matched comparison of PLS+1SE, RRR, and LG-AR1 (user request 2026-09-23,
following up on "Can you compare... are they comparable? Do they take
the same input data?" -- answer was no; this makes them so): SAME data,
SAME window, SAME feature scaling for both decoders, and LG-AR1 applied
to BOTH decoders' outputs (previously only paired with RRR in `082`).

**Window: the baseline/ITI window, not sensory** (user's explicit
request) -- `WIDE_BASELINE_WINDOW = (-2.000, -0.010)`, this project's own
canonical baseline window (`062`'s window-comparison study, §3 of the
recap). Chosen deliberately by the user as the harder, more informative
test: §3 already found baseline is the MOST autocorrelated/highest-null
window (raw scores highest, but above-null gain lowest of the three
windows tested) -- exactly the regime where "does time-resolving within
the window help" (RRR's structural advantage over PLS) is most likely to
show a real difference, and where a naive unsmoothed comparison would be
most misleading (matches `082`'s own already-demonstrated null-inflation
risk).

**What's now matched between PLS and RRR that wasn't in `082`:**
- Same window (`WIDE_BASELINE_WINDOW`) for both -- previously PLS used
  the sensory window and RRR used a different, wider peri-stimulus window.
- Same feature scaling: RRR now uses the SAME floored `StandardScaler`
  (`SCALE_FLOOR=1e-3`, the `052` bug fix) PLS already used -- `082`'s RRR
  only mean-centered, giving high-firing-rate neurons disproportionate
  weight in the ridge fit.
- Same null convention (linear shift, non-wrapping) -- already matched,
  unchanged.
- **LG-AR1 now applied to PLS's out-of-fold outputs too**, not just
  RRR's -- the missing pairing flagged in the comparison discussion.

**What's still NOT matched, deliberately** (would require changing each
method's own defining property, not a fair thing to force): RRR gets
T=20 time bins across the baseline window (its whole structural point is
learning a temporal kernel instead of assuming a flat aggregate); PLS
still aggregates the window into one flat mean per neuron (that
aggregation IS what PLS is, in this project's design). Component-
selection philosophy also stays as each method's own convention (PLS:
1-SE rule; RRR: plain CV-best grid) -- forcing RRR through the 1-SE rule
would need a real algorithmic change (1-SE across a 2D R x lambda grid),
out of scope for this matched-input comparison specifically.

Same 3 pilot sessions (MH070/MH069/MH031), same 3 targets, `N_SHUF_NULL
=20`, whole_brain.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
WIDE_BASELINE_WINDOW = (-2.000, -0.010)
RRR_BIN_WIDTH = 0.100  # 100ms bins over the ~2s baseline window -> T~20
SCALE_FLOOR = 1e-3
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
MIN_TRIALS_FOR_REGRESSION = 25
N_FOLDS = 5
N_REPEATS = 2
COMPONENT_GRID = (2, 5, 10, 15, 20, 30)
R_GRID = (1, 2, 3)
LAM_GRID = (3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0)
# 2026-09-23: widened after the "plot the curves" figure revealed wild
# held-out-fold outliers (RRR predictions reaching ~250-300x the
# standardized target's own scale) for at least one session -- the
# selected lambda for that cell (100.0) turned out to be the OLD grid's
# ceiling, meaning the search never actually found an interior optimum,
# just the best available option in a grid that didn't extend far enough.
# With R up to 3 and N up to ~2000+ units, the U-step alone fits up to
# ~R*N~6000 parameters from ~80 training trials per fold -- under-
# regularization here doesn't just overfit, it can make the alternating-
# ridge iteration itself numerically unstable on held-out data.
N_FOLDS_INNER = 3
N_SHUF_NULL = 20
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5

RECIPIENTS = [
    ("MH070", "MH070_20260121_140848"),
    ("MH069", "MH069_20260122_111455"),
    ("MH031", "MH031_20250507_104425"),
]


def _floored_scaler():
    from sklearn.preprocessing import StandardScaler

    class _FlooredScaler(StandardScaler):
        def fit(self, X, y=None):
            super().fit(X, y)
            self.scale_ = np.maximum(self.scale_, SCALE_FLOOR)
            return self
    return _FlooredScaler()


# ---------------------------------------------------------------------------
# PLS+1SE (single aggregated window -- copied from 058/078, unchanged recipe)
# ---------------------------------------------------------------------------

def select_pls_1se_components(X, Y, rng, n_folds=5, grid=COMPONENT_GRID):
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score

    max_comp = max(1, min(30, X.shape[1] - 1, int(X.shape[0] * 0.6)))
    candidates = sorted(set(min(c, max_comp) for c in grid if c <= max_comp)) or [max_comp]
    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X))
    mean_scores, se_scores = [], []
    for n_comp in candidates:
        fold_scores = []
        for tr, te in splits:
            scaler = _floored_scaler()
            scaler.fit(X[tr])
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(scaler.transform(X[tr]), Y[tr])
            Y_pred = pls.predict(scaler.transform(X[te]))
            fold_scores.append(r2_score(Y[te], Y_pred))
        mean_scores.append(np.mean(fold_scores))
        se_scores.append(np.std(fold_scores) / np.sqrt(n_folds))
    best_idx = int(np.argmax(mean_scores))
    threshold = mean_scores[best_idx] - se_scores[best_idx]
    return next(c for i, c in enumerate(candidates) if mean_scores[i] >= threshold)


def _pls_fit_predict_1d(X_train, y_train, X_test, n_components):
    from sklearn.cross_decomposition import PLSRegression
    scaler = _floored_scaler()
    scaler.fit(X_train)
    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(scaler.transform(X_train), y_train)
    return pls.predict(scaler.transform(X_test)).ravel()


def pls_cv_predict(X, y, n_components, rng, n_repeats=N_REPEATS, n_folds=N_FOLDS):
    from sklearn.model_selection import KFold
    y_pred_accum = np.zeros_like(y, dtype=float)
    for _ in range(n_repeats):
        seed = int(rng.integers(0, 2**31 - 1))
        y_pred_rep = np.empty_like(y, dtype=float)
        for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            y_pred_rep[te] = _pls_fit_predict_1d(X[tr], y[tr], X[te], n_components)
        y_pred_accum += y_pred_rep
    return y_pred_accum / n_repeats


# ---------------------------------------------------------------------------
# RRR (dual-ridge alternating fit -- same as 082, now WITH floored scaling)
# ---------------------------------------------------------------------------

def _dual_ridge_solve(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    K = F.shape[0]
    alpha = np.linalg.solve(F @ F.T + lam * np.eye(K), y)
    return F.T @ alpha


def _scale_X(X: np.ndarray) -> np.ndarray:
    """Floored per-(neuron,bin) scaling -- matches PLS's SCALE_FLOOR
    convention, unlike `082`'s mean-centering-only RRR."""
    mean = X.mean(axis=0, keepdims=True)
    std = np.maximum(X.std(axis=0, keepdims=True), SCALE_FLOOR)
    return (X - mean) / std, mean, std


def fit_rrr(X: np.ndarray, y: np.ndarray, R: int, lam: float, n_iter: int = 30, seed: int = 0):
    K, N, T = X.shape
    Xc, x_mean, x_std = _scale_X(X)
    y_mean = y.mean()
    yc = y - y_mean
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((N, R)) / np.sqrt(N)
    for _ in range(n_iter):
        Z = np.einsum("nr,knt->krt", U, Xc).reshape(K, R * T)
        V = _dual_ridge_solve(Z, yc, lam).reshape(R, T)
        F = np.einsum("knt,rt->knr", Xc, V).reshape(K, N * R)
        U = _dual_ridge_solve(F, yc, lam).reshape(N, R)
    return U, V, y_mean, x_mean, x_std


def predict_rrr(X: np.ndarray, U, V, y_mean, x_mean, x_std) -> np.ndarray:
    Xc = (X - x_mean) / x_std
    return np.einsum("knt,nt->k", Xc, U @ V) + y_mean


def rrr_cv_predict(X, y, R, lam, n_folds, rng):
    from sklearn.model_selection import KFold
    d = np.full(len(y), np.nan)
    seed = int(rng.integers(0, 2**31 - 1))
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
        U, V, y_mean, x_mean, x_std = fit_rrr(X[tr], y[tr], R, lam, seed=int(rng.integers(0, 2**31 - 1)))
        d[te] = predict_rrr(X[te], U, V, y_mean, x_mean, x_std)
    return d


def select_rrr_hyperparams(X, y, rng):
    from sklearn.model_selection import KFold
    best_score, best_R, best_lam = -np.inf, R_GRID[0], LAM_GRID[0]
    seed = int(rng.integers(0, 2**31 - 1))
    splits = list(KFold(n_splits=N_FOLDS_INNER, shuffle=True, random_state=seed).split(X))
    for R in R_GRID:
        if R > min(X.shape[1], X.shape[0] - 1):
            continue
        for lam in LAM_GRID:
            scores = []
            for tr, te in splits:
                U, V, y_mean, x_mean, x_std = fit_rrr(X[tr], y[tr], R, lam, seed=0)
                pred = predict_rrr(X[te], U, V, y_mean, x_mean, x_std)
                scores.append(-1.0 if np.std(pred) < 1e-8 else pearsonr(pred, y[te])[0])
            mean_score = np.nanmean(scores)
            if mean_score > best_score:
                best_score, best_R, best_lam = mean_score, R, lam
    return best_R, best_lam


# ---------------------------------------------------------------------------
# LG-AR1 (unchanged from 082)
# ---------------------------------------------------------------------------

def lgar1_filter_loglik(d, rho, theta, mu, q, r):
    K = len(d)
    m_f, P_f, m_p, P_p = (np.zeros(K) for _ in range(4))
    P0 = q / max(1 - rho ** 2, 1e-6)
    loglik = 0.0
    m, P = 0.0, P0
    for k in range(K):
        mp, Pp = (0.0, P0) if k == 0 else (rho * m, rho ** 2 * P + q)
        m_p[k], P_p[k] = mp, Pp
        S = theta ** 2 * Pp + r
        resid = d[k] - (theta * mp + mu)
        loglik += -0.5 * np.log(2 * np.pi * S) - 0.5 * resid ** 2 / S
        G = Pp * theta / S
        m = mp + G * resid
        P = (1 - G * theta) * Pp
        m_f[k], P_f[k] = m, P
    return m_f, P_f, m_p, P_p, loglik


def lgar1_smooth(d, rho, theta, mu, q, r):
    m_f, P_f, m_p, P_p, _ = lgar1_filter_loglik(d, rho, theta, mu, q, r)
    K = len(d)
    m_s = m_f.copy()
    for k in range(K - 2, -1, -1):
        denom = P_p[k + 1] if P_p[k + 1] > 1e-10 else 1e-10
        J = P_f[k] * rho / denom
        m_s[k] = m_f[k] + J * (m_s[k + 1] - m_p[k + 1])
    return m_s


def fit_lgar1(d):
    d_var = max(np.var(d), 1e-6)

    def negloglik(params):
        u_rho, u_theta, mu, u_q, u_r = params
        try:
            _, _, _, _, ll = lgar1_filter_loglik(d, np.tanh(u_rho), np.exp(u_theta), mu, np.exp(u_q), np.exp(u_r))
        except (FloatingPointError, ZeroDivisionError):
            return 1e10
        return 1e10 if not np.isfinite(ll) else -ll

    x0 = np.array([np.arctanh(0.5), 0.0, float(np.mean(d)), np.log(d_var / 2), np.log(d_var / 2)])
    res = minimize(negloglik, x0, method="Nelder-Mead", options=dict(maxiter=2000, xatol=1e-5, fatol=1e-6))
    u_rho, u_theta, mu, u_q, u_r = res.x
    return dict(rho=float(np.tanh(u_rho)), theta=float(np.exp(u_theta)), mu=float(mu),
                q=float(np.exp(u_q)), r=float(np.exp(u_r)))


def calibrate_to_target(d: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rescale `d` to `y`'s scale via OLS (`a*d+b`, fit by regressing d
    against y) -- PLOTTING ONLY. Pearson r is invariant to any affine
    transform with the SAME SIGN as the correlation, and the OLS slope
    `a = cov(d,y)/var(d)` always has that same sign by construction, so
    `corr(a*d+b, y) == corr(d, y)` exactly -- this changes nothing about
    any reported above-null number, only how the raw (unsmoothed) curve
    looks next to `y` in the figure. Fixes the user-flagged "RRR line is
    on a totally different y-axis scale than true/PLS" visual issue,
    distinct from the earlier lambda-grid bug (§23) -- this is the
    scale-invariant-statistic-vs-scale-sensitive-plot gap, not a fit
    problem."""
    if np.std(d) < 1e-10:
        return d
    a, b = np.polyfit(d, y, 1)
    return a * d + b


def apply_lgar1(d, y, rng, n_shuf, real_trial_len):
    """Fits LG-AR1 on real d, smooths, scores real + null (shift applied
    to y and d together, matching them positionally -- the null here
    reuses the ALREADY-computed shift-null decoder outputs, see caller)."""
    params = fit_lgar1(d)
    smoothed = lgar1_smooth(d, **params)
    r_smoothed = pearsonr(smoothed, y)[0]
    return r_smoothed, params, smoothed


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------

def process_session(mouse: str, session_id: str) -> dict | None:
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, load_session_unit_spikes,
        prep_perfquant_curve_targets, sliding_bin_population_matrices,
    )

    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    area_labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))

    targets_df = prep_perfquant_curve_targets(dataset_root, session_id, sessions_tbl, trials_tbl)
    if targets_df is None or len(targets_df) < MIN_TRIALS_FOR_REGRESSION:
        print(f"{mouse}/{session_id}: no usable targets, skipped")
        return None
    unit_ids = area_units(session_id, "whole_brain", "All units", area_labels)
    unit_spikes = load_session_unit_spikes(dataset_root, session_id)
    start_time = targets_df["start_time"].to_numpy()
    is_whisker = np.ones(len(targets_df), dtype=bool)

    # PLS: one aggregated window over the FULL baseline span.
    X_pls_list = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker,
                                                  [WIDE_BASELINE_WINDOW], dead_zone=None)
    X_pls = X_pls_list[0]  # (K, N)

    # RRR: T bins across the SAME baseline span.
    lo, hi = WIDE_BASELINE_WINDOW
    n_bins = int(round((hi - lo) / RRR_BIN_WIDTH))
    bin_edges = [(round(lo + i * RRR_BIN_WIDTH, 10), round(lo + (i + 1) * RRR_BIN_WIDTH, 10)) for i in range(n_bins)]
    mats = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges, dead_zone=None)
    X_rrr = np.stack(mats, axis=-1)  # (K, N, T)

    valid = ~np.isnan(X_pls).any(axis=1) & ~np.isnan(X_rrr).any(axis=(1, 2))
    X_pls, X_rrr = X_pls[valid], X_rrr[valid]
    keep_units = (np.nanstd(X_pls, axis=0) > 1e-6) & (np.nanstd(X_rrr, axis=(0, 2)) > 1e-6)
    X_pls, X_rrr = X_pls[:, keep_units], X_rrr[:, keep_units, :]
    targets_valid = targets_df[TARGETS].to_numpy()[valid]
    if len(X_pls) < MIN_TRIALS_FOR_REGRESSION or X_pls.shape[1] < 2:
        print(f"{mouse}/{session_id}: too few valid trials/units, skipped")
        return None
    print(f"{mouse}/{session_id}: K={len(X_pls)} trials, N={X_pls.shape[1]} units, "
          f"RRR T={X_rrr.shape[2]} bins ({RRR_BIN_WIDTH*1000:.0f}ms each)", flush=True)

    out = dict(mouse=mouse, session_id=session_id, n_trials=len(X_pls), n_units=X_pls.shape[1],
               n_bins=X_rrr.shape[2], targets={})
    rng_master = np.random.default_rng(abs(hash(session_id)) % (2**31))

    for target in TARGETS:
        y_raw = targets_valid[:, TARGETS.index(target)]
        y = (y_raw - y_raw.mean()) / (y_raw.std() if y_raw.std() > 0 else 1.0)
        n = len(y)
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))

        # --- fix hyperparameters on REAL data only ---
        n_1se = select_pls_1se_components(X_pls, y, rng)
        R, lam = select_rrr_hyperparams(X_rrr, y, rng)
        if lam == LAM_GRID[-1]:
            print(f"  WARNING: {mouse}/{session_id}/{target}: selected lambda={lam:g} is the grid's MAX "
                  f"-- search likely hit a ceiling rather than finding an interior optimum, widen LAM_GRID further",
                  flush=True)

        # --- real fits ---
        d_pls_real = pls_cv_predict(X_pls, y, n_1se, rng)
        d_rrr_real = rrr_cv_predict(X_rrr, y, R, lam, N_FOLDS, rng)
        pls_r = pearsonr(d_pls_real, y)[0]
        rrr_r = pearsonr(d_rrr_real, y)[0]
        pls_smoothed_r, pls_params, pls_smoothed_curve = apply_lgar1(d_pls_real, y, rng, N_SHUF_NULL, n)
        rrr_smoothed_r, rrr_params, rrr_smoothed_curve = apply_lgar1(d_rrr_real, y, rng, N_SHUF_NULL, n)
        # Affine-correct the smoothed latent z back to the observation (target)
        # scale via the fitted theta/mu -- for PLOTTING only (r itself is already
        # scale-invariant and unaffected); avoids the visually-misleading raw-
        # latent-scale plot §21 flagged.
        # theta/mu map the latent z back to d's OWN native scale, which can
        # itself still differ from y's scale (ridge regression calibrates
        # for prediction error, not variance-matching) -- a second,
        # correlation-preserving OLS calibration to y is layered on top for
        # plotting (composition of two same-signed affine maps is still a
        # same-signed affine map, so r is still exactly unchanged).
        pls_smoothed_curve_obs = calibrate_to_target(
            pls_params["theta"] * pls_smoothed_curve + pls_params["mu"], y)
        rrr_smoothed_curve_obs = calibrate_to_target(
            rrr_params["theta"] * rrr_smoothed_curve + rrr_params["mu"], y)

        # --- null: shift Y, rerun the WHOLE pipeline (both decoders, both with/without smoothing) ---
        null_pls, null_rrr, null_pls_smoothed, null_rrr_smoothed = [], [], [], []
        min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
        for _ in range(N_SHUF_NULL):
            shift = int(rng.integers(min_shift, max_shift + 1))
            if rng.random() < 0.5:
                Xp_s, Xr_s, y_s = X_pls[: n - shift], X_rrr[: n - shift], y[shift:]
            else:
                Xp_s, Xr_s, y_s = X_pls[shift:], X_rrr[shift:], y[: n - shift]
            d_pls_n = pls_cv_predict(Xp_s, y_s, n_1se, rng)
            d_rrr_n = rrr_cv_predict(Xr_s, y_s, R, lam, N_FOLDS, rng)
            null_pls.append(pearsonr(d_pls_n, y_s)[0])
            null_rrr.append(pearsonr(d_rrr_n, y_s)[0])
            null_pls_smoothed.append(apply_lgar1(d_pls_n, y_s, rng, 0, len(y_s))[0])
            null_rrr_smoothed.append(apply_lgar1(d_rrr_n, y_s, rng, 0, len(y_s))[0])

        def summarize(real, nulls):
            nulls = np.array(nulls)
            return dict(real=real, null_mean=float(np.nanmean(nulls)), above_null=real - float(np.nanmean(nulls)))

        out["targets"][target] = dict(
            n_components=n_1se, R=R, lam=lam,
            pls=summarize(pls_r, null_pls),
            rrr=summarize(rrr_r, null_rrr),
            pls_lgar1=summarize(pls_smoothed_r, null_pls_smoothed),
            rrr_lgar1=summarize(rrr_smoothed_r, null_rrr_smoothed),
            pls_rho=pls_params["rho"], rrr_rho=rrr_params["rho"],
            y=y, d_pls=d_pls_real, d_rrr=d_rrr_real,
            d_pls_obs=calibrate_to_target(d_pls_real, y), d_rrr_obs=calibrate_to_target(d_rrr_real, y),
            pls_smoothed_curve_obs=pls_smoothed_curve_obs, rrr_smoothed_curve_obs=rrr_smoothed_curve_obs,
        )
        t = out["targets"][target]
        print(f"  {target}: PLS above-null={t['pls']['above_null']:+.3f} | "
              f"PLS+LGAR1 above-null={t['pls_lgar1']['above_null']:+.3f} (rho={t['pls_rho']:.3f}) | "
              f"RRR above-null={t['rrr']['above_null']:+.3f} | "
              f"RRR+LGAR1 above-null={t['rrr_lgar1']['above_null']:+.3f} (rho={t['rrr_rho']:.3f})", flush=True)

    return out


def _process_session_worker(args):
    mouse, sid, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    return process_session(mouse, sid)


def main():
    import pickle
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cache_path = OUT_DIR / "083_pls_rrr_lgar1_cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        print(f"loaded {len(results)} sessions from cache ({cache_path.name}) -- delete it to force a recompute",
              flush=True)
        results.sort(key=lambda r: [m for m, _ in RECIPIENTS].index(r["mouse"]))
        _finish(results)
        return

    results = []
    with ProcessPoolExecutor(max_workers=len(RECIPIENTS)) as ex:
        futures = {ex.submit(_process_session_worker, (m, s, SCRIPTS_DIR)): (m, s) for m, s in RECIPIENTS}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                print(f"  {futures[fut]} raised {type(e).__name__}: {e}", flush=True)
                continue
            if res is not None:
                results.append(res)
    results.sort(key=lambda r: [m for m, _ in RECIPIENTS].index(r["mouse"]))
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    print(f"cached {len(results)} sessions -> {cache_path.name}", flush=True)
    _finish(results)


def _finish(results):
    records = []
    for res in results:
        for target, t in res["targets"].items():
            rec = dict(mouse=res["mouse"], session_id=res["session_id"], target=target,
                       n_trials=res["n_trials"], n_units=res["n_units"], n_bins=res["n_bins"],
                       n_components=t["n_components"], R=t["R"], lam=t["lam"],
                       pls_rho=t["pls_rho"], rrr_rho=t["rrr_rho"])
            for method in ("pls", "rrr", "pls_lgar1", "rrr_lgar1"):
                for k, v in t[method].items():
                    rec[f"{method}_{k}"] = v
            records.append(rec)
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(OUT_DIR / "083_pls_rrr_lgar1_comparable_summary.csv", index=False)
    print("\nsaved 083_pls_rrr_lgar1_comparable_summary.csv")

    print("\n=== MATCHED COMPARISON: baseline/ITI window, same data, same scaling ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['mouse']:<8} {row['target']:<20} "
              f"PLS={row['pls_above_null']:+.3f}  PLS+LGAR1={row['pls_lgar1_above_null']:+.3f}  "
              f"RRR={row['rrr_above_null']:+.3f}  RRR+LGAR1={row['rrr_lgar1_above_null']:+.3f}")

    for method in ("pls_above_null", "pls_lgar1_above_null", "rrr_above_null", "rrr_lgar1_above_null"):
        print(f"  mean {method}: {summary_df[method].mean():+.3f}")

    # --- Figure: 4-method bar comparison, one panel per (session,target) ---
    methods = ["pls", "rrr", "pls_lgar1", "rrr_lgar1"]
    method_labels = ["PLS+1SE", "RRR", "PLS+LGAR1", "RRR+LGAR1"]
    colors = ["#1f77b4", "#ff7f0e", "#1f77b4", "#ff7f0e"]
    hatches = ["", "", "//", "//"]
    n_sessions = len(results)
    fig, axes = plt.subplots(n_sessions, len(TARGETS), figsize=(4.2 * len(TARGETS), 3.2 * n_sessions),
                              constrained_layout=True, squeeze=False)
    for row_i, res in enumerate(results):
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i][col_i]
            t = res["targets"][target]
            vals = [t["pls"]["above_null"], t["rrr"]["above_null"],
                    t["pls_lgar1"]["above_null"], t["rrr_lgar1"]["above_null"]]
            x = np.arange(4)
            ax.bar(x, vals, color=colors, hatch=hatches, edgecolor="black", linewidth=0.5)
            ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, fontsize=6.5, rotation=20, ha="right")
            ax.set_title(f"{res['mouse']} -- {target}", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("above-null Pearson r", fontsize=8.5)
    fig.suptitle("Matched PLS+1SE vs RRR (+/- LG-AR1), baseline/ITI window, same data+scaling", fontsize=12)
    fig_path = OUT_DIR / "083_pls_rrr_lgar1_comparable.png"
    fig.savefig(fig_path, dpi=140)
    print(f"saved {fig_path.name}")

    # --- Figure 2: true vs each method's per-trial curve, one panel per (session, target) ---
    fig2, axes2 = plt.subplots(n_sessions, len(TARGETS), figsize=(4.8 * len(TARGETS), 3.2 * n_sessions),
                                constrained_layout=True, squeeze=False)
    for row_i, res in enumerate(results):
        for col_i, target in enumerate(TARGETS):
            ax = axes2[row_i][col_i]
            t = res["targets"][target]
            trial_idx = np.arange(len(t["y"]))
            ax.plot(trial_idx, t["y"], color="#000000", lw=1.4, label="true", zorder=5)
            ax.plot(trial_idx, t["d_pls_obs"], color="#1f77b4", lw=0.7, alpha=0.5, label="PLS+1SE")
            ax.plot(trial_idx, t["d_rrr_obs"], color="#ff7f0e", lw=0.7, alpha=0.5, label="RRR")
            ax.plot(trial_idx, t["pls_smoothed_curve_obs"], color="#1f77b4", lw=1.6, linestyle="--", label="PLS+LGAR1")
            ax.plot(trial_idx, t["rrr_smoothed_curve_obs"], color="#ff7f0e", lw=1.6, linestyle="--", label="RRR+LGAR1")
            ax.set_title(f"{res['mouse']} -- {target}", fontsize=8)
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=6, frameon=False, ncol=2)
    fig2.suptitle("True vs per-trial curves: PLS+1SE, RRR, and their LG-AR1-smoothed versions "
                  "(baseline/ITI window)", fontsize=12)
    fig2_path = OUT_DIR / "083_pls_rrr_lgar1_curves.png"
    fig2.savefig(fig2_path, dpi=140)
    print(f"saved {fig2_path.name}")
    print("DONE_083")


if __name__ == "__main__":
    main()
