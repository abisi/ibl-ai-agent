"""RRR (single-session reduced-rank regression decoder) + LG-AR1 (linear-
Gaussian AR(1) Kalman smoother) pilot, per user spec 2026-09-23 -- a
from-scratch alternative to this project's PLS+1SE single-window
pipeline: RRR decodes a per-trial estimate from a full per-trial spike
TIME COURSE (N units x T bins), not one aggregated window; LG-AR1 then
smooths the sequence of per-trial estimates across trials using an AR(1)
state-space model, exploiting trial-to-trial persistence.

**Deviations from the user's minimal sketch, and why:**

1. **Dual-ridge (Woodbury) reformulation, not the sketch's primal solve.**
   The sketch's U-step ridge-solves a system of size N*R (features =
   neuron x rank). At whole_brain scale (N up to ~2500, R~2) that's a
   ~5000-dim linear system PER alternating-ridge iteration, repeated
   across CV folds x hyperparameters x null draws x targets x sessions --
   computationally infeasible as written. `w = (F^T F + lam I)^-1 F^T y`
   is mathematically identical to `w = F^T (F F^T + lam I_K)^-1 y` (the
   standard ridge dual form), and the latter solves a K x K system
   (K = n_trials ~ 100-250) instead -- orders of magnitude cheaper here
   since K << N*R. Used for the U-step (where it matters); the V-step
   solves in R*T-space, small enough (~20-60) not to need it, but the
   same helper is used for both for simplicity/uniformity.

2. **Out-of-fold RRR predictions via explicit K-fold CV** (5-fold here)
   -- the user's own stated requirement ("use cross-validated RRR
   predictions for d so the smoother never sees in-sample fits"). Rank R
   and ridge lambda are chosen once via a lighter 3-fold inner CV grid
   search on the REAL (unshifted) data only, then reused for both the
   real 5-fold out-of-fold fit and every null shuffle -- same "fix
   hyperparameters once from real data" convention this project uses
   throughout (e.g. PLS's component count).

3. **LG-AR1 parameters (rho, theta, mu, q, r) fit by directly maximizing
   the Kalman marginal likelihood** (`scipy.optimize.minimize`,
   Nelder-Mead) rather than EM -- the user's own prose names marginal-
   likelihood maximization as an equivalent method ("maximizing the
   marginal likelihood with EM" -- EM IS one way to do that maximization;
   direct optimization is the same objective, simpler to implement
   correctly under this pilot's time budget). Parameters transformed
   (rho via tanh, q/r/theta via exp) to respect their natural domains.

4. **Null: user's explicit caveat, implemented literally** -- "Pass
   every null session through the exact same RRR + LG-AR1 pipeline, and
   compare the smoothed correlation to that null rather than to the
   unsmoothed baseline." A linear-shift null (this project's standing
   confound-control convention, matches `058` etc.) is applied to the
   TARGET before EVERY step (RRR out-of-fold fit AND LG-AR1 fit AND
   smoothing), not just at the final correlation step -- exactly the
   failure mode the user flagged (a smoother naively compared to an
   unsmoothed baseline would look great even on pure drift). RRR
   hyperparameters (R, lambda) are reused from the real fit for cost
   reasons; LG-AR1 parameters are refit fresh per null draw (cheap,
   Nelder-Mead over 5 scalars).

**Scope, deliberately small ("test on a few sessions")**: 3 sessions
(MH070, MH069, MH031 -- the same top R+ learning-stage sessions already
used in `079`/`081`), all 3 targets, whole_brain scheme, `N_SHUF_NULL=20`.
Bin window (-0.05, 0.25)s relative to stim onset, 20ms disjoint bins via
`dead_zone_offset_bin_edges` (respects the mandatory whisker-artifact
dead zone, drops the one bin containing it) -- T~14 bins, much shorter
than the original paper's 2s/100-bin ITI-based design, since this
project's task is a fast sensory-evoked response, not a slow prior.
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
BIN_WINDOW = (-0.05, 0.25)
BIN_WIDTH = 0.02
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
MIN_TRIALS_FOR_REGRESSION = 25
N_FOLDS = 5
N_FOLDS_INNER = 3
R_GRID = (1, 2, 3)
LAM_GRID = (3.0, 10.0, 30.0, 100.0)
N_SHUF_NULL = 20
MIN_SHIFT_FRAC, MAX_SHIFT_FRAC = 0.1, 0.5

RECIPIENTS = [
    ("MH070", "MH070_20260121_140848"),
    ("MH069", "MH069_20260122_111455"),
    ("MH031", "MH031_20250507_104425"),
]


# ---------------------------------------------------------------------------
# RRR: dual-ridge alternating fit
# ---------------------------------------------------------------------------

def _dual_ridge_solve(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """w minimizing ||y - F w||^2 + lam ||w||^2, via the K x K dual form
    (K = F.shape[0]) instead of the P x P primal (P = F.shape[1]) --
    identical solution, cheap when K << P (always true here: K = n_trials
    ~100-250, P = N_units*R up to ~5000+)."""
    K = F.shape[0]
    alpha = np.linalg.solve(F @ F.T + lam * np.eye(K), y)
    return F.T @ alpha


def fit_rrr(X: np.ndarray, y: np.ndarray, R: int, lam: float, n_iter: int = 30, seed: int = 0):
    """X: (K, N, T), y: (K,). Returns U (N,R), V (R,T), b (scalar)."""
    K, N, T = X.shape
    Xc, y_mean = X - X.mean(axis=0, keepdims=True), y.mean()
    yc = y - y_mean
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((N, R)) / np.sqrt(N)
    for _ in range(n_iter):
        Z = np.einsum("nr,knt->krt", U, Xc).reshape(K, R * T)
        V = _dual_ridge_solve(Z, yc, lam).reshape(R, T)
        F = np.einsum("knt,rt->knr", Xc, V).reshape(K, N * R)
        U = _dual_ridge_solve(F, yc, lam).reshape(N, R)
    b = y_mean - float(np.einsum("knt,nt->k", X, U @ V).mean())
    return U, V, b


def predict_rrr(X: np.ndarray, U: np.ndarray, V: np.ndarray, b: float) -> np.ndarray:
    return np.einsum("knt,nt->k", X, U @ V) + b


def rrr_cv_predict(X: np.ndarray, y: np.ndarray, R: int, lam: float, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Out-of-fold RRR predictions, one per trial, IN ORIGINAL TRIAL ORDER."""
    from sklearn.model_selection import KFold
    d = np.full(len(y), np.nan)
    seed = int(rng.integers(0, 2**31 - 1))
    for tr, te in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
        U, V, b = fit_rrr(X[tr], y[tr], R, lam, seed=int(rng.integers(0, 2**31 - 1)))
        d[te] = predict_rrr(X[te], U, V, b)
    return d


def select_rrr_hyperparams(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[int, float]:
    """Light grid search (R x lambda), `N_FOLDS_INNER`-fold CV, real data only."""
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
                U, V, b = fit_rrr(X[tr], y[tr], R, lam, seed=0)
                pred = predict_rrr(X[te], U, V, b)
                if np.std(pred) < 1e-8:
                    scores.append(-1.0)
                    continue
                scores.append(pearsonr(pred, y[te])[0])
            mean_score = np.nanmean(scores)
            if mean_score > best_score:
                best_score, best_R, best_lam = mean_score, R, lam
    return best_R, best_lam


# ---------------------------------------------------------------------------
# LG-AR1: Kalman filter/smoother + marginal-likelihood parameter fit
# ---------------------------------------------------------------------------

def lgar1_filter_loglik(d: np.ndarray, rho: float, theta: float, mu: float, q: float, r: float):
    """Forward Kalman filter; returns (m_f, P_f, m_p, P_p, loglik)."""
    K = len(d)
    m_f, P_f, m_p, P_p = (np.zeros(K) for _ in range(4))
    m0 = 0.0
    P0 = q / max(1 - rho ** 2, 1e-6)
    loglik = 0.0
    m, P = m0, P0
    for k in range(K):
        if k == 0:
            mp, Pp = m0, P0
        else:
            mp, Pp = rho * m, rho ** 2 * P + q
        m_p[k], P_p[k] = mp, Pp
        S = theta ** 2 * Pp + r
        resid = d[k] - (theta * mp + mu)
        loglik += -0.5 * np.log(2 * np.pi * S) - 0.5 * resid ** 2 / S
        G = Pp * theta / S
        m = mp + G * resid
        P = (1 - G * theta) * Pp
        m_f[k], P_f[k] = m, P
    return m_f, P_f, m_p, P_p, loglik


def lgar1_smooth(d: np.ndarray, rho: float, theta: float, mu: float, q: float, r: float) -> np.ndarray:
    m_f, P_f, m_p, P_p, _ = lgar1_filter_loglik(d, rho, theta, mu, q, r)
    K = len(d)
    m_s = m_f.copy()
    for k in range(K - 2, -1, -1):
        denom = P_p[k + 1] if P_p[k + 1] > 1e-10 else 1e-10
        J = P_f[k] * rho / denom
        m_s[k] = m_f[k] + J * (m_s[k + 1] - m_p[k + 1])
    return m_s


def fit_lgar1(d: np.ndarray) -> dict:
    """MLE of (rho, theta, mu, q, r) by direct maximization of the Kalman
    marginal likelihood -- the same objective EM maximizes, via
    `scipy.optimize.minimize` instead (see module docstring point 3)."""
    d_var = max(np.var(d), 1e-6)

    def negloglik(params):
        u_rho, u_theta, mu, u_q, u_r = params
        rho = np.tanh(u_rho)
        theta = np.exp(u_theta)
        q = np.exp(u_q)
        r = np.exp(u_r)
        try:
            _, _, _, _, ll = lgar1_filter_loglik(d, rho, theta, mu, q, r)
        except (FloatingPointError, ZeroDivisionError):
            return 1e10
        if not np.isfinite(ll):
            return 1e10
        return -ll

    x0 = np.array([np.arctanh(0.5), 0.0, float(np.mean(d)), np.log(d_var / 2), np.log(d_var / 2)])
    res = minimize(negloglik, x0, method="Nelder-Mead",
                    options=dict(maxiter=2000, xatol=1e-5, fatol=1e-6))
    u_rho, u_theta, mu, u_q, u_r = res.x
    return dict(rho=float(np.tanh(u_rho)), theta=float(np.exp(u_theta)), mu=float(mu),
                q=float(np.exp(u_q)), r=float(np.exp(u_r)))


# ---------------------------------------------------------------------------
# One session
# ---------------------------------------------------------------------------

def process_session(mouse: str, session_id: str) -> dict | None:
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, add_whole_brain_column, area_units, dead_zone_offset_bin_edges,
        load_session_unit_spikes, prep_perfquant_curve_targets, sliding_bin_population_matrices,
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

    bin_edges = dead_zone_offset_bin_edges(BIN_WINDOW, bin_width=BIN_WIDTH)
    mats = sliding_bin_population_matrices(unit_spikes, unit_ids, start_time, is_whisker, bin_edges, dead_zone=None)
    X = np.stack(mats, axis=-1)  # (K, N, T)
    valid = ~np.isnan(X).any(axis=(1, 2))
    X = X[valid]
    col_std = np.nanstd(X, axis=(0, 2))
    keep_units = col_std > 1e-6
    X = X[:, keep_units, :]
    targets_valid = targets_df[TARGETS].to_numpy()[valid]
    if len(X) < MIN_TRIALS_FOR_REGRESSION or X.shape[1] < 2:
        print(f"{mouse}/{session_id}: too few valid trials/units after binning, skipped")
        return None
    print(f"{mouse}/{session_id}: K={len(X)} trials, N={X.shape[1]} units, T={X.shape[2]} bins", flush=True)

    out = dict(mouse=mouse, session_id=session_id, n_trials=len(X), n_units=X.shape[1], n_bins=X.shape[2],
               targets={})
    rng_master = np.random.default_rng(abs(hash(session_id)) % (2**31))

    for k_idx, target in enumerate(TARGETS):
        y_raw = targets_valid[:, k_idx]
        y = (y_raw - y_raw.mean()) / (y_raw.std() if y_raw.std() > 0 else 1.0)

        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        R, lam = select_rrr_hyperparams(X, y, rng)
        d_real = rrr_cv_predict(X, y, R, lam, N_FOLDS, rng)
        rrr_only_r = pearsonr(d_real, y)[0]
        params_real = fit_lgar1(d_real)
        smoothed_real = lgar1_smooth(d_real, **params_real)
        smoothed_r = pearsonr(smoothed_real, y)[0]

        null_rrr_r, null_smoothed_r = [], []
        n = len(y)
        min_shift, max_shift = max(1, int(MIN_SHIFT_FRAC * n)), max(2, int(MAX_SHIFT_FRAC * n))
        for s in range(N_SHUF_NULL):
            shift = int(rng.integers(min_shift, max_shift + 1))
            if rng.random() < 0.5:
                X_shift, y_shift = X[: n - shift], y[shift:]
            else:
                X_shift, y_shift = X[shift:], y[: n - shift]
            d_null = rrr_cv_predict(X_shift, y_shift, R, lam, N_FOLDS, rng)
            null_rrr_r.append(pearsonr(d_null, y_shift)[0])
            params_null = fit_lgar1(d_null)
            smoothed_null = lgar1_smooth(d_null, **params_null)
            null_smoothed_r.append(pearsonr(smoothed_null, y_shift)[0])

        null_rrr_r, null_smoothed_r = np.array(null_rrr_r), np.array(null_smoothed_r)
        out["targets"][target] = dict(
            R=R, lam=lam, rrr_only_r=rrr_only_r, smoothed_r=smoothed_r,
            null_rrr_mean=float(np.nanmean(null_rrr_r)), null_smoothed_mean=float(np.nanmean(null_smoothed_r)),
            above_null_rrr=rrr_only_r - float(np.nanmean(null_rrr_r)),
            above_null_smoothed=smoothed_r - float(np.nanmean(null_smoothed_r)),
            lgar1_params=params_real, d_real=d_real, y=y, smoothed_real=smoothed_real,
        )
        print(f"  {target}: R={R} lam={lam:g} | RRR-only r={rrr_only_r:+.3f} (null={np.nanmean(null_rrr_r):+.3f}) "
              f"| LG-AR1 r={smoothed_r:+.3f} (null={np.nanmean(null_smoothed_r):+.3f}) "
              f"| rho={params_real['rho']:.3f}", flush=True)

    return out


def _process_session_worker(args: tuple):
    mouse, sid, scripts_dir = args
    sys.path.insert(0, scripts_dir)
    return process_session(mouse, sid)


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed

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

    if not results:
        print("no usable sessions, exiting")
        return

    records = []
    for res in results:
        for target, t in res["targets"].items():
            records.append(dict(mouse=res["mouse"], session_id=res["session_id"], target=target,
                                 n_trials=res["n_trials"], n_units=res["n_units"], n_bins=res["n_bins"],
                                 R=t["R"], lam=t["lam"], rho=t["lgar1_params"]["rho"],
                                 rrr_only_r=t["rrr_only_r"], null_rrr_mean=t["null_rrr_mean"],
                                 above_null_rrr=t["above_null_rrr"], smoothed_r=t["smoothed_r"],
                                 null_smoothed_mean=t["null_smoothed_mean"], above_null_smoothed=t["above_null_smoothed"]))
    summary_df = pd.DataFrame(records)
    summary_df.to_csv(OUT_DIR / "082_rrr_lgar1_pilot_summary.csv", index=False)
    print(f"\nsaved 082_rrr_lgar1_pilot_summary.csv")

    print("\n=== SUMMARY: RRR-only vs RRR+LG-AR1, real vs shift-null ===")
    for _, row in summary_df.iterrows():
        print(f"  {row['mouse']:<8} {row['target']:<20} RRR above-null={row['above_null_rrr']:+.3f}  "
              f"LG-AR1 above-null={row['above_null_smoothed']:+.3f}  (rho={row['rho']:.3f})")

    # --- Figure: true vs RRR-only vs LG-AR1-smoothed, per (session, target) ---
    fig, axes = plt.subplots(len(results), len(TARGETS), figsize=(4.5 * len(TARGETS), 3.2 * len(results)),
                              constrained_layout=True, squeeze=False)
    for row_i, res in enumerate(results):
        for col_i, target in enumerate(TARGETS):
            ax = axes[row_i][col_i]
            t = res["targets"][target]
            trial_idx = np.arange(len(t["y"]))
            ax.plot(trial_idx, t["y"], color="#333333", lw=1.2, label="true")
            ax.plot(trial_idx, t["d_real"], color="#999999", lw=0.7, alpha=0.6, label="RRR (per-trial)")
            ax.plot(trial_idx, t["smoothed_real"], color="#d62728", lw=1.4, label="LG-AR1 smoothed")
            ax.set_title(f"{res['mouse']} -- {target}\nRRR r={t['rrr_only_r']:.2f} (null={t['null_rrr_mean']:.2f}) | "
                          f"smoothed r={t['smoothed_r']:.2f} (null={t['null_smoothed_mean']:.2f})", fontsize=7.5)
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
            if row_i == 0 and col_i == 0:
                ax.legend(fontsize=6.5, frameon=False)
    fig.suptitle("RRR + LG-AR1 pilot: true vs per-trial RRR estimate vs Kalman-smoothed estimate", fontsize=12)
    fig_path = OUT_DIR / "082_rrr_lgar1_pilot.png"
    fig.savefig(fig_path, dpi=140)
    print(f"saved {fig_path.name}")
    print("DONE_082")


if __name__ == "__main__":
    main()
