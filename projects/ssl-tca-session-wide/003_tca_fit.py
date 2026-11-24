"""Non-negative TCA per session, per epoch tensor and normalisation, ranks 1..R with replicates and a held-out check.

Epoch tensors (from the 5-ms tensor of 002; bins overlapping the artefact window -10..+5 ms are never used):
  baseline  -1.00 .. -0.05 s, 50-ms bins (19)     pre-stimulus state
  evoked     0.005 .. 0.050 s, 5-ms bins (9)      early sensory response (user: 5-50 ms)
  late       0.050 .. 1.000 s, 25-ms bins (38)    decision / action / reward
  whole     -0.50 .. 1.00 s, 10-ms bins, bins -10..+10 ms dropped (148)
  trial     -0.10 .. 0.50 s, 10-ms bins, bins -10..+10 ms dropped (58)   (user 2026-11-21; the megamouse window was
            -0.1..0.2 / 0.4 s)
Normalisation (per unit, rates in Hz):
  minmax    (x - min) / (max - min) over trials x time (the user's megamouse choice)
  soft      x / (max of the trial-averaged rate + 5 Hz)   (soft normalisation, Williams et al. 2018)
  bsub      per-trial baseline subtracted (unit rate in -1.0..-0.010 s of the same trial), then / (max |trial-averaged
            change| + 5 Hz); signed, so fitted with unconstrained CP (cp_als; held-out with mcp_als). Removes the slow
            pre-stimulus state from the evoked / late epochs (not run for the baseline epoch).
Units with mean rate < 0.5 Hz inside the epoch are dropped (normalisation would amplify noise).
Fit: tensortools ncp_hals (minmax, soft) or cp_als (bsub), ranks 1..12, 8 replicates on all entries; held-out: 3 fits per rank with 10 % of
(trial, unit) fibres masked, test error on the masked fibres (relative SSE).

Run: python 003_tca_fit.py [--pilot] [--epochs ...] [--norms ...] [--n-jobs 60] [--skip-done]
     -> OUT/artifacts/tca/<norm>/<epoch>/<session_id>.npz (factors per rank, objectives, similarity, held-out error)
"""
import argparse
import time
import traceback

import numpy as np
import pandas as pd
import tensortools as tt
from joblib import Parallel, delayed

import tca_lib as L

EPOCHS = {"baseline": (-1.0, -0.05, 0.05), "evoked": (0.005, 0.050, 0.005), "late": (0.050, 1.0, 0.025),
          "whole": (-0.5, 1.0, 0.010), "trial": (-0.1, 0.5, 0.010),
          # 50-ms versions (user 2026-11-22: 10-ms single-trial bins too noisy for signed CP); rate over the valid
          # (non-artefact) part of each bin
          "trial_b50": (-0.1, 0.5, 0.050), "late_b50": (0.050, 1.0, 0.050), "whole_b50": (-0.5, 1.0, 0.050)}
BASE_WIN = (-1.0, -0.010)
DROP_WHOLE = (-0.010, 0.010)
NORMS = ["minmax", "soft", "bsub"]
SIGNED = {"bsub"}
RANKS = range(1, 13)
N_REP = 8
N_CV = 3
CV_FRAC = 0.10
MIN_RATE = 0.5
TCA_DIR = L.OUT / "artifacts" / "tca"


def epoch_tensor(X, tc, epoch):
    """Re-bin the 5-ms count tensor to the epoch's bins; returns rates (Hz), bin centres."""
    a, b, w = EPOCHS[epoch]
    k = int(round(w / L.CONFIG["bin_s"]))
    sel = (tc > a) & (tc < b)
    Xe, te = X[:, :, sel].astype(np.float32), tc[sel]
    if epoch.endswith("_b50"):   # rate over the valid part of each bin (artefact 5-ms bins excluded)
        lo, hi = L.CONFIG["artifact_window_s"]
        ok = ~((te > lo) & (te < hi))
        nb = len(te) // k
        nv = ok[:nb * k].reshape(nb, k).sum(1)
        Xe = (Xe[:, :, :nb * k] * ok[:nb * k]).reshape(Xe.shape[0], Xe.shape[1], nb, k).sum(3)
        te = te[:nb * k].reshape(nb, k).mean(1)
        return Xe[:, :, nv > 0] / (nv[nv > 0] * L.CONFIG["bin_s"]), te[nv > 0]
    nb = Xe.shape[2] // k
    Xe = Xe[:, :, :nb * k].reshape(Xe.shape[0], Xe.shape[1], nb, k).sum(3) / w
    te = te[:nb * k].reshape(nb, k).mean(1)
    if epoch in ("whole", "trial"):
        keep = (te + w / 2 <= DROP_WHOLE[0]) | (te - w / 2 >= DROP_WHOLE[1])
        Xe, te = Xe[:, :, keep], te[keep]
    return Xe, te


def baseline_rate(X, tc):
    """Per-trial, per-unit rate (Hz) in BASE_WIN (5-ms tensor; the artefact window is already blank)."""
    sel = (tc > BASE_WIN[0]) & (tc < BASE_WIN[1])
    return X[:, :, sel].sum(2, dtype=np.float32) / (sel.sum() * L.CONFIG["bin_s"])


def normalise(Xe, norm, base=None):
    if norm == "bsub":
        D = Xe - base[:, :, None]
        return D / (np.abs(D.mean(0, keepdims=True)).max(2, keepdims=True) + 5.0)
    if norm == "minmax":
        mn, mx = Xe.min((0, 2), keepdims=True), Xe.max((0, 2), keepdims=True)
        return (Xe - mn) / (mx - mn + 1e-9)
    if norm == "soft":
        return Xe / (Xe.mean(0, keepdims=True).max(2, keepdims=True) + 5.0)
    raise ValueError(norm)


def fit_session(sid, epoch, norm, seed=0):
    out = TCA_DIR / norm / epoch / f"{sid}.npz"
    try:
        t0 = time.time()
        d = np.load(L.TENSORS / f"{sid}.npz")
        Xe, te = epoch_tensor(d["X"], d["t_centres"], epoch)
        if norm == "bsub" and epoch == "baseline":
            raise ValueError("bsub is not defined for the baseline epoch")
        keep_u = Xe.mean((0, 2)) >= MIN_RATE
        base = baseline_rate(d["X"], d["t_centres"])[:, keep_u] if norm == "bsub" else None
        Y = normalise(Xe[:, keep_u], norm, base).astype(np.float64)
        signed = norm in SIGNED

        def fit(r, mask=None, seed_=None):
            kw = dict(random_state=seed_, verbose=False, max_iter=500, tol=1e-6)
            if signed:
                return tt.mcp_als(Y, r, mask, **kw) if mask is not None else tt.cp_als(Y, r, **kw)
            return tt.ncp_hals(Y, r, mask=mask, **kw)
        rng = np.random.default_rng(seed)
        res = {"t_centres": te, "units_kept": np.flatnonzero(keep_u), "shape": np.array(Y.shape)}
        nrm = np.linalg.norm(Y)
        for r in RANKS:
            fits = [fit(r, seed_=int(rng.integers(1 << 31))) for _ in range(N_REP)]
            fits.sort(key=lambda f: f.obj)
            best = fits[0].factors
            best.rebalance()
            if signed:   # sign convention: unit and time factors with positive mean, sign moved to the trial factor
                for k in range(r):
                    for m in (1, 2):
                        sg = np.sign(best.factors[m][:, k].sum()) or 1.0
                        best.factors[m][:, k] *= sg
                        best.factors[0][:, k] *= sg
            res[f"r{r}_obj"] = np.array([f.obj for f in fits])
            res[f"r{r}_sim"] = np.array([tt.kruskal_align(best, f.factors, permute_U=False, permute_V=False)
                                         for f in fits[1:]]) if r > 0 else np.array([])
            for m, U in enumerate(best.factors):
                res[f"r{r}_f{m}"] = U.astype(np.float32)
            # held-out: mask 10 % of (trial, unit) fibres
            cv = []
            for _ in range(N_CV):
                fib = rng.random(Y.shape[:2]) < CV_FRAC
                M = np.repeat(~fib[:, :, None], Y.shape[2], 2)
                f = fit(r, mask=M, seed_=int(rng.integers(1 << 31)))
                Yh = f.factors.full()
                cv.append([np.linalg.norm((Y - Yh)[~M]) / np.linalg.norm(Y[~M]),
                           np.linalg.norm((Y - Yh)[M]) / np.linalg.norm(Y[M])])
            res[f"r{r}_cv"] = np.array(cv)   # columns: test, train relative error
        res["norm_Y"] = nrm
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res)
        msg = dict(session_id=sid, epoch=epoch, norm=norm, ok=True, secs=round(time.time() - t0), shape=str(Y.shape))
    except Exception:
        msg = dict(session_id=sid, epoch=epoch, norm=norm, ok=False, error=traceback.format_exc()[-600:])
    print(msg, flush=True)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--epochs", nargs="*", default=list(EPOCHS))
    ap.add_argument("--norms", nargs="*", default=NORMS)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-jobs", type=int, default=60)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    if a.pilot:
        s = s[s.pilot]
    if a.sessions:
        s = s[s.session_id.isin(a.sessions)]
    s = s[[(L.TENSORS / f"{x}_trials.parquet").exists() for x in s.session_id]]
    jobs = [(sid, e, n) for n in a.norms for e in a.epochs for sid in s.session_id
            if not (n == "bsub" and e == "baseline") and not (a.skip_done and (TCA_DIR / n / e / f"{sid}.npz").exists())]
    print(f"{len(jobs)} fits", flush=True)
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(fit_session)(*j) for j in jobs)
    f = L.OUT / "tables" / "tca_fit_log.csv"
    out = pd.DataFrame(res)
    if f.exists():
        out = pd.concat([pd.read_csv(f), out]).drop_duplicates(["session_id", "epoch", "norm"], keep="last")
    out.to_csv(f, index=False)


if __name__ == "__main__":
    main()
