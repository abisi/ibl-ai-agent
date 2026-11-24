"""Jaw-onset-aligned TCA per session (pilot): decision / pre-movement signals (user 2026-11-21).

Tensor: re-indexed from the 5-ms stimulus-aligned tensor of 002 (-1..+1 s, so no NWB reload), window -0.30..+0.20 s
around jaw onset, 10-ms bins (50). Alignment error <= 2.5 ms (5-ms grid).
Jaw onset: user's per-mouse table combined_results_ks4/<mouse>/dlc_jaw_onset_times.pkl (`jaw_dlc_onset`, s from
trial start; dlc_helpers.movement_onset_times, 2026-11-12 parameters), joined on start_time.
Variants:
  jaw      active lick trials (whisker hits, auditory hits, false alarms) with a valid onset:
           0.015 s <= onset <= 0.79 s (window inside the tensor; pre-stimulus / artefact-time onsets dropped) and
           onset <= corrected first lick.
  jaw_all  all selected trials: lick trials as above, every other trial (misses, correct rejections, passive trials,
           lick trials without a valid onset) pseudo-aligned to the session median onset of the hits of the same
           trial type (no-stim: median of all valid onsets). Allows hit vs miss on the same component.
Stimulus artefact (-10..+5 ms from stimulus, spikes removed in 002) lands inside the window on most trials: those bins
are masked (missing data, masked CP), not treated as zero. Normalisation bsub (as 003): per-trial baseline in
-1.0..-0.010 s before the stimulus subtracted, / (max |trial-averaged change| + 5 Hz); signed, masked CP (mcp_als).
Note: with median whisker-hit jaw onsets ~0.1 s, the pre-onset part of the window contains the sensory response;
compare components with the stimulus-aligned `trial` epoch before calling them decision signals.

Run: python 003b_tca_jaw.py [--variants jaw jaw_all] [--sessions ...] [--n-jobs 28] [--skip-done]   (pilot sessions)
     -> OUT/artifacts/tca/bsub/<variant>/<sid>.npz (as 003, plus trials_kept, jaw_onset, pseudo)
"""
import argparse
import importlib
import time
import traceback

import numpy as np
import pandas as pd
import tensortools as tt
from joblib import Parallel, delayed

import tca_lib as L

F = importlib.import_module("003_tca_fit")
WIN = (-0.30, 0.20)
W = 0.010
W_VARIANT = {"jaw": 0.010, "jaw_all": 0.010, "jaw_b50": 0.050, "jaw_all_b50": 0.050}   # 50 ms: user 2026-11-22
ONSET_RANGE = (0.015, 0.79)
NORM = "bsub"


def jaw_table(mouse, sid):
    d = pd.read_pickle(L.ROOT / "combined_results_ks4" / mouse / "dlc_jaw_onset_times.pkl")
    d = d[d.session_id == sid][["start_time", "jaw_dlc_onset"]]
    return d.astype(float).sort_values("start_time")


def build(sid, mouse, variant):
    d = np.load(L.TENSORS / f"{sid}.npz")
    X, tc = d["X"], d["t_centres"]
    t = pd.read_parquet(L.TENSORS / f"{sid}_trials.parquet").sort_values("trial_index").reset_index(drop=True)
    j = jaw_table(mouse, sid)
    if not len(j):
        raise RuntimeError("no jaw onsets for this session")
    t = pd.merge_asof(t.sort_values("start_time"), j, on="start_time", direction="nearest", tolerance=1e-3)
    t = t.sort_values("trial_index").reset_index(drop=True)
    on = t.jaw_dlc_onset.to_numpy(float)
    act_lick = ((t.epoch == "active") & (t.lick_flag == 1)).to_numpy()
    lat = t.lick_latency.to_numpy(float) if "lick_latency" in t else np.full(len(t), np.inf)
    valid = act_lick & (on >= ONSET_RANGE[0]) & (on <= ONSET_RANGE[1]) & ~(on > lat + 1e-9)
    W = W_VARIANT[variant]
    if variant.startswith("jaw_b") or variant == "jaw":
        keep = valid
        align = on.copy()
    else:
        keep = np.ones(len(t), bool)
        align = np.where(valid, on, np.nan)
        hits = valid & (t.trial_type != "no_stim_trial").to_numpy()
        allmed = np.nanmedian(on[valid])
        for tt_ in t.trial_type.unique():
            m = (t.trial_type == tt_).to_numpy()
            med = np.nanmedian(on[hits & m]) if (hits & m).sum() >= 3 else allmed
            align[m & ~valid] = med
    if keep.sum() < 20:
        raise RuntimeError(f"{keep.sum()} trials with a valid jaw onset")
    idx = np.flatnonzero(keep)
    nb5 = int(round((WIN[1] - WIN[0]) / L.CONFIG["bin_s"]))
    k = int(round(W / L.CONFIG["bin_s"]))
    art = (tc > L.CONFIG["artifact_window_s"][0]) & (tc < L.CONFIG["artifact_window_s"][1])
    i0 = np.rint((align[idx] + WIN[0] - tc[0] + L.CONFIG["bin_s"] / 2) / L.CONFIG["bin_s"]).astype(int)
    if i0.min() < 0 or (i0 + nb5).max() > X.shape[2]:
        raise RuntimeError("jaw window outside the stimulus tensor")
    sl = i0[:, None] + np.arange(nb5)[None]
    Xe = np.take_along_axis(X[idx].astype(np.float32), sl[:, None, :].repeat(X.shape[1], 1), 2)
    nv = (~art[sl]).reshape(len(idx), nb5 // k, k).sum(2)    # valid 5-ms sub-bins per trial x bin
    Xe = Xe.reshape(len(idx), X.shape[1], nb5 // k, k).sum(3) / np.maximum(nv, 1)[:, None, :] / L.CONFIG["bin_s"]
    mask_t = nv == 0   # trials x bins, True = bin entirely inside the artefact (missing)
    te = WIN[0] + W * (np.arange(nb5 // k) + 0.5)
    base = F.baseline_rate(X[idx], tc)
    info = t.iloc[idx].assign(jaw_align=align[idx], pseudo=~valid[idx])
    return Xe, te, mask_t, base, info


def fit_session(sid, mouse, variant, seed=0):
    out = F.TCA_DIR / NORM / variant / f"{sid}.npz"
    try:
        t0 = time.time()
        Xe, te, mask_t, base, info = build(sid, mouse, variant)
        Mt = np.repeat(mask_t[:, None, :], Xe.shape[1], 1)
        Xn = np.where(Mt, np.nan, Xe)
        keep_u = np.nanmean(Xn, (0, 2)) >= F.MIN_RATE
        D = Xn[:, keep_u] - base[:, keep_u, None]
        D = D / (np.nanmax(np.abs(np.nanmean(D, 0, keepdims=True)), 2, keepdims=True) + 5.0)
        obs = ~np.isnan(D)
        Y = np.nan_to_num(D).astype(np.float64)

        def fit(r, M, seed_):
            return tt.mcp_als(Y, r, M, random_state=seed_, verbose=False, max_iter=500, tol=1e-6)
        rng = np.random.default_rng(seed)
        res = {"t_centres": te, "units_kept": np.flatnonzero(keep_u), "shape": np.array(Y.shape),
               "trials_kept": info.trial_index.to_numpy(), "jaw_onset": info.jaw_align.to_numpy(float),
               "pseudo": info.pseudo.to_numpy(), "artefact_mask": mask_t}
        for r in F.RANKS:
            fits = sorted((fit(r, obs, int(rng.integers(1 << 31))) for _ in range(F.N_REP)), key=lambda f: f.obj)
            best = fits[0].factors
            best.rebalance()
            for c in range(r):
                for m in (1, 2):
                    sg = np.sign(best.factors[m][:, c].sum()) or 1.0
                    best.factors[m][:, c] *= sg
                    best.factors[0][:, c] *= sg
            res[f"r{r}_obj"] = np.array([f.obj for f in fits])
            res[f"r{r}_sim"] = np.array([tt.kruskal_align(best, f.factors, permute_U=False, permute_V=False)
                                         for f in fits[1:]])
            for m, U in enumerate(best.factors):
                res[f"r{r}_f{m}"] = U.astype(np.float32)
            cv = []
            for _ in range(F.N_CV):
                fib = np.repeat((rng.random(Y.shape[:2]) < F.CV_FRAC)[:, :, None], Y.shape[2], 2)
                f = fit(r, obs & ~fib, int(rng.integers(1 << 31)))
                E = Y - f.factors.full()
                te_m, tr_m = obs & fib, obs & ~fib
                cv.append([np.linalg.norm(E[te_m]) / np.linalg.norm(Y[te_m]),
                           np.linalg.norm(E[tr_m]) / np.linalg.norm(Y[tr_m])])
            res[f"r{r}_cv"] = np.array(cv)
        res["norm_Y"] = np.linalg.norm(Y)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res)
        msg = dict(session_id=sid, epoch=variant, norm=NORM, ok=True, secs=round(time.time() - t0), shape=str(Y.shape),
                   n_pseudo=int(info.pseudo.sum()), masked_frac=round(float(mask_t.mean()), 4))
    except Exception:
        msg = dict(session_id=sid, epoch=variant, norm=NORM, ok=False, error=traceback.format_exc()[-600:])
    print(msg, flush=True)
    return msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["jaw_b50"])
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-jobs", type=int, default=28)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    s = s[s.pilot] if not a.sessions else s[s.session_id.isin(a.sessions)]
    jobs = [(r.session_id, r.mouse_id, v) for v in a.variants for r in s.itertuples()
            if not (a.skip_done and (F.TCA_DIR / NORM / v / f"{r.session_id}.npz").exists())]
    print(f"{len(jobs)} fits", flush=True)
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(fit_session)(*j) for j in jobs)
    f = L.OUT / "tables" / "tca_fit_log_jaw.csv"
    out = pd.DataFrame(res)
    if f.exists():
        out = pd.concat([pd.read_csv(f), out]).drop_duplicates(["session_id", "epoch", "norm"], keep="last")
    out.to_csv(f, index=False)


if __name__ == "__main__":
    main()
