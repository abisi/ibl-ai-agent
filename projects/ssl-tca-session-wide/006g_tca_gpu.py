"""GPU version of 006 (TCA on the event-aligned tensors of 005), for Kuma (user 2026-11-22).

Same input, settings and output keys as 006_tca_aligned.py, so 007 reads either. Differences: fitting with tensorly
parafac (ALS) on the GPU (pytorch backend, float32; benchmark kuma_bench_gpu.py: same errors as tensortools, 40-90x
faster at rank 10-20) instead of tensortools cp_als on CPU; restart similarity computed with the tensortools
kruskal_align rule re-implemented (mean over modes of |cosine|, Hungarian matching, mean of matched pairs; identical to
kruskal_align to 4 decimals on AB123). Self-contained (no project imports): Kuma has no NAS mount; tensors are copied
in and results out with rclone by the job script (kuma_tca_array.sh).

Input per trial / unit: firing rate (Hz) minus that trial's baseline rate (start - 1.0 .. start - 0.010 s); units with
mean rate < 0.5 Hz (window + baseline) dropped; normalisations rate / soft / zscore (see 006). Types = 4 alignments x
normalisations (main: z-score). Ranks 1..12, 10 restarts each; best restart kept (rebalanced, sign convention: unit and time factors
positive sum, sign moved to the trial factor). Rank choice (Williams et al. 2018 trade-off): largest rank with mean
restart similarity >= 0.85 whose added component lowers the error by >= 5 % of the total decrease from rank 1 to 12.
Outliers: no capping and no outlier-trial masking (user 2026-11-22); both remain optional (CLIP_SD, OUTLIER_Z)
(mask) and given trial weights afterwards by least squares; flagged as `outlier_trials`. Errors on fitted entries only.
In-sample condition-mean model (context x trial type x lick) as reference.

Variants (user 2026-11-22, against cancelling components = CP degeneracy): --methods cp [--l2 ridge strengths]
(tensorly l2_reg) and splitnn (unit split into increase / decrease channels, non-negative CP, HALS).

Run (Kuma): venv/bin/python 006g_tca_gpu.py --session <sid> [--methods cp splitnn] [--l2 0 10 100]
     -> <out-root>/<sid>/<alignment>_<norm>[_l2-<l2> | _splitnn].npz, fit_log_<variants>.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorly as tl
from scipy.optimize import linear_sum_assignment
from scipy.optimize import nnls
from tensorly.decomposition import non_negative_parafac_hals, parafac

ALIGNMENTS = ["stim_long", "stim_sensory", "jaw", "lick", "stim_long_ns", "jaw_all"]   # new ones appended (seeds of the others unchanged)
NORMS = ["zscore"]          # main normalisation (user 2026-11-22); rate / soft still available with --norms
RANKS = range(1, 13)        # ranks 1-12 per mouse (user 2026-11-22); comparisons across mice use one fixed rank
CLIP_SD = None              # no capping (user 2026-11-22; was 10 SD: <= 0.06 % of entries, 0.5-4.4 % of the signal)
OUTLIER_Z = None            # no outlier-trial masking (user 2026-11-22; was 4 robust SD: 0-7 trials per fit)
N_REP = 10
TOL = 1e-6
MAX_ITER = 500
MIN_RATE = 0.5
SIM_MIN = 0.85
GAIN_MIN = 0.05


def prepare(root, sid, align, norm, robust=True, bsub=True):
    """006_tca_aligned.prepare (validated on AB123), plus outlier handling (user 2026-11-22, AB123 trials 162-163):
    both optional and off (user 2026-11-22): z-scored entries clipped at +-CLIP_SD if set; trials with RMS signal >
    OUTLIER_Z robust SD (median / MAD over trials) masked out of the fit if set. Returns Y, obs (fit mask), keep_u, cls, d, outlier (bool per trial)."""
    d = np.load(root / sid / f"{align}.npz")
    R, base = d["rate"], d["baseline"]
    keep_u = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= MIN_RATE
    # bsub=False (method 'nn', user 2026-11-22): rates as they are (no per-trial baseline subtraction), non-negative
    D = R[:, keep_u] - base[:, keep_u, None] if bsub else R[:, keep_u].astype(np.float64)
    if norm == "soft":
        D = D / (np.nanmax(np.abs(np.nanmean(D, 0, keepdims=True)), 2, keepdims=True) + 5.0)
    elif norm == "zscore":
        D = D / (np.nanstd(D, (0, 2), keepdims=True) + 1e-9)
    obs = ~np.isnan(D)
    outlier = np.zeros(D.shape[0], bool)
    if robust:
        if norm == "zscore" and CLIP_SD:
            D = np.clip(D, -CLIP_SD, CLIP_SD)
        if OUTLIER_Z:
            e = np.sqrt(np.nanmean(D ** 2, axis=(1, 2)))
            mad = 1.4826 * np.median(np.abs(e - np.median(e)))
            outlier = (e - np.median(e)) / max(mad, 1e-12) > OUTLIER_Z
            obs[outlier] = False
    t = pd.read_parquet(root / sid / "trials.parquet").set_index("trial_index").loc[d["trial_index"]]
    cls = (t.epoch.str.startswith("passive").map({True: "P", False: "A"}) + "_" + t.trial_type + "_"
           + t.lick_flag.fillna(0).astype(int).astype(str)).to_numpy()
    return np.nan_to_num(D).astype(np.float64), obs, keep_u, cls, d, outlier


def cond_mean_error(Y, obs, cls):
    P = np.zeros_like(Y)
    for c in np.unique(cls):
        m = cls == c
        w = obs[m]
        P[m] = (Y[m] * w).sum(0, keepdims=True) / np.maximum(w.sum(0, keepdims=True), 1)
    return np.linalg.norm((Y - P)[obs]) / np.linalg.norm(Y[obs]), len(np.unique(cls)) * Y.shape[1] * Y.shape[2]


def choose_rank(err, sim):
    R = np.array(list(RANKS))
    gain = np.r_[np.inf, -np.diff(err)] / max(err[0] - err[-1], 1e-12)
    ok = (sim >= SIM_MIN) & (gain >= GAIN_MIN)
    ok[0] = True
    return int(R[np.flatnonzero(ok)[-1]]), gain


def unit_norm(M):
    return M / (np.linalg.norm(M, axis=0, keepdims=True) + 1e-12)


def tt_similarity(fa, fb):
    """tensortools kruskal_align score: mean over modes of |cos|, Hungarian, mean of matched pairs"""
    S = np.mean([np.abs(unit_norm(a).T @ unit_norm(b)) for a, b in zip(fa, fb)], axis=0)
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean())


def to_factors(cp):
    """weights folded in, rebalanced like tensortools KTensor.rebalance (equal column norms across modes)"""
    w, fs = cp
    fs = [tl.to_numpy(f).astype(np.float64) for f in fs]
    fs[0] = fs[0] * tl.to_numpy(w).astype(np.float64)[None]
    norms = [np.linalg.norm(f, axis=0) for f in fs]
    lam = np.prod(norms, axis=0) ** (1 / len(fs))
    return [f * (lam / np.maximum(n, 1e-30))[None] for f, n in zip(fs, norms)]


def variant_tag(method, l2):
    if method == "cp":
        return f"_l2-{l2:g}" if l2 else ""
    return f"_{method}"   # _splitnn, _seminn


def congruence_min(F):
    """most negative signed Tucker congruence between two components (product of cosines over the 3 modes)"""
    if F[0].shape[1] < 2:
        return np.nan
    un = [unit_norm(f) for f in F]
    C = (un[0].T @ un[0]) * (un[1].T @ un[1]) * (un[2].T @ un[2])
    return float(C[~np.eye(len(C), dtype=bool)].min())


def fit_type(root, sid, align, norm, method="cp", l2=0.0, seed=0):
    """method 'cp': signed CP (ALS, optional ridge l2_reg against degenerate cancelling components);
    'splitnn': each unit split into an increase channel max(D, 0) and a decrease channel max(-D, 0), non-negative CP
    (HALS), so components cannot cancel; unit factor has 2 x units rows (first half increases, second decreases).
    'nn' (user 2026-11-22): plain non-negative TCA (HALS) on rates WITHOUT baseline subtraction (scaled by the unit's
    SD for zscore, not centred), errors on those rates (not comparable with the baseline-subtracted variants).
    'seminn' (user 2026-11-22): semi-non-negative CP (HALS, nn_modes = trial and time): trial and time weights >= 0,
    unit weights signed (each unit goes up or down in a component); limits cancelling components without tuning.
    Errors are always on the signed baseline-subtracted data (split model: increases minus decreases)."""
    import torch   # imported here so that haas (no torch) can import the helpers
    t0 = time.time()
    Y, obs, keep_u, cls, d, outlier = prepare(root, sid, align, norm, bsub=method != "nn")
    J = Y.shape[1]
    Yg = torch.tensor(Y, dtype=torch.float32, device="cuda")
    Og = torch.tensor(obs, dtype=torch.bool, device="cuda")
    ny = torch.linalg.norm(Yg[Og])
    Mg = None if obs.all() else torch.tensor(obs, dtype=torch.float32, device="cuda")
    fit_rows = ~outlier
    if method == "splitnn":
        Y2 = np.concatenate([np.maximum(Y, 0), np.maximum(-Y, 0)], axis=1)
        Y2g = torch.tensor(Y2[fit_rows], dtype=torch.float32, device="cuda")
        Yfit = Yg[torch.tensor(fit_rows, device="cuda")]
        nfit = torch.linalg.norm(Yfit)
    cm_err, cm_par = cond_mean_error(Y, obs, cls)
    res = {"units_kept": np.flatnonzero(keep_u), "trial_index": d["trial_index"], "t_centres": d["t_centres"],
           "classes": cls.astype(str), "norm_Y": np.linalg.norm(Y), "shape": np.array(Y.shape),
           "condmean_err": cm_err, "condmean_params": cm_par, "tol": TOL, "max_iter": MAX_ITER,
           "engine": "tensorly-gpu-float32", "outlier_trials": outlier, "clip_sd": CLIP_SD, "outlier_z": OUTLIER_Z,
           "method": method, "l2": l2, "split": method == "splitnn", "bsub": method != "nn"}
    for r in RANKS:
        tr = time.time()
        rng = np.random.default_rng([seed, r, ALIGNMENTS.index(align), NORMS.index(norm)])
        fits = []
        n_singular = 0
        for _ in range(N_REP):
            rs = int(rng.integers(1 << 31))
            if method == "splitnn":
                cp, err = non_negative_parafac_hals(Y2g, r, n_iter_max=MAX_ITER, tol=TOL, init="random",
                                                    random_state=rs, return_errors=True)
                Yh = tl.cp_to_tensor(cp)
                e_obs = float(torch.linalg.norm(Yfit - (Yh[:, :J] - Yh[:, J:])) / nfit)
            elif method == "nn":   # plain non-negative TCA on rates without baseline subtraction (Williams 2018)
                if Mg is not None:
                    raise ValueError("nn (HALS) has no missing-data mask; masked entries present")
                for attempt in range(5):
                    try:
                        cp, err = non_negative_parafac_hals(Yg, r, n_iter_max=MAX_ITER, tol=TOL, init="random",
                                                            random_state=rs, return_errors=True)
                        break
                    except torch.linalg.LinAlgError:
                        n_singular += 1
                        rs = int(rng.integers(1 << 31))
                else:
                    raise RuntimeError(f"nn rank {r}: singular solve in 5 random starts")
                e_obs = float(torch.linalg.norm(Yg - tl.cp_to_tensor(cp)) / ny)
            elif method == "seminn":
                if Mg is not None:
                    raise ValueError("seminn (HALS) has no missing-data mask; masked entries present")
                for attempt in range(5):   # a component collapsing to zero makes the signed-mode solve singular:
                    try:                   # restart from a new random start (counted in r{r}_n_singular)
                        cp, err = non_negative_parafac_hals(Yg, r, n_iter_max=MAX_ITER, tol=TOL, init="random",
                                                            random_state=rs, nn_modes=[0, 2], return_errors=True)
                        break
                    except torch.linalg.LinAlgError:
                        n_singular += 1
                        rs = int(rng.integers(1 << 31))
                else:
                    raise RuntimeError(f"seminn rank {r}: singular solve in 5 random starts")
                e_obs = float(torch.linalg.norm(Yg - tl.cp_to_tensor(cp)) / ny)
            else:
                cp, err = parafac(Yg, r, n_iter_max=MAX_ITER, tol=TOL, init="random", mask=Mg, l2_reg=l2,
                                  random_state=rs, return_errors=True)
                e_obs = float(torch.linalg.norm((Yg - tl.cp_to_tensor(cp))[Og]) / ny)   # fitted entries only
            fits.append((e_obs, len(err), to_factors(cp)))
        torch.cuda.synchronize()
        fits.sort(key=lambda x: x[0])
        best = fits[0][2]
        # seminn / splitnn: trial and time weights are >= 0, no sign flipping (seminn: sign is in the unit weights)
        if method == "cp":
            for c in range(r):   # sign convention (006): unit and time factors positive sum, sign on the trial factor
                for m in (1, 2):
                    sg = np.sign(best[m][:, c].sum()) or 1.0
                    best[m][:, c] *= sg
                    best[0][:, c] *= sg
        res[f"r{r}_obj"] = np.array([f[0] for f in fits])
        res[f"r{r}_sim"] = np.array([tt_similarity(best, f[2]) for f in fits[1:]])
        res[f"r{r}_iters"] = np.array([f[1] for f in fits])
        res[f"r{r}_secs"] = time.time() - tr
        res[f"r{r}_mincong"] = congruence_min(best)
        res[f"r{r}_n_singular"] = n_singular
        A_ = best[0]
        if method == "splitnn":   # trial factor rows exist only for fitted trials: put them back, outliers by NNLS
            A_ = np.zeros((len(outlier), r))
            A_[fit_rows] = best[0]
            if outlier.any():
                KR = np.einsum("jr,kr->jkr", best[1], best[2]).reshape(-1, r)
                for i in np.flatnonzero(outlier):
                    A_[i] = nnls(KR, Y2[i].reshape(-1))[0]
        elif outlier.any():       # outlier trials: weights by least squares on the fitted unit x time factors
            KR = np.einsum("jr,kr->jkr", best[1], best[2]).reshape(-1, r)
            A_[outlier] = np.linalg.lstsq(KR, Y[outlier].reshape(outlier.sum(), -1).T, rcond=None)[0].T
        best[0] = A_
        for m, U in enumerate(best):
            res[f"r{r}_f{m}"] = U.astype(np.float32)
    err = np.array([res[f"r{r}_obj"].min() for r in RANKS])
    sim = np.array([res[f"r{r}_sim"].mean() if len(res[f"r{r}_sim"]) else 1.0 for r in RANKS])
    res["rank_chosen"], res["gain"] = choose_rank(err, sim)
    res["sim_min"], res["gain_min"] = SIM_MIN, GAIN_MIN
    rc = res["rank_chosen"]
    return res, dict(session_id=sid, alignment=align, norm=norm, method=method, l2=l2, ok=True, shape=str(Y.shape),
                     rank_chosen=rc, err_chosen=round(float(err[rc - 1]), 4), sim_chosen=round(float(sim[rc - 1]), 3),
                     mincong_r7=round(res.get("r7_mincong", np.nan), 3), err_r7=round(float(err[6]), 4),
                     condmean_err=round(float(cm_err), 4), secs=round(time.time() - t0, 1),
                     n_outlier_trials=int(outlier.sum()), n_trials=int(len(outlier)),
                     max_iters_hit=int(sum((res[f"r{r}_iters"] >= MAX_ITER).sum() for r in RANKS)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-root", default="out")
    ap.add_argument("--alignments", nargs="*", default=ALIGNMENTS)
    ap.add_argument("--norms", nargs="*", default=NORMS)
    ap.add_argument("--methods", nargs="*", default=["splitnn"],
                    help="splitnn (main, user 2026-11-22), cp (signed; --l2 ridge), seminn, nn (no baseline "
                         "subtraction)")
    ap.add_argument("--l2", nargs="*", type=float, default=[0.0], help="ridge strengths for method cp")
    a = ap.parse_args()
    import torch
    tl.set_backend("pytorch")
    root, out = Path(a.data_root), Path(a.out_root) / a.session
    out.mkdir(parents=True, exist_ok=True)
    variants = [(m, l2) for m in a.methods for l2 in (a.l2 if m == "cp" else [0.0])]
    log = []
    for al in a.alignments:
        for n in a.norms:
            for m, l2 in variants:
                try:
                    res, msg = fit_type(root, a.session, al, n, m, l2)
                    np.savez_compressed(out / f"{al}_{n}{variant_tag(m, l2)}.npz", **res)
                except Exception:   # noqa: BLE001
                    import traceback
                    msg = dict(session_id=a.session, alignment=al, norm=n, method=m, l2=l2, ok=False,
                               error=traceback.format_exc()[-800:])
                print(msg, flush=True)
                log.append(msg)
    tag = "_".join(f"{m}{variant_tag(m, l2)}" for m, l2 in variants)
    (out / f"fit_log_{tag}.json").write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(0),
                                                             torch=torch.__version__, tensorly=tl.__version__,
                                                             fits=log), indent=1, default=float))


if __name__ == "__main__":
    main()
