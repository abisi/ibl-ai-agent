"""001b -- Refit the ORIGINAL learning-curve model (PyMC state-space model of
`beh_plotting_functions.plot_single_mouse_session_learning_curve`) for every
session in artifacts/001_inputs.pkl, whisker and no-stim trials, under two
sampler settings:
  orig : pm.sample(10, tune=10, chains=4)          (as in the stored curves)
  conv : pm.sample(1000, tune=1000, chains=4, target_accept=0.95)
Model (unchanged): tau ~ Gamma(10,10), sigma = 1/sqrt(tau),
x ~ GaussianRandomWalk(mu=0, sigma, init_dist=Normal(0,100) [PyMC default]),
p = invlogit(x), outcome ~ Bernoulli(p).

Run with the behaviour_analysis conda env (PyMC 5.23):
  /home/bisi/anaconda3/envs/behaviour_analysis/bin/python 001b_... [n_jobs]

Per (session, trial_type, setting): posterior mean/80% CI of p, sigma
posterior mean, max R-hat and min bulk ESS over p (arviz), # divergences;
500 thinned posterior samples of p (float32) kept for joint whisker-vs-FA
probabilities. FA curve interpolated at whisker trial times two ways:
  fa_orig : `learning_utils.interp_fa_curve` logic -- no-stim p_mean placed
            on an EVENLY SPACED time grid spanning the session, cubic spline
  fa_time : same p_mean placed at the ACTUAL no-stim trial times, linear
            interpolation (edges held constant)
Output: artifacts/001_fits.pkl (dict session_id -> results).
"""

from __future__ import annotations

import os
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ART = Path(__file__).resolve().parents[1] / "artifacts"
SETTINGS = {"orig": dict(draws=10, tune=10), "conv": dict(draws=1000, tune=1000, target_accept=0.95)}
N_KEEP = 500
CONF = 80


def _init():
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir=/tmp/pytensor_lt_{os.getpid()},cxx="
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def fit_curve(outcomes: np.ndarray, setting: str, seed: int) -> dict:
    import arviz as az
    import pymc as pm

    n = len(outcomes)
    with pm.Model():
        tau = pm.Gamma("tau", alpha=10, beta=10)
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))
        x = pm.GaussianRandomWalk("x", mu=0, sigma=sigma, init_dist=pm.Normal.dist(0, 100), shape=n)
        p = pm.Deterministic("p", pm.math.invlogit(x))
        pm.Bernoulli("obs", p=p, observed=outcomes)
        tr = pm.sample(chains=4, cores=1, random_seed=seed, progressbar=False, compute_convergence_checks=False,
                       **SETTINGS[setting])
    ps = tr.posterior["p"].values.reshape(-1, n)
    summ = az.summary(tr, var_names=["p"], kind="diagnostics")
    lo, hi = np.percentile(ps, [(100 - CONF) / 2, 100 - (100 - CONF) / 2], axis=0)
    keep = ps[np.linspace(0, len(ps) - 1, min(N_KEEP, len(ps))).astype(int)].astype(np.float32)
    return dict(p_mean=ps.mean(0), p_low=lo, p_high=hi, samples=keep, n_samples=len(ps),
                sigma_mean=float(tr.posterior["sigma"].values.mean()),
                rhat_max=float(np.nanmax(summ["r_hat"])), ess_bulk_min=float(np.nanmin(summ["ess_bulk"])),
                n_divergent=int(tr.sample_stats["diverging"].values.sum()))


def fa_interp(fa_p: np.ndarray, n_start: np.ndarray, w_start: np.ndarray, w_stop: np.ndarray) -> dict:
    from scipy.interpolate import CubicSpline

    grid = np.linspace(min(w_start.min(), n_start.min()), max(w_stop.max(), n_start.max()), len(fa_p))
    orig = CubicSpline(grid, fa_p, extrapolate=False)(w_start)
    t = np.interp(w_start, n_start, fa_p)
    return dict(fa_orig=orig, fa_time=t)


def process(item) -> tuple[str, dict]:
    sid, d = item
    seed = zlib.crc32(sid.encode()) % (2**31)
    res = {"meta": {k: d[k] for k in ("session_id", "mouse_id", "reward_group", "learning_category")}}
    t0 = time.time()
    for setting in SETTINGS:
        w = fit_curve(d["w_outcomes"], setting, seed)
        n = fit_curve(d["n_outcomes"], setting, seed + 1)
        fa = fa_interp(n["p_mean"], d["n_start"], d["w_start"], d["w_stop"])
        fa_samples_time = np.stack([np.interp(d["w_start"], d["n_start"], s) for s in n["samples"]])
        res[setting] = dict(whisker=w, nostim={k: v for k, v in n.items()}, **fa,
                            p_above_time=(w["samples"] > fa_samples_time).mean(0))
    res["fit_s"] = time.time() - t0
    return sid, res


def main():
    n_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    with open(ART / "001_inputs.pkl", "rb") as f:
        inputs = pickle.load(f)
    out_path = ART / "001_fits.pkl"
    out = pickle.load(open(out_path, "rb")) if out_path.exists() else {}
    todo = [(k, v) for k, v in inputs.items() if k not in out]
    print(f"{len(todo)} sessions to fit ({len(out)} done), {n_jobs} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init) as ex:
        futs = [ex.submit(process, it) for it in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            sid, res = fut.result()
            out[sid] = res
            with open(out_path, "wb") as f:
                pickle.dump(out, f)
            o, c = res["orig"]["whisker"], res["conv"]["whisker"]
            print(f"[{i}/{len(todo)}] {sid} orig rhat {o['rhat_max']:.2f} ess {o['ess_bulk_min']:.0f} | conv rhat "
                  f"{c['rhat_max']:.3f} ess {c['ess_bulk_min']:.0f} div {c['n_divergent']} sigma {c['sigma_mean']:.2f} "
                  f"({res['fit_s']:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
