"""001c -- Validate the exact grid posterior (lt_lib, 'orig' prior) against the
ORIGINAL PyMC model sampled to convergence, and show what the stored
10-tune/10-draw setting gives, on a few sessions (whisker trials).
Run with the behaviour_analysis conda env (PyMC 5.23), e.g.:
  /home/bisi/anaconda3/envs/behaviour_analysis/bin/python 001c_validate_grid_vs_pymc.py
Output: artifacts/001c_validation.csv, exploratory-analyses/001c_validation.png
"""

from __future__ import annotations

import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
N_SESS = 4
SETTINGS = {"orig_10x10": dict(draws=10, tune=10), "conv_1000x1000": dict(draws=1000, tune=1000, target_accept=0.95)}


def _init():
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir=/tmp/pytensor_lt_{os.getpid()}"
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def fit(args):
    sid, y, setting = args
    import arviz as az
    import pymc as pm

    n = len(y)
    with pm.Model():
        tau = pm.Gamma("tau", alpha=10, beta=10)
        sigma = pm.Deterministic("sigma", 1 / pm.math.sqrt(tau))
        x = pm.GaussianRandomWalk("x", mu=0, sigma=sigma, init_dist=pm.Normal.dist(0, 100), shape=n)
        pm.Deterministic("p", pm.math.invlogit(x))
        pm.Bernoulli("obs", p=pm.math.invlogit(x), observed=y)
        tr = pm.sample(chains=4, cores=1, random_seed=zlib.crc32(sid.encode()) % 2**31, progressbar=False,
                       compute_convergence_checks=False, **SETTINGS[setting])
    ps = tr.posterior["p"].values.reshape(-1, n)
    s = az.summary(tr, var_names=["p"], kind="diagnostics")
    lo, hi = np.percentile(ps, [10, 90], axis=0)
    return sid, setting, dict(p_mean=ps.mean(0), p_low=lo, p_high=hi, rhat_max=float(s.r_hat.max()),
                              ess_min=float(s.ess_bulk.min()), n_div=int(tr.sample_stats.diverging.values.sum()))


def main():
    import pandas as pd

    sys.path.insert(0, str(HERE))
    import lt_lib as L
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    keys = sorted(inputs, key=lambda k: len(inputs[k]["w_outcomes"]))
    sel = [keys[int(q * (len(keys) - 1))] for q in np.linspace(0.1, 0.9, N_SESS)]
    jobs = [(k, inputs[k]["w_outcomes"], s) for k in sel for s in SETTINGS]
    with ProcessPoolExecutor(max_workers=len(jobs), initializer=_init) as ex:
        res = list(ex.map(fit, jobs))
    fig, axes = plt.subplots(len(sel), 1, figsize=(11, 2.6 * len(sel)), constrained_layout=True)
    rows = []
    for ax, k in zip(axes, sel):
        y = inputs[k]["w_outcomes"]
        g = L.fit_curve(y, "orig")
        glo, ghi = L.quantiles(g["marg"], [0.1, 0.9])
        x = np.arange(len(y))
        ax.fill_between(x, glo, ghi, color="k", alpha=0.15, lw=0, label="exact grid 80% CI")
        ax.plot(x, g["p_mean"], color="k", lw=1.5, label="exact grid mean")
        for sid, setting, r in res:
            if sid != k:
                continue
            col = "#d62728" if setting.startswith("orig") else "#1f77b4"
            ax.plot(x, r["p_mean"], color=col, lw=1, label=f"PyMC {setting} (R-hat {r['rhat_max']:.2f}, ESS {r['ess_min']:.0f})")
            ax.plot(x, r["p_low"], color=col, lw=0.6, ls=":")
            ax.plot(x, r["p_high"], color=col, lw=0.6, ls=":")
            rows.append(dict(session_id=k, setting=setting, rhat_max=r["rhat_max"], ess_min=r["ess_min"], n_div=r["n_div"],
                             mae_mean_vs_grid=np.abs(r["p_mean"] - g["p_mean"]).mean(),
                             mae_ci_vs_grid=(np.abs(r["p_low"] - glo).mean() + np.abs(r["p_high"] - ghi).mean()) / 2))
        ax.scatter(x, np.where(y == 1, 1.05, -0.05), s=3, color="k", marker="|")
        ax.set_title(k, fontsize=9)
        ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="upper right")
    fig.savefig(HERE / "001c_validation.png", dpi=150)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "001c_validation.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
