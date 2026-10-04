"""011b -- Check the grid joint continuous model (lt_joint.fit_continuous) against PyMC on a few
of the 011 sessions. Same model: g, d independent Gaussian random walks over all whisker +
no-stim trials in time order, logit P(no-stim lick) = g, logit P(whisker lick) = g + d,
g_1 ~ N(logit 0.2, 1.5), d_1 ~ N(0, 1.5), log sigma_g, log sigma_d ~ Uniform(log 0.02, log 1)
(continuous version of the grid's log-spaced sigma values). Non-centred parametrisation,
1000 tune / 1000 draws x 4 chains, target_accept 0.95.
Run with the behaviour_analysis conda env:
  /home/bisi/anaconda3/envs/behaviour_analysis/bin/python 011b_joint_pymc_check.py
Outputs: artifacts/011b_pymc_check.csv, exploratory-analyses/011b_joint_pymc_check.png
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
CHECK = ["AB107", "MH028", "AB082"]


def _init():
    os.environ["PYTENSOR_FLAGS"] = f"base_compiledir=/tmp/pytensor_joint_{os.getpid()}"
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def fit(args):
    sid, y, is_w = args
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    N = len(y)
    with pm.Model():
        lsg = pm.Uniform("log_sigma_g", np.log(0.02), np.log(1.0))
        lsd = pm.Uniform("log_sigma_d", np.log(0.02), np.log(1.0))
        g0 = pm.Normal("g0", np.log(0.2 / 0.8), 1.5)
        d0 = pm.Normal("d0", 0.0, 1.5)
        zg = pm.Normal("zg", 0, 1, shape=N - 1)
        zd = pm.Normal("zd", 0, 1, shape=N - 1)
        g = pm.Deterministic("g", pt.concatenate([pt.stack([g0]), g0 + pt.cumsum(pt.exp(lsg) * zg)]))
        d = pm.Deterministic("d", pt.concatenate([pt.stack([d0]), d0 + pt.cumsum(pt.exp(lsd) * zd)]))
        pm.Bernoulli("obs", logit_p=g + d * is_w.astype(float), observed=y)
        idata = pm.sample(1000, tune=1000, chains=4, cores=1, target_accept=0.95, progressbar=False,
                          random_seed=zlib.crc32(sid.encode()) % 2**31)
    s = az.summary(idata, var_names=["g", "d", "log_sigma_g", "log_sigma_d"], kind="diagnostics")
    return sid, dict(g=idata.posterior["g"].values.reshape(-1, N), d=idata.posterior["d"].values.reshape(-1, N),
                     rhat_max=float(s.r_hat.max()), ess_min=float(s.ess_bulk.min()),
                     n_div=int(idata.sample_stats.diverging.values.sum()),
                     sigma_g=float(np.exp(idata.posterior["log_sigma_g"]).mean()),
                     sigma_d=float(np.exp(idata.posterior["log_sigma_d"]).mean()))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    sys.path.insert(0, str(HERE))
    import lt_joint as J

    fits = pickle.load(open(ART / "011_joint_fits.pkl", "rb"))
    sel = [next(k for k in fits if k.startswith(p)) for p in CHECK]
    with ProcessPoolExecutor(max_workers=len(sel), initializer=_init) as ex:
        res = dict(ex.map(fit, [(s, fits[s]["y"], fits[s]["is_w"]) for s in sel]))
    rows = []
    fig, axes = plt.subplots(len(sel), 2, figsize=(14, 3.6 * len(sel)), constrained_layout=True, squeeze=False)
    for r, sid in enumerate(sel):
        f, m = fits[sid], res[sid]
        is_w = f["is_w"]
        x = np.cumsum(is_w) - 1.0
        x[~is_w] += 0.5
        for c, (name, marg, grid, samp) in enumerate((("g", f["cont"]["marg_g"], J.G_GRID, m["g"]),
                                                      ("d", f["cont"]["marg_d"], J.D_GRID, m["d"]))):
            ax = axes[r, c]
            (q10, q50, q90), mean = J.marg_summary(marg, grid)
            ax.fill_between(x, q10, q90, color="k", alpha=0.15, lw=0, label="grid 80%")
            ax.plot(x, mean, color="k", lw=1.5, label="grid mean")
            ax.plot(x, samp.mean(0), color="#1f77b4", lw=1.1, label="PyMC mean")
            lo, hi = np.percentile(samp, [10, 90], axis=0)
            ax.plot(x, lo, color="#1f77b4", lw=0.6, ls="--")
            ax.plot(x, hi, color="#1f77b4", lw=0.6, ls="--", label="PyMC 80%")
            mae = np.abs(mean - samp.mean(0)).mean()
            ax.set_title(f"{sid[:14]} -- {name}: |grid - PyMC| mean = {mae:.3f} logit; PyMC R-hat {m['rhat_max']:.3f}, "
                         f"ESS {m['ess_min']:.0f}, divergences {m['n_div']}", fontsize=8.5, loc="left")
            ax.set_ylabel(f"{name} (logit)")
            ax.legend(fontsize=7, frameon=False)
            rows.append(dict(session_id=sid, variable=name, mae_mean=mae, rhat_max=m["rhat_max"], ess_min=m["ess_min"],
                             n_div=m["n_div"], sigma_g_pymc=m["sigma_g"], sigma_d_pymc=m["sigma_d"],
                             sigma_g_grid=f["cont"]["sigma_g_mean"], sigma_d_grid=f["cont"]["sigma_d_mean"]))
    for a in axes[-1]:
        a.set_xlabel("whisker trial index")
    fig.suptitle("Joint continuous model: exact grid vs PyMC (same model; PyMC sigma prior continuous log-uniform)", fontsize=11)
    fig.savefig(HERE / "011b_joint_pymc_check.png", dpi=150)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "011b_pymc_check.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
