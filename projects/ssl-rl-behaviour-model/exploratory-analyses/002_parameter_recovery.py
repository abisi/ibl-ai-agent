"""002 -- Parameter recovery of the joint fit (no data). For each model, N_TRUTH parameter sets are drawn around the default values (each
parameter log-uniform within +-RANGE decades of its default; b, a, v, tr from the previous group-specific values), R+ and R- targets are
simulated with one common pretraining, and the joint fit is rerun from the default starting point (the average of the previous
group-specific values, default initial weights). Recovery = true vs fitted parameter, and loss at the fitted parameters vs at truth;
also checks that the simulated pretraining is identical across groups.
Output: tables/002_recovery_<model>.csv (one row per truth and parameter)
Run (haas): python 002_parameter_recovery.py --model shared [--n-truth 12] [--maxiter 1500] [--n-jobs 24]
"""
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

import common as C
import rl_joint as J

RANGE = 0.3


def one(args):
    name, k, maxiter = args
    mod, Model = J.load_model(name)
    layout = J.Layout(J.DYN_KEYS + J.weight_keys(mod))
    rng = np.random.default_rng(k)
    base = {**{q: getattr(mod.Params(), q) for q in J.weight_keys(mod)},
            **{q: float(np.mean([s[q] for s in J.START_DYN[name]])) for q in J.DYN_KEYS}}
    truth = {q: v * 10 ** rng.uniform(-RANGE, RANGE) for q, v in base.items()}
    sims = J.simulate_joint(mod, Model, truth, {})
    same_pre = bool(np.allclose(sims['R+'][0]['auditory'], sims['R-'][0]['auditory']))
    targets = {g: {'auditory': np.r_[sims[g][0]['auditory'], sims[g][1]['auditory']],
                   'no_stim': np.r_[sims[g][0]['nostim'], sims[g][1]['nostim']],
                   'whisker': np.r_[np.full(mod.N_PRE, np.nan), sims[g][1]['whisker']]} for g in J.GROUPS}
    r = J.fit(mod, Model, layout, targets, [layout.x0(base)], maxiter=maxiter)
    fitted, _ = layout.split(r.x)
    return [dict(model=name, truth_id=k, param=q, true=truth[q], fitted=fitted[q], start=base[q], loss=r.fun, n_eval=r.nfev,
                 same_pretraining=same_pre) for q in layout.common]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(J.MODELS), default="shared")
    ap.add_argument("--n-truth", type=int, default=12)
    ap.add_argument("--maxiter", type=int, default=1500)
    ap.add_argument("--n-jobs", type=int, default=24)
    a = ap.parse_args()
    with mp.get_context("spawn").Pool(a.n_jobs) as pool:
        rows = pool.map(one, [(a.model, k, a.maxiter) for k in range(a.n_truth)])
    df = pd.DataFrame([x for r in rows for x in r])
    df.to_csv(C.TAB / f"002_recovery_{a.model}.csv", index=False)
    d = df[df.param.isin(J.DYN_KEYS)]
    print(d.groupby("param").apply(lambda g: np.corrcoef(np.log10(g.true), np.log10(g.fitted))[0, 1]).round(2))
    print("pretraining identical across groups:", df.same_pretraining.all())
