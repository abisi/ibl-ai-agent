"""021 -- model v5 (rl6_model, model_v5_design.md): cohort-level fits (separate R+ and R- parameter sets; v4 019: cohort > shared in every
variant), 5-fold cross-validation over mice, recovery including g > 0 (user 2026-10-23: "yes go").
Trials: 011 sequences (active, perf != 6, no trimming). Folds: same mouse split as 019.
Models: rl6_model.model_names() (transfer init/overlap x FA E0-E3 x engagement no/yes x sym/asym = 32).
Two passes as 019 (pass 2 restarts each model from the pass-1 optima of its nested models).
Generators for recovery (chosen from the real fits): best model; best with g set to 0.5 (tests recovery at g > 0); the other transfer
variant, the engagement toggle and the sym/asym toggle of the best model, each at their own fitted values.
Output: fits/021_fits.csv, tables/021_model_comparison.csv, tables/021_latents/<model>.parquet, fits/021_recovery_fits.csv,
tables/021_recovery_traj.csv, tables/021_generators.csv
Run (haas): OMP_NUM_THREADS=1 python 021_fit_v5.py --fit --recover --n-jobs 100
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd

import common as C
import rl6_model as M

F019 = importlib.import_module("019_fit_v4")
N_FOLD = 5
COHORTS = ["R+", "R-"]
LAT_DIR = C.TAB / "021_latents"
D = None


def subset(d, cs):
    return d[d.reward_group == cs].reset_index(drop=True)


def nested_start(sub, p_sub, name, extra):
    p = dict(p_sub)
    for k, v in extra.items():
        p[k] = p.get("eps0", 0.1) if v is None else v
    p.setdefault("alpha_neg", p["alpha_pos"])
    return [float(np.clip(p.get(k, M.DEFAULT[k]), *M.BOUNDS[k])) for k in M.free_params(name)]


def job(args):
    name, cs, fold, extra, data = args
    d = subset(D if data is None else data, cs)
    A = M.arrays(d)
    train = np.ones(len(d), np.int64) if fold == "full" else (d.fold != fold).to_numpy(np.int64)
    rng = np.random.default_rng(zlib.crc32(f"{name}{cs}{fold}".encode()))
    p, nll = M.fit(name, A, train, rng, n_random=6 if not extra else 0, extra_starts=extra)
    x = [p[k] for k in M.free_params(name)]
    ll_test = np.nan if fold == "full" else M.loglik(name, x, A, 1 - train)
    return dict(model=name, cohort_set=cs, fold=str(fold), k=len(x), n_train=int(train.sum()), n_test=int((1 - train).sum()),
                nll_train=nll, ll_test=ll_test, **{f"p_{k}": p[k] for k in M.P_ALL})


def run_jobs(jobs, n_jobs):
    with mp.get_context("fork").Pool(n_jobs) as pool:
        return pool.map(job, jobs, chunksize=1)


def two_pass(n_jobs, folds, models, data=None):
    R = pd.DataFrame(run_jobs([(m, cs, f, (), data) for m in models for cs in COHORTS for f in folds], n_jobs))
    key = ["model", "cohort_set", "fold"]
    first = R.set_index(key)
    jobs2 = []
    for (m, cs, f), _ in first.iterrows():
        starts = [nested_start(s, {k: first.loc[(s, cs, f), f"p_{k}"] for k in M.free_params(s)}, m, ex)
                  for s, ex in M.nested_of(m) if (s, cs, f) in first.index]
        if starts:
            jobs2.append((m, cs, f, tuple(starts), data))
    if jobs2:
        R2 = pd.DataFrame(run_jobs(jobs2, n_jobs)).set_index(key)
        better = R2.index[R2.nll_train < first.loc[R2.index, "nll_train"] - 1e-6]
        print(f"pass 2 improved {len(better)} of {len(R2)} fits", flush=True)
        first.loc[better] = R2.loc[better]
    return first.reset_index()


def params_of(F, name, cs, fold="full"):
    r = F[(F.model == name) & (F.cohort_set == cs) & (F.fold.astype(str) == str(fold))].iloc[0]
    return {k: r[f"p_{k}"] for k in M.P_ALL}


def run_latents(d, F, name, params=None):
    out = []
    for cs in COHORTS:
        sub = subset(d, cs)
        p = params[cs] if params else params_of(F, name, cs)
        L = pd.DataFrame(M.latents(name, p, M.arrays(sub)), columns=M.LATENTS)
        out.append(pd.concat([sub[["mouse_id", "session_id", "day", "trial_type", "lick_flag", "reward_group"]].reset_index(drop=True), L],
                             axis=1))
    return pd.concat(out, ignore_index=True)


def binned(L, nb=5):
    L = L.copy()
    pos = L.groupby("session_id").cumcount(); n = L.groupby("session_id").session_id.transform("size")
    L["x"] = L.day + np.floor(nb * pos / n) / nb
    return L.groupby(["reward_group", "x"])[["V_A", "V_W", "eps", "eng", "p_lick", "thirst"]].mean().reset_index()


def comparison(F, n):
    cv = F[F.fold != "full"].groupby("model").agg(ll_test=("ll_test", "sum"), n_test=("n_test", "sum")).reset_index()
    full = F[F.fold == "full"].groupby("model").agg(nll=("nll_train", "sum"), k=("k", "sum")).reset_index()
    S = cv.merge(full, how="right")
    S["ll_test_per_trial"] = S.ll_test / S.n_test
    S["bic"] = 2 * S.nll + S.k * np.log(n)
    return S


def fit_mode(a):
    F = two_pass(a.n_jobs, ["full"] + list(range(N_FOLD)), M.model_names())
    F.to_csv(C.FITS / "021_fits.csv", index=False)
    LAT_DIR.mkdir(exist_ok=True)
    for m in M.model_names():
        run_latents(D, F, m).to_parquet(LAT_DIR / f"{m}.parquet")
    S = comparison(F, len(D)).sort_values("ll_test_per_trial", ascending=False)
    S.to_csv(C.TAB / "021_model_comparison.csv", index=False)
    print(S.round(5).to_string(), flush=True)


def generators(F, S):
    best = S.sort_values("ll_test_per_trial", ascending=False).model.iloc[0]
    tr, fa, en, lr = best.split("-")
    gens = [(best, "fitted", {cs: params_of(F, best, cs) for cs in COHORTS})]
    g05 = {cs: dict(params_of(F, best, cs), g=0.5) for cs in COHORTS}
    gens.append((best, "g=0.5", g05))
    for alt in [f"{'overlap' if tr == 'init' else 'init'}-{fa}-{en}-{lr}", f"{tr}-{fa}-{'noeng' if en == 'eng' else 'eng'}-{lr}",
                f"{tr}-{fa}-{en}-{'sym' if lr == 'asym' else 'asym'}"]:
        gens.append((alt, "fitted", {cs: params_of(F, alt, cs) for cs in COHORTS}))
    return gens


def recover_mode(a):
    F = pd.read_csv(C.FITS / "021_fits.csv"); F["fold"] = F.fold.astype(str)
    S = pd.read_csv(C.TAB / "021_model_comparison.csv")
    gens = generators(F, S)
    pd.DataFrame([dict(generator=f"{m} ({lab})", model=m, label=lab, cohort_set=cs, **{f"p_{k}": v for k, v in p[cs].items()})
                  for m, lab, p in gens for cs in COHORTS]).to_csv(C.TAB / "021_generators.csv", index=False)
    rows, traj = [], []
    for gi, (gen, lab, P) in enumerate(gens):
        gname = f"{gen} ({lab})"
        rng = np.random.default_rng(2000 + gi)
        sim = D.copy()
        for cs in COHORTS:
            idx = D.index[D.reward_group == cs].to_numpy()
            lk, _ = M.simulate(gen, P[cs], M.arrays(D.loc[idx].reset_index(drop=True)), rng)
            sim.loc[idx, "lick_flag"] = lk
        traj.append(binned(run_latents(sim, None, gen, params=P)).assign(generator=gname, source="true"))
        R = two_pass(a.n_jobs, ["full"], M.model_names(), data=sim).assign(generator=gname)
        rows.append(R)
        Sr = comparison(R, len(sim))
        best = Sr.sort_values("bic").model.iloc[0]
        for m in {gen, best}:
            traj.append(binned(run_latents(sim, R, m)).assign(generator=gname, source=f"refit {m}"))
        print(f"generator {gname} done; BIC winner {best}", flush=True)
        pd.concat(rows).to_csv(C.FITS / "021_recovery_fits.csv", index=False)
        pd.concat(traj).to_csv(C.TAB / "021_recovery_traj.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=100)
    a = ap.parse_args()
    D = F019.load()
    if a.fit:
        fit_mode(a)
    if a.recover:
        recover_mode(a)
