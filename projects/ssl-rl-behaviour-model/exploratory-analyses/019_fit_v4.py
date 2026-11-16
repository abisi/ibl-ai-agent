"""019 -- model v4 (model_v4_design.md): cohort-level fits of the whole training process, 5-fold cross-validation over mice, and
recovery (user 2026-10-23: "Yes test this").
Trials: 011 sequences (active, perf != 6, no trimming; pretraining + whisker days, each mouse ordered by day and start time).
Models: rl5_model.model_names() (5 FA variants x sym/asym x 1ts/2ts = 20), each fitted with two groupings:
  shared  one parameter set for all mice (R+ and R- differ only through the whisker outcome)
  cohort  separate parameter sets for R+ and R- mice
Fits: pooled likelihood over the mice of a group; fold 'full' = all mice, folds 0-4 = mice split 5-fold within cohort (held-out mice
scored). Two passes: pass 2 restarts every model from the pass-1 optima of its nested models (keeps the better optimum).
Modes:
  --fit       fits/019_fits.csv (model x grouping x cohort_set x fold: params, nll_train, ll_test), tables/019_latents/<model>_<grouping>.parquet
  --recover   simulate from generators (cohort-grouping full fits), refit all 40 model x grouping (full only):
              fits/019_recovery_fits.csv, tables/019_recovery_traj.csv (binned true vs refit trajectories)
Run (haas): OMP_NUM_THREADS=1 python 019_fit_v4.py --fit --n-jobs 100
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd

import common as C
import rl5_model as M

N_FOLD = 5
GROUPINGS = {"shared": ["all"], "cohort": ["R+", "R-"]}
LAT_DIR = C.TAB / "019_latents"
GENERATORS = ["E1-asym-1ts", "E1-sym-2ts", "E2-asym-1ts", "E3-asym-1ts", "B-asym-1ts", "E0-sym-1ts"]
D = None          # sequence table (set before forking)


def load():
    d = pd.read_parquet(C.TAB / "011_sequences.parquet").reset_index(drop=True)
    mice = d.drop_duplicates("mouse_id")[["mouse_id", "reward_group"]]
    fold = {}
    for c, g in mice.groupby("reward_group"):
        ids = sorted(g.mouse_id)
        rng = np.random.default_rng(zlib.crc32(c.encode()))
        for i, m in enumerate(rng.permutation(ids)):
            fold[m] = i % N_FOLD
    d["fold"] = d.mouse_id.map(fold)
    return d


def subset(d, cohort_set):
    return d if cohort_set == "all" else d[d.reward_group == cohort_set].reset_index(drop=True)


def nested_start(sub, p_sub, name):
    """parameter dict of a nested model -> start vector for `name` that reproduces it"""
    p = dict(p_sub)
    p.setdefault("alpha_neg", p["alpha_pos"])
    if "eps_inf" in M.free_params(name) and "eps_inf" not in M.free_params(sub):
        p["eps_inf"] = p.get("eps0", 0.1)
    if name.startswith("E2") and sub.startswith("E0"):
        p["alpha_eps"] = 1e-4
    if name.startswith("B") and sub.startswith("E0"):
        p["kappa"] = 0.0
    if name.startswith("E1") and sub.startswith("E0"):
        p["tau"] = 3000.0
    return [float(np.clip(p.get(k, M.DEFAULT[k]), *M.BOUNDS[k])) for k in M.free_params(name)]


def nested_of(name):
    fa, lr, ts = name.split("-")
    out = []
    if lr == "asym":
        out.append(f"{fa}-sym-{ts}")
    if ts == "2ts":
        out.append(f"{fa}-{lr}-1ts")
    if fa in ("E1", "E2", "B"):
        out.append(f"E0-{lr}-{ts}")
    return out


def job(args):
    name, grouping, cohort_set, fold, extra, data = args
    d = subset(D if data is None else data, cohort_set)
    A = M.arrays(d)
    train = np.ones(len(d), np.int64) if fold == "full" else (d.fold != fold).to_numpy(np.int64)
    rng = np.random.default_rng(zlib.crc32(f"{name}{grouping}{cohort_set}{fold}".encode()))
    p, nll = M.fit(name, A, train, rng, n_random=6 if not extra else 0, extra_starts=extra)
    x = [p[k] for k in M.free_params(name)]
    ll_test = np.nan if fold == "full" else M.loglik(name, x, A, 1 - train)
    return dict(model=name, grouping=grouping, cohort_set=cohort_set, fold=fold, k=len(x), n_train=int(train.sum()),
                n_test=int((1 - train).sum()), nll_train=nll, ll_test=ll_test, **{f"p_{k}": p[k] for k in M.P_ALL})


def run_jobs(jobs, n_jobs):
    with mp.get_context("fork").Pool(n_jobs) as pool:
        return pool.map(job, jobs, chunksize=1)


def two_pass(n_jobs, folds, data=None):
    jobs = [(m, g, cs, f, (), data) for m in M.model_names() for g, css in GROUPINGS.items() for cs in css for f in folds]
    R = pd.DataFrame(run_jobs(jobs, n_jobs))
    key = ["model", "grouping", "cohort_set", "fold"]
    first = R.set_index(key)
    jobs2 = []
    for (m, g, cs, f), r in first.iterrows():
        starts = [nested_start(s, {k: first.loc[(s, g, cs, f), f"p_{k}"] for k in M.free_params(s)}, m) for s in nested_of(m)]
        if starts:
            jobs2.append((m, g, cs, f, tuple(starts), data))
    if jobs2:
        R2 = pd.DataFrame(run_jobs(jobs2, n_jobs)).set_index(key)
        better = R2.index[R2.nll_train < first.loc[R2.index, "nll_train"] - 1e-6]
        print(f"pass 2 improved {len(better)} of {len(R2)} fits")
        first.loc[better] = R2.loc[better]
    return first.reset_index()


def params_of(F, name, grouping, cohort_set, fold="full"):
    r = F[(F.model == name) & (F.grouping == grouping) & (F.cohort_set == cohort_set) & (F.fold.astype(str) == str(fold))].iloc[0]
    return {k: r[f"p_{k}"] for k in M.P_ALL}


def run_latents(d, F, name, grouping):
    parts = []
    for cs in GROUPINGS[grouping]:
        sub = subset(d, cs)
        A = M.arrays(sub)
        out = M.latents(name, params_of(F, name, grouping, cs), A)
        g = params_of(F, name, grouping, cs)["gamma"]
        L = pd.DataFrame(out, columns=M.LATENTS)
        L["V_A"] = g * L.w_S + (1 - g) * L.w_A
        L["V_W"] = g * L.w_S + (1 - g) * L.w_W
        parts.append(pd.concat([sub[["mouse_id", "session_id", "day", "trial_type", "lick_flag", "reward_group"]].reset_index(drop=True), L],
                               axis=1))
    return pd.concat(parts, ignore_index=True)


def binned(L, nb=5):
    """cohort-mean trajectories: x = day + within-session bin / nb"""
    L = L.copy()
    pos = L.groupby("session_id").cumcount(); n = L.groupby("session_id").session_id.transform("size")
    L["x"] = L.day + np.floor(nb * pos / n) / nb
    return L.groupby(["reward_group", "x"])[["V_A", "V_W", "eps", "p_lick", "thirst"]].mean().reset_index()


def fit_mode(a):
    folds = ["full"] + list(range(N_FOLD))
    F = two_pass(a.n_jobs, folds)
    F.to_csv(C.FITS / "019_fits.csv", index=False)
    LAT_DIR.mkdir(exist_ok=True)
    for m in M.model_names():
        for g in GROUPINGS:
            run_latents(D, F, m, g).to_parquet(LAT_DIR / f"{m}_{g}.parquet")
    cv = F[F.fold != "full"].groupby(["model", "grouping"]).agg(ll_test=("ll_test", "sum"), n_test=("n_test", "sum")).reset_index()
    full = F[F.fold == "full"].groupby(["model", "grouping"]).agg(nll=("nll_train", "sum"), k=("k", "sum")).reset_index()
    S = cv.merge(full)
    S["ll_test_per_trial"] = S.ll_test / S.n_test
    S["bic"] = 2 * S.nll + S.k * np.log(len(D))
    S.sort_values("ll_test_per_trial", ascending=False).to_csv(C.TAB / "019_model_comparison.csv", index=False)
    print(S.sort_values("ll_test_per_trial", ascending=False).round(5).to_string())


def recover_mode(a):
    F = pd.read_csv(C.FITS / "019_fits.csv")
    F["fold"] = F.fold.astype(str)
    rows, traj = [], []
    for gi, gen in enumerate(GENERATORS):
        rng = np.random.default_rng(1000 + gi)
        sim = D.copy()
        for cs in ["R+", "R-"]:
            idx = D.index[D.reward_group == cs].to_numpy()
            lk, _ = M.simulate(gen, params_of(F, gen, "cohort", cs), M.arrays(D.loc[idx].reset_index(drop=True)), rng)
            sim.loc[idx, "lick_flag"] = lk
        T_true = run_latents(sim, F, gen, "cohort")                 # generator latents on its own simulated licks
        traj.append(binned(T_true).assign(generator=gen, source="true"))
        R = two_pass(a.n_jobs, ["full"], data=sim)
        R["generator"] = gen
        rows.append(R)
        Fr = R.assign(fold=R.fold.astype(str))
        B = R.groupby(["model", "grouping"]).agg(nll=("nll_train", "sum"), k=("k", "sum"))
        best = (2 * B.nll + B.k * np.log(len(sim))).idxmin()[0]
        for m in {gen, best}:
            for g in GROUPINGS:
                traj.append(binned(run_latents(sim, Fr, m, g)).assign(generator=gen, source=f"refit {m} {g}"))
        print(f"generator {gen} done")
        pd.concat(rows).to_csv(C.FITS / "019_recovery_fits.csv", index=False)
        pd.concat(traj).to_csv(C.TAB / "019_recovery_traj.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=100)
    a = ap.parse_args()
    D = load()
    if a.fit:
        fit_mode(a)
    if a.recover:
        recover_mode(a)
