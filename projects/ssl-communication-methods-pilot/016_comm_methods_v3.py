"""Communication between area groups, v3: several methods on the same draws, including methods that take the rest of
the recording out (user request 2026-11-09: "update the large scale, comparing several methods, including methods that
allow to partialize other areas out. Also run an analysis where you just have somatosensory cortex, auditory, midbrain,
thalamus, hippocampus, frontal cortex, striatum").

Data, trials, units, binning, draws and null as 012 (v2): whisker day 0, stable good + mua units, conditions W_lick,
W_nolick, A_hit (>= 5 trials, up to 20 per draw), 50-ms bins, 300-ms windows every 100 ms (centres -0.4..0.8 s),
condition PSTH removed per unit and bin, 15 units per group, 5 draws, 10 trial shuffles of the source, within-area
ceiling for groups with >= 30 units.
Area sets (--areas):
  all    area_group_coarse_v3 (as 012);
  seven  Somatosensory cortex (SSp-w + SSp-orofacial), Auditory areas, Midbrain, Thalamus, Hippocampus, Motor and
         frontal areas (frontal cortex), Striatum and pallidum; units of other groups (PPC, retrosplenial, pons and
         medulla, unassigned) are not analysed as groups but are part of the "rest".
Methods per ordered pair (source -> target), window and permutation (source trials permuted for the null):
  r2    ridge-RRR held-out R^2 at the CV rank (comm_lib.rrr_cv; Wu & Pillow 2025), with rank and cf (communication
        fraction) as in 010;
  cca   first held-out canonical correlation, PCA (10) -> CCA (comm_lib.pca_cca_cv);
  pcca  the same, partial CCA conditioned on Z (comm_lib.pcca_cv, vendored partial_CCA);
  prrr  partial RRR: source and target residualised on Z (least squares with intercept over the window's rows), then
        ridge-RRR held-out R^2 (rrr_cv);
  dr2   unique contribution of the source: held-out R^2 of ridge [source, Z] -> target minus that of ridge Z -> target
        (same trial folds; ridge penalty chosen for the full model by CV over comm_lib ridge_alphas).
  Z = the top 5 PCs of every other unit of the session outside the source and target groups (same bins, residualised
  like the populations; PCs from all trials of the draw); Z stays aligned to the target trials in the null.
Preprocessing (--prep; user 2026-11-10: "do the normalization; also check baseline-subtracted data (trial-by-trial)"):
  raw    spike counts minus the condition mean per unit and bin (as 012);
  z      the same residuals divided per unit by (SD + soft constant), SD over the draw's trials and all bins (soft
         normalisation, Churchland et al. 2012: every unit weighs about equally, near-silent units are not inflated;
         soft constant = Poisson SD of a 2-Hz unit in one 50-ms bin); Z is built from the normalised residuals;
  zbase  each trial's own baseline (mean count per bin, -1.1..-0.6 s before the stimulus, ahead of every analysis
         window) subtracted per unit before the condition mean, then as z: trial-to-trial deviations from each trial's
         baseline, so slow state shared across areas is removed.
Outputs: OUT/comm_v3_<areas>[_<prep>]/sessions/<session>.parquet (+ _meta.json); same columns as 012.
Run (haas, repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python
    projects/ssl-communication-methods-pilot/016_comm_methods_v3.py --areas all|seven [--prep raw|z|zbase] [--n-proc 56] [--quick --sessions S]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comm_lib as cl  # noqa: E402

_spec = importlib.util.spec_from_file_location("rrr_v2", HERE / "012_rrr_timeresolved_v2.py")
v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v2)
hyp, WITHIN = v2.hyp, v2.WITHIN

SEVEN = {"SSp-w": "Somatosensory cortex", "SSp-orofacial": "Somatosensory cortex", "Auditory areas": "Auditory areas",
         "Midbrain": "Midbrain", "Thalamus": "Thalamus", "Hippocampus": "Hippocampus",
         "Motor and frontal areas": "Motor and frontal areas", "Striatum and pallidum": "Striatum and pallidum"}
CONFIG = {**v2.CONFIG, "version": 3,
          "methods": ["r2 / rank / cf (ridge-RRR)", "cca (PCA 10 -> CCA cc1)", "pcca (partial CCA | Z, cc1)",
                      "prrr (RRR on residuals of Z)", "dr2 (R2[source + Z] - R2[Z], ridge)"],
          "Z": "top 5 PCs of all units outside the source and target groups",
          "soft_sd": float(np.sqrt(2.0 * v2.CONFIG["bin_s"])), "baseline_s": (-1.1, -0.6)}
PREPS = ("raw", "z", "zbase")
QUANT = ("r2", "rank", "cf", "cca", "pcca", "prrr", "dr2")


def out_dir(areas, prep="raw"):
    return cl.OUT / (f"comm_v3_{areas}" if prep == "raw" else f"comm_v3_{areas}_{prep}")


def residualise_on(XT, ZT):
    """[trial, bin, unit] residuals of XT on ZT (+ intercept), least squares over the rows."""
    X, Z = XT.reshape(-1, XT.shape[2]), ZT.reshape(-1, ZT.shape[2])
    A = np.column_stack([np.ones(len(Z)), Z])
    return (X - A @ np.linalg.lstsq(A, X, rcond=None)[0]).reshape(XT.shape)


def unique_r2(XT, YT, ZT, rng):
    """Held-out R^2 (pooled over target units, folds by trial) of ridge [X, Z] -> Y minus ridge Z -> Y; the ridge alpha
    (relative to the mean input variance) chosen for the full model by CV."""
    n, k = len(XT), cl.CONFIG["cv_folds"]
    folds = cl._folds(n, k, rng)
    full = np.concatenate([XT, ZT], axis=2)

    def cv(IT, alpha):
        sse = sst = 0.0
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            Itr, Ytr, Ite, Yte = cl._rows(IT, train), cl._rows(YT, train), cl._rows(IT, test), cl._rows(YT, test)
            mi, my = Itr.mean(0), Ytr.mean(0)
            Itr, Ytr, Ite, Yte = Itr - mi, Ytr - my, Ite - mi, Yte - my
            scale = np.trace(Itr.T @ Itr) / Itr.shape[1]
            B = cl._ridge(Itr, Ytr, alpha * scale)
            sse += ((Yte - Ite @ B) ** 2).sum()
            sst += (Yte ** 2).sum()
        return 1 - sse / max(sst, 1e-12)

    perf = {a: cv(full, a) for a in cl.CONFIG["ridge_alphas"]}
    alpha = max(perf, key=perf.get)
    return perf[alpha] - cv(ZT, alpha)


def run_session(args):
    sid, cohort, learner, quick, areas, prep = args
    t0 = time.time()
    C = CONFIG
    cov3 = cl._coverage_lib()
    units, trials, cond_idx, unit_rule = v2.load(sid)
    starts = trials.start_time.to_numpy()
    X = hyp.counts([np.sort(np.asarray(s, float)) for s in units.spike_times], starts)
    rate = X.sum((0, 1)) / (len(starts) * (C["edges_s"][1] - C["edges_s"][0]))
    keep = rate >= C["min_rate_hz"]
    units, X = units[keep].reset_index(drop=True), X[:, :, keep]
    if prep == "zbase":  # mean count per bin in each trial's own baseline window, per unit
        b0, b1 = C["baseline_s"]
        base = np.stack([np.diff(np.searchsorted(np.sort(np.asarray(st, float)),
                                                 np.stack([starts + b0, starts + b1], 1)), axis=1)[:, 0]
                         for st in units.spike_times], 1) * C["bin_s"] / (b1 - b0)
        X = X - base[:, None, :].astype(X.dtype)
    group = units.area_acronym_custom.map(cov3.COARSE_GROUP_V3_OF)
    if areas == "seven":
        group = group.map(SEVEN)
    group = group.to_numpy()
    rest_label = np.where(pd.isna(group), "__none__", group).astype(object)  # unassigned units are always rest
    members = {g: np.flatnonzero(group == g) for g in pd.unique(group[pd.notna(group)])}
    groups = sorted(g for g, m in members.items() if len(m) >= C["n_units"])
    ceil_groups = [g for g in groups if len(members[g]) >= C["ceiling_min_units"]]
    pairs = [(a, b) for a in groups for b in groups if a != b] + [(b + WITHIN, b) for b in ceil_groups]
    conds = [c for c in C["conditions"] if len(cond_idx[c]) >= C["min_trials"]]
    centers, widx = hyp.windows()
    n_draws, n_sh = (1, 2) if quick else (C["n_draws"], C["n_shuffles"])
    meta = dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=cohort, learner=learner, areas=areas,
                unit_rule=unit_rule, prep=prep, n_units={g: int(len(m)) for g, m in members.items()}, groups=groups,
                ceiling_groups=ceil_groups, n_trials={c: int(len(v)) for c, v in cond_idx.items()}, conditions=conds,
                config=C)
    rows = []
    rng0 = zlib.crc32(sid.encode())
    for draw in range(n_draws if len(groups) > 1 else 0):
        rng = np.random.default_rng(rng0 + draw)
        for cond in conds:
            n_tr = min(len(cond_idx[cond]), C["n_trials"])
            tr = rng.choice(cond_idx[cond], n_tr, replace=False)
            cl.CONFIG["cv_folds"] = min(C["cv_folds"], n_tr)
            sel = {}
            for g in groups:
                pick = rng.permutation(members[g])
                sel[g] = pick[:C["n_units"]]
                if g in ceil_groups:
                    sel[g + WITHIN] = pick[C["n_units"]:2 * C["n_units"]]
            R_all = cl.residualise(X[tr], np.zeros(len(tr)))
            if prep != "raw":
                R_all /= R_all.std((0, 1), keepdims=True) + C["soft_sd"]
            R = {g: R_all[:, :, u] for g, u in sel.items()}
            rest_cache = {}
            perms = [np.arange(len(tr))] + [rng.permutation(len(tr)) for _ in range(n_sh)]
            for a, b in pairs:
                Z = v2.rest_pcs(R_all, rest_label, {a.replace(WITHIN, ""), b}, rest_cache, C["rest_pcs"])
                for wc, idx in zip(centers, widx):
                    vals = []
                    for perm in perms:
                        xa, yb = R[a][perm][:, idx], R[b][:, idx]
                        r = cl.rrr_cv(xa, yb, rng)
                        cf = v2.v1.comm_fraction(xa, yb, r["alpha"], r["rank"])
                        cca = cl.pca_cca_cv(xa, yb, rng)["cc1"]
                        if Z is None:
                            pcca = prrr = dr2 = np.nan
                        else:
                            zw = Z[:, idx]
                            pcca = cl.pcca_cv(xa, yb, zw, rng)["cc1"]
                            prrr = cl.rrr_cv(residualise_on(xa, zw), residualise_on(yb, zw), rng)["perf_rrr"]
                            dr2 = unique_r2(xa, yb, zw, rng)
                        vals.append((r["perf_rrr"], r["rank"], cf, cca, pcca, prrr, dr2, r["alpha"]))
                    v = np.array(vals, float)
                    for j, q in enumerate(QUANT):
                        rows.append(dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=cohort, learner=learner,
                                         draw=draw, condition=cond, n_trials_used=n_tr, source=a, target=b,
                                         window_center=float(wc), quantity=q, value=float(v[0, j]),
                                         null_mean=float(np.nanmean(v[1:, j])), null_sd=float(np.nanstd(v[1:, j])),
                                         alpha=float(v[0, -1]),
                                         p_upper=float((1 + np.sum(v[1:, j] >= v[0, j])) / len(v))))
        print(sid, "draw", draw, f"{time.time() - t0:.0f} s", flush=True)
    od = out_dir(areas, prep) / "sessions"
    od.mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if quick else ""
    pd.DataFrame(rows).to_parquet(od / f"{sid}{sfx}.parquet")
    json.dump(meta, open(od / f"{sid}{sfx}_meta.json", "w"), indent=1, default=str)
    return sid, len(groups), len(ceil_groups), len(rows), round(time.time() - t0)


def run_safe(job):
    try:
        return run_session(job)
    except Exception as err:  # noqa: BLE001 - logged, run continues
        import traceback
        traceback.print_exc()
        return job[0], f"ERROR {type(err).__name__}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", choices=["all", "seven"], required=True)
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-proc", type=int, default=56)
    ap.add_argument("--prep", choices=PREPS, default="raw")
    ap.add_argument("--quick", action="store_true", help="1 draw, 2 shuffles")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    s = pd.read_csv(cl.OUT / "coverage" / "session_coverage.csv")
    s = s[(s.exclude_ephys != 1) & s.cohort.isin(["R+", "R-"])]
    if args.sessions:
        s = s[s.session_id.isin(args.sessions)]
    od = out_dir(args.areas, args.prep)
    done = {p.stem for p in (od / "sessions").glob("*.parquet")} if not args.quick else set()
    jobs = [(r.session_id, r.cohort, bool(r.learner), args.quick, args.areas, args.prep) for r in s.itertuples()
            if r.session_id not in done]
    od.mkdir(parents=True, exist_ok=True)
    json.dump({**CONFIG, "areas": args.areas, "prep": args.prep}, open(od / "config.json", "w"), indent=1, default=str)
    print(f"{len(jobs)} sessions to run ({len(done)} already done)", flush=True)
    with Pool(min(args.n_proc, max(1, len(jobs)))) as pool:
        for res in pool.imap_unordered(run_safe, jobs):
            print("done", res, flush=True)


if __name__ == "__main__":
    main()
