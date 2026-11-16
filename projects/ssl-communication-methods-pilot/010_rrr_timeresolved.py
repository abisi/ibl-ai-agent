"""Time-resolved reduced-rank regression (RRR) between all area groups (user request 2026-11-09: "rerun the RRR
communication pilot, use more area groups (e.g. hippocampus, midbrain), use information in Wu & Pillow 2025
(arXiv 2512.12467, RRR tutorial), show time-resolved communication, highlight R+ and R- differences").

Method (Wu & Pillow 2025; Semedo et al. 2019):
  ridge-RRR: W_ridge = (X'X + lambda I)^-1 X'Y, then projection on the top-r right singular vectors of X W_ridge
  (= eigenvectors of W'X'XW; RRR is not the truncated SVD of W unless the input covariance is spherical); lambda and the
  rank r chosen by cross-validation (folds split by trial), r = smallest rank within one s.e.m. of the peak.
  Centering per time bin (condition PSTH removed per unit and bin): the fit is on trial-to-trial co-fluctuations,
  so it does not just reflect shared stimulus or movement responses.
  Quantities per window: held-out R^2 of the RRR at the chosen rank (predictive performance), the chosen rank
  (communication dimensionality), and the communication fraction CF = Tr(W' S_x W) / Tr(S_y) of the chosen model refit
  on all trials (share of the target's variance that the source accounts for; in-sample, as in the tutorial).
  Null: the same with the source trials shuffled (N_SHUFFLES permutations); the session value is the excess over the
  null mean.
Area groups: area_group_coarse_v3 (ssl-cross-area-communication 000_coverage_lib), every group with >= N_UNITS good
units (>= 1 Hz); every ordered pair (source -> target). Same-bin RRR is not directional: A->B and B->A differ through the
populations, not through time order.
Trials: day-0 whisker sessions (not exclude_ephys, with a jaw-onset table), active, perf != 6, early_lick == 0, rule A1
trim (as 007); conditions W_lick, W_nolick, A_hit (>= N_TRIALS trials each). Per draw: N_TRIALS trials per condition and
N_UNITS units per group at random (seed crc32(session) + draw), so cohorts and pairs are matched in sample size.
Time: 20-ms bins, 200-ms windows, 50-ms steps, centres -0.4..+0.8 s from trial start.
Outputs: OUT/rrr_timeresolved/sessions/<session>.parquet + _meta.json (config, units per group, trial counts).
Run (haas, repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python
    projects/ssl-communication-methods-pilot/010_rrr_timeresolved.py [--n-proc 48] [--sessions S ...] [--quick]
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

_spec = importlib.util.spec_from_file_location("hyp", HERE / "007_run_hypothesis.py")
hyp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hyp)

OUT = cl.OUT / "rrr_timeresolved"
CONFIG = {
    "version": 1,
    "reference": "Wu & Pillow 2025, arXiv 2512.12467 (ridge-RRR, CV rank by one-s.e.m. rule, communication fraction)",
    "area_level": "area_group_coarse_v3",
    "conditions": ["W_lick", "W_nolick", "A_hit"],
    "n_units": 20, "n_trials": 20, "n_draws": 3, "n_shuffles": 10, "min_rate_hz": 1.0,
    "bin_s": 0.02, "edges_s": (-0.6, 1.0), "window_bins": 10, "step_s": 0.05, "centers_s": (-0.4, 0.8),
    "cv_folds": cl.CONFIG["cv_folds"], "ridge_alphas": cl.CONFIG["ridge_alphas"], "max_rank": cl.CONFIG["max_rank"],
    "residuals": "condition mean per unit and bin removed (centering per time bin)",
    "trials": "context != passive, perf != 6, early_lick == 0, rule A1 trim (007 load)",
}
for k in ("bin_s", "edges_s", "window_bins", "step_s", "centers_s"):  # 007 helpers read its CONFIG2
    hyp.CONFIG2[k] = CONFIG[k]


def comm_fraction(XT, YT, alpha, rank):
    """CF = Tr(W' S_x W) / Tr(S_y) of the ridge-RRR (alpha relative to the mean input variance, as rrr_cv) on all rows."""
    X, Y = XT.reshape(-1, XT.shape[2]), YT.reshape(-1, YT.shape[2])
    X, Y = X - X.mean(0), Y - Y.mean(0)
    scale = np.trace(X.T @ X) / X.shape[1]
    B = np.linalg.solve(X.T @ X + alpha * scale * np.eye(X.shape[1]), X.T @ Y)
    _, _, Vt = np.linalg.svd(X @ B, full_matrices=False)
    W = B @ Vt[:rank].T @ Vt[:rank]
    return float(((X @ W) ** 2).sum() / max((Y ** 2).sum(), 1e-12))


def run_session(args):
    sid, cohort, learner, quick = args
    t0 = time.time()
    C = CONFIG
    cov3 = cl._coverage_lib()
    units, trials, cond_idx = hyp.load(sid)
    starts = trials.start_time.to_numpy()
    X = hyp.counts([np.sort(np.asarray(s, float)) for s in units.spike_times], starts)
    rate = X.sum((0, 1)) / (len(starts) * (C["edges_s"][1] - C["edges_s"][0]))
    keep = rate >= C["min_rate_hz"]
    units, X = units[keep].reset_index(drop=True), X[:, :, keep]
    group = units.area_acronym_custom.map(cov3.COARSE_GROUP_V3_OF).to_numpy()
    members = {g: np.flatnonzero(group == g) for g in pd.unique(group[pd.notna(group)])}
    groups = sorted(g for g, m in members.items() if len(m) >= C["n_units"])
    pairs = [(a, b) for a in groups for b in groups if a != b]
    conds = [c for c in C["conditions"] if len(cond_idx[c]) >= C["n_trials"]]
    centers, widx = hyp.windows()
    n_draws, n_sh = (1, 2) if quick else (C["n_draws"], C["n_shuffles"])
    meta = dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=cohort, learner=learner,
                n_units={g: int(len(m)) for g, m in members.items()}, groups=groups,
                n_trials={c: int(len(v)) for c, v in cond_idx.items()}, conditions=conds, config=C)
    rows = []
    rng0 = zlib.crc32(sid.encode())
    for draw in range(n_draws if pairs else 0):
        rng = np.random.default_rng(rng0 + draw)
        for cond in conds:
            tr = rng.choice(cond_idx[cond], C["n_trials"], replace=False)
            sel = {g: rng.choice(members[g], C["n_units"], replace=False) for g in groups}
            R = {g: cl.residualise(X[tr][:, :, sel[g]], np.zeros(len(tr))) for g in groups}
            perms = [np.arange(len(tr))] + [rng.permutation(len(tr)) for _ in range(n_sh)]
            for a, b in pairs:
                for wc, idx in zip(centers, widx):
                    vals = []
                    for perm in perms:
                        xa, yb = R[a][perm][:, idx], R[b][:, idx]
                        r = cl.rrr_cv(xa, yb, rng)
                        vals.append((r["perf_rrr"], r["rank"], comm_fraction(xa, yb, r["alpha"], r["rank"]), r["alpha"]))
                    v = np.array(vals, float)
                    for j, q in enumerate(("r2", "rank", "cf")):
                        rows.append(dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=cohort, learner=learner,
                                         draw=draw, condition=cond, source=a, target=b, window_center=float(wc),
                                         quantity=q, value=float(v[0, j]), null_mean=float(v[1:, j].mean()),
                                         null_sd=float(v[1:, j].std()), alpha=float(v[0, 3]),
                                         p_upper=float((1 + np.sum(v[1:, j] >= v[0, j])) / len(v))))
        print(sid, "draw", draw, f"{time.time() - t0:.0f} s", flush=True)
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if quick else ""
    pd.DataFrame(rows).to_parquet(OUT / "sessions" / f"{sid}{sfx}.parquet")
    json.dump(meta, open(OUT / "sessions" / f"{sid}{sfx}_meta.json", "w"), indent=1, default=str)
    return sid, len(groups), len(rows), round(time.time() - t0)


def run_safe(job):
    try:
        return run_session(job)
    except Exception as err:  # noqa: BLE001 - logged, run continues
        import traceback
        traceback.print_exc()
        return job[0], f"ERROR {type(err).__name__}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-proc", type=int, default=48)
    ap.add_argument("--quick", action="store_true", help="1 draw, 2 shuffles")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    s = pd.read_csv(cl.OUT / "coverage" / "session_coverage.csv")
    s = s[s.has_jaw_table & (s.exclude_ephys != 1) & s.cohort.isin(["R+", "R-"])]
    if args.sessions:
        s = s[s.session_id.isin(args.sessions)]
    done = {p.stem for p in (OUT / "sessions").glob("*.parquet")} if not args.quick else set()
    jobs = [(r.session_id, r.cohort, bool(r.learner), args.quick) for r in s.itertuples() if r.session_id not in done]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(CONFIG, open(OUT / "config.json", "w"), indent=1, default=str)
    print(f"{len(jobs)} sessions to run ({len(done)} already done)", flush=True)
    with Pool(min(args.n_proc, max(1, len(jobs)))) as pool:
        for res in pool.imap_unordered(run_safe, jobs):
            print("done", res, flush=True)


if __name__ == "__main__":
    main()
