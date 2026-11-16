"""Full-dataset test of the routing hypothesis (user request 2026-11-06; pre-registered in question.md, section
"Hypothesis test v2"): in R+, a feedforward Go signal leaves SSp-w towards motor areas / striatum on whisker lick
trials; in R-, frontal areas send a top-down NoGo signal to SSp-w / striatum on whisker no-lick trials; auditory hits
are Go in both cohorts (within-session reference).

Per whisker day-0 session (ephys, not exclude_ephys), per pre-registered pair with >= N_UNITS good units (>= 1 Hz) in
both area sets, per condition with >= N_TRIALS trials, per draw (N_DRAWS; seeds crc32(session) + draw): N_UNITS units
per area and N_TRIALS trials, then in sliding windows (20-ms bins, 200-ms windows, 50-ms steps):
  lagRRR A->B and B->A (source lags 1..3 bins; direction index A->B minus B->A); lagRRR | past + rest (target's own
  past and the rest of the recording regressed out first; benchmark 005: removes recorded common input and most shared
  gain, at lower power); Granger A<->B on 3 PCs with time reversal (TRGC), conditional Granger | rest, pCCA | rest
  (partial_CCA), PCA->CCA, same-bin RRR, all against N_SHUFFLES trial shuffles of A (same permutations for every method).
  "rest" = top 5 PCs of every other good unit of the session (not in A or B), lags 0..3 bins.
Residuals: condition mean per unit and bin (the gain-corrected variant of benchmark 005 lowered power without removing
shared gain for the lagged methods, so it is not used).
Conditions: W_lick, W_nolick, A_hit, NS_cr (active, perf != 6, early_lick == 0, rule A1 trim, as 006) and the
pre-movement subsets W_lick_pm150 / A_hit_pm150 (jaw onset >= 150 ms; windows ending <= 150 ms only).
Outputs: OUT/hypothesis_v2/sessions/<session>.parquet (+ _meta.json)
Run (haas): OMP_NUM_THREADS=1 python projects/ssl-communication-methods-pilot/007_run_hypothesis.py [--n-proc 48]
            [--sessions S ...] [--quick]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402
from common import REFERENCE_SHEET  # noqa: E402

OUT = cl.OUT / "hypothesis_v2"
_spec = importlib.util.spec_from_file_location("cov", Path(__file__).resolve().parent / "006_coverage_table.py")
cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov)

CONFIG2 = {
    "version": 2,
    "area_sets": cov.AREA_SETS,
    # pre-registered pairs (A, B); direction tested both ways. VPM-PO -> SSp-w: anatomical positive control.
    "pairs": [("SSp-w", "wMO"), ("SSp-w", "ALM-tjM1"), ("SSp-w", "Striatum"), ("Frontal", "SSp-w"),
              ("Frontal", "Striatum"), ("wMO", "Striatum"), ("ALM-tjM1", "Striatum"), ("Frontal", "wMO"),
              ("Frontal", "ALM-tjM1"), ("VPM-PO", "SSp-w")],
    "conditions": ["W_lick", "W_nolick", "A_hit", "NS_cr", "W_lick_pm150", "A_hit_pm150"],
    "premovement_conditions": {"W_lick_pm150": 0.15, "A_hit_pm150": 0.15},  # latest window end (s)
    "n_units": 20, "n_trials": 15, "n_draws": 5, "n_shuffles": 20, "min_rate_hz": 1.0,
    "bin_s": 0.02, "edges_s": (-0.6, 0.8), "window_bins": 10, "step_s": 0.05, "centers_s": (-0.3, 0.5),
    "lags_bins": 3, "granger_pcs": 3, "granger_order": 3, "granger_bins": 8, "rest_pcs": 5,
    "residuals": "condition mean per unit and bin (comm_lib.residualise)",
    "trials": "context != passive (NaN in 23 day-0 NWBs), perf != 6, early_lick == 0, rule A1 trim; jaw onsets from behaviour_analysis pkl",
}


def edges():
    lo, hi = CONFIG2["edges_s"]
    return np.round(np.arange(lo, hi + 1e-9, CONFIG2["bin_s"]), 4)


def windows():
    e = edges()
    c0, c1 = CONFIG2["centers_s"]
    centers = np.round(np.arange(c0, c1 + 1e-9, CONFIG2["step_s"]), 4)
    w = CONFIG2["window_bins"]
    mids = (e[:-1] + e[1:]) / 2
    out = []
    for c in centers:
        i0 = int(np.argmin(np.abs(mids - (c - (w / 2 - 0.5) * CONFIG2["bin_s"]))))
        out.append(np.arange(i0, i0 + w))
    return centers, out


def load(session_id):
    from ephys_utilities.helpers import data_utils
    from ephys_utilities.neural_utils import unit_metrics_utils
    import allen_utils
    cov3 = cl._coverage_lib()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = data_utils.process_single_nwb(str(cl.NWB_DIR / f"{session_id}.nwb"), day_to_analyze="learning")
    units, trials = res["unit_table"], res["trial_table"]
    units = unit_metrics_utils.classify_units_quality(units, thresholds=cov3.THRESHOLDS, exclude=cov3.EXCLUDE,
                                                      label_col="quality_label")
    units = allen_utils.process_allen_labels(units, split_merge_areas=True)
    units = units[units.quality_label == "good"].reset_index(drop=True)
    t = trials[(trials.context != "passive") & (trials.perf != 6) & (trials.early_lick == 0)].sort_values("start_time")
    t = cov.trim_a1(t.reset_index(drop=True)).copy()
    jaw = pd.read_pickle(cl.ROOT / "combined_results_ks4" / session_id.split("_")[0] / "dlc_jaw_onset_times.pkl")
    jaw = jaw[jaw.session_id == session_id][["start_time", "jaw_dlc_onset"]]
    t = pd.merge_asof(t.sort_values("start_time"), jaw.sort_values("start_time"), on="start_time",
                      tolerance=0.005, direction="nearest")
    j = pd.to_numeric(t.jaw_dlc_onset, errors="coerce")
    w, a, n = (t.trial_type == f"{k}_trial" for k in ("whisker", "auditory", "no_stim"))
    lk = t.lick_flag == 1
    cond = {"W_lick": w & lk, "W_nolick": w & ~lk, "A_hit": a & lk, "NS_cr": n & ~lk,
            "W_lick_pm150": w & lk & (j >= 0.15), "A_hit_pm150": a & lk & (j >= 0.15)}
    return units, t, {k: np.flatnonzero(v.to_numpy()) for k, v in cond.items()}


def counts(spike_times, starts):
    e = edges()
    q = starts[:, None] + e[None, :]
    out = np.zeros((len(starts), len(e) - 1, len(spike_times)), dtype=np.float32)
    for u, st in enumerate(spike_times):
        out[:, :, u] = np.diff(np.searchsorted(st, q), axis=1)
    return out


def pcs(x, k):
    flat = x.reshape(-1, x.shape[2])
    flat = flat - flat.mean(0)
    _, _, Vt = np.linalg.svd(flat, full_matrices=False)
    z = flat @ Vt[:k].T
    return (z / (z.std(0) + 1e-12)).reshape(x.shape[0], x.shape[1], k)


def run_session(args):
    sid, cohort, learner, quick = args
    t0 = time.time()
    C = CONFIG2
    units, trials, cond_idx = load(sid)
    starts = trials.start_time.to_numpy()
    spikes = [np.sort(np.asarray(s, float)) for s in units.spike_times]
    X = counts(spikes, starts)
    rate = X.sum((0, 1)) / (len(starts) * (edges()[-1] - edges()[0]))
    keep = rate >= C["min_rate_hz"]
    units, X = units[keep].reset_index(drop=True), X[:, :, keep]
    acr = units.area_acronym_custom.to_numpy()
    members = {a: np.flatnonzero(np.isin(acr, s)) for a, s in C["area_sets"].items()}
    pairs = [(a, b) for a, b in C["pairs"] if len(members[a]) >= C["n_units"] and len(members[b]) >= C["n_units"]]
    conds = [c for c in C["conditions"] if len(cond_idx[c]) >= C["n_trials"]]
    centers, widx = windows()
    L, order, gb = C["lags_bins"], C["granger_order"], C["granger_bins"]
    n_draws, n_sh = (1, 3) if quick else (C["n_draws"], C["n_shuffles"])
    rows = []
    meta = dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=cohort, learner=learner,
                n_units=dict((a, int(len(v))) for a, v in members.items()), n_trials={c: int(len(v)) for c, v in
                cond_idx.items()}, pairs=pairs, conditions=conds, config=C)
    rng0 = zlib.crc32(sid.encode())
    for draw in range(n_draws):
        rng = np.random.default_rng(rng0 + draw)
        for cond in conds:
            tr = rng.choice(cond_idx[cond], C["n_trials"], replace=False)
            Xc = X[tr]
            zero = np.zeros(len(tr))
            for a, b in pairs:
                ua = rng.choice(members[a], C["n_units"], replace=False)
                ub = rng.choice(members[b], C["n_units"], replace=False)
                rest = np.setdiff1d(np.arange(X.shape[2]), np.r_[members[a], members[b]])
                A, B = cl.residualise(Xc[:, :, ua], zero), cl.residualise(Xc[:, :, ub], zero)
                R = cl.residualise(Xc[:, :, rest], zero) if len(rest) >= C["rest_pcs"] else None
                za, zb = pcs(A, C["granger_pcs"]), pcs(B, C["granger_pcs"])
                zr = pcs(R, C["rest_pcs"]) if R is not None else None
                perms = [np.arange(len(tr))] + [rng.permutation(len(tr)) for _ in range(n_sh)]
                for wc, idx in zip(centers, widx):
                    if cond in C["premovement_conditions"] and wc + C["window_bins"] * C["bin_s"] / 2 > \
                            C["premovement_conditions"][cond] + 1e-9:
                        continue
                    Z = cl.lag_stack(zr, idx, range(L + 1)) if zr is not None else None
                    lo = max(order, int(idx[0] + len(idx) / 2 - gb / 2))
                    t_index = np.arange(lo, lo + gb)
                    t_rev = np.sort(za.shape[1] - 1 - t_index)
                    vals = []
                    for perm in perms:
                        Ap, zap = A[perm], za[perm]
                        o = {}
                        o["RRR"] = cl.rrr_cv(Ap[:, idx], B[:, idx], rng)["perf_rrr"]
                        o["lagRRR_ab"] = cl.rrr_cv(cl.lag_stack(Ap, idx, range(1, L + 1)), B[:, idx], rng)["perf_rrr"]
                        o["lagRRR_ba"] = cl.rrr_cv(cl.lag_stack(B, idx, range(1, L + 1)), Ap[:, idx], rng)["perf_rrr"]
                        o["lagRRR_dir"] = o["lagRRR_ab"] - o["lagRRR_ba"]
                        if Z is not None:
                            o["lagRRR|rest_ab"] = cl.lagged_rrr_conditional(Ap, B, idx, L, rng, Z)["perf_rrr"]
                            o["lagRRR|rest_ba"] = cl.lagged_rrr_conditional(B, Ap, idx, L, rng, Z)["perf_rrr"]
                            o["lagRRR|rest_dir"] = o["lagRRR|rest_ab"] - o["lagRRR|rest_ba"]
                        o["CCA"] = cl.pca_cca_cv(Ap[:, idx], B[:, idx], rng)["cc1"]
                        if Z is not None:
                            o["pCCA|rest"] = cl.pcca_cv(Ap[:, idx], B[:, idx], Z, rng)["cc1"]
                        ab, ba = cl.granger_ensemble(zap, zb, order, t_index)
                        abr, bar = cl.granger_ensemble(zap[:, ::-1], zb[:, ::-1], order, t_rev)
                        o["Granger_ab"], o["Granger_ba"], o["Granger_dir"] = ab, ba, (ab - ba) - (abr - bar)
                        if zr is not None:
                            cab, cba = cl.granger_conditional(zap, zb, zr, order, t_index)
                            cabr, cbar = cl.granger_conditional(zap[:, ::-1], zb[:, ::-1], zr[:, ::-1], order, t_rev)
                            o["Granger|rest_ab"], o["Granger|rest_ba"] = cab, cba
                            o["Granger|rest_dir"] = (cab - cba) - (cabr - cbar)
                        vals.append(o)
                    for q in vals[0]:
                        v, nl = vals[0][q], np.array([x[q] for x in vals[1:]], float)
                        rows.append(dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=cohort, learner=learner,
                                         draw=draw, condition=cond, area_a=a, area_b=b, n_rest=int(len(rest)),
                                         window_center=float(wc), quantity=q, value=float(v),
                                         null_mean=float(np.nanmean(nl)), null_sd=float(np.nanstd(nl)),
                                         p_upper=float((1 + np.sum(nl >= v)) / (1 + len(nl))),
                                         p_lower=float((1 + np.sum(nl <= v)) / (1 + len(nl)))))
        print(sid, "draw", draw, f"{time.time() - t0:.0f} s", flush=True)
    OUT.joinpath("sessions").mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if quick else ""
    pd.DataFrame(rows).to_parquet(OUT / "sessions" / f"{sid}{sfx}.parquet")
    json.dump(meta, open(OUT / "sessions" / f"{sid}{sfx}_meta.json", "w"), indent=1, default=str)
    return sid, len(rows), round(time.time() - t0)


def run_safe(job):
    try:
        return run_session(job)
    except Exception as err:
        import traceback
        traceback.print_exc()
        return job[0], f"ERROR {type(err).__name__}: {err}"


def session_list():
    """Day-0 sessions with units in at least one pre-registered pair (006 coverage), jaw table, not exclude_ephys."""
    s = pd.read_csv(cl.OUT / "coverage" / "session_coverage.csv")
    s = s[s.has_jaw_table & (s.exclude_ephys != 1) & s.cohort.isin(["R+", "R-"])]
    ok = np.zeros(len(s), bool)
    for a, b in CONFIG2["pairs"]:
        ok |= (s[f"units_{a}"] >= CONFIG2["n_units"]).to_numpy() & (s[f"units_{b}"] >= CONFIG2["n_units"]).to_numpy()
    return s[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-proc", type=int, default=48)
    ap.add_argument("--quick", action="store_true", help="1 draw, 3 shuffles")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    s = session_list()
    if args.sessions:
        s = s[s.session_id.isin(args.sessions)]
    done = {p.stem for p in (OUT / "sessions").glob("*.parquet")} if not args.quick else set()
    jobs = [(r.session_id, r.cohort, bool(r.learner), args.quick) for r in s.itertuples() if r.session_id not in done]
    print(f"{len(jobs)} sessions to run ({len(done)} already done)", flush=True)
    with Pool(min(args.n_proc, max(1, len(jobs)))) as pool:
        for res in pool.imap_unordered(run_safe, jobs):
            print("done", res, flush=True)


if __name__ == "__main__":
    main()
