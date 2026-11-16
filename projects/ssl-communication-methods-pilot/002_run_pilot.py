"""Inter-areal communication pilot: per session, time-resolved RRR, PCA->CCA and ensemble Granger between pairs of the
area groups motor/frontal, SSp-w and striatum, per trial condition, with trial-shuffle nulls and controls
(see comm_lib docstring and CONFIG).

Sessions: whisker day 0 (learning), learners (reference sheet good/moderate), 8 R+ and 8 R- mice, one session each,
chosen as those with the most good units in all three area groups (>= 40 each; ssl-cross-area-communication unit
metadata v3, 2026-11-05).
Per session and draw (n_draws, seeds from crc32(session) + draw): n_trials trials per condition (conditions with fewer
are skipped) and, per area, n_units_source units (+ n_units_target other units of the same area for the within-area
ceiling when available), units with mean rate >= min_rate_hz.
Outputs (combined_results_ks4/ssl-communication-methods-pilot/sessions/): <session>.parquet (one row per session x
draw x condition x pair x direction x method x metric x window, with true value, shuffle mean / sd and metadata),
<session>_meta.json (unit and trial counts, rates per area, config).
Run (haas): OMP_NUM_THREADS=1 python projects/ssl-communication-methods-pilot/002_run_pilot.py [--sessions S ...] [--n-proc 16] [--quick]
"""

import argparse
import itertools
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402

SESSIONS = {  # session_id: cohort (learners; most good units in all three area groups)
    "AB087_20231017_141901": "R+", "AB127_20240821_103757": "R+", "AB130_20240902_123634": "R+",
    "AB149_20241217_113458": "R+", "AB121_20240813_125401": "R+", "MH032_20250507_143741": "R+",
    "AB150_20241219_144646": "R+", "AB147_20241212_151234": "R+",
    "AB085_20231005_152636": "R-", "AB093_20231207_111207": "R-", "AB116_20240724_102941": "R-",
    "AB122_20240804_134554": "R-", "AB126_20240822_114405": "R-", "AB128_20240829_112813": "R-",
    "AB140_20241129_104154": "R-", "MH034_20250514_104756": "R-",
}
OUT = cl.OUT / "sessions"


def regress_out(XT, M):
    """Remove the linear contribution of movement covariates M [trial, bin, k] from XT [trial, bin, unit]."""
    X, Z = XT.reshape(-1, XT.shape[2]), M.reshape(-1, M.shape[2])
    ok = np.isfinite(Z).all(1)
    if ok.sum() < 20:
        return None
    Z = np.c_[np.ones(len(Z)), np.nan_to_num(Z - np.nanmean(Z, 0))]
    B = np.linalg.lstsq(Z[ok], X[ok], rcond=None)[0]
    return (X - Z @ B).reshape(XT.shape)


def run_session(args):
    sid, quick = args
    t0 = time.time()
    C = cl.CONFIG
    rng0 = zlib.crc32(sid.encode())
    units, trials = cl.load_session(sid)
    starts = trials.start_time.to_numpy()
    spikes = [np.sort(np.asarray(s, float)) for s in units.spike_times]
    fine_all = cl.fine_tensor(spikes, starts)  # [trial, bin, unit]
    dur = cl.bin_edges()[-1] - cl.bin_edges()[0]
    rate = fine_all.sum((0, 1)) / (len(starts) * dur)
    keep = rate >= C["min_rate_hz"]
    units, fine_all, rate = units[keep].reset_index(drop=True), fine_all[:, :, keep], rate[keep]
    by_area = {a: np.flatnonzero(units.area_group.to_numpy() == a) for a in C["areas"]}
    mvt = None
    edges = cl.bin_edges()
    mv = cl.movement_covariates(sid, starts, edges[:-1], edges[1:])  # [trial, 2, bin]
    if mv is not None:
        mvt = np.transpose(mv, (0, 2, 1))  # [trial, bin, 2]
    centers, widx = cl.window_grid()
    gbins = int(round(C["granger_window_s"] / C["bin_s"]))
    if quick:
        centers, widx = centers[::5], widx[::5]
    conds = [c for c in C["conditions"] if (trials.condition == c).sum() >= C["n_trials"]]
    pairs = [(a, b) for a, b in itertools.combinations(C["areas"], 2)
             if len(by_area[a]) >= C["n_units_source"] and len(by_area[b]) >= C["n_units_source"]]
    meta = {"session_id": sid, "mouse_id": sid.split("_")[0], "cohort": SESSIONS[sid],
            "n_units_by_area": {a: int(len(v)) for a, v in by_area.items()},
            "mean_rate_hz_by_area": {a: float(rate[v].mean()) if len(v) else None for a, v in by_area.items()},
            "n_trials_by_condition": trials.condition.value_counts().to_dict(), "conditions_used": conds,
            "pairs": pairs, "movement_covariates": mvt is not None, "config": C}
    rows = []
    base = dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=SESSIONS[sid])

    def add(**kw):
        rows.append({**base, **kw})

    n_draws = 1 if quick else C["n_draws"]
    n_sh = 3 if quick else C["n_shuffles_rrr_cca"]
    n_sh_g = 10 if quick else C["n_shuffles_granger"]
    for draw in range(n_draws):
        rng = np.random.default_rng(rng0 + draw)
        pick = {a: rng.permutation(v) for a, v in by_area.items()}
        src = {a: p[:C["n_units_source"]] for a, p in pick.items()}
        within = {a: p[C["n_units_source"]:C["n_units_source"] + C["n_units_target"]] for a, p in pick.items()}
        for cond in conds:
            tr = rng.choice(np.flatnonzero(trials.condition.to_numpy() == cond), C["n_trials"], replace=False)
            R = cl.residualise(fine_all[tr], np.zeros(len(tr)))  # one condition: subtract its mean per unit and bin
            Rm = None
            if mvt is not None:
                Rm = R  # movement-regressed per window below
            pcs = {a: cl.population_pcs(fine_all[tr][:, :, src[a]], np.zeros(len(tr)), C["granger_pcs"])
                   for a in C["areas"] if len(src[a]) >= C["n_units_source"]}
            for a, b in pairs:
                tag = dict(draw=draw, condition=cond, pair=f"{a} | {b}", area_a=a, area_b=b, n_trials=len(tr),
                           n_units_a=len(src[a]), n_units_b=len(src[b]))
                for wc, idx in zip(centers, widx):
                    XA, XB = R[:, idx][:, :, src[a]], R[:, idx][:, :, src[b]]
                    for d, (S, T, s_name, t_name) in enumerate([(XA, XB, a, b), (XB, XA, b, a)]):
                        res = cl.rrr_cv(S, T, rng)
                        null = [cl.rrr_cv(S[rng.permutation(len(S))], T, rng)["perf_rrr"] for _ in range(n_sh)]
                        add(**tag, window_center=wc, method="RRR", metric="perf_rrr", source=s_name, target=t_name,
                            value=res["perf_rrr"], null_mean=float(np.mean(null)), null_sd=float(np.std(null)),
                            rank=res["rank"], alpha=res["alpha"], perf_full=res["perf_full"])
                        wsrc = within[s_name]
                        if len(wsrc) >= C["n_units_target"]:
                            W = R[:, idx][:, :, wsrc]
                            rw = cl.rrr_cv(S, W, rng)
                            add(**tag, window_center=wc, method="RRR_within", metric="perf_rrr", source=s_name,
                                target=s_name, value=rw["perf_rrr"], rank=rw["rank"], alpha=rw["alpha"])
                        if mvt is not None:
                            M = mvt[tr][:, idx]
                            S2, T2 = regress_out(S, M), regress_out(T, M)
                            if S2 is not None and T2 is not None:
                                rm = cl.rrr_cv(S2, T2, rng)
                                add(**tag, window_center=wc, method="RRR_mvt_regressed", metric="perf_rrr",
                                    source=s_name, target=t_name, value=rm["perf_rrr"], rank=rm["rank"])
                    cc = cl.pca_cca_cv(XA, XB, rng)
                    null = [cl.pca_cca_cv(XA[rng.permutation(len(XA))], XB, rng)["cc1"] for _ in range(n_sh)]
                    add(**tag, window_center=wc, method="PCA_CCA", metric="cc1", source=a, target=b, value=cc["cc1"],
                        null_mean=float(np.mean(null)), null_sd=float(np.std(null)), cc_top3=cc["cc_top3"])
                    if mvt is not None:
                        M = mvt[tr][:, idx]
                        A2, B2 = regress_out(XA, M), regress_out(XB, M)
                        if A2 is not None and B2 is not None:
                            add(**tag, window_center=wc, method="PCA_CCA_mvt_regressed", metric="cc1", source=a,
                                target=b, value=cl.pca_cca_cv(A2, B2, rng)["cc1"])
                    # Granger: bins within +-granger_window/2 of the centre that have `order` past bins
                    lo = max(C["granger_order"], int(idx[0] + len(idx) / 2 - gbins / 2))
                    t_index = np.arange(lo, min(lo + gbins, R.shape[1]))
                    za, zb = pcs[a], pcs[b]
                    ab, ba = cl.granger_ensemble(za, zb, C["granger_order"], t_index)
                    nb = za.shape[1]
                    t_rev = np.sort(nb - 1 - t_index)
                    t_rev = t_rev[t_rev >= C["granger_order"]]
                    ab_r, ba_r = cl.granger_ensemble(za[:, ::-1], zb[:, ::-1], C["granger_order"], t_rev)
                    nulls = np.array([cl.granger_ensemble(za[rng.permutation(len(za))], zb, C["granger_order"], t_index)
                                      for _ in range(n_sh_g)])
                    for s_name, t_name, val, rev, nl in [(a, b, ab, ab_r, nulls[:, 0]), (b, a, ba, ba_r, nulls[:, 1])]:
                        add(**tag, window_center=wc, method="Granger", metric="gc", source=s_name, target=t_name,
                            value=val, null_mean=float(nl.mean()), null_sd=float(nl.std()), value_reversed=rev)
        print(sid, "draw", draw, f"{time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if quick else ""
    df.to_parquet(OUT / f"{sid}{sfx}.parquet")
    json.dump(meta, open(OUT / f"{sid}{sfx}_meta.json", "w"), indent=1, default=str)
    return sid, len(df), round(time.time() - t0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", nargs="*")
    parser.add_argument("--n-proc", type=int, default=16)
    parser.add_argument("--quick", action="store_true", help="1 draw, every 5th window, few shuffles (smoke test)")
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sessions = args.sessions or list(SESSIONS)
    jobs = [(s, args.quick) for s in sessions]
    with Pool(min(args.n_proc, len(jobs))) as pool:
        for res in pool.imap_unordered(run_session_safe, jobs):
            print("done", res, flush=True)


def run_session_safe(job):
    try:
        return run_session(job)
    except Exception as err:
        import traceback
        traceback.print_exc()
        return job[0], f"ERROR {type(err).__name__}: {err}"


if __name__ == "__main__":
    main()
