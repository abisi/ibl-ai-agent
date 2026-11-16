"""Time-resolved RRR between area groups, v2 (user request 2026-11-09: "does communication rise after stimulus the same
in both cohorts for all trial types and tested areas? Use larger bins, normalise by a within-area ceiling, and test
within-session contrasts"). Same method as 010 (ridge-RRR, CV rank by the one-s.e.m. rule, communication fraction,
trial-shuffle null, condition PSTH removed per unit and bin; Wu & Pillow 2025), with:
  - 50-ms bins (Wu & Pillow recommend 50-100 ms for RRR on spike counts), 300-ms windows (6 bins), 100-ms steps,
    centres -0.4..+0.8 s;
  - 15 units per group (was 20), so that groups with >= 30 units also get a within-area ceiling: the same 15 target
    units predicted from 15 other units of the target's own group (disjoint), same trials and windows. Cross-area
    values are later divided by this ceiling (013), so cohort differences in which sub-areas, depths or probes were
    recorded scale out of the comparison;
  - 5 draws (was 3);
  - units: good AND mua that pass the stability criteria of 137_stable_units (coverage >= 0.9, presence >= 0.5, no
    probe-drift dependence; ssl-whisker-hitmiss-timeresolved-decoding tables; user 2026-11-09); sessions absent from
    that table (no drift file) use all good / mua units (as ssl-prelick-convergence 127);
  - trials: a condition enters with >= 5 trials (user 2026-11-09; was 20); each draw takes min(available, 20) trials,
    recorded in n_trials_used; CV folds = min(10, trials) (leave-one-trial-out below 10 trials).
  - methods compared on the same draws (user 2026-11-09: "can you still compare the two methods? also with CCA"):
    besides RRR (r2, rank, cf), held-out first canonical correlation of PCA (10 PCs) -> CCA ("cca", comm_lib
    pca_cca_cv) and of partial CCA conditioned on the rest of the recording ("pcca", comm_lib pcca_cv with the vendored
    partial_CCA): Z = top 5 PCs of every other unit of the session outside the source and target groups (same bins,
    residualised like the populations; PCs from all trials of the draw, unsupervised). Same trial-shuffle null (source
    trials permuted, Z kept with the target).
Each group's 15-unit set is the same whether the group is source or target in a draw.
Outputs: OUT/rrr_timeresolved_v2/sessions/<session>.parquet (+ _meta.json); rows with source == "<target> (within)" are
the ceiling.
Run (haas, repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python
    projects/ssl-communication-methods-pilot/012_rrr_timeresolved_v2.py [--n-proc 48] [--sessions S ...] [--quick]
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


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v1 = _load("rrr_v1", "010_rrr_timeresolved.py")
hyp = v1.hyp
OUT = cl.OUT / "rrr_timeresolved_v2"
CONFIG = {**v1.CONFIG, "version": 2,
          "n_units": 15, "ceiling_min_units": 30, "min_trials": 5, "n_trials": 20,
          "unit_set": "stable good + mua (137_stable_units; all good / mua where the session is absent from it)", "n_draws": 5, "n_shuffles": 10,
          "bin_s": 0.05, "edges_s": (-0.6, 1.0), "window_bins": 6, "step_s": 0.1, "centers_s": (-0.4, 0.8),
          "ceiling": "same target units predicted from n_units other units of the target group (disjoint)",
          "methods": ["rrr", "cca (PCA 10 -> CCA, cc1)", "pcca (partial CCA | rest top 5 PCs, cc1)"], "rest_pcs": 5}
for k in ("bin_s", "edges_s", "window_bins", "step_s", "centers_s"):  # 007 helpers read its CONFIG2
    hyp.CONFIG2[k] = CONFIG[k]
WITHIN = " (within)"
STABLE = cl.ROOT / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet"


def load(session_id):
    """007 load with the unit set of v2: stable good + mua units (137_stable_units joined on session_id,
    electrode_group, NWB cluster id); all good / mua units where the session is absent from that table."""
    import warnings
    from ephys_utilities.helpers import data_utils
    from ephys_utilities.neural_utils import unit_metrics_utils
    import allen_utils
    cov3 = cl._coverage_lib()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = data_utils.process_single_nwb(str(cl.NWB_DIR / f"{session_id}.nwb"), day_to_analyze="learning")
    units = unit_metrics_utils.classify_units_quality(res["unit_table"], thresholds=cov3.THRESHOLDS, exclude=cov3.EXCLUDE,
                                                      label_col="quality_label")
    units = allen_utils.process_allen_labels(units, split_merge_areas=True)
    units = units[units.quality_label.isin(["good", "mua"])].reset_index(drop=True)
    S = pd.read_parquet(STABLE, columns=["session_id", "electrode_group", "nwb_cluster_id", "stable"])
    S = S[S.session_id == session_id]
    rule = "all good/mua (no drift file)"
    if len(S):
        S = S.assign(cluster_id=S.nwb_cluster_id.astype(int).astype(str), electrode_group=S.electrode_group.astype(str))
        key = units.assign(cluster_id=units.cluster_id.astype(str), electrode_group=units.electrode_group.astype(str))
        key = key[["electrode_group", "cluster_id"]].merge(
            S.drop_duplicates(["electrode_group", "cluster_id"])[["electrode_group", "cluster_id", "stable"]],
            on=["electrode_group", "cluster_id"], how="left")
        units = units[key.stable.fillna(False).astype(bool).to_numpy()].reset_index(drop=True)
        rule = "stable good/mua"
    trials = res["trial_table"]  # trial selection and conditions as 007 load
    t = trials[(trials.context != "passive") & (trials.perf != 6) & (trials.early_lick == 0)].sort_values("start_time")
    t = hyp.cov.trim_a1(t.reset_index(drop=True)).copy()
    w, a = (t.trial_type == f"{k}_trial" for k in ("whisker", "auditory"))
    lk = t.lick_flag == 1
    cond = {"W_lick": w & lk, "W_nolick": w & ~lk, "A_hit": a & lk}
    return units, t, {k: np.flatnonzero(v.to_numpy()) for k, v in cond.items()}, rule


def rest_pcs(R_all, group, exclude, cache, k):
    """[trial, bin, k] top-k PCs of every unit outside the excluded groups (cached per excluded set)."""
    key = frozenset(exclude)
    if key not in cache:
        cols = np.flatnonzero(~pd.Series(group).isin(list(exclude)).to_numpy())
        if len(cols) < k:
            cache[key] = None
        else:
            flat = R_all[:, :, cols].reshape(-1, len(cols))
            flat = flat - flat.mean(0)
            _, _, Vt = np.linalg.svd(flat, full_matrices=False)
            cache[key] = (flat @ Vt[:k].T).reshape(R_all.shape[0], R_all.shape[1], k)
    return cache[key]


def run_session(args):
    sid, cohort, learner, quick = args
    t0 = time.time()
    C = CONFIG
    cov3 = cl._coverage_lib()
    units, trials, cond_idx, unit_rule = load(sid)
    starts = trials.start_time.to_numpy()
    X = hyp.counts([np.sort(np.asarray(s, float)) for s in units.spike_times], starts)
    rate = X.sum((0, 1)) / (len(starts) * (C["edges_s"][1] - C["edges_s"][0]))
    keep = rate >= C["min_rate_hz"]
    units, X = units[keep].reset_index(drop=True), X[:, :, keep]
    group = units.area_acronym_custom.map(cov3.COARSE_GROUP_V3_OF).to_numpy()
    members = {g: np.flatnonzero(group == g) for g in pd.unique(group[pd.notna(group)])}
    groups = sorted(g for g, m in members.items() if len(m) >= C["n_units"])
    ceil_groups = [g for g in groups if len(members[g]) >= C["ceiling_min_units"]]
    pairs = [(a, b) for a in groups for b in groups if a != b] + [(b + WITHIN, b) for b in ceil_groups]
    conds = [c for c in C["conditions"] if len(cond_idx[c]) >= C["min_trials"]]
    centers, widx = hyp.windows()
    n_draws, n_sh = (1, 2) if quick else (C["n_draws"], C["n_shuffles"])
    meta = dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=cohort, learner=learner,
                unit_rule=unit_rule, n_units={g: int(len(m)) for g, m in members.items()}, groups=groups, ceiling_groups=ceil_groups,
                n_trials={c: int(len(v)) for c, v in cond_idx.items()}, conditions=conds, config=C)
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
            R = {g: cl.residualise(X[tr][:, :, u], np.zeros(len(tr))) for g, u in sel.items()}
            R_all = cl.residualise(X[tr], np.zeros(len(tr)))
            rest_cache = {}
            perms = [np.arange(len(tr))] + [rng.permutation(len(tr)) for _ in range(n_sh)]
            for a, b in pairs:
                Z = rest_pcs(R_all, group, {a.replace(WITHIN, ""), b}, rest_cache, C["rest_pcs"])
                for wc, idx in zip(centers, widx):
                    vals = []
                    for perm in perms:
                        xa, yb = R[a][perm][:, idx], R[b][:, idx]
                        r = cl.rrr_cv(xa, yb, rng)
                        cca = cl.pca_cca_cv(xa, yb, rng)["cc1"]
                        pcca = cl.pcca_cv(xa, yb, None if Z is None else Z[:, idx], rng)["cc1"]
                        vals.append((r["perf_rrr"], r["rank"], v1.comm_fraction(xa, yb, r["alpha"], r["rank"]), cca, pcca,
                                     r["alpha"]))
                    v = np.array(vals, float)
                    for j, q in enumerate(("r2", "rank", "cf", "cca", "pcca")):
                        rows.append(dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=cohort, learner=learner,
                                         draw=draw, condition=cond, n_trials_used=n_tr, source=a, target=b, window_center=float(wc),
                                         quantity=q, value=float(v[0, j]), null_mean=float(v[1:, j].mean()),
                                         null_sd=float(v[1:, j].std()), alpha=float(v[0, 5]),
                                         p_upper=float((1 + np.sum(v[1:, j] >= v[0, j])) / len(v))))
        print(sid, "draw", draw, f"{time.time() - t0:.0f} s", flush=True)
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if quick else ""
    pd.DataFrame(rows).to_parquet(OUT / "sessions" / f"{sid}{sfx}.parquet")
    json.dump(meta, open(OUT / "sessions" / f"{sid}{sfx}_meta.json", "w"), indent=1, default=str)
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
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-proc", type=int, default=48)
    ap.add_argument("--quick", action="store_true", help="1 draw, 2 shuffles")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    s = pd.read_csv(cl.OUT / "coverage" / "session_coverage.csv")
    s = s[(s.exclude_ephys != 1) & s.cohort.isin(["R+", "R-"])]  # no jaw table needed (v2 has no pre-movement conditions)
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
