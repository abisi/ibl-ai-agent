"""076 -- Per-session geometry extras for the supplements (user review 2026-10-07), from the 051 per-event rates.

Same sessions, events, units and normalisation as 057 (whole brain, variant all; good + mua units with mean raw pre-lick
rate >= 0.1 Hz; >= 5 units; >= 4 events per class; unit scaling NORM, default the current pooled SD, see 074).
Per session:
  geometry      d(WH,ref), d(WH,AH), d(AH,ref), delta d, along = [d(WH,ref) + d(AH,ref) - d(WH,AH)] / 2, lambda (057 lam)
  linear shift  the same quantities with the event labels shifted against the activity (events in time order; label of
                event i paired with the activity of event i + k, no wrap-around; |k| = 5 .. n/3, 40 shifts; within_day
                001 shifts / shifted, the decoders' null); null median per quantity; corrected = observed - null median
                for d, delta d and along; lambda_corrected = along_corrected / d(AH,ref)_corrected (numerator and
                denominator corrected separately, as for the decoder ratios)
  coding direction  (a) lambda (split-half, 057); (b) unified split-half trial scores (coding_direction.split_half_scores):
                mean WH score (= lambda by construction) and its D; (c) Part II 5-fold coding direction (within_day 002):
                mean WH score and held-out d'
Output: across_days/<ref>/extras/extras_sessions.csv (+ provenance)
"""
import argparse
import importlib
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "within_day"))
m51 = importlib.import_module("051_roc_prelick")
m57 = importlib.import_module("057_roc_prelick_lambda")
m74 = importlib.import_module("074_geometry_normalisation")
m001 = importlib.import_module("001_within_session_halves")
CD = importlib.import_module("coding_direction")
OUT = m51.OUTROOT / "extras"
KEYS = ["d_WH_FA", "d_WH_AH", "d_AH_FA", "dd", "along", "lam"]


def geometry(Z, lab, seed=0):
    r = m57.lam(Z, lab, np.random.default_rng(seed))
    if not r:
        return None
    dd = r["d_WH_FA"] - r["d_WH_AH"]
    along = (r["d_WH_FA"] + r["d_AH_FA"] - r["d_WH_AH"]) / 2
    return dict(d_WH_FA=r["d_WH_FA"], d_WH_AH=r["d_WH_AH"], d_AH_FA=r["d_AH_FA"], dd=dd, along=along, lam=r["lam"])


def session(args):
    meta, f, Kq, norm = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])                                    # time order (needed by the shift null)
    X, raw, lab, rt = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o], z["rt"][o]
    if min((lab == c).sum() for c in m51.CLASSES) < 4:
        return None
    ok0 = (raw.mean(1) >= m51.MIN_FR) & Kq
    Z, ok = m74.scale(X, lab, norm)
    u = ok0 & ok
    if u.sum() < m57.MIN_UNITS:
        return None
    Z = Z[u]
    g = geometry(Z, lab)
    if g is None:
        return None
    row = dict(meta, norm=norm, n_units=int(u.sum()), **g)
    null = []
    for k in m001.shifts(len(lab)):
        Xs, ls, _ = m001.shifted(Z.T, lab, int(k))
        if min((ls == c).sum() for c in m51.CLASSES) < 4:
            continue
        gs_ = geometry(Xs.T, ls, seed=int(k) + 1000)
        if gs_:
            null.append(gs_)
    row["n_shifts"] = len(null)
    if null:
        N = pd.DataFrame(null)
        for kk in KEYS:
            row[f"{kk}_null_median"] = float(N[kk].median())
        for kk in ["d_WH_FA", "d_WH_AH", "d_AH_FA", "dd", "along"]:
            row[f"{kk}_corrected"] = row[kk] - row[f"{kk}_null_median"]
        den = row["d_AH_FA_corrected"]
        row["lam_corrected"] = row["along_corrected"] / den if den > 0 else np.nan
    sh = CD.split_half_scores(Z, lab)
    if sh is not None:
        row["lam_split_trials"], row["D_split"] = sh[2], sh[1]
    kf = CD.kfold_scores(Z.T, lab)
    md = kf["md"]
    row["cd_kfold_wh"] = float(np.nanmean(md[lab == "WH"])) if np.isfinite(md).any() else np.nan
    row["cd_kfold_dprime"] = float(kf["dprime"])
    return row


def main(a):
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        info = W[W.session_id == r.session_id][["electrode_group", "cluster_id", "cohort", "stage", "mouse_id", "quality_label"]]
        K = K.merge(info, on=["electrode_group", "cluster_id"], how="left")
        if K.cohort.isna().all():
            continue
        meta = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
                    stage=K.stage.dropna().iloc[0])
        jobs.append((meta, f, K.quality_label.isin(m57.UNIT_SET).to_numpy(), a.norm))
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        rows = [r for r in pool.imap_unordered(session, jobs) if r]
    D = pd.DataFrame(rows)
    D.to_csv(OUT / f"extras_sessions_{a.norm}.csv", index=False)
    json.dump(dict(script="076_session_geometry_extras.py", ref=m51.REF, norm=a.norm, n_sessions=len(D),
                   shifts="within_day 001 shifts (40, |k| 5..n/3)", splits=CD.N_SPLIT, date=time.strftime("%Y-%m-%d %H:%M")),
              open(OUT / f"provenance_{a.norm}.json", "w"), indent=1)
    cols = ["lam", "lam_split_trials", "cd_kfold_wh", "lam_corrected"]
    print(D[cols].corr(method="spearman").round(3).to_string())
    print(f"{len(D)} sessions, {round((time.time() - t0) / 60, 1)} min -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", default="pooled", choices=m74.NORMS)
    ap.add_argument("--n-proc", type=int, default=20)
    main(ap.parse_args())
