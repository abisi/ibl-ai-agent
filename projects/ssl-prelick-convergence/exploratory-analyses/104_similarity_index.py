"""104 -- Per-neuron similarity index of the whisker-hit response to the auditory-hit vs spontaneous-lick responses (user 2026-10-11),
and COSYNE across days v6 (103 with panel c = this index; 103 kept).
Neurons: the reward-lick neurons of 095 (selected on AH vs SL-A, label permutation p < 0.05, s > 0); the same random SL split per
session (seed crc32("<sid>|095")), so only SL-B is used here. Data: 051 per-event baseline-corrected pre-lick rates ([-100, 0) ms).
Cross-validated selectivity magnitudes (N_SPLIT = 50 random halves of each class's events):
  |s_WS| = sqrt(max(mean_k s_WS^a s_WS^b, 0)),  s_WS = 2 AUC - 1 of WH vs SL-B on half a / b
  |s_WA| likewise for WH vs AH
  SI = (|s_WS| - |s_WA|) / (|s_WS| + |s_WA|)  in [-1, 1]; +1: WH like AH (differs from SL only); -1: WH like SL; NaN if both are 0
Noise alone gives |s| ~ 0 (the product of independent halves averages to zero), so SI is not pulled toward 0 by few events.
Per session: mean SI and fraction SI > 0 over its reward-lick neurons (>= 3 with a defined SI).
Outputs: across_days/<ref>/similarity_index/units.parquet, sessions.csv; publication/<pop>/COSYNE_across_days_v6.{png,pdf,svg} + _stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../104_similarity_index.py [--n-proc 30] [--replot] [--pop learners]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m103 = importlib.import_module("103_cosyne_across_days_v5")
m96, m89, m93, m88 = m103.m96, m103.m89, m103.m93, m103.m88
m51, m61, m62, m81, m83 = m88.m51, m88.m61, m88.m62, m88.m81, m88.m83
m80 = importlib.import_module("080_single_unit_geometry")
OUT = m51.OUTROOT / "similarity_index"
N_SPLIT = 50


def cv_abs(X, ia, ib, rng):
    """X units x events; cross-validated |s| of ia vs ib per unit"""
    acc = np.zeros(len(X))
    for _ in range(N_SPLIT):
        pa, pb = rng.permutation(ia), rng.permutation(ib)
        a1, a2, b1, b2 = pa[: len(pa) // 2], pa[len(pa) // 2:], pb[: len(pb) // 2], pb[len(pb) // 2:]
        acc += m80.auc_sel(X, a1, b1) * m80.auc_sel(X, a2, b2)
    return np.sqrt(np.clip(acc / N_SPLIT, 0, None))


def session(args):
    sid, f, units = args
    z = np.load(f, allow_pickle=True)
    lab = z["cls"]; X = z["rates"].astype(float)
    iw, ia, isl = (np.where(lab == c)[0] for c in ("WH", "AH", "FA"))
    if min(len(iw), len(ia)) < 4 or len(isl) < 16:
        return []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|095".encode()))
    p = rng.permutation(isl); slB = np.sort(p[len(p) // 2:])                  # same split as 095
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))); K["row"] = np.arange(len(K))
    K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    if K.empty:
        return []
    Xs = X[K.row.to_numpy()]
    r2 = np.random.default_rng(zlib.crc32(f"{sid}|104".encode()))
    sWS, sWA = cv_abs(Xs, iw, slB, r2), cv_abs(Xs, iw, ia, r2)
    den = sWS + sWA
    si = np.where(den > 0, (sWS - sWA) / np.where(den > 0, den, 1), np.nan)
    return [dict(session_id=sid, electrode_group=e, cluster_id=c, abs_sWS=a, abs_sWA=b, SI=s) for e, c, a, b, s in zip(K.electrode_group, K.cluster_id, sWS, sWA, si)]


def compute(n_proc):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(m96.SPLIT / "selectivity_index.parquet")
    Un = D.drop_duplicates(["session_id", "electrode_group", "cluster_id"])[["session_id", "electrode_group", "cluster_id", "mouse_id", "cohort", "stage", "nclass"]]
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(Un.session_id.unique())]
    jobs = [(r.session_id, r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", Un[Un.session_id == r.session_id][["electrode_group", "cluster_id"]])
            for r in ss.itertuples()]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, [j for j in jobs if j[1].exists()]) for x in r]
    U = pd.DataFrame(res).merge(Un, on=["session_id", "electrode_group", "cluster_id"], how="left")
    U.to_parquet(OUT / "units.parquet", index=False)
    S = U.dropna(subset=["SI"]).groupby(["session_id", "mouse_id", "cohort", "stage"]).agg(SI=("SI", "mean"), frac_pos=("SI", lambda x: (x > 0).mean()), n=("SI", "size")).reset_index()
    S[S.n >= 3].to_csv(OUT / "sessions.csv", index=False)
    print("computed", len(U), "units", S.session_id.nunique(), "sessions", flush=True)


def figure(pop):
    flt = m61.learner_filter if pop == "learners" else (lambda d: d)
    SI = flt(pd.read_csv(OUT / "sessions.csv"))
    # reuse 103 main with panel c replaced: patch mean_panel2 for the "WS" call
    orig = m88.mean_panel2

    def patched(ax, d, col, ylab, rng, panel, title, ref=None):
        if panel == "103 WS":
            return orig(ax, SI, "SI", "similarity index SI\n(−1 SL-like … +1 AH-like)", rng, "104 SI", "Whisker hits: AH- vs SL-like\n(per neuron, cross-validated)", ref=0)
        return orig(ax, d, col, ylab, rng, panel, title, ref)
    m88.mean_panel2 = patched
    orig_save = m62.save
    m62.save = lambda fig, out, name: orig_save(fig, out, "COSYNE_across_days_v6")
    out = m51.OUTROOT / "publication" / pop
    v5 = out / "COSYNE_across_days_v5_stats.csv"; bak = v5.read_bytes() if v5.exists() else None     # 103 main writes this file: keep v5's
    try:
        m103.main(argparse.Namespace(pop=pop))
    finally:
        m88.mean_panel2 = orig; m62.save = orig_save
        if v5.exists():
            v5.replace(out / "COSYNE_across_days_v6_stats.csv")
        if bak is not None:
            v5.write_bytes(bak)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("104")])
    St.to_csv(out / "COSYNE_across_days_v6_SI_stats.csv", index=False)
    S2 = SI.groupby(["cohort", "stage"]).agg(SI=("SI", "mean"), frac_pos=("frac_pos", "mean"), n=("session_id", "nunique"))
    print(S2.round(3)); print(St[["mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "interaction_p_perm", "R+ L vs E p_MWU",
                                  "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    fr = SI.copy(); g = m62.group_stats(fr, "frac_pos", "frac SI > 0", np.random.default_rng(1), "104 frac")
    print("frac SI>0 interaction p", round(g["interaction"][1], 4), {k: round(v, 3) for k, v in g["means"].items()})


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    ap.add_argument("--replot", action="store_true")
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
