"""095 -- 094 without the shared-reference bias (user 2026-10-10): spontaneous licks split at random into halves A and B per session
(seed fixed); neurons selected on AH vs SL-A, selectivity over time measured against SL-B only.
Selection: tested units of 081 (stable, mean pre-lick rate >= 0.1 Hz); pre-lick baseline-corrected rates (051, [-100, 0) ms);
s = 2 AUC - 1 for AH vs SL-A, two-sided label-permutation p (N_PERM = 1000; note: 051 used a linear-shift null for the SL
reference, here a plain permutation within the session); reward-lick (split) = p < 0.05 and s > 0.
Measurement: as 092 (50-ms windows, 5-ms stride, -500 .. +200 ms from the first lick, rate minus the event's baseline), AH vs SL-B,
WH vs SL-B, WH vs AH. Summaries and statistics as 094. Learners rule.
Output: across_days/<ref>/timeresolved_roc/split_sl/{selectivity.npy, selectivity_index.parquet, t.npy, ws_in_rl_split.{png,pdf}, stats.csv}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../095_timeresolved_split_sl.py [--n-proc 25] [--replot]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m92 = importlib.import_module("092_timeresolved_prelick_roc")
m51, m81 = m92.m51, m92.m81
OUT = m92.OUT / "split_sl"
TC, WIN = m92.TC, m92.WIN
N_PERM = 1000


def perm_sel(X, ia, ib, rng):
    """X units x events; s = 2 AUC - 1 (ia > ib) and two-sided permutation p per unit"""
    idx = np.r_[ia, ib]; na, nb = len(ia), len(ib)
    R = np.apply_along_axis(rankdata, 1, X[:, idx])
    f = lambda cols: 2 * (R[:, cols].sum(1) - na * (na + 1) / 2) / (na * nb) - 1
    obs = f(np.arange(na))
    cnt = np.zeros(len(X))
    for _ in range(N_PERM):
        p = rng.permutation(na + nb)[:na]
        cnt += np.abs(f(p)) >= np.abs(obs) - 1e-12
    return obs, (1 + cnt) / (1 + N_PERM)


def session(args):
    sid, f, units = args
    from pynwb import NWBHDF5IO
    z = np.load(f, allow_pickle=True)
    lab = z["cls"]; ev = z["trial_start"].astype(float) + np.nan_to_num(z["rt"].astype(float))
    rates = z["rates"].astype(float); base = z["raw"].astype(float) - rates
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K["row"] = np.arange(len(K)); K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    iw, ia, isl = (np.where(lab == c)[0] for c in ("WH", "AH", "FA"))
    if K.empty or min(len(iw), len(ia)) < 4 or len(isl) < 16:
        return []
    rng = np.random.default_rng(zlib.crc32(f"{sid}|095".encode()))
    p = rng.permutation(isl); slA, slB = np.sort(p[: len(p) // 2]), np.sort(p[len(p) // 2:])
    s, pv = perm_sel(rates[K.row.to_numpy()], ia, slA, rng)
    K = K.assign(selA=s, pA=pv); K = K[(K.pA < 0.05) & (K.selA > 0)]
    if K.empty:
        return []
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        U, _ = m51.ru.process_nwb_tables(io.read())
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str))
    sp = {(r.electrode_group, r.cluster_id): np.sort(np.asarray(r.spike_times)) for r in U.itertuples()}
    out = []
    for r in K.itertuples():
        st = sp.get((r.electrode_group, r.cluster_id))
        if st is None:
            continue
        lo = np.searchsorted(st, ev[:, None] + TC[None] - WIN / 2); hi = np.searchsorted(st, ev[:, None] + TC[None] + WIN / 2)
        Rm = (hi - lo) / WIN - base[r.row][:, None]
        for pair, (a, b) in (("AS", (ia, slB)), ("WS", (iw, slB)), ("WA", (iw, ia))):
            out.append(dict(session_id=sid, electrode_group=r.electrode_group, cluster_id=r.cluster_id, pair=pair,
                            sel=m92.auc_sel(Rm[a], Rm[b]).astype(np.float32)))
    return out


def compute(n_proc):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(U0.session_id.unique())]
    jobs = [(r.session_id, r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz",
             U0[U0.session_id == r.session_id][["electrode_group", "cluster_id"]]) for r in ss.itertuples()]
    jobs = [j for j in jobs if j[1].exists()]
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, jobs) for x in r]
    D = pd.DataFrame(res)
    D = D.merge(U0[["session_id", "electrode_group", "cluster_id", "mouse_id", "cohort", "stage", "nclass"]], on=["session_id", "electrode_group", "cluster_id"], how="left")
    S = np.stack(D.sel.to_numpy()); D = D.drop(columns="sel").assign(rl=True)
    np.save(OUT / "selectivity.npy", S); D.to_parquet(OUT / "selectivity_index.parquet", index=False); np.save(OUT / "t.npy", TC)
    print("computed", S.shape, flush=True)


def main(a):
    if not a.replot:
        compute(a.n_proc)
    m94 = importlib.import_module("094_timeresolved_ws_in_rl")
    m93, m88, m61, m62 = m94.m93, m94.m88, m94.m61, m94.m62
    m93.OUT = OUT
    plt = m62.setup(); rng = np.random.default_rng(0)
    D = pd.read_parquet(OUT / "selectivity_index.parquet"); S = np.load(OUT / "selectivity.npy"); t = np.load(OUT / "t.npy")
    keep = D.index.isin(m61.learner_filter(D).index); D, S = D[keep].reset_index(drop=True), S[keep]
    rl = np.ones(len(D), bool)
    rows = []
    for pr in ("WS", "AS"):
        mm = (D.pair == pr).to_numpy()
        for w, (lo, hi) in (("early", (-0.3, -0.1)), ("late", (-0.1, 0.0))):
            rows.append(D[mm][["session_id", "mouse_id", "cohort", "stage"]].assign(pair=pr, win=w, v=S[mm][:, (t >= lo) & (t < hi)].mean(1)))
    R = pd.concat(rows).groupby(["session_id", "mouse_id", "cohort", "stage", "pair", "win"]).v.mean().unstack(["pair", "win"])
    R.columns = [f"{p}_{w}" for p, w in R.columns]; R = R.reset_index(); R["ratio"] = R.WS_late / R.AS_late
    fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.32))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.4, 1, 1, 1], wspace=0.6, left=0.06, right=0.99, top=0.78, bottom=0.2)
    m93.insets(fig, gs[0, 0], D, S, t, rl, "a")
    fig.text(0.06, 0.9, "a  Reward-lick neurons selected on AH vs SL-A; selectivity vs SL-B (held-out half)", fontsize=5.6)
    for k, (col, ylab, ttl) in enumerate((("WS_early", "WH vs SL-B selectivity", "WH vs SL, −300 to −100 ms"),
                                           ("WS_late", "WH vs SL-B selectivity", "WH vs SL, −100 to 0 ms"),
                                           ("ratio", "WH vs SL-B / AH vs SL-B", "Generalisation ratio\n(−100 to 0 ms)"))):
        ax = fig.add_subplot(gs[0, 1 + k]); ax.set_box_aspect(1)
        m88.mean_panel2(ax, R, col, ylab, rng, f"095 {col}", ttl, ref=0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"ws_in_rl_split.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("095")]); St.to_csv(OUT / "stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning",
              "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]].round(4).to_string())
    print("ALL DONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=25)
    ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
