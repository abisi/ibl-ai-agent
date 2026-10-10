"""092 -- Time-resolved pre-lick ROC selectivity of reward-lick neurons (user 2026-10-10: "time-resolved ROC, 50-ms windows, 5-ms
stride"; do converging neurons acquire their selectivity at the same time for both modalities?).
Per session (051 events: WH, AH, SL; first lick = start + RT, SL = the lick), units: tested units of 081 (stable, rate >= 0.1 Hz)
that are reward-lick neurons (pre-lick AH > SL significant, positive selectivity), plus a random sample of other tested units
(control, up to N_CTRL per session). Spike counts in 50-ms windows stepped by 5 ms, centres from -500 to +200 ms around the lick;
rate minus the event's own baseline (051 baseline: raw - corrected pre-lick rate); per window ROC selectivity s = 2 AUC - 1 for
AH vs SL, WH vs SL and WH vs AH (no permutation test; descriptive).
Output: across_days/<ref>/timeresolved_roc/selectivity.parquet (mouse_id, session_id, electrode_group, cluster_id, cohort, stage,
rl, conv, nclass, pair, t, sel). Figure / onsets: 093.
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../092_timeresolved_prelick_roc.py [--n-proc 30]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m51 = importlib.import_module("051_roc_prelick")
m81 = importlib.import_module("081_single_unit_classes")
OUT = m51.OUTROOT / "timeresolved_roc"
WIN, STEP, T0, T1, N_CTRL = 0.050, 0.005, -0.500, 0.200, 50
TC = np.round(np.arange(T0, T1 + 1e-9, STEP), 3)


def auc_sel(A, B):
    """A: events_a x windows, B: events_b x windows -> selectivity 2 AUC - 1 per window (A > B positive)"""
    na, nb = len(A), len(B)
    R = np.apply_along_axis(rankdata, 0, np.vstack([A, B]))
    u = R[:na].sum(0) - na * (na + 1) / 2
    return 2 * u / (na * nb) - 1


def session(args):
    sid, f, units = args
    from pynwb import NWBHDF5IO
    z = np.load(f, allow_pickle=True)
    lab = z["cls"]; ev = z["trial_start"].astype(float) + np.nan_to_num(z["rt"].astype(float))
    base = (z["raw"].astype(float) - z["rates"].astype(float))           # units x events (Hz)
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K["row"] = np.arange(len(K))
    K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    if K.empty or min((lab == c).sum() for c in ("WH", "AH", "FA")) < 4:
        return []
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        U, _ = m51.ru.process_nwb_tables(io.read())
    U = U.assign(electrode_group=U.electrode_group.astype(str), cluster_id=U.cluster_id.astype(str))
    sp = {(r.electrode_group, r.cluster_id): np.sort(np.asarray(r.spike_times)) for r in U.itertuples()}
    rows = []
    idx = {c: np.where(lab == c)[0] for c in ("WH", "AH", "FA")}
    for r in K.itertuples():
        s = sp.get((r.electrode_group, r.cluster_id))
        if s is None:
            continue
        lo = np.searchsorted(s, ev[:, None] + TC[None] - WIN / 2); hi = np.searchsorted(s, ev[:, None] + TC[None] + WIN / 2)
        Rm = (hi - lo) / WIN - base[r.row][:, None]                        # events x windows
        for pair, (a, b) in (("AS", ("AH", "FA")), ("WS", ("WH", "FA")), ("WA", ("WH", "AH"))):
            sel = auc_sel(Rm[idx[a]], Rm[idx[b]])
            rows.append(dict(electrode_group=r.electrode_group, cluster_id=r.cluster_id, pair=pair, sel=sel.astype(np.float32)))
    return [dict(session_id=sid, **x) for x in rows]


def main(a):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    OUT.mkdir(parents=True, exist_ok=True)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda c: U0[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U0["rl"] = sig(f"sig:{m81.SEL_AS}") & (U0[f"sel:{m81.SEL_AS}"] > 0)
    U0["conv"] = U0.rl & sig(f"sig:{m81.SEL_WS}") & (U0[f"sel:{m81.SEL_WS}"] > 0)
    U0["electrode_group"] = U0.electrode_group.astype(str); U0["cluster_id"] = U0.cluster_id.astype(str)
    rng = np.random.default_rng(0)
    keep = []
    for sid, g in U0.groupby("session_id"):
        ctrl = g[~g.rl]
        keep.append(pd.concat([g[g.rl], ctrl.sample(min(N_CTRL, len(ctrl)), random_state=int(rng.integers(1e9)))]))
    U1 = pd.concat(keep)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(U1.session_id.unique())]
    jobs = [(r.session_id, r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz",
             U1[U1.session_id == r.session_id][["electrode_group", "cluster_id"]]) for r in ss.itertuples()]
    jobs = [j for j in jobs if j[1].exists()]
    print(len(jobs), "sessions", len(U1), "units", flush=True)
    with mp.get_context("fork").Pool(a.n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, jobs) for x in r]
    D = pd.DataFrame(res)
    meta = U1[["session_id", "electrode_group", "cluster_id", "mouse_id", "cohort", "stage", "rl", "conv", "nclass", "area_group"]]
    D = D.merge(meta, on=["session_id", "electrode_group", "cluster_id"], how="left")
    S = np.stack(D.sel.to_numpy()); D = D.drop(columns="sel")
    np.save(OUT / "selectivity.npy", S); D.to_parquet(OUT / "selectivity_index.parquet", index=False)
    np.save(OUT / "t.npy", TC)
    print("ALL DONE", OUT, S.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    main(ap.parse_args())
