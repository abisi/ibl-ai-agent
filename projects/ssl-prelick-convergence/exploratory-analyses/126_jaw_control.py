"""126 -- Trial-level jaw control for the single-neuron convergence (user 2026-10-14: "is the jaw opening control convincing, are there
stats to show the movements are similar? trial variance of movement vs mean?"; 2026-10-15: same neuron selection as 127).
Events, neurons and rates exactly as 127 (main figure 128, shared baseline): events of the 051 per-event files (WH, AH = corrected
first lick, SL = spontaneous-lick bout onset), kept if the shared baseline window [lick - 1.5, lick - 1.0) s is lick-free and pre-stimulus;
neurons = 127 units_table, non-sensory classes (081), mean raw pre-lick rate >= 0.1 Hz, no p-value selection; rate = spike count in
WIN = [-100, 0) ms before the lick minus the shared-baseline rate.
Jaw: 113 load_series "jaw opening" (mm re. the session's resting jaw) at the same lick times, 116 grid; per event mean opening in WIN
(events with > 50 % valid samples).
1 kinematic ROC   per session signed s = 2 AUC - 1 on per-event jaw opening (all jaw-valid events): s_WA_jaw (WH vs AH), s_WS_jaw (WH vs
                  SL); per-class trial SD of jaw opening and the SD ratio WH / AH.
2 neural s        mean over neurons of s_WA / s_WS on jaw-valid events, count-matched as 127 (N_DRAW draws of 4 WH / 4 AH / 8 SL, "_all"),
                  and jaw-matched ("_jm"): per draw WH and AH (or WH and SL) events paired 1:1 within JAW_BIN-mm bins, then N_WH of each
                  drawn; the matched draws' jaw s is saved to verify the matching (should be ~0).
Statistics: 062 group_stats (MWU + Welch, mouse-level cohort x stage permutation), unit = session, learners rule.
Output: <OUTROOT>/jaw_control/{sessions.csv, stats_<pop>.csv, jaw_control_<pop>.{png,pdf,svg}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable PRELICK_SELECT=all python .../126_jaw_control.py [--n-proc 20] [--replot]
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
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m80 = importlib.import_module("080_single_unit_geometry")
m81 = importlib.import_module("081_single_unit_classes")
m113 = importlib.import_module("113_dlc_passive_pre")
m116 = importlib.import_module("116_prelick_kinematics")
m127 = importlib.import_module("127_unified_single_neuron")
OUT = m51.OUTROOT / "jaw_control"
WIN = (-0.1, 0.0)
JAW_BIN, N_DRAW, MIN_UNITS = 0.2, 200, 3
N_WH, N_AH, N_SL = m127.N_WH, m127.N_AH, m127.N_SL


def auc1(a, b):
    return float(m80.auc_sel(np.r_[a, b][None], np.arange(len(a)), np.arange(len(a), len(a) + len(b)))[0])


def match(ia, ib, jaw, rng):
    ba, bb = np.floor(jaw[ia] / JAW_BIN), np.floor(jaw[ib] / JAW_BIN)
    sa, sb = [], []
    for b in np.intersect1d(ba, bb):
        a, c = ia[ba == b], ib[bb == b]; k = min(len(a), len(c))
        sa.append(rng.choice(a, k, replace=False)); sb.append(rng.choice(c, k, replace=False))
    return (np.concatenate(sa), np.concatenate(sb)) if sa else (ia[:0], ib[:0])


def neural_rates(sid, K, lick):
    """127 rates: spikes in [lick - 100, lick) ms minus the shared-baseline rate; lick-free shared-baseline mask"""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        Uq, _ = m51.ru.process_nwb_tables(nwb)
        lk = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"].time_series["piezo_lick_times"].data[:], float))
    Uq = Uq.assign(electrode_group=Uq.electrode_group.astype(str), cluster_id=Uq.cluster_id.astype(str)).set_index(["electrode_group", "cluster_id"])
    X = np.full((len(K), len(lick)), np.nan); B0, B1 = m127.SHARED
    for u, r in enumerate(K.itertuples()):
        if (r.electrode_group, r.cluster_id) not in Uq.index:
            continue
        st = np.sort(np.asarray(Uq.loc[(r.electrode_group, r.cluster_id)].spike_times))
        n = lambda a, b: np.searchsorted(st, lick + b) - np.searchsorted(st, lick + a)
        X[u] = n(*WIN) / (WIN[1] - WIN[0]) - n(B0, B1) / (B1 - B0)
    lickfree = np.searchsorted(lk, lick + B1) == np.searchsorted(lk, lick + B0)
    return X[np.isfinite(X).all(1)], lickfree


def session(args):
    sid, f, units = args
    z = np.load(f, allow_pickle=True)
    lab = z["cls"].astype(str); start = z["trial_start"].astype(float); rt = z["rt"].astype(float)
    lick = start + np.nan_to_num(rt); trial = lab != "FA"
    try:
        info, tr, series = m113.load_series(sid)
    except Exception:
        return None
    if "jaw opening" not in series:
        return None
    t, v = series["jaw opening"]; v = v - np.nanmedian(v)
    G = m113.to_grid(t, v, lick, None, grid=m116.GRID)
    w = (m116.GRID >= WIN[0]) & (m116.GRID < WIN[1])
    jok = np.isfinite(G[:, w]).mean(1) > 0.5
    jaw = np.where(jok, np.nanmean(G[:, w], 1), np.nan)
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))); K["row"] = np.arange(len(K))
    K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    K = K[K.nclass.isin(m81.CLASSES) & (K.nclass != "sensory")]
    K = K[z["raw"].astype(float)[K.row.to_numpy()].mean(1) >= m51.MIN_FR].reset_index(drop=True)
    X, lickfree = neural_rates(sid, K, lick)
    ok = jok & lickfree & (~trial | (lick + m127.SHARED[1] <= start))
    iw, ia, isl = (np.where((lab == c) & ok)[0] for c in ("WH", "AH", "FA"))
    if len(iw) < N_WH or len(ia) < N_AH or len(isl) < N_SL:
        return None
    row = dict(session_id=sid, n_wh=len(iw), n_ah=len(ia), n_sl=len(isl), n_units=len(X),
               jaw_WH=jaw[iw].mean(), jaw_AH=jaw[ia].mean(), jaw_SL=jaw[isl].mean(),
               sd_WH=jaw[iw].std(ddof=1), sd_AH=jaw[ia].std(ddof=1), sd_SL=jaw[isl].std(ddof=1),
               sWA_jaw=auc1(jaw[iw], jaw[ia]), sWS_jaw=auc1(jaw[iw], jaw[isl]))
    row["sd_ratio_WA"] = row["sd_WH"] / row["sd_AH"] if row["sd_AH"] > 0 else np.nan
    if len(X) < MIN_UNITS:
        return row
    rng = np.random.default_rng(zlib.crc32(f"{sid}|126".encode()))
    va, vs = [], []
    for _ in range(N_DRAW):
        w_, a_, s_ = rng.choice(iw, N_WH, replace=False), rng.choice(ia, N_AH, replace=False), rng.choice(isl, N_SL, replace=False)
        va.append(m80.auc_sel(X, w_, a_).mean()); vs.append(m80.auc_sel(X, w_, s_).mean())
    row.update(sWA_all=float(np.mean(va)), sWS_all=float(np.mean(vs)))
    for key, (a_, b_) in [("WA", (iw, ia)), ("WS", (iw, isl))]:
        vals, jv = [], []
        for _ in range(N_DRAW):
            ma, mb = match(a_, b_, jaw, rng)
            if min(len(ma), len(mb)) >= N_WH:
                k = rng.choice(len(ma), N_WH, replace=False); ma, mb = ma[k], mb[k]
                vals.append(m80.auc_sel(X, ma, mb).mean()); jv.append(auc1(jaw[ma], jaw[mb]))
        row[f"n_draw_jm_{key}"] = len(vals)
        if len(vals) >= N_DRAW / 2:
            row[f"s{key}_jm"], row[f"s{key}_jaw_jm"] = float(np.mean(vals)), float(np.mean(jv))
    return row


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    ss = m127.meta_all(); U0 = m127.units_table()
    jobs = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse_id}_roc_prelick{m51.TAG}_trials.npz"
        u = U0[U0.session_id == r.session_id][["electrode_group", "cluster_id", "nclass", "area_group"]].drop_duplicates()
        if f.exists() and len(u) and isinstance(r.cohort, str):
            jobs.append((r.session_id, f, u))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [r for r in pool.imap_unordered(session, jobs) if r is not None]
    meta = ss[["session_id", "mouse_id", "cohort", "stage"]]
    S = pd.DataFrame(res).merge(meta, on="session_id"); S.to_csv(OUT / "sessions.csv", index=False)
    print("sessions with jaw", len(S), flush=True)


def figure(pop):
    plt = m62.setup()
    m125 = importlib.import_module("125_single_neuron_figure")
    S = pd.read_csv(OUT / "sessions.csv"); S = S[S.cohort.isin(["R+", "R-"])]
    if pop == "learners":
        S = m61.learner_filter(S)
    rng = np.random.default_rng(1); m62.STATS.clear()
    panels = [("sWA_jaw", "Jaw s_WA\n(WH vs AH, per trial)"), ("sWS_jaw", "Jaw s_WS\n(WH vs SL, per trial)"),
              ("sd_ratio_WA", "Jaw trial SD\nWH / AH"),
              ("sWA_all", "Neural s_WA\n(jaw-valid events)"), ("sWA_jm", "Neural s_WA\n(jaw-matched)"),
              ("sWS_all", "Neural s_WS\n(jaw-valid events)"), ("sWS_jm", "Neural s_WS\n(jaw-matched)")]
    fig, axs = plt.subplots(1, len(panels), figsize=(7.8, 2.2))
    fig.subplots_adjust(left=0.06, right=0.975, top=0.76, bottom=0.27, wspace=1.0)
    for ax, (col, lab) in zip(axs, panels):
        m125.means_panel(ax, S.dropna(subset=[col]), col, lab, rng, f"126 {col}", show_x=True)
        if col == "sd_ratio_WA":
            ax.axhline(1, color="0.7", lw=0.5, ls=(0, (2, 2)))
    fig.suptitle(f"Jaw control, [−100, 0) ms before the first lick; 127 non-sensory neurons, shared baseline (matched jaw s_WA = "
                 f"{S.sWA_jaw_jm.mean():+.3f}, s_WS = {S.sWS_jaw_jm.mean():+.3f})", fontsize=6, y=0.98)
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"jaw_control_{pop}.{ext}", dpi=300)
    St = pd.DataFrame(m62.STATS); St.to_csv(OUT / f"stats_{pop}.csv", index=False)
    cols = ["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert",
            "n_R− learning", "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R+ L vs E p_Welch", "R- L vs E p_MWU",
            "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]
    pd.set_option("display.width", 300)
    print(St[[c for c in cols if c in St]].round(4).to_string())
    print("matched jaw s (should be ~0): WA", S.sWA_jaw_jm.describe().round(3).to_dict(), "WS", S.sWS_jaw_jm.describe().round(3).to_dict())


def main(a):
    if not a.replot:
        compute(a.n_proc)
    figure(a.pop); print("ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--n-proc", type=int, default=20); p.add_argument("--replot", action="store_true")
    p.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(p.parse_args())
