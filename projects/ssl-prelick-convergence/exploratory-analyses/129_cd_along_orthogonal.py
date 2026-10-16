"""129 -- Population geometry of the WH - AH difference relative to the reward-lick coding direction (CD), on the data of the unified
single-neuron analysis (127: same events, shared lick-relative baseline, count-matched draws) (user 2026-10-14): do whisker hits converge
onto auditory hits along the CD while becoming different from them in another (orthogonal) dimension?
Per session, pre-lick window [-100, 0) ms, units z-scored over the session's valid events; CD Delta = AH - SL, v = WH - SL
(population mean vectors). Count-matched draws (N_WH WH, N_AH AH, N_SL SL; N_DRAW), each split into halves A / B; cross-validated
products summed over draws: Delta.Delta = Delta_A . Delta_B, v.Delta = (v_A . Delta_B + v_B . Delta_A) / 2, v.v = v_A . v_B. Then
  lambda = v.Delta / |Delta|^2       position of WH along the CD (0 = SL, 1 = AH); along gap of WH - AH = 1 - lambda
  omega  = sqrt(max(v.v - (v.Delta)^2 / |Delta|^2, 0)) / |Delta|      WH - AH difference orthogonal to the CD (same for v and WH - AH)
  |Delta|^2 / n_units                CD length (AH - SL separation per unit)
Unit sets (127 units_table, mean raw pre-lick rate >= 0.1 Hz): non-sensory (class not "sensory"; as the single-neuron figure) and all units. Baselines: shared (main), raw.
Sessions: >= MIN_UNITS units and the count-matched event minimum; |Delta|^2 > 0. Learners rule; cohort and stage from 127 meta_all.
Statistics: 062 group_stats via 125 means_panel (MWU + Welch, mouse-level cohort x stage permutation), unit = session.
Output: <OUTROOT>/cd_along_orthogonal/{sessions.csv, stats_<pop>.csv, fig_cd_along_orthogonal_<baseline>_<pop>.{png,pdf,svg}}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../129_cd_along_orthogonal.py [--n-proc 25] [--replot]
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
sys.path[:0] = [str(HERE), str(HERE / "within_day")]
m127 = importlib.import_module("127_unified_single_neuron")
m125 = importlib.import_module("125_single_neuron_figure")
m51, m61, m62, m81 = m127.m51, m127.m61, m125.m62, m127.m81
OUT = m51.OUTROOT / "cd_along_orthogonal"
N_DRAW, MIN_UNITS = 200, 10                     # 200 draws final (user 2026-10-15)
N_WH, N_AH, N_SL = m127.N_WH, m127.N_AH, m127.N_SL
WIN = (-0.1, 0.0)
SETS = ["nonsensory", "all"]
BASES = ["shared", "raw"]


def session(args):
    sid, f, units = args
    from pynwb import NWBHDF5IO
    z = np.load(f, allow_pickle=True)
    lab = z["cls"].astype(str); start = z["trial_start"].astype(float); lick = start + np.nan_to_num(z["rt"].astype(float))
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))); K["row"] = np.arange(len(K))
    K = K.merge(units, on=["electrode_group", "cluster_id"], how="inner")
    K = K[z["raw"].astype(float)[K.row.to_numpy()].mean(1) >= m51.MIN_FR].reset_index(drop=True)
    if len(K) < MIN_UNITS:
        return []
    with NWBHDF5IO(str(m51.NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read(); Uq, _ = m51.ru.process_nwb_tables(nwb)
        lk = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"].time_series["piezo_lick_times"].data[:], float))
    Uq = Uq.assign(electrode_group=Uq.electrode_group.astype(str), cluster_id=Uq.cluster_id.astype(str)).set_index(["electrode_group", "cluster_id"])
    s0, s1 = m127.SHARED
    W = np.full((len(K), len(lab)), np.nan); B = np.full_like(W, np.nan)
    for u, r in enumerate(K.itertuples()):
        if (r.electrode_group, r.cluster_id) not in Uq.index:
            continue
        st = np.sort(np.asarray(Uq.loc[(r.electrode_group, r.cluster_id)].spike_times))
        W[u] = (np.searchsorted(st, lick + WIN[1]) - np.searchsorted(st, lick + WIN[0])) / (WIN[1] - WIN[0])
        B[u] = (np.searchsorted(st, lick + s1) - np.searchsorted(st, lick + s0)) / (s1 - s0)
    ok_shared = (np.searchsorted(lk, lick + s1) == np.searchsorted(lk, lick + s0)) & ((lab == "FA") | (lick + s1 <= start))
    rng = np.random.default_rng(zlib.crc32(f"{sid}|129".encode()))
    rows = []
    for base in BASES:
        X = W - B if base == "shared" else W.copy()
        valid = ok_shared if base == "shared" else np.ones(len(lab), bool)
        iw, ia, isl = (np.where((lab == c) & valid)[0] for c in ("WH", "AH", "FA"))
        if len(iw) < N_WH or len(ia) < N_AH or len(isl) < N_SL:
            continue
        ev = np.r_[iw, ia, isl]
        for us in SETS:
            um = np.isfinite(X[:, ev]).all(1) & (np.nanstd(X[:, ev], 1) > 0)
            if us == "nonsensory":
                um &= (K.nclass != "sensory").to_numpy()
            if um.sum() < MIN_UNITS:
                continue
            Xz = X[um]; mu, sd = Xz[:, ev].mean(1, keepdims=True), Xz[:, ev].std(1, keepdims=True); Xz = (Xz - mu) / sd
            DD = VD = VV = 0.0
            for _ in range(N_DRAW):
                w, a, s = (rng.permutation(rng.choice(i_, n_, replace=False)) for i_, n_ in ((iw, N_WH), (ia, N_AH), (isl, N_SL)))
                h = lambda e: (Xz[:, e[: len(e) // 2]].mean(1), Xz[:, e[len(e) // 2:]].mean(1))
                (wA, wB), (aA, aB), (sA, sB) = h(w), h(a), h(s)
                dA, dB, vA, vB = aA - sA, aB - sB, wA - sA, wB - sB
                DD += dA @ dB; VD += (vA @ dB + vB @ dA) / 2; VV += vA @ vB
            if DD <= 0:
                continue
            lam = VD / DD; omega = np.sqrt(max(VV - VD ** 2 / DD, 0.0) / DD)
            rows.append(dict(session_id=sid, baseline=base, units=us, n_units=int(um.sum()), lam=lam, gap=1 - lam, omega=omega,
                             rho=np.sqrt(max(VV, 0.0) / DD), cd_len=DD / N_DRAW / um.sum()))
    return rows


def compute(n_proc):
    OUT.mkdir(parents=True, exist_ok=True)
    ss = m127.meta_all()
    U0 = m127.units_table()                                              # same units as 127 / 128
    jobs = []
    for r in ss[ss.cohort.isin(["R+", "R-"])].itertuples():
        f = r.file.parent / f"{r.mouse_id}_roc_prelick{m51.TAG}_trials.npz"
        u = U0[U0.session_id == r.session_id][["electrode_group", "cluster_id", "nclass"]].drop_duplicates()
        if f.exists() and len(u):
            jobs.append((r.session_id, f, u))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = [x for r in pool.imap_unordered(session, jobs) for x in r]
    S = pd.DataFrame(res).merge(ss[["session_id", "mouse_id", "cohort", "stage"]], on="session_id", how="left")
    S.to_csv(OUT / "sessions.csv", index=False); print("computed", S.groupby(["baseline", "units"]).session_id.nunique().to_string(), flush=True)


def schematic(ax):
    ax.set_xlim(-0.2, 1.3); ax.set_ylim(-0.3, 0.75); ax.axis("off"); ax.set_aspect("equal")
    SL, AH, WH = np.array([0, 0]), np.array([1, 0]), np.array([0.55, 0.42]); P = np.array([WH[0], 0])
    ax.annotate("", xy=AH + [0.22, 0], xytext=SL - [0.1, 0], arrowprops=dict(arrowstyle="-|>", color="0.55", lw=0.8, mutation_scale=6))
    ax.text(1.24, -0.09, "CD", fontsize=5.2, color="0.4", ha="center")
    ax.plot([P[0], AH[0]], [0, 0], color="#2c2cdb", lw=2.2, solid_capstyle="butt", zorder=2)
    ax.plot([WH[0], P[0]], [WH[1], 0], color="#d95f02", lw=2.2, solid_capstyle="butt", zorder=2)
    ax.plot([WH[0], AH[0]], [WH[1], 0], color="0.3", lw=0.7, ls=(0, (2, 1.5)), zorder=1)
    for pt, c, lab in [(SL, "#262626", "SL"), (AH, "#2c2cdb", "AH"), (WH, "#f7b519", "WH")]:
        ax.scatter(*pt, s=38, color=c, zorder=3, edgecolor="white", lw=0.6)
        ax.text(pt[0], pt[1] + (0.09 if lab == "WH" else -0.13), lab, ha="center", fontsize=5.6, color=c if lab != "WH" else "#b07e00")
    ax.text((P[0] + 1) / 2, 0.05, "along gap\n1 − λ", ha="center", va="bottom", fontsize=4.8, color="#2c2cdb")
    ax.text(P[0] - 0.04, WH[1] / 2, "orthogonal\nω", ha="right", va="center", fontsize=4.8, color="#d95f02")
    ax.text(0.85, 0.3, "WH − AH", fontsize=4.6, color="0.3", rotation=-38, ha="center")
    ax.set_title("WH − AH relative to the CD\n(normalised by |AH − SL|)", fontsize=5.4, pad=2)


def panel_plane(ax, S, rng):
    gx = m62.group_stats(S, "lam", "lambda", rng, "129 plane lam"); gy = m62.group_stats(S, "omega", "omega", rng, "129 plane omega")
    ax.scatter([0, 1], [0, 0], s=40, color=["#262626", "#2c2cdb"], zorder=4, edgecolor="white", lw=0.6)
    ax.text(0, -0.06, "SL", ha="center", va="top", fontsize=5.2, color="#262626"); ax.text(1, -0.06, "AH", ha="center", va="top", fontsize=5.2, color="#2c2cdb")
    for c in ["R+", "R-"]:
        p0 = (gx["means"][(c, "learning")], gy["means"][(c, "learning")]); p1 = (gx["means"][(c, "expert")], gy["means"][(c, "expert")])
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=m125.COH[c], lw=0.9, mutation_scale=6, shrinkA=3, shrinkB=3))
        for st in ["learning", "expert"]:
            k = (c, st)
            ax.errorbar(gx["means"][k], gy["means"][k], xerr=gx["sems"][k], yerr=gy["sems"][k], fmt="o", ms=3.4, color=m125.gcolor(c, st), lw=0.8, capsize=0, zorder=3)
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.set_xlabel("Along CD: λ (0 = SL, 1 = AH)"); ax.set_ylabel("Orthogonal: ω", labelpad=1)
    ax.set_xlim(-0.15, 1.15); ax.set_ylim(bottom=-0.12)
    ax.set_title("WH position relative to SL and AH", fontsize=5.2, pad=2)


def figure(pop, base):
    plt = m62.setup(); rng = np.random.default_rng(1)
    S0 = pd.read_csv(OUT / "sessions.csv"); S0 = S0[(S0.baseline == base) & S0.cohort.isin(["R+", "R-"])]
    if pop == "learners":
        S0 = m61.learner_filter(S0)
    m62.STATS.clear()
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.15, 1.15, 0.75, 0.75, 0.75], hspace=0.75, wspace=0.6, left=0.06, right=0.985, top=0.86, bottom=0.12)
    letters = []
    for i, us in enumerate(SETS):
        S = S0[S0.units == us]
        if i == 0:
            ax0 = fig.add_subplot(gs[i, 0]); schematic(ax0); letters.append((ax0, "a"))
        else:
            ax0 = fig.add_subplot(gs[i, 0]); ax0.axis("off")
            n = S.groupby(["cohort", "stage"]).session_id.nunique()
            ax0.text(0.0, 0.5, "All tested units\n(incl. sensory)\n\n" + "\n".join(f"{m125.cname(c)} {m125.STL[s_]}: {n.get((c, s_), 0)} sessions"
                                                                                 for c in ["R+", "R-"] for s_ in ["learning", "expert"]),
                     fontsize=5.2, va="center", transform=ax0.transAxes)
        axp = fig.add_subplot(gs[i, 1]); panel_plane(axp, S, rng)
        axs = [fig.add_subplot(gs[i, 2 + j]) for j in range(3)]
        for ax, (col, lab) in zip(axs, [("gap", "Along gap 1 − λ\n(WH − AH on the CD)"), ("omega", "Orthogonal ω\n(WH − AH off the CD)"),
                                        ("cd_len", "CD length\n|AH − SL|² per unit")]):
            m125.means_panel(ax, S, col, lab, rng, f"129 {us} {col}", show_x=True)
        fig.text(0.075, [0.9, 0.455][i], {"nonsensory": "Non-sensory neurons (as the single-neuron figure)", "all": "All tested units"}[us],
                 fontsize=6, weight="bold", color="0.25")
        if i == 0:
            letters += [(axp, "b"), (axs[0], "c")]
    h = [plt.Line2D([], [], color=m125.gcolor(c, s_), lw=0, marker="o", ms=3.2) for c in ["R+", "R-"] for s_ in ["learning", "expert"]]
    fig.legend(h, [f"{m125.cname(c)} {m125.STL[s_]}" for c in ["R+", "R-"] for s_ in ["learning", "expert"]], frameon=False, fontsize=4.8, ncol=4,
               loc="upper right", bbox_to_anchor=(0.985, 0.958))
    bl = "shared baseline [−1.5, −1.0) s before the lick" if base == "shared" else "no baseline"
    fig.text(0.06, 0.975, f"Population geometry, [−100, 0) ms before the first lick; units z-scored; count-matched draws ({N_WH} WH / {N_AH} AH / "
             f"{N_SL} SL × {N_DRAW}), split-half cross-validated; {bl}; mean ± SEM over sessions", fontsize=4.4, color="0.35", va="top")
    for ax, l in letters:
        fig.text(ax.get_position().x0 - 0.035, 0.875, l, fontsize=9, weight="bold", va="top")
    name = f"fig_cd_along_orthogonal_{base}_{pop}"
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    St = pd.DataFrame(m62.STATS); St.to_csv(OUT / f"{name}_stats.csv", index=False)
    cols = ["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert",
            "n_R− learning", "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "R- L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"]
    pd.set_option("display.width", 300); print(St[[c for c in cols if c in St]].round(4).to_string())


def main(a):
    if not a.replot:
        compute(a.n_proc)
    for base in BASES:
        figure(a.pop, base)
    print("ALL DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--n-proc", type=int, default=25); p.add_argument("--replot", action="store_true")
    p.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(p.parse_args())
