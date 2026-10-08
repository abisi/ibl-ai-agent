"""085 -- Example sessions in neural state space for the reward-lick axis (user 2026-10-08: "produce a similar view [to the 161
schematic], finding good axes, with an example session").
Per example session (R+ and R-, learning day; stable units with mean pre-lick rate >= 0.1 Hz, as 002):
  axes (orthonormal): x = u, the unit reward-lick direction mean_z(AH) - mean_z(SL) in the pre-lick window [-100, 0) ms;
  y = the part of the whisker-hit displacement mean_z(WH) - mean_z(SL) orthogonal to u (the omega direction of 084);
  z = the first principal component of the class-mean trajectories (-250 ms to the lick) orthogonal to x and y
  curves: class-mean lick-aligned trajectories (051 per-unit PSTHs, 10-ms bins, 30-ms boxcar), z-scored with the mean / SD of the
  per-trial window rates over AH + SL trials; thick: the pre-lick window; dots: window means
  WH early / late: mean of the whisker hits in the first / second half of the session (normalised time tau < / >= 0.5), from the
  per-trial window rates, with their projections on u (diamonds)
  origin: the SL window mean (so x = 0 at SL).
Raw window rates are used for curves and points alike (the analysis itself subtracts each trial's baseline); illustration, in-sample
axis (the analysis axis is cross-validated).
Examples: AB138 (R+) and MH008 (R-), the sessions shown in the within-day COSYNE figure (largest WH slope with stable SL and AH).
Outputs: within_day/<ref>/example_state_space/example_state_space.{svg,pdf,png} (SVG text as text)
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../085_example_state_space.py
"""
import importlib
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
st26 = importlib.import_module("026_roc_rates_all_sessions")
OUT = m51.WITHIN / "example_state_space"
EXAMPLES = [("AB138", "R+"), ("MH008", "R-")]
COL = {"WH": "#f7b519", "AH": "#2c2cdb", "FA": "#262626"}
LAB = {"WH": "WH", "AH": "AH", "FA": "SL"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
T0, T1, W0 = -0.25, 0.0, -0.10
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7})


def session_data(mouse):
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    ss = st26.all_sessions()
    r = ss[(ss.mouse == mouse) & ss.session_id.isin(W[W.stage == "learning"].session_id)].iloc[0]
    z = np.load(r.file.parent / f"{mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
    Ws = W[W.session_id == r.session_id].copy()
    Ws["electrode_group"] = Ws.electrode_group.astype(str); Ws["cluster_id"] = Ws.cluster_id.astype(str)
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(Ws[["electrode_group", "cluster_id", "quality_label"]], on=["electrode_group", "cluster_id"], how="left")
    raw = z["raw"].astype(float)
    ok = K.quality_label.isin(["good", "mua"]).to_numpy() & (raw.mean(1) >= m51.MIN_FR)
    o = np.argsort(z["trial_start"])
    return r.session_id, raw[ok][:, o], z["cls"][o], z["trial_start"].astype(float)[o], z["psth"].astype(float)[ok], z["psth_edges"]


def axes_for(raw, cls, psth, edges):
    m_ = np.isin(cls, ["AH", "FA"])
    mu, sd = raw[:, m_].mean(1), raw[:, m_].std(1); sd[sd == 0] = 1
    Z = (raw - mu[:, None]) / sd[:, None]
    u = Z[:, cls == "AH"].mean(1) - Z[:, cls == "FA"].mean(1); u /= np.linalg.norm(u)
    t = (edges[:-1] + edges[1:]) / 2
    k = np.ones(5) / 5
    P = np.stack([np.apply_along_axis(lambda x: np.convolve(x, k, "same"), 1, psth[:, c]) for c in range(3)], 1)
    Pz = (P - mu[:, None, None]) / sd[:, None, None]                      # units x class x time
    sel = (t >= T0) & (t < T1)
    traj = Pz[:, :, sel]                                                   # units x 3 x T
    M = traj.reshape(traj.shape[0], -1).T                                  # samples x units
    v = Z[:, cls == "WH"].mean(1) - Z[:, cls == "FA"].mean(1)
    w = v - (v @ u) * u; w /= np.linalg.norm(w)
    M = M - M.mean(0)
    M = M - np.outer(M @ u, u) - np.outer(M @ w, w)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    q = Vt[0] - (Vt[0] @ u) * u - (Vt[0] @ w) * w
    B = np.stack([u, w, q / np.linalg.norm(q)], 1)
    return Z, traj, t[sel], B


def panel(ax, mouse, coh):
    sid, raw, cls, ts, psth, edges = session_data(mouse)
    Z, traj, t, B = axes_for(raw, cls, psth, edges)
    win = t >= W0
    wm = {c: traj[:, ["WH", "AH", "FA"].index(c)][:, win].mean(1) for c in ("WH", "AH", "FA")}   # trajectory window means
    zSL = wm["FA"]
    off = wm["WH"] - Z[:, cls == "WH"].mean(1)       # per-trial points shifted onto the trajectory scale (same window)
    for c in ("FA", "AH", "WH"):
        Y = (traj[:, ["WH", "AH", "FA"].index(c)].T - zSL) @ B                  # T x 3
        ax.plot(*Y[~win].T, color=COL[c], lw=0.8, alpha=0.5)
        ax.plot(*Y[win].T, color=COL[c], lw=2.4)
        m = (wm[c] - zSL) @ B
        ax.scatter(*m, color=COL[c], s=24, depthshade=False, edgecolor="white", linewidth=0.5)
        ax.text(*(m + np.r_[0, 0, 0.6]), LAB[c], color=COL[c], fontsize=7, ha="center")
    tau = (ts - ts.min()) / max(ts.max() - ts.min(), 1e-9)
    for half, ls, txt in ((tau < 0.5, "o", "WH early"), (tau >= 0.5, "s", "WH late")):
        k = (cls == "WH") & half
        if k.sum() < 3:
            continue
        p = (Z[:, k].mean(1) + off - zSL) @ B
        ax.scatter(*p, marker=ls, s=26, facecolor="white" if txt.endswith("early") else "#f7b519", edgecolor="#b07e00",
                   linewidth=1.2, depthshade=False, zorder=6)
        ax.plot([p[0], p[0]], [p[1], 0], [p[2], 0], color="#b07e00", lw=0.7, ls=(0, (2, 1.5)))
        ax.scatter(p[0], 0, 0, marker="D", s=12, color="#b07e00", depthshade=False)
        ax.text(*(p + np.r_[0, 0, -0.7]), f"{txt} (n={k.sum()})", fontsize=6, color="#b07e00", ha="center")
    a = (wm["AH"] - zSL) @ B
    ax.quiver(0, 0, 0, a[0] * 1.15, 0, 0, color="black", lw=1.4, arrow_length_ratio=0.06)
    ax.text(a[0] * 1.22, 0, 0, "u", fontsize=8, weight="bold")
    ax.set_xlabel("reward-lick axis u (SD)", labelpad=-6); ax.set_ylabel("WH orthogonal to u", labelpad=-6)
    ax.set_zlabel("PC (remaining)", labelpad=-6)
    for a_ in (ax.xaxis, ax.yaxis, ax.zaxis):
        a_.pane.fill = False; a_.pane.set_edgecolor("0.85"); a_._axinfo["grid"]["color"] = (0.92, 0.92, 0.92, 1)
    ax.tick_params(labelsize=5, pad=-2)
    ax.view_init(elev=22, azim=-60)
    ax.set_title(f"{coh.replace('-', '−')} example: {sid}\n{Z.shape[0]} units; WH {int((cls == 'WH').sum())}, AH {int((cls == 'AH').sum())}, "
                 f"SL {int((cls == 'FA').sum())} licks", color=COH[coh], fontsize=7)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 3.6))
    for i, (mouse, coh) in enumerate(EXAMPLES):
        panel(fig.add_subplot(1, 2, i + 1, projection="3d"), mouse, coh)
    fig.text(0.5, 0.02, "curves: class-mean trajectories, −250 ms to the first lick (thick: pre-lick window [−100, 0) ms); "
             "dots: window means; origin = SL window mean\nWH early / late: whisker hits in the first / second half of the session, "
             "diamonds: their projection on u; axes orthonormal: x = u, y = whisker-hit displacement orthogonal to u", ha="center", fontsize=5.8)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.88, bottom=0.1, wspace=0.05)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"example_state_space.{ext}", dpi=300)
    print("ALL DONE", OUT / "example_state_space.png")


if __name__ == "__main__":
    main()
