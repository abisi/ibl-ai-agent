"""101 -- Heatmaps of z-scored responses per area group (user 2026-10-11: "still a lot, differences of differences; z-score first;
heatmaps?"). Exploratory.
z: Δ rate of each unit (rate - its pre-stimulus baseline mean for that class) divided by the across-trial SD of that baseline rate
(floor 0.5 Hz); mean over units per session x area group (>= 5 units), then mean over sessions.
  WH, WM, AH: 099 psth_z (pre-stimulus baseline [-1, -0.015] s before trial start); misses at the pseudo-lick (stimulus + median WH RT)
  SL: 051 lick-aligned PSTHs with the 051 spontaneous-lick baseline ([-1, -0.5] s before the lick), same z definition
Rows: R+ day 0, R+ expert, R- day 0, R- expert, then expert - day 0 for R+ and R- (difference rows, diverging scale).
Columns: stimulus-aligned WH, WM, AH; lick-aligned WH, AH, SL. Areas (heatmap rows) in the custom order, those with >= 3 sessions per
cohort x stage. Colour scale shared within the four group rows (symmetric, 98th percentile) and within the difference rows.
Population: --pop learners | all. Output: across_days/<ref>/area_heatmaps/area_heatmaps_z_<pop>.{png,pdf}
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../101_area_heatmaps_z.py --pop learners
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m99 = importlib.import_module("099_area_psth_misses")
m88 = importlib.import_module("088_cosyne_across_days_v2")
m51, m61, m62, m81 = m88.m51, m88.m61, m88.m62, m88.m81
OUT = m51.OUTROOT / "area_heatmaps"
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
GL = {("R+", "learning"): "R+ day 0", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− day 0", ("R-", "expert"): "R− expert"}
COLS = [("stim", "WH"), ("stim", "WM"), ("stim", "AH"), ("lick", "WH"), ("lick", "AH"), ("lick", "SL")]


def load_sl(U):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(U.session_id.unique())]
    rows, edges = [], None
    for r in ss.itertuples():
        z = np.load(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        lab = z["cls"]; isl = lab == "FA"
        if isl.sum() < 3:
            continue
        bt = (z["raw"].astype(float) - z["rates"].astype(float))[:, isl]
        mu, sd = bt.mean(1), np.maximum(bt.std(1), m99.SD_MIN)
        P = z["psth"].astype(float)[:, 2]                               # class order WH, AH, FA
        P = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, "same"), 1, P)[:, 1:-1]
        Z = (P - mu[:, None]) / sd[:, None]; edges = z["psth_edges"]
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str))); K["row"] = np.arange(len(K))
        K = K.merge(U[U.session_id == r.session_id][["electrode_group", "cluster_id", "area_group"]], on=["electrode_group", "cluster_id"], how="inner")
        for ag, g in K.groupby("area_group"):
            if len(g) >= m99.MIN_UNITS:
                rows.append(dict(session_id=r.session_id, area_group=ag, align="lick", cls="SL", psth_z=Z[g.row.to_numpy()].mean(0)))
    tc = ((edges[:-1] + np.diff(edges) / 2) * 1e3)[1:-1]
    return pd.DataFrame(rows), tc


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    OUT.mkdir(parents=True, exist_ok=True)
    plt = m62.setup()
    U = pd.read_parquet(m81.OUT / "units.parquet")
    U["electrode_group"] = U.electrode_group.astype(str); U["cluster_id"] = U.cluster_id.astype(str)
    meta = U.drop_duplicates("session_id")[["session_id", "mouse_id", "cohort", "stage"]]
    P = pd.read_parquet(m99.OUT / "psth.parquet")[["session_id", "area_group", "align", "cls", "psth_z"]]
    P["psth_z"] = P.psth_z.apply(np.asarray)
    S, tc_sl = load_sl(U)
    D = pd.concat([P, S], ignore_index=True).merge(meta, on="session_id", how="left")
    D = m61.learner_filter(D) if a.pop == "learners" else D
    tt = {w: ((np.arange(lo, hi + m99.BIN / 2, m99.BIN)[:-1] + m99.BIN / 2) * 1e3)[1:-1] for w, (lo, hi) in m99.WINS.items()}
    n = D[D.cls == "WH"].groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack([1, 2]).fillna(0)
    areas = [g for g in au.get_area_group_custom_order() if g in n.index and all(n.loc[g].get(k, 0) >= 3 for k in GROUPS)]
    M = {}
    for (w, c) in COLS:
        x = tc_sl if c == "SL" else tt[w]
        lim = (x >= -500) & (x <= 300) if w == "lick" else np.ones(len(x), bool)
        for k in GROUPS:
            d = D[(D["align"] == w) & (D.cls == c) & (D.cohort == k[0]) & (D.stage == k[1])]
            M[(w, c, k)] = (x[lim], np.vstack([np.stack(d[d.area_group == ag].psth_z.to_numpy())[:, lim].mean(0) if (d.area_group == ag).any()
                                               else np.full(lim.sum(), np.nan) for ag in areas]))
    v = np.nanpercentile(np.abs(np.concatenate([m[1].ravel() for m in M.values()])), 98)
    dv = np.nanpercentile(np.abs(np.concatenate([(M[(w, c, (co, "expert"))][1] - M[(w, c, (co, "learning"))][1]).ravel()
                                                 for (w, c) in COLS for co in ("R+", "R-")])), 98)
    rows = [(k, "group") for k in GROUPS] + [(co, "diff") for co in ("R+", "R-")]
    fig, axs = plt.subplots(len(rows), len(COLS), figsize=(m62.W_IN, 0.16 * len(areas) * len(rows) + 1.0), squeeze=False)
    for i, (k, kind) in enumerate(rows):
        for j, (w, c) in enumerate(COLS):
            ax = axs[i, j]
            if kind == "group":
                x, Z = M[(w, c, k)]; im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, extent=(x[0], x[-1], len(areas) - 0.5, -0.5), interpolation="nearest")
                lab = GL[k]; colr = m62.COH[k[0]]
            else:
                x, Ze = M[(w, c, (k, "expert"))]; Zl = M[(w, c, (k, "learning"))][1]
                im2 = ax.imshow(Ze - Zl, aspect="auto", cmap="PuOr_r", vmin=-dv, vmax=dv, extent=(x[0], x[-1], len(areas) - 0.5, -0.5), interpolation="nearest")
                lab = f"{k.replace('-', '−')} expert − day 0"; colr = m62.COH[k]
            ax.axvline(0, color="k", lw=0.4, ls=":")
            if w == "lick":
                ax.axvline(-100, color="0.3", lw=0.3, ls=":")
            ax.set_yticks(range(len(areas)))
            ax.set_yticklabels([m88.m83.m62_short(g) for g in areas] if j == 0 else [], fontsize=3.8)
            ax.tick_params(labelsize=3.8, length=1.5)
            if i == 0:
                ax.set_title(f"{'stim' if w == 'stim' else 'lick'}: {c}", fontsize=5)
            if j == 0:
                ax.set_ylabel(lab, fontsize=4.8, color=colr)
            if i != len(rows) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("from stimulus (ms)" if w == "stim" else "from (pseudo-)lick (ms)", fontsize=4.2)
        if i == 3:
            cb = fig.colorbar(im, ax=axs[:4, :].ravel().tolist(), fraction=0.015, pad=0.01); cb.set_label("z (baseline SD)", fontsize=4.6); cb.ax.tick_params(labelsize=4)
    cb2 = fig.colorbar(im2, ax=axs[4:, :].ravel().tolist(), fraction=0.015, pad=0.01); cb2.set_label("Δz (expert − day 0)", fontsize=4.6); cb2.ax.tick_params(labelsize=4)
    fig.text(0.5, 1.0, f"z-scored responses per area ({a.pop}); mean over sessions", fontsize=5.8, ha="center", va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"area_heatmaps_z_{a.pop}.{ext}", dpi=300, bbox_inches="tight")
    print("ALL DONE", a.pop, len(areas), "areas")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
