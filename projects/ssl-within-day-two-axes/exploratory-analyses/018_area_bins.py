"""018 -- Per-area reward-lick CD within the session with event-based bins (user 2026-10-16: "do equal-count bins, or per trial (or averages of
5 trials)"). Replaces the equal-duration fifths of 017, in which a fifth can hold very few whisker hits and a session without whisker hits in
the first fifth drops out.
Scores: per session and area group exactly as 017 (each area's own CD, estimator of within_day 002, baseline-corrected rates, held-out
SL = 0 / AH = 1); stored per event (tables/018_area_events.parquet).
Binnings (whisker-hit anchored, so that the references are measured at the same moments as the whisker hits):
  quint   the session's whisker hits split in time order into 5 bins of equal count (n_WH / 5 each); SL and AH scores averaged over the events
          falling in each bin's time window (from the first to the last whisker hit of the bin, extended to the midpoints between bins)
  blocks  consecutive blocks of 5 whisker hits (block 1 = whisker hits 1-5, ...); SL and AH averaged over each block's time window (midpoints
          as above); blocks up to MAX_BLOCK
Per session, area and bin: WH, WH - SL and WH - AH (bins without SL or AH events: difference undefined). Curves relative to the session's own
baseline: the first quintile (quint) or the mean of blocks 1-2 (blocks). Mean +- s.e.m. over sessions (points with >= 3 sessions); per-session
OLS slope over bins (>= 3 defined bins), Wilcoxon vs 0 per stage and Mann-Whitney day 0 vs expert.
Output: tables/018_area_events.parquet, 018_bins_<mode>.csv, 018_slopes_<mode>.csv; figures/018_area_<mode>.{png,pdf,svg}
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import numpy as np
import pandas as pd
from scipy import stats

import common as C

m002 = importlib.import_module("002_trial_slopes")
st26 = importlib.import_module("026_roc_rates_all_sessions")
MIN_UNITS, MIN_SESS, BLOCK, MAX_BLOCK = 10, 3, 5, 12
MEAS = [("WH", "whisker hits"), ("WH-SL", "WH − SL"), ("WH-AH", "WH − AH")]


def session(args):
    f, meta, W_s, seed = args
    z = np.load(f, allow_pickle=True)
    o = np.argsort(z["trial_start"])
    X, raw, lab, t = z["rates"].astype(float)[:, o], z["raw"].astype(float)[:, o], z["cls"][o].astype(str), z["trial_start"].astype(float)[o]
    if (lab == "WH").sum() < m002.MIN_WH or min((lab == c).sum() for c in ("AH", "FA")) < m002.MIN_EV:
        return []
    K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
    K = K.merge(W_s, on=["electrode_group", "cluster_id"], how="left")
    ok = K.quality_label.isin(m002.UNIT_SET).to_numpy() & (raw.mean(1) >= C.m51.MIN_FR)
    cls = np.where(lab == "FA", "SL", lab)
    out = []
    for area, m in [("whole_brain", ok)] + [(a, ok & (K.area_group == a).to_numpy()) for a in K.area_group.dropna().unique()]:
        if m.sum() < MIN_UNITS:
            continue
        S = m002.trial_scores(X[m].T, lab, seed)["md"]
        if np.isfinite(S).any():
            out.append(pd.DataFrame(dict(meta, area=area, cls=cls, t=t, cd=S)))
    return out


def compute(n_proc):
    W = pd.read_parquet(C.m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])].copy()
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    jobs = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{C.m51.TAG}_trials.npz"
        if f.exists():
            Ws = W[W.session_id == r.session_id]
            meta = dict(session_id=r.session_id, mouse_id=Ws.mouse_id.iloc[0], cohort=Ws.cohort.iloc[0], stage=Ws.stage.iloc[0])
            jobs.append((f, meta, Ws[["electrode_group", "cluster_id", "quality_label", "area_group"]], zlib.crc32(r.session_id.encode())))
    with mp.get_context("fork").Pool(n_proc) as pool:
        res = pool.map(session, jobs)
    E = C.population(pd.concat([d for r in res for d in r], ignore_index=True))
    E.to_parquet(C.TAB / "018_area_events.parquet", index=False)
    return E


def bins_one(E, mode):
    E = E.dropna(subset=["cd"]).sort_values("t")
    w = E[E.cls == "WH"]
    n = len(w)
    if n < 5:
        return []
    if mode == "quint":
        idx = np.array_split(np.arange(n), 5)
    else:
        idx = [np.arange(i, min(i + BLOCK, n)) for i in range(0, min(n, BLOCK * MAX_BLOCK), BLOCK)]
        idx = [i for i in idx if len(i) == BLOCK]
    tw = w.t.to_numpy(); edges = []
    for k, i in enumerate(idx):
        lo = -np.inf if k == 0 else 0.5 * (tw[idx[k - 1][-1]] + tw[i[0]])
        hi = np.inf if k == len(idx) - 1 else 0.5 * (tw[i[-1]] + tw[idx[k + 1][0]])
        edges.append((lo, hi))
    rows = []
    for k, (i, (lo, hi)) in enumerate(zip(idx, edges)):
        win = E[(E.t >= lo) & (E.t < hi)]
        sl, ah = win[win.cls == "SL"].cd, win[win.cls == "AH"].cd
        whm = w.cd.to_numpy()[i].mean()
        rows.append(dict(bin=k + 1, WH=whm, **{"WH-SL": whm - sl.mean() if len(sl) else np.nan, "WH-AH": whm - ah.mean() if len(ah) else np.nan},
                         n_SL=len(sl), n_AH=len(ah)))
    return rows


def binned(E, mode):
    rows = []
    for (sid, area), g in E.groupby(["session_id", "area"]):
        meta = g.iloc[0][["session_id", "mouse_id", "cohort", "stage", "area"]].to_dict()
        rows += [dict(meta, **r) for r in bins_one(g, mode)]
    B = pd.DataFrame(rows)
    for m, _ in MEAS:                                              # relative to the session's own baseline
        base = B[B.bin <= (1 if mode == "quint" else 2)].groupby(["session_id", "area"])[m].mean().rename("base")
        B = B.merge(base, on=["session_id", "area"], how="left"); B[f"rel_{m}"] = B[m] - B.base; B = B.drop(columns="base")
    return B


def slopes(B):
    rows = []
    for (sid, area, c, s_), g in B.groupby(["session_id", "area", "cohort", "stage"]):
        for m, _ in MEAS:
            v = g.dropna(subset=[m])
            if len(v) >= 3:
                rows.append(dict(session_id=sid, area=area, cohort=c, stage=s_, measure=m, slope=np.polyfit(v.bin, v[m], 1)[0]))
    Sl = pd.DataFrame(rows); St = []
    w = lambda v: stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan
    for (m, area, c), g in Sl.groupby(["measure", "area", "cohort"]):
        d0, e = g[g.stage == "learning"].slope, g[g.stage == "expert"].slope
        St.append(dict(measure=m, area=area, cohort=c, n_d0=len(d0), n_e=len(e), slope_d0=d0.mean(), slope_e=e.mean(), p_d0=w(d0), p_e=w(e),
                       p_stage=stats.mannwhitneyu(d0, e).pvalue if len(d0) >= 3 and len(e) >= 3 else np.nan))
    return pd.DataFrame(St)


def figure(B, St, mode):
    import ephys_utilities.allen_utils.allen_utils as au
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.5, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    n = B.drop_duplicates(["session_id", "area"]).groupby(["area", "cohort", "stage"]).size().unstack(["cohort", "stage"]).fillna(0)
    ok_area = {c: [a for a in n.index if n.loc[a].get((c, "learning"), 0) >= MIN_SESS and n.loc[a].get((c, "expert"), 0) >= MIN_SESS] for c in ("R+", "R-")}
    areas = ["whole_brain"] + [g for g in au.get_area_group_custom_order() if g in set(ok_area["R+"]) | set(ok_area["R-"])]
    gcol = au.get_custom_area_groups_colors()
    cols = [(m, c) for c in ("R+", "R-") for m, _ in MEAS]
    fig, axs = plt.subplots(len(areas), len(cols), figsize=(7.8, 1.05 * len(areas) + 0.6), gridspec_kw=dict(wspace=0.45, hspace=0.75))
    for i, area in enumerate(areas):
        axs[i, 0].annotate(area.replace("whole_brain", "Whole brain").replace("Somatosensory-", "Ss-").replace(" areas", ""), xy=(-0.42, 0.5),
                           xycoords="axes fraction", ha="right", va="center", fontsize=5.6, color=gcol.get(area, "k"), annotation_clip=False)
        for j, (m, c) in enumerate(cols):
            ax = axs[i, j]
            if area != "whole_brain" and area not in ok_area[c]:
                ax.axis("off"); continue
            for s_, ls, mfc, alpha in (("learning", (0, (3, 1.5)), "white", 0.7), ("expert", "-", None, 1.0)):
                d = B[(B.area == area) & (B.cohort == c) & (B.stage == s_)].pivot_table(index="session_id", columns="bin", values=f"rel_{m}")
                if d.empty:
                    continue
                nn = d.notna().sum(); keep = nn >= 3
                x = d.columns.to_numpy()[keep.to_numpy()]
                ax.errorbar(x + (0.08 if s_ == "expert" else -0.08), d.mean()[keep], d.sem()[keep], color=C.COH[c], ls=ls, marker="o", ms=2.0,
                            mfc=mfc or C.COH[c], lw=0.8, capsize=0, alpha=alpha, label=f"{'D0' if s_ == 'learning' else 'E'} ({len(d)})")
            ax.axhline(0, color="0.75", lw=0.4, ls=":")
            st = St[(St.measure == m) & (St.area == area) & (St.cohort == c)]
            if len(st):
                st = st.iloc[0]
                ax.text(0.02, 1.0, f"D0 {C.fp(st.p_d0)} · E {C.fp(st.p_e)} · D0≠E {C.fp(st.p_stage)}", transform=ax.transAxes, fontsize=3.6, va="bottom", color="0.25")
            ax.legend(frameon=False, fontsize=3.6, loc="lower right", handlelength=1.4, borderaxespad=0.1); ax.tick_params(labelsize=4.2, length=1.5)
            if i == 0:
                ax.set_title(f"{c.replace('-', '−')}: {dict(MEAS)[m]}", color=C.COH[c], fontsize=5.6, pad=10)
            if i == len(areas) - 1:
                ax.set_xlabel("whisker-hit quintile" if mode == "quint" else f"block of {BLOCK} whisker hits", fontsize=4.8)
    what = ("whisker-hit quintiles (equal numbers of whisker hits; references in the same time windows), relative to the first quintile"
            if mode == "quint" else f"blocks of {BLOCK} consecutive whisker hits (references in the same time windows), relative to the mean of blocks 1-2")
    fig.suptitle(f"Reward-lick CD within the session per area group: {what} (score units: held-out SL = 0, AH = 1); dashed open = day 0, solid filled = "
                 "expert; above each panel: per-session slope over bins, Wilcoxon vs 0 and day 0 vs expert (MWU)", fontsize=5.6, x=0.01, y=0.995, ha="left", wrap=True)
    fig.subplots_adjust(top=0.94, left=0.12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"018_area_{mode}.{ext}", dpi=300, bbox_inches="tight")


def main(a):
    E = pd.read_parquet(C.TAB / "018_area_events.parquet") if a.replot else compute(a.n_proc)
    pd.set_option("display.width", 220)
    for mode in ("quint", "blocks"):
        B = binned(E, mode); B.to_csv(C.TAB / f"018_bins_{mode}.csv", index=False)
        St = slopes(B); St.to_csv(C.TAB / f"018_slopes_{mode}.csv", index=False)
        figure(B, St, mode)
        print("==", mode); print(St[St.p_stage < 0.1].round(3).to_string(index=False))
        print(St[St.area == "whole_brain"].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=20); ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
