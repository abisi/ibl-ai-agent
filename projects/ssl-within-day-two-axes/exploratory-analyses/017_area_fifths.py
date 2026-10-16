"""017 -- Per-area within-session drift on the reward-lick CD, day 0 vs expert, per fifth of the session (user 2026-10-16: "make a figure
that compares day 0 drift (whisker only, and differences relative to a reference) for areas and learning stages. Show fifths").
Per session and area group (units of that group: stable good / mua, mean raw pre-lick rate >= 0.1 Hz, >= MIN_UNITS), the reward-lick CD and
per-event scores exactly as within_day 002 (5-fold CV over AH + SL, units z-scored on the training folds, unit-norm AH - SL axis; held-out
AH / SL and fold-averaged WH projections; session normalisation held-out SL = 0, AH = 1; held-out AH-vs-SL d' >= 0.3); baseline-corrected
rates (051). Whole brain = all units (as 008).
Per session, area and fifth of the session (normalised time tau): mean WH, SL and AH scores; curves relative to the first fifth for
  WH        whisker hits alone
  WH - SL   whisker hits relative to spontaneous licks measured in the same fifth
  WH - AH   whisker hits relative to auditory hits measured in the same fifth
Displayed: mean +- s.e.m. over sessions per fifth (fifths with >= 3 sessions), day 0 (dashed, open) vs expert (solid, filled); areas with
>= MIN_SESS sessions at both stages for a cohort, allen_utils order; rows = areas, columns = measure x cohort. Annotation: per-session slope
of the five fifth values (per session), day 0 vs 0 and expert vs 0 (Wilcoxon) and day 0 vs expert (Mann-Whitney). No shift null for these
per-event scores (the 026 slopes beyond the shift null per area are in 003).
Output: tables/017_area_fifths.csv, 017_area_slopes.csv; figures/017_area_fifths.{png,pdf,svg}
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
MIN_UNITS, MIN_SESS, NB = 10, 3, 5
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
    tau = (t - t.min()) / max(t.max() - t.min(), 1e-9); b = np.minimum((tau * NB).astype(int), NB - 1)
    cls = np.where(lab == "FA", "SL", lab)
    rows = []
    groups = [("whole_brain", ok)] + [(a, ok & (K.area_group == a).to_numpy()) for a in K.area_group.dropna().unique()]
    for area, m in groups:
        if m.sum() < MIN_UNITS:
            continue
        S = m002.trial_scores(X[m].T, lab, seed)["md"]
        if not np.isfinite(S).any():
            continue
        for k in range(NB):
            r = dict(meta, area=area, fifth=k + 1)
            for c in ("WH", "SL", "AH"):
                v = S[(cls == c) & (b == k)]; v = v[np.isfinite(v)]
                r[c] = v.mean() if len(v) else np.nan
            rows.append(r)
    return rows


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
    F = C.population(pd.DataFrame([x for r in res for x in r]))
    F["WH-SL"], F["WH-AH"] = F.WH - F.SL, F.WH - F.AH
    F.to_csv(C.TAB / "017_area_fifths.csv", index=False)
    return F


def rel(F, m):
    """per session x area: curve relative to its first fifth (sessions with the first fifth defined)"""
    P = F.pivot_table(index=["session_id", "cohort", "stage", "area"], columns="fifth", values=m)
    P = P.sub(P[1], axis=0)
    return P


def slopes(F):
    rows = []
    x = np.arange(1, NB + 1)
    for m, _ in MEAS:
        P = F.pivot_table(index=["session_id", "cohort", "stage", "area"], columns="fifth", values=m)
        for idx, v in P.iterrows():
            ok = np.isfinite(v.to_numpy())
            if ok.sum() >= 3:
                rows.append(dict(zip(["session_id", "cohort", "stage", "area"], idx), measure=m, slope=np.polyfit(x[ok], v.to_numpy()[ok], 1)[0]))
    return pd.DataFrame(rows)


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    F = pd.read_csv(C.TAB / "017_area_fifths.csv") if a.replot else compute(a.n_proc)
    Sl = slopes(F)
    St = []
    for (m, area, c), g in Sl.groupby(["measure", "area", "cohort"]):
        d0, e = g[g.stage == "learning"].slope, g[g.stage == "expert"].slope
        w = lambda v: stats.wilcoxon(v).pvalue if len(v) >= 5 else np.nan
        St.append(dict(measure=m, area=area, cohort=c, n_d0=len(d0), n_e=len(e), slope_d0=d0.mean(), slope_e=e.mean(), p_d0=w(d0), p_e=w(e),
                       p_stage=stats.mannwhitneyu(d0, e).pvalue if len(d0) >= 3 and len(e) >= 3 else np.nan))
    St = pd.DataFrame(St); St.to_csv(C.TAB / "017_area_slopes.csv", index=False)
    n = F.drop_duplicates(["session_id", "area"]).groupby(["area", "cohort", "stage"]).size().unstack(["cohort", "stage"]).fillna(0)
    ok_area = {c: [a_ for a_ in n.index if n.loc[a_].get((c, "learning"), 0) >= MIN_SESS and n.loc[a_].get((c, "expert"), 0) >= MIN_SESS] for c in ("R+", "R-")}
    areas = ["whole_brain"] + [g for g in au.get_area_group_custom_order() if g in set(ok_area["R+"]) | set(ok_area["R-"])]
    gcol = au.get_custom_area_groups_colors()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.5, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    cols = [(m, c) for c in ("R+", "R-") for m, _ in MEAS]
    fig, axs = plt.subplots(len(areas), len(cols), figsize=(7.6, 1.05 * len(areas) + 0.6), sharex=True,
                            gridspec_kw=dict(wspace=0.45, hspace=0.75))
    R = {m: rel(F, m) for m, _ in MEAS}
    x = np.arange(1, NB + 1)
    for i, area in enumerate(areas):
        axs[i, 0].annotate(area.replace("whole_brain", "Whole brain").replace("Somatosensory-", "Ss-").replace(" areas", ""), xy=(-0.42, 0.5),
                           xycoords="axes fraction", ha="right", va="center", fontsize=5.6, color=gcol.get(area, "k"), annotation_clip=False)
        for j, (m, c) in enumerate(cols):
            ax = axs[i, j]
            if area != "whole_brain" and area not in ok_area[c]:
                ax.axis("off"); continue
            P = R[m]
            for s_, ls, mfc, alpha in (("learning", (0, (3, 1.5)), "white", 0.7), ("expert", "-", None, 1.0)):
                try:
                    d = P.xs((c, s_, area), level=["cohort", "stage", "area"])
                except KeyError:
                    continue
                nn = d.notna().sum(); mu, se = d.mean().where(nn >= 3), d.sem().where(nn >= 3)
                ax.errorbar(x + (0.06 if s_ == "expert" else -0.06), mu, se, color=C.COH[c], ls=ls, marker="o", ms=2.0, mfc=mfc or C.COH[c],
                            lw=0.8, capsize=0, alpha=alpha, label=f"{'D0' if s_ == 'learning' else 'E'} ({len(d)})")
            ax.axhline(0, color="0.75", lw=0.4, ls=":")
            st = St[(St.measure == m) & (St.area == area) & (St.cohort == c)]
            if len(st):
                st = st.iloc[0]
                ax.text(0.02, 1.0, f"D0 {C.fp(st.p_d0)} · E {C.fp(st.p_e)} · D0≠E {C.fp(st.p_stage)}", transform=ax.transAxes, fontsize=3.6,
                        va="bottom", color="0.25")
            ax.legend(frameon=False, fontsize=3.6, loc="lower right", handlelength=1.4, borderaxespad=0.1)
            ax.tick_params(labelsize=4.2, length=1.5)
            if i == 0:
                ax.set_title(f"{c.replace('-', '−')}: {dict(MEAS)[m]}", color=C.COH[c], fontsize=5.6, pad=10)

            if i == len(areas) - 1:
                ax.set_xlabel("fifth of session", fontsize=4.8); ax.set_xticks(x)
    fig.suptitle("Reward-lick CD within the session per area group: change of the score relative to the first fifth (score units: held-out SL = 0, "
                 "AH = 1); dashed open = day 0, solid filled = expert; above each panel: per-session slope over fifths, Wilcoxon vs 0 and "
                 "day 0 vs expert (MWU)", fontsize=5.6, x=0.01, y=0.995, ha="left", wrap=True)
    fig.subplots_adjust(top=0.94, left=0.12)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"017_area_fifths.{ext}", dpi=300, bbox_inches="tight")
    pd.set_option("display.width", 220)
    print(St[St.area.isin(areas)].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n-proc", type=int, default=20); ap.add_argument("--replot", action="store_true")
    main(ap.parse_args())
