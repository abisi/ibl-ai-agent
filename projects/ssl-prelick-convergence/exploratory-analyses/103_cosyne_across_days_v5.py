"""103 -- COSYNE abstract figure, across days, version 5 (user 2026-10-11): 096 without the single-cell / population correlation, with
a per-area single-cell panel, the population on the 087 CD with fold-wise z-scoring (as within-day 026), and --pop learners | all.
083 / 088 / 089 / 096 kept unchanged.
Row 1, single neurons
  a  example converging non-sensory neuron (R+ expert): PSTH above raster
  b  time-resolved selectivity of reward-lick neurons (095: selected on AH vs SL-A, measured against the held-out SL-B)
  c  WH vs SL-B selectivity in [-100, 0) ms, per session
  d  generalisation ratio (WH vs SL-B) / (AH vs SL-B) in [-100, 0) ms
  e  per area group: change (expert - day 0) of the converging fraction among reward-lick neurons (081; sessions with >= 3 reward-lick
     neurons in the area), R+ and R- (mean +- combined s.e.m.); * BH q < 0.05 of the per-area cohort x stage interaction (mouse
     permutation), (*) p < 0.05
Row 2, population on the reward-lick CD (087, fold-wise z-scoring; 027 time courses): f schematic slot, g projection time courses,
h WH - SL projection, i angle / length decomposition, j per-area change of the WH - SL projection.
Population: --pop learners (day-0 sessions of non-learner mice dropped) or all.
Output: across_days/<ref>[_units]/publication/<pop>/COSYNE_across_days_v5.{png,pdf,svg} + _stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../103_cosyne_across_days_v5.py --pop learners
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m96 = importlib.import_module("096_cosyne_across_days_v4")
m89, m93, m88 = m96.m89, m96.m93, m96.m88
m51, m61, m62, m81, m83 = m88.m51, m88.m61, m88.m62, m88.m81, m88.m83
COH, CL, GROUPS = m88.COH, m88.CL, m88.GROUPS


def bh(p):
    p = np.asarray(p, float); q = np.full(len(p), np.nan); ok = np.isfinite(p)
    if ok.sum():
        v = p[ok]; o = np.argsort(v); r = v[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty_like(r); out[o] = np.minimum(r, 1); q[ok] = out
    return q


def sc_area_panel(ax, U, rows, rng):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    S = U[U.rl].groupby(["session_id", "mouse_id", "cohort", "stage", "area_group"]).agg(conv=("conv", "mean"), n=("conv", "size")).reset_index()
    S = S[S.n >= 3]
    n = S.groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack([1, 2]).fillna(0)
    order = [g for g in au.get_area_group_custom_order() if g in n.index and all(n.loc[g].get(k, 0) >= 3 for k in GROUPS)]
    y = np.arange(len(order))[::-1]; pv = []
    for i, g in enumerate(order):
        d = S[S.area_group == g]
        for k, c in enumerate(("R+", "R-")):
            a, b = (d[(d.cohort == c) & (d.stage == s_)].conv for s_ in ("learning", "expert"))
            ax.errorbar(b.mean() - a.mean(), y[i] + 0.16 - 0.32 * k, xerr=np.sqrt(a.sem() ** 2 + b.sem() ** 2), fmt="o", ms=2.8,
                        color=COH[c], lw=0.8, capsize=0, mfc="white")
            rows.append(dict(panel="e", area=g, cohort=c, change=b.mean() - a.mean(), n_day0=len(a), n_expert=len(b)))
        pv.append(m62.group_stats(d, "conv", f"area {g}", rng, "103 e")["interaction"][1])
    qv = bh(pv)
    for i, (g, p_, q_) in enumerate(zip(order, pv, qv)):
        rows.append(dict(panel="e q", area=g, p_perm=p_, q_BH=q_))
        mk = "*" if q_ < 0.05 else ("(*)" if p_ < 0.05 else "")
        if mk:
            ax.text(0.99, y[i], mk, transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", ha="right", weight="bold")
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y, [m83.m62_short(g) for g in order], fontsize=4.6)
    for tl, g in zip(ax.get_yticklabels(), order):
        tl.set_color(gcol.get(g, "k"))
    ax.set_xlabel("Δ converging fraction\n(expert − day 0)", fontsize=5)
    ax.set_title("Per area: converging neurons\n* c×s q < 0.05, (*) p < 0.05", fontsize=4.8, pad=2)


def main(a):
    learn = a.pop == "learners"
    flt = m61.learner_filter if learn else (lambda d: d)
    plt = m62.setup(); rng = np.random.default_rng(0)
    D = pd.read_parquet(m96.SPLIT / "selectivity_index.parquet"); S = np.load(m96.SPLIT / "selectivity.npy"); t = np.load(m96.SPLIT / "t.npy")
    keep = D.index.isin(flt(D).index); D, S = D[keep].reset_index(drop=True), S[keep]
    R = m96.session_ratio(D, S, t)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda D_, c: D_[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U0["rl"] = sig(U0, f"sig:{m81.SEL_AS}") & (U0[f"sel:{m81.SEL_AS}"] > 0)
    U0["conv"] = U0.rl & sig(U0, f"sig:{m81.SEL_WS}") & (U0[f"sel:{m81.SEL_WS}"] > 0)
    for k in ("electrode_group", "cluster_id"):
        U0[k] = U0[k].astype(str)
    W = m62.load_population(a.pop)["W"].copy()
    for k in ("electrode_group", "cluster_id"):
        W[k] = W[k].astype(str)
    Wc = W.merge(U0[["session_id", "electrode_group", "cluster_id", "nclass", "rl", "conv"]], on=["session_id", "electrode_group", "cluster_id"], how="left")
    Wc["mi"] = np.nan
    PA = flt(pd.read_csv(m88.H087)); P = PA[PA.area == "whole_brain"]
    T = flt(pd.read_parquet(m89.PROJ_T)); T = T[T.session_id.isin(P.session_id)]
    rows = []
    with plt.rc_context({"font.size": 5.4, "axes.titlesize": 5.6, "axes.labelsize": 5.3, "xtick.labelsize": 4.9, "ytick.labelsize": 4.9, "legend.fontsize": 4.8}):
        fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.6))
        gs = fig.add_gridspec(2, 5, hspace=0.9, wspace=0.55, left=0.05, right=0.975, top=0.82, bottom=0.1)

        def cell_title(spec, text, dy=0.04):
            bb = spec.get_position(fig); fig.text(bb.x0 + 0.03, bb.y1 + dy, text, fontsize=5.6, va="bottom")
        L = []
        ex = [e for e in m88.pick_examples(Wc) if e[0].startswith("Converging, non-sensory")]
        if ex:
            ttl, r = ex[0]
            ap, ar = m62.raster_psth(fig, gs[0, 0], r, rng, f"Example converging, non-sensory\n{r.get('area_acronym_custom', '')}, R+ expert", "0.15")
            ap.set_ylabel("Hz", fontsize=4.8); ar.set_xlabel("time from first lick (ms)", fontsize=4.8); L.append(ap)
        ib = m93.insets(fig, gs[0, 1], D, S, t, np.ones(len(D), bool), "b"); L.append(ib[0]); cell_title(gs[0, 1], "Reward-lick neurons over time")
        ax = fig.add_subplot(gs[0, 2]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, R, "WS", "WH vs SL selectivity\n(−100 to 0 ms)", rng, "103 WS", "Whisker hits vs spont. licks", ref=0)
        ax = fig.add_subplot(gs[0, 3]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, R, "ratio", "(WH vs SL) / (AH vs SL)", rng, "103 ratio", "Generalisation to whisker hits", ref=0)
        ax = fig.add_subplot(gs[0, 4]); ax.set_box_aspect(1); L.append(ax)
        sc_area_panel(ax, flt(U0), rows, rng)
        axf = fig.add_subplot(gs[1, 0]); axf.set_xticks([]); axf.set_yticks([]); L.append(axf)
        for sp in axf.spines.values():
            sp.set_visible(True); sp.set_color("0.85")
        axf.text(0.5, 0.5, "schematic: CD = AH − SL\n(100 ms before the lick)", ha="center", va="center", fontsize=5, color="0.6", transform=axf.transAxes)
        ig = m89.proj_insets(fig, gs[1, 1], T, rows); L.append(ig[0]); cell_title(gs[1, 1], "Projection on the reward-lick CD")
        ax = fig.add_subplot(gs[1, 2]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, P, "d_WS", "WH − SL projection on CD\n(per neuron)", rng, "103 d_WS", "WH move toward AH", ref=0)
        gi = gs[1, 3].subgridspec(1, 2, wspace=1.1); ai = [fig.add_subplot(gi[i]) for i in range(2)]
        m88.small_panel(ai[0], P, "theta", "θ (°)", rng, "Angle θ"); m88.small_panel(ai[1], P, "rho_v", "ρ (per neuron)", rng, "Length ρ")
        L.append(ai[0]); cell_title(gs[1, 3], "Decomposition of WH − SL", dy=0.085)
        ax = fig.add_subplot(gs[1, 4]); ax.set_box_aspect(1); L.append(ax)
        m88.area_panel(ax, PA, rows, rng)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color=CL[k_], lw=1.4, marker="o", ms=2.6) for k_ in ("WH", "AH", "FA")],
                   ["Whisker hit", "Auditory hit", "Spontaneous lick"], frameon=False, fontsize=5.2, ncol=3, loc="upper left", bbox_to_anchor=(0.05, 0.965), handlelength=1.4)
        m62.letter_row(fig, L[:5], "abcde", dx_in=0.3, dy_in=0.22); m62.letter_row(fig, L[5:], "fghij", dx_in=0.3, dy_in=0.22)
        fig.suptitle(f"Reward recruits neurons across the brain into a stimulus-independent reward-lick code ({a.pop})", x=0.02, y=0.995,
                     ha="left", va="top", fontsize=7, weight="bold")
        out = m51.OUTROOT / "publication" / a.pop
        m62.save(fig, out, "COSYNE_across_days_v5"); plt.close(fig)
    St = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith(("103", "088 theta", "088 rho_v"))])
    St.to_csv(out / "COSYNE_across_days_v5_stats.csv", index=False)
    pd.set_option("display.width", 250)
    s = St[St.panel.astype(str).str.startswith("103 ") & ~St.panel.astype(str).str.contains("e")]
    cols = [c for c in ["panel", "mean_R+ learning", "mean_R+ expert", "mean_R− learning", "mean_R− expert", "n_R+ learning", "n_R+ expert", "n_R− learning",
                        "n_R− expert", "interaction_p_perm", "R+ L vs E p_MWU", "learning R+ vs R- p_MWU", "expert R+ vs R- p_MWU"] if c in St]
    print(St[St.panel.astype(str).isin(["103 WS", "103 ratio", "103 d_WS", "088 theta", "088 rho_v"])][cols].round(4).to_string())
    print(St[St.panel == "e q"][["area", "p_perm", "q_BH"]].round(3).to_string(index=False))
    print("ALL DONE", out / "COSYNE_across_days_v5.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", default="learners", choices=["learners", "all"])
    main(ap.parse_args())
