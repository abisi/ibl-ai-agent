"""089 -- COSYNE abstract figure, across days, version 3 (user 2026-10-10): two rows of five panels, projections onto the CD only
(no 0-1 scaling). 083 and 088 kept unchanged.
Row 1, single neurons
  a  first-lick-aligned population PSTHs, all tested neurons (rate minus the unit's pre-trial baseline; mean over units within
     session, mean +- s.e.m. over sessions), R+ / R- x day 0 / expert
  b  example converging non-sensory neuron (R+ expert): PSTH above raster, WH / AH / SL; shaded: pre-lick window [-100, 0) ms
  c  population PSTHs of the converging non-sensory neurons (081 classes: not significant in whisker_active / auditory_active
     nor whisker_choice; pre-lick AH > SL and WH > SL significant)
  d  fraction of tested neurons that are reward-lick neurons (pre-lick AH > SL significant, 051 ROC), per session
  e  fraction of reward-lick neurons that converge (WH > SL also significant), per session; squares: within the three stimulus
     classes of 081 (sensory / lick-gated / non-sensory)
Row 2, population, reward-lick coding direction (within-day definition: 026 folds, units z-scored over AH + SL, unit-norm
CD = mean AH - mean SL in the pre-lick window, cross-validated; per neuron, / sqrt(N))
  f  schematic slot (user's own)
  g  time course of the projection on the CD, -400 to +200 ms from the first lick (027, all stages): WH, AH (held out), SL (held out),
     mean +- s.e.m. over sessions, R+ / R- x day 0 / expert
  h  session-mean projection of whisker hits relative to spontaneous licks on the CD, d_WS = p_WH - p_SL (087)
  i  decomposition of WH - SL: angle theta to the CD and length rho (cross-validated, per neuron)
  j  per area group: change of d_WS (expert - day 0), R+ and R-; * BH q < 0.05 of the per-area cohort x stage interaction;
     title: ANOVA d_WS ~ cohort x stage x area (cohort labels permuted across mice)
Brackets (Mann-Whitney U): day 0 vs expert per cohort, R+ vs R- at day 0 and at expert; titles: cohort x stage interaction
(mouse-level permutation, 10,000). Learners rule; unit = session.
Output: across_days/<ref>[_units]/publication/learners/COSYNE_across_days_v3.{png,pdf,svg} + _stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../089_cosyne_across_days_v3.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m88 = importlib.import_module("088_cosyne_across_days_v2")
m51, m61, m62, m81, m83 = m88.m51, m88.m61, m88.m62, m88.m81, m88.m83
COH, CL, GROUPS, GLAB = m88.COH, m88.CL, m88.GROUPS, m88.GLAB
PROJ_T = m51.WITHIN / "cd_psth" / "cd_psth_projection_allstages.parquet"


def proj_insets(fig, spec, T, rows):
    sub = spec.subgridspec(2, 2, hspace=0.35, wspace=0.12)
    axs = []
    for j, g in enumerate(GROUPS):
        ax = fig.add_subplot(sub[j // 2, j % 2]); axs.append(ax)
        d = T[(T.cohort == g[0]) & (T.stage == g[1])]
        for cl, key in (("SL", "FA"), ("AH", "AH"), ("WH", "WH")):
            piv = d[d.cls == cl].pivot_table(index="session_id", columns="t", values="value")
            x = piv.columns.to_numpy() * 1e3; m, se = piv.mean(0).to_numpy(), piv.sem(0).to_numpy()
            ax.fill_between(x, m - se, m + se, color=CL[key], alpha=0.25, lw=0); ax.plot(x, m, color=CL[key], lw=0.9)
        ax.axvspan(-100, 0, color="#FDD49E", alpha=0.5, lw=0, zorder=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
        ax.axhline(0, color="0.85", lw=0.4)
        ax.set_xlim(-400, 200); ax.set_xticks([-300, 0]); ax.set_xticklabels(["−300", "0"] if j >= 2 else [])
        n = d.session_id.nunique()
        ax.set_title(f"{GLAB[g]} ({n})", color=COH[g[0]], fontsize=4.8, pad=1.5)
        rows.append(dict(panel="g", group=GLAB[g], n_sessions=n))
    yl = (min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs))
    for j, a in enumerate(axs):
        a.set_ylim(*yl); a.tick_params(labelsize=4.4)
        if j % 2:
            a.set_yticklabels([])
    axs[0].set_ylabel("proj. on CD", fontsize=4.8); axs[2].set_ylabel("proj. on CD", fontsize=4.8)
    return axs


def main():
    plt = m62.setup()
    rng = np.random.default_rng(0)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda D_, c: D_[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U0["rl"] = sig(U0, f"sig:{m81.SEL_AS}") & (U0[f"sel:{m81.SEL_AS}"] > 0)
    U0["conv"] = U0.rl & sig(U0, f"sig:{m81.SEL_WS}") & (U0[f"sel:{m81.SEL_WS}"] > 0)
    for k in ("electrode_group", "cluster_id"):
        U0[k] = U0[k].astype(str)
    C0 = pd.read_csv(m81.OUT / "classes.csv")
    D = m62.load_population("learners")
    W = D["W"].copy()
    for k in ("electrode_group", "cluster_id"):
        W[k] = W[k].astype(str)
    PB, KB, tc = m62.load_psth_prestart(W)
    KB = KB.assign(electrode_group=KB.electrode_group.astype(str), cluster_id=KB.cluster_id.astype(str))
    W = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    Wc = W.merge(U0[["session_id", "electrode_group", "cluster_id", "nclass", "rl", "conv"]], on=["session_id", "electrode_group", "cluster_id"], how="left")
    Wc["mi"] = np.nan
    U = m61.learner_filter(U0); C = m61.learner_filter(C0)
    Fr = U.groupby(["session_id", "mouse_id", "cohort", "stage"]).agg(frac_rl=("rl", "mean"), n=("rl", "size")).reset_index()
    Fr = Fr[Fr.n >= 10]
    S = U[U.rl & U.nclass.isin(m81.CLASSES)].groupby(["session_id", "mouse_id", "cohort", "stage"]).agg(
        conv=("conv", "mean"), n=("conv", "size")).reset_index()
    S = S[S.n >= m81.MIN_CLASS]
    PA = m61.learner_filter(pd.read_csv(m88.H087)); P = PA[PA.area == "whole_brain"]
    T = m61.learner_filter(pd.read_parquet(PROJ_T)); T = T[T.session_id.isin(P.session_id)]
    rows = []
    with plt.rc_context({"font.size": 5.4, "axes.titlesize": 5.6, "axes.labelsize": 5.3, "xtick.labelsize": 4.9,
                         "ytick.labelsize": 4.9, "legend.fontsize": 4.8}):
        fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.58))
        gs = fig.add_gridspec(2, 5, hspace=0.85, wspace=0.55, left=0.05, right=0.985, top=0.83, bottom=0.1)

        def cell_title(spec, text, dy=0.04):
            bb = spec.get_position(fig)
            fig.text(bb.x0 + 0.03, bb.y1 + dy, text, fontsize=5.6, va="bottom")
        L = []
        # ---------------- row 1
        tested = (Wc["sig:auditory_hit_vs_fa_prelick@all"].notna() & Wc["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
        ia = m83.psth_insets(fig, gs[0, 0], Wc, PB, tc, tested, "a", rows); L.append(ia[0]); cell_title(gs[0, 0], "All neurons")
        ex = [e for e in m88.pick_examples(Wc) if e[0].startswith("Converging, non-sensory")]
        if ex:
            ttl, r = ex[0]
            ap, ar = m62.raster_psth(fig, gs[0, 1], r, rng, f"Example converging, non-sensory\n{r.get('area_acronym_custom', '')}, R+ expert", "0.15")
            ap.set_ylabel("Hz", fontsize=4.8); ar.set_xlabel("time from first lick (ms)", fontsize=4.8); L.append(ap)
            rows.append(dict(panel="b", session_id=r.session_id, electrode_group=r.electrode_group, cluster_id=r.cluster_id))
        ic = m83.psth_insets(fig, gs[0, 2], Wc, PB, tc, (Wc.conv.fillna(False) & (Wc.nclass == "non-sensory")).to_numpy(), "c", rows)
        L.append(ic[0]); cell_title(gs[0, 2], "Converging, non-sensory neurons")
        ax = fig.add_subplot(gs[0, 3]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, Fr, "frac_rl", "Fraction of tested neurons", rng, "089 frac_rl", "Reward-lick neurons (AH > SL)")
        ax = fig.add_subplot(gs[0, 4]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, S, "conv", "Fraction of reward-lick neurons", rng, "089 conv", "Converging (WH > SL too)")
        for cl, dx in (("sensory", -0.18), ("lick-gated", 0.0), ("non-sensory", 0.18)):
            d = C[C.nclass == cl].dropna(subset=["conv"])
            for (c, s_), x in zip(GROUPS, (0, 1, 2.4, 3.4)):
                v = d[(d.cohort == c) & (d.stage == s_)]["conv"]
                ax.plot(x + 0.28 + dx * 0.5, v.mean(), marker="s", ms=2.0, color=m88.CCOL[cl], ls="", alpha=0.9,
                        label=cl if (c, s_) == GROUPS[0] else None)
        ax.legend(frameon=False, fontsize=4.0, loc="center", bbox_to_anchor=(0.6, 0.42), handlelength=0.8, borderaxespad=0.1, title="neuron class", title_fontsize=4.0)
        # ---------------- row 2
        fr_ = m62.frame if hasattr(m62, "frame") else None
        axf = fig.add_subplot(gs[1, 0]); axf.set_xticks([]); axf.set_yticks([]); L.append(axf)
        for sp in axf.spines.values():
            sp.set_visible(True); sp.set_color("0.85")
        axf.text(0.5, 0.5, "schematic: CD = AH − SL\n(100 ms before the lick)", ha="center", va="center", fontsize=5, color="0.6", transform=axf.transAxes)
        ig = proj_insets(fig, gs[1, 1], T, rows); L.append(ig[0]); cell_title(gs[1, 1], "Projection on the reward-lick CD")
        ax = fig.add_subplot(gs[1, 2]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, P, "d_WS", "WH − SL projection on CD\n(per neuron)", rng, "089 d_WS", "WH move toward AH", ref=0)
        gi = gs[1, 3].subgridspec(1, 2, wspace=1.1)
        ai = [fig.add_subplot(gi[i]) for i in range(2)]
        m88.small_panel(ai[0], P, "theta", "θ (°)", rng, "Angle θ")
        m88.small_panel(ai[1], P, "rho_v", "ρ (per neuron)", rng, "Length ρ")
        L.append(ai[0]); cell_title(gs[1, 3], "Decomposition of WH − SL", dy=0.085)
        ax = fig.add_subplot(gs[1, 4]); ax.set_box_aspect(1); L.append(ax)
        m88.area_panel(ax, PA, rows, rng)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color=CL[k_], lw=1.4, marker="o", ms=2.6) for k_ in ("WH", "AH", "FA")],
                   ["Whisker hit", "Auditory hit", "Spontaneous lick"], frameon=False, fontsize=5.2, ncol=3, loc="upper left",
                   bbox_to_anchor=(0.05, 0.965), handlelength=1.4)
        m62.letter_row(fig, L[:5], "abcde", dx_in=0.3, dy_in=0.22)
        m62.letter_row(fig, L[5:], "fghij", dx_in=0.3, dy_in=0.22)
        fig.suptitle("Reward recruits neurons across the brain into a stimulus-independent reward-lick code", x=0.02, y=0.995,
                     ha="left", va="top", fontsize=7, weight="bold")
        out = m51.OUTROOT / "publication" / "learners"
        m62.save(fig, out, "COSYNE_across_days_v3"); plt.close(fig)
    R = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith("089")])
    R.to_csv(out / "COSYNE_across_days_v3_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("089") and not str(r.get("measure", "")).startswith("area")]).round(4).to_string())
    print("ALL DONE", out / "COSYNE_across_days_v3.png")


if __name__ == "__main__":
    main()
