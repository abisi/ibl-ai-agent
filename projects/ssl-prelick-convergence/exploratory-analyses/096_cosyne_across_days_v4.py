"""096 -- COSYNE abstract figure, across days, version 4 (user 2026-10-10): single neurons from the split-SL time-resolved ROC (095)
and the population geometry of v3 (089), linked by a session-level correlation. 083 / 088 / 089 kept unchanged.
Row 1, single neurons (reward-lick neurons selected on AH vs SL-A, selectivity against the held-out SL-B; 095)
  a  example converging non-sensory neuron (R+ expert): PSTH above raster
  b  time-resolved selectivity (50-ms windows, 5-ms stride): AH vs SL-B, WH vs SL-B, WH vs AH; R+ / R- x day 0 / expert
  c  WH vs SL-B selectivity in [-100, 0) ms, per session
  d  generalisation ratio = (WH vs SL-B) / (AH vs SL-B) in [-100, 0) ms, per session
  e  link: per session, generalisation ratio (d) vs population WH - SL projection on the CD (h); OLS line with 95 % CI (solid if
     p < 0.05), Spearman and Pearson r pooled over all sessions and within cohort
Row 2, population on the reward-lick CD (as 089 f-j)
Output: across_days/<ref>[_units]/publication/learners/COSYNE_across_days_v4.{png,pdf,svg} + _stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../096_cosyne_across_days_v4.py
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m89 = importlib.import_module("089_cosyne_across_days_v3")
m93 = importlib.import_module("093_timeresolved_roc_figure")
m88, m51, m61, m62, m81, m83 = m89.m88, m89.m51, m89.m61, m89.m62, m89.m81, m89.m83
COH, CL, GROUPS, GLAB = m89.COH, m89.CL, m89.GROUPS, m89.GLAB
SPLIT = m51.OUTROOT / "timeresolved_roc" / "split_sl"


def session_ratio(D, S, t):
    rows = []
    for pr in ("WS", "AS"):
        mm = (D.pair == pr).to_numpy()
        rows.append(D[mm][["session_id", "mouse_id", "cohort", "stage"]].assign(pair=pr, v=S[mm][:, (t >= -0.1) & (t < 0)].mean(1)))
    R = pd.concat(rows).groupby(["session_id", "mouse_id", "cohort", "stage", "pair"]).v.mean().unstack("pair").reset_index()
    R["ratio"] = R.WS / R.AS
    return R


def corr_panel(ax, M, x, y, xl, yl, rows):
    import statsmodels.formula.api as smf
    d = M.dropna(subset=[x, y])
    for (c, s_) in GROUPS:
        g = d[(d.cohort == c) & (d.stage == s_)]
        ax.plot(g[x], g[y], "o", ms=3, color=COH[c], mfc="white" if s_ == "learning" else COH[c], mew=0.7, alpha=0.85, ls="")
    m = smf.ols(f"{y} ~ {x}", d).fit()
    xx = np.linspace(d[x].min(), d[x].max(), 50); pr = m.get_prediction(pd.DataFrame({x: xx})).summary_frame(alpha=0.05)
    rp, pp = stats.pearsonr(d[x], d[y]); rs, ps = stats.spearmanr(d[x], d[y])
    ax.fill_between(xx, pr.mean_ci_lower, pr.mean_ci_upper, color="0.5", alpha=0.2, lw=0)
    ax.plot(xx, pr["mean"], color="0.2", lw=1.0, ls="-" if pp < 0.05 else "--")
    within = {c: stats.spearmanr(*d[d.cohort == c][[x, y]].to_numpy().T) for c in ("R+", "R-")}
    rows.append(dict(panel="e", n=len(d), pearson_r=rp, pearson_p=pp, spearman_r=rs, spearman_p=ps,
                     **{f"spearman_r_{c}": w[0] for c, w in within.items()}, **{f"spearman_p_{c}": w[1] for c, w in within.items()}))
    ax.axhline(0, color="0.85", lw=0.4); ax.axvline(0, color="0.85", lw=0.4)
    ax.set_xlabel(xl, fontsize=5); ax.set_ylabel(yl)
    ax.set_title(f"Single cells ↔ population ({len(d)} sessions)\nSpearman r = {rs:.2f}, {m62.fmt_p(ps)}\n"
                 f"within R+ {within['R+'][0]:.2f}, R− {within['R-'][0]:.2f}", fontsize=4.8, pad=3)


def main():
    plt = m62.setup(); rng = np.random.default_rng(0)
    # single-cell split-SL selectivity
    D = pd.read_parquet(SPLIT / "selectivity_index.parquet"); S = np.load(SPLIT / "selectivity.npy"); t = np.load(SPLIT / "t.npy")
    keep = D.index.isin(m61.learner_filter(D).index); D, S = D[keep].reset_index(drop=True), S[keep]
    R = session_ratio(D, S, t)
    # example neuron
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda D_, c: D_[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U0["rl"] = sig(U0, f"sig:{m81.SEL_AS}") & (U0[f"sel:{m81.SEL_AS}"] > 0)
    U0["conv"] = U0.rl & sig(U0, f"sig:{m81.SEL_WS}") & (U0[f"sel:{m81.SEL_WS}"] > 0)
    for k in ("electrode_group", "cluster_id"):
        U0[k] = U0[k].astype(str)
    W = m62.load_population("learners")["W"].copy()
    for k in ("electrode_group", "cluster_id"):
        W[k] = W[k].astype(str)
    Wc = W.merge(U0[["session_id", "electrode_group", "cluster_id", "nclass", "rl", "conv"]], on=["session_id", "electrode_group", "cluster_id"], how="left")
    Wc["mi"] = np.nan
    # population
    PA = m61.learner_filter(pd.read_csv(m88.H087)); P = PA[PA.area == "whole_brain"]
    T = m61.learner_filter(pd.read_parquet(m89.PROJ_T)); T = T[T.session_id.isin(P.session_id)]
    M = R.merge(P[["session_id", "d_WS"]], on="session_id", how="inner")
    rows = []
    with plt.rc_context({"font.size": 5.4, "axes.titlesize": 5.6, "axes.labelsize": 5.3, "xtick.labelsize": 4.9,
                         "ytick.labelsize": 4.9, "legend.fontsize": 4.8}):
        fig = plt.figure(figsize=(m62.W_IN * 1.2, m62.W_IN * 0.6))
        gs = fig.add_gridspec(2, 5, hspace=0.9, wspace=0.55, left=0.05, right=0.975, top=0.82, bottom=0.1)

        def cell_title(spec, text, dy=0.04):
            bb = spec.get_position(fig)
            fig.text(bb.x0 + 0.03, bb.y1 + dy, text, fontsize=5.6, va="bottom")
        L = []
        ex = [e for e in m88.pick_examples(Wc) if e[0].startswith("Converging, non-sensory")]
        if ex:
            ttl, r = ex[0]
            ap, ar = m62.raster_psth(fig, gs[0, 0], r, rng, f"Example converging, non-sensory\n{r.get('area_acronym_custom', '')}, R+ expert", "0.15")
            ap.set_ylabel("Hz", fontsize=4.8); ar.set_xlabel("time from first lick (ms)", fontsize=4.8); L.append(ap)
        ib = m93.insets(fig, gs[0, 1], D, S, t, np.ones(len(D), bool), "b"); L.append(ib[0])
        cell_title(gs[0, 1], "Reward-lick neurons over time")
        ax = fig.add_subplot(gs[0, 2]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, R, "WS", "WH vs SL selectivity\n(−100 to 0 ms)", rng, "096 WS", "Whisker hits vs spont. licks", ref=0)
        ax = fig.add_subplot(gs[0, 3]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, R, "ratio", "(WH vs SL) / (AH vs SL)", rng, "096 ratio", "Generalisation to whisker hits", ref=0)
        ax = fig.add_subplot(gs[0, 4]); ax.set_box_aspect(1); L.append(ax)
        corr_panel(ax, M, "ratio", "d_WS", "single-cell generalisation ratio", "population WH − SL\nprojection on CD (per neuron)", rows)
        # row 2 (as 089)
        axf = fig.add_subplot(gs[1, 0]); axf.set_xticks([]); axf.set_yticks([]); L.append(axf)
        for sp in axf.spines.values():
            sp.set_visible(True); sp.set_color("0.85")
        axf.text(0.5, 0.5, "schematic: CD = AH − SL\n(100 ms before the lick)", ha="center", va="center", fontsize=5, color="0.6", transform=axf.transAxes)
        ig = m89.proj_insets(fig, gs[1, 1], T, rows); L.append(ig[0]); cell_title(gs[1, 1], "Projection on the reward-lick CD")
        ax = fig.add_subplot(gs[1, 2]); ax.set_box_aspect(1); L.append(ax)
        m88.mean_panel2(ax, P, "d_WS", "WH − SL projection on CD\n(per neuron)", rng, "096 d_WS", "WH move toward AH", ref=0)
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
        m62.save(fig, out, "COSYNE_across_days_v4"); plt.close(fig)
    St = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith("096")])
    St.to_csv(out / "COSYNE_across_days_v4_stats.csv", index=False)
    print(pd.DataFrame([r for r in rows if r.get("panel") == "e"]).round(4).to_string())
    print("ALL DONE", out / "COSYNE_across_days_v4.png")


if __name__ == "__main__":
    main()
