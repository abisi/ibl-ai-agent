"""088 -- COSYNE abstract figure, across days, version 2 (user 2026-10-09): 083 updated to the current definition of the reward-lick
coding direction (087: the within-day 026 CD across days), with example neurons, a modulation index instead of AH-likeness, and
day-0 R+ vs R- comparisons. 083 and its outputs are kept unchanged.
Row 1  a  example neurons (raster + PSTH aligned to the corrected first lick; WH yellow, AH blue, SL black; shaded: pre-lick window):
          converging non-sensory neuron (R+ expert), converging neuron (R+ day 0), reward-lick neuron whose whisker-hit response is
          auditory-hit-like (MI near +1, R+ expert) and one whose whisker-hit response is spontaneous-lick-like (MI near -1, R- expert)
Row 2  b  first-lick-aligned population PSTHs, all tested neurons (as 083 a)
       c  converging non-sensory neurons (as 083 b)
       d  converging fraction among reward-lick (AH > SL) neurons per session (as 083 c); squares: the same fraction within the three
          stimulus classes of 081 (sensory: significant in whisker_active or auditory_active; lick-gated: not sensory, whisker_choice
          significant; non-sensory: neither)
       e  whisker-hit modulation index MI of reward-lick neurons, per-session mean (replaces AH-likeness; no class squares):
              MI = (|m_WH - m_SL| - |m_WH - m_AH|) / (|m_WH - m_SL| + |m_WH - m_AH|),  in [-1, 1]
          m = the neuron's mean baseline-corrected pre-lick rate per event class ([-100, 0) ms before the first lick, 051);
          MI = +1: whisker hits evoke the auditory-hit response; -1: they evoke the spontaneous-lick response; 0: halfway
Row 3 (current CD definition, 087: 026 folds, units z-scored over AH + SL, unit-norm axis, per neuron)
       f  session-mean projections of the three event classes on the CD (per neuron; 0 = mean of the AH + SL events), mean +- s.e.m.
          over sessions; day 0 open, expert filled (2026-10-09: replaces the geometry plane and the 0-1 normalised lambda)
       g  whisker hits relative to spontaneous licks on the CD, d_WS = p_WH - p_SL (per neuron), session means
       h  angle theta between WH - SL and the CD, and length rho of WH - SL (cross-validated, per neuron)
       i  per area group: change of d_WS (expert - day 0, learners rule), R+ and R-; * = BH q < 0.05 of the per-area cohort x stage interaction (mouse
          permutation); title: ANOVA d_WS ~ cohort x stage x area (cohort labels permuted across mice)
Brackets (Mann-Whitney U): day 0 vs expert within cohort; R+ vs R- at day 0; R+ vs R- at expert. Titles: cohort x stage interaction
(mouse-level permutation). Learners rule (day-0 sessions of non-learners dropped); unit = session.
Output: across_days/<ref>[_units]/publication/learners/COSYNE_across_days_v2.{png,pdf,svg} + COSYNE_across_days_v2_stats.csv
Run (haas): PRELICK_REF=sl PRELICK_UNITS=stable python .../088_cosyne_across_days_v2.py [--n-proc 30]
"""
import argparse
import importlib
import multiprocessing as mp
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m81 = importlib.import_module("081_single_unit_classes")
m83 = importlib.import_module("083_cosyne_across_days")
COH, CL, GROUPS, GLAB = m62.COH, m62.CL, m83.GROUPS, m83.GLAB
CCOL = m83.CCOL
H087 = m51.OUTROOT / "harmonised" / "sessions_areas.csv"
XS = {GROUPS[0]: 0.0, GROUPS[1]: 1.0, GROUPS[2]: 2.4, GROUPS[3]: 3.4}


# ------------------------------------------------------------------ modulation index
def session_mi(args):
    f, sid = args
    z = np.load(f, allow_pickle=True)
    X, lab = z["rates"].astype(float), z["cls"]
    m = {c: X[:, lab == c].mean(1) if (lab == c).any() else np.full(len(X), np.nan) for c in ("WH", "AH", "FA")}
    a, b = np.abs(m["WH"] - m["FA"]), np.abs(m["WH"] - m["AH"])
    mi = np.where(a + b > 0, (a - b) / np.where(a + b > 0, a + b, 1), np.nan)
    return pd.DataFrame(dict(session_id=sid, electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str),
                             mi=mi, m_WH=m["WH"], m_AH=m["AH"], m_SL=m["FA"]))


def all_mi(n_proc):
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions()
    jobs = [(r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", r.session_id) for r in ss.itertuples()]
    jobs = [j for j in jobs if j[0].exists()]
    with mp.get_context("fork").Pool(n_proc) as pool:
        return pd.concat(pool.map(session_mi, jobs), ignore_index=True)


# ------------------------------------------------------------------ panels
def mean_panel2(ax, d, col, ylabel, rng, panel, title, ref=None):
    """062 mean_panel plus the day-0 R+ vs R- bracket"""
    S = m62.group_stats(d, col, title, rng, panel)
    for c, (x0, x1) in {"R+": (0.0, 1.0), "R-": (2.4, 3.4)}.items():
        mm = [S["means"][(c, s)] for s in ("learning", "expert")]; ee = [S["sems"][(c, s)] for s in ("learning", "expert")]
        ax.plot([x0, x1], mm, color=COH[c], lw=1.1, zorder=2)
        for x, m_, e_, s in zip((x0, x1), mm, ee, ("learning", "expert")):
            ax.errorbar(x, m_, e_, fmt="o", ms=4.0, color=COH[c], mfc="white" if s == "learning" else COH[c], mew=1.0, lw=1.1,
                        capsize=0, zorder=3)
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.6, ls=(0, (2, 2)), zorder=1)
    lo, hi = ax.get_ylim(); r_ = hi - lo
    for name, (x0, x1), c, yb in [("R+ L vs E", (0, 1), COH["R+"], hi + 0.08 * r_), ("R- L vs E", (2.4, 3.4), COH["R-"], hi + 0.08 * r_),
                                  ("learning R+ vs R-", (0, 2.4), "0.45", hi + 0.3 * r_), ("expert R+ vs R-", (1, 3.4), "0.15", hi + 0.52 * r_)]:
        if name in S:
            ax.plot([x0, x0, x1, x1], [yb - 0.03 * r_, yb, yb, yb - 0.03 * r_], color=c, lw=0.6)
            lab = m62.fmt_p(S[name][0]) + (" (D0)" if name.startswith("learning") else " (E)" if name.startswith("expert") else "")
            ax.text((x0 + x1) / 2, yb + 0.01 * r_, lab, ha="center", va="bottom", fontsize=4.5, color=c)
    ax.set_ylim(lo, hi + 0.72 * r_); ax.set_xlim(-0.5, 3.9)
    ax.set_xticks([0, 1, 2.4, 3.4], [f"{l}\n({S['n'][g]})" for l, g in zip(["D0", "E", "D0", "E"], GROUPS)], fontsize=4.6)   # n sessions
    for x, c in [(0.5, "R+"), (2.9, "R-")]:
        ax.annotate(c.replace("-", "−"), xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -17), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color=COH[c], weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\ncohort × stage {m62.fmt_p(S['interaction'][1])}", fontsize=5.2, pad=3)
    return S


def triangles(axs, P):
    lim = 0
    for ax, c in zip(axs, ("R+", "R-")):
        for s, ls, mfc in (("learning", (0, (2, 1.5)), "white"), ("expert", "-", None)):
            d = P[(P.cohort == c) & (P.stage == s)]
            dAS, dWS, rho = d.d_AS.mean(), d.d_WS.mean(), d.rho_v.mean()
            wh = np.r_[dWS, np.sqrt(max(rho ** 2 - dWS ** 2, 0))]
            pts = {"FA": np.r_[0, 0], "AH": np.r_[dAS, 0], "WH": wh}
            tri = np.vstack([pts["FA"], pts["AH"], pts["WH"], pts["FA"]])
            ax.plot(*tri.T, color=COH[c], lw=0.9, ls=ls)
            for k, p in pts.items():
                ax.plot(*p, "o", ms=3.2, color=CL[k], mfc=mfc or CL[k], mew=0.8, zorder=4)
            lim = max(lim, dAS, wh[1], abs(dWS))
        ax.axhline(0, color="0.85", lw=0.4)
        ax.set_title(f"{c.replace('-', '−')}: day 0 (open, dashed) → expert", color=COH[c], fontsize=4.6, pad=1.5)
        ax.set_ylabel("orthogonal (per neuron)", fontsize=4.4); ax.tick_params(labelsize=4.2)
    for ax in axs:
        ax.set_xlim(-0.08 * lim, 1.15 * lim); ax.set_ylim(-0.08 * lim, 1.2 * lim); ax.set_aspect("equal", adjustable="box")
    axs[-1].set_xlabel("along the CD, SL → AH (per neuron)", fontsize=4.4)


def proj_levels(ax, P):
    """session-mean projection of each class on the CD, per cohort x stage"""
    for c, (x0, x1) in {"R+": (0.0, 1.0), "R-": (2.4, 3.4)}.items():
        for k, (cl, col) in enumerate((("SL", "p_SL"), ("AH", "p_AH"), ("WH", "p_WH"))):
            key = "FA" if cl == "SL" else cl
            mm, ee = [], []
            for s_ in ("learning", "expert"):
                v = P[(P.cohort == c) & (P.stage == s_)][col].dropna()
                mm.append(v.mean()); ee.append(v.sem())
            dx = (k - 1) * 0.12
            ax.plot([x0 + dx, x1 + dx], mm, color=CL[key], lw=1.0)
            for x, m_, e_, s_ in zip((x0 + dx, x1 + dx), mm, ee, ("learning", "expert")):
                ax.errorbar(x, m_, e_, fmt="o", ms=3.2, color=CL[key], mfc="white" if s_ == "learning" else CL[key], lw=0.9, capsize=0)
    ax.axhline(0, color="0.8", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1, 2.4, 3.4], ["D0", "E", "D0", "E"]); ax.set_xlim(-0.5, 3.9)
    for x, c in [(0.5, "R+"), (2.9, "R-")]:
        ax.annotate(c.replace("-", "−"), xy=(x, 0), xycoords=ax.get_xaxis_transform(), xytext=(0, -11), textcoords="offset points",
                    ha="center", va="top", fontsize=5.5, color=COH[c], weight="bold")
    ax.set_ylabel("session-mean projection on CD\n(per neuron)")
    ax.set_title("Mean projection per event class\n(0 = mean of AH + SL events)", fontsize=5.2, pad=3)


def small_panel(ax, P, col, ylab, rng, title):
    S = m62.group_stats(P, col, title, rng, f"088 {col}")
    for c, (x0, x1) in {"R+": (0.0, 0.6), "R-": (1.6, 2.2)}.items():
        mm = [S["means"][(c, s)] for s in ("learning", "expert")]; ee = [S["sems"][(c, s)] for s in ("learning", "expert")]
        ax.plot([x0, x1], mm, color=COH[c], lw=0.9)
        for x, m_, e_, s in zip((x0, x1), mm, ee, ("learning", "expert")):
            ax.errorbar(x, m_, e_, fmt="o", ms=3.2, color=COH[c], mfc="white" if s == "learning" else COH[c], lw=0.9, capsize=0)
    p = lambda k: m62.fmt_p(S[k][0]) if k in S else "n/a"
    ax.set_title(f"{title}\nc×s {m62.fmt_p(S['interaction'][1])}\nD0 R+/R− {p('learning R+ vs R-')}\nE R+/R− {p('expert R+ vs R-')}",
                 fontsize=4.3, pad=2)
    ax.set_xticks([0.3, 1.9], [f"{c.replace('-', '−')}\n({S['n'][(c, 'learning')]}/{S['n'][(c, 'expert')]})" for c in ("R+", "R-")], fontsize=4.4)
    for t, c in zip(ax.get_xticklabels(), ("R+", "R-")):
        t.set_color(COH[c])
    ax.set_xlim(-0.4, 2.6); ax.set_ylabel(ylab, fontsize=4.6); ax.tick_params(labelsize=4.4); ax.margins(y=0.15)


def area_panel(ax, PA, rows, rng):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    LA = PA.rename(columns={"area": "region"})
    Q = m62.area_qualify(LA, col="d_WS")
    order = [g for g in au.get_area_group_custom_order() if g in Q["R+"] and g in Q["R-"]]
    y = np.arange(len(order))[::-1]; pv = []
    for i, g in enumerate(order):
        da = LA[LA.region == g]
        for k, c in enumerate(("R+", "R-")):
            d = da[da.cohort == c].dropna(subset=["d_WS"])
            a, b = d[d.stage == "learning"].d_WS, d[d.stage == "expert"].d_WS
            diff, se = b.mean() - a.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            ax.errorbar(diff, y[i] + 0.16 - 0.32 * k, xerr=se, fmt="o", ms=2.8, color=COH[c], lw=0.8, capsize=0, mfc="white")
            rows.append(dict(panel="i", area=g, cohort=c, change=diff, sem=se, n_day0=len(a), n_expert=len(b)))
        S = m62.group_stats(da.dropna(subset=["d_WS"]), "d_WS", f"area {g}", rng, "088 i")
        pv.append(S["interaction"][1])
    pv = np.array(pv, float); qv = np.full_like(pv, np.nan); ok = np.isfinite(pv)
    if ok.any():
        o = np.argsort(pv[ok]); r_ = pv[ok][o] * ok.sum() / (np.arange(ok.sum()) + 1)
        qv[np.where(ok)[0][o]] = np.minimum.accumulate(r_[::-1])[::-1].clip(max=1)
    for g, p_, q_ in zip(order, pv, qv):
        rows.append(dict(panel="i q", area=g, p_perm=p_, q_BH=q_))
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y, [m83.m62_short(g) + (" *" if q_ < 0.05 else "") for g, q_ in zip(order, qv)], fontsize=4.6)
    for tl, g in zip(ax.get_yticklabels(), order):
        tl.set_color(gcol.get(g, "k"))
    ax.set_xlabel("Δ WH − SL projection on CD\n(expert − day 0, per neuron)", fontsize=5)
    AN = m62.area_anova(LA, order, col="d_WS")
    rows.append(dict(panel="i ANOVA", **{f"{k} p_perm": v["p_perm"] for k, v in (AN or {}).items()}, n_areas=len(order)))
    ax.set_title(f"Per area: cohort × stage × area\n{m62.fmt_p(AN['cohort x stage x area']['p_perm'])}; * q < 0.05", fontsize=4.8, pad=2)


def pick_examples(Wc):
    af, wf = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
    G = Wc[(Wc.quality_label == "good")].copy()
    G["fr"] = G[f"fr_window:{af}"]; sa, sw = G[f"sel:{af}"], G[f"sel:{wf}"]
    G["score"] = (sa.abs() + sw.abs()) * np.log1p(G.fr)
    spec = [("Converging, non-sensory (R+ expert)", G.conv.fillna(False) & (G.nclass == "non-sensory") & (G.cohort == "R+") & (G.stage == "expert")),
            ("Converging (R+ day 0)", G.conv.fillna(False) & (G.cohort == "R+") & (G.stage == "learning")),
            ("AH-like whisker hits, MI > 0.7 (R+ expert)", G.rl.fillna(False) & (G.mi > 0.7) & (G.cohort == "R+") & (G.stage == "expert")),
            ("SL-like whisker hits, MI < −0.7 (R− expert)", G.rl.fillna(False) & (G.mi < -0.7) & (G.cohort == "R-") & (G.stage == "expert"))]
    out = []
    for ttl, m in spec:
        g = G[m & (G.fr >= 5)]
        if len(g):
            r = g.loc[g.score.idxmax()]
            out.append((ttl, r))
    return out


def main(a):
    plt = m62.setup()
    rng = np.random.default_rng(0)
    MI = all_mi(a.n_proc)
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda D_, c: D_[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U0["rl"] = sig(U0, f"sig:{m81.SEL_AS}") & (U0[f"sel:{m81.SEL_AS}"] > 0)
    U0["conv"] = U0.rl & sig(U0, f"sig:{m81.SEL_WS}") & (U0[f"sel:{m81.SEL_WS}"] > 0)
    for k in ("electrode_group", "cluster_id"):
        U0[k] = U0[k].astype(str)
    U0 = U0.merge(MI, on=["session_id", "electrode_group", "cluster_id"], how="left")
    C0 = pd.read_csv(m81.OUT / "classes.csv")
    D = m62.load_population("learners")
    W = D["W"].copy()
    for k in ("electrode_group", "cluster_id"):
        W[k] = W[k].astype(str)
    PB, KB, tc = m62.load_psth_prestart(W)
    KB = KB.assign(electrode_group=KB.electrode_group.astype(str), cluster_id=KB.cluster_id.astype(str))
    W = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
    Wc = W.merge(U0[["session_id", "electrode_group", "cluster_id", "nclass", "rl", "conv", "mi"]], on=["session_id", "electrode_group", "cluster_id"], how="left")
    U = m61.learner_filter(U0); C = m61.learner_filter(C0)
    S = U[U.rl & U.nclass.isin(m81.CLASSES)].groupby(["session_id", "mouse_id", "cohort", "stage"]).agg(
        conv=("conv", "mean"), mi=("mi", "mean"), n=("conv", "size")).reset_index()
    S = S[S.n >= m81.MIN_CLASS]
    PA = m61.learner_filter(pd.read_csv(H087)); P = PA[PA.area == "whole_brain"]
    rows = []
    with plt.rc_context({"font.size": 5.4, "axes.titlesize": 5.6, "axes.labelsize": 5.3, "xtick.labelsize": 4.9,
                         "ytick.labelsize": 4.9, "legend.fontsize": 4.8}):
        fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.92))
        gs = fig.add_gridspec(3, 4, height_ratios=[0.8, 1, 1], hspace=0.8, wspace=0.5, left=0.065, right=0.975, top=0.9, bottom=0.06)

        def cell_title(spec, text, dy=0.03):
            bb = spec.get_position(fig)
            fig.text(bb.x0 + 0.035, bb.y1 + dy, text, fontsize=5.6, va="bottom")
        letters = []
        # row 1: examples
        for k, (ttl, r) in enumerate(pick_examples(Wc)):
            ap, ar = m62.raster_psth(fig, gs[0, k], r, rng, f"{ttl}\n{r.get('area_acronym_custom', r.get('area_group', ''))}, MI = {r.mi:+.2f}", "0.15")
            ap.set_ylabel("Hz", fontsize=4.8)
            if k == 0:
                letters.append(ap)
            ar.set_xlabel("time from first lick (ms)", fontsize=4.8)
            rows.append(dict(panel="a", example=ttl, session_id=r.session_id, electrode_group=r.electrode_group, cluster_id=r.cluster_id, mi=r.mi))
        # row 2
        tested = (Wc["sig:auditory_hit_vs_fa_prelick@all"].notna() & Wc["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
        ib = m83.psth_insets(fig, gs[1, 0], Wc, PB, tc, tested, "b", rows); letters.append(ib[0]); cell_title(gs[1, 0], "All neurons")
        ic = m83.psth_insets(fig, gs[1, 1], Wc, PB, tc, (Wc.conv.fillna(False) & (Wc.nclass == "non-sensory")).to_numpy(), "c", rows)
        letters.append(ic[0]); cell_title(gs[1, 1], "Converging, non-sensory neurons")
        ax = fig.add_subplot(gs[1, 2]); ax.set_box_aspect(1); letters.append(ax)
        mean_panel2(ax, S, "conv", "Fraction of reward-lick neurons", rng, "088 conv", "Converging neurons")
        for cl, dx in (("sensory", -0.18), ("lick-gated", 0.0), ("non-sensory", 0.18)):
            d = C[C.nclass == cl].dropna(subset=["conv"])
            for (c, s), x in zip(GROUPS, (0, 1, 2.4, 3.4)):
                v = d[(d.cohort == c) & (d.stage == s)]["conv"]
                ax.plot(x + 0.28 + dx * 0.5, v.mean(), marker="s", ms=2.0, color=CCOL[cl], ls="", alpha=0.9,
                        label=cl if (c, s) == GROUPS[0] else None)
        ax.legend(frameon=False, fontsize=4.0, loc="center right", handlelength=0.8, borderaxespad=0.1, title="neuron class (squares)", title_fontsize=4.0)
        ax = fig.add_subplot(gs[1, 3]); ax.set_box_aspect(1); letters.append(ax)
        mean_panel2(ax, S, "mi", "Whisker-hit modulation index MI\n(−1 SL-like … +1 AH-like)", rng, "088 mi", "Whisker hits like auditory hits", ref=0)
        # row 3: current CD definition
        ax = fig.add_subplot(gs[2, 0]); ax.set_box_aspect(1); letters.append(ax); proj_levels(ax, P)
        ax = fig.add_subplot(gs[2, 1]); ax.set_box_aspect(1); letters.append(ax)
        mean_panel2(ax, P, "d_WS", "WH − SL projection on CD\n(per neuron)", rng, "088 d_WS", "WH move toward AH", ref=0)
        gh = gs[2, 2].subgridspec(1, 2, wspace=1.1)
        ah = [fig.add_subplot(gh[i]) for i in range(2)]
        small_panel(ah[0], P, "theta", "θ (°)", rng, "Angle θ")
        small_panel(ah[1], P, "rho_v", "ρ (per neuron)", rng, "Length ρ")
        letters.append(ah[0]); cell_title(gs[2, 2], "Angle and length of WH − SL", dy=0.07)
        ax = fig.add_subplot(gs[2, 3]); ax.set_box_aspect(1); letters.append(ax)
        area_panel(ax, PA, rows, rng)
        from matplotlib.lines import Line2D
        fig.legend([Line2D([], [], color=CL[k_], lw=1.4, marker="o", ms=2.6) for k_ in ("WH", "AH", "FA")],
                   ["Whisker hit", "Auditory hit", "Spontaneous lick"], frameon=False, fontsize=5.2, ncol=3, loc="upper left",
                   bbox_to_anchor=(0.06, 0.975), handlelength=1.4)
        m62.letter_row(fig, letters[:1], "a", dx_in=0.3, dy_in=0.22)
        m62.letter_row(fig, letters[1:5], "bcde", dx_in=0.3, dy_in=0.22)
        m62.letter_row(fig, letters[5:], "fghi", dx_in=0.3, dy_in=0.22)
        fig.suptitle("Reward recruits neurons across the brain into a stimulus-independent reward-lick code", x=0.02, y=0.998,
                     ha="left", va="top", fontsize=7, weight="bold")
        out = m51.OUTROOT / "publication" / "learners"
        m62.save(fig, out, "COSYNE_across_days_v2"); plt.close(fig)
    R = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith("088")])
    R.to_csv(out / "COSYNE_across_days_v2_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("088") and not str(r.get("measure", "")).startswith("area")]).round(4).to_string())
    print("ALL DONE", out / "COSYNE_across_days_v2.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-proc", type=int, default=30)
    main(ap.parse_args())
