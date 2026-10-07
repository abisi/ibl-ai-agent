"""083 -- COSYNE abstract figure, across days (user 2026-10-08): 2 rows x 4 square panels, single neurons (row 1) and
population geometry (row 2), last panel = per-area breakdown of the population geometry. Few methods shown.

  a  Same lick, different expected outcome: first-lick-aligned population PSTHs (all tested units; rate minus the unit's
     pre-trial baseline; mean over units within session, mean +- s.e.m. over sessions), insets R+ / R- x day 0 / expert.
  b  Converging non-sensory neurons (081 classes: not significant in whisker_active / auditory_active nor whisker_choice;
     pre-lick AH > SL and WH > SL significant): PSTH insets for the four groups.
  c  Recruitment: converging fraction among reward-lick (AH > SL) neurons per session, day 0 -> expert per cohort (mean +-
     s.e.m.; brackets: Mann-Whitney U; title: learning x cohort interaction, mouse-level permutation, Eq. 15); small
     markers: the three stimulus classes (081).
  d  Position of each reward-lick neuron between SL and AH: AH-likeness c (Eq. 3), per-session mean.
  e  Group geometry: triangles from the group-mean cross-validated distances (Eq. 5), day 0 dashed -> expert solid.
  f  Distance difference Delta-d = d(WH, SL) - d(WH, AH) per session (057).
  g  Length grows, no detectable rotation (user wording 2026-10-08; the angle CI is wide): angle theta between WH - SL and AH - SL and length ratio rho (Eq. 10) per group, mean and
     95 % hierarchical-bootstrap CI (mice, then sessions; 5,000 resamples); text: change day 0 -> expert per cohort and
     the cohort difference of the change (interaction), each with its 95 % CI.
  h  Per area group: change of Delta-d from day 0 to expert (mean +- s.e.m.), R+ and R-; right: FDR q (Benjamini-Hochberg
     over the area groups) of the per-area interaction; area groups with >= 3 sessions at
     both stages in both cohorts; right: interaction p (mouse permutation); title: ANOVA Delta-d ~ cohort x stage x area
     (cohort-label permutation across mice).
Output: across_days/<ref>[_units]/publication/<pop>/COSYNE_across_days.{png,pdf,svg} + COSYNE_across_days_stats.csv.
"""
import argparse
import importlib
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
COH, CL = m62.COH, m62.CL
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
GLAB = {("R+", "learning"): "R+ day 0", ("R+", "expert"): "R+ expert", ("R-", "learning"): "R− day 0", ("R-", "expert"): "R− expert"}
CCOL = {"sensory": "#4C72B0", "lick-gated": "#DD8452", "non-sensory": "#55A868"}
N_BOOT = 5000


def hboot(d, col, rng, B=N_BOOT):
    """hierarchical bootstrap of the group mean: mice with replacement, then sessions within mouse"""
    g = {k: v[col].dropna().to_numpy() for k, v in d.groupby("mouse_id")}
    g = {k: v for k, v in g.items() if len(v)}
    mice = list(g)
    return np.array([np.mean(np.concatenate([rng.choice(g[m], len(g[m])) for m in rng.choice(mice, len(mice))]))
                     for _ in range(B)])


def psth_insets(fig, spec, W, PB, tc, mask, title, rows):
    sub = spec.subgridspec(2, 2, hspace=0.35, wspace=0.12)
    axs = []
    for j, g in enumerate(GROUPS):
        ax = fig.add_subplot(sub[j // 2, j % 2]); axs.append(ax)
        mk = mask & ((W.cohort == g[0]) & (W.stage == g[1])).to_numpy()
        ns, nu = m62.mouse_psth(ax, W, PB, tc, mk, min_units=1)
        ax.set_xlim(-400, 200); ax.set_xticks([-300, 0])
        ax.set_xticklabels(["−300", "0"] if j >= 2 else [])
        ax.set_title(GLAB[g], color=COH[g[0]], fontsize=4.8, pad=1.5)
        rows.append(dict(panel=title, group=GLAB[g], n_sessions=ns, n_units=nu))
    yl = (min(a.get_ylim()[0] for a in axs), max(a.get_ylim()[1] for a in axs))
    for j, a in enumerate(axs):
        a.set_ylim(*yl)
        if j % 2:
            a.set_yticklabels([])
        a.tick_params(labelsize=4.4)
    axs[0].set_ylabel("Hz", fontsize=4.8); axs[2].set_ylabel("Hz", fontsize=4.8)
    return axs


def angle_table(Lwb):
    d = Lwb.copy()
    ok = (d.d_WH_FA > 0) & (d.d_AH_FA >= 0.01)
    a = (d.d_WH_FA + d.d_AH_FA - d.d_WH_AH) / 2
    d["theta"] = np.where(ok, np.degrees(np.arccos(np.clip(a / np.sqrt(d.d_WH_FA.clip(lower=1e-12) * d.d_AH_FA), -1, 1))), np.nan)
    d["rho"] = np.where(ok, np.sqrt(d.d_WH_FA.clip(lower=0) / d.d_AH_FA), np.nan)
    return d


def ci_panel(fig, spec, A, rng, rows):
    sub = spec.subgridspec(1, 2, wspace=0.55)
    axs = []
    for k, (col, lab, unit) in enumerate((("theta", "Angle θ", "°"), ("rho", "Length ratio ρ", ""))):   # mean, 95% CI
        ax = fig.add_subplot(sub[k]); axs.append(ax)
        boots = {}
        for i, g in enumerate(GROUPS):
            d = A[(A.cohort == g[0]) & (A.stage == g[1])]
            b = hboot(d, col, rng); boots[g] = b
            m = d.groupby("mouse_id")[col].mean().mean()
            x = [0, 0.6, 1.6, 2.2][i]
            ax.errorbar(x, np.mean(b), [[np.mean(b) - np.percentile(b, 2.5)], [np.percentile(b, 97.5) - np.mean(b)]],
                        fmt="o", ms=3.5, color=COH[g[0]], lw=1.1, capsize=0, mfc="white" if g[1] == "learning" else COH[g[0]])
        for c, xs in (("R+", (0, 0.6)), ("R-", (1.6, 2.2))):
            ax.plot(xs, [np.mean(boots[(c, "learning")]), np.mean(boots[(c, "expert")])], color=COH[c], lw=0.8)
        dP = boots[("R+", "expert")] - boots[("R+", "learning")]
        dM = boots[("R-", "expert")] - boots[("R-", "learning")]
        inter = dP - dM
        ci = lambda v: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        r = dict(panel="g", measure=col, dRplus=dP.mean(), dRplus_lo=ci(dP)[0], dRplus_hi=ci(dP)[1], dRminus=dM.mean(),
                 dRminus_lo=ci(dM)[0], dRminus_hi=ci(dM)[1], interaction=inter.mean(), inter_lo=ci(inter)[0],
                 inter_hi=ci(inter)[1], p_boot=min(1, 2 * min((inter <= 0).mean(), (inter >= 0).mean())))
        rows.append(r)
        fmt = (lambda v: f"{v:+.0f}") if col == "theta" else (lambda v: f"{v:+.2f}")
        ax.set_title(lab + "\ncohort × stage\n" + fmt(r["interaction"]) + unit + " [" + fmt(r["inter_lo"]) + ", " + fmt(r["inter_hi"]) + "]", fontsize=4.6, pad=2)
        ax.set_ylabel("θ (degrees)" if col == "theta" else "ρ = |WH − SL| / |AH − SL|", fontsize=4.6)
        ax.set_xticks([0.3, 1.9], ["R+", "R−"], fontsize=4.8)
        for t, c in zip(ax.get_xticklabels(), ("R+", "R-")):
            t.set_color(COH[c])
        ax.set_xlim(-0.4, 2.6); ax.tick_params(labelsize=4.4); ax.margins(y=0.15)
    return axs


def area_panel(ax, LA, rows):
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    Q = m62.area_qualify(LA, col="dd")
    order = [g for g in au.get_area_group_custom_order() if g in Q["R+"] and g in Q["R-"]]
    T = pd.read_csv(m51.OUTROOT / "learners" / "tests_learners.csv") if POP == "learners" else pd.read_csv(m51.OUTROOT / "lambda" / "lambda_tests.csv")
    T = T[(T.metric == "dd") & (T.variant == "all") & (T.level == "area_group")].set_index("region")
    y = np.arange(len(order))[::-1]
    for k, c in enumerate(("R+", "R-")):
        for i, g in enumerate(order):
            d = LA[(LA.region == g) & (LA.cohort == c)].dropna(subset=["dd"])
            a, b = d[d.stage == "learning"].dd, d[d.stage == "expert"].dd
            diff, se = b.mean() - a.mean(), np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
            ax.errorbar(diff, y[i] + 0.16 - 0.32 * k, xerr=se, fmt="o", ms=2.8, color=COH[c], lw=0.8, capsize=0, mfc="white")
            rows.append(dict(panel="h", area=g, cohort=c, change=diff, sem=se, n_day0=len(a), n_expert=len(b)))
    # FDR (Benjamini-Hochberg) over the area groups shown, of the per-area interaction p (user 2026-10-08)
    pv = np.array([T.loc[g, "interaction_p"] if g in T.index else np.nan for g in order], float)
    ok = np.isfinite(pv); qv = np.full_like(pv, np.nan)
    if ok.any():
        o = np.argsort(pv[ok]); r_ = pv[ok][o] * ok.sum() / (np.arange(ok.sum()) + 1)
        qv[np.where(ok)[0][o]] = np.minimum.accumulate(r_[::-1])[::-1].clip(max=1)
    ax.text(1.02, len(order) - 0.4, "q", transform=ax.get_yaxis_transform(), va="bottom", fontsize=4.4, weight="bold")
    for i, g in enumerate(order):
        q = qv[i]
        ax.text(1.02, y[i], f"{q:.2f}".lstrip("0") if np.isfinite(q) else "", transform=ax.get_yaxis_transform(),
                va="center", fontsize=4.2, color="#D62728" if q < 0.05 else "0.4")
        rows.append(dict(panel="h q", area=g, p_perm=pv[i], q_BH=q))
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y, [m62_short(g) for g in order], fontsize=4.6)
    for tl, g in zip(ax.get_yticklabels(), order):
        tl.set_color(gcol.get(g, "k"))
    ax.set_xlabel("Δd change, expert − day 0", fontsize=5)
    AN = m62.area_anova(LA, order, col="dd")
    pos = sum(1 for g in order if g in T.index and T.loc[g, "interaction"] > 0)
    rows.append(dict(panel="h ANOVA", **{f"{k} p_perm": v["p_perm"] for k, v in (AN or {}).items()}, n_areas=len(order),
                     n_positive_interaction=pos))
    ax.set_title(f"Across the brain: interaction > 0 in {pos}/{len(order)}\n"
                 f"cohort × stage × area {m62.fmt_p(AN['cohort x stage x area']['p_perm'])}", fontsize=4.8, pad=2)


def m62_short(g):
    return (g.replace(" areas", "").replace("Somatosensory-", "SS-").replace("Lateral septal complex", "LSX")
            .replace("Posterior parietal", "PPC").replace("Retrosplenial", "RSP").replace("Hippocampus", "HPF")
            .replace("Amygdala and hypothalamus", "Amyg."))


POP = "learners"


def main(a):
    global POP
    plt = m62.setup()
    U0 = pd.read_parquet(m81.OUT / "units.parquet")
    sig = lambda D_, c: D_[c].astype(str).str.lower().isin(["true", "1", "1.0"])
    U0["rl"] = sig(U0, f"sig:{m81.SEL_AS}") & (U0[f"sel:{m81.SEL_AS}"] > 0)
    U0["conv"] = U0.rl & sig(U0, f"sig:{m81.SEL_WS}") & (U0[f"sel:{m81.SEL_WS}"] > 0)
    C0 = pd.read_csv(m81.OUT / "classes.csv")
    for pop in ("learners", "all"):
        POP = pop
        rng = np.random.default_rng(0)
        flt = m61.learner_filter if pop == "learners" else (lambda d: d)
        D = m62.load_population(pop)
        W = D["W"].copy()
        PB, KB, tc = m62.load_psth_prestart(W)
        W = W.merge(KB, on=["session_id", "electrode_group", "cluster_id"], how="inner")
        Wc = W.merge(U0[["session_id", "electrode_group", "cluster_id", "nclass", "rl", "conv"]],
                     on=["session_id", "electrode_group", "cluster_id"], how="left")
        L = D["lam"]; Lwb = L[(L.level == "all") & (L.variant == "all")]
        LA = L[(L.level == "area_group") & (L.variant == "all")]
        U = flt(U0); C = flt(C0)
        S = U[U.rl & U.nclass.isin(m81.CLASSES)].groupby(["session_id", "mouse_id", "cohort", "stage"]).agg(
            conv=("conv", "mean"), like=("like", "mean"), n=("conv", "size")).reset_index()
        S = S[S.n >= m81.MIN_CLASS]
        rows = []
        with plt.rc_context({"font.size": 5.4, "axes.titlesize": 5.6, "axes.labelsize": 5.3, "xtick.labelsize": 4.9,
                             "ytick.labelsize": 4.9, "legend.fontsize": 4.8}):
            fig = plt.figure(figsize=(m62.W_IN, m62.W_IN * 0.6))
            gs = fig.add_gridspec(2, 4, hspace=0.75, wspace=0.45, left=0.065, right=0.975, top=0.84, bottom=0.08)

            def cell_title(spec, text, dy=0.045):            # heading of a panel made of insets, above its cell
                bb = spec.get_position(fig)
                fig.text(bb.x0 + 0.035, bb.y1 + dy, text, fontsize=5.6, va="bottom")
            letters_ax = []
            tested = (Wc["sig:auditory_hit_vs_fa_prelick@all"].notna() & Wc["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
            ia = psth_insets(fig, gs[0, 0], Wc, PB, tc, tested, "a", rows); letters_ax.append(ia[0])
            cell_title(gs[0, 0], "All neurons")
            ib = psth_insets(fig, gs[0, 1], Wc, PB, tc, (Wc.conv.fillna(False) & (Wc.nclass == "non-sensory")).to_numpy(), "b", rows)
            letters_ax.append(ib[0])
            cell_title(gs[0, 1], "Converging, non-sensory neurons")
            # c, d
            for k, (col, ylab, ttl) in enumerate((("conv", "Fraction of reward-lick neurons", "Converging neurons"),
                                                  ("like", "AH-likeness (−1 SL … +1 AH)", "Whisker hits like auditory hits"))):
                ax = fig.add_subplot(gs[0, 2 + k]); ax.set_box_aspect(1); letters_ax.append(ax)
                m62.mean_panel(ax, S, col, ylab, rng, f"083 {col}", ttl, ref=[(0, "0.6")] if col == "like" else None)
                ax.set_title(ax.get_title(), fontsize=5.0)
                ax.set_xticklabels(["D0", "E", "D0", "E"])
                for cl, dx in (("sensory", -0.18), ("lick-gated", 0.0), ("non-sensory", 0.18)):
                    d = C[C.nclass == cl].dropna(subset=[col])
                    for (c, s), x in zip(GROUPS, (0, 1, 2.4, 3.4)):
                        v = d[(d.cohort == c) & (d.stage == s)][col]
                        ax.plot(x + 0.28 + dx * 0.5, v.mean(), marker="s", ms=2.0, color=CCOL[cl], ls="", alpha=0.9)
            # e triangles
            ge = gs[1, 0].subgridspec(2, 1, hspace=0.55)
            axs_e = [fig.add_subplot(ge[i]) for i in range(2)]
            for i_, (ax, c) in enumerate(zip(axs_e, ("R+", "R-"))):
                m62.draw_triangles(ax, Lwb, c, "SL")
                ax.set_title(f"{c.replace('-', '−')}: day 0 (open, dashed) → expert", color=COH[c], fontsize=4.6, pad=1.5)
                ax.set_ylabel("Orthogonal", fontsize=4.4)
                ax.set_xlabel("Along SL → AH" if i_ == 1 else "", fontsize=4.4)
                ax.tick_params(labelsize=4.2)
            m62.share_triangle_limits(axs_e); letters_ax.append(axs_e[0])
            cell_title(gs[1, 0], "Population geometry")
            from matplotlib.lines import Line2D
            fig.legend([Line2D([], [], color=CL[k_], lw=1.4, marker="o", ms=2.6) for k_ in ("WH", "AH", "FA")],
                       ["Whisker hit", "Auditory hit", "Spontaneous lick"], frameon=False, fontsize=5.2, ncol=3,
                       loc="upper left", bbox_to_anchor=(0.06, 0.965), handlelength=1.4)
            # f
            ax = fig.add_subplot(gs[1, 1]); ax.set_box_aspect(1); letters_ax.append(ax)
            m62.mean_panel(ax, Lwb, "dd", "Δd = d(WH,SL) − d(WH,AH)", rng, "083 dd", "WH move toward AH", ref=[(0, "0.6")])
            ax.set_title(ax.get_title(), fontsize=5.0); ax.set_xticklabels(["D0", "E", "D0", "E"])
            # g angle / length with CI
            A = angle_table(Lwb)
            ig = ci_panel(fig, gs[1, 2], A, rng, rows); letters_ax.append(ig[0])
            cell_title(gs[1, 2], "Length grows, no detectable rotation", dy=0.085)
            # h areas
            ax = fig.add_subplot(gs[1, 3]); ax.set_box_aspect(1); letters_ax.append(ax)
            area_panel(ax, LA, rows)
            m62.letter_row(fig, letters_ax[:4], "abcd", dx_in=0.3, dy_in=0.22)
            m62.letter_row(fig, letters_ax[4:], "efgh", dx_in=0.3, dy_in=0.22)
            fig.suptitle("Reward recruits neurons across the brain into a stimulus-independent reward-lick code",
                         x=0.02, y=0.995, ha="left", va="top", fontsize=7, weight="bold")
            out = m51.OUTROOT / "publication" / pop
            m62.save(fig, out, "COSYNE_across_days"); plt.close(fig)
        R = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith("083")])
        R.to_csv(out / "COSYNE_across_days_stats.csv", index=False)
        print(pop); print(pd.DataFrame([r for r in rows if r.get("panel") in ("g", "h ANOVA")]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
