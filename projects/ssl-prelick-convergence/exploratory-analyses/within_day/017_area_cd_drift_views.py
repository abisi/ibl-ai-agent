"""017 -- Per-area within-session CD drift (015 area_sessions.csv), other views (user 2026-10-07):
  1. one forest plot per stage (day 0 only, expert only): drift of WH relative to SL and to AH per area group, R+ and R-,
     mean +- s.e.m. over sessions; per stage an area group enters a cohort with >= 3 sessions of that stage (stage-specific
     rule, so day-0 plots can show areas without enough expert sessions); R+ vs R- per area where both cohorts enter:
     Mann-Whitney U, Welch t and mouse-level cohort permutation (5000), Benjamini-Hochberg over areas (permutation p).
  2. matrices (as the across-day COSYNE area heatmaps): area group x [R+ day 0, R+ expert, R- day 0, R- expert], mean drift,
     '*' = drift differs from 0 (Wilcoxon p < 0.05); right-hand columns: R+ vs R- p on day 0, in experts, and the
     cohort x stage interaction (015 tests; 015 rule: >= 3 sessions at both stages in both cohorts).
Output: within_day/<ref>/area_cd_drift/<pop>/area_cd_drift_{day0,expert,matrix}.{png,pdf,svg}, area_stage_tests.csv;
strict duplicates (user 2026-10-07: only areas with >= 3 sessions in every cohort x stage condition, BH over those areas):
area_cd_drift_{day0,expert}_strict.*, area_stage_tests_strict.csv (the matrix already uses that rule).
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m001 = importlib.import_module("001_within_session_halves")
m015 = importlib.import_module("015_area_cd_drift")
OUT = m015.OUT
COH = m62.COH
MEAS = [("drift_WH_SL", "WH relative to SL\n(β_WH − β_SL)"), ("drift_WH_AH", "WH relative to AH\n(β_WH − β_AH)")]
SHORT = lambda g: (g.replace(" areas", "").replace("Somatosensory-", "Ss-").replace("Lateral septal complex", "LSX")
                   .replace("Posterior parietal", "PPC").replace("Retrosplenial", "RSP").replace("Hippocampus", "HPF")
                   .replace("Amygdala and hypothalamus", "Amyg."))
fp = lambda p: "" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}".lstrip("0"))


def stage_tests(A, col, st, min_sess=3):
    rows = []
    d0 = A[A.stage == st].dropna(subset=[col])
    for g, d in d0.groupby("area_group"):
        r = dict(measure=col, stage=st, area_group=g)
        ok = {}
        for c in ("R+", "R-"):
            v = d[d.cohort == c][col]
            ok[c] = len(v) >= min_sess
            r[f"n {c}"], r[f"n mice {c}"] = len(v), d[d.cohort == c].mouse_id.nunique()
            if ok[c]:
                r[f"mean {c}"], r[f"sem {c}"] = v.mean(), v.sem()
                r[f"p_wilcoxon {c}"] = stats.wilcoxon(v).pvalue if (v != 0).any() else np.nan
        if ok["R+"] and ok["R-"]:
            x, y = d[d.cohort == "R+"][col], d[d.cohort == "R-"][col]
            r["p_MWU"] = stats.mannwhitneyu(x, y).pvalue
            r["p_Welch"] = stats.ttest_ind(x, y, equal_var=False).pvalue
            r["R+ - R-"], r["p_perm"] = m001.perm_interaction(d.assign(stage="learning"), col, n_perm=5000, kind="learning")
        if ok["R+"] or ok["R-"]:
            rows.append(r)
    T = pd.DataFrame(rows)
    if "p_perm" in T:
        T["q_BH"] = m015.bh(T["p_perm"])
    return T


def order_of(groups):
    import ephys_utilities.allen_utils.allen_utils as au
    return [g for g in au.get_area_group_custom_order() if g in set(groups)]


def forest(ax, T, order, show_labels, gcol, xlabel=True):
    y = np.arange(len(order))[::-1]
    T = T.set_index("area_group")
    for i, g in enumerate(order):
        if g not in T.index:
            continue
        r = T.loc[g]
        for k, c in enumerate(("R+", "R-")):
            m_ = r.get(f"mean {c}", np.nan)
            if not np.isfinite(m_):
                continue
            p0 = r.get(f"p_wilcoxon {c}", np.nan)
            sig = np.isfinite(p0) and p0 < 0.05
            ax.errorbar(m_, y[i] + 0.14 - 0.28 * k, xerr=r[f"sem {c}"], fmt="o", ms=3, color=COH[c], lw=0.8, capsize=0,
                        mfc=COH[c] if sig else "white")
    ax.axvline(0, color="0.6", lw=0.5)
    ax.set_yticks(y, [SHORT(g) for g in order] if show_labels else [], fontsize=5.2)
    if show_labels:
        for tl, g in zip(ax.get_yticklabels(), order):
            tl.set_color(gcol.get(g, "k"))
    ax.set_ylim(-0.6, len(order) - 0.4)
    if xlabel:
        ax.set_xlabel("Change over one session (CD units)", fontsize=5.2)
    return y, T


def pcol(ax, y, T, order, title="R+ vs R−"):
    """p-value columns: mouse-level permutation, its Benjamini-Hochberg q over the areas shown, MWU, Welch"""
    ax.axis("off")
    ax.text(0, len(order) - 0.45, f"{title}\n perm  q_FDR   MWU  Welch", fontsize=4.4, va="bottom", weight="bold",
            family="DejaVu Sans Mono")
    for i, g in enumerate(order):
        if g in T.index and np.isfinite(T.loc[g].get("p_perm", np.nan)):
            r = T.loc[g]
            q = r.get("q_BH", np.nan)
            ax.text(0, y[i], f"{fp(r['p_perm']):>5}  {fp(q):>5}  {fp(r['p_MWU']):>5}  {fp(r['p_Welch']):>5}", fontsize=4.3,
                    va="center", family="DejaVu Sans Mono", color="#D62728" if r["p_perm"] < 0.05 else "0.25",
                    weight="bold" if (np.isfinite(q) and q < 0.05) else "normal")


def strict_areas(A, col, min_sess=3):
    """area groups with >= min_sess sessions in every condition (R+ / R- x day 0 / expert) for this measure"""
    n = A.dropna(subset=[col]).groupby(["area_group", "cohort", "stage"]).session_id.nunique().unstack(["cohort", "stage"])
    need = [(c, s) for c in ("R+", "R-") for s in ("learning", "expert")]
    n = n.reindex(columns=need).fillna(0)
    return set(n.index[(n >= min_sess).all(1)])


def stage_area_anova(A, areas, col, stage, n_perm=1000, seed=0):
    """two-way ANOVA within one stage on session x area drifts: y ~ C(cohort) * C(area) (type II, OLS), areas shown only.
    Terms: cohort (R+ vs R- across areas), area, cohort x area (does the cohort difference depend on the area). A session
    contributes several areas (not independent), so cohort and cohort x area are also tested by permuting cohort labels
    across mice (n_perm, F re-estimated); the area term is not changed by that permutation (parametric p only)."""
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    d = A[(A.stage == stage) & A.area_group.isin(areas)].dropna(subset=[col]).rename(columns={col: "y", "area_group": "area"})
    terms = {"cohort": "C(cohort)", "area": "C(area)", "cohort x area": "C(cohort):C(area)"}
    fit = lambda dd: anova_lm(smf.ols("y ~ C(cohort) * C(area)", data=dd).fit(), typ=2)
    tab = fit(d)
    out = {k: dict(F=tab.loc[v, "F"], p=tab.loc[v, "PR(>F)"]) for k, v in terms.items()}
    rng = np.random.default_rng(seed)
    mice = d.mouse_id.unique(); mc = d.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    midx = pd.Index(mice).get_indexer(d.mouse_id)
    null = {k: [] for k in ("cohort", "cohort x area")}
    for _ in range(n_perm):
        t = fit(d.assign(cohort=rng.permutation(mc)[midx]))
        for k in null:
            null[k].append(t.loc[terms[k], "F"])
    for k in null:
        nl = np.array(null[k])
        out[k]["p_perm"] = (1 + np.sum(nl >= out[k]["F"])) / (1 + len(nl))
    out["area"]["p_perm"] = np.nan
    out["n_sessions"], out["n_mice"], out["n_values"] = d.session_id.nunique(), d.mouse_id.nunique(), len(d)
    return out


def anova_line(o):
    return (f"ANOVA cohort {m62.fmt_p(o['cohort']['p_perm'])}, area {m62.fmt_p(o['area']['p'])}, "
            f"cohort × area {m62.fmt_p(o['cohort x area']['p_perm'])}")


def anova_rows(o, **meta):
    return dict(**meta, **{f"{k} {s_}": o[k][s_] for k in ("cohort", "area", "cohort x area") for s_ in ("F", "p", "p_perm")},
                n_sessions=o["n_sessions"], n_mice=o["n_mice"], n_values=o["n_values"])


ANOVA = []          # filled by stage_figure / main, written to area_anova.csv


def stage_figure(plt, A, out, pop, st, sl, gcol, strict=False):
    TT = pd.concat([stage_tests(A, c, st) for c, _ in MEAS], ignore_index=True)
    if strict:                                               # user 2026-10-07: duplicate with only fully sampled areas
        TT = pd.concat([TT[(TT.measure == c) & TT.area_group.isin(strict_areas(A, c))] for c, _ in MEAS], ignore_index=True)
        TT["q_BH"] = pd.concat([pd.Series(m015.bh(TT[TT.measure == c].p_perm), index=TT[TT.measure == c].index) for c, _ in MEAS])
    order = order_of(TT.area_group)
    fig = plt.figure(figsize=(m62.W_IN, 0.25 * len(order) + 1.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.8, 1.3, 0.8], wspace=0.12, left=0.1, right=0.99, top=0.84, bottom=0.11)
    axs = []
    for j, (col, ttl) in enumerate(MEAS):
        ax = fig.add_subplot(gs[0, 2 * j]); axs.append(ax)
        y, T = forest(ax, TT[TT.measure == col], order, j == 0, gcol)
        AN = stage_area_anova(A, [g for g in order if g in set(TT[(TT.measure == col) & TT.p_perm.notna()].area_group)], col, st)
        ANOVA.append(anova_rows(AN, measure=col, stage=st, strict=strict))
        ax.set_title(f"{sl}: drift of {ttl}\n" + anova_line(AN), fontsize=5.4)
        axp = fig.add_subplot(gs[0, 2 * j + 1], sharey=ax)
        pcol(axp, y, T, order)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=COH[c], ms=3, label=c.replace("-", "−")) for c in ("R+", "R-")]
    axs[0].legend(handles=h, frameon=False, fontsize=4.6, loc="lower left")
    m62.letter_row(fig, axs, "ab", dx_in=0.5, dy_in=0.2)
    fig.suptitle(f"Within-session drift on the reward-lick CD per area group, {sl.lower()} sessions ({pop}); mean ± s.e.m. over "
                 "sessions;\nfilled: drift ≠ 0 (Wilcoxon p < 0.05); areas with ≥ 3 sessions per cohort; red: R+ vs R− mouse-level "
                 "permutation p < 0.05, bold: FDR q < 0.05 (Benjamini-Hochberg over the areas shown)", x=0.02, y=0.995, ha="left", va="top", fontsize=5.8, weight="bold")
    m62.save(fig, out, f"area_cd_drift_{'day0' if st == 'learning' else 'expert'}{'_strict' if strict else ''}"); plt.close(fig)
    return TT


def matrix(ax, A, T15, col, order, cbar_label, fig):
    M = np.full((len(order), 4), np.nan); star = np.zeros_like(M, bool)
    groups = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
    for i, g in enumerate(order):
        for j, (c, st) in enumerate(groups):
            v = A[(A.area_group == g) & (A.cohort == c) & (A.stage == st)][col].dropna()
            if len(v) >= 3:
                M[i, j] = v.mean()
                star[i, j] = (v != 0).any() and stats.wilcoxon(v).pvalue < 0.05
    vlim = float(np.nanpercentile(np.abs(M), 95))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(4):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.2f}".replace("-", "−") + ("*" if star[i, j] else ""), ha="center", va="center",
                        fontsize=4.3, color="white" if abs(M[i, j]) > 0.6 * vlim else "k")
    ax.set_xticks(range(4), ["R+ D0", "R+ E", "R− D0", "R− E"], fontsize=5); ax.xaxis.tick_top()
    for t, c in zip(ax.get_xticklabels(), ["R+", "R+", "R-", "R-"]):
        t.set_color(COH[c])
    ax.set_yticks(range(len(order)), [SHORT(g) for g in order], fontsize=5)
    T = T15[T15.measure == col].set_index("area_group").reindex(order)
    keys = ("learning p_perm", "expert p_perm", "cxs p_perm")
    Q = {k: pd.Series(m015.bh(T[k].to_numpy()), index=T.index) for k in keys}     # FDR over the areas in the matrix
    for i, g in enumerate(order):
        if np.isfinite(T.loc[g].get("learning p_perm", np.nan)):
            for k, key in enumerate(keys):
                p_, q_ = T.loc[g, key], Q[key][g]
                ax.text(4.05 + 0.95 * k, i, f"{fp(p_)}\n({fp(q_)})", ha="center", va="center", fontsize=4.0, linespacing=1.0,
                        color="#D62728" if p_ < 0.05 else "0.3", weight="bold" if q_ < 0.05 else "normal", clip_on=False)
    for k, lab in enumerate(("D0", "E", "×")):
        ax.annotate(lab, xy=(4.05 + 0.95 * k, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=4.8, weight="bold")
    ax.annotate("R+ vs R− p (FDR q)", xy=(5.0, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 13), textcoords="offset points", ha="center", va="bottom", fontsize=4.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, location="bottom")
    cb.set_label(cbar_label, fontsize=4.8); cb.ax.tick_params(labelsize=4.5)


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup()
    gcol = au.get_custom_area_groups_colors()
    A0 = pd.read_csv(OUT / "area_sessions.csv")
    for pop in ("learners", "all"):
        A = m61.learner_filter(A0) if pop == "learners" else A0
        out = OUT / pop
        ST = pd.concat([stage_figure(plt, A, out, pop, st, sl, gcol) for st, sl in (("learning", "Day 0"), ("expert", "Expert"))])
        ST.to_csv(out / "area_stage_tests.csv", index=False)
        SS = pd.concat([stage_figure(plt, A, out, pop, st, sl, gcol, strict=True) for st, sl in (("learning", "Day 0"), ("expert", "Expert"))])
        SS.to_csv(out / "area_stage_tests_strict.csv", index=False)
        T15 = pd.read_csv(out / "area_tests.csv")
        Q = set(T15[T15.qualifies_Rplus & T15.qualifies_Rminus].area_group)
        order = order_of(Q)
        fig = plt.figure(figsize=(m62.W_IN * 0.9, 0.2 * len(order) + 1.9))
        gs = fig.add_gridspec(1, 2, wspace=1.5, left=0.09, right=0.84, top=0.76, bottom=0.12)
        axs = []
        for j, (col, ttl) in enumerate(MEAS):
            ax = fig.add_subplot(gs[0, j]); axs.append(ax)
            matrix(ax, A, T15, col, order, f"Mean drift of {ttl.replace(chr(10), ' ')}", fig)
            AN3 = m62.area_anova(A.rename(columns={"area_group": "region"}), order, col=col)
            ANOVA.append(dict(measure=col, stage="both", model="cohort x stage x area",
                              **{f"{k} {s_}": v[s_] for k, v in AN3.items() for s_ in ("F", "p", "p_perm")}))
            ax.set_title(f"Drift of {ttl}\nANOVA: cohort {m62.fmt_p(AN3['cohort']['p_perm'])}, cohort × area "
                         f"{m62.fmt_p(AN3['cohort x area']['p_perm'])},\ncohort × stage {m62.fmt_p(AN3['cohort x stage']['p_perm'])}, "
                         f"cohort × stage × area {m62.fmt_p(AN3['cohort x stage x area']['p_perm'])}", pad=22, fontsize=5.2)
        m62.letter_row(fig, axs, "ab", dx_in=0.45, dy_in=0.35)
        fig.suptitle(f"Within-session drift per area group and group ({pop}); D0 = day 0, E = expert; '*': drift ≠ 0 (Wilcoxon "
                     "p < 0.05);\nareas with ≥ 3 sessions at both stages in both cohorts; p: mouse-level permutation, "
                     "×: cohort × stage interaction;\nq: Benjamini-Hochberg over the areas shown, bold q < 0.05", x=0.02, y=0.995, ha="left", va="top", fontsize=5.6, weight="bold")
        m62.save(fig, out, "area_cd_drift_matrix"); plt.close(fig)
        pd.DataFrame(ANOVA).to_csv(out / "area_anova.csv", index=False); ANOVA.clear()
        with pd.option_context("display.width", 250):
            c = ["measure", "stage", "area_group", "n R+", "mean R+", "n R-", "mean R-", "p_perm", "p_MWU", "p_Welch", "q_BH"]
            print(pop); print(ST[[x for x in c if x in ST]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners")          # both populations are always written
    main(ap.parse_args())
