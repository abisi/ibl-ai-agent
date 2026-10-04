"""115 -- COSYNE abstract figure for the within-session split-half hit/miss decoding, R+ vs R- (user request
2026-09-30). Data: 114_halves_single_decoder.parquet (learning stage; one whole-session decoder per session and area,
held-out predictions scored separately in the 1st and 2nd half; a half is scored only with >= 3 hits and >= 3 misses;
values = balanced accuracy - linear-shift null scored the same way). Unit = session (one learning session per mouse).
Areas excluded (user): Olfactory areas, Cortical subplate, Pons and medulla, Amygdala and hypothalamus, Visual areas,
Somatosensory-body. Six square panels, no schematic:
  a,b  whole brain, 1st vs 2nd half time course (mean +- SEM across sessions with both halves), R+ and R-;
  c    whole brain change (2nd - 1st) time course, R+ vs R- (mean +- SEM);
  d    whole brain in the window W, 1st vs 2nd half per cohort (mean +- SEM; paired Wilcoxon | paired t within cohort; change
       R+ vs R-: Mann-Whitney | Welch);
  e    per area change (2nd - 1st) in the window W, R+ vs R- (mean +- SEM); * = both MW and Welch
       BH-FDR q < 0.05 across areas;
  f    ANOVA across cohort, half and area (sensory window, session x area x half rows): (i) OLS type II three-way
       ANOVA (as in 113) and (ii) linear mixed model with a random intercept per session (Wald chi2 per term, type III
       sum-to-zero contrasts) -- the mixed model respects the repeated measures (halves and areas within a session);
       bars = -log10 p per term, both methods.
Outputs: figures/publication/115_cosyne_halves_figure.pdf/.png, 115_stats.csv (every test with n, statistic, p,
         method), 115_values.csv (per session x area x half values with session_id, subject_id, cohort).
Variants (user 2026-09-30): SOURCE "114" = one whole-session decoder scored per half (area level stores only the
5-50 ms window, so only W = 5-50 there); SOURCE "024" = separate decoder per half (024 "half" condition; full curves at
area level too, so any window, e.g. 5-100 ms; value = real - mean of the linear-shift null curves).
Run (haas): python .../115_cosyne_halves_figure.py [source=114|024] [window=5-50|5-100]
Outputs are suffixed _<source>_<window> (e.g. 115_cosyne_halves_figure_024_5-100.pdf, 115_stats_024_5-100.csv).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
ROOT_REPO = OUT.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
FIGDIR = OUT / "figures" / "publication"
COL = {"R+": "#00B400", "R-": "#C800C8"}
HALF_STYLE = {"first": dict(ls="--", alpha=0.18), "second": dict(ls="-", alpha=0.28)}
EXCLUDE = {"Olfactory areas", "Cortical subplate", "Pons and medulla", "Amygdala and hypothalamus", "Visual areas",
           "Somatosensory-body"}
ABBR = {"Motor areas": "Motor", "Frontal areas": "Frontal", "Somatosensory-orofacial": "SS-orofacial",
        "Somatosensory-whisker": "SS-whisker", "Auditory areas": "Auditory", "Retrosplenial areas": "RSP",
        "Posterior parietal areas": "PPC", "Insular areas": "Insular", "Hippocampus": "HPC", "Striatum": "Striatum",
        "Pallidum": "Pallidum", "Lateral septal complex": "LSX", "Thalamus": "Thalamus", "Midbrain": "Midbrain"}
STATS = []


def style():
    from matplotlib import font_manager
    for f in list(Path.home().glob(".local/share/fonts/arial*.ttf")) + list(Path("C:/Windows/Fonts").glob("arial*.ttf")):
        font_manager.fontManager.addfont(str(f))
    fam = "Arial" if "Arial" in {f.name for f in font_manager.fontManager.ttflist} else "DejaVu Sans"
    plt.rcParams.update({"font.family": fam, "font.size": 7, "axes.titlesize": 7.2, "axes.labelsize": 7,
                         "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "legend.fontsize": 6, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                         "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False})


def allen():
    from axel_bisi_paths import axel_bisi_root
    sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))
    import ephys_utilities.allen_utils.allen_utils as au
    return au.get_custom_area_groups_colors(), au.get_area_group_custom_order()


def bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def rec(**kw):
    STATS.append(kw)


def sem(a, axis=0):
    a = np.asarray(a, float)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.sum(np.isfinite(a), axis=axis))


def stars(q):
    return "**" if q < 0.01 else "*" if q < 0.05 else ""


def load(source, W):
    """Unified frames: wb (session x half, whole-brain curve 'dcurve' = real - null) and ag (session x area x half,
    window value 'y')."""
    if source == "114":
        d = pd.read_parquet(OUT / "114_halves_single_decoder.parquet")
        d = d[d.skipped_reason.isna()].copy()
        t = np.array([b[1] for b in json.load(open(OUT / "114_bin_edges.json"))]) * 1000
        wb = d[d.area_col == "whole_brain"].copy()
        wb["dcurve"] = [np.asarray(r, float) - np.asarray(n, float) for r, n in zip(wb.real_curve, wb.null_mean_curve)]
        ag = d[(d.area_col == "area_group") & ~d.area_value.isin(EXCLUDE)].copy()
        if tuple(W) != (5, 50):
            raise SystemExit("114 stores only the 5-50 ms window at area level; use source 024 for other windows")
        ag["y"] = ag.corr_sensory
        return wb, ag, t
    out = []
    for tag in ("whole_brain", "area_group"):
        f = OUT / f"024_master_results_hitmiss_stim_{tag}.parquet"
        d = pd.read_parquet(f, columns=["session_id", "subject_id", "reward_group", "area_value", "condition_type",
                                        "condition_value", "skipped_reason", "n_units", "n_trials", "real_curve",
                                        "shift_null_curves"])
        d = d[d.skipped_reason.isna() & (d.condition_type == "half")].copy()
        t = np.array([b[1] for b in json.load(open(OUT / f"024_bin_edges_hitmiss_stim_{tag}.json"))]) * 1000
        d["dcurve"] = [np.asarray(r, float) - np.nanmean(np.stack([np.asarray(x, float) for x in L]), 0)
                       for r, L in zip(d.real_curve, d.shift_null_curves)]
        m = (t >= W[0]) & (t <= W[1])
        d["y"] = d.dcurve.map(lambda c: float(np.nanmean(c[m])))
        d = d.drop(columns=["real_curve", "shift_null_curves"])
        out.append((d, t))
    (wb, t), (ag, _) = out
    ag = ag[~ag.area_value.isin(EXCLUDE)].copy()
    return wb, ag, t


def paired_curves(wb, rg):
    g = wb[wb.reward_group == rg]
    f = g[g.condition_value == "first"].set_index("session_id")["dcurve"]
    s = g[g.condition_value == "second"].set_index("session_id")["dcurve"]
    ids = sorted(set(f.index) & set(s.index))
    return np.stack([f[i] for i in ids]), np.stack([s[i] for i in ids]), ids


def win_mean(C, t, w):
    m = (t >= w[0]) & (t <= w[1])
    return np.nanmean(C[:, m], 1)


def main():
    style()
    source = sys.argv[1] if len(sys.argv) > 1 else "114"
    wtxt = sys.argv[2] if len(sys.argv) > 2 else "5-50"
    W = tuple(float(v) for v in wtxt.split("-"))
    tag = f"{source}_{wtxt}"
    colors, order = allen()
    wb, ag, t = load(source, W)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.1))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.9, bottom=0.1, hspace=0.6, wspace=0.6)
    for ax in axes.flat:
        ax.set_box_aspect(1)
    (a_, b_, c_), (d_, e_, f_) = axes
    # a, b: halves time course per cohort
    ylim = [np.inf, -np.inf]
    for ax, rg in ((a_, "R+"), (b_, "R-")):
        F, S, ids = paired_curves(wb, rg)
        for lab, M in (("first", F), ("second", S)):
            mu, se = np.nanmean(M, 0), sem(M)
            ax.fill_between(t, mu - se, mu + se, color=COL[rg], alpha=HALF_STYLE[lab]["alpha"], lw=0)
            ax.plot(t, mu, color=COL[rg], ls=HALF_STYLE[lab]["ls"], lw=0.9, label=f"{lab} half")
            ylim = [min(ylim[0], np.nanmin(mu - se)), max(ylim[1], np.nanmax(mu + se))]
        ax.set_title(f"whole brain, {rg} ({len(ids)} sessions)", pad=6)
        ax.legend(loc="upper left", handlelength=1.8)
    for ax in (a_, b_):
        ax.set_ylim(ylim[0] - 0.02, ylim[1] + 0.02)
        ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
        ax.axvline(0, color="k", lw=0.5)
        ax.axvspan(*W, color="0.9", lw=0, zorder=0)
        ax.set_xlim(t[0], t[-1])
        ax.set_xlabel("time from stimulus (ms)")
    a_.set_ylabel("hit vs miss\nbalanced accuracy - null")
    # c: change over time, R+ vs R-
    for rg in ("R+", "R-"):
        F, S, ids = paired_curves(wb, rg)
        D = S - F
        mu, se = np.nanmean(D, 0), sem(D)
        c_.fill_between(t, mu - se, mu + se, color=COL[rg], alpha=0.22, lw=0)
        c_.plot(t, mu, color=COL[rg], lw=0.9, label=f"{rg} (n = {len(ids)})")
    c_.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    c_.axvline(0, color="k", lw=0.5)
    c_.axvspan(*W, color="0.9", lw=0, zorder=0)
    c_.set_xlim(t[0], t[-1])
    c_.set_xlabel("time from stimulus (ms)")
    c_.set_ylabel("change, 2nd - 1st half")
    c_.set_title("whole brain, within-session change", pad=6)
    c_.legend(loc="upper left", handlelength=1.4)
    # d: whole-brain window, halves per cohort + tests
    chg = {}
    for k, rg in enumerate(("R+", "R-")):
        F, S, ids = paired_curves(wb, rg)
        f, s = win_mean(F, t, W), win_mean(S, t, W)
        chg[rg] = s - f
        x0 = k * 2.2
        for j, (v, lab) in enumerate(((f, "1st"), (s, "2nd"))):
            d_.bar(x0 + j, v.mean(), yerr=sem(v), color=COL[rg], alpha=0.45 if j == 0 else 0.9, width=0.8, lw=0,
                   error_kw=dict(lw=0.6, capsize=0))
        pw, pt = stats.wilcoxon(s, f).pvalue, stats.ttest_rel(s, f).pvalue
        rec(panel="d", test=f"1st vs 2nd half, whole brain {wtxt} ms", cohort=rg, n=len(ids), mean_first=f.mean(),
            mean_second=s.mean(), p_wilcoxon=pw, p_paired_t=pt)
        top = max(f.mean() + sem(f), s.mean() + sem(s))
        d_.text(x0 + 0.5, top + 0.012, f"W p={pw:.2g}\nt p={pt:.2g}", ha="center", fontsize=5)
    pm = stats.mannwhitneyu(chg["R+"], chg["R-"]).pvalue
    pwl = stats.ttest_ind(chg["R+"], chg["R-"], equal_var=False).pvalue
    rec(panel="d", test=f"change (2nd - 1st) R+ vs R-, whole brain {wtxt} ms", n_rplus=len(chg["R+"]), n_rminus=len(chg["R-"]),
        mean_change_rplus=chg["R+"].mean(), mean_change_rminus=chg["R-"].mean(), p_mannwhitney=pm, p_welch=pwl)
    d_.set_xticks([0, 1, 2.2, 3.2], ["1st", "2nd", "1st", "2nd"])
    d_.set_xlabel(f"R+                R\u2212\nchange R+ vs R\u2212: MW p={pm:.2g} | Welch p={pwl:.2g}")
    d_.set_ylabel(f"balanced accuracy - null\n(whole brain, {wtxt} ms)")
    d_.set_title("whole brain, 1st vs 2nd half", pad=6)
    d_.set_ylim(0, d_.get_ylim()[1] * 1.25)
    d_.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    # e: per-area change, R+ vs R-
    piv = ag.pivot_table(index=["session_id", "subject_id", "reward_group", "area_value"], columns="condition_value",
                         values="y").dropna().reset_index()
    piv["change"] = piv["second"] - piv["first"]
    areas = [a for a in order if a in set(piv.area_value)]
    rows = []
    for a in areas:
        x, y = (piv[(piv.area_value == a) & (piv.reward_group == rg)].change.to_numpy() for rg in ("R+", "R-"))
        if len(x) < 2 or len(y) < 2:
            continue
        rows.append(dict(area=a, n_rplus=len(x), n_rminus=len(y), m_rplus=x.mean(), se_rplus=sem(x), m_rminus=y.mean(),
                         se_rminus=sem(y), p_mw=stats.mannwhitneyu(x, y).pvalue,
                         p_welch=stats.ttest_ind(x, y, equal_var=False).pvalue))
    E = pd.DataFrame(rows)
    E["q_mw"], E["q_welch"] = bh(E.p_mw), bh(E.p_welch)
    for r in E.itertuples():
        rec(panel="e", test=f"change (2nd - 1st) R+ vs R-, window {wtxt} ms", area=r.area, n_rplus=r.n_rplus,
            n_rminus=r.n_rminus, mean_change_rplus=r.m_rplus, mean_change_rminus=r.m_rminus, p_mannwhitney=r.p_mw,
            p_welch=r.p_welch, q_mannwhitney_bh=r.q_mw, q_welch_bh=r.q_welch)
    E = E.sort_values("m_rplus").reset_index(drop=True)
    for k, r in E.iterrows():
        e_.plot([r.m_rplus, r.m_rminus], [k, k], color="0.8", lw=0.8, zorder=1)
        e_.errorbar(r.m_rplus, k + 0.14, xerr=r.se_rplus, fmt="o", ms=2.4, color=COL["R+"], elinewidth=0.5, capsize=0)
        e_.errorbar(r.m_rminus, k - 0.14, xerr=r.se_rminus, fmt="o", ms=2.4, color=COL["R-"], elinewidth=0.5, capsize=0)
        s = stars(max(r.q_mw, r.q_welch))
        if s:
            e_.text(1.0, k, s, transform=e_.get_yaxis_transform(), va="center", fontsize=6.5)
    e_.set_yticks(range(len(E)))
    e_.set_yticklabels([f"{ABBR.get(a, a)} ({n1}/{n2})" for a, n1, n2 in zip(E.area, E.n_rplus, E.n_rminus)], fontsize=4.8)
    for lab, a in zip(e_.get_yticklabels(), E.area):
        lab.set_color(colors.get(a, "0.3"))
    e_.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
    e_.tick_params(axis="y", length=0)
    e_.set_ylim(-0.7, len(E) - 0.3)
    e_.set_xlabel(f"change, 2nd - 1st half\n({wtxt} ms)")
    e_.set_title("per area (sessions R+/R\u2212)", pad=6)
    # f: ANOVA across cohort x half x area
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    L = ag[ag.area_value.isin(areas)].rename(columns={"reward_group": "cohort", "condition_value": "half",
                                                       "area_value": "area"})
    L = L[["session_id", "subject_id", "cohort", "half", "area", "y"]].dropna()
    ols = smf.ols("y ~ C(cohort) * C(half) * C(area)", data=L).fit()
    A = anova_lm(ols, typ=2)
    terms = {"C(cohort)": "cohort", "C(half)": "half", "C(area)": "area", "C(cohort):C(half)": "cohort x half",
             "C(cohort):C(area)": "cohort x area", "C(half):C(area)": "half x area",
             "C(cohort):C(half):C(area)": "cohort x half x area"}
    p_ols = {terms[k]: A.loc[k, "PR(>F)"] for k in terms}
    for k, lab in terms.items():
        rec(panel="f", test=f"three-way ANOVA (OLS type II), window {wtxt} ms", term=lab, F=A.loc[k, "F"],
            df=A.loc[k, "df"], df_resid=A.loc["Residual", "df"], p=A.loc[k, "PR(>F)"], n_rows=len(L),
            n_sessions=L.session_id.nunique())
    lmm = smf.mixedlm("y ~ C(cohort, Sum) * C(half, Sum) * C(area, Sum)", data=L, groups=L["session_id"]).fit(reml=True)
    names = list(lmm.params.index)                 # fixed effects + "Group Var" (random-intercept variance)
    p_lmm = {}
    var_of = lambda piece: piece.split("(", 1)[1].split(",")[0].split(")")[0]      # "C(cohort, Sum)[S.R+]" -> "cohort"
    for k, lab in terms.items():
        want = sorted(var_of(p) for p in k.split(":"))
        idx = [i for i, n in enumerate(names) if n.startswith("C(") and sorted(var_of(p) for p in n.split(":")) == want]
        R = np.zeros((len(idx), len(names)))
        R[np.arange(len(idx)), idx] = 1
        wt = lmm.wald_test(R, scalar=True)
        p_lmm[lab] = float(wt.pvalue)
        rec(panel="f", test=f"linear mixed model, random intercept per session (Wald), window {wtxt} ms", term=lab,
            chi2=float(wt.statistic), df=len(idx), p=float(wt.pvalue), n_rows=len(L), n_sessions=L.session_id.nunique())
    labs = list(terms.values())
    yy = np.arange(len(labs))[::-1]
    lp = lambda p: -np.log10(max(p, 1e-12))
    f_.barh(yy + 0.18, [lp(p_ols[l]) for l in labs], height=0.34, color="0.55", lw=0, label="OLS ANOVA (type II)")
    f_.barh(yy - 0.18, [lp(p_lmm[l]) for l in labs], height=0.34, color="0.15", lw=0, label="mixed model (session)")
    f_.axvline(-np.log10(0.05), color="#c0392b", lw=0.6, ls="--")
    f_.set_yticks(yy)
    f_.set_yticklabels(labs, fontsize=5.2)
    f_.tick_params(axis="y", length=0)
    f_.set_xlabel("-log10 p")
    f_.set_title(f"cohort x half x area ({L.session_id.nunique()} sessions)", pad=6)
    f_.legend(loc="lower right", fontsize=5, handlelength=1.0)
    for ax, L_ in zip(axes.flat, "abcdef"):
        ax.text(-0.42 if ax in (e_, f_) else -0.3, 1.1, L_, transform=ax.transAxes, fontweight="bold", fontsize=9)
    meth = ("one decoder per session (all trials), held-out predictions scored per half" if source == "114" else
            "separate decoder per session half")
    fig.suptitle(f"Within-session change of hit vs miss decoding, learning day: {meth}\nbalanced accuracy - linear-shift "
                 f"null; mean \u00b1 SEM across sessions; window {wtxt} ms (grey); * = Mann-Whitney and Welch BH q < 0.05",
                 fontsize=6, y=0.995)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"115_cosyne_halves_figure_{tag}.{ext}", dpi=300)
    src = "114_halves_single_decoder.parquet" if source == "114" else "024_master_results_hitmiss_stim_{whole_brain,area_group}.parquet (half)"
    pd.DataFrame(STATS).assign(source=src, window_ms=wtxt, stage="learning",
                               excluded_areas=";".join(sorted(EXCLUDE))).to_csv(OUT / f"115_stats_{tag}.csv", index=False)
    ag[["session_id", "subject_id", "reward_group", "area_value", "condition_value", "n_units", "y"]].rename(
        columns={"y": f"acc_minus_null_{wtxt}ms"}).assign(source=src).to_csv(OUT / f"115_values_{tag}.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(STATS).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
