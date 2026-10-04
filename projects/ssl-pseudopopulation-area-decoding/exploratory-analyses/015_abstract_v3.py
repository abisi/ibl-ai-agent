"""015 -- COSYNE abstract figure v3 (user 2026-09-30, from 014 v2): stimulus-aligned modality panel and onset latency
removed; 2 rows x 4 square panels, one row per decoding type, same panel types in both rows.
  a / e  area overlay (pseudo-populations, R+; real - paired shift null, mean over iterations) + R+ reaction-time band;
         row 1 hit vs miss zoomed on the first 150 ms after the stimulus; row 2 whisker vs auditory, first-lick-aligned
         (full -600..200 ms);
  b / f  session halves, whole brain, single-session decoding with a separate decoder per half (024 "half"): the square
         cell holds two tall sub-plots side by side, R+ and R-, 1st half light / 2nd half dark cohort shade, mean +- SEM
         across sessions, window shaded, that cohort's reaction-time histogram; zoom -100..300 ms (stimulus) and
         -300..100 ms (first lick); no difference trace (user);
  c / g  window value per half, group mean +- SEM; within cohort paired Wilcoxon (1st vs 2nd); between cohorts
         Mann-Whitney on the per-session change (user: keep Mann-Whitney; paired t, the mixed-ANOVA interaction and
         Welch are still written to the csv);
  d / h  per area: split-half change computed per session (2nd - 1st, window value), mean +- SEM per area and cohort,
         scatter R+ change (x) vs R- change (y), one dot per area group (area colour), dashed identity line; ring =
         per-area Mann-Whitney R+ vs R- on the change p < 0.05 (uncorrected); LMM (session random intercept) cohort x
         half and cohort x half x area p in the corner.
Windows: hit vs miss 5-100 ms, whisker vs auditory -100..0 ms from the first lick. Areas: 005 EXCLUDE.
Outputs: figures/015_abstract_v3.pdf/.png, figures/015_panels/*, ../artifacts/015_stats.csv
Run (haas, repo root): python projects/ssl-pseudopopulation-area-decoding/exploratory-analyses/015_abstract_v3.py [n_iter]
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
ART, FIG = HERE.parent / "artifacts", HERE / "figures"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V2 = _load("v2", "014_abstract_v2.py")          # shared helpers: halves024, wb_pairs, shade, sem, pfmt, COL, CLAB
S003, S005, S011, S012 = V2.S003, V2.S005, V2.S011, V2.S012
COL, CLAB, shade, sem, pfmt = V2.COL, V2.CLAB, V2.shade, V2.sem, V2.pfmt
ROW = {"hitmiss": dict(win=(5, 100), wkey="sensory100", wlab="5-100 ms", ref="stimulus", zoom=(-100, 300),
                       overlay_view=(0, 150), label="Hit vs miss"),
       "modality_lick": dict(win=(-100, 0), wkey="prelick", wlab="-100-0 ms", ref="first lick", zoom=(-300, 100),
                             overlay_view=(-300, 5), label="Whisker vs auditory\n(lick-aligned)")}
V2.ROW.update({k: dict(V2.ROW[k], **{kk: v[kk] for kk in ("win", "wkey", "wlab")}) for k, v in ROW.items()})
STATS = []
REC = {"on": True}
FS_L, FS_M, FS_S = 8, 6, 5          # the only three font sizes (user 2026-09-30)
TITLE_PAD = 11                       # same title height in every panel of a row
YLAB_X = -0.30                       # y-label x position (axes fraction), identical per column across rows


def rec(**kw):
    if REC["on"]:
        STATS.append(kw)


def p_overlay(ax, X, colors, target):
    R = ROW[target]
    S011.tc(ax, X, colors, R["overlay_view"], R["ref"], f"{R['label']}\nacc. − null",
            S012.rt_values(target, "R+"))
    ax.set_title("areas, R+ (pseudo-pop.)" + (", first 150 ms" if target == "hitmiss" else ", 300 ms pre-lick"), fontsize=FS_M, pad=TITLE_PAD,
                 color=COL["R+"])


def p_halves_split(ax, d, t, target):
    R = ROW[target]
    view = R["zoom"]
    z = (t >= view[0]) & (t <= view[1])
    ax.set_axis_off()
    subs = {"R+": ax.inset_axes([0.0, 0.0, 0.44, 1.0]), "R-": ax.inset_axes([0.56, 0.0, 0.44, 1.0])}
    subs["R-"].sharey(subs["R+"])
    for rg, sa in subs.items():
        g = d[(d.area_col == "whole_brain") & (d.reward_group == rg)].copy()
        g["dc"] = [np.asarray(r, float) - np.asarray(n, float) for r, n in zip(g.real_curve, g.null_mean_curve)]
        F = g[g.condition_value == "first"].set_index("session_id").dc
        S = g[g.condition_value == "second"].set_index("session_id").dc
        ids = sorted(set(F.index) & set(S.index))
        sa.axvspan(*R["win"], color="0.9", lw=0, zorder=0)
        for M, f, lab in ((np.stack([F[i] for i in ids]), 0.35, "1st"), (np.stack([S[i] for i in ids]), 1.0, "2nd")):
            mu, se = np.nanmean(M, 0), sem(M)
            sa.fill_between(t[z], (mu - se)[z], (mu + se)[z], color=shade(COL[rg], f), alpha=0.28, lw=0)
            sa.plot(t[z], mu[z], color=shade(COL[rg], f), lw=0.9, label=f"{lab} half")
        sa.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
        sa.axvline(0, color="k", lw=0.5)
        sa.set_xlim(*view)
        sa.tick_params(labelsize=FS_S)
        sa.set_title(f"{CLAB[rg]} (n = {len(ids)})", fontsize=FS_M, color=COL[rg], pad=2)
        sa.legend(loc="upper left", fontsize=FS_S, handlelength=1.0, borderaxespad=0.1, labelspacing=0.15)
    subs["R+"].set_ylabel("whole brain, acc. − null", fontsize=FS_M)
    subs["R-"].tick_params(labelleft=False)
    for rg, sa in subs.items():
        S012.add_rt_band(sa, [(S012.rt_values(target, rg), shade(COL[rg], 0.6))], view)
        sa.set_xticks([0, 200] if target == "hitmiss" else [-200, 0])
    subs["R+"].text(1.06, -0.2, f"time from {R['ref']} (ms)", transform=subs["R+"].transAxes, ha="center", va="top",
                    fontsize=FS_M)
    ax.set_title(f"session halves ({R['wlab']} shaded)", fontsize=FS_M, pad=TITLE_PAD)


def p_halves_stats(ax, d, target):
    R = ROW[target]
    key = f"corr_{R['wkey']}"
    chg, xs, tops = {}, {"R+": (0, 1), "R-": (2.4, 3.4)}, []
    for rg in ("R+", "R-"):
        f, s, ids = V2.wb_pairs(d, rg, key)
        ok = np.isfinite(f.to_numpy()) & np.isfinite(s.to_numpy())
        f, s = f.to_numpy()[ok], s.to_numpy()[ok]
        chg[rg] = s - f
        x0, x1 = xs[rg]
        ax.plot([x0, x1], [f.mean(), s.mean()], color=COL[rg], lw=0.8, zorder=1)
        for x, v, fsh in ((x0, f, 0.35), (x1, s, 1.0)):
            ax.errorbar(x, v.mean(), yerr=sem(v), fmt="o", ms=4, color=shade(COL[rg], fsh), elinewidth=0.9, capsize=1.5,
                        mew=0, zorder=2)
        pw, pt = stats.wilcoxon(s, f).pvalue, stats.ttest_rel(s, f).pvalue
        tops.append((x0, x1, max(f.mean() + sem(f), s.mean() + sem(s)), pw, rg, len(f)))
        rec(row=target, panel="c/g", test=f"1st vs 2nd half, whole brain {R['wlab']}", cohort=rg, n=len(f),
            mean_first=f.mean(), sem_first=sem(f), mean_second=s.mean(), sem_second=sem(s), p_wilcoxon=pw,
            p_paired_t_csv_only=pt)
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    for x0, x1, tp, pw, rg, n in tops:
        ax.plot([x0, x1], [tp + 0.06 * span] * 2, color=COL[rg], lw=0.6)
        ax.text((x0 + x1) / 2, tp + 0.08 * span, f"Wilcoxon\n{pfmt(pw)}", ha="center", va="bottom", fontsize=FS_S,
                color=COL[rg])
        ax.text((x0 + x1) / 2, y0 + 0.02 * span, f"n = {n}", ha="center", va="bottom", fontsize=FS_S, color=COL[rg])
    pm = stats.mannwhitneyu(chg["R+"], chg["R-"]).pvalue
    tt = stats.ttest_ind(chg["R+"], chg["R-"], equal_var=True)
    pwl = stats.ttest_ind(chg["R+"], chg["R-"], equal_var=False).pvalue
    rec(row=target, panel="c/g", test="change (2nd - 1st) R+ vs R-, whole brain", n_rplus=len(chg["R+"]),
        n_rminus=len(chg["R-"]), mean_change_rplus=chg["R+"].mean(), mean_change_rminus=chg["R-"].mean(),
        p_mannwhitney=pm, p_welch_csv_only=pwl, F_mixed_anova_csv_only=tt.statistic ** 2, p_mixed_anova_csv_only=tt.pvalue)
    ax.set_ylim(y0, y1 + 0.5 * span)
    ax.set_xticks([0, 1, 2.4, 3.4], ["1st", "2nd", "1st", "2nd"])
    for lab, c in zip(ax.get_xticklabels(), ["R+", "R+", "R-", "R-"]):
        lab.set_color(COL[c])
    ax.set_xlim(-0.6, 4.0)
    ax.set_ylabel(f"acc. − null, {R['wlab']}")
    ax.set_xlabel(f"change {CLAB['R+']} vs {CLAB['R-']}: Mann-Whitney {pfmt(pm)}", fontsize=FS_S)
    ax.set_title("whole brain, mean ± SEM", fontsize=FS_M, pad=TITLE_PAD)


def p_area_scatter(ax, d, target, colors, order):
    R = ROW[target]
    key = f"corr_{R['wkey']}"
    a = d[(d.area_col == "area_group") & ~d.area_value.isin(S005.EXCLUDE)]
    P = a.pivot_table(index=["session_id", "reward_group", "area_value"], columns="condition_value", values=key).dropna()
    P = P.reset_index()
    P["change"] = P["second"] - P["first"]
    areas = [x for x in order if x in set(P.area_value)]
    lim = 0
    for ar in areas:
        x = P[(P.area_value == ar) & (P.reward_group == "R+")].change.to_numpy()
        y = P[(P.area_value == ar) & (P.reward_group == "R-")].change.to_numpy()
        if len(x) < 2 or len(y) < 2:
            continue
        c = colors.get(ar, "0.5")
        pm = stats.mannwhitneyu(x, y).pvalue
        rec(row=target, panel="d/h", test=f"per-area change R+ vs R- ({R['wlab']})", area=ar, n_rplus=len(x),
            n_rminus=len(y), mean_change_rplus=x.mean(), sem_rplus=sem(x), mean_change_rminus=y.mean(), sem_rminus=sem(y),
            p_mannwhitney=pm)
        ax.errorbar(x.mean(), y.mean(), xerr=sem(x), yerr=sem(y), fmt="none", ecolor=shade(c, 0.6), elinewidth=0.6,
                    capsize=0, zorder=1)
        ax.scatter(x.mean(), y.mean(), s=16, color=c, lw=0, zorder=3)
        if pm < 0.05:
            ax.scatter(x.mean(), y.mean(), s=42, facecolor="none", edgecolor="k", lw=0.6, zorder=4)
        lim = max(lim, abs(x.mean()) + sem(x), abs(y.mean()) + sem(y))
    lim *= 1.12
    ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=0.5, ls="--", zorder=0)
    ax.axhline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.axvline(0, color="#bbbbbb", lw=0.5, ls=":")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(f"{CLAB['R+']} change, 2nd − 1st", color=COL["R+"])
    ax.set_ylabel(f"{CLAB['R-']} change, 2nd − 1st", color=COL["R-"])
    import statsmodels.formula.api as smf
    L = P.melt(id_vars=["session_id", "reward_group", "area_value"], value_vars=["first", "second"], var_name="half",
               value_name="y").rename(columns={"reward_group": "cohort", "area_value": "area"})
    m = smf.mixedlm("y ~ C(cohort, Sum) * C(half, Sum) * C(area, Sum)", L, groups=L["session_id"]).fit(reml=True)
    names = list(m.params.index)
    var_of = lambda piece: piece.split("(", 1)[1].split(",")[0].split(")")[0]
    out = {}
    for lab, want in (("cohort x half", ["cohort", "half"]), ("cohort x half x area", ["area", "cohort", "half"])):
        idx = [i for i, n in enumerate(names) if n.startswith("C(") and sorted(var_of(p) for p in n.split(":")) == want]
        Rm = np.zeros((len(idx), len(names)))
        Rm[np.arange(len(idx)), idx] = 1
        w = m.wald_test(Rm, scalar=True)
        out[lab] = float(w.pvalue)
        rec(row=target, panel="d/h", test=f"LMM (session random intercept) Wald, {R['wlab']}", term=lab,
            chi2=float(w.statistic), df=len(idx), p=float(w.pvalue), n_rows=len(L), n_sessions=L.session_id.nunique())
    ax.text(0.97, 0.03, f"LMM cohort×half {pfmt(out['cohort x half'])}\ncohort×half×area "
                        f"{pfmt(out['cohort x half x area'])}\n○ per-area MW p < 0.05",
            transform=ax.transAxes, fontsize=FS_S, va="bottom", ha="right", linespacing=1.1)
    ax.set_title("per area, split-half change", fontsize=FS_M, pad=TITLE_PAD)


def main():
    import ssl_timeresolved_decoding as T
    S003.style()
    plt.rcParams.update({"svg.fonttype": "none", "font.size": FS_M, "axes.titlesize": FS_M, "axes.labelsize": FS_M, "xtick.labelsize": FS_S,
                         "ytick.labelsize": FS_S, "legend.fontsize": FS_S})
    n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    colors, order = S005.allen()
    fig, axes = plt.subplots(2, 4, figsize=(8.27, 4.55))          # A4 width (210 mm)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.16, hspace=0.72, wspace=0.62)
    for ax in axes.flat:
        ax.set_box_aspect(1)
    X = {tg: S011.compute("R+", n_iter, order, T)[tg] for tg in ("hitmiss", "modality_lick")}
    D = {tg: V2.halves024(tg) for tg in ("hitmiss", "modality_lick")}
    for r, tg in enumerate(("hitmiss", "modality_lick")):
        a, b, c, e = axes[r]
        p_overlay(a, X[tg], colors, tg)
        p_halves_split(b, *D[tg], tg)
        p_halves_stats(c, D[tg][0], tg)
        p_area_scatter(e, D[tg][0], tg, colors, order)
    for ax in axes[:, [0, 2, 3]].flat:                      # y labels at the same x in every column
        ax.yaxis.set_label_coords(YLAB_X, 0.5)
    for ax in axes[:, 1]:
        ax.child_axes[0].yaxis.set_label_coords(YLAB_X / 0.44, 0.5)
    for ax, L in zip(axes.flat, "abcdefgh"):
        ax.text(-0.38, 1.16, L, transform=ax.transAxes, fontweight="bold", fontsize=FS_L)
    names = [x for x in order if x not in S005.EXCLUDE]
    handles = [Patch(color=colors.get(x, "0.6"), label=S005.ABBR.get(x, x)) for x in names]
    handles += [Patch(color="#b0b0b0", label="reaction times")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=FS_S, handlelength=0.7, handleheight=0.6,
               handletextpad=0.25, columnspacing=0.55, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Learning day. a, e: pseudo-populations (20 mice x 10 neurons, {n_iter} iterations x 10 shifts"
                 f"{', PILOT' if n_iter < 1000 else ''}); b-d, f-h: single sessions, separate decoder per session half;\n"
                 "acc. − null = balanced accuracy − linear-shift null", fontsize=FS_S, y=0.998)
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"015_abstract_v3.{ext}", dpi=300)
    plt.close(fig)
    REC["on"] = False
    pan = FIG / "015_panels"
    pan.mkdir(exist_ok=True)
    for tg, pre in (("hitmiss", "abcd"), ("modality_lick", "efgh")):
        jobs = {f"{pre[0]}_{tg}_overlay": lambda ax, tg=tg: p_overlay(ax, X[tg], colors, tg),
                f"{pre[1]}_{tg}_halves": lambda ax, tg=tg: p_halves_split(ax, *D[tg], tg),
                f"{pre[2]}_{tg}_halves_stats": lambda ax, tg=tg: p_halves_stats(ax, D[tg][0], tg),
                f"{pre[3]}_{tg}_area_scatter": lambda ax, tg=tg: p_area_scatter(ax, D[tg][0], tg, colors, order)}
        for name, fn in jobs.items():
            f1, ax1 = plt.subplots(figsize=(2.5, 2.5))
            ax1.set_box_aspect(1)
            fn(ax1)
            f1.tight_layout()
            for ext in ("pdf", "png", "svg"):
                f1.savefig(pan / f"015_{name}.{ext}", dpi=300)
            plt.close(f1)
    S = pd.DataFrame(STATS)
    S.to_csv(ART / "015_stats.csv", index=False)
    pd.set_option("display.width", 250)
    keep = S[S.test.str.contains("LMM|whole brain")]
    print(keep.dropna(axis=1, how="all").round(4).to_string(index=False))
    sig = S[S.test.str.contains("per-area") & (S.p_mannwhitney < 0.05)]
    print("per-area MW p<0.05:", sig[["row", "area", "mean_change_rplus", "mean_change_rminus", "p_mannwhitney"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
