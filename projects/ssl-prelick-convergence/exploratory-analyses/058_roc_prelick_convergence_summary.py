"""Summary figure: do whisker hits become more like auditory hits before the lick, more in R+ (and where)?

Inputs: 054 session metrics, 056 PSTHs (via its loader), 057 lambda tables. Output:
combined_results_ks4/_roc_prelick/summary/convergence_summary.{png,pdf}
  a  lambda scale: whole-brain mean position of whisker hits between FA (0) and AH (1), per cohort x stage
  b  population first-lick PSTHs (all tested units, Δ rate vs -600..-400 ms): WH, AH, FA per cohort x stage
  c  lambda (whole brain) per session      d  Δd = d(WH,FA) - d(WH,AH) per session
  e  % units WH vs FA significant           f  r(sel WH-FA, sel AH-FA) (RT-matched)
  g  lambda, RT-matched                     h  |sel| WH vs AH (RT-matched)
  i  area groups: change of lambda (expert - learning) per cohort (filled = MWU p < .05), interaction p
     (uncorrected / family-wise max-T)
  j  lambda in motor areas and hippocampus per session
Unit = session; MWU (and Welch) for stage / cohort, mouse-level cohort permutation for the interaction.
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m56 = importlib.import_module("056_roc_prelick_psth")
BASE = m51.OUTROOT
OUT = BASE / "summary"
COH = {"R+": "#00B400", "R-": "#C800C8"}
CCOL = m56.CCOL
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
XS = {GROUPS[0]: 0, GROUPS[1]: 1, GROUPS[2]: 2.4, GROUPS[3]: 3.4}


def fmt(p):
    return "p<.001" if p < 0.001 else f"p={p:.3f}" if p < 0.01 else f"p={p:.2f}"


def dots(ax, d, col, ylabel, tests, rng, title):
    for k, x in XS.items():
        v = d[(d.cohort == k[0]) & (d.stage == k[1])][col].dropna().to_numpy()
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(v)), v, s=5, color=COH[k[0]],
                   alpha=0.35 if k[1] == "learning" else 0.85, lw=0)
        if len(v):
            ax.errorbar(x + 0.28, v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0, fmt="o", ms=3.5,
                        color=COH[k[0]], mfc="white" if k[1] == "learning" else COH[k[0]], lw=1)
    ax.set_xticks(list(XS.values()), ["learn.", "expert", "learn.", "expert"])
    ax.text(0.5, -0.2, "R+", transform=ax.get_xaxis_transform(), ha="center", color=COH["R+"], fontsize=6, weight="bold")
    ax.text(2.9, -0.2, "R−", transform=ax.get_xaxis_transform(), ha="center", color=COH["R-"], fontsize=6, weight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title + "\n" + tests, fontsize=4.8)


def test_text(d, col, rng, n_perm=5000):
    g = {k: d[(d.cohort == k[0]) & (d.stage == k[1])][col].dropna().to_numpy() for k in GROUPS}
    out = []
    for lab, (a, b) in [("R+ E−L", (GROUPS[0], GROUPS[1])), ("R− E−L", (GROUPS[2], GROUPS[3])),
                        ("expert R+−R−", (GROUPS[3], GROUPS[1]))]:
        if len(g[a]) >= 3 and len(g[b]) >= 3:
            out.append(f"{lab}: MWU {fmt(stats.mannwhitneyu(g[a], g[b]).pvalue)}, "
                       f"Welch {fmt(stats.ttest_ind(g[a], g[b], equal_var=False).pvalue)}")
    dd = d.dropna(subset=[col])
    if all(len(v) >= 3 for v in g.values()):
        mice = dd.mouse_id.unique(); mc = dd.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
        midx = pd.Index(mice).get_indexer(dd.mouse_id); st = dd.stage.to_numpy(); y = dd[col].to_numpy()

        def stat(coh):
            mm = {k: y[(coh == k[0]) & (st == k[1])] for k in GROUPS}
            if min(len(v) for v in mm.values()) == 0:
                return np.nan
            return (mm[GROUPS[1]].mean() - mm[GROUPS[0]].mean()) - (mm[GROUPS[3]].mean() - mm[GROUPS[2]].mean())
        obs = stat(dd.cohort.to_numpy())
        null = np.array([stat(rng.permutation(mc)[midx]) for _ in range(n_perm)]); null = null[np.isfinite(null)]
        out.append(f"interaction {obs:+.3f}: perm {fmt((1 + np.sum(np.abs(null) >= abs(obs))) / (1 + len(null)))}")
    return "\n".join(out)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "xtick.labelsize": 5.2, "ytick.labelsize": 5.2, "pdf.fonttype": 42})
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    Lm = pd.read_csv(BASE / "lambda" / "lambda_sessions.csv")
    LT = pd.read_csv(BASE / "lambda" / "lambda_tests.csv")
    S = pd.read_csv(BASE / "convergence" / "session_metrics_all.csv")
    W, P, tc = m56.load()
    fig = plt.figure(figsize=(7.4, 10.2))
    gs = fig.add_gridspec(5, 4, height_ratios=[0.75, 1, 1, 1, 1.15], hspace=0.95, wspace=0.55,
                          left=0.08, right=0.98, top=0.935, bottom=0.04)
    # a: lambda scale
    ax = fig.add_subplot(gs[0, 0:2]); ax.axis("off")
    wb = Lm[(Lm.level == "all") & (Lm.variant == "all")]
    ax.plot([0, 1], [0, 0], color="0.3", lw=1.2, transform=ax.transData)
    ax.set_xlim(-0.3, 1.75); ax.set_ylim(-1.2, 1.45)
    ax.scatter([0, 1], [0, 0], s=40, color=[CCOL["FA"], CCOL["AH"]], zorder=3)
    ax.text(0, -0.45, "FA\n(unrewarded lick)", ha="center", va="top", fontsize=5.5, color=CCOL["FA"])
    ax.text(1, -0.45, "AH\n(rewarded lick)", ha="center", va="top", fontsize=5.5, color=CCOL["AH"])
    for k, (c, s_) in enumerate(GROUPS):
        v = wb[(wb.cohort == c) & (wb.stage == s_)].lam.dropna()
        y = 0.35 + 0.25 * k
        ax.errorbar(v.mean(), y, xerr=v.std(ddof=1) / np.sqrt(len(v)), fmt="o", ms=4, color=COH[c],
                    mfc="white" if s_ == "learning" else COH[c], lw=0.8)
        ax.text(1.12, y, f"{c.replace('-', '−')} {s_}: λ={v.mean():.2f}", fontsize=5, va="center", color=COH[c])
    ax.set_title("a  position of whisker hits between FA (λ=0) and AH (λ=1)\n    whole brain, mean ± SEM over sessions",
                 loc="left", fontsize=6)
    # b: PSTHs
    sub = gs[0, 2:4].subgridspec(1, 4, wspace=0.35)
    tested = (W["sig:auditory_hit_vs_fa_prelick@all"].notna() & W["sig:whisker_hit_vs_fa_prelick@all"].notna()).to_numpy()
    for j, (c, s_) in enumerate(GROUPS):
        ax = fig.add_subplot(sub[j]); m56.deco(ax, tc)
        idx = W.row.to_numpy()[tested & ((W.cohort == c) & (W.stage == s_)).to_numpy()]
        for k, cl in enumerate(m51.CLASSES):
            ax.plot(tc, P[idx, k].mean(0), color=CCOL[cl], lw=0.8, label=cl)
        ax.set_title(f"{c.replace('-', '−')} {s_[:5]}.", fontsize=5.5, color=COH[c]); ax.tick_params(labelsize=4.3)
        ax.set_xticks([-400, 0, 400])
        if j == 0:
            ax.set_ylabel("Δ rate (Hz)", fontsize=5); ax.legend(frameon=False, fontsize=4.3, loc="upper left")
            ax.text(-0.7, 1.12, "b", transform=ax.transAxes, fontsize=8, weight="bold")
    # c-h
    Lwb = Lm[(Lm.level == "all") & (Lm.variant == "all")]; Lrt = Lm[(Lm.level == "all") & (Lm.variant == "rt_matched")]
    Sall = S[S.variant == "all"]; Srt = S[S.variant == "rt_matched"]
    panels = [(gs[1, 0], Lwb, "lam", "λ (WH on FA→AH axis)", "c  λ, whole brain"),
              (gs[1, 1], Lwb, "dd", "d(WH,FA) − d(WH,AH)", "d  Δd (> 0: WH nearer AH)"),
              (gs[1, 2], Sall, "frac_sig_WHvsFA", "fraction sig. WH vs FA", "e  WH vs FA units"),
              (gs[1, 3], Srt, "r_shared", "r(sel WH−FA, sel AH−FA)", "f  shared hit code (RT-matched)"),
              (gs[2, 0], Lrt, "lam", "λ", "g  λ, RT-matched"),
              (gs[2, 1], Srt, "abs_sel_WHvsAH", "|sel| WH vs AH", "h  modality selectivity (RT-matched)")]
    for spec, d, col, yl, title in panels:
        ax = fig.add_subplot(spec)
        dots(ax, d, col, yl, test_text(d, col, rng), rng, title)
    # i: area forest
    ax = fig.add_subplot(gs[2:4, 2:4])
    A = LT[(LT.metric == "lam") & (LT.variant == "all") & (LT.level == "area_group")]
    A = A[A[["stage:R+_diff", "stage:R-_diff"]].notna().any(axis=1)]
    import ephys_utilities.allen_utils.allen_utils as au
    order = [g for g in au.get_area_group_custom_order() if g in set(A.region)]
    A = A.set_index("region").reindex(order)
    y = np.arange(len(order))
    for k, c in enumerate(["R+", "R-"]):
        d_, p_ = A[f"stage:{c}_diff"], A[f"stage:{c}_p_mwu"]
        ax.scatter(d_, y + (k - 0.5) * 0.3, s=16, color=[COH[c] if pp < 0.05 else "white" for pp in p_.fillna(1)],
                   edgecolors=COH[c], lw=0.8, zorder=3, label=f"{c.replace('-', '−')}: expert − learning")
    for i, reg in enumerate(order):
        r = A.loc[reg]
        if np.isfinite(r.get("interaction_p", np.nan)):
            ax.text(1.02, i, f"int. {fmt(r.interaction_p)} / FW {fmt(r.interaction_p_maxT)}  "
                             f"(n E: {int(r['n_R+_expert'])}/{int(r['n_R-_expert'])})",
                    transform=ax.get_yaxis_transform(), fontsize=4.3, va="center")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(y, [g.replace(" areas", "").replace("Somatosensory", "SS") for g in order], fontsize=5); ax.invert_yaxis()
    ax.set_xlabel("Δ λ (expert − learning), session means"); ax.legend(frameon=False, fontsize=5, loc="lower right")
    ax.set_title("i  area groups: Δλ (filled = MWU p < .05)\n    right: interaction p uncorrected / family-wise (max-T)",
                 fontsize=5.5, loc="left")
    ax.set_position([ax.get_position().x0 + 0.04, ax.get_position().y0, ax.get_position().width * 0.55,
                     ax.get_position().height])
    # j: example areas
    for j, reg in enumerate(["Motor areas", "Hippocampus"]):
        ax = fig.add_subplot(gs[3, j])
        d = Lm[(Lm.level == "area_group") & (Lm.variant == "all") & (Lm.region == reg)]
        dots(ax, d, "lam", "λ", test_text(d, "lam", rng), rng, f"j  λ, {reg}" if j == 0 else f"λ, {reg}")
    # k: area PSTH examples (all tested units), expert
    sub = gs[4, :].subgridspec(1, 4, wspace=0.35)
    for j, (reg, (c, s_)) in enumerate([("Motor areas", ("R+", "expert")), ("Motor areas", ("R-", "expert")),
                                        ("Hippocampus", ("R+", "expert")), ("Hippocampus", ("R-", "expert"))]):
        ax = fig.add_subplot(sub[j]); m56.deco(ax, tc)
        idx = W.row.to_numpy()[tested & ((W.cohort == c) & (W.stage == s_) & (W.area_group == reg)).to_numpy()]
        for k, cl in enumerate(m51.CLASSES):
            mu, se = m56.mean_sem(P[idx, k])
            ax.fill_between(tc, mu - se, mu + se, color=CCOL[cl], alpha=0.2, lw=0); ax.plot(tc, mu, color=CCOL[cl], lw=0.8)
        ax.set_title(f"{reg}, {c.replace('-', '−')} {s_} (n={len(idx)})", fontsize=5.5, color=COH[c])
        ax.set_xlabel("ms from first lick", fontsize=5)
        if j == 0:
            ax.set_ylabel("Δ rate (Hz)", fontsize=5)
            ax.text(-0.3, 1.15, "k", transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle("Do whisker hits become more like auditory hits before the lick, more in R+? (pre-lick 100 ms window; "
                 "unit = session)", fontsize=7.2, y=0.997)
    fig.savefig(OUT / "convergence_summary.png", dpi=230); fig.savefig(OUT / "convergence_summary.pdf"); plt.close(fig)
    print("ALL DONE")


if __name__ == "__main__":
    main()
