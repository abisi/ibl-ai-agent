"""Learning-stage x area summary of the pre-lick ROC types (047 run on 053's table; all units passing min FR).

Figures -> combined_results_ks4/ssl-prelick-convergence/across_days/fa/figures/
  P1_overview.png       measures (type x trial variant) x comparisons: change in % significant and in mean |sel|
                        (pooled over areas), stars = permutation p (047: stage labels across sessions, cohort labels
                        across mice), uncorrected
  P1b_overview_sign.png same for % positive / % negative units
  P2_pooled.png         pooled % significant (stacked by sign; three-class unsigned) and mean |sel| per cohort x stage,
                        session dots, for each type; one row per trial variant
  P3_area_groups_<variant>.png  area-group change (expert - learning) per cohort for each type (% sig.), stars = perm. p
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
st47 = importlib.import_module("047_roc_stage_stats")
f48 = importlib.import_module("048_roc_stage_type_figures")
m51 = importlib.import_module("051_roc_prelick")
BASE = m51.OUTROOT
FIG = BASE / "figures"
TYPES = ["wh_vs_aud_hit_prelick", "whisker_hit_vs_fa_prelick", "auditory_hit_vs_fa_prelick"]   # three-class removed
TLAB = {"wh_vs_aud_hit_prelick": "WH vs AH (+ = AH > WH)", "whisker_hit_vs_fa_prelick": "FA vs WH (+ = WH > FA)",
        "auditory_hit_vs_fa_prelick": "FA vs AH (+ = AH > FA)", "three_class_prelick": "three-class D3"}
VARIANTS = ["all", "long_rt", "rt_matched", "short_rt"]
COMPS = [("stage:R+", "R+: expert−learning"), ("stage:R-", "R−: expert−learning"), ("interaction", "interaction"),
         ("cohort:learning", "learning: R+−R−"), ("cohort:expert", "expert: R+−R−")]


def pstar(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def heat(plt, A, metrics, name, title):
    rows = [f"{t}@{v}" for t in TYPES for v in VARIANTS]
    fig, axs = plt.subplots(1, len(metrics), figsize=(7.4, 5.2), gridspec_kw=dict(wspace=0.9))
    for ax, (m, lab, sc) in zip(axs, metrics):
        Mx = np.full((len(rows), len(COMPS)), np.nan); ann = [[""] * len(COMPS) for _ in rows]
        for j, (c, _) in enumerate(COMPS):
            cc = f"interaction:{m}" if c == "interaction" else c
            sub = A[A.comparison == cc].set_index("measure")
            if f"{m}_diff" not in sub:
                continue
            for i, r in enumerate(rows):
                if r in sub.index and np.isfinite(sub.loc[r, f"{m}_diff"]):
                    d, p = sub.loc[r, f"{m}_diff"] * sc, sub.loc[r, f"{m}_p"]
                    Mx[i, j] = d; ann[i][j] = (f"{d:+.1f}" if sc == 100 else f"{d:+.3f}") + pstar(p)
        v = np.nanpercentile(np.abs(Mx), 95) if np.isfinite(Mx).any() else 1
        ax.imshow(Mx, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(COMPS)):
                ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=4.2)
        ax.set_yticks(range(len(rows)), [r.replace("_prelick@", " @ ") for r in rows], fontsize=4.8)
        ax.set_xticks(range(len(COMPS)), [c[1] for c in COMPS], rotation=40, ha="right", fontsize=5)
        for y in range(len(VARIANTS), len(rows), len(VARIANTS)):
            ax.axhline(y - 0.5, color="k", lw=0.6)
        ax.set_title(lab, fontsize=6)
    fig.suptitle(title, fontsize=7)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.9, bottom=0.14)
    fig.savefig(FIG / f"{name}.png", dpi=220); fig.savefig(FIG / f"{name}.pdf"); plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "xtick.labelsize": 5.2, "ytick.labelsize": 5.2, "pdf.fonttype": 42, "hatch.linewidth": 0.4})
    FIG.mkdir(parents=True, exist_ok=True)
    R = pd.read_csv(BASE / "stats_all" / "region_stats.csv")
    A = R[(R.level == "all") & (R.region == "all")]
    heat(plt, A, [("frac_sig", "Δ % significant", 100), ("mean_abs_sel", "Δ mean |selectivity| (3-class: D3)", 1)],
         "P1_overview", "Pre-lick ROC: learning-stage and cohort effects (all units, min FR 0.1 Hz; * perm. p, uncorrected)")
    heat(plt, A, [("frac_pos", "Δ % significant, positive", 100), ("frac_neg", "Δ % significant, negative", 100)],
         "P1b_overview_sign", "Pre-lick ROC per sign (positive: AH > WH, WH > FA, AH > FA)")
    # pooled bars with session dots
    U = pd.read_parquet(st47.BASE / "units.parquet")
    U = U[U.cohort.isin(["R+", "R-"])].reset_index(drop=True); U["uid"] = np.arange(len(U))
    L = pd.read_parquet(BASE / "roc_long_prelick.parquet"); L["cluster_id"] = L.cluster_id.astype(str)
    U["cluster_id"] = U.cluster_id.astype(str); U["electrode_group"] = U.electrode_group.astype(str)
    L["electrode_group"] = L.electrode_group.astype(str)
    L = L.merge(U[st47.KEYS + ["uid"]], on=st47.KEYS, how="inner"); U = U.set_index("uid")
    M = st47.build_measures(U, L)
    fig, axs = plt.subplots(len(VARIANTS), 2 * len(TYPES), figsize=(7.4, 8.4), gridspec_kw=dict(hspace=0.9, wspace=0.75))
    for i, v in enumerate(VARIANTS):
        for j, t in enumerate(TYPES):
            mname = f"{t}@{v}"
            if mname not in M:
                continue
            Rm = R[R.measure == mname]
            cat = t == "three_class_prelick"
            for k, metric in enumerate(["frac", "abs"]):
                ax = axs[i, 2 * j + k]
                f48.panel_pooled(ax, U, M[mname], Rm, metric, "% sig." if metric == "frac" else ("D3" if cat else "|sel|"), cat)
                ax.tick_params(labelsize=4.2); ax.yaxis.label.set_size(5)
                for tx in ax.texts:
                    tx.set_fontsize(min(tx.get_fontsize(), 4.2))
                if i == 0 and k == 0:
                    ax.text(1.25, 1.75, TLAB[t], transform=ax.transAxes, ha="center", fontsize=5.5, weight="bold")
            axs[i, 0].text(-0.9, 0.5, v, transform=axs[i, 0].transAxes, rotation=90, va="center", fontsize=6.5, weight="bold")
    fig.suptitle("Pre-lick ROC: pooled % significant (hatched = negative) and |selectivity| by cohort x stage", fontsize=7, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.05)
    fig.savefig(FIG / "P2_pooled.png", dpi=220); fig.savefig(FIG / "P2_pooled.pdf"); plt.close(fig)
    # area groups
    import ephys_utilities.allen_utils.allen_utils as au
    G = R[(R.level == "area_group") & R.included]
    order = [g for g in au.get_area_group_custom_order() if g in set(G.region)]
    for v in ["all", "rt_matched"]:
        fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.6), gridspec_kw=dict(wspace=0.15))
        for ax, comp in zip(axs, ["stage:R+", "stage:R-", "interaction:frac_sig"]):
            Mx = np.full((len(order), len(TYPES)), np.nan); ann = [[""] * len(TYPES) for _ in order]
            for j, t in enumerate(TYPES):
                sub = G[(G.measure == f"{t}@{v}") & (G.comparison == comp)].set_index("region")
                for i, g in enumerate(order):
                    if g in sub.index:
                        Mx[i, j] = sub.loc[g, "frac_sig_diff"] * 100; ann[i][j] = pstar(sub.loc[g, "frac_sig_p"])
            im = ax.imshow(Mx, cmap="RdBu_r", vmin=-15, vmax=15, aspect="auto")
            for i in range(len(order)):
                for j in range(len(TYPES)):
                    if ann[i][j]:
                        ax.text(j, i, ann[i][j], ha="center", va="center", fontsize=5)
            ax.set_xticks(range(len(TYPES)), ["WH vs AH", "FA vs WH", "FA vs AH"], rotation=40, ha="right")
            ax.set_yticks(range(len(order)), [g.replace(" areas", "") for g in order] if ax is axs[0] else [], fontsize=5)
            ax.set_title({"stage:R+": "R+: expert − learning", "stage:R-": "R−: expert − learning",
                          "interaction:frac_sig": "interaction Δ(R+) − Δ(R−)"}[comp], fontsize=6)
        cb = fig.colorbar(im, ax=axs, fraction=0.02); cb.set_label("Δ % significant", fontsize=5.5)
        fig.suptitle(f"Pre-lick ROC by area group ({v}; * perm. p < .05; blank = < 10 units or < 3 sessions per group)", fontsize=7)
        fig.subplots_adjust(left=0.17, right=0.9, top=0.86, bottom=0.2)
        fig.savefig(FIG / f"P3_area_groups_{v}.png", dpi=220); plt.close(fig)
    print("ALL DONE")


if __name__ == "__main__":
    main()
