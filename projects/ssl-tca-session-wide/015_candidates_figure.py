"""Candidate learning components in the main fit and in the reference fit (exploratory).

Rows: split clusters whose learning effect beyond outcome differs between cohorts or is strong in R+ (013 / 014), each
next to its best-matching cluster of the reference fit (014_method_agreement). Columns: mean time factor per cohort
(main), mean time factor per cohort (reference), per-mouse excess of the outcome-controlled learning correlation over
its shift null (main), same (reference). Black bar = median; p: one-sample Wilcoxon per cohort, R+ vs R- Mann-Whitney
and Welch (from clusters.csv, uncorrected).

Run: python 015_candidates_figure.py -> OUT/figures/aligned_gpu/candidates_main_vs_reference.png|pdf
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tca_lib as L  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
CAND = [("stim_long", 67), ("stim_long", 56), ("stim_sensory", 68)]   # split clusters (013 numbering)


def pstr(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def main():
    mem = pd.read_parquet(L.OUT / "tables" / "cluster_members.parquet")
    mem = mem[mem.stage == "learning"]
    cl = pd.read_csv(L.OUT / "tables" / "clusters.csv").set_index(["method", "alignment", "cluster"])
    ag = pd.read_csv(L.OUT / "tables" / "method_agreement.csv").set_index(["alignment", "split_cluster"])
    fig, axs = plt.subplots(len(CAND), 4, figsize=(8.0, 2.1 * len(CAND)), gridspec_kw=dict(wspace=0.55, hspace=1.1))
    for i, (al, k) in enumerate(CAND):
        k2 = int(ag.loc[(al, k), "ref_cluster"])
        for j, (meth, kk, tag) in enumerate((("_splitnn", k, "main fit"), ("_nn", k2, "reference fit"))):
            m = mem[(mem.method == meth) & (mem.alignment == al) & (mem.cluster == kk)]
            c = cl.loc[(meth, al, kk)]
            ax = axs[i, j]
            t = np.array(json.loads(m.t_centres_ms.iloc[0]))
            for coh, g in m.groupby("cohort"):
                TF = np.array([json.loads(x) for x in g.time_factor])
                ax.plot(t, TF.mean(0), color=COH_COL[coh], lw=1)
            nm = m.groupby("cohort").mouse_id.nunique()
            ax.axvline(0, color="0.6", lw=0.4)
            ax.set_title(f"{al}, {tag}, cl. {kk}\npeak {c.peak_ms_median:.0f} ms, top {c.top_group}\n"
                         f"R+ {nm.get('R+', 0)} / R- {nm.get('R-', 0)} mice", fontsize=5.5)
            ax.set_xlabel("Time from stimulus (ms)")
            if j == 0:
                ax.set_ylabel("Time factor (mean)")
            ax = axs[i, 2 + j]
            pm = m.groupby("mouse_id").agg(cohort=("cohort", "first"), v=("learn_ctrl_excess", "mean"))
            for x, coh in enumerate(("R+", "R-")):
                v = pm[pm.cohort == coh].v.dropna().to_numpy()
                jit = (np.random.default_rng(x).random(len(v)) - 0.5) * 0.3
                ax.scatter(x + jit, v, s=3, color=COH_COL[coh], lw=0)
                ax.plot([x - 0.22, x + 0.22], [np.median(v)] * 2, color="k", lw=1)
                ax.text(x, 1.02, pstr(c[f"all_learnctrl_{coh}_wilcoxon_p"]), transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=5)
            ax.axhline(0, color="0.6", lw=0.4)
            ax.set_xticks([0, 1], ["R+", "R-"])
            ax.set_xlim(-0.6, 1.6)
            ax.set_title(f"{tag}\nR+ vs R-: MW {pstr(c['all_learnctrl_RpRm_mw_p'])}\n"
                         f"Welch {pstr(c['all_learnctrl_RpRm_welch_p'])}", fontsize=5.5, pad=9)
            if j == 0:
                ax.set_ylabel("Learning rho excess\n(outcome-controlled)")
    fig.suptitle("Candidate learning components, main fit vs reference fit (whisker day 0; exploratory, uncorrected)",
                 fontsize=7, y=0.995)
    out = L.OUT / "figures" / "aligned_gpu"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"candidates_main_vs_reference.{ext}", dpi=230, bbox_inches="tight")
    print("saved", out / "candidates_main_vs_reference.png")


if __name__ == "__main__":
    main()
