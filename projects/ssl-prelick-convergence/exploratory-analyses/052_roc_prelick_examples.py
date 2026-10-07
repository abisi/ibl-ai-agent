"""Pilot figures for the pre-lick ROC types (051): example selective neurons and two-class vs three-class ROC.

Inputs: 051 outputs (<mouse>_roc_prelick_results.csv / _trials.npz) of the pilot sessions; good units only
(units.parquet quality_label, joined on session_id, electrode_group, cluster_id).
Figures -> combined_results_ks4/ssl-prelick-convergence/across_days/fa/pilot/
  examples_<category>.png  one row per unit: (a) PSTH aligned to the corrected first lick per class (window shaded),
     (b) single-trial baseline-corrected pre-lick rates per class, pairwise selectivity and p, (c) ROC curves of the
     3 pairs, (d) three-class D3 vs its joint-permutation null.
     categories: all_sig (3-class and all three 2-class significant), three_only (3-class significant, no 2-class),
     two_only (>= 1 two-class significant, 3-class not), pref_WH / pref_AH / pref_FA (3-class significant,
     preferred class by one-vs-rest selectivity).
  two_vs_three_class.png  (a) schematic definition, (b) overlap of significance (3-class vs number of significant
     2-class types), (c) D3 vs max pairwise |selectivity|, (d) fraction significant per type and session,
     (e) reaction-time distributions per class with the pre-lick window relative to stimulus onset.
"""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
st26 = importlib.import_module("026_roc_rates_all_sessions")
OUT = m51.OUTROOT / "pilot"
PILOT = ["AB150_20241219_144646", "MH065_20260114_154021", "MH007_20250202_165003", "AB092_20231205_140109",
         "MH030_20250503_154256"]
CCOL = {"WH": "#1f77b4", "AH": "#d62728", "FA": "#7f7f7f"}
CLAB = {"WH": "whisker hit", "AH": "auditory hit", "FA": "false alarm"}
TWO = list(m51.TWO_CLASS)
TLAB = {"wh_vs_aud_hit_prelick": "WH vs AH", "whisker_hit_vs_fa_prelick": "FA vs WH", "auditory_hit_vs_fa_prelick": "FA vs AH"}


def load():
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(PILOT)]
    U = pd.read_parquet(m51.RES / "_roc_stage_analysis" / "units.parquet",
                        columns=["session_id", "electrode_group", "cluster_id", "quality_label", "cohort", "area_acronym_custom"])
    U["cluster_id"] = U.cluster_id.astype(str); U["electrode_group"] = U.electrode_group.astype(str)
    W, T = [], {}
    for r in ss.itertuples():
        d = r.file.parent
        res = pd.read_csv(d / f"{r.mouse}_roc_prelick{m51.TAG}_results.csv")
        res["cluster_id"] = res.cluster_id.astype(str); res["electrode_group"] = res.electrode_group.astype(str)
        if "variant" in res:                                           # 051 with trial variants: examples use all trials
            res = res[res.variant == "all"]
        w = res.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type",
                            values=["selectivity", "significant", "p_value"], aggfunc="first")
        w.columns = [f"{a}:{b}" for a, b in w.columns]
        extra = res[res.analysis_type == "three_class_prelick"].set_index(["session_id", "electrode_group", "cluster_id"])
        keep = [c for c in extra.columns if c.startswith(("pair_sel", "ovr_sel", "preferred"))]
        w = w.join(extra[keep]).reset_index()
        W.append(w)
        z = np.load(d / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz", allow_pickle=True)
        T[r.session_id] = dict(z)
        T[r.session_id]["day"] = r.day
        T[r.session_id]["key"] = {(e, c): i for i, (e, c) in enumerate(zip(z["electrode_group"], z["cluster_id"]))}
    W = pd.concat(W, ignore_index=True).merge(U, on=["session_id", "electrode_group", "cluster_id"], how="left")
    W = W[W.quality_label == "good"].reset_index(drop=True)
    for t in TWO + ["three_class_prelick"]:
        W[f"significant:{t}"] = W[f"significant:{t}"].astype(str).eq("True")
    W["n_two_sig"] = W[[f"significant:{t}" for t in TWO]].sum(1)
    W["sig3"] = W["significant:three_class_prelick"]
    W["max_pair"] = W[[f"selectivity:{t}" for t in TWO]].abs().max(1)
    return W, T


def d3_null(x, lab, rng, n=1000):
    X = x[None]
    D3, p, pair, ovr, null = m51.three_class(X, lab, rng)
    return D3[0], p[0], null[0]


def example_row(fig, gs, row, T, rng):
    import matplotlib.pyplot as plt
    z = T[row.session_id]; i = z["key"][(row.electrode_group, row.cluster_id)]
    lab, x = z["cls"], z["rates"][i].astype(float)
    sub = gs.subgridspec(1, 4, wspace=0.45, width_ratios=[1.3, 1, 1, 1])
    ax = fig.add_subplot(sub[0])
    e = z["psth_edges"]; tc = (e[:-1] + np.diff(e) / 2) * 1e3
    for k, c in enumerate(m51.CLASSES):
        y = np.convolve(z["psth"][i, k], np.ones(3) / 3, "same")
        ax.plot(tc, y, color=CCOL[c], lw=1, label=f"{CLAB[c]} ({(lab == c).sum()})")
    ax.axvspan(m51.PRELICK[0] * 1e3, m51.PRELICK[1] * 1e3, color="#FDD49E", alpha=0.6, lw=0)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("ms from first lick (corrected)"); ax.set_ylabel("rate (Hz)")
    ax.set_title(f"{row.session_id} {row.electrode_group} c{row.cluster_id}\n{row.area_acronym_custom} ({row.cohort}, day {z['day']})",
                 fontsize=5.5)
    ax.legend(frameon=False, fontsize=4.5, loc="upper left")
    ax = fig.add_subplot(sub[1])
    for k, c in enumerate(m51.CLASSES):
        v = x[lab == c]
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=3, color=CCOL[c], alpha=0.6, lw=0)
        ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color="k", lw=1)
    ax.set_xticks(range(3), m51.CLASSES); ax.axhline(0, color="0.7", lw=0.4)
    ax.set_ylabel("pre-lick rate − baseline (Hz)")
    txt = [f"{TLAB[t]}: {row[f'selectivity:{t}']:+.2f} p={row[f'p_value:{t}']:.3f}{' *' if row[f'significant:{t}'] else ''}"
           for t in TWO]
    ax.set_title("\n".join(txt), fontsize=4.6)
    ax = fig.add_subplot(sub[2])
    for t in TWO:
        c1, c2 = m51.TWO_CLASS[t]
        mm = np.isin(lab, [c1, c2])
        fpr, tpr, _ = roc_curve((lab[mm] == c2).astype(int), x[mm])
        ax.plot(fpr, tpr, lw=1, color={"wh_vs_aud_hit_prelick": "#9467bd", "whisker_hit_vs_fa_prelick": CCOL["WH"],
                                        "auditory_hit_vs_fa_prelick": CCOL["AH"]}[t],
                label=f"{TLAB[t]} AUC {row[f'selectivity:{t}'] / 2 + 0.5:.2f}")
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.5, ls="--")
    ax.set_xlabel("FPR (class 1)"); ax.set_ylabel("TPR (class 2)"); ax.legend(frameon=False, fontsize=4.3, loc="lower right")
    ax.set_title("two-class ROC curves", fontsize=5.5)
    ax = fig.add_subplot(sub[3])
    D3, _, null = d3_null(x, lab, rng)                           # null redrawn for display; p = stored result
    p = float(row["p_value:three_class_prelick"])
    ax.hist(null, bins=30, color="0.75")
    ax.axvline(D3, color="k", lw=1.2)
    ax.set_title(f"three-class D3={D3:.2f}, p={p:.3f}{' *' if p < 0.05 else ''}\npreferred: {row.preferred_class}", fontsize=5.5)
    ax.set_xlabel("D3 null (redrawn; p = stored run)")


def examples(W, T, plt):
    rng = np.random.default_rng(1)
    cats = {
        "all_sig": W[W.sig3 & (W.n_two_sig == 3)].sort_values("selectivity:three_class_prelick", ascending=False),
        "three_only": W[W.sig3 & (W.n_two_sig == 0)].sort_values("p_value:three_class_prelick"),
        "two_only": W[~W.sig3 & (W.n_two_sig >= 1)].sort_values("max_pair", ascending=False),
        "pref_WH": W[W.sig3 & (W.preferred_class == "WH")].sort_values("ovr_sel_WH", ascending=False),
        "pref_AH": W[W.sig3 & (W.preferred_class == "AH")].sort_values("ovr_sel_AH", ascending=False),
        "pref_FA": W[W.sig3 & (W.preferred_class == "FA")].sort_values("ovr_sel_FA", ascending=False)}
    desc = {"all_sig": "significant in the three-class and all three two-class ROCs",
            "three_only": "significant in the three-class ROC only (no single pair significant)",
            "two_only": "significant in >= 1 two-class ROC but not in the three-class ROC",
            "pref_WH": "three-class significant, preferring whisker hits (one-vs-rest)",
            "pref_AH": "three-class significant, preferring auditory hits", "pref_FA": "three-class significant, preferring false alarms"}
    counts = {}
    for k, d in cats.items():
        counts[k] = len(d)
        d = d.groupby("session_id").head(2).head(5)                          # spread over sessions
        if not len(d):
            continue
        fig = plt.figure(figsize=(7.4, 1.75 * len(d) + 0.5))
        gs = fig.add_gridspec(len(d), 1, hspace=1.1, top=1 - 0.75 / (1.75 * len(d) + 0.5), bottom=0.06, left=0.07, right=0.98)
        for j, (_, row) in enumerate(d.iterrows()):
            example_row(fig, gs[j], row, T, rng)
        fig.suptitle(f"Pre-lick ROC examples: {desc[k]} (n = {counts[k]} good units in 5 pilot sessions)", fontsize=7, y=0.998)
        fig.savefig(OUT / f"examples_{k}.png", dpi=220); plt.close(fig)
    return counts


def overview(W, T, plt):
    fig = plt.figure(figsize=(7.4, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.45, left=0.07, right=0.98, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.text(0, 1, "Two-class ROC (per pair)\n"
            "  AUC = P(rate class 2 > rate class 1)\n  sel = 2 AUC − 1 (signed)\n"
            "  p: one-tailed label permutation on the side\n  of sel (as in the main ROC); sig if p < .05\n"
            "  → chance rate of 'significant' ≈ 10 %\n\n"
            "Three-class ROC (WH, AH, FA together)\n"
            "  D3 = mean over the 3 pairs of |2 AUC − 1|\n  (balanced, unsigned, 0..1)\n"
            "  p: the 3 labels permuted jointly; sig if\n  P(D3null ≥ D3) < .05 → chance rate 5 %\n"
            "  preferred class: largest one-vs-rest sel.\n\n"
            "D3 is one test per unit: it picks up\nconsistent graded differences across all\n"
            "pairs (three_only) but dilutes a single\nstrong pair (two_only).",
            va="top", fontsize=5.4, family="monospace", transform=ax.transAxes)
    ax.set_title("a  definitions", loc="left", fontsize=6.5)
    ax = fig.add_subplot(gs[0, 1])
    tab = pd.crosstab(W.n_two_sig, W.sig3).reindex(columns=[False, True], fill_value=0)
    x = np.arange(len(tab))
    ax.bar(x - 0.2, tab[False] / len(W) * 100, 0.4, color="0.75", label="3-class n.s.")
    ax.bar(x + 0.2, tab[True] / len(W) * 100, 0.4, color="k", label="3-class sig.")
    ax.set_xticks(x, tab.index); ax.set_xlabel("# significant two-class types"); ax.set_ylabel("% of good units")
    ax.legend(frameon=False, fontsize=5); ax.set_title(f"b  overlap (n = {len(W)} good units)", loc="left", fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2])
    col = np.where(W.sig3 & (W.n_two_sig == 0), "#e6550d", np.where(~W.sig3 & (W.n_two_sig > 0), "#3182bd",
                   np.where(W.sig3, "k", "0.8")))
    ax.scatter(W.max_pair, W["selectivity:three_class_prelick"], s=2, c=col, lw=0, alpha=0.7, rasterized=True)
    ax.plot([0, 1], [0, 1], color="0.6", lw=0.4, ls=":")
    ax.set_xlabel("max pairwise |selectivity|"); ax.set_ylabel("three-class D3")
    for lab, c in [("both sig.", "k"), ("3-class only", "#e6550d"), ("2-class only", "#3182bd"), ("neither", "0.8")]:
        ax.scatter([], [], s=6, color=c, label=lab)
    ax.legend(frameon=False, fontsize=4.8, loc="upper left"); ax.set_title("c  D3 vs strongest pair", loc="left", fontsize=6.5)
    ax = fig.add_subplot(gs[1, 0:2])
    F = W.groupby("session_id")[[f"significant:{t}" for t in TWO + ["three_class_prelick"]]].mean() * 100
    xs = np.arange(F.shape[1])
    for s, r in F.iterrows():
        ax.plot(xs, r.values, "-o", ms=2.5, lw=0.6, label=f"{s[:5]} ({W[W.session_id == s].cohort.iloc[0]}, d{T[s]['day']})")
    ax.axhline(10, color="0.6", lw=0.5, ls="--"); ax.axhline(5, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(xs, [TLAB[t] for t in TWO] + ["three-class"]); ax.set_ylabel("% significant (good units)")
    ax.legend(frameon=False, fontsize=4.6, ncol=2); ax.set_title("d  fraction significant per session (dashed: 2-class chance 10 %, dotted: 5 %)",
                                                               loc="left", fontsize=6.5)
    ax = fig.add_subplot(gs[1, 2])
    bins = np.arange(0, 1.0, 0.025)
    for c in ["WH", "AH", "FA"]:
        rt = np.concatenate([T[s]["rt"][T[s]["cls"] == c] for s in T])
        ax.hist(rt * 1e3, bins * 1e3, histtype="step", color=CCOL[c], lw=1, density=True,
                label=f"{CLAB[c]} (median {np.median(rt) * 1e3:.0f} ms)")
    ax.axvspan(0, 100, color="#FDD49E", alpha=0.5, lw=0)
    ax.set_xlabel("reaction time (ms from stimulus / trial onset)"); ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=4.6)
    ax.set_title("e  RT: pre-lick window = [RT−100, RT] ms;\nshaded = first 100 ms after stimulus", loc="left", fontsize=6)
    fig.suptitle("Pre-lick ROC pilot (5 sessions): two-class vs three-class", fontsize=7.5)
    fig.savefig(OUT / "two_vs_three_class.png", dpi=220); plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 6, "axes.titlesize": 6, "axes.labelsize": 5.8, "xtick.labelsize": 5.2,
                         "ytick.labelsize": 5.2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
    OUT.mkdir(parents=True, exist_ok=True)
    W, T = load()
    counts = examples(W, T, plt)
    overview(W, T, plt)
    summ = dict(n_good=len(W), category_counts=counts,
                frac_sig={t: float(W[f"significant:{t}"].mean()) for t in TWO + ["three_class_prelick"]},
                rt_median_ms={c: float(np.median(np.concatenate([T[s]["rt"][T[s]["cls"] == c] for s in T])) * 1e3)
                              for c in m51.CLASSES})
    (OUT / "pilot_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
