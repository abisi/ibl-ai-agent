"""Reward-free MMD witness (016) per METACLUSTER, with a mouse-level R+ vs R- test per metacluster x PSTH type.

Witness = mouse-LOO witness of 016 (mean of both CV directions; R+ > 0 > R-), centered per PSTH type by the population
mean witness (what remains is each metacluster's relative lean). Per metacluster x PSTH type: mouse mean witness
(mice with >= MIN_NEURONS neurons in the metacluster), R+ vs R- with Mann-Whitney U and Welch t (both reported,
uncorrected, exploratory); `sig_both_tests` = both p < 0.05.
Metaclusters = <method_dir>/metacluster_stats/metacluster_map.csv (045 / 004 contiguous agglomerative cut);
names from <variant>/metacluster_description/metacluster_names_all.csv when present.
Outputs -> <method_dir>/<run-dir>/metacluster_witness/{witness_per_metacluster.csv, witness_per_metacluster.{png,pdf,svg}}
Usage: 023_mmd_metacluster_witness.py --variant V [--run-dir mmd_rewardfree]
"""
import argparse
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu, ttest_ind

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
d016 = importlib.import_module("016_mmd_rewardfree_cv")

MIN_NEURONS, ALPHA = 3, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", default="mmd_rewardfree")
    a = ap.parse_args()
    root, _, md = d016.paths(a.variant)
    run = md / a.run_dir
    out = run / "metacluster_witness"
    out.mkdir(parents=True, exist_ok=True)
    mc_map = pd.read_csv(md / "metacluster_stats" / "metacluster_map.csv")[["cluster", "metacluster"]]
    names_f = root / "metacluster_description" / "metacluster_names_all.csv"
    names = (pd.read_csv(names_f).set_index("family")["name"].to_dict() if names_f.exists() else {})

    wn = pd.read_parquet(run / "witness_per_neuron.parquet")
    wn = wn[wn.loo == "mouse"].merge(mc_map, left_on="rm_cluster", right_on="cluster", how="left", validate="many_to_one")
    assert wn.metacluster.notna().all(), "neurons without metacluster"
    versions = list(dict.fromkeys(wn.feature_version))
    wn["w_c"] = wn.witness_mean_dirs - wn.groupby("feature_version").witness_mean_dirs.transform("mean")

    rows = []
    for (v, m), d in wn.groupby(["feature_version", "metacluster"]):
        cnt = d.groupby("mouse_id").size()
        dm = d[d.mouse_id.isin(cnt[cnt >= MIN_NEURONS].index)]
        mm = dm.groupby(["mouse_id", "cohort"]).w_c.mean().reset_index()
        p_, n_ = mm[mm.cohort == "R+"].w_c.to_numpy(), mm[mm.cohort == "R-"].w_c.to_numpy()
        ok = len(p_) >= 3 and len(n_) >= 3
        rows.append(dict(feature_version=v, metacluster=int(m), name=names.get(int(m), ""), n_neurons=len(d),
                         share_rplus_neurons=float((d.cohort == "R+").mean()),
                         witness_centered_mean=float(d.w_c.mean()), n_mice_pos=len(p_), n_mice_neg=len(n_),
                         mouse_mean_rplus=float(np.mean(p_)) if len(p_) else np.nan,
                         mouse_mean_rminus=float(np.mean(n_)) if len(n_) else np.nan,
                         p_mannwhitney=mannwhitneyu(p_, n_).pvalue if ok else np.nan,
                         p_welch_t=ttest_ind(p_, n_, equal_var=False).pvalue if ok else np.nan))
    df = pd.DataFrame(rows)
    df["sig_both_tests"] = (df.p_mannwhitney < ALPHA) & (df.p_welch_t < ALPHA)
    df.to_csv(out / "witness_per_metacluster.csv", index=False)

    # figure: centered witness per metacluster (rows, largest first) x PSTH type; ring = both tests p < 0.05
    sizes = wn[wn.feature_version == versions[0]].groupby("metacluster").size().sort_values(ascending=False)
    W = df.pivot(index="metacluster", columns="feature_version", values="witness_centered_mean").loc[sizes.index, versions]
    S = df.pivot(index="metacluster", columns="feature_version", values="sig_both_tests").loc[sizes.index, versions]
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "Arial"})
    fig, ax = plt.subplots(figsize=((40 + 7 * len(versions)) / 25.4, (20 + 5 * len(W)) / 25.4))
    v = np.nanpercentile(np.abs(W.values), 98)
    cmap = LinearSegmentedColormap.from_list("rpm", [d016.COLORS["R-"], "white", d016.COLORS["R+"]])
    im = ax.imshow(W.values, cmap=cmap, vmin=-v, vmax=v, aspect="auto", interpolation="none")
    yy, xx = np.nonzero(S.fillna(False).values.astype(bool))
    ax.scatter(xx, yy, s=14, facecolors="none", edgecolors="k", linewidths=0.6)
    ax.set_xticks(range(len(versions))); ax.set_xticklabels(versions, rotation=50, ha="right")
    ax.set_yticks(range(len(W)))
    ax.set_yticklabels([f"MC {m} {names.get(m, '')} (n={sizes[m]})" for m in W.index])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("centered witness (R+ > 0 > R−)")
    ax.set_title(f"{a.variant}: reward-free witness per metacluster (ring: mouse-level MWU and Welch p<0.05, "
                 "uncorrected)", loc="left", fontsize=6.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"witness_per_metacluster.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(df[df.sig_both_tests].sort_values(["feature_version", "metacluster"])[
        ["feature_version", "metacluster", "name", "witness_centered_mean", "p_mannwhitney", "p_welch_t"]]
          .round(4).to_string(index=False))
    print("ALL DONE")


if __name__ == "__main__":
    main()
