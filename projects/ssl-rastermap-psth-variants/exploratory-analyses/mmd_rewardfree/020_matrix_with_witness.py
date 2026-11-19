"""Publication matrix with reward-free MMD witness columns + R+ share vs witness scatter (exploratory view).

Witness per cluster = mean over the cluster's neurons of the mouse-LOO witness (mean of both CV directions, from 016),
CENTERED by subtracting the population mean witness of that feature version (removes the global offset shared by all
clusters; what remains is each cluster's relative R+/R- lean). R+ share = mean F_mk(R+) / (mean F_mk(R+) + mean
F_mk(R-)) per cluster (reward-inclusive clustering; positive control).
Outputs -> <method_dir>/<run-dir>/matrix_with_witness/
"""
import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import rastermap_psth.population_matrix_summary as pms            # noqa: E402
import rastermap_psth.rastermap_utils as rmu                       # noqa: E402

VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CONFIG = "/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", default="mmd_rewardfree")
    ap.add_argument("--versions", nargs="*", default=["Whisker miss", "Auditory hit", "Auditory hit (lick)"])
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    md = sorted((VARIANTS_ROOT / a.variant).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]
    out = md / a.run_dir / "matrix_with_witness"
    out.mkdir(parents=True, exist_ok=True)
    qc = ["good", "mua"] if a.variant.startswith("qc_good_mua") else ["good"]
    cfg = pms.load_cfg(CONFIG, unit_quality_label=qc)
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS, rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = \
        rmu.get_conditions(cfg)
    R = pms.load_rastermap(md); roc = pms.build_roc_categories(md, R["unit_ids"])
    meta = pms.metaclustering(md); fmk = pms.load_fmk_stats(md)

    wn = pd.read_parquet(md / a.run_dir / "witness_per_neuron.parquet")
    wn = wn[wn.loo == "mouse"]
    witness, rows = {}, []
    for v in a.versions:
        d = wn[wn.feature_version == v]
        pop = d.witness_mean_dirs.mean()
        cm = d.groupby("rm_cluster").witness_mean_dirs.mean() - pop
        witness[v] = cm
        rows.append(pd.DataFrame(dict(feature_version=v, rm_cluster=cm.index, witness_centered=cm.values,
                                      population_mean=pop)))
    pd.concat(rows).to_csv(out / "witness_per_cluster_centered.csv", index=False)

    title = a.title or a.variant
    for rm in ("cluster_mean_uniform", "neuron"):
        pms.fig5_population_matrix_publication(R, roc, meta, fmk, cfg, out / f"pub_matrix_{rm}_witness",
                                               row_mode=rm, title=title, witness=witness)

    # scatter: R+ share (reward-inclusive clusters) vs centered reward-free witness, one point per cluster
    F, g = fmk["F"], fmk["groups"]
    mrp, mrm = F[g == "R+"].mean(0), F[g == "R-"].mean(0)
    share = pd.Series(mrp / (mrp + mrm))
    fam = meta["families"]
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(share))])
    pms.use_pub_font()
    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    n = len(a.versions)
    fig, axes = plt.subplots(1, n, figsize=(n * 50 / 25.4, 52 / 25.4), squeeze=False)
    stats = []
    for ax, v in zip(axes[0], a.versions):
        w = witness[v].reindex(share.index)
        ok = np.isfinite(w.values) & np.isfinite(share.values)
        sig = np.asarray(fmk["reject"], bool)
        ax.scatter(share[ok & ~sig], w[ok & ~sig], s=8, c=[pal[fam[k]] for k in share.index[ok & ~sig]],
                   edgecolors="none", alpha=0.85)
        ax.scatter(share[ok & sig], w[ok & sig], s=14, c=[pal[fam[k]] for k in share.index[ok & sig]],
                   edgecolors="k", linewidths=0.5)
        rho, p = spearmanr(share[ok], w[ok])
        stats.append(dict(feature_version=v, spearman_rho=rho, p=p, n_clusters=int(ok.sum())))
        ax.axvline(0.5, color="0.6", lw=0.4); ax.axhline(0, color="0.6", lw=0.4)
        ax.set_xlabel("R+ share (reward-inclusive clusters)")
        ax.set_title(f"{v}\nSpearman ρ = {rho:.2f} (p = {p:.2g})", loc="left", fontsize=6.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.set_box_aspect(1)
    axes[0, 0].set_ylabel("reward-free witness\n(centered; >0 R+-like)")
    fig.text(0.01, -0.02, "one point per rastermap cluster, colour = metacluster; black edge = R+/R− post-hoc "
                          "Mann-Whitney FDR<0.05 (reward-inclusive)", fontsize=5.5, color="0.35")
    fig.tight_layout()
    fig.savefig(out / "share_vs_witness.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / "share_vs_witness.pdf", bbox_inches="tight")
    pd.DataFrame(stats).to_csv(out / "share_vs_witness_stats.csv", index=False)
    print(pd.DataFrame(stats).round(3).to_string(index=False))
    print("ALL DONE")


if __name__ == "__main__":
    main()
