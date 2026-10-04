"""F_mk per METACLUSTER, computed directly from neuron counts (not by summing fine-cluster F_mk):

    F[m, M] = n_neurons(mouse m, metacluster M) / n_neurons(mouse m)

Metaclusters = run_meta_clustering families (recomputed + checked by
population_matrix_summary.metaclustering), numbered 1..F in dendrogram leaf order oriented to
the rastermap direction (same order as the metacluster-order matrix figures).
Mouse set / reward group / neuron set identical to run_reward_group_stats (CV results npz).

Outputs per variant, in <method_dir>/metacluster_stats/:
    f_matrix_metacluster.npz        f_matrix (mice x metaclusters), mouse_ids, reward_groups,
                                    neuron_counts, metacluster_ids
    fmk_metacluster_long.csv        mouse_id, reward_group, metacluster, n_in_metacluster,
                                    n_total_mouse, f_mk  (+ variant)
    metacluster_map.csv             fine cluster (1-based) -> metacluster (1..F) -> original family id
    fig_fmk_metacluster.png/pdf/svg descriptive: per-mouse F_mk per metacluster, R+ vs R-
Sanity check (printed + saved): direct F equals the sum of fine-cluster F over member clusters.
"""
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rastermap_psth import population_matrix_summary as pms
from ephys_utilities.plotting_utils.plotting_utils import GROUP_COLORS

VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")


def run(md, variant):
    out = md / "metacluster_stats"
    out.mkdir(exist_ok=True)
    d = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    labels = d["cluster_labels"].astype(int)
    mouse_arr, reward_arr = d["mouse_arr"].astype(str), d["reward_arr"].astype(str)

    meta = pms.metaclustering(md)
    leaves, flipped = pms.leaf_order(meta)
    fam_order = list(dict.fromkeys(meta["families"][c] for c in leaves))     # families in leaf order
    fam_to_mc = {f: i + 1 for i, f in enumerate(fam_order)}                  # 1..F
    mc_of_cluster = {c: fam_to_mc[meta["families"][c]] for c in meta["clusters"]}
    pd.DataFrame({"cluster": np.array(list(mc_of_cluster)) + 1, "metacluster": list(mc_of_cluster.values()),
                  "original_family_id": [meta["families"][c] for c in mc_of_cluster],
                  "leaf_position": [leaves.index(c) for c in mc_of_cluster]}).sort_values("leaf_position") \
        .to_csv(out / "metacluster_map.csv", index=False)
    mc_arr = np.array([mc_of_cluster[c] for c in labels])
    n_mc = len(fam_order)

    # same mouse set / group assignment as run_reward_group_stats
    mice = np.unique(mouse_arr)
    grp = {m: pd.Series(reward_arr[mouse_arr == m]).mode()[0] for m in mice}
    mice = [m for m in mice if grp[m] in ("R+", "R-")]
    F = np.zeros((len(mice), n_mc)); counts = np.zeros(len(mice), int); rows = []
    for i, m in enumerate(mice):
        sel = mouse_arr == m
        counts[i] = sel.sum()
        for j in range(n_mc):
            n_in = int((mc_arr[sel] == j + 1).sum())
            F[i, j] = n_in / counts[i]
            rows.append(dict(variant=variant, mouse_id=m, reward_group=grp[m], metacluster=j + 1,
                             n_in_metacluster=n_in, n_total_mouse=int(counts[i]), f_mk=F[i, j]))
    groups = np.array([grp[m] for m in mice])
    np.savez_compressed(out / "f_matrix_metacluster.npz", f_matrix=F, mouse_ids=np.array(mice),
                        reward_groups=groups, neuron_counts=counts, metacluster_ids=np.arange(1, n_mc + 1))
    pd.DataFrame(rows).to_csv(out / "fmk_metacluster_long.csv", index=False)

    # sanity: direct metacluster F == sum of fine-cluster F over member clusters (same denominators)
    fz = np.load(md / "stats" / "f_matrix.npz", allow_pickle=True)
    assert list(fz["mouse_ids"].astype(str)) == list(mice), "mouse set differs from run_reward_group_stats"
    F_fine = fz["f_matrix"]
    F_sum = np.column_stack([F_fine[:, [c for c, mc in mc_of_cluster.items() if mc == j + 1]].sum(1)
                             for j in range(n_mc)])
    max_dev = float(np.abs(F - F_sum).max())
    assert np.allclose(F.sum(1), 1), "per-mouse F_mk must sum to 1"

    # descriptive figure: per-mouse dots + mean ± SEM per metacluster, R+ vs R-
    fig, ax = plt.subplots(figsize=(max(6, 0.38 * n_mc + 2), 3.4), dpi=250)
    for off, g, col in [(-0.18, "R+", GROUP_COLORS["rplus"]), (0.18, "R-", GROUP_COLORS["rminus"])]:
        Fg = F[groups == g]
        x = np.arange(1, n_mc + 1) + off
        rng = np.random.default_rng(0)
        for j in range(n_mc):
            ax.scatter(np.full(len(Fg), x[j]) + rng.uniform(-0.06, 0.06, len(Fg)), Fg[:, j], s=3, color=col,
                       alpha=0.35, lw=0)
        ax.errorbar(x, Fg.mean(0), yerr=Fg.std(0, ddof=1) / np.sqrt(len(Fg)), fmt="o", ms=3, color=col,
                    elinewidth=0.8, label=f"{g} (n={len(Fg)} mice)")
    ax.set_xticks(np.arange(1, n_mc + 1)); ax.tick_params(labelsize=6)
    ax.set_xlabel("Metacluster (dendrogram leaf order)"); ax.set_ylabel(r"$F_{m,M}$")
    ax.set_title(f"{variant}: per-mouse fractional representation per metacluster", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"fig_fmk_metacluster.{ext}", bbox_inches="tight")
    plt.close(fig)

    summary = dict(variant=variant, n_metaclusters=n_mc, n_mice=len(mice), n_rplus=int((groups == "R+").sum()),
                   n_rminus=int((groups == "R-").sum()), n_neurons=int(counts.sum()),
                   leaf_order_flipped=bool(flipped),
                   max_abs_diff_direct_vs_summed_fine_fmk=max_dev)
    json.dump(summary, open(out / "fmk_metacluster_summary.json", "w"), indent=2)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()
    for md in sorted(VARIANTS_ROOT.glob("qc_*/rastermap_clustering/*/*/clustering/*/rastermap")):
        variant = next(p for p in md.parts if p.startswith("qc_"))
        if a.variants and variant not in a.variants:
            continue
        if not (md / "stats" / "f_matrix.npz").exists():
            print(f"skip {variant}: no stats/f_matrix.npz"); continue
        run(md, variant)
    print("ALL DONE")
