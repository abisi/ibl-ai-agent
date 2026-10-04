"""Reward-free R+ vs R- MMD^2 per AREA GROUP x trial type (odd/even CV, mouse-level permutation, no correction).

Same features / cuts / PCA (99%, fit split) / median-heuristic RBF / mouse-level permutations as 016, restricted to the
neurons of one area group. Within an area group, a mouse contributes only if it has >= MIN_NEURONS neurons there; the
area group is tested only if each cohort then has >= MIN_MICE mice. `significant_both` = p < 0.05 in both CV
directions (uncorrected).
Output -> <method_dir>/<run-dir>/per_area/{mmd_per_area.csv, mmd_per_area_summary.csv, mmd_per_area_heatmap.png,
          mmd_per_area_ranking.png}
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

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis/rastermap_psth"))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import cluster_mmd_analysis_new as mmd                                 # noqa: E402
d016 = importlib.import_module("016_mmd_rewardfree_cv")
import ephys_utilities.allen_utils.allen_utils as allen_utils          # noqa: E402

MIN_NEURONS, MIN_MICE, ALPHA = 5, 3, 0.05
DIRS = ("fit_odd_test_even", "fit_even_test_odd")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", default="mmd_rewardfree")
    ap.add_argument("--n-perm", type=int, default=9999)
    a = ap.parse_args()
    root, fm_path, md = d016.paths(a.variant)
    out = md / a.run_dir / "per_area"
    out.mkdir(parents=True, exist_ok=True)
    X, meta, Xo, Xe = mmd._load_npz_data(fm_path)
    import yaml
    with np.load(fm_path, allow_pickle=True) as z:
        t_ctrs = [z[f"t_ctr_{i}"] for i in range(int(z["n_conds"]))]
    pipe_cfg = yaml.unsafe_load(open(fm_path.parent / "config_used.yaml"))
    _, cond_labels, _, _, _ = d016.rmu.get_conditions(pipe_cfg)
    cond_labels = list(cond_labels)
    rm = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(meta["unit_ids"])), index=meta["unit_ids"]).loc[rm["unit_ids"]].to_numpy()
    mice, cohorts = meta["mouse"][pos], meta["reward_group"][pos].astype(str)
    areas = meta["area_group"][pos].astype(str)
    slices = mmd._build_default_trial_type_slices(meta["n_bins_list"], cond_labels=cond_labels)
    exclude = mmd.time_cut_exclude_bins(t_ctrs, cond_labels, d016.REWARD_FREE_CUTS, pipe_cfg.get("bin_ms", 10) / 1000)
    V = {"fit_odd_test_even": (d016.build_versions(Xo[pos], cond_labels, slices, exclude),
                               d016.build_versions(Xe[pos], cond_labels, slices, exclude))}
    V["fit_even_test_odd"] = (V["fit_odd_test_even"][1], V["fit_odd_test_even"][0])
    versions = cond_labels + ["full"]
    cfg = {**mmd.DEFAULT_MMD_CFG, "cohort_labels": ("R+", "R-"), "pca_test_n_components": None,
           "pca_test_variance_retained": 0.99}

    rows = []
    order_ref = allen_utils.get_area_group_custom_order()
    area_list = [g for g in order_ref if g in set(areas)] + sorted(set(areas) - set(order_ref))
    for ai, ag in enumerate(area_list):
        in_area = areas == ag
        cnt = pd.Series(mice[in_area]).value_counts()
        ok_mice = cnt[cnt >= MIN_NEURONS].index
        sel = in_area & np.isin(mice, ok_mice)
        coh_m = pd.Series(cohorts[sel], index=mice[sel]).groupby(level=0).first()
        n_pos, n_neg = int((coh_m == "R+").sum()), int((coh_m == "R-").sum())
        if min(n_pos, n_neg) < MIN_MICE:
            print(f"[021] skip {ag}: {n_pos} R+ / {n_neg} R- mice with >= {MIN_NEURONS} neurons", flush=True)
            continue
        for vi, v in enumerate(versions):
            for di, d in enumerate(DIRS):
                fit, test = V[d][0][v][sel], V[d][1][v][sel]
                try:
                    res = mmd.run_cv_direction_blocks(fit, test, mice[sel], cohorts[sel], {**cfg, "mmd_n_perm": a.n_perm},
                                                      seed=100000 * ai + 100 * vi + di)
                except ValueError as e:
                    print(f"[021] {ag} / {v}: {e}", flush=True)
                    continue
                rows.append(dict(area_group=ag, feature_version=v, direction=d, n_neurons=int(sel.sum()),
                                 n_mice_pos=n_pos, n_mice_neg=n_neg, mmd2_obs=res["mmd2_obs"], mmd2_z=res["mmd2_z"],
                                 p_value=res["p_value"], n_components=res["n_components"]))
        print(f"[021] {ag}: {int(sel.sum())} neurons, {n_pos} R+ / {n_neg} R- mice done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "mmd_per_area.csv", index=False)
    summ = (df.groupby(["area_group", "feature_version"])
            .agg(n_neurons=("n_neurons", "first"), n_mice_pos=("n_mice_pos", "first"), n_mice_neg=("n_mice_neg", "first"),
                 z_mean=("mmd2_z", "mean"), p_max=("p_value", "max")).reset_index())
    summ["significant_both"] = summ.p_max < ALPHA
    summ.to_csv(out / "mmd_per_area_summary.csv", index=False)
    figures(summ, area_list, versions, out)
    piv = summ.pivot(index="area_group", columns="feature_version", values="z_mean").reindex(
        [g for g in area_list if g in set(summ.area_group)])[versions]
    print(piv.round(1).to_string())
    for v in versions:
        s = summ[(summ.feature_version == v)].sort_values("z_mean", ascending=False)
        top = ", ".join(f"{r.area_group} ({r.z_mean:.1f}{'*' if r.significant_both else ''})" for r in s.head(4).itertuples())
        print(f"{v:>20s}: {top}")
    print("ALL DONE")


def figures(summ, area_list, versions, out):
    import rastermap_psth.population_matrix_summary as pms
    pms.use_pub_font()
    fs = 6.5
    plt.rcParams.update({"font.size": fs, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
    rows = [g for g in area_list if g in set(summ.area_group)]
    Z = summ.pivot(index="area_group", columns="feature_version", values="z_mean").reindex(rows)[versions]
    S = summ.pivot(index="area_group", columns="feature_version", values="significant_both").reindex(rows)[versions]
    N = summ.groupby("area_group")[["n_neurons", "n_mice_pos", "n_mice_neg"]].first().reindex(rows)
    fig, ax = plt.subplots(figsize=(120 / 25.4, (20 + 5.5 * len(rows)) / 25.4))
    vmax = max(4.0, float(np.nanpercentile(Z.values, 98)))
    im = ax.imshow(Z.values, cmap="magma_r", vmin=0, vmax=vmax, aspect="auto")
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            v = Z.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}{'*' if S.values[i, j] else ''}", ha="center", va="center", fontsize=fs - 1.5,
                        color="w" if v > 0.55 * vmax else "k")
    ax.set_xticks(range(len(versions))); ax.set_xticklabels(versions, rotation=45, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{g}  (n={int(N.loc[g, 'n_neurons'])}; {int(N.loc[g, 'n_mice_pos'])}/{int(N.loc[g, 'n_mice_neg'])} mice)"
                        for g in rows], fontsize=fs - 1)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("MMD² z (mean of CV directions)")
    ax.set_title("Reward-free R+ vs R− MMD² per area group (*: p<0.05 both halves, uncorrected)", loc="left")
    fig.savefig(out / "mmd_per_area_heatmap.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / "mmd_per_area_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    # ranking: per trial type, area groups sorted by z
    nc = 4
    nr = int(np.ceil(len(versions) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(183 / 25.4, nr * 45 / 25.4), squeeze=False)
    colors = allen_utils.get_custom_area_groups_colors()
    for ax, v in zip(axes.ravel(), versions):
        s = summ[summ.feature_version == v].sort_values("z_mean")
        ax.barh(range(len(s)), s.z_mean, color=[colors.get(g, "0.6") for g in s.area_group],
                edgecolor=["k" if b else "none" for b in s.significant_both], linewidth=0.6)
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.area_group, fontsize=fs - 2)
        ax.axvline(0, color="0.5", lw=0.4); ax.set_title(v, loc="left", fontsize=fs)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axes.ravel()[len(versions):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("MMD² z")
    fig.suptitle("Area groups ranked by reward-free cohort difference (black edge: p<0.05 both halves)",
                 fontsize=fs + 0.5, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out / "mmd_per_area_ranking.png", dpi=400, bbox_inches="tight")
    fig.savefig(out / "mmd_per_area_ranking.pdf", bbox_inches="tight")
    plt.close(fig); plt.rcdefaults()


if __name__ == "__main__":
    main()
