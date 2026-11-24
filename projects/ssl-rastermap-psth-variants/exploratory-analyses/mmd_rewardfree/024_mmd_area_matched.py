"""Control: reward-free R+ vs R- MMD^2 (016) after matching the AREA-GROUP composition of the two cohorts.

Pooled MMD^2 can be driven by a cohort difference in where neurons were recorded. Per repeat, each area group a keeps
a random floor(p_a * N_c) neurons of each cohort c, with p_a = min(share of a in R+, share of a in R-), so both cohorts
then have identical area-group proportions. Then exactly as 016: reward-free cuts, PCA (99%) + median-heuristic sigma^2
fit on the FIT split, mouse-level permutation on the TEST split, both CV directions. Mice keep their identity
(permutations still move whole mice). Repeats with different subsamples; per PSTH type: median z, fraction of repeats
with p < 0.05 in both directions. Compared with the unmatched 016 result.
Outputs -> <method_dir>/<run-dir>/area_matched/{area_matched_runs.csv, area_matched_summary.csv, area_matched.{png,pdf,svg}}
Usage: 024_mmd_area_matched.py --variant V [--n-rep 10] [--n-perm 1999]
"""
import argparse
import importlib
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
d016 = importlib.import_module("016_mmd_rewardfree_cv")
mmd = d016.mmd
DIRS = ("fit_odd_test_even", "fit_even_test_odd")


def matched_subset(areas, cohorts, rng):
    keep = []
    N = {c: int((cohorts == c).sum()) for c in ("R+", "R-")}
    for ag in np.unique(areas):
        idx = {c: np.flatnonzero((areas == ag) & (cohorts == c)) for c in N}
        p = min(len(idx[c]) / N[c] for c in N)
        for c in N:
            k = min(int(np.floor(p * N[c])), len(idx[c]))
            if k:
                keep.append(rng.choice(idx[c], k, replace=False))
    return np.sort(np.concatenate(keep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", default="mmd_rewardfree")
    ap.add_argument("--n-rep", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=1999)
    a = ap.parse_args()
    root, fm_path, md = d016.paths(a.variant)
    out = md / a.run_dir / "area_matched"
    out.mkdir(parents=True, exist_ok=True)
    X, meta, Xo, Xe = mmd._load_npz_data(fm_path)
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
    Vo = d016.build_versions(Xo[pos], cond_labels, slices, exclude)
    Ve = d016.build_versions(Xe[pos], cond_labels, slices, exclude)
    versions = cond_labels + ["full"]
    cfg = {**mmd.DEFAULT_MMD_CFG, "cohort_labels": ("R+", "R-"), "pca_test_n_components": None,
           "pca_test_variance_retained": 0.99, "mmd_n_perm": a.n_perm}

    comp = pd.crosstab(areas, cohorts, normalize="columns")
    comp.to_csv(out / "area_composition_unmatched.csv")
    rows = []
    for r in range(a.n_rep):
        sel = matched_subset(areas, cohorts, np.random.default_rng(r))
        for vi, v in enumerate(versions):
            for di, (d, F, T) in enumerate(((DIRS[0], Vo[v], Ve[v]), (DIRS[1], Ve[v], Vo[v]))):
                res = mmd.run_cv_direction_blocks(F[sel], T[sel], mice[sel], cohorts[sel], cfg,
                                                  seed=10000 * r + 10 * vi + di)
                rows.append(dict(rep=r, feature_version=v, direction=d, n_neurons=len(sel),
                                 n_mice=len(np.unique(mice[sel])), mmd2_obs=res["mmd2_obs"], mmd2_z=res["mmd2_z"],
                                 p_value=res["p_value"]))
        print(f"[024] {a.variant} repeat {r + 1}/{a.n_rep}: {len(sel)} of {len(pos)} neurons", flush=True)
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "area_matched_runs.csv", index=False)
    per = runs.groupby(["rep", "feature_version"]).agg(z=("mmd2_z", "mean"), p_max=("p_value", "max")).reset_index()
    summ = per.groupby("feature_version").agg(z_median=("z", "median"), z_min=("z", "min"), z_max=("z", "max"),
                                              frac_sig_both=("p_max", lambda p: float((p < d016.ALPHA).mean())),
                                              p_max_median=("p_max", "median")).reindex(versions)
    ref = pd.read_csv(md / a.run_dir / "mmd_cv_summary.csv").set_index("feature_version")
    summ["z_unmatched"] = ref[["mmd2_z__fit_odd_test_even", "mmd2_z__fit_even_test_odd"]].mean(1).reindex(versions)
    summ["sig_both_unmatched"] = ref["significant_both"].reindex(versions)
    summ = summ.reset_index()
    summ.to_csv(out / "area_matched_summary.csv", index=False)
    json.dump(dict(variant=a.variant, n_rep=a.n_rep, n_perm=a.n_perm, rule="per area group keep floor(p_a*N_c) per "
                   "cohort, p_a = min cohort share", n_neurons_total=int(len(pos))),
              open(out / "config.json", "w"), indent=2)

    plt.rcParams.update({"font.size": 6.5, "axes.linewidth": 0.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(110 / 25.4, 55 / 25.4))
    x = np.arange(len(versions))
    for i, v in enumerate(versions):
        zz = per[per.feature_version == v]
        ax.scatter(np.full(len(zz), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(zz)), zz.z, s=5,
                   c=np.where(zz.p_max < d016.ALPHA, "k", "0.6"), edgecolors="none", zorder=3)
    ax.scatter(x, summ.z_unmatched, marker="_", s=120, c="#b00020", linewidths=1.2, zorder=4, label="unmatched (016)")
    ax.axhline(1.645, color="0.5", lw=0.4, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(versions, rotation=50, ha="right")
    ax.set_ylabel("MMD² z (mean of CV directions)")
    ax.set_title(f"{a.variant}: area-composition-matched subsamples ({a.n_rep} repeats; black: p<0.05 both halves)",
                 loc="left", fontsize=6.5)
    ax.legend(frameon=False, fontsize=6)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"area_matched.{ext}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(summ.round(3).to_string(index=False))
    print("ALL DONE")


if __name__ == "__main__":
    main()
