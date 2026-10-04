"""Neuron-level R+ vs R- MMD^2 on REWARD-FREE features, odd/even cross-validated, with per-neuron witness
mapped onto the (reward-inclusive) rastermap clusters.

Uses rastermap_psth/cluster_mmd_analysis_new.py (mouse-block extension, 2026-09-27):
  - feature matrix = the variant's feature_matrix.npz (X_odd / X_even), restricted to the neurons in the
    rastermap CV result; no extra normalisation (already baseline-normalised)
  - reward-free bins: lick-aligned whisker / auditory hits keep bins ending <= +50 ms after the corrected
    first lick; stimulus-aligned whisker hits, whisker misses and auditory hits keep bins ending <= +200 ms after stimulus onset;
    all other conditions unchanged (REWARD_FREE_CUTS)
  - feature versions: every single condition + groups (full, whisker, auditory, passive, active_stim, lick)
  - per version, both directions: PCA (99% variance) + sigma2 (median heuristic) fit label-blind on one
    split, mouse-level permutation test (9999) + witness on the other split
  - no multiple-comparison correction; `significant_both` = raw p < 0.05 in BOTH directions
  - witness: neuron-LOO and mouse-LOO per neuron and direction; per rastermap cluster the mean (no stats)

Outputs -> <method_dir>/mmd_rewardfree/: mmd_cv_results.csv (one row per version x direction),
mmd_cv_summary.csv (one row per version), witness_per_neuron.parquet (keys mouse_id, session_id,
electrode_group, cluster_id, unit_id + rastermap cluster), witness_per_cluster.csv, config.json, figures.
Usage: 016_mmd_rewardfree_cv.py --variant <rastermap variant name> [--n-perm 9999]
"""
import argparse
import json
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis/rastermap_psth"))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import cluster_mmd_analysis_new as mmd                               # noqa: E402
import rastermap_psth.rastermap_utils as rmu                          # noqa: E402

VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
REWARD_FREE_CUTS = {"Whisker hit (lick)": 0.05, "Auditory hit (lick)": 0.05,   # s after corrected first lick
                    "Whisker hit": 0.2, "Auditory hit": 0.2, "Whisker miss": 0.2}   # s after stimulus onset (miss: same cut for a fair whisker comparison)
GROUPS = {
    "full": None,                                                              # all conditions
    "whisker": ["Whisker pre", "Whisker post", "Whisker miss", "Whisker hit", "Whisker hit (lick)"],
    "auditory": ["Auditory pre", "Auditory post", "Auditory hit", "Auditory hit (lick)"],
    "passive": ["Whisker pre", "Whisker post", "Auditory pre", "Auditory post"],
    "active_stim": ["Whisker miss", "Whisker hit", "Auditory hit"],
    "lick": ["Spont. lick", "Whisker hit (lick)", "Auditory hit (lick)"],
}
ALPHA = 0.05
COLORS = {"R+": "#00B400", "R-": "#C800C8"}


def paths(variant):
    root = VARIANTS_ROOT / variant
    fm = sorted(root.glob("rastermap_clustering/*/*/*/feature_matrix.npz"))[0]
    md = sorted(root.glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]
    return root, fm, md


def build_versions(Xs, cond_labels, slices, exclude_bins):
    """{version: (N, d) matrix} for one split: single conditions + groups, reward-free bins removed."""
    per = mmd._slice_feature_matrix(Xs, slices, exclude_bins=exclude_bins)
    out = {c: per[c] for c in cond_labels}
    for g, conds in GROUPS.items():
        out[g] = np.concatenate([per[c] for c in (cond_labels if conds is None else conds)], axis=1)
    return out


def fig_nulls(records, out):
    names = list(dict.fromkeys(r["feature_version"] for r in records))
    nc = 4
    nr = int(np.ceil(len(names) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc, 2.3 * nr), dpi=150, squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        for r in [r for r in records if r["feature_version"] == name]:
            c = "0.3" if r["direction"] == "fit_odd_test_even" else "tab:orange"
            ax.hist(r["null"], bins=50, color=c, alpha=0.45)
            ax.axvline(r["mmd2_obs"], color=c, lw=1.5)
        rr = [r for r in records if r["feature_version"] == name]
        both = all(r["p_value"] < ALPHA for r in rr)
        ax.set_title(f"{name}\n" + ", ".join(f"p={r['p_value']:.3g}" for r in rr) + ("  *" if both else ""),
                     fontsize=7.5, color="k" if not both else "#b00020")
        ax.tick_params(labelsize=6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle("Reward-free MMD², mouse-level permutation null (grey: fit odd → test even; orange: fit even → "
                 "test odd; line = observed; * = p<0.05 both, uncorrected)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "mmd_cv_null_histograms.png"); fig.savefig(out / "mmd_cv_null_histograms.pdf")
    plt.close(fig)


def fig_cluster_witness(wc, versions, out, loo="mouse"):
    col = f"witness_{loo}_loo_mean_dirs"
    piv = wc.pivot(index="rm_cluster", columns="feature_version", values=col)[versions]
    v = np.nanpercentile(np.abs(piv.values), 98)
    fig, ax = plt.subplots(figsize=(0.45 * len(versions) + 2, 9), dpi=200)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rpm", [COLORS["R-"], "white", COLORS["R+"]])
    im = ax.imshow(piv.values, aspect="auto", cmap=cmap, vmin=-v, vmax=v, interpolation="none")
    ax.set_xticks(range(len(versions))); ax.set_xticklabels(versions, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("rastermap cluster (rastermap order)"); ax.tick_params(axis="y", labelsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(f"mean witness ({loo}-LOO; R+ > 0 > R−), mean of both CV directions", fontsize=7)
    ax.set_title("Reward-free witness per cluster (descriptive, no cluster-level statistics)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"witness_per_cluster_{loo}_loo.png"); fig.savefig(out / f"witness_per_cluster_{loo}_loo.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--n-perm", type=int, default=9999)
    ap.add_argument("--versions", nargs="*", default=None, help="subset of feature versions (default: all)")
    ap.add_argument("--out-name", default="mmd_rewardfree")
    a = ap.parse_args()
    t0 = time.time()
    root, fm_path, md = paths(a.variant)
    out = md / a.out_name
    out.mkdir(exist_ok=True)

    # data: feature matrix restricted to the rastermap neurons (same rows as the clusters)
    X, meta, Xo, Xe = mmd._load_npz_data(fm_path)
    with np.load(fm_path, allow_pickle=True) as z:
        n_conds = int(z["n_conds"])
        t_ctrs = [z[f"t_ctr_{i}"] for i in range(n_conds)]
    import yaml
    pipe_cfg = yaml.unsafe_load(open(fm_path.parent / "config_used.yaml"))   # pipeline-written (python tuples)
    _, cond_labels, _, _, _ = rmu.get_conditions(pipe_cfg)
    cond_labels = list(cond_labels)
    assert len(cond_labels) == n_conds, (cond_labels, n_conds)
    rm = np.load(md / "rastermap_results_cv.npz", allow_pickle=True)
    rm_units, rm_labels = rm["unit_ids"], rm["cluster_labels"].astype(int)
    pos = pd.Series(np.arange(len(meta["unit_ids"])), index=meta["unit_ids"]).loc[rm_units].to_numpy()
    mice, cohorts = meta["mouse"][pos], meta["reward_group"][pos].astype(str)
    Xo, Xe = Xo[pos], Xe[pos]

    cfg = {**mmd.DEFAULT_MMD_CFG, "mmd_n_perm": a.n_perm, "cohort_labels": ("R+", "R-"),
           "pca_test_n_components": None, "pca_test_variance_retained": 0.99}
    slices = mmd._build_default_trial_type_slices(meta["n_bins_list"], cond_labels=cond_labels)
    bin_s = float(pipe_cfg.get("bin_ms", 10)) / 1000
    exclude = mmd.time_cut_exclude_bins(t_ctrs, cond_labels, REWARD_FREE_CUTS, bin_s)
    kept = {c: int(slices[c][1] - slices[c][0] - sum(e - s for s, e in exclude.get(c, []))) for c in cond_labels}
    print(f"[016] {a.variant}: {len(pos)} rastermap neurons, {len(np.unique(mice))} mice; bins kept per "
          f"condition: {kept}", flush=True)
    V_odd, V_even = build_versions(Xo, cond_labels, slices, exclude), build_versions(Xe, cond_labels, slices, exclude)
    versions = a.versions or list(V_odd)

    records, wit = [], {}
    for i, v in enumerate(versions):
        for j, (direction, fit, test) in enumerate((("fit_odd_test_even", V_odd[v], V_even[v]),
                                                    ("fit_even_test_odd", V_even[v], V_odd[v]))):
            t1 = time.time()
            res = mmd.run_cv_direction_blocks(fit, test, mice, cohorts, cfg, seed=1000 * i + j)
            wit[(v, direction, "neuron")] = res.pop("witness_neuron_loo")
            wit[(v, direction, "mouse")] = res.pop("witness_mouse_loo")
            records.append(dict(feature_version=v, direction=direction, n_bins=int(fit.shape[1]),
                                null=res.pop("null_mmd2"), **res))
            print(f"[016] {v:>20s} {direction}: mmd2={res['mmd2_obs']:.4g} z={res['mmd2_z']:.2f} "
                  f"p={res['p_value']:.4g} ({res['n_components']} PCs, {time.time() - t1:.0f}s)", flush=True)

    res_df = pd.DataFrame([{k: v for k, v in r.items() if k != "null"} for r in records])
    res_df.to_csv(out / "mmd_cv_results.csv", index=False)
    summ = res_df.pivot(index="feature_version", columns="direction",
                        values=["mmd2_obs", "mmd2_z", "p_value", "n_components"])
    summ.columns = [f"{a_}__{b_}" for a_, b_ in summ.columns]
    summ["significant_both"] = ((summ["p_value__fit_odd_test_even"] < ALPHA)
                                & (summ["p_value__fit_even_test_odd"] < ALPHA))
    summ = summ.reindex(versions).reset_index()
    summ.to_csv(out / "mmd_cv_summary.csv", index=False)
    np.savez_compressed(out / "mmd_cv_nulls.npz", **{f"{r['feature_version']}__{r['direction']}": r["null"]
                                                    for r in records})

    # witness per neuron (join keys) and per rastermap cluster
    keys = pd.read_csv(sorted(root.glob("rastermap_clustering/*/*/*/neuron_metadata.csv"))[0])[
        ["unit_id", "mouse_id", "session_id", "electrode_group", "cluster_id"]]
    base = pd.DataFrame(dict(unit_id=rm_units, rm_cluster=rm_labels, cohort=cohorts)).merge(
        keys, on="unit_id", how="left", validate="one_to_one")
    rows = []
    for v in versions:
        for loo in ("neuron", "mouse"):
            d = base.copy()
            d["feature_version"], d["loo"] = v, loo
            d["witness_fit_odd_test_even"] = wit[(v, "fit_odd_test_even", loo)]
            d["witness_fit_even_test_odd"] = wit[(v, "fit_even_test_odd", loo)]
            rows.append(d)
    wn = pd.concat(rows, ignore_index=True)
    wn["witness_mean_dirs"] = wn[["witness_fit_odd_test_even", "witness_fit_even_test_odd"]].mean(1)
    wn.to_parquet(out / "witness_per_neuron.parquet", index=False)
    wc = (wn.groupby(["feature_version", "loo", "rm_cluster"])
          .agg(n_neurons=("unit_id", "size"), witness_mean_dirs=("witness_mean_dirs", "mean"),
               witness_fit_odd_test_even=("witness_fit_odd_test_even", "mean"),
               witness_fit_even_test_odd=("witness_fit_even_test_odd", "mean")).reset_index())
    wcw = wc.pivot_table(index=["feature_version", "rm_cluster", "n_neurons"], columns="loo",
                         values="witness_mean_dirs").reset_index()
    wcw.columns = [c if c in ("feature_version", "rm_cluster", "n_neurons") else f"witness_{c}_loo_mean_dirs"
                   for c in wcw.columns]
    wc.to_csv(out / "witness_per_cluster_long.csv", index=False)
    wcw.to_csv(out / "witness_per_cluster.csv", index=False)
    # consistency of the witness across CV directions
    cons = (wn.groupby(["feature_version", "loo"])
            .apply(lambda d: np.corrcoef(d.witness_fit_odd_test_even, d.witness_fit_even_test_odd)[0, 1])
            .rename("r_between_directions").reset_index())
    cons.to_csv(out / "witness_direction_consistency.csv", index=False)

    fig_nulls(records, out)
    for loo in ("mouse", "neuron"):
        fig_cluster_witness(wcw, versions, out, loo=loo)
    json.dump(dict(variant=a.variant, feature_matrix=str(fm_path), method_dir=str(md), n_neurons=int(len(pos)),
                   n_mice=int(len(np.unique(mice))), reward_free_cuts_s=REWARD_FREE_CUTS,
                   cut_rule="drop bins whose end (t_ctr + bin/2) is after the cut", bins_kept=kept,
                   groups=GROUPS, n_perm=a.n_perm, alpha=ALPHA, multiple_comparisons="none",
                   pca="99% variance, fit on the fit split", sigma2="median heuristic on the fit split "
                   "(random 8000-neuron subset when larger)", permutation="mouse-level Monte Carlo",
                   normalisation="none added (feature matrix as built)",
                   runtime_min=round((time.time() - t0) / 60, 1)),
              open(out / "config.json", "w"), indent=2)
    print(summ.round(4).to_string(index=False))
    print(cons.round(3).to_string(index=False))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
