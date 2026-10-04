"""Sensitivity of the reward-free MMD^2 result (016) to the kernel bandwidth and the distance metric.

Same data, reward-free cuts, PCA (99%, fit split), mouse-level permutations and odd/even CV as 016; only the kernel
changes:  RBF with sigma^2 x {0.25, 0.5, 1, 2, 4} (median heuristic = 1), Laplacian (L1, median scale), cosine
(RBF on unit-norm rows: response shape only), energy distance (Euclidean, no bandwidth). Scales are label-blind,
fit on the FIT split. Output -> <method_dir>/<run-dir>/kernel_sensitivity/{kernel_sensitivity.csv, *.png/pdf}
"""
import argparse
import json
import importlib
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis/rastermap_psth"))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
import cluster_mmd_analysis_new as mmd                                    # noqa: E402
d017 = importlib.import_module("017_mmd_method_figures")

SETTINGS = [("rbf", 0.25), ("rbf", 0.5), ("rbf", 1.0), ("rbf", 2.0), ("rbf", 4.0),
            ("laplacian", 1.0), ("cosine", 1.0), ("energy", None)]
METRIC = {"rbf": "sqeuclidean", "cosine": "sqeuclidean", "laplacian": "cityblock", "energy": "euclidean"}
VERSIONS = ["Whisker miss", "Auditory hit", "Auditory hit (lick)", "Auditory post", "Whisker hit",
            "Whisker hit (lick)", "Spont. lick", "Whisker pre", "full"]


def label(k, f):
    if k == "rbf":
        return f"RBF σ²×{f:g}"
    return {"laplacian": "Laplacian (L1)", "cosine": "Cosine RBF", "energy": "Energy (L2)"}[k]


def fit_scale(fit_proj, kernel, factor, seed):
    if kernel == "energy":
        return None
    X = fit_proj
    if kernel == "cosine":
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    sub = X if len(X) <= 8000 else X[rng.choice(len(X), 8000, replace=False)]
    return float(np.median(pdist(sub, METRIC[kernel]))) * factor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-perm", type=int, default=999)
    ap.add_argument("--versions", nargs="*", default=VERSIONS)
    ap.add_argument("--from-csv", action="store_true", help="rebuild summary/figure from kernel_sensitivity.csv")
    a = ap.parse_args()
    _, _, md = d017.d016.paths(a.variant)
    out = md / a.run_dir / "kernel_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**mmd.DEFAULT_MMD_CFG, "cohort_labels": ("R+", "R-"), "pca_test_n_components": None,
           "pca_test_variance_retained": 0.99}
    rows = []
    for vi, v in enumerate([] if a.from_csv else a.versions):
        base = d017.prepare(a.variant, v, "fit_odd_test_even", None)        # loads data once per version
        for di, direction in enumerate(("fit_odd_test_even", "fit_even_test_odd")):
            D = base if direction == "fit_odd_test_even" else d017.prepare(a.variant, v, direction, None)
            for si, (kernel, factor) in enumerate(SETTINGS):
                seed = 10000 * vi + 100 * di + si
                scale = fit_scale(D["fit_proj"], kernel, factor, seed)
                S, kdiag, _ = mmd.kernel_mouse_sums_general(D["test_proj"], D["codes"], len(D["uniq"]),
                                                            kernel=kernel, scale=scale)
                res = mmd.mouse_block_permutation_test(S, D["codes"], D["is_pos"], n_perm=a.n_perm,
                                                       rng=np.random.default_rng(seed), kdiag=kdiag)
                _, w = mmd.witness_from_mouse_sums(S, D["codes"], D["is_pos"], kdiag=kdiag)
                wc = pd.Series(w).groupby(D["rm_labels"]).mean()
                rows.append(dict(feature_version=v, direction=direction, kernel=kernel, factor=factor,
                                 setting=label(kernel, factor), mmd2_obs=res["mmd2_obs"], mmd2_z=res["mmd2_z"],
                                 p_value=res["p_value"], scale=scale,
                                 witness_cluster_means=wc.to_json()))
                print(f"[018] {v:>20s} {direction} {label(kernel, factor):>15s}: z={res['mmd2_z']:.2f} "
                      f"p={res['p_value']:.3g}", flush=True)
    if a.from_csv:
        df = pd.read_csv(out / "kernel_sensitivity.csv")
    else:
        df = pd.DataFrame(rows)
        df.to_csv(out / "kernel_sensitivity.csv", index=False)
    as_series = lambda js: pd.Series(json.loads(js)).rename(index=int).sort_index()

    # witness pattern stability: correlation of (centered) cluster means with the default RBF, per version
    stab = []
    for (v, d), g in df.groupby(["feature_version", "direction"]):
        ref = as_series(g[g.setting == "RBF σ²×1"].witness_cluster_means.iloc[0])
        for _, r in g.iterrows():
            s = as_series(r.witness_cluster_means)
            stab.append(dict(feature_version=v, direction=d, setting=r.setting,
                             r_centered_cluster_witness=np.corrcoef(ref - ref.mean(), s - s.mean())[0, 1]))
    stab = pd.DataFrame(stab)
    stab.to_csv(out / "witness_pattern_stability.csv", index=False)

    # figure: z (mean of directions) per version x setting; * = p<0.05 in both directions
    settings = [label(k, f) for k, f in SETTINGS]
    z = df.groupby(["feature_version", "setting"]).mmd2_z.mean().unstack()[settings].loc[a.versions]
    both = df.groupby(["feature_version", "setting"]).p_value.max().unstack()[settings].loc[a.versions] < 0.05
    r = stab.groupby(["feature_version", "setting"]).r_centered_cluster_witness.mean().unstack()[settings].loc[a.versions]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=200)
    for ax, M, cmap, vmin, vmax, ttl in ((axes[0], z, "Greys", 0, max(3, np.nanmax(z.values)), "MMD² z (mean of CV directions)"),
                                          (axes[1], r, "viridis", 0, 1, "Witness pattern vs RBF σ²×1\n(r of centered cluster means)")):
        im = ax.imshow(M.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(settings))); ax.set_xticklabels(settings, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(len(M))); ax.set_yticklabels(M.index, fontsize=7)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                txt = f"{M.values[i, j]:.1f}" if M is z else f"{M.values[i, j]:.2f}"
                if M is z and both.values[i, j]:
                    txt += "*"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6,
                        color="w" if (M.values[i, j] - vmin) / (vmax - vmin + 1e-9) > 0.55 else "k")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
        ax.set_title(ttl, fontsize=8)
    fig.suptitle("Kernel sensitivity (* = p<0.05 in both CV directions, uncorrected)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "kernel_sensitivity.png"); fig.savefig(out / "kernel_sensitivity.pdf")
    print(z.round(1).to_string()); print(both.to_string()); print(r.round(2).to_string())
    print("ALL DONE")


if __name__ == "__main__":
    main()
