"""046 -- Bayesian GMM (spectral embedding + DP-GMM) on the main rastermap variant, for Supplementary Figure 3
(user 2026-11-19: "show that my rastermap clusters are found by the alternate method, the Bayesian GMM").

Same method and defaults as unit_spikes_analysis rastermap_psth.run_clustering_new.run_gmm (gmm_utils_new), on the
same matrix rastermap was fitted on (cross-validated run: X_odd of feature_matrix.npz; rastermap clusters
= neuron_cluster_labels_cv.csv):
  1. N_PCA from the permutation test of gmm_utils_new.determine_n_pca (100 within-row shuffles, Bonferroni),
  2. row-centred PCA scores, Gaussian similarity S = exp(-d / sigma) with sigma tuned so that mean(S) = 0.5,
  3. normalised affinity D^-1/2 S D^-1/2: 14 leading eigenvectors (= the 14 smallest of the normalised Laplacian), the
     trivial one dropped -> 13-dimensional spectral embedding (gmm_n_spectral),
  4. BayesianGaussianMixture: 200 components, Dirichlet-process prior, concentration 1, diagonal covariances,
     2000 iterations, 10 initialisations, seed 42; components with weight > 1 / (10 * 200) kept, other neurons
     reassigned to the nearest active component (gmm_utils_new.bayesian_active_labels).
Memory-lean re-implementation of steps 2-3 (float32, in place, no identity matrix): the dense pipeline would need
~90 GB for 47,736 neurons; the result is the same up to float32 precision.
Output: <method_dir>/../gmm_bayesian_main/{bgmm_labels.csv (keys + rastermap and BGMM labels), bgmm_results.npz,
        bgmm_provenance.json}
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=. nice -n 10 .venv/bin/python \
  ~/code/ibl-ai-agent/projects/ssl-rastermap-psth-variants/exploratory-analyses/046_bgmm_main_variant.py
"""
import json
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.sparse.linalg import eigsh
from scipy.stats import ttest_1samp
from sklearn.decomposition import PCA
from sklearn.mixture import BayesianGaussianMixture

VARIANT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/"
                       "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin")
ROOT = VARIANT / "rastermap_clustering/passive_active/whisker_auditory"
FM = ROOT / "unmerge_nobl_baseline_good_mua/feature_matrix.npz"
METHOD = ROOT / "clustering/n100_passive_active_unmerge_baseline_whisker_auditory_combined_nobl/rastermap"
OUT = METHOD.parent / "gmm_bayesian_main"
P = dict(n_perm=100, n_spectral=13, target_mean_s=0.5, n_components=200, concentration=1.0, cov="diag",
         max_iter=2000, n_init=10, seed=42)
CHUNK = 2000


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def n_pca_permutation(Xc, n_perm, rng):
    """gmm_utils_new.determine_n_pca, vectorised shuffles"""
    n_max = min(Xc.shape[0] - 1, Xc.shape[1] - 1)
    real = PCA(n_components=n_max, svd_solver="full").fit(Xc).explained_variance_
    null0 = np.array([PCA(n_components=1, svd_solver="randomized", random_state=j).fit(rng.permuted(Xc, axis=1))
                      .explained_variance_[0] for j in range(n_perm)])
    p = np.array([ttest_1samp(null0, popmean=r).pvalue for r in real])
    idx = np.where((real > null0.mean()) & (p < 0.05 / n_max))[0]
    return (int(idx.max() + 1) if len(idx) else 10), real, null0


def sq_dist_rows(Z, i0, i1, sq):
    d = sq[i0:i1, None] + sq[None, :] - 2.0 * (Z[i0:i1] @ Z.T)
    np.maximum(d, 0, out=d)
    return np.sqrt(d, out=d)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    fm = np.load(FM, allow_pickle=True)
    X = fm["X_odd"].astype(np.float64)
    unit_ids = fm["unit_ids"]
    lab = pd.read_csv(METHOD / "neuron_cluster_labels_cv.csv")
    assert np.array_equal(lab.unit_ids.to_numpy(), unit_ids), "label table and feature matrix unit order differ"
    Xc = X - X.mean(1, keepdims=True)
    log("N_PCA permutation test", Xc.shape)
    n_pca, real, null0 = n_pca_permutation(Xc, P["n_perm"], rng)
    log("n_pca", n_pca)
    Z = PCA(n_components=n_pca, svd_solver="full").fit_transform(Xc).astype(np.float32)
    N = len(Z)
    sq = (Z.astype(np.float64) ** 2).sum(1).astype(np.float32)
    D = np.empty((N, N), np.float32)
    for i in range(0, N, CHUNK):
        D[i:i + CHUNK] = sq_dist_rows(Z, i, min(N, i + CHUNK), sq)
    np.fill_diagonal(D, 0)
    log("distances done")

    def mean_s(sig):
        tot = 0.0
        for i in range(0, N, CHUNK):
            tot += np.exp(-D[i:i + CHUNK] / sig, dtype=np.float32).sum(dtype=np.float64)
        return (tot - N) / (N * (N - 1)) - P["target_mean_s"]

    sigma = brentq(mean_s, 1e-5, 1e4, xtol=1e-6, maxiter=300)
    log("sigma", sigma)
    for i in range(0, N, CHUNK):                       # D -> S in place
        blk = D[i:i + CHUNK]
        np.divide(blk, -sigma, out=blk)
        np.exp(blk, out=blk)
    deg = D.sum(1, dtype=np.float64)
    dis = (deg ** -0.5).astype(np.float32)
    for i in range(0, N, CHUNK):                       # S -> D^-1/2 S D^-1/2 in place
        D[i:i + CHUNK] *= dis[i:i + CHUNK, None]
        D[i:i + CHUNK] *= dis[None, :]
    vals, vecs = eigsh(D, k=P["n_spectral"] + 1, which="LA", tol=1e-6)
    del D
    o = np.argsort(-vals)
    eig_L = 1.0 - vals[o]                              # eigenvalues of the normalised Laplacian, ascending
    spectral = vecs[:, o][:, 1:].astype(np.float64)
    log("eigenvalues of L", np.round(eig_L, 5).tolist())

    bg = BayesianGaussianMixture(n_components=P["n_components"], covariance_type=P["cov"],
                                 weight_concentration_prior_type="dirichlet_process",
                                 weight_concentration_prior=P["concentration"], max_iter=P["max_iter"],
                                 n_init=P["n_init"], random_state=P["seed"]).fit(spectral)
    thr = 1.0 / (10 * P["n_components"])
    active = np.where(bg.weights_ > thr)[0]
    raw = bg.predict(spectral)
    remap = {k: (int(np.searchsorted(active, k)) if k in active else
                 int(np.argmin(np.linalg.norm(bg.means_[k] - bg.means_[active], axis=1))))
             for k in range(P["n_components"])}
    bl = np.array([remap[r] for r in raw])
    log("active components", len(active), "converged", bg.converged_)
    keys = ["unit_ids", "mouse_id", "session_id", "electrode_group", "cluster_id", "reward_group",
            "area_acronym_custom"]
    lab[keys].assign(rastermap_cluster=lab.cluster_label, bgmm_component=bl).to_csv(OUT / "bgmm_labels.csv", index=False)
    np.savez_compressed(OUT / "bgmm_results.npz", spectral=spectral, eig_L=eig_L, bgmm_labels=bl,
                        rastermap_labels=lab.cluster_label.to_numpy(), weights=bg.weights_, pca_var=real,
                        pca_null0=null0, unit_ids=unit_ids)
    json.dump(dict(script="046_bgmm_main_variant.py", matrix="feature_matrix.npz X_odd (rastermap CV fit matrix)",
                   n_neurons=int(N), n_pca=n_pca, sigma=float(sigma), n_active=int(len(active)),
                   converged=bool(bg.converged_), weight_threshold=thr, params=P,
                   date=time.strftime("%Y-%m-%d %H:%M")), open(OUT / "bgmm_provenance.json", "w"), indent=1)
    log("saved", OUT)


if __name__ == "__main__":
    main()
