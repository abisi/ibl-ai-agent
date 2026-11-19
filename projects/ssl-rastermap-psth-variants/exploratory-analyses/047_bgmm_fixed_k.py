"""047 -- Bayesian GMM with a fixed number of components (K = 24 metaclusters or K = 100 rastermap clusters) on the
spectral embedding of 046, for the one-to-one comparison with rastermap in Supplementary Figure 3
(user 2026-11-19: "the Bayesian GMM must have the same number of clusters as metaclusters or raw 100 clusters").

The Dirichlet-process run of 046 chooses its own number of components (19). Here the embedding of 046
(bgmm_results.npz: 13 spectral dimensions of the odd-trial PSTHs) is reused and a BayesianGaussianMixture with a finite
Dirichlet prior over exactly K components is fitted (other settings as 046 / run_gmm: diagonal covariances,
concentration 1, 2000 iterations, 10 initialisations, seed 42). Labels = predict (every neuron to its most likely
component); the number of components that receive neurons is reported.
The Bayesian weight prior still prunes components (K = 100 -> 29 used, run 2026-11-19), so --mode em (default) fits the
same mixture by maximum likelihood (sklearn GaussianMixture: diagonal covariances, k-means init, 10 inits, seed 42),
which keeps exactly K components; --mode bayes is the finite-Dirichlet run (outputs suffixed _bayes).
Output: <gmm_bayesian_main>/bgmm_k<K>_labels.csv (unit_ids, rastermap_cluster, bgmm_component) + _provenance.json
Run (haas): cd ~/code/unit_spikes_analysis; nice -n 10 .venv/bin/python <this> --k 24   (and --k 100)
"""
import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture

OUT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/"
                   "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin/rastermap_clustering/"
                   "passive_active/whisker_auditory/clustering/"
                   "n100_passive_active_unmerge_baseline_whisker_auditory_combined_nobl/gmm_bayesian_main")
P = dict(prior="dirichlet_distribution", concentration=1.0, cov="diag", max_iter=2000, n_init=10, seed=42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--mode", choices=["em", "bayes"], default="em")
    a = ap.parse_args()
    K, sfx = a.k, ("" if a.mode == "em" else "_bayes")
    r = np.load(OUT / "bgmm_results.npz", allow_pickle=True)
    t0 = time.time()
    if a.mode == "bayes":
        bg = BayesianGaussianMixture(n_components=K, covariance_type=P["cov"], weight_concentration_prior_type=P["prior"],
                                     weight_concentration_prior=P["concentration"], max_iter=P["max_iter"],
                                     n_init=P["n_init"], random_state=P["seed"]).fit(r["spectral"])
    else:
        bg = GaussianMixture(n_components=K, covariance_type=P["cov"], max_iter=P["max_iter"], n_init=P["n_init"],
                             random_state=P["seed"]).fit(r["spectral"])
    lab = bg.predict(r["spectral"])
    pd.DataFrame(dict(unit_ids=r["unit_ids"], rastermap_cluster=r["rastermap_labels"], bgmm_component=lab)).to_csv(
        OUT / f"bgmm_k{K}{sfx}_labels.csv", index=False)
    prov = dict(script="047_bgmm_fixed_k.py", mode=a.mode, embedding="bgmm_results.npz spectral (046)", K=K,
                n_used=int(len(np.unique(lab))), n_weight_above_1_over_10K=int((bg.weights_ > 1 / (10 * K)).sum()),
                converged=bool(bg.converged_), params=P, minutes=round((time.time() - t0) / 60, 1),
                date=time.strftime("%Y-%m-%d %H:%M"))
    json.dump(prov, open(OUT / f"bgmm_k{K}{sfx}_provenance.json", "w"), indent=1)
    print(prov, flush=True)


if __name__ == "__main__":
    main()
