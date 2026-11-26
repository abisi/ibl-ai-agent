"""GMM on the Figure 3 population matrix (rastermap v2 main) for SuppFigure3_robustness panel c (2026-11-26).
Reuses ssl-rastermap-psth-variants 046 (spectral embedding of X_odd + DP-GMM; method of unit_spikes_analysis
rastermap_psth gmm_utils_new.run_gmm) and 047 (fixed-K GMM on that embedding, --mode em = maximum likelihood, keeps K
components) with their paths redirected from the v1 archive variant to v2 main. Outputs:
<v2 main method_dir>/../gmm_bayesian_main/{bgmm_results.npz, bgmm_labels.csv, bgmm_k<K>_labels.csv, *_provenance.json}
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=. nice -n 10 .venv/bin/python <this> [--skip-embedding] --k 100 14
"""
import argparse
import importlib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "projects" / "ssl-rastermap-psth-variants" / "exploratory-analyses"))
V2 = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/v2/main")
ROOT = V2 / "rastermap_clustering/passive_active/whisker_auditory"
FM = ROOT / "unmerge_nobl_baseline_good_mua_stable/feature_matrix.npz"
METHOD = ROOT / "clustering/n100_passive_active_unmerge_baseline_whisker_auditory_combined_nobl/rastermap"
OUT = METHOD.parent / "gmm_bayesian_main"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-embedding", action="store_true")
    ap.add_argument("--k", type=int, nargs="+", default=[100, 14])
    a = ap.parse_args()
    m046 = importlib.import_module("046_bgmm_main_variant")
    m046.VARIANT, m046.ROOT, m046.FM, m046.METHOD, m046.OUT = V2, ROOT, FM, METHOD, OUT
    if not a.skip_embedding:
        m046.main()
    m047 = importlib.import_module("047_bgmm_fixed_k")
    m047.OUT = OUT
    for k in a.k:
        sys.argv = ["047", "--k", str(k), "--mode", "em"]
        m047.main()
    print("GMM V2 DONE", OUT, flush=True)
