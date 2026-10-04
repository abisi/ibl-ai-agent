"""Regenerate every matrix summary with the current population_matrix_summary defaults
(2026-09-27: ROC columns = modality preference, motor (lick-related, def. C), decision, gated;
per-column colour ranges; no height-proportional cluster rows). Old matrix_* figures in each target
folder are deleted first so no stale layouts remain.

Targets: variant summaries (matrix_summary/) and contiguous-segment summaries
(matrix_summary_segments[_cutX]/ built from segment_stats[_cutX]/segment_map.csv + segment_linkage.npz).
"""
import pathlib
import traceback

from rastermap_psth.population_matrix_summary import run_matrix_summary

ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CONFIG = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml")
VARIANTS = ["qc_good__learncat_all", "qc_good__learncat_good_moderate",
            "qc_good__learncat_good_moderate__noFR", "qc_good__learncat_good_moderate__grid10_nodes",
            "qc_good_mua__learncat_all", "qc_good_mua__learncat_good_moderate",
            "qc_good_mua__learncat_good_moderate__grid10_nodes"]
SEGMENTS = [("qc_good__learncat_good_moderate__grid10_nodes", ""),
            ("qc_good__learncat_good_moderate__grid10_nodes", "_cut0p40"),
            ("qc_good__learncat_good_moderate__grid10_nodes", "_cut0p45"),
            ("qc_good_mua__learncat_good_moderate__grid10_nodes", "_cut0p40")]


def method_dir(v):
    return sorted((ROOT / v).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]


def clear(folder):
    n = 0
    if folder.exists():
        for f in folder.glob("matrix_*"):
            f.unlink(); n += 1
    print(f"  removed {n} old matrix_* files from {folder.name}", flush=True)


for v in VARIANTS:
    md = method_dir(v)
    print(f"\n=== {v} ===", flush=True)
    try:
        clear(md / "matrix_summary")
        qc = ["good"] if v.startswith("qc_good__") else ["good", "mua"]
        run_matrix_summary(md, CONFIG, dict(unit_quality_label=qc), title_prefix=f"{v}: ")
        print(f"=== {v} done", flush=True)
    except Exception:
        print(f"=== {v} FAILED ===\n{traceback.format_exc()}", flush=True)

for v, tag in SEGMENTS:
    md = method_dir(v)
    seg = md / f"segment_stats{tag}"
    print(f"\n=== {v} segments{tag or ' (cut 0.35)'} ===", flush=True)
    try:
        clear(md / f"matrix_summary_segments{tag}")
        cut = tag.replace("_cut", "").replace("p", ".") or "0.35"
        qc = ["good"] if v.startswith("qc_good__") else ["good", "mua"]
        run_matrix_summary(md, CONFIG, dict(unit_quality_label=qc), title_prefix=f"{v}: ",
                           families_csv=seg / "segment_map.csv", out_name=f"matrix_summary_segments{tag}",
                           family_label=f"Segment (contiguous, 1-r<={cut})", linkage_npz=seg / "segment_linkage.npz")
        print(f"=== {v} segments{tag} done", flush=True)
    except Exception:
        print(f"=== {v} segments{tag} FAILED ===\n{traceback.format_exc()}", flush=True)
print("ALL DONE", flush=True)
