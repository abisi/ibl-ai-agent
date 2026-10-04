"""Run rastermap_psth.population_matrix_summary.run_matrix_summary for every rastermap variant
(000_run_rastermap_variants.py outputs) → <method_dir>/matrix_summary/.

Run on haas from ~/code/unit_spikes_analysis:
    PYTHONPATH=~/code/NWB_reader:. .venv/bin/python -u <this file> [--test] [--variants ...]
"""
import argparse
import pathlib
import traceback

from rastermap_psth.population_matrix_summary import run_matrix_summary

VARIANTS_ROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants")
CONFIG_PATH = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/unit_spikes_analysis/rastermap_psth/config.yaml")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None)
    a = ap.parse_args()
    root = VARIANTS_ROOT / "_test" if a.test else VARIANTS_ROOT
    dirs = sorted(p for p in root.glob("qc_*/rastermap_clustering/*/*/clustering/*/rastermap")
                  if (p / "stats" / "f_matrix.npz").exists())
    for md in dirs:
        variant = next(p for p in md.parts if p.startswith("qc_"))
        if a.variants and variant not in a.variants:
            continue
        print(f"\n=== {variant} ===", flush=True)
        try:
            qc = ["good"] if variant.startswith("qc_good__") else ["good", "mua"]
            out = run_matrix_summary(md, CONFIG_PATH, dict(unit_quality_label=qc), title_prefix=f"{variant}: ")
            print(f"=== {variant} done → {out}", flush=True)
        except Exception:
            print(f"=== {variant} FAILED ===\n{traceback.format_exc()}", flush=True)
    print("ALL DONE", flush=True)
