"""Regenerate the ROC-dependent figures after the ROC results were converted to firing rates (026, 2026-09-30).

For each long-window lick-fixed variant: matrix summary (incl. publication layout + metacluster PSTH summary;
ROC categories read from <mouse>/whisker_<day>/roc_analysis/, now rate-based) and the sig-cluster figures (023).
Usage: python 043_regen_roc_figures.py [--variants ...] [--skip-023]
"""
import argparse
import importlib
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
drv = importlib.import_module("000_run_rastermap_variants")
from rastermap_psth.population_matrix_summary import run_matrix_summary  # noqa: E402

VARIANTS = ["qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin",
            "qc_good__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin",
            "qc_good_mua__learncat_all__loc0p25_grid10_nodes__lickfix_longwin",
            "qc_good__learncat_all__loc0p25_grid10_nodes__lickfix_longwin"]


def method_dir(variant):
    return sorted((drv.VARIANTS_ROOT / variant).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=VARIANTS)
    ap.add_argument("--skip-023", action="store_true")
    a = ap.parse_args()
    for v in a.variants:
        qc = ["good"] if v.startswith("qc_good__") else ["good", "mua"]
        md = method_dir(v)
        print(f"=== {v}: matrix summary ({md}) ===", flush=True)
        run_matrix_summary(md, drv.CONFIG_PATH, dict(unit_quality_label=qc, correct_lick_time=True), title_prefix=f"{v}: ")
        if not a.skip_023:
            print(f"=== {v}: sig-cluster figures (023) ===", flush=True)
            subprocess.run([sys.executable, "-u", str(HERE / "023_sig_cluster_figures.py"), "--variant", v], check=True)
    print("ALL DONE", flush=True)
