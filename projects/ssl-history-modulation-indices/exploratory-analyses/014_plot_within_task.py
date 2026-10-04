"""Publication-quality figures for the within-task (auditory-only) R+ vs R-
comparison (012_compute_within_task_auditory.py / 013_stats_within_task.py
output). Reuses 006's location/area plots and 011's ECDF/proportion plots
with within-task paths and a distinct filename prefix ("014").
"""
import sys
from pathlib import Path

sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from importlib import import_module

plot_mod = import_module("006_plot_publication_figure")
dist_mod = import_module("011_plot_distribution_proportion")

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/within_task_auditory_day0_learners.parquet")
STATS_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/013_stats_results.json")
DIST_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/013_distribution_proportion_results.json")
OUT_DIR = Path("projects/ssl-history-modulation-indices/exploratory-analyses")
TITLE_SUFFIX = "within task (auditory-only), day 0, learners only"

if __name__ == "__main__":
    plot_mod.run(in_path=IN_PATH, stats_path=STATS_PATH, out_dir=OUT_DIR, prefix="014", title_suffix=TITLE_SUFFIX)
    dist_mod.run(in_path=IN_PATH, results_path=DIST_PATH, out_dir=OUT_DIR, prefix="014", title_suffix=TITLE_SUFFIX)
