"""Publication-quality RHMI/EHMI figures, day 0 (learning-stage) sessions
and learner mice only (007_filter_day0_learners.py / 008_stats_day0_learners.py
output). Reuses 006_plot_publication_figure.py's plotting code with
different input/output paths.
"""
import sys
from pathlib import Path

sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from importlib import import_module

plot_mod = import_module("006_plot_publication_figure")

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/day0_learners_unit_indices.parquet")
STATS_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/008_stats_results.json")
OUT_DIR = Path("projects/ssl-history-modulation-indices/exploratory-analyses")

if __name__ == "__main__":
    plot_mod.run(in_path=IN_PATH, stats_path=STATS_PATH, out_dir=OUT_DIR, prefix="009",
                 title_suffix="day 0, learners only")
