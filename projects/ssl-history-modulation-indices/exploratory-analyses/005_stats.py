"""Cohort (R+ vs R-) difference tests for RHMI/EHMI, full cohort
(004_compute_full_cohort.py output). See history_lib.run_stats for the two
complementary approaches (mouse-level Mann-Whitney+Welch, neuron-level LMM
with mouse random intercept).
"""
import sys
from pathlib import Path

sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from history_lib import run_stats

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/full_cohort_unit_indices.parquet")
OUT_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/005_stats_results.json")

if __name__ == "__main__":
    run_stats(IN_PATH, OUT_PATH)
