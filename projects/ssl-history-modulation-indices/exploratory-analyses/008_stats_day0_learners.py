"""Cohort (R+ vs R-) difference tests for RHMI/EHMI, day 0 (learning-stage)
sessions and learner mice only (007_filter_day0_learners.py output). See
history_lib.run_stats for the two complementary approaches.
"""
import sys
from pathlib import Path

sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from history_lib import run_stats

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/day0_learners_unit_indices.parquet")
OUT_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/008_stats_results.json")

if __name__ == "__main__":
    run_stats(IN_PATH, OUT_PATH)
