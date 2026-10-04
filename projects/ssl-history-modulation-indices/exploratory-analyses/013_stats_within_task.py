"""Cohort (R+ vs R-) difference tests for RHMI/EHMI, within-task
(auditory-only) day 0/learners data (012_compute_within_task_auditory.py
output). Same test battery as 008/010, run on the within-task table.
"""
import sys
from pathlib import Path

sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from history_lib import run_stats, run_distribution_proportion_tests

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/within_task_auditory_day0_learners.parquet")
STATS_OUT = Path("projects/ssl-history-modulation-indices/exploratory-analyses/013_stats_results.json")
DIST_OUT = Path("projects/ssl-history-modulation-indices/exploratory-analyses/013_distribution_proportion_results.json")

if __name__ == "__main__":
    run_stats(IN_PATH, STATS_OUT)
    print("\n" + "=" * 60 + "\n")
    run_distribution_proportion_tests(IN_PATH, DIST_OUT)
