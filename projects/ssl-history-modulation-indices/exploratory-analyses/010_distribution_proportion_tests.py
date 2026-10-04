"""Proportion and distribution difference tests, R+ vs R-, day 0 / learners
only (007_filter_day0_learners.py output -- the latest subset). Three tests
per metric, see history_lib for definitions:

1. Mouse-level proportion: fraction of units with index>0 per mouse,
   Mann-Whitney + Welch between cohorts -- "does the balance of enhanced-
   vs suppressed-modulation units differ by cohort" (distinct from "does
   the typical index value differ", already covered by 008_stats.py).
2. Neuron-level KS test (pooled units): standard two-sample distributional
   test, pseudoreplication-caveated.
3. Mouse-block permutation KS test: the rigorous version of #2, null built
   by permuting the R+/R- label across mice (not units).
"""
import sys
from pathlib import Path

sys.path.insert(0, "projects/ssl-history-modulation-indices/exploratory-analyses")
from history_lib import run_distribution_proportion_tests

IN_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/day0_learners_unit_indices.parquet")
OUT_PATH = Path("projects/ssl-history-modulation-indices/exploratory-analyses/010_distribution_proportion_results.json")

if __name__ == "__main__":
    run_distribution_proportion_tests(IN_PATH, OUT_PATH)
