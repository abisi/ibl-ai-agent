"""Smoke test the Path B coverage pipeline on 5 known multi-probe sessions
(picked from ssl_ephys's compressed sessions.parquet n_probes>=2, cheap
metadata-only lookup, before committing to a full-population Path B run) --
times the load so a full-population runtime estimate can be given to Axel
per AGENTS.md's 'estimate before large run' rule.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
lib = importlib.import_module("000_coverage_lib")

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

# 5 multi-probe sessions spanning learning/expert, from the compressed
# ssl_ephys sessions.parquet n_probes ranking (metadata-only pick, see
# question.md -- does not substitute for the mandatory Path B analysis itself).
SMOKE_FILES = [
    "AB130_20240902_123634.nwb",  # learning (whisker_0), 4 probes, 3787 units
    "MH023_20250316_110814.nwb",  # learning (whisker_0), 4 probes, 3773 units
    "MH030_20250501_151231.nwb",  # learning (whisker_0), 4 probes, 3800 units
    "MH030_20250503_154256.nwb",  # expert (whisker_+2), 4 probes, 3211 units
    "MH062_20260113_125836.nwb",  # expert (whisker_+4), 5 probes, 3949 units
]


def main() -> None:
    t0 = time.time()
    unit_table, trial_table = lib.load_units(SMOKE_FILES, day_to_analyze="all", max_workers=5)
    elapsed = time.time() - t0
    print(f"\nLoad took {elapsed:.1f}s for {len(SMOKE_FILES)} sessions "
          f"({elapsed / len(SMOKE_FILES):.1f}s/session)")

    ref_df = pd.read_excel(lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = lib.apply_mouse_filters(unit_table, ref_df)

    print("\nUnits per session x probe (electrode_group):")
    print(unit_table.groupby(["session_id", "electrode_group"]).size())

    print("\nUnits per session x area_group_coarse:")
    print(unit_table.groupby(["session_id", "area_group_coarse"]).size())

    print("\nUnits per session x area_acronym_custom (top rows):")
    print(unit_table.groupby(["session_id", "area_acronym_custom"]).size().sort_values(ascending=False).head(20))

    for area_col in ["area_group_coarse", "area_acronym_custom"]:
        counts = lib.area_unit_counts(unit_table, area_col)
        pairs = lib.valid_area_pairs(counts, area_col)
        print(f"\n[{area_col}] valid area-pairs (>=20 units each, this 5-session smoke set): {len(pairs)}")
        print(pairs.to_string(index=False))

    safe_cols = [c for c in unit_table.columns if c not in ("spike_times", "location")]
    unit_table[safe_cols].to_parquet(ARTIFACTS_DIR / "smoke_unit_table.parquet", index=False)
    print(f"\nWrote {ARTIFACTS_DIR / 'smoke_unit_table.parquet'}")


if __name__ == "__main__":
    main()
