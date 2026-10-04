"""Attach Allen custom area labels (area_acronym_custom) and large-area
groups to the ssl_ephys units table, using the user's own allen_utils
(process_allen_labels with subdivide_areas=True, get_custom_area_groups),
per M:\\analysis\\Axel_Bisi\\Github\\ephys_utilities\\allen_utils\\allen_utils.py.

target_region comes from scripts/extract_ssl_target_region.py's output
(raw NWB electrode_group.location), merged on (session_id, probe_name).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities\ephys_utilities\allen_utils")
import allen_utils  # noqa: E402

DATASET_DIR = Path("reports/datasets/ssl_ephys/1.0.0")
DERIVED_DIR = Path("reports/ssl_analysis/derived")
OUT_PATH = DERIVED_DIR / "unit_area_labels.parquet"


def main() -> None:
    units = pd.read_parquet(DATASET_DIR / "metadata/units.parquet")
    sessions = pd.read_parquet(DATASET_DIR / "metadata/sessions.parquet")
    target_region = pd.read_parquet(DERIVED_DIR / "electrode_group_target_region.parquet")

    df = units.merge(sessions[["session_id", "subject_id"]], on="session_id", how="left")
    df = df.rename(columns={"subject_id": "mouse_id"})

    tr = target_region[["session_id", "probe_name", "target_region"]].drop_duplicates()
    before = len(df)
    df = df.merge(tr, on=["session_id", "probe_name"], how="left")
    assert len(df) == before, "target_region merge changed row count -- (session_id, probe_name) not unique in target_region table"

    n_missing_target = df["target_region"].isna().sum()
    print(f"Units missing target_region after merge: {n_missing_target} / {len(df)}")

    labeled = allen_utils.process_allen_labels(df, split_merge_areas=True)
    area_groups_from_name = allen_utils.get_custom_area_groups_from_name()
    labeled["area_group"] = labeled["area_acronym_custom"].map(area_groups_from_name)

    n_missing_group = labeled["area_group"].isna().sum()
    print(f"Units with area_acronym_custom not in a custom group: {n_missing_group} / {len(labeled)}")
    print()
    print("area_acronym_custom counts:")
    print(labeled["area_acronym_custom"].value_counts(dropna=False))
    print()
    print("area_group counts:")
    print(labeled["area_group"].value_counts(dropna=False))

    keep_cols = [
        "session_id", "cluster_id", "mouse_id", "probe_name", "target_region",
        "ccf_atlas_acronym", "ccf_atlas_parent_acronym", "area_acronym_custom", "area_group",
        "bc_label", "ks_label", "firing_rate",
    ]
    labeled[keep_cols].to_parquet(OUT_PATH, index=False)
    print(f"\nWrote {len(labeled)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
