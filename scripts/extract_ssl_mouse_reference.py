"""Load the mouse-level reward_group/learning_category reference sheet
(ssl-load/references/ssl_loading_policy.md Path B source) and cross-check
its subject coverage against the built ssl_ephys dataset's subjects table.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

XLSX_PATH = Path(r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx")
SUBJECTS_PATH = Path("reports/datasets/ssl_ephys/1.0.0/metadata/subjects.parquet")
OUT_DIR = Path("reports/ssl_analysis/derived")
OUT_PATH = OUT_DIR / "mouse_reference.parquet"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = pd.read_excel(XLSX_PATH, sheet_name="Sheet1")
    ref = ref.rename(columns={"mouse_id": "subject_id"})
    ref.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(ref)} mouse reference rows to {OUT_PATH}")

    subjects = pd.read_parquet(SUBJECTS_PATH)
    subject_col = "subject_id" if "subject_id" in subjects.columns else subjects.columns[0]
    built_ids = set(subjects[subject_col].unique())
    ref_ids = set(ref["subject_id"].dropna().unique())

    missing_from_ref = sorted(built_ids - ref_ids)
    missing_from_built = sorted(ref_ids - built_ids)
    print(f"Subjects in built ssl_ephys dataset: {len(built_ids)}")
    print(f"Subjects in reference sheet: {len(ref_ids)}")
    print(f"Built subjects missing from reference sheet ({len(missing_from_ref)}): {missing_from_ref}")

    have_reward = ref[ref["subject_id"].isin(built_ids)]
    print()
    print("reward_group among built subjects:")
    print(have_reward["reward_group"].value_counts(dropna=False))
    print()
    print("learning_category among built subjects:")
    print(have_reward["learning_category"].value_counts(dropna=False))
    print()
    print("exclude / exclude_ephys among built subjects:")
    print(have_reward[["exclude", "exclude_ephys"]].value_counts(dropna=False))


if __name__ == "__main__":
    main()
