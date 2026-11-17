"""Build the final filtered unit table for the whisker/auditory PSTH + LMM
analysis: merges coverage, area labels, and mouse-level reward_group /
learning_category / exclusion flags, then applies the inclusion filter.

Inclusion criteria (documented judgment calls, not project-established
defaults -- see reports/ssl_analysis/ANALYSIS_NOTES.md):
- quality_label in {good, mua} (bc_label up to 2026-11-09).
- coverage_ratio >= 0.9 (last_spike - first_spike, over session duration).
  Chosen from the observed distribution: Q1=0.982, median=0.998, so 0.9
  only removes units clearly in the degraded/drop-out tail, not the bulk.
- Direct cross-check: >=1 spike in passive_pre AND >=1 spike in passive_post
  (coverage_ratio is used as the primary continuous proxy per the user's
  request; this direct count is reported alongside it, not substituted).
- Mouse-level: exclude==0 AND exclude_ephys==0 (joint_mouse_reference_weight.xlsx).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DERIVED_DIR = Path("reports/ssl_analysis/derived")
COVERAGE_THRESHOLD = 0.9


def main() -> None:
    coverage = pd.read_parquet(DERIVED_DIR / "unit_coverage.parquet")
    # coverage here = unit_coverage.parquet (span over session duration); the area table also carries the quality-label
    # presence / coverage columns since 2026-11-06, dropped to keep one definition
    area = pd.read_parquet(DERIVED_DIR / "unit_area_labels.parquet").drop(columns=["presence_ratio", "coverage_ratio"], errors="ignore")
    mouse_ref = pd.read_parquet(DERIVED_DIR / "mouse_reference.parquet")

    df = area.merge(coverage, on=["session_id", "cluster_id"], how="inner")
    print(f"area x coverage merge: {len(df)} rows (area table had {len(area)}, coverage had {len(coverage)})")

    df = df.merge(
        mouse_ref[["subject_id", "reward_group", "learning_category", "exclude", "exclude_ephys"]],
        left_on="mouse_id", right_on="subject_id", how="left",
    )

    n_total = len(df)
    # quality_label, not bc_label (user 2026-09-11: "it is not the bc_label used but the quality_label"; bc_label up to 2026-11-09)
    quality_ok = df["quality_label"].isin(["good", "mua"])
    coverage_ok = df["coverage_ratio"] >= COVERAGE_THRESHOLD
    direct_pre_post_ok = (df["n_spikes_passive_pre"] > 0) & (df["n_spikes_passive_post"] > 0)
    mouse_ok = (df["exclude"] == 0) & (df["exclude_ephys"] == 0)

    print()
    print(f"n_total (area x coverage x mouse_ref merged): {n_total}")
    print(f"pass quality_label good/mua: {quality_ok.sum()}")
    print(f"pass coverage_ratio >= {COVERAGE_THRESHOLD}: {coverage_ok.sum()}")
    print(f"pass direct pre&post spike presence: {direct_pre_post_ok.sum()}")
    print(f"coverage_ok but NOT direct_pre_post_ok: {(coverage_ok & ~direct_pre_post_ok).sum()}")
    print(f"direct_pre_post_ok but NOT coverage_ok: {(direct_pre_post_ok & ~coverage_ok).sum()}")
    print(f"pass mouse exclude filters: {mouse_ok.sum()}")

    keep = quality_ok & coverage_ok & mouse_ok
    final = df[keep].copy()
    # passive pre / post analyses: ephys_utilities data_utils.keep_passive_data = mice with passive pre AND post data
    # (keep_passive_mice) and the session x probe exclusions of keep_passive_units (e.g. AB155_20250208_175103 imec4);
    # reminder: always select passive data there. keep_passive_units alone (up to 2026-11-09) kept AB144 / MH013.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from axel_bisi_paths import axel_bisi_root  # M:\ on Windows, /mnt/lsens-analysis on haas
    sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))
    from ephys_utilities.helpers.data_utils import keep_passive_data
    n0 = len(final)
    final = keep_passive_data(final.assign(electrode_group=final["probe_name"])).drop(columns="electrode_group")
    print(f"passive selection (keep_passive_data): {n0 - len(final)} units dropped")
    print()
    print(f"Final analysis unit set: {len(final)} / {n_total}")
    print(f"Mice retained: {final['mouse_id'].nunique()}")
    print(f"Sessions retained: {final['session_id'].nunique()}")
    print()
    print("By reward_group:")
    print(final.groupby("reward_group")["mouse_id"].nunique())
    print()
    print("By area_group (units):")
    print(final["area_group"].value_counts(dropna=False))

    out_path = DERIVED_DIR / "analysis_units.parquet"
    final.to_parquet(out_path, index=False)
    print(f"\nWrote {len(final)} rows to {out_path}")


if __name__ == "__main__":
    main()
