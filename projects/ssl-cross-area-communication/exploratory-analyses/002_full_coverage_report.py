"""Step 1 deliverable: recording-pair coverage report, full population.
Path B load (all 119 has_ephys sessions, day_to_analyze='all'), mouse
filters, both area hierarchy levels, >=20-unit floor, valid-pair
enumeration, tabulated by cohort x day-stage x population scope. This is
the mandatory checkpoint before any CCA code is written (question.md).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
lib = importlib.import_module("000_coverage_lib")

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

_DAY_RE = re.compile(r"whisker(?:_on_\d+_opto|_off_\d+_opto)?_([+-]?\d+)")


def day_stage_from_description(desc: str) -> str | None:
    m = _DAY_RE.match(str(desc))
    if not m:
        return None
    day = int(m.group(1))
    if day < 0:
        return None
    return "learning" if day == 0 else "expert"


def main() -> None:
    filenames = pd.read_csv(ARTIFACTS_DIR / "ephys_session_filenames.csv")["source_filename"].tolist()
    print(f"Loading {len(filenames)} ephys sessions via Path B (day_to_analyze='all', max_workers=12)...")
    t0 = time.time()
    unit_table, trial_table = lib.load_units(filenames, day_to_analyze="all", max_workers=12)
    print(f"Full load took {time.time() - t0:.1f}s")

    ref_df = pd.read_excel(lib.REF_XLSX, sheet_name="Sheet1")
    unit_table = lib.apply_mouse_filters(unit_table, ref_df)
    unit_table = unit_table[unit_table["quality_label"] != "non-soma"].copy()

    # Checkpoint save immediately after the ~20min Path B load, before any
    # further processing -- insurance against losing the load to a downstream
    # bug (happened once already: an ibl_ai_agent import not on this conda env).
    safe_cols_ckpt = [c for c in unit_table.columns if c not in ("spike_times", "location")]
    unit_table[safe_cols_ckpt].to_parquet(ARTIFACTS_DIR / "full_unit_table_metadata_checkpoint.parquet", index=False)
    print(f"Checkpoint-saved {len(unit_table)} units to full_unit_table_metadata_checkpoint.parquet")

    sess_desc = trial_table.drop_duplicates("session_id").set_index("session_id")
    # session_description isn't on trial_table; pull it from the sessions.parquet mapping instead.
    # (Local parquet path, not ibl_ai_agent.data_locations.resolve_dataset_dir -- that package
    # isn't on the bwa conda env's path, only .venv's; see 004_small_subset_pipeline.py's same fix.)
    REPO_ROOT = Path(__file__).resolve().parents[3]
    ssl_sessions = pd.read_parquet(REPO_ROOT / "reports" / "datasets" / "ssl_ephys" / "1.0.0" / "metadata" / "sessions.parquet")
    desc_map = ssl_sessions.set_index("session_id")["session_description"].to_dict()
    unit_table["day_stage"] = unit_table["session_id"].map(lambda s: day_stage_from_description(desc_map.get(s)))
    n_before = len(unit_table)
    unit_table = unit_table[unit_table["day_stage"].notna()]
    print(f"Day-stage parse (whisker-training only, day>=0): {n_before} -> {len(unit_table)} units, "
          f"{unit_table['session_id'].nunique()} sessions")

    safe_cols = [c for c in unit_table.columns if c not in ("spike_times", "location")]
    unit_table[safe_cols].to_parquet(ARTIFACTS_DIR / "full_unit_table_metadata.parquet", index=False)
    print(f"Wrote {ARTIFACTS_DIR / 'full_unit_table_metadata.parquet'} ({len(unit_table)} units)")

    scopes = {
        "entire_dataset": unit_table["reward_group"].isin(["R+", "R-"]),
        "learners_only": unit_table["learning_category"].isin(["good", "moderate"]),
    }

    summary_rows = []
    pair_frames = []
    for area_col, level_name in [("area_group_coarse", "coarse"), ("area_acronym_custom", "fine")]:
        for scope_name, scope_mask in scopes.items():
            scoped = unit_table[scope_mask]
            for day_stage in ["learning", "expert"]:
                ds = scoped[scoped["day_stage"] == day_stage]
                counts = lib.area_unit_counts(ds, area_col)
                pairs = lib.valid_area_pairs(counts, area_col)
                if len(pairs):
                    pairs["level"] = level_name
                    pairs["scope"] = scope_name
                    pairs["day_stage"] = day_stage
                    sess_cohort = ds.drop_duplicates("session_id").set_index("session_id")["reward_group"]
                    sess_mouse = ds.drop_duplicates("session_id").set_index("session_id")["mouse_id"]
                    pairs["reward_group"] = pairs["session_id"].map(sess_cohort)
                    pairs["mouse_id"] = pairs["session_id"].map(sess_mouse)
                    pair_frames.append(pairs)
                n_sessions_with_ephys = ds["session_id"].nunique()
                for cohort in ["R+", "R-"]:
                    coh_pairs = pairs[pairs.get("reward_group", pd.Series(dtype=object)) == cohort] if len(pairs) else pd.DataFrame()
                    n_sess_cohort = ds.loc[ds["reward_group"] == cohort, "session_id"].nunique()
                    n_mice_cohort = ds.loc[ds["reward_group"] == cohort, "mouse_id"].nunique()
                    summary_rows.append({
                        "level": level_name, "scope": scope_name, "day_stage": day_stage, "reward_group": cohort,
                        "n_sessions_total": n_sess_cohort, "n_mice_total": n_mice_cohort,
                        "n_sessions_with_valid_pair": coh_pairs["session_id"].nunique() if len(coh_pairs) else 0,
                        "n_distinct_area_pairs": coh_pairs[["area_a", "area_b"]].drop_duplicates().shape[0] if len(coh_pairs) else 0,
                        "n_pair_instances": len(coh_pairs),
                    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ARTIFACTS_DIR / "coverage_summary.csv", index=False)
    print("\n=== Coverage summary (sessions/mice with >=1 valid area-pair, >=20 units/area) ===")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    all_pairs = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    all_pairs.to_parquet(ARTIFACTS_DIR / "valid_area_pairs.parquet", index=False)
    print(f"\nWrote {ARTIFACTS_DIR / 'valid_area_pairs.parquet'} ({len(all_pairs)} rows)")

    # Most common pairs, fine level, entire dataset, both day-stages pooled -- useful to eyeball which pairs are richest.
    fine_all = all_pairs[(all_pairs["level"] == "fine") & (all_pairs["scope"] == "entire_dataset")]
    if len(fine_all):
        top_pairs = fine_all.groupby(["area_a", "area_b"]).agg(
            n_sessions=("session_id", "nunique"), n_mice=("mouse_id", "nunique"),
            n_Rplus_sessions=("reward_group", lambda s: (s == "R+").sum()),
            n_Rminus_sessions=("reward_group", lambda s: (s == "R-").sum()),
        ).sort_values("n_sessions", ascending=False)
        top_pairs.to_csv(ARTIFACTS_DIR / "top_fine_area_pairs.csv")
        print("\nTop 25 fine-level area-pairs by session count (entire-dataset scope, both day-stages):")
        print(top_pairs.head(25).to_string())


if __name__ == "__main__":
    main()
