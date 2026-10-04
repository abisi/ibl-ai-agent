"""Drives the full `024_updated_figures.py` pipeline across all FINE-level
area pairs (`area_acronym_custom`, 69 fine areas) meeting >=5 sessions in
EACH cohort, per Axel's 2026-09-01 "do the same but for the fine area
level. Only do it for area pairs with at least 5 recordings of each."

Unlike the coarse-level driver (`026`), there is no pre-existing fine-pair
CSV to start from -- pairs are derived directly from the >=5-sessions/
cohort criterion (superseding the earlier, separate >=3-mice/cohort
criterion used for `pairs_final_fine.csv`), same as the coarse level's
`031_windowed_summary.py` re-filter. 69 fine areas x combinatorics, at
>=20 good+MUA units/area and >=5 sessions/cohort, gives 82 qualifying
pairs (1291 total session-instances -- ~35% more than the coarse batch's
956, so expect a proportionally longer run, roughly 5-6h based on the
coarse batch's ~4.45h).

Each pair gets its own run subfolder
(`artifacts/runs/fine_<area_a>_vs_<area_b>/`) with the full 01/02/03/05/
06/10 figure set; results also land in the same cache
(`artifacts/cache/main_pass_*.pkl`) that `031`-style windowed summary
scripts read from, keyed distinctly from coarse-level results via
`AREA_COLUMN` now being part of the pair fingerprint.
"""
from __future__ import annotations

import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
fig024 = importlib.import_module("024_updated_figures")

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
MIN_UNITS = fig024.MIN_UNITS
MIN_SESSIONS_PER_COHORT = 5
AREA_COLUMN = "area_acronym_custom"
SUMMARY_CSV = RUNS_DIR / "all_fine_pairs_summary.csv"


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def derive_fine_pairs() -> tuple[list[tuple[str, str]], dict[tuple[str, str], list[str]]]:
    ut = pd.read_parquet(ARTIFACTS_DIR / "full_unit_table_metadata_v3.parquet")
    ut = ut[(ut["exclude"] == 0) & (ut["exclude_ephys"] == 0) & (ut["reward_group"].isin(["R+", "R-"]))
            & (ut["quality_label"].isin(["good", "mua"])) & (ut["day_stage"] == "learning")]
    counts = ut.groupby(["session_id", AREA_COLUMN]).size().unstack(fill_value=0)
    rg = ut.drop_duplicates("session_id").set_index("session_id")["reward_group"]
    areas = sorted(ut[AREA_COLUMN].unique())

    pairs, session_lists = [], {}
    for i, a in enumerate(areas):
        for b in areas[i + 1:]:
            qualifying = counts[(counts.get(a, 0) >= MIN_UNITS) & (counts.get(b, 0) >= MIN_UNITS)].index
            n_p = (rg.reindex(qualifying) == "R+").sum()
            n_m = (rg.reindex(qualifying) == "R-").sum()
            if n_p >= MIN_SESSIONS_PER_COHORT and n_m >= MIN_SESSIONS_PER_COHORT:
                sess = sorted(qualifying.tolist())
                pairs.append((a, b))
                session_lists[(a, b)] = sess
    return pairs, session_lists


def main() -> None:
    import os
    pairs, session_lists = derive_fine_pairs()
    only_n = os.environ.get("SSL_PAIRS_LIMIT")
    pairs = sorted(pairs, key=lambda p: len(session_lists[p]))  # cheapest first
    if only_n:
        pairs = pairs[: int(only_n)]
        print(f"[fine driver] [LIMITED] running only the first {len(pairs)} pairs")

    print(f"[fine driver] {len(pairs)} fine-area pairs to run "
          f"({sum(len(session_lists[p]) for p in pairs)} total session-instances)")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for i, (area_a, area_b) in enumerate(pairs):
        session_ids = session_lists[(area_a, area_b)]
        run_dir = RUNS_DIR / f"fine_{slugify(area_a)}_vs_{slugify(area_b)}"
        print(f"\n[fine driver] ({i+1}/{len(pairs)}) {area_a} vs {area_b} -- {len(session_ids)} sessions -> {run_dir}")
        t0 = time.time()
        try:
            result = fig024.main(area_a=area_a, area_b=area_b, session_ids=session_ids, run_dir=run_dir,
                                  area_column=AREA_COLUMN)
            status = result.get("status", "unknown")
        except Exception as e:
            status = "error"
            print(f"[fine driver] FAILED {area_a} vs {area_b}: {e}")
            traceback.print_exc()
        elapsed = time.time() - t0
        summary_rows.append({"area_a": area_a, "area_b": area_b, "n_sessions": len(session_ids),
                              "status": status, "elapsed_s": round(elapsed, 1), "run_dir": str(run_dir)})
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
        print(f"[fine driver] {area_a} vs {area_b}: {status} in {elapsed:.0f}s")

    print(f"\n[fine driver] Done. Summary at {SUMMARY_CSV}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
