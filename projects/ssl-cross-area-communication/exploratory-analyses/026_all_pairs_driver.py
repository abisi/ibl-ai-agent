"""Drives the full `024_updated_figures.py` pipeline across all 39 coarse
area pairs (`pairs_final_coarse_v3.csv`), per Axel's 2026-09-01 request
("run for the full dataset, all pairs"). Each pair gets its own run
subfolder (`artifacts/runs/<area_a>_vs_<area_b>/`) with the full 6-figure
menu (01/02/03/05/06/10 -- the figures currently built by 024; 07/lag-
sweep and 11/unit-count-confound live in 023 and are NOT run per-pair here
per the earlier "representative subset, not all 39" scoping default).

Session lists are derived directly from the cached
`full_unit_table_metadata_v3.parquet` (no NWB reload needed for this
step) using the SAME criterion as the existing
`motor_striatum_session_list.csv` -- verified to reproduce it exactly
(63/63 sessions, byte-for-byte match) before trusting this for the other
38 pairs: per pair, sessions with >=MIN_UNITS good+MUA units in BOTH
areas, mouse-filtered, day_stage=='learning'.

Each pair is wrapped in try/except so one pair's failure doesn't kill the
batch; a per-pair status/timing log is written to
`artifacts/runs/all_pairs_summary.csv` as the batch progresses (appended
after each pair, not just at the end, so progress survives an interrupted
run).
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
SUMMARY_CSV = RUNS_DIR / "all_pairs_summary.csv"


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return s


def derive_session_lists() -> tuple[pd.DataFrame, dict[tuple[str, str], list[str]]]:
    pairs = pd.read_csv(ARTIFACTS_DIR / "pairs_final_coarse_v3.csv")
    ut = pd.read_parquet(ARTIFACTS_DIR / "full_unit_table_metadata_v3.parquet")
    ut = ut[(ut["exclude"] == 0) & (ut["exclude_ephys"] == 0) & (ut["reward_group"].isin(["R+", "R-"]))
            & (ut["quality_label"].isin(["good", "mua"])) & (ut["day_stage"] == "learning")]
    counts = ut.groupby(["session_id", "area_group_coarse_v3"]).size().unstack(fill_value=0)

    session_lists: dict[tuple[str, str], list[str]] = {}
    for _, row in pairs.iterrows():
        a, b = row["area_a"], row["area_b"]
        qualifying = counts[(counts.get(a, 0) >= MIN_UNITS) & (counts.get(b, 0) >= MIN_UNITS)].index.tolist()
        session_lists[(a, b)] = sorted(qualifying)
    return pairs, session_lists


def main() -> None:
    import os
    pairs, session_lists = derive_session_lists()
    only_n = os.environ.get("SSL_PAIRS_LIMIT")
    pairs["_n_sessions"] = pairs.apply(lambda r: len(session_lists[(r["area_a"], r["area_b"])]), axis=1)
    pairs = pairs.sort_values("_n_sessions")  # cheapest pairs first -- surfaces bugs before the expensive ones run
    pair_rows = pairs.to_dict("records")
    if only_n:
        pair_rows = pair_rows[: int(only_n)]
        print(f"[driver] [LIMITED] running only the first {len(pair_rows)} pairs")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for i, row in enumerate(pair_rows):
        area_a, area_b = row["area_a"], row["area_b"]
        session_ids = session_lists[(area_a, area_b)]
        run_dir = RUNS_DIR / f"{slugify(area_a)}_vs_{slugify(area_b)}"
        print(f"\n[driver] ({i+1}/{len(pair_rows)}) {area_a} vs {area_b} -- {len(session_ids)} sessions -> {run_dir}")
        t0 = time.time()
        try:
            result = fig024.main(area_a=area_a, area_b=area_b, session_ids=session_ids, run_dir=run_dir)
            status = result.get("status", "unknown")
        except Exception as e:
            status = "error"
            print(f"[driver] FAILED {area_a} vs {area_b}: {e}")
            traceback.print_exc()
        elapsed = time.time() - t0
        summary_rows.append({"area_a": area_a, "area_b": area_b, "n_sessions": len(session_ids),
                              "status": status, "elapsed_s": round(elapsed, 1), "run_dir": str(run_dir)})
        pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
        print(f"[driver] {area_a} vs {area_b}: {status} in {elapsed:.0f}s")

    print(f"\n[driver] Done. Summary at {SUMMARY_CSV}")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
