"""024b -- Fill-in targets for MIN_TRIALS_PER_CLASS 5 -> 3 (user request 2026-09-28): per 024 tag, every
(session, area, condition_type) group with at least one row skipped as "class counts hit=X/miss=Y (< 5 each)" with
min(X, Y) >= 3. The whole group is recomputed (both halves + cross-generalisation), via
`SSL_FILLIN_TARGETS=024_fillin_targets_<tag>.parquet python 024_master_sweep.py ...`.
Run: python 024b_min3_fillin_targets.py tag [tag ...]
"""

import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
for tag in sys.argv[1:]:
    d = pd.read_parquet(OUT / f"024_master_results_{tag}.parquet",
                        columns=["session_id", "reward_group", "area_col", "area_value", "condition_type", "condition_value", "skipped_reason"])
    m = d.skipped_reason.fillna("").str.extract(r"class counts hit=(\d+)/miss=(\d+)").astype(float)
    ok = m.min(axis=1) >= 3
    t = d[ok][["session_id", "area_col", "area_value", "condition_type"]].drop_duplicates()
    t.to_parquet(OUT / f"024_fillin_targets_{tag}.parquet", index=False)
    rows = d[ok]
    print(f"{tag}: {len(t)} groups in {t.session_id.nunique()} sessions; rescued rows by condition/cohort:",
          rows.groupby(["condition_type", "reward_group"]).size().to_dict(), flush=True)
