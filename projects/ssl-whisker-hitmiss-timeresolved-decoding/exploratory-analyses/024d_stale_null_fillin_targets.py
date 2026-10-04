"""024d -- Fill-in targets for rows whose stored linear-shift null is entirely NaN although the real curve exists
(user 2026-09-30). Cause: rows computed before 2026-09-28 used a fixed 5-fold CV for the null decoders; with exactly 5
trials of the minority class in a condition, every shift (dropping 10-50% of the trials) left <= 4 and all 25 nulls
failed, while the unshifted real decode (5 trials) succeeded. The current code uses n_folds = min(5, minority) and
computes these nulls. These groups were not in the 2026-09-28 recompute sets (024b/024c).
Per tag: every (session_id, area_col, area_value, condition_type) group containing such a row is listed; 024 in
fill-in mode (SSL_FILLIN_TARGETS) recomputes and replaces exactly those groups.
Writes 024_fillin_stalenull_<tag>.parquet and prints the counts.
Run (haas): python 024d_stale_null_fillin_targets.py tag [tag ...]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
for tag in sys.argv[1:]:
    d = pd.read_parquet(OUT / f"024_master_results_{tag}.parquet",
                        columns=["session_id", "area_col", "area_value", "condition_type", "skipped_reason",
                                 "shift_null_curves"])
    d = d[d.skipped_reason.isna() & d.shift_null_curves.notna()]
    bad = d.shift_null_curves.map(lambda L: all(np.isnan(np.asarray(x, float)).all() for x in L))
    t = d[bad][["session_id", "area_col", "area_value", "condition_type"]].drop_duplicates()
    t.to_parquet(OUT / f"024_fillin_stalenull_{tag}.parquet", index=False)
    print(f"{tag}: {int(bad.sum())} rows, {len(t)} groups, {t.session_id.nunique()} sessions "
          f"{t.groupby('condition_type').size().to_dict()}")
