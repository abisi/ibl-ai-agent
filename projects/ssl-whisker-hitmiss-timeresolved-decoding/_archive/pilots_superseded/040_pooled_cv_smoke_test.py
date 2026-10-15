"""Quick smoke test for `039_pooled_cv_pilot_wholebrain_whole.py` (user
request 2026-09-14): 2 sessions (1 R+, 1 R-), both hitmiss and perfstate,
plotted with the exact same comparison figure the full 89-session sweep
will use -- a fast visual preview while the full pilot is still queued.

Usage: python 040_pooled_cv_smoke_test.py
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
pilot = importlib.import_module("039_pooled_cv_pilot_wholebrain_whole")

from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import causal_bin_edges, hitmiss_session_list  # noqa: E402

OUT_DIR = pilot.OUT_DIR
SESSIONS = [
    ("AB080_20230622_152205", "AB080", "R+"),
    ("AB085_20231005_152636", "AB085", "R-"),
]


def main():
    scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
    bin_edges = causal_bin_edges(pilot.STIM_WINDOW, bin_width=pilot.BIN_WIDTH, stride=pilot.STRIDE)

    for decode_target in ("hitmiss", "perfstate"):
        rows = []
        for session_id, subject_id, reward_group in SESSIONS:
            print(f"[{decode_target}] running {session_id} ({reward_group})...", flush=True)
            row = pilot.process_one_session((session_id, subject_id, reward_group, scripts_dir, decode_target))
            if row is not None:
                rows.append(row)
        df = pd.DataFrame(rows)
        partial_path = OUT_DIR / f"039_pooled_cv_pilot_{decode_target}_smoke_stim_whole_brain.parquet"
        edges_path = OUT_DIR / f"039_bin_edges_{decode_target}_smoke.json"
        df.to_parquet(partial_path, index=False)
        edges_path.write_text(json.dumps(bin_edges))
        pilot.plot_comparison(decode_target, suffix="_smoke", partial_path=partial_path, edges_path=edges_path)


if __name__ == "__main__":
    main()
