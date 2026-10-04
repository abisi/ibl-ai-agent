"""024c -- Sessions to recompute after the 2026-09-28 trial-prep rule (first whisker trial never removed; exactly
1 trial kept before it; skills/ssl-analyze/references/ssl_auditory_warmup_block.md). Per 024 tag: a session is
affected if its stored whole-session trial count differs from the count the current trial prep gives, or if any
of its rows was recomputed by the 2026-09-28 min-3 fill-in (interim trial prep). For affected sessions every
(session, area, condition_type) group present in the tag is listed, and 024 in fill-in mode replaces them.
Writes 024_fillin_firstwhisker_<tag>.parquet and prints the per-tag counts.
Run (haas): python 024c_first_whisker_fillin_targets.py tag [tag ...]
"""

import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT.parents[2] / "scripts"))
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import prep_hitmiss_trials, prep_lick_aligned_trials, prep_modality_trials  # noqa: E402

PREP = {"hitmiss": prep_hitmiss_trials, "modality_stim": prep_modality_trials, "modality_lick": prep_lick_aligned_trials}
root = resolve_dataset_dir("ssl_ephys")
st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
cache = {}
for tag in sys.argv[1:]:
    kind = "hitmiss" if tag.startswith("hitmiss") else "modality_lick" if tag.startswith("modality_lick") else "modality_stim"
    d = pd.read_parquet(OUT / f"024_master_results_{tag}.parquet",
                        columns=["session_id", "reward_group", "area_col", "area_value", "condition_type", "n_trials"])
    old = d[d.condition_type == "whole"].dropna(subset=["n_trials"]).groupby("session_id").n_trials.first()
    changed = set()
    for sid in d.session_id.unique():
        if (kind, sid) not in cache:
            tr = PREP[kind](root, sid, st, tt)
            cache[(kind, sid)] = None if tr is None else len(tr)
        new = cache[(kind, sid)]
        if sid in old.index and new is not None and int(old[sid]) != new:
            changed.add(sid)
    fill = OUT / f"024_fillin_targets_{tag}.parquet"
    min3 = set(pd.read_parquet(fill).session_id) if fill.exists() else set()
    affected = changed | min3
    t = d[d.session_id.isin(affected)][["session_id", "area_col", "area_value", "condition_type"]].drop_duplicates()
    t.to_parquet(OUT / f"024_fillin_firstwhisker_{tag}.parquet", index=False)
    rg = d.drop_duplicates("session_id").set_index("session_id").reward_group
    print(f"{tag}: {d.session_id.nunique()} sessions; trial set changed {len(changed)} "
          f"({pd.Series([rg[s] for s in changed]).value_counts().to_dict() if changed else {}}); min-3 fill-in {len(min3)}; "
          f"recompute {len(affected)} sessions / {len(t)} groups", flush=True)
