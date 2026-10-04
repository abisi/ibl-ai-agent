"""Extract reward-group (cohort) and training day for every ssl_ks2_ephys
ephys session, by re-opening each source NWB file's experiment_description
(not currently stored in the compressed dataset).

Must be run with M:\\analysis\\Axel_Bisi\\unit_spikes_analysis's own venv python.

- reward_group ("cohort") = experiment_description['wh_reward'], parsed via
  ast.literal_eval since the NWB field is a stringified Python dict repr, not
  real JSON.
- day is parsed directly from the already-stored session_description
  ('whisker_0', 'whisker_+1', ...) -- no NWB re-read needed for this part,
  kept here anyway for a single combined output.
"""
import ast
import re
import sys

sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_combined")  # not a package, just documenting the source root
import pandas as pd

NWB_ROOT = r"M:\analysis\Axel_Bisi\NWB_combined"
DATASET_DIR = r"C:\Users\bisi\Github\int-brain-lab\ibl-ai-agent\reports\datasets\ssl_ks2_ephys\1.0.0"
OUTPUT_PATH = r"C:\Users\bisi\Github\int-brain-lab\ibl-ai-agent\projects\ssl-ks2-dataset-description\artifacts\session_metadata_extra.parquet"


def parse_day(session_description: str) -> int | None:
    m = re.match(r"^whisker_([+-]?\d+)$", session_description)
    return int(m.group(1)) if m else None


def main():
    from pynwb import NWBHDF5IO
    import warnings
    warnings.filterwarnings("ignore")
    import os

    sessions = pd.read_parquet(os.path.join(DATASET_DIR, "metadata", "sessions.parquet"))
    eph = sessions[sessions["has_ephys"]].copy()
    print(f"{len(eph)} ephys sessions to process")

    rows = []
    for i, row in enumerate(eph.itertuples(), start=1):
        path = os.path.join(NWB_ROOT, row.source_filename)
        try:
            with NWBHDF5IO(path, "r", load_namespaces=True) as io:
                nwb = io.read()
                desc_raw = nwb.experiment_description
                desc = ast.literal_eval(desc_raw) if isinstance(desc_raw, str) else (desc_raw or {})
                reward_group = desc.get("wh_reward")
        except Exception as exc:
            print(f"  [{i}/{len(eph)}] FAILED {row.source_filename}: {exc}")
            reward_group = None
        day = parse_day(row.session_description)
        rows.append({
            "session_id": row.session_id,
            "subject_id": row.subject_id,
            "session_description": row.session_description,
            "day": day,
            "day_stage": ("learning" if day == 0 else "expert") if day is not None else None,
            "reward_group": reward_group,
        })
        if i % 20 == 0:
            print(f"  [{i}/{len(eph)}]")

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"saved {OUTPUT_PATH} ({len(out)} rows)")
    print(out["day_stage"].value_counts(dropna=False))
    print(out["reward_group"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
