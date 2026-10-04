"""PERMANOVA (mouse-block permutation) testing whether the joint
[whisker_AUC, auditory_AUC] responsiveness profile differs by reward_group,
separately per (day_stage, passive_epoch). Overall (all areas pooled) then
per-area post-hoc with BH-FDR -- see permanova.py and question.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from permanova import run_with_posthoc  # noqa: E402

ARTIFACTS_DIR = Path("projects/ssl-passive-coselectivity/artifacts")
WINDOW = "w5_35"


def main() -> None:
    auc = pd.read_parquet(ARTIFACTS_DIR / "responsiveness_auc.parquet")
    idx_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "day_stage", "passive_epoch"]
    wide = auc.pivot_table(index=idx_cols, columns="trial_type", values=f"auc_{WINDOW}", aggfunc="first").reset_index()
    wide = wide.rename(columns={"whisker_trial": "whisker_auc", "auditory_trial": "auditory_auc"})
    wide = wide.dropna(subset=["whisker_auc", "auditory_auc"])
    wide.to_parquet(ARTIFACTS_DIR / f"responsiveness_auc_wide_{WINDOW}.parquet", index=False)

    all_main = []
    for day_stage in ["learning", "expert"]:
        for epoch in ["passive_pre", "passive_post"]:
            sub = wide[(wide["day_stage"] == day_stage) & (wide["passive_epoch"] == epoch)]
            label = f"responsiveness_{day_stage}_{epoch}"
            print(f"\n===== {label}: n_units={len(sub)}, n_mice={sub['mouse_id'].nunique()} =====")
            main_result, posthoc = run_with_posthoc(sub, ["whisker_auc", "auditory_auc"], area_col="area_group", label_prefix=label)
            all_main.append(main_result)
            posthoc.to_parquet(ARTIFACTS_DIR / f"permanova_responsiveness_posthoc_{day_stage}_{epoch}.parquet", index=False)

    with open(ARTIFACTS_DIR / "permanova_responsiveness_main.json", "w") as f:
        json.dump(all_main, f, indent=2, default=str)
    print("\nWrote permanova_responsiveness_main.json")


if __name__ == "__main__":
    main()
