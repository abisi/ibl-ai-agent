"""PERMANOVA (mouse-block permutation) testing whether selectivity_auc
differs by reward_group, separately per (day_stage, passive_epoch). With a
single index column this reduces to a mouse-block-permuted one-way
ANOVA-equivalent F-test -- same permanova_euclidean function, p=1.
Overall then per-area post-hoc with BH-FDR.
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
    sel = pd.read_parquet(ARTIFACTS_DIR / "selectivity_auc.parquet")
    sel = sel.dropna(subset=[f"selectivity_auc_{WINDOW}"])

    all_main = []
    for day_stage in ["learning", "expert"]:
        for epoch in ["passive_pre", "passive_post"]:
            sub = sel[(sel["day_stage"] == day_stage) & (sel["passive_epoch"] == epoch)]
            label = f"selectivity_{day_stage}_{epoch}"
            print(f"\n===== {label}: n_units={len(sub)}, n_mice={sub['mouse_id'].nunique()} =====")
            main_result, posthoc = run_with_posthoc(sub, [f"selectivity_auc_{WINDOW}"], area_col="area_group", label_prefix=label)
            all_main.append(main_result)
            posthoc.to_parquet(ARTIFACTS_DIR / f"permanova_selectivity_posthoc_{day_stage}_{epoch}.parquet", index=False)

    with open(ARTIFACTS_DIR / "permanova_selectivity_main.json", "w") as f:
        json.dump(all_main, f, indent=2, default=str)
    print("\nWrote permanova_selectivity_main.json")


if __name__ == "__main__":
    main()
