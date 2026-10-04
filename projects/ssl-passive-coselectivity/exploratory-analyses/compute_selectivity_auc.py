"""Selectivity AUC + permutation significance per unit, per passive_epoch,
per response window -- see question.md's "Selectivity AUC" definition. AUC
compares per-trial baseline-corrected whisker responses (label 1) against
per-trial baseline-corrected auditory responses (label 0), within the same
unit and epoch. AUC>0.5 = whisker-preferring, AUC<0.5 = auditory-preferring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from roc_utils import auc_and_perm_p  # noqa: E402

ARTIFACTS_DIR = Path("projects/ssl-passive-coselectivity/artifacts")
WINDOWS = ["w5_35"]
N_PERM = 1000
SEED = 20260815


def main() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "trial_rates.parquet")
    for w in WINDOWS:
        df[f"corrected_{w}"] = df[f"response_rate_hz_{w}"] - df["baseline_rate_hz"]
    rng = np.random.default_rng(SEED)

    group_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "day_stage", "passive_epoch"]
    grouped = df.groupby(group_cols, observed=True, sort=False)

    rows = []
    n_groups = 0
    for key, g in grouped:
        wh = g[g["trial_type"] == "whisker_trial"]
        au = g[g["trial_type"] == "auditory_trial"]
        if wh.empty or au.empty:
            continue
        n_groups += 1
        row = dict(zip(group_cols, key))
        row["n_whisker_trials"] = len(wh)
        row["n_auditory_trials"] = len(au)
        for w in WINDOWS:
            auc, p = auc_and_perm_p(wh[f"corrected_{w}"].to_numpy(), au[f"corrected_{w}"].to_numpy(), rng, n_perm=N_PERM)
            row[f"selectivity_auc_{w}"] = auc
            row[f"selectivity_p_{w}"] = p
            row[f"selective_{w}"] = bool(p < 0.05) if not np.isnan(p) else False
        rows.append(row)
        if n_groups % 50000 == 0:
            print(f"{n_groups} groups done", flush=True)

    result = pd.DataFrame(rows)
    out_path = ARTIFACTS_DIR / "selectivity_auc.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nWrote {len(result)} rows to {out_path}")
    for w in WINDOWS:
        print(f"\n{w}:")
        print(result.groupby("passive_epoch")[f"selectivity_auc_{w}"].describe())
        print(result.groupby("passive_epoch")[f"selective_{w}"].mean())


if __name__ == "__main__":
    main()
