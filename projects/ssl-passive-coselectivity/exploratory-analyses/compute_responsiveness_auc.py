"""Responsiveness AUC + permutation significance per unit, per modality
(trial_type), per passive_epoch, per response window -- see question.md's
"Responsiveness AUC" definition. AUC compares that unit's per-trial
response-window rate (label 1) against its per-trial baseline-window rate
(label 0), response window 5-35ms post-stim.
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
    rng = np.random.default_rng(SEED)

    group_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "day_stage", "trial_type", "passive_epoch"]
    grouped = df.groupby(group_cols, observed=True, sort=False)

    rows = []
    n_groups = 0
    for key, g in grouped:
        n_groups += 1
        base = g["baseline_rate_hz"].to_numpy()
        row = dict(zip(group_cols, key))
        row["n_trials"] = len(g)
        for w in WINDOWS:
            resp = g[f"response_rate_hz_{w}"].to_numpy()
            auc, p = auc_and_perm_p(resp, base, rng, n_perm=N_PERM)
            row[f"auc_{w}"] = auc
            row[f"p_{w}"] = p
            row[f"responsive_{w}"] = bool(p < 0.05) if not np.isnan(p) else False
        rows.append(row)
        if n_groups % 50000 == 0:
            print(f"{n_groups} groups done", flush=True)

    result = pd.DataFrame(rows)
    out_path = ARTIFACTS_DIR / "responsiveness_auc.parquet"
    result.to_parquet(out_path, index=False)
    print(f"\nWrote {len(result)} rows to {out_path}")
    for w in WINDOWS:
        print(f"\n{w}: responsive fraction overall = {result[f'responsive_{w}'].mean():.3f}")
        print(result.groupby(["trial_type", "passive_epoch"])[f"responsive_{w}"].mean())


if __name__ == "__main__":
    main()
