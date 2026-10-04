"""Bi-responsive fraction: units responsive to both whisker and auditory in
the same epoch (see question.md). Computed per mouse first, then reported
by epoch x cohort x area x day_stage x window, per the mouse-level-
aggregation lesson from ssl-passive-sensory-selectivity (pooling units
directly, even with a random-intercept term, previously gave a spuriously
"highly significant" cohort effect that dropped to borderline once
aggregated to one row per mouse first).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path("projects/ssl-passive-coselectivity/artifacts")
WINDOWS = ["w5_35"]


def main() -> None:
    auc = pd.read_parquet(ARTIFACTS_DIR / "responsiveness_auc.parquet")

    idx_cols = ["session_id", "cluster_id", "mouse_id", "area_group", "reward_group", "day_stage", "passive_epoch"]
    for w in WINDOWS:
        wide = auc.pivot_table(index=idx_cols, columns="trial_type", values=f"responsive_{w}", aggfunc="first").reset_index()
        wide = wide.rename(columns={"whisker_trial": "wh_resp", "auditory_trial": "au_resp"}).dropna(subset=["wh_resp", "au_resp"])
        wide[f"bi_responsive_{w}"] = wide["wh_resp"] & wide["au_resp"]
        wide[f"wh_only_{w}"] = wide["wh_resp"] & ~wide["au_resp"]
        wide[f"au_only_{w}"] = ~wide["wh_resp"] & wide["au_resp"]
        wide[f"neither_{w}"] = ~wide["wh_resp"] & ~wide["au_resp"]
        wide = wide.drop(columns=["wh_resp", "au_resp"])
        out_name = f"bi_responsive_{w}.parquet"
        wide.to_parquet(ARTIFACTS_DIR / out_name, index=False)
        print(f"Wrote {len(wide)} unit-level rows to {out_name}")

        # per-mouse fraction, then per (day_stage, epoch, cohort) summary
        per_mouse = wide.groupby(["day_stage", "passive_epoch", "mouse_id", "reward_group"])[f"bi_responsive_{w}"].mean().reset_index()
        per_mouse.to_parquet(ARTIFACTS_DIR / f"bi_responsive_per_mouse_{w}.parquet", index=False)
        print(f"\n[{w}] Per-mouse bi-responsive fraction, by day_stage x epoch x cohort:")
        print(per_mouse.groupby(["day_stage", "passive_epoch", "reward_group"])[f"bi_responsive_{w}"].agg(["mean", "std", "count"]))

        # per-mouse, per area
        per_mouse_area = wide.groupby(["day_stage", "passive_epoch", "area_group", "mouse_id", "reward_group"])[f"bi_responsive_{w}"].mean().reset_index()
        per_mouse_area.to_parquet(ARTIFACTS_DIR / f"bi_responsive_per_mouse_area_{w}.parquet", index=False)


if __name__ == "__main__":
    main()
