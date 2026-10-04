"""Per-area distribution comparison (R+ vs R-) for the 6 combinations where
the pooled PERMANOVA was significant -- descriptive follow-up showing the
actual value distributions in the areas the post-hoc ranked most suggestive,
even though none survived BH-FDR (see 004_statistics.py's summary). Uses
already-computed full-population parquets and post-hoc results -- no rerun.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
N_TOP_AREAS = 6

SIGNIFICANT_LABELS = [
    "learning_all_mice_good_burst_index_auditory",
    "learning_learners_only_good_burst_index_auditory",
    "expert_all_mice_good_burst_index_whisker",
    "expert_all_mice_good_mua_burst_index_whisker",
    "expert_learners_only_good_burst_index_whisker",
    "expert_learners_only_good_mua_burst_index_whisker",
]


def parse_label(label: str) -> dict:
    m = re.match(r"(learning|expert)_(all_mice|learners_only)_(good_mua|good)_(.+)", label)
    day_stage, scope, tier, metric = m.groups()
    return {"day_stage": day_stage, "scope": scope, "tier": tier, "metric": metric}


def main() -> None:
    full_cache = {}
    for label in SIGNIFICANT_LABELS:
        info = parse_label(label)
        posthoc = pd.read_parquet(ARTIFACTS_DIR / f"posthoc_{label}.parquet")
        top_areas = posthoc.nsmallest(N_TOP_AREAS, "p_value")["area_acronym_custom"].tolist()

        if info["day_stage"] not in full_cache:
            full_cache[info["day_stage"]] = pd.read_parquet(ARTIFACTS_DIR / f"full_{info['day_stage']}_unit_metrics.parquet")
        df = full_cache[info["day_stage"]]

        scope_filter = df["reward_group"].isin(["R+", "R-"]) if info["scope"] == "all_mice" else df["learning_category"].isin(["good", "moderate"])
        tier_filter = df["quality_label"] == "good" if info["tier"] == "good" else df["quality_label"].isin(["good", "mua"])
        metric = info["metric"]
        sub = df[scope_filter & tier_filter & df[f"{metric}_valid"] & df[metric].notna()]
        sub = sub[sub["area_acronym_custom"].isin(top_areas)]

        fig, ax = plt.subplots(figsize=(1.4 * len(top_areas) + 2, 5))
        positions, labels_x = [], []
        for i, area in enumerate(top_areas):
            area_p = posthoc.loc[posthoc.area_acronym_custom == area, "p_value"].iloc[0]
            for j, (rg, color, offset) in enumerate([("R+", "#4C72B0", -0.18), ("R-", "#DD8452", 0.18)]):
                vals = sub.loc[(sub.area_acronym_custom == area) & (sub.reward_group == rg), metric]
                if len(vals) == 0:
                    continue
                parts = ax.violinplot(vals, positions=[i + offset], widths=0.32, showmedians=True)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)
                for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                    parts[key].set_color(color)
            labels_x.append(f"{area}\n(p={area_p:.3f})")
        ax.set_xticks(range(len(top_areas)))
        ax.set_xticklabels(labels_x, fontsize=8)
        ax.set_ylabel(metric)
        ax.axhline(0, color="0.7", linewidth=0.6) if "burst_index" in metric else None
        import matplotlib.patches as mpatches
        ax.legend(handles=[mpatches.Patch(color="#4C72B0", label="R+"), mpatches.Patch(color="#DD8452", label="R-")], fontsize=8)
        ax.set_title(f"{label}\n(top {len(top_areas)} areas by post-hoc p-value; main PERMANOVA p<0.05, none survive BH-FDR)", fontsize=9)
        fig.tight_layout()
        out_path = ARTIFACTS_DIR / f"area_distributions_{label}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
