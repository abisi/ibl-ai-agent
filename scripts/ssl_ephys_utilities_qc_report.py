"""SSL KS4 area-level QC report, built via the mandatory ephys_utilities
Path B pipeline (process_single_nwb/combine_ephys_nwb + classify_units_quality),
per skills/ssl-load/references/ssl_loading_policy.md.

For the 'learning' and 'expert' day arms separately, counts per
area_acronym_custom, split by reward_group (R+/R-): unique mice, unique
insertions, good units, good+MUA units (non-soma units excluded throughout).
The 'learning' arm is additionally split into all mice vs. the subset whose
learning_category (per joint_mouse_reference_weight.xlsx) is 'moderate' or 'good'.

Run with the `bwa` conda env (has pynwb + ephys_utilities + allen_utils all
importable together; verified 2026-08-26 -- `iblenv2`/`ephys_utils` envs are
missing pynwb):
    C:\\Users\\bisi\\AppData\\Local\\anaconda3\\envs\\bwa\\python.exe scripts\\ssl_ephys_utilities_qc_report.py

Insertion key: this pipeline's unit_table has no separate probe_name column
(unlike ibl_ai_agent's compressed-dataset builder) -- only `electrode_group`
(e.g. 'imec0' or 'imec0_shank0', from process_single_nwb's
convert_electrode_group_object_to_columns). (session_id, electrode_group) is
used as the "unique insertion" key here.

Area labels use allen_utils.process_allen_labels(..., split_merge_areas=True)
-- the real parameter is split_merge_areas, not subdivide_areas (verified
against source) -- since split_merge_areas=True is the canonical setting for
area_acronym_custom used for quantification in this project (per Axel Bisi,
2026-08-26): it further applies create_areas_subdivisions/create_area_groupings/
create_thalamic_groupings on top of the base custom-acronym mapping, which
*overwrites* area_acronym_custom in place (e.g. individual thalamic nuclei
collapse into functional groups like "ATN"), not just add extra columns.

reward_group source: the native `reward_group` column process_single_nwb
attaches to unit_table directly from session metadata's `wh_reward` field
(int 0/1 -- verified against source: NWB_reader_functions.get_session_metadata),
mapped here to 'R-'/'R+'. This is NOT the same source as the reference
sheet's per-mouse `reward_group` (R+/R-/R+proba levels) used for
learning_category filtering below -- see ssl_loading_policy.md's
"Cohort/reward_group source ambiguity" section. The two can disagree,
in particular R+proba mice: wh_reward only encodes a binary, so an R+proba
mouse's sessions appear as plain 'R+' here, not distinguished as probabilistic.
"""
from __future__ import annotations

import sys
import json
import warnings
from pathlib import Path

sys.path.insert(0, r"M:\analysis\Axel_Bisi\NWB_reader")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\ephys_utilities")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ephys_utilities.helpers import data_utils
from ephys_utilities.neural_utils import unit_metrics_utils
import allen_utils

NWB_ROOT = Path(r"M:\analysis\Axel_Bisi\NWB_ks4")
REF_XLSX = Path(r"M:\share_internal\Axel_Bisi_Share\dataset_info\joint_mouse_reference_weight.xlsx")
OUT_DIR = Path(r"C:\Users\bisi\Github\int-brain-lab\ibl-ai-agent\reports\ssl_analysis\ephys_utilities_qc_report")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 12
LABEL_COL = "quality_label"
THRESHOLDS = unit_metrics_utils.DEFAULT_METRIC_THRESHOLDS
EXCLUDE = ["Lratio", "isolationDistance", "presenceRatio", "maxDriftEstimate"]
SPLIT_MERGE_AREAS = True
REWARD_LABELS = {1: "R+", 0: "R-"}
REWARD_ORDER = ["R+", "R-"]


def list_nwb_files() -> list[str]:
    return [str(f) for f in sorted(NWB_ROOT.glob("*.nwb"))]


def build_and_classify(day_to_analyze: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    nwb_list = list_nwb_files()
    trial_table, unit_table, ephys_nwb_list = data_utils.combine_ephys_nwb(
        nwb_list, day_to_analyze=day_to_analyze, max_workers=MAX_WORKERS
    )
    meta = {
        "day_to_analyze": day_to_analyze,
        "n_nwb_files_scanned": len(nwb_list),
        "n_nwb_files_with_ephys_after_day_filter": len(ephys_nwb_list),
    }
    if len(unit_table) == 0:
        return trial_table, unit_table, meta

    if "presence_ratio" not in unit_table.columns and "spike_times" in unit_table.columns:
        unit_table = unit_metrics_utils.compute_presence_coverage_metrics(unit_table)
        meta["presence_coverage_metrics_computed"] = True
    else:
        meta["presence_coverage_metrics_computed"] = False

    joint_drift_cols_present = all(c in unit_table.columns for c in ("drift_abs_r", "drift_shift_test_pval"))
    meta["drift_dredge_merge_performed"] = False
    meta["joint_drift_criterion_evaluated"] = joint_drift_cols_present

    unit_table = unit_metrics_utils.classify_units_quality(
        unit_table, thresholds=THRESHOLDS, exclude=EXCLUDE, label_col=LABEL_COL
    )
    meta["thresholds_used"] = "DEFAULT_METRIC_THRESHOLDS"
    meta["thresholds_literal"] = {k: list(v) for k, v in THRESHOLDS.items()}
    meta["exclude_used"] = EXCLUDE
    meta["label_col"] = LABEL_COL
    meta["quality_label_counts"] = unit_table[LABEL_COL].value_counts(dropna=False).to_dict()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        labeled = allen_utils.process_allen_labels(unit_table, split_merge_areas=SPLIT_MERGE_AREAS)
        for w in caught:
            print(f"  allen_utils warning: {w.message}")

    meta["split_merge_areas"] = SPLIT_MERGE_AREAS
    meta["n_units_before_area_labeling"] = len(unit_table)
    meta["n_units_after_area_labeling"] = len(labeled)

    labeled["_reward_label"] = labeled["reward_group"].map(REWARD_LABELS)
    unexpected = sorted(labeled.loc[labeled["_reward_label"].isna(), "reward_group"].unique().tolist())
    meta["reward_group_source"] = "native unit_table['reward_group'] (session metadata wh_reward, int 0/1)"
    meta["reward_group_unexpected_values"] = [str(v) for v in unexpected]
    if unexpected:
        print(f"  Warning: unexpected reward_group values (not 0/1): {unexpected} -- dropped from reward-group split")

    return trial_table, labeled, meta


def summarize_by_area_reward(unit_table: pd.DataFrame) -> pd.DataFrame:
    """Long-form: one row per (area_acronym_custom, reward_group)."""
    df = unit_table[unit_table[LABEL_COL] != "non-soma"].copy()
    df = df[df["_reward_label"].isin(REWARD_ORDER)]
    df["_insertion_key"] = df["session_id"].astype(str) + "::" + df["electrode_group"].astype(str)

    rows = []
    for (area, rg), g in df.groupby(["area_acronym_custom", "_reward_label"], dropna=False):
        rows.append(
            {
                "area_acronym_custom": area,
                "reward_group": rg,
                "n_mice": g["mouse_id"].nunique(),
                "n_insertions": g["_insertion_key"].nunique(),
                "n_good": int((g[LABEL_COL] == "good").sum()),
                "n_good_mua": int(g[LABEL_COL].isin(["good", "mua"]).sum()),
            }
        )
    long_df = pd.DataFrame(rows)
    if len(long_df) == 0:
        return long_df, []

    area_order = (
        long_df.groupby("area_acronym_custom")["n_good_mua"].sum().sort_values(ascending=False).index.tolist()
    )
    return long_df, area_order


def to_wide(long_df: pd.DataFrame, area_order: list[str]) -> pd.DataFrame:
    """area_acronym_custom, n_mice_R+, n_mice_R-, n_insertions_R+, ... (one row per area)."""
    if len(long_df) == 0:
        return pd.DataFrame(columns=["area_acronym_custom"])
    metrics = ["n_mice", "n_insertions", "n_good", "n_good_mua"]
    wide = long_df.pivot_table(index="area_acronym_custom", columns="reward_group", values=metrics, fill_value=0)
    cols = []
    for m in metrics:
        for rg in REWARD_ORDER:
            cols.append((m, rg))
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(cols))
    wide.columns = [f"{m}_{rg.replace('+', 'plus').replace('-', 'minus')}" for m, rg in wide.columns]
    wide = wide.reindex(area_order).fillna(0).astype(int)
    wide = wide.reset_index()
    return wide


def plot_four_panel_grouped(long_df: pd.DataFrame, area_order: list[str], out_path: Path, suptitle: str) -> None:
    if len(area_order) == 0:
        return
    metrics = ["n_mice", "n_insertions", "n_good", "n_good_mua"]
    titles = ["# Mice", "# Insertions", "# Good units", "# Good+MUA units"]
    colors = {"R+": "#4C72B0", "R-": "#DD8452"}
    n_areas = len(area_order)
    y_base = np.arange(n_areas)
    bar_h = 0.38
    fig, axes = plt.subplots(1, 4, figsize=(19, max(6, 0.32 * n_areas)), sharey=True)
    for ax, metric, title in zip(axes, metrics, titles):
        for rg in REWARD_ORDER:
            sub = long_df[long_df["reward_group"] == rg].set_index("area_acronym_custom").reindex(area_order)
            offset = bar_h / 2 if rg == "R+" else -bar_h / 2
            ax.barh(y_base + offset, sub[metric].fillna(0), height=bar_h, color=colors[rg], label=rg)
        ax.set_title(title)
        ax.set_xlabel("count")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y_base)
    axes[0].set_yticklabels(area_order)
    axes[0].legend(loc="lower right")
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_good_vs_goodmua_by_reward(long_df: pd.DataFrame, area_order: list[str], out_path: Path, suptitle: str) -> None:
    if len(area_order) == 0:
        return
    n_areas = len(area_order)
    y_pos = np.arange(n_areas)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(6, 0.32 * n_areas)), sharey=True)
    for ax, rg in zip(axes, REWARD_ORDER):
        sub = long_df[long_df["reward_group"] == rg].set_index("area_acronym_custom").reindex(area_order).fillna(0)
        ax.barh(y_pos, sub["n_good_mua"], color="#C44E52", label="Good+MUA")
        ax.barh(y_pos, sub["n_good"], color="#4C72B0", label="Good")
        ax.set_title(rg)
        ax.set_xlabel("# units")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(area_order)
    axes[0].legend()
    fig.suptitle(suptitle)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def process_and_save(unit_table: pd.DataFrame, prefix: str, suptitle_suffix: str, all_meta_entry: dict) -> None:
    long_df, area_order = summarize_by_area_reward(unit_table)
    wide_df = to_wide(long_df, area_order)
    wide_df.to_csv(OUT_DIR / f"{prefix}_area_summary.csv", index=False)
    plot_four_panel_grouped(
        long_df, area_order, OUT_DIR / f"{prefix}_panels.png",
        f"{suptitle_suffix} (non-soma excluded, split by reward_group)",
    )
    plot_good_vs_goodmua_by_reward(
        long_df, area_order, OUT_DIR / f"{prefix}_good_vs_goodmua.png",
        f"Good vs Good+MUA per area -- {suptitle_suffix}",
    )
    all_meta_entry["n_areas"] = len(area_order)


def run_arm(day_to_analyze: str, arm_label: str, ref_df: pd.DataFrame, all_meta: dict) -> None:
    print(f"=== Building {arm_label} arm (day_to_analyze={day_to_analyze!r}) ===")
    trial_table, unit_table, meta = build_and_classify(day_to_analyze)
    all_meta[arm_label] = meta
    if len(unit_table) == 0:
        print(f"  No units for {arm_label}; skipping.")
        return

    all_meta[arm_label]["n_mice_all"] = unit_table["mouse_id"].nunique()
    all_meta[arm_label]["n_sessions_all"] = unit_table["session_id"].nunique()
    per_mouse_reward = unit_table.groupby("mouse_id")["_reward_label"].nunique()
    inconsistent_mice = sorted(per_mouse_reward[per_mouse_reward > 1].index.tolist())
    if inconsistent_mice:
        print(f"  Warning: mice with >1 distinct reward_group across sessions in {arm_label}: {inconsistent_mice}")
    all_meta[arm_label]["mice_with_inconsistent_reward_group"] = inconsistent_mice

    dedup_mice = unit_table.drop_duplicates("mouse_id")
    all_meta[arm_label]["n_mice_per_reward_group"] = dedup_mice.groupby("_reward_label")["mouse_id"].nunique().to_dict()

    all_mice_meta: dict = {}
    process_and_save(unit_table, f"{arm_label}_all_mice", f"{arm_label} -- all mice", all_mice_meta)
    all_meta[arm_label]["all_mice"] = all_mice_meta

    if arm_label == "learning":
        merged = unit_table.merge(
            ref_df[["mouse_id", "learning_category"]], on="mouse_id", how="left"
        )
        n_unmatched = merged["learning_category"].isna().sum()
        if n_unmatched:
            unmatched_mice = sorted(merged.loc[merged["learning_category"].isna(), "mouse_id"].unique())
            print(f"  Warning: {n_unmatched} units from mice with no learning_category match in reference sheet: {unmatched_mice}")
        subset = merged[merged["learning_category"].isin(["moderate", "good"])]
        all_meta[arm_label]["n_mice_moderate_good_subset"] = subset["mouse_id"].nunique()
        all_meta[arm_label]["mice_excluded_from_subset"] = sorted(
            set(unit_table["mouse_id"].unique()) - set(subset["mouse_id"].unique())
        )

        subset_meta: dict = {}
        process_and_save(
            subset, f"{arm_label}_moderate_good_mice",
            f"{arm_label} -- learning_category in {{moderate, good}}", subset_meta,
        )
        all_meta[arm_label]["moderate_good_mice"] = subset_meta


def main() -> None:
    ref_df = pd.read_excel(REF_XLSX, sheet_name="Sheet1")
    all_meta: dict = {
        "ephys_utilities_root": r"M:\analysis\Axel_Bisi\Github\ephys_utilities",
        "nwb_root": str(NWB_ROOT),
        "reference_xlsx": str(REF_XLSX),
        "insertion_key": "(session_id, electrode_group)",
        "non_soma_exclusion": "quality_label == 'non-soma' dropped before all counts",
        "split_merge_areas": SPLIT_MERGE_AREAS,
        "reward_group_labels": REWARD_LABELS,
    }
    run_arm("learning", "learning", ref_df, all_meta)
    run_arm("expert", "expert", ref_df, all_meta)

    with open(OUT_DIR / "run_metadata.json", "w") as f:
        json.dump(all_meta, f, indent=2, default=str)
    print("Wrote run_metadata.json")


if __name__ == "__main__":
    main()
