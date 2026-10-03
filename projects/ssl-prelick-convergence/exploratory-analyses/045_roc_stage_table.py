"""Master table for the multi-ROC learning-stage analysis (rate-based ROC results, 2026-10-01).

Inputs
  combined_results_ks4/<mouse>/whisker_<day>/roc_analysis/<mouse>_roc_results_new.csv   (124 sessions; rates)
  rastermap_variants/_cache/tables_all_days.pkl   (unit_table of all days: quality_label with the corrected drift
                                                    rule, area_acronym_custom, reward_group, day)
  joint_mouse_reference_weight.xlsx               (learning_category)
Conventions
  stage   : learning = day 0, expert = day >= 1
  cohort  : R+ / R- (reward_group)
  units   : quality_label == 'good' (all labels kept in the table; analyses filter)
  join    : mouse_id, session_id, electrode_group, cluster_id (str)
  selectivity sign unified to "positive = second-named condition higher" with interpretable labels; the choice
  types (spikes_1 = lick, spikes_2 = no-lick) are FLIPPED so that positive = lick / hit > no-lick / miss.
  Passive analyses are absent (NaN) for sessions without passive blocks -> excluded from those denominators.
Outputs (combined_results_ks4/_roc_stage_analysis/)
  units.parquet     one row per unit (keys, stage, cohort, day, area_acronym_custom, area_group, ccf_atlas_*, labels)
  roc_long.parquet  one row per unit x analysis_type (sel, abs_sel, sig, pos, neg)
  provenance.json
"""
import glob
import json
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
CACHE = RES / "rastermap_variants" / "_cache" / "tables_all_days.pkl"
MOUSE_INFO = "/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx"
OUT = RES / "_roc_stage_analysis"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
FLIP = {"choice", "whisker_choice", "auditory_choice", "baseline_choice", "baseline_whisker_choice",
        "baseline_auditory_choice"}
# meaning of positive selectivity after unification (for figure labels)
POS_MEANING = {
    "whisker_passive_pre": "excited", "whisker_passive_post": "excited", "whisker_active": "excited",
    "auditory_passive_pre": "excited", "auditory_passive_post": "excited", "auditory_active": "excited",
    "whisker_pre_vs_post_learning": "post > pre session", "auditory_pre_vs_post_learning": "post > pre session",
    "wh_vs_aud_pre_vs_post_learning": "auditory(post) > whisker(pre)",
    "wh_vs_aud_passive_pre": "auditory > whisker", "wh_vs_aud_passive_post": "auditory > whisker",
    "wh_vs_aud_active": "auditory > whisker", "spontaneous_licks": "excited",
    "spontaneous_licks_vs_cr": "lick > CR", "choice": "lick > no-lick", "whisker_choice": "hit > miss",
    "auditory_choice": "hit > miss", "baseline_choice": "lick > no-lick (pre-stim)",
    "baseline_whisker_choice": "hit > miss (pre-stim)", "baseline_auditory_choice": "hit > miss (pre-stim)",
    "baseline_pre_vs_post_learning": "post > pre session (pre-stim)", "whisker_sensory": "miss > CR",
    "auditory_sensory": "miss > CR", "whisker_hit_vs_cr": "hit > CR", "auditory_hit_vs_cr": "hit > CR",
    "whisker_hit_vs_spontaneous": "hit > spont. lick", "auditory_hit_vs_spontaneous": "hit > spont. lick"}


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
    import ephys_utilities.allen_utils.allen_utils as au
    uinfo = OUT / "unit_info_all_days.parquet"
    if uinfo.exists():
        ut = pd.read_parquet(uinfo)
    else:
        print("loading", CACHE, flush=True)
        with open(CACHE, "rb") as f:
            T = pickle.load(f)
        ut = T["unit_table"]
        cols = KEYS + [c for c in ["quality_label", "bc_label", "area_acronym_custom", "day", "reward_group",
                                   "firing_rate", "waveform_type", "target_region"] if c in ut.columns]
        ut = ut[cols].copy()
        del T
        ut["cluster_id"] = ut.cluster_id.astype(str)
        ut.to_parquet(uinfo, index=False)
    ut["cluster_id"] = ut.cluster_id.astype(str)
    dup = ut.duplicated(KEYS, keep=False)
    if dup.any():
        D = ut[dup]
        same = D.drop_duplicates().duplicated(KEYS, keep=False).sum() == 0
        print(f"WARNING: {int(dup.sum())} unit_table rows with duplicated keys in sessions "
              f"{D.session_id.unique().tolist()}; identical rows: {same}", flush=True)
        ut = ut.drop_duplicates() if same else ut[~dup]               # ambiguous keys dropped
    a2g = {a: g for g, acs in au.get_custom_area_groups().items() for a in acs}
    ut["area_group"] = ut.area_acronym_custom.map(a2g).fillna("Other")
    ut["reward_group"] = ut.reward_group.map(lambda r: "R+" if r in (1, "R+", "1") else "R-" if r in (0, "R-", "0") else r)
    print("unit_table:", len(ut), "units;", ut.groupby("day").session_id.nunique().to_dict(), flush=True)
    mi = pd.read_excel(MOUSE_INFO).rename(columns={"mouse_name": "mouse_id"})[["mouse_id", "learning_category"]]
    rows = []
    files = sorted(glob.glob(str(RES / "*" / "whisker_*" / "roc_analysis" / "*_roc_results_new.csv")))
    for i, f in enumerate(files):
        f = pathlib.Path(f)
        dname = f.parent.parent.name.split("_")[1]
        if not dname.lstrip("-+").isdigit() or int(dname) < 0:
            continue
        d = pd.read_csv(f, usecols=["mouse_id", "session_id", "electrode_group", "cluster_id", "neuron_id",
                                    "analysis_type", "auc", "selectivity", "significant", "p_value_to_show",
                                    "ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv", "modality_preference",
                                    "sensory_label", "whisker_decision", "whisker_gated_decision",
                                    "auditory_decision", "auditory_gated_decision"])
        d["day_folder"] = int(dname)
        rows.append(d)
        if i % 20 == 0:
            print(f"  read {i + 1}/{len(files)}", flush=True)
    R = pd.concat(rows, ignore_index=True)
    R["cluster_id"] = R.cluster_id.astype(str)
    R = R[R.auc.notna()]                                            # analysis not computed for this unit
    R["sel"] = np.where(R.analysis_type.isin(FLIP), -R.selectivity, R.selectivity)
    R["sig"] = R.significant.astype(bool)
    R["pos"] = R.sig & (R.sel > 0)
    R["neg"] = R.sig & (R.sel < 0)
    R["abs_sel"] = R.sel.abs()
    units = R.drop_duplicates(KEYS)[KEYS + ["neuron_id", "day_folder", "ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv",
                                             "modality_preference", "sensory_label", "whisker_decision",
                                             "whisker_gated_decision", "auditory_decision", "auditory_gated_decision"]]
    units = units.merge(ut, on=KEYS, how="left", validate="one_to_one").merge(mi, on="mouse_id", how="left")
    units["day"] = units.day.fillna(units.day_folder)
    units["stage"] = np.where(units.day == 0, "learning", "expert")
    units["cohort"] = units.reward_group
    print("ROC units:", len(units), "| matched to unit_table:", int(units.quality_label.notna().sum()),
          "| good:", int((units.quality_label == "good").sum()), flush=True)
    print(units[units.quality_label == "good"].groupby(["cohort", "stage"]).agg(
        sessions=("session_id", "nunique"), mice=("mouse_id", "nunique"), units=("cluster_id", "size")).to_string(), flush=True)
    units.to_parquet(OUT / "units.parquet", index=False)
    R[KEYS + ["analysis_type", "sel", "abs_sel", "sig", "pos", "neg", "p_value_to_show"]].to_parquet(
        OUT / "roc_long.parquet", index=False)
    json.dump(dict(created="2026-10-02", roc_source="rate-based roc_analysis (026, ROC_USES_RATES=True)",
                   units_source=str(CACHE), stage_definition="learning = day 0, expert = day >= 1",
                   unit_selection="analyses use quality_label == 'good'", join_keys=KEYS,
                   flipped_types=sorted(FLIP), positive_meaning=POS_MEANING, n_roc_files=len(files),
                   runtime_min=round((time.time() - t0) / 60, 1)), open(OUT / "provenance.json", "w"), indent=2)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
