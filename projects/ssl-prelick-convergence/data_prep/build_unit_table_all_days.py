"""Build the all-days SSL unit / trial tables (cache used by 045 -> 053 -> pre-lick analyses), version 2.

Same pipeline as ssl-rastermap-psth-variants/000_run_rastermap_variants.load_tables (NWB_ks4 files ->
data_utils.combine_ephys_nwb -> DREDge drift-shift test merge -> presence / coverage -> classify_units_quality ->
process_allen_labels -> keep_shared_areas -> hierarchy / waveform merges), with three rule changes (user 2026-10-03):
  1. The mouse reference sheet column `recording` applies to day 0 only: mice are selected on exclude == 0,
     exclude_ephys == 0 and reward_group in {R+, R-}; day-0 sessions of mice with recording != 1 are dropped after
     loading; expert (day >= 1) sessions are kept whatever `recording` says.
  2. reward_group is the mouse-level value of the sheet (not the session-level NWB wh_reward); session values that
     disagree are reported (sessions_cohort_check.csv).
  3. Exact duplicate unit rows (found 2026-10-02) are dropped before saving.
Drift-shift results: combined_results_ks4/<mouse>/<behaviour>_<day>/single_neuron_motion_shift_test/ (latest test:
unit_fr_motion_shift_test_harris.py); sessions still without results are listed in the provenance (their quality_label
has no drift check).
Output (atomic writes): rastermap_variants/_cache/tables_all_days_v2.pkl (unit_table, trial_table, lick_df, nwb_files,
subject_ids) and _roc_stage_analysis/unit_info_all_days_v2.parquet (metadata columns read by 045) + provenance json.
"""
import json
import os
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "code" / "unit_spikes_analysis"))
import ephys_utilities                                                    # noqa: E402
import ephys_utilities.helpers.load_helpers                               # noqa: E402
from ephys_utilities.helpers import data_utils, load_helpers              # noqa: E402
from ephys_utilities.neural_utils import unit_metrics_utils               # noqa: E402
import ephys_utilities.allen_utils.allen_utils as allen_utils             # noqa: E402

N_WORKERS = 100
NWB_DIR = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
INFO_PATH = pathlib.Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info")
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
CACHE = RES / "rastermap_variants" / "_cache" / "tables_all_days_v2.pkl"
UINFO = RES / "_roc_stage_analysis" / "unit_info_all_days_v2.parquet"
EXCLUDED_MICE = ["AB068", "AB077"]
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]


def atomic_pickle(obj, path):
    tmp = path.with_suffix(".pkl.partial")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main():
    t0 = time.time()
    X = pd.read_excel(INFO_PATH / "joint_mouse_reference_weight.xlsx").rename(columns={"mouse_name": "mouse_id"})
    X = X.drop_duplicates("mouse_id").set_index("mouse_id")
    sel = X[(X.exclude == 0) & (X.exclude_ephys == 0) & X.reward_group.isin(["R+", "R-"])]
    nwb_names = os.listdir(NWB_DIR)
    subject_ids = [m for m in sel.index if m not in EXCLUDED_MICE and any(n.startswith(m + "_") for n in nwb_names)]
    nwb_list = sorted(str(NWB_DIR / n) for n in nwb_names if n.split("_")[0] in subject_ids)
    print(f"{len(subject_ids)} subjects, {len(nwb_list)} NWB files", flush=True)

    trial_table, unit_table, nwb_files = data_utils.combine_ephys_nwb(nwb_list, day_to_analyze="all", max_workers=N_WORKERS)
    # rule 1: `recording` only gates day-0 sessions
    rec = unit_table.mouse_id.map(X.recording).fillna(0).astype(float)
    drop_d0 = (unit_table.day.astype(int) == 0) & (rec != 1)
    dropped = sorted(unit_table[drop_d0].session_id.unique())
    unit_table = unit_table[~drop_d0]
    trial_table = trial_table[~trial_table.session_id.isin(dropped)]
    nwb_files = [f for f in nwb_files if not any(s in f for s in dropped)]
    print(f"day-0 sessions dropped (recording != 1): {dropped}", flush=True)

    dredge = load_helpers.load_motion_dredge_shift_test_results(nwb_files, day_to_analyze="all", max_workers=N_WORKERS)
    dredge = dredge[["mouse_id", "session_id", "cluster_id", "electrode_group", "p_conservative", "r"]].rename(
        columns={"p_conservative": "drift_shift_test_pval"})
    dredge["drift_abs_r"] = dredge["r"].abs()
    unit_table["cluster_id"] = unit_table["cluster_id"].astype(str); dredge["cluster_id"] = dredge["cluster_id"].astype(str)
    unit_table = unit_table.merge(dredge, on=KEYS, how="left", validate="one_to_one")
    no_drift = sorted(unit_table.groupby("session_id").drift_abs_r.apply(lambda s: s.isna().all()).pipe(lambda s: s[s].index))
    unit_table = unit_metrics_utils.compute_presence_coverage_metrics(unit_table)
    unit_table = unit_metrics_utils.classify_units_quality(unit_table, label_col="quality_label")
    # rule 2: mouse-level cohort from the sheet (encoded like the session field: 1 = R+, 0 = R-)
    sess = unit_table.groupby(["mouse_id", "session_id", "day"]).reward_group.first().reset_index()
    sess["sheet"] = sess.mouse_id.map(X.reward_group)
    sess["session_value"] = sess.reward_group.map(lambda r: "R+" if r in (1, "1", "R+") else "R-" if r in (0, "0", "R-") else None)
    sess["status"] = np.where(sess.session_value.isna(), "session label missing",
                              np.where(sess.session_value != sess.sheet, "MISMATCH (sheet used)", "ok"))
    sess.to_csv(RES / "_roc_stage_analysis" / "sessions_cohort_check.csv", index=False)
    print(sess[sess.status != "ok"].to_string(index=False), flush=True)
    unit_table["reward_group"] = unit_table.mouse_id.map(X.reward_group).map({"R+": 1, "R-": 0})
    unit_table = allen_utils.process_allen_labels(unit_table, split_merge_areas=True)
    unit_table, _ = data_utils.keep_shared_areas(unit_table, nomenclature="area_acronym_custom", n_min_units=5, n_min_mice=3)
    unit_table = allen_utils.merge_liu_avg_ipsi(unit_table)
    unit_table = allen_utils.merge_hierarchy_columns_from_gao(unit_table)
    unit_table = allen_utils.merge_hierarchy_from_harris(unit_table)
    lick_df = load_helpers.load_spontaneous_reward_lick_times(nwb_files, day_to_analyze="all", max_workers=N_WORKERS,
                                                              load_summary=False)
    wf = ephys_utilities.helpers.load_helpers.load_wf_analysis_data(nwb_files=nwb_files, experimenter="AB")
    unit_table = unit_table.merge(wf[wf.mouse_id.isin(unit_table.mouse_id.unique())], on=KEYS, how="left")
    # rule 3: exact duplicate unit rows
    ut_key = unit_table.assign(_c=unit_table.cluster_id.astype(str))
    dup = ut_key.duplicated(subset=["mouse_id", "session_id", "electrode_group", "_c"])
    unit_table = unit_table[~dup]
    print(f"dropped {int(dup.sum())} duplicate unit rows", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle(dict(unit_table=unit_table, trial_table=trial_table, lick_df=lick_df, nwb_files=nwb_files,
                       subject_ids=subject_ids), CACHE)
    cols = KEYS + [c for c in ["quality_label", "bc_label", "area_acronym_custom", "day", "reward_group", "firing_rate",
                               "waveform_type", "target_region"] if c in unit_table.columns]
    ui = unit_table[cols].copy(); ui["cluster_id"] = ui.cluster_id.astype(str)
    tmp = UINFO.with_suffix(".partial.parquet"); ui.to_parquet(tmp, index=False); os.replace(tmp, UINFO)
    prov = dict(script="build_unit_table_all_days.py", created=time.strftime("%Y-%m-%d %H:%M"), n_subjects=len(subject_ids),
                n_sessions=int(unit_table.session_id.nunique()), n_units=int(len(unit_table)),
                day0_dropped_recording=dropped, sessions_without_drift_test=no_drift,
                sessions_per_day=unit_table.groupby("day").session_id.nunique().to_dict(),
                runtime_min=round((time.time() - t0) / 60, 1))
    (CACHE.parent / "tables_all_days_v2_provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print(json.dumps(prov, indent=1, default=str), flush=True)


if __name__ == "__main__":
    main()
