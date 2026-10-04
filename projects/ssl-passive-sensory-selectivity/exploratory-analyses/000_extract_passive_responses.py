"""Extract per-unit passive sensory-response summaries, before vs after the active task.

Must be run with M:\\analysis\\Axel_Bisi\\unit_spikes_analysis's own venv
python (has pynwb/statsmodels/etc pinned; NOT the ibl-ai-agent project venv):
  & "M:\\analysis\\Axel_Bisi\\unit_spikes_analysis\\.venv\\Scripts\\python.exe" 000_extract_passive_responses.py

See ../question.md for the full resolved analysis design. Summary of the
per-unit computation:
- passive_pre trials = context=='passive' and before the session's first
  context=='active' trial; passive_post = context=='passive' and after the
  last context=='active' trial (NOT based on epoch-table timestamps, which
  were found unreliable for some sessions in an earlier QA pass).
- response(period, modality) = mean firing rate in the post-stim window minus
  mean firing rate in the baseline window, averaged over that period's trials
  of that modality. Windows (updated 2026-08-06 per user request): baseline
  -55..-5ms; response 5..30ms ("w5_30", the single window of interest). This
  supersedes the original 10ms-dead-zone windows (10-50ms/10-30ms) used in the
  first version of this analysis -- the baseline was shifted to end at -5ms to
  stay symmetric with the new 5ms response-window start, though only the
  response-window change was explicitly requested.
- diff(modality) = response(passive_post) - response(passive_pre).
- selectivity_index = (diff_whisker - diff_auditory) / (diff_whisker + diff_auditory).
- Time-course bins (for the visual PSTH-style figure only) are unchanged from
  the original version: 10ms bins from -50 to +100ms with a -10..10ms
  dead-zone excluded. This is a separate visualization, not the scalar
  "window of interest" used for diff/index/GLMM.

Sessions are processed in small batches via combine_ephys_nwb, and only the
compact per-unit summary rows are kept -- raw spike arrays are discarded
after each batch to bound peak memory across ~204 candidate ephys sessions.
"""
import sys
sys.path.insert(0, r"M:\analysis\Axel_Bisi\unit_spikes_analysis")
sys.path.insert(0, r"M:\analysis\Axel_Bisi\Github\allen_utils")

import glob
import os
import traceback

import numpy as np
import pandas as pd

BASELINE_WINDOW = (-0.055, -0.005)  # 50ms window ending 5ms before stim, symmetric with the response window below
RESPONSE_WINDOWS = {"w5_30": (0.005, 0.030)}  # single window of interest, per user's 2026-08-06 request
MIN_TRIALS_PER_MODALITY_PERIOD = 3

# Time-course bins for the peri-stimulus firing-rate-change figure: 10ms bins
# from -50ms to +100ms, skipping the 0-10ms dead-zone (per user's exclusion
# rule) so no bin straddles or sits inside the excluded window.
_TC_EDGES_MS = [-50, -40, -30, -20, -10, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
TIME_COURSE_BINS = [(a / 1000.0, b / 1000.0) for a, b in zip(_TC_EDGES_MS[:-1], _TC_EDGES_MS[1:]) if not (a == -10)]
TIME_COURSE_BIN_CENTERS = [float((a + b) / 2000.0) for a, b in zip(_TC_EDGES_MS[:-1], _TC_EDGES_MS[1:]) if not (a == -10)]
BATCH_SIZE = 12
NWB_ROOT = r"M:\analysis\Axel_Bisi\NWB_ks4"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "passive_unit_summary.parquet")
FAILURE_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "extraction_failures.csv")
SESSION_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "artifacts", "session_log.csv")


def firing_rate_in_window(spike_times, align_time, window):
    lo, hi = align_time + window[0], align_time + window[1]
    n = np.sum((spike_times >= lo) & (spike_times < hi))
    return n / (window[1] - window[0])


def response_for_trials(spike_times, align_times, window):
    """Mean (response_window_rate - baseline_window_rate) across trials."""
    if len(align_times) == 0:
        return np.nan
    vals = [
        firing_rate_in_window(spike_times, t, window) - firing_rate_in_window(spike_times, t, BASELINE_WINDOW)
        for t in align_times
    ]
    return float(np.mean(vals))


def time_course_for_trials(spike_times, align_times):
    """Per-bin mean (bin_rate - baseline_rate) across trials, one value per TIME_COURSE_BINS entry."""
    if len(align_times) == 0:
        return [np.nan] * len(TIME_COURSE_BINS)
    baseline = np.mean([firing_rate_in_window(spike_times, t, BASELINE_WINDOW) for t in align_times])
    return [
        float(np.mean([firing_rate_in_window(spike_times, t, window) for t in align_times]) - baseline)
        for window in TIME_COURSE_BINS
    ]


def session_passive_trials(trials_session):
    t = trials_session.sort_values("start_time")
    active = t[t["context"] == "active"]
    passive = t[t["context"] == "passive"]
    if len(active) == 0 or len(passive) == 0:
        return None, None, 0, 0
    first_active, last_active = active["start_time"].min(), active["start_time"].max()
    passive_pre = passive[passive["start_time"] < first_active]
    passive_post = passive[passive["start_time"] > last_active]
    n_wh = int((active["trial_type"] == "whisker_trial").sum())
    n_aud = int((active["trial_type"] == "auditory_trial").sum())
    return passive_pre, passive_post, n_wh, n_aud


def process_batch(nwb_batch, day_to_analyze, area_groups_from_name):
    from neural_utils import combine_ephys_nwb
    import allen_utils as allen

    trial_table, unit_table, _ = combine_ephys_nwb(nwb_batch, day_to_analyze=day_to_analyze, max_workers=4)
    if unit_table.empty:
        return [], []

    unit_table = allen.create_area_custom_column(unit_table)

    rows = []
    session_rows = []
    for session_id, unit_sess in unit_table.groupby("session_id"):
        trials_sess = trial_table[trial_table["session_id"] == session_id]
        if trials_sess.empty:
            continue
        passive_pre, passive_post, n_wh_active, n_aud_active = session_passive_trials(trials_sess)
        session_rows.append({
            "session_id": session_id,
            "mouse_id": unit_sess["mouse_id"].iloc[0],
            "day": unit_sess["day"].iloc[0],
            "n_units": len(unit_sess),
            "has_valid_passive": passive_pre is not None and len(passive_pre) > 0 and len(passive_post) > 0,
            "n_whisker_active": n_wh_active,
            "n_auditory_active": n_aud_active,
        })
        if passive_pre is None or len(passive_pre) == 0 or len(passive_post) == 0:
            continue

        align_times = {}
        skip_modality = set()
        for modality, trial_type in [("whisker", "whisker_trial"), ("auditory", "auditory_trial")]:
            pre_t = passive_pre.loc[passive_pre["trial_type"] == trial_type, "stim_onset"].dropna().to_numpy()
            post_t = passive_post.loc[passive_post["trial_type"] == trial_type, "stim_onset"].dropna().to_numpy()
            align_times[modality] = (pre_t, post_t)
            if len(pre_t) < MIN_TRIALS_PER_MODALITY_PERIOD or len(post_t) < MIN_TRIALS_PER_MODALITY_PERIOD:
                skip_modality.add(modality)

        for _, u in unit_sess.iterrows():
            spike_times = np.asarray(u["spike_times"], dtype=np.float64)
            area_custom = u.get("area_acronym_custom")
            area_group = area_groups_from_name.get(area_custom, "Unassigned")
            row = {
                "unit_key": f"{session_id}_{u['unit_id']}",
                "session_id": session_id,
                "mouse_id": u["mouse_id"],
                "reward_group": u.get("reward_group"),
                "day": u["day"],
                "day_stage": "learning" if u["day"] == 0 else "expert",
                "bc_label": u.get("bc_label"),
                "area_acronym_custom": area_custom,
                "area_group": area_group,
                "n_whisker_active": n_wh_active,
                "n_auditory_active": n_aud_active,
            }
            any_modality = False
            for modality in ["whisker", "auditory"]:
                if modality in skip_modality:
                    for wname in RESPONSE_WINDOWS:
                        row[f"response_pre_{modality}_{wname}"] = np.nan
                        row[f"response_post_{modality}_{wname}"] = np.nan
                        row[f"diff_{modality}_{wname}"] = np.nan
                    row[f"tc_pre_{modality}"] = [np.nan] * len(TIME_COURSE_BINS)
                    row[f"tc_post_{modality}"] = [np.nan] * len(TIME_COURSE_BINS)
                    continue
                pre_t, post_t = align_times[modality]
                for wname, window in RESPONSE_WINDOWS.items():
                    r_pre = response_for_trials(spike_times, pre_t, window)
                    r_post = response_for_trials(spike_times, post_t, window)
                    row[f"response_pre_{modality}_{wname}"] = r_pre
                    row[f"response_post_{modality}_{wname}"] = r_post
                    row[f"diff_{modality}_{wname}"] = r_post - r_pre
                row[f"tc_pre_{modality}"] = time_course_for_trials(spike_times, pre_t)
                row[f"tc_post_{modality}"] = time_course_for_trials(spike_times, post_t)
                any_modality = True
            if not any_modality:
                continue
            for wname in RESPONSE_WINDOWS:
                dw, da = row.get(f"diff_whisker_{wname}"), row.get(f"diff_auditory_{wname}")
                denom = (dw + da) if (dw is not None and da is not None) else np.nan
                row[f"selectivity_index_{wname}"] = (dw - da) / denom if denom and abs(denom) > 1e-9 else np.nan
            rows.append(row)

    return rows, session_rows


def main():
    import json

    all_files = sorted(glob.glob(os.path.join(NWB_ROOT, "*.nwb")))
    print(f"{len(all_files)} total NWB files found under {NWB_ROOT}")

    bin_centers_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "time_course_bin_centers_s.json")
    with open(bin_centers_path, "w") as f:
        json.dump(TIME_COURSE_BIN_CENTERS, f)

    import allen_utils as allen
    area_groups_from_name = allen.get_custom_area_groups_from_name()

    all_rows = []
    all_session_rows = []
    failures = []
    batches = [all_files[i:i + BATCH_SIZE] for i in range(0, len(all_files), BATCH_SIZE)]
    for bi, batch in enumerate(batches, start=1):
        print(f"=== batch {bi}/{len(batches)} ({len(batch)} files) ===")
        try:
            rows, session_rows = process_batch(batch, day_to_analyze="all", area_groups_from_name=area_groups_from_name)
            all_rows.extend(rows)
            all_session_rows.extend(session_rows)
            print(f"  -> {len(rows)} units, {len(session_rows)} sessions so far this batch")
        except Exception as exc:
            failures.append({"batch": bi, "files": ";".join(batch), "error": str(exc), "traceback": traceback.format_exc()})
            print(f"  BATCH FAILED: {exc}")

        # checkpoint after every batch so partial progress survives a crash
        if all_rows:
            pd.DataFrame(all_rows).to_parquet(OUTPUT_PATH, index=False)
        if all_session_rows:
            pd.DataFrame(all_session_rows).to_csv(SESSION_LOG_PATH, index=False)
        if failures:
            pd.DataFrame(failures).to_csv(FAILURE_LOG_PATH, index=False)

    print()
    print(f"DONE. {len(all_rows)} unit rows, {len(all_session_rows)} session rows, {len(failures)} batch failures.")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
