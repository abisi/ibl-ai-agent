"""Shared input glm_v3: run Myriam Hamon's behaviour GLM v3 (behaviour_analysis/glm_behavior/glm_behavior_v3.py,
process_one_mouse) for whisker day 0 on haas, without editing her file. Settings are her __main__ defaults (bin 0.2 s,
pre-stimulus window -0.2..0 s, 10 CV folds, RT head, nested comparison, shuffle importance with 10 repeats, whole
session); only paths change (POSIX NAS paths instead of UNC) and the mouse list: reference-sheet learners
(learning_category good/moderate, user 2026-11-02) instead of her hard-coded non-learner list.
Outputs: Figures/_cache/glm_v3/<mouse>/... and all_sessions_summary.csv.

Run on haas (repo root), about an hour:
    nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/glm_v3_run.py [--n-jobs 24] [--mice AB164]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, EXTERNAL_REPOS, NWB_DIR, axel_bisi_root, import_external_repos  # noqa: E402
from shared_behaviour import mouse_reference  # noqa: E402

OUT = CACHE_DIR / "glm_v3"
SETTINGS = dict(bin_size=0.2, pre_stim_window=(-0.2, 0.0), n_cv_folds=10, fit_rt_head=True, use_movement_energy=False,
                run_shuffle_importance=True, n_shuffle_repeats=10, run_nested_comparison=True, day=0,
                trial_type="whisker_trial", session_half=None)
HISTORY = ["lick_whisker_trial_n-1", "lick_whisker_trial_n-2", "lick_whisker_trial_n-3", "spontaneous_lick_rate",
           "time_since_reward", "cumulative_reward_fraction"]
MOVEMENT = ["jaw_speed_pre", "nose_speed_pre", "whisker_speed_pre", "pupil_area_pre"]


def glm_module():
    import_external_repos()
    sys.path.insert(0, str(EXTERNAL_REPOS["behaviour_analysis"] / "glm_behavior"))
    import glm_behavior_v3 as glm
    glm.ROOT_PATH = str(axel_bisi_root() / "combined_results")  # was a UNC path (only used by unused loaders)
    return glm


def run_mouse(mouse_id, cohort, nwb_files):
    glm = glm_module()
    groups = {name: [name] for name in HISTORY + MOVEMENT}
    groups["all_movement"] = list(MOVEMENT)
    return glm.process_one_mouse(
        mouse_id, cohort, nwb_files, str(OUT), SETTINGS["bin_size"], SETTINGS["pre_stim_window"],
        SETTINGS["n_cv_folds"], HISTORY, groups, SETTINGS["run_nested_comparison"], SETTINGS["fit_rt_head"],
        SETTINGS["use_movement_energy"], SETTINGS["run_shuffle_importance"], SETTINGS["n_shuffle_repeats"],
        SETTINGS["day"], SETTINGS["trial_type"], str(axel_bisi_root() / "combined_results"), SETTINGS["session_half"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--mice", nargs="*")
    args = parser.parse_args()
    ref = mouse_reference()
    ref = ref[ref.learning_category.isin(["good", "moderate"])]
    if args.mice:
        ref = ref[ref.mouse_id.isin(args.mice)]
    jobs = []
    for r in ref.itertuples():
        files = sorted(str(f) for f in NWB_DIR.glob(f"{r.mouse_id}_*.nwb"))
        if files:
            jobs.append((r.mouse_id, r.cohort, files))
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} mice -> {OUT}")
    results = Parallel(n_jobs=min(args.n_jobs, len(jobs)), verbose=5)(delayed(run_mouse)(*j) for j in jobs)
    rows = [r for mouse in results for r in mouse]
    if rows:
        df = pd.DataFrame(rows).drop(columns=["nested_results"], errors="ignore")
        tmp = OUT / "all_sessions_summary.tmp"
        df.to_csv(tmp, index=False)
        tmp.rename(OUT / "all_sessions_summary.csv")
        print(f"{len(df)} sessions, {df.mouse_id.nunique()} mice")


if __name__ == "__main__":
    main()
