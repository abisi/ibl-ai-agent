"""Exploration (user 2026-11-10): redo the panel-i alternatives with behaviour_analysis' own
plot_multi_mouse_session_aligned (beh_plotting_functions, NAS checkout, which loads its DLC helpers from ephys_utilities;
the function that made combined_results_ks4/mean_performance_over_trials_day0/perf_aligned_first_whisker_hit_*), with the
options added to it on 2026-11-10: auditory / no-stim panels aligned at their own first trial at or after the first
whisker hit, per-mouse bins, and the R+ vs R- test on per-mouse bin means (Mann-Whitney, Benjamini-Hochberg over bins;
"test differences between cohorts, not between bins").

Input: the shared behaviour table (shared_behaviour.load: behaviour_analysis build_standard_behavior_table on NWB_ks4;
perf != 6, passive removed, day-0 warm-up trimmed), whisker day 0, learners (reference sheet good / moderate), early licks
dropped (Figure 1 convention; the original run kept them); reward_group from the sheet cohort.
Runs (n_trials_shown 50, 10 trials before the alignment in row 2, align first_whisker_hit, variable perf):
  every trial (seaborn, as the original) and bins of 5 trials, both with the per-mouse cohort test.
Output: Figures/_explore/session_aligned_<date>/ (figures + _stats.csv from the function).
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_session_aligned.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, import_external_repos  # noqa: E402


def main():
    from compute import day0_table
    from plot import group_color
    import_external_repos()
    import types
    for name in ("pymc", "arviz"):  # module-level imports used only by curve fitting, not installed in the haas env
        sys.modules.setdefault(name, types.ModuleType(name))
    import beh_plotting_functions as bpf
    d0 = day0_table(shared_behaviour.load(), learners_only=True)
    d0 = d0[d0.early_lick == 0].sort_values(["session_id", "start_time"]).copy()
    d0["reward_group"] = d0.cohort.map({"R+": 1, "R-": 0})
    colors = dict(whisker_reward=group_color("whisker_trial", "R+"), whisker_no_reward=group_color("whisker_trial", "R-"),
                  auditory_reward=group_color("auditory_trial", "R+"),
                  auditory_no_reward=group_color("auditory_trial", "R-"),
                  no_stim=group_color("no_stim_trial", "R+"), no_stim_no_reward=group_color("no_stim_trial", "R-"))
    out = FIGURES_ROOT / "_explore" / f"session_aligned_{datetime.date.today()}"
    out.mkdir(parents=True, exist_ok=True)
    for bin_size in (None, 5):
        params = dict(n_trials_shown=50, align_method="first_whisker_hit", variable="perf", across_days=False,
                      n_trials_max=None, pre_trials=10, align_other_types_in_time=True, bin_size=bin_size,
                      stats="mouse_bins", min_mice=8)
        bpf.plot_multi_mouse_session_aligned(bhv_data=d0, color_palette=colors, params=params, saving_path=str(out))
    print(out, d0.groupby("cohort").mouse_id.nunique().to_dict())


if __name__ == "__main__":
    main()
