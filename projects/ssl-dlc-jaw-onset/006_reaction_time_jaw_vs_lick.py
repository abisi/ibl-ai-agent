"""Jaw vs lick reaction-time figures (user's plot_group_behavior 'reaction_time_jaw_vs_lick' branch) from the
combined_results_ks4 per-mouse pkls (user 2026-11-12: redo the jaw / lick reaction-time figures with the offset-corrected
lick; piezo_lick_time = lick_time - response_window_start_time since the 2026-11-12 dlc_helpers fix).

Calls beh_plotting_functions.plot_multi_mouse_piezo_vs_dlc_reaction_times and ..._hist unchanged, per whisker day 0, 1,
2, with the branch's colors. Only the inputs differ from the branch: pkls from combined_results_ks4/<mouse>/ (the branch
reads the 2025 combined_results ones), all mice with a pkl (the branch drops AB080, AB082, AB085, which have none here).
Output: combined_results_ks4/reaction_time_jaw_vs_lick/reaction_time_jaw_vs_lick_day<d>/ (figures +
avg_reaction_times_day<d>.csv)
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-dlc-jaw-onset/006_reaction_time_jaw_vs_lick.py
"""

import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "projects" / "ssl-paper-figures"))
from common import axel_bisi_root, import_external_repos  # noqa: E402

ROOT = axel_bisi_root() / "combined_results_ks4"
OUT = ROOT / "reaction_time_jaw_vs_lick"


def main():
    import_external_repos()
    sys.modules.setdefault("pymc", types.ModuleType("pymc"))  # not installed on haas, unused here
    sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))
    import beh_plotting_functions as bpf
    colors = dict(auditory_reward='#0e458a', auditory_no_reward='#3d8beb',
                  whisker_reward=bpf.GROUP_COLORS['rplus'], whisker_no_reward=bpf.GROUP_COLORS['rminus'],
                  no_stim='#262626', correct_rejection='#737373', no_stim_no_reward='#737373')
    pkls = sorted(ROOT.glob("*/dlc_jaw_onset_times.pkl"))
    df = pd.concat([pd.read_pickle(p) for p in pkls], ignore_index=True)
    print(len(pkls), "mice,", len(df), "trials")
    for day in [0, 1, 2]:
        out = OUT / f"reaction_time_jaw_vs_lick_day{day}"
        out.mkdir(parents=True, exist_ok=True)
        d = df[df["day"] == day]
        avg = bpf.plot_multi_mouse_piezo_vs_dlc_reaction_times(data_df=d, color_palette=colors, params={},
                                                               saving_path=str(out))
        avg.to_csv(out / f"avg_reaction_times_day{day}.csv")
        bpf.plot_multi_mouse_piezo_vs_dlc_reaction_times_hist(data_df=d, color_palette=colors,
                                                              params={'time_start': -1, 'time_stop': 2},
                                                              saving_path=str(out))
        print("day", day, "saved", out)


if __name__ == "__main__":
    main()
