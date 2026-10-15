"""177 -- Report figure, across days (user 2026-10-15: report of ssl-hitmiss-axis-passive; auditory control and areas shown at every level).
Plot only, from 175 (sessions table) and 168 (time-resolved projections), with the panels of ssl-prelick-convergence 119 row 1 and 088:
  a  held-out active hits / misses and passive-post whisker trials on the hit - miss axis over time (2 x 2: cohort x stage)
  b  passive pre / post projections, real (colour) vs shifted axes (grey), whisker | auditory (same y scale)
  c  passive pre (whisker), d passive post (whisker), e passive post (auditory), f within-session change (whisker), g within-session
     change (auditory): real - shifted, day 0 vs expert per cohort (088 mean_panel2: Mann-Whitney brackets, cohort x stage permutation)
  h  per area: change of the passive-post whisker projection (expert - day 0), omnibus cohort x stage x area + post hoc (119 area_forest)
Output: figures/publication/177_report_across_days.{png,pdf,svg} + tables 177_report_across_days_stats.csv
Run (haas, ssl-prelick-convergence env): cd ~/code/unit_spikes_analysis && PRELICK_REF=sl PRELICK_UNITS=stable \
  PYTHONPATH=~/code/NWB_reader:.:$HOME/code/ephys_utilities ./.venv/bin/python .../177_report_across_days.py
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"))
m119 = importlib.import_module("119_cosyne_three_rows")
m88, m62 = m119.m88, m119.m62
FIGDIR = EA / "figures" / "publication"
S, GAP, L0, A, TOP, B, T = 1.2, 0.5, 0.45, 0.6, 0.12, 0.42, 0.62


def main():
    import ephys_utilities.allen_utils.allen_utils as au
    gcol = au.get_custom_area_groups_colors()
    plt = m62.setup(); rng = np.random.default_rng(0); rows = []
    HMD = pd.read_csv(EA / "175_across_days_passive_projection_sessions.csv")
    for k in ("W", "A"):
        HMD[f"d{k}_ex"] = HMD[f"post_{k}_ex"] - HMD[f"pre_{k}_ex"]
    HP = []
    for st_, sfx in (("learning", ""), ("expert", "_expert")):
        q = pd.read_parquet(EA / f"168_cd_psth_projection{sfx}_min2.parquet"); q = q[q.area == "All units"].copy(); q["stage"] = st_; HP.append(q)
    HP = pd.concat(HP).rename(columns={"reward_group": "cohort"}); HP = HP[HP.session_id.isin(HMD.session_id)]
    AD = m119.area_data_e(); AQ = m119.qualify(AD)
    ORD = [g for g in au.get_area_group_custom_order() if g in AQ["R+"] and g in AQ["R-"]]
    rc = {"font.size": 5.2, "axes.titlesize": 5.2, "axes.labelsize": 5.0, "xtick.labelsize": 4.6, "ytick.labelsize": 4.6,
          "legend.fontsize": 4.4, "svg.fonttype": "none"}
    with plt.rc_context(rc):
        # one row: a (2 x 2 insets) | b (whisker | auditory) | c-g squares | h areas (extra room for names)
        ncell = 8
        xs = [L0 + c * (S + GAP) for c in range(ncell - 1)]
        xs.append(xs[-1] + S + GAP + A)
        W_, H_ = xs[-1] + S + 0.1, TOP + T + S + B + 0.1
        fig = plt.figure(figsize=(W_, H_)); y0 = B + 0.1
        box = lambda c: [xs[c] / W_, y0 / H_, S / W_, S / H_]
        spec = lambda c: fig.add_gridspec(1, 1, left=xs[c] / W_, right=(xs[c] + S) / W_, bottom=y0 / H_, top=(y0 + S) / H_)[0]
        LS = []
        ia = m119.hm_insets(fig, spec(0), HP, rows); LS.append(ia[0])
        sc = spec(1).subgridspec(1, 2, wspace=0.45); ac = [fig.add_subplot(sc[i]) for i in range(2)]; LS.append(ac[0])
        for j, (k, tt) in enumerate((("W", "Whisker"), ("A", "Auditory"))):
            m119.four(ac[j], HMD, k, tt, show_x=True, show_n=(j == 0)); m119.four_stats(HMD, k, rows)
        ac[0].set_ylabel("projection on hit − miss axis\n(per neuron)", fontsize=4.6)
        yl = (min(a.get_ylim()[0] for a in ac), max(a.get_ylim()[1] for a in ac))
        for a in ac:
            a.set_ylim(*yl)
        ac[1].tick_params(labelleft=False)
        for c, (q, tt) in enumerate((("pre_W_ex", "Passive pre (whisker)"), ("post_W_ex", "Passive post (whisker)"),
                                     ("post_A_ex", "Passive post (auditory)"), ("dW_ex", "Post − pre (whisker)"),
                                     ("dA_ex", "Post − pre (auditory)")), start=2):
            ax = fig.add_axes(box(c)); LS.append(ax)
            m88.mean_panel2(ax, HMD, q, "real − shifted (per neuron)", rng, f"177 {q}", tt, ref=0)
        ax = fig.add_axes(box(7)); LS.append(ax)
        m119.area_forest(ax, AD, ORD, AQ, "Δ passive-post whisker projection\n(real − shifted; expert − day 0)",
                         "Per area: passive post (whisker)", rng, rows, "177h", gcol)
        m62.letter_row(fig, LS, "abcdefgh", dx_in=0.3, dy_in=0.12)
        for ext in ("png", "pdf", "svg"):
            fig.savefig(FIGDIR / f"177_report_across_days.{ext}", dpi=300)
        plt.close(fig)
    R = pd.DataFrame(rows + [r for r in m62.STATS if str(r.get("panel", "")).startswith("177")])
    R.to_csv(EA / "177_report_across_days_stats.csv", index=False)
    print("ALL DONE", FIGDIR / "177_report_across_days.png")


if __name__ == "__main__":
    main()
