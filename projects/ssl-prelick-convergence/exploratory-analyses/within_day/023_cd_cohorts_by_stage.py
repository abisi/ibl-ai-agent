"""023 -- Time-normalised reward-lick CD figures with the cohorts contrasted within each learning stage (user 2026-10-07: the
011 / 020 figures contrast the stages within each cohort; "do the same contrasting cohorts per learning stage, for
whole-brain and per area, in the same folder").

Same data and quantities as 011 (whole brain, 002 trial_scores.parquet) and 020 (per area group, 015
area_trial_scores.parquet; area groups with >= 3 sessions in every cohort x stage condition): 5 bins of normalised
session time tau, per-session class means, mean +- s.e.m. over sessions; drifts D_SL = beta_WH - beta_SL and
D_AH = beta_WH - beta_AH per session (Eq. 17).
  row 1  WH - SL and AH - WH across the session, R+ and R- overlaid: day 0 | expert
  row 2  the class scores (WH, AH, SL) of R+ and R- overlaid, day 0 | expert, for WH (solid) only plus the two references
         (dashed) -- one panel per cohort x stage would repeat 011; here WH of both cohorts with their own SL / AH levels
  row 3  per-session drifts, R+ vs R- within each stage (dots = sessions, mean +- s.e.m.; p: mouse-level cohort
         permutation, MWU and Welch in the stats table)
Output: whole brain within_day/<ref>/cd_trajectories/<pop>/cd_trajectories_by_stage.{png,pdf,svg};
        per area within_day/<ref>/area_cd_drift/<pop>/per_area/<NN>_<area>_by_stage.{png,pdf,svg}; stats_by_stage.csv.
"""
import argparse
import importlib
import pathlib
import re
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
m011 = importlib.import_module("011_cd_trajectories")
m015 = importlib.import_module("015_area_cd_drift")
m017 = importlib.import_module("017_area_cd_drift_views")
m022 = importlib.import_module("022_wh_trial_trajectories")
COH, CL = m62.COH, m62.CL
STG = m011.STG


def figure(plt, T, out, name, title):
    S, _, D = m011.session_table(T)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(m62.W_IN, 5.9))
    gs = fig.add_gridspec(3, 4, hspace=0.8, wspace=0.55, left=0.08, right=0.98, top=0.9, bottom=0.07)
    panels = [(c, st, sl) for c in ("WH_SL", "AH_WH") for st, sl in STG]
    axs1 = []
    for j, (col, st, sl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j]); axs1.append(ax); ax.set_box_aspect(1)
        for coh in ("R+", "R-"):
            m011.traj(ax, D, coh, st, [col], [COH[coh]], [f"{coh.replace('-', '−')} ({D[(D.cohort == coh) & (D.stage == st)].session_id.nunique()})"])
        ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.set_xlim(0, 1)
        ax.set_xlabel("Time in session τ", fontsize=5)
        ax.set_ylabel("WH − SL (score)" if col == "WH_SL" else "AH − WH (distance to AH)", fontsize=5)
        ax.set_title(f"{sl}: " + ("WH relative to SL" if col == "WH_SL" else "WH distance to AH"), fontsize=5.6)
        ax.legend(frameon=False, fontsize=4.5, loc="upper left")
    for a_, b_ in ((0, 1), (2, 3)):
        yl = (min(axs1[a_].get_ylim()[0], axs1[b_].get_ylim()[0]), max(axs1[a_].get_ylim()[1], axs1[b_].get_ylim()[1]))
        axs1[a_].set_ylim(*yl); axs1[b_].set_ylim(*yl)
    axs2 = []
    for j, (coh, (st, sl)) in enumerate([(c, s) for s in STG for c in ("R+", "R-")]):
        ax = fig.add_subplot(gs[1, j]); axs2.append(ax); ax.set_box_aspect(1)
        n = m011.traj(ax, D, coh, st, ["WH", "AH", "FA"], [CL["WH"], CL["AH"], CL["FA"]], ["WH", "AH", "SL"])
        ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.axhline(1, color=CL["AH"], lw=0.4, ls=":")
        ax.set_xlim(0, 1); ax.set_xlabel("Time in session τ", fontsize=5)
        ax.set_title(f"{sl}, {coh.replace('-', '−')} ({n})", color=COH[coh], fontsize=5.6)
        if j == 0:
            ax.set_ylabel("Score (SL = 0, AH = 1)", fontsize=5); ax.legend(frameon=False, fontsize=4.4, loc="upper left")
    yl = (min(a.get_ylim()[0] for a in axs2), max(a.get_ylim()[1] for a in axs2))
    for a in axs2:
        a.set_ylim(*yl)
    axs3 = []
    for j, (col, st, sl) in enumerate(panels):
        ax = fig.add_subplot(gs[2, j]); axs3.append(ax); ax.set_box_aspect(1)
        m = "drift_WH_SL" if col == "WH_SL" else "drift_WH_AH"
        m022.cohort_dots(ax, S, st, m, rng, "Drift over one session" if j == 0 else "")
        ax.set_title(f"{sl}: β_WH − β_" + ("SL" if col == "WH_SL" else "AH"), fontsize=5.4)
    m62.letter_row(fig, axs1, "abcd"); m62.letter_row(fig, axs2, "efgh"); m62.letter_row(fig, axs3, "ijkl")
    fig.suptitle(title, x=0.02, y=0.995, ha="left", va="top", fontsize=6.8, weight="bold")
    m62.save(fig, out, name); plt.close(fig)


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup()
    gcol = au.get_custom_area_groups_colors()
    T0 = pd.read_parquet(m51.WITHIN / "slopes" / "trial_scores.parquet")
    TA0 = pd.read_parquet(m015.OUT / "area_trial_scores.parquet")
    A0 = pd.read_csv(m015.OUT / "area_sessions.csv")
    for pop in ("learners", "all"):
        flt = m61.learner_filter if pop == "learners" else (lambda d: d)
        out = m011.OUT / pop; out.mkdir(parents=True, exist_ok=True)
        figure(plt, flt(T0), out, "cd_trajectories_by_stage",
               f"Reward-lick CD within sessions: R+ vs R− within each stage (whole brain; {pop})")
        pd.DataFrame(m62.STATS).to_csv(out / "stats_by_stage.csv", index=False); m62.STATS.clear()
        TA, A = flt(TA0), flt(A0)
        keep = m017.strict_areas(A, "drift_WH_SL") & m017.strict_areas(A, "drift_WH_AH")
        pa = m015.OUT / pop / "per_area"; pa.mkdir(parents=True, exist_ok=True)
        for i, g in enumerate(m017.order_of(keep)):
            figure(plt, TA[TA.area_group == g], pa, f"{i + 1:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', g).strip('_')}_by_stage",
                   f"{g}: reward-lick CD from this area's units, R+ vs R− within each stage ({pop})")
            print(pop, g, flush=True)
        pd.DataFrame(m62.STATS).to_csv(pa / "stats_by_stage.csv", index=False); m62.STATS.clear()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
