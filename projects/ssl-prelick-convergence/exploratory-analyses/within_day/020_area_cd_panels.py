"""020 -- The reward-lick CD analyses of 011 / 019 repeated within each area group, learning day vs expert sessions
(user 2026-10-07: "repeat the same types of figures per area group, in particular also contrast learning day and expert").

Data: 015 area_trial_scores.parquet -- every event's score on the CD built from the units of one area group in that
session (>= 5 units, held-out AH-vs-SL d' >= 0.3; SL = 0, AH = 1). Area groups: >= 3 sessions in every cohort x stage
condition (017 strict rule) for both drifts.
One figure per area group (square panels):
  row 1  score of every event type across normalised session time (5 bins, mean +- s.e.m. over sessions): R+ day 0,
         R+ expert, R- day 0, R- expert;
  row 2  WH - SL and AH - WH across the session, day 0 (dashed, open) vs expert (solid), R+ and R-;
  row 3  drift per session beta_WH - beta_SL and beta_WH - beta_AH (OLS slopes against tau): day 0 vs expert per cohort
         (MWU), R+ vs R- per stage, cohort x stage interaction (mouse-level permutation; 062 dots_panel), and the per-class
         slope of WH.
Summary across area groups: within_day/<ref>/area_cd_drift/<pop>/per_area/area_stage_contrasts.csv (011 tests per area:
drift vs 0 per group, day 0 vs expert per cohort, R+ vs R- per stage, cohort x stage interaction).
Output: within_day/<ref>/area_cd_drift/<pop>/per_area/<NN>_<area>.{png,pdf,svg}.
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
COH, CL = m62.COH, m62.CL
STG = m011.STG


def area_figure(plt, T, area, out, idx, pop, gcol):
    S, _, D = m011.session_table(T)
    TT = m011.tests(S).assign(area_group=area)
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(m62.W_IN, 5.6))
    gs = fig.add_gridspec(3, 4, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.9, bottom=0.07)
    axs1 = []
    for j, (coh, (st, sl)) in enumerate([(c, s) for c in ("R+", "R-") for s in STG]):
        ax = fig.add_subplot(gs[0, j]); axs1.append(ax); ax.set_box_aspect(1)
        n = m011.traj(ax, D, coh, st, ["WH", "AH", "FA"], [CL["WH"], CL["AH"], CL["FA"]], ["WH", "AH", "SL"])
        ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.axhline(1, color=CL["AH"], lw=0.4, ls=":")
        ax.set_xlim(0, 1); ax.set_title(f"{coh.replace('-', '−')} {sl.lower()} ({n})", color=COH[coh], fontsize=5.6)
        ax.set_xlabel("Time in session τ", fontsize=5)
        if j == 0:
            ax.set_ylabel("Pre-lick score\n(SL = 0, AH = 1)"); ax.legend(frameon=False, fontsize=4.4, loc="upper left")
    yl = (min(a.get_ylim()[0] for a in axs1), max(a.get_ylim()[1] for a in axs1))
    for a in axs1:
        a.set_ylim(*yl)
    axs2 = []
    for j, (col, coh) in enumerate([(c, h) for c in ("WH_SL", "AH_WH") for h in ("R+", "R-")]):
        ax = fig.add_subplot(gs[1, j]); axs2.append(ax); ax.set_box_aspect(1)
        for st, sl in STG:
            m011.traj(ax, D, coh, st, [col], [COH[coh]], [sl], ls="--" if st == "learning" else "-",
                      mfc="white" if st == "learning" else None)
        ax.axhline(0, color="0.6", lw=0.4, ls=":"); ax.set_xlim(0, 1)
        ax.set_xlabel("Time in session τ", fontsize=5)
        ax.set_ylabel("WH − SL" if col == "WH_SL" else "AH − WH (distance to AH)", fontsize=5)
        ax.set_title(f"{coh.replace('-', '−')}: " + ("WH relative to SL" if col == "WH_SL" else "WH distance to AH"),
                     color=COH[coh], fontsize=5.6)
        ax.legend(frameon=False, fontsize=4.4, loc="upper left")
    for a_, b_ in ((0, 1), (2, 3)):
        yl = (min(axs2[a_].get_ylim()[0], axs2[b_].get_ylim()[0]), max(axs2[a_].get_ylim()[1], axs2[b_].get_ylim()[1]))
        axs2[a_].set_ylim(*yl); axs2[b_].set_ylim(*yl)
    axs3 = []
    for j, (col, ttl) in enumerate([("drift_WH_SL", "Drift β_WH − β_SL"), ("drift_WH_AH", "Drift β_WH − β_AH"),
                                    ("beta_WH", "Slope of WH β_WH"), ("beta_FA", "Slope of SL β_SL")]):
        ax = fig.add_subplot(gs[2, j]); axs3.append(ax); ax.set_box_aspect(1)
        m62.dots_panel(ax, S, col, "Change over one session" if j == 0 else "", rng, f"A-{area} {col}", ttl, ref=[(0, "0.6")])
        ax.set_title(ax.get_title(), fontsize=5.0)
    m62.letter_row(fig, axs1, "abcd"); m62.letter_row(fig, axs2, "efgh"); m62.letter_row(fig, axs3, "ijkl")
    fig.suptitle(f"{area}: reward-lick CD from this area's units, learning day vs expert ({pop}; "
                 f"{S.session_id.nunique()} sessions)", x=0.02, y=0.995, ha="left", va="top", fontsize=6.8, weight="bold",
                 color=gcol.get(area, "k"))
    m62.save(fig, out, f"{idx:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', area).strip('_')}"); plt.close(fig)
    return TT


def main(a):
    import ephys_utilities.allen_utils.allen_utils as au
    plt = m62.setup()
    gcol = au.get_custom_area_groups_colors()
    T0 = pd.read_parquet(m015.OUT / "area_trial_scores.parquet")
    A0 = pd.read_csv(m015.OUT / "area_sessions.csv")
    for pop in ("learners", "all"):
        flt = m61.learner_filter if pop == "learners" else (lambda d: d)
        T, A = flt(T0), flt(A0)
        keep = m017.strict_areas(A, "drift_WH_SL") & m017.strict_areas(A, "drift_WH_AH")
        out = m015.OUT / pop / "per_area"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, g in enumerate(m017.order_of(keep)):
            rows.append(area_figure(plt, T[T.area_group == g], g, out, i + 1, pop, gcol))
            print(pop, g, flush=True)
        R = pd.concat(rows, ignore_index=True)
        R.to_csv(out / "area_stage_contrasts.csv", index=False)
        pd.DataFrame(m62.STATS).to_csv(out / "stats_dots.csv", index=False); m62.STATS.clear()
        cols = ["area_group", "measure", "learning p_perm", "expert p_perm", "cxs p_perm", "R+ day0 vs expert p_MWU",
                "R- day0 vs expert p_MWU"]
        with pd.option_context("display.width", 250):
            print(R[R.measure.isin(["drift_WH_SL", "drift_WH_AH"])][[c for c in cols if c in R]].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    main(ap.parse_args())
