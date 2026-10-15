"""175 -- Hit - miss axis across days: passive whisker responses projected on each session's hit - miss axis, day 0 vs expert
(user 2026-10-11: "port the hit miss coding direction analysis of the within day to the across days, projecting passive data onto that
axis and comparing"). Exploratory. 173 row 1 kept (angles, >= 10 hits and misses).
Axis and projections: 135 with SSL_135_HARM=1 (unit mean / SD from the active trials), tracked stable units (137b), whole brain
("All units"), 5-35 ms after stimulus onset, and SSL_MIN_CLASS=2: >= 4 licked and >= 4 unlicked active whisker trials (2 per half), so
that expert sessions (R+ rarely miss, R- rarely hit) can enter; the axis is mean(hit) - mean(miss) on one random half of the active
trials (unit norm), averaged over 50 splits; passive whisker (W) and auditory (A) pre / post means projected on it, per neuron (/ sqrt(N)).
Linear-shift null (135 / shift_null.py): axes rebuilt from hit / miss labels shifted against the time-ordered active whisker trials
(10-50 %, non-wrapping); null = mean over shifts. Every quantity is real - null.
Panels (whole brain, learners on day 0, all mice at expert; session = unit):
  a  passive whisker projection, pre and post, at day 0 and expert: real (filled / open) and shift null (grey), mean +- s.e.m.
  b  pre: passive whisker projection before the session, real - null (does the expert passive response already sit on the hit side?)
  c  post: same after the session
  d  within-session change post - pre, real - null (the within-day quantity of 165 d), day 0 vs expert
  e  auditory control: within-session change post - pre of the passive auditory projection, real - null
Statistics (173): per cohort day 0 vs expert Mann-Whitney U (legend; Welch t in the csv); title: cohort x stage interaction
[E - L](R+) - [E - L](R-), cohort labels permuted across mice (10,000; a mouse keeps its sessions).
Output: figures/publication/175_across_days_passive_projection.{svg,pdf,png} + tables 175_across_days_passive_projection_{stats,sessions}.csv
Run (haas, repo root; after 135 with SSL_MIN_CLASS=2 SSL_135_UNITSET=tracked SSL_135_HARM=1 for SSL_STAGE=learning and expert):
  python .../175_across_days_passive_projection.py [v2]   (v2: the figure with 088 panels; run with the ssl-prelick-convergence env)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EA = Path(__file__).resolve().parent
sys.path[:0] = [str(EA), str(EA.parents[2] / "scripts")]
M173 = importlib.import_module("173_cosyne_across_days")
H, M165 = M173.H, M173.M165
COL, COH, FIGDIR, STG, STL = H.COL, H.COH, (Path(__file__).resolve().parent / "figures" / "publication"), M173.STG, M173.STL
TAG = "_min2_tracked_harm"
plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


def load():
    rows = []
    for st_, sfx in (("learning", ""), ("expert", "_expert")):
        A = pd.read_parquet(EA / f"135_alignment_epochs{TAG}{sfx}.parquet")
        A = A[(A.area == "All units") & A.skipped_reason.isna()].copy(); A["stage"] = st_
        rows.append(A)
    D = pd.concat(rows).rename(columns={"reward_group": "cohort"})
    D = D[(D.stage == "expert") | H.mouse_of(D).isin(H.learners())].copy()
    for e in ("pre", "post"):
        for k in ("W", "A"):
            D[f"{e}_{k}_real"], D[f"{e}_{k}_null"] = D[f"shift_real_{e}_{k}"], D[f"shift_null_{e}_{k}_mean"]
            D[f"{e}_{k}_ex"] = D[f"{e}_{k}_real"] - D[f"{e}_{k}_null"]
    for k in ("W", "A"):
        D[f"d{k}_ex"] = D[f"post_{k}_ex"] - D[f"pre_{k}_ex"]
    return D


def four(ax, D):
    xs = [0, 1, 2.4, 3.4]
    for j, c in enumerate(COH):
        for k, s_ in enumerate(STG):
            d = D[(D.cohort == c) & (D.stage == s_)]
            x = np.array(xs[2 * k: 2 * k + 2]) + (j - 0.5) * 0.14
            for kind, col, ls in (("null", "0.6", (0, (2, 1.5))), ("real", COL[c], "-")):
                v = d[[f"pre_W_{kind}", f"post_W_{kind}"]]
                ax.errorbar(x, v.mean(), v.sem(), fmt="o", ls=ls, ms=2.8, color=col, lw=1.0, capsize=0,
                            mfc=col if s_ == "expert" else "white", mew=0.7, zorder=3 if kind == "real" else 2)
            ax.text(xs[2 * k] + 0.5, 0.92 - 0.07 * j, f"n = {len(d)}", color=COL[c], fontsize=4.3, ha="center", transform=ax.get_xaxis_transform())
    ax.axhline(0, color="0.6", lw=0.5, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels(["pre", "post", "pre", "post"], fontsize=5)
    for x0, s_ in ((0.5, "learning"), (2.9, "expert")):
        ax.text(x0, -0.2, STL[s_], ha="center", fontsize=5.2, transform=ax.get_xaxis_transform())
    ax.set_ylabel("passive whisker projection on\nhit − miss axis (per neuron)", fontsize=5.5)
    ax.set_title("real (colour) vs shifted axes (grey)", fontsize=5.5, pad=4)


def main():
    D = load()
    with plt.rc_context({"font.size": 5.5, "xtick.labelsize": 5, "ytick.labelsize": 5}):
        fig, axs = plt.subplots(1, 5, figsize=(7.2, 1.95), gridspec_kw=dict(wspace=0.75, width_ratios=[1.35, 1, 1, 1, 1]))
        four(axs[0], D)
        for ax, (q, yl, key) in zip(axs[1:], (("pre_W_ex", "pre: projection\nreal − shifted", "pre W"),
                                             ("post_W_ex", "post: projection\nreal − shifted", "post W"),
                                             ("dW_ex", "post − pre (whisker)\nreal − shifted", "change W"),
                                             ("dA_ex", "post − pre (auditory)\nreal − shifted", "change A"))):
            M173.stage_lines(ax, D, q, yl, key, ref=0)
        for ax, l in zip(axs, "abcde"):
            ax.text(-0.42, 1.32 if l != "a" else 1.12, l, transform=ax.transAxes, fontsize=8, weight="bold")
        fig.subplots_adjust(left=0.07, right=0.99, top=0.72, bottom=0.17)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(FIGDIR / f"175_across_days_passive_projection.{ext}", dpi=300)
    S = pd.DataFrame(M173.R); S.to_csv(EA / "175_across_days_passive_projection_stats.csv", index=False)
    D[["session_id", "mouse_id", "cohort", "stage", "n_units", "n_wh_licked", "n_wh_unlicked"] +
      [c for c in D.columns if c.endswith(("_real", "_null", "_ex"))]].to_csv(EA / "175_across_days_passive_projection_sessions.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(4).to_string(index=False))
    print(D.groupby(["cohort", "stage"]).agg(n=("session_id", "nunique"), hits=("n_wh_licked", "median"), misses=("n_wh_unlicked", "median")))


def figure2():
    """v2 (user 2026-10-11 "plot all of that into a figure"): b-e as the ssl-prelick-convergence panels (088 mean_panel2): day 0 vs
    expert per cohort, R+ vs R- at day 0 and at expert (Mann-Whitney), cohort x stage (mouse permutation), n per group; dashed: 0"""
    sys.path.insert(0, str(EA.parents[1] / "ssl-prelick-convergence" / "exploratory-analyses"))
    m88 = importlib.import_module("088_cosyne_across_days_v2"); m62 = m88.m62
    D = load(); rng = np.random.default_rng(0)
    plt_ = m62.setup()
    with plt_.rc_context({"font.size": 5.4, "axes.labelsize": 5.4, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8, "svg.fonttype": "none"}):
        fig = plt_.figure(figsize=(7.2, 2.15))
        gs = fig.add_gridspec(1, 5, width_ratios=[1.3, 1, 1, 1, 1], wspace=0.75, left=0.065, right=0.99, top=0.74, bottom=0.2)
        ax = fig.add_subplot(gs[0, 0]); four(ax, D); ax.set_box_aspect(1); L = [ax]
        for j, (q, yl, tt) in enumerate((("pre_W_ex", "projection, real − shifted", "Passive pre (whisker)"),
                                         ("post_W_ex", "projection, real − shifted", "Passive post (whisker)"),
                                         ("dW_ex", "post − pre, real − shifted", "Within session (whisker)"),
                                         ("dA_ex", "post − pre, real − shifted", "Within session (auditory)"))):
            ax = fig.add_subplot(gs[0, 1 + j]); ax.set_box_aspect(1); L.append(ax)
            m88.mean_panel2(ax, D, q, yl, rng, f"175 {q}", tt, ref=0)
        m62.letter_row(fig, L, "abcde", dx_in=0.26, dy_in=0.14)
        fig.suptitle("Passive whisker responses on each session's hit − miss axis (5–35 ms; per neuron; learners)", x=0.01, y=0.995,
                     ha="left", fontsize=6.4, weight="bold")
        for ext in ("svg", "pdf", "png"):
            fig.savefig(FIGDIR / f"175_across_days_passive_projection_v2.{ext}", dpi=300)
    St = pd.DataFrame([r for r in m62.STATS if str(r.get("panel", "")).startswith("175")])
    St.to_csv(EA / "175_across_days_passive_projection_v2_stats.csv", index=False)
    pd.set_option("display.width", 250)
    print(St[[c for c in St.columns if c.startswith(("panel", "mean_", "n_", "interaction", "R+ L", "R- L", "learning R", "expert R"))]].round(4).to_string())


if __name__ == "__main__":
    figure2() if sys.argv[1:] == ["v2"] else main()
