"""152 -- Full results deck (user 2026-10-05: "a slide deck that shows all results, from active-only within-session changes to
results that compare passive and active ... from simple to complex, not too many methods at once ... go into a single method
and show it is robust, then show results hold across methods").
Flow: A active-only decoding within the session (one method: hit / miss decoding vs a linear-shift null; halves -> hit-median
split -> learning trial vs placebo -> robustness); B passive vs active with ONE method (lick-axis alignment), made robust step by
step (same units, session-time null, auditory control, learners, noise, area groups); C convergence across methods (coding
direction, decoder readout, state space), controls, synthesis, caveats; backup figures.
Numbers: Part II from the report's numbers.json (same values as the report), Part III re-computed from the per-session tables.
Helpers (slide, panels, schematics) from 148_part3_digest.py.
Output: combined_results_ks4/<slug>/report/<SSL_152_OUT or results_deck.pdf>, figures/deck/152_slide_XX.png
Run (haas, repo root): python .../152_full_deck.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
M = importlib.import_module("148_part3_digest")
H = M.H
from axel_bisi_paths import axel_bisi_root  # noqa: E402

COL, COH, WC, AC = M.COL, M.COH, M.WC, M.AC
FIG, PUB = M.FIG, M.PUB
RDIR = axel_bisi_root() / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "report"
OUT_PDF = RDIR / os.environ.get("SSL_152_OUT", "results_deck.pdf")
M.SLIDES = FIG / "deck"
N = json.loads((RDIR / "numbers.json").read_text())
slide, bullets, takehome, save = M.slide, M.bullets, M.takehome, M.save


def n(key, fmt="{:+.3f}"):
    v = N.get(key)
    return "n/a" if v is None else fmt.format(v)


def p(key):
    v = N.get(key)
    return "n/a" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def pp(k1, k2):
    return f"{p(k1)} | {p(k2)}"


def section(pdf, title, sub):
    fig = plt.figure(figsize=(M.W_IN, M.H_IN))
    fig.text(0.06, 0.55, title, fontsize=34, weight="bold", va="center")
    fig.text(0.06, 0.43, sub, fontsize=16, color="0.35", va="center")
    save(pdf, fig, 0)


def img_slide(pdf, title, sub, path, items, th, img_w=0.6):
    fig = slide(title, sub)
    if path.exists():
        ax = fig.add_axes([0.02, 0.05, img_w, 0.8]); ax.imshow(plt.imread(path)); ax.axis("off")
    else:
        fig.text(0.05, 0.5, f"[{path.name} missing]", fontsize=14, color="r")
    x0 = img_w + 0.04
    bullets(fig, x0, 0.8, items, size=11.5, dy=0.12, width=int((0.96 - x0) * 115))
    takehome(fig, th, x=x0, y=0.05, w=0.96 - x0, h=0.22)
    save(pdf, fig, 0)


# ------------------------------------------------------------------------------------------------------------------- intro
def s_title(pdf):
    fig = plt.figure(figsize=(M.W_IN, M.H_IN))
    fig.text(0.06, 0.74, "Within-session changes of choice and sensory coding\nduring single-session whisker learning", fontsize=28,
             weight="bold", va="center")
    fig.text(0.06, 0.58, "Whole-brain Neuropixels, learning session, R+ (whisker licks rewarded) vs R- (not rewarded)", fontsize=15,
             color="0.3")
    for i, (h, t) in enumerate((("A", "Does choice decoding change within the session? (active trials only)"),
                                ("B", "Does the early whisker response move relative to the choice axis? One method, made robust"),
                                ("C", "Does the result hold across methods? Controls, synthesis, caveats"))):
        fig.text(0.06, 0.42 - i * 0.08, h, fontsize=20, weight="bold", color="0.4")
        fig.text(0.1, 0.42 - i * 0.08, t, fontsize=16)
    save(pdf, fig, 0)


def s_task(pdf):
    fig = slide("Task and question", "Same stimuli, same action, different contingency")
    ax = fig.add_axes([0.04, 0.3, 0.42, 0.5]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, (c, txt) in enumerate((("R+", "whisker -> lick -> reward\n(hit rate rises)"), ("R-", "whisker -> lick -> no reward\n(learn to withhold)"))):
        ax.add_patch(Rectangle((0.02, 0.6 - i * 0.45), 0.96, 0.32, facecolor="white", edgecolor=COL[c], lw=2.5))
        ax.text(0.06, 0.76 - i * 0.45, c, fontsize=20, color=COL[c], weight="bold", va="center")
        ax.text(0.25, 0.76 - i * 0.45, txt, fontsize=14, va="center")
    ax.text(0.02, 0.02, "auditory -> lick -> reward in both cohorts", fontsize=12, color=AC)
    bullets(fig, 0.52, 0.8, ["Hit = lick after the whisker stimulus: the same label in both cohorts (trained response in R+, error in R-).",
                             "H1: choice information after the whisker stimulus grows when R+ mice learn.",
                             "H2: it does not grow, or declines, in R-.",
                             "H3: the early (5-35 ms) whisker response is re-mapped relative to the lick representation, depending on "
                             "the contingency.",
                             "Session = unit of analysis; both a non-parametric and a parametric test, uncorrected (exploratory)."],
            size=13, dy=0.12, width=62)
    save(pdf, fig, 0)


# ------------------------------------------------------------------------------------------------------------------- part A
def s_a_method(pdf):
    fig = slide("A1. One method: decode the upcoming choice from early activity",
                "Hit vs miss on active whisker trials, from the response 5-50 / 5-100 ms after the stimulus")
    ax = fig.add_axes([0.04, 0.35, 0.45, 0.45]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.0, 0.62), 0.38, 0.25, color="#cfe8cf", lw=0)); ax.text(0.19, 0.745, "active whisker trials\nhit | miss", ha="center", fontsize=12)
    ax.annotate("", xy=(0.55, 0.745), xytext=(0.4, 0.745), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.75, 0.745, "cross-validated\nlinear decoder", ha="center", fontsize=12)
    ax.annotate("", xy=(0.75, 0.42), xytext=(0.75, 0.62), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.75, 0.32, "accuracy minus\nlinear-shift null", ha="center", fontsize=12, weight="bold")
    ax.text(0.0, 0.05, "Null: labels shifted against the trials by 10-50 % (no wrap):\nkeeps slow drift, so drift alone is not 'decoding'",
            fontsize=11, color="0.3")
    bullets(fig, 0.54, 0.8, [f"Session-wide, choice is decodable in both cohorts already 5-50 ms after the stimulus: R+ {n('k0_rp')}, "
                             f"R- {n('k0_rm')} above the null.",
                             f"No cohort difference early on (Mann-Whitney | Welch p = {pp('k0_mw', 'k0_w')}).",
                             "Question for part A: does this information change within the session, and differently in the two cohorts?"],
            size=13, dy=0.14, width=60)
    takehome(fig, "The choice after a whisker stimulus is predictable from early whole-brain activity, rewarded or not.", x=0.54, y=0.06, w=0.42, h=0.15)
    save(pdf, fig, 0)


def s_part_a(pdf):
    img_slide(pdf, "A2. Split the session in halves: no change",
              "Count-matched decoders in the first and second half (equal hits and misses per half)",
              FIG / "117b_halves_matched_full.png",
              [f"R+ change {n('h117_rp')} (p = {p('h117_rp_p')}, n = {n('h117_rp_n', '{:.0f}')}); R- {n('h117_rm')} (p = {p('h117_rm_p')}).",
               f"Cohorts do not differ (p = {pp('r1_mw', 'r1_welch')}).",
               f"Pseudo-populations (equal neurons and trials): R+ {n('pp_rp')} (p = {p('pp_rp_p')}), R- {n('pp_rm')} (p = {p('pp_rm_p')})."],
              "Decodability of the choice does not change between session halves, in either cohort.")
    img_slide(pdf, "A3. Equal numbers of hits before and after: hit-median split",
              "The split falls at the middle hit, so a hit-rate change cannot drive the comparison",
              PUB / "144_hitmedian_split_all.png",
              [f"Hit / miss decoding: no change in any window (smallest p = {p('o144_hm_pmin')}).",
               f"Pre-lick modality decoding falls in R+ ({n('k8_a', '{:.3f}')} -> {n('k8_b', '{:.3f}')}, p = {pp('k8_pw', 'k8_pt')}), more than in R- "
               f"(p = {pp('k8_c', 'k8_cw')})."],
              "Still no change in choice decoding; only the R+ pre-lick modality signal declines.")
    img_slide(pdf, "A4. Split at the learning trial, against placebo splits",
              "Change at the behavioural learning trial vs the changes at every other split of the same session",
              PUB / "138_cosyne_lt_placebo_all.png",
              [f"R+: change at the learning trial {n('o138_all_rp_real')} vs placebo {n('o138_all_rp_pl')} (p = {pp('o138_all_rp_pw', 'o138_all_rp_pt')}).",
               f"R+ vs R- beyond placebo: p = {pp('o138_all_exc_mw', 'o138_all_exc_w')}, cohort-label permutation p = {p('o138_all_exc_perm')}.",
               f"Learners only: R+ p = {pp('o138_learners_rp_pw', 'o138_learners_rp_pt')}."],
              "A small R+-specific gain appears when the split is tied to behaviour, not at an arbitrary split.")
    img_slide(pdf, "A5. Is the learning-trial effect robust? Windows and kind of learner",
              "Change point of the whisker hit sequence (L5), three windows; step vs gradual learners",
              PUB / "143_lt_split_L5_windows_all.png",
              [f"R+ beyond placebo at the change point: 5-35 ms p = {pp('o143_5_35ms_rp_pw', 'o143_5_35ms_rp_pt')}, 5-50 ms p = "
               f"{pp('o143_5_50ms_rp_pw', 'o143_5_50ms_rp_pt')}, 5-100 ms p = {pp('o143_5_100ms_rp_pw', 'o143_5_100ms_rp_pt')}; R- none.",
               f"R+ vs R- (5-100 ms) p = {pp('o143_5_100ms_ex_mw', 'o143_5_100ms_ex_w')}, permutation p = {p('o143_5_100ms_ex_perm')}.",
               f"Carried by step learners (step vs gradual, 5-50 ms: p = {pp('o143sg_5_50ms_Rp_mw', 'o143sg_5_50ms_Rp_w')})."],
              "The R+ gain holds across windows but belongs to abrupt (step) learners; gradual learners show no change.")
    fig = slide("Part A in one slide", "Active-only decoding of the choice within the learning session")
    bullets(fig, 0.06, 0.78, ["Choice is decodable early after the whisker stimulus, equally in both cohorts.",
                              "Over the session it does not change on average: halves, pseudo-populations, hit-median split.",
                              "It increases in R+ only at the behavioural change point, and only in mice that learn abruptly.",
                              "So decodability is mostly stable. Does the relation between the sensory response and the choice change "
                              "instead? Part B uses passive stimuli before and after the task to ask this."],
            size=16, dy=0.13, width=110)
    save(pdf, fig, 0)


# ------------------------------------------------------------------------------------------------------------------- part B
def s_b_tools(pdf):
    fig = slide("B1. One pattern, one axis: cosine and projection", "Everything in part B compares a passive response pattern with an "
                "active hit / miss axis, before vs after the task")
    ax = fig.add_axes([0.04, 0.12, 0.4, 0.68]); M.neural_space(ax)
    M.arrow(ax, (0, 0), (1.05, 0), "0.15"); ax.text(0.85, -0.07, "lick axis L = mean(hits) - mean(misses)\n(active whisker trials)", fontsize=11, ha="center", va="top")
    pw_ = (0.6, 0.7); M.arrow(ax, (0, 0), pw_, WC, label="passive whisker\npattern p", lpos=1.2, fs=11)
    ax.plot([pw_[0], pw_[0]], [0, pw_[1]], color="0.5", lw=1, ls=":"); ax.plot([0, pw_[0]], [-0.03, -0.03], color="0.3", lw=4)
    ax.text(0.17, 0.08, "θ", fontsize=16)
    bullets(fig, 0.5, 0.8, ["Pattern p: mean evoked 5-35 ms response to a passive stimulus (one value per unit, z units).",
                            "cos θ = direction of p relative to the lick axis (size ignored).",
                            "projection = |p| cos θ / √n: how far p reaches along the axis (linear in p).",
                            "Prediction (H3): after the task, the R- whisker pattern points less along the lick axis; R+ does not; the "
                            "auditory pattern (rewarded in both) does not change."],
            size=13, dy=0.13, width=62)
    save(pdf, fig, 0)


def s_b_raw(pdf, D):
    fig = slide("B2. Lick-axis alignment across the session (135)", "Raw cosine of the passive / active evoked patterns with the lick axis; "
                "the same tracked stable units in every epoch")
    d = D[(D.area == "All units") & D.skipped_reason.isna()]
    ep = ["passive_pre", "active_1", "active_2", "passive_post"]
    for j, (s, t) in enumerate((("W", "whisker-evoked pattern"), ("A", "auditory-evoked pattern (control)"))):
        a = fig.add_axes([0.07 + j * 0.47, 0.34, 0.38, 0.44])
        M.change_panel(a, d, [f"evoked{s}_cos_{e}" for e in ep], M.EPL4, t, "cosine with the lick axis", legend="upper right" if j == 0 else None)
    fig.text(0.07, 0.04, "Active halves are descriptive: an active-half pattern averages hits and misses, so it moves along the axis whenever "
             "the hit rate changes. Inference uses passive pre -> post.", fontsize=11, color="0.3")
    save(pdf, fig, 0)


def s_b_time(pdf, D):
    fig = slide("B3. The result beyond session time (135, linear-shift null)", "Passive pre -> post change minus the change produced by "
                "lick axes rebuilt from time-shifted labels")
    d = D[(D.area == "All units") & D.skipped_reason.isna()]
    for j, (col, t) in enumerate((("shift_excess_dWR", "whisker-evoked\nraw cosine"), ("shift_excess_dAR", "auditory-evoked\nraw cosine (control)"),
                                  ("shift_excess_dWAP", "whisker - auditory\nprojection"))):
        M.excess_panel(fig.add_axes([0.08 + j * 0.3, 0.36, 0.18, 0.44]), d, col, t, "excess post - pre\n(real - shift null)" if j == 0 else "")
    takehome(fig, "Beyond session time, the R- whisker pattern moves away from the lick axis after the task; R+ does not; the auditory "
             "pattern does not differ; whisker relative to auditory (a linear contrast in which shared drift cancels) differs.",
             x=0.08, y=0.04, w=0.84, h=0.14)
    save(pdf, fig, 0)


def s_b_robust(pdf, D):
    fig = slide("B4. Robust to the population and to the noise metric (135)", "Learners only; noise-corrected cosine as a sensitivity check")
    d = D[(D.area == "All units") & D.skipped_reason.isna()]
    L = d[H.mouse_of(d).isin(H.learners())]
    for j, (dd, col, t) in enumerate(((L, "shift_excess_dWR", "learners: whisker\nraw cosine"), (L, "shift_excess_dWAP", "learners: whisker -\nauditory projection"),
                                      (d, "shift_excess_dWN", "all: whisker noise-\ncorrected cosine"))):
        M.excess_panel(fig.add_axes([0.08 + j * 0.3, 0.36, 0.18, 0.44]), dd, col, t, "excess post - pre\n(real - shift null)" if j == 0 else "")
    takehome(fig, "The effect holds in learners only and with the noise-corrected cosine (whose null is noisier, because most shifted axes "
             "are unreliable).", x=0.08, y=0.04, w=0.84, h=0.12)
    save(pdf, fig, 0)


def s_b_area(pdf):
    img_slide(pdf, "B5. Where in the brain? Area groups (151)", "Each area with its own axes; >= 3 sessions per cohort, >= 20 units",
              PUB / "151_area_heatmap_all.png",
              ["Read the last two columns (lick axis beyond session time) and the R- minus R+ panel.",
               "Cohort differences (both tests p < 0.05) in somatosensory-whisker cortex, motor areas, midbrain and thalamus.",
               "Most areas point in the R- direction; single cells are small and uncorrected."],
              "The decoupling is distributed but strongest in whisker sensorimotor areas, midbrain and thalamus.", img_w=0.66)


# ------------------------------------------------------------------------------------------------------------------- part C
def s_c_cd(pdf, D140):
    fig = slide("C1. Method 2: the late hit / miss coding direction (140)", "CD = mean(hits) - mean(misses) in the second half of a hit-median "
                "split: the direction at the end of the task")
    d = D140[(D140.split == "hitmedian") & D140.skipped_reason.isna()]
    for j, (col, t) in enumerate((("shift_excess_daxisR", "passive whisker - auditory\naxis, raw cosine"), ("shift_excess_dWAP", "whisker - auditory\nprojection"),
                                  ("shift_excess_dAR", "auditory-evoked\nraw cosine (control)"))):
        M.excess_panel(fig.add_axes([0.08 + j * 0.3, 0.36, 0.18, 0.44]), d, col, t, "excess post - pre\n(real - shift null)" if j == 0 else "")
    takehome(fig, "Same answer with a different axis: beyond session time, the R- passive whisker response turns away from the late coding "
             "direction, relative to auditory.", x=0.08, y=0.04, w=0.84, h=0.12)
    save(pdf, fig, 0)


def s_c_dec(pdf, D146):
    fig = slide("C2. Method 3: a choice decoder read out on passive trials (146)", "Train hit vs miss on active trials; score passive trials; "
                "readout in SD units (+ hit-like, - miss-like)")
    v = D146[(D146.area == "whole_brain") & (D146.unit_set == "stable") & D146.skipped_reason.isna()]
    for j, (resp, col, t) in enumerate((("epochbase", "shift_excess_dW", "whisker readout"), ("trialbase", "shift_excess_dWA", "whisker - auditory\n(per-trial baseline)"),
                                        ("baseline", "shift_excess_dW", "baseline window alone\n(state), whisker"))):
        M.excess_panel(fig.add_axes([0.08 + j * 0.3, 0.36, 0.18, 0.44]), v[v.response == resp], col, t,
                       "excess post - pre\n(real - shift null)" if j == 0 else "")
    takehome(fig, "The decoder agrees only relative to auditory: its whisker readout alone no longer separates the cohorts beyond session "
             "time, and part of the R- change is a pre-stimulus state shift common to both stimuli.", x=0.08, y=0.04, w=0.84, h=0.14)
    save(pdf, fig, 0)


def s_c_ss(pdf):
    img_slide(pdf, "C3. Method 4: the state space - where do the passive responses move? (151)",
              "x = choice axis; y = passive-pre whisker, whisker - auditory or auditory pattern; passive pre at the origin",
              PUB / "151_state_space_centered_all.png",
              ["Both responses shrink along their own pre-task pattern, equally in both cohorts (shared; adaptation / state).",
               "Only the R- whisker response slides along the choice axis, to the level of the active misses (dashed lines).",
               "The cohort difference lives on the choice axis for every y axis."],
              "Geometry of the effect: after the task, an R- whisker stimulus evokes a pattern like a whisker trial without a lick.",
              img_w=0.5)


def s_c_table(pdf, D135, D140, D146, S):
    fig = slide("C4. The result across methods", "Passive pre -> post, excess over the linear-shift null unless noted (R+ vs R-: Mann-Whitney | Welch)")
    rows = []

    def add(name, d, col, unit=""):
        x = {c: d[d.reward_group == c][col].to_numpy(float) for c in COH}
        mw, we = H.unpaired(x["R+"], x["R-"])
        rows.append([name, f"{np.nanmean(x['R+']):+.3f}", f"{np.nanmean(x['R-']):+.3f}", M.pfmt(mw, we)])
    a = D135[(D135.area == "All units") & D135.skipped_reason.isna()]
    c = D140[(D140.split == "hitmedian") & D140.skipped_reason.isna()]
    v = D146[(D146.area == "whole_brain") & (D146.unit_set == "stable") & D146.skipped_reason.isna()]
    add("lick axis (135): whisker raw cosine", a, "shift_excess_dWR")
    add("lick axis (135): whisker - auditory projection", a, "shift_excess_dWAP")
    add("late coding direction (140): passive axis raw cosine", c, "shift_excess_daxisR")
    add("late coding direction (140): whisker - auditory projection", c, "shift_excess_dWAP")
    add("decoder (146): whisker readout", v[v.response == "epochbase"], "shift_excess_dW")
    add("decoder (146): whisker - auditory, per-trial baseline", v[v.response == "trialbase"], "shift_excess_dWA")
    if S is not None:
        for m_, lab in (("dx_W", "state space (raw): whisker along choice axis"), ("dx_WA", "state space (raw): whisker - auditory along choice axis")):
            r = S[(S.scope == "all") & (S.panel == "whole-brain state space") & (S.measure == m_)]
            rp, rm, rc = (r[r.cohort == k].iloc[0] for k in ("R+", "R-", "R- minus R+"))
            rows.append([lab, f"{rp['mean']:+.2f}", f"{rm['mean']:+.2f}", M.pfmt(rc.p_nonparam, rc.p_param)])
    tab = fig.add_axes([0.05, 0.12, 0.9, 0.7]); tab.axis("off")
    t = tab.table(cellText=rows, colLabels=["method: measure", "R+", "R-", "R+ vs R-"], loc="upper left", colWidths=[0.58, 0.12, 0.12, 0.18], cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 2.0)
    for (i, j), c_ in t.get_celld().items():
        c_.set_edgecolor("0.85")
        if i == 0:
            c_.set_facecolor("0.93"); c_.set_text_props(weight="bold")
    fig.text(0.05, 0.06, "Mean-difference axes (lick axis, coding direction) and the state space agree; the decoder agrees relative to auditory.",
             fontsize=13, weight="bold")
    save(pdf, fig, 0)


def s_synth(pdf):
    fig = slide("Synthesis", "From decodability to geometry")
    bullets(fig, 0.05, 0.8, ["A. Choice is decodable early in both cohorts; its decodability is stable within the session, except an R+ "
                             "gain at abrupt behavioural transitions.",
                             "B. With one method (lick-axis alignment), after the task the R- passive whisker response moves away from "
                             "the active lick axis, beyond session time, relative to the auditory response, in learners too, strongest in "
                             "whisker sensorimotor areas, midbrain and thalamus.",
                             "C. The same holds for the late coding direction and in the state space (the R- whisker response slides to "
                             "the miss level); the choice decoder shows it relative to auditory; R- also shows a pre-stimulus state shift.",
                             "Interpretation: in R-, where licking to the whisker is not rewarded, the earliest whisker response is "
                             "re-mapped away from the motor (lick) representation; decodability alone misses this."],
            size=14, dy=0.16, width=125)
    save(pdf, fig, 0)


def s_caveats(pdf):
    fig = slide("Caveats and next steps")
    bullets(fig, 0.05, 0.8, ["Session time: hit / miss labels drift; all part B / C inference is on the excess over a linear-shift null "
                             "(conservative).",
                             "State and exposure: R- shows a pre-stimulus state shift; R- mice receive more whisker stimuli and fewer "
                             "rewards (by design); stimulus-specific adaptation is not covered by the null or the auditory contrast.",
                             "Noise: 5-35 ms hit / miss axes are unreliable, so raw cosines and projections carry the tests.",
                             "Part A analyses 117 / 123b predate the invalid-trial exclusion; learning-trial definitions are not locked.",
                             "Exploratory, uncorrected. Next: exposure covariate, time-matched decoder, held-out mice / expert sessions."],
            size=14, dy=0.12, width=125)
    save(pdf, fig, 0)


def main():
    M.setup()
    D135 = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    D140 = pd.read_parquet(EA / "140_coding_direction_noise.parquet")
    D146 = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    S147 = pd.read_csv(EA / "147_stats.csv")
    S151 = pd.read_csv(EA / "151_stats.csv") if (EA / "151_stats.csv").exists() else None
    M.SLIDES.mkdir(parents=True, exist_ok=True)
    for f in M.SLIDES.glob("148_slide_*.png"):
        f.unlink()
    with PdfPages(OUT_PDF) as pdf:
        s_title(pdf); s_task(pdf)
        section(pdf, "A. Active-only: does choice decoding change within the session?", "One method: hit / miss decoding against a linear-shift null")
        s_a_method(pdf); s_part_a(pdf)
        section(pdf, "B. Passive vs active: one method, made robust", "Lick-axis alignment of the early (5-35 ms) passive responses, before vs after the task")
        s_b_tools(pdf); M.s_data(pdf, 0, D146); s_b_raw(pdf, D135); M.s_null(pdf, 0); s_b_time(pdf, D135); s_b_robust(pdf, D135); s_b_area(pdf)
        section(pdf, "C. Does it hold across methods?", "Late coding direction, choice decoder, state space; controls; synthesis")
        s_c_cd(pdf, D140); s_c_dec(pdf, D146); s_c_ss(pdf); s_c_table(pdf, D135, D140, D146, S151); M.s_step6(pdf, 0, D146, S147)
        s_synth(pdf); s_caveats(pdf)
        for p_, t in ((FIG / "126_lt_variants_modality_step1_pl100.png", "A: real vs placebo, all learning-trial definitions (126)"),
                      (FIG / "131_cohort_permutation_excess.png", "A: cohort-label permutation (131)"),
                      (PUB / "143_step_vs_gradual_all.png", "A: step vs gradual learners (143)"),
                      (FIG / "127_margin_vs_learning_curve.png", "A: single-trial margins vs learning curve (127)"),
                      (FIG / "133_whole_brain_tracked.png", "B: whisker vs auditory identity across epochs (133)"),
                      (FIG / "135b_alignment_no_bad_rplus_tracked.png", "B: lick-axis alignment, all panels (135b)"),
                      (PUB / "150_axis_alignment_shift_null_all.png", "B / C: lick axis and coding direction vs the shift null (150)"),
                      (PUB / "149_passive_readout_shift_null_all.png", "C: decoder readout vs the shift null (149)"),
                      (PUB / "151_state_space_variants_all.png", "C: state space, three y axes (151)"),
                      (PUB / "151_state_space_heatmap_all.png", "C: state-space displacements (151)"),
                      (PUB / "145_coding_direction_all.png", "C: coding direction, rotation and noise (145)"),
                      (PUB / "147_choice_axis_readout_stable.png", "C: choice readout, all panels (147)"),
                      (PUB / "147_controls.png", "C: state and engagement controls (147)")):
            M.s_backup(pdf, 0, p_, t)
    for f in sorted(M.SLIDES.glob("148_slide_*.png")):
        f.rename(f.with_name(f.name.replace("148_slide_", "152_slide_")))
    print(OUT_PDF, M._K[0], "slides")


if __name__ == "__main__":
    main()
