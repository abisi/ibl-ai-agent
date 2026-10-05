"""148 -- Part III digest: a step-by-step slide deck (16:9 PDF + one PNG per slide) of the stimulus-onset geometry analyses
(user 2026-10-05: "a digest of Part III ... clear on the methods with schematics and additional figures, per analysis, that
help me present that to an audience step by step").
Slides: title / roadmap; question and prediction (schematic); session structure and analysis window (schematic); vector
toolbox (schematic); then per analysis a method schematic + the key result re-plotted from the per-session tables:
  1 (133) is the stimulus identity kept across epochs?   2 (134) whisker-specific gain and lick-axis alignment in the active
  epoch   3 (135) alignment of the whisker axis / evoked patterns with the lick axis across the four epochs   4 (140) the
  hit / miss coding direction across halves and the passive whisker axis   5 (146) the choice decoder read out on passive trials
  6 (146 controls) state, timing and behaviour; synthesis (schematic + summary table); caveats; backup slides (full figures).
Numbers on slides come from the per-session tables and the stats files (no typed numbers). Session = unit; tests: within cohort
post - pre Wilcoxon | paired t, R+ vs R- on the change Mann-Whitney | Welch; uncorrected.
Output: combined_results_ks4/<slug>/report/part3_digest.pdf, figures/digest/148_slide_XX.png
Run (haas, repo root): python .../148_part3_digest.py
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA)); sys.path.insert(0, str(EA.parents[2] / "scripts"))
H = importlib.import_module("143_lt_split_windows_figures")
from axel_bisi_paths import axel_bisi_root  # noqa: E402

COL, COH = H.COL, H.COH
WC, AC = "#f7b519", "#2c2cdb"
FIG, PUB = EA / "figures", EA / "figures" / "publication"
OUT_PDF = axel_bisi_root() / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "report" / "part3_digest.pdf"
SLIDES = FIG / "digest"
W_IN, H_IN = 13.33, 7.5
EPL4 = ["passive\npre", "active\n1st half", "active\n2nd half", "passive\npost"]


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 13, "axes.labelsize": 12,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})


def slide(title, subtitle=None):
    fig = plt.figure(figsize=(W_IN, H_IN))
    fig.text(0.04, 0.93, title, fontsize=22, weight="bold", va="bottom")
    if subtitle:
        fig.text(0.04, 0.885, subtitle, fontsize=13, color="0.35", va="bottom")
    fig.add_artist(plt.Line2D([0.04, 0.96], [0.875, 0.875], color="0.8", lw=1))
    return fig


def bullets(fig, x, y, items, size=13, dy=0.062, width=None):
    for i, t in enumerate(items):
        t = textwrap.fill(t, width, subsequent_indent="    ") if width else t
        fig.text(x, y - i * dy, "•  " + t, fontsize=size, va="top", wrap=width is None)


def takehome(fig, text, x=0.03, y=0.04, w=0.31, h=0.19):
    """take-home message box in the bottom-left corner of an analysis slide"""
    fig.add_artist(Rectangle((x, y), w, h, transform=fig.transFigure, facecolor="#fff6dc", edgecolor=WC, lw=1.2))
    fig.text(x + 0.01, y + h - 0.012, "Take-home", fontsize=11, weight="bold", va="top")
    fig.text(x + 0.01, y + h - 0.045, textwrap.fill(text, 56), fontsize=10.5, va="top")


def arrow(ax, p0, p1, color, lw=2.5, ls="-", label=None, lpos=1.05, fs=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=18, color=color, lw=lw, linestyle=ls))
    if label:
        ax.text(p0[0] + lpos * (p1[0] - p0[0]), p0[1] + lpos * (p1[1] - p0[1]), label, color=color, fontsize=fs, ha="center", va="center")


def pfmt(a, b):
    return f"{H.pnum(a)} | {H.pnum(b)}"


def change_panel(ax, d, cols, labels, title, ylab, ref0=True, ls_cols=None, legend="best"):
    """mean +- s.e.m. per cohort across columns (epochs); stats on last - first"""
    x = np.arange(len(cols)); out = {}
    for c in COH:
        g = d[d.reward_group == c]; M = g[cols].to_numpy(float)
        ax.errorbar(x + (0.04 if c == "R-" else -0.04), np.nanmean(M, 0), [H.sem(M[:, j]) for j in range(len(cols))], color=COL[c],
                    lw=2.4, marker="o", ms=7, capsize=0, label=f"{c} (n = {len(g)})")
        out[c] = M[:, -1] - M[:, 0]
        if ls_cols is not None:
            M2 = g[ls_cols].to_numpy(float)
            ax.errorbar(x + (0.04 if c == "R-" else -0.04), np.nanmean(M2, 0), [H.sem(M2[:, j]) for j in range(len(ls_cols))], color=COL[c],
                        lw=1.6, ls="--", marker="o", ms=5, mfc="white", capsize=0)
    mw, we = H.unpaired(out["R+"], out["R-"])
    win = {c: H.paired(d[d.reward_group == c][cols[-1]].to_numpy(float), d[d.reward_group == c][cols[0]].to_numpy(float)) for c in COH}
    if ref0:
        ax.axhline(0, color="0.6", lw=1, ls=(0, (3, 3)))
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=13)
    if legend:
        ax.legend(frameon=False, fontsize=10.5, loc=legend)
    lab = lambda s: s.replace("\n", " ")
    txt = (f"{lab(labels[-1])} vs {lab(labels[0])}\np (Wilcoxon | t):\nR+ {pfmt(win['R+'][0], win['R+'][1])}\nR- {pfmt(win['R-'][0], win['R-'][1])}\n"
           f"change R+ vs R- (MW | Welch):\n{pfmt(mw, we)}")
    ax.text(0, -0.2, txt, transform=ax.transAxes, fontsize=9.5, color="0.25", va="top")
    return dict(within=win, cohort=(mw, we), change={c: np.nanmean(v) for c, v in out.items()})


def excess_panel(ax, d, col, title, ylab="excess change\n(real - shift null)"):
    """per-session excess of the passive post - pre change over the linear-shift null, mean +- s.e.m. and dots per cohort"""
    v = {}
    for i, c in enumerate(COH):
        v[c] = d[d.reward_group == c][col].to_numpy(float); v[c] = v[c][np.isfinite(v[c])]
        x = i + np.random.default_rng(i).uniform(-0.13, 0.13, len(v[c]))
        ax.plot(x, v[c], "o", ms=3.5, color=COL[c], alpha=0.35, mew=0)
        ax.errorbar(i, np.mean(v[c]), H.sem(v[c]), fmt="o", ms=8, color=COL[c], lw=2.4, capsize=0, zorder=5)
    ax.axhline(0, color="0.6", lw=1, ls=(0, (3, 3)))
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"R+\n{len(v['R+'])}", f"R-\n{len(v['R-'])}"]); ax.set_xlim(-0.6, 1.6)
    ax.set_title(title, fontsize=11.5); ax.set_ylabel(ylab, fontsize=10.5)
    o = {c: H.one_sample(v[c]) for c in COH}; mw, we = H.unpaired(v["R+"], v["R-"])
    ax.text(0, -0.2, f"vs 0 (Wilcoxon | t):\nR+ {pfmt(o['R+'][0], o['R+'][1])}\nR- {pfmt(o['R-'][0], o['R-'][1])}\nR+ vs R- (MW | Welch):\n{pfmt(mw, we)}",
            transform=ax.transAxes, fontsize=9.5, color="0.25", va="top")


def neural_space(ax, title=""):
    ax.set_xlim(-0.2, 1.25); ax.set_ylim(-0.2, 1.15); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13)


# ---------------------------------------------------------------------------------------------------------------- slides
def s_title(pdf, k):
    fig = plt.figure(figsize=(W_IN, H_IN))
    fig.text(0.06, 0.72, "Part III: does the early whisker response stay\nlinked to the choice readout?", fontsize=30, weight="bold", va="center")
    fig.text(0.06, 0.55, "Stimulus-onset (5-35 ms) population geometry across passive pre, active and passive post epochs,\n"
             "learning session, R+ vs R- cohorts (ssl-whisker-hitmiss-timeresolved-decoding)", fontsize=15, color="0.3", va="center")
    steps = ["Question, data and tools", "Null: session time (linear shift)", "1  Is stimulus identity kept? (133)",
             "2  Whisker-specific gain (134)", "3  Alignment with the lick axis (135)", "4  Coding direction across halves (140)",
             "5  Choice decoder on passive trials (146)", "6  Controls: state, timing, behaviour", "Synthesis and caveats"]
    for i, s in enumerate(steps):
        fig.text(0.06 + (i // 5) * 0.45, 0.38 - (i % 5) * 0.055, s, fontsize=15)
    save(pdf, fig, k)


def s_question(pdf, k):
    fig = slide("The question", "In R+ a lick after the whisker is rewarded, in R- it is not. Does the early sensory response re-map relative to the choice?")
    for i, (c, ang_post, lab) in enumerate((("R+", 35, "R+: stays aligned"), ("R-", 95, "R-: turns away"))):
        ax = fig.add_axes([0.05 + i * 0.3, 0.12, 0.27, 0.68]); neural_space(ax, lab)
        o = (0, 0)
        arrow(ax, o, (1.05, 0), "0.15"); ax.text(0.85, -0.1, "choice axis (hit - miss)", fontsize=11, ha="center", va="top")
        a0 = np.deg2rad(32)
        arrow(ax, o, (0.85 * np.cos(a0), 0.85 * np.sin(a0)), WC, label="whisker response\nbefore the task", lpos=1.25, fs=11)
        a1 = np.deg2rad(ang_post if c == "R-" else 20)
        arrow(ax, o, (0.75 * np.cos(a1), 0.75 * np.sin(a1)), COL[c], ls="--", label="after the task", lpos=1.3 if c == "R+" else 1.15, fs=11)
        ax.text(0.5, -0.12, "neural population space (schematic)", fontsize=10, color="0.45", ha="center", transform=ax.transAxes)
    bullets(fig, 0.66, 0.76, ["Choice axis: the population direction that separates active whisker hits (lick) from misses.",
                              "Hypothesis: in R-, where licking to the whisker is not rewarded, the early whisker response turns away from that "
                              "axis during and after the task; in R+ it stays aligned.",
                              "Controls: the auditory response (always rewarded in both cohorts) should not change; a shared change of "
                              "state should affect both stimuli.",
                              "Measured at 5-35 ms after stimulus onset: before any lick (trials with a lick before 35 ms excluded)."],
            size=13, dy=0.15)
    save(pdf, fig, k)


def s_data(pdf, k, D146):
    fig = slide("Data and analysis window", "Learning session (whisker day 0), one per mouse; passive stimuli before and after the active task")
    ax = fig.add_axes([0.05, 0.5, 0.9, 0.3]); ax.set_xlim(0, 100); ax.set_ylim(0, 3); ax.axis("off")
    for x0, x1, lab, colr in ((2, 18, "passive pre\n(whisker + auditory,\nno reward)", "0.85"), (20, 50, "active 1st half", "#cfe8cf"),
                              (50, 80, "active 2nd half", "#bfe0bf"), (82, 98, "passive post\n(whisker + auditory,\nno reward)", "0.85")):
        ax.add_patch(Rectangle((x0, 1), x1 - x0, 1, color=colr, lw=0))
        ax.text((x0 + x1) / 2, 1.5, lab, ha="center", va="center", fontsize=12)
    ax.annotate("", xy=(98, 0.6), xytext=(2, 0.6), arrowprops=dict(arrowstyle="->", color="0.3", lw=1.5))
    ax.text(50, 0.25, "session time (active: whisker, auditory and no-stim trials; R+ rewarded for whisker licks, both for auditory licks)",
            ha="center", fontsize=11, color="0.3")
    ax2 = fig.add_axes([0.07, 0.1, 0.45, 0.3]); ax2.set_xlim(-80, 60); ax2.set_ylim(0, 3); ax2.axis("off")
    ax2.add_patch(Rectangle((-55, 1), 35, 1, color="0.8", lw=0)); ax2.text(-37.5, 2.3, "baseline\n-55 to -20 ms", ha="center", fontsize=11)
    ax2.add_patch(Rectangle((-10, 1), 15, 1, color="#f4c6c6", lw=0)); ax2.text(-2.5, 0.55, "artefact dead zone", ha="center", fontsize=10, color="#b04040")
    ax2.add_patch(Rectangle((5, 1), 30, 1, color=WC, lw=0, alpha=0.8)); ax2.text(20, 2.3, "response\n5-35 ms", ha="center", fontsize=11)
    ax2.axvline(0, ymin=0.25, ymax=0.75, color="k", lw=2); ax2.text(0, 0.1, "stimulus", ha="center", fontsize=11)
    ax2.annotate("", xy=(58, 0.7), xytext=(-78, 0.7), arrowprops=dict(arrowstyle="->", color="0.3"))
    d = D146[(D146.area == "whole_brain") & (D146.response == "epochbase") & (D146.unit_set == "stable") & D146.skipped_reason.isna()]
    nu = d.groupby("reward_group").n_units.median()
    bullets(fig, 0.56, 0.4, [f"Sessions: R+ {int((d.reward_group == 'R+').sum())}, R- {int((d.reward_group == 'R-').sum())} (passive before and "
                             "after the task, >= 3 active whisker hits and misses)",
                             f"Units: the same tracked stable units in every step (coverage, presence, drift test; >= 0.5 Hz in pre, post and both active halves); median "
                             f"{int(nu.get('R+', 0))} (R+) / {int(nu.get('R-', 0))} (R-) per session",
                             "Response = rate 5-35 ms minus the unit's baseline in that epoch, z-scored per unit",
                             "Trials: invalid trials removed, warm-up block cut, disengaged tail trimmed (rule A1)",
                             "Inference: passive pre -> post change, as excess over a linear-shift null; active epoch defines the axes"],
            size=12, dy=0.07)
    save(pdf, fig, k)


def s_tools(pdf, k):
    fig = slide("Toolbox: comparing population vectors", "All measures compare a response pattern (one value per unit) with an axis, across epochs of the same session")
    ax = fig.add_axes([0.03, 0.12, 0.3, 0.68]); neural_space(ax, "size, angle, projection")
    arrow(ax, (0, 0), (1.05, 0), "0.15")
    ax.text(1.08, 0.06, "axis u\n(unit length)", fontsize=11, ha="center")
    p = (0.6, 0.7); arrow(ax, (0, 0), p, WC, label="pattern p", lpos=1.12, fs=11)
    ax.plot([p[0], p[0]], [0, p[1]], color="0.5", lw=1, ls=":"); ax.plot([0, p[0]], [-0.05, -0.05], color="0.3", lw=4)
    ax.text(0.3, -0.16, "projection = size x cos θ", color="0.3", fontsize=11, ha="center")
    ax.text(0.17, 0.08, "θ", fontsize=16)
    ax.text(-0.15, -0.38, "size = |p| / sqrt(n units);  cos θ: direction only;\nprojection: how far p reaches along u", fontsize=10.5, color="0.3")
    ax2 = fig.add_axes([0.36, 0.12, 0.28, 0.68]); neural_space(ax2, "noise-corrected cosine\n(two estimates per vector, disjoint trials)")
    for dx, dy in ((0.04, 0.05), (-0.05, 0.02)):
        arrow(ax2, (0, 0), (1.0 + dx, 0.1 + dy), "0.4", lw=1.6)
        arrow(ax2, (0, 0), (0.55 + dx, 0.75 + dy), WC, lw=1.6)
    ax2.text(1.1, 0.0, "u_A, u_B", color="0.4", fontsize=11); ax2.text(0.45, 0.9, "p_A, p_B", color=WC, fontsize=11)
    ax2.text(-0.15, -0.38, "cos_norm = cos(p_A, u_B) / sqrt(rel_p x rel_u)\nrel = cos(p_A, p_B), cos(u_A, u_B): corrects for noise",
             fontsize=10.5, color="0.3")
    ax3 = fig.add_axes([0.69, 0.3, 0.27, 0.45])
    xx = np.linspace(-4, 4, 300)
    for m, cc, lab in ((-1.2, "0.55", "active\nmisses"), (1.2, "0.1", "active\nhits")):
        ax3.fill_between(xx, np.exp(-(xx - m) ** 2 / 2), color=cc, alpha=0.35, lw=0, edgecolor="none"); ax3.text(m * 1.9, 0.75, lab, ha="center", fontsize=11)
    ax3.axvline(-0.4, color=WC, lw=3); ax3.text(-0.4, 1.12, "passive whisker trial", color=WC, ha="center", fontsize=11)
    ax3.set_yticks([]); ax3.set_xlabel("decoder score (SD units of active scores)\n- miss-like       + hit-like")
    ax3.set_title("choice-decoder readout", fontsize=13, pad=26)
    ax3.spines["left"].set_visible(False)
    save(pdf, fig, k)


def s_null(pdf, k):
    fig = slide("Session time is a confound: the linear-shift null",
                "Hit rate drifts during the session, so any axis built from hits vs misses can partly encode 'early vs late'")
    ax = fig.add_axes([0.04, 0.42, 0.55, 0.38]); ax.set_xlim(-2, 102); ax.set_ylim(-0.5, 4.3); ax.axis("off")
    rng = np.random.default_rng(3); n = 60
    p_hit = np.linspace(0.75, 0.15, n)                          # R-: hits early, misses late
    y = rng.random(n) < p_hit
    xs = np.linspace(0, 70, n)
    ax.text(-2, 3.75, "active whisker trials (R- example)", fontsize=11, color="0.3")
    ax.scatter(xs, np.full(n, 3.2), c=["0.15" if h else "0.75" for h in y], s=26, marker="s")
    ax.text(71.5, 3.2, "real labels: hit (dark) / miss (light)", va="center", fontsize=10.5)
    ax.scatter(xs, np.full(n, 2.3), c="#9ecae1", s=26, marker="s"); ax.text(71.5, 2.3, "neural trials", va="center", fontsize=10.5)
    k_ = 18
    ax.scatter(xs[: n - k_], np.full(n - k_, 1.1), c=["0.15" if h else "0.75" for h in y[k_:]], s=26, marker="s")
    ax.scatter(xs[: n - k_], np.full(n - k_, 0.3), c="#9ecae1", s=26, marker="s")
    ax.text(71.5, 0.7, "null: labels shifted by k trials\n(10-50 %), no wrap-around;\nhits still mostly early", va="center", fontsize=10.5)
    ax.annotate("", xy=(xs[0], 1.45), xytext=(xs[k_], 2.75), arrowprops=dict(arrowstyle="->", color=AC, lw=1.5))
    ax.text(xs[k_ // 2] - 2, 2.0, "shift", color=AC, fontsize=10.5)
    bullets(fig, 0.04, 0.33, ["An axis or decoder fitted to drifting labels can learn session time; passive post, later still, then reads "
                              "miss-like in R- (hits early) or hit-like in R+ (hits late), with no sensory change.",
                              "Null: rebuild the axis / decoder from labels shifted against the neural trials (50 shifts per session); it keeps the slow "
                              "drift of both series. Report the excess = real post - pre change minus the mean null change.",
                              "Every session with >= 3 hits and >= 3 misses gets a null (only shifts keeping enough hits are drawn). Conservative: "
                              "real learning is time-correlated too."], size=12, dy=0.085, width=150)
    ax2 = fig.add_axes([0.64, 0.42, 0.33, 0.38]); ax2.axis("off")
    ax2.text(0, 0.95, "Why not the alternatives?", fontsize=12, weight="bold", va="top")
    ax2.text(0, 0.8, textwrap.fill("Shuffled labels destroy trial order: a time-learning decoder is not in that null.", 52) + "\n\n" +
             textwrap.fill("A circular shift turns the trend into a sawtooth (hits land late in half the shifts): the null centres "
                           "near 0 and is too lenient.", 52) + "\n\n" +
             textwrap.fill("Whisker - auditory removes a shared drift only for linear measures (projections), not for cosines.", 52),
             fontsize=11, va="top")
    save(pdf, fig, k)


def s_step1(pdf, k, D133):
    fig = slide("Step 1 (133): is the stimulus identity kept across epochs?",
                "Decode whisker vs auditory trials from the 5-35 ms response in each epoch; transfer a decoder across epochs")
    ax = fig.add_axes([0.03, 0.3, 0.32, 0.5]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, lab in enumerate(("passive pre", "active", "passive post")):
        ax.add_patch(Rectangle((0.05 + i * 0.32, 0.55), 0.26, 0.2, color="0.88", lw=0)); ax.text(0.18 + i * 0.32, 0.65, lab, ha="center", fontsize=12)
        ax.text(0.18 + i * 0.32, 0.45, "W vs A\ndecoder", ha="center", fontsize=11, color=WC)
    ax.annotate("", xy=(0.82, 0.8), xytext=(0.18, 0.8), arrowprops=dict(arrowstyle="->", color=AC, connectionstyle="arc3,rad=-0.4"))
    ax.text(0.5, 0.95, "train on one epoch, test on another", ha="center", color=AC, fontsize=11)
    ax.text(0.0, 0.05, "accuracy minus trial-shuffle null; equal trial\ncounts per class and epoch; the shared tracked\n"
            "stable units (same units as steps 2-6)", fontsize=10.5, color="0.3")
    d = D133[(D133.area == "All units") & D133.skipped_reason.isna()]
    ax1 = fig.add_axes([0.42, 0.34, 0.25, 0.44])
    change_panel(ax1, d, ["within_passive_pre_corr", "within_active_corr", "within_passive_post_corr"], ["passive\npre", "active", "passive\npost"],
                 "whisker vs auditory decoding", "accuracy - null", legend="lower left")
    ax2 = fig.add_axes([0.75, 0.34, 0.2, 0.44])
    change_panel(ax2, d, ["cross_passive_pre__active_corr", "cross_passive_pre__passive_post_corr"], ["pre -> active", "pre -> post"],
                 "decoder trained on passive pre", "accuracy - null", legend=None)
    takehome(fig, "Whisker and auditory stay equally separable in every epoch and in both cohorts: the sensory code itself is not lost or rewritten.")
    save(pdf, fig, k)


def s_step2(pdf, k, D134):
    fig = slide("Step 2 (134): does the whisker response change specifically?",
                "Gain change of each unit in its preferred direction, whisker minus auditory (auditory = within-mouse control)")
    ax = fig.add_axes([0.04, 0.15, 0.3, 0.6]); ax.axis("off")
    bullets(fig, 0.04, 0.75, ["Each unit's preferred sign from half of the passive-pre trials",
                              "Change of its response (other half vs active / post) in that direction: gain change g",
                              "Specific change = g(whisker) - g(auditory): removes changes shared by both stimuli",
                              "Lick-axis alignment (active only): cos(whisker axis, licked - unlicked), noise-corrected",
                              "Shown for the 5-35 ms window (sliding windows in the backup figure)"], size=12, dy=0.1, width=46)
    takehome(fig, "Response gains change little and alike in both cohorts; what differs is the relation to licking: in the active epoch the "
             "whisker axis is weakly aligned with the lick axis in R+ and anti-aligned in R- (active epoch, descriptive).")
    d = D134[(D134.area == "All units") & (D134.win_start == 5) & D134.skipped_reason.isna()]
    for j, (col, lab) in enumerate((("spec_state", "whisker - auditory gain change,\nactive vs pre (z)"), ("spec_plast", "whisker - auditory gain change,\npost vs pre (z)"),
                                     ("lick_cosnorm", "whisker axis vs lick axis,\nactive (normalised cosine)"))):
        a = fig.add_axes([0.42 + j * 0.19, 0.34, 0.13, 0.44])
        v = {}
        for i, c in enumerate(COH):
            v[c] = d[d.reward_group == c][col].to_numpy(float)
            a.errorbar(i, np.nanmean(v[c]), H.sem(v[c]), fmt="o", ms=9, color=COL[c], lw=2.4, capsize=0)
        mw, we = H.unpaired(v["R+"], v["R-"])
        a.axhline(0, color="0.6", lw=1, ls=(0, (3, 3))); a.set_xticks([0, 1]); a.set_xticklabels(["R+", "R-"]); a.set_xlim(-0.6, 1.6)
        a.set_title(lab, fontsize=11.5); a.text(0.5, -0.12, f"R+ vs R- (MW | Welch)\n{pfmt(mw, we)}", ha="center", va="top",
                                         transform=a.transAxes, fontsize=10.5)
    save(pdf, fig, k)


def s_step3(pdf, k, D135):
    fig = slide("Step 3 (135): alignment with the lick axis across the four epochs",
                "Lick axis = active licked - unlicked whisker trials; compared with the whisker-evoked and auditory-evoked patterns per epoch")
    ax = fig.add_axes([0.02, 0.3, 0.28, 0.52]); neural_space(ax)
    arrow(ax, (0, 0), (1.05, 0), "0.15"); ax.text(0.85, -0.06, "lick axis (active)", fontsize=11, ha="center", va="top")
    for ang, lab, cc in ((30, "pre", WC), (12, "active 1", WC), (80, "active 2 / post\n(R-)", COL["R-"])):
        a = np.deg2rad(ang); arrow(ax, (0, 0), (0.8 * np.cos(a), 0.8 * np.sin(a)), cc, lw=2, label=lab, lpos=1.22, fs=10)
    ax.text(0.0, -0.05, "lick axis and evoked patterns from disjoint\ntrials (50 random splits), noise-corrected", fontsize=10.5, color="0.3", transform=ax.transAxes)
    d = D135[(D135.area == "All units") & D135.skipped_reason.isna()]
    ep = ["passive_pre", "active_1", "active_2", "passive_post"]
    a1 = fig.add_axes([0.35, 0.34, 0.25, 0.44])
    change_panel(a1, d, [f"evokedW_cosnorm_{e}" for e in ep], EPL4, "whisker-evoked vs lick axis\n(descriptive, noise-corrected)", "noise-corrected cosine")
    for x0, col, t in ((0.69, "shift_excess_dWR", "whisker-evoked\nraw cos"), (0.86, "shift_excess_dWAP", "whisker - auditory\nprojection")):
        excess_panel(fig.add_axes([x0, 0.34, 0.11, 0.44]), d, col, t, "excess post - pre\n(real - shift null)" if x0 < 0.8 else "")
    fig.text(0.69, 0.84, "passive pre -> post, beyond session time", fontsize=11.5, weight="bold")
    takehome(fig, "Beyond session time (linear-shift null), the passive whisker-evoked pattern moves away from the lick axis after the "
             "task in R-, also relative to auditory (linear contrast); R+ does not. Active halves are descriptive (trial composition).")
    save(pdf, fig, k)


def s_step4(pdf, k, D140):
    fig = slide("Step 4 (140): the hit / miss coding direction across session halves",
                "Coding direction CD = mean(hit) - mean(miss) per half of the active epoch (hit-median split: equal hits per half)")
    ax = fig.add_axes([0.02, 0.3, 0.28, 0.52]); neural_space(ax)
    arrow(ax, (0, 0), (1.0, 0.08), "0.15", label="CD half 1", lpos=1.15, fs=11)
    arrow(ax, (0, 0), (0.85, 0.5), "0.4", label="CD half 2", lpos=1.12, fs=11)
    arrow(ax, (0, 0), (0.3, 0.8), WC, label="passive whisker axis\n(whisker - auditory)", lpos=1.2, fs=10)
    ax.text(0.0, -0.05, "rotation = similarity of CD half 1 and half 2;\npassive axis compared with CD half 2,\npassive pre vs post", fontsize=10.5,
            color="0.3", transform=ax.transAxes)
    d = D140[(D140.split == "hitmedian") & D140.skipped_reason.isna()]
    a1 = fig.add_axes([0.37, 0.34, 0.12, 0.44]); v = {}
    for i, c in enumerate(COH):
        v[c] = d[d.reward_group == c].cosnorm_between.to_numpy(float)
        a1.errorbar(i, np.nanmean(v[c]), H.sem(v[c]), fmt="o", ms=9, color=COL[c], lw=2.4, capsize=0)
    a1.axhline(1, color="0.6", lw=1, ls=(0, (3, 3))); a1.set_xticks([0, 1]); a1.set_xticklabels(["R+", "R-"]); a1.set_xlim(-0.6, 1.6)
    a1.set_title("CD half 1 vs half 2\n(1 = same direction)", fontsize=11.5)
    a1.set_title("CD half 1 vs half 2\n(1 = same; descriptive)", fontsize=11.5)
    for x0, col, t in ((0.57, "shift_excess_daxisR", "passive whisker axis\nraw cos"), (0.72, "shift_excess_dWAP", "whisker - auditory\nprojection"),
                       (0.87, "shift_excess_dAR", "auditory-evoked\nraw cos (control)")):
        excess_panel(fig.add_axes([x0, 0.34, 0.1, 0.44]), d, col, t, "excess post - pre\n(real - shift null)" if x0 < 0.6 else "")
    fig.text(0.57, 0.84, "passive pre -> post vs CD half 2, beyond session time", fontsize=11.5, weight="bold")
    takehome(fig, "The hit / miss direction is partly re-drawn between halves. Beyond session time, the passive whisker axis turns away "
             "from the late coding direction after the task in R-, also relative to auditory; R+ does not, auditory alone does not.")
    save(pdf, fig, k)


def s_step5(pdf, k, D146):
    fig = slide("Step 5 (146): read out passive trials with the active choice decoder",
                "Train hit vs miss on active whisker trials; score passive whisker and auditory trials (never used for training)")
    ax = fig.add_axes([0.03, 0.3, 0.3, 0.52]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0.05, 0.62), 0.4, 0.22, color="#cfe8cf", lw=0)); ax.text(0.25, 0.73, "active whisker trials\nhits vs misses", ha="center", fontsize=11)
    ax.annotate("", xy=(0.62, 0.73), xytext=(0.46, 0.73), arrowprops=dict(arrowstyle="->", lw=2)); ax.text(0.78, 0.73, "decoder\n(cross-validated)", ha="center", fontsize=11)
    for i, lab in enumerate(("passive pre", "passive post")):
        ax.add_patch(Rectangle((0.05 + i * 0.5, 0.25), 0.4, 0.18, color="0.88", lw=0)); ax.text(0.25 + i * 0.5, 0.34, lab + "\nwhisker | auditory", ha="center", fontsize=10.5)
        ax.annotate("", xy=(0.25 + i * 0.5, 0.44), xytext=(0.78, 0.62), arrowprops=dict(arrowstyle="->", color="0.4"))
    ax.text(0.0, 0.05, "readout in SD units of the held-out active scores\n(+ hit-like, - miss-like); 20 balanced repetitions", fontsize=10.5, color="0.3")
    d = D146[(D146.area == "whole_brain") & (D146.response == "epochbase") & (D146.unit_set == "stable") & D146.skipped_reason.isna()].copy()
    ep = ["passive_pre", "active_1", "active_2", "passive_post"]
    a1 = fig.add_axes([0.37, 0.34, 0.25, 0.44])
    change_panel(a1, d, [f"ro_std_{e}_W" for e in ep], EPL4, "readout, whisker solid / auditory\ndashed (descriptive)", "readout (SD units)",
                 ls_cols=[f"ro_std_{e}_A" for e in ep])
    tb = D146[(D146.area == "whole_brain") & (D146.response == "trialbase") & (D146.unit_set == "stable") & D146.skipped_reason.isna()]
    for x0, dd, col, t in ((0.71, d, "shift_excess_dW", "whisker readout\n(main)"), (0.86, tb, "shift_excess_dWA", "whisker - auditory\nper-trial baseline")):
        excess_panel(fig.add_axes([x0, 0.34, 0.11, 0.44]), dd, col, t, "excess post - pre\n(real - shift null)" if x0 < 0.8 else "")
    fig.text(0.71, 0.84, "passive pre -> post, beyond session time", fontsize=11.5, weight="bold")
    takehome(fig, "Session time explains part of the readout shift. Beyond it, R- passive whisker trials still move miss-ward, but the "
             "cohort difference holds only for whisker relative to auditory (per-trial baseline), not for the whisker readout alone.")
    save(pdf, fig, k)


def s_step6(pdf, k, D146, S147):
    fig = slide("Step 6: controls for state, timing and behaviour",
                "A decoder trained on the pre-stimulus baseline tests state; whisker - auditory removes what both stimuli share")
    v = D146[(D146.area == "whole_brain") & (D146.unit_set == "stable") & D146.skipped_reason.isna()].copy()
    for e in ("passive_pre", "passive_post"):
        v[f"wa_{e}"] = v[f"ro_std_{e}_W"] - v[f"ro_std_{e}_A"]
    for j, (resp, lab) in enumerate((("baseline", "baseline window alone\n(-55 to -20 ms): state"), ("trialbase", "5-35 ms, per-trial baseline\nremoved"))):
        g = v[v.response == resp]
        a = fig.add_axes([0.06 + j * 0.24, 0.36, 0.16, 0.42])
        change_panel(a, g, ["ro_std_passive_pre_W", "ro_std_passive_post_W"], ["passive\npre", "passive\npost"], lab, "readout (SD units)",
                     ls_cols=["ro_std_passive_pre_A", "ro_std_passive_post_A"], legend="upper right" if j == 0 else None)
    e = v[v.response == "epochbase"]
    a = fig.add_axes([0.57, 0.36, 0.13, 0.42])
    excess_panel(a, v[v.response == "baseline"], "shift_excess_dW", "state readout, whisker\nbeyond session time", "excess post - pre\n(real - shift null)")
    a = fig.add_axes([0.79, 0.36, 0.18, 0.42])
    g = e[e.reward_group == "R-"].copy(); g["d_ro"] = g.ro_std_passive_post_W - g.ro_std_passive_pre_W; g["d_hit"] = g.hit_rate_2 - g.hit_rate_1
    a.plot(g.d_hit, g.d_ro, "o", color=COL["R-"], ms=6, alpha=0.75)
    from scipy import stats
    lr = stats.linregress(g.d_hit, g.d_ro); xx = np.linspace(g.d_hit.min(), g.d_hit.max(), 20)
    a.plot(xx, lr.intercept + lr.slope * xx, color=COL["R-"], lw=2)
    a.axhline(0, color="0.6", lw=1); a.set_xlabel("whisker hit rate, 2nd - 1st half"); a.set_ylabel("readout change, post - pre")
    a.set_title(f"R-: learning to withhold vs shift\nr = {lr.rvalue:+.2f}, p = {H.pnum(lr.pvalue)}", fontsize=11.5)
    c = S147[S147.scope == "all controls"] if "scope" in S147 else S147.iloc[:0]
    if len(c):
        r = lambda m: c[(c.measure == m) & (c.cohort == "R+ vs R-")].iloc[0]
        ad, rw = r("active_dur_min"), r("n_rewards_active")
        fig.text(0.04, 0.035, f"Timing and rewards: R- active epochs longer ({ad.mean_b:.0f} vs {ad.mean_a:.0f} min, p {pfmt(ad.p_nonparam, ad.p_param)}); "
                 f"R- collect fewer rewards ({rw.mean_b:.0f} vs {rw.mean_a:.0f}, p {pfmt(rw.p_nonparam, rw.p_param)}) by task design;\n"
                 "baseline firing rises equally in both cohorts. Solid = whisker, dashed = auditory trials.", fontsize=10.5, color="0.25")
    save(pdf, fig, k)


def s_synthesis(pdf, k, rows):
    fig = slide("Synthesis", "Beyond session time: R- whisker responses move away from the hit / miss axes after the task (relative to auditory), "
                "plus an R- state shift")
    ax = fig.add_axes([0.03, 0.12, 0.3, 0.68]); neural_space(ax)
    arrow(ax, (0, 0), (1.05, 0), "0.15"); ax.text(0.9, -0.06, "choice axis", fontsize=11, ha="center", va="top")
    arrow(ax, (0, 0), (0.75, 0.55), WC, lw=2.5, label="whisker pre\n(both cohorts)", lpos=1.28, fs=10)
    arrow(ax, (0, 0), (0.8, 0.32), COL["R+"], lw=2, ls="--", label="R+ post", lpos=1.2, fs=10)
    arrow(ax, (0, 0), (0.12, 0.85), COL["R-"], lw=2, ls="--", label="R- post: turned away", lpos=1.15, fs=10)
    arrow(ax, (0, 0), (-0.15, -0.12), "0.5", lw=1.5)
    ax.text(0.05, -0.2, "+ a shared state shift (both stimuli, larger in R-)", fontsize=10, color="0.4")
    tab = fig.add_axes([0.36, 0.1, 0.61, 0.72]); tab.axis("off")
    cell = [[r[0], r[1], r[2], r[3]] for r in rows]
    t = tab.table(cellText=cell, colLabels=["passive pre -> post; excess over the shift null unless noted", "R+", "R-", "R+ vs R- (MW | Welch)"],
                  loc="upper left", colWidths=[0.55, 0.12, 0.12, 0.21], cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(10.5); t.scale(1, 1.6)
    for (i, j), c_ in t.get_celld().items():
        c_.set_edgecolor("0.85")
        if i == 0:
            c_.set_facecolor("0.93"); c_.set_text_props(weight="bold")
    save(pdf, fig, k)


def s_caveats(pdf, k):
    fig = slide("Caveats and next steps")
    bullets(fig, 0.05, 0.8, ["Session time: hit / miss labels drift, so axes and decoders partly encode it; inference is on the excess over a "
                             "linear-shift null (conservative: real learning is time-correlated too). Time explains part, not most, of the R- changes.",
                             "State: the pre-stimulus window alone shifts miss-ward after the task in R- beyond time, equally for whisker and auditory; "
                             "the whisker-specific claim rests on whisker - auditory contrasts.",
                             "Exposure: R- mice get more whisker stimuli and longer sessions, and collect fewer rewards (by design); stimulus-specific "
                             "adaptation is not covered by the shift null or by the auditory contrast.",
                             "Noise: hit / miss axes at 5-35 ms have low split-half reliability; most shifted axes fall below the 0.05 floor, "
                             "so raw cosines and projections carry the null tests, noise-corrected cosines are a sensitivity check.",
                             "Same tracked stable units in every step (137b); sessions differ only by each analysis's hit / miss minimums.",
                             "Exploratory, uncorrected. Next: exposure covariate, time-matched decoder, then held-out mice / expert sessions."],
            size=13, dy=0.11, width=150)
    save(pdf, fig, k)


def s_backup(pdf, k, path, title):
    if not path.exists():
        return k
    fig = slide(f"Backup: {title}", path.name)
    im = plt.imread(path)
    ax = fig.add_axes([0.04, 0.03, 0.92, 0.83]); ax.imshow(im); ax.axis("off")
    save(pdf, fig, k)
    return k + 1


_K = [0]


def save(pdf, fig, k):
    SLIDES.mkdir(parents=True, exist_ok=True)
    _K[0] += 1
    fig.text(0.96, 0.02, str(_K[0]), fontsize=10, color="0.5", ha="right")
    pdf.savefig(fig); fig.savefig(SLIDES / f"148_slide_{_K[0]:02d}.png", dpi=150); plt.close(fig)


def summary_rows(D133, D135, D140, D146):
    rows = []
    def add(name, d, a, b):
        ch = {c: (d[d.reward_group == c][b] - d[d.reward_group == c][a]).to_numpy(float) for c in COH}
        mw, we = H.unpaired(ch["R+"], ch["R-"])
        rows.append([name, f"{np.nanmean(ch['R+']):+.2f}", f"{np.nanmean(ch['R-']):+.2f}", pfmt(mw, we)])
    def add_ex(name, d, col):
        x = {c: d[d.reward_group == c][col].to_numpy(float) for c in COH}
        mw, we = H.unpaired(x["R+"], x["R-"])
        rows.append([name, f"{np.nanmean(x['R+']):+.3f}", f"{np.nanmean(x['R-']):+.3f}", pfmt(mw, we)])
    d = D133[(D133.area == "All units") & D133.skipped_reason.isna()]
    add("1  W vs A decoding (raw change; no hit / miss axis)", d, "within_passive_pre_corr", "within_passive_post_corr")
    d = D135[(D135.area == "All units") & D135.skipped_reason.isna()]
    add_ex("3  whisker-evoked vs lick axis, raw cos", d, "shift_excess_dWR")
    add_ex("3  auditory-evoked vs lick axis, raw cos (control)", d, "shift_excess_dAR")
    add_ex("3  whisker - auditory on lick axis, projection", d, "shift_excess_dWAP")
    d = D140[(D140.split == "hitmedian") & D140.skipped_reason.isna()]
    add_ex("4  passive whisker axis vs CD half 2, raw cos", d, "shift_excess_daxisR")
    add_ex("4  whisker - auditory on CD half 2, projection", d, "shift_excess_dWAP")
    v = D146[(D146.area == "whole_brain") & (D146.unit_set == "stable") & D146.skipped_reason.isna()].copy()
    add_ex("5  decoder readout, whisker (main)", v[v.response == "epochbase"], "shift_excess_dW")
    add_ex("5  decoder readout, whisker - auditory, per-trial baseline", v[v.response == "trialbase"], "shift_excess_dWA")
    add_ex("6  baseline-window (state) readout, whisker", v[v.response == "baseline"], "shift_excess_dW")
    add_ex("6  baseline-window (state) readout, whisker - auditory", v[v.response == "baseline"], "shift_excess_dWA")
    return rows


def main():
    setup()
    D133 = pd.read_parquet(EA / "133_modality_stim_epochs_tracked.parquet")
    D134 = pd.read_parquet(EA / "134_whisker_specific_change_tracked.parquet")
    D135 = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    D140 = pd.read_parquet(EA / "140_coding_direction_noise.parquet")
    D146 = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    S147 = pd.read_csv(EA / "147_stats.csv")
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        s_title(pdf, 0); s_question(pdf, 0); s_data(pdf, 0, D146); s_tools(pdf, 0); s_null(pdf, 0)
        s_step1(pdf, 0, D133); s_step2(pdf, 0, D134); s_step3(pdf, 0, D135); s_step4(pdf, 0, D140); s_step5(pdf, 0, D146)
        s_step6(pdf, 0, D146, S147); s_synthesis(pdf, 0, summary_rows(D133, D135, D140, D146)); s_caveats(pdf, 0)
        for p, t in ((PUB / "150_axis_alignment_shift_null_all.png", "150 lick axis / coding direction vs the shift null"),
                     (PUB / "149_passive_readout_shift_null_all.png", "149 decoder readout vs the shift null"),
                     (FIG / "133_whole_brain_tracked.png", "133 whisker vs auditory across epochs"), (FIG / "134_whole_brain_tracked.png", "134 gain changes, sliding windows"),
                     (FIG / "134c_illustration_tracked.png", "134 illustration of the measures"), (FIG / "135b_alignment_no_bad_rplus_tracked.png", "135b lick-axis alignment"),
                     (PUB / "145_coding_direction_all.png", "145 coding direction"), (PUB / "145_evoked_alignment_all.png", "145 evoked patterns vs CD"),
                     (PUB / "147_choice_axis_readout_stable.png", "147 choice-axis readout"), (PUB / "147_state_space_whisker_auditory_axis.png", "147 state space"),
                     (PUB / "147_controls.png", "147 controls"), (PUB / "147_choice_axis_readout_area_groups.png", "147 area groups")):
            s_backup(pdf, 0, p, t)
    print(OUT_PDF, _K[0], "slides")


if __name__ == "__main__":
    main()
