"""152 -- Full results deck (user 2026-10-05: "a slide deck that shows all results, from active-only within-session changes to
results that compare passive and active ... from simple to complex, not too many methods at once ... go into a single method
and show it is robust, then show results hold across methods"; "make deck nicer, add latex equations to describe metrics,
quantities being plotted").
Flow: A active-only decoding within the session (one method: hit / miss decoding vs a linear-shift null; halves -> hit-median
split -> learning trial vs placebo -> robustness); B passive vs active with ONE method (lick-axis alignment), made robust step by
step (same units, session-time null, auditory control, learners, noise, area groups); C convergence across methods (coding
direction, decoder readout, state space), controls, synthesis, caveats; backup figures.
Style: section accent colours (A teal, B orange, C purple), header band, footer, rounded take-home boxes; every method / result
slide carries an equation box (matplotlib mathtext) defining the plotted quantity.
Numbers: Part II from the report's numbers.json (same values as the report), Part III re-computed from the per-session tables.
Helpers (panels, schematics) from 148_part3_digest.py; its slide / take-home functions are replaced by the styled ones here.
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
from matplotlib.patches import FancyBboxPatch, Rectangle

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
SEC = {"intro": ("", "#5c677d"), "A": ("A  active only", "#2a9d8f"), "B": ("B  passive vs active: one method", "#e07a3f"),
       "C": ("C  across methods", "#7b5ea7"), "bk": ("backup", "#8d99ae")}
CUR = {"sec": "intro"}
INK, SOFT = "#1d2433", "#5c677d"


# ------------------------------------------------------------------------------------------------------------------ style
def slide(title, subtitle=None):
    tag, col = SEC[CUR["sec"]]
    fig = plt.figure(figsize=(M.W_IN, M.H_IN), facecolor="white")
    fig.add_artist(Rectangle((0, 0.885), 1, 0.115, transform=fig.transFigure, facecolor="#f4f6f9", lw=0, zorder=-2))
    fig.add_artist(Rectangle((0, 0.885), 0.012, 0.115, transform=fig.transFigure, facecolor=col, lw=0))
    if tag:
        fig.text(0.035, 0.975, tag.upper(), fontsize=9.5, color=col, weight="bold", va="top")
    fig.text(0.035, 0.935, title, fontsize=21, weight="bold", color=INK, va="center")
    if subtitle:
        fig.text(0.035, 0.898, subtitle, fontsize=11.5, color=SOFT, va="center")
    fig.add_artist(plt.Line2D([0.035, 0.965], [0.035, 0.035], color="#d5dae1", lw=0.8))
    fig.text(0.035, 0.018, "Single-session whisker learning  |  within-session coding changes", fontsize=8.5, color="#8d99ae", va="center")
    return fig


def takehome(fig, text, x=0.03, y=0.05, w=0.31, h=0.19):
    col = SEC[CUR["sec"]][1]
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", transform=fig.transFigure,
                                  facecolor="#fffaf0", edgecolor=col, lw=1.4))
    fig.text(x + 0.012, y + h - 0.016, "TAKE-HOME", fontsize=9, weight="bold", color=col, va="top")
    fig.text(x + 0.012, y + h - 0.05, textwrap.fill(text, max(30, int(w * 135))), fontsize=11, color=INK, va="top")


def eqbox(fig, x, y, w, h, lines, title="Definitions", fs=12.5):
    """light box with mathtext lines (each a (label, latex) pair or a latex string)"""
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012", transform=fig.transFigure,
                                  facecolor="#f7f8fb", edgecolor="#c9d1dc", lw=1))
    fig.text(x + 0.012, y + h - 0.016, title.upper(), fontsize=9, weight="bold", color=SOFT, va="top")
    yy = y + h - 0.05
    step = (h - 0.06) / max(len(lines), 1)
    for ln in lines:
        lab, tex = ln if isinstance(ln, tuple) else ("", ln)
        if lab:
            fig.text(x + 0.012, yy, lab, fontsize=9.5, color=SOFT, va="top")
            fig.text(x + 0.012, yy - 0.028, tex, fontsize=fs, color=INK, va="top")      # equation hangs below its label
        else:
            fig.text(x + 0.012, yy, tex, fontsize=fs, color=INK, va="top")
        yy -= step


M.slide, M.takehome = slide, takehome          # the 148 helpers (s_data, s_null, s_step6, s_backup) use the styled versions
bullets, save = M.bullets, M.save


def n(key, fmt="{:+.3f}"):
    v = N.get(key)
    return "n/a" if v is None else fmt.format(v)


def p(key):
    v = N.get(key)
    return "n/a" if v is None else ("<0.001" if v < 0.001 else f"{v:.3f}")


def pp(k1, k2):
    return f"{p(k1)} | {p(k2)}"


def section(pdf, key, title, sub):
    CUR["sec"] = key
    col = SEC[key][1]
    fig = plt.figure(figsize=(M.W_IN, M.H_IN), facecolor="white")
    fig.add_artist(Rectangle((0, 0), 0.36, 1, transform=fig.transFigure, facecolor=col, lw=0))
    fig.text(0.18, 0.5, key, fontsize=120, weight="bold", color="white", ha="center", va="center", alpha=0.9)
    fig.text(0.41, 0.56, textwrap.fill(title, 34), fontsize=28, weight="bold", color=INK, va="center")
    fig.text(0.41, 0.36, textwrap.fill(sub, 60), fontsize=15, color=SOFT, va="center")
    save(pdf, fig, 0)


def frame_image(fig, rect, path):
    ax = fig.add_axes(rect)
    if path.exists():
        ax.imshow(plt.imread(path))
    else:
        ax.text(0.5, 0.5, f"[{path.name} missing]", ha="center", color="r", transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)


def img_slide(pdf, title, sub, path, items, th, eqs=None, img_w=0.58, eq_h=0.2):
    fig = slide(title, sub)
    frame_image(fig, [0.025, 0.06, img_w, 0.8], path)
    x0 = img_w + 0.05; w = 0.965 - x0
    ytop = 0.85
    if eqs:
        eqbox(fig, x0, ytop - eq_h, w, eq_h, eqs)
        ytop -= eq_h + 0.03
    bullets(fig, x0, ytop, items, size=11, dy=0.105, width=int(w * 118))
    takehome(fig, th, x=x0, y=0.055, w=w, h=0.17)
    save(pdf, fig, 0)


def panel_slide(pdf, title, sub, panels, eqs, th, eq_h=0.48, th_h=0.13):
    """up to three excess panels on the left, an equation box on the right, a take-home at the bottom"""
    fig = slide(title, sub)
    for j, (d, col, t) in enumerate(panels):
        M.excess_panel(fig.add_axes([0.075 + j * 0.205, 0.37, 0.13, 0.43]), d, col, t,
                       "excess post - pre\n(real - shift null)" if j == 0 else "", fs=8.5)
    eqbox(fig, 0.69, 0.85 - eq_h, 0.275, eq_h, eqs, fs=12)
    takehome(fig, th, x=0.69, y=0.055, w=0.275, h=0.85 - eq_h - 0.09)
    save(pdf, fig, 0)


EXCESS = r"$\mathrm{excess} = \Delta M - \dfrac{1}{K}\sum_{k=1}^{K}\Delta M^{(k)}_{\mathrm{shift}}$"
DELTA = r"$\Delta M = M^{\mathrm{post}} - M^{\mathrm{pre}}$"


# ------------------------------------------------------------------------------------------------------------------- intro
def s_title(pdf):
    CUR["sec"] = "intro"
    fig = plt.figure(figsize=(M.W_IN, M.H_IN), facecolor="white")
    fig.add_artist(Rectangle((0, 0.62), 1, 0.38, transform=fig.transFigure, facecolor="#1d2433", lw=0))
    fig.text(0.06, 0.82, "Within-session changes of choice and sensory coding\nduring single-session whisker learning", fontsize=27,
             weight="bold", color="white", va="center")
    fig.text(0.06, 0.67, "Whole-brain Neuropixels, learning session; R+ (whisker licks rewarded) vs R- (not rewarded)", fontsize=14,
             color="#c9d1dc", va="center")
    for i, k in enumerate("ABC"):
        col = SEC[k][1]
        t = {"A": "Does choice decoding change within the session? (active trials only)",
             "B": "Does the early whisker response move relative to the choice axis? One method, made robust",
             "C": "Does the result hold across methods? Controls, synthesis, caveats"}[k]
        fig.add_artist(FancyBboxPatch((0.06, 0.44 - i * 0.13), 0.06, 0.09, boxstyle="round,pad=0.004,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor=col, lw=0))
        fig.text(0.09, 0.485 - i * 0.13, k, fontsize=24, weight="bold", color="white", ha="center", va="center")
        fig.text(0.14, 0.485 - i * 0.13, t, fontsize=16, color=INK, va="center")
    save(pdf, fig, 0)


def s_task(pdf):
    CUR["sec"] = "intro"
    fig = slide("Task and question", "Same stimuli, same action, different contingency")
    ax = fig.add_axes([0.04, 0.3, 0.42, 0.5]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, (c, txt) in enumerate((("R+", "whisker -> lick -> reward\n(hit rate rises)"), ("R-", "whisker -> lick -> no reward\n(learn to withhold)"))):
        ax.add_patch(FancyBboxPatch((0.02, 0.6 - i * 0.45), 0.96, 0.32, boxstyle="round,pad=0.01,rounding_size=0.03", facecolor="white",
                                    edgecolor=COL[c], lw=2.5))
        ax.text(0.07, 0.76 - i * 0.45, c, fontsize=22, color=COL[c], weight="bold", va="center")
        ax.text(0.25, 0.76 - i * 0.45, txt, fontsize=14, va="center", color=INK)
    ax.text(0.02, 0.02, "auditory -> lick -> reward in both cohorts", fontsize=12, color=AC)
    bullets(fig, 0.52, 0.8, ["Hit = lick after the whisker stimulus: the same label in both cohorts (trained response in R+, error in R-).",
                             "H1: choice information after the whisker stimulus grows when R+ mice learn.",
                             "H2: it does not grow, or declines, in R-.",
                             "H3: the early (5-35 ms) whisker response is re-mapped relative to the lick representation, depending on "
                             "the contingency.",
                             "Session = unit of analysis; a non-parametric and a parametric test, uncorrected (exploratory)."],
            size=13, dy=0.12, width=62)
    save(pdf, fig, 0)


# ------------------------------------------------------------------------------------------------------------------- part A
def s_a_method(pdf):
    fig = slide("A1. One method: decode the upcoming choice from early activity",
                "Hit vs miss on active whisker trials, from the response 5-50 / 5-100 ms after the stimulus")
    ax = fig.add_axes([0.04, 0.42, 0.45, 0.4]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.0, 0.62), 0.38, 0.25, boxstyle="round,pad=0.01,rounding_size=0.03", color="#cfe8e3", lw=0))
    ax.text(0.19, 0.745, "active whisker trials\nhit | miss", ha="center", va="center", fontsize=12)
    ax.annotate("", xy=(0.55, 0.745), xytext=(0.4, 0.745), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.75, 0.745, "cross-validated\nlinear decoder", ha="center", va="center", fontsize=12)
    ax.annotate("", xy=(0.75, 0.42), xytext=(0.75, 0.62), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(0.75, 0.32, "accuracy minus\nlinear-shift null", ha="center", fontsize=12, weight="bold")
    eqbox(fig, 0.04, 0.07, 0.45, 0.3, [("balanced accuracy over held-out trials", r"$\mathrm{acc} = \frac{1}{2}\left(P(\hat y=\mathrm{hit}\,|\,\mathrm{hit}) + P(\hat y=\mathrm{miss}\,|\,\mathrm{miss})\right)$"),
                                      ("plotted: above the linear-shift null (labels shifted by k trials)", r"$\Delta\mathrm{acc} = \mathrm{acc}_{\mathrm{real}} - \langle \mathrm{acc}^{(k)}_{\mathrm{shift}} \rangle_k$")])
    bullets(fig, 0.54, 0.8, [f"Session-wide, choice is decodable in both cohorts already 5-50 ms after the stimulus: R+ {n('k0_rp')}, "
                             f"R- {n('k0_rm')} above the null.",
                             f"No cohort difference early on (Mann-Whitney | Welch p = {pp('k0_mw', 'k0_w')}).",
                             "Question for part A: does this information change within the session, and differently in the two cohorts?"],
            size=13, dy=0.14, width=60)
    takehome(fig, "The choice after a whisker stimulus is predictable from early whole-brain activity, rewarded or not.", x=0.54, y=0.07, w=0.425, h=0.15)
    save(pdf, fig, 0)


def s_part_a(pdf):
    img_slide(pdf, "A2. Split the session in halves: no change",
              "Count-matched decoders in the first and second half (equal hits and misses per half)",
              FIG / "117b_halves_matched_full.png",
              [f"R+ change {n('h117_rp')} (p = {p('h117_rp_p')}, n = {n('h117_rp_n', '{:.0f}')}); R- {n('h117_rm')} (p = {p('h117_rm_p')}).",
               f"Cohorts do not differ (p = {pp('r1_mw', 'r1_welch')}).",
               f"Pseudo-populations (equal neurons and trials): R+ {n('pp_rp')} (p = {p('pp_rp_p')}), R- {n('pp_rm')} (p = {p('pp_rm_p')})."],
              "Decodability of the choice does not change between session halves, in either cohort.",
              eqs=[("plotted per session", r"$\Delta\mathrm{acc}_{h},\ h \in \{1, 2\}$;  change $= \Delta\mathrm{acc}_2 - \Delta\mathrm{acc}_1$")], eq_h=0.13)
    img_slide(pdf, "A3. Equal numbers of hits before and after: hit-median split",
              "The split falls at the middle hit, so a hit-rate change cannot drive the comparison",
              PUB / "144_hitmedian_split_all.png",
              [f"Hit / miss decoding: no change in any window (smallest p = {p('o144_hm_pmin')}).",
               f"Pre-lick modality decoding falls in R+ ({n('k8_a', '{:.3f}')} -> {n('k8_b', '{:.3f}')}, p = {pp('k8_pw', 'k8_pt')}), more than in R- "
               f"(p = {pp('k8_c', 'k8_cw')})."],
              "Still no change in choice decoding; only the R+ pre-lick modality signal declines.",
              eqs=[("with H whisker hits in the session, split at hit", r"$\lfloor H/2 \rfloor + 1$"),
                   ("plotted", r"$\Delta\mathrm{acc}$ before vs after (within-half shift null)")], eq_h=0.2)
    img_slide(pdf, "A4. Split at the learning trial, against placebo splits",
              "Change at the behavioural learning trial vs the changes at every other split of the same session",
              PUB / "138_cosyne_lt_placebo_all.png",
              [f"R+: change at the learning trial {n('o138_all_rp_real')} vs placebo {n('o138_all_rp_pl')} (p = {pp('o138_all_rp_pw', 'o138_all_rp_pt')}).",
               f"R+ vs R- beyond placebo: p = {pp('o138_all_exc_mw', 'o138_all_exc_w')}, permutation p = {p('o138_all_exc_perm')}.",
               f"Learners only: R+ p = {pp('o138_learners_rp_pw', 'o138_learners_rp_pt')}."],
              "A small R+-specific gain appears when the split is tied to behaviour, not at an arbitrary split.",
              eqs=[("change at a split s", r"$\delta_s = \Delta\mathrm{acc}^{\mathrm{after}}_s - \Delta\mathrm{acc}^{\mathrm{before}}_s$"),
                   ("beyond placebo (splits >= 10 trials away)", r"$\delta_{\mathrm{LT}} - \langle \delta_s \rangle_{s \in \mathrm{placebo}}$")], eq_h=0.24)
    img_slide(pdf, "A5. Is the learning-trial effect robust? Windows and kind of learner",
              "Change point of the whisker hit sequence (L5), three windows; step vs gradual learners",
              PUB / "143_lt_split_L5_windows_all.png",
              [f"R+ beyond placebo: 5-35 ms p = {pp('o143_5_35ms_rp_pw', 'o143_5_35ms_rp_pt')}, 5-50 ms p = "
               f"{pp('o143_5_50ms_rp_pw', 'o143_5_50ms_rp_pt')}, 5-100 ms p = {pp('o143_5_100ms_rp_pw', 'o143_5_100ms_rp_pt')}; R- none.",
               f"R+ vs R- (5-100 ms) p = {pp('o143_5_100ms_ex_mw', 'o143_5_100ms_ex_w')}, permutation p = {p('o143_5_100ms_ex_perm')}.",
               f"Carried by step learners (step vs gradual, 5-50 ms: p = {pp('o143sg_5_50ms_Rp_mw', 'o143sg_5_50ms_Rp_w')})."],
              "The R+ gain holds across windows but belongs to abrupt (step) learners; gradual learners show no change.",
              eqs=[("L5: Bayesian change point of the whisker hit sequence", r"$\delta_{\mathrm{L5}} - \langle \delta_s \rangle_{\mathrm{placebo}}$ per window")], eq_h=0.13)
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
    fig = slide("B1. One pattern, one axis", "Everything in part B compares a passive response pattern with an active hit / miss axis, "
                "before vs after the task")
    ax = fig.add_axes([0.03, 0.1, 0.36, 0.72]); M.neural_space(ax)
    M.arrow(ax, (0, 0), (1.05, 0), "0.15"); ax.text(0.85, -0.07, "lick axis L", fontsize=12, ha="center", va="top")
    pw_ = (0.6, 0.7); M.arrow(ax, (0, 0), pw_, WC, label="passive whisker\npattern p", lpos=1.2, fs=11)
    ax.plot([pw_[0], pw_[0]], [0, pw_[1]], color="0.5", lw=1, ls=":"); ax.plot([0, pw_[0]], [-0.03, -0.03], color="0.3", lw=4)
    ax.text(0.17, 0.08, r"$\theta$", fontsize=16)
    eqbox(fig, 0.42, 0.08, 0.545, 0.76, [
        ("response of unit i on trial t (5-35 ms, minus the epoch's baseline, -55 to -20 ms)",
         r"$r_{it} = f^{[5,35]}_{it} - \langle f^{[-55,-20]}_{it'} \rangle_{t' \in \mathrm{epoch}}$"),
        ("z-scored (axes) and evoked (patterns) versions", r"$z_{it} = \dfrac{r_{it}-\mu_i}{\sigma_i}, \qquad \varepsilon_{it} = \dfrac{r_{it}}{\sigma_i}$"),
        ("passive pattern (mean evoked response to a stimulus)", r"$\mathbf{p}_W = \langle \varepsilon_{\cdot t} \rangle_{t \in \mathrm{whisker}}$"),
        ("lick axis (active whisker trials)", r"$\mathbf{L} = \bar{\mathbf{z}}_{\mathrm{hit}} - \bar{\mathbf{z}}_{\mathrm{miss}}$"),
        ("direction and reach along the axis", r"$\cos\theta = \dfrac{\mathbf{p}\cdot\mathbf{L}}{\|\mathbf{p}\|\,\|\mathbf{L}\|}, \qquad \pi = \dfrac{\mathbf{p}\cdot\hat{\mathbf{L}}}{\sqrt{n}} = \dfrac{\|\mathbf{p}\|}{\sqrt{n}}\cos\theta$"),
        ("prediction (H3): after the task, in R- only", r"$\cos\theta(\mathbf{p}_W^{\mathrm{post}}, \mathbf{L}) < \cos\theta(\mathbf{p}_W^{\mathrm{pre}}, \mathbf{L})$")],
        title="Quantities", fs=13)
    save(pdf, fig, 0)


def s_b_raw(pdf, D):
    fig = slide("B2. Lick-axis alignment across the session (135)", "Raw cosine of the evoked patterns with the lick axis; the same "
                "tracked stable units in every epoch")
    d = D[(D.area == "All units") & D.skipped_reason.isna()]
    ep = ["passive_pre", "active_1", "active_2", "passive_post"]
    for j, (s, t) in enumerate((("W", "whisker-evoked pattern"), ("A", "auditory-evoked pattern (control)"))):
        a = fig.add_axes([0.06 + j * 0.33, 0.36, 0.26, 0.44])
        M.change_panel(a, d, [f"evoked{s}_cos_{e}" for e in ep], M.EPL4, t, r"$\cos\theta(\mathbf{p}, \mathbf{L})$",
                       legend="upper right" if j == 0 else None)
    eqbox(fig, 0.72, 0.45, 0.245, 0.38, [("plotted", r"$\cos\theta(\mathbf{p}^{e}_{S}, \mathbf{L})$"),
                                         ("S = whisker, auditory", r"$e \in$ {pre, act 1, act 2, post}"),
                                         ("tested", DELTA)], fs=12)
    takehome(fig, "R- whisker alignment falls during the task and stays low in passive post; auditory does not differ between cohorts. "
             "Active halves are descriptive (they average hits and misses).", x=0.72, y=0.07, w=0.245, h=0.33)
    save(pdf, fig, 0)


def s_b_time(pdf, D):
    d = D[(D.area == "All units") & D.skipped_reason.isna()]
    panel_slide(pdf, "B3. The result beyond session time (135)", "Passive pre -> post change minus the change produced by lick axes "
                "rebuilt from time-shifted labels",
                [(d, "shift_excess_dWR", "whisker\nraw cosine"), (d, "shift_excess_dAR", "auditory\nraw cosine"),
                 (d, "shift_excess_dWAP", "whisker - auditory\nprojection")],
                [("change of a metric M", DELTA), ("excess over K = 50 shifted axes", EXCESS),
                 ("linear contrast (shared drift cancels)", r"$\Delta\pi_W - \Delta\pi_A$")],
                "Beyond session time, the R- whisker pattern moves away from the lick axis after the task; R+ does not; auditory "
                "does not differ; whisker relative to auditory differs.")


def s_b_robust(pdf, D):
    d = D[(D.area == "All units") & D.skipped_reason.isna()]
    L = d[H.mouse_of(d).isin(H.learners())]
    panel_slide(pdf, "B4. Robust to the population and to the noise metric (135)", "Learners only; noise-corrected cosine as a "
                "sensitivity check",
                [(L, "shift_excess_dWR", "learners: whisker\nraw cosine"), (L, "shift_excess_dWAP", "learners: W - A\nprojection"),
                 (d, "shift_excess_dWN", "all: whisker\nnoise-corrected")],
                [("noise-corrected cosine (disjoint trial halves 1, 2)", r"$\cos_{\mathrm{norm}} = \dfrac{\cos(\mathbf{p}^{(1)},\mathbf{L}^{(2)})}{\sqrt{\rho_p\,\rho_L}}$"),
                 ("split-half reliability", r"$\rho_p = \cos(\mathbf{p}^{(1)}, \mathbf{p}^{(2)})$, floored at 0.05")],
                "Holds in learners only and with the noise-corrected cosine (noisier null: most shifted axes are unreliable).")


def s_b_area(pdf):
    img_slide(pdf, "B5. Where in the brain? Area groups (151)", "Each area with its own axes; >= 3 sessions per cohort, >= 20 units",
              PUB / "151_area_heatmap_all.png",
              ["Read the last two columns (lick axis beyond session time) and the R- minus R+ panel.",
               "Cohort differences (both tests) in somatosensory-whisker cortex, motor areas, midbrain and thalamus.",
               "Most areas point in the R- direction; single cells are small and uncorrected."],
              "Distributed, strongest in whisker sensorimotor areas, midbrain and thalamus.",
              eqs=[("cell value", r"$\langle \mathrm{excess} \rangle_{\mathrm{sessions}}$ or $\langle \Delta x \rangle$"),
                   ("stars", r"* one, ** both tests $p<0.05$")], eq_h=0.19, img_w=0.64)


# ------------------------------------------------------------------------------------------------------------------- part C
def s_c_cd(pdf, D140):
    d = D140[(D140.split == "hitmedian") & D140.skipped_reason.isna()]
    panel_slide(pdf, "C1. Method 2: the late hit / miss coding direction (140)", "Coding direction of the second half of a hit-median "
                "split: the direction at the end of the task",
                [(d, "shift_excess_daxisR", "passive W - A axis\nraw cosine"), (d, "shift_excess_dWAP", "whisker - auditory\nprojection"),
                 (d, "shift_excess_dAR", "auditory\nraw cosine")],
                [("late coding direction", r"$\mathbf{CD}_2 = \bar{\mathbf{z}}_{\mathrm{hit},2} - \bar{\mathbf{z}}_{\mathrm{miss},2}$"),
                 ("passive stimulus axis", r"$\mathbf{D} = \bar{\mathbf{z}}_W - \bar{\mathbf{z}}_A$"),
                 ("tested", r"excess of $\Delta\cos(\mathbf{D}, \mathbf{CD}_2)$, $\Delta\pi_W - \Delta\pi_A$")],
                "Same answer with a different axis: beyond session time, the R- passive whisker response turns away from the late "
                "coding direction, relative to auditory.")


def s_c_dec(pdf, D146):
    v = D146[(D146.area == "whole_brain") & (D146.unit_set == "stable") & D146.skipped_reason.isna()]
    panel_slide(pdf, "C2. Method 3: a choice decoder read out on passive trials (146)", "Train hit vs miss on active trials, score "
                "passive trials; + hit-like, - miss-like",
                [(v[v.response == "epochbase"], "shift_excess_dW", "whisker\nreadout"),
                 (v[v.response == "trialbase"], "shift_excess_dWA", "W - A, per-trial\nbaseline"),
                 (v[v.response == "baseline"], "shift_excess_dW", "baseline window\n(state), whisker")],
                [("decoder (L2 logistic regression)", r"$s(\mathbf{z}) = \mathbf{w}\cdot\mathbf{z} + b, \quad \mathbf{w} \propto \Sigma^{-1}(\mu_{\mathrm{hit}} - \mu_{\mathrm{miss}})$"),
                 ("readout (m: hit / miss midpoint, held-out SD)", r"$R = \dfrac{s(\mathbf{z}) - m}{\sigma_s}$"),
                 ("tested", r"excess of $\Delta \langle R \rangle_W$, $\Delta(\langle R \rangle_W - \langle R \rangle_A)$")],
                "Agrees only relative to auditory; part of the R- change is a pre-stimulus state shift common to both stimuli.", eq_h=0.5)


def s_c_ss(pdf):
    img_slide(pdf, "C3. Method 4: the state space (151)", "Passive pre at the origin; both cohorts overlaid; dashed: active miss / hit level",
              PUB / "151_state_space_centered_all.png",
              ["Both responses shrink along their own pre-task pattern, in both cohorts (shared).",
               "Only the R- whisker response slides along the choice axis, to the active miss level.",
               "The cohort difference lives on the choice axis for every y axis."],
              "After the task, an R- whisker stimulus evokes a pattern like a whisker trial without a lick.",
              eqs=[("axes", r"$\hat{\mathbf{x}} = \dfrac{\mathbf{L}}{\|\mathbf{L}\|}, \quad \hat{\mathbf{y}} \propto \mathbf{v} - (\mathbf{v}\cdot\hat{\mathbf{x}})\,\hat{\mathbf{x}}$"),
                   ("v = pre whisker, W - A, or auditory pattern", r"$\Delta x = (\mathbf{p}^{\mathrm{post}} - \mathbf{p}^{\mathrm{pre}})\cdot\hat{\mathbf{x}}$")],
              eq_h=0.27, img_w=0.47)


def s_c_table(pdf, D135, D140, D146, S):
    fig = slide("C4. The result across methods", "Passive pre -> post, excess over the linear-shift null unless noted (R+ vs R-: Mann-Whitney | Welch)")
    rows, sig = [], []

    def add(name, d, col):
        x = {c: d[d.reward_group == c][col].to_numpy(float) for c in COH}
        mw, we = H.unpaired(x["R+"], x["R-"])
        rows.append([name, f"{np.nanmean(x['R+']):+.3f}", f"{np.nanmean(x['R-']):+.3f}", M.pfmt(mw, we)]); sig.append(mw < 0.05 and we < 0.05)
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
            sig.append(rc.p_nonparam < 0.05 and rc.p_param < 0.05)
    tab = fig.add_axes([0.05, 0.14, 0.9, 0.68]); tab.axis("off")
    t = tab.table(cellText=rows, colLabels=["method: measure", "R+", "R-", "R+ vs R-"], loc="upper left", colWidths=[0.58, 0.12, 0.12, 0.18], cellLoc="left")
    t.auto_set_font_size(False); t.set_fontsize(12.5); t.scale(1, 2.1)
    for (i, j), c_ in t.get_celld().items():
        c_.set_edgecolor("#e1e5eb")
        if i == 0:
            c_.set_facecolor("#f4f6f9"); c_.set_text_props(weight="bold", color=INK)
        elif sig[i - 1]:
            c_.set_facecolor("#fbefe6" if j < 3 else "#f6dccb")
    fig.text(0.05, 0.075, "Shaded: R+ vs R- with both tests p < 0.05. Mean-difference axes and the state space agree; the decoder agrees "
             "relative to auditory.", fontsize=12.5, weight="bold", color=INK)
    save(pdf, fig, 0)


def s_synth(pdf):
    CUR["sec"] = "intro"
    fig = slide("Synthesis", "From decodability to geometry")
    for i, (k, t) in enumerate((("A", "Choice is decodable early in both cohorts; its decodability is stable within the session, except an R+ "
                                      "gain at abrupt behavioural transitions."),
                                ("B", "With one method (lick-axis alignment), after the task the R- passive whisker response moves away from "
                                      "the active lick axis, beyond session time, relative to auditory, in learners too, strongest in whisker "
                                      "sensorimotor areas, midbrain and thalamus."),
                                ("C", "The same holds for the late coding direction and in the state space (R- whisker response slides to the "
                                      "miss level); the decoder shows it relative to auditory; R- also shows a pre-stimulus state shift."))):
        col = SEC[k][1]
        fig.add_artist(FancyBboxPatch((0.05, 0.66 - i * 0.2), 0.05, 0.13, boxstyle="round,pad=0.004,rounding_size=0.01",
                                      transform=fig.transFigure, facecolor=col, lw=0))
        fig.text(0.075, 0.725 - i * 0.2, k, fontsize=24, weight="bold", color="white", ha="center", va="center")
        fig.text(0.12, 0.725 - i * 0.2, textwrap.fill(t, 115), fontsize=14, color=INK, va="center")
    takehome(fig, "In R-, where licking to the whisker is not rewarded, the earliest whisker response is re-mapped away from the motor "
             "(lick) representation; decodability alone misses this.", x=0.05, y=0.07, w=0.9, h=0.13)
    save(pdf, fig, 0)


def s_caveats(pdf):
    CUR["sec"] = "intro"
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
    plt.rcParams.update({"mathtext.fontset": "dejavusans", "axes.edgecolor": "#5c677d", "axes.labelcolor": INK,
                         "xtick.color": SOFT, "ytick.color": SOFT})
    D135 = pd.read_parquet(EA / "135_alignment_epochs_tracked.parquet")
    D140 = pd.read_parquet(EA / "140_coding_direction_noise.parquet")
    D146 = pd.read_parquet(EA / "146_choice_axis_readout.parquet")
    S147 = pd.read_csv(EA / "147_stats.csv")
    S151 = pd.read_csv(EA / "151_stats.csv") if (EA / "151_stats.csv").exists() else None
    M.SLIDES.mkdir(parents=True, exist_ok=True)
    for f in list(M.SLIDES.glob("148_slide_*.png")) + list(M.SLIDES.glob("152_slide_*.png")):
        f.unlink()
    with PdfPages(OUT_PDF) as pdf:
        s_title(pdf); s_task(pdf)
        section(pdf, "A", "Active only: does choice decoding change within the session?", "One method: hit / miss decoding against a linear-shift null")
        s_a_method(pdf); s_part_a(pdf)
        section(pdf, "B", "Passive vs active: one method, made robust", "Lick-axis alignment of the early (5-35 ms) passive responses, before vs after the task")
        s_b_tools(pdf); M.s_data(pdf, 0, D146); s_b_raw(pdf, D135); M.s_null(pdf, 0); s_b_time(pdf, D135); s_b_robust(pdf, D135); s_b_area(pdf)
        section(pdf, "C", "Does it hold across methods?", "Late coding direction, choice decoder, state space; controls; synthesis")
        s_c_cd(pdf, D140); s_c_dec(pdf, D146); s_c_ss(pdf); s_c_table(pdf, D135, D140, D146, S151); M.s_step6(pdf, 0, D146, S147)
        s_synth(pdf); s_caveats(pdf)
        CUR["sec"] = "bk"
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
