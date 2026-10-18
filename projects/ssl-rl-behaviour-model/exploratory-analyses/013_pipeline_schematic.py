"""013 -- Pipeline schematic of model v3 (user 2026-10-18: "have you done figures to explain the pipeline? do you first simulate then
fit?"): data preparation, validation by simulation (recovery: simulate with known parameters on the real trial sequences, refit, compare)
BEFORE fitting real mice, real-data fitting with leave-one-session-out, model selection, evaluation; with the cost of each step.
Output: figures/013_pipeline_schematic.{png,pdf,svg}
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import common as C  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "pdf.fonttype": 42, "svg.fonttype": "none"})


def box(ax, x, y, w, h, title, body, fc, status=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008", fc=fc, ec="k", lw=0.6, clip_on=False))
    ax.text(x, y + h / 2 - 0.012, title, ha="center", va="top", fontsize=6.2, weight="bold")
    ax.text(x, y + h / 2 - 0.042, body, ha="center", va="top", fontsize=5.1, linespacing=1.3)
    if status:
        col = {"done": "#38761d", "running": "#b45f06", "to do": "#990000"}[status]
        ax.text(x + w / 2 - 0.006, y - h / 2 + 0.008, status, ha="right", va="bottom", fontsize=5, color=col, weight="bold")


def arr(ax, a, b, lab=None):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.9, color="0.25", clip_on=False))
    if lab:
        ax.text((a[0] + b[0]) / 2 + 0.01, (a[1] + b[1]) / 2, lab, fontsize=5, color="0.25", va="center")


def main():
    fig, ax = plt.subplots(figsize=(7.4, 8.0)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.995, "Model v3 pipeline: validate by simulation first, then fit real mice", ha="center", va="top", fontsize=8, weight="bold")
    box(ax, 0.5, 0.905, 0.62, 0.11, "0  Data (001)",
        "behaviour NWBs → one trial sequence per mouse (111 mice, 226,670 trials): pretraining + whisker days,\n"
        "active trials, perf ≠ 6, no trimming; per trial: stimulus, lick, session; outcome of a lick fixed by the\n"
        "cohort design (A +1 water, W +1 water R+ / −1 R−, no stim −0.5)", "#f4f4f4", "done")
    # validation column
    box(ax, 0.25, 0.70, 0.46, 0.20, "1  Validation by simulation (recovery)",
        "a. pick generating parameters (realistic ranges, or a pilot fit)\n"
        "b. SIMULATE licks with each model on each mouse's real trial\n"
        "    sequence (same stimuli, sessions, cohort outcomes)\n"
        "c. FIT all models to the simulated licks (same procedure as 2)\n"
        "d. compare with the truth:\n"
        "    model recovery: is the generating model selected? (confusion)\n"
        "    parameter recovery: true vs fitted parameters",
        "#fff2cc", "to do")
    box(ax, 0.25, 0.47, 0.46, 0.14, "1e  Decide what the data can tell apart",
        "components never recovered → merge / redefine / fix\n"
        "parameters not recovered → fix or report as products/ratios\n"
        "(v2: γ weakly identified; β·α only as a product)", "#fff2cc", "to do")
    # real data column
    box(ax, 0.75, 0.70, 0.46, 0.20, "2  Fit real mice (011)",
        "64 models (core + any of G, M, N, Z, R, P)\n"
        "maximum likelihood of each lick, trial by trial\n"
        "full data: 1 fit per model and mouse\n"
        "leave-one-session-out: ~8 more fits per model and mouse\n"
        "(fit on other sessions, score the left-out one)\n"
        "starts: default + 2 random + optima of sub-models",
        "#cfe2f3", "running")
    box(ax, 0.75, 0.47, 0.46, 0.14, "3  Model selection (012)",
        "held-out log-likelihood per mouse (all sessions; day 0)\n"
        "random-effects BMS: model frequencies, exceedance\n"
        "component families: P(component used)", "#cfe2f3", "to do")
    box(ax, 0.5, 0.26, 0.9, 0.15, "4  Evaluation and interpretation",
        "predictive checks: one-step and closed-loop simulations vs learning curves, within-session curves, the day-0 transition\n"
        "swapped contingency: each mouse simulated under the other cohort's outcomes (cohort = design only?)\n"
        "parameters R+ vs R− (Mann-Whitney + Welch); per-trial latents (values, thirst, prediction errors) for neural alignment",
        "#d9ead3", "to do")
    arr(ax, (0.36, 0.85), (0.27, 0.80)); arr(ax, (0.64, 0.85), (0.73, 0.80))
    arr(ax, (0.25, 0.60), (0.25, 0.54)); arr(ax, (0.75, 0.60), (0.75, 0.54))
    arr(ax, (0.48, 0.47), (0.52, 0.47), "trust only\nrecoverable\ncomponents")
    arr(ax, (0.75, 0.40), (0.62, 0.335)); arr(ax, (0.25, 0.40), (0.38, 0.335))
    ax.text(0.5, 0.155, "Cost of step 2 (why it is slow)", ha="center", fontsize=6.8, weight="bold")
    ax.text(0.5, 0.135, "per mouse: 64 models × 9 fits (full + ~8 left-out sessions) × ~4-9 starts × ~100-200 optimiser iterations × ~19 model "
                        "runs per iteration\n(numerical gradient over 18 parameters) × ~2,000 trials  →  ~10⁷ passes through the mouse's data "
                        "≈ 1 h per mouse on a loaded core\n"
                        "Faster options: analytical / automatic gradients (10-20×), fewer models (full, full − one component, core + one "
                        "component: 13 instead of 64),\nleave-one-session-out only for the top models, fewer starts once recovery shows "
                        "the landscape is benign", ha="center", va="top", fontsize=5.3, linespacing=1.4)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"013_pipeline_schematic.{ext}", dpi=300, bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
