"""010 -- Schematic of the v3 model design (model_v3_design.md) for review: generative loop per trial, components, and illustrative curves
of each component with example parameter values (not fitted).
Output: figures/010_design_schematic.{png,pdf,svg}
Run (haas): python 010_design_schematic.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import common as C  # noqa: E402

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})
COL = dict(core="#cfe2f3", gen="#d9ead3", mot="#f4cccc", nov="#fff2cc", ctx="#e6d7f0", per="#e0e0e0", dec="#fde0c5", obs="#ffffff")


def box(ax, x, y, w, h, txt, fc, fs=5.6, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008", fc=fc, ec="k", lw=0.6, clip_on=False))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arr(ax, a, b, col="0.25", ls="-", lab=None, off=(0, 0.012), rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=col, ls=ls, clip_on=False,
                                 connectionstyle=f"arc3,rad={rad}"))
    if lab:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], lab, fontsize=4.9, color=col, ha="center")


def main():
    fig = plt.figure(figsize=(7.2, 8.4))
    ax = fig.add_axes([0.01, 0.40, 0.98, 0.59]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 0.995, "A  Generative loop per trial (one learning rule for every mouse; cohorts differ only in the outcome table)",
            fontsize=7, weight="bold", va="top")
    # stimulus and features
    box(ax, 0.08, 0.86, 0.14, 0.06, "stimulus s_t\nA / W / N", COL["obs"], bold=True)
    box(ax, 0.08, 0.70, 0.14, 0.05, "bias  (context c)", COL["core"])
    box(ax, 0.08, 0.62, 0.14, 0.05, "stimulus present S", COL["gen"])
    box(ax, 0.08, 0.54, 0.14, 0.05, "auditory A", COL["core"])
    box(ax, 0.08, 0.46, 0.14, 0.05, "whisker W", COL["core"])
    arr(ax, (0.08, 0.83), (0.08, 0.725))
    # learned value
    box(ax, 0.34, 0.58, 0.20, 0.13, "learned value of licking\nQ(s) = w_b,c + γ·w_S + (1−γ)·w_s\nQ(N) = w_b,c", COL["core"])
    for y in (0.70, 0.62, 0.54, 0.46):
        arr(ax, (0.155, y), (0.24, 0.58 + (y - 0.58) * 0.4))
    # motivation
    box(ax, 0.34, 0.86, 0.20, 0.09, "thirst T_t (reset to 1 each session)\nT = 1 − ρ·Σ water absorbed (τ)", COL["mot"])
    box(ax, 0.60, 0.86, 0.17, 0.07, "Pavlovian lick bias\nB(T) ≈ s_B·[T − θ]₊", COL["mot"])
    arr(ax, (0.44, 0.86), (0.515, 0.86))
    # novelty, perseveration
    box(ax, 0.34, 0.30, 0.20, 0.08, "novelty bonus\nNov(s) = ν·exp(−n(s)/λ_ν)\nn(s): presentations so far", COL["nov"])
    box(ax, 0.34, 0.18, 0.20, 0.07, "perseveration\nC ← C + φ(l − C);  π(2C − 1)", COL["per"])
    # decision
    box(ax, 0.66, 0.55, 0.22, 0.13, "decision variable\nDV = β·T·Q(s) + B(T)\n        + Nov(s) + π(2C−1) − c", COL["dec"], fs=5.8)
    arr(ax, (0.44, 0.58), (0.55, 0.57), lab="Q", off=(0, 0.012))
    arr(ax, (0.34, 0.815), (0.62, 0.62), col="#990000", lab="× T", off=(0.03, 0.0))
    arr(ax, (0.60, 0.825), (0.66, 0.62), col="#990000", lab="+ B(T)", off=(0.04, 0.0))
    arr(ax, (0.44, 0.30), (0.60, 0.485))
    arr(ax, (0.44, 0.18), (0.64, 0.485))
    box(ax, 0.66, 0.36, 0.22, 0.08, "P(lick) = (1−λ)·[1 − (1−p_sp)(1 − σ(DV))]\nλ disengagement, p_sp spontaneous", COL["dec"], fs=5.5)
    arr(ax, (0.66, 0.485), (0.66, 0.40))
    box(ax, 0.66, 0.22, 0.14, 0.05, "lick l_t ∈ {0, 1}", COL["obs"], bold=True)
    arr(ax, (0.66, 0.32), (0.66, 0.245))
    # outcome table
    ax.text(0.80, 0.30, "outcome of a lick (design):", fontsize=5.6, weight="bold")
    ax.text(0.80, 0.105, "w = water delivered", fontsize=5.0, color="0.3")
    tab = ax.table(cellText=[["", "A", "W", "N"], ["R+", "+1 w", "+1 w", "−1"], ["R−", "+1 w", "−1", "−1"]], cellLoc="center",
                   bbox=[0.79, 0.12, 0.2, 0.16])
    tab.auto_set_font_size(False); tab.set_fontsize(5.3)
    for (_, _), c_ in tab.get_celld().items():
        c_.set_linewidth(0.4)
    arr(ax, (0.73, 0.22), (0.79, 0.20))
    # updates
    box(ax, 0.40, 0.05, 0.50, 0.075, "after a lick:  δ = o − Q(s),  α = α± ;   w_b,c += α·κ·δ,   w_S += α·γ·δ,   w_s += α·(1−γ)·δ\n"
                                       "water (rewarded lick, at the corrected lick time) → absorbed with τ → lowers T", COL["core"], fs=5.4)
    arr(ax, (0.79, 0.12), (0.655, 0.07), col="0.25", rad=-0.2, lab="o_t, water", off=(0.05, -0.03))
    arr(ax, (0.15, 0.05), (0.035, 0.435), col="0.25", rad=-0.25)
    arr(ax, (0.47, 0.088), (0.40, 0.815), col="#990000", ls=":", rad=0.25)
    # context switch annotation
    box(ax, 0.10, 0.30, 0.16, 0.085, "context switch at day 0\nw_b,whisker ← (1−ζ)·w_b,auditory", COL["ctx"], fs=5.2)
    arr(ax, (0.10, 0.345), (0.08, 0.675), col="#674ea7", ls=":")
    # legend of components
    for i, (k, lab) in enumerate((("core", "core learning (always on)"), ("gen", "G generalisation γ"), ("mot", "M motivation ρ, τ, s_B, θ"),
                                  ("nov", "N novelty ν, λ_ν"), ("ctx", "Z context ζ"), ("per", "P perseveration π, φ"), ("dec", "decision β, c, λ, p_sp"))):
        ax.add_patch(plt.Rectangle((0.79 + 0.0, 0.94 - i * 0.032), 0.018, 0.022, fc=COL[k], ec="k", lw=0.4, clip_on=False))
        ax.text(0.815, 0.951 - i * 0.032, lab, fontsize=5.2, va="center")
    # B: illustrative curves
    ax0 = fig.add_axes([0.01, 0.355, 0.98, 0.03]); ax0.axis("off")
    ax0.text(0.0, 0.5, "B  Component behaviour with example values (illustration, not fitted)", fontsize=7, weight="bold", va="center")
    gs = fig.add_gridspec(2, 4, left=0.07, right=0.98, bottom=0.04, top=0.33, hspace=0.75, wspace=0.45)
    # thirst within a session
    a = fig.add_subplot(gs[0, 0]); t = np.arange(0, 3600, 5.0)
    for rho, tau, ls in ((0.006, 300, "-"), (0.006, 60, "--"), (0.003, 300, ":")):
        rew = t[(np.random.default_rng(0).random(len(t)) < 0.35)]
        T = np.array([max(0, 1 - rho * np.sum(1 - np.exp(-(x - rew[rew < x]) / tau))) for x in t])
        a.plot(t / 60, T, ls=ls, color="#990000", lw=0.9, label=f"ρ={rho}, τ={tau}s")
    a.set_xlabel("time in session (min)"); a.set_ylabel("thirst T"); a.set_title("satiation (reward rate 0.07/s)", fontsize=6)
    a.legend(frameon=False, fontsize=4.5)
    a = fig.add_subplot(gs[0, 1]); T = np.linspace(0, 1, 200)
    for th, ls in ((0.3, "-"), (0.6, "--")):
        a.plot(T, 2 * np.log1p(np.exp(20 * (T - th))) / 20, color="#990000", ls=ls, lw=0.9, label=f"θ={th}, s_B=2")
    a.set_xlabel("thirst T"); a.set_ylabel("B(T)"); a.set_title("Pavlovian lick bias", fontsize=6); a.legend(frameon=False, fontsize=4.5)
    a = fig.add_subplot(gs[0, 2]); n = np.arange(0, 300)
    for lam, ls in ((20, "-"), (80, "--")):
        a.plot(n, 1.5 * np.exp(-n / lam), color="#b45f06", ls=ls, lw=0.9, label=f"ν=1.5, λ_ν={lam}")
    a.set_xlabel("presentations of the stimulus"); a.set_ylabel("Nov"); a.set_title("novelty bonus", fontsize=6); a.legend(frameon=False, fontsize=4.5)
    a = fig.add_subplot(gs[0, 3]); dv = np.linspace(-6, 6, 200)
    for lam, ps in ((0, 0), (0.1, 0), (0, 0.15), (0.1, 0.15)):
        a.plot(dv, (1 - lam) * (1 - (1 - ps) * (1 - 1 / (1 + np.exp(-dv)))), lw=0.9, label=f"λ={lam}, p_sp={ps}")
    a.set_xlabel("DV"); a.set_ylabel("P(lick)"); a.set_title("choice with lapses", fontsize=6); a.legend(frameon=False, fontsize=4.5)
    # generalisation: whisker value at onset vs gamma
    a = fig.add_subplot(gs[1, 0]); g = np.linspace(0, 1, 50)
    for wS, wA in ((0.8, 0.8), (0.4, 0.8)):
        a.plot(g, g * wS, lw=0.9, label=f"w_S={wS}")
    a.set_xlabel("γ"); a.set_ylabel("Q(W) − bias at onset"); a.set_title("generalisation at day 0", fontsize=6); a.legend(frameon=False, fontsize=4.5)
    a = fig.add_subplot(gs[1, 1])
    x = np.arange(-200, 200); wb = np.where(x < 0, 0.3, np.nan)
    for z, ls in ((0, "-"), (0.5, "--"), (1, ":")):
        a.plot(x, np.where(x < 0, 0.3, 0.3 * (1 - z) * np.exp(0 * x)), ls=ls, color="#674ea7", lw=0.9, label=f"ζ={z}")
    a.axvline(0, color="r", lw=0.5, ls="--"); a.set_xlabel("trial (0 = first whisker session)"); a.set_ylabel("bias weight")
    a.set_title("context switch (before relearning)", fontsize=6); a.legend(frameon=False, fontsize=4.5)
    a = fig.add_subplot(gs[1, 2]); rng = np.random.default_rng(1); l = (rng.random(60) < 0.5).astype(float); Cv = [0.5]
    for x_ in l:
        Cv.append(Cv[-1] + 0.3 * (x_ - Cv[-1]))
    a.step(range(60), l, color="0.7", lw=0.6, where="post", label="licks"); a.plot(range(61), Cv, color="k", lw=0.9, label="C (φ=0.3)")
    a.set_xlabel("trial"); a.set_title("perseveration kernel", fontsize=6); a.set_ylim(-0.05, 1.35); a.legend(frameon=False, fontsize=4.5, loc="upper center", ncol=2)
    a = fig.add_subplot(gs[1, 3]); a.axis("off")
    a.text(0, 1, "Model space: core + any of\nG, M, N, Z, P  →  2⁵ = 32 models\n\nSelection: leave-one-session-out\nheld-out log-likelihood (total and\n"
                 "day 0), random-effects model\nselection, component families.\n\nValidation first: model and\nparameter recovery on real\n"
                 "trial sequences.", fontsize=5.5, va="top")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"010_design_schematic.{ext}", dpi=300, bbox_inches="tight")
    print("saved")


if __name__ == "__main__":
    main()
