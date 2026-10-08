"""161 -- Vector schematic of the two coding directions in neural state space (user 2026-10-08: "schematic of the coding direction
definition in neural state space, 3D with three axes for three neurons, curves for the trial types and the vector linking the two;
vectorized and editable, text as text").
Illustration only (no data): three neurons as axes; condition-mean trajectories as curves; the analysis window highlighted; the
coding direction u joins the means of the two defining classes in that window; other events are projected on u.
  a  reward-lick axis (pre-lick): spontaneous lick (SL) and auditory hit (AH) define u; whisker hits early / late in the session
     are projected on it
  b  hit - miss axis (stimulus onset): active whisker hits and misses define u; passive whisker responses before / after the task
     are projected on it
Outputs (SVG text as text, PDF TrueType): figures/publication/161_state_space_schematic.{svg,pdf,png}
Run (anywhere): python 161_state_space_schematic.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7})
OUT = Path(__file__).resolve().parent / "figures" / "publication"
WC, NEG, POS, RP, RM, GR = "#f7b519", "#262626", "#2c2cdb", "#00B400", "#C800C8", "0.55"


def traj(start, end, bend, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.asarray(start) + 2 * (1 - t) * t * np.asarray(bend) + t ** 2 * np.asarray(end)


def neuron_axes(ax, L=1.25):
    for v, lab in zip(np.eye(3) * L, ("neuron 1", "neuron 2", "neuron 3")):
        ax.quiver(0, 0, 0, *v, color="0.35", lw=0.8, arrow_length_ratio=0.06)
        ax.text(*(v * 1.07), lab, fontsize=6.5, color="0.3", ha="center")
    ax.set_axis_off(); ax.set_xlim(0, 1.15); ax.set_ylim(0, 1.15); ax.set_zlim(0, 1.15)
    ax.view_init(elev=20, azim=40)
    ax.set_box_aspect((1, 1, 1), zoom=1.12)


def curve(ax, P, col, lw=1.2, frac=0.25, ls="-", label=None, lab_off=(0, 0, 0.05)):
    k = int(len(P) * (1 - frac))
    ax.plot(*P[:k + 1].T, color=col, lw=lw * 0.7, alpha=0.55, ls=ls)
    ax.plot(*P[k:].T, color=col, lw=lw * 2.2, ls=ls, solid_capstyle="round")       # analysis window
    m = P[k:].mean(0)
    ax.scatter(*m, color=col, s=22, depthshade=False, edgecolor="white", linewidth=0.5, zorder=5)
    if label:
        ax.text(*(m + np.asarray(lab_off)), label, color=col, fontsize=6.5, ha="center")
    return m


def project(ax, p, a, b, col, txt=None):
    d = b - a; t = (p - a) @ d / (d @ d); q = a + t * d
    ax.plot(*np.c_[p, q], color=col, lw=0.7, ls=(0, (2, 1.5)))
    ax.scatter(*q, color=col, s=10, marker="D", depthshade=False, zorder=6)
    if txt:
        ax.text(*(q + np.r_[0, 0, -0.09]), txt, color=col, fontsize=6, ha="center")
    return t


def axis_arrow(ax, a, b, lab):
    d = b - a
    ax.quiver(*a, *(d * 1.12), color="black", lw=1.6, arrow_length_ratio=0.1)
    ax.text(*(a + d * 1.2 + np.r_[0, 0, 0.03]), lab, fontsize=8, weight="bold")


def main():
    fig = plt.figure(figsize=(7.2, 3.8))
    base = np.r_[0.15, 0.15, 0.1]
    # ---------------------------------------------------------------- a: reward-lick axis
    ax = fig.add_subplot(1, 2, 1, projection="3d"); neuron_axes(ax)
    S = curve(ax, traj(base, [0.95, 0.2, 0.35], [0.6, 0.1, 0.15]), NEG, label="SL", lab_off=(0.06, 0, -0.1))
    A = curve(ax, traj(base, [0.3, 0.95, 0.85], [0.15, 0.6, 0.4]), POS, label="AH", lab_off=(0, 0, 0.08))
    W1 = curve(ax, traj(base, [0.85, 0.35, 0.95], [0.5, 0.2, 0.55]), WC, ls=(0, (3, 1.5)), label="WH early", lab_off=(0.04, 0, 0.08))
    W2 = curve(ax, traj(base, [0.55, 0.75, 1.05], [0.35, 0.4, 0.65]), WC, label="WH late (R+)", lab_off=(0, 0, 0.08))
    axis_arrow(ax, S, A, "u")
    project(ax, W1, S, A, "#b07e00"); project(ax, W2, S, A, "#b07e00")
    ax.text(*(base + np.r_[0, 0, -0.08]), "trial start", fontsize=6, color=GR, ha="center")
    ax.set_title("a  Reward-lick axis (pre-lick window)\n    SL = spontaneous lick, AH = auditory hit, WH = whisker hit", fontsize=8, loc="left", pad=18)
    # ---------------------------------------------------------------- b: hit - miss axis
    ax = fig.add_subplot(1, 2, 2, projection="3d"); neuron_axes(ax)
    Mi = curve(ax, traj(base, [0.95, 0.25, 0.4], [0.6, 0.15, 0.2]), "0.6", lw=1.0, label="active miss", lab_off=(0.06, 0, -0.1))
    Hi = curve(ax, traj(base, [0.3, 0.9, 0.9], [0.15, 0.55, 0.45]), "0.15", lw=1.0, label="active hit", lab_off=(0, 0, 0.08))
    Pp = curve(ax, traj(base, [0.5, 0.55, 1.05], [0.3, 0.3, 0.6]), WC, ls=(0, (3, 1.5)), label="passive pre", lab_off=(-0.02, 0, 0.08))
    Po = curve(ax, traj(base, [0.85, 0.3, 0.85], [0.5, 0.2, 0.5]), WC, label="passive post (R−)", lab_off=(0.08, 0, 0.08))
    axis_arrow(ax, Mi, Hi, "u")
    project(ax, Pp, Mi, Hi, "#b07e00"); project(ax, Po, Mi, Hi, "#b07e00")
    ax.text(*(base + np.r_[0, 0, -0.08]), "stimulus onset", fontsize=6, color=GR, ha="center")
    ax.set_title("b  Hit − miss axis (5–35 ms after stimulus)\n    active whisker trials define u; passive responses projected", fontsize=8, loc="left", pad=18)
    fig.text(0.5, 0.02, "curves: condition-mean population trajectories; thick: analysis window; dots: window means\n"
             "u: unit vector from the mean of the − class to the mean of the + class; diamonds: projections on u",
             ha="center", fontsize=6.2)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.8, bottom=0.1, wspace=0.05)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"161_state_space_schematic.{ext}", dpi=300)
    print("ALL DONE", OUT / "161_state_space_schematic.svg")


if __name__ == "__main__":
    main()
