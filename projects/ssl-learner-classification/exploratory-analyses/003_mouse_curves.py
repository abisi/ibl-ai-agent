"""003 -- Whisker day-0 learning curves of every sheet 'bad' mouse and of every mouse whose agnostic label (001) disagrees
with the sheet (user 2026-11-19: "make a figure showing the learning curves for all these mice").

Per mouse, as Fig. 1E: exact learning curves (Figure 1 session_curves) of whisker trials (cohort colour, 80% posterior
interval), no-stimulus trials (chance, black) and auditory trials (blue) on whisker trials; ticks, whisker hits (top) and
misses (bottom). Title: mouse, sheet category, criteria met (of 9) and the agnostic label; day-0 and day +1 d'.
Groups (rows of the figure): R+ bad fixed by the user, R+ bad flexible, R+ good/moderate called non-learners, R- bad,
R- good/moderate called non-learners.
Outputs: combined_results_ks4/ssl-learner-classification/figures/003_mouse_curves.{png,pdf}.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/003_mouse_curves.py
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PF = REPO / "projects" / "ssl-paper-figures"
sys.path.insert(0, str(PF))
sys.path.insert(0, str(REPO / "scripts"))
import paperstyle as ps  # noqa: E402
import shared_behaviour  # noqa: E402
from axel_bisi_paths import axel_bisi_root  # noqa: E402

_spec = importlib.util.spec_from_file_location("fig1_compute", PF / "figures" / "Figure1_behaviour" / "compute.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

RES = axel_bisi_root() / "combined_results_ks4" / "ssl-learner-classification"
FIXED_BAD = ["MH020", "MH013", "AB142", "MH068", "AB151", "AB117", "AB107"]  # as 001
FLEX = ["AB141", "MH007", "AB104", "MH026", "MH009"]
C = ps.COLORS
NCOL = 4
MM = 1 / 25.4


def groups(pm):
    pm = pm.set_index("mouse_id")
    disagree = pm[pm.learner.astype(str) != pm.sheet_learner.astype(str)]
    by = lambda ids: list(pm.loc[ids].sort_values("consensus", ascending=False).index)  # noqa: E731
    gm = lambda c: by(list(disagree[(disagree.cohort == c) & (disagree.learning_category != "bad")].index))  # noqa: E731
    return [("R+ bad, fixed non-learners", by(FIXED_BAD)),
            ("R+ bad, flexible", by(FLEX)),
            ("R+ good / moderate called non-learners by the agnostic label", gm("R+")),
            ("R− bad", by(list(pm[(pm.cohort == "R-") & (pm.learning_category == "bad")].index))),
            ("R− good / moderate called non-learners by the agnostic label", gm("R-"))], pm


def draw(ax, cur, cohort):
    col = C[cohort]
    x = cur.whisker_trial.to_numpy()
    ax.fill_between(x, cur.p_low, cur.p_high, color=ps.light(col, 0.8), lw=0, zorder=1)
    ax.plot(x, cur.p_mean, color=col, lw=0.7, zorder=3)
    ax.plot(x, cur.p_chance, color=C["no_stim"], lw=0.5, zorder=2)
    if np.isfinite(np.asarray(cur.p_auditory, float)).any():
        ax.plot(x, cur.p_auditory, color=ps.TRIAL_TYPE_COLORS["auditory_trial"], lw=0.5, zorder=2)
    lick = cur.lick.to_numpy() == 1
    ax.scatter(x[lick], np.full(lick.sum(), 1.07), s=1.2, marker="|", lw=0.3, color=col)
    ax.scatter(x[~lick], np.full((~lick).sum(), -0.07), s=1.2, marker="|", lw=0.3, color="0.6")
    ax.set_ylim(-0.12, 1.12)
    ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_xlim(0, x.max() + 1)
    ax.tick_params(labelsize=5, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    pm = pd.read_csv(RES / "tables" / "001_per_mouse.csv")
    grp, pm = groups(pm)
    d0 = fig1.day0_table(shared_behaviour.load())
    rows_per = [int(np.ceil(len(ids) / NCOL)) for _, ids in grp]
    panel_h, head_h, w = 24, 7, 183
    H = sum(r * panel_h + head_h for r in rows_per) + 6
    fig = plt.figure(figsize=(w * MM, H * MM))
    y = 3
    for (title, ids), nr in zip(grp, rows_per):
        fig.text(8 * MM / (w * MM), 1 - y / H, title, fontsize=7, fontweight="bold", va="top")
        y += head_h
        for k, m in enumerate(ids):
            r, c = divmod(k, NCOL)
            left, bottom = (10 + c * 44) / w, 1 - (y + r * panel_h + panel_h - 6) / H
            ax = fig.add_axes([left, bottom, 34 / w, (panel_h - 10) / H])
            st = d0[d0.mouse_id == m]
            cur, _, _ = fig1.session_curves(st)
            p = pm.loc[m]
            draw(ax, cur, p.cohort)
            lab = "learner" if str(p.learner) == "True" else "non-learner"
            # peak within-session d' (user 2026-11-23): dotted line at the trial of the peak
            dpc = np.clip(cur.p_mean.to_numpy(float), 0.01, 0.99), np.clip(cur.p_chance.to_numpy(float), 0.01, 0.99)
            dpc = stats.norm.ppf(dpc[0]) - stats.norm.ppf(dpc[1])
            ax.axvline(cur.whisker_trial.to_numpy()[int(np.argmax(dpc))], color="0.3", lw=0.4, ls=":", zorder=0)
            ax.set_title(f"{m} ({p.cohort.replace('-', '−')}, {p.learning_category}) · {int(round(p.consensus * 9))}/9 → {lab}\n"
                         f"d′ day 0 {p.dprime:.2f}, day +1 {p.dprime_day1:.2f}, peak {p.peak_dprime:.2f}", fontsize=5,
                         pad=2, loc="left")
            if c == 0:
                ax.set_ylabel("P(lick)", fontsize=5, labelpad=1)
            if r == nr - 1 or k + NCOL >= len(ids):
                ax.set_xlabel("Whisker trial", fontsize=5, labelpad=1)
        y += nr * panel_h
    handles = [plt.Line2D([], [], color="0.3", lw=0.8, label="Whisker (cohort colour)"),
               plt.Line2D([], [], color=C["no_stim"], lw=0.8, label="Chance (no stimulus)"),
               plt.Line2D([], [], color=ps.TRIAL_TYPE_COLORS["auditory_trial"], lw=0.8, label="Auditory")]
    fig.legend(handles=handles, loc="upper right", fontsize=5, frameon=False, ncol=3, bbox_to_anchor=(0.99, 1 - 2 / H))
    out = RES / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"003_mouse_curves.{ext}", dpi=300)
    print(out / "003_mouse_curves.png", f"{H:.0f} mm")


if __name__ == "__main__":
    main()
