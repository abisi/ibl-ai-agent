"""Figure 4E candidates (user 2026-11-26: "remove z-score? split / combined stim types, show jaw speed, show until 60 ms only;
have several variants in _explore").

Data: 207 (tables/207_passive_whisker_jaw_traces.parquet): whisker speed (deg/s, top view), jaw distance (mm, side view), jaw
speed (mm/s); DLC via ephys_utilities dlc_helpers; passive trials of the Figure 4 pipeline, learners, whisker day 0; each trial
minus its -0.2..0 s mean; per session and epoch; stim W, A or both pooled. Plotted: change post - pre, R+ and R- (mean +-
s.e.m. over sessions), -50..60 ms, grey band 5-35 ms; p = R+ vs R- on the 5-35 ms mean (larger of Mann-Whitney / Welch).
Two pages: physical units, and z (s.d. of the session's passive windows).
Output: Figures/_explore/Figure4_passive_cd_<date>/movement/movement_variants_{raw,z}.{png,pdf,svg}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_movement_variants.py
"""
import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import shared_passive_cd as S  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402
from paperstyle import MM, light, save_figure, set_nature_rc  # noqa: E402

FS = 5
COH = ["R+", "R-"]
COL = {"R+": "#00B400", "R-": "#C800C8"}
SIGS = [("whisker_speed", "Whisker speed", "°/s"), ("jaw_distance", "Jaw opening", "mm"), ("jaw_speed", "Jaw speed", "mm/s")]
STIMS = [("W", "Whisker trials"), ("A", "Auditory trials"), ("all", "Both pooled")]


def page(t, col, unit_fn, name, out):
    fig, axes = plt.subplots(3, 3, figsize=(150 * MM, 120 * MM))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.9, wspace=0.5, hspace=0.75)
    w = t.pivot_table(index=["session_id", "cohort", "signal", "stim", "time_s"], columns="epoch", values=col).reset_index()
    w["d"] = w["post"] - w["pre"]
    w = w[(w.time_s >= -0.05 - 1e-9) & (w.time_s <= 0.06 + 1e-9)]
    for i, (s, lab, unit) in enumerate(SIGS):
        for j, (st, slab) in enumerate(STIMS):
            a = axes[i, j]
            g = w[(w.signal == s) & (w.stim == st)]
            gg = g.groupby(["cohort", "time_s"]).d.agg(["mean", "sem"]).reset_index()
            for c in COH:
                h = gg[gg.cohort == c]
                a.fill_between(h.time_s * 1000, h["mean"] - h["sem"], h["mean"] + h["sem"], color=light(COL[c], 0.7), lw=0,
                               edgecolor="none")
                a.plot(h.time_s * 1000, h["mean"], color=COL[c], lw=0.7, label=c.replace("-", "−"))
            q = g[(g.time_s >= 0.005 - 1e-9) & (g.time_s <= 0.035 + 1e-9)].groupby(["session_id", "cohort"]).d.mean()
            q = q.reset_index().dropna()
            x, y = q[q.cohort == "R+"].d, q[q.cohort == "R-"].d
            p = max(stats.mannwhitneyu(x, y).pvalue, stats.ttest_ind(x, y, equal_var=False).pvalue)
            a.axvspan(5, 35, facecolor="0.93", lw=0, edgecolor="none", zorder=0)
            a.axvline(0, color="0.4", lw=0.4, ls=(0, (2, 2)))
            a.axhline(0, color="0.6", lw=0.4, ls=(0, (2, 2)))
            a.set_xlim(-50, 60)
            a.set_xticks([0, 50])
            a.text(0.98, 0.97, f"5–35 ms: {S.p_text(p)}", transform=a.transAxes, ha="right", va="top", fontsize=FS - 1)
            a.set_title(f"{lab}, {slab.lower()}", fontsize=FS + 0.5, pad=2)
            a.set_xlabel("Time from stimulus (ms)", fontsize=FS)
            a.set_ylabel(f"Δ {lab.lower()}\n({unit_fn(unit)}, post − pre)", fontsize=FS)
            a.tick_params(labelsize=FS, length=1.5)
            if i == 0 and j == 0:
                a.legend(fontsize=FS - 0.5, frameon=False, loc="lower left", handlelength=1)
    fig.suptitle(f"Evoked movement change, post − pre ({name}); mean ± s.e.m. over sessions", fontsize=FS + 1.5)
    save_figure(fig, out, f"movement_variants_{name}")


def main():
    set_nature_rc()
    t = pd.read_parquet(S.TAB / "207_passive_whisker_jaw_traces.parquet")
    t = t[t.mouse_id.isin(S.learners())]
    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}" / "movement"
    out.mkdir(parents=True, exist_ok=True)
    page(t, "raw_bc", lambda u: u, "raw", out)
    page(t, "z_bc", lambda u: "z", "z", out)
    print("saved", out, "sessions", t.session_id.nunique())


if __name__ == "__main__":
    main()
