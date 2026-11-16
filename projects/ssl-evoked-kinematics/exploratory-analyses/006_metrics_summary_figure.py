"""006 -- Summary figure of the per-mouse evoked-movement metrics of 000, day 0 and expert side by side (no recomputation).

Inputs: tables/000_{day0,expert}_metrics_mouse.csv (per-mouse medians over hit trials, written by 000).
Metrics shown: first-reach latency, jaw-opening onset latency, reach duration, maximal tongue extension, peak tongue
speed, whisker sweep amplitude (0 .. 0.3 s). Whisker and nose onset latencies are not shown: on whisker trials the
whisker onset is dominated by the stimulus deflection itself.
Layout: one panel per metric; within a panel, day 0 then expert, each with R+ whisker, R+ auditory, R- whisker,
R- auditory hits (R+ #00B400, R- #C800C8; whisker = filled circles, auditory = open squares). Dots = mice, black =
mean +- SEM across mice. No statistics on the figure (they are in tables/000_<stage>_stats.csv).
Outputs (combined_results_ks4/ssl-evoked-kinematics/figures/): 006_metrics_summary.{png,pdf}
Run (haas, after 000): python 006_metrics_summary_figure.py
"""
import importlib.util
import pathlib

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('f001', HERE / '001_trajectory_figures.py')
f1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f1)
k = f1.k

METRICS = [('reach_latency', 'first reach latency (s)'), ('lat_jaw', 'jaw onset latency (s)'),
           ('reach_duration', 'reach duration (s)'), ('reach_max_ext', 'max tongue extension (mm)'),
           ('reach_peak_speed', 'peak tongue speed (mm/s)'), ('wh_amp', 'whisker sweep amplitude (deg)')]
GROUPS = [('R+', 'whisker'), ('R+', 'auditory'), ('R-', 'whisker'), ('R-', 'auditory')]
STAGE_SHORT = {'day0': 'day 0', 'expert': 'expert'}


def main():
    tabs = {st: pd.read_csv(k.OUT / 'tables' / f'000_{st}_metrics_mouse.csv') for st in k.STAGES}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 6.0))
    rng = np.random.default_rng(0)
    for ax, (col, lab) in zip(axs.flat, METRICS):
        ticks = []
        for si, st in enumerate(k.STAGES):
            t = tabs[st]
            for gi, (c, mod) in enumerate(GROUPS):
                x = si * 5 + gi
                v = t[(t.cohort == c) & (t['mod'] == mod)][col].dropna()
                filled = mod == 'whisker'
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(v)), v, s=5, marker='o' if filled else 's',
                           facecolor=k.COHORT_COLOR[c] if filled else 'none', edgecolor=k.COHORT_COLOR[c], lw=0.4,
                           alpha=0.5, zorder=2)
                if len(v):
                    ax.errorbar(x + 0.32, v.mean(), v.sem(), fmt='o' if filled else 's', color='k',
                                mfc='k' if filled else 'w', ms=3.5, lw=1, zorder=3)
            ticks.append((si * 5 + 1.5, STAGE_SHORT[st]))
        ax.set_xticks([p for p, _ in ticks], [s for _, s in ticks])
        ax.set_xlim(-0.7, 8.8)
        lo, hi = np.nanpercentile(pd.concat([tabs[s][col] for s in tabs]).dropna(), [0.5, 99.5])
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_ylabel(lab)
        ax.set_box_aspect(1)
    h = [plt.Line2D([], [], ls='', marker='o' if m == 'whisker' else 's', mfc=k.COHORT_COLOR[c] if m == 'whisker'
                    else 'none', mec=k.COHORT_COLOR[c], label=f'{c} {m} hit') for c, m in GROUPS]
    fig.legend(handles=h, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f1.wrap('Evoked-movement metrics per mouse (median over hit trials), day 0 and expert. Dots = mice '
                         '(filled = whisker hits, open = auditory hits); black = mean +- SEM across mice. Points '
                         'beyond the 0.5-99.5th percentile are outside the axis', fig), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=2.0)
    f1.save(fig, '006_metrics_summary')


if __name__ == '__main__':
    main()
