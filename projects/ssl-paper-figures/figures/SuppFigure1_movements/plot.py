"""SuppFigure1_movements -- one plot_<key>(ax, df) per panel; style and helpers from the Figure 1 plot module."""

import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("fig1_plot", Path(__file__).resolve().parents[1] / "Figure1_behaviour" / "plot.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

LABELS = {**fig1.HITMISS_LABELS, "tongue_out": ("Tongue out", "Δ P"), "jaw_opening": ("Jaw distance", "Δ mm"),
          "nose_distance": ("Nose distance", "Δ mm"), "tongue_distance": ("Tongue distance", "mm")}
STIM_COLS = [("whisker_trial", "Whisker", ("Hit", "Miss")), ("auditory_trial", "Auditory", ("Hit", "Miss")),
             ("no_stim_trial", "No stimulus", ("FA", "CR"))]
LICK_TYPES = [("whisker_trial", "W hit"), ("auditory_trial", "A hit"), ("no_stim_trial", "FA")]


def _bars(a, d, y, label):
    """R+ vs R- cluster bars of one condition, grey to black by P, stars and P above (Figure 1 _shaded_bars); the
    condition is named once at the left end of its row (user 2026-11-22: less text over the traces)."""
    on = d[d.cluster_sig.astype(bool)].drop_duplicates("time_s")
    if len(on):
        step = float(np.min(np.diff(np.sort(d.time_s.unique()))))
        fig1._shaded_bars(a, on, "time_s", y, step, lw=1)
        a.text(1.01, y, label, transform=a.transAxes, ha="left", va="center", fontsize=fig1.FS_NOTE, color="0.3")


def plot_evoked(ax, df):
    """User 2026-11-22: per signal, stimulus-aligned whisker / auditory / no-stimulus trials (lick solid, no lick dashed;
    both cohorts, group_color) and lick-aligned licked trials (W hits, A hits, false alarms), mean +- s.e.m. over mice.
    One y range per signal. Bars, R+ vs R- clusters (top: lick / W hit, below: no lick / A hit / FA), stars and P above."""
    sig = df.signal.iloc[0]
    label, unit = LABELS[sig]
    axes = fig1._row_axes(ax, 4, gap_mm=[7.0, 7.0, 11.0], square=False)
    stim, lick = df[df["align"] == "stimulus"], df[df["align"] == "lick"]
    if not len(stim):  # user 2026-11-23: tongue distance, lick-aligned only
        for a in axes[:3]:
            a.axis("off")
        axes = axes[3:]
    for a, (tt, title, outcomes) in zip(axes[:3] if len(stim) else [], STIM_COLS):
        d = stim[stim.trial_type == tt]
        for lk, ls in ((1, "-"), (0, "--")):
            for c in fig1.COHORTS:
                dd = d[(d.lick == lk) & (d.cohort == c)]
                if dd.mouse_id.nunique() >= 3:
                    fig1._mean_band(a, dd, "time_s", "value", fig1.group_color(tt, c), ls=ls, lw=1.0, fill=0.8)  # user 2026-11-23: thicker
        a.set_title(title, fontsize=fig1.FS_TITLE, pad=14)
        a.axvline(0, color="0.6", lw=0.4, ls=":")
        a.set_xlim(d.time_s.min(), d.time_s.max())
        a.set_xlabel("Time from stimulus (s)", labelpad=1)
    a = axes[-1]
    for tt, _ in LICK_TYPES:
        for c in fig1.COHORTS:
            dd = lick[(lick.trial_type == tt) & (lick.cohort == c) & (lick.lick == 1)]
            if dd.mouse_id.nunique() >= 3:
                fig1._mean_band(a, dd, "time_s", "value", fig1.group_color(tt, c), lw=1.0, fill=0.8)
    a.set_title("Licked trials", fontsize=fig1.FS_TITLE, pad=14)
    a.axvline(0, color="0.6", lw=0.4, ls=":")
    a.set_xlim(lick.time_s.min(), lick.time_s.max())
    a.set_xlabel("Time from first lick (s)", labelpad=1)
    lo = min(x.get_ylim()[0] for x in axes)
    hi = max(x.get_ylim()[1] for x in axes)
    for k, a in enumerate(axes):
        a.set_ylim(lo, hi + 0.8 * (hi - lo))  # room for the cluster bars, stars and P
        a.axhline(0, color="0.8", lw=0.3)
        a.xaxis.set_major_locator(fig1.mpl.ticker.MaxNLocator(3))
        a.yaxis.set_major_locator(fig1.mpl.ticker.MaxNLocator(3))
        if k in (1, 2) and len(stim):
            a.set_yticklabels([])
    axes[0].set_ylabel(f"{label}\n({unit})", labelpad=1)  # two lines: clear of the panel letter
    if len(stim):
        for a, (tt, _, outcomes) in zip(axes[:3], STIM_COLS):
            d = stim[stim.trial_type == tt]
            _bars(a, d[d.lick == 1], 0.74, outcomes[0])
            _bars(a, d[d.lick == 0], 0.64, outcomes[1])
        if sig == "whisker_speed":  # user 2026-11-23: legends once, in the first row (room above the traces)
            fig1._legend(axes[2], [("Lick", "0.2"), ("No lick", "0.2", dict(ls="--"))], loc="center left",
                         bbox_to_anchor=(0, 1.0))
    for (tt, name), y in zip(LICK_TYPES, (0.74, 0.64, 0.54)):
        _bars(axes[-1], lick[(lick.trial_type == tt) & (lick.lick == 1)], y, name)
    if sig == "whisker_speed":
        fig1._legend(axes[-1], [(n, fig1.group_color(tt, "R+")) for tt, n in LICK_TYPES], loc="upper left",
                     bbox_to_anchor=(0, 1.0))  # top left, in the headroom above the traces; lighter colours, R-


plot_evoked_whisker_speed = plot_evoked_jaw_opening = plot_evoked_tongue_out = plot_evoked
plot_evoked_nose_speed = plot_evoked_pupil_area = plot_evoked_nose_distance = plot_evoked_tongue_distance = plot_evoked
