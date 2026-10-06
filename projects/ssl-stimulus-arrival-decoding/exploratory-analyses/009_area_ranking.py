"""009 -- Which areas decode whisker vs auditory better? (user 2026-10-06) Two figures, areas ordered by decreasing value,
area groups (top row) and fine areas (bottom row), passive trials (left) and task trials (right), N = 200:
  area_peak_accuracy  -- peak corrected balanced accuracy: maximum over the post-stimulus wide bins (causal 50-ms bins,
                         5-ms steps, ending 5..600 ms) of the mean over iterations; error bar: 2.5-97.5 percentiles over
                         iterations in that bin; the time of the peak is printed next to each bar;
  area_onset_accuracy -- corrected balanced accuracy in the onset bin (zoom resolution, 20-ms bins, 2-ms steps; onset
                         rule of 004); error bar as above; unreliable onsets (004 rule) hatched with a dagger; areas
                         without an onset are omitted.
Source per epoch and level: the final N = 200 run when complete, else the N-sweep run (006 source()).
Output: <home>/figures/area_peak_accuracy.*, area_onset_accuracy.*, <home>/tables/area_ranking.csv
"""
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
S = importlib.import_module("_style")
AR = importlib.import_module("_areas")
m4 = importlib.import_module("004_main_figures")
m6 = importlib.import_module("006_active_vs_passive")
HOME = AR.HOME
N_MAIN = 200
EPOCHS = ("passive", "active")
EW = {"passive": "passive trials", "active": "task (active) trials"}
LEVELS = ("area_group", "area_acronym_custom")


def metrics(epoch, level):
    folder, kind = m6.source(epoch, level)
    b = pd.read_parquet(folder / "summary_bins.parquet")
    b = b[(b.level == level) & (b.N == N_MAIN)]
    ob = pd.read_csv(folder / "onset_bootstrap_N200.csv")
    ob = ob[ob.level == level].assign(flag=lambda d: m4.unreliable(d)).set_index("area")
    rows = []
    for a, g in b.groupby("area"):
        w = g[(g.resolution == "wide") & (g.t > 0.004)].sort_values("t")
        k = w["mean"].idxmax()
        z = g[g.resolution == "zoom"]
        on = ob.onset_ms.get(a, np.nan)
        zo = z[np.isclose(1000 * z.t, on)] if np.isfinite(on) else z.iloc[:0]
        rows.append(dict(epoch=epoch, level=level, area=a, source=kind, n_iter=int(w.n_iter.iloc[0]),
                         peak=w.loc[k, "mean"], peak_lo=w.loc[k, "lo"], peak_hi=w.loc[k, "hi"], peak_t_ms=1000 * w.loc[k, "t"],
                         onset_ms=on, onset_flag=bool(ob.flag.get(a, True)),
                         onset_acc=zo["mean"].iloc[0] if len(zo) else np.nan,
                         onset_acc_lo=zo["lo"].iloc[0] if len(zo) else np.nan,
                         onset_acc_hi=zo["hi"].iloc[0] if len(zo) else np.nan))
    return pd.DataFrame(rows)


def panel(ax, T, key, col, note, flag=None):
    q = T.dropna(subset=[key]).sort_values(key, ascending=False).reset_index(drop=True)
    y = np.arange(len(q))
    ax.barh(y, q[key], color=[col.get(a, "0.5") for a in q.area], height=0.72, lw=0)
    if flag is not None:
        for yi, r in q.iterrows():
            if r[flag]:
                ax.barh(yi, r[key], color="white", alpha=0.6, height=0.72, lw=0)
                ax.barh(yi, r[key], color="none", edgecolor=col.get(r.area, "0.5"), hatch="////", height=0.72, lw=0.4)
    ax.errorbar(q[key], y, xerr=[q[key] - q[f"{key}_lo"], q[f"{key}_hi"] - q[key]], fmt="none", ecolor="0.25", lw=0.5)
    fs = 4.2 if len(q) > 20 else 5
    ax.set_yticks(y, [S.short(a) for a in q.area], fontsize=fs)
    for tl, a in zip(ax.get_yticklabels(), q.area):
        tl.set_color(col.get(a, "k"))
    for yi, r in q.iterrows():
        ax.text(r[f"{key}_hi"] + 0.008, yi, note(r), va="center", fontsize=fs - 0.6, color="0.35")
    ax.set_ylim(len(q) - 0.5, -0.6)
    ax.set_xlim(0, 0.62)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0.5, color="0.6", lw=0.4, ls=":")


def figure(plt, T, key, name, title, xlabel, note, flag=None):
    nG = T[T.level == "area_group"].area.nunique()
    nF = T[T.level == "area_acronym_custom"].area.nunique()
    H = 0.095 * (nG + nF) + 1.6
    fig = plt.figure(figsize=(S.W_IN, H))
    gs = fig.add_gridspec(2, 2, height_ratios=[nG + 2, nF + 2], wspace=0.45, hspace=0.12 + 0.6 / H, left=0.13, right=0.98,
                          top=1 - 0.55 / H, bottom=0.45 / H)
    col = AR.colors()
    axs = []
    for r, level in enumerate(LEVELS):
        for c, e in enumerate(EPOCHS):
            ax = fig.add_subplot(gs[r, c])
            q = T[(T.level == level) & (T.epoch == e)]
            panel(ax, q, key, col, note, flag)
            n_iter = int(q.n_iter.iloc[0])
            ax.set_title(f"{m4.LEVEL_NAME[level].capitalize()}, {EW[e]} ({n_iter} iterations)", loc="left", fontsize=5.6)
            ax.set_xlabel(xlabel if r == 1 else "")
            axs.append(ax)
    S.letter_row(fig, axs[:2], "ab"); S.letter_row(fig, axs[2:], "cd")
    fig.suptitle(title, x=0.02, y=1 - 0.05 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, HOME / "figures", name)
    plt.close(fig)


def main():
    plt = S.setup()
    T = pd.concat([metrics(e, lv) for e in EPOCHS for lv in LEVELS], ignore_index=True)
    (HOME / "tables").mkdir(exist_ok=True)
    T.to_csv(HOME / "tables" / "area_ranking.csv", index=False)
    figure(plt, T, "peak", "area_peak_accuracy",
           f"Whisker vs auditory decoding: peak accuracy per area (N = {N_MAIN}), ordered by decreasing peak",
           "Peak corrected balanced accuracy (50-ms bins)", lambda r: f"{r.peak_t_ms:.0f} ms")
    U = T.rename(columns={"onset_acc": "oacc", "onset_acc_lo": "oacc_lo", "onset_acc_hi": "oacc_hi"})
    figure(plt, U, "oacc", "area_onset_accuracy",
           f"Whisker vs auditory decoding: accuracy at onset per area (N = {N_MAIN}), ordered by decreasing value",
           "Corrected balanced accuracy in the onset bin (20-ms bins)",
           lambda r: f"{r.onset_ms:.0f} ms" + (" †" if r.onset_flag else ""), flag="onset_flag")
    print(T.groupby(["epoch", "level"])[["peak", "onset_acc"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
