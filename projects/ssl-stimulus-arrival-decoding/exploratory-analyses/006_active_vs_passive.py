"""006 -- Task (active) vs passive trials: does stimulus-modality information arrive at the same time, and with the same
strength, when the stimuli are delivered in the task and outside it? One figure per level (area groups / areas).

Source per epoch: the final N = 200 run (<home>/<epoch>_final_n200/, 500 iterations x 20 shuffles) when it covers every
area of the level, otherwise the N-sweep run (<home>/<epoch>/, 100 x 10, N = 200 rows); the source is printed in the
figure title and recorded in the provenance.
  a -- onset per area, task (active, filled) and passive (open) trials (N = 200), 95 % range over resamples of the
       iterations; rows sorted by the active onset; grey rows: onset unreliable in either epoch (004 rule), not tested;
  b -- mean corrected accuracy 5-50 ms, same layout, 95 % range over iterations;
  c -- pre-stimulus corrected accuracy (mean over the 50-ms bins ending -150..0 ms), same layout: expected 0;
  d -- first 50 ms (20-ms bins, 2-ms steps) of the 6 best-sampled areas, active solid, passive dashed (mean +- s.d.);
  e -- the same over -200..600 ms (50-ms bins, 5-ms steps).
Tests over areas (paired, n = areas): Wilcoxon signed-rank and paired t-test on passive - active (onset in ms, accuracy,
baseline).
Output: <home>/figures/active_vs_passive_<level>.{png,pdf,svg}, <home>/tables/active_vs_passive_<level>.csv,
<home>/tables/active_vs_passive_tests.csv, <home>/provenance_006.json
"""
import importlib
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
S = importlib.import_module("_style")
AR = importlib.import_module("_areas")
m4 = importlib.import_module("004_main_figures")
HOME = AR.HOME
N_MAIN, N_SHOW = 200, 6
BASE = (-0.150, 0.0)        # pre-stimulus window: wide bins whose end lies in -150..0 ms (causal 50-ms bins: no post-stimulus spikes)
EPOCHS = ("active", "passive")
EPOCH_C = {"active": "k", "passive": "0.55"}


def source(epoch, level):
    """final folder if it covers every area of the level, else the N-sweep folder"""
    fin, sweep = HOME / f"{epoch}_final_n200", HOME / epoch
    need = set(AR.LEVELS[level])
    f = fin / "onset_bootstrap_N200.csv"
    if f.exists() and (fin / "summary_bins.parquet").exists():
        ob = pd.read_csv(f)
        if need <= set(ob[ob.level == level].area):
            return fin, "final"
    return sweep, "sweep"


def load(epoch, level):
    folder, kind = source(epoch, level)
    ob = pd.read_csv(folder / "onset_bootstrap_N200.csv")
    ob = ob[ob.level == level].assign(flag=lambda d: m4.unreliable(d))
    w = pd.read_csv(folder / "window_accuracy.csv")
    w = w[(w.level == level) & (w.N == N_MAIN)]
    b = pd.read_parquet(folder / "summary_bins.parquet")
    b = b[(b.level == level) & (b.N == N_MAIN)]
    n_iter = int(w.n_iter.iloc[0])
    return dict(ob=ob.set_index("area"), w=w.set_index("area"), b=b, kind=kind, folder=str(folder), n_iter=n_iter)


def paired_tests(x, y):
    d = np.asarray(y) - np.asarray(x)
    ok = np.isfinite(d)
    d = d[ok]
    if len(d) < 3:
        return dict(n=len(d), mean_diff=np.nan, median_diff=np.nan, p_wilcoxon=np.nan, p_paired_t=np.nan)
    pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    pt = stats.ttest_1samp(d, 0).pvalue if np.std(d) > 0 else (1.0 if np.mean(d) == 0 else 0.0)
    return dict(n=int(len(d)), mean_diff=float(d.mean()), median_diff=float(np.median(d)), p_wilcoxon=float(pw),
                p_paired_t=float(pt))


def dumbbell(ax, T, key, col, fmt, flag=None, xlabel=""):
    """one row per area (T order): active filled, passive open, 95 % ranges as lines"""
    y = np.arange(len(T))
    for yi, (a, r) in zip(y, T.iterrows()):
        c = "0.75" if flag is not None and r[flag] else col.get(a, "0.4")
        for e, dy, mfc in (("active", -0.17, c), ("passive", 0.17, "white")):
            ax.plot([r[f"{key}_{e}_lo"], r[f"{key}_{e}_hi"]], [yi + dy] * 2, color=c, lw=0.6)
            ax.plot(r[f"{key}_{e}"], yi + dy, "o", ms=2.6, mfc=mfc, mec=c, mew=0.7, zorder=3)
    fs = 4.3 if len(T) > 20 else 5
    ax.set_yticks(y, [m4.label(a) for a in T.index], fontsize=fs)
    for tl, a in zip(ax.get_yticklabels(), T.index):
        tl.set_color("0.6" if flag is not None and T.loc[a, flag] else col.get(a, "k"))
    ax.set_ylim(len(T) - 0.5, -0.7)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color="0.92", lw=0.4)
    ax.set_axisbelow(True)


def test_text(ax, t, unit, fmt):
    """test result under the x label (never over the data)"""
    ax.set_xlabel(ax.get_xlabel() + f"\npassive - active: median {format(t['median_diff'], fmt)}{unit}, n = {t['n']}\n"
                  f"Wilcoxon {S.fmt_p(t['p_wilcoxon'])}; paired t {S.fmt_p(t['p_paired_t'])}", fontsize=5)


def curve(ax, b, area, res, c, ls):
    q = b[(b.area == area) & (b.resolution == res)].sort_values("t")
    t = 1000 * q.t.to_numpy()
    ax.fill_between(t, q["mean"] - q.sd, q["mean"] + q.sd, color=c, alpha=0.12, lw=0)
    ax.plot(t, q["mean"], color=c, ls=ls, lw=0.8)


def figure(plt, level, E, col):
    A, P = E["active"], E["passive"]
    areas = sorted(set(A["ob"].index) & set(P["ob"].index))
    T = pd.DataFrame(index=pd.Index(areas, name="area"))
    for e, X in E.items():
        T[f"onset_{e}"] = X["ob"].onset_ms.reindex(areas)
        T[f"onset_{e}_lo"] = X["ob"].lo.reindex(areas)
        T[f"onset_{e}_hi"] = X["ob"].hi.reindex(areas)
        T[f"flag_{e}"] = X["ob"].flag.reindex(areas).astype(bool)
        T[f"acc_{e}"] = X["w"]["mean"].reindex(areas)
        T[f"acc_{e}_lo"] = X["w"].lo.reindex(areas)
        T[f"acc_{e}_hi"] = X["w"].hi.reindex(areas)
        T[f"n_sessions_{e}"] = X["w"].n_eligible_sessions.reindex(areas)
        bw = X["b"][(X["b"].resolution == "wide") & (X["b"].t > BASE[0] - 1e-9) & (X["b"].t <= BASE[1] + 1e-9)]
        bm = bw.groupby("area")["mean"].mean()
        bsd = bw.groupby("area")["sd"].mean() / np.sqrt(X["n_iter"])
        T[f"base_{e}"] = bm.reindex(areas)
        T[f"base_{e}_lo"] = (bm - 1.96 * bsd).reindex(areas)
        T[f"base_{e}_hi"] = (bm + 1.96 * bsd).reindex(areas)
        T[f"source_{e}"] = X["kind"]
        T[f"n_iter_{e}"] = X["n_iter"]
    T["flag_any"] = T.flag_active | T.flag_passive
    rel = T[~T.flag_any]
    tests = [dict(level=level, measure="onset_ms", **paired_tests(rel.onset_active, rel.onset_passive)),
             dict(level=level, measure="accuracy_5_50ms", **paired_tests(T.acc_active, T.acc_passive)),
             dict(level=level, measure="baseline_-150_0ms", **paired_tests(T.base_active, T.base_passive))]
    (HOME / "tables").mkdir(exist_ok=True)
    T.reset_index().assign(level=level).to_csv(HOME / "tables" / f"active_vs_passive_{level}.csv", index=False)

    top = list(T.n_sessions_active.sort_values(ascending=False).index[:N_SHOW])
    top = sorted(top, key=lambda a: T.loc[a, "onset_active"])
    T = T.assign(_k=T.onset_active.fillna(1e9)).sort_values(["_k", "acc_active"], ascending=[True, False]).drop(columns="_k")
    W = S.W_IN
    h0 = max(2.4, 0.105 * len(T) + 0.7)
    H = h0 + 2.9
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(3, 1, height_ratios=[h0, 1.05, 1.05], hspace=0.62, left=0.12, right=0.98, top=1 - 0.55 / H,
                          bottom=0.35 / H)
    g0 = gs[0].subgridspec(1, 3, wspace=0.12)
    axa, axb, axc = (fig.add_subplot(g0[k]) for k in range(3))
    dumbbell(axa, T, "onset", col, ".0f", flag="flag_any", xlabel="Onset (ms)")
    axa.set_xlim(0, min(40, np.nanmax(T[["onset_active_hi", "onset_passive_hi"]].to_numpy()) + 2))
    axa.set_title("Onset (filled: task, open: passive)", loc="left", fontsize=5.6)
    test_text(axa, tests[0], " ms", "+.1f")
    dumbbell(axb, T, "acc", col, ".2f", xlabel="Corrected accuracy, 5-50 ms")
    axb.set_yticklabels([]); axb.set_title("Early accuracy", loc="left", fontsize=5.6)
    test_text(axb, tests[1], "", "+.3f")
    dumbbell(axc, T, "base", col, ".2f", xlabel="Corrected accuracy, -150..0 ms")
    axc.axvline(0, color="0.5", lw=0.5)
    axc.set_yticklabels([]); axc.set_title("Pre-stimulus baseline (expected 0)", loc="left", fontsize=5.6)
    test_text(axc, tests[2], "", "+.3f")
    rows = []
    for gi, (res, xlim) in enumerate((("zoom", (-20, 50)), ("wide", (-200, 600))), start=1):
        g = gs[gi].subgridspec(1, N_SHOW, wspace=0.18)
        axs = [fig.add_subplot(g[k]) for k in range(N_SHOW)]
        for k, (ax, a) in enumerate(zip(axs, top)):
            for e, ls in (("active", "-"), ("passive", "--")):
                curve(ax, E[e]["b"], a, res, col.get(a, "0.4"), ls)
            ax.axvline(0, color="0.5", lw=0.4, ls=":"); ax.axhline(0, color="0.6", lw=0.4)
            ax.set_xlim(xlim); ax.set_ylim(-0.03, 0.52)
            if gi == 1:
                ax.set_title(m4.label(a), fontsize=5.4, color=col.get(a, "k"))
            if k:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Corrected accuracy")
            ax.set_xlabel("Time (ms)", fontsize=5)
        rows.append(axs)
    rows[0][-1].plot([], [], "k-", lw=0.8, label="task (active)")
    rows[0][-1].plot([], [], "k--", lw=0.8, label="passive")
    rows[0][-1].legend(loc="lower right", frameon=False, fontsize=4.6, handlelength=1.8)
    S.letter_row(fig, [axa, axb, axc], "abc", dx_in=0.5)
    S.letter_row(fig, [rows[0][0]], "d")
    S.letter_row(fig, [rows[1][0]], "e")
    src = "; ".join(f"{e}: {E[e]['n_iter']} iterations" for e in EPOCHS)
    fig.suptitle(f"Task (active) vs passive trials, {m4.LEVEL_NAME[level]} (N = {N_MAIN}; {src})",
                 x=0.02, y=1 - 0.05 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, HOME / "figures", f"active_vs_passive_{level}")
    plt.close(fig)
    return tests


def main():
    plt = S.setup()
    col = AR.colors()
    tests, prov = [], {}
    for level in ("area_group", "area_acronym_custom"):
        E = {e: load(e, level) for e in EPOCHS}
        prov[level] = {e: dict(folder=E[e]["folder"], kind=E[e]["kind"], n_iter=E[e]["n_iter"]) for e in EPOCHS}
        tests += figure(plt, level, E, col)
        print(level, {e: E[e]["kind"] for e in EPOCHS}, flush=True)
    pd.DataFrame(tests).to_csv(HOME / "tables" / "active_vs_passive_tests.csv", index=False)
    print(pd.DataFrame(tests).round(4).to_string(index=False))
    (HOME / "provenance_006.json").write_text(json.dumps(dict(
        script="006_active_vs_passive.py", N=N_MAIN, sources=prov, unreliable_rule=dict(
            range_ms=m4.WIDE_RANGE_MS, min_defined=m4.MIN_DEFINED), tests="Wilcoxon signed-rank + paired t, over areas",
        date=time.strftime("%Y-%m-%d %H:%M")), indent=1))


if __name__ == "__main__":
    main()
