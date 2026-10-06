"""003 -- How many neurons would area B need to match the early accuracy of a reference area A?

Reference areas (user, 2026-10-04): Somatosensory-whisker, Midbrain, Auditory areas, each at N_REF = 100 neurons.
Target = the reference's corrected balanced accuracy in 5-50 ms (002 window_accuracy.csv). For every other area (both
levels), the matched N is read from its accuracy-vs-N curve (monotone envelope, linear in log N between the measured
counts; beyond 500, extrapolated linearly in log N from the two largest counts, capped at N_MAX; areas whose curve never
reaches the target within N_MAX are marked "not reached"). Matched N is rounded to the nearest 10.
  --plan : writes matched_n.csv and the 001 runs still needed (OUT/matched_runs.txt, one "level|area|N" per line)
  --plot : figure matched_accuracy (per reference: wide and zoom curves of every area at its matched N, and the matched
           N per area), using the 001 runs at those N; checks that the window accuracy at matched N is near the target.
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
S = importlib.import_module("_style")
m2 = importlib.import_module("002_arrival_summary")
OUT, FIG, LEVELS = m2.OUT, m2.FIG, m2.LEVELS
REFS = ["Somatosensory-whisker", "Midbrain", "Auditory areas"]
N_REF, N_MAX = 100, 2000


def matched_n(q, target):
    q = q.sort_values("N")
    N, acc = q.N.to_numpy(float), np.maximum.accumulate(q["mean"].to_numpy())
    x = np.log(N)
    if target <= acc[0]:
        return float(N[0]), "at or below the smallest N"
    i = np.searchsorted(acc, target)
    if i < len(acc):
        f = (target - acc[i - 1]) / max(1e-9, acc[i] - acc[i - 1])
        return float(np.exp(x[i - 1] + f * (x[i] - x[i - 1]))), "interpolated"
    slope = (acc[-1] - acc[-2]) / (x[-1] - x[-2])
    if slope <= 1e-6:
        return np.nan, "not reached"
    n = float(np.exp(x[-1] + (target - acc[-1]) / slope))
    return (n, "extrapolated") if n <= N_MAX else (np.nan, "not reached")


def plan():
    W = pd.read_csv(OUT / "window_accuracy.csv")
    W = W[W.N.isin(m2.N_GRID)]
    rows = []
    for ref in REFS:
        t = W[(W.level == "area_group") & (W.area == ref) & (W.N == N_REF)]["mean"]
        if not len(t):
            continue
        target = float(t.iloc[0])
        for level, areas in {"area_group": LEVELS["area_group"]}.items():     # matched N for area groups only
            for a in areas:
                q = W[(W.level == level) & (W.area == a)]
                if len(q) < 2:
                    continue
                n, how = matched_n(q, target)
                n10 = int(max(20, round(n / 10) * 10)) if np.isfinite(n) else np.nan
                if level == "area_group" and a == ref:
                    n10, how = N_REF, "reference"
                rows.append(dict(reference=ref, target_accuracy=target, level=level, area=a, matched_N=n10, matched_N_exact=n,
                                 how=how))
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "matched_n.csv", index=False)
    need = sorted({(r.level, r.area, int(r.matched_N)) for r in P.itertuples() if np.isfinite(r.matched_N)})
    have = {p.name for p in (OUT / "raw").glob("*.parquet")}
    todo = [f"{l}|{a}|{n}" for l, a, n in need if f"{l}__{a.replace(' ', '_')}__N{n}.parquet" not in have]
    (OUT / "matched_runs.txt").write_text("\n".join(todo) + "\n")
    print(P.pivot_table(index=["level", "area"], columns="reference", values="matched_N").to_string())
    print(len(todo), "runs needed ->", OUT / "matched_runs.txt")


def plot():
    plt = S.setup()
    P = pd.read_csv(OUT / "matched_n.csv")
    D = m2.load_raw()
    B, O, W = m2.summarise(D)
    H = 1.9 * len(REFS) + 0.7
    fine = (P.level == "area_acronym_custom").any()   # fine-area matched runs exist only when 003 --plan was run for them
    panels = [("area_group", "wide"), ("area_group", "zoom")] + ([("area_acronym_custom", "zoom")] if fine else [])
    titles = ["Area groups, 50-ms bins", "Area groups, 20-ms bins", "Areas, 20-ms bins"]
    # columns: a (groups, 50 ms) | b (groups, 20 ms) | legend b | [c (areas, 20 ms) | legend c] | neurons needed
    widths = [1.4, 1.0, 0.55] + ([1.0, 0.55] if fine else []) + [0.9]
    plot_cols, leg_cols = ([0, 1, 3], {1: 2, 2: 4}) if fine else ([0, 1], {1: 2})
    fig = plt.figure(figsize=(S.W_IN * (1.3 if fine else 1.0), H))
    gs = fig.add_gridspec(len(REFS), len(widths), width_ratios=widths, wspace=0.42, hspace=0.75, left=0.08,
                          right=0.99, top=1 - 0.6 / H, bottom=0.45 / H)
    chk = []
    for r, ref in enumerate(REFS):
        p = P[P.reference == ref]
        if not len(p):
            continue
        target = p.target_accuracy.iloc[0]
        axs = [fig.add_subplot(gs[r, k]) for k in plot_cols + [len(widths) - 1]]
        for col, (level, res) in enumerate(panels):
            ax = axs[col]
            for x in p[p.level == level].itertuples():
                if not np.isfinite(x.matched_N):
                    continue
                q = B[(B.level == level) & (B.area == x.area) & (B.N == int(x.matched_N)) & (B.resolution == res)]
                if not len(q):
                    continue
                tt = 1000 * q.t.to_numpy()
                lw = 1.3 if x.area == ref else 0.7
                ax.fill_between(tt, q["mean"] - q.sd, q["mean"] + q.sd, color=S.AREA_C[x.area], alpha=0.15, lw=0, edgecolor="none")
                ax.plot(tt, q["mean"], color=S.AREA_C[x.area], lw=lw, label=f"{S.short(x.area)} ({int(x.matched_N)})")
                w = W[(W.level == level) & (W.area == x.area) & (W.N == int(x.matched_N))]
                if len(w) and col != 0:
                    chk.append(dict(reference=ref, level=level, area=x.area, matched_N=int(x.matched_N), target=target,
                                    achieved=float(w["mean"].iloc[0])))
            ax.axhline(0, color="0.5", lw=0.4)
            ax.axvline(0, color="0.3", lw=0.4, ls="--")
            if res == "zoom":
                ax.axvspan(5, 50, color="0.88", lw=0, edgecolor="none", zorder=0)
                ax.set_xlim(-20, 100)
            else:
                ax.set_xlim(-200, 600)
            ax.set_xlabel("Time from stimulus (ms)", fontsize=5.5)
            ax.set_title(titles[col], loc="left", fontsize=5.6)
            if col > 0:                                # panel a has the same areas and N as b: one legend, beside b
                lax = fig.add_subplot(gs[r, leg_cols[col]]); lax.set_axis_off()
                h, lab = ax.get_legend_handles_labels()
                lax.legend(h, lab, frameon=False, fontsize=4.2, handlelength=1, loc="upper left", borderaxespad=0,
                           title="area (neurons)", title_fontsize=4.4, alignment="left")
        axs[0].set_ylabel(f"Reference: {S.short(ref)}, {N_REF} neurons\nCorrected balanced accuracy")
        ax = axs[-1]
        q = p.sort_values("matched_N", na_position="last")
        y = np.arange(len(q))
        vals = q.matched_N.fillna(N_MAX * 1.15)
        ax.barh(y, vals, color=[S.AREA_C[a] for a in q.area], height=0.7, lw=0, edgecolor="none")
        for yi, v, how in zip(y, q.matched_N, q.how):
            ax.text((v if np.isfinite(v) else N_MAX * 1.15) * 1.08, yi, ("> %d" % N_MAX) if not np.isfinite(v) else
                    (f"{int(v)}*" if how == "extrapolated" else f"{int(v)}"), va="center", fontsize=4.4)
        ax.set_yticks(y, [S.short(a) for a in q.area], fontsize=4.6)
        ax.set_xscale("log")
        ax.axvline(N_REF, color="0.3", lw=0.4, ls=":")
        ax.set_xlabel("Neurons needed", fontsize=5.5)
        ax.set_title(f"to reach {target:.2f} (5-50 ms)", loc="left", fontsize=5.6)
        ax.invert_yaxis()
        k = len(axs)
        S.letter_row(fig, axs, "abcdefghijkl"[r * k:(r + 1) * k])
    fig.suptitle("Neurons needed to match the early whisker vs auditory decoding of a reference area, task trials (* extrapolated beyond 500)",
                 x=0.02, y=1 - 0.05 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, FIG, "matched_accuracy")
    plt.close(fig)
    pd.DataFrame(chk).to_csv(OUT / "matched_accuracy_check.csv", index=False)
    print("ALL DONE", FIG / "matched_accuracy.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plan:
        plan()
    if a.plot:
        plot()
