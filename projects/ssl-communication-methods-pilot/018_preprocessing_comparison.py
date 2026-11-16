"""Does the communication result depend on preprocessing? (user request 2026-11-10: "do the normalization; also check
baseline-subtracted data (trial-by-trial) and see deviations from that"). Compares three 016 runs of one area set:
  raw    condition-mean residuals of spike counts (comm_v3_<areas>);
  z      the same, soft-normalised per unit (comm_v3_<areas>_z);
  zbase  each trial's own -1.1..-0.6 s baseline subtracted per unit, then as z (comm_v3_<areas>_zbase).
All measures are ceiling-normalised (excess / within-area ceiling of the target group, 013), which makes them
comparable across preprocessing (the raw R^2 scale changes with the normalisation).
Figure (183 mm wide; ssl-paper-figures paperstyle):
  a-c  RRR R^2 / ceiling over time on whisker hits, one panel per preprocessing; mean over each session's pairs, then
       mean +- s.e.m. over sessions per cohort; bars: cohort cluster p < 0.05 (013);
  d-f  the same for partial RRR (rest of the recording regressed out);
  g    early rise on whisker hits per method and preprocessing, mean +- s.e.m. over sessions per cohort;
  h    pre-stimulus level on whisker hits, same layout (what the trial baseline takes away);
  i    pairs with an early rise != 0 on whisker hits (Wilcoxon per pair, BH q < 0.05) per method and preprocessing.
Table: OUT/report_figures/preprocessing_<areas>_<pop>.csv, the pooled tests (013 stats_pooled, source = target = all)
for every method, condition and variable, one block per preprocessing.
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-communication-methods-pilot/018_preprocessing_comparison.py
    [--areas seven] [--pop learners]
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "ssl-paper-figures")]
import comm_lib as cl  # noqa: E402
import paperstyle as ps  # noqa: E402

_spec = importlib.util.spec_from_file_location("pubfig", HERE / "017_publication_figure.py")
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)

PREPS = {"raw": "Counts (raw)", "z": "Normalised", "zbase": "Normalised, trial baseline removed"}
PREP_SHORT = {"raw": "raw", "z": "norm.", "zbase": "norm. − base"}
METHODS = ["r2n", "ccan", "pccan", "prrrn", "dr2n"]
METHOD_LAB = {"r2n": "RRR", "ccan": "CCA", "pccan": "pCCA", "prrrn": "pRRR", "dr2n": "ΔR²"}
COL, COHORTS = pf.COL, pf.COHORTS


def run_dir(areas, prep, pop):
    return cl.OUT / (f"comm_v3_{areas}" if prep == "raw" else f"comm_v3_{areas}_{prep}") / pop


def method_dots(ax, ep, var, out):
    """Per method (groups on x) and preprocessing (3 positions within a group): session means over pairs, mean +- s.e.m."""
    rows = []
    for i, q in enumerate(METHODS):
        for j, prep in enumerate(PREPS):
            d = ep[prep]
            d = d[(d.quantity == q) & (d.condition == "W_lick") & (d.source != "all")]
            s = d.groupby(["session_id", "cohort"])[var].mean().reset_index()
            for c, dx in (("R+", -0.07), ("R-", 0.07)):
                v = s[s.cohort == c][var]
                x = i + (j - 1) * 0.24 + dx
                ax.errorbar(x, v.mean(), v.sem(), fmt="os^"[j], ms=2.4, color=COL[c], mfc=COL[c] if j == 0 else "white",
                            elinewidth=0.6, mew=0.6, lw=0)
                rows.append(dict(method=q, prep=prep, cohort=c, n=len(v), mean=v.mean(), sem=v.sem()))
    ax.axhline(0, color="0.7", lw=0.4)
    ax.set_xticks(range(len(METHODS)), [METHOD_LAB[q] for q in METHODS])
    out.append(pd.DataFrame(rows).assign(variable=var))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--areas", default="seven")
    ap.add_argument("--pop", default="learners")
    a = ap.parse_args()
    ps.set_nature_rc()
    sl = {p: pd.read_parquet(run_dir(a.areas, p, a.pop) / "session_level.parquet") for p in PREPS}
    ep = {p: pd.read_parquet(run_dir(a.areas, p, a.pop) / "epochs_session.parquet") for p in PREPS}
    clus = {p: pd.read_csv(run_dir(a.areas, p, a.pop) / "stats_clusters.csv") for p in PREPS}
    clus = {p: (c[c.source == "all"] if "source" in c else c) for p, c in clus.items()}
    counts = {p: pd.read_csv(run_dir(a.areas, p, a.pop) / "method_counts.csv") for p in PREPS}
    n_s = {c: sl["z"][sl["z"].cohort == c].session_id.nunique() for c in COHORTS}

    fig = plt.figure(figsize=(183 * ps.MM, 150 * ps.MM))
    gs = fig.add_gridspec(3, 3, left=0.08, right=0.98, top=0.93, bottom=0.08, hspace=0.75, wspace=0.38)
    out_csv = {}
    for i, (q, ylab) in enumerate((("r2n", "RRR R² / ceiling"), ("prrrn", "pRRR R² / ceiling"))):
        axs = []
        for j, prep in enumerate(PREPS):
            ax = fig.add_subplot(gs[i, j])
            d = pf.pooled(sl[prep], q, "W_lick")
            out_csv[f"{'abcdef'[3 * i + j]}_{q}_{prep}"] = d
            pf.time_panel(ax, d, clus[prep], q, "W_lick", ylab if j == 0 else None)
            ax.set_title(f"{PREPS[prep]}" + (", rest of brain removed" if i == 1 else ""), fontsize=6)
            pf.label(ax, "abcdef"[3 * i + j])
            axs.append(ax)
        lo, hi = min(x.get_ylim()[0] for x in axs), max(x.get_ylim()[1] for x in axs)
        for x in axs:
            x.set_ylim(lo, hi)
    dots = []
    for k, (var, ttl, yl) in enumerate((("rise_early", "Early rise, whisker hit", "Rise / ceiling"),
                                        ("pre", "Pre-stimulus level, whisker hit", "Level / ceiling"))):
        ax = fig.add_subplot(gs[2, k])
        method_dots(ax, ep, var, dots)
        ax.set_title(ttl, fontsize=6)
        ax.set_ylabel(yl)
        pf.label(ax, "gh"[k])
    for j, prep in enumerate(PREPS):  # marker key, once
        ax.plot([], [], "os^"[j], ms=2.4, color="0.3", mfc="0.3" if j == 0 else "white", mew=0.6, label=PREP_SHORT[prep])
    ax.legend(loc="upper right", fontsize=4.5)
    out_csv["gh_method_dots"] = pd.concat(dots)
    ax = fig.add_subplot(gs[2, 2])
    rows = []
    for i, q in enumerate(METHODS):
        for j, prep in enumerate(PREPS):
            c = counts[prep]
            r = c[(c.quantity == q) & (c.condition == "W_lick") & (c.variable == "rise_early")]
            if not len(r):
                continue
            r = r.iloc[0]
            for cc, dx, v in (("R+", -0.06, r.rplus_rise), ("R-", 0.06, r.rminus_rise)):
                ax.bar(i + (j - 1) * 0.26 + dx, v / r.n_pairs, 0.11, color=COL[cc], alpha=[1, 0.7, 0.4][j], lw=0)
            rows.append(dict(method=q, prep=prep, rplus=r.rplus_rise, rminus=r.rminus_rise, n_pairs=r.n_pairs))
    ax.set_xticks(range(len(METHODS)), [METHOD_LAB[q] for q in METHODS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pairs with early rise ≠ 0")
    ax.set_title("Whisker hit (bars: raw, norm., norm. − base)", fontsize=6)
    pf.label(ax, "i")
    out_csv["i_method_counts"] = pd.DataFrame(rows)
    fig.text(0.5, 0.985, f"Communication by preprocessing, {a.areas} area groups ({a.pop}: R+ {n_s['R+']}, R- {n_s['R-']} "
             "sessions; R+ green, R- magenta)", ha="center", va="top", fontsize=7)
    od = cl.OUT / "report_figures"
    name = f"fig_preprocessing_{a.areas}_{a.pop}"
    ps.save_figure(fig, od, name)
    (od / f"{name}_data").mkdir(parents=True, exist_ok=True)
    for k, v in out_csv.items():
        v.to_csv(od / f"{name}_data" / f"{k}.csv")
    tab = []
    for prep in PREPS:
        t = pd.read_csv(run_dir(a.areas, prep, a.pop) / "stats_pooled.csv")
        tab.append(t[(t.source == "all") & (t.target == "all")].assign(prep=prep))
    pd.concat(tab).to_csv(od / f"preprocessing_{a.areas}_{a.pop}.csv", index=False)
    print(od / name)


if __name__ == "__main__":
    main()
