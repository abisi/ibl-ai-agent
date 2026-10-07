"""012 -- First-lick-aligned PSTHs per session half for all four groups: R+ and R-, learning day (day 0) and expert sessions
(user review 2026-10-07: R+ equivalent of the R- PSTH panel of the COSYNE within-day figure, one figure per analysis).

Data: 004 psth_halves.npz (per session, class WH / AH / SL, half early / late: rate minus baseline, mean over units, 10 ms
bins) and psth_halves/sessions.csv. A session enters a class / half trace with >= 3 events of that class in that half.
Traces: 3-bin boxcar smoothing, mean +- s.e.m. over sessions. Rows: cohorts; columns: day 0 early, day 0 late, expert
early, expert late; one y-axis range per row. Last column: WH - SL difference trace, early (dashed) vs late (solid) half,
day 0 and expert.
Output: within_day/<ref>/psth_halves/<pop>/psth_halves_groups.{png,pdf,svg}.
"""
import argparse
import importlib
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
CONV = HERE.parent
sys.path[:0] = [str(HERE), str(CONV)]
m51 = importlib.import_module("051_roc_prelick")
m61 = importlib.import_module("061_roc_prelick_learners")
m62 = importlib.import_module("062_pub_convergence_figures")
BASE = m51.WITHIN / "psth_halves"
COH, CL, CLAB = m62.COH, m62.CL, m62.CLAB
COLS = [("learning", 0), ("learning", 1), ("expert", 0), ("expert", 1)]
sm = lambda x: np.convolve(x, np.ones(3) / 3, "same")


def main(a):
    plt = m62.setup()
    out = BASE / a.population
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(BASE / "psth_halves.npz", allow_pickle=True)
    S = pd.read_csv(BASE / "sessions.csv")
    keep = np.ones(len(S), bool)
    if a.population == "learners":
        keep = S.index.isin(m61.learner_filter(S).index)
    P, Nev, edges = z["psth"], z["n"], z["edges"]
    tc = (edges[:-1] + np.diff(edges) / 2) * 1e3
    iw, isl = m51.CLASSES.index("WH"), m51.CLASSES.index("FA")
    fig = plt.figure(figsize=(m62.W_IN, 3.9))
    gs = fig.add_gridspec(2, 5, hspace=0.75, wspace=0.35, left=0.07, right=0.985, top=0.88, bottom=0.11,
                          width_ratios=[1, 1, 1, 1, 1.15])
    first = []
    for i, coh in enumerate(("R+", "R-")):
        row = []
        for j, (stg, hh) in enumerate(COLS):
            ax = fig.add_subplot(gs[i, j]); row.append(ax)
            msk = keep & ((S.cohort == coh) & (S.stage == stg)).to_numpy()
            ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
            for k, c in enumerate(m51.CLASSES):
                ok = msk & (Nev[:, k, hh] >= 3)
                A = np.array([sm(p) for p in P[ok, k, hh]])
                if len(A) < 2:
                    continue
                mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                ax.fill_between(tc, mu - se, mu + se, color=CL[c], alpha=0.25, lw=0)
                ax.plot(tc, mu, color=CL[c], lw=0.9, label=CLAB[c])
            ax.set_xlim(-500, 300); ax.set_xticks([-400, 0, 200]); ax.set_xticklabels(["−400", "0", "200"])
            ax.set_title(f"{coh.replace('-', '−')} {'day 0' if stg == 'learning' else 'expert'}, "
                         f"{'early' if hh == 0 else 'late'} half\n({int(msk.sum())} sessions)", color=COH[coh], fontsize=5.2, pad=2)
            ax.set_xlabel("Time from first lick (ms)", fontsize=5)
            if j == 0:
                ax.set_ylabel("Rate − baseline (Hz)")
                if i == 0:
                    ax.legend(frameon=False, loc="upper left", handlelength=1.0, borderaxespad=0.1, fontsize=4.6)
        yl = (min(x.get_ylim()[0] for x in row), max(x.get_ylim()[1] for x in row))
        for j, x in enumerate(row):
            x.set_ylim(*yl)
            if j:
                x.set_yticklabels([])
        # WH - SL difference trace, early vs late, day 0 and expert
        ax = fig.add_subplot(gs[i, 4]); row.append(ax)
        ax.axvspan(-100, 0, color="#FDD49E", alpha=0.6, lw=0); ax.axvline(0, color="0.3", lw=0.5, ls=(0, (2, 2)))
        for stg, alpha in (("learning", 1.0), ("expert", 0.45)):
            msk = keep & ((S.cohort == coh) & (S.stage == stg)).to_numpy()
            for hh, ls in ((0, (0, (2.5, 1.5))), (1, "-")):
                ok = msk & (Nev[:, iw, hh] >= 3) & (Nev[:, isl, hh] >= 3)
                A = np.array([sm(p) for p in P[ok, iw, hh] - P[ok, isl, hh]])
                if len(A) < 2:
                    continue
                mu, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(A))
                ax.fill_between(tc, mu - se, mu + se, color=COH[coh], alpha=0.12 * alpha, lw=0)
                ax.plot(tc, mu, color=COH[coh], lw=0.9, ls=ls, alpha=alpha,
                        label=f"{'day 0' if stg == 'learning' else 'expert'} {'early' if hh == 0 else 'late'}")
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlim(-500, 300); ax.set_xticks([-400, 0, 200]); ax.set_xticklabels(["−400", "0", "200"])
        ax.set_title(f"{coh.replace('-', '−')}: WH − SL", color=COH[coh], fontsize=5.2, pad=2)
        ax.set_xlabel("Time from first lick (ms)", fontsize=5); ax.set_ylabel("Δ rate (Hz)", fontsize=5)
        ax.legend(frameon=False, fontsize=4.2, loc="upper left", handlelength=1.6)
        first.append(row)
    m62.letter_row(fig, first[0], "abcde", dx_in=0.25, dy_in=0.12)
    m62.letter_row(fig, first[1], "fghij", dx_in=0.25, dy_in=0.12)
    fig.suptitle(f"First-lick-aligned activity per session half: R+ and R−, day 0 and expert ({a.population})",
                 x=0.02, y=0.995, ha="left", va="top", fontsize=7, weight="bold")
    m62.save(fig, out, "psth_halves_groups"); plt.close(fig)
    print("done", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", default="learners", choices=["all", "learners"])
    main(ap.parse_args())
