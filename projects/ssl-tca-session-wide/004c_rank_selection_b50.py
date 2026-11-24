"""Rank selection, 10-ms vs 50-ms bins (user 2026-11-22), pilot sessions.

Columns: window pairs (10 ms | 50 ms) for trial, late, whole, jaw; evoked (5 ms) for reference.
Rows: relative error on all entries, held-out error (log scale; 1 = no better than predicting zero), similarity of
replicates to the best fit. Lines: per session (thin, cohort colour, dotted = expert), median across sessions (thick)
for bsub (black) and soft (grey, stimulus-aligned only). Medians, not means: a few sessions blow up at high rank.
Also writes the per-window rank at the median held-out minimum.

Run: python 004c_rank_selection_b50.py -> OUT/figures/pilot/rank_selection_b50.png|pdf, tables/rank_choice_b50.csv
"""
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tca_lib as L  # noqa: E402

F = importlib.import_module("003_tca_fit")
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False})
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
COLS = [("trial", "-100..500 ms, 10 ms"), ("trial_b50", "-100..500 ms, 50 ms"), ("late", "50 ms..1 s, 10 ms"),
        ("late_b50", "50 ms..1 s, 50 ms"), ("whole", "-500 ms..1 s, 10 ms"), ("whole_b50", "-500 ms..1 s, 50 ms"),
        ("jaw", "jaw -300..200 ms, 10 ms"), ("jaw_b50", "jaw -300..200 ms, 50 ms"), ("evoked", "5-50 ms, 5 ms")]
ROWS = ["Relative error, all entries", "Held-out error (log)", "Similarity to best fit"]


def curves(norm, epoch, sid):
    f = F.TCA_DIR / norm / epoch / f"{sid}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    R = list(F.RANKS)
    return (np.array([d[f"r{k}_obj"].min() for k in R]), np.array([d[f"r{k}_cv"][:, 0].mean() for k in R]),
            np.array([d[f"r{k}_sim"].mean() if len(d[f"r{k}_sim"]) else 1 for k in R]))


def main():
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    s = s[s.pilot]
    R = np.array(list(F.RANKS))
    fig, ax = plt.subplots(3, len(COLS), figsize=(10, 4.2), squeeze=False)
    rows = []
    for j, (ep, lab) in enumerate(COLS):
        for norm, mc in (("bsub", "k"), ("soft", "0.55")):
            got = {r.session_id: (r, curves(norm, ep, r.session_id)) for r in s.itertuples()}
            got = {k: v for k, v in got.items() if v[1] is not None}
            if not got:
                continue
            for i in range(3):
                M = np.array([v[1][i] for v in got.values()])
                if norm == "bsub":
                    for r, c in got.values():
                        ax[i, j].plot(R, c[i], color=COH_COL[r.cohort], lw=0.4, alpha=0.5,
                                      ls="-" if r.stage == "learning" else ":")
                ax[i, j].plot(R, np.median(M, 0), color=mc, lw=1.3, label=norm)
            med_te = np.median(np.array([v[1][1] for v in got.values()]), 0)
            rows.append(dict(epoch=ep, norm=norm, n_sessions=len(got), rank_min_heldout=int(R[np.argmin(med_te)]),
                             heldout_min=float(med_te.min()), heldout_r1=float(med_te[0]),
                             sim_at_min=float(np.median([v[1][2][np.argmin(med_te)] for v in got.values()]))))
        ax[0, j].set_title(lab, fontsize=6)
        ax[1, j].set_yscale("log")
        ax[1, j].axhline(1, color="grey", lw=0.4, ls="--")
        for i in range(3):
            ax[i, j].set_xlabel("Rank" if i == 2 else "")
            if j == 0:
                ax[i, j].set_ylabel(ROWS[i])
    ax[0, 0].plot([], [], color=COH_COL["R+"], lw=0.6, label="R+ session")
    ax[0, 0].plot([], [], color=COH_COL["R-"], lw=0.6, label="R- session")
    ax[0, 0].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    out = L.OUT / "figures" / "pilot"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"rank_selection_b50.{ext}", dpi=250)
    pd.DataFrame(rows).to_csv(L.OUT / "tables" / "rank_choice_b50.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
