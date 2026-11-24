"""Rank selection per cohort on the full fits (user 2026-11-22: fixed rank picked after fitting the final datasets,
cohort-specific). Methods: split non-negative on single-trial baseline-subtracted data ("_splitnn", main) and
non-negative on rates without baseline subtraction ("_nn", reference); z-scored; 4 alignments; ranks 1-12.

Per cohort (R+, R-; whisker day 0 = learning stage; expert sessions shown as reference, not used for the choice):
median over sessions of the relative error (normalised to rank 1), of the mean restart similarity and of the error
decrease per added component (fraction of the rank 1 -> 12 decrease). Proposed rank (Williams et al. 2018 trade-off,
same rule as 006g, on the cohort medians): largest rank with median restart similarity >= 0.85 whose added component
still lowers the median error by >= 5 % of the total decrease.

Run: python 011_rank_by_cohort.py -> OUT/figures/aligned_gpu/rank_by_cohort.png|pdf, tables/rank_by_cohort.csv,
     tables/rank_curves_by_session.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tca_lib as L  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
METHODS = [("_splitnn", "split non-negative (baseline-subtracted)"), ("_nn", "non-negative (rates)")]
import os as _os
# optional subset of alignments written to separate tables (user 2026-11-23: go-signal tensors), e.g.
# TCA_ALIGNMENTS=stim_long_ns,jaw_all TCA_SUFFIX=_go; default: the four original alignments, original tables
SUFFIX = _os.environ.get("TCA_SUFFIX", "")
ALIGNMENTS = _os.environ.get("TCA_ALIGNMENTS", "stim_long,stim_sensory,jaw,lick").split(",")
D = L.OUT / "artifacts" / "tca_aligned_gpu"
SIM_MIN, GAIN_MIN = 0.85, 0.05
RANKS = np.arange(1, 13)


def choose(err, sim):
    gain = np.r_[np.inf, -np.diff(err)] / max(err[0] - err[-1], 1e-12)
    ok = (sim >= SIM_MIN) & (gain >= GAIN_MIN)
    ok[0] = True
    return int(RANKS[np.flatnonzero(ok)[-1]]), gain


def main():
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    rows = []
    for r in s.itertuples():
        for tag, _ in METHODS:
            for al in ALIGNMENTS:
                f = D / r.session_id / f"{al}_zscore{tag}.npz"
                if not f.exists():
                    continue
                d = np.load(f)
                if not all(f"r{k}_obj" in d.files for k in RANKS):
                    continue
                err = np.array([d[f"r{k}_obj"].min() for k in RANKS])
                sim = np.array([d[f"r{k}_sim"].mean() if len(d[f"r{k}_sim"]) else 1.0 for k in RANKS])
                for k, e, si in zip(RANKS, err, sim):
                    rows.append(dict(session_id=r.session_id, mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage,
                                     learner=r.learner, method=tag, alignment=al, rank=int(k), err=e,
                                     err_rel=e / err[0], sim=si))
    c = pd.DataFrame(rows)
    c.to_csv(L.OUT / "tables" / f"rank_curves_by_session{SUFFIX}.csv", index=False)
    fig, ax = plt.subplots(3, 2 * len(ALIGNMENTS), figsize=(13, 5.0), squeeze=False)
    out = []
    for jm, (tag, lab) in enumerate(METHODS):
        for ja, al in enumerate(ALIGNMENTS):
            j = jm * len(ALIGNMENTS) + ja
            for coh in ("R+", "R-"):
                for stage, ls in (("learning", "-"), ("expert", ":")):
                    x = c[(c.method == tag) & (c.alignment == al) & (c.cohort == coh) & (c.stage == stage)]
                    if x.empty:
                        continue
                    m = x.groupby("rank").agg(err=("err_rel", "median"), sim=("sim", "median"))
                    rk, gain = choose(m.err.to_numpy(), m.sim.to_numpy())
                    lw = 1.1 if stage == "learning" else 0.7
                    ax[0, j].plot(RANKS, m.err, color=COH_COL[coh], lw=lw, ls=ls)
                    ax[1, j].plot(RANKS, m.sim, color=COH_COL[coh], lw=lw, ls=ls)
                    ax[2, j].plot(RANKS[1:], gain[1:], color=COH_COL[coh], lw=lw, ls=ls)
                    if stage == "learning":
                        for i in range(3):
                            ax[i, j].axvline(rk, color=COH_COL[coh], lw=0.5, alpha=0.7)
                    out.append(dict(method=tag, alignment=al, cohort=coh, stage=stage,
                                    n_sessions=x.session_id.nunique(), proposed_rank=rk,
                                    sim_at_rank=float(m.sim.iloc[rk - 1]), err_rel_at_rank=float(m.err.iloc[rk - 1])))
            ax[1, j].axhline(SIM_MIN, color="0.5", lw=0.4, ls=":")
            ax[2, j].axhline(GAIN_MIN, color="0.5", lw=0.4, ls=":")
            ax[0, j].set_title(f"{al}\n{lab}", fontsize=5.5)
            for i in range(3):
                ax[i, j].set_xticks([1, 4, 8, 12])
            ax[2, j].set_xlabel("Rank")
    for i, lab in enumerate(["Error / error at rank 1\n(median)", "Restart similarity\n(median)",
                             "Error decrease\n(fraction of total)"]):
        ax[i, 0].set_ylabel(lab)
    ax[0, 0].plot([], [], color=COH_COL["R+"], label="R+ day 0")
    ax[0, 0].plot([], [], color=COH_COL["R-"], label="R- day 0")
    ax[0, 0].plot([], [], color="0.4", ls=":", label="expert (reference)")
    ax[0, 0].legend(frameon=False, fontsize=5)
    fig.suptitle("Rank per cohort: vertical lines = proposed rank (median restart similarity >= 0.85, error decrease "
                 ">= 5 %), whisker day 0", fontsize=6.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.align_ylabels(ax[:, 0])
    od = L.OUT / "figures" / "aligned_gpu"
    od.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(od / f"rank_by_cohort{SUFFIX}.{ext}", dpi=230)
    t = pd.DataFrame(out)
    t.to_csv(L.OUT / "tables" / f"rank_by_cohort{SUFFIX}.csv", index=False)
    print(t[t.stage == "learning"].pivot_table(index=["method", "alignment"], columns="cohort",
                                                values=["proposed_rank", "sim_at_rank"]).round(3).to_string())


if __name__ == "__main__":
    main()
