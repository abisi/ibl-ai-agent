"""Jaw-aligned TCA overview, one row per pilot session (fits of 003b_tca_jaw.py).

Columns: (a) time factors of the `jaw_all` fit (components ordered by peak time; jaw onset = 0, median stimulus time
marked), (b) mean trial weight per trial class (hits, pseudo-aligned misses / correct rejections, passive trials),
each component scaled by its SD over trials, (c) Spearman rho between trial weight and jaw reaction time (lick trials
with a real onset), (d) time factors of the lick-trials-only `jaw` fit.

Run: python 004b_jaw_overview.py [--rank 6] [--dir tca|tca_preview]
     -> OUT/figures/pilot/jaw_overview_r<rank>[_preview].png|pdf
"""
import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import tca_lib as L  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False})
CLASSES = [("Whisker hit", "active", "whisker_trial", 1), ("Whisker miss*", "active", "whisker_trial", 0),
           ("Auditory hit", "active", "auditory_trial", 1), ("False alarm", "active", "no_stim_trial", 1),
           ("Correct rej.*", "active", "no_stim_trial", 0), ("Whisker passive*", "passive", "whisker_trial", None),
           ("Auditory passive*", "passive", "auditory_trial", None)]
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}


def load(d, variant, sid):
    f = L.OUT / "artifacts" / d / "bsub" / variant / f"{sid}.npz"
    return dict(np.load(f)) if f.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=6)
    ap.add_argument("--dir", default="tca")
    a = ap.parse_args()
    R = a.rank
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    s = s[s.pilot].sort_values(["stage", "cohort", "learning_category"], ascending=[False, True, True])
    s = s[[load(a.dir, "jaw", x) is not None for x in s.session_id]]
    cmap = plt.get_cmap("viridis")
    fig, ax = plt.subplots(len(s), 4, figsize=(7.4, 1.05 * len(s) + 0.4),
                           gridspec_kw={"width_ratios": [1.3, 1.6, 0.8, 1.3]}, squeeze=False)
    for i, r in enumerate(s.itertuples()):
        d = load(a.dir, "jaw_all", r.session_id)
        src = "all trials"
        if d is None:   # fall back to the lick-trials fit (no pseudo-aligned classes)
            d, src = load(a.dir, "jaw", r.session_id), "lick trials"
        t = pd.read_parquet(L.TENSORS / f"{r.session_id}_trials.parquet").set_index("trial_index")
        t = t.loc[d["trials_kept"]].reset_index()
        A, C, tc = d[f"r{R}_f0"], d[f"r{R}_f2"], d["t_centres"] * 1e3
        o = np.argsort(tc[np.argmax(np.abs(C), 0)])
        A, C = A[:, o], C[:, o]
        cols = [cmap(k / max(R - 1, 1)) for k in range(R)]
        real = ~d["pseudo"]
        stim_ms = -np.median(d["jaw_onset"][real]) * 1e3
        b = ax[i, 0]
        for k in range(R):
            b.plot(tc, C[:, k], color=cols[k], lw=0.8, label=f"C{k + 1}")
        b.axvline(0, color="k", lw=0.4)
        b.axvline(stim_ms, color="grey", lw=0.4, ls="--")
        b.set_ylabel(f"{r.session_id}\n{r.cohort}, {r.stage}\n{r.learning_category}\n({src} fit)", fontsize=5,
                     color=COH_COL[r.cohort])
        # class means
        ctx = np.where(t.epoch == "active", "active", "passive")
        M = np.full((R, len(CLASSES)), np.nan)
        ns = []
        Az = A / (A.std(0, keepdims=True) + 1e-12)
        for j, (lab, c_, tt_, lk) in enumerate(CLASSES):
            m = (ctx == c_) & (t.trial_type == tt_).to_numpy()
            if lk is not None:
                m &= (t.lick_flag == lk).to_numpy()
            if lab in ("Whisker hit", "Auditory hit", "False alarm"):
                m &= real
            ns.append(int(m.sum()))
            if m.sum() >= 3:
                M[:, j] = Az[m].mean(0)
        v = np.nanmax(np.abs(M))
        b = ax[i, 1]
        b.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
        b.set_yticks(range(R))
        b.set_yticklabels([f"C{k + 1}" for k in range(R)], fontsize=5)
        for k in range(R):
            b.get_yticklabels()[k].set_color(cols[k])
        b.set_xticks(range(len(CLASSES)))
        b.set_xticklabels([f"{c[0]} ({n})" for c, n in zip(CLASSES, ns)] if i == len(s) - 1 else
                          [str(n) for n in ns], rotation=40 if i == len(s) - 1 else 0, ha="right" if i == len(s) - 1
                          else "center", fontsize=4.5)
        # RT correlation
        b = ax[i, 2]
        rt = d["jaw_onset"]
        rho = [stats.spearmanr(A[real, k], rt[real])[0] for k in range(R)]
        b.barh(range(R), rho, color=cols)
        b.axvline(0, color="k", lw=0.4)
        b.set_xlim(-0.8, 0.8)
        b.invert_yaxis()
        b.set_yticks([])
        # lick-only fit
        b = ax[i, 3]
        dj = load(a.dir, "jaw", r.session_id)
        if dj is not None:
            Cj = dj[f"r{R}_f2"]
            oj = np.argsort(tc[np.argmax(np.abs(Cj), 0)])
            for k in range(R):
                b.plot(tc, Cj[:, oj[k]], color=cols[k], lw=0.8)
            b.axvline(0, color="k", lw=0.4)
            b.axvline(stim_ms, color="grey", lw=0.4, ls="--")
    ax[0, 0].set_title("Time factors (all-trials fit if available)\n(black: jaw onset, dashed: median stimulus)", fontsize=6)
    ax[0, 1].set_title("Mean trial weight per class (SD units)\n* pseudo-aligned; n trials", fontsize=6)
    ax[0, 2].set_title("rho(trial weight,\njaw reaction time)", fontsize=6)
    ax[0, 3].set_title("Lick trials only: time factors", fontsize=6)
    ax[-1, 0].set_xlabel("Time from jaw onset (ms)")
    ax[-1, 3].set_xlabel("Time from jaw onset (ms)")
    ax[-1, 2].set_xlabel("Spearman rho")
    ax[0, 0].legend(frameon=False, fontsize=4.5, ncol=2, loc="upper left")
    fig.tight_layout(h_pad=0.4)
    out = L.OUT / "figures" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    tag = "_preview" if a.dir != "tca" else ""
    for ext in ("png", "pdf"):
        fig.savefig(out / f"jaw_overview_r{R}{tag}.{ext}", dpi=250)
    print("saved", out / f"jaw_overview_r{R}{tag}.png")


if __name__ == "__main__":
    main()
