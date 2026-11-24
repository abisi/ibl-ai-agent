"""Pilot summary across mice (Kuma GPU fits, 006g): per TCA type (alignment x normalisation), every mouse's error and
restart-similarity curves, error decrease per added component, and the chosen rank (Williams et al. 2018 rule);
grid of chosen ranks (mice x types) and TCA error at the chosen rank relative to the condition-mean model.

Run: python 009_pilot_summary.py --sessions <sid> ... [--tca-dir tca_aligned_gpu]
     -> OUT/figures/aligned_gpu/pilot_summary_<n>mice.png|pdf (Kuma fits), tables/pilot_summary_<n>mice.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tca_lib as L  # noqa: E402

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False})
COH_COL = {"R+": "#00B400", "R-": "#C800C8"}
ALIGNMENTS = ["stim_long", "stim_sensory", "jaw", "lick"]
NORMS = ["rate", "soft", "zscore"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--tca-dir", default="tca_aligned_gpu")
    a = ap.parse_args()
    meta = pd.read_csv(L.OUT / "tables" / "sessions.csv").set_index("session_id")
    D = L.OUT / "artifacts" / a.tca_dir
    types = [(al, n) for al in ALIGNMENTS for n in NORMS]
    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(4, len(types), height_ratios=[1, 1, 1, 1.25])
    rows = []
    for j, (al, n) in enumerate(types):
        axs = [fig.add_subplot(gs[i, j]) for i in range(3)]
        for sid in a.sessions:
            f = D / sid / f"{al}_{n}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            R = np.array([int(k[1:-4]) for k in d.files if k.endswith("_obj")])
            R.sort()
            err = np.array([d[f"r{r}_obj"].min() for r in R])
            sim = np.array([d[f"r{r}_sim"].mean() if len(d[f"r{r}_sim"]) else 1 for r in R])
            col = COH_COL[meta.loc[sid, "cohort"]]
            ls = "-" if meta.loc[sid, "learning_category"] in ("good", "moderate") else ":"
            rc = int(d["rank_chosen"])
            axs[0].plot(R, err / err[0], color=col, lw=0.7, ls=ls)
            axs[1].plot(R, sim, color=col, lw=0.7, ls=ls)
            axs[2].plot(R[1:], np.array(d["gain"])[1:], color=col, lw=0.7, ls=ls)
            for ax_, y in ((axs[0], err[rc - 1] / err[0]), (axs[1], sim[rc - 1])):
                ax_.scatter([rc], [y], s=8, color=col, zorder=3)
            rows.append(dict(session_id=sid, cohort=meta.loc[sid, "cohort"],
                             learning_category=meta.loc[sid, "learning_category"], alignment=al, norm=n,
                             shape=str(tuple(d["shape"])), rank_chosen=rc, err_chosen=float(err[rc - 1]),
                             sim_chosen=float(sim[rc - 1]), condmean_err=float(d["condmean_err"]),
                             err_minus_condmean=float(err[rc - 1] - d["condmean_err"])))
        axs[1].axhline(0.85, color="grey", lw=0.4, ls=":")
        axs[2].axhline(0.05, color="grey", lw=0.4, ls=":")
        axs[0].set_title(f"{al}\n{n}", fontsize=6)
        for i, lab in enumerate(["Error / error at rank 1", "Restart similarity", "Error decrease\n(fraction of total)"]):
            axs[i].set_ylabel(lab if j == 0 else "")
            axs[i].set_xticks([1, 5, 10, 15, 20])
        axs[2].set_xlabel("Rank")
    df = pd.DataFrame(rows)
    ax = fig.add_subplot(gs[3, : len(types) // 2])
    M = df.pivot_table(index="session_id", columns=["alignment", "norm"], values="rank_chosen").reindex(
        index=a.sessions, columns=pd.MultiIndex.from_tuples(types))
    ax.imshow(M.to_numpy(float), cmap="viridis", vmin=1, vmax=20, aspect="auto")
    for (i, k), v in np.ndenumerate(M.to_numpy(float)):
        if np.isfinite(v):
            ax.text(k, i, f"{int(v)}", ha="center", va="center", fontsize=5, color="w" if v < 12 else "k")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{al}\n{n}" for al, n in types], fontsize=4.5)
    ax.set_yticks(range(len(a.sessions)))
    ax.set_yticklabels([f"{s.split('_')[0]} {meta.loc[s, 'cohort']} {meta.loc[s, 'learning_category']}"
                        for s in a.sessions], fontsize=5)
    for t, lab in zip(ax.get_yticklabels(), a.sessions):
        t.set_color(COH_COL[meta.loc[lab, "cohort"]])
    ax.set_title("Chosen rank (Williams et al. 2018 trade-off: restart similarity >= 0.85, error decrease >= 5 %)",
                 fontsize=6, loc="left")
    ax = fig.add_subplot(gs[3, len(types) // 2 + 1:])
    for k, (al, n) in enumerate(types):
        x = df[(df.alignment == al) & (df.norm == n)]
        ax.scatter(np.full(len(x), k) + np.linspace(-0.2, 0.2, max(len(x), 1))[:len(x)], x.err_minus_condmean,
                   c=[COH_COL[c] for c in x.cohort], s=8)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{al}\n{n}" for al, n in types], fontsize=4.5)
    ax.set_ylabel("TCA error at chosen rank\nminus condition-mean error")
    ax.set_title("Below 0: TCA (few parameters) fits better than per-class trial averages (in-sample)", fontsize=6,
                 loc="left")
    fig.text(0.01, 0.99, "R+ green, R- magenta; solid = learner (good / moderate), dotted = non-learner", fontsize=6,
             va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = L.OUT / "figures" / ("aligned_gpu" if a.tca_dir.endswith("_gpu") else "aligned")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{len(a.sessions)}mice"
    for ext in ("png", "pdf"):
        fig.savefig(out / f"pilot_summary_{tag}.{ext}", dpi=220)
    df.to_csv(L.OUT / "tables" / f"pilot_summary_{tag}.csv", index=False)
    print(df.groupby(["alignment", "norm"]).rank_chosen.describe()[["mean", "min", "max"]].round(1).to_string())


if __name__ == "__main__":
    main()
