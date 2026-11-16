"""024 -- which v5 model predicts each mouse best? (user 2026-10-23: "are there mice that are best fit by some models and not others?")
For every v5 model (021) and mouse, held-out log-likelihood of the mouse's whole sequence under the cohort parameters fitted on the
other 4 folds (the mouse was not used in fitting). No per-mouse fitting: this asks which cohort-level model generalises best to each
mouse, per factor (FA variant, engagement, transfer, sym/asym) by the best model within each level.
Output: tables/024_mouse_model_ll.csv (mouse x model), tables/024_mouse_best.csv, figures/024_per_mouse_models
Run (haas): OMP_NUM_THREADS=1 python 024_per_mouse_heldout.py
"""
from __future__ import annotations

import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
import rl6_model as M

F019 = importlib.import_module("019_fit_v4")
COH = {"R+": "#00B400", "R-": "#C800C8"}
FACTORS = {"FA": 1, "engagement": 2, "learning rates": 3, "transfer": 0}


def mouse_ll(D, F):
    rows = []
    for (cs, fold), d in D.groupby(["reward_group", "fold"]):
        d = d.reset_index(drop=True)
        A = M.arrays(d)
        for m in M.model_names():
            r = F[(F.model == m) & (F.cohort_set == cs) & (F.fold == str(fold))].iloc[0]
            p = M.latents(m, {k: r[f"p_{k}"] for k in M.P_ALL}, A)[:, 0]
            ll = np.where(d.lick_flag == 1, np.log(p), np.log(1 - p))
            s = pd.Series(ll).groupby(d.mouse_id.values).agg(["sum", "size"])
            rows += [dict(mouse_id=mid, reward_group=cs, model=m, ll=v["sum"], n=v["size"]) for mid, v in s.iterrows()]
    return pd.DataFrame(rows)


def main():
    D = F019.load()
    F = pd.read_csv(C.FITS / "021_fits.csv"); F["fold"] = F.fold.astype(str)
    L = mouse_ll(D, F)
    L.to_csv(C.TAB / "024_mouse_model_ll.csv", index=False)
    W = L.pivot_table(index=["mouse_id", "reward_group", "n"], columns="model", values="ll").reset_index()
    models = M.model_names()
    best_pooled = {cs: L[L.reward_group == cs].groupby("model").ll.sum().idxmax() for cs in ("R+", "R-")}
    out = W[["mouse_id", "reward_group", "n"]].copy()
    out["best_model"] = W[models].idxmax(axis=1)
    out["pooled_best"] = out.reward_group.map(best_pooled)
    out["d_best_vs_pooled"] = W[models].max(axis=1) - np.array([W.loc[i, best_pooled[c]] for i, c in zip(W.index, W.reward_group)])
    for fac, pos in FACTORS.items():
        lev = sorted({m.split("-")[pos] for m in models})
        best_in = pd.DataFrame({l: W[[m for m in models if m.split("-")[pos] == l]].max(axis=1) for l in lev})
        out[f"best_{fac}"] = best_in.idxmax(axis=1)
        srt = np.sort(best_in.to_numpy(), axis=1)
        out[f"margin_{fac}"] = srt[:, -1] - srt[:, -2]   # nats by which the winning level beats the runner-up level
    out.to_csv(C.TAB / "024_mouse_best.csv", index=False)
    pd.set_option("display.width", 200)
    print("pooled best:", best_pooled)
    for fac in FACTORS:
        print(f"\n{fac}: winners per cohort (all mice | margin > 10 nats)")
        print(pd.concat([pd.crosstab(out.reward_group, out[f"best_{fac}"]),
                         pd.crosstab(out[out[f"margin_{fac}"] > 10].reward_group, out[out[f"margin_{fac}"] > 10][f"best_{fac}"]).add_suffix(" >10")],
                        axis=1).fillna(0).astype(int).to_string())
    print("\nnats/trial gain of each mouse's best model over the pooled-best model (quantiles):")
    print(out.groupby("reward_group").apply(lambda e: (e.d_best_vs_pooled / e.n).quantile([.25, .5, .75, .9]).round(4)).to_string())
    print("\nmice with gain > 20 nats:")
    print(out[out.d_best_vs_pooled > 20].sort_values("d_best_vs_pooled", ascending=False)[
        ["mouse_id", "reward_group", "n", "best_model", "d_best_vs_pooled"]].round(1).to_string())

    # figure: per mouse, best level per factor and its margin
    fig, axs = plt.subplots(1, len(FACTORS), figsize=(7.2, 2.2), gridspec_kw=dict(wspace=0.45))
    for ax, fac in zip(axs, FACTORS):
        lev = sorted(out[f"best_{fac}"].unique())
        for i, cs in enumerate(("R+", "R-")):
            e = out[out.reward_group == cs]
            for j, l in enumerate(lev):
                y = e[e[f"best_{fac}"] == l][f"margin_{fac}"].clip(lower=0.1)
                x = j + (i - 0.5) * 0.3 + np.random.default_rng(j + 10 * i).uniform(-0.08, 0.08, len(y))
                ax.scatter(x, y, s=5, color=COH[cs], lw=0, alpha=0.8, label=cs.replace("-", "−") if j == 0 else None)
        ax.axhline(10, color="0.6", lw=0.5, ls=":")
        ax.set_yscale("log"); ax.set_xticks(range(len(lev)), lev, rotation=30)
        ax.set_title(fac)
    axs[0].set_ylabel("held-out margin over next level (nats)")
    axs[0].legend(frameon=False)
    fig.suptitle("Per mouse: which v5 model level predicts the held-out mouse best (cohort parameters fitted without it)", y=1.04)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"024_per_mouse_models.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
