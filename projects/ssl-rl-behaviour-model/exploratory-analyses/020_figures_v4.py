"""020 -- figures for model v4 (019): model comparison, fit vs data over training, weight trajectories across models, recovery.
Inputs: fits/019_fits.csv, tables/019_model_comparison.csv, tables/019_latents/*.parquet, fits/019_recovery_fits.csv,
tables/019_recovery_traj.csv. Output: figures/020_*.{png,pdf,svg}, tables/020_params.csv, tables/020_recovery_confusion.csv
Trajectories: x = day + within-session fifth (sessions split in 5 equal bins); a point is shown when >= 10 mice contribute.
Run (haas): python 020_figures_v4.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
import rl5_model as M

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.titlesize": 6.5, "legend.fontsize": 5.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
TC = {"auditory": "#1f78b4", "whisker": "#d95f02", "no_stim": "0.35"}
NB = 5
XMIN = -4


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def xpos(L):
    pos = L.groupby("session_id").cumcount(); n = L.groupby("session_id").session_id.transform("size")
    return L.day + np.floor(NB * pos / n) / NB


def traj(L, cols, by=("reward_group",)):
    L = L.assign(x=xpos(L))
    L = L[L.x >= XMIN]
    g = L.groupby(list(by) + ["x"])
    T = g[cols].mean()
    T["n_mice"] = g.mouse_id.nunique()
    return T[T.n_mice >= 10].reset_index()


def lat(m, g):
    return pd.read_parquet(C.TAB / "019_latents" / f"{m}_{g}.parquet")


def fig_comparison(S):
    ref = S[(S.model == "E0-sym-1ts") & (S.grouping == "shared")].iloc[0]
    S = S.assign(d_ll=S.ll_test_per_trial - ref.ll_test_per_trial, d_bic=S.bic - ref.bic)
    fams = list(M.FA_MODES)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.35))
    for ax, q, lab in [(axs[0], "d_ll", "held-out LL per trial vs E0-sym-1ts shared\n(5-fold over mice; higher = better)"),
                       (axs[1], "d_bic", "ΔBIC vs E0-sym-1ts shared (full fit; lower = better)")]:
        for i, fa in enumerate(fams):
            for j, v in enumerate(["sym-1ts", "asym-1ts", "sym-2ts", "asym-2ts"]):
                for gk, (g, mk) in enumerate([("shared", "o"), ("cohort", "s")]):
                    r = S[(S.model == f"{fa}-{v}") & (S.grouping == g)]
                    if len(r):
                        ax.scatter(i + (j - 1.5) * 0.18, r[q].iloc[0], marker=mk, s=12, color=plt.cm.viridis(j / 3),
                                   edgecolor="k" if g == "cohort" else "none", lw=0.4,
                                   label=f"{v} {g}" if i == 0 else None)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(range(len(fams)), ["E0 const", "E1 decay", "E2 FA-learn", "E3 reward rate", "B bias learn"])
        ax.set_title(lab)
    axs[0].legend(frameon=False, ncol=2, loc="lower right")
    best = S.sort_values("ll_test_per_trial", ascending=False).iloc[0]
    fig.suptitle(f"Model v4 comparison (111 mice; best held-out: {best.model} {best.grouping})", y=1.03)
    save(fig, "020_1_model_comparison")


def fig_fit_vs_data(models):
    fig, axs = plt.subplots(len(models), 2, figsize=(7.2, 1.7 * len(models)), sharex=True, sharey=True,
                            gridspec_kw=dict(hspace=0.5, wspace=0.12))
    axs = np.atleast_2d(axs)
    for i, (m, g) in enumerate(models):
        T = traj(lat(m, g), ["lick_flag", "p_lick"], by=("reward_group", "trial_type"))
        for j, c in enumerate(["R+", "R-"]):
            ax = axs[i, j]
            for t in TC:
                e = T[(T.reward_group == c) & (T.trial_type == t)]
                for d0, e2 in e.groupby(np.floor(e.x)):
                    ax.plot(e2.x, e2.lick_flag, color=TC[t], lw=1.6, alpha=0.45)
                    ax.plot(e2.x, e2.p_lick, color=TC[t], lw=0.8)
            ax.axvline(0, color="0.7", lw=0.5, ls=":")
            ax.set_title(f"{c.replace('-', '−')} — {m} ({g})")
            if j == 0:
                ax.set_ylabel("lick rate")
    for ax in axs[-1]:
        ax.set_xlabel("training day (session fifths; 0 = first whisker day)")
    fig.suptitle("Data (thick, light) vs model (thin): blue auditory, orange whisker, grey no stimulus", y=1.0)
    save(fig, "020_2_fit_vs_data")


def fig_weights(models):
    qs = [("V_A", "auditory value V(A)"), ("V_W", "whisker value V(W)"), ("eps", "exploration ε (FA drive)")]
    fig, axs = plt.subplots(len(qs), len(models), figsize=(1.55 * len(models), 4.2), sharex=True, sharey="row",
                            gridspec_kw=dict(hspace=0.35, wspace=0.15))
    for j, (m, g) in enumerate(models):
        T = traj(lat(m, g), [q for q, _ in qs])
        for i, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            for c in ["R+", "R-"]:
                e = T[T.reward_group == c]
                for _, e2 in e.groupby(np.floor(e.x)):
                    ax.plot(e2.x, e2[q], color=COH[c], lw=1)
            ax.axvline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(0, color="0.85", lw=0.5)
            if i == 0:
                ax.set_title(f"{m}\n{g}", fontsize=5.8)
            if j == 0:
                ax.set_ylabel(lab)
    for ax in axs[-1]:
        ax.set_xlabel("day")
    fig.suptitle("Learned weights over training (cohort means; green R+, magenta R−)", y=1.0)
    save(fig, "020_3_weights_across_models")


def fig_recovery():
    R = pd.read_csv(C.FITS / "019_recovery_fits.csv")
    B = R.groupby(["generator", "model", "grouping"]).agg(nll=("nll_train", "sum"), k=("k", "sum")).reset_index()
    n = 226670
    B["bic"] = 2 * B.nll + B.k * np.log(n)
    B["dbic"] = B.bic - B.groupby("generator").bic.transform("min")
    win = B.loc[B.groupby("generator").bic.idxmin()]
    true = B[B.model == B.generator].groupby("generator").dbic.min()
    conf = win[["generator", "model", "grouping"]].rename(columns=dict(model="bic_winner")).assign(dbic_generator=true.reindex(win.generator).values)
    conf.to_csv(C.TAB / "020_recovery_confusion.csv", index=False)
    print(conf.to_string())
    # parameter recovery: generator model refit (cohort grouping) vs generating values
    F = pd.read_csv(C.FITS / "019_fits.csv"); F = F[F.fold.astype(str) == "full"]
    rows = []
    for gen in R.generator.unique():
        for cs in ["R+", "R-"]:
            t = F[(F.model == gen) & (F.grouping == "cohort") & (F.cohort_set == cs)].iloc[0]
            r = R[(R.generator == gen) & (R.model == gen) & (R.grouping == "cohort") & (R.cohort_set == cs)].iloc[0]
            for k in M.free_params(gen):
                rows.append(dict(generator=gen, cohort=cs, param=k, true=t[f"p_{k}"], refit=r[f"p_{k}"]))
    P = pd.DataFrame(rows); P.to_csv(C.TAB / "020_param_recovery.csv", index=False)
    Tr = pd.read_csv(C.TAB / "019_recovery_traj.csv")
    gens = list(R.generator.unique())
    fig, axs = plt.subplots(2, len(gens), figsize=(1.6 * len(gens), 3.2), sharex=True, sharey="row", gridspec_kw=dict(hspace=0.4, wspace=0.15))
    for j, gen in enumerate(gens):
        for i, q in enumerate(["V_W", "eps"]):
            ax = axs[i, j]
            for src, ls in [("true", "-"), (f"refit {gen} cohort", "--")]:
                e = Tr[(Tr.generator == gen) & (Tr.source == src) & (Tr.x >= XMIN)]
                for c in ["R+", "R-"]:
                    for _, e2 in e[e.reward_group == c].groupby(np.floor(e[e.reward_group == c].x)):
                        ax.plot(e2.x, e2[q], color=COH[c], ls=ls, lw=0.9)
            w = conf[conf.generator == gen].iloc[0]
            if i == 0:
                ax.set_title(f"gen {gen}\nBIC winner {w.bic_winner} {w.grouping}", fontsize=5.5)
            if j == 0:
                ax.set_ylabel({"V_W": "whisker value V(W)", "eps": "exploration ε"}[q])
    fig.suptitle("Recovery on simulated data: true (solid) vs refit generator model (dashed)", y=1.02)
    save(fig, "020_4_recovery")


def main():
    S = pd.read_csv(C.TAB / "019_model_comparison.csv")
    fig_comparison(S)
    best = S.sort_values("ll_test_per_trial", ascending=False).iloc[0]
    fa, lr, ts = best.model.split("-")
    sel = [(best.model, best.grouping)]
    sel += [(f"{fa}-{a}-{b}", best.grouping) for a in ("sym", "asym") for b in ("1ts", "2ts") if f"{fa}-{a}-{b}" != best.model]
    sel += [(f"{f}-{lr}-{ts}", best.grouping) for f in M.FA_MODES if f != fa]
    fig_fit_vs_data([sel[0], (f"B-{lr}-{ts}", best.grouping), (f"E0-{lr}-{ts}", best.grouping)])
    fig_weights(sel)
    F = pd.read_csv(C.FITS / "019_fits.csv")
    F = F[(F.fold.astype(str) == "full")]
    F[[(m, g) in sel for m, g in zip(F.model, F.grouping)]].to_csv(C.TAB / "020_params.csv", index=False)
    try:
        fig_recovery()
    except FileNotFoundError as e:
        print("no recovery yet:", e)


if __name__ == "__main__":
    main()
