"""022 -- figures for model v5 (021): model comparison, fit vs data over training, day-0 whisker trials (R- extinction), weight
trajectories across models, recovery.
Inputs: fits/021_fits.csv, tables/021_model_comparison.csv, tables/021_latents/*.parquet, fits/021_recovery_fits.csv,
tables/021_recovery_traj.csv, tables/021_generators.csv
Output: figures/022_*.{png,pdf,svg}, tables/022_params.csv, tables/022_recovery_confusion.csv, tables/022_param_recovery.csv
Run (haas): python 022_figures_v5.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
import rl6_model as M

plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.titlesize": 6.5, "legend.fontsize": 5.5})
COH = {"R+": "#00B400", "R-": "#C800C8"}
TC = {"auditory": "#1f78b4", "whisker": "#d95f02", "no_stim": "0.35"}
NB, XMIN = 5, -4
N_TRIALS = 226670


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def lat(m):
    return pd.read_parquet(C.TAB / "021_latents" / f"{m}.parquet")


def traj(L, cols, by=("reward_group",)):
    pos = L.groupby("session_id").cumcount(); n = L.groupby("session_id").session_id.transform("size")
    L = L.assign(x=L.day + np.floor(NB * pos / n) / NB)
    L = L[L.x >= XMIN]
    g = L.groupby(list(by) + ["x"])
    T = g[cols].mean(); T["n_mice"] = g.mouse_id.nunique()
    return T[T.n_mice >= 10].reset_index()


def segs(ax, e, y, **kw):
    for _, e2 in e.groupby(np.floor(e.x)):
        ax.plot(e2.x, e2[y], **kw)


def fig_comparison(S):
    ref = S[S.model == "init-E0-noeng-sym"].iloc[0]
    S = S.assign(d_ll=S.ll_test_per_trial - ref.ll_test_per_trial, d_bic=S.bic - ref.bic)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw=dict(wspace=0.35))
    fams = list(M.FA_MODES)
    variants = [(tr, en, lr) for tr in M.TRANSFER for en in ("noeng", "eng") for lr in ("sym", "asym")]
    for ax, q, lab in [(axs[0], "d_ll", "held-out LL per trial vs init-E0-noeng-sym\n(5-fold over mice; higher = better)"),
                       (axs[1], "d_bic", "ΔBIC vs init-E0-noeng-sym (lower = better)")]:
        for i, fa in enumerate(fams):
            for j, (tr, en, lr) in enumerate(variants):
                r = S[S.model == f"{tr}-{fa}-{en}-{lr}"]
                ax.scatter(i + (j - 3.5) * 0.09, r[q].iloc[0], s=11, marker="o" if tr == "init" else "^",
                           color=("#d95f02" if en == "eng" else "0.5"), alpha=1.0 if lr == "asym" else 0.5, lw=0,
                           label=f"{tr} {en} {lr}" if i == 0 else None)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xticks(range(len(fams)), ["E0 const", "E1 decay", "E2 FA-learn", "E3 reward rate"])
        ax.set_title(lab)
    axs[0].legend(frameon=False, ncol=2, fontsize=4.8)
    best = S.sort_values("ll_test_per_trial", ascending=False).iloc[0]
    fig.suptitle(f"Model v5 comparison (111 mice, cohort-specific parameters; best held-out: {best.model})", y=1.03)
    save(fig, "022_1_model_comparison")


def fig_fit_vs_data(models):
    fig, axs = plt.subplots(len(models), 2, figsize=(7.2, 1.7 * len(models)), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.5, wspace=0.12))
    for i, m in enumerate(models):
        T = traj(lat(m), ["lick_flag", "p_lick"], by=("reward_group", "trial_type"))
        for j, c in enumerate(COH):
            ax = axs[i, j]
            for t in TC:
                e = T[(T.reward_group == c) & (T.trial_type == t)]
                segs(ax, e, "lick_flag", color=TC[t], lw=1.6, alpha=0.45)
                segs(ax, e, "p_lick", color=TC[t], lw=0.8)
            ax.axvline(0, color="0.7", lw=0.5, ls=":")
            ax.set_title(f"{c.replace('-', '−')} — {m}")
            if j == 0:
                ax.set_ylabel("lick rate")
    for ax in axs[-1]:
        ax.set_xlabel("training day (session fifths; 0 = first whisker day)")
    fig.suptitle("Data (thick, light) vs model (thin): blue auditory, orange whisker, grey no stimulus", y=1.0)
    save(fig, "022_2_fit_vs_data")


def fig_day0(models, bin_=10, n_max=150):
    """day-0 whisker (and no-stimulus) trials, by order within day 0: data vs models"""
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True, gridspec_kw=dict(wspace=0.1))
    cols = plt.cm.tab10(np.arange(len(models)))
    for j, c in enumerate(COH):
        ax = axs[j]
        for k, m in enumerate(models):
            L = lat(m)
            L = L[(L.day == 0) & (L.reward_group == c)]
            for t, ls in [("whisker", "-"), ("no_stim", ":")]:
                e = L[L.trial_type == t].copy()
                e["i"] = e.groupby("mouse_id").cumcount() // bin_ * bin_
                e = e[e.i < n_max].groupby("i")[["lick_flag", "p_lick"]].mean()
                if k == 0:
                    ax.plot(e.index, e.lick_flag, color="k", lw=1.8, alpha=0.35, ls=ls, label=f"data {t.replace('_', ' ')}")
                ax.plot(e.index, e.p_lick, color=cols[k], lw=0.9, ls=ls, label=m if t == "whisker" else None)
        ax.set_title(f"{c.replace('-', '−')}: day 0, by trial order within type (bins of {bin_})")
        ax.set_xlabel("whisker / no-stimulus trial number on day 0")
    axs[0].set_ylabel("lick rate"); axs[1].legend(frameon=False, fontsize=4.8)
    save(fig, "022_3_day0_whisker")


def fig_weights(models):
    qs = [("V_A", "auditory value V(A)"), ("V_W", "whisker value V(W)"), ("eps", "exploration ε (FA)"), ("eng", "engagement e")]
    fig, axs = plt.subplots(len(qs), len(models), figsize=(1.55 * len(models), 5.4), sharex=True, sharey="row",
                            gridspec_kw=dict(hspace=0.35, wspace=0.15))
    for j, m in enumerate(models):
        T = traj(lat(m), [q for q, _ in qs])
        for i, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            for c in COH:
                segs(ax, T[T.reward_group == c], q, color=COH[c], lw=1)
            ax.axvline(0, color="0.7", lw=0.5, ls=":"); ax.axhline(0, color="0.85", lw=0.5)
            if i == 0:
                ax.set_title(m.replace("-", "\n", 1), fontsize=5.6)
            if j == 0:
                ax.set_ylabel(lab)
    for ax in axs[-1]:
        ax.set_xlabel("day")
    fig.suptitle("Learned weights and latent factors over training (cohort means; green R+, magenta R−)", y=1.0)
    save(fig, "022_4_weights_across_models")


def fig_recovery():
    R = pd.read_csv(C.FITS / "021_recovery_fits.csv")
    G = pd.read_csv(C.TAB / "021_generators.csv")
    B = R.groupby(["generator", "model"]).agg(nll=("nll_train", "sum"), k=("k", "sum")).reset_index()
    B["bic"] = 2 * B.nll + B.k * np.log(N_TRIALS)
    B["dbic"] = B.bic - B.groupby("generator").bic.transform("min")
    gm = G.drop_duplicates("generator").set_index("generator").model
    rows = []
    for gen, e in B.groupby("generator"):
        e = e.sort_values("dbic")
        rows.append(dict(generator=gen, bic_winner=e.model.iloc[0], runner_up=e.model.iloc[1], dbic_runner_up=e.dbic.iloc[1],
                         dbic_generator=e[e.model == gm[gen]].dbic.iloc[0]))
    conf = pd.DataFrame(rows); conf.to_csv(C.TAB / "022_recovery_confusion.csv", index=False); print(conf.to_string())
    P = []
    for _, g in G.iterrows():
        r = R[(R.generator == g.generator) & (R.model == g.model) & (R.cohort_set == g.cohort_set)].iloc[0]
        for k in M.free_params(g.model):
            P.append(dict(generator=g.generator, cohort=g.cohort_set, param=k, true=g[f"p_{k}"], refit=r[f"p_{k}"]))
    P = pd.DataFrame(P); P.to_csv(C.TAB / "022_param_recovery.csv", index=False)
    Tr = pd.read_csv(C.TAB / "021_recovery_traj.csv")
    gens = list(conf.generator)
    qs = [("V_W", "whisker value V(W)"), ("V_A", "auditory value V(A)"), ("eps", "exploration ε"), ("eng", "engagement e")]
    fig, axs = plt.subplots(len(qs), len(gens), figsize=(1.65 * len(gens), 5.2), sharex=True, sharey="row", gridspec_kw=dict(hspace=0.4, wspace=0.15))
    for j, gen in enumerate(gens):
        for i, (q, lab) in enumerate(qs):
            ax = axs[i, j]
            for src, ls in [("true", "-"), (f"refit {gm[gen]}", "--")]:
                e = Tr[(Tr.generator == gen) & (Tr.source == src) & (Tr.x >= XMIN)]
                for c in COH:
                    segs(ax, e[e.reward_group == c], q, color=COH[c], ls=ls, lw=0.9)
            if i == 0:
                w = conf[conf.generator == gen].iloc[0]
                ax.set_title(f"gen {gen}\nBIC winner {w.bic_winner}", fontsize=5.0)
            if j == 0:
                ax.set_ylabel(lab)
    fig.suptitle("Recovery on simulated data: true (solid) vs refit generator model (dashed)", y=1.02)
    save(fig, "022_5_recovery")


def main():
    S = pd.read_csv(C.TAB / "021_model_comparison.csv")
    fig_comparison(S)
    best = S.sort_values("ll_test_per_trial", ascending=False).model.iloc[0]
    tr, fa, en, lr = best.split("-")
    other_tr = "overlap" if tr == "init" else "init"
    sel = [best, f"{other_tr}-{fa}-{en}-{lr}", f"{tr}-{fa}-{'noeng' if en == 'eng' else 'eng'}-{lr}",
           f"{tr}-{fa}-{en}-{'sym' if lr == 'asym' else 'asym'}"] + [f"{tr}-{f}-{en}-{lr}" for f in M.FA_MODES if f != fa]
    fig_fit_vs_data(sel[:3])
    fig_day0(sel[:4])
    fig_weights(sel)
    F = pd.read_csv(C.FITS / "021_fits.csv")
    F[(F.fold.astype(str) == "full") & F.model.isin(sel)].to_csv(C.TAB / "022_params.csv", index=False)
    try:
        fig_recovery()
    except (FileNotFoundError, IndexError) as e:
        print("recovery incomplete:", e)


if __name__ == "__main__":
    main()
