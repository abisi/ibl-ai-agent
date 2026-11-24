"""023 -- learning ethograms (haas, full run): session-wide behaviour composition next to the learning curves.

Input: tables/full_trial_measures.parquet (015; per kept trial x epoch: fraction of frames in each lightning-action class
and kpms side meta-state, all frames), full_sessions.csv (cohort, learner), learning trials
(ssl-learning-trial-identification 017 lt_final = whisker-trial index).
Epochs (bm_lib.trial_windows): pre (2 s), resp, post (response end -> +5 s), iti (+5 s -> next quiet window).
Whisker-trial axis: every kept trial gets the number of whisker trials before it (auditory / no-stim trials sit between
whisker trials), so all trial types share one axis.
Views:
 1. aligned to the learning trial (mice with an LT): whisker trials -40..+60 relative to the LT, bins of 5 trials;
 2. normalised session position (all mice): 10 bins of the whisker-trial axis;
 per view and cohort: P(lick) per trial type (mean over mice of the per-mouse bin means) and, per epoch, the stacked
 composition (mean over mice) of the lightning-action classes (still / whisking / jaw / lick / unclassified) or of the
 kpms side meta-states; whisker trials for the composition rows.
 3. per-mouse ethograms: rows = mice (sorted by LT, then mice without an LT by whisker P(lick)), columns = whisker trials
 (bins of 3 trials from the first): hit / miss, pre-stim P(still), resp P(whisking), resp P(lick), post P(lick).
Outputs: figures/023_learning_ethogram_{actions,meta}.png|pdf|svg, figures/023_mouse_ethograms.png|pdf|svg,
tables/023_ethogram_aligned.csv, tables/023_ethogram_normalised.csv.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
T = L.TABLES
LT_TABLE = L.ROOT / "combined_results_ks4" / "ssl-learning-trial-identification" / "artifacts" / "017_learning_trials_final.csv"
EPOCHS = [("pre", "pre-stim (2 s)"), ("resp", "response window"), ("post", "post-response (0-5 s)"), ("iti", "ITI (+5 s → next quiet)")]
ACT = [("act_still", "still", "#4d4d4d"), ("act_whisking", "whisking", "#1f77b4"), ("act_jaw", "jaw", "#ff7f0e"),
       ("act_lick", "lick", "#d62728")]
META_COL = ["#4d4d4d", "#17becf", "#ff7f0e", "#9467bd"]
ALIGN = np.arange(-40, 61, 5)
NBIN = 10


def load():
    D = pd.read_parquet(T / "full_trial_measures.parquet")
    S = pd.read_csv(T / "full_sessions.csv")
    lt = pd.read_csv(LT_TABLE)
    S["lt_trial"] = S.session_id.map(dict(zip(lt.session_id, lt.lt_final)))
    D = D[D.session_id.isin(S[S.rate_side > 150].session_id)].copy()
    D["cohort"] = D.session_id.map(dict(zip(S.session_id, S.reward_group)))
    # whisker-trial axis: whisker trials before each trial (per session, over kept trials)
    tr = D.drop_duplicates(["session_id", "trial"])[["session_id", "trial", "stim"]].sort_values(["session_id", "trial"])
    tr["wpos"] = tr.groupby("session_id").stim.transform(lambda s: (s == "whisker").cumsum().shift(fill_value=0))
    tr.loc[tr.stim == "whisker", "wpos"] = tr.groupby("session_id").stim.transform(lambda s: (s == "whisker").cumsum() - 1)
    D = D.merge(tr[["session_id", "trial", "wpos"]], on=["session_id", "trial"])
    nw = D[D.stim == "whisker"].groupby("session_id").wpos.max() + 1
    D["wnorm"] = D.wpos / D.session_id.map(nw)
    D["rel"] = D.wpos - D.session_id.map(dict(zip(S.session_id, S.lt_trial)))
    return D, S


def binned(D, xcol, edges, meas):
    """per mouse x bin mean, then mean / sem over mice; returns dict measure -> (mean, sem, n)"""
    g = D.assign(bin=np.digitize(D[xcol], edges) - 1)
    g = g[(g.bin >= 0) & (g.bin < len(edges) - 1)]
    pm = g.groupby(["session_id", "bin"])[meas].mean()
    out = {}
    for m in meas:
        p = pm[m].unstack("bin").reindex(columns=range(len(edges) - 1))
        n = p.notna().sum(0)
        out[m] = (p.mean(0).to_numpy(), (p.std(0) / np.sqrt(n.clip(1))).to_numpy(), n.to_numpy())
    return out


def ethogram_figure(D, kind, name):
    if kind == "actions":
        comp = ACT
    else:
        mn = list(pd.read_csv(T / "full_009_metastates.csv").name)
        comp = [(f"meta_side_M{k}", n, META_COL[k]) for k, n in enumerate(mn) if f"meta_side_M{k}" in D]
    meas = [c for c, _, _ in comp]
    views = [("aligned to the learning trial", "rel", np.r_[ALIGN, ALIGN[-1] + 5], "whisker trials from the learning trial",
              lambda d: d[d.rel.notna()]),
             ("normalised session position", "wnorm", np.linspace(0, 1.0001, NBIN + 1), "fraction of whisker trials",
              lambda d: d)]
    fig = plt.figure(figsize=(7.4, 9.6))
    outer = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.18, left=0.07, right=0.98, top=0.93, bottom=0.05)
    rows = []
    for vi, (vtitle, xcol, edges, xlab, sel) in enumerate(views):
        for ci, cg in enumerate(("R+", "R-")):
            d0 = sel(D[D.cohort == cg])
            inner = outer[vi, ci].subgridspec(1 + len(EPOCHS), 1, hspace=0.25, height_ratios=[1.3] + [1] * len(EPOCHS))
            ctr = (edges[:-1] + edges[1:]) / 2
            ax = fig.add_subplot(inner[0])
            for stim in ("whisker", "auditory", "no_stim"):
                dd = d0[(d0.stim == stim) & (d0.epoch == "resp")]
                b = binned(dd, xcol, edges, ["lick_flag"])["lick_flag"]
                ax.fill_between(ctr, b[0] - b[1], b[0] + b[1], color=L.TASK_COL[stim], alpha=0.2, lw=0)
                ax.plot(ctr, b[0], color=L.TASK_COL[stim], lw=0.9)
                for x, m_, e_, n_ in zip(ctr, *b):
                    rows.append(dict(view=vtitle, cohort=cg, panel="P(lick)", measure=stim, x=x, mean=m_, sem=e_, n_mice=n_))
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel("P(lick)", fontsize=5)
            ax.set_xticklabels([])
            nm = d0.session_id.nunique()
            ax.set_title(f"{cg} — {vtitle} ({nm} mice)", fontsize=6, color=L.COH_COL[cg])
            if xcol == "rel":
                ax.axvline(0, color="r", lw=0.6, ls="--")
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            for ei, (ep, elab) in enumerate(EPOCHS):
                ax = fig.add_subplot(inner[1 + ei])
                dd = d0[(d0.stim == "whisker") & (d0.epoch == ep)]
                b = binned(dd, xcol, edges, meas)
                base = np.zeros(len(ctr))
                for c, lab, col in comp:
                    v = np.nan_to_num(b[c][0])
                    ax.fill_between(ctr, base, base + v, color=col, lw=0, alpha=0.9, label=lab, step=None)
                    base = base + v
                    for x, m_, e_, n_ in zip(ctr, *b[c]):
                        rows.append(dict(view=vtitle, cohort=cg, panel=ep, measure=c, x=x, mean=m_, sem=e_, n_mice=n_))
                ax.set_ylim(0, 1)
                ax.set_xlim(ctr[0], ctr[-1])
                ax.set_ylabel(elab, fontsize=4.5)
                ax.tick_params(labelsize=4.5, length=1.5, pad=1)
                if xcol == "rel":
                    ax.axvline(0, color="r", lw=0.6, ls="--")
                if ei < len(EPOCHS) - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(xlab, fontsize=5)
                if vi == 0 and ci == 1 and ei == 0:
                    ax.legend(frameon=False, fontsize=4.5, ncol=len(comp), loc="lower right", bbox_to_anchor=(1, 1.0),
                              handlelength=0.8)
    what = "lightning-action classes (white = unclassified)" if kind == "actions" else "kpms side meta-states"
    fig.suptitle(f"Learning ethogram: P(lick) per trial type (orange whisker, blue auditory, black no-stim) and the composition of "
                 f"behaviour on whisker trials per epoch, {what}; mean over mice of per-mouse bin means; red dashed = learning trial",
                 fontsize=5.8, x=0.01, ha="left", wrap=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"023_learning_ethogram_{name}.{ext}", dpi=300)
    plt.close(fig)
    return pd.DataFrame(rows)


def mouse_ethograms(D, S):
    W = D[D.stim == "whisker"]
    panels = [("hit", "resp", "lick_flag", "Greys", (0, 1)), ("pre-stim P(still)", "pre", "act_still", "Greys", (0, 1)),
              ("resp P(whisking)", "resp", "act_whisking", "Blues", (0, 0.8)), ("resp P(lick)", "resp", "act_lick", "Reds", (0, 0.8)),
              ("post P(lick)", "post", "act_lick", "Reds", (0, 0.6))]
    BINW = 3
    fig = plt.figure(figsize=(7.4, 9.6))
    gs = fig.add_gridspec(2, len(panels), hspace=0.12, wspace=0.08, left=0.06, right=0.97, top=0.94, bottom=0.05)
    lt = dict(zip(S.session_id, S.lt_trial))
    for ci, cg in enumerate(("R+", "R-")):
        ss = S[(S.reward_group == cg) & S.session_id.isin(W.session_id)]
        ss = ss.assign(o1=ss.lt_trial.isna(), o2=np.where(ss.lt_trial.notna(), ss.lt_trial, -ss.p_lick_whisker)).sort_values(["o1", "o2"])
        sids = list(ss.session_id)
        nmax = int(W[W.session_id.isin(sids)].wpos.max() // BINW + 1)
        nmax = min(nmax, 100)
        for pi, (ptitle, ep, col, cmap, lim) in enumerate(panels):
            ax = fig.add_subplot(gs[ci, pi])
            M = np.full((len(sids), nmax), np.nan)
            for i, s in enumerate(sids):
                g = W[(W.session_id == s) & (W.epoch == ep)]
                b = (g.wpos // BINW).astype(int)
                v = g.groupby(b)[col].mean()
                v = v[v.index < nmax]
                M[i, v.index] = v.to_numpy()
            ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest",
                      extent=[0, nmax * BINW, len(sids), 0])
            for i, s in enumerate(sids):
                if np.isfinite(lt[s]):
                    ax.plot([lt[s], lt[s]], [i, i + 1], color="#00c0ff", lw=1.0)
            ax.set_title(ptitle if ci == 0 else "", fontsize=5.5)
            ax.set_xlim(0, min(nmax * BINW, 200))
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            if pi == 0:
                ax.set_ylabel(f"{cg} mice (sorted by learning trial; no LT at the bottom)", fontsize=5, color=L.COH_COL[cg])
                ax.set_yticks(np.arange(len(sids)) + 0.5)
                ax.set_yticklabels([s.split("_")[0] for s in sids], fontsize=3.2)
            else:
                ax.set_yticks([])
            if ci == 1:
                ax.set_xlabel("whisker trial", fontsize=5)
    fig.suptitle(f"Per-mouse learning ethograms: whisker trials in bins of {BINW} (blue tick = learning trial); hit rate, "
                 "pre-stimulus stillness, response-window whisking and licking, post-response licking (lightning-action)",
                 fontsize=5.8, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"023_mouse_ethograms.{ext}", dpi=300)
    plt.close(fig)


def main():
    D, S = load()
    a = ethogram_figure(D, "actions", "actions")
    m = ethogram_figure(D, "meta", "meta")
    out = pd.concat([a.assign(kind="actions"), m.assign(kind="meta")])
    out[out.view.str.startswith("aligned")].to_csv(T / "023_ethogram_aligned.csv", index=False)
    out[out.view.str.startswith("normalised")].to_csv(T / "023_ethogram_normalised.csv", index=False)
    mouse_ethograms(D, S)
    print(D.groupby("cohort").session_id.nunique().to_dict(), "mice with LT:",
          D[D.rel.notna()].groupby("cohort").session_id.nunique().to_dict())


if __name__ == "__main__":
    main()
