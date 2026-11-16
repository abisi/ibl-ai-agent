"""034 -- model evaluation on binned data (user 2026-10-29: "evaluate model performance not on single trials but on binned data, e.g.
10-trial bins").

Bins: consecutive blocks of B same-type trials within a session (auditory, whisker, no stimulus separately; a last partial block is
kept if it has >= B/2 trials). Per bin: n trials, k licks, data rate y = k/n, model rate p = mean causal P(lick) over the bin.
Models:
  held-out  cohort-level fits from the 030 leave-mice-out folds (each mouse predicted with the parameters fitted on the other folds;
            the per-fold parameters are stored in tables/030_cv*.csv): lean-abs-stim (final), +hom, -nodis, simple, simple-m1, simple-gen
  in-sample per-mouse hierarchical fits (028_<tag>_trial_latents.parquet): 025S (final), 025G (simple-gen)
Per mouse (and per mouse x trial type), on days -1..2:
  deviance D = 2 sum[k log(y/p) + (n-k) log((1-y)/(1-p))] (binomial, vs the saturated bin model); dispersion phi = D / n_bins
  (~1 if the bins are consistent with the model up to binomial noise); same for the strong baseline: each mouse's per-day per-type lick
  rate (in-sample); R2_dev = 1 - D_model / D_day (within-day dynamics explained beyond the daily rates); explained = (D_day - D_model) /
  (D_day - n_bins) (fraction of the explainable deviance, the binomial noise floor being D ~ n_bins); R2_bin = 1 - SSE / SST of the
  bin rates around the mouse's per-type mean (classic variance explained of the learning curves).
Output: tables/034_binned_B<B>_mouse.csv, tables/034_binned_B<B>_summary.csv, tables/034_binned_B<B>_bins.parquet,
figures/pub_simple/figS5_binned_eval_B<B>.{png,pdf,svg}
Run (haas): python 034_binned_eval.py [--bin 10] [--n-jobs 40]
  core screen (2026-10-30): --cv-files 030_cv_core.csv 030_cv.csv --insample --order cv:core cv:core-rec ... --key cv:core cv:lean-abs-stim
  --tag core (outputs 034<tag>_binned_B<B>_*, figure figS5<tag>_binned_eval_B<B>)
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy import stats

import common as C
import rl7_model as M

L027 = importlib.import_module("027_lean_model")
F019 = importlib.import_module("019_fit_v4")
F023 = importlib.import_module("023_v6_prediction_check")
F = importlib.import_module("026_pub_figures")
plt = F.plt
CV_FILES = ["030_cv.csv", "030_cv_nodis.csv", "030_cv_simple.csv", "030_cv_gen.csv"]
INSAMPLE = {"025S": "final, per-mouse fit", "025G": "simple-gen, per-mouse fit"}
MLAB = {"core": "core (simple-gen + HMM)", "lean-abs-stim": "final (HMM, lag, κ, θ, ctx)", "lean-abs-stim-hom": "final + thirst-scaled reward",
        "lean-abs-stim-nodis": "final without HMM", "simple": "simple", "simple-m1": "simple, unrewarded −1",
        "simple-gen": "simple-gen", "simple-genS": "simple-gen, shared g_W0"}
SHORT = {"cv:core": "core", "cv:core-rec": "core\n+rec", "cv:core-fast": "core\n+fast", "cv:core-pers": "core\n+pers",
         "cv:core-gw": "core\n+gw", "cv:core-hab": "core\n+hab", "cv:core-habr": "core\n+habr", "cv:lean-abs-stim": "final", "cv:lean-abs-stim-hom": "final\n+hom", "cv:lean-abs-stim-nodis": "final\n−HMM",
         "cv:simple": "simple", "cv:simple-gen": "simple\n-gen", "mouse:025S": "final\nmouse", "mouse:025G": "simple\n-gen\nmouse"}
DAYS = range(-1, 3)
KEY = ["cv:lean-abs-stim", "cv:simple-gen"]
TAG = ""
EPS = 1e-4
D = None
CV = None


def predict(args):
    """held-out causal P(lick) of one mouse from the fold parameters fitted without it"""
    variant, cs, m = args
    name, fp = L027.configure(variant, cs)
    r = CV[(CV.variant == variant) & (CV.cohort_set == cs) & (CV.mouse_id == m)].iloc[0]
    p = dict(M.FIXED) | {q: float(r[f"p_{q}"]) for q in fp}
    d = F023.subset(D, cs)
    d = d[d.mouse_id == m]
    o = d[["mouse_id", "session_id", "day", "trial_type", "lick_flag", "reward_group"]].copy()
    o["p_lick"] = M.latents(name, p, M.arrays(d))["p_lick"]
    o["model"] = variant
    return o


def bins(L, B):
    L = L.copy()
    L["b"] = L.groupby(["model", "session_id", "trial_type"]).cumcount() // B
    G = (L.groupby(["model", "reward_group", "mouse_id", "day", "session_id", "trial_type", "b"], observed=True)
         .agg(n=("lick_flag", "size"), k=("lick_flag", "sum"), p=("p_lick", "mean")).reset_index())
    G = G[G.n >= max(2, B // 2)].copy()
    rate = L.groupby(["model", "mouse_id", "day", "trial_type"], observed=True).lick_flag.mean().rename("p_day")
    G = G.join(rate, on=["model", "mouse_id", "day", "trial_type"])
    G["y"] = G.k / G.n
    return G


def dev(k, n, p):
    p = np.clip(p, EPS, 1 - EPS); y = k / n
    a = np.where(k > 0, k * np.log(np.where(k > 0, y, 1) / p), 0.0)
    b = np.where(n - k > 0, (n - k) * np.log(np.where(n - k > 0, 1 - y, 1) / (1 - p)), 0.0)
    return 2 * (a + b)


def scores(G):
    G = G.assign(dm=dev(G.k, G.n, G.p), dd=dev(G.k, G.n, G.p_day))
    G["ybar"] = G.groupby(["model", "mouse_id", "trial_type"]).y.transform("mean")
    rows = []
    for by, typ in ((["model", "reward_group", "mouse_id"], "all"), (["model", "reward_group", "mouse_id", "trial_type"], None)):
        for key, g in G.groupby(by, observed=True):
            nb = len(g); Dm, Dd = g.dm.sum(), g.dd.sum()
            sst = ((g.y - g.ybar) ** 2).sum()
            rows.append(dict(model=key[0], reward_group=key[1], mouse_id=key[2], trial_type=typ or key[3], n_bins=nb,
                             dev_model=Dm, dev_day=Dd, phi_model=Dm / nb, phi_day=Dd / nb, r2_dev=1 - Dm / Dd if Dd > 0 else np.nan,
                             explained=(Dd - Dm) / (Dd - nb) if Dd - nb > 0 else np.nan,
                             r2_bin=1 - ((g.y - g.p) ** 2).sum() / sst if sst > 0 else np.nan))
    return pd.DataFrame(rows)


def summary(S):
    rows = []
    for (m, t), g in S.groupby(["model", "trial_type"]):
        r = dict(model=m, trial_type=t)
        for cs in ("R+", "R-"):
            h = g[g.reward_group == cs]
            for q in ("phi_model", "phi_day", "r2_dev", "explained", "r2_bin"):
                r[f"{q}_{cs}"] = h[q].median()
            r[f"n_{cs}"] = len(h)
            r[f"frac_phi_le_1.5_{cs}"] = (h.phi_model <= 1.5).mean()
            r[f"frac_r2dev_gt0_{cs}"] = (h.r2_dev > 0).mean()
        a, b = g[g.reward_group == "R+"].r2_dev.dropna(), g[g.reward_group == "R-"].r2_dev.dropna()
        if len(a) > 2 and len(b) > 2:
            r["r2_dev_mwu_p"] = stats.mannwhitneyu(a, b).pvalue
            r["r2_dev_welch_p"] = stats.ttest_ind(a, b, equal_var=False).pvalue
        rows.append(r)
    return pd.DataFrame(rows)


def strip(ax, S, models, col, ylab, ref=None):
    for i, m in enumerate(models):
        for j, cs in enumerate(F.COH):
            y = S[(S.model == m) & (S.reward_group == cs) & (S.trial_type == "all")][col].dropna()
            x = i + (j - 0.5) * 0.36
            ax.scatter(x + np.random.default_rng(i * 2 + j).uniform(-0.08, 0.08, len(y)), y, s=2.5, lw=0, color=F.COH[cs], alpha=0.5)
            if len(y):
                ax.errorbar(x, y.median(), yerr=[[y.median() - y.quantile(0.25)], [y.quantile(0.75) - y.median()]], fmt="o", ms=2.2,
                            color="k", lw=0.6)
    if ref is not None:
        ax.axhline(ref, color="0.6", lw=0.5, ls="--")
    ax.set_xticks(range(len(models)), [SHORT.get(m, m) for m in models], fontsize=5)
    ax.set_ylabel(ylab)


def figure(S, G, B, models):
    fig = plt.figure(figsize=(F.W, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.75, wspace=0.42, left=0.07, right=0.985, top=0.93, bottom=0.08)
    ax = fig.add_subplot(gs[0, 0]); strip(ax, S, models, "phi_model", "dispersion φ = deviance / bin", ref=1)
    ax.set_yscale("log"); ax.set_title("bins vs model: φ ≈ 1 = binomial noise only", fontsize=6)
    ax2 = fig.add_subplot(gs[0, 1]); strip(ax2, S, models, "r2_dev", "deviance R² vs per-day rate", ref=0)
    ax2.set_title("within-day dynamics beyond daily rates", fontsize=6)
    lo = S[S.trial_type == "all"].r2_dev.quantile(0.02)
    ax2.set_ylim(max(lo, -1.0) - 0.05, 1.0)
    ax3 = fig.add_subplot(gs[0, 2]); strip(ax3, S, models, "r2_bin", "R² of bin rates (around mouse mean)", ref=0)
    ax3.set_ylim(max(S[S.trial_type == "all"].r2_bin.quantile(0.02), -1.0) - 0.05, 1.0)
    ax3.set_title("learning-curve variance explained", fontsize=6)
    # per trial type for two key models
    key = [m for m in KEY if m in models]
    ax4 = fig.add_subplot(gs[1, 0:2])
    xt, xl = [], []
    for i, m in enumerate(key):
        for j, t in enumerate(F.TC):
            for c, cs in enumerate(F.COH):
                y = S[(S.model == m) & (S.trial_type == t) & (S.reward_group == cs)].r2_dev.dropna()
                x = i * 4 + j + (c - 0.5) * 0.36
                ax4.scatter(x + np.random.default_rng(i * 100 + j * 10 + c).uniform(-0.07, 0.07, len(y)), y, s=2.5, lw=0, color=F.COH[cs],
                            alpha=0.5)
                if len(y):
                    ax4.errorbar(x, y.median(), yerr=[[y.median() - y.quantile(0.25)], [y.quantile(0.75) - y.median()]], fmt="o",
                                 ms=2.2, color=F.TC[t], lw=0.7)
            xt.append(i * 4 + j); xl.append(f"{F.TLAB[t].split(' (')[0]}")
        ax4.text(i * 4 + 1, 1.04, MLAB.get(m.split(":")[-1], m) + " (held-out)", ha="center", fontsize=5.5, transform=ax4.get_xaxis_transform())
    ax4.axhline(0, color="0.6", lw=0.5, ls="--"); ax4.set_xticks(xt, xl, fontsize=5)
    ax4.set_ylim(-1.05, 1.0); ax4.set_ylabel("deviance R² vs per-day rate")
    # example: binned data vs model, one R+ and one R- mouse, days 0-1, whisker and auditory
    ax5 = fig.add_subplot(gs[1, 2])
    g = G[(G.model == key[0]) & G.day.isin([0, 1])]
    s0 = S[(S.model == key[0]) & (S.trial_type == "all")]
    for cs, off in (("R+", 0.0), ("R-", -1.15)):
        h = s0[s0.reward_group == cs]
        m = h.loc[(h.r2_dev - h.r2_dev.median()).abs().idxmin(), "mouse_id"]
        for t in ("whisker", "auditory", "no_stim"):
            e = g[(g.mouse_id == m) & (g.trial_type == t)].sort_values(["day", "session_id", "b"])
            x = np.arange(len(e))
            se = np.sqrt(np.clip(e.y * (1 - e.y), 0.02, None) / e.n)
            ax5.errorbar(x, e.y + off, yerr=se, fmt="o", ms=1.5, lw=0.4, color=F.TC[t], alpha=0.6)
            ax5.plot(x, e.p + off, color=F.TC[t], lw=0.8)
            ax5.plot(x, e.p_day + off, color=F.TC[t], lw=0.5, ls=":")
        ax5.text(0, 1.06 + off, f"{m} ({F.cs_lab(cs) if hasattr(F, 'cs_lab') else cs}), median mouse", color=F.COH[cs], fontsize=5)
    ax5.set_yticks([-1.15, -0.65, -0.15, 0, 0.5, 1], ["0", "", "1", "0", "", "1"])
    ax5.set_xlabel(f"bin of {B} same-type trials (days 0–1)"); ax5.set_ylabel("P(lick)")
    ax5.set_title(f"dots: data ± s.e.; line: held-out {SHORT.get(key[0], key[0]).replace(chr(10), ' ')};\ndotted: per-day rate",
                  fontsize=5.5)
    F.letter_row(fig, [ax, ax2, ax3], "abc"); F.letter_row(fig, [ax4, ax5], "de")
    fig.text(0.07, 0.535, "final = lean-abs-stim (HMM, absorption lag, κ, θ, context step); +hom = thirst-scaled reward; −HMM = no "
             "engagement states. Held-out: predicted with cohort parameters fitted on\nthe other folds of mice; mouse: per-mouse "
             "hierarchical fit (in-sample). Green R+, magenta R−; dots = mice.", fontsize=5, color="0.35", va="top")
    fig.suptitle(f"Model evaluation on bins of {B} same-type trials (days −1 to 2; per mouse; median, IQR)", fontsize=7, y=0.995)
    F.OUT = C.FIG / "pub_simple"; F.OUT.mkdir(exist_ok=True)
    F.save(fig, f"figS5{TAG}_binned_eval_B{B}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=int, default=10)
    ap.add_argument("--n-jobs", type=int, default=40)
    ap.add_argument("--fig-only", action="store_true", help="redraw from the saved bins / mouse tables")
    ap.add_argument("--cv-files", nargs="+", default=CV_FILES)
    ap.add_argument("--insample", nargs="*", default=list(INSAMPLE))
    ap.add_argument("--order", nargs="+", default=["cv:lean-abs-stim", "cv:lean-abs-stim-hom", "cv:lean-abs-stim-nodis", "cv:simple",
                                                   "cv:simple-gen", "mouse:025S", "mouse:025G"])
    ap.add_argument("--key", nargs=2, default=["cv:lean-abs-stim", "cv:simple-gen"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    order, KEY, TAG = a.order, a.key, a.tag
    INSAMPLE = {t: INSAMPLE.get(t, t) for t in a.insample}
    MLAB.update({"025S": "final, per-mouse\n(in-sample)", "025G": "simple-gen, per-mouse\n(in-sample)"})
    if a.fig_only:
        S = pd.read_csv(C.TAB / f"034{TAG}_binned_B{a.bin}_mouse.csv")
        G = pd.read_parquet(C.TAB / f"034{TAG}_binned_B{a.bin}_bins.parquet")
        figure(S, G, a.bin, [m for m in order if m in set(S.model)])
        raise SystemExit
    D = F019.load()
    CV = pd.concat([pd.read_csv(C.TAB / f) for f in a.cv_files if (C.TAB / f).exists()], ignore_index=True)
    CV = CV[CV.variant.isin([m[3:] for m in order if m.startswith("cv:")])]
    CV = CV.drop_duplicates(["variant", "cohort_set", "mouse_id"], keep="last")
    jobs = [tuple(r) for r in CV[["variant", "cohort_set", "mouse_id"]].itertuples(index=False)]
    with mp.get_context("fork").Pool(a.n_jobs) as pool:
        L = pd.concat(pool.map(predict, jobs, chunksize=4), ignore_index=True)
    L["model"] = "cv:" + L.model
    parts = [L]
    for tag in INSAMPLE:
        f = C.TAB / f"028_{tag}_trial_latents.parquet"
        if f.exists():
            e = pd.read_parquet(f)[["mouse_id", "session_id", "day", "trial_type", "lick_flag", "reward_group", "p_lick"]]
            parts.append(e.assign(model=f"mouse:{tag}"))
    L = pd.concat(parts, ignore_index=True)
    L = L[L.day.isin(DAYS)]
    G = bins(L, a.bin)
    G.to_parquet(C.TAB / f"034{TAG}_binned_B{a.bin}_bins.parquet")
    S = scores(G)
    S.to_csv(C.TAB / f"034{TAG}_binned_B{a.bin}_mouse.csv", index=False)
    Q = summary(S)
    Q.to_csv(C.TAB / f"034{TAG}_binned_B{a.bin}_summary.csv", index=False)
    with pd.option_context("display.width", 250):
        print(Q[Q.trial_type == "all"].round(3).to_string(), flush=True)
        print(Q[Q.trial_type != "all"][["model", "trial_type", "phi_model_R+", "phi_model_R-", "r2_dev_R+", "r2_dev_R-"]].round(3)
              .to_string(), flush=True)
    figure(S, G, a.bin, [m for m in order if m in set(S.model)])
