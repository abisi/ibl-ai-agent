"""023 -- model v6 prediction check before the full run (model_v6_design.md; user 2026-10-23: "before running the full thing, let's
check predictions"). Cohort-level fits on all trials (no CV) of 7 rl7_model variants, two passes (pass 2 restarts from nested optima).
Predictions: causal predictive (rho_s posterior from earlier trials of the session) and simulation with rho_s drawn from the session's
smoothed posterior (learning on simulated licks). Compared with the data and with v5 best (021 latents) within session (session fifths)
and aligned to the session end.
Output: fits/023_fits.csv, tables/023_latents/<model>.parquet, figures/023_1_fit_vs_data, 023_2_session_end
Run (haas): OMP_NUM_THREADS=1 python 023_v6_prediction_check.py --n-jobs 14
"""
from __future__ import annotations

import argparse
import importlib
import multiprocessing as mp
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C
import rl7_model as M

F019 = importlib.import_module("019_fit_v4")
COHORTS = ["R+", "R-"]
MODELS = ["pop-lin-inst-noeng", "pop-thr-inst-noeng", "sess-lin-inst-noeng", "sess-thr-inst-noeng", "sess-thr-abs-noeng",
          "sess-lin-inst-eng", "sess-thr-inst-eng"]
V5 = "overlap-E3-eng-asym"
TAG = ""
N_RANDOM = 5
LAT_DIR = C.TAB / "023_latents"
COH = {"R+": "#00B400", "R-": "#C800C8"}
TC = {"auditory": "#1f78b4", "whisker": "#d95f02", "no_stim": "0.35"}
END_BINS = [0, 10, 20, 40, 80, 160, 100000]
NB, XMIN = 5, -4
D = None
plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.titlesize": 6.5, "legend.fontsize": 5.5})


def subset(d, cs):
    return d[d.reward_group == cs].reset_index(drop=True)


def job(args):
    name, cs, extra = args
    A = M.arrays(subset(D, cs))
    rng = np.random.default_rng(zlib.crc32(f"{name}{cs}".encode()))
    p, nll = M.fit(name, A, np.ones(len(A[0]), np.int64), rng, n_random=N_RANDOM if not extra else 0, extra_starts=extra)
    k = len(M.free_params(name))
    return dict(model=name, cohort_set=cs, k=k, n=len(A[0]), nll=nll, bic=2 * nll + k * np.log(len(A[0])), **{f"p_{q}": p[q] for q in M.P_ALL})


def start_from(sub_row, name, extra):
    p = {q: sub_row.get(f"p_{q}", M.FIXED[q]) for q in M.P_ALL}
    p.update(extra)
    return [float(np.clip(p[q], *M.BOUNDS[q])) for q in M.free_params(name)]


def fit_all(n_jobs, seed=None):
    with mp.get_context("fork").Pool(n_jobs) as pool:
        F = pd.DataFrame(pool.map(job, [(m, cs, ()) for m in MODELS for cs in COHORTS], chunksize=1))
        src = F if seed is None else pd.concat([F, seed[~seed.model.isin(MODELS)]], ignore_index=True)
        jobs = []
        for m in MODELS:
            for cs in COHORTS:
                st = [start_from(src[(src.model == s) & (src.cohort_set == cs)].iloc[0], m, ex) for s, ex in M.nested_of(m)
                      if s in set(src.model)]
                if m.split("-")[1] == "thr":  # threshold bias from the linear-bias optimum (theta below the thirst range)
                    lin = m.replace("-thr-", "-lin-")
                    if lin in set(src.model):
                        st.append(start_from(src[(src.model == lin) & (src.cohort_set == cs)].iloc[0], m, dict(theta=-0.5)))
                if st:
                    jobs.append((m, cs, tuple(st)))
        F2 = pd.DataFrame(pool.map(job, jobs, chunksize=1))
    F = F.set_index(["model", "cohort_set"]); F2 = F2.set_index(["model", "cohort_set"])
    better = F2.index[F2.nll < F.loc[F2.index, "nll"] - 1e-6]
    print(f"pass 2 improved {len(better)} of {len(F2)} fits", flush=True)
    F.loc[better] = F2.loc[better]
    return F.reset_index()


def latents(F, name):
    out = []
    for cs in COHORTS:
        d = subset(D, cs)
        A = M.arrays(d)
        r = F[(F.model == name) & (F.cohort_set == cs)].iloc[0]
        p = {q: r.get(f"p_{q}", M.FIXED[q]) for q in M.P_ALL}
        L = M.latents(name, p, A)
        sim, p_sim = M.simulate(name, p, A, np.random.default_rng(zlib.crc32(f"sim{name}{cs}".encode())), w_smooth=L["w_smooth"])
        o = d[["mouse_id", "session_id", "day", "trial_type", "lick_flag", "reward_group"]].copy()
        o["p_lick"] = L["p_lick"]; o["sim_lick"] = sim; o["p_sim"] = p_sim
        o["lrho_smooth"] = L["lrho_smooth"]; o["lrho_causal"] = L["lrho_causal"]; o["p_engaged"] = L["p_engaged"]
        for i, q in enumerate(M.LATENTS):
            o[q] = L["lat"][:, i]
        out.append(o)
    return pd.concat(out, ignore_index=True)


def add_x(L):
    g = L.groupby("session_id")
    pos = g.cumcount(); n = g.session_id.transform("size")
    return L.assign(x=L.day + np.floor(NB * pos / n) / NB, toend=g.cumcount(ascending=False))


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(C.FIG / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_fit_vs_data(lats, col):
    names = list(lats)
    fig, axs = plt.subplots(len(names), 2, figsize=(7.2, 1.55 * len(names)), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.55, wspace=0.1))
    for i, m in enumerate(names):
        L = lats[m]; L = L[L.x >= XMIN]
        c_ = col if col in L else "p_lick"
        g = L.groupby(["reward_group", "trial_type", "x"])
        T = g[["lick_flag", c_]].mean().assign(n_mice=g.mouse_id.nunique())
        T = T[T.n_mice >= 10].reset_index()
        for j, cs in enumerate(COHORTS):
            ax = axs[i, j]
            for t in TC:
                e = T[(T.reward_group == cs) & (T.trial_type == t)]
                for _, e2 in e.groupby(np.floor(e.x)):
                    ax.plot(e2.x, e2.lick_flag, color=TC[t], lw=1.6, alpha=0.4)
                    ax.plot(e2.x, e2[c_], color=TC[t], lw=0.8)
            ax.axvline(0, color="0.7", lw=0.5, ls=":")
            ax.set_title(f"{cs.replace('-', '−')} — {m}")
        axs[i, 0].set_ylabel("lick rate")
    for ax in axs[-1]:
        ax.set_xlabel("training day (session fifths; 0 = first whisker day)")
    lab = "causal predictive" if col == "p_lick" else "simulation (ρ_s from session posterior)"
    fig.suptitle(f"Data (thick, light) vs model {lab} (thin): blue auditory, orange whisker, grey no stimulus", y=1.0)
    return fig


def fig_session_end(lats):
    names = list(lats)
    cols = plt.cm.tab10(np.arange(len(names)))
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True, sharey="col", gridspec_kw=dict(hspace=0.45, wspace=0.3))
    mid = np.array([5, 15, 30, 60, 120, 200])
    for i, cs in enumerate(COHORTS):
        for j, t in enumerate(TC):
            ax = axs[i, j]
            for k, m in enumerate(names):
                L = lats[m]; L = L[(L.reward_group == cs) & (L.trial_type == t) & (L.day >= XMIN)]
                b = pd.cut(L.toend, END_BINS, right=False)
                e = L.groupby(b, observed=True)[["lick_flag", "p_lick"] + (["sim_lick"] if "sim_lick" in L else [])].mean()
                if k == 0:
                    ax.plot(mid[:len(e)], e.lick_flag, color="k", lw=2, alpha=0.35, label="data")
                ax.plot(mid[:len(e)], e.p_lick, color=cols[k], lw=0.9, label=f"{m} (pred)")
                if "sim_lick" in e:
                    ax.plot(mid[:len(e)], e.sim_lick, color=cols[k], lw=0.7, ls="--")
            ax.set_xscale("log"); ax.invert_xaxis()
            ax.set_title(f"{cs.replace('-', '−')} {t.replace('_', ' ')}")
            if i == 1:
                ax.set_xlabel("trials before session end")
        axs[i, 0].set_ylabel("lick rate")
    axs[0, 2].legend(frameon=False, fontsize=4.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Session-end collapse (days ≥ −4): data vs causal predictive (solid) and posterior simulation (dashed)", y=1.0)
    return fig


def figures(F, sel_models):
    lats = {f"v5 {V5}": add_x(pd.read_parquet(C.TAB / "021_latents" / f"{V5}.parquet"))}
    for m in MODELS:
        lats[m] = add_x(pd.read_parquet(LAT_DIR / f"{m}.parquet"))
    sel = {k: lats[k] for k in [f"v5 {V5}"] + sel_models}
    save(fig_fit_vs_data(sel, "p_lick"), f"023{TAG}_1_fit_vs_data")
    save(fig_fit_vs_data({k: v for k, v in sel.items() if not k.startswith("v5")}, "sim_lick"), f"023{TAG}_1b_fit_vs_data_sim")
    save(fig_session_end(lats), f"023{TAG}_2_session_end")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=14)
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--models", nargs="+", default=None, help="v6b check: model list (default: the 7 v6 models)")
    ap.add_argument("--tag", default="", help="output tag, e.g. b -> fits/023b_fits.csv, tables/023b_latents, figures/023b_*")
    ap.add_argument("--seed", default=None, help="earlier fits file whose models seed the nested starts (e.g. 023_fits.csv)")
    ap.add_argument("--n-random", type=int, default=5)
    ap.add_argument("--show", nargs="+", default=["pop-lin-inst-noeng", "sess-lin-inst-noeng", "sess-thr-inst-noeng", "sess-thr-inst-eng"])
    a = ap.parse_args()
    TAG = a.tag
    N_RANDOM = a.n_random
    LAT_DIR = C.TAB / f"023{TAG}_latents"
    MODELS = a.models or MODELS
    D = F019.load()
    fits = C.FITS / f"023{TAG}_fits.csv"
    if not a.figures_only:
        F = fit_all(a.n_jobs, seed=pd.read_csv(C.FITS / a.seed) if a.seed else None)
        F.to_csv(fits, index=False)
        print(F[["model", "cohort_set", "k", "nll", "bic"]].round(1).to_string(), flush=True)
        LAT_DIR.mkdir(exist_ok=True)
        for m in MODELS:
            latents(F, m).to_parquet(LAT_DIR / f"{m}.parquet")
    figures(pd.read_csv(fits), [m for m in a.show if m in MODELS])
