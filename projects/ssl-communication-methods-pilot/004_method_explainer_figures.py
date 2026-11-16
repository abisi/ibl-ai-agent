"""Explainer figures for the communication methods pilot (user request 2026-11-06: "make a lot of figures for each method
that summarise the process and help me understand the different steps and interpretation").

Every step of every method is rerun, with its intermediate quantities, on one worked example: session AB127 (R+),
whisker-hit trials, MO/FRP <-> SSp-w, draw 0 of 002. The units and trials are the same as in 002, using the same
seeds. Time-resolved and group panels read the 002/003 outputs.
Figures (combined_results_ks4/ssl-communication-methods-pilot/figures/explainer/):
  00_pipeline            the whole analysis, step by step
  01_data_residuals      spikes -> 10-ms bins -> condition-mean residuals; why residuals
  02_samples_controls    windows, samples, unit/trial matching, firing rates, trial counts
  03_rrr_steps           reduced-rank regression on one window: ridge, rank, prediction, subspace, null
  04_rrr_results         RRR in time, nulls, ceiling, movement, cohorts
  05_cca_steps           PCA -> CCA on one window: PCs, train vs held-out, folds and signs, weights, null
  06_cca_results         CCA in time, nulls, movement, cohorts
  07_granger_steps       ensemble Granger on one window: PCs, lags, restricted vs full model, order, null, reversal
  08_granger_results     Granger in time, time-reversed directionality, cohorts
  09_methods_compared    simulation sensitivity, agreement, power, what each method measures
Area colours: ephys_utilities allen_utils get_custom_area_groups_colors(). MO/FRP = 'Motor and frontal areas',
SSp-w = 'Somatosensory-whisker', striatum and pallidum = 'Striatum'.
Run (haas): OMP_NUM_THREADS=4 ~/code/unit_spikes_analysis/.venv/bin/python \
    projects/ssl-communication-methods-pilot/004_method_explainer_figures.py [--replot]
"""

import argparse
import json
import pickle
import sys
import textwrap
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402

OUT = cl.OUT
FIG = OUT / "figures" / "explainer"
CACHE = OUT / "explainer_cache.pkl"
C = cl.CONFIG
SID, COHORT = "AB127_20240821_103757", "R+"
COND = "W_hit"
A, B = "Motor and frontal areas", "SSp-w"
POST, PRE = 0.1, -0.2  # example window centres (s from trial start)
N_NULL = 200  # shuffles for the example null distributions
COLORS = {"R+": "#00B400", "R-": "#C800C8"}
AREA_COL = {"Motor and frontal areas": "#1f9d5a", "SSp-w": "#f7b519", "Striatum and pallidum": "#54c2ff"}
SHORT = {"Motor and frontal areas": "MO/FRP", "SSp-w": "SSp-w", "Striatum and pallidum": "STR"}
COND_LABEL = {"W_hit": "Whisker hit", "W_miss": "Whisker miss", "A_hit": "Auditory hit", "NS_cr": "No-stim CR"}
COND_COL = {"W_hit": "#d4880f", "W_miss": "#9a9a9a", "A_hit": "#2c6fdb", "NS_cr": "#333333"}
NULL_COL, CEIL_COL, MVT_COL = "#8c8c8c", "#4d4d4d", "#c0392b"

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 200})


# ---------------------------------------------------------------- worked example: compute

def select_example():
    """Units, trials and residual tensors of draw 0 of 002 for SID (same seeds and order as 002_run_pilot)."""
    units, trials = cl.load_session(SID)
    starts = trials.start_time.to_numpy()
    spikes = [np.sort(np.asarray(s, float)) for s in units.spike_times]
    fine_all = cl.fine_tensor(spikes, starts)
    dur = cl.bin_edges()[-1] - cl.bin_edges()[0]
    rate = fine_all.sum((0, 1)) / (len(starts) * dur)
    keep = rate >= C["min_rate_hz"]
    units, fine_all, rate = units[keep].reset_index(drop=True), fine_all[:, :, keep], rate[keep]
    spikes = [s for s, k in zip(spikes, keep) if k]
    by_area = {a: np.flatnonzero(units.area_group.to_numpy() == a) for a in C["areas"]}
    rng = np.random.default_rng(zlib.crc32(SID.encode()) + 0)
    pick = {a: rng.permutation(v) for a, v in by_area.items()}
    src = {a: p[:C["n_units_source"]] for a, p in pick.items()}
    within = {a: p[C["n_units_source"]:C["n_units_source"] + C["n_units_target"]] for a, p in pick.items()}
    tr = rng.choice(np.flatnonzero(trials.condition.to_numpy() == COND), C["n_trials"], replace=False)
    return dict(units=units, trials=trials, starts=starts, spikes=spikes, fine_all=fine_all, rate=rate,
                by_area=by_area, src=src, within=within, tr=tr)


def window_idx(center):
    centers, widx = cl.window_grid()
    return widx[int(np.argmin(np.abs(centers - center)))]


def rrr_detail(XT, YT, rng):
    """cl.rrr_cv with its intermediate quantities: per-fold performance per alpha and rank, the chosen alpha and
    rank, held-out predictions at that rank, and the performance of regressing on the top-k source PCs."""
    n, k, max_rank = len(XT), C["cv_folds"], min(C["max_rank"], XT.shape[2], YT.shape[2])
    folds = cl._folds(n, k, rng)
    X_all = cl._rows(XT, np.arange(n))
    scale = np.trace(X_all.T @ X_all) / X_all.shape[1]
    perf = {}
    for alpha in C["ridge_alphas"]:
        p = np.zeros((k, max_rank + 1))
        for f, test in enumerate(folds):
            train = np.setdiff1d(np.arange(n), test)
            Xtr, Ytr, Xte, Yte = cl._rows(XT, train), cl._rows(YT, train), cl._rows(XT, test), cl._rows(YT, test)
            mx, my = Xtr.mean(0), Ytr.mean(0)
            Xtr, Ytr, Xte, Yte = Xtr - mx, Ytr - my, Xte - mx, Yte - my
            Bm = cl._ridge(Xtr, Ytr, alpha * scale)
            _, _, Vt = np.linalg.svd(Xtr @ Bm, full_matrices=False)
            sst = (Yte ** 2).sum()
            for r in range(1, max_rank + 1):
                V = Vt[:r].T
                p[f, r - 1] = 1 - ((Yte - Xte @ Bm @ V @ V.T) ** 2).sum() / sst
            p[f, max_rank] = 1 - ((Yte - Xte @ Bm) ** 2).sum() / sst
        perf[alpha] = p
    alpha = max(perf, key=lambda a: perf[a][:, -1].mean())
    m, s = perf[alpha][:, :-1].mean(0), perf[alpha][:, :-1].std(0) / np.sqrt(k)
    peak = int(np.argmax(m))
    opt = int(np.argmax(m >= m[peak] - s[peak])) + 1
    # second pass at the chosen alpha: held-out predictions at the optimal rank, and top-k source PC regression
    Y_true, Y_pred, test_trial = [], [], []
    pc_perf = np.zeros((k, max_rank))
    for f, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        Xtr, Ytr, Xte, Yte = cl._rows(XT, train), cl._rows(YT, train), cl._rows(XT, test), cl._rows(YT, test)
        mx, my = Xtr.mean(0), Ytr.mean(0)
        Xtr, Ytr, Xte, Yte = Xtr - mx, Ytr - my, Xte - mx, Yte - my
        Bm = cl._ridge(Xtr, Ytr, alpha * scale)
        _, _, Vt = np.linalg.svd(Xtr @ Bm, full_matrices=False)
        V = Vt[:opt].T
        Y_true.append(Yte)
        Y_pred.append(Xte @ Bm @ V @ V.T)
        test_trial.append(np.repeat(test, XT.shape[1]))
        _, _, Vx = np.linalg.svd(Xtr, full_matrices=False)
        sst = (Yte ** 2).sum()
        for r in range(1, max_rank + 1):
            P = Vx[:r].T
            Bp = cl._ridge(Xtr @ P, Ytr, alpha * scale)
            pc_perf[f, r - 1] = 1 - ((Yte - Xte @ P @ Bp) ** 2).sum() / sst
    return dict(perf=perf, alpha=alpha, opt=opt, mean=m, sem=s, perf_full=perf[alpha][:, -1],
                Y_true=np.vstack(Y_true), Y_pred=np.vstack(Y_pred), test_trial=np.concatenate(test_trial),
                pc_perf=pc_perf, scale=scale)


def cca_detail(XT, YT, rng, kpc=None):
    """cl.pca_cca_cv with its intermediates: training canonical correlations (all), held-out correlations per fold and
    dimension, the projections of fold 0, the pooled-over-folds correlation (the sign problem) and neuron weights."""
    kpc = kpc or C["cca_pcs"]
    n, k = len(XT), C["cv_folds"]
    kx, ky = min(kpc, XT.shape[2]), min(kpc, YT.shape[2])
    train_cc, test_cc, pooled_a, pooled_b, fold0 = [], [], [], [], None
    for f, test in enumerate(cl._folds(n, k, rng)):
        train = np.setdiff1d(np.arange(n), test)
        Xtr, Ytr, Xte, Yte = cl._rows(XT, train), cl._rows(YT, train), cl._rows(XT, test), cl._rows(YT, test)
        mx, Px = cl._pca_fit(Xtr, kx)
        my, Py = cl._pca_fit(Ytr, ky)
        Aa, Bb = (Xtr - mx) @ Px, (Ytr - my) @ Py
        Cxx, Cyy, Cxy = Aa.T @ Aa / len(Aa), Bb.T @ Bb / len(Bb), Aa.T @ Bb / len(Aa)
        Wx, Wy = cl._inv_sqrt(Cxx, C["cca_reg"]), cl._inv_sqrt(Cyy, C["cca_reg"])
        U, S, Vt = np.linalg.svd(Wx @ Cxy @ Wy)
        d = min(kx, ky)
        a, b = Wx @ U[:, :d], Wy @ Vt[:d].T
        pa, pb = ((Xte - mx) @ Px) @ a, ((Yte - my) @ Py) @ b
        train_cc.append(S[:d])
        test_cc.append([np.corrcoef(pa[:, j], pb[:, j])[0, 1] for j in range(d)])
        pooled_a.append(pa[:, 0])
        pooled_b.append(pb[:, 0])
        if f == 0:
            fold0 = dict(train_a=Aa @ a[:, 0], train_b=Bb @ b[:, 0], test_a=pa[:, 0], test_b=pb[:, 0])
    # weights on neurons from a fit on all trials (illustration only)
    Xall, Yall = cl._rows(XT, np.arange(n)), cl._rows(YT, np.arange(n))
    mx, Px = cl._pca_fit(Xall, kx)
    my, Py = cl._pca_fit(Yall, ky)
    Aa, Bb = (Xall - mx) @ Px, (Yall - my) @ Py
    Wx = cl._inv_sqrt(Aa.T @ Aa / len(Aa), C["cca_reg"])
    Wy = cl._inv_sqrt(Bb.T @ Bb / len(Bb), C["cca_reg"])
    U, S, Vt = np.linalg.svd(Wx @ (Aa.T @ Bb / len(Aa)) @ Wy)
    w_a, w_b = Px @ (Wx @ U[:, 0]), Py @ (Wy @ Vt[0])
    sv_x = np.linalg.svd(Xall - Xall.mean(0), compute_uv=False) ** 2
    sv_y = np.linalg.svd(Yall - Yall.mean(0), compute_uv=False) ** 2
    return dict(train_cc=np.array(train_cc), test_cc=np.array(test_cc), fold0=fold0,
                pooled=float(np.corrcoef(np.concatenate(pooled_a), np.concatenate(pooled_b))[0, 1]),
                w_a=w_a / np.abs(w_a).max(), w_b=w_b / np.abs(w_b).max(),
                var_x=sv_x / sv_x.sum(), var_y=sv_y / sv_y.sum())


def gc_parts(src, tgt, order, t_index):
    """Restricted (target past) and full (target + source past) VAR residual covariances, as in cl.granger_ensemble."""
    def design(src_list):
        rows_y, rows_x = [], []
        for t in t_index:
            rows_y.append(tgt[:, t, :])
            rows_x.append(np.concatenate([s[:, t - lag, :] for s in src_list for lag in range(1, order + 1)], 1))
        return np.vstack(rows_x), np.vstack(rows_y)

    out = {}
    for name, lst in [("restricted", [tgt]), ("full", [tgt, src])]:
        Xd, Yd = design(lst)
        Xd = np.c_[np.ones(len(Xd)), Xd]
        Bm = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
        E = Yd - Xd @ Bm
        out[name] = dict(E=E, cov=np.cov(E.T), n=len(Yd), p=Xd.shape[1] * Yd.shape[1])
    out["gc"] = float(np.log(np.linalg.det(out["restricted"]["cov"]) / np.linalg.det(out["full"]["cov"])))
    return out


def granger_t_index(center):
    idx = window_idx(center)
    gbins = int(round(C["granger_window_s"] / C["bin_s"]))
    lo = max(C["granger_order"], int(idx[0] + len(idx) / 2 - gbins / 2))
    return np.arange(lo, lo + gbins)


def compute():
    ex = select_example()
    fine, tr, src, within = ex["fine_all"], ex["tr"], ex["src"], ex["within"]
    R = cl.residualise(fine[tr], np.zeros(len(tr)))
    rng = np.random.default_rng(1)
    res = {"meta": {"session_id": SID, "cohort": COHORT, "condition": COND, "area_a": A, "area_b": B,
                    "post": POST, "pre": PRE, "n_null": N_NULL, "config": C}}
    # data panels
    edges = cl.bin_edges()
    t_raster = int(tr[0])
    res["raster"] = {a: [ex["spikes"][u] - ex["starts"][t_raster] for u in src[a]] for a in (A, B)}
    cond_all = ex["trials"].condition.to_numpy()
    res["psth"] = {(a, c): fine[cond_all == c][:, :, src[a]].mean((0, 2)) / C["bin_s"]
                   for a in (A, B) for c in C["conditions"]}
    resp = fine[tr][:, (edges[:-1] >= 0) & (edges[:-1] < 0.1)][:, :, src[B]].mean((0, 1)) \
        - fine[tr][:, (edges[:-1] < 0)][:, :, src[B]].mean((0, 1))
    u_ex = int(src[B][int(np.argmax(resp))])
    res["unit_example"] = dict(raw=fine[tr][:, :, u_ex], resid=R[:, :, u_ex])
    win = (edges[:-1] >= 0) & (edges[:-1] < 0.3)
    pop = {a: fine[:, win][:, :, src[a]].sum((1, 2)) for a in (A, B)}
    res["pop_counts"] = dict(raw=pop, cond=cond_all,
                             resid={a: cl.residualise(pop[a][:, None], cond_all)[:, 0] for a in (A, B)})
    idx_post = window_idx(POST)
    XA, XB = R[:, idx_post][:, :, src[A]], R[:, idx_post][:, :, src[B]]
    res["noise_corr"] = np.corrcoef(np.c_[cl._rows(XA, np.arange(len(XA))), cl._rows(XB, np.arange(len(XB)))].T)
    res["X_rows"], res["Y_rows"] = cl._rows(XA, np.arange(len(XA))), cl._rows(XB, np.arange(len(XB)))

    # RRR (A -> B), within-area ceiling (A -> other A units), shuffle
    WA = R[:, idx_post][:, :, within[A]]
    res["rrr"] = {"post": rrr_detail(XA, XB, rng), "within": rrr_detail(XA, WA, rng),
                  "shuffle": rrr_detail(XA[rng.permutation(len(XA))], XB, rng),
                  "pre": rrr_detail(R[:, window_idx(PRE)][:, :, src[A]], R[:, window_idx(PRE)][:, :, src[B]], rng)}
    res["rrr"]["null"] = np.array([cl.rrr_cv(XA[rng.permutation(len(XA))], XB, rng)["perf_rrr"]
                                   for _ in range(N_NULL)])

    # PCA -> CCA
    res["cca"] = {"post": cca_detail(XA, XB, rng), "shuffle": cca_detail(XA[rng.permutation(len(XA))], XB, rng)}
    res["cca"]["null"] = np.array([cl.pca_cca_cv(XA[rng.permutation(len(XA))], XB, rng)["cc1"]
                                   for _ in range(N_NULL)])
    ks = [1, 2, 3, 5, 8, 10, 15, 20]
    res["cca"]["by_k"] = {k: (cca_detail(XA, XB, rng, k)["test_cc"][:, 0].mean(),
                              np.mean([cca_detail(XA[rng.permutation(len(XA))], XB, rng, k)["test_cc"][:, 0].mean()
                                       for _ in range(10)])) for k in ks}

    # Granger
    za = cl.population_pcs(fine[tr][:, :, src[A]], np.zeros(len(tr)), C["granger_pcs"])
    zb = cl.population_pcs(fine[tr][:, :, src[B]], np.zeros(len(tr)), C["granger_pcs"])
    ti = granger_t_index(POST)
    order = C["granger_order"]
    g = {"za": za, "zb": zb, "t_index": ti, "ab": gc_parts(za, zb, order, ti), "ba": gc_parts(zb, za, order, ti)}
    nb = za.shape[1]
    t_rev = np.sort(nb - 1 - ti)
    t_rev = t_rev[t_rev >= order]
    g["rev"] = cl.granger_ensemble(za[:, ::-1], zb[:, ::-1], order, t_rev)
    g["null"] = np.array([cl.granger_ensemble(za[rng.permutation(len(za))], zb, order, ti) for _ in range(N_NULL)])
    g["orders"] = {}
    for o in range(1, 9):
        t_o = ti[ti >= o]
        pa, pb = gc_parts(za, zb, o, t_o), gc_parts(zb, za, o, t_o)
        bic = {k: np.log(np.linalg.det(p["full"]["cov"])) + p["full"]["p"] * np.log(p["full"]["n"]) / p["full"]["n"]
               for k, p in (("ab", pa), ("ba", pb))}
        g["orders"][o] = dict(ab=pa["gc"], ba=pb["gc"], bic_ab=bic["ab"], bic_ba=bic["ba"])
    lags = np.arange(-10, 11)
    xc = []
    for lag in lags:
        t_ok = ti[(ti + lag >= 0) & (ti + lag < nb)]
        xc.append(np.corrcoef(za[:, t_ok, 0].ravel(), zb[:, t_ok + lag, 0].ravel())[0, 1])
    g["xcorr"] = (lags * C["bin_s"] * 1000, np.array(xc))
    for k in ("ab", "ba"):  # keep only what is plotted
        for m in ("restricted", "full"):
            g[k][m] = {kk: vv for kk, vv in g[k][m].items() if kk != "E"} | {"E0": g[k][m]["E"][:, 0]}
    res["granger"] = g
    res["units_example"] = dict(n_by_area={a: len(v) for a, v in ex["by_area"].items()}, unit=u_ex)
    pickle.dump(res, open(CACHE, "wb"))
    return res


# ---------------------------------------------------------------- shared data from 002 / 003

def load_group():
    sl = pd.read_parquet(OUT / "session_level.parquet")
    ses = pd.read_parquet(OUT / "sessions" / f"{SID}.parquet")
    metas = [json.load(open(f)) for f in sorted((OUT / "sessions").glob("*_meta.json")) if "_quick" not in f.name]
    sim = pd.read_csv(OUT / "simulation" / "simulation.csv")
    return sl, ses, metas, sim


# ---------------------------------------------------------------- plotting helpers

def label(ax, s):
    ax.text(-0.18, 1.08, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def note(ax, title, text, width=62):
    ax.axis("off")
    body = "\n".join(textwrap.fill(p, width) for p in text.strip().split("\n"))
    ax.text(0, 1, title, fontweight="bold", fontsize=7.5, va="top", transform=ax.transAxes)
    ax.text(0, 0.93, body, fontsize=6.4, va="top", transform=ax.transAxes, linespacing=1.35)


def stim_line(ax):
    ax.axvline(0, color="k", lw=0.6, ls=":")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def null_hist(ax, null, val, xlabel):
    ax.hist(null, bins=30, color=NULL_COL, alpha=0.7, label=f"trial-shuffle null (n={len(null)})")
    ax.axvline(val, color="k", lw=1.2, label="true value")
    z = (val - null.mean()) / null.std()
    p = (np.sum(null >= val) + 1) / (len(null) + 1)
    ax.set(xlabel=xlabel, ylabel="shuffles")
    ax.set_title(f"Is it above chance?  z = {z:.1f}, p = {p:.3f}")
    ax.legend(frameon=False, loc="upper right")


def cohort_ts(ax, d, ycol="excess"):
    """mean +- s.e.m. over mice per cohort of a [mouse, window] long table."""
    for c in ("R+", "R-"):
        w = d[d.cohort == c].groupby(["mouse_id", "window_center"])[ycol].mean().unstack()
        if w.empty:
            continue
        m, s = w.mean(), w.std() / np.sqrt(w.notna().sum())
        ax.plot(m.index, m, color=COLORS[c], lw=1.2, label=f"{c} (n={len(w)})")
        ax.fill_between(m.index, m - s, m + s, color=COLORS[c], alpha=0.2, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    stim_line(ax)


def cohort_dots(ax, d, groups, ycol="excess"):
    """post-window per-mouse values per cohort for each group (x positions), with MW and Welch p."""
    for i, (gname, gd) in enumerate(groups(d)):
        per = gd.groupby(["mouse_id", "cohort"])[ycol].mean().reset_index()
        vals = {}
        for j, c in enumerate(("R+", "R-")):
            v = per[per.cohort == c][ycol].to_numpy()
            vals[c] = v
            x = i + (j - 0.5) * 0.35
            ax.scatter(x + np.random.default_rng(i).uniform(-0.06, 0.06, len(v)), v, s=7, color=COLORS[c],
                       alpha=0.75, lw=0)
            if len(v):
                ax.errorbar(x, v.mean(), v.std() / np.sqrt(len(v)), fmt="_", color="k", ms=8, lw=1)
        if len(vals["R+"]) > 2 and len(vals["R-"]) > 2:
            pm = stats.mannwhitneyu(vals["R+"], vals["R-"]).pvalue
            pw = stats.ttest_ind(vals["R+"], vals["R-"], equal_var=False).pvalue
            ax.text(i, 1.0, f"MW {pm:.2f}\nWelch {pw:.2f}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=5.2)
    names = [g for g, _ in groups(d)]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.axhline(0, color="k", lw=0.5)


def post_window(d):
    return d[(d.window_center >= 0) & (d.window_center <= 0.3)]


def directed_groups(method):
    def f(d):
        out = []
        for pair in sorted(d.pair.unique()):
            a, b = pair.split(" | ")
            for s, t in ((a, b), (b, a)):
                out.append((f"{SHORT[s]}→{SHORT[t]}", d[(d.pair == pair) & (d.source == s)]))
        return out
    return f


def pair_groups(d):
    return [(" ↔ ".join(SHORT[x] for x in p.split(" | ")), d[d.pair == p]) for p in sorted(d.pair.unique())]


# ---------------------------------------------------------------- figures

def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f2f2f2", ec="#555555"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5", fc=fc, ec=ec, lw=0.8))
        ax.text(x + 1, y + h - 1.2, title, fontweight="bold", fontsize=7, va="top")
        ax.text(x + 1, y + h - 5.2, textwrap.fill(body, int(w * 1.55)), fontsize=5.8, va="top", linespacing=1.3)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))

    box(1, 84, 30, 14, "1. Session",
        "Whisker day 0, learner. Good units (project QC), rate >= 1 Hz, grouped into MO/FRP, SSp-w, STR. "
        "Active trials, no early lick, perf != 6.")
    box(35, 84, 30, 14, "2. Trials and units",
        "4 conditions: whisker hit/miss, auditory hit, no-stim CR. 30 random trials per condition and 20 random "
        "units per area; done twice (2 draws).")
    box(69, 84, 30, 14, "3. Binning",
        "Spike counts in 10-ms bins from -0.5 to 1.1 s around trial start: tensor [trial x bin x unit] per area.")
    box(69, 63, 30, 16, "4. Residuals",
        "Subtract each unit's mean across trials of the same condition, bin by bin. The PSTH (stimulus/lick-locked "
        "part) is gone; what remains are trial-to-trial fluctuations.", fc="#fff3d6")
    box(35, 63, 30, 16, "5. Sliding windows",
        "100-ms windows every 20 ms (-0.4 to 1.0 s). Samples = the 30 trials x 10 bins = 300 rows. Cross-validation "
        "folds split by trial, never by bin.")
    box(1, 63, 30, 16, "6. Trial-shuffle null",
        "Same method after permuting the source area's trials: each area keeps its own statistics, the trial-by-trial "
        "pairing is broken.", fc="#ececec")
    box(1, 34, 30, 22, "RRR (Semedo 2019)",
        "Predict target residuals from source residuals with ridge regression whose rank is limited. Value: "
        "cross-validated variance explained (1 - NSE) at the best rank. Directed (A→B and B→A) but zero-lag.",
        fc="#e3f1e3", ec="#1f9d5a")
    box(35, 34, 30, 22, "PCA → CCA",
        "Reduce each area to its top 10 PCs, find the pair of axes (one per area) whose projections are most "
        "correlated. Value: held-out correlation of the first pair. Symmetric, zero-lag.",
        fc="#e3eef9", ec="#2c6fdb")
    box(69, 34, 30, 22, "Ensemble Granger",
        "Top 3 PCs per area. Does the source's past (3 bins = 30 ms) improve prediction of the target beyond the "
        "target's own past? Value: log variance ratio. Directed and lagged; time-reversal check.",
        fc="#f8e6f3", ec="#a03b8b")
    box(1, 4, 47, 22, "7. Session value",
        "excess = true value − mean of the trial-shuffle null, averaged over the 2 draws. Controls: within-area RRR "
        "(ceiling), jaw and whisker movement regressed out, firing rates, trial counts, synthetic data with known "
        "coupling.")
    box(52, 4, 47, 22, "8. Cohort test",
        "One session per mouse, 8 R+ vs 8 R-. Per window: Mann-Whitney and Welch. Over time: cluster permutation "
        "(1,000 cohort-label permutations). Per pair, direction, condition and method → about 72 time courses.",
        fc="#fde9e9")
    for (x0, y0, x1, y1) in [(31.5, 91, 34.5, 91), (65.5, 91, 68.5, 91), (84, 83.5, 84, 79.5), (68.5, 71, 65.5, 71),
                             (34.5, 71, 31.5, 71), (16, 62.5, 16, 56.5), (50, 62.5, 50, 56.5), (66, 64, 84, 56.5),
                             (16, 33.5, 24, 26.5), (50, 33.5, 30, 26.5), (84, 33.5, 40, 26.5), (48.5, 15, 51.5, 15)]:
        arrow(x0, y0, x1, y1)
    ax.set_title("Inter-areal communication pilot: from spikes to a cohort test", fontsize=9, fontweight="bold")
    save(fig, "00_pipeline")


def fig_data(res, metas):
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(4, 4, hspace=0.75, wspace=0.55)
    edges = cl.bin_edges()
    tc = edges[:-1] + C["bin_s"] / 2
    # A: raster
    ax = fig.add_subplot(gs[0, :2])
    row = 0
    for a in (A, B):
        for st in res["raster"][a]:
            s = st[(st >= edges[0]) & (st <= edges[-1])]
            ax.vlines(s, row, row + 0.8, color=AREA_COL[a], lw=0.6)
            row += 1
    ax.set(xlim=(edges[0], edges[-1]), ylim=(0, row), xlabel="time from trial start (s)", ylabel="unit")
    ax.text(1.02, 0.25, SHORT[A], color=AREA_COL[A], transform=ax.transAxes, fontweight="bold")
    ax.text(1.02, 0.75, SHORT[B], color=AREA_COL[B], transform=ax.transAxes, fontweight="bold")
    stim_line(ax)
    ax.set_title("One whisker-hit trial: 20 + 20 selected units")
    label(ax, "a")
    # B: PSTHs
    for j, a in enumerate((A, B)):
        ax = fig.add_subplot(gs[0, 2 + j])
        for c in C["conditions"]:
            ax.plot(tc, res["psth"][(a, c)], color=COND_COL[c], lw=0.9, label=COND_LABEL[c])
        stim_line(ax)
        ax.set(xlabel="time (s)", ylabel="mean rate (Hz)", title=f"{SHORT[a]} PSTH by condition")
        if j == 1:
            ax.legend(frameon=False, loc="upper right")
        if j == 0:
            label(ax, "b")
    # C-E: one unit raw, mean, residual
    ue = res["unit_example"]
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(ue["raw"], aspect="auto", cmap="Greys", extent=(edges[0], edges[-1], len(ue["raw"]), 0),
              interpolation="nearest", vmin=0, vmax=max(1, ue["raw"].max()))
    stim_line(ax)
    ax.set(xlabel="time (s)", ylabel="trial", title="One SSp-w unit: counts")
    label(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(tc, ue["raw"].mean(0) / C["bin_s"], color=AREA_COL[B])
    stim_line(ax)
    ax.set(xlabel="time (s)", ylabel="Hz", title="Condition mean (the PSTH)")
    ax = fig.add_subplot(gs[1, 2])
    lim = np.abs(ue["resid"]).max()
    ax.imshow(ue["resid"], aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
              extent=(edges[0], edges[-1], len(ue["resid"]), 0), interpolation="nearest")
    stim_line(ax)
    ax.set(xlabel="time (s)", ylabel="trial", title="Residual = counts − mean")
    ax = fig.add_subplot(gs[1, 3])
    note(ax, "What is analysed",
         "Every bin of every trial minus that unit's mean for the condition. A spike where the unit "
         "usually fires adds little; a spike where it usually doesn't adds a lot. Communication is "
         "then: do residuals in area A co-vary with residuals in area B on the same trials? "
         "Shared stimulus responses can no longer create it.", width=34)
    # F: why residuals (scatter)
    pc = res["pop_counts"]
    for j, kind in enumerate(("raw", "resid")):
        ax = fig.add_subplot(gs[2, j])
        for c in C["conditions"]:
            m = pc["cond"] == c
            ax.scatter(pc[kind][A][m], pc[kind][B][m], s=4, color=COND_COL[c], alpha=0.6, lw=0,
                       label=COND_LABEL[c])
        r = np.corrcoef(pc[kind][A], pc[kind][B])[0, 1]
        ax.set(xlabel=f"{SHORT[A]} spikes 0–0.3 s", ylabel=f"{SHORT[B]} spikes",
               title=f"{'Raw counts' if kind == 'raw' else 'Residuals'}: r = {r:.2f}")
        if j == 0:
            label(ax, "d")
            ax.legend(frameon=False, markerscale=2, loc="upper left")
    # G: noise correlation matrix
    ax = fig.add_subplot(gs[2, 2])
    nc = res["noise_corr"].copy()
    np.fill_diagonal(nc, np.nan)
    lim = np.nanpercentile(np.abs(nc), 99)
    ax.imshow(nc, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.axhline(19.5, color="k", lw=0.6)
    ax.axvline(19.5, color="k", lw=0.6)
    ax.set_xticks([10, 30])
    ax.set_xticklabels([SHORT[A], SHORT[B]])
    ax.set_yticks([10, 30])
    ax.set_yticklabels([SHORT[A], SHORT[B]])
    ax.set_title("Residual correlations, 0.05–0.15 s")
    label(ax, "e")
    ax = fig.add_subplot(gs[2, 3])
    n = 20
    iu = np.triu_indices(n, 1)
    within_a = res["noise_corr"][:n, :n][iu]
    within_b = res["noise_corr"][n:, n:][iu]
    between = res["noise_corr"][:n, n:].ravel()
    bins = np.linspace(-0.15, 0.25, 40)
    for v, lab, col in [(within_a, f"within {SHORT[A]}", AREA_COL[A]), (within_b, f"within {SHORT[B]}", AREA_COL[B]),
                        (between, "between", "k")]:
        ax.hist(v, bins=bins, histtype="step", density=True, color=col, label=f"{lab} ({np.mean(v):.3f})")
    ax.set(xlabel="pairwise r (10-ms bins)", ylabel="density", title="Pairs are weakly correlated")
    ax.legend(frameon=False, loc="upper right")
    # H: interpretation
    ax = fig.add_subplot(gs[3, :])
    note(ax, "How to read this figure",
         "(a-b) Areas respond to the stimulus and to the lick on hit trials. "
         "(c) Spike counts per 10-ms bin are sparse (mostly 0 or 1). "
         "(d) Left: raw counts, all trials of the session. Trials of different conditions differ in mean activity in "
         "both areas, so the areas look correlated even with no trial-by-trial interaction (the classic confound). "
         "Right: after subtracting the condition means, only the within-condition co-fluctuation remains. That is "
         "the 'noise correlation' that RRR, CCA and Granger measure. "
         "(e) In the window used below, single pairs of 10-ms bins correlate at about r = 0.01. Single pairs are "
         "therefore uninformative. The methods instead look for a few population directions that carry the shared "
         "signal, which is why they pool 20 units per area. "
         "Interpretation limit: a shared residual can come from direct communication or from a common input to "
         "both areas (e.g. arousal, movement). None of the methods can tell these apart. Movement regression (fig. "
         "02, 04, 06) only removes the measured part.", width=150)
    fig.suptitle(f"Data and residuals (example {SID}, {COHORT}, {COND_LABEL[COND]}, 30 trials)", fontweight="bold",
                 y=0.995)
    save(fig, "01_data_residuals")


def fig_samples_controls(res, metas, sl):
    fig = plt.figure(figsize=(7.2, 6.6))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.5)
    # A: window grid
    ax = fig.add_subplot(gs[0, :2])
    centers, widx = cl.window_grid()
    edges = cl.bin_edges()
    for i, ce in enumerate(centers[:12]):
        ax.add_patch(Rectangle((ce - 0.05, i), 0.1, 0.7, color="#bbbbbb"))
    k = int(np.argmin(np.abs(centers - POST)))
    ax.add_patch(Rectangle((POST - 0.05, 13.5), 0.1, 0.9, color="#1f77b4"))
    ax.text(POST + 0.07, 13.9, "RRR / CCA window (100 ms)", va="center", fontsize=6)
    ti = granger_t_index(POST)
    ax.add_patch(Rectangle((edges[ti[0]], 15.2), len(ti) * C["bin_s"], 0.9, color="#a03b8b"))
    ax.add_patch(Rectangle((edges[ti[0]] - C["granger_order"] * C["bin_s"], 15.2), C["granger_order"] * C["bin_s"],
                           0.9, color="#a03b8b", alpha=0.35))
    ax.text(edges[ti[-1]] + 0.03, 15.65, "Granger window (150 ms) + 30 ms of past", va="center", fontsize=6)
    ax.text(centers[11] + 0.08, 5.5, f"... {len(centers)} windows, every 20 ms, to 1.0 s", fontsize=6)
    ax.set(xlim=(-0.5, 1.1), ylim=(-0.5, 17), yticks=[], xlabel="time from trial start (s)",
           title=f"Sliding windows (centre {POST} s highlighted)")
    stim_line(ax)
    label(ax, "a")
    # B: sample matrix
    ax = fig.add_subplot(gs[0, 2])
    Xr = res["X_rows"]
    lim = np.percentile(np.abs(Xr), 99)
    ax.imshow(Xr, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    for t in range(10, len(Xr), 10):
        ax.axhline(t - 0.5, color="w", lw=0.2)
    ax.set(xlabel=f"{SHORT[A]} unit", ylabel="sample (trial x bin)", title="One window = 300 x 20 matrix")
    label(ax, "b")
    # C-E: across sessions: units, rates, trials
    m = pd.DataFrame([{"cohort": x["cohort"], "mouse": x["mouse_id"], **{f"n_{SHORT[a]}": x["n_units_by_area"][a]
                       for a in C["areas"]}, **{f"r_{SHORT[a]}": x["mean_rate_hz_by_area"][a] for a in C["areas"]},
                       **{f"t_{c}": x["n_trials_by_condition"].get(c, 0) for c in C["conditions"]}} for x in metas])
    for j, (pref, ylab, title) in enumerate([("n_", "good units (>= 1 Hz)", "Units available (20 used)"),
                                             ("r_", "mean rate (Hz)", "Firing rates by cohort")]):
        ax = fig.add_subplot(gs[1, j])
        for i, a in enumerate(C["areas"]):
            for jj, c in enumerate(("R+", "R-")):
                v = m[m.cohort == c][pref + SHORT[a]]
                x = i + (jj - 0.5) * 0.35
                ax.scatter(np.full(len(v), x), v, s=8, color=COLORS[c], alpha=0.7, lw=0,
                           label=c if i == 0 else None)
        if pref == "n_":
            ax.axhline(C["n_units_source"] * 2, color="k", ls="--", lw=0.6)
            ax.text(2.3, C["n_units_source"] * 2, " 40 = 20 + 20\n (ceiling units)", fontsize=5, va="center")
        ax.set_xticks(range(3))
        ax.set_xticklabels([SHORT[a] for a in C["areas"]])
        ax.set(ylabel=ylab, title=title)
        if j == 0:
            ax.legend(frameon=False)
            label(ax, "c")
    ax = fig.add_subplot(gs[1, 2])
    for i, c in enumerate(C["conditions"]):
        for jj, co in enumerate(("R+", "R-")):
            v = m[m.cohort == co][f"t_{c}"]
            x = i + (jj - 0.5) * 0.35
            ax.scatter(np.full(len(v), x), v, s=8, color=COLORS[co], alpha=0.7, lw=0)
            ax.text(x, -0.2, f"{(v >= C['n_trials']).sum()}", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=5, color=COLORS[co])
    ax.axhline(C["n_trials"], color="k", ls="--", lw=0.6)
    ax.set_xticks(range(4))
    ax.set_xticklabels([COND_LABEL[c] for c in C["conditions"]], rotation=25, ha="right")
    ax.set(ylabel="trials in session", yscale="log", title="Trials per condition (>= 30 needed)")
    ax.text(1.0, -0.42, "numbers: mice with >= 30", transform=ax.transAxes, ha="right", fontsize=5)
    # F: within-area ceiling vs between
    ax = fig.add_subplot(gs[2, 0])
    p = post_window(sl)
    bet = p[p.method == "RRR"].groupby(["mouse_id", "cohort", "source"]).value.mean()
    wit = p[p.method == "RRR_within"].groupby(["mouse_id", "cohort", "source"]).value.mean()
    j = pd.concat([bet.rename("between"), wit.rename("within")], axis=1).dropna().reset_index()
    for c in ("R+", "R-"):
        d = j[j.cohort == c]
        ax.scatter(d.within, d.between, s=8, color=COLORS[c], alpha=0.7, lw=0, label=c)
    lim = max(j.within.max(), j.between.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="k", lw=0.5)
    ax.plot([0, lim], [0, lim / 2], color="k", lw=0.5, ls=":")
    ax.set(xlabel="within-area RRR (source → 20 other source units)", ylabel="between-area RRR",
           title="Ceiling: between ≈ half of within", xlim=(0, lim), ylim=(0, lim))
    ax.legend(frameon=False)
    label(ax, "d")
    # G: movement regression
    ax = fig.add_subplot(gs[2, 1])
    for meth, base, col in [("RRR", "RRR_mvt_regressed", "#1f9d5a"), ("PCA_CCA", "PCA_CCA_mvt_regressed", "#2c6fdb")]:
        raw = p[p.method == meth].groupby(["session_id", "condition", "pair", "source"]).value.mean()
        reg = p[p.method == base].groupby(["session_id", "condition", "pair", "source"]).value.mean()
        jj = pd.concat([raw.rename("raw"), reg.rename("reg")], axis=1).dropna()
        sc = jj.raw.abs().max()
        ax.scatter(jj.raw / sc, jj.reg / sc, s=5, alpha=0.6, lw=0, color=col,
                   label=f"{meth.replace('_', '→')}: slope {np.polyfit(jj.raw, jj.reg, 1)[0]:.2f}")
    ax.plot([-0.2, 1], [-0.2, 1], color="k", lw=0.5)
    ax.set(xlabel="raw value (scaled to max)", ylabel="movement regressed out", title="Movement barely matters")
    ax.legend(frameon=False, loc="upper left")
    label(ax, "e")
    ax = fig.add_subplot(gs[2, 2])
    note(ax, "Why these controls",
         "(a-b) Each window is a 300 x 20 matrix per area; folds keep a trial's 10 bins together, so neighbouring "
         "bins never sit in both training and test sets. "
         "(c) Every area has >= 40 good units, so 20 + 20 can be drawn and unit count is identical across mice "
         "(RRR and CCA grow with unit number). "
         "(d) Within-area prediction is the most a 20-unit source can explain; between-area values are about half "
         "of it. "
         "(e) Mean jaw opening and whisker speed per 10-ms bin regressed out of both areas: values barely change, so "
         "the shared fluctuations are not mainly these measured movements. "
         "Trial counts: whisker hits in R- (rewarded only in R+) rarely reach 30, so that cohort test rests on 3 "
         "mice.", width=40)
    fig.suptitle("Samples, matching and controls (all 16 sessions; post-stimulus 0–0.3 s in d-e)",
                 fontweight="bold", y=0.995)
    save(fig, "02_samples_controls")


def fig_rrr_steps(res):
    r = res["rrr"]
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 3, hspace=0.7, wspace=0.55)
    post = r["post"]
    # A: X and Y matrices
    ax = fig.add_subplot(gs[0, 0])
    big = np.c_[res["X_rows"][:60], np.full((60, 1), np.nan), res["Y_rows"][:60]]
    lim = np.nanpercentile(np.abs(big), 99)
    ax.imshow(big, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set(xlabel=f"X: {SHORT[A]} units  |  Y: {SHORT[B]} units", ylabel="sample (first 6 trials)",
           title="Y ≈ X · B   (B: 20 x 20 weights)")
    label(ax, "a")
    # B: ridge alpha
    ax = fig.add_subplot(gs[0, 1])
    al = sorted(post["perf"])
    mm = [post["perf"][a][:, -1].mean() for a in al]
    ss = [post["perf"][a][:, -1].std() / np.sqrt(C["cv_folds"]) for a in al]
    ax.errorbar(al, mm, ss, marker="o", ms=3, color="k", lw=0.8)
    ax.scatter([post["alpha"]], [post["perf_full"].mean()], s=40, facecolor="none", edgecolor="r", zorder=5)
    ax.set(xscale="log", xlabel="ridge alpha (× mean source variance)", ylabel="CV performance, full rank",
           title="Step 1: choose the ridge penalty")
    label(ax, "b")
    # C: performance vs rank
    ax = fig.add_subplot(gs[0, 2])
    ranks = np.arange(1, len(post["mean"]) + 1)
    for key, lab, col in [("post", f"{SHORT[A]}→{SHORT[B]}", "k"), ("within", f"{SHORT[A]}→{SHORT[A]} (ceiling)",
                                                                     CEIL_COL), ("shuffle", "trial-shuffled", NULL_COL)]:
        d = r[key]
        ls = "--" if key == "within" else "-"
        ax.errorbar(ranks, d["mean"], d["sem"], color=col, lw=0.9, ls=ls, marker="o", ms=2.5, label=lab)
        ax.plot(d["opt"], d["mean"][d["opt"] - 1], marker="v", color="r", ms=5)
    ax.axhline(post["perf_full"].mean(), color="k", lw=0.5, ls=":")
    ax.axhline(0, color="k", lw=0.4)
    ax.set(xlabel="rank (number of communication dimensions)", ylabel="CV performance (1 − NSE)",
           title="Step 2: limit the rank (▼ = chosen)")
    ax.legend(frameon=False, loc="center right")
    label(ax, "c")
    # D: held-out prediction for the best-predicted target unit
    ax = fig.add_subplot(gs[1, :2])
    Yt, Yp = post["Y_true"], post["Y_pred"]
    r2 = 1 - ((Yt - Yp) ** 2).sum(0) / (Yt ** 2).sum(0)
    u = int(np.argmax(r2))
    order = np.argsort(post["test_trial"], kind="stable")
    sel = order[:80]
    xs = np.arange(len(sel))
    ax.step(xs, Yt[sel, u], where="mid", color=AREA_COL[B], lw=0.8, label="actual residual (held out)")
    ax.plot(xs, Yp[sel, u], color="k", lw=1, label=f"prediction from {SHORT[A]} (rank {post['opt']})")
    for t in range(10, len(sel), 10):
        ax.axvline(t - 0.5, color="#dddddd", lw=0.5)
    ax.set(xlabel="10-ms bins of 8 held-out trials (0.05–0.15 s each)", ylabel="spike-count residual",
           title=f"Step 3: held-out prediction, best {SHORT[B]} unit (R² = {r2[u]:.3f}); "
                 f"median over 20 units R² = {np.median(r2):.4f}")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    label(ax, "d")
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(r2, bins=15, color=AREA_COL[B])
    ax.axvline(0, color="k", lw=0.5)
    ax.set(xlabel="held-out R² per target unit", ylabel="units", title="Prediction is weak and spread thin")
    # E: communication subspace vs dominant source dimensions
    ax = fig.add_subplot(gs[2, 0])
    pcm = post["pc_perf"].mean(0)
    ax.plot(ranks, post["mean"], color="k", marker="o", ms=2.5, label="RRR: best predictive dims")
    ax.plot(ranks, pcm, color=AREA_COL[A], marker="s", ms=2.5, label=f"top-k {SHORT[A]} PCs (most variance)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set(xlabel="number of source dimensions", ylabel="CV performance",
           title="Step 4: is it the dominant activity?")
    ax.legend(frameon=False)
    label(ax, "e")
    # F: null
    ax = fig.add_subplot(gs[2, 1])
    null_hist(ax, r["null"], post["mean"][post["opt"] - 1], "CV performance at optimal rank")
    label(ax, "f")
    ax = fig.add_subplot(gs[2, 2])
    note(ax, "Reading RRR",
         f"Value = variance of held-out {SHORT[B]} residuals explained from {SHORT[A]} residuals in the same bin "
         f"(here {post['mean'][post['opt'] - 1]:.4f}, i.e. {100 * post['mean'][post['opt'] - 1]:.2f}%). "
         "Small numbers are expected: 10-ms counts are mostly Poisson noise. "
         "Rank = how many source dimensions carry it (the 'communication subspace'). If a few dimensions do as "
         "well as all 20, communication is low-dimensional. "
         "(e) If the best predictive dimensions do better than the top-variance PCs, the shared signal is not "
         "simply the source's largest fluctuation (Semedo 2019). "
         f"Pre-stimulus (−0.2 s) value: {r['pre']['mean'][r['pre']['opt'] - 1]:.4f}. "
         "RRR is directed only in the sense of which area is the predictor: same-bin prediction carries no "
         "temporal order.", width=38)
    fig.suptitle(f"Reduced-rank regression, step by step ({SID}, {COND_LABEL[COND]}, window {POST - 0.05:.2f}–"
                 f"{POST + 0.05:.2f} s, {SHORT[A]} → {SHORT[B]})", fontweight="bold", y=0.995)
    save(fig, "03_rrr_steps")


def example_ts(ax, ses, method, source, target, vals=("value",), title=""):
    d = ses[(ses.method == method) & (ses.condition == COND) & (ses.pair == f"{A} | {B}")]
    if source is not None:
        d = d[(d.source == source) & (d.target == target)]
    g = d.groupby("window_center")[["value", "null_mean", "null_sd"]].mean()
    ax.plot(g.index, g.value, color="k", lw=1, label="true")
    ax.plot(g.index, g.null_mean, color=NULL_COL, lw=0.8, label="shuffle null mean ± sd")
    ax.fill_between(g.index, g.null_mean - g.null_sd, g.null_mean + g.null_sd, color=NULL_COL, alpha=0.25, lw=0)
    stim_line(ax)
    ax.set(xlabel="time from trial start (s)", title=title)
    return g


def zdist(ax, sl_raw, method):
    d = sl_raw[sl_raw.method == method]
    z = ((d.value - d.null_mean) / d.null_sd).replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(z.clip(-5, 15), bins=60, color="#777777")
    ax.axvline(1.96, color="r", lw=0.7)
    ax.set(xlabel="z = (true − null mean) / null sd, every window", ylabel="count",
           title=f"{100 * (z > 1.96).mean():.0f}% of windows above z = 1.96")


def fig_rrr_results(sl, ses, raw):
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(4, 4, hspace=0.85, wspace=0.6)
    for j, (s, t) in enumerate(((A, B), (B, A))):
        ax = fig.add_subplot(gs[0, j])
        example_ts(ax, ses, "RRR", s, t, title=f"Example: {SHORT[s]}→{SHORT[t]}")
        w = ses[(ses.method == "RRR_within") & (ses.condition == COND) & (ses.pair == f"{A} | {B}") &
                (ses.source == s)].groupby("window_center").value.mean()
        ax.plot(w.index, w, color=CEIL_COL, ls="--", lw=0.8, label="within-area ceiling")
        m = ses[(ses.method == "RRR_mvt_regressed") & (ses.condition == COND) & (ses.pair == f"{A} | {B}") &
                (ses.source == s)].groupby("window_center").value.mean()
        ax.plot(m.index, m, color=MVT_COL, lw=0.7, label="movement regressed")
        ax.set_ylabel("CV performance")
        if j == 0:
            ax.legend(frameon=False, loc="upper left", fontsize=5)
            label(ax, "a")
    ax = fig.add_subplot(gs[0, 2])
    d = ses[(ses.method == "RRR") & (ses.condition == COND) & (ses.pair == f"{A} | {B}") & (ses.source == A)]
    g = d.groupby("window_center")["rank"].mean()
    ax.plot(g.index, g, color="k")
    stim_line(ax)
    ax.set(xlabel="time (s)", ylabel="optimal rank", title="Rank over time (example)")
    label(ax, "b")
    ax = fig.add_subplot(gs[0, 3])
    zdist(ax, raw, "RRR")
    label(ax, "c")
    rr = sl[sl.method == "RRR"]
    for j, c in enumerate(C["conditions"]):
        ax = fig.add_subplot(gs[1, j])
        cohort_ts(ax, rr[(rr.condition == c) & (rr.pair == f"{A} | {B}") & (rr.source == A)])
        ax.set(title=COND_LABEL[c], xlabel="time (s)")
        if j == 0:
            ax.set_ylabel(f"excess RRR {SHORT[A]}→{SHORT[B]}")
            ax.legend(frameon=False, loc="upper left")
            label(ax, "d")
    for j, c in enumerate(("W_hit", "A_hit")):
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2])
        cohort_dots(ax, post_window(rr[rr.condition == c]), directed_groups("RRR"))
        ax.set(ylabel="excess, 0–0.3 s", title=f"{COND_LABEL[c]}: every pair and direction, one dot per mouse")
        if j == 0:
            label(ax, "e")
    ax = fig.add_subplot(gs[3, :2])
    ra = raw[raw.method == "RRR"]
    ax.hist(ra["rank"], bins=np.arange(0.5, 11.5), color="k", alpha=0.7)
    ax.set(xlabel="optimal rank (all sessions, windows, pairs)", ylabel="count",
           title=f"Communication is low-dimensional: median rank {ra['rank'].median():.0f}")
    label(ax, "f")
    ax = fig.add_subplot(gs[3, 2:])
    note(ax, "Interpretation",
         "(a) The true value rises above the shuffle band after the stimulus and around the lick; the shuffle keeps "
         "every PSTH, so the rise is co-fluctuation, not shared response. "
         "(d) excess = true − null mean, averaged per mouse; shading = s.e.m. over mice. "
         "(e) Post-stimulus averages per mouse; p values uncorrected, Mann-Whitney and Welch. No direction or pair "
         "differs consistently between cohorts. "
         "(f) Most windows need 1-3 dimensions. "
         "Caveat: at 10-ms bins RRR explains < 1% of variance and is the least sensitive method on synthetic data "
         "(fig. 09).", width=60)
    fig.suptitle("Reduced-rank regression: in time and across mice", fontweight="bold", y=0.995)
    save(fig, "04_rrr_results")


def fig_cca_steps(res):
    c = res["cca"]
    post = c["post"]
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 3, hspace=0.7, wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    for v, a in ((post["var_x"], A), (post["var_y"], B)):
        ax.plot(np.arange(1, len(v) + 1), np.cumsum(v), color=AREA_COL[a], marker="o", ms=2, label=SHORT[a])
    ax.axvline(C["cca_pcs"], color="k", ls=":", lw=0.6)
    ax.set(xlabel="number of PCs", ylabel="cumulative variance", title="Step 1: keep the top 10 PCs per area")
    ax.legend(frameon=False)
    label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    d = np.arange(1, post["train_cc"].shape[1] + 1)
    ax.plot(d, post["train_cc"].mean(0), color="k", marker="o", ms=2.5, label="training trials")
    ax.plot(d, np.nanmean(post["test_cc"], 0), color="#2c6fdb", marker="o", ms=2.5, label="held-out trials")
    ax.plot(d, c["shuffle"]["train_cc"].mean(0), color=NULL_COL, ls="--", label="shuffled: training")
    ax.plot(d, np.nanmean(c["shuffle"]["test_cc"], 0), color=NULL_COL, label="shuffled: held-out")
    ax.axhline(0, color="k", lw=0.4)
    ax.set(xlabel="canonical pair", ylabel="correlation", title="Step 2: CCA overfits; only held-out counts")
    ax.legend(frameon=False)
    label(ax, "b")
    ax = fig.add_subplot(gs[0, 2])
    f0 = post["fold0"]
    ax.scatter(f0["train_a"], f0["train_b"], s=2, color="#bbbbbb", lw=0,
               label=f"training r = {np.corrcoef(f0['train_a'], f0['train_b'])[0, 1]:.2f}")
    ax.scatter(f0["test_a"], f0["test_b"], s=4, color="#2c6fdb", lw=0,
               label=f"held-out r = {np.corrcoef(f0['test_a'], f0['test_b'])[0, 1]:.2f}")
    ax.set(xlabel=f"{SHORT[A]} canonical variate 1", ylabel=f"{SHORT[B]} canonical variate 1",
           title="Fold 1: projections on the first pair")
    ax.legend(frameon=False, loc="upper left")
    label(ax, "c")
    ax = fig.add_subplot(gs[1, 0])
    tc = post["test_cc"][:, 0]
    ax.scatter(np.arange(1, len(tc) + 1), tc, color="#2c6fdb", s=10)
    ax.axhline(tc.mean(), color="#2c6fdb", lw=1, label=f"mean of folds = {tc.mean():.3f} (used)")
    ax.axhline(post["pooled"], color="r", lw=1, ls="--", label=f"pooled across folds = {post['pooled']:.3f}")
    ax.axhline(0, color="k", lw=0.4)
    ax.set(xlabel="fold", ylabel="held-out canonical r", title="Step 3: one value per fold, then average")
    ax.legend(frameon=False, loc="lower right")
    label(ax, "d")
    for j, (w, a) in enumerate(((post["w_a"], A), (post["w_b"], B))):
        ax = fig.add_subplot(gs[1, 1 + j])
        o = np.argsort(w)
        ax.bar(np.arange(len(w)), w[o], color=AREA_COL[a])
        ax.axhline(0, color="k", lw=0.4)
        ax.set(xlabel=f"{SHORT[a]} units (sorted)", ylabel="weight (max = 1)",
               title=f"Step 4: which {SHORT[a]} units carry it")
        if j == 0:
            label(ax, "e")
    ax = fig.add_subplot(gs[2, 0])
    ks = sorted(c["by_k"])
    ax.plot(ks, [c["by_k"][k][0] for k in ks], color="#2c6fdb", marker="o", ms=2.5, label="true")
    ax.plot(ks, [c["by_k"][k][1] for k in ks], color=NULL_COL, marker="o", ms=2.5, label="shuffle (10)")
    ax.axvline(C["cca_pcs"], color="k", ls=":", lw=0.6)
    ax.set(xlabel="PCs kept per area", ylabel="held-out canonical r 1", title="Why reduce with PCA first")
    ax.legend(frameon=False)
    label(ax, "f")
    ax = fig.add_subplot(gs[2, 1])
    null_hist(ax, c["null"], post["test_cc"][:, 0].mean(), "held-out canonical r 1")
    label(ax, "g")
    ax = fig.add_subplot(gs[2, 2])
    note(ax, "Reading PCA → CCA",
         "CCA finds one weighted sum of each area's activity, chosen so that the two sums correlate as much as "
         "possible across samples. With 300 samples and 10 + 10 dimensions this overfits: (b) training "
         "correlations are high even for shuffled data. So the axes are fitted on 9/10 of trials and evaluated on "
         "the held-out tenth. "
         "(d) The sign of a canonical pair is arbitrary in each fold. Pooling held-out projections across folds "
         "mixes signs and cancels the correlation (red, the first version's bug). Hence one correlation per fold, "
         "then the mean. "
         "(f) Keeping few PCs removes noise dimensions but may drop the shared one; 10 was chosen in advance. "
         "Symmetric: CCA says how strongly the areas co-fluctuate, not who drives whom.", width=38)
    fig.suptitle(f"PCA → CCA, step by step ({SID}, {COND_LABEL[COND]}, window {POST - 0.05:.2f}–"
                 f"{POST + 0.05:.2f} s, {SHORT[A]} ↔ {SHORT[B]})", fontweight="bold", y=0.995)
    save(fig, "05_cca_steps")


def fig_cca_results(sl, ses, raw):
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(3, 4, hspace=0.8, wspace=0.6)
    ax = fig.add_subplot(gs[0, :2])
    example_ts(ax, ses, "PCA_CCA", None, None, title=f"Example: {SHORT[A]} ↔ {SHORT[B]}, {COND_LABEL[COND]}")
    m = ses[(ses.method == "PCA_CCA_mvt_regressed") & (ses.condition == COND) &
            (ses.pair == f"{A} | {B}")].groupby("window_center").value.mean()
    ax.plot(m.index, m, color=MVT_COL, lw=0.7, label="movement regressed")
    t3 = ses[(ses.method == "PCA_CCA") & (ses.condition == COND) &
             (ses.pair == f"{A} | {B}")].groupby("window_center").cc_top3.mean()
    ax.plot(t3.index, t3, color="#2c6fdb", lw=0.7, ls="--", label="mean of pairs 1-3")
    ax.set_ylabel("held-out canonical r")
    ax.legend(frameon=False, fontsize=5, loc="upper left")
    label(ax, "a")
    ax = fig.add_subplot(gs[0, 2])
    zdist(ax, raw, "PCA_CCA")
    label(ax, "b")
    ax = fig.add_subplot(gs[0, 3])
    nm = raw[raw.method == "PCA_CCA"]
    ax.hist(nm.null_mean, bins=40, color=NULL_COL)
    ax.set(xlabel="shuffle-null mean", ylabel="count", title="The null is not zero")
    cc = sl[sl.method == "PCA_CCA"]
    for j, c in enumerate(C["conditions"]):
        ax = fig.add_subplot(gs[1, j])
        cohort_ts(ax, cc[(cc.condition == c) & (cc.pair == f"{A} | {B}")])
        ax.set(title=COND_LABEL[c], xlabel="time (s)")
        if j == 0:
            ax.set_ylabel(f"excess CCA {SHORT[A]}↔{SHORT[B]}")
            ax.legend(frameon=False, loc="upper left")
            label(ax, "c")
    ax = fig.add_subplot(gs[2, :2])
    pw = post_window(cc)

    def cond_pair_groups(d):
        return [(f"{COND_LABEL[c].split()[0][0]}{COND_LABEL[c].split()[1][0]} {g}", gd[gd.condition == c])
                for c in ("W_hit", "A_hit", "NS_cr") for g, gd in pair_groups(d)]
    cohort_dots(ax, pw, cond_pair_groups)
    ax.set(ylabel="excess, 0–0.3 s", title="Per mouse: WH whisker hit, AH auditory hit, NC no-stim CR")
    label(ax, "d")
    ax = fig.add_subplot(gs[2, 2:])
    note(ax, "Interpretation",
         "(a) The true canonical correlation sits above the shuffle band after the stimulus. "
         "(c) Post-stimulus rises in both cohorts; on hits the R- mean is often higher, mainly on "
         "auditory hits (also no-stim CR), but not significantly (d). "
         "(b) The shuffle null has a positive mean (about 0.0-0.1): with 300 samples a held-out correlation "
         "is biased upwards by chance alignment. That is why every value is reported as excess over the null. "
         "Movement regression (red, a) leaves the curve almost unchanged. "
         "CCA is the most sensitive symmetric measure here (fig. 09).", width=60)
    fig.suptitle("PCA → CCA: in time and across mice", fontweight="bold", y=0.995)
    save(fig, "06_cca_results")


def fig_granger_steps(res):
    g = res["granger"]
    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.55)
    edges = cl.bin_edges()
    tc = edges[:-1] + C["bin_s"] / 2
    ax = fig.add_subplot(gs[0, :2])
    for i in range(3):
        ax.plot(tc, g["za"][i, :, 0] + i * 8, color=AREA_COL[A], lw=0.7)
        ax.plot(tc, g["zb"][i, :, 0] + i * 8 + 3.5, color=AREA_COL[B], lw=0.7)
    ti = g["t_index"]
    ax.axvspan(edges[ti[0]], edges[ti[-1] + 1], color="#a03b8b", alpha=0.12)
    stim_line(ax)
    ax.set(xlabel="time from trial start (s)", yticks=[], title=f"Step 1: residual PC 1 of {SHORT[A]} (green) and "
           f"{SHORT[B]} (yellow), 3 trials; shaded = Granger window")
    label(ax, "a")
    ax = fig.add_subplot(gs[0, 2])
    lags, xc = g["xcorr"]
    ax.bar(lags, xc, width=8, color=["#a03b8b" if l > 0 else "#cccccc" for l in lags])
    ax.axvline(0, color="k", lw=0.5)
    ax.set(xlabel=f"lag (ms; > 0: {SHORT[A]} leads)", ylabel="r (PC 1)", title="Lagged correlation in the window")
    label(ax, "b")
    ax = fig.add_subplot(gs[1, 0])
    vals = []
    for k, (s, t) in (("ab", (A, B)), ("ba", (B, A))):
        vr, vf = np.linalg.det(g[k]["restricted"]["cov"]), np.linalg.det(g[k]["full"]["cov"])
        vals.append((f"{SHORT[s]}→{SHORT[t]}", vr, vf, g[k]["gc"]))
    x = np.arange(len(vals))
    ax.bar(x - 0.18, [v[1] for v in vals], 0.34, color="#cccccc", label="own past only")
    ax.bar(x + 0.18, [v[2] for v in vals], 0.34, color="#a03b8b", label="+ other area's past")
    for i, v in enumerate(vals):
        ax.text(i, max(v[1], v[2]) * 1.005, f"GC = ln(ratio)\n= {v[3]:.4f}", ha="center", va="bottom", fontsize=5.5)
    lo = min(min(v[1], v[2]) for v in vals)
    hi = max(max(v[1], v[2]) for v in vals)
    ax.set(ylim=(lo * 0.98, hi * 1.03), xticks=x, xticklabels=[v[0] for v in vals],
           ylabel="det(residual covariance)", title="Step 2: does the other's past help?")
    ax.legend(frameon=False, loc="lower right")
    label(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    o = sorted(g["orders"])
    ax.plot(o, [g["orders"][k]["ab"] for k in o], color="#a03b8b", marker="o", ms=2.5,
            label=f"{SHORT[A]}→{SHORT[B]}")
    ax.plot(o, [g["orders"][k]["ba"] for k in o], color="#555555", marker="o", ms=2.5,
            label=f"{SHORT[B]}→{SHORT[A]}")
    ax.axvline(C["granger_order"], color="k", ls=":", lw=0.6)
    ax.set(xlabel="model order (past bins of 10 ms)", ylabel="GC", title="Step 3: how much past to use")
    ax2 = ax.twinx()
    ax2.plot(o, [g["orders"][k]["bic_ab"] for k in o], color="#a03b8b", ls=":", lw=0.7)
    ax2.set_ylabel("BIC, full model (dotted)", fontsize=5.5)
    ax2.tick_params(labelsize=5)
    ax.legend(frameon=False, loc="upper left")
    label(ax, "d")
    ax = fig.add_subplot(gs[1, 2])
    null_hist(ax, g["null"][:, 0], g["ab"]["gc"], f"GC {SHORT[A]}→{SHORT[B]}")
    label(ax, "e")
    ax = fig.add_subplot(gs[2, 0])
    fwd = [g["ab"]["gc"], g["ba"]["gc"]]
    rev = list(g["rev"])
    x = np.arange(2)
    ax.bar(x - 0.18, fwd, 0.34, color="#a03b8b", label="forward time")
    ax.bar(x + 0.18, rev, 0.34, color="#d9b3d0", label="time reversed")
    ax.set(xticks=x, xticklabels=[f"{SHORT[A]}→{SHORT[B]}", f"{SHORT[B]}→{SHORT[A]}"], ylabel="GC",
           title="Step 4: time-reversal check")
    ax.legend(frameon=False)
    net = (fwd[0] - fwd[1]) - (rev[0] - rev[1])
    ax.text(0.5, 0.97, f"net TRGC = (fwd A→B − B→A) − (rev A→B − B→A)\n= {net:+.4f}  "
                       f"({'>0: ' + SHORT[A] + ' leads' if net > 0 else '<0: ' + SHORT[B] + ' leads'})",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.3)
    label(ax, "f")
    ax = fig.add_subplot(gs[2, 1:])
    note(ax, "Reading Granger",
         "Granger asks a temporal question: knowing the target's own last 30 ms, does the source's last 30 ms "
         "improve the prediction of the target's next bin? GC = ln(error without source / error with source); 0 means "
         "no help. Fitted on all 30 trials x 15 time points together ('ensemble'), on 3 residual PCs per area, so "
         "slow shared trends are already removed by the residuals. "
         "(c) The improvement is tiny in absolute terms (ratio close to 1), but (e) it is far outside the trial-"
         "shuffle null. "
         "(d) Order 3 (30 ms) was fixed in advance; GC changes slowly with order. "
         "(f) Granger is fooled by common input with different delays or by unequal noise. Time reversal flips true "
         "causal asymmetries but not these artefacts, so the net TRGC is the robust directionality score "
         "(Haufe 2013, Winkler 2016). Still, 'Granger-causal' means 'temporally predictive', not anatomical "
         "causation.", width=78)
    fig.suptitle(f"Ensemble Granger causality, step by step ({SID}, {COND_LABEL[COND]}, window "
                 f"{edges[ti[0]]:.2f}–{edges[ti[-1] + 1]:.2f} s)", fontweight="bold", y=0.995)
    save(fig, "07_granger_steps")


def fig_granger_results(sl, ses, raw):
    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(4, 4, hspace=0.85, wspace=0.6)
    for j, (s, t) in enumerate(((A, B), (B, A))):
        ax = fig.add_subplot(gs[0, j])
        example_ts(ax, ses, "Granger", s, t, title=f"Example: {SHORT[s]}→{SHORT[t]}")
        d = ses[(ses.method == "Granger") & (ses.condition == COND) & (ses.pair == f"{A} | {B}") & (ses.source == s)]
        r = d.groupby("window_center").value_reversed.mean()
        ax.plot(r.index, r, color="#d9b3d0", lw=0.8, label="time reversed")
        ax.set_ylabel("GC")
        if j == 0:
            ax.legend(frameon=False, fontsize=5, loc="upper left")
            label(ax, "a")
    ax = fig.add_subplot(gs[0, 2])
    zdist(ax, raw, "Granger")
    label(ax, "b")
    ax = fig.add_subplot(gs[0, 3])
    nt = sl[(sl.method == "Granger_net_TRGC") & (sl.session_id == SID) & (sl.condition == COND)]
    for pair, d in nt.groupby("pair"):
        a_, b_ = pair.split(" | ")
        ax.plot(d.window_center, d.excess, lw=0.8, label=f"{SHORT[a_]}→{SHORT[b_]} (+)")
    ax.axhline(0, color="k", lw=0.5)
    stim_line(ax)
    ax.set(xlabel="time (s)", ylabel="net TRGC", title="Example: net directionality")
    ax.legend(frameon=False, fontsize=5)
    gr = sl[sl.method == "Granger"]
    for j, c in enumerate(C["conditions"]):
        ax = fig.add_subplot(gs[1, j])
        cohort_ts(ax, gr[(gr.condition == c) & (gr.pair == f"{A} | {B}") & (gr.source == A)])
        ax.set(title=COND_LABEL[c], xlabel="time (s)")
        if j == 0:
            ax.set_ylabel(f"excess GC {SHORT[A]}→{SHORT[B]}")
            ax.legend(frameon=False, loc="upper left")
            label(ax, "c")
    nt = sl[sl.method == "Granger_net_TRGC"]
    for j, c in enumerate(C["conditions"]):
        ax = fig.add_subplot(gs[2, j])
        cohort_ts(ax, nt[(nt.condition == c) & (nt.pair == f"{A} | {B}")])
        ax.set(title=COND_LABEL[c], xlabel="time (s)")
        if j == 0:
            ax.set_ylabel(f"net TRGC (+: {SHORT[A]} leads)")
            label(ax, "d")
    ax = fig.add_subplot(gs[3, :2])
    cohort_dots(ax, post_window(nt[nt.condition.isin(["W_hit", "A_hit"])]),
                lambda d: [(f"{'WH' if c == 'W_hit' else 'AH'} {g}", gd[gd.condition == c])
                           for c in ("W_hit", "A_hit") for g, gd in pair_groups(d)])
    ax.set(ylabel="net TRGC, 0–0.3 s", title="Per mouse: direction of the first area in each pair = +")
    label(ax, "e")
    ax = fig.add_subplot(gs[3, 2:])
    note(ax, "Interpretation",
         "(a) GC in both directions rises after the stimulus; the reversed-time GC (pink) rises too, so part of "
         "the rise is not directional. "
         "(c) Excess GC is above zero in both cohorts and every condition. "
         "(d, e) Net TRGC isolates the asymmetry: it hovers near 0 with a few transient deflections. "
         "The cohort clusters found in 003 (auditory hits, MO/FRP→STR around 0 s; GC MO/FRP→SSp-w at 0.74-0.84 s) "
         "are the scattered ones expected by chance across about 72 tests. "
         "Granger is the most sensitive method in simulation (fig. 09) but needs the 10-ms structure; it measures "
         "a different thing from CCA and agrees with it weakly.", width=60)
    fig.suptitle("Ensemble Granger: in time, directionality and across mice", fontweight="bold", y=0.995)
    save(fig, "08_granger_results")


def fig_methods(sl, sim):
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(3, 4, hspace=0.8, wspace=0.65)
    s = sim.groupby("coupling").agg(["mean", "sem"])
    for j, (col, null, lab) in enumerate([("rrr", "rrr_shuffle", "RRR"), ("cca", "cca_shuffle", "PCA→CCA"),
                                          ("gc_ab", "gc_null", "Granger A→B")]):
        ax = fig.add_subplot(gs[0, j])
        x = s.index
        ax.errorbar(x, s[(col, "mean")], s[(col, "sem")], color="k", marker="o", ms=3, lw=0.9, label="true")
        ax.errorbar(x, s[(null, "mean")], s[(null, "sem")], color=NULL_COL, marker="o", ms=3, lw=0.9,
                    label="shuffle")
        if col == "gc_ab":
            ax.errorbar(x, s[("gc_ba", "mean")], s[("gc_ba", "sem")], color="#a03b8b", marker="o", ms=3, lw=0.9,
                        label="B→A (wrong way)")
        ax.set(xlabel="true coupling A→B", title=lab)
        if j == 0:
            ax.set_ylabel("value")
            label(ax, "a")
        ax.legend(frameon=False, fontsize=5)
    ax = fig.add_subplot(gs[0, 3])
    sens = {}
    for col, null, lab in [("rrr", "rrr_shuffle", "RRR"), ("cca", "cca_shuffle", "CCA"), ("gc_ab", "gc_null", "GC"),
                           ("net_trgc", None, "net TRGC")]:
        e = sim[col] - (sim[null] if null else 0)
        e0 = e[sim.coupling == 0]
        sens[lab] = [(e[sim.coupling == c].mean() - e0.mean()) / e0.std() for c in sorted(sim.coupling.unique())]
    for lab, v in sens.items():
        ax.plot(sorted(sim.coupling.unique()), v, marker="o", ms=3, label=lab)
    ax.axhline(2, color="k", ls=":", lw=0.6)
    ax.set(xlabel="true coupling", ylabel="signal / noise (z vs coupling 0)", title="Sensitivity")
    ax.legend(frameon=False)
    # B: agreement scatter
    pw = post_window(sl)
    per = {}
    for meth in ("RRR", "PCA_CCA", "Granger"):
        per[meth] = pw[pw.method == meth].groupby(["session_id", "cohort", "condition", "pair"]).excess.mean()
    J = pd.concat(per, axis=1).dropna().reset_index()
    for j, (x, y) in enumerate((("RRR", "PCA_CCA"), ("RRR", "Granger"), ("PCA_CCA", "Granger"))):
        ax = fig.add_subplot(gs[1, j])
        for c in C["conditions"]:
            d = J[J.condition == c]
            ax.scatter(d[x], d[y], s=5, color=COND_COL[c], alpha=0.7, lw=0, label=COND_LABEL[c])
        rho = stats.spearmanr(J[x], J[y]).statistic
        ax.set(xlabel=x.replace("_", "→"), ylabel=y.replace("_", "→"), title=f"Spearman ρ = {rho:.2f}")
        if j == 0:
            ax.legend(frameon=False, fontsize=5, markerscale=2)
            label(ax, "b")
    ax = fig.add_subplot(gs[1, 3])
    note(ax, "Agreement",
         "One dot = one session x pair x condition, mean excess 0–0.3 s (directed methods averaged over the two "
         "directions). The condition ordering (hits > misses > no-stim) is shared; within a condition, sessions "
         "rank differently across methods. Zero-lag (RRR, CCA) and lagged (Granger) measures capture different "
         "aspects of coupling.", width=30)
    # C: power
    ax = fig.add_subplot(gs[2, :2])
    rows = []
    for meth in ("RRR", "PCA_CCA", "Granger"):
        for c in C["conditions"]:
            d = pw[(pw.method == meth) & (pw.condition == c)]
            per_m = d.groupby(["mouse_id", "cohort", "pair", "source"]).excess.mean().reset_index()
            for (pair, srcn), dd in per_m.groupby(["pair", "source"]):
                n1, n2 = (dd.cohort == "R+").sum(), (dd.cohort == "R-").sum()
                if n1 < 3 or n2 < 3:
                    continue
                sd = np.sqrt((dd[dd.cohort == "R+"].excess.var() + dd[dd.cohort == "R-"].excess.var()) / 2)
                mdd = (stats.norm.ppf(0.975) + stats.norm.ppf(0.8)) * sd * np.sqrt(1 / n1 + 1 / n2)
                rows.append(dict(method=meth, condition=c, pct=100 * mdd / abs(dd.excess.mean())))
    P = pd.DataFrame(rows).groupby(["method", "condition"]).pct.median().unstack()[list(C["conditions"])]
    im = ax.imshow(P.to_numpy(), cmap="Greys", vmin=0, vmax=150)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            ax.text(j, i, f"{P.iloc[i, j]:.0f}%", ha="center", va="center", fontsize=6,
                    color="w" if P.iloc[i, j] > 80 else "k")
    ax.set_xticks(range(P.shape[1]))
    ax.set_xticklabels([COND_LABEL[c] for c in P.columns], rotation=20, ha="right")
    ax.set_yticks(range(P.shape[0]))
    ax.set_yticklabels([m.replace("_", "→") for m in P.index])
    ax.set_title("Smallest R+/R- difference detectable (80% power), % of mean excess")
    plt.colorbar(im, ax=ax, fraction=0.03)
    label(ax, "c")
    ax = fig.add_subplot(gs[2, 2:])
    note(ax, "Which method answers what",
         "RRR: how much of B's trial-to-trial activity can be read out linearly from A in the same 10 ms, and in "
         "how many dimensions. Directed by construction, not by time. Least sensitive at this bin size.\n"
         "PCA→CCA: how strongly the most-coupled pair of population axes co-fluctuates. Symmetric, sensitive, "
         "needs the shuffle-null subtraction (biased upward).\n"
         "Granger: whether A's recent past predicts B's next 10 ms beyond B's own past. Directed and lagged; "
         "check with time reversal.\n"
         "(c) With 8 vs 8 mice (3 R- for whisker hits), only cohort differences of about the size of the effect "
         "itself are detectable: absence of a difference here is weak evidence.", width=60)
    fig.suptitle("Comparing the methods: simulation, agreement and power", fontweight="bold", y=0.995)
    save(fig, "09_methods_compared")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true", help="reuse the cached worked example")
    args = parser.parse_args()
    res = pickle.load(open(CACHE, "rb")) if args.replot and CACHE.exists() else compute()
    sl, ses, metas, sim = load_group()
    raw = pd.read_parquet(OUT / "pilot_long.parquet") if (OUT / "pilot_long.parquet").exists() else ses
    fig_pipeline()
    fig_data(res, metas)
    fig_samples_controls(res, metas, sl)
    fig_rrr_steps(res)
    fig_rrr_results(sl, ses, raw)
    fig_cca_steps(res)
    fig_cca_results(sl, ses, raw)
    fig_granger_steps(res)
    fig_granger_results(sl, ses, raw)
    fig_methods(sl, sim)


if __name__ == "__main__":
    main()
