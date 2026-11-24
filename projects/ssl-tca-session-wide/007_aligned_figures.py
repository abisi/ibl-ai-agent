"""Figures for the event-aligned TCA (006g Kuma GPU fits; 006 haas fits with --tca-dir tca_aligned), per session.

1. rank_model_<sid>: per TCA type: reconstruction error per restart (best = line) with the condition-mean model,
   restart similarity to the best fit, error decrease per added component; thresholds and chosen rank marked.
2. factors_<sid>_<type>_r<rank>: one row per component: unit weights (sorted by area group, group names under the last
   row, palette from ephys_utilities), time factor (event at 0; jaw / lick: median stimulus time in red, lick: median
   jaw onset in green), trial weights across the session (aligned with the P(lick) curves in the header row).
3. recon_<sid>_<type>_r<rank>: full-model variance explained, loss of variance explained when each component alone is
   removed (larger than the full model = components cancelling each other), signed component overlap (Tucker
   congruence; < -0.5 flags a cancelling pair = CP degeneracy); then for the 4 best-fitted units: class-averaged data
   (lines) vs model (dots), and single trials, data vs model, sorted by class (colour strip) and session order.
Trial markers (user 2026-11-22): colour = trial type (whisker #f7b519, auditory #2c2cdb, no stim #262626); filled = lick,
open = no lick; passive trials in a lightened version of the colour; trials left out of the fit as outliers = grey x.

Run: python 007_aligned_figures.py --sessions <sid> ... [--rank N] [--tca-dir tca_aligned_gpu | tca_aligned]
     -> OUT/figures/aligned_gpu/<sid>/... (Kuma fits) or OUT/figures/aligned/<sid>/... (haas fits)
"""
import argparse
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import tca_lib as L  # noqa: E402

A = importlib.import_module("005_build_aligned")
G = importlib.import_module("006g_tca_gpu")        # prepare (incl. outlier handling), ranks, normalisations
P = importlib.import_module("004_pilot_figures")   # area_colours, shade_passive
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "font.size": 6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                     "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2,
                     "ytick.major.size": 2, "axes.labelpad": 2})
TT_COL = {"whisker_trial": "#f7b519", "auditory_trial": "#2c2cdb", "no_stim_trial": "#262626"}
TT_LAB = {"whisker_trial": "whisker", "auditory_trial": "auditory", "no_stim_trial": "no stim"}
EV_LAB = {"stim": "stimulus", "jaw": "jaw onset", "lick": "first lick"}
CLASSES = [("Whisker hit", "A", "whisker_trial", 1), ("Whisker miss", "A", "whisker_trial", 0),
           ("Auditory hit", "A", "auditory_trial", 1), ("Auditory miss", "A", "auditory_trial", 0),
           ("False alarm", "A", "no_stim_trial", 1), ("Correct rejection", "A", "no_stim_trial", 0),
           ("Whisker passive", "P", "whisker_trial", None), ("Auditory passive", "P", "auditory_trial", None)]
TCA_DIR = L.OUT / "artifacts" / "tca_aligned_gpu"
VARIANT = ""   # "" (signed CP), "_l2-<strength>" (ridge) or "_splitnn" (006g variants)
FIG = L.OUT / "figures" / "aligned_gpu"
DATA = A.OUT_DIR


def light(c, amount=0.6):
    c = np.array(mcolors.to_rgb(c))
    return tuple(c + (1 - c) * amount)


def trial_colour(tt_, passive):
    return light(TT_COL[tt_]) if passive else TT_COL[tt_]


def scatter_trials(ax, x, y, t, outlier=None):
    passive = t.epoch.str.startswith("passive").to_numpy()
    lick = t.lick_flag.fillna(0).to_numpy() == 1
    ok = np.ones(len(t), bool) if outlier is None else ~outlier
    for tt_ in TT_COL:
        for pv in (False, True):
            m = (t.trial_type == tt_).to_numpy() & (passive == pv) & ok
            c = trial_colour(tt_, pv)
            ax.scatter(x[m & lick], y[m & lick], s=3.5, color=c, lw=0, rasterized=False)
            ax.scatter(x[m & ~lick], y[m & ~lick], s=3.5, facecolor="none", edgecolor=c, lw=0.45)
    if outlier is not None and outlier.any():
        ax.scatter(x[outlier], y[outlier], s=6, marker="x", color="0.45", lw=0.6)


def trial_legend_handles(with_outlier):
    h = [Line2D([], [], ls="", marker="o", ms=2.5, color=TT_COL[k], label=f"{TT_LAB[k]}") for k in TT_COL]
    h += [Line2D([], [], ls="", marker="o", ms=2.5, color=light(TT_COL["whisker_trial"]), label="passive (light)"),
          Line2D([], [], ls="", marker="o", ms=2.5, color="k", label="lick (filled)"),
          Line2D([], [], ls="", marker="o", ms=2.5, mfc="none", mec="k", mew=0.5, label="no lick (open)")]
    if with_outlier:
        h.append(Line2D([], [], ls="", marker="x", ms=3, color="0.45", label="outlier, not fitted"))
    return h


def load(sid, typ):
    f = TCA_DIR / sid / f"{typ}{VARIANT}.npz"
    return dict(np.load(f, allow_pickle=True)) if f.exists() else None


def unit_factor(d, r):
    """(increase weights, decrease weights or None, signed effective weights) of the unit factor; split fits store
    increases in the first half of the rows and decreases in the second half"""
    Fu = d[f"r{r}_f1"]
    J = len(d["units_kept"])
    if Fu.shape[0] == 2 * J:
        return Fu[:J], Fu[J:], Fu[:J] - Fu[J:]
    return Fu, None, Fu


def shared_ylim(axes, vals, symmetric=False, pad=0.06):
    """one y-range for all panels of a column (factors are balanced across modes, so components compare directly)"""
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if symmetric:
        m = max(abs(lo), abs(hi))
        lo, hi = -m, m
    span = max(hi - lo, 1e-9)
    for a in axes:
        a.set_ylim(min(lo, 0) - pad * span, max(hi, 0) + pad * span)


def types():
    return [(a, n) for a in A.ALIGN for n in G.NORMS]


def ranks_in(d):
    return np.array(sorted(int(k[1:-4]) for k in d if k.startswith("r") and k.endswith("_obj")))


def rank_model(sid):
    """Williams et al. 2018 rank selection: error and restart similarity vs rank (+ gain per added component)."""
    ty = [t for t in types() if load(sid, f"{t[0]}_{t[1]}") is not None]
    fig, ax = plt.subplots(4, len(ty), figsize=(1.85 * len(ty) + 0.6, 5.6), squeeze=False)
    for j, (al, n) in enumerate(ty):
        d = load(sid, f"{al}_{n}")
        R = ranks_in(d)
        err = np.array([d[f"r{k}_obj"].min() for k in R])
        si = [d[f"r{k}_sim"] for k in R]
        rc = int(d["rank_chosen"])
        for k, o in zip(R, (d[f"r{k}_obj"] for k in R)):
            ax[0, j].scatter(np.full(len(o), k), o, s=1.5, color="0.65", lw=0)
        ax[0, j].plot(R, err, color="k", lw=0.9, label="TCA, best restart")
        ax[0, j].axhline(float(d["condmean_err"]), color="tab:red", lw=0.7, ls="--", label="condition-mean model")
        for k, x in zip(R, si):
            ax[1, j].scatter(np.full(len(x), k), x, s=1.5, color="0.65", lw=0)
        ax[1, j].plot(R, [x.mean() if len(x) else 1 for x in si], color="k", lw=0.9)
        ax[1, j].axhline(float(d["sim_min"]), color="tab:blue", lw=0.5, ls=":")
        ax[1, j].set_ylim(None, 1.02)
        ax[2, j].bar(R[1:], np.array(d["gain"])[1:], color="0.55", width=0.7)
        ax[2, j].axhline(float(d["gain_min"]), color="tab:blue", lw=0.5, ls=":")
        if f"r{R[1]}_mincong" in d:
            ax[3, j].plot(R[1:], [float(d[f"r{k}_mincong"]) for k in R[1:]], color="k", lw=0.9, marker="o", ms=1.5)
        ax[3, j].axhline(-0.5, color="tab:red", lw=0.5, ls=":")
        ax[3, j].set_ylim(-1.02, 1.02)
        for i in range(4):
            ax[i, j].axvline(rc, color="tab:blue", lw=0.6)
            ax[i, j].set_xticks([1, 4, 8, 12] if R.max() <= 12 else [1, 5, 10, 15, 20])
            ax[i, j].set_xlim(0.3, R.max() + 0.7)
        n_out = int(np.sum(d.get("outlier_trials", [])))
        ax[0, j].set_title(f"{al}, {n}\nchosen rank {rc}\n{d['shape'][0]} trials x {d['shape'][1]} units"
                           + (f"\n{n_out} outlier trials" if n_out else "\n"), fontsize=5.5)
        ax[3, j].set_xlabel("Rank")
        ax[3, j].set_xticks([1, 4, 8, 12] if R.max() <= 12 else [1, 5, 10, 15, 20])
        ax[3, j].set_xlim(0.3, R.max() + 0.7)
    for i, lab in enumerate(["Relative error", "Restart similarity", "Error decrease\n(fraction of total)",
                             "Most negative\ncomponent overlap"]):
        ax[i, 0].set_ylabel(lab)
    ax[0, 0].legend(frameon=False, fontsize=4.5, loc="upper right")
    fig.suptitle(f"{sid}{VARIANT}: rank selection (blue line = chosen; dotted = similarity 0.85 / decrease 5 %; "
                 f"overlap < -0.5 = cancelling pair)", fontsize=6.5,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.align_ylabels(ax[:, 0])
    fig.savefig(FIG / sid / f"rank_model_{sid}{VARIANT}.png", dpi=250)
    fig.savefig(FIG / sid / f"rank_model_{sid}{VARIANT}.pdf")
    plt.close(fig)


SPELL = {"Somatosensory-whisker": "Ss-whisker", "Somatosensory-orofacial": "Ss-orofacial",
         "Somatosensory-body": "Ss-body"}   # skills/ssl-figure-style rule 8


def area_axis(ax, grp_sorted, cols_sorted, min_frac=0.03):
    """group names (style spelling) under the unit-weight axis at the group centres; groups with < min_frac of the
    units keep their colour but get no name (avoids overlapping labels)"""
    g = np.asarray(grp_sorted)
    edges = np.flatnonzero(np.r_[True, g[1:] != g[:-1], True])
    ticks, labs, cols = [], [], []
    for a0, a1 in zip(edges[:-1], edges[1:]):
        if (a1 - a0) / len(g) >= min_frac:
            ticks.append((a0 + a1 - 1) / 2)
            labs.append(SPELL.get(g[a0], g[a0]))
            cols.append(cols_sorted[a0])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, rotation=60, ha="right", fontsize=4.5, rotation_mode="anchor")
    for lab, c in zip(ax.get_xticklabels(), cols):
        lab.set_color(c)
    return edges


def factors(sid, al, n, rank=None):
    d = load(sid, f"{al}_{n}")
    if d is None:
        return
    r = rank or int(d["rank_chosen"])
    if f"r{r}_f0" not in d:
        return
    Fa, Ft = d[f"r{r}_f0"], d[f"r{r}_f2"]
    Fu, Fd, _ = unit_factor(d, r)
    out = np.asarray(d.get("outlier_trials", np.zeros(len(Fa), bool)), bool)
    full = pd.read_parquet(DATA / sid / "trials.parquet")
    t = full.set_index("trial_index").loc[d["trial_index"]].reset_index()
    u = pd.read_parquet(DATA / sid / "units.parquet").iloc[d["units_kept"]].reset_index(drop=True)
    ucol, uord, grp = P.area_colours(u.area_acronym_custom.str.strip().tolist())
    o = np.lexsort((u.area_acronym_custom.to_numpy(), np.array(uord)))
    ucol_s, grp_s = np.array(ucol)[o], np.array(grp)[o]
    x = t.trial_index.to_numpy()
    xf = full.trial_index.to_numpy()
    tc = d["t_centres"] * 1e3
    ev = A.ALIGN[al]["event"]
    stim_ms = jaw_ms = None
    if ev in ("jaw", "lick"):
        e_abs = (t.jaw_time if ev == "jaw" else t.lick_abs).to_numpy()
        stim_ms = np.median(t.start_time.to_numpy() - e_abs) * 1e3
        if ev == "lick":
            jaw_ms = np.median(t.jaw_time.to_numpy() - e_abs) * 1e3
    row_h = 0.62
    fig = plt.figure(figsize=(7.4, 1.05 + row_h * r + 0.45))
    gs = fig.add_gridspec(r + 1, 3, width_ratios=[2.0, 1.25, 3.15], height_ratios=[1.25] + [1] * r,
                          hspace=0.32, wspace=0.28, left=0.08, right=0.985, top=0.965,
                          bottom=min(0.45 / (1.05 + row_h * r + 0.55) + 0.05, 0.16))
    # header row: info, legend, P(lick)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(0, 1, f"{sid}\n{t.cohort.iloc[0]}, whisker day 0\naligned to {EV_LAB[ev]}, {n}{VARIANT}\nrank {r}"
             f"{' (chosen)' if rank is None else ' (fixed)'}; {len(t)} trials x {len(u)} units"
             + (f"\n{out.sum()} outlier trials (x) not fitted" if out.any() else ""), fontsize=5.5, va="top")
    axl = fig.add_subplot(gs[0, 1])
    axl.axis("off")
    axl.legend(handles=trial_legend_handles(out.any()), frameon=False, fontsize=4.5, loc="upper left", ncol=2,
               handletextpad=0.2, borderaxespad=0, labelspacing=0.3, columnspacing=0.6)
    b = fig.add_subplot(gs[0, 2])
    for c, lab in (("p_wh_i", "whisker_trial"), ("p_au_i", "auditory_trial"), ("p_ns_i", "no_stim_trial")):
        if c in full:
            b.plot(xf, full[c], color=TT_COL[lab], lw=0.9, label=TT_LAB[lab] + (" (false alarm)" if
                                                                             lab == "no_stim_trial" else ""))
    P.shade_passive(b, full, xf)
    b.set_ylabel("P(lick)")
    b.yaxis.set_label_position("right")   # keeps the label clear of the trial legend on the left
    b.set_ylim(-0.02, 1.02)
    b.set_xlim(xf.min() - 2, xf.max() + 2)
    b.tick_params(labelbottom=False)
    b.legend(frameon=False, fontsize=4.5, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0,
             handlelength=1.2, columnspacing=0.8)
    axu = axt = axs = None
    col_axes = [[], [], []]
    for k in range(r):
        last = k == r - 1
        a = fig.add_subplot(gs[k + 1, 0], sharex=axu)
        axu = axu or a
        a.bar(np.arange(len(o)), Fu[o, k], color=ucol_s, width=1.0, lw=0)
        if Fd is not None:   # split fit: decreases drawn downwards
            a.bar(np.arange(len(o)), -Fd[o, k], color=ucol_s, width=1.0, lw=0, alpha=0.55)
        a.axhline(0, color="0.6", lw=0.3)
        a.set_xlim(-0.5, len(o) - 0.5)
        a.set_ylabel(f"C{k + 1}", rotation=0, ha="right", va="center", fontsize=6, fontweight="bold")
        if last:
            area_axis(a, grp_s, ucol_s)
        else:
            a.tick_params(labelbottom=False, bottom=False)
        col_axes[0].append(a)
        a = fig.add_subplot(gs[k + 1, 1], sharex=axt)
        axt = axt or a
        a.plot(tc, Ft[:, k], color="k", lw=0.8)
        a.axhline(0, color="0.6", lw=0.3)
        a.axvline(0, color="0.4", lw=0.4)
        if stim_ms is not None:
            a.axvline(stim_ms, color="tab:red", lw=0.5, ls=":")
        if jaw_ms is not None:
            a.axvline(jaw_ms, color="tab:green", lw=0.5, ls=":")
        if not last:
            a.tick_params(labelbottom=False)
        col_axes[1].append(a)
        a = fig.add_subplot(gs[k + 1, 2], sharex=b)
        scatter_trials(a, x, Fa[:, k], t, out)
        P.shade_passive(a, full, xf)
        a.axhline(0, color="0.6", lw=0.3)
        if not last:
            a.tick_params(labelbottom=False)
        col_axes[2].append(a)
    shared_ylim(col_axes[0], np.r_[Fu.ravel(), (-Fd).ravel() if Fd is not None else []], symmetric=Fd is None)
    shared_ylim(col_axes[1], Ft)
    shared_ylim(col_axes[2], Fa)
    col_axes[0][0].set_title("Unit weight" + (" (up: increase, down: decrease)" if Fd is not None else ""),
                             fontsize=6)
    col_axes[1][0].set_title("Time weight", fontsize=6)
    col_axes[2][0].set_title("Trial weight", fontsize=6)
    col_axes[1][-1].set_xlabel(f"Time from {EV_LAB[ev]} (ms)" + ("\nred: median stimulus" if stim_ms is not None
                                                               else "") + ("; green: median jaw" if jaw_ms is not None
                                                                           else ""))
    col_axes[2][-1].set_xlabel("Trial in session (passive blocks shaded)")
    fig.align_ylabels(col_axes[0])
    fig.align_ylabels(col_axes[1])
    fig.align_ylabels(col_axes[2])
    base = FIG / sid / f"factors_{sid}_{al}_{n}{VARIANT}_r{r}"
    fig.savefig(f"{base}.png", dpi=220)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)


def recon(sid, al, n, rank=None, n_units=4):
    d = load(sid, f"{al}_{n}")
    if d is None:
        return
    r = rank or int(d["rank_chosen"])
    if f"r{r}_f0" not in d:
        return
    bsub = bool(d.get("bsub", True))   # method 'nn' fits rates without baseline subtraction
    Y, obs, keep_u, cls, _, outlier = G.prepare(DATA, sid, al, n, bsub=bsub)
    Fa, Ft = d[f"r{r}_f0"], d[f"r{r}_f2"]
    _, _, Fu = unit_factor(d, r)   # split fits: increases minus decreases = signed prediction
    Yh = np.einsum("ir,jr,kr->ijk", Fa, Fu, Ft)
    ss = np.sum(Y[obs] ** 2)
    r2 = 1 - np.sum((Y - Yh)[obs] ** 2) / ss
    drop = []
    for k in range(r):   # loss of variance explained when component k alone is removed (% of data variance)
        Yk = Yh - np.einsum("i,j,k->ijk", Fa[:, k], Fu[:, k], Ft[:, k])
        drop.append(r2 - (1 - np.sum((Y - Yk)[obs] ** 2) / ss))
    # signed Tucker congruence between components (product of cosines over the 3 modes): near -1 = two components
    # cancelling each other (CP degeneracy)
    un = [f / np.linalg.norm(f, axis=0, keepdims=True) for f in (Fa, Fu, Ft)]
    Cg = (un[0].T @ un[0]) * (un[1].T @ un[1]) * (un[2].T @ un[2])
    tc = d["t_centres"] * 1e3
    ev = A.ALIGN[al]["event"]
    u = pd.read_parquet(DATA / sid / "units.parquet").iloc[d["units_kept"]].reset_index(drop=True)
    fit_ = ~outlier
    r2u = 1 - ((Y - Yh)[fit_] ** 2).sum((0, 2)) / np.maximum((Y[fit_] ** 2).sum((0, 2)), 1e-12)
    ex = list(np.argsort(r2u)[::-1][:n_units])   # example units: best fitted by the model
    key = {f"{c_}_{tt_}_{lk if lk is not None else 0}": i for i, (_, c_, tt_, lk) in enumerate(CLASSES)}
    ci = np.array([key.get(c, 99) for c in cls])
    srt = np.lexsort((d["trial_index"], ci))
    cs = ci[srt]
    bounds = np.flatnonzero(np.diff(cs)) + 0.5
    fig = plt.figure(figsize=(7.4, 1.75 + 1.3 * len(ex)))
    gs = fig.add_gridspec(1 + len(ex), 5, width_ratios=[1.9, 0.3, 0.06, 2.6, 0.05],
                          height_ratios=[1.15] + [1] * len(ex), hspace=0.5, wspace=0.12, left=0.085, right=0.95,
                          top=0.93, bottom=0.07)
    top = gs[0, :].subgridspec(1, 3, width_ratios=[1.3, 0.8, 1.2], wspace=0.45)
    a = fig.add_subplot(top[0])
    a.bar(np.arange(1, r + 1), np.array(drop) * 100, color="0.5", width=0.7)
    a.axhline(r2 * 100, color="tab:red", lw=0.6, ls="--")
    a.text(0.4, r2 * 100, "full model", color="tab:red", fontsize=4.5, va="bottom", ha="left")
    a.set_xticks(np.arange(1, r + 1))
    a.set_xticklabels([f"C{k}" for k in range(1, r + 1)])
    a.set_ylabel("Loss of variance explained\nwhen removed (% of data)")
    a = fig.add_subplot(top[1])
    a.imshow(Cg, cmap="RdBu_r", vmin=-1, vmax=1)
    for (i_, j_), v in np.ndenumerate(Cg):
        if i_ != j_ and abs(v) >= 0.3:
            a.text(j_, i_, f"{v:.1f}", ha="center", va="center", fontsize=3.8)
    a.set_xticks(range(r))
    a.set_xticklabels([f"C{k}" for k in range(1, r + 1)], fontsize=4.5)
    a.set_yticks(range(r))
    a.set_yticklabels([f"C{k}" for k in range(1, r + 1)], fontsize=4.5)
    a.set_title("Component overlap (congruence;\n< -0.5: cancelling pair)", fontsize=5)
    al_ = fig.add_subplot(top[2])
    al_.axis("off")
    hs = [Line2D([], [], color=TT_COL[tt_] if c_ == "A" else light(TT_COL[tt_]),
                 ls="-" if lk in (1, None) else "--", lw=0.9, label=lab) for lab, c_, tt_, lk in CLASSES]
    hs += [Line2D([], [], color="k", lw=0.9, label="data (line)"),
           Line2D([], [], color="k", lw=0, marker="o", ms=1.8, label="model (dots)")]
    al_.legend(handles=hs, frameon=False, fontsize=4.5, ncol=2, loc="center left", handlelength=1.6)
    fig.suptitle(f"{sid}, aligned to {EV_LAB[ev]}, {n}{VARIANT}, rank {r}: full model explains {r2 * 100:.1f} % of the "
                 f"{'baseline-subtracted' if bsub else 'rate (no baseline subtraction)'} variance (fitted trials)", fontsize=6,
                 x=0.01, ha="left")
    class_col = [(light(TT_COL[tt_]) if c_ == "P" else TT_COL[tt_]) for _, c_, tt_, _ in CLASSES]
    strip = np.array([mcolors.to_rgb(class_col[c]) if c < len(CLASSES) else (1, 1, 1) for c in cs])[:, None, :]
    hit = np.array([CLASSES[c][3] in (1, None) if c < len(CLASSES) else True for c in cs])
    strip[~hit] = 1 - 0.45 * (1 - strip[~hit])   # misses / correct rejections paler in the strip
    unit_lab = {"rate": "Hz", "zscore": "SD"}.get(n, "norm.")
    for i, j in enumerate(ex):
        a = fig.add_subplot(gs[1 + i, 0])
        for lab, c_, tt_, lk in CLASSES:
            m = np.array([c.startswith(c_ + "_" + tt_) and (lk is None or c.endswith(f"_{lk}")) for c in cls]) & fit_
            if m.sum() < 3:
                continue
            col = light(TT_COL[tt_]) if c_ == "P" else TT_COL[tt_]
            ls = "-" if (lk == 1 or lk is None) else "--"
            a.plot(tc, Y[m, j].mean(0), color=col, lw=0.8, ls=ls)
            a.plot(tc, Yh[m, j].mean(0), color=col, lw=0, marker="o", ms=1.6, alpha=0.8)
        a.axvline(0, color="0.4", lw=0.4)
        a.axhline(0, color="0.6", lw=0.3)
        a.set_ylabel(f"{u.area_acronym_custom.iloc[j].strip()} unit ({u.electrode_group.iloc[j]})\n"
                     f"{'change' if bsub else 'rate'} ({unit_lab}), R2 {r2u[j]:.2f}", fontsize=5)
        if i == len(ex) - 1:
            a.set_xlabel(f"Time from {EV_LAB[ev]} (ms)")
        v = max(np.nanpercentile(np.abs(Y[:, j]), 99), 1e-6)
        ext = [tc[0] - (tc[1] - tc[0]) / 2, tc[-1] + (tc[1] - tc[0]) / 2, len(srt), 0]
        sa = fig.add_subplot(gs[1 + i, 2])
        sa.imshow(strip, aspect="auto", interpolation="nearest", extent=[0, 1, len(srt), 0])
        sa.set_xticks([])
        mids = [(lo + hi) / 2 for lo, hi in zip(np.r_[0, bounds], np.r_[bounds, len(srt)])]
        sizes = np.diff(np.r_[0, bounds, len(srt)])
        short = ["W hit", "W miss", "A hit", "A miss", "FA", "CR", "W pass.", "A pass."]
        names = [short[c] if c < len(CLASSES) else "" for c in np.unique(cs)]
        keep = sizes >= 0.06 * len(srt)
        sa.set_yticks(np.array(mids)[keep])
        sa.set_yticklabels(np.array(names)[keep], fontsize=4.2)
        pair = gs[1 + i, 3].subgridspec(1, 2, wspace=0.04)   # data and model matrices side by side
        a_data = None
        for q, (M, lab) in enumerate(((Y[srt, j], "data"), (Yh[srt, j], "model"))):
            a2 = fig.add_subplot(pair[q], sharey=a_data if q else None)
            im = a2.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest", extent=ext)
            for y_ in bounds:
                a2.axhline(y_, color="k", lw=0.3)
            a2.axvline(0, color="k", lw=0.3, ls=":")
            a2.set_xticks([0, 500, 1000] if tc[-1] > 1000 else np.round(np.linspace(0, tc[-1] // 50 * 50, 3)))
            a2.set_yticks([])
            a_data = a_data or a2
            if i == 0:
                a2.set_title(f"Single trials, {lab}", fontsize=5.5)
            if i == len(ex) - 1:
                a2.set_xlabel(f"Time from {EV_LAB[ev]} (ms)")
        cb = fig.colorbar(im, cax=fig.add_subplot(gs[1 + i, 4]))
        cb.ax.tick_params(labelsize=4.5)
        cb.outline.set_linewidth(0.3)
    base = FIG / sid / f"recon_{sid}_{al}_{n}{VARIANT}_r{r}"
    fig.savefig(f"{base}.png", dpi=220)
    fig.savefig(f"{base}.pdf")
    plt.close(fig)


def main():
    global TCA_DIR, FIG, VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--rank", type=int, help="fixed rank for all types (default: each type's chosen rank)")
    ap.add_argument("--tca-dir", default="tca_aligned_gpu", help="tca_aligned_gpu (006g, Kuma) or tca_aligned (006)")
    ap.add_argument("--variant", default="", help='"" (signed CP), "_l2-100" (ridge), "_splitnn" (split non-negative)')
    ap.add_argument("--alignments", nargs="*", default=list(A.ALIGN))
    a = ap.parse_args()
    TCA_DIR = L.OUT / "artifacts" / a.tca_dir
    VARIANT = a.variant
    FIG = L.OUT / "figures" / ("aligned_gpu" if a.tca_dir.endswith("_gpu") else "aligned")
    for sid in a.sessions:
        (FIG / sid).mkdir(parents=True, exist_ok=True)
        rank_model(sid)
        for al, n in [t for t in types() if t[0] in a.alignments]:
            factors(sid, al, n, a.rank)
            recon(sid, al, n, a.rank)
        print("done", sid, flush=True)


if __name__ == "__main__":
    main()
