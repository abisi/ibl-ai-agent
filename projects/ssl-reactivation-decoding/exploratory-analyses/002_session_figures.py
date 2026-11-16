"""002 -- Pilot figures 1-4 for one session and target (outputs of 001; question.md figure list, user 2026-10-23).

fig1_template : per key area group: held-out ROC (low-FPR zoom), held-out readout around every event class (-200..600 ms) with the
                detection threshold, decoder weights, DLC whisker speed in the training windows per class (state check).
fig2_continuous: 60-s stretch of task trials with the most detections and a 5-s zoom on the strongest Ss-whisker event: raster of
                the Ss-whisker decoder units (sorted by weight), readout per group with threshold and events, stimuli / licks,
                whisker speed and jaw.
fig3_threshold : event rate vs threshold for the data and both session-own nulls (weight shuffle, linear time shift), per-group
                threshold (the same in every session and template); detections per minute at the threshold vs nulls.
fig4_sensitivity: fraction of events of each class whose readout crosses the threshold 10-100 ms after onset (held-out models
                for training events), and the detection latency of target stimuli.
Run (haas): ... ./.venv/bin/python .../002_session_figures.py --session MH030_20250503_154256 --target whisker
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
sys.path.insert(0, str(pathlib.Path.home() / "code" / "ephys_utilities"))

WH_C, AUD_C = "#f7b519", "#2c2cdb"
# event classes shown (key, label, colour, line style); task / passive split by context
CLASSES = [("W_hit", "whisker hit", WH_C, "-"), ("W_miss", "whisker miss", WH_C, "--"), ("W_pas", "whisker passive", "#c98a00", ":"),
           ("A_hit", "auditory hit", AUD_C, "-"), ("A_miss", "auditory miss", AUD_C, "--"), ("A_pas", "auditory passive", "#7f7fe8", ":"),
           ("N", "correct rejection", "0.35", "-"), ("FA", "false alarm", "0.0", "--"), ("sham", "passive sham", "0.6", ":")]


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 6,
                         "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5, "ytick.labelsize": 5.5,
                         "legend.fontsize": 5.2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5,
                         "lines.linewidth": 0.9, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": "tight"})
    return plt


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=250)


def short(g):
    return {"Somatosensory-whisker": "Ss-whisker", "Motor areas": "Motor", "Auditory areas": "Auditory"}.get(g, g)


def colors():
    from ephys_utilities.allen_utils import allen_utils as au
    return au.get_custom_area_groups_colors()


def ev_key(r):
    if r.cls in ("SW", "SA"):
        return "sham"
    if r.cls in ("W", "A"):
        return f"{r.cls}_pas" if r.context == "passive" else f"{r.cls}_{'hit' if r.licked else 'miss'}"
    return r.cls


def load(out):
    P = json.loads((out / "params.json").read_text())
    V = pd.read_csv(out / "validation.csv")
    pe = pd.read_parquet(out / "peri_event.parquet")
    pe["key"] = [ev_key(r) for r in pe.itertuples()]
    C = {g: dict(np.load(out / f"continuous_{g.replace(' ', '_')}.npz")) for g in V.group}
    for g, th in zip(V.group, V.thr):          # per-group threshold of apply_thresholds (001)
        C[g]["thr"] = th
    E = pd.read_parquet(out / "events.parquet")
    N = pd.read_parquet(out / "null_curves.parquet")
    T = pd.read_parquet(out / "task_events.parquet")
    B = dict(np.load(out / "behaviour.npz"))
    return P, V, pe, C, E, N, T, B


def interp(B, k, t):
    if f"{k}_t" not in B:
        return np.full(len(t), np.nan)
    return np.interp(t, B[f"{k}_t"], B[f"{k}_v"], left=np.nan, right=np.nan)


# ------------------------------------------------------------------------------------------------ figure 1
def fig1(plt, out, P, V, pe, C, B, target, col):
    from sklearn.metrics import roc_curve
    G = list(V.group)
    fig, axs = plt.subplots(len(G), 4, figsize=(7.4, 1.55 * len(G)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1, 2.2, 1.2, 1.1]))
    for i, g in enumerate(G):
        c, v = C[g], V[V.group == g].iloc[0]
        ax = axs[i, 0]
        fpr, tpr, _ = roc_curve(c["y"], c["oof"])
        ax.plot(fpr, tpr, color=col.get(g, "k")); ax.plot([0, 1], [0, 1], color="0.7", lw=0.4, ls=":")
        ax.axvline(m1.MAX_FPR, color="0.5", lw=0.4, ls="--")
        ax.set_xscale("symlog", linthresh=0.01); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_title(f"{short(g)}: pAUC {v.pauc:.2f}, AUC {v.auc:.2f}\nsens. @ FPR 1 % {v.sens_fpr01:.2f}, 5 % {v.sens_fpr05:.2f}",
                     fontsize=5.2, loc="left", color=col.get(g, "k"))
        ax.set_xlabel("false-positive rate (held out)"); ax.set_ylabel("hit rate")
        ax = axs[i, 1]
        q = pe[pe.group == g]
        for key, lab, cc, ls in CLASSES:
            rows = q[q.key == key].ev.to_numpy()
            if len(rows) < 3:
                continue
            R = c["peri"][rows]
            m, se = R.mean(0), R.std(0) / np.sqrt(len(rows))
            ax.plot(c["peri_t"] * 1000, m, color=cc, ls=ls, label=f"{lab} ({len(rows)})")
            ax.fill_between(c["peri_t"] * 1000, m - se, m + se, color=cc, alpha=0.12, lw=0)
        ax.axhline(float(c["thr"]), color="crimson", lw=0.6, ls="--", label="threshold")
        ax.axvspan(m1.TRAIN_WIN[0] * 1000 + m1.WIDTH * 1000 - 30, m1.TRAIN_WIN[1] * 1000, color="0.85", lw=0, zorder=0)
        ax.axvline(0, color="0.5", lw=0.4, ls=":")
        ax.set_xlabel("window end re. onset (ms; 30-ms windows)"); ax.set_ylabel("readout (decision value)")
        ax.set_title("readout around events (training events: held-out models); grey: training window", fontsize=5.2, loc="left")
        if i == 0:
            ax.legend(ncol=2, frameon=False, fontsize=4.4, loc="upper right")
        ax = axs[i, 2]
        w = np.sort(c["w"])[::-1]
        ax.bar(np.arange(len(w)), w, color=np.where(w > 0, col.get(g, "k"), "0.6"), width=0.9, lw=0)
        ax.axhline(0, color="0.4", lw=0.4)
        ax.set_xlabel("unit (sorted)"); ax.set_ylabel("weight (z-scored counts)")
        ax.set_title(f"weights, {int(v.n_units)} of {int(v.n_available)} units, C = {v.C:g}", fontsize=5.2, loc="left")
        ax = axs[i, 3]
        q = q[q.train]
        tt = q.t.to_numpy()
        sp = np.array([np.nanmean(interp(B, "whisker_speed", t + np.arange(-0.5, 0.0, 0.005))) for t in tt])
        sp2 = np.array([np.nanmean(interp(B, "whisker_speed", t + np.arange(m1.TRAIN_WIN[0], m1.TRAIN_WIN[1], 0.005))) for t in tt])
        for k, (vals, x0) in enumerate(((sp, 0), (sp2, 2))):
            for yv, dx, cc in ((1, 0, WH_C if target == "whisker" else AUD_C), (0, 0.8, "0.4")):
                d = vals[(q.y.to_numpy() == yv) & np.isfinite(vals)]
                if len(d):
                    ax.boxplot(d, positions=[x0 + dx], widths=0.6, showfliers=False, patch_artist=True,
                               boxprops=dict(facecolor=cc, alpha=0.5, lw=0.4), medianprops=dict(color="k", lw=0.6),
                               whiskerprops=dict(lw=0.4), capprops=dict(lw=0.4))
        ax.set_xticks([0.4, 2.4], ["-500..0 ms", "training\nwindow"])
        ax.set_ylabel("whisker speed (°/s, DLC)")
        ax.set_title("state check: stimulus (colour) vs no stimulus (grey)", fontsize=5.2, loc="left")
    fig.suptitle(f"{P['session_id']} ({P['model']}): {target} vs no stimulus template per area group ({P['n_units']} units, {int(1000 * m1.TRAIN_WIN[0])}-"
                 f"{int(1000 * m1.TRAIN_WIN[1])} ms, quiet events; training: {P['n_train']})", x=0.01, ha="left", fontsize=6.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    save(fig, out / "figures", "fig1_template")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 2
def fig2(plt, out, P, V, C, E, T, B, target, col, units_spikes):
    G = list(V.group)
    g0 = "Somatosensory-whisker" if "Somatosensory-whisker" in G else G[0]
    task = T[T.context == "task"]
    lo, hi = task.t.min(), task.t.max()
    best, bt = -1, lo
    for s in np.arange(lo, hi - 60, 5):
        n = ((E.t_peak >= s) & (E.t_peak < s + 60)).sum()
        if n > best:
            best, bt = n, s
    e0 = E[(E.group == g0) & (E.t_peak >= bt) & (E.t_peak < bt + 60)]
    zc = e0.sort_values("peak").t_peak.iloc[-1] if len(e0) else bt + 30
    wins = [(bt, bt + 60), (zc - 2.5, zc + 2.5)]
    nr = len(G) + 4                                     # raster, one row per group, trials + licks, whisker speed, jaw
    fig, axs = plt.subplots(nr, 2, figsize=(7.4, 0.62 * nr + 1.2), sharex="col", squeeze=False,
                            gridspec_kw=dict(height_ratios=[2.2] + [1] * len(G) + [0.7, 1, 1], width_ratios=[2.2, 1]))
    c0 = C[g0]
    order = np.argsort(c0["w"])[::-1]
    for j, (a, b) in enumerate(wins):
        ax = axs[0, j]
        for k, u in enumerate(order):
            st = units_spikes[int(c0["unit_index"][u])]
            st = st[(st >= a) & (st < b)]
            ax.vlines(st, k, k + 0.8, color=col.get(g0, "k") if c0["w"][u] > 0 else "0.55", lw=0.35 if j == 0 else 0.6)
        ax.set_ylim(0, len(order)); ax.set_yticks([])
        ax.set_ylabel(f"{short(g0)} units\n(sorted by weight)" if j == 0 else "")
        for r, g in enumerate(G, start=1):
            ax = axs[r, j]
            c = C[g]
            m = (c["t"] >= a) & (c["t"] < b)
            tr = np.where(c["valid"][m], c["trace"][m], np.nan)
            ax.plot(c["t"][m], tr, color=col.get(g, "k"), lw=0.4 if j == 0 else 0.7)
            ax.plot(c["t"][m], np.where(c["valid"][m], np.nan, c["trace"][m]), color="0.8", lw=0.4)
            ax.axhline(float(c["thr"]), color="crimson", lw=0.5, ls="--")
            ee = E[(E.group == g) & (E.t_peak >= a) & (E.t_peak < b)]
            ax.plot(ee.t_peak, ee.peak, "v", color="crimson", ms=2.5 if j == 0 else 3.5)
            ax.set_ylabel(short(g), rotation=0, ha="right", va="center", color=col.get(g, "k"))
        ax = axs[len(G) + 1, j]
        tt = T[(T.t >= a) & (T.t < b)]
        for cl, cc in (("W", WH_C), ("A", AUD_C), ("N", "0.35"), ("FA", "k")):
            ax.vlines(tt[tt.cls == cl].t, 1, 2, color=cc, lw=1.0)
        lk = B["licks"][(B["licks"] >= a) & (B["licks"] < b)]
        ax.vlines(lk, 0, 0.8, color="k", lw=0.4)
        ax.set_ylim(0, 2.1); ax.set_yticks([0.4, 1.5], ["licks", "trials"])
        for r, (k, lab) in enumerate((("whisker_speed", "whisker\nspeed (°/s)"), ("jaw", "jaw (Δpx)")), start=len(G) + 2):
            ax = axs[r, j]
            if f"{k}_t" in B:
                m = (B[f"{k}_t"] >= a) & (B[f"{k}_t"] < b)
                ax.plot(B[f"{k}_t"][m], B[f"{k}_v"][m], color="0.3", lw=0.4)
            ax.set_ylabel(lab if j == 0 else "")
        axs[-1, j].set_xlabel("session time (s)")
        axs[0, j].set_title("60 s of task trials with the most detections" if j == 0 else "5-s zoom: strongest Ss-whisker detection",
                            loc="left", fontsize=5.6)
    fig.suptitle(f"{P['session_id']}: continuous {target} readout (30-ms windows, 5-ms steps); red: threshold and detections; "
                 f"grey trace: excluded around trial onsets (-100..600 ms); trials: whisker (orange), auditory (blue), CR (grey), FA (black)",
                 x=0.01, ha="left", fontsize=5.8, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out / "figures", "fig2_continuous")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 3
def fig3(plt, out, P, V, N, target, col):
    G = list(V.group)
    fig, axs = plt.subplots(2, len(G), figsize=(7.4, 3.6), squeeze=False)
    for i, g in enumerate(G):
        q, v = N[N.group == g], V[V.group == g].iloc[0]
        ax = axs[0, i]
        ax.plot(q.thr, q.real, color=col.get(g, "k"), label="data")
        ax.plot(q.thr, q.null_weight, color="0.3", ls="--", label="weight shuffle (mean)")
        ax.fill_between(q.thr, 0, q.null_weight_hi, color="0.6", alpha=0.25, lw=0, label="weight shuffle 95 %")
        ax.plot(q.thr, q.null_shift, color="0.3", ls=":", label="time shift (mean)")
        ax.axhline(m1.NULL_RATE, color="0.5", lw=0.4)
        ax.axvline(v.thr, color="crimson", lw=0.6, ls="--")
        ax.set_yscale("log"); ax.set_ylim(0.01, None)
        ax.set_xlim(q.thr.min(), q.thr.max())
        ax.set_title(short(g), color=col.get(g, "k"), loc="left")
        ax.set_xlabel("threshold (readout)"); ax.set_ylabel("events / min" if i == 0 else "")
        if i == 0:
            ax.legend(frameon=False, fontsize=4.4)
    ax = plt.subplot2grid((2, len(G)), (1, 0), colspan=len(G), fig=fig)
    for a in axs[1]:
        a.remove()
    x = np.arange(len(G))
    ax.bar(x - 0.25, V.rate_real, 0.25, color=[col.get(g, "k") for g in G], label="data")
    ax.bar(x, V.rate_null_weight, 0.25, color="0.55", label="weight shuffle")
    ax.bar(x + 0.25, V.rate_null_shift, 0.25, color="0.8", label="time shift")
    for xi, r in zip(x, V.itertuples()):
        ax.text(xi - 0.25, r.rate_real * 1.05, f"{r.rate_real:.1f}\n×{r.rate_real / max(r.rate_null_shift, 1e-3):.1f}",
                ha="center", va="bottom", fontsize=4.6)
    ax.set_xticks(x, [short(g) for g in G]); ax.set_ylabel("detections / min at the threshold")
    ax.set_title(f"per-group threshold, the same in every session ({V.thr_rule.iloc[0]}; valid time {P['valid_min']:.0f} min); "
                 "text: rate and ratio to the time-shift null", loc="left", fontsize=5.4)
    ax.legend(frameon=False)
    fig.suptitle(f"{P['session_id']} ({P['model']} template): {target} event rate vs threshold, data and session-own nulls ({m1.N_NULL} weight permutations, "
                 f"{m1.N_NULL} linear time shifts of {m1.SHIFT_S[0]:.0f}-{m1.SHIFT_S[1]:.0f} s)", x=0.01, ha="left", fontsize=6.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out / "figures", "fig3_threshold")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ figure 4
def fig4(plt, out, P, V, pe, target, col):
    G = list(V.group)
    keys = [k for k in CLASSES if (pe.key == k[0]).any()]
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 2.6), gridspec_kw=dict(width_ratios=[2.4, 1]))
    ax = axs[0]
    wd = 0.8 / len(G)
    for i, g in enumerate(G):
        q = pe[pe.group == g]
        fr = [q[q.key == k].detected.mean() for k, *_ in keys]
        ax.bar(np.arange(len(keys)) + (i - len(G) / 2 + 0.5) * wd, fr, wd, color=col.get(g, "k"), label=short(g))
    ax.set_xticks(np.arange(len(keys)), [f"{lab}\n(n = {(pe[pe.group == G[0]].key == k).sum()})" for k, lab, *_ in keys], fontsize=4.8)
    ax.set_ylabel("fraction detected (10-100 ms)"); ax.set_ylim(0, 1)
    ax.set_title("detections at the threshold per event class (training events: held-out models)", loc="left", fontsize=5.6)
    ax.legend(frameon=False, ncol=3)
    ax = axs[1]
    stim = {"whisker": ("W_hit", "W_miss", "W_pas"), "auditory": ("A_hit", "A_miss", "A_pas")}[target]
    for g in G:
        q = pe[(pe.group == g) & pe.key.isin(stim) & pe.detected]
        if len(q) > 3:
            v = np.sort(q.latency * 1000)
            ax.step(v, np.arange(1, len(v) + 1) / len(v), color=col.get(g, "k"), where="post", label=short(g))
    ax.set_xlabel("detection latency (window end, ms)"); ax.set_ylabel("cumulative fraction")
    ax.set_title(f"latency of detected {target} stimuli", loc="left", fontsize=5.6)
    fig.suptitle(f"{P['session_id']}: sensitivity and cross-talk of the {target} detector", x=0.01, ha="left", fontsize=6.2, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out / "figures", "fig4_sensitivity")
    plt.close(fig)


def main(sid, target, model):
    out = m1.HOME / "pilot" / sid / target / model
    plt = setup()
    col = colors()
    P, V, pe, C, E, N, T, B = load(out)
    fig1(plt, out, P, V, pe, C, B, target, col)
    fig3(plt, out, P, V, N, target, col)
    fig4(plt, out, P, V, pe, target, col)
    units, *_ = m1.load_session(sid)
    fig2(plt, out, P, V, C, E, T, B, target, col, [np.asarray(s) for s in units.spike_times])
    print("FIGS DONE", out / "figures")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--target", default="whisker")
    ap.add_argument("--model", default="clean")
    a = ap.parse_args()
    main(a.session, a.target, a.model)
