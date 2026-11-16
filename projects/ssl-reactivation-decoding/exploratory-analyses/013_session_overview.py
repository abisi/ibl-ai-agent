"""013 -- Two A4 pages per session x target: the whole reactivation pipeline at a glance (user 2026-10-30; v7 2026-11-01:
decoder and correlation readouts side by side, much more continuous data).

Data: 009 v7 saved files (stable units, whole-brain "All"; readouts: decoder = decision value, correlation = Pearson correlation
across units with the evoked template; each with its own threshold = median over 100 template shuffles of the 99.9th
percentile), spikes from 006 prepare (artefact-corrected, as 009).
Page 1:
  A  session timeline (minutes): epochs, valid time (outside onset - 10 .. + 35 ms windows), still, evoked tail; trial onsets by
     class; decoder and correlation readouts (max per 1-s bin, raw units: decision value, r) with thresholds and events (triangles: still, ticks:
     other); whisker speed; lick rate; shaded: the stretches of page 2.
  B  training: training set, out-of-fold decision values (nested CV), decoder weights sorted (colour = area group), evoked
     template vs decoder weight per unit, |weight| vs rate with the shuffle strata.
  C  readouts around onsets (mean +- sem; decoder top row, correlation bottom row): task trials, passive pre and post (in
     training); passive ROC at the training window; passive peaks (10-100 ms) with the threshold.
  E  statistics from the 009 counts: rate vs threshold (real vs 100 shuffles, all valid and still) per readout; rates at the
     threshold by mask; shuffle distributions with p; whisker speed around event peaks.
  F  this session among the cohort (same target and stage): rates and excess per readout; this session ringed.
Page 2 (continuous data): one 3-min stretch (readouts, population activity, behaviour) and STRETCHES stretches of 30 s
  (raster of the top N_TOP units by |weight|, decoder and correlation readouts with the 2.5-97.5 % band of N_ZSHUF template
  shuffles drawn here, thresholds, events, population activity = mean z-scored count across units, whisker / jaw speed,
  licks, stimulus onsets; grey = invalid, gold = still, blue = evoked tail).
Output: figures/full_<unit_set>/overview/<session>_<target>.pdf (2 pages) + _p1.png / _p2.png. Sessions: --sessions, default
010 example sessions.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m6 = importlib.import_module("006_fdr_threshold_heldout_controls")
m7 = importlib.import_module("007_content_threshold_participation")
m9 = importlib.import_module("009_full_dataset_events")
m10 = importlib.import_module("010_quiet_window_examples")
GROUP = m9.m8.ALL
N_TOP = 50
N_ZSHUF = 20                                     # template shuffles drawn for the bands of page 2
LONG_S, SHORT_S, STRETCHES = 180.0, 30.0, 5
PERI = (-0.3, 0.6)                               # s around onsets (row C)
WPOS, WNEG = "#b2182b", "#2166ac"
COH = m10.COH
RD = {"": dict(name="decoder", col="k", band="0.7", unit="decision value"),
      "_corr": dict(name="correlation", col="#8c510a", band="#dfc27d", unit="r"),
      "_pcorr": dict(name="partial corr. | whisker vel.", col="#e7298a", band="#fbb4d9", unit="r")}   # raw units (009 v7); soft-normalised r


def clip_hi(C, x):
    """upper axis limit: stimulus responses clip (median + 2.5 x (threshold - median), or the 99.5th percentile if lower)"""
    md, thr = float(C["med"]), float(C["thr"])
    return min(np.nanpercentile(x, 99.5), md + 2.5 * (thr - md))


def qi(G, q=99.9):
    return [float(x) for x in G["null_qs"]].index(q)
CLS_COL = {"W": "#e6550d", "A": "#3182bd", "N": "0.35", "FA": "0.65", "SW": "0.6", "SA": "0.6"}
EPO_COL = {"passive_pre": "#c6dbef", "active": "#fdd0a2", "passive_post": "#9ecae1"}
SHADE = (("inv", "0.85", 1.0), ("evoked", "#c6dbef", 0.6), ("still", "gold", 0.3))


def runs(mask):
    d = np.diff(np.r_[0, np.asarray(mask, int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def binned(t, v, edges, fn):
    i = np.searchsorted(edges, t) - 1
    ok = (i >= 0) & (i < len(edges) - 1) & np.isfinite(v)
    out = np.full(len(edges) - 1, np.nan)
    s = pd.Series(v[ok]).groupby(i[ok]).agg(fn)
    out[s.index.to_numpy()] = s.to_numpy()
    return out


def load(sid, target, unit_set):
    out = m9.ROOT / unit_set / sid
    g = GROUP.replace(" ", "_")
    S = dict(np.load(out / "session.npz"))
    B = dict(np.load(out / "behaviour.npz", allow_pickle=True))
    Pm = json.loads((out / "params.json").read_text())
    pk = pd.read_parquet(out / f"peaks_{g}.parquet")
    Dd = m9.decoders_table(sid, unit_set)
    R = {}
    for sfx in RD:
        if not (out / f"g_{g}_{target}{sfx}.npz").exists():                  # e.g. no partial correlation (no movement data)
            continue
        G = dict(np.load(out / f"g_{g}_{target}{sfx}.npz", allow_pickle=True))
        C = dict(np.load(out / f"continuous_{target}_{g}{sfx}.npz", allow_pickle=True))
        E = pd.read_parquet(out / f"events_{target}_{g}{sfx}.parquet")
        d = Dd[(Dd.group == GROUP) & (Dd.target == target) & (Dd.thr_source == m9.PREFIX[sfx] + m9.THR_MAIN)].iloc[0]
        R[sfx] = dict(G=G, C=C, E=E, d=d)
    return dict(S=S, B=B, P=Pm, pk=pk, R=R, t=S["t"])


def label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=7, weight="bold", ha="right", va="bottom")


def speed(B, key):
    if f"{key}_t" not in B:
        return None, None
    t, v = B[f"{key}_t"], B[f"{key}_v"].astype(float)
    return (t, np.abs(np.gradient(v, t))) if key == "jaw" else (t, v)


# ------------------------------------------------------------------------------------------------------------- row A
def row_timeline(sf, Q, D, ev, zooms):
    S, B, t = Q["S"], Q["B"], Q["t"]
    valid = S["valid"]
    tm = lambda x: np.asarray(x) / 60
    nr = len(Q["R"])
    axs = sf.subplots(4 + nr, 1, sharex=True, gridspec_kw=dict(height_ratios=[0.9, 1.2] + [1.4] * nr + [0.6, 0.5]))
    a0, a1 = axs[0], axs[1]
    for nm, lab in (("active", "task"), ("passive_pre", "passive pre"), ("passive_post", "passive post")):
        for a, b in D["epochs"].get(nm, []):
            a0.broken_barh([(a / 60, (b - a) / 60)], (3, 0.8), color=EPO_COL[nm])
            a0.text((a + b) / 120, 3.4, lab, fontsize=4.5, ha="center", va="center")
    for y, mk, col in ((2, valid, "k"), (1, S["still"] & valid, "gold"), (0, S.get("evoked", np.zeros_like(valid)) & valid, "#6baed6")):
        st, en = runs(mk)
        a0.broken_barh(list(zip(t[st] / 60, (t[np.minimum(en, len(t) - 1)] - t[st]) / 60)), (y, 0.8), color=col, lw=0)
    a0.set_yticks([0.4, 1.4, 2.4, 3.4], ["evoked", "still", "valid", "epoch"], fontsize=4.5)
    a0.set_ylim(-0.1, 3.9)
    rows = [("task", "W"), ("task", "A"), ("task", "N"), ("task", "FA"), ("passive", "W"), ("passive", "A")]
    for k, (cx, cl) in enumerate(rows):
        a1.vlines(tm(ev.t[(ev.context == cx) & (ev.cls == cl)].to_numpy()), k, k + 0.8, color=CLS_COL[cl], lw=0.3)
    a1.set_yticks(np.arange(len(rows)) + 0.4, [f"{'task' if c == 'task' else 'pas.'} {l}" for c, l in rows], fontsize=3.6)
    a1.set_ylim(len(rows), 0)
    edges = np.arange(t[0], t[-1] + 1.0, 1.0)
    for ax, sfx in zip(axs[2:2 + nr], Q["R"]):
        C, E = Q["R"][sfx]["C"], Q["R"][sfx]["E"]
        zb = binned(t, np.where(valid, C["z"], np.nan), edges, "max")
        ax.plot(tm(edges[:-1]), zb, color=RD[sfx]["col"], lw=0.3)
        ax.axhline(float(C["thr"]), color="r", lw=0.5, ls="--")
        top = clip_hi(C, zb[np.isfinite(zb)]) if np.isfinite(zb).any() else 1
        bot = np.nanpercentile(zb, 0.5) if np.isfinite(zb).any() else 0
        top = top + 0.1 * (top - bot)
        st_ = E.still.to_numpy(bool)
        ax.vlines(tm(E.t_peak[~st_]), top - 0.08 * (top - bot), top, color="0.6", lw=0.3)
        ax.scatter(tm(E.t_peak[st_]), np.full(st_.sum(), top + 0.06 * (top - bot)), marker="v", s=6, lw=0, color=RD[sfx]["col"], zorder=4, clip_on=False)
        ax.set_ylim(bot - 0.05 * (top - bot), top + 0.12 * (top - bot))
        ax.set_ylabel(f"{RD[sfx]['name']}\nmax / s ({RD[sfx]['unit']})", fontsize=4.5)
        for lo, hi in zooms:
            ax.axvspan(lo / 60, hi / 60, color="tab:cyan", alpha=0.25, lw=0)
    tv, vv = speed(B, "whisker_speed")
    if tv is not None:
        axs[-2].plot(tm(edges[:-1]), binned(tv, vv, edges, "mean"), color="0.3", lw=0.3)
    axs[-2].set_ylabel("whisk\n(deg/s)", fontsize=4.5)
    e10 = np.arange(t[0], t[-1] + 10, 10.0)
    axs[-1].plot(tm(e10[:-1]), np.histogram(B["licks"], e10)[0] / 10, color="tab:orange", lw=0.4)
    axs[-1].set_ylabel("licks/s", fontsize=4.5)
    axs[-1].set_xlabel("time (min)", fontsize=5)
    axs[-1].set_xlim(t[0] / 60, t[-1] / 60)
    for a in axs:
        a.tick_params(labelsize=4.2)
    label(a0, "A")
    a0.set_title("session timeline: epochs, valid / still / evoked-tail time; trial onsets; decoder, correlation and partial correlation (| whisker velocity) readouts (max per s, "
                 "valid time) with thresholds (triangles: still events, grey: other events; cyan: stretches of page 2); whisker speed; lick rate",
                 fontsize=5, loc="left")


# ------------------------------------------------------------------------------------------------------------- row B
def row_training(sf, Q, ev, target, units_sel):
    G, Gr = Q["R"][""]["G"], Q["R"]["_corr"]["G"]
    axs = sf.subplots(1, 5, gridspec_kw=dict(width_ratios=[0.8, 1, 1.3, 1, 1]))
    stim = m1.TARGET_CLS[target][0]
    tmask = m9.train_mask(ev, target)
    task = ev[ev.context == "task"]
    n_all = {c: int((task.cls == c).sum()) for c in (stim, "N")}
    n_tr = {c: int(((ev.cls == c) & tmask & (ev.context == "task")).sum()) for c in (stim, "N")}
    n_pas = {e: int(((ev.cls == stim) & tmask & (ev.epoch == f"passive_{e}")).sum()) for e in ("pre", "post")}
    n_sh = {e: int(((ev.cls == m1.TARGET_CLS[target][1]) & tmask & (ev.epoch == f"passive_{e}")).sum()) for e in ("pre", "post")}
    ax = axs[0]
    ax.axis("off")
    txt = (f"training: {target} vs no stimulus\ntask: {stim} vs correct rejections\npassive pre / post: {stim} vs shams\n"
           f"(kept active trials: perf != 6, warm-up\ncut, rule A1); classes balanced\nwithin context\n\n{stim}: task {n_tr[stim]}, "
           f"passive {n_pas['pre']} + {n_pas['post']}\nN: {n_tr['N']} / {n_all['N']}; shams {n_sh['pre']} + {n_sh['post']}\n\nfeatures: counts "
           f"{m1.TRAIN_WIN[0] * 1000:.0f}-{m1.TRAIN_WIN[1] * 1000:.0f} ms, z-scored on\ncontinuous valid time; {int(G['n_units'])} units\n"
           f"decoder: L2 logistic, C = {float(G['C']):.3g}\ntemplate: mean z of {stim} - mean z\nof no-stimulus events")
    ax.text(0, 1, txt, fontsize=4.4, va="top", transform=ax.transAxes)
    label(ax, "B")
    ax = axs[1]
    y, oof = G["y"].astype(int), G["oof"]
    bins = np.linspace(np.nanmin(oof), np.nanmax(oof), 30)
    for v, cl, lab in ((1, CLS_COL[stim], f"{stim} (n {int((y == 1).sum())})"), (0, CLS_COL["N"], f"no stim. (n {int((y == 0).sum())})")):
        ax.hist(oof[y == v], bins, color=cl, alpha=0.6, label=lab)
    ax.set_title(f"decoder, out-of-fold (nested CV)\nbalanced acc. {m9.bal_acc(G['y'], G['oof']):.2f}, pAUC {float(G['pauc']):.2f}", fontsize=5)
    ax.set_xlabel("decision value (log-odds)", fontsize=4.5)
    ax.legend(frameon=False, fontsize=4)
    w, tmpl = G["w"], Gr["template"]
    ag = units_sel.area_group.astype(str).to_numpy()
    try:
        col = m2.colors()
    except Exception:
        col = {}
    cols = [col.get(a, "0.5") for a in ag]
    ax = axs[2]
    o = np.argsort(w)[::-1]
    ax.bar(np.arange(len(w)), w[o], width=1.0, color=[cols[i] for i in o], lw=0)
    ax.axhline(0, color="k", lw=0.3)
    ax.set_title("decoder weights (z-scored units), sorted; colour = area group", fontsize=5)
    ax.set_xlabel("unit rank", fontsize=4.5)
    gs = pd.Series(np.abs(w)).groupby(ag).sum().sort_values(ascending=False)
    ax.text(0.99, 0.98, "sum |w| by area group\n" + "\n".join(f"{m2.short(k)} {v / np.abs(w).sum():.0%}" for k, v in gs.head(6).items()),
            transform=ax.transAxes, ha="right", va="top", fontsize=3.8)
    ax = axs[3]
    ax.scatter(tmpl, w, s=1.5, lw=0, c=cols)
    ax.axhline(0, color="0.7", lw=0.3)
    ax.axvline(0, color="0.7", lw=0.3)
    rr = np.corrcoef(tmpl, w)[0, 1] if np.std(tmpl) > 0 and np.std(w) > 0 else np.nan
    ax.set_title(f"evoked template vs decoder weight\n(per unit; r = {rr:.2f})", fontsize=5)
    ax.set_xlabel("template (z, stimulus - no stimulus)", fontsize=4.5)
    ax.set_ylabel("decoder weight", fontsize=4.5)
    ax = axs[4]
    mu = G["mu"]
    st = m9.rate_strata(mu, m9.SHUF_STRATA)
    ax.scatter(mu / m1.WIDTH, np.abs(w), s=1.5, lw=0, c=[f"C{k}" for k in st])
    for k in range(1, m9.SHUF_STRATA):
        ax.axvline(mu[st == k].min() / m1.WIDTH, color="0.6", lw=0.3, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("continuous mean rate (Hz, valid time)", fontsize=4.5)
    ax.set_ylabel("|weight|", fontsize=4.5)
    ax.set_title(f"shuffles: weights / template permuted\nwithin {m9.SHUF_STRATA} rate quantiles (colours)", fontsize=5)
    for a in axs[1:]:
        a.tick_params(labelsize=4.2)


# ------------------------------------------------------------------------------------------------------------- row C
def aligned(z, t, ts, lags):
    if not len(ts):
        return np.zeros((0, len(lags)))
    return z[np.clip(np.searchsorted(t, np.asarray(ts)[:, None] + lags[None, :] - 1e-6), 0, len(t) - 1)]


def onsets_row(axs, Q, ev, target, sfx, first):
    from sklearn.metrics import roc_curve
    C, pk, t = Q["R"][sfx]["C"], Q["pk"], Q["t"]
    z, thr = C["z"], float(C["thr"])
    stim, sham = m1.TARGET_CLS[target]
    other = "A" if stim == "W" else "W"
    lags = np.round(np.arange(PERI[0], PERI[1] + 1e-9, m1.STEP), 6)
    sets = [("active task (in training)", ev.context == "task",
             [("W", ev.licked, "W hit"), ("W", ~ev.licked, "W miss"), ("A", None, "A"), ("N", None, "N"), ("FA", None, "FA")]),
            ("passive pre (in training)", ev.epoch == "passive_pre", [(stim, None, f"{stim} (target)"), (other, None, other), (sham, None, "sham")]),
            ("passive post (in training)", ev.epoch == "passive_post", [(stim, None, f"{stim} (target)"), (other, None, other), (sham, None, "sham")])]
    lo, hi = np.inf, -np.inf
    for ax, (ttl, m, cls) in zip(axs[:3], sets):
        for cl, extra, lab in cls:
            k = m & (ev.cls == cl)
            if extra is not None:
                k &= extra
            X = aligned(z, t, ev.t[k].to_numpy(), lags)
            if not len(X):
                continue
            mu, se = X.mean(0), X.std(0) / np.sqrt(len(X))
            ls = "--" if cl in ("SW", "SA") else (":" if lab.endswith("miss") else "-")
            ax.plot(lags * 1000, mu, color=CLS_COL[cl], ls=ls, lw=0.7, label=f"{lab} ({len(X)})")
            ax.fill_between(lags * 1000, mu - se, mu + se, color=CLS_COL[cl], alpha=0.2, lw=0)
            lo, hi = min(lo, np.nanmin(mu - se)), max(hi, np.nanmax(mu + se))
        ax.axhline(thr, color="r", lw=0.5, ls="--")
        ax.axvspan(m1.TRAIN_WIN[1] * 1000 - 0.5, m1.TRAIN_WIN[1] * 1000 + 0.5, color="k", alpha=0.3, lw=0)
        ax.axvspan(m1.EXCL[0] * 1000, m1.EXCL[1] * 1000, color="0.9", lw=0, zorder=0)
        ax.axvline(0, color="0.6", lw=0.3, ls=":")
        if first:
            ax.set_title(ttl, fontsize=5)
        ax.legend(frameon=False, fontsize=3.6, loc="upper right")
        ax.set_xlabel("window end from onset (ms)", fontsize=4.5)
    lo = min(lo, float(np.nanmedian(z)))
    span = max(hi, thr) - lo
    for ax in axs[:3]:
        ax.set_ylim(lo - 0.05 * span, max(hi, thr) + 0.1 * span)
    axs[0].set_ylabel(f"{RD[sfx]['name']} ({RD[sfx]['unit']})\nmean +- sem", fontsize=4.5)
    win = m9.win_values(z, t, pk.t.to_numpy())
    ax = axs[3]
    for ep, ls in (("passive_pre", "-"), ("passive_post", "--")):
        for neg, cl in ((sham, "0.4"), (other, CLS_COL[other])):
            a = win[(pk.epoch == ep) & (pk.cls == stim)]
            b = win[(pk.epoch == ep) & (pk.cls == neg)]
            if len(a) > 1 and len(b) > 1:
                fpr, tpr, _ = roc_curve(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
                ax.plot(fpr, tpr, color=cl, ls=ls, lw=0.7, label=f"{ep[8:]} vs {'sham' if neg == sham else other}: {m9.auc(a, b):.2f}")
    ax.plot([0, 1], [0, 1], color="0.8", lw=0.3)
    if first:
        ax.set_title("passive ROC, in training\n(readout at the 5-35 ms window)", fontsize=5)
    ax.set_xlabel("false positive rate", fontsize=4.5)
    ax.legend(frameon=False, fontsize=3.6, loc="lower right")
    ax = axs[4]
    col = f"peak_{target}{sfx}"
    for k, (ep, cl) in enumerate([(e, c) for e in ("passive_pre", "passive_post") for c in (stim, other, sham)]):
        v = pk.loc[(pk.epoch == ep) & (pk.cls == cl), col].to_numpy() if col in pk else np.zeros(0)
        x = k + np.random.default_rng(k).uniform(-0.25, 0.25, len(v))
        ax.scatter(x, v, s=1.5, lw=0, color=CLS_COL[cl])
        if len(v):
            ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color="k", lw=0.7)
            ax.text(k, np.nanmax(v), f"{np.mean(v > thr):.0%}", fontsize=3.6, ha="center", va="bottom")
    ax.axhline(thr, color="r", lw=0.5, ls="--")
    ax.set_xticks(range(6), [f"{e[8:]}\n{c}" for e in ("passive_pre", "passive_post") for c in (stim, other, sham)], fontsize=3.8)
    ax.set_title(f"passive peaks (10-100 ms)\nthreshold {thr:.3g} ({RD[sfx]['unit']}); % above", fontsize=5)
    for a in axs:
        a.tick_params(labelsize=4.2)


def row_onsets(sf, Q, ev, target):
    axs = sf.subplots(len(Q["R"]), 5, squeeze=False)
    for i, sfx in enumerate(Q["R"]):
        onsets_row(axs[i], Q, ev, target, sfx, i == 0)
    label(axs[0, 0], "C")


# ------------------------------------------------------------------------------------------------------------- row E
def row_stats(sf, Q):
    S, B = Q["S"], Q["B"]
    nr = len(Q["R"])
    axs = sf.subplots(1, 3 + nr)
    for ax, sfx in zip(axs[:nr], Q["R"]):
        G, C = Q["R"][sfx]["G"], Q["R"][sfx]["C"]
        masks = [str(x) for x in G["masks"]]
        ns = int(G["n_shuf"])
        for mk, ls in (("all", "-"), ("still", "--")):
            k = masks.index(mk)
            if G["secs"][k] < 30:
                continue
            R = G["counts"][:, k, :] / G["secs"][k]
            lo_, hi_ = np.percentile(R[1:1 + ns], [2.5, 97.5], 0)
            ax.fill_between(G["thr"], lo_, hi_, color=RD[sfx]["band"], alpha=0.6, lw=0)
            ax.plot(G["thr"], R[0], color=RD[sfx]["col"], lw=0.8, ls=ls, label=f"real, {mk}")
        ax.axvline(float(C["thr"]), color="r", lw=0.5, ls="--")
        ax.set_yscale("log")
        ax.set_xlim(G["thr"][0], float(C["med"]) + 3 * (float(C["thr"]) - float(C["med"])))
        ax.set_xlabel(f"threshold ({RD[sfx]['unit']})", fontsize=4.5)
        ax.set_ylabel("events / s", fontsize=4.5)
        ax.legend(frameon=False, fontsize=3.8)
        ax.set_title(f"{RD[sfx]['name']}:\nrate vs threshold (95 % of {ns} shuffles)", fontsize=4.6)
    label(axs[0], "E")
    ax = axs[nr]
    show = [("all", "all"), ("nonevoked", "non-evoked"), ("qw", "QW (no lick)"), ("qw_still", "QW no movement"), ("iti", "ITI"),
            ("iti_still", "ITI no movement"), ("evoked", "evoked tail")]
    for o, sfx in zip(np.linspace(-0.25, 0.25, nr), Q["R"]):
        G, C = Q["R"][sfx]["G"], Q["R"][sfx]["C"]
        masks = [str(x) for x in G["masks"]]
        if not np.isfinite(float(C["thr"])):
            continue
        ns = int(G["n_shuf"])
        for j, (mk, lab) in enumerate(show):
            if mk not in masks or G["secs"][masks.index(mk)] < 30:
                continue
            kk = masks.index(mk)
            r = G["counts_q"][:, kk, qi(G)] / G["secs"][kk]
            ax.scatter([j + o], [r[0]], color=RD[sfx]["col"], s=8, zorder=3)
            ax.errorbar([j + o + 0.08], [r[1:1 + ns].mean()], [r[1:1 + ns].std()], color=RD[sfx]["band"], fmt="o", ms=2.5, lw=0.6)
    G0 = Q["R"][""]["G"]
    m0 = [str(x) for x in G0["masks"]]
    mins = [f"\n{G0['secs'][m0.index(mk)] / 60:.0f} min" if mk in m0 else "" for mk, _ in show]
    ax.set_xticks(range(len(show)), [s[1].replace("\n", " ") + n for s, n in zip(show, mins)], fontsize=3.6, rotation=60, ha="right")
    ax.set_yscale("symlog", linthresh=m9.LINTHRESH)
    ax.set_ylim(bottom=0)
    ax.set_title("rate by condition (QW = quiet window):\nreal (dot) vs shuffles (mean +- s.d.)", fontsize=4.6)
    ax = axs[nr + 1]
    txt = []
    for sfx in Q["R"]:
        G, C, d = Q["R"][sfx]["G"], Q["R"][sfx]["C"], Q["R"][sfx]["d"]
        if not np.isfinite(float(C["thr"])):
            continue
        ns = int(G["n_shuf"])
        r = G["counts_q"][:, 0, qi(G)] / G["secs"][0]
        ax.hist(r[1:1 + ns], 20, color=RD[sfx]["band"], alpha=0.7)
        ax.axvline(r[0], color=RD[sfx]["col"], lw=0.9)
        txt.append(f"{RD[sfx]['name']}: all p {d.get('p_shuf', np.nan):.2f}, still p {d.get('p_shuf_still', np.nan):.2f}")
    ax.set_xlabel("events / s (all valid time)", fontsize=4.5)
    ax.set_title("real (line) vs shuffles at threshold\n" + "\n".join(txt), fontsize=4.4)
    ax = axs[nr + 2]                                                    # movement tuning (009 wz / jz / lk masks)
    x0 = 0
    xt, xl = [], []
    for key, lab in (("wz", "whisker z"), ("jz", "jaw z"), ("lk", "lick dt (s)")):
        nb = m9.TUNE[key]
        edges = m9.Z_EDGES if key != "lk" else m9.LICK_EDGES
        for sfx in Q["R"]:
            G = Q["R"][sfx]["G"]
            masks = [str(x) for x in G["masks"]]
            ks = [masks.index(f"{key}{b}") if f"{key}{b}" in masks else None for b in range(nb)]
            if any(k is None for k in ks):
                continue
            ns, qq = int(G["n_shuf"]), qi(G)
            sec = np.array([G["secs"][k] for k in ks])
            r = np.array([G["counts_q"][:, k, qq] for k in ks]).T / np.where(sec >= 10, sec, np.nan)
            xs = x0 + np.arange(nb)
            ax.plot(xs, r[0], color=RD[sfx]["col"], lw=0.7, marker="o", ms=1.5)
            ax.plot(xs, r[1:1 + ns].mean(0), color=RD[sfx]["band"], lw=0.7, ls="--")
        xt += list(x0 + np.arange(nb))
        xl += [f"{a:g}" if np.isfinite(a) else "" for a in edges[:-1]]
        ax.text(x0 + (nb - 1) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=4)
        x0 += nb + 1
    ax.set_yscale("log")
    ax.set_xticks(xt, xl, fontsize=3.4)
    ax.set_ylabel("events / s", fontsize=4.5)
    ax.set_title("rate vs movement (outside evoked tail): real (solid)\nvs shuffles (dashed); bins: lower edges\n", fontsize=4.6)
    for a in axs:
        a.tick_params(labelsize=4.2)


# ------------------------------------------------------------------------------------------------------------- row F
def row_cohort(sf, sid, target, unit_set, stage):
    f = m1.HOME / f"full_decoders_{unit_set}.csv"
    cols = [(sfx, c, f"{RD[sfx]['name']}: {lab}") for sfx in RD for c, lab in
            (("rate", "rate, all (/s)"), ("rate_still", "rate, still (/s)"), ("excess", "excess vs shuffles (/s)"))] + \
           [("", "bacc", "decoder balanced acc."), ("", "auc_sham_pre", "AUC vs sham, pre")]
    Dd = pd.read_csv(f) if f.exists() else None
    if Dd is not None and "readout" in Dd:
        cols = [c for c in cols if (Dd.thr_source == m9.PREFIX[c[0]] + m9.THR_MAIN).any()]
    axs = sf.subplots(1, len(cols))
    if Dd is None or "readout" not in Dd:
        for ax in axs:
            ax.axis("off")
        axs[0].text(0, 0.5, "cohort comparison: needs the full-dataset table of this 009 version", fontsize=6, transform=axs[0].transAxes)
        return
    st0 = "day 0" if stage.startswith("day 0") else "expert"
    for ax, (sfx, c, lab) in zip(axs, cols):
        q = Dd[(Dd.group == GROUP) & (Dd.target == target) & (Dd.thr_source == m9.PREFIX[sfx] + m9.THR_MAIN)
               & Dd.stage.str.startswith(st0)]
        if c not in q:
            ax.axis("off")
            continue
        for x_, coh in enumerate(COH):
            s_ = q[q.cohort == coh]
            jit = np.random.default_rng(x_).uniform(-0.18, 0.18, len(s_))
            ax.scatter(x_ + jit, s_[c], s=4, lw=0.3, edgecolors=COH[coh],
                       facecolors=[COH[coh] if p >= m9.GOOD_BACC else "w" for p in s_.bacc], zorder=3)
            if s_[c].notna().any():
                ax.plot([x_ - 0.28, x_ + 0.28], [s_[c].median()] * 2, color="k", lw=0.8, zorder=4)
            me = (s_.session_id == sid).to_numpy()
            if me.any():
                ax.scatter(x_ + jit[me], s_[c][me], s=40, facecolors="none", edgecolors="k", lw=0.8, zorder=5)
        a_, b_ = q[q.cohort == "R+"][c].dropna(), q[q.cohort == "R-"][c].dropna()
        if len(a_) >= m9.MIN_SESS and len(b_) >= m9.MIN_SESS:
            p_mw, p_t = m9.m8.two_tests(a_, b_)
            ax.text(0.5, 1.0, f"p {p_mw:.2g} / {p_t:.2g}", transform=ax.transAxes, fontsize=3.8, ha="center", va="bottom")
        if c.startswith(("rate", "excess")):
            ax.set_yscale("symlog", linthresh=m9.LINTHRESH)
        ax.set_xticks([0, 1], list(COH), fontsize=4.2)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(lab, fontsize=4.4, pad=8)
        ax.tick_params(labelsize=4.2)
    label(axs[0], "F")
    sf.suptitle(f"this session (ringed) among {st0} {target} sessions (whole brain; filled: decoder balanced accuracy >= {m9.GOOD_BACC}); "
                f"bar: median; p: R+ vs R- Mann-Whitney / Welch, uncorrected", fontsize=5, x=0.01, ha="left")


# ------------------------------------------------------------------------------------------------------------- page 2
def zoom_data(spikes, Q, lo, hi, rng):
    """counts over [lo, hi]; real and N_ZSHUF shuffled decoder / correlation readouts (009 procedure, real readout's units);
    population activity = mean z-scored count across units; partial correlation | whisker velocity when 009 saved it"""
    t = Q["t"]
    i0, i1 = np.searchsorted(t, lo), np.searchsorted(t, hi)
    ts = t[i0:i1]
    G, Gr = Q["R"][""]["G"], Q["R"]["_corr"]["G"]
    X = np.stack([m1.counts(s, ts) for s in spikes], 1).astype(np.float32)
    w, mu, sd, b = G["w"], G["mu"], G["sd"], float(G["b"])
    st = m9.rate_strata(mu, m9.SHUF_STRATA)
    W = np.c_[w, np.stack([m7.perm_within(w, st, rng) for _ in range(N_ZSHUF)], 1)]
    dec = X @ (W / sd[:, None]).astype(np.float32) + (b - (mu / sd) @ W).astype(np.float32)          # decision values
    tc, mur, sdr = Gr["tc"], Gr["mu"], Gr["sd"]
    TC = np.c_[tc, np.stack([m7.perm_within(tc, st, rng) for _ in range(N_ZSHUF)], 1)]
    s1, s2 = m9.unit_moments(X, mur, sdr)
    cor = m9.corr_readout(X, TC, mur, sdr, s1, s2)                                                 # Pearson r (soft-normalised)
    out = dict(i0=i0, i1=i1, ts=ts, dec=dec, cor=cor, pop=s1 / len(mur))
    if "_pcorr" in Q["R"]:                                                                          # partial correlation | whisker velocity
        Gp = Q["R"]["_pcorr"]["G"]
        Dg = m9.vel_design(Gp["vel"][i0:i1].astype(float), float(Gp["vel_fill"]))
        sv1, sv2 = m9.resid_moments(X, Dg, Gp["vel_beta"], sdr)
        out["pcor"] = m9.corr_resid(X, TC, Dg, Gp["vel_beta"], sdr, sv1, sv2)
    return out


def stretch(sf, Q, ev, Z, lo, hi, spikes, order, title, raster=True):
    S, B = Q["S"], Q["B"]
    i0, i1, ts = Z["i0"], Z["i1"], Z["ts"]
    hr = ([2.0] if raster else []) + [2.2, 0.6, 0.8]
    axs = sf.subplots(len(hr), 1, sharex=True, gridspec_kw=dict(height_ratios=hr))
    k0 = 0
    if raster:
        a_r = axs[0]
        w = Q["R"][""]["G"]["w"]
        for k, u in enumerate(order):
            s = spikes[u]
            s = s[(s >= lo) & (s < hi)]
            a_r.vlines(s - lo, k, k + 0.85, color=WPOS if w[u] > 0 else WNEG, lw=0.25)
        a_r.set_ylim(len(order), 0)
        a_r.set_yticks([0, len(order) - 1], ["max w", "min w"], fontsize=3.6)
        k0 = 1
    a_z, a_p, a_b = axs[k0], axs[k0 + 1], axs[k0 + 2]
    axs[0].set_title(title, fontsize=5, loc="left")
    masks = dict(inv=~S["valid"][i0:i1], evoked=(S.get("evoked", np.zeros(len(S["valid"]), bool)) & S["valid"])[i0:i1],
                 still=(S["still"] & S["valid"])[i0:i1])
    x = ts - lo
    for a in (a_z, a_p):
        for nm, col, al in SHADE:
            st_, en_ = runs(masks[nm])
            for p, q in zip(st_, en_):
                a.axvspan(x[p], x[min(q, len(x) - 1)], color=col, alpha=al, lw=0, zorder=0)
    a_c = a_z.twinx()                                                    # correlations (r) on the right axis
    lims = {}
    for ax_, key, sfx, ypos in ((a_z, "dec", "", 1.0), (a_c, "cor", "_corr", 1.08), (a_c, "pcor", "_pcorr", 1.16)):
        if key not in Z or sfx not in Q["R"]:
            continue
        Y = Z[key]
        C = Q["R"][sfx]["C"]
        lo_, hi_ = np.percentile(Y[:, 1:], [2.5, 97.5], 1)
        ax_.fill_between(x, lo_, hi_, color=RD[sfx]["band"], alpha=0.45, lw=0, zorder=1)
        ax_.plot(x, Y[:, 0], color=RD[sfx]["col"], lw=0.45, zorder=3, label=f"{RD[sfx]['name']} ({RD[sfx]['unit']})")
        ax_.axhline(float(C["thr"]), color=RD[sfx]["col"], lw=0.5, ls="--", zorder=2)
        E = Q["R"][sfx]["E"]
        e = E[(E.t_peak >= lo) & (E.t_peak < hi)]
        ax_.scatter(e.t_peak - lo, np.full(len(e), ypos), transform=ax_.get_xaxis_transform(), marker="v", s=9,
                    color=RD[sfx]["col"], lw=0, zorder=4, clip_on=False)
        md, thr = float(C["med"]), float(C["thr"])
        bot = np.nanpercentile(Y[:, 0], 0.5)
        top = max(md + 1.6 * (thr - md), np.nanpercentile(Y[:, 0], 99.5))                         # stimulus responses clip
        b0, t0 = lims.get(ax_, (bot, top))
        lims[ax_] = (min(b0, bot), max(t0, top))
        ax_.set_ylim(lims[ax_][0] - 0.05 * (lims[ax_][1] - lims[ax_][0]), lims[ax_][1])
        ax_.tick_params(labelsize=4.0, colors=RD[sfx]["col"] if ax_ is a_z else RD["_corr"]["col"])
    a_z.set_ylabel("decoder\n(decision value)", fontsize=4.3)
    a_c.set_ylabel("correlation (r)", fontsize=4.3, color=RD["_corr"]["col"])
    h1, l1 = a_z.get_legend_handles_labels()
    h2, l2 = a_c.get_legend_handles_labels()
    a_z.legend(h1 + h2, l1 + l2, frameon=False, fontsize=3.8, loc="upper left", ncol=2)
    a_p.plot(x, Z["pop"], color="0.35", lw=0.4)
    a_p.set_ylabel("pop.\n(mean z)", fontsize=4.3)
    tv, vv = speed(B, "whisker_speed")
    if tv is not None:
        m = (tv >= lo) & (tv < hi)
        a_b.plot(tv[m] - lo, vv[m] / max(np.nanpercentile(vv, 99), 1e-9), color="0.3", lw=0.4, label="whisker speed")
    tj, vj = speed(B, "jaw")
    if tj is not None:
        m = (tj >= lo) & (tj < hi)
        a_b.plot(tj[m] - lo, vj[m] / max(np.nanpercentile(vj[np.isfinite(vj)], 99), 1e-9), color="tab:purple", lw=0.4, label="jaw speed")
    lk = B["licks"][(B["licks"] >= lo) & (B["licks"] < hi)] - lo
    a_b.vlines(lk, 1.05, 1.35, color="tab:orange", lw=0.6)
    a_b.set_ylim(0, 1.4)
    a_b.set_ylabel("behav.\n(norm.)", fontsize=4.3)
    a_b.legend(frameon=False, fontsize=3.6, loc="upper right", ncol=2)
    for a in axs:
        for r in ev[(ev.t >= lo) & (ev.t < hi)].itertuples():
            a.axvline(r.t - lo, color=CLS_COL.get(r.cls, "k"), lw=0.5, ls="--" if r.cls in ("SW", "SA") else "-", zorder=5)
        a.tick_params(labelsize=4.0)
    a_b.set_xlim(0, hi - lo)
    a_b.set_xlabel(f"time (s) from {lo / 60:.2f} min", fontsize=4.3)


def pick_stretches(Q, ev, target, rng):
    """(lo, hi, title): passive pre, strongest non-evoked decoder / correlation event, typical still decoder / correlation
    event, random task stretch; overlapping picks skipped"""
    t, valid = Q["t"], Q["S"]["valid"]
    out = []

    def add(c, title):
        lo = c - SHORT_S / 2
        if all(abs(lo - a) >= SHORT_S for a, _, _ in out):
            out.append((lo, lo + SHORT_S, title))

    stim = m1.TARGET_CLS[target][0]
    pp = ev[(ev.epoch == "passive_pre") & (ev.cls == stim)].t.to_numpy()
    if len(pp):
        add(pp[len(pp) // 2], "passive pre (stimuli are training data; their -10..+35 ms windows invalid)")
    for sfx in Q["R"]:
        E = Q["R"][sfx]["E"]
        Ene = E[~E.evoked.to_numpy(bool)] if "evoked" in E else E
        if len(Ene):
            r = Ene.loc[Ene.peak.idxmax()]
            add(float(r.t_peak), f"strongest non-evoked {RD[sfx]['name']} event (peak {r.peak:.3g} {RD[sfx]['unit']}, movement {r.movement})")
        Es = E[E.still.to_numpy(bool)]
        if len(Es):
            r = Es.iloc[int(np.argmin(np.abs(Es.peak.to_numpy() - Es.peak.median())))]
            add(float(r.t_peak), f"typical still {RD[sfx]['name']} event (peak {r.peak:.3g} {RD[sfx]['unit']})")
    task = ev[ev.context == "task"].t.to_numpy()
    for _ in range(20):
        if len(out) >= STRETCHES or not len(task):
            break
        add(float(rng.uniform(task.min() + SHORT_S, task.max() - SHORT_S)), "random task stretch")
    return out[:STRETCHES]


# ------------------------------------------------------------------------------------------------------------- pages
def page(sid, target, unit_set):
    from matplotlib.backends.backend_pdf import PdfPages
    plt = m2.setup()
    Q = load(sid, target, unit_set)
    Sx = m9.sessions_table().set_index("session_id")
    Q["coh"] = str(Sx.cohort.get(sid, "?"))
    stage = m9.stage_of(sid)
    D = m6.prepare(sid)
    ev = D["ev"]
    assert len(ev) == len(Q["pk"]) and np.allclose(ev.t.to_numpy(), Q["pk"].t.to_numpy()), "events differ from 009 peaks"
    keep = m9.unit_set_mask(sid, D["units"], unit_set)
    sp_all = [s for s, k in zip(D["spikes"], keep) if k]
    units = D["units"][keep].reset_index(drop=True)
    sel = Q["R"][""]["G"]["unit_index"]
    spikes = [np.asarray(sp_all[i]) for i in sel]
    units_sel = units.iloc[sel].reset_index(drop=True)
    w = Q["R"][""]["G"]["w"]
    top = np.argsort(np.abs(w))[::-1][:N_TOP]
    order = top[np.argsort(w[top])[::-1]]
    rng = np.random.default_rng(zlib.crc32(f"{sid}|{target}|overview".encode()))
    picks = pick_stretches(Q, ev, target, rng)
    task = ev[ev.context == "task"].t.to_numpy()
    lmid = float(np.median(task)) if len(task) else float(Q["t"][len(Q["t"]) // 2])
    long_ = (lmid - LONG_S / 2, lmid + LONG_S / 2)
    fdir = m1.HOME / "figures" / f"full_{unit_set}" / "overview"
    fdir.mkdir(parents=True, exist_ok=True)
    dd, dr = Q["R"][""]["d"], Q["R"]["_corr"]["d"]
    P = Q["P"]
    # ---- page 1
    fig = plt.figure(figsize=(8.27, 11.69), layout="constrained")
    sfs = fig.subfigures(6, 1, height_ratios=[0.5, 2.3, 1.15, 2.4, 1.35, 1.1])
    head = sfs[0]
    head.text(0.0, 0.97, f"{sid}  |  {Q['coh']}, {stage}  |  whole-brain {target}, {int(Q['R']['']['G']['n_units'])} {unit_set} units  |  decoder "
              f"balanced acc. {dd.bacc:.2f} (task {dd.bacc_task:.2f}, passive pre {dd.bacc_passive_pre:.2f} / post {dd.bacc_passive_post:.2f})",
              fontsize=7, weight="bold", va="top")
    lines = []
    for sfx in Q["R"]:
        d = Q["R"][sfx]["d"]
        lines.append(f"{RD[sfx]['name']}: threshold {d.thr_raw:.3g} {RD[sfx]['unit']} (= median + {d.thr_norm:.1f} robust s.d.); events {d.get('n_events', np.nan):.0f} = "
                     f"{d.get('rate', np.nan):.4f}/s (shuffles {d.get('rate_shuf', np.nan):.4f}, p {d.get('p_shuf', np.nan):.2f}); still "
                     f"{d.get('rate_still', np.nan):.4f}/s (shuffles {d.get('rate_shuf_still', np.nan):.4f}, p {d.get('p_shuf_still', np.nan):.2f}); "
                     f"passive AUC vs sham {d.auc_sham_pre:.2f} / {d.auc_sham_post:.2f}")
    sec = P["secs"]
    lines.append(f"valid {sec.get('all', 0) / 60:.1f} min (still {sec.get('still', 0) / 60:.1f}, whisking {sec.get('whisk', 0) / 60:.1f}, lick "
                 f"{sec.get('lick', 0) / 60:.1f}, evoked tail {sec.get('evoked', 0) / 60:.1f}, passive {sec.get('passive', 0) / 60:.1f}); "
                 f"{m9.THR_TEXT}; readout = 30-ms window ending at t, 5-ms stride; raw units: decoder decision value (log-odds), correlation r")
    lines.append(m9.STILL_NOTE)
    head.text(0.0, 0.6, "\n".join(lines), fontsize=4.8, va="top")
    head.patch.set_alpha(0)
    row_timeline(sfs[1], Q, D, ev, [long_] + [(a, b) for a, b, _ in picks])
    row_training(sfs[2], Q, ev, target, units_sel)
    row_onsets(sfs[3], Q, ev, target)
    row_stats(sfs[4], Q)
    row_cohort(sfs[5], sid, target, unit_set, stage)
    # ---- page 2
    fig2 = plt.figure(figsize=(8.27, 11.69), layout="constrained")
    sfs2 = fig2.subfigures(1 + len(picks), 1, height_ratios=[1.3] + [1.9] * len(picks))
    Z = zoom_data(spikes, Q, *long_, rng)
    err = float(np.nanmax(np.abs(Z["dec"][:, 0] - Q["R"][""]["C"]["z"][Z["i0"]:Z["i1"]]))) if len(Z["ts"]) else np.nan
    err_c = float(np.nanmax(np.abs(Z["cor"][:, 0] - Q["R"]["_corr"]["C"]["z"][Z["i0"]:Z["i1"]]))) if len(Z["ts"]) else np.nan
    err_p = float(np.nanmax(np.abs(Z["pcor"][:, 0] - Q["R"]["_pcorr"]["C"]["z"][Z["i0"]:Z["i1"]]))) if "pcor" in Z and len(Z["ts"]) else np.nan
    stretch(sfs2[0], Q, ev, Z, *long_, spikes, order, f"{sid} {target}: 3-min task stretch -- decoder (black), correlation (brown), partial correlation | whisker velocity (pink) "
            f"readouts with the 95 % band of {N_ZSHUF} template shuffles and thresholds (dashed), events (triangles); population activity; "
            f"whisker / jaw speed, licks (orange); stimulus onsets (lines; dashed = sham)", raster=False)
    del Z
    for sf, (lo, hi, ttl) in zip(sfs2[1:], picks):
        Z = zoom_data(spikes, Q, lo, hi, rng)
        stretch(sf, Q, ev, Z, lo, hi, spikes, order, f"{ttl}; raster: top {len(order)} units by |decoder weight| (red +, blue -)")
    for ext_fig, nm in ((fig, "p1"), (fig2, "p2")):
        ext_fig.savefig(fdir / f"{sid}_{target}_{nm}.png", dpi=200)
    with PdfPages(fdir / f"{sid}_{target}.pdf") as pp:
        pp.savefig(fig)
        pp.savefig(fig2)
    plt.close(fig)
    plt.close(fig2)
    print(f"overview {sid} {target}: reconstruction max |err| decoder {err:.2e}, correlation {err_c:.2e}, partial {err_p:.2e} -> {fdir / f'{sid}_{target}.pdf'}",
          flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs="*", default=["whisker", "auditory"])
    ap.add_argument("--unit-set", default="stable")
    ap.add_argument("--sessions", nargs="*", default=None, help="default: 010 example sessions")
    a = ap.parse_args()
    for tg in a.target:
        ss = a.sessions
        if ss is None:
            Dd = pd.read_csv(m1.HOME / f"full_decoders_{a.unit_set}.csv")
            D = Dd[(Dd.group == GROUP) & (Dd.target == tg) & (Dd.thr_source == m9.THR_MAIN)
                   & (Dd.bacc >= m9.GOOD_BACC) & Dd.stage.str.startswith("day 0")].dropna(subset=["rate_still"])
            ss = [r.session_id for c in COH for r in m10.pick(D, c)]
        for s in ss:
            try:
                page(s, tg, a.unit_set)
            except Exception as e:
                import traceback
                print("FAILED overview", s, tg, repr(e), flush=True)
                traceback.print_exc()
