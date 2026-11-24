"""Figure pages for Jules Lebert's context dataset (S1M2_DATASET=jl), used by 005_replication_figure.py and
007_hires_figure.py. Groups are the two context blocks (rewarded / non-rewarded) instead of SSL cohort x stage; contexts
are compared within session (paired Wilcoxon + paired t-test). Whisker trials in the rewarded context are matched to the
paper's Go trials, in the non-rewarded context to NoGo (lick = hit / FA, no lick = miss / CR)."""
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import s1m2_lib as L

PAIR = "SSp-bfd__wMO"
CTX_COL = {"rew": "#00B400", "nonrew": "#C800C8", "both": "0.15"}
CTX_LAB = {"rew": "rewarded context", "nonrew": "non-rewarded context", "both": "both contexts"}
PAPER_JL = {("All", "both"): "All", ("W_all", "rew"): "Go", ("W_all", "nonrew"): "NoGo", ("W_hit", "rew"): "hit",
            ("W_hit", "nonrew"): "FA", ("W_miss", "rew"): "miss", ("W_miss", "nonrew"): "CR"}
TRIAL_TYPES_JL = [("All", "All trials"), ("W_all", "Whisker trials (all outcomes)"),
                  ("W_hit", "Whisker trials with lick"), ("W_miss", "Whisker trials without lick"),
                  ("A_hit", "Auditory hits"), ("NS_all", "No-stimulus trials")]


def wil(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def paired_p(a, b):
    j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(j) < 5:
        return len(j), np.nan, np.nan, np.nan
    d = j.a - j.b
    return len(j), d.mean(), float(stats.wilcoxon(d).pvalue), float(stats.ttest_rel(j.a, j.b).pvalue)


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    return "p<1e-4" if p < 1e-4 else "p=" + (f"{p:.2f}" if p >= 0.01 else f"{p:.1g}").lstrip("0")


def mean_sem(x):
    x = np.asarray(x, float)
    return np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(np.sum(np.isfinite(x), 0).clip(1))


def crop(m, T, D, tlim=(-0.15, 0.25), dlim=0.05):
    w, tm = np.abs(D) <= dlim + 1e-9, (T >= tlim[0] - 1e-9) & (T <= tlim[1] + 1e-9)
    return m[np.ix_(w, tm)], [T[tm][0] * 1e3, T[tm][-1] * 1e3, D[w][0] * 1e3, D[w][-1] * 1e3]


def readout_panel(ax, df, col, lab, ctxs, paper=None, rng=None):
    """paper: list of (label, values) drawn first in grey. JL values per context with paired lines; title = paired
    test rewarded vs non-rewarded."""
    rng = np.random.default_rng(0) if rng is None else rng
    xt, xl, xc, x0 = [], [], [], 0
    for plab, v in (paper or []):
        v = v.dropna()
        ax.errorbar(x0, v.mean(), v.std() / np.sqrt(len(v)), color="0.4", marker="s", ms=3.5, capsize=2, lw=0.8)
        xt.append(x0); xl.append(f"paper\n{plab}\nn={len(v)}\n{fmt_p(wil(v))}"); xc.append("0.3")
        x0 += 1
    xs = {}
    for c in ctxs:
        v = df[df.ctx == c].set_index("session")[col].dropna()
        if not len(v):
            continue
        xs[c] = (x0, v)
        ax.scatter(x0 - .15 + rng.uniform(-.05, .05, len(v)), v, s=4, c=CTX_COL[c], alpha=.5, lw=0)
        ax.errorbar(x0 + .15, v.mean(), v.std() / np.sqrt(len(v)), color=CTX_COL[c], capsize=2, lw=0.8, marker="o",
                    ms=3.5)
        xt.append(x0); xc.append(CTX_COL[c]); xl.append(f"JL\n{c}\nn={len(v)}\n{fmt_p(wil(v))}")
        x0 += 1
    ttl = ""
    if "rew" in xs and "nonrew" in xs:
        (xa, va), (xb, vb) = xs["rew"], xs["nonrew"]
        j_ = pd.concat([va, vb], axis=1, keys=["a", "b"]).dropna()
        for _, r in j_.iterrows():
            ax.plot([xa - .15, xb - .15], [r.a, r.b], color="0.75", lw=0.4, zorder=0)
        n, _, pw, pt = paired_p(va, vb)
        if np.isfinite(pw):
            ttl = f"rew vs nonrew, {n} pairs:\nWilcoxon {fmt_p(pw)}, t {fmt_p(pt)}"
    ax.set_title(ttl, fontsize=5.5)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=5)
    for t_, c_ in zip(ax.get_xticklabels(), xc):
        t_.set_color(c_)
    ax.set_xlim(-0.6, max(x0, 1) - 0.4)
    ax.set_ylabel(lab)


def page_v3(cond, title, kind, df_all, maps, traces, T, D, tr, ref_all, clean=False):
    """005 page: A maps (paper matching trial types, JL per context), B time courses, C per-session readouts."""
    pk = "Total" if kind == "total" else "Noise"
    sfx = "_clean" if clean else ""
    early_lo = 30 if clean else 0

    def mask(m, Tm, Dm):
        if not clean:
            return m
        m = np.where(L.gap_mask(Dm, Tm), m, np.nan)
        b = (Tm >= -0.25) & (Tm <= 0)
        return m - np.nanmean(m[:, b], axis=1, keepdims=True)

    df = df_all[(df_all.pair == PAIR) & (df_all.condition == cond) & (df_all.kind == kind)]
    ctxs = [c for c in ("rew", "nonrew", "both") if (df.ctx == c).any()]
    pcs = {c: PAPER_JL.get((cond, c)) for c in ctxs}
    fig = plt.figure(figsize=(7.4, 7.0))
    g1 = fig.add_gridspec(1, 5, wspace=0.55, left=0.08, right=0.9, top=0.89, bottom=0.71)
    g2 = fig.add_gridspec(1, 4, wspace=0.55, left=0.09, right=0.98, top=0.6, bottom=0.45)
    g3 = fig.add_gridspec(1, 3, wspace=0.6, left=0.09, right=0.98, top=0.33, bottom=0.16)
    panels = []
    for c in ctxs:
        pc = pcs[c]
        if pc and f"{pk}_{pc}_map_mean" in tr:
            Tp = tr[f"{pk}_{pc}_t"]
            pm = tr[f"{pk}_{pc}_map_mean"]
            pm = pm - np.nanmean(pm[:, (Tp >= -0.25) & (Tp <= 0)], axis=1, keepdims=True)
            panels.append((f"Paper, expert\n{pc} trials (n = 50)", *crop(mask(pm, Tp, tr["delays"]), Tp, tr["delays"]),
                           "k"))
    for c in ctxs:
        ms = [maps[k] for k in [(s, PAIR, cond, kind) for s in df[df.ctx == c].session_id] if k in maps]
        m = crop(mask(np.nanmean(ms, axis=0), T, D), T, D) if ms else (None, None)
        panels.append((f"JL, {CTX_LAB[c]}\n(n = {len(ms)} sessions)", *m, CTX_COL[c]))
    vals = np.concatenate([p[1].ravel() for p in panels if p[1] is not None])
    vmax = np.nanpercentile(np.abs(vals), 99)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.82")
    for j, (ttl, m, ext, col) in enumerate(panels[:5]):
        ax = fig.add_subplot(g1[0, j])
        ax.set_title(ttl, color=col)
        if m is None:
            ax.set_axis_off()
            continue
        im = ax.imshow(np.ma.masked_invalid(m), aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                       extent=ext, interpolation="nearest")
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.set_xticks([-100, 0, 100, 200])
        ax.set_xlabel("Time from stimulus (ms)")
        ax.set_ylabel("Delay (ms; > 0: S1 leads)")
    cax = fig.add_axes([0.915, 0.73, 0.01, 0.14])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("CCV minus\nbaseline", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    # B
    idx = 5 if clean else 0
    axes_b, ylo, yhi, tr_by = [], np.inf, -np.inf, {}
    for j, c in enumerate(ctxs):
        ax = fig.add_subplot(g2[0, j])
        axes_b.append(ax)
        pc = pcs[c]
        if pc and f"{pk}_{pc}_FF{sfx}" in tr:
            m, se = mean_sem(tr[f"{pk}_{pc}_FF{sfx}"])
            Tp = tr[f"{pk}_{pc}_t"]
            ax.plot(Tp * 1e3, m, color="0.55", lw=0.9, label=f"paper {pc}")
            ax.fill_between(Tp * 1e3, m - se, m + se, color="0.55", alpha=.2, lw=0, edgecolor="none")
            ylo, yhi = min(ylo, np.nanmin(m - se)), max(yhi, np.nanmax(m + se))
        x = [traces[k][idx] for k in [(s, PAIR, cond, kind) for s in df[df.ctx == c].session_id] if k in traces]
        if x:
            m, se = mean_sem(x)
            tr_by[c] = (m, se, len(x))
            ax.plot(T * 1e3, m, color=CTX_COL[c], lw=1, label=f"JL {CTX_LAB[c]}")
            ax.fill_between(T * 1e3, m - se, m + se, color=CTX_COL[c], alpha=.2, lw=0, edgecolor="none")
            sel = (T >= -0.25) & (T <= 0.4)
            ylo, yhi = min(ylo, np.nanmin((m - se)[sel])), max(yhi, np.nanmax((m + se)[sel]))
        ax.set_title(f"JL {CTX_LAB[c]} (n = {len(x)})", color=CTX_COL[c])
    ax = fig.add_subplot(g2[0, len(ctxs)])
    axes_b.append(ax)
    for c, (m, se, n) in tr_by.items():
        ax.plot(T * 1e3, m, color=CTX_COL[c], lw=1, label=f"{CTX_LAB[c]} (n={n})")
        ax.fill_between(T * 1e3, m - se, m + se, color=CTX_COL[c], alpha=.2, lw=0, edgecolor="none")
    ax.set_title("JL, contexts overlaid")
    for ax in axes_b:
        ax.axvspan(early_lo, 100, facecolor="gold", alpha=.15, lw=0, edgecolor="none")
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(0, color="k", lw=0.4, ls="--")
        ax.set_xlim(-250, 400)
        ax.set_xlabel("Time from stimulus (ms)")
        ax.set_ylabel("S1-leading CCV\nminus baseline")
        ax.legend(frameon=False, loc="upper right", handlelength=1.2)
        if np.isfinite(ylo):
            pad = 0.08 * (yhi - ylo)
            ax.set_ylim(ylo - pad, yhi + 3 * pad)
    # C
    win = f"{early_lo}-100 ms"
    specs = ((f"FF{sfx}", f"S1 -> M2 coupling, {win}\n(in-sample CCV minus baseline)", True),
             (f"CV_FF{sfx}", f"S1 -> M2 coupling, {win}\n(cross-validated CCV minus baseline)", False),
             (f"FF_minus_FB{sfx}", "Feedforward minus feedback\n(S1 leading minus M2 leading)", True))
    vcol = "value_clean" if clean else "value"
    for j, (col, lab, with_paper) in enumerate(specs):
        paper = []
        if with_paper:
            for c in ctxs:
                pc = pcs[c]
                if not pc:
                    continue
                ref = ref_all[(ref_all["corr"] == pk) & (ref_all.condition == pc)]
                v = ref[ref.direction == "FF"].set_index("session_idx")[vcol]
                if col.startswith("FF_minus_FB"):
                    v = v - ref[ref.direction == "FB"].set_index("session_idx")[vcol]
                paper.append((pc, v))
        readout_panel(fig.add_subplot(g3[0, j]), df, col, lab, ctxs, paper)
    for y, l in ((0.925, "A"), (0.635, "B"), (0.365, "C")):
        fig.text(0.01, y, l, fontsize=9, fontweight="bold")
    fig.suptitle(f"{title}: S1 (SSp-bfd) -> M2 (MO-wM1 + MO-wM2), Jules Lebert context task, {kind} coupling"
                 + (", artefact-free windows only" if clean else ""), fontsize=8, y=0.985)
    cap = ("Same method as the SSL replication (Ranjbar-Slamloo & Kamigaki 2026 recipe: 5-ms bins, 50-ms windows every "
           "10 ms, delays +-100 ms; each area's stable good + MUA units, >= 10 per area, reduced to its top 10 PCs "
           "before CCA). Jules Lebert's context task, expert mice: whisker / auditory / no-stimulus trials in "
           "alternating rewarded (green) and non-rewarded (magenta) context blocks. Whisker trials in the rewarded "
           "context are set against the paper's Go trials, in the non-rewarded context against NoGo (lick = hit / FA, "
           "no lick = miss / CR). Readout: CCV over delays 0..+15 ms minus its -250..0 ms mean, averaged over 0-100 ms "
           "(yellow). Spikes in -10..+5 ms around every trial start removed (whisker artefact). Session = unit; mean +- "
           "s.e.m. over sessions; p under the ticks: Wilcoxon signed-rank vs 0; contexts compared within session "
           "(paired Wilcoxon and paired t-test, grey lines), uncorrected. Paper values recomputed from their Zenodo "
           "maps. Conditions with < 20 trials in a session are not computed for that session.")
    if clean:
        cap = cap.replace("averaged over 0-100 ms (yellow)", "averaged over +30..+100 ms (yellow), using only map "
                          "cells whose S1 and M2 windows are clear of the artefact gap (grey in A), also for the paper")
    fig.text(0.07, 0.012, "\n".join(textwrap.wrap(cap, 190)), fontsize=5.2, va="bottom")
    return fig


def remap_hires(df):
    """007 rows: condition 'W_hit_rew' -> condition 'W_hit', ctx 'rew'; the real session id goes to 'session'."""
    sp = df.condition.map(L.split_condition)
    return df.assign(session=df.session_id, cond_full=df.condition, condition=[b for b, _ in sp],
                     ctx=[c for _, c in sp])


def page_hires(cond, title, kind, df_all, maps, ws, D, epochs, base_label):
    """007 page: A maps per context, B S1-leading / M2-leading traces, C per-session epoch readouts."""
    df = df_all[df_all.condition == cond]
    ctxs = [c for c in ("rew", "nonrew", "both") if (df.ctx == c).any()]
    fig = plt.figure(figsize=(7.4, 7.6))
    g1 = fig.add_gridspec(1, 4, wspace=0.5, left=0.09, right=0.9, top=0.89, bottom=0.71)
    g2 = fig.add_gridspec(1, 4, wspace=0.55, left=0.09, right=0.98, top=0.6, bottom=0.45)
    g3 = fig.add_gridspec(1, 5, wspace=0.75, left=0.08, right=0.98, top=0.32, bottom=0.15)
    tm = (ws >= -0.050 - 1e-9) & (ws <= 0.100 + 1e-9)
    ext = [ws[tm][0] * 1e3, ws[tm][-1] * 1e3, D[0] * 1e3, D[-1] * 1e3]
    mm = {}
    for c in ctxs:
        h = df[df.ctx == c]
        ms = [maps[k] for k in zip(h.session, h.cond_full) if k in maps]
        mm[c] = (np.nanmean(ms, axis=0)[:, tm] if ms else None, len(ms))
    vals = np.concatenate([v[0].ravel() for v in mm.values() if v[0] is not None])
    vmax = np.nanpercentile(np.abs(vals), 99)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.82")
    for j, c in enumerate(ctxs):
        ax = fig.add_subplot(g1[0, j])
        m, n = mm[c]
        ax.set_title(f"JL, {CTX_LAB[c]}\n(n = {n} sessions)", color=CTX_COL[c])
        if m is None:
            ax.set_axis_off()
            continue
        im = ax.imshow(np.ma.masked_invalid(m), aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax,
                       extent=ext, interpolation="nearest")
        ax.axhline(0, color="k", lw=0.4)
        ax.axvline(5, color="r", lw=0.5, ls="--")
        ax.set_xlabel("Window start (ms from stimulus)")
        ax.set_ylabel("Delay (ms; > 0: S1 leads)")
    cax = fig.add_axes([0.915, 0.73, 0.01, 0.14])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("CCV minus\nbaseline", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    axes_b, lo, hi = [], np.inf, -np.inf
    for j, c in enumerate(ctxs):
        ax = fig.add_subplot(g2[0, j])
        axes_b.append(ax)
        h = df[df.ctx == c]
        for key, ls, lab in (("is_ff", "-", "S1 leads (0..15 ms)"), ("is_fb", ":", "M2 leads (0..15 ms)")):
            x = [t[key] for t in h.traces]
            if not x:
                continue
            m, se = mean_sem(x)
            ax.plot(ws * 1e3, m, color=CTX_COL[c], ls=ls, lw=1, label=lab)
            ax.fill_between(ws * 1e3, m - se, m + se, color=CTX_COL[c], alpha=.15, lw=0, edgecolor="none")
            sel = (ws >= -0.05) & (ws <= 0.1)
            lo, hi = min(lo, np.nanmin((m - se)[sel])), max(hi, np.nanmax((m + se)[sel]))
        ax.set_title(f"JL {CTX_LAB[c]} (n = {len(h)})", color=CTX_COL[c])
        ax.legend(frameon=False, loc="upper left")
    for ax in axes_b:
        ax.axvspan(-20, 5, facecolor="0.9", lw=0, edgecolor="none", zorder=0)
        ax.axvline(5, color="r", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlim(-50, 100)
        ax.set_xlabel("Window start (ms from stimulus)")
        ax.set_ylabel("CCV minus baseline")
        if np.isfinite(lo):
            pad = 0.1 * (hi - lo)
            ax.set_ylim(lo - pad, hi + 2.5 * pad)
    el = [f"{int(round(a * 1e3))}-{int(round(b * 1e3))} ms" for a, b in epochs]
    specs = [("is_ff_e0", f"S1-leading CCV, in-sample\nwindows starting {el[0]}"),
             ("cv_ff_e0", f"S1-leading CCV, cross-validated\nwindows starting {el[0]}"),
             ("cv_ff_e1", f"S1-leading CCV, cross-validated\nwindows starting {el[1]}"),
             ("cv_ff_e2", f"S1-leading CCV, cross-validated\nwindows starting {el[2]}"),
             ("is_ff_minus_fb", "S1 leading minus M2 leading\n(windows starting 25-95 ms)")]
    for j, (col, lab) in enumerate(specs):
        readout_panel(fig.add_subplot(g3[0, j]), df, col, lab, ctxs)
    for y, l in ((0.925, "A"), (0.635, "B"), (0.365, "C")):
        fig.text(0.01, y, l, fontsize=9, fontweight="bold")
    fig.suptitle(f"{title}: S1 (SSp-bfd) -> M2 (MO-wM1 + MO-wM2), high-resolution time-shifted CCA, Jules Lebert "
                 f"context task, {kind} coupling", fontsize=8, y=0.985)
    cap = ("1-ms bins, 10-ms windows stepped every 1 ms and labelled by their start, M2 window shifted by the delay "
           "(-20..+20 ms), first canonical correlation (CCV) after reducing each area (stable good + MUA units) to its "
           "top 10 PCs. Only cells whose S1 and M2 windows are both clear of the -10..+5 ms artefact gap (grey) are "
           f"used: the first S1-leading window is 5-15 ms (red line). Baseline: {base_label}, per delay. Rewarded "
           "(green) and non-rewarded (magenta) context blocks. Session = unit; mean +- s.e.m. over sessions; p under "
           "ticks: Wilcoxon signed-rank vs 0; contexts compared within session (paired Wilcoxon and paired t-test, "
           "grey lines), uncorrected.")
    fig.text(0.08, 0.012, "\n".join(textwrap.wrap(cap, 185)), fontsize=5.2, va="bottom")
    return fig
