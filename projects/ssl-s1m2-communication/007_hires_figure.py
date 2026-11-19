"""High-resolution S1 -> M2 time-shifted CCA (profile "hires": 1-ms bins, 10-ms windows stepped every 1 ms, delays
-20..+20 ms), artefact-free cells only (both S1 and M2 windows clear of -10..+5 ms). Windows labelled by their START,
so the first S1-leading point is the 5-15 ms window. One page per trial type, multi-page PDF + PNG per page.

  A  mean maps (delay x window start), baseline-subtracted per delay (clean windows starting -100..-25 ms), grey =
     excluded (overlaps the artefact gap).
  B  S1-leading (delays 0..+15 ms, solid) and M2-leading (-15..0 ms, dotted) CCV over window start, per cohort x stage.
  C  per-session S1-leading CCV in three epochs of window start (5-15, 15-35, 35-95 ms): in-sample and 10-fold CV.
  D  S1-leading minus M2-leading, window starts 25-95 ms (first epoch where both are artefact-free).

Run (after 003 --profile hires): python 007_hires_figure.py [--kind total]  ->  OUT/replication/hires_<kind>.pdf
"""
import argparse
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import s1m2_lib as L

L.set_profile("hires")
COL = {"R+": "#00B400", "R-": "#C800C8"}
GROUPS = [("learning", "R+"), ("learning", "R-"), ("expert", "R+"), ("expert", "R-")]
PAIR = "SSp-bfd__wMO"
TRIAL_TYPES = [("All", "All active trials"), ("W_all", "Whisker trials (all outcomes)"), ("W_hit", "Whisker hits"),
               ("W_miss", "Whisker misses"), ("A_hit", "Auditory hits")]
BASE = (-0.100, -0.025)                     # window starts used as baseline
EPOCHS = [(0.005, 0.015), (0.015, 0.035), (0.035, 0.095)]
DIR_EPOCH = (0.025, 0.095)
FF, FB = (0.0, 0.015), (-0.015, 0.0)
plt.rcParams.update({"font.family": "Arial", "font.size": 6.5, "axes.titlesize": 6.5, "axes.labelsize": 6.5,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 5.5, "pdf.fonttype": 42,
                     "ps.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.0})


def wil(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def fmt_p(p):
    if not np.isfinite(p):
        return ""
    return "p<1e-4" if p < 1e-4 else "p=" + (f"{p:.2f}" if p >= 0.01 else f"{p:.1g}").lstrip("0")


def mean_sem(x):
    x = np.asarray(x, float)
    return np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(np.sum(np.isfinite(x), 0).clip(1))


def load(kind):
    """per (session, condition): clean baseline-subtracted map, S1-/M2-leading traces (in-sample, CV), metadata"""
    rows, maps = [], {}
    for f in sorted((L.OUT / "maps_v4_hires").glob(f"*__{PAIR}__*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        T, ws, D, Dcv = z["times"], z["window_starts"], z["delays"], z["cv_delays"]
        out = {}
        for lab, m, dd in (("is", z[f"{kind}__ccv"].astype(float), D), ("cv", z[f"{kind}__cv"].astype(float), Dcv)):
            m = np.where(L.gap_mask(dd, T), m, np.nan)
            b = (ws >= BASE[0] - 1e-9) & (ws <= BASE[1] + 1e-9)
            m = m - np.nanmean(m[:, b], axis=1, keepdims=True)
            for nm, rng in (("ff", FF), ("fb", FB)):
                sel = (dd >= rng[0] - 1e-9) & (dd <= rng[1] + 1e-9)
                # a time point counts only if every delay in the range is artefact-free (else the M2-leading trace
                # would average only the small delays at early times)
                ok = L.gap_mask(dd, T)[sel].all(axis=0)
                out[f"{lab}_{nm}"] = np.where(ok, np.nanmean(m[sel], axis=0), np.nan)
            if lab == "is":
                maps[(meta["session_id"], meta["condition"])] = m
        r = dict(meta)
        for i, (a, b_) in enumerate(EPOCHS):
            e = (ws >= a - 1e-9) & (ws <= b_ + 1e-9)
            r[f"is_ff_e{i}"] = np.nanmean(out["is_ff"][e])
            r[f"cv_ff_e{i}"] = np.nanmean(out["cv_ff"][e])
        e = (ws >= DIR_EPOCH[0] - 1e-9) & (ws <= DIR_EPOCH[1] + 1e-9)
        r["is_ff_minus_fb"] = np.nanmean(out["is_ff"][e] - out["is_fb"][e])
        r["traces"] = out
        rows.append(r)
    return pd.DataFrame(rows), maps, ws, D


def page(cond, title, kind, df_all, maps, ws, D):
    df = df_all[df_all.condition == cond]
    fig = plt.figure(figsize=(7.4, 7.6))
    g1 = fig.add_gridspec(1, 4, wspace=0.5, left=0.09, right=0.9, top=0.89, bottom=0.71)
    g2 = fig.add_gridspec(1, 4, wspace=0.55, left=0.09, right=0.98, top=0.6, bottom=0.45)
    g3 = fig.add_gridspec(1, 3, wspace=0.45, left=0.09, right=0.98, top=0.32, bottom=0.17, width_ratios=[1.35, 1.35, 1])
    tm = (ws >= -0.050 - 1e-9) & (ws <= 0.100 + 1e-9)
    ext = [ws[tm][0] * 1e3, ws[tm][-1] * 1e3, D[0] * 1e3, D[-1] * 1e3]
    mm = {}
    for st, co in GROUPS:
        sess = df[(df.stage == st) & (df.cohort == co)].session_id
        ms = [maps[(s, cond)] for s in sess if (s, cond) in maps]
        mm[(st, co)] = (np.nanmean(ms, axis=0)[:, tm] if ms else None, len(ms))
    vals = np.concatenate([v[0].ravel() for v in mm.values() if v[0] is not None])
    vmax = np.nanpercentile(np.abs(vals), 99)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.82")
    for j, (st, co) in enumerate(GROUPS):
        ax = fig.add_subplot(g1[0, j])
        m, n = mm[(st, co)]
        ax.set_title(f"SSL {co}, {st}\n(n = {n} sessions)", color=COL[co])
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
    # B
    axes_b, lo, hi = [], np.inf, -np.inf
    for j, (st, co) in enumerate(GROUPS):
        ax = fig.add_subplot(g2[0, j])
        axes_b.append(ax)
        h = df[(df.stage == st) & (df.cohort == co)]
        for key, ls, lab in (("is_ff", "-", "S1 leads (0..15 ms)"), ("is_fb", ":", "M2 leads (0..15 ms)")):
            x = [t[key] for t in h.traces]
            if not x:
                continue
            m, se = mean_sem(x)
            ax.plot(ws * 1e3, m, color=COL[co], ls=ls, lw=1, label=lab)
            ax.fill_between(ws * 1e3, m - se, m + se, color=COL[co], alpha=.15, lw=0, edgecolor="none")
            sel = (ws >= -0.05) & (ws <= 0.1)
            lo, hi = min(lo, np.nanmin((m - se)[sel])), max(hi, np.nanmax((m + se)[sel]))
        ax.axvspan(-20, 5, color="0.9", lw=0, edgecolor="none", zorder=0)
        ax.axvline(5, color="r", lw=0.5, ls="--")
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlim(-50, 100)
        ax.set_xlabel("Window start (ms from stimulus)")
        ax.set_ylabel("CCV minus baseline")
        ax.set_title(f"SSL {co}, {st} (n = {len(h)})", color=COL[co])
        if j == 0:
            ax.legend(frameon=False, loc="upper left")
    for ax in axes_b:
        if np.isfinite(lo):
            pad = 0.1 * (hi - lo)
            ax.set_ylim(lo - pad, hi + 2.5 * pad)
    # C, D
    rng = np.random.default_rng(0)
    elab = [f"{int(a * 1e3)}-{int(b * 1e3)}" for a, b in EPOCHS]
    for j, (pref, lab) in enumerate((("is", "S1-leading CCV (in-sample)\nminus baseline"),
                                     ("cv", "S1-leading CCV (cross-validated)\nminus baseline"))):
        ax = fig.add_subplot(g3[0, j])
        xt, xl, xc = [], [], []
        for gi, (st, co) in enumerate(GROUPS):
            h = df[(df.stage == st) & (df.cohort == co)]
            for ei in range(len(EPOCHS)):
                x0 = gi * 4 + ei
                v = h[f"{pref}_ff_e{ei}"].dropna()
                if not len(v):
                    continue
                ax.scatter(x0 + rng.uniform(-.15, .15, len(v)), v, s=3, c=COL[co], alpha=.35, lw=0)
                ax.errorbar(x0, v.mean(), v.std() / np.sqrt(len(v)), color=COL[co], capsize=1.5, lw=0.8, marker="o",
                            ms=3, mfc="white" if st == "learning" else COL[co])
                xt.append(x0); xl.append(elab[ei]); xc.append(COL[co])
                ax.text(x0, 0.99, fmt_p(wil(v)), transform=ax.get_xaxis_transform(), rotation=90, ha="center",
                        va="top", fontsize=4.5, color=COL[co])
            if len(h) == 0:
                continue
            ax.text(gi * 4 + 1, 1.01, f"{co} {st[:5]}.\nn={len(h)}", transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=5.5, color=COL[co])
        ax.axhline(0, color="k", lw=0.4)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.35 * (hi_ - lo_))  # room for the p-values
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=4.5, rotation=90)
        for t_, c_ in zip(ax.get_xticklabels(), xc):
            t_.set_color(c_)
        ax.set_xlabel("Window start epoch (ms)", fontsize=5.5)
        ax.set_ylabel(lab)
    ax = fig.add_subplot(g3[0, 2])
    xt, xl, xc = [], [], []
    for gi, (st, co) in enumerate(GROUPS):
        v = df[(df.stage == st) & (df.cohort == co)].is_ff_minus_fb.dropna()
        if not len(v):
            continue
        ax.scatter(gi + rng.uniform(-.12, .12, len(v)) - .15, v, s=4, c=COL[co], alpha=.4, lw=0)
        ax.errorbar(gi + .15, v.mean(), v.std() / np.sqrt(len(v)), color=COL[co], capsize=2, lw=0.8, marker="o", ms=3,
                    mfc="white" if st == "learning" else COL[co])
        xt.append(gi); xl.append(f"{co}\n{st[:5]}.\nn={len(v)}\n{fmt_p(wil(v))}"); xc.append(COL[co])
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=5)
    for t_, c_ in zip(ax.get_xticklabels(), xc):
        t_.set_color(c_)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylabel("S1 leading minus M2 leading\n(window starts 25-95 ms)")
    for y, l in ((0.925, "A"), (0.635, "B"), (0.365, "C")):
        fig.text(0.01, y, l, fontsize=9, fontweight="bold")
    fig.suptitle(f"{title}: S1 (SSp-bfd) -> M2 (MO-wM1 + MO-wM2), high-resolution time-shifted CCA, {kind} coupling",
                 fontsize=8, y=0.985)
    cap = ("1-ms bins, 10-ms windows stepped every 1 ms and labelled by their start, M2 window shifted by the delay "
           "(-20..+20 ms, 1-ms steps), first canonical correlation (CCV) after reducing each area (all stable good + MUA "
           "units) to its top 10 PCs. Only cells whose S1 and M2 windows are both clear of the -10..+5 ms artefact gap "
           "(grey) are used, so the first S1-leading window is 5-15 ms (red line) and the M2-leading side is clean from "
           "~25 ms. Baseline: windows starting -100..-25 ms, per delay. The paper's maps (50-ms windows) are not "
           "shown: different resolution. Session = unit of analysis; mean +- s.e.m. over sessions; p: Wilcoxon "
           "signed-rank vs 0, uncorrected. Open markers learning day, filled expert days.")
    fig.text(0.09, 0.012, "\n".join(textwrap.wrap(cap, 185)), fontsize=5.2, va="bottom")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="total")
    a = ap.parse_args()
    df, maps, ws, D = load(a.kind)
    out = L.OUT / "replication"
    out.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / f"hires_{a.kind}.pdf") as pdf:
        for cond, title in TRIAL_TYPES:
            fig = page(cond, title, a.kind, df, maps, ws, D)
            pdf.savefig(fig)
            fig.savefig(out / f"hires_{a.kind}_{cond}.png", dpi=300)
            plt.close(fig)
    df.drop(columns="traces").to_csv(out / f"hires_{a.kind}_per_session.csv", index=False)
    print(out / f"hires_{a.kind}.pdf")


if __name__ == "__main__":
    main()
