"""Supplementary Figure 2: example of probe motion (< 20 um) before and after correction (DREDge, spikeinterface).

Example: MH069 imec2, 7-12 min (the motion excursion of the session); picked by scan_motion.py (session max |displacement| < 20 um, largest 100-s range at mid
depths, motion coherent across depths). Detected peaks (spikeinterface locally_exclusive, threshold 8, monopolar
triangulation; <imec>/dredge/motion/peaks.npy + peak_locations.npy), top 50 % amplitudes in the band, shown as density (2-s x 2-um bins, smoothed, each time bin
scaled to its maximum) and as depth profiles of two epochs (before / during the motion excursion). Corrected
depth = depth - displacement(t, depth) (DREDge non-rigid motion, linear interpolation in time and depth).
Top: displacement at the band depths (excerpt; whole recording with the excerpt shaded). Bottom: one 100-um depth band (2.00 mm); left before, right after correction; the dashed
line on the left is band centre + estimated displacement.
Outputs: manuscript Figures/SuppFigure2_recording_<date>/SuppFigure2_motion_example.{pdf,svg,eps,ai,png,jpg} + csv
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python motion_example.py
"""

import datetime
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import paperstyle as ps  # noqa: E402

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi")
PROBE = ROOT / "data/MH069/MH069_20260125_094519/Ephys/catgt_MH069_g0/MH069_g0_imec2"
T0, DUR = 420.0, 300.0
BANDS = [1985.0]
TAG = ""
if len(sys.argv) > 2:  # preview of another example: <style> <probe dir> <t0 s> <band um> <tag>
    PROBE, T0, BANDS, TAG = Path(sys.argv[2]), float(sys.argv[3]), [float(sys.argv[4])], sys.argv[5]  # band centres (um, probe depth from the tip)
HALF = 50.0
AMP_PCT = 50  # top 50 % amplitudes per band used
TBIN, YBIN = 2.0, 2.0  # density bins (s, um)
def _mins(lo, hi):  # epoch label in session minutes, e.g. "7-9 min" / "9.3-11 min"
    f = lambda s: f"{(T0 + s) / 60:.1f}".rstrip("0").rstrip(".")  # noqa: E731
    return f"{f(lo)}-{f(hi)} min"


EPOCHS = [((lo, hi), _mins(lo, hi), ls) for (lo, hi), ls in
          [((0.0, 120.0), (0, (1, 0))), ((140.0, 240.0), (0, (3, 1.5)))]]  # s from T0
BAND_COLORS = ["#0072B2"]
STYLE = sys.argv[1] if len(sys.argv) > 1 else "density"  # "density" or "dots"
TRACK_BIN = 5.0  # s, bins for the tracked depth of the band
C_TRACK = "#D55E00"
OUT = ROOT / "manuscripts/Bisi_Hamon/Figures" / f"SuppFigure2_recording_{datetime.date.today():%Y-%m-%d}"


def load():
    d = PROBE / "dredge" / "motion"
    fs = json.load(open(d / "parameters.json"))["sampling_frequency"]
    pk = np.load(d / "peaks.npy", mmap_mode="r")
    lo, hi = np.searchsorted(pk["sample_index"], [T0 * fs, (T0 + DUR) * fs])
    p = np.array(pk[lo:hi])
    loc = np.array(np.load(d / "peak_locations.npy", mmap_mode="r")[lo:hi])
    D = np.load(d / "motion" / "displacement_seg0.npy")
    tb = np.load(d / "motion" / "temporal_bins_s_seg0.npy")
    yb = np.load(d / "motion" / "spatial_bins_um.npy")
    return p["sample_index"] / fs - T0, loc["y"], np.abs(p["amplitude"]), D, tb - T0, yb


def disp_at(t, y, D, tb, yb):
    Dt = np.stack([np.interp(t, tb, D[:, j]) for j in range(len(yb))], 1)
    j = np.clip(np.searchsorted(yb, y) - 1, 0, len(yb) - 2)
    w = np.clip((y - yb[j]) / (yb[j + 1] - yb[j]), 0, 1)
    i = np.arange(len(t))
    return Dt[i, j] * (1 - w) + Dt[i, j + 1] * w


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ps.set_nature_rc()
    t, y, a, D, tb, yb = load()
    yc = y - disp_at(t, y, D, tb, yb)
    tt = np.linspace(0, DUR, 400)

    W = 120.0
    L, R, GAP, TOP, BOT = 16.0, 2.0, 5.0, 6.0, 9.0
    TR_GAP, BAND_GAP = 11.0, 2.5
    if STYLE == "density":
        PG, PW = 4.0, 13.0  # gap and width of the depth-profile column
        cw = (W - L - R - GAP - PG - PW) / 2
    else:  # third square column: tracked shift vs estimated motion
        PG = 9.0
        cw = PW = (W - L - R - GAP - PG) / 3
    TR_H = bh = cw  # square panels
    H = TOP + TR_H + TR_GAP + len(BANDS) * bh + BAND_GAP * (len(BANDS) - 1) + BOT
    fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))

    def axmm(x, ytop, w, h):
        return fig.add_axes([x / W, 1 - (ytop + h) / H, w / W, h / H])

    # motion traces
    ax = axmm(L + cw + GAP, TOP, cw, TR_H)
    src = {"time_s": tt + T0}
    for c, col in zip(BANDS, BAND_COLORS):
        dd = disp_at(tt, np.full_like(tt, c), D, tb, yb)
        ax.plot(tt, dd, color=col, lw=0.8)
        src[f"displacement_um_at_{c:.0f}"] = dd
    ax.axhline(0, color="0.75", lw=0.4, zorder=0)
    ax.set_xlim(0, DUR)
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_title(f"Estimated motion, {T0 / 60:.0f}-{(T0 + DUR) / 60:.0f} min", loc="left", fontsize=7)

    rows = []
    TRK = {}
    for r, (c, col) in enumerate(zip(BANDS, BAND_COLORS)):
        ytop = TOP + TR_H + TR_GAP + r * (bh + BAND_GAP)
        for k, (yy, ttl) in enumerate(((y, "Before correction"), (yc, "After correction"))):
            m = np.flatnonzero(np.abs(yy - c) < HALF)
            m = m[a[m] >= np.percentile(a[m], AMP_PCT)]
            axb = axmm(L + k * (cw + GAP), ytop, cw, bh)
            from scipy.ndimage import gaussian_filter
            # tracked depth: peak of the smoothed depth histogram per TRACK_BIN, relative to the first 2 min
            tc_, pk_ = [], []
            for b0 in np.arange(0, DUR - 1e-9, TRACK_BIN):
                w = m[(t[m] >= b0) & (t[m] < b0 + TRACK_BIN)]
                if len(w) < 30:
                    continue
                hh, e = np.histogram(yy[w] - c, bins=np.arange(-HALF, HALF + 1e-9, 1.0))
                hh = gaussian_filter(hh.astype(float), 3.0)
                tc_.append(b0 + TRACK_BIN / 2)
                pk_.append(((e[:-1] + e[1:]) / 2)[np.argmax(hh)])
            tc_, pk_ = np.array(tc_), np.array(pk_)
            base = np.median(pk_[tc_ < 120])
            TRK[k] = (tc_, pk_ - base, base)
            if STYLE == "dots":
                axb.scatter(t[m], yy[m] - c, s=0.15, c="k", alpha=0.35, lw=0, rasterized=True)
                axb.plot(tc_, pk_, color=C_TRACK, lw=0.9)
            else:  # spike density per time bin, each time bin scaled to its maximum (removes firing-rate stripes)
                h, _, _ = np.histogram2d(t[m], yy[m] - c, bins=[np.arange(0, DUR + 1e-9, TBIN),
                                                             np.arange(-HALF, HALF + 1e-9, YBIN)])
                h = gaussian_filter(h, (0.7, 1.5))
                h /= h.max(1, keepdims=True) + 1e-12
                IM = axb.imshow(h.T, origin="lower", aspect="auto", extent=[0, DUR, -HALF, HALF], cmap="Greys", vmin=0,
                           vmax=1, interpolation="nearest")
            prof = {}
            for (lo, hi), lab, ls in EPOCHS:
                w = m[(t[m] >= lo) & (t[m] < hi)]
                pr, e = np.histogram(yy[w] - c, bins=np.arange(-HALF, HALF + 1e-9, YBIN))
                pr = gaussian_filter(pr.astype(float), 1.5)
                prof[lab] = pr / pr.max()
                rows.append(dict(band_um=c, corrected=bool(k), epoch=lab,
                                 peak_um=float(((e[:-1] + e[1:]) / 2)[np.argmax(pr)])))
            if r == 0 and STYLE == "density":
                if k == 0:
                    pax = axmm(L + 2 * cw + GAP + PG, ytop, PW, bh)
                    ymid = (np.arange(-HALF, HALF, YBIN) + YBIN / 2)
                for (lo, hi), lab, ls in EPOCHS:
                    pax.plot(prof[lab] + 1.3 * k, ymid, color="k" if k == 0 else col, lw=0.8, ls=ls)
            if k == 0:
                axb.plot(tt, disp_at(tt, np.full_like(tt, c), D, tb, yb), color=col, lw=0.8, ls=(0, (3, 2)))
            else:
                axb.axhline(0, color=col, lw=0.8, ls=(0, (3, 2)))
            axb.set_xlim(0, DUR)
            axb.set_ylim(-HALF, HALF)
            axb.set_xticks([])
            axb.set_yticks([])
            for s in axb.spines.values():
                s.set_visible(False)
            if r == 0:
                axb.set_title(ttl, fontsize=7)
            if k == 0:
                axb.text(-0.02, 0.5, f"{c / 1000:.2f} mm", transform=axb.transAxes, ha="right", va="center",
                         fontsize=6, color=col)
    if STYLE == "dots":
        sax = axmm(L + 2 * cw + GAP + PG, TOP + TR_H + TR_GAP, PW, bh)
        c = BANDS[0]
        for k, (colk, lab) in enumerate((("k", "Before"), (BAND_COLORS[0], "After"))):
            tc_, sh, _ = TRK[k]
            mo = disp_at(tc_, np.full_like(tc_, c), D, tb, yb)
            mo = mo - np.median(mo[tc_ < 120])
            sl_, ic_ = np.polyfit(mo, sh, 1)
            r_ = np.corrcoef(mo, sh)[0, 1]
            sax.scatter(mo, sh, s=4, color=colk, lw=0, alpha=0.8)
            xx = np.array([mo.min(), mo.max()])
            sax.plot(xx, sl_ * xx + ic_, color=colk, lw=0.8)
            sax.text(0.03, 0.97 - 0.09 * k, f"{lab}: slope {sl_:.2f}", transform=sax.transAxes, fontsize=5.5,
                     va="top", color=colk)
            rows.append(dict(band_um=c, corrected=bool(k), slope=sl_, r=r_, n_bins=len(tc_)))
        allv = np.r_[[np.r_[TRK[k][1], disp_at(TRK[k][0], np.full_like(TRK[k][0], c), D, tb, yb) - np.median(disp_at(TRK[k][0], np.full_like(TRK[k][0], c), D, tb, yb)[TRK[k][0] < 120])] for k in (0, 1)]].ravel()
        lim = [np.floor(allv.min()) - 1, np.ceil(allv.max()) + 1]
        sax.plot(lim, lim, color="0.7", lw=0.5, ls=":", zorder=0)
        sax.set_xlim(lim)
        sax.set_ylim(lim)
        sax.set_aspect("equal")
        sax.set_xlabel("Estimated motion (µm)", labelpad=1)
        sax.set_ylabel("Tracked depth shift (µm)", labelpad=1)
        sax.set_title(f"Per {TRACK_BIN:.0f}-s bin", fontsize=7)
        sax.spines[["top", "right"]].set_visible(False)
    else:
        pax.set_ylim(-HALF, HALF)
        pax.set_xlim(-0.1, 2.4)
        pax.axis("off")
        from matplotlib.transforms import blended_transform_factory
        bt = blended_transform_factory(pax.transData, pax.transAxes)  # x data, y axes fraction
        pax.text(0.5, -0.02, "Before", ha="center", va="top", fontsize=6, transform=bt)
        pax.text(1.8, -0.02, "After", ha="center", va="top", fontsize=6, color=BAND_COLORS[0], transform=bt)
        pax.set_title("Spike density\nby depth", fontsize=7)
        for j, ((lo, hi), lab, ls) in enumerate(EPOCHS):
            ly = -0.11 - 0.055 * j
            pax.plot([0, 0.5], [ly, ly], color="k", lw=0.8, ls=ls, clip_on=False, transform=bt)
            pax.text(0.6, ly, lab, fontsize=5.5, va="center", transform=bt)
    y0 = TOP + TR_H + TR_GAP
    y1 = y0 + len(BANDS) * bh + (len(BANDS) - 1) * BAND_GAP
    fig.text(1.5 / W, 1 - (y0 + y1) / 2 / H, "Depth along the probe", rotation=90, ha="left", va="center", fontsize=7)
    # scale bars (bottom-left band): 20 um, 10 s
    last = TOP + TR_H + TR_GAP + (len(BANDS) - 1) * (bh + BAND_GAP) + bh
    sb = axmm(L, last + 1.0, cw, 4.0)
    sb.set_xlim(0, DUR)
    sb.set_ylim(0, 1)
    sb.plot([0, 30], [0.8, 0.8], color="k", lw=1)
    sb.text(15, 0.35, "30 s", ha="center", va="top", fontsize=6)
    sb.axis("off")
    if STYLE == "density":  # what the grey level means
        cax = axmm(L + cw + GAP + 0.55 * cw, last + 1.5, 0.45 * cw, 1.3)
        cb = fig.colorbar(IM, cax=cax, orientation="horizontal", ticks=[0, 1])
        cb.outline.set_linewidth(0.4)
        cax.tick_params(labelsize=5.5, length=1.5, width=0.4, pad=1)
        cax.text(-0.04, 0.5, f"Spike density\n(norm. per {TBIN:.0f}-s bin)", transform=cax.transAxes, ha="right",
                 va="center", fontsize=6)
    axv = fig.axes[1 + 2 * (len(BANDS) - 1)]  # last band, before correction: vertical 20-um bar at its left edge
    axv.plot([-4.5, -4.5], [-HALF, -HALF + 20], color="k", lw=1, clip_on=False)
    axv.text(-7.5, -HALF + 10, "20 µm", rotation=90, ha="right", va="center", fontsize=6)
    # full-session overview (top left): displacement at the band depths, excerpt shaded
    ov = axmm(L, TOP, cw, TR_H)
    ov.axvspan(T0 / 60, (T0 + DUR) / 60, color="0.88", lw=0, zorder=0)
    for c, col in zip(BANDS, BAND_COLORS):
        ov.plot((tb + T0) / 60, disp_at(tb, np.full_like(tb, c), D, tb, yb), color=col, lw=0.5)
    ov.axhline(0, color="0.75", lw=0.4, zorder=0)
    ov.set_xlim(0, (tb[-1] + T0) / 60)
    ov.set_xlabel("Time in session (min)", labelpad=1)
    ov.set_title("Whole recording", loc="left", fontsize=7)
    ov.set_ylabel("Motion (µm)")
    lo = min(ax.get_ylim()[0], ov.get_ylim()[0]); hi = max(ax.get_ylim()[1], ov.get_ylim()[1])
    ax.set_ylim(lo, hi); ov.set_ylim(lo, hi)
    ov.spines[["top", "right"]].set_visible(False)

    name = "SuppFigure2_motion_example" + ("" if STYLE == "density" else "_dots") + (f"_{TAG}" if TAG else "")
    out = OUT if not TAG else OUT / "_explore"  # candidate examples next to the main figure
    out.mkdir(parents=True, exist_ok=True)
    ps.save_figure(fig, out, name)
    pd.DataFrame(src).to_csv(out / f"{name}_traces.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"{name}_stats.csv", index=False)
    print("saved", out / name)


if __name__ == "__main__":
    main()
