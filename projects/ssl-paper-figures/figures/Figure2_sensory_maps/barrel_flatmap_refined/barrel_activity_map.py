"""Barrel-column activity maps on the Blue Brain barrel flatmap: global (CCF annotation at the unit) vs refined (D2,
user-picked) barrel identity, same units (SSp-bfd units of the 27 use_local mice)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path.home() / "code/wS1_barrel_registration"))
from barrelreg.atlas import BARREL_NAMES, SEPTA  # noqa: E402

FIG = Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/Figure2_sensory_maps_explore_2026-11-17/G_maps")
OUT = FIG / "barrel_flatmap_refined"
T = np.load("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ws1-barrel-registration/D2_all_brains/template.npz")
glo, pix, alab = T["lo"], float(T["pix"]), T["alab"]
MIN_N = 1
C_W, C_A = "#E69F00", "#0072B2"

s = pd.read_csv(OUT / "barrel_refined_units.csv")
for c in ("barrel_global", "barrel_refined"):
    s[c] = s[c].fillna("")
# final = refined where the local fit was accepted (use_local, unit matched), else the global fit (use_global / redo)
loc = s.ref_ap.notna()
s["source"] = np.where(loc, "local", "global")
s["barrel_final"] = np.where(loc, s.barrel_refined, s.barrel_global)
s["e1_final"] = np.where(loc, s.e1_refined, s.e1_global)
s["e2_final"] = np.where(loc, s.e2_refined, s.e2_global)
s.to_csv(OUT / "barrel_final_units.csv", index=False)
print("units per source:", s.source.value_counts().to_dict(), "mice:", s.groupby("source").mouse_id.nunique().to_dict())
s["w_exc"] = s.sig_whisker_active.fillna(False).astype(bool) & (s.sel_whisker_active > 0)
s["a_exc"] = s.sig_auditory_active.fillna(False).astype(bool) & (s.sel_auditory_active > 0)
inb = lambda c: s[c] != ""
print(f"{len(s)} units, {s.mouse_id.nunique()} mice; in a barrel column: global {inb('barrel_global').sum()}, "
      f"refined {inb('barrel_final').sum()}; both in a column: {(inb('barrel_global') & inb('barrel_final')).sum()}, "
      f"same barrel {((s.barrel_global == s.barrel_final) & inb('barrel_final')).sum()}")

cols = np.where((alab > 0) & (alab < SEPTA), alab, 0)
field = alab > 0
gi, gj = np.meshgrid(np.arange(alab.shape[0] + 1), np.arange(alab.shape[1] + 1), indexing="ij")
G1, G2 = glo[0] + gi * pix, glo[1] + gj * pix
cen = {BARREL_NAMES[k - 1]: glo + (np.argwhere(cols == k).mean(0) + 0.5) * pix for k in np.unique(cols[cols > 0])}
av = np.mean([cen[f"{r}4"] - cen[f"{r}1"] for r in "BCD"], 0)
ang = np.arctan2(av[1], av[0])
Rm = np.array([[np.cos(ang), np.sin(ang)], [-np.sin(ang), np.cos(ang)]])
fy = -1.0 if (Rm @ (cen["E2"] - cen["A2"]))[1] > 0 else 1.0


def rot(p1, p2):
    q = np.tensordot(Rm, np.stack([np.asarray(p1, float), np.asarray(p2, float)]), 1)
    return q[0] / 1000, fy * q[1] / 1000


GX, GY = rot(G1, G2)
edges = (ndimage.grey_dilation(cols, 3) != ndimage.grey_erosion(cols, 3)) & field
FX, FY = rot(*(glo[:, None] + (np.argwhere(field).T + 0.5) * pix))
XLIM, YLIM = np.array([FX.min(), FX.max()]) + [-0.2, 0.2], np.array([FY.min(), FY.max()]) + [-0.2, 0.2]
GREEK = {"Alpha": "α", "Beta": "β", "Gamma": "γ", "Delta": "δ"}


def frame(ax):
    ax.set_aspect("equal"); ax.set_xlim(*XLIM); ax.set_ylim(*YLIM); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def outlines(ax, labels=True):
    ed = np.ma.masked_where(~edges, np.ones(alab.shape))
    ax.pcolormesh(GX, GY, ed, cmap=matplotlib.colors.ListedColormap(["0.3"]), rasterized=True, zorder=3)
    if labels:
        for nm, c in cen.items():
            ax.text(*rot(c[0], c[1]), GREEK.get(nm, nm), fontsize=3, ha="center", va="center", color="0.2", zorder=4)


def choropleth(ax, vals, cmap, vmin, vmax, n, lab=None, min_n=MIN_N, norm=None):
    img = np.full(alab.shape, np.nan)
    for k in np.unique(cols[cols > 0]):
        nm = BARREL_NAMES[k - 1]
        if n.get(nm, 0) >= min_n:
            img[cols == k] = vals[nm]
    under = np.ma.masked_where(~(cols > 0), np.ones(alab.shape))
    ax.pcolormesh(GX, GY, under, cmap=matplotlib.colors.ListedColormap(["0.94"]), rasterized=True, zorder=0)
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    m = ax.pcolormesh(GX, GY, np.ma.masked_invalid(img), cmap=cmap, rasterized=True, zorder=1, **kw)
    outlines(ax, labels=False)
    for nm, c in cen.items():
        k = (lab if lab is not None else n).get(nm, 0)
        ax.text(*rot(c[0], c[1]), f"{GREEK.get(nm, nm)}\n{k}" if k else GREEK.get(nm, nm), fontsize=2.6, ha="center",
                va="center", color="0.15", zorder=4, linespacing=0.9)
    return m


plt.rcParams.update({"font.size": 6, "font.family": "Arial"})
fig, axs = plt.subplots(2, 7, figsize=(183 / 25.4, 62 / 25.4), gridspec_kw=dict(hspace=0.3, wspace=0.08))
cm_w = matplotlib.colors.LinearSegmentedColormap.from_list("w", ["#fff6e5", C_W, "#5a3d00"])
cm_a = matplotlib.colors.LinearSegmentedColormap.from_list("a", ["#e8f2fa", C_A, "#002a45"])
rows = [("barrel_global", "e1_global", "e2_global", "Global registration (CCF annotation at the unit)"),
        ("barrel_final", "e1_final", "e2_final", "Refined where accepted (D2 local fit), global fit otherwise")]
for r, (bc, c1, c2, name) in enumerate(rows):
    g = s[s[bc] != ""].groupby(bc)
    n = g.size().to_dict()
    fw, fa, mw = g.w_exc.mean(), g.a_exc.mean(), g.sel_whisker_active.mean()
    ma = g.sel_auditory_active.mean()
    ax = axs[r, 0]
    x, y = rot(s[c1], s[c2])
    outlines(ax)
    w = s.w_exc.to_numpy()
    ax.scatter(x[~w], y[~w], s=0.12, c="0.75", lw=0, rasterized=True, zorder=2)
    ax.scatter(x[w], y[w], s=0.25, c=C_W, lw=0, rasterized=True, zorder=2)
    ax.set_title("Neurons (orange:\nwhisker-excited)", fontsize=5)
    nm_ = g.mouse_id.nunique().to_dict()
    m4 = choropleth(axs[r, 1], {k: v for k, v in n.items()}, matplotlib.colors.LinearSegmentedColormap.from_list("g", ["0.97", "0.5"]), None, None, n, min_n=1,
                    norm=matplotlib.colors.LogNorm(5, 1000))
    axs[r, 1].set_title("Coverage: neurons", fontsize=5)
    m5 = choropleth(axs[r, 2], nm_, "Greens", 0, 12, n, lab=nm_, min_n=1)
    axs[r, 2].set_title("Coverage: mice", fontsize=5)
    m1 = choropleth(axs[r, 3], fw, cm_w, 0, 0.8, n)
    axs[r, 3].set_title("Fraction whisker-excited", fontsize=5)
    m2 = choropleth(axs[r, 4], mw, "bwr", -0.2, 0.2, n)
    axs[r, 4].set_title("Mean whisker selectivity", fontsize=5)
    m3 = choropleth(axs[r, 5], fa, cm_a, 0, 0.5, n)
    axs[r, 5].set_title("Fraction auditory-excited", fontsize=5)
    m6 = choropleth(axs[r, 6], ma, "bwr", -0.2, 0.2, n)
    axs[r, 6].set_title("Mean auditory selectivity", fontsize=5)
    for ax in axs[r]:
        frame(ax)
    fig.text(0.1, [0.99, 0.53][r], f"{name}: {sum(n.values())} neurons in a barrel column", fontsize=6)
    if r == 1:
        for ax, m, lab in [(axs[r, 1], m4, "neurons"), (axs[r, 2], m5, "mice"), (axs[r, 3], m1, "fraction"),
                          (axs[r, 4], m2, "ROC selectivity"), (axs[r, 5], m3, "fraction"), (axs[r, 6], m6, "ROC selectivity")]:
            cb = fig.colorbar(m, ax=ax, orientation="horizontal", fraction=0.05, pad=0.02, shrink=0.75)
            cb.set_label(lab, fontsize=5); cb.ax.tick_params(labelsize=4.5, length=1.5)
fig.text(0.5, -0.06, f"All {len(s)} SSp-bfd units of the barrel row, {s.mouse_id.nunique()} mice (same units in both rows); bottom row: "
         f"{(s.source == 'local').sum()} units ({s[s.source == 'local'].mouse_id.nunique()} mice) with the local fit, "
         f"{(s.source == 'global').sum()} ({s[s.source == 'global'].mouse_id.nunique()} mice) with the global fit. "
         f"Numbers: neurons per barrel column (mice in the mice panel); every barrel with at least one neuron is filled (no cutoff). Whisker-/auditory-excited: ROC "
         "significant with selectivity > 0 (active trials). Arcs 1-8 left to right, rows A-E top to bottom.",
         fontsize=4.5, ha="center", wrap=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"barrel_activity_map.{ext}", dpi=600, bbox_inches="tight")
print("saved", OUT)
