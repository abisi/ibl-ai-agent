"""CCF maps of the fraction of significant units per learning stage and cohort, for every ROC type / category.

Atlas: allen_mouse_bluebrain_barrels_10um_v1.0 (annotation.tiff: grey region contours with cortical layers and barrel
columns merged into their area, darker brain outline, white background; axis order AP, DV, ML; 10 um voxels). Units: ccf_atlas_ap / ml / dv (um), good units tested for the measure.
Slabs: 6 coronal (AP) and 4 sagittal (ML) slabs tiling the 2nd-98th percentile range of recorded positions; within a
slab all units are projected. Map = Gaussian-smoothed (sigma 150 um, 50 um grid) count of significant units /
smoothed count of tested units (metric frac_sig / frac_pos / frac_neg), or smoothed sum of |sel| / sel+ / sel- over
smoothed count of units with a selectivity (mean_abs_sel / mean_sel_pos / mean_sel_neg), shown where the smoothed
tested-unit density >= DENS_MIN (~3 nearby units); dots = recorded (tested) unit positions. Rows: R+ learning, R+
expert, R- learning, R- expert (single-hue white -> cohort colour, shared scale), then expert - learning (R+, R-) on a
diverging scale. Categories: frac_sig only.
Output: combined_results_ks4/_roc_stage_analysis/figures/ccf/<metric>/<measure>.{png,pdf} (density, trimmed to the brain)
Other styles (--styles): regions -> figures/ccf_regions/<metric>/: choropleth of the 047 area-level statistics
(area_acronym_custom; each layer-merged atlas region takes the majority custom label of the units inside it; light grey =
sampled but below the >= 10 units / >= 3 sessions threshold; Δ rows outline areas with stage perm. p < .05);
units -> figures/ccf_units/: every tested unit at its position (n.s. light grey; significant coloured by signed
selectivity, size ~ |sel|, strongest on top).
"""
import argparse
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
st47 = importlib.import_module("047_roc_stage_stats")
BASE = st47.BASE
FIG = BASE / "figures" / "ccf"
ATLAS = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/allen_mouse_bluebrain_barrels_10um_v1.0")
GRID = 50.0
SIGMA_UM = 150.0
DENS_MIN = 0.05
N_COR, N_SAG = 6, 4
ROWS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
COH = {"R+": "#00B400", "R-": "#C800C8"}


COH_DARK = {"R+": "#005A00", "R-": "#5A005A"}
# metric -> (unit weight function, denominator: all tested units or units with a selectivity value, label)
CCF_METRICS = {
    "frac_sig": (lambda m: m.flag.to_numpy(float), "all", "fraction significant"),
    "frac_pos": (lambda m: m.pos.to_numpy(float), "all", "fraction significant, positive"),
    "frac_neg": (lambda m: m.neg.to_numpy(float), "all", "fraction significant, negative"),
    "mean_abs_sel": (lambda m: m.abs_sel.to_numpy(float), "sel", "mean |selectivity|"),
    "mean_sel_pos": (lambda m: np.clip(m.sel.to_numpy(float), 0, None), "sel", "mean positive selectivity"),
    "mean_sel_neg": (lambda m: np.clip(-m.sel.to_numpy(float), 0, None), "sel", "mean negative selectivity (magnitude)"),
}


def structure_parent_map():
    """annotation id -> id used for contours: cortical layers and barrel columns merged into their area"""
    S = pd.read_csv(ATLAS / "structures.csv")
    name = dict(zip(S.id, S.name)); acr = dict(zip(S.id, S.acronym))
    par = dict(zip(S.id, S.parent_structure_id.fillna(-1).astype(int)))
    bfd = S.loc[S.acronym == "SSp-bfd", "id"]
    bfd = int(bfd.iloc[0]) if len(bfd) else None
    out = {}
    for i in S.id:
        j = i
        while j in name and ("layer" in str(name[j]).lower()) and par.get(j, -1) in name:
            j = par[j]
        if bfd is not None and str(acr.get(j, "")).startswith("SSp-bfd"):
            j = bfd
        out[i] = j
    return out


class Slabs:
    def __init__(self, xyz, uxyz=None):
        ann = tifffile.imread(ATLAS / "annotation.tiff")
        self.shape = ann.shape                                   # (AP, DV, ML) voxels
        pm = structure_parent_map()
        ap, ml, dv = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        self.cor = self._edges(ap, N_COR); self.sag = self._edges(ml, N_SAG)
        self.bg = {}; self.lab = {}
        for kind, edges in [("cor", self.cor), ("sag", self.sag)]:
            for i in range(len(edges) - 1):
                mid = int(round((edges[i] + edges[i + 1]) / 2 / 10))
                if kind == "cor":
                    a = ann[np.clip(mid, 0, self.shape[0] - 1)]                 # (DV, ML)
                else:
                    a = ann[:, :, np.clip(mid, 0, self.shape[2] - 1)].T         # (DV, AP)
                u, inv = np.unique(a, return_inverse=True)
                a = np.array([pm.get(int(k), int(k)) for k in u])[inv].reshape(a.shape)
                bnd = (np.diff(a, axis=0, prepend=a[:1]) != 0) | (np.diff(a, axis=1, prepend=a[:, :1]) != 0)
                bnd = ndimage.binary_dilation(bnd & (a != 0))
                self.bg[(kind, i)] = (bnd, ndimage.binary_fill_holes(a != 0))
                self.lab[(kind, i)] = a
        if uxyz is not None:                                     # collapsed atlas id at each unit position (0 = none)
            ok = np.isfinite(uxyz).all(1)
            ijk = np.zeros((len(uxyz), 3), int)
            ijk[ok] = np.round(uxyz[ok][:, [0, 2, 1]] / 10).astype(int)          # (AP, DV, ML) voxel
            ok &= (ijk >= 0).all(1) & (ijk < np.array(self.shape)).all(1)
            raw = np.zeros(len(uxyz), int); raw[ok] = ann[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            self.unit_id = np.array([pm.get(int(k), int(k)) for k in raw])
        del ann

    @staticmethod
    def _edges(v, n):
        lo, hi = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        return np.linspace(lo, hi, n + 1)


def smooth_map(x, y, w, xr, yr):
    bx = np.arange(xr[0], xr[1] + GRID, GRID); by = np.arange(yr[0], yr[1] + GRID, GRID)
    H, _, _ = np.histogram2d(y, x, bins=[by, bx], weights=w)
    return ndimage.gaussian_filter(H, SIGMA_UM / GRID), (bx[0], bx[-1], by[-1], by[0])


def draw_atlas(ax, sl, kind, i):
    bnd, inb = sl.bg[(kind, i)]
    ex = (0, (sl.shape[2] if kind == "cor" else sl.shape[0]) * 10, sl.shape[1] * 10, 0)
    rgba = np.zeros(bnd.shape + (4,)); rgba[..., :3] = 0.55; rgba[..., 3] = bnd * 0.55
    ax.imshow(rgba, extent=ex, interpolation="antialiased", zorder=2)
    yy = (np.arange(inb.shape[0]) + 0.5) * 10; xx = (np.arange(inb.shape[1]) + 0.5) * 10
    ax.contour(xx, yy, inb.astype(float), levels=[0.5], colors="0.3", linewidths=0.45, zorder=3)


def make(mname, U, meas, sl, out, metric="frac_sig"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 5.5, "axes.titlesize": 5.5, "pdf.fonttype": 42})
    wfun, denom, mlabel = CCF_METRICS[metric]
    xyz = U[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].to_numpy(float)
    v = meas.valid.to_numpy() & np.isfinite(xyz).all(1)
    w = wfun(meas)
    if denom == "sel":
        v = v & np.isfinite(w)
    w = np.nan_to_num(w)
    cmaps = {c: LinearSegmentedColormap.from_list(f"w{c}", ["#FFFFFF", COH[c], COH_DARK[c]]) for c in COH}
    ncol = N_COR + N_SAG
    fig, axs = plt.subplots(6, ncol, figsize=(7.4, 5.2), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    maps = {}
    panels = [("cor", i) for i in range(N_COR)] + [("sag", i) for i in range(N_SAG)]
    H, W = sl.shape[1] * 10, (sl.shape[2] * 10, sl.shape[0] * 10)
    for (c, g) in ROWS:
        m = v & (U.cohort == c).to_numpy() & (U.stage == g).to_numpy()
        for kind, i in panels:
            edges = sl.cor if kind == "cor" else sl.sag
            ax_ = 0 if kind == "cor" else 1
            inside = m & (xyz[:, ax_] >= edges[i]) & (xyz[:, ax_] < edges[i + 1])
            xr = (0, W[0]) if kind == "cor" else (0, W[1])
            xx = xyz[inside, 1] if kind == "cor" else xyz[inside, 0]
            yy = xyz[inside, 2]
            dall, ext = smooth_map(xx, yy, np.ones(inside.sum()), xr, (0, H))
            dsig, _ = smooth_map(xx, yy, w[inside], xr, (0, H))
            with np.errstate(invalid="ignore", divide="ignore"):
                val = np.where(dall >= DENS_MIN, dsig / dall, np.nan)
            inb = sl.bg[(kind, i)][1]                            # trim to the brain (grid centres inside the brain)
            yi = np.clip(((np.arange(val.shape[0]) + 0.5) * GRID / 10).astype(int), 0, inb.shape[0] - 1)
            xi = np.clip(((np.arange(val.shape[1]) + 0.5) * GRID / 10).astype(int), 0, inb.shape[1] - 1)
            val[~inb[np.ix_(yi, xi)]] = np.nan
            maps[(c, g, kind, i)] = (val, ext, xx, yy)
    vals = np.concatenate([f[0][np.isfinite(f[0])].ravel() for f in maps.values()] + [np.array([0.0])])
    vmax = max(np.nanpercentile(vals, 97), 0.02)
    dvals = []
    for c in ["R+", "R-"]:
        for kind, i in panels:
            a, b = maps[(c, "learning", kind, i)][0], maps[(c, "expert", kind, i)][0]
            dvals.append((b - a)[np.isfinite(b - a)].ravel())
    dmax = max(np.nanpercentile(np.abs(np.concatenate(dvals + [np.array([0.0])])), 97), 0.01)
    for r in range(6):
        for j, (kind, i) in enumerate(panels):
            ax = axs[r, j]; ax.axis("off"); ax.set_facecolor("white")
            if r < 4:
                c, g = ROWS[r]
                val, ext, xx, yy = maps[(c, g, kind, i)]
                ax.imshow(val, cmap=cmaps[c], vmin=0, vmax=vmax, extent=ext, interpolation="bilinear", zorder=1)
                if len(xx):
                    k = np.random.default_rng(0).choice(len(xx), min(len(xx), 600), replace=False)
                    ax.scatter(xx[k], yy[k], s=0.12, color="0.2", alpha=0.35, lw=0, rasterized=True, zorder=4)
            else:
                c = ["R+", "R-"][r - 4]
                a, ext = maps[(c, "learning", kind, i)][0], maps[(c, "learning", kind, i)][1]
                b = maps[(c, "expert", kind, i)][0]
                ax.imshow(b - a, cmap="RdBu_r", vmin=-dmax, vmax=dmax, extent=ext, interpolation="bilinear", zorder=1)
            draw_atlas(ax, sl, kind, i)
            allx = xyz[v, 1] if kind == "cor" else xyz[v, 0]
            ax.set_xlim(np.nanpercentile(allx, 0.5) - 800, np.nanpercentile(allx, 99.5) + 800)
            ax.set_ylim(sl.shape[1] * 10, 0)
            if r == 0:
                e = sl.cor if kind == "cor" else sl.sag
                ax.set_title(f"{'coronal, AP' if kind == 'cor' else 'sagittal, ML'}\n{e[i] / 1000:.1f}–{e[i + 1] / 1000:.1f} mm", fontsize=4.6)
            if j == 0:
                c = ROWS[r][0] if r < 4 else ["R+", "R-"][r - 4]
                lab = f"{c} {ROWS[r][1]}" if r < 4 else f"{c} Δ (exp.-learn.)"
                ax.text(-0.08, 0.5, lab.replace("-", "−"), transform=ax.transAxes, rotation=90,
                        ha="right", va="center", fontsize=5.5, color=COH[c])
    cb = []
    for k, c in enumerate(["R+", "R-"]):
        sm = plt.cm.ScalarMappable(cmap=cmaps[c], norm=plt.Normalize(0, vmax))
        ca = fig.add_axes([0.915 + 0.03 * k, 0.42, 0.008, 0.45]); cb.append(ca)
        bar = fig.colorbar(sm, cax=ca)
        if k == 1:
            bar.set_label(mlabel, fontsize=5)
        else:
            ca.set_yticks([])
        ca.set_title(c.replace("-", "−"), fontsize=5, color=COH[c])
    sm2 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-dmax, dmax))
    c2 = fig.add_axes([0.93, 0.12, 0.008, 0.22]); cb.append(c2)
    fig.colorbar(sm2, cax=c2).set_label("Δ " + mlabel.split(",")[0] + " (expert − learning)", fontsize=4.6)
    for c_ in cb:
        c_.tick_params(labelsize=4.5)
    fig.suptitle(f"{mname}: {mlabel} in CCF slabs (coronal | sagittal); dots = recorded units",
                 fontsize=6.5, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.03)
    safe = mname.replace(":", "_")
    d = out / metric; d.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(d / f"{safe}.{ext}", dpi=260, facecolor="white")
    plt.close(fig)


def panel_labels(sl, U, kind, i, gmaj):
    """custom-area label image for a section: each collapsed atlas region gets the majority area_acronym_custom of the
    units located in it (units inside the slab if >= 5, else all units); regions without units -> ''"""
    L = sl.lab[(kind, i)]
    u, inv = np.unique(L, return_inverse=True)
    edges = sl.cor if kind == "cor" else sl.sag
    col = "ccf_atlas_ap" if kind == "cor" else "ccf_atlas_ml"
    inslab = ((U[col] >= edges[i]) & (U[col] < edges[i + 1])).to_numpy()
    names = []
    for k in u:
        if k == 0:
            names.append(""); continue
        m = inslab & (sl.unit_id == k)
        if m.sum() >= 5:
            names.append(U.area_acronym_custom.to_numpy()[m].astype(str))
            names[-1] = pd.Series(names[-1]).mode().iloc[0]
        else:
            names.append(gmaj.get(int(k), ""))
    names = np.array(names, dtype=object)
    return names[inv].reshape(L.shape)


def setup_panel(ax, sl, kind, i, xyz, v, r, j, rowlab, rowcol):
    ax.axis("off")
    draw_atlas(ax, sl, kind, i)
    allx = xyz[v, 1] if kind == "cor" else xyz[v, 0]
    ax.set_xlim(np.nanpercentile(allx, 0.5) - 800, np.nanpercentile(allx, 99.5) + 800)
    ax.set_ylim(sl.shape[1] * 10, 0)
    if r == 0:
        e = sl.cor if kind == "cor" else sl.sag
        ax.set_title(f"{'coronal, AP' if kind == 'cor' else 'sagittal, ML'}\n{e[i] / 1000:.1f}–{e[i + 1] / 1000:.1f} mm", fontsize=4.6)
    if j == 0:
        ax.text(-0.08, 0.5, rowlab.replace("-", "−"), transform=ax.transAxes, rotation=90, ha="right", va="center",
                fontsize=5.5, color=rowcol)


def make_region(mname, U, meas, sl, out, metric, RS, gmaj):
    """choropleth of the area-level statistics (047 region_stats, area_acronym_custom) painted on atlas sections"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.size": 5.5, "pdf.fonttype": 42})
    _, _, mlabel = CCF_METRICS[metric]
    sc = 100.0 if metric.startswith("frac") else 1.0
    xyz = U[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].to_numpy(float)
    v = meas.valid.to_numpy() & np.isfinite(xyz).all(1)
    R = RS[(RS.measure == mname) & (RS.level == "area_acronym_custom")]
    st = {c: R[R.comparison == f"stage:{c}"].set_index("region") for c in COH}
    if not all(len(x) for x in st.values()) or f"{metric}_A" not in R:
        return False
    sampled = {c: set(U.area_acronym_custom[v & (U.cohort == c).to_numpy()].astype(str)) for c in COH}
    vals = {}
    for c in COH:
        inc = st[c][st[c].included.astype(bool)]
        vals[(c, "learning")] = (inc[f"{metric}_A"] * sc).to_dict()
        vals[(c, "expert")] = (inc[f"{metric}_B"] * sc).to_dict()
        vals[(c, "diff")] = (inc[f"{metric}_diff"] * sc).to_dict()
        vals[(c, "sig")] = set(inc.index[inc[f"{metric}_p"] < 0.05])
    lv = np.concatenate([list(vals[k].values()) for k in vals if k[1] in ("learning", "expert")] + [[0.0]])
    vmax = max(np.nanpercentile(lv, 97), 1e-3)
    dv_ = np.concatenate([list(vals[(c, "diff")].values()) for c in COH] + [[0.0]])
    dmax = max(np.nanpercentile(np.abs(dv_), 95), 1e-3)
    cmaps = {c: LinearSegmentedColormap.from_list(f"w{c}", ["#FFFFFF", COH[c], COH_DARK[c]]) for c in COH}
    panels = [("cor", i) for i in range(N_COR)] + [("sag", i) for i in range(N_SAG)]
    labs = {p: panel_labels(sl, U, *p, gmaj) for p in panels}
    fig, axs = plt.subplots(6, len(panels), figsize=(7.4, 5.2), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for r in range(6):
        if r < 4:
            c, g = ROWS[r]; cmap, norm = cmaps[c], plt.Normalize(0, vmax); d = vals[(c, g)]; lab = f"{c} {g}"
        else:
            c = ["R+", "R-"][r - 4]; cmap, norm = plt.get_cmap("RdBu_r"), plt.Normalize(-dmax, dmax)
            d = vals[(c, "diff")]; lab = f"{c} Δ (exp.-learn.)"
        for j, (kind, i) in enumerate(panels):
            ax = axs[r, j]
            A = labs[(kind, i)]
            inb = sl.bg[(kind, i)][1]
            u, inv = np.unique(A, return_inverse=True)
            V = np.array([d.get(a, np.nan) for a in u])[inv].reshape(A.shape)
            grey = np.array([(a in sampled[c]) and (a not in d) for a in u])[inv].reshape(A.shape)
            rgba = cmap(norm(np.nan_to_num(V)))
            rgba[~np.isfinite(V)] = (1, 1, 1, 1)
            rgba[grey] = (0.86, 0.86, 0.86, 1)
            rgba[~inb] = (1, 1, 1, 0)
            ex = (0, (sl.shape[2] if kind == "cor" else sl.shape[0]) * 10, sl.shape[1] * 10, 0)
            ax.imshow(rgba, extent=ex, interpolation="antialiased", zorder=1)
            if r >= 4 and vals[(c, "sig")]:
                sig = np.array([a in vals[(c, "sig")] for a in u])[inv].reshape(A.shape)
                if sig.any():
                    yy = (np.arange(sig.shape[0]) + 0.5) * 10; xx = (np.arange(sig.shape[1]) + 0.5) * 10
                    ax.contour(xx, yy, sig.astype(float), levels=[0.5], colors="k", linewidths=0.7, zorder=5)
            setup_panel(ax, sl, kind, i, xyz, v, r, j, lab, COH[c])
    cb = []
    for k, c in enumerate(["R+", "R-"]):
        ca = fig.add_axes([0.915 + 0.03 * k, 0.42, 0.008, 0.45]); cb.append(ca)
        bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmaps[c], norm=plt.Normalize(0, vmax)), cax=ca)
        if k == 1:
            bar.set_label(mlabel + (" (%)" if sc == 100 else ""), fontsize=5)
        else:
            ca.set_yticks([])
        ca.set_title(c.replace("-", "−"), fontsize=5, color=COH[c])
    c2 = fig.add_axes([0.93, 0.12, 0.008, 0.22]); cb.append(c2)
    fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-dmax, dmax)), cax=c2).set_label(
        "Δ (expert − learning)", fontsize=4.8)
    for c_ in cb:
        c_.tick_params(labelsize=4.5)
    fig.suptitle(f"{mname}: {mlabel} per area (area_acronym_custom)\n"
                 "areas with >= 10 units & >= 3 sessions per stage; light grey = sampled, below threshold; "
                 "Δ rows: black outline = stage permutation p < .05", fontsize=6, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.9, bottom=0.03)
    d = out.parent / "ccf_regions" / metric; d.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(d / f"{mname.replace(':', '_')}.{ext}", dpi=260, facecolor="white")
    plt.close(fig)
    return True


def make_units(mname, U, meas, sl, out):
    """single-unit map: every tested unit at its position; significant units coloured by signed selectivity and
    sized by |sel| (strongest on top); non-significant units small light grey"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.5, "pdf.fonttype": 42})
    xyz = U[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].to_numpy(float)
    v = meas.valid.to_numpy() & np.isfinite(xyz).all(1) & (sl.unit_id != 0)
    flag = meas.flag.to_numpy(bool); sel = meas.sel.to_numpy(float)
    signed = np.isfinite(sel).any() and not mname.startswith("cat:")
    SMAX = 0.5
    panels = [("cor", i) for i in range(N_COR)] + [("sag", i) for i in range(N_SAG)]
    fig, axs = plt.subplots(4, len(panels), figsize=(7.4, 3.7), gridspec_kw=dict(wspace=0.04, hspace=0.12))
    for r, (c, g) in enumerate(ROWS):
        m = v & (U.cohort == c).to_numpy() & (U.stage == g).to_numpy()
        for j, (kind, i) in enumerate(panels):
            ax = axs[r, j]
            edges = sl.cor if kind == "cor" else sl.sag
            a_ = 0 if kind == "cor" else 1
            ins = m & (xyz[:, a_] >= edges[i]) & (xyz[:, a_] < edges[i + 1])
            xx = xyz[:, 1] if kind == "cor" else xyz[:, 0]
            ns = ins & ~flag
            ax.scatter(xx[ns], xyz[ns, 2], s=0.12, color="0.75", lw=0, rasterized=True, zorder=3)
            sg = np.where(ins & flag)[0]
            if len(sg):
                if signed:
                    sg = sg[np.argsort(np.abs(sel[sg]))]
                    ax.scatter(xx[sg], xyz[sg, 2], s=0.15 + 2.2 * np.clip(np.abs(sel[sg]), 0, SMAX) / SMAX, c=sel[sg],
                               cmap="RdBu_r", vmin=-SMAX, vmax=SMAX, lw=0, alpha=0.85, rasterized=True, zorder=4)
                else:
                    ax.scatter(xx[sg], xyz[sg, 2], s=0.8, color=COH[c], lw=0, alpha=0.85, rasterized=True, zorder=4)
            setup_panel(ax, sl, kind, i, xyz, v, r, j, f"{c} {g}", COH[c])
            ax.text(0.98, 0.02, f"{ins.sum()}", transform=ax.transAxes, ha="right", va="bottom", fontsize=3.8, color="0.4")
    if signed:
        ca = fig.add_axes([0.925, 0.3, 0.008, 0.4])
        fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-SMAX, SMAX)), cax=ca).set_label(
            "selectivity of significant units", fontsize=5)
        ca.tick_params(labelsize=4.5)
    fig.suptitle(f"{mname}: single units (grey = n.s.; coloured = significant, size ∝ |sel|; number = units in slab)",
                 fontsize=6.5, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.9, top=0.86, bottom=0.03)
    d = out.parent / "ccf_units"; d.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(d / f"{mname.replace(':', '_')}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


def main(a):
    FIG.mkdir(parents=True, exist_ok=True)
    U = pd.read_parquet(BASE / "units.parquet")
    U = U[(U.quality_label == "good") & U.cohort.isin(["R+", "R-"])].reset_index(drop=True)
    U["uid"] = np.arange(len(U))
    L = pd.read_parquet(BASE / "roc_long.parquet"); L["cluster_id"] = L.cluster_id.astype(str)
    L = L.merge(U[st47.KEYS + ["uid"]], on=st47.KEYS, how="inner")
    U = U.set_index("uid")
    M = st47.build_measures(U, L)
    uxyz = U[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].to_numpy(float)
    xyz = uxyz[np.isfinite(uxyz).all(1)]
    print("coords range AP/ML/DV:", np.nanpercentile(xyz, [1, 99], axis=0).round(0).tolist(), flush=True)
    sl = Slabs(xyz, uxyz)
    print(f"units inside the brain: {(sl.unit_id != 0).mean():.3f}", flush=True)
    t = pd.DataFrame(dict(k=sl.unit_id, a=U.area_acronym_custom.astype(str).to_numpy()))
    gmaj = t[t.k != 0].groupby("k").a.agg(lambda x: x.mode().iloc[0]).to_dict()
    RS = pd.read_csv(BASE / "stats" / "region_stats.csv") if "regions" in a.styles else None
    print("atlas shape", sl.shape, "coronal edges", sl.cor.round(0).tolist(), "sagittal edges", sl.sag.round(0).tolist(), flush=True)
    for n in (a.measures or list(M)):
        for metric in a.metrics:
            if n.startswith("cat:") and metric != "frac_sig":
                continue
            if "density" in a.styles:
                make(n, U, M[n], sl, FIG, metric)
            if "regions" in a.styles:
                make_region(n, U, M[n], sl, FIG, metric, RS, gmaj)
            print("saved", n, metric, flush=True)
        if "units" in a.styles:
            make_units(n, U, M[n], sl, FIG)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measures", nargs="*", default=None)
    ap.add_argument("--metrics", nargs="*", default=list(CCF_METRICS))
    ap.add_argument("--styles", nargs="*", default=["regions", "units", "density"], help="regions | units | density")
    main(ap.parse_args())
