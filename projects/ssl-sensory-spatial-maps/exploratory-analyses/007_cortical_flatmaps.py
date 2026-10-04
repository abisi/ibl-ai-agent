"""007 -- All sensory quantifications on a flatmap of the isocortex (Allen dorsal-cortex flatmap), side by side.

Neurons: good + mua in the isocortex (merged atlas label under Isocortex), all sessions pooled, ML folded onto the right
hemisphere. Flatmap: iblatlas FlatMap('dorsal_cortex', 25 um): each flat pixel holds the 80 volume voxels sampled along
its cortical streamline (pia -> white matter). A neuron is placed at the flat pixel whose streamline contains its 25-um
voxel; neurons whose voxel is not sampled take the nearest sampled voxel within 100 um (others dropped). Area boundaries:
Allen labels (layers merged) at mid-depth of each streamline.
Quantities (as in 003): whisker / auditory responsiveness (selectivity, significant neurons), modality preference,
whisker / auditory response latency (responsive neurons), bimodal (fraction of responsive neurons).
Per quantity: top -- significant (or responsive) neurons coloured by the quantity, others light grey; bottom -- density:
sum of the quantity over neurons and number of neurons with a value on the flat grid, each smoothed with a 2-D Gaussian
(sigma 150 um along the surface), divided (normalised by the recorded-neuron density); shown where >= 3 neurons fall within
the kernel; lines: whisker and auditory projection zones (002, ZONE_PCT contour; a flat pixel is in a zone when >= half of
its streamline voxels are).
Output: combined_results_ks4/_sensory_spatial_maps/figures<ZTAG>/cortical_flatmaps.{png,pdf,svg} + flatmap_units.parquet
"""
import importlib
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m3 = importlib.import_module("003_spatial_maps")
S, OUT, FIG = m3.S, m3.OUT, m3.FIG
RES, SIGMA_UM, DENS_MIN, MAXD_VOX = 25.0, 150.0, 3.0, 4
QUANTS = ["whisker", "auditory", "modality", "latency_whisker", "latency_auditory", "bimodal"]
LAYER = re.compile(r"(1|2/3|2|3|4|5|6a|6b)$")


def flat_lookup(fm):
    """flat (row, col) for every sampled 25-um voxel of the right half; KD-tree on sampled voxels for misses"""
    F = fm.flatmap
    half = F.shape[1] // 2
    Fr = F[:, half:, :]
    pos = np.flatnonzero(Fr.ravel() > 0)
    lin = Fr.ravel()[pos]
    r, c, d = np.unravel_index(pos, Fr.shape)
    order = np.argsort(lin, kind="stable")
    lin, r, c, d = lin[order], r[order], c[order], d[order]
    first = np.r_[True, lin[1:] != lin[:-1]]                       # one flat pixel per voxel (first along the streamline)
    return dict(lin=lin[first], r=r[first], c=c[first], half=half, F=Fr)


def place_units(U, fm, L):
    ccf = np.c_[U.ml_f, U.ccf_atlas_ap, U.ccf_atlas_dv].astype(float)
    ijk = fm.bc.xyz2i(fm.ccf2xyz(ccf, ccf_order="mlapdv"))           # (ml, ap, dv) indices
    lin = np.ravel_multi_index((ijk[:, 1], ijk[:, 0], ijk[:, 2]), fm.image.shape, mode="clip")
    k = np.searchsorted(L["lin"], lin)
    k = np.minimum(k, len(L["lin"]) - 1)
    hit = L["lin"][k] == lin
    rr, cc = np.where(hit, L["r"][k], -1), np.where(hit, L["c"][k], -1)
    miss = np.where(~hit)[0]
    if len(miss):
        ap, ml, dv = np.unravel_index(L["lin"], fm.image.shape)
        tree = cKDTree(np.c_[ap, ml, dv])
        dist, j = tree.query(np.c_[ijk[miss, 1], ijk[miss, 0], ijk[miss, 2]], distance_upper_bound=MAXD_VOX)
        ok = np.isfinite(dist)
        rr[miss[ok]], cc[miss[ok]] = L["r"][j[ok]], L["c"][j[ok]]
    return rr, cc


def flat_labels(fm, L, depth=40):
    lin = L["F"][:, :, depth]
    ids = np.zeros(lin.shape, int)
    ok = lin > 0
    ids[ok] = fm.label.ravel()[lin[ok]]
    acr = np.array([LAYER.sub("", a) for a in fm.regions.acronym])
    names = np.where(ok, acr[ids], "")
    u, inv = np.unique(names, return_inverse=True)
    return u, inv.reshape(names.shape), ok


def flat_zone(fm, L, mask50):
    """fraction of each flat pixel's streamline voxels inside a 50-um CCF mask (AP, DV, ML)"""
    F = L["F"]
    ok = F > 0
    ap, ml, dv = np.unravel_index(F[ok], fm.image.shape)
    xyz = fm.bc.i2xyz(np.c_[ml, ap, dv])
    ccf = fm.xyz2ccf(xyz, ccf_order="apdvml")
    i = np.clip(np.floor(ccf / 50).astype(int), 0, np.array(mask50.shape) - 1)
    ml_f = np.where(i[:, 2] < 114, 2 * 114 - 1 - i[:, 2], i[:, 2])          # fold onto the right hemisphere
    val = np.zeros(F.shape, np.float32)
    val[ok] = mask50[i[:, 0], i[:, 1], np.clip(ml_f, 0, mask50.shape[2] - 1)]
    n = ok.sum(2)
    return np.where(n > 0, val.sum(2) / np.maximum(n, 1), 0) >= 0.5


def main():
    from iblatlas.flatmaps import FlatMap
    plt = S.setup()
    fm = FlatMap(flatmap="dorsal_cortex", res_um=int(RES))
    L = flat_lookup(fm)
    U = m3.load_units()
    A = m3.Atlas()
    U["atlas_id"] = A.unit_ids(U)
    iso = {k for k, p in A.path.items() if 315 in p}
    U = U[U.atlas_id.isin(iso)].reset_index(drop=True)
    U["flat_r"], U["flat_c"] = place_units(U, fm, L)
    print(f"isocortex neurons {len(U)}, placed {np.mean(U.flat_r >= 0):.3f}")
    U = U[U.flat_r >= 0].reset_index(drop=True)
    names, lab, inside = flat_labels(fm, L)
    Z = np.load(m3.ZONES_NPZ)
    filled = ndimage.binary_fill_holes(inside)
    zw = flat_zone(fm, L, Z["zone70_whisker"]) & filled          # clipped to the flat cortex
    za = flat_zone(fm, L, Z["zone70_auditory"]) & filled
    U.drop(columns=[c for c in U.columns if U[c].dtype == object and c not in ("mouse_id", "session_id", "electrode_group",
                                                                                 "cluster_id", "area_acronym_custom")]).to_parquet(
        OUT / f"flatmap_units{m3.ZTAG}.parquet")
    figure(plt, U, names, lab, inside, zw, za)
    print("ALL DONE", FIG / "cortical_flatmaps.png")


def boundaries(lab, inside):
    b = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
    return b & inside


def figure(plt, U, names, lab, inside, zw, za):
    import matplotlib.patheffects as pe
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    rows, cols = np.where(inside)
    r0, r1, c0, c1 = rows.min() - 5, rows.max() + 5, cols.min() - 5, cols.max() + 5
    crop = (slice(r0, r1), slice(c0, c1))
    bnd = boundaries(lab, inside)[crop]
    ins = inside[crop]
    H_, W_ = bnd.shape
    ncol = len(QUANTS) + 1
    W = S.W_IN
    pw = (W - 0.3) / ncol
    ph = pw * H_ / W_
    H = 0.55 + 2 * ph + 0.55
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, ncol, left=0.25 / W, right=1 - 0.05 / W, top=1 - 0.5 / H, bottom=0.5 / H, wspace=0.04, hspace=0.08)
    rgba_b = np.zeros(bnd.shape + (4,)); rgba_b[..., :3] = 0.55; rgba_b[..., 3] = bnd * 0.7
    y, x = U.flat_r.to_numpy() - r0, U.flat_c.to_numpy() - c0
    s = SIGMA_UM / RES
    norm = 2 * np.pi * s ** 2

    def base(ax):
        ax.imshow(rgba_b, interpolation="nearest", zorder=2)
        ax.contour(ndimage.binary_fill_holes(ins).astype(float), levels=[0.5], colors="0.3", linewidths=0.5, zorder=3)
        ax.set_xlim(-0.5, W_ - 0.5); ax.set_ylim(H_ - 0.5, -0.5); ax.set_aspect("equal"); ax.set_axis_off()

    def zones(ax):
        for z, col, ls in ((zw[crop], m3.SRC_STYLE["whisker"][0], "-"), (za[crop], m3.SRC_STYLE["auditory"][0], "--")):
            if z.any():
                ax.contour(ndimage.gaussian_filter(z.astype(float), 1.0), levels=[0.5], colors=[col], linewidths=0.7,
                           linestyles=[ls], zorder=5)

    # column 0: recorded neurons + area labels / projection zones
    ax = fig.add_subplot(gs[0, 0]); base(ax)
    ax.scatter(x, y, s=0.08, c="0.3", lw=0, zorder=4, rasterized=True)
    labc = lab[crop]
    for k, nm in enumerate(names):
        if not nm or nm in ("root",):
            continue
        yy, xx = np.where(labc == k)
        if len(yy) < 900:
            continue
        ax.text(np.median(xx), np.median(yy), nm, fontsize=3.4, ha="center", va="center", zorder=7,
                path_effects=[pe.withStroke(linewidth=1.0, foreground="white")])
    ax.set_title(f"Recorded neurons\n(n = {len(U)})", fontsize=5.4, pad=2)
    ax.annotate("", xy=(0.80, 0.30), xytext=(0.80, 0.16), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.2"))
    ax.text(0.80, 0.32, "A", transform=ax.transAxes, ha="center", va="bottom", fontsize=4.6)
    ax.annotate("", xy=(0.94, 0.16), xytext=(0.80, 0.16), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.2"))
    ax.text(0.96, 0.16, "L", transform=ax.transAxes, ha="left", va="center", fontsize=4.6)
    ax = fig.add_subplot(gs[1, 0]); base(ax)
    for z, col in ((zw[crop], m3.WH_C if hasattr(m3, "WH_C") else "#f7b519"), (za[crop], "#2c2cdb")):
        ax.imshow(np.ma.masked_where(~z, z.astype(float)), cmap=ListedColormap([col]), alpha=0.35, interpolation="nearest", zorder=1)
    ov = zw[crop] & za[crop]
    ax.imshow(np.ma.masked_where(~ov, ov.astype(float)), cmap=ListedColormap(["#7b3294"]), alpha=0.75, interpolation="nearest", zorder=1.2)
    ax.set_title(f"Projection zones ({m3.ZONE_PCT} %)", fontsize=5.4, pad=2)
    # quantities
    for j, qk in enumerate(QUANTS):
        q = m3.QUANT[qk]
        v, sig = m3.values(U, qk)
        if q["cmap"] == "modality":
            cmap = m3.cmap_modality()
        elif q["cmap"] == "inferno_light_fast":
            cmap = ListedColormap(plt.get_cmap("inferno_r")(np.linspace(0.1, 1.0, 256)))
        elif q["cmap"] == "bimodal":
            cmap = LinearSegmentedColormap.from_list("white_bimodal", ["#ffffff", "#c2a5cf", "#7b3294", "#40004b"])
        else:
            cmap = plt.get_cmap(q["cmap"])
        ax = fig.add_subplot(gs[0, j + 1]); base(ax)
        tested = np.isfinite(v) if not q.get("categorical") else (U.bimodal_cat.to_numpy() >= 0)
        ax.scatter(x[tested & ~sig], y[tested & ~sig], s=0.06, c="0.8", lw=0, zorder=3.5, rasterized=True)
        if q.get("categorical"):
            sc = ax.scatter(x[sig], y[sig], s=0.35, c="#7b3294", lw=0, zorder=4, rasterized=True)
        else:
            o = np.argsort(np.abs(v[sig]))
            sc = ax.scatter(x[sig][o], y[sig][o], s=0.35, c=v[sig][o], cmap=cmap, vmin=q["vmin"], vmax=q["vmax"], lw=0, zorder=4,
                            rasterized=True)
        ax.set_title(q["title"].replace(" (whisker and auditory)", "").replace(" responsiveness", "\nresponsiveness")
                     .replace(" response latency", "\nresponse latency").replace(" preference", "\npreference"),
                     fontsize=5.4, pad=2)
        ax.text(0.98, 0.02, f"{int(sig.sum())}", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.2, color="0.4")
        # density
        ok = np.isfinite(v)
        num = np.zeros((H_, W_)); den = np.zeros((H_, W_))
        yi, xi = y[ok].astype(int), x[ok].astype(int)
        good = (yi >= 0) & (yi < H_) & (xi >= 0) & (xi < W_)
        np.add.at(num, (yi[good], xi[good]), v[ok][good])
        np.add.at(den, (yi[good], xi[good]), 1)
        num, den = ndimage.gaussian_filter(num, s), ndimage.gaussian_filter(den, s)
        M = np.where((den * norm >= DENS_MIN) & ins, num / np.maximum(den, 1e-12), np.nan)
        ax = fig.add_subplot(gs[1, j + 1]); base(ax)
        dlo, dhi = m3.density_range(q)
        im = ax.imshow(M, cmap=cmap, vmin=dlo, vmax=dhi, interpolation="bilinear", zorder=1)
        zones(ax)
        cax = fig.add_axes([ax.get_position().x0 + 0.1 * pw / W, ax.get_position().y0 - 0.06 / H, 0.8 * pw / W, 0.045 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=4.0, length=1.2, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
        lbl = {"whisker": "selectivity", "auditory": "selectivity", "modality": "W < 0 < A", "latency_whisker": "ms",
               "latency_auditory": "ms", "bimodal": "bimodal fraction"}[qk]
        cb.set_label(lbl, fontsize=4.4, labelpad=1)
    fig.text(0.05 / W, 1 - 0.5 / H - ph / 2 / H, "Neurons", rotation=90, va="center", fontsize=6)
    fig.text(0.05 / W, 1 - 0.5 / H - 1.5 * ph / H, "Density", rotation=90, va="center", fontsize=6)
    h = [plt.Line2D([], [], color=m3.SRC_STYLE["whisker"][0], lw=1, label=f"whisker-cortex zone ({m3.ZONE_PCT} %)"),
         plt.Line2D([], [], color=m3.SRC_STYLE["auditory"][0], lw=1, ls="--", label=f"auditory-cortex zone ({m3.ZONE_PCT} %)"),
         plt.Rectangle((0, 0), 1, 1, color="#7b3294", alpha=0.75, lw=0, label="overlap")]
    fig.legend(handles=h, loc="lower left", ncol=1, frameon=False, fontsize=4.8, bbox_to_anchor=(0.25 / W, 0.0))
    fig.suptitle("Sensory responses across the isocortex (dorsal flatmap, right hemisphere; all sessions pooled)",
                 x=0.25 / W, y=1 - 0.05 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, FIG, "cortical_flatmaps")
    plt.close(fig)
    caption(U)


def caption(U):
    txt = f"""# Sensory responses across the isocortex (flatmap)

**Sensory responses across the isocortex.** Dorsal flatmap of the isocortex (Allen CCF streamline flatmap, iblatlas
FlatMap 'dorsal_cortex', 25 um; right hemisphere, neurons folded onto it; anterior up, lateral right). Neurons: good and
multi-unit clusters in the isocortex, all sessions (both cohorts, learning day and expert days; n = {len(U)},
{U.session_id.nunique()} sessions). Each neuron is placed at the flat pixel whose cortical streamline contains its 25-um
voxel (nearest sampled voxel within 100 um otherwise); neurons of one probe therefore form streaks along the flat surface.
Top row: neurons with a significant response (responsiveness: rate-based ROC, 5-35 ms after stimulus vs pre-trial
baseline, permutation p < 0.05; modality preference: whisker vs auditory, active trials; latencies: half-time to peak of
responsive neurons; bimodal: responsive to both modalities, Bonferroni over the modality's active / passive-pre /
passive-post tests in the session) coloured by the quantity; light grey: tested, not significant; grey number: neurons
shown in colour. First column, top: all recorded neurons with Allen area labels (layers merged, mid-depth).
Bottom row: density -- sum of the quantity over neurons and number of neurons with a value on the flat grid, each
smoothed with a 2-D Gaussian (sigma 150 um along the surface) and divided (normalised by the recorded-neuron density);
shown where >= 3 neurons fall within the kernel. Lines: whisker-cortex (SSp-bfd + SSs, teal) and auditory-cortex
(AUDp + AUDd/v, brown dashed) projection zones ({m3.ZONE_PCT} % contours of the Allen anterograde projection density; a
flat pixel belongs to a zone when >= half of its streamline voxels do; the source areas are excluded from their own zone).
First column, bottom: whisker zone (yellow), auditory zone (blue), overlap (purple). The dorsal flatmap truncates the most
lateral cortex (ventral auditory and temporal association areas), where much of the auditory zone lies.
"""
    (OUT / f"cortical_flatmaps_caption{m3.ZTAG}.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
