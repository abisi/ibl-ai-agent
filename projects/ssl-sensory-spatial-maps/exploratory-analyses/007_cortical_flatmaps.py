"""007 -- All sensory quantifications on a flatmap of the isocortex (Allen CCF butterfly flatmap), side by side.

Flatmap (user, 2026-10-04): the Allen CCFv3 "butterfly" flatmap (Wang et al. 2020 Cell; Harris et al. 2019 Nature),
built from cortical streamlines (Laplace solution between pia and white matter) and geodesic distances on the cortical
surface to two pairs of anchor points; it unfolds the whole isocortex (lateral and medial walls included). Allen's
ccf_streamlines package and assets (download.alleninstitute.org/.../cortical_coordinates/ccf_2017/ccf_streamlines_assets,
copied to Anatomy/ccf_streamlines_assets): each neuron's CCF position is assigned to its closest streamline and placed at
that streamline's flatmap position (IsocortexCoordinateProjector). Left hemisphere shown (neurons folded onto it),
anterior up, lateral left, as seen from above; flatmap units 10 um. The embedding does not preserve area (frontal pole and
medial-posterior cortex are distorted): densities are computed on the flatmap for display only.
Neurons: good + mua in the isocortex (merged atlas label under Isocortex), all sessions pooled.
Quantities (as in 003): whisker / auditory responsiveness (selectivity, significant neurons), modality preference,
whisker / auditory response latency (responsive neurons), bimodal (fraction of responsive neurons).
Per quantity: top -- significant (or responsive) neurons coloured by the quantity, others light grey; bottom -- density:
sum of the quantity over neurons and number of neurons with a value on the flat grid, each smoothed with a 2-D Gaussian
(sigma 150 um), divided (normalised by the recorded-neuron density); shown inside the isocortex where >= 3 neurons fall
within the kernel; lines: whisker and auditory projection zones (002, ZONE_PCT contour), projected with the same streamlines
(a flat pixel belongs to a zone when >= half of its streamline lies in it).
Output: combined_results_ks4/_sensory_spatial_maps/figures<ZTAG>/cortical_flatmaps{,_nozones}.{png,pdf,svg} (_nozones: no zone
contours; recorded-neuron density instead of the zone panel),
cortical_flatmaps_caption<ZTAG>.md, flatmap_units<ZTAG>.parquet
"""
import importlib
import pathlib
import sys

import numpy as np
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "py_extra"))          # ccf_streamlines (installed with --target)
m3 = importlib.import_module("003_spatial_maps")
S, OUT, FIG = m3.S, m3.OUT, m3.FIG
ASSETS = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/ccf_streamlines_assets")
PX_UM, SIGMA_UM, DENS_MIN, MID = 10.0, 150.0, 3.0, 5700.0
QUANTS = ["whisker", "auditory", "modality", "latency_whisker", "latency_auditory", "bimodal"]
VIEW = dict(view_space_for_other_hemisphere="flatmap_butterfly")


def projector():
    import ccf_streamlines.projection as ccfproj
    return ccfproj.IsocortexCoordinateProjector(
        projection_file=str(ASSETS / "flatmap_butterfly.h5"), surface_paths_file=str(ASSETS / "surface_paths_10_v3.h5"),
        closest_surface_voxel_reference_file=str(ASSETS / "closest_surface_voxel_lookup.h5"),
        streamline_layer_thickness_file=str(ASSETS / "cortical_layers_10_v2.h5"))


def boundaries():
    import ccf_streamlines.projection as ccfproj
    bf = ccfproj.BoundaryFinder(projected_atlas_file=str(ASSETS / "flatmap_butterfly.nrrd"),
                                labels_file=str(ASSETS / "labelDescription_ITKSNAPColor.txt"))
    return {k: np.asarray(v) for k, v in bf.region_boundaries(hemisphere="left", **VIEW).items()}


def zone_flat(mask50):
    """project a 50-um CCF mask (right hemisphere) mirrored onto the left hemisphere; fraction of each streamline inside"""
    import ccf_streamlines.projection as ccfproj
    p2 = ccfproj.Isocortex2dProjector(str(ASSETS / "flatmap_butterfly.h5"), str(ASSETS / "surface_paths_10_v3.h5"),
                                      hemisphere="left", **VIEW)
    m = mask50[:, :, ::-1]                                           # mirror ML: right -> left hemisphere
    vol = np.repeat(np.repeat(np.repeat(m.astype(np.float32), 5, 0), 5, 1), 5, 2)[:1320, :800, :1140]
    return np.asarray(p2.project_volume(vol, kind="mean"))


def main():
    plt = S.setup()
    U = m3.load_units()
    A = m3.Atlas()
    U["atlas_id"] = A.unit_ids(U)
    iso = {k for k, p in A.path.items() if 315 in p}
    U = U[U.atlas_id.isin(iso)].reset_index(drop=True)
    ml_left = MID - (U.ml_f.to_numpy() - MID)                        # fold onto the left hemisphere
    P = projector().project_coordinates(np.c_[U.ccf_atlas_ap, U.ccf_atlas_dv, ml_left].astype(float), scale="voxels",
                                        hemisphere="left", **VIEW)
    U["flat_x"], U["flat_y"], U["flat_depth"] = P[:, 0], P[:, 1], P[:, 2]
    ok = np.isfinite(P).all(1)
    print(f"isocortex neurons {len(U)}, projected {ok.mean():.4f}")
    U = U[ok].reset_index(drop=True)
    B = boundaries()
    Z = np.load(m3.ZONES_NPZ)
    zw, za = zone_flat(Z["zone70_whisker"]), zone_flat(Z["zone70_auditory"])
    print("zone image shape", zw.shape)
    keep = [c for c in U.columns if U[c].dtype != object or c in ("mouse_id", "session_id", "electrode_group", "cluster_id",
                                                                    "area_acronym_custom")]
    U[keep].to_parquet(OUT / f"flatmap_units{m3.ZTAG}.parquet")
    figure(plt, U, B, zw, za)
    figure(plt, U, B, zw, za, show_zones=False)                      # sensory-coding slides / report: no anatomy yet
    caption(U)
    print("ALL DONE", FIG / "cortical_flatmaps.png")


def figure(plt, U, B, zw, za, show_zones=True):
    import matplotlib.patheffects as pe
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm
    from matplotlib.path import Path
    allp = np.vstack(list(B.values()))
    x0, x1 = int(allp[:, 0].min()) - 10, int(allp[:, 0].max()) + 10
    y0, y1 = int(allp[:, 1].min()) - 10, int(allp[:, 1].max()) + 10
    nx, ny = x1 - x0, y1 - y0
    gx, gy = np.meshgrid(np.arange(nx) + x0 + 0.5, np.arange(ny) + y0 + 0.5)
    pts = np.c_[gx.ravel(), gy.ravel()]
    inside = np.zeros(len(pts), bool)
    for v in B.values():
        inside |= Path(v).contains_points(pts)
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(inside.reshape(ny, nx), iterations=4))   # close gaps between areas

    def zcrop(z):
        """projector image is indexed [x, y] like the coordinates -> crop to [y, x]"""
        zz = np.asarray(z).T
        out = np.zeros((ny, nx), bool)
        ys, xs = np.arange(ny) + y0, np.arange(nx) + x0
        okx, oky = (xs >= 0) & (xs < zz.shape[1]), (ys >= 0) & (ys < zz.shape[0])
        out[np.ix_(oky, okx)] = zz[np.ix_(ys[oky], xs[okx])] >= 0.5
        return out & inside
    zW, zA = zcrop(zw), zcrop(za)
    x = U.flat_x.to_numpy() - x0
    y = U.flat_y.to_numpy() - y0
    ncol = len(QUANTS) + 1
    W = S.W_IN
    pw = (W - 0.3) / ncol
    ph = pw * ny / nx
    H = 0.55 + 2 * ph + 0.6
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, ncol, left=0.25 / W, right=1 - 0.05 / W, top=1 - 0.5 / H, bottom=0.55 / H, wspace=0.04,
                          hspace=0.1)
    s = SIGMA_UM / PX_UM
    norm = 2 * np.pi * s ** 2

    def base(ax):
        for v in B.values():
            ax.plot(v[:, 0] - x0, v[:, 1] - y0, color="0.7", lw=0.25, zorder=2)
        ax.contour(inside.astype(float), levels=[0.5], colors="0.25", linewidths=0.6, zorder=3)
        ax.set_xlim(0, nx); ax.set_ylim(ny, 0); ax.set_aspect("equal"); ax.set_axis_off()

    def zones(ax):
        if not show_zones:
            return
        for z, col, ls in ((zW, m3.SRC_STYLE["whisker"][0], "-"), (zA, m3.SRC_STYLE["auditory"][0], "--")):
            if z.any():
                ax.contour(ndimage.gaussian_filter(z.astype(float), 3), levels=[0.5], colors=[col], linewidths=0.7,
                           linestyles=[ls], zorder=5)

    # column 0: recorded neurons with area labels; projection zones
    ax = fig.add_subplot(gs[0, 0]); base(ax)
    ax.scatter(x, y, s=0.05, c="0.3", lw=0, zorder=4, rasterized=True)
    for nm, v in B.items():
        if len(v) < 450 or nm.startswith(("VISp", "SSp-un", "SSp-tr", "AUDpo", "VISli", "VISpor", "VISpl")) and nm != "VISp":
            continue
        cx, cy = v[:, 0].mean(), v[:, 1].mean()
        ax.text(cx - x0, cy - y0, nm, fontsize=3.0, ha="center", va="center", zorder=7,
                path_effects=[pe.withStroke(linewidth=0.9, foreground="white")])
    ax.set_title(f"Recorded neurons\n(n = {len(U)})", fontsize=5.4, pad=2)
    ax.annotate("", xy=(0.93, 0.95), xytext=(0.93, 0.80), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(0.93, 0.97, "A", transform=ax.transAxes, ha="center", va="bottom", fontsize=4.6)
    ax.annotate("", xy=(0.78, 0.80), xytext=(0.93, 0.80), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.text(0.76, 0.80, "L", transform=ax.transAxes, ha="right", va="center", fontsize=4.6)
    ax = fig.add_subplot(gs[1, 0]); base(ax)
    if not show_zones:                                               # recorded-neuron density (sampling) instead of the zones
        xi, yi = x.astype(int), y.astype(int)
        g = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        den = np.zeros((ny, nx))
        np.add.at(den, (yi[g], xi[g]), 1)
        den = ndimage.gaussian_filter(den, s) * norm
        im = ax.imshow(np.where(inside & (den >= DENS_MIN), den, np.nan), cmap="Greys", norm=LogNorm(
            vmin=DENS_MIN, vmax=np.nanpercentile(np.where(den >= DENS_MIN, den, np.nan), 99)), interpolation="bilinear", zorder=1)
        ax.set_title("Recorded-neuron density", fontsize=5.4, pad=2)
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 - 0.07 / H, 0.8 * pos.width, 0.045 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=4.0, length=1.2, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
        cb.set_label("neurons per kernel", fontsize=4.4, labelpad=1)
    for z, col in (((zW, "#f7b519"), (zA, "#2c2cdb")) if show_zones else ()):
        ax.imshow(np.ma.masked_where(~z, z.astype(float)), cmap=ListedColormap([col]), alpha=0.35, interpolation="nearest", zorder=1)
    ov = zW & zA & show_zones
    ax.imshow(np.ma.masked_where(~ov, ov.astype(float)), cmap=ListedColormap(["#7b3294"]), alpha=0.75, interpolation="nearest",
              zorder=1.2)
    if show_zones:
        ax.set_title(f"Projection zones ({m3.ZONE_PCT} %)", fontsize=5.4, pad=2)
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
        ax.scatter(x[tested & ~sig], y[tested & ~sig], s=0.04, c="0.8", lw=0, zorder=3.5, rasterized=True)
        if q.get("categorical"):
            ax.scatter(x[sig], y[sig], s=0.25, c="#7b3294", lw=0, zorder=4, rasterized=True)
        else:
            o = np.argsort(np.abs(v[sig]))
            ax.scatter(x[sig][o], y[sig][o], s=0.25, c=v[sig][o], cmap=cmap, vmin=q["vmin"], vmax=q["vmax"], lw=0, zorder=4,
                       rasterized=True)
        ax.set_title(q["title"].replace(" (whisker and auditory)", "").replace(" responsiveness", "\nresponsiveness")
                     .replace(" response latency", "\nresponse latency").replace(" preference", "\npreference"), fontsize=5.4, pad=2)
        ax.text(0.98, 0.02, f"{int(sig.sum())}", transform=ax.transAxes, ha="right", va="bottom", fontsize=4.2, color="0.4")
        okv = np.isfinite(v)
        xi, yi = x[okv].astype(int), y[okv].astype(int)
        g = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        num, den = np.zeros((ny, nx)), np.zeros((ny, nx))
        np.add.at(num, (yi[g], xi[g]), v[okv][g])
        np.add.at(den, (yi[g], xi[g]), 1)
        num, den = ndimage.gaussian_filter(num, s), ndimage.gaussian_filter(den, s)
        M = np.where((den * norm >= DENS_MIN) & inside, num / np.maximum(den, 1e-12), np.nan)
        ax = fig.add_subplot(gs[1, j + 1]); base(ax)
        dlo, dhi = m3.density_range(q)
        im = ax.imshow(M, cmap=cmap, vmin=dlo, vmax=dhi, interpolation="bilinear", zorder=1)
        zones(ax)
        pos = ax.get_position()
        cax = fig.add_axes([pos.x0 + 0.1 * pos.width, pos.y0 - 0.07 / H, 0.8 * pos.width, 0.045 / H])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=4.0, length=1.2, width=0.4, pad=1); cb.outline.set_linewidth(0.4)
        cb.set_label({"whisker": "selectivity", "auditory": "selectivity", "modality": "W < 0 < A", "latency_whisker": "ms",
                      "latency_auditory": "ms", "bimodal": "bimodal fraction"}[qk], fontsize=4.4, labelpad=1)
    fig.text(0.05 / W, 1 - 0.5 / H - ph / 2 / H, "Neurons", rotation=90, va="center", fontsize=6)
    fig.text(0.05 / W, 1 - 0.5 / H - 1.5 * ph / H, "Density", rotation=90, va="center", fontsize=6)
    h = [plt.Line2D([], [], color=m3.SRC_STYLE["whisker"][0], lw=1, label=f"whisker-cortex zone ({m3.ZONE_PCT} %)"),
         plt.Line2D([], [], color=m3.SRC_STYLE["auditory"][0], lw=1, ls="--", label=f"auditory-cortex zone ({m3.ZONE_PCT} %)"),
         plt.Rectangle((0, 0), 1, 1, color="#7b3294", alpha=0.75, lw=0, label="overlap")]
    if show_zones:
        fig.legend(handles=h, loc="lower left", ncol=1, frameon=False, fontsize=4.8, bbox_to_anchor=(0.25 / W, 0.0))
    fig.suptitle("Sensory responses across the isocortex (Allen butterfly flatmap, left hemisphere; all sessions pooled)",
                 x=0.25 / W, y=1 - 0.05 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, FIG, "cortical_flatmaps" if show_zones else "cortical_flatmaps_nozones")
    plt.close(fig)


def caption(U):
    txt = f"""# Sensory responses across the isocortex (flatmap)

**Sensory responses across the isocortex.** Allen CCFv3 butterfly flatmap of the isocortex (cortical streamlines from the
Laplace solution between pia and white matter; 2-D positions from geodesic distances on the cortical surface to two pairs
of anchor points; Wang et al. 2020, Harris et al. 2019; Allen ccf_streamlines assets). Left hemisphere, seen from above
(anterior up, lateral left); recorded neurons, folded onto the left hemisphere, are placed at the flatmap position of
their closest cortical streamline. The embedding does not preserve area (frontal pole and medial-posterior cortex
distorted). Neurons: good and multi-unit clusters in the isocortex, all sessions (both cohorts, learning day and expert
days; n = {len(U)} neurons, {U.session_id.nunique()} sessions).
Top row: neurons with a significant response (responsiveness: rate-based ROC, 5-35 ms after stimulus vs pre-trial
baseline, permutation p < 0.05; modality preference: whisker vs auditory, active trials; latencies: half-time to peak of
responsive neurons; bimodal: responsive to both modalities, Bonferroni over the modality's active / passive-pre /
passive-post tests in the session) coloured by the quantity; light grey: tested, not significant; grey number: neurons in
colour. First column, top: all recorded neurons with Allen area boundaries and labels.
Bottom row: density -- sum of the quantity over neurons and number of neurons with a value on the flat grid (10 um), each
smoothed with a 2-D Gaussian (sigma 150 um) and divided (normalised by the recorded-neuron density); shown inside the
isocortex where >= 3 neurons fall within the kernel. Lines: whisker-cortex (SSp-bfd + SSs, teal) and auditory-cortex
(AUDp + AUDd/v, brown dashed) projection zones ({m3.ZONE_PCT} % contours of the Allen anterograde projection density,
projected along the same streamlines; a flat pixel belongs to a zone when >= half of its streamline does; the source areas
are excluded from their own zone). First column, bottom: whisker zone (yellow), auditory zone (blue), overlap (purple).
"""
    (OUT / f"cortical_flatmaps_caption{m3.ZTAG}.md").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
