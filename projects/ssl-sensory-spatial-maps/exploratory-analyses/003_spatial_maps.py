"""003 -- Spatial maps of whisker / auditory responsiveness, modality preference and response latency in 500-um slabs,
with the projection zones of whisker and auditory cortex (layout after Chen et al. 2024, Cell, "Brain-wide neural
activity underlying memory-guided movement", supplementary slab figures).

Units: good + mua, all sessions pooled (both cohorts, learning day and expert days); positions ccf_atlas_ap / ml / dv (um,
CCF; ML folded onto the right hemisphere). Quantities (rate-based ROC, 045 roc_long; response window 5-35 ms vs pre-trial
baseline; selectivity = 2 AUC - 1, significant if permutation p < 0.05):
  whisker   whisker_active   (+ excited, - inhibited)
  auditory  auditory_active  (+ excited, - inhibited)
  modality  wh_vs_aud_active (+ auditory-preferring, - whisker-preferring)
  latency_whisker / latency_auditory  half-time to peak (001) of the responsive units
Slab sets (500 um thick; units within +-250 um of the slab centre are projected onto its central section):
  coronal   tiling the AP range of the recorded units
  sagittal  tiling the ML range (lateral distance from the midline)
  targets   coronal slabs centred on projection zones / areas (SSp-bfd, SSs, wM1, wM2, ORB, anterior / caudal / tail of
            the striatum, SCm, AUDp, TEa, PO, MG)
Panels per slab (one row): 1 schematic (sagittal section with Allen colours for coronal slabs, coronal section for sagittal
slabs; the slab drawn as a band); 2 all recorded neurons; 3 neurons coloured by the quantity, with the projection zones
(002; top 10 % of the Allen anterograde projection density of SSp-bfd, SSs, AUDp, AUDd+AUDv, union over the slab);
4 density map: mean of the quantity over the neurons in a 550 x 550 um window (in-plane, 50-um grid, Gaussian smoothing
sigma 50 um; shown where >= MIN_N neurons contribute); 5 significant neurons only (latency: responsive neurons).
Output: combined_results_ks4/_sensory_spatial_maps/figures/<quantity>/<set>_p<k>.{png,pdf,svg} + slab_units.csv
"""
import argparse
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ssl-stimulus-arrival-decoding" / "exploratory-analyses"))
S = importlib.import_module("_style")
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
ROC = RES / "_roc_stage_analysis"
OUT = RES / "_sensory_spatial_maps"
FIG = OUT / "figures"
ATLAS = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/allen_mouse_bluebrain_barrels_10um_v1.0")
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
MID, BREGMA_AP = 5700.0, 5400.0               # CCF midline (ML) and approximate bregma (AP), um
SLAB, BOX, GRID, SMOOTH, MIN_N = 500.0, 550.0, 50.0, 50.0, 5
ROWS_PER_PAGE = 5
SRC_STYLE = {"SSp-bfd": ("#1b7837", "-"), "SSs": ("#1b7837", (0, (3, 1.5))), "AUDp": ("#8c2d04", "-"),
             "AUD-sec": ("#8c2d04", (0, (3, 1.5)))}
SRC_LABEL = {"SSp-bfd": "SSp-bfd projections", "SSs": "SSs projections", "AUDp": "AUDp projections",
             "AUD-sec": "AUDd/AUDv projections"}


def cmap_modality():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("wh_aud", [S.WH_C, "#fbe3a6", "#f4f4f4", "#b9b9f0", S.AUD_C])


QUANT = {
    "whisker": dict(atype="whisker_active", title="Whisker responsiveness", cbar="Whisker selectivity (inhibited < 0 < excited)",
                    cmap="RdBu_r", vmin=-0.5, vmax=0.5),
    "auditory": dict(atype="auditory_active", title="Auditory responsiveness", cbar="Auditory selectivity (inhibited < 0 < excited)",
                     cmap="RdBu_r", vmin=-0.5, vmax=0.5),
    "modality": dict(atype="wh_vs_aud_active", title="Modality preference",
                     cbar="Modality selectivity (whisker-preferring < 0 < auditory-preferring)", cmap="modality",
                     vmin=-0.5, vmax=0.5),
    "latency_whisker": dict(col="latency_whisker_ms", title="Whisker response latency",
                            cbar="Half-time to peak after whisker stimulus (ms)", cmap="viridis", vmin=5, vmax=40),
    "latency_auditory": dict(col="latency_auditory_ms", title="Auditory response latency",
                             cbar="Half-time to peak after auditory stimulus (ms)", cmap="viridis", vmin=5, vmax=40),
}
# coronal slabs centred on projection zones / areas: (label, how, what); atlas = centroid (right hemisphere) of the Allen
# structure(s); units = centroid of the units with that area_acronym_custom; bregma = fixed AP (mm from bregma, approx.)
TARGETS = [("SSp-bfd", "atlas", ["SSp-bfd"]), ("SSs", "atlas", ["SSs"]), ("wM1", "units", "MO-wM1"),
           ("wM2", "units", "MO-wM2"), ("ORB", "atlas", ["ORB"]), ("Anterior striatum", "bregma", 1.0),
           ("Caudal striatum", "bregma", -0.5), ("Striatum tail", "bregma", -1.6), ("SCm", "atlas", ["SCm"]),
           ("AUDp", "atlas", ["AUDp"]), ("TEa", "atlas", ["TEa"]), ("PO (thalamus)", "atlas", ["PO"]),
           ("MG (thalamus)", "atlas", ["MG"])]


# ------------------------------------------------------------------------------------------------ data
def load_units():
    U = pd.read_parquet(ROC / "units.parquet")
    U = U[U.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
    R = pd.read_parquet(ROC / "roc_long.parquet")
    R = R[R.analysis_type.isin([q["atype"] for q in QUANT.values() if "atype" in q])].drop_duplicates(KEYS + ["analysis_type"])
    for q in QUANT.values():
        if "atype" in q:
            r = R[R.analysis_type == q["atype"]][KEYS + ["sel", "sig"]].rename(columns={"sel": f"sel_{q['atype']}", "sig": f"sig_{q['atype']}"})
            U = U.merge(r, on=KEYS, how="left")
    f = OUT / "unit_latency.parquet"
    if f.exists():
        L = pd.read_parquet(f)
        L["cluster_id"] = L.cluster_id.astype(str)
        U = U.merge(L[KEYS + ["latency_whisker_ms", "latency_auditory_ms"]], on=KEYS, how="left")
    else:
        U["latency_whisker_ms"] = U["latency_auditory_ms"] = np.nan
    U["ml_f"] = MID + np.abs(U.ccf_atlas_ml - MID)
    U["lat_mm"] = (U.ml_f - MID) / 1000
    U["ap_mm"] = U.ccf_atlas_ap / 1000
    U["dv_mm"] = U.ccf_atlas_dv / 1000
    return U


def values(U, qk):
    q = QUANT[qk]
    if "atype" in q:
        v, sig = U[f"sel_{q['atype']}"].to_numpy(float), U[f"sig_{q['atype']}"].to_numpy(float) == 1
    else:
        v = U[q["col"]].to_numpy(float)
        sig = np.isfinite(v)
    return v, sig


def parent_map(S_):
    name = dict(zip(S_.id, S_.name)); acr = dict(zip(S_.id, S_.acronym))
    par = dict(zip(S_.id, S_.parent_structure_id.fillna(-1).astype(int)))
    bfd = int(S_.loc[S_.acronym == "SSp-bfd", "id"].iloc[0])
    out = {}
    for i in S_.id:
        j = i
        while j in name and "layer" in str(name[j]).lower() and par.get(j, -1) in name:
            j = par[j]
        if str(acr.get(j, "")).startswith("SSp-bfd"):
            j = bfd
        out[i] = j
    return out


class Atlas:
    def __init__(self):
        import tifffile
        self.ann = tifffile.imread(ATLAS / "annotation.tiff")          # (AP, DV, ML), 10 um
        self.S = pd.read_csv(ATLAS / "structures.csv")
        self.pm = parent_map(self.S)
        js = json.load(open(ATLAS / "structures.json"))
        self.rgb = {s["id"]: np.array(s["rgb_triplet"]) / 255 for s in js}
        self.path = {s["id"]: s["structure_id_path"] for s in js}
        self.acr = {s["id"]: s["acronym"] for s in js}
        Z = np.load(OUT / "projection_zones.npz")
        self.zres = float(Z["res_um"])
        self.zones = {k[5:]: Z[k] for k in Z.files if k.startswith("mask_")}

    def merged(self, a):
        u, inv = np.unique(a, return_inverse=True)
        return np.array([self.pm.get(int(k), int(k)) for k in u])[inv].reshape(a.shape)

    def section(self, kind, c_um):
        """kind 'cor': (DV, lateral) right hemisphere at AP c_um; 'sag': (DV, AP) at folded ML c_um; 'corfull': (DV, ML)"""
        if kind in ("cor", "corfull"):
            a = self.ann[int(np.clip(round(c_um / 10), 0, self.ann.shape[0] - 1))]
            if kind == "cor":
                a = a[:, int(MID / 10):]
        else:
            a = self.ann[:, :, int(np.clip(round(c_um / 10), 0, self.ann.shape[2] - 1))].T
        return self.merged(a)

    def boundaries(self, lab):
        bnd = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
        return bnd & (lab != 0), ndimage.binary_fill_holes(lab != 0)

    def colour_image(self, lab):
        u, inv = np.unique(lab, return_inverse=True)
        cols = np.array([self.rgb.get(int(k), np.ones(3)) if k != 0 else np.ones(3) for k in u])
        return cols[inv].reshape(lab.shape + (3,))

    def zone_section(self, src, kind, c_um):
        m = self.zones[src]
        r = self.zres
        if kind == "cor":
            i0, i1 = int(np.floor((c_um - SLAB / 2) / r)), int(np.ceil((c_um + SLAB / 2) / r))
            return m[max(i0, 0):i1].any(0)[:, int(MID / r):]               # (DV, lateral)
        i0, i1 = int(np.floor((c_um - SLAB / 2) / r)), int(np.ceil((c_um + SLAB / 2) / r))
        return m[:, :, max(i0, 0):i1].any(2).T                              # (DV, AP)

    def centroid_ap(self, acronyms):
        ids = {i for i, p in self.path.items() if any(self.S.loc[self.S.acronym == a, "id"].iloc[0] in p for a in acronyms)}
        a = self.ann[::5, ::5, ::5]
        m = np.isin(a, list(ids))
        m[:, :, : int(MID / 50)] = False
        return float(np.mean(np.where(m)[0]) * 50 + 25)


# ------------------------------------------------------------------------------------------------ slabs
def slab_sets(U, A):
    ap = U.ccf_atlas_ap.to_numpy()
    lo, hi = np.floor(np.nanpercentile(ap, 0.5) / SLAB) * SLAB, np.ceil(np.nanpercentile(ap, 99.5) / SLAB) * SLAB
    cor = [(f"AP {(BREGMA_AP - c) / 1000:+.2f} mm", c) for c in np.arange(lo + SLAB / 2, hi, SLAB)]
    ml = U.ml_f.to_numpy()
    hi = np.ceil(np.nanpercentile(ml, 99.5) / SLAB) * SLAB
    sag = [(f"ML {(c - MID) / 1000:.2f} mm", c) for c in np.arange(MID + SLAB / 2, hi, SLAB)]
    tg = []
    for lab, how, what in TARGETS:
        if how == "atlas":
            c = A.centroid_ap(what)
        elif how == "units":
            c = float(U.loc[U.area_acronym_custom == what, "ccf_atlas_ap"].median())
        else:
            c = BREGMA_AP - 1000 * what
        tg.append((f"{lab}, AP {(BREGMA_AP - c) / 1000:+.2f} mm", c))
    return {"coronal": ("cor", cor), "sagittal": ("sag", sag), "targets": ("cor", tg)}


def in_slab(U, kind, c):
    if kind == "cor":
        m = (U.ccf_atlas_ap - c).abs() <= SLAB / 2
        return m.to_numpy(), U.lat_mm.to_numpy(), U.dv_mm.to_numpy()
    m = (U.ml_f - c).abs() <= SLAB / 2
    return m.to_numpy(), U.ap_mm.to_numpy(), U.dv_mm.to_numpy()


def density_range(q):
    """density colour range: half the neuron range for selectivities (means are smaller), same range for latencies"""
    if q["vmin"] < 0:
        return q["vmin"] / 2, q["vmax"] / 2
    return q["vmin"], q["vmax"]


def density(x, y, v, xr, yr):
    bx = np.arange(xr[0], xr[1] + 1e-9, GRID / 1000)
    by = np.arange(yr[0], yr[1] + 1e-9, GRID / 1000)
    ok = np.isfinite(v)
    Sm, _, _ = np.histogram2d(y[ok], x[ok], bins=[by, bx], weights=v[ok])
    C, _, _ = np.histogram2d(y[ok], x[ok], bins=[by, bx])
    k = int(round(BOX / GRID))
    Sm, C = ndimage.uniform_filter(Sm, k, mode="constant") * k * k, ndimage.uniform_filter(C, k, mode="constant") * k * k
    Sm, C2 = ndimage.gaussian_filter(Sm, SMOOTH / GRID), ndimage.gaussian_filter(C, SMOOTH / GRID)
    M = np.where(C2 >= MIN_N, Sm / np.maximum(C2, 1e-9), np.nan)
    return M, (bx[0], bx[-1], by[-1], by[0])


# ------------------------------------------------------------------------------------------------ drawing
def draw_section(ax, A, lab, extent, colour=False):
    bnd, inside = A.boundaries(lab)
    if colour:
        img = A.colour_image(lab)
        img[~inside] = 1.0
        ax.imshow(img, extent=extent, interpolation="nearest", zorder=1)
        rgba = np.zeros(bnd.shape + (4,)); rgba[..., 3] = bnd * 0.35
    else:
        rgba = np.zeros(bnd.shape + (4,)); rgba[..., :3] = 0.6; rgba[..., 3] = bnd * 0.6
    ax.imshow(rgba, extent=extent, interpolation="antialiased", zorder=2)
    yy = extent[3] + (np.arange(lab.shape[0]) + 0.5) * (extent[2] - extent[3]) / lab.shape[0]
    xx = extent[0] + (np.arange(lab.shape[1]) + 0.5) * (extent[1] - extent[0]) / lab.shape[1]
    ax.contour(xx, yy, inside.astype(float), levels=[0.5], colors="0.25", linewidths=0.5, zorder=3)


def draw_zones(ax, A, kind, c, extent):
    for src, (col, ls) in SRC_STYLE.items():
        z = A.zone_section(src, kind, c)
        if not z.any():
            continue
        z = ndimage.binary_opening(z, iterations=1) | z
        yy = (np.arange(z.shape[0]) + 0.5) * A.zres / 1000
        xx = extent[0] + (np.arange(z.shape[1]) + 0.5) * A.zres / 1000
        ax.contour(xx, yy, ndimage.gaussian_filter(z.astype(float), 0.7), levels=[0.5], colors=[col], linewidths=0.6,
                   linestyles=[ls], zorder=6)


def scalebar(ax, x0, y0):
    ax.plot([x0, x0 + 1], [y0, y0], color="k", lw=1.0, solid_capstyle="butt", zorder=8)
    ax.text(x0 + 0.5, y0 - 0.12, "1 mm", ha="center", va="bottom", fontsize=4.6)


def make_page(plt, A, U, qk, set_name, kind, slabs, page, n_pages, schem):
    q = QUANT[qk]
    cmap = cmap_modality() if q["cmap"] == "modality" else plt.get_cmap(q["cmap"])
    v_all, sig_all = values(U, qk)
    if kind == "cor":
        ext_sec, xlim = (0, 5.7, 8.0, 0), (0, 5.8)
        ratio_sec = 5.8 / 7.4
        ext_sch, xlim_sch = (0, 13.2, 8.0, 0), (0.6, 12.6)
        ratio_sch = 12.0 / 7.4
    else:
        ext_sec, xlim = (0, 13.2, 8.0, 0), (0.6, 12.6)
        ratio_sec = 12.0 / 7.4
        ext_sch, xlim_sch = (0, 11.4, 8.0, 0), (0, 11.4)
        ratio_sch = 11.4 / 7.4
    ylim = (7.4, 0)
    ratios = [ratio_sch] + [ratio_sec] * 4
    W = S.W_IN
    h = (W - 0.75) / sum(ratios)
    H = h * len(slabs) + 1.0
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(slabs), 5, width_ratios=ratios, wspace=0.04, hspace=0.12, left=0.6 / W, right=1 - 0.05 / W,
                          top=1 - 0.5 / H, bottom=0.5 / H)
    rows = []
    for r, (lab, c) in enumerate(slabs):
        m, x, y = in_slab(U, kind, c)
        v, sig = v_all[m], sig_all[m]
        xs, ys = x[m], y[m]
        sec = A.section(kind, c)
        axs = [fig.add_subplot(gs[r, k]) for k in range(5)]
        # 1 schematic
        ax = axs[0]
        if kind == "cor":
            draw_section(ax, A, schem, ext_sch, colour=True)
            ax.axvspan((c - SLAB / 2) / 1000, (c + SLAB / 2) / 1000, color="k", alpha=0.35, lw=0, edgecolor="none", zorder=5)
        else:
            draw_section(ax, A, schem, ext_sch, colour=True)
            for side in (+1, -1):
                ax.axvspan((MID + side * (c - MID) - SLAB / 2) / 1000, (MID + side * (c - MID) + SLAB / 2) / 1000,
                           color="k", alpha=0.35 if side > 0 else 0.12, lw=0, edgecolor="none", zorder=5)
        ax.set_xlim(*xlim_sch); ax.set_ylim(*ylim)
        ax.text(-0.04, 0.5, f"{lab}\n{m.sum()} neurons", transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=5.4)
        # 2 all neurons
        draw_section(axs[1], A, sec, ext_sec)
        axs[1].scatter(xs, ys, s=0.25, c="0.35", lw=0, zorder=4, rasterized=True)
        # 3 coloured by the quantity (+ projection zones)
        draw_section(axs[2], A, sec, ext_sec)
        fin = np.isfinite(v)
        axs[2].scatter(xs[~fin], ys[~fin], s=0.2, c="0.8", lw=0, zorder=3.5, rasterized=True)
        o = np.argsort(np.abs(v[fin] - (0 if q["vmin"] < 0 else np.nanmedian(v))))
        axs[2].scatter(xs[fin][o], ys[fin][o], s=0.9, c=v[fin][o], cmap=cmap, vmin=q["vmin"], vmax=q["vmax"], lw=0, zorder=4,
                       rasterized=True)
        draw_zones(axs[2], A, kind, c, ext_sec)
        # 4 density
        draw_section(axs[3], A, sec, ext_sec)
        M, ext = density(xs, ys, v, (xlim[0], xlim[1]), (0, 8.0))
        _, inside = A.boundaries(sec)
        gy = ((np.arange(M.shape[0]) + 0.5) * GRID / 10).astype(int)            # density grid -> 10-um section pixels
        gx = ((np.arange(M.shape[1]) + 0.5) * GRID / 10 + xlim[0] * 100).astype(int)
        ok = (gy[:, None] < inside.shape[0]) & (gx[None, :] < inside.shape[1])
        inb = np.zeros(M.shape, bool)
        inb[ok] = inside[np.minimum(gy, inside.shape[0] - 1)[:, None].repeat(M.shape[1], 1)[ok],
                         np.minimum(gx, inside.shape[1] - 1)[None, :].repeat(M.shape[0], 0)[ok]]
        M = np.where(inb, M, np.nan)
        dlo, dhi = density_range(q)
        imd = axs[3].imshow(M, extent=ext, cmap=cmap, vmin=dlo, vmax=dhi, interpolation="bilinear", zorder=1.5)
        im = axs[4].scatter([], [], c=[], cmap=cmap, vmin=q["vmin"], vmax=q["vmax"])
        draw_zones(axs[3], A, kind, c, ext_sec)
        # 5 significant only
        draw_section(axs[4], A, sec, ext_sec)
        o = np.argsort(np.abs(v[sig]))
        axs[4].scatter(xs[sig][o], ys[sig][o], s=1.1, c=v[sig][o], cmap=cmap, vmin=q["vmin"], vmax=q["vmax"], lw=0, zorder=4,
                       rasterized=True)
        draw_zones(axs[4], A, kind, c, ext_sec)
        for ax in axs[1:]:
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        for ax in axs:
            ax.set_aspect("equal"); ax.set_axis_off()
        if r == len(slabs) - 1:
            scalebar(axs[1], xlim[0] + 0.2, 7.25)
        if r == 0:
            heads = ["Slab position", "All recorded neurons", f"{q['title']}, all neurons", "Density (550-um window)",
                     "Significant neurons" if "atype" in q else "Responsive neurons"]
            for ax, t in zip(axs, heads):
                ax.set_title(t, fontsize=5.8, pad=3)
        rows.append(dict(quantity=qk, set=set_name, slab=lab, centre_um=c, n_units=int(m.sum()),
                         n_with_value=int(np.isfinite(v).sum()), n_significant=int(sig.sum()),
                         n_sessions=int(U.session_id[m].nunique())))
    for xc, mappable, lab in [(0.12, im, q["cbar"] + ", neurons"), (0.45, imd, "Mean over the 550-um window (density)")]:
        cax = fig.add_axes([xc, 0.22 / H, 0.25, 0.06 / H])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=5.0); cb.ax.tick_params(labelsize=4.8, width=0.4, length=1.5); cb.outline.set_linewidth(0.4)
    h_ = [plt.Line2D([], [], color=col, ls=ls, lw=0.8, label=SRC_LABEL[s]) for s, (col, ls) in SRC_STYLE.items()]
    fig.legend(handles=h_, loc="lower right", ncol=2, frameon=False, fontsize=4.8, bbox_to_anchor=(0.99, 0.0), handlelength=2.2)
    set_word = {"coronal": "coronal slabs", "sagittal": "sagittal slabs", "targets": "coronal slabs centred on projection zones"}[set_name]
    fig.suptitle(f"{q['title']} -- {set_word} (500 um), all sessions pooled" + (f"  [{page}/{n_pages}]" if n_pages > 1 else ""),
                 x=0.6 / W, y=1 - 0.12 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, FIG / qk, f"{set_name}_p{page}")
    plt.close(fig)
    return rows


def main(a):
    plt = S.setup()
    U = load_units()
    A = Atlas()
    sets = slab_sets(U, A)
    schem = {"cor": A.section("sag", MID + 2300), "sag": A.section("corfull", 6000)}
    rows = []
    for qk in (a.quantities.split(",") if a.quantities else QUANT):
        for set_name, (kind, slabs) in sets.items():
            if a.sets and set_name not in a.sets.split(","):
                continue
            pages = [slabs[i:i + ROWS_PER_PAGE] for i in range(0, len(slabs), ROWS_PER_PAGE)]
            for k, sl in enumerate(pages):
                rows += make_page(plt, A, U, qk, set_name, kind, sl, k + 1, len(pages), schem["cor" if kind == "cor" else "sag"])
                print(qk, set_name, k + 1, "/", len(pages), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "slab_units.csv", index=False)
    json.dump({k: [(l, float(c)) for l, c in v[1]] for k, v in sets.items()}, open(OUT / "slabs.json", "w"), indent=1)
    print("ALL DONE", FIG)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantities", default=None)
    ap.add_argument("--sets", default=None)
    main(ap.parse_args())
