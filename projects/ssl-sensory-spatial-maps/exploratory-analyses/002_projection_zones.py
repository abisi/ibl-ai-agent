"""002 -- Projection zones of whisker and auditory cortical areas (Allen Mouse Brain Connectivity Atlas).

Experiments: anterograde AAV tracing with the primary injection in the source area, wild-type (C57BL/6J, pan-neuronal
AAV) or Emx1-IRES-Cre (all cortical excitatory neurons) -- Cre lines restricted to one layer or cell type are not used.
Sources: SSp-bfd, SSs (whisker); AUDp, AUD-secondary (AUDd + AUDv) (auditory).
Volume: projection_density (fraction of each voxel's volume with labelled axons), 50-um CCF grid, downloaded through the
Allen API (grid_data/download_file). Injections in the left hemisphere are mirrored so that every injection is in the
right hemisphere (the hemisphere the recorded units are folded onto); the ipsilateral (right) hemisphere is used.
Per experiment the density is divided by its sum over the candidate voxels (shape of the projection, independent of the
injection size), then averaged over experiments of the source.
Candidate voxels: right hemisphere, inside the brain, excluding fibre tracts, ventricles and the source area itself
(with its layers / barrels), and receiving signal (density > 0) in at least half of the source's experiments.
Projection zone (user, 2026-10-04, third version): the mean density is smoothed (Gaussian, sigma 50 um) and the zone
is the 70 % contour = the highest-density voxels that together hold 70 % of the source's projection (candidate voxels).
The first version (top 10 % of candidate voxels, unsmoothed) is kept as mask_<source>.
Merged zones (user 2026-10-04): whisker = 70 % contour of the mean of the SSp-bfd and SSs densities (each normalised),
auditory = same for AUDp and AUDd/AUDv (zone70_whisker / zone70_auditory). Overlap = whisker zone & auditory zone;
overlap volume per Allen structure (layers merged) -> projection_overlap.csv.
Output: combined_results_ks4/_sensory_spatial_maps/projection_zones.npz (zone70_<source>, mask_<source>,
density_<source> smoothed; AP x DV x ML, 50 um) + projection_experiments.csv + projection_overlap.csv +
figures/projection_zones_{coronal,sagittal}.{png,pdf,svg}: 500-um slabs (mean density, union of the zone), rows =
sources + overlap of the whisker and auditory zones, labels = the 3 structures holding most of each zone in the slab, right
column = composition of the whole zone (% of its volume per structure, projection_zone_composition*.csv).
"""
import json
import os
import pathlib
import time
import urllib.request

import numpy as np
import pandas as pd

OUT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/_sensory_spatial_maps")
RAW = OUT / "allen_projection_density_50um"
ATLAS = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/allen_mouse_bluebrain_barrels_10um_v1.0")
SOURCES = {"SSp-bfd": [329], "SSs": [378], "AUDp": [1002], "AUD-sec": [1011, 1018]}
LINES_OK = {"", "Emx1-IRES-Cre"}
ZONE_PCT = int(os.environ.get("ZONE_PCT", "70"))      # projection-zone contour level (user: 70, repeated with 90)
ZTAG = "" if ZONE_PCT == 70 else f"_zone{ZONE_PCT}"
RES_UM, MID_UM, TOP, MASS, SMOOTH_UM = 50, 5700, 0.10, ZONE_PCT / 100, 50.0
SLAB_UM, CONTOUR_SIGMA = 500.0, 0.8           # figure slabs (um) and contour-line smoothing (50-um voxels)
WHISKER_SRC, AUDITORY_SRC = ["SSp-bfd", "SSs"], ["AUDp", "AUD-sec"]
SRC_NAME = {"SSp-bfd": "SSp-bfd\n(barrel cortex)", "SSs": "SSs (secondary\nsomatosensory)", "AUDp": "AUDp (primary\nauditory)",
            "AUD-sec": "AUDd + AUDv\n(secondary auditory)", "overlap": "Overlap of the\nwhisker and\nauditory zones"}
API = "https://api.brain-map.org/api/v2/data/query.json?criteria=service::mouse_connectivity_injection_structure"


def experiments():
    rows = []
    for src, ids in SOURCES.items():
        for sid in ids:
            for attempt in range(5):
                d = json.load(urllib.request.urlopen(f"{API}[injection_structures$eq{sid}][primary_structure_only$eqtrue]", timeout=120))["msg"]
                if isinstance(d, list):
                    break
                time.sleep(10)
            else:
                raise RuntimeError(f"Allen API error for structure {sid}: {d}")
            for e in d:
                line = e.get("transgenic-line") or ""
                if line in LINES_OK:
                    rows.append(dict(source=src, experiment_id=e["id"], line=line or "wild type",
                                     structure=e["structure-abbrev"], injection_volume=e["injection-volume"],
                                     inj_ap=e["injection-coordinates"][0], inj_dv=e["injection-coordinates"][1],
                                     inj_ml=e["injection-coordinates"][2]))
    return pd.DataFrame(rows)


def download(eid):
    import nrrd
    f = RAW / f"{eid}.nrrd"
    if not f.exists():
        url = f"https://api.brain-map.org/grid_data/download_file/{eid}?image=projection_density&resolution={RES_UM}"
        data = urllib.request.urlopen(url, timeout=600).read()
        f.with_suffix(".tmp").write_bytes(data)
        f.with_suffix(".tmp").rename(f)
    v, _ = nrrd.read(str(f))
    return v.astype(np.float32)                                   # (AP, DV, ML)


def atlas_50um():
    import tifffile
    ann = tifffile.imread(ATLAS / "annotation.tiff")              # (AP, DV, ML), 10 um
    s = RES_UM // 10
    return ann[s // 2::s, s // 2::s, s // 2::s]


def descendants(structs, ids):
    ids = set(ids)
    return {s["id"] for s in structs if ids & set(s["structure_id_path"])}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    (OUT / f"figures{ZTAG}").mkdir(parents=True, exist_ok=True)
    E = experiments()
    E.to_csv(OUT / "projection_experiments.csv", index=False)
    print(E.groupby("source").agg(n=("experiment_id", "size"), lines=("line", lambda x: dict(x.value_counts()))).to_string())
    structs = json.load(open(ATLAS / "structures.json"))
    ann = atlas_50um()
    excl_common = descendants(structs, [1009, 73])                # fibre tracts, ventricular systems
    vol = {}
    for e in E.itertuples():
        v = download(e.experiment_id)
        if e.inj_ml < MID_UM:
            v = v[:, :, ::-1]                                      # mirror: injection in the right hemisphere
        vol[e.experiment_id] = v
        print("loaded", e.source, e.experiment_id, v.shape, flush=True)
    shape = next(iter(vol.values())).shape
    assert ann.shape == shape, (ann.shape, shape)
    right = np.zeros(shape, bool)
    right[:, :, int(MID_UM / RES_UM):] = True
    brain = (ann != 0) & ~np.isin(ann, list(excl_common)) & right
    out, summ = {}, []
    for src, ids in SOURCES.items():
        ex = E[E.source == src].experiment_id.tolist()
        cand = brain & ~np.isin(ann, list(descendants(structs, ids)))
        V = np.stack([vol[i] for i in ex])
        V = np.where(cand[None], V, 0)
        V = V / V.sum(axis=(1, 2, 3), keepdims=True)
        sig = (V > 0).mean(0) >= 0.5
        m = V.mean(0)
        c = cand & sig
        thr = np.quantile(m[c], 1 - TOP)
        mask = c & (m >= thr)
        from scipy import ndimage
        ms = ndimage.gaussian_filter(m, SMOOTH_UM / RES_UM) * cand
        v = np.sort(ms[cand])[::-1]
        thr70 = v[np.searchsorted(np.cumsum(v) / v.sum(), MASS)]
        zone = cand & (ms >= thr70)
        out[f"mask_{src}"] = mask
        out[f"zone70_{src}"] = zone
        out[f"density_{src}"] = ms.astype(np.float32)
        out[f"thr70_{src}"] = np.float32(thr70)
        out[f"source_{src}"] = np.isin(ann, list(descendants(structs, ids))) & right          # injected area (excluded)
        top_regions = pd.Series(ann[zone]).map({s["id"]: s["acronym"] for s in structs}).value_counts().head(12)
        summ.append(dict(source=src, n_experiments=len(ex), n_candidate_voxels=int(c.sum()),
                         zone70_volume_mm3=float(zone.sum() * (RES_UM / 1000) ** 3),
                         top10_volume_mm3=float(mask.sum() * (RES_UM / 1000) ** 3), top_regions_zone70=top_regions.to_dict()))
        print(src, summ[-1], flush=True)
    np.savez_compressed(OUT / f"projection_zones{ZTAG}.npz", res_um=RES_UM, **out)
    pd.DataFrame(summ).to_csv(OUT / f"projection_zones_summary{ZTAG}.csv", index=False)
    finish(out, ann, structs)
    print("ALL DONE", OUT / f"projection_zones{ZTAG}.npz")


def m3_module():
    import importlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    return importlib.import_module("003_spatial_maps")


def merged_ids(ann):
    """annotation -> structure id with cortical layers and barrel columns merged (as the section contours)"""
    pm = m3_module().parent_map(pd.read_csv(ATLAS / "structures.csv"))
    u, inv = np.unique(ann, return_inverse=True)
    return np.array([pm.get(int(k), int(k)) for k in u])[inv].reshape(ann.shape)


DIVISIONS = [(315, "Isocortex"), (698, "Olfactory"), (1089, "Hippocampal formation"), (703, "Cortical subplate"),
             (477, "Striatum"), (803, "Pallidum"), (549, "Thalamus"), (1097, "Hypothalamus"), (313, "Midbrain"),
             (771, "Pons"), (354, "Medulla"), (512, "Cerebellum")]


def merged_zones(out, ann, structs):
    """whisker zone = 70 % contour of the mean of the SSp-bfd and SSs densities (each normalised to its own total);
    auditory zone = same for AUDp and AUDd/AUDv (user, 2026-10-04: merge the sources of a modality)"""
    right = np.zeros(ann.shape, bool)
    right[:, :, int(MID_UM / RES_UM):] = True
    brain = (ann != 0) & ~np.isin(ann, list(descendants(structs, [1009, 73]))) & right
    for g, srcs in (("whisker", WHISKER_SRC), ("auditory", AUDITORY_SRC)):
        src_ids = [i for s_ in srcs for i in SOURCES[s_]]
        cand = brain & ~np.isin(ann, list(descendants(structs, src_ids)))
        d = np.mean([np.where(cand, out[f"density_{s_}"], 0) / out[f"density_{s_}"][cand].sum() for s_ in srcs], axis=0)
        v = np.sort(d[cand])[::-1]
        thr = v[np.searchsorted(np.cumsum(v) / v.sum(), MASS)]
        out[f"zone70_{g}"] = cand & (d >= thr)
        out[f"density_{g}"] = d.astype(np.float32)
        out[f"thr70_{g}"] = np.float32(thr)
        out[f"source_{g}"] = np.isin(ann, list(descendants(structs, src_ids))) & right
    return out


def zone_sets(out):
    z = {s: out[f"zone70_{s}"] for s in WHISKER_SRC + AUDITORY_SRC}
    w, a = out["zone70_whisker"], out["zone70_auditory"]
    z["whisker"], z["auditory"] = w, a
    z["overlap"] = w & a
    return z, w, a


def composition(out, lab, structs):
    """share of each zone's volume per Allen structure (layers merged) and per major division"""
    acr = {s["id"]: s["acronym"] for s in structs}
    name = {s["id"]: s["name"] for s in structs}
    path = {s["id"]: s["structure_id_path"] for s in structs}
    div = {k: next((n for d, n in DIVISIONS if d in path.get(k, [])), "other") for k in path}
    z, _, _ = zone_sets(out)
    v = (RES_UM / 1000) ** 3
    rows = []
    for src, m in z.items():
        ids, cnt = np.unique(lab[m], return_counts=True)
        for k, n in zip(ids, cnt):
            rows.append(dict(zone=src, structure=acr.get(int(k), str(k)), name=name.get(int(k), ""),
                             division=div.get(int(k), "other"), volume_mm3=n * v, frac_of_zone=n / m.sum()))
    C = pd.DataFrame(rows).sort_values(["zone", "frac_of_zone"], ascending=[True, False])
    rec = m3_module().recorded_structures()
    C["recorded"] = C.structure.isin(rec) if rec is not None else True
    C.to_csv(OUT / f"projection_zone_composition{ZTAG}.csv", index=False)
    D = C.groupby(["zone", "division"]).frac_of_zone.sum().unstack(fill_value=0).round(3)
    D.to_csv(OUT / f"projection_zone_composition_divisions{ZTAG}.csv")
    print(D.to_string())
    print(C.groupby("zone").head(6).round(3).to_string())
    return C


def overlap_table(out, lab, structs):
    z, w, a = zone_sets(out)
    ov = z["overlap"]
    acr = {s["id"]: s["acronym"] for s in structs}
    name = {s["id"]: s["name"] for s in structs}
    v = (RES_UM / 1000) ** 3
    right = np.zeros(lab.shape, bool)
    right[:, :, int(MID_UM / RES_UM):] = True
    rows = []
    for k in np.unique(lab[ov]):
        inside = (lab == k) & right
        rows.append(dict(structure=acr.get(int(k), str(k)), name=name.get(int(k), ""), overlap_mm3=float((ov & inside).sum() * v),
                         structure_mm3=float(inside.sum() * v),
                         frac_structure_in_overlap=float((ov & inside).sum() / max(1, inside.sum())),
                         frac_structure_in_whisker_zone=float((w & inside).sum() / max(1, inside.sum())),
                         frac_structure_in_auditory_zone=float((a & inside).sum() / max(1, inside.sum())),
                         ap_centre_um=float(np.mean(np.where(ov & inside)[0]) * RES_UM + RES_UM / 2)))
    T = pd.DataFrame(rows).sort_values("overlap_mm3", ascending=False)
    rec = m3_module().recorded_structures()
    T["recorded"] = T.structure.isin(rec) if rec is not None else True
    pairs = {f"{x}&{y}": float((out[f"zone70_{x}"] & out[f"zone70_{y}"]).sum() * v) for x in WHISKER_SRC for y in AUDITORY_SRC}
    T.to_csv(OUT / f"projection_overlap{ZTAG}.csv", index=False)
    json.dump(dict(whisker_union_mm3=float(w.sum() * v), auditory_union_mm3=float(a.sum() * v), overlap_mm3=float(ov.sum() * v),
                   pairwise_overlap_mm3=pairs), open(OUT / f"projection_overlap_summary{ZTAG}.json", "w"), indent=1)
    print("overlap", float(ov.sum() * v), "mm3;", pairs)


def finish(out, ann, structs):
    out = merged_zones(out, ann, structs)
    out["res_um"] = RES_UM
    np.savez_compressed(OUT / f"projection_zones{ZTAG}.npz", **out)
    lab = merged_ids(ann)
    overlap_table(out, lab, structs)
    C = composition(out, lab, structs)
    for kind in ("cor", "sag"):
        figure(out, lab, structs, C, kind)


def figure(out, lab, structs, C, kind):
    """rows: the 4 sources and their overlap; columns: 500-um slabs (coronal: AP, sagittal: ML) + zone composition.
    Density = mean over the slab (white = no projection), lines = smoothed 70 % contours (union over the slab),
    labels = the 3 structures holding most of the zone in that slab."""
    import matplotlib.patheffects as pe
    from matplotlib.colors import ListedColormap, LogNorm
    from scipy import ndimage
    m3 = m3_module()
    S = m3.S
    plt = S.setup()
    A = m3.Atlas(load_zones=False)
    acr = {s["id"]: s["acronym"] for s in structs}
    REC = m3.recorded_structures()
    z, w_all, a_all = zone_sets(out)
    from matplotlib.colors import LinearSegmentedColormap
    # user 2026-10-04: white -> modality colour -> dark grey (white = no projection)
    cmw = LinearSegmentedColormap.from_list("white_whisker_grey", ["#ffffff", "#f7b519", "#3a3a3a"])
    cma = LinearSegmentedColormap.from_list("white_auditory_grey", ["#ffffff", "#2c2cdb", "#3a3a3a"])
    cm = {**{s: cmw for s in WHISKER_SRC}, **{s: cma for s in AUDITORY_SRC}}
    wcol, acol, ocol = "#d99a00", "#2c2cdb", "#9b6bb5"
    rows = WHISKER_SRC + AUDITORY_SRC + ["overlap"]
    r50, i0, half = RES_UM / 1000, int(MID_UM / RES_UM), int(round(SLAB_UM / 2 / RES_UM))
    if kind == "cor":
        centres = np.arange(2750, 10251, 1000)
        sec_ext, xlim, ylim, aspect_w = (0, 5.7, 8.0, 0), (0, 5.8), (7.3, 0), 5.8
    else:
        centres = MID_UM + np.arange(750, 4751, 1000)
        sec_ext, xlim, ylim, aspect_w = (0, 13.2, 8.0, 0), (0.6, 12.4), (7.3, 0), 11.8
    W = S.W_IN
    bar_w = 1.15
    pw = (W - 0.55 - bar_w) / len(centres)
    ph = pw * 7.3 / aspect_w
    H = ph * len(rows) + 1.0
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(rows), len(centres) + 1, width_ratios=[1] * len(centres) + [bar_w / pw * 0.92], wspace=0.02,
                          hspace=0.06, left=0.5 / W, right=1 - 0.05 / W, top=1 - 0.34 / H, bottom=0.62 / H)
    ims = {}

    def slab(vol, c):
        i = int(round(c / RES_UM))
        if kind == "cor":
            return vol[max(i - half, 0):i + half].mean(0)[:, i0:] if vol.dtype != bool else vol[max(i - half, 0):i + half].any(0)[:, i0:]
        sl = vol[:, :, max(i - half, 0):i + half]
        return (sl.mean(2) if vol.dtype != bool else sl.any(2)).T

    def lab_slab(c):
        i = int(round(c / RES_UM))
        if kind == "cor":
            return lab[i][:, i0:]
        return lab[:, :, i].T

    def grid_xy(shape):
        return (np.arange(shape[1]) + 0.5) * r50, (np.arange(shape[0]) + 0.5) * r50

    def contour(ax, m2, col, lw=0.7):
        if m2.any():
            xx, yy = grid_xy(m2.shape)
            ax.contour(xx, yy, ndimage.gaussian_filter(m2.astype(float), CONTOUR_SIGMA), levels=[0.5], colors=[col],
                       linewidths=lw, zorder=5)

    def annotate(ax, zm, lb):
        """the (up to) 3 structures holding most of the zone in this slab; labels closer than 0.7 mm are skipped"""
        ids, cnt = np.unique(lb[zm], return_counts=True)
        keep = [(k, n) for k, n in zip(ids, cnt) if k != 0 and n >= 15 and (REC is None or acr.get(int(k), "") in REC)]
        placed = []
        for k, n in sorted(keep, key=lambda t: -t[1]):
            yy, xx = np.where(zm & (lb == k))
            x, y = (np.median(xx) + 0.5) * r50, (np.median(yy) + 0.5) * r50
            if any(np.hypot(x - a, y - b) < 0.7 for a, b in placed):
                continue
            placed.append((x, y))
            ax.text(x, y, acr.get(int(k), ""), fontsize=3.9, ha="center", va="center", zorder=7, color="k",
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
            if len(placed) == 3:
                break

    for r, src in enumerate(rows):
        for c_i, c in enumerate(centres):
            ax = fig.add_subplot(gs[r, c_i])
            sec = A.section("cor" if kind == "cor" else "sag", c)
            zm = slab(z[src], c)
            ext = (0, zm.shape[1] * r50, zm.shape[0] * r50, 0)
            if src != "overlap":
                d = slab(out[f"density_{src}"], c)
                thr = float(out[f"thr70_{src}"])
                vmax = float(np.quantile(out[f"density_{src}"][out[f"density_{src}"] > 0], 0.999))
                ims[src] = ax.imshow(np.ma.masked_where(d < thr / 30, d), extent=ext, cmap=cm[src], norm=LogNorm(thr / 30, vmax),
                                     interpolation="bilinear", zorder=1)
                sm = slab(out[f"source_{src}"], c)
                if sm.any():
                    ax.imshow(np.ma.masked_where(~sm, sm.astype(float)), extent=ext, cmap=ListedColormap(["#cfcfcf"]),
                              interpolation="nearest", zorder=1.2)
                m3.draw_section(ax, A, sec, sec_ext)
                contour(ax, zm, "k", 0.6)
            else:
                ov = ndimage.gaussian_filter(zm.astype(float), CONTOUR_SIGMA) >= 0.5
                ax.imshow(np.ma.masked_where(~ov, ov.astype(float)), extent=ext, cmap=ListedColormap([ocol]), alpha=0.6,
                          interpolation="nearest", zorder=1)
                m3.draw_section(ax, A, sec, sec_ext)
                contour(ax, slab(w_all, c), wcol)
                contour(ax, slab(a_all, c), acol)
            annotate(ax, zm, lab_slab(c))
            ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.set_axis_off()
            if r == 0:
                ax.set_title(f"AP {(5400 - c) / 1000:+.2f} mm" if kind == "cor" else f"ML {(c - MID_UM) / 1000:.2f} mm",
                             fontsize=5.6, pad=1)
            if c_i == 0:
                ax.text(-0.04, 0.5, SRC_NAME[src], transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=5.3)
            if r == len(rows) - 1 and c_i == 0:
                ax.plot([xlim[0] + 0.2, xlim[0] + 1.2], [7.15, 7.15], color="k", lw=0.9, solid_capstyle="butt")
                ax.text(xlim[0] + 0.7, 7.0, "1 mm", ha="center", va="bottom", fontsize=4.8)
        # composition of the whole zone (all slabs): top structures, % of the zone volume
        axb = fig.add_subplot(gs[r, -1])
        q = C[(C.zone == src) & (C.recorded if "recorded" in C else True)].reset_index(drop=True)
        need = [int(q.index[q.structure == k][0]) + 1 for k in ("SCm", "MRN") if (q.structure == k).any()]
        q = q.head(min(16, max([8] + need)))                      # enough rows to reach SCm and MRN
        col = ocol if src == "overlap" else (wcol if src in WHISKER_SRC else acol)
        y = np.arange(len(q))
        axb.barh(y, 100 * q.frac_of_zone, color=col, height=0.72, lw=0, edgecolor="none")
        fs = 4.3 if len(q) <= 8 else 3.6 if len(q) <= 12 else 3.1
        for yi, (s_, f_) in enumerate(zip(q.structure, q.frac_of_zone)):
            axb.text(100 * f_ + 1, yi, f"{s_} {100 * f_:.1f}%" if f_ < 0.1 else f"{s_} {100 * f_:.0f}%", va="center",
                     fontsize=fs)
        axb.set_ylim(len(q) - 0.4, -0.6)
        axb.set_xlim(0, max(60, 100 * q.frac_of_zone.max() * 1.9))
        axb.set_yticks([]); axb.spines["left"].set_visible(False)
        axb.tick_params(axis="x", labelsize=4.4, length=1.5)
        vol = float(z[src].sum() * r50 ** 3)
        axb.text(0.98, 0.02, f"zone: {vol:.1f} mm³", transform=axb.transAxes, ha="right", va="bottom", fontsize=4.5,
                 color="0.3")
        if r == len(rows) - 1:
            axb.set_xlabel("% of the zone volume", fontsize=4.8)
        else:
            axb.set_xticklabels([])
        if r == 0:
            axb.set_title("Largest recorded structures", fontsize=5.4, pad=1, loc="left")
    for k, (src, x0) in enumerate([(WHISKER_SRC[0], 0.07), (AUDITORY_SRC[0], 0.40)]):
        cax = fig.add_axes([x0 * 0.85, 0.3 / H, 0.22, 0.07 / H])
        cb = fig.colorbar(ims[src], cax=cax, orientation="horizontal")
        cb.set_label(("Whisker" if k == 0 else "Auditory") + "-cortex projection density (normalised, log)", fontsize=5.0)
        cb.ax.tick_params(labelsize=4.6, width=0.4, length=1.5); cb.outline.set_linewidth(0.4)
    h = [plt.Line2D([], [], color="k", lw=0.8, label=f"{ZONE_PCT} % contour"),
         plt.Rectangle((0, 0), 1, 1, color="#cfcfcf", lw=0, label="injected area (excluded)"),
         plt.Line2D([], [], color=wcol, lw=1, label=f"whisker zone (SSp-bfd + SSs merged, {ZONE_PCT} %)"),
         plt.Line2D([], [], color=acol, lw=1, label=f"auditory zone (AUDp + AUDd/v merged, {ZONE_PCT} %)"),
         plt.Rectangle((0, 0), 1, 1, color=ocol, alpha=0.6, lw=0, label="overlap")]
    fig.legend(handles=h, loc="lower right", bbox_to_anchor=(0.995, 0.0), frameon=False, fontsize=4.7, ncol=2,
               columnspacing=0.8, handlelength=1.6)
    word = "coronal" if kind == "cor" else "sagittal"
    fig.suptitle(f"Projection zones of whisker and auditory cortex, {word} 500-um slabs (Allen anterograde tracing, right "
                 f"hemisphere; white = no projection)", x=0.5 / W, y=1 - 0.06 / H, ha="left", va="top", fontsize=6.8,
                 weight="bold")
    S.save(fig, OUT / f"figures{ZTAG}", f"projection_zones_{'coronal' if kind == 'cor' else 'sagittal'}")
    plt.close(fig)


def main_figure_only():
    Z = np.load(OUT / f"projection_zones{ZTAG}.npz")
    finish({k: Z[k] for k in Z.files}, atlas_50um(), json.load(open(ATLAS / "structures.json")))


if __name__ == "__main__":
    import sys
    t0 = time.time()
    main_figure_only() if "--figure" in sys.argv else main()
    print(f"{time.time() - t0:.0f}s")
