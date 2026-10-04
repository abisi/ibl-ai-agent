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
Projection zone (user, 2026-10-04, second version): the mean density is smoothed (Gaussian, sigma 100 um) and the zone
is the 70 % contour = the highest-density voxels that together hold 70 % of the source's projection (candidate voxels).
The first version (top 10 % of candidate voxels, unsmoothed) is kept as mask_<source>.
Overlap of whisker and auditory projections: union of the SSp-bfd and SSs zones vs union of the AUDp and AUDd/AUDv zones;
overlap volume per Allen structure (layers merged) -> projection_overlap.csv.
Output: combined_results_ks4/_sensory_spatial_maps/projection_zones.npz (zone70_<source>, mask_<source>,
density_<source> smoothed; AP x DV x ML, 50 um) + projection_experiments.csv + projection_overlap.csv +
figures/projection_zones.{png,pdf,svg} (densities and 70 % contours on coronal sections; last row: overlap)
"""
import io
import json
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
RES_UM, MID_UM, TOP, MASS, SMOOTH_UM = 50, 5700, 0.10, 0.70, 100.0
WHISKER_SRC, AUDITORY_SRC = ["SSp-bfd", "SSs"], ["AUDp", "AUD-sec"]
SRC_NAME = {"SSp-bfd": "SSp-bfd\n(barrel cortex)", "SSs": "SSs (secondary\nsomatosensory)", "AUDp": "AUDp (primary\nauditory)",
            "AUD-sec": "AUDd + AUDv\n(secondary auditory)", "overlap": "Overlap: whisker\nand auditory zones"}
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
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
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
    np.savez_compressed(OUT / "projection_zones.npz", res_um=RES_UM, **out)
    pd.DataFrame(summ).to_csv(OUT / "projection_zones_summary.csv", index=False)
    overlap_table(out, ann, structs)
    figure(out)
    print("ALL DONE", OUT / "projection_zones.npz")


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


def overlap_table(out, ann, structs):
    w = np.logical_or.reduce([out[f"zone70_{s}"] for s in WHISKER_SRC])
    a = np.logical_or.reduce([out[f"zone70_{s}"] for s in AUDITORY_SRC])
    ov = w & a
    lab = merged_ids(ann)
    acr = {s["id"]: s["acronym"] for s in structs}
    name = {s["id"]: s["name"] for s in structs}
    v = (RES_UM / 1000) ** 3
    right = np.zeros(ann.shape, bool)
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
    pairs = {f"{x}&{y}": float((out[f"zone70_{x}"] & out[f"zone70_{y}"]).sum() * v) for x in WHISKER_SRC for y in AUDITORY_SRC}
    T.to_csv(OUT / "projection_overlap.csv", index=False)
    json.dump(dict(whisker_union_mm3=float(w.sum() * v), auditory_union_mm3=float(a.sum() * v), overlap_mm3=float(ov.sum() * v),
                   pairwise_overlap_mm3=pairs), open(OUT / "projection_overlap_summary.json", "w"), indent=1)
    print("overlap", float(ov.sum() * v), "mm3;", pairs)
    print(T.head(25).round(3).to_string())


def figure(out):
    """densities (white = no projection) and smoothed 70 % contours on coronal sections of the right hemisphere"""
    import cmasher as cmr
    from matplotlib.colors import ListedColormap, LogNorm
    from scipy import ndimage
    m3 = m3_module()
    S = m3.S
    plt = S.setup()
    A = m3.Atlas(load_zones=False)
    cm = {s: cmr.get_sub_cmap("cmr.sunburst_r", 0.0, 0.9) for s in WHISKER_SRC}
    cm.update({s: cmr.get_sub_cmap("cmr.freeze_r", 0.0, 0.9) for s in AUDITORY_SRC})
    line = {s: cm[s](0.8) for s in cm}
    aps = np.arange(2750, 10251, 1000)
    rows = WHISKER_SRC + AUDITORY_SRC + ["overlap"]
    W = S.W_IN
    pw = (W - 0.6) / len(aps)
    ph = pw * 7.3 / 5.8
    H = ph * len(rows) + 0.85
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(len(rows), len(aps), wspace=0.02, hspace=0.03, left=0.5 / W, right=1 - 0.1 / W,
                          top=1 - 0.32 / H, bottom=0.55 / H)
    r50 = RES_UM / 1000
    i0 = int(MID_UM / RES_UM)
    ny, nx = out[f"density_{WHISKER_SRC[0]}"].shape[1], out[f"density_{WHISKER_SRC[0]}"].shape[2] - i0
    yy, xx = (np.arange(ny) + 0.5) * r50, (np.arange(nx) + 0.5) * r50
    ext = (0, nx * r50, ny * r50, 0)
    ims = {}

    def contour(ax, mask2d, col, lw=0.8):
        if mask2d.any():
            ax.contour(xx, yy, ndimage.gaussian_filter(mask2d.astype(float), 1.5), levels=[0.5], colors=[col], linewidths=lw,
                       zorder=5)

    for r, src in enumerate(rows):
        for c, ap in enumerate(aps):
            ax = fig.add_subplot(gs[r, c])
            sec = A.section("cor", ap)
            i = int(round(ap / RES_UM))
            if src != "overlap":
                d = out[f"density_{src}"][i][:, i0:]
                thr = float(out[f"thr70_{src}"])
                vmax = float(np.quantile(out[f"density_{src}"][out[f"density_{src}"] > 0], 0.999))
                ims[src] = ax.imshow(np.ma.masked_where(d < thr / 30, d), extent=ext, cmap=cm[src],
                                     norm=LogNorm(thr / 30, vmax), interpolation="bilinear", zorder=1)
                srcm = out[f"source_{src}"][i][:, i0:]
                if srcm.any():
                    ax.imshow(np.ma.masked_where(~srcm, srcm.astype(float)), extent=ext, cmap=ListedColormap(["#cfcfcf"]),
                              interpolation="nearest", zorder=1.2)
                m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
                contour(ax, out[f"zone70_{src}"][i][:, i0:], "k", lw=0.6)
            else:
                w = np.logical_or.reduce([out[f"zone70_{s}"][i] for s in WHISKER_SRC])[:, i0:]
                a = np.logical_or.reduce([out[f"zone70_{s}"][i] for s in AUDITORY_SRC])[:, i0:]
                ov = ndimage.gaussian_filter((w & a).astype(float), 1.5) >= 0.5
                ax.imshow(np.ma.masked_where(~ov, ov.astype(float)), extent=ext, cmap=ListedColormap(["#9b6bb5"]), alpha=0.6,
                          interpolation="nearest", zorder=1)
                m3.draw_section(ax, A, sec, (0, 5.7, 8.0, 0))
                contour(ax, w, line[WHISKER_SRC[0]])
                contour(ax, a, line[AUDITORY_SRC[0]])
            ax.set_xlim(0, 5.8); ax.set_ylim(7.3, 0); ax.set_aspect("equal"); ax.set_axis_off()
            if r == 0:
                ax.set_title(f"AP {(5400 - ap) / 1000:+.2f} mm", fontsize=5.6, pad=1)
            if c == 0:
                ax.text(-0.04, 0.5, SRC_NAME[src], transform=ax.transAxes, rotation=90, ha="right", va="center", fontsize=5.4)
            if r == len(rows) - 1 and c == 0:
                ax.plot([0.2, 1.2], [7.15, 7.15], color="k", lw=0.9, solid_capstyle="butt")
                ax.text(0.7, 7.0, "1 mm", ha="center", va="bottom", fontsize=4.8)
    for k, (src, x0) in enumerate([(WHISKER_SRC[0], 0.07), (AUDITORY_SRC[0], 0.42)]):
        cax = fig.add_axes([x0, 0.25 / H, 0.27, 0.07 / H])
        cb = fig.colorbar(ims[src], cax=cax, orientation="horizontal")
        cb.set_label(("Whisker" if k == 0 else "Auditory") + "-cortex projection density (log)", fontsize=5.0)
        cb.ax.tick_params(labelsize=4.6, width=0.4, length=1.5); cb.outline.set_linewidth(0.4)
    h = [plt.Line2D([], [], color=line[WHISKER_SRC[0]], lw=1, label="whisker zone (70 % contour)"),
         plt.Line2D([], [], color=line[AUDITORY_SRC[0]], lw=1, label="auditory zone (70 % contour)"),
         plt.Rectangle((0, 0), 1, 1, color="#9b6bb5", alpha=0.6, lw=0, label="overlap"),
         plt.Line2D([], [], color="k", lw=0.8, label="70 % contour (rows 1-4)"),
         plt.Rectangle((0, 0), 1, 1, color="#cfcfcf", lw=0, label="injected area (excluded)")]
    fig.legend(handles=h, loc="lower right", bbox_to_anchor=(0.995, 0.005), frameon=False, fontsize=4.8, ncol=1)
    fig.suptitle("Projection zones of whisker and auditory cortex (Allen anterograde tracing, right hemisphere; white = no projection)",
                 x=0.5 / W, y=1 - 0.06 / H, ha="left", va="top", fontsize=7, weight="bold")
    S.save(fig, OUT / "figures", "projection_zones")
    plt.close(fig)


def main_figure_only():
    Z = np.load(OUT / "projection_zones.npz")
    out = {k: Z[k] for k in Z.files}
    import tifffile  # noqa: F401
    overlap_table(out, atlas_50um(), json.load(open(ATLAS / "structures.json")))
    figure(out)


if __name__ == "__main__":
    import sys
    t0 = time.time()
    main_figure_only() if "--figure" in sys.argv else main()
    print(f"{time.time() - t0:.0f}s")
