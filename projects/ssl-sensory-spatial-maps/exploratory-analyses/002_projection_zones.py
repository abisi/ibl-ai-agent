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
Projection zone = top 10 % of the candidate voxels by mean normalised density (user, 2026-10-04).
Output: combined_results_ks4/_sensory_spatial_maps/projection_zones.npz (mask_<source>, density_<source>; AP x DV x ML,
50 um) + projection_experiments.csv + figures/projection_zones_qc.png
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
RES_UM, MID_UM, TOP = 50, 5700, 0.10
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
        out[f"mask_{src}"] = mask
        out[f"density_{src}"] = m.astype(np.float32)
        top_regions = pd.Series(ann[mask]).map({s["id"]: s["acronym"] for s in structs}).value_counts().head(12)
        summ.append(dict(source=src, n_experiments=len(ex), n_candidate_voxels=int(c.sum()), n_zone_voxels=int(mask.sum()),
                         zone_volume_mm3=float(mask.sum() * (RES_UM / 1000) ** 3), threshold=float(thr),
                         top_regions=top_regions.to_dict()))
        print(src, summ[-1], flush=True)
    np.savez_compressed(OUT / "projection_zones.npz", res_um=RES_UM, **out)
    pd.DataFrame(summ).to_csv(OUT / "projection_zones_summary.csv", index=False)
    qc(out, ann, structs)
    print("ALL DONE", OUT / "projection_zones.npz")


def qc(out, ann, structs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    srcs = list(SOURCES)
    aps = np.linspace(40, 220, 8).astype(int)
    fig, axs = plt.subplots(len(srcs), len(aps), figsize=(12, 1.6 * len(srcs)))
    for i, s in enumerate(srcs):
        for j, a in enumerate(aps):
            ax = axs[i, j]
            ax.imshow(ann[a] != 0, cmap="Greys", alpha=0.15, vmax=3)
            ax.imshow(np.ma.masked_where(out[f"density_{s}"][a] <= 0, np.log10(out[f"density_{s}"][a] + 1e-9)), cmap="magma_r")
            ax.contour(out[f"mask_{s}"][a], levels=[0.5], colors="c", linewidths=0.6)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(f"AP {a * RES_UM} um", fontsize=6)
        axs[i, 0].text(-0.1, 0.5, s, transform=axs[i, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "projection_zones_qc.png", dpi=150)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"{time.time() - t0:.0f}s")
