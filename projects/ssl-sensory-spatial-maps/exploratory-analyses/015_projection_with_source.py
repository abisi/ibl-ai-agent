"""015 -- Allen projection densities WITH the source areas included (user, 2026-11-19, Figure 2G round 3).

Same recipe as 002 (same experiments and cached volumes, injections mirrored to the right hemisphere, per-experiment
normalisation by the total over the candidate voxels, mean within each Cre line then over lines, Gaussian smoothing
sigma 50 um on the 50-um grid, right hemisphere, fibre tracts and ventricles excluded), except that the source area
(with its layers / barrels) is NOT excluded from its own map: candidate voxels = the whole right-hemisphere brain.
Merged modalities as 002: whisker = mean of the SSp-bfd and SSs densities, auditory = mean of AUDp and AUDd/v, each
normalised to its own total. Zone = 90 % mass zone (highest-density voxels holding 90 % of the projection mass, as the
002 ZONE_PCT=90 run), recomputed on these maps. Note that the injection site itself (projection_density includes the
labelled cell bodies / dendrites at the injection) is now part of the map and of the zone.
Output (002 outputs untouched): combined_results_ks4/_sensory_spatial_maps/projection_with_source/
  projection_with_source_zone90.npz: density_<src>, zone90_<src>, thr90_<src>, source_<src> for src in SSp-bfd, SSs,
  AUDp, AUD-sec, whisker, auditory (AP x DV x ML, 50 um) + summary.csv
Run (haas): ~/code/ibl-ai-agent/.venv/bin/python 015_projection_with_source.py
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("p002", HERE / "002_projection_zones.py")
p2 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(p2)
import json  # noqa: E402

MASS = 0.90
OUTD = p2.OUT / "projection_with_source"


def main():
    OUTD.mkdir(parents=True, exist_ok=True)
    E = pd.read_csv(p2.OUT / "projection_experiments.csv")         # the 002 experiment list (no API query)
    structs = json.load(open(p2.ATLAS / "structures.json"))
    ann = p2.atlas_50um()
    right = np.zeros(ann.shape, bool)
    right[:, :, int(p2.MID_UM / p2.RES_UM):] = True
    brain = (ann != 0) & ~np.isin(ann, list(p2.descendants(structs, [1009, 73]))) & right
    out, summ = {}, []

    def zone(d):
        v = np.sort(d[brain])[::-1]
        thr = v[np.searchsorted(np.cumsum(v) / v.sum(), MASS)]
        return brain & (d >= thr), np.float32(thr)

    for src, ids in p2.SOURCES.items():
        Es = E[E.source == src]
        L = []
        for line, g in Es.groupby("line"):
            V = []
            for eid, ml in zip(g.experiment_id, g.inj_ml):
                v = p2.download(eid)
                if ml < p2.MID_UM:
                    v = v[:, :, ::-1]
                v = np.where(brain, v, 0)
                V.append(v / v.sum())
            L.append(np.mean(V, 0))
        m = np.mean(L, 0)
        ms = (ndimage.gaussian_filter(m, p2.SMOOTH_UM / p2.RES_UM) * brain).astype(np.float32)
        z, thr = zone(ms)
        out[f"density_{src}"], out[f"zone90_{src}"], out[f"thr90_{src}"] = ms, z, thr
        out[f"source_{src}"] = np.isin(ann, list(p2.descendants(structs, ids))) & right
        summ.append(dict(source=src, n_experiments=len(Es), n_lines=int(Es.line.nunique()),
                         zone90_volume_mm3=float(z.sum() * (p2.RES_UM / 1000) ** 3),
                         source_share_of_mass=float(ms[out[f"source_{src}"]].sum() / ms.sum())))
        print(summ[-1], flush=True)
    for g, srcs in (("whisker", p2.WHISKER_SRC), ("auditory", p2.AUDITORY_SRC)):
        d = np.mean([out[f"density_{s_}"] / out[f"density_{s_}"][brain].sum() for s_ in srcs], axis=0).astype(np.float32)
        z, thr = zone(d)
        out[f"density_{g}"], out[f"zone90_{g}"], out[f"thr90_{g}"] = d, z, thr
        out[f"source_{g}"] = np.any([out[f"source_{s_}"] for s_ in srcs], axis=0)
        summ.append(dict(source=g, zone90_volume_mm3=float(z.sum() * (p2.RES_UM / 1000) ** 3),
                         source_share_of_mass=float(d[out[f"source_{g}"]].sum() / d.sum())))
        print(summ[-1], flush=True)
    np.savez_compressed(OUTD / "projection_with_source_zone90.npz", res_um=p2.RES_UM, **out)
    pd.DataFrame(summ).to_csv(OUTD / "summary.csv", index=False)
    print("ALL DONE", OUTD)


if __name__ == "__main__":
    main()
