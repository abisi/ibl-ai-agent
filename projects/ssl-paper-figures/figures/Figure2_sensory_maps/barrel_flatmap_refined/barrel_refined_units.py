"""Refined (D2, user-picked) barrel identity + refined flat position per SSp-bfd unit of the Fig 2G barrel row.
unit CCF (NWB, global) -> sample voxel of the mouse's brain (nearest voxel whose GLOBAL brainreg CCF matches, among
barrel-field-column voxels) -> barrels_corrected_full (refined barrel) + ccf_field (refined CCF) -> Blue Brain flatmap ->
D2 template plane. Only use_local mice; others keep NaN."""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / "code/wS1_barrel_registration"))
from barrelreg.atlas import BARREL_NAMES  # noqa: E402

FIG = Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/Figure2_sensory_maps_explore_2026-11-17/G_maps")
REG = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ws1-barrel-registration/D2_final_user_picked")
DATA = Path("/mnt/lsens-analysis/Axel_Bisi/data")
OUT = FIG / "barrel_flatmap_refined"
MID = 5700.0
TOL_UM = 25.0


def fold(ml):
    return MID + np.abs(ml - MID)


def per_mouse(args):
    mouse, units = args
    exp = sorted(DATA.glob(f"{mouse}/*/Anatomy/wS1_barrel_registration/registration_info.json"))
    if not exp:
        return mouse, None, "no export"
    info = json.load(open(exp[0]))
    if info["decision"] != "use_local":
        return mouse, None, info["decision"]
    d = exp[0].parent
    rp = pd.read_csv(REG.parent / "D2_all_brains/registration_params.csv").set_index("brain")
    regdir = Path(rp.loc[info["brain"], "reg_dir"])
    c0 = tifffile.imread(d / "ccf_field_0.tif")
    idx = np.flatnonzero(np.isfinite(c0))
    ref = np.empty((len(idx), 3), np.float32)
    ref[:, 0] = c0.ravel()[idx]
    del c0
    for k in (1, 2):
        ref[:, k] = tifffile.imread(d / f"ccf_field_{k}.tif").ravel()[idx]
    glob_ = np.empty_like(ref)
    for k in range(3):
        glob_[:, k] = tifffile.memmap(regdir / f"deformation_field_{k}.tiff", mode="r").ravel()[idx]
    bar = tifffile.imread(d / "barrels_corrected_full.tif").ravel()[idx]
    G = glob_ * 1000.0
    G[:, 2] = fold(G[:, 2])
    tree = cKDTree(G)
    P = np.c_[units.ccf_atlas_ap, units.ccf_atlas_dv, fold(units.ccf_atlas_ml.to_numpy())]
    dist, j = tree.query(P)
    R = ref[j] * 1000.0
    R[:, 2] = fold(R[:, 2])
    res = pd.DataFrame({"nn_um": dist, "ref_ap": R[:, 0], "ref_dv": R[:, 1], "ref_ml": R[:, 2],
                        "barrel_refined": [BARREL_NAMES[b - 1] if 0 < b <= len(BARREL_NAMES) else "" for b in bar[j]]},
                       index=units.index)
    res.loc[dist > TOL_UM, ["ref_ap", "ref_dv", "ref_ml"]] = np.nan
    res.loc[dist > TOL_UM, "barrel_refined"] = ""
    return mouse, res, f"{info['decision']} n={len(units)} matched={(dist <= TOL_UM).sum()} med_nn={np.median(dist):.0f}um"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    s = pd.read_csv(FIG / "barrel_flatmap_example_bfd_only/barrel_flatmap_units.csv")   # SSp-bfd units, global flat pos
    s = s.rename(columns={"barrel": "barrel_global", "e1": "e1_global", "e2": "e2_global"})
    parts = []
    with ProcessPoolExecutor(4) as ex:
        for mouse, res, msg in ex.map(per_mouse, list(s.groupby("mouse_id"))):
            print(mouse, msg, flush=True)
            if res is not None:
                parts.append(res)
    s = s.join(pd.concat(parts))
    # refined CCF -> Blue Brain flatmap -> template plane
    CROP = Path("/raid0/lmsmith/local_barrel_reg/D2_allen")
    FM = np.load(CROP / "flatmap_atlas_crop.npy", mmap_mode="r")
    lo = np.load(CROP / "flatmap_atlas_crop_lo.npy")
    L = np.load(REG.parent / "D2_all_brains/template.npz")["L"]
    ok = s.ref_ap.notna().to_numpy()
    ijk = np.floor(s.loc[ok, ["ref_ap", "ref_dv", "ref_ml"]].to_numpy() / 10).astype(int) - lo
    F = FM[ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float64)
    T = np.c_[F, np.ones(len(F))] @ L
    good = np.isfinite(F).all(1) & (F >= 0).all(1)
    s["e1_refined"], s["e2_refined"] = np.nan, np.nan
    s.loc[s.index[ok][good], "e1_refined"] = T[good, 0]
    s.loc[s.index[ok][good], "e2_refined"] = T[good, 1]
    s.to_csv(OUT / "barrel_refined_units.csv", index=False)
    r = s[s.barrel_refined.fillna("") != ""]
    print(f"{len(s)} SSp-bfd units; refined barrel for {len(r)} ({r.mouse_id.nunique()} mice); "
          f"same barrel as global: {(r.barrel_refined == r.barrel_global).mean():.0%}")
