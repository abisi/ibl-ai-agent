"""005 -- Is the barrel localization right? Global (NWB CCF) vs local (D2) registration, on each brain's own tissue and
against the whisker responses.

1. Tissue (per brain, compute step): the L4 slab of the brain's own autofluorescence image (channel 0, voxels within
   30 um of the L4 depth Lana's fit used, depth below the image pia from her pia_depth), averaged per flat pixel, in two
   frames on the same template grid:
     global frame = the brain's own flatmap before the barrel correction (flatcoords G): the Blue Brain atlas barrels
       land here as the coarse (brainreg) registration placed them, i.e. the registration behind the NWB CCF labels
       (the IBL alignment picks lie on that registration's atlas-space tracks);
     D2 frame = after Lana's barrel correction (G_corrected): the corrected barrels are the template itself.
   DiI (channel 1): 99th percentile per flat pixel over 150-1000 um below the pia (the probe's footprint in the column).
   Units: global = NWB CCF -> inverse deformation field -> sample voxel (001 `z, y, x`); D2 = the same unit on the
   physical DiI track (001 `trk_z, trk_y, trk_x`). Both are looked up in both frames, so the registration effect
   (frame) and the track effect (voxel) can be separated.
2. Figures: per brain, side by side on the same tissue: global outlines + global units | D2 outlines + D2 units, DiI in
   red, clicked DiI track points as a line. Population: Lana's fit QC (landmark error and Dice, coarse placement vs D2)
   and the functional test: the stimulus is the right C2 whisker (left barrel field), so whisker responses should
   peak in C2 and fall with distance from C2 under the right registration.
Outputs: tables/005_units_flat_<mouse>.parquet, tables/005_l4_<mouse>.npz, figures/validation/.
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 005_registration_validation.py compute [mice]   then   005_plot.py
"""
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

D2 = Path("/raid0/lmsmith/local_barrel_reg/D2_neuropixels")
_memmap = tifffile.memmap
tifffile.memmap = lambda *a, **k: _memmap(*a, **{"mode": "r", **k})  # Lana's files are read only
sys.path.insert(0, str(D2))
sys.path.insert(0, str(D2 / "code_snapshot"))
import tracks_to_barrels as ttb  # noqa: E402  (read only)
from barrelreg.imageproc import bandpass, tissue_mask_padded  # noqa: E402
from barrelreg.register import pia_depth  # noqa: E402

_spec = importlib.util.spec_from_file_location("maps003", Path(__file__).with_name("003_maps.py"))
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)

TABLES = m3.TABLES
OUT = m3.FIGS / "validation"
TPL = m3.TPL
LO, PIX, SHAPE = np.asarray(TPL["lo"], float), float(TPL["pix"]), TPL["alab"].shape  # (anterior, dorsal) grid


def bin_flat(F, v, how="mean"):
    """Values v at flat positions F (um, [anterior, dorsal]) -> template-grid image (mean or 99th percentile)."""
    ij = np.floor((F - LO) / PIX).astype(int)
    ok = np.isfinite(F).all(1) & (ij >= 0).all(1) & (ij < np.array(SHAPE)).all(1)
    idx = np.ravel_multi_index(ij[ok].T, SHAPE)
    out = np.full(np.prod(SHAPE), np.nan, np.float32)
    if how == "mean":
        s = np.bincount(idx, v[ok], np.prod(SHAPE))
        n = np.bincount(idx, minlength=np.prod(SHAPE))
        out[n > 0] = s[n > 0] / n[n > 0]
    else:
        o = np.argsort(idx, kind="stable")
        idx, val = idx[o], v[ok][o]
        starts = np.r_[0, np.flatnonzero(np.diff(idx)) + 1]
        for a, b in zip(starts, np.r_[starts[1:], len(idx)]):
            out[idx[a]] = np.percentile(val[a:b], 99)
    return out.reshape(SHAPE)


def mode_flat(F, lab, K=35):
    """Most frequent label (0 = none, 1-33 barrels, 34 septa) per template-grid pixel of the voxels at F."""
    ij = np.floor((F - LO) / PIX).astype(int)
    ok = np.isfinite(F).all(1) & (ij >= 0).all(1) & (ij < np.array(SHAPE)).all(1)
    idx = np.ravel_multi_index(ij[ok].T, SHAPE)
    n = np.bincount(idx * K + lab[ok].astype(int), minlength=np.prod(SHAPE) * K).reshape(-1, K)
    out = np.where(n.sum(1) > 0, n.argmax(1), 0)
    return out.reshape(SHAPE).astype(np.uint8)


def flat_at(G, lo_vox, V):
    g = np.rint(V).astype(int) - lo_vox
    ok = np.isfinite(V).all(1)
    ok[ok] = (g[ok] >= 0).all(1) & (g[ok] < np.array(G.shape[:3])).all(1)
    F = np.full((len(V), 2), np.nan)
    F[ok] = G[tuple(g[ok].T)]
    return F


def compute(mouse):
    inv = pd.read_csv(TABLES / "000_inventory.csv")
    folder = inv[inv.mouse_id == mouse].d2_folder.dropna().iloc[0]
    bd = D2 / folder
    met = json.load(open(bd / "h1" / "metrics.json"))
    fz = np.load(bd / "h1" / "flatcoords.npz")
    G, Gc, lo_vox = fz["G"], fz["G_corrected"], np.asarray(fz["crop20_lo"])
    sl = tuple(slice(int(a), int(a) + n) for a, n in zip(lo_vox, G.shape[:3]))
    reg = ttb.cfg.reg_dir(folder)
    ch0 = np.asarray(tifffile.memmap(ttb.cfg.ds_file(reg, 0))[sl]).astype(np.float32)
    ch1 = np.asarray(tifffile.memmap(ttb.cfg.ds_file(reg, 1))[sl]).astype(np.float32)
    atl = np.asarray(tifffile.memmap(f"{reg}/registered_atlas.tiff")[sl]).astype(np.int64)
    d2lab = np.asarray(tifffile.imread(bd / "barrels_corrected_full.tif")[sl])
    glab = np.zeros(atl.shape, np.uint8)  # coarse-registration (Blue Brain) barrel index, septa 34 inside SSp-bfd
    for i in np.unique(atl):
        k = ttb.BARREL_OF_ID.get(int(i), 0)
        glab[atl == i] = k if k else (34 if int(i) in ttb.BFD else 0)
    tissue = tissue_mask_padded(ch0, atl)
    depth, _ = pia_depth(tissue)
    have = np.isfinite(G[..., 0]) & tissue
    slab = have & (np.abs(depth - met["l4_depth_um"]) <= 30)
    col = have & (depth >= 150) & (depth <= 1000)
    out = {}
    for frame, F in (("glob", G), ("d2", Gc)):
        out[f"l4_{frame}"] = bin_flat(F[slab], ch0[slab])
        out[f"dii_{frame}"] = bin_flat(F[col], ch1[col], how="p99")
        out[f"lab_glob_{frame}"] = mode_flat(F[slab], glab[slab])
        out[f"lab_d2_{frame}"] = mode_flat(F[slab], d2lab[slab])
    # clicked DiI tracks (sample space) in both frames
    B = {"reg": reg}
    for space, rel, f in ttb.track_files(reg):
        if space == "sample_space":
            P = ttb.read_points(f)
            for frame, F in (("glob", G), ("d2", Gc)):
                out[f"track_{frame}::{rel}"] = flat_at(F, lo_vox, P)
    np.savez_compressed(TABLES / f"005_l4_{mouse}.npz", l4_depth_um=met["l4_depth_um"], **out)
    u = pd.read_parquet(TABLES / f"001_units_{mouse}.parquet")
    u = u[u.track_file.notna()].copy()
    for vox, cols in (("direct", ["z", "y", "x"]), ("trk", ["trk_z", "trk_y", "trk_x"])):
        V = u[cols].to_numpy(float)
        for frame, F in (("glob", G), ("d2", Gc)):
            f = flat_at(F, lo_vox, V)
            u[f"{frame}_{vox}_ant"], u[f"{frame}_{vox}_dor"] = f[:, 0], f[:, 1]
    u.to_parquet(TABLES / f"005_units_flat_{mouse}.parquet")
    del B
    return mouse, len(u)


def main_compute(mice):
    mice = mice or sorted(p.stem.split("_")[-1] for p in TABLES.glob("002_responses_*.parquet"))
    with ProcessPoolExecutor(6) as ex:
        for m, r in zip(mice, ex.map(_safe, mice)):
            print(m, r, flush=True)


def _safe(m):
    try:
        return compute(m)
    except Exception as e:  # noqa: BLE001
        return f"FAILED {type(e).__name__}: {e}"


if __name__ == "__main__":
    main_compute(sys.argv[2:] if sys.argv[1:2] == ["compute"] else sys.argv[1:])
