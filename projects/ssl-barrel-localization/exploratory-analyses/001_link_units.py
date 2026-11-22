"""001 -- NWB units -> global CCF (IBL ephys alignment) -> brain voxel -> final user-picked D2 barrel registration.

Chain (verified 2026-11-11 on AB156, AB164, MH064):
- Each unit's NWB `ccf_atlas_ap/dv/ml` is the aligned position of its peak channel in channel_locations.json (IBL ephys
  alignment, surface and white matter used), converted from IBL coordinates (um from bregma) with the bregma offset only:
  ap = 5400 - y, dv = 332 - z, ml = 5739 + x (checked here per unit against the electrodes table).
- The alignment's track picks (xyz_picks.json) lie on the atlas-space DiI tracks of the brainreg registration Lana's
  barrel fit starts from (median 5-11 um), so that registration's deformation field maps the units back into the brain.
1. Global CCF -> atlas voxel (10 um) -> nearest brain voxel (inverse of the brainreg deformation field, 20 um brain grid);
   `residual_um` = atlas distance between the unit and that voxel's atlas position.
2. Global labels: Blue Brain barrels annotation (the registration atlas) at the global CCF position -> `glob_region`,
   `glob_barrel`, `glob_layer`, and the global flat position `glob_flat_{anterior,dorsal}_um` (Blue Brain flatmap ->
   template frame). Brains whose local fit was rejected (final decision use_global) are placed with these.
3. Barrel-specific labels at the voxel (Lana's sample_table, tracks_to_barrels.py, imported read only): D2 barrel, D2
   region / layer (annotation at the corrected CCF), corrected CCF (`ccf_*_um`), corrected flat L4 coordinate, brainreg
   region. `shift_*_um` = corrected - global CCF (the local barrel correction at the unit).
4. Track transfer (`trk_` columns): the atlas-space and sample-space DiI tracks of one brainglobe segmentation disagree
   by a median 85-176 um under the deformation field (checked 2026-11-11), so step 1 can land off the physical track.
   Each unit is projected on the atlas-space track (the alignment's picks lie on it), its arc fraction between the
   first (dorsal) and last clicked points is carried to the same fraction of the sample-space track (extrapolated past
   the ends with the sample / atlas length ratio), and Lana's labels are read there; `trk_shift_*` = corrected - global.
5. QC: perpendicular distance from the voxel to the probe's sample-space DiI track line (PC1 through the clicked points;
   AB imec<N>.points, MH <session>/imec<N>.points), in brain um.

Output: combined_results_ks4/ssl-barrel-localization/tables/001_units_<mouse>.parquet (keys mouse_id, session_id,
electrode_group, cluster_id) and 001_link_qc.csv (one row per mouse).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 001_link_units.py AB156 AB164 MH064   [no args: all mice]
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

D2 = Path("/raid0/lmsmith/local_barrel_reg/D2_neuropixels")
_memmap = tifffile.memmap
tifffile.memmap = lambda *a, **k: _memmap(*a, **{"mode": "r", **k})  # Lana's files are read only (default mode is r+)
sys.path.insert(0, str(D2))
import tracks_to_barrels as ttb  # noqa: E402  (Lana's module: config, atlas tables, load_brain, sample_table; read only)

# Barrel fits: the final user-picked registration (ws1-barrel-registration/D2_final_user_picked, 2026-11-22), not Lana's
# D2_neuropixels run: Lana's lookup functions read <brain>/results there (use_local brains; use_global brains have no
# fit, so their D2 columns stay empty). Brainreg folders from the registration's brain list (covers every brain).
FINAL = axel_bisi_root() / "combined_results_ks4" / "ws1-barrel-registration" / "D2_final_user_picked"
_REG = pd.read_csv(FINAL.parent / "D2_all_brains" / "registration_params.csv").set_index("brain").reg_dir.to_dict()
ttb.cfg.brain_dir = lambda b: str(FINAL / b / "results")
ttb.cfg.reg_dir = lambda b: _REG[b]

NWB = axel_bisi_root() / "NWB_ks4"
OUT = axel_bisi_root() / "combined_results_ks4" / "ssl-barrel-localization" / "tables"
CCF = ["ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml"]
UNIT_COLS = ["cluster_id", "bc_label", "ks_label", "peak_channel", "depth", "firing_rate", "ccf_atlas_acronym"] + CCF


def _col(u, name):
    x = u[name][:]
    if x.dtype.kind in "SO":
        x = np.array([v.decode() if isinstance(v, bytes) else v for v in x], dtype=object)
        try:
            return x.astype(float)
        except ValueError:
            return x
    return x


def load_units(session_id, groups):
    """Units of the session's wS1-targeted probes (`groups`), plus any unit labelled SSp-bfd; `peak_channel_check_um` =
    distance between the unit position and its peak channel's position in the electrodes table."""
    with h5py.File(NWB / f"{session_id}.nwb", "r") as f:
        u = f["units"]
        df = pd.DataFrame({c: _col(u, c) for c in UNIT_COLS})
        df["electrode_group"] = [f[r].name.split("/")[-1] for r in u["electrode_group"][:]]
        e = f["general/extracellular_ephys/electrodes"]
        el = pd.DataFrame({"group": _col(e, "group_name"), "ch": e["index_on_probe"][:],
                           **{c: e[c][:].astype(float) for c in CCF}})
    df["session_id"], df["mouse_id"] = session_id, session_id.split("_")[0]
    probe = df.electrode_group.str.split("_").str[0]
    ch = el.set_index(["group", "ch"])[CCF]
    look = ch.reindex(list(zip(probe, df.peak_channel.astype(int)))).to_numpy()
    df["peak_channel_check_um"] = np.linalg.norm(look - df[CCF].to_numpy(float), axis=1)
    keep = df.electrode_group.isin(groups) | df.ccf_atlas_acronym.astype(str).str.startswith("SSp-bfd")
    return df[keep].reset_index(drop=True)


class Inverse:
    """Atlas voxel (10 um) -> nearest brain voxel of the brainreg deformation field (20 um brain grid), with residual."""

    def __init__(self, reg):
        self.F = np.stack([tifffile.memmap(f"{reg}/deformation_field_{k}.tiff")[::2, ::2, ::2] for k in range(3)],
                          -1).astype(np.float32) * 100  # mm -> atlas voxel

    def __call__(self, P):
        lo, hi = np.nanmin(P, 0) - 30, np.nanmax(P, 0) + 30
        m = ((self.F >= lo) & (self.F <= hi)).all(-1)
        vox, res = np.full((len(P), 3), np.nan), np.full(len(P), np.nan)
        ok = np.isfinite(P).all(1)
        if m.any() and ok.any():
            d, i = cKDTree(self.F[m]).query(P[ok])
            vox[ok], res[ok] = np.argwhere(m)[i] * 2, d * 10
        return vox, res


def track_line_distance(vox, P):
    """Perpendicular distance (um, brain space) of voxels to the PC1 line through the clicked track points P."""
    c = P.mean(0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    r = vox - c
    return np.linalg.norm(r - np.outer(r @ u, u), axis=1) * 10.0


def ordered(P):
    """Clicked points ordered along their PC1, and the polyline arc length."""
    c = P.mean(0)
    u = np.linalg.svd(P - c, full_matrices=False)[2][0]
    P = P[np.argsort((P - c) @ u)]
    return P, np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]


def project_arc(X, P, arc):
    """Arc position of points X on polyline P (nearest segment; extrapolated along the end segments)."""
    out = np.full(len(X), np.nan)
    for j, x in enumerate(X):
        best = (np.inf, np.nan)
        for i in range(len(P) - 1):
            a, b = P[i], P[i + 1]
            L = np.linalg.norm(b - a)
            t = (x - a) @ (b - a) / max(L ** 2, 1e-9)
            tt = t if (i == 0 and t < 0) or (i == len(P) - 2 and t > 1) else np.clip(t, 0, 1)
            d = np.linalg.norm(x - (a + tt * (b - a)))
            if d < best[0]:
                best = (d, arc[i] + tt * L)
        out[j] = best[1]
    return out


def point_at_arc(P, arc, s):
    """Points at arc positions s along polyline P (linear extrapolation past the ends)."""
    i = np.clip(np.searchsorted(arc, s) - 1, 0, len(P) - 2)
    L = arc[i + 1] - arc[i]
    t = (s - arc[i]) / np.where(L > 0, L, 1)
    return P[i] + t[:, None] * (P[i + 1] - P[i])


def transfer_to_sample_track(units_ccf_vox, A, S, field):
    """Atlas-space track A, sample-space track S (clicked points of the same track): units -> sample voxels."""
    A, arcA = ordered(A)
    S, arcS = ordered(S)
    q = np.clip(np.rint(S[[0, -1]]).astype(int), 0, np.array(field[0].shape) - 1)
    dv_ends = np.asarray(field[1][tuple(q.T)])  # atlas DV of the two sample ends: smaller = dorsal (surface) end
    if dv_ends[0] > dv_ends[1]:
        S, arcS = S[::-1], arcS[-1] - arcS[::-1]
    if A[0, 1] > A[-1, 1]:
        A, arcA = A[::-1], arcA[-1] - arcA[::-1]
    f = project_arc(units_ccf_vox, A, arcA) / arcA[-1]
    return point_at_arc(S, arcS, f * arcS[-1])


_FM = {}


def global_flat(v):
    """global CCF voxels (10 um, ap / dv / ml) -> position in the template flat frame (um): Blue Brain flatmap at the
    voxel (one CCF hemisphere, ml index >= 572; the other is mirrored) through the template's flatmap -> plane affine L.
    Every isocortex voxel carries its column's flat position, so this is the global registration's flat position."""
    if not _FM:
        _FM.update(F=np.load(ttb.cfg.flatmap_crop, mmap_mode="r"), lo=np.load(ttb.cfg.flatmap_crop_lo))
    v = v.copy()
    mid = ttb.ann.shape[2] // 2
    v[:, 2] = np.where(v[:, 2] < mid, ttb.ann.shape[2] - 1 - v[:, 2], v[:, 2])
    g = v - _FM["lo"]
    ok = (g >= 0).all(1) & (g < np.array(_FM["F"].shape[:3])).all(1)
    F = np.full((len(v), 2), np.nan)
    F[ok] = np.asarray(_FM["F"][tuple(g[ok].T)], np.float64)
    F[~(np.isfinite(F).all(1) & (F >= 0).all(1))] = np.nan
    T = np.c_[F, np.ones(len(F))] @ ttb.TPL["L"]
    return T[:, 0], T[:, 1]


def global_labels(units):
    """Labels and flat position under the global registration (the NWB CCF): used for brains whose local fit was
    rejected (decision use_global)."""
    v = np.clip(np.rint(units[CCF].to_numpy(float) / 10).astype(int), 0, np.array(ttb.ann.shape) - 1)
    rid = np.asarray(ttb.ann[tuple(v.T)]).astype(np.int64)
    fa, fd = global_flat(v)
    return pd.DataFrame({"glob_region": [ttb.acronym(i) for i in rid], "glob_barrel": [ttb.barrel(i) for i in rid],
                         "glob_layer": [ttb.layer(i) for i in rid], "glob_flat_anterior_um": fa,
                         "glob_flat_dorsal_um": fd}, index=units.index)


def link_mouse(mouse, inv_table):
    rows = inv_table[(inv_table.mouse_id == mouse) & inv_table.session_id.notna()]
    folder = rows.d2_folder.iloc[0]
    ws1 = rows[rows.target_area == "wS1"]  # targets are per session: the same imec<N> can go elsewhere on other days
    units = pd.concat([load_units(s, set(ws1[ws1.session_id == s].electrode_group)) for s in sorted(rows.session_id.unique())],
                      ignore_index=True)
    on_ws1 = pd.Series(list(zip(units.session_id, units.electrode_group))).isin(set(zip(ws1.session_id, ws1.electrode_group))).to_numpy()
    B = ttb.load_brain(folder)
    vox, res = Inverse(B["reg"])(units[CCF].to_numpy(float) / 10.0)
    T = ttb.sample_table(B, np.nan_to_num(vox)).drop(columns=["dist_from_surface_um", "dist_from_track_end_um"])
    T = T.where(pd.Series(np.isfinite(vox).all(1), index=T.index), axis=0)  # units outside the field: NaN
    units = units.join(global_labels(units)).join(T)
    units["residual_um"] = res
    for k, c in zip(("ap", "dv", "ml"), CCF):
        units[f"shift_{k}_um"] = units[f"ccf_{k}_um"] - units[c]
    units["shift_um"] = np.linalg.norm(units[["shift_ap_um", "shift_dv_um", "shift_ml_um"]].to_numpy(float), axis=1)
    rel_tracks = {rel: f for space, rel, f in ttb.track_files(B["reg"]) if space == "sample_space"}
    atl_tracks = {rel: f for space, rel, f in ttb.track_files(B["reg"]) if space == "atlas_space"}
    atl_tracks.update({rel: f"{B['reg']}/segmentation/atlas_space/tracks/{rel}" for rel in rel_tracks
                       if Path(f"{B['reg']}/segmentation/atlas_space/tracks/{rel}").exists()})
    field = [tifffile.memmap(f"{B['reg']}/deformation_field_{k}.tiff") for k in range(3)]
    tvox = np.full((len(units), 3), np.nan)
    units["track_file"], units["track_line_dist_um"] = None, np.nan
    for (sid, grp), idx in units.groupby(["session_id", "electrode_group"]).groups.items():
        n = grp.split("_")[0]
        rel = next((r for r in (f"{sid}/{n}.points", f"{n}.points") if r in rel_tracks), None)
        if rel is None:
            continue
        units.loc[idx, "track_file"] = rel
        units.loc[idx, "track_line_dist_um"] = track_line_distance(vox[units.index.get_indexer(idx)],
                                                                   ttb.read_points(rel_tracks[rel]))
        if rel in atl_tracks:
            tvox[units.index.get_indexer(idx)] = transfer_to_sample_track(
                units.loc[idx, CCF].to_numpy(float) / 10.0, ttb.read_points(atl_tracks[rel]), ttb.read_points(rel_tracks[rel]), field)
    T = ttb.sample_table(B, np.nan_to_num(tvox)).drop(columns=["dist_from_surface_um", "dist_from_track_end_um"])
    T = T.where(pd.Series(np.isfinite(tvox).all(1), index=T.index), axis=0)
    units = units.join(T.add_prefix("trk_"))
    for k, c in zip(("ap", "dv", "ml"), CCF):
        units[f"trk_shift_{k}_um"] = units[f"trk_ccf_{k}_um"] - units[c]
    units["trk_shift_um"] = np.linalg.norm(units[["trk_shift_ap_um", "trk_shift_dv_um", "trk_shift_ml_um"]].to_numpy(float), axis=1)
    # which registration places this brain's units: its final user-picked D2 fit, or the global one (use_global)
    units["registration"] = "D2" if B["met"] is not None else "global"
    OUT.mkdir(parents=True, exist_ok=True)
    units.to_parquet(OUT / f"001_units_{mouse}.parquet")
    bar = on_ws1 & (units.d2_barrel.fillna("") != "")
    both = bar & (units.glob_barrel.fillna("") != "")
    return pd.DataFrame([dict(
        mouse_id=mouse, d2_folder=folder, n_units=len(units), n_ws1_probe=int(on_ws1.sum()),
        peak_channel_check_max_um=float(units.peak_channel_check_um.max()),
        residual_median_um=float(np.nanmedian(res)),
        track_line_dist_median_um=float(np.nanmedian(units.track_line_dist_um[on_ws1])),
        n_in_d2_barrel=int(bar.sum()), n_in_glob_barrel=int((on_ws1 & (units.glob_barrel.fillna("") != "")).sum()),
        barrel_same=float((units.glob_barrel == units.d2_barrel)[both].mean()) if both.any() else np.nan,
        layer_same=float((units.glob_layer == units.d2_layer)[bar].mean()) if bar.any() else np.nan,
        shift_median_um=float(np.nanmedian(units.shift_um[bar])) if bar.any() else np.nan,
        trk_n_in_d2_barrel=int((on_ws1 & (units.trk_d2_barrel.fillna("") != "")).sum()),
        trk_barrel_same_glob=float((units.trk_d2_barrel == units.glob_barrel)[on_ws1 & (units.trk_d2_barrel.fillna("") != "") & (units.glob_barrel.fillna("") != "")].mean()),
        trk_barrel_same_direct=float((units.trk_d2_barrel == units.d2_barrel)[on_ws1 & (units.trk_d2_barrel.fillna("") != "") & bar].mean()),
        trk_shift_median_um=float(np.nanmedian(units.trk_shift_um[on_ws1 & (units.trk_d2_barrel.fillna("") != "")])))])


def main(mice):
    inv = pd.read_csv(OUT / "000_inventory.csv")
    inv = inv[~inv.d2_variant]  # second registrations of the same brain (e.g. MH025cmdcds) are not used
    mice = mice or sorted(inv[inv.target_area == "wS1"].mouse_id.unique())
    qcs = []
    for m in mice:
        try:
            q = link_mouse(m, inv)
        except Exception as err:  # e.g. no D2 fit or registration folder for this brain: logged, next brain
            print(f"{m}: FAILED {type(err).__name__}: {err}", flush=True)
            continue
        print(q.to_string(index=False), flush=True)
        qcs.append(q)
    qc_file = OUT / "001_link_qc.csv"
    old = pd.read_csv(qc_file) if qc_file.exists() else pd.DataFrame()
    keep = old[~old.mouse_id.isin(mice)] if len(old) and "barrel_same" in old else pd.DataFrame()
    pd.concat([keep, *qcs], ignore_index=True).to_csv(qc_file, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
