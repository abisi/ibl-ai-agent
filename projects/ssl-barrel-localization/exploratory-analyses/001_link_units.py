"""001 -- NWB units -> brain voxel -> Lana Smith's D2 barrel registration (barrel, layer, corrected CCF, flat L4 coordinate).

Per mouse with a D2 fit (000 inventory) and NWB units: every unit of a wS1-targeted probe, plus any unit labelled SSp-bfd.
1. Unit CCF position (NWB, um) -> atlas voxel (10 um) -> nearest brain voxel through the inverse of the brainreg
   deformation field (20 um brain grid, as Lana's AtlasToSample); `residual_um` = distance in the atlas between the unit
   and the atlas position of that voxel. Both NWB coordinate sets are mapped (`ccf_atlas_*` -> prefix `atl_`, `ccf_*` ->
   `alt_`) to find which one belongs to this registration.
2. At the voxel, Lana's `sample_table` (tracks_to_barrels.py, imported read-only): brainreg region / barrel, D2 barrel,
   D2 region and layer (Allen annotation at the corrected CCF position), corrected CCF (um), corrected flat L4 coordinate.
3. Track route (registration independent; the MH brains, and partly AB164, were registered again by Lana, so the NWB
   CCF positions, from the earlier registration, are offset in her brain space): each unit's distance from the brain
   surface along its probe, from the NWB channel line in the atlas (`d_surface_um`: PC1 line through the unit
   positions of the probe, entry where the Allen annotation leaves the brain), placed on the probe's own DiI track
   (Lana's line_samples: 10 um samples from the brain surface; extrapolated past the deepest clicked point). Cleared
   brains are shrunk relative to the atlas (~15%), so the distance is converted along the track: every track sample's
   atlas position (deformation field) gives its atlas arc length from the surface, and a unit goes to the sample at
   its own atlas distance (the registration sets only the scale along the track, not the position) -> voxel ->
   sample_table, prefix `trk_`. `d_insertion_um` = insertion depth (electrode-group metadata) minus the unit depth from
   the tip, for comparison. Track per probe: AB imec<N>.points, MH <session>/imec<N>.points.
4. QC: distance from the voxel to the nearest annotated DiI track (sample-space .points, 10 um, polyline through the
   clicked points ordered along their main axis) and its name; NWB area label vs the brainreg region at the voxel.

Output: combined_results_ks4/ssl-barrel-localization/tables/001_units_<mouse>.parquet (keys mouse_id, session_id,
electrode_group, cluster_id) and 001_link_qc.csv (one row per mouse x coordinate set).
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

NWB = axel_bisi_root() / "NWB_ks4"
OUT = axel_bisi_root() / "combined_results_ks4" / "ssl-barrel-localization" / "tables"
COORDS = {"atl": ("ccf_atlas_ap", "ccf_atlas_dv", "ccf_atlas_ml", "ccf_atlas_acronym"),
          "alt": ("ccf_ap", "ccf_dv", "ccf_ml", "ccf_acronym")}
UNIT_COLS = ["cluster_id", "bc_label", "ks_label", "peak_channel", "depth", "firing_rate"]


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
    """Units of the session's wS1-targeted probes (`groups`), plus any unit labelled SSp-bfd."""
    with h5py.File(NWB / f"{session_id}.nwb", "r") as f:
        u = f["units"]
        df = pd.DataFrame({c: _col(u, c) for c in UNIT_COLS + [c for cs in COORDS.values() for c in cs] if c in u})
        df["electrode_group"] = [f[r].name.split("/")[-1] for r in u["electrode_group"][:]]
    df["session_id"], df["mouse_id"] = session_id, session_id.split("_")[0]
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


def track_polylines(reg):
    """Sample-space DiI tracks: densified polyline (10 um voxels) through the clicked points, ordered along their PC1."""
    out = {}
    for space, rel, f in ttb.track_files(reg):
        if space != "sample_space":
            continue
        P = ttb.read_points(f)
        if len(P) < 2:
            continue
        c = P - P.mean(0)
        P = P[np.argsort(c @ np.linalg.svd(c, full_matrices=False)[2][0])]
        seg = [np.linspace(a, b, max(2, int(np.ceil(np.linalg.norm(b - a))) + 1)) for a, b in zip(P[:-1], P[1:])]
        out[rel] = np.concatenate(seg)
    return out


def nearest_track(vox, tracks):
    names, dist = np.full(len(vox), "", dtype=object), np.full(len(vox), np.nan)
    ok = np.isfinite(vox).all(1)
    for name, L in tracks.items():
        d, _ = cKDTree(L).query(vox[ok])
        d = d * 10.0
        better = ~(dist[ok] <= d)
        idx = np.flatnonzero(ok)[better]
        dist[idx], names[idx] = d[better], name
    return names, dist


def surface_distance(P):
    """Distance (um) of atlas-voxel points P from the brain surface along their PC1 line (Allen annotation == 0 outside)."""
    ok = np.isfinite(P).all(1)
    c = P[ok].mean(0)
    u = np.linalg.svd(P[ok] - c)[2][0]
    t = (P - c) @ u
    shp = np.array(ttb.ann.shape)

    def outside(q):
        q = np.rint(q).astype(int)
        return not ((q >= 0).all() and (q < shp).all()) or ttb.ann[tuple(q)] == 0

    def steps_out(p, d):
        return next((s for s in range(600) if outside(p + d * s)), 10 ** 6)

    ends = [(c + u * np.nanmin(t), -u), (c + u * np.nanmax(t), u)]
    (p0, out), = [min(ends, key=lambda e: steps_out(*e))]
    entry = p0 + out * steps_out(p0, out)
    return ((P - entry) @ -out) * 10.0


def track_for(rel_tracks, session_id, group):
    n = group.split("_")[0]  # imec<N>
    for rel in (f"{session_id}/{n}.points", f"{n}.points"):
        if rel in rel_tracks:
            return rel
    return None


def place_on_track(B, P_track, d_um):
    """Voxel of the DiI track whose atlas distance from the brain surface is d_um (atlas um); past the deepest clicked
    point the track is extended along its last 200 um with the local scale. Returns voxels and the sample/atlas scale."""
    Sm, _ = ttb.line_samples(P_track, B["brain"])
    if Sm is None:
        return np.full((len(d_um), 3), np.nan), np.nan
    q = np.clip(np.rint(Sm).astype(int), 0, np.array(B["field"][0].shape) - 1)
    A = np.stack([np.asarray(f[tuple(q.T)], np.float64) for f in B["field"]], 1) * 1000.0  # atlas um
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(A, axis=0), axis=1))]
    arc = np.maximum.accumulate(arc)
    n = min(20, len(Sm) - 1)
    um_per_sample = (arc[-1] - arc[-1 - n]) / n  # atlas um per 10 um track sample near the deep end
    k = np.where(d_um <= arc[-1], np.interp(d_um, arc, np.arange(len(Sm))), len(Sm) - 1 + (d_um - arc[-1]) / um_per_sample)
    step = (Sm[-1] - Sm[-1 - n]) / n
    vox = np.array([Sm[int(round(i))] if i <= len(Sm) - 1 else Sm[-1] + step * (i - len(Sm) + 1) for i in np.nan_to_num(k)])
    vox[~np.isfinite(d_um)] = np.nan
    return vox, 10.0 / um_per_sample


def link_mouse(mouse, inv_table):
    rows = inv_table[(inv_table.mouse_id == mouse) & inv_table.session_id.notna()]
    folder = rows.d2_folder.iloc[0]
    ws1 = rows[rows.target_area == "wS1"]  # targets are per session: the same imec<N> can go elsewhere on other days
    units = pd.concat([load_units(s, set(ws1[ws1.session_id == s].electrode_group)) for s in sorted(rows.session_id.unique())],
                      ignore_index=True)
    on_ws1 = pd.Series(list(zip(units.session_id, units.electrode_group))).isin(set(zip(ws1.session_id, ws1.electrode_group))).to_numpy()
    depth_of = rows.set_index(["session_id", "electrode_group"]).insertion_depth_um.to_dict()
    B = ttb.load_brain(folder)
    inverse, tracks = Inverse(B["reg"]), track_polylines(B["reg"])
    qc = []
    for tag, (ap, dv, ml, acr) in COORDS.items():
        P = units[[ap, dv, ml]].to_numpy(float) / 10.0
        vox, res = inverse(P)
        T = ttb.sample_table(B, np.nan_to_num(vox)).drop(columns=["dist_from_surface_um", "dist_from_track_end_um"])
        T = T.where(pd.Series(np.isfinite(vox).all(1), index=T.index), axis=0)  # units outside the field: NaN
        T["residual_um"] = res
        T["track"], T["track_dist_um"] = nearest_track(vox, tracks)
        T["label_agrees"] = [str(a) == str(b) for a, b in zip(units[acr], T.brainreg_region)]
        T["parent_agrees"] = [str(a).rstrip("123456ab/") == str(b).rstrip("123456ab/") for a, b in
                              zip(units[acr], T.brainreg_region)]
        units = units.join(T.add_prefix(f"{tag}_"))
        bfd = on_ws1
        qc.append(dict(mouse_id=mouse, d2_folder=folder, coords=tag, n_units=len(units),
                       residual_median_um=float(np.nanmedian(res)), track_dist_median_um=float(np.nanmedian(T.track_dist_um[bfd])),
                       label_agree=float(T.label_agrees.mean()), parent_agree=float(T.parent_agrees.mean()),
                       n_in_d2_barrel=int((T.d2_barrel.fillna("") != "").sum()), n_left=int((T.hemisphere == 1).sum())))
    # track route (per probe)
    rel_tracks = {rel: f for space, rel, f in ttb.track_files(B["reg"]) if space == "sample_space"}
    B["field"] = [tifffile.memmap(f"{B['reg']}/deformation_field_{k}.tiff") for k in range(3)]
    units["trk_scale"] = np.nan
    units["d_surface_um"], units["d_insertion_um"], units["trk_file"] = np.nan, np.nan, None
    vox = np.full((len(units), 3), np.nan)
    for (sid, grp), idx in units.groupby(["session_id", "electrode_group"]).groups.items():
        P = units.loc[idx, list(COORDS["atl"][:3])].to_numpy(float) / 10.0
        if np.isfinite(P).all(1).sum() < 3:
            continue
        units.loc[idx, "d_surface_um"] = surface_distance(P)
        units.loc[idx, "d_insertion_um"] = float(depth_of.get((sid, grp), np.nan)) - units.loc[idx, "depth"].astype(float)
        rel = track_for(rel_tracks, sid, grp)
        if rel is None or (sid, grp) not in set(zip(ws1.session_id, ws1.electrode_group)):
            continue
        units.loc[idx, "trk_file"] = rel
        v, scale = place_on_track(B, ttb.read_points(rel_tracks[rel]), units.loc[idx, "d_surface_um"].to_numpy())
        vox[units.index.get_indexer(idx)], units.loc[idx, "trk_scale"] = v, scale
    T = ttb.sample_table(B, np.nan_to_num(vox)).drop(columns=["dist_from_surface_um", "dist_from_track_end_um"])
    T = T.where(pd.Series(np.isfinite(vox).all(1), index=T.index), axis=0)
    units = units.join(T.add_prefix("trk_"))
    on = units.trk_d2_barrel.fillna("") != ""
    for tag in COORDS:
        both = on & (units[f"{tag}_d2_barrel"].fillna("") != "")
        qc[[q["coords"] for q in qc].index(tag)].update(
            trk_n_in_d2_barrel=int(on.sum()), barrel_agrees_with_trk=float((units[f"{tag}_d2_barrel"] == units.trk_d2_barrel)[both].mean()))
    OUT.mkdir(parents=True, exist_ok=True)
    units.to_parquet(OUT / f"001_units_{mouse}.parquet")
    return pd.DataFrame(qc)


def main(mice):
    inv = pd.read_csv(OUT / "000_inventory.csv")
    inv = inv[~inv.d2_variant]  # second registrations of the same brain (e.g. MH025cmdcds) are not used
    mice = mice or sorted(inv[inv.target_area == "wS1"].mouse_id.unique())
    qcs = []
    for m in mice:
        q = link_mouse(m, inv)
        print(q.to_string(index=False), flush=True)
        qcs.append(q)
    qc_file = OUT / "001_link_qc.csv"
    old = pd.read_csv(qc_file) if qc_file.exists() else pd.DataFrame()
    pd.concat([old[~old.mouse_id.isin(mice)] if len(old) else old, *qcs], ignore_index=True).to_csv(qc_file, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
