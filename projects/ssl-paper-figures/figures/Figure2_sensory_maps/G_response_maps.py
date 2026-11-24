"""Figure 2G: ROC response maps of individual neurons, whisker and auditory, five brain regions x six columns
(user spec 2026-11-18, revisions 2026-11-19 rounds 2 and 3). Standalone 183-mm-wide figure (placed under 2A-F in Figma).

Rows (views "mixed per region"):
  Cortex          Allen CCFv3 butterfly flatmap of the isocortex (ssl-sensory-spatial-maps 007 coordinates,
                  ccf_streamlines): left half = whisker (left hemisphere seen from above), right half = auditory (mirror)
  Barrel field    top-down CCF view of SSp-bfd (x = ML lateral of the midline, folded; y = AP, anterior up); outlines
                  = layer-4 barrels of the 10-um barrel atlas (allen_mouse_bluebrain_barrels_10um; most dorsal L4 voxel
                  of each AP-ML column). Neurons are plotted at their own CCF (ccf_atlas_ap, ccf_atlas_ml folded) from
                  _roc_stage_analysis/units.parquet (probe-track registration), with no projection onto a plane or
                  along the cortical column; depth (DV) is not shown, so all layers are included and overlaid. Units:
                  area_acronym_custom == SSp-bfd (all layers) + units on the flatmap (isocortex by atlas voxel) whose
                  (AP, ML) falls within 200 um outside the SSp-bfd footprint, all probes. The row therefore contains
                  SSp-bfd (~6,700) plus neighbouring cortical areas under or around the footprint (SSs, SSp-oro,
                  PPC-relabelled SSp-bfd / SSp-tr units of allen_utils 7ae712d, AUD); only isocortex area groups are
                  kept (CORTICAL_GROUPS), so subcortical units below the footprint (~175: TS, GPe, DLS) are dropped;
                  units need not be in SSp-bfd.
                  Neighbouring areas in the schematic = the first isocortex structure
                  from the dorsal surface, labelled in the schematic). Projection: L4 voxels inside, 400 um below the
                  cortical surface in the rim. Barrel identity is from the CCF atlas, not from the per-mouse wS1
                  barrel registration (ws1-barrel-registration D2_final; per-unit barrel labels do not exist yet).
                  Whisker | auditory side by side.
  Striatum        sagittal view, Striatum-group units of the slab ML 0.8-3.6 mm (folded; covers VS ~1 mm, DMS
                  ~1.5-2 mm, DLS ~2.5-3.3 mm and TS ~3-3.5 mm); background section at ML 2.0 mm, black contour = striatum
                  (area group) in any section of the slab. Chosen over a coronal 1-mm slab at the striatal-unit centroid
                  (AP +0.1: 3,663 vs 15,027 striatal units shown; G_STRIATUM_VIEW=cor switches)
  Frontal         SLAB-DEFINED: every stable day-0 neuron in the coronal slab AP +0.80 to +2.50 mm from bregma
                  (1.7 mm thick; section drawn at the slab centre, AP +1.65), inside the brain section (no area-group
                  filter: wM2 / ALM / MOs / frontal units all count); left half = whisker, right = auditory
  Midbrain        SLAB-DEFINED: every neuron in the 500-um sagittal slab centred on the SCs + SCm unit centroid (folded
                  ML 0.87 mm), inside the brain section; cropped from the dorsal brain surface to below the midbrain
                  units; SCs, SCm and MRN outlined
Columns:
  Area groups     the section / view coloured by area group (allen_utils palette, lightened), left hemisphere only;
                  left inset: dorsal brain outline with the plane (line) or region (box) and its coordinate (bregma)
  Recorded neurons  every neuron of the row (grey dots, alpha 0.35, rasterized). Neurons per row: Cortex = isocortex
                  units on the flatmap; Barrel field = SSp-bfd units + the 200-um rim; Striatum = Striatum area group;
                  Frontal / Midbrain = slab-defined (above). In every row, neurons whose 2-D position falls outside the
                  brain contour of the shown section / view (barrel: outside footprint + rim) are dropped
  Allen projection  projection density WITH the source areas included (015: the 002 recipe -- Allen anterograde
                  tracing, line-balanced, per-experiment normalised, smoothed sigma 50 um, 50-um grid -- without excluding
                  the source and its layers / barrels from its own map; projection_with_source/
                  projection_with_source_zone90.npz); whisker = SSp-bfd + SSs, auditory = AUDp + AUDd/v, in every row
                  including the barrel row. The source holds 26-42 % of the projection mass, so the colour scale tops
                  at the 99.5th percentile of the (non-zero) brain voxels (higher values saturate) instead of 99.9;
                  slab mean, square-root scale, white -> modality colour -> near black. Dotted grey = source outline.
                  Dashed = 90 % mass zone recomputed on these maps (highest-density voxels holding 90 % of the mass,
                  injection site included; Chen et al. 2024 Cell Fig. 4 / S4 instead draw "a contour line 90 % of
                  peak density"), drawn where >= half of the slab (flatmap: of the streamline) is inside the zone
  Selectivity     every neuron coloured by its signed ROC selectivity (whisker_active / auditory_active), one
                  symmetric bwr scale shared by both modalities, strongest on top
  Selective neurons  ROC-significant neurons (either sign) coloured by |selectivity| on a light -> modality colour ->
                  dark ramp (whisker orange, auditory blue), strongest on top; non-significant neurons grey
  Response onset (latenZy)  latenZy onset latency (ZETA pipeline, unit_spikes_analysis/zeta_analysis:
                  combined_results_ks4/<mouse>/whisker_0/zeta/<mouse>_zeta_results.csv, analysis_type whisker_active /
                  auditory_active, column latency_s; latenZy window -100..200 ms, negative latencies not allowed),
                  joined on session_id + electrode_group + NWB cluster_id. Units: ROC-responsive (sig) AND excited
                  (sel > 0) AND latenzy_p < 0.05 with a non-NaN onset; other neurons light grey. Log colour scale
                  5-100 ms (clipped below 5 ms: whisker spikes before +5 ms are artefact-replaced; and above 100 ms),
                  inferno reversed (early = bright), earliest on top
Units: day 0, stable set (skill ssl-valid-data, 137_stable_units `stable`, joined on nwb_cluster_id), good + mua, cohorts
pooled, ML folded onto one hemisphere. Dot layers rasterized, text vector.
Insets: dorsal brain outline per row with the plane / slab (shaded band, true bounds in the text, relative to bregma);
a band narrower than 0.8 mm on the page (the 500-um slabs) is drawn 0.8 mm wide for visibility.
Dot positions: unit positions are channel positions and overplot, so every dot is jittered by a Gaussian offset (SD
10 um, clipped at +-25 um) in each plotted dimension (seeded per row, JITTER_SEED; flatmap: in flatmap units); the same
jitter in every column; applied before the in-brain test. Dot sizes (pt^2): recorded neurons 0.10, coloured data dots
0.18, grey background 0.028; alpha 0.6 for the recorded-neuron dots (was 0.35); coloured data dots and grey background dots opaque (all
rasterized).
Area-group labels: abbreviations (MO, FRO, OLF, RSP, Ss-w/-o/-b, Hip, MB, Str, Pal, TH, Amy/Hy, P/M, ...) beside the
section, joined to their region by horizontal leader lines (as SuppFig 2 area_group_schematic.label_rows).
(Earlier versions showed the 010 PSTH peak time t_peak; replaced by the latenZy onset, user 2026-11-19.)
Variant --onset-all (G_response_maps_onset_all.*, + _counts.csv): the onset column shows every neuron of the row with a
significant latenZy onset (latenzy_p < 0.05, latency_s >= 0), regardless of ROC significance and response sign; ZETA
`direction` negative (decreases) = open circles under the filled dots (increases; direction unknown also filled).
Output: manuscript Figures/Figure2_sensory_maps_explore_2026-11-17/G_maps/G_response_maps.* + G_maps_units.csv (source
data, per unit and row) + G_maps_summary.csv. Cache: combined_results_ks4/ssl-paper-figures/Figure2/G_maps.pkl
Run (haas): PYTHONPATH=<NAS ephys_utilities>:~/py_extra ~/code/unit_spikes_analysis/.venv/bin/python G_response_maps.py
            [--plot-only] [--png-only]
  compute (units, maps, flatmap projection) -> CACHE; prep (contour lines, background / schematic images, label
  positions; rebuilt when CACHE changes or PREP_VERSION is bumped) -> G_maps_prep.pkl; plot only draws.
  --png-only writes G_response_maps_quick.png (iterations); without it all formats (final).
"""
import argparse
import importlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, Normalize, PowerNorm, to_rgb  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
EA = REPO / "projects/ssl-sensory-spatial-maps/exploratory-analyses"
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(EA))
sys.path.insert(0, str(Path.home() / "py_extra"))
import paperstyle as ps  # noqa: E402
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402

m3 = importlib.import_module("003_spatial_maps")
ROOT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
ROC = ROOT / "_roc_stage_analysis"
STABLE = ROOT / "ssl-whisker-hitmiss-timeresolved-decoding/tables/137_stable_units.parquet"
SSM = ROOT / "_sensory_spatial_maps"
LAT = SSM / "fast_slow_peaks/units.parquet"
FLAT = SSM / "flatmap_units_zone90.parquet"
ZONES = SSM / "projection_with_source/projection_with_source_zone90.npz"   # 015: sources included, 90 % mass zones
PROJ_PCT = 99.5                                           # top of the projection colour scale (percentile of the brain)
ASSETS = Path("/mnt/lsens-analysis/Axel_Bisi/Anatomy/ccf_streamlines_assets")
OUT = ROOT.parent / "manuscripts/Bisi_Hamon/Figures/Figure2_sensory_maps_explore_2026-11-17/G_maps"
CACHE = ROOT / "ssl-paper-figures/Figure2/G_maps.pkl"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
MODS = {"w": "whisker", "a": "auditory"}
C_W, C_A = ps.COLORS["whisker"], ps.COLORS["auditory"]
Z_MIN = 5.0
MID, BREGMA_AP, G = m3.MID, m3.BREGMA_AP, m3.GRID          # um
SLAB = 1000.0
FR_AP = (0.80, 2.50)                                      # frontal slab, mm anterior of bregma (user 2026-11-19)
MB_SLAB, STR_COR_SLAB = 500.0, 1000.0
STRIATUM_VIEW = __import__("os").environ.get("G_STRIATUM_VIEW", "sag")                                   # "cor" or "sag" (both cached; see the report)
STR_ML = (800.0, 3600.0)                                  # striatum sagittal slab, lateral of the midline (um)
STR_DRAW_ML = 2000.0
RIM_UM, RIM_NORMAL_UM = 200.0, 1000.0
RESPELL = {"Somatosensory-orofacial": "Ss-orofacial", "Somatosensory-body": "Ss-body",
           "Somatosensory-whisker": "Ss-whisker"}
ROW_LABELS = {"striatum": {"Str"}, "midbrain": {"MB"},             # schematic labels kept per row (group level)
              "cortex": {"MO", "FRO", "Ss-w", "Ss-o", "Ss-b", "AUD", "VIS", "RSP", "PPC", "INS"}}
FRONTAL_LABELS = {"MOs": "MOs", "MOp": "MOp", "ACAd": "ACAd", "ACAv": "ACAv", "PL": "PL", "ILA": "ILA",
                  "ORBm": "ORBm", "ORBl": "ORBl", "ORBvl": "ORBvl", "FRP": "FRP"}
NAME = "G_response_maps"
ROWS_ORDER = ["cortex", "barrel", "striatum", "frontal", "midbrain"]
LABEL_FIRST = {"frontal": "FRO", "striatum": "Str", "midbrain": "MB"}   # schematic label kept first
CORTICAL_GROUPS = ["Motor areas", "Frontal areas", "Somatosensory-orofacial", "Somatosensory-body",
                   "Somatosensory-whisker", "Auditory areas", "Retrosplenial areas", "Posterior parietal areas",
                   "Visual areas", "Insular areas"]
ROW_GROUP = {"striatum": "Striatum"}         # area-group filter per row (frontal / midbrain rows are slab-defined)
ROWS = [("cortex", "Cortex"), ("barrel", "Barrel field"), ("striatum", "Striatum"), ("frontal", "Frontal"),
        ("midbrain", "Midbrain")]
COLS = ["schematic", "coverage", "projection", "sel", "frac", "lat"]
COL_TITLES = {"schematic": "Area groups", "coverage": "Recorded neurons", "projection": "Allen projection",
              "sel": "Selectivity", "frac": "Selective neurons", "lat": "Response onset"}


# ----------------------------------------------------------------------------------------------------- data
def load_units():
    u = pd.read_parquet(ROC / "units.parquet", columns=KEYS + ["day", "quality_label", "area_acronym_custom", "area_group",
                                                              "ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"])
    u = u[(u.day == 0) & u.quality_label.isin(["good", "mua"])].drop_duplicates(KEYS)
    u["cluster_id"] = u.cluster_id.astype(str)
    st = pd.read_parquet(STABLE, columns=KEYS[:3] + ["nwb_cluster_id", "stable"])
    st["cluster_id"] = st.nwb_cluster_id.astype(str)
    u = u.merge(st[st.stable][KEYS], on=KEYS)
    r = pd.read_parquet(ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sel", "sig"],
                        filters=[("analysis_type", "in", ["whisker_active", "auditory_active"])])
    r["cluster_id"] = r.cluster_id.astype(str)
    r = r.drop_duplicates(KEYS + ["analysis_type"])
    for k, m in MODS.items():
        x = r[r.analysis_type == f"{m}_active"][KEYS + ["sel", "sig"]].rename(columns={"sel": f"sel_{k}", "sig": f"sig_{k}"})
        u = u.merge(x, on=KEYS, how="left")
    L = pd.read_parquet(LAT, columns=KEYS + ["modality", "t_peak_ms", "z_peak", "cls"])
    L["cluster_id"] = L.cluster_id.astype(str)
    L = L[(L.z_peak >= Z_MIN) & (L.cls != "edge")]
    for k, m in MODS.items():
        x = L[L.modality == m][KEYS + ["t_peak_ms"]].drop_duplicates(KEYS).rename(columns={"t_peak_ms": f"lat_{k}"})
        u = u.merge(x, on=KEYS, how="left")
    f = pd.read_parquet(FLAT, columns=KEYS + ["flat_x", "flat_y"])
    f["cluster_id"] = f.cluster_id.astype(str)
    u = u.merge(f.drop_duplicates(KEYS), on=KEYS, how="left")
    u["ml_f"] = MID + np.abs(u.ccf_atlas_ml - MID)
    u["lat_mm"], u["ap_mm"], u["dv_mm"] = (u.ml_f - MID) / 1000, u.ccf_atlas_ap / 1000, u.ccf_atlas_dv / 1000
    return u.reset_index(drop=True)


def group_lut(A):
    """atlas id -> area group (pipeline labelling: allen_utils.create_area_custom_column)"""
    S = A.S
    acr = dict(zip(S.id, S.acronym))
    rows = []
    for i, p in zip(S.id, S.structure_id_path):
        path = [int(x) for x in str(p).strip("/").split("/") if x]
        rows.append(dict(id=i, ccf_atlas_acronym=acr[i], target_region=None,
                         ccf_atlas_parent_acronym=acr.get(path[-2], acr[i]) if len(path) > 1 else acr[i]))
    df = pd.DataFrame(rows)
    keep = ~df.ccf_atlas_acronym.isin(au.get_excluded_areas())
    lab = au.create_area_custom_column(df[keep].copy())
    df["area_acronym_custom"] = lab["area_acronym_custom"].reindex(df.index)
    df["area_group"] = df.area_acronym_custom.map(au.get_custom_area_groups_from_name())
    return dict(zip(df.id, df.area_group)), dict(zip(df.ccf_atlas_acronym, df.area_group))


def to_group(lab, lut):
    uq, inv = np.unique(lab, return_inverse=True)
    return np.array([lut.get(int(i)) for i in uq], dtype=object)[inv].reshape(lab.shape)


def slab_mean(vol, kind, lo_um, hi_um):
    """50-um volume (AP, DV, ML) -> cor (DV, lateral of the midline) / sag (DV, AP), mean over [lo_um, hi_um]
    (cor: AP um; sag: folded ML um)"""
    i0, i1 = int(np.floor(lo_um / G)), int(np.ceil(hi_um / G))
    if kind == "cor":
        return vol[max(i0, 0):i1].astype(np.float32).mean(0)[:, int(MID / G):]
    return vol[:, :, max(i0, 0):i1].astype(np.float32).mean(2).T


# ----------------------------------------------------------------------------------------------------- rows
def maps_section(d, P, Zs, kind, lo, hi):
    for k in MODS:
        d[f"projection_{k}"] = slab_mean(P[k], kind, lo, hi)
        d[f"zone_{k}"] = slab_mean(Zs[f"zone_{k}"], kind, lo, hi) >= 0.5
        d[f"source_{k}"] = slab_mean(Zs[f"source_{k}"], kind, lo, hi) >= 0.5
    d["map_extent"] = (0, d[f"projection_w"].shape[1] * G / 1000, d[f"projection_w"].shape[0] * G / 1000, 0)


def cor_row(u, A, P, Zs, c, lut, slab=SLAB):
    lab = A.section("cor", c)                                        # 10 um (DV, lateral), right hemisphere
    d = {"kind": "cor", "c": c, "slab": slab, "lab": lab, "inside": lab > 0, "group": to_group(lab, lut),
         "extent": (0, lab.shape[1] / 100, lab.shape[0] / 100, 0), "outlines": []}
    maps_section(d, P, Zs, "cor", c - slab / 2, c + slab / 2)
    s = u[(u.ccf_atlas_ap - c).abs() <= slab / 2]
    d["units"] = s.assign(row_x_mm=s.lat_mm, row_y_mm=s.dv_mm)
    w, h = d["extent"][1], d["extent"][2]
    d["crop"] = (0, w, h, 0)
    return d


def sag_row(u, A, P, Zs, c_draw, lo, hi, lut):
    lab = A.section("sag", c_draw)                                   # 10 um (DV, AP)
    d = {"kind": "sag", "c": c_draw, "ml": (lo, hi), "lab": lab, "inside": lab > 0, "group": to_group(lab, lut),
         "extent": (0, lab.shape[1] / 100, lab.shape[0] / 100, 0), "outlines": []}
    maps_section(d, P, Zs, "sag", lo, hi)
    s = u[(u.ml_f >= lo) & (u.ml_f <= hi)]
    d["units"] = s.assign(row_x_mm=s.ap_mm, row_y_mm=s.dv_mm)
    return d


def flat_row(u, P, Zs, acr_group):
    import ccf_streamlines.projection as ccfproj
    from matplotlib.path import Path as MPath
    view = dict(view_space_for_other_hemisphere="flatmap_butterfly")
    bf = ccfproj.BoundaryFinder(projected_atlas_file=str(ASSETS / "flatmap_butterfly.nrrd"),
                                labels_file=str(ASSETS / "labelDescription_ITKSNAPColor.txt"))
    B = {k: np.asarray(v) for k, v in bf.region_boundaries(hemisphere="left", **view).items()}
    allp = np.vstack(list(B.values()))
    x0, x1 = int(allp[:, 0].min()) - 10, int(allp[:, 0].max()) + 10
    y0, y1 = int(allp[:, 1].min()) - 10, int(allp[:, 1].max()) + 10
    nx, ny = x1 - x0, y1 - y0
    gx, gy = np.meshgrid(np.arange(nx) + x0 + 0.5, np.arange(ny) + y0 + 0.5)
    pts = np.c_[gx.ravel(), gy.ravel()]
    inside = np.zeros(len(pts), bool)
    lab = np.zeros(len(pts), object)
    for nm, v in B.items():
        m = MPath(v).contains_points(pts)
        inside |= m
        lab[m] = nm
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(inside.reshape(ny, nx), iterations=4))
    lab = lab.reshape(ny, nx)
    group = np.vectorize(lambda a: acr_group.get(a) if isinstance(a, str) else None, otypes=[object])(lab)
    s = u[u.flat_x.notna()]
    d = {"kind": "flat", "B": {k: (v - [x0, y0]) * 0.01 for k, v in B.items()}, "inside": inside, "group": group,
         "extent": (0, nx * 0.01, ny * 0.01, 0)}
    d["map_extent"] = d["extent"]
    d["crop"] = (0, nx * 0.01, ny * 0.01, 0)
    p2 = ccfproj.Isocortex2dProjector(str(ASSETS / "flatmap_butterfly.h5"), str(ASSETS / "surface_paths_10_v3.h5"),
                                      hemisphere="left", **view)

    def flat(vol50):
        vol = np.repeat(np.repeat(np.repeat(vol50[:, :, ::-1].astype(np.float32), 5, 0), 5, 1), 5, 2)[:1320, :800, :1140]
        img = np.asarray(p2.project_volume(vol, kind="mean")).T                     # [y, x]
        del vol
        out = np.full((ny, nx), np.nan)
        ys, xs = np.arange(ny) + y0, np.arange(nx) + x0
        okx, oky = (xs >= 0) & (xs < img.shape[1]), (ys >= 0) & (ys < img.shape[0])
        out[np.ix_(oky, okx)] = img[np.ix_(ys[oky], xs[okx])]
        return np.where(inside, out, np.nan)
    for k in MODS:
        d[f"projection_{k}"] = flat(P[k])
        d[f"zone_{k}"] = np.nan_to_num(flat(Zs[f"zone_{k}"])) >= 0.5
        d[f"source_{k}"] = np.nan_to_num(flat(Zs[f"source_{k}"])) >= 0.5
    d["units"] = s.assign(row_x_mm=(s.flat_x - x0) * 0.01, row_y_mm=(s.flat_y - y0) * 0.01)
    return d


def barrel_row(u, A, Z, Zs, norm99, lut):
    """Top-down CCF view of the barrel field (x = lateral ML of the folded right hemisphere, y = AP, anterior up).
    Outlines: layer-4 barrels of the 10-um barrel atlas, the most dorsal L4 voxel of each (AP, ML) column. Neurons are
    placed at their own (AP, ML) CCF coordinates (no projection onto a plane; DV is not shown)."""
    S = A.S
    l4 = S[S.acronym.str.match(r"^SSp-bfd-.+-4$")]
    sept = [int(i) for i in S.loc[S.acronym == "SSp-bfd4", "id"]]
    bname = {int(i): a.replace("SSp-bfd-", "").replace("-4", "") for i, a in zip(l4.id, l4.acronym)}
    ids = list(bname) + sept
    ann = A.ann
    mid10 = int(MID / 10)
    sub = ann[:, :, mid10:]
    cand = np.where(np.isin(sub[::4, ::4, ::4], ids))
    lo = np.maximum(np.array([c.min() for c in cand]) * 4 - 8, 0)
    hi = np.array([c.max() for c in cand]) * 4 + 8
    blk = sub[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    vi = np.where(np.isin(blk, ids))
    xyz = (np.c_[vi[0] + lo[0], vi[1] + lo[1], vi[2] + lo[2] + mid10] * 10.0 + 5.0)   # um (AP, DV, ML)
    lid = blk[vi]
    PX, M = 20.0, 320.0
    Xmin, Xmax = (xyz[:, 2] - MID).min() - M, (xyz[:, 2] - MID).max() + M      # lateral of the midline (um)
    Ymin, Ymax = xyz[:, 0].min() - M, xyz[:, 0].max() + M                      # AP (um), anterior = small
    nx, ny = int((Xmax - Xmin) / PX), int((Ymax - Ymin) / PX)
    ix, iy = ((xyz[:, 2] - MID - Xmin) / PX).astype(int), ((xyz[:, 0] - Ymin) / PX).astype(int)
    bar = np.zeros((ny, nx), np.int64)
    isb = np.isin(lid, list(bname))
    o = np.lexsort((-xyz[:, 1], isb))                                # barrels over septa, the most dorsal voxel last
    bar[iy[o], ix[o]] = lid[o]
    inside = ndimage.binary_fill_holes(ndimage.binary_closing(bar > 0, iterations=3))
    bar = np.where(inside & (bar == 0), sept[0] if sept else 1, bar)
    dil = ndimage.binary_dilation(inside, iterations=int(RIM_UM / PX))
    rim = dil & ~inside
    # rim: area of the first isocortex voxel from the dorsal surface in each column; sampled 400 um below it
    iso = {i for i, pth in A.path.items() if 315 in pth}
    gy_, gx_ = np.where(dil)
    ap10 = np.clip(((Ymin + (gy_ + 0.5) * PX) / 10).astype(int), 0, ann.shape[0] - 1)
    ml10 = np.clip(((MID + Xmin + (gx_ + 0.5) * PX) / 10).astype(int), 0, ann.shape[2] - 1)
    rlab = np.zeros((ny, nx), np.int64)
    depth = np.full(len(gy_), -1)
    for n_, (a_, m_) in enumerate(zip(ap10, ml10)):
        if not rim[gy_[n_], gx_[n_]]:
            continue
        col = ann[a_, :, m_]
        k = np.flatnonzero(np.isin(col, list(iso)))
        if len(k):
            depth[n_] = k[0]
            if rim[gy_[n_], gx_[n_]]:
                rlab[gy_[n_], gx_[n_]] = int(A.merged(np.array([col[k[0]]]))[0])
    group = to_group(rlab, lut)
    group[inside] = "Somatosensory-whisker"
    d = {"kind": "barrel", "bar": bar, "bname": bname, "inside": inside, "dil": dil, "rlab": rlab, "group": group,
         "extent": (0, nx * PX / 1000, ny * PX / 1000, 0), "origin_um": (Xmin, Ymin)}
    d["map_extent"] = d["extent"]
    d["crop"] = d["extent"]
    g50_l4 = np.clip((xyz / G).astype(int), 0, np.array(m3.VOL_SHAPE) - 1)
    okd = depth >= 0
    rim_sel = rim[gy_, gx_] & okd
    rim_xyz = np.c_[ap10 * 10.0, (depth + 40) * 10.0, ml10 * 10.0][rim_sel]
    g50_rim = np.clip((rim_xyz / G).astype(int), 0, np.array(m3.VOL_SHAPE) - 1)

    def sample(vol, scale=1.0):
        num, den = np.zeros((ny, nx)), np.zeros((ny, nx))
        np.add.at(num, (iy, ix), vol[g50_l4[:, 0], g50_l4[:, 1], g50_l4[:, 2]].astype(float) / scale)
        np.add.at(den, (iy, ix), 1)
        img = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        img = np.where(inside, ndimage.generic_filter(img, np.nanmean, size=3, mode="nearest"), np.nan)
        img[gy_[rim_sel], gx_[rim_sel]] = vol[g50_rim[:, 0], g50_rim[:, 1], g50_rim[:, 2]].astype(float) / scale
        return img
    for k, m in MODS.items():                                        # full whisker / auditory maps (015)
        d[f"projection_{k}"] = sample(Z[f"density_{m}"], norm99[f"density_{m}"])
        d[f"zone_{k}"] = np.nan_to_num(sample(Zs[f"zone_{k}"])) >= 0.5
        d[f"source_{k}"] = np.nan_to_num(sample(Zs[f"source_{k}"])) >= 0.5
    # units at their own (AP, ML): SSp-bfd units (all layers) + isocortex units in the 200-um rim (all layers)
    X = (u.ml_f.to_numpy() - MID - Xmin) / PX
    Y = (u.ccf_atlas_ap.to_numpy() - Ymin) / PX
    jx, jy = np.floor(X).astype(int), np.floor(Y).astype(int)
    okp = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
    in_dil, in_bfd = np.zeros(len(u), bool), np.zeros(len(u), bool)
    in_dil[okp] = dil[jy[okp], jx[okp]]
    in_bfd[okp] = inside[jy[okp], jx[okp]]
    keep = in_dil & ((u.area_acronym_custom == "SSp-bfd") | u.flat_x.notna()).to_numpy()
    s = u[keep].copy()
    s["row_x_mm"], s["row_y_mm"] = X[keep] * PX / 1000, Y[keep] * PX / 1000
    s["in_bfd_contour"] = in_bfd[keep]
    d["units"] = s
    cen = {}
    for i, nm in bname.items():
        w = np.where(bar == i)
        if len(w[0]):
            cen[nm] = (w[1].mean() * PX / 1000, w[0].mean() * PX / 1000)
    d["centroids"] = cen
    d["rim_acr"] = {int(i): A.acr.get(int(i), "") for i in np.unique(rlab) if i}
    return d


def load_projection():
    """015 maps (sources included): densities, 90 % mass zones, source masks; P = density / PROJ_PCT percentile"""
    Z = np.load(ZONES)
    Zs = {"zone_w": Z["zone90_whisker"], "zone_a": Z["zone90_auditory"],
          "source_w": Z["source_whisker"], "source_a": Z["source_auditory"]}
    norm99 = {}
    for src in ("density_whisker", "density_auditory"):
        v = Z[src]
        norm99[src] = float(np.percentile(v[v > 0], PROJ_PCT))
    P = {"w": Z["density_whisker"] / norm99["density_whisker"], "a": Z["density_auditory"] / norm99["density_auditory"]}
    return Z, Zs, norm99, P


def frontal_row(u, A, P, Zs, lut):
    """coronal slab AP +0.80 to +2.50 mm from bregma (FR_AP), section drawn at the slab centre (AP +1.65)"""
    c = BREGMA_AP - 1000 * (FR_AP[0] + FR_AP[1]) / 2
    return cor_row(u, A, P, Zs, c, lut, slab=1000 * (FR_AP[1] - FR_AP[0]))


def compute():
    A = m3.Atlas(load_zones=False)
    u = load_units()
    ids = A.unit_ids(u)
    print(f"stable day-0 units {len(u)}, outside the brain {(ids == 0).sum()}", flush=True)
    u = u[ids != 0].reset_index(drop=True)
    lut, acr_group = group_lut(A)
    Z, Zs, norm99, P = load_projection()
    R = {}
    R["frontal"] = frontal_row(u, A, P, Zs, lut)
    # striatum: (a) coronal slab at the striatum-unit centroid, (b) sagittal slab ML 0.8-3.6 mm; STRIATUM_VIEW picks one
    st = u[u.area_group == "Striatum"]
    R["striatum_cor"] = cor_row(u, A, P, Zs, float(st.ccf_atlas_ap.mean()), lut, slab=STR_COR_SLAB)
    lo, hi = MID + STR_ML[0], MID + STR_ML[1]
    R["striatum_sag"] = sag_row(u, A, P, Zs, MID + STR_DRAW_ML, lo, hi, lut)
    sl = A.ann[:, :, int(lo / 10):int(hi / 10)]
    uq = np.unique(sl)
    mg = A.merged(uq)
    str_ids = [int(i) for i, m in zip(uq, mg) if lut.get(int(m)) == "Striatum"]
    smask = np.isin(sl[:, :, ::2], str_ids).any(2).T                  # (DV, AP) 10 um
    R["striatum_sag"]["outlines"].append(("Striatum", smask, "k"))
    R["striatum_sag"]["in_mask"] = smask | (R["striatum_sag"]["lab"] > 0)
    rr, cc = np.where(smask)
    R["striatum_sag"]["crop"] = (cc.min() / 100 - 0.3, cc.max() / 100 + 0.3, rr.max() / 100 + 0.3,
                                 max(rr.min() / 100 - 0.3, 0))
    g = R["striatum_cor"]["group"] == "Striatum"
    rr, cc = np.where(g)
    R["striatum_cor"]["crop"] = (0, cc.max() / 100 + 0.4, rr.max() / 100 + 0.3, max(rr.min() / 100 - 0.4, 0))
    # midbrain: 500-um sagittal slab at the superior-colliculus unit centroid (SCs + SCm, folded ML)
    c_mb = float(u.loc[u.area_acronym_custom.isin(["SCs", "SCm"]), "ml_f"].mean())
    R["midbrain"] = sag_row(u, A, P, Zs, c_mb, c_mb - MB_SLAB / 2, c_mb + MB_SLAB / 2, lut)
    lab = R["midbrain"]["lab"]
    for nm in ("SCs", "SCm", "MRN"):
        sid = int(A.S.loc[A.S.acronym == nm, "id"].iloc[0])
        desc = [i for i, p in A.path.items() if sid in p]
        R["midbrain"]["outlines"].append((nm, np.isin(lab, desc), "k"))
    mb = R["midbrain"]["units"]
    mb = mb[mb.area_group == "Midbrain"]
    ap0, ap1 = np.percentile(mb.ap_mm, 1) - 0.5, np.percentile(mb.ap_mm, 99) + 0.5
    cols_ = slice(int(ap0 * 100), int(ap1 * 100))
    top = np.where((lab[:, cols_] > 0).any(1))[0].min() / 100
    R["midbrain"]["crop"] = (ap0, ap1, np.percentile(mb.dv_mm, 99) + 0.4, max(top - 0.1, 0))
    print("sections done", flush=True)
    R["barrel"] = barrel_row(u, A, Z, Zs, norm99, lut)
    print("barrel done", flush=True)
    R["cortex"] = flat_row(u, P, Zs, acr_group)
    print("flatmap done", flush=True)
    R["dorsal_outline"] = (A.ann[::5, ::5, ::5] > 0).any(1)          # (AP, ML) 50 um
    R["n_units"] = len(u)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE.with_suffix(".tmp"), "wb") as f:
        pickle.dump(R, f)
    CACHE.with_suffix(".tmp").replace(CACHE)
    return R


# ----------------------------------------------------------------------------------------------------- drawing
# Everything that does not depend on colours / norms / layout is precomputed once (prep(), cached in PREP): contour
# lines (contourpy), background and schematic images, label positions. plot() only draws images, lines and dots.
PREP = ROOT / "ssl-paper-figures/Figure2/G_maps_prep.pkl"
PREP_VERSION = 12


def white_to(c):
    return LinearSegmentedColormap.from_list("w", ["#ffffff", c])


def trunc(name, lo, hi):
    return ListedColormap(plt.get_cmap(name)(np.linspace(lo, hi, 256)))


def edges(lbl):
    b = np.zeros(lbl.shape, bool)
    b[1:, :] |= lbl[1:, :] != lbl[:-1, :]
    b[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    return b


def T(d, side, x):
    """row x coordinate (mm) -> plotted x for a side (mirrored rows: whisker left, auditory right)"""
    if d["kind"] == "cor":
        return -x if side == "w" else x
    if d["kind"] == "flat":
        w = d["extent"][1]
        return x - w if side == "w" else w - x
    return x


def show(ax, d, side, rgb, ext):
    a, b = T(d, side, ext[0]), T(d, side, ext[1])
    ax.imshow(rgb[:, ::-1] if b < a else rgb, extent=(min(a, b), max(a, b), ext[2], ext[3]),
              interpolation="nearest", rasterized=True, zorder=1)


def lines_of(mask, ext, step=1, smooth=0.7):
    """contour lines (row coordinates, mm) of a boolean mask on the grid `ext`"""
    import contourpy
    m = mask[::step, ::step]
    if not m.any():
        return []
    h, w = mask.shape
    xs = ext[0] + (np.arange(0, w, step) + 0.5) * (ext[1] - ext[0]) / w
    ys = ext[3] + (np.arange(0, h, step) + 0.5) * (ext[2] - ext[3]) / h
    z = ndimage.gaussian_filter(m.astype(np.float32), smooth)
    return [v for v in contourpy.contour_generator(xs, ys, z).lines(0.5) if len(v) > 2]


def draw_lines(ax, d, side, lines, **kw):
    from matplotlib.collections import LineCollection
    if lines:
        ax.add_collection(LineCollection([np.c_[T(d, side, v[:, 0]), v[:, 1]] for v in lines], **kw))


def group_rgb(group, inside, cols, edge_lbl=None):
    gs = group.astype(str)
    uq, gi = np.unique(gs, return_inverse=True)
    gi = gi.reshape(gs.shape)
    pal = np.ones((len(uq), 3))
    for i, g in enumerate(uq):
        if g in cols:
            pal[i] = to_rgb(ps.light(cols[g], 0.55))
    rgb = pal[gi]
    rgb[edges(gi) & inside] = 0.35
    if edge_lbl is not None:
        e = edges(edge_lbl) & inside
        rgb[e] = np.minimum(rgb[e], 0.7)
    rgb[~inside] = 1.0
    return rgb


ABBR = {"Motor areas": "MO", "Frontal areas": "FRO", "Olfactory areas": "OLF", "Retrosplenial areas": "RSP",
        "Somatosensory-whisker": "Ss-w", "Somatosensory-orofacial": "Ss-o", "Somatosensory-body": "Ss-b",
        "Hippocampus": "Hip", "Midbrain": "MB", "Striatum": "Str", "Pallidum": "Pal", "Thalamus": "TH",
        "Amygdala and hypothalamus": "Amy/Hy", "Pons and medulla": "P/M", "Visual areas": "VIS",
        "Auditory areas": "AUD", "Insular areas": "INS", "Posterior parietal areas": "PPC", "Cortical subplate": "CTXsp",
        "Lateral septal complex": "LS"}


def short(g):
    return ABBR.get(g, RESPELL.get(g, g))


def anchors_of(m, xs, ys):
    """candidate label anchors of a region: in each image row, its most interior pixel -> (n, 2) (x, y), best first"""
    dt = ndimage.distance_transform_edt(m)
    rows = np.flatnonzero(m.any(1))
    if not len(rows):
        return np.zeros((0, 3))
    cols = dt[rows].argmax(1)
    q = dt[rows, cols]
    o = np.argsort(-q)
    return np.c_[xs[cols[o]], ys[rows[o]], q[o]]


def label_positions(d, group, inside, ext, crop, max_n=5, min_frac=0.03, names=None, placed=None, step=1):
    """(x, y, text) for the largest piece of each group (or `names` key) inside the crop (side 'w' coordinates),
    skipping labels that would collide with already placed ones"""
    g_ = group[::step, ::step]
    ins0 = inside[::step, ::step]
    h, w = ins0.shape
    H, Wd = inside.shape
    xs = T(d, "w", ext[0] + (np.arange(0, Wd, step) + 0.5) * (ext[1] - ext[0]) / Wd)
    ys = ext[3] + (np.arange(0, H, step) + 0.5) * (ext[2] - ext[3]) / H
    x0, x1, y1, y0 = crop
    box = ((xs >= min(x0, x1)) & (xs <= max(x0, x1)))[None, :] & ((ys >= y0) & (ys <= y1))[:, None]
    ins = ins0 & box
    tot = max(ins.sum(), 1)
    keys = names if names is not None else {g: short(g) for g in set(g_[ins].tolist()) if isinstance(g, str) and g != "None"}
    uq, code = np.unique(g_.astype(str), return_inverse=True)          # integer codes: no per-key string compares
    code = code.reshape(g_.shape)
    cnt = np.bincount(code[ins], minlength=len(uq))
    idx = {k: i for i, k in enumerate(uq)}
    cand = []
    for key, text in keys.items():
        i = idx.get(str(key))
        if i is not None and cnt[i] >= min_frac * tot:
            cand.append((cnt[i], text, (code == i) & ins))
    cand.sort(key=lambda t: -t[0])
    placed = [(px, py, 2) for px, py in (placed or [])]
    out = []
    ch = 0.045 * abs(x1 - x0)                                        # ~ width of one 5-pt character (data units)
    dy = 0.075 * abs(y1 - y0)
    for _, text, m in cand:
        if len(out) >= max_n:
            break
        lb, n = ndimage.label(m)
        big = 1 + np.argmax(ndimage.sum(m, lb, range(1, n + 1)))
        dt = ndimage.distance_transform_edt(lb == big)
        r, c = np.unravel_index(np.argmax(dt), dt.shape)
        x, y = xs[c], ys[r]
        placed.append((x, y, len(text)))
        out.append((x, y, text, False, anchors_of(lb == big, xs, ys)))
    return out


def prep_row(d, cols, r=None):
    kind = d["kind"]
    p = {}
    ext = d["extent"]
    if kind in ("cor", "sag"):
        lab = d["lab"]
        l2 = lab[::2, ::2]                                           # 20-um display grid
        bg = np.ones(l2.shape + (3,), np.float32)
        bg[edges(l2) & (l2 > 0)] = 0.84
        p["bg"] = bg
        p["outline"] = lines_of(ndimage.binary_fill_holes(lab > 0), ext, step=2)
        ins = d["lab"][2::5, 2::5] > 0
        p["proj_ins"] = ins[:d["projection_w"].shape[0], :d["projection_w"].shape[1]]
        p["schem"] = group_rgb(d["group"][::2, ::2], l2 > 0, cols, l2)
        crop = (T(d, "w", d["crop"][1]), T(d, "w", d["crop"][0]), d["crop"][2], d["crop"][3]) if kind == "cor" \
            else d["crop"]
        if r == "frontal":                                            # frontal / motor structures in the slab
            S_ = pd.read_csv(m3.ATLAS / "structures.csv")
            acr = dict(zip(S_.id, S_.acronym))
            uq, inv = np.unique(lab, return_inverse=True)
            aimg = np.array([acr.get(int(i), "") for i in uq], dtype=object)[inv].reshape(lab.shape)
            p["labels"] = label_positions(d, aimg, lab > 0, ext, crop, max_n=8, min_frac=0.01, step=2,
                                          names={a_: FRONTAL_LABELS.get(a_, a_) for a_ in FRONTAL_LABELS})
        else:
            p["labels"] = [t for t in label_positions(d, d["group"], lab > 0, ext, crop, max_n=12, step=2,
                                                      min_frac=0.005)
                           if t[2] in ROW_LABELS.get(r, {t[2]})]
    elif kind == "flat":
        p["bg"] = None
        p["outline"] = lines_of(d["inside"], ext)
        p["proj_ins"] = d["inside"]
        p["schem"] = group_rgb(d["group"], d["inside"], cols)
        crop = (T(d, "w", d["crop"][1]), T(d, "w", d["crop"][0]), d["crop"][2], d["crop"][3])
        p["labels"] = label_positions(d, d["group"], d["inside"], ext, crop, max_n=8)
    else:
        bg = np.ones(d["bar"].shape + (3,), np.float32)
        bg[edges(d["bar"]) & d["inside"]] = 0.82
        bg[edges(d["rlab"]) & d["dil"] & ~d["inside"]] = 0.88
        p["bg"] = bg
        p["outline"] = lines_of(d["inside"], ext)
        p["dil"] = lines_of(d["dil"], ext)
        p["proj_ins"] = d["dil"]
        p["schem"] = group_rgb(d["group"], d["dil"], cols, d["bar"])
        x0, x1, y1, y0 = d["crop"]
        rows_ = {}
        for nm, (cx, cy) in d["centroids"].items():
            if nm[0] in "ABCDE" and nm[1:].isdigit():
                rows_.setdefault(nm[0], []).append((cx, cy))
        xl = min(c[0] for v in rows_.values() for c in v) - 0.22          # rows run medio-laterally (top-down)
        lab = [(xl, float(np.mean([c[1] for c in v])), rk, False, None) for rk, v in sorted(rows_.items()) if rk in "ACE"]
        rl = np.zeros(d["rlab"].shape, object)
        for i, acr in d["rim_acr"].items():
            rl[d["rlab"] == i] = acr
        names = {a_: a_ for a_ in set(d["rim_acr"].values()) if a_ and not a_.startswith("SSp-bfd")}
        xs_b = (np.arange(d["inside"].shape[1]) + 0.5) * (ext[1] - ext[0]) / d["inside"].shape[1]
        ys_b = (np.arange(d["inside"].shape[0]) + 0.5) * (ext[2] - ext[3]) / d["inside"].shape[0]
        a_b = anchors_of(d["inside"], xs_b, ys_b)
        lab.append((a_b[0, 0], a_b[0, 1], "SSp-bfd", False, a_b))
        lab += label_positions(d, rl, d["dil"] & ~d["inside"], ext, d["crop"], max_n=8, min_frac=0.02, names=names,
                               placed=[(x_, y_) for x_, y_, _, _, _ in lab])
        p["labels"] = lab
    p["extra"] = [(nm, lines_of(m, ext, step=2 if kind in ("cor", "sag") else 1)) for nm, m, _ in d.get("outlines", [])]
    if kind == "sag":
        for nm, m, _ in d.get("outlines", []):
            if nm in ("SCs", "SCm", "MRN"):
                m2 = m[::2, ::2]
                if m2.any():
                    xs2 = (np.arange(0, m.shape[1], 2) + 0.5) / 100
                    ys2 = (np.arange(0, m.shape[0], 2) + 0.5) / 100
                    a_ = anchors_of(m2, xs2, ys2)
                    p["labels"].append((a_[0, 0], a_[0, 1], nm, True, a_))
    for k in MODS:
        for q in ("zone", "source"):
            p[f"{q}_{k}"] = lines_of(d[f"{q}_{k}"] & p["proj_ins"], d["map_extent"])
    if kind == "flat":
        p["B"] = list(d["B"].values())
    return p


def prep(R):
    if PREP.exists():
        with open(PREP, "rb") as f:
            P = pickle.load(f)
        if P.get("version") == PREP_VERSION and P.get("cache_mtime") == CACHE.stat().st_mtime and                 P.get("str_view") == STRIATUM_VIEW:
            return P
    cols = au.get_custom_area_groups_colors()
    P = {"version": PREP_VERSION, "cache_mtime": CACHE.stat().st_mtime, "str_view": STRIATUM_VIEW}
    for r, _ in ROWS:
        P[r] = prep_row(R[r], cols, r)
        print("prep", r, flush=True)
    with open(PREP.with_suffix(".tmp"), "wb") as f:
        pickle.dump(P, f)
    PREP.with_suffix(".tmp").replace(PREP)
    return P


def background(ax, d, p, side):
    if d["kind"] == "flat":
        draw_lines(ax, d, side, p["B"], colors="0.82", linewidths=0.2, zorder=1)
    else:
        ext = d["extent"]
        show(ax, d, side, p["bg"], ext)
    lines_over(ax, d, p, side)


def lines_over(ax, d, p, side):
    draw_lines(ax, d, side, p["outline"], colors="0.15" if d["kind"] == "barrel" else "0.3",
               linewidths=0.5 if d["kind"] == "barrel" else 0.4, zorder=2)
    if d["kind"] == "barrel":
        draw_lines(ax, d, side, p["dil"], colors="0.6", linewidths=0.3, linestyles=":", zorder=2)
    for nm, ln in p["extra"]:
        draw_lines(ax, d, side, ln, colors="k", linewidths=0.45, zorder=6)


DOT_S = {"coverage": 0.10, "bg": 0.028, "data": 0.18, "open": 0.6}     # pt^2 (round 8: ~0.7 x the round-7 sizes 0.14 / 0.04 / 0.22-0.3)
DOT_ALPHA = 0.6                                            # recorded-neuron dots (was 0.35); coloured + grey dots opaque


def dots(ax, d, side, col, norms, cmaps):
    un = d["units"]
    x, y = T(d, side, un.row_x_mm.to_numpy()), un.row_y_mm.to_numpy()
    kw = dict(lw=0, rasterized=True)
    if col == "coverage":
        ax.scatter(x, y, s=DOT_S["coverage"], c="0.2", alpha=DOT_ALPHA, zorder=3, **kw)
        return
    ax.scatter(x, y, s=DOT_S["bg"], c="0.72" if col == "frac" else "0.88", zorder=3, **kw)   # background: opaque, light
    if col == "sel":
        v = un[f"sel_{side}"].to_numpy(float)
        m = np.isfinite(v)
        o = np.argsort(np.abs(v[m]))
        ax.scatter(x[m][o], y[m][o], s=DOT_S["data"], c=v[m][o], cmap=cmaps["sel"][side], norm=norms["sel"], zorder=4, **kw)
    elif col == "frac":
        sig = (un[f"sig_{side}"] == 1).to_numpy()
        v = np.abs(un[f"sel_{side}"].to_numpy(float))[sig]
        o = np.argsort(v)
        ax.scatter(x[sig][o], y[sig][o], s=DOT_S["data"], c=v[o], cmap=cmaps["frac"][side], norm=norms["frac"],
                   zorder=4, **kw)
    elif col == "lat":
        v = un[f"lat_{side}"].to_numpy(float)
        m = np.isfinite(v)
        if ONSET_ALL:                                                 # decreases: open circles, under the filled
            dec = m & (un[f"dir_{side}"] == "negative").to_numpy()
            od = np.argsort(-v[dec])
            ax.scatter(x[dec][od], y[dec][od], s=DOT_S["open"], facecolors="none",
                       edgecolors=cmaps["lat"](norms["lat"](v[dec][od])), linewidths=0.2, zorder=4, rasterized=True)
            m = m & ~dec
        o = np.argsort(-v[m])                                         # fastest drawn last (on top)
        ax.scatter(x[m][o], y[m][o], s=DOT_S["data"], c=v[m][o], cmap=cmaps["lat"], norm=norms["lat"], zorder=4, **kw)


def projection(ax, d, p, side, norms, cmaps):
    M = d[f"projection_{side}"]
    rgb = cmaps["projection"][side](norms["projection"](np.nan_to_num(M, nan=0.0)))[..., :3]
    rgb[~p["proj_ins"]] = 1.0
    show(ax, d, side, rgb, d["map_extent"])
    if d["kind"] == "flat":
        draw_lines(ax, d, side, p["B"], colors="0.75", linewidths=0.2, zorder=2)
    lines_over(ax, d, p, side)
    draw_lines(ax, d, side, p[f"source_{side}"], colors="0.45", linewidths=0.4, linestyles=":", zorder=5)
    draw_lines(ax, d, side, p[f"zone_{side}"], colors="0.1", linewidths=0.5, linestyles=[(0, (2.0, 1.2))], zorder=5)


def in_view(d, un):
    """True for units whose 2-D position lies inside the brain contour of the shown section / view"""
    x, y = un.row_x_mm.to_numpy(), un.row_y_mm.to_numpy()
    if d["kind"] in ("cor", "sag"):
        m, res = d.get("in_mask", d["lab"] > 0), 0.01
    elif d["kind"] == "flat":
        m, res = d["inside"], 0.01
    else:
        m, res = d["dil"], (d["extent"][1] - d["extent"][0]) / d["dil"].shape[1]
    i, j = np.floor(y / res).astype(int), np.floor(x / res).astype(int)
    ok = (i >= 0) & (i < m.shape[0]) & (j >= 0) & (j < m.shape[1])
    out = np.zeros(len(un), bool)
    out[ok] = m[i[ok], j[ok]]
    return out


def leader_labels(ax, labels, first, right=False, Hmm=110.0, gap_mm=0.8, row_mm=1.9):
    """labels beside the panel joined to their region by horizontal leader lines (as SuppFig 2 label_rows): each label
    takes the candidate anchor row closest to its best anchor that is >= row_mm from the rows already used; single
    letters (barrel rows) are drawn in place"""
    x0, x1 = ax.get_xlim()
    yb, yt = ax.get_ylim()                                           # inverted: yb > yt
    pos = ax.get_position()
    h_mm = pos.height * Hmm
    w_mm = pos.width * (183.0)
    dy = row_mm * (yb - yt) / h_mm
    gx = gap_mm * (x1 - x0) / w_mm
    xe = x1 + gx if right else x0 - gx
    used = []
    order = sorted(labels, key=lambda t: (len(t[2]) == 1, not t[3], t[2] != first))
    for lx, ly, txt, bold, cand in order:
        if len(txt) == 1 or cand is None:
            ax.text(lx, ly, txt, fontsize=5, ha="center", va="center", zorder=9)
            continue
        c = cand[(cand[:, 1] > yt + 0.3 * dy) & (cand[:, 1] < yb - 0.3 * dy)]
        if not len(c):
            continue
        best = c[0]
        ok = [k for k in np.argsort(np.abs(c[:, 1] - best[1])) if all(abs(c[k, 1] - u) >= dy for u in used)]
        if not ok:
            continue
        ax_, ay = c[ok[0], 0], c[ok[0], 1]
        used.append(ay)
        ax.plot([xe, ax_], [ay, ay], color="k", lw=0.3, solid_capstyle="butt", clip_on=False, zorder=9)
        ax.plot(ax_, ay, "o", ms=1.0, color="k", mec="none", clip_on=False, zorder=9)
        ax.text(xe + (0.3 if right else -0.3) * (x1 - x0) / w_mm, ay, txt, fontsize=5, ha="left" if right else "right",
                va="center", clip_on=False, zorder=9)


JITTER_SD_MM, JITTER_CLIP_MM, JITTER_SEED = 0.010, 0.025, 0


ONSETS = ROOT / "ssl-paper-figures/Figure2/G_latenzy_onsets_v2.parquet"   # + ZETA response direction
ONSET_ALL = False                                       # --onset-all: onset column = every neuron with a latenZy onset


def load_onsets():
    """latenZy onsets (ms) of whisker_active / auditory_active from the per-mouse ZETA tables (cached)"""
    if ONSETS.exists():
        return pd.read_parquet(ONSETS)
    rows = []
    for f in sorted(ROOT.glob("*/whisker_0/zeta/*_zeta_results.csv")):
        z = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "analysis_type", "latency_s",
                                    "latenzy_p", "direction"])
        rows.append(z[z.analysis_type.isin(["whisker_active", "auditory_active"])])
    z = pd.concat(rows, ignore_index=True)
    z["cluster_id"] = z.cluster_id.astype(str)
    ok = (z.latenzy_p < 0.05) & z.latency_s.notna() & (z.latency_s >= 0)
    z["onset_ms"] = np.where(ok, 1000 * z.latency_s, np.nan)
    o = z.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type", values="onset_ms",
                      aggfunc="first").reset_index()
    o = o.rename(columns={"whisker_active": "onset_w", "auditory_active": "onset_a"})
    dr = z.pivot_table(index=["session_id", "electrode_group", "cluster_id"], columns="analysis_type",
                       values="direction", aggfunc="first").reset_index()
    dr = dr.rename(columns={"whisker_active": "dir_w", "auditory_active": "dir_a"})
    o = o.merge(dr, on=["session_id", "electrode_group", "cluster_id"], how="left")
    o.to_parquet(ONSETS)
    return o


def row_units(R, r):
    un = R[r]["units"]
    on = load_onsets()
    un = un.drop(columns=[c for c in ("onset_w", "onset_a", "dir_w", "dir_a") if c in un]).merge(
        on, on=["session_id", "electrode_group", "cluster_id"], how="left")
    for k in MODS:                                                    # latency column = latenZy onset
        resp = (un[f"sig_{k}"] == 1) & (un[f"sel_{k}"] > 0)
        un[f"lat_{k}"] = un[f"onset_{k}"] if ONSET_ALL else np.where(resp, un[f"onset_{k}"], np.nan)
    rng = np.random.default_rng([JITTER_SEED, ROWS_ORDER.index(r)])
    j = np.clip(rng.normal(0, JITTER_SD_MM, (len(un), 2)), -JITTER_CLIP_MM, JITTER_CLIP_MM)   # Gaussian SD 10 um, |j| <= 25 um
    un = un.assign(row_x_mm=un.row_x_mm + j[:, 0], row_y_mm=un.row_y_mm + j[:, 1])
    if r in ROW_GROUP:
        un = un[un.area_group == ROW_GROUP[r]]
    if r == "barrel":                                                 # cortical units only (isocortex area groups)
        un = un[un.area_group.isin(CORTICAL_GROUPS)]
    return un[in_view(R[r], un)]


def plot(R, quick=False):
    cols = au.get_custom_area_groups_colors()
    R["striatum"] = R[f"striatum_{STRIATUM_VIEW}"]
    for r, _ in ROWS:                                                 # row's area group only, inside the view
        R[r] = dict(R[r], units=row_units(R, r))
    PP = prep(R)
    ps.set_nature_rc()
    allu = pd.concat([R[r]["units"] for r, _ in ROWS])
    sel_hi = float(np.nanpercentile(np.abs(np.r_[allu.sel_w, allu.sel_a]), 99))
    sel_hi = np.ceil(sel_hi * 10) / 10
    sg = np.abs(np.r_[allu.sel_w[allu.sig_w == 1], allu.sel_a[allu.sig_a == 1]])
    frac_hi = np.ceil(float(np.nanpercentile(sg, 95)) * 10) / 10
    norms = {"projection": PowerNorm(0.5, 0, 1, clip=True), "sel": Normalize(-sel_hi, sel_hi, clip=True),
             "frac": Normalize(0, frac_hi, clip=True), "lat": LogNorm(5, 100, clip=True)}
    ramp = lambda c, lo, hi: LinearSegmentedColormap.from_list("r", [lo, c, hi])  # noqa: E731
    cmaps = {"projection": {"w": ramp(C_W, "#ffffff", "#241800"), "a": ramp(C_A, "#ffffff", "#03031f")},
             "sel": {"w": plt.get_cmap("bwr"), "a": plt.get_cmap("bwr")},
             "frac": {"w": ramp(C_W, "#fdeec4", "#3d2900"), "a": ramp(C_A, "#d4d4f7", "#05053a")},
             "lat": trunc("inferno_r", 0.08, 0.88)}
    Wmm, left, inset_w, gap, top, cb_h, rgap = 183.0, 1.0, 14.0, 1.2, 7.5, 9.5, 2.0
    LBL, BOXW = 7.0, 17.0                                             # schematic column: label space + schematic box
    CS = LBL + BOXW
    cw = (Wmm - left - inset_w - 1.0 - CS - 5 * gap) / 5             # data columns
    fr = R["frontal"]["lab"] > 0                                      # frontal: crop the coronal section to the brain
    rr_, cc_ = np.where(fr)
    R["frontal"] = dict(R["frontal"], crop=(0, cc_.max() / 100 + 0.15, rr_.max() / 100 + 0.15,
                                            max(rr_.min() / 100 - 0.15, 0)))

    def dims(r):
        d = R[r]
        x0, x1, y1, y0 = d["crop"]
        ww, hh = abs(x1 - x0), y1 - y0
        return (2 * ww, hh) if d["kind"] in ("cor", "flat") else (2 * ww + 0.6 * ww / 6.5, hh)
    heights = []
    for r, _ in ROWS:
        ww, hh = dims(r)
        heights.append(float(np.clip(cw * hh / ww, 12.0, 22.0)))
    Hmm = top + sum(heights) + rgap * (len(ROWS) - 1) + cb_h
    fig = plt.figure(figsize=(Wmm * ps.MM, Hmm * ps.MM))

    def ax_mm(x, y, w, h):
        return fig.add_axes([x / Wmm, 1 - (y + h) / Hmm, w / Wmm, h / Hmm])

    x_s = left + inset_w + 1.0
    xcol = [x_s] + [x_s + CS + gap + j * (cw + gap) for j in range(5)]
    colw = [CS] + [cw] * 5
    for j, c in enumerate(COLS):
        fig.text((xcol[j] + colw[j] / 2) / Wmm, 1 - 1.6 / Hmm, COL_TITLES[c], ha="center", va="top", fontsize=6)
        if c not in ("schematic", "coverage"):
            for k, xx, ha in (("w", xcol[j] + 0.5, "left"), ("a", xcol[j] + cw - 0.5, "right")):
                fig.text(xx / Wmm, 1 - 4.3 / Hmm, MODS[k].capitalize(), ha=ha, va="top", fontsize=5,
                         color=C_W if k == "w" else C_A)
    y = top
    IW, IH = inset_w - 0.6, min(heights) - 6.2                      # one inset size for every row (fits the shortest)
    SEC = ("striatum", "frontal", "midbrain")                         # section rows: schematics at one common scale
    BOXH = min(heights)                                               # one schematic box for every row
    s_sec = min(min(BOXW / abs(R[r_]["crop"][1] - R[r_]["crop"][0]), BOXH / (R[r_]["crop"][2] - R[r_]["crop"][3]))
                for r_ in SEC)
    dor = R["dorsal_outline"]
    summary = []
    for (r, title), h in zip(ROWS, heights):
        d = R[r]
        pp = PP[r]
        kind = d["kind"]
        un = d["units"]
        x0, x1, y1, y0 = d["crop"]
        fig.text((left + inset_w / 2) / Wmm, 1 - (y + 0.2) / Hmm, title, ha="center", va="top", fontsize=6)
        fig.text((left + inset_w / 2) / Wmm, 1 - (y + 2.5) / Hmm, f"n = {len(un):,}", ha="center", va="top", fontsize=5,
                 color="0.3")
        iw, ih = IW, IH
        ia = ax_mm(left + 0.1, y + 4.4, iw, ih)
        ia.contour(np.arange(dor.shape[1]) * 0.05, np.arange(dor.shape[0]) * 0.05, dor.astype(float), levels=[0.5],
                   colors="0.45", linewidths=0.35)
        pg = min(iw / (dor.shape[1] * 0.05), ih / (dor.shape[0] * 0.05 + 0.2))   # page mm per brain mm
        half_min = 0.4 / pg                                           # drawn band >= 0.8 mm on the page

        def band(lo_um, hi_um, vertical):
            cen, half = (lo_um + hi_um) / 2000, max((hi_um - lo_um) / 2000, half_min)
            (ia.axvspan if vertical else ia.axhspan)(cen - half, cen + half, color="0.75", lw=0, zorder=0)
            for v_ in (cen - half, cen + half):
                (ia.axvline if vertical else ia.axhline)(v_, color="k", lw=0.4)
        if kind == "cor":
            sl_ = d.get("slab", SLAB)
            band(d["c"] - sl_ / 2, d["c"] + sl_ / 2, False)
            coord = (f"AP {(BREGMA_AP - d['c'] - sl_ / 2) / 1000:+.2f}–"
                     f"{(BREGMA_AP - d['c'] + sl_ / 2) / 1000:.2f} mm")
        elif kind == "sag":
            lo, hi = d["ml"]
            band(lo, hi, True)
            coord = f"ML {(lo - MID) / 1000:.2f}–{(hi - MID) / 1000:.2f} mm"
        elif kind == "barrel":
            pu = un[un.area_acronym_custom == "SSp-bfd"]
            xs_, ys_ = pu.ml_f / 1000, pu.ccf_atlas_ap / 1000
            ia.add_patch(plt.Rectangle((np.percentile(xs_, 2), np.percentile(ys_, 2)),
                                       np.percentile(xs_, 98) - np.percentile(xs_, 2),
                                       np.percentile(ys_, 98) - np.percentile(ys_, 2), fill=False, lw=0.6, ec="k"))
            coord = "Top-down CCF"
        else:
            coord = "Isocortex flatmap"
        ia.set_aspect("equal"); ia.set_ylim(dor.shape[0] * 0.05, -0.2); ia.axis("off")
        ia.text(0.5, -0.02, coord, transform=ia.transAxes, ha="center", va="top", fontsize=5)
        for j, c in enumerate(COLS):
            mirrored = kind in ("cor", "flat") and c not in ("schematic", "coverage")   # coverage: left half only
            if c == "schematic" or c == "coverage" or mirrored:
                sides = ["w"] if c == "schematic" else (["w", "a"] if mirrored else ["w"])
                ww = 2 * abs(x1 - x0) if mirrored else abs(x1 - x0)
                hh = y1 - y0
                if c == "schematic":                                  # equal boxes; one scale for sections
                    s = s_sec if r in SEC else min(BOXW / ww, BOXH / hh)
                    box_l = xcol[0] + LBL
                    ax = ax_mm(box_l + (BOXW - ww * s) / 2, y + (h - hh * s) / 2, ww * s, hh * s)
                else:
                    s = min(cw / ww, h / hh)
                    ax = ax_mm(xcol[j] + (cw - ww * s) / 2, y + (h - hh * s) / 2, ww * s, hh * s)
                axes = [(ax, sides)]
                if mirrored:
                    ax.set_xlim(-abs(x1 - x0), abs(x1 - x0))
                elif kind in ("cor", "flat"):                         # left hemisphere only
                    ax.set_xlim(-abs(x1 - x0), 0)
                else:
                    ax.set_xlim(x0, x1)
            else:                                                     # sagittal / barrel: whisker | auditory
                ww, hh = abs(x1 - x0), y1 - y0
                sub = (cw - 0.6) / 2
                s = min(sub / ww, h / hh)
                axes = []
                for si, k in enumerate(("w", "a")):
                    ax = ax_mm(xcol[j] + si * (sub + 0.6) + (sub - ww * s) / 2, y + (h - hh * s) / 2, ww * s, hh * s)
                    ax.set_xlim(x0, x1)
                    axes.append((ax, [k]))
            for ax, sides in axes:
                ax.set_ylim(y1, y0)
                ax.axis("off")
                for k in sides:
                    if c == "schematic":
                        show(ax, d, "w", pp["schem"], d["extent"])
                        lines_over(ax, d, pp, "w")
                        if kind == "flat":
                            draw_lines(ax, d, "w", pp["B"], colors="0.45", linewidths=0.2, zorder=2)
                        if kind != "barrel":                          # barrel schematic: no annotations
                            gap_l = ax.get_position().x0 * Wmm - (xcol[0] + LBL) + 0.8   # labels aligned at the box
                            leader_labels(ax, pp["labels"], LABEL_FIRST.get(r), Hmm=Hmm, gap_mm=gap_l)
                        xa, xb = ax.get_xlim()
                        ph = ax.get_position().height * Hmm               # bar 1.2 mm below, centred
                        xl, yb = (xa + xb) / 2 - 0.5, y1 + 1.2 * (y1 - y0) / ph
                        ax.plot([xl, xl + 1.0], [yb, yb], color="k", lw=0.8, solid_capstyle="butt", clip_on=False,
                                zorder=9)
                        ax.text(xl + 1.1, yb, "1 mm" + (" (flatmap)" if kind == "flat" else ""), fontsize=5,
                                ha="left", va="center", clip_on=False)
                    elif c == "projection":
                        projection(ax, d, pp, k, norms, cmaps)
                    else:
                        background(ax, d, pp, k)
                        dots(ax, d, k, c, norms, cmaps)
                    if c in ("sel", "frac", "lat"):
                        for k2 in sides:
                            col_ = {"sel": f"sel_{k2}", "frac": f"sig_{k2}", "lat": f"lat_{k2}"}[c]
                            v = un[col_]
                            summary.append(dict(row=r, column=c, modality=MODS[k2], n_units=len(un),
                                                n_with_value=int(v.notna().sum()),
                                                n_significant=int((un[f"sig_{k2}"] == 1).sum()),
                                                median=float(v.median()) if v.notna().any() else np.nan))
                if mirrored:
                    ax.axvline(0, color="0.6", lw=0.3, ls=":", zorder=7)
        y += h + rgap
    # colour bars
    yb = Hmm - cb_h + 3.4
    for j, c in enumerate(COLS):
        if c in ("schematic",):
            continue
        if c == "coverage":
            fig.text((xcol[j] + cw / 2) / Wmm, 1 - (yb + 0.3) / Hmm, "Dots: stable day-0 neurons; positions\njittered (Gaussian, SD 10 \u00b5m)", ha="center",
                     va="top", fontsize=5)
            continue
        specs = [(cmaps["lat"], "")] if c == "lat" else [(cmaps["sel"]["w"], "")] if c == "sel" else \
            [(cmaps[c]["w"], "W"), (cmaps[c]["a"], "A")]
        for si, (cm, tag) in enumerate(specs):
            cax = ax_mm(xcol[j] + 3, yb + si * 1.4, cw - 6, 1.1)
            cb = fig.colorbar(plt.cm.ScalarMappable(norm=norms[c], cmap=cm), cax=cax, orientation="horizontal")
            cb.outline.set_linewidth(0.3)
            cb.ax.tick_params(labelsize=5, length=1.2, width=0.3, pad=0.5)
            if c in ("projection", "frac"):
                cb.set_ticks([0, 0.5, 1.0] if c == "projection" else [0, 0.4, 0.8])
            if c == "lat":
                cb.set_ticks([5, 10, 20, 50, 100]); cb.set_ticklabels(["5", "10", "20", "50", "100"]); cb.ax.minorticks_off()
            if si < len(specs) - 1:
                cb.ax.set_xticklabels([])
                cb.ax.tick_params(length=0)
            if tag:
                cax.text(-0.02, 0.5, tag, transform=cax.transAxes, ha="right", va="center", fontsize=5,
                         color=C_W if tag == "W" else C_A)
        lab = {"projection": f"Density (/{PROJ_PCT:g}th pct); - - 90 % zone", "sel": "ROC selectivity",
               "frac": "|Selectivity| (significant)",
               "lat": "Onset (ms); \u25cb decrease" if ONSET_ALL else "Onset (ms)"}[c]
        fig.text((xcol[j] + cw / 2) / Wmm, 1 - (yb - 0.3) / Hmm, lab, ha="center", va="bottom", fontsize=5)
    OUT.mkdir(parents=True, exist_ok=True)
    if quick:                                                         # iteration: PNG only
        fig.savefig(OUT / f"{NAME}_quick.png", dpi=600)
        plt.close(fig)
        return
    ps.save_figure(fig, OUT, NAME)
    plt.close(fig)
    if ONSET_ALL:
        rows_ = []
        for r, _ in ROWS:
            un = R[r]["units"]
            for k in MODS:
                on = un[f"onset_{k}"].notna()
                resp = (un[f"sig_{k}"] == 1) & (un[f"sel_{k}"] > 0)
                rows_.append(dict(row=r, modality=MODS[k], n_units=len(un), n_onset=int(on.sum()),
                                  n_onset_roc_excited=int((on & resp).sum()), n_onset_not_roc_excited=int((on & ~resp).sum()),
                                  n_onset_decrease=int((on & (un[f"dir_{k}"] == "negative")).sum()),
                                  n_onset_direction_unknown=int((on & un[f"dir_{k}"].isna()).sum())))
        T_ = pd.DataFrame(rows_)
        T_.to_csv(OUT / f"{NAME}_counts.csv", index=False)
        print(T_.to_string(index=False))
        return
    pd.DataFrame(summary).drop_duplicates().to_csv(OUT / "G_maps_summary.csv", index=False)
    keep = KEYS + ["area_group", "area_acronym_custom", "ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv", "sel_w", "sig_w",
                   "sel_a", "sig_a", "lat_w", "lat_a", "row_x_mm", "row_y_mm"]
    pd.concat([R[r]["units"][keep].assign(row=r) for r, _ in ROWS]).rename(
        columns={"sel_w": "sel_whisker_active", "sig_w": "sig_whisker_active", "sel_a": "sel_auditory_active",
                 "sig_a": "sig_auditory_active", "lat_w": "onset_latenzy_whisker_ms", "lat_a": "onset_latenzy_auditory_ms"}).to_csv(
        OUT / "G_maps_units.csv", index=False)
    print(f"figure {Wmm:.0f} x {Hmm:.1f} mm; heights {np.round(heights, 1)}")
    print("norms: sel", sel_hi, "frac", frac_hi, "onset 5-100 ms log")
    for r, _ in ROWS:
        d = R[r]
        un = d["units"]
        extra = ""
        if d["kind"] == "cor":
            extra = f"AP {(BREGMA_AP - d['c']) / 1000:+.2f} mm"
        elif d["kind"] == "sag":
            extra = f"ML {(d['ml'][0] - MID) / 1000:.2f}-{(d['ml'][1] - MID) / 1000:.2f} mm"
        elif d["kind"] == "barrel":
            extra = f"in contour {int(un.in_bfd_contour.sum())}, rim {int((~un.in_bfd_contour).sum())}"
        print(r, extra, "units", len(un), "sig w", int((un.sig_w == 1).sum()), "sig a", int((un.sig_a == 1).sum()),
              "lat w/a", int(un.lat_w.notna().sum()), int(un.lat_a.notna().sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    ap.add_argument("--onset-all", action="store_true",
                    help="variant G_response_maps_onset_all: onset column = every neuron with a latenZy onset")
    ap.add_argument("--png-only", dest="quick", action="store_true",
                    help="iteration: PNG only (G_response_maps_quick.png); all formats only for the final render")
    a = ap.parse_args()
    if a.onset_all:
        ONSET_ALL, NAME = True, "G_response_maps_onset_all"
        COL_TITLES["lat"] = "Response onset (all)"
    if a.plot_only:
        with open(CACHE, "rb") as f:
            R = pickle.load(f)
    else:
        R = compute()
    plot(R, a.quick)
    print("saved", OUT / "G_response_maps")
