"""Supp Fig 2: schematic of the area grouping (area_group) on the Allen CCF, several sections and views, with the areas
annotated (user requests 2026-11-18).

Every Allen structure in the 25-um annotation volume is passed through the pipeline's own labelling
(ephys_utilities.allen_utils: excluded areas -> create_area_custom_column with the structure-tree parent as
ccf_atlas_parent_acronym, no target region) and mapped to its area group with get_custom_area_groups_from_name().
Fill = area group colour (get_custom_area_groups_colors(), skill ssl-figure-style); dark lines = group borders, mid-grey
lines = area_acronym_custom borders (the labelled areas), light lines = finer Allen structures. Labels = the
area_acronym_custom names (the area level used in the analyses), black, beside each panel, joined to their area by
horizontal (parallel) leader lines; areas without a free label row are not labelled. White = not in any group (cerebellum, olfactory bulb, entorhinal
cortex...); fibre tracts and ventricles are white too.
Target-dependent rules (MOs/MOp split by target, PPC-probe relabelling of SSp-bfd / SSp-tr / SSp-un, SSp-w on wS1/wS2
probes) cannot be drawn on an atlas and are not applied. Default labels as of ephys_utilities 7ae712d (2026-11-18):
SSp-un and SSp-tr -> Somatosensory-body (SSp-un was orofacial before).
Coordinates relative to bregma (IBL estimate, as allen_utils.create_bregma_centric_coords_from_ccf: CCF ap 5400,
ml 5739, dv 332 um). Sections show the right hemisphere (midline on the left), except the sagittal and dorsal views.

Figures (manuscript Figures/SuppFigure2_recording_<date>/_explore/area_groups/):
  area_groups_two_views (sagittal ML 0.9 + lateral surface: main schematic); _tracks: + every insertion's atlas
  track projected on the sagittal panel
  area_groups_all_views (surface dorsal + lateral, sagittal, coronal, horizontal, oblique; one scale)
  + area_groups_mapping.csv (acronym -> area_acronym_custom -> area_group, voxel counts)
  --per-view also writes area_groups_sagittal / _coronal / _horizontal / _surface / _oblique with more sections.
Run (haas): PYTHONPATH=<NAS ephys_utilities until 7ae712d is pushed> \
  ~/code/unit_spikes_analysis/.venv/bin/python area_group_schematic.py [--per-view]
"""

import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import paperstyle as ps  # noqa: E402
from ephys_utilities.allen_utils import allen_utils as au  # noqa: E402

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi")
OUT = ROOT / "manuscripts/Bisi_Hamon/Figures" / f"SuppFigure2_recording_{datetime.date.today():%Y-%m-%d}" / "_explore" / "area_groups"
RES = 25.0  # um per voxel
BREGMA = dict(ap=5400.0, ml=5739.0, dv=332.0)  # CCF um (allen_utils)
W = 183.0  # mm
C_FINE, C_AREA, C_GROUP, C_OUTLINE = (0.86, 0.86, 0.86), (0.6, 0.6, 0.6), (0.3, 0.3, 0.3), (0.15, 0.15, 0.15)
RESPELL = {"Somatosensory-orofacial": "Ss-orofacial", "Somatosensory-body": "Ss-body",
           "Somatosensory-whisker": "Ss-whisker"}
LABEL_PT = 5.0
HIDE_TWO_VIEWS = ["Pons and medulla", "Amygdala and hypothalamus"]  # not shown in the two-view schematic
TRACK_UNITS = ROOT / "combined_results_ks4/ssl-ssp-un-review/unit_labels_all_days_v2.parquet"
TRACK_SUB = "Anatomy/fused/registered/segmentation/atlas_space/tracks"
MIN_LABEL_PX = 120  # smallest area piece labelled (voxels in the plane, 25 x 25 um)


# ----------------------------------------------------------------------------------------------------- labelling
def structure_table(atlas):
    rows = []
    for sid, s in atlas.structures.items():
        path = s["structure_id_path"]
        parent = atlas.structures[path[-2]]["acronym"] if len(path) > 1 else s["acronym"]
        rows.append(dict(id=sid, ccf_atlas_acronym=s["acronym"], ccf_atlas_parent_acronym=parent, target_region=None))
    df = pd.DataFrame(rows)
    keep = ~df.ccf_atlas_acronym.isin(au.get_excluded_areas())
    lab = au.create_area_custom_column(df[keep].copy())
    df["area_acronym_custom"] = lab["area_acronym_custom"].reindex(df.index)
    df["area_group"] = df.area_acronym_custom.map(au.get_custom_area_groups_from_name())
    df["excluded"] = ~keep
    return df


class Lut:
    """Structure id -> (group index, area index); 0 = none."""

    def __init__(self, df, order):
        gidx = {g: i + 1 for i, g in enumerate(order)}
        areas = sorted(df.loc[df.area_group.notna(), "area_acronym_custom"].unique())
        self.areas = [""] + areas
        aidx = {a: i + 1 for i, a in enumerate(areas)}
        d = df.sort_values("id")
        self.ids = d.id.to_numpy()
        self.g = d.area_group.map(gidx).fillna(0).astype(int).to_numpy()
        self.a = np.where(d.area_group.notna(), d.area_acronym_custom.map(aidx), 0).astype(int)

    def __call__(self, ids2d):
        i = np.clip(np.searchsorted(self.ids, ids2d), 0, len(self.ids) - 1)
        ok = (ids2d > 0) & (self.ids[i] == ids2d)
        g = np.where(ok, self.g[i], 0)
        a = np.where(ok, self.a[i], 0)
        return g, a


def edges(lbl):
    b = np.zeros(lbl.shape, bool)
    b[1:, :] |= lbl[1:, :] != lbl[:-1, :]
    b[:, 1:] |= lbl[:, 1:] != lbl[:, :-1]
    return b


# ----------------------------------------------------------------------------------------------------- drawing
MARG = 9.0  # mm of label space left and right of every panel
LINE_MM = 2.0  # vertical spacing of stacked labels (mm)


def anchors(a, names, label_mask=None, min_px=MIN_LABEL_PX):
    """(size, name, col, row) at the most interior point of each label's largest piece."""
    cand = []
    for k in np.unique(a[a > 0]):
        m = a == k
        if label_mask is not None:
            m &= label_mask
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        big = int(np.argmax(sizes)) + 1
        if sizes[big - 1] < min_px:
            continue
        dist = ndimage.distance_transform_edt(lab == big)
        r, c = np.unravel_index(np.argmax(dist), dist.shape)
        cand.append((sizes[big - 1], names[k], c, r))
    return cand


def _stack(ys, lo, hi, d):
    y = list(ys)
    for i in range(len(y)):
        y[i] = max(y[i], lo if i == 0 else y[i - 1] + d)
    if y and y[-1] > hi:
        y[-1] = hi
        for i in range(len(y) - 2, -1, -1):
            y[i] = min(y[i], y[i + 1] - d)
    return y


def label_outside(ax, cand, w, h, s, fs=LABEL_PT, only=None):
    """Black labels in columns left and right of a w x h (voxel) image, leader lines to the anchors.
    s = mm per voxel. Labels that do not fit (smallest areas) are dropped."""
    d, gap = LINE_MM / s, 1.2 / s
    nmax = int((h - 1) // d) + 1
    side = {"L": [x for x in cand if x[2] < w / 2], "R": [x for x in cand if x[2] >= w / 2]}
    if only:  # all labels on one side
        side = {only: list(cand), ("R" if only == "L" else "L"): []}
    for a_, b_ in (() if only else (("L", "R"), ("R", "L"))):  # move overflow to the other side, nearest-the-centre first
        while len(side[a_]) > nmax and len(side[b_]) < nmax:
            x = min(side[a_], key=lambda t: abs(t[2] - w / 2))
            side[a_].remove(x)
            side[b_].append(x)
    for k, items in side.items():
        items = sorted(items, reverse=True)[:nmax]
        items.sort(key=lambda t: t[3])
        ys = _stack([t[3] for t in items], 0, h - 1, d)
        xt = -0.5 - gap if k == "L" else w - 0.5 + gap
        xb = -0.5 - 0.35 * gap if k == "L" else w - 0.5 + 0.35 * gap
        xs = xt + (0.12 * gap if k == "L" else -0.12 * gap)
        for (_, name, c, r), y in zip(items, ys):
            ax.plot([xs, xb, c], [y, y, r], color="k", lw=0.3, solid_capstyle="round", clip_on=False)
            ax.plot(c, r, "o", ms=1.1, color="k", mec="none", clip_on=False)
            ax.text(xt, y, name, fontsize=fs, ha="right" if k == "L" else "left", va="center", color="k")


def label_rows(ax, a, names, w, h, s, label_mask=None, only=None, fs=LABEL_PT):
    """Labels beside the image with purely horizontal leader lines (all parallel): each area gets a label row that
    crosses its largest piece, anchored at the most interior point of that row. Rows on one side are >= LINE_MM apart;
    the most constrained areas (fewest usable rows) are placed first; areas without a free row are not labelled.
    only: 'L' / 'R' puts every label on that side, else the side nearer to the anchor."""
    d, gap = LINE_MM / s, 1.2 / s
    areas = []
    for k in np.unique(a[a > 0]):
        m = a == k
        if label_mask is not None:
            m &= label_mask
        lab, n = ndimage.label(m)
        if n == 0:
            continue
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        big = int(np.argmax(sizes)) + 1
        if sizes[big - 1] < MIN_LABEL_PX:
            continue
        dist = ndimage.distance_transform_edt(lab == big)
        best = dist.max(1)
        rows = np.flatnonzero(best >= 1.5)
        if rows.size == 0:
            continue
        rows = rows[np.argsort(-best[rows])]
        areas.append((rows.size, names[k], [(int(r), int(np.argmax(dist[r]))) for r in rows]))
    taken = {"L": [], "R": []}
    placed = []
    for _, name, opts in sorted(areas, key=lambda t: t[0]):
        for r, c in opts:
            side = only or ("L" if c < w / 2 else "R")
            if all(abs(r - y) >= d for y in taken[side]):
                taken[side].append(r)
                placed.append((name, r, c, side))
                break
    for name, r, c, side in placed:
        xt = -0.5 - gap if side == "L" else w - 0.5 + gap
        xs = xt + (0.12 * gap if side == "L" else -0.12 * gap)
        ax.plot([xs, c], [r, r], color="k", lw=0.3, solid_capstyle="butt", clip_on=False)
        ax.plot(c, r, "o", ms=1.1, color="k", mec="none", clip_on=False)
        ax.text(xt, r, name, fontsize=fs, ha="right" if side == "L" else "left", va="center", color="k")


def render(ax, ids2d, lut, palette, M):
    """ids2d: (rows = page vertical, cols = page horizontal) structure ids; M = label margin in voxels."""
    g, a = lut(ids2d)
    img = palette[g]
    inside = ids2d > 0
    img[~inside] = 1.0
    img[edges(ids2d) & inside] = C_FINE
    img[edges(a) & inside] = C_AREA
    img[edges(g) & inside] = C_GROUP
    img[edges(inside)] = C_OUTLINE
    ax.imshow(img, interpolation="nearest", aspect="equal", rasterized=True)
    ax.set_xlim(-0.5 - M, ids2d.shape[1] - 0.5 + M)
    ax.set_ylim(ids2d.shape[0] - 0.5, -0.5)
    ax.axis("off")
    return a


def figure(rows, name, lut, palette, order, overlays=None):
    """rows: list of rows, each a list of panels (ids2d, title, label_mask, side). One scale (mm per voxel) for the
    whole figure, set by the most crowded row; rows are centred."""
    ps.set_nature_rc()
    L, R, gap, tgap, rgap = 2.0, 2.0, 2.0, 4.0, 3.0
    avail = lambda r: W - L - R - (len(r) - 1) * gap - len(r) * 2 * MARG  # noqa: E731
    s = min(avail(r) / sum(p[0].shape[1] for p in r) for r in rows)  # mm per voxel
    row_h = [max(p[0].shape[0] for p in r) * s for r in rows]
    LEG_H = 16.0
    H = sum(h + tgap + rgap for h in row_h) + LEG_H
    fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
    y = 0.0
    sb_at = None
    for r, h in zip(rows, row_h):
        y += tgap
        widths = [p[0].shape[1] * s + 2 * MARG for p in r]
        x = L + (W - L - R - sum(widths) - (len(r) - 1) * gap) / 2
        for (ids2d, title, lm, side), w_ in zip(r, widths):
            h_ = ids2d.shape[0] * s
            ax = fig.add_axes([x / W, 1 - (y + h_) / H, w_ / W, h_ / H])
            a = render(ax, ids2d, lut, palette, MARG / s)
            if overlays and title in overlays:
                overlays[title](ax, *CROP_OFF.get(title, (0, 0)))
            label_rows(ax, a, lut.areas, ids2d.shape[1], ids2d.shape[0], s, lm, side)
            ax.set_title(title, fontsize=6, pad=2)
            sb_at = sb_at or (x + MARG, y + h_)
            x += w_ + gap
        y += h + rgap
    # 1-mm scale bar under the first panel
    sb = fig.add_axes([sb_at[0] / W, 1 - (sb_at[1] + 1.8) / H, (1000 / RES * s) / W, 1.0 / H])
    sb.plot([0, 1], [0.5, 0.5], color="k", lw=1, solid_capstyle="butt")
    sb.set_xlim(0, 1)
    sb.axis("off")
    sb.text(1.15, 0.5, "1 mm", transform=sb.transAxes, fontsize=5.5, va="center", ha="left")
    lg = fig.add_axes([L / W, 0, (W - L - R) / W, LEG_H / H])
    lg.axis("off")
    cols = au.get_custom_area_groups_colors()
    handles = [Patch(facecolor=cols[g], edgecolor="none", label=RESPELL.get(g, g)) for g in order]
    handles.append(Patch(facecolor="white", edgecolor="0.5", lw=0.4, label="Not grouped"))
    lg.legend(handles=handles, ncol=6, loc="upper center", frameon=False, fontsize=6, handlelength=1.2,
              handleheight=1.0, columnspacing=1.5, labelspacing=0.5)
    ps.save_figure(fig, OUT, name)
    plt.close(fig)
    print("saved", OUT / name)


CROP_OFF = {}  # panel title -> (row, col) offset of the crop in the full section


def crop(panels):
    """Crop all panels of a figure to the union of their non-empty extents (same size, same scale)."""
    shp = np.array([p[0].shape for p in panels]).max(0)
    inside = np.zeros(shp, bool)
    for p in panels:
        inside[:p[0].shape[0], :p[0].shape[1]] |= p[0] > 0
    r = np.flatnonzero(inside.any(1))
    c = np.flatnonzero(inside.any(0))
    r0, r1, c0, c1 = max(r[0] - 2, 0), r[-1] + 3, max(c[0] - 2, 0), c[-1] + 3
    out = []
    for ids2d, t, lm, side in panels:
        CROP_OFF[t] = (r0, c0)
        out.append((ids2d[r0:r1, c0:c1], t, None if lm is None else lm[r0:r1, c0:c1], side))
    return out


def load_tracks():
    """Atlas-space probe tracks (brainreg-segment, CCF um as AP, DV, LR) of every insertion in the unit table;
    identical files (same-mouse sessions share one track) drawn once."""
    import hashlib
    u = pd.read_parquet(TRACK_UNITS, columns=["mouse_id", "session_id", "electrode_group"])
    ins = u.assign(probe=u.electrode_group.str.split("_").str[0]).drop_duplicates(["session_id", "probe"])
    tracks, seen, missing = [], set(), 0
    for m, sess, pr in ins[["mouse_id", "session_id", "probe"]].itertuples(index=False):
        f = next((ROOT.parent / o / "data" / m / sess / TRACK_SUB / f"{pr}.npy" for o in ("Axel_Bisi", "Myriam_Hamon")
                  if (ROOT.parent / o / "data" / m / sess / TRACK_SUB / f"{pr}.npy").exists()), None)
        if f is None:
            missing += 1
            continue
        xyz = np.load(f)
        h = hashlib.md5(xyz.tobytes()).hexdigest()
        if h not in seen:
            seen.add(h)
            tracks.append(xyz)
    print(f"tracks: {len(tracks)} unique from {len(ins)} insertions ({missing} without a track file)")
    return tracks


def track_overlay(tracks):
    def fn(ax, r0, c0):  # sagittal panel: rows = DV, cols = AP voxel index
        for xyz in tracks:
            ax.plot(xyz[:, 0] / RES - c0, xyz[:, 1] / RES - r0, color="0.3", lw=0.25,
                    solid_capstyle="butt", zorder=3)
    return fn


def fmt(v):
    return f"{v:+.1f}".replace("+0.0", "0.0").replace("-", "−")


# ----------------------------------------------------------------------------------------------------- views
def main():
    from brainglobe_atlasapi import BrainGlobeAtlas
    atlas = BrainGlobeAtlas("allen_mouse_25um", check_latest=False)
    ann = np.asarray(atlas.annotation)  # (AP, DV, LR)
    nap, ndv, nlr = ann.shape
    ap_i = lambda ap_mm: int(round((BREGMA["ap"] - ap_mm * 1000) / RES))  # noqa: E731  ap > 0 anterior
    ml_i = lambda ml_mm: int(round((BREGMA["ml"] + ml_mm * 1000) / RES))  # noqa: E731
    dv_i = lambda dv_mm: int(round((BREGMA["dv"] + dv_mm * 1000) / RES))  # noqa: E731  dv > 0 ventral
    mid = ml_i(0.0)

    df = structure_table(atlas)
    order = au.get_area_group_custom_order()
    cols = au.get_custom_area_groups_colors()
    missing = [g for g in order if g not in cols]
    if missing:
        raise KeyError(f"area groups without a colour: {missing}")
    lut = Lut(df, order)
    palette = np.array([(1.0, 1.0, 1.0)] + [to_rgb(cols[g]) for g in order])

    OUT.mkdir(parents=True, exist_ok=True)
    u, n = np.unique(ann[:, :, mid:], return_counts=True)
    df["n_voxels_hemi"] = df.id.map(dict(zip(u, n))).fillna(0).astype(int)
    df.drop(columns="target_region").sort_values(["area_group", "area_acronym_custom", "ccf_atlas_acronym"]).to_csv(
        OUT / "area_groups_mapping.csv", index=False)

    # sagittal: (DV, AP), anterior left
    sag = lambda mls: crop([(ann[:, :, ml_i(ml)].T, f"Sagittal, ML {ml:.1f} mm", None, None) for ml in mls])  # noqa
    # coronal: (DV, LR) right hemisphere, midline left
    cor = lambda aps: crop([(ann[ap_i(ap), :, mid:], f"Coronal, AP {fmt(ap)} mm", None, None) for ap in aps])  # noqa
    # horizontal: (AP, LR) right hemisphere, anterior up
    hor = lambda dvs: crop([(ann[:, dv_i(dv), mid:], f"Horizontal, DV {dv:.1f} mm", None, None) for dv in dvs])  # noqa

    # surface views: first labelled (non-excluded) voxel seen from above / from the right side
    excl = df.loc[df.excluded, "id"].to_numpy()
    valid = (ann > 0) & ~np.isin(ann, excl)
    k = np.argmax(valid, axis=1)  # (AP, LR) first from dorsal
    dors = np.take_along_axis(ann, k[:, None, :], axis=1)[:, 0, :] * valid.any(1)
    vr = valid[:, :, ::-1]
    k = np.argmax(vr, axis=2)  # (AP, DV) first from the right
    lat = np.take_along_axis(ann[:, :, ::-1], k[:, :, None], axis=2)[:, :, 0] * vr.any(2)
    right = np.zeros(dors.shape, bool)
    right[:, mid:] = True
    surf = [crop([(dors, "Dorsal view (labels: right hemisphere)", right, "R")])[0],
            crop([(lat.T, "Lateral view (right)", None, None)])[0]]

    # oblique planes: coronal tilted about the ML axis, sagittal tilted about the AP axis
    def plane(p0, u, v, nu, nv):
        uu, vv = np.meshgrid(np.arange(nu), np.arange(nv))
        co = [p0[i] + vv * v[i] + uu * u[i] for i in range(3)]
        return ndimage.map_coordinates(ann, co, order=0, mode="constant", cval=0).astype(ann.dtype)

    def obl_cor(ang):
        t = np.deg2rad(ang)  # positive: deeper is more posterior
        p0 = (ap_i(-1.0) - dv_i(0.0) * np.tan(t), 0.0, mid)
        return (plane(p0, (0, 0, 1), (np.sin(t), np.cos(t), 0), nlr - mid, ndv),
                f"Coronal, {abs(ang)}° tilt ({'deeper = posterior' if ang > 0 else 'deeper = anterior'})", None, None)

    def obl_sag(ang):
        t = np.deg2rad(ang)  # deeper is more lateral
        return (plane((0.0, 0.0, ml_i(1.0)), (1, 0, 0), (0, np.cos(t), np.sin(t)), nap, ndv),
                f"Sagittal, {ang}° tilt (deeper = lateral)", None, None)

    # one figure with every view
    figure([surf, sag((0.6, 1.8, 3.2)), cor((2.0, 0.6, -1.4, -3.0)), hor((1.0, 2.2, 3.4, 4.6)),
            crop([obl_cor(-30), obl_cor(30)]) + crop([obl_sag(30)])],
           "area_groups_all_views", lut, palette, order)
    # main schematic: the two views covering most recorded areas (19/20 area groups incl. wS1, wS2, A1, PPC)
    # pons / medulla and amygdala / hypothalamus hidden here (drawn white, unlabelled, not in the legend)
    order2 = [g for g in order if g not in HIDE_TWO_VIEWS]
    lut2 = Lut(df.assign(area_group=df.area_group.where(~df.area_group.isin(HIDE_TWO_VIEWS))), order2)
    palette2 = np.array([(1.0, 1.0, 1.0)] + [to_rgb(cols[g]) for g in order2])
    two = [[sag((0.9,))[0], surf[1]]]
    figure(two, "area_groups_two_views", lut2, palette2, order2)
    figure(two, "area_groups_two_views_tracks", lut2, palette2, order2,
           overlays={two[0][0][1]: track_overlay(load_tracks())})
    if "--per-view" in sys.argv:  # more sections per view, one figure each
        chunk = lambda p, n: [p[i:i + n] for i in range(0, len(p), n)]  # noqa: E731
        figure(chunk(sag((0.5, 1.2, 2.0, 2.8, 3.5, 4.2)), 2), "area_groups_sagittal", lut, palette, order)
        figure(chunk(cor((2.8, 2.0, 1.2, 0.4, -0.5, -1.4, -2.3, -3.2)), 4), "area_groups_coronal", lut, palette, order)
        figure(chunk(hor((0.8, 1.6, 2.4, 3.2, 4.0, 4.8)), 4), "area_groups_horizontal", lut, palette, order)
        figure([surf], "area_groups_surface", lut, palette, order)
        figure([crop([obl_cor(-30), obl_cor(30)]), crop([obl_sag(20), obl_sag(40)])], "area_groups_oblique", lut,
               palette, order)


if __name__ == "__main__":
    main()
