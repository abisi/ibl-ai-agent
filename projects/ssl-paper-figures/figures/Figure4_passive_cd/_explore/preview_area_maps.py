"""Figure 4G candidates, part 3: effects on Allen CCF contour overlays + more alternatives (user 2026-11-25: "I meant heatmaps
on allen contour overlays; provide more alternatives").

Effect per fine area (area_acronym_custom; >= 10 neurons per session, >= 3 sessions per cohort; preview_area_alternatives2):
delta = mean_R+(W - A) - mean_R-(W - A) of the passive evoked change post - pre (5-35 ms, Hz).
Region painting: atlas regions (Allen CCF 25 um, brainglobe allen_mouse_25um_v1.2; cortical layers merged into their area,
barrel columns into SSp-bfd, as 049) are assigned to the fine area holding most of the recorded units located in them (ROC
table ccf_atlas_id); a region takes its fine area's delta; regions without data are light grey; grey lines, region contours.
Panels:
  a  coronal sections (8, at unit-coverage quantiles; one hemisphere), painted with delta
  b  dorsal view (first labelled voxel from the top: cortex and dorsal structures)
  c  sagittal sections (3)
  d  smoothed single-neuron maps (no area boundaries): per unit w - a change (post - pre, whisker minus auditory), Gaussian-
     weighted mean (sigma 200 um) within 500-um coronal slabs, R+ and R- and R+ - R-; pixels with low unit density blank
  e  volcano: delta vs -log10 P (larger P of Mann-Whitney and Welch on sessions), fine areas
  f  ranked delta (waterfall), colour = parent area group
Output: Figures/_explore/Figure4_passive_cd_<date>/area_effect_maps.{png,pdf,svg}
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/preview_area_maps.py
"""
import datetime
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tifffile  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from scipy import ndimage  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
import shared_passive_cd as S  # noqa: E402
from common import FIGURES_ROOT, axel_bisi_root  # noqa: E402
from paperstyle import MM, save_figure, set_nature_rc  # noqa: E402

spec = importlib.util.spec_from_file_location("alt2", HERE / "preview_area_alternatives2.py")
A2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A2)

FS = 5
RES = 25.0
ATL25 = Path.home() / ".brainglobe" / "allen_mouse_25um_v1.2"
BB = axel_bisi_root() / "Anatomy" / "allen_mouse_bluebrain_barrels_10um_v1.0"
MID = 5700.0
CMAP = plt.get_cmap("PuOr_r")


def parent_map():
    st = pd.read_csv(BB / "structures.csv")
    name, acr = dict(zip(st.id, st.name)), dict(zip(st.id, st.acronym))
    par = dict(zip(st.id, st.parent_structure_id.fillna(-1).astype(int)))
    bfd = int(st.loc[st.acronym == "SSp-bfd", "id"].iloc[0])
    out = {}
    for i in st.id:
        j = i
        while j in name and "layer" in str(name[j]).lower() and par.get(j, -1) in name:
            j = par[j]
        if str(acr.get(j, "")).startswith("SSp-bfd"):
            j = bfd
        out[i] = j
    return out


def units_ccf():
    u = A2.units()
    root = axel_bisi_root() / "combined_results_ks4"
    rows = []
    for m in sorted(u.mouse_id.unique()):
        f = root / m / "whisker_0" / "roc_analysis" / f"{m}_roc_stimulus.csv"
        if f.exists():
            r = pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "ccf_atlas_id", "ccf_atlas_ap",
                                        "ccf_atlas_ml", "ccf_atlas_dv"])
            rows.append(r.drop_duplicates(["session_id", "electrode_group", "cluster_id"]))
    c = pd.concat(rows, ignore_index=True).rename(columns={"cluster_id": "cid"})
    c["cid"] = c.cid.astype(int)
    return u.merge(c, on=["session_id", "electrode_group", "cid"], how="left")


def collapse(a, pm):
    uq, inv = np.unique(a, return_inverse=True)
    return np.array([pm.get(int(k), int(k)) for k in uq])[inv].reshape(a.shape)


def paint(ax, lab, val, vmax, extent=None):
    """lab: 2D collapsed region ids; val: region id -> delta"""
    rgb = np.ones(lab.shape + (3,))
    inside = lab != 0
    rgb[inside] = 0.93
    norm = mcolors.Normalize(-vmax, vmax)
    for rid, v in val.items():
        m = lab == rid
        if m.any():
            rgb[m] = CMAP(norm(v))[:3]
    bnd = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
    bnd &= ndimage.binary_dilation(inside)
    rgb[bnd] = 0.55
    ax.imshow(rgb, interpolation="nearest", extent=extent)
    ax.set_axis_off()


def main():
    set_nature_rc()
    pm = parent_map()
    u = units_ccf()
    mf = A2.per_area(u, "area_acronym_custom")
    # P per fine area
    pv = {}
    d = u.groupby(["session_id", "cohort", "area_acronym_custom"]).agg(n=("cid", "size"), W=("ev_post_W", "mean"),
                                                                       W0=("ev_pre_W", "mean"), A=("ev_post_A", "mean"),
                                                                       A0=("ev_pre_A", "mean")).reset_index()
    d = d[d.n >= 10]
    d["WA"] = (d.W - d.W0) - (d.A - d.A0)
    for a_, g in d[d.area_acronym_custom.isin(mf.index)].groupby("area_acronym_custom"):
        t = S.two_group_tests(g[g.cohort == "R+"].WA, g[g.cohort == "R-"].WA)
        pv[a_] = S.pmax(t["mannwhitney_p"], t["welch_p"])
    mf["p"] = pd.Series(pv)
    # region -> fine area (majority of units), -> delta
    u["rid"] = u.ccf_atlas_id.map(lambda x: pm.get(int(x), int(x)) if np.isfinite(x) else 0)
    maj = u[u.area_acronym_custom.isin(mf.index) & (u.rid != 0)].groupby("rid").area_acronym_custom.agg(
        lambda x: x.value_counts().index[0])
    val = {int(r): float(mf.loc[a_, "delta"]) for r, a_ in maj.items()}
    vmax = float(np.nanpercentile(np.abs(mf.delta), 95))
    ann = tifffile.imread(ATL25 / "annotation.tiff")                    # (AP, DV, ML)
    nap, ndv, nml = ann.shape
    ok = u.ccf_atlas_ap.notna()
    aps = np.nanpercentile(u.loc[ok, "ccf_atlas_ap"], np.linspace(6, 94, 8))
    W, H = 183, 200
    fig = plt.figure(figsize=(W * MM, H * MM))

    def ax_at(x, y, w, h):
        return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])

    def letter(x, y, s, t):
        fig.text(x / W, 1 - y / H, s, fontsize=8, fontweight="bold", va="top")
        fig.text((x + 4) / W, 1 - y / H, t, fontsize=6, va="top")

    letter(0, 2, "a", "Coronal sections, Δ per fine area (one hemisphere)")
    midv = int(MID / RES)
    for k, ap in enumerate(aps):
        sl = collapse(ann[int(np.clip(ap / RES, 0, nap - 1))], pm)[:, midv:]          # (DV, ML right)
        ax = ax_at(2 + (k % 4) * 30, 9 + (k // 4) * 26, 28, 24)
        paint(ax, sl, val, vmax)
        ax.set_title(f"AP {ap / 1000:.1f} mm", fontsize=FS - 0.5, pad=1)
    cax = ax_at(124, 12, 1.5, 40)
    cb = fig.colorbar(plt.cm.ScalarMappable(mcolors.Normalize(-vmax, vmax), CMAP), cax=cax)
    cb.set_label("Δ (R+ − R−) of whisker − auditory change (Hz)", fontsize=FS - 0.5)
    cb.ax.tick_params(labelsize=FS - 1, length=1)

    letter(135, 2, "b", "Dorsal view")
    lab3 = ann != 0
    first = np.argmax(lab3, axis=1)                                     # (AP, ML) first labelled DV index
    top = np.take_along_axis(ann, first[:, None, :], axis=1)[:, 0, :]
    top = collapse(top, pm)
    ax = ax_at(138, 9, 42, 50)
    paint(ax, top, val, vmax)

    letter(0, 62, "c", "Sagittal sections")
    mls = np.nanpercentile(MID + np.abs(u.loc[ok, "ccf_atlas_ml"] - MID), [20, 50, 80])
    for k, ml in enumerate(mls):
        sl = collapse(ann[:, :, int(np.clip(ml / RES, 0, nml - 1))], pm)       # (AP, DV)
        ax = ax_at(2 + k * 46, 69, 44, 30)
        paint(ax, sl.T, val, vmax)
        ax.set_title(f"ML {(ml - MID) / 1000:.1f} mm from midline", fontsize=FS - 0.5, pad=1)

    letter(0, 104, "d", "Smoothed single-neuron maps (w − a change, Hz), coronal slabs ±250 µm")
    uu = u[ok].copy()
    uu["wa"] = (uu.ev_post_W - uu.ev_pre_W) - (uu.ev_post_A - uu.ev_pre_A)
    uu["mlf"] = MID + np.abs(uu.ccf_atlas_ml - MID)
    slabs = np.nanpercentile(uu.ccf_atlas_ap, [25, 50, 75])
    sig, step = 200.0, 50.0
    for k, ap in enumerate(slabs):
        s_ = uu[np.abs(uu.ccf_atlas_ap - ap) <= 250]
        sl = collapse(ann[int(np.clip(ap / RES, 0, nap - 1))], pm)[:, midv:]
        gx, gy = np.arange(MID, nml * RES, step), np.arange(0, ndv * RES, step)
        maps = {}
        for c in ("R+", "R-"):
            q = s_[s_.cohort == c]
            num = np.zeros((len(gy), len(gx)))
            den = np.zeros_like(num)
            ix = np.clip(((q.mlf - MID) / step).astype(int), 0, len(gx) - 1)
            iy = np.clip((q.ccf_atlas_dv / step).astype(int), 0, len(gy) - 1)
            np.add.at(num, (iy, ix), q.wa.to_numpy())
            np.add.at(den, (iy, ix), 1)
            num, den = ndimage.gaussian_filter(num, sig / step), ndimage.gaussian_filter(den, sig / step)
            mp = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
            maps[c] = mp
        maps["R+ − R−"] = maps["R+"] - maps["R-"]
        for j, (nm, mp) in enumerate(maps.items()):
            ax = ax_at(2 + j * 30 + k * 0, 112 + k * 28, 28, 25) if False else ax_at(2 + k * 61 + j * 20, 112, 19, 22)
            ext = [0, (nml - midv) * RES, ndv * RES, 0]
            bg = np.ones(sl.shape + (3,))
            bg[sl != 0] = 0.95
            bnd = (np.diff(sl, axis=0, prepend=sl[:1]) != 0) | (np.diff(sl, axis=1, prepend=sl[:, :1]) != 0)
            bg[bnd & (sl != 0)] = 0.6
            ax.imshow(bg, extent=ext, interpolation="nearest")
            v = 4.0
            ax.imshow(mp, extent=[0, len(gx) * step, len(gy) * step, 0], cmap="RdBu_r" if j < 2 else "PuOr_r",
                      vmin=-v, vmax=v, alpha=0.85, interpolation="bilinear")
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_axis_off()
            ax.set_title(f"{nm}" + (f"\nAP {ap / 1000:.1f} mm" if j == 0 else ""), fontsize=FS - 1, pad=1)

    letter(0, 140, "e", "Volcano, fine areas")
    col, _, _ = S.area_palette()
    parent = u.groupby("area_acronym_custom").area_group.agg(lambda x: x.mode().iat[0])
    ax = ax_at(12, 148, 45, 45)
    for a_, r in mf.iterrows():
        c = col.get(parent.get(a_), "0.4")
        ax.plot(r.delta, -np.log10(r.p), "o", ms=3, color=c)
        if r.p < 0.1 or abs(r.delta) > 4:
            ax.text(r.delta, -np.log10(r.p), " " + a_, fontsize=FS - 1.5, color=c, va="center")
    ax.axhline(-np.log10(0.05), color="0.6", lw=0.4, ls=(0, (2, 2)))
    ax.axvline(0, color="0.8", lw=0.4)
    ax.set_xlabel("Δ (R+ − R−) (Hz)", fontsize=FS)
    ax.set_ylabel("−log10 P", fontsize=FS)
    ax.tick_params(labelsize=FS, length=1.5)

    letter(70, 140, "f", "Ranked Δ, fine areas")
    ax = ax_at(80, 148, 98, 38)
    s = mf.sort_values("delta")
    ax.bar(range(len(s)), s.delta, color=[col.get(parent.get(a_), "0.4") for a_ in s.index], width=0.8)
    ax.errorbar(range(len(s)), s.delta, yerr=[s.delta - s.lo, s.hi - s.delta], fmt="none", ecolor="0.3", lw=0.4)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=90, fontsize=FS - 1.5)
    for tl, a_ in zip(ax.get_xticklabels(), s.index):
        tl.set_color(col.get(parent.get(a_), "0.3"))
    ax.axhline(0, color="0.5", lw=0.4)
    ax.set_ylabel("Δ (Hz)", fontsize=FS)
    ax.set_ylim(np.nanpercentile(s.lo, 2) - 1, np.nanpercentile(s.hi, 98) + 1)
    ax.tick_params(labelsize=FS, length=1.5)

    out = FIGURES_ROOT / "_explore" / f"Figure4_passive_cd_{datetime.date.today().isoformat()}"
    save_figure(fig, out, "area_effect_maps")
    print("regions painted", len(val), "fine areas", len(mf), "| saved", out / "area_effect_maps.png")


if __name__ == "__main__":
    main()
