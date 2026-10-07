"""Spatial distribution of pre-lick ROC neuron groups on coronal atlas slices (density maps).

Groups (good + mua units with both pre-lick ROCs tested; 051, all trials):
  all_neurons    every tested neuron (the sampling)
  reward_lick    significant auditory hit vs false alarm
  generalizing   significant auditory hit vs FA AND whisker hit vs FA, same sign (modality-generalizing neurons)
Figures: rows = cohort x stage (R+ learning, R+ expert, R- learning, R- expert), columns = coronal slabs centred on
AP_POSITIONS (CCF um). A neuron belongs to the slab whose centre is within SLAB_UM / 2 of its ccf_atlas_ap
(contiguous slabs: every neuron in exactly one slab); the whole slab is projected onto the atlas slice at its centre.
Three versions per group, each in its own subfolder:
  density/                the group's smoothed density
  points/                 one dot per neuron of the group
  density_with_sampling/  the group's density over the sampling density (all tested neurons, grey); not made for
                          all_neurons, which is the sampling itself
Density = 2D histogram of (ML, DV) in BIN_UM bins, Gaussian-smoothed with fixed SIGMA_UM, divided by the group's total
neuron count (that cohort x stage, all slabs) and the bin area: fraction of the group's neurons per mm2. Groups are
comparable despite different counts and share one colour scale per figure. Neurons are pooled across sessions (sessions
with many neurons weigh more).
Coordinates: Allen CCF (um) as stored, drawn on allen_mouse_bluebrain_barrels_10um_v1.0 annotation (grey region
contours with layers / barrel columns merged, darker brain outline, white background). FOLD_HEMISPHERES mirrors neurons
across the midline onto the right side.
Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/generalizing_units/density_maps/<version>/<group>.{png,pdf,svg}
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
m62 = importlib.import_module("062_pub_convergence_figures")
m49 = importlib.import_module("049_roc_stage_ccf")
OUT = m51.OUTROOT / "generalizing_units" / "density_maps"
AF, WF = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"
SLAB_UM = 800
AP_POSITIONS = np.arange(3000, 9500, SLAB_UM)          # slab centres (CCF AP, um)
BIN_UM = 50
SIGMA_UM = 150
MIDLINE_UM = 5700
FOLD_HEMISPHERES = True
ROWS = m62.GROUPS
from matplotlib.colors import LinearSegmentedColormap as _LSC
GREY = _LSC.from_list("grey_sampling", ["#FFFFFF", "#8C8C8C"])


def atlas_slices():
    import tifffile
    ann = tifffile.imread(m49.ATLAS / "annotation.tiff")      # (AP, DV, ML), 10 um
    pm = m49.structure_parent_map()
    out = {}
    for ap in AP_POSITIONS:
        a = ann[int(np.clip(round(ap / 10), 0, ann.shape[0] - 1))]
        u, inv = np.unique(a, return_inverse=True)
        a = np.array([pm.get(int(k), int(k)) for k in u])[inv].reshape(a.shape)
        bnd = (np.diff(a, axis=0, prepend=a[:1]) != 0) | (np.diff(a, axis=1, prepend=a[:, :1]) != 0)
        out[ap] = (ndimage.binary_dilation(bnd & (a != 0)), ndimage.binary_fill_holes(a != 0))
    return out, ann.shape


def draw_slice(ax, sl, shape):
    bnd, inb = sl
    ex = (0, shape[2] * 10, shape[1] * 10, 0)
    rgba = np.zeros(bnd.shape + (4,)); rgba[..., :3] = 0.6; rgba[..., 3] = bnd * 0.5
    ax.imshow(rgba, extent=ex, interpolation="antialiased", zorder=2)
    ax.contour((np.arange(inb.shape[1]) + 0.5) * 10, (np.arange(inb.shape[0]) + 0.5) * 10, inb.astype(float),
               levels=[0.5], colors="0.3", linewidths=0.45, zorder=3)
    ax.set_xlim(MIDLINE_UM - 300 if FOLD_HEMISPHERES else 0, shape[2] * 10)
    ax.set_ylim(shape[1] * 10, 0); ax.set_aspect("equal"); ax.axis("off")


def density(ml, dv, n_total, shape):
    bx = np.arange(0, shape[2] * 10 + BIN_UM, BIN_UM); by = np.arange(0, shape[1] * 10 + BIN_UM, BIN_UM)
    H, _, _ = np.histogram2d(dv, ml, bins=[by, bx])
    H = ndimage.gaussian_filter(H, SIGMA_UM / BIN_UM) / max(n_total, 1) / (BIN_UM / 1000) ** 2   # fraction per mm2
    return H, (bx[0], bx[-1], by[-1], by[0])


def make_group(plt, W, grp_mask, name, label, slices, shape, versions):
    from matplotlib.colors import LinearSegmentedColormap
    cm = {c: LinearSegmentedColormap.from_list(c, ["#FFFFFF", m62.COH[c], m49.COH_DARK[c]]) for c in m62.COH}
    ml = W.ccf_atlas_ml.to_numpy(float); dv = W.ccf_atlas_dv.to_numpy(float); ap = W.ccf_atlas_ap.to_numpy(float)
    if FOLD_HEMISPHERES:
        ml = MIDLINE_UM + np.abs(ml - MIDLINE_UM)
    slab = np.full(len(W), -1)
    for i, c in enumerate(AP_POSITIONS):
        slab[np.abs(ap - c) <= SLAB_UM / 2] = i
    ok = np.isfinite(ml) & np.isfinite(dv) & (slab >= 0)
    sampling = W.tested.to_numpy() & ok
    D, S = {}, {}
    for k in ROWS:
        rk = ((W.cohort == k[0]) & (W.stage == k[1])).to_numpy()
        g, s = grp_mask & ok & rk, sampling & rk
        for i in range(len(AP_POSITIONS)):
            D[(k, i)] = density(ml[g & (slab == i)], dv[g & (slab == i)], g.sum(), shape)
            S[(k, i)] = density(ml[s & (slab == i)], dv[s & (slab == i)], s.sum(), shape)
        D[(k, "n")] = int(g.sum()); D[(k, "pts")] = (ml[g], dv[g], slab[g])
    vmax = np.nanpercentile(np.concatenate([D[(k, i)][0].ravel() for k in ROWS for i in range(len(AP_POSITIONS))]), 99.7)
    smax = np.nanpercentile(np.concatenate([S[(k, i)][0].ravel() for k in ROWS for i in range(len(AP_POSITIONS))]), 99.7)
    for version in versions:
        fig, axs = plt.subplots(len(ROWS), len(AP_POSITIONS), figsize=(m62.W_IN, 4.3),
                                gridspec_kw=dict(wspace=0.02, hspace=0.06))
        for r, k in enumerate(ROWS):
            for i, apc in enumerate(AP_POSITIONS):
                ax = axs[r, i]
                H, ext = D[(k, i)]
                if version == "points":
                    x, y, sb = D[(k, "pts")]
                    ax.scatter(x[sb == i], y[sb == i], s=0.6, color=m62.COH[k[0]], lw=0, alpha=0.8, rasterized=True, zorder=4)
                else:
                    if version == "density_with_sampling":
                        Hs, exs = S[(k, i)]
                        ax.imshow(np.ma.masked_less(Hs, smax * 0.02), cmap=GREY, vmin=0, vmax=smax * 0.7, extent=exs,
                                  interpolation="bilinear", zorder=1, alpha=0.9)
                    Hm = np.ma.masked_less(H, vmax * 0.03)
                    ax.imshow(Hm, cmap=cm[k[0]], vmin=0, vmax=vmax, extent=ext, interpolation="bilinear", zorder=1.5,
                              alpha=0.75 if version == "density_with_sampling" else 1)
                draw_slice(ax, slices[apc], shape)
                if r == 0:
                    ax.set_title(f"AP {apc / 1000:.1f} ± {SLAB_UM / 2000:.1f} mm", fontsize=4.8, pad=2)
                if i == 0:
                    ax.text(-0.05, 0.5, f"{m62.GLAB[k]}\n(n = {D[(k, 'n')]})", transform=ax.transAxes, rotation=90,
                            ha="right", va="center", fontsize=5.3, color=m62.COH[k[0]])
        if version != "points":
            for j, c in enumerate(["R+", "R-"]):
                ca = fig.add_axes([0.905 + 0.025 * j, 0.3, 0.008, 0.4])
                cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cm[c], norm=plt.Normalize(0, vmax)), cax=ca)
                ca.tick_params(labelsize=4.5); ca.set_title(c.replace("-", "−"), fontsize=5, color=m62.COH[c])
                if j == 0:
                    ca.set_yticks([])
                else:
                    cb.set_label("Fraction of the group's\nneurons per mm²", fontsize=5)
        sub = {"density": "smoothed density", "points": "one dot per neuron",
               "density_with_sampling": "density (colour) over the sampling density of all tested neurons (grey)"}[version]
        fig.suptitle(f"{label}\n{sub}; coronal slabs, hemispheres folded, Gaussian σ = {SIGMA_UM} µm", fontsize=6.3,
                     x=0.06, ha="left", y=0.995)
        fig.subplots_adjust(left=0.05, right=0.88, top=0.86, bottom=0.02)
        d = OUT / version; d.mkdir(parents=True, exist_ok=True)
        m62.save(fig, d, name); plt.close(fig)


def main():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"]) & W.quality_label.isin(["good", "mua"])].reset_index(drop=True)
    W["tested"] = W[f"sig:{AF}"].notna() & W[f"sig:{WF}"].notna()
    saf, swf = np.sign(W[f"sel:{AF}"]), np.sign(W[f"sel:{WF}"])
    rl = (W.tested & (W[f"sig:{AF}"] == 1)).to_numpy()
    gen = (W.tested & (W[f"sig:{AF}"] == 1) & (W[f"sig:{WF}"] == 1) & (saf == swf)).to_numpy()
    plt = m62.setup()
    slices, shape = atlas_slices()
    make_group(plt, W, W.tested.to_numpy(), "all_neurons", "All tested neurons (sampling)", slices, shape,
               ["density", "points"])
    make_group(plt, W, rl, "reward_lick", "Reward-lick neurons (auditory hit ≠ false alarm)", slices, shape,
               ["density", "points", "density_with_sampling"])
    make_group(plt, W, gen, "generalizing", "Modality-generalizing neurons (auditory and whisker hit ≠ false alarm, same sign)",
               slices, shape, ["density", "points", "density_with_sampling"])
    print("ALL DONE")


if __name__ == "__main__":
    main()
