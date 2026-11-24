"""Figure 2 exploration: bivariate CCF maps (no aggregation per area, no mirroring).
Per row two categories A / B (stimulus: whisker_active | auditory_active significant; licks: spontaneous_licks enhanced |
suppressed; decision: whisker_decision | auditory_decision). Neurons folded onto one hemisphere. Within each coronal slab
(or the whole brain for the sagittal panel) the counts of recorded neurons, A neurons and B neurons are binned on a 25-um
grid and smoothed with a 2-D Gaussian (sigma 150 um); fA, fB = smoothed A / smoothed recorded (fractions).
Colour per pixel: hue = preference fA / (fA + fB) between the A and B colours (grey when equal), intensity = fA + fB
(white = recorded neurons but none in A or B, full colour = fA + fB >= SAT); light grey = tissue with < N_MIN neurons within
the kernel (not sampled). Structure boundaries: merged Allen structures (10-um barrel atlas, 003_spatial_maps).
Figure 1 (grid): 3 coronal slabs (1 mm; frontal AP +2.0, wS1 = median AP of SSp-bfd neurons, midbrain = median AP of
Midbrain neurons) + sagittal (all neurons, ML collapsed). Figure 2 (AP series): 8 slabs of 0.9 mm from AP +2.5 to -4.0.
Units: good + mua, day 0, R+ and R- pooled; neurons outside the brain or outside the slab-centre section removed.
Run on haas (ibl-ai-agent .venv): python explore_bivariate.py"""
import datetime
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[3] / 'projects' / 'ssl-paper-figures'))
sys.path.insert(0, str(REPO / 'projects' / 'ssl-sensory-spatial-maps' / 'exploratory-analyses'))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage  # noqa: E402

import paperstyle as ps  # noqa: E402
import explore_ccf_maps as cm  # noqa: E402  (load_units, ROWS colours)
m3 = cm.m3

BIN_UM, SIGMA_UM, N_MIN = 25.0, 150.0, 5.0
SAT = {'stimulus': 0.45, 'lick': 0.6, 'decision': 0.4}      # fA + fB giving full colour
GREY_UNSAMPLED = np.array(mcolors.to_rgb('#e4e4e4'))
NEUTRAL = np.array(mcolors.to_rgb('#7a7a7a'))
ROWS = [('stimulus', 'Stimulus response', ('sig_whisker_active', 'Whisker', cm.C_W), ('sig_auditory_active', 'Auditory', cm.C_A)),
        ('lick', 'Spontaneous licks', ('lick_enh', 'Enhanced', cm.C_EN), ('lick_sup', 'Suppressed', cm.C_SU)),
        ('decision', 'Decision', ('whisker_decision', 'Whisker decision', cm.C_W), ('auditory_decision', 'Auditory decision', cm.C_A))]


def bivariate_rgb(fa, fb, sampled, ca, cb, sat):
    ca, cb = np.array(mcolors.to_rgb(ca)), np.array(mcolors.to_rgb(cb))
    tot = np.nan_to_num(fa + fb)
    t = np.where(tot > 0, np.nan_to_num(fa) / np.maximum(tot, 1e-9), 0.5)[..., None]
    # hue: B (t=0) -> neutral grey (t=0.5) -> A (t=1)
    hue = np.where(t >= 0.5, NEUTRAL + (ca - NEUTRAL) * (2 * t - 1), NEUTRAL + (cb - NEUTRAL) * (1 - 2 * t))
    s = np.clip(tot / sat, 0, 1)[..., None]
    rgb = 1 - s * (1 - hue)
    rgb[~sampled] = GREY_UNSAMPLED
    return rgb


def smooth_maps(x_mm, y_mm, masks, xr, yr):
    """2-D binned counts (all recorded + one per mask) smoothed with a Gaussian; returns fractions, sampled mask, extent"""
    bx = np.arange(xr[0], xr[1] + 1e-9, BIN_UM / 1000)
    by = np.arange(yr[0], yr[1] + 1e-9, BIN_UM / 1000)
    s = SIGMA_UM / BIN_UM
    den = ndimage.gaussian_filter(np.histogram2d(y_mm, x_mm, bins=[by, bx])[0], s)
    sampled = den * 2 * np.pi * s ** 2 >= N_MIN
    fr = []
    for m in masks:
        num = ndimage.gaussian_filter(np.histogram2d(y_mm[m], x_mm[m], bins=[by, bx])[0], s)
        fr.append(np.where(sampled, num / np.maximum(den, 1e-12), np.nan))
    return fr, sampled, (bx[0], bx[-1], by[-1], by[0])


def inside_mask(lab, extent, shape):
    """brain-section mask (10-um label image with its extent) resampled to a map of `shape` with the same extent"""
    yy = np.clip(((np.arange(shape[0]) + 0.5) / shape[0] * (extent[2] - extent[3]) + extent[3]) * 100, 0, lab.shape[0] - 1).astype(int)
    xx = np.clip(((np.arange(shape[1]) + 0.5) / shape[1] * (extent[1] - extent[0]) + extent[0]) * 100, 0, lab.shape[1] - 1).astype(int)
    return (lab > 0)[np.ix_(yy, xx)]


def draw_bounds(ax, A, lab, extent):
    bnd, inside = A.boundaries(lab)
    rgba = np.zeros(bnd.shape + (4,)); rgba[..., :3] = 0.45; rgba[..., 3] = bnd * 0.55
    ax.imshow(rgba, extent=extent, interpolation='antialiased', zorder=3)
    yy = extent[3] + (np.arange(lab.shape[0]) + 0.5) * (extent[2] - extent[3]) / lab.shape[0]
    xx = extent[0] + (np.arange(lab.shape[1]) + 0.5) * (extent[1] - extent[0]) / lab.shape[1]
    ax.contour(xx, yy, inside.astype(float), levels=[0.5], colors='0.15', linewidths=0.6, zorder=4)


def coronal_panel(ax, A, u, c, slab, row, label=True):
    key, title, (ka, la, ca), (kb, lb, cb) = row
    lab = A.section('cor', c)                                         # (DV, lateral), right hemisphere, 10 um
    ext_lab = (0, lab.shape[1] / 100, lab.shape[0] / 100, 0)
    sl = u[(u['ccf_atlas_ap'] - c).abs() <= slab / 2]
    di = np.clip(np.round(sl['dv_mm'].to_numpy() * 100).astype(int), 0, lab.shape[0] - 1)
    li = np.clip(np.round(sl['lat_mm'].to_numpy() * 100).astype(int), 0, lab.shape[1] - 1)
    sl = sl[lab[di, li] > 0]
    (fa, fb), sampled, ext = smooth_maps(sl['lat_mm'].to_numpy(), sl['dv_mm'].to_numpy(),
                                         [sl[ka].to_numpy(), sl[kb].to_numpy()], (0, ext_lab[1]), (0, ext_lab[2]))
    inside = inside_mask(lab, ext, fa.shape)
    img = bivariate_rgb(fa, fb, sampled & inside, ca, cb, SAT[key])
    img = np.concatenate([img, np.where(inside, 1.0, 0.0)[..., None]], axis=2)       # transparent outside the brain
    ax.imshow(img, extent=ext, interpolation='bilinear', zorder=1)
    draw_bounds(ax, A, lab, ext_lab)
    ax.set_xlim(0, 5.9); ax.set_ylim(7.9, 0.1)
    cm.axes_off(ax)
    return sl, fa, fb


def sagittal_panel(ax, A, u, row):
    key, title, (ka, la, ca), (kb, lb, cb) = row
    lab = A.section('sag', m3.MID + 1500)
    ext_lab = (0, lab.shape[1] / 100, lab.shape[0] / 100, 0)
    (fa, fb), sampled, ext = smooth_maps(u['ap_mm'].to_numpy(), u['dv_mm'].to_numpy(), [u[ka].to_numpy(), u[kb].to_numpy()],
                                         (0, ext_lab[1]), (0, ext_lab[2]))
    brain = (A.ann[::5, ::5, ::5] > 0).any(2).T                       # (DV, AP) silhouette, 50 um
    yy = np.clip(((np.arange(fa.shape[0]) + 0.5) * BIN_UM / 50).astype(int), 0, brain.shape[0] - 1)
    xx = np.clip(((np.arange(fa.shape[1]) + 0.5) * BIN_UM / 50).astype(int), 0, brain.shape[1] - 1)
    inside = brain[np.ix_(yy, xx)]
    img = bivariate_rgb(fa, fb, sampled & inside, ca, cb, SAT[key])
    img = np.concatenate([img, np.where(inside, 1.0, 0.0)[..., None]], axis=2)
    ax.imshow(img, extent=ext, interpolation='bilinear', zorder=1)
    bnd, _ = A.boundaries(lab)
    rgba = np.zeros(bnd.shape + (4,)); rgba[..., :3] = 0.45; rgba[..., 3] = bnd * 0.4
    ax.imshow(rgba, extent=ext_lab, interpolation='antialiased', zorder=3)
    ax.contour((np.arange(brain.shape[1]) + 0.5) * 0.05, (np.arange(brain.shape[0]) + 0.5) * 0.05, brain.astype(float),
               levels=[0.5], colors='0.15', linewidths=0.6, zorder=4)
    ax.set_xlim(1.0, 12.4); ax.set_ylim(7.9, 0.1)
    cm.axes_off(ax)


def colour_key(ax, row):
    key, title, (ka, la, ca), (kb, lb, cb) = row
    t = np.linspace(0, 1, 60)
    s = np.linspace(0, 1, 40)
    T, Sg = np.meshgrid(t, s)
    fa, fb = T * Sg * SAT[key], (1 - T) * Sg * SAT[key]
    img = bivariate_rgb(fa, fb, np.ones_like(fa, bool), ca, cb, SAT[key])
    ax.imshow(img, extent=(0, 1, 0, SAT[key]), origin='lower', aspect='auto')
    ax.set_xticks([0, 1], [lb, la], fontsize=4.5)
    ax.set_yticks([0, SAT[key]], ['0', f'{SAT[key]:g}'], fontsize=4.5)
    ax.set_ylabel('fraction A+B', fontsize=4.5, labelpad=1)
    ax.tick_params(length=1.5, pad=1)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.3)
    ax.text(0.5, -0.5 * SAT[key], 'preference', ha='center', va='top', fontsize=4.5)
    ax.add_patch(plt.Rectangle((0, SAT[key] * 1.25), 0.25, SAT[key] * 0.25, color=GREY_UNSAMPLED, clip_on=False))
    ax.text(0.3, SAT[key] * 1.37, 'not sampled', fontsize=4.5, va='center')


def main():
    A = m3.Atlas(load_zones=False)
    u = cm.load_units()
    ids = A.unit_ids(u)
    u = u[ids != 0].copy()
    print('units', len(u))
    centres = [('Frontal', m3.BREGMA_AP - 2000.0),
               ('wS1', float(u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ccf_atlas_ap'].median())),
               ('Midbrain', float(u.loc[u['area_group'] == 'Midbrain', 'ccf_atlas_ap'].median()))]
    ps.set_nature_rc()
    out = cm.OUT
    # ---- grid: 3 coronal slabs + sagittal + colour key
    fig = plt.figure(figsize=(183 * ps.MM, 150 * ps.MM))
    gs = fig.add_gridspec(3, 5, width_ratios=[0.62, 0.62, 0.62, 1.2, 0.32], wspace=0.05, hspace=0.12,
                          left=0.05, right=0.985, top=0.93, bottom=0.04)
    for i, row in enumerate(ROWS):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            name, c = centres[j]
            sl, _, _ = coronal_panel(ax, A, u, c, 1000.0, row)
            if i == 0:
                ax.set_title(f'{name}\nAP {(m3.BREGMA_AP - c) / 1000:+.1f} mm', pad=1)
            ax.text(5.8, 7.8, f'n = {len(sl)}', ha='right', va='bottom', fontsize=4.5, color='0.35')
            if j == 0:
                ax.text(-0.04, 0.5, row[1], transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
                if i == 2:
                    ax.plot([0.3, 1.3], [7.6, 7.6], color='k', lw=0.8); ax.text(0.8, 7.5, '1 mm', ha='center', va='bottom', fontsize=4.5)
        ax = fig.add_subplot(gs[i, 3]); sagittal_panel(ax, A, u, row)
        if i == 0:
            ax.set_title('Sagittal, all neurons\n(ML collapsed)', pad=1)
        kax = fig.add_subplot(gs[i, 4])
        pos = kax.get_position(); kax.set_position([pos.x0 + 0.02, pos.y0 + pos.height * 0.35, pos.width * 0.6, pos.height * 0.35])
        colour_key(kax, row)
    fig.suptitle('Bivariate CCF maps: hue = preference between the two categories, intensity = fraction of recorded neurons '
                 'in either (2-D Gaussian, sigma 150 um); day 0, R+ and R- pooled', fontsize=6, y=0.99)
    ps.save_figure(fig, out, 'Figure2_explore_bivariate_grid', raster_dpi=600)
    plt.close(fig)
    print('saved grid', flush=True)
    # ---- AP series
    aps = np.linspace(2.5, -4.0, 8)
    fig = plt.figure(figsize=(183 * ps.MM, 105 * ps.MM))
    gs = fig.add_gridspec(3, 9, width_ratios=[1] * 8 + [0.55], wspace=0.04, hspace=0.08, left=0.04, right=0.99, top=0.9, bottom=0.03)
    for i, row in enumerate(ROWS):
        for j, a in enumerate(aps):
            ax = fig.add_subplot(gs[i, j])
            sl, _, _ = coronal_panel(ax, A, u, m3.BREGMA_AP - a * 1000, 900.0, row)
            if i == 0:
                ax.set_title(f'AP {a:+.1f}', pad=1)
            ax.text(5.8, 7.8, f'{len(sl)}', ha='right', va='bottom', fontsize=4, color='0.35')
            if j == 0:
                ax.text(-0.05, 0.5, row[1], transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
        kax = fig.add_subplot(gs[i, 8])
        pos = kax.get_position(); kax.set_position([pos.x0 + 0.015, pos.y0 + pos.height * 0.3, pos.width * 0.6, pos.height * 0.4])
        colour_key(kax, row)
    fig.suptitle('AP series (0.9-mm slabs, one hemisphere): same bivariate maps; numbers = neurons per slab', fontsize=6, y=0.985)
    ps.save_figure(fig, out, 'Figure2_explore_bivariate_APseries', raster_dpi=600)
    plt.close(fig)
    print('saved AP series', flush=True)


if __name__ == '__main__':
    main()
