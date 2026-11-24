"""Figure 2 exploration: track ribbons on CCF sections (no interpolation between tracks, no aggregation per area).
Every probe shank (session x electrode_group) with neurons in the slab is drawn at its CCF position (one hemisphere,
neurons folded) as two parallel ribbons offset across the track: category A (left of the track direction) and B (right),
in their own colours. Ribbon width at each point = fraction of neurons significant in that category among the WIN nearest
neurons along the shank (sorted on the shank's main axis); the thin grey line is the sampled extent. Rows / categories /
slabs as explore_bivariate.py (stimulus whisker | auditory; licks enhanced | suppressed; decision whisker | auditory).
Units: good + mua, day 0, R+ and R- pooled. Run on haas (ibl-ai-agent .venv): python explore_ribbons.py"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402

import explore_bivariate as bv  # noqa: E402
cm, m3, ps = bv.cm, bv.m3, bv.ps

WIN, MIN_UNITS, OFFSET_MM, W_MAX = 30, 20, 0.035, 2.4       # neurons per window, shank minimum, ribbon offset, width at f=1 (pt)
F_FULL = {'stimulus': 0.6, 'lick': 0.8, 'decision': 0.6}    # fraction drawn at full width


def shank_profiles(df, xcol, ycol, cols):
    """per shank: units sorted along the main axis in the (x, y) plane, rolling fractions of each column"""
    out = []
    for _, g in df.groupby(['session_id', 'electrode_group']):
        if len(g) < MIN_UNITS:
            continue
        xy = g[[xcol, ycol]].to_numpy(float)
        c = xy.mean(0)
        d = np.linalg.svd(xy - c, full_matrices=False)[2][0]
        if d[1] < 0:
            d = -d                                                   # main axis pointing down (increasing DV)
        o = np.argsort((xy - c) @ d)
        xy = xy[o]
        fr = []
        for col in cols:
            v = g[col].to_numpy(float)[o]
            fr.append(uniform_filter1d(v, size=min(WIN, len(v)), mode='nearest'))
        # smooth positions too (unit coordinates jitter around the track)
        xs, ys = uniform_filter1d(xy[:, 0], 9, mode='nearest'), uniform_filter1d(xy[:, 1], 9, mode='nearest')
        out.append((np.c_[xs, ys], d, fr))
    return out


def draw_ribbons(ax, profiles, colours, f_full, flip_x=False):
    for xy, d, fr in profiles:
        n = np.array([-d[1], d[0]])                                  # in-plane normal
        ax.plot(xy[:, 0], xy[:, 1], color='0.55', lw=0.25, zorder=4, solid_capstyle='round')
        for side, f, col in zip((-1, 1), fr, colours):
            p = xy + side * OFFSET_MM * n
            seg = np.stack([p[:-1], p[1:]], axis=1)
            w = np.clip((f[:-1] + f[1:]) / 2 / f_full, 0, 1) * W_MAX
            keep = w > 0.05
            ax.add_collection(LineCollection(seg[keep], linewidths=w[keep], colors=col, capstyle='butt', zorder=5))


def coronal(ax, A, u, c, slab, row):
    key, title, (ka, la, ca), (kb, lb, cb) = row
    lab = A.section('cor', c)
    ext_lab = (0, lab.shape[1] / 100, lab.shape[0] / 100, 0)
    sl = u[(u['ccf_atlas_ap'] - c).abs() <= slab / 2]
    di = np.clip(np.round(sl['dv_mm'].to_numpy() * 100).astype(int), 0, lab.shape[0] - 1)
    li = np.clip(np.round(sl['lat_mm'].to_numpy() * 100).astype(int), 0, lab.shape[1] - 1)
    sl = sl[lab[di, li] > 0]
    bnd, inside = A.boundaries(lab)
    fill = np.zeros(lab.shape + (4,)); fill[..., :3] = 0.97; fill[..., 3] = inside
    ax.imshow(fill, extent=ext_lab, zorder=1)
    bv.draw_bounds(ax, A, lab, ext_lab)
    prof = shank_profiles(sl, 'lat_mm', 'dv_mm', [ka, kb])
    draw_ribbons(ax, prof, (ca, cb), F_FULL[key])
    ax.set_xlim(0, 5.9); ax.set_ylim(7.9, 0.1)
    cm.axes_off(ax)
    return sl, len(prof)


def sagittal(ax, A, u, row):
    key, title, (ka, la, ca), (kb, lb, cb) = row
    brain = (A.ann[::5, ::5, ::5] > 0).any(2).T
    lab = A.section('sag', m3.MID + 1500)
    ext_lab = (0, lab.shape[1] / 100, lab.shape[0] / 100, 0)
    fill = np.zeros(brain.shape + (4,)); fill[..., :3] = 0.97; fill[..., 3] = brain
    ax.imshow(fill, extent=(0, brain.shape[1] * 0.05, brain.shape[0] * 0.05, 0), zorder=1)
    bnd, _ = A.boundaries(lab)
    rgba = np.zeros(bnd.shape + (4,)); rgba[..., :3] = 0.45; rgba[..., 3] = bnd * 0.35
    ax.imshow(rgba, extent=ext_lab, interpolation='antialiased', zorder=3)
    ax.contour((np.arange(brain.shape[1]) + 0.5) * 0.05, (np.arange(brain.shape[0]) + 0.5) * 0.05, brain.astype(float),
               levels=[0.5], colors='0.15', linewidths=0.6, zorder=4)
    prof = shank_profiles(u, 'ap_mm', 'dv_mm', [ka, kb])
    draw_ribbons(ax, prof, (ca, cb), F_FULL[key])
    ax.set_xlim(1.0, 12.4); ax.set_ylim(7.9, 0.1)
    cm.axes_off(ax)
    return len(prof)


def key_panel(ax, row):
    key, title, (ka, la, ca), (kb, lb, cb) = row
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    y0 = 0.82
    for f in (0.1, 0.3, 0.6):
        w = min(f / F_FULL[key], 1) * W_MAX
        ax.plot([0.05, 0.3], [y0, y0], color=ca, lw=w, solid_capstyle='butt')
        ax.plot([0.05, 0.3], [y0 - 0.06, y0 - 0.06], color=cb, lw=w, solid_capstyle='butt')
        ax.text(0.36, y0 - 0.03, f'{f:g}', va='center', fontsize=4.5)
        y0 -= 0.2
    ax.text(0.05, 0.97, 'fraction significant\n(ribbon width)', fontsize=4.5, va='top')
    ax.text(0.05, 0.12, la, color=ca, fontsize=5); ax.text(0.05, 0.03, lb, color=cb, fontsize=5)


def main():
    A = m3.Atlas(load_zones=False)
    u = cm.load_units()
    u = u[A.unit_ids(u) != 0].copy()
    centres = [('Frontal', m3.BREGMA_AP - 2000.0),
               ('wS1', float(u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ccf_atlas_ap'].median())),
               ('Midbrain', float(u.loc[u['area_group'] == 'Midbrain', 'ccf_atlas_ap'].median()))]
    ps.set_nature_rc()
    fig = plt.figure(figsize=(183 * ps.MM, 150 * ps.MM))
    gs = fig.add_gridspec(3, 5, width_ratios=[0.62, 0.62, 0.62, 1.2, 0.3], wspace=0.05, hspace=0.12,
                          left=0.05, right=0.99, top=0.93, bottom=0.03)
    for i, row in enumerate(bv.ROWS):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j]); name, c = centres[j]
            sl, ns = coronal(ax, A, u, c, 1000.0, row)
            if i == 0:
                ax.set_title(f'{name}\nAP {(m3.BREGMA_AP - c) / 1000:+.1f} mm', pad=1)
            ax.text(5.8, 7.8, f'{ns} shanks, {len(sl)} neurons', ha='right', va='bottom', fontsize=4.3, color='0.35')
            if j == 0:
                ax.text(-0.04, 0.5, row[1], transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
        ax = fig.add_subplot(gs[i, 3]); ns = sagittal(ax, A, u, row)
        if i == 0:
            ax.set_title('Sagittal, all shanks\n(ML collapsed)', pad=1)
        key_panel(fig.add_subplot(gs[i, 4]), row)
    fig.suptitle(f'Track ribbons: per shank, fraction of the {WIN} nearest neurons significant (left ribbon = first category, '
                 'right = second); day 0, R+ and R- pooled', fontsize=6, y=0.99)
    ps.save_figure(fig, cm.OUT, 'Figure2_explore_ribbons_grid', raster_dpi=600)
    print('saved', flush=True)


if __name__ == '__main__':
    main()
