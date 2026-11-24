"""Figure 2 exploration: same layout as explore_sparse.py (half-A4 grid, 0.4-mm slabs, rows whisker / auditory / spontaneous
licks), but smooth maps instead of dots. 2-D Gaussian sigma SIGMA_UM on a 25-um grid within each slab:
  density:   responsive neurons per mm^2 (whisker / auditory rows); licks row = enhanced - suppressed density (diverging)
  selectivity: mean signed ROC selectivity of all recorded neurons (sum of sel / number of neurons, both smoothed);
             whisker / auditory one-sided (white -> modality colour), licks diverging (suppressed blue, enhanced red)
Tissue with < N_MIN recorded neurons within the kernel is not sampled (light grey). Colour range per row = 98th
percentile of |value| over the row's four panels. Units: good + mua, day 0, R+ and R- pooled.
Run on haas (ibl-ai-agent .venv): python explore_density_maps.py
Output: Figure2_explore_density_grid, Figure2_explore_meansel_grid"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import explore_sparse as es  # noqa: E402
bv, cm, m3, ps = es.bv, es.cm, es.m3, es.ps

SIGMA_UM, BIN_UM, N_MIN = 100.0, 25.0, 3.0
UNSAMPLED = np.array(mcolors.to_rgb('#eeeeee'))


def one_sided(c):
    return mcolors.LinearSegmentedColormap.from_list('o', ['#ffffff', c])


def diverging(cn, cp):
    return mcolors.LinearSegmentedColormap.from_list('d', [cn, '#ffffff', cp])


def smooth(x, y, w, xr, yr):
    bx = np.arange(xr[0], xr[1] + 1e-9, BIN_UM / 1000)
    by = np.arange(yr[0], yr[1] + 1e-9, BIN_UM / 1000)
    return bv.ndimage.gaussian_filter(np.histogram2d(y, x, bins=[by, bx], weights=w)[0], SIGMA_UM / BIN_UM), \
        (bx[0], bx[-1], by[-1], by[0])


def compute(A, df, plane, c, row, kind):
    lab = A.section(plane, c)
    ext_lab = (0, lab.shape[1] / 100, lab.shape[0] / 100, 0)
    xcol = 'lat_mm' if plane == 'cor' else 'ap_mm'
    pos = df['ccf_atlas_ap'] if plane == 'cor' else df['ml_f']
    sl = df[(pos - c).abs() <= es.SLAB / 2]
    sl = sl[es.section_inside(lab, sl[xcol].to_numpy(), sl['dv_mm'].to_numpy())]
    x, y = sl[xcol].to_numpy(), sl['dv_mm'].to_numpy()
    xr, yr = (0, ext_lab[1]), (0, ext_lab[2])
    den, ext = smooth(x, y, None, xr, yr)
    s = SIGMA_UM / BIN_UM
    sampled = den * 2 * np.pi * s ** 2 >= N_MIN
    cats = row[2]
    if kind == 'density':
        vals = [smooth(x, y, sl[k].to_numpy(float), xr, yr)[0] / (BIN_UM / 1000) ** 2 for k, _, _, _ in cats]
        v = vals[0] if len(vals) == 1 else vals[0] - vals[1]
    else:
        sel = np.nan_to_num(sl[cats[0][1]].to_numpy(float))
        v = smooth(x, y, sel, xr, yr)[0] / np.maximum(den, 1e-12)
    inside = bv.inside_mask(lab, ext, v.shape)
    v = np.where(sampled & inside, v, np.nan)
    return dict(v=v, ext=ext, inside=inside, lab=lab, ext_lab=ext_lab, n=len(sl),
                cnt=[int(sl[k].sum()) for k, _, _, _ in cats])


def cmap_norm(row, kind, vmax):
    cats = row[2]
    if len(cats) == 2:
        return diverging(cats[1][3], cats[0][3]), mcolors.Normalize(-vmax, vmax)
    return one_sided(cats[0][3]), mcolors.Normalize(0, vmax)


def draw(ax, A, P, cmap, norm, lims):
    rgba = cmap(norm(np.nan_to_num(P['v'])))
    rgba[np.isnan(P['v']), :3] = UNSAMPLED
    rgba[..., 3] = P['inside']
    ax.imshow(rgba, extent=P['ext'], interpolation='bilinear', zorder=1)
    bv.draw_bounds(ax, A, P['lab'], P['ext_lab'])
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    cm.axes_off(ax)


def key(fig, rect, row, kind, cmap, norm, fs=4.2):
    cax = fig.add_axes(rect)
    cb_ = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='horizontal')
    lo, hi = norm.vmin, norm.vmax
    cb_.set_ticks([lo, 0, hi] if lo < 0 else [0, hi])
    cb_.ax.tick_params(labelsize=fs - 0.5, length=1.5, pad=1); cb_.outline.set_linewidth(0.3)
    if kind == 'density':
        lab = 'enhanced - suppressed\nneurons / mm$^2$' if len(row[2]) == 2 else 'responsive\nneurons / mm$^2$'
    else:
        lab = 'mean selectivity\n(all neurons)'
    cb_.set_label(lab, fontsize=fs, labelpad=1)
    if len(row[2]) == 2:
        cax.text(0, 1.6, row[2][1][2], transform=cax.transAxes, fontsize=fs - 0.5, color=row[2][1][3], ha='left')
        cax.text(1, 1.6, row[2][0][2], transform=cax.transAxes, fontsize=fs - 0.5, color=row[2][0][3], ha='right')


def main():
    A = m3.Atlas(load_zones=False)
    u = cm.load_units()
    u = u[A.unit_ids(u) != 0].copy()
    centres = [('Frontal', m3.BREGMA_AP - 2000.0),
               ('wS1', float(u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ccf_atlas_ap'].median())),
               ('Midbrain', float(u.loc[u['area_group'] == 'Midbrain', 'ccf_atlas_ap'].median()))]
    grid = np.arange(m3.MID + 300, m3.MID + 5000, 100.0)
    ml_c = grid[np.argmax([((u['ml_f'] - g).abs() <= es.SLAB / 2).sum() for g in grid])]
    planes = [('cor', c) for _, c in centres] + [('sag', ml_c)]
    lims = []
    for pl, c in planes:
        s = u[((u['ccf_atlas_ap'] if pl == 'cor' else u['ml_f']) - c).abs() <= es.SLAB / 2]
        lims.append(es.zoom(s['lat_mm'] if pl == 'cor' else s['ap_mm'], s['dv_mm']))
    ps.set_nature_rc()
    W, L, R, LEG, GAP, TOP, RGAP, BOT = 105.0, 4.0, 0.5, 15.0, 1.0, 6.5, 4.0, 1.0
    wd = [l[0][1] - l[0][0] for l in lims]
    hd = [l[1][0] - l[1][1] for l in lims]
    k = (W - L - R - LEG - GAP * 4) / sum(wd)
    rowh = max(hd) * k
    H = TOP + 3 * rowh + 3 * RGAP + BOT
    FS = 5.0
    for kind, name in (('density', 'Figure2_explore_density_grid'), ('meansel', 'Figure2_explore_meansel_grid')):
        fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
        for i, row in enumerate(es.ROWS):
            P = [compute(A, u, pl, c, row, kind) for pl, c in planes]
            allv = np.concatenate([p['v'][np.isfinite(p['v'])] for p in P])
            vmax = float(np.percentile(np.abs(allv), 98))
            vmax = float(f'{vmax:.2g}')
            cmap, norm = cmap_norm(row, kind, vmax)
            y_top = H - TOP - i * (rowh + RGAP)
            x = L
            for j, (pl, c) in enumerate(planes):
                w, h = wd[j] * k, hd[j] * k
                ax = fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])
                draw(ax, A, P[j], cmap, norm, lims[j])
                if i == 0:
                    ttl = (f'{centres[j][0]}\nAP {(m3.BREGMA_AP - c) / 1000:+.1f} mm' if pl == 'cor'
                           else f'Sagittal\nML {(c - m3.MID) / 1000:.1f} mm')
                    ax.set_title(ttl, pad=1, fontsize=FS)
                ax.text(1.0, -0.01, ' | '.join(map(str, P[j]['cnt'])) + f" / {P[j]['n']}", transform=ax.transAxes,
                        ha='right', va='top', fontsize=3.8, color='0.3')
                if j == 0:
                    ax.text(-0.03, 0.5, row[1], transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=FS)
                    if i == 2:
                        x0 = lims[0][0][0] + 0.1; y0 = lims[0][1][0] - 0.1
                        ax.plot([x0, x0 + 1], [y0, y0], color='k', lw=0.6)
                        ax.text(x0 + 0.5, y0 - 0.06, '1 mm', ha='center', va='bottom', fontsize=4)
                x += w + GAP
            key(fig, [(W - R - LEG + 1.5) / W, (y_top - rowh * 0.3) / H, (LEG - 3.5) / W, 1.2 / H], row, kind, cmap, norm)
        lax = fig.add_axes([(W - R - LEG + 1.5) / W, (BOT + 3) / H, 2.5 / W, 1.5 / H]); lax.axis('off')
        lax.add_patch(plt.Rectangle((0, 0), 1, 1, color=UNSAMPLED, transform=lax.transAxes))
        fig.text((W - R - LEG + 4.5) / W, (BOT + 3.75) / H, 'not sampled', fontsize=4.2, va='center')
        ps.save_figure(fig, cm.OUT, name, raster_dpi=600)
        plt.close(fig)
        print('saved', name, flush=True)


if __name__ == '__main__':
    main()
