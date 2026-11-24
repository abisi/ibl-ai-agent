"""Figure 2 exploration, alternatives to the dot grid (explore_slice_grid.py, same units, slabs and rows):
  A  fraction maps: share of recorded units that are significant, 0.25-mm bins, counts smoothed (Gaussian, 1 bin) before
     the ratio, bins with < MIN_N units hidden; coronal mirrored (left = first category, right = second); sagittal =
     difference of the two fractions (diverging)
  B  preference maps: fraction(first) - fraction(second) per bin, diverging, both hemispheres pooled (mirror image)
  C  area bubbles: one circle per area (area_acronym_custom) per slab at the median unit position; size = units,
     colour = fraction significant; coronal mirrored as in A; sagittal = difference, colour diverging
Run on haas (brainrender env): python explore_slice_variants.py [--all-days]
Output: Figures/Figure2_sensory_maps_explore_<date>/Figure2_explore_variant_{A,B,C}.*"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402
from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402

import explore_slice_grid as g  # noqa: E402  (load_units, ROWS, constants, OUT)
import paperstyle as ps  # noqa: E402

BIN, MIN_N, SIG = 0.25, 15, 1.0
LAT = np.arange(0, 6.0 + BIN, BIN)
DV = np.arange(-0.5, 7.75 + BIN, BIN)
AP = np.arange(-7.75, 4.5 + BIN, BIN)


def cmap_to(c):
    return LinearSegmentedColormap.from_list('w', ['#ffffff', c])


def diverging(c_neg, c_pos):
    return LinearSegmentedColormap.from_list('d', [c_neg, '#ffffff', c_pos])


def frac_map(x, y, m, xe, ye):
    n, _, _ = np.histogram2d(x, y, [xe, ye])
    k, _, _ = np.histogram2d(x[m], y[m], [xe, ye])
    ns, ks = gaussian_filter(n, SIG), gaussian_filter(k, SIG)
    f = np.where(ns > 0, ks / np.maximum(ns, 1e-9), np.nan)
    f[gaussian_filter(n, SIG) < MIN_N * 0.5] = np.nan       # smoothed count of at least MIN_N/2
    f[n + gaussian_filter(n, SIG) == 0] = np.nan
    return f


class Atlas:
    def __init__(self):
        a = BrainGlobeAtlas('allen_mouse_25um')
        self.ann, self.res = a.annotation, a.resolution[0]
        lut = np.zeros(int(self.ann.max()) + 1, dtype=np.int64)
        for sid, st in a.structures.items():
            if isinstance(sid, int) and sid < len(lut):
                p = st['structure_id_path']; lut[sid] = p[min(len(p), 7) - 1]
        self.lut = lut

    def coronal(self, ax, ap_mm):
        sec = self.ann[int(round((g.BREGMA_AP - ap_mm * 1000) / self.res))]
        ml = (np.arange(sec.shape[1]) * self.res + self.res / 2 - g.MID) / 1000
        dv = (np.arange(sec.shape[0]) * self.res + self.res / 2 - g.BREGMA_DV) / 1000
        lab = self.lut[sec]
        ax.contour(ml, dv, lab, levels=np.unique(lab)[1:] - 0.5, colors=['#c8c8c8'], linewidths=0.25, zorder=5)
        ax.contour(ml, dv, (sec > 0).astype(float), levels=[0.5], colors='k', linewidths=0.5, zorder=6)
        return sec

    def sagittal(self, ax, lat_mm):
        proj = (self.ann > 0).any(axis=2)
        ap = (g.BREGMA_AP - (np.arange(self.ann.shape[0]) * self.res + self.res / 2)) / 1000
        dv = (np.arange(self.ann.shape[1]) * self.res + self.res / 2 - g.BREGMA_DV) / 1000
        lab = self.lut[self.ann[:, :, int(round((g.MID + lat_mm * 1000) / self.res))]]
        ax.contour(ap, dv, lab.T, levels=np.unique(lab)[1:] - 0.5, colors=['#c8c8c8'], linewidths=0.25, zorder=5)
        ax.contour(ap, dv, proj.T.astype(float), levels=[0.5], colors='k', linewidths=0.5, zorder=6)


def style(ax, coronal):
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if coronal:
        ax.set_xlim(-5.9, 5.9); ax.set_ylim(7.8, -1.2)
    else:
        ax.set_xlim(4.4, -7.6); ax.set_ylim(7.8, -1.2)


def slab_units(u, atlas, c):
    sl = u[np.abs(u['ap_mm'] - c) <= g.SLAB_MM / 2]
    sec = atlas.ann[int(round((g.BREGMA_AP - c * 1000) / atlas.res))]
    di = np.clip(np.round((sl['dv_mm'] * 1000 + g.BREGMA_DV) / atlas.res).astype(int), 0, sec.shape[0] - 1)
    mi = np.clip(np.round((g.MID + sl['lat_mm'] * 1000) / atlas.res).astype(int), 0, sec.shape[1] - 1)
    return sl[(sec[di, mi] > 0)]


def main():
    atlas = Atlas()
    u = g.load_units()
    idx = np.clip(np.round(u[['ccf_atlas_ap', 'ccf_atlas_dv', 'ccf_atlas_ml']].to_numpy(float) / atlas.res).astype(int), 0,
                  np.array(atlas.ann.shape) - 1)
    u = u[atlas.ann[idx[:, 0], idx[:, 1], idx[:, 2]] > 0].copy()
    centres = [('Frontal', 2.0), ('wS1', float(u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ap_mm'].median())),
               ('Midbrain', float(u.loc[u['area_group'] == 'Midbrain', 'ap_mm'].median()))]
    lat_sag = float(u['lat_mm'].median())
    ps.set_nature_rc()
    for variant in 'ABC':
        fig = plt.figure(figsize=(183 * ps.MM, 132 * ps.MM))
        gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1.25, 0.06], wspace=0.08, hspace=0.22,
                              left=0.06, right=0.97, top=0.93, bottom=0.03)
        for i, (rkey, rtitle, cats) in enumerate(g.ROWS):
            (ka, la, ca), (kb, lb, cb) = cats
            vmax = 0.5 if rkey == 'lick' else 0.4
            dmax = 0.3
            for j in range(4):
                ax = fig.add_subplot(gs[i, j])
                if j < 3:
                    name, c = centres[j]
                    sl = slab_units(u, atlas, c)
                    x, y = sl['lat_mm'].to_numpy(), sl['dv_mm'].to_numpy()
                    if variant == 'A':
                        for side, k, col in ((-1, ka, ca), (1, kb, cb)):
                            f = frac_map(x, y, sl[k].to_numpy(), LAT, DV)
                            xe = side * LAT
                            ax.pcolormesh(np.sort(xe) if side > 0 else xe[::-1], DV, (f if side > 0 else f[::-1]).T,
                                          cmap=cmap_to(col), vmin=0, vmax=vmax, shading='flat', rasterized=True)
                    elif variant == 'B':
                        d = frac_map(x, y, sl[ka].to_numpy(), LAT, DV) - frac_map(x, y, sl[kb].to_numpy(), LAT, DV)
                        for side in (-1, 1):
                            ax.pcolormesh(side * LAT if side > 0 else (-LAT)[::-1], DV, (d if side > 0 else d[::-1]).T,
                                          cmap=diverging(cb, ca), vmin=-dmax, vmax=dmax, shading='flat', rasterized=True)
                    else:
                        a = sl.groupby('area_acronym_custom').agg(n=('lat_mm', 'size'), lat=('lat_mm', 'median'), dv=('dv_mm', 'median'),
                                                                  fa=(ka, 'mean'), fb=(kb, 'mean')).reset_index()
                        a = a[a['n'] >= 30]
                        for side, fcol, col in ((-1, 'fa', ca), (1, 'fb', cb)):
                            ax.scatter(side * a['lat'], a['dv'], s=np.sqrt(a['n']) * 0.9, c=a[fcol], cmap=cmap_to(col), vmin=0,
                                       vmax=vmax, edgecolors='k', linewidths=0.2, zorder=7)
                    atlas.coronal(ax, c)
                    ax.axvline(0, color='#bbbbbb', lw=0.3, ls=':', zorder=6)
                    if i == 0:
                        ax.set_title(f'{name}: AP {c:+.1f} mm', pad=2)
                    if variant != 'B':
                        ax.text(-5.8, -0.6, la, color=ca, fontsize=5, ha='left'); ax.text(5.8, -0.6, lb, color=cb, fontsize=5, ha='right')
                    else:
                        ax.text(0, -0.6, f'{la} - {lb}', fontsize=5, ha='center')
                    style(ax, True)
                else:
                    x, y = u['ap_mm'].to_numpy(), u['dv_mm'].to_numpy()
                    if variant in 'AB':
                        d = frac_map(x, y, u[ka].to_numpy(), AP, DV) - frac_map(x, y, u[kb].to_numpy(), AP, DV)
                        im = ax.pcolormesh(AP, DV, d.T, cmap=diverging(cb, ca), vmin=-dmax, vmax=dmax, shading='flat', rasterized=True)
                    else:
                        a = u.groupby('area_acronym_custom').agg(n=('ap_mm', 'size'), ap=('ap_mm', 'median'), dv=('dv_mm', 'median'),
                                                                 fa=(ka, 'mean'), fb=(kb, 'mean')).reset_index()
                        a = a[a['n'] >= 100]
                        im = ax.scatter(a['ap'], a['dv'], s=np.sqrt(a['n']) * 0.6, c=a['fa'] - a['fb'], cmap=diverging(cb, ca),
                                        vmin=-dmax, vmax=dmax, edgecolors='k', linewidths=0.2, zorder=7)
                    atlas.sagittal(ax, lat_sag)
                    if i == 0:
                        ax.set_title('Sagittal: all units', pad=2)
                    ax.text(-1.6, -0.6, f'{la} - {lb}', fontsize=5, ha='center')
                    style(ax, False)
                    cax = fig.add_subplot(gs[i, 4])
                    cb_ = fig.colorbar(im, cax=cax); cb_.set_ticks([-dmax, 0, dmax]); cb_.ax.tick_params(labelsize=5)
                    cb_.set_label('fraction difference', fontsize=5)
                if j == 0:
                    ax.text(-0.04, 0.5, rtitle, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
        desc = {'A': f'fraction of units significant per {BIN} mm bin (smoothed, >= {MIN_N} units); colour scale 0 to 0.4 (licks 0.5)',
                'B': 'preference: fraction(first) - fraction(second) per bin, both hemispheres pooled',
                'C': 'areas: circle at the median unit position, size = units, colour = fraction significant (sagittal: difference)'}[variant]
        fig.suptitle(f'Variant {variant}: {desc}; {"all days" if g.ALL_DAYS else "day 0"}, R+ and R- pooled', fontsize=6, y=0.985)
        ps.save_figure(fig, g.OUT, f'Figure2_explore_variant_{variant}', raster_dpi=600)
        plt.close(fig)
        print('saved variant', variant)


if __name__ == '__main__':
    main()
