"""Figure 2 exploration: thin slabs, zoomed to the recorded territory, sparse signed dots over a light preference map.
Rows (ROWS): whisker responsive, auditory responsive (active trials), spontaneous licks enhanced | suppressed. Neurons folded onto one hemisphere.
  background: density of recorded neurons (neurons / mm^2, 2-D Gaussian sigma 150 um, log grey scale); white = < N_MIN
              neurons within the kernel (not sampled)
  dots:       significant neurons only, coloured A / B (dark grey = both), size steeply increasing with |ROC selectivity|
              (dot_size), alpha ALPHA, random drawing order (no colour hides the other)
Figure 1 (grid, 105 mm wide = half A4): 3 coronal slabs of SLAB um (frontal AP +2.0, wS1 = median AP of SSp-bfd neurons, midbrain = median AP of
Midbrain neurons) + a sagittal slab of SLAB um at the most densely sampled ML band. Figure 2: coronal AP series.
Units: good + mua, day 0, R+ and R- pooled; neurons outside the brain in the slab-centre section removed.
Run on haas (ibl-ai-agent .venv): python explore_sparse.py"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import explore_bivariate as bv  # noqa: E402
cm, m3, ps = bv.cm, bv.m3, bv.ps

SLAB = 400.0
DMIN, DMAX = 35.0, 4000.0                          # recorded-neuron density colour range (neurons / mm^2, log)
DENS_CMAP = mcolors.LinearSegmentedColormap.from_list('dens', ['#fafafa', '#cfcfcf'])
C_BOTH = '#3a3a3a'
ALPHA = 0.45
ROWS = [('whisker', 'Whisker', [('sig_whisker_active', 'sel_whisker_active', 'Responsive', cm.C_W)]),
        ('auditory', 'Auditory', [('sig_auditory_active', 'sel_auditory_active', 'Responsive', cm.C_A)]),
        ('lick', 'Spontaneous licks', [('lick_enh', 'sel_spontaneous_licks', 'Enhanced', cm.C_EN),
                                       ('lick_sup', 'sel_spontaneous_licks', 'Suppressed', cm.C_SU)])]
RNG = np.random.default_rng(0)


SMIN, SMAX = 0.05, 0.6                              # |selectivity| colour range
S_RESP, S_NR, C_NR, A_NR = 1.6, 0.5, '#3c3c3c', 0.4   # dot sizes (x scale), non-responsive colour / alpha


def sel_cmap(c):
    c = np.array(mcolors.to_rgb(c))
    return mcolors.LinearSegmentedColormap.from_list('s', [1 - 0.15 * (1 - c), c])


def dot_size(sel):
    return 0.03 + 22 * np.nan_to_num(sel) ** 2.2


def density(x, y, xr, yr):
    """recorded neurons per mm^2 (2-D Gaussian, sigma SIGMA_UM, explore_bivariate grid); NaN below N_MIN per kernel"""
    bx = np.arange(xr[0], xr[1] + 1e-9, bv.BIN_UM / 1000)
    by = np.arange(yr[0], yr[1] + 1e-9, bv.BIN_UM / 1000)
    sg = bv.SIGMA_UM / bv.BIN_UM
    den = bv.ndimage.gaussian_filter(np.histogram2d(y, x, bins=[by, bx])[0], sg)
    d = den / (bv.BIN_UM / 1000) ** 2
    d[den * 2 * np.pi * sg ** 2 < bv.N_MIN] = np.nan
    return d, (bx[0], bx[-1], by[-1], by[0])


def dots(ax, x, y, df, row, scale):
    """non-responsive neurons small dark grey; significant neurons per category coloured by |ROC selectivity|
    (light = weak, dark = strong), drawn in increasing |selectivity| so the strongest are on top"""
    m_all, counts, pts = np.zeros(len(df), bool), [], []
    for k, ks, _, c in row[2]:
        m = df[k].to_numpy(bool)
        m_all |= m; counts.append(int(m.sum()))
        sv = df[ks].abs().to_numpy()[m]
        rgba = sel_cmap(c)(np.clip((sv - SMIN) / (SMAX - SMIN), 0, 1))
        pts.append(np.c_[x[m], y[m], sv, rgba])
    ax.scatter(x[~m_all], y[~m_all], s=S_NR * scale, c=C_NR, lw=0, alpha=A_NR, zorder=5, rasterized=True)
    P = np.concatenate(pts)
    P = P[np.argsort(P[:, 2])]
    ax.scatter(P[:, 0], P[:, 1], s=S_RESP * scale, c=P[:, 3:7], lw=0, alpha=0.9, zorder=6, rasterized=True)
    return counts


ZONE_LEVEL = 0.9                                   # 002 projection zone (mass rank <= 0.9, line-balanced Allen tracing)
_ZONES = {}


def zones():
    """90 % projection zones of whisker (SSp-bfd + SSs) and auditory (AUDp + AUDd/v) cortex, 50-um CCF grid"""
    if not _ZONES:
        Z = np.load(m3.OUT / 'projection_zones.npz')
        for mod in ('whisker', 'auditory'):
            d = Z[f'density_{mod}'].astype(float).ravel()
            o = np.argsort(-d, kind='stable')
            r = np.empty_like(d); r[o] = np.cumsum(d[o]) / d.sum(); r[d <= 0] = 1.0
            _ZONES[mod] = r.reshape(Z[f'density_{mod}'].shape) <= ZONE_LEVEL
    return _ZONES


def draw_zone(ax, mod, plane, c):
    z, h, i, mid = zones()[mod], int(round(SLAB / 2 / 50)), int(round(c / 50)), int(m3.MID / 50)
    m = z[max(i - h, 0):i + h].any(0)[:, mid:] if plane == 'cor' else z[:, :, max(i - h, 0):i + h].any(2).T
    if m.any():
        yy, xx = (np.arange(m.shape[0]) + 0.5) * 0.05, (np.arange(m.shape[1]) + 0.5) * 0.05
        ax.contour(xx, yy, bv.ndimage.gaussian_filter(m.astype(float), 0.8), levels=[0.5], colors='#1a1a1a',
                   linewidths=0.45, linestyles=[(0, (2.2, 1.3))], zorder=7)


def label_structures(ax, A, sl, xcol, n_max=4, min_n=150):
    ids = A.unit_ids(sl)
    vc = pd.Series(ids[ids != 0]).value_counts()
    for k in vc[vc >= min_n].index[:n_max]:
        m = ids == k
        ax.text(np.median(sl[xcol].to_numpy()[m]), np.median(sl['dv_mm'].to_numpy()[m]), A.acr.get(int(k), ''), fontsize=5,
                color='k', ha='center', va='center', zorder=9)          # plain text (svg.fonttype none: editable)


# expected false-positive fraction per category: each ROC test is a 1000-permutation test, p < 0.05 in the tail of the
# neuron's selectivity sign (unit_spikes_analysis roc_utils_new, uncorrected); 'responsive' counts both signs -> ~10 %,
# lick enhanced / suppressed one sign each -> ~5 %
CHANCE = {'whisker': (0.10,), 'auditory': (0.10,), 'lick': (0.05, 0.05)}


def pct_text(k, n, chance):
    """percentage; '(n.s.)' when not above chance (one-sided binomial test, p >= 0.001)"""
    return f'{100 * k / max(n, 1):.0f} %'


def section_inside(lab, x_mm, y_mm):
    yi = np.clip(np.round(y_mm * 100).astype(int), 0, lab.shape[0] - 1)
    xi = np.clip(np.round(x_mm * 100).astype(int), 0, lab.shape[1] - 1)
    return lab[yi, xi] > 0


def panel(ax, A, df, plane, c, row, lims, scale=1.0, labels=False):
    """plane 'cor' (x = lateral, c = ccf AP) or 'sag' (x = AP, c = ccf ML); returns counts"""
    lab = A.section(plane, c)
    ext_lab = (0, lab.shape[1] / 100, lab.shape[0] / 100, 0)
    xcol = 'lat_mm' if plane == 'cor' else 'ap_mm'
    pos = df['ccf_atlas_ap'] if plane == 'cor' else df['ml_f']
    sl = df[(pos - c).abs() <= SLAB / 2]
    sl = sl[section_inside(lab, sl[xcol].to_numpy(), sl['dv_mm'].to_numpy())]
    x, y = sl[xcol].to_numpy(), sl['dv_mm'].to_numpy()
    d, ext = density(x, y, (0, ext_lab[1]), (0, ext_lab[2]))
    inside = bv.inside_mask(lab, ext, d.shape)
    d[~inside] = np.nan
    ax.imshow(d, extent=ext, cmap=DENS_CMAP, norm=mcolors.LogNorm(DMIN, DMAX), interpolation='bilinear', zorder=1)
    if np.isfinite(d).any():
        print('max density', int(np.nanmax(d)), flush=True)
    bv.draw_bounds(ax, A, lab, ext_lab)
    n = dots(ax, x, y, sl, row, scale)
    if labels:
        label_structures(ax, A, sl, xcol)
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    cm.axes_off(ax)
    return len(sl), n


def zoom(x, y, pad=0.35):
    return ((0.0 if x.name == 'lat_mm' else max(0, np.percentile(x, 0.5) - pad), np.percentile(x, 99.5) + pad),
            (np.percentile(y, 99.5) + pad, max(0.0, np.percentile(y, 0.5) - pad)))


def legend(ax, row, scale, fs=4.5):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    g = np.linspace(0, 1, 64)[None, :]
    y = 0.92
    for _, _, t, c in row[2]:
        ax.text(0.02, y, t, fontsize=fs, va='center')
        ax.imshow(g, cmap=sel_cmap(c), extent=(0.05, 0.85, y - 0.17, y - 0.09), aspect='auto')
        ax.text(0.05, y - 0.2, f'{SMIN:g}', fontsize=fs - 0.5, va='top', ha='center')
        ax.text(0.85, y - 0.2, f'{SMAX:g}', fontsize=fs - 0.5, va='top', ha='center')
        y -= 0.36
    ax.text(0.45, y + 0.07, '|selectivity|', fontsize=fs - 0.5, ha='center', va='center')
    ax.scatter(0.08, y - 0.06, s=S_NR * scale * 4, c=C_NR, lw=0, alpha=0.8)
    ax.text(0.17, y - 0.06, 'not responsive', fontsize=fs, va='center')


def density_bar(fig, rect, fs=4.5):
    cax = fig.add_axes(rect)
    cb_ = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.LogNorm(DMIN, DMAX), cmap=DENS_CMAP), cax=cax, orientation='horizontal')
    cb_.set_ticks([100, 1000]); cb_.set_ticklabels(['100', '1000']); cb_.minorticks_off()
    cb_.ax.tick_params(labelsize=fs - 0.5, length=1.5, pad=1); cb_.outline.set_linewidth(0.3)
    cb_.set_label('recorded neurons\nper mm$^2$', fontsize=fs, labelpad=1)


def main():
    A = m3.Atlas(load_zones=False)
    u = cm.load_units()
    u = u[A.unit_ids(u) != 0].copy()
    centres = [('Frontal', m3.BREGMA_AP - 2000.0),
               ('wS1', float(u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ccf_atlas_ap'].median())),
               ('Midbrain', float(u.loc[u['area_group'] == 'Midbrain', 'ccf_atlas_ap'].median()))]
    # sagittal through wS1: median folded ML of the SSp-bfd neurons (100-um grid)
    ml_c = m3.MID + round((u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ml_f'].median() - m3.MID) / 100) * 100
    print('sagittal ML', (ml_c - m3.MID) / 1000, flush=True)
    ps.set_nature_rc()

    # ---- grid
    lims_c, lims_s = [], None
    for _, c in centres:
        s = u[(u['ccf_atlas_ap'] - c).abs() <= SLAB / 2]
        lims_c.append(zoom(s['lat_mm'], s['dv_mm']))
    s = u[(u['ml_f'] - ml_c).abs() <= SLAB / 2]
    lims_s = zoom(s['ap_mm'], s['dv_mm'])
    # half-A4 width (105 mm), panels placed in mm so every panel has the same mm-per-brain-mm scale
    W, L, R, LEG, GAP, TOP, RGAP, BOT = 105.0, 4.0, 0.5, 18.0, 1.0, 8.0, 5.0, 1.0
    lims = lims_c + [lims_s]
    wd = [l[0][1] - l[0][0] for l in lims]
    hd = [l[1][0] - l[1][1] for l in lims]
    k = (W - L - R - LEG - GAP * 4) / sum(wd)                     # figure mm per brain mm
    rowh = max(hd) * k
    H = TOP + 3 * rowh + 3 * RGAP + BOT
    fig = plt.figure(figsize=(W * ps.MM, H * ps.MM))
    DS, FS = 0.4, 5.5                                             # dot-size scale, font size
    rec = []
    for i, row in enumerate(ROWS):
        y_top = H - TOP - i * (rowh + RGAP)
        x = L
        for j in range(4):
            w, h = wd[j] * k, hd[j] * k
            ax = fig.add_axes([x / W, (y_top - h) / H, w / W, h / H])
            if j < 3:
                name, c = centres[j]
                n, cnt = panel(ax, A, u, 'cor', c, row, lims[j], scale=DS, labels=False)
                ttl = f'{name}\nAP {(m3.BREGMA_AP - c) / 1000:+.1f} mm'
            else:
                n, cnt = panel(ax, A, u, 'sag', ml_c, row, lims[j], scale=DS, labels=False)
                ttl = f'Sagittal (wS1)\nML {(ml_c - m3.MID) / 1000:.1f} mm'
            if i == 0:
                ax.set_title(ttl, pad=1, fontsize=FS)
            ax.text(1.0, -0.01, ' | '.join(pct_text(k_, n, ch) for k_, ch in zip(cnt, CHANCE[row[0]])), transform=ax.transAxes,
                    ha='right', va='top', fontsize=5, color='0.25')
            rec.append(dict(row=row[0], panel=ttl.replace('\n', ' '), n=n, **{f'k_{t}': k_ for (_, _, t, _), k_ in zip(row[2], cnt)}))
            if j == 0:
                ax.text(-0.03, 0.5, row[1], transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=FS)
                if i == 2:
                    x0 = lims[0][0][0] + 0.1; y0 = lims[0][1][0] - 0.1
                    ax.plot([x0, x0 + 1], [y0, y0], color='k', lw=0.6)
                    ax.text(x0 + 0.5, y0 - 0.06, '1 mm', ha='center', va='bottom', fontsize=5)
            x += w + GAP
        lax = fig.add_axes([(W - R - LEG + 0.5) / W, (y_top - rowh * 0.75) / H, (LEG - 0.5) / W, rowh * 0.75 / H])
        legend(lax, row, DS, fs=5.5)
    density_bar(fig, [(W - R - LEG + 1.5) / W, (BOT + 8.5) / H, (LEG - 3.5) / W, 1.2 / H], fs=5.5)
    pd.DataFrame(rec).to_csv(cm.OUT / 'Figure2_explore_sparse_grid_counts.csv', index=False)
    ps.save_figure(fig, cm.OUT, 'Figure2_explore_sparse_grid', raster_dpi=600)
    plt.close(fig)
    print('saved grid', W, 'x', round(H, 1), 'mm', flush=True)

    # ---- AP series (common zoom, full width)
    aps = np.arange(2.8, -3.61, -0.8)
    s = u[((m3.BREGMA_AP - u['ccf_atlas_ap']) / 1000).between(aps[-1] - 0.2, aps[0] + 0.2)]
    lims = zoom(s['lat_mm'], s['dv_mm'])
    fig = plt.figure(figsize=(183 * ps.MM, 95 * ps.MM))
    gs = fig.add_gridspec(3, len(aps) + 1, width_ratios=[1] * len(aps) + [1.4], wspace=0.03, hspace=0.06,
                          left=0.035, right=0.995, top=0.9, bottom=0.02)
    for i, row in enumerate(ROWS):
        for j, a_ in enumerate(aps):
            ax = fig.add_subplot(gs[i, j])
            n, _ = panel(ax, A, u, 'cor', m3.BREGMA_AP - a_ * 1000, row, lims, scale=0.5)
            if i == 0:
                ax.set_title(f'AP {a_:+.1f}', pad=1)
            ax.text(0.97, -0.01, f'{n}', transform=ax.transAxes, ha='right', va='top', fontsize=4, color='0.3')
            if j == 0:
                ax.text(-0.05, 0.5, row[1], transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
        lax = fig.add_subplot(gs[i, len(aps)]); legend(lax, row, 0.5)
    density_bar(fig, [0.905, 0.04, 0.08, 0.012])
    fig.suptitle(f'AP series of {SLAB / 1000:g}-mm coronal slabs every 0.8 mm (one hemisphere, same zoom); numbers = neurons per slab',
                 fontsize=6, y=0.985)
    ps.save_figure(fig, cm.OUT, 'Figure2_explore_sparse_APseries', raster_dpi=600)
    plt.close(fig)
    print('saved AP series', flush=True)


if __name__ == '__main__':
    main()
