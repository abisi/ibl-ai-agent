"""Figure 2 exploration in the style of Chen et al. 2024 (Cell), Fig. 4 / Fig. S4 ("encoding of choice in ALM projection
zones"), with the whisker- and auditory-cortex projections of this project (002: Allen anterograde tracing, SSp-bfd + SSs
= whisker, AUDp + AUDd/v = auditory, line-balanced, smoothed 50-um density, right hemisphere).

Figure A (Figure2_explore_projection_slabs): one row per target (wS1 / wS2 = SSp-bfd + SSs, SC = SCs + SCm, wM1 = MO-wM1), one coronal
slab of SLAB um each, centred on the AP where most recorded target neurons lie inside the union of the two 90 % zones.
Columns: recorded neurons (target neurons dark, others light) | whisker projection density (slab mean, normalised to the
99.9th percentile of the brain; dashed = 90 % zone, mass rank <= 0.9) | whisker-responsive neurons (colour = |ROC
selectivity|, light -> whisker colour) | proportion of whisker-responsive neurons (003 machinery: 3-D Gaussian sigma
150 um, smoothed responsive count / smoothed recorded count, slab mean, >= 3 neurons within the kernel) | same three for
auditory. Target structure outlined in black.
Figure B (Figure2_explore_projection_quartiles): proportion of responsive neurons as a function of projection density,
per area group (Chen Fig. 4C): neurons of the group divided into quartiles (Q1-Q4) of the projection density at their
50-um voxel; whisker projection (yellow) and auditory projection (blue); rows = whisker-responsive, auditory-
responsive, spontaneous-lick-modulated (enhanced or suppressed). Shading = Wilson 95 % CI (binomial).
p = Cochran-Armitage trend test across quartiles. Inside SSp-bfd / SSs the whisker projection is the input from the
other whisker source (SSs -> SSp-bfd, SSp-bfd -> SSs; 'whisker_x'); auditory-source neurons are excluded for auditory.
Units: good + mua, day 0, R+ and R- pooled (explore_ccf_maps.load_units), ML folded onto the right hemisphere.
Run on haas (ibl-ai-agent .venv): python explore_projection_selectivity.py
Outputs: combined_results_ks4/_sensory_spatial_maps/figure2_maps/ (figures + projection_quartiles.csv)."""
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
from scipy import ndimage, stats  # noqa: E402

import explore_sparse as es  # noqa: E402
bv, cm, m3, ps = es.bv, es.cm, es.m3, es.ps

SLAB = 400.0
m3.SLAB = SLAB
ZONE_LEVEL = 0.9
# groups = area_acronym_custom sets (user 2026-11-17: focus on wS1 / wS2, SC / midbrain, frontal wM1)
TARGETS = [('wS1 / wS2', ['SSp-bfd', 'SSs']), ('SC', ['SCs', 'SCm']), ('wM1', ['MO-wM1'])]
QGROUPS = [('SSp-bfd', ['SSp-bfd']), ('SSs', ['SSs']), ('SCs', ['SCs']), ('SCm', ['SCm']), ('MRN', ['MRN']),
           ('wM1', ['MO-wM1'])]
WS = ['SSp-bfd', 'SSs']        # whisker sources: inside them the whisker projection = input from the other source
RESP = [('sig_whisker_active', 'Whisker-responsive'), ('sig_auditory_active', 'Auditory-responsive'),
        ('sig_spontaneous_licks', 'Lick-modulated')]
MOD = {'whisker': cm.C_W, 'auditory': cm.C_A}


def white_to(c):
    return mcolors.LinearSegmentedColormap.from_list('w', ['#ffffff', c])


def mass_rank(d):
    f = d.ravel().astype(float)
    o = np.argsort(-f, kind='stable')
    cs = np.cumsum(f[o]) / f.sum()
    r = np.empty_like(cs)
    r[o] = cs
    r[f <= 0] = 1.0
    return r.reshape(d.shape)


def wilson(k, n, z=1.96):
    p = k / np.maximum(n, 1)
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return p, c - h, c + h


def cochran_armitage(k, n):
    k, n = np.asarray(k, float), np.asarray(n, float)
    t = np.arange(len(k), dtype=float)
    N, K = n.sum(), k.sum()
    p = K / N
    T = np.sum(t * (k - n * p))
    V = p * (1 - p) * (np.sum(n * t ** 2) - np.sum(n * t) ** 2 / N)
    z = T / np.sqrt(V) if V > 0 else np.nan
    return z, 2 * stats.norm.sf(abs(z))


def load():
    A = m3.Atlas(load_zones=False)
    u = cm.load_units()
    u = u[A.unit_ids(u) != 0].copy()
    Z = np.load(m3.OUT / 'projection_zones.npz')
    P = {}
    for mod in MOD:
        d = Z[f'density_{mod}'].astype(float)
        P[mod] = dict(d=d / np.percentile(d[d > 0], 99.9), r=mass_rank(d), src=Z[f'source_{mod}'])
    # cross-projection inside the whisker sources: SSp-bfd -> SSs and SSs -> SSp-bfd (each source's own area is 0)
    dx = sum(np.where(Z[f'source_{s_}'], 0, Z[f'density_{s_}'].astype(float)) / np.percentile(Z[f'density_{s_}'][Z[f'density_{s_}'] > 0], 99.9)
             for s_ in WS) / len(WS)
    P['whisker_x'] = dict(d=dx / np.percentile(dx[dx > 0], 99.9), r=mass_rank(dx), src=np.zeros(dx.shape, bool))
    ijk = np.floor(np.c_[u.ccf_atlas_ap, u.ccf_atlas_dv, u.ml_f].astype(float) / 50).astype(int)
    ijk = np.clip(ijk, 0, np.array(P['whisker']['d'].shape) - 1)
    for mod in MOD:
        u[f'proj_{mod}'] = P[mod]['d'][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        u[f'rank_{mod}'] = P[mod]['r'][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        u[f'src_{mod}'] = P[mod]['src'][ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    ws = u['area_acronym_custom'].isin(WS).to_numpy()
    for f, key in (('proj', 'd'), ('rank', 'r')):
        u.loc[ws, f'{f}_whisker'] = P['whisker_x'][key][ijk[ws, 0], ijk[ws, 1], ijk[ws, 2]]
    u.loc[ws, 'src_whisker'] = False
    return A, u, P


def slab_proj(P, mod, c):
    i, h, mid = int(round(c / 50)), int(round(SLAB / 2 / 50)), int(m3.MID / 50)
    d = P[mod]['d'][max(i - h, 0):i + h].mean(0)[:, mid:]
    z = (P[mod]['r'][max(i - h, 0):i + h] <= ZONE_LEVEL).any(0)[:, mid:]
    return d, z


def target_mask(A, lab, group, u_slab):
    """structures (merged ids) holding the group's neurons in this slab -> mask of the section"""
    ids = np.unique(A.unit_ids(u_slab[u_slab['area_acronym_custom'].isin(group)]))
    return np.isin(lab, ids[ids != 0])


def contour(ax, m, res_mm, **kw):
    if m.any():
        yy = (np.arange(m.shape[0]) + 0.5) * res_mm
        xx = (np.arange(m.shape[1]) + 0.5) * res_mm
        ax.contour(xx, yy, ndimage.gaussian_filter(m.astype(float), 0.8), levels=[0.5], **kw)


def figure_slabs(A, u, P):
    vols = {}
    for k in ('sig_whisker_active', 'sig_auditory_active'):
        vols[k] = m3.density_volumes(u, u[k].astype(float).to_numpy())
    rows = []
    for name, group in TARGETS:
        g = u[u['area_acronym_custom'].isin(group)]
        aps = np.arange(g['ccf_atlas_ap'].quantile(0.02), g['ccf_atlas_ap'].quantile(0.98), 50.0)
        inz = g[(g['rank_whisker'] <= ZONE_LEVEL) | (g['rank_auditory'] <= ZONE_LEVEL)]
        c = aps[np.argmax([((inz['ccf_atlas_ap'] - a).abs() <= SLAB / 2).sum() for a in aps])]
        rows.append((name, group, c))
        print(name, 'AP', round((m3.BREGMA_AP - c) / 1000, 2), flush=True)
    ncol = 7
    fig = plt.figure(figsize=(183 * ps.MM, 3 * 34 * ps.MM))
    gs = fig.add_gridspec(3, ncol, wspace=0.04, hspace=0.18, left=0.04, right=0.995, top=0.85, bottom=0.03)
    vmax = {'proj': 1.0, 'frac': 0.5}
    for i, (name, group, c) in enumerate(rows):
        lab = A.section('cor', c)
        ext = (0, lab.shape[1] / 100, lab.shape[0] / 100, 0)
        sl = u[(u['ccf_atlas_ap'] - c).abs() <= SLAB / 2]
        sl = sl[es.section_inside(lab, sl['lat_mm'].to_numpy(), sl['dv_mm'].to_numpy())]
        tm = target_mask(A, lab, group, sl)
        isg = sl['area_acronym_custom'].isin(group).to_numpy()
        gs_ = sl[isg]
        pad = 0.6
        xl = (max(0, gs_['lat_mm'].quantile(0.01) - pad), gs_['lat_mm'].quantile(0.99) + pad)
        yl = (gs_['dv_mm'].quantile(0.99) + pad, max(0, gs_['dv_mm'].quantile(0.01) - pad))
        panels = ['recorded', ('proj', 'whisker'), ('resp', 'whisker'), ('frac', 'whisker'),
                  ('proj', 'auditory'), ('resp', 'auditory'), ('frac', 'auditory')]
        for j, p in enumerate(panels):
            ax = fig.add_subplot(gs[i, j])
            inside = ndimage.binary_fill_holes(lab != 0)
            bg = np.zeros(lab.shape + (4,)); bg[..., :3] = 1; bg[..., 3] = inside
            ax.imshow(bg, extent=ext, zorder=0)
            if p == 'recorded':
                ax.scatter(sl['lat_mm'][~isg], sl['dv_mm'][~isg], s=0.08, c='#c8c8c8', lw=0, rasterized=True, zorder=5)
                ax.scatter(gs_['lat_mm'], gs_['dv_mm'], s=0.12, c='#303030', lw=0, rasterized=True, zorder=5)
                ttl = f'Recorded (n = {len(gs_)})'
            else:
                kind, mod = p
                col = MOD[mod]
                d, z = slab_proj(P, 'whisker_x' if (mod == 'whisker' and group == TARGETS[0][1]) else mod, c)
                if kind == 'proj':
                    ax.imshow(np.where(inside[::5, ::5][:d.shape[0], :d.shape[1]], d, np.nan), extent=(0, d.shape[1] * 0.05, d.shape[0] * 0.05, 0),
                              cmap=white_to(col), vmin=0, vmax=vmax['proj'], interpolation='bilinear', zorder=1)
                    ttl = f'{mod.capitalize()} cortex\nprojection density'
                elif kind == 'resp':
                    k = f'sig_{mod}_active'
                    m = sl[k].to_numpy(bool)
                    ax.scatter(sl['lat_mm'][~m], sl['dv_mm'][~m], s=0.06, c=es.C_NR, alpha=0.3, lw=0, rasterized=True, zorder=5)
                    sv = sl[f'sel_{mod}_active'].abs().to_numpy()[m]
                    o = np.argsort(sv)
                    ax.scatter(sl['lat_mm'].to_numpy()[m][o], sl['dv_mm'].to_numpy()[m][o], s=0.9,
                               c=es.sel_cmap(col)(np.clip((sv[o] - es.SMIN) / (es.SMAX - es.SMIN), 0, 1)), lw=0,
                               rasterized=True, zorder=6)
                    nt = int(sl[k].to_numpy(bool)[isg].sum())
                    ttl = f'{mod.capitalize()}-responsive\n({nt} / {len(gs_)} in target)'
                else:
                    num, den = vols[f'sig_{mod}_active']
                    M, ext50 = m3.slab_ratio(num, den, 'cor', c)
                    ax.imshow(M, extent=ext50, cmap=white_to(col), vmin=0, vmax=vmax['frac'], interpolation='bilinear', zorder=1)
                    ttl = f'Proportion\n{mod}-responsive'
                contour(ax, z, 0.05, colors='k', linewidths=0.6, linestyles=[(0, (2.5, 1.5))], zorder=8)
            bv.draw_bounds(ax, A, lab, ext)
            contour(ax, tm[::2, ::2], 0.02, colors='#000000', linewidths=0.45, zorder=7)
            ax.set_xlim(*xl); ax.set_ylim(*yl)
            cm.axes_off(ax)
            ax.set_title(ttl, fontsize=4.8, pad=1)
            if j == 0:
                ax.text(-0.06, 0.5, f'{name}\nAP {(m3.BREGMA_AP - c) / 1000:+.1f}', transform=ax.transAxes, rotation=90,
                        ha='right', va='center', fontsize=5.5)
                if i == 2:
                    x0, y0 = xl[0] + 0.1, yl[0] - 0.1
                    ax.plot([x0, x0 + 1], [y0, y0], color='k', lw=0.6)
                    ax.text(x0 + 0.5, y0 - 0.05, '1 mm', ha='center', va='bottom', fontsize=4)
    # colour keys
    for jj, (mod, kind, lab_) in enumerate([('whisker', 'proj', 'projection (norm.)'), ('whisker', 'frac', 'proportion'),
                                            ('auditory', 'proj', 'projection (norm.)'), ('auditory', 'frac', 'proportion')]):
        cax = fig.add_axes([0.06 + jj * 0.2, 0.935, 0.08, 0.008])
        cb_ = fig.colorbar(plt.cm.ScalarMappable(norm=mcolors.Normalize(0, vmax[kind]), cmap=white_to(MOD[mod])),
                           cax=cax, orientation='horizontal')
        cb_.ax.tick_params(labelsize=4, length=1.2, pad=0.8); cb_.outline.set_linewidth(0.3)
        cb_.ax.xaxis.set_ticks_position('top'); cb_.set_label(f'{mod} {lab_}', fontsize=4.3, labelpad=1)
        cb_.ax.xaxis.set_label_position('top')
    fig.text(0.995, 0.985, f'dashed: 90 % projection zone; black: target structures; {SLAB / 1000:g}-mm coronal slabs; day 0',
             ha='right', fontsize=4.3, color='0.3')
    ps.save_figure(fig, cm.OUT, 'Figure2_explore_projection_slabs', raster_dpi=600)
    plt.close(fig)
    print('saved slabs', flush=True)


def figure_quartiles(u):
    rec = []
    fig, axs = plt.subplots(len(RESP), len(QGROUPS), figsize=(183 * ps.MM, 85 * ps.MM), sharex=True)
    for i, (k, rl) in enumerate(RESP):
        for j, (gname, group) in enumerate(QGROUPS):
            ax = axs[i, j]
            g = u[u['area_acronym_custom'].isin(group)]
            for mod, col in MOD.items():
                h = g[~g[f'src_{mod}']]
                if len(h) < 40:
                    continue
                q = pd.qcut(h[f'proj_{mod}'].rank(method='first'), 4, labels=False)
                kk = h.groupby(q)[k].sum().to_numpy(float)
                nn = h.groupby(q)[k].size().to_numpy(float)
                p, lo, hi = wilson(kk, nn)
                z, pv = cochran_armitage(kk, nn)
                x = np.arange(1, 5) + (-0.06 if mod == 'whisker' else 0.06)
                ax.fill_between(x, lo, hi, color=col, alpha=0.18, lw=0)
                ax.plot(x, p, '-o', color=col, ms=1.8, lw=0.8)
                star = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else 'n.s.'
                ax.text(0.03 if mod == 'whisker' else 0.97, 0.97, star, transform=ax.transAxes, color=col, fontsize=4.5,
                        ha='left' if mod == 'whisker' else 'right', va='top')
                for qi in range(4):
                    rec.append(dict(response=k, area_group=gname, projection=mod, quartile=qi + 1, n=int(nn[qi]),
                                    k=int(kk[qi]), prop=p[qi], ci_lo=lo[qi], ci_hi=hi[qi], trend_z=z, trend_p=pv))
            ax.set_xticks([1, 2, 3, 4], ['Q1', 'Q2', 'Q3', 'Q4'])
            ax.tick_params(labelsize=4.5, length=1.5, pad=1)
            ax.set_ylim(bottom=0)
            for s_ in ('top', 'right'):
                ax.spines[s_].set_visible(False)
            if i == 0:
                ax.set_title(gname, fontsize=5)
            if j == 0:
                ax.set_ylabel(f'{rl}\n(proportion)', fontsize=5)
            if i == len(RESP) - 1:
                ax.set_xlabel('projection-density quartile', fontsize=4.5)
    fig.text(0.5, 0.995, 'Proportion of responsive neurons vs cortical projection density at the neuron '
             '(yellow: whisker cortex, blue: auditory cortex; Wilson 95 % CI; Cochran-Armitage trend)',
             ha='center', va='top', fontsize=5)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.6, w_pad=0.4)
    ps.save_figure(fig, cm.OUT, 'Figure2_explore_projection_quartiles')
    plt.close(fig)
    pd.DataFrame(rec).to_csv(cm.OUT / 'projection_quartiles.csv', index=False)
    print('saved quartiles', flush=True)


def main():
    A, u, P = load()
    ps.set_nature_rc()
    figure_quartiles(u)
    figure_slabs(A, u, P)


if __name__ == '__main__':
    main()
