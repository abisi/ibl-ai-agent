"""Figure 2 exploration: CCF-coordinate maps (no aggregation per area), built on the ssl-sensory-spatial-maps functions
(003_spatial_maps: 10-um barrel atlas, merged-structure boundaries, 3-D Gaussian kernel sigma 150 um on the 50-um CCF
grid, map = smoothed sum / smoothed recorded-neuron count, shown where >= 3 neurons fall within the kernel).
Rows: stimulus (whisker_active | auditory_active), licks (spontaneous_licks enhanced | suppressed), decision
(whisker_decision | auditory_decision; decision = significant M_choice AND significant M_hit_vs_spontaneous).
Coronal slabs 1 mm: frontal (AP +2.0), wS1 (median AP of the SSp-bfd units), midbrain (median AP of the Midbrain
units), mirrored: left half = first category, right half = second (neurons folded onto one hemisphere).
Variants:
  1 dots: significant neurons coloured by their signed ROC selectivity (stimulus: whisker / auditory selectivity; licks:
    lick selectivity; decision: M_choice selectivity), strongest on top; non-significant neurons light grey
  2 fraction: smoothed fraction of recorded neurons that are significant in the category (single-hue scale)
  3 selectivity: smoothed mean signed selectivity of all neurons (diverging; decision row: M_choice selectivity)
  4 projections: maximum projection of the smoothed fraction volume, sagittal (first | second category), horizontal
    and coronal (mirrored)
Units: good + mua, day 0, R+ and R- pooled; neurons outside the brain, or outside the section at the slab centre, removed.
Run on haas (ibl-ai-agent .venv): python explore_ccf_maps.py
Output: Figures/Figure2_sensory_maps_explore_<date>/Figure2_explore_ccf_{1_dots,2_fraction,3_selectivity,4_projections}.*"""
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
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

import paperstyle as ps  # noqa: E402
m3 = importlib.import_module('003_spatial_maps')

m3.SLAB = 1000.0                                        # 1-mm slabs (003 default 500 um)
KEYS = m3.KEYS
OUT = m3.OUT / 'figure2_maps'                           # combined_results_ks4/_sensory_spatial_maps/figure2_maps
C_W, C_A = ps.COLORS['whisker'], ps.COLORS['auditory']
C_EN, C_SU = '#c0392b', '#2471a3'
ATYPES = ['whisker_active', 'auditory_active', 'spontaneous_licks', 'whisker_choice', 'auditory_choice']
# per row: (key, title, [(mask column, value column for dots, label, colour)] x 2, signed column(s) for variant 3)
ROWS = [('stimulus', 'Stimulus response', [('sig_whisker_active', 'sel_whisker_active', 'Whisker', C_W),
                                           ('sig_auditory_active', 'sel_auditory_active', 'Auditory', C_A)],
         ('sel_whisker_active', 'sel_auditory_active')),
        ('lick', 'Spontaneous licks', [('lick_enh', 'sel_spontaneous_licks', 'Enhanced', C_EN),
                                       ('lick_sup', 'sel_spontaneous_licks', 'Suppressed', C_SU)],
         ('sel_spontaneous_licks', 'sel_spontaneous_licks')),
        ('decision', 'Decision', [('whisker_decision', 'sel_whisker_choice', 'Whisker decision', C_W),
                                  ('auditory_decision', 'sel_auditory_choice', 'Auditory decision', C_A)],
         ('sel_whisker_choice', 'sel_auditory_choice'))]
SEL_CMAP = 'RdBu_r'


def to_white(c):
    return LinearSegmentedColormap.from_list('w', ['#ffffff', c])


def load_units():
    u = pd.read_parquet(m3.ROC / 'units.parquet')
    u = u[u['quality_label'].isin(['good', 'mua']) & (u['day'] == 0)].copy()
    u['cluster_id'] = u['cluster_id'].astype(str)
    u = u.drop_duplicates(KEYS)
    r = pd.read_parquet(m3.ROC / 'roc_long.parquet', columns=KEYS + ['analysis_type', 'sel', 'sig'],
                        filters=[('analysis_type', 'in', ATYPES)])
    r['cluster_id'] = r['cluster_id'].astype(str)
    r = r.drop_duplicates(KEYS + ['analysis_type'])
    for at in ATYPES:
        x = r[r['analysis_type'] == at][KEYS + ['sel', 'sig']].rename(columns={'sel': f'sel_{at}', 'sig': f'sig_{at}'})
        u = u.merge(x, on=KEYS, how='left')
    for at in ATYPES:
        u[f'sig_{at}'] = u[f'sig_{at}'] == 1
    u['lick_enh'] = u['sig_spontaneous_licks'] & (u['sel_spontaneous_licks'] > 0)
    u['lick_sup'] = u['sig_spontaneous_licks'] & (u['sel_spontaneous_licks'] < 0)
    for c in ('whisker_decision', 'auditory_decision'):
        u[c] = u[c].fillna(False).astype(bool)
    u['ml_f'] = m3.MID + np.abs(u['ccf_atlas_ml'] - m3.MID)
    u['lat_mm'] = (u['ml_f'] - m3.MID) / 1000
    u['ap_mm'] = u['ccf_atlas_ap'] / 1000
    u['dv_mm'] = u['ccf_atlas_dv'] / 1000
    return u


def norm3d():
    return (2 * np.pi) ** 1.5 * (m3.SIGMA3D / m3.GRID) ** 3


def axes_off(ax):
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def cor_background(ax, A, c):
    lab = A.section('cor', c)
    w, h = lab.shape[1] / 100, lab.shape[0] / 100
    m3.draw_section(ax, A, lab, (0, w, h, 0))
    m3.draw_section(ax, A, lab[:, ::-1], (-w, 0, h, 0))
    ax.axvline(0, color='0.75', lw=0.3, ls=':', zorder=4)
    ax.set_xlim(-5.9, 5.9); ax.set_ylim(8.0, -0.2)
    return lab


def sag_background(ax, A):
    lab = A.section('sag', m3.MID + 1500)
    m3.draw_section(ax, A, lab, (0, lab.shape[1] / 100, lab.shape[0] / 100, 0))
    ax.set_xlim(1.0, 12.4); ax.set_ylim(8.0, -0.2)                     # CCF AP mm, anterior left


def keep_in_section(sl, lab):
    di = np.clip(np.round(sl['dv_mm'].to_numpy() * 100).astype(int), 0, lab.shape[0] - 1)
    li = np.clip(np.round(sl['lat_mm'].to_numpy() * 100).astype(int), 0, lab.shape[1] - 1)
    return sl[lab[di, li] > 0]


def ratio_cor(num, den, c):
    M, ext = m3.slab_ratio(num, den, 'cor', c)
    return M, ext


def ratio_sag_all(num, den):
    n, d = num.sum(2).T, den.sum(2).T
    M = np.where(d * norm3d() >= m3.DENS_MIN, n / np.maximum(d, 1e-12), np.nan)
    return M, (0, M.shape[1] * m3.GRID / 1000, M.shape[0] * m3.GRID / 1000, 0)


def show_mirrored(ax, ML, MR, ext, cmL, cmR, vmin, vmax):
    w = ext[1]
    ax.imshow(MR, extent=(0, w, ext[2], 0), cmap=cmR, vmin=vmin, vmax=vmax, interpolation='bilinear', zorder=1.5)
    return ax.imshow(ML[:, ::-1], extent=(-w, 0, ext[2], 0), cmap=cmL, vmin=vmin, vmax=vmax, interpolation='bilinear', zorder=1.5)


def main():
    A = m3.Atlas(load_zones=False)
    u = load_units()
    ids = A.unit_ids(u)
    print(f'units {len(u)}, outside the brain {(ids == 0).sum()}')
    u = u[ids != 0].copy()
    centres = [('Frontal', m3.BREGMA_AP - 2000.0),
               ('wS1', float(u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ccf_atlas_ap'].median())),
               ('Midbrain', float(u.loc[u['area_group'] == 'Midbrain', 'ccf_atlas_ap'].median()))]
    vols = {}
    for _, _, cats, sels in ROWS:
        for col, _, _, _ in cats:
            vols[col] = m3.density_volumes(u, u[col].astype(float).to_numpy())
        for s in sels:
            if s not in vols:
                vols[s] = m3.density_volumes(u, u[s].to_numpy(float))
    ps.set_nature_rc()
    data_dir = OUT / 'data'; data_dir.mkdir(parents=True, exist_ok=True)
    for vname in ('1_dots', '2_fraction', '3_selectivity', '4_projections'):
        fig = plt.figure(figsize=(183 * ps.MM, 132 * ps.MM))
        gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 1.15, 0.05], wspace=0.06, hspace=0.18,
                              left=0.06, right=0.965, top=0.92, bottom=0.03)
        for i, (rkey, rtitle, cats, sels) in enumerate(ROWS):
            (ka, va, la, ca), (kb, vb, lb, cb) = cats
            fmax = 0.6 if rkey == 'lick' else 0.4
            im = None
            for j in range(4):
                ax = fig.add_subplot(gs[i, j]); axes_off(ax)
                if vname == '4_projections':
                    R = {}
                    for k in (ka, kb):
                        num, den = vols[k]
                        R[k] = np.where(den * norm3d() >= m3.DENS_MIN, num / np.maximum(den, 1e-12), np.nan)
                    brain = A.ann[::5, ::5, ::5] > 0
                    mid = int(m3.MID / m3.GRID)
                    if j < 2:                                      # sagittal max projection, one category each
                        k, col = (ka, ca) if j == 0 else (kb, cb)
                        P = np.nanmax(R[k][:, :, mid:], axis=2).T
                        im = ax.imshow(P, extent=(0, P.shape[1] * 0.05, P.shape[0] * 0.05, 0), cmap=to_white(col), vmin=0,
                                       vmax=fmax, interpolation='bilinear')
                        ax.contour(np.arange(brain.shape[0]) * 0.05 + 0.025, np.arange(brain.shape[1]) * 0.05 + 0.025,
                                   brain.any(2).T.astype(float), levels=[0.5], colors='0.25', linewidths=0.5)
                        ax.set_xlim(1.0, 12.4); ax.set_ylim(8.0, -0.2)
                        if i == 0:
                            ax.set_title('Sagittal max projection', pad=2)
                        ax.text(6.7, -0.15, la if j == 0 else lb, color=col, ha='center', va='bottom', fontsize=5)
                    else:                                          # horizontal (j=2) / coronal (j=3) max projections, mirrored
                        ax_ = 1 if j == 2 else 0
                        PL = np.nanmax(R[ka][:, :, mid:], axis=ax_)
                        PR = np.nanmax(R[kb][:, :, mid:], axis=ax_)
                        b = brain.any(ax_)
                        if j == 2:                                 # rows = AP (anterior up), cols = lateral
                            ext = (0, PL.shape[1] * 0.05, PL.shape[0] * 0.05, 0)
                        else:                                      # rows = DV, cols = lateral
                            ext = (0, PL.shape[1] * 0.05, PL.shape[0] * 0.05, 0)
                        im = show_mirrored(ax, PL, PR, ext, to_white(ca), to_white(cb), 0, fmax)
                        full = np.concatenate([b[:, mid:][:, ::-1], b[:, mid:]], axis=1).astype(float)
                        xs = (np.arange(full.shape[1]) - full.shape[1] / 2 + 0.5) * 0.05
                        ys = (np.arange(full.shape[0]) + 0.5) * 0.05
                        ax.contour(xs, ys, full, levels=[0.5], colors='0.25', linewidths=0.5)
                        ax.axvline(0, color='0.75', lw=0.3, ls=':')
                        ax.set_xlim(-5.9, 5.9)
                        ax.set_ylim((13.2, 0) if j == 2 else (8.0, -0.2))
                        if i == 0:
                            ax.set_title('Horizontal max projection' if j == 2 else 'Coronal max projection', pad=2)
                        ax.text(-5.8, 0.1 if j == 3 else 0.3, la, color=ca, fontsize=5, va='top'); ax.text(5.8, 0.1 if j == 3 else 0.3, lb, color=cb, fontsize=5, ha='right', va='top')
                elif j < 3:
                    name, c = centres[j]
                    lab = cor_background(ax, A, c)
                    sl = keep_in_section(u[(u['ccf_atlas_ap'] - c).abs() <= m3.SLAB / 2], lab)
                    if i == 0:
                        ax.set_title(f'{name}: AP {(m3.BREGMA_AP - c) / 1000:+.1f} mm', pad=2)
                    if vname == '1_dots':
                        for side, (k, v, lbl, col) in zip((-1, 1), cats):
                            ax.scatter(side * sl['lat_mm'], sl['dv_mm'], s=0.08, color='0.86', lw=0, rasterized=True, zorder=3)
                            s = sl[sl[k]].copy(); s = s.iloc[np.argsort(s[v].abs().to_numpy())]
                            im = ax.scatter(side * s['lat_mm'], s['dv_mm'], s=0.35, c=s[v], cmap=SEL_CMAP, vmin=-0.5, vmax=0.5,
                                            lw=0, rasterized=True, zorder=4)
                            s.assign(side=side, category=k).to_csv(data_dir / f'Fig2_ccf1_{rkey}_{name}_{k}.csv', index=False)
                    elif vname == '2_fraction':
                        ML, ext = ratio_cor(*vols[ka], c); MR, _ = ratio_cor(*vols[kb], c)
                        im = show_mirrored(ax, ML, MR, ext, to_white(ca), to_white(cb), 0, fmax)
                    else:
                        ML, ext = ratio_cor(*vols[sels[0]], c); MR, _ = ratio_cor(*vols[sels[1]], c)
                        im = show_mirrored(ax, ML, MR, ext, SEL_CMAP, SEL_CMAP, -0.2, 0.2)
                    ax.text(-5.8, -0.1, la, color=ca if vname != '3_selectivity' else 'k', fontsize=5, va='bottom')
                    ax.text(5.8, -0.1, lb, color=cb if vname != '3_selectivity' else 'k', fontsize=5, va='bottom', ha='right')
                else:                                              # sagittal, all neurons (ML collapsed)
                    sag_background(ax, A)
                    if i == 0:
                        ax.set_title('Sagittal: all neurons', pad=2)
                    if vname == '1_dots':
                        ax.scatter(u['ap_mm'], u['dv_mm'], s=0.05, color='0.88', lw=0, rasterized=True, zorder=3)
                        s = pd.concat([u[u[ka]].assign(_v=u.loc[u[ka], va]), u[u[kb]].assign(_v=u.loc[u[kb], vb])])
                        s = s.iloc[np.argsort(s['_v'].abs().to_numpy())]
                        im = ax.scatter(s['ap_mm'], s['dv_mm'], s=0.25, c=s['_v'], cmap=SEL_CMAP, vmin=-0.5, vmax=0.5, lw=0,
                                        rasterized=True, zorder=4)
                        ax.text(6.7, -0.15, f'{la} + {lb}', ha='center', va='bottom', fontsize=5)
                    elif vname == '2_fraction':
                        Ma, ext = ratio_sag_all(*vols[ka]); Mb, _ = ratio_sag_all(*vols[kb])
                        im = ax.imshow(Ma - Mb, extent=ext, cmap=LinearSegmentedColormap.from_list('d', [cb, '#ffffff', ca]),
                                       vmin=-0.3, vmax=0.3, interpolation='bilinear', zorder=1.5)
                        ax.text(6.7, -0.15, f'fraction {la} - {lb}', ha='center', va='bottom', fontsize=5)
                    else:
                        M, ext = ratio_sag_all(*vols[sels[0]])
                        im = ax.imshow(M, extent=ext, cmap=SEL_CMAP, vmin=-0.2, vmax=0.2, interpolation='bilinear', zorder=1.5)
                        ax.text(6.7, -0.15, sels[0].replace('sel_', '').replace('_', ' ') + ' selectivity', ha='center',
                                va='bottom', fontsize=5)
                if j == 0:
                    ax.text(-0.03, 0.5, rtitle, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
            cax = fig.add_subplot(gs[i, 4])
            cbar = fig.colorbar(im, cax=cax); cbar.ax.tick_params(labelsize=5)
            cbar.set_label({'1_dots': 'ROC selectivity', '2_fraction': 'fraction (sagittal: difference)',
                            '3_selectivity': 'mean selectivity', '4_projections': 'fraction (max)'}[vname], fontsize=5)
        title = {'1_dots': 'significant neurons coloured by signed ROC selectivity (grey: all recorded)',
                 '2_fraction': 'smoothed fraction of recorded neurons significant (3-D Gaussian, sigma 150 um, >= 3 neurons/kernel)',
                 '3_selectivity': 'smoothed mean signed selectivity, all neurons (3-D Gaussian, sigma 150 um)',
                 '4_projections': 'maximum projections of the smoothed fraction volume'}[vname]
        fig.suptitle(f'CCF maps, variant {vname[0]}: {title}; day 0, R+ and R- pooled, good + mua', fontsize=6, y=0.985)
        ps.save_figure(fig, OUT, f'Figure2_explore_ccf_{vname}', raster_dpi=600)
        plt.close(fig)
        print('saved', vname, flush=True)


if __name__ == '__main__':
    main()
