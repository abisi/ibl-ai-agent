"""Figure 2 exploration: where are stimulus, lick and decision neurons? Grid of 3 rows x 4 columns.
Rows (single-unit ROC, roc_utils_new; _roc_stage_analysis tables):
  1  stimulus: whisker_active vs auditory_active significant units (left hemisphere = whisker, right = auditory, mirrored)
  2  licks: spontaneous_licks significant, enhanced (left) vs suppressed (right)
  3  decision: whisker_decision (left) vs auditory_decision (right); M_decision = significant M_choice (hit vs miss) AND
     significant M_hit_vs_spontaneous (roc_utils_new.compute_neuron_labels)
Columns: three 1-mm coronal slabs (AP +2.0 mm; the median AP of the SSp-bfd units = wS1; the median AP of the Midbrain
units), and one sagittal projection of all units (both categories overlaid, ML collapsed).
Units: good + mua, day 0 (learning), R+ and R- pooled. Units whose CCF voxel is outside the brain, or (coronal slabs)
whose projected position falls outside the brain section at the slab centre, are removed.
Dots = significant units (light grey = all recorded units in the slab, for coverage); contours = Gaussian KDE of the
significant units (40 / 70 % of the peak density).
Run on haas (brainrender env): python explore_slice_grid.py [--all-days]
Output: Figures/Figure2_sensory_maps_explore_<date>/ (figure + one CSV per panel)"""
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / 'projects' / 'ssl-paper-figures'))
sys.path.insert(0, str(HERE.parents[3] / 'scripts'))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from brainglobe_atlasapi import BrainGlobeAtlas  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

import paperstyle as ps  # noqa: E402

ROC = axel_bisi_root() / 'combined_results_ks4' / '_roc_stage_analysis'
OUT = axel_bisi_root() / 'combined_results_ks4' / '_sensory_spatial_maps' / 'figure2_maps'
KEYS = ['mouse_id', 'session_id', 'electrode_group', 'cluster_id']
BREGMA_AP, BREGMA_DV, MID = 5400.0, 332.0, 5700.0          # um, CCF (25-um atlas midline = 5700)
SLAB_MM = 1.0
ALL_DAYS = '--all-days' in sys.argv
C_LICK = {'enhanced': '#c0392b', 'suppressed': '#2471a3'}
ROWS = [('stimulus', 'Stimulus response', [('whisker', 'Whisker', ps.COLORS['whisker']), ('auditory', 'Auditory', ps.COLORS['auditory'])]),
        ('lick', 'Spontaneous licks', [('enhanced', 'Enhanced', C_LICK['enhanced']), ('suppressed', 'Suppressed', C_LICK['suppressed'])]),
        ('decision', 'Decision', [('whisker_decision', 'Whisker decision', ps.COLORS['whisker']), ('auditory_decision', 'Auditory decision', ps.COLORS['auditory'])])]


def load_units():
    u = pd.read_parquet(ROC / 'units.parquet')
    u = u[u['quality_label'].isin(['good', 'mua'])]
    if not ALL_DAYS:
        u = u[u['day'] == 0]
    u['cluster_id'] = u['cluster_id'].astype(str)
    u = u.drop_duplicates(KEYS)                                   # the all-days table has duplicate unit rows
    r = pd.read_parquet(ROC / 'roc_long.parquet', columns=KEYS + ['analysis_type', 'sig', 'pos', 'neg'],
                        filters=[('analysis_type', 'in', ['whisker_active', 'auditory_active', 'spontaneous_licks'])])
    r['cluster_id'] = r['cluster_id'].astype(str)
    r = r.drop_duplicates(KEYS + ['analysis_type'])
    for at, cols in (('whisker_active', ['sig']), ('auditory_active', ['sig']), ('spontaneous_licks', ['sig', 'pos', 'neg'])):
        x = r[r['analysis_type'] == at][KEYS + cols].rename(columns={c: f'{at}_{c}' for c in cols})
        u = u.merge(x, on=KEYS, how='left')
    u['whisker'] = u['whisker_active_sig'] == 1
    u['auditory'] = u['auditory_active_sig'] == 1
    u['enhanced'] = (u['spontaneous_licks_sig'] == 1) & (u['spontaneous_licks_pos'] == 1)
    u['suppressed'] = (u['spontaneous_licks_sig'] == 1) & (u['spontaneous_licks_neg'] == 1)
    u['whisker_decision'] = u['whisker_decision'].fillna(False).astype(bool)
    u['auditory_decision'] = u['auditory_decision'].fillna(False).astype(bool)
    u['ap_mm'] = (BREGMA_AP - u['ccf_atlas_ap']) / 1000
    u['dv_mm'] = (u['ccf_atlas_dv'] - BREGMA_DV) / 1000
    u['lat_mm'] = np.abs(u['ccf_atlas_ml'] - MID) / 1000
    return u


def main():
    atlas = BrainGlobeAtlas('allen_mouse_25um')
    ann, res = atlas.annotation, atlas.resolution[0]                # AP, DV, ML
    # outline id map: every structure -> its ancestor at depth <= 6 (major areas), for light region boundaries
    coarse = {}
    for sid, st in atlas.structures.items():
        if isinstance(sid, int):
            p = st['structure_id_path']
            coarse[sid] = p[min(len(p), 7) - 1]
    lut = np.zeros(int(ann.max()) + 1, dtype=np.int64)
    for k, v in coarse.items():
        if k < len(lut):
            lut[k] = v
    u = load_units()
    idx = np.clip(np.round(u[['ccf_atlas_ap', 'ccf_atlas_dv', 'ccf_atlas_ml']].to_numpy(float) / res).astype(int), 0,
                  np.array(ann.shape) - 1)
    inside = ann[idx[:, 0], idx[:, 1], idx[:, 2]] > 0
    print(f'units: {len(u)} ({"all days" if ALL_DAYS else "day 0"}), outside the brain: {(~inside).sum()}')
    u = u[inside].copy()
    centres = [('Frontal', 2.0),
               ('wS1', float(u.loc[u['area_acronym_custom'] == 'SSp-bfd', 'ap_mm'].median())),
               ('Midbrain', float(u.loc[u['area_group'] == 'Midbrain', 'ap_mm'].median()))]
    print('slab centres (AP mm from bregma):', [(n, round(c, 2)) for n, c in centres])

    ps.set_nature_rc()
    fig = plt.figure(figsize=(183 * ps.MM, 128 * ps.MM))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.25], wspace=0.08, hspace=0.25, left=0.06, right=0.99, top=0.93, bottom=0.04)
    data_dir = OUT / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i, (rkey, rtitle, cats) in enumerate(ROWS):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            rows = []
            if j < 3:                                              # coronal slab, mirrored
                name, c = centres[j]
                sl = u[np.abs(u['ap_mm'] - c) <= SLAB_MM / 2].copy()
                ai = int(round((BREGMA_AP - c * 1000) / res))
                sec = ann[ai]                                      # DV x ML
                ml = (np.arange(sec.shape[1]) * res + res / 2 - MID) / 1000
                dv = (np.arange(sec.shape[0]) * res + res / 2 - BREGMA_DV) / 1000
                lab = lut[sec]
                ax.contour(ml, dv, lab, levels=np.unique(lab)[1:] - 0.5, colors=[ps.light('k', 0.82)], linewidths=0.25)
                ax.contour(ml, dv, (sec > 0).astype(float), levels=[0.5], colors='k', linewidths=0.5)
                # drop units whose projection falls outside this section
                di = np.clip(np.round((sl['dv_mm'] * 1000 + BREGMA_DV) / res).astype(int), 0, sec.shape[0] - 1)
                mi_l = np.clip(np.round((MID - sl['lat_mm'] * 1000) / res).astype(int), 0, sec.shape[1] - 1)
                mi_r = np.clip(np.round((MID + sl['lat_mm'] * 1000) / res).astype(int), 0, sec.shape[1] - 1)
                ok = (sec[di, mi_l] > 0) & (sec[di, mi_r] > 0)
                sl = sl[ok.to_numpy() if hasattr(ok, 'to_numpy') else ok]
                for side, (ck, clab, col) in zip((-1, 1), cats):
                    x = side * sl['lat_mm'].to_numpy()
                    ax.scatter(x, sl['dv_mm'], s=0.15, color=ps.light('k', 0.85), lw=0, rasterized=True, zorder=1)
                    m = sl[ck].to_numpy()
                    ax.scatter(x[m], sl['dv_mm'][m], s=0.25, color=col, lw=0, rasterized=True, zorder=3)
                    if m.sum() >= 10:
                        pts = np.vstack([x[m], sl['dv_mm'].to_numpy()[m]])
                        k = gaussian_kde(pts, bw_method=0.25)
                        gx, gy = np.meshgrid(np.linspace(side * 0.0, side * 5.8, 80), np.linspace(-0.5, 7.5, 100))
                        z = k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
                        ax.contour(gx, gy, z, levels=[0.4 * z.max(), 0.7 * z.max()], colors=['k'], linewidths=[0.35, 0.6], zorder=4)
                    rows.append(sl.assign(side='left' if side < 0 else 'right', category=ck, plotted_x_mm=x, significant=m)
                                [KEYS + ['reward_group', 'day', 'area_acronym_custom', 'ap_mm', 'dv_mm', 'lat_mm', 'side', 'category',
                                         'plotted_x_mm', 'significant']])
                    n_sig, n_all = int(m.sum()), len(m)
                    ax.text(side * 5.8, -0.55, f'{clab}\n{n_sig}/{n_all}', ha='right' if side > 0 else 'left', va='bottom',
                            fontsize=5, color=col)
                ax.axvline(0, color=ps.light('k', 0.6), lw=0.3, ls=':')
                ax.set_xlim(-5.9, 5.9); ax.set_ylim(7.8, -1.6)
                if i == 0:
                    ax.set_title(f'{name}: AP {c:+.1f} mm ({SLAB_MM:g}-mm slab)', pad=2)
            else:                                                  # sagittal projection, all units, both categories
                proj = (ann > 0).any(axis=2)                       # AP x DV silhouette
                ap = (BREGMA_AP - (np.arange(ann.shape[0]) * res + res / 2)) / 1000
                dv = (np.arange(ann.shape[1]) * res + res / 2 - BREGMA_DV) / 1000
                mi = int(round((MID + float(u['lat_mm'].median()) * 1000) / res))
                lab = lut[ann[:, :, mi]]
                ax.contour(ap, dv, lab.T, levels=np.unique(lab)[1:] - 0.5, colors=[ps.light('k', 0.82)], linewidths=0.25)
                ax.contour(ap, dv, proj.T.astype(float), levels=[0.5], colors='k', linewidths=0.5)
                ax.scatter(u['ap_mm'], u['dv_mm'], s=0.1, color=ps.light('k', 0.88), lw=0, rasterized=True, zorder=1)
                xs, ys, cs = [], [], []
                for ck, clab, col in cats:                         # both categories, drawn in random order
                    m = u[ck].to_numpy()
                    xs.append(u['ap_mm'].to_numpy()[m]); ys.append(u['dv_mm'].to_numpy()[m]); cs += [col] * int(m.sum())
                xs, ys = np.concatenate(xs), np.concatenate(ys); o = rng.permutation(len(xs))
                ax.scatter(xs[o], ys[o], s=0.25, c=list(np.array(cs, dtype=object)[o]), lw=0, rasterized=True, zorder=3)
                for ck, clab, col in cats:
                    m = u[ck].to_numpy()
                    rows.append(u.assign(category=ck, significant=m)[KEYS + ['reward_group', 'day', 'area_acronym_custom', 'ap_mm',
                                                                               'dv_mm', 'lat_mm', 'category', 'significant']])
                ax.set_xlim(4.4, -7.6); ax.set_ylim(7.8, -1.6)             # anterior left
                if i == 0:
                    ax.set_title('Sagittal: all units (ML collapsed)', pad=2)
                ax.plot([-7.0, -6.0], [7.4, 7.4], color='k', lw=0.75)
                ax.text(-6.5, 7.25, '1 mm', ha='center', va='bottom', fontsize=5)
            if j == 0:
                ax.text(-0.04, 0.5, rtitle, transform=ax.transAxes, rotation=90, ha='right', va='center', fontsize=6)
            pd.concat(rows).to_csv(data_dir / f'Fig2_explore_{rkey}_{"sagittal" if j == 3 else "coronal" + str(j + 1)}.csv', index=False)
    fig.suptitle(f'Figure 2 exploration: single-unit ROC maps, {"all days" if ALL_DAYS else "day 0"}, R+ and R- pooled, good + mua',
                 fontsize=6, y=0.985)
    ps.save_figure(fig, OUT, 'Figure2_explore_slice_grid', raster_dpi=600)
    print('saved', OUT)


if __name__ == '__main__':
    main()
