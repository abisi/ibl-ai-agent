"""Entry points on the atlas surface: dorsal and lateral views of the CCF surface coloured by area group (labels via
allen_utils), with target centroids and flagged insertions (from 4b). Output: 4d_entry_points_atlas_surface.*"""
import contextlib
import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas

import ephys_utilities.allen_utils.allen_utils as au
from probe_viz_trials import OUT, RC, NONE, TARGET_ORDER, TARGET_COLORS, gcol, save

BREGMA = dict(ap=5400, ml=5739, dv=332)
atlas = BrainGlobeAtlas('allen_mouse_25um')
res = atlas.resolution[0]
ann = atlas.annotation                                   # AP, DV, ML
excluded = set(au.get_excluded_areas())

# area group per structure id (layers -> parent), as for units
ids = np.unique(ann)
ids = ids[ids > 0]
rows = []
for i in ids:
    acr = atlas.structures[i]['acronym']
    path = atlas.structures[i]['structure_id_path']
    rows.append(dict(id=i, ccf_atlas_acronym=acr,
                     ccf_atlas_parent_acronym=atlas.structures[path[-2]]['acronym'] if len(path) > 1 else acr))
lab = pd.DataFrame(rows).assign(target_region='none', ccf_atlas_ap=0.0, ccf_atlas_ml=0.0, ccf_atlas_dv=0.0, mouse_id='x',
                                cluster_id=lambda d: d['id'])
with contextlib.redirect_stdout(io.StringIO()):
    out = au.process_allen_labels(lab.copy(), split_merge_areas=False)
id2group = dict(zip(out['id'], out['area_group']))
id2area = dict(zip(out['id'], out['area_acronym_custom']))
groups = [g for g in au.get_area_group_custom_order()] + ['Other']
gidx = {g: k for k, g in enumerate(groups)}


def surface(axis, from_end):
    """Structure id of the first labelled, non-excluded voxel seen along `axis` (DV from top, ML from one side)."""
    a = np.moveaxis(ann, axis, 0)
    if from_end:
        a = a[::-1]
    keep = np.isin(a, [i for i in ids if id2group.get(i) is not None and i in id2area])
    keep &= a > 0
    first = keep.argmax(axis=0)
    surf = np.take_along_axis(a, first[None], axis=0)[0]
    surf[~keep.any(axis=0)] = 0
    return surf


def rgb(surf, mode='group'):
    """Surface colours: 'group' = light tint of the area-group colour, 'allen' = Allen atlas colour of the structure."""
    img = np.ones(surf.shape + (3,))
    for i in np.unique(surf):
        if i == 0:
            continue
        if mode == 'plain':
            continue                                     # white; the brain outline is drawn as a black contour
            continue
        if mode == 'allen':
            c = np.array(atlas.structures[i]['rgb_triplet']) / 255
            img[surf == i] = 0.75 * c + 0.25
        else:
            g = id2group.get(i, 'Other')
            c = mcolors.to_rgb(gcol(g) if g != 'Other' else '#e6e6e6')
            img[surf == i] = 0.45 * np.array(c) + 0.55      # light tint
    return img


def boundaries(ax, surf, extent, key, lw, color):
    lab_img = np.vectorize(lambda i: hash(key.get(i, '')) % 100003 if i else -1)(surf).astype(float)
    edge = (np.diff(lab_img, axis=0, prepend=lab_img[:1]) != 0) | (np.diff(lab_img, axis=1, prepend=lab_img[:, :1]) != 0)
    ax.imshow(np.ma.masked_where(~edge, edge), extent=extent, cmap=mcolors.ListedColormap([color]),
              interpolation='nearest', alpha=lw)


def label_groups(ax, surf, extent, transform, xlim, ylim, min_px=400):
    """Area-group name at the median position of its visible surface; only groups with >= min_px pixels in view."""
    for g in set(id2group.get(i) for i in np.unique(surf) if i):
        if g in (None, 'Other'):
            continue
        rc = np.argwhere(np.isin(surf, [i for i in ids if id2group.get(i) == g]))
        xy = np.array([transform(r, c) for r, c in rc])
        inside = (xy[:, 0] >= min(xlim)) & (xy[:, 0] <= max(xlim)) & (xy[:, 1] >= min(ylim)) & (xy[:, 1] <= max(ylim))
        if inside.sum() < min_px:
            continue
        x, y = np.median(xy[inside], axis=0)
        ax.text(x, y, g.replace(' areas', '').replace('Somatosensory-', 'SS-'), fontsize=6, color='0.25',
                ha='center', va='center', alpha=0.85)


e = pd.read_csv(OUT / '4b_insertion_points_centroids.csv')
e['dv_mm'] = (e['dv_ccf_um'] - BREGMA['dv']) / 1000
off = e[e['off_target']].sort_values('dist_mm', ascending=False).reset_index(drop=True)
off['n'] = off.index + 1
cross = pd.read_csv(OUT / '4c_entry_crossref.csv')[['session', 'probe', 'hits_target_sample', 'hits_target_atlas']]
e = e.merge(cross, on=['session', 'probe'], how='left')
miss = e[(e['hits_target_atlas'].astype(str) == 'False')]
targets = [t for t in TARGET_ORDER if t in set(e['target_region'])]
cent = e.groupby('target_region')[['ml_mm', 'ap_mm', 'dv_mm']].median()

# dorsal surface: rows = AP index, cols = ML index; lateral surface (from high-ML side): rows = AP, cols = DV
dors = surface(axis=1, from_end=False)
lat = surface(axis=2, from_end=True)
n_ap, n_ml = dors.shape
ml_ext = ((0 - BREGMA['ml']) / 1000, (n_ml * res - BREGMA['ml']) / 1000)
ap_ext = ((BREGMA['ap'] - n_ap * res) / 1000, BREGMA['ap'] / 1000)
dv_ext = ((ann.shape[1] * res - BREGMA['dv']) / 1000, (0 - BREGMA['dv']) / 1000)


# day (sessions table: whisker_<day>) and cohort (reference sheet, share_internal) per insertion
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from matplotlib.ticker import MultipleLocator
REPO = Path('/home/bisi/code/ibl-ai-agent')
ses = pd.read_parquet(REPO / 'reports/datasets/ssl_ephys/1.0.0/metadata/sessions.parquet', columns=['session_id', 'session_description'])
ses['day'] = pd.to_numeric(ses['session_description'].str.extract(r'whisker_([-+]?\d+)')[0], errors='coerce')
sheet = pd.read_excel('/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx', sheet_name='Sheet1')
e = (e.merge(ses[['session_id', 'day']], left_on='session', right_on='session_id', how='left')
      .merge(sheet[['mouse_id', 'reward_group']], on='mouse_id', how='left'))
e['stage'] = np.where(e['day'] == 0, 'day 0', np.where(e['day'] > 0, 'expert', 'unknown'))
print('insertions per stage:', e['stage'].value_counts().to_dict(), '| per cohort:', e['reward_group'].value_counts(dropna=False).to_dict())
DORS_LIM, LAT_LIM = ((-0.6, 5.8), (-5.2, 3.9)), ((-4.5, 1.0), (-0.3, 5.0))
ext_d = (ml_ext[0], ml_ext[1], ap_ext[0], ap_ext[1])
ext_l = (ap_ext[1], ap_ext[0], dv_ext[0], dv_ext[1])


def surface_panel(a_, which, MODE):
    """background (area colours or white), outline (plain), 1-mm grid clipped to the brain"""
    PLAIN = MODE == 'plain'
    if which == 'dorsal':
        a_.imshow(rgb(dors, MODE)[::-1], extent=ext_d, origin='lower', interpolation='nearest')
        if not PLAIN:
            boundaries(a_, dors[::-1], ext_d, id2area if MODE == 'allen' else id2group, 0.8, '0.35')
            label_groups(a_, dors, ext_d, lambda r, c: (ml_ext[0] + c * res / 1000, ap_ext[1] - r * res / 1000), *DORS_LIM)
        X = ml_ext[0] + (np.arange(dors.shape[1]) + 0.5) * res / 1000
        Y = ap_ext[1] - (np.arange(dors.shape[0]) + 0.5) * res / 1000
        M, (xl, yl) = (dors > 0).astype(float), DORS_LIM
    else:
        a_.imshow(rgb(lat, MODE).transpose(1, 0, 2), extent=ext_l, origin='upper', interpolation='nearest', aspect='equal')
        if not PLAIN:
            boundaries(a_, lat.T, ext_l, id2area if MODE == 'allen' else id2group, 0.8, '0.35')
            label_groups(a_, lat, ext_l, lambda r, c: (BREGMA['ap'] / 1000 - r * res / 1000, (c * res - BREGMA['dv']) / 1000), *LAT_LIM)
        X = BREGMA['ap'] / 1000 - (np.arange(lat.shape[0]) + 0.5) * res / 1000
        Y = (np.arange(lat.shape[1]) * res + res / 2 - BREGMA['dv']) / 1000
        M, (xl, yl) = (lat > 0).astype(float).T, LAT_LIM
    if PLAIN:
        a_.contour(X, Y, M, levels=[0.5], colors='k', linewidths=0.8, zorder=3)
    cf = a_.contourf(X, Y, M, levels=[0.5, 1.5], colors='none')
    clip = PathPatch(MPath.make_compound_path(*cf.get_paths()), transform=a_.transData, facecolor='none', edgecolor='none')
    a_.add_patch(clip)
    a_.xaxis.set_major_locator(MultipleLocator(1)); a_.yaxis.set_major_locator(MultipleLocator(1)); a_.grid(False)
    for v in np.arange(np.ceil(min(xl)), np.floor(max(xl)) + 1):
        ln, = a_.plot([v, v], [min(yl), max(yl)], color='0.75', lw=0.4, zorder=2); ln.set_clip_path(clip)
    for v in np.arange(np.ceil(min(yl)), np.floor(max(yl)) + 1):
        ln, = a_.plot([min(xl), max(xl)], [v, v], color='0.75', lw=0.4, zorder=2); ln.set_clip_path(clip)
    a_.set_xlim(*xl)
    if which == 'dorsal':
        a_.set_ylim(*yl); a_.set_xlabel('ML from bregma (mm)'); a_.set_ylabel('AP from bregma (mm)')
    else:
        a_.set_ylim(max(yl), min(yl)); a_.set_xlabel('AP from bregma (mm, anterior right)'); a_.set_ylabel('DV from bregma (mm)')


def points(a_, which, sub, MODE):
    """entries: day 0 filled, expert hollow; centroid of the shown entries per target (diamond / X)"""
    PLAIN = MODE == 'plain'
    xcol, ycol = ('ml_mm', 'ap_mm') if which == 'dorsal' else ('ap_mm', 'dv_mm')
    if which == 'lateral':
        sub = sub[sub['ml_mm'] > 3.0]                         # entries on the lateral surface
    for t in targets:
        d = sub[sub['target_region'] == t]
        c = TARGET_COLORS.get(t, '0.3')
        f, h = d[d['stage'] != 'expert'], d[d['stage'] == 'expert']
        a_.scatter(f[xcol], f[ycol], s=8, color=c, edgecolor='none' if PLAIN else 'k', lw=0 if PLAIN else 0.2,
                   alpha=0.55 if PLAIN else 1, zorder=4)
        a_.scatter(h[xcol], h[ycol], s=8, facecolor='none', edgecolor=c, lw=0.7, alpha=0.8 if PLAIN else 1, zorder=4)
        if len(d):
            a_.scatter(d[xcol].median(), d[ycol].median(), s=30 if PLAIN else 45, marker='D' if PLAIN else 'X',
                       edgecolor='none' if PLAIN else 'k', lw=0 if PLAIN else 0.5, color=c, zorder=5)


def legend(fig, sub, MODE, bottom):
    PLAIN = MODE == 'plain'
    h = [plt.Line2D([], [], marker='o', ls='', mfc=TARGET_COLORS.get(t, '0.3'), mec='none' if PLAIN else 'k', mew=0.3, ms=5)
         for t in targets]
    l = [f'{t} (n={int((sub.target_region == t).sum())})' for t in targets]
    h += [plt.Line2D([], [], marker='o', ls='', mfc='0.5', mec='none', ms=5),
          plt.Line2D([], [], marker='o', ls='', mfc='none', mec='0.5', mew=0.8, ms=5),
          plt.Line2D([], [], marker='D' if PLAIN else 'X', ls='', mfc='0.6', mec='none' if PLAIN else 'k', ms=5.5 if PLAIN else 6)]
    l += ['day 0', 'expert', 'target centroid']
    fig.legend(h, l, loc='lower center', ncol=8, frameon=False, fontsize=7, handletextpad=0.1, columnspacing=0.8,
               bbox_to_anchor=(0.5, 0))


RC4 = {**RC, 'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5}
SPLITS = [('', None), ('_day0_vs_expert', [('day 0', e['stage'] == 'day 0'), ('expert', e['stage'] == 'expert')]),
          ('_Rplus_vs_Rminus', [('R+', e['reward_group'] == 'R+'), ('R-', e['reward_group'] == 'R-')])]
for MODE, SUFFIX in [('group', ''), ('allen', '_allen_colors'), ('plain', '_plain')]:
    for SPLIT, groups in SPLITS:
        with plt.rc_context(RC4):
            if groups is None:                                  # all insertions: dorsal | lateral
                fig, (ax, bx) = plt.subplots(1, 2, figsize=(18 / 2.54, 10.5 / 2.54), gridspec_kw={'width_ratios': [1, 1.25]})
                for a_, which in ((ax, 'dorsal'), (bx, 'lateral')):
                    surface_panel(a_, which, MODE); points(a_, which, e, MODE)
                ax.set_title('Entry points, dorsal surface (atlas space, all days)')
                bx.set_title('Lateral surface (entries with ML > 3 mm)')
                legend(fig, e, MODE, 0.11)
                fig.tight_layout(rect=(0, 0.11, 1, 1))
            else:                                               # two groups side by side: rows dorsal / lateral
                fig, axs = plt.subplots(2, 2, figsize=(18 / 2.54, 21 / 2.54), gridspec_kw={'height_ratios': [9.1, 5.3]})
                for j, (name, mask) in enumerate(groups):
                    sub = e[mask]
                    for i, which in enumerate(('dorsal', 'lateral')):
                        surface_panel(axs[i, j], which, MODE); points(axs[i, j], which, sub, MODE)
                    axs[0, j].set_title(f'{name}: {len(sub)} insertions, {sub.mouse_id.nunique()} mice')
                    axs[1, j].set_title('lateral surface (ML > 3 mm)', fontsize=7.5)
                legend(fig, e, MODE, 0.06)
                fig.tight_layout(rect=(0, 0.06, 1, 1))
            save(fig, '4d_entry_points_atlas_surface' + SUFFIX + SPLIT,
                 e[['mouse_id', 'session', 'probe', 'day', 'stage', 'reward_group', 'target_region', 'ml_mm', 'ap_mm',
                    'dv_mm', 'entry_area', 'off_target', 'hits_target_atlas']])
            plt.close(fig)
