"""Exploratory visualisations of the probe-track / unit dataset (user 2026-11-16: "try all those in a temp folder").
Outputs: combined_results_ks4/anatomy/_viz_trials_20261116/<n>_<name>.{png,svg,csv}

 1 3D overview (copies of the brainrender area_group screenshots)
 2 probe stripe plot (depth along each track, coloured by area group, grouped by target)
 3 target x area-group coverage: track length heatmap, unit-count heatmap, alluvial target -> area group (units)
 4 2D atlas projections: entry points (top view) + tracks (sagittal / coronal) coloured by area group
 5 units: 2D projections coloured by area group + Swanson flatmap of unit counts
 6 small multiples per target (sagittal and coronal projections)

Run (PC, brainrender env): PYTHONPATH=<ephys_utilities> python probe_viz_trials.py
"""
import contextlib
import io
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath

import ephys_utilities.allen_utils.allen_utils as au
import ephys_utilities.allen_utils.generate_probe_tracks_visualization_colors_single_parallel as g

ROOT = Path('/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/anatomy')
OUT = ROOT / '_viz_trials_20261116'
OUT.mkdir(parents=True, exist_ok=True)
UNIT_CACHE = Path('/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/data_summary/unit_table_cache/unit_table_all_mice.parquet')
RC = {'font.family': 'sans-serif', 'font.sans-serif': ['Arial'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
      'font.size': 6, 'axes.titlesize': 7, 'axes.labelsize': 6.5, 'xtick.labelsize': 5.5, 'ytick.labelsize': 5.5,
      'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5}
GROUP_COLORS = au.get_custom_area_groups_colors()
NONE = 'Other / no area'
NONE_COLOR = '#d9d9d9'
TARGET_ORDER = au.get_target_region_order()
TARGET_COLORS = g.TARGET_AREA_CUSTOM_CMAP


def gcol(group):
    return GROUP_COLORS.get(group, NONE_COLOR) if isinstance(group, str) and group not in ('Other', NONE) else NONE_COLOR


def group_order(present):
    order = [x for x in au.get_area_group_custom_order() if x in present]
    return order + [x for x in sorted(present) if x not in order and x not in (NONE, 'Other')] + \
        ([NONE] if NONE in present or 'Other' in present else [])


def save(fig, name, table=None):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    if table is not None:
        table.to_csv(OUT / f'{name}.csv', index=False)
    print('saved', name, flush=True)


def group_legend(ax, groups, **kw):
    ax.legend(handles=[Patch(facecolor=gcol(x), label=x) for x in groups], frameon=False, fontsize=5,
              handlelength=1, handleheight=0.8, **kw)


# ------------------------------------------------------------------ data: per-point track table
def collect_track_points():
    """Same probe selection as generate_brain_visualization (all recording days, valid probes, sheet exclusions), one row per
    track point with its depth along the track and area labels (allen_utils.process_allen_labels)."""
    cache = OUT / 'track_points.parquet'  # all recording days (day-0 version kept as track_points_day0.parquet)
    if cache.exists():
        return pd.read_parquet(cache)
    p = g.params
    atlas = BrainGlobeAtlas('allen_mouse_bluebrain_barrels_10um')
    probes = pd.read_excel(g.resolve_path(p['probe_info_path']))
    probes = probes[probes['valid'] == 1]
    mice = pd.read_excel(g.resolve_path(p['mouse_info_path'])).rename(columns={'mouse_name': 'mouse_id'})
    mice = mice[(mice['exclude'] == 0) & (mice['exclude_ephys'] == 0) & mice['reward_group'].isin(['R+', 'R-'])
                & (mice['recording'] == 1)]
    rows = []
    for m_name in [m for m in probes['mouse_name'].unique() if m in set(mice['mouse_id'])]:
        base = g.resolve_path(p['input_data_path_axel'] if m_name.startswith('AB') else p['input_data_path_myriam'])
        mouse_probes = probes[probes['mouse_name'] == m_name]
        for date_val, date_probes in mouse_probes.groupby('date'):
            with contextlib.redirect_stdout(io.StringIO()):
                session = g.find_session_folder(base, m_name, date_val)
            if session is None:
                continue
            if m_name.startswith('AB') and int(m_name[2:]) < 80:
                folder = session / 'brainreg' / 'manual_segmentation' / 'standard_space' / 'tracks'
            else:
                folder = session / 'Anatomy' / 'fused' / 'registered' / 'segmentation' / 'atlas_space' / 'tracks'
            if not folder.exists():
                continue
            for npy in sorted(f for f in folder.iterdir() if f.suffix == '.npy' and f.name.startswith('imec')
                              and '_keep' not in f.name):  # *_keep = duplicate copies (user 2026-11-17: ignore)
                probe_name = npy.name.split('.')[0]
                info = date_probes[date_probes['probe_id'] == int(probe_name.split('_')[0][-1])]
                if info.empty:
                    continue
                track = np.load(npy, allow_pickle=True)
                track = np.linspace(track[0], track[-1], num=len(track))
                target = info['target_area'].values[0]
                if target == 'SC' and info['depth'].values[0] > 4000:
                    track = track[-200:]
                areas, groups = g.get_row_area_labels(atlas, track, target, m_name)
                depth = np.linalg.norm(track - track[0], axis=1)
                rows.append(pd.DataFrame({
                    'mouse_id': m_name, 'session': session.name, 'probe': probe_name, 'target_region': target,
                    'reward_group': info['reward_group'].values[0], 'day': int(info['day_of_recording'].values[0]),
                    'point': np.arange(len(track)), 'depth_um': depth,
                    'ap': track[:, 0], 'dv': track[:, 1], 'ml': track[:, 2],
                    'area_acronym_custom': areas, 'area_group': groups}))
                print('  track', m_name, probe_name, target, len(track), flush=True)
    pts = pd.concat(rows, ignore_index=True)
    pts['area_group'] = pts['area_group'].where(pts['area_group'].notna() & (pts['area_group'] != 'Other'), NONE)
    pts['area_acronym_custom'] = pts['area_acronym_custom'].fillna('none')
    pts.to_parquet(cache, index=False)
    return pts


def load_units():
    """Good + mua units with positions and labels from the current NWB_ks4 (unit_coords_atlas.csv, built by
    unit_coords_atlas.py with the current allen_utils); target from the probe sheet."""
    u = pd.read_csv(OUT / 'unit_coords_atlas.csv', dtype={'cluster_id': str}).dropna(subset=['ap', 'ml', 'dv'])
    u['ccf_atlas_ap'], u['ccf_atlas_ml'], u['ccf_atlas_dv'] = 5400 - u['ap'], u['ml'] + 5739, u['dv'] + 332
    sheet = collect_track_points()[['session', 'probe', 'target_region']].drop_duplicates()
    sheet = dict(zip(zip(sheet['session'], sheet['probe']), sheet['target_region']))
    probe = u['electrode_group'].astype(str).str.split('_').str[0]
    u['target_region'] = [sheet.get((s_, p), t) for s_, p, t in zip(u['session_id'], probe, u['target_region'])]
    u['area_group'] = u['area_group'].where(u['area_group'] != 'Other', NONE)
    return u[['mouse_id', 'session_id', 'electrode_group', 'cluster_id', 'target_region', 'ccf_atlas_acronym',
              'ccf_atlas_ap', 'ccf_atlas_ml', 'ccf_atlas_dv', 'area_acronym_custom', 'area_group']]


# ------------------------------------------------------------------ atlas outlines (25 um annotation)
class Outlines:
    def __init__(self):
        a = BrainGlobeAtlas('allen_mouse_25um')
        self.res = 25.0
        brain = a.annotation > 0                     # axes: AP, DV, ML
        self.top = brain.any(axis=1)                 # AP x ML
        self.sag = brain.any(axis=2)                 # AP x DV
        self.cor = brain.any(axis=0)                 # DV x ML

    def draw(self, ax, view):
        r = self.res
        if view == 'top':      # x = ML, y = AP (anterior up)
            m = self.top
            ax.contour(np.arange(m.shape[1]) * r, np.arange(m.shape[0]) * r, m, levels=[0.5], colors='0.6',
                       linewidths=0.5)
            ax.set_xlabel('ML (µm)'); ax.set_ylabel('AP (µm, anterior up)'); ax.invert_yaxis()
        elif view == 'sagittal':   # x = AP (anterior left), y = DV (dorsal up)
            m = self.sag.T
            ax.contour(np.arange(m.shape[1]) * r, np.arange(m.shape[0]) * r, m, levels=[0.5], colors='0.6',
                       linewidths=0.5)
            ax.set_xlabel('AP (µm, anterior left)'); ax.set_ylabel('DV (µm)'); ax.invert_yaxis()
        else:                       # coronal: x = ML, y = DV
            m = self.cor
            ax.contour(np.arange(m.shape[1]) * r, np.arange(m.shape[0]) * r, m, levels=[0.5], colors='0.6',
                       linewidths=0.5)
            ax.set_xlabel('ML (µm)'); ax.set_ylabel('DV (µm)'); ax.invert_yaxis()
        ax.set_aspect('equal')


def view_xy(df, view):
    return {'top': ('ml', 'ap'), 'sagittal': ('ap', 'dv'), 'coronal': ('ml', 'dv')}[view]


def track_segments(pts, view):
    """LineCollection segments between consecutive points of each probe, coloured by area group."""
    x, y = view_xy(pts, view)
    segs, cols = [], []
    for _, p in pts.groupby(['session', 'probe'], sort=False):
        xy = p[[x, y]].to_numpy()
        segs.append(np.stack([xy[:-1], xy[1:]], axis=1))
        cols += [gcol(gr) for gr in p['area_group'].to_numpy()[:-1]]
    return np.concatenate(segs), cols


# ------------------------------------------------------------------ figures
def fig1_copy_3d():
    src = ROOT / 'area_group'
    for f in sorted(src.glob('all_probes_in_atlas_*area_group*_transparent_learners.png')):
        view = f.stem.split('_')[-3]
        shutil.copy2(f, OUT / f'1_3d_overview_{view}.png')
    legend = next(src.glob('*_legend.svg'), None)
    if legend:
        shutil.copy2(legend, OUT / '1_3d_overview_legend.svg')
    print('copied 3D views', flush=True)


def probe_order(pts):
    meta = pts.groupby(['session', 'probe'], sort=False).agg(
        mouse_id=('mouse_id', 'first'), target_region=('target_region', 'first'),
        length=('depth_um', 'max')).reset_index()
    rank = {t: i for i, t in enumerate(TARGET_ORDER)}
    meta['t_rank'] = meta['target_region'].map(rank).fillna(len(rank))
    return meta.sort_values(['t_rank', 'target_region', 'mouse_id', 'probe']).reset_index(drop=True)


def fig2_stripes(pts):
    meta = probe_order(pts)
    gap = 3
    x, xs, last = 0, [], None
    for t in meta['target_region']:
        if last is not None and t != last:
            x += gap
        xs.append(x); x += 1; last = t
    meta['x'] = xs
    xmap = {(r.session, r.probe): r.x for r in meta.itertuples()}
    runs = []
    for (sess, probe), p in pts.groupby(['session', 'probe'], sort=False):
        grp = p['area_group'].to_numpy(); d = p['depth_um'].to_numpy()
        step = np.median(np.diff(d)) if len(d) > 1 else 10
        start = 0
        for i in range(1, len(grp) + 1):
            if i == len(grp) or grp[i] != grp[start]:
                runs.append((xmap[(sess, probe)], d[start], d[i - 1] + step, grp[start], sess, probe))
                start = i
    runs = pd.DataFrame(runs, columns=['x', 'depth_start_um', 'depth_end_um', 'area_group', 'session', 'probe'])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(18 / 2.54, 7 / 2.54))
        ax.bar(runs['x'], runs['depth_end_um'] - runs['depth_start_um'], bottom=runs['depth_start_um'], width=0.85,
               color=[gcol(x) for x in runs['area_group']], edgecolor='none')
        ax.invert_yaxis()
        ax.set_ylabel('Depth along track (µm)')
        centers = meta.groupby('target_region', sort=False)['x'].agg(['min', 'max'])
        ax.set_xticks((centers['min'] + centers['max']) / 2, centers.index)
        ax.tick_params(axis='x', length=0)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlim(-1, meta['x'].max() + 1)
        group_legend(ax, group_order(set(runs['area_group'])), loc='upper left', bbox_to_anchor=(1.0, 1.0))
        ax.set_title(f'{len(meta)} probes, {meta.mouse_id.nunique()} mice (all recording days), grouped by target region')
        save(fig, '2_probe_stripes', runs.merge(meta[['session', 'probe', 'mouse_id', 'target_region']]))


def fig3_coverage(pts, units):
    step = pts.groupby(['session', 'probe'])['depth_um'].transform(lambda d: np.median(np.diff(d)) if len(d) > 1 else 0)
    pts = pts.assign(len_mm=step / 1000)
    groups = group_order(set(pts['area_group']) | set(units['area_group']))
    targets = [t for t in TARGET_ORDER if t in set(pts['target_region'])]
    trk = pts.pivot_table(index='target_region', columns='area_group', values='len_mm', aggfunc='sum',
                          fill_value=0).reindex(index=targets, columns=groups, fill_value=0)
    n_probes = pts.groupby('target_region')[['session', 'probe']].apply(lambda d: len(d.drop_duplicates()))
    trk_per_probe = trk.div(n_probes.reindex(targets), axis=0)
    unt = units.pivot_table(index='target_region', columns='area_group', values='cluster_id', aggfunc='size',
                            fill_value=0).reindex(index=targets, columns=groups, fill_value=0)
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(18 / 2.54, 6.5 / 2.54))
        for ax, mat, title, fmt, cmap in [(axs[0], trk_per_probe, 'Track length per probe (mm)', '{:.1f}', 'Greys'),
                                          (axs[1], unt, 'Units (good + mua)', '{:.0f}', 'Purples')]:
            vals = mat.to_numpy(float)
            shown = np.log10(vals + 1) if 'Units' in title else vals
            ax.imshow(shown, cmap=cmap, aspect='auto')
            for (i, j), v in np.ndenumerate(vals):
                if v > 0:
                    ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=4,
                            color='white' if shown[i, j] > 0.6 * shown.max() else 'black')
            ax.set_xticks(range(len(groups)), groups, rotation=60, ha='right', rotation_mode='anchor')
            for lbl in ax.get_xticklabels():
                lbl.set_color(gcol(lbl.get_text()) if lbl.get_text() != NONE else '0.4')
            ax.set_yticks(range(len(targets)), targets)
            ax.set_title(title)
            ax.spines[:].set_visible(False)
            ax.tick_params(length=0)
        axs[1].set_yticklabels([])
        fig.tight_layout()
        out = pd.concat({'track_mm_per_probe': trk_per_probe.stack(), 'units': unt.stack()}, axis=1)
        save(fig, '3a_coverage_heatmaps', out.reset_index().rename(columns={'level_1': 'area_group'}))
        fig3_alluvial(unt, targets, groups)


def fig3_alluvial(unt, targets, groups):
    """Units flowing from target region (left) to area group (right); band width = units."""
    total = unt.to_numpy().sum()
    pad = 0.012 * total
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10 / 2.54, 9 / 2.54))
        def stack(values, order):
            pos, y = {}, 0
            for k in order:
                pos[k] = y; y += values[k] + pad
            return pos
        left_tot, right_tot = unt.sum(axis=1), unt.sum(axis=0)
        lpos, rpos = stack(left_tot, targets), stack(right_tot, groups)
        loff, roff = dict(lpos), dict(rpos)
        for t in targets:
            for gr in groups:
                w = unt.loc[t, gr]
                if w <= 0:
                    continue
                y0l, y0r = loff[t], roff[gr]
                verts = [(0.1, y0l), (0.5, y0l), (0.5, y0r), (0.9, y0r), (0.9, y0r + w), (0.5, y0r + w),
                         (0.5, y0l + w), (0.1, y0l + w), (0.1, y0l)]
                codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
                         MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
                ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=gcol(gr), alpha=0.55, edgecolor='none'))
                loff[t] += w; roff[gr] += w
        for t in targets:
            ax.add_patch(plt.Rectangle((0.07, lpos[t]), 0.03, left_tot[t], color=TARGET_COLORS.get(t, '0.3')))
            ax.text(0.06, lpos[t] + left_tot[t] / 2, f'{t} ({left_tot[t]:,})', ha='right', va='center')
        for gr in groups:
            if right_tot[gr] <= 0:
                continue
            ax.add_patch(plt.Rectangle((0.9, rpos[gr]), 0.03, right_tot[gr], color=gcol(gr)))
            ax.text(0.94, rpos[gr] + right_tot[gr] / 2, f'{gr} ({right_tot[gr]:,})', ha='left', va='center',
                    color=gcol(gr) if gr != NONE else '0.4')
        ax.set_xlim(-0.25, 1.35); ax.set_ylim(max(max(loff.values()), max(roff.values())) + pad, -pad)
        ax.axis('off')
        ax.set_title('Units: target region → area group')
        save(fig, '3b_alluvial_target_to_area_group',
             unt.stack().rename('units').reset_index().rename(columns={'level_1': 'area_group'}))


def fig4_projections(pts, outl):
    entry = (pts[pts['area_group'] != NONE].sort_values('point').groupby(['session', 'probe']).head(1))
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(18 / 2.54, 6 / 2.54), gridspec_kw={'width_ratios': [0.9, 1.3, 1]})
        outl.draw(axs[0], 'top')
        for t, e in entry.groupby('target_region'):
            axs[0].scatter(e['ml'], e['ap'], s=4, color=TARGET_COLORS.get(t, '0.3'), label=t, lw=0)
        axs[0].legend(frameon=False, fontsize=4.5, markerscale=1.5, loc='lower right', handletextpad=0.1)
        axs[0].set_title('Entry points (first labelled point)')
        for ax, view in [(axs[1], 'sagittal'), (axs[2], 'coronal')]:
            outl.draw(ax, view)
            segs, cols = track_segments(pts, view)
            ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.4, rasterized=False))
            ax.set_title(f'Tracks, {view} projection')
        group_legend(axs[2], group_order(set(pts['area_group'])), loc='upper left', bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        save(fig, '4_projections_tracks', entry[['mouse_id', 'session', 'probe', 'target_region', 'ap', 'dv', 'ml',
                                                 'area_acronym_custom']])


def fig5_units(units, outl):
    u = units.rename(columns={'ccf_atlas_ap': 'ap', 'ccf_atlas_dv': 'dv', 'ccf_atlas_ml': 'ml'}).dropna(
        subset=['ap', 'dv', 'ml'])
    u = u.sample(frac=1, random_state=0)  # random draw order so no group sits on top
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 3, figsize=(18 / 2.54, 6 / 2.54), gridspec_kw={'width_ratios': [0.9, 1.3, 1]})
        for ax, view in zip(axs, ['top', 'sagittal', 'coronal']):
            outl.draw(ax, view)
            x, y = view_xy(u, view)
            ax.scatter(u[x], u[y], s=0.3, c=[gcol(gr) for gr in u['area_group']], lw=0, alpha=0.5, rasterized=True)
            ax.set_title(f'Units ({len(u):,}), {view}')
        group_legend(axs[2], group_order(set(u['area_group'])), loc='upper left', bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        save(fig, '5a_units_projections')
    counts = units.groupby('ccf_atlas_acronym').size().rename('units').reset_index()
    try:
        from iblatlas.atlas import BrainRegions
        from iblatlas.plots import plot_swanson_vector
        br = BrainRegions()
        beryl = br.acronym2acronym(counts['ccf_atlas_acronym'].to_numpy(), mapping='Beryl')
        bc = pd.DataFrame({'beryl': beryl, 'units': counts['units']}).groupby('beryl')['units'].sum()
        bc = bc[~bc.index.isin(['void', 'root'])]
        with plt.rc_context(RC):
            fig, ax = plt.subplots(figsize=(12 / 2.54, 7 / 2.54))
            plot_swanson_vector(bc.index.to_numpy(), np.log10(bc.to_numpy()), cmap='magma_r', br=br, ax=ax,
                                empty_color='white', linewidth=0.1)
            ax.axis('off')
            sm = plt.cm.ScalarMappable(cmap='magma_r', norm=plt.Normalize(np.log10(bc.min()), np.log10(bc.max())))
            cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
            cb.set_label('log10 units (good + mua)')
            ax.set_title('Units per Allen (Beryl) area, Swanson flatmap')
            save(fig, '5b_units_swanson', bc.rename('units').reset_index())
    except Exception as err:  # needs iblatlas swanson files (downloaded once from IBL)
        print('Swanson flatmap skipped:', repr(err), flush=True)


def fig6_small_multiples(pts, outl):
    targets = [t for t in TARGET_ORDER if t in set(pts['target_region'])]
    for view in ['sagittal', 'coronal']:
        with plt.rc_context(RC):
            fig, axs = plt.subplots(3, 4, figsize=(18 / 2.54, 11 / 2.54))
            for ax, t in zip(axs.flat, targets):
                p = pts[pts['target_region'] == t]
                outl.draw(ax, view)
                segs, cols = track_segments(p, view)
                ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.6))
                n = p[['session', 'probe']].drop_duplicates().shape[0]
                ax.set_title(f'{t}  ({n} probes, {p.mouse_id.nunique()} mice)', color=TARGET_COLORS.get(t, 'k'))
                ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(''); ax.set_ylabel('')
                ax.spines[:].set_visible(False)
            for ax in list(axs.flat)[len(targets):]:
                ax.axis('off')
            group_legend(list(axs.flat)[-1], group_order(set(pts['area_group'])), loc='center')
            fig.tight_layout()
            save(fig, f'6_small_multiples_{view}')


if __name__ == '__main__':
    fig1_copy_3d()
    pts = collect_track_points()
    print('track points:', len(pts), 'probes:', len(pts[['session', 'probe']].drop_duplicates()), flush=True)
    units = load_units()
    print('units:', len(units), flush=True)
    outl = Outlines()
    fig2_stripes(pts)
    fig3_coverage(pts, units)
    fig4_projections(pts, outl)
    fig5_units(units, outl)
    fig6_small_multiples(pts, outl)
    print('ALL DONE', flush=True)
