"""Insertion (entry) points per probe, top view in bregma coordinates, centroid per target, off-target insertions
flagged and labelled with their session. Output: anatomy/_viz_trials_20261116/4b_insertion_points_centroids.*"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from probe_viz_trials import OUT, RC, TARGET_ORDER, TARGET_COLORS, Outlines, save

BREGMA = dict(ap=5400, ml=5739)          # IBL bregma estimate in CCF um (as allen_utils)
MAD_K, MIN_MM = 3.0, 0.4                 # flag: d > median + 3 robust SD and d > 0.4 mm

pts = pd.read_parquet(OUT / 'track_points.parquet')
e = (pts[pts['area_acronym_custom'] != 'none'].sort_values('point')
     .groupby(['session', 'probe']).head(1).copy())      # first point inside a labelled brain area
e['ap_mm'] = (BREGMA['ap'] - e['ap']) / 1000              # anterior positive
e['ml_mm'] = (e['ml'] - BREGMA['ml']) / 1000              # lateral positive (this hemisphere)
targets = [t for t in TARGET_ORDER if t in set(e['target_region'])]

cent = e.groupby('target_region')[['ml_mm', 'ap_mm']].median()
e = e.join(cent, on='target_region', rsuffix='_centroid')
e['dist_mm'] = np.hypot(e['ml_mm'] - e['ml_mm_centroid'], e['ap_mm'] - e['ap_mm_centroid'])
stats = e.groupby('target_region')['dist_mm'].agg(
    med='median', mad=lambda d: 1.4826 * np.median(np.abs(d - np.median(d))))
stats['threshold_mm'] = np.maximum(stats['med'] + MAD_K * stats['mad'], MIN_MM)
e = e.join(stats['threshold_mm'], on='target_region')
e['off_target'] = e['dist_mm'] > e['threshold_mm']
e['label'] = e['session'] + ' ' + e['probe']
off = e[e['off_target']].sort_values('dist_mm', ascending=False).reset_index(drop=True)
off['n'] = off.index + 1
print(f'{len(off)} of {len(e)} insertions flagged:')
print(off[['target_region', 'label', 'ml_mm', 'ap_mm', 'dist_mm', 'threshold_mm']].round(2).to_string(index=False))

outl = Outlines()
top = outl.top                                            # AP x ML (25 um voxels)
ml_mm = (np.arange(top.shape[1]) * outl.res - BREGMA['ml']) / 1000
ap_mm = (BREGMA['ap'] - np.arange(top.shape[0]) * outl.res) / 1000

with plt.rc_context(RC):
    fig, (ax, lx, bx) = plt.subplots(1, 3, figsize=(18 / 2.54, 9 / 2.54), gridspec_kw={'width_ratios': [1.25, 0.55, 1]})
    ax.contour(ml_mm, ap_mm, top, levels=[0.5], colors='0.6', linewidths=0.5)
    ax.axvline(0, color='0.85', lw=0.4, zorder=0)
    ax.axhline(0, color='0.85', lw=0.4, zorder=0)
    for t in targets:
        d = e[e['target_region'] == t]
        c = TARGET_COLORS.get(t, '0.3')
        ax.scatter(d['ml_mm'], d['ap_mm'], s=6, color=c, lw=0, alpha=0.8, label=f'{t} (n={len(d)})')
        cx, cy = cent.loc[t]
        ax.scatter(cx, cy, s=40, marker='X', color=c, edgecolor='k', lw=0.5, zorder=4)
        ax.text(cx, cy + 0.17, t, color=c, ha='center', va='bottom', fontsize=5.5, fontweight='bold', zorder=5)
    for r in off.itertuples():
        ax.scatter(r.ml_mm, r.ap_mm, s=30, facecolor='none', edgecolor='red', lw=0.6, zorder=5)
        ax.plot([r.ml_mm, r.ml_mm_centroid], [r.ap_mm, r.ap_mm_centroid], color='red', lw=0.4, ls=':', zorder=3)
        ax.text(r.ml_mm - 0.12, r.ap_mm + 0.12, str(r.n), color='red', fontsize=5, fontweight='bold',
                ha='right', va='bottom', zorder=6)
    ax.set_xlim(-0.5, 6.2); ax.set_ylim(-5.0, 3.8)
    ax.set_aspect('equal')
    ax.set_xlabel('ML from bregma (mm)'); ax.set_ylabel('AP from bregma (mm, anterior up)')
    ax.set_title(f'Insertion points ({len(e)} probes), X = target centroid (median)')
    lx.axis('off')
    h, l = ax.get_legend_handles_labels()
    lx.legend(h, l, frameon=False, fontsize=5, loc='upper left', handletextpad=0.1, markerscale=1.3,
              title='Target', title_fontsize=5.5, alignment='left')
    lines = [f'{r.n}  {r.session} {r.probe}\n    {r.target_region}, {r.dist_mm:.2f} mm' for r in off.itertuples()]
    lx.text(0.0, 0.47, 'Flagged insertions', fontsize=5.5, fontweight='bold', transform=lx.transAxes, va='top')
    lx.text(0.0, 0.43, '\n'.join(lines), fontsize=4.5, color='red', transform=lx.transAxes, va='top',
            linespacing=1.25)

    xs = {t: i for i, t in enumerate(targets)}
    jit = np.random.default_rng(0).uniform(-0.18, 0.18, len(e))
    bx.scatter(e['target_region'].map(xs) + jit, e['dist_mm'], s=5, lw=0,
               color=[TARGET_COLORS.get(t, '0.3') for t in e['target_region']])
    for t in targets:
        bx.plot([xs[t] - 0.35, xs[t] + 0.35], [stats.loc[t, 'threshold_mm']] * 2, color='red', lw=0.6)
    for r in off.itertuples():
        bx.annotate(str(r.n), (xs[r.target_region], r.dist_mm), xytext=(4, 0), textcoords='offset points',
                    fontsize=5, fontweight='bold', color='red', va='center')
    bx.set_xticks(range(len(targets)), targets, rotation=45, ha='right', rotation_mode='anchor')
    for lbl in bx.get_xticklabels():
        lbl.set_color(TARGET_COLORS.get(lbl.get_text(), 'k'))
    bx.set_ylabel('Distance to target centroid (mm)')
    bx.set_title(f'Red line: median + {MAD_K:g} robust SD (min {MIN_MM} mm)')
    bx.set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, '4b_insertion_points_centroids',
         e[['mouse_id', 'session', 'probe', 'target_region', 'ml_mm', 'ap_mm', 'dv', 'area_acronym_custom',
            'ml_mm_centroid', 'ap_mm_centroid', 'dist_mm', 'threshold_mm', 'off_target']]
         .rename(columns={'dv': 'dv_ccf_um', 'area_acronym_custom': 'entry_area'})
         .sort_values(['target_region', 'dist_mm'], ascending=[True, False]))
