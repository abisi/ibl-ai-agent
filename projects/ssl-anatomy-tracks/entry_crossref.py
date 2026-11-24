"""Cross-reference probe entry points and track labels between sample space (traced on the mouse's own brain,
labels from registered_atlas.tiff) and atlas space (segmentation/{sample_space,atlas_space}/tracks/imecN.csv).
Output: anatomy/_viz_trials_20261116/4c_entry_crossref.{png,svg,csv}"""
import contextlib
import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainglobe_atlasapi import BrainGlobeAtlas

import ephys_utilities.allen_utils.allen_utils as au
from probe_viz_trials import OUT, RC, NONE, TARGET_ORDER, TARGET_COLORS, gcol, group_order, group_legend, save

FLAGGED = pd.read_csv(OUT / '4b_insertion_points_centroids.csv')
EXPECTED = {'wS1': 'Somatosensory-whisker', 'wS2': 'Somatosensory-whisker', 'A1': 'Auditory areas',
            'PPC': 'Posterior parietal areas', 'DLS': 'Striatum', 'wM1': 'Motor areas', 'wM2': 'Motor areas',
            'tjM1': 'Motor areas', 'ALM': 'Motor areas', 'OFC': 'Frontal areas', 'SC': 'Midbrain'}
N_BINS = 100  # tracks compared at 100 relative positions (sample and atlas spacing differ)
atlas = BrainGlobeAtlas('allen_mouse_bluebrain_barrels_10um')
EXCLUDED = set(au.get_excluded_areas())


def parent(acr):
    try:
        path = atlas.structures[acr]['structure_id_path']
        return atlas.structures[path[-2]]['acronym'] if len(path) > 1 else acr
    except KeyError:
        return acr


def label_track(csv_path, target, mouse):
    t = pd.read_csv(csv_path)
    acr = t['Region acronym'].fillna('root').astype(str).to_numpy()
    df = pd.DataFrame({'_row': range(len(acr)), 'cluster_id': range(len(acr)), 'ccf_atlas_acronym': acr,
                       'ccf_atlas_parent_acronym': [parent(a) for a in acr], 'target_region': target,
                       'ccf_atlas_ap': 0.0, 'ccf_atlas_ml': 0.0, 'ccf_atlas_dv': 0.0, 'mouse_id': mouse})
    with contextlib.redirect_stdout(io.StringIO()):
        lab = au.process_allen_labels(df, split_merge_areas=False)
    lab = lab.set_index('_row').reindex(range(len(acr)))
    grp = lab['area_group'].where(lab['area_group'].notna() & (lab['area_group'] != 'Other'), NONE).to_numpy()
    no_layer = [parent(a) if au.contains_layer(a) else a for a in acr]
    return pd.DataFrame({'acronym': acr, 'acronym_no_layer': no_layer, 'area_group': grp})


def entry(lab):
    inside = [i for i, a in enumerate(lab['acronym']) if a not in EXCLUDED and a not in ('root', 'Not found in brain')]
    return (lab.iloc[inside[0]] if inside else None)


def resample(groups, n=N_BINS):
    idx = np.minimum((np.arange(n) + 0.5) / n * len(groups), len(groups) - 1).astype(int)
    return np.asarray(groups)[idx]


pts = pd.read_parquet(OUT / 'track_points.parquet')
probes = pts.groupby(['session', 'probe']).agg(mouse_id=('mouse_id', 'first'),
                                                target_region=('target_region', 'first')).reset_index()
rows, profiles = [], {}
for r in probes.itertuples():
    base = Path('/mnt/lsens-analysis') / \
        ('Axel_Bisi' if r.mouse_id.startswith('AB') else 'Myriam_Hamon') / 'data' / r.mouse_id / r.session
    seg = base / 'Anatomy' / 'fused' / 'registered' / 'segmentation'
    s_csv, a_csv = seg / 'sample_space' / 'tracks' / f'{r.probe}.csv', seg / 'atlas_space' / 'tracks' / f'{r.probe}.csv'
    if not (s_csv.exists() and a_csv.exists()):
        rows.append(dict(session=r.session, probe=r.probe, mouse_id=r.mouse_id, target_region=r.target_region,
                         status='no sample_space track'))
        continue
    s_lab, a_lab = label_track(s_csv, r.target_region, r.mouse_id), label_track(a_csv, r.target_region, r.mouse_id)
    s_e, a_e = entry(s_lab), entry(a_lab)
    s_res, a_res = resample(s_lab['area_group']), resample(a_lab['area_group'])
    exp = EXPECTED.get(r.target_region)
    profiles[(r.session, r.probe)] = (s_res, a_res)
    rows.append(dict(
        session=r.session, probe=r.probe, mouse_id=r.mouse_id, target_region=r.target_region, status='ok',
        expected_group=exp,
        entry_sample=None if s_e is None else s_e['acronym'], entry_atlas=None if a_e is None else a_e['acronym'],
        entry_group_sample=None if s_e is None else s_e['area_group'],
        entry_group_atlas=None if a_e is None else a_e['area_group'],
        entry_area_agree=(s_e is not None and a_e is not None and s_e['acronym_no_layer'] == a_e['acronym_no_layer']),
        group_agreement=float(np.mean(s_res == a_res)),
        hits_target_sample=bool(exp in set(s_lab['area_group'])), hits_target_atlas=bool(exp in set(a_lab['area_group'])),
        groups_sample=' > '.join(pd.unique(s_lab['area_group'][s_lab['area_group'] != NONE])),
        groups_atlas=' > '.join(pd.unique(a_lab['area_group'][a_lab['area_group'] != NONE]))))
    print('  ', r.session, r.probe, flush=True)

tab = pd.DataFrame(rows).merge(FLAGGED[['session', 'probe', 'dist_mm', 'off_target']], on=['session', 'probe'],
                               how='left')
ok = tab[tab['status'] == 'ok']
print(f"\n{len(ok)}/{len(tab)} probes with sample-space tracks; entry area agrees (no layer): "
      f"{ok.entry_area_agree.mean():.0%}; median group agreement along track: {ok.group_agreement.median():.0%}; "
      f"hits target group: sample {ok.hits_target_sample.mean():.0%}, atlas {ok.hits_target_atlas.mean():.0%}")
cols = ['session', 'probe', 'target_region', 'dist_mm', 'entry_sample', 'entry_atlas', 'group_agreement',
        'hits_target_sample', 'hits_target_atlas', 'groups_sample']
print('\nFLAGGED:'); print(tab[tab.off_target == True][cols].sort_values('dist_mm', ascending=False).to_string(index=False))
print('\nNOT FLAGGED but missing target group in sample space:')
print(ok[(ok.off_target != True) & ~ok.hits_target_sample][cols].to_string(index=False))

# figure: flagged probes, sample vs atlas space track profiles; right: group agreement for all probes
fl = tab[(tab.off_target == True) & (tab.status == 'ok')].sort_values('dist_mm', ascending=False)
with plt.rc_context(RC):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(18 / 2.54, 8.5 / 2.54), gridspec_kw={'width_ratios': [2, 1]})
    for i, r in enumerate(fl.itertuples()):
        s_res, a_res = profiles[(r.session, r.probe)]
        for j, (prof, dx) in enumerate([(s_res, -0.2), (a_res, 0.2)]):
            ax.bar(np.full(N_BINS, i + dx), np.ones(N_BINS), bottom=np.arange(N_BINS), width=0.36,
                   color=[gcol(x) for x in prof], edgecolor='none')
        ax.text(i, -3, f"{'OK' if r.hits_target_sample else 'MISS'}", ha='center', va='bottom', fontsize=5,
                fontweight='bold', color='0.3' if r.hits_target_sample else 'red')
    ax.set_xticks(range(len(fl)), [f'{r.session}\n{r.probe}  {r.target_region}' for r in fl.itertuples()],
                  rotation=60, ha='right', rotation_mode='anchor', fontsize=4.5)
    ax.set_ylim(N_BINS, -8)
    ax.set_yticks([0, 50, 100], ['top', 'mid', 'tip'])
    ax.set_ylabel('Relative position along track')
    ax.set_title('Flagged insertions: sample space (left bar) vs atlas space (right bar)\n'
                 'OK / MISS = track passes through the target area group in sample space')
    ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x', length=0)
    present = set().union(*[set(p[0]) | set(p[1]) for k, p in profiles.items() if k in set(zip(fl.session, fl.probe))])
    group_legend(ax, group_order(present), loc='upper left', bbox_to_anchor=(1.0, 1.0))
    others = ok[ok.off_target != True]
    bx.hist(others['group_agreement'], bins=np.linspace(0, 1, 21), color='0.75', label='other probes')
    for r in fl.itertuples():
        bx.axvline(r.group_agreement, color='red', lw=0.5)
    bx.set_xlabel('Area-group agreement along track\n(sample vs atlas space)')
    bx.set_ylabel('Probes')
    bx.set_title('Red lines: flagged insertions')
    bx.legend(frameon=False, fontsize=5, loc='upper left')
    fig.tight_layout()
    save(fig, '4c_entry_crossref', tab)
