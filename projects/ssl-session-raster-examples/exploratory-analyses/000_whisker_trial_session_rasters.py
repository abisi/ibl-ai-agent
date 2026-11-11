"""Example session rasters around a whisker trial, for sessions with >= 4 valid probes.

Runs brain_wide_analysis.plotting_functions.plot_session_raster (unchanged) twice per session on one epoch
around the whisker hit trial with the largest population response, and puts both versions side by side:
  left  = default look, all spikes black (area colours switched off);
  right = spikes coloured by area_group (ephys_utilities allen_utils.get_custom_area_groups_colors).

Units: unit set "all" (quality_label good/mua) from 137_stable_units.parquet. Probes counted as distinct imec
electrode groups with >= 1 unit of that set. DLC traces via ephys_utilities dlc_helpers (not NWB_reader).
Output: combined_results_ks4/<mouse>/whisker_<day>/session_raster/ (per session) and a ranking table in
combined_results_ks4/ssl-session-raster-examples/tables/.

Usage: python 000_whisker_trial_session_rasters.py [session_id ...]   (no argument: all sessions)
"""
import os
import sys
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'scripts'))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

for p in [axel_bisi_path('brain_wide_analysis'), axel_bisi_path('Github', 'ephys_utilities'),
          axel_bisi_path('NWB_reader')]:
    sys.path.insert(0, str(p))
sys.path.append(str(axel_bisi_path('Github', 'allen_utils')))  # standalone copy, imported by brain_wide_analysis.neural_utils
import plotting_functions as pf  # noqa: E402
import NWB_reader_functions as nwb_reader  # noqa: E402
import ephys_utilities.helpers.data_utils as data_utils  # noqa: E402
import ephys_utilities.helpers.dlc_helpers as dlc  # noqa: E402
import ephys_utilities.allen_utils.allen_utils as allen_utils  # noqa: E402

SLUG = 'ssl-session-raster-examples'
RESULTS = axel_bisi_path('combined_results_ks4')
NWB_DIR = axel_bisi_path('NWB_ks4')
# unit set 'all' of ssl-whisker-hitmiss-timeresolved-decoding/tables/137_stable_units.parquet, exported to csv
# (the brain_wide_analysis env has no parquet engine)
UNITS_TABLE = RESULTS / SLUG / 'tables' / 'units_all_from_137.csv.gz'
MIN_PROBES = 4

T_PRE, T_POST = 1.0, 3.0           # plotted epoch around the selected whisker stimulus (s)
SCORE_PRE = (-0.5, 0.0)            # baseline window for trial scoring (s)
SCORE_POST = (0.0, 0.3)            # response window for trial scoring (s)

TRIAL_COLORS = dict(auditory_reward='#0e458a', auditory_no_reward='#3d8beb', whisker_reward='#238443',
                    whisker_no_reward='#bf3435', no_stim='#262626', correct_rejection='#737373',
                    no_stim_no_reward='#737373')
RASTER_PARAMS = {'marker_size': 3.5, 'marker_alpha': 0.7, 'combine_probes': True, 'order_type': 'neuron_id',
                 'behaviour_flag': True, 'area_legend_flag': False, 'video_start_time': 0.0}


def valid_sessions():
    """Sessions with >= MIN_PROBES probes carrying >= 1 unit of the 'all' unit set."""
    units = pd.read_csv(UNITS_TABLE)
    units['probe'] = units['electrode_group'].str.split('_').str[0]
    counts = units.groupby('session_id').agg(mouse_id=('mouse_id', 'first'), n_probes=('probe', 'nunique'),
                                             n_units=('cluster_id', 'size'))
    return units, counts[counts['n_probes'] >= MIN_PROBES].sort_values('n_units', ascending=False)


class _BlackColors(dict):
    """Stand-in for get_custom_area_groups_colors(): every group black (default, uncoloured raster)."""
    def get(self, key, default=None):
        return 'k'


def dlc_time_series(nwb_file):
    """Series dict in the plot_session_raster format ({name: {'timestamps', 'data'}}) from dlc_helpers."""
    views = dlc.load_dlc(nwb_file)
    side, top = views['side'], views['top']
    out = {}
    for name in ['jaw_distance', 'tongue_distance']:
        out[name] = {'timestamps': side.timestamps, 'data': dlc.clean_series(side, name)[0].copy()}
    out['pupil_area'] = {'timestamps': side.timestamps, 'data': dlc.clean_series(side, 'pupil_area')[0].copy()}
    out['whisker_angle'] = {'timestamps': top.timestamps, 'data': dlc.clean_series(top, 'whisker_angle')[0].copy()}
    nose, ts = dlc.nose_norm(views)
    out['nose_norm_distance'] = {'timestamps': ts, 'data': nose.copy()}
    out['tongue_likelihood'] = {'timestamps': side.timestamps,
                                'data': side.likelihood['tongue_likelihood'].to_numpy()}
    # pupil QC already applied by clean_series: no second likelihood mask inside plot_session_raster
    out['pupil_likelihood'] = {'timestamps': side.timestamps, 'data': np.ones(len(side.timestamps))}
    return out


def score_whisker_trials(trial_table, unit_table):
    """Population rate change (Hz, summed over units) after vs before each active whisker hit trial."""
    trials = trial_table[(trial_table['whisker_stim'] == 1) & (trial_table['lick_flag'] == 1)
                         & (trial_table['context'] == 'active') & (trial_table['perf'] != 6)].copy()
    spikes = np.sort(np.concatenate(unit_table['spike_times'].to_numpy()))
    t0 = trials['start_time'].to_numpy()

    def count(a, b):
        return np.searchsorted(spikes, t0 + b) - np.searchsorted(spikes, t0 + a)
    pre = count(*SCORE_PRE) / (SCORE_PRE[1] - SCORE_PRE[0])
    post = count(*SCORE_POST) / (SCORE_POST[1] - SCORE_POST[0])
    trials['pop_rate_pre'] = pre
    trials['pop_rate_post'] = post
    trials['score'] = post - pre
    return trials.sort_values('score', ascending=False)


def render(trial_table, unit_table, events, series, t_stim, coloured):
    params = dict(RASTER_PARAMS, time_start=t_stim - T_PRE, time_stop=t_stim + T_POST)
    original = pf.allen_utils.get_custom_area_groups_colors
    if not coloured:
        pf.allen_utils.get_custom_area_groups_colors = lambda: _BlackColors()
    try:
        series = {k: {'timestamps': v['timestamps'], 'data': v['data'].copy()} for k, v in series.items()}
        fig = pf.plot_session_raster(trial_table, unit_table, events, series, params=params,
                                     color_palette=TRIAL_COLORS, saving_path=None)
    finally:
        pf.allen_utils.get_custom_area_groups_colors = original
    # Probe boundary lines / labels are drawn at cumulative per-probe counts, but units are sorted by area_group:
    # they do not mark real probe boundaries, so they are removed here.
    ax = fig.axes[1]
    for line in list(ax.lines):
        if line.get_linestyle() == '--':
            line.remove()
    for txt in list(ax.texts):
        txt.remove()
    return fig


def fig_to_array(fig, dpi=300):
    fig.set_dpi(dpi)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return img


def run_session(session_id, units_all):
    nwb_file = NWB_DIR / f'{session_id}.nwb'
    res = data_utils.process_single_nwb(str(nwb_file), day_to_analyze='all')
    trial_table, unit_table = res['trial_table'], res['unit_table']
    beh, day = nwb_reader.get_bhv_type_and_training_day_index(str(nwb_file))
    unit_table = allen_utils.process_allen_labels(unit_table, split_merge_areas=True)

    # Unit set 'all' from the canonical table (dataset cluster_id = probe index * 1e6 + NWB cluster id)
    keep = units_all[units_all['session_id'] == session_id][['electrode_group', 'nwb_cluster_id']]
    eg = unit_table['electrode_group'].astype(str)
    eg = eg.where(eg.str.contains('_shank'), eg + '_shank0')
    key = pd.MultiIndex.from_arrays([eg, unit_table['cluster_id'].astype(int)])
    sel = key.isin(pd.MultiIndex.from_arrays([keep['electrode_group'], keep['nwb_cluster_id'].astype(int)]))
    if sel.sum() != len(keep):
        print(f'{session_id}: matched {sel.sum()} of {len(keep)} units of the unit table')
    unit_table = unit_table[sel].copy()
    unit_table['quality_label'] = 'good'  # all kept units are good/mua; plot_session_raster filters on this column

    trials = score_whisker_trials(trial_table, unit_table)
    events = nwb_reader.get_behavioral_events(str(nwb_file))
    series = dlc_time_series(str(nwb_file))

    best = trials.iloc[0]
    t_stim = float(best['start_time'])
    imgs = [fig_to_array(render(trial_table, unit_table, events, series, t_stim, c)) for c in (False, True)]

    fig, axs = plt.subplots(1, 2, figsize=(12.5, 8), dpi=150, gridspec_kw={'wspace': 0.0})
    for ax, img, title in zip(axs, imgs, ['Default', 'Coloured by area group']):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title, fontsize=10)
    groups = unit_table['area_acronym_custom'].map(allen_utils.get_custom_area_groups_from_name()).fillna('Other')
    colours = allen_utils.get_custom_area_groups_colors()
    present = sorted(groups.unique())
    handles = [Line2D([0], [0], marker='o', ls='', markerfacecolor=colours.get(g, 'grey'), markeredgewidth=0,
                      markersize=6, label=f'{g} ({(groups == g).sum()})') for g in present]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7,
               title='Area group (n units)', title_fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'{session_id}  whisker day {day}  trial {int(best["trial_id"]) if "trial_id" in best else best.name}'
                 f'  stim t = {t_stim:.2f} s  |  {len(unit_table)} units',
                 fontsize=10)
    out_dir = RESULTS / session_id.split('_')[0] / f'{beh}_{day}' / 'session_raster'
    os.makedirs(out_dir, exist_ok=True)
    stem = f'session_raster_whisker_trial_{session_id}_t{t_stim:.0f}s_default_vs_area_group'
    for fmt in ['png', 'pdf']:
        fig.savefig(out_dir / f'{stem}.{fmt}', bbox_inches='tight', dpi=300)
    plt.close(fig)

    trials.head(10).assign(session_id=session_id).to_csv(out_dir / f'{stem}_top_trials.csv', index=False)
    return dict(session_id=session_id, mouse_id=session_id.split('_')[0], day=day,
                n_units=len(unit_table), n_area_groups=len(present), t_stim=t_stim,
                score=best['score'], pop_rate_pre=best['pop_rate_pre'], pop_rate_post=best['pop_rate_post'],
                figure=str(out_dir / f'{stem}.png'))


def main():
    units_all, sessions = valid_sessions()
    todo = sys.argv[1:] or list(sessions.index)
    print(f'{len(sessions)} sessions with >= {MIN_PROBES} probes; running {len(todo)} on {socket.gethostname()}')
    rows = []
    for sid in todo:
        try:
            rows.append(run_session(sid, units_all))
            print('done', sid, rows[-1]['figure'])
        except Exception as e:
            print('FAILED', sid, repr(e))
            rows.append(dict(session_id=sid, error=repr(e)))
    tab_dir = RESULTS / SLUG / 'tables'
    os.makedirs(tab_dir, exist_ok=True)
    name = 'session_ranking.csv' if not sys.argv[1:] else f'session_ranking_{"_".join(todo)}.csv'
    pd.DataFrame(rows).to_csv(tab_dir / name, index=False)


if __name__ == '__main__':
    main()
