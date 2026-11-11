"""Publication-style whole-brain raster around one whisker hit trial (style of the Figma schematic
manuscripts/Bisi_Hamon/Figures/FromFigma/schematic_rec_session_raster.pdf).

Two versions side by side on one Nature-width figure: (left) black spikes, (right) spikes coloured by area_group;
both with the area-group bar and labels on the left, lick ticks on top, DLC traces below (jaw, tongue, whisking,
pupil, snout = side-camera nose_distance), trial start / jaw opening / lick onset lines and scale bars.

Data: as 000 (unit set 'all' from 137_stable_units, artifact-corrected spike times, dlc_helpers series).
Jaw opening: dlc_helpers.movement_onset_times on the cleaned jaw_distance (behaviour_analysis defaults).
Lick onset: first lick of the trial, corrected_lick = start_time + (lick_time - response_window_start_time).

Usage: python 001_paper_style_whisker_trial_raster.py [session_id ...]   (no argument: all sessions >= 4 probes)
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('rasters000', HERE / '000_whisker_trial_session_rasters.py')
r0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r0)
sys.path.insert(0, str(r0.REPO / 'projects' / 'ssl-paper-figures'))
import paperstyle  # noqa: E402

dlc, allen_utils, nwb_reader, data_utils = r0.dlc, r0.allen_utils, r0.nwb_reader, r0.data_utils

T_PRE, T_POST = 0.5, 2.0                 # plotted epoch around the whisker stimulus (s)
WHISKER_COLOR = paperstyle.COLORS['whisker']
LICK_COLOR = '#1e90ff'
MIN_BOUT_LICKS, BOUT_S = 5, 2.0      # preferred trials: >= 5 piezo licks within 2 s of lick onset
N_EXAMPLES = 5                       # example trials per session (best first)
PRE_LICK_TOL_S = 0.03                # ... and no piezo lick from the window start to lick onset - 30 ms

# Top-to-bottom order of area groups in the raster: allen_utils.get_area_group_custom_order(); short labels
SPIKE_SIZE, SPIKE_ALPHA = 0.45, 0.6     # raster marker area (pt^2) and transparency
RASTER_LAYER_DPI = 1200                 # resolution of the rasterized spike layer in PDF / SVG / EPS / AI
AREA_SHORT = {'Thalamus': 'TH', 'Midbrain': 'MB', 'Pons and medulla': 'HB', 'Amygdala and hypothalamus': 'AMY/HY',
              'Lateral septal complex': 'LSX', 'Pallidum': 'PAL', 'Striatum': 'STR', 'Hippocampus': 'HPF',
              'Cortical subplate': 'CTXsp', 'Olfactory areas': 'OLF', 'Somatosensory-whisker': 'SSp-bfd',
              'Somatosensory-orofacial': 'SSp-orof', 'Somatosensory-body': 'SSp-body', 'Somatosensory areas': 'SS',
              'Auditory areas': 'AUD', 'Visual areas': 'VIS', 'Posterior parietal areas': 'PTLp',
              'Retrosplenial areas': 'RSP', 'Insular areas': 'AI', 'Frontal areas': 'FRO', 'Motor areas': 'MO',
              'Motor and frontal areas': 'MO/FRO'}
TRACES = [('jaw_distance', 'Jaw'), ('tongue_distance', 'Tongue'), ('whisker_angle', 'Whisking'),
          ('pupil_area', 'Pupil size'), ('nose_side_distance', 'Snout')]

# Layout of one version (mm)
W_COL = 56.0          # width of the time axis
H_LICK, H_RASTER, H_TRACE, GAP = 3.0, 72.0, 6.0, 1.0
X_AREA_BAR = 1.2      # width of the area colour bar


def load_session(session_id, units_all):
    nwb_file = str(r0.NWB_DIR / f'{session_id}.nwb')
    res = data_utils.process_single_nwb(nwb_file, day_to_analyze='all')
    trial_table, unit_table = res['trial_table'], res['unit_table']
    beh, day = nwb_reader.get_bhv_type_and_training_day_index(nwb_file)
    unit_table = allen_utils.process_allen_labels(unit_table, split_merge_areas=True)
    keep = units_all[units_all['session_id'] == session_id][['electrode_group', 'nwb_cluster_id']]
    eg = unit_table['electrode_group'].astype(str)
    eg = eg.where(eg.str.contains('_shank'), eg + '_shank0')
    key = pd.MultiIndex.from_arrays([eg, unit_table['cluster_id'].astype(int)])
    sel = key.isin(pd.MultiIndex.from_arrays([keep['electrode_group'], keep['nwb_cluster_id'].astype(int)]))
    unit_table = unit_table[sel].copy()
    unit_table['area_group'] = unit_table['area_acronym_custom'].map(
        allen_utils.get_custom_area_groups_from_name()).fillna('Other')

    views = dlc.load_dlc(nwb_file)
    side, top = views['side'], views['top']
    series = {}
    for name in ['jaw_distance', 'tongue_distance']:
        series[name] = (side.timestamps, dlc.clean_series(side, name)[0])
    tongue_out = dlc.tongue_presence(side) == 1
    series['tongue_distance'][1][~tongue_out] = np.nan
    series['pupil_area'] = (side.timestamps, dlc.clean_series(side, 'pupil_area')[0])
    series['nose_side_distance'] = (side.timestamps, dlc.clean_series(side, 'nose_distance')[0])
    series['whisker_angle'] = (top.timestamps, dlc.clean_series(top, 'whisker_angle')[0])
    jaw = series['jaw_distance']
    events = nwb_reader.get_behavioral_events(nwb_file)
    return dict(trial_table=trial_table, unit_table=unit_table, series=series, jaw=jaw,
                licks=np.asarray(events['piezo_lick_times']), beh=beh, day=day)


def order_units(unit_table):
    order = allen_utils.get_area_group_custom_order()
    rank = {g: i for i, g in enumerate(order)}
    ut = unit_table.assign(_rank=unit_table['area_group'].map(rank).fillna(len(order)),
                           _depth=unit_table['depth'].astype(float))
    # top of the raster = first group; within a group, deepest unit at the bottom
    ut = ut.sort_values(['_rank', 'electrode_group', '_depth'], ascending=[True, True, False])
    return ut.iloc[::-1].reset_index(drop=True)  # row 0 = bottom of the axis


def draw_version(fig, x0_mm, y0_mm, page_mm, d, trial, coloured):
    """One raster column; x0, y0 = bottom-left of the time axes block (mm from the bottom-left of the figure)."""
    pw, ph = page_mm
    t0 = trial['start_time']
    t_start, t_stop = t0 - T_PRE, t0 + T_POST
    colours = allen_utils.get_custom_area_groups_colors()

    def ax_at(y_mm, h_mm, x_mm=x0_mm, w_mm=W_COL):
        ax = fig.add_axes([x_mm / pw, y_mm / ph, w_mm / pw, h_mm / ph])
        ax.set_axis_off()
        return ax

    n_tr = len(TRACES)
    y_trace0 = y0_mm
    y_raster = y_trace0 + n_tr * H_TRACE + GAP
    y_lick = y_raster + H_RASTER + GAP
    y_top = y_lick + H_LICK

    # Raster
    ut = d['ordered_units']
    ax_r = ax_at(y_raster, H_RASTER)
    xs, ys, cs = [], [], []
    for i, row in enumerate(ut.itertuples()):
        st = row.spike_times
        st = st[(st >= t_start) & (st <= t_stop)]
        xs.append(st)
        ys.append(np.full(len(st), i))
        if coloured:
            cs.append(np.repeat([colours.get(row.area_group, '#808080')], len(st)))
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    c = np.concatenate(cs) if coloured else 'k'
    ax_r.scatter(xs, ys, s=SPIKE_SIZE, marker='o', c=c, alpha=SPIKE_ALPHA, linewidths=0, rasterized=True)
    ax_r.set_xlim(t_start, t_stop)
    ax_r.set_ylim(-0.5, len(ut) - 0.5)

    # Area bar and labels
    ax_b = ax_at(y_raster, H_RASTER, x_mm=x0_mm - X_AREA_BAR - 0.6, w_mm=X_AREA_BAR)
    ax_b.set_ylim(-0.5, len(ut) - 0.5)
    ax_b.set_xlim(0, 1)
    groups = ut['area_group'].to_numpy()
    starts = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1]
    stops = np.r_[starts[1:], len(groups)]
    min_label = 0.025 * len(ut)
    for a, b in zip(starts, stops):
        g = groups[a]
        ax_b.add_patch(plt.Rectangle((0, a - 0.5), 1, b - a, color=colours.get(g, '#808080'), lw=0))
        if b - a >= min_label:
            ax_b.text(-0.8, (a + b) / 2, AREA_SHORT.get(g, g), ha='right', va='center', fontsize=5)

    # Unit scale bar (right of the raster)
    n_bar = 500 if len(ut) >= 1500 else 200
    ax_s = ax_at(y_raster, H_RASTER, x_mm=x0_mm + W_COL + 0.8, w_mm=1.0)
    ax_s.set_ylim(-0.5, len(ut) - 0.5)
    ax_s.set_xlim(0, 1)
    ax_s.plot([0.5, 0.5], [0, n_bar], c='k', lw=1.0, solid_capstyle='butt')
    ax_s.text(1.3, n_bar / 2, f'{n_bar} units', rotation=90, va='center', ha='left', fontsize=5)

    # Licks
    ax_l = ax_at(y_lick, H_LICK)
    lk = d['licks'][(d['licks'] >= t_start) & (d['licks'] <= t_stop)]
    ax_l.vlines(lk, 0, 1, colors=LICK_COLOR, lw=0.75)
    ax_l.set_xlim(t_start, t_stop)
    ax_l.set_ylim(0, 1)
    ax_l.text(1.0, 0.5, 'Licks', transform=ax_l.transAxes, ha='left', va='center', fontsize=5)

    # DLC traces, each scaled to its own range in the epoch
    for k, (name, label) in enumerate(TRACES):
        ts, x = d['series'][name]
        m = (ts >= t_start) & (ts <= t_stop)
        ax = ax_at(y_trace0 + (n_tr - 1 - k) * H_TRACE, H_TRACE)
        ax.plot(ts[m], x[m], c='k', lw=0.9)
        if np.any(np.isfinite(x[m])):
            lo, hi = np.nanpercentile(x[m], [0.5, 99.5])
            pad = 0.1 * (hi - lo) if hi > lo else 1
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(t_start, t_stop)
        ax.text(-0.01, 0.35, label, transform=ax.transAxes, ha='right', va='center', fontsize=5)

    # Event lines through the whole column, labels below
    ax_o = ax_at(y_trace0, y_top - y_trace0)
    ax_o.set_xlim(t_start, t_stop)
    ax_o.set_ylim(0, 1)
    ax_o.axvline(t0, c=WHISKER_COLOR, lw=0.75, zorder=5)
    ev = [(t0, 'Trial start')]
    if np.isfinite(trial['jaw_onset']):
        ax_o.axvline(trial['jaw_onset'], c='k', lw=0.6, ls=(0, (2, 1.5)), zorder=5)
        ev.append((trial['jaw_onset'], 'Jaw open'))
    if np.isfinite(trial['lick_onset']):
        ax_o.axvline(trial['lick_onset'], c='k', lw=0.6, ls=(0, (2, 1.5)), zorder=5)
        ev.append((trial['lick_onset'], 'Lick onset'))
    for t, lab in ev:
        ax_o.text(t, -0.01, lab, rotation=60, ha='right', va='top', fontsize=5, clip_on=False)

    # Time scale bar (1 s, bottom right)
    ax_o.plot([t_stop - 1.0, t_stop], [-0.025, -0.025], c='k', lw=1.0, solid_capstyle='butt', clip_on=False)
    ax_o.text(t_stop - 0.5, -0.035, '1 s', ha='center', va='top', fontsize=5)

    # Title
    ax_o.text(t0 + 0.05, (y_top - y_trace0 + 2.0) / (y_top - y_trace0), 'Whisker hit trial', ha='left',
              va='bottom', fontsize=7)
    ax_o.text(x=t_stop, y=(y_top - y_trace0 + 2.0) / (y_top - y_trace0),
              s='Area-group colours' if coloured else '', ha='right', va='bottom', fontsize=5, color='0.4')


def pick_trial(d):
    trials = r0.score_whisker_trials(d['trial_table'], d['unit_table'])
    trials['lick_onset'] = trials['start_time'] + (trials['lick_time'] - trials['response_window_start_time'])
    ts, jaw = d['jaw']
    on = dlc.movement_onset_times(jaw, ts, trials)
    trials['jaw_onset'] = trials['start_time'] + on['onset'].reindex(trials.index).to_numpy()
    licks = d['licks']
    lo = trials['lick_onset'].to_numpy()
    trials['n_licks_bout'] = (np.searchsorted(licks, lo + BOUT_S) - np.searchsorted(licks, lo - 0.05))
    # piezo licks in the plotted window before the trial's lick onset (none expected on a clean hit)
    trials['n_licks_pre'] = (np.searchsorted(licks, lo - PRE_LICK_TOL_S)
                             - np.searchsorted(licks, trials['start_time'].to_numpy() - T_PRE))
    i_first = np.searchsorted(licks, trials['start_time'].to_numpy())
    trials['first_piezo_minus_onset'] = np.where(i_first < len(licks), licks[np.minimum(i_first, len(licks) - 1)],
                                                 np.nan) - lo
    # preferred: a lick bout, no piezo lick before lick onset, and jaw opening detected between stimulus and first
    # lick; then largest population response
    trials['preferred'] = ((trials['n_licks_bout'] >= MIN_BOUT_LICKS) & (trials['n_licks_pre'] == 0)
                           & (trials['jaw_onset'] < trials['lick_onset'])
                           & (trials['jaw_onset'] > trials['start_time']))
    return trials.sort_values(['preferred', 'score'], ascending=[False, False])


def save_example(d, trial, out_dir, name, full_formats):
    """Both versions side by side; full_formats: all paperstyle formats + all-vector SVG, else PNG, PDF and SVG."""
    paperstyle.set_nature_rc()
    page = (150.0, 132.0)
    fig = plt.figure(figsize=(page[0] * paperstyle.MM, page[1] * paperstyle.MM))
    for x0, coloured in [(13.0, False), (13.0 + W_COL + 17.0, True)]:
        draw_version(fig, x0, 16.0, page, d, trial, coloured)
    out_dir.mkdir(parents=True, exist_ok=True)
    if full_formats:
        paperstyle.save_figure(fig, out_dir, name, raster_dpi=600)
    else:
        fig.savefig(out_dir / f'{name}.png', dpi=600)
    # paperstyle.save_figure writes vector formats at the figure dpi (100): the rasterized spike layer was pixelated
    # in Figma. Vector formats are (re-)saved with the spike layer at RASTER_LAYER_DPI.
    for ext in ('pdf', 'svg', 'eps') if full_formats else ('pdf', 'svg'):
        fig.savefig(out_dir / f'{name}.{ext}', dpi=RASTER_LAYER_DPI)
    if full_formats:
        shutil.copyfile(out_dir / f'{name}.pdf', out_dir / f'{name}.ai')
        for coll in fig.findobj(matplotlib.collections.PathCollection):
            coll.set_rasterized(False)
        fig.savefig(out_dir / f'{name}_vector.svg')  # every spike as a vector dot
    plt.close(fig)


def run_session(session_id, units_all):
    d = load_session(session_id, units_all)
    d['ordered_units'] = order_units(d['unit_table'])
    trials = pick_trial(d)
    examples = trials[trials['preferred']].head(N_EXAMPLES)
    if examples.empty:
        examples = trials.head(1)
    out_dir = r0.RESULTS / session_id.split('_')[0] / f'{d["beh"]}_{d["day"]}' / 'session_raster'
    rows = []
    for rank, (_, trial) in enumerate(examples.iterrows(), start=1):
        name = f'paper_raster_whisker_hit_{session_id}_t{trial["start_time"]:.0f}s'
        save_example(d, trial, out_dir, name, full_formats=(rank == 1))
        rows.append(dict(session_id=session_id, rank=rank, day=d['day'], n_units=len(d['unit_table']),
                         t_stim=trial['start_time'], rt_lick=trial['lick_onset'] - trial['start_time'],
                         rt_jaw=trial['jaw_onset'] - trial['start_time'], score=trial['score'],
                         score_per_unit=trial['score'] / len(d['unit_table']), n_licks_bout=trial['n_licks_bout'],
                         n_licks_pre=trial['n_licks_pre'], first_piezo_minus_onset=trial['first_piezo_minus_onset'],
                         preferred=trial['preferred'], n_area_groups=d['unit_table']['area_group'].nunique(),
                         figure=str(out_dir / f'{name}.png')))
    cols = ['start_time', 'score', 'pop_rate_pre', 'pop_rate_post', 'lick_onset', 'jaw_onset', 'n_licks_bout',
            'n_licks_pre', 'first_piezo_minus_onset', 'preferred']
    trials[cols].head(20).assign(session_id=session_id).to_csv(
        out_dir / f'paper_raster_whisker_hit_{session_id}_top_trials.csv', index=False)
    return rows


def main():
    units_all, sessions = r0.valid_sessions()
    todo = sys.argv[1:] or list(sessions.index)
    rows = []
    for sid in todo:
        try:
            rows.extend(run_session(sid, units_all))
            print('done', sid, rows[-1]['rank'], 'examples')
        except Exception as e:
            import traceback
            traceback.print_exc()
            rows.append(dict(session_id=sid, error=repr(e)))
    tab_dir = r0.RESULTS / r0.SLUG / 'tables'
    os.makedirs(tab_dir, exist_ok=True)
    name = '001_session_ranking.csv' if not sys.argv[1:] else f'001_ranking_{"_".join(todo)}.csv'
    pd.DataFrame(rows).to_csv(tab_dir / name, index=False)


if __name__ == '__main__':
    main()
