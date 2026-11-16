"""012 -- Raw top-camera DLC traces around stimuli: is there a fixed-latency, stimulus-driven component (usable to align
the top camera), separate from the mouse's own movements? (user 2026-11-16, after 011: "there could be movements not
triggered by stimuli")

Raw NWB_ks4 pixel series (no likelihood gating or cleaning, on purpose: this is an alignment check), current NWB pairing.
Speeds = frame-to-frame displacement (px / frame) of: top_particle (the particle glued to the whisker, moved by the coil),
whisker_tip, whisker_base, top_nose_tip (top camera), jaw (side camera, behaviour reference).
Trial groups: whisker hit / whisker miss (active), whisker passive, auditory (all), no-stim (all); time 0 = stimulus
time (whisker_stim_time / auditory_stim_time / no_stim_time).
Page per session: (a) mean speed per keypoint and trial group; (b) particle-speed and nose-speed heatmaps per group
(session order); (c) raw particle x, y and likelihood of single whisker trials (early / mid / late) frame by frame.
Sessions: in-register controls (AB095, AB085, AB116, MH028 checked in 010; MH064, AB151 never flagged) and the 3 drift
sessions (top camera not re-fitted: MH017, MH013, AB092).
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/drift_review/top_stimulus_traces_<date>.pdf (+ figures/012_*.png)
Run (haas, ibl-ai-agent .venv): python 012_top_stimulus_traces.py [SESSION_ID ...]
"""
import datetime
import importlib.util
import pathlib
import sys

import h5py
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m006', HERE / '006_pairing_evidence.py')
m006 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m006)
OUT = m006.OUT / 'drift_review'
SESSIONS = {
    'AB095_20231212_141728': 'control (top in register, 010)',
    'AB085_20231005_152636': 'control (top in register, 010/011)',
    'AB116_20240726_094206': 'control (top end-aligned, 010)',
    'MH028_20250503_112316': 'control (end-aligned, 009/010)',
    'MH064_20260114_102210': 'control (never flagged)',
    'AB151_20250130_131124': 'control (never flagged)',
    'MH017_20250302_120918': 'DRIFT: top clause misses the final step',
    'MH013_20250219_170437': 'DRIFT: top clause misses the final step',
    'AB092_20231205_140109': 'DRIFT: top not fixed (+8,115 pulses)',
}
KEYPOINTS = [('top_particle', 'top'), ('whisker_tip', 'top'), ('whisker_base', 'top'), ('top_nose_tip', 'top'),
             ('jaw', 'side')]
GROUPS = {'whisker hit': '#00B400', 'whisker miss': '#C8A000', 'whisker passive': '#1f77b4', 'auditory': '#C800C8',
          'no stim': '0.5'}
GRID = np.round(np.arange(-0.3, 0.5001, 0.005), 3)


def s(v):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])


def load(session_id):
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        ts = {'top': b['whisker_angle']['timestamps'][:], 'side': b['jaw_angle']['timestamps'][:]}
        kp = {}
        for name, cam in KEYPOINTS:
            if f'{name}_x' not in b:
                continue
            x, y = b[f'{name}_x']['data'][:].astype(float), b[f'{name}_y']['data'][:].astype(float)
            n = min(len(x), len(ts[cam]))
            kp[name] = dict(x=x[:n], y=y[:n], t=ts[cam][:n],
                            lik=b[f'{name}_likelihood']['data'][:n].astype(float) if f'{name}_likelihood' in b else None,
                            speed=np.r_[np.nan, np.hypot(np.diff(x[:n]), np.diff(y[:n]))])
        tr = f['intervals/trials']
        t = dict(type=s(tr['trial_type'][:]), context=s(tr['context'][:]), lick=tr['lick_flag'][:],
                 w=tr['whisker_stim_time'][:], a=tr['auditory_stim_time'][:], n=tr['no_stim_time'][:])
    passive = t['context'] == 'passive'
    wh = t['type'] == 'whisker_trial'
    groups = {'whisker hit': t['w'][wh & ~passive & (t['lick'] == 1)],
              'whisker miss': t['w'][wh & ~passive & (t['lick'] == 0)],
              'whisker passive': t['w'][wh & passive],
              'auditory': t['a'][t['type'] == 'auditory_trial'],
              'no stim': t['n'][t['type'] == 'no_stim_trial']}
    return kp, {g: np.sort(v[np.isfinite(v)]) for g, v in groups.items()}


def page(pdf, session_id, label):
    kp, groups = load(session_id)
    fig = plt.figure(figsize=(16.5, 15))
    gs = fig.add_gridspec(4, 5, height_ratios=[0.8, 1.1, 1.1, 1.0], hspace=0.5, wspace=0.3)
    # (a) mean speed per keypoint and group
    for c, (name, cam) in enumerate(KEYPOINTS):
        ax = fig.add_subplot(gs[0, c])
        if name not in kp:
            ax.set_axis_off()
            continue
        k = kp[name]
        for g, ev in groups.items():
            if len(ev) < 3:
                continue
            m = m006.triggered(k['speed'], k['t'], ev, GRID)
            ax.plot(GRID * 1e3, np.nanmean(m, 0), color=GROUPS[g], lw=1.2, label=f'{g} ({len(ev)})')
        ax.axvline(0, color='k', lw=0.5, ls=':')
        ax.set_title(f'{name} speed ({cam} cam)', fontsize=9)
        ax.set_xlabel('time from stimulus (ms)', fontsize=7)
        ax.set_ylabel('px / frame (mean)', fontsize=7)
        ax.tick_params(labelsize=6)
        if c == 0:
            ax.legend(fontsize=6)
    # (b) heatmaps: particle speed and nose speed per group
    for r, name in ((1, 'top_particle'), (2, 'top_nose_tip')):
        for c, (g, ev) in enumerate(groups.items()):
            ax = fig.add_subplot(gs[r, c])
            if name not in kp or len(ev) < 3:
                ax.set_axis_off()
                continue
            k = kp[name]
            m = m006.triggered(k['speed'], k['t'], ev, GRID)
            hi = np.nanpercentile(kp[name]['speed'], 99.5)
            ax.imshow(m, aspect='auto', cmap='magma', vmin=0, vmax=hi, extent=[GRID[0] * 1e3, GRID[-1] * 1e3, len(ev), 0],
                      interpolation='nearest')
            ax.axvline(0, color='c', lw=0.5)
            ax.set_title(f'{name} speed: {g}', fontsize=8, color=GROUPS[g])
            ax.set_xlabel('time from stimulus (ms)', fontsize=7)
            ax.set_ylabel('trial (session order)', fontsize=7)
            ax.tick_params(labelsize=6)
    # (c) raw single whisker trials: particle x, y, likelihood, frame by frame
    ev = np.sort(np.r_[groups['whisker hit'], groups['whisker miss'], groups['whisker passive']])
    k = kp.get('top_particle')
    if k is not None and len(ev):
        pick = ev[np.unique(np.linspace(0, len(ev) - 1, 9).astype(int))]
        for c, (what, lab) in enumerate((('x', 'particle x (px)'), ('y', 'particle y (px)'), ('lik', 'particle likelihood'))):
            ax = fig.add_subplot(gs[3, c])
            cmap = plt.get_cmap('viridis')
            for i, e in enumerate(pick):
                j = np.flatnonzero((k['t'] >= e - 0.1) & (k['t'] <= e + 0.2))
                if not len(j) or k[what] is None:
                    continue
                v = k[what][j]
                base = 0 if what == 'lik' else np.nanmedian(v[k['t'][j] < e])
                ax.plot((k['t'][j] - e) * 1e3, v - base, '.-', ms=2.5, lw=0.7, color=cmap(i / max(len(pick) - 1, 1)),
                        label=f'{e:.0f} s')
            ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.set_title(f'{lab}{"" if what == "lik" else " minus pre-stim median"}: 9 whisker trials', fontsize=8)
            ax.set_xlabel('time from stimulus (ms); one dot = one frame', fontsize=7)
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.legend(fontsize=5, ncol=3, title='trial time', title_fontsize=6)
    ax = fig.add_subplot(gs[3, 3:])
    ax.set_axis_off()
    txt = [f'{session_id}: {label}', '', 'trials: ' + ', '.join(f'{g} {len(v)}' for g, v in groups.items()), '',
           'Raw NWB_ks4 pixels, current NWB pairing, no cleaning.',
           'A mechanical (coil) deflection should appear in top_particle on every',
           'whisker trial (hit, miss and passive) at a fixed latency, and not on',
           'auditory / no-stim trials; movements made by the mouse (nose, jaw,',
           'whisking) come later, vary in latency and also follow auditory stimuli.']
    ax.text(0, 1, '\n'.join(txt), va='top', fontsize=9, family='monospace')
    fig.suptitle(f'{session_id} ({label}): top-camera keypoints around stimuli', fontsize=12, x=0.02, ha='left')
    pdf.savefig(fig)
    fig.savefig(OUT / 'figures' / f'012_{session_id}.png', dpi=70)
    plt.close(fig)
    print(session_id, {g: len(v) for g, v in groups.items()}, flush=True)


def main():
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    sessions = sys.argv[1:] or list(SESSIONS)
    out = OUT / f"top_stimulus_traces_{datetime.date.today()}{'_test' if sys.argv[1:] else ''}.pdf"
    with PdfPages(out) as pdf:
        for sid in sessions:
            try:
                page(pdf, sid, SESSIONS.get(sid, ''))
            except Exception as ex:
                print(sid, 'ERROR', type(ex).__name__, ex, flush=True)
    print(out)


if __name__ == '__main__':
    main()
