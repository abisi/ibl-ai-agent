"""000 -- Whisker- vs auditory-evoked orofacial kinematics in R+ and R- mice (2 x 2: cohort x modality), day 0 and expert
(whisker_+1 / +2): time courses of posture and speed, tongue reach trajectories and whisker sweep (skeleton
whisker_base -> top_particle -> whisker_tip) step by step in a head-centred frame (mm). Methods: ../question.md.

Inputs: NWB_ks4 (*.nwb, DLC via ephys_utilities dlc_helpers, side + top cameras, mm); cohort per mouse from
  share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx (R+ / R-, exclude != 1).
Failed top-view tracking (2026-11-10): MH025_20250421 and MH027_20250422 are dropped in DROP_VIEWS; MH009_20250207 /
  _20250208 (no top head frame) and MH025_20250422 (almost no frame passes the skeleton rule) yield no whisker data
  through the validity rules below, not through an explicit list.

Outputs (combined_results_ks4/ssl-evoked-kinematics/):
  cache/000v1_<session>.npz, cache/000v1_<session>_trials.parquet
  figures/000_<stage>_timecourses.png       2 x 2 (cohort x modality): time courses, mice thin, mean +- SEM
  figures/000_<stage>_tongue_reach.png      2 x 2: mean first-reach trajectory step by step (5 ms) relative to the jaw
                                            at rest, per mouse (thin) and across mice; spout position
  figures/000_<stage>_tongue_reach_spout.png same, per-session spout frame (x = jaw at rest -> spout)
  figures/000_<stage>_tongue_examples.png   2 x 2: single reaches and their mean, example session per cohort
  figures/000_<stage>_whisker_sweep.png     2 x 2: mean whisker skeleton at successive times + tip path step by step
  figures/000_<stage>_whisker_examples.png  2 x 2: example session per cohort, single-trial skeletons
  figures/000_<stage>_metrics.png           per-mouse metrics, R+ vs R- and whisker vs auditory, tests
  tables/000_<stage>_metrics_mouse.csv, tables/000_<stage>_stats.csv, tables/000_<stage>_n.csv
Run (haas, first): python 000_evoked_kinematics.py [--plot-only] [--stage day0|expert] [--pilot N]
"""
import glob
import os
import pathlib
import socket
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

if 'haas' in socket.gethostname():
    sys.path.insert(0, '/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities')
else:
    sys.path.insert(0, r'M:\analysis\Axel_Bisi\Github\ephys_utilities')
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'scripts'))
from axel_bisi_paths import axel_bisi_path  # noqa: E402

NWB_DIR = axel_bisi_path('NWB_ks4')
OUT = axel_bisi_path('combined_results_ks4') / 'ssl-evoked-kinematics'
CACHE = OUT / 'cache'
SHEET = '/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx'
NOT_SSL = ('AB091', 'AB096')
DROP_SESSIONS = {'AB155_20250208_175103'}
DROP_VIEWS = {'MH025_20250421_115741': ('top',), 'MH027_20250422_111013': ('top',)}
STAGES = {'day0': ('whisker_0',), 'expert': ('whisker_+1', 'whisker_+2')}
STAGE_LABEL = {'day0': 'Day 0', 'expert': 'Expert (whisker days +1, +2)'}
COHORT_COLOR = {'R+': '#00B400', 'R-': '#C800C8'}
COHORTS = ('R+', 'R-')
MODS = ('whisker', 'auditory')
PERF = {'whisker hit': 2, 'whisker miss': 0, 'auditory hit': 3}
HIT = {'whisker': 'whisker hit', 'auditory': 'auditory hit'}
WIN, DT = (-0.5, 1.5), 0.01
SK_WIN, SK_DT = (-0.1, 0.4), 0.005           # whisker skeleton / tip path
SK_STEPS = (-0.05, 0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3)
REACH_GRID = np.arange(0, 0.3001, 0.005)     # step-by-step reach positions (5 ms)
TONGUE_OUT, TONGUE_BRIDGE = dh.TONGUE_PRESENCE_CUTOFF, 0.3
CUT = dict(whisker_base=0.1, top_particle=0.1, whisker_tip=0.5)
VARS = [  # (key, label, unit, baseline-subtract)
    ('jaw', 'jaw opening', 'mm', True),
    ('tongue_out', 'tongue out (P)', '', False),
    ('tongue_ext', 'tongue extension\n(when out)', 'mm', False),
    ('nose_disp', 'nose displ.\n(side)', 'mm', False),
    ('nose_speed', 'nose speed\n(side)', 'mm/s', False),
    ('wh_angle', 'whisker angle\n(- = protraction)', 'deg', True),
    ('wh_speed', 'whisker\nang. speed', 'deg/s', False),
    ('wh_curv', 'whisker\ncurvature', 'deg', True),
    ('topnose_speed', 'nose speed (top)', 'mm/s', False),
]
METRICS = [('reach_latency', 'first reach latency', 's'), ('reach_duration', 'reach duration', 's'),
           ('reach_max_ext', 'max tongue extension', 'mm'), ('reach_peak_speed', 'peak tongue speed', 'mm/s'),
           ('lat_jaw', 'jaw onset latency', 's'), ('lat_whisker', 'whisker onset latency', 's'),
           ('lat_nose', 'nose onset latency', 's'), ('wh_amp', 'whisker sweep amplitude (0-0.3 s)', 'deg')]


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


# ---------------------------------------------------------------- geometry
def xy(v, kp):
    return np.c_[v.data[f'{kp}_x'].to_numpy(), -v.data[f'{kp}_y'].to_numpy()]  # image y down -> up


def lik(v, kp):
    return v.likelihood[f'{kp}_likelihood'].to_numpy()


def head_frame(v, tip, base, sign_kps):
    """origin, rotation (forward = +x), flip of y so that the median of sign_kps is on +y."""
    ok = (lik(v, tip) >= 0.1) & (lik(v, base) >= 0.5)
    if ok.mean() < 0.05:
        return None
    t, b = xy(v, tip)[ok], xy(v, base)[ok]
    o = np.nanmedian(t, 0)
    f = np.nanmedian(t - b, 0)
    f = f / np.linalg.norm(f)
    r = np.array([[f[0], f[1]], [-f[1], f[0]]])
    pts = []
    for kp in sign_kps:
        g = lik(v, kp) >= 0.1
        if g.mean() > 0.05:
            pts.append(np.nanmedian((xy(v, kp)[g] - o) @ r.T, 0))
    flip = bool(pts) and np.mean([p[1] for p in pts]) < 0
    return o, r, flip


def to_head(p, hf):
    o, r, flip = hf
    q = (p - o) @ r.T
    if flip:
        q[:, 1] *= -1
    return q


def run_median(x, n=3):
    return pd.Series(x).rolling(n, center=True, min_periods=2).median().to_numpy()


def angle(d):
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def wrap(a):
    return (a + 180) % 360 - 180


# ---------------------------------------------------------------- sessions
def sessions_and_cohorts():
    sheet = pd.read_excel(SHEET)
    sheet = sheet[sheet.reward_group.isin(['R+', 'R-']) & (sheet.exclude != 1)]
    cohort = dict(zip(sheet.mouse_id, sheet.reward_group))
    out = []
    for p in sorted(glob.glob(str(NWB_DIR / '*.nwb'))):
        s = os.path.basename(p)[:-4]
        if '_backup' in p or s[:5] in NOT_SSL or s in DROP_SESSIONS or s[:5] not in cohort:
            continue
        with h5py.File(p, 'r') as f:
            d = f['session_description'][()].decode()
        st = [k for k, v in STAGES.items() if d in v]
        if st:
            out.append(dict(path=p, session=s, mouse=s[:5], cohort=cohort[s[:5]], stage=st[0], day=d))
    return pd.DataFrame(out)


def trial_events(f):
    t = f['intervals/trials']
    n = len(t['start_time'])
    ctx = dec(t['context'][:]) if 'context' in t else np.array(['nan'] * n)
    perf = t['perf'][:]
    w, a = t['whisker_stim'][:] == 1, t['auditory_stim'][:] == 1
    ev = np.where(w, t['whisker_stim_time'][:], np.where(a, t['auditory_stim_time'][:], t['start_time'][:]))
    act = ctx != 'passive'
    return {k: ev[act & (perf == p) & np.isfinite(ev)] for k, p in PERF.items()}


# ---------------------------------------------------------------- per session
def onset_latency(m, grid, sd):
    """first of 3 consecutive bins in 0 .. 1 s above baseline mean + 5 sd (m: trials x bins, baseline-subtracted)."""
    out = np.full(len(m), np.nan)
    if not np.isfinite(sd) or sd <= 0:
        return out
    idx = np.flatnonzero((grid >= 0) & (grid <= 1.0))
    above = m[:, idx] > 5 * sd
    run = above[:, :-2] & above[:, 1:-1] & above[:, 2:]
    has = run.any(1)
    out[has] = grid[idx][np.argmax(run[has], 1)]
    return out


def reaches(v):
    """runs of tongue out: (start index, end index inclusive) with single bridged frames."""
    lk = lik(v, 'tongue')
    out = lk >= TONGUE_OUT
    gap = (~out[1:-1]) & out[:-2] & out[2:] & (lk[1:-1] >= TONGUE_BRIDGE)
    out[1:-1] |= gap
    d = np.diff(np.r_[0, out.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def one(row):
    s, path = row['session'], row['path']
    f_npz, f_tr = CACHE / f'000v1_{s}.npz', CACHE / f'000v1_{s}_trials.parquet'
    if f_npz.exists() and f_tr.exists():
        return s
    try:
        views = dh.load_dlc(path, mask_likelihood=False)
        for vname in DROP_VIEWS.get(s, ()):
            views.pop(vname, None)
        with h5py.File(path, 'r') as f:
            evs = trial_events(f)
        store, trows = {}, []
        series, ts_of = {}, {}
        side, top = views.get('side'), views.get('top')
        hf_side = hf_top = None
        if side is not None and 'tongue_likelihood' in side.likelihood and 'nose_base_x' in side.data:
            hf_side = head_frame(side, 'nose_tip', 'nose_base', ['pupil_top', 'pupil_bottom', 'pupil_left', 'pupil_right'])
        if top is not None and 'whisker_tip_x' in top.data and 'top_nose_base_x' in top.data:
            hf_top = head_frame(top, 'top_nose_tip', 'top_nose_base', ['whisker_base'])
        if hf_side is not None:
            ts = side.timestamps
            jaw, _ = dh.clean_series(side, 'jaw_distance')
            series['jaw'] = jaw
            series['tongue_out'] = dh.tongue_presence(side)
            tg = to_head(xy(side, 'tongue'), hf_side)
            tg[lik(side, 'tongue') < TONGUE_OUT] = np.nan
            jw = to_head(xy(side, 'jaw'), hf_side)
            ns = to_head(xy(side, 'nose_tip'), hf_side)
            ns[lik(side, 'nose_tip') < 0.1] = np.nan
            ns = np.c_[run_median(ns[:, 0]), run_median(ns[:, 1])]
            series['nose_speed'] = dh.keypoint_speed(ns[:, 0], ns[:, 1], ts)
            for k, a in (('tg_x', tg[:, 0]), ('tg_y', tg[:, 1]), ('jw_x', jw[:, 0]), ('jw_y', jw[:, 1]),
                         ('ns_x', ns[:, 0]), ('ns_y', ns[:, 1])):
                series[k] = a
            for k in ('jaw', 'tongue_out', 'nose_speed', 'tg_x', 'tg_y', 'jw_x', 'jw_y', 'ns_x', 'ns_y'):
                ts_of[k] = ts
            if 'spout_x' in side.data:
                g = lik(side, 'spout') >= 0.5
                if g.mean() > 0.05:
                    store['spout_head'] = np.nanmedian(to_head(xy(side, 'spout')[g], hf_side), 0)
            rs, re_ = reaches(side)
            tg_speed = dh.keypoint_speed(tg[:, 0], tg[:, 1], ts)
        if hf_top is not None:
            ts2 = top.timestamps
            pts = {kp: to_head(xy(top, kp), hf_top) for kp in CUT}
            ok = np.ones(len(ts2), bool)
            for kp, c in CUT.items():
                ok &= lik(top, kp) >= c
            l1 = np.linalg.norm(pts['top_particle'] - pts['whisker_base'], axis=1)
            l2 = np.linalg.norm(pts['whisker_tip'] - pts['top_particle'], axis=1)
            for ln in (l1, l2):
                med = np.nanmedian(ln[ok]) if ok.any() else np.nan
                ok &= (ln > 0.6 * med) & (ln < 1.6 * med)
            for kp in CUT:
                p = pts[kp].copy()
                p[~ok] = np.nan
                pts[kp] = np.c_[run_median(p[:, 0]), run_median(p[:, 1])]
            a_bt = angle(pts['whisker_tip'] - pts['whisker_base'])
            a1 = angle(pts['top_particle'] - pts['whisker_base'])
            a2 = angle(pts['whisker_tip'] - pts['top_particle'])
            series['wh_angle'] = a_bt
            series['wh_speed'] = dh.derivative(a_bt, ts2, kind='speed')
            series['wh_curv'] = wrap(a2 - a1)
            tn = xy(top, 'top_nose_tip')
            tn[lik(top, 'top_nose_tip') < 0.1] = np.nan
            series['topnose_speed'] = dh.keypoint_speed(run_median(tn[:, 0]), run_median(tn[:, 1]), ts2)
            for kp, short in (('whisker_base', 'wb'), ('top_particle', 'wp'), ('whisker_tip', 'wt')):
                series[f'{short}_x'], series[f'{short}_y'] = pts[kp][:, 0], pts[kp][:, 1]
            for k in ('wh_angle', 'wh_speed', 'wh_curv', 'topnose_speed', 'wb_x', 'wb_y', 'wp_x', 'wp_y', 'wt_x', 'wt_y'):
                ts_of[k] = ts2
            store['whisker_valid_frac'] = np.array([ok.mean()])
        if not series:
            return None
        # baseline SDs for onset detection (session median over all trials of all types)
        all_ev = np.sort(np.concatenate([e for e in evs.values()]))
        sd = {}
        for k in ('jaw', 'wh_angle'):
            if k in series and len(all_ev):
                g, m = dh.peth(series[k], ts_of[k], all_ev, (-0.5, 0), dt=DT)
                sd[k] = np.nanmedian(np.nanstd(m, 1))
        if 'ns_x' in series and len(all_ev):
            g, mx = dh.peth(series['ns_x'], ts_of['ns_x'], all_ev, (-0.5, 0), dt=DT)
            _, my = dh.peth(series['ns_y'], ts_of['ns_y'], all_ev, (-0.5, 0), dt=DT)
            d = np.hypot(mx - np.nanmean(mx, 1, keepdims=True), my - np.nanmean(my, 1, keepdims=True))
            sd['nose'] = np.nanmedian(np.nanstd(d, 1))
        for tt, ev in evs.items():
            if not len(ev):
                continue
            trial = pd.DataFrame({'session': s, 'trial_type': tt, 'event_time': ev})
            mats = {}
            for k in ('jaw', 'tongue_out', 'nose_speed', 'wh_angle', 'wh_speed', 'wh_curv', 'topnose_speed',
                      'tg_x', 'tg_y', 'jw_x', 'jw_y', 'ns_x', 'ns_y'):
                if k in series:
                    grid, mats[k] = dh.peth(series[k], ts_of[k], ev, WIN, dt=DT)
            if not mats:
                continue
            store['grid'] = grid
            base = grid < 0
            out = {}
            for k in ('jaw', 'wh_angle', 'wh_curv'):
                if k in mats:
                    out[k] = mats[k] - np.nanmean(mats[k][:, base], 1, keepdims=True)
            for k in ('tongue_out', 'nose_speed', 'wh_speed', 'topnose_speed'):
                if k in mats:
                    out[k] = mats[k]
            if 'tg_x' in mats:
                jx = np.nanmedian(mats['jw_x'][:, base], 1, keepdims=True)
                jy = np.nanmedian(mats['jw_y'][:, base], 1, keepdims=True)
                out['tongue_ext'] = np.hypot(mats['tg_x'] - jx, mats['tg_y'] - jy)
                nd = np.hypot(mats['ns_x'] - np.nanmean(mats['ns_x'][:, base], 1, keepdims=True),
                              mats['ns_y'] - np.nanmean(mats['ns_y'][:, base], 1, keepdims=True))
                out['nose_disp'] = nd
                trial['lat_nose'] = onset_latency(nd, grid, sd.get('nose', np.nan))
                trial['jaw_rest_x'], trial['jaw_rest_y'] = jx[:, 0], jy[:, 0]
            if 'jaw' in out:
                trial['lat_jaw'] = onset_latency(out['jaw'], grid, sd.get('jaw', np.nan))
            if 'wh_angle' in out:
                trial['lat_whisker'] = onset_latency(np.abs(out['wh_angle']), grid, sd.get('wh_angle', np.nan))
                w03 = (grid >= 0) & (grid <= 0.3)
                with np.errstate(all='ignore'):
                    trial['wh_amp'] = np.nanmax(np.abs(out['wh_angle'][:, w03]), 1)
            for k, m in out.items():
                okt = np.isfinite(m).mean(1) >= (0.5 if k != 'tongue_ext' else 0)
                if k == 'tongue_ext':
                    okt = np.ones(len(m), bool)
                if okt.any():
                    with np.errstate(all='ignore'):
                        store[f'tc|{k}|{tt}'] = np.nanmean(m[okt], 0)
                    store[f'n|{k}|{tt}'] = np.array([okt.sum()])
            # whisker skeleton / tip path
            if 'wb_x' in series:
                skel = []
                for short in ('wb', 'wp', 'wt'):
                    g2, mxs = dh.peth(series[f'{short}_x'], ts_of['wb_x'], ev, SK_WIN, dt=SK_DT)
                    _, mys = dh.peth(series[f'{short}_y'], ts_of['wb_x'], ev, SK_WIN, dt=SK_DT)
                    skel.append(np.stack([mxs, mys], -1))
                skel = np.stack(skel, 2)  # trials x time x 3 points x 2
                store[f'skel|{tt}'] = skel.astype(np.float32)
                store['sk_grid'] = g2
            # first tongue reach
            if hf_side is not None:
                rows = []
                for k_ev, e in enumerate(ev):
                    i0 = np.searchsorted(ts[rs], e)
                    r = dict(reach_latency=np.nan)
                    traj = np.full((len(REACH_GRID), 2), np.nan)
                    if i0 < len(rs) and ts[rs[i0]] - e <= 1.0:
                        a, b = rs[i0], re_[i0]
                        dur = (b - a + 1) / side.rate
                        if b - a + 1 >= 3 and dur <= 0.3:
                            seg = tg[a:b + 1]
                            jr = np.array([trial['jaw_rest_x'].iat[k_ev], trial['jaw_rest_y'].iat[k_ev]]) \
                                if 'jaw_rest_x' in trial else np.array([np.nan, np.nan])
                            rel = seg - jr
                            ext = np.hypot(rel[:, 0], rel[:, 1])
                            q = ts[a] + REACH_GRID
                            j = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
                            j = np.where(np.abs(q - ts[j - 1]) <= np.abs(ts[j] - q), j - 1, j)
                            good = (j >= a) & (j <= b) & (np.abs(ts[j] - q) <= 0.5 / side.rate + 1e-6)
                            traj[good] = tg[j[good]] - jr
                            k_max = int(np.nanargmax(ext)) if np.isfinite(ext).any() else None
                            r = dict(reach_latency=ts[a] - e, reach_duration=dur,
                                     reach_max_ext=float(np.nanmax(ext)) if k_max is not None else np.nan,
                                     reach_peak_speed=float(np.nanmax(tg_speed[a + 1:b + 1])) if b > a and
                                     np.isfinite(tg_speed[a + 1:b + 1]).any() else np.nan,
                                     reach_end_x=rel[k_max, 0] if k_max is not None else np.nan,
                                     reach_end_y=rel[k_max, 1] if k_max is not None else np.nan)
                    rows.append(r)
                    if k_ev == 0:
                        trajs = []
                    trajs.append(traj)
                trial = pd.concat([trial, pd.DataFrame(rows, index=trial.index)], axis=1)
                store[f'reach|{tt}'] = np.array(trajs, dtype=np.float32)
            trows.append(trial)
        np.savez_compressed(f_npz, **store)
        pd.concat(trows, ignore_index=True).to_parquet(f_tr)
        return s
    except Exception as e:
        print(s, 'ERROR', type(e).__name__, e, flush=True)
        return None


# ---------------------------------------------------------------- aggregation
def mean_sem(mat):
    mat = np.asarray(mat, float)
    with np.errstate(all='ignore'):
        m = np.nanmean(mat, 0)
        n = np.sum(np.isfinite(mat), 0)
        se = np.nanstd(mat, 0, ddof=1) / np.sqrt(np.maximum(n, 1)) if len(mat) > 1 else m * 0
    return m, se


def collect(sess):
    """per (cohort, key, trial type): mouse x ... array (mean over the mouse's sessions), mouse ids, n trials."""
    acc, extras, trials = {}, {}, []
    for r in sess.itertuples():
        f = CACHE / f'000v1_{r.session}.npz'
        if not f.exists():
            continue
        d = np.load(f)
        for k in d.files:
            if k.startswith(('tc|', 'n|', 'skel|', 'reach|')):
                acc.setdefault((r.cohort, k), {}).setdefault(r.mouse, []).append(d[k])
            elif k in ('grid', 'sk_grid'):
                extras[k] = d[k]
            elif k == 'spout_head':
                extras.setdefault(('spout', r.cohort), {}).setdefault(r.mouse, []).append(d[k])
        t = pd.read_parquet(CACHE / f'000v1_{r.session}_trials.parquet')
        t['mouse'], t['cohort'] = r.mouse, r.cohort
        trials.append(t)
        sv = spout_vector(d, t)
        if sv is not None:
            extras.setdefault(('spout_rel', r.cohort), {}).setdefault(r.mouse, []).append(sv)
            rot = spout_rotation(sv)
            for k in d.files:
                if k.startswith('reach|'):
                    acc.setdefault((r.cohort, 'reachS|' + k.split('|', 1)[1]), {}).setdefault(r.mouse, []).append(
                        d[k] @ rot.T)
    trials = pd.concat(trials, ignore_index=True) if trials else pd.DataFrame()
    return acc, extras, trials


def spout_vector(d, t):
    """spout position relative to the session's median jaw at rest (side head frame, mm), or None."""
    if 'spout_head' not in d.files or 'jaw_rest_x' not in t:
        return None
    jr = np.array([np.nanmedian(t.jaw_rest_x), np.nanmedian(t.jaw_rest_y)])
    v = d['spout_head'] - jr
    return v if np.isfinite(v).all() and np.linalg.norm(v) > 0.5 else None


def spout_rotation(v):
    """rotation mapping the jaw-at-rest -> spout direction to +x (keeps handedness: +y stays on the 'up' side)."""
    u = v / np.linalg.norm(v)
    return np.array([[u[0], u[1]], [-u[1], u[0]]])


def mouse_matrix(acc, cohort, key, reducer=None):
    per = acc.get((cohort, key), {})
    mice = sorted(per)
    if reducer is None:
        mat = [np.nanmean(np.array(per[m]), 0) for m in mice]
    else:
        mat = [reducer(per[m]) for m in mice]
    return mice, np.array(mat) if mat else None


def n_trials(acc, cohort, var, tt):
    per = acc.get((cohort, f'n|{var}|{tt}'), {})
    return len(per), int(sum(int(x[0]) for v in per.values() for x in v))


# ---------------------------------------------------------------- figures
def grid22(fig, nrow_inner, ncol_inner=1, hspace=0.15, wspace=0.25, **kw):
    outer = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.18, **kw)
    cells = {}
    for i, c in enumerate(COHORTS):
        for j, m in enumerate(MODS):
            cells[(c, m)] = outer[i, j].subgridspec(nrow_inner, ncol_inner, hspace=hspace, wspace=wspace)
    return cells


def fig_timecourses(acc, extras, stage):
    g = extras['grid']
    vars_ = [v for v in VARS if any((c, f'tc|{v[0]}|whisker hit') in acc for c in COHORTS)]
    fig = plt.figure(figsize=(14, 2.0 * len(vars_) + 1.5))
    cells = grid22(fig, len(vars_), top=0.94, bottom=0.04, left=0.08, right=0.98)
    lims = {}
    axes = {}
    for (c, mod), sg in cells.items():
        for r, (k, lab, unit, _) in enumerate(vars_):
            ax = fig.add_subplot(sg[r, 0])
            axes[(c, mod, k)] = ax
            tts = [HIT[mod]] + (['whisker miss'] if mod == 'whisker' else [])
            for tt in tts:
                mice, mat = mouse_matrix(acc, c, f'tc|{k}|{tt}')
                if mat is None:
                    continue
                hit = tt == HIT[mod]
                if hit:
                    for row in mat:
                        ax.plot(g, row, color=COHORT_COLOR[c], lw=0.35, alpha=0.3)
                m, se = mean_sem(mat)
                col = 'k' if hit else '0.5'
                ax.fill_between(g, m - se, m + se, color=col, alpha=0.2, lw=0)
                ax.plot(g, m, color=col, lw=1.2, ls='-' if hit else '--',
                        label=f'{tt}: {len(mice)} mice ({n_trials(acc, c, k, tt)[1]} trials)')
                lo, hi = np.nanmin(m - se), np.nanmax(m + se)
                l0 = lims.get(k, (np.inf, -np.inf))
                lims[k] = (min(l0[0], lo), max(l0[1], hi))
            ax.axvline(0, color='k', lw=0.5, ls=':')
            ax.set_xlim(-0.2, 1.0)
            ax.tick_params(labelsize=6)
            ax.spines[['top', 'right']].set_visible(False)
            if mod == 'whisker':
                ax.set_ylabel(f'{lab}\n({unit})' if unit else lab, fontsize=6.5)
            if r == 0:
                ax.set_title(f'{c}, {mod} trials', fontsize=10, color=COHORT_COLOR[c])
                ax.legend(fontsize=5.5, loc='upper right', frameon=False)
            if r < len(vars_) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('time from stimulus (s)', fontsize=7)
    for (c, mod, k), ax in axes.items():
        if k in lims and np.isfinite(lims[k]).all():
            lo, hi = lims[k]
            pad = 0.08 * (hi - lo) if hi > lo else 1
            ax.set_ylim(lo - pad, hi + pad)
    fig.suptitle(f'{STAGE_LABEL[stage]}: stimulus-evoked orofacial kinematics. Black = hits (mean +- SEM across mice; '
                 f'thin = mice), grey dashed = whisker misses.\nWhisker trials: the whisker transient at 0 ms (hits and misses) is the '
                 f'stimulus moving the whisker, not an evoked movement', fontsize=10)
    fig.savefig(OUT / 'figures' / f'000_{stage}_timecourses.png', dpi=110)
    plt.close(fig)


def reach_mouse_mean(trajs):
    """per mouse: mean over sessions of the mean over reaches (time x 2), steps with >= 50% of reaches present."""
    per = []
    for t in trajs:
        if len(t) == 0:
            continue
        ok = np.isfinite(t[:, :, 0]).any(1)
        if ok.sum() < 3:
            continue
        t = t[ok]
        frac = np.isfinite(t[:, :, 0]).mean(0)
        with np.errstate(all='ignore'):
            m = np.nanmean(t, 0)
        m[frac < 0.5] = np.nan
        per.append(m)
    if not per:
        return np.full((len(REACH_GRID), 2), np.nan)
    with np.errstate(all='ignore'):
        return np.nanmean(np.array(per), 0)


def step_plot(ax, traj, cmap='viridis', label_every=4, ms=18, lw=0.8, color=None):
    ok = np.isfinite(traj[:, 0])
    if not ok.any():
        return
    t = REACH_GRID[ok] * 1000
    ax.plot(traj[ok, 0], traj[ok, 1], '-', color=color or '0.3', lw=lw, zorder=1)
    sc = ax.scatter(traj[ok, 0], traj[ok, 1], c=t, cmap=cmap, s=ms, vmin=0, vmax=150, zorder=2, edgecolor='k', lw=0.3)
    for i in np.flatnonzero(ok)[::label_every]:
        ax.annotate(f'{REACH_GRID[i] * 1000:.0f}', traj[i], fontsize=5, xytext=(3, 3), textcoords='offset points')
    return sc


def fig_tongue(acc, extras, stage, frame='head'):
    pref = 'reach' if frame == 'head' else 'reachS'
    fig = plt.figure(figsize=(12, 11))
    cells = grid22(fig, 1, top=0.92, bottom=0.06, left=0.07, right=0.9)
    sc = None
    axes = []
    for (c, mod), sg in cells.items():
        ax = fig.add_subplot(sg[0, 0])
        axes.append(ax)
        per = acc.get((c, f'{pref}|{HIT[mod]}'), {})
        mm = [reach_mouse_mean(per[m]) for m in sorted(per)]
        mm = [m for m in mm if np.isfinite(m).any()]
        for m in mm:
            ok = np.isfinite(m[:, 0])
            ax.plot(m[ok, 0], m[ok, 1], color=COHORT_COLOR[c], lw=0.5, alpha=0.5)
        if mm:
            arr = np.array(mm)
            with np.errstate(all='ignore'):
                mean = np.nanmean(arr, 0)
                n = np.isfinite(arr[:, :, 0]).sum(0)
                se = np.nanstd(arr, 0, ddof=1) / np.sqrt(np.maximum(n, 1))[:, None]
            mean[n < max(2, 0.5 * len(mm))] = np.nan
            okm = np.isfinite(mean[:, 0])
            ax.errorbar(mean[okm, 0], mean[okm, 1], xerr=se[okm, 0], yerr=se[okm, 1], fmt='none', ecolor='0.6', lw=0.5)
            sc = step_plot(ax, mean)
        sp = [np.nanmean(np.array(v), 0) for v in extras.get(('spout_rel', c), {}).values()]
        if sp:
            sp = np.array(sp)
            if frame == 'spout':
                sp = np.c_[np.linalg.norm(sp, axis=1), np.zeros(len(sp))]
            ax.plot(sp[:, 0], sp[:, 1], 'o', mfc='none', mec='0.5', ms=4, mew=0.5)
            ax.plot(*np.nanmean(sp, 0), 'D', color='orange', mec='k', ms=7)
            ax.annotate('spout (mean; circles = mice)', np.nanmean(sp, 0), fontsize=6, xytext=(5, 5),
                        textcoords='offset points')
        ax.plot(0, 0, 'k+', ms=10, mew=1.5)
        ax.annotate('jaw at rest', (0, 0), fontsize=6, xytext=(4, -10), textcoords='offset points')
        ntr = sum(len(t) for v in per.values() for t in v)
        nre = sum(int(np.isfinite(t[:, :, 0]).any(1).sum()) for v in per.values() for t in v)
        ax.set_title(f'{c}, {HIT[mod]}: {len(mm)} mice, {nre} first reaches / {ntr} trials', fontsize=9,
                     color=COHORT_COLOR[c])
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('forward (mm, head axis)' if frame == 'head' else 'towards the spout (mm)', fontsize=7)
        ax.set_ylabel('up (mm)' if frame == 'head' else 'orthogonal, up side (mm)', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(lw=0.3, alpha=0.4)
    if axes:
        xs = [a.get_xlim() for a in axes]
        ys = [a.get_ylim() for a in axes]
        for a in axes:
            a.set_xlim(min(x[0] for x in xs), max(x[1] for x in xs))
            a.set_ylim(min(y[0] for y in ys), max(y[1] for y in ys))
    if sc is not None:
        cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
        fig.colorbar(sc, cax=cax, label='time from reach onset (ms)')
    what = ('head frame, origin = jaw at rest' if frame == 'head' else
            'spout frame per session: origin = jaw at rest, x = towards the spout')
    fig.suptitle(f'{STAGE_LABEL[stage]}: first tongue reach after the stimulus (side view, {what}).\nDots = mean '
                 f'position every 5 ms across mice (+- SEM; steps with >= 50% of reaches still out), thin = mice. '
                 f'Tongue frames: likelihood >= {TONGUE_OUT}', fontsize=9)
    fig.savefig(OUT / 'figures' / f'000_{stage}_tongue_reach{"" if frame == "head" else "_spout"}.png', dpi=110)
    plt.close(fig)


def pick_examples(sess, stage):
    ex = {}
    for c in COHORTS:
        best, score = None, -1
        for r in sess[(sess.cohort == c) & (sess.stage == stage)].itertuples():
            f = CACHE / f'000v1_{r.session}.npz'
            if not f.exists():
                continue
            d = np.load(f)
            ns = []
            for mod in MODS:
                k = f'reach|{HIT[mod]}'
                ns.append(int(np.isfinite(d[k][:, :, 0]).any(1).sum()) if k in d.files else 0)
                k2 = f'skel|{HIT[mod]}'
                ns.append(int(np.isfinite(d[k2][:, :, 2, 0]).mean(1).__ge__(0.8).sum()) if k2 in d.files else 0)
            if min(ns) > score:
                best, score = r.session, min(ns)
        ex[c] = best
    return ex


def fig_examples(ex, stage):
    fig = plt.figure(figsize=(12, 11))
    cells = grid22(fig, 1, top=0.92, bottom=0.06, left=0.07, right=0.9)
    fig2 = plt.figure(figsize=(12, 11))
    cells2 = grid22(fig2, 1, top=0.92, bottom=0.06, left=0.07, right=0.9)
    sc = sc2 = None
    for (c, mod), sg in cells.items():
        ax = fig.add_subplot(sg[0, 0])
        ax2 = fig2.add_subplot(cells2[(c, mod)][0, 0])
        s = ex.get(c)
        if s is None:
            continue
        d = np.load(CACHE / f'000v1_{s}.npz')
        k = f'reach|{HIT[mod]}'
        if k in d.files:
            t = d[k]
            ok = np.isfinite(t[:, :, 0]).any(1)
            for tr in t[ok]:
                o = np.isfinite(tr[:, 0])
                ax.plot(tr[o, 0], tr[o, 1], '.-', color=COHORT_COLOR[c], lw=0.4, ms=2, alpha=0.4)
            sc = step_plot(ax, reach_mouse_mean([t]))
            ax.set_title(f'{s} ({c}), {HIT[mod]}: {ok.sum()} reaches', fontsize=9, color=COHORT_COLOR[c])
            sv = spout_vector(d, pd.read_parquet(CACHE / f'000v1_{s}_trials.parquet'))
            if sv is not None:
                ax.plot(*sv, 'D', color='orange', mec='k', ms=7)
                ax.annotate('spout', sv, fontsize=6, xytext=(5, 5), textcoords='offset points')
        ax.plot(0, 0, 'k+', ms=10, mew=1.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_xlabel('forward (mm)', fontsize=7)
        ax.set_ylabel('up (mm)', fontsize=7)
        ax.grid(lw=0.3, alpha=0.4)
        k2 = f'skel|{HIT[mod]}'
        if k2 in d.files:
            sk = d[k2]
            g2 = d['sk_grid']
            good = np.isfinite(sk[:, :, :, 0]).all(2).mean(1) >= 0.8
            sk = sk[good]
            idx = [int(np.argmin(np.abs(g2 - t))) for t in SK_STEPS]
            cm = plt.cm.viridis(np.linspace(0, 1, len(idx)))
            for tr in sk[:30]:
                for ci, i in enumerate(idx):
                    ax2.plot(tr[i, :, 0], tr[i, :, 1], '-', color=cm[ci], lw=0.3, alpha=0.35)
            with np.errstate(all='ignore'):
                m = np.nanmean(sk, 0)
            for ci, i in enumerate(idx):
                ax2.plot(m[i, :, 0], m[i, :, 1], 'o-', color=cm[ci], lw=1.6, ms=3,
                         label=f'{SK_STEPS[ci] * 1000:+.0f} ms')
            sc2 = True
            ax2.set_title(f'{s} ({c}), {HIT[mod]}: {len(sk)} trials (30 shown)', fontsize=9, color=COHORT_COLOR[c])
            ax2.legend(fontsize=5.5, ncol=3, frameon=False)
        ax2.plot(0, 0, 'k^', ms=6)
        ax2.annotate('nose tip', (0, 0), fontsize=6, xytext=(4, 4), textcoords='offset points')
        ax2.set_aspect('equal', adjustable='datalim')
        ax2.set_xlabel('forward (mm, head axis)', fontsize=7)
        ax2.set_ylabel('lateral (mm)', fontsize=7)
        ax2.grid(lw=0.3, alpha=0.4)
    if sc is not None:
        cax = fig.add_axes([0.92, 0.3, 0.012, 0.4])
        fig.colorbar(sc, cax=cax, label='time from reach onset (ms)')
    fig.suptitle(f'{STAGE_LABEL[stage]}: example sessions, single first reaches (thin) and their mean every 5 ms',
                 fontsize=10)
    fig.savefig(OUT / 'figures' / f'000_{stage}_tongue_examples.png', dpi=110)
    plt.close(fig)
    fig2.suptitle(f'{STAGE_LABEL[stage]}: example sessions, whisker skeleton (base - particle - tip, top view, head '
                  f'frame) at successive times from the stimulus; thin = single trials, thick = mean', fontsize=10)
    fig2.savefig(OUT / 'figures' / f'000_{stage}_whisker_examples.png', dpi=110)
    plt.close(fig2)


def fig_whisker(acc, extras, stage):
    g2 = extras.get('sk_grid')
    if g2 is None:
        return
    fig = plt.figure(figsize=(12, 11))
    cells = grid22(fig, 1, top=0.92, bottom=0.06, left=0.07, right=0.88)
    idx = [int(np.argmin(np.abs(g2 - t))) for t in SK_STEPS]
    cm = plt.cm.viridis(np.linspace(0, 1, len(idx)))
    axes = []

    def red(lst):
        ms = []
        for sk in lst:
            good = np.isfinite(sk[:, :, :, 0]).all(2).mean(1) >= 0.8
            if good.sum() >= 3:
                with np.errstate(all='ignore'):
                    ms.append(np.nanmean(sk[good], 0))
        with np.errstate(all='ignore'):
            return np.nanmean(np.array(ms), 0) if ms else np.full(lst[0].shape[1:], np.nan)

    bound = []
    for (c, mod), sg in cells.items():
        ax = fig.add_subplot(sg[0, 0])
        mice, mat = mouse_matrix(acc, c, f'skel|{HIT[mod]}', reducer=red)
        if mat is None:
            continue
        ok = np.isfinite(mat).mean(axis=(1, 2, 3)) >= 0.8  # single missing bins tolerated (nan-means below)
        mat = mat[ok]
        if not len(mat):
            continue
        axes.append(ax)
        # mouse means aligned on the baseline whisker base (shape comparison across mice), then averaged
        b0 = np.nanmean(mat[:, g2 < 0, 0, :], 1)
        al = mat - b0[:, None, None, :]
        with np.errstate(all='ignore'):
            m = np.nanmean(al, 0)
        for mi in al:
            ax.plot(mi[g2 >= 0, 2, 0], mi[g2 >= 0, 2, 1], color=COHORT_COLOR[c], lw=0.4, alpha=0.4)
        bound.append(m.reshape(-1, 2))
        for ci, i in enumerate(idx):
            ax.plot(m[i, :, 0], m[i, :, 1], 'o-', color=cm[ci], lw=1.8, ms=3.5, label=f'{SK_STEPS[ci] * 1000:+.0f} ms')
        tip = m[(g2 >= -0.05) & (g2 <= 0.3), 2]
        tt = g2[(g2 >= -0.05) & (g2 <= 0.3)]
        scat = ax.scatter(tip[:, 0], tip[:, 1], c=tt * 1000, cmap='viridis', s=10, edgecolor='k', lw=0.2, zorder=3)
        ntr = sum(len(s) for v in acc.get((c, f'skel|{HIT[mod]}'), {}).values() for s in v)
        ax.set_title(f'{c}, {HIT[mod]}: {len(al)} mice ({ntr} trials)', fontsize=9, color=COHORT_COLOR[c])
        ax.plot(0, 0, 'ks', ms=4)
        ax.annotate('whisker base at baseline', (0, 0), fontsize=6, xytext=(4, -10), textcoords='offset points')
        ax.set_xlabel('forward (mm, head axis)', fontsize=7)
        ax.set_ylabel('lateral (mm)', fontsize=7)
        ax.grid(lw=0.3, alpha=0.4)
        ax.legend(fontsize=5.5, ncol=3, frameon=False, loc='lower left')
    if axes:
        bb = np.vstack(bound + [np.zeros((1, 2))])
        lo, hi = np.nanmin(bb, 0) - 2.0, np.nanmax(bb, 0) + 2.0  # mean skeletons + 2 mm (mouse paths may leave)
        half = max(hi - lo) / 2
        mid = (lo + hi) / 2
        for a in axes:  # same equal-aspect square window in every panel
            a.set_xlim(mid[0] - half, mid[0] + half)
            a.set_ylim(mid[1] - half, mid[1] + half)
            a.set_aspect('equal', adjustable='box')
        cax = fig.add_axes([0.9, 0.3, 0.012, 0.4])
        fig.colorbar(scat, cax=cax, label='tip, time from stimulus (ms)')
    fig.suptitle(f'{STAGE_LABEL[stage]}: whisker sweep, mean skeleton (base - particle - tip) across mice at successive '
                 f'times, tip path every 5 ms (dots), thin = mouse tip paths 0-0.4 s', fontsize=9)
    fig.savefig(OUT / 'figures' / f'000_{stage}_whisker_sweep.png', dpi=110)
    plt.close(fig)


def fig_metrics(trials, stage):
    t = trials[trials.trial_type.isin(HIT.values())].copy()
    t['mod'] = t.trial_type.map({v: k for k, v in HIT.items()})
    cols = [m for m, _, _ in METRICS if m in t]
    mouse = t.groupby(['cohort', 'mouse', 'mod'])[cols].median().reset_index()
    mouse.to_csv(OUT / 'tables' / f'000_{stage}_metrics_mouse.csv', index=False)
    rows = []
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    pos = {('R+', 'whisker'): 0, ('R+', 'auditory'): 1, ('R-', 'whisker'): 2.5, ('R-', 'auditory'): 3.5}
    for ax, (mcol, lab, unit) in zip(axs.flat, [m for m in METRICS if m[0] in cols]):
        for (c, mod), x in pos.items():
            v = mouse[(mouse.cohort == c) & (mouse['mod'] == mod)][mcol].dropna()
            ax.scatter(np.full(len(v), x) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, s=10,
                       color=COHORT_COLOR[c], alpha=0.6 if mod == 'whisker' else 0.35,
                       marker='o' if mod == 'whisker' else 's')
            if len(v):
                ax.errorbar(x + 0.25, v.mean(), v.sem() if len(v) > 1 else 0, fmt='k_', ms=12, lw=1.2)
        txt = []
        for mod in MODS:
            a = mouse[(mouse.cohort == 'R+') & (mouse['mod'] == mod)][mcol].dropna()
            b = mouse[(mouse.cohort == 'R-') & (mouse['mod'] == mod)][mcol].dropna()
            if len(a) > 2 and len(b) > 2:
                pu = stats.mannwhitneyu(a, b).pvalue
                pt = stats.ttest_ind(a, b, equal_var=False).pvalue
                rows.append(dict(metric=mcol, test='R+ vs R-', subset=mod, n1=len(a), n2=len(b), p_mwu=pu, p_welch=pt))
                txt.append(f'R+ vs R- {mod[0].upper()}: MWU p={pu:.3f}, Welch p={pt:.3f}')
        for c in COHORTS:
            w = mouse[(mouse.cohort == c)].pivot(index='mouse', columns='mod', values=mcol).dropna()
            if len(w) > 2 and {'whisker', 'auditory'} <= set(w.columns):
                pw = stats.wilcoxon(w.whisker, w.auditory).pvalue
                pt = stats.ttest_rel(w.whisker, w.auditory).pvalue
                rows.append(dict(metric=mcol, test='W vs A (paired)', subset=c, n1=len(w), n2=len(w), p_mwu=pw,
                                 p_welch=pt))
                txt.append(f'W vs A {c}: Wilcoxon p={pw:.3f}, paired t p={pt:.3f}')
        ax.set_xticks(list(pos.values()), [f'{c}\n{m}' for c, m in pos])
        ax.set_ylabel(f'{lab} ({unit})', fontsize=8)
        ax.set_title('\n'.join(txt), fontsize=5.5)
        ax.tick_params(labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'{STAGE_LABEL[stage]}: per-mouse medians over hit trials (dots = mice; mean +- SEM). Tests '
                 f'uncorrected', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / 'figures' / f'000_{stage}_metrics.png', dpi=110)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / 'tables' / f'000_{stage}_stats.csv', index=False)
    return pd.DataFrame(rows)


def main():
    for d in ('figures', 'tables', 'cache'):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    stages = [sys.argv[sys.argv.index('--stage') + 1]] if '--stage' in sys.argv else list(STAGES)
    sess = sessions_and_cohorts()
    sess = sess[sess.stage.isin(stages)]
    if '--pilot' in sys.argv:  # first N sessions per stage x cohort
        n = int(sys.argv[sys.argv.index('--pilot') + 1])
        sess = sess.groupby(['stage', 'cohort']).head(n)
    print(sess.groupby(['stage', 'cohort']).agg(sessions=('session', 'size'), mice=('mouse', 'nunique')).to_string(),
          flush=True)
    if '--plot-only' not in sys.argv:
        with ProcessPoolExecutor(12) as ex:
            done = [s for s in ex.map(one, sess.to_dict('records')) if s]
        print('sessions cached', len(done), 'of', len(sess), flush=True)
    for st in stages:
        ss = sess[sess.stage == st]
        acc, extras, trials = collect(ss)
        if 'grid' not in extras:
            continue
        n = trials.groupby(['cohort', 'trial_type']).agg(mice=('mouse', 'nunique'), sessions=('session', 'nunique'),
                                                         trials=('event_time', 'size'),
                                                         reaches=('reach_latency', lambda x: int(x.notna().sum())))
        n.to_csv(OUT / 'tables' / f'000_{st}_n.csv')
        print(st, '\n', n.to_string(), flush=True)
        fig_timecourses(acc, extras, st)
        fig_tongue(acc, extras, st)
        fig_tongue(acc, extras, st, frame='spout')
        fig_whisker(acc, extras, st)
        fig_examples(pick_examples(sess, st), st)
        s = fig_metrics(trials, st)
        print(s.round(4).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
