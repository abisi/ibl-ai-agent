"""013 -- Evidence that frames are lost in chunks: a lost chunk should leave a discontinuity between two consecutive
saved frames (the mouse "teleports"), at the steps placed by the lick-based fit (010) (user 2026-11-16).

(1) DLC jump detector over the whole session, per camera: per frame f and moving keypoint k (likelihood >= 0.6 at f-1
    and f), z_k(f) = |p_k(f) - p_k(f-1)| / median_f |p_k(f) - p_k(f-1)|; score S(f) = fraction of valid keypoints with
    z >= Z_JUMP (>= 3 valid keypoints). Fixed points (spout, jaw_ref) excluded. Controls: AB151 0130, MH064 0114
    (never flagged, frames = pulses).
(2) Raw video around each fitted side step (+-STEP_WIN frames, decoded with ffmpeg at 1/4 size, grey): mean absolute
    difference between consecutive frames; the two frames on either side of the largest difference.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/drift_review/chunk_drop_evidence_<date>.pdf,
tables/chunk_candidates.csv (top frames by S per session / camera, with the nearest fitted step)
Run (haas, ibl-ai-agent .venv; ffmpeg on PATH): python 013_chunk_drop_evidence.py
"""
import datetime
import glob
import importlib.util
import pathlib
import subprocess

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m006', HERE / '006_pairing_evidence.py')
m006 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m006)
OUT = m006.OUT / 'drift_review'
SESSIONS = ['AB092_20231205_140109', 'MH017_20250302_120918', 'MH013_20250219_170437',
            'AB151_20250130_131124', 'MH064_20260114_102210']
MOVING = {'side': ['jaw', 'nose_tip', 'nose_base', 'pupil_top', 'pupil_bottom', 'pupil_left', 'pupil_right', 'particle',
                   'tongue'],
          'top': ['top_nose_tip', 'top_nose_base', 'whisker_base', 'whisker_tip', 'top_particle']}
Z_JUMP, MIN_LIK, STEP_WIN, N_TOP = 8.0, 0.6, 400, 25
VIDEO_ROOTS = ['/mnt/lsens-data/{m}/Recording/{s}/Video', '/mnt/lsens-analysis/Myriam_Hamon/data/{m}/{s}/Video',
               '/mnt/lsens-analysis/Axel_Bisi/data/{m}/{s}/Video']


def jump_score(session_id, view):
    with h5py.File(m006.NWB_DIR / f'{session_id}.nwb', 'r') as f:
        b = f['processing/behavior/BehavioralTimeSeries']
        zs, valid = [], []
        for k in MOVING[view]:
            if f'{k}_x' not in b:
                continue
            x, y = b[f'{k}_x']['data'][:].astype(float), b[f'{k}_y']['data'][:].astype(float)
            lik = b[f'{k}_likelihood']['data'][:].astype(float)
            d = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
            ok = (lik >= MIN_LIK) & (np.r_[0, lik[:-1]] >= MIN_LIK) & np.isfinite(d)
            typ = np.nanmedian(d[ok]) if ok.any() else np.nan
            zs.append(np.where(ok, d / max(typ, 1e-3), np.nan))
            valid.append(ok)
    z = np.array(zs)
    n_ok = np.sum(np.array(valid), 0)
    S = np.where(n_ok >= 3, np.nansum(z >= Z_JUMP, 0) / np.maximum(n_ok, 1), np.nan)
    return S, n_ok


def video_file(session_id, view):
    m = session_id[:5]
    for r in VIDEO_ROOTS:
        hits = [h for h in glob.glob(r.format(m=m, s=session_id) + f'/*{view}*')
                if h.endswith(('.avi', '.mp4')) and 'DLC' not in h and 'backup' not in h]
        if hits:
            return hits[0]
    return None


def frames(video, f0, n, fps=200.0):
    """Grey frames f0 .. f0 + n - 1 at 1/4 size (accurate seek)."""
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height',
                            '-of', 'csv=p=0', video], capture_output=True, text=True).stdout.strip().split(',')
    w, h = int(probe[0]) // 4, int(probe[1]) // 4
    cmd = ['ffmpeg', '-v', 'error', '-ss', f'{f0 / fps:.4f}', '-i', video, '-frames:v', str(n),
           '-vf', f'scale={w}:{h},format=gray', '-f', 'rawvideo', '-']
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.int16)


def steps_of(session_id, view):
    p = OUT / 'tables' / f'refit_{session_id}_{view}.csv'
    if not p.exists():
        return pd.DataFrame(columns=['frame', 'size'])
    sg = pd.read_csv(p)
    return pd.DataFrame(dict(frame=sg.frame_start.iloc[1:].to_numpy(), size=np.diff(sg.D.to_numpy())))


def main():
    rows = []
    out = OUT / f'chunk_drop_evidence_{datetime.date.today()}.pdf'
    scores = {}
    with PdfPages(out) as pdf:
        # page 1: whole-session jump scores, fitted steps marked
        fig, axs = plt.subplots(len(SESSIONS), 2, figsize=(16.5, 3 * len(SESSIONS)), squeeze=False)
        for r, s in enumerate(SESSIONS):
            for c, view in enumerate(('side', 'top')):
                S, n_ok = jump_score(s, view)
                scores[(s, view)] = S
                st = steps_of(s, view if view == 'side' else 'side')  # top: side steps shown for reference
                ax = axs[r, c]
                f = np.arange(len(S))
                ax.plot(f / 200, S, lw=0.4, color='0.3')
                for fr, sz in zip(st.frame, st['size']):
                    ax.axvline(fr / 200, color='r', lw=0.8, alpha=0.6)
                    ax.text(fr / 200, 1.02, f'+{int(sz)}', color='r', fontsize=6, ha='center')
                ax.set_ylim(0, 1.1)
                ax.set_title(f'{s} {view}: fraction of moving keypoints jumping >= {Z_JUMP:.0f}x typical '
                             f'(red = side steps from the lick fit)', fontsize=8)
                ax.set_xlabel('video time (frame / 200 Hz, s)', fontsize=7)
                ax.tick_params(labelsize=6)
                top = np.argsort(np.nan_to_num(S, nan=-1))[::-1][:N_TOP]
                for fr in top:
                    near = st.frame.to_numpy()
                    d = int(np.min(np.abs(near - fr))) if len(near) else np.nan
                    rows.append(dict(session_id=s, view=view, frame=int(fr), score=float(S[fr]),
                                     n_valid=int(n_ok[fr]), frames_to_nearest_side_step=d))
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(OUT / 'figures' / '013_scores.png', dpi=70)
        plt.close(fig)
        # pages 2+: around each fitted side step: S (both cameras) + raw video frame difference + frames across max
        for s in SESSIONS[:3]:
            st = steps_of(s, 'side')
            for fr, sz in zip(st.frame, st['size']):
                fig, axs = plt.subplots(2, 4, figsize=(16.5, 7.5), gridspec_kw=dict(width_ratios=[1.4, 1.4, 1, 1]))
                for c, view in enumerate(('side', 'top')):
                    S = scores[(s, view)]
                    a, b = max(fr - STEP_WIN, 0), min(fr + STEP_WIN, len(S))
                    ax = axs[0, c]
                    ax.plot(np.arange(a, b) - fr, S[a:b], '.-', ms=2, lw=0.5)
                    ax.axvline(0, color='r', lw=0.6)
                    ax.set_title(f'{view} DLC jump score around side step (frame {fr:,}, +{int(sz)})', fontsize=8)
                    ax.set_xlabel('frames from fitted step', fontsize=7)
                    ax.set_ylim(0, 1.05)
                    vid = video_file(s, view)
                    ax = axs[1, c]
                    if vid is None:
                        ax.set_title(f'{view}: video not found', fontsize=8)
                        continue
                    fr_view = fr  # top frame index differs after earlier top drops: scan the same window
                    im = frames(vid, max(fr_view - STEP_WIN, 0), 2 * STEP_WIN)
                    if len(im) < 3:
                        ax.set_title(f'{view}: decode failed', fontsize=8)
                        continue
                    diff = np.abs(np.diff(im, axis=0)).mean((1, 2))
                    x0 = max(fr_view - STEP_WIN, 0) - fr_view
                    ax.plot(np.arange(len(diff)) + x0 + 1, diff, '.-', ms=2, lw=0.5)
                    ax.axvline(0, color='r', lw=0.6)
                    k = int(np.argmax(diff))
                    ax.set_title(f'{view} raw video: mean |frame - previous frame|; max at {k + x0 + 1:+d} '
                                 f'({diff[k] / np.median(diff):.1f}x median)', fontsize=8)
                    ax.set_xlabel('frames from fitted step', fontsize=7)
                    rows.append(dict(session_id=s, view=view, frame=int(fr_view + k + x0 + 1), score=np.nan,
                                     video_diff_ratio=float(diff[k] / np.median(diff)), side_step=int(fr),
                                     step_size=int(sz)))
                    for j, (dk, lab) in enumerate(((k, 'before'), (k + 1, 'after'))):
                        axi = axs[j, 2 + c]
                        axi.imshow(im[dk], cmap='gray')
                        axi.set_title(f'{view} frame {fr_view + x0 + dk:,} ({lab} the largest change)', fontsize=7)
                        axi.set_axis_off()
                fig.suptitle(f'{s}: side step at frame {fr:,} (+{int(sz)} pulses)', fontsize=10, x=0.02, ha='left')
                fig.tight_layout()
                pdf.savefig(fig)
                fig.savefig(OUT / 'figures' / f'013_{s}_{fr}.png', dpi=70)
                plt.close(fig)
                print(s, fr, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'tables' / 'chunk_candidates.csv', index=False)
    print(out)


if __name__ == '__main__':
    main()
