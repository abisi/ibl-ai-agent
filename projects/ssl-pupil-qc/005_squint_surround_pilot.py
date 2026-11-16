"""Squint pilot, step 2 (user, 2026-11-04): remove global brightness changes (illumination, exposure, head position)
from the eye-ROI brightness by a surround reference. A larger box (outer = eye ROI x OUTER_SCALE, same centre, clipped
to the frame) is measured per frame (video_cache/eye_intensity_outer.npy, ~12 min per session); the surround ring mean
= (outer sum - eye sum) / (outer area - eye area); contrast = eye mean / ring mean (an open eye is darker than the fur
around it; a closing lid brings the ratio towards 1).
Run (haas): python 005_squint_surround_pilot.py [session ...]
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import pupil_qc as pq
import video

PILOT = ['AB144_20241205_111557', 'AB126_20240822_114405', 'AB140_20241129_104154', 'MH021_20250309_122552',
         'MH037_20250524_143522']
OUTER_SCALE = 2.2


def outer_box(roi, frame_shape=(480, 640)):
    x, y, w, h = roi
    cx, cy = x + w / 2, y + h / 2
    hw, hh = OUTER_SCALE * w / 2, OUTER_SCALE * h / 2
    x0, y0 = int(max(cx - hw, 0)), int(max(cy - hh, 0))
    x1, y1 = int(min(cx + hw, frame_shape[1])), int(min(cy + hh, frame_shape[0]))
    return [x0, y0, x1 - x0, y1 - y0]


def one(session_id):
    d = next(Path(p).parent for p in pq.RESULTS.glob(f'{session_id[:5]}/*/pupil_qc/{session_id}_frames.parquet'))
    cache = d / 'video_cache'
    meta = json.loads((cache / 'roi.json').read_text())
    out = cache / 'eye_intensity_outer.npy'
    box = outer_box(meta['roi'])
    if not out.exists():
        np.save(out, video.eye_intensity(meta['video'], box))
    (cache / 'roi_outer.json').write_text(json.dumps({'roi_outer': box, 'scale': OUTER_SCALE, 'video': meta['video']}))
    return session_id, box


if __name__ == '__main__':
    with Pool(5) as pool:
        for sid, box in pool.imap_unordered(one, sys.argv[1:] or PILOT):
            print(sid, box, flush=True)
