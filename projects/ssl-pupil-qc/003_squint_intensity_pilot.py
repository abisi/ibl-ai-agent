"""Pilot (user, 2026-11-04): detect extreme squints (eye nearly closed, the four pupil points collapse together so the
height / width ratio stays ~1 and the QC calls the frames valid) from the eye-ROI video intensity: a closing lid
replaces the dark eye by brighter fur / lid. Step 1 (this script): eye-ROI mean intensity of every frame
(video.VideoCache, cached as eye_intensity.npy in the session's pupil_qc folder; ~12 min per session) for the pilot
sessions. The squint rule is chosen on these sessions afterwards.
Pilot: AB144, AB126, AB140 (narrowing seen in ssl-dlc-utils-validation 015), MH021 (narrow eye), MH037 (clean control).
Run (haas): python 003_squint_intensity_pilot.py [session ...]
"""
import sys
from multiprocessing import Pool

import pupil_qc as pq
import video

PILOT = ['AB144_20241205_111557', 'AB126_20240822_114405', 'AB140_20241129_104154', 'MH021_20250309_122552',
         'MH037_20250524_143522']


def one(session_id):
    r = pq.run_session(session_id)
    vid = pq.find_video(r['meta'], r['df'])
    if not vid or not vid['video']:
        return session_id, 'no video'
    cache = video.VideoCache(r['out'], vid['video'])
    roi = cache.roi(r['df'], r['c']['state'] == pq.STATE['valid'])
    x = cache.intensity(roi, compute=True)
    return session_id, f'{len(x)} frames, roi {roi}, video {vid["video"]}, offset {vid["offset"]}'


if __name__ == '__main__':
    with Pool(5) as pool:
        for sid, msg in pool.imap_unordered(one, sys.argv[1:] or PILOT):
            print(sid, msg, flush=True)
