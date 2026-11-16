"""Facemap lick decoding, ROI step 2 (Kuma, one SLURM array task per ROI video): Facemap SVDs of one ROI.

Input: roi/videos/<session>_<roi>.mp4 from 004_roi_auto.py (automatic eye / snout / whisker pad / jaw ROIs).
Same Facemap call as the full-frame runs (no ROI inside Facemap = the whole cropped video; motion SVD and movie SVD)
but sbin 1, since the ROIs are only ~60x60 px. Saved: first 50 components of each, motion energy per frame, average
frame (small npz); Facemap's full proc is deleted.

Run on Kuma:  bash 005_kuma_roi_submit.sh <n_videos>   (index into sorted videos/*.mp4)
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
from facemap import process

BASE = Path("/work/lsens/bisi/facemap_roi_full")
N_KEEP = 50
SBIN = 1


def main(idx):
    video = sorted((BASE / "videos").glob("*.mp4"))[idx]
    name = video.stem  # <session>_<roi>
    small = BASE / "results" / f"{name}_facemap_small.npz"
    if small.exists():
        print(name, "exists")
        return
    out = BASE / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    savename = process.run([[str(video)]], sbin=SBIN, motSVD=True, movSVD=True, GUIobject=None, parent=None,
                           proc=None, savepath=str(out))
    proc = np.load(savename, allow_pickle=True).item()
    np.savez(small, motSVD=np.asarray(proc["motSVD"][0])[:, :N_KEEP].astype(np.float32),
             movSVD=np.asarray(proc["movSVD"][0])[:, :N_KEEP].astype(np.float32),
             motion=np.asarray(proc["motion"][0]).astype(np.float32),
             avgframe=np.asarray(proc["avgframe_reshape"]).astype(np.float32), sbin=SBIN, video=str(video),
             seconds=time.time() - t0)
    shutil.rmtree(out, ignore_errors=True)
    print(name, "seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
