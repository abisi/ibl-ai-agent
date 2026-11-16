"""Facemap lick decoding, step 2 (Kuma, one SLURM array task per session): Facemap SVDs of the peri-trial clips video.

Same Facemap choices as the ssl-facemap-pilot (full-frame, no ROI; sbin 2 -> 160x120; motion SVD and movie SVD,
500 components computed), applied to <session>_clips.mp4 from step 1. The spatial basis is therefore fit on
peri-whisker-trial frames only. Saved: first 100 components of each, summed motion energy per frame, average frame and
the first 10 motion masks (small npz); Facemap's full proc is deleted to save space.

Run on Kuma:  sbatch --array=0-<n-1>%<k> 002_kuma_job.sh   (index into sorted clips/*_clips.mp4)
"""

import shutil
import sys
import time
from pathlib import Path

import numpy as np
from facemap import process

BASE = Path("/work/lsens/bisi/facemap_full")
N_KEEP = 100
SBIN = 2


def main(idx):
    videos = sorted((BASE / "clips").glob("*_clips.mp4"))
    video = videos[idx]
    sid = video.name.replace("_clips.mp4", "")
    small = BASE / "results" / f"{sid}_facemap_small.npz"
    if small.exists():
        print(sid, "exists")
        return
    out = BASE / "results" / sid
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    savename = process.run([[str(video)]], sbin=SBIN, motSVD=True, movSVD=True, GUIobject=None, parent=None,
                           proc=None, savepath=str(out))
    proc = np.load(savename, allow_pickle=True).item()
    mot = np.asarray(proc["motSVD"][0])[:, :N_KEEP].astype(np.float32)
    mov = np.asarray(proc["movSVD"][0])[:, :N_KEEP].astype(np.float32)
    energy = np.asarray(proc["motion"][0]).astype(np.float32)
    masks = np.asarray(proc["motMask_reshape"][0])[..., :10].astype(np.float32) \
        if "motMask_reshape" in proc else np.array([])
    np.savez(small, motSVD=mot, movSVD=mov, motion=energy, avgframe=np.asarray(proc["avgframe_reshape"]).astype(
        np.float32), motMask=masks, sbin=SBIN, n_frames=mot.shape[0], video=str(video), seconds=time.time() - t0)
    shutil.rmtree(out, ignore_errors=True)
    print(sid, "frames", mot.shape[0], "seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
