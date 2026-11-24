"""033 (Kuma) -- Facemap (MouseLand) motion / movie SVDs of the WHOLE raw side video (pilot, 2026-11-22), for the
representation comparison with BEAST and the keypoints. The existing Facemap outputs (ssl-facemap-lick-decoding) cover
peri-trial clips only.

Same choices as ssl-facemap-lick-decoding 002 (full frame, no ROI, motion SVD + movie SVD), with sbin 4 on the raw
640 x 480 avi (= the 2x-binned clips at sbin 2 there, 160 x 120). The SVD basis is fit per session by Facemap on its own
frame subsample. Kept: first 100 components of each, motion energy per frame, mean frame, first 10 motion masks.
Output: ext_data/facemap/<view>/<sid>_<view>_facemap.npz (frame i = video frame i = DLC frame i)
Run: python 033_facemap_full.py <sid>
"""
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path("/work/lsens/bisi/behaviour_motifs")
VIEW = os.environ.get("BM_VIEW", "side")
OUT = BASE / "ext_data" / "facemap" / VIEW
VID = BASE / "ext_data" / "beast" / VIEW / "videos"  # symlinks written by 032 links
N_KEEP, SBIN = 100, 4


def main(sid):
    from facemap import process
    OUT.mkdir(parents=True, exist_ok=True)
    small = OUT / f"{sid}_{VIEW}_facemap.npz"
    if small.exists():
        print("exists", small)
        return
    tmp = OUT / f"tmp_{sid}"
    tmp.mkdir(exist_ok=True)
    t0 = time.time()
    savename = process.run([[str(VID / f"{sid}.avi")]], sbin=SBIN, motSVD=True, movSVD=True, GUIobject=None, parent=None,
                           proc=None, savepath=str(tmp))
    proc = np.load(savename, allow_pickle=True).item()
    mot = np.asarray(proc["motSVD"][0])[:, :N_KEEP].astype(np.float32)
    mov = np.asarray(proc["movSVD"][0])[:, :N_KEEP].astype(np.float32)
    energy = np.asarray(proc["motion"][0]).astype(np.float32)
    masks = np.asarray(proc["motMask_reshape"][0])[..., :10].astype(np.float32) if "motMask_reshape" in proc else np.array([])
    np.savez(small, motSVD=mot, movSVD=mov, motion=energy, avgframe=np.asarray(proc["avgframe_reshape"]).astype(np.float32),
             motMask=masks, sbin=SBIN, n_frames=mot.shape[0], seconds=time.time() - t0)
    shutil.rmtree(tmp, ignore_errors=True)
    print(sid, "frames", mot.shape[0], "seconds", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
