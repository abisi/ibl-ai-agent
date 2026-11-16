"""Copy the per-session Facemap outputs into the per-mouse results folders (user 2026-11-10).

Layout (same day folder as learning_curve / pupil_qc: NWB session_description without '+', e.g. whisker_0):
  combined_results_ks4/<mouse>/<day>/facemap/
    <session>_facemap_fullframe.npz        002: full-frame motion SVD + movie SVD (first 100 components each), motion
                                           energy per frame, mean frame, first 10 spatial masks
    <session>_facemap_roi_<roi>.npz        005: eye / snout / whisker / jaw ROI motion + movie SVD (first 50), energy
    <session>_clips_index.npz              001: clip frame -> trial / time index (needed to read both files above)
    params.json                            Facemap settings and provenance (sources, copy date)
Not copied: the clip videos (clips/*.mp4, input only) and the Kuma working folders. Sources are left in place.
Existing destination files of the same size are skipped. Log: ssl-facemap-lick-decoding/copy_to_mouse_folders.csv
Run on haas: nice -n 10 ~/code/ibl-ai-agent/.venv/bin/python 009_copy_to_mouse_folders.py [--dry-run]
"""

import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import h5py
import pandas as pd

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
SRC = ROOT / "ssl-facemap-lick-decoding"
NWB = Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
ROIS = ("eye", "snout", "whisker", "jaw")
PARAMS = {
    "clips": "peri-trial clips of the raw side video, -2.5..+0.5 s around active whisker trials (context != passive, "
             "perf != 6, early_lick == 0), 2x binned (320 x 240), H.264 CRF 18 (001_extract_clips.py)",
    "fullframe": "Facemap process.run, no ROI (full frame), sbin 2 (160 x 120), motion SVD + movie SVD, first 100 "
                 "components saved, motion energy per frame, mean frame, 10 spatial masks (002_kuma_facemap.py)",
    "roi": "automatic eye / snout / whisker-pad / jaw ROIs (004_roi_auto.py) cropped from the clips, Facemap sbin 1, "
           "first 50 motion / movie SVD components, motion energy per frame (005_kuma_roi_facemap.py)",
    "clips_index": "clip frame -> trial and time from trial start (001_extract_clips.py)",
    "code": "ibl-ai-agent projects/ssl-facemap-lick-decoding",
}


def day_folder(session):
    with h5py.File(NWB / f"{session}.nwb", "r") as f:
        d = f["session_description"][()]
    return (d.decode() if isinstance(d, bytes) else str(d)).replace("+", "")


def main(dry_run):
    rows = []
    sessions = sorted(p.name.replace("_facemap_small.npz", "") for p in (SRC / "facemap").glob("*_facemap_small.npz"))
    for session in sessions:
        mouse = session.split("_")[0]
        try:
            dest = ROOT / mouse / day_folder(session) / "facemap"
        except (OSError, KeyError) as err:
            rows.append({"session": session, "file": "", "status": f"no NWB / description: {err}"})
            continue
        pairs = [(SRC / "facemap" / f"{session}_facemap_small.npz", f"{session}_facemap_fullframe.npz"),
                 (SRC / "clips" / f"{session}_clips.npz", f"{session}_clips_index.npz")]
        pairs += [(SRC / "roi" / "facemap" / f"{session}_{r}_facemap_small.npz", f"{session}_facemap_roi_{r}.npz")
                  for r in ROIS]
        for src, name in pairs:
            out = dest / name
            if not src.exists():
                status = "source missing"
            elif out.exists() and out.stat().st_size == src.stat().st_size:
                status = "exists"
            else:
                status = "would copy" if dry_run else "copied"
                if not dry_run:
                    dest.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(".npz.part")
                    shutil.copyfile(src, tmp)  # copyfile: the NAS refuses utime (copy2)
                    tmp.replace(out)
            rows.append({"session": session, "dest": str(dest), "file": name, "source": str(src), "status": status})
        if not dry_run and dest.exists():
            (dest / "params.json").write_text(json.dumps({**PARAMS, "session": session, "copied": str(date.today()),
                                                          "sources": [str(s) for s, _ in pairs if s.exists()]},
                                                         indent=1))
    log = pd.DataFrame(rows)
    if not dry_run:
        log.to_csv(SRC / "copy_to_mouse_folders.csv", index=False)
    print(log.groupby("status").size().to_string())
    print(log.dest.nunique() if "dest" in log else 0, "destination folders")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    main(ap.parse_args().dry_run)
