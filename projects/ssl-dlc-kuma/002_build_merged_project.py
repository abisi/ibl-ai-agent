"""Build the merged SSL DLC labelling project (2026-11-04, user): Tracking2D-Axel-2026-11-04 on the NAS.

Source: Myriam_Hamon/DLCprojects/Tracking2D-Axel-2024-08-10 (its labeled-data is a superset of the Axel_Bisi copy:
300 folders, 4,542 labelled frames, 3,444 AB + 1,098 MH; copied unchanged into the new project beforehand).
This script (run on haas) writes the new config.yaml and fixes labels; it never touches the source projects.
  - Task / scorer / bodyparts / skeleton / plotting settings as the source (scorer Axel, 13 bodyparts)
  - project_path = UNC path of the new project (the GUI runs on Windows, M: = \\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab)
  - date Nov4; TrainingFraction 0.90 (larger test set for snapshot selection; training not started)
  - video_sets: one entry per labeled-data folder whose raw video is found in lsens-data (<mouse>/{Recording,
    Training}/<session>/Video/<folder name>.avi or .mp4), UNC paths, crop 0, 640, 0, 480
  - AB118_top 24-07-30 10-45-50: bodypart 'snout' (early label scheme, verified on images = nose tip) renamed to
    nose_tip; that CollectedData is re-indexed to the 13 config bodyparts (missing ones NaN), h5 + csv
Re-run with --add <video paths> to append new video_sets entries (frame extraction for never-labelled videos).
Run on haas: ~/code/unit_spikes_analysis/.venv/bin/python 002_build_merged_project.py
"""

import argparse
import glob
import os
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/Tracking2D-Axel-2026-11-04")
UNC = "\\\\sv-nas1.rcp.epfl.ch\\Petersen-Lab"
MOUNTS = {"/mnt/lsens-data": UNC + "\\data", "/mnt/lsens-analysis": UNC + "\\analysis"}
BODYPARTS = ["pupil_top", "pupil_left", "pupil_right", "pupil_bottom", "nose_tip", "nose_base", "jaw", "jaw_ref",
             "spout", "particle", "tongue", "whisker_base", "whisker_tip"]
SNOUT_FOLDER = "AB118_top 24-07-30 10-45-50"


def to_unc(p):
    for m, u in MOUNTS.items():
        if str(p).startswith(m):
            return u + str(p)[len(m):].replace("/", "\\")
    raise ValueError(p)


def find_video(folder):
    mouse = folder[:5]
    stem = folder
    for ext in (".avi", ".mp4"):
        hits = glob.glob(f"/mnt/lsens-data/{mouse}/*/*/Video/{glob.escape(stem)}{ext}")
        if hits:
            return sorted(hits)[0]
    return None


def fix_snout():
    d = ROOT / "labeled-data" / SNOUT_FOLDER
    h5 = glob.glob(str(d / "CollectedData_*.h5"))[0]
    x = pd.read_hdf(h5)
    if "snout" not in set(x.columns.get_level_values(1)):
        return "already fixed"
    x = x.rename(columns={"snout": "nose_tip"}, level=1)
    scorer = x.columns.get_level_values(0)[0]
    cols = pd.MultiIndex.from_product([[scorer], BODYPARTS, ["x", "y"]], names=x.columns.names)
    x = x.reindex(columns=cols)
    x.to_hdf(h5, key="df_with_missing", mode="w")
    x.to_csv(h5.replace(".h5", ".csv"))
    return f"renamed snout -> nose_tip ({int(x[(scorer, 'nose_tip', 'x')].notna().sum())} frames)"


def write_config(entries):
    src = (ROOT / "config_source_myriam.yaml").read_text()
    tail = src[src.index("\nskeleton:"):]
    tail = tail.replace("TrainingFraction:\n- 0.95", "TrainingFraction:\n- 0.9")
    vs = "".join(f"\n  ? {v}\n  : crop: 0, 640, 0, 480" for v in entries)
    head = ("    # Project definitions (do not edit)\nTask: Tracking2D\nscorer: Axel\ndate: Nov4\n"
            "multianimalproject: false\nidentity:\n\n\n    # Project path (change when moving around)\n"
            f"project_path: {to_unc(ROOT)}\n\n\n"
            "    # Default DeepLabCut engine to use for shuffle creation (either pytorch or tensorflow)\n"
            "engine: pytorch\n\n\n    # Annotation data set configuration (and individual video cropping parameters)\n"
            f"video_sets:{vs}\nbodyparts:\n" + "".join(f"- {b}\n" for b in BODYPARTS) + "\n\n")
    head = head.replace("    # ", "# ")
    (ROOT / "config.yaml").write_text(head + tail.lstrip("\n"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--add", nargs="*", default=[], help="extra raw video paths (/mnt/...) to add to video_sets")
    a = ap.parse_args()
    print(fix_snout())
    rows, entries = [], []
    for d in sorted(p for p in (ROOT / "labeled-data").iterdir() if p.is_dir()):
        n_png = len(list(d.glob("*.png")))
        v = find_video(d.name)
        rows.append(dict(folder=d.name, n_png=n_png, video=v))
        if v and (n_png > 0 or v in a.add):
            entries.append(to_unc(v))
    for v in a.add:
        u = to_unc(v)
        if u not in entries:
            entries.append(u)
    write_config(entries)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "merged_project_folders.csv", index=False)
    print(f"{len(df)} labeled-data folders, {int((df.n_png > 0).sum())} with frames, "
          f"{int(df.video.notna().sum())} with raw video found; video_sets entries: {len(entries)}")
    print("folders with frames but no raw video:", df[(df.n_png > 0) & df.video.isna()].folder.tolist())


if __name__ == "__main__":
    main()
