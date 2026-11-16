"""Label corrections in the merged DLC project (user 2026-11-11: "Yes correct it").

Found by the per-mouse test error (019 panel q: AB083, AB096 above the human error) and a geometric scan of every side
label (pupil points > 60 px apart, nose_tip < 120 px from the pupil or behind nose_base), then checked against the
model predictions and the label coordinates:
  - AB096 0221 13-10 (22 frames) and AB117 0724 (5 frames): nose_tip placed on the pupil_bottom point -> removed;
  - AB083 img035328 / img053904: pupil x4 + nose_tip are other body parts' positions (nose_base, jaw, jaw_ref, tongue,
    particle; labels shifted by one body part list) -> removed;
  - AB083 img065450: pupil_bottom and nose_tip swapped -> swapped back;
  - AB133 img0242467 (pupil_top / left / right) and AB134 img0075654 / img0086187 (pupil_top / left): pupil points
    on the jaw / chin / whiskers -> removed.
Removed = set to NaN (unlabelled); placing them correctly needs the DLC labelling GUI. Applied to CollectedData_Axel
.h5 + .csv of each folder (backups *_backup_20261111.*) and to the exported split file kuma/split_ground_truth.h5
(backup, same frames and split). The Kuma project copy is synced separately (rclone of the five folders' files).
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 041_fix_labels.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

PROJ = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/Tracking2D-Axel-2026-11-04")
SPLIT = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/merged_training_2026-11-09/kuma/split_ground_truth.h5")
PUPIL = ["pupil_top", "pupil_left", "pupil_right", "pupil_bottom"]
NOSE_ON_PUPIL_FOLDERS = ["AB096_side 24-02-21 13-10-14-converted", "AB117_side 24-07-24 12-20-59"]
REMOVE = {
    ("AB083_side-converted", "img035328.png"): PUPIL + ["nose_tip"],
    ("AB083_side-converted", "img053904.png"): PUPIL + ["nose_tip"],
    ("AB133_side 24-11-07 10-15-38", "img0242467.png"): ["pupil_top", "pupil_left", "pupil_right"],
    ("AB134_side 24-11-12 14-24-26", "img0075654.png"): ["pupil_top", "pupil_left"],
    ("AB134_side 24-11-12 14-24-26", "img0086187.png"): ["pupil_top", "pupil_left"],
}
SWAP = {("AB083_side-converted", "img065450.png"): ("pupil_bottom", "nose_tip")}
TAG = "_backup_20261111"


def fix(df, folder_of, image_of):
    """Apply the corrections to a CollectedData-like frame (columns scorer / bodyparts / coords). Returns a log."""
    log = []
    sc = df.columns.get_level_values(0)[0]
    for i in df.index:
        key = (folder_of(i), image_of(i))
        if key[0] in NOSE_ON_PUPIL_FOLDERS:
            nx, ny = df.loc[i, (sc, "nose_tip", "x")], df.loc[i, (sc, "nose_tip", "y")]
            px = np.nanmean([df.loc[i, (sc, p, "x")] for p in PUPIL])
            py = np.nanmean([df.loc[i, (sc, p, "y")] for p in PUPIL])
            if np.isfinite(nx) and np.hypot(nx - px, ny - py) < 120:
                df.loc[i, [(sc, "nose_tip", "x"), (sc, "nose_tip", "y")]] = np.nan
                log.append((*key, "nose_tip removed (on the pupil)"))
        for b in REMOVE.get(key, []):
            df.loc[i, [(sc, b, "x"), (sc, b, "y")]] = np.nan
            log.append((*key, f"{b} removed"))
        if key in SWAP:
            a, b = SWAP[key]
            va = df.loc[i, [(sc, a, "x"), (sc, a, "y")]].to_numpy().copy()
            df.loc[i, [(sc, a, "x"), (sc, a, "y")]] = df.loc[i, [(sc, b, "x"), (sc, b, "y")]].to_numpy()
            df.loc[i, [(sc, b, "x"), (sc, b, "y")]] = va
            log.append((*key, f"{a} <-> {b} swapped"))
    return log


def main():
    folders = sorted({k[0] for k in REMOVE} | {k[0] for k in SWAP} | set(NOSE_ON_PUPIL_FOLDERS))
    for f in folders:
        h5 = PROJ / "labeled-data" / f / "CollectedData_Axel.h5"
        for ext in (".h5", ".csv"):
            src, bak = h5.with_suffix(ext), h5.with_name(f"CollectedData_Axel{TAG}{ext}")
            if not bak.exists():
                shutil.copyfile(src, bak)
        df = pd.read_hdf(h5)
        log = fix(df, lambda i: i[1], lambda i: i[2])
        df.to_hdf(h5, key="keypoints", mode="w")
        df.to_csv(h5.with_suffix(".csv"))
        print(f, len(log), "changes")
    bak = SPLIT.with_name(SPLIT.stem + TAG + ".h5")
    if not bak.exists():
        shutil.copyfile(SPLIT, bak)
    sp = pd.read_hdf(SPLIT)
    split = sp.pop(("split", "", ""))
    log = fix(sp, lambda i: i[1], lambda i: i[2])
    sp[("split", "", "")] = split
    sp.to_hdf(SPLIT, key="df", mode="w")
    out = pd.DataFrame(log, columns=["folder", "image", "change"])
    out.to_csv(SPLIT.parent / "label_fixes_20261111.csv", index=False)
    print(len(out), "changes in split_ground_truth.h5 (", out.image.nunique(), "frames )")
    print(out.groupby(["folder", "change"]).size().to_string())


if __name__ == "__main__":
    main()
