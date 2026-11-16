"""Shuffle 3 / 5 / 6 benchmark, step 1 (haas): ground truth + training membership of every labelled frame (2026-11-04).

Ground truth = every labelled frame of the merged project Tracking2D-Axel-2026-11-04 (CollectedData_*.h5, frames with
>= 1 labelled bodypart). Training membership of each shuffle = images listed in its training .mat file:
shuffle 3 from Axel's copy (3,444-frame dataset), shuffles 5 and 6 from Myriam's copy (model files of shuffles 3/5
are identical in both copies). A frame is "held out" for a shuffle when the shuffle never trained on it (its own
test split, or labelled after / outside its dataset).
Writes combined_results_ks4/ssl-dlc-kuma/shuffle_bench/{gt.csv, images.txt (rclone --files-from list)}.
Usage (haas): ~/code/ibl-ai-agent/.venv/bin/python 007_shuffle_bench_prepare.py
"""

from pathlib import Path

import pandas as pd
import scipy.io as sio

PROJ = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/Tracking2D-Axel-2026-11-04")
OUT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma/shuffle_bench")
TD = "training-datasets/iteration-0/UnaugmentedDataSet_Tracking2DAug10/Tracking2D_Axel95shuffle{}.mat"
MATS = {3: Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/Tracking2D-Axel-2024-08-10"),
        5: Path("/mnt/lsens-analysis/Myriam_Hamon/DLCprojects/Tracking2D-Axel-2024-08-10"),
        6: Path("/mnt/lsens-analysis/Myriam_Hamon/DLCprojects/Tracking2D-Axel-2024-08-10")}


def norm_folder(f):
    return "MH" + f[1:] if f.startswith("M0") else f  # MH006 folders renamed from M006_


def train_set(shuffle):
    d = sio.loadmat(MATS[shuffle] / TD.format(shuffle))["dataset"][0]
    return {(norm_folder(str(e["image"][0, 1]).strip()), str(e["image"][0, 2]).strip()) for e in d}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in sorted(PROJ.glob("labeled-data/*/CollectedData_*.h5")):
        df = pd.read_hdf(h)
        df = df[df.notna().any(axis=1)]
        df.columns = [f"{bp}_{c}" for _, bp, c in df.columns]
        if isinstance(df.index, pd.MultiIndex):
            df.index = [f"{a}/{b}" for _, a, b in df.index]
        else:
            df.index = [str(i).replace("\\", "/").split("labeled-data/")[-1] for i in df.index]
        df.insert(0, "folder", [i.split("/")[0] for i in df.index])
        df.insert(1, "image", [i.split("/")[1] for i in df.index])
        rows.append(df.reset_index(drop=True))
    gt = pd.concat(rows, ignore_index=True)
    gt.insert(2, "mouse_id", gt.folder.str.split("_").str[0])
    gt.insert(3, "view", gt.folder.str.extract(r"_(side|top)")[0])
    gt.insert(4, "cohort", gt.mouse_id.str[:2])
    for s in MATS:
        tr = train_set(s)
        gt[f"train_s{s}"] = [(f, i) in tr for f, i in zip(gt.folder, gt.image)]
        print(f"shuffle {s}: {len(tr)} training images, {gt[f'train_s{s}'].sum()} found in merged ground truth")
    gt.to_csv(OUT / "gt.csv", index=False)
    (OUT / "images.txt").write_text("\n".join(gt.folder + "/" + gt.image) + "\n")
    print(f"{len(gt)} labelled frames, {gt.folder.nunique()} folders; held out of all three: "
          f"{(~gt[[f'train_s{s}' for s in MATS]].any(axis=1)).sum()}")
    print(gt.groupby(["cohort", "view"])[[f"train_s{s}" for s in MATS]].agg(["sum", "size"]))


if __name__ == "__main__":
    main()
