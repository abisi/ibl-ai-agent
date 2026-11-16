"""Human reference (training_plan.md, user 2026-11-09): a separate DLC project with 100 frames drawn at random (seed 0)
from the test split of the merged model, renamed and shuffled, without labels, for blind relabelling in the DLC GUI.

Project: Axel_Bisi/DLCProjects/Tracking2D-Axel-relabel-2026-11-09 (same bodyparts, colormap and scorer as the merged
project), one labelled-data folder `relabel_100` (img000.png .. img099.png). The key linking each image to its source
frame is written OUTSIDE the project, in DLCProjects/merged_training_2026-11-09/relabel_100_key.csv.

Run on the PC (DEEPLABCUT env): python 012_make_relabel_project.py
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(r"M:\analysis\Axel_Bisi\DLCProjects")
SRC = ROOT / "Tracking2D-Axel-2026-11-04"
NEW = ROOT / "Tracking2D-Axel-relabel-2026-11-09"
SIDE = ROOT / "merged_training_2026-11-09"
N, SEED = 100, 0


def main():
    if NEW.exists():
        raise SystemExit(f"{NEW} exists: not overwritten")
    split = pd.read_csv(SIDE / "split.csv")
    test = split[split.set == "test"].reset_index(drop=True)
    pick = test.sample(N, random_state=SEED).reset_index(drop=True)  # random order = presentation order
    folder = NEW / "labeled-data" / "relabel_100"
    folder.mkdir(parents=True)
    rows = []
    for i, r in pick.iterrows():
        src = SRC / Path(r.image)
        dst = folder / f"img{i:03d}.png"
        shutil.copy2(src, dst)
        parts = Path(r.image).parts
        rows.append({"relabel_image": f"labeled-data/relabel_100/{dst.name}", "source_image": r.image,
                     "source_folder": parts[-2], "mouse": parts[-2].split("_")[0],
                     "view": "top" if "_top" in parts[-2] else "side", "split_index": r["index"]})
    pd.DataFrame(rows).to_csv(SIDE / "relabel_100_key.csv", index=False)
    cfg = yaml.safe_load(open(SRC / "config.yaml", encoding="utf-8"))
    cfg.update({"Task": "Tracking2D-relabel", "date": "Nov9", "project_path": str(NEW),
                "video_sets": {str(NEW / "videos" / "relabel_100.mp4"): {"crop": "0, 640, 0, 480"}}})
    (NEW / "videos").mkdir()
    yaml.safe_dump(cfg, open(NEW / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    key = pd.DataFrame(rows)
    print(f"{NEW}: {len(rows)} frames ({key.view.value_counts().to_dict()}), {key.mouse.nunique()} mice; key in "
          f"{SIDE / 'relabel_100_key.csv'}")


if __name__ == "__main__":
    main()
