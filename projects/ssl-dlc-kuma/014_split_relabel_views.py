"""Relabel project (user 2026-11-09): split the 100 blind frames into one folder per view, 50 each
(labeled-data/relabel_side_50, labeled-data/relabel_top_50; img000..img049 in the existing random order). Only allowed
while relabel_100 has no labels. The key file outside the project is rewritten with the new names.

Run on the PC (DEEPLABCUT env): python 014_split_relabel_views.py
"""

import shutil
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(r"M:\analysis\Axel_Bisi\DLCProjects")
PROJ = ROOT / "Tracking2D-Axel-relabel-2026-11-09"
KEY = ROOT / "merged_training_2026-11-09" / "relabel_100_key.csv"


def main():
    old = PROJ / "labeled-data" / "relabel_100"
    if list(old.glob("CollectedData_*")):
        raise SystemExit("relabel_100 already has labels: not split")
    key = pd.read_csv(KEY)
    rows = []
    for view in ("side", "top"):
        k = key[key.view == view].reset_index(drop=True)
        assert len(k) == 50, (view, len(k))
        new = PROJ / "labeled-data" / f"relabel_{view}_50"
        new.mkdir()
        for i, r in k.iterrows():
            src = PROJ / r.relabel_image
            dst = new / f"img{i:03d}.png"
            shutil.move(str(src), str(dst))
            rows.append({**r.to_dict(), "relabel_image": f"labeled-data/{new.name}/{dst.name}"})
    old.rmdir()
    out = pd.DataFrame(rows)
    out.to_csv(KEY.with_name("relabel_key.csv"), index=False)
    KEY.unlink()
    cfg = yaml.safe_load(open(PROJ / "config.yaml", encoding="utf-8"))
    cfg["video_sets"] = {str(PROJ / "videos" / f"relabel_{v}_50.mp4"): {"crop": "0, 640, 0, 480"} for v in ("side", "top")}
    yaml.safe_dump(cfg, open(PROJ / "config.yaml", "w", encoding="utf-8"), sort_keys=False)
    print(out.groupby("view").size().to_dict(), "->", KEY.with_name("relabel_key.csv"))
    print(sorted(p.name for p in (PROJ / "labeled-data").iterdir()))


if __name__ == "__main__":
    main()
