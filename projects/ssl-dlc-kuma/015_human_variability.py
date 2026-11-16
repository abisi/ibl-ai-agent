"""Human reference (training_plan.md): blind relabelling (Tracking2D-Axel-relabel-2026-11-09, 50 side + 50 top test
frames) vs the original labels of the same frames in Tracking2D-Axel-2026-11-04.

Per frame and keypoint: labelled in both / only original / only relabel / neither, and the distance (px) where both.
Output: DLCProjects/merged_training_2026-11-09/human_variability_px.csv (one row per frame x keypoint); mm conversion
is added on haas by 016 (dlc_helpers NWB conversion).

Run on the PC (DEEPLABCUT env): python 015_human_variability.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"M:\analysis\Axel_Bisi\DLCProjects")
ORIG = ROOT / "Tracking2D-Axel-2026-11-04"
RELAB = ROOT / "Tracking2D-Axel-relabel-2026-11-09"
SIDE = ROOT / "merged_training_2026-11-09"


def xy(df, row):
    s = df.loc[row]
    s.index = s.index.droplevel(0)  # scorer
    return s.unstack()  # bodypart x (x, y)


def main():
    key = pd.read_csv(SIDE / "relabel_key.csv")
    relab = {f.parent.name: pd.read_hdf(f) for f in (RELAB / "labeled-data").glob("*/CollectedData_*.h5")}
    orig_cache, rows = {}, []
    for r in key.itertuples():
        folder, img = r.relabel_image.split("/")[1:]
        rl = relab[folder]
        rl_row = next(i for i in rl.index if (i[-1] if isinstance(i, tuple) else Path(i).name) == img)
        src_folder, src_img = Path(r.source_image).parts[-2:]
        if src_folder not in orig_cache:
            orig_cache[src_folder] = pd.read_hdf(next((ORIG / "labeled-data" / src_folder).glob("CollectedData_*.h5")))
        od = orig_cache[src_folder]
        od_row = next(i for i in od.index if (i[-1] if isinstance(i, tuple) else Path(i).name) == src_img)
        a, b = xy(od, od_row), xy(rl, rl_row)
        for bp in a.index:
            ao, bo = a.loc[bp].notna().all(), b.loc[bp].notna().all()
            d = float(np.hypot(*(a.loc[bp, ["x", "y"]].to_numpy(float) - b.loc[bp, ["x", "y"]].to_numpy(float)))) if ao and bo else np.nan
            rows.append({"relabel_image": r.relabel_image, "source_image": r.source_image, "source_folder": src_folder,
                         "mouse": r.mouse, "view": r.view, "bodypart": bp,
                         "state": "both" if ao and bo else "original_only" if ao else "relabel_only" if bo else "neither",
                         "dist_px": d, "x_orig": a.loc[bp, "x"], "y_orig": a.loc[bp, "y"],
                         "x_relabel": b.loc[bp, "x"], "y_relabel": b.loc[bp, "y"]})
    df = pd.DataFrame(rows)
    df.to_csv(SIDE / "human_variability_px.csv", index=False)
    lab = df[df.state != "neither"]
    summ = lab.groupby(["view", "bodypart"]).agg(
        n_both=("state", lambda s: (s == "both").sum()), only_orig=("state", lambda s: (s == "original_only").sum()),
        only_relabel=("state", lambda s: (s == "relabel_only").sum()), median_px=("dist_px", "median"),
        mean_px=("dist_px", "mean"), p90_px=("dist_px", lambda v: v.quantile(0.9)))
    summ = summ[summ.n_both + summ.only_orig + summ.only_relabel > 0].round(2)
    print(summ.to_string())
    for v in ("side", "top"):
        d = df[(df.view == v) & (df.state == "both")].dist_px
        print(v, "all keypoints: median", round(d.median(), 2), "mean", round(d.mean(), 2), "RMSE",
              round(float(np.sqrt((d ** 2).mean())), 2), "n", len(d))


if __name__ == "__main__":
    main()
