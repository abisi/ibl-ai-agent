"""mm per pixel for every labeled-data folder of the merged DLC project (training_plan.md: errors in px and mm), from
the NWB DLC series `conversion` attribute (mm per pixel of each camera) read with ephys_utilities dlc_helpers
(TS_PATH, camera grouping). Folder -> NWB: same mouse and recording date (folder name 'MOUSE_view yy-mm-dd ...');
folders without a date (e.g. 'AB082_side-converted') or without an NWB of that date take the mouse's median over its
NWBs; mice without any NWB DLC take the median of the view. NWB sources: NWB_ks4, then NWB_combined, NWBFull.

Output: DLCProjects/merged_training_2026-11-09/mm_per_px.csv (folder, mouse, view, date, mm_per_px, source)
and human_variability_mm.csv (human_variability_px.csv + mm columns).
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 016_mm_per_px.py
"""

import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.home() / "code/ephys_utilities"))
from ephys_utilities.helpers import dlc_helpers  # noqa: E402

AB = Path("/mnt/lsens-analysis/Axel_Bisi")
PROJ = AB / "DLCProjects/Tracking2D-Axel-2026-11-04"
SIDE = AB / "DLCProjects/merged_training_2026-11-09"
NWB_DIRS = [AB / "NWB_ks4", AB / "NWB_combined", AB / "NWBFull"]
PAT = re.compile(r"^([A-Z]+\d+)_(side|top)(?: (\d\d)-(\d\d)-(\d\d))?")


def nwb_conversion(nwb):
    """{view: mm per pixel} of one NWB (first non-likelihood position series of each camera)."""
    out = {}
    with h5py.File(nwb, "r") as f:
        if dlc_helpers.TS_PATH not in f:
            return out
        grp = f[dlc_helpers.TS_PATH]
        for view, names in dlc_helpers._camera_groups(grp).items():
            for n in sorted(names):
                if not n.endswith("likelihood"):
                    c = grp[n]["data"].attrs.get("conversion")
                    if c is not None and float(c) not in (0.0, 1.0):
                        out[view] = float(c)
                        break
    return out


def main():
    folders = sorted(p.name for p in (PROJ / "labeled-data").iterdir() if p.is_dir() and not p.name.endswith("_labeled"))
    rows, cache = [], {}
    for name in folders:
        m = PAT.match(name)
        if not m:
            continue
        mouse, view = m[1], m[2]
        date = f"20{m[3]}{m[4]}{m[5]}" if m[3] else None
        if mouse not in cache:
            files = {}
            for d in NWB_DIRS:
                for f in d.glob(f"{mouse}_*.nwb"):
                    if dlc_helpers is not None and "backup" not in f.name:
                        files.setdefault(f.name.split("_")[1], f)
            cache[mouse] = {day: nwb_conversion(f) for day, f in files.items()}
        conv = cache[mouse]
        if date and view in conv.get(date, {}):
            rows.append({"folder": name, "mouse": mouse, "view": view, "date": date,
                         "mm_per_px": conv[date][view], "source": "nwb same date"})
        else:
            vals = [c[view] for c in conv.values() if view in c]
            rows.append({"folder": name, "mouse": mouse, "view": view, "date": date,
                         "mm_per_px": float(np.median(vals)) if vals else np.nan,
                         "source": "mouse median" if vals else "view median"})
    df = pd.DataFrame(rows)
    for v in ("side", "top"):
        miss = (df.view == v) & df.mm_per_px.isna()
        df.loc[miss, "mm_per_px"] = df[(df.view == v) & ~miss].mm_per_px.median()
    df.to_csv(SIDE / "mm_per_px.csv", index=False)
    print(df.groupby(["view", "source"]).mm_per_px.agg(["count", "median", "min", "max"]).round(4).to_string())
    hv = pd.read_csv(SIDE / "human_variability_px.csv")
    hv = hv.merge(df[["folder", "mm_per_px"]], left_on="source_folder", right_on="folder", how="left").drop(columns="folder")
    hv["dist_mm"] = hv.dist_px * hv.mm_per_px
    hv.to_csv(SIDE / "human_variability_mm.csv", index=False)
    s = hv[hv.state == "both"].groupby(["view", "bodypart"]).agg(n=("dist_px", "size"), median_px=("dist_px", "median"),
                                                                 median_mm=("dist_mm", "median"), max_px=("dist_px", "max"))
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
