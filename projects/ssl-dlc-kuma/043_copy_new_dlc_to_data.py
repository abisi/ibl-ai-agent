"""Copy the new DLC files (CSPNeXt-M snapshot-330, filtered) into the data folders next to the old ones, as
*_filtered_new.h5 (user 2026-11-10: "copy back in tmp before copying to /data of my analysis folder, but with _new";
approved 2026-11-11 after the assessment, 036 / 042).

Source: DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330/<session>/*_filtered.h5 (034 staging).
Destination: the converter's DLC folder (NWB_converter utils/server_paths.get_dlc_file_path): AB mice
Axel_Bisi/data/<mouse>/<session>/Video, MH mice Myriam_Hamon/data/<mouse>/<session>/Video (created if missing: sessions
without any old DLC). Name: <video stem>DLC_CspnextM_Tracking2DNov4shuffle6_snapshot_330_filtered_new.h5. Never
overwrites: an existing destination with the same size is skipped, a different size stops the copy (reported).
The converter picks *_filtered_new.h5 per camera when present (server_paths.get_dlc_file_path, 2026-11-11).
Output: DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330/copy_to_data_log.csv
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 043_copy_new_dlc_to_data.py [--dry-run]
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path("/mnt/lsens-analysis")
STAGE = ROOT / "Axel_Bisi/DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330"
DATA = {"AB": ROOT / "Axel_Bisi/data", "MH": ROOT / "Myriam_Hamon/data"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    rows = []
    for src in sorted(STAGE.glob("*/*_filtered.h5")):
        session = src.parent.name
        dest_dir = DATA[session[:2]] / session.split("_")[0] / session / "Video"
        dest = dest_dir / src.name.replace("_filtered.h5", "_filtered_new.h5")
        view = "side" if "_side" in src.name else "top"
        old = [p.name for p in dest_dir.glob("*filtered.h5") if view in p.name] if dest_dir.exists() else []
        r = dict(session_id=session, view=view, source=str(src), dest=str(dest), size=src.stat().st_size,
                 old_files=" | ".join(old), folder_created=not dest_dir.exists())
        if dest.exists():
            r["status"] = "exists, same size" if dest.stat().st_size == r["size"] else "EXISTS, DIFFERENT SIZE"
        elif args.dry_run:
            r["status"] = "to copy"
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(dest.name + ".part")
            shutil.copyfile(src, tmp)  # NAS refuses utime: copyfile, not copy2
            tmp.rename(dest)
            r["status"] = "copied"
        rows.append(r)
    log = pd.DataFrame(rows)
    if not args.dry_run:
        log.to_csv(STAGE / "copy_to_data_log.csv", index=False)
    print(log.groupby(["status", "folder_created"]).size().to_string())
    print(len(log), "files,", log.session_id.nunique(), "sessions,", round(log["size"].sum() / 1e9, 1), "GB")
    print("without an old file of that camera:", (log.old_files == "").sum())
    bad = log[log.status.str.startswith("EXISTS")]
    if len(bad):
        print(bad.to_string())


if __name__ == "__main__":
    main()
