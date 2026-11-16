"""Labelling status of every SSL whisker session x view in the merged project Tracking2D-Axel-2026-11-04 (2026-11-04).

Whisker sessions = sessions with an exact whisker learning curve (combined_results_ks4/<mouse>/whisker_<day>/...).
For each session x view, the video stems come from ssl-dlc-kuma/inventory.csv and are looked up in
labeled-data/<stem>:
  labelled           CollectedData_*.h5 with >= 1 frame carrying a label
  extracted_old      frames (png) but no labelled frame, folder already in the project when it was built
  extracted_kmeans   frames written by today's k-means extraction (folder not in merged_project_folders.csv)
  not_extracted      video exists, no labeled-data folder / no frames
  no_video           no video in the inventory
Writes <project>/label_status_whisker_sessions.csv and prints counts.
Usage (haas): ~/code/ibl-ai-agent/.venv/bin/python 006_label_status.py
"""

import glob
import os
from pathlib import Path

import pandas as pd

PROJ = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/Tracking2D-Axel-2026-11-04")
LAB = PROJ / "labeled-data"
KUMA = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma")


def whisker_sessions():
    rows = []
    for p in glob.glob("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/*/whisker_*/learning_curve/*_whisker_trial_learning_curve_exact.h5"):
        sid = pd.read_hdf(p).session_id.iloc[0]
        rows.append(dict(session_id=sid, mouse_id=sid.split("_")[0], day=int(Path(p).parts[-3].split("_")[1])))
    return pd.DataFrame(rows).drop_duplicates("session_id")


def folder_status(stem, built):
    d = LAB / stem
    if not d.is_dir() and stem.startswith("M0"):  # MH006 videos renamed from M006_ after the inventory was built
        d = LAB / ("MH" + stem[1:])
    if not d.is_dir():
        return "not_extracted", 0, 0
    n_png = len(list(d.glob("*.png")))
    n_lab = 0
    for h in d.glob("CollectedData_*.h5"):
        n_lab += int(pd.read_hdf(h).notna().any(axis=1).sum())
    if n_lab:
        return "labelled", n_png, n_lab
    if n_png == 0:
        return "not_extracted", 0, 0
    return ("extracted_old" if stem in built else "extracted_kmeans"), n_png, 0


def main():
    ws = whisker_sessions()
    inv = pd.read_csv(KUMA / "inventory.csv")
    built = set(pd.read_csv(PROJ / "merged_project_folders.csv").folder)
    rank = ["labelled", "extracted_old", "extracted_kmeans", "not_extracted", "no_video"]
    rows = []
    for s in ws.itertuples():
        for view in ("side", "top"):
            r = inv[(inv.session_id == s.session_id) & (inv.view == view)]
            vids = [] if r.empty or pd.isna(r.video_files.iloc[0]) else [v.strip() for v in str(r.video_files.iloc[0]).split(";")]
            if not vids:
                rows.append(dict(mouse_id=s.mouse_id, session_id=s.session_id, day=s.day, view=view, status="no_video",
                                 n_png=0, n_labelled=0, folders=""))
                continue
            st = [(os.path.splitext(v)[0], *folder_status(os.path.splitext(v)[0], built)) for v in vids]
            best = min(st, key=lambda x: rank.index(x[1]))
            rows.append(dict(mouse_id=s.mouse_id, session_id=s.session_id, day=s.day, view=view, status=best[1],
                             n_png=sum(x[2] for x in st), n_labelled=sum(x[3] for x in st),
                             folders=";".join(x[0] for x in st)))
    df = pd.DataFrame(rows).sort_values(["mouse_id", "session_id", "view"])
    df.to_csv(PROJ / "label_status_whisker_sessions.csv", index=False)
    print(f"{df.session_id.nunique()} whisker sessions, {df.mouse_id.nunique()} mice")
    print(pd.crosstab(df.status, df.view, margins=True).reindex(rank + ["All"]).fillna(0).astype(int))
    print("\nlabelled frames per view:", df.groupby("view").n_labelled.sum().to_dict())
    mv = df.groupby(["mouse_id", "view"]).status.agg(lambda x: "labelled" if (x == "labelled").any() else min(x, key=rank.index))
    print("\nmouse x view without any labelled session (best status):")
    print(mv[mv != "labelled"].unstack().fillna("-").to_string())


if __name__ == "__main__":
    main()
