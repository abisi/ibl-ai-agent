"""Extract frames for labelling into the merged project Tracking2D-Axel-2026-11-04 (2026-11-04, user). Run on haas.

Two modes; frames are written as DLC does (labeled-data/<video stem>/img<frame index zero-padded to
ceil(log10(n_frames)) digits>.png, 3-channel 640x480), existing files are never overwritten, and every extracted
frame is listed in labeled-data/extraction_manifest_<mode>.csv (folder, image, frame, reason).

  kmeans   videos never in the dataset: for every mouse x view with video but no labelled frame
           (combined_results_ks4/ssl-dlc-kuma/label_coverage.csv), one session (whisker day 0 if available, else the
           first whisker day, else the first session with video); N_KMEANS frames = DLC-style k-means: N_SAMPLE frames
           sampled uniformly over the video, downscaled to 30 px width (grey), k-means with N_KMEANS clusters, the frame
           closest to each centroid
  tongue   surplus tongue frames for every video already in the dataset (labeled-data folders with frames): frames
           LICK_WIN after piezo lick times (NWB piezo_lick_times; video frame = NWB camera-timestamp index, NWB series
           of that view), ranked by the existing DLC tongue likelihood when the h5 is found, top N_TONGUE_POOL kept,
           N_TONGUE frames chosen by k-means for diversity; fallback without NWB / licks: frames with DLC tongue
           likelihood > 0.5
Usage: ~/code/ibl-ai-agent/.venv/bin/python 003_extract_frames.py {kmeans,tongue} --jobs 12
"""

import argparse
import glob
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

PROJ = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/Tracking2D-Axel-2026-11-04")
LAB = PROJ / "labeled-data"
KUMA = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma")
NWB = Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
DLC_DIRS = {"AB": Path("/mnt/lsens-analysis/Axel_Bisi/data"), "MH": Path("/mnt/lsens-analysis/Myriam_Hamon/data")}
N_SAMPLE, N_KMEANS = 1000, 20
N_TONGUE, N_TONGUE_POOL, LICK_WIN = 5, 300, (0.015, 0.045)
TS_SERIES = {"side": "jaw_x", "top": "top_nose_tip_x"}


def n_frames(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def save_frames(video, folder, idx, reason):
    """Write the given frame indices of the video as DLC-named pngs; returns manifest rows."""
    out = LAB / folder
    out.mkdir(exist_ok=True)
    width = int(np.ceil(np.log10(n_frames(video))))
    cap = cv2.VideoCapture(video)
    rows = []
    for f in sorted(set(int(i) for i in idx)):
        name = f"img{str(f).zfill(width)}.png"
        if (out / name).exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            cv2.imwrite(str(out / name), img)
            rows.append(dict(folder=folder, image=name, frame=f, reason=reason))
    cap.release()
    return rows


def thumbs(video, idx):
    cap = cv2.VideoCapture(video)
    X, keep = [], []
    for f in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            X.append(cv2.resize(g, (30, int(30 * g.shape[0] / g.shape[1])), interpolation=cv2.INTER_AREA).ravel())
            keep.append(int(f))
    cap.release()
    return np.array(X, float), np.array(keep)


def kmeans_pick(X, frames, k, seed=0):
    k = min(k, len(frames))
    km = KMeans(k, n_init=4, random_state=seed).fit(X)
    pick = []
    for c in range(k):
        m = np.flatnonzero(km.labels_ == c)
        pick.append(frames[m[np.argmin(((X[m] - km.cluster_centers_[c]) ** 2).sum(1))]])
    return pick


def job_kmeans(args):
    video, folder = args
    try:
        n = n_frames(video)
        X, fr = thumbs(video, np.linspace(0, n - 1, N_SAMPLE).astype(int))
        return save_frames(video, folder, kmeans_pick(X, fr, N_KMEANS), "kmeans_never_labelled")
    except Exception as e:
        return [dict(folder=folder, error=f"{type(e).__name__}: {e}")]


def dlc_h5(folder):
    mouse = folder[:5]
    hits = [h for h in glob.glob(str(DLC_DIRS.get(mouse[:2], Path("/x")) / mouse / "*" / "Video" /
                                     (glob.escape(folder) + "*filtered.h5"))) if "_backup_" not in h]
    return hits[0] if hits else None


def job_tongue(args):
    video, folder, session = args
    try:
        n = n_frames(video)
        view = "top" if "_top" in folder else "side"
        lik = None
        h5 = dlc_h5(folder)
        if h5:
            d = pd.read_hdf(h5)
            d.columns = d.columns.droplevel(0)
            if ("tongue", "likelihood") in d.columns:
                lik = d[("tongue", "likelihood")].to_numpy()
        cand, reason = np.array([], int), ""
        nwbp = NWB / f"{session}.nwb" if isinstance(session, str) else None
        if nwbp is not None and nwbp.exists():
            with h5py.File(nwbp, "r") as f:
                ev = f.get("processing/behavior/BehavioralEvents/piezo_lick_times")
                ts = f.get(f"processing/behavior/BehavioralTimeSeries/{TS_SERIES[view]}")
                if ev is not None and ts is not None:
                    licks, t = ev["timestamps"][:], ts["timestamps"][:]
                    if abs(len(t) - n) <= 5:
                        idx = np.concatenate([np.arange(*np.searchsorted(t, [l + LICK_WIN[0], l + LICK_WIN[1]]))
                                              for l in licks]) if len(licks) else np.array([], int)
                        cand, reason = np.unique(idx[(idx >= 0) & (idx < n)]), "tongue_after_piezo_lick"
        if not len(cand) and lik is not None:
            cand, reason = np.flatnonzero(lik > 0.5), "tongue_dlc_likelihood"
        if not len(cand):
            return [dict(folder=folder, error="no lick times and no DLC tongue detections")]
        if lik is not None and len(lik) >= cand.max() + 1:
            cand = cand[np.argsort(-lik[cand])][:N_TONGUE_POOL]
        else:
            cand = np.random.default_rng(0).choice(cand, min(N_TONGUE_POOL, len(cand)), replace=False)
        X, fr = thumbs(video, np.sort(cand))
        return save_frames(video, folder, kmeans_pick(X, fr, N_TONGUE), reason)
    except Exception as e:
        return [dict(folder=folder, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-300:])]


def sessions_by_video():
    inv = pd.read_csv(KUMA / "inventory.csv")
    m = {}
    for _, r in inv.dropna(subset=["video_files"]).iterrows():
        for v in str(r.video_files).split(";"):
            m[os.path.splitext(v.strip())[0]] = r.session_id
    return m, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["kmeans", "tongue"])
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    vid2sess, inv = sessions_by_video()
    if a.mode == "kmeans":
        cov = pd.read_csv(KUMA / "label_coverage.csv")
        todo = cov[cov.labelled == 0]
        days = {}
        for p in glob.glob("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/*/whisker_*/learning_curve/*_whisker_trial_learning_curve_exact.h5"):
            days[pd.read_hdf(p).session_id.iloc[0]] = int(Path(p).parts[-3].split("_")[1])
        jobs = []
        for _, r in todo.iterrows():
            c = inv[(inv.mouse_id == r.mouse) & (inv.view == r.view) & (inv.n_videos > 0)].copy()
            c["day"] = c.session_id.map(days)
            c["rank"] = np.where(c.day == 0, 0, np.where(c.day.notna(), 1, 2))
            c = c.sort_values(["rank", "session_id"])
            if c.empty:
                continue
            row = c.iloc[0]
            v = os.path.join(row.raw_folder, str(row.video_files).split(";")[0].strip())
            jobs.append((v, os.path.splitext(os.path.basename(v))[0]))
        fn = job_kmeans
    else:
        folders = [p for p in LAB.iterdir() if p.is_dir() and any(p.glob("*.png"))]
        cfg = pd.read_csv(PROJ / "merged_project_folders.csv")
        vids = dict(zip(cfg.folder, cfg.video))
        jobs = [(vids[p.name], p.name, vid2sess.get(p.name)) for p in folders if isinstance(vids.get(p.name), str)]
        fn = job_tongue
    print(f"{a.mode}: {len(jobs)} videos", flush=True)
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for rs in ex.map(fn, jobs):
            rows.extend(rs)
            print(rs[0].get("folder"), len([r for r in rs if "image" in r]), rs[0].get("error", ""), flush=True)
    m = pd.DataFrame(rows)
    out = LAB / f"extraction_manifest_{a.mode}.csv"
    if out.exists():
        m = pd.concat([pd.read_csv(out), m], ignore_index=True)
    m.to_csv(out, index=False)
    ok = m[m.get("image").notna()] if "image" in m else m.iloc[:0]
    print(f"DONE {a.mode}: {len(ok)} frames in {ok.folder.nunique()} folders; errors: "
          f"{m.error.notna().sum() if 'error' in m else 0}")
    if a.mode == "kmeans":
        print("new videos:", sorted(set(v for v, _ in jobs)))


if __name__ == "__main__":
    main()
