"""Manifest of the full-dataset re-inference with the merged model (CSPNeXt-M snapshot-330; user 2026-11-10).

One row per camera-session of the DLC inventory (combined_results_ks4/ssl-dlc-kuma/inventory.csv, n_videos > 0) with
its video on Kuma (/work/lsens/bisi/data/<mouse>/<session>/Video/), matched by session folder and camera name
('_side ' / '_side_' / '_side-', same for top). Several files for one camera: the largest (complete) one. Exceptions:
MH019_20250306_121844 has no video of its own (its raw folder held misplaced copies of MH020 0306, which are run under
MH020); MH007_20250203 side runs separately (copied 2026-11-10, job 4884034).
Tier by frame count (inventory video_n_frames; else size / frame size): time limit = frames / 200 fps x 1.25 + 15 min,
rounded up to the tier: S <= 2 h, M <= 3 h, L <= 4.5 h, XL beyond (6 h). Rows sorted longest first within a tier.
Videos already run with the same model and snapshot (023 pilot_inference, 026 pilot_models/s6) are marked reuse=1:
the job copies their h5 instead of running DLC.
Input: /tmp/kuma_videos_sizes.tsv (size \\t data-relative path, from `find ... -printf`). Output: manifest_<tier>.tsv
(task, session_id, view, video, n_frames, reuse_dir) in combined_results_ks4/ssl-dlc-kuma/inference_cspnext_m_330/.
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 032_inference_manifest.py
"""

import math
import re
from pathlib import Path

import pandas as pd

INV = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma/inventory.csv")
OUT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma/inference_cspnext_m_330")
DATA = "/work/lsens/bisi/data"
MERGED = "/work/lsens/bisi/dlc_merged"
PILOT = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/merged_training_2026-11-09/kuma")
SKIP = {("MH019_20250306_121844", "side"), ("MH019_20250306_121844", "top"), ("MH007_20250203_140412", "side")}
TIERS = [("S", 2.0), ("M", 3.0), ("L", 4.5), ("XL", 6.0)]
FRAME_BYTES = 640 * 480  # uncompressed 8-bit grey AVI


def frames_of(row, size):
    s = str(row.video_n_frames)
    n = sum(int(i) for i in re.findall(r"\d+", s)) if s not in ("nan", "") else 0
    return n if n > 0 else int(size / FRAME_BYTES)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    k = pd.read_csv("/tmp/kuma_videos_sizes.tsv", sep="\t", names=["size", "path"])
    k["session_id"] = k.path.str.split("/").str[1]
    inv = pd.read_csv(INV)
    inv = inv[inv.n_videos > 0]
    reuse = {}
    for d in [PILOT / "pilot_inference", PILOT / "pilot_models" / "s6"]:
        for h5 in d.glob("*/*_filtered.h5"):
            v = "side" if "_side" in h5.name else "top"
            reuse.setdefault((h5.parent.name, v), f"{MERGED}/{d.relative_to(PILOT)}/{h5.parent.name}")
    rows, missing = [], []
    for r in inv.itertuples():
        if (r.session_id, r.view) in SKIP:
            continue
        cand = k[(k.session_id == r.session_id)
                 & k.path.str.contains(rf"_{r.view}[ _\-]", regex=True)]
        if cand.empty:
            missing.append((r.session_id, r.view))
            continue
        best = cand.sort_values("size").iloc[-1]
        n = frames_of(r, best["size"])
        hours = n / 200 / 3600 * 1.25 + 0.25
        tier = next(t for t, lim in TIERS if hours <= lim or t == "XL")
        rows.append(dict(session_id=r.session_id, view=r.view, video=f"{DATA}/{best.path}", n_frames=n, tier=tier,
                         reuse_dir=reuse.get((r.session_id, r.view), "")))
    m = pd.DataFrame(rows)
    for t, _ in TIERS:
        mt = m[m.tier == t].sort_values("n_frames", ascending=False).reset_index(drop=True)
        mt.insert(0, "task", range(len(mt)))
        mt.drop(columns="tier").to_csv(OUT / f"manifest_{t}.tsv", sep="\t", index=False)
    print(m.groupby("tier").agg(videos=("video", "size"), frames_M=("n_frames", lambda x: x.sum() / 1e6),
                                reuse=("reuse_dir", lambda x: (x != "").sum())).round(1).to_string())
    print("total", len(m), "videos,", round(m.n_frames.sum() / 1e6, 1), "M frames,",
          round(m[m.reuse_dir == ""].n_frames.sum() / 222 / 3600, 0), "GPU-h to run (222 fr/s)")
    print("no Kuma video:", missing)


if __name__ == "__main__":
    main()
