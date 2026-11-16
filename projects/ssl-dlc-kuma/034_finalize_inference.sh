#!/bin/bash
# After the full re-inference (033 arrays + the MH007 side job): per video, check the job JSON (ok, frames video vs h5),
# write $ROOT/summary.tsv, and copy the h5 files (raw + filtered) and JSONs to the NAS
# (DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330/<session>/). The data folders are not touched.
# Submitted with --dependency=afterany on the inference jobs (runs even if some failed; failures listed).
set -uo pipefail
module load rclone/1.65.2
ROOT=/work/lsens/bisi/dlc_merged/inference_cspnext_m_330
DST=nas-sv:Petersen-Lab/analysis/Axel_Bisi/DLCProjects/merged_training_2026-11-09/inference_cspnext_m_330
source /work/lsens/bisi/dlc_project/venv_dlc/bin/activate
python - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
man = pd.concat([pd.read_csv(f, sep="\t").assign(tier=f.stem.split("_")[1]) for f in (root / "manifests").glob("manifest_*.tsv")])
rows = []
for r in man.itertuples():
    j = root / r.session_id / (Path(r.video).stem + "_job.json")
    d = json.loads(j.read_text()) if j.exists() else {}
    rows.append(dict(session_id=r.session_id, view=r.view, tier=r.tier, task=r.task, ok=d.get("ok", False),
                     n_frames_video=d.get("n_frames_video"), n_frames_h5=d.get("n_frames_h5"), source=d.get("source", "missing"),
                     frames_per_s=d.get("frames_per_s")))
s = pd.DataFrame(rows)
s.to_csv(root / "summary.tsv", sep="\t", index=False)
print(f"{s.ok.sum()} / {len(s)} videos ok; median {s.frames_per_s.median():.0f} fr/s")
print(s[~s.ok].to_string())
PY
rclone copy "$ROOT" "$DST" --include "*.h5" --include "*.json" --include "*.tsv" --log-level NOTICE
rclone ls "$DST" | wc -l
