"""Onsets for the lick-aligned whisker-vs-auditory decoder (modality_lick, per-session 024 area-group decoders, rerun
2026-11) -- helper of explore_modality_lick.py, run in the hit/miss project's env (~/code/ibl-ai-agent/.venv; 042 imports
brainwidemap). Two rules, both 042 cluster_latency (cluster-mass, mouse-block sign-flip null, 2000 permutations,
p < 0.05, linear-shift null curves):
  raw       real - shift null, as the project reports it (search over the whole window, -600..+200 ms);
  baseline  each session's real curve minus its mean (real - shift null) over the first 200 ms of the window
            (bins ending -600..-400 ms) -> deviation from the session's own early level.
Usage: python explore_modality_lick_onsets.py <rows_dir> <out_json> <area> [<area> ...]
"""
import glob
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HM = pathlib.Path(__file__).resolve().parents[3] / "ssl-whisker-hitmiss-timeresolved-decoding" / "exploratory-analyses"
sys.path.insert(0, str(HM))
m042 = importlib.import_module("042_area_decoding_latency")

rows_dir, out_json, areas = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
rows = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(rows_dir / "*.parquet")))], ignore_index=True)
rows = rows[(rows.condition_type == "whole") & rows.real_curve.notna()].copy()
rows["surrogate_curve_used"] = [m042.session_surrogate_curve(r)[0] for r in rows.itertuples()]
edges = json.load(open(next(rows_dir.glob("024_bin_edges_*.json"))))
t = np.array([e[1] for e in edges])
base = (t >= -0.6 - 1e-9) & (t <= -0.4 + 1e-9)
bl = rows.copy()
bl["real_curve"] = [np.asarray(r, float) - np.nanmean((np.asarray(r, float) - np.asarray(s, float))[base])
                    for r, s in zip(rows.real_curve, rows.surrogate_curve_used)]
out = {}
for mode, R in (("raw", rows), ("baseline", bl)):
    rng = np.random.default_rng(0)
    for a in areas:
        sub = R[R.area_value == a]
        res = m042.cluster_latency(sub, t * 1000, rng) if len(sub) else dict(latency_ms=np.nan)
        out[f"{mode}|{a}"] = None if not np.isfinite(res["latency_ms"]) else float(res["latency_ms"]) / 1000
        print(mode, a, out[f"{mode}|{a}"], res.get("n_mice"), res.get("reason"), flush=True)
json.dump(out, open(out_json, "w"), indent=1)
