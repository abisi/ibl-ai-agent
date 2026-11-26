"""Hit vs miss significance onsets per area group for Figure 3 C (called by compute.py _sig_onsets / _sig_onsets_baseline in
the hit/miss project's environment, ~/code/ibl-ai-agent/.venv, because 042 imports brainwidemap / ssl_timeresolved_decoding).
Rule = ssl-whisker-hitmiss-timeresolved-decoding 042 cluster_latency: earliest cluster-mass-significant cluster (p < 0.05,
mouse-block sign-flip null, 2000 permutations, linear-shift null curves), rerun rows (hitmiss_stim_area_group, 'whole').
--baseline (user 2026-11-26): each session's accuracy - null is first referenced to its own pre-stimulus level (mean over
bins ending in [-200, -10] ms: the real curve is lowered by that mean), and the test is run on the bins ending after 0
only (search from the stimulus).
Usage: python hm_onsets.py [--baseline] <rows_dir> <out_json> <area> [<area> ...]
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

args = sys.argv[1:]
baseline = "--baseline" in args
args = [a for a in args if a != "--baseline"]
rows_dir, out_json, areas = pathlib.Path(args[0]), args[1], args[2:]
rows = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(rows_dir / "*.parquet")))], ignore_index=True)
rows = rows[(rows.condition_type == "whole") & rows.real_curve.notna()].copy()
rows["surrogate_curve_used"] = [m042.session_surrogate_curve(r)[0] for r in rows.itertuples()]
edges = json.load(open(next(rows_dir.glob("024_bin_edges_*.json"))))
t_s = np.round(np.array([e[1] for e in edges]), 3)
t_ms = t_s * 1000
if baseline:
    b = (t_s >= -0.200 - 1e-9) & (t_s <= -0.010 + 1e-9)
    post = t_s > 1e-9
    real, surr = [], []
    for r in rows.itertuples():
        x, s = np.asarray(r.real_curve, float), np.asarray(r.surrogate_curve_used, float)
        lev = np.nanmean((x - s)[b])
        real.append((x - lev)[post]); surr.append(s[post])
    rows["real_curve"], rows["surrogate_curve_used"] = real, surr
    t_ms = t_ms[post]
rng = np.random.default_rng(0)
out = {}
for a in areas:
    sub = rows[rows.area_value == a]
    res = m042.cluster_latency(sub, t_ms, rng) if len(sub) else dict(latency_ms=np.nan)
    out[a] = None if not np.isfinite(res["latency_ms"]) else float(res["latency_ms"]) / 1000
    print(a, out[a], res.get("n_mice"), res.get("reason"), flush=True)
json.dump(out, open(out_json, "w"), indent=1)
