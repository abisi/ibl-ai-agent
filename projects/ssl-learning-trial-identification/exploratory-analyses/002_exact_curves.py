"""002 -- Exact (grid forward-backward) learning curves for all sessions in
artifacts/001_inputs.pkl, under the stored model's prior ('orig', sigma ~ 1)
and a broad prior letting the data choose smoothness ('eb'), for whisker and
no-stim (FA) trials. See `lt_lib.py`.

Per session and prior:
  whisker p_mean, 80%/90% CI; FA curve at whisker times two ways:
    fa_evengrid : stored pipeline's `interp_fa_curve` (FA p_mean placed on an
                  evenly spaced time grid, cubic spline) -- reproduces the
                  stored p_chance
    fa_time     : FA posterior placed at the actual no-stim trial times
                  (marginals mixed linearly in time)
  p_above = P(p_whisker > p_FA) per whisker trial (FA at actual times,
  independent posteriors).
  sigma posterior mean, log evidence (-> log Bayes factor eb vs orig).
Also: |fa_evengrid - fa_time| (size of the interpolation-grid issue) and
agreement of the orig-prior exact curve with the stored (40-sample) curve.

Output: artifacts/002_curves.pkl (session_id -> dict, marginals float32),
        artifacts/002_summary.csv
"""

from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lt_lib as L  # noqa: E402

ART = Path(__file__).resolve().parents[1] / "artifacts"


def process(item):
    sid, d = item
    out = {"meta": {k: d[k] for k in ("session_id", "mouse_id", "reward_group", "learning_category")}}
    ws, ns = d["w_start"], d["n_start"]
    for prior in ("orig", "eb"):
        w = L.fit_curve(d["w_outcomes"], prior)
        n = L.fit_curve(d["n_outcomes"], prior)
        grid = np.linspace(min(ws.min(), ns.min()), max(d["w_stop"].max(), ns.max()), len(n["p_mean"]))
        fa_even = CubicSpline(grid, n["p_mean"], extrapolate=False)(ws)
        fa_marg = L.interp_marginals(n["marg"], ns, ws)
        lo80, hi80, lo90, hi90 = L.quantiles(w["marg"], [0.1, 0.9, 0.05, 0.95])
        out[prior] = dict(p_mean=w["p_mean"], p_low80=lo80, p_high80=hi80, p_low90=lo90, p_high90=hi90,
                          fa_evengrid=fa_even, fa_time=fa_marg @ L.P_GRID, p_above=L.prob_greater(w["marg"], fa_marg),
                          sigma=w["sigma_post_mean"], sigma_fa=n["sigma_post_mean"], logev=w["log_evidence"],
                          logev_fa=n["log_evidence"], marg_w=w["marg"].astype(np.float32), marg_fa=fa_marg.astype(np.float32))
    return sid, out


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = dict(ex.map(process, inputs.items()))
    pickle.dump(res, open(ART / "002_curves.pkl", "wb"))
    rows = []
    for sid, r in res.items():
        st = inputs[sid]["stored"]
        o, e = r["orig"], r["eb"]
        rows.append(dict(session_id=sid, reward_group=r["meta"]["reward_group"], n_trials=len(o["p_mean"]),
                         sigma_orig=o["sigma"], sigma_eb=e["sigma"], sigma_fa_eb=e["sigma_fa"],
                         log_bf_eb_vs_orig=e["logev"] - o["logev"],
                         stored_vs_exact_orig_mae=np.abs(st["p_mean"] - o["p_mean"]).mean(),
                         stored_pchance_vs_fa_even_mae=np.nanmean(np.abs(st["p_chance"] - o["fa_evengrid"])),
                         fa_even_vs_time_mae=np.nanmean(np.abs(o["fa_evengrid"] - o["fa_time"])),
                         fa_even_vs_time_max=np.nanmax(np.abs(o["fa_evengrid"] - o["fa_time"])),
                         roughness_stored=np.median(np.abs(np.diff(np.log(st["p_mean"] / (1 - st["p_mean"]))))),
                         roughness_eb=np.median(np.abs(np.diff(np.log(e["p_mean"] / (1 - e["p_mean"])))))))
    df = pd.DataFrame(rows)
    df.to_csv(ART / "002_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(df.drop(columns=["session_id"]).groupby("reward_group").median().T.round(3).to_string())
    print("sessions with log BF(eb vs orig) > 3:", (df.log_bf_eb_vs_orig > 3).sum(), "of", len(df),
          "| < -3:", (df.log_bf_eb_vs_orig < -3).sum())


if __name__ == "__main__":
    main()
