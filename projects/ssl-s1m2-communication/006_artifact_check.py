"""Artefact check (user 2026-11-19): does the whisker-stimulus artefact end by +5 ms after trial start?

Raw spikes (no Poisson fill, no blanking), all stable good + mua units of the sessions in sessions_v2.csv, 1-ms bins
from -30 to +40 ms around trial start, active trials (s1m2_lib trial rules). Per session, trial type (whisker, auditory,
no-stim) and area set (S1 = SSp-bfd, M2 = MO-wM1 + MO-wM2, other = every other recorded area):
  - population rate (spikes / s / unit), and the same divided by the -30..-12 ms mean;
  - S1-M2 zero-lag trial-to-trial correlation of the summed population counts in each 1-ms bin (an artefact is
    synchronous across areas, so it shows up as a correlation peak on whisker trials only).
An artefact predicts a whisker-only, all-area (including non-responsive "other" areas) bump in rate and in S1-M2
synchrony; a sensory response predicts a later, area-specific rise (S1 first).

Run on haas: python 006_artifact_check.py --n-jobs 14  ->  OUT/artifact_check/{per_session.parquet, artifact_check.png}
"""
import argparse
import importlib
import json
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import s1m2_lib as L

OUTD = L.OUT / "artifact_check"
EDGES = np.round(np.arange(-0.030, 0.0405, 0.001), 4)
TYPES = {"whisker": "whisker_trial", "auditory": "auditory_trial", "no-stim": "no_stim_trial"}


def run(sid):
    su = importlib.import_module("002_sessions").stable_units()
    keys = su[su.session_id == sid][["electrode_group", "cluster_id", "area_acronym_custom", "target_region",
                                     "quality_label", "firing_rate"]]
    units, trials = L.load_session(sid, keys, blank=False)
    grp = np.where(units.area_acronym_custom == "SSp-bfd", "S1",
                   np.where(units.area_acronym_custom.isin(["MO-wM1", "MO-wM2"]), "M2", "other"))
    rows = []
    for tname, tt in TYPES.items():
        st = trials[trials.trial_type == tt].start_time.to_numpy(float)
        if len(st) < 10:
            continue
        X = L.count_tensor(units.spike_times.tolist(), st, EDGES)  # trial, bin, unit
        pops = {g: X[:, :, grp == g].sum(2) for g in ("S1", "M2", "other") if (grp == g).any()}
        for g, P in pops.items():
            n_u = int((grp == g).sum())
            rate = P.mean(0) / n_u / 0.001
            base = rate[(EDGES[:-1] >= -0.030) & (EDGES[:-1] < -0.012)].mean()
            rows.append(dict(session_id=sid, trial_type=tname, area=g, n_units=n_u, n_trials=len(st),
                             rate=rate.tolist(), rate_norm=(rate / base if base > 0 else rate * np.nan).tolist()))
        if "S1" in pops and "M2" in pops:
            a, b = pops["S1"], pops["M2"]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = [np.corrcoef(a[:, k], b[:, k])[0, 1] if a[:, k].std() > 0 and b[:, k].std() > 0 else np.nan
                     for k in range(a.shape[1])]
            rows.append(dict(session_id=sid, trial_type=tname, area="S1-M2 sync", n_units=np.nan, n_trials=len(st),
                             rate=r, rate_norm=r))
    return rows


def safe(sid):
    try:
        r = run(sid)
        print(f"done {sid}", flush=True)
        return r
    except Exception:
        print(f"FAIL {sid}\n{traceback.format_exc()}", flush=True)
        return []


def figure(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = (EDGES[:-1] + 0.0005) * 1e3
    col = {"whisker": "#f7b519", "auditory": "#2c2cdb", "no-stim": "#262626"}
    panels = [("S1", "rate_norm", "SSp-bfd rate / baseline"), ("M2", "rate_norm", "wM1 + wM2 rate / baseline"),
              ("other", "rate_norm", "Other areas rate / baseline"), ("S1-M2 sync", "rate", "S1-M2 zero-lag correlation")]
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.8), constrained_layout=True)
    for ax, (area, colname, lab) in zip(axs, panels):
        for tname, c in col.items():
            x = np.array([v for v in df[(df.area == area) & (df.trial_type == tname)][colname]], float)
            if not len(x):
                continue
            m, se = np.nanmean(x, 0), np.nanstd(x, 0) / np.sqrt(np.sum(np.isfinite(x), 0).clip(1))
            ax.plot(t, m, color=c, lw=1, label=f"{tname} (n={len(x)})")
            ax.fill_between(t, m - se, m + se, color=c, alpha=.2, lw=0, edgecolor="none")
        ax.axvspan(-10, 5, color="0.85", lw=0, edgecolor="none", zorder=0)
        ax.axvline(5, color="r", lw=0.6, ls="--")
        ax.set_xlabel("Time from trial start (ms)")
        ax.set_ylabel(lab)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Raw spikes (1-ms bins, no artefact correction), mean +- s.e.m. over sessions; grey: -10..+5 ms removed in the analyses; red: +5 ms", fontsize=9)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=14)
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    sess = pd.read_csv(L.OUT / "sessions_v2.csv").session_id.unique()
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(delayed(safe)(s) for s in sess)
    df = pd.DataFrame([r for rr in res for r in rr])
    df.to_parquet(OUTD / "per_session.parquet")
    (OUTD / "config.json").write_text(json.dumps({"edges_s": EDGES.tolist(), "units": "stable good + mua",
                                                 "trials": L.CONFIG["trials"], "spikes": "raw, no correction"}))
    fig = figure(df)
    for ext in ("png", "pdf"):
        fig.savefig(OUTD / f"artifact_check.{ext}", dpi=200)


if __name__ == "__main__":
    main()
