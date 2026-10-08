"""166 -- Per-trial passive projections on the hit / miss axis for the example sessions (user 2026-10-08: "include the example sessions that
show what is being quantified").
Example sessions: those of 163 (learners, >= 300 tracked units: R+ closest to the R+ median change, R- most negative change).
Harmonised scaling (as 135 with SSL_135_HARM=1): 5-35 ms response minus the epoch's mean baseline (-55 to -20 ms), unit mean / SD from the
active trials. Real axis u = unit(mean_z(active whisker hits) - mean_z(misses)) from all active whisker trials (active trials with a
lick before 35 ms excluded). Every passive trial (whisker and auditory, pre and post) is projected on u, per neuron (/ sqrt(N)).
Shifted-label axes: hit / miss labels shifted by k = 10-50 % of the time-ordered active whisker trials (non-wrapping, both directions,
20 shifts, >= 10 trials per class), axis rebuilt, the same passive trials projected; null = mean over shifts per trial.
State space (2026-10-08): y = the passive-pre whisker pattern (mean evoked response over the pre whisker trials) orthogonalised to u,
unit length; each pre whisker trial uses the pattern of the other pre whisker trials (leave-one-out), so it is not projected on itself.
Alternative y2 (2026-10-08): the passive whisker change pattern (mean post - mean pre) orthogonalised to u (uses all trials, so
trials are not left out: illustration only).
Shifted axes: the same y; x on each shifted axis u_k; per shift the mean (x, y) of each epoch x stimulus (166_example_shifts.csv).
Output: 166_example_trials.csv (session, cohort, epoch, stim, trial index, real, null, y), 166_example_shifts.csv. Run (haas, repo root): python .../166_example_trials.py
"""
from __future__ import annotations

import importlib
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
EA = Path(__file__).resolve().parent
SCRIPTS = str(EA.parents[2] / "scripts")
sys.path[:0] = [str(EA), SCRIPTS]
M135 = importlib.import_module("135_whisker_lick_alignment_epochs")
M163 = importlib.import_module("163_example_state_space")
WIN, BASE, DZ = M135.WIN, M135.BASE, M135.DZ
N_SHIFT, FRAC, MIN_CLASS = 20, (0.10, 0.50), 10


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def run(sid, coh):
    os.chdir(EA.parents[2])
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    import ssl_timeresolved_decoding as T
    M132 = importlib.import_module("132_modality_stim_passive_active")
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = T.add_whole_brain_column(pd.read_parquet(T.AREA_LABELS_PATH))
    trs = M132.session_trials(sid, st, tt, T)
    act = trs["active"].reset_index(drop=True)
    rt = (act.lick_time - act.response_window_start_time).to_numpy()
    licked = (act.lick_flag == 1).to_numpy() & np.isfinite(rt)
    keep = ~licked | (rt > WIN[1]); act = act[keep].reset_index(drop=True); licked = licked[keep]
    trs = dict(trs, active=act)
    spikes = T.load_session_unit_spikes(root, sid)
    ids = importlib.import_module("tracked_units").load("stable").get(sid, np.array([], dtype=np.int64))
    units = np.intersect1d(T.area_units(sid, "whole_brain", "All units", labels), ids)
    X = {}
    for e in ("passive_pre", "active", "passive_post"):
        t0 = trs[e].start_time.to_numpy()
        r, b = T.sliding_bin_population_matrices(spikes, units, t0, np.ones(len(t0), bool), [WIN, BASE], dead_zone=DZ)
        X[e] = r - np.nanmean(b, 0, keepdims=True)
    mu, sd = X["active"].mean(0), X["active"].std(0); ok = sd > 1e-6        # units silent in every active window: float-residue SD
    Z = {e: (v[:, ok] - mu[ok]) / sd[ok] for e, v in X.items()}
    E = {e: v[:, ok] / sd[ok] for e, v in X.items()}
    N = int(ok.sum())
    isw = (act.trial_type == "whisker_trial").to_numpy()
    widx = np.where(isw)[0]; widx = widx[np.argsort(act.start_time.to_numpy()[widx], kind="stable")]
    y = licked[widx]
    u = unit(Z["active"][widx[y]].mean(0) - Z["active"][widx[~y]].mean(0))
    n = len(widx); ks = np.unique(np.linspace(FRAC[0] * n, FRAC[1] * n, N_SHIFT // 2).astype(int))
    us = []
    for k in np.r_[ks, -ks]:
        i = np.arange(max(0, -k), min(n, n - k)); ys, xi = y[i], widx[i + k]
        if min(ys.sum(), (~ys).sum()) >= MIN_CLASS:
            us.append(unit(Z["active"][xi[ys]].mean(0) - Z["active"][xi[~ys]].mean(0)))
    rows, srows = [], []
    isw_pre = (trs["passive_pre"].trial_type == "whisker_trial").to_numpy()
    Epw = E["passive_pre"][isw_pre]; S_ = Epw.sum(0); npw = len(Epw)
    def yax(pat):
        v = pat - (pat @ u) * u
        return v / np.linalg.norm(v)
    y_all = yax(S_ / npw)
    isw_post = (trs["passive_post"].trial_type == "whisker_trial").to_numpy()
    y2 = yax(E["passive_post"][isw_post].mean(0) - S_ / npw)
    for e in ("passive_pre", "passive_post"):
        stim = np.where((trs[e].trial_type == "whisker_trial").to_numpy(), "W", "A")
        real = E[e] @ u / np.sqrt(N)
        null = np.mean([E[e] @ uk for uk in us], 0) / np.sqrt(N)
        ycoord = E[e] @ y_all / np.sqrt(N)
        if e == "passive_pre":                                       # leave-one-out y for the pre whisker trials
            for jj, j in enumerate(np.where(isw_pre)[0]):
                ycoord[j] = E[e][j] @ yax((S_ - Epw[jj]) / (npw - 1)) / np.sqrt(N)
        y2c = E[e] @ y2 / np.sqrt(N)
        rows += [dict(session_id=sid, cohort=coh, epoch=e, stim=s_, trial=j, real=a, null=b, y=c_, y2=d_)
                 for j, (s_, a, b, c_, d_) in enumerate(zip(stim, real, null, ycoord, y2c))]
        for k, uk in enumerate(us):
            xk = E[e] @ uk / np.sqrt(N)
            for s_ in ("W", "A"):
                m = stim == s_
                srows.append(dict(session_id=sid, cohort=coh, shift=k, epoch=e, stim=s_, x=xk[m].mean(), y=ycoord[m].mean(), y2=y2c[m].mean()))
    print(sid, coh, N, "units,", len(us), "shifts", flush=True)
    return rows, srows


def main():
    rows, srows = [], []
    for sid, _, coh in M163.pick():
        a, b = run(sid, coh); rows += a; srows += b
    pd.DataFrame(rows).to_csv(EA / "166_example_trials.csv", index=False)
    pd.DataFrame(srows).to_csv(EA / "166_example_shifts.csv", index=False)
    print("ALL DONE", EA / "166_example_trials.csv")


if __name__ == "__main__":
    main()
