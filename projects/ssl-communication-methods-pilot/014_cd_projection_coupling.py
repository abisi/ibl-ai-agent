"""Coupling between area groups along task coding directions (CDs), Chen et al. 2024 (Cell 187:676) style (user request
2026-11-09: "could we do something similar with the prelick CD or the hit-miss CD? then compare" [with the RRR v2]).

Chen et al.: per area, a coding direction (mean lick-right minus lick-left), single-trial projections on held-out trials
(leave-one-trial-out), Spearman correlation of the projections between simultaneously recorded areas in 200-ms bins,
both trial types pooled, mean over sessions. Here the same, with two CDs of this project family and controls they did
not report:
  hitmiss  whisker hit minus whisker miss; per-trial rate 5-35 ms after stimulus minus the unit's -55..-20 ms baseline,
           z-scored per unit over the whisker trials (ssl-hitmiss-axis-passive 140). Whisker trials with early_lick == 0
           and no lick before 35 ms. Projections stimulus-aligned.
  prelick  auditory hit minus spontaneous lick (SL; = the false alarms of the project, user rule); per-event rate in
           [-100, 0) ms before the (corrected) first lick minus the 051 baseline, z-scored per unit over AH + SL
           (ssl-prelick-convergence 051 / 086, PRELICK_REF=sl). Projections lick-aligned; whisker hits (WH) are projected
           too (not used to build the CD).
  Trials / events: 051 active_trials / classify (active, perf != 6, warm-up cut, rule A1) and 051 spontaneous_licks.
  Leave-one-out: a trial of a CD class is projected on the CD built without it; other trials on the full CD.
  Projection in a window: spike counts in the 200-ms window, scaled per unit by the SD of its CD-window rate, dot the
  unit-norm CD (no per-window re-centring: correlations are across trials, the mean cancels).
Coupling per unordered pair of area groups, window and trial set:
  r          Spearman correlation of the two areas' projections across trials;
  trial sets: within class (hitmiss: W_hit, W_miss; prelick: AH, SL, WH) = trial-to-trial co-fluctuation along the CD,
             and pooled over the two CD classes (hitmiss: hm_pooled; prelick: ahsl_pooled) = Chen et al.'s measure, which
             also contains the class difference both areas encode;
  null       N_PERM permutations of one area's trials within class: r_null_mean / r_null_sd / p (two-sided);
  r_rand     the same correlation on N_RAND random unit axes per area (k-th axis of A with k-th of B), mean: coupling of
             an arbitrary direction (near 0: random axes average out a common gain change);
  r_gain     the same correlation on the population-mean axis (all units weighted equally): shared gain / arousal; the CD
             coupling is task-axis specific if it exceeds this;
  r_partial  Spearman partial correlation after regressing both projections on the top REST_PCS PCs of every other unit
             of the session (outside the two groups; same window and trials): coupling not explained by the recorded rest
             of the brain (as partial CCA in 012).
Per area and CD also cd_dprime: held-out (leave-one-out) d' between the two CD classes (is the CD meaningful there).
Units and areas: as 012 (stable good + mua, >= 1 Hz, area_group_coarse_v3), >= MIN_UNITS units per group (Chen: 10),
whisker artefact correction (roc_utils_new.correct_neuron_spike_train, all whisker onsets). >= MIN_TRIALS per class.
Output: OUT/cd_coupling/sessions/<session>.parquet (+ _meta.json with per-area d').
Run (haas, repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python
    projects/ssl-communication-methods-pilot/014_cd_projection_coupling.py [--n-proc 48] [--sessions S ...]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import warnings
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import comm_lib as cl  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# area set (env CD_AREAS, user 2026-11-09): "all" = area_group_coarse_v3 (cd_coupling/); "seven" = 016 SEVEN groups
# (somatosensory cortex = SSp-w + SSp-orofacial, auditory, midbrain, thalamus, hippocampus, frontal, striatum;
# cd_coupling_seven/); units of other groups stay in the rest of the recording
AREAS = os.environ.get("CD_AREAS", "all")
OUT = cl.OUT / ("cd_coupling" if AREAS == "all" else f"cd_coupling_{AREAS}")
SEVEN = {"SSp-w": "Somatosensory cortex", "SSp-orofacial": "Somatosensory cortex", "Auditory areas": "Auditory areas",
         "Midbrain": "Midbrain", "Thalamus": "Thalamus", "Hippocampus": "Hippocampus",
         "Motor and frontal areas": "Motor and frontal areas", "Striatum and pallidum": "Striatum and pallidum"}
CONFIG = {"version": 1, "reference": "Chen et al. 2024 Cell 187:676 (CD projections, Spearman across areas, 200-ms bins)",
          "min_units": 10, "min_trials": 5, "min_rate_hz": 1.0, "win_s": 0.2, "step_s": 0.05,
          "stim_centers": (-0.4, 0.8), "lick_centers": (-0.5, 0.3), "hitmiss_window": (0.005, 0.035),
          "hitmiss_baseline": (-0.055, -0.020), "prelick_window": (-0.100, 0.0), "n_perm": 100, "n_rand": 20,
          "rest_pcs": 5, "area_level": "area_group_coarse_v3"}


def modules():
    m012 = _load("rrr_v2", HERE / "012_rrr_timeresolved_v2.py")
    m051 = _load("m051", HERE.parent / "ssl-prelick-convergence" / "exploratory-analyses" / "051_roc_prelick.py")
    return m012, m051


def load(sid, m012, m051):
    """012 unit set (stable good + mua, area labels) with whisker-artefact-corrected spike trains; 051 trials and SL."""
    from pynwb import NWBHDF5IO
    units, _, _, rule = m012.load(sid)
    with NWBHDF5IO(str(cl.NWB_DIR / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        raw = nwb.trials.to_dataframe()
        t, log = m051.active_trials(raw, sid)
        sl = m051.spontaneous_licks(nwb, raw, log["epoch"])
    onsets = raw.loc[raw.whisker_stim == 1, "start_time"].to_numpy()
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    spikes = [m051.ru.correct_neuron_spike_train(np.sort(np.asarray(s, float)), onsets, rng) for s in units.spike_times]
    lick_t, _ = m051.classify(t, dict(log))
    return units, spikes, t, lick_t, sl, rule


def counts(spikes, events, lo, hi):
    """[event, unit] spike counts in [event + lo, event + hi)."""
    out = np.zeros((len(events), len(spikes)), np.float32)
    for u, s in enumerate(spikes):
        out[:, u] = np.searchsorted(s, events + hi) - np.searchsorted(s, events + lo)
    return out


def loo_axes(Z, a, b):
    """Z [trial, unit] (z-scored CD-window responses); a, b boolean class masks. Per trial the unit-norm CD
    (mean a - mean b), leaving the trial out if it belongs to a or b; and the held-out d'."""
    sa, sb, na, nb = Z[a].sum(0), Z[b].sum(0), a.sum(), b.sum()
    full = sa / na - sb / nb
    D = np.tile(full, (len(Z), 1))
    D[a] = (sa - Z[a]) / (na - 1) - sb / nb
    D[b] = sa / na - (sb - Z[b]) / (nb - 1)
    D /= np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    proj = (Z * D).sum(1)
    pa, pb = proj[a], proj[b]
    dp = (pa.mean() - pb.mean()) / np.sqrt(0.5 * (pa.var(ddof=1) + pb.var(ddof=1)) + 1e-12)
    return D, float(dp)


def _rank(x):
    return stats.rankdata(x, axis=0)


def spearman_perm(x, y, groups, n_perm, rng):
    """Spearman r of x, y; null by permuting y within groups."""
    rx, ry = _rank(x), _rank(y)
    r = np.corrcoef(rx, ry)[0, 1]
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = np.arange(len(y))
        for g in np.unique(groups):
            m = np.flatnonzero(groups == g)
            p[m] = rng.permutation(m)
        null[k] = np.corrcoef(rx, ry[p])[0, 1]
    return r, null


def partial_spearman(x, y, Z):
    rx, ry = _rank(x), _rank(y)
    A = np.column_stack([np.ones(len(x)), Z])
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1]) if ex.std() > 0 and ey.std() > 0 else np.nan


def run_session(args):
    sid, cohort, learner = args
    t0 = time.time()
    C = CONFIG
    warnings.simplefilter("ignore")
    m012, m051 = modules()
    cov3 = cl._coverage_lib()
    units, spikes, t, lick_t, sl, rule = load(sid, m012, m051)
    rng = np.random.default_rng(zlib.crc32(sid.encode()))
    # rate filter on all analysed trials (as 012: >= 1 Hz over -0.6..1.0 s around trial starts)
    rate = counts(spikes, t.start_time.to_numpy(), -0.6, 1.0).mean(0) / 1.6
    keep = rate >= C["min_rate_hz"]
    units, spikes = units[keep].reset_index(drop=True), [s for s, k in zip(spikes, keep) if k]
    group = units.area_acronym_custom.map(cov3.COARSE_GROUP_V3_OF)
    group = (group.map(SEVEN) if AREAS == "seven" else group).to_numpy()
    members = {g: np.flatnonzero(group == g) for g in pd.unique(group[pd.notna(group)])}
    groups = sorted(g for g, m in members.items() if len(m) >= C["min_units"])
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
    rows, dprimes = [], []

    def centers(rng_):
        return np.round(np.arange(rng_[0], rng_[1] + 1e-9, C["step_s"]), 3)

    # ---------------------------------------------------------------- the two CDs: events, classes, CD-window responses
    specs = {}
    w = t[(t.trial_type == "whisker_trial") & (t.early_lick == 0)].copy()
    rt = w.lick_time - w.response_window_start_time
    w = w[~((w.lick_flag == 1) & (rt < C["hitmiss_window"][1]))]
    ev = w.start_time.to_numpy()
    resp = (counts(spikes, ev, *C["hitmiss_window"]) / np.diff(C["hitmiss_window"])[0]
            - counts(spikes, ev, *C["hitmiss_baseline"]) / np.diff(C["hitmiss_baseline"])[0])
    lab = np.where(w.lick_flag.to_numpy() == 1, "W_hit", "W_miss")
    specs["hitmiss"] = dict(events=ev, resp=resp, lab=lab, a="W_hit", b="W_miss", cd_mask=np.ones(len(ev), bool),
                            centers=centers(C["stim_centers"]), pooled="hm_pooled")
    lk = lick_t[lick_t.cls.isin(["WH", "AH"])]
    ev = np.r_[lk.first_lick_time.to_numpy(), sl]
    lab = np.r_[lk.cls.to_numpy(), np.full(len(sl), "SL")]
    base = np.r_[np.column_stack([lk.base_lo, lk.base_hi]), np.column_stack([sl + m051.SL_BASE[0], sl + m051.SL_BASE[1]])]
    win = counts(spikes, ev, *C["prelick_window"]) / np.diff(C["prelick_window"])[0]
    bas = np.zeros_like(win)
    for u, s in enumerate(spikes):
        bas[:, u] = (np.searchsorted(s, base[:, 1]) - np.searchsorted(s, base[:, 0])) / (base[:, 1] - base[:, 0])
    specs["prelick"] = dict(events=ev, resp=win - bas, lab=lab, a="AH", b="SL", cd_mask=np.isin(lab, ["AH", "SL"]),
                            centers=centers(C["lick_centers"]), pooled="ahsl_pooled")
    n_cls = {cd: {c: int((s["lab"] == c).sum()) for c in np.unique(s["lab"])} for cd, s in specs.items()}

    for cd, s in specs.items():
        lab, m = s["lab"], s["cd_mask"]
        if min((lab == s["a"]).sum(), (lab == s["b"]).sum()) < C["min_trials"] or len(groups) < 2:
            continue
        sd = s["resp"][m].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Zcd = (s["resp"] - s["resp"][m].mean(0)) / sd
        axes, rand = {}, {}
        for g in groups:
            u = members[g]
            D = np.zeros((len(lab), len(u)))
            D[m], dp = loo_axes(Zcd[m][:, u], lab[m] == s["a"], lab[m] == s["b"])
            full = Zcd[m][:, u][lab[m] == s["a"]].mean(0) - Zcd[m][:, u][lab[m] == s["b"]].mean(0)
            D[~m] = full / max(np.linalg.norm(full), 1e-12)
            axes[g] = D
            R = rng.standard_normal((C["n_rand"], len(u)))
            rand[g] = R / np.linalg.norm(R, axis=1, keepdims=True)
            dprimes.append(dict(session_id=sid, cd=cd, area=g, n_units=int(len(u)), cd_dprime=dp))
        sets = {c: lab == c for c in np.unique(lab)}
        sets[s["pooled"]] = m
        for wc in s["centers"]:
            X = counts(spikes, s["events"], wc - C["win_s"] / 2, wc + C["win_s"] / 2) / sd  # unit-SD scaled
            proj = {g: (X[:, members[g]] * axes[g]).sum(1) for g in groups}
            prand = {g: X[:, members[g]] @ rand[g].T for g in groups}
            pgain = {g: X[:, members[g]].mean(1) for g in groups}
            for name, sm in sets.items():
                if sm.sum() < C["min_trials"]:
                    continue
                gl = lab[sm]
                for a, b in pairs:
                    x, y = proj[a][sm], proj[b][sm]
                    if x.std() == 0 or y.std() == 0:
                        continue
                    r, null = spearman_perm(x, y, gl, C["n_perm"], rng)
                    rr = np.nanmean([stats.spearmanr(prand[a][sm, k], prand[b][sm, k]).statistic for k in range(C["n_rand"])])
                    rg = stats.spearmanr(pgain[a][sm], pgain[b][sm]).statistic
                    rest = np.flatnonzero(~np.isin(group, [a, b]))
                    rp = np.nan
                    if len(rest) >= C["rest_pcs"] and sm.sum() > C["rest_pcs"] + 3:
                        Rr = X[sm][:, rest]
                        Rr = Rr - Rr.mean(0)
                        _, _, Vt = np.linalg.svd(Rr, full_matrices=False)
                        rp = partial_spearman(x, y, Rr @ Vt[:C["rest_pcs"]].T)
                    rows.append(dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=cohort, learner=learner, cd=cd,
                                     trial_set=name, area_a=a, area_b=b, window_center=float(wc), n_trials=int(sm.sum()),
                                     r=float(r), r_null_mean=float(np.nanmean(null)), r_null_sd=float(np.nanstd(null)),
                                     p=float((1 + np.sum(np.abs(null) >= abs(r))) / (1 + len(null))),
                                     r_rand=float(rr), r_gain=float(rg), r_partial=float(rp)))
    meta = dict(session_id=sid, cohort=cohort, learner=learner, unit_rule=rule, groups=groups,
                n_units={g: int(len(v)) for g, v in members.items()}, n_events=n_cls, config=C)
    (OUT / "sessions").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "sessions" / f"{sid}.parquet")
    pd.DataFrame(dprimes).to_parquet(OUT / "sessions" / f"{sid}_dprime.parquet")
    json.dump(meta, open(OUT / "sessions" / f"{sid}_meta.json", "w"), indent=1, default=str)
    return sid, len(groups), len(rows), round(time.time() - t0)


def run_safe(job):
    try:
        return run_session(job)
    except Exception as err:  # noqa: BLE001 - logged, run continues
        import traceback
        traceback.print_exc()
        return job[0], f"ERROR {type(err).__name__}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--n-proc", type=int, default=24)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    s = pd.read_csv(cl.OUT / "coverage" / "session_coverage.csv")
    s = s[(s.exclude_ephys != 1) & s.cohort.isin(["R+", "R-"])]
    if args.sessions:
        s = s[s.session_id.isin(args.sessions)]
    done = {p.stem for p in (OUT / "sessions").glob("*.parquet")}
    jobs = [(r.session_id, r.cohort, bool(r.learner)) for r in s.itertuples() if r.session_id not in done]
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(CONFIG, open(OUT / "config.json", "w"), indent=1, default=str)
    print(f"{len(jobs)} sessions to run ({len(done)} already done)", flush=True)
    with Pool(min(args.n_proc, max(1, len(jobs)))) as pool:
        for res in pool.imap_unordered(run_safe, jobs):
            print("done", res, flush=True)


if __name__ == "__main__":
    main()
