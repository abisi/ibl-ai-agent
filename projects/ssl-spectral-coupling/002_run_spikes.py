"""Spectral coupling between areas from binned spikes of stable units (5-ms bins), whisker day 0, all sessions.

Per session, draw and condition: trials drawn (ALL: 40, events: 15), condition mean removed per unit and bin, soft
normalisation (spec_lib.residuals); per area 15 (area group, area_group_coarse_v3) or 8 (area acronym,
area_acronym_custom of units with a v3 group) random stable units with rate >= 1 Hz; sum signal + 3 PCs; rest = top 3
PCs of up to 200 random units outside both areas. Measures (spec_lib): msc, imc, cancoh, pcoh, gc_ab, gc_ba per band
and on a 2-Hz grid (epochs, group level), psi per band; observed vs source-trial shuffle null.
Windows: epochs quiet / early / execution (both levels), 200-ms time-resolved windows (group level, event conditions).
Conditions: ALL (quiet only), W_lick, W_nolick, A_hit, NS_cr (needs >= 15 trials).
Sessions: whisker day 0 from the pilot coverage table, cohort / learner / exclusions re-read from the current sheet.
Outputs: OUT/<source>/sessions/<session>_{bands,spectra}.parquet + _meta.json  (source = spikes)
Run (haas, repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python \
    projects/ssl-spectral-coupling/002_run_spikes.py [--n-proc 40] [--sessions S ...] [--quick]
"""

import argparse
import json
import os
import sys
import time
import zlib
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spec_lib as sl  # noqa: E402

CONFIG = {
    "version": 1, "source": "spikes", "bin_s": sl.BIN, "edges_s": [float(sl.EDGES[0]), float(sl.EDGES[-1])],
    "epochs": sl.EPOCHS, "time_resolved": sl.TR, "bands": sl.BANDS, "min_rate_hz": 1.0,
    "levels": {"group": {"n_units": 15, "n_draws": 3, "time_resolved": True, "ceiling": True},
               "acronym": {"n_units": 8, "n_draws": 2, "time_resolved": False, "ceiling": True}},
    "conditions": {"ALL": 40, "W_lick": 15, "W_nolick": 15, "A_hit": 15, "NS_cr": 15},
    "n_shuffles_epoch": 20, "n_shuffles_tr": 10, "n_pc": sl.N_PC, "n_rest_pc": sl.N_REST_PC,
    "n_rest_units": sl.N_REST_UNITS, "var_order": sl.VAR_ORDER, "soft_sd": sl.SOFT_SD,
    "unit_set": "stable good + mua (137_stable_units via pilot 012 load); all good/mua if no drift file",
    "trials": "context active (per-trial rule), perf != 6, early_lick == 0, warm-up cut (1 trial before first whisker), A1 trim",
}
WITHIN = " (within)"
# AP envelope: the artefact-corrected binary has a deterministic dip in the 0-5 ms bin; epochs kept 10 ms away from it
AP_EPOCHS = {"quiet": (-1.0, -0.01, 3.0), "early": (0.01, 0.16, 1.5), "execution": (0.2, 0.5, 3.0)}
# v2 (user 2026-11-18): jaw-onset alignment of lick trials (DLC jaw onset, behaviour_analysis per-mouse pkl);
# time-domain coupling in 50-ms bins ("corr50": Pearson r of the area sum residuals over trials x 50-ms bins)
JAW_EPOCHS = {"pre": (-0.3, 0.0, 1.5), "post": (0.0, 0.3, 1.5)}
JAW_TR = {"win": 0.2, "step": 0.05, "centers": (-0.6, 0.6), "NW": 2.0}
JAW_CONDITIONS = {"W_lick": 15, "A_hit": 15, "NS_fa": 10}
BIN50 = 10  # 5-ms bins per 50-ms bin


def jaw_onsets(sid):
    """{round(start_time, 3): jaw onset (s from trial start)} from the mouse's dlc_jaw_onset_times.pkl."""
    d = pd.read_pickle(sl.ROOT / "combined_results_ks4" / sid.split("_")[0] / "dlc_jaw_onset_times.pkl")
    d = d[d.session_id == sid]
    return dict(zip(d.start_time.round(3), d.jaw_dlc_onset))


def corr50(xa, xb, perms):
    """Pearson r of two [trial, 5-ms bin] signals summed into 50-ms bins (window cut to whole 50-ms bins), per perm."""
    nb = xa.shape[1] // BIN50
    if nb < 1:
        return np.full(len(perms), np.nan)
    a = xa[:, :nb * BIN50].reshape(len(xa), nb, BIN50).sum(2)
    b = xb[:, :nb * BIN50].reshape(len(xb), nb, BIN50).sum(2).ravel()
    b = b - b.mean()
    out = []
    for perm in perms:
        x = a[perm].ravel()
        x = x - x.mean()
        out.append(float(x @ b / (np.sqrt((x @ x) * (b @ b)) + 1e-30)))
    return np.array(out)


def session_table():
    sys.path.insert(0, str(sl.PILOT.parent / "ssl-paper-figures"))
    from common import REFERENCE_SHEET
    s = pd.read_csv(sl.cl.OUT / "coverage" / "session_coverage.csv")[["session_id", "mouse_id"]]
    sheet = pd.read_excel(REFERENCE_SHEET).drop_duplicates("mouse_id").set_index("mouse_id")
    for c in ("reward_group", "learning_category", "exclude", "exclude_ephys", "recording"):
        s[c] = s.mouse_id.map(sheet[c])
    s["cohort"] = s.reward_group
    s["learner"] = s.learning_category.isin(["good", "moderate"])
    keep = (s.exclude == 0) & (s.exclude_ephys == 0) & s.cohort.isin(["R+", "R-"]) & (s.recording == 1) & \
        ~s.mouse_id.isin(["AB068", "AB077"])
    return s[keep].reset_index(drop=True), str(REFERENCE_SHEET)


def run_session(args):
    sid, cohort, learner, quick, levels, source, align = args
    t0 = time.time()
    C = {**CONFIG, "source": source, "align": align}
    units, trials, rule = sl.load_session(sid)
    if align == "jaw":
        C["conditions"] = JAW_CONDITIONS
        on = trials.start_time.round(3).map(jaw_onsets(sid))
        ok = trials.condition.isin(list(JAW_CONDITIONS)) & (on > 0) & (on < 1.0)  # stimulus-evoked onsets only
        trials = trials[ok].assign(jaw_onset=on[ok]).reset_index(drop=True)
        trials["start_time"] = trials.start_time + trials.jaw_onset  # t = 0 is the jaw onset
    if source == "ap":  # channels (peak channels of stable units) play the role of units; envelope in place of counts
        cache = np.load(sl.OUT / "ap" / "cache" / f"{sid}.npz")
        X = cache["E"]
        trials = trials.iloc[cache["trial_index"]].reset_index(drop=True)
        assert np.allclose(trials.start_time.to_numpy(), cache["start_time"])
        units = pd.read_parquet(sl.OUT / "ap" / "cache" / f"{sid}_channels.parquet")
        ok = np.isfinite(X).all((1, 2))  # trials whose chunk fell outside the recording
        X, trials = X[ok], trials[ok].reset_index(drop=True)
        soft = 0.0
        wins_override = AP_EPOCHS
    else:
        starts = trials.start_time.to_numpy()
        X = sl.counts(units.spike_times.tolist(), starts)
        rate = X.sum((0, 1)) / (len(starts) * (sl.EDGES[-1] - sl.EDGES[0]))
        keep = rate >= C["min_rate_hz"]
        units, X = units[keep].reset_index(drop=True), X[:, :, keep]
        soft = sl.SOFT_SD
        wins_override = None
    labels = {"group": units.area_group.to_numpy(object),
              "acronym": np.where(units.area_group.notna(), units.area_acronym_custom, None).astype(object)}
    cond_all = trials.condition.to_numpy()
    wins = sl.windows()
    if align == "jaw":
        wins = {k: (sl.window_idx(a, b), nw) for k, (a, b, nw) in JAW_EPOCHS.items()}
        for c in np.round(np.arange(JAW_TR["centers"][0], JAW_TR["centers"][1] + 1e-9, JAW_TR["step"]), 3):
            wins[f"tr_{c:+.2f}"] = (sl.window_idx(c - JAW_TR["win"] / 2, c + JAW_TR["win"] / 2), JAW_TR["NW"])
    if wins_override:
        wins.update({k: (sl.window_idx(a, b), nw) for k, (a, b, nw) in wins_override.items()})
    if quick:
        wins = {k: v for k, v in wins.items() if not k.startswith("tr_") or k in ("tr_-0.50", "tr_+0.10")}
    acc = defaultdict(list)       # (level, cond, window, a, b, measure, band) -> [(obs, null_mean, null_sd)]
    spec_acc = defaultdict(list)  # (level, cond, window, a, b, measure) -> [(obs grid, null grid)]
    meta = dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=cohort, learner=learner, unit_rule=rule,
                n_trials={c: int((cond_all == c).sum()) for c in np.unique(cond_all)}, n_trials_total=int(len(trials)),
                config=C)
    rng0 = zlib.crc32(sid.encode())
    for level in levels:
        L = C["levels"][level]
        lab = labels[level]
        members = {g: np.flatnonzero(lab == g) for g in pd.unique(lab[pd.notna(lab)])}
        areas = sorted(g for g, m in members.items() if len(m) >= L["n_units"])
        ceil = [g for g in areas if L["ceiling"] and len(members[g]) >= 2 * L["n_units"]]
        pairs = [(a, b) for i, a in enumerate(areas) for b in areas[i + 1:]] + [(g + WITHIN, g) for g in ceil]
        meta[f"n_units_{level}"] = {g: int(len(m)) for g, m in members.items()}
        meta[f"areas_{level}"] = areas
        if len(areas) < 2:
            continue
        for draw in range(1 if quick else L["n_draws"]):
            rng = np.random.default_rng(rng0 + 7919 * draw + (0 if level == "group" else 104729))
            for cond, n_tr in C["conditions"].items():
                pool = np.arange(len(trials)) if cond == "ALL" else np.flatnonzero(cond_all == cond)
                if len(pool) < n_tr:
                    continue
                tr = np.sort(rng.choice(pool, n_tr, replace=False))
                Xtr = X[tr]
                R_all = sl.residuals(Xtr, cond_all[tr], soft)
                sel = {}
                for g in areas:
                    pick = rng.permutation(members[g])
                    sel[g] = pick[:L["n_units"]]
                    if g in ceil:
                        sel[g + WITHIN] = pick[L["n_units"]:2 * L["n_units"]]
                sig = {g: sl.area_signals(R_all[:, :, u]) for g, u in sel.items()}
                rest = {}
                for a, b in pairs:
                    excl = np.isin(lab, [a.replace(WITHIN, ""), b])
                    cand = np.flatnonzero(~excl)
                    if len(cand) >= 2 * sl.N_REST_PC:
                        cand = rng.choice(cand, min(len(cand), sl.N_REST_UNITS), replace=False)
                        rest[(a, b)] = sl.pcs(R_all[:, :, cand], sl.N_REST_PC)
                    else:
                        rest[(a, b)] = None
                for wname, (idx, nw) in wins.items():
                    is_tr = wname.startswith("tr_")
                    if (cond == "ALL" and wname != "quiet") or (is_tr and not L["time_resolved"]):
                        continue
                    n_sh = (2 if quick else (C["n_shuffles_tr"] if is_tr else C["n_shuffles_epoch"]))
                    bands = sl.bands_for(len(idx) * sl.BIN)
                    F, x = {}, {}
                    for g, s in sig.items():
                        F[g], f = sl.tfft(s, idx, nw)
                        xs = s[:, idx, 0].astype(np.float64)
                        x[g] = xs - xs.mean(1, keepdims=True)
                    for g in areas:  # power of the sum signal and firing rate (confound checks)
                        pw = (np.abs(F[g][..., 0]) ** 2).mean((0, 1))
                        for band, v in sl.band_means(pw, f, bands).items():
                            acc[(level, cond, wname, g, g, "power", band)].append((v, np.nan, np.nan))
                        fr = Xtr[:, idx][:, :, sel[g]].mean() / sl.BIN
                        acc[(level, cond, wname, g, g, "rate", "all")].append((float(fr), np.nan, np.nan))
                    perms = [np.arange(n_tr)] + [rng.permutation(n_tr) for _ in range(n_sh)]
                    for a, b in pairs:
                        Z = rest[(a, b)]
                        FZ = None if Z is None else sl.tfft(Z, idx, nw)[0]
                        spec, psi = sl.pair_all(F[a], F[b], FZ, x[a], x[b], f, perms, bands)
                        for m, v in spec.items():
                            bm = np.array([[d[k] for k in bands] for d in (sl.band_means(r, f, bands) for r in v)])
                            for j, band in enumerate(bands):
                                acc[(level, cond, wname, a, b, m, band)].append(
                                    (bm[0, j], np.nanmean(bm[1:, j]), np.nanstd(bm[1:, j])))
                            if not is_tr and level == "group":
                                spec_acc[(level, cond, wname, a, b, m)].append(
                                    (sl.to_grid(v[0], f), sl.to_grid(np.nanmean(v[1:], 0), f)))
                        for band, v in psi.items():
                            acc[(level, cond, wname, a, b, "psi", band)].append((v[0], v[1:].mean(), v[1:].std()))
                        sa, sb = sig[a][:, idx, 0].astype(np.float64), sig[b][:, idx, 0].astype(np.float64)
                        r = corr50(sa, sb, perms)
                        acc[(level, cond, wname, a, b, "corr50", "all")].append((r[0], np.nanmean(r[1:]), np.nanstd(r[1:])))
            print(sid, level, "draw", draw, f"{time.time() - t0:.0f} s", flush=True)
    rows = [dict(level=k[0], condition=k[1], window=k[2], source=k[3], target=k[4], measure=k[5], band=k[6],
                 value=float(np.nanmean([v[0] for v in vals])), null_mean=float(np.nanmean([v[1] for v in vals])),
                 null_sd=float(np.nanmean([v[2] for v in vals])), n_draws=len(vals)) for k, vals in acc.items()]
    srows = []
    for k, vals in spec_acc.items():
        obs, nul = np.nanmean([v[0] for v in vals], 0), np.nanmean([v[1] for v in vals], 0)
        for fq, o, n in zip(sl.SPEC_GRID, obs, nul):
            if np.isfinite(o):
                srows.append(dict(level=k[0], condition=k[1], window=k[2], source=k[3], target=k[4], measure=k[5],
                                  freq=float(fq), value=float(o), null_mean=float(n)))
    od = sl.OUT / outdir(source, align) / "sessions"
    od.mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if quick else ""
    for name, rr in (("bands", rows), ("spectra", srows)):
        d = pd.DataFrame(rr)
        d.insert(0, "session_id", sid)
        d.insert(1, "mouse_id", meta["mouse_id"])
        d.insert(2, "cohort", cohort)
        d.insert(3, "learner", learner)
        tmp = od / f"{sid}{sfx}_{name}.tmp.parquet"
        d.to_parquet(tmp)
        os.replace(tmp, od / f"{sid}{sfx}_{name}.parquet")
    meta["runtime_s"] = round(time.time() - t0)
    json.dump(meta, open(od / f"{sid}{sfx}_meta.json", "w"), indent=1, default=str)
    return sid, len(rows), len(srows), meta["runtime_s"]


def outdir(source, align):
    return source if align is None else f"v2_{align}"


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
    ap.add_argument("--levels", nargs="*", default=["group", "acronym"])
    ap.add_argument("--n-proc", type=int, default=40)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--source", choices=["spikes", "ap"], default="spikes")
    ap.add_argument("--align", choices=["stim", "jaw"], default=None, help="v2 run (outputs in v2_<align>)")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    s, sheet = session_table()
    if args.sessions:
        s = s[s.session_id.isin(args.sessions)]
    od = sl.OUT / outdir(args.source, args.align)
    done = {p.name.replace("_meta.json", "") for p in (od / "sessions").glob("*_meta.json")} if not args.quick else set()
    if args.source == "ap":
        s = s[s.session_id.map(lambda x: (sl.OUT / "ap" / "cache" / f"{x}.npz").exists())]
    jobs = [(r.session_id, r.cohort, bool(r.learner), args.quick, args.levels, args.source, args.align) for r in s.itertuples()
            if r.session_id not in done]
    od.mkdir(parents=True, exist_ok=True)
    s.to_csv(od / "sessions_included.csv", index=False)
    json.dump({**CONFIG, "source": args.source, "align": args.align,
               "jaw": dict(epochs=JAW_EPOCHS, tr=JAW_TR, conditions=JAW_CONDITIONS, onset_rule="0 < onset < 1 s",
                           file="combined_results_ks4/<mouse>/dlc_jaw_onset_times.pkl") if args.align == "jaw" else None, "ap_epochs": AP_EPOCHS if args.source == "ap" else None,
               "reference_sheet": sheet}, open(od / "config.json", "w"), indent=1, default=str)
    print(f"{len(jobs)} sessions to run ({len(done)} already done)", flush=True)
    with Pool(min(args.n_proc, max(1, len(jobs)))) as pool:
        for res in pool.imap_unordered(run_safe, jobs):
            print("done", res, flush=True)


if __name__ == "__main__":
    main()
