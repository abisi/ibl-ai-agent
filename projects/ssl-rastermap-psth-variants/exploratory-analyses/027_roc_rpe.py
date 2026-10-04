"""Single-neuron RPE ROC (separate from the main ROC results), one file per session.

Timing: first lick L1 = start_time + (lick_time - response_window_start_time) (corrected NWB lick time); reward contact
~ second lick -> OUTCOME window = [L1 + ILI, L1 + ILI + 150 ms], ILI = session median piezo inter-lick interval.
BASELINE = [start_time - 200 ms, start_time] (pre-trial). All values are firing RATES (spikes/s). Spike trains are
artifact-corrected as in roc_utils_new.process_nwb_tables; all units (no QC filter, as the main ROC).
ROC as roc_utils_new: AUC with class 2 as the positive class, selectivity = 2(AUC - 0.5) (+ = class 2 higher),
1000 label permutations, one-tailed p in the direction of the observed selectivity, significant if p < 0.05.
Vs-baseline comparisons use raw rates (outcome vs baseline of the same trials); between-trial-type comparisons use
baseline-subtracted outcome rates.

comparison (class1 -> class2)                                    RPE prediction
reward_vs_baseline            baseline -> outcome, rewarded hits (auditory hits + R+ whisker hits)       +
reward_vs_baseline_whisker    same, R+ whisker hits                                                       +
reward_vs_baseline_auditory   same, auditory hits                                                         +
early_vs_late_whisker         late -> early rewarded whisker hits (R+), split at the learning trial       +
                              (learning_curve whisker learning_trial; median split if NaN)
early_vs_late_auditory        late -> early auditory hits (median split; control)                         +
unexpected_vs_expected        auditory hits -> early whisker hits (R+; same reward, different expectation) +
unexpected_vs_expected_all    auditory hits -> all whisker hits (R+)                                      +
unrewarded_lick_vs_baseline   baseline -> outcome, false alarms + R- whisker hits                         -
fa_vs_baseline                baseline -> outcome, false alarms (no-stim licks)                           -
rminus_whisker_hit_vs_baseline  baseline -> outcome, R- whisker hits                                      -
Columns: keys (mouse_id, session_id, electrode_group, cluster_id, neuron_id), cohort, comparison, n1, n2, auc,
selectivity, p_value, significant, rpe_prediction (+1/-1), rpe_consistent (significant AND sign == prediction),
split / timing metadata.
Output -> combined_results_ks4/<mouse>/whisker_<day>/roc_analysis_rpe/<mouse>_roc_rpe.csv (+ _config.json)
"""
import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy.stats import rankdata

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
from roc_analysis import roc_utils_new as ru                              # noqa: E402

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
OUT_W = 0.15
BASE = (-0.2, 0.0)
N_PERM, ALPHA, MIN_TRIALS, SEED = ru.N_PERMUTATIONS, ru.ALPHA, 5, 0


def rates(st, t0, t1):
    return (np.searchsorted(st, t1) - np.searchsorted(st, t0)) / np.maximum(t1 - t0, 1e-3)


def roc_perm(x1, x2, rng):
    """x1 (N, n1), x2 (N, n2): AUC of class 2, selectivity, one-tailed permutation p (roc_utils_new convention)"""
    n1, n2 = x1.shape[1], x2.shape[1]
    r = rankdata(np.concatenate([x1, x2], 1), axis=1)
    auc = (r[:, n1:].sum(1) - n2 * (n2 + 1) / 2) / (n1 * n2)
    mem = np.zeros((n1 + n2, N_PERM))
    for b in range(N_PERM):
        mem[rng.permutation(n1 + n2)[:n2], b] = 1
    null = (r @ mem - n2 * (n2 + 1) / 2) / (n1 * n2)
    sel = 2 * (auc - 0.5)
    p = np.where(sel >= 0, (null >= auc[:, None]).mean(1), (null <= auc[:, None]).mean(1))
    return auc, sel, p


def learning_split_time(mouse, day, whisker_starts):
    f = RES / mouse / f"whisker_{day}" / "learning_curve" / f"{mouse}_whisker_{day}_whisker_trial_learning_curve.h5"
    if f.exists():
        lt = pd.read_hdf(f).iloc[0].learning_trial
        if np.isfinite(lt) and 0 < int(lt) < len(whisker_starts):
            return float(whisker_starts[int(lt)]), f"learning_trial={int(lt)}"
    return None, "median split (no learning trial)"


def run_session(session_id, day, rng):
    mouse = session_id.split("_")[0]
    nwb = NWBHDF5IO(str(NWB / f"{session_id}.nwb"), "r").read()
    units, _ = ru.process_nwb_tables(nwb)                       # artifact-corrected spike trains
    trials = nwb.trials.to_dataframe()
    ctx = trials["context"].astype(str).str.strip().str.lower()
    # sessions without passive blocks store context as the STRING 'nan' -> all trials are active
    # (process_nwb_tables only catches real NaN and would label them passive_post)
    trials["context"] = "active" if ctx.isin(["nan", "none", ""]).all() else ctx
    spikes = [np.sort(np.asarray(s)) for s in units["spike_times"]]
    pz = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:]))
    d = np.diff(pz); ili = float(np.median(d[(d > 0.02) & (d < 0.3)]))
    act = trials[trials["context"] == "active"].copy()
    act["L1"] = act["start_time"] + act["lick_time"] - act["response_window_start_time"]
    wh = act[act["whisker_stim"] == 1]
    cohort = "R+" if wh["reward_available"].mean() > 0.5 else "R-"
    lick = act["lick_flag"] == 1
    whisker_hits = act[(act["whisker_stim"] == 1) & lick]
    aud_hits = act[(act["auditory_stim"] == 1) & lick]
    fas = act[(act["no_stim"] == 1) & lick]
    t_split, split_how = learning_split_time(mouse, day, np.sort(wh["start_time"].to_numpy()))
    if t_split is None and len(whisker_hits):
        t_split = float(np.median(whisker_hits["start_time"]))

    def outcome(tr, sub_baseline=False):
        l1 = tr["L1"].to_numpy(float); st0 = tr["start_time"].to_numpy(float)
        o = np.stack([rates(s, l1 + ili, l1 + ili + OUT_W) for s in spikes])
        if sub_baseline:
            o = o - np.stack([rates(s, st0 + BASE[0], st0 + BASE[1]) for s in spikes])
        return o

    def baseline(tr):
        st0 = tr["start_time"].to_numpy(float)
        return np.stack([rates(s, st0 + BASE[0], st0 + BASE[1]) for s in spikes])

    rew = pd.concat([aud_hits] + ([whisker_hits] if cohort == "R+" else []))
    early_w = whisker_hits[whisker_hits["start_time"] < t_split] if t_split is not None else whisker_hits.iloc[:0]
    late_w = whisker_hits[whisker_hits["start_time"] >= t_split] if t_split is not None else whisker_hits.iloc[:0]
    a_med = float(np.median(aud_hits["start_time"])) if len(aud_hits) else np.inf
    comps = {
        "reward_vs_baseline": (lambda: baseline(rew), lambda: outcome(rew), +1, len(rew), len(rew)),
        "reward_vs_baseline_auditory": (lambda: baseline(aud_hits), lambda: outcome(aud_hits), +1, len(aud_hits), len(aud_hits)),
        "early_vs_late_auditory": (lambda: outcome(aud_hits[aud_hits.start_time >= a_med], True),
                                   lambda: outcome(aud_hits[aud_hits.start_time < a_med], True), +1,
                                   int((aud_hits.start_time >= a_med).sum()), int((aud_hits.start_time < a_med).sum())),
        "fa_vs_baseline": (lambda: baseline(fas), lambda: outcome(fas), -1, len(fas), len(fas)),
    }
    if cohort == "R+":
        comps.update({
            "reward_vs_baseline_whisker": (lambda: baseline(whisker_hits), lambda: outcome(whisker_hits), +1,
                                           len(whisker_hits), len(whisker_hits)),
            "early_vs_late_whisker": (lambda: outcome(late_w, True), lambda: outcome(early_w, True), +1,
                                      len(late_w), len(early_w)),
            "unexpected_vs_expected": (lambda: outcome(aud_hits, True), lambda: outcome(early_w, True), +1,
                                       len(aud_hits), len(early_w)),
            "unexpected_vs_expected_all": (lambda: outcome(aud_hits, True), lambda: outcome(whisker_hits, True), +1,
                                           len(aud_hits), len(whisker_hits)),
            "unrewarded_lick_vs_baseline": (lambda: baseline(fas), lambda: outcome(fas), -1, len(fas), len(fas)),
        })
    else:
        unrew = pd.concat([fas, whisker_hits])
        comps.update({
            "unrewarded_lick_vs_baseline": (lambda: baseline(unrew), lambda: outcome(unrew), -1, len(unrew), len(unrew)),
            "rminus_whisker_hit_vs_baseline": (lambda: baseline(whisker_hits), lambda: outcome(whisker_hits), -1,
                                               len(whisker_hits), len(whisker_hits)),
        })
    keys = units[["electrode_group", "cluster_id", "neuron_id"]].copy()
    keys.insert(0, "session_id", session_id); keys.insert(0, "mouse_id", mouse)
    rows = []
    for name, (f1, f2, pred, n1, n2) in comps.items():
        k = keys.copy()
        k["cohort"], k["comparison"], k["n1"], k["n2"], k["rpe_prediction"] = cohort, name, n1, n2, pred
        if min(n1, n2) >= MIN_TRIALS:
            auc, sel, p = roc_perm(f1(), f2(), rng)
            k["auc"], k["selectivity"], k["p_value"] = auc, sel, p
            k["significant"] = p < ALPHA
        else:
            k["auc"] = k["selectivity"] = k["p_value"] = np.nan
            k["significant"] = False
        k["rpe_consistent"] = k["significant"] & (np.sign(k["selectivity"]) == pred)
        rows.append(k)
    df = pd.concat(rows, ignore_index=True)
    meta = dict(session_id=session_id, day=day, cohort=cohort, ili_s=ili, outcome_window="[L1+ILI, L1+ILI+0.15] s",
                baseline_window_s=BASE, whisker_early_late_split=split_how, n_whisker_hits=len(whisker_hits),
                n_auditory_hits=len(aud_hits), n_false_alarms=len(fas), n_early_whisker=len(early_w),
                n_late_whisker=len(late_w), n_perm=N_PERM, alpha=ALPHA, min_trials=MIN_TRIALS, measure="rates (spikes/s)",
                lick_time="corrected: start_time + lick_time - response_window_start_time")
    return df, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", type=int, default=None, help="default: all days, day 0 first")
    a = ap.parse_args()
    files = sorted(glob.glob(str(RES / "*" / "whisker_*" / "roc_analysis" / "*_roc_results_new.csv")))
    todo = []
    for f in files:
        f = pathlib.Path(f)
        dname = f.parent.parent.name.split("_")[1]
        if not dname.lstrip("-+").isdigit():
            continue
        day = int(dname)
        if a.days is not None and day not in a.days:
            continue
        sid = pd.read_csv(f, usecols=["session_id"], nrows=1).session_id.iloc[0]
        todo.append((abs(day), day, sid, f.parent.parent))
    todo.sort()
    print(f"[027] {len(todo)} sessions", flush=True)
    rng = np.random.default_rng(SEED)
    for i, (_, day, sid, dayf) in enumerate(todo):
        out = dayf / "roc_analysis_rpe"
        mouse = sid.split("_")[0]
        if (out / f"{mouse}_roc_rpe.csv").exists():
            print(f"[027] {i + 1}/{len(todo)} {sid}: exists, skip", flush=True)
            continue
        t0 = time.time()
        try:
            df, meta = run_session(sid, day, rng)
            out.mkdir(exist_ok=True)
            df.to_csv(out / f"{mouse}_roc_rpe.csv", index=False)
            json.dump(meta, open(out / f"{mouse}_roc_rpe_config.json", "w"), indent=2)
            frac = df.groupby("comparison").rpe_consistent.mean().round(3).to_dict()
            print(f"[027] {i + 1}/{len(todo)} {sid} ({meta['cohort']}, day {day}, {meta['whisker_early_late_split']}): "
                  f"{(time.time() - t0) / 60:.1f} min; RPE-consistent fractions {frac}", flush=True)
        except Exception as e:
            print(f"[027] {i + 1}/{len(todo)} {sid}: FAILED {type(e).__name__}: {e}", flush=True)
    print("ALL DONE", flush=True)
