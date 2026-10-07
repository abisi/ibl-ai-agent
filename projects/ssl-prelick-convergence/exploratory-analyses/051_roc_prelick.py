"""Pre-lick ROC types (rate-based, active trials), run separately from the main ROC table (026 / roc_utils_new).

Trials (NWB trials table; skills ssl-trial-exclusion, ssl-lick-alignment):
  active context (per-trial rule resolve_context: fixed ~3 s ITI = passive, else the context column; unlabelled = active), perf != 6, auditory warm-up cut (keep 1 trial
  before the first whisker trial), end-of-session disengagement rule A1 (tail after the last lick dropped if it holds
  >= 5 whisker and >= 1 auditory trials).
  Classes: whisker hit (whisker_trial, lick_flag 1), auditory hit (auditory_trial, lick_flag 1), false alarm (FA:
  no_stim_trial, lick_flag 1); lick_time not NaN.
  First lick (corrected): first_lick_time = start_time + (lick_time - response_window_start_time).
Rate: spikes in [first_lick - 100 ms, first_lick) / 0.1 s minus the trial's baseline rate in [start_time - 1.0 s,
  start_time - 0.015 s] (roc_utils_new.BASELINE_WINDOW; same single-trial baseline correction as the main ROC).
  Spike trains whisker-artefact corrected by roc_utils_new.process_nwb_tables (same as the main ROC).
Two-class ROCs (AUC, selectivity = 2 AUC - 1, positive = class 2 higher; 1000 label permutations; one-tailed p on the
  side of the observed selectivity, significant if p < 0.05 -- identical to roc_utils_new.process_unit):
  wh_vs_aud_hit_prelick        class 1 whisker hit, class 2 auditory hit (positive = auditory > whisker, as wh_vs_aud_*)
  whisker_hit_vs_fa_prelick    class 1 FA, class 2 whisker hit (positive = whisker hit > FA)
  auditory_hit_vs_fa_prelick   class 1 FA, class 2 auditory hit (positive = auditory hit > FA)
Three-class ROC (three_class_prelick): D3 = mean over the 3 class pairs of |2 AUC_pair - 1| (class-balanced, unsigned
  pairwise discriminability, 0..1); null from 1000 joint permutations of the 3 labels; significant if
  P(D3_null >= D3) < 0.05. Also stored: the 3 signed pairwise selectivities, one-vs-rest selectivities per class and
  the preferred class (largest one-vs-rest selectivity).
Minimum trials: N_MIN per class (else NaN for that analysis). Minimum firing: a unit is tested in an analysis only if
  its mean raw rate in the pre-lick window over the analysed trials (classes pooled, label-independent) >= MIN_FR.
Variants (column `variant`): all (all trials); short_rt / long_rt (RT < / >= RT_SPLIT = 250 ms: with a short RT the
  pre-lick window starts < 150 ms after stimulus onset and overlaps the stimulus-evoked response; FA RT is counted from
  the no-stim response window, as for hits); rt_matched (per analysis, trials subsampled so every class has the same
  RT histogram in 50 ms bins; one draw, seed 0).
All units are processed (no quality filter); quality is joined downstream.
Output per session: <mouse>/whisker_<day>/roc_analysis/<mouse>_roc_prelick_results.csv (+ _roc_prelick_config.json);
  _roc_prelick_trials.npz: per-trial rates (baseline-corrected and raw) and first-lick PSTHs.
"""
import argparse
import importlib
import json
import os
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
from roc_analysis import roc_utils_new as ru                             # noqa: E402

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
# Unrewarded-lick reference (env PRELICK_REF): "fa" = false alarms (default); "sl" = spontaneous licks, which take the
# place of the FA class (label kept as "FA" in arrays so downstream code is unchanged; outputs go to separate files /
# folders tagged "_sl"). Spontaneous lick = bout-onset piezo lick (>= SL_GAP s after the previous lick), outside every
# trial window [start_time - 0.2 s, response_window_stop_time + 0.5 s], within the analysed active epoch (after the
# warm-up cut and A1 trim). Its baseline is the rate in SL_BASE before the lick (no trial start); trial classes keep
# the pre-trial baseline. Piezo and trial clocks are treated as in sync (user, 2026-10-02). RT variants are not
# defined for spontaneous licks -> only the "all" variant is run.
REF = os.environ.get("PRELICK_REF", "fa")
TAG = "" if REF == "fa" else f"_{REF}"
HOME = RES / "ssl-prelick-convergence"   # project results home (2026-10-07; old _roc_prelick{,_sl} / _within_day_sl are symlinks)
OUTROOT = HOME / "across_days" / REF      # across-day analyses (Part I), per reference (sl / fa)
WITHIN = HOME / "within_day" / REF        # within-day analyses (Part II), per reference
SL_GAP = 1.0
THREE_CLASS = False         # three-class ROC removed from the analyses (user 2026-10-03)
SL_BASE = (-1.0, -0.5)
PRELICK = (-0.100, 0.0)
BASE = ru.BASELINE_WINDOW
N_PERM = ru.N_PERMUTATIONS
ALPHA = ru.ALPHA
N_MIN = 3
CLASSES = ["WH", "AH", "FA"]
TWO_CLASS = {"wh_vs_aud_hit_prelick": ("WH", "AH"), "whisker_hit_vs_fa_prelick": ("FA", "WH"),
             "auditory_hit_vs_fa_prelick": ("FA", "AH")}
PAIRS = [("WH", "AH"), ("FA", "WH"), ("FA", "AH")]
PSTH_WIN, PSTH_BIN = (-0.6, 0.4), 0.010


# ------------------------------------------------------------------ trials
# Exception to the warm-up cut (user 2026-10-03): for these sessions the last k auditory hits of the warm-up block (before
# the first whisker trial) are kept, so that the session has >= 4 auditory hits in each half for the within-session
# analyses (ssl-within-day-remapping). MH065 R- expert day +1: 3 early-half auditory hits without the exception.
WARMUP_KEEP_AH = {"MH065_20260115_163926": 1}


def resolve_context(t):
    """per-trial context (user, 2026-10-04): a trial in a fixed ~3 s ITI sequence (gap to the previous or next trial
    3.0 +- 0.3 s) is passive; otherwise the context column decides (passive stays passive; active and unlabelled "nan"
    are active; perf == 6 trials inside active blocks stay active and are dropped by the perf rule). Changes vs the
    labels: MH062_20260113 (377 unlabelled training trials -> active), MH064_20260114 (21 -> passive), AB128_20240829
    (2 -> passive). t sorted by start_time."""
    st = t["start_time"].to_numpy()
    dp, dn = np.r_[np.inf, np.diff(st)], np.r_[np.diff(st), np.inf]
    fixed = (np.abs(dp - 3.0) < 0.3) | (np.abs(dn - 3.0) < 0.3)
    lab = t["context"].astype(str).to_numpy()
    return pd.Series(np.where(fixed | (lab == "passive"), "passive", "active"), index=t.index)


def select_trials(trials, sid=None):
    """active (resolve_context), perf != 6, warm-up cut (with the WARMUP_KEEP_AH exception), rule A1; returns
    (classified trials, log)"""
    t, log = active_trials(trials, sid)
    return classify(t, log)


def active_trials(trials, sid=None):
    """all trials of the analysed task epoch (any type / outcome): active (resolve_context), perf != 6, warm-up cut
    (with the WARMUP_KEEP_AH exception), rule A1; returns (trials, log). Split out of select_trials (2026-10-07) so that
    behaviour (hit / FA rates) is computed on exactly the analysed trials."""
    t = trials.sort_values("start_time").reset_index(drop=True)
    log = dict(n_all=len(t))
    ctx = resolve_context(t)
    if (ctx == "active").any():
        t = t[ctx == "active"].reset_index(drop=True)
    log["n_active"] = len(t)
    if "perf" in t:
        t = t[t["perf"] != 6].reset_index(drop=True)
    log["n_perf_ok"] = len(t)
    wi = np.where(t["trial_type"].to_numpy() == "whisker_trial")[0]
    if len(wi):
        warm = t.iloc[:max(0, wi[0] - 1)]
        t = t.iloc[max(0, wi[0] - 1):].reset_index(drop=True)
        k = WARMUP_KEEP_AH.get(sid, 0)
        if k:
            keep = warm[(warm.trial_type == "auditory_trial") & (warm.lick_flag == 1)].tail(k)
            t = pd.concat([keep, t]).sort_values("start_time").reset_index(drop=True)
            log["warmup_auditory_hits_kept"] = int(len(keep))
    log["n_after_warmup_cut"] = len(t)
    licked = np.where(t["lick_flag"].to_numpy() == 1)[0]
    log["a1_trimmed"] = 0
    if len(licked):
        tail = t.iloc[licked[-1] + 1:]
        if (tail.trial_type == "whisker_trial").sum() >= 5 and (tail.trial_type == "auditory_trial").sum() >= 1:
            t = t.iloc[:licked[-1] + 1].reset_index(drop=True)
            log["a1_trimmed"] = len(tail)
    log["epoch"] = (float(t.start_time.min()), float(t.stop_time.max())) if len(t) else (np.nan, np.nan)
    return t, log


def classify(t, log):
    """events of the analysed trials: corrected first lick, pre-trial baseline window, class WH / AH / FA"""
    t = t.copy()
    t["reaction_time"] = t["lick_time"] - t["response_window_start_time"]
    t["first_lick_time"] = t["start_time"] + t["reaction_time"]
    t["base_lo"], t["base_hi"] = t.start_time + BASE[0], t.start_time + BASE[1]
    ok = (t["lick_flag"] == 1) & t["first_lick_time"].notna()
    cls = np.select([ok & (t.trial_type == "whisker_trial"), ok & (t.trial_type == "auditory_trial"),
                     ok & (t.trial_type == "no_stim_trial")], CLASSES, default="")
    t["cls"] = cls
    t = t[t.cls != ""].reset_index(drop=True)
    log.update({f"n_{c}": int((t.cls == c).sum()) for c in CLASSES})
    log.update({f"rt_median_{c}": float(t.reaction_time[t.cls == c].median()) for c in CLASSES})
    log["frac_WH_rt_lt_100ms"] = float((t.reaction_time[t.cls == "WH"] < 0.1).mean()) if (t.cls == "WH").any() else np.nan
    return t, log


# ------------------------------------------------------------------ rates
def spontaneous_licks(nwb, trials_raw, epoch):
    """bout-onset piezo licks outside all trial windows, within the analysed active epoch"""
    lk = np.sort(np.asarray(nwb.processing["behavior"].data_interfaces["BehavioralEvents"]
                            .time_series["piezo_lick_times"].data[:], float))
    if len(lk) == 0:
        return lk
    onset = lk[np.r_[True, np.diff(lk) >= SL_GAP]]
    lo = trials_raw.start_time.to_numpy() - 0.2
    hi = trials_raw.response_window_stop_time.to_numpy() + 0.5
    inside = np.zeros(len(onset), bool)
    for a, b in zip(lo, hi):
        inside |= (onset >= a) & (onset <= b)
    keep = ~inside & (onset >= epoch[0]) & (onset <= epoch[1])
    return onset[keep]


def unit_rates(spikes, t):
    """(n_units, n_trials) raw pre-lick window rate and baseline-corrected pre-lick rate (spikes/s); per-event baseline
    window [base_lo, base_hi] (absolute times)"""
    fl = t.first_lick_time.to_numpy()
    blo, bhi = t.base_lo.to_numpy(), t.base_hi.to_numpy()
    W = np.zeros((len(spikes), len(t))); B = np.zeros_like(W)
    dw, db = PRELICK[1] - PRELICK[0], bhi - blo
    for u, s in enumerate(spikes):
        W[u] = (np.searchsorted(s, fl + PRELICK[1]) - np.searchsorted(s, fl + PRELICK[0])) / dw
        B[u] = (np.searchsorted(s, bhi) - np.searchsorted(s, blo)) / db
    return W, W - B


def lick_psth(spikes, t):
    edges = np.arange(PSTH_WIN[0], PSTH_WIN[1] + PSTH_BIN / 2, PSTH_BIN)
    fl = t.first_lick_time.to_numpy()
    out = np.zeros((len(spikes), len(CLASSES), len(edges) - 1), np.float32)
    for u, s in enumerate(spikes):
        for k, c in enumerate(CLASSES):
            ev = fl[t.cls.to_numpy() == c]
            if len(ev) == 0:
                continue
            lo, hi = np.searchsorted(s, ev + PSTH_WIN[0]), np.searchsorted(s, ev + PSTH_WIN[1])
            rel = np.concatenate([s[a:b] - e for a, b, e in zip(lo, hi, ev)]) if len(ev) else np.array([])
            out[u, k] = np.histogram(rel, edges)[0] / len(ev) / PSTH_BIN
    return out, edges


# ------------------------------------------------------------------ ROC statistics
def auc_two(X, y, rng):
    """X (U, n) values, y (n,) 0/1 (class 2 = 1). Returns auc (U,), null aucs (U, P). AUC = Mann-Whitney U/(n1 n2)
    (= sklearn roc_auc_score)"""
    Rk = rankdata(X, axis=1)
    n1, n2 = int((y == 0).sum()), int((y == 1).sum())
    obs = (Rk[:, y == 1].sum(1) - n2 * (n2 + 1) / 2) / (n1 * n2)
    Pm = np.stack([rng.permutation(y) for _ in range(N_PERM)]).astype(float)       # (P, n)
    null = (Rk @ Pm.T - n2 * (n2 + 1) / 2) / (n1 * n2)
    return obs, null


ROC_SHIFT_NULL = False      # single-neuron ROC uses label permutation for both references (user 2026-10-03)
N_SHIFT_ROC = 200          # linear shifts for the spontaneous-lick reference (|k| from MIN_SHIFT_ROC to n // 3)
MIN_SHIFT_ROC = 5


def auc_shift(Xall, lab, c1, c2):
    """Linear-shift null for autocorrelated events (spontaneous-lick reference). Xall (U, n) rates of ALL events in time
    order, lab (n,) labels. Observed AUC on the c1 / c2 events; for each shift k the label of event i is paired with the
    activity of event i + k on the overlap (no wrap-around) and the AUC recomputed. Returns auc (U,), null (U, S)."""
    def _auc(V, y):
        Rk = rankdata(V, axis=1); n1, n2 = int((y == 0).sum()), int((y == 1).sum())
        return (Rk[:, y == 1].sum(1) - n2 * (n2 + 1) / 2) / (n1 * n2)
    n = len(lab)
    m = np.isin(lab, [c1, c2]); obs = _auc(Xall[:, m], (lab[m] == c2).astype(int))
    kmax = n // 3
    ks = np.unique(np.r_[np.linspace(MIN_SHIFT_ROC, kmax, N_SHIFT_ROC // 2).astype(int),
                         -np.linspace(MIN_SHIFT_ROC, kmax, N_SHIFT_ROC // 2).astype(int)]) if kmax > MIN_SHIFT_ROC else []
    null = []
    for k in ks:
        i = np.arange(max(0, -k), min(n, n - k)); li = lab[i]; mm = np.isin(li, [c1, c2]); y = (li[mm] == c2).astype(int)
        if min(y.sum(), (1 - y).sum()) < N_MIN:
            continue
        null.append(_auc(Xall[:, i[mm] + k], y))
    return obs, (np.column_stack(null) if null else np.full((len(obs), 0), np.nan))


def pair_auc_matrix(G, A, B):
    """G (U, n, n): 1 if x_a > x_b, 0.5 tie; A (P, n), B (P, n) class indicators -> AUC(B > A) (U, P)"""
    na, nb = A.sum(1), B.sum(1)
    T = np.einsum("uab,pb->upa", G, A, optimize=True)                 # sum over b in A of [x_a > x_b]
    return np.einsum("upa,pa->up", T, B, optimize=True) / (na * nb)


def three_class(X, lab, rng, batch=100):
    """D3 = mean_pairs |2 AUC - 1| with joint 3-label permutation null; pairwise and one-vs-rest selectivities"""
    U, n = X.shape
    G = (X[:, :, None] > X[:, None, :]).astype(np.float32) + 0.5 * (X[:, :, None] == X[:, None, :]).astype(np.float32)
    ind = {c: (lab == c).astype(np.float32)[None] for c in CLASSES}
    pair_sel = {}
    for a, b in PAIRS:
        pair_sel[(a, b)] = 2 * pair_auc_matrix(G, ind[a], ind[b])[:, 0] - 1
    D3 = np.mean([np.abs(v) for v in pair_sel.values()], axis=0)
    null = np.zeros((U, N_PERM), np.float32)
    for s in range(0, N_PERM, batch):
        perm = np.stack([rng.permutation(lab) for _ in range(min(batch, N_PERM - s))])
        pi = {c: (perm == c).astype(np.float32) for c in CLASSES}
        acc = 0
        for a, b in PAIRS:
            acc = acc + np.abs(2 * pair_auc_matrix(G, pi[a], pi[b]) - 1)
        null[:, s:s + perm.shape[0]] = acc / len(PAIRS)
    p = (null >= D3[:, None] - 1e-12).mean(1)
    Rk = rankdata(X, axis=1)
    ovr = {}
    for c in CLASSES:
        m = lab == c; k = m.sum()
        ovr[c] = 2 * ((Rk[:, m].sum(1) - k * (k + 1) / 2) / (k * (n - k))) - 1
    return D3, p, pair_sel, ovr, null


# ------------------------------------------------------------------ variants
VARIANTS = ["all", "short_rt", "long_rt", "rt_matched"]
RT_SPLIT = 0.250            # s; short RT: the pre-lick window [RT-100, RT] starts < 150 ms after stimulus onset
RT_BIN = 0.050              # s; RT matching bins
MIN_FR = 0.1                # Hz; mean raw pre-lick window rate over the analysed trials (pooled across classes)
                            # (user 2026-10-02; was 1.0 Hz in the first run)


def trial_set(lab, rt, classes, variant, rng):
    """indices of the trials of `classes` used by `variant`"""
    idx = np.where(np.isin(lab, classes))[0]
    if variant == "short_rt":
        idx = idx[rt[idx] < RT_SPLIT]
    elif variant == "long_rt":
        idx = idx[rt[idx] >= RT_SPLIT]
    elif variant == "rt_matched":                                     # same RT histogram (50 ms bins) in every class
        bins = np.floor(rt[idx] / RT_BIN).astype(int)
        keep = []
        for b in np.unique(bins):
            mem = [idx[(bins == b) & (lab[idx] == c)] for c in classes]
            n = min(len(m) for m in mem)
            if n:
                keep += [rng.choice(m, n, replace=False) for m in mem]
        idx = np.sort(np.concatenate(keep)) if keep else np.array([], int)
    return idx


# ------------------------------------------------------------------ session
def run_session(sid, out_dir, mouse, save_trials=True, seed=0):
    from pynwb import NWBHDF5IO
    rng = np.random.default_rng(seed)
    with NWBHDF5IO(str(NWB / f"{sid}.nwb"), "r", load_namespaces=True) as io:
        nwb = io.read()
        units, _ = ru.process_nwb_tables(nwb)                       # artefact-corrected spike trains (main ROC)
        trials_raw = nwb.trials.to_dataframe()
        t, log = select_trials(trials_raw, sid)
        if REF == "sl":                                             # spontaneous licks replace false alarms
            sl = spontaneous_licks(nwb, trials_raw, log["epoch"])
            t = t[t.cls != "FA"]
            t = pd.concat([t, pd.DataFrame(dict(first_lick_time=sl, start_time=sl, reaction_time=np.nan, cls="FA",
                                                base_lo=sl + SL_BASE[0], base_hi=sl + SL_BASE[1]))],
                          ignore_index=True)
            log["n_FA_trials_replaced"] = log["n_FA"]; log["n_FA"] = int(len(sl)); log["n_SL"] = int(len(sl))
            t = t.sort_values("start_time").reset_index(drop=True)   # time order (needed by the linear-shift null)
    spikes = [np.sort(np.asarray(s)) for s in units.spike_times]
    Wraw, X = unit_rates(spikes, t)
    lab, rt = t.cls.to_numpy(), t.reaction_time.to_numpy()
    meta_cols = ["electrode_group", "cluster_id", "neuron_id", "firing_rate", "target_region"] + \
        [c for c in units.columns if "ccf" in c]
    meta = units[meta_cols].reset_index(drop=True).assign(mouse_id=mouse, session_id=sid)
    rows = []
    for variant in (["all"] if REF == "sl" else VARIANTS):
        for name, (c1, c2) in TWO_CLASS.items():
            idx = trial_set(lab, rt, [c1, c2], variant, rng)
            y = (lab[idx] == c2).astype(int)
            n1, n2 = int((y == 0).sum()), int((y == 1).sum())
            fr = Wraw[:, idx].mean(1) if len(idx) else np.zeros(len(meta))
            r = meta.copy(); r["variant"] = variant; r["analysis_type"] = name
            r["n_class1"], r["n_class2"], r["fr_window"] = n1, n2, fr
            r["tested"] = (fr >= MIN_FR) & (n1 >= N_MIN) & (n2 >= N_MIN)
            r["auc"] = np.nan; r["selectivity"] = np.nan; r["p_value"] = np.nan; r["significant"] = False
            r["direction"] = None
            tu = r["tested"].to_numpy()
            if tu.any():
                if REF == "sl" and ROC_SHIFT_NULL:                  # optional linear-shift null (off)
                    auc, null = auc_shift(X[tu], lab, c1, c2)
                else:                                               # randomised trials: label permutation
                    auc, null = auc_two(X[tu][:, idx], y, rng)
                sel = 2 * auc - 1
                if null.shape[1] < 20:
                    p = np.full(len(auc), np.nan)
                else:
                    p = np.where(sel >= 0, (1 + (null >= auc[:, None]).sum(1)) / (1 + null.shape[1]),
                                 (1 + (null <= auc[:, None]).sum(1)) / (1 + null.shape[1]))
                sig = p < ALPHA
                dirs = ("auditory", "whisker") if "wh_vs_aud" in name else ("positive", "negative")
                r.loc[tu, "auc"], r.loc[tu, "selectivity"], r.loc[tu, "p_value"] = auc, sel, p
                r.loc[tu, "significant"] = sig
                r.loc[tu, "direction"] = np.where(sig, np.where(sel > 0, dirs[0], dirs[1]), None)
            rows.append(r)
        if not THREE_CLASS:                                          # removed (user 2026-10-03)
            continue
        idx = trial_set(lab, rt, CLASSES, variant, rng)
        l3 = lab[idx]
        fr = Wraw[:, idx].mean(1) if len(idx) else np.zeros(len(meta))
        r = meta.copy(); r["variant"] = variant; r["analysis_type"] = "three_class_prelick"; r["fr_window"] = fr
        for c in CLASSES:
            r[f"n_{c}"] = int((l3 == c).sum())
        r["tested"] = (fr >= MIN_FR) & all((l3 == c).sum() >= N_MIN for c in CLASSES)
        r["auc"] = np.nan; r["selectivity"] = np.nan; r["p_value"] = np.nan; r["significant"] = False
        r["direction"] = None; r["preferred_class"] = None
        tu = r["tested"].to_numpy()
        if tu.any():
            D3, p, pair_sel, ovr, _ = three_class(X[tu][:, idx], l3, rng)
            r.loc[tu, "selectivity"], r.loc[tu, "p_value"], r.loc[tu, "significant"] = D3, p, p < ALPHA
            for (a, b), v in pair_sel.items():
                r.loc[tu, f"pair_sel_{b}_vs_{a}"] = v
            for c, v in ovr.items():
                r.loc[tu, f"ovr_sel_{c}"] = v
            pref = np.array(CLASSES)[np.argmax(np.column_stack([ovr[c] for c in CLASSES]), 1)]
            r.loc[tu, "preferred_class"] = pref
            r.loc[tu, "direction"] = np.where(p < ALPHA, pref, None)
        rows.append(r)
        log[f"{variant}_n3"] = {c: int((l3 == c).sum()) for c in CLASSES}
    res = pd.concat(rows, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{mouse}_roc_prelick{TAG}_results.tmp.csv"
    res.to_csv(tmp, index=False); tmp.replace(out_dir / f"{mouse}_roc_prelick{TAG}_results.csv")
    cfg = dict(session_id=sid, script="051_roc_prelick.py", prelick_window_s=PRELICK, baseline_window_s=BASE,
               n_permutations=N_PERM, alpha=ALPHA, n_min_per_class=N_MIN, seed=seed, variants=VARIANTS,
               rt_split_s=RT_SPLIT, rt_match_bin_s=RT_BIN, min_fr_hz=MIN_FR,
               min_fr_definition="mean raw rate in the pre-lick window over the analysed trials (classes pooled)",
               lick_definition="first_lick_time = start_time + (lick_time - response_window_start_time)",
               trial_exclusions="active; perf != 6; warm-up cut (keep 1 trial before first whisker); rule A1",
               spike_preprocessing="roc_utils_new.process_nwb_tables (whisker artefact correction)",
               units="all units (no quality filter)", two_class=TWO_CLASS,
               three_class="D3 = mean |2 AUC_pair - 1| over (WH,AH),(FA,WH),(FA,AH); joint label permutation",
               unrewarded_lick_reference=REF,
               roc_null=(f"linear shift ({N_SHIFT_ROC} shifts, |k| {MIN_SHIFT_ROC}..n//3)" if (REF == "sl" and ROC_SHIFT_NULL) else f"label permutation ({N_PERM})"),
               spontaneous_lick=dict(gap_s=SL_GAP, baseline_s=SL_BASE,
               exclusion="outside [start - 0.2, response_window_stop + 0.5] of every trial; within analysed active epoch",
               clock="piezo lick times, treated as in sync with trial licks") if REF == "sl" else None,
               trial_log=log, created=time.strftime("%Y-%m-%d %H:%M"))
    (out_dir / f"{mouse}_roc_prelick{TAG}_config.json").write_text(json.dumps(cfg, indent=1, default=str))
    if save_trials:
        P, edges = lick_psth(spikes, t)
        np.savez_compressed(out_dir / f"{mouse}_roc_prelick{TAG}_trials.npz", rates=X.astype(np.float32),
                            raw=Wraw.astype(np.float32), cls=lab, rt=rt, trial_start=t.start_time.to_numpy(),
                            electrode_group=units.electrode_group.astype(str).to_numpy(),
                            cluster_id=units.cluster_id.astype(str).to_numpy(), psth=P, psth_edges=edges)
    return res, log


def _job(args):
    sid, out, mouse, day = args
    t0 = time.time()
    try:
        res, log = run_session(sid, out, mouse)
        a = res[res.variant == "all"]
        s = (a.groupby("analysis_type").significant.sum() / a.groupby("analysis_type").tested.sum()).round(3).to_dict()
        return (f"{sid} day {day}: {log['n_WH']} WH / {log['n_AH']} AH / {log['n_FA']} FA; "
                f"frac sig (all trials, tested units) {s} ({time.time() - t0:.0f} s)")
    except Exception as e:                                             # noqa: BLE001
        return f"{sid} FAILED: {e!r}"


def main(a):
    import multiprocessing
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions()
    if a.sessions:
        ss = ss[ss.session_id.isin(a.sessions)]
    jobs = [(r.session_id, r.file.parent, r.mouse, r.day) for r in ss.itertuples()
            if not (a.skip_done and (r.file.parent / f"{r.mouse}_roc_prelick{TAG}_config.json").exists()
                    and "variants" in json.loads((r.file.parent / f"{r.mouse}_roc_prelick{TAG}_config.json").read_text()))]
    print(f"[051] {len(jobs)} sessions to run ({len(ss)} listed), {a.n_proc} processes", flush=True)
    with multiprocessing.Pool(a.n_proc, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(_job, jobs)):
            print(f"[051] {i + 1}/{len(jobs)} {msg}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--n-proc", type=int, default=6)
    main(ap.parse_args())
