"""RPE ROC v2: ROC-based classification of positive/negative-RPE, reward, reward-cue, cue-decay and novelty neurons.
Specification: Axel Bisi, 2026-09-29 (see README written by `aggregate`). Builds on 027_roc_rpe.py (same loading, the
corrected first lick, rate helper); 027 outputs are left untouched.

Windows (firing RATES, spikes/s, per trial and neuron)
  baseline  before trial start (whisker/auditory stimulus onset == start_time), SAME trial (single-trial baseline, so
            slow baseline changes are accounted for), length-matched to the analysed window:
            [-110, -10] ms for cue analyses, [-280, -10] ms for outcome analyses
  units     all units computed; quality_label (classify_units_quality incl. DREDge drift test, presence/coverage ratio)
            stored, aggregate keeps quality_label == 'good' by default
  cue       [50, 150] ms after stimulus onset
  outcome   [30, 300] ms after the corrected first lick L1 = start_time + lick_time - response_window_start_time
            (same window for unrewarded R- whisker licks; includes the pre-reward period and the lick itself)
  Whisker magnetic-artifact correction (roc_utils_new.process_nwb_tables): spikes in [-10, +5] ms around every whisker
  onset are replaced by Poisson spikes at the [-200, -10] ms rate; none of the windows above overlaps [-10, +5] ms.
  "set vs baseline": window rates of the set's trials vs the same trials' baseline rates.
  "set vs set": baseline-subtracted rates (window - same-trial baseline).
Trial sets (active trials; catch/no-stim trials never used)
  W_all    all whisker trials (hits + misses), thirds T1/T2/T3 by within-session order
  W_first  the first N_FIRST whisker trials of the session (licked or not)
  W_next   whisker trials after W_first up to the end of W_all T1 (early-session reference for the novelty decrement)
  W_lick   whisker trials with a lick (R+: rewarded hits; R-: unrewarded whisker licks), thirds
  A_hit    auditory hits, HALVES H1/H2 (thirds too small: day-0 R+ median 8 hits per third); the auditory warm-up block is
           KEPT (deliberate exception to the ssl-analyze warm-up trim rule), only the first N_AUD_SKIP auditory trials of
           the session are excluded
  A_all    all auditory trials (same exclusion), halves -- cue reference for the population plot
Comparisons (AUC = P(first set > second set) + 0.5 P(tie); AUC > 0.5 = first set higher)
  O1 R+  W_lick T1 outcome vs baseline                     pred +
  O2 R+  W_lick T1 vs W_lick T3 (outcome)                  pred +
  O3 R+  W_lick T1 vs A_hit H1 (outcome)                   pred +
  O4 R-  W_lick T1 outcome vs baseline                     pred -
  O5 R-  W_lick T1 vs W_lick T3 (outcome)                  pred -
  O6 all A_hit H1 vs A_hit H2 (outcome)                    pred 0 (control)
  C1 all W_first cue vs baseline (cue response to the first presentations)          pred +
  C1b all W_first vs W_next (cue; novelty DECREMENT: first presentations > following)  pred +
  C2 R+  W_all T3 vs W_all T1 (cue)                        pred +
  C3 R-  W_all T3 vs W_all T1 (cue)                        pred -
  extras: O3tm (R+) W_lick T1 vs auditory hits in the same time span as W_lick T1; IX (all) interaction
  AUC(W_lick T1 vs T3) - AUC(A_hit H1 vs H2), pred + (R+) / - (R-).
Monotonicity: Spearman rho(baseline-subtracted rate, third index) for W_lick (outcome) and W_all (cue).
Novelty curve: per-trial cue response of the first N_CURVE whisker trials, z-scored per neuron by the SD of its
  baseline-subtracted cue response over all whisker trials (cuez_w01..).
Statistics: two-sided label-permutation p (N_PERM shuffles, (b+1)/(N+1)); asymptotic Mann-Whitney p also stored.
  NO multiple-comparison correction (comparison scheme not fixed yet): classes use uncorrected p < ALPHA in the predicted
  direction. CHANCE: every comparison is also run once on shuffled set labels through the identical pipeline (p_null)
  -> chance class proportions. Skipped if either set has < N_MIN trials (skip_<cmp>, skip_reason_<cmp>).
Usage: run [--days ...] [--sessions ...] -> <mouse>/whisker_<day>/roc_analysis_rpe_v2/<mouse>_roc_rpe_v2.csv
       aggregate -> combined_results_ks4/_roc_rpe_v2_summary/ ; examples -> .../examples/
"""
import argparse
import glob
import importlib
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
r27 = importlib.import_module("027_roc_rpe")
ru, RES, NWB = r27.ru, r27.RES, r27.NWB

BASE_W = (-0.28, -0.01)          # outcome baseline (270 ms, same length as the outcome window)
BASE_W_CUE = (-0.11, -0.01)      # cue baseline (100 ms, same length as the cue window: unequal window lengths bias the
                                 # AUC, e.g. pre-trial 100 ms vs 300 ms gives mean AUC 0.47 under the null)
CUE_W = (0.05, 0.15)
OUT_W = (0.03, 0.30)             # 30-300 ms after the very first (corrected) lick
N_MIN = 2
N_FIRST = 5
N_CURVE = 20
N_AUD_SKIP = 5
N_PERM = 1000
ALPHA = 0.05
SUBDIR = "roc_analysis_rpe_v2"
SUMMARY = RES / "_roc_rpe_v2_summary"
PARAMS = dict(baseline_window_s_outcome=BASE_W, baseline_window_s_cue=BASE_W_CUE, cue_window_s=CUE_W,
              unit_selection="aggregate: quality_label == 'good' (unit_metrics_utils.classify_units_quality: bombcell "
                             "metrics + presence/coverage ratio + DREDge motion shift test); all units computed", outcome_window_s_from_first_lick=OUT_W, n_min=N_MIN,
              n_first_whisker=N_FIRST, n_novelty_curve=N_CURVE, n_auditory_trials_skipped=N_AUD_SKIP,
              auditory_split="halves", whisker_split="thirds", n_perm=N_PERM, alpha=ALPHA,
              multiple_comparisons="none (uncorrected p; chance from shuffled-label run)",
              p_value="two-sided label permutation (b+1)/(N+1); p_mw = asymptotic Mann-Whitney",
              measure="firing rate (spikes/s)", units="all units (no QC filter)",
              artifact_correction="roc_utils_new: whisker onset [-10, +5] ms replaced by Poisson at [-200, -10] ms rate",
              lick_time="corrected first lick: start_time + lick_time - response_window_start_time")
COMPS = {   # name: (cohort, predicted sign, window, (set1, set2), kind)   kind: base = set vs its baseline, bs = set vs set
    "O1": ("R+", +1, "outcome", ("W_lick_T1", "W_lick_T1"), "base"),
    "O2": ("R+", +1, "outcome", ("W_lick_T1", "W_lick_T3"), "bs"),
    "O3": ("R+", +1, "outcome", ("W_lick_T1", "A_hit_H1"), "bs"),
    "O4": ("R-", -1, "outcome", ("W_lick_T1", "W_lick_T1"), "base"),
    "O5": ("R-", -1, "outcome", ("W_lick_T1", "W_lick_T3"), "bs"),
    "O6": ("both", 0, "outcome", ("A_hit_H1", "A_hit_H2"), "bs"),
    "C1": ("both", +1, "cue", ("W_first", "W_first"), "base"),
    "C1b": ("both", +1, "cue", ("W_first", "W_next"), "bs"),
    "C2": ("R+", +1, "cue", ("W_all_T3", "W_all_T1"), "bs"),
    "C3": ("R-", -1, "cue", ("W_all_T3", "W_all_T1"), "bs"),
    "O3tm": ("R+", +1, "outcome", ("W_lick_T1", "A_hit_tm"), "bs"),
    "IX": ("both", None, "outcome", ("W_lick_T1", "W_lick_T3"), "ix"),
}
CLASS_RULES = {
    "positive_RPE": ("R+", lambda s: s("O1", +1) & s("O2", +1) & s("O3", +1)),
    "negative_RPE": ("R-", lambda s: s("O4", -1) & s("O5", -1)),
    "reward": ("R+", lambda s: s("O1", +1) & ~s("O2", +1)),
    "reward_cue": ("R+", lambda s: s("C2", +1)),
    "cue_decay": ("R-", lambda s: s("C3", -1)),
    "novelty_response": ("both", lambda s: s("C1", +1)),
    "novelty_decrement": ("both", lambda s: s("C1b", +1)),
}


# ------------------------------------------------------------------ ROC machinery
def auc_first(x1, x2):
    """x1 (N, n1), x2 (N, n2) -> AUC = P(x1 > x2) + 0.5 ties, plus the pooled ranks (for permutations)"""
    n1, n2 = x1.shape[1], x2.shape[1]
    r = stats.rankdata(np.concatenate([x1, x2], 1), axis=1)
    return (r[:, :n1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n2), r


def perm_null(r, n1, n2, rng):
    mem = np.zeros((n1 + n2, N_PERM))
    for b in range(N_PERM):
        mem[rng.permutation(n1 + n2)[:n1], b] = 1
    return (r @ mem - n1 * (n1 + 1) / 2) / (n1 * n2)


def two_sided(obs, null, center=0.5):
    return (1 + (np.abs(null - center) >= np.abs(obs - center)[:, None] - 1e-12).sum(1)) / (N_PERM + 1)


def roc_test(x1, x2, rng):
    auc, r = auc_first(x1, x2)
    p = two_sided(auc, perm_null(r, x1.shape[1], x2.shape[1], rng))
    try:
        p_mw = stats.mannwhitneyu(x1, x2, axis=1, method="asymptotic").pvalue
    except ValueError:
        p_mw = np.full(len(auc), np.nan)
    return auc, p, np.where(np.isnan(p_mw), 1.0, p_mw)


def interaction_test(w1, w3, a1, a3, rng):
    aw, rw = auc_first(w1, w3)
    aa, ra = auc_first(a1, a3)
    null = perm_null(rw, w1.shape[1], w3.shape[1], rng) - perm_null(ra, a1.shape[1], a3.shape[1], rng)
    return aw - aa, two_sided(aw - aa, null, 0.0)


def spearman_rows(x, idx):
    """row-wise Spearman rho between x (N, n) and idx (n,), t-approximation p"""
    n = x.shape[1]
    rx = stats.rankdata(x, axis=1); ri = stats.rankdata(idx)
    rx = rx - rx.mean(1, keepdims=True); ri = ri - ri.mean()
    den = np.sqrt((rx ** 2).sum(1) * (ri ** 2).sum())
    rho = np.where(den > 0, (rx @ ri) / np.where(den > 0, den, 1), 0.0)
    t = rho * np.sqrt((n - 2) / np.maximum(1 - rho ** 2, 1e-12))
    return rho, 2 * stats.t.sf(np.abs(t), n - 2)


def split(tr, k):
    tr = tr.sort_values("start_time")
    return [tr.iloc[ix] for ix in np.array_split(np.arange(len(tr)), k)]


# ------------------------------------------------------------------ session
def region_columns(units):
    df = units[[c for c in units.columns if c.startswith("ccf") or c == "target_region"]].copy()
    try:
        import ephys_utilities.allen_utils.allen_utils as au
        df = au.create_area_custom_column(df)
        try:
            df = au.create_areas_subdivisions(df)
        except Exception:
            pass
        a2g = {a: g for g, acs in au.get_custom_area_groups().items() for a in acs}
        df["area_group"] = df["area_acronym_custom"].map(a2g).fillna("unassigned")
    except Exception as e:
        print("  region mapping failed:", e, flush=True)
        df["area_acronym_custom"], df["area_group"] = df.get("ccf_acronym", "?"), "unassigned"
    return df[["ccf_acronym", "area_acronym_custom", "area_group"]]


def quality_columns(nwb, units, session_id, mouse, day):
    """quality_label as in unit_spikes_analysis.py: DREDge motion shift test (drift_shift_test_pval = p_conservative,
    drift_abs_r = |r|) + presence/coverage ratio + unit_metrics_utils.classify_units_quality (bombcell thresholds)"""
    from ephys_utilities.neural_utils import unit_metrics_utils as umu
    raw = nwb.units.to_dataframe().loc[units.neuron_id.to_numpy()].copy()
    raw["session_id"] = session_id
    raw["electrode_group"] = units.electrode_group.astype(str).to_numpy()
    raw["cluster_id"] = units.cluster_id.astype(str).to_numpy()
    f = (RES / mouse / f"whisker_{day}" / "single_neuron_motion_shift_test"
         / f"{mouse}_whisker_{day}_motion_shift_test_results.csv")
    if f.exists():
        dd = pd.read_csv(f, usecols=["cluster_id", "electrode_group", "p_conservative", "r"])
        dd["cluster_id"] = dd.cluster_id.astype(str)
        dd = dd.rename(columns={"p_conservative": "drift_shift_test_pval"})
        dd["drift_abs_r"] = dd.pop("r").abs()
        raw = raw.reset_index(drop=True).merge(dd, on=["cluster_id", "electrode_group"], how="left", validate="one_to_one")
    else:
        raw = raw.reset_index(drop=True)
        raw["drift_shift_test_pval"], raw["drift_abs_r"] = np.nan, np.nan
    raw["drift_test_available"] = raw["drift_shift_test_pval"].notna()
    raw = umu.compute_presence_coverage_metrics(raw)
    raw = umu.classify_units_quality(raw, label_col="quality_label")
    return raw[["bc_label", "quality_label", "presence_ratio", "coverage_ratio", "drift_shift_test_pval", "drift_abs_r",
                "drift_test_available"]].reset_index(drop=True)


def load_session(session_id):
    nwb = NWBHDF5IO(str(NWB / f"{session_id}.nwb"), "r").read()
    units, _ = ru.process_nwb_tables(nwb)                        # artifact-corrected spike trains, all units
    units = units.reset_index(drop=True)
    trials = nwb.trials.to_dataframe()
    ctx = trials["context"].astype(str).str.strip().str.lower()
    trials["context"] = "active" if ctx.isin(["nan", "none", ""]).all() else ctx   # string-'nan' = no passive
    act = trials[trials["context"] == "active"].sort_values("start_time").copy()
    act["L1"] = act["start_time"] + act["lick_time"] - act["response_window_start_time"]
    act["aud_rank"] = np.nan
    a = act["auditory_stim"] == 1
    act.loc[a, "aud_rank"] = np.arange(a.sum())
    return nwb, units, act


def sets_for(act):
    W_all = act[act.whisker_stim == 1].sort_values("start_time")
    cohort = "R+" if W_all.reward_available.mean() > 0.5 else "R-"
    W_lick = W_all[W_all.lick_flag == 1]
    A_all = act[(act.auditory_stim == 1) & (act.aud_rank >= N_AUD_SKIP)]
    A_hit = A_all[A_all.lick_flag == 1]
    S = dict(W_all=W_all, W_lick=W_lick, A_all=A_all, A_hit=A_hit)
    for k in ["W_all", "W_lick"]:
        for i, t in enumerate(split(S[k], 3)):
            S[f"{k}_T{i + 1}"] = t
    for k in ["A_all", "A_hit"]:
        for i, t in enumerate(split(S[k], 2)):
            S[f"{k}_H{i + 1}"] = t
    S["W_first"] = W_all.iloc[:N_FIRST]
    S["W_next"] = W_all.iloc[N_FIRST:max(len(S["W_all_T1"]), N_FIRST)]
    t1 = S["W_lick_T1"]
    S["A_hit_tm"] = (A_hit[(A_hit.start_time >= t1.start_time.min()) & (A_hit.start_time <= t1.start_time.max())]
                     if len(t1) else A_hit.iloc[:0])
    return cohort, S


class Rates:
    def __init__(self, spikes):
        self.spikes = spikes

    def win(self, tr, which):
        ref = tr.L1 if which == "outcome" else tr.start_time
        w = {"baseline_outcome": BASE_W, "baseline_cue": BASE_W_CUE, "cue": CUE_W, "outcome": OUT_W}[which]
        t0 = ref.to_numpy(float)
        if len(t0) == 0:
            return np.zeros((len(self.spikes), 0))
        return np.stack([r27.rates(s, t0 + w[0], t0 + w[1]) for s in self.spikes])

    def bs(self, tr, which):                                    # baseline-subtracted (length-matched baseline)
        return self.win(tr, which) - self.win(tr, f"baseline_{which}")


def run_session(session_id, day, rng):
    mouse = session_id.split("_")[0]
    nwb, units, act = load_session(session_id)
    cohort, S = sets_for(act)
    R = Rates([np.sort(np.asarray(s)) for s in units["spike_times"]])
    out = units[["electrode_group", "cluster_id", "neuron_id"]].copy()
    out.insert(0, "session_id", session_id); out.insert(0, "mouse_id", mouse)
    out.insert(2, "day", day); out.insert(3, "cohort", cohort)
    out = pd.concat([out, region_columns(units), quality_columns(nwb, units, session_id, mouse, day)], axis=1)
    for k, v in S.items():
        out[f"n_{k}"] = len(v)
    N = len(out)
    for c, (coh, pred, win, (s1, s2), kind) in COMPS.items():
        applies = coh in ("both", cohort)
        n1, n2 = len(S[s1]), len(S[s2])
        need = [n1, n2] + ([len(S["A_hit_H1"]), len(S["A_hit_H2"])] if kind == "ix" else [])
        skip = (not applies) or min(need) < N_MIN
        out[f"skip_{c}"] = skip
        out[f"skip_reason_{c}"] = "" if not skip else ("not applicable to cohort" if not applies else f"< {N_MIN} trials")
        cols = {k: np.full(N, np.nan) for k in ["auc", "p", "p_mw", "auc_null", "p_null", "p_mw_null"]}
        if not skip:
            if kind == "ix":
                args = (R.bs(S["W_lick_T1"], win), R.bs(S["W_lick_T3"], win), R.bs(S["A_hit_H1"], win), R.bs(S["A_hit_H2"], win))
                cols["auc"], cols["p"] = interaction_test(*args, rng)
                sh = []                                          # chance: shuffle labels within each modality
                for x1, x2 in (args[:2], args[2:]):
                    pool = np.concatenate([x1, x2], 1)[:, rng.permutation(x1.shape[1] + x2.shape[1])]
                    sh += [pool[:, :x1.shape[1]], pool[:, x1.shape[1]:]]
                cols["auc_null"], cols["p_null"] = interaction_test(*sh, rng)
            else:
                if kind == "base":
                    x1, x2 = R.win(S[s1], win), R.win(S[s1], f"baseline_{win}")
                else:
                    x1, x2 = R.bs(S[s1], win), R.bs(S[s2], win)
                cols["auc"], cols["p"], cols["p_mw"] = roc_test(x1, x2, rng)
                pool = np.concatenate([x1, x2], 1)[:, rng.permutation(x1.shape[1] + x2.shape[1])]
                cols["auc_null"], cols["p_null"], cols["p_mw_null"] = roc_test(pool[:, :x1.shape[1]], pool[:, x1.shape[1]:], rng)
        for k, v in cols.items():
            out[f"{k}_{c}"] = v
        out[f"pred_{c}"] = pred if pred is not None else (+1 if cohort == "R+" else -1)
    for name, key, win in [("mono_W_lick_outcome", "W_lick", "outcome"), ("mono_W_all_cue", "W_all", "cue")]:
        parts = [S[f"{key}_T{i}"] for i in (1, 2, 3)]
        if min(len(p) for p in parts) >= N_MIN:
            idx = np.concatenate([np.full(len(p), i + 1) for i, p in enumerate(parts)])
            out[f"rho_{name}"], out[f"p_{name}"] = spearman_rows(np.concatenate([R.bs(p, win) for p in parts], 1), idx)
        else:
            out[f"rho_{name}"], out[f"p_{name}"] = np.nan, np.nan
    # descriptive AUC vs same-trial baseline per set (population plots)
    for win, keys in [("cue", ["W_first", "W_next", "W_all_T1", "W_all_T2", "W_all_T3", "A_all_H1", "A_all_H2"]),
                      ("outcome", ["W_lick_T1", "W_lick_T2", "W_lick_T3", "A_hit_H1", "A_hit_H2"])]:
        for k in keys:
            out[f"popauc_{win}_{k}"] = (auc_first(R.win(S[k], win), R.win(S[k], f"baseline_{win}"))[0]
                                        if len(S[k]) >= N_MIN else np.nan)
    # novelty curve: z-scored cue response of the first N_CURVE whisker trials
    cue_all = R.bs(S["W_all"], "cue")
    sd = cue_all.std(1); sd[sd == 0] = np.nan
    for i in range(N_CURVE):
        out[f"cuez_w{i + 1:02d}"] = cue_all[:, i] / sd if i < cue_all.shape[1] else np.nan
    first_lick = int(np.argmax(S["W_all"].lick_flag.to_numpy() == 1)) if len(S["W_lick"]) else -1
    meta = dict(PARAMS, session_id=session_id, day=day, cohort=cohort, n_trials={k: len(v) for k, v in S.items()},
                first_whisker_lick_index=first_lick,
                first_whisker_licks=S["W_first"].lick_flag.astype(int).tolist())
    return out, meta


def cmd_run(a):
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
        if a.sessions and sid not in a.sessions:
            continue
        todo.append((abs(day), day, sid, f.parent.parent))
    todo.sort()
    print(f"[029] {len(todo)} sessions", flush=True)
    rng = np.random.default_rng(0)
    for i, (_, day, sid, dayf) in enumerate(todo):
        out_dir = dayf / SUBDIR
        mouse = sid.split("_")[0]
        f_out = out_dir / f"{mouse}_roc_rpe_v2.csv"
        if f_out.exists() and not a.overwrite:
            print(f"[029] {i + 1}/{len(todo)} {sid}: exists, skip", flush=True)
            continue
        t0 = time.time()
        try:
            df, meta = run_session(sid, day, rng)
            out_dir.mkdir(exist_ok=True)
            tmp = f_out.with_suffix(".tmp.csv"); df.to_csv(tmp, index=False); tmp.replace(f_out)
            json.dump(meta, open(out_dir / f"{mouse}_roc_rpe_v2_config.json", "w"), indent=2, default=str)
            sig = {c: round(float(np.nanmean(df[f"p_{c}"] < ALPHA)), 3) for c in COMPS if not df[f"skip_{c}"].all()}
            print(f"[029] {i + 1}/{len(todo)} {sid} ({meta['cohort']}, day {day}): {(time.time() - t0) / 60:.1f} min; "
                  f"frac p<.05 {sig}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[029] {i + 1}/{len(todo)} {sid}: FAILED {type(e).__name__}: {e}", flush=True)
    print("ALL DONE", flush=True)


# ------------------------------------------------------------------ aggregate
def classify(d, suffix=""):
    """uncorrected p < ALPHA in the predicted direction. suffix '' = real data, '_null' = shuffled-label chance run"""
    def s(c, sign):
        auc = d[f"auc{suffix}_{c}"]
        return ((d[f"p{suffix}_{c}"] < ALPHA) & (np.sign(auc - 0.5) == sign)).fillna(False).to_numpy()
    lab = {}
    for k, (coh, rule) in CLASS_RULES.items():
        ok = (d.cohort == coh).to_numpy() if coh != "both" else np.ones(len(d), bool)
        lab[k] = rule(s) & ok
    return pd.DataFrame(lab, index=d.index)


def load_all(sessions=None):
    fs = sorted(glob.glob(str(RES / "*" / "whisker_*" / SUBDIR / "*_roc_rpe_v2.csv")))
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if sessions:
        d = d[d.session_id.isin(sessions)]
    d["stage"] = np.where(d.day == 0, "learning", "expert")
    return d


def cmd_aggregate(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = SUMMARY / a.tag if a.tag else SUMMARY
    out.mkdir(parents=True, exist_ok=True)
    d = load_all(a.sessions)
    n_all = len(d)
    print("[029-agg] quality_label counts:", d.quality_label.value_counts().to_dict(),
          "| drift test available:", round(float(d.drift_test_available.mean()), 3), flush=True)
    if a.quality != "all":
        d = d[d.quality_label == a.quality]
    print(f"[029-agg] {d.session_id.nunique()} sessions, {len(d)} / {n_all} units (quality {a.quality}) -> {out}", flush=True)
    lab = classify(d); labn = classify(d, "_null")
    for k in CLASS_RULES:
        d[f"class_{k}"] = lab[k]; d[f"null_class_{k}"] = labn[k]
    d["O6_equivalent_0.1"] = (d["auc_O6"] - 0.5).abs() < 0.1
    d["class_labels"] = lab.apply(lambda r: "|".join(k for k in CLASS_RULES if r[k]), axis=1)
    d.to_parquet(out / "rpe_v2_neurons.parquet", index=False)
    # --- per comparison: direction-resolved fraction p < ALPHA, real vs chance
    cs = []
    for (coh, stg), g in d.groupby(["cohort", "stage"]):
        for c in COMPS:
            v = g[~g[f"skip_{c}"].astype(bool)]
            if not len(v):
                continue
            ctr = 0.0 if c == "IX" else 0.5
            fr = lambda p, auc, sgn: float(((p < ALPHA) & (np.sign(auc - ctr) == sgn)).mean())   # noqa: E731
            cs.append(dict(cohort=coh, stage=stg, comparison=c, n_units=len(v), n_sessions=v.session_id.nunique(),
                           mean_auc=v[f"auc_{c}"].mean(),
                           frac_up=fr(v[f"p_{c}"], v[f"auc_{c}"], 1), frac_down=fr(v[f"p_{c}"], v[f"auc_{c}"], -1),
                           chance_up=fr(v[f"p_null_{c}"], v[f"auc_null_{c}"], 1),
                           chance_down=fr(v[f"p_null_{c}"], v[f"auc_null_{c}"], -1)))
    C = pd.DataFrame(cs); C.to_csv(out / "rpe_v2_comparison_summary.csv", index=False)
    # --- class totals (real vs chance) and overlaps
    O = pd.DataFrame([dict(cohort=coh, stage=stg, n_units=len(g), n_sessions=g.session_id.nunique(),
                           **{k: g[f"class_{k}"].mean() for k in CLASS_RULES},
                           **{f"{k}_chance": g[f"null_class_{k}"].mean() for k in CLASS_RULES})
                      for (coh, stg), g in d.groupby(["cohort", "stage"])])
    O.to_csv(out / "rpe_v2_class_totals.csv", index=False)
    (d[d.class_labels != ""].groupby(["cohort", "stage", "class_labels"]).size().rename("n_units").reset_index()
     .sort_values(["cohort", "stage", "n_units"], ascending=[True, True, False])
     .to_csv(out / "rpe_v2_class_overlaps.csv", index=False))
    # --- class proportions per region vs chance (one-sided binomial vs the pooled chance proportion, uncorrected)
    rows = []
    for level in ["area_group", "area_acronym_custom"]:
        for (coh, stg), g in d.groupby(["cohort", "stage"]):
            for k, (kc, _) in CLASS_RULES.items():
                if kc not in ("both", coh):
                    continue
                chance = g[f"null_class_{k}"].mean()
                for reg, gr in g.groupby(level):
                    n, x = len(gr), int(gr[f"class_{k}"].sum())
                    rows.append(dict(level=level, cohort=coh, stage=stg, cls=k, region=reg, n_units=n,
                                     n_sessions=gr.session_id.nunique(), n_class=x, prop=x / n,
                                     chance_prop_pooled=chance, chance_prop_region=gr[f"null_class_{k}"].mean(),
                                     p_binom_vs_chance=stats.binomtest(x, n, max(chance, 1e-6), alternative="greater").pvalue))
    P = pd.DataFrame(rows); P.to_csv(out / "rpe_v2_class_proportions_by_region.csv", index=False)
    # --- population plots (per-session mean AUC vs same-trial baseline, mean +- SEM across sessions)
    for stg in sorted(d.stage.unique()):
        fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), sharey="row")
        for j, coh in enumerate(["R+", "R-"]):
            g = d[(d.cohort == coh) & (d.stage == stg)]
            if not len(g):
                continue
            ps = g.groupby("session_id")[[c for c in d.columns if c.startswith("popauc_")]].mean()
            specs = {"cue": [("whisker trials", "#2ca02c", [(0, "W_first"), (1, "W_next"), (2, "W_all_T1"), (3, "W_all_T2"), (4, "W_all_T3")]),
                             ("auditory trials", "#1f77b4", [(2.5, "A_all_H1"), (3.5, "A_all_H2")])],
                     "outcome": [("whisker " + ("hits" if coh == "R+" else "licks"), "#2ca02c", [(2, "W_lick_T1"), (3, "W_lick_T2"), (4, "W_lick_T3")]),
                                 ("auditory hits", "#1f77b4", [(2.5, "A_hit_H1"), (3.5, "A_hit_H2")])]}
            for i, win in enumerate(["cue", "outcome"]):
                ax = axs[i, j]
                for lb, col, pts in specs[win]:
                    x = [p for p, _ in pts]
                    v = [ps[f"popauc_{win}_{k}"].dropna() for _, k in pts]
                    ax.errorbar(x, [s.mean() for s in v], [s.std() / np.sqrt(max(len(s), 1)) for s in v], marker="o",
                                color=col, label=lb, capsize=3)
                ax.axhline(0.5, color="k", lw=0.6, ls="--")
                ax.set_xticks([0, 1, 2, 3, 4, 2.5, 3.5], [f"First{N_FIRST}", "Next", "T1", "T2", "T3", "H1", "H2"], fontsize=7)
                ax.set_title(f"{coh} {win} window (n={len(ps)} sessions)", fontsize=9); ax.legend(fontsize=7)
                if j == 0:
                    ax.set_ylabel("mean AUC vs same-trial baseline")
        fig.suptitle(f"population AUC ({stg}); whisker thirds, auditory halves (plotted between thirds)", fontsize=10)
        fig.tight_layout(); fig.savefig(out / f"rpe_v2_population_auc_{stg}.png", dpi=150); plt.close(fig)
    # --- novelty curve: z-scored cue response vs whisker trial index, cue-responsive neurons (defined on T2, not
    #     on the first trials -> no circularity) and all neurons
    zc = [c for c in d.columns if c.startswith("cuez_w")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, (lab_, sel) in zip(axs, [("all neurons", np.ones(len(d), bool)),
                                     ("cue-responsive neurons (AUC cue vs baseline on whisker T2 > 0.6)",
                                      (d["popauc_cue_W_all_T2"] > 0.6).to_numpy())]):
        for coh, col in [("R+", "#00B400"), ("R-", "#C800C8")]:
            g = d[sel & (d.cohort == coh).to_numpy() & (d.stage == "learning").to_numpy()]
            if not len(g):
                continue
            ps = g.groupby("session_id")[zc].mean()
            ax.errorbar(np.arange(1, len(zc) + 1), ps.mean(), ps.std() / np.sqrt(len(ps)), color=col, marker="o", ms=3,
                        label=f"{coh} (n={len(ps)} sessions)", capsize=2)
        ax.axvline(N_FIRST + 0.5, color="0.5", ls=":"); ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("whisker trial index (session order)"); ax.set_title(lab_, fontsize=9); ax.legend(fontsize=7)
    axs[0].set_ylabel("cue response - baseline (z, per neuron)")
    fig.suptitle(f"novelty curve (learning day); dotted = first-{N_FIRST} boundary", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "rpe_v2_novelty_curve.png", dpi=150); plt.close(fig)
    write_readme(out, d)
    pd.set_option("display.width", 220)
    print(C.round(3).to_string()); print(O.round(4).T.to_string())
    print("AGG DONE", flush=True)


def write_readme(out, d):
    txt = f"""# RPE ROC v2 (029_roc_rpe_v2.py)

Specification by Axel Bisi (2026-09-29). All units (no QC filter). Learning = day 0, expert = day > 0.
Catch (no-stim) trials are never used. See the script docstring for the full definition of sets and comparisons.

## Windows (per trial, per neuron, firing rate in spikes/s)
- Baseline, from the SAME trial (single-trial baseline), length-matched to the analysed window (unequal lengths bias
  the AUC: pre-trial 100 ms vs 300 ms gives mean AUC 0.47 under the null):
  cue analyses {int(-BASE_W_CUE[0] * 1e3)} to {int(-BASE_W_CUE[1] * 1e3)} ms, outcome analyses
  {int(-BASE_W[0] * 1e3)} to {int(-BASE_W[1] * 1e3)} ms before trial start (stimulus onset = start_time in all sessions).

## Units
All units are computed; summaries/classes/examples use quality_label == 'good' only (stable, well-isolated units):
unit_metrics_utils.classify_units_quality on bombcell metrics + presence ratio (>= 0.5 of 60-s bins) + coverage ratio
(>= 0.9 of the recording) + DREDge motion shift test (fails only if |r| < 0.5 AND p < 0.01 jointly; test file missing
for some sessions -> drift check skipped for those units, drift_test_available = False).
- Cue: {int(CUE_W[0] * 1e3)}-{int(CUE_W[1] * 1e3)} ms after stimulus onset.
- Outcome: {int(OUT_W[0] * 1e3)}-{int(OUT_W[1] * 1e3)} ms after the corrected first lick
  (start_time + lick_time - response_window_start_time); same window for unrewarded R- whisker licks.
  **The outcome window includes the pre-reward period and the lick itself** (reward is received around the 2nd lick).
- Whisker magnetic-artifact correction is applied to all spike trains (roc_utils_new: [-10, +5] ms around each whisker
  onset replaced by Poisson spikes at the [-200, -10] ms rate); no analysis window overlaps [-10, +5] ms.

## Trial sets
- Whisker trials (hits + misses) and whisker licks (R+ rewarded hits / R- unrewarded licks): thirds T1/T2/T3.
- Auditory hits / auditory trials: halves H1/H2 (thirds too small: day-0 R+ median 8 auditory hits per third).
  Warm-up block kept (exception to the ssl-analyze warm-up trim rule); first {N_AUD_SKIP} auditory trials excluded.
- First = first {N_FIRST} whisker trials (licked or not); Next = the following whisker trials up to the end of T1.

## Comparisons
O1-O6, C1-C3 as specified (O3: whisker T1 vs auditory H1; O6: auditory H1 vs H2), plus C1b (novelty decrement:
First vs Next, cue), O3tm (time-matched auditory), IX (whisker T1-T3 change minus auditory H1-H2 change).

## Statistics
- Two-sided label-permutation p ({N_PERM} shuffles; minimum p {1 / (N_PERM + 1):.4f}); asymptotic Mann-Whitney p_mw stored.
- **No multiple-comparison correction** (comparison scheme not fixed yet): classes use uncorrected p < {ALPHA} in the
  predicted direction. Chance = the same pipeline on shuffled set labels (p_null, null_class_*).
- Skipped if either set has < {N_MIN} trials (with 2 vs 2 trials the minimum two-sided p is 0.33).

## Classes (a neuron can hold several)
positive_RPE (R+): O1 & O2 & O3 | negative_RPE (R-): O4 & O5 | reward (R+): O1 & not O2 | reward_cue (R+): C2 |
cue_decay (R-): C3 | novelty_response: C1 | novelty_decrement: C1b.

## Parameters
```
{json.dumps(PARAMS, indent=2, default=str)}
```
Sessions: {d.session_id.nunique()}, units: {len(d)}.
"""
    (out / "README.md").write_text(txt)


# ------------------------------------------------------------------ examples
def cmd_examples(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = SUMMARY / a.tag if a.tag else SUMMARY
    d = pd.read_parquet(out / "rpe_v2_neurons.parquet")
    d = d[d.stage == "learning"]
    ex_dir = out / "examples"; ex_dir.mkdir(exist_ok=True)
    score = {"positive_RPE": lambda g: g.auc_O1 + g.auc_O2 + g.auc_O3, "negative_RPE": lambda g: -(g.auc_O4 + g.auc_O5),
             "reward": lambda g: g.auc_O1, "reward_cue": lambda g: g.auc_C2, "cue_decay": lambda g: -g.auc_C3,
             "novelty_response": lambda g: g.auc_C1, "novelty_decrement": lambda g: g.auc_C1b}
    c3 = ["#98df8a", "#2ca02c", "#0b4d0b"]; a2 = ["#6baed6", "#08306b"]
    cache = {}
    for k, f in score.items():
        g = d[d[f"class_{k}"]]
        if not len(g):
            print(f"[029-ex] {k}: no neurons", flush=True); continue
        g = g.assign(score=f(g)).sort_values("score", ascending=False)
        g = g.groupby("cohort", group_keys=False).head(a.n_examples)          # top per cohort
        fig, axs = plt.subplots(2 * len(g), 2, figsize=(11, 4.0 * len(g)), gridspec_kw=dict(height_ratios=[2, 1] * len(g)))
        axs = np.atleast_2d(axs)
        for i, (_, row) in enumerate(g.iterrows()):
            if row.session_id not in cache:
                cache = {row.session_id: load_session(row.session_id)}         # keep one session in memory
            _, units, act = cache[row.session_id]
            _, S = sets_for(act)
            kk = int(np.where(units.neuron_id.to_numpy() == row.neuron_id)[0][0])
            st = np.sort(np.asarray(units.spike_times.iloc[kk]))
            for j, (align, groups, win) in enumerate([
                    ("start_time", [(f"first {N_FIRST}", S["W_first"], "#ff7f0e")] +
                     [(f"W T{t}", S[f"W_all_T{t}"], c3[t - 1]) for t in (1, 2, 3)] +
                     [(f"A H{t}", S[f"A_all_H{t}"], a2[t - 1]) for t in (1, 2)], CUE_W),
                    ("L1", [(f"W lick T{t}", S[f"W_lick_T{t}"], c3[t - 1]) for t in (1, 2, 3)] +
                     [(f"A hit H{t}", S[f"A_hit_H{t}"], a2[t - 1]) for t in (1, 2)], OUT_W)]):
                ax, axp = axs[2 * i, j], axs[2 * i + 1, j]; y = 0; bins = np.arange(-0.5, 0.8, 0.02)
                for lb, tr, col in groups:
                    t0 = tr[align].to_numpy(float)
                    rel = [st[(st >= t - 0.5) & (st < t + 0.8)] - t for t in t0]
                    for r_ in rel:
                        ax.plot(r_, np.full(len(r_), y), "|", ms=1.5, color=col); y += 1
                    y += 2
                    if len(t0):
                        h = np.histogram(np.concatenate(rel), bins)[0] / len(t0) / 0.02
                        axp.plot(bins[:-1] + 0.01, np.convolve(h, np.ones(3) / 3, "same"), color=col, label=f"{lb} (n={len(t0)})")
                for aa in (ax, axp):
                    aa.axvspan(*win, color="orange", alpha=0.2); aa.axvline(0, color="k", lw=0.5)
                    if align == "start_time":
                        aa.axvspan(*BASE_W_CUE, color="0.85", alpha=0.6)
                ax.set_ylim(y, -1); ax.set_xlim(-0.5, 0.8); ax.set_yticks([])
                axp.set_xlim(-0.5, 0.8); axp.legend(fontsize=5.5, ncol=2); axp.set_ylabel("Hz", fontsize=7)
                axp.set_xlabel("time from stimulus (s)" if align == "start_time" else "time from first lick (s)", fontsize=7)
                ax.set_title(f"{row.session_id} n{row.neuron_id} {row.area_acronym_custom} ({row.cohort}) "
                             + ("cue (grey = baseline)" if j == 0 else "outcome"), fontsize=8)
            aucs = "  ".join(f"{c}={row[f'auc_{c}']:.2f}(p={row[f'p_{c}']:.3f})"
                             for c in ["O1", "O2", "O3", "O4", "O5", "O6", "C1", "C1b", "C2", "C3"] if pd.notna(row[f"auc_{c}"]))
            axs[2 * i, 0].text(0, 1.2, aucs, transform=axs[2 * i, 0].transAxes, fontsize=6.5)
        fig.suptitle(f"class {k}: top examples per cohort (uncorrected p)", fontsize=11, y=1.0)
        fig.tight_layout(); fig.savefig(ex_dir / f"examples_{k}.png", dpi=110); plt.close(fig)
        print(f"[029-ex] {k}: {len(g)} examples", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--days", nargs="*", type=int); r.add_argument("--sessions", nargs="*")
    r.add_argument("--overwrite", action="store_true")
    g = sp.add_parser("aggregate"); g.add_argument("--sessions", nargs="*"); g.add_argument("--tag", default="")
    g.add_argument("--quality", default="good", help="quality_label to keep ('good', 'mua', or 'all')")
    e = sp.add_parser("examples"); e.add_argument("--n-examples", type=int, default=3); e.add_argument("--tag", default="")
    a = ap.parse_args()
    {"run": cmd_run, "aggregate": cmd_aggregate, "examples": cmd_examples}[a.cmd](a)
