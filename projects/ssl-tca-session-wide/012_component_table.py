"""One row per TCA component (session x method x alignment x component) at the cohort's rank (011), with descriptors
and within-session relations to behaviour. Exploratory (user 2026-11-22: all sessions, exploratory).

Methods: "_splitnn" (main; split non-negative on single-trial baseline-subtracted z-scored data; unit weight reported
as net = increase - decrease) and "_nn" (reference; non-negative on rates, no baseline subtraction).
Descriptors
- time: peak latency (ms from the event), width (ms above half of the peak), pre-event share (mean weight before the
  event / mean weight), flatness (CV of the time factor; < 0.2 = flat).
- units: directionality (split only; median over the top half of units by total weight of (inc - dec) / (inc + dec);
  about 0 = variability / noise component); area profile = mean net weight per area group (JSON); top group;
  specificity = |top group mean| / sum of |group means|, p against 300 permutations of area labels across units.
- trials: mean trial weight per class (A_/P_ x whisker / auditory x lick).
- behaviour (trial weights):
  * perf_auc: active whisker trials, hit vs miss (stimulus alignments; >= 5 of each);
  * learn_rho: Spearman correlation of trial weights with the exact whisker learning curve (p_wh_i), both residualised
    on trial index (partial on time); trials: active whisker trials (stimulus alignments) or whisker hits (jaw / lick);
  * null for both: non-wrapping linear shifts of the behaviour against the trial weights in trial order, lags 10-50 %
    of the trials, both directions (shift_null.py convention), up to 100 shifts; reported as real, null mean, excess,
    two-sided p (|value - chance| against the null);
  * learn_ctrl (stimulus alignments): as learn_rho, with the weights also residualised on the trial's own outcome
    (lick 0 / 1), i.e. learning beyond 'higher on hit trials' (the curve is a smoothed hit rate);
  * passive_change: whisker passive trials, mean trial weight post - pre, in units of the trial-weight SD (stimulus
    alignments; no null, time-confounded by construction).

Run: python 012_component_table.py [--n-jobs 40] -> OUT/tables/components.parquet
"""
import argparse
import importlib
import json
import os
import traceback
import zlib

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

import tca_lib as L  # noqa: E402

P = importlib.import_module("004_pilot_figures")   # area_colours (ephys_utilities palette / groups)
A = importlib.import_module("005_build_aligned")
D = L.OUT / "artifacts" / "tca_aligned_gpu"
METHODS = ["_splitnn", "_nn"]
import os as _os
# optional subset of alignments written to separate tables (user 2026-11-23: go-signal tensors), e.g.
# TCA_ALIGNMENTS=stim_long_ns,jaw_all TCA_SUFFIX=_go; default: the four original alignments, original tables
SUFFIX = _os.environ.get("TCA_SUFFIX", "")
ALIGNMENTS = _os.environ.get("TCA_ALIGNMENTS", "stim_long,stim_sensory,jaw,lick").split(",")
N_PERM_AREA = 300
N_SHIFT = 100
LO, HI = 0.1, 0.5
CLASSES = {"A_whisker_1": ("A", "whisker_trial", 1), "A_whisker_0": ("A", "whisker_trial", 0),
           "A_auditory_1": ("A", "auditory_trial", 1), "A_auditory_0": ("A", "auditory_trial", 0),
           "P_whisker": ("P", "whisker_trial", None), "P_auditory": ("P", "auditory_trial", None),
           # no-stimulus trials (only in stim_long_ns / jaw_all; NaN elsewhere): false alarms / correct rejections
           "A_no_stim_1": ("A", "no_stim_trial", 1), "A_no_stim_0": ("A", "no_stim_trial", 0)}


def shifts(n, rng):
    lo, hi = max(1, int(LO * n)), max(1, int(HI * n))
    pairs = [(k, dr) for k in range(lo, hi + 1) for dr in (0, 1)]
    if len(pairs) > N_SHIFT:
        pairs = [pairs[i] for i in rng.choice(len(pairs), N_SHIFT, replace=False)]
    return pairs


def cut(n, k, dr):
    """positions of (behaviour, weights) paired by a non-wrapping shift of lag k"""
    return (np.arange(k, n), np.arange(n - k)) if dr else (np.arange(n - k), np.arange(k, n))


def auc(w, y):
    y = np.asarray(y, bool)
    if y.sum() < 5 or (~y).sum() < 5:
        return np.nan
    return float(stats.mannwhitneyu(w[y], w[~y]).statistic / (y.sum() * (~y).sum()))


def partial_rho(w, p, idx, outcome=None):
    """Spearman correlation of trial weights and learning curve, both residualised on trial index; with outcome
    (lick 0 / 1 per trial), the weights are also residualised on the trial's own outcome, so the correlation is not
    just 'higher on hit trials, and hits are dense where the curve is high'."""
    ok = np.isfinite(p)
    if ok.sum() < 10:
        return np.nan
    w, p, idx = w[ok], p[ok], idx[ok].astype(float)
    X = np.c_[np.ones_like(idx), idx] if outcome is None else np.c_[np.ones_like(idx), idx, outcome[ok]]
    rw = w - X @ np.linalg.lstsq(X, w, rcond=None)[0]
    rp = p - np.polyval(np.polyfit(idx, p, 1), idx)
    if np.std(rp) < 1e-9 or np.std(rw) < 1e-12:
        return np.nan
    return float(stats.spearmanr(rw, rp)[0])


def with_null(fn, w, b, idx, chance, rng, extra=None):
    """real value and non-wrapping shift null of fn(weights, behaviour, trial index[, extra]) in trial order; extra
    (e.g. the trial's own outcome) stays paired with the weights"""
    real = fn(w, b, idx) if extra is None else fn(w, b, idx, extra)
    n = len(w)
    null = []
    for k, dr in shifts(n, rng):
        pb, pw = cut(n, k, dr)
        null.append(fn(w[pw], b[pb], idx[pw]) if extra is None else fn(w[pw], b[pb], idx[pw], extra[pw]))
    null = np.asarray([x for x in null if np.isfinite(x)])
    if not np.isfinite(real) or len(null) < 10:
        return dict(real=real, null_mean=np.nan, excess=np.nan, p=np.nan, n_null=len(null))
    p = (np.sum(np.abs(null - chance) >= abs(real - chance)) + 1) / (len(null) + 1)
    return dict(real=real, null_mean=float(null.mean()), excess=float(real - null.mean()), p=float(p),
                n_null=len(null))


def one_session(sid, meta, ranks):
    out = []
    rng = np.random.default_rng(zlib.crc32(sid.encode()))   # reproducible per session
    try:
        trials = pd.read_parquet(A.OUT_DIR / sid / "trials.parquet").set_index("trial_index")
        units = pd.read_parquet(A.OUT_DIR / sid / "units.parquet")
    except Exception:
        return [dict(session_id=sid, error=traceback.format_exc()[-400:])]
    for tag in METHODS:
        for al in ALIGNMENTS:
            f = D / sid / f"{al}_zscore{tag}.npz"
            if not f.exists():
                continue
            try:
                d = np.load(f)
                r = int(ranks.get((tag, al, meta["cohort"]), 7))
                if f"r{r}_f0" not in d.files:
                    continue
                Fa, Fu, Ft = d[f"r{r}_f0"], d[f"r{r}_f1"], d[f"r{r}_f2"]
                J = len(d["units_kept"])
                split = Fu.shape[0] == 2 * J
                up, dn = (Fu[:J], Fu[J:]) if split else (Fu, None)
                net = up - dn if split else up
                u = units.iloc[d["units_kept"]].reset_index(drop=True)
                _, _, grp = P.area_colours(u.area_acronym_custom.str.strip().tolist())
                grp = np.asarray(grp)
                groups = sorted(set(grp))
                t = trials.loc[d["trial_index"]].reset_index()
                tc = d["t_centres"] * 1e3
                ctx = np.where(t.epoch.str.startswith("passive"), "P", "A")
                lick = t.lick_flag.fillna(0).to_numpy().astype(int)
                tt = t.trial_type.to_numpy()
                idx = t.trial_index.to_numpy()
                stim_al = A.ALIGN[al]["event"] == "stim"
                for k in range(r):
                    w, T = Fa[:, k], Ft[:, k]
                    row = dict(session_id=sid, mouse_id=meta["mouse_id"], cohort=meta["cohort"], stage=meta["stage"],
                               learner=meta["learner"], learning_category=meta["learning_category"], method=tag,
                               alignment=al, rank=r, comp=k + 1, n_trials=len(w), n_units=J)
                    # time
                    row["peak_ms"] = float(tc[np.argmax(T)])
                    row["width_ms"] = float((T >= 0.5 * T.max()).sum() * np.median(np.diff(tc)))
                    row["pre_share"] = float(T[tc < 0].mean() / max(T.mean(), 1e-12)) if (tc < 0).any() else np.nan
                    row["flat_cv"] = float(T.std() / max(abs(T.mean()), 1e-12))
                    row["time_factor"] = json.dumps(np.round(T / max(np.linalg.norm(T), 1e-12), 5).tolist())
                    row["t_centres_ms"] = json.dumps(np.round(tc, 2).tolist())
                    # units
                    if split:
                        tot = up[:, k] + dn[:, k]
                        top = tot > np.percentile(tot, 50)
                        row["directionality"] = float(np.median((up[top, k] - dn[top, k]) / np.maximum(tot[top], 1e-12)))
                    else:
                        row["directionality"] = np.nan
                    gm = np.array([net[grp == g, k].mean() for g in groups])
                    spec = np.abs(gm).max() / max(np.abs(gm).sum(), 1e-12)
                    null = []
                    for _ in range(N_PERM_AREA):
                        pg = rng.permutation(grp)
                        nm = np.array([net[pg == g, k].mean() for g in groups])
                        null.append(np.abs(nm).max() / max(np.abs(nm).sum(), 1e-12))
                    row["area_profile"] = json.dumps({g: round(float(v), 5) for g, v in zip(groups, gm)})
                    row["top_group"] = groups[int(np.argmax(np.abs(gm)))]
                    row["top_group_sign"] = float(np.sign(gm[np.argmax(np.abs(gm))]))
                    row["specificity"] = float(spec)
                    row["specificity_p"] = float((np.sum(np.array(null) >= spec) + 1) / (N_PERM_AREA + 1))
                    row["net_weight_mean"] = float(net[:, k].mean())
                    # trial classes
                    for name, (c_, ty, lk) in CLASSES.items():
                        m = (ctx == c_) & (tt == ty) & ((lick == lk) if lk is not None else True)
                        row[f"cls_{name}"] = float(w[m].mean()) if m.sum() >= 3 else np.nan
                        row[f"n_{name}"] = int(m.sum())
                    # behaviour
                    if stim_al:
                        aw = np.flatnonzero((ctx == "A") & (tt == "whisker_trial"))
                    else:
                        aw = np.flatnonzero((tt == "whisker_trial") & (lick == 1))
                    curve = t.p_wh_i.to_numpy(float) if "p_wh_i" in t else np.full(len(t), np.nan)
                    if stim_al and len(aw) >= 10:
                        res = with_null(lambda w_, y_, i_: auc(w_, y_), w[aw], lick[aw].astype(float), idx[aw], 0.5, rng)
                        row.update({f"perf_auc_{kk}": v for kk, v in res.items()})
                    if len(aw) >= 10:
                        res = with_null(partial_rho, w[aw], curve[aw], idx[aw], 0.0, rng)
                        row.update({f"learn_rho_{kk}": v for kk, v in res.items()})
                    if stim_al and len(aw) >= 10:   # learning beyond the trial's own outcome (hit / miss)
                        lk_aw = lick[aw].astype(float)
                        res = with_null(lambda w_, p_, i_, lk_=None: partial_rho(w_, p_, i_, lk_), w[aw], curve[aw],
                                        idx[aw], 0.0, rng, extra=lk_aw)
                        row.update({f"learn_ctrl_{kk}": v for kk, v in res.items()})
                    if stim_al:
                        ep = t.epoch.to_numpy()
                        pre = (ep == "passive_pre") & (tt == "whisker_trial")
                        post = (ep == "passive_post") & (tt == "whisker_trial")
                        if pre.sum() >= 5 and post.sum() >= 5:
                            row["passive_change"] = float((w[post].mean() - w[pre].mean()) / max(w.std(), 1e-12))
                    out.append(row)
            except Exception:
                out.append(dict(session_id=sid, method=tag, alignment=al, error=traceback.format_exc()[-600:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=40)
    a = ap.parse_args()
    s = pd.read_csv(L.OUT / "tables" / "sessions.csv")
    rk = pd.read_csv(L.OUT / "tables" / f"rank_by_cohort{SUFFIX}.csv")
    rk = rk[rk.stage == "learning"]
    ranks = {(r.method, r.alignment, r.cohort): int(r.proposed_rank) for r in rk.itertuples()}
    res = Parallel(n_jobs=a.n_jobs, backend="loky")(
        delayed(one_session)(r.session_id, dict(mouse_id=r.mouse_id, cohort=r.cohort, stage=r.stage, learner=r.learner,
                                                learning_category=r.learning_category), ranks)
        for r in s.itertuples())
    df = pd.DataFrame([x for xs in res for x in xs])
    err = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    print(f"{len(df) - len(err)} components, {len(err)} errors")
    if len(err):
        print(err[["session_id", "error"]].head().to_string())
    df.to_parquet(L.OUT / "tables" / f"components{SUFFIX}.parquet", index=False)


if __name__ == "__main__":
    main()
