"""Figure1_behaviour -- compute every filled panel's tidy table from scratch and write it to the snapshot's data/ folder.

Panels are found by their stable `key` (spec.yaml); letters (`id`) follow the planner layout and change with it.
Run on haas (repo root):
    nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/compute.py [--panels e f] [--rebuild-cache] [--date D]
then build.py figures/Figure1_behaviour [--date D].
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import learning_curve_exact as lce  # noqa: E402
import shared_behaviour  # noqa: E402
from common import import_external_repos, load_spec, snapshot_dir, two_group_tests  # noqa: E402

import_external_repos()
import learning_utils  # noqa: E402

SPEC = load_spec(FIG_DIR)
PANELS = {p.get("key", p["id"]): p for p in SPEC["panels"]}  # by stable key (letters follow the layout)
CURVE_PARAMS = SPEC["shared"]["day0_curves"]["params"]
STATE_PARAMS = SPEC["shared"]["perf_states"]["params"]
COHORTS = ["R+", "R-"]
TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]
LEARNER = {"good": True, "moderate": True, "bad": False}  # reference sheet learning_category (user 2026-11-02)
MOVEMENT_COLS = ["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]


# ---------------------------------------------------------------- shared inputs

def day0_table(table, learners_only=False):
    d0 = table[(table.behavior == "whisker") & (table.day == 0)]
    return d0[d0.learning_category.map(LEARNER).fillna(False).astype(bool)] if learners_only else d0


def session_curves(session_table, ctx=None):
    """Exact whisker curve, chance (no-stim) and auditory curves on whisker trials, and performance states.
    Cached per session in ctx (the exact fit is the slow step and several panels use it)."""
    sid = session_table.session_id.iloc[0]
    if ctx is not None and sid in ctx.setdefault("curves", {}):
        return ctx["curves"][sid]
    st = session_table[session_table.early_lick == 0].sort_values("start_time")
    out = lce.session_learning_curves_exact(st, sigma=CURVE_PARAMS["sigma"], conf_int=CURVE_PARAMS["conf_int"])
    w = out["whisker_trial"]
    n = len(w["outcomes"])
    curves = pd.DataFrame({"whisker_trial": np.arange(1, n + 1), "lick": w["outcomes"], "p_mean": w["p_mean"],
                           "p_low": w["p_low"], "p_high": w["p_high"], "p_chance": w["p_chance"],
                           "p_auditory": out.get("auditory_trial", {}).get("at_whisker", {}).get("p_mean", np.nan)})
    blocks = learning_utils.assign_expertise_blocks(
        curves.assign(mouse_id=st.mouse_id.iloc[0], reward_group=int(st.reward_group.iloc[0]),
                      trial_id=np.arange(n), trial_type="whisker_trial"),
        n_consecutive=STATE_PARAMS["n_consecutive"])
    curves["state"] = blocks.block_perf_type.to_numpy()
    res = (curves, w["learning_trial"], w["mouse_cat"])
    if ctx is not None:
        ctx["curves"][sid] = res
    return res


def n_by_cohort(df):
    return {c: {"mice": int(df[df.cohort == c].mouse_id.nunique()),
                "sessions": int(df[df.cohort == c].session_id.nunique()) if "session_id" in df else None}
            for c in COHORTS}


def vs_zero(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {"n": len(v), "wilcoxon_p": float(stats.wilcoxon(v).pvalue) if len(v) > 5 else None,
            "t_p": float(stats.ttest_1samp(v, 0).pvalue) if len(v) > 2 else None}


def cluster_test(df, x, value, n_perm=1000, alpha=0.05, seed=0):
    """Cluster-based permutation test, R+ vs R- along `x` (one row per mouse per x). Statistic: Welch t per x; clusters
    of contiguous x with p < alpha and the same sign; mass = sum of t; null = largest |mass| after permuting the cohort
    labels of mice. Returns [{start, end, mass, p}] for every observed cluster."""
    wide = df.pivot_table(index=["mouse_id", "cohort"], columns=x, values=value)
    xs = wide.columns.to_numpy()
    cohort = wide.index.get_level_values("cohort").to_numpy()
    data = wide.to_numpy(float)

    def tvals(labels):
        a, b = data[labels == "R+"], data[labels == "R-"]
        with np.errstate(all="ignore"):
            res = stats.ttest_ind(a, b, axis=0, equal_var=False, nan_policy="omit")
        return np.asarray(res.statistic, float), np.asarray(res.pvalue, float)

    def clusters(t, p):
        sig = np.nan_to_num(p, nan=1) < alpha
        out, i = [], 0
        while i < len(t):
            if not sig[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(t) and sig[j + 1] and np.sign(t[j + 1]) == np.sign(t[i]):
                j += 1
            out.append((i, j, float(np.nansum(t[i:j + 1]))))
            i = j + 1
        return out

    t, p = tvals(cohort)
    observed = clusters(t, p)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for k in range(n_perm):
        cl = clusters(*tvals(rng.permutation(cohort)))
        null[k] = max([abs(c[2]) for c in cl], default=0)
    return [{"start": float(xs[s]), "end": float(xs[e]), "mass": m, "p": float((np.sum(null >= abs(m)) + 1) / (n_perm + 1))}
            for s, e, m in observed]


def mark_clusters(df, x, clusters, alpha=0.05):
    """Boolean column: x inside a cluster with permutation p < alpha (for the significance bar of the panel)."""
    sig = np.zeros(len(df), bool)
    for c in clusters:
        if c["p"] < alpha:
            sig |= (df[x] >= c["start"]).to_numpy() & (df[x] <= c["end"]).to_numpy()
    return sig


# ---------------------------------------------------------------- row 1: task, day-0 performance, RTs, pretraining

def compute_days(ctx):
    """Lick probability per mouse and trial type on auditory days -2 and -1, whisker day 0 and after day 0 (mean of
    whisker days +1, +2 and +3 per mouse); planner 2026-11-06: line plot, days -2, -1, 0, >0 (= 1, 2, 3)."""
    t = ctx["table"]
    t = t[t.day.isin([-2, -1, 0, 1, 2, 3]) & (t.early_lick == 0)]
    per_session = t.groupby(["mouse_id", "cohort", "day", "session_id", "trial_type"]).lick_flag.mean().reset_index()
    per_day = per_session.groupby(["mouse_id", "cohort", "day", "trial_type"]).lick_flag.mean().reset_index()
    per_day["day_group"] = per_day.day.map(lambda d: str(d) if d <= 0 else ">0")
    df = (per_day.groupby(["mouse_id", "cohort", "day_group", "trial_type"]).lick_flag.mean()
          .rename("lick_probability").reset_index())
    tests = {}
    for dg in ["-2", "-1", "0", ">0"]:  # every day: the R+ - R- panel marks each (planner 2026-11-09)
        for tt in TRIAL_TYPES:
            d = df[(df.day_group == dg) & (df.trial_type == tt)]
            if len(d):
                tests[f"day{dg} {tt}"] = two_group_tests(d[d.cohort == "R+"].lick_probability,
                                                         d[d.cohort == "R-"].lick_probability)
    n = {c: {dg: int(df[(df.cohort == c) & (df.day_group == dg)].mouse_id.nunique()) for dg in ["-2", "-1", "0", ">0"]}
         for c in COHORTS}
    return df, {"n": n, "stats": tests}


def compute_rt(ctx):
    d0 = day0_table(ctx["table"])
    licks = d0[(d0.lick_flag == 1) & (d0.early_lick == 0) & d0.trial_type.isin(TRIAL_TYPES)]
    df = licks[["mouse_id", "session_id", "cohort", "trial_type", "computed_reaction_time"]].rename(
        columns={"computed_reaction_time": "reaction_time_s"})
    df = df[df.reaction_time_s > 0].reset_index(drop=True)
    med = (df.groupby(["mouse_id", "session_id", "cohort", "trial_type"]).reaction_time_s
           .agg(median_reaction_time_s="median", n_licks="size").reset_index())  # panel: per-mouse medians
    tests = {}
    for tt in TRIAL_TYPES:
        m = med[med.trial_type == tt]
        tests[tt] = two_group_tests(m[m.cohort == "R+"].median_reaction_time_s, m[m.cohort == "R-"].median_reaction_time_s)
    across = {}  # within cohort, between trial types (paired over mice with both)
    for c in COHORTS:
        w = med[med.cohort == c].pivot_table(index="mouse_id", columns="trial_type", values="median_reaction_time_s")
        for a, b in [("whisker_trial", "auditory_trial"), ("whisker_trial", "no_stim_trial"),
                     ("auditory_trial", "no_stim_trial")]:
            if a in w and b in w:
                pair = w[[a, b]].dropna()
                if len(pair) > 5:
                    across[f"{c} {a} vs {b}"] = {"n_mice": len(pair),
                                                 "wilcoxon_p": float(stats.wilcoxon(pair[a], pair[b]).pvalue),
                                                 "paired_t_p": float(stats.ttest_rel(pair[a], pair[b]).pvalue)}
    return med, {"n": n_by_cohort(df),
                 "n_trials": {f"{c} {t}": int(n) for (c, t), n in df.groupby(["cohort", "trial_type"]).size().items()},
                 "stats_mouse_median": tests, "stats_across_trial_types": across}


def compute_pretraining(ctx):
    t = ctx["table"]
    aud = t[(t.behavior == "auditory") & (t.day < 0)]
    df = (aud.groupby(["mouse_id", "cohort"])
          .agg(n_pretraining_trials=("trial_id", "size"), n_auditory_days=("day", "nunique"),
               session_id=("session_id", lambda s: ";".join(sorted(set(s)))))
          .reset_index())
    col = "n_pretraining_trials"
    return df, {"n": n_by_cohort(df),
                "stats": two_group_tests(df[df.cohort == "R+"][col], df[df.cohort == "R-"][col]),
                "stats_days": two_group_tests(df[df.cohort == "R+"].n_auditory_days, df[df.cohort == "R-"].n_auditory_days),
                "median_trials": {c: float(df[df.cohort == c][col].median()) for c in COHORTS},
                "median_days": {c: float(df[df.cohort == c].n_auditory_days.median()) for c in COHORTS}}


RT_HITS, RT_BIN, RT_SHIFT_EXCLUDE, RT_N_SHIFT = 30, 1, 10, 500  # RT_BIN 1: hit by hit (user 2026-11-10; was 5)
RT_START, RT_LATE, RT_MIN_MICE = 5, (21, 30), 8  # change-from-start variant (explore_rt_time_b column 7, user 2026-11-10)


def day0_hits(ctx):
    """Licked trials of whisker day 0 per learner session (early licks dropped), numbered per trial type in time order
    (hit_number counts every licked trial of the type); rt_ms = computed_reaction_time in ms, nan outside 0-1 s.
    Cached in ctx."""
    if "day0_hits" in ctx:
        return ctx["day0_hits"]
    rows = []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        rt = (st.computed_reaction_time.where(st.computed_reaction_time.between(0, 1)) * 1000).to_numpy()
        for tt in ("whisker_trial", "auditory_trial"):
            sel = np.flatnonzero((st.trial_type == tt).to_numpy() & (st.lick_flag == 1).to_numpy())
            rows.append(pd.DataFrame({"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                                      "trial_type": tt, "hit_number": np.arange(1, len(sel) + 1), "rt_ms": rt[sel]}))
    ctx["day0_hits"] = pd.concat(rows, ignore_index=True)
    return ctx["day0_hits"]


def sign_flip_clusters(wide, n_perm=1000, alpha=0.05, seed=0):
    """One-sample cluster test vs 0 along the columns of wide [mouse, x]: one-sample t per x (nan-aware); clusters of
    contiguous x with p < alpha and the same sign, mass = sum of t; null = largest |mass| after flipping the sign of
    whole mice. Returns [{start, end, mass, p}] for every observed cluster."""
    data, xs = wide.to_numpy(float), wide.columns.to_numpy()

    def tval(d):
        n = np.sum(np.isfinite(d), 0)
        with np.errstate(all="ignore"):
            t = np.nanmean(d, 0) / (np.nanstd(d, 0, ddof=1) / np.sqrt(n))
        return t, 2 * stats.t.sf(np.abs(t), np.maximum(n - 1, 1))

    def clusters(t, p):
        out, i = [], 0
        while i < len(t):
            if not (np.isfinite(p[i]) and p[i] < alpha):
                i += 1
                continue
            j = i
            while j + 1 < len(t) and np.isfinite(p[j + 1]) and p[j + 1] < alpha and np.sign(t[j + 1]) == np.sign(t[i]):
                j += 1
            out.append((i, j, float(np.nansum(t[i:j + 1]))))
            i = j + 1
        return out

    obs = clusters(*tval(data))
    rng = np.random.default_rng(seed)
    null = np.array([max([abs(m) for *_, m in clusters(*tval(data * rng.choice([-1, 1], (len(data), 1))))] or [0])
                     for _ in range(n_perm)])
    return [{"start": float(xs[i]), "end": float(xs[j]), "mass": m, "p": float((1 + np.sum(null >= abs(m))) / (n_perm + 1))}
            for i, j, m in obs]


def _summary_vs_zero(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    w = stats.wilcoxon(v)
    return {"mean": float(v.mean()), "sem": float(stats.sem(v)), "n": len(v), "wilcoxon_W": float(w.statistic),
            "wilcoxon_p": float(w.pvalue), "t_p": float(stats.ttest_1samp(v, 0).pvalue)}


def _trial_lmm(h, formula, term):
    """Trial-based linear mixed model (user 2026-11-10): `formula` on single hits, random intercept and hit slope per
    mouse (optimisers lbfgs, powell, nm in turn; random intercept only if none converges); REML. Returns the fixed effect
    `term` (ms per 10 hits)."""
    import warnings
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    h = h.assign(hit10=(h.hit_number - 1) / 10)
    for re_formula, method in [("~hit10", m) for m in ("lbfgs", "powell", "nm")] + [("1", "lbfgs")]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                fit = smf.mixedlm(formula, h, groups=h.mouse_id, re_formula=re_formula).fit(reml=True, method=method)
                break
            except (ConvergenceWarning, np.linalg.LinAlgError):
                continue
    return {"slope": float(fit.params[term]), "se": float(fit.bse[term]), "z": float(fit.tvalues[term]),
            "p": float(fit.pvalues[term]), "intercept": float(fit.params["Intercept"]), "n_mice": int(h.mouse_id.nunique()),
            "n_trials": len(h), "random_effects": "intercept + slope" if re_formula == "~hit10" else "intercept", "optimiser": method}


def rt_change_from_start(ctx, trial_type):
    """Reaction-time change from the start of whisker day 0 (user 2026-11-10; explore_rt_time_b column 7), learners,
    first RT_HITS licked trials of `trial_type` with a reaction time: per trial, reaction time minus the mouse's median
    over its first RT_START; per mouse, the median per bin of RT_BIN (no smoothing; the first bin is 0 by construction).
    Per-mouse summaries on the unbinned values: late change = mean over hits RT_LATE; slope = least-squares slope on hit
    number, per 10 hits (mice with >= 10 hits); within cohort Wilcoxon signed-rank (and one-sample t) vs 0; R+ vs R-
    Mann-Whitney and Welch. Time-resolved, on the per-mouse bin medians (bins with >= RT_MIN_MICE mice per cohort):
    cohort vs 0 sign-flip cluster test on the bins after the reference bin; R+ vs R- cluster permutation test
    (cluster_test); 1,000 permutations, cluster P < 0.05.
    Rows: kind 'bin' (per mouse and bin; sig_vs0 / sig_cohorts mark significant clusters) and kind 'mouse'."""
    h = day0_hits(ctx)
    h = h[(h.trial_type == trial_type) & (h.hit_number <= RT_HITS)].dropna(subset=["rt_ms"])
    start = h[h.hit_number <= RT_START].groupby("mouse_id").rt_ms.median()
    h = h.assign(rt_change_ms=h.rt_ms - h.mouse_id.map(start), hit_bin=(h.hit_number - 1) // RT_BIN)
    h["hit_bin_centre"] = h.hit_bin * RT_BIN + (RT_BIN + 1) // 2
    keys = ["mouse_id", "session_id", "cohort"]
    bins = (h.groupby(keys + ["hit_bin", "hit_bin_centre"]).rt_change_ms.agg(value="median", n_hits_in_bin="size")
            .reset_index().assign(kind="bin", trial_type=trial_type))
    bins["hit_from"] = bins.hit_bin * RT_BIN + 1
    wide = {c: bins[bins.cohort == c].pivot_table(index="mouse_id", columns="hit_bin_centre", values="value")
            for c in COHORTS}
    wide = {c: w.loc[:, w.notna().sum() >= RT_MIN_MICE] for c, w in wide.items()}
    bins["sig_vs0"] = False
    clusters_vs0 = {}
    for c in COHORTS:
        w = wide[c]
        clusters_vs0[c] = sign_flip_clusters(w.loc[:, w.columns > w.columns.min()])  # after the reference bin
        bins.loc[bins.cohort == c, "sig_vs0"] = mark_clusters(bins[bins.cohort == c], "hit_bin_centre", clusters_vs0[c])
    both = pd.concat(wide, names=["cohort"]).stack().rename("v").reset_index()
    both = both[both.groupby("hit_bin_centre").cohort.transform("nunique") == 2]
    clusters_cohorts = cluster_test(both, "hit_bin_centre", "v")
    bins["sig_cohorts"] = mark_clusters(bins, "hit_bin_centre", clusters_cohorts)
    mice = []
    for (m, s, c), g in h.groupby(keys):
        b, a = np.polyfit((g.hit_number - 1) / 10, g.rt_change_ms, 1) if len(g) >= 10 else (np.nan, np.nan)
        mice.append(dict(mouse_id=m, session_id=s, cohort=c, kind="mouse", trial_type=trial_type, n_hits=len(g),
                         slope_ms_per_10_hits=b, intercept_ms=a,
                         late_change_ms=g[g.hit_number.between(*RT_LATE)].rt_change_ms.mean()))
    mice = pd.DataFrame(mice)
    tests = {v: {**{c: _summary_vs_zero(mice[mice.cohort == c][v]) for c in COHORTS},
                 "rplus_vs_rminus": two_group_tests(mice[mice.cohort == "R+"][v], mice[mice.cohort == "R-"][v])}
             for v in ("late_change_ms", "slope_ms_per_10_hits")}
    sig = {**{c: [k for k in clusters_vs0[c] if k["p"] < 0.05] for c in COHORTS},
           "rplus_vs_rminus": [k for k in clusters_cohorts if k["p"] < 0.05]}
    lmm = {c: _trial_lmm(h[h.cohort == c], "rt_change_ms ~ hit10", "hit10") for c in COHORTS}
    lmm["rplus_vs_rminus"] = _trial_lmm(h.assign(rminus=(h.cohort == "R-").astype(int)),
                                        "rt_change_ms ~ hit10 * rminus", "hit10:rminus")
    fit = pd.DataFrame([dict(kind="fit", trial_type=trial_type, cohort=c,  # panel: per-mouse hit-by-hit slopes
                             slope_ms_per_10_hits=mice[mice.cohort == c].slope_ms_per_10_hits.mean(),
                             intercept_ms=mice[mice.cohort == c].intercept_ms.mean(),
                             wilcoxon_p=tests["slope_ms_per_10_hits"][c]["wilcoxon_p"], lmm_p=lmm[c]["p"])
                        for c in COHORTS])
    return pd.concat([bins, mice, fit], ignore_index=True), {
        "n": n_by_cohort(mice), "late": tests["late_change_ms"], "slope": tests["slope_ms_per_10_hits"], "lmm": lmm,
        "clusters_vs0": clusters_vs0, "clusters_rplus_vs_rminus": clusters_cohorts,
        "n_sig_clusters": {k: len(v) for k, v in sig.items()},
        "sig_clusters_text": {k: "; ".join(f"hits {int(x['start']) - RT_BIN // 2}-{int(x['end']) + RT_BIN // 2}, P = {x['p']:.3g}" for x in v)
                              or "none" for k, v in sig.items()}}


def _rank_shift_null(x, y):
    """Spearman rho of x, y and the median rho over circular shifts of x by more than RT_SHIFT_EXCLUDE trials."""
    x, y = stats.rankdata(x), stats.rankdata(y)
    x, y = (x - x.mean()) / x.std(), (y - y.mean()) / y.std()
    n = len(x)
    sh = np.array([s for s in range(-n + 1, n) if RT_SHIFT_EXCLUDE < abs(s) < n - RT_SHIFT_EXCLUDE])
    if len(sh) > RT_N_SHIFT:
        sh = sh[np.linspace(0, len(sh) - 1, RT_N_SHIFT).astype(int)]
    null = [np.mean(np.roll(x, s) * y) for s in sh]
    return float(np.mean(x * y)), float(np.median(null)) if len(null) else np.nan


def compute_rt_learning(ctx):
    """Reaction time and learning (user 2026-11-09; left part replaced 2026-11-10). Learners, whisker day 0, early licks
    dropped; reaction time = computed_reaction_time (0-1 s) on licked trials.
    Left (kind 'bin' / 'mouse'): whisker-hit reaction time change from the mouse's first 5 hits (rt_change_from_start).
    Right (kind 'rho'): per mouse and licked trial type (whisker hits, auditory hits, false alarms; >= 10 per mouse),
    Spearman rho between reaction time and the whisker learning curve (at the trial; other trial types at the preceding
    whisker trial) minus the median rho over linear shifts (|shift| > 10 trials, hits in time order)."""
    rows = []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        curves, _, _ = session_curves(st, ctx)
        widx = np.cumsum(st.trial_type.to_numpy() == "whisker_trial") - 1
        curve = np.where(widx >= 0, curves.p_mean.to_numpy()[np.clip(widx, 0, len(curves) - 1)], np.nan)
        rt = (st.computed_reaction_time.where(st.computed_reaction_time.between(0, 1)) * 1000).to_numpy()
        base = dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
        for tt in TRIAL_TYPES:
            sel = (st.trial_type == tt).to_numpy() & (st.lick_flag == 1).to_numpy()
            r, c = rt[sel], curve[sel]
            ok = np.isfinite(r) & np.isfinite(c)
            if ok.sum() >= 10 and np.std(c[ok]) > 0:
                rho, null = _rank_shift_null(c[ok], r[ok])
                rows.append({**base, "kind": "rho", "trial_type": tt, "rho": rho, "shift_median": null,
                             "value": rho - null, "n_licked": int(ok.sum())})
    rho = pd.DataFrame(rows)
    stats_rho = {tt: {**{f"{c}_vs_0": vs_zero(rho[(rho.trial_type == tt) & (rho.cohort == c)].value) for c in COHORTS},
                      "rplus_vs_rminus": two_group_tests(rho[(rho.trial_type == tt) & (rho.cohort == "R+")].value,
                                                         rho[(rho.trial_type == tt) & (rho.cohort == "R-")].value)}
                 for tt in TRIAL_TYPES}
    change, info = rt_change_from_start(ctx, "whisker_trial")
    return pd.concat([change, rho], ignore_index=True), {"n": n_by_cohort(rho), "rt_change": info, "stats_rho": stats_rho}


def compute_first_lick(ctx):
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        w = st[(st.trial_type == "whisker_trial") & (st.early_lick == 0)].sort_values("start_time")
        licks = np.flatnonzero(w.lick_flag.to_numpy() == 1)
        rows.append(dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0],
                         whisker_trials_before_first_lick=licks[0] if len(licks) else np.nan))
    df = pd.DataFrame(rows)
    col = "whisker_trials_before_first_lick"
    return df, {"n": n_by_cohort(df), "stats": two_group_tests(df[df.cohort == "R+"][col], df[df.cohort == "R-"][col]),
                "no_whisker_lick": int(df[col].isna().sum()),
                "median": {c: float(df[df.cohort == c][col].median()) for c in COHORTS}}


# ---------------------------------------------------------------- row 2: example sessions

def compute_example(ctx, pid):
    mouse = PANELS[pid]["params"]["example"]
    st = day0_table(ctx["table"])
    st = st[st.mouse_id == mouse]
    curves, lt, cat = session_curves(st, ctx)
    df = curves.assign(mouse_id=mouse, session_id=st.session_id.iloc[0], cohort=st.cohort.iloc[0],
                       learning_category=st.learning_category.iloc[0])
    df = df[["mouse_id", "session_id", "cohort", "learning_category"] + list(curves.columns)]
    return df, {"mouse_id": mouse, "session_id": st.session_id.iloc[0], "cohort": st.cohort.iloc[0],
                "learning_category": st.learning_category.iloc[0], "n_whisker_trials": len(df),
                "whisker_hit_rate": float(df.lick.mean()),
                "fa_rate": float(st[(st.trial_type == "no_stim_trial") & (st.early_lick == 0)].lick_flag.mean()),
                "frac_trials_high": float((df.state == "high").mean()),
                "rule_learning_trial": None if np.isnan(lt) else int(lt) + 1, "rule_category": cat}


def compute_ex_rplus_learner(ctx):
    return compute_example(ctx, "ex_rplus_learner")


def compute_ex_rplus_nonlearner(ctx):
    return compute_example(ctx, "ex_rplus_nonlearner")


def compute_ex_rminus_learner(ctx):
    return compute_example(ctx, "ex_rminus_learner")


def compute_ex_rminus_nonlearner(ctx):
    return compute_example(ctx, "ex_rminus_nonlearner")


# ---------------------------------------------------------------- row 3: states, learners, session dynamics, movement

def _per_state(ctx):
    if "per_state" in ctx:
        return ctx["per_state"]
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time").reset_index(drop=True)
        curves, _, _ = session_curves(st, ctx)
        st["block_perf_type"] = pd.Series([None] * len(st), dtype=object)
        st.loc[st.trial_type == "whisker_trial", "block_perf_type"] = curves.state.to_numpy()
        st = learning_utils.propagate_expertise_inplace(st)  # other trial types: state of the nearest whisker trial
        keys = ["block_perf_type", "trial_type"]
        rate = st.groupby(keys).lick_flag.agg(["mean", "size"]).rename(columns={"mean": "lick_probability",
                                                                                "size": "n_trials"})
        licks = st[(st.lick_flag == 1) & (st.computed_reaction_time > 0)]
        rate["median_reaction_time_s"] = licks.groupby(keys).computed_reaction_time.median()
        rate["sd_reaction_time_s"] = licks.groupby(keys).computed_reaction_time.std()
        rate = rate.reset_index().rename(columns={"block_perf_type": "state"})
        rows.append(rate.assign(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0]))
    df = pd.concat(rows, ignore_index=True)[["mouse_id", "session_id", "cohort", "state", "trial_type",
                                             "lick_probability", "median_reaction_time_s", "sd_reaction_time_s",
                                             "n_trials"]]
    ctx["per_state"] = df
    return df


def _paired_states(df, value):
    tests = {}
    for c in COHORTS:
        for tt in TRIAL_TYPES:
            w = (df[(df.cohort == c) & (df.trial_type == tt) & df.state.isin(["low", "high"])]
                 .pivot_table(index="mouse_id", columns="state", values=value))
            w = w.dropna() if {"low", "high"} <= set(w.columns) else w.iloc[:0]
            if len(w) < 3:
                continue
            t = stats.ttest_rel(w.high, w.low)
            tests[f"{c} {tt}"] = {"n_mice": len(w), "wilcoxon_p": float(stats.wilcoxon(w.high, w.low).pvalue),
                                  "paired_t": float(t.statistic), "paired_t_p": float(t.pvalue)}
    return tests


def compute_lick_states(ctx):
    df = _per_state(ctx)
    return df, {"n": n_by_cohort(df), "stats_high_vs_low": _paired_states(df, "lick_probability")}


def compute_rt_states(ctx):
    df = _per_state(ctx)
    return df, {"n": n_by_cohort(df), "stats_high_vs_low": _paired_states(df, "median_reaction_time_s"),
                "stats_sd_high_vs_low": _paired_states(df, "sd_reaction_time_s")}


def compute_learners(ctx):
    d0 = day0_table(ctx["table"])
    mice = d0.groupby("mouse_id").agg(cohort=("cohort", "first"), learning_category=("learning_category", "first"),
                                      session_id=("session_id", "first")).reset_index()
    mice["learner"] = mice.learning_category.map(LEARNER)
    high = {}
    for sid, st in d0.groupby("session_id"):  # alternative definition (planner text): reaches a high state on day 0
        curves, _, _ = session_curves(st, ctx)
        high[st.mouse_id.iloc[0]] = bool((curves.state == "high").any())
    mice["reaches_high_state"] = mice.mouse_id.map(high)
    df = mice.groupby("cohort").agg(learners=("learner", "sum"), mice=("learner", "size"),
                                    reach_high_state=("reaches_high_state", "sum")).reset_index()
    df["fraction_learners"] = df.learners / df.mice
    df["fraction_reach_high_state"] = df.reach_high_state / df.mice
    for cat in ["good", "moderate", "bad"]:
        df[f"n_{cat}"] = [int((mice[mice.cohort == c].learning_category == cat).sum()) for c in df.cohort]
    table = [[int(r.learners), int(r.mice - r.learners)] for r in df.set_index("cohort").loc[COHORTS].itertuples()]
    res = stats.fisher_exact(table)
    table_h = [[int(r.reach_high_state), int(r.mice - r.reach_high_state)] for r in df.set_index("cohort").loc[COHORTS].itertuples()]
    res_h = stats.fisher_exact(table_h)
    agree = float((mice.learner == mice.reaches_high_state).mean())
    return df, {"n": n_by_cohort(mice), "stats": {"fisher_odds_ratio": float(res.statistic), "fisher_p": float(res.pvalue)},
                "stats_high_state": {"fisher_odds_ratio": float(res_h.statistic), "fisher_p": float(res_h.pvalue)},
                "sheet_vs_high_state_agreement": agree}


def compute_perf_session(ctx):
    """Learners and non-learners (planner 2026-11-06: learners left, non-learners right): performance (whisker - chance
    curve) against session time normalised from the first to the last whisker trial; cluster-based permutation test
    R+ vs R- within each group."""
    grid = np.linspace(0, 1, 51)
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        curves, _, _ = session_curves(st, ctx)
        tw = st[st.trial_type == "whisker_trial"].start_time.to_numpy()
        x = (tw - tw[0]) / (tw[-1] - tw[0])
        perf = np.interp(grid, x, (curves.p_mean - curves.p_chance).to_numpy())
        learner = bool(LEARNER.get(st.learning_category.iloc[0], False))
        rows.append(pd.DataFrame({"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                                  "group": "learner" if learner else "non-learner",
                                  "session_time_norm": np.round(grid, 3), "performance": perf}))
    df = pd.concat(rows, ignore_index=True)
    df["cluster_sig"] = False
    clusters, n = {}, {}
    for grp, d in df.groupby("group"):
        clusters[grp] = cluster_test(d, "session_time_norm", "performance")
        df.loc[d.index, "cluster_sig"] = mark_clusters(d, "session_time_norm", clusters[grp])
        n[grp] = n_by_cohort(d)
    return df, {"n": n, "clusters": clusters}


def compute_movement_peth(ctx):
    """Learners with dual-view video: movement traces per session, trial type, lick / no lick, aligned at the stimulus
    and at the lick (shared_movement_peth v4); cluster test R+ vs R- within each signal x trial type x lick x alignment
    (planner 2026-11-06: mice, within trial type, across cohorts)."""
    import shared_movement_peth
    peth = shared_movement_peth.load()
    learners = set(day0_table(ctx["table"], learners_only=True).mouse_id)
    df = peth[peth.mouse_id.isin(learners)]
    # planner 2026-11-06 (re-arranged): stimulus-aligned only, auditory misses not shown
    df = df[(df["align"] == "stimulus") & ~((df.trial_type == "auditory_trial") & (df.lick == 0))].reset_index(drop=True)
    tests, n = {}, {}
    df["cluster_sig"] = False
    for (sig, tt, lick, align), d in df.groupby(["signal", "trial_type", "lick", "align"]):
        key = f"{sig} {tt} lick={lick} {align}"
        tests[key] = cluster_test(d, "time_s", "value_delta")
        df.loc[d.index, "cluster_sig"] = mark_clusters(d, "time_s", tests[key])
        n[key] = n_by_cohort(d)
    return df, {"n": n_by_cohort(df), "n_by_group": n, "clusters": tests, "params": shared_movement_peth.PARAMS}


# ---------------------------------------------------------------- row 4: false alarms, curve correlations, posture

def compute_aligned_curves(ctx):
    """Whisker, false-alarm (no-stimulus) and auditory learning, all aligned at the first whisker hit (planner
    2026-11-09: 'show data aligned at 1st whisker hit, not 1st lick of that trial type'). Each trial type stays on its own
    trials: trial 0 of whisker is the first whisker hit; for no-stimulus and auditory trials, trial 0 is the first trial
    of that type after the first whisker hit (negative indices, the trials before it). Data: the lick outcome of every trial
    of that type (no binning; as the data of behaviour_analysis plot_multi_mouse_session_aligned, mean_performance_over_
    trials_day0), mean +- s.e.m. over mice per trial index; whisker from trial 0 only. Curves: exact curve of that trial
    type on its own trials (learning_curve_exact). Cluster test R+ vs R- per trial type on the curves."""
    lo, hi = PANELS["aligned_curves"]["params"]["window_trials"]
    rows = []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):  # planner: learners only
        s0 = st[st.early_lick == 0].sort_values("start_time")
        out = lce.session_learning_curves_exact(s0, sigma=CURVE_PARAMS["sigma"], conf_int=CURVE_PARAMS["conf_int"])
        if "whisker_trial" not in out or not (out["whisker_trial"]["outcomes"] == 1).any():
            continue
        w = out["whisker_trial"]
        t_hit = w["start_time"][np.argmax(w["outcomes"] == 1)]
        base = dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
        for tt in ("whisker_trial", "no_stim_trial", "auditory_trial"):
            if tt not in out:
                continue
            y = np.asarray(out[tt]["outcomes"])
            times = np.asarray(out[tt]["start_time"])
            rel = np.arange(len(y)) - int(np.sum(times < t_hit))  # 0: first trial of the type at/after the first W hit
            keep = (rel >= (0 if tt == "whisker_trial" else lo)) & (rel <= hi)
            rows.append(pd.DataFrame({**base, "trial_type": tt, "kind": "curve", "trial_from_first_whisker_hit": rel[keep],
                                      "p_lick": np.asarray(out[tt]["p_mean"])[keep]}))
            rows.append(pd.DataFrame({**base, "trial_type": tt, "kind": "data", "trial_from_first_whisker_hit": rel[keep],
                                      "p_lick": y[keep].astype(float)}))
    df = pd.concat(rows, ignore_index=True)
    df["cluster_sig"] = False
    clusters = {}
    for tt in ("whisker_trial", "no_stim_trial", "auditory_trial"):
        cur = df[(df.kind == "curve") & (df.trial_type == tt)]
        clusters[tt] = cluster_test(cur, "trial_from_first_whisker_hit", "p_lick")
        df.loc[cur.index, "cluster_sig"] = mark_clusters(cur, "trial_from_first_whisker_hit", clusters[tt])
    return df, {"n": n_by_cohort(df), "clusters": clusters}


FIRST_HIT_SUBPANELS = {"whisker": "whisker_trial", "spont_licks": None, "auditory": "auditory_trial"}  # plot order


def _ranges_text(starts, width=1):
    """'0 to 1 and 3 to 20' from sorted bin starts (contiguous bins merged); 'none' if empty."""
    out = []
    for x in sorted(int(v) for v in starts):
        if out and x == out[-1][1]:
            out[-1][1] = x + width
        else:
            out.append([x, x + width])
    if not out:
        return "none"
    return (" and " if len(out) < 3 else ", ").join(f"{a} to {b}" for a, b in out)


def _first_hit_tests(fa, kind, b, pp, vs_pre=True):
    """Bin tests (R+ vs R- per bin, Mann-Whitney + Welch, BH over bins) and pre / post tests of one sub-panel, as in
    explore_fa_divergence. vs_pre False (whisker: pre = 0 by definition, all misses): no bin-vs-pre, post-vs-pre or change
    tests, only R+ vs R- on post."""
    bins = fa.bin_tests(kind, b, pp)
    if not vs_pre:
        bins = bins.drop(columns=[c for c in bins if "_vs_pre" in c or "_n_paired" in c])
        pp = pp.dropna(subset=["post"])
        tests = {"R+ vs R- post": two_group_tests(pp[pp.cohort == "R+"].post, pp[pp.cohort == "R-"].post)}
    else:
        pp, t = fa.prepost_tests(kind, pp)
        tests = {r.test.replace(" (Wilcoxon signed-rank)", ""): {k: v for k, v in r.items() if pd.notna(v) and k != "row"}
                 for _, r in t.iterrows()}
    post = bins[bins.bin >= 0]
    info = {"prepost": tests,
            "pre_post": {c: {"n": int((pp.cohort == c).sum()), "pre_mean": float(pp[pp.cohort == c].pre.mean()),
                             "pre_sem": float(pp[pp.cohort == c].pre.sem()), "post_mean": float(pp[pp.cohort == c].post.mean()),
                             "post_sem": float(pp[pp.cohort == c].post.sem())} for c in COHORTS},
            "bins_sig_after_hit": int((post.mannwhitney_q_bh < 0.05).sum()), "bins_after_hit": int(len(post)),
            "bins_sig_before_hit": int((bins[bins.bin < 0].mannwhitney_q_bh < 0.05).sum()),
            "sig_minutes": [int(x) for x in bins[bins.mannwhitney_q_bh < 0.05].bin],
            "sig_minutes_text": _ranges_text(bins[bins.mannwhitney_q_bh < 0.05].bin),
            "bin_tests": bins.drop(columns=["row"]).to_dict(orient="records")}
    return bins, info


def compute_first_hit_licks(ctx):
    """Whisker | spontaneous licks | auditory around the first whisker hit (user 2026-11-10: 'like for FA_divergence'),
    all with the explore_fa_divergence time-course logic. Untrimmed whisker day 0 (auditory warm-up block kept, so every
    mouse has a pre-hit baseline), learners, active trials; first whisker hit = first licked whisker trial; 1-min bins
    from it (-6..20 min). Whisker and auditory: fraction of licked trials of the type per mouse and bin (row 2 'fa_time'
    of explore_fa_divergence for these trial types; early-lick trials dropped), pre -5..0 min (>= 3 trials), post
    0..5 min. Spontaneous licks (row 4 'licks', unchanged from the former spont_licks panel): NWB piezo lick times outside
    every trial's 0-1 s window and, for rewarded trials (licked auditory; licked whisker in R+), outside trial start to
    5 s after the reward; per minute of time outside those windows, bins >= 20 s covered, pre / post >= 60 s covered.
    R+ vs R- per bin (Mann-Whitney, BH over bins; Welch saved); pre vs post per cohort (Wilcoxon signed-rank), R+ vs R-
    on pre, post and change; whisker: post only (pre = 0 by definition)."""
    import explore_fa_divergence as fa
    trials, sessions = fa.session_tables(fa.day0_active(), trial_types=("whisker_trial", "auditory_trial"))
    sid = {s["mouse_id"]: s["session_id"] for s in sessions}
    rows, info = [], {}
    for sub, tt in FIRST_HIT_SUBPANELS.items():
        if tt is None:
            kind, measure = "licks", "spontaneous_licks_per_min"
            b, pp = fa.per_mouse(kind, None, sessions)
        else:
            kind, measure = "fa_time", "lick_probability"
            tr = trials[trials.trial_type == tt]
            b, pp = fa.per_mouse(kind, tr, None)
            n_tr = tr.assign(bin=np.floor(tr.minutes)).groupby(["mouse_id", "bin"]).size().rename("n_trials")
            b = b.merge(n_tr.reset_index(), on=["mouse_id", "bin"], how="left")
        bins, info[sub] = _first_hit_tests(fa, kind, b, pp, vs_pre=sub != "whisker")
        info[sub]["n"] = n_by_cohort(b.assign(session_id=b.mouse_id.map(sid)))
        info[sub]["measure"] = measure
        q = bins[["bin", "mannwhitney_q_bh", "welch_q_bh"]].rename(columns={"mannwhitney_q_bh": "cohort_q_bh",
                                                                           "welch_q_bh": "cohort_welch_q_bh"})
        rows.append(b.merge(q, on="bin", how="left").assign(sub_panel=sub, trial_type=tt or "spontaneous", measure=measure))
    df = pd.concat(rows, ignore_index=True).rename(columns={"bin": "minute"})
    df["session_id"] = df.mouse_id.map(sid)
    df["cohort_sig"] = df.cohort_q_bh < 0.05
    df = df[["mouse_id", "session_id", "cohort", "sub_panel", "trial_type", "measure", "minute", "value", "n_trials",
             "cohort_q_bh", "cohort_welch_q_bh", "cohort_sig"]]
    info["n"] = n_by_cohort(df)
    info["params"] = {"response_window_s": fa.RESP_WIN, "reward_exclusion_s": fa.REWARD_EXCL, "bin_min": 1,
                      "edges_min": [int(fa.ROWS["licks"]["edges"][0]), int(fa.ROWS["licks"]["edges"][-1])],
                      "pre_min": [-5, 0], "post_min": [0, 5], "min_cover_s": fa.MIN_COVER,
                      "min_pre_trials": fa.ROWS["fa_time"]["min_pre"], "min_mice_per_cohort_bin_test": fa.MIN_MICE}
    return df, info


LICK_COURSE_EDGES = np.round(np.arange(-0.21, 1.5 + 1e-9, 0.03), 3)  # 30-ms bins, 0 on an edge (user 2026-11-10)


def compute_lick_course(ctx):
    """Jaw opening and piezo licks after the stimulus on whisker and auditory lick trials (user 2026-11-10). Learners,
    whisker day 0, early licks dropped. Jaw (measure 'jaw'): DLC jaw_distance PETH (shared_movement_peth, dlc_helpers,
    mice with dual-view video), each trial minus its pre-stimulus mean, mean over trials per session; -0.1 to 0.3 s,
    5-ms steps. Piezo (measure 'piezo'): per lick trial, whether at least one NWB piezo lick falls in each 30-ms bin
    from -0.21 to 1.5 s, mean over trials per session. Per mouse: mean over sessions. Column value: mm (jaw) or
    probability (piezo); time_s = bin centre / sample time."""
    from multiprocessing import Pool
    import shared_movement_peth
    from explore_lick_proba import session_licks
    d0 = day0_table(ctx["table"], learners_only=True)
    d0 = d0[(d0.early_lick == 0) & (d0.lick_flag == 1) & d0.trial_type.isin(["whisker_trial", "auditory_trial"])]
    jaw = shared_movement_peth.load()
    jaw = jaw[(jaw.signal == "jaw_opening") & (jaw["align"] == "stimulus") & (jaw.lick == 1)
              & jaw.trial_type.isin(["whisker_trial", "auditory_trial"]) & jaw.session_id.isin(set(d0.session_id))]
    rows = [jaw.assign(measure="jaw", value=jaw.value_delta)[["mouse_id", "session_id", "cohort", "measure", "trial_type",
                                                              "time_s", "value", "n_trials"]]]
    with Pool(16) as pool:
        piezo = {sid: p for sid, p, _ in pool.map(session_licks, list(d0.session_id.unique()))}
    edges, centre = LICK_COURSE_EDGES, np.round(LICK_COURSE_EDGES[:-1] + 0.015, 3)
    for sid, st in d0.groupby("session_id"):
        lk = piezo[sid]
        if not len(lk):
            continue
        for tt, g in st.groupby("trial_type"):
            m = np.array([np.histogram(lk[np.searchsorted(lk, t + edges[0]):np.searchsorted(lk, t + edges[-1])] - t,
                                       edges)[0] > 0 for t in g.start_time.to_numpy()])
            rows.append(pd.DataFrame({"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                                      "measure": "piezo", "trial_type": tt, "time_s": centre, "value": m.mean(0),
                                      "n_trials": len(g)}))
    d = pd.concat(rows, ignore_index=True)
    d = (d.groupby(["mouse_id", "cohort", "measure", "trial_type", "time_s"])
         .agg(session_id=("session_id", lambda v: ";".join(sorted(set(v)))), value=("value", "mean"),
              n_trials=("n_trials", "sum"))
         .reset_index())
    return d, {"n": {m: n_by_cohort(d[d.measure == m]) for m in ("jaw", "piezo")}, "piezo_bin_ms": 30}


def _anchors(ctx):
    """Early-FA anchors, as in behaviour_analysis beh_plotting_functions.plot_fa_rate_day_change and
    plot_cumulative_fa_vs_whisker_trials. Per mouse: day-0 session (untrimmed, so warm-up auditory rewards count) in
    start-time order, the index of its first whisker lick and the auditory rewards before it; day -1 session and the
    index where the same number of auditory rewards is reached (0 rewards: trial 0). Early licks removed."""
    if "fa_anchors" in ctx:
        return ctx["fa_anchors"]
    d0 = shared_behaviour.load_untrimmed_day0()
    d0 = d0[d0.early_lick == 0]
    dm1 = ctx["table"][(ctx["table"].behavior == "auditory") & (ctx["table"].day == -1) & (ctx["table"].early_lick == 0)]
    out = {}
    for mouse, s0 in d0.groupby("mouse_id"):
        s0 = s0.sort_values("start_time").reset_index(drop=True)
        wl = s0.index[(s0.trial_type == "whisker_trial") & (s0.lick_flag == 1)]
        if len(wl) == 0:
            continue
        a0 = wl[0]
        n_aud = int(((s0.index < a0) & (s0.trial_type == "auditory_trial") & (s0.lick_flag == 1)).sum())
        rec = {"cohort": s0.cohort.iloc[0], "day0": s0, "anchor0": a0, "n_aud": n_aud}
        sm = dm1[dm1.mouse_id == mouse]
        if len(sm):
            sm = sm.sort_values("start_time").reset_index(drop=True)
            hits = sm.index[(sm.trial_type == "auditory_trial") & (sm.lick_flag == 1)]
            if n_aud == 0:
                rec.update(daym1=sm, anchor_m1=0)
            elif len(hits) >= n_aud:
                rec.update(daym1=sm, anchor_m1=hits[n_aud - 1])
        out[mouse] = rec
    ctx["fa_anchors"] = out
    return out


def compute_cumulative_fa(ctx):
    """Cumulative false alarms after the anchor: day 0 per whisker trial after the first whisker lick; day -1 per
    no-stimulus trial after the matched auditory-reward count."""
    n_trials = PANELS["cumulative_fa"]["params"]["n_trials"]
    rows = []
    for mouse, r in _anchors(ctx).items():
        post = r["day0"][r["day0"].index > r["anchor0"]]
        whisk = post.index[post.trial_type == "whisker_trial"]
        if len(whisk) >= n_trials:
            ns = post[post.trial_type == "no_stim_trial"]
            rows += [dict(mouse_id=mouse, cohort=r["cohort"], day=0, trial_after_anchor=i + 1,
                          cumulative_fa=int(ns[ns.index <= whisk[i]].lick_flag.sum()), n_aud_rewards_at_anchor=r["n_aud"])
                     for i in range(n_trials)]
        if "daym1" in r:
            ns = r["daym1"][(r["daym1"].index > r["anchor_m1"]) & (r["daym1"].trial_type == "no_stim_trial")]
            if len(ns) >= n_trials:
                cum = ns.lick_flag.iloc[:n_trials].cumsum().to_numpy()
                rows += [dict(mouse_id=mouse, cohort=r["cohort"], day=-1, trial_after_anchor=i + 1, cumulative_fa=int(c),
                              n_aud_rewards_at_anchor=r["n_aud"]) for i, c in enumerate(cum)]
    df = pd.DataFrame(rows)
    last = df[df.trial_after_anchor == n_trials]
    tests = {f"day{d}": two_group_tests(last[(last.day == d) & (last.cohort == "R+")].cumulative_fa,
                                       last[(last.day == d) & (last.cohort == "R-")].cumulative_fa) for d in (0, -1)}
    return df, {"n": {f"day{d}": n_by_cohort(df[df.day == d]) for d in (0, -1)}, f"stats_at_{n_trials}": tests}


# Lagged correlation and linear-shift null as in behaviour_analysis/beh_correlates/beh_correlates_lick_decoding.py
LAGS = np.arange(-10, 11)
N_SHIFTS, SHIFT_EXCLUDE = 500, 10


def lagged_corr(x, y, lag):
    """Pearson r of x and y with y shifted by `lag` trials (positive: x leads y)."""
    if lag > 0:
        x, y = x[:-lag], y[lag:]
    elif lag < 0:
        x, y = x[-lag:], y[:lag]
    ok = np.isfinite(x) & np.isfinite(y)
    return stats.pearsonr(x[ok], y[ok])[0] if ok.sum() > 4 else np.nan


def shift_null(x, y, rng):
    """Correlations of x with y linearly shifted by more than SHIFT_EXCLUDE trials (keeps each series' autocorrelation)."""
    n = len(y)
    valid = [s for s in range(1, n) if min(s, n - s) > SHIFT_EXCLUDE] or list(range(1, n))
    return np.array([lagged_corr(x, y, s) for s in rng.choice(valid, size=N_SHIFTS)])


def compute_xcorr(ctx):
    """Learners: cross-correlation of the whisker and chance curves, with a linear-shift null at lag 0."""
    rng = np.random.default_rng(0)
    rows = []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        curves, _, _ = session_curves(st, ctx)
        w, fa = curves.p_mean.to_numpy(), curves.p_chance.to_numpy()
        null = shift_null(w, fa, rng)
        for lag in LAGS:
            rows.append(dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0], lag=int(lag),
                             r=lagged_corr(w, fa, lag), null_mean_lag0=float(np.nanmean(null)),
                             null_sd_lag0=float(np.nanstd(null))))
    df = pd.DataFrame(rows)
    lag0 = df[df.lag == 0].assign(r_minus_null=lambda d: d.r - d.null_mean_lag0)
    vs_null = {c: {"median_r": float(lag0[lag0.cohort == c].r.median()),
                   **vs_zero(lag0[lag0.cohort == c].r_minus_null)} for c in COHORTS}
    return df, {"n": n_by_cohort(df), "lag0_vs_shift_null": vs_null,
                "lag0_rplus_vs_rminus": two_group_tests(lag0[lag0.cohort == "R+"].r, lag0[lag0.cohort == "R-"].r)}


def _movement_sessions(ctx):
    """Learner day-0 sessions with video: exact curves, reaction times and pre-stimulus movements, trial by trial."""
    if "movement_sessions" in ctx:
        return ctx["movement_sessions"]
    import shared_movements
    mov = shared_movements.load()
    out = {}
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        m = mov[mov.session_id == sid].sort_values("start_time")
        if m.empty:
            continue
        curves, _, _ = session_curves(st, ctx)
        w = st[(st.trial_type == "whisker_trial") & (st.early_lick == 0)].sort_values("start_time")
        if not (len(curves) == len(m) == len(w)):
            print("skip", sid, "curve, movement and trial counts differ", len(curves), len(m), len(w))
            continue
        rt = np.where((w.lick_flag.to_numpy() == 1) & (w.computed_reaction_time.to_numpy() > 0),
                      w.computed_reaction_time.to_numpy(), np.nan)
        out[sid] = pd.concat([curves.reset_index(drop=True),
                              m[["mouse_id", "session_id", "cohort", "trial_id"] + MOVEMENT_COLS].reset_index(drop=True)],
                             axis=1).assign(reaction_time_s=rt)
    ctx["movement_sessions"] = out
    return out


def compute_movement_example(ctx):
    mouse = PANELS["movement_example"]["params"]["example"]
    sessions = _movement_sessions(ctx)
    df = next(d for d in sessions.values() if d.mouse_id.iloc[0] == mouse).copy()
    win = PANELS["movement_example"]["params"]["smoothing_trials"]
    for c in MOVEMENT_COLS:
        z = (df[c] - df[c].mean()) / df[c].std()
        df[f"{c}_z"] = z
        df[f"{c}_z_smooth"] = z.rolling(win, center=True, min_periods=1).mean()
    df["performance"] = df.p_mean - df.p_chance
    return df, {"mouse_id": mouse, "n_whisker_trials": len(df), "smoothing_trials": win}


def _corr_vs_null(ctx, target):
    """Per session and movement: Pearson r between `target` and the movement across whisker trials, minus the mean
    of a linear-shift null."""
    rng = np.random.default_rng(0)
    rows = []
    for sid, d in _movement_sessions(ctx).items():
        y = (d.p_mean - d.p_chance).to_numpy() if target == "performance" else d[target].to_numpy(float)
        for c in MOVEMENT_COLS:
            x = d[c].to_numpy(float)
            if (np.isfinite(x) & np.isfinite(y)).sum() < 20:
                continue
            null = shift_null(y, x, rng)
            r = lagged_corr(y, x, 0)
            rows.append(dict(mouse_id=d.mouse_id.iloc[0], session_id=sid, cohort=d.cohort.iloc[0], movement=c, r=r,
                             null_mean=float(np.nanmean(null)), null_sd=float(np.nanstd(null)),
                             r_minus_null=r - float(np.nanmean(null))))
    df = pd.DataFrame(rows)
    tests = {}
    for c in MOVEMENT_COLS:
        d = df[df.movement == c]
        tests[c] = {"rplus_vs_rminus": two_group_tests(d[d.cohort == "R+"].r_minus_null, d[d.cohort == "R-"].r_minus_null),
                    **{f"{k}_vs_null": vs_zero(d[d.cohort == k].r_minus_null) for k in COHORTS}}
    return df, {"n": n_by_cohort(df), "stats": tests}


def compute_movement_perf(ctx):
    return _corr_vs_null(ctx, "performance")


def compute_movement_rt(ctx):
    return _corr_vs_null(ctx, "reaction_time_s")


# ---------------------------------------------------------------- row 5: behavioural GLM (glm_v4_run.py)
# glm_v4 (user 2026-11-09): glm_v3 design matrix refitted with contiguous-block CV (gap 3 whisker trials, 10 repeats with
# shifted blocks), no class weighting (calibrated probabilities); same file names as glm_v3

def _glm_dir():
    from common import CACHE_DIR
    # user 2026-11-09: glm_v5 (5 outer blocks, drop-and-refit delta LL) with one penalty per session chosen on the other
    # sessions (leave-one-session-out) and a recent-licking baseline model (glm_v6_run.py)
    return CACHE_DIR / "glm_v6"


GLM_ORDER = ["cumulative_reward_fraction", "time_since_reward", "whisker_speed_pre", "nose_speed_pre", "pupil_area_pre",
             "jaw_speed_pre", "spontaneous_lick_rate", "lick_whisker_trial_n-3", "lick_whisker_trial_n-2",
             "lick_whisker_trial_n-1"]  # planner panel o, 2026-11-09


def _glm_feature(name):
    return name.replace("Shuffled ", "").replace("Without ", "").removesuffix("_t0")


def _glm_files(ctx, kind):
    d0 = day0_table(ctx["table"])
    sessions = d0.groupby("session_id").agg(mouse_id=("mouse_id", "first"), cohort=("cohort", "first")).reset_index()
    out = []
    for r in sessions.itertuples():
        f = _glm_dir() / r.mouse_id / f"{r.mouse_id}_{r.session_id}_{kind}"
        if f.exists():
            out.append((r, f))
    return out


def _glm_example_mouse(ctx):
    """A well predicted but typical R+ learner session (user 2026-11-09: not the best): among sessions with >= 100
    whisker trials, the one whose test AUC is closest to the 75th percentile of R+ learner sessions."""
    metrics = pd.DataFrame([dict(mouse_id=r.mouse_id, test_auc=pd.read_csv(f).test_auc.iloc[0],
                                 n_trials=pd.read_csv(f).n_trials.iloc[0])
                            for r, f in _glm_files(ctx, "results_metrics.csv") if r.cohort == "R+"])
    q75 = metrics.test_auc.quantile(0.75)
    m = metrics[metrics.n_trials >= 100]
    return m.iloc[(m.test_auc - q75).abs().argsort()].mouse_id.iloc[0]


def compute_glm_schematic(ctx):
    """Design-matrix schematic (user 2026-11-10): the session of the GLM example (_glm_example_mouse), its whisker-trial
    design matrix rebuilt as in glm_v6_run (glm_v5_run.load_session on Myriam Hamon's module), each predictor z-scored
    over the session for display (the fits standardise within the training blocks), and the lick outcome."""
    import glm_v5_run as v5
    from common import NWB_DIR
    mouse = _glm_example_mouse(ctx)
    r = next(r for r, _ in _glm_files(ctx, "results.pkl") if r.mouse_id == mouse)
    X, y, _, names = v5.load_session(v5.glm_module(v5.glm_source_dir()), str(NWB_DIR / f"{r.session_id}.nwb"))
    with np.errstate(all="ignore"):
        z = np.nan_to_num((X - X.mean(0)) / X.std(0))
    df = (pd.DataFrame(z, columns=names).assign(whisker_trial=np.arange(1, len(y) + 1), lick=y)
          .melt(id_vars=["whisker_trial", "lick"], var_name="feature", value_name="z"))
    df.insert(0, "cohort", r.cohort)
    df.insert(0, "session_id", r.session_id)
    df.insert(0, "mouse_id", mouse)
    return df, {"mouse_id": mouse, "session_id": r.session_id, "n_whisker_trials": int(len(y)), "features": list(names)}


def compute_glm_example(ctx):
    """Well predicted, typical R+ learner session (_glm_example_mouse): held-out prediction of each trial averaged over
    the 10 repeats of the cross-validation (each repeat: the model of the test block containing the trial), with the
    spread over repeats (min-max)."""
    import pickle
    mouse = _glm_example_mouse(ctx)
    r, f = next((r, f) for r, f in _glm_files(ctx, "results.pkl") if r.mouse_id == mouse)
    res = pickle.load(open(f, "rb"))
    y, pred = np.asarray(res["y_all"]), np.asarray(res["y_pred_proba_oof"])
    pr = pd.read_parquet(str(f).replace("results.pkl", "predictions.parquet")).pivot(index="trial", columns="repeat",
                                                                                     values="p_lick")
    df = pd.DataFrame({"mouse_id": mouse, "session_id": r.session_id, "cohort": r.cohort,
                       "whisker_trial": np.arange(1, len(y) + 1), "lick": y, "predicted_p_lick": pred,
                       "predicted_min": pr.min(axis=1).to_numpy(), "predicted_max": pr.max(axis=1).to_numpy(),
                       "observed_rolling10": pd.Series(y).rolling(10, center=True, min_periods=3).mean(),
                       "test_balanced_accuracy": float(res["result"].get("test_balanced_accuracy", np.nan))})
    return df, {"mouse_id": mouse, "n_whisker_trials": len(df), "test_auc": float(res["result"]["test_auc"]),
                "test_balanced_accuracy": float(res["result"].get("test_balanced_accuracy", np.nan))}


def compute_glm_fit(ctx):
    """Cross-validated balanced accuracy (planner: not AUC)."""
    rows = [pd.read_csv(f).assign(mouse_id=r.mouse_id, session_id=r.session_id, cohort=r.cohort)
            for r, f in _glm_files(ctx, "results_metrics.csv")]
    df = pd.concat(rows, ignore_index=True)[["mouse_id", "session_id", "cohort", "test_balanced_accuracy", "test_auc",
                                             "test_pseudo_r2", "n_trials", "lick_rate", "C_star", "baseline_test_auc",
                                             "baseline_test_ll_per_trial", "test_ll_per_trial", "delta_ll_vs_baseline",
                                             "delta_auc_vs_baseline"]]
    df["is_example"] = df.mouse_id == _glm_example_mouse(ctx)  # circled: the mouse of the GLM example panel
    ba = {c: df[df.cohort == c].test_balanced_accuracy.dropna() for c in COHORTS}
    vs_chance = {c: {"median": float(ba[c].median()), **vs_zero(ba[c] - 0.5)} for c in COHORTS}
    vs_base = {c: {k: {"mean": float(df[df.cohort == c][k].mean()), **vs_zero(df[df.cohort == c][k].dropna())}
                   for k in ("delta_ll_vs_baseline", "delta_auc_vs_baseline")} for c in COHORTS}
    return df, {"n": n_by_cohort(df), "ba_vs_chance": vs_chance, "vs_baseline": vs_base,
                "C_star": sorted(float(c) for c in df.C_star.dropna().unique()),
                "stats": two_group_tests(ba["R+"], ba["R-"])}


def _feature_tests(df, value):
    return {f: {"rplus_vs_rminus": two_group_tests(df[(df.feature == f) & (df.cohort == "R+")][value],
                                                   df[(df.feature == f) & (df.cohort == "R-")][value]),
                **{f"{c}_vs_0": vs_zero(df[(df.feature == f) & (df.cohort == c)][value]) for c in COHORTS}}
            for f in df.feature.unique()}


def compute_glm_weights(ctx):
    rows = [pd.read_csv(f).assign(mouse_id=r.mouse_id, session_id=r.session_id, cohort=r.cohort)
            for r, f in _glm_files(ctx, "coefficients.csv")]
    df = pd.concat(rows, ignore_index=True)
    df["feature"] = df.feature.map(_glm_feature)
    df = df[["mouse_id", "session_id", "cohort", "feature", "coefficient"]]
    # planner 2026-11-09 (panel n note): fixed order, top to bottom; predictors not listed follow by mean |weight|
    by_mag = df.groupby("feature").coefficient.apply(lambda v: v.abs().mean()).sort_values(ascending=False).index
    order = [f for f in GLM_ORDER if f in set(df.feature)] + [f for f in by_mag if f not in GLM_ORDER]
    df["rank_by_mean_abs"] = df.feature.map({f: i for i, f in enumerate(order)})
    ctx["glm_order"] = list(order)
    return df, {"n": n_by_cohort(df), "order": list(order), "stats": _feature_tests(df, "coefficient")}


def compute_glm_weights_auc(ctx):
    """Planner 2026-11-06 (panel p): GLM weights, then the same for the shuffle importance (delta AUC), side by side,
    predictors in the order of mean absolute weight. Long format: metric = weight | delta_auc."""
    w, w_info = compute_glm_weights(ctx)
    s, s_info = compute_glm_shuffle(ctx)
    rank = lambda d: d.feature.map({f: i for i, f in enumerate(w_info["order"])}).fillna(len(w_info["order"]))
    keep = ["mouse_id", "session_id", "cohort", "feature"]
    rows = []  # drop-and-refit: delta held-out LL per trial when the predictor is removed (nested file)
    for r, f in _glm_files(ctx, "nested_model_comparison.csv"):
        nm = pd.read_csv(f)
        nm = nm[nm.model != "Full Model"]
        rows.append(pd.DataFrame({"mouse_id": r.mouse_id, "session_id": r.session_id, "cohort": r.cohort,
                                  "feature": nm.model.map(_glm_feature), "value": nm.delta_test_ll_per_trial.to_numpy()}))
    drop = pd.concat(rows, ignore_index=True)
    df = pd.concat([w.rename(columns={"coefficient": "value"}).assign(metric="weight"),
                    s[keep + ["delta_test_auc"]].rename(columns={"delta_test_auc": "value"})
                     .assign(metric="delta_auc", rank_by_mean_abs=rank),
                    drop.assign(metric="delta_ll_drop", rank_by_mean_abs=rank)], ignore_index=True)
    return df, {"n": w_info["n"], "order": w_info["order"], "stats_weight": w_info["stats"],
                "stats_delta_auc_shuffle": _feature_tests(s, "delta_test_auc"),
                "stats_delta_ll_drop": _feature_tests(drop, "value")}


def compute_glm_ll(ctx):
    """Change in held-out log-likelihood when each predictor (or all movement predictors) is removed and the model
    refitted (nested comparison of glm_behavior_v3). Its files give the change in test pseudo-R2 (McFadden); in nats
    per held-out trial, delta LL = delta pseudo-R2 x |null LL per trial|, with null LL = LL / (1 - pseudo-R2) of the
    full model (results_metrics: test_ll, summed over a test fold; test fold = n_trials / n_folds trials)."""
    metrics = {r.session_id: pd.read_csv(f).iloc[0] for r, f in _glm_files(ctx, "results_metrics.csv")}
    rows = []
    for r, f in _glm_files(ctx, "nested_model_comparison.csv"):
        if r.session_id not in metrics:
            continue
        m = metrics[r.session_id]
        null_ll_per_trial = abs(m.test_ll / (1 - m.test_pseudo_r2)) / (m.n_trials / m.n_folds)
        nested = pd.read_csv(f)
        nested = nested[nested.model != "Full Model"]
        rows.append(pd.DataFrame({"mouse_id": r.mouse_id, "session_id": r.session_id, "cohort": r.cohort,
                                  "feature": nested.model.map(_glm_feature),
                                  "delta_test_pseudo_r2": nested.delta_test_pseudo_r2.to_numpy(),
                                  "delta_ll_nats_per_trial": nested.delta_test_pseudo_r2.to_numpy() * null_ll_per_trial}))
    df = pd.concat(rows, ignore_index=True)
    order = ctx.get("glm_order") or compute_glm_weights(ctx)[1]["order"]
    df["rank"] = df.feature.map({f: i for i, f in enumerate(order)}).fillna(len(order))  # all_movement last
    return df, {"n": n_by_cohort(df), "order": order, "stats": _feature_tests(df, "delta_ll_nats_per_trial")}


def compute_glm_shuffle(ctx):
    """Shuffle importance of glm_behavior_v3: drop in held-out AUC when each predictor is shuffled (10 repeats;
    planner 2026-11-06: then same for delta AUC (shuffled importance)), same predictor order as the weights."""
    rows = []
    for r, f in _glm_files(ctx, "shuffle_importance.csv"):
        sh = pd.read_csv(f)
        sh = sh[sh.model != "Full Model"]
        rows.append(pd.DataFrame({"mouse_id": r.mouse_id, "session_id": r.session_id, "cohort": r.cohort,
                                  "feature": sh.model.map(_glm_feature), "delta_test_auc": sh.delta_test_auc.to_numpy(),
                                  "delta_test_balanced_accuracy": sh.delta_test_balanced_accuracy.to_numpy(),
                                  "delta_test_ll_per_trial": sh.get("delta_test_ll_per_trial", np.nan)}))
    df = pd.concat(rows, ignore_index=True)
    order = ctx.get("glm_order") or compute_glm_weights(ctx)[1]["order"]
    df["rank"] = df.feature.map({f: i for i, f in enumerate(order)}).fillna(len(order))
    return df, {"n": n_by_cohort(df), "order": order, "stats": _feature_tests(df, "delta_test_auc")}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="*")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--date", help="snapshot date (default today)")
    args = parser.parse_args()

    ctx = {"table": shared_behaviour.load(rebuild=args.rebuild_cache)}
    out = snapshot_dir(SPEC, args.date)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "_build").mkdir(exist_ok=True)
    meta_file = out / "_build" / "panel_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    for panel in SPEC["panels"]:
        pid, key = panel["id"], panel.get("key", panel["id"])
        fn = globals().get(f"compute_{key}")
        if fn is None or "data" not in panel or (args.panels and pid not in args.panels and key not in args.panels):
            continue
        df, info = fn(ctx)
        df.to_csv(out / "data" / panel["data"], index=False)
        meta[pid] = info
        print(pid, key, panel["data"], len(df), "rows", flush=True)
    meta_file.write_text(json.dumps(meta, indent=1, default=str))  # serialise fully before writing


if __name__ == "__main__":
    main()
