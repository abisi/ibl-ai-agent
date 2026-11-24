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


def perm_test_per_x(df, x, value, n_perm=5000, seed=0, min_mice=8):
    """User 2026-11-16: permutation test R+ vs R- at every x (trial resolution): statistic = difference of the cohort
    means over mice; null = the same with the cohort (reward group) labels of the mice shuffled (one shuffle applied to
    all x); two-sided p = (1 + #|null| >= |obs|) / (n_perm + 1). x values with fewer than `min_mice` mice in a cohort get
    no test. Uncorrected over x. Returns a DataFrame [x, n_rplus, n_rminus, diff, p]."""
    wide = df.pivot_table(index=["mouse_id", "cohort"], columns=x, values=value)
    lab = wide.index.get_level_values("cohort").to_numpy()
    data = wide.to_numpy(float)
    ok = np.isfinite(data)

    def diff(labels):
        a, b = labels == "R+", labels == "R-"
        with np.errstate(all="ignore"):
            return np.nanmean(np.where(a[:, None], data, np.nan), 0) - np.nanmean(np.where(b[:, None], data, np.nan), 0)

    obs = diff(lab)
    rng = np.random.default_rng(seed)
    exceed = np.zeros(len(obs))
    for _ in range(n_perm):
        exceed += np.abs(diff(rng.permutation(lab))) >= np.abs(obs) - 1e-12
    n_p, n_m = ok[lab == "R+"].sum(0), ok[lab == "R-"].sum(0)
    p = (1 + exceed) / (n_perm + 1)
    p[(n_p < min_mice) | (n_m < min_mice)] = np.nan
    return pd.DataFrame({x: wide.columns.to_numpy(), "n_rplus": n_p, "n_rminus": n_m, "diff": obs, "p": p})


def cluster_p(df, x, clusters, alpha=0.05):
    """Permutation p of the significant cluster containing each row's x, nan elsewhere (planner 2026-11-12: significance
    bars annotated with their p-value)."""
    p = np.full(len(df), np.nan)
    for c in clusters:
        if c["p"] < alpha:
            p[(df[x] >= c["start"]).to_numpy() & (df[x] <= c["end"]).to_numpy()] = c["p"]
    return p


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
    whisker days +1, +2 and +3 per mouse); planner 2026-11-06: line plot, days -2, -1, 0, >0 (= 1, 2, 3); learners only
    (planner 2026-11-11)."""
    t = ctx["table"]
    t = t[t.day.isin([-2, -1, 0, 1, 2, 3]) & (t.early_lick == 0) & t.learning_category.map(LEARNER).fillna(False).astype(bool)]
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
    d0 = day0_table(ctx["table"], learners_only=True)  # planner 2026-11-11
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


RT_FIRST_BIN, RT_LAST_BIN = (1, 5), (26, 30)  # first vs last display bin of 5 hits (planner 2026-11-11)
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
        bins.loc[bins.cohort == c, "sig_vs0_p"] = cluster_p(bins[bins.cohort == c], "hit_bin_centre", clusters_vs0[c])
    both = pd.concat(wide, names=["cohort"]).stack().rename("v").reset_index()
    both = both[both.groupby("hit_bin_centre").cohort.transform("nunique") == 2]
    clusters_cohorts = cluster_test(both, "hit_bin_centre", "v")
    bins["sig_cohorts"] = mark_clusters(bins, "hit_bin_centre", clusters_cohorts)
    bins["sig_cohorts_p"] = cluster_p(bins, "hit_bin_centre", clusters_cohorts)
    mice = []
    for (m, s, c), g in h.groupby(keys):
        b, a = np.polyfit((g.hit_number - 1) / 10, g.rt_change_ms, 1) if len(g) >= 10 else (np.nan, np.nan)
        mice.append(dict(mouse_id=m, session_id=s, cohort=c, kind="mouse", trial_type=trial_type, n_hits=len(g),
                         slope_ms_per_10_hits=b, intercept_ms=a,
                         late_change_ms=g[g.hit_number.between(*RT_LATE)].rt_change_ms.mean(),
                         last_minus_first_ms=(g[g.hit_number.between(*RT_LAST_BIN)].rt_change_ms.mean()
                                              - g[g.hit_number.between(*RT_FIRST_BIN)].rt_change_ms.mean())))
    mice = pd.DataFrame(mice)
    tests = {v: {**{c: _summary_vs_zero(mice[mice.cohort == c][v]) for c in COHORTS},
                 "rplus_vs_rminus": two_group_tests(mice[mice.cohort == "R+"][v], mice[mice.cohort == "R-"][v])}
             for v in ("late_change_ms", "slope_ms_per_10_hits", "last_minus_first_ms")}
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
        "n": n_by_cohort(mice), "late": tests["late_change_ms"], "slope": tests["slope_ms_per_10_hits"],
        "last_vs_first": tests["last_minus_first_ms"], "lmm": lmm,
        "clusters_vs0": clusters_vs0, "clusters_rplus_vs_rminus": clusters_cohorts,
        "n_sig_clusters": {k: len(v) for k, v in sig.items()},
        "sig_clusters_text": {k: "; ".join(f"hits {int(x['start']) - RT_BIN // 2}-{int(x['end']) + RT_BIN // 2}, P = {x['p']:.3g}" for x in v)
                              or "none" for k, v in sig.items()}}


def _rank_shift_null(x, y):
    """Spearman rho of x, y and its linear-shift null (user 2026-11-11: the former circular np.roll null was biased; now the
    project's non-wrapping linear shift, as ssl-whisker-hitmiss-timeresolved-decoding shift_null.py): y shifted against x by
    k = 10-50 % of the trials in both directions, rho recomputed on the overlap (x[k:] vs y[:n-k] and x[:n-k] vs y[k:]).
    Returns rho, the null median and all null rhos."""
    n = len(x)
    lags = range(max(1, int(0.1 * n)), max(1, int(0.5 * n)) + 1)
    null = np.array([stats.spearmanr(x[k:], y[:n - k])[0] if d else stats.spearmanr(x[:n - k], y[k:])[0]
                     for k in lags for d in (0, 1) if n - k >= 5])
    null = null[np.isfinite(null)]
    return float(stats.spearmanr(x, y)[0]), float(np.median(null)) if len(null) else np.nan, null


RT_N_NULL_DRAWS = 1000  # cohort-mean null: one random shift per mouse per draw (planner 2026-11-11)


def _cohort_null(nulls, rng):
    """Distribution of the cohort mean rho under the shift null: per draw, one shifted rho per mouse, averaged."""
    nulls = [n for n in nulls if len(n)]  # mice too short for a shift > RT_SHIFT_EXCLUDE have no null
    return np.array([np.mean([rng.choice(n) for n in nulls]) for _ in range(RT_N_NULL_DRAWS)])




def _cut_session(st):
    """Whisker day 0 of one session, early licks dropped, cut at session_end_time (as Fig. 1G-J)."""
    st = st[st.early_lick == 0].sort_values("start_time")
    return st[st.start_time <= session_end_time(st[st.trial_type == "whisker_trial"].start_time)]


def rt_change_aligned(ctx):
    """User 2026-11-22 ('K aligned with I'): whisker-hit reaction time change on the trial axis of Fig. 1I. Learners,
    whisker day 0 cut at session_end_time, early licks dropped; each whisker hit with a reaction time (0-1 s) gets its
    whisker-trial index from the first whisker hit (0 = the first hit) and its change = RT minus the mouse's median over
    its first RT_START hits; per mouse, the median per bin of Fig. 1I (0 alone, 1-5, ..., 26-30). R+ vs R- per bin
    (Mann-Whitney + Welch, Benjamini-Hochberg over the bins, >= RT_MIN_MICE mice per cohort); late change = per-mouse
    mean over whisker trials 21-30 (Wilcoxon vs 0 within cohort; R+ vs R- Mann-Whitney and Welch).
    Rows: kind 'bin' (mouse x bin), 'bin_test', 'mouse'."""
    from statsmodels.stats.multitest import multipletests
    rows = []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        w = _cut_session(st)
        w = w[w.trial_type == "whisker_trial"]
        lick = w.lick_flag.to_numpy() == 1
        if not lick.any():
            continue
        idx = np.arange(len(w)) - int(np.argmax(lick))
        rt = (w.computed_reaction_time.where(w.computed_reaction_time.between(0, 1)) * 1000).to_numpy()
        ok = lick & np.isfinite(rt) & (idx >= 0)
        rows.append(pd.DataFrame({"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                                  "trial_from_first_hit": idx[ok], "hit_number": np.arange(1, ok.sum() + 1),
                                  "rt_ms": rt[ok]}))
    h = pd.concat(rows, ignore_index=True)
    ref = h[h.hit_number <= RT_START].groupby("mouse_id").rt_ms.median()
    h["rt_change_ms"] = h.rt_ms - h.mouse_id.map(ref)
    lo_e, hi_e = ALIGN_BIN_EDGES[:-1], ALIGN_BIN_EDGES[1:] - 1
    b = np.searchsorted(ALIGN_BIN_EDGES, h.trial_from_first_hit, side="right") - 1
    h = h[(b >= 0) & (b < len(lo_e))].assign(bin=b[(b >= 0) & (b < len(lo_e))])
    keys = ["mouse_id", "session_id", "cohort"]
    bins = h.groupby(keys + ["bin"]).rt_change_ms.agg(value="median", n_hits_in_bin="size").reset_index()
    bins["bin_start"], bins["bin_end"] = lo_e[bins.bin], hi_e[bins.bin]
    bins["bin_centre"] = (bins.bin_start + bins.bin_end) / 2
    bins["kind"] = "bin"
    tests = []
    for (b0, b1, bc), g in bins.groupby(["bin_start", "bin_end", "bin_centre"]):
        a_, b_ = g[g.cohort == "R+"].value, g[g.cohort == "R-"].value
        if min(len(a_), len(b_)) >= RT_MIN_MICE and b0 > 0:
            tests.append({"bin_start": b0, "bin_end": b1, "bin_centre": bc, "n_rplus": len(a_), "n_rminus": len(b_),
                          **two_group_tests(a_, b_)})
    tests = pd.DataFrame(tests)
    tests["mannwhitney_q_bh"] = multipletests(tests.mannwhitney_p, method="fdr_bh")[1]
    tests["welch_q_bh"] = multipletests(tests.welch_p.fillna(1), method="fdr_bh")[1]
    mice = (h[h.trial_from_first_hit.between(*RT_LATE)].groupby(keys).rt_change_ms.mean().rename("late_change_ms")
            .reset_index().assign(kind="mouse"))
    late = {**{c: _summary_vs_zero(mice[mice.cohort == c].late_change_ms) for c in COHORTS},
            "rplus_vs_rminus": two_group_tests(mice[mice.cohort == "R+"].late_change_ms,
                                               mice[mice.cohort == "R-"].late_change_ms)}
    sig = tests[tests.mannwhitney_q_bh < 0.05]
    return pd.concat([bins, tests.assign(kind="bin_test"), mice], ignore_index=True), {
        "n": n_by_cohort(bins), "late": late, "bin_tests": tests.to_dict(orient="records"),
        "sig_bins_text": ", ".join(f"{int(r.bin_start)}-{int(r.bin_end)} (q = {r.mannwhitney_q_bh:.2g})"
                                   for r in sig.itertuples()) or "none",
        "params": {"reference": f"median of the first {RT_START} hits", "late_trials": list(RT_LATE),
                   "bin_edges": [int(v) for v in ALIGN_BIN_EDGES], "session_keep": SESSION_KEEP}}


def rt_curve_lmm(d):
    """User 2026-11-22: linear mixed model of whisker-hit reaction time on the whisker curve, one cohort: rt_ms ~ curve,
    random intercept and curve slope per mouse (optimisers lbfgs, powell, nm in turn; random intercept only if none
    converges), REML. Returns the fixed effect of the curve (ms per unit of curve) and its p."""
    import warnings
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    fit, re_used = None, None
    for re_formula, method in [("~curve", m) for m in ("lbfgs", "powell", "nm")] + [("1", "lbfgs")]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                fit = smf.mixedlm("rt_ms ~ curve", d, groups=d.mouse_id, re_formula=re_formula).fit(reml=True, method=method)
                re_used = re_formula
                break
            except (ConvergenceWarning, np.linalg.LinAlgError):
                continue
    return {"slope": float(fit.params["curve"]), "se": float(fit.bse["curve"]), "z": float(fit.tvalues["curve"]),
            "p": float(fit.pvalues["curve"]), "intercept": float(fit.params["Intercept"]), "n_mice": int(d.mouse_id.nunique()),
            "n_trials": len(d), "random_effects": "intercept + slope" if re_used == "~curve" else "intercept",
            "optimiser": method}


def rt_curve_lmm_rank(d):
    """User 2026-11-23 ('show the correlation coefficient; is it possible to have Spearman?'): rank-based linear mixed
    model, one cohort. Per mouse, reaction time and curve are ranked (ties averaged) and z-scored within the mouse, so the
    least-squares slope of one mouse is its Spearman rho; z_rt ~ z_curve with a random curve slope per mouse (random
    intercept only if the fit does not converge), REML. The fixed slope is the mouse-weighted mean Spearman rho
    (rho_lmm), tested against 0 with the mice as random effects."""
    import warnings
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import ConvergenceWarning

    def z(v):
        r = stats.rankdata(v)
        return (r - r.mean()) / r.std()
    d = d.groupby("mouse_id").filter(lambda g: len(g) >= 5 and g.curve.nunique() > 1 and g.rt_ms.nunique() > 1)
    d = d.assign(z_rt=d.groupby("mouse_id").rt_ms.transform(z), z_curve=d.groupby("mouse_id").curve.transform(z))
    fit = None
    # z-scored within mouse: no intercept variance, so a random slope only ("0 + z_curve"); last resort, the random
    # slope fit with its convergence warning recorded
    # lbfgs returns the start value (slope 0) on these data (checked 2026-11-23); powell, nm and bfgs agree with each
    # other and with the mean per-mouse Spearman rho
    for method in ("powell", "nm", "bfgs"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            try:
                fit = smf.mixedlm("z_rt ~ z_curve", d, groups=d.mouse_id, re_formula="~0 + z_curve").fit(reml=True,
                                                                                                        method=method)
            except np.linalg.LinAlgError:
                continue
        if fit.converged:
            break
    per_mouse = d.groupby("mouse_id").apply(lambda g: stats.spearmanr(g.curve, g.rt_ms)[0])
    return {"rho": float(fit.params["z_curve"]), "se": float(fit.bse["z_curve"]), "p": float(fit.pvalues["z_curve"]),
            "n_mice": int(d.mouse_id.nunique()), "n_trials": len(d),
            "random_effects": "slope", "optimiser": method, "converged": bool(fit.converged),
            "mean_per_mouse_rho": float(per_mouse.mean())}


def _shift_p(rho, null):
    """Two-sided p of rho against its linear-shift null distribution (user 2026-11-22: per-mouse significance)."""
    null = np.asarray(null, float)
    if not len(null):
        return np.nan
    return float(min(1, 2 * min(np.sum(null >= rho) + 1, np.sum(null <= rho) + 1) / (len(null) + 1)))


def compute_rt_trials(ctx):
    """User 2026-11-23 ('the previous version was better; the first bin should be 0'): whisker-hit RT change from the
    mouse's first 5 hits on the hit axis (rt_change_from_start), its own panel."""
    return rt_change_from_start(ctx, "whisker_trial")


def compute_rt_curve(ctx):
    """Reaction time and learning curve (user 2026-11-23: own panel; scatter of both cohorts overlaid with a rank-based
    LMM rho, then the per-mouse Spearman rho). Was compute_rt_learning (user 2026-11-09, 2026-11-22). Learners,
    whisker day 0 cut at session_end_time (user 2026-11-22: trimmed as Fig. 1G-J; curves refitted on the cut session),
    early licks dropped; reaction time = computed_reaction_time (0-1 s) on licked trials.
    Left (kind 'bin' / 'bin_test' / 'mouse'): whisker-hit reaction time change on the trial axis of Fig. 1I
    (rt_change_aligned).
    Scatter (kind 'scatter'): reaction time of every whisker hit against the whisker curve at that trial, all mice;
    linear mixed model per cohort (rt_curve_lmm; user 2026-11-22).
    Right (kind 'rho'): per mouse and licked trial type (whisker hits, auditory hits, false alarms; >= 10 per mouse),
    Spearman rho between reaction time and the whisker learning curve (at the trial; other trial types at the preceding
    whisker trial), its linear-shift null (shifts of 10-50 % of the licked trials) and per-mouse two-sided shift p;
    fraction of mice with p < 0.05 per cohort (binomial test against 0.05)."""
    rows, scatter = [], []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        st = _cut_session(st)
        curves, _, _ = session_curves(st, None)  # refitted on the cut session (not cached: differs from E, F)
        widx = np.cumsum(st.trial_type.to_numpy() == "whisker_trial") - 1
        curve = np.where(widx >= 0, curves.p_mean.to_numpy()[np.clip(widx, 0, len(curves) - 1)], np.nan)
        rt = (st.computed_reaction_time.where(st.computed_reaction_time.between(0, 1)) * 1000).to_numpy()
        base = dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
        for tt in TRIAL_TYPES:
            sel = (st.trial_type == tt).to_numpy() & (st.lick_flag == 1).to_numpy()
            r, c = rt[sel], curve[sel]
            ok = np.isfinite(r) & np.isfinite(c)
            if ok.sum() >= 10 and np.std(c[ok]) > 0:
                rho, null, null_all = _rank_shift_null(c[ok], r[ok])
                rows.append({**base, "kind": "rho", "trial_type": tt, "rho": rho, "shift_median": null,
                             "value": rho - null, "n_licked": int(ok.sum()), "shift_p": _shift_p(rho, null_all),
                             "_null": null_all})
            if tt == "whisker_trial" and ok.any():
                scatter.append(pd.DataFrame({**base, "kind": "scatter", "trial_type": tt, "curve": c[ok], "rt_ms": r[ok]}))
    rho = pd.DataFrame(rows)
    # planner 2026-11-11: raw rho, with the cohort-mean distribution under the shift null in grey
    rng, null_rows, null_p = np.random.default_rng(0), [], {}
    for (tt, c), g in rho.groupby(["trial_type", "cohort"]):
        dist = _cohort_null(list(g._null), rng)
        obs = g.rho.mean()
        null_p[f"{tt} {c}"] = {"mean_rho": float(obs), "null_mean": float(dist.mean()),
                               "p_two_sided": float(min(1, 2 * min(np.sum(dist >= obs) + 1, np.sum(dist <= obs) + 1)
                                                        / (len(dist) + 1)))}
        null_rows.append(pd.DataFrame({"kind": "rho_null", "trial_type": tt, "cohort": c, "draw": np.arange(len(dist)),
                                       "rho": dist}))
    rho = pd.concat([rho.drop(columns="_null")] + null_rows, ignore_index=True)
    real = rho[rho.kind == "rho"]
    stats_rho = {tt: {**{f"{c}_vs_0": vs_zero(real[(real.trial_type == tt) & (real.cohort == c)].value) for c in COHORTS},
                      "rplus_vs_rminus": two_group_tests(real[(real.trial_type == tt) & (real.cohort == "R+")].value,
                                                         real[(real.trial_type == tt) & (real.cohort == "R-")].value)}
                 for tt in TRIAL_TYPES}
    frac = {}  # user 2026-11-22: fraction of mice whose rho differs from their own shift null
    for (tt, c), g in real.groupby(["trial_type", "cohort"]):
        k, n = int((g.shift_p < 0.05).sum()), int(g.shift_p.notna().sum())
        neg = int(((g.shift_p < 0.05) & (g.rho < g.shift_median)).sum())
        frac[f"{tt} {c}"] = {"n_sig": k, "n": n, "fraction": k / n if n else np.nan, "n_sig_negative": neg,
                             "binomial_p": float(stats.binomtest(k, n, 0.05, alternative="greater").pvalue)}
    sc = pd.concat(scatter, ignore_index=True)
    lmm = {c: rt_curve_lmm(sc[sc.cohort == c]) for c in COHORTS}
    lmm_rank = {c: rt_curve_lmm_rank(sc[sc.cohort == c]) for c in COHORTS}
    sc = sc.assign(lmm_slope=sc.cohort.map({c: v["slope"] for c, v in lmm.items()}),
                   lmm_intercept=sc.cohort.map({c: v["intercept"] for c, v in lmm.items()}),
                   lmm_p=sc.cohort.map({c: v["p"] for c, v in lmm.items()}),
                   rho_lmm=sc.cohort.map({c: v["rho"] for c, v in lmm_rank.items()}),
                   rho_lmm_p=sc.cohort.map({c: v["p"] for c, v in lmm_rank.items()}))
    pooled = {c: dict(zip(("rho", "p"), map(float, stats.spearmanr(sc[sc.cohort == c].curve, sc[sc.cohort == c].rt_ms))))
              for c in COHORTS}  # descriptive only (hits of all mice pooled, not independent)
    return pd.concat([rho, sc], ignore_index=True), {
        "n": n_by_cohort(real), "stats_rho": stats_rho, "fraction_sig": frac, "lmm": lmm, "lmm_rank": lmm_rank,
        "pooled_spearman": pooled,
        "cohort_mean_vs_shift_null": null_p, "n_null_draws": RT_N_NULL_DRAWS, "session_keep": SESSION_KEEP}


def compute_first_lick(ctx):
    """Whisker trials before the first licked whisker trial of whisker day 0, learners (planner 2026-11-11)."""
    rows = []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
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


SESSION_KEEP = 0.85  # user 2026-11-17: the last 15% of each session (time from first to last whisker trial) is cut;
# the last kept trial is the session end (normalised session time 1)


def session_end_time(whisker_times):
    """End of the kept session: first whisker trial + SESSION_KEEP x (last - first whisker trial)."""
    tw = np.sort(np.asarray(whisker_times, float))
    return tw[0] + SESSION_KEEP * (tw[-1] - tw[0])


def compute_perf_session(ctx):
    """User 2026-11-17: whisker day 0 cut at session_end_time (last 15% of the session dropped), learning curves refitted
    on the cut session. Performance (whisker minus chance curve) and the whisker, chance (no-stimulus) and auditory curves
    on whisker trials, against normalised session time: first whisker trial (0) to session end (1), and first whisker
    hit (0) to session end (1). Cluster-based permutation test R+ vs R- of the performance within each group."""
    grid = np.round(np.arange(0, 1 + 1e-9, 0.02), 3)
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        st = st[st.start_time <= session_end_time(st[st.trial_type == "whisker_trial"].start_time)]
        curves, _, _ = session_curves(st, None)  # not cached: the cut session differs from E, F
        tw = st[st.trial_type == "whisker_trial"].start_time.to_numpy()
        vals = {"performance": (curves.p_mean - curves.p_chance).to_numpy(), "p_whisker": curves.p_mean.to_numpy(),
                "p_chance": curves.p_chance.to_numpy(), "p_auditory": np.asarray(curves.p_auditory, float)}
        learner = bool(LEARNER.get(st.learning_category.iloc[0], False))
        base = {"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                "group": "learner" if learner else "non-learner"}
        x = (tw - tw[0]) / (tw[-1] - tw[0])
        rows.append(pd.DataFrame({**base, "alignment": "first_whisker_trial", "session_time_norm": grid,
                                  **{k: np.interp(grid, x, v) for k, v in vals.items()}}))
        hits = np.flatnonzero(curves.lick.to_numpy() == 1)
        if len(hits) and hits[0] < len(tw) - 1:  # from the first whisker hit (0) to the session end (1)
            k = hits[0]
            xh = (tw[k:] - tw[k]) / (tw[-1] - tw[k])
            rows.append(pd.DataFrame({**base, "alignment": "first_whisker_hit", "session_time_norm": grid,
                                      **{kk: np.interp(grid, xh, v[k:]) for kk, v in vals.items()}}))
    df = pd.concat(rows, ignore_index=True)
    df["cluster_sig"] = False
    clusters, n = {}, {}
    for (grp, align), d in df.groupby(["group", "alignment"]):
        d = d.dropna(subset=["performance"])
        enough = d.groupby(["session_time_norm", "cohort"]).mouse_id.nunique().unstack().reindex(columns=COHORTS).fillna(0).min(1)
        d = d[d.session_time_norm.isin(enough[enough >= 8].index)]
        key = f"{grp} {align}"
        clusters[key] = cluster_test(d, "session_time_norm", "performance")
        df.loc[d.index, "cluster_sig"] = mark_clusters(d, "session_time_norm", clusters[key])
        df.loc[d.index, "cluster_p"] = cluster_p(d, "session_time_norm", clusters[key])
        n[key] = n_by_cohort(d)
    return df, {"n": n, "clusters": clusters, "session_keep": SESSION_KEEP}


def _perf(ctx, align):
    if "perf_session" not in ctx:
        ctx["perf_session"] = compute_perf_session(ctx)
    df, info = ctx["perf_session"]
    key = f"learner {align}"
    d = df[(df.group == "learner") & (df.alignment == align)]
    sig = [c for c in info["clusters"][key] if c["p"] < 0.05]
    return d, {"n": info["n"][key], "clusters": info["clusters"][key], "session_keep": SESSION_KEEP,
               "sig_clusters_text": "; ".join(f"{c['start']:.2f}-{c['end']:.2f}, P = {c['p']:.3g}" for c in sig) or "none"}


def compute_perf_trial(ctx):
    return _perf(ctx, "first_whisker_trial")


def compute_perf_hit(ctx):
    return _perf(ctx, "first_whisker_hit")


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


ALIGNED_WINDOW = (-10, 30)  # trials of the type from the alignment trial (user 2026-11-12: -10 to +30)
ALIGNED_MIN_MICE = 8  # per cohort and trial index, for the cluster test
ALIGNMENTS = {"first_whisker_trial": "1st W trial", "first_whisker_hit": "1st W hit"}  # top row first (user 2026-11-16)


ALIGN_BIN_EDGES = np.r_[-10, -5, 0, 1, np.arange(6, 32, 5)]  # bins -10..-6, -5..-1, 0 alone, 1-5, 6-10, ..., 26-30


def _aligned_lick_trials(ctx):
    """Per-trial lick outcomes of learners on the untrimmed whisker day 0 (auditory warm-up kept, early licks dropped),
    each trial type on its own trials: index 0 = the first trial of the type at or after the alignment trial (first
    whisker trial or first whisker hit), negative = trials of the type before it; window ALIGNED_WINDOW. Cached in ctx."""
    if "aligned_lick_trials" in ctx:
        return ctx["aligned_lick_trials"]
    lo, hi = ALIGNED_WINDOW
    learners = set(day0_table(ctx["table"], learners_only=True).mouse_id)
    d0 = shared_behaviour.load_untrimmed_day0()
    d0 = d0[d0.mouse_id.isin(learners) & (d0.early_lick == 0)]
    rows = []
    for sid, st in d0.groupby("session_id"):
        st = st.sort_values("start_time")
        st = st[st.start_time <= session_end_time(st[st.trial_type == "whisker_trial"].start_time)]  # user 2026-11-17
        out = lce.session_learning_curves_exact(st, sigma=CURVE_PARAMS["sigma"], conf_int=CURVE_PARAMS["conf_int"])
        if "whisker_trial" not in out or not (np.asarray(out["whisker_trial"]["outcomes"]) == 1).any():
            continue
        w = out["whisker_trial"]
        ref = {"first_whisker_trial": w["start_time"][0],
               "first_whisker_hit": w["start_time"][int(np.argmax(np.asarray(w["outcomes"]) == 1))]}
        base = dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0])
        for align, t_ref in ref.items():
            for tt in TRIAL_TYPES:
                if tt not in out:
                    continue
                y = np.asarray(out[tt]["outcomes"], float)
                rel = np.arange(len(y)) - int(np.sum(np.asarray(out[tt]["start_time"]) < t_ref))
                keep = (rel >= lo) & (rel <= hi)
                rows.append(pd.DataFrame({**base, "alignment": align, "trial_type": tt, "kind": "data",
                                          "trial_from_alignment": rel[keep], "p_lick": y[keep]}))
    ctx["aligned_lick_trials"] = pd.concat(rows, ignore_index=True)
    return ctx["aligned_lick_trials"]


def compute_aligned(ctx, align):
    """User 2026-11-17: whisker, false alarm and auditory P(lick) around one alignment (G, first whisker trial; I, first
    whisker hit): per-mouse lick fraction in bins (-10..-6, -5..-1, 0 alone, 1-5, ..., 26-30 trials of the type), mean
    +- s.e.m. over mice, dots connected (kind 'bin'). Sessions cut at session_end_time. R+ vs R- per bin (Mann-Whitney U
    and Welch t, bins with >= ALIGNED_MIN_MICE mice per cohort), Benjamini-Hochberg over the bins of each subplot
    (trial type); significance = Mann-Whitney q < 0.05 (Welch q saved)."""
    from statsmodels.stats.multitest import multipletests
    data = _aligned_lick_trials(ctx)
    # user 2026-11-23: trial 0 (the alignment trial itself) not shown for false-alarm and auditory trials; kept for
    # whisker trials (the first hit, P(lick) = 1). The bins after it stay 1-5, 6-10, ... (the former 0-4, 5-9 bins
    # shifted by one trial)
    data = data[(data.alignment == align)
                & ((data.trial_from_alignment != 0) | (data.trial_type == "whisker_trial"))]
    lo_e, hi_e = ALIGN_BIN_EDGES[:-1], ALIGN_BIN_EDGES[1:] - 1  # first and last trial of each bin
    idx = np.searchsorted(ALIGN_BIN_EDGES, data.trial_from_alignment, side="right") - 1
    ok = (idx >= 0) & (idx < len(lo_e))
    data = data[ok].assign(bin=idx[ok])
    bins = (data.groupby(["mouse_id", "session_id", "cohort", "alignment", "trial_type", "bin"])
            .agg(p_lick=("p_lick", "mean"), n_trials=("p_lick", "size")).reset_index())
    bins["bin_start"], bins["bin_end"] = lo_e[bins.bin], hi_e[bins.bin]
    bins["bin_centre"] = (bins.bin_start + bins.bin_end) / 2
    bins["kind"] = "bin"
    tests, n = [], {}
    for tt, g in bins.groupby("trial_type"):
        rows = []
        for (b0, b1, bc), gb in g.groupby(["bin_start", "bin_end", "bin_centre"]):
            a_, b_ = gb[gb.cohort == "R+"].p_lick, gb[gb.cohort == "R-"].p_lick
            if min(len(a_), len(b_)) < ALIGNED_MIN_MICE or np.ptp(np.r_[a_, b_]) == 0:
                continue
            rows.append({"trial_type": tt, "bin_start": b0, "bin_end": b1, "bin_centre": bc, **two_group_tests(a_, b_)})
        t = pd.DataFrame(rows)
        if len(t):
            t["mannwhitney_q_bh"] = multipletests(t.mannwhitney_p, method="fdr_bh")[1]
            t["welch_q_bh"] = multipletests(t.welch_p.fillna(1), method="fdr_bh")[1]
            tests.append(t)
        n[tt] = n_by_cohort(g)
    tests = pd.concat(tests, ignore_index=True).assign(kind="bin_test", alignment=align)
    sig = tests[tests.mannwhitney_q_bh < 0.05]
    text = {tt: (", ".join(f"{int(r.bin_start)}" + (f"-{int(r.bin_end)}" if r.bin_end > r.bin_start else "")
                           + f" (q = {r.mannwhitney_q_bh:.2g})" for r in sig[sig.trial_type == tt].itertuples()) or "none")
            for tt in TRIAL_TYPES}
    return pd.concat([bins, tests], ignore_index=True), {
        "n": n_by_cohort(bins), "n_by_trial_type": n, "sig_bins_text": text,
        "bin_tests": tests.to_dict(orient="records"),
        "params": {"window_trials": list(ALIGNED_WINDOW), "bin_edges": [int(v) for v in ALIGN_BIN_EDGES],
                   "min_mice_per_cohort": ALIGNED_MIN_MICE, "session_keep": SESSION_KEEP,
                   "session": "untrimmed whisker day 0 cut at session_end_time, early licks dropped",
                   "trial_0": "whisker trials only (user 2026-11-23)",
                   "test": "per bin Mann-Whitney U and Welch t, R+ vs R-, Benjamini-Hochberg within each subplot"}}


def compute_aligned_trial(ctx):
    return compute_aligned(ctx, "first_whisker_trial")


def compute_aligned_hit(ctx):
    return compute_aligned(ctx, "first_whisker_hit")


FIRST_HIT_SUBPANELS = {"whisker": "whisker_trial", "auditory": "auditory_trial", "false_alarm": "no_stim_trial"}
# user 2026-11-11: trial 0 alone (whisker: the hit), then 1-5, 6-10, ...; same bins for every trial type (user 2026-11-11)
FIRST_HIT_WHISKER_EDGES = np.r_[-10, -5, 0, np.arange(1, 52, 5)]
FIRST_HIT_SESSION_FRACTION = 1.0  # planner 2026-11-12: whole session (was 0.9, first 90% of session time)  # plot order (planner 2026-11-11)


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
            "sig_minutes_text": _ranges_text(bins[bins.mannwhitney_q_bh < 0.05].bin, 5 if kind == "fa_count" else 1),
            "bin_tests": bins.drop(columns=["row"]).to_dict(orient="records")}
    return bins, info


def compute_first_hit_licks(ctx):
    """Whisker | auditory | spontaneous licks around the first whisker hit (user 2026-11-10: 'like for FA_divergence'),
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
    # planner 2026-11-11 (14:09): whisker, auditory and false-alarm P(lick) by trial of the type from the first whisker hit
    # (explore_fa_divergence 'fa_count', 5-trial bins); spontaneous licks per minute of time (unchanged)
    trials, sessions = fa.session_tables(fa.day0_active(), trial_types=("whisker_trial", "auditory_trial", "no_stim_trial"))
    span = trials.groupby("session_id").start_time.agg(["min", "max"])
    t_end = trials.session_id.map(span["min"] + FIRST_HIT_SESSION_FRACTION * (span["max"] - span["min"]))
    trials = trials[trials.start_time <= t_end]
    sid = {s["mouse_id"]: s["session_id"] for s in sessions}
    rows, info = [], {}
    for sub, tt in FIRST_HIT_SUBPANELS.items():
        if tt is None:
            kind, measure = "licks", "spontaneous_licks_per_min"
            b, pp = fa.per_mouse(kind, None, sessions)
        else:
            kind, measure = "fa_count", "lick_probability"
            tr = trials[trials.trial_type == tt]
            default = fa.ROWS[kind]["edges"]
            fa.ROWS[kind]["edges"] = FIRST_HIT_WHISKER_EDGES  # all trial types aligned like whisker (user 2026-11-11)
            try:
                b, pp = fa.per_mouse(kind, tr, None)
                e = fa.ROWS[kind]["edges"]
                n_tr = (tr[(tr.fa_index >= e[0]) & (tr.fa_index < e[-1])]
                        .assign(bin=lambda d: e[np.searchsorted(e, d.fa_index, side="right") - 1])
                        .groupby(["mouse_id", "bin"]).size().rename("n_trials"))
                b = b.merge(n_tr.reset_index(), on=["mouse_id", "bin"], how="left")
                b["bin_end"] = b.bin.map(dict(zip(e[:-1], e[1:])))  # exclusive end of the bin (trials)
                bins, info[sub] = _first_hit_tests(fa, kind, b, pp, vs_pre=sub != "whisker")
            finally:
                fa.ROWS[kind]["edges"] = default
        info[sub]["n"] = n_by_cohort(b.assign(session_id=b.mouse_id.map(sid)))
        info[sub]["measure"] = measure
        q = bins[["bin", "mannwhitney_q_bh", "welch_q_bh"]].rename(columns={"mannwhitney_q_bh": "cohort_q_bh",
                                                                           "welch_q_bh": "cohort_welch_q_bh"})
        rows.append(b.merge(q, on="bin", how="left").assign(sub_panel=sub, trial_type=tt or "spontaneous", measure=measure,
                                                             x_unit="minute" if tt is None else "trial"))
    df = pd.concat(rows, ignore_index=True).rename(columns={"bin": "bin_start"})
    df["session_id"] = df.mouse_id.map(sid)
    df["cohort_sig"] = df.cohort_q_bh < 0.05
    df = df[["mouse_id", "session_id", "cohort", "sub_panel", "trial_type", "measure", "x_unit", "bin_start", "bin_end",
             "value", "n_trials", "cohort_q_bh", "cohort_welch_q_bh", "cohort_sig"]]
    info["n"] = n_by_cohort(df)
    info["params_trials"] = {"session_fraction": FIRST_HIT_SESSION_FRACTION, "bin_trials": 5, "edges_trials": [int(fa.ROWS["fa_count"]["edges"][0]),
                                                                int(fa.ROWS["fa_count"]["edges"][-1])],
                             "pre_trials": [int(v) for v in fa.ROWS["fa_count"]["pre"]],
                             "post_trials": [int(v) for v in fa.ROWS["fa_count"]["post"]]}
    info["params"] = {"response_window_s": fa.RESP_WIN, "reward_exclusion_s": fa.REWARD_EXCL, "bin_min": 1,
                      "edges_min": [int(fa.ROWS["licks"]["edges"][0]), int(fa.ROWS["licks"]["edges"][-1])],
                      "pre_min": [-5, 0], "post_min": [0, 5], "min_cover_s": fa.MIN_COVER,
                      "min_pre_trials": fa.ROWS["fa_time"]["min_pre"], "min_mice_per_cohort_bin_test": fa.MIN_MICE}
    return df, info


def compute_movement_course(ctx):
    """DLC movements after the stimulus on whisker and auditory lick trials (planner 2026-11-11, replaces the piezo lick
    course): whisker speed, jaw opening and pupil area from shared_movement_peth (dlc_helpers, mice with dual-view video),
    stimulus-aligned, each trial minus its pre-stimulus mean, mean over trials per session, then over sessions per mouse.
    Learners, whisker day 0, sessions with early-lick-free lick trials. Cluster test R+ vs R- per signal and trial type
    (mice; Welch t, cohort-label permutations)."""
    import shared_movement_peth
    d0 = day0_table(ctx["table"], learners_only=True)
    d0 = d0[(d0.early_lick == 0) & (d0.lick_flag == 1) & d0.trial_type.isin(["whisker_trial", "auditory_trial"])]
    peth = shared_movement_peth.load()
    peth = peth[(peth["align"] == "stimulus") & (peth.lick == 1) & peth.trial_type.isin(["whisker_trial", "auditory_trial"])
                & peth.session_id.isin(set(d0.session_id))]
    d = (peth.groupby(["mouse_id", "cohort", "signal", "unit", "trial_type", "time_s"])
         .agg(session_id=("session_id", lambda v: ";".join(sorted(set(v)))), value=("value_delta", "mean"),
              n_trials=("n_trials", "sum"))
         .reset_index())
    d["cluster_sig"] = False
    tests, n = {}, {}
    for (sig, tt), g in d.groupby(["signal", "trial_type"]):
        tests[f"{sig} {tt}"] = cluster_test(g, "time_s", "value")
        d.loc[g.index, "cluster_sig"] = mark_clusters(g, "time_s", tests[f"{sig} {tt}"])
        d.loc[g.index, "cluster_p"] = cluster_p(g, "time_s", tests[f"{sig} {tt}"])
        n[f"{sig} {tt}"] = n_by_cohort(g)
    names = {"whisker_speed": "whisker speed", "jaw_opening": "jaw opening", "pupil_area": "pupil area"}
    text = [f"{names.get(k.split()[0], k.split()[0])}, {k.split()[1].split('_')[0]} trials, {c['start']:.2f} to {c['end']:.2f} s "
            f"(P = {c['p']:.3f})" for k, cl in tests.items() for c in cl if c["p"] < 0.05]
    return d, {"n": n_by_cohort(d), "n_by_group": n, "clusters": tests, "cluster_text": "; ".join(text) or "none",
               "params": shared_movement_peth.PARAMS}


HITMISS_WINDOW = {"whisker_speed": (-0.05, 0.1), "jaw_opening": (-0.1, 0.5), "pupil_area": (-1.0, 3.0),
                  "nose_speed": (-0.1, 0.5)}


def compute_movement_hitmiss(ctx, signal):
    """User 2026-11-16 (Supp. Fig. 1, one panel per DLC measure): stimulus-aligned `signal` on whisker hits and whisker
    misses of whisker day 0, per cohort (shared_movement_peth, dlc_helpers, mice with dual-view video; each trial minus
    its pre-stimulus mean, mean over trials per session, then over sessions per mouse). Learners, early-lick trials
    dropped. Within each cohort, hit minus miss per mouse (mice with both), sign-flip cluster test vs 0 (1,000
    permutations, cluster P < 0.05); cluster P on the bars."""
    import shared_movement_peth
    learners = set(day0_table(ctx["table"], learners_only=True).session_id)
    peth = shared_movement_peth.load()
    peth = peth[(peth["align"] == "stimulus") & (peth.trial_type == "whisker_trial") & (peth.signal == signal)
                & peth.session_id.isin(learners) & peth.time_s.between(*HITMISS_WINDOW[signal])]
    d = (peth.groupby(["mouse_id", "cohort", "signal", "unit", "lick", "time_s"])
         .agg(session_id=("session_id", lambda v: ";".join(sorted(set(v)))), value=("value_delta", "mean"),
              n_trials=("n_trials", "sum"))
         .reset_index())
    d["outcome"] = d.lick.map({1: "hit", 0: "miss"})
    d["cluster_sig"], d["cluster_p"] = False, np.nan
    clusters, n = {}, {}
    for c in COHORTS:
        g = d[d.cohort == c]
        wide = g.pivot_table(index="mouse_id", columns=["lick", "time_s"], values="value")
        if 0 not in wide.columns.get_level_values(0) or 1 not in wide.columns.get_level_values(0):
            continue
        diff = (wide[1] - wide[0]).dropna(how="all")
        clusters[c] = sign_flip_clusters(diff)
        idx = g.index
        d.loc[idx, "cluster_sig"] = mark_clusters(g, "time_s", clusters[c])
        d.loc[idx, "cluster_p"] = cluster_p(g, "time_s", clusters[c])
        n[c] = {"mice_hit_and_miss": int(len(diff)), "trials_hit": int(g[g.lick == 1].groupby("mouse_id").n_trials.first().sum()),
                "trials_miss": int(g[g.lick == 0].groupby("mouse_id").n_trials.first().sum())}
    text = "; ".join(f"{c}, {k['start']:.2f} to {k['end']:.2f} s (P = {k['p']:.3f})"
                     for c, cl in clusters.items() for k in cl if k["p"] < 0.05) or "none"
    return d, {"n": n_by_cohort(d), "n_hit_miss": n, "clusters": clusters, "cluster_text": text,
               "window_s": list(HITMISS_WINDOW[signal]), "unit": d.unit.iloc[0] if len(d) else None,
               "params": shared_movement_peth.PARAMS}


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


def lagged_corr(x, y, lag, method="pearson"):
    """Pearson r (or Spearman rho) of x and y with y shifted by `lag` trials (positive: x leads y)."""
    if lag > 0:
        x, y = x[:-lag], y[lag:]
    elif lag < 0:
        x, y = x[-lag:], y[:lag]
    ok = np.isfinite(x) & np.isfinite(y)
    f = stats.spearmanr if method == "spearman" else stats.pearsonr
    return f(x[ok], y[ok])[0] if ok.sum() > 4 else np.nan


def shift_null(x, y, rng):
    """Correlations of x with y linearly shifted by more than SHIFT_EXCLUDE trials (keeps each series' autocorrelation)."""
    n = len(y)
    valid = [s for s in range(1, n) if min(s, n - s) > SHIFT_EXCLUDE] or list(range(1, n))
    return np.array([lagged_corr(x, y, s) for s in rng.choice(valid, size=N_SHIFTS)])


def compute_xcorr(ctx):
    """Learners: cross-correlation of the whisker and chance curves (refitted on the cut session, user 2026-11-22), with a
    linear-shift null at lag 0. Also on linearly detrended curves (r_detrended; user 2026-11-22: the asymmetry of the raw
    cross-correlation comes from slow trends, checks/003)."""
    rng = np.random.default_rng(0)
    rows = []
    for sid, st in day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        curves, _, _ = session_curves(st, ctx)  # user 2026-11-23: whole session (the end kept), as the posture panels
        w, fa = curves.p_mean.to_numpy(), curves.p_chance.to_numpy()
        null = shift_null(w, fa, rng)
        t = np.arange(len(w))
        wd, fad = w - np.polyval(np.polyfit(t, w, 1), t), fa - np.polyval(np.polyfit(t, fa, 1), t)
        for lag in LAGS:
            rows.append(dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0], lag=int(lag),
                             r=lagged_corr(w, fa, lag), r_detrended=lagged_corr(wd, fad, lag),
                             rho=lagged_corr(w, fa, lag, "spearman"),  # user 2026-11-23: Spearman version
                             rho_detrended=lagged_corr(wd, fad, lag, "spearman"),
                             null_mean_lag0=float(np.nanmean(null)), null_sd_lag0=float(np.nanstd(null))))
    df = pd.DataFrame(rows)
    lag0 = df[df.lag == 0].assign(r_minus_null=lambda d: d.r - d.null_mean_lag0)
    vs_null = {c: {"median_r": float(lag0[lag0.cohort == c].r.median()),
                   **vs_zero(lag0[lag0.cohort == c].r_minus_null)} for c in COHORTS}
    asym = {}  # mean r at negative lags (FA curve leads) minus positive lags (whisker curve leads), per mouse
    for col in ("r", "r_detrended", "rho", "rho_detrended"):
        a = df.groupby(["mouse_id", "cohort"]).apply(
            lambda g: g[g.lag < 0][col].mean() - g[g.lag > 0][col].mean(), include_groups=False).rename("a").reset_index()
        asym[col] = {c: _summary_vs_zero(a[a.cohort == c].a) for c in COHORTS}
    return df, {"n": n_by_cohort(df), "lag0_vs_shift_null": vs_null, "asymmetry": asym,
                "lag0_rplus_vs_rminus": two_group_tests(lag0[lag0.cohort == "R+"].r, lag0[lag0.cohort == "R-"].r),
                "lag0_rho": {c: _summary_vs_zero(lag0[lag0.cohort == c].rho) for c in COHORTS},
                "lag0_rho_rplus_vs_rminus": two_group_tests(lag0[lag0.cohort == "R+"].rho, lag0[lag0.cohort == "R-"].rho)}


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
    # user 2026-11-17: glm_v7 lick head (= glm_v6 refitted on the reconverted NWBs, pupil QC; glm_v7_run.py)
    return CACHE_DIR / "glm_v7"


GLM_ORDER = ["cumulative_reward_fraction", "time_since_reward", "whisker_speed_pre", "nose_speed_pre", "pupil_area_pre",
             "jaw_speed_pre", "spontaneous_lick_rate", "lick_whisker_trial_n-3", "lick_whisker_trial_n-2",
             "lick_whisker_trial_n-1"]  # planner panel o, 2026-11-09


GLM_GROUPS = {"reward": ["cumulative_reward_fraction", "time_since_reward"],  # glm_v7_groups.py (user 2026-11-23)
              "movement": ["whisker_speed_pre", "nose_speed_pre", "pupil_area_pre", "jaw_speed_pre"],
              "history": ["spontaneous_lick_rate", "lick_whisker_trial_n-3", "lick_whisker_trial_n-2",
                          "lick_whisker_trial_n-1"]}


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


GLM_EXAMPLE_MOUSE = "MH011"  # user 2026-11-12 "change prediction example": explore_glm_examples.py, best balanced accuracy
                             # among R+ learner sessions with >= 100 trials and lick rate 0.25-0.75 (MH070 licked 75 %)


def _glm_example_mouse(ctx):
    """GLM_EXAMPLE_MOUSE when set; else a well predicted but typical R+ learner session (user 2026-11-09: not the best):
    among sessions with >= 100 whisker trials, the one whose test AUC is closest to the 75th percentile of R+ learners."""
    if GLM_EXAMPLE_MOUSE:
        return GLM_EXAMPLE_MOUSE
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
    # user 2026-11-22: the model's held-out prediction under the lick row (same session as the GLM example)
    ex, _ = compute_glm_example(ctx)
    if len(ex) == len(y) and (ex.lick.to_numpy() == y).all():
        df = df.merge(ex[["whisker_trial", "predicted_p_lick"]], on="whisker_trial", how="left")
    else:
        print("glm_schematic: example and design-matrix trials differ", len(ex), len(y))
    return df, {"mouse_id": mouse, "session_id": r.session_id, "n_whisker_trials": int(len(y)), "features": list(names),
                "same_session_as_example": bool("predicted_p_lick" in df)}


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
    # user 2026-11-16: whisker exact learning curve instead of the rolling data
    st = day0_table(ctx["table"])
    curves, _, _ = session_curves(st[st.session_id == r.session_id], ctx)
    if len(curves) == len(df) and (curves.lick.to_numpy() == y).all():
        df["curve_p_mean"], df["curve_p_low"], df["curve_p_high"] = (curves.p_mean.to_numpy(), curves.p_low.to_numpy(),
                                                                     curves.p_high.to_numpy())
    else:  # different trial sets: curve on its own trials, matched by order of whisker trials
        print("glm_example: curve and GLM trials differ", len(curves), len(df))
        m = min(len(curves), len(df))
        for c, col in (("curve_p_mean", "p_mean"), ("curve_p_low", "p_low"), ("curve_p_high", "p_high")):
            df[c] = np.r_[curves[col].to_numpy()[:m], np.full(len(df) - m, np.nan)]
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
    """Planner 2026-11-06 (panel p): GLM weights, then the same for the shuffle importance, side by side, predictors in
    the order of mean absolute weight. Shuffle importance in balanced accuracy (planner 2026-11-11; was AUC). Long format:
    metric = weight | delta_bacc | delta_ll_drop."""
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
    # user 2026-11-23: groups of predictors removed together (glm_v7_groups.py: reward, movement, history)
    grp = pd.read_csv(_glm_dir() / "_groups" / "group_drop_all.csv")
    grp = grp[grp.session_id.isin(set(w.session_id))]
    grp = grp.rename(columns={"group": "feature", "delta_test_ll_per_trial": "value"})[keep + ["value"]]
    sch, sch_info = compute_glm_schematic(ctx)  # user 2026-11-23: minimal design-matrix schematic above the weights
    df = pd.concat([w.rename(columns={"coefficient": "value"}).assign(metric="weight"),
                    s[keep + ["delta_test_balanced_accuracy"]].rename(columns={"delta_test_balanced_accuracy": "value"})
                     .assign(metric="delta_bacc", rank_by_mean_abs=rank),
                    drop.assign(metric="delta_ll_drop", rank_by_mean_abs=rank),
                    grp.assign(metric="delta_ll_group", group_members=grp.feature.map(lambda g: ";".join(GLM_GROUPS[g]))),
                    sch.assign(metric="schematic")], ignore_index=True)
    return df, {"n": w_info["n"], "order": w_info["order"], "stats_weight": w_info["stats"],
                "stats_delta_ll_group": _feature_tests(grp, "value"), "groups": GLM_GROUPS, "schematic": sch_info,
                "stats_delta_bacc_shuffle": _feature_tests(s, "delta_test_balanced_accuracy"),
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
