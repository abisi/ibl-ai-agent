"""Figure1_behaviour -- compute every filled panel's tidy table from scratch and write it to the snapshot's data/ folder.

Run on haas (repo root):
    nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/compute.py [--panels e f] [--rebuild-cache]
then build.py figures/Figure1_behaviour.
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
PANELS = {p["id"]: p for p in SPEC["panels"]}
CURVE_PARAMS = SPEC["shared"]["day0_curves"]["params"]
STATE_PARAMS = SPEC["shared"]["perf_states"]["params"]
COHORTS = ["R+", "R-"]
LEARNER = {"good": True, "moderate": True, "bad": False}  # reference sheet learning_category (user 2026-11-02)


# ---------------------------------------------------------------- shared inputs

def day0_table(table):
    return table[(table.behavior == "whisker") & (table.day == 0)]


def session_curves(session_table):
    """Exact whisker curve, chance (no-stim) and auditory curves on whisker trials, and performance states."""
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
    return curves, w["learning_trial"], w["mouse_cat"]


def n_by_cohort(df):
    return {c: {"mice": int(df[df.cohort == c].mouse_id.nunique()),
                "sessions": int(df[df.cohort == c].session_id.nunique()) if "session_id" in df else None}
            for c in COHORTS}


# ---------------------------------------------------------------- panels

def compute_example(ctx, pid):
    mouse = PANELS[pid]["params"]["example"]
    st = day0_table(ctx["table"])
    st = st[st.mouse_id == mouse]
    curves, lt, cat = session_curves(st)
    df = curves.assign(mouse_id=mouse, session_id=st.session_id.iloc[0], cohort=st.cohort.iloc[0],
                       learning_category=st.learning_category.iloc[0])
    df = df[["mouse_id", "session_id", "cohort", "learning_category"] + list(curves.columns)]
    return df, {"mouse_id": mouse, "session_id": st.session_id.iloc[0], "cohort": st.cohort.iloc[0],
                "learning_category": st.learning_category.iloc[0], "n_whisker_trials": len(df),
                "whisker_hit_rate": float(df.lick.mean()),
                "fa_rate": float(st[(st.trial_type == "no_stim_trial") & (st.early_lick == 0)].lick_flag.mean()),
                "frac_trials_high": float((df.state == "high").mean()),
                "rule_learning_trial": None if np.isnan(lt) else int(lt) + 1, "rule_category": cat}


def compute_e(ctx):
    return compute_example(ctx, "e")


def compute_f(ctx):
    return compute_example(ctx, "f")


def compute_g(ctx):
    return compute_example(ctx, "g")


def compute_h(ctx):
    return compute_example(ctx, "h")


def compute_b(ctx):
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


def compute_d(ctx):
    d0 = day0_table(ctx["table"])
    licks = d0[(d0.lick_flag == 1) & (d0.early_lick == 0)
               & d0.trial_type.isin(["whisker_trial", "auditory_trial", "no_stim_trial"])]
    df = licks[["mouse_id", "session_id", "cohort", "trial_type", "computed_reaction_time"]].rename(
        columns={"computed_reaction_time": "reaction_time_s"})
    df = df[df.reaction_time_s > 0].reset_index(drop=True)
    med = df.groupby(["mouse_id", "cohort", "trial_type"]).reaction_time_s.median().reset_index()
    tests = {}
    for tt in ["whisker_trial", "auditory_trial", "no_stim_trial"]:
        m = med[med.trial_type == tt]
        tests[tt] = two_group_tests(m[m.cohort == "R+"].reaction_time_s, m[m.cohort == "R-"].reaction_time_s)
    return df, {"n": n_by_cohort(df),
                "n_trials": {f"{c} {t}": int(n) for (c, t), n in df.groupby(["cohort", "trial_type"]).size().items()},
                "stats_mouse_median": tests}


def compute_i(ctx):
    d0 = day0_table(ctx["table"])
    mice = d0.groupby("mouse_id").agg(cohort=("cohort", "first"), learning_category=("learning_category", "first"),
                                      session_id=("session_id", "first")).reset_index()
    mice["learner"] = mice.learning_category.map(LEARNER)
    df = mice.groupby("cohort").agg(learners=("learner", "sum"), mice=("learner", "size")).reset_index()
    df["fraction_learners"] = df.learners / df.mice
    for cat in ["good", "moderate", "bad"]:
        df[f"n_{cat}"] = [int((mice[mice.cohort == c].learning_category == cat).sum()) for c in df.cohort]
    table = [[int(r.learners), int(r.mice - r.learners)] for r in df.set_index("cohort").loc[COHORTS].itertuples()]
    res = stats.fisher_exact(table)
    return df, {"n": n_by_cohort(mice), "stats": {"fisher_odds_ratio": float(res.statistic), "fisher_p": float(res.pvalue)}}


def compute_c(ctx):
    t = ctx["table"]
    t = t[t.day.isin([-1, 0]) & (t.early_lick == 0)]
    per_session = t.groupby(["mouse_id", "cohort", "day", "session_id", "trial_type"]).lick_flag.mean().reset_index()
    df = (per_session.groupby(["mouse_id", "cohort", "day", "trial_type"])
          .agg(lick_probability=("lick_flag", "mean"), session_id=("session_id", lambda s: ";".join(sorted(s))))
          .reset_index())
    tests = {}
    for tt in ["whisker_trial", "auditory_trial", "no_stim_trial"]:
        d = df[(df.day == 0) & (df.trial_type == tt)]
        tests[tt] = two_group_tests(d[d.cohort == "R+"].lick_probability, d[d.cohort == "R-"].lick_probability)
    return df, {"n": {c: {int(d): int(df[(df.cohort == c) & (df.day == d)].mouse_id.nunique()) for d in (-1, 0)}
                      for c in COHORTS}, "stats_day0": tests}


def compute_j(ctx):
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time").reset_index(drop=True)
        curves, _, _ = session_curves(st)
        st["block_perf_type"] = pd.Series([None] * len(st), dtype=object)
        st.loc[st.trial_type == "whisker_trial", "block_perf_type"] = curves.state.to_numpy()
        st = learning_utils.propagate_expertise_inplace(st)  # other trial types: state of the nearest whisker trial
        rate = st.groupby(["block_perf_type", "trial_type"]).lick_flag.agg(["mean", "size"]).reset_index()
        rate = rate.rename(columns={"mean": "lick_probability", "size": "n_trials", "block_perf_type": "state"})
        rows.append(rate.assign(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0]))
    df = pd.concat(rows, ignore_index=True)[["mouse_id", "session_id", "cohort", "state", "trial_type",
                                             "lick_probability", "n_trials"]]
    tests = {}
    for c in COHORTS:
        for tt in ["whisker_trial", "auditory_trial", "no_stim_trial"]:
            w = (df[(df.cohort == c) & (df.trial_type == tt) & df.state.isin(["low", "high"])]
                 .pivot_table(index="mouse_id", columns="state", values="lick_probability").dropna())
            if len(w) < 3:
                continue
            t = stats.ttest_rel(w.high, w.low)
            tests[f"{c} {tt}"] = {"n_mice": len(w), "wilcoxon_p": float(stats.wilcoxon(w.high, w.low).pvalue),
                                  "paired_t": float(t.statistic), "paired_t_p": float(t.pvalue)}
    return df, {"n": n_by_cohort(df), "stats_high_vs_low": tests}


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


def compute_l(ctx):
    n_trials = PANELS["l"]["params"]["n_trials"]
    rows = []
    for mouse, r in _anchors(ctx).items():
        post = r["day0"][r["day0"].index > r["anchor0"]]
        whisk = post.index[post.trial_type == "whisker_trial"]
        if len(whisk) < n_trials:
            continue
        ns = post[post.trial_type == "no_stim_trial"]
        for i in range(n_trials):
            rows.append(dict(mouse_id=mouse, cohort=r["cohort"], whisker_trial_after_anchor=i + 1,
                             cumulative_fa=int(ns[ns.index <= whisk[i]].lick_flag.sum()),
                             n_aud_rewards_at_anchor=r["n_aud"]))
    df = pd.DataFrame(rows)
    return df, {"n": n_by_cohort(df)}


def compute_m(ctx):
    n_trials = PANELS["m"]["params"]["n_trials"]
    rows = []
    for mouse, r in _anchors(ctx).items():
        if "daym1" not in r:
            continue
        ns = r["daym1"][(r["daym1"].index > r["anchor_m1"]) & (r["daym1"].trial_type == "no_stim_trial")]
        if len(ns) < n_trials:
            continue
        cum = ns.lick_flag.iloc[:n_trials].cumsum().to_numpy()
        rows += [dict(mouse_id=mouse, cohort=r["cohort"], no_stim_trial_after_anchor=i + 1, cumulative_fa=int(c),
                      n_aud_rewards_matched=r["n_aud"]) for i, c in enumerate(cum)]
    df = pd.DataFrame(rows)
    return df, {"n": n_by_cohort(df)}


def compute_n(ctx):
    n_early = PANELS["n"]["params"]["n_trials_early"]
    rows = []
    for mouse, r in _anchors(ctx).items():
        if "daym1" not in r:
            continue
        ns0 = r["day0"][(r["day0"].index > r["anchor0"]) & (r["day0"].trial_type == "no_stim_trial")]
        nsm = r["daym1"][(r["daym1"].index > r["anchor_m1"]) & (r["daym1"].trial_type == "no_stim_trial")]
        if len(ns0) < n_early or len(nsm) < n_early:
            continue
        fa0, fam = ns0.lick_flag.iloc[:n_early].mean(), nsm.lick_flag.iloc[:n_early].mean()
        rows.append(dict(mouse_id=mouse, cohort=r["cohort"], fa_day0=fa0, fa_daym1=fam, delta_fa=fa0 - fam,
                         n_aud_anchor=r["n_aud"]))
    df = pd.DataFrame(rows)
    a, b = df[df.cohort == "R+"].delta_fa.to_numpy(), df[df.cohort == "R-"].delta_fa.to_numpy()
    # one-sided permutation test R+ > R- on the difference of means, as in plot_fa_rate_day_change
    rng = np.random.default_rng(0)
    pooled, obs = np.r_[a, b], a.mean() - b.mean()
    perm = np.empty(10000)
    for i in range(len(perm)):
        x = rng.permutation(pooled)
        perm[i] = x[:len(a)].mean() - x[len(a):].mean()
    within = {c: {"wilcoxon_p": float(stats.wilcoxon(df[df.cohort == c].delta_fa).pvalue),
                  "t_p": float(stats.ttest_1samp(df[df.cohort == c].delta_fa, 0).pvalue)} for c in COHORTS}
    return df, {"n": n_by_cohort(df), "stats": two_group_tests(a, b),
                "permutation": {"diff_means": float(obs), "p_one_sided": float((np.sum(perm >= obs) + 1) / (len(perm) + 1)),
                                "n_perm": len(perm)},
                "stats_delta_vs_0": within}


def compute_r(ctx):
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


def compute_o(ctx):
    rng = np.random.default_rng(0)
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        curves, _, _ = session_curves(st)
        w, fa = curves.p_mean.to_numpy(), curves.p_chance.to_numpy()
        null = shift_null(w, fa, rng)
        for lag in LAGS:
            rows.append(dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0], lag=int(lag),
                             r=lagged_corr(w, fa, lag), null_mean_lag0=float(np.nanmean(null)),
                             null_sd_lag0=float(np.nanstd(null))))
    df = pd.DataFrame(rows)
    lag0 = df[df.lag == 0].assign(r_minus_null=lambda d: d.r - d.null_mean_lag0)
    vs_null = {c: {"median_r": float(lag0[lag0.cohort == c].r.median()),
                   "wilcoxon_p": float(stats.wilcoxon(lag0[lag0.cohort == c].r_minus_null.dropna()).pvalue),
                   "t_p": float(stats.ttest_1samp(lag0[lag0.cohort == c].r_minus_null.dropna(), 0).pvalue)}
               for c in COHORTS}
    return df, {"n": n_by_cohort(df), "lag0_vs_shift_null": vs_null,
                "lag0_rplus_vs_rminus": two_group_tests(lag0[lag0.cohort == "R+"].r, lag0[lag0.cohort == "R-"].r)}


MOVEMENT_COLS = ["jaw_motion", "nose_motion", "whisker_motion", "pupil_area"]


def _movement_sessions(ctx):
    """Per day-0 session with video: exact curves aligned trial by trial with the pre-stimulus movements."""
    if "movement_sessions" in ctx:
        return ctx["movement_sessions"]
    import shared_movements
    mov = shared_movements.load()
    out = {}
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        m = mov[mov.session_id == sid].sort_values("start_time")
        if m.empty:
            continue
        curves, _, _ = session_curves(st)
        if len(curves) != len(m):
            print("skip", sid, "curve and movement trials differ", len(curves), len(m))
            continue
        out[sid] = pd.concat([curves.reset_index(drop=True),
                              m[["mouse_id", "session_id", "cohort", "trial_id"] + MOVEMENT_COLS].reset_index(drop=True)],
                             axis=1)
    ctx["movement_sessions"] = out
    return out


def compute_p(ctx):
    mouse = PANELS["e"]["params"]["example"]  # same mouse as the R+ example
    sessions = _movement_sessions(ctx)
    df = next(d for d in sessions.values() if d.mouse_id.iloc[0] == mouse).copy()
    for c in MOVEMENT_COLS:
        df[f"{c}_z"] = (df[c] - df[c].mean()) / df[c].std()
    df["performance"] = df.p_mean - df.p_chance
    return df, {"mouse_id": mouse, "n_whisker_trials": len(df)}


def compute_q(ctx):
    rng = np.random.default_rng(0)
    rows = []
    for sid, d in _movement_sessions(ctx).items():
        perf = (d.p_mean - d.p_chance).to_numpy()
        for c in MOVEMENT_COLS:
            x = d[c].to_numpy(float)
            if np.isfinite(x).sum() < 20:
                continue
            null = shift_null(perf, x, rng)
            r = lagged_corr(perf, x, 0)
            rows.append(dict(mouse_id=d.mouse_id.iloc[0], session_id=sid, cohort=d.cohort.iloc[0], movement=c, r=r,
                             null_mean=float(np.nanmean(null)), null_sd=float(np.nanstd(null)),
                             r_minus_null=r - float(np.nanmean(null)),
                             z=(r - np.nanmean(null)) / np.nanstd(null)))
    df = pd.DataFrame(rows)
    tests = {}
    for c in MOVEMENT_COLS:
        d = df[df.movement == c]
        tests[c] = {"rplus_vs_rminus": two_group_tests(d[d.cohort == "R+"].r_minus_null, d[d.cohort == "R-"].r_minus_null),
                    **{f"{k}_vs_null": {"n": int((d.cohort == k).sum()),
                                        "wilcoxon_p": float(stats.wilcoxon(d[d.cohort == k].r_minus_null.dropna()).pvalue),
                                        "t_p": float(stats.ttest_1samp(d[d.cohort == k].r_minus_null.dropna(), 0).pvalue)}
                       for k in COHORTS}}
    return df, {"n": n_by_cohort(df), "stats": tests}


# ---- behavioural GLM v3 (glm_v3_run.py writes Figures/_cache/glm_v3/<mouse>/<mouse>_<session>_*)
GLM_FEATURES = {"lick_whisker_trial_n-1": "Lick n-1", "lick_whisker_trial_n-2": "Lick n-2",
                "lick_whisker_trial_n-3": "Lick n-3", "spontaneous_lick_rate": "Spont. lick rate",
                "time_since_reward": "Time since reward", "cumulative_reward_fraction": "Cum. reward",
                "jaw_speed_pre": "Jaw speed", "nose_speed_pre": "Nose speed", "whisker_speed_pre": "Whisker speed",
                "pupil_area_pre": "Pupil area", "all_movement": "All movements"}


def _glm_dir():
    from common import CACHE_DIR
    return CACHE_DIR / "glm_v3"


def _glm_feature(name):
    return name.replace("Shuffled ", "").removesuffix("_t0")


def _glm_mice(ctx):
    d0 = day0_table(ctx["table"])
    return d0.groupby("session_id").agg(mouse_id=("mouse_id", "first"), cohort=("cohort", "first")).reset_index()


def _glm_files(ctx, kind):
    out = []
    for r in _glm_mice(ctx).itertuples():
        f = _glm_dir() / r.mouse_id / f"{r.mouse_id}_{r.session_id}_{kind}"
        if f.exists():
            out.append((r, f))
    return out


def compute_t(ctx):
    import pickle
    mouse = PANELS["e"]["params"]["example"]
    r, f = next((r, f) for r, f in _glm_files(ctx, "results.pkl") if r.mouse_id == mouse)
    res = pickle.load(open(f, "rb"))
    y, pred = np.asarray(res["y_all"]), np.asarray(res["y_pred_proba_oof"])
    df = pd.DataFrame({"mouse_id": mouse, "session_id": r.session_id, "cohort": r.cohort,
                       "whisker_trial": np.arange(1, len(y) + 1), "lick": y, "predicted_p_lick": pred,
                       "observed_rolling10": pd.Series(y).rolling(10, center=True, min_periods=3).mean()})
    return df, {"mouse_id": mouse, "n_whisker_trials": len(df), "test_auc": float(res["result"]["test_auc"])}


def compute_u(ctx):
    rows = [pd.read_csv(f).assign(mouse_id=r.mouse_id, session_id=r.session_id, cohort=r.cohort)
            for r, f in _glm_files(ctx, "results_metrics.csv")]
    df = pd.concat(rows, ignore_index=True)[["mouse_id", "session_id", "cohort", "test_auc", "test_balanced_accuracy",
                                             "test_pseudo_r2", "n_trials", "lick_rate"]]
    auc = {c: df[df.cohort == c].test_auc.dropna() for c in COHORTS}
    vs_chance = {c: {"n": len(auc[c]), "median_auc": float(auc[c].median()),
                     "wilcoxon_p": float(stats.wilcoxon(auc[c] - 0.5).pvalue),
                     "t_p": float(stats.ttest_1samp(auc[c], 0.5).pvalue)} for c in COHORTS}
    return df, {"n": n_by_cohort(df), "n_missing_auc": int(df.test_auc.isna().sum()), "auc_vs_chance": vs_chance,
                "stats": two_group_tests(df[df.cohort == "R+"].test_auc, df[df.cohort == "R-"].test_auc)}


def compute_v(ctx):
    rows = [pd.read_csv(f).assign(mouse_id=r.mouse_id, session_id=r.session_id, cohort=r.cohort)
            for r, f in _glm_files(ctx, "coefficients.csv")]
    df = pd.concat(rows, ignore_index=True)
    df["feature"] = df.feature.map(_glm_feature)
    df = df[["mouse_id", "session_id", "cohort", "feature", "coefficient"]]
    order = df.groupby("feature").coefficient.apply(lambda v: v.abs().mean()).sort_values(ascending=False).index
    df["rank_by_mean_abs"] = df.feature.map({f: i for i, f in enumerate(order)})
    tests = {f: {"rplus_vs_rminus": two_group_tests(df[(df.feature == f) & (df.cohort == "R+")].coefficient,
                                                    df[(df.feature == f) & (df.cohort == "R-")].coefficient),
                 **{f"{c}_vs_0_wilcoxon_p": float(stats.wilcoxon(df[(df.feature == f) & (df.cohort == c)].coefficient).pvalue)
                    for c in COHORTS}}
             for f in order}
    return df, {"n": n_by_cohort(df), "order": list(order), "stats": tests}


def compute_w(ctx):
    rows = [pd.read_csv(f).assign(mouse_id=r.mouse_id, session_id=r.session_id, cohort=r.cohort)
            for r, f in _glm_files(ctx, "shuffle_importance.csv")]
    df = pd.concat(rows, ignore_index=True)
    df = df[df.model != "Full Model"]
    df["feature"] = df.model.map(_glm_feature)
    df = df[["mouse_id", "session_id", "cohort", "feature", "delta_test_auc", "delta_test_balanced_accuracy"]]
    tests = {f: {"rplus_vs_rminus": two_group_tests(df[(df.feature == f) & (df.cohort == "R+")].delta_test_auc,
                                                    df[(df.feature == f) & (df.cohort == "R-")].delta_test_auc),
                 **{f"{c}_vs_0_wilcoxon_p": float(stats.wilcoxon(df[(df.feature == f) & (df.cohort == c)].delta_test_auc).pvalue)
                    for c in COHORTS}}
             for f in df.feature.unique()}
    return df, {"n": n_by_cohort(df), "stats": tests}


def compute_k(ctx):
    grid = np.linspace(0, 1, 51)
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        curves, _, _ = session_curves(st)
        tw = st[st.trial_type == "whisker_trial"].start_time.to_numpy()
        x = (tw - tw[0]) / (tw[-1] - tw[0])  # session time from the first to the last whisker trial
        perf = np.interp(grid, x, (curves.p_mean - curves.p_chance).to_numpy())
        rows.append(pd.DataFrame({"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0],
                                  "learning_category": st.learning_category.iloc[0], "session_time_norm": grid,
                                  "performance": perf}))
    df = pd.concat(rows, ignore_index=True)
    return df, {"n": n_by_cohort(df)}


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
        pid = panel["id"]
        fn = globals().get(f"compute_{pid}")
        if fn is None or "data" not in panel or (args.panels and pid not in args.panels):
            continue
        df, info = fn(ctx)
        df.to_csv(out / "data" / panel["data"], index=False)
        meta[pid] = info
        print(pid, panel["data"], len(df), "rows")
    meta_file.write_text(json.dumps(meta, indent=1, default=str))  # serialise fully before writing


if __name__ == "__main__":
    main()
