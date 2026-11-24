"""SuppFigure1_behaviour -- one compute_<key>(ctx) per panel, returning (data frame for the panel CSV, info for the caption).
Reuses the Figure 1 functions (same shared behaviour table and conventions).

Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/SuppFigure1_behaviour/compute.py [--date D]
then build.py figures/SuppFigure1_behaviour.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as stats_mod

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import shared_behaviour  # noqa: E402
from common import load_spec, snapshot_dir, two_group_tests  # noqa: E402

SPEC = load_spec(FIG_DIR)
PANELS = {p.get("key", p["id"]): p for p in SPEC["panels"]}
_spec = importlib.util.spec_from_file_location("fig1_compute", FIG_DIR.parent / "Figure1_behaviour" / "compute.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

TRIAL_TYPES = ["whisker_trial", "auditory_trial", "no_stim_trial"]
# panels moved here from Figure 1 (movement) and the deleted SuppFigure2 (lick course, latencies, RT-curve and
# posture-curve correlations; planner 2026-11-09): helpers in correlates_compute.py / correlates_plot.py
_spec2 = importlib.util.spec_from_file_location("s1_correlates_compute", FIG_DIR / "correlates_compute.py")
s2 = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(s2)


def compute_movement_peth(ctx):
    return fig1.compute_movement_peth(ctx)


def compute_first_lick(ctx):  # moved from Figure 1 g (user 2026-11-16)
    return fig1.compute_first_lick(ctx)


def compute_hitmiss_whisker_speed(ctx):  # user 2026-11-16: hit / miss per cohort, one panel per DLC measure
    return fig1.compute_movement_hitmiss(ctx, "whisker_speed")


def compute_hitmiss_jaw_opening(ctx):
    return fig1.compute_movement_hitmiss(ctx, "jaw_opening")


def compute_hitmiss_pupil_area(ctx):
    return fig1.compute_movement_hitmiss(ctx, "pupil_area")


def compute_hitmiss_nose_speed(ctx):
    return fig1.compute_movement_hitmiss(ctx, "nose_speed")


def compute_lick_course(ctx):
    return s2.compute_lick_course(ctx)


def compute_learners(ctx):
    return fig1.compute_learners(ctx)  # moved from Figure 1 (planner 2026-11-11)


def compute_rt_curves(ctx):
    return s2.compute_rt_curves(ctx)


def compute_xcorr(ctx):
    """Moved from Figure 1 (planner 2026-11-11)."""
    return fig1.compute_xcorr(ctx)


AUD_RT_WINDOWS = {"early": (0, 5), "late": (21, 30)}  # whisker trials from the first whisker hit (Fig. 1K bins)


def compute_rt_change_auditory(ctx):
    """User 2026-11-22 ('same timescale as the whisker counterpart, two dots'): auditory hits placed on the whisker-trial
    axis of Fig. 1I / K (index of the preceding whisker trial from the first whisker hit; cut session, early licks
    dropped, reaction time 0-1 s). Per mouse, median auditory RT in the early (whisker trials 0-5) and late (21-30)
    windows and their difference (late - early). Within cohort, Wilcoxon (and one-sample t) of the difference vs 0;
    R+ vs R- Mann-Whitney and Welch on the difference. The same for whisker hits (the Fig. 1K counterpart)."""
    rows = []
    for sid, st in fig1.day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        st = fig1._cut_session(st)
        isw = (st.trial_type == "whisker_trial").to_numpy()
        lick = (st.lick_flag == 1).to_numpy()
        if not (isw & lick).any():
            continue
        widx = np.cumsum(isw) - 1  # index of the current / preceding whisker trial
        rel = widx - widx[np.argmax(isw & lick)]
        rt = (st.computed_reaction_time.where(st.computed_reaction_time.between(0, 1)) * 1000).to_numpy()
        for tt in ("auditory_trial", "whisker_trial"):
            sel = (st.trial_type == tt).to_numpy() & lick & np.isfinite(rt)
            for w, (a, b) in AUD_RT_WINDOWS.items():
                m = sel & (rel >= a) & (rel <= b)
                rows.append(dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0], trial_type=tt,
                                 window=w, n_hits=int(m.sum()), median_rt_ms=np.median(rt[m]) if m.any() else np.nan))
    df = pd.DataFrame(rows)
    w = df.pivot_table(index=["mouse_id", "cohort", "trial_type"], columns="window", values="median_rt_ms").dropna()
    w["late_minus_early_ms"] = w["late"] - w["early"]
    w = w.reset_index()
    tests = {}
    for tt, g in w.groupby("trial_type"):
        tests[tt] = {**{c: {"n": int((g.cohort == c).sum()),
                            "mean_diff": float(g[g.cohort == c].late_minus_early_ms.mean()),
                            "sem_diff": float(g[g.cohort == c].late_minus_early_ms.sem()),
                            "wilcoxon_p": float(stats_mod.wilcoxon(g[g.cohort == c].late_minus_early_ms).pvalue),
                            "t_p": float(stats_mod.ttest_1samp(g[g.cohort == c].late_minus_early_ms, 0).pvalue)}
                        for c in ("R+", "R-")},
                     "rplus_vs_rminus": two_group_tests(g[g.cohort == "R+"].late_minus_early_ms,
                                                        g[g.cohort == "R-"].late_minus_early_ms)}
    out = df.merge(w[["mouse_id", "trial_type", "late_minus_early_ms"]], on=["mouse_id", "trial_type"], how="left")
    out["paired"] = out.late_minus_early_ms.notna()
    return out, {"n": fig1.n_by_cohort(w[w.trial_type == "auditory_trial"]), "tests": tests,
                 "windows": {k: list(v) for k, v in AUD_RT_WINDOWS.items()}}


def compute_glm_schematic(ctx):  # moved from Figure 1 (user 2026-11-22: only the weights panel stays there)
    return fig1.compute_glm_schematic(ctx)


def compute_glm_example(ctx):
    return fig1.compute_glm_example(ctx)


def compute_glm_fit(ctx):
    return fig1.compute_glm_fit(ctx)


def compute_posture_curve(ctx):
    return s2.compute_posture_curve(ctx)


def compute_movement_curve(ctx):  # user 2026-11-23: posture and movement as two panels (same data)
    return s2.compute_posture_curve(ctx)


LEARNER = ["good", "moderate"]


def learners(table):
    return table[table.learning_category.isin(LEARNER)]


def compute_pretraining(ctx):
    """Planner 2026-11-09: learners."""
    return fig1.compute_pretraining({"table": learners(ctx["table"])})


def _example(ctx, key):
    fig1.PANELS[key] = {"params": {"example": PANELS[key]["params"]["example"]}}
    return fig1.compute_example(ctx, key)


def compute_ex_rplus_nonlearner(ctx):
    return _example(ctx, "ex_rplus_nonlearner")


def compute_ex_rminus_nonlearner(ctx):
    return _example(ctx, "ex_rminus_nonlearner")


def compute_day0_closeup(ctx):
    """Per mouse: lick probability per trial type on whisker day 0 (early licks dropped; session mean, then mouse mean),
    learner mice (planner 2026-11-09)."""
    t = learners(ctx["table"])
    d0 = t[(t.behavior == "whisker") & (t.day == 0) & (t.early_lick == 0) & t.trial_type.isin(TRIAL_TYPES)]
    s = d0.groupby(["mouse_id", "cohort", "session_id", "trial_type"]).lick_flag.mean().reset_index()
    df = (s.groupby(["mouse_id", "cohort", "trial_type"]).lick_flag.mean().rename("lick_probability").reset_index())
    stats = {tt: two_group_tests(df[(df.trial_type == tt) & (df.cohort == "R+")].lick_probability,
                                 df[(df.trial_type == tt) & (df.cohort == "R-")].lick_probability) for tt in TRIAL_TYPES}
    return df, {"n": fig1.n_by_cohort(df), "stats": stats}


def compute_day0_dprime(ctx):
    """User 2026-11-23: as day0_closeup, d' per mouse on whisker day 0: whisker d' = z(whisker lick rate) - z(false-alarm
    rate), auditory d' = z(auditory lick rate) - z(false-alarm rate), log-linear correction ((k + 0.5) / (n + 1)), per
    session, then mouse mean; learners, early licks dropped."""
    t = learners(ctx["table"])
    d0 = t[(t.behavior == "whisker") & (t.day == 0) & (t.early_lick == 0) & t.trial_type.isin(TRIAL_TYPES)]
    g = d0.groupby(["mouse_id", "cohort", "session_id", "trial_type"]).lick_flag.agg(["sum", "count"]).unstack("trial_type")
    z = lambda tt: stats_mod.norm.ppf(_rate(g[("sum", tt)], g[("count", tt)]))
    fa = z("no_stim_trial")
    rows = []
    for tt in ("whisker_trial", "auditory_trial"):
        dp = pd.Series(np.asarray(z(tt) - fa), index=g.index).dropna()
        rows.append(dp.groupby(["mouse_id", "cohort"]).mean().rename("dprime").reset_index().assign(trial_type=tt))
    df = pd.concat(rows, ignore_index=True)
    tests = {tt: {"rplus_vs_rminus": two_group_tests(df[(df.trial_type == tt) & (df.cohort == "R+")].dprime,
                                                     df[(df.trial_type == tt) & (df.cohort == "R-")].dprime),
                  **{c: fig1.vs_zero(df[(df.trial_type == tt) & (df.cohort == c)].dprime) for c in ("R+", "R-")}}
             for tt in ("whisker_trial", "auditory_trial")}
    return df, {"n": fig1.n_by_cohort(df), "stats": tests}


def compute_glm_shuffle_groups(ctx):
    """User 2026-11-23: shuffle controls of the GLM: drop in held-out log-likelihood per trial when a predictor
    (glm_v7 shuffle_importance) or a group of predictors (glm_v7_groups.py group_shuffle: reward history, movement, lick
    history) is permuted in the training and test trials and the model refitted (10 permutations x 10 repeats)."""
    s, info = fig1.compute_glm_shuffle(ctx)
    s = s[~s.feature.isin(["all_movement"])]
    single = s.rename(columns={"delta_test_ll_per_trial": "value"})[["mouse_id", "session_id", "cohort", "feature",
                                                                      "value"]].assign(metric="shuffle_single")
    grp = pd.read_csv(fig1._glm_dir() / "_groups" / "group_shuffle_all.csv")
    grp = grp[grp.session_id.isin(set(single.session_id))]
    grp = (grp.rename(columns={"group": "feature", "delta_test_ll_per_trial": "value"})
           [["mouse_id", "session_id", "cohort", "feature", "value"]]
           .assign(metric="shuffle_group", group_members=lambda d: d.feature.map(lambda g: ";".join(fig1.GLM_GROUPS[g]))))
    df = pd.concat([single, grp], ignore_index=True)
    df["rank"] = df.feature.map({f: i for i, f in enumerate(fig1.GLM_ORDER)})
    return df, {"n": fig1.n_by_cohort(single), "order": fig1.GLM_ORDER,
                "stats_single": fig1._feature_tests(single, "value"), "stats_group": fig1._feature_tests(grp, "value")}


PREV = ["prev_whisker_hit", "prev_whisker_miss", "prev_auditory_hit", "prev_auditory_miss"]


def compute_history(ctx):
    """Trial history as behaviour_analysis beh_history/behavior_history_analysis (prepare, _hit_rate_given_prev,
    _rt_given_prev): whisker and auditory trials of whisker day 0 in sequence (learners; early licks dropped), each trial's
    previous stimulus trial (whisker hit / miss, auditory hit / miss, within the session); per mouse, P(lick) given each
    previous condition and the reaction-time shift (median RT given the condition minus the mouse's median RT of that
    trial type; hits, RT = computed_reaction_time, 0-1 s)."""
    d0 = fig1.day0_table(ctx["table"], learners_only=True)
    d0 = d0[(d0.early_lick == 0) & d0.trial_type.isin(["whisker_trial", "auditory_trial"])].sort_values("start_time")
    rows = []
    for sid, s in d0.groupby("session_id"):
        s = s.copy()
        pt, pl = s.trial_type.shift(1), s.lick_flag.shift(1)
        s["prev"] = np.select([(pt == "whisker_trial") & (pl == 1), (pt == "whisker_trial") & (pl == 0),
                               (pt == "auditory_trial") & (pl == 1), (pt == "auditory_trial") & (pl == 0)], PREV, "none")
        rt = s.computed_reaction_time.where((s.lick_flag == 1) & s.computed_reaction_time.between(0, 1))
        s["rt"] = rt
        for tt in ("whisker_trial", "auditory_trial"):
            cur = s[s.trial_type == tt]
            med, overall = cur.rt.median(), cur.lick_flag.mean()
            for cond in PREV:
                c = cur[cur.prev == cond]
                rows.append({"mouse_id": s.mouse_id.iloc[0], "session_id": sid, "cohort": s.cohort.iloc[0],
                             "trial_type": tt, "prev": cond, "n_trials": len(c),
                             "p_lick": c.lick_flag.mean() if len(c) else np.nan,
                             "p_lick_shift": (c.lick_flag.mean() - overall) if len(c) else np.nan,
                             "rt_shift_s": (c.rt.median() - med) if c.rt.notna().any() else np.nan})
    df = pd.DataFrame(rows)
    stats = {f"{tt} {cond} {v}": two_group_tests(df[(df.trial_type == tt) & (df.prev == cond) & (df.cohort == "R+")][v],
                                                 df[(df.trial_type == tt) & (df.prev == cond) & (df.cohort == "R-")][v])
             for tt in ("whisker_trial", "auditory_trial") for cond in PREV for v in ("p_lick_shift", "rt_shift_s")}
    return df, {"n": fig1.n_by_cohort(df), "stats": stats}


RATE_VARS = {"aud_day0": "Auditory P(lick) day 0", "fa_day0": "No-stim P(lick) day 0",
             "aud_daym1": "Auditory P(lick) day -1", "fa_daym1": "No-stim P(lick) day -1",
             "wh_day1": "Whisker P(lick) day +1", "wh_day2": "Whisker P(lick) day +2",
             "n_pretraining_days": "Pretraining days", "n_pretraining_trials": "Pretraining trials",
             "rt_wh_day0": "Whisker RT day 0 (s)"}  # planner 2026-11-09


def compute_rates_correlation(ctx):
    """Per mouse, whisker P(lick) on day 0 against other session measures, as behaviour_analysis
    beh_plotting_functions.plot_multi_mouse_whisker_session_rates_correlation (relative weight not available in the shared
    table); Pearson r and OLS per cohort. Learner mice (planner 2026-11-09), early licks dropped."""
    t = learners(ctx["table"])
    t = t[t.early_lick == 0]
    p = (t.groupby(["mouse_id", "cohort", "day", "session_id", "trial_type"]).lick_flag.mean()
         .groupby(["mouse_id", "cohort", "day", "trial_type"]).mean().unstack(["day", "trial_type"]))
    get = lambda d, tt: p.get((d, tt))
    w = pd.DataFrame({"wh_day0": get(0, "whisker_trial"), "fa_day0": get(0, "no_stim_trial"),
                      "aud_day0": get(0, "auditory_trial"), "wh_day1": get(1, "whisker_trial"),
                      "wh_day2": get(2, "whisker_trial"), "aud_daym1": get(-1, "auditory_trial"),
                      "aud_daym2": get(-2, "auditory_trial"), "fa_daym1": get(-1, "no_stim_trial"),
                      "fa_daym2": get(-2, "no_stim_trial")}).reset_index()
    pre, _ = fig1.compute_pretraining({"table": t})
    w = w.merge(pre[["mouse_id", "n_pretraining_trials", "n_auditory_days"]].rename(
        columns={"n_auditory_days": "n_pretraining_days"}), on="mouse_id", how="left")
    d0 = t[(t.behavior == "whisker") & (t.day == 0) & (t.trial_type == "whisker_trial") & (t.lick_flag == 1)]
    rt = d0[d0.computed_reaction_time > 0].groupby("mouse_id").computed_reaction_time.median().rename("rt_wh_day0")
    w = w.merge(rt, on="mouse_id", how="left")
    stats = {}
    for v in RATE_VARS:
        for c in ("R+", "R-"):
            d = w[(w.cohort == c)][["wh_day0", v]].dropna()
            if len(d) > 3:
                r = stats_mod.linregress(d[v], d.wh_day0)
                stats[f"{v} {c}"] = {"r": float(r.rvalue), "p": float(r.pvalue), "n": len(d)}
    return w, {"n": fig1.n_by_cohort(w), "stats": stats}


# ---------------------------------------------------------------- user 2026-11-18: reorganised Supp. Fig. 1

def _learner_mice(ctx):
    return set(fig1.day0_table(ctx["table"], learners_only=True).mouse_id)


def compute_cumulative_fa(ctx):
    """Day -1 versus day 0 false alarms (user 2026-11-18): cumulative false alarms after the anchor (Figure 1 compute
    _anchors: day 0 from the first whisker lick, day -1 from the matched auditory-reward count), learners."""
    n = PANELS["cumulative_fa"]["params"]["n_trials"]
    fig1.PANELS["cumulative_fa"] = {"params": {"n_trials": n}}
    df, _ = fig1.compute_cumulative_fa(ctx)
    df = df[df.mouse_id.isin(_learner_mice(ctx))].reset_index(drop=True)
    last = df[df.trial_after_anchor == n]
    tests = {f"day{d}": two_group_tests(last[(last.day == d) & (last.cohort == "R+")].cumulative_fa,
                                       last[(last.day == d) & (last.cohort == "R-")].cumulative_fa) for d in (0, -1)}
    paired = {}
    for c in ("R+", "R-"):  # day 0 minus day -1 within mouse (mice with both days)
        w = last[last.cohort == c].pivot_table(index="mouse_id", columns="day", values="cumulative_fa").dropna()
        paired[c] = {"n": int(len(w)), "mean_diff": float((w[0] - w[-1]).mean()) if len(w) else np.nan,
                     "wilcoxon_p": float(stats_mod.wilcoxon(w[0], w[-1]).pvalue) if len(w) > 5 else np.nan}
    return df, {"n": {f"day{d}": fig1.n_by_cohort(df[df.day == d]) for d in (0, -1)}, "stats": tests,
                "day0_vs_daym1": paired, "n_trials": n}


def _rate(k, n):
    """Lick rate with the log-linear correction (k + 0.5) / (n + 1), for d'."""
    return (k + 0.5) / (n + 1)


def compute_days_expert(ctx):
    """User 2026-11-18: every day separately, learners, early licks dropped. Lick probability per trial type on auditory
    days -2, -1 and whisker days 0..3 (session mean, then mouse mean); d' on whisker days, z(whisker hit rate) -
    z(false-alarm rate) per session with the log-linear correction, then mouse mean. R+ vs R- per day."""
    t = learners(ctx["table"])
    t = t[t.day.isin([-2, -1, 0, 1, 2]) & (t.early_lick == 0) & t.trial_type.isin(TRIAL_TYPES)]  # user 2026-11-23: to day 2
    g = t.groupby(["mouse_id", "cohort", "day", "session_id", "trial_type"]).lick_flag.agg(["sum", "count", "mean"])
    p = g["mean"].groupby(["mouse_id", "cohort", "day", "trial_type"]).mean().rename("value").reset_index()
    p["measure"] = "p_lick"
    w = g.unstack("trial_type")
    w = w[w.index.get_level_values("day") >= 0].dropna(subset=[("count", "whisker_trial"), ("count", "no_stim_trial")])
    z = stats_mod.norm.ppf
    dp = (z(_rate(w[("sum", "whisker_trial")], w[("count", "whisker_trial")]))
          - z(_rate(w[("sum", "no_stim_trial")], w[("count", "no_stim_trial")])))
    dp = pd.Series(np.asarray(dp), index=w.index).groupby(["mouse_id", "cohort", "day"]).mean().rename("value").reset_index()
    dp["measure"], dp["trial_type"] = "dprime", "whisker_trial"
    df = pd.concat([p, dp], ignore_index=True)
    tests = {}
    for (m, tt, day), d in df.groupby(["measure", "trial_type", "day"]):
        a, b = d[d.cohort == "R+"].value, d[d.cohort == "R-"].value
        if len(a) > 2 and len(b) > 2:
            tests[f"{m} {tt} day{day}"] = two_group_tests(a, b)
    n = {c: {int(day): int(df[(df.cohort == c) & (df.day == day)].mouse_id.nunique()) for day in sorted(df.day.unique())}
         for c in ("R+", "R-")}
    dprime = {k.split()[-1]: v for k, v in tests.items() if k.startswith("dprime")}
    return df, {"n": n, "stats": tests, "dprime_by_day": dprime}


DPRIME_CLIP = (0.01, 0.99)


def _dprime_session(ctx):
    """User 2026-11-18: d' = z(whisker curve) - z(chance curve) on whisker trials (curves refitted on the cut session as
    Figure 1 H/J, clipped to DPRIME_CLIP), learners; against normalised session time (alignment 0 to session end 1) and
    absolute time (min from the alignment, 1-min grid, inside the session only) and whisker trials from the alignment, aligned at the first whisker trial and
    at the first whisker hit. Cluster-based permutation test R+ vs R- per alignment x time axis (times with >= 8 mice per
    cohort)."""
    if "dprime_session" in ctx:
        return ctx["dprime_session"]
    z = lambda v: stats_mod.norm.ppf(np.clip(np.asarray(v, float), *DPRIME_CLIP))
    grid = np.round(np.arange(0, 1 + 1e-9, 0.02), 3)
    rows = []
    for sid, st in fig1.day0_table(ctx["table"], learners_only=True).groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        st = st[st.start_time <= fig1.session_end_time(st[st.trial_type == "whisker_trial"].start_time)]
        curves, _, _ = fig1.session_curves(st, None)
        tw = st[st.trial_type == "whisker_trial"].start_time.to_numpy()
        dprime = z(curves.p_mean) - z(curves.p_chance)
        base = {"mouse_id": st.mouse_id.iloc[0], "session_id": sid, "cohort": st.cohort.iloc[0]}
        hits = np.flatnonzero(curves.lick.to_numpy() == 1)
        starts = {"first_whisker_trial": 0}
        if len(hits) and hits[0] < len(tw) - 1:
            starts["first_whisker_hit"] = hits[0]
        for align, k in starts.items():
            t, v = tw[k:], dprime[k:]
            rows.append(pd.DataFrame({**base, "alignment": align, "axis": "norm", "time": grid,
                                      "dprime": np.interp(grid, (t - t[0]) / (t[-1] - t[0]), v)}))
            rows.append(pd.DataFrame({**base, "alignment": align, "axis": "trial", "time": np.arange(len(v)),
                                      "dprime": v}))  # user 2026-11-22: whisker trials from the alignment
            minutes = np.arange(0, np.floor((t[-1] - t[0]) / 60) + 1)
            rows.append(pd.DataFrame({**base, "alignment": align, "axis": "abs_min", "time": minutes,
                                      "dprime": np.interp(minutes, (t - t[0]) / 60, v)}))
    df = pd.concat(rows, ignore_index=True)
    df["cluster_sig"], df["cluster_p"], df["in_test"] = False, np.nan, False
    clusters, n = {}, {}
    for (axis, align), d in df.groupby(["axis", "alignment"]):
        enough = d.groupby(["time", "cohort"]).mouse_id.nunique().unstack().reindex(columns=["R+", "R-"]).fillna(0).min(1)
        d = d[d.time.isin(enough[enough >= 8].index)]
        key = f"{axis}_{align}"
        clusters[key] = fig1.cluster_test(d, "time", "dprime")
        df.loc[d.index, "in_test"] = True
        df.loc[d.index, "cluster_sig"] = fig1.mark_clusters(d, "time", clusters[key])
        df.loc[d.index, "cluster_p"] = fig1.cluster_p(d, "time", clusters[key])
        n[key] = fig1.n_by_cohort(d)
    text = {k: "; ".join(f"{c['start']:g}-{c['end']:g}, P = {c['p']:.3g}" for c in v if c["p"] < 0.05) or "none"
            for k, v in clusters.items()}
    ctx["dprime_session"] = (df, {"n_by_key": n, "clusters": clusters, "sig_clusters_text": text,
                                  "clip": list(DPRIME_CLIP), "session_keep": fig1.SESSION_KEEP})
    return ctx["dprime_session"]


def compute_dprime_norm(ctx):
    df, info = _dprime_session(ctx)
    d = df[df.axis == "norm"].reset_index(drop=True)
    return d, {**info, "n": fig1.n_by_cohort(d)}


DPRIME_MAX_TRIALS = 100  # whisker trials shown from the alignment (user 2026-11-22)


def compute_dprime_trials(ctx):
    """User 2026-11-22: d-prime (whisker vs chance curve) against whisker trials from the first whisker trial / hit (cut
    session), trials with >= 8 mice per cohort, at most DPRIME_MAX_TRIALS."""
    df, info = _dprime_session(ctx)
    d = df[(df.axis == "trial") & (df.time <= DPRIME_MAX_TRIALS)].reset_index(drop=True)
    return d, {**info, "n": fig1.n_by_cohort(d), "max_trials": DPRIME_MAX_TRIALS}


def compute_dprime_abs(ctx):
    df, info = _dprime_session(ctx)
    d = df[df.axis == "abs_min"].reset_index(drop=True)
    return d, {**info, "n": fig1.n_by_cohort(d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*")
    ap.add_argument("--date")
    a = ap.parse_args()
    ctx = {"table": shared_behaviour.load()}
    out = snapshot_dir(SPEC, a.date)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "_build").mkdir(exist_ok=True)
    meta_file = out / "_build" / "panel_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    for panel in SPEC["panels"]:
        pid, key = panel["id"], panel.get("key", panel["id"])
        fn = globals().get(f"compute_{key}")
        if fn is None or "data" not in panel or (a.panels and pid not in a.panels and key not in a.panels):
            continue
        df, info = fn(ctx)
        df.to_csv(out / "data" / panel["data"], index=False)
        meta[pid] = info
        print(pid, key, panel["data"], len(df), "rows", flush=True)
    meta_file.write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
