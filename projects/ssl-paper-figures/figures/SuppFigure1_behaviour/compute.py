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


def compute_lick_course(ctx):
    return s2.compute_lick_course(ctx)


def compute_rt_hits_other(ctx):
    return s2.compute_rt_hits_other(ctx)


def compute_rt_curves(ctx):
    return s2.compute_rt_curves(ctx)


def compute_rt_change_auditory(ctx):
    """Auditory-hit version of Figure 1 k left (user 2026-11-10): reaction-time change from the mouse's first 5 hits."""
    return fig1.rt_change_from_start(ctx, "auditory_trial")


def compute_posture_curve(ctx):
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
