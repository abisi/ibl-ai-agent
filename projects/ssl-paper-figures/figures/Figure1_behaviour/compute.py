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


def compute_w(ctx):
    return compute_example(ctx, "w")


def compute_c(ctx):
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


def compute_h(ctx):
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
