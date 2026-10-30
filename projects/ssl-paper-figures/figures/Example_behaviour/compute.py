"""Example_behaviour -- compute every panel's tidy table from scratch and write it to the snapshot's data/ folder.

Run on haas (repo root):
    ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Example_behaviour/compute.py [--panels b c] [--rebuild-cache]
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
import shared_behaviour  # noqa: E402
from common import import_external_repos, load_spec, snapshot_dir, two_group_tests  # noqa: E402

import_external_repos()
import behavior_analysis_utils as bhv_utils  # noqa: E402
import learning_curve_exact as lce  # noqa: E402

SPEC = load_spec(FIG_DIR)
CURVE_PARAMS = SPEC["shared"]["day0_learning_curves"]["params"]
COHORTS = ["R+", "R-"]


# ---------------------------------------------------------------- shared inputs

def day0_table(table):
    return table[(table.behavior == "whisker") & (table.day == 0)]


def day0_curves(table):
    """Exact learning curves and learning trial of every whisker day-0 session (one per mouse)."""
    curves = {}
    for session_id, st in day0_table(table).groupby("session_id"):
        st = st[st.early_lick == 0]
        out = lce.session_learning_curves_exact(st, sigma=CURVE_PARAMS["sigma"], conf_int=CURVE_PARAMS["conf_int"])
        if "whisker_trial" not in out or "p_chance" not in out["whisker_trial"]:
            continue
        curves[session_id] = dict(mouse_id=st.mouse_id.iloc[0], cohort=st.cohort.iloc[0], **out["whisker_trial"])
    return curves


def example_sessions(curves):
    """Per cohort, the learner whose learning trial is closest to the cohort median."""
    rows = pd.DataFrame([dict(session_id=s, cohort=c["cohort"], ltrial=c["learning_trial"]) for s, c in curves.items()])
    rows = rows.dropna(subset=["ltrial"]).sort_values("session_id")
    out = {}
    for cohort, grp in rows.groupby("cohort"):
        out[cohort] = grp.loc[(grp.ltrial - grp.ltrial.median()).abs().idxmin(), "session_id"]
    return out


def n_by_cohort(df):
    return {c: {"mice": int(df[df.cohort == c].mouse_id.nunique()),
                "sessions": int(df[df.cohort == c].session_id.nunique())} for c in COHORTS if "session_id" in df}


# ---------------------------------------------------------------- panels

def compute_b(ctx):
    sid = ctx["examples"]["R+"]
    st = ctx["table"][(ctx["table"].session_id == sid) & (ctx["table"].early_lick == 0)].sort_values("start_time")
    df = pd.DataFrame({"mouse_id": st.mouse_id, "session_id": sid, "cohort": st.cohort,
                       "trial_number": np.arange(1, len(st) + 1), "trial_type": st.trial_type,
                       "lick": st.lick_flag.astype(int), "start_time_s": st.start_time,
                       "reaction_time_s": st.computed_reaction_time.where(st.lick_flag == 1)})
    return df, {"n": n_by_cohort(df)}


def _block_table(table, session_id, block_size):
    st, _, _ = bhv_utils.get_standard_single_session_table(table, session_id, block_size=block_size, verbose=False)
    blocks = st.groupby("block").agg(first_trial=("trial", "min"), hr_w=("hr_w", "first"), hr_a=("hr_a", "first"),
                                     hr_n=("hr_n", "first")).reset_index()
    blocks.insert(0, "session_id", session_id)
    blocks.insert(0, "mouse_id", st.mouse_id.iloc[0])
    blocks.insert(2, "cohort", st.cohort.iloc[0])
    return blocks


def compute_c(ctx, cohort="R+", panel="c"):
    block_size = next(p for p in SPEC["panels"] if p["id"] == panel)["params"]["block_size"]
    df = _block_table(day0_table(ctx["table"]), ctx["examples"][cohort], block_size)
    return df, {"n": n_by_cohort(df)}


def compute_d(ctx):
    return compute_c(ctx, cohort="R-", panel="d")


def compute_e(ctx):
    t = ctx["table"][ctx["table"].early_lick == 0]
    per_session = t.groupby(["mouse_id", "cohort", "day", "session_id", "trial_type"]).lick_flag.mean().reset_index()
    df = (per_session.groupby(["mouse_id", "cohort", "day", "trial_type"])
          .agg(lick_probability=("lick_flag", "mean"), session_id=("session_id", lambda s: ";".join(sorted(s))))
          .reset_index())
    return df, {"n": {c: {int(d): int(df[(df.cohort == c) & (df.day == d)].mouse_id.nunique()) for d in sorted(df.day.unique())}
                      for c in COHORTS}}


def compute_f(ctx):
    p = next(p for p in SPEC["panels"] if p["id"] == "f")["params"]
    d0 = day0_table(ctx["table"])
    df = pd.concat([_block_table(d0, s, p["block_size"]) for s in d0.session_id.unique()], ignore_index=True)
    df = df[df.block < p["max_blocks"]]
    return df, {"n": n_by_cohort(df)}


def compute_g(ctx):
    rows = []
    for sid, st in day0_table(ctx["table"]).groupby("session_id"):
        w = st[(st.trial_type == "whisker_trial") & (st.early_lick == 0)].sort_values("start_time")
        licks = np.flatnonzero(w.lick_flag.to_numpy() == 1)
        rows.append(dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0],
                         whisker_trials_before_first_lick=licks[0] if len(licks) else np.nan))
    df = pd.DataFrame(rows)
    col = "whisker_trials_before_first_lick"
    return df, {"n": n_by_cohort(df), "stats": two_group_tests(df[df.cohort == "R+"][col], df[df.cohort == "R-"][col]),
                "no_whisker_lick": int(df[col].isna().sum())}


def compute_h(ctx):
    sid = ctx["examples"]["R+"]
    c = ctx["curves"][sid]
    df = pd.DataFrame({"mouse_id": c["mouse_id"], "session_id": sid, "cohort": c["cohort"],
                       "whisker_trial": np.arange(1, len(c["outcomes"]) + 1), "lick": c["outcomes"],
                       "p_mean": c["p_mean"], "p_low": c["p_low"], "p_high": c["p_high"], "p_chance": c["p_chance"],
                       "learning_trial": c["learning_trial"] + 1})
    return df, {"n": n_by_cohort(df)}


def _learning_table(ctx):
    return pd.DataFrame([dict(mouse_id=c["mouse_id"], session_id=s, cohort=c["cohort"], category=c["mouse_cat"],
                              learning_trial=c["learning_trial"] + 1, learner=not np.isnan(c["learning_trial"]))
                         for s, c in ctx["curves"].items()])


def compute_i(ctx):
    df = _learning_table(ctx)
    lr = df[df.learner]
    return df, {"n": n_by_cohort(df),
                "stats": two_group_tests(lr[lr.cohort == "R+"].learning_trial, lr[lr.cohort == "R-"].learning_trial)}


def compute_j(ctx):
    lt = _learning_table(ctx)
    df = lt.groupby("cohort").learner.agg(learners="sum", mice="count").reset_index()
    df["fraction_learners"] = df.learners / df.mice
    table = [[int(r.learners), int(r.mice - r.learners)] for r in df.set_index("cohort").loc[COHORTS].itertuples()]
    res = stats.fisher_exact(table)
    return df, {"n": n_by_cohort(lt), "stats": {"fisher_odds_ratio": float(res.statistic), "fisher_p": float(res.pvalue)}}


def compute_k(ctx):
    lo, hi = next(p for p in SPEC["panels"] if p["id"] == "k")["params"]["window"]
    rows = []
    for sid, c in ctx["curves"].items():
        if np.isnan(c["learning_trial"]):
            continue
        rel = np.arange(len(c["p_mean"])) - int(c["learning_trial"])
        keep = (rel >= lo) & (rel <= hi)
        rows.append(pd.DataFrame({"mouse_id": c["mouse_id"], "session_id": sid, "cohort": c["cohort"],
                                  "trial_from_learning_trial": rel[keep], "p_mean": c["p_mean"][keep]}))
    df = pd.concat(rows, ignore_index=True)
    return df, {"n": n_by_cohort(df)}


def compute_l(ctx):
    d0 = day0_table(ctx["table"])
    hits = d0[(d0.lick_flag == 1) & (d0.early_lick == 0) & d0.trial_type.isin(["whisker_trial", "auditory_trial"])]
    df = hits[["mouse_id", "session_id", "cohort", "trial_type", "computed_reaction_time"]].rename(
        columns={"computed_reaction_time": "reaction_time_s"})
    df = df[df.reaction_time_s > 0]
    med = df.groupby(["mouse_id", "cohort", "trial_type"]).reaction_time_s.median().reset_index()
    w = med[med.trial_type == "whisker_trial"]
    return df, {"n": n_by_cohort(df), "n_trials": {f"{c} {t}": int(n) for (c, t), n in df.groupby(["cohort", "trial_type"]).size().items()},
                "stats_whisker_mouse_median": two_group_tests(w[w.cohort == "R+"].reaction_time_s,
                                                              w[w.cohort == "R-"].reaction_time_s)}


def compute_m(ctx):
    t = ctx["table"]
    hits = t[(t.lick_flag == 1) & (t.early_lick == 0) & (t.computed_reaction_time > 0)
             & t.trial_type.isin(["whisker_trial", "auditory_trial"])]
    df = (hits.groupby(["mouse_id", "cohort", "day", "trial_type"])
          .agg(median_reaction_time_s=("computed_reaction_time", "median"),
               session_id=("session_id", lambda s: ";".join(sorted(set(s))))).reset_index())
    return df, {"n": {c: int(df[df.cohort == c].mouse_id.nunique()) for c in COHORTS}}


def compute_n(ctx):
    lt = _learning_table(ctx)
    e, _ = compute_e(ctx)
    day1 = e[(e.day == 1) & (e.trial_type == "whisker_trial")][["mouse_id", "lick_probability"]]
    df = lt[lt.learner].merge(day1.rename(columns={"lick_probability": "day1_whisker_lick_probability"}), on="mouse_id")
    out = {}
    for c in COHORTS:
        d = df[df.cohort == c]
        r = stats.pearsonr(d.learning_trial, d.day1_whisker_lick_probability)
        out[c] = {"n_mice": len(d), "pearson_r": float(r.statistic), "p": float(r.pvalue)}
    return df, {"n": n_by_cohort(df), "stats": out}


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", nargs="*")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--date", help="snapshot date (default today)")
    args = parser.parse_args()

    table = shared_behaviour.load(rebuild=args.rebuild_cache)
    curves = day0_curves(table)
    ctx = {"table": table, "curves": curves, "examples": example_sessions(curves)}
    print("examples:", ctx["examples"])

    out = snapshot_dir(SPEC, args.date)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "_build").mkdir(exist_ok=True)
    meta_file = out / "_build" / "panel_meta.json"
    meta = json.load(open(meta_file)) if meta_file.exists() else {}
    meta["_shared"] = {"examples": ctx["examples"], "n_day0_sessions_with_curves": len(curves)}
    for panel in SPEC["panels"]:
        pid = panel["id"]
        if "data" not in panel or (args.panels and pid not in args.panels):
            continue
        df, info = globals()[f"compute_{pid}"](ctx)
        df.to_csv(out / "data" / panel["data"], index=False)
        meta[pid] = info
        print(pid, panel["data"], len(df), "rows")
    meta_file.write_text(json.dumps(meta, indent=1, default=str))  # serialise fully before writing


if __name__ == "__main__":
    main()
