"""SuppFigure1_learner_definition -- learners versus non-learners within each cohort (user 2026-11-18).

Learner = reference-sheet learning_category good or moderate, non-learner = bad (both cohorts; Figure 1 convention).
The session quantifications of Figure 1 and Supplementary Figure 1 are rerun on every mouse (the learner filter of those
functions is lifted) and each output row gets the mouse's learner label; tests compare learners with non-learners within
each cohort (Mann-Whitney U and Welch t; cluster-based permutation tests along time), only where both groups have enough
mice.

Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/SuppFigure1_learner_definition/compute.py
then build.py figures/SuppFigure1_learner_definition.
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
_spec = importlib.util.spec_from_file_location("s1_compute", FIG_DIR.parent / "SuppFigure1_behaviour" / "compute.py")
s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s1)
fig1 = s1.fig1  # the Figure 1 compute module loaded by s1 (shared PANELS, caches)

COHORTS = ["R+", "R-"]
TRIAL_TYPES = fig1.TRIAL_TYPES
MIN_TEST = 3  # mice per group for a two-group test
MIN_CLUSTER = 5  # mice per group at each x for the cluster test
CATEGORY_COLUMN = "learning_category_new"  # user 2026-11-19: sheet column with AB104, AB141, MH009, MH026 bad -> moderate


def with_sheet_category(table, column=CATEGORY_COLUMN):
    """Replace learning_category by `column` of the reference sheet (per mouse); the old label kept as
    learning_category_old."""
    ref = pd.read_excel(shared_behaviour.REFERENCE_SHEET)[["mouse_id", column]].drop_duplicates("mouse_id")
    t = table.drop(columns=[column], errors="ignore").merge(ref, on="mouse_id", how="left")
    changed = t[t.learning_category != t[column]].groupby("mouse_id").first()
    print("learner category from", column, "; changed:",
          {m: f"{r.learning_category} -> {r[column]}" for m, r in changed.iterrows()}, flush=True)
    return t.assign(learning_category_old=t.learning_category, learning_category=t[column]).drop(columns=[column])


# ---------------------------------------------------------------- learner labels and the all-mice table

def learner_map(table):
    """mouse_id -> True (learner: good / moderate) or False (non-learner: bad), from the whisker day-0 sessions."""
    d0 = fig1.day0_table(table)
    cat = d0.groupby("mouse_id").learning_category.first()
    return cat.map(fig1.LEARNER).dropna().astype(bool)


def all_mice_ctx(ctx):
    """ctx whose table marks every labelled mouse as a learner, so the Figure 1 / Supp. Fig. 1 functions keep all mice."""
    if "all" not in ctx:
        t = ctx["table"].copy()
        t["sheet_category"] = t.learning_category
        t.loc[t.learning_category.notna(), "learning_category"] = "good"
        ctx["all"] = {"table": t}
    return ctx["all"]


def label(ctx, df):
    if "learner_map" not in ctx:
        ctx["learner_map"] = learner_map(ctx["table"])
    lm = ctx["learner_map"]
    df = df[df.mouse_id.isin(lm.index)].copy()
    df["learner"] = df.mouse_id.map(lm).astype(bool)
    df["group"] = df.cohort + np.where(df.learner, " learner", " non-learner")
    return df.reset_index(drop=True)


def n_groups(df):
    return {c: {"learners": int(df[(df.cohort == c) & df.learner].mouse_id.nunique()),
                "non_learners": int(df[(df.cohort == c) & ~df.learner].mouse_id.nunique())} for c in COHORTS}


def within(df, value, by=()):
    """Learners vs non-learners within each cohort (and each `by` level): Mann-Whitney and Welch, both groups >= MIN_TEST."""
    out = {}
    groups = df.groupby(list(by)) if by else [((), df)]
    for keys, d in groups:
        keys = keys if isinstance(keys, tuple) else (keys,)
        for c in COHORTS:
            a = d[(d.cohort == c) & d.learner][value].dropna()
            b = d[(d.cohort == c) & ~d.learner][value].dropna()
            if len(a) >= MIN_TEST and len(b) >= MIN_TEST:
                r = two_group_tests(a, b)
                r["n_learners"], r["n_non_learners"] = r.pop("n_R+"), r.pop("n_R-")
                out[" ".join([c, *map(str, keys)])] = r
    return out


def cluster_within(df, x, value, by=()):
    """Cluster-based permutation test (Figure 1 cluster_test: Welch t, mouse labels permuted) of learners vs non-learners
    within each cohort and `by` level, at x values with >= MIN_CLUSTER mice per group. Adds cluster_sig, cluster_p,
    in_test; returns {key: clusters}."""
    df["cluster_sig"], df["cluster_p"], df["in_test"] = False, np.nan, False
    clusters = {}
    for keys, d in df.groupby(["cohort", *by]):
        keys = keys if isinstance(keys, tuple) else (keys,)
        d = d.dropna(subset=[value])
        n = d.groupby([x, "learner"]).mouse_id.nunique().unstack().reindex(columns=[True, False]).fillna(0).min(1)
        d = d[d[x].isin(n[n >= MIN_CLUSTER].index)]
        if not len(d):
            continue
        key = " ".join(map(str, keys))
        d2 = d.assign(cohort=np.where(d.learner, "R+", "R-"))  # cluster_test compares the labels "R+" and "R-"
        clusters[key] = fig1.cluster_test(d2, x, value)
        df.loc[d.index, "in_test"] = True
        df.loc[d.index, "cluster_sig"] = fig1.mark_clusters(d, x, clusters[key])
        df.loc[d.index, "cluster_p"] = fig1.cluster_p(d, x, clusters[key])
    return clusters


def clusters_text(clusters, fmt="{:.2f}"):
    out = {}
    for k, v in clusters.items():
        sig = [c for c in v if c["p"] < 0.05]
        out[k.replace(" ", "_")] = "; ".join(f"{fmt.format(c['start'])}-{fmt.format(c['end'])}, P = {c['p']:.3g}"
                                             for c in sig) or "none"
    return out


# ---------------------------------------------------------------- panels

def compute_learners(ctx):
    """Mice per sheet category (good, moderate, bad) per cohort; Fisher test of the learner fraction R+ vs R-."""
    return fig1.compute_learners(ctx)


def _example(ctx, key):
    fig1.PANELS[key] = {"params": {"example": PANELS[key]["params"]["example"]}}
    return fig1.compute_example(ctx, key)


def compute_ex_rplus_nonlearner(ctx):
    return _example(ctx, "ex_rplus_nonlearner")


def compute_ex_rminus_nonlearner(ctx):
    return _example(ctx, "ex_rminus_nonlearner")


def compute_day0_closeup(ctx):
    df, _ = s1.compute_day0_closeup(all_mice_ctx(ctx))
    df = label(ctx, df)
    return df, {"n": n_groups(df), "stats": within(df, "lick_probability", ["trial_type"])}


def compute_dprime_day0(ctx):
    """User 2026-11-19: d' on whisker day 0 per mouse, whisker vs no-stimulus and auditory vs no-stimulus trials:
    z(lick rate) - z(false-alarm rate) per session (log-linear correction (k + 0.5) / (n + 1), early licks dropped),
    mean over sessions per mouse."""
    t = fig1.day0_table(all_mice_ctx(ctx)["table"])
    t = t[(t.early_lick == 0) & t.trial_type.isin(TRIAL_TYPES)]
    g = t.groupby(["mouse_id", "cohort", "session_id", "trial_type"]).lick_flag.agg(["sum", "count"]).unstack("trial_type")
    z = lambda tt: pd.Series(stats_mod.norm.ppf(s1._rate(g[("sum", tt)], g[("count", tt)])), index=g.index)  # noqa: E731
    rows = []
    for tt, name in (("whisker_trial", "whisker"), ("auditory_trial", "auditory")):
        d = (z(tt) - z("no_stim_trial")).rename("dprime").dropna().groupby(["mouse_id", "cohort"]).mean().reset_index()
        rows.append(d.assign(comparison=name))
    df = label(ctx, pd.concat(rows, ignore_index=True))
    return df, {"n": n_groups(df), "stats": within(df, "dprime", ["comparison"])}


def compute_days(ctx):
    df, _ = s1.compute_days_expert(all_mice_ctx(ctx))
    df = label(ctx, df)
    n = {g: {int(d): int(x[x.day == d].mouse_id.nunique()) for d in sorted(df.day.unique())}
         for g, x in df.groupby("group")}
    return df, {"n": n_groups(df), "n_by_day": n, "stats": within(df, "value", ["measure", "trial_type", "day"])}


def _perf_session(ctx):
    if "perf" not in ctx:
        df, _ = fig1.compute_perf_session(all_mice_ctx(ctx))
        df = label(ctx, df.drop(columns=["group", "cluster_sig", "cluster_p"], errors="ignore"))
        clusters = cluster_within(df, "session_time_norm", "performance", ["alignment"])
        ctx["perf"] = (df, clusters)
    return ctx["perf"]


def _perf(ctx, align):
    df, clusters = _perf_session(ctx)
    d = df[df.alignment == align].reset_index(drop=True)
    cl = {k: v for k, v in clusters.items() if k.endswith(align)}
    text = clusters_text(cl)
    return d, {"n": n_groups(d), "clusters": cl, "session_keep": fig1.SESSION_KEEP,
               "sig_clusters_text": {c: text.get(f"{c}_{align}", "not tested") for c in COHORTS}}


def compute_perf_trial(ctx):
    return _perf(ctx, "first_whisker_trial")


def compute_perf_hit(ctx):
    return _perf(ctx, "first_whisker_hit")


def _dprime(ctx, axis):
    df, _ = s1._dprime_session(all_mice_ctx(ctx))
    df = label(ctx, df[df.axis == axis].drop(columns=["cluster_sig", "cluster_p", "in_test"]))
    clusters = cluster_within(df, "time", "dprime", ["alignment"])
    text = clusters_text(clusters, "{:.2f}" if axis == "norm" else "{:g}")
    text = {f"{c}_{al}": text.get(f"{c}_{al}", "not tested") for c in COHORTS
            for al in ("first_whisker_trial", "first_whisker_hit")}
    return df, {"n": n_groups(df), "clusters": clusters, "sig_clusters_text": text, "clip": list(s1.DPRIME_CLIP),
                "session_keep": fig1.SESSION_KEEP}


def compute_dprime_norm(ctx):
    return _dprime(ctx, "norm")


def compute_dprime_abs(ctx):
    return _dprime(ctx, "abs_min")


def compute_cumulative_fa(ctx):
    df, _ = s1.compute_cumulative_fa(all_mice_ctx(ctx))
    df = label(ctx, df)
    n = PANELS["cumulative_fa"]["params"]["n_trials"]
    last = df[df.trial_after_anchor == n]
    return df, {"n": {f"day{d}": n_groups(df[df.day == d]) for d in (0, -1)}, "n_trials": n,
                "stats": within(last, "cumulative_fa", ["day"])}


def compute_pretraining(ctx):
    df, _ = fig1.compute_pretraining(ctx)
    df = label(ctx, df)
    return df, {"n": n_groups(df), "stats": within(df, "n_pretraining_trials")}


def compute_first_lick(ctx):
    df, _ = fig1.compute_first_lick(all_mice_ctx(ctx))
    df = label(ctx, df)
    col = "whisker_trials_before_first_lick"
    return df, {"n": n_groups(df), "stats": within(df, col),
                "no_whisker_lick": {g: int(x[col].isna().sum()) for g, x in df.groupby("group")}}


def compute_rt(ctx):
    df, _ = fig1.compute_rt(all_mice_ctx(ctx))
    df = label(ctx, df)
    df = df[df.trial_type.isin(["whisker_trial", "auditory_trial"])].reset_index(drop=True)
    return df, {"n": n_groups(df), "stats": within(df, "median_reaction_time_s", ["trial_type"])}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", nargs="*")
    ap.add_argument("--date")
    a = ap.parse_args()
    ctx = {"table": with_sheet_category(shared_behaviour.load())}
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
