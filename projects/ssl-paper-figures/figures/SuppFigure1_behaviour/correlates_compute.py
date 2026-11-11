"""Helpers of SuppFigure1_behaviour panels d, i, j, k (from the deleted SuppFigure2_learning_correlates, 2026-11-09);
not a figure on its own (no main). One compute_<key>(ctx) per panel. Panel a reuses Figure 1 compute_first_lick; panels
b-f read the latest outputs of the Figure 1 exploration scripts in Figures/_explore/ (run those first on haas).

Imported by SuppFigure1_behaviour/compute.py (run that one).


"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, load_spec, snapshot_dir, two_group_tests  # noqa: E402

SPEC = load_spec(FIG_DIR)
_spec = importlib.util.spec_from_file_location("fig1_compute", FIG_DIR.parent / "Figure1_behaviour" / "compute.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)
EXPLORE = FIGURES_ROOT / "_explore"
LEARNER = ["good", "moderate"]
MIN_MICE, BIN, N_HITS = 8, 5, 30


def latest(pattern):
    files = sorted(EXPLORE.glob(pattern))
    if not files:
        raise FileNotFoundError(f"{pattern} in {EXPLORE}: run the exploration script first")
    return files[-1]


def learners(ctx):
    return set(ctx["table"][ctx["table"].learning_category.isin(LEARNER)].mouse_id)


def compute_first_lick(ctx):
    return fig1.compute_first_lick(ctx)


def _binned(df, value, by):
    """Per-mouse median per bin of 5 hits (first 30); keep bins with >= MIN_MICE mice per cohort."""
    d = df[df.hit_number <= N_HITS].assign(hit_bin=lambda x: (x.hit_number - 1) // BIN)
    m = d.groupby(["mouse_id", "cohort", *by, "hit_bin"])[value].median().rename("value").reset_index().dropna()
    cnt = m.groupby([*by, "hit_bin", "cohort"]).mouse_id.nunique().unstack().fillna(0)
    ok = cnt[(cnt >= MIN_MICE).all(axis=1)].reset_index()[[*by, "hit_bin"]]
    return m.merge(ok, on=[*by, "hit_bin"])


def compute_rt_hits_other(ctx):
    f = latest("rt_hits_20*-*-??.csv")
    h = pd.read_csv(f)
    h = h[h.mouse_id.isin(learners(ctx))]
    parts = [_binned(h[h.trial_type.isin(["auditory_trial", "no_stim_trial"])], "rt_ms", ["trial_type"]).assign(latency="rt_ms"),
             _binned(h[h.trial_type == "whisker_trial"], "jaw_ms", ["trial_type"]).assign(latency="jaw_ms")]
    df = pd.concat(parts, ignore_index=True)
    return df, {"n": fig1.n_by_cohort(df), "source": f.name}


def compute_lick_course(ctx):
    f = latest("lick_proba_variants_20*-*-??.csv")
    d = pd.read_csv(f)
    d = d[d.lick_trials_only & (d.state == "all") & (d.half == "all") & d.trial_type.isin(["whisker_trial", "auditory_trial"])
          & d.mouse_id.isin(learners(ctx))]
    return d, {"n": fig1.n_by_cohort(d), "source": f.name}


def compute_rt_curves(ctx):
    f = latest("rt_learning_20*-*-??_corr.csv")
    d = pd.read_csv(f)
    d = d[(d.latency == "rt_ms") & d.mouse_id.isin(learners(ctx))][
        ["mouse_id", "cohort", "trial_type", "curve", "rho", "rho_vs_shift", "rho_partial_time", "n_hits"]]
    tests = {f"{tt} {cu} {c}": float(stats.wilcoxon(g.rho_vs_shift.dropna()).pvalue)
             for (tt, cu, c), g in d.groupby(["trial_type", "curve", "cohort"]) if g.rho_vs_shift.notna().sum() > 5}
    return d, {"n": fig1.n_by_cohort(d), "wilcoxon_vs_0": tests, "source": f.name}


def compute_bouts(ctx):
    f = latest("lick_learning_hits_20*-*-??.csv")
    h = pd.read_csv(f)
    h = h[h.mouse_id.isin(learners(ctx))]
    df = pd.concat([_binned(h, "n_licks", ["trial_type"]).assign(measure="n_licks"),
                    _binned(h[h.trial_type == "whisker_trial"], "expectation_index", ["trial_type"]).assign(
                        measure="expectation_index")], ignore_index=True)
    return df, {"n": fig1.n_by_cohort(df), "source": f.name}


def compute_posture_curve(ctx):
    f = latest("movement_corr_20*-*-??.csv")
    d = pd.read_csv(f)
    d = d[(d.curve == "whisker") & d.metric.str.contains("_pre_") & d.mouse_id.isin(learners(ctx))][
        ["mouse_id", "cohort", "metric", "spearman", "spearman_null", "spearman_vs_null", "pearson_vs_null", "n_trials"]]
    tests = {m: {**{f"{c}_vs_0": float(stats.wilcoxon(g[g.cohort == c].spearman_vs_null).pvalue) for c in ("R+", "R-")},
                 "rplus_vs_rminus": two_group_tests(g[g.cohort == "R+"].spearman_vs_null, g[g.cohort == "R-"].spearman_vs_null)}
             for m, g in d.groupby("metric")}
    return d, {"n": fig1.n_by_cohort(d), "stats": tests, "source": f.name}


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
