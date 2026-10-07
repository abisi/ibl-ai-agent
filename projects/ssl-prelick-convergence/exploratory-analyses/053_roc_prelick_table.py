"""Collect the pre-lick ROC results (051) of all sessions into tables for the learning-stage statistics (047).

Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/
  roc_long_prelick.parquet  one row per unit x measure, measure = "<analysis_type>@<variant>"; columns as the 045
                            roc_long table (sel, abs_sel, sig, pos, neg) + p_value_to_show; units not tested
                            (min FR / min trials) have sig = NaN (-> not valid in 047)
  prelick_units.parquet     wide unit table (join keys + cohort, stage, quality, areas) with every selectivity,
                            significance, the three-class pairwise / one-vs-rest selectivities and preferred class
  provenance.json
Cohort: one per mouse, from the mouse reference sheet (see mouse_cohort; checked against the session labels,
report in cohort_check.csv).
Selectivity sign: as stored by 051 (wh_vs_aud_hit: positive = auditory > whisker; *_hit_vs_fa: positive = hit > FA);
three-class: sel = D3 >= 0 (unsigned), pos = significant.
"""
import glob
import json
import pathlib
import time

import numpy as np
import pandas as pd

import os
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
TAG = "" if os.environ.get("PRELICK_REF", "fa") == "fa" else "_" + os.environ["PRELICK_REF"]   # same switch as 051
OUT = RES / "ssl-prelick-convergence" / "across_days" / os.environ.get("PRELICK_REF", "fa")   # = 051 OUTROOT
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
MOUSE_REF = pathlib.Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")


def mouse_cohort(U, out):
    """Cohort rule (user 2026-10-03): the cohort is a property of the mouse, taken from the mouse reference sheet
    (joint_mouse_reference_weight.xlsx, column reward_group; R+ / R- only, other groups such as R+proba -> excluded).
    Mice flagged exclude or exclude_ephys in the sheet are excluded (cohort NaN). The session-level reward_group of the
    unit table (from NWB wh_reward) is checked against the sheet: every mismatch and every mouse missing from the sheet is
    written to cohort_check.csv and printed; the sheet wins. Fails if an analysed mouse is missing from the sheet."""
    X = pd.read_excel(MOUSE_REF).drop_duplicates("mouse_id").set_index("mouse_id")
    flag = lambda c: X[c].fillna(0).astype(str).isin(["1", "1.0", "True", "yes"]) if c in X else False
    ref = X.reward_group.where(X.reward_group.isin(["R+", "R-"]) & ~flag("exclude") & ~flag("exclude_ephys"))
    S = U.groupby(["mouse_id", "session_id", "stage"]).reward_group.agg(
        lambda v: ",".join(sorted(set(v.dropna())))).reset_index().rename(columns={"reward_group": "session_label"})
    missing = sorted(set(S.mouse_id) - set(X.index))
    if missing:
        raise ValueError(f"mice missing from the mouse reference sheet: {missing}")
    S["reference_group"] = S.mouse_id.map(X.reward_group)
    S["cohort"] = S.mouse_id.map(ref)
    S["excluded_by_sheet"] = S.mouse_id.map(flag("exclude") | flag("exclude_ephys"))
    S["status"] = np.select([S.session_label == "", S.session_label != S.reference_group],
                            ["session label missing", "MISMATCH (sheet used)"], "ok")
    S.to_csv(out / "cohort_check.csv", index=False)
    bad = S[S.status != "ok"]
    if len(bad):
        print("[053] cohort check (sheet wins):\n" + bad.to_string(index=False), flush=True)
    return U.mouse_id.map(ref)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(RES / "*" / "whisker_*" / "roc_analysis" / f"*_roc_prelick{TAG}_results.csv")))
    R = []
    for f in files:
        r = pd.read_csv(f, low_memory=False)
        if "variant" not in r:
            continue                                                       # pilot-format file
        R.append(r)
    R = pd.concat(R, ignore_index=True)
    R["cluster_id"] = R.cluster_id.astype(str); R["electrode_group"] = R.electrode_group.astype(str)
    U = pd.read_parquet(RES / "_roc_stage_analysis" / "units.parquet")
    U["cluster_id"] = U.cluster_id.astype(str); U["electrode_group"] = U.electrode_group.astype(str)
    U = U.drop_duplicates(KEYS)
    U["cohort"] = mouse_cohort(U, OUT)                              # mouse-level cohort from the reference sheet
    R = R.drop(columns=["mouse_id"]).merge(U[KEYS], on=["session_id", "electrode_group", "cluster_id"], how="inner")
    R = R[R.analysis_type != "three_class_prelick"]                 # three-class ROC removed (user 2026-10-03)
    R["measure"] = R.analysis_type + "@" + R.variant
    tested = R.tested.astype(str).eq("True")
    R["sel"] = np.where(tested, R.selectivity, np.nan)
    R["sig"] = np.where(tested, R.significant.astype(str).eq("True"), np.nan)
    R["pos"] = np.where(tested, (R.sig == 1) & (R.sel > 0), np.nan)
    R["neg"] = np.where(tested, (R.sig == 1) & (R.sel < 0), np.nan)
    R["abs_sel"] = R.sel.abs()
    L = R[KEYS + ["measure", "sel", "abs_sel", "sig", "pos", "neg", "p_value"]].rename(
        columns={"measure": "analysis_type", "p_value": "p_value_to_show"})
    L["sig"] = L.sig.astype(float); L["pos"] = L.pos.astype(float); L["neg"] = L.neg.astype(float)
    # 047 build_measures expects sig/pos/neg as booleans where defined; keep NaN for untested
    L["sig"] = L.sig.map({1.0: True, 0.0: False}); L["pos"] = L.pos.map({1.0: True, 0.0: False})
    L["neg"] = L.neg.map({1.0: True, 0.0: False})
    L.to_parquet(OUT / "roc_long_prelick.parquet", index=False)
    wide_cols = ["sel", "sig", "p_value", "fr_window"]
    W = R.pivot_table(index=KEYS, columns="measure", values=wide_cols, aggfunc="first")
    W.columns = [f"{a}:{b}" for a, b in W.columns]
    W = W.reset_index().merge(
        U[KEYS + ["cohort", "stage", "day", "quality_label", "area_group", "area_acronym_custom", "ccf_atlas_ap",
                  "ccf_atlas_ml", "ccf_atlas_dv", "reward_group"]], on=KEYS, how="left")
    W.to_parquet(OUT / "prelick_units.parquet", index=False)
    cnt = W.groupby(["cohort", "stage"]).agg(sessions=("session_id", "nunique"), mice=("mouse_id", "nunique"),
                                              units=("cluster_id", "size")).reset_index()
    prov = dict(script="053_roc_prelick_table.py", n_files=len(files), n_rows=len(R), measures=sorted(R.measure.unique()),
                counts=cnt.to_dict("records"), unit_table="_roc_stage_analysis/units.parquet (045)",
                created=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print(cnt.to_string())
    print(len(prov["measures"]), "measures")


if __name__ == "__main__":
    main()
