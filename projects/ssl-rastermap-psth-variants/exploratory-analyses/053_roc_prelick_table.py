"""Collect the pre-lick ROC results (051) of all sessions into tables for the learning-stage statistics (047).

Output: combined_results_ks4/_roc_prelick/
  roc_long_prelick.parquet  one row per unit x measure, measure = "<analysis_type>@<variant>"; columns as the 045
                            roc_long table (sel, abs_sel, sig, pos, neg) + p_value_to_show; units not tested
                            (min FR / min trials) have sig = NaN (-> not valid in 047)
  prelick_units.parquet     wide unit table (join keys + cohort, stage, quality, areas) with every selectivity,
                            significance, the three-class pairwise / one-vs-rest selectivities and preferred class
  provenance.json
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
OUT = RES / f"_roc_prelick{TAG}"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]


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
