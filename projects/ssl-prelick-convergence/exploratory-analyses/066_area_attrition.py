"""Where are sessions / areas lost? Attrition per cohort x stage x area group, step by step (pre-lick analyses).

Steps (counts of sessions, and of units where relevant):
  s0  recorded: sessions of the group with >= 1 unit in the area group (any quality; 051 unit list)
  s1  trials: session has >= 3 WH, >= 3 AH and >= 3 FA trials after exclusions (051 trial log; needed by every
      pre-lick analysis; lambda needs >= 4 per class)
  s2  quality: >= 1 good or mua unit in the area
  s3  firing: >= 5 good / mua units with mean raw pre-lick rate >= 0.1 Hz and both ROCs tested
      (single-cell area statistics, 065)
  s4  population: lambda defined for the session x area (>= 5 units, >= 4 trials / class, AH-FA axis >= 0.01; 057)
Inclusion for a cohort in area figures: >= 3 sessions passing at both stages.
Output: combined_results_ks4/ssl-prelick-convergence/across_days/fa/attrition/attrition_area_group.csv (+ printed summary)
"""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
OUT = m51.OUTROOT / "attrition"
AF, WF = "auditory_hit_vs_fa_prelick@all", "whisker_hit_vs_fa_prelick@all"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    tl = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_config.json"
        if f.exists():
            log = json.loads(f.read_text())["trial_log"]
            tl.append(dict(session_id=r.session_id, n_WH=log["n_WH"], n_AH=log["n_AH"], n_FA=log["n_FA"]))
    TL = pd.DataFrame(tl).set_index("session_id")
    TL["trials_ok"] = TL[["n_WH", "n_AH", "n_FA"]].min(1) >= 3
    W["good_mua"] = W.quality_label.isin(["good", "mua"])
    W["tested"] = W[f"sig:{AF}"].notna() & W[f"sig:{WF}"].notna() & W.good_mua
    S = W.groupby(["session_id", "mouse_id", "cohort", "stage", "area_group"]).agg(
        n_units=("cluster_id", "size"), n_good_mua=("good_mua", "sum"), n_tested=("tested", "sum")).reset_index()
    S = S.join(TL, on="session_id")
    L = pd.read_csv(m51.OUTROOT / "lambda" / "lambda_sessions.csv")
    L = L[(L.level == "area_group") & (L.variant == "all")][["session_id", "region", "lam", "n_units", "d_AH_FA"]]
    S = S.merge(L.rename(columns={"region": "area_group", "n_units": "n_units_lambda"}), on=["session_id", "area_group"],
                how="left")
    S["s0"] = True
    S["s1"] = S.trials_ok.fillna(False)
    S["s2"] = S.s1 & (S.n_good_mua >= 1)
    S["s3"] = S.s2 & (S.n_tested >= 5)
    S["s4"] = S.s3 & S.lam.notna()
    S.to_csv(OUT / "attrition_session_area.csv", index=False)
    rows = []
    for (ag, c, st), d in S.groupby(["area_group", "cohort", "stage"]):
        row = dict(area_group=ag, cohort=c, stage=st, mice_s0=d.mouse_id.nunique())
        for k in ["s0", "s1", "s2", "s3", "s4"]:
            row[k] = int(d[k].sum())
        row["median_tested_units"] = float(d.loc[d.s2, "n_tested"].median()) if d.s2.any() else np.nan
        row["median_FA_trials"] = float(d.n_FA.median())
        rows.append(row)
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "attrition_area_group.csv", index=False)
    piv = A.pivot_table(index="area_group", columns=["cohort", "stage"], values=["s0", "s1", "s3", "s4"], aggfunc="first")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    for c in ["R+", "R-"]:
        print(f"===== {c}: sessions recorded (s0) -> trials ok (s1) -> >=5 tested units (s3) -> lambda valid (s4)")
        t = pd.DataFrame({f"{st} {k}": piv[(k, c, st)] for st in ["learning", "expert"] for k in ["s0", "s1", "s3", "s4"]
                          if (k, c, st) in piv})
        print(t.fillna(0).astype(int).to_string())
    print(A[(A.area_group == "Frontal areas")].to_string(index=False))
    # session-level trial losses
    TLg = TL.join(W.drop_duplicates("session_id").set_index("session_id")[["cohort", "stage"]])
    print(TLg.groupby(["cohort", "stage"]).agg(sessions=("trials_ok", "size"), trials_ok=("trials_ok", "sum"),
                                                median_FA=("n_FA", "median"), min_FA=("n_FA", "min")).to_string())


if __name__ == "__main__":
    main()
