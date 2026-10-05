"""137 -- Unit-set table (skills/ssl-valid-data "Unit sets", user definition 2026-10-05):
  unit_set_all  quality_label in {good, mua} (bombcell non-soma excluded);
  stable        good or mua passing exactly the three stability criteria: coverage_ratio >= 0.9, presence_ratio >= 0.5 and
                independence from probe drift (DREDge drift-shift joint test: fail = |r| > 0.5 AND p < 0.01); a unit without a
                drift result is not stable;
  good          quality_label == 'good' AND stable (the label table has no drift check, hence the intersection).
  stable_v1     the first version used by 140 on 2026-10-05 (kept for provenance): also required nSpikes >= 300 and spikes
                missing <= 20 %, and admitted units without a drift result if quality_label == 'good' (none did).
The >= 0.5 Hz in every analysed epoch criterion is analysis-specific and applied by each analysis.
Sources: ssl_ephys 1.0.0 units.parquet (bombcell metrics; cluster_id = probe index * 1e6 + NWB cluster id),
reports/ssl_analysis/derived/unit_area_labels.parquet (presence / coverage, area_group, quality_label), DREDge drift-shift
CSVs read directly for every NWB file (load_motion_dredge_shift_test_results), so units outside the v2 table's shared-area
filter keep their drift result. Join: (session_id, electrode_group, cluster_id mod 1e6) -- unit-level join keys
mouse_id, session_id, electrode_group, cluster_id are all saved.
Output: combined_results_ks4/<slug>/tables/137_stable_units.parquet + 137_stable_units_provenance.json
Run (haas, repo root): python projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses/137_stable_units.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(pathlib.Path.home() / "code" / "unit_spikes_analysis"))
sys.path.insert(0, str(pathlib.Path.home() / "code" / "ephys_utilities"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402
from ephys_utilities.helpers import load_helpers  # noqa: E402

SLUG = "ssl-whisker-hitmiss-timeresolved-decoding"
TAB = axel_bisi_root() / "combined_results_ks4" / SLUG / "tables"
NWB_DIR = axel_bisi_root() / "NWB_ks4"
UNITS = REPO / "reports/datasets/ssl_ephys/1.0.0/metadata/units.parquet"
LABELS = REPO / "reports/ssl_analysis/derived/unit_area_labels.parquet"
THR = dict(nSpikes=300, missing_max=20.0, coverage_min=0.9, presence_min=0.5, drift_r_max=0.5, drift_p_min=0.01)
N_WORKERS = 60


def main():
    t0 = time.time()
    U = pd.read_parquet(UNITS, columns=["session_id", "cluster_id", "electrode_group", "bc_label", "nSpikes",
                                        "percentageSpikesMissing_gaussian", "fractionRPVs_estimatedTauR", "firing_rate"])
    for c in ("nSpikes", "percentageSpikesMissing_gaussian", "fractionRPVs_estimatedTauR", "firing_rate"):
        U[c] = pd.to_numeric(U[c], errors="coerce")
    U["cluster_id"] = U.cluster_id.astype(np.int64)
    U["nwb_cluster_id"] = U.cluster_id % 1_000_000
    L = pd.read_parquet(LABELS, columns=["session_id", "cluster_id", "mouse_id", "area_group", "quality_label",
                                         "presence_ratio", "coverage_ratio"])
    L["cluster_id"] = L.cluster_id.astype(np.int64)
    U = U.merge(L, on=["session_id", "cluster_id"], how="left", validate="one_to_one")

    sessions = set(U.session_id)
    nwb_files = sorted(str(NWB_DIR / f) for f in os.listdir(NWB_DIR) if f.endswith(".nwb") and f[:-4] in sessions)
    D = load_helpers.load_motion_dredge_shift_test_results(nwb_files, day_to_analyze="all", max_workers=N_WORKERS)
    D = D[["session_id", "electrode_group", "cluster_id", "p_conservative", "r"]]
    # units tested later by 142 (missing from the original files): appended after the originals (originals win on duplicates)
    rr = sorted((axel_bisi_root() / "combined_results_ks4" / "_drift_rerun_20261005").glob("*/*/single_neuron_motion_shift_test/*_results.csv"))
    if rr:
        R = pd.concat([pd.read_csv(f, usecols=["session_id", "electrode_group", "cluster_id", "p_conservative", "r"]) for f in rr])
        D = pd.concat([D.assign(_src=0), R.assign(_src=1)], ignore_index=True)
        D = D.sort_values("_src").drop(columns="_src")
        D = D[pd.to_numeric(D.r, errors="coerce").notna() | ~D.duplicated(["session_id", "electrode_group", "cluster_id"], keep=False)]
    D = D.rename(columns={"p_conservative": "drift_shift_test_pval"})
    D["drift_abs_r"] = pd.to_numeric(D.r, errors="coerce").abs()
    D["drift_shift_test_pval"] = pd.to_numeric(D.drift_shift_test_pval, errors="coerce")
    D["nwb_cluster_id"] = D.cluster_id.astype(np.int64)
    D["electrode_group"] = D.electrode_group.astype(str)
    D = D.drop(columns=["cluster_id", "r"]).drop_duplicates(["session_id", "electrode_group", "nwb_cluster_id"])
    U["electrode_group"] = U.electrode_group.astype(str)
    U = U.merge(D, on=["session_id", "electrode_group", "nwb_cluster_id"], how="left", validate="one_to_one")

    has_drift = U.drift_abs_r.notna() & U.drift_shift_test_pval.notna()
    drift_fail = (U.drift_abs_r > THR["drift_r_max"]) & (U.drift_shift_test_pval < THR["drift_p_min"])
    somatic = (U.bc_label != "non-soma") & U.quality_label.isin(["good", "mua"])
    U["has_drift_test"] = has_drift
    U["drift_fail"] = has_drift & drift_fail
    U["unit_set_all"] = somatic
    U["stable"] = somatic & (U.coverage_ratio >= THR["coverage_min"]) & (U.presence_ratio >= THR["presence_min"]) & has_drift & ~drift_fail
    U["good"] = U.stable & (U.quality_label == "good")
    base_v1 = ((U.bc_label != "non-soma") & (U.nSpikes >= THR["nSpikes"]) & (U.percentageSpikesMissing_gaussian.fillna(0) <= THR["missing_max"])
               & (U.coverage_ratio >= THR["coverage_min"]) & (U.presence_ratio >= THR["presence_min"]))
    U["stable_v1"] = np.where(has_drift, base_v1 & ~drift_fail, (U.quality_label == "good"))
    out = U[["mouse_id", "session_id", "electrode_group", "cluster_id", "nwb_cluster_id", "area_group", "bc_label", "quality_label",
             "nSpikes", "percentageSpikesMissing_gaussian", "fractionRPVs_estimatedTauR", "firing_rate", "presence_ratio",
             "coverage_ratio", "drift_abs_r", "drift_shift_test_pval", "has_drift_test", "drift_fail", "unit_set_all", "stable", "good",
             "stable_v1"]]
    TAB.mkdir(parents=True, exist_ok=True)
    tmp = TAB / "137_stable_units.partial.parquet"
    out.to_parquet(tmp, index=False)
    os.replace(tmp, TAB / "137_stable_units.parquet")
    per = out.groupby("session_id").agg(n=("stable", "size"), all=("unit_set_all", "sum"), stable=("stable", "sum"), good=("good", "sum"),
                                        stable_v1=("stable_v1", "sum"), no_drift=("has_drift_test", lambda x: int((~x).sum())),
                                        drift_fail=("drift_fail", "sum"))
    prov = dict(script=pathlib.Path(__file__).name, created=time.strftime("%Y-%m-%d %H:%M"),
                thresholds=dict(coverage_min=THR["coverage_min"], presence_min=THR["presence_min"], drift_r_max=THR["drift_r_max"],
                                drift_p_min=THR["drift_p_min"]),
                n_units=int(len(out)), n_all=int(out.unit_set_all.sum()), n_stable=int(out.stable.sum()), n_good=int(out.good.sum()),
                n_stable_v1=int(out.stable_v1.sum()), n_stable_v1_not_stable=int((out.stable_v1 & ~out.stable).sum()),
                n_stable_not_v1=int((out.stable & ~out.stable_v1).sum()),
                n_without_drift=int((~out.has_drift_test).sum()), n_drift_fail=int(out.drift_fail.sum()),
                sessions_without_any_drift=sorted(per.index[per.no_drift == per.n].tolist()),
                per_session_median=per.median().round(1).to_dict(), sources=[str(UNITS), str(LABELS), "DREDge CSVs via load_helpers"],
                runtime_min=round((time.time() - t0) / 60, 1))
    (TAB / "137_stable_units_provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    print(json.dumps(prov, indent=1, default=str))


if __name__ == "__main__":
    main()
