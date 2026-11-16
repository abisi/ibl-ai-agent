"""05 -- Curation schemes and threshold sweeps on every sorting (Figures 4-5). Pipeline container or any env with
pandas.

Schemes (config `curation`, exact thresholds there): IBL, Bombcell (pipeline Bombcell unitType GOOD), Allen, and the
user's own criteria = ephys_utilities classify_units_quality (vendored copy, default thresholds and exclusions) on
Bombcell metrics + presence / coverage ratio + the joint DREDge drift shift test. A NaN metric fails a rule-based
scheme (IBL, Allen); classify_units_quality ignores NaN metrics by design.
Outputs tables/: 05_units_curated.csv (every unit, one boolean column per scheme), 05_retained.csv (counts per
recording x run x scheme x area, area = 'all' for the total), 05_sweeps.csv (single-criterion retention).
Usage: python 05_curation.py [--slice]
"""
import argparse
import operator
import sys

import numpy as np
import pandas as pd

import npw

sys.path.insert(0, str(npw.HERE / "vendor"))

OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge, "==": operator.eq}


def rule_scheme(units, rules):
    keep = pd.Series(True, index=units.index)
    for metric, (op, thr) in rules.items():
        v = units[metric]
        if metric == "amplitude_median":   # SI keeps the sign of the peak (negative troughs); IBL thresholds |amp|
            v = v.abs()
        keep &= OPS[op](v, thr) & v.notna()
    return keep


def own_scheme(units, cfg):
    import unit_metrics_utils as umu
    t = units.rename(columns={c: c[3:] for c in units.columns if c.startswith("bc_") and c != "bc_unitType"})
    t = t.drop(columns=["presence_ratio"]).rename(columns={"presence_ratio_eu": "presence_ratio"})
    bc = units.bc_unitType.astype(str).str.lower()
    t["bc_label"] = np.where(bc.str.contains("non-soma"), "non-soma", bc)
    lab = umu.classify_units_quality(t)["quality_label"]
    return lab == cfg["curation"]["axel"]["keep_label"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="store_true")
    a = ap.parse_args()
    cfg = npw.load_config()
    sfx = "_slice" if a.slice else ""
    frames = []
    for rec_id, run_id, sorter in npw.all_runs(cfg, a.slice):
        f = npw.post_dir(cfg, rec_id, run_id, a.slice) / "units.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    units = pd.concat(frames, ignore_index=True)
    for scheme, sc in cfg["curation"].items():
        if scheme == "axel":
            units[f"keep_{scheme}"] = pd.concat([own_scheme(g, cfg) for _, g in units.groupby(["rec_id", "run_id"])])
        else:
            units[f"keep_{scheme}"] = rule_scheme(units, sc["rules"])
    units["area"] = units.area.fillna("unassigned")
    units.to_csv(npw.out(cfg, "tables", f"05_units_curated{sfx}.csv"), index=False)
    rows = []
    for (rec_id, run_id, sorter), g in units.groupby(["rec_id", "run_id", "sorter"]):
        for scheme in ["none", *cfg["curation"]]:
            k = g if scheme == "none" else g[g[f"keep_{scheme}"]]
            rows.append(dict(rec_id=rec_id, run_id=run_id, sorter=sorter, scheme=scheme, area="all", n=len(k)))
            rows += [dict(rec_id=rec_id, run_id=run_id, sorter=sorter, scheme=scheme, area=ar, n=int(n))
                     for ar, n in k.area.value_counts().items()]
    ret = pd.DataFrame(rows)
    ret.to_csv(npw.out(cfg, "tables", f"05_retained{sfx}.csv"), index=False)
    sw = []
    for metric, s in cfg["sweeps"].items():
        thr = np.logspace(np.log10(s["min"]), np.log10(s["max"]), s["n"])
        for (rec_id, run_id, sorter), g in units.groupby(["rec_id", "run_id", "sorter"]):
            v = g[metric].to_numpy(float)
            sw += [dict(rec_id=rec_id, run_id=run_id, sorter=sorter, metric=metric, threshold=t,
                        n=int(np.sum(v < t))) for t in thr]
    pd.DataFrame(sw).to_csv(npw.out(cfg, "tables", f"05_sweeps{sfx}.csv"), index=False)
    tot = ret[(ret.area == "all")].pivot_table(index=["rec_id", "run_id"], columns="scheme", values="n")
    print(tot.to_string())


if __name__ == "__main__":
    main()
