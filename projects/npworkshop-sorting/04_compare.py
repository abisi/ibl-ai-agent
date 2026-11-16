"""04 -- Sorter agreement (Figure 2) and KS4 run-to-run variability (Figure 3), per recording. Pipeline container.

Matching: SpikeInterface compare_two_sorters / compare_multiple_sorters, delta_time 0.4 ms, match score 0.5, chance 0.1
(config `comparison`), all units of every sorting (no curation).
Outputs tables/: 04_pairwise_<rec>.csv (fraction of row-sorter units matched in column sorter),
04_agreement_units_<rec>.csv (one row per sorter unit: number of sorters that found it, SNR),
04_agreement_sets_<rec>.csv (unit count per sorter combination, UpSet), 04_ks4_variability_<rec>.csv (per run),
04_ks4_scores_<rec>.csv (best agreement score of every default-run unit in each run).
Usage: python 04_compare.py [--slice]
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import npw


def best_scores(cmp):
    """Best agreement score of each unit of sorting 1 in sorting 2, and whether it has a Hungarian match."""
    sc = cmp.agreement_scores
    best = sc.max(axis=1) if sc.shape[1] else pd.Series(0.0, index=sc.index)
    matched = cmp.hungarian_match_12 != -1
    return best, matched.reindex(best.index).fillna(False)


def main():
    import spikeinterface.comparison as sc
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", action="store_true")
    a = ap.parse_args()
    cfg = npw.load_config()
    c = cfg["comparison"]
    kw = dict(delta_time=c["delta_time_ms"], match_score=c["match_score"], chance_score=c["chance_score"])
    sfx = "_slice" if a.slice else ""
    for rec_id in list(cfg["recordings"])[:1] if a.slice else cfg["recordings"]:
        names = [s for s in cfg["sorters"] if (npw.sorting_dir(cfg, rec_id, s, a.slice) / "sorting").exists()]
        sortings = {s: npw.load_sorting(cfg, rec_id, s, a.slice) for s in names}
        units = {s: pd.read_csv(npw.post_dir(cfg, rec_id, s, a.slice) / "units.csv") for s in names
                 if (npw.post_dir(cfg, rec_id, s, a.slice) / "units.csv").exists()}
        # pairwise: fraction of units of sorter i with a match in sorter j
        pw = pd.DataFrame(np.nan, index=names, columns=names)
        for i, j in itertools.permutations(names, 2):
            cmp = sc.compare_two_sorters(sortings[i], sortings[j], sorting1_name=i, sorting2_name=j, **kw)
            pw.loc[i, j] = (cmp.hungarian_match_12 != -1).mean()
        pw.to_csv(npw.out(cfg, "tables", f"04_pairwise_{rec_id}{sfx}.csv"))
        # multi-sorter agreement: each agreement unit = one connected set of matched units across sorters
        mc = sc.compare_multiple_sorters([sortings[s] for s in names], name_list=names, **kw)
        agr = mc.get_agreement_sorting(minimum_agreement_count=1)
        rows, sets = [], []
        for au in agr.unit_ids:
            members = agr.get_unit_property(au, "unit_ids")
            k = int(agr.get_unit_property(au, "agreement_number"))
            sets.append(tuple(sorted(members)))
            for s, u in members.items():
                snr = np.nan
                if s in units:
                    m = units[s][units[s].unit_id.astype(str) == str(u)]
                    snr = float(m.snr.iloc[0]) if len(m) else np.nan
                rows.append(dict(rec_id=rec_id, sorter=s, unit_id=u, agreement_unit=au, n_sorters=k, snr=snr))
        pd.DataFrame(rows).to_csv(npw.out(cfg, "tables", f"04_agreement_units_{rec_id}{sfx}.csv"), index=False)
        st = pd.Series(sets).value_counts().rename_axis("sorters").reset_index(name="n_units")
        st["sorters"] = st.sorters.map("+".join)
        st["n_sorters"] = st.sorters.str.count(r"\+") + 1
        st.to_csv(npw.out(cfg, "tables", f"04_agreement_sets_{rec_id}{sfx}.csv"), index=False)
        print(rec_id, "agreement units by n sorters:", st.groupby("n_sorters").n_units.sum().to_dict())
        # KS4 variability: every variant against the default run
        if "kilosort4" not in sortings:
            continue
        ref = sortings["kilosort4"]
        var_rows, score_rows = [], []
        runs = [r["run_id"] for r in npw.run_table(cfg) if r["rec_id"] == rec_id and r["sorter"] == "kilosort4"]
        runs += [cfg["asrun"]["run_id"]] if not a.slice else []
        for run_id in runs:
            if run_id != cfg["asrun"]["run_id"] and not (npw.sorting_dir(cfg, rec_id, run_id, a.slice) /
                                                         "sorting").exists():
                continue
            s2 = npw.load_sorting(cfg, rec_id, run_id, a.slice)
            u2 = npw.post_dir(cfg, rec_id, run_id, a.slice) / "units.csv"
            n_good = int((pd.read_csv(u2).bc_unitType == "GOOD").sum()) if u2.exists() else np.nan
            if run_id == "kilosort4":
                best, matched = pd.Series(1.0, index=ref.unit_ids), pd.Series(True, index=ref.unit_ids)
            else:
                best, matched = best_scores(sc.compare_two_sorters(ref, s2, "kilosort4", run_id, **kw))
            var_rows.append(dict(rec_id=rec_id, run_id=run_id, n_units=s2.get_num_units(), n_good=n_good,
                                 frac_default_matched=float(matched.mean()), median_best_score=float(best.median())))
            score_rows += [dict(rec_id=rec_id, run_id=run_id, unit_id=u, best_score=float(v)) for u, v in best.items()]
        pd.DataFrame(var_rows).to_csv(npw.out(cfg, "tables", f"04_ks4_variability_{rec_id}{sfx}.csv"), index=False)
        pd.DataFrame(score_rows).to_csv(npw.out(cfg, "tables", f"04_ks4_scores_{rec_id}{sfx}.csv"), index=False)
        print(pd.DataFrame(var_rows).to_string(index=False))


if __name__ == "__main__":
    main()
