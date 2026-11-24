"""187 -- Selection variants of 179 (user 2026-11-24, Method-1 review): does session-end trimming matter, does it change the
hit - miss axis reliability, and does the passive-epoch rule (ephys_utilities middle-trial split vs passive blocks) matter?
Variants (179 outputs): default (no trim, passive 'mid'), _trim-a1 (rule A1), _trim-end10 (last 10 % of the active epoch),
_passive-blocks; plus the previous selection (_sel132_20261124 backup: 132.session_trials, A1, >= 20 units, >= 10 trials).
Per variant, learners, whole brain, whisker: sessions per cohort, hits / misses, held-out d' (axis reliability), and R+ vs R-
on the angle change (data - shifted), the rotation and gain terms and the projection change. Paired across variants
(same sessions): d' difference vs the default (Wilcoxon). Uncorrected.
Output: 187_selection_variants.csv (+ printed). Run (haas, repo root): .venv/bin/python .../187_selection_variants.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA.parents[1] / "ssl-paper-figures"))
import shared_passive_cd as S  # noqa: E402

VAR = {"default (none, mid)": "", "trim A1": "_trim-a1", "trim end 10%": "_trim-end10", "passive blocks": "_passive-blocks",
       "previous (132, A1)": "_sel132_20261124"}


def load(sfx):
    d = pd.read_parquet(S.TAB / f"179_cd_clean{sfx}.parquet")
    d = d[d.skipped_reason.isna() & (d.area == "All units") & (d.stim == "W") & d.mouse_id.isin(S.learners())].copy()
    d["cohort"] = d.reward_group
    d["qth"] = (S.theta(d.cos_post) - S.theta(d.cos_pre)) - (S.theta(d.cos_post_null) - S.theta(d.cos_pre_null))
    d["q"] = d.d_along - d.d_along_null
    d["rotation"] = d.rho_post * ((d.cos_post - d.cos_pre) - (d.cos_post_null - d.cos_pre_null))
    d["gain"] = (d.rho_post - d.rho_pre) * (d.cos_pre - d.cos_pre_null)
    return d.set_index("session_id")


def main():
    D = {k: load(v) for k, v in VAR.items() if (S.TAB / f"179_cd_clean{v}.parquet").exists()}
    base = D["default (none, mid)"]
    rows = []
    for k, d in D.items():
        r = dict(variant=k)
        for c in S.COH:
            g = d[d.cohort == c]
            r.update({f"n_{c}": len(g), f"hits_{c}": g.n_hit.mean(), f"misses_{c}": g.n_miss.mean(), f"dprime_{c}": g.dprime_loo.mean(),
                      f"angle_change_{c}": g.qth.mean()})
        for m in ("qth", "rotation", "gain", "q"):
            t = S.two_group_tests(d[d.cohort == "R+"][m], d[d.cohort == "R-"][m])
            r[f"p_{m}"] = max(t["mannwhitney_p"], t["welch_p"])
        common = d.index.intersection(base.index)
        if k != "default (none, mid)" and len(common) > 5:
            x, y = d.loc[common, "dprime_loo"], base.loc[common, "dprime_loo"]
            r["dprime_minus_default"] = float((x - y).mean())
            r["p_dprime_vs_default"] = float(stats.wilcoxon(x, y).pvalue) if (x != y).any() else 1.0
            r["n_common"] = len(common)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(EA / "187_selection_variants.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).T.to_string())
    only_new = base.index.difference(D.get("previous (132, A1)", base).index)
    only_old = D.get("previous (132, A1)", base).index.difference(base.index)
    print("\nsessions only in the default:", list(only_new), "\nsessions only in the previous selection:", list(only_old))


if __name__ == "__main__":
    main()
