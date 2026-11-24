"""188 -- Axis quality / trial counts as covariates of the cohort effects (user 2026-11-24, Method-4 review: "What about number
of hits directly as covariate?"; "have the results saved somewhere explicit but do not add to figure yet").
Learners, whole brain, whisker (179 table; SSL_188_SFX selects a 179 output, default = current). OLS y ~ cohort [+ covariate],
covariates: n_hit, n_miss, log n_eff (n_eff = 1 / (1/n_hit + 1/n_miss); axis-noise variance scales with 1/n_eff) and the held-out
d'. Outcomes: angle change, projection change, rotation and gain terms (data - shifted) and the pre-task angle.
Also: means of the covariates per cohort and their point-biserial correlation with cohort.
Output: 188_axis_covariates{sfx}.csv. Run (haas, repo root): .venv/bin/python .../188_axis_covariates.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

EA = Path(__file__).resolve().parent
sys.path.insert(0, str(EA.parents[1] / "ssl-paper-figures"))
import shared_passive_cd as S  # noqa: E402

SFX = os.environ.get("SSL_188_SFX", "")
COVS = ("n_hit", "n_miss", "lneff", "dprime_loo")
YS = {"qth": "angle change (data - shifted)", "q": "projection change (data - shifted)", "rotation": "rotation term",
      "gain": "gain term", "th_pre": "pre-task angle"}


def main():
    d = pd.read_parquet(S.TAB / f"179_cd_clean{SFX}.parquet")
    d = d[d.skipped_reason.isna() & (d.area == "All units") & (d.stim == "W") & d.mouse_id.isin(S.learners())].copy()
    d["cohort"] = d.reward_group
    d["lneff"] = np.log(1 / (1 / d.n_hit + 1 / d.n_miss))
    d["q"] = d.d_along - d.d_along_null
    d["qth"] = (S.theta(d.cos_post) - S.theta(d.cos_pre)) - (S.theta(d.cos_post_null) - S.theta(d.cos_pre_null))
    d["rotation"] = d.rho_post * ((d.cos_post - d.cos_pre) - (d.cos_post_null - d.cos_pre_null))
    d["gain"] = (d.rho_post - d.rho_pre) * (d.cos_pre - d.cos_pre_null)
    d["th_pre"] = S.theta(d.cos_pre)
    rows = []
    for c in COVS:
        r = stats.pointbiserialr((d.cohort == "R-").astype(int), d[c])
        rows.append(dict(kind="covariate vs cohort", outcome=c, covariate="", mean_Rp=d[d.cohort == "R+"][c].mean(),
                         mean_Rm=d[d.cohort == "R-"][c].mean(), r=r[0], p_cohort=r[1]))
    for y, lab in YS.items():
        for cov in ("",) + COVS:
            f = smf.ols(f"{y} ~ C(cohort)" + (f" + {cov}" if cov else ""), data=d).fit()
            pc = [p for k, p in f.pvalues.items() if "cohort" in k][0]
            bc = [b for k, b in f.params.items() if "cohort" in k][0]
            rows.append(dict(kind="OLS", outcome=lab, covariate=cov or "none", n=int(f.nobs), cohort_coef_Rm=bc, p_cohort=pc,
                             p_covariate=f.pvalues[cov] if cov else np.nan))
    out = pd.DataFrame(rows)
    out.to_csv(EA / f"188_axis_covariates{SFX}.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
