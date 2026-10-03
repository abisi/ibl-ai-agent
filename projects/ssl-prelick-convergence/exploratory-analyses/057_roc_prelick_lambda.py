"""Do whisker hits become more like auditory hits before the lick (more in R+), in some areas?

Measure (per session x area, pre-lick 100 ms window, corrected first lick, baseline-corrected rates from 051):
  lambda = cross-validated projection of the whisker-hit response onto the false-alarm -> auditory-hit axis:
      lambda = mean_s[(WH_a - FA_a) . (AH_b - FA_b)] / mean_s[(AH_a - FA_a) . (AH_b - FA_b)]
  a / b = random halves of each class's trials (s = 50 splits); units z-scored over the used trials; units tested if
  mean raw pre-lick rate >= 0.1 Hz, quality good or mua. lambda = 0: WH like FA (unrewarded lick); lambda = 1: WH like AH (rewarded lick).
  Kept only if the AH-FA axis is reliable (cv squared distance d_AF > 0) and >= MIN_UNITS units, >= 4 trials per class.
  Also stored: cv distances d_WH_AH, d_AH_FA, d_WH_FA (per unit).
  Robustness: lambda is a ratio, so it is used only where the AH-FA axis is reliable (d_AH_FA >= AXIS_MIN = 0.02,
  ~85 % of cases) and clipped to [-1, 2]; ratio-free companion dd = d_WH_FA - d_WH_AH (> 0: WH closer to AH than FA).
Levels: whole brain (all units of a session), area_group, area_acronym_custom. Trial variants: all, rt_matched.
Tests (unit = session): stage within cohort (MWU + Welch), cohort at expert (MWU + Welch), learning x cohort
  interaction [E-L](R+) - [E-L](R-) with cohort labels permuted across mice (10000). Across areas: max-statistic
  permutation (same cohort permutations for all areas; per-area interaction standardised by its null SD; family-wise
  p = P(max_area |z_null| >= |z_obs|)) -> "in some areas" claims control the family-wise error.
Output: combined_results_ks4/_roc_prelick/lambda/
"""
import importlib
import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m51 = importlib.import_module("051_roc_prelick")
OUT = m51.OUTROOT / "lambda"
N_SPLIT = 50
MIN_UNITS = 5                     # user 2026-10-02 (was 10)
UNIT_SET = ("good", "mua")      # quality_label kept (non-soma and unlabelled units excluded)
MIN_SESS = 3
N_PERM = 10000
AXIS_MIN = 0.01            # min cv squared AH-FA distance per unit (user 2026-10-02: smaller than the first 0.02)
LAM_CLIP = (-1.0, 2.0)
METRICS = ["lam", "dd"]
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]


def lam(Z, lab, rng):
    idx = {c: np.where(lab == c)[0] for c in m51.CLASSES}
    if min(len(v) for v in idx.values()) < 4:
        return None
    num, den, dwa, dwf = [], [], [], []
    for _ in range(N_SPLIT):
        h = {}
        for c, v in idx.items():
            p = rng.permutation(v); k = len(p) // 2
            h[c] = (Z[:, p[:k]].mean(1), Z[:, p[k:]].mean(1))
        wa, wb = h["WH"][0] - h["FA"][0], h["WH"][1] - h["FA"][1]
        aa, ab = h["AH"][0] - h["FA"][0], h["AH"][1] - h["FA"][1]
        num.append((wa @ ab + wb @ aa) / 2); den.append(aa @ ab)
        dwa.append((h["WH"][0] - h["AH"][0]) @ (h["WH"][1] - h["AH"][1])); dwf.append(wa @ wb)
    n = len(Z)
    d_af = np.mean(den) / n
    return dict(lam=np.mean(num) / np.mean(den) if d_af > 0 else np.nan, d_AH_FA=d_af, d_WH_AH=np.mean(dwa) / n,
                d_WH_FA=np.mean(dwf) / n, n_units=n)


def compute():
    W = pd.read_parquet(m51.OUTROOT / "prelick_units.parquet")
    W = W[W.cohort.isin(["R+", "R-"])]
    W["electrode_group"] = W.electrode_group.astype(str); W["cluster_id"] = W.cluster_id.astype(str)
    st26 = importlib.import_module("026_roc_rates_all_sessions")
    ss = st26.all_sessions(); ss = ss[ss.session_id.isin(W.session_id.unique())]
    rng = np.random.default_rng(0)
    rows = []
    for r in ss.itertuples():
        f = r.file.parent / f"{r.mouse}_roc_prelick{m51.TAG}_trials.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        K = pd.DataFrame(dict(electrode_group=z["electrode_group"].astype(str), cluster_id=z["cluster_id"].astype(str)))
        info = W[W.session_id == r.session_id][["electrode_group", "cluster_id", "area_group", "area_acronym_custom",
                                                 "cohort", "stage", "mouse_id", "quality_label"]]
        K = K.merge(info, on=["electrode_group", "cluster_id"], how="left")
        if K.cohort.isna().all():
            continue
        meta = dict(session_id=r.session_id, mouse_id=K.mouse_id.dropna().iloc[0], cohort=K.cohort.dropna().iloc[0],
                    stage=K.stage.dropna().iloc[0])
        X, raw, lab, rt = z["rates"].astype(float), z["raw"].astype(float), z["cls"], z["rt"]
        for v in ["all", "rt_matched"]:
            idx = m51.trial_set(lab, rt, m51.CLASSES, v, np.random.default_rng(0))
            if len(idx) < 12:
                continue
            ok = (raw[:, idx].mean(1) >= m51.MIN_FR) & K.quality_label.isin(UNIT_SET).to_numpy()
            Z = X[:, idx]; sd = Z.std(1); ok &= sd > 0
            Z = (Z - Z.mean(1, keepdims=True)) / np.where(sd > 0, sd, 1)[:, None]
            l = lab[idx]
            for level in ["all", "area_group", "area_acronym_custom"]:
                regs = {"all": np.ones(len(K), bool)} if level == "all" else \
                    {a: (K[level] == a).to_numpy() for a in K[level].dropna().unique()}
                for reg, m in regs.items():
                    u = ok & m
                    if u.sum() < MIN_UNITS:
                        continue
                    res = lam(Z[u], l, rng)
                    if res:
                        rows.append(dict(meta, variant=v, level=level, region=reg, **res))
    return pd.DataFrame(rows)


def group_tests(d, metric="lam"):
    out = {}
    g = {k: d[(d.cohort == k[0]) & (d.stage == k[1])][metric].dropna().to_numpy() for k in GROUPS}
    for name, (A, B) in {"stage:R+": (GROUPS[0], GROUPS[1]), "stage:R-": (GROUPS[2], GROUPS[3]),
                         "cohort:learning": (GROUPS[2], GROUPS[0]), "cohort:expert": (GROUPS[3], GROUPS[1])}.items():
        x, y = g[A], g[B]
        if len(x) >= MIN_SESS and len(y) >= MIN_SESS:
            out[name] = dict(diff=y.mean() - x.mean(), p_mwu=stats.mannwhitneyu(x, y).pvalue,
                             p_welch=stats.ttest_ind(x, y, equal_var=False).pvalue, n_A=len(x), n_B=len(y))
    out["means"] = {f"{c}_{s}": float(g[(c, s)].mean()) if len(g[(c, s)]) else np.nan for c, s in GROUPS}
    out["n"] = {f"{c}_{s}": len(g[(c, s)]) for c, s in GROUPS}
    return out


def interaction_perm(L, regions, perm_coh, mice, metric="lam"):
    """observed interaction and its null (P x R) per region with the same cohort permutations"""
    obs, null = {}, {}
    for reg in regions:
        d = L[L.region == reg].dropna(subset=[metric])
        n = {k: ((d.cohort == k[0]) & (d.stage == k[1])).sum() for k in GROUPS}
        if min(n.values()) < MIN_SESS:
            continue
        midx = pd.Index(mice).get_indexer(d.mouse_id); st = d.stage.to_numpy(); y = d[metric].to_numpy()

        def stat(coh):
            mm = {k: y[(coh == k[0]) & (st == k[1])] for k in GROUPS}
            if min(len(v) for v in mm.values()) == 0:
                return np.nan
            return (mm[GROUPS[1]].mean() - mm[GROUPS[0]].mean()) - (mm[GROUPS[3]].mean() - mm[GROUPS[2]].mean())
        obs[reg] = stat(d.cohort.to_numpy())
        null[reg] = np.array([stat(pc[midx]) for pc in perm_coh])
    return obs, null


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "lambda_sessions_raw.csv"
    L = pd.read_csv(f) if f.exists() else compute()
    L.to_csv(f, index=False)
    L["axis_ok"] = (L.d_AH_FA > 0) & (L.d_AH_FA >= AXIS_MIN)
    L["lam"] = np.where(L.axis_ok, L.lam.clip(*LAM_CLIP), np.nan)
    L["dd"] = L.d_WH_FA - L.d_WH_AH                   # > 0: WH closer to AH than to FA (cv squared distances / unit)
    L.to_csv(OUT / "lambda_sessions.csv", index=False)
    rng = np.random.default_rng(1)
    mice = L.mouse_id.unique(); mc = L.groupby("mouse_id").cohort.first().reindex(mice).to_numpy()
    perm_coh = [rng.permutation(mc) for _ in range(N_PERM)]
    rows = []
    for metric, v in [(m_, v_) for m_ in METRICS for v_ in ["all", "rt_matched"]]:
        for level in ["all", "area_group", "area_acronym_custom"]:
            D = L[(L.variant == v) & (L.level == level)]
            obs, null = interaction_perm(D, D.region.unique(), perm_coh, mice, metric)
            zs = {r: obs[r] / np.nanstd(null[r]) for r in obs}
            Z0 = np.column_stack([null[r] / np.nanstd(null[r]) for r in obs]) if obs else np.zeros((N_PERM, 0))
            maxz = np.nanmax(np.abs(Z0), 1) if Z0.size else np.array([np.nan])
            for reg in D.region.unique():
                t = group_tests(D[D.region == reg], metric)
                row = dict(metric=metric, variant=v, level=level, region=reg, **{f"mean_{k}": x for k, x in t["means"].items()},
                           **{f"n_{k}": x for k, x in t["n"].items()})
                for c in ["stage:R+", "stage:R-", "cohort:expert", "cohort:learning"]:
                    if c in t:
                        for k in ["diff", "p_mwu", "p_welch"]:
                            row[f"{c}_{k}"] = t[c][k]
                if reg in obs:
                    nl = null[reg][np.isfinite(null[reg])]
                    row["interaction"] = obs[reg]
                    row["interaction_p"] = (1 + np.sum(np.abs(nl) >= abs(obs[reg]))) / (1 + len(nl))
                    row["interaction_z"] = zs[reg]
                    row["interaction_p_maxT"] = (1 + np.sum(maxz >= abs(zs[reg]))) / (1 + len(maxz))
                    row["n_regions_family"] = len(obs)
                rows.append(row)
    T = pd.DataFrame(rows); T.to_csv(OUT / "lambda_tests.csv", index=False)
    json.dump(dict(script="057_roc_prelick_lambda.py", n_split=N_SPLIT, min_units=MIN_UNITS, min_sessions=MIN_SESS,
                   n_perm=N_PERM, min_fr_hz=m51.MIN_FR, unit_of_analysis="session", axis_min=AXIS_MIN,
                   lambda_clip=LAM_CLIP, dd="d_WH_FA - d_WH_AH (cv squared distances per unit)",
                   family_wise="max-|z| permutation across regions within level x variant"),
              open(OUT / "provenance.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    cols = ["metric", "variant", "level", "region", "mean_R+_learning", "mean_R+_expert", "mean_R-_learning", "mean_R-_expert",
            "stage:R+_diff", "stage:R+_p_mwu", "stage:R-_diff", "stage:R-_p_mwu", "interaction", "interaction_p",
            "interaction_p_maxT", "n_R+_expert", "n_R-_expert"]
    print(T[T.level != "area_acronym_custom"][cols].round(3).to_string())
    print(T[(T.level == "area_acronym_custom") & (T.interaction_p < 0.05)][cols].round(3).to_string())


if __name__ == "__main__":
    main()
