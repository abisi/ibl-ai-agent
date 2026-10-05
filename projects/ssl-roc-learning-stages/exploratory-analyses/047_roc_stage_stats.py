"""Statistics of the multi-ROC learning-stage analysis (good units; inputs from 045).

Measures
  ROC types (27): frac_sig, frac_pos, frac_neg (direction after sign unification, see 045), mean |sel| (all tested
  units) and mean |sel| of significant units.
  Combined tuning categories (binary per unit; denominator = units for which the needed analyses exist):
    resp_whisker / resp_auditory (whisker_active / auditory_active sig), bimodal (both), whisker_only, auditory_only,
    pref_whisker / pref_auditory (wh_vs_aud_active sig, whisker > auditory / auditory > whisker),
    whisker_decision / whisker_gated / auditory_decision / auditory_gated (roc_utils_new labels), lick_resp
    (spontaneous_licks), motor (spontaneous_licks_vs_cr), choice_in_whisker_resp / decision_in_whisker_resp
    (conditional: among whisker-responsive units), choice_in_auditory_resp, mixed_n (number of significant core types).
Comparisons (statistic; permutation unit)
  stage:<cohort>   expert - learning within a cohort; stage labels permuted across that cohort's sessions
  cohort:<stage>   R+ - R- within a stage; cohort labels permuted across mice (a mouse keeps all its sessions)
  interaction      [expert - learning](R+) - [expert - learning](R-); cohort labels permuted across mice
Metrics are pooled over units of a group (sum of counts / sum of tested units). 95% CI: bootstrap of sessions within
each group. Region inclusion per comparison & measure: >= 10 tested units in total and >= 3 sessions in EACH group
(interaction: included in both stage comparisons). Levels: all (pooled), area_group, area_acronym_custom.
No multiple-comparison correction (as requested).
Also: two-way weighted ANOVA (frac ~ group x region, session x region rows, weights = n units) with a permutation p
for the group main effect; focality across regions (Gini; entropy focality 1 - H(p)/ln R) with bootstrap CI and
permutation p of the change; selectivity-distribution shift (Wasserstein distance of signed selectivity, permutation
of session labels); matched mice (both stages): paired sign-flip test of the pooled per-mouse fraction.
Output: combined_results_ks4/ssl-roc-learning-stages/stats/
"""
import argparse
import itertools
import json
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
BASE = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-roc-learning-stages")
OUT = BASE / "stats"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
MIN_UNITS, MIN_SESS = 10, 3
CORE = ["whisker_active", "auditory_active", "wh_vs_aud_active", "whisker_choice", "auditory_choice",
        "whisker_sensory", "spontaneous_licks", "spontaneous_licks_vs_cr", "whisker_hit_vs_spontaneous"]
LEVELS = ["all", "area_group", "area_acronym_custom"]


# ------------------------------------------------------------------ measures
def build_measures(U, L):
    """returns dict measure -> DataFrame(unit index aligned to U: valid, flag, pos, neg, abs_sel, sel)"""
    M = {}
    W = {c: L.pivot_table(index="uid", columns="analysis_type", values=c, aggfunc="first") for c in ["sig", "sel", "abs_sel", "pos", "neg"]}
    W = {k: v.reindex(U.index) for k, v in W.items()}
    for t in W["sig"].columns:
        v = W["sig"][t].notna()
        M[t] = pd.DataFrame(dict(valid=v, flag=W["sig"][t].fillna(False).astype(bool), pos=W["pos"][t].fillna(False).astype(bool),
                                 neg=W["neg"][t].fillna(False).astype(bool), abs_sel=W["abs_sel"][t], sel=W["sel"][t]))
    s, sel = W["sig"], W["sel"]

    def has(*ts):
        return np.all([s[t].notna() for t in ts if t in s], axis=0) if all(t in s for t in ts) else np.zeros(len(U), bool)

    def sg(t):
        return s[t].fillna(False).astype(bool) if t in s else pd.Series(False, index=U.index)
    if "whisker_active" not in s or "wh_vs_aud_active" not in s:          # other ROC tables (e.g. pre-lick): no categories
        return M
    wa, aa = sg("whisker_active"), sg("auditory_active")
    cats = {
        "cat:resp_whisker": (has("whisker_active"), wa), "cat:resp_auditory": (has("auditory_active"), aa),
        "cat:bimodal": (has("whisker_active", "auditory_active"), wa & aa),
        "cat:whisker_only": (has("whisker_active", "auditory_active"), wa & ~aa),
        "cat:auditory_only": (has("whisker_active", "auditory_active"), aa & ~wa),
        "cat:pref_whisker": (has("wh_vs_aud_active"), sg("wh_vs_aud_active") & (sel["wh_vs_aud_active"] < 0)),
        "cat:pref_auditory": (has("wh_vs_aud_active"), sg("wh_vs_aud_active") & (sel["wh_vs_aud_active"] > 0)),
        "cat:whisker_decision": (has("whisker_choice", "whisker_hit_vs_spontaneous"), U.whisker_decision.fillna(False).astype(bool)),
        "cat:whisker_gated": (has("whisker_choice", "whisker_hit_vs_spontaneous", "whisker_sensory", "spontaneous_licks_vs_cr"),
                              U.whisker_gated_decision.fillna(False).astype(bool)),
        "cat:auditory_decision": (has("auditory_choice", "auditory_hit_vs_spontaneous"), U.auditory_decision.fillna(False).astype(bool)),
        "cat:auditory_gated": (has("auditory_choice", "auditory_hit_vs_spontaneous", "auditory_sensory", "spontaneous_licks_vs_cr"),
                               U.auditory_gated_decision.fillna(False).astype(bool)),
        "cat:lick_resp": (has("spontaneous_licks"), sg("spontaneous_licks")),
        "cat:motor": (has("spontaneous_licks_vs_cr"), sg("spontaneous_licks_vs_cr")),
        "cat:choice_in_whisker_resp": (has("whisker_choice") & wa.to_numpy(), sg("whisker_choice")),
        "cat:decision_in_whisker_resp": (has("whisker_choice", "whisker_hit_vs_spontaneous") & wa.to_numpy(),
                                         U.whisker_decision.fillna(False).astype(bool)),
        "cat:choice_in_auditory_resp": (has("auditory_choice") & aa.to_numpy(), sg("auditory_choice")),
    }
    for k, (v, f) in cats.items():
        M[k] = pd.DataFrame(dict(valid=np.asarray(v, bool), flag=np.asarray(f, bool), pos=np.asarray(f, bool),
                                 neg=np.zeros(len(U), bool), abs_sel=np.nan, sel=np.nan), index=U.index)
    core = [c for c in CORE if c in s]
    mixed = sum(sg(c).astype(int) for c in core)
    M["cat:mixed_n"] = pd.DataFrame(dict(valid=has(*core), flag=mixed >= 3, pos=mixed >= 3, neg=np.zeros(len(U), bool),
                                         abs_sel=mixed.astype(float), sel=mixed.astype(float)), index=U.index)
    return M


# ------------------------------------------------------------------ session x region sums
class Sums:
    """per measure & level: arrays (S, R) of n, k, kp, kn, sa (sum abs_sel), ns (n with sel), ssig (sum abs_sel of sig)"""

    def __init__(self, U, meas, level, sess):
        self.S = len(sess)
        if level == "all":
            reg = np.zeros(len(U), int); self.regions = np.array(["all"])
        else:
            lab = U[level].astype(str)
            ok = ~lab.isin(["nan", "None", "Other", "unassigned", "root", ""])
            self.regions = np.array(sorted(lab[ok].unique()))
            rmap = {r: i for i, r in enumerate(self.regions)}
            reg = lab.map(rmap).fillna(-1).astype(int).to_numpy()
        si = U.session_id.map({s_: i for i, s_ in enumerate(sess)}).to_numpy()
        R = len(self.regions)
        ok_u = (reg >= 0) & meas.valid.to_numpy()
        a = {}
        sel = np.nan_to_num(meas.sel.to_numpy(float)); absel = np.nan_to_num(meas.abs_sel.to_numpy(float))
        for name, v in [("n", np.ones(len(U))), ("k", meas.flag.to_numpy(float)), ("kp", meas.pos.to_numpy(float)),
                        ("kn", meas.neg.to_numpy(float)), ("sa", absel),
                        ("ns", np.isfinite(meas.abs_sel.to_numpy(float)).astype(float)),
                        ("ssig", absel * meas.flag.to_numpy(float)),
                        ("sp", np.clip(sel, 0, None)), ("sn", np.clip(-sel, 0, None)),        # |sel| = sel+ + sel-
                        ("ssp", absel * meas.pos.to_numpy(float)), ("ssn", absel * meas.neg.to_numpy(float))]:
            arr = np.zeros((self.S, R)); np.add.at(arr, (si[ok_u], reg[ok_u]), v[ok_u]); a[name] = arr
        self.a = a

    def group(self, w):
        """w: (..., S) session weights -> dict of (..., R) sums"""
        return {k: w @ v for k, v in self.a.items()}


def metrics(g):
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(frac_sig=g["k"] / g["n"], frac_pos=g["kp"] / g["n"], frac_neg=g["kn"] / g["n"],
                    mean_abs_sel=g["sa"] / g["ns"], mean_sel_pos=g["sp"] / g["ns"], mean_sel_neg=g["sn"] / g["ns"],
                    mean_abs_sel_sig=g["ssig"] / g["k"], mean_abs_sel_sig_pos=g["ssp"] / g["kp"],
                    mean_abs_sel_sig_neg=g["ssn"] / g["kn"])


METRICS = ["frac_sig", "frac_pos", "frac_neg", "mean_abs_sel", "mean_sel_pos", "mean_sel_neg", "mean_abs_sel_sig",
           "mean_abs_sel_sig_pos", "mean_abs_sel_sig_neg"]
INTERACTION_METRICS = ["frac_sig", "frac_pos", "frac_neg", "mean_abs_sel", "mean_sel_pos", "mean_sel_neg"]
ANOVA_METRICS = {"frac_sig": ("k", "n"), "frac_pos": ("kp", "n"), "frac_neg": ("kn", "n"), "mean_abs_sel": ("sa", "ns"),
                 "mean_sel_pos": ("sp", "ns"), "mean_sel_neg": ("sn", "ns")}


def gini(x):
    x = np.sort(np.clip(np.asarray(x, float), 0, None)); n = len(x)
    return np.nan if n == 0 or x.sum() == 0 else (2 * np.sum(np.arange(1, n + 1) * x) / (n * x.sum()) - (n + 1) / n)


def entropy_focality(x):
    x = np.clip(np.asarray(x, float), 0, None)
    if len(x) < 2 or x.sum() == 0:
        return np.nan
    p = x / x.sum(); p = p[p > 0]
    return 1 - (-(p * np.log(p)).sum()) / np.log(len(x))


# ------------------------------------------------------------------ comparison engine
def compare(sm, wA, wB, perm_pairs, boot_pairs):
    """wA, wB: (S,) 0/1 masks of the two groups; perm_pairs: list of (wA_p, wB_p) (P,S); boot: (B,S) multiplicities"""
    gA, gB = sm.group(wA), sm.group(wB)
    nA, nB = gA["n"], gB["n"]
    sessA = (sm.a["n"] * wA[:, None] > 0).sum(0); sessB = (sm.a["n"] * wB[:, None] > 0).sum(0)
    incl = (nA + nB >= MIN_UNITS) & (sessA >= MIN_SESS) & (sessB >= MIN_SESS)
    mA, mB = metrics(gA), metrics(gB)
    pA, pB = perm_pairs
    mpA, mpB = metrics(sm.group(pA)), metrics(sm.group(pB))
    bA, bB = boot_pairs
    mbA, mbB = metrics(sm.group(bA)), metrics(sm.group(bB))
    out = dict(incl=incl, nA=nA, nB=nB, sessA=sessA, sessB=sessB)
    for k in mA:
        d = mB[k] - mA[k]; dp = mpB[k] - mpA[k]; db = mbB[k] - mbA[k]
        out[f"{k}_A"], out[f"{k}_B"], out[f"{k}_diff"] = mA[k], mB[k], d
        with np.errstate(invalid="ignore"):
            out[f"{k}_p"] = (1 + np.nansum(np.abs(dp) >= np.abs(d)[None] - 1e-12, 0)) / (1 + np.sum(np.isfinite(dp), 0))
        out[f"{k}_lo"], out[f"{k}_hi"] = np.nanpercentile(db, 2.5, 0), np.nanpercentile(db, 97.5, 0)
        # focality of frac_sig across included regions (computed by caller for frac_sig only)
        out[f"_{k}_perm"] = (mpA[k], mpB[k]); out[f"_{k}_boot"] = (mbA[k], mbB[k])
    return out


def focality_block(res, key="frac_sig"):
    inc = res["incl"]
    if inc.sum() < 3:
        return None
    A, B = res[f"{key}_A"][inc], res[f"{key}_B"][inc]
    (pA, pB), (bA, bB) = res[f"_{key}_perm"], res[f"_{key}_boot"]
    r = {}
    for name, fn in [("gini", gini), ("entropy_focality", entropy_focality)]:
        fa, fb = fn(A), fn(B)
        dp = np.array([fn(b[inc]) - fn(a[inc]) for a, b in zip(pA, pB)])
        db = np.array([fn(b[inc]) - fn(a[inc]) for a, b in zip(bA, bB)])
        r.update({f"{name}_A": fa, f"{name}_B": fb, f"{name}_diff": fb - fa,
                  f"{name}_p": (1 + np.nansum(np.abs(dp) >= abs(fb - fa) - 1e-12)) / (1 + np.isfinite(dp).sum()),
                  f"{name}_lo": np.nanpercentile(db, 2.5), f"{name}_hi": np.nanpercentile(db, 97.5)})
    r["n_regions"] = int(inc.sum())
    return r


def main(a):
    global OUT
    t0 = time.time()
    OUT = pathlib.Path(a.out) if a.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    U = pd.read_parquet(BASE / "units.parquet")
    U = U[U.cohort.isin(["R+", "R-"]) & ((U.quality_label == "good") if a.quality == "good" else True)].reset_index(drop=True)
    U["uid"] = np.arange(len(U))
    L = pd.read_parquet(a.roc_long or (BASE / "roc_long.parquet"))
    L["cluster_id"] = L.cluster_id.astype(str)
    L = L.merge(U[KEYS + ["uid"]], on=KEYS, how="inner")
    U = U.set_index("uid")
    M = build_measures(U, L)
    M = {k: v for k, v in M.items() if v.valid.any()}                  # categories need the main ROC types
    if a.measures:
        M = {k: v for k, v in M.items() if k in a.measures}
    sess_tab = U.groupby("session_id").agg(mouse=("mouse_id", "first"), cohort=("cohort", "first"), stage=("stage", "first"))
    sess = sess_tab.index.to_numpy(); S = len(sess)
    coh = sess_tab.cohort.to_numpy(); stg = sess_tab.stage.to_numpy(); mouse = sess_tab.mouse.to_numpy()
    mice = np.unique(mouse); m_idx = np.searchsorted(mice, mouse)
    mouse_coh = pd.Series(coh, index=mouse).groupby(level=0).first().reindex(mice).to_numpy()
    print(f"{S} sessions ({pd.crosstab(coh, stg).to_dict()}), {len(mice)} mice, {len(U)} units ({a.quality}), {len(M)} measures",
          flush=True)
    P, B = a.n_perm, a.n_boot

    def boot(mask):
        idx = np.where(mask)[0]
        W = np.zeros((B, S))
        for b in range(B):
            np.add.at(W[b], rng.choice(idx, len(idx), replace=True), 1)
        return W

    # comparison definitions: name -> (maskA, maskB, perm (PA, PB), boot (BA, BB))
    comps = {}
    for c in ["R+", "R-"]:
        sel = coh == c
        A, Bm = (sel & (stg == "learning")).astype(float), (sel & (stg == "expert")).astype(float)
        lab = np.where(sel, (stg == "expert").astype(float), np.nan)
        PA, PB = np.zeros((P, S)), np.zeros((P, S))
        for p in range(P):
            l = lab.copy(); l[sel] = rng.permutation(lab[sel]); PB[p] = np.nan_to_num(l); PA[p] = sel * (1 - np.nan_to_num(l))
        comps[f"stage:{c}"] = (A, Bm, (PA, PB), (boot(A > 0), boot(Bm > 0)))
    for s_ in ["learning", "expert"]:
        sel = stg == s_
        A, Bm = (sel & (coh == "R-")).astype(float), (sel & (coh == "R+")).astype(float)
        PA, PB = np.zeros((P, S)), np.zeros((P, S))
        for p in range(P):
            mc = rng.permutation(mouse_coh)[m_idx]
            PB[p] = sel * (mc == "R+"); PA[p] = sel * (mc == "R-")
        comps[f"cohort:{s_}"] = (A, Bm, (PA, PB), (boot(A > 0), boot(Bm > 0)))
    # interaction permutation: cohort labels across mice
    perm_coh = np.array([rng.permutation(mouse_coh)[m_idx] for _ in range(P)])
    boots = {(c, s_): boot((coh == c) & (stg == s_)) for c in ["R+", "R-"] for s_ in ["learning", "expert"]}

    region_rows, foc_rows, anova_rows, dist_rows, matched_rows = [], [], [], [], []
    for mi_, (mname, meas) in enumerate(M.items()):
        meas.attrs["name"] = mname
        for level in LEVELS:
            sm = Sums(U, meas, level, sess)
            res_by = {}
            for cname, (A, Bm, perm_pairs, boot_pairs) in comps.items():
                r = compare(sm, A, Bm, perm_pairs, boot_pairs)
                res_by[cname] = r
                for j, reg in enumerate(sm.regions):
                    row = dict(measure=mname, level=level, comparison=cname, region=reg, included=bool(r["incl"][j]),
                               n_units_A=int(r["nA"][j]), n_units_B=int(r["nB"][j]), n_sess_A=int(r["sessA"][j]),
                               n_sess_B=int(r["sessB"][j]))
                    for k in METRICS:
                        for suf in ["A", "B", "diff", "p", "lo", "hi"]:
                            row[f"{k}_{suf}"] = float(r[f"{k}_{suf}"][j])
                    region_rows.append(row)
                if level != "all":
                    for key in ["frac_sig", "mean_abs_sel"]:
                        fb = focality_block(r, key)
                        if fb:
                            foc_rows.append(dict(measure=mname, level=level, comparison=cname, metric=key, **fb))
            # interaction: delta(R+) - delta(R-), for fraction and |selectivity|, together and per sign
            sP, sM = res_by["stage:R+"], res_by["stage:R-"]
            inc = sP["incl"] & sM["incl"]
            masks = {(c, s_): ((perm_coh == c) & (stg == s_)[None]).astype(float) for c in ["R+", "R-"]
                     for s_ in ["learning", "expert"]}
            mperm = {key: metrics(sm.group(w)) for key, w in masks.items()}          # (P, R) per metric
            mboot = {key: metrics(sm.group(w)) for key, w in boots.items()}
            for k in INTERACTION_METRICS:
                obs = sP[f"{k}_diff"] - sM[f"{k}_diff"]
                null = (mperm[("R+", "expert")][k] - mperm[("R+", "learning")][k]) - \
                       (mperm[("R-", "expert")][k] - mperm[("R-", "learning")][k])
                bd = (mboot[("R+", "expert")][k] - mboot[("R+", "learning")][k]) - \
                     (mboot[("R-", "expert")][k] - mboot[("R-", "learning")][k])
                for j, reg in enumerate(sm.regions):
                    region_rows.append(dict(measure=mname, level=level, comparison=f"interaction:{k}", region=reg,
                                            included=bool(inc[j]), **{f"{k}_diff": float(obs[j]),
                                            f"{k}_p": float((1 + np.nansum(np.abs(null[:, j]) >= abs(obs[j]) - 1e-12)) /
                                                            (1 + np.isfinite(null[:, j]).sum())),
                                            f"{k}_lo": float(np.nanpercentile(bd[:, j], 2.5)),
                                            f"{k}_hi": float(np.nanpercentile(bd[:, j], 97.5))}))
            # ANOVA (area_group level): frac ~ group * region, session x region rows weighted by n
            if level == "area_group" and not a.skip_anova:
                for metric in ANOVA_METRICS:
                    if mname.startswith("cat:") and metric not in ("frac_sig",):
                        continue
                    anova_rows += run_anova(sm, mname, comps, coh, stg, mouse_coh, m_idx, rng, a.n_perm_anova, metric)
        # selectivity distribution shift (ROC types only, all units pooled)
        if not mname.startswith("cat:"):
            dist_rows += sel_distribution(U, meas, sess, coh, stg, m_idx, mouse_coh, rng, a.n_perm_dist)
        matched_rows += matched_mice(U, meas, rng)
        print(f"[{mi_ + 1}/{len(M)}] {mname} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    pd.DataFrame(region_rows).to_csv(OUT / "region_stats.csv", index=False)
    pd.DataFrame(foc_rows).to_csv(OUT / "focality.csv", index=False)
    pd.DataFrame(anova_rows).to_csv(OUT / "anova.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUT / "selectivity_distribution.csv", index=False)
    pd.DataFrame(matched_rows).to_csv(OUT / "matched_mice.csv", index=False)
    json.dump(dict(n_perm=P, n_boot=B, min_units_total=MIN_UNITS, min_sessions_per_group=MIN_SESS, levels=LEVELS,
                   core_types_mixed=CORE, unit_selection=f"quality: {a.quality}", roc_long=str(a.roc_long or BASE / "roc_long.parquet"), stage="learning day 0 / expert day >= 1",
                   permutation={"stage": "stage labels across sessions within cohort",
                                "cohort": "cohort labels across mice within stage",
                                "interaction": "cohort labels across mice"},
                   correction="none", runtime_min=round((time.time() - t0) / 60, 1)),
              open(OUT / "stats_provenance.json", "w"), indent=2)
    print("ALL DONE", flush=True)


def run_anova(sm, mname, comps, coh, stg, mouse_coh, m_idx, rng, n_perm, metric="frac_sig"):
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    num, den = ANOVA_METRICS[metric]
    rows = []
    for cname in comps:
        kind, val = cname.split(":")
        sel = (coh == val) if kind == "stage" else (stg == val)
        grp = stg if kind == "stage" else coh
        S_idx = np.where(sel)[0]
        recs = []
        for s_ in S_idx:
            for j, reg in enumerate(sm.regions):
                n = sm.a[den][s_, j]
                if n >= 3:
                    recs.append(dict(sess=s_, g=grp[s_], region=reg, y=sm.a[num][s_, j] / n, w=n))
        D = pd.DataFrame(recs)
        if D.empty or D.g.nunique() < 2 or D.region.nunique() < 2:
            continue
        keep = D.groupby("region").g.nunique() == 2
        D = D[D.region.isin(keep[keep].index)]
        try:
            fit = smf.wls("y ~ C(g) * C(region)", data=D, weights=D.w).fit()
            tab = anova_lm(fit, typ=2)
        except Exception:                                            # noqa: BLE001
            continue
        F_obs = tab.loc["C(g)", "F"]; F_int = tab.loc["C(g):C(region)", "F"]
        null_g, null_i = [], []
        sess_ids = D.sess.unique()
        g_of = dict(zip(D.sess, D.g))
        for _ in range(n_perm):
            if kind == "stage":
                perm = dict(zip(sess_ids, rng.permutation([g_of[x] for x in sess_ids])))
            else:
                mc = rng.permutation(mouse_coh)[m_idx]; perm = {x: mc[x] for x in sess_ids}
            Dp = D.assign(g=D.sess.map(perm))
            try:
                tp = anova_lm(smf.wls("y ~ C(g) * C(region)", data=Dp, weights=Dp.w).fit(), typ=2)
                null_g.append(tp.loc["C(g)", "F"]); null_i.append(tp.loc["C(g):C(region)", "F"])
            except Exception:                                        # noqa: BLE001
                pass
        rows.append(dict(measure=mname, metric=metric, comparison=cname, n_rows=len(D), n_regions=D.region.nunique(),
                         F_group=F_obs, p_group=tab.loc["C(g)", "PR(>F)"], F_region=tab.loc["C(region)", "F"],
                         p_region=tab.loc["C(region)", "PR(>F)"], F_interaction=F_int,
                         p_interaction=tab.loc["C(g):C(region)", "PR(>F)"],
                         p_perm_group=(1 + np.sum(np.array(null_g) >= F_obs)) / (1 + len(null_g)),
                         p_perm_interaction=(1 + np.sum(np.array(null_i) >= F_int)) / (1 + len(null_i))))
    return rows


def sel_distribution(U, meas, sess, coh, stg, m_idx, mouse_coh, rng, n_perm):
    v = meas.valid.to_numpy() & np.isfinite(meas.sel.to_numpy(float))
    x = meas.sel.to_numpy(float)[v]; s_of = U.session_id.to_numpy()[v]
    si = pd.Series(np.arange(len(sess)), index=sess)[s_of].to_numpy()
    rows = []
    defs = {"stage:R+": (coh == "R+", stg == "expert", "stage"), "stage:R-": (coh == "R-", stg == "expert", "stage"),
            "cohort:learning": (stg == "learning", coh == "R+", "cohort"), "cohort:expert": (stg == "expert", coh == "R+", "cohort")}
    for name, (scope, isB, kind) in defs.items():
        in_scope = scope[si]
        xs, sis = x[in_scope], si[in_scope]
        b = isB[sis]
        if b.sum() < 20 or (~b).sum() < 20:
            continue
        obs = stats.wasserstein_distance(xs[b], xs[~b])
        dmean = np.mean(np.abs(xs[b])) - np.mean(np.abs(xs[~b]))
        null_w, null_m = [], []
        sc = np.where(scope)[0]
        for _ in range(n_perm):
            if kind == "stage":
                lab = isB.copy(); lab[sc] = rng.permutation(isB[sc])
            else:
                lab = (rng.permutation(mouse_coh)[m_idx] == "R+")
            bp = lab[sis]
            if bp.sum() == 0 or (~bp).sum() == 0:
                continue
            null_w.append(stats.wasserstein_distance(xs[bp], xs[~bp]))
            null_m.append(np.mean(np.abs(xs[bp])) - np.mean(np.abs(xs[~bp])))
        q = np.percentile
        rows.append(dict(measure=meas.attrs.get("name", ""), comparison=name, n_A=int((~b).sum()), n_B=int(b.sum()),
                         wasserstein=obs, p_wasserstein=(1 + np.sum(np.array(null_w) >= obs)) / (1 + len(null_w)),
                         d_mean_abs_sel=dmean, p_mean_abs_sel=(1 + np.sum(np.abs(null_m) >= abs(dmean))) / (1 + len(null_m)),
                         median_A=np.median(xs[~b]), median_B=np.median(xs[b]), q10_A=q(xs[~b], 10), q90_A=q(xs[~b], 90),
                         q10_B=q(xs[b], 10), q90_B=q(xs[b], 90)))
    return rows


def matched_mice(U, meas, rng):
    rows = []
    d = pd.DataFrame(dict(mouse=U.mouse_id, cohort=U.cohort, stage=U.stage, v=meas.valid.to_numpy(),
                          f=meas.flag.to_numpy(float), a=meas.abs_sel.to_numpy(float)))
    d = d[d.v]
    for c, g in d.groupby("cohort"):
        both = g.groupby("mouse").stage.nunique()
        mm = both[both == 2].index
        if len(mm) < 3:
            continue
        pm = g[g.mouse.isin(mm)].groupby(["mouse", "stage"]).agg(f=("f", "mean"), a=("a", "mean")).unstack()
        for k in ["f", "a"]:
            diff = (pm[(k, "expert")] - pm[(k, "learning")]).dropna().to_numpy()
            if len(diff) < 3:
                continue
            signs = np.array(list(itertools.product([-1, 1], repeat=len(diff)))) if len(diff) <= 14 else \
                rng.choice([-1, 1], size=(10000, len(diff)))
            null = (signs * diff).mean(1)
            rows.append(dict(measure=meas.attrs.get("name", ""), cohort=c, metric="frac_sig" if k == "f" else "mean_abs_sel",
                             n_mice=len(diff), mean_diff=diff.mean(), p_signflip=(np.sum(np.abs(null) >= abs(diff.mean()) - 1e-12)) / len(null),
                             p_wilcoxon=stats.wilcoxon(diff).pvalue if np.any(diff != 0) else np.nan))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm-anova", type=int, default=100)
    ap.add_argument("--n-perm-dist", type=int, default=500)
    ap.add_argument("--measures", nargs="*", default=None)
    ap.add_argument("--skip-anova", action="store_true")
    ap.add_argument("--roc-long", default=None, help="alternative unit x analysis_type table (default: 045 roc_long)")
    ap.add_argument("--out", default=None, help="output directory (default: ssl-roc-learning-stages/stats)")
    ap.add_argument("--quality", default="good", choices=["good", "all"])
    main(ap.parse_args())
