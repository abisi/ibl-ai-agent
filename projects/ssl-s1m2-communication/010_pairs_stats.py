"""Readouts and statistics of the connected-pairs CCA (009 maps). Session = unit of analysis; exploratory, uncorrected.

Readouts per (session, pair, condition), total CCV (noise kept in the table):
- v3 profile, option a (s1m2_lib.readout_clean: only map cells whose source and target 50-ms windows are clear of the
  -10..+5 ms gap): v3_FF (in-sample, source leads 0..+15 ms, baseline-subtracted, mean 0..100 ms), v3_FB (target leads),
  v3_CV_FF (10-fold CV), v3_shuffle_FF (trial-shuffle map), v3_FF_minus_shuffle, v3_FF_minus_FB.
- hires profile (as 007): artefact-free cells, baseline = window starts -100..-25 ms per delay; source-leading trace
  (delays 0..+15 ms, a time point counts only if every delay is clear); hr_cv_e0 / hr_is_e0 = mean over window starts
  5-15 ms (CV / in-sample), e1 15-35 ms, e2 35-95 ms; hr_is_ff_minus_fb = source- minus target-leading, starts 25-95 ms.
Stats per pair x condition x readout x stage: per cohort mean, Wilcoxon vs 0; R+ vs R-: Mann-Whitney and Welch.
Cohort-level rows carry pass_<cohort> (>= 3 sessions at both stages, 008). Projection strength vs coupling: (1) per
session with >= 3 pairs, Spearman rho across its pairs between log10 Allen enrichment and the readout, Wilcoxon of rho vs 0;
(2) mixed model readout ~ log10 enrichment + same_probe_frac + (1 | session) per cohort x stage, and adjusted for log10
unit counts of both areas (lmm_adj); (3) pair level: mean
readout per pair vs log10 enrichment, Spearman and Pearson, all pairs and without shared-probe pairs (same_probe > 0.2).
Probe exclusion (pairs_lib.PAIR_CONFIG exclude_probes_csv or --flagged): if a flagged table exists, everything is also
written with flagged session-pairs removed (suffix _noflag).

Output: OUT/pairs/tables/readouts.parquet, stats.csv, scaling_session.csv, scaling_pair.csv (+ _noflag).
Run (haas): ../../.venv/bin/python 010_pairs_stats.py [--flagged path.csv]
"""
import argparse
import copy
import json

import numpy as np
import pandas as pd
from scipy import stats

import pairs_lib as P

L = P.L
BASE_CONFIG = copy.deepcopy(L.CONFIG)
BASE = (-0.100, -0.025)
EPOCHS = [(0.005, 0.015), (0.015, 0.035), (0.035, 0.095)]
DIR_EPOCH = (0.025, 0.095)
FF, FB = (0.0, 0.015), (-0.015, 0.0)
READOUTS = ["v3_FF", "v3_CV_FF", "v3_FF_minus_shuffle", "v3_FF_minus_FB", "hr_cv_e0", "hr_is_e0", "hr_cv_e1", "hr_cv_e2",
            "hr_is_ff_minus_fb"]
KEY = ["hr_cv_e0", "v3_CV_FF"]


def use_profile(name):
    L.CONFIG.clear()
    L.CONFIG.update(copy.deepcopy(BASE_CONFIG))
    return L.set_profile(name)


def wil(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(stats.wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


def hires_traces(z, kind):
    T, ws, D, Dcv = z["times"], z["window_starts"], z["delays"], z["cv_delays"]
    out, maps = {}, {}
    for lab, m, dd in (("is", z[f"{kind}__ccv"].astype(float), D), ("cv", z[f"{kind}__cv"].astype(float), Dcv)):
        gm = L.gap_mask(dd, T)
        m = np.where(gm, m, np.nan)
        b = (ws >= BASE[0] - 1e-9) & (ws <= BASE[1] + 1e-9)
        m = m - np.nanmean(m[:, b], axis=1, keepdims=True)
        maps[lab] = m
        for nm, rng in (("ff", FF), ("fb", FB)):
            sel = (dd >= rng[0] - 1e-9) & (dd <= rng[1] + 1e-9)
            out[f"{lab}_{nm}"] = np.where(gm[sel].all(axis=0), np.nanmean(m[sel], axis=0), np.nan)
    r = {}
    for i, (a, b_) in enumerate(EPOCHS):
        e = (ws >= a - 1e-9) & (ws <= b_ + 1e-9)
        r[f"hr_is_e{i}"] = np.nanmean(out["is_ff"][e])
        r[f"hr_cv_e{i}"] = np.nanmean(out["cv_ff"][e])
    e = (ws >= DIR_EPOCH[0] - 1e-9) & (ws <= DIR_EPOCH[1] + 1e-9)
    r["hr_is_ff_minus_fb"] = np.nanmean(out["is_ff"][e] - out["is_fb"][e])
    return r, out, maps["is"], ws, D


def load():
    """one row per (session, pair, condition, kind) with both profiles' readouts; hires traces / maps for figures"""
    rows, extra = {}, {}
    use_profile("v3")
    for f in sorted((P.OUT / "maps_v3").glob("*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=True)  # own files: unit arrays are object dtype
        meta = json.loads(str(z["meta"]))
        t, d, dcv = z["times"], z["delays"], z["cv_delays"]
        for kind in ("total", "noise"):
            c = z[f"{kind}__ccv"].astype(float)
            ff = L.readout_clean(c, d, t, FF)[0]
            fb = L.readout_clean(c, d, t, FB)[0]
            sh = L.readout_clean(z[f"{kind}__shuffle"].astype(float), d, t, FF)[0]
            cv = L.readout_clean(z[f"{kind}__cv"].astype(float), dcv, t, FF)[0]
            k = (meta["session_id"], meta["pair"], meta["condition"], kind)
            rows[k] = dict({x: meta[x] for x in meta if x not in ("profile", "elapsed_s")}, kind=kind,
                           source_units=";".join("/".join(u) for u in z["source_units"]),
                           target_units=";".join("/".join(u) for u in z["target_units"]),
                           v3_FF=ff, v3_FB=fb, v3_CV_FF=cv, v3_shuffle_FF=sh, v3_FF_minus_shuffle=ff - sh,
                           v3_FF_minus_FB=ff - fb)
    use_profile("hires")
    for f in sorted((P.OUT / "maps_v4_hires").glob("*.npz")):
        if ".tmp" in f.name:
            continue
        z = np.load(f, allow_pickle=True)  # own files: unit arrays are object dtype
        meta = json.loads(str(z["meta"]))
        for kind in ("total", "noise"):
            r, tr, m, ws, D = hires_traces(z, kind)
            k = (meta["session_id"], meta["pair"], meta["condition"], kind)
            rows.setdefault(k, dict({x: meta[x] for x in meta if x not in ("profile", "elapsed_s")}, kind=kind)).update(r)
            if kind == "total":
                extra[k[:3]] = (tr, m, ws, D)
    df = pd.DataFrame(list(rows.values()))
    C = pd.read_csv(P.TABLES / "connectivity_pairs.csv")
    keep = ["pair", "class", "allen_enrichment", "allen_frac_zone70", "liu", "liu_class", "anat_direction",
            "pass_R+", "pass_R-", "same_probe_frac_mean"]
    df = df.merge(C[keep], on="pair", how="left")
    df["log_enrichment"] = np.log10(df.allen_enrichment)
    df["config"] = json.dumps(dict(pair_config=P.PAIR_CONFIG, v3=BASE_CONFIG, hires=L.PROFILES["hires"],
                                   readouts=dict(FF=FF, FB=FB, hires_baseline=BASE, epochs=EPOCHS, dir_epoch=DIR_EPOCH)))
    return df, extra


def flag_filter(df, flagged):
    if not flagged:
        return None
    bad = np.zeros(len(df), bool)
    for i, (sid, su, tu) in enumerate(zip(df.session_id, df.source_units.fillna(""), df.target_units.fillna(""))):
        probes = {u.split("/")[0] for u in (su + ";" + tu).split(";") if u}
        bad[i] = any((sid, p) in flagged for p in probes)
    return df[~bad]


def stats_table(df):
    out = []
    d = df[df.kind == "total"]
    for (pair, cond, stage), g in d.groupby(["pair", "condition", "stage"]):
        info = dict(pair=pair, condition=cond, stage=stage, **{k: g[k].iloc[0] for k in ("class", "allen_enrichment")})
        for col in READOUTS:
            for coh in ("R+", "R-"):
                h = g[g.cohort == coh][col].dropna()
                out.append(dict(info, readout=col, test=f"{coh} vs 0 (Wilcoxon)", cohort=coh,
                                cohort_passes=bool(g[f"pass_{coh}"].iloc[0]), n_sessions=len(h),
                                n_mice=g[(g.cohort == coh) & g[col].notna()].mouse_id.nunique(),
                                mean=h.mean(), sem=h.std() / np.sqrt(max(len(h), 1)), p=wil(h)))
            a, b = g[g.cohort == "R+"][col].dropna(), g[g.cohort == "R-"][col].dropna()
            if len(a) >= 3 and len(b) >= 3:
                out.append(dict(info, readout=col, test="R+ vs R-", cohort="R+ vs R-",
                                cohort_passes=bool(g["pass_R+"].iloc[0] and g["pass_R-"].iloc[0]),
                                n_sessions=len(a) + len(b), mean=a.mean() - b.mean(),
                                p_mannwhitney=stats.mannwhitneyu(a, b).pvalue,
                                p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue))
    return pd.DataFrame(out)


def scaling(df):
    import statsmodels.formula.api as smf
    d = df[df.kind == "total"]
    srows, prows = [], []
    for (cond, stage, coh), g in d.groupby(["condition", "stage", "cohort"]):
        for col in READOUTS:
            # (1) within-session rho
            rhos = []
            for sid, h in g.groupby("session_id"):
                h = h[[col, "log_enrichment"]].dropna()
                if len(h) >= 3 and h.log_enrichment.nunique() >= 3:
                    rhos.append(stats.spearmanr(h.log_enrichment, h[col])[0])
            rhos = np.array(rhos, float)
            r = dict(condition=cond, stage=stage, cohort=coh, readout=col, n_sessions_rho=int(np.isfinite(rhos).sum()),
                     mean_rho=np.nanmean(rhos) if len(rhos) else np.nan, p_rho_wilcoxon=wil(rhos))
            # (2) mixed model
            h = g[[col, "log_enrichment", "same_probe_frac", "session_id"]].dropna().rename(columns={col: "y"})
            if h.session_id.nunique() >= 5 and len(h) >= 10:
                try:
                    fit = smf.mixedlm("y ~ log_enrichment + same_probe_frac", h, groups=h.session_id).fit(reml=True)
                    r.update(lmm_slope=fit.params["log_enrichment"], lmm_p=fit.pvalues["log_enrichment"],
                             lmm_same_probe=fit.params["same_probe_frac"], lmm_same_probe_p=fit.pvalues["same_probe_frac"],
                             lmm_n=len(h), lmm_sessions=h.session_id.nunique())
                except Exception as e:  # noqa: BLE001
                    r["lmm_error"] = str(e)[:80]
            # (2b) same, adjusted for unit counts (CV CCV depends on how many units / dims each area has)
            h = g[[col, "log_enrichment", "same_probe_frac", "session_id", "n_source", "n_target"]].dropna()
            h = h.rename(columns={col: "y"}).assign(ls=lambda x: np.log10(x.n_source), lt=lambda x: np.log10(x.n_target))
            if h.session_id.nunique() >= 5 and len(h) >= 10:
                try:
                    fit = smf.mixedlm("y ~ log_enrichment + same_probe_frac + ls + lt", h, groups=h.session_id).fit(reml=True)
                    r.update(lmm_adj_slope=fit.params["log_enrichment"], lmm_adj_p=fit.pvalues["log_enrichment"],
                             lmm_adj_n_target_slope=fit.params["lt"], lmm_adj_n_target_p=fit.pvalues["lt"])
                except Exception as e:  # noqa: BLE001
                    r["lmm_adj_error"] = str(e)[:80]
            srows.append(r)
            # (3) pair level
            pm = g.groupby("pair").agg(y=(col, "mean"), x=("log_enrichment", "first"), n=(col, "count"),
                                       sp=("same_probe_frac", "mean"), cls=("class", "first"))
            pm = pm[pm.n >= 3].dropna(subset=["y", "x"])
            for subset, q in (("all", pm), ("different_probes", pm[pm.sp <= 0.2])):
                if len(q) >= 4:
                    prows.append(dict(condition=cond, stage=stage, cohort=coh, readout=col, subset=subset, n_pairs=len(q),
                                      spearman_rho=stats.spearmanr(q.x, q.y)[0], spearman_p=stats.spearmanr(q.x, q.y)[1],
                                      pearson_r=stats.pearsonr(q.x, q.y)[0], pearson_p=stats.pearsonr(q.x, q.y)[1],
                                      pairs=";".join(q.index)))
    return pd.DataFrame(srows), pd.DataFrame(prows)


def find_flagged(path):
    if path:
        return P.flagged_probes(path)
    f = P.flagged_probes()
    if f is not None:
        return f
    scr = P.L.ROOT / "combined_results_ks4" / "ssl-whisker-artefact-gap"
    for c in sorted(scr.rglob("*flag*.csv")):
        try:
            x = pd.read_csv(c, nrows=2)
        except Exception:  # noqa: BLE001
            continue
        if {"session_id", "electrode_group"} <= set(x.columns):
            print("flagged-probe table:", c)
            return P.flagged_probes(c)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagged")
    a = ap.parse_args()
    df, _ = load()
    df.to_parquet(P.TABLES / "readouts.parquet", index=False)
    variants = [("", df)]
    fl = find_flagged(a.flagged)
    if fl:
        dn = flag_filter(df, fl)
        print(f"flagged probes: {len(fl)}; rows {len(df)} -> {len(dn)}")
        variants.append(("_noflag", dn))
    else:
        print("no flagged-probe table found: no exclusion variant")
    for suf, d in variants:
        stats_table(d).to_csv(P.TABLES / f"stats{suf}.csv", index=False)
        s, p = scaling(d)
        s.to_csv(P.TABLES / f"scaling_session{suf}.csv", index=False)
        p.to_csv(P.TABLES / f"scaling_pair{suf}.csv", index=False)
    S = pd.read_csv(P.TABLES / "stats.csv")
    pd.set_option("display.width", 250)
    for ro in KEY:
        x = S[(S.readout == ro) & (S.condition == "W_all") & (S.stage == "learning")]
        print(f"\n== {ro}, W_all, learning")
        print(x[["pair", "class", "allen_enrichment", "cohort", "cohort_passes", "n_sessions", "mean", "p", "p_mannwhitney",
                 "p_welch"]].round(4).to_string(index=False))
    s = pd.read_csv(P.TABLES / "scaling_session.csv")
    p = pd.read_csv(P.TABLES / "scaling_pair.csv")
    print(s[s.readout.isin(KEY) & s.condition.isin(["W_all", "All"])].round(4).to_string(index=False))
    print(p[p.readout.isin(KEY) & p.condition.isin(["W_all", "All"])].drop(columns="pairs").round(4).to_string(index=False))


if __name__ == "__main__":
    main()
