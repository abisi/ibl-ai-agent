"""009 -- EDA: which ROC selectivities are enriched where whisker and auditory cortex both project strongly?

Shared-input level (user question 2026-10-08: "regions with lots of input from whisker and auditory cortices"):
per voxel and modality, the mass rank r(v) = fraction of the modality's projection mass (002 merged, line-balanced,
smoothed density) held by voxels at least as dense as v (0 = densest, 1 = no projection or source area); the voxel is in
the X % zone iff r <= X. Shared level L = max(r_whisker, r_auditory) = the strictest contour at which v lies in BOTH zones.
Categories: shared core (L <= CORE, default 0.5), shared shell (CORE < L <= 0.9), whisker zone only, auditory zone only
(90 % zones), outside both.
Units: ROC stage analysis (_roc_stage_analysis: units.parquet + roc_long.parquet), good + mua; UNITS=stable (default)
keeps the tracked stable units (hit/miss 137, via the pre-lick table's `stable` column), UNITS=all keeps all.
ROC: every analysis type; per unit sig (pos / neg) and |sel|.
Quantities (per analysis type): fraction significant (total, positive, negative) and mean |sel| per category and per
shared-level bin; enrichment = core vs rest of brain.
Tests (core vs rest; session = unit of analysis): per-session difference (sessions with >= MIN_UNITS units in the core and
outside), Wilcoxon signed-rank and paired t-test; area control = Mantel-Haenszel odds ratio of sig (core vs not) within
Allen structures (area_acronym_custom with >= MIN_UNITS units on each side) with the CMH test. Benjamini-Hochberg over
analysis types within each measure. Pooled over cohorts and stages, and per cohort x stage (descriptive).
Outputs: combined_results_ks4/_sensory_spatial_maps/shared_zone_roc<_UNITS>/ (unit table with KEYS, tests, bins,
provenance) and figures (heatmap, forest, per group, dose-response).
"""
import importlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m3 = importlib.import_module("003_spatial_maps")
m6 = importlib.import_module("006_colocation_figure")
S, KEYS = m3.S, m3.KEYS
UNITS = os.environ.get("UNITS", "stable")
CORE = float(os.environ.get("CORE", "0.5"))
OUTER = 0.9
MIN_UNITS = 10
OUTD = m3.OUT / f"shared_zone_roc_{UNITS}"
FIGD = OUTD / "figures"
STABLE_SRC = m3.RES / "ssl-prelick-convergence/across_days/sl_stable/prelick_units.parquet"
ZONES = m3.OUT / "projection_zones.npz"          # densities are identical at every contour level
CATS = ["shared core", "shared shell", "whisker zone only", "auditory zone only", "outside both"]
BINS = [0, 0.3, 0.5, 0.7, 0.9, 1.0001]
BIN_LAB = ["≤0.3", "0.3–0.5", "0.5–0.7", "0.7–0.9", ">0.9"]
FAMILY = [("Sensory (active)", ["whisker_active", "auditory_active", "wh_vs_aud_active", "whisker_sensory", "auditory_sensory"]),
          ("Sensory (passive)", ["whisker_passive_pre", "whisker_passive_post", "auditory_passive_pre", "auditory_passive_post",
                                 "wh_vs_aud_passive_pre", "wh_vs_aud_passive_post"]),
          ("Choice / lick", ["whisker_choice", "auditory_choice", "choice", "whisker_hit_vs_cr", "auditory_hit_vs_cr",
                             "spontaneous_licks", "spontaneous_licks_vs_cr", "whisker_hit_vs_spontaneous",
                             "auditory_hit_vs_spontaneous"]),
          ("Baseline", ["baseline_choice", "baseline_whisker_choice", "baseline_auditory_choice"]),
          ("Passive pre vs post", ["whisker_pre_vs_post_learning", "auditory_pre_vs_post_learning",
                                   "wh_vs_aud_pre_vs_post_learning", "baseline_pre_vs_post_learning"])]
ORDER = [a for _, l in FAMILY for a in l]
GROUPS = [("R+", "learning"), ("R+", "expert"), ("R-", "learning"), ("R-", "expert")]
MEAS = [("sig", "Fraction significant"), ("pos", "Fraction excited (sel > 0)"), ("neg", "Fraction inhibited (sel < 0)"),
        ("abs_sel", "Mean |selectivity|")]


def mass_rank(d):
    f = d.ravel().astype(float)
    o = np.argsort(-f, kind="stable")
    cs = np.cumsum(f[o]) / f.sum()
    r = np.empty_like(cs)
    r[o] = cs
    r[f <= 0] = 1.0
    return r.reshape(d.shape)


def load():
    U = pd.read_parquet(m3.ROC / "units.parquet")
    U = U[U.quality_label.isin(["good", "mua"]) & U.cohort.isin(["R+", "R-"])].drop_duplicates(KEYS)
    if UNITS == "stable":
        st = pd.read_parquet(STABLE_SRC, columns=["session_id", "electrode_group", "cluster_id", "stable"])
        st["cluster_id"] = st.cluster_id.astype(str)
        U["cluster_id"] = U.cluster_id.astype(str)
        U = U.merge(st.drop_duplicates(["session_id", "electrode_group", "cluster_id"]),
                    on=["session_id", "electrode_group", "cluster_id"], how="left")
        U = U[U.stable.fillna(False).astype(bool)]
    U = U[U[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].notna().all(1)].reset_index(drop=True)
    U["ml_f"] = m3.MID + np.abs(U.ccf_atlas_ml - m3.MID)
    Z = np.load(ZONES)
    rw, ra = mass_rank(Z["density_whisker"]), mass_rank(Z["density_auditory"])
    ijk, ok = m6.unit_voxels(U, rw.shape)
    U["r_whisker"] = np.where(ok, rw[ijk[:, 0], ijk[:, 1], ijk[:, 2]], 1.0)
    U["r_auditory"] = np.where(ok, ra[ijk[:, 0], ijk[:, 1], ijk[:, 2]], 1.0)
    U["L_shared"] = U[["r_whisker", "r_auditory"]].max(1)
    w, a = U.r_whisker <= OUTER, U.r_auditory <= OUTER
    U["zone_cat"] = np.select([U.L_shared <= CORE, w & a, w, a], CATS[:4], CATS[4])
    U["L_bin"] = pd.cut(U.L_shared, BINS, labels=BIN_LAB, right=True)
    R = pd.read_parquet(m3.ROC / "roc_long.parquet", columns=KEYS + ["analysis_type", "sel", "abs_sel", "sig", "pos", "neg"])
    R["cluster_id"] = R.cluster_id.astype(str)
    U["cluster_id"] = U.cluster_id.astype(str)
    R = R[R.sel.notna()].drop_duplicates(KEYS + ["analysis_type"])
    D = R.merge(U[KEYS + ["cohort", "stage", "area_group", "area_acronym_custom", "r_whisker", "r_auditory", "L_shared",
                          "zone_cat", "L_bin"]], on=KEYS, how="inner")
    for c in ("sig", "pos", "neg"):
        D[c] = D[c].astype(float)
    return U, D


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.sum():
        pv = p[ok]
        o = np.argsort(pv)
        r = pv[o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        out = np.empty_like(r)
        out[o] = np.minimum(r, 1)
        q[ok] = out
    return q


def cmh(G, col):
    """Mantel-Haenszel OR and CMH test of binary `col` (core vs not) within area_acronym_custom strata"""
    num = den = a_sum = e_sum = v_sum = 0.0
    n_str = 0
    for _, g in G.groupby("area_acronym_custom"):
        c = g.zone_cat == "shared core"
        if c.sum() < MIN_UNITS or (~c).sum() < MIN_UNITS:
            continue
        y = g[col].to_numpy() > 0
        a, b = np.sum(y & c), np.sum(~y & c)
        cc, d = np.sum(y & ~c), np.sum(~y & ~c)
        n = a + b + cc + d
        num += a * d / n
        den += b * cc / n
        r1, c1 = a + b, a + cc
        a_sum += a
        e_sum += r1 * c1 / n
        v_sum += r1 * (n - r1) * c1 * (n - c1) / (n * n * (n - 1)) if n > 1 else 0
        n_str += 1
    if n_str == 0 or den == 0 or v_sum == 0:
        return np.nan, np.nan, n_str
    chi = (abs(a_sum - e_sum) - 0.5) ** 2 / v_sum
    return num / den, float(stats.chi2.sf(chi, 1)), n_str


def session_test(G, col):
    c = G.zone_cat == "shared core"
    A = G.assign(core=c).groupby(["session_id", "core"])[col].agg(["mean", "size"]).unstack()
    A = A[(A[("size", True)] >= MIN_UNITS) & (A[("size", False)] >= MIN_UNITS)]
    d = (A[("mean", True)] - A[("mean", False)]).to_numpy()
    if len(d) < 5:
        return dict(n_sess=len(d))
    return dict(n_sess=len(d), diff_mean=float(d.mean()), diff_sem=float(d.std(ddof=1) / np.sqrt(len(d))),
                p_wilcoxon=float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0, p_ttest=float(stats.ttest_1samp(d, 0).pvalue))


def run_tests(D):
    rows = []
    for at, G in D.groupby("analysis_type"):
        c = G.zone_cat == "shared core"
        for col, _ in MEAS:
            r = dict(analysis_type=at, measure=col, n_core=int(c.sum()), n_rest=int((~c).sum()),
                     core=float(G.loc[c, col].mean()), rest=float(G.loc[~c, col].mean()))
            r.update(session_test(G, col))
            if col != "abs_sel":
                r["or_mh"], r["p_cmh"], r["n_strata"] = cmh(G, col)
            rows.append(r)
    T = pd.DataFrame(rows)
    for col, _ in MEAS:
        m = T.measure == col
        T.loc[m, "q_wilcoxon"] = bh(T.loc[m, "p_wilcoxon"])
        T.loc[m, "q_ttest"] = bh(T.loc[m, "p_ttest"])
        if col != "abs_sel":
            T.loc[m, "q_cmh"] = bh(T.loc[m, "p_cmh"])
    T["log2_ratio"] = np.log2(T.core / T.rest)
    return T


def tables(D):
    C = D.groupby(["analysis_type", "zone_cat"])[["sig", "pos", "neg", "abs_sel"]].mean().reset_index()
    C["n"] = D.groupby(["analysis_type", "zone_cat"]).size().to_numpy()
    B = D.groupby(["analysis_type", "L_bin"], observed=True)[["sig", "pos", "neg", "abs_sel"]].mean().reset_index()
    B["n"] = D.groupby(["analysis_type", "L_bin"], observed=True).size().to_numpy()
    Gr = []
    for coh, stg in GROUPS:
        g = D[(D.cohort == coh) & (D.stage == stg)]
        c = g.zone_cat == "shared core"
        t = g.groupby("analysis_type").apply(lambda x: pd.Series(dict(
            core=x.loc[x.zone_cat == "shared core", "sig"].mean(), rest=x.loc[x.zone_cat != "shared core", "sig"].mean(),
            n_core=int((x.zone_cat == "shared core").sum()))), include_groups=False).reset_index()
        t["group"] = f"{coh} {stg}"
        Gr.append(t)
    Gr = pd.concat(Gr, ignore_index=True)
    Gr["log2_ratio"] = np.log2(Gr.core / Gr.rest)
    return C, B, Gr


# ------------------------------------------------------------------------------------------------ figures
def label(at):
    return at.replace("_", " ").replace("wh vs aud", "whisker vs auditory")


def stars(q):
    return "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else ""


def fig_overview(plt, C, T, U):
    from matplotlib.colors import TwoSlopeNorm
    ats = [a for a in ORDER if a in set(C.analysis_type)]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 0.17 * len(ats) + 1.4), gridspec_kw=dict(width_ratios=[1.3, 1, 1]))
    allf = C.groupby("analysis_type").apply(lambda x: np.average(x.sig, weights=x.n), include_groups=False)
    M = C.pivot(index="analysis_type", columns="zone_cat", values="sig").loc[ats, CATS]
    E = np.log2(M.div(allf.loc[ats], axis=0))
    ax = axs[0]
    im = ax.imshow(E.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    ax.set_yticks(range(len(ats)), [label(a) for a in ats], fontsize=5)
    ncat = U.zone_cat.value_counts()
    ax.set_xticks(range(len(CATS)), [f"{c}\n(n={ncat.get(c, 0)})" for c in CATS], fontsize=4.8, rotation=40, ha="right")
    ax.set_title("Fraction significant,\nlog2 ratio to all units", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=5)
    for k, (col, ttl) in enumerate((("sig", "Core − rest of brain\n(session mean ± s.e.m.)"), ("abs_sel", "Core − rest, mean |sel|"))):
        ax = axs[k + 1]
        t = T[T.measure == col].set_index("analysis_type").reindex(ats)
        y = np.arange(len(ats))
        ax.errorbar(t.diff_mean, y, xerr=t.diff_sem, fmt="o", ms=2.2, lw=0.7, color="k")
        ax.axvline(0, color="0.6", lw=0.6)
        for yi, (q, qc) in enumerate(zip(t.q_wilcoxon, t.get("q_cmh", pd.Series(np.nan, index=t.index)))):
            s = stars(q) + ("" if col == "abs_sel" or not np.isfinite(qc) or qc >= 0.05 else " †")
            if s:
                ax.text(1.02, yi, s, transform=ax.get_yaxis_transform(), fontsize=5, va="center")
        ax.set_ylim(len(ats) - 0.5, -0.5)
        ax.set_yticks(y, [""] * len(ats))
        ax.set_title(ttl, fontsize=6)
        ax.tick_params(labelsize=5)
    fig.text(0.99, 0.005, f"* Wilcoxon over sessions, BH q; † area-stratified (Mantel-Haenszel) q < 0.05. Core: shared level L ≤ {CORE}.",
             fontsize=4.5, ha="right")
    fig.tight_layout()
    return fig


def fig_signs(plt, T):
    ats = [a for a in ORDER if a in set(T.analysis_type)]
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 0.17 * len(ats) + 1.2), sharey=True)
    for ax, (col, ttl) in zip(axs, MEAS[:3]):
        t = T[T.measure == col].set_index("analysis_type").reindex(ats)
        y = np.arange(len(ats))
        ax.barh(y, t.log2_ratio, color=np.where(t.q_wilcoxon < 0.05, "#d94801", "0.7"), height=0.7)
        ax.axvline(0, color="0.4", lw=0.6)
        ax.set_title(f"{ttl}\nlog2(core / rest)", fontsize=6)
        ax.tick_params(labelsize=5)
    axs[0].set_yticks(range(len(ats)), [label(a) for a in ats], fontsize=5)
    axs[0].set_ylim(len(ats) - 0.5, -0.5)
    fig.text(0.99, 0.005, "orange: Wilcoxon over sessions q < 0.05", fontsize=4.5, ha="right")
    fig.tight_layout()
    return fig


def fig_groups(plt, Gr):
    from matplotlib.colors import TwoSlopeNorm
    ats = [a for a in ORDER if a in set(Gr.analysis_type)]
    M = Gr.pivot(index="analysis_type", columns="group", values="log2_ratio").loc[ats, [f"{c} {s}" for c, s in GROUPS]]
    N = Gr.groupby("group").n_core.max()
    fig, ax = plt.subplots(figsize=(3.4, 0.17 * len(ats) + 1.2))
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    ax.set_yticks(range(len(ats)), [label(a) for a in ats], fontsize=5)
    ax.set_xticks(range(4), [f"{g.replace('-', '−')}\n(core n≤{N.get(g, 0)})" for g in M.columns], fontsize=5, rotation=30, ha="right")
    ax.set_title("Fraction significant, log2(core / rest)\nper cohort × stage (descriptive)", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.05).ax.tick_params(labelsize=5)
    fig.tight_layout()
    return fig


def fig_dose(plt, B, T):
    t = T[T.measure == "sig"].sort_values("p_wilcoxon")
    top = t.analysis_type.head(8).tolist()
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True)
    for ax, at in zip(axs.ravel(), top):
        b = B[B.analysis_type == at].set_index("L_bin").reindex(BIN_LAB)
        for col, c in (("sig", "k"), ("pos", "#d6604d"), ("neg", "#4393c3")):
            ax.plot(range(len(BIN_LAB)), b[col], "-o", ms=2, lw=0.9, color=c, label={"sig": "all", "pos": "excited", "neg": "inhibited"}[col])
        ax.set_title(label(at), fontsize=5.5)
        ax.set_xticks(range(len(BIN_LAB)), BIN_LAB, fontsize=4.5, rotation=40)
        ax.tick_params(labelsize=5)
    axs[0, 0].legend(fontsize=4.5, frameon=False)
    for ax in axs[1]:
        ax.set_xlabel("Shared level L (strong → no shared input)", fontsize=5)
    for ax in axs[:, 0]:
        ax.set_ylabel("Fraction significant", fontsize=5)
    fig.tight_layout()
    return fig


def main():
    FIGD.mkdir(parents=True, exist_ok=True)
    plt = S.setup()
    U, D = load()
    print(U.zone_cat.value_counts(), flush=True)
    T = run_tests(D)
    C, B, Gr = tables(D)
    T.to_csv(OUTD / "core_vs_rest_tests.csv", index=False)
    C.to_csv(OUTD / "zone_category_fractions.csv", index=False)
    B.to_csv(OUTD / "shared_level_bins.csv", index=False)
    Gr.to_csv(OUTD / "per_group_core_vs_rest.csv", index=False)
    U[KEYS + ["cohort", "stage", "day", "area_group", "area_acronym_custom", "ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv",
              "r_whisker", "r_auditory", "L_shared", "zone_cat"]].to_parquet(OUTD / "units_shared_level.parquet")
    top = U[U.zone_cat == "shared core"].area_acronym_custom.value_counts().head(15)
    top.to_csv(OUTD / "core_unit_areas.csv")
    for name, f in (("overview", fig_overview(plt, C, T, U)), ("per_sign", fig_signs(plt, T)),
                    ("per_group", fig_groups(plt, Gr)), ("dose_response", fig_dose(plt, B, T))):
        for ext in ("png", "pdf", "svg"):
            f.savefig(FIGD / f"shared_zone_roc_{name}.{ext}", dpi=300)
    prov = dict(script="009_shared_zone_roc_enrichment.py", units=UNITS, core_level=CORE, outer_level=OUTER,
                zones=str(ZONES), roc=str(m3.ROC), stable_source=str(STABLE_SRC) if UNITS == "stable" else None,
                n_units=len(U), n_sessions=int(U.session_id.nunique()), n_mice=int(U.mouse_id.nunique()),
                n_core=int((U.zone_cat == "shared core").sum()), n_core_sessions=int(U[U.zone_cat == "shared core"].session_id.nunique()),
                min_units=MIN_UNITS)
    (OUTD / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(top.to_string())
    s = T[T.measure == "sig"].sort_values("p_wilcoxon")
    print(s[["analysis_type", "core", "rest", "log2_ratio", "n_sess", "diff_mean", "q_wilcoxon", "q_ttest", "or_mh", "q_cmh", "n_strata"]].round(4).to_string())
    print("ALL DONE", OUTD)


if __name__ == "__main__":
    main()
