"""SuppFigure3_areas_cohorts -- areas, mice and cohorts of the rastermap clusters (user 2026-11-26: "S-D"; message:
differences in area sampling between the cohorts do not drive the response-type structure). One compute_<key>() per
panel, each writing data/<csv> of the dated snapshot; numbers for the captions in _build/panel_meta.json.

Data: rastermap v2 main (stable good + MUA units, learner mice, whisker day 0, one session per mouse; 100 clusters in
rastermap order), metaclusters contiguous at cut 1 - r = 0.40 (as Figure 3, same colours). Cohort per mouse from the
reference sheet (share_internal dataset_info). Mouse (= session) is the unit for every cohort test; R+ vs R- always
Mann-Whitney U and Welch t (common.two_group_tests), both reported. CCF positions from the decoding-rerun unit table.
Reused analyses (user's rastermap post hoc, unit_spikes_analysis rastermap_psth): f_{m,k} matrix + per-cluster cohort
tests + PERMANOVA (rastermap_utils stats), per-mouse dispersion metrics (rastermap_utils.compute_dispersion_metrics),
area specialization as KL divergence vs a label shuffle (SuppFigure3_rastermap_clustering compute_area_specialization).
New here: f_{m,k} expected from each mouse's area sampling (sampling-only model) and the cohort tests on the residual;
per-area cohort comparison of the distribution over metaclusters (Jensen-Shannon, cohort labels permuted across mice).

Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
  .venv/bin/python ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure3_areas_cohorts/compute.py [--only key ...]
then build.py figures/SuppFigure3_areas_cohorts.
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import matplotlib.colors as mc
import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
REPO = FIG_DIR.parents[3]
V2 = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/v2")
os.environ["RASTERMAP_META_CUT"] = "0.40"
os.environ["RASTERMAP_VARIANTS_ROOT"] = str(V2)
sys.path.insert(0, str(FIG_DIR.parents[1]))
sys.path.insert(0, str(REPO / "projects" / "ssl-rastermap-psth-variants" / "exploratory-analyses"))
sys.path.insert(0, str(Path.home() / "code" / "unit_spikes_analysis"))
from common import load_spec, snapshot_dir, two_group_tests  # noqa: E402
import rastermap_psth.population_matrix_summary as pms  # noqa: E402
import rastermap_psth.rastermap_utils as ru  # noqa: E402
import ephys_utilities.allen_utils.allen_utils as allen_utils  # noqa: E402

SPEC = load_spec(FIG_DIR)
PANELS = {p.get("key", p["id"]): p for p in SPEC["panels"]}
CUT = 0.40
SHEET = Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
UNITS = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-decoding-rerun-202611/units/units_stable.parquet")
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
COH = {"R+": "#00B400", "R-": "#C800C8"}
N_PERM = 2000
MIN_MICE_COHORT = 3          # area x cohort comparisons: >= 3 mice of each cohort with >= MIN_NEURONS_MOUSE neurons
MIN_NEURONS_MOUSE = 5
FINE_MIN_MICE = 5            # fine areas shown: >= 5 mice with >= MIN_NEURONS_MOUSE neurons, >= FINE_MIN_NEURONS neurons
FINE_MIN_NEURONS = 200
YIELD_MIN = 3                # a mouse "contributes" to a cluster with >= 3 neurons (as 045 MIN_NEURONS_MOUSE)
rng = np.random.default_rng(0)
META = {}
_c = {}


def base():
    if _c:
        return _c
    m050 = importlib.import_module("050_rastermap_variants_v2")
    md = m050.method_dir("main")
    fam = pms.metaclustering(md, cut=CUT)["families"]
    R = pms.load_rastermap(md)
    lab = pd.read_csv(md / "neuron_cluster_labels_cv.csv")
    assert np.array_equal(lab.unit_ids.to_numpy(), R["unit_ids"])
    sh = pd.read_excel(SHEET).drop_duplicates("mouse_id").set_index("mouse_id")
    lab["cohort"] = lab.mouse_id.map(sh.reward_group)
    mism = lab.drop_duplicates("mouse_id").query("cohort != reward_group")[["mouse_id", "reward_group", "cohort"]]
    K = int(R["labels"].max()) + 1
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(K)])
    df = pd.DataFrame(dict(mouse_id=lab.mouse_id.astype(str), session_id=lab.session_id.astype(str),
                           electrode_group=lab.electrode_group.astype(str), cluster_id=lab.cluster_id.astype(str),
                           cohort=lab.cohort, cluster=R["labels"], metacluster=[fam[k] for k in R["labels"]],
                           area=R["area_arr"], area_group=R["area_group_arr"]))
    acol = allen_utils.get_custom_area_groups_colors()
    order = allen_utils.get_area_group_custom_order()
    _c.update(df=df, K=K, fam=fam, pal=pal, acol=acol, order=order, mism=mism)
    META["_shared"] = dict(method_dir=str(md), cut=CUT, n_neurons=int(len(df)), n_mice=int(df.mouse_id.nunique()),
                           n_mice_rplus=int(df.drop_duplicates("mouse_id").cohort.eq("R+").sum()),
                           n_mice_rminus=int(df.drop_duplicates("mouse_id").cohort.eq("R-").sum()),
                           n_metaclusters=int(len(set(fam.values()))),
                           cohort_label_mismatch_vs_sheet=mism.to_dict("records"))
    return _c


def mc_hex(f):
    return mc.to_hex(base()["pal"][f])


def group_order(groups):
    o = base()["order"]
    return sorted(groups, key=lambda g: o.index(g) if g in o else 999)


def mouse_cohort():
    return base()["df"].drop_duplicates("mouse_id").set_index("mouse_id").cohort


# ------------------------------------------------------------------------------------------------ a: area sampling
def compute_area_sampling():
    """per mouse x area group: neurons and fraction of the mouse's neurons; per group R+ vs R- (MW + Welch) on the
    fractions; overall: distance between the cohort-mean composition vectors vs cohort labels permuted across mice"""
    df = base()["df"]
    C = pd.crosstab(df.mouse_id, df.area_group)
    F = C.div(C.sum(1), axis=0)
    coh = mouse_cohort().reindex(F.index)
    groups = group_order(F.columns.tolist())
    rows, tests = [], {}
    for g in groups:
        t = two_group_tests(F.loc[coh == "R+", g], F.loc[coh == "R-", g])
        tests[g] = t
        for m in F.index:
            rows.append(dict(mouse_id=m, cohort=coh[m], area_group=g, color=base()["acol"].get(g, "#aaaaaa"),
                             n=int(C.loc[m, g]), frac=float(F.loc[m, g]), mw_p=t["mannwhitney_p"], welch_p=t["welch_p"]))
    X, y = F[groups].to_numpy(), (coh == "R+").to_numpy()
    dist = lambda yy: np.linalg.norm(X[yy].mean(0) - X[~yy].mean(0))
    obs = dist(y)
    null = np.array([dist(rng.permutation(y)) for _ in range(N_PERM)])
    META["area_sampling"] = dict(n_groups=len(groups), dist=float(obs), p_perm=float((1 + (null >= obs).sum()) / (1 + N_PERM)),
                                 n_perm=N_PERM, n_groups_p05_both=int(sum(t["mannwhitney_p"] < 0.05 and t["welch_p"] < 0.05
                                                                          for t in tests.values())),
                                 n_groups_p05_any=int(sum(t["mannwhitney_p"] < 0.05 or t["welch_p"] < 0.05
                                                          for t in tests.values())),
                                 groups_p05=[g for g, t in tests.items() if t["mannwhitney_p"] < 0.05 or t["welch_p"] < 0.05])
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ b: CCF positions
def compute_ccf():
    df = base()["df"]
    U = pd.read_parquet(UNITS, columns=KEYS + ["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]).astype(
        {"cluster_id": str, "electrode_group": str, "session_id": str, "mouse_id": str}).drop_duplicates(KEYS)
    d = df.merge(U, on=KEYS, how="left")
    ok = d[["ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]].notna().all(1)
    d = d[ok]
    d["color"] = d.area_group.map(lambda g: base()["acol"].get(g, "#aaaaaa"))
    META["ccf"] = dict(n_with_ccf=int(ok.sum()), n_total=int(len(ok)),
                       ml_range=[float(d.ccf_atlas_ml.min()), float(d.ccf_atlas_ml.max())],
                       ap_range=[float(d.ccf_atlas_ap.min()), float(d.ccf_atlas_ap.max())])
    return d[["mouse_id", "session_id", "electrode_group", "cluster_id", "cohort", "area_group", "color",
              "ccf_atlas_ap", "ccf_atlas_ml", "ccf_atlas_dv"]]


# ------------------------------------------------------------------------------------------------ c: mouse yield
def fmatrix():
    df = base()["df"]
    K = base()["K"]
    C = pd.crosstab(df.mouse_id, df.cluster).reindex(columns=range(K), fill_value=0)
    return C, C.div(C.sum(1), axis=0)


def compute_mouse_yield():
    """f_{m,k} = fraction of mouse m's neurons in cluster k (user's F matrix); per cluster: mice with >= YIELD_MIN neurons
    (per cohort), mice contributing >= 1 % of the cluster's neurons, and the cluster's R+ share of neurons vs the R+
    share of all neurons"""
    df, K, fam = base()["df"], base()["K"], base()["fam"]
    C, F = fmatrix()
    coh = mouse_cohort().reindex(C.index)
    share_all = float((df.cohort == "R+").mean())
    rows = []
    for m in C.index:
        for k in range(K):
            rows.append(dict(kind="cell", mouse_id=m, cohort=coh[m], cluster=k, metacluster=fam[k], mc_color=mc_hex(fam[k]),
                             n=int(C.loc[m, k]), frac=float(F.loc[m, k]), n_mouse=int(C.loc[m].sum())))
    per = []
    for k in range(K):
        col = C[k]
        per.append(dict(kind="cluster", cluster=k, metacluster=fam[k], mc_color=mc_hex(fam[k]), n=int(col.sum()),
                        n_mice_ge3=int((col >= YIELD_MIN).sum()),
                        n_mice_ge3_rplus=int(((col >= YIELD_MIN) & (coh == "R+")).sum()),
                        n_mice_ge3_rminus=int(((col >= YIELD_MIN) & (coh == "R-")).sum()),
                        n_mice_ge1pct=int((col / col.sum() >= 0.01).sum()),
                        max_mouse_share=float((col / col.sum()).max()),
                        rplus_share=float(col[coh == "R+"].sum() / col.sum()), rplus_share_all=share_all))
    p = pd.DataFrame(per)
    nm = len(C)
    META["mouse_yield"] = dict(n_mice=nm, yield_min=YIELD_MIN, median_mice_ge3=float(p.n_mice_ge3.median()),
                               min_mice_ge3=int(p.n_mice_ge3.min()),
                               n_clusters_half_mice=int((p.n_mice_ge3 >= nm / 2).sum()),
                               n_clusters_both_cohorts_ge5=int(((p.n_mice_ge3_rplus >= 5) & (p.n_mice_ge3_rminus >= 5)).sum()),
                               median_mice_ge1pct=float(p.n_mice_ge1pct.median()),
                               max_mouse_share_median=float(p.max_mouse_share.median()),
                               n_clusters_max_share_gt_half=int((p.max_mouse_share > 0.5).sum()),
                               rplus_share_all=share_all, rplus_share_iqr=[float(p.rplus_share.quantile(0.25)),
                                                                           float(p.rplus_share.quantile(0.75))])
    return pd.concat([pd.DataFrame(rows), p], ignore_index=True)


# ------------------------------------------------------------------------------------------------ d: dispersion
def compute_dispersion():
    """per-mouse concentration of the cluster repertoire (user's rastermap_utils.compute_dispersion_metrics on f_{m,k}),
    R+ vs R-"""
    C, F = fmatrix()
    coh = mouse_cohort().reindex(F.index)
    d = ru.compute_dispersion_metrics(F.to_numpy())
    d.insert(0, "mouse_id", F.index)
    d.insert(1, "cohort", coh.to_numpy())
    d["n_neurons"] = C.sum(1).to_numpy()
    META["dispersion"] = {m: two_group_tests(d.loc[d.cohort == "R+", m], d.loc[d.cohort == "R-", m])
                          for m in ("shannon_entropy_norm", "ipr", "gini", "max_frac")}
    for m, t in META["dispersion"].items():
        d[f"mw_p_{m}"], d[f"welch_p_{m}"] = t["mannwhitney_p"], t["welch_p"]
    return d


# ------------------------------------------------------------------------------------------------ e/f: occupancy vs sampling
def _bh(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def _permanova(X, y, n=N_PERM):
    """Euclidean PERMANOVA (two groups), pseudo-F and permutation p (as the user's rastermap_utils._permanova)"""
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    N = len(X)

    def F(yy):
        sst = D2.sum() / (2 * N)
        ssw = sum(D2[np.ix_(yy == g, yy == g)].sum() / (2 * (yy == g).sum()) for g in (True, False))
        return ((sst - ssw) / 1) / (ssw / (N - 2))
    f0 = F(y)
    null = np.array([F(rng.permutation(y)) for _ in range(n)])
    return float(f0), float((1 + (null >= f0).sum()) / (1 + n))


def compute_occupancy():
    if "occupancy" in _c:
        return _c["occupancy"]
    return _occupancy()


def _occupancy():
    """per cluster: mean f_{m,k} per cohort (observed); expected f_{m,k} from the mouse's area sampling alone,
    E_{m,k} = sum_a (n_{m,a} / n_m) p(k | a), p(k | a) from all OTHER mice's neurons of fine area a (leave-mouse-out);
    residual = observed - expected. R+ vs R- (MW + Welch, BH across clusters) on observed, expected and residual;
    PERMANOVA on each matrix"""
    df, K, fam = base()["df"], base()["K"], base()["fam"]
    C, F = fmatrix()
    mice = F.index.to_numpy()
    coh = mouse_cohort().reindex(mice)
    A = pd.crosstab([df.mouse_id, df.area], df.cluster).reindex(columns=range(K), fill_value=0)
    tot_a = A.groupby(level=1).sum()
    E = np.zeros_like(F.to_numpy())
    for i, m in enumerate(mice):
        Am = A.loc[m]
        other = tot_a.loc[Am.index] - Am                       # leave-mouse-out area -> cluster distribution
        pka = other.div(other.sum(1).replace(0, np.nan), axis=0).fillna(0)
        w = Am.sum(1) / Am.sum().sum()
        E[i] = (pka.mul(w, axis=0)).sum(0).to_numpy()
    Fo = F.to_numpy()
    Res = Fo - E
    y = (coh == "R+").to_numpy()
    rows, pv = [], {}
    for name, M in (("observed", Fo), ("expected", E), ("residual", Res)):
        ts = [two_group_tests(M[y, k], M[~y, k]) for k in range(K)]
        qm, qw = _bh([t["mannwhitney_p"] for t in ts]), _bh([t["welch_p"] for t in ts])
        pv[name] = (qm, qw)
        for k, t in enumerate(ts):
            rows.append(dict(matrix=name, cluster=k, metacluster=fam[k], mc_color=mc_hex(fam[k]),
                             mean_rplus=float(M[y, k].mean()), mean_rminus=float(M[~y, k].mean()),
                             diff=float(M[y, k].mean() - M[~y, k].mean()),
                             sem_diff=float(np.sqrt(M[y, k].var(ddof=1) / y.sum() + M[~y, k].var(ddof=1) / (~y).sum())),
                             mw_p=t["mannwhitney_p"], welch_p=t["welch_p"], mw_q=qm[k], welch_q=qw[k],
                             sig_both=bool(qm[k] < 0.05 and qw[k] < 0.05)))
    d = pd.DataFrame(rows)
    meta = {}
    for name, M in (("observed", Fo), ("expected", E), ("residual", Res)):
        f, p = _permanova(M, y)
        meta[name] = dict(permanova_F=f, permanova_p=p, n_sig_both=int(d[(d.matrix == name)].sig_both.sum()),
                          n_sig_mw=int((d[(d.matrix == name)].mw_q < 0.05).sum()),
                          n_sig_welch=int((d[(d.matrix == name)].welch_q < 0.05).sum()))
    ob = d[d.matrix == "observed"].set_index("cluster")["diff"]
    ex = d[d.matrix == "expected"].set_index("cluster")["diff"]
    from scipy.stats import pearsonr
    r, p = pearsonr(ob, ex)
    meta["r_obs_vs_expected_diff"] = float(r)
    meta["p_obs_vs_expected_diff"] = float(p)
    meta["var_explained"] = float(1 - ((ob - ex) ** 2).sum() / ((ob - ob.mean()) ** 2).sum())
    meta["n_perm"] = N_PERM
    for name in ("observed", "expected", "residual"):
        d[f"permanova_F_{name}"], d[f"permanova_p_{name}"] = meta[name]["permanova_F"], meta[name]["permanova_p"]
    d["r_obs_vs_expected"], d["p_obs_vs_expected"] = meta["r_obs_vs_expected_diff"], meta["p_obs_vs_expected_diff"]
    META["occupancy"] = meta
    META["occupancy_scatter"] = meta
    _c["occupancy"] = d
    return d


def compute_occupancy_scatter():
    return compute_occupancy()


# ------------------------------------------------------------------------------------------------ g: area enrichment by cohort
def _js(p, q):
    m = (p + q) / 2
    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a > 0, a * np.log2(a / b), 0).sum()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compute_area_enrichment():
    """per area group (>= MIN_MICE_COHORT mice of each cohort with >= MIN_NEURONS_MOUSE neurons): distribution of its
    neurons over the metaclusters per cohort, log2 enrichment over the pooled dataset distribution; cohort difference =
    Jensen-Shannon divergence (bits) between the R+ and R- distributions, null = cohort labels permuted across the
    area's mice"""
    df = base()["df"]
    ks = sorted(set(base()["fam"].values()))
    kix = {k: i for i, k in enumerate(ks)}
    p0 = np.bincount([kix[k] for k in df.metacluster], minlength=len(ks)) / len(df)
    coh = mouse_cohort()
    rows, out = [], {}
    for g in group_order(df.area_group.unique().tolist()):
        dg = df[df.area_group == g]
        nm = dg.groupby("mouse_id").size()
        mice = nm.index[nm >= MIN_NEURONS_MOUSE]
        cm = coh.reindex(mice)
        if (cm == "R+").sum() < MIN_MICE_COHORT or (cm == "R-").sum() < MIN_MICE_COHORT:
            continue
        dg = dg[dg.mouse_id.isin(mice)]
        M = pd.crosstab(dg.mouse_id, dg.metacluster).reindex(columns=ks, fill_value=0).reindex(mice).to_numpy(float)
        yy = (cm == "R+").to_numpy()

        def dist(y_):
            a, b = M[y_].sum(0), M[~y_].sum(0)
            return _js(a / a.sum(), b / b.sum())
        obs = dist(yy)
        null = np.array([dist(rng.permutation(yy)) for _ in range(N_PERM)])
        p = float((1 + (null >= obs).sum()) / (1 + N_PERM))
        out[g] = dict(js=float(obs), null_p95=float(np.percentile(null, 95)), p_perm=p, n_rplus=int(yy.sum()),
                      n_rminus=int((~yy).sum()))
        for c, sel in (("R+", yy), ("R-", ~yy)):
            v = M[sel].sum(0)
            fr = v / v.sum()
            for j, k in enumerate(ks):
                rows.append(dict(area_group=g, color=base()["acol"].get(g, "#aaaaaa"), cohort=c, metacluster=k,
                                 mc_color=mc_hex(k), n=int(v[j]), frac=float(fr[j]), frac_dataset=float(p0[j]),
                                 log2_enrichment=float(np.log2((v[j] + 0.5) / (v.sum() * p0[j] + 0.5))),
                                 js=float(obs), null_p95=out[g]["null_p95"], p_perm=p, n_mice_rplus=int(yy.sum()),
                                 n_mice_rminus=int((~yy).sum())))
    d = pd.DataFrame(rows)
    q = _bh([v["p_perm"] for v in out.values()])
    qd = dict(zip(out, q))
    d["q_bh"] = d.area_group.map(qd)
    corr = {g: float(np.corrcoef(d[(d.area_group == g) & (d.cohort == "R+")].log2_enrichment,
                                 d[(d.area_group == g) & (d.cohort == "R-")].log2_enrichment)[0, 1]) for g in out}
    META["area_enrichment"] = dict(n_groups=len(out), n_p05=int(sum(v["p_perm"] < 0.05 for v in out.values())),
                                   n_q05=int((q < 0.05).sum()), groups_p05=[g for g, v in out.items() if v["p_perm"] < 0.05],
                                   r_profiles_median=float(np.median(list(corr.values()))),
                                   r_profiles_min=float(min(corr.values())), n_perm=N_PERM,
                                   min_mice_cohort=MIN_MICE_COHORT, min_neurons_mouse=MIN_NEURONS_MOUSE)
    return d


# ------------------------------------------------------------------------------------------------ h/i: specialization
def _kl_bits(counts, p0):
    p = counts / counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * np.log2(p / p0), 0.0).sum(1)


def _specialization(area_col, areas, level_col):
    """KL divergence (bits) of each area's distribution over the clusters (level_col) from the dataset distribution;
    null: area labels permuted across all neurons (as SuppFigure3 compute_area_specialization)"""
    df = base()["df"]
    lv = df[level_col].to_numpy()
    ks = np.unique(lv)
    li = np.searchsorted(ks, lv)
    p0 = np.bincount(li, minlength=len(ks)) / len(lv)
    ar = df[area_col].to_numpy()
    ai = {a: i for i, a in enumerate(areas)}

    def table(a_):
        m = np.isin(a_, areas)
        idx = np.array([ai[x] for x in a_[m]]) * len(ks) + li[m]
        return np.bincount(idx, minlength=len(areas) * len(ks)).reshape(len(areas), len(ks)).astype(float)
    kl = _kl_bits(table(ar), p0)
    null = np.array([_kl_bits(table(rng.permutation(ar)), p0) for _ in range(200)])
    return kl, null


def compute_specialization():
    """area groups: specialization over the fine clusters and over the metaclusters"""
    df = base()["df"]
    groups = group_order(df.area_group.value_counts().index[df.area_group.value_counts() >= FINE_MIN_NEURONS].tolist())
    groups = [g for g in groups if g != "Other"]
    rows = []
    for lvl, col in (("fine clusters", "cluster"), ("metaclusters", "metacluster")):
        kl, null = _specialization("area_group", groups, col)
        for i, g in enumerate(groups):
            rows.append(dict(level=lvl, area_group=g, color=base()["acol"].get(g, "#aaaaaa"),
                             n=int((df.area_group == g).sum()), n_mice=int(df[df.area_group == g].mouse_id.nunique()),
                             kl_bits=float(kl[i]), null_p95=float(np.percentile(null[:, i], 95)),
                             p_perm=float((1 + (null[:, i] >= kl[i]).sum()) / 201)))
    d = pd.DataFrame(rows)
    META["specialization"] = {lvl: dict(n_groups=len(groups), n_sig=int((d[d.level == lvl].p_perm < 0.05).sum()),
                                        kl_median=float(d[d.level == lvl].kl_bits.median()))
                              for lvl in ("fine clusters", "metaclusters")}
    META["specialization"]["min_neurons"] = FINE_MIN_NEURONS
    return d


def compute_fine_areas():
    """best-sampled fine areas (>= FINE_MIN_MICE mice with >= MIN_NEURONS_MOUSE neurons and >= FINE_MIN_NEURONS neurons):
    specialization over the metaclusters and the fine clusters, and the metacluster carrying most of the area's
    neurons relative to the dataset (max log2 enrichment)"""
    df = base()["df"]
    nm = df.groupby(["area", "mouse_id"]).size()
    n_mice = (nm >= MIN_NEURONS_MOUSE).groupby(level=0).sum()
    n = df.area.value_counts()
    areas = [a for a in n.index if n[a] >= FINE_MIN_NEURONS and n_mice.get(a, 0) >= FINE_MIN_MICE and a not in ("unknown", "nan")]
    grp = df.groupby("area").area_group.agg(lambda x: x.value_counts().index[0])
    ks = sorted(set(base()["fam"].values()))
    p0 = df.metacluster.value_counts(normalize=True).reindex(ks).to_numpy()
    rows = []
    res = {lvl: _specialization("area", areas, col) for lvl, col in (("fine clusters", "cluster"),
                                                                       ("metaclusters", "metacluster"))}
    for i, a in enumerate(areas):
        v = df[df.area == a].metacluster.value_counts().reindex(ks, fill_value=0).to_numpy(float)
        enr = np.log2((v + 0.5) / (v.sum() * p0 + 0.5))
        top = ks[int(np.argmax(enr))]
        for lvl, (kl, null) in res.items():
            rows.append(dict(level=lvl, area=a, area_group=grp[a], color=base()["acol"].get(grp[a], "#aaaaaa"),
                             n=int(n[a]), n_mice=int(n_mice[a]), kl_bits=float(kl[i]),
                             null_p95=float(np.percentile(null[:, i], 95)),
                             p_perm=float((1 + (null[:, i] >= kl[i]).sum()) / 201),
                             top_metacluster=int(top), top_mc_color=mc_hex(top), top_log2_enrichment=float(enr.max())))
    d = pd.DataFrame(rows)
    META["fine_areas"] = dict(n_areas=len(areas), n_areas_total=int(df.area.nunique()), min_mice=FINE_MIN_MICE,
                              min_neurons=FINE_MIN_NEURONS, min_neurons_mouse=MIN_NEURONS_MOUSE,
                              n_sig_meta=int((d[d.level == "metaclusters"].p_perm < 0.05).sum()))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    out = snapshot_dir(SPEC)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "_build").mkdir(exist_ok=True)
    mf = out / "_build" / "panel_meta.json"
    allmeta = json.load(open(mf)) if mf.exists() else {}
    for key, p in PANELS.items():
        fn = globals().get(f"compute_{key}")
        if fn is None or "data" not in p or (a.only and key not in a.only):
            continue
        df = fn()
        df.to_csv(out / "data" / p["data"], index=False)
        allmeta[p["id"]] = META.get(key, {})
        print(p["id"], key, len(df), json.dumps(META.get(key), default=str)[:1500], flush=True)
    base()
    allmeta["_shared"] = META["_shared"]
    json.dump(allmeta, open(mf, "w"), indent=1, default=str)
    print("shared", META["_shared"])


if __name__ == "__main__":
    main()
