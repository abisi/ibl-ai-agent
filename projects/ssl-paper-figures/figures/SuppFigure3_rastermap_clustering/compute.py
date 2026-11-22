"""SuppFigure3_rastermap_clustering -- one compute_<key>() per panel, each writing data/<csv> of the dated snapshot
(user 2026-11-19: remaining metaclusters, metaclustering details, GMM families recover the metaclusters (S4),
cluster position vs latency / hierarchy).

Main rastermap run: rastermap_variants/<VARIANT> (fit on odd trials, 100 clusters; clusters numbered 0..99 in
rastermap order); metaclusters from population_matrix_summary.metaclustering (contiguous, 1 - r cut 0.40).
Panel a is the 045_metacluster_description.py figure (--order 17 15 16 9 20 21 --name supp_metaclusters), placed as
an image. GMM labels: ssl-rastermap-psth-variants/046_bgmm_main_variant.py (spectral embedding) and
047_bgmm_fixed_k.py (K = 100, maximum likelihood on that embedding).

Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=~/code/NWB_reader:. .venv/bin/python \
  ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure3_rastermap_clustering/compute.py [--only key ...]
then build.py figures/SuppFigure3_rastermap_clustering.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.colors
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
from common import CACHE_DIR, load_spec, snapshot_dir  # noqa: E402
from rastermap_psth import population_matrix_summary as pms  # noqa: E402

SPEC = load_spec(FIG_DIR)
PANELS = {p.get("key", p["id"]): p for p in SPEC["panels"]}
RES = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
VARIANT = "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin"
MD = sorted((RES / "rastermap_variants" / VARIANT).glob("rastermap_clustering/*/*/clustering/*/rastermap"))[0]
BGMM = MD.parent / "gmm_bayesian_main" / "bgmm_labels.csv"
ZETA = RES / "ssl-zeta-responses" / "zeta_long.parquet"
KEYS = ["mouse_id", "session_id", "electrode_group", "cluster_id"]
MIN_NEURONS_MOUSE = 3   # as 045
N_PERM = 200
FLOW_EDGE_TRIM = 1     # as run_clustering_new (fig8_flow_umap, edge_trim=1)
MIN_AREA_NEURONS = 300  # areas shown in the specialization panel
N_PERM_AMI_FACTOR = 20   # AMI is chance-corrected; the null only checks it is ~0
N_MOUSE_SPLITS = 20     # random halves of the mice
MIN_SPLIT_NEURONS = 10
CUTS = np.round(np.arange(0.05, 0.951, 0.025), 3)
rng = np.random.default_rng(0)
META = {}
_cache = {}


def base():
    """rastermap labels, metaclusters, cluster means (full trials: the metaclustering input)"""
    if not _cache:
        R = pms.load_rastermap(MD)
        meta = pms.metaclustering(MD)
        clusters, means = pms._cluster_means(MD)
        names, name_tab = pms.family_names_psth(R, meta["families"])
        lab = pd.read_csv(MD / "neuron_cluster_labels_cv.csv")
        assert np.array_equal(lab.unit_ids.to_numpy(), R["unit_ids"])
        lab["cluster_id"] = lab.cluster_id.astype(str)
        lab["metacluster"] = [meta["families"][k] for k in R["labels"]]
        _cache.update(R=R, meta=meta, clusters=clusters, means=means, name_tab=name_tab, lab=lab)
    return _cache


def r_matrix(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    B /= np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def group_means(X, labels, ks):
    return np.vstack([X[labels == k].mean(0) for k in ks])


# ---------------------------------------------------------------- panels

UMAP_PARAMS = dict(n_neighbors=15, min_dist=0.0, metric="euclidean", random_state=42)


def _umap():
    """neuron UMAP on the mean of the odd- and even-trial PSTHs (raw, no PCA), min_dist 0 (user 2026-11-20: the run's
    UMAP, even trials only with min_dist 0.11, was blobby; a parameter screen on haas showed PCA or z-scoring merge the
    response arms, while averaging the two halves and min_dist 0 separate them; screen: rastermap_variants/
    umap_param_test*.npz, variant J). Cached: ~75 s for 47k neurons. Random draw order."""
    if "umap" in _cache:
        return _cache["umap"]
    b = base()
    R = b["R"]
    cache = CACHE_DIR / f"suppfig3_umap_{VARIANT}_mean_nn15_md0.npz"
    if cache.exists() and np.array_equal(np.load(cache, allow_pickle=True)["unit_ids"], R["unit_ids"]):
        emb = np.load(cache, allow_pickle=True)["emb"]
    else:
        import umap
        Xm = (np.nan_to_num(R["X"]) + np.nan_to_num(R["X_odd"])) / 2
        emb = umap.UMAP(n_components=2, **UMAP_PARAMS).fit_transform(Xm)
        np.savez(cache, emb=emb, unit_ids=R["unit_ids"], params=json.dumps(UMAP_PARAMS))
    o = rng.permutation(len(emb))
    _cache["umap"] = (b, R, emb, o)
    return _cache["umap"]


def compute_umap_clusters():
    b, R, emb, o = _umap()
    META["umap_clusters"] = dict(n_neurons=int(len(emb)), input="mean of odd- and even-trial PSTHs", **UMAP_PARAMS)
    return pd.DataFrame(dict(u1=emb[o, 0], u2=emb[o, 1], cluster=R["labels"][o],
                             metacluster=[b["meta"]["families"][k] for k in R["labels"][o]]))


def compute_umap_areas():
    """as 045: major area groups with their allen_utils colours, minor groups pooled in Other"""
    b, R, emb, o = _umap()
    major, acol = pms._area_categories(R)
    acol = dict(acol, Other=pms.OTHER_AREA_COLOR)
    area = np.where(np.isin(R["area_group_arr"], major), R["area_group_arr"], "Other")
    META["umap_areas"] = dict(n_groups=len(major), n_neurons=int(len(emb)))
    return pd.DataFrame(dict(u1=emb[o, 0], u2=emb[o, 1], area_group=area[o],
                             color=[matplotlib.colors.to_hex(acol[a]) for a in area[o]]))


def _flow():
    """population state per time bin (rows = bins of the held-out even-trial PSTHs, columns = neurons) projected on
    its first two PCs; as run_clustering_new.fig8_flow_umap (PCA rows): stimulus-aligned conditions (lick-aligned
    excluded), edge_trim = 1 bin per condition end, activity-weighted rastermap cluster index per bin"""
    if "flow" in _cache:
        return _cache["flow"]
    b = base()
    R = b["R"]
    X = R["X"]
    labels_c, colors_c = list(pms.rmu.COND_LABELS), list(pms.rmu.COND_COLORS)
    off = np.r_[0, np.cumsum(R["n_bins_list"])]
    rows, cols = [], []
    for c, lab in enumerate(labels_c):
        if "passive" in lab.lower() or "lick" in lab.lower():
            continue
        lo, hi = FLOW_EDGE_TRIM, int(R["n_bins_list"][c]) - FLOW_EDGE_TRIM
        t = np.asarray(R["t_ctrs"][c])[lo:hi]
        for i, tt in enumerate(t):
            rows.append(dict(condition=lab, color=colors_c[c], cond_index=c, t=float(tt),
                             frac=i / max(1, len(t) - 1)))
        cols.extend(range(off[c] + lo, off[c] + hi))
    Xb = np.nan_to_num(X[:, cols].T)
    pca = PCA(n_components=2).fit(Xb)
    pc = pca.transform(Xb)
    K = int(R["labels"].max()) + 1
    cm = np.stack([Xb[:, R["labels"] == k].mean(1) for k in range(K)], 1)
    w = np.clip(cm, 0, None)
    ws = w.sum(1)
    wci = np.where(ws > 0, (w @ np.arange(K)) / np.where(ws > 0, ws, 1), (K - 1) / 2)
    var = pca.explained_variance_ratio_
    d = pd.DataFrame(rows).assign(pc1=pc[:, 0], pc2=pc[:, 1], weighted_cluster=wci, pc1_var=var[0], pc2_var=var[1])
    _cache["flow"] = (d, pca.explained_variance_ratio_)
    return _cache["flow"]


def compute_pca_flow():
    d, var = _flow()
    META["pca_flow"] = dict(pc1_var=float(var[0]), pc2_var=float(var[1]), n_conditions=int(d.condition.nunique()),
                            edge_trim=FLOW_EDGE_TRIM, n_bins=int(len(d)))
    return d


def compute_pca_flow_clusters():
    d, var = _flow()
    META["pca_flow_clusters"] = dict(pc1_var=float(var[0]), pc2_var=float(var[1]))
    return d


def compute_meta_matrix():
    b = base()
    C = r_matrix(b["means"].copy(), b["means"].copy())
    fam = np.array([b["meta"]["families"][k] for k in b["clusters"]])
    K = len(C)
    rows = [dict(kind="r", i=i, j=j, value=C[i, j], fam_i=fam[i], fam_j=fam[j]) for i in range(K) for j in range(K)]
    rows += [dict(kind="Z", i=int(z[0]), j=int(z[1]), value=z[2], n=int(z[3])) for z in b["meta"]["Z"]]
    off = ~np.eye(K, dtype=bool)
    same = (fam[:, None] == fam[None, :]) & off
    META["meta_matrix"] = dict(n_clusters=K, n_metaclusters=int(fam.max()), cut=b["meta"]["cut"],
                               r_within_median=float(np.median(C[same])),
                               r_between_median=float(np.median(C[off & ~same])))
    return pd.DataFrame(rows)


def compute_meta_cut():
    b = base()
    n = [int(pms.contiguous_metaclustering(b["means"], c)[0].max()) for c in CUTS]
    META["meta_cut"] = dict(cut=b["meta"]["cut"], n_at_cut=int(max(b["meta"]["families"].values())))
    return pd.DataFrame(dict(cut=CUTS, n_metaclusters=n))


def compute_meta_sizes():
    b = base()
    lab, nt = b["lab"], b["name_tab"].rename(columns={"family": "metacluster"})
    g = lab.groupby("metacluster")
    mice = g.apply(lambda x: int((x.mouse_id.value_counts() >= MIN_NEURONS_MOUSE).sum()))
    d = pd.DataFrame(dict(metacluster=g.size().index, n_neurons=g.size().values, n_mice=mice.values,
                          n_clusters=g.cluster_label.nunique().values))
    d = d.merge(nt[["metacluster", "name", "kind", "modality"]], on="metacluster", how="left")
    META["meta_sizes"] = dict(n_mice_total=int(lab.mouse_id.nunique()), n_neurons=int(len(lab)))
    return d


def compute_cv():
    """per fine cluster: r(mean PSTH on odd trials, mean PSTH on even trials) of the same neurons; best other cluster"""
    b = base()
    R, ks = b["R"], b["clusters"]
    odd, even = group_means(R["X_odd"], R["labels"], ks), group_means(R["X"], R["labels"], ks)
    C = r_matrix(odd, even)
    self_r = np.diag(C).copy()
    np.fill_diagonal(C, -np.inf)
    d = pd.DataFrame(dict(cluster=ks, r_self=self_r, r_best_other=C.max(1),
                          n_neurons=[(R["labels"] == k).sum() for k in ks],
                          metacluster=[b["meta"]["families"][k] for k in ks]))
    META["cv"] = dict(n_clusters=len(ks), r_self_median=float(np.median(self_r)),
                      frac_self_beats_others=float((d.r_self > d.r_best_other).mean()))
    return d


def _bgmm(K):
    """labels of the fixed-K Bayesian GMM (047), checked against the rastermap labels"""
    b = base()
    g = pd.read_csv(BGMM.parent / f"bgmm_k{K}_labels.csv")
    assert np.array_equal(g.unit_ids.to_numpy(), b["lab"].unit_ids.to_numpy())
    assert np.array_equal(g.rastermap_cluster.to_numpy(), b["R"]["labels"])
    return g.bgmm_component.to_numpy()


def _means(X, labels):
    """mean PSTH per label (one-hot product), labels sorted"""
    ks, inv = np.unique(labels, return_inverse=True)
    M = np.zeros((len(ks), len(labels)), np.float32)
    M[inv, np.arange(len(labels))] = 1
    return ks, (M @ X) / M.sum(1, keepdims=True)


def _xd():
    if "Xd" not in _cache:
        X = base()["R"]["X"].astype(np.float32)          # even (held-out) trials
        _cache["Xd"] = X                                 # raw mean PSTHs, population mean kept (user 2026-11-21)
    return _cache["Xd"]


def _xfull():
    """full-trial neuron PSTHs (the metaclustering input of pms._cluster_means), rows in R["unit_ids"] order"""
    if "Xfull" not in _cache:
        cpa = pms.cpa
        cdf = cpa.load_cluster_table(MD)
        uids, conds, psth, _ = cpa.load_psth_npz(MD)
        X, sub, _ = cpa._build_neuron_matrix(cdf, conds, psth, uids)
        pos = pd.Series(np.arange(len(sub)), sub.unit_ids.to_numpy()).reindex(base()["R"]["unit_ids"]).to_numpy()
        assert np.isfinite(pos).all(), "full-trial PSTHs missing for some neurons"
        _cache["Xfull"] = X[pos.astype(int)].astype(np.float32)
    return _cache["Xfull"]


def _gmm_families():
    """metaclustering applied to the 100-component GMM (047, spectral embedding): components smaller than the smallest
    rastermap cluster dropped (their neurons excluded), component-mean PSTHs (full trials, as for rastermap) grouped
    by the same average linkage on 1 - Pearson r but WITHOUT the contiguity constraint (GMM components have no order),
    cut to the number of rastermap metaclusters. Families are matched one-to-one (Hungarian) to the metaclusters on
    the Pearson r of family-mean PSTHs (average of the neurons' held-out even-trial PSTHs)."""
    if "gfam" in _cache:
        return _cache["gfam"]
    from sklearn.cluster import AgglomerativeClustering
    b = base()
    rl = b["R"]["labels"]
    ml = np.array([b["meta"]["families"][k] for k in rl])
    n_meta = int(ml.max())
    min_size = int(np.bincount(rl).min())
    gl = _bgmm(100)
    sz = pd.Series(gl).value_counts()
    keep = np.isin(gl, sz.index[sz >= min_size])
    kc, Mc = _means(_xfull()[keep], gl[keep])
    C = Mc - Mc.mean(1, keepdims=True)
    fam = AgglomerativeClustering(n_clusters=n_meta, metric="cosine", linkage="average").fit(C).labels_
    gf = np.full(len(gl), -1)
    gf[keep] = pd.Series(fam, kc).reindex(gl[keep]).to_numpy()
    ka, Ma = _means(_xd(), ml)
    kf, Mf = _means(_xd()[keep], gf[keep])
    R = r_matrix(Ma.copy(), Mf.copy())
    i, j = linear_sum_assignment(-R)
    _cache["gfam"] = dict(ml=ml, gf=gf, keep=keep, ka=ka[i], kf=kf[j], r=R[i, j], Ma=Ma[i], Mf=Mf[j],
                          min_size=min_size, n_dropped=int((sz < min_size).sum()), n_neurons_dropped=int((~keep).sum()),
                          n_components=int(len(sz)), n_meta=n_meta)
    return _cache["gfam"]


def _match_table():
    g = _gmm_families()
    nt = base()["name_tab"].rename(columns={"family": "metacluster"})
    nm, nf = pd.Series(g["ml"]).value_counts(), pd.Series(g["gf"][g["keep"]]).value_counts()
    d = pd.DataFrame(dict(metacluster=g["ka"], gmm_family=g["kf"], r=g["r"], n_neurons=nm.reindex(g["ka"]).to_numpy(),
                          n_neurons_gmm_family=nf.reindex(g["kf"]).to_numpy()))
    return d.merge(nt[["metacluster", "name", "kind", "modality"]], on="metacluster", how="left")


def compute_gmm_family_match():
    """per rastermap metacluster: r with its matched GMM family (family-mean PSTHs, held-out trials) vs its size"""
    g, d = _gmm_families(), _match_table()
    w = d.n_neurons / d.n_neurons.sum()
    META["gmm_family_match"] = dict(
        n_meta=g["n_meta"], n_components=g["n_components"], min_size=g["min_size"], n_dropped=g["n_dropped"],
        n_neurons_dropped=g["n_neurons_dropped"], r_median=float(d.r.median()), r_weighted=float((w * d.r).sum()),
        n_r_gt_0p8=int((d.r > 0.8).sum()), frac_neurons_r_gt_0p8=float(w[d.r > 0.8].sum()),
        min_size_all_matched=int(d.n_neurons[d.r <= 0.8].max() + 1) if (d.r <= 0.8).any() else 0)
    return d.sort_values("metacluster")


def compute_gmm_family_psth():
    """family-mean PSTHs (held-out even trials, concatenated conditions) of each metacluster and its matched GMM
    family, rows in metacluster (rastermap) order"""
    g, d = _gmm_families(), _match_table().set_index("metacluster")
    R = base()["R"]
    off = np.r_[0, np.cumsum(R["n_bins_list"])]
    cond = np.repeat(np.arange(len(R["n_bins_list"])), R["n_bins_list"])
    rows = []
    for a, Ma, Mf in sorted(zip(g["ka"], g["Ma"], g["Mf"]), key=lambda t: t[0]):
        for method, M in (("rastermap", Ma), ("gmm", Mf)):
            rows.append(pd.DataFrame(dict(metacluster=a, method=method, col=np.arange(len(M)), cond=cond, value=M,
                                          r=d.loc[a, "r"], n_neurons=d.loc[a, "n_neurons"], kind=d.loc[a, "kind"],
                                          modality=d.loc[a, "modality"])))
    META["gmm_family_psth"] = dict(cond_labels=list(pms.rmu.COND_LABELS), cond_edges=off.tolist(),
                                   **{k: g[k] for k in ("n_meta", "n_components", "min_size", "n_dropped",
                                                        "n_neurons_dropped")})
    return pd.concat(rows, ignore_index=True)


def _zeta(types=("whisker_active", "auditory_active")):
    z = pd.read_parquet(ZETA, columns=KEYS + ["analysis_type", "significant", "direction", "latency_s"],
                        filters=[("analysis_type", "in", list(types))])
    z["cluster_id"] = z.cluster_id.astype(str)
    return base()["lab"][KEYS + ["cluster_label", "metacluster"]].merge(z, on=KEYS)


def _perm_rho(values, groups, keep, agg, n=N_PERM):
    """Spearman rho of cluster position vs cluster aggregate after shuffling the neuron values across clusters"""
    out = []
    for _ in range(n):
        v = pd.Series(rng.permutation(values)).groupby(groups).agg(agg).reindex(keep)
        out.append(spearmanr(keep, v)[0])
    return np.array(out)


def _p(null, rho):
    return float((1 + np.sum(np.abs(null) >= abs(rho))) / (1 + len(null)))


def compute_latency():
    """per fine cluster: median latenZy onset of its ZETA-positive neurons (whisker / auditory active trials);
    Spearman rho, permutation p (onsets shuffled across neurons), least-squares line"""
    m = _zeta()
    pos = m[m.significant & (m.direction == "positive")].dropna(subset=["latency_s"])
    rows, meta = [], {}
    for at, x in pos.groupby("analysis_type"):
        d = (x.groupby("cluster_label").agg(latency_ms=("latency_s", lambda v: 1000 * v.median()),
                                            n=("latency_s", "size"), metacluster=("metacluster", "first"))
             .reset_index().rename(columns={"cluster_label": "cluster"}))
        d = d[d.n >= 10]
        rho, p = spearmanr(d.cluster, d.latency_ms)
        null = _perm_rho(1000 * x.latency_s.to_numpy(), x.cluster_label.to_numpy(), d.cluster.to_numpy(), "median")
        slope, icpt = np.polyfit(d.cluster, d.latency_ms, 1)
        meta[at] = dict(rho=float(rho), p=float(p), p_perm=_p(null, rho), slope_ms_per_cluster=float(slope),
                        intercept=float(icpt), n_clusters=int(len(d)))
        rows.append(d.assign(analysis_type=at, fit=icpt + slope * d.cluster, rho=rho, p_perm=meta[at]["p_perm"]))
    META["latency"] = {**meta, "n_perm": N_PERM}
    return pd.concat(rows, ignore_index=True)


def compute_hierarchy():
    """per fine cluster: mean Harris et al. (2019) hierarchy score of its neurons with a score"""
    d, meta = _anatomy_vs_order(pms.ANATOMY_COLS[1])
    META["hierarchy"] = meta
    return d


def compute_liu_projection():
    """as compute_hierarchy, for the wS1/wS2 projection strength to the neuron's area (Liu et al., 2024; avg_ipsi),
    log10 (heavy-tailed: 0.004 to 1006, median 225) -- user 2026-11-20, panel next to the hierarchy panel"""
    d, meta = _anatomy_vs_order(pms.ANATOMY_COLS[0], log10=True)
    META["liu_projection"] = meta
    return d


def _anatomy_vs_order(column, log10=False):
    """per fine cluster: mean anatomy score (column of R['anatomy']) of its neurons with a score; clusters split into
    whisker- and auditory-dominated (larger fraction of ZETA-positive neurons in active whisker vs auditory trials);
    Spearman rho per group, permutation p (scores shuffled across neurons of that group's clusters), least-squares line"""
    b = base()
    R = b["R"]
    h = R["anatomy"][column].astype(float)
    if log10:
        h = np.where(h > 0, np.log10(np.where(h > 0, h, 1)), np.nan)
    ok = np.isfinite(h)
    hl, ll = h[ok], R["labels"][ok]
    d = (pd.DataFrame(dict(cluster=ll, h=hl)).groupby("cluster").h.agg(["mean", "sem", "size"])
         .reset_index().rename(columns={"mean": "hierarchy", "size": "n"}))
    d = d[d.n >= 20].reset_index(drop=True)
    z = _zeta()
    z["pos"] = z.significant & (z.direction == "positive")
    fr = z.pivot_table(index="cluster_label", columns="analysis_type", values="pos", aggfunc="mean")
    d["frac_whisker"] = d.cluster.map(fr["whisker_active"])
    d["frac_auditory"] = d.cluster.map(fr["auditory_active"])
    d["dominant"] = np.where(d.frac_whisker > d.frac_auditory, "whisker", "auditory")
    d["metacluster"] = d.cluster.map(b["meta"]["families"])
    meta = dict(n_neurons=int(ok.sum()), column=column, log10=log10, n_perm=N_PERM)
    rho, _ = spearmanr(d.cluster, d.hierarchy)
    meta["all"] = dict(rho=float(rho), p_perm=_p(_perm_rho(hl, ll, d.cluster.to_numpy(), "mean"), rho),
                       n_clusters=int(len(d)))
    d["fit"] = np.nan
    for g, x in d.groupby("dominant"):
        rho, p = spearmanr(x.cluster, x.hierarchy)
        keep = x.cluster.to_numpy()
        sel = np.isin(ll, keep)
        slope, icpt = np.polyfit(x.cluster, x.hierarchy, 1)
        meta[g] = dict(rho=float(rho), p=float(p), p_perm=_p(_perm_rho(hl[sel], ll[sel], keep, "mean"), rho),
                       n_clusters=int(len(x)), slope=float(slope))
        d.loc[x.index, "fit"] = icpt + slope * x.cluster
        d.loc[x.index, "rho"] = rho
        d.loc[x.index, "p_perm"] = meta[g]["p_perm"]
    return d, meta


def _kl_bits(counts, p0):
    """KL divergence (bits) of each row's distribution from p0; counts: (n_rows, K)"""
    p = counts / counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log2(p / p0), 0.0)
    return t.sum(1)


def compute_area_specialization():
    """per area (>= MIN_AREA_NEURONS neurons): distribution of its neurons over the metaclusters, log2 enrichment
    over the dataset distribution, specialization = KL divergence (bits) from the dataset distribution; null: area
    labels shuffled across all neurons of the dataset (user 2026-11-19)"""
    b = base()
    R, lab = b["R"], b["lab"]
    meta = lab.metacluster.to_numpy()
    area = R["area_arr"]
    cnt = pd.Series(area).value_counts()
    areas = cnt.index[cnt >= MIN_AREA_NEURONS].to_numpy()
    keep = np.isin(area, areas)
    ks = np.sort(np.unique(meta))
    p0 = np.bincount(np.searchsorted(ks, meta), minlength=len(ks)) / len(meta)
    ai = {a: i for i, a in enumerate(areas)}

    def table(ar):
        m = keep & np.isin(ar, areas)
        idx = np.array([ai[a] for a in ar[m]]) * len(ks) + np.searchsorted(ks, meta[m])
        return np.bincount(idx, minlength=len(areas) * len(ks)).reshape(len(areas), len(ks)).astype(float)

    C = table(area)
    kl = _kl_bits(C, p0)
    null = [_kl_bits(table(rng.permutation(area)), p0) for _ in range(N_PERM)]
    null = np.array(null)
    grp = pd.Series(R["area_group_arr"]).groupby(area).first()
    major, acol = pms._area_categories(R)
    acol = dict(acol, Other=pms.OTHER_AREA_COLOR)
    nt = b["name_tab"].set_index("family")
    rows = []
    for i, a in enumerate(areas):
        frac = C[i] / C[i].sum()
        g = grp[a]
        for j, k in enumerate(ks):
            rows.append(dict(area=a, area_group=g, color=matplotlib.colors.to_hex(acol.get(g, acol["Other"])),
                             n_area=int(C[i].sum()), metacluster=int(k), kind=nt.kind.get(k), modality=nt.modality.get(k),
                             n=int(C[i, j]), frac=frac[j], frac_dataset=p0[j],
                             log2_enrichment=np.log2((C[i, j] + 0.5) / (C[i].sum() * p0[j] + 0.5)),
                             kl_bits=kl[i], null_mean=null[:, i].mean(), null_p95=np.percentile(null[:, i], 95),
                             p_perm=(1 + (null[:, i] >= kl[i]).sum()) / (1 + N_PERM)))
    d = pd.DataFrame(rows)
    q = d.drop_duplicates("area")
    META["area_specialization"] = dict(n_areas=int(len(areas)), min_neurons=MIN_AREA_NEURONS,
                                       n_neurons=int(keep.sum()), n_sig=int((q.p_perm < 0.05).sum()),
                                       kl_median=float(q.kl_bits.median()), null_median=float(q.null_mean.median()),
                                       n_perm=N_PERM)
    return d


def compute_area_ami():
    """adjusted mutual information between cluster labels (fine / metaclusters) and area, area group, mouse (= session:
    one session per mouse here) and reward group; null: factor labels permuted across neurons"""
    b = base()
    R, lab = b["R"], b["lab"]
    factors = dict(area=R["area_arr"], area_group=R["area_group_arr"], mouse=R["mouse_arr"],   # 1 session per mouse
                   reward_group=lab.reward_group.astype(str).to_numpy())
    rows = []
    for level, y in (("fine clusters", R["labels"]), ("metaclusters", lab.metacluster.to_numpy())):
        for f, x in factors.items():
            real = adjusted_mutual_info_score(y, x)
            null = [adjusted_mutual_info_score(y, rng.permutation(x)) for _ in range(N_PERM_AMI_FACTOR)]
            rows.append(dict(level=level, factor=f, n_labels=int(len(np.unique(x))), AMI=real,
                             null_p95=np.percentile(null, 95),
                             p_perm=(1 + np.sum(np.array(null) >= real)) / (1 + len(null))))
    d = pd.DataFrame(rows)
    META["area_ami"] = {f"{r.level.split()[0]}_{r.factor}": float(r.AMI) for r in d.itertuples()}
    META["area_ami"]["n_perm"] = N_PERM_AMI_FACTOR
    return d


ROBUST = [   # (label, reference variant, compared variant); "" = the main variant; "<variant>:all" = fit on all trials
    # (no all-trials fit in the main variant: its neuron_cluster_labels.csv is all 0, rastermap run in CV mode only)
    ("Good units only", "", "qc_good__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin"),
    ("All learner categories", "", "qc_good_mua__learncat_all__loc0p25_grid10_nodes__lickfix_longwin"),
    ("No passive conditions", "", "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin__nopassive"),
    ("No whisker-reward bins", "", "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes__lickfix_longwin__noWhiskerReward"),
    ("Uncorrected lick times", "", "qc_good_mua__learncat_good_moderate__loc0p25_grid10_nodes"),
    ("No firing-rate threshold", "qc_good__learncat_good_moderate", "qc_good__learncat_good_moderate__noFR"),
    ("Locality 0 vs 0.25", "qc_good__learncat_good_moderate__loc0p25", "qc_good__learncat_good_moderate__loc0p00"),
    ("Locality 0.75 vs 0.25", "qc_good__learncat_good_moderate__loc0p25", "qc_good__learncat_good_moderate__loc0p75"),
]


def _variant_labels(v):
    v, _, mode = (v or VARIANT).partition(":")
    v = v or VARIANT
    name = "neuron_cluster_labels.csv" if mode == "all" else "neuron_cluster_labels_cv.csv"   # all trials vs CV (odd) fit
    f = sorted((RES / "rastermap_variants" / v).glob(f"rastermap_clustering/*/*/clustering/*/rastermap/{name}"))[0]
    d = pd.read_csv(f, usecols=KEYS + ["cluster_label"])
    d["cluster_id"] = d.cluster_id.astype(str)
    return d


def compute_robustness():
    """agreement of the rastermap clusters between analysis variants, on the neurons present in both (unit keys):
    AMI of the 100 fine-cluster labels and Spearman correlation of the cluster position (rastermap order)"""
    rows = []
    for label, ref, other in ROBUST:
        m = _variant_labels(ref).merge(_variant_labels(other), on=KEYS, suffixes=("_a", "_b"))
        rows.append(dict(comparison=label, reference=ref or VARIANT, variant=other, n_common=int(len(m)),
                         AMI=adjusted_mutual_info_score(m.cluster_label_a, m.cluster_label_b),
                         ARI=adjusted_rand_score(m.cluster_label_a, m.cluster_label_b),
                         spearman_position=spearmanr(m.cluster_label_a, m.cluster_label_b)[0]))
    d = pd.DataFrame(rows)
    gl = _bgmm(100)
    META["robustness"] = dict(ami_min=float(d.AMI.min()), ami_max=float(d.AMI.max()),
                              rho_min=float(d.spearman_position.min()),
                              gmm_ami=float(adjusted_mutual_info_score(base()["R"]["labels"], gl)))
    return d.assign(gmm_ami=META["robustness"]["gmm_ami"])


def compute_mouse_split():
    """per fine cluster: its mean PSTH (held-out even trials) from one random half of the mice vs the other half (r_self)
    and vs the most similar other cluster's other-half mean (r_best_other); median over N_MOUSE_SPLITS splits; clusters
    with < MIN_SPLIT_NEURONS neurons in a half skipped in that split"""
    b = base()
    R = b["R"]
    X, lab, mouse = R["X"], R["labels"], R["mouse_arr"]
    mice = np.unique(mouse)
    ks = np.arange(int(lab.max()) + 1)
    rs, ro = [], []
    for _ in range(N_MOUSE_SPLITS):
        half = np.isin(mouse, rng.permutation(mice)[: len(mice) // 2])
        ok = np.array([min(((lab == k) & half).sum(), ((lab == k) & ~half).sum()) >= MIN_SPLIT_NEURONS for k in ks])
        A = group_means(X[half], lab[half], ks[ok])
        B = group_means(X[~half], lab[~half], ks[ok])
        C = r_matrix(A, B)
        s = np.full(len(ks), np.nan); o = np.full(len(ks), np.nan)
        s[ok] = np.diag(C)
        np.fill_diagonal(C, -np.inf)
        o[ok] = C.max(1)
        rs.append(s); ro.append(o)
    d = pd.DataFrame(dict(cluster=ks, r_self=np.nanmedian(rs, 0), r_best_other=np.nanmedian(ro, 0),
                          n_splits=np.isfinite(rs).sum(0))).dropna()
    META["mouse_split"] = dict(n_clusters=int(len(d)), n_splits=N_MOUSE_SPLITS, r_self_median=float(d.r_self.median()),
                               frac_self_beats_others=float((d.r_self > d.r_best_other).mean()),
                               min_neurons=MIN_SPLIT_NEURONS)
    return d


def compute_mouse_share():
    """same table as mouse_mixing (panel with the largest single-mouse share)"""
    d = compute_mouse_mixing()
    META["mouse_share"] = META["mouse_mixing"]
    return d


def compute_mouse_mixing():
    """per fine cluster: effective number of contributing mice (exp of the Shannon entropy of its neurons' mouse
    distribution) and the largest single-mouse share; null: cluster labels permuted across neurons (same sizes)"""
    b = base()
    R = b["R"]
    lab, mouse = R["labels"], R["mouse_arr"]
    ks = np.arange(int(lab.max()) + 1)
    mi = np.unique(mouse, return_inverse=True)[1]
    nm = mi.max() + 1

    def stats(l):
        C = np.bincount(l * nm + mi, minlength=len(ks) * nm).reshape(len(ks), nm).astype(float)
        p = C / C.sum(1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            H = -np.where(p > 0, p * np.log(p), 0).sum(1)
        return np.exp(H), p.max(1), (C >= 1).sum(1)

    if "mouse_mixing" in _cache:
        return _cache["mouse_mixing"]
    eff, mx, nmice = stats(lab)
    null = [stats(rng.permutation(lab)) for _ in range(N_PERM)]
    ne = np.array([n[0] for n in null]); nx = np.array([n[1] for n in null])
    # within-area null: cluster labels permuted among the neurons of each area (keeps every area's cluster mix and
    # every mouse's area mix): is the mouse concentration more than the areas each mouse was recorded in?
    by_area = pd.Series(np.arange(len(lab))).groupby(R["area_arr"]).indices.values()
    na = []
    for _ in range(N_PERM):
        sh = lab.copy()
        for ix in by_area:
            sh[ix] = lab[rng.permutation(ix)]
        na.append(stats(sh)[0])
    na = np.array(na)
    d = pd.DataFrame(dict(cluster=ks, n_neurons=np.bincount(lab, minlength=len(ks)), eff_mice=eff, max_share=mx,
                          n_mice=nmice, null_eff_mean=ne.mean(0), null_eff_p5=np.percentile(ne, 5, 0),
                          null_eff_p95=np.percentile(ne, 95, 0), null_max_p95=np.percentile(nx, 95, 0),
                          area_null_eff_mean=na.mean(0), area_null_eff_p5=np.percentile(na, 5, 0),
                          area_null_eff_p95=np.percentile(na, 95, 0),
                          metacluster=[b["meta"]["families"][k] for k in ks]))
    META["mouse_mixing"] = dict(eff_median=float(d.eff_mice.median()), null_eff_median=float(d.null_eff_mean.median()),
                                n_mice_total=int(nm), max_share_median=float(d.max_share.median()),
                                n_max_share_above_half=int((d.max_share > 0.5).sum()),
                                frac_eff_below_null_p5=float((d.eff_mice < d.null_eff_p5).mean()), n_perm=N_PERM,
                                area_null_eff_median=float(d.area_null_eff_mean.median()),
                                frac_eff_below_area_null_p5=float((d.eff_mice < d.area_null_eff_p5).mean()))
    _cache["mouse_mixing"] = d
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
        if key.startswith("gmm") and not (BGMM.parent / "bgmm_k100_labels.csv").exists():
            print("GMM labels not ready, skipped", key)
            continue
        df = fn()
        df.to_csv(out / "data" / p["data"], index=False)
        allmeta[p["id"]] = META.get(key, {})
        print(p["id"], key, len(df), META.get(key), flush=True)
    allmeta["_shared"] = dict(variant=VARIANT, method_dir=str(MD))
    json.dump(allmeta, open(mf, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
