"""SuppFigure3_rastermap_clustering -- one compute_<key>() per panel, each writing data/<csv> of the dated snapshot
(user 2026-11-19: remaining metaclusters, metaclustering details, Bayesian GMM recovers the rastermap clusters,
cluster position vs latency / hierarchy).

Main rastermap run: rastermap_variants/<VARIANT> (fit on odd trials, 100 clusters; clusters numbered 0..99 in
rastermap order); metaclusters from population_matrix_summary.metaclustering (contiguous, 1 - r cut 0.40).
Panel a is the 045_metacluster_description.py figure (--order 17 15 16 9 20 21 --name supp_metaclusters), placed as
an image. GMM labels: ssl-rastermap-psth-variants/046_bgmm_main_variant.py (embedding, Dirichlet-process run) and
047_bgmm_fixed_k.py (K = 100 and K = n metaclusters, maximum likelihood on the same embedding).

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
from common import load_spec, snapshot_dir  # noqa: E402
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
N_PERM_AGREE = 50     # AMI is slow at K = 100
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

def _umap():
    """neuron UMAP of the run (fitted on the held-out even-trial PSTHs, rastermap_results_cv.npz), random draw order"""
    b = base()
    R = b["R"]
    d = np.load(MD / "rastermap_results_cv.npz", allow_pickle=True)
    assert np.array_equal(d["unit_ids"], R["unit_ids"])
    emb = d["umap_embedding"]
    o = rng.permutation(len(emb))
    return b, R, emb, o


def compute_umap_clusters():
    b, R, emb, o = _umap()
    META["umap_clusters"] = dict(n_neurons=int(len(emb)))
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
    d = pd.DataFrame(rows).assign(pc1=pc[:, 0], pc2=pc[:, 1], weighted_cluster=wci)
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


def _levels():
    """(level, K, rastermap labels, BGMM labels): fine clusters vs K = 100, metaclusters vs K = n metaclusters"""
    b = base()
    rl = b["R"]["labels"]
    fam = np.array([b["meta"]["families"][k] for k in rl])
    return [("fine clusters", 100, rl, _bgmm(100)), ("metaclusters", int(fam.max()), fam, _bgmm(int(fam.max())))]


def _means(X, labels):
    """mean PSTH per label (one-hot product), labels sorted"""
    ks, inv = np.unique(labels, return_inverse=True)
    M = np.zeros((len(ks), len(labels)), np.float32)
    M[inv, np.arange(len(labels))] = 1
    return ks, (M @ X) / M.sum(1, keepdims=True)


def _match(Xd, a, c):
    """one-to-one (Hungarian) matching of rastermap groups a and BGMM components c on the Pearson r of their mean
    PSTHs (Xd: held-out trials, population mean subtracted); returns matched a keys, c keys, r per pair"""
    ka, Ma = _means(Xd, a)
    kc, Mc = _means(Xd, c)
    C = r_matrix(Ma, Mc)
    i, j = linear_sum_assignment(-C)
    return ka[i], kc[j], C[i, j]


def _xd():
    if "Xd" not in _cache:
        X = base()["R"]["X"].astype(np.float32)          # even (held-out) trials
        _cache["Xd"] = X - X.mean(0, keepdims=True)      # deviation from the population-mean PSTH
    return _cache["Xd"]


def compute_bgmm_contingency():
    """fraction of each rastermap cluster's neurons in each K = 100 component; components in the order of their
    matched rastermap cluster (diagonal = one-to-one agreement)"""
    b = base()
    _, K, rl, gl = _levels()[0]
    ka, kc, _ = _match(_xd(), rl, gl)
    order = list(kc[np.argsort(ka)]) + [c for c in np.unique(gl) if c not in set(kc)]
    T = pd.crosstab(rl, gl).reindex(columns=order, fill_value=0)
    Tn = T.div(T.sum(axis=1), axis=0)
    long = Tn.stack().rename("frac_of_cluster").reset_index()
    long.columns = ["cluster", "component", "frac_of_cluster"]
    long["component_rank"] = long.component.map({c: i for i, c in enumerate(order)})
    long["metacluster"] = long.cluster.map(b["meta"]["families"])
    kmeta = _levels()[1][1]
    prov = {k: json.load(open(BGMM.parent / f"bgmm_k{k}_provenance.json")) for k in (100, kmeta)}
    dp = json.load(open(BGMM.parent / "bgmm_provenance.json"))
    META["bgmm_contingency"] = dict(n_components=int(len(order)), n_clusters=int(T.shape[0]),
                                    frac_on_diagonal=float(sum(T.loc[x, y] for x, y in zip(ka, kc)) / T.to_numpy().sum()),
                                    n_dp_components=dp["n_active"], fixed_k=prov, dp=dp)
    return long


def compute_bgmm_agreement():
    """AMI, ARI and fraction of neurons in matched pairs, rastermap vs BGMM with the same number of groups;
    null: BGMM labels permuted across neurons (matching redone)"""
    Xd = _xd()
    rows, meta = [], {}
    for level, K, a, c in _levels():
        def stats(cc):
            ka, kc, _ = _match(Xd, a, cc)
            m = dict(zip(kc, ka))
            return dict(AMI=adjusted_mutual_info_score(a, cc), ARI=adjusted_rand_score(a, cc),
                        matched=float(np.mean(pd.Series(cc).map(m).to_numpy() == a)))
        real = stats(c)
        null = pd.DataFrame([stats(rng.permutation(c)) for _ in range(N_PERM_AGREE)])
        for metric in real:
            rows.append(dict(level=level, K=K, metric=metric, value=real[metric], null_mean=null[metric].mean(),
                             null_p95=null[metric].quantile(0.95),
                             p_perm=(1 + (null[metric] >= real[metric]).sum()) / (1 + len(null))))
        meta[level.split()[0]] = {**real, "K": K}
    META["bgmm_agreement"] = dict(fine_matched=meta["fine"]["matched"], meta_matched=meta["metaclusters"]["matched"],
                                  n_perm=N_PERM_AGREE, levels=meta)
    return pd.DataFrame(rows)


def compute_bgmm_matched():
    """r between matched rastermap and BGMM mean PSTHs (held-out even trials, population mean subtracted), per pair;
    null: BGMM labels permuted, matching redone"""
    Xd = _xd()
    rows, meta = [], {}
    for level, K, a, c in _levels():
        ka, kc, r = _match(Xd, a, c)
        na, nc = pd.Series(a).value_counts(), pd.Series(c).value_counts()
        rows += [dict(level=level, kind="real", rastermap=x, bgmm=y, r=v, n_rastermap=int(na[x]), n_bgmm=int(nc[y]))
                 for x, y, v in zip(ka, kc, r)]
        med = []
        for p in range(N_PERM):
            _, _, rp = _match(Xd, a, rng.permutation(c))
            med.append(np.median(rp))
            rows += [dict(level=level, kind="perm", r=v, perm=p) for v in rp]
        meta[level.split()[0]] = dict(r_median=float(np.median(r)), null_median=float(np.median(med)),
                                      p_perm=float((1 + np.sum(np.array(med) >= np.median(r))) / (1 + N_PERM)),
                                      n_pairs=int(len(r)))
    META["bgmm_matched"] = dict(fine_r=meta["fine"]["r_median"], meta_r=meta["metaclusters"]["r_median"],
                                fine_null=meta["fine"]["null_median"], meta_null=meta["metaclusters"]["null_median"],
                                fine_p=meta["fine"]["p_perm"], meta_p=meta["metaclusters"]["p_perm"], n_perm=N_PERM)
    return pd.DataFrame(rows)


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
    """per fine cluster: mean Harris et al. (2019) hierarchy score of its neurons with a score; clusters split into
    whisker- and auditory-dominated (larger fraction of ZETA-positive neurons in active whisker vs auditory trials);
    Spearman rho per group, permutation p (scores shuffled across neurons of that group's clusters), least-squares line"""
    b = base()
    R = b["R"]
    h = R["anatomy"][pms.ANATOMY_COLS[1]]
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
    meta = dict(n_neurons=int(ok.sum()), column=pms.ANATOMY_COLS[1], n_perm=N_PERM)
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
    META["hierarchy"] = meta
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
        if key.startswith("bgmm") and not (BGMM.parent / "bgmm_k100_labels.csv").exists():
            print("BGMM labels not ready, skipped", key)
            continue
        df = fn()
        df.to_csv(out / "data" / p["data"], index=False)
        allmeta[p["id"]] = META.get(key, {})
        print(p["id"], key, len(df), META.get(key), flush=True)
    allmeta["_shared"] = dict(variant=VARIANT, method_dir=str(MD))
    json.dump(allmeta, open(mf, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
