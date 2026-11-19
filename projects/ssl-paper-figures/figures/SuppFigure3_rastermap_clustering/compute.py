"""SuppFigure3_rastermap_clustering -- one compute_<key>() per panel, each writing data/<csv> of the dated snapshot
(user 2026-11-19: remaining metaclusters, metaclustering details, Bayesian GMM recovers the rastermap clusters,
cluster position vs latency / hierarchy).

Main rastermap run: rastermap_variants/<VARIANT> (fit on odd trials, 100 clusters; clusters numbered 0..99 in
rastermap order); metaclusters from population_matrix_summary.metaclustering (contiguous, 1 - r cut 0.40).
Panel a is the 045_metacluster_description.py figure (--order 17 15 16 9 20 21 --name supp_metaclusters), placed as
an image. Bayesian GMM labels: ssl-rastermap-psth-variants/046_bgmm_main_variant.py.

Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=~/code/NWB_reader:. .venv/bin/python \
  ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure3_rastermap_clustering/compute.py [--only key ...]
then build.py figures/SuppFigure3_rastermap_clustering.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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


def _bgmm():
    b = base()
    g = pd.read_csv(BGMM)
    assert np.array_equal(g.unit_ids.to_numpy(), b["lab"].unit_ids.to_numpy())
    assert np.array_equal(g.rastermap_cluster.to_numpy(), b["R"]["labels"])
    return b, g.bgmm_component.to_numpy()


def _comp_order(rl, gl):
    """BGMM components ordered by the mean rastermap position of their neurons"""
    comps = np.unique(gl)
    return comps[np.argsort([rl[gl == c].mean() for c in comps])]


def compute_bgmm_contingency():
    b, gl = _bgmm()
    rl = b["R"]["labels"]
    comps = _comp_order(rl, gl)
    T = pd.crosstab(rl, gl).reindex(columns=comps)
    Tn = T.div(T.sum(1), axis=0)
    long = Tn.stack().rename("frac_of_cluster").reset_index()
    long.columns = ["cluster", "component", "frac_of_cluster"]
    long["component_rank"] = long.component.map({c: i for i, c in enumerate(comps)})
    long["metacluster"] = long.cluster.map(b["meta"]["families"])
    META["bgmm_contingency"] = dict(n_components=int(len(comps)), n_clusters=int(T.shape[0]),
                                    provenance=json.load(open(BGMM.parent / "bgmm_provenance.json")))
    return long


def compute_bgmm_agreement():
    b, gl = _bgmm()
    rl = b["R"]["labels"]
    fam_r = np.array([b["meta"]["families"][k] for k in rl])
    # BGMM components -> metacluster by majority of their neurons
    comp_meta = pd.Series(fam_r).groupby(gl).agg(lambda s: s.value_counts().index[0])
    fam_g = comp_meta.reindex(gl).to_numpy()
    rows = []
    for level, a, c in (("fine clusters", rl, gl), ("metaclusters", fam_r, fam_g)):
        for metric, f in (("AMI", adjusted_mutual_info_score), ("ARI", adjusted_rand_score)):
            real = f(a, c)
            null = [f(a, rng.permutation(c)) for _ in range(N_PERM if metric == "ARI" else 50)]
            rows.append(dict(level=level, metric=metric, value=real, null_mean=np.mean(null),
                             null_p95=np.percentile(null, 95), p_perm=(1 + np.sum(np.array(null) >= real)) / (1 + len(null))))
    META["bgmm_agreement"] = dict(frac_same_metacluster=float((fam_r == fam_g).mean()),
                                  n_perm_ari=N_PERM, n_perm_ami=50)
    return pd.DataFrame(rows)


def compute_bgmm_matched():
    """per BGMM component: max r of its held-out (even-trial) mean PSTH with any rastermap cluster mean; null:
    components of permuted labels"""
    b, gl = _bgmm()
    R = b["R"]
    ks, comps = b["clusters"], np.unique(gl)
    rm = group_means(R["X"], R["labels"], ks)
    C = r_matrix(group_means(R["X"], gl, comps), rm.copy())
    best = C.argmax(1)
    rows = [dict(kind="real", component=c, r_best=C[i, best[i]], best_cluster=ks[best[i]],
                 n_neurons=int((gl == c).sum())) for i, c in enumerate(comps)]
    for p in range(20):
        gp = rng.permutation(gl)
        Cp = r_matrix(group_means(R["X"], gp, comps), rm.copy())
        rows += [dict(kind="null", component=c, r_best=Cp[i].max(), perm=p) for i, c in enumerate(comps)]
    d = pd.DataFrame(rows)
    META["bgmm_matched"] = dict(r_median=float(d[d.kind == "real"].r_best.median()),
                                null_median=float(d[d.kind == "null"].r_best.median()),
                                frac_above_null_max=float((d[d.kind == "real"].r_best > d[d.kind == "null"].r_best.max()).mean()))
    return d


def compute_latency():
    """per fine cluster: median latenZy onset of its ZETA-positive neurons (whisker / auditory active trials)"""
    b = base()
    lab = b["lab"]
    z = pd.read_parquet(ZETA, columns=KEYS + ["analysis_type", "significant", "direction", "latency_s"],
                        filters=[("analysis_type", "in", ["whisker_active", "auditory_active"])])
    z["cluster_id"] = z.cluster_id.astype(str)
    z = z[z.significant & (z.direction == "positive")].dropna(subset=["latency_s"])
    m = lab[KEYS + ["cluster_label", "metacluster"]].merge(z, on=KEYS)
    d = (m.groupby(["analysis_type", "cluster_label"]).agg(latency_ms=("latency_s", lambda x: 1000 * x.median()),
                                                           n=("latency_s", "size"), metacluster=("metacluster", "first"))
         .reset_index().rename(columns={"cluster_label": "cluster"}))
    d = d[d.n >= 10]
    META["latency"] = {at: dict(zip(("rho", "p"), map(float, spearmanr(x.cluster, x.latency_ms))), n_clusters=len(x))
                       for at, x in d.groupby("analysis_type")}
    META["latency"]["matched_neurons"] = int(m[["mouse_id", "session_id", "electrode_group", "cluster_id"]]
                                             .drop_duplicates().shape[0])
    return d


def compute_hierarchy():
    """per fine cluster: mean Harris et al. (2019) hierarchy score of its neurons with a score (cortex)"""
    b = base()
    R = b["R"]
    h = R["anatomy"][pms.ANATOMY_COLS[1]]
    ok = np.isfinite(h)
    d = (pd.DataFrame(dict(cluster=R["labels"][ok], h=h[ok])).groupby("cluster").h.agg(["mean", "sem", "size"])
         .reset_index().rename(columns={"mean": "hierarchy", "size": "n"}))
    d = d[d.n >= 20]
    rho, p = spearmanr(d.cluster, d.hierarchy)
    null = []
    for _ in range(N_PERM):   # neurons shuffled across clusters (sizes kept)
        x = pd.Series(rng.permutation(h[ok])).groupby(R["labels"][ok]).mean().reindex(d.cluster)
        null.append(spearmanr(d.cluster, x)[0])
    d["metacluster"] = d.cluster.map(b["meta"]["families"])
    META["hierarchy"] = dict(rho=float(rho), p=float(p), p_perm=float((1 + np.sum(np.abs(null) >= abs(rho))) / (1 + N_PERM)),
                             n_clusters=int(len(d)), n_neurons=int(ok.sum()), column=pms.ANATOMY_COLS[1])
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
        if key.startswith("bgmm") and not BGMM.exists():
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
