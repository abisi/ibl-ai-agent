"""SuppFigure3_embedding -- how the Figure 3 response types are built (user 2026-11-26): pipeline schematic,
reproducibility of cluster-mean PSTHs across trial halves, pairwise cluster similarity used for the metaclustering, and
the choice of the cut. One compute_<key>() per panel writing data/<csv> of the dated snapshot.

Data: rastermap v2 main (stable good + MUA units, learner mice, whisker day 0, 63 sessions; locality 0.25, time-lag
window 20, 100 k-means nodes, fitted on odd-trial PSTHs), metaclusters contiguous at cut 1 - r = 0.40 (as Figure 3).
Ported from SuppFigure3_rastermap_clustering compute (v1 archive variant) to v2 main; those folders are not edited.

Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
  .venv/bin/python ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure3_embedding/compute.py [--only key ...]
then build.py figures/SuppFigure3_embedding.
"""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CUT = 0.40
V2 = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/v2")
os.environ["RASTERMAP_META_CUT"] = str(CUT)
os.environ["RASTERMAP_VARIANTS_ROOT"] = str(V2)
FIG_DIR = Path(__file__).resolve().parent
REPO = FIG_DIR.parents[3]
sys.path.insert(0, str(FIG_DIR.parents[1]))
sys.path.insert(0, str(REPO / "projects" / "ssl-rastermap-psth-variants" / "exploratory-analyses"))
sys.path.insert(0, str(Path.home() / "code" / "unit_spikes_analysis"))
from common import load_spec, snapshot_dir  # noqa: E402
from rastermap_psth import population_matrix_summary as pms  # noqa: E402

SPEC = load_spec(FIG_DIR)
PANELS = {p.get("key", p["id"]): p for p in SPEC["panels"]}
CUTS = np.round(np.arange(0.05, 0.951, 0.025), 3)
THUMB_STEP = 4            # schematic thumbnail: average of 4 bins
META = {}
_cache = {}


def r_matrix(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def group_means(X, labels, ks):
    return np.vstack([np.nanmean(X[labels == k], 0) for k in ks])


def base():
    """rastermap v2 main: labels, metaclusters at CUT, cluster means (full trials: the metaclustering input),
    Figure 3 A metacluster colours, condition labels"""
    if not _cache:
        m050 = importlib.import_module("050_rastermap_variants_v2")
        md = m050.method_dir("main")
        R = pms.load_rastermap(md)
        meta = pms.metaclustering(md, cut=CUT)
        clusters, means = pms._cluster_means(md)
        fam = meta["families"]
        pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(fam))])
        import matplotlib.colors as mc
        import rastermap_psth.rastermap_utils as ru
        drv = m050._drv()
        cfg = drv.load_cfg(drv.CONFIG_PATH, **m050.cfg_of("main"))
        cond_labels = list(ru.get_conditions(cfg)[1])
        feat = np.load(pms.find_data_folder(md) / "feature_matrix.npz", allow_pickle=True)
        assert np.array_equal(feat["unit_ids"], R["unit_ids"])
        _cache.update(md=md, R=R, meta=meta, clusters=clusters, means=means, fam=fam, cfg=cfg,
                      color={f: mc.to_hex(c) for f, c in pal.items()}, cond_labels=cond_labels,
                      odd_even_r=feat["odd_even_r"] if "odd_even_r" in feat.files else None)
    return _cache


def compute_schematic():
    """thumbnail of the cluster-mean matrix (100 clusters x concatenated conditions, bins averaged by THUMB_STEP) with
    condition blocks and metacluster colours, for the pipeline schematic"""
    b = base()
    R, means = b["R"], b["means"]
    nb = list(R["n_bins_list"])
    off = np.r_[0, np.cumsum(nb)]
    rows = []
    col = 0
    for c, (lo, hi) in enumerate(zip(off[:-1], off[1:])):
        for s in range(lo, hi, THUMB_STEP):
            v = means[:, s:min(s + THUMB_STEP, hi)].mean(1)
            rows += [dict(kind="thumb", i=int(k), j=col, value=float(v[k]), cond=c) for k in range(len(means))]
            col += 1
    rows += [dict(kind="cond", i=c, label=lab, n_bins=int(n)) for c, (lab, n) in enumerate(zip(b["cond_labels"], nb))]
    rows += [dict(kind="fam", i=int(k), value=int(b["fam"][k]), color=b["color"][b["fam"][k]]) for k in b["clusters"]]
    cfg = b["cfg"]
    META["schematic"] = dict(n_neurons=int(len(R["labels"])), n_clusters=int(len(b["clusters"])),
                             n_metaclusters=int(max(b["fam"].values())), n_mice=int(len(set(R["mouse_arr"]))),
                             n_conditions=len(nb), bin_ms=cfg.get("bin_ms"), stride_ms=cfg.get("stride_ms"),
                             locality=cfg.get("locality"), time_lag_window=cfg.get("time_lag_window"))
    return pd.DataFrame(rows)


def compute_cv_cluster():
    """per cluster: r(mean PSTH on odd trials, mean PSTH on even trials) of the same neurons (r_self), and the highest r
    with any OTHER cluster's even-trial mean (r_best_other)"""
    b = base()
    R, ks = b["R"], b["clusters"]
    odd, even = group_means(R["X_odd"], R["labels"], ks), group_means(R["X"], R["labels"], ks)
    C = r_matrix(odd, even)
    self_r = np.diag(C).copy()
    other = C.copy()
    np.fill_diagonal(other, np.nan)
    d = pd.DataFrame(dict(cluster=ks, r_self=self_r, r_best_other=np.nanmax(other, 1),
                          r_median_other=np.nanmedian(other, 1),
                          n_neurons=[int((R["labels"] == k).sum()) for k in ks],
                          metacluster=[int(b["fam"][k]) for k in ks]))
    d["color"] = d.metacluster.map(b["color"])
    META["cv_cluster"] = dict(n_clusters=len(ks), r_self_median=float(np.median(self_r)),
                              r_self_min=float(self_r.min()), r_best_other_median=float(np.median(d.r_best_other)),
                              frac_self_beats_others=float((d.r_self > d.r_best_other).mean()))
    return d


def compute_cv_neuron():
    """per neuron: Pearson r between its odd-trial and even-trial PSTH vectors (feature_matrix odd_even_r, else
    recomputed), with its metacluster"""
    b = base()
    R = b["R"]
    r = b["odd_even_r"]
    if r is None:
        r = np.array([np.corrcoef(np.nan_to_num(a), np.nan_to_num(c))[0, 1] for a, c in zip(R["X_odd"], R["X"])])
    fam = np.array([b["fam"][k] for k in R["labels"]])
    d = pd.DataFrame(dict(unit_id=R["unit_ids"], cluster=R["labels"], metacluster=fam, r_odd_even=np.asarray(r, float)))
    META["cv_neuron"] = dict(n_neurons=len(d), r_median=float(np.nanmedian(d.r_odd_even)),
                             r_iqr=[float(np.nanpercentile(d.r_odd_even, 25)), float(np.nanpercentile(d.r_odd_even, 75))],
                             source="feature_matrix odd_even_r" if b["odd_even_r"] is not None else "recomputed")
    return d


def compute_meta_matrix():
    b = base()
    C = r_matrix(b["means"], b["means"])
    fam = np.array([b["fam"][k] for k in b["clusters"]])
    K = len(C)
    rows = [dict(kind="r", i=i, j=j, value=float(C[i, j]), fam_i=int(fam[i])) for i in range(K) for j in range(K)]
    rows += [dict(kind="Z", i=int(z[0]), j=int(z[1]), value=float(z[2]), n=int(z[3])) for z in b["meta"]["Z"]]
    rows += [dict(kind="fam", i=int(k), fam_i=int(fam[k]), color=b["color"][fam[k]]) for k in range(K)]
    off = ~np.eye(K, dtype=bool)
    same = (fam[:, None] == fam[None, :]) & off
    META["meta_matrix"] = dict(n_clusters=K, n_metaclusters=int(fam.max()), cut=CUT,
                               r_within_median=float(np.median(C[same])),
                               r_between_median=float(np.median(C[off & ~same])))
    return pd.DataFrame(rows)


def compute_meta_cut():
    """number of metaclusters, median r within and between metaclusters, and median metacluster size, per cut"""
    b = base()
    C = r_matrix(b["means"], b["means"])
    K = len(C)
    off = ~np.eye(K, dtype=bool)
    rows = []
    for c in CUTS:
        fam = pms.contiguous_metaclustering(b["means"], c)[0]
        same = (fam[:, None] == fam[None, :]) & off
        rows.append(dict(cut=float(c), n_metaclusters=int(fam.max()),
                         r_within=float(np.median(C[same])) if same.any() else np.nan,
                         r_between=float(np.median(C[off & ~same])) if (off & ~same).any() else np.nan))
    d = pd.DataFrame(rows)
    META["meta_cut"] = dict(cut=CUT, n_at_cut=int(max(b["fam"].values())))
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
        print(p["id"], key, len(df), META.get(key), flush=True)
    allmeta["_shared"] = dict(variant="v2 main", method_dir=str(base()["md"]), cut=CUT)
    json.dump(allmeta, open(mf, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
