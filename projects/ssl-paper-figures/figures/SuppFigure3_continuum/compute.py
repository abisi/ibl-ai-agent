"""SuppFigure3_continuum -- do the Figure 3 response types form a continuum along the rastermap order? (user 2026-11-26,
separate figure). Same data as SuppFigure3_embedding (rastermap v2 main, 100 clusters, metaclusters at 1 - r = 0.40);
base() is imported from that figure's compute.py (one source).

  decay      Pearson r between the cluster-mean PSTHs of clusters k and k + d (d = distance along the rastermap order),
             median and IQR over k, against 1000 random orders of the clusters (null: no relation between order and
             similarity; band = 2.5-97.5 % of the per-order mean r at each distance)
  adjacent   mean r of the 99 neighbouring pairs vs the same mean in 1000 random orders (permutation p)
  nearest    per cluster: r with its best rastermap neighbour (k - 1 or k + 1) vs r with its nearest cluster in PSTH space;
             fraction of clusters whose nearest cluster is a rastermap neighbour (null 2/99 for a random order)
  assignment per neuron: Pearson r of its EVEN-trial PSTH with the ODD-trial centroid of every cluster (held out from the
             fit); distance along the order between its own cluster and its second-best cluster, against the
             distance expected if the second-best cluster were random; margin own - second best
Cluster means = full-trial means (the metaclustering input, pms._cluster_means).
Run on haas: cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
  .venv/bin/python ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure3_continuum/compute.py
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIG_DIR = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("emb_compute", FIG_DIR.parent / "SuppFigure3_embedding" / "compute.py")
emb = importlib.util.module_from_spec(_s)
_s.loader.exec_module(emb)
from common import load_spec, snapshot_dir  # noqa: E402  (path set by the embedding module)

SPEC = load_spec(FIG_DIR)
PANELS = {p.get("key", p["id"]): p for p in SPEC["panels"]}
N_PERM = 1000
D_MAX = 40
rng = np.random.default_rng(0)
META = {}


def _C():
    b = emb.base()
    return emb.r_matrix(b["means"], b["means"])


def compute_decay():
    C = _C()
    K = len(C)
    rows = []
    perms = [rng.permutation(K) for _ in range(N_PERM)]
    for d in range(1, D_MAX + 1):
        obs = np.array([C[k, k + d] for k in range(K - d)])
        null = np.array([np.mean([C[p[k], p[k + d]] for k in range(K - d)]) for p in perms])
        rows.append(dict(d=d, median=float(np.median(obs)), q25=float(np.percentile(obs, 25)),
                         q75=float(np.percentile(obs, 75)), mean=float(obs.mean()),
                         null_lo=float(np.percentile(null, 2.5)), null_hi=float(np.percentile(null, 97.5)),
                         null_mean=float(null.mean())))
    d = pd.DataFrame(rows)
    above = d[d["mean"] > d.null_hi]
    META["decay"] = dict(n_perm=N_PERM, r_d1_median=float(d.loc[0, "median"]), r_null_mean=float(d.null_mean.mean()),
                         d_last_above_null=int(above.d.max()) if len(above) else 0,
                         d_first_within_null=int(d[d["mean"] <= d.null_hi].d.min()) if (d["mean"] <= d.null_hi).any() else None)
    return d


def compute_adjacent():
    C = _C()
    K = len(C)
    obs = np.array([C[k, k + 1] for k in range(K - 1)])
    null = np.array([np.mean([C[p[k], p[k + 1]] for k in range(K - 1)]) for p in (rng.permutation(K) for _ in range(N_PERM))])
    p = (1 + (null >= obs.mean()).sum()) / (1 + N_PERM)
    META["adjacent"] = dict(r_adjacent_mean=float(obs.mean()), r_adjacent_median=float(np.median(obs)),
                            null_mean=float(null.mean()), null_p95=float(np.percentile(null, 95)), p=float(p), n_perm=N_PERM)
    return pd.concat([pd.DataFrame(dict(kind="obs", value=obs)), pd.DataFrame(dict(kind="null_mean", value=null))])


def compute_nearest():
    b = emb.base()
    C = _C()
    K = len(C)
    Cn = C.copy()
    np.fill_diagonal(Cn, -np.inf)
    nn = Cn.argmax(1)
    r_nn = Cn.max(1)
    r_rm = np.array([max(C[k, k - 1] if k > 0 else -np.inf, C[k, k + 1] if k < K - 1 else -np.inf) for k in range(K)])
    dist = np.abs(nn - np.arange(K))
    d = pd.DataFrame(dict(cluster=np.arange(K), r_nearest=r_nn, r_rastermap_neighbour=r_rm, nearest=nn, distance=dist,
                          metacluster=[int(b["fam"][k]) for k in range(K)]))
    d["color"] = d.metacluster.map(b["color"])
    META["nearest"] = dict(frac_nearest_is_neighbour=float((dist == 1).mean()), frac_within_3=float((dist <= 3).mean()),
                           null_neighbour=2 / (K - 1), median_distance=float(np.median(dist)))
    return d


def compute_assignment():
    b = emb.base()
    R = b["R"]
    lab = R["labels"]
    K = int(lab.max()) + 1
    cent = emb.group_means(R["X_odd"], lab, range(K))
    X = np.nan_to_num(R["X"])
    Xz = X - X.mean(1, keepdims=True)
    Xz /= np.maximum(np.linalg.norm(Xz, axis=1, keepdims=True), 1e-12)
    Cz = cent - cent.mean(1, keepdims=True)
    Cz /= np.linalg.norm(Cz, axis=1, keepdims=True)
    S = Xz @ Cz.T                                         # neurons x clusters
    own = S[np.arange(len(lab)), lab]
    S2 = S.copy()
    S2[np.arange(len(lab)), lab] = -np.inf
    second = S2.argmax(1)
    margin = own - S2.max(1)
    best = S.argmax(1)
    dist = np.abs(second - lab)
    edges = list(range(1, 11)) + [K]                      # distances 1..10, then > 10
    rows = []
    for i, d0 in enumerate(edges[:-1]):
        sel = dist == d0
        # null: second-best cluster uniform over the other K - 1 clusters, given each neuron's own position
        pnull = np.mean([((np.abs(np.arange(K) - k) == d0).sum()) / (K - 1) for k in lab])
        rows.append(dict(kind="dist", d=str(d0), frac_obs=float(sel.mean()), frac_null=float(pnull)))
    rows.append(dict(kind="dist", d=">10", frac_obs=float((dist > 10).mean()),
                     frac_null=float(np.mean([((np.abs(np.arange(K) - k) > 10).sum()) / (K - 1) for k in lab]))))
    h, e = np.histogram(margin, bins=np.linspace(-0.3, 0.6, 46))
    rows += [dict(kind="margin", bin_lo=float(a), bin_hi=float(c), count=int(n)) for a, c, n in zip(e[:-1], e[1:], h)]
    META["assignment"] = dict(n_neurons=int(len(lab)), frac_own_best=float((best == lab).mean()),
                              frac_second_adjacent=float((dist == 1).mean()),
                              frac_second_within_3=float((dist <= 3).mean()),
                              null_second_adjacent=float(rows[0]["frac_null"]),
                              margin_median=float(np.median(margin)), frac_margin_lt_005=float((margin < 0.05).mean()))
    return pd.DataFrame(rows)


def compute_margin():
    """panel e reads the same table as d (margin histogram rows); its caption numbers are d's"""
    df = compute_assignment()
    META["margin"] = META["assignment"]
    return df


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
    allmeta["_shared"] = dict(variant="v2 main", method_dir=str(emb.base()["md"]), cut=emb.CUT)
    json.dump(allmeta, open(mf, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
