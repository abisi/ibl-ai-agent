"""SuppFigure3_robustness -- panel tables (one CSV per panel, read by plot.py). Design: user 2026-11-26 ("S-C").

  a param_grid    054 fits (v2 units_good, locality x time_lag_window): cluster-mean PSTHs (even trials) of the reference
                  fit's 100 clusters + each fit's position of every reference cluster (fits drawn in the reference
                  direction, 054 is_flipped); 058 positions()
  b grid_summary  per fit: 058 order RSA vs the reference fit and vs the PSTH dissimilarity (+ null), 057 matched r
  c variants      per v2 variant: the variant's partition evaluated on the MAIN even-trial PSTHs of the shared neurons
                  (one PSTH space for all variants; noWhiskerReward has fewer bins), Hungarian match of its 100 cluster
                  means to the main cluster means (Pearson r, population mean kept; memory rule), order RSA = Spearman
                  between the main order RDM |i - j| and the matched variant clusters' order RDM
  d gmm_matrices  GMM (run_gmm_v2.py: 046 spectral embedding of X_odd + 047 EM GMM, K = 100 and K = 14) component means
                  (even trials) beside the rastermap cluster / metacluster means, components ordered by their matched
                  rastermap cluster
  e gmm_match     matched r vs size for both K
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure3_robustness/compute.py [key ...]
"""
import importlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

FIG_DIR = pathlib.Path(__file__).resolve().parent
REPO = FIG_DIR.parents[3]
sys.path.insert(0, str(FIG_DIR.parents[1]))
from common import load_spec, snapshot_dir  # noqa: E402

V2 = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/v2")
os.environ["RASTERMAP_VARIANTS_ROOT"] = str(V2)
os.environ["RASTERMAP_META_CUT"] = "0.40"
META_CUT = 0.40
EXPL = REPO / "projects" / "ssl-rastermap-psth-variants" / "exploratory-analyses"
sys.path.insert(0, str(EXPL))
sys.path.insert(0, str(pathlib.Path.home() / "code" / "unit_spikes_analysis"))
GRID = V2 / "_param_grid_units_good"
VARIANTS = ["mice_all", "units_all", "units_good", "noFR", "noWhiskerReward", "loc0p00", "loc0p75"]
VLABEL = {"mice_all": "All mice", "units_all": "All units (incl. unstable)", "units_good": "Good units only",
          "noFR": "No firing-rate threshold", "noWhiskerReward": "No whisker-reward bins", "loc0p00": "Locality 0",
          "loc0p75": "Locality 0.75"}
SPEC = load_spec(FIG_DIR)
PANELS = {p.get("key", p["id"]): p for p in SPEC["panels"]}
META, _cache = {}, {}


def r_matrix(A, B):
    """Pearson r between the rows of A and the rows of B (each PSTH centred on its own mean; population mean kept)"""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return A @ B.T


def means(X, lab, keys=None):
    keys = np.unique(lab) if keys is None else keys
    return keys, np.vstack([np.nanmean(X[lab == k], 0) for k in keys])


def bin_columns(n_bins_list):
    return [f"c{ci}_b{j}" for ci, n in enumerate(n_bins_list) for j in range(n)]


def main_run():
    if "main" in _cache:
        return _cache["main"]
    import rastermap_psth.population_matrix_summary as pms
    m050 = importlib.import_module("050_rastermap_variants_v2")
    md = m050.method_dir("main")
    R = pms.load_rastermap(md)
    fam = pms.metaclustering(md, cut=META_CUT)["families"]
    _cache["main"] = dict(m050=m050, pms=pms, md=md, R=R, fam=fam)
    return _cache["main"]


# ------------------------------------------------------------------------------------------------------------ a, b
def compute_param_grid():
    g054 = importlib.import_module("054_locality_timelag_grid")
    g058 = importlib.import_module("058_grid_order_rsa")
    m050, fm_path, out = g054.paths()
    d = np.load(fm_path, allow_pickle=True)
    Xe, nb = d["X_even"], [int(x) for x in d["n_bins_list"]]
    fits = {(l, t): np.load(GRID / "fits" / f"loc{l:.2f}_tlw{t}.npz") for l in g054.LOCALITY for t in g054.TLW}
    ref = fits[g054.REF]
    ref_node = ref["node"]
    K = g054.N_CLUSTERS
    M = g054.cluster_means(Xe, ref_node)
    df = pd.DataFrame(M, columns=bin_columns(nb))
    df.insert(0, "cluster_ref", np.arange(K))
    for (l, t), f in fits.items():
        pos = g058.positions(f, ref_node)
        if g054.is_flipped(f, ref):
            pos = K - 1 - pos
        df[f"pos_l{l:.2f}_t{t}"] = pos.astype(int)
    META["param_grid"] = dict(n_neurons=int(len(ref_node)), n_fits=len(fits), reference=list(g054.REF),
                              variant="v2 units_good", bins=int(sum(nb)))
    return df


def compute_grid_summary():
    rsa = pd.read_csv(GRID / "order_rsa.csv")
    btw = pd.read_csv(GRID / "order_rsa_between_fits.csv", index_col=0)
    cm = pd.read_csv(GRID / "cluster_matching.csv")
    met = pd.read_csv(GRID / "grid_metrics.csv")
    ref = "0.25/20"
    rows = []
    for r in rsa.itertuples():
        key = f"{r.locality:g}/{int(r.time_lag_window)}"
        key = key if key in btw.index else f"{r.locality}/{int(r.time_lag_window)}"
        m = cm[(cm.locality == r.locality) & (cm.time_lag_window == r.time_lag_window)].iloc[0]
        g = met[(met.locality == r.locality) & (met.time_lag_window == r.time_lag_window)].iloc[0]
        rows.append(dict(locality=r.locality, time_lag_window=int(r.time_lag_window),
                         rsa_order_vs_reference=float(btw.loc[key, ref]), rsa_order_vs_psth=float(r.rsa_psth),
                         rsa_order_vs_psth_null_mean=float(r.rsa_psth_null_mean),
                         rsa_order_vs_psth_null_sd=float(r.rsa_psth_null_sd),
                         matched_r_mean=float(m.mean_matched_r), matched_r_min=float(m.min_matched_r),
                         smoothness=float(g.smoothness)))
    df = pd.DataFrame(rows)
    META["grid_summary"] = dict(matched_r_min=float(df.matched_r_min.min()),
                                rsa_vs_ref_min=float(df.rsa_order_vs_reference.min()),
                                rsa_psth_range=[float(df.rsa_order_vs_psth.min()), float(df.rsa_order_vs_psth.max())],
                                rsa_psth_reference=float(df[(df.locality == 0.25) & (df.time_lag_window == 20)]
                                                         .rsa_order_vs_psth.iloc[0]))
    return df


# ------------------------------------------------------------------------------------------------------------ c
def compute_variants():
    b = main_run()
    m050, pms, R = b["m050"], b["pms"], b["R"]
    Xe, lab = R["X"], R["labels"]                         # X = even trials (checked: equals feature_matrix X_even)
    uid = pd.Index(R["unit_ids"])
    K = 100
    _, Mm = means(Xe, lab, np.arange(K))
    iu = np.triu_indices(K, 1)
    rdm_main = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])[iu]
    rows = []
    for v in VARIANTS:
        Rv = pms.load_rastermap(m050.method_dir(v))
        common = uid.get_indexer(pd.Index(Rv["unit_ids"]))
        ok = common >= 0
        lv = Rv["labels"][ok]
        Xv = Xe[common[ok]]
        kv, Mv = means(Xv, lv)
        C = r_matrix(Mm.copy(), Mv.copy())
        i, j = linear_sum_assignment(-C)
        pos_v = np.full(K, np.nan)
        pos_v[i] = kv[j]                                  # variant labels are positions along its order
        okp = np.isfinite(pos_v)
        rdm_v = np.abs(pos_v[:, None] - pos_v[None, :])[iu]
        good = np.isfinite(rdm_v)
        rsa = spearmanr(rdm_main[good], rdm_v[good])[0]
        sz_v = pd.Series(lv).value_counts()
        for a_, b_ in zip(i, j):
            rows.append(dict(variant=v, label=VLABEL[v], main_cluster=int(a_), variant_cluster=int(kv[b_]),
                             r=float(C[a_, b_]), n_main=int((lab == a_).sum()), n_variant_shared=int(sz_v.get(kv[b_], 0)),
                             n_neurons_variant=int(len(Rv["unit_ids"])), n_shared=int(ok.sum()),
                             frac_main_shared=float(ok.sum() / len(uid)), rsa_order=float(rsa),
                             n_matched=int(okp.sum())))
        print(v, "shared", int(ok.sum()), "median r", round(float(np.median(C[i, j])), 3), "rsa", round(float(rsa), 3),
              flush=True)
    df = pd.DataFrame(rows)
    META["variants"] = {v: dict(median_r=float(g.r.median()), rsa_order=float(g.rsa_order.iloc[0]),
                                n_shared=int(g.n_shared.iloc[0])) for v, g in df.groupby("variant")}
    return df


# ------------------------------------------------------------------------------------------------------------ d, e
def _gmm():
    if "gmm" in _cache:
        return _cache["gmm"]
    b = main_run()
    R, fam = b["R"], b["fam"]
    Xe, lab = R["X"], R["labels"]
    gdir = b["md"].parent / "gmm_bayesian_main"
    ml = np.array([fam[k] for k in lab])
    n_meta = int(len(set(fam.values())))
    out = {}
    for K, rl in ((100, lab), (n_meta, ml)):
        g = pd.read_csv(gdir / f"bgmm_k{K}_labels.csv")
        assert np.array_equal(g.unit_ids.to_numpy(), R["unit_ids"]), "GMM label order != rastermap order"
        gl = g.bgmm_component.to_numpy()
        kr, Mr = means(Xe, rl)
        kg, Mg = means(Xe, gl)
        C = r_matrix(Mr.copy(), Mg.copy())
        i, j = linear_sum_assignment(-C)
        out[K] = dict(kr=kr, Mr=Mr, kg=kg, Mg=Mg, i=i, j=j, r=C[i, j], rl=rl, gl=gl)
    _cache["gmm"] = dict(out=out, n_meta=n_meta, nb=[int(x) for x in R["n_bins_list"]])
    return _cache["gmm"]


def compute_gmm_matrices():
    G = _gmm()
    cols = bin_columns(G["nb"])
    blocks = []
    for K, o in G["out"].items():
        lvl = "cluster" if K == 100 else "metacluster"
        szr, szg = pd.Series(o["rl"]).value_counts(), pd.Series(o["gl"]).value_counts()
        dr = pd.DataFrame(o["Mr"], columns=cols)
        dr.insert(0, "row", np.arange(len(o["kr"])))
        dr.insert(0, "id", o["kr"]); dr.insert(0, "n", szr.reindex(o["kr"]).to_numpy())
        dr.insert(0, "block", f"rastermap_{lvl}")
        dr.insert(0, "K", K)
        # GMM components in the order of their matched rastermap cluster; unmatched components appended
        order = list(o["j"][np.argsort(o["i"])])
        order += [k for k in range(len(o["kg"])) if k not in set(order)]
        rmatch = dict(zip(o["j"], o["r"]))
        dg = pd.DataFrame(o["Mg"][order], columns=cols)
        dg.insert(0, "row", np.arange(len(order)))
        dg.insert(0, "id", o["kg"][order]); dg.insert(0, "n", szg.reindex(o["kg"][order]).to_numpy())
        dg.insert(0, "block", f"gmm_{lvl}")
        dg.insert(0, "K", K)
        dg["matched_r"] = [rmatch.get(k, np.nan) for k in order]
        blocks += [dr, dg]
    return pd.concat(blocks, ignore_index=True)


def compute_gmm_match():
    G = _gmm()
    rows = []
    for K, o in G["out"].items():
        szr, szg = pd.Series(o["rl"]).value_counts(), pd.Series(o["gl"]).value_counts()
        for a, b, r in zip(o["i"], o["j"], o["r"]):
            rows.append(dict(K=K, level="cluster" if K == 100 else "metacluster", rastermap=int(o["kr"][a]),
                             gmm=int(o["kg"][b]), r=float(r), n_rastermap=int(szr[o["kr"][a]]),
                             n_gmm=int(szg[o["kg"][b]]), n_gmm_components_used=int(len(o["kg"]))))
    df = pd.DataFrame(rows)
    META["gmm_match"] = {str(K): dict(median_r=float(g.r.median()), frac_r_gt_0p9=float((g.r > 0.9).mean()),
                                      frac_neurons_r_gt_0p9=float(g.n_rastermap[g.r > 0.9].sum() / g.n_rastermap.sum()),
                                      n_components_used=int(g.n_gmm_components_used.iloc[0]))
                         for K, g in df.groupby("K")}
    return df


def main():
    which = sys.argv[1:] or list(PANELS)
    out = snapshot_dir(SPEC)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "_build").mkdir(exist_ok=True)
    mf = out / "_build" / "panel_meta.json"
    allmeta = json.load(open(mf)) if mf.exists() else {}
    for key in which:
        p = PANELS[key]
        df = globals()[f"compute_{key}"]()
        df.to_csv(out / "data" / p["data"], index=False)
        allmeta[p["id"]] = META.get(key, {})
        print(p["id"], key, df.shape, json.dumps(META.get(key, {}), default=str)[:600], flush=True)
    allmeta["_shared"] = dict(root=str(V2), meta_cut=META_CUT)
    json.dump(allmeta, open(mf, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
