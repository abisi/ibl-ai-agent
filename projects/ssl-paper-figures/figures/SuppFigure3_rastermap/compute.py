"""SuppFigure3_rastermap (user 2026-11-26): ONE compact A4 supplementary figure merging the embedding, continuum,
annotated-matrix and areas / cohorts figures built the same day (their folders stay as the sources; their data CSVs are
copied from their snapshots, nothing is recomputed there).

New here:
  units   burstiness B and memory M per rastermap neuron (Goh & Barabasi 2008, EPL 81:48002, as used for single neurons
          by Le Merre et al. 2026 Nat Neurosci, prefrontal activity map -- definitions not checked against that paper's
          methods): B = (sigma - mu) / (sigma + mu) of the inter-spike intervals (= (CV - 1) / (CV + 1), a deterministic
          function of the ISI CV); M = Pearson r between consecutive ISIs (ISI_i, ISI_i+1). All spikes of the session;
          >= 50 ISIs required. -> _cache/suppfig3_rastermap_isi_stats.parquet (+ the CV / firing-rate / quality columns
          of SuppFigure3_annotated_matrix's unit cache)
  matrix  the Figure 3 A matrix (rastermap v2 main, metaclusters cut 0.40) with area group, Gao hierarchy, ROC function
          columns, unit columns (fraction good, firing rate, ISI CV, burstiness, memory; one shared colour key, each
          column scaled to its own 2-98th percentile range, given in the column title) and the R+ share, with the cohort
          of every mouse from the REFERENCE SHEET (MH035 is R- there, R+ in the rastermap label table): f_mk R+ share and
          the post hoc per-cluster Mann-Whitney (BH-FDR) flags recomputed with the corrected labels.
  isi     per-neuron M and B vs ISI CV (panel data).
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python <this> [copy] [units] [matrix] [isi]
"""
import importlib
import os
import pathlib
import pickle
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]
EXPL = REPO / "projects" / "ssl-rastermap-psth-variants" / "exploratory-analyses"
FIGROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures")
CACHE = FIGROOT / "_cache"
V2 = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/v2")
SHEET = pathlib.Path("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
VARIANT, META_CUT = "main", 0.40
os.environ["RASTERMAP_META_CUT"] = str(META_CUT)
os.environ["RASTERMAP_VARIANTS_ROOT"] = str(V2)
sys.path.insert(0, str(EXPL))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
sys.path.insert(0, str(HERE.parents[1]))
UNIT_CACHE = CACHE / "suppfig3b_unit_features.parquet"              # SuppFigure3_annotated_matrix compute_units
ISI_CACHE = CACHE / "suppfig3_rastermap_isi_stats.parquet"
SRC_DATE = "2026-11-26"
SOURCES = {"SuppFigure3_embedding": ["FigS_emb_schematic.csv", "FigS_emb_cv_cluster.csv", "FigS_emb_meta_matrix.csv",
                                     "FigS_emb_meta_cut.csv"],
           "SuppFigure3_continuum": ["FigS_cont_decay.csv", "FigS_cont_adjacent.csv", "FigS_cont_assignment.csv"],
           "SuppFigure3_areas_cohorts": ["FigS3a_area_sampling.csv", "FigS3c_mouse_yield.csv", "FigS3e_occupancy.csv",
                                         "FigS3h_specialization.csv"]}


def spec():
    return yaml.safe_load(open(HERE / "spec.yaml", encoding="utf-8"))


def cell(pid):
    return next(p for p in spec()["panels"] if p["id"] == pid)["cell"]


def out_dir():
    from common import snapshot_dir
    d = snapshot_dir(spec()) / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_copy():
    """source panel CSVs -> this snapshot; the continuum decay CSV gets the neighbouring-cluster test (adjacent CSV)"""
    out = out_dir()
    for fig, files in SOURCES.items():
        for f in files:
            shutil.copyfile(FIGROOT / f"{fig}_{SRC_DATE}" / "data" / f, out / f)
    adj = pd.read_csv(out / "FigS_cont_adjacent.csv")
    null, obs = adj[adj.kind == "null_mean"].value, adj[adj.kind == "obs"].value.mean()
    d = pd.read_csv(out / "FigS_cont_decay.csv")
    d["adj_obs"], d["adj_null_mean"] = obs, null.mean()
    d["adj_p"] = (1 + (null >= obs).sum()) / (1 + len(null))
    d["adj_n_perm"] = len(null)
    d.to_csv(out / "FigS_cont_decay.csv", index=False)
    print("copied", sum(len(v) for v in SOURCES.values()), "CSVs ->", out)


def _rastermap():
    import rastermap_psth.population_matrix_summary as pms
    m050 = importlib.import_module("050_rastermap_variants_v2")
    drv = m050._drv()
    cfg = drv.load_cfg(drv.CONFIG_PATH, **m050.cfg_of(VARIANT))
    drv.set_conditions(cfg)
    md = m050.method_dir(VARIANT)
    return pms, cfg, md, pms.load_rastermap(md)


def compute_units():
    pms, cfg, md, R = _rastermap()
    tables = pickle.load(open(V2 / "_tables" / "tables_day0.pkl", "rb"))
    U = tables["unit_table"].loc[R["unit_ids"]]
    B, M, n = [], [], []
    for st in U.spike_times:
        isi = np.diff(np.sort(np.asarray(st, float)))
        isi = isi[isi > 0]
        if len(isi) < 50:
            B.append(np.nan); M.append(np.nan); n.append(len(isi)); continue
        mu, sd = isi.mean(), isi.std()
        B.append((sd - mu) / (sd + mu))
        M.append(np.corrcoef(isi[:-1], isi[1:])[0, 1])
        n.append(len(isi))
    df = pd.read_parquet(UNIT_CACHE)
    assert np.array_equal(df.unit_id.to_numpy(), R["unit_ids"]), "unit cache order != rastermap order"
    df["burstiness"], df["memory"], df["n_isi"] = B, M, n
    df.to_parquet(ISI_CACHE, index=False)
    print("units ->", ISI_CACHE, int(np.isfinite(df.burstiness).sum()), "neurons with >= 50 ISIs")


def sheet_cohort():
    sh = pd.read_excel(SHEET)
    return dict(zip(sh.mouse_id.astype(str), sh.reward_group.astype(str)))


def fmk_sheet(pms, md):
    """f_mk stats with the reference-sheet cohort per mouse; post hoc per-cluster Mann-Whitney, BH-FDR (as the rastermap
    run's 'reject', which drives the filled R+ share markers)"""
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests
    fmk = pms.load_fmk_stats(md)
    z = np.load(md / "stats" / "f_matrix.npz", allow_pickle=True)
    mice = z["mouse_ids"].astype(str)
    old = fmk["groups"].copy()
    coh = sheet_cohort()
    new = np.array([coh.get(m, g) for m, g in zip(mice, old)])
    F = fmk["F"]

    def reject(g):
        p = np.array([mannwhitneyu(F[g == "R+", k], F[g == "R-", k]).pvalue for k in range(F.shape[1])])
        return multipletests(p, method="fdr_bh")[0], p
    r_old, _ = reject(old)
    r_new, p_new = reject(new)
    changed = [m for m, a, b in zip(mice, old, new) if a != b]
    fmk.update(groups=new, reject=r_new, p_fdr=multipletests(p_new, method="fdr_bh")[1])
    info = dict(relabelled=changed, n_sig_old=int(r_old.sum()), n_sig_new=int(r_new.sum()),
                gained=[int(k) + 1 for k in np.flatnonzero(r_new & ~r_old)],
                lost=[int(k) + 1 for k in np.flatnonzero(r_old & ~r_new)], n_rplus=int((new == "R+").sum()),
                n_rminus=int((new == "R-").sum()))
    return fmk, info


def compute_matrix():
    import json
    import matplotlib
    import matplotlib.colors as mcolors
    pms, cfg, md, R = _rastermap()
    roc = pms.build_roc_categories(md, R["unit_ids"])
    meta = pms.metaclustering(md, cut=META_CUT)
    fmk, info = fmk_sheet(pms, md)
    print("cohort relabel (reference sheet):", info, flush=True)
    df = pd.read_parquet(ISI_CACHE)
    df["frac_good"] = (df.quality_label == "good").astype(float)
    g = df.groupby("cluster")
    feats = [("frac_good", "Good", g.frac_good.mean(), False, lambda x: f"{x:.0%}"),
             ("firing_rate", "FR", g.firing_rate.median(), True, lambda x: f"{10 ** x:.2g}"),
             ("cv_isi", "ISI CV", g.cv_isi.median(), False, lambda x: f"{x:.2g}"),
             ("burstiness", "Burst.", g.burstiness.median(), False, lambda x: f"{x:.2f}"),
             ("memory", "Memory", g.memory.median(), False, lambda x: f"{x:.2f}")]
    cmap = matplotlib.colormaps["cividis"]
    unit_cols = {}
    for i, (c, title, v, log, fmt) in enumerate(feats):
        vv = np.log10(v.clip(lower=1e-6)) if log else v
        lo, hi = np.nanpercentile(vv, 2), np.nanpercentile(vv, 98)
        unit_cols[c] = dict(values=vv, cmap=cmap, norm=mcolors.Normalize(lo, hi, clip=True), ticks=[lo, hi],
                            ticklabels=["low", "high"], title=f"{title} {fmt(lo)}–{fmt(hi)}", extend="neither",
                            key=(i == 0), key_title="Units (column range)")
    c = cell("matrix")
    out = CACHE / "suppfig3_rastermap_matrix"
    cols = ["area_group", "cc_hierarchy_score_columns"] + list(pms.PUB_FUNCTION) + ["reward_dot"]
    pms.fig5_population_matrix_publication(R, roc, meta, fmk, cfg, out, columns=cols, row_mode="cluster_mean_uniform",
                                           width_mm=c["w_mm"], height_mm=c["h_mm"], base_fs=6.5,
                                           col_widths={"area_group": 1.0, "matrix": 8.0}, show_title=False,
                                           show_legends=False, aligned_keys=True, matrix_label="Δ rate (Hz)",
                                           show_block_headers=True, area_text_legend=True, unit_cols=unit_cols,
                                           gs_box=dict(left=0.04, right=0.80, top=0.86, bottom=0.17))
    json.dump(info, open(out_dir() / "cohort_relabel.json", "w"), indent=1)
    print("matrix ->", out.with_suffix(".pdf"))


def compute_isi():
    """per-neuron M, B and ISI CV (random 6000 neurons for the scatter + all for the stats); per-metacluster medians"""
    from scipy.stats import spearmanr
    df = pd.read_parquet(ISI_CACHE).dropna(subset=["cv_isi", "burstiness", "memory"])
    rho_m = spearmanr(df.cv_isi, df.memory)
    rho_b = spearmanr(df.cv_isi, df.burstiness)
    s = df.sample(min(6000, len(df)), random_state=0)
    pms, cfg, md, R = _rastermap()
    fam = pms.metaclustering(md, cut=META_CUT)["families"]
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(fam))])
    out = pd.DataFrame(dict(cv=s.cv_isi, burstiness=s.burstiness, memory=s.memory, metacluster=s.metacluster,
                            color=[mcolors_hex(pal[f]) for f in s.metacluster]))
    out["rho_cv_memory"], out["p_cv_memory"] = rho_m.correlation, rho_m.pvalue
    out["rho_cv_burst"] = rho_b.correlation
    out["n_neurons"] = len(df)
    out.to_csv(out_dir() / "FigS_isi_stats.csv", index=False)
    med = df.groupby("metacluster")[["cv_isi", "burstiness", "memory", "firing_rate"]].median()
    med["n"] = df.groupby("metacluster").size()
    med.to_csv(out_dir() / "FigS_isi_metacluster_medians.csv")
    print(f"isi: n {len(df)}; Spearman CV-memory {rho_m.correlation:.2f}, CV-burstiness {rho_b.correlation:.2f}")
    print(med.round(3).to_string())


def mcolors_hex(c):
    import matplotlib.colors as mc
    return mc.to_hex(c)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["copy", "units", "matrix", "isi"]:
        globals()[f"compute_{w}"]()
