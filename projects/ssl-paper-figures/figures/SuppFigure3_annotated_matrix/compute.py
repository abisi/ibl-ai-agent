"""Supplementary figure "S-B" (user 2026-11-26): the Figure 3 A population matrix (rastermap v2 main, 100 clusters in
rastermap order, metaclusters at cut 0.40, same strip / colours) with the columns removed from Figure 3: waveform type,
anatomy scores (wS1/2 projection, Liu '24; hierarchy, Harris '19; hierarchy, Gao '26), reward-group enrichment (R+ share),
plus per-cluster unit properties chosen after the IBL ephys-atlas unit features (see UNIT_FEATURES).

  a  annotated matrix: unit_spikes_analysis rastermap_psth.population_matrix_summary.fig5_population_matrix_publication
     (opt-in unit_cols, 2026-11-26) -> _cache/suppfig3b_annotated_matrix.pdf
  b  per-metacluster summary of every unit feature (median per metacluster, z-scored across metaclusters) + fraction good
  c  per-cluster table of every feature (median, IQR) -> CSV (source data)
Unit features (per neuron), from the day-0 unit table the rastermap was built from (rastermap_variants/v2/_tables):
bombcell metrics + waveform slopes from the mean waveform + ISI statistics from the spike times (full tables pickle).
Run (haas): cd ~/code/unit_spikes_analysis; PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/SuppFigure3_annotated_matrix/compute.py [units] [a] [b]
"""
import importlib
import os
import pathlib
import pickle
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
VARIANT, META_CUT = "main", 0.40                       # as Figure 3
os.environ["RASTERMAP_META_CUT"] = str(META_CUT)
os.environ["RASTERMAP_VARIANTS_ROOT"] = str(V2)
sys.path.insert(0, str(EXPL))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
UNIT_CACHE = CACHE / "suppfig3b_unit_features.parquet"

# feature: (label, IBL ephys-atlas unit feature it stands for, log10 scale)
UNIT_FEATURES = {
    "firing_rate": ("Firing rate (spk/s)", "firing_rate", True),
    "rawAmplitude": ("Amplitude (a.u.)", "amp_median", True),
    "waveformDuration_peakTrough": ("Peak-trough (µs)", "peak_to_trough_duration", False),
    "repol_slope": ("Repolarisation slope", "repolarisation_slope", True),
    "recovery_slope": ("Recovery slope", "recovery_slope", False),
    "pt_ratio": ("Peak / trough", "polarity / peak-trough ratio", False),
    "spatialDecaySlope": ("Spatial decay", "spatial spread", False),
    "cv_isi": ("ISI CV", "-- (not an atlas feature)", False),
    "fractionRPVs_estimatedTauR": ("RP violations", "contamination / slidingRP_viol", False),
    "percentageSpikesMissing_gaussian": ("Missed spikes (%)", "missed_spikes_est / noise_cutoff", False),
    "presenceRatio": ("Presence ratio", "presence_ratio", False),
    "maxDriftEstimate": ("Max drift (µm)", "drift", False),
    "signalToNoiseRatio": ("SNR", "-- (bombcell)", True),
}
MATRIX_UNIT_COLS = ["frac_good", "firing_rate", "rawAmplitude", "waveformDuration_peakTrough", "cv_isi",
                    "signalToNoiseRatio"]      # shown as matrix columns; every feature is in panel b / the CSV


def spec():
    return yaml.safe_load(open(HERE / "spec.yaml", encoding="utf-8"))


def cell(pid):
    return next(p for p in spec()["panels"] if p["id"] == pid)["cell"]


def _rastermap():
    import rastermap_psth.population_matrix_summary as pms
    m050 = importlib.import_module("050_rastermap_variants_v2")
    drv = m050._drv()
    cfg = drv.load_cfg(drv.CONFIG_PATH, **m050.cfg_of(VARIANT))
    drv.set_conditions(cfg)
    md = m050.method_dir(VARIANT)
    return pms, cfg, md, pms.load_rastermap(md)


def _slopes(w, fs):
    """IBL-style waveform slopes on the mean waveform (peak channel): baseline = median of the first 10 samples;
    trough = minimum; repolarisation slope = (post-trough peak - trough) / time between them; recovery slope = linear
    slope over the 0.3 ms after the post-trough peak. Units: amplitude units per ms."""
    w = np.asarray(w, float)
    if w.ndim != 1 or len(w) < 20 or not np.isfinite(w).all():
        return np.nan, np.nan
    w = w - np.median(w[:10])
    it = int(np.argmin(w))
    if it >= len(w) - 3:
        return np.nan, np.nan
    ip = it + int(np.argmax(w[it:]))
    dt = 1000.0 / fs                                        # ms per sample
    repol = (w[ip] - w[it]) / max((ip - it) * dt, dt)
    n = max(2, int(round(0.3 / dt)))
    seg = w[ip:ip + n]
    rec = np.polyfit(np.arange(len(seg)) * dt, seg, 1)[0] if len(seg) >= 3 else np.nan
    return repol, rec


def compute_units():
    """per-neuron features of the rastermap neurons -> _cache/suppfig3b_unit_features.parquet (ISI from the full pickle)"""
    pms, cfg, md, R = _rastermap()
    tables = pickle.load(open(V2 / "_tables" / "tables_day0.pkl", "rb"))
    U = tables["unit_table"].loc[R["unit_ids"]].copy()
    fam = pms.metaclustering(md, cut=META_CUT)["families"]
    df = pd.DataFrame(dict(unit_id=R["unit_ids"], cluster=R["labels"], metacluster=[fam[k] for k in R["labels"]],
                           mouse_id=U.mouse_id.to_numpy(), session_id=U.session_id.to_numpy(),
                           electrode_group=U.electrode_group.to_numpy(), cluster_id=U.cluster_id.astype(str).to_numpy(),
                           quality_label=U.quality_label.astype(str).to_numpy(),
                           waveform_type=U.waveform_type.astype(str).to_numpy()))
    for c in [c for c in UNIT_FEATURES if c in U.columns]:
        df[c] = pd.to_numeric(U[c], errors="coerce").to_numpy()
    fs = pd.to_numeric(U.sampling_rate, errors="coerce").fillna(30000.0).to_numpy()
    sl = [_slopes(w, f) for w, f in zip(U.waveform_mean, fs)]
    df["repol_slope"] = [a for a, _ in sl]
    df["recovery_slope"] = [b for _, b in sl]
    cv, burst = [], []
    for st in U.spike_times:
        isi = np.diff(np.asarray(st, float))
        isi = isi[isi > 0]
        cv.append(isi.std() / isi.mean() if len(isi) > 10 else np.nan)
        burst.append((isi < 0.01).mean() if len(isi) > 10 else np.nan)
    df["cv_isi"], df["frac_isi_lt10ms"] = cv, burst
    df.to_parquet(UNIT_CACHE, index=False)
    print("units ->", UNIT_CACHE, len(df), "neurons")


def cluster_table():
    df = pd.read_parquet(UNIT_CACHE)
    df["frac_good"] = (df.quality_label == "good").astype(float)
    g = df.groupby("cluster")
    out = pd.DataFrame(dict(n=g.size(), metacluster=g.metacluster.first(), frac_good=g.frac_good.mean(),
                            frac_narrow=g.waveform_type.apply(lambda s: (s == "fsu").mean())))
    for c in UNIT_FEATURES:
        out[f"{c}_median"] = g[c].median()
        out[f"{c}_q25"] = g[c].quantile(0.25)
        out[f"{c}_q75"] = g[c].quantile(0.75)
    return df, out.reset_index()


def compute_a():
    import matplotlib
    import matplotlib.colors as mcolors
    pms, cfg, md, R = _rastermap()
    roc = pms.build_roc_categories(md, R["unit_ids"])
    meta = pms.metaclustering(md, cut=META_CUT)
    fmk = pms.load_fmk_stats(md)
    _, ct = cluster_table()
    ct = ct.set_index("cluster")
    cmap = matplotlib.colormaps["cividis"]
    unit_cols = {}
    for c in MATRIX_UNIT_COLS:
        if c == "frac_good":
            v, title, log = ct.frac_good, "Good (vs MUA)", False
        else:
            v, title, log = ct[f"{c}_median"], UNIT_FEATURES[c][0], UNIT_FEATURES[c][2]
        vv = np.log10(v.clip(lower=1e-6)) if log else v
        lo, hi = np.nanpercentile(vv, 2), np.nanpercentile(vv, 98)
        fmt = (lambda x: f"{10 ** x:.3g}") if log else ((lambda x: f"{x:.0%}") if c == "frac_good" else (lambda x: f"{x:.3g}"))
        unit_cols[c] = dict(values=vv, cmap=cmap, norm=mcolors.Normalize(lo, hi, clip=True), ticks=[lo, hi],
                            ticklabels=[fmt(lo), fmt(hi)], title=title + (" (log)" if log else ""), extend="both")
    c = cell("a")
    out = CACHE / "suppfig3b_annotated_matrix"
    cols = (["area_group", "waveform_type", "avg_ipsi", "cc_tc_ct_iterated", "cc_hierarchy_score_columns"]
            + list(pms.PUB_FUNCTION) + ["reward_dot"])
    pms.fig5_population_matrix_publication(R, roc, meta, fmk, cfg, out, columns=cols, row_mode="cluster_mean_uniform",
                                           width_mm=c["w_mm"], height_mm=c["h_mm"], base_fs=6.5,
                                           col_widths={"area_group": 0.8, "matrix": 6.5}, show_title=False,
                                           show_legends=True, matrix_label="Δ firing rate (spk/s)", unit_cols=unit_cols)
    print("a ->", out.with_suffix(".pdf"))


def compute_b():
    """per metacluster (rastermap order): median of every unit feature, z-scored across metaclusters; + fraction good;
    and the per-cluster table (median, IQR) as source data"""
    sys.path.insert(0, str(HERE.parents[1]))
    from common import snapshot_dir
    pms, cfg, md, R = _rastermap()
    fam = pms.metaclustering(md, cut=META_CUT)["families"]
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(fam))])
    df, ct = cluster_table()
    out = snapshot_dir(spec()) / "data"
    out.mkdir(parents=True, exist_ok=True)
    ct.to_csv(out / "SuppFig3b_cluster_unit_properties.csv", index=False)
    order = list(dict.fromkeys(fam[k] for k in range(len(fam))))          # rastermap order of metaclusters
    df["frac_good"] = (df.quality_label == "good").astype(float)
    rows = []
    for f in order:
        d = df[df.metacluster == f]
        for c in ["frac_good"] + list(UNIT_FEATURES):
            v = d[c].mean() if c == "frac_good" else d[c].median()
            rows.append(dict(metacluster=int(f), feature=c, label="Good (vs MUA)" if c == "frac_good" else UNIT_FEATURES[c][0],
                             ibl_feature="--" if c == "frac_good" else UNIT_FEATURES[c][1], value=float(v), n=len(d),
                             mc_color=matplotlib_hex(pal[f]), mc_rank=order.index(f)))
    m = pd.DataFrame(rows)
    m["z"] = m.groupby("feature").value.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    m.to_csv(out / "SuppFig3b_metacluster_unit_properties.csv", index=False)
    print("b ->", out, m.metacluster.nunique(), "metaclusters")


def matplotlib_hex(c):
    import matplotlib.colors as mcolors
    return mcolors.to_hex(c)


if __name__ == "__main__":
    for w in sys.argv[1:] or ["units", "a", "b"]:
        globals()[f"compute_{w}"]()
