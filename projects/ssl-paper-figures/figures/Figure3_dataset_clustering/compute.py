"""Figure 3 (dataset clustering) -- panel images rendered by their analysis functions at final panel size (as SuppFigure3
panel a), embedded by build.py from Figures/_cache/ (PDF source -> 1200-dpi PNG).

  a  population matrix: unit_spikes_analysis rastermap_psth.population_matrix_summary.fig5_population_matrix_publication,
     v2 main (cluster means, rastermap order), side columns = area group (wider, user 2026-11-25) + all ROC function
     columns; no reward-group, waveform or anatomy columns (user 2026-11-25)
  b  example metaclusters: ssl-rastermap-psth-variants/045_metacluster_description.py, v2 main, the 5 largest metaclusters
     (user 2026-11-25) in rastermap order, one per row, panel width
Run (haas): cd ~/code/unit_spikes_analysis; RASTERMAP_META_CUT=0.30 PYTHONPATH=/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities:. \
            .venv/bin/python ~/code/ibl-ai-agent/projects/ssl-paper-figures/figures/Figure3_dataset_clustering/compute.py [a] [b]
"""
import importlib
import os
import pathlib
import shutil
import subprocess
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[3]                                   # ibl-ai-agent
EXPL = REPO / "projects" / "ssl-rastermap-psth-variants" / "exploratory-analyses"
FIGROOT = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures")
CACHE = FIGROOT / "_cache"
V2 = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/rastermap_variants/v2")
VARIANT = "main"
META_CUT = 0.40         # user 2026-11-25: "rerun figure with r = 0.4" (metacluster cut 1 - r = 0.40)
N_EXAMPLES = 6          # user 2026-11-25: the 6 largest metaclusters
N_AREAS = 10           # area groups shown in panel C (most neurons)
A_LEGEND_MM = 0.0      # no legends below the matrix any more (user 2026-11-25); colour bars stay inside
os.environ["RASTERMAP_META_CUT"] = str(META_CUT)
os.environ["RASTERMAP_VARIANTS_ROOT"] = str(V2)
sys.path.insert(0, str(EXPL))
sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))


def spec():
    return yaml.safe_load(open(HERE / "spec.yaml", encoding="utf-8"))


def cell(pid):
    return next(p for p in spec()["panels"] if p["id"] == pid)["cell"]


def compute_a():
    import rastermap_psth.population_matrix_summary as pms
    m050 = importlib.import_module("050_rastermap_variants_v2")
    drv = m050._drv()
    cfg_full = m050.cfg_of(VARIANT)
    cfg = drv.load_cfg(drv.CONFIG_PATH, **cfg_full)
    drv.set_conditions(cfg)
    md = m050.method_dir(VARIANT)
    R = pms.load_rastermap(md)
    roc = pms.build_roc_categories(md, R["unit_ids"])
    meta = pms.metaclustering(md, cut=META_CUT)
    fmk = pms.load_fmk_stats(md)
    c = cell("a")
    out = CACHE / "fig3a_population_matrix"
    pms.fig5_population_matrix_publication(R, roc, meta, fmk, cfg, out, anatomy=["area_group"],
                                           function=pms.PUB_FUNCTION, enrichment=None, row_mode="cluster_mean_uniform",
                                           width_mm=c["w_mm"], height_mm=c["h_mm"] - A_LEGEND_MM, base_fs=6.5,
                                           col_widths={"area_group": 1.2, "matrix": 10.0}, show_title=False,
                                           show_legends=False, aligned_keys=True, matrix_label="Δ firing rate (spks/s)",
                                           show_block_headers=False, area_text_legend=True,
                                           gs_box=dict(left=0.03, right=0.855, top=0.93, bottom=0.15))   # right: area names
    print("a ->", out.with_suffix(".pdf"))


def families():
    """metacluster of every rastermap cluster at META_CUT (v2 main), plus cluster sizes"""
    import numpy as np
    import rastermap_psth.population_matrix_summary as pms
    m050 = importlib.import_module("050_rastermap_variants_v2")
    md = m050.method_dir(VARIANT)
    fam = pms.metaclustering(md, cut=META_CUT)["families"]
    labels = pms.load_rastermap(md)["labels"]
    return fam, np.bincount(labels, minlength=len(fam)), md, pms


def compute_b(cols=None):
    """the metacluster summary (045) of the N_EXAMPLES largest metaclusters at META_CUT in rastermap order, square panels
    (no --height-mm), passive pre and post merged, compact (no legends, metacluster number only; user 2026-11-25);
    written to its own folder so the variant's 0.30 tables stay untouched."""
    import pandas as pd
    cols = int(cols or os.environ.get("B_COLS", 2))
    fam, size, _, _ = families()
    mc_size = pd.Series(size).groupby(pd.Series(fam)).sum()
    # decreasing metacluster number (user 2026-11-26): sensory types (12, 7, 5) first, then lick-related (4, 3, 2)
    order = sorted(mc_size.sort_values(ascending=False).index[:N_EXAMPLES].astype(int), reverse=True)
    c = cell("b")
    name = f"fig3b_example_metaclusters_cols{cols}"
    outdir = CACHE / "fig3_045"
    cmd = [sys.executable, "-u", str(EXPL / "045_metacluster_description.py"), "--variant", VARIANT,
           "--order", *map(str, order), "--name", name, "--cols", str(cols),
           "--fs", os.environ.get("B_FS", "5.5" if cols == 1 else "5.0"), "--merge-passive", "--compact", "--no-names", "--col-major", "--mc-colors", "--meta-cut", str(META_CUT),
           "--out-dir", str(outdir), "--width-mm", str(c["w_mm"]), "--height-mm", str(c["h_mm"])]   # user 2026-11-26: B taller
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, "RASTERMAP_META_CUT": str(META_CUT)})   # 050 import resets it
    shutil.copyfile(outdir / f"{name}.pdf", CACHE / f"{name}.pdf")    # copyfile: CIFS refuses copy2 metadata
    print("b ->", CACHE / f"{name}.pdf", "metaclusters", order, "cut", META_CUT)


C_SRC = FIGROOT / "_explore" / "Figure3_C_options_2026-11-25"     # option (iv) computed by explore_c_option_iv.py


ZOOM_DEC = ("wdet", "adet", "wva")     # user 2026-11-25: zoom on the first 50 ms (2-ms grid, 20-ms bins)
EXAMPLE = dict(decoder="wdet", metacluster=12)   # worked example (panel d): strongest whisker-detection cluster of MC 12


RERUN = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-decoding-rerun-202611")
PSEUDO_SRC = {"wdet": "detection/whisker_active", "adet": "detection/auditory_active", "wva": "arrival/active_final_n200",
              "move": "detection/movement_jaw_active"}     # 2026-11-26: jaw-aligned FA vs CR (THE movement decoder)
WIDE_GRID = {"move": (-1.0, 0.5)}                       # wide grid of each pseudo-population run (default -0.2..0.6 s)
# baseline-referenced significance onsets (user 2026-11-26; rules in _sig_onsets_baseline). Baseline windows = bin END
# times (s): zoom grid of the stimulus decoders, wide grid of the stimulus decoders, wide grid of the jaw movement run
BL = dict(zoom=(-0.020, 0.000), wide=(-0.200, -0.010), move=(-1.000, -0.500))
SEARCH = dict(stim=0.0, move=-0.500)                    # stimulus: first bin ending after 0; movement: from -0.5 s


def _pseudo_D(folder, area):
    """iterations x bins of accuracy - null mean (merged raw file, N = 200, area-group level)"""
    import numpy as np
    import pandas as pd
    f = RERUN / folder / "raw" / f"area_group__{area.replace(' ', '_')}__N200.parquet"
    if not f.exists():
        return None
    r = pd.read_parquet(f, columns=["curve", "null_mean", "skipped_reason"])
    r = r[r.skipped_reason.isna()]
    return np.stack(r.curve.map(np.asarray)).astype(float) - np.stack(r.null_mean.map(np.asarray)).astype(float)


def _run_onset(t, sig, span, need, start, strict):
    import numpy as np
    ok = (t > start + 1e-9) if strict else (t >= start - 1e-9)
    for i in np.where(ok & sig)[0]:
        if i + span > len(t):
            return np.nan
        if sig[i:i + span].sum() >= need:
            return float(t[i])
    return np.nan


def _sig_onsets_baseline(areas):
    """Baseline-referenced significance onsets (user 2026-11-26), cached in Figures/_cache/fig3_sig_onsets_baseline.json
    (the previous rule stays in fig3_sig_onsets.json):
    pseudo-population decoders (N = 200): per iteration i, d_i(t) = accuracy_i(t) - null_i(t) and
      D_i(t) = d_i(t) - mean of d_i over the baseline bins; bin significant if the 5th percentile over iterations of
      D_i(t) > 0; onset = first significant bin at/after the search start with >= 11 of the 13 bins from it significant
      (zoom grid, 2-ms steps) or >= 4 of 5 (wide grid, 5-ms steps). Baselines (bin end times): stimulus decoders zoom grid
      [-20, 0] ms (window -40..0 ms), wide grid [-200, -10] ms; movement (jaw, wide grid) [-1000, -500] ms. Search:
      stimulus = first bin ending after 0; movement = from -500 ms. Stimulus decoders report the zoom-grid onset (and
      '<key>_wide' the wide-grid one); movement the wide-grid one.
    hit vs miss (per-session area-group decoders, rerun 024 rows): per session D_s(t) = d_s(t) - mean over bins ending in
      [-200, -10] ms; 042 cluster_latency (cluster mass, mouse-block sign-flip null, 2000 permutations, alpha 0.05) on
      D_s restricted to bins ending after 0 (hm_onsets.py --baseline)."""
    import json as _json
    import numpy as np
    cache = CACHE / "fig3_sig_onsets_baseline.json"
    keys = list(PSEUDO_SRC) + ["hm"]
    if cache.exists():
        got = _json.load(open(cache))
        if all(f"{k}|{a}" in got for k in keys for a in areas):
            return got
    zoom_t = np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 3)
    out = {}
    for key, folder in PSEUDO_SRC.items():
        w0, w1 = WIDE_GRID.get(key, (-0.2, 0.6))
        wide_t = np.round(np.arange(w0, w1 + 1e-9, 0.005), 3)
        for a in areas:
            D = _pseudo_D(folder, a)
            if D is None:
                out[f"{key}|{a}"] = None
                continue
            if key == "move":
                grids = [("wide", wide_t, D[:, :len(wide_t)], BL["move"], 5, 4, SEARCH["move"], False)]
            else:
                grids = [("zoom", zoom_t, D[:, len(wide_t):len(wide_t) + len(zoom_t)], BL["zoom"], 13, 11, SEARCH["stim"], True),
                         ("wide", wide_t, D[:, :len(wide_t)], BL["wide"], 5, 4, SEARCH["stim"], True)]
            for g, t, x, bl, span, need, start, strict in grids:
                b = (t >= bl[0] - 1e-9) & (t <= bl[1] + 1e-9)
                dx = x - x[:, b].mean(1, keepdims=True)
                o = _run_onset(t, np.percentile(dx, 5, axis=0) > 0, span, need, start, strict)
                name = key if (key == "move" or g == "zoom") else f"{key}_wide"
                out[f"{name}|{a}"] = None if not np.isfinite(o) else o
    tmp = CACHE / "fig3_hm_onsets_baseline.json"
    hm_py = pathlib.Path.home() / "code/ibl-ai-agent/.venv/bin/python"
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'scripts'}"}
    subprocess.run([str(hm_py), str(HERE / "hm_onsets.py"), "--baseline",
                    str(RERUN / "hitmiss" / "rows" / "hitmiss_stim_area_group"), str(tmp), *areas],
                   check=True, env=env, cwd=str(REPO))
    for a, v in _json.load(open(tmp)).items():
        out[f"hm|{a}"] = v
    _json.dump(out, open(cache, "w"), indent=1)
    return out


def _move_jaw_curves():
    """area-group curves of the jaw-aligned movement decoder (mean over iterations of accuracy - null, wide grid)"""
    import numpy as np
    import pandas as pd
    w0, w1 = WIDE_GRID["move"]
    t = np.round(np.arange(w0, w1 + 1e-9, 0.005), 3)
    rows = []
    for f in sorted((RERUN / PSEUDO_SRC["move"] / "raw").glob("area_group__*__N200.parquet")):
        a = f.name.split("__")[1].replace("_", " ")
        D = _pseudo_D(PSEUDO_SRC["move"], a)
        if D is None:
            continue
        rows.append(pd.DataFrame(dict(decoder="move", area_group=a, t=t, acc_minus_null=D[:, :len(t)].mean(0))))
    return pd.concat(rows, ignore_index=True)


def _sig_onsets(areas):
    """Significance-based onset per decoder x area group, as each project reports it (user 2026-11-25):
    detection / arrival / movement (pseudo-populations, N = 200): bin significant when the 5th percentile over iterations
      of (accuracy - linear-shift null) > 0; onset = first significant bin with >= 11 of the 13 fine bins (2-ms steps)
      from it significant (stimulus decoders, zoom grid, searched after the stimulus) or >= 4 of the 5 wide bins
      (movement, wide grid, searched from -200 ms) -- ssl-stimulus-detection-decoding 002 onset();
    hit vs miss (per-session area-group decoders): earliest cluster-mass-significant cluster (p < 0.05, mouse-block
      sign-flip null, 2000 permutations) -- ssl-whisker-hitmiss-timeresolved-decoding 042 cluster_latency().
    Cached in Figures/_cache/fig3_sig_onsets.json."""
    import glob
    import json as _json
    import numpy as np
    import pandas as pd
    cache = CACHE / "fig3_sig_onsets.json"
    if cache.exists():
        got = _json.load(open(cache))
        if all(f"{k}|{a}" in got for k in list(PSEUDO_SRC) + ["hm"] for a in areas):
            return got

    def onset(t, sig, span, need, after):
        for i in np.where((t > after) & sig)[0]:
            if i + span > len(t):
                return np.nan
            if sig[i:i + span].sum() >= need:
                return float(t[i])
        return np.nan
    out = {}
    for key, folder in PSEUDO_SRC.items():
        w0, w1 = WIDE_GRID.get(key, (-0.2, 0.6))
        wide_t = np.round(np.arange(w0, w1 + 1e-9, 0.005), 3)
        zoom_t = np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 3)
        for a in areas:
            fs = sorted(glob.glob(str(RERUN / folder / "raw_chunks" / f"area_group__{a.replace(' ', '_')}__N200__*.parquet")))
            if not fs:
                out[f"{key}|{a}"] = None
                continue
            r = pd.concat([pd.read_parquet(f, columns=["curve", "null_mean"]) for f in fs], ignore_index=True)
            D = np.stack(r.curve.map(np.asarray)) - np.stack(r.null_mean.map(np.asarray))
            if key == "move":
                x, t, span, need, after = D[:, :len(wide_t)], wide_t, 5, 4, -0.2
            else:
                x, t, span, need, after = D[:, len(wide_t):len(wide_t) + len(zoom_t)], zoom_t, 13, 11, 0.0
            sig = np.percentile(x, 5, axis=0) > 0
            o = onset(t, sig, span, need, after)
            out[f"{key}|{a}"] = None if not np.isfinite(o) else o
    # hit vs miss: project 042 cluster-mass latency, run in the hit/miss project's environment (brainwidemap)
    tmp = CACHE / "fig3_hm_onsets.json"
    hm_py = pathlib.Path.home() / "code/ibl-ai-agent/.venv/bin/python"
    env = {**os.environ, "PYTHONPATH": f"{REPO}:{REPO / 'scripts'}"}
    subprocess.run([str(hm_py), str(HERE / "hm_onsets.py"), str(RERUN / "hitmiss" / "rows" / "hitmiss_stim_area_group"),
                    str(tmp), *areas], check=True, env=env, cwd=str(REPO))
    for a, v in _json.load(open(tmp)).items():
        out[f"hm|{a}"] = v
    _json.dump(out, open(cache, "w"), indent=1)
    return out


def _c_common():
    import numpy as np
    import pandas as pd
    import matplotlib.colors as mc
    import ephys_utilities.allen_utils.allen_utils as allen_utils
    sys.path.insert(0, str(HERE.parents[1]))
    from common import snapshot_dir
    fam, size, md, pms = families()
    pal = pms.family_palette(fam.values(), display_order=[fam[k] for k in range(len(fam))])
    R = pms.load_rastermap(md)
    cur = pd.read_csv(C_SRC / "option_iv_area_curves.csv")
    cur = pd.concat([cur[cur.decoder != "move"], _move_jaw_curves()], ignore_index=True)   # 2026-11-26: jaw FA vs CR
    # population information onset per decoder (wide grid, same resolution as the weight curves): first time the mean
    # over the shown area groups reaches half its maximum -- the single time reference of the dot colours (user)
    n_area = pd.Series(R["area_group_arr"]).value_counts()
    top = [a for a in n_area.index if a in set(cur.area_group)][:N_AREAS]
    sig = _sig_onsets_baseline(top)          # 2026-11-26: baseline-referenced onsets (old rule: _sig_onsets)
    # decoder information onset (single time reference of the dot colours) = earliest significant area-group onset
    pop = {}
    for key in set(cur.decoder):
        v = [sig.get(f"{key}|{a}") for a in top]
        v = [x for x in v if x is not None]
        pop[key] = float(min(v)) if v else np.nan
    out = snapshot_dir(spec()) / "data"
    out.mkdir(parents=True, exist_ok=True)
    return dict(np=np, pd=pd, mc=mc, fam=fam, pal=pal, R=R, cur=cur, top=top, n_area=n_area, pop=pop, out=out, sig=sig,
                acol=allen_utils.get_custom_area_groups_colors(), order=allen_utils.get_area_group_custom_order())


def compute_c():
    """Fig3c: accuracy - null over time per area group (the N_AREAS most sampled), one row per decoder; whisker /
    auditory detection and whisker vs auditory on the zoom grid (2-ms steps, 20-ms bins, -20..100 ms; user: first 50 ms),
    hit vs miss and movement on the wide grid (5-ms steps, 50-ms bins)."""
    c = _c_common(); pd = c["pd"]
    cur = c["cur"].assign(grid="wide")
    zoom = pd.read_csv(C_SRC / "option_iv_area_curves_zoom.csv").assign(grid="zoom")
    df = pd.concat([cur[~cur.decoder.isin(ZOOM_DEC)], zoom[zoom.decoder.isin(ZOOM_DEC)]], ignore_index=True)
    df = df[df.area_group.isin(c["top"])].copy()
    df["area_color"] = df.area_group.map(lambda a: c["acol"].get(a, "#aaaaaa"))
    df["area_rank"] = df.area_group.map(lambda a: c["order"].index(a) if a in c["order"] else 99)
    df["area_n_neurons"] = df.area_group.map(c["n_area"])
    df["pop_onset_s"] = df.decoder.map(c["pop"])
    df["sig_onset_s"] = [c["sig"].get(f"{k}|{a}") for k, a in zip(df.decoder, df.area_group)]
    df.to_csv(c["out"] / "Fig3c_area_decoding.csv", index=False)
    print("c ->", c["out"] / "Fig3c_area_decoding.csv", "population onsets", {k: round(v, 3) for k, v in c["pop"].items()})


def compute_e():
    """Fig3e: per fine cluster and decoder, peak weight z (vs within-session label permutation) and the half-rise time of
    its weight-z curve relative to the decoder's population information onset; metacluster at META_CUT + strip colour."""
    c = _c_common()
    # 2026-11-26: the 2000-shuffle stats of explore_e_proposal.py (as panel D; movement = jaw run); display rule unchanged
    # (filled if z >= 3) until the user decides on the FDR rule
    dots = c["pd"].read_csv(E_SRC / "option_e_cluster_stats.csv").drop(columns=["metacluster", "onset_rel_s", "ref_s"])
    dots["metacluster"] = dots.cluster.map(c["fam"])
    dots["mc_color"] = dots.metacluster.map(lambda f: c["mc"].to_hex(c["pal"][f]))
    dots["pop_onset_s"] = dots.decoder.map(c["pop"])
    dots["onset_rel_s"] = dots.onset_half_s - dots.pop_onset_s
    mc_size = c["pd"].Series(families()[1]).groupby(c["pd"].Series(c["fam"])).sum()
    dots["in_b"] = dots.metacluster.isin(set(mc_size.sort_values(ascending=False).index[:N_EXAMPLES].astype(int)))
    dots.to_csv(c["out"] / "Fig3e_cluster_weights.csv", index=False)
    print("e ->", c["out"] / "Fig3e_cluster_weights.csv")


E_SRC = FIGROOT / "_explore" / "Figure3_E_proposal_2026-11-26"   # explore_e_proposal.py (2000 shuffles, zoom-grid z)


def compute_d():
    """Fig3d (user 2026-11-26: explain the weight z; recommendation agreed): one example cluster (the strongest
    whisker-detection cluster of MC 12) --
      schematic: its cluster weight (wide grid, 25-ms smoothed) vs 80 same-size shuffled 'clusters' (labels shuffled
        within session), and its peak vs the 2000 shuffled peaks -> z (dot size in E);
      pair: its mean response (even trials, whisker hit) raw and averaged in the decoder's own causal 20-ms bins
        (labelled at the bin end, 2-ms steps), above its weight z on the same zoom grid (20-ms bins, 2-ms steps).
    Long CSV: part, series, t, y + scalar columns."""
    import json as _json
    import numpy as np
    c = _c_common(); pd = c["pd"]
    st = pd.read_csv(E_SRC / "option_e_cluster_stats.csv")
    k = int(pd.read_csv(E_SRC / "option_e_example_null_peaks.csv").cluster.iloc[0])
    r = st[(st.decoder == EXAMPLE["decoder"]) & (st.cluster == k)].iloc[0]
    parts = []
    cur = pd.read_csv(E_SRC / "option_e_example_curve.csv")
    parts.append(pd.DataFrame(dict(part="obs_curve", series="obs", t=cur.t, y=cur.obs_s)))
    nc = pd.read_csv(E_SRC / "option_e_example_null_curves.csv")
    for col in [x for x in nc.columns if x.startswith("s")]:
        parts.append(pd.DataFrame(dict(part="null_curve", series=col, t=nc.t, y=nc[col])))
    npk = pd.read_csv(E_SRC / "option_e_example_null_peaks.csv")
    parts.append(pd.DataFrame(dict(part="null_peak", series="null", t=np.nan, y=npk.null_peak)))
    zt = pd.read_parquet(E_SRC / "option_e_z_time.parquet")
    zz = zt[(zt.decoder == EXAMPLE["decoder"]) & (zt.cluster == k) & (zt.grid == "zoom")]
    parts.append(pd.DataFrame(dict(part="z_zoom", series="z", t=zz.t, y=zz.z_smooth)))
    zw = zt[(zt.decoder == EXAMPLE["decoder"]) & (zt.cluster == k) & (zt.grid == "wide")]
    # cluster mean response (rastermap PSTHs, even trials, 5-ms bins centred) raw and in the decoder's causal bins
    import rastermap_psth.rastermap_utils as rmu
    fam, size, md, pms = families()
    R = pms.load_rastermap(md)
    prov = _json.load(open(V2 / VARIANT / "variant_provenance.json"))
    m045 = importlib.import_module("045_metacluster_description")
    cfg = rmu.load_cfg(m045.CONFIG, **prov["config_overrides"])
    (rmu.CONDITIONS, rmu.COND_LABELS, rmu.COND_COLORS, rmu.COND_LABELS_MATRIX, rmu.COND_ALIGN_COLS) = rmu.get_conditions(cfg)
    off = np.r_[0, np.cumsum(R["n_bins_list"])]
    rows = R["labels"] == k
    zoom_t = np.round(np.arange(-0.02, 0.1 + 1e-9, 0.002), 3)
    lag = {}
    for name in ("Whisker hit", "Whisker miss"):
        i = list(rmu.COND_LABELS).index(name)
        tc = R["t_ctrs"][i]; y = R["X"][rows, off[i]:off[i + 1]].mean(0)
        parts.append(pd.DataFrame(dict(part="psth_raw", series=name, t=tc, y=y)))
        t1 = np.round(np.arange(tc[0], tc[-1] + 1e-9, 0.001), 3)
        y1 = np.interp(t1, tc, y)
        box = lambda T, w: np.array([y1[(t1 > x - w + 1e-9) & (t1 <= x + 1e-9)].mean() for x in T])
        parts.append(pd.DataFrame(dict(part="psth_dec", series=name, t=zoom_t, y=box(zoom_t, 0.020))))
        if name == "Whisker hit":
            wide_t = np.round(np.arange(-0.05, 0.3 + 1e-9, 0.005), 3)
            sel = (tc >= -0.02) & (tc <= 0.1)
            lag = dict(psth_raw_peak_ms=float(tc[sel][np.argmax(y[sel])] * 1000),
                       psth_box20_peak_ms=float(zoom_t[np.argmax(box(zoom_t, 0.020))] * 1000),
                       z_zoom_peak_ms=float(zz.t.iloc[int(np.argmax(zz.z_smooth.to_numpy()))] * 1000),
                       psth_box50_peak_ms=float(wide_t[np.argmax(box(wide_t, 0.050))] * 1000),
                       z_wide_peak_ms=float(zw.t.iloc[int(np.argmax(zw.z_smooth.to_numpy()))] * 1000))
    out = pd.concat(parts, ignore_index=True).assign(
        cluster=k, metacluster=EXAMPLE["metacluster"], mc_color=c["mc"].to_hex(c["pal"][EXAMPLE["metacluster"]]),
        n_neurons=int(r.n), peak_z=float(r.peak_z), peak_obs=float(r.peak_obs), n_null_ge=int(r.n_null_ge),
        p=float(r.p), n_perm=int(len(npk)))
    out.to_csv(c["out"] / "Fig3d_weight_example.csv", index=False)
    print("d ->", c["out"] / "Fig3d_weight_example.csv", "cluster", k, "peak z", round(float(r.peak_z), 1), "lags", lag)


def compute_d_v1():
    """previous D (2026-11-25): weight-z time course of the example cluster with peak / half-rise annotations"""
    c = _c_common(); pd = c["pd"]
    dots = pd.read_csv(C_SRC / "option_iv_cluster_dots.csv")
    dots["metacluster"] = dots.cluster.map(c["fam"])
    d = dots[(dots.decoder == EXAMPLE["decoder"]) & (dots.metacluster == EXAMPLE["metacluster"])]
    k = int(d.sort_values("peak_z", ascending=False).cluster.iloc[0])
    zt = pd.read_parquet(C_SRC / "option_iv_cluster_z_time.parquet")
    zt = zt[(zt.decoder == EXAMPLE["decoder"]) & (zt.cluster == k)].copy()
    r = d[d.cluster == k].iloc[0]
    zt = zt.assign(metacluster=EXAMPLE["metacluster"], mc_color=c["mc"].to_hex(c["pal"][EXAMPLE["metacluster"]]),
                   n_neurons=int(r.n), peak_z=float(r.peak_z), onset_half_s=float(r.onset_half_s),
                   pop_onset_s=c["pop"][EXAMPLE["decoder"]],
                   onset_rel_s=float(r.onset_half_s) - c["pop"][EXAMPLE["decoder"]])
    zt.to_csv(c["out"] / "Fig3d_weight_example.csv", index=False)
    print("d ->", c["out"] / "Fig3d_weight_example.csv", "cluster", k, "peak z", round(float(r.peak_z), 1))


if __name__ == "__main__":
    which = sys.argv[1:] or ["a", "b"]
    for w in which:
        f, *arg = w.split(":")                            # e.g. b:2 = two metaclusters per row
        globals()[f"compute_{f}"](*arg)
