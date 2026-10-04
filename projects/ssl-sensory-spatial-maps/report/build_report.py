"""Build the sensory-maps + stimulus-arrival report (Quarto source) from the result tables (run on haas).
Writes combined_results_ks4/_sensory_spatial_maps/report/sensory_maps_report.qmd and the list of figures it uses
(report_figures.txt); render.sh copies them and renders the PDF locally (Quarto / typst)."""
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
SM, AR_ACT, AR_PAS = RES / "_sensory_spatial_maps", RES / "_stimulus_arrival", RES / "_stimulus_arrival_passive"
REPO = pathlib.Path.home() / "code" / "ibl-ai-agent" / "projects"
sys.path.insert(0, str(REPO / "ssl-sensory-spatial-maps" / "exploratory-analyses"))
OUTD = SM / "report"
FIGS = []


def fig(src, name, caption, width="100%"):
    FIGS.append((str(src), name))
    return f"![{caption}](fig/{name}){{width={width}}}\n"


def p_txt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def short(a):
    return {"Somatosensory-whisker": "SS-whisker", "Somatosensory-orofacial": "SS-orofacial", "Somatosensory-body": "SS-body",
            "Auditory areas": "Auditory", "Motor areas": "Motor", "Frontal areas": "Frontal", "Retrosplenial areas": "Retrosplenial",
            "Posterior parietal areas": "Posterior parietal", "Lateral septal complex": "Lateral septum", "Visual areas": "Visual",
            "Insular areas": "Insular", "Olfactory areas": "Olfactory", "Amygdala and hypothalamus": "Amygdala + hypothalamus"}.get(a, a)


def onset_table(OB, W, level):
    q = OB[OB.level == level].copy()
    q["key"] = q.onset_ms.fillna(1e9)
    q = q.sort_values(["key", "area"])
    w = W[W.level == level].pivot_table(index="area", columns="N", values="mean")
    rows = ["| Area | onset N = 200 (ms) | early accuracy N = 100 / 200 / 500 | eligible sessions |", "|---|---|---|---|"]
    for r in q.itertuples():
        on = f"{r.onset_ms:.0f} ({r.lo:.0f}-{r.hi:.0f})" if np.isfinite(r.onset_ms) else "n.s."
        acc = " / ".join(f"{w.loc[r.area, n]:.2f}" if r.area in w.index and n in w.columns and np.isfinite(w.loc[r.area, n]) else "-"
                         for n in (100, 200, 500))
        rows.append(f"| {short(r.area)} | {on} | {acc} | {r.n_eligible_sessions} |")
    return "\n".join(rows) + "\n"


def decoding_section(base, title, epoch_word):
    OB = pd.read_csv(base / "onset_bootstrap_N200.csv")
    W = pd.read_csv(base / "window_accuracy.csv")
    st = pd.read_csv(base / "onset_vs_window_stats.csv")
    rho = st[st.resolution == "zoom"].set_index("level").rho
    g = OB[OB.level == "area_group"].dropna(subset=["onset_ms"]).sort_values("onset_ms")
    f = OB[OB.level == "area_acronym_custom"].dropna(subset=["onset_ms"]).sort_values("onset_ms")
    txt = f"## {title}\n\n"
    txt += (f"With 200-neuron pseudo-populations ({epoch_word}), the first significant 20-ms bin ends at "
            + ", ".join(f"{r.onset_ms:.0f} ms in {short(r.area)}" for r in g.head(4).itertuples())
            + f" (area groups; last: {short(g.iloc[-1].area)}, {g.iloc[-1].onset_ms:.0f} ms). Among areas, the earliest are "
            + ", ".join(f"{r.area} ({r.onset_ms:.0f} ms)" for r in f.head(5).itertuples())
            + f". Onset and early accuracy are related across areas and neuron counts (Spearman rho = {rho.get('area_group', np.nan):.2f} "
            f"for area groups, {rho.get('area_acronym_custom', np.nan):.2f} for areas).\n\n")
    n_ng = OB[(OB.level == "area_group")].onset_ms.isna().sum()
    n_nf = OB[(OB.level == "area_acronym_custom")].onset_ms.isna().sum()
    txt += f"Area groups without a significant onset: {n_ng}; areas: {n_nf}.\n\n**Area groups.**\n\n" + onset_table(OB, W, "area_group")
    txt += "\n**Areas (40 best-sampled).**\n\n" + onset_table(OB, W, "area_acronym_custom") + "\n"
    P = base / "matched_n.csv"
    if P.exists():
        M = pd.read_csv(P)
        M = M[(M.reference == "Somatosensory-whisker") & (M.level == "area_group")].sort_values("matched_N")
        txt += ("Neurons each area group needs to reach the early accuracy of whisker somatosensory cortex at 100 neurons: "
                + ", ".join(f"{short(r.area)} {r.matched_N:.0f}" if np.isfinite(r.matched_N) else f"{short(r.area)} > 2000"
                            for r in M.itertuples()) + ".\n\n")
    tag = "passive_" if "passive" in epoch_word else ""
    txt += fig(base / "figures" / "arrival_summary_area_group.png", f"{tag}arrival_summary_area_group.png",
               f"**Where and when can stimulus modality be decoded? Area groups, {epoch_word}.** a, Method. b, Corrected "
               "accuracy over the first 50 ms (N = 200), rows sorted by onset; grey: not above chance; tick: onset. c, Onset "
               "ranking (95 % range). d, Early accuracy vs number of neurons. e, Onset vs early accuracy (area x N). f, Matched "
               "early accuracy (or best-sampled curves). g, Real and shuffled balanced accuracy. h, Time course with bars "
               "marking the bins above chance.")
    txt += fig(base / "figures" / "arrival_summary_area_acronym_custom.png", f"{tag}arrival_summary_area_acronym_custom.png",
               f"Same for the 40 best-sampled areas ({epoch_word}).")
    txt += fig(base / "figures" / "arrival_main_N200_area_group.png", f"{tag}arrival_main_N200_area_group.png",
               f"**Main figure, area groups ({epoch_word}, N = 200).** Heatmaps of the whole time course and the first 50 ms "
               "(rows sorted by onset), onset ranking, and time courses of the 8 best-sampled groups with significance bars.")
    txt += fig(base / "figures" / "arrival_main_N200_area_acronym_custom.png", f"{tag}arrival_main_N200_area_acronym_custom.png",
               f"Main figure for the areas ({epoch_word}).")
    return txt, OB


def numbers(U, ses, fr, n_r, n_b, lat, LG, PZ, OVS, OVS70, OV, T70, T90, MC, nrec, OBa, OBp):
    """every number quoted in the deck -> report/report_numbers.json (deck/build_deck.py reads it; nothing hard-coded)"""
    from scipy import stats

    def coloc(T):
        return dict(P_in=100 * T.P_in, P_ref=100 * T.P_ref, diff=100 * T["diff"], ci_lo=100 * T.diff_ci_lo,
                    ci_hi=100 * T.diff_ci_hi, p=float(T.p_boot), n_in=int(T.n_resp_in))

    def onsets(OB, level):
        q = OB[OB.level == level].sort_values("onset_ms")
        return [[short(r.area), None if not np.isfinite(r.onset_ms) else float(r.onset_ms)] for r in q.itertuples()]
    st = pd.read_csv(AR_ACT / "onset_vs_window_stats.csv")
    rho = st[st.resolution == "zoom"].set_index("level").rho
    M = pd.read_csv(AR_ACT / "matched_n.csv") if (AR_ACT / "matched_n.csv").exists() else pd.DataFrame()
    if len(M):
        M = M[(M.reference == "Somatosensory-whisker") & (M.level == "area_group")]
    link = {}
    lf = SM / "deck" / "link_latency_onset.csv"
    if lf.exists():
        K = pd.read_csv(lf)
        r, p = stats.spearmanr(K["first"], K.onset_ms)
        link = dict(rho=float(r), p=float(p), n=len(K))
    sig = MC[MC.p_distance_holm < 0.05]
    N = dict(
        n_neurons=len(U), n_sessions=int(ses.session_id.nunique()), n_mice=int(U.mouse_id.nunique()), n_structures=nrec,
        resp=dict(whisker=fr["whisker_active"][:2], auditory=fr["auditory_active"][:2],
                  modality=[fr["wh_vs_aud_active"][0], fr["wh_vs_aud_active"][1], fr["wh_vs_aud_active"][2]]),
        bimodal_pct=100 * n_b / n_r, latency_median=dict(whisker=float(lat["whisker"].median()), auditory=float(lat["auditory"].median())),
        latency_groups=[[short(a), float(r.latency_whisker_ms), float(r.latency_auditory_ms)] for a, r in LG.sort_values("latency_whisker_ms").iterrows()],
        zones=dict(sources=[[r.source, int(r.n_experiments), int(r.n_lines)] for r in PZ.itertuples()],
                   overlap90=OVS["overlap_mm3"], overlap70=OVS70["overlap_mm3"], whisker90=OVS["whisker_union_mm3"],
                   auditory90=OVS["auditory_union_mm3"],
                   pieces=list(OV[~OV.structure.isin(["root", "grey", "MB", "TH", "HY", "CTX"])].head(6).structure)),
        coloc90=coloc(T90), coloc70=coloc(T70),
        modality_offset=dict(n_sig=len(sig), n=len(MC), dmin=float(sig.centroid_distance_um.min()) if len(sig) else None,
                             dmax=float(sig.centroid_distance_um.max()) if len(sig) else None),
        arrival=dict(groups=onsets(OBa, "area_group"), areas=onsets(OBa, "area_acronym_custom"),
                     rho_groups=float(rho.get("area_group", np.nan)), rho_areas=float(rho.get("area_acronym_custom", np.nan)),
                     matched=[[short(r.area), None if not np.isfinite(r.matched_N) else float(r.matched_N)] for r in M.itertuples()],
                     passive_groups=onsets(OBp, "area_group") if OBp is not None else None),
        link=link, generated=f"{pd.Timestamp.now():%Y-%m-%d %H:%M}")
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "report_numbers.json").write_text(json.dumps(N, indent=1, default=float))


def main():
    m3 = importlib.import_module("003_spatial_maps")
    U = m3.load_units()
    ses = U.drop_duplicates("session_id")
    fr = {}
    for q in ("whisker_active", "auditory_active", "wh_vs_aud_active"):
        s, v = U[f"sig_{q}"], U[f"sel_{q}"]
        ok = s.notna()
        fr[q] = (100 * (s[ok] == 1).mean(), 100 * ((s == 1) & (v > 0))[ok].mean(), 100 * ((s == 1) & (v < 0))[ok].mean())
    c = U.bimodal_cat
    n_w, n_a, n_b = int((c == 1).sum()), int((c == 2).sum()), int((c == 3).sum())
    n_r = n_w + n_a + n_b
    L = pd.read_parquet(SM / "unit_latency.parquet")
    lat = {m: L[f"latency_{m}_ms"].dropna() for m in ("whisker", "auditory")}
    LG = U.groupby("area_group")[["latency_whisker_ms", "latency_auditory_ms"]].median().dropna(how="all")
    # 90 % projection zones are the main result, 70 % the stricter supplementary version (user 2026-10-04)
    PZ = pd.read_csv(SM / "projection_zones_summary_zone90.csv")
    OVS = json.load(open(SM / "projection_overlap_summary_zone90.json"))
    OVS70 = json.load(open(SM / "projection_overlap_summary.json"))
    OV = pd.read_csv(SM / "projection_overlap_zone90.csv")
    OV = OV[OV.recorded] if "recorded" in OV else OV
    OV = OV[~OV.structure.isin(["MB", "TH", "HY", "CTX", "grey"])]           # generic remainder labels
    T70 = pd.read_csv(SM / "colocation_tests.csv").query("kind == 'global'").iloc[0]
    T90 = pd.read_csv(SM / "colocation_tests_zone90.csv").query("kind == 'global'").iloc[0]
    MC = pd.read_csv(SM / "modality_contours_zone90.csv")
    nrec = int(pd.read_csv(SM / "recorded_structures.csv").recorded.sum())
    nflat = len(pd.read_parquet(SM / "flatmap_units_zone90.parquet", columns=["session_id"]))
    FZ = SM / "figures_zone90"                                                     # figures drawn with the 90 % zones
    act, OBa = decoding_section(AR_ACT, "Task (active) trials", "task (active) trials")
    has_pas = (AR_PAS / "onset_bootstrap_N200.csv").exists()
    pas, OBp = decoding_section(AR_PAS, "Passive trials", "passive trials") if has_pas else ("", None)
    ga = OBa[OBa.level == "area_group"].dropna(subset=["onset_ms"]).sort_values("onset_ms")
    lat_sw, lat_mb = LG.loc["Somatosensory-whisker"], LG.loc["Midbrain"]
    sig_mc = MC[MC.p_distance_holm < 0.05]
    numbers(U, ses, fr, n_r, n_b, lat, LG, PZ, OVS, OVS70, OV, T70, T90, MC, nrec, OBa, OBp)
    lines = []
    A = lines.append
    A(f"""---
title: "Whisker and auditory responses across the mouse brain: where, when, and where the two streams converge"
subtitle: "SSL dataset (Neuropixels, KS4) -- report generated {pd.Timestamp.now():%Y-%m-%d %H:%M}"
format:
  typst:
    papersize: a4
    margin:
      x: 1.8cm
      y: 2cm
    fontsize: 9.5pt
    toc: true
    toc-depth: 2
    section-numbering: "1.1"
---

# Key conclusions

**Sensory coding**

1. **Sensory responses are widespread but spatially organised.** Of {len(U):,} good and multi-unit neurons ({ses.session_id.nunique()}
   sessions, {U.mouse_id.nunique()} mice), {fr['whisker_active'][0]:.1f} % respond to the whisker stimulus and {fr['auditory_active'][0]:.1f} % to
   the auditory stimulus in the active task (5-35 ms after onset); {fr['wh_vs_aud_active'][0]:.1f} % prefer one modality (Where).
2. **Each modality reaches its own sensory system first.** Median half-time to peak: whisker {lat_sw.latency_whisker_ms:.1f} ms in whisker
   somatosensory cortex (auditory there {lat_sw.latency_auditory_ms:.1f} ms); auditory {lat_mb.latency_auditory_ms:.1f} ms in the midbrain
   (whisker there {lat_mb.latency_whisker_ms:.1f} ms) (When).
3. **Stimulus modality can be decoded within {ga.onset_ms.min():.0f}-{ga.onset_ms.max():.0f} ms of stimulus onset (area groups, task trials)**, earliest in
   {', '.join(short(a) for a in ga[ga.onset_ms == ga.onset_ms.min()].area)}; faster areas also carry more early information (Population coding).""")
    if has_pas:
        gp = OBp[OBp.level == "area_group"].dropna(subset=["onset_ms"]).sort_values("onset_ms")
        A(f"""4. **Passive trials (no task, no licks)** give onsets of {gp.onset_ms.min():.0f}-{gp.onset_ms.max():.0f} ms, earliest in
   {', '.join(short(a) for a in gp[gp.onset_ms == gp.onset_ms.min()].area)} (Population coding, passive).""")
    A(f"""
**Convergence**

5. **Whisker and auditory cortex project to partly overlapping targets.** The 90 % projection zones of whisker cortex
   ({OVS['whisker_union_mm3']:.1f} mm³) and auditory cortex ({OVS['auditory_union_mm3']:.1f} mm³) overlap in {OVS['overlap_mm3']:.1f} mm³ (70 % zones:
   {OVS70['overlap_mm3']:.1f} mm³), mainly in {', '.join(OV[~OV.structure.isin(["root", "grey", "MB", "TH", "HY", "CTX"])].head(6).structure)}.
6. **Bimodal neurons are enriched where these projections converge.** {100 * n_b / n_r:.1f} % of sensory-responsive neurons respond to both
   modalities; inside the overlap (90 % zones) {100 * T90.P_in:.1f} % vs {100 * T90.P_ref:.1f} % of all responsive neurons ({100 * T90['diff']:+.1f} points,
   95 % CI {100 * T90.diff_ci_lo:+.1f} to {100 * T90.diff_ci_hi:+.1f}; hierarchical bootstrap {p_txt(T90.p_boot)}); with the stricter 70 % zones
   {100 * T70['diff']:+.1f} points ({p_txt(T70.p_boot)}). A co-location, not evidence of causation.
7. **Within shared targets, whisker- and auditory-preferring neurons are spatially offset** in {len(sig_mc)} of {len(MC)} target slabs
   (within-session permutation, Holm p < 0.05).

**Notes.** Pseudo-population iterations (100 x 10 shuffles) are pilot values. All analyses pool both cohorts and both stages.

# Data

- **Recordings.** Neuropixels, Kilosort 4 (`NWB_ks4`); {ses.session_id.nunique()} whisker-training sessions ({(ses.stage == 'learning').sum()} learning-day,
  {(ses.stage == 'expert').sum()} expert; {(ses.cohort == 'R+').sum()} R+ and {(ses.cohort == 'R-').sum()} R- sessions; {U.mouse_id.nunique()} mice); inclusion and cohort labels
  from the dataset record (`skills/ssl-valid-data`).
- **Neurons.** Good and multi-unit clusters (v2 unit table with drift test), {len(U):,} neurons; CCF positions folded onto one
  hemisphere; custom area groups / areas (`allen_utils`); {nrec} structures recorded (>= 10 neurons).
- **Trials.** Active: context active, `perf` != 6, auditory warm-up removed (trial before the first whisker trial kept), end-of-session
  disengagement trimmed (rule A1). Passive: fixed ~3 s ITI sequence or labelled passive (pre- and post-task blocks). Unlabelled trials
  in a session with context labels are active.
- **Whisker artefact.** Spikes in -10 to +5 ms around every whisker onset replaced by a Poisson train at the pre-onset rate.
- **Maps.** Density $\\rho_q(x) = (G_\\sigma * \\sum_i q_i \\delta_{{x_i}})(x) \\,/\\, (G_\\sigma * \\sum_i \\delta_{{x_i}})(x)$ (Gaussian, sigma 150 µm;
  normalised by the recorded-neuron density), shown where >= 3 neurons fall within the kernel. Isocortex flatmap: Allen CCFv3 butterfly
  flatmap (cortical streamlines; geodesic embedding to two pairs of anchor points; Wang et al. 2020, Harris et al. 2019) via
  `ccf_streamlines`; {nflat:,} isocortex neurons placed at their closest streamline; left hemisphere, anterior up; area not preserved
  (numbers computed in 3-D).

# Where: single-neuron sensory responses

## Responsiveness and modality preference (rate-based ROC)

Spike counts in the response window (5-35 ms) and the baseline window (-1 s to -15 ms) give $\\mathrm{{sel}} = 2\\,\\mathrm{{AUC}}(A, B) - 1$
(positive when B is higher); significance from 1000 label permutations, one-sided, p < 0.05.

| Quantity | A vs B | positive | significant | positive / negative |
|---|---|---|---|---|
| whisker responsiveness | baseline vs whisker (active) | excited | {fr['whisker_active'][0]:.1f} % | {fr['whisker_active'][1]:.1f} / {fr['whisker_active'][2]:.1f} % |
| auditory responsiveness | baseline vs auditory (active) | excited | {fr['auditory_active'][0]:.1f} % | {fr['auditory_active'][1]:.1f} / {fr['auditory_active'][2]:.1f} % |
| modality preference | whisker vs auditory (active) | auditory-preferring | {fr['wh_vs_aud_active'][0]:.1f} % | {fr['wh_vs_aud_active'][1]:.1f} / {fr['wh_vs_aud_active'][2]:.1f} % |

## Bimodal neurons

Responsive to modality $m$ if any stimulus-vs-baseline test in the $k_m$ epochs of the session (active, passive pre, passive post) is
significant after Bonferroni correction: $R_m = \\exists\\, e:\\; p_{{m,e}} < 0.05 / k_m$. Bimodal: $R_\\text{{whisker}} \\wedge R_\\text{{auditory}}$.
Of {n_r:,} responsive neurons, {n_w:,} respond to whiskers only, {n_a:,} to sound only and {n_b:,} to both ($B$ = {100 * n_b / n_r:.1f} %).

## Maps

""")
    A(fig(SM / "figures" / "cortical_flatmaps_nozones.png", "cortical_flatmaps_nozones.png",
          "**Sensory responses across the isocortex** (Allen butterfly flatmap). Top: significant neurons coloured by the quantity "
          "(grey number: neurons in colour); bottom: density normalised by recorded-neuron density; first column: recorded neurons and "
          "their density. Projection zones are added in the Convergence section."))
    A(f"""
# When: response latency

$r(t) = s\\,[\\mathrm{{PSTH}}(t) - \\overline{{\\mathrm{{PSTH}}}}_{{[-100,-10]}}]$ (1-ms bins, Gaussian sigma 2 ms, $s$ = sign of the selectivity); latency =
last upward crossing of $r(t_\\text{{peak}})/2$ before $t_\\text{{peak}} = \\arg\\max_{{5 \\le t \\le 100}} r(t)$ (whisker: after +5 ms). Medians:
whisker {lat['whisker'].median():.1f} ms ({len(lat['whisker']):,} neurons), auditory {lat['auditory'].median():.1f} ms ({len(lat['auditory']):,}).
The latency flatmaps are columns 5-6 of the flatmap figure above.

| Area group | whisker (ms) | auditory (ms) |
|---|---|---|
""" + "\n".join(f"| {short(a)} | {r.latency_whisker_ms:.1f} | {r.latency_auditory_ms:.1f} |"
                for a, r in LG.sort_values("latency_whisker_ms").iterrows()) + "\n")
    A("""
# Population coding: when does stimulus information arrive?

**Method.** One iteration: 20 sessions with replacement, $N/20$ neurons of the area per session, pseudo-trials per class by balanced
reuse; L2 logistic regression per time bin (3-fold CV on real trials, inner 2-fold for C); $\\mathrm{BA} = (\\mathrm{TPR} + \\mathrm{TNR})/2$;
corrected accuracy $d(t) = \\mathrm{BA}_\\text{real}(t) - \\frac{1}{10}\\sum_k \\mathrm{BA}_{\\text{shuffle},k}(t)$ (labels shuffled within
sessions); 100 iterations; $N$ = 20-500. Bin above chance: 5th percentile of $d(t)$ > 0. Onset: first $t > 0$ above chance with >= 80 % of
the bins in $[t, t + 25\\,\\text{ms}]$ above chance (20-ms bins, 2-ms steps); 95 % range from 1000 resamples of the iterations. Early accuracy:
mean $d$ over bins ending 5-50 ms. Areas: all 18 area groups and the 40 best-sampled areas.

""" + act + ("\n" + pas if has_pas else "\n*Passive-trial decoding: not finished at report time.*\n"))
    link = SM / "deck" / "img" / "link_latency_onset.png"
    if link.exists():
        A("\n## Single-neuron latency and population onset\n\n")
        A(fig(link, "link_latency_onset.png",
              "**Population decoding onset (N = 200) vs median single-neuron latency** per area group (left: faster modality; right: "
              "whisker); Spearman rho, OLS line with 95 % CI (solid when p < 0.05)."))
    A(f"""
# Convergence: where do the whisker and auditory streams meet?

## Projection anatomy

Allen Mouse Brain Connectivity Atlas: anterograde injections in wild-type mice and every Cre line labelling projection
neurons (interneuron lines excluded; """ +
      ", ".join(f"{r.source} {r.n_experiments} experiments in {r.n_lines} lines" for r in PZ.itertuples()) + """). Per source,
experiments are normalised, averaged within each line $l$ and then over the $K$ lines (no line dominates):
$\\bar D(v) = \\frac{1}{K}\\sum_l \\frac{1}{E_l}\\sum_{e \\in l} d_e(v) / \\sum_{v' \\in C} d_e(v')$, smoothed ($\\tilde D = G_{50\\,\\mu m} * \\bar D$); 90 % zone
$Z = \\{v : \\tilde D(v) \\ge \\tau\\}$ with $\\sum_{Z} \\tilde D = 0.9 \\sum_C \\tilde D$ (main; 70 % as the stricter version); merged whisker
(SSp-bfd + SSs) and auditory (AUDp + AUDd/v) zones; overlap $Z_w \\cap Z_a$. """ + f"""Overlap {OVS['overlap_mm3']:.1f} mm³ (70 %: {OVS70['overlap_mm3']:.1f} mm³); largest recorded pieces: """ +
      ", ".join(f"{r.structure} {r.overlap_mm3:.2f} mm³" for r in OV.head(8).itertuples()) + ".\n\n")
    A(fig(FZ / "projection_zones_coronal.png", "projection_zones_coronal.png",
          "**Projection zones of whisker and auditory cortex**, coronal 500-µm slabs (density, 90 % contours, overlap; right: largest "
          "recorded structures as % of the zone)."))
    A(fig(FZ / "projection_zones_sagittal.png", "projection_zones_sagittal.png", "Same, sagittal slabs."))
    A("\n## Responses relative to the projection zones\n\n")
    A(fig(FZ / "cortical_flatmaps.png", "cortical_flatmaps.png",
          "**The flatmaps of the Where section with the projection zones** (90 % contours of whisker (teal) and auditory (brown dashed) "
          "cortex projections; first column, bottom: whisker zone yellow, auditory zone blue, overlap purple)."))
    A(f"""
## Bimodal neurons and projection overlap

$\\Delta = P_\\text{{in}} - P_\\text{{ref}}$ (bimodal fraction of responsive neurons inside the overlap minus among all); hierarchical bootstrap
over sessions then neurons ($B$ = 2000), $p = (1 + \\#\\{{\\Delta^* \\le 0\\}})/(1 + B)$; Fisher's exact test alongside. Sub-regions described,
not tested. 90 % zones (main): {100 * T90.P_in:.1f} % of {int(T90.n_resp_in):,} vs {100 * T90.P_ref:.1f} % of {int(T90.n_resp_ref):,},
{100 * T90['diff']:+.1f} points (95 % CI {100 * T90.diff_ci_lo:+.1f} to {100 * T90.diff_ci_hi:+.1f}), {p_txt(T90.p_boot)}. 70 % zones (stricter):
{100 * T70.P_in:.1f} % of {int(T70.n_resp_in):,} vs {100 * T70.P_ref:.1f} %, {100 * T70['diff']:+.1f} points ({100 * T70.diff_ci_lo:+.1f} to
{100 * T70.diff_ci_hi:+.1f}), {p_txt(T70.p_boot)}.

""")
    A(fig(FZ / "colocation_figure.png", "colocation_figure.png",
          "**Bimodal neurons and the convergence of whisker- and auditory-cortex projections** (90 % zones). a, zones and sub-regions; "
          "b, responsive / bimodal neurons; c, bimodal fraction; d, inside vs all responsive neurons."))
    A(f"""
## Modality preference within shared targets

80 % highest-density contours of whisker- and auditory-preferring neurons per target slab and area; centroid distance and axis shifts;
labels permuted within sessions (5000), Holm correction.

| Slab / area | whisker / auditory-pref. | sessions | distance (µm) | depth shift A-W (µm) | p (Holm) |
|---|---|---|---|---|---|
""" + "\n".join(f"| {r.slab.split(',')[0]} / {r.area} | {r.n_whisker_pref} / {r.n_auditory_pref} | {r.n_sessions} | {r.centroid_distance_um:.0f} | "
                f"{r.shift_depth_um:+.0f} | {r.p_distance_holm:.3f} |" for r in MC.itertuples()) + "\n")
    A(fig(FZ / "modality_contours.png", "modality_contours.png",
          "**Location of whisker- vs auditory-preferring neurons within areas** (80 % contours, centroids)."))
    sch = SM / "deck" / "img" / "summary_schematic.png"
    if sch.exists():
        A("\n## Summary\n\n")
        A(fig(sch, "summary_schematic.png",
              "**Separate entry routes, shared targets (draft).** a, Whisker route via thalamus to barrel cortex; sound via midbrain and "
              "thalamus to auditory cortex; both cortices project to the shared targets, where bimodal neurons are enriched; times: "
              "population decoding onset (N = 200). Routes: textbook anatomy and the Allen projection zones. b, Onsets with 95 % bootstrap CI."))
    A("""
# Caveats

- Co-location, not causation (tracing from other mice, axons of passage, CCF uncertainty of ~100-200 µm).
- Cohorts and stages are pooled throughout (scope of this report).
- The bimodal enrichment depends on the zone level: clear with the 90 % zones (main), smaller and not significant with the
  stricter 70 % zones.
- Task-trial decoding after ~100 ms can use lick preparation (auditory trials nearly always licked); passive trials and onsets in the
  first 20 ms are unaffected.
- Whisker-trial spikes in -10 to +5 ms are replaced by baseline-rate Poisson spikes: whisker information cannot appear before ~5 ms.
- 100 iterations x 10 shuffles are pilot values (final N = 200 onsets: 1000 x 20, separate run).
""")
    # target-slab figures dropped from the report (user 2026-10-04); the within-target offset test (004) is kept
    OUTD.mkdir(parents=True, exist_ok=True)
    (OUTD / "sensory_maps_report.qmd").write_text("\n".join(lines), encoding="utf-8")
    (OUTD / "report_figures.txt").write_text("\n".join(f"{s}\t{n}" for s, n in FIGS) + "\n")
    print("wrote", OUTD / "sensory_maps_report.qmd", len(FIGS), "figures, passive:", has_pas)


if __name__ == "__main__":
    main()
