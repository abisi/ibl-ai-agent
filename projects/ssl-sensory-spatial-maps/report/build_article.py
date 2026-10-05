"""Article-style report of ssl-sensory-spatial-maps (skills/project-report): run on haas.

Reads the result tables of combined_results_ks4/_sensory_spatial_maps (until the folder migration) and writes, into
combined_results_ks4/ssl-sensory-spatial-maps/report/: report.md (single source), numbers.json (every quoted number),
figures/ (copies, downscaled to <= 2000 px), build.sh (copy of skills/project-report/build.sh). The PDF / HTML are built
with build.sh where TinyTeX is installed (render_article.sh does it locally).
"""
import importlib
import json
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
SM = RES / "_sensory_spatial_maps"
FZ, F70 = SM / "figures_zone90", SM / "figures"
REP = RES / "ssl-sensory-spatial-maps" / "report"
REPO = pathlib.Path.home() / "code" / "ibl-ai-agent"
sys.path.insert(0, str(REPO / "projects" / "ssl-sensory-spatial-maps" / "exploratory-analyses"))
FIGS = []                                   # (source, name)
GENERIC = {"root", "grey", "MB", "TH", "HY", "CTX"}


def P(p):
    return "n/a" if p is None or not np.isfinite(p) else "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def short(a):
    return {"Somatosensory-whisker": "SS-whisker", "Somatosensory-orofacial": "SS-orofacial", "Somatosensory-body": "SS-body",
            "Auditory areas": "Auditory", "Motor areas": "Motor", "Frontal areas": "Frontal", "Retrosplenial areas": "Retrosplenial",
            "Posterior parietal areas": "Posterior parietal", "Lateral septal complex": "Lateral septum", "Visual areas": "Visual",
            "Insular areas": "Insular", "Olfactory areas": "Olfactory", "Amygdala and hypothalamus": "Amygdala + hypothalamus"}.get(a, a)


def fig(src, name, caption, width="100%"):
    name = name.replace(".png", ".jpg")                    # JPEG copies (quality 90): keeps PDF / HTML small
    FIGS.append((pathlib.Path(src), name))
    return f"![{caption}](figures/{name}){{width={width}}}\n"


def caption_md(path):
    """caption text of a *_caption.md file (heading dropped, paragraphs joined)"""
    if not path.exists():
        return ""
    t = path.read_text(encoding="utf-8").split("\n")
    t = [l for l in t if not l.startswith("# ")]
    return " ".join(l.strip() for l in t if l.strip() and not l.startswith("- ") and not l.startswith("Sub-regions"))


def table(df, cols, fmt=None):
    fmt = fmt or {}
    out = ["| " + " | ".join(cols.values()) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        d = r._asdict()
        out.append("| " + " | ".join(fmt[c](d[c]) if c in fmt else str(d[c]) for c in cols) + " |")
    return "\n".join(out) + "\n"


def main():
    m3 = importlib.import_module("003_spatial_maps")
    U = m3.load_units()
    ses = U.drop_duplicates("session_id")
    N = {}
    # responsiveness and preference
    fr = {}
    for q in ("whisker_active", "auditory_active", "wh_vs_aud_active"):
        s, v = U[f"sig_{q}"], U[f"sel_{q}"]
        ok = s.notna()
        fr[q] = dict(sig=(s[ok] == 1).mean(), pos=((s == 1) & (v > 0))[ok].mean(), neg=((s == 1) & (v < 0))[ok].mean(), n=int(ok.sum()))
    c = U.bimodal_cat
    n_w, n_a, n_b = int((c == 1).sum()), int((c == 2).sum()), int((c == 3).sum())
    n_r = n_w + n_a + n_b
    # per area group
    G = U.groupby("area_group").agg(n=("cluster_id", "size"), sessions=("session_id", "nunique"),
                                    wh=("sig_whisker_active", lambda s: (s == 1).sum() / max(s.notna().sum(), 1)),
                                    au=("sig_auditory_active", lambda s: (s == 1).sum() / max(s.notna().sum(), 1)),
                                    lat_w=("latency_whisker_ms", "median"), lat_a=("latency_auditory_ms", "median")).reset_index()
    G = G[G.n >= 200].sort_values("lat_w")
    L = pd.read_parquet(SM / "unit_latency.parquet")
    lat = {m: L[f"latency_{m}_ms"].dropna() for m in ("whisker", "auditory")}
    # anatomy
    PZ = pd.read_csv(SM / "projection_zones_summary_zone90.csv")
    PZ70 = pd.read_csv(SM / "projection_zones_summary.csv")
    E = pd.read_csv(SM / "projection_experiments.csv")
    OV9 = json.load(open(SM / "projection_overlap_summary_zone90.json"))
    OV7 = json.load(open(SM / "projection_overlap_summary.json"))
    OV = pd.read_csv(SM / "projection_overlap_zone90.csv")
    OV = OV[OV.recorded & ~OV.structure.isin(GENERIC)] if "recorded" in OV else OV[~OV.structure.isin(GENERIC)]
    DIV = pd.read_csv(SM / "projection_zone_composition_divisions_zone90.csv")
    T9 = pd.read_csv(SM / "colocation_tests_zone90.csv")
    T7 = pd.read_csv(SM / "colocation_tests.csv")
    g9, g7 = T9[T9.kind == "global"].iloc[0], T7[T7.kind == "global"].iloc[0]
    SR9 = T9[T9.kind == "sub-region"].sort_values("id")
    MC = pd.read_csv(SM / "modality_contours_zone90.csv")
    sig_mc = MC[MC.p_distance_holm < 0.05]
    nrec = int(pd.read_csv(SM / "recorded_structures.csv").recorded.sum())
    nflat = len(pd.read_parquet(SM / "flatmap_units_zone90.parquet", columns=["session_id"]))
    PL = RES / "_roc_prelick_sl" / "projection_colocation"
    PT = pd.read_csv(PL / "prelick_colocation_tests_zone90.csv").set_index("group") if (PL / "prelick_colocation_tests_zone90.csv").exists() else None
    snap = RES / "_snapshots" / "2026-10-04_2200" / "sensory_spatial_maps"
    N.update(n_neurons=len(U), n_sessions=int(ses.session_id.nunique()), n_mice=int(U.mouse_id.nunique()), n_structures=nrec,
             n_flatmap=nflat, resp=fr, n_resp=n_r, n_bimodal=n_b, bimodal_frac=n_b / n_r,
             latency_median=dict(whisker=float(lat["whisker"].median()), auditory=float(lat["auditory"].median())),
             overlap90=OV9, overlap70=OV7, coloc90=g9.to_dict(), coloc70=g7.to_dict(),
             modality_offset=dict(n_sig=len(sig_mc), n=len(MC)))
    sw, mb = G.set_index("area_group").loc["Somatosensory-whisker"], G.set_index("area_group").loc["Midbrain"]
    wl = U.wh_vs_aud_active if "wh_vs_aud_active" in U else None
    lines = []
    A = lines.append
    # ------------------------------------------------------------------ front matter
    A(f"""---
title: "Whisker and auditory responses across the mouse brain and the convergence of whisker- and auditory-cortex projections"
subtitle: "ssl-sensory-spatial-maps -- SSL dataset (Neuropixels, Kilosort 4), report generated {pd.Timestamp.now():%Y-%m-%d %H:%M}"
author: "Axel Bisi (data); analysis with the IBL AI agent"
date: "{pd.Timestamp.now():%Y-%m-%d}"
abstract: |
  We mapped single-neuron responses to a whisker and an auditory stimulus across {len(U):,} neurons recorded with Neuropixels
  probes in {ses.session_id.nunique()} sessions ({U.mouse_id.nunique()} mice) of a whisker/auditory Go/NoGo task, registered to the Allen
  CCF. About {round(100 * (fr['whisker_active']['sig'] + fr['auditory_active']['sig']) / 2)} % of neurons responded to each
  stimulus within 5-35 ms, in partly distinct regions, and {pct(fr['wh_vs_aud_active']['sig'])} preferred one modality. Each
  stimulus reached its own sensory system first: whisker responses in whisker somatosensory cortex (median half-time to peak
  {sw.lat_w:.1f} ms), auditory responses in the midbrain ({mb.lat_a:.1f} ms). Using anterograde tracing from the Allen
  Connectivity Atlas (all projection-neuron lines, balanced per line), the projection zones of whisker and auditory cortex
  overlapped in {OV9['overlap_mm3']:.1f} mm³ (90 % zones). Inside this overlap, {pct(g9.P_in)} of sensory-responsive neurons
  were bimodal vs {pct(g9.P_ref)} overall ({100 * g9['diff']:+.1f} points, hierarchical bootstrap {P(g9.p_boot)}); with stricter
  70 % zones the difference was {100 * g7['diff']:+.1f} points ({P(g7.p_boot)}). Within shared targets, whisker- and
  auditory-preferring neurons were spatially offset in {len(sig_mc)} of {len(MC)} slabs.
---
""")
    A(f"""# Key results

1. **Sensory responses are widespread and spatially organised.** Whisker: {pct(fr['whisker_active']['sig'])} of tested neurons
   significant ({pct(fr['whisker_active']['pos'])} excited); auditory: {pct(fr['auditory_active']['sig'])}
   ({pct(fr['auditory_active']['pos'])} excited); whisker vs auditory preference: {pct(fr['wh_vs_aud_active']['sig'])}
   ({pct(fr['wh_vs_aud_active']['pos'])} auditory-preferring, {pct(fr['wh_vs_aud_active']['neg'])} whisker-preferring).
   Rate-based ROC, 5-35 ms after onset vs pre-trial baseline, 1000 permutations, p < 0.05; n = {fr['whisker_active']['n']:,} neurons.
2. **Each modality reaches its own sensory system first.** Median latency of whisker responses {sw.lat_w:.1f} ms in
   SS-whisker (auditory there {sw.lat_a:.1f} ms); auditory responses {mb.lat_a:.1f} ms in the midbrain (whisker there
   {mb.lat_w:.1f} ms); all responsive neurons: whisker {lat['whisker'].median():.1f} ms (n = {len(lat['whisker']):,}),
   auditory {lat['auditory'].median():.1f} ms (n = {len(lat['auditory']):,}).
3. **Whisker- and auditory-cortex projections overlap** in {OV9['overlap_mm3']:.1f} mm³ (90 % zones; 70 %: {OV7['overlap_mm3']:.1f} mm³),
   mainly in {', '.join(OV.head(6).structure)}.
4. **Bimodal neurons are enriched in the overlap** (90 % zones): {pct(g9.P_in)} vs {pct(g9.P_ref)}, {100 * g9['diff']:+.1f} points
   (95 % CI {100 * g9.diff_ci_lo:+.1f} to {100 * g9.diff_ci_hi:+.1f}), hierarchical bootstrap {P(g9.p_boot)}, Fisher {P(g9.p_fisher)};
   {int(g9.n_resp_in):,} responsive neurons inside, {int(g9.n_sessions_in)} sessions, {int(g9.n_mice_in)} mice. Stricter 70 % zones:
   {100 * g7['diff']:+.1f} points ({P(g7.p_boot)}).
5. **Within shared targets the two preferences are spatially offset** in {len(sig_mc)} of {len(MC)} target slabs (within-session
   label permutation, Holm-corrected p < 0.05).
""" + (f"""6. **Pre-lick converging neurons are not enriched in the overlap** (ssl-prelick-convergence, spontaneous-lick reference):
   {pct(PT.loc['All sessions', 'P_in'])} inside vs {pct(PT.loc['All sessions', 'P_ref'])} overall ({P(PT.loc['All sessions', 'p_boot'])}).
""" if PT is not None else ""))
    # ------------------------------------------------------------------ introduction
    A(f"""# Introduction

In the SSL task, head-fixed mice receive a brief whisker deflection or an auditory tone and learn to lick for reward;
the auditory stimulus is rewarded from the start, and the whisker stimulus is rewarded only in the R+ cohort (R- mice
learn not to respond to it). Whisker and auditory information therefore enter the brain through separate sensory
pathways but must reach the same motor output. This project describes, at single-neuron resolution and across the
whole brain, *where* and *when* each stimulus evokes responses, and asks whether neurons responding to both modalities
sit where the outputs of whisker and auditory cortex converge anatomically.

Three questions organise the report. (i) Where are whisker- and auditory-responsive neurons, and how is modality
preference distributed? (ii) In which order do areas respond, and does each modality reach its canonical sensory system
first? (iii) Where do the projections of whisker cortex (SSp-bfd, SSs) and auditory cortex (AUDp, AUDd/v) overlap, and
are bimodal neurons enriched there? The population counterpart -- when stimulus identity becomes decodable from
pseudo-populations of each area -- is reported in the companion project *ssl-stimulus-arrival-decoding*.

# Methods

## Data

- **Recordings.** Neuropixels probes, spike sorting with Kilosort 4 (`NWB_ks4`); {ses.session_id.nunique()} sessions
  ({(ses.stage == 'learning').sum()} learning-day, {(ses.stage == 'expert').sum()} expert-day; {(ses.cohort == 'R+').sum()} R+ and
  {(ses.cohort == 'R-').sum()} R- sessions) from {U.mouse_id.nunique()} mice. Cohort is a property of the mouse, taken from the mouse
  reference sheet; inclusion follows the SSL dataset record (`skills/ssl-valid-data`).
- **Neurons.** Good and multi-unit clusters of the v2 unit table (quality label with drift test): {len(U):,} neurons. CCF
  positions from probe-track reconstruction, folded onto one hemisphere (ML mirrored about the midline); {nrec} Allen
  structures recorded with >= 10 neurons. Area groups and areas follow the custom grouping of `allen_utils`.
- **Trials.** Active trials: context active (per-trial rule: a trial inside a fixed ~3 s ITI sequence is passive, otherwise
  the context column; unlabelled trials are active), `perf` != 6, the auditory warm-up block removed (the trial before the
  first whisker trial kept), the disengaged end of the session trimmed (rule A1: tail after the last lick with >= 5 whisker
  and >= 1 auditory trials). Passive trials: the pre- and post-task passive blocks.
- **Whisker stimulus artefact.** Spikes within -10 to +5 ms of each whisker onset are replaced by a Poisson train at the
  neuron's pre-onset rate (per-session seed); whisker responses before +5 ms cannot be detected.
- **Pooling.** Cohorts and stages are pooled throughout (scope of this project).

## Single-neuron responsiveness and modality preference

For a neuron and two trial sets $A$, $B$, spike counts in the response window (5-35 ms after stimulus onset) and the
baseline window (-1 s to -15 ms) give the area under the ROC curve and the selectivity
""")
    A(r"""$$\mathrm{sel} = 2\,\mathrm{AUC}(A, B) - 1 \in [-1, 1],$$

positive when $B$ has higher counts. Significance: 1000 permutations of the trial labels, one-sided, $p < 0.05$.
Responsiveness compares baseline ($A$) with stimulus trials ($B$) of the active task (whisker or auditory; positive =
excited); modality preference compares whisker ($A$) with auditory trials ($B$) in the response window (positive =
auditory-preferring).

## Bimodal neurons

A neuron is responsive to modality $m$ if any of its stimulus-vs-baseline tests in the $k_m$ epochs recorded in its session
(active, passive pre, passive post; $k_m \le 3$) is significant after Bonferroni correction,
$$R_m = \exists\, e:\; p_{m,e} < 0.05 / k_m,$$
and bimodal if $R_\text{whisker} \wedge R_\text{auditory}$. Among responsive neurons ($R_\text{whisker} \vee R_\text{auditory}$),
the bimodal fraction is $B = n_\text{both} / n_\text{responsive}$.

## Response latency

For a responsive neuron, the peri-stimulus time histogram (1-ms bins, Gaussian smoothing $\sigma$ = 2 ms) is baseline-corrected
and signed by the selectivity $s$: $r(t) = s\,[\mathrm{PSTH}(t) - \overline{\mathrm{PSTH}}_{[-100,-10]\,\mathrm{ms}}]$. With
$t_\mathrm{peak} = \arg\max_{5 \le t \le 100\,\mathrm{ms}} r(t)$, the latency is the last upward crossing of $r(t_\mathrm{peak})/2$
before $t_\mathrm{peak}$ (half-time to peak); whisker latencies are searched after +5 ms (artefact window).

## Spatial maps

For a quantity $q_i$ of neuron $i$ at CCF position $x_i$ (50-µm grid), the map is the ratio of Gaussian-smoothed sums,
$$\rho_q(x) = \frac{(G_\sigma * \sum_i q_i\,\delta_{x_i})(x)}{(G_\sigma * \sum_i \delta_{x_i})(x)},\qquad \sigma = 150\ \mu\mathrm{m},$$
i.e. the local mean of $q$ normalised by the density of recorded neurons with a value; it is shown where at least 3 neurons
fall within the kernel ($\sum_i G_\sigma \ge 3$), averaged over 500-µm slabs (coronal slabs tiling the recorded AP range).
**Isocortex flatmap.** Neurons in the isocortex are placed on the Allen CCFv3 butterfly flatmap (cortical streamlines from
the Laplace solution between pia and white matter; 2-D positions from geodesic distances to two pairs of anchor points;
`ccf_streamlines`, Wang et al. 2020, Harris et al. 2019) at the position of their closest streamline, left hemisphere,
anterior up; the same ratio is computed on the 10-µm flat grid with a 2-D Gaussian ($\sigma$ = 150 µm). The flatmap does
not preserve area; all statistics are computed in 3-D.

## Projection zones

Anterograde tracing experiments of the Allen Mouse Brain Connectivity Atlas with the primary injection in the source area,
from wild-type mice and every Cre line labelling projection neurons (interneuron lines excluded: Sst, Pvalb, Vip, Gad2,
Calb2, Nos1, Htr3a, Crh, Cort, ...). Per experiment $e$, the projection density $d_e(v)$ (fraction of voxel $v$ occupied by
labelled axons, 50-µm grid, injections mirrored into the right hemisphere) is normalised over the candidate voxels $C$
(right hemisphere, outside fibre tracts, ventricles and the source area), averaged within each Cre line $l$ ($E_l$
experiments) and then over the $K$ lines, so that lines with many injections do not dominate:
$$\bar D(v) = \frac{1}{K}\sum_{l=1}^{K}\frac{1}{E_l}\sum_{e \in l}\frac{d_e(v)}{\sum_{v' \in C} d_e(v')},\qquad \tilde D = G_{50\,\mu\mathrm{m}} * \bar D.$$
The zone at level $\alpha$ holds the highest-density voxels that together contain a fraction $\alpha$ of the projection:
$$Z_\alpha = \{v \in C : \tilde D(v) \ge \tau_\alpha\},\qquad \sum_{v \in Z_\alpha}\tilde D(v) = \alpha \sum_{v \in C}\tilde D(v),$$
with $\alpha$ = 0.9 (main) and 0.7 (stricter). Whisker zone: from the mean of the normalised SSp-bfd and SSs densities;
auditory zone: AUDp and AUDd/v; overlap $Z_w \cap Z_a$.
""")
    A(table(PZ.merge(PZ70[["source", "zone70_volume_mm3"]].rename(columns={"zone70_volume_mm3": "v70"}), on="source"),
            {"source": "Source", "n_experiments": "Experiments", "n_lines": "Lines", "zone70_volume_mm3": "90 % zone (mm³)", "v70": "70 % zone (mm³)"},
            {"zone70_volume_mm3": lambda v: f"{v:.1f}", "v70": lambda v: f"{v:.1f}"}))
    A("\n**Table 1.** Allen experiments per source (all projection-neuron lines), number of lines and zone volumes.\n")
    A(r"""
## Co-location test

For the bimodal fraction, $\Delta = P_\mathrm{in} - P_\mathrm{ref}$, where $P_\mathrm{in}$ is the pooled bimodal fraction of
responsive neurons inside the overlap and $P_\mathrm{ref}$ that of all responsive neurons. Main test: a hierarchical
bootstrap that keeps the clustering of neurons within sessions without pairing sessions (sampling differs strongly between
areas): $B$ = 2000 resamples of the sessions with replacement, then of the neurons within each session (binomial),
recomputing $P_\mathrm{in}$ and $P_\mathrm{ref}$; one-sided $p = (1 + \#\{\Delta^* \le 0\})/(1 + B)$ and the 95 % percentile
interval. Secondary: Fisher's exact test (neurons independent). Sub-regions of the overlap (connected 3-D pieces of
>= 0.04 mm³ per Allen structure, holding >= 10 recorded neurons) are described, with Holm-corrected bootstrap p-values and a
control against the rest of the same structure outside the overlap.

## Location of whisker- vs auditory-preferring neurons within targets

In 500-µm coronal slabs centred on target areas, the 80 % highest-density contours (Gaussian KDE) of whisker- and
auditory-preferring neurons are compared: centroid distance and shifts along the lateral and depth axes. Null: the
preference labels are permuted within sessions (5000 permutations), which removes offsets caused by which sessions
recorded which preference; Holm correction across slabs.
""")
    A("""
## Parameters

| Parameter | Value |
|---|---|
| Response / baseline window | 5-35 ms / -1 s to -15 ms |
| ROC permutations, alpha | 1000, 0.05 (one-sided) |
| Bimodal correction | Bonferroni over the k_m <= 3 epochs of the session |
| Latency PSTH | 1-ms bins, Gaussian sigma 2 ms, baseline -100 to -10 ms, peak search 5-100 ms |
| Map kernel | 3-D Gaussian sigma 150 µm, >= 3 neurons in the kernel, 500-µm slabs |
| Projection smoothing | Gaussian 50 µm, 50-µm grid |
| Zone levels | 90 % (main), 70 % (stricter) |
| Co-location bootstrap | 2000 resamples (sessions, then neurons) |
| Contour test | 80 % KDE contours, 5000 within-session permutations, Holm |
| Whisker artefact window | -10 to +5 ms (Poisson replacement) |

**Table 2.** Analysis parameters.
""")
    # ------------------------------------------------------------------ results
    A(f"""
# Results

## Where neurons respond

Of {fr['whisker_active']['n']:,} tested neurons, {pct(fr['whisker_active']['sig'])} responded to the whisker stimulus
({pct(fr['whisker_active']['pos'])} excited, {pct(fr['whisker_active']['neg'])} inhibited) and {pct(fr['auditory_active']['sig'])}
to the auditory stimulus ({pct(fr['auditory_active']['pos'])} excited, {pct(fr['auditory_active']['neg'])} inhibited) in the active
task. {pct(fr['wh_vs_aud_active']['sig'])} preferred one modality, auditory-preferring neurons outnumbering whisker-preferring ones
({pct(fr['wh_vs_aud_active']['pos'])} vs {pct(fr['wh_vs_aud_active']['neg'])}). On the isocortex flatmap (Figure 1; {nflat:,}
isocortical neurons), whisker responses and whisker preference concentrate in somatosensory cortex, while auditory
responses are more widespread. Of {n_r:,} neurons responsive to at least one modality, {n_w:,} responded to whiskers only,
{n_a:,} to sound only and {n_b:,} to both (bimodal fraction {pct(n_b / n_r)}).

""")
    A(fig(FZ / "cortical_flatmaps_nozones.png", "fig1_flatmaps.png",
          "**Figure 1. Sensory responses across the isocortex.** " + caption_md(SM / "cortical_flatmaps_caption_zone90.md").replace(
              "Lines: whisker-cortex", "(Projection zones are added in Figure 3.) Lines in Figure 3: whisker-cortex")))
    A("\n" + table(G.assign(area=G.area_group.map(short)),
                   {"area": "Area group", "n": "Neurons", "sessions": "Sessions", "wh": "Whisker resp.", "au": "Auditory resp.",
                    "lat_w": "Whisker latency (ms)", "lat_a": "Auditory latency (ms)"},
                   {"wh": lambda v: pct(v), "au": lambda v: pct(v), "lat_w": lambda v: f"{v:.1f}", "lat_a": lambda v: f"{v:.1f}",
                    "n": lambda v: f"{v:,}"}))
    A("\n**Table 3.** Responsiveness (fraction of tested neurons, active task) and median latency of responsive neurons per area "
      "group (groups with >= 200 neurons), sorted by whisker latency.\n")
    A(f"""
## When neurons respond

Whisker responses were fastest in whisker somatosensory cortex (median {sw.lat_w:.1f} ms) and auditory responses in the
midbrain ({mb.lat_a:.1f} ms); each modality was slower in the other's sensory system (auditory {sw.lat_a:.1f} ms in SS-whisker,
whisker {mb.lat_w:.1f} ms in the midbrain; Table 3; latency columns of Figure 1). Motor and striatal areas responded later
to both modalities.

## Where whisker- and auditory-cortex projections overlap

The line-balanced projection zones of whisker and auditory cortex (Table 1) cover {OV9['whisker_union_mm3']:.1f} and
{OV9['auditory_union_mm3']:.1f} mm³ (90 % zones) and overlap in {OV9['overlap_mm3']:.1f} mm³ (70 % zones: {OV7['whisker_union_mm3']:.1f},
{OV7['auditory_union_mm3']:.1f} and {OV7['overlap_mm3']:.1f} mm³; Figure 2, Figure S1). The largest recorded pieces of the
overlap are {', '.join(f"{r.structure} ({r.overlap_mm3:.2f} mm³)" for r in OV.head(8).itertuples())}.

""")
    A(fig(FZ / "projection_zones_coronal.png", "fig2_projection_zones.png",
          "**Figure 2. Projection zones of whisker and auditory cortex** (Allen anterograde tracing, all projection-neuron lines "
          "averaged per line). Coronal 500-µm slabs: line-balanced projection density of SSp-bfd, SSs, AUDp and AUDd/v (rows), 90 % "
          "contours, and the overlap of the merged whisker and auditory zones (bottom row); right: largest recorded structures as "
          "a fraction of each zone."))
    A(fig(FZ / "cortical_flatmaps.png", "fig3_flatmaps_zones.png",
          "**Figure 3. Cortical responses against the projection zones.** Same flatmaps as Figure 1 with the 90 % projection zones of "
          "whisker cortex (teal) and auditory cortex (brown, dashed); first column, bottom: whisker zone (yellow), auditory zone (blue) "
          "and overlap (purple)."))
    A(f"""
## Bimodal neurons are enriched where the projections converge

Inside the 90 % overlap, {pct(g9.P_in)} of {int(g9.n_resp_in):,} responsive neurons were bimodal, against {pct(g9.P_ref)} of
{int(g9.n_resp_ref):,} responsive neurons overall: {100 * g9['diff']:+.1f} points (95 % CI {100 * g9.diff_ci_lo:+.1f} to
{100 * g9.diff_ci_hi:+.1f}; hierarchical bootstrap {P(g9.p_boot)}; Fisher {P(g9.p_fisher)}; {int(g9.n_sessions_in)} sessions,
{int(g9.n_mice_in)} mice inside; Figure 4). With the stricter 70 % zones the difference was smaller and not significant:
{pct(g7.P_in)} of {int(g7.n_resp_in):,} vs {pct(g7.P_ref)}, {100 * g7['diff']:+.1f} points ({100 * g7.diff_ci_lo:+.1f} to
{100 * g7.diff_ci_hi:+.1f}), {P(g7.p_boot)} (Figure S2). The enrichment is therefore carried by the broader convergence
territory rather than by the core of the overlap. Sub-regions are listed in Table 4 (descriptive).

""")
    A(fig(FZ / "colocation_figure.png", "fig4_colocation.png",
          "**Figure 4. Bimodal neurons and the convergence of whisker- and auditory-cortex projections (90 % zones).** " +
          caption_md(SM / "colocation_figure_caption_zone90.md")))
    if len(SR9):
        A("\n" + table(SR9.assign(name=SR9.region), {"name": "Sub-region", "n_resp_in": "Responsive", "P_in": "Bimodal",
                                                      "P_rest_structure": "Rest of structure", "p_boot_holm": "p (Holm)"},
                       {"P_in": lambda v: pct(v), "P_rest_structure": lambda v: pct(v) if np.isfinite(v) else "n/a",
                        "p_boot_holm": lambda v: P(v)}))
        A("\n**Table 4.** Sub-regions of the 90 % overlap: responsive neurons inside, bimodal fraction, bimodal fraction in the rest "
          "of the same structure outside the overlap, and the Holm-corrected bootstrap p of inside vs all responsive neurons "
          "(descriptive).\n")
    A(f"""
## Whisker- and auditory-preferring neurons are spatially offset within shared targets

In {len(sig_mc)} of {len(MC)} target slabs, the 80 % contours of whisker- and auditory-preferring neurons had significantly
different centroids (within-session permutation, Holm p < 0.05; Figure 5, Table 5).

""")
    A(fig(FZ / "modality_contours.png", "fig5_modality_contours.png",
          "**Figure 5. Location of whisker- vs auditory-preferring neurons within target areas.** 80 % highest-density contours "
          "(Gaussian KDE) of whisker-preferring (yellow) and auditory-preferring (blue) neurons in 500-µm coronal slabs centred on "
          "target areas; crosses: centroids; text: neurons per class, centroid distance and Holm-corrected permutation p."))
    A("\n" + table(MC.assign(slab_=MC.slab.str.split(",").str[0]),
                   {"slab_": "Slab", "area": "Area", "n_whisker_pref": "Whisker-pref.", "n_auditory_pref": "Auditory-pref.",
                    "n_sessions": "Sessions", "centroid_distance_um": "Distance (µm)", "shift_depth_um": "Depth shift A-W (µm)",
                    "p_distance_holm": "p (Holm)"},
                   {"centroid_distance_um": lambda v: f"{v:.0f}", "shift_depth_um": lambda v: f"{v:+.0f}", "p_distance_holm": lambda v: P(v)}))
    A("\n**Table 5.** Offset of whisker- vs auditory-preferring neurons per target slab.\n")
    if PT is not None:
        A(f"""
## Pre-lick converging neurons are not enriched in the overlap

Neurons whose activity in the 100 ms before the first lick converges across modalities (ssl-prelick-convergence:
whisker hit vs spontaneous lick and auditory hit vs spontaneous lick both significant, same sign) were not enriched in the
overlap: {pct(PT.loc['All sessions', 'P_in'])} inside vs {pct(PT.loc['All sessions', 'P_ref'])} of all tested neurons
({P(PT.loc['All sessions', 'p_boot'])}), and no cohort x stage group showed an enrichment (Holm p >= {PT.loc[PT.index != 'All sessions', 'p_boot_holm'].min():.2f}).
The full analysis is reported in ssl-prelick-convergence (supplementary figure).
""")
    # ------------------------------------------------------------------ discussion
    A(f"""
# Discussion

Single-neuron responses to the whisker and the auditory stimulus are found throughout the recorded brain, but their
spatial distributions differ: whisker responses and whisker preference are concentrated in somatosensory cortex,
auditory responses extend further. Latencies follow the canonical pathways -- whisker cortex first for the whisker
stimulus, midbrain first for the auditory stimulus -- and downstream motor and striatal areas respond later to both.

Anatomically, the outputs of whisker and auditory cortex converge in a restricted set of targets (caudal striatum,
posterior parietal and associated visual areas, superior colliculus and midbrain reticular nucleus, temporal association
cortex). Bimodal neurons are more frequent inside this convergence territory when it is defined broadly (90 % zones), but
not significantly so in its core (70 % zones), so the association is graded rather than confined to the densest overlap.
Within shared targets, whisker- and auditory-preferring neurons occupy partly separate sub-territories, consistent with
topographically organised inputs. Neurons with modality-convergent pre-lick activity, in contrast, show no relation to the
projection overlap, suggesting that this task-related convergence is not inherited from converging cortical inputs.

# Caveats and limitations

- **Co-location, not causation.** Projection zones come from other mice (population-averaged tracing of excitatory cortical
  axons, including axons of passage); overlap marks where cortical inputs can converge, not that they drive the recorded
  responses. CCF positions of recorded neurons carry an uncertainty of roughly 100-200 µm.
- **Dependence on the zone level.** The bimodal enrichment is significant with 90 % zones and not with 70 % zones.
- **Line balance.** Averaging per Cre line gives each projection class equal weight; lines with a single, possibly atypical
  injection get the same weight as well-sampled lines.
- **Pooling.** Cohorts and stages are pooled; differences in sampled areas between cohorts and stages are not modelled.
- **Artefact window.** Whisker responses cannot be detected before +5 ms.
- **Sampling.** Responsiveness and latency per area depend on how many neurons were recorded there (Table 3 lists n).

# Supplementary figures

""")
    A(fig(FZ / "projection_zones_sagittal.png", "figS1_projection_zones_sagittal.png",
          "**Figure S1. Projection zones, sagittal slabs** (as Figure 2, 90 % zones)."))
    A(fig(F70 / "colocation_figure.png", "figS2_colocation_70.png",
          f"**Figure S2. Co-location with the stricter 70 % zones** (as Figure 4, 70 % contours): bimodal {pct(g7.P_in)} of "
          f"{int(g7.n_resp_in):,} responsive neurons inside the overlap vs {pct(g7.P_ref)} overall, {100 * g7['diff']:+.1f} points, "
          f"hierarchical bootstrap {P(g7.p_boot)}."))
    A(fig(F70 / "projection_zones_coronal.png", "figS3_projection_zones_70.png",
          "**Figure S3. Projection zones with 70 % contours** (as Figure 2)."))
    k = 4
    for q, ttl in (("whisker", "Whisker responsiveness"), ("auditory", "Auditory responsiveness"), ("modality", "Modality preference"),
                   ("bimodal", "Bimodal neurons"), ("latency_whisker", "Whisker response latency"),
                   ("latency_auditory", "Auditory response latency")):
        for p in sorted((FZ / q).glob("coronal_p*.png")):
            A(fig(p, f"figS{k}_{q}_{p.stem}.png",
                  f"**Figure S{k}. {ttl}, coronal 500-µm slabs ({p.stem.split('_')[-1]}).** Neurons (significant ones coloured by "
                  "the quantity) and density maps (3-D Gaussian, sigma 150 µm, normalised by recorded-neuron density); lines: 90 % "
                  "projection zones of whisker and auditory cortex."))
            k += 1
    # ------------------------------------------------------------------ appendix
    old = snap / "colocation_tests.csv"
    A(f"""
# Appendix

## Files and code

- Code: `projects/ssl-sensory-spatial-maps/` (ibl-ai-agent fork): `exploratory-analyses/001_unit_latency.py`,
  `002_projection_zones.py`, `003_spatial_maps.py`, `004_modality_contours.py`, `006_colocation_figure.py`,
  `007_cortical_flatmaps.py`, `008_prelick_convergence_colocation.py`; report: `report/build_article.py`.
- Results: `combined_results_ks4/_sensory_spatial_maps/` (to be moved to `combined_results_ks4/ssl-sensory-spatial-maps/`):
  `figures_zone90/` (main), `figures/` (70 %), `colocation_tests*.csv`, `projection_*`, `modality_contours*.csv`,
  `unit_latency.parquet`.
- Frozen statistics before the 2026-10-04 overnight reruns: `combined_results_ks4/_snapshots/2026-10-04_2200/`.

## Version history

- 2026-10-04: projection zones rebuilt from all projection-neuron lines averaged per line (before: wild type and
  Emx1-IRES-Cre only, 20 experiments). With the first zones the 70 % co-location gave +5.7 points (p = 0.008); with the
  line-balanced zones it is {100 * g7['diff']:+.1f} points ({P(g7.p_boot)}) at 70 % and {100 * g9['diff']:+.1f} points ({P(g9.p_boot)}) at 90 %,
  which became the main level.
- 2026-10-04: ROC re-run for three sessions with corrected passive / active trial labels (MH062, MH064, AB128); no
  conclusion changed (bimodal co-location and modality offsets identical within 0.1 points).
- 2026-10-04: co-location statistics: spatial-shift null and session-based tests removed (user decision); target-slab
  figures dropped.
""")
    # ------------------------------------------------------------------ write
    REP.mkdir(parents=True, exist_ok=True)
    if (REP / "figures").exists():
        shutil.rmtree(REP / "figures")                     # figures/ holds only the copies of this build
    (REP / "figures").mkdir()
    for src, name in FIGS:
        if not src.exists():
            print("MISSING", src)
            continue
        im = Image.open(src)
        if im.width > 2000:
            im = im.resize((2000, round(im.height * 2000 / im.width)), Image.LANCZOS)
        im.convert("RGB").save(REP / "figures" / name, quality=90, optimize=True)
    (REP / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (REP / "numbers.json").write_text(json.dumps(N, indent=1, default=lambda o: o if not hasattr(o, "item") else o.item()))
    shutil.copyfile(REPO / "skills" / "project-report" / "build.sh", REP / "build.sh")
    print("wrote", REP, len(FIGS), "figures")


if __name__ == "__main__":
    main()
