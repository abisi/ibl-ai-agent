"""Report of ssl-barrel-localization (skills/project-report), on the final user-picked wS1 barrel registration only
(ws1-barrel-registration/D2_final_user_picked, 2026-11-22). Reads the result tables of 000-005 and writes
combined_results_ks4/ssl-barrel-localization/report/ (report.md, numbers.json, figures/, build.sh); build.sh makes the PDF
and HTML. Run on haas (unit_spikes_analysis venv, needs Lana's code_snapshot on the path for 003_maps):
  PYTHONPATH=/raid0/lmsmith/local_barrel_reg/D2_neuropixels/code_snapshot python build_report.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "skills" / "project-report"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE.parent / "exploratory-analyses"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402
from report_lib import Report  # noqa: E402

RES = axel_bisi_root() / "combined_results_ks4" / "ssl-barrel-localization"
T, F, OUT = RES / "tables", RES / "figures", RES / "report"
FINAL = axel_bisi_root() / "combined_results_ks4" / "ws1-barrel-registration" / "D2_final_user_picked"
_spec = importlib.util.spec_from_file_location("m3", HERE.parent / "exploratory-analyses" / "003_maps.py")
m3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m3)

R = Report(banned=("Lana's fit",))
R.register([("eq-auc", "Eq. (1)"), ("eq-lat", "Eq. (2)"), ("eq-dist", "Eq. (3)"),
            ("fig-probes", "Figure 1"), ("fig-tissue", "Figure 2"), ("fig-maps", "Figure 3"), ("fig-c2", "Figure 4"),
            ("tbl-sizes", "Table 1"), ("tbl-class", "Table 2"), ("tbl-c2", "Table 3"),
            ("fig-s-qc", "Figure S1"), ("fig-s-dii", "Figure S2"), ("fig-s-class", "Figure S3"),
            ("tbl-s-mice", "Table S1"), ("tbl-s-registry", "Table S2")])
num, pv, ref, eq = R.num, R.pv, R.ref, R.eq

# ---------------------------------------------------------------- data
final = pd.read_csv(FINAL / "final_registration_table.csv", keep_default_na=False)
inv = pd.read_csv(T / "000_inventory.csv")
mice = sorted(p.stem.split("_")[-1] for p in T.glob("002_responses_*.parquet"))
df = pd.concat([m3.load(m) for m in mice], ignore_index=True)
df = df.merge(inv.drop_duplicates("mouse_id")[["mouse_id", "barrels_visible"]], on="mouse_id", how="left")
ref_sheet = pd.read_excel("/mnt/share_internal/Axel_Bisi_Share/dataset_info/joint_mouse_reference_weight.xlsx")
df = df.merge(ref_sheet[["mouse_id", "reward_group"]], on="mouse_id", how="left")
onmap = df[np.isfinite(df.trk_flat_anterior_um)]
reg_of = df.groupby("mouse_id").registration.first()
n_d2, n_gl = int((reg_of == "D2").sum()), int((reg_of == "global").sum())
probes = pd.read_csv(T / "004_probe_positions.csv")
qc = pd.read_csv(T / "001_link_qc.csv")
qc = qc[qc.mouse_id.isin(mice)].drop_duplicates("mouse_id", keep="last")
qcd = qc[qc.mouse_id.map(reg_of) == "D2"]
cls = pd.read_csv(T / "003_summary_all_allbrains_pooled.csv")
c2t = pd.read_csv(T / "005_functional_c2.csv")
c2a, c2v = c2t[c2t.set == "all"].set_index("route"), c2t[c2t.set == "visible"].set_index("route")
dii = pd.read_csv(T / "005_dii_check.csv")
dv = dii[dii.dii_visible.astype(bool) & dii.dist_global_um.notna() & dii.dist_D2_um.notna()]
w_dii = stats.wilcoxon(dv.dist_global_um, dv.dist_D2_um)
fits = []
for b in inv[~inv.d2_variant].d2_folder.unique():
    f = FINAL / b / "results" / "h1" / "metrics.json"
    if f.exists() and b in final.brain.values:
        m = json.load(open(f))
        fits.append(dict(brain=b, **{k: m.get(k) for k in ("err_initial_um", "err_tps_loo_um", "dice_initial", "dice_after",
                                                           "n_matched", "contrast_after")}))
fits = pd.DataFrame(fits)
w_err = stats.wilcoxon(fits.err_initial_um, fits.err_tps_loo_um)

# barrel-class rows used in the text (all layers pooled from the class x layer table, unit-weighted)
def pooled(c, col):
    x = cls[cls.barrel_class == c]
    return float(np.average(x[col], weights=x.n_units))


def at(c, layer, col):
    return float(cls[(cls.barrel_class == c) & (cls.layer == layer)][col].iloc[0])


l4 = onmap[onmap.layer.astype(str) == "4"]
in_c2 = l4.barrel_class.eq("C2")
other = l4.barrel_class.isin(["other barrels"])
tab_l4 = [[int(l4[in_c2].whisker_responsive.fillna(False).astype(bool).sum()), int((~l4[in_c2].whisker_responsive.fillna(False).astype(bool)).sum())],
          [int(l4[other].whisker_responsive.fillna(False).astype(bool).sum()), int((~l4[other].whisker_responsive.fillna(False).astype(bool)).sum())]]
fisher = stats.fisher_exact(tab_l4)

sizes = (onmap.groupby(["registration", "reward_group"]).agg(mice=("mouse_id", "nunique"), probes=("track_file", lambda s: s.dropna().nunique()),
                                                         units=("cluster_id", "size")).reset_index())
sizes["probes"] = onmap.groupby(["registration", "reward_group"]).apply(
    lambda g: g[["session_id", "electrode_group"]].drop_duplicates().shape[0], include_groups=False).values
sizes = sizes.rename(columns={"registration": "registration", "reward_group": "cohort"})
sizes.loc[len(sizes)] = dict(registration="all", cohort="all", mice=onmap.mouse_id.nunique(),
                             probes=onmap[["session_id", "electrode_group"]].drop_duplicates().shape[0], units=len(onmap))

per_mouse = (onmap.groupby("mouse_id").agg(registration=("registration", "first"), cohort=("reward_group", "first"),
                                           units=("cluster_id", "size"), in_barrel=("barrel", lambda s: int((s != "").sum())))
             .reset_index().merge(final[["brain", "picked_variant"]].rename(columns={"brain": "mouse_id"}), on="mouse_id", how="left")
             .merge(qc[["mouse_id", "trk_barrel_same_glob", "trk_shift_median_um"]], on="mouse_id", how="left"))
per_mouse["trk_barrel_same_glob"] = per_mouse.trk_barrel_same_glob.round(2)
per_mouse["trk_shift_median_um"] = per_mouse.trk_shift_median_um.round(0)
per_mouse = per_mouse.rename(columns={"picked_variant": "fit (variant)", "trk_barrel_same_glob": "same barrel as global",
                                      "trk_shift_median_um": "shift (µm)", "in_barrel": "units in a barrel"})

class_tbl = cls[cls.layer.isin(["2/3", "4", "5", "6a"])].copy()
class_tbl = class_tbl.assign(whisker_responsive=class_tbl.whisker_responsive.round(2), whisker_auc=class_tbl.whisker_auc.round(3),
                             whisker_latency_ms=class_tbl.whisker_latency_ms.round(0), auditory_responsive=class_tbl.auditory_responsive.round(2))
class_tbl = class_tbl[["barrel_class", "layer", "n_units", "n_mice", "whisker_responsive", "whisker_auc", "whisker_latency_ms",
                       "auditory_responsive"]].rename(columns={"barrel_class": "class", "n_units": "units", "n_mice": "mice",
                                                               "whisker_responsive": "whisker resp.", "whisker_auc": "whisker AUC",
                                                               "whisker_latency_ms": "latency (ms)", "auditory_responsive": "auditory resp."})
c2_tbl = c2t.assign(spearman_auc_vs_dist=c2t.spearman_auc_vs_dist.round(3), p=c2t.p.map(lambda x: f"{x:.1e}"),
                    frac_resp_within150=c2t.frac_resp_within150.round(2), frac_resp_beyond500=c2t.frac_resp_beyond500.round(2))
c2_tbl = c2_tbl[["set", "route", "n_units", "n_mice", "spearman_auc_vs_dist", "p", "frac_resp_within150", "frac_resp_beyond500"]].rename(
    columns={"n_units": "units", "n_mice": "mice", "spearman_auc_vs_dist": "Spearman r", "frac_resp_within150": "resp. < 150 µm",
             "frac_resp_beyond500": "resp. > 500 µm"})

g_all, d_all = c2a.loc["global"], c2a.loc["D2"]
g_vis, d_vis = c2v.loc["global"], c2v.loc["D2"]
mix = c2a.loc["global barrels, DiI-track units"]
n_vis = int(c2v.loc["D2", "n_mice"])
probes_per_barrel = probes.barrel.value_counts()

# ---------------------------------------------------------------- text
md = f"""---
title: "Where the wS1 probes are: units on the final barrel registration"
subtitle: "SSL dataset, NWB_ks4 (Kilosort 4), good + multi-units on wS1 probes; final user-picked D2 barrel fits (2026-11-22); built {pd.Timestamp.now():%Y-%m-%d %H:%M}"
---

# Abstract

The whisker barrel field (wS1) is where the SSL task's whisker stimulus arrives, and the barrel a probe records from
decides which whisker drives its units. The whole-brain (global) registration places barrels with an error of about one
barrel, so each brain's barrel field was registered locally (method D2) and the best fit per brain was chosen by eye
({num("n_brains", len(final))} brains: {num("n_local", int((final.decision == "use_local").sum()))} with an accepted local
fit, {num("n_global", int((final.decision == "use_global").sum()))} where the global registration was kept). Units of
{num("n_mice_map", onmap.mouse_id.nunique())} mice ({num("n_units_map", len(onmap))} good and multi-units on wS1 probes) are placed
on the common barrel map: {num("n_mice_d2", n_d2)} mice through their local fit and {num("n_mice_gl", n_gl)} through the global
registration of the NWB files ({ref("tbl-sizes")}). Probes are concentrated around C2 (C3, the D row, C1), with few L4 sites in C2 itself ({ref("fig-probes")}).
Whisker responses are most frequent in and around C2, the stimulated whisker's barrel ({ref("fig-maps")}), and fall off
with distance from C2 under both registrations; the local fits do not make this tuning sharper than the global
registration ({ref("fig-c2")}, {ref("tbl-c2")}).

**Key results**

1. {R.claim("c1", f"The final local fits lower the barrel landmark error from {num('err0', fits.err_initial_um.median(), '{:.0f}')} µm (global placement) to {num('err1', fits.err_tps_loo_um.median(), '{:.0f}')} µm (leave-one-out), median over {num('n_fits', len(fits))} brains (Wilcoxon p {pv('p_err', w_err.pvalue)})", "Figure S1", "Wilcoxon signed-rank", f"{len(fits)} brains")}.
2. {R.claim("c2", f"On the probe's own DiI track, units change barrel relative to the global labels: the same barrel for a median {num('same_med', qcd.trk_barrel_same_glob.median() * 100, '{:.0f}')} % of units per mouse, with a median shift of {num('shift_med', qcd.trk_shift_median_um.median(), '{:.0f}')} µm", "Table S1", "per-mouse medians", f"{len(qcd)} mice")}.
3. {R.claim("c3", f"L4 units in C2 are more often whisker-responsive than L4 units in other barrels ({num('c2_l4', at('C2', '4', 'whisker_responsive') * 100, '{:.0f}')} % vs {num('oth_l4', at('other barrels', '4', 'whisker_responsive') * 100, '{:.0f}')} %; Fisher's exact p {pv('p_fisher', fisher.pvalue)}, units pooled) and respond with a median latency of {num('c2_l4_lat', at('C2', '4', 'whisker_latency_ms'), '{:.0f}')} ms", "Figure 3, Table 2", "Fisher exact (units pooled)", f"{int(sum(map(sum, tab_l4)))} L4 units", status="fragile", note="units pooled across mice")}.
4. {R.claim("c4", f"Whisker AUC falls with distance from the C2 centre under the local fits (Spearman r = {num('r_d2', d_all.spearman_auc_vs_dist, '{:.2f}')}) and equally under the global registration (r = {num('r_gl', g_all.spearman_auc_vs_dist, '{:.2f}')}); in the {num('n_vis', n_vis)} mice with clearly visible barrels the global map is steeper (r = {num('r_gl_v', g_vis.spearman_auc_vs_dist, '{:.2f}')} vs {num('r_d2_v', d_vis.spearman_auc_vs_dist, '{:.2f}')})", "Figure 4", "Spearman", f"{int(d_all.n_units)} units, {int(d_all.n_mice)} mice")}.
5. {R.claim("c5", f"Placing units on their DiI track sharpens the C2 tuning more than replacing the barrel map: global barrels with DiI-track units give r = {num('r_mix', mix.spearman_auc_vs_dist, '{:.2f}')} and {num('mix_150', mix.frac_resp_within150 * 100, '{:.0f}')} % responsive within 150 µm of C2 (local fits {num('d2_150', d_all.frac_resp_within150 * 100, '{:.0f}')} %)", "Figure 4", "Spearman", f"{int(mix.n_units)} units")}.

# Introduction

In the SSL task the C2 whisker is stimulated, so units in the C2 barrel column should respond first and most, and units
farther away less. Testing this, and interpreting any wS1 result by barrel, needs the barrel each unit sits in. The NWB
files carry each unit's position in the Allen CCF from the global brainreg registration and the IBL ephys alignment;
barrel by barrel this placement can be off by a barrel or more. The local registration (ws1-barrel-registration, method
D2) fits the atlas barrel map onto the barrels visible in each brain's own L4 image and was reviewed brain by brain; this
report uses only the final, user-picked fits.

Questions: where are the probes and units in the barrel field (Section 4.1)? How much does the local fit move units
(4.2)? How do whisker and auditory responses vary across barrels and layers (4.3)? Does the barrel map pass the C2 test,
and is the local fit better than the global registration on it (4.4)?

# Methods

**Data.** {R.methods("units").strip()} Mice: every SSL mouse with a wS1-targeted probe (NWB `electrode_group` location) and
a registered brain ({num("n_mice_inv", inv[inv.target_area == "wS1"].mouse_id.nunique())} mice in the inventory). Units of the
wS1 probes with an annotated DiI track, left hemisphere. Cohort (R+ / R-) from the mouse reference sheet.

**Barrel registration (final fits).** Per brain, the D2 fit chosen on the review page
(`ws1-barrel-registration/D2_final_user_picked/`): `use_local` = the picked fit's barrel per voxel
(`barrels_corrected_full.tif`), corrected CCF position (`ccf_field_*`) and flat coordinates (`flatcoords.npz`);
`use_global` = the local fit was rejected and the global registration is kept. For a mouse with several brains, the brain
with annotated probe tracks is used. Methods of the fit and the review: repository `abisi/wS1_barrel_registration` and its
wiki.

**Placing units.** `use_local` mice: each unit's global CCF (NWB `ccf_atlas_*`, peak channel of the IBL alignment) is
projected on the atlas-space DiI track, carried at the same arc fraction to the sample-space DiI track, and the final fit's
barrel, layer (Blue Brain annotation at the corrected CCF) and flat position are read there (track transfer; 001).
`use_global` mice: the barrel and layer of the annotation at the unit's NWB CCF, and its flat position from the Blue Brain
flatmap at that voxel through the template's flatmap-to-plane affine (the frame of all local fits). Barrel classes: C2;
C2 neighbours (B1-B3, C1, C3, D1-D3); other barrels; septa / between columns (in SSp-bfd but not in a barrel column);
outside SSp-bfd.

**Responses (002).** Active whisker and auditory trials (perf ≠ 6, no early lick), aligned at the stimulus; whisker window
5-50 ms, auditory 5-100 ms, baseline the 100 ms before. Responsiveness compares evoked and baseline spike counts trial by
trial with the area under the ROC curve,
{eq("eq-auc", r"\mathrm{AUC} = P\left(n_{\mathrm{evoked}} > n_{\mathrm{baseline}}\right) + \tfrac{1}{2}\,P\left(n_{\mathrm{evoked}} = n_{\mathrm{baseline}}\right),")}
where $n$ are spike counts of the same trial in the two windows (0.5 = no change); a unit is responsive when the Wilcoxon
signed-rank test gives p < 0.01 and the rate increases. The latency of a responsive unit is the first 1-ms bin $t$ of the
trial-averaged PSTH $r(t)$, without smoothing, with
{eq("eq-lat", r"r(t) > \mu_b + 3\,\sigma_b \quad\text{and}\quad r(t+1) > \mu_b + 3\,\sigma_b,")}
where $\mu_b$ and $\sigma_b$ are the mean and standard deviation of the baseline bins.

**C2 test (005).** For each unit, the distance to the C2 centre in the flat L4 plane,
{eq("eq-dist", r"d_{C2} = \left\lVert \mathbf{u} - \mathbf{c}_{C2} \right\rVert,")}
where $\mathbf{{u}}$ is the unit's flat position and $\mathbf{{c}}_{{C2}}$ the centroid of C2 in the same map. Four routes
combine a barrel map (global or local fit) with a unit placement (NWB CCF or DiI track), all in the brain's own flatmap
before the barrel correction, so the tissue image is identical. L2/3 and L4 units (that map's layers) of the
`use_local` mice; Spearman correlation of the whisker AUC ({ref("eq-auc")}) with $d_{{C2}}$ ({ref("eq-dist")}). The DiI
check compares, for probes with a visible DiI spot in the brain's own L4 image, the distance from the spot to the
probe's L4 units under each placement (Wilcoxon signed-rank).

**Statistics.** Exploratory; fractions and medians pooled over units, with the number of mice given; tests as stated,
uncorrected. Sample sizes in {ref("tbl-sizes")}, per mouse in {ref("tbl-s-mice")}.

**Code.** `projects/ssl-barrel-localization/exploratory-analyses/000_inventory.py` … `005_plot.py` (haas); this report:
`report/build_report.py`.

# Results

## Where are the probes?

{num("n_probes_map", len(probes))} probe-sessions ({num("n_mice_probes", probes.mouse_id.nunique())} mice) cross L4 of the
barrel field ({ref("fig-probes")}). By the barrel of its L4 units, the most frequent is {probes_per_barrel.index[0]}
({num("top_barrel_n", int(probes_per_barrel.iloc[0]))} probe-sessions), then {probes_per_barrel.index[1]}
({num("second_barrel_n", int(probes_per_barrel.iloc[1]))}); C2 itself holds {num("c2_probes", int(probes_per_barrel.get("C2", 0)))},
but counting every barrel a probe has units in, C3 and C2 are crossed most often (right panel).
The probes enter at an angle, so one probe usually crosses two or three barrels between L2/3 and L6. The two cohorts are
spread over the same barrels ({ref("tbl-sizes")}).

## How much does the local fit move units?

On the brain's own tissue the local fit sits on the visible barrels where the global registration is offset
({ref("fig-tissue")}). Its landmark error is lower ({ref("fig-s-qc")}), and on the probe's own DiI track a median
{num("same_med2", qcd.trk_barrel_same_glob.median() * 100, '{:.0f}')} % of a mouse's units keep their global barrel
({ref("tbl-s-mice")}). For probes whose DiI spot is visible in L4 ({num("n_dii", len(dv))} probes), the L4 units lie
{num("dii_gl", dv.dist_global_um.median(), '{:.0f}')} µm from the spot with the NWB placement and
{num("dii_d2", dv.dist_D2_um.median(), '{:.0f}')} µm with the DiI-track placement (Wilcoxon p {pv("p_dii", w_dii.pvalue)};
{ref("fig-s-dii")}).

## How do responses vary across barrels and layers?

Whisker-responsive units are most frequent in C2 and its neighbours ({ref("fig-maps")}, {ref("tbl-class")}):
{num("c2_all", pooled("C2", "whisker_responsive") * 100, '{:.0f}')} % in C2 over all layers,
{num("nb_all", pooled("C2 neighbours", "whisker_responsive") * 100, '{:.0f}')} % in its neighbours,
{num("ot_all", pooled("other barrels", "whisker_responsive") * 100, '{:.0f}')} % in other barrels and
{num("sep_all", pooled("septa / between columns", "whisker_responsive") * 100, '{:.0f}')} % between columns. Latencies are
shortest in C2 (median {num("c2_23_lat", at("C2", "2/3", "whisker_latency_ms"), '{:.0f}')} ms in L2/3 and
{num("c2_l4_lat2", at("C2", "4", "whisker_latency_ms"), '{:.0f}')} ms in L4, against
{num("nb_l4_lat", at("C2 neighbours", "4", "whisker_latency_ms"), '{:.0f}')} ms in L4 of its neighbours). Auditory responses are
rarer ({num("aud_min", cls[cls.layer.isin(["2/3", "4", "5"])].auditory_responsive.min() * 100, '{:.0f}')}-{num("aud_max", cls[cls.layer.isin(["2/3", "4", "5"])].auditory_responsive.max() * 100, '{:.0f}')} % of units in L2/3-L5) and do not peak at C2 ({ref("fig-s-class")}).

## Does the barrel map pass the C2 test?

Under every route the whisker AUC ({ref("eq-auc")}) and the fraction of responsive units fall with the distance from C2
({ref("eq-dist")}; {ref("fig-c2")}, {ref("tbl-c2")}), so all maps place C2 roughly right. The local fits are not sharper than
the global registration (r = {num("r_d2b", d_all.spearman_auc_vs_dist, '{:.2f}')} vs {num("r_glb", g_all.spearman_auc_vs_dist, '{:.2f}')});
the steepest gradient and the highest responsive fraction near C2 come from the global barrels with units on their DiI
track (r = {num("r_mixb", mix.spearman_auc_vs_dist, '{:.2f}')}). Latency ({ref("eq-lat")}) is shortest within 300 µm of C2
under the local fits but too sparse beyond 800 µm to compare.

# Discussion

The final registration changes the barrel of most units: the local fits sit on the tissue barrels, their landmark error is
lower, and units placed on their DiI track lie closer to the visible DiI spot. The functional C2 test does not separate the
two maps: responses fall off with distance from C2 under both, and the gain comes from placing units on the physical
track rather than from the barrel map. Two readings are possible: either the global barrel map is already close in the
barrels the probes target (C2, C3, D row, where the field is centred), or the response gradient is broad (several hundred
µm, the whisker drives surround barrels at these latencies) and does not resolve a one-barrel difference.

# Caveats

- Units are pooled across mice for fractions and tests; a few mice contribute many units ({ref("tbl-s-mice")}).
- The C2 test uses L2/3 and L4 units of `use_local` mice only; `use_global` mice have no local map to compare.
- Probe tracks were annotated by hand on DiI; where the DiI is weak, the track transfer inherits that uncertainty.
- Responses are from active trials of all recording days; learning-stage differences are not separated here.
- Barrel classes depend on the map; septa are thin, so units near a border change class with small shifts.

# Figures

{R.figure("fig-probes", F / "topdown" / "004_probe_locations.png", f"Probe locations on the barrel map (final registration). Left: each wS1 probe's path through the barrel field from its superficial to its deep units (line) and its L4 units (dot), green R+ and magenta R-; the common flat frame of the D2 template, anterior to the right, row A at the top. Right: probes with units in each barrel | mice. {len(probes)} probe-sessions, {probes.mouse_id.nunique()} mice; `use_global` mice placed with the NWB global registration.", "004_topdown_maps.py")}

{R.figure("fig-tissue", F / "validation" / "005_tissue_AB140.png", "Global and local registration on one brain's own tissue (AB140). The brain's L4 autofluorescence (grey) in its own flatmap, the DiI column maximum (red) and the clicked DiI track (dotted). Middle: global barrel outlines and units at their NWB CCF; right: the final local fit's barrels and the units on the DiI track, coloured by layer; thick outline = C2; text: barrels crossed by each probe.", "005_plot.py")}

{R.figure("fig-maps", F / "topdown" / "004_response_maps_all.png", f"Responses by barrel and layer, all mice ({onmap.mouse_id.nunique()} mice, {len(onmap)} units on the map). Columns: fraction of whisker-responsive units, median whisker AUC (Eq. (1)), median whisker latency of responsive units (Eq. (2)), fraction of auditory-responsive units. Rows: all layers, L2/3, L4, L5/6. Barrels with at least 10 units are filled (latency: at least 3 responsive units); light grey = fewer.", "004_topdown_maps.py")}

{R.figure("fig-c2", F / "validation" / "005_functional_c2_all.png", f"C2 test: whisker response against the distance from the C2 centre (Eq. (3)), L2/3 and L4 units of {int(d_all.n_mice)} `use_local` mice. Solid: global (barrels and NWB-CCF units) and D2 (final fit, DiI-track units); dashed: the two mixed routes. Panels: fraction responsive, median AUC, median latency of responsive units per distance bin; right: Spearman r of AUC vs distance (more negative = tuned to C2). Numbers in Table 3.", "005_plot.py")}

{R.table("tbl-sizes", sizes, "Mice, probe-sessions and units (good + multi-units on wS1 probes, placed on the barrel map) by registration and cohort.")}

{R.table("tbl-class", class_tbl, "Responses by barrel class and layer (all mice, units pooled): fraction whisker-responsive, median whisker AUC (Eq. (1)), median latency of responsive units (Eq. (2)), fraction auditory-responsive.")}

{R.table("tbl-c2", c2_tbl, "C2 test per route (Figure 4): Spearman correlation of whisker AUC with distance from C2 (Eq. (3)), fraction responsive within 150 µm and beyond 500 µm; set all = all `use_local` mice, visible = mice with clearly visible barrels.")}

# Supplementary material

{R.figure("fig-s-qc", F / "validation" / "005_fit_qc.png", f"Final fit quality, {len(fits)} brains with a local fit: barrel landmark error against the tissue barrels (global placement vs local fit, leave-one-out) and Dice overlap with the tissue barrels; lines join the same brain.", "005_plot.py", kind="supplementary")}

{R.figure("fig-s-dii", F / "validation" / "005_dii_distance.png", f"Distance from the visible DiI spot in L4 to the probe's L4 units, NWB placement (x) vs DiI-track placement (y), {len(dv)} probes with a visible DiI spot; Wilcoxon signed-rank.", "005_plot.py", kind="supplementary")}

{R.figure("fig-s-class", F / "003_summary_all_allbrains_pooled.png", "Whisker and auditory responses by barrel class and layer, all mice (units pooled): fraction responsive, AUC and latency (Table 2).", "003_maps.py", kind="supplementary")}

{R.table("tbl-s-mice", per_mouse, "Per mouse: registration used (D2 = final local fit, global = NWB), cohort, units on the map, units in a barrel column, the picked fit variant, and for D2 mice the fraction of DiI-track units in the same barrel as their global label and the median shift between the two positions.")}

{R.registry_table("tbl-s-registry")}

# Appendix

Every claim of the report, with its figure, statistic and status, is listed in {ref("tbl-s-registry")}. Result tables: `combined_results_ks4/ssl-barrel-localization/tables/` (000_inventory.csv, 001_units_<mouse>.parquet with
the `registration` column, 002_responses_<mouse>.parquet, 003_summary_*.csv, 004_barrel_maps_*.csv,
004_probe_positions.csv, 005_functional_c2.csv, 005_dii_check.csv). Figures: `figures/topdown/` (maps for all, R+, R-
and visible-barrel mice), `figures/per_mouse/` (flat maps per mouse), `figures/validation/` (tissue figure per brain,
C2 test, DiI and fit QC). The analysis before the final registration (Lana Smith's earlier D2 run, 2026-11-11) is in
`_archive_lana_d2_neuropixels_20261111/`; it is superseded and not used here.
"""
R.write(OUT, md)
