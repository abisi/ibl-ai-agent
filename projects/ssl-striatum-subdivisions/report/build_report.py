"""Report of ssl-striatum-subdivisions (skills/project-report): reads the result tables of 001_striatum_split.py and writes
combined_results_ks4/ssl-striatum-subdivisions/report/ (report.md, numbers.json, figures/, build.sh); build.sh makes the
PDF and HTML. Run locally (NAS mounted) or on haas:  python build_report.py"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "skills" / "project-report"))
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402
from report_lib import Report  # noqa: E402

RES = axel_bisi_root() / "combined_results_ks4" / "ssl-striatum-subdivisions"
T, F, OUT = RES / "tables", RES / "figures", RES / "report"
SUBS = ["DMS", "DLS", "VS", "TS"]
TYPES = {"whisker_passive_pre": "passive whisker", "auditory_passive_pre": "passive auditory",
         "whisker_active": "whisker (active)", "auditory_active": "auditory (active)",
         "whisker_choice": "whisker choice", "spontaneous_licks": "spontaneous licks"}

R = Report()
R.register([("fig-split", "Figure 1"), ("fig-s-maps", "Figure S1"), ("tbl-counts", "Table 1"), ("tbl-tests", "Table 2"),
            ("tbl-s-boxes", "Table S1")])
num, pv, ref = R.num, R.pv, R.ref

prov = json.load(open(RES / "provenance.json"))
counts = pd.read_csv(T / "counts_old_vs_new.csv")
trans = pd.read_csv(T / "transitions_old_to_new.csv", index_col=0)
tests = pd.read_csv(T / "roc_subdivision_tests.csv")
new = tests[tests.labels == "new"].set_index("analysis_type")
old = tests[tests.labels == "old"].set_index("analysis_type")
c = counts.pivot(index="subdivision", columns="labels", values="units").reindex(SUBS)
n_units = int(trans.to_numpy().sum())
n_changed = n_units - int(sum(trans.loc[s, s] for s in SUBS))
dms_vs = int(trans.loc["DMS", "VS"])
vs_dms = int(trans.loc["VS", "DMS"])
m = lambda ty, s: new.loc[ty, f"mean_{s}"]
BOXES = pd.DataFrame([dict(order=1, subdivision="VS", AP="0 to 6", ML="0 to 6", DV="4 to 7"),
                      dict(order=2, subdivision="DLS", AP="-1.5 to 0.5", ML="2.4 to 6", DV="0 to 7"),
                      dict(order=3, subdivision="DMS", AP="-1.5 to 6", ML="0 to 3", DV="0 to 7"),
                      dict(order=4, subdivision="TS", AP="-6 to -1.5", ML="0 to 6", DV="0 to 7")])
bonf_kw = [TYPES[t] for t in TYPES if new.loc[t, "kw_p"] * len(TYPES) < 0.05]
bonf_an = [TYPES[t] for t in TYPES if new.loc[t, "anova_p"] * len(TYPES) < 0.05]

md = f"""---
title: "Striatal subdivisions after the bregma-coordinate fix"
subtitle: "SSL dataset, NWB_ks4 (Kilosort 4), good + multi-units; built {pd.Timestamp.now():%Y-%m-%d %H:%M}"
---

# Abstract

The striatum of the SSL recordings is split into dorsomedial (DMS), dorsolateral (DLS), ventral (VS) and tail (TS)
striatum by coordinate boxes in `allen_utils.create_areas_subdivisions`. Until 2026-11-17, the bregma-centred coordinates
used for the boxes were computed from the `ccf_ap/ml/dv` columns of the NWB files, which are not in the space of the
atlas labels (`ccf_atlas_acronym`). They are now computed from `ccf_atlas_ap/ml/dv`. Of {num("n_units", n_units)} striatal
units, {num("n_changed", n_changed)} ({num("pct_changed", 100 * n_changed / n_units, "{:.1f}")} %) change subdivision,
mostly between DMS and VS ({ref("tbl-counts")}). At the session level, the four subdivisions differ in sensory response
prevalence on day 0 (whisker and auditory, passive and active; {ref("tbl-tests")}), as before the fix, while choice and
spontaneous-lick responses do not differ significantly.

**Key results**

1. {R.claim("c1", f"{num('n_changed_k', n_changed)} of {num('n_units_k', n_units)} striatal units change subdivision; {num('dms_vs', dms_vs)} move DMS -> VS and {num('vs_dms', vs_dms)} VS -> DMS", "Figure 1b", "counts", f"{n_units} units")}.
2. {R.claim("c2", f"Whisker (active) responses are most frequent in DLS ({num('wa_dls', m('whisker_active', 'DLS'), '{:.1f}')} % of units per session) and TS ({num('wa_ts', m('whisker_active', 'TS'), '{:.1f}')} %), rare in DMS ({num('wa_dms', m('whisker_active', 'DMS'), '{:.1f}')} %) and VS ({num('wa_vs', m('whisker_active', 'VS'), '{:.1f}')} %); Kruskal-Wallis p {pv('kw_wa', new.loc['whisker_active', 'kw_p'])}, ANOVA p {pv('an_wa', new.loc['whisker_active', 'anova_p'])}", "Figure 1e", "KW + one-way ANOVA", "sessions")}.
3. {R.claim("c3", f"Auditory (active) responses are most frequent in DMS ({num('aa_dms', m('auditory_active', 'DMS'), '{:.1f}')} %); Kruskal-Wallis p {pv('kw_aa', new.loc['auditory_active', 'kw_p'])}, ANOVA p {pv('an_aa', new.loc['auditory_active', 'anova_p'])}", "Figure 1e", "KW + one-way ANOVA", "sessions")}.
4. {R.claim("c4", f"Whisker-choice and spontaneous-lick responses do not differ between subdivisions (Kruskal-Wallis p {pv('kw_wc', new.loc['whisker_choice', 'kw_p'])} and p {pv('kw_sl', new.loc['spontaneous_licks', 'kw_p'])})", "Figure 1e", "KW + one-way ANOVA", "sessions", status="n.s.")}.

# Introduction

Striatal subdivisions receive different cortical inputs: DLS from sensorimotor cortex (including whisker cortex), DMS
from frontal and associative cortex, TS from sensory (auditory, visual) cortex, VS from limbic areas. The subdivision
labels are used in the area-level analyses of the project (response prevalence, decoding, convergence). This report
checks how the coordinate fix changes the labels and whether the subdivisions still differ in function.

# Methods

**Data.** {R.methods("task", "units").strip()} Here, all good and multi-units (Bombcell labels) of every NWB_ks4 session
with units in the striatum are used ({num("n_sessions", prov["n_sessions"])} sessions, {num("n_mice", prov["n_mice"])}
mice), without the stability filter, as in the area labels of the shared tables.

**Subdivision labels.** `process_allen_labels(split_merge_areas=True)` maps striatal units (Allen CP, ACB, FS, OT, ...)
to DMS / DLS / VS / TS with coordinate boxes in bregma-centred coordinates ({ref("tbl-s-boxes")}). The bregma-centred
coordinates are AP = 5400 − ccf_ap, ML = ccf_ml − 5739, DV = ccf_dv − 332 µm. "Old" labels use the NWB `ccf_ap/ml/dv`
columns; "new" labels use `ccf_atlas_ap/ml/dv`, the coordinates that also give the atlas acronym of each unit
(`ephys_utilities` {prov["ephys_utilities"].split()[0]}). Both are recomputed here from the same NWB files.

**Response prevalence.** Single-unit ROC selectivity (`_roc_stage_analysis/roc_long.parquet`; significance from the
permutation test of the ROC pipeline) on day 0 for six comparisons: passive whisker and auditory (pre-training passive
block), whisker and auditory in the active task, whisker choice (hit vs miss) and spontaneous licks. For each session
and subdivision with at least {num("min_units", prov["min_units"])} units, the percentage of significant units is computed;
the session is the unit of analysis. Subdivisions are compared with the Kruskal-Wallis test and a one-way ANOVA, on the
new and on the old labels.

**Code.** `projects/ssl-striatum-subdivisions/001_striatum_split.py` (haas), the area-summary panels with
`brain_wide_analysis/plotting_functions.plot_area_proj_brain`; this report: `report/build_report.py`.

# Results

## How many units change subdivision?

The fix changes the label of {num("n_changed2", n_changed)} units. DMS loses units to VS and VS gains units overall
(old {num("vs_old", int(c.loc["VS", "old"]))}, new {num("vs_new", int(c.loc["VS", "new"]))}), while DLS and TS are almost
unchanged ({ref("fig-split")}a-b, {ref("tbl-counts")}). VS is the anterior striatum below DV 4 mm, assigned first, so units near
that depth switch between DMS and VS when the DV coordinate changes by a few hundred µm. Unit positions per subdivision are shown in
{ref("fig-split")}c-d and on the atlas projections in {ref("fig-s-maps")}.

## Do the subdivisions differ in function?

Sensory responses differ between subdivisions ({ref("fig-split")}e, {ref("tbl-tests")}): whisker responses (passive and
active) are most frequent in DLS and TS, auditory responses in DMS. Whisker-choice and spontaneous-lick responses are
similar in all four subdivisions. The tests give the same conclusions with the old labels; the VS percentages change
the most (e.g. auditory active VS {num("aa_vs_old", old.loc["auditory_active", "mean_VS"], "{:.1f}")} % old,
{num("aa_vs_new", new.loc["auditory_active", "mean_VS"], "{:.1f}")} % new), as expected from the units moved between
DMS and VS.

# Discussion

The coordinate fix mostly moves units across the DMS / VS border and leaves DLS and TS nearly unchanged. The
functional profile matches the known input pattern: whisker responses where whisker cortex projects (DLS, and TS), auditory
responses in DMS and TS. Analyses that use the subdivision labels (anything at the `area_acronym_custom` level in the
striatum) need the rebuilt tables; the shared tables are being rebuilt (ssl-table-rebuild-20261117).

# Caveats

- Units are pooled across days for the counts; the response prevalence is day 0 only.
- A session contributes to several subdivisions, so the groups compared are not independent; the tests treat them as
  independent samples.
- The boxes are fixed coordinates, not atlas structures; units near a box border can be assigned to either side
  within the alignment error of the probe (about 100-200 µm).
- No correction across the six comparisons in the tables; with Bonferroni over six, significant comparisons are
  {", ".join(bonf_kw)} (Kruskal-Wallis) and {", ".join(bonf_an)} (ANOVA).

# Figures

{R.figure("fig-split", F / "striatum_split.png", f"Striatal subdivisions before and after the coordinate fix. a, Units (good + multi-units, all days) per subdivision with the old (pale) and new (dark) labels. b, Number of units per old (rows) and new (columns) label. c-d, Unit positions (new labels), top view (absolute ML vs AP) and side view (AP vs DV), bregma-centred. e, Percentage of significant ROC units per session on day 0 for six comparisons (dots: sessions with at least {prov['min_units']} units in the subdivision; markers: mean ± s.e.m.); x labels give the Kruskal-Wallis and one-way ANOVA p across subdivisions.", "projects/ssl-striatum-subdivisions/001_striatum_split.py")}

# Supplementary material

{R.figure("fig-s-maps", F / "area_summary_subdivisions.png", "Positions of the units of each subdivision (new labels) on coronal, sagittal and horizontal projections of the Allen reference volume, with density contours and centroid (brain_wide_analysis plot_area_proj_brain, area_summary).", "projects/ssl-striatum-subdivisions/001_striatum_split.py", kind="supplementary")}

{R.table("tbl-counts", counts.pivot(index="subdivision", columns="labels", values=["units", "sessions", "mice"]).reindex(SUBS).pipe(lambda d: d.set_axis([f"{a} ({b})" for a, b in d.columns], axis=1)).reset_index(), "Units, sessions and mice per subdivision with the old and new labels (all days).")}

{R.table("tbl-tests", new.reset_index().assign(comparison=lambda d: d.analysis_type.map(TYPES))[["comparison"] + [f"mean_{s}" for s in SUBS] + ["kw_p", "anova_p"]].round(3).rename(columns={f"mean_{s}": f"{s} (%)" for s in SUBS} | {"kw_p": "Kruskal-Wallis p", "anova_p": "ANOVA p"}), "Day-0 percentage of significant units per session (mean over sessions, new labels) and tests across subdivisions (sessions = samples).")}

{R.table("tbl-s-boxes", BOXES, "Coordinate boxes of create_areas_subdivisions (allen_utils), bregma-centred, mm (DV positive downwards). Striatal units (CP, STR, STRd, STRv, ACB, FS) are assigned in the order VS, DLS, DMS, TS: the first box that contains the unit wins.")}

# Appendix

Result tables: `combined_results_ks4/ssl-striatum-subdivisions/tables/` (striatal_units.csv, counts_old_vs_new.csv,
transitions_old_to_new.csv, roc_frac_sig_per_session.csv, roc_subdivision_tests.csv); per-subdivision area-summary plots:
`area_summary/striatal_subdivision/<subdivision>/`; provenance.json.
"""
R.write(OUT, md)
