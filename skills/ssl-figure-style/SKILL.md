---
name: ssl-figure-style
description: Use this skill whenever making publication, abstract (e.g. COSYNE), report or summary figures for SSL analyses. It is the single home of the SSL plotting rules - colours (task, cohort, area palette), edge-free shading, PSTH baseline and session averaging, quantification and correlation panels, maps and heatmaps, explicit labels and area names, no-clip margins, layout, and exports (editable text, dense rasters). Manuscript organisation is in ssl-paper-figures.
---

# SSL Figure Style

## Use this skill when
- Building or revising figures for papers, abstracts, posters, talks or project reports from SSL data (matplotlib).
- The manuscript figures additionally follow `ssl-paper-figures` (grid, per-panel CSVs, snapshots, captions); every
  style rule below applies to them, with the Nature format of rule 11 replacing the general layout.

## Rules
1. **Shaded areas have no edges.** Every `fill_between`, `axvspan`, `axhspan`, `hist(histtype="stepfilled")` and
   patch used as a shaded area gets `lw=0, edgecolor="none"` (s.e.m. bands, analysis windows, density fills).
   Matplotlib otherwise draws a thin edge that shows up in PDF/SVG exports.
2. **PSTHs**
   - Baseline: subtract the pre-trial baseline (rate in [−1, −0.015] s before trial start, per trial, averaged per unit
     and condition), not a window of the aligned trace itself, unless the figure states otherwise.
   - Average over **sessions** (the unit of analysis for SSL learning-stage analyses; user rule 2026-10-02): mean
     over units within each session, then mean ± s.e.m. across sessions. The learning day has one session per mouse,
     expert mice can have several, so sessions ≠ mice at the expert stage. State "mean ± s.e.m. over sessions
     (n = …)" in the panel or caption.
   - Shade the analysis window (e.g. the pre-lick window) and mark the alignment event with a thin dashed line.
3. **Quantification panels**: plot the group mean ± s.e.m. (no individual session dots unless asked), connect
   learning → expert within a cohort with a line, open marker = learning, filled = expert. Put the test p-values as
   brackets (within-cohort change in the cohort colour, between-cohort comparison in black) and the interaction in
   the title. Report the unit of analysis (session / mouse) and n in the caption.
4. **Single-cell quantifications** (fractions of units, selectivity): add the average response of the units being
   quantified (e.g. small PSTHs below the panel), averaged as in rule 2, sign-aligned to the selectivity if units of
   both signs are pooled.
5. **Correlation panels** (user, 2026-09): scatter + OLS line + faint 95 % confidence band of the regression line
   (analytic: `t(n−2, 0.975) · s_err · sqrt(1/n + (x − x̄)² / Sxx)`, not bootstrap); line solid if p < 0.05, dashed
   otherwise; n, r and p annotated; dotted reference lines at x = 0 and y = 0. Reference implementation:
   `regress_and_plot` in `projects/ssl-whisker-hitmiss-timeresolved-decoding/exploratory-analyses/030_wholebrain_behavior_correlation.py`.
6. **Explicit labels**: axis labels and titles say what is measured in words, with the reference points (e.g.
   "Similarity of whisker hits to auditory hits (0 = false-alarm-like, 1 = auditory-hit-like)"). Avoid symbols or
   jargon (λ, d′, "transfer") in abstract figures; define them in the caption if used in full figures. Say
   "whisker vs auditory decoding", not "stimulus modality".
7. **Colours**
   - Task: whisker #f7b519, auditory #2c2cdb, false alarm / no-stim #262626 (very dark grey). Cohorts: R+ #00B400,
     R− #C800C8, never matplotlib defaults.
   - Areas (user, 2026-10-06; figures were rejected twice for wrong colours): area groups take exactly
     `ephys_utilities.allen_utils.get_custom_area_groups_colors()[area_group]`; fine areas (`area_acronym_custom`)
     take shades of their group's colour (membership from `get_custom_area_groups_from_name()`). Never the standalone
     `allen_utils` repo palette (older, merged group names). A group missing from the palette raises an error, no
     silent fallback. Area groups are ordered by `allen_utils.get_area_group_custom_order()` unless a figure sorts by
     a value.
   - Maps and heatmaps: CCF maps have a white background and grey Allen contours (not a grey reference image),
     single-hue colormaps for levels (white → cohort colour) and a diverging colormap for differences. Decoding
     accuracy heatmaps use `magma` (dark = low).
8. **Area names** (user, 2026-10-13 / 10-15): somatosensory area groups are written `Ss-orofacial`, `Ss-whisker`,
   `Ss-body` (from `Somatosensory-orofacial` / `-whisker` / `-body`), never "S1 orofacial", "S1 whisker", "S1-whisker",
   "SS-whisker" or "wS1" for the group. They are area groups, not single Allen areas, so "S1" is wrong. Fine areas keep
   their Allen acronyms (e.g. SSp-bfd, SSp-m). Same for labels, captions, reports and abstract text (reports respell
   automatically, skills/project-report `SPELLING`).
9. **No clipped markers or borders** (user, 2026-09-16): a literal axis bound at a reachable data limit gets ~2 %
   headroom (`(0.4, 1.02)` instead of `(0.4, 1.0)` for accuracy, `(-0.02, 1.02)` for fractions); x bounds at the data
   min/max get a small padding. Categorical limits and limits computed with a pad term are already fine.
10. **Statistics shown**: never pick the best of several variants post hoc for display; state thresholds and
    corrections in the caption. Decoding publication figure sets (hit/miss project `112_publication_sets.py`) show the
    learning stage only.
11. **Layout and exports**
    - General figures: two-column width 7.4 in, Arial/Helvetica 5–7.5 pt, panel letters bold capitals (A, B, …) 9 pt aligned on one row
      per figure row (same y, left of each axis), no overlapping text; png at 300 dpi, pdf and svg.
    - Manuscript figures: Nature format instead (183 / 89 mm wide, Arial 5–7 pt, 8 pt bold capital letters, lines
      ≥ 0.25 pt, PNG/JPG at 1200 dpi); details and helpers in `ssl-paper-figures` and its `paperstyle.py`.
    - Editable text everywhere: `pdf.fonttype = 42`, `ps.fonttype = 42`, `svg.fonttype = "none"`.
    - Transparency: figures that are exported to EPS (all manuscript figures) use opaque light colours
      (`paperstyle.light()`) instead of `alpha`, since EPS has no transparency. Other figures may use alpha.
    - Dense scatter / raster layers (e.g. whole-session spike rasters with ~10⁵ dots): draw them with
      `rasterized=True` and save every vector format (pdf, svg, eps) with `dpi=1200`. Matplotlib rasterizes such
      layers at the savefig dpi (default 100), which looks pixelated once imported in Figma. Also save an all-vector
      `<name>_vector.svg` (no rasterization) for editing. Example:
      `projects/ssl-session-raster-examples/exploratory-analyses/001_paper_style_whisker_trial_raster.py`.

## Implementation pointers
- `projects/ssl-paper-figures/paperstyle.py`: Nature rcParams, colours, `light()`, square-cell grid, `save_figure`
  (all formats, rasterized layers at 1200 dpi), source-data workbook.
- `projects/ssl-rastermap-psth-variants/exploratory-analyses/062_pub_convergence_figures.py`: `setup()` (rcParams),
  `letter_row()` (aligned panel letters), `save()`, `mean_panel()` (rule 3), `mouse_psth()` (rules 1–2),
  `load_psth_prestart()` (pre-trial baseline).
