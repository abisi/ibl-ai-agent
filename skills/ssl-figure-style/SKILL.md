---
name: ssl-figure-style
description: Use this skill whenever making publication, abstract (e.g. COSYNE) or summary figures for SSL analyses. Gives the plotting conventions (colours, edge-free shaded areas, PSTH baseline and mouse-level averaging, explicit labels, mean ± s.e.m. quantification panels, aligned panel letters) the user asked for.
---

# SSL Figure Style

## Use this skill when
- Building figures meant for papers, abstracts, posters or talks from SSL data (matplotlib).
- Revising an existing figure script (e.g. `projects/ssl-rastermap-psth-variants/exploratory-analyses/062_pub_convergence_figures.py`).

## Rules (user requests, 2026-10-02)
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
5. **Explicit labels**: axis labels and titles say what is measured in words, with the reference points (e.g.
   "Similarity of whisker hits to auditory hits (0 = false-alarm-like, 1 = auditory-hit-like)"). Avoid symbols or
   jargon (λ, d′, "transfer") in abstract figures; define them in the caption if used in full figures.
6. **Colours**: whisker #f7b519, auditory #2c2cdb, false alarm / no-stim #262626 (very dark grey), R+ #00B400,
   R− #C800C8 ([[ssl-cross-area-rplus-rminus-colors]] memory). CCF maps: white background, grey Allen contours,
   single-hue or diverging colormaps.
7. **Layout**: journal width 7.4 in (two-column), Arial/Helvetica 5–7.5 pt, panel letters bold 9 pt aligned on one
   row per figure row (same y, left of each axis), no overlapping text, figures exported as png (300 dpi), pdf and svg
   with editable text (`pdf.fonttype = 42`, `svg.fonttype = "none"`).
8. **Statistics shown**: never pick the best of several variants post hoc for display; state thresholds and
   corrections in the caption.

## Implementation pointers
- `062_pub_convergence_figures.py`: `setup()` (rcParams), `letter_row()` (aligned panel letters), `save()` (png / pdf /
  svg), `mean_panel()` (rule 3), `mouse_psth()` (rules 1–2), `load_psth_prestart()` (pre-trial baseline).
