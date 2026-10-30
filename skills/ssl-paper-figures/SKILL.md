---
name: ssl-paper-figures
description: Use this skill to design and build the main and supplementary figures of the Bisi & Hamon SSL manuscript (Nature format) panel by panel - figure spec, square-panel grid, panel-to-analysis mapping, per-panel source-data CSVs, generation notes, dated output snapshots - and to keep the figure code ready for the clean standalone repo written at submission.
---

# SSL paper figures

## Use this skill when
- Designing, filling, revising or exporting a main or supplementary figure of the Bisi & Hamon manuscript.
- Bringing an analysis from an exploratory project or another repo into a figure.
- Preparing the clean standalone code, source data or generation notes for submission.

## Organisation (user decisions, 2026-10-30)

| Layer | Where | When |
|---|---|---|
| Analysis code | existing repos (`behaviour_analysis`, `ephys_utilities`, `ibl-ai-agent/projects/...`), imported as they are | now |
| Figure code: `spec.yaml`, `compute.py`, `plot.py`, `make_figure.py` per figure, shared `paperstyle.py` | `projects/ssl-paper-figures/` in ibl-ai-agent (public fork; same copy on haas) | now |
| Outputs, one dated snapshot per build | `M:\analysis\Axel_Bisi\manuscripts\Bisi_Hamon\Figures\<FigureN_topic>_<YYYY-MM-DD>\` (`/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/` on haas; resolve with `scripts/axel_bisi_paths.py`) | now |
| Shared caches (e.g. behaviour table over all NWBs) | `.../Bisi_Hamon/Figures/_cache/` | now |
| Clean standalone repo: NWB -> intermediate data -> analyses -> figures | new repo | at submission only |

Figures are named `Figure<N>_<topic>` and `SuppFigure<N>_<topic>` (e.g. `Figure1_behaviour`). A rebuild on the same
day overwrites that day's snapshot; older snapshots are never edited (hand edits from Figma go elsewhere, e.g.
`Figures/FromFigma/`). Each snapshot contains:
```
Figure1_behaviour_2026-10-30/
  Figure1_behaviour.svg  .pdf  .png  .tiff   # A4 page, figure block at final Nature size, editable text
  Figure1_behaviour.md                       # generation note (see below)
  Figure1_source_data.xlsx                   # Nature source data, one sheet per panel
  data/Fig1a_<name>.csv ...                  # one tidy CSV per panel = exactly what is plotted
```

The fork is public: `spec.yaml` and code hold design and parameters only. The figure's message, numbers and the
generation note stay in the snapshot on the NAS. `.gitignore` tracks `projects/ssl-paper-figures/**/*.yaml`.

## Workflow per figure
1. **Design together** (no code yet). Agree on the figure's single message, the panels, their order and letters,
   the grid placement, and for each panel the analysis that produces it (repo, module, function, parameters, data
   scope). Write it into `figures/<FigureN_topic>/spec.yaml` (template: `references/figure_spec_template.yaml`) and
   render the layout immediately: unfilled panels show as labelled placeholder boxes. Ask before changing an agreed
   design.
2. **Fill panel by panel.** For one panel at a time: write `compute_<id>()` returning one tidy DataFrame, run it
   **from scratch on haas** (writes `data/Fig<N><id>_*.csv`), write `plot_<id>(ax, df)` that reads only that CSV,
   re-render, and show the user before moving on. Set the panel's `status` to `filled`, then `approved` once the user
   accepts it. Commit and push the figure code at each filled panel.
3. **Snapshot.** Run the quality gates, write the generation note and a draft legend, export all formats and the
   source-data workbook into the dated snapshot folder.

## Panel-to-analysis mapping (three layers)
- **Shared inputs**: expensive steps several panels need (behaviour table from all NWBs, unit tables) run once per
  build and are cached in `_cache/` with their parameters in a sidecar json.
- **`compute_<id>()`**: imports the analysis function from its repo, calls it with explicit parameters, applies the
  SSL rules the data need (`ssl-trial-exclusion`, `ssl-lick-alignment`, `ssl-valid-data`, mouse-level cohort from the
  reference sheet), and returns a tidy table with the join keys (`mouse_id`, `session_id`; `electrode_group`,
  `cluster_id` for units) plus `cohort` / `reward_group`.
- **`plot_<id>(ax, df)`**: draws from the CSV only, no computation beyond trivial reshaping. Existing functions that
  compute, plot and save in one call (e.g. `behaviour_analysis/beh_plotting_functions.py`) are not called for figures:
  their compute part is imported or factored out, their drawing is rewritten as a `plot_<id>`.

## Keeping the submission cleanup cheap
- Every repo a figure imports from is under git, and the generation note records its commit and whether the working
  tree was dirty. A repo without git (e.g. `~/code/behaviour_analysis` on haas, 2026-10-30) is put under git before its
  first panel is filled; ask the user.
- `spec.yaml` lists the exact function and parameters of every panel: this is the to-do list of the clean repo.
- Plotting reads only CSVs, so `plot.py`, `paperstyle.py` and the layout move to the clean repo unchanged.
- At submission: the clean repo regenerates every panel's CSV from the NWB files, and a diff against the submitted
  snapshot's `data/` is the acceptance test.
- Dropping a panel removes its compute/plot code in the same commit.

## Layout: square grid (user rule, 2026-10-30)
- Figures are built on a grid of **square unit cells**. A panel's axes box is square by default (one cell); a
  rectangular panel spans whole cells (2x1, 1x2, 2x2, ...), so its edges align with the neighbouring panels.
  Axes are positioned by the axes box, not the bounding box, so tick labels never shift alignment.
- Nature format: figure block 183 mm wide (89 mm for a single-column figure), at most ~170 mm tall, placed on an A4
  page at final size for import into Figma. Arial 5-7 pt, panel letters 8 pt bold lowercase on one line per grid row,
  lines >= 0.25 pt. This overrides the 7.4 in width and 9 pt letters of `ssl-figure-style` rule 7 for the paper; all
  other `ssl-figure-style` rules (colours, edge-free shading, PSTH baseline and averaging, quantification panels,
  explicit labels, area names), the area colour palette and the no-clip axis margins apply.
- Exports: SVG with `svg.fonttype = "none"` (text editable in Figma), PDF with `pdf.fonttype = 42`, PNG and TIFF at
  300 dpi minimum. Grid, rcParams, placeholder and export helpers: `references/paperstyle_template.py`
  (copied to `projects/ssl-paper-figures/paperstyle.py`).

## Generation note (`<FigureN_topic>.md`, in the snapshot)
Header: figure message, build date, machine, ibl-ai-agent commit, commit (+ dirty flag) of every imported repo, key
package versions, input data version (NWB set, unit-table version). One section per panel: title; imported
function and repo; parameters; data scope (cohorts, population scope, stage, trial and mouse/session exclusions); n
(mice, sessions, units, trials) per group; unit of analysis; statistics (test, both the non-parametric and the
parametric unpaired test for R+ vs R-, correction, exact p); CSV name and columns. Then the draft legend.

## Quality gates
- [ ] Every non-placeholder panel has a CSV in `data/` and a sheet in the source-data workbook with identical content.
- [ ] Every panel is square or an integer multiple of the unit cell; letters aligned per row.
- [ ] Figure block within Nature size; SVG text editable (`<text>` elements present).
- [ ] Generation note lists commits, parameters, n, unit of analysis and statistics for every panel.
- [ ] No results or figure message in tracked files; figure code committed and pushed.
- [ ] Decoding figure sets show the learning stage only (user rule); learner definition stated when a panel depends on it.
