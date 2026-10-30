---
name: ssl-paper-figures
description: Use this skill to design and build the main and supplementary figures of the Bisi & Hamon SSL manuscript (Nature format) panel by panel - figure spec, square-panel grid, panel-to-analysis mapping, per-panel source-data CSVs, generation notes - with all code in a standalone, publishable paper repo that reruns every analysis from scratch.
---

# SSL paper figures

## Use this skill when
- Designing, filling, revising or exporting a main or supplementary figure of the Bisi & Hamon manuscript.
- Moving an analysis from an exploratory project into the paper.
- Preparing figure code, source data or generation notes for submission.

## Where things live (user decisions, 2026-10-30)

| What | Where |
|---|---|
| Figure code, specs, shared style, vendored analysis code | paper repo `bisi-hamon-figures` (`C:\Users\bisi\Github\bisi-hamon-figures`, `~/code/bisi-hamon-figures` on haas), private on GitHub until publication |
| Figure outputs | `M:\analysis\Axel_Bisi\manuscripts\Bisi_Hamon\Figures\<FigureN_topic>\` (`/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/` on haas; resolve, never hardcode) |
| Shared caches (e.g. the behaviour table over all NWBs) | `.../Bisi_Hamon/Figures/_cache/`, never inside a figure's `data/` |

Figures are named `Figure<N>_<topic>` and `SuppFigure<N>_<topic>` (e.g. `Figure1_behaviour`). This replaces the
`projects/<slug>` + `combined_results_ks4/<slug>` homes of `skills/project-organization/SKILL.md` for paper figures only.

Each figure folder contains:
```
Figure1_behaviour/
  Figure1_behaviour.svg  .pdf  .png  .tiff   # A4 page, figure block at final Nature size, editable text
  Figure1_behaviour.md                       # generation note (see below)
  Figure1_source_data.xlsx                   # Nature source data, one sheet per panel
  data/Fig1a_<name>.csv ...                  # one tidy CSV per panel = exactly what is plotted
```

## Workflow per figure
1. **Design together** (no code yet). Agree on the figure's single message, the panels, their order and letters,
   the grid placement, and for each panel the analysis that produces it (repo, module, function, parameters, data
   scope). Write it into `figures/<FigureN_topic>/spec.yaml` (template: `references/figure_spec_template.yaml`) and
   render the layout immediately: unfilled panels show as labelled placeholder boxes. Ask before changing an agreed
   design.
2. **Fill panel by panel.** For one panel at a time: vendor the compute function(s) it needs into `src/analysis/`,
   write `compute_<id>()` returning one tidy DataFrame, run it **from scratch on haas** (writes `data/Fig<N><id>_*.csv`),
   write `plot_<id>(ax, df)` that reads only that CSV, re-render, and show the user before moving on. Set the panel's
   `status` to `filled`, then `approved` once the user accepts it.
3. **Finalise.** Run the quality gates below, write the generation note and a draft legend, export all formats and the
   source-data workbook.

## Panel-to-analysis mapping (three layers)
- **Shared inputs**: expensive steps several panels need (behaviour table from all NWBs, unit tables) run once per
  figure build and are cached in `_cache/` with their parameters in the cache file name or a sidecar json.
- **`compute_<id>()`**: calls vendored functions with explicit parameters, applies the SSL rules that the panel's
  data need (`ssl-trial-exclusion`, `ssl-lick-alignment`, `ssl-valid-data`, mouse-level cohort from the reference
  sheet), and returns a tidy table with the join keys (`mouse_id`, `session_id`, and `electrode_group`, `cluster_id`
  for units) plus `cohort` / `reward_group`.
- **`plot_<id>(ax, df)`**: draws from the CSV only, no computation beyond trivial reshaping. Existing plotting code that
  computes, plots and saves in one call (e.g. `behaviour_analysis/beh_plotting_functions.py`) is not reused as is:
  its compute part is vendored, its drawing is rewritten as a `plot_<id>`.

## Publishable-code rules (keep the final cleanup cheap)
- The paper repo never imports from `ibl-ai-agent/projects/` or from unversioned checkouts (e.g.
  `~/code/behaviour_analysis`, not under git). Code is **vendored** into `src/analysis/` when its panel is filled
  (record the origin file in the module docstring), or **pinned** to a tagged release of a versioned library
  (e.g. `ephys_utilities`). Vendor one copy only: pick between duplicates such as `learning_methods` / `learning_methods_bis` with the user.
- Exploration stays outside the paper repo (in `ibl-ai-agent/projects/` or git-ignored `scratch/`); a variant enters
  the paper repo only when it becomes a panel.
- `spec.yaml` is the source of truth: dropping a panel removes its compute/plot code and CSV in the same commit;
  `tools/check_orphans.py` flags functions in `src/analysis/` that no panel reaches.
- Data paths come from `config.yaml` only (NWB / DANDI, reference sheet, output root), resolved per machine.
- Commit and push at each filled panel. Before submission: remove unused code, fresh clone + fresh environment,
  `make_all.py` on haas, and diff the regenerated CSVs against the submitted source data.

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
  300 dpi minimum. Grid, rcParams, placeholder and export helpers: `references/paperstyle_template.py`.

## Generation note (`<FigureN_topic>.md`)
Written from the spec and the run, one section per panel: title; compute module/function and origin of the vendored
code; parameters; data scope (cohorts, population scope, stage, trial and mouse/session exclusions); n (mice,
sessions, units, trials) per group; unit of analysis; statistics (test, both the non-parametric and the parametric unpaired test for R+ vs
R-, correction, exact p); CSV name and columns. Header: paper-repo commit, run date, machine, environment
(key package versions), input data version (NWB set, unit-table version).

## Quality gates
- [ ] Every non-placeholder panel has a CSV in `data/` and a sheet in the source-data workbook with identical content.
- [ ] Every panel is square or an integer multiple of the unit cell; letters aligned per row.
- [ ] Figure block within Nature size; SVG text editable (open the SVG and check for `<text>` elements).
- [ ] Generation note lists parameters, n, unit of analysis and statistics for every panel.
- [ ] No import from `ibl-ai-agent/projects/` or unversioned paths; commit pushed.
- [ ] Decoding figure sets show the learning stage only (user rule); learner definition stated when a panel depends on it.
