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
| Figure code: `spec.yaml`, `compute.py`, `plot.py` per figure; shared `build.py` (assembly, exports, note, caption), `paperstyle.py`, `common.py` | `projects/ssl-paper-figures/` in ibl-ai-agent (public fork; same copy on haas) | now |
| Outputs, one dated snapshot per build | `M:\analysis\Axel_Bisi\manuscripts\Bisi_Hamon\Figures\<FigureN_topic>_<YYYY-MM-DD>\` (`/mnt/lsens-analysis/Axel_Bisi/manuscripts/Bisi_Hamon/Figures/` on haas; resolve with `scripts/axel_bisi_paths.py`) | now |
| Shared caches (e.g. behaviour table over all NWBs) | `.../Bisi_Hamon/Figures/_cache/` | now |
| Clean standalone repo: NWB -> intermediate data -> analyses -> figures | new repo | at submission only |

Figures are named `Figure<N>_<topic>` and `SuppFigure<N>_<topic>` (e.g. `Figure1_behaviour`). A rebuild on the same
day overwrites that day's snapshot; older snapshots are never edited (hand edits from Figma go elsewhere, e.g.
`Figures/FromFigma/`). Each snapshot contains:
```
Figure1_behaviour_2026-10-30/
  Figure1_behaviour.pdf .svg .ai .eps .png .jpg  # A4 page, figure block at final Nature size, editable text
  Figure1_behaviour.md                       # generation note (see below)
  Figure1_behaviour_caption.md               # figure legend draft, n and statistics filled in (see below)
  Figure1_source_data.xlsx                   # Nature source data, one sheet per panel
  data/Fig1a_<name>.csv ...                  # one tidy CSV per panel = exactly what is plotted
```

The fork is public: `spec.yaml` and code hold design and parameters only. The figure's message, numbers and the
generation note stay in the snapshot on the NAS. `.gitignore` tracks `projects/ssl-paper-figures/**/*.yaml`.

## Workflow per figure
1. **Design together** (no code yet). Agree on the figure's single message, the panels, their order and letters,
   the grid placement, and for each panel the analysis that produces it (repo, module, function, parameters, data
   scope). Write it into `figures/<FigureN_topic>/spec.yaml` (template: `references/figure_spec_template.yaml`; worked example `figures/Example_behaviour/spec.yaml`) and
   render the layout immediately (`build.py figures/<FigureN_topic>`): unfilled panels show as labelled placeholder
   boxes. The user may draft panels in the figure planner page (see "Figure planner" below); read it with ArtifactData. Letters follow
   reading order (left to right, top to bottom); re-letter once the panel list is final. Ask before changing an agreed
   design.
2. **Fill panel by panel.** For one panel at a time: write `compute_<id>()` returning one tidy DataFrame, run it
   **from scratch on haas** (writes `data/Fig<N><id>_*.csv`), write `plot_<id>(ax, df)` that reads only that CSV,
   re-render, and show the user before moving on. Set the panel's `status` to `filled`, then `approved` once the user
   accepts it. Commit and push the figure code at each filled panel.
   Write the panel's `caption` in the spec at the same time as its plot.
3. **Snapshot.** Run the quality gates, write the generation note and the caption draft, export all formats and the
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
- Every repo a figure imports from is under git, and the generation note records its commit and the files whose
  content differs from it. Import the canonical git checkouts on the NAS (`<Axel_Bisi>/behaviour_analysis`,
  `<Axel_Bisi>/NWB_reader`, GitHub `abisi/*`; `common.EXTERNAL_REPOS`), never the old `~/code` copies on haas. Never
  commit to those shared repos yourself: code they lack (e.g. `learning_curve_exact.py`) is vendored in
  `projects/ssl-paper-figures/` with a header saying where it belongs, and the user moves it.
- Learning-curve panels use the exact re-fit (`learning_curve_exact`, sigma 1, 80% CI) computed from the NWB files,
  not the stored PyMC curves (user, 2026-11-02; comparison in `checks/001_exact_vs_stored_curves.py`).
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
  lines >= 0.25 pt. Every other style rule (colours and area palette, shading, PSTHs, quantification and correlation
  panels, labels, area names, margins, dense rasters) is in `ssl-figure-style`, the single home of SSL style rules.
- Exports (user, 2026-10-30): PDF, SVG, AI, EPS, PNG, JPG. SVG with `svg.fonttype = "none"` (text editable in Figma);
  PDF, AI and EPS with `pdf.fonttype` / `ps.fonttype = 42` (embedded TrueType, editable text). The `.ai` file is the
  PDF saved under `.ai`: Illustrator opens it as a PDF-compatible AI document with editable text (Arial must be
  installed). EPS has no transparency, so shaded bands and scatter use opaque light colours (`paperstyle.light()`),
  never `alpha`. PNG and JPG at 1200 dpi (Nature line-art resolution); rasterized layers inside PDF, SVG and EPS are
  also written at 1200 dpi. Grid, rcParams, colours, placeholder and export
  helpers: `projects/ssl-paper-figures/paperstyle.py` (canonical); paths, external-repo imports, provenance and the
  R+ vs R- test pair: `projects/ssl-paper-figures/common.py`. Worked example (dummy figure, 14 behaviour panels):
  `projects/ssl-paper-figures/figures/Example_behaviour/`.

## Generation note (`<FigureN_topic>.md`, in the snapshot)
Header: figure message, build date, machine, ibl-ai-agent commit, commit (+ dirty flag) of every imported repo, key
package versions, input data version (NWB set, unit-table version). One section per panel: title; imported
function and repo; parameters; data scope (cohorts, population scope, stage, trial and mouse/session exclusions); n
(mice, sessions, units, trials) per group; unit of analysis; statistics (test, both the non-parametric and the
parametric unpaired test for R+ vs R-, correction, exact p); CSV name and columns.

## Caption (`<FigureN_topic>_caption.md`, in the snapshot)
A Nature-style legend draft, deliberately exhaustive so the user can cut it down: `**Fig. N | Title.**` then
`**a**, ...` for every panel, then a footer with the shared methods (trial exclusions, cohort definition, colours,
error bars, tests, multiple-comparison handling). Written by `common.write_caption` from the spec:
- `caption.title` and `caption.footer` at figure level, `caption` per panel (YAML folded block `>-`).
- Each panel caption says what is plotted (axes, measure, unit of analysis, what dots/lines/shading are), the
  analysis and its parameters in words, n per group, and the test with its statistic and exact P.
- Numbers are never typed: `{path[:fmt]}` fields are filled from the panel's run meta (`_build/panel_meta.json`;
  `_shared` fields available everywhere), e.g. `{n.R+.mice}`, `{stats.welch_t:.2f}`, `{stats.mannwhitney_p:p}`
  (`:p` writes `*P* = 0.025` or `*P* = 9.9 × 10⁻⁴`). An unknown field prints `[?path]`; a panel without caption
  prints `[caption missing: ...]`; both must be gone before submission.
- Check every methods statement against the code it describes (rule, thresholds, window) before writing it.
- Spec captions stay descriptive (public fork); wording the claim and final edits happen in the NAS caption or the
  manuscript. Snapshots are regenerated, so carry lasting wording changes back into the spec.

## Quality gates
- [ ] Every non-placeholder panel has a CSV in `data/` and a sheet in the source-data workbook with identical content.
- [ ] Every panel is square or an integer multiple of the unit cell; letters aligned per row.
- [ ] Figure block within Nature size; SVG text editable (`<text>` elements present).
- [ ] Generation note lists commits, parameters, n, unit of analysis and statistics for every panel.
- [ ] Caption has every panel, no `[?...]` or `[caption missing` left, and its n and P match the generation note.
- [ ] No results or figure message in tracked files; figure code committed and pushed.
- [ ] Decoding figure sets show the learning stage only (user rule); learner definition stated when a panel depends on it.

## Figure planner
Source: `assets/figure_planner.html`, published as a private claude.ai artifact (URL in the user's memory). Republish
from this file with the Artifact tool, keeping the same URL.
- **Data** (ArtifactData; shared db):
  - `figures/<Figure>` holds the figure (name, message, grid `n_cols`/`n_rows`, `order`). `deleted: true` marks a figure
    in the trash.
  - `figures/<Figure>/panels/<pid>` holds one panel: `key`, `letter` (follows the layout), `title`,
    `x_mm`/`y_mm`/`w_mm`/`h_mm`, `analysis`, `params`, `shows`, `stats`, `style`, `notes`, `status`.
  - `history/<id>` logs every saved change as `{ts, path, figure, panel, kind (edit|create|delete), fields {old, new}
    or data, user}`. The page offers Revert and Restore from it.
- **Updating a figure from the planner:**
  1. List the figure's panels.
  2. Read `history` since the last build to see what changed.
  3. Map planner keys to spec keys (they can differ, e.g. `fa_curves` -> `aligned_curves`).
  4. Copy the cells and letters into `spec.yaml`.
  5. Fix caption cross-references.
  6. Re-run the **full** compute: caption meta is stored per letter, so re-lettered panels otherwise keep stale values.
  7. Rebuild.
- **Deleted figures:** move the code to `figures/_deleted/<Figure>_<date>/` and the snapshots to `Figures/_deleted/`,
  then set `cleanup: done` on the figure doc.
- No comments or requests on the page (user 2026-11-09); the history replaces them.
