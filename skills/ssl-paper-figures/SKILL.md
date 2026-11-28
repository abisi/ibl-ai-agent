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
| Clean standalone repo: NWB -> intermediate data -> analyses -> figures | new repo | later, once figures are approved; plan in `references/clean_code_repo_plan.md` |

Figures are named `Figure<N>_<topic>` and `SuppFigure<N>_<topic>` (e.g. `Figure1_behaviour`). A rebuild on the same
day overwrites that day's snapshot; older snapshots are never edited (hand edits from Figma go elsewhere, e.g.
`Figures/FromFigma/`). Each snapshot contains:
```
Figure1_behaviour_2026-10-30/
  Figure1_behaviour.pdf .svg .ai .eps .png .jpg  # A4 page, figure block at final Nature size, editable text
  Figure1_behaviour.md                       # generation note (see below)
  Figure1_behaviour_caption.md / .pdf         # figure legend draft, n and statistics filled in; LaTeX-generated (see below)
  Figure1_behaviour_methods.md / .pdf         # methods for independent reproduction (skill ssl-figure-methods); LaTeX-generated
  Figure1_behaviour_full.pdf                  # figure + caption + methods in one PDF, hyperref / cleveref cross-references
  _build/latex/*.tex                          # the LaTeX sources of the three PDFs (the .md are converted from them)
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
   Write or update the panel's paragraph in `figures/<FigureN_topic>/methods.md` at the same time (skill
   `ssl-figure-methods`).
3. **Snapshot.** Run the quality gates, write the generation note, the caption draft and the methods, export all formats and the
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
  page at final size for import into Figma. Arial 5-7 pt, panel letters 8 pt bold capitals (A, B, …; user 2026-11-11; drawn and captioned from the lowercase panel `id`, which
  CSV, sheet and meta names keep, e.g. `Fig1a`) on one line per grid row,
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
- **Section titles** (user rule, 2026-11-25): a figure made of several row blocks (e.g. whole brain / areas / learning)
  gets one short title per block, centred, spanning the full figure width, in a light grey box (`0.92` fill, no
  edge, 6.5 pt). Declared in the spec as `sections: [{title, y_mm[, h_mm]}]` (planner coordinates, box 3.0 mm tall by
  default -- user 2026-11-26: thin bar, was 4.5) and drawn by `build.py figure_sections()`; the block's first row of
  panels starts 1 mm below the bar, and the previous block ends 1 mm above the next bar (user 2026-11-26: tighter, was
  1.5 / 2 mm). The panels' own bottom margins (axis labels) already separate the blocks; no extra white band.
  The title states the block's finding or question in a few words, not a method.
- **Panel headings explicit but short** (user rule, 2026-11-26): a panel's `heading` names what is measured and, when
  relevant, the comparison, in about 3-6 words, e.g. "Baseline rate change", "Evoked responses, pre vs post",
  "Cohort × stimulus interaction", "Effect vs choice coding, per area" -- not a bare noun ("Baseline", "Pupil",
  "Interaction") and not a sentence, a result or a method.
- **Panel headings aligned with the panel** (user rule, 2026-11-25): a panel's `heading` starts at the left edge of
  its axes (cell `x_mm` + left margin), not right after the letter, which stays at the cell's top-left corner.
  `build.py panel_heading()` does this from the cell's `margins_mm` (order: left, bottom, top, right).
- **Firing rates in spks/s, never Hz** (user rule, 2026-11-26): every axis label, colour-bar label, legend and panel
  text that shows a firing rate (or a change of one) writes the unit as "spks/s", e.g. "Evoked rate (spks/s)",
  "Baseline change (spks/s)"; captions write "spikes/s". "Hz" stays only for true frequencies (camera, stimulus,
  oscillations).

## Generation note (`<FigureN_topic>.md`, in the snapshot)
Header: figure message, build date, machine, ibl-ai-agent commit, commit (+ dirty flag) of every imported repo, key
package versions, input data version (NWB set, unit-table version). One section per panel: title; imported
function and repo; parameters; data scope (cohorts, population scope, stage, trial and mouse/session exclusions); n
(mice, sessions, units, trials) per group; unit of analysis; statistics (test, both the non-parametric and the
parametric unpaired test for R+ vs R-, correction, exact p); CSV name and columns.

## Caption (`<FigureN_topic>_caption.md`, in the snapshot)
**LaTeX-generated** (user rule 2026-11-28): the caption and methods `.md` and `.pdf` both come from LaTeX, and the
snapshot also holds `<Figure>_full.pdf` = figure + caption + methods. `build.py` calls `latex_docs.build_docs`:
spec captions and the filled methods template -> LaTeX fragments (Pandoc) -> three XeLaTeX documents (latexmk; Arial,
A4): `_full` (figure cropped to its rows, one invisible anchor per panel letter, legend, Methods), `_caption` and
`_methods` (each alone; labels of the other parts through xr-hyper, links open `_full.pdf`). The `.md` files are
Pandoc conversions of the same LaTeX, every reference resolved from the full document's `.aux`.
- Cross-references use `hyperref` + `cleveref` (`\cref`): "panel B", "panels B, D-J", "Panel F:" and "As f," are
  linked to the panel in the figure automatically; the caption's panel letters link to the panel's methods paragraph
  (a methods line starting `**B**`); "details in Methods" links to the Methods. Equations: `\begin{equation} ...
  \label{eq:name} \end{equation}` in the template, cited `\cref{eq:name}`; tables: Pandoc pipe table with caption
  `Table: text \label{tab:name}`, cited `\cref{tab:name}`; the figure: `\cref{fig:main}`. Raw LaTeX passes through
  the markdown templates. Check the build output: undefined references and glyphs missing from Arial are printed.
- Toolchain: Pandoc (Quarto's bundled one) + TinyTeX (`quarto install tinytex`), present on the PC, not on haas. On
  haas `build.py` skips this step with a warning; then run locally
  `python projects/ssl-paper-figures/latex_docs.py figures/<Figure> <snapshot_dir>`.
- A figure taller than A4 needs `grid: page_mm: [210, H]` in the spec, otherwise its lower rows fall off the page.
A concise Nature-style legend (user rule 2026-11-27: "way too long", match what papers show): `**Fig. N | Title.**`
then `**A**, ...` per panel, then a short footer (learner definition, cohort colours, error bars, test pair, star code,
"details in Methods"). Written by `common.write_caption` from the spec:
- `caption.title` and `caption.footer` at figure level, `caption` per panel (YAML folded block `>-`).
- Each panel caption: one or two sentences on what is plotted (measure, alignment, what lines / shading / bars mark),
  the main test(s) with exact P (R+ versus R-: Mann-Whitney and Welch both), and n per group. Typically 30-80 words.
  No algorithms, windows, bin edges, permutation counts, secondary tests or history: those go in the methods
  (`ssl-figure-methods`). Long lists of significant bins / clusters are replaced by the marks on the plot.
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
