---
name: ssl-figure-methods
description: Use this skill whenever an SSL figure is generated or rebuilt (manuscript main / supplementary figures, project report figures). Every figure gets a concise but precise methods section, written so a colleague can reproduce every panel independently from the NWB files without reading our code - data and inclusion, trial selection, each measure's definition, each panel's computation, statistics.
---

# SSL figure methods

## Rule (user, 2026-11-27)
For each figure we generate, also write its methods section. Purpose: the user hands it to a colleague who reproduces
the results independently. It is a recipe, not a summary: someone with the NWB files, the reference sheet and a
standard scientific Python stack must be able to recompute every plotted number from it alone.

## Where
- Manuscript figures (`ssl-paper-figures`): template `projects/ssl-paper-figures/figures/<Figure>/methods.md`
  (public fork: methods and parameters only, no results); `build.py` fills it into the snapshot as
  `<Figure>_methods.md` (`common.write_methods`), with `{panel.path[:fmt]}` fields taken from the run meta
  (`_build/panel_meta.json`, top-level keys = panel letters, e.g. `{b.n.R+.0}`, `{j.n_null_draws}`; `:p` formats a
  P value). No code-provenance block (user 2026-11-27: not needed for an independent reproduction; commits and
  environment stay in the generation note). An unknown field prints `[?path]`; none may remain in a delivered file.
- **Output formats** (user 2026-11-27, 2026-11-28): the methods (and the caption) are delivered as markdown AND PDF,
  both **generated from LaTeX** (`latex_docs.build_docs`, called by `build.py`; Pandoc + XeLaTeX), plus
  `<Figure>_full.pdf` with the figure, caption and methods. References inside the methods are LaTeX cross-references
  (`hyperref` + `cleveref`): write "panel B" / "panels D-J" (linked automatically), `\cref{eq:name}` for an equation
  (`\begin{equation} ... \label{eq:name} \end{equation}`), `\cref{tab:name}` for a table (`Table: caption
  \label{tab:name}` under a pipe table); start each per-panel paragraph with `**B**` so the caption's letters link to it.
  Formulas that are cited or central go in numbered equations, not inline prose. Details: skill `ssl-paper-figures`,
  Caption. Report figures: export the Methods section as a PDF too (project-report build, also LaTeX).
- Project report figures: a `Methods` section per figure in the report, same content rules.

## Content, in this order
1. **Data**: NWB set (folder, conversion), mice and sessions (selection rule, reference-sheet columns used, learner
   definition), days, n per cohort (filled fields, not typed numbers).
2. **Trial selection**: every exclusion, with the column and value (`perf != 6`, `context != 'passive'`,
   `early_lick == 0`, warm-up block, session-end cut), and where each panel deviates.
3. **Definitions**: every derived quantity a panel uses (reaction time, lick probability, learning curve, chance,
   performance, movement signals and their units), as formulas or exact verbal rules with windows in s / ms and
   thresholds. Name the NWB fields read (`trials.lick_flag`, `start_time`, ...), and the corrections applied to them
   (e.g. the `lick_time` artefact offset, `ssl-lick-alignment`).
4. **Per panel** (`**B**`, `**C**`, ...): input subset -> per-trial value -> aggregation levels (trial -> session ->
   mouse -> cohort, with mean vs median at each level) -> what is plotted (axes, error bars) -> the test (statistic,
   unit of analysis, sidedness, minimum n per group, permutations and seed, multiple-comparison correction). State
   bin edges, alignment rules and index conventions explicitly (which trial is 0, inclusive / exclusive edges).
5. **Statistics conventions** shared by the figure (two-sided; R+ vs R- = Mann-Whitney U and Welch t; cluster tests;
   BH scope), once, so panels can refer to them.

## Writing rules
- Check every statement against the code that produced the snapshot (compute function, shared loaders, external
  functions it calls), not against the spec text or the caption: those drift. Where code and caption disagree, the
  methods follow the code and the disagreement is reported to the user.
- Concise: one short paragraph per panel; no history ("was 15%"), no user/date notes, no motivation beyond one clause.
- Precise: numbers with units; name the library function only where the algorithm is not standard (e.g. the exact
  grid HMM, statsmodels MixedLM REML); give the algorithm itself in words or a formula.
- Self-contained: no repo paths or internal function names; describe
  the operation instead (a colleague may reimplement it).
- Mark anything the code does not determine (task hardware, stimulus parameters, sheet categories assigned by the
  experimenters) as `[TO COMPLETE: ...]` rather than guessing.
- Regenerated with every build; lasting wording changes go into the template.

## Quality gates
- [ ] Every non-drawing panel has a paragraph; every test in the caption appears with its unit of analysis and n.
- [ ] The caption stays short (see `ssl-paper-figures`, Caption): details belong here, not there.
- [ ] Every window, threshold, bin edge and minimum count matches the code constants.
- [ ] No `[?...]` fields left; `[TO COMPLETE]` items listed to the user.
- [ ] LaTeX build clean: no undefined references, no missing glyphs; every `\cref` resolves in `<Figure>_full.pdf`.
- [ ] Code/caption disagreements found while writing reported to the user.
