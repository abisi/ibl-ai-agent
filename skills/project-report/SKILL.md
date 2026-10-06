---
name: project-report
description: Each project keeps a scientific-article report (PDF built from LaTeX, with a Markdown and an HTML equivalent) in combined_results_ks4/<slug>/report/. Regularly ask the user whether it is a good time to build it; rebuild automatically after many changes; publish PDF/md/html privately. Use in every project session.
---

# Project report (article-style PDF + Markdown)

## Use this skill when

- Working on any project (user rule, 2026-10-04: applies to all projects).
- A checkpoint is reached (see "When to ask"), or the user asks for a report, PDF, write-up or summary of a project.

## When to ask, when to rebuild

- **Ask regularly**: at natural checkpoints -- a group of analyses or figures completed, a result changed after a rerun,
  a methodology decision made, the end of a working session -- ask once: *"Is this a good time to build the report PDF of
  `<slug>`?"* Do not ask more than once per checkpoint, and not in the middle of a running computation whose results the
  report would need.
- **Rebuild automatically** (without asking) after a lot of changes since the last build: a new main or supplementary
  figure, changed main statistics, a methodology or scope change, or several result-changing reruns. Tell the user it was
  rebuilt and what changed.
- If the user asked not to render for now, hold the rebuild until they lift it.
- Record every build in the project's `change-log.md` (date, what changed in the report).

## Where and what

`combined_results_ks4/<slug>/report/` is self-contained -- everything needed to rebuild the PDF is inside:

```
report/
  build.sh              # copy of skills/project-report/build.sh: report.md -> report.tex (Pandoc) -> report.pdf (latexmk -xelatex) + report.html
  report.tex            # LaTeX source (standalone article)
  references.bib        # if citations are used
  figures/              # copies of every figure used (png for the PDF / html, plus pdf if vector is wanted)
  report.pdf            # built article
  report.md             # Markdown equivalent of the PDF (same text, equations in $...$, figures referenced from figures/)
  report.html           # self-contained HTML (images embedded)
  numbers.json          # every number quoted in the text, read from the result tables
```

- The generator is versioned code: `projects/<slug>/report/build_report.py` (+ `render.sh`). It reads the result tables,
  writes `numbers.json`, `report.md` (single source: Pandoc Markdown with LaTeX math and figure/table captions), converts it
  to `report.tex` with Pandoc (`quarto pandoc report.md -s -o report.tex` with the article template), copies the figures,
  and builds `report.pdf` with XeLaTeX (`latexmk -xelatex`, TinyTeX installed through Quarto; Arial, so Greek letters, µ, −, ≥
  render) and a self-contained `report.html` -- all done by `build.sh` (copy it into the report folder). Captions carry their
  own labels ("**Figure 3.**", "**Figure S2.**"; LaTeX auto-labels are off). No number in the text is typed by hand.
- Compute and table generation run on haas; LaTeX / Pandoc build where the toolchain is (locally: TinyTeX via Quarto).

## Content: read like a scientific article, project in its latest form

1. **Title, date, data version** (spike sorting, unit table version, sessions / mice, populations).
2. **Abstract** (5-8 sentences) and **key results** (numbered, each with effect size, test, p, n).
3. **Introduction**: motivation, question, hypotheses, how the analyses address them.
4. **Methods** in detail: data and inclusion (trial / unit / session rules, cohort source), every quantity with its
   equation, every parameter (windows, bins, thresholds, smoothing, iterations, permutations), statistics (test, unit of
   analysis, null, correction), software / scripts used.
5. **Results** organised by question, each subsection: claim, quantification (numbers with CI / error), test, figure
   reference; tables for per-area numbers.
6. **Discussion**: interpretation, relation to other projects, alternative explanations.
7. **Caveats and limitations**; **open questions / next steps**.
8. **Figures** (main) and **Supplementary figures** (controls, per-area, variants), numbered, each with a full caption
   (what is plotted, n, statistics, colour code); main figures are the publication-style figures of the project.
9. **Appendix**: parameter table, file / script index (paths of results and code), version history of major changes.

Describe the current state only (no history of abandoned versions in the main text; superseded results go to the
version-history appendix).

### Defaults for every report (user, 2026-10-06)

- **Cross-reference everything.** Every figure and table (main and supplementary) is cited in the text where its result
  is stated ("Figure 2b", "Table S1"), with clickable links: give each figure / table a Pandoc id
  (`![caption](figures/x.png){#fig:x}`, table caption `: caption {#tbl:x}`) and link with `[Figure 2](#fig:x)`. No figure or
  table may be left uncited; check this before building.
- **Paragraphs.** One idea per paragraph, separated by a blank line in `report.md` (never one long block per section).
- **Thorough captions.** Every caption states what is plotted in each panel, the data (epoch, trials, n sessions / mice /
  units or areas), the statistic (mean ± what, error bars, bands), the test and n, the colour / marker code, and the
  sampling (iterations, shuffles) -- readable without the main text.
- **All generated figures appear.** Every figure an analysis script writes ends up in the report, at least as a
  supplementary figure; promote a supplementary figure to the main text only when the argument needs it.
- **Sample sizes.** A supplementary table with the number of sessions, mice and units per area (and per condition /
  epoch / level analysed), cited from the Methods.
- **Order follows the argument** the user sets (e.g. control condition before the condition of interest, then their
  comparison); ask when unclear.

## Publishing

- Reports are published to the **private** repository `abisi/ibl-ai-agent-reports` under `docs/<slug>/`: `report.pdf`,
  `report.md`, `report.html` only -- **no figure files, no data, no tables** (user, 2026-10-04); update `docs/index.html`.
- Clone or update the repository in a scratch / publish folder, copy the three files, commit
  ("Update <slug> report"), push. Never push to a public repository without an explicit opt-in in this session.
- The code fork (public) never receives report sources, PDFs or result-bearing Markdown.

## Quality gates

- [ ] Asked at the checkpoint (or rebuilt automatically after many changes) and logged in `change-log.md`.
- [ ] `report/` rebuilds from its own contents; PDF, md and html agree; all numbers from `numbers.json`.
- [ ] Article structure complete: introduction, methods (equations, parameters, tests), results, discussion, caveats,
      main + supplementary figures with captions.
- [ ] Every figure and table cited and linked in the text; every generated figure included (main or supplementary);
      sample-size table present; paragraphs separated; captions complete.
- [ ] Published to the private reports repo (pdf, md, html only).
