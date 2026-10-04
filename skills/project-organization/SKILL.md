---
name: project-organization
description: Where a project's code, results, figures and report live (one slug everywhere), how folders are named, and when code is committed and pushed to the user's fork. Use for every project: when creating one, writing outputs, or finishing a work step.
---

# Project organization

## Use this skill when

- Starting a new project, or writing any output (tables, figures, caches, reports) for an existing one.
- Finishing a work step that changed code (commit + push cadence below).
- Renaming or moving project folders.

## One slug, three homes (user decision, 2026-10-04)

Every project has one slug, `ssl-<topic>` (SSL dataset) or `<dataset>-<topic>` otherwise: lowercase, hyphen-separated,
no leading underscore. The same slug is used everywhere:

| What | Where | Versioned |
|---|---|---|
| Code, docs (`question.md`, `TODO.md`, `change-log.md`, `README.md`), build scripts | `projects/<slug>/` in the ibl-ai-agent checkout | yes, pushed to the user's fork |
| Results: tables, caches, figures, provenance | `combined_results_ks4/<slug>/` on the NAS (`/mnt/lsens-analysis/Axel_Bisi/...` on haas, `M:\analysis\Axel_Bisi\...` on Windows; resolve with `scripts/axel_bisi_paths.py`) | no |
| Report: self-contained LaTeX build, PDF, `.md`, `.html` | `combined_results_ks4/<slug>/report/` (see `skills/project-report/SKILL.md`) | PDF / md / html only, to the private reports repo |

- Variants of one project are subfolders of its results folder, never sibling folders with suffixes:
  `ssl-stimulus-arrival-decoding/active/`, `.../passive/`, `.../final_n200/`; `ssl-prelick-convergence/ref_sl/`,
  `.../ref_fa/`; contour levels as `figures_zone90/` inside the project folder.
- Inside a results folder: `figures/` (png + pdf + svg), `tables/` (csv / parquet), `cache/`, `report/`, `provenance*.json`,
  `_snapshots/<YYYY-MM-DD_HHMM>/` for frozen statistics.
- Never write project outputs into another project's folder; a cross-project analysis writes into the folder of the
  project that owns the question and records the source folders in its provenance.
- Older folders that do not follow this (underscore prefixes such as `_roc_prelick_sl`, `_sensory_spatial_maps`,
  `_stimulus_arrival*`, `_within_day_sl`, `_roc_*`; pre-agent plain folders such as `mean_learning_curves`) are migrated
  with a mapping table the user approves first; the old path is kept as a symlink to the new one so running jobs and old
  scripts keep working. Do not rename while jobs that write to the folder are running.

## Versioning: commit and push regularly

- The `projects/` code is versioned in the user's fork of ibl-ai-agent (`.gitignore` tracks `projects/**/*.py`, `*.sh`,
  `*.md`). The fork is **public**; SSL data are unpublished.
- Commit and push at every natural checkpoint: after a script works end to end, after a methodology or scope decision,
  before launching a long remote run, and at the end of a session. Do not let a day of changes accumulate unpushed.
- Stage explicit paths only (never `git add` a whole folder: tracked generated files get swept in). Before every commit,
  check staged `.md` files for result statistics (p-values, effect sizes, percentages, numbers of significant units) and
  keep result-bearing files out; generated report sources (`.qmd`, `.tex`, report `.md`) never go to the fork.
- Keep the local checkout and the haas copy of a script identical (copy before running; check checksums when in doubt).
- Commit messages end with the attribution line required by the session.

## Quality gates

- [ ] Code under `projects/<slug>/`, outputs under `combined_results_ks4/<slug>/`, report under `.../<slug>/report/`.
- [ ] No new underscore-prefixed or suffix-variant result folders.
- [ ] Staged files checked for result numbers; pushed to the fork after the checkpoint.
