# change-log

- 2026-10-03: project created by splitting the pre-lick convergence analyses (051–071, shared loaders 026/045/047/048/049,
  run drivers, article builder) out of `ssl-rastermap-psth-variants`; question.md, TODO.md, LOCKED.md written; next
  step added: within-session remapping test (072).
- 2026-10-03: within-session test (072) designed with the user (split at median AH, count-matched halves, Δd/λ/decoders, odd/even null, controls; SL only) and run; TODO updated.
- 2026-10-03: 072 moved to the separate project ssl-within-day-remapping (001) at the user's request.
- 2026-10-05: article-style report built (skills/project-report): combined_results_ks4/ssl-prelick-convergence/report/ (report.tex, report.pdf, report.md, report.html, figures/); published (PDF, md, html) to the private repo abisi/ibl-ai-agent-reports, docs/ssl-prelick-convergence/.
  Generator: exploratory-analyses/build_report_article.py (main text: build_article.py in Markdown mode, variant SL / learners; variants, archives v1-v3, FA and all-mice controls, projection-zone null added).


## Part II: within days (history of ssl-within-day-remapping)


- 2026-10-03: project created (user request: within-day change with expert comparison as a separate analysis); 072 moved here as 001; 002 trial-level slopes added; question.md, TODO.md written.
- 2026-10-03: 003 epoch comparison (within-day vs across-day, common footing) added at the user's request (no mouse pairing).
- 2026-10-03: 003 rerun with 4 events per class (user request, to recover R− expert sessions).
- 2026-10-03: 004 per-half PSTHs and 005 COSYNE within-day summary figure added (user request).
- 2026-10-04: mixed model (009), decoder schemes (003 options, 010), expanded figure (008), plain-language captions with CD wording and equations (user request).
- 2026-10-05: article-style report built (skills/project-report): combined_results_ks4/ssl-within-day-remapping/report/ (report.tex, report.pdf, report.md, report.html, figures/); published (PDF, md, html) to the private repo abisi/ibl-ai-agent-reports, docs/ssl-within-day-remapping/.

- 2026-10-07 Projects merged (user): ssl-within-day-remapping -> Part II of ssl-prelick-convergence. Code `exploratory-analyses/within_day/` (CONV = parent folder); results moved to `combined_results_ks4/ssl-prelick-convergence/{across_days/{sl,fa},within_day/sl}/` with symlinks at `_roc_prelick{,_sl}`, `_within_day_sl` (manifest results_home_manifest_20261007.tsv); 051 defines HOME / OUTROOT / WITHIN; references updated in ssl-sensory-spatial-maps 008 + report, hit/miss 154, dataset record. Smoke test: imports on haas, within_day/005 end to end.
- 2026-10-07 Merged report (report_lib): Part I across days (Figures 1-5) and Part II within days, learning day and expert sessions (Figures 6-9, mixed model / halves / epoch tables), 19 supplementary figures, sample-size tables, math definitions; learners, SL reference, current settings (rerun pending). report/build_report.py + render.sh.
