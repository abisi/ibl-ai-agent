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
- 2026-10-07 within_day/011: reward-lick CD trajectories per class, WH - SL / WH - AH drift and per-class slopes, learning day AND expert sessions, both cohorts (learners, all); mixed-model estimates for both stages. Report skill: numbered equations (report_lib.eq).

## 2026-10-07 (evening)
- Report rebuilt (report/, good + mua units): Part II one section per analysis (011 CD drift, 001 halves, 012 PSTHs per half, 002 slopes, 013 change size, 003 epochs, 015/017 areas with ANOVA, 014 cohort specificity, 016 day-0 vs expert outcome, summary figures 019/006/008/005); numbered equations (19); 020 per-area figures and 018 variants in the supplement.
- Stable-unit switch (PRELICK_UNITS=stable: 051 roots *_stable, 053 relabels non-stable units); full pipeline rerun on stable units started (run/run_stable_pipeline.sh); good + mua results kept as records in sl/.
- within_day 021 figure index (FIGURES.md per results folder).

## 2026-10-08
- 080 single-neuron geometry (cross-validated ROC products), 081 neuron classes (sensory = whisker_active or auditory_active; lick-gated = whisker_choice), 082 class PSTHs, 083 across-days COSYNE figure (2 x 4 square panels, angle with bootstrap CI, per-area geometry with BH q).
- Cleanup (user request): superseded within-day scripts 005, 006, 007, 018 moved to within_day/archive/ (README); stable driver updated (022-024, 080-083 added; archived scripts removed).
- Reports rebuilt without redundant figures (old COSYNE versions, timelines, non-strict area variants, unit-sampling variants removed); report/ (good + mua record) and report_stable/.
- Projection colocation (ssl-sensory-spatial-maps 008) now follows PRELICK_UNITS; stricter contours run (002 at ZONE_PCT=50 and 30; 008 at 50 on stable units). 50 % chosen as the strict level (30 % leaves too few units in some groups).
- Report: Relation to the literature section + reference list (citations checked online); discussion wording on the angle aligned with 083 (no detectable rotation). Both reports rebuilt and rendered.
- within_day/025 (user question): CD normalisation variants (current SL = 0 / AH = 1, unit axis, per unit, not normalised, noise units, Hz without z-scoring) on the same folds; cross-validated length / projection / orthogonal / angle of WH early vs late half; schematic with simulated sessions (grow / rotate / orthogonal; N and event-count scaling). Stable units.
- 2026-10-08: stable driver fixed: 054/055/058 dropped (superseded, need the removed long_rt variant); 059 and 061 moved after 069 (059 needs the decoder-transfer sessions; its empty fallback lacked cohort); 059 / 061 rerun on stable units (exit 0).
- 084: along vs orthogonal summary figure (within day 0 / expert halves from 025, across days from 057 distances): λ = ρ cos θ, ω = ρ sin θ, ρ, θ; schematics, equations, tests.
- 2026-10-08: report: new Part II section 'Along the axis or orthogonal to it?' (084 main; 025 normalisation and half-geometry supplements); 025 and 084 rerun on the good + mua record set; driver runs w025 and 084; superseded pre-merge report generators archived (exploratory-analyses/archive/reports_pre_merge/).
- 2026-10-08: report pruned (user: redundancies out; 56 -> 36 figures): removed the COSYNE summary figures (083, 019), the 9 per-area figures (020), cohorts-by-stage (023) and compact-drift (024) layouts, the 10-WH trial figure (kept 20), epochs with 4 events, decoder schemes, the 080 effect-size duplicate and the 025 half-geometry figures (summarised by 084); day-0 vs expert outcome (016) moved to the supplement. Text points to the result folders instead.
- 2026-10-08: 085 example sessions in state space for the reward-lick axis (x = u, y = WH displacement orthogonal to u, z = remaining PC; class trajectories, WH early / late).
- 2026-10-08: within_day 026 linear-shift null for the within-session WH drift (AH / SL labels shifted 10-50 % against the time-ordered AH + SL events, axis rebuilt, WH projected; unit-norm and per-unit scales).
- 2026-10-08: within_day 026 extended: SL slopes on real / shifted CDs (held-out), WH - SL contrast, binned time courses (real vs shifted), per-area axes (sessions_areas.csv, bins.csv).
- 2026-10-08: within_day 026 saves per-trial real / shifted projections of the example sessions (example_trials.csv).
