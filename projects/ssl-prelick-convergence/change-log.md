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
- 2026-10-08: 026 example trials include AH (real / shifted held-out projections).
- 2026-10-08: 086 reward-lick CD definition on real data (AB138: AH / SL trajectories from a common start, before / after orthogonalising the whisker-hit direction).
- 2026-10-08: within_day 026 uses 50 linear shifts (as hit/miss 135; was 20); 086 redrawn in 3D (u = AH - SL, w = WH - SL, d = WH late - early half; common start offset; before / after orthogonalisation; no grid / panes).
- 2026-10-08: within_day 027 time-resolved projection on the reward-lick CD (026 folds and units; held-out AH / SL, WH on every fold; -400 to +200 ms around the first lick; within_day/<ref>/cd_psth/cd_psth_projection.parquet).
- 2026-10-08: 026 example sessions re-picked as representative (closest to the cohort mean excess, >= 20 WH, |null| < 0.03; SSL_026_EXAMPLES override).
- 2026-10-08: 087 across days with the within-day metric definitions (026 folds / scaling; per-neuron WH - SL position on the unit-norm CD, lambda, cross-validated angle and length; cohort x stage mouse permutation); across_days/<ref>/harmonised/.
- 2026-10-09: within_day 026 / 027 SSL_026_RAW=1: CD from centred but unscaled rates (no z-scoring); outputs cd_shift_null_raw, cd_psth_raw (examples AB147 / MH062 via SSL_026_EXAMPLES).
- 2026-10-09: 088 COSYNE across days v2 (083 kept): example neurons (raster + PSTH), whisker-hit modulation index MI = (|WH - SL| - |WH - AH|) / (|WH - SL| + |WH - AH|) replaces AH-likeness (no class squares), class-square legend on the converging fraction, day-0 R+ vs R- brackets, population geometry / lambda / angle / length / per-area change from 087 (current CD definition).
- 2026-10-09: 088: f = session-mean projections of WH / AH / SL on the CD (replaces the geometry plane), g = WH - SL projection (per neuron; replaces the 0-1 normalised lambda), i labelled as the change of the WH - SL projection.
- 2026-10-10: 089 COSYNE across days v3 (083 / 088 kept): row 1 PSTHs, example converging non-sensory neuron, converging population PSTH, fraction of reward-lick neurons, converging fraction; row 2 schematic slot, CD projection time courses at both stages (027 SSL_027_ALLSTAGES=1), WH - SL projection, angle / length decomposition, per-area change; projections only, no 0-1 scaling.
- 2026-10-10: 088 / 089: number of sessions under each group tick (D0 / E per cohort).
- 2026-10-10: 090 exploratory RT adjustments of the reward-lick CD across days (WH / AH with RT >= 150 ms; WH vs AH RT-matched in 20-ms bins, 10 draws; spontaneous licks unchanged); across_days/<ref>/rt_adjusted/.
