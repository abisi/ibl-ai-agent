# TODO / progress log: ssl-whisker-perfquant-pls-regression

## 2026-09-24: project created (split off from ssl-whisker-hitmiss-timeresolved-decoding)

User: "For the performance curves regression, copy all of this apart in a
different project. list caveats and remind briefly the method, and
provide a report." Scope confirmed via AskUserQuestion: the WHOLE
perfquant/PLS pipeline (all 3 regression targets -- whisker_curve,
falsealarm_curve, performance_curve), not performance_curve alone; folder
name `ssl-whisker-perfquant-pls-regression`.

Copied from `ssl-whisker-hitmiss-timeresolved-decoding/exploratory-
analyses/`: all 147 files numbered `049`-`088` that belong to the
PLS/RRR regression branch (excluded `069_perfstate_*` and
`073_hitmiss_*`, which are classification-pipeline files that happen to
share the numeric range). Also copied `METHOD_perfquant_decoding.md` and
`RESULTS_RECAP_perfquant.md` (both already perfquant-scoped in the
source project) -- appended sections 26-28 to the copied
`RESULTS_RECAP_perfquant.md` to cover `086`/`087`/`088`, which postdated
the source doc's last update.

Wrote `REPORT.md`: brief method reminder, a consolidated 12-item caveat
list (pulled from across the source project's `RESULTS_RECAP_perfquant.
md` §5/§7/§12/§19/§21-23/§25/§27/§28), and a results report summarizing
the full arc through §28, plus a "what's still open" section.

This project has NO independent compute of its own yet -- every script
here was written and run in the source project; this is a snapshot/fork,
not a fresh start. Any future work here should follow the same
conventions documented in `METHOD_perfquant_decoding.md` (floored
scaler, valid null construction, PLS+1SE as the working method, always
report R2/Pearson/Spearman together).

## 2026-09-24 (continued): cross-target specificity + weight-vector comparison

User: "how to address drift vs code" -> recommended (2-3 sentence,
exploratory-question convention) a cross-target specificity null +
PLS weight-vector cosine comparison as the cheapest builds on existing
machinery -> user: "Implement cross-target specificity null, compare
weight vector across targets (PLS) only. Make figures of these."

New scripts `089_perfquant_crosstarget_specificity.py` (compute, haas --
3 separate single-output PLS+1SE fits/session instead of 058's one joint
3-output fit, so each target finds its own neuron-weighting
independently; 12-session pilot, same EXAMPLE_SESSIONS as 059-061,
N_SHUF_NULL=200) and `090_perfquant_crosstarget_figures.py` (pure
post-hoc figures + stats from 089's CSVs, local).

Deployed 089 to haas (scp to the source project's exploratory-analyses/,
since that's where the shared `scripts/` dependency resolves correctly
at the same path depth) and ran it there in the background
(`nohup ... &`, launched from the repo root -- the first attempt failed
on every session because it was launched from inside exploratory-
analyses/, breaking a relative-path dependency the same way 034's
REF_PATH issue did earlier this session). 12/12 sessions succeeded.
Pulled the 4 output files back to this project's exploratory-analyses/.

**Result: strong evidence for target-specific coding, not pure shared
drift.** Paired own-target-vs-cross-target test: 33/36 (session x
train_target) rows favor own target, Wilcoxon p=3.2e-9. Weight-vector
cosine similarity is nuanced, not just "shared vs not": whisker/
performance share substantial overlap (+0.629) despite negative
cross-target transfer; falsealarm/performance are significantly
ANTI-correlated (-0.437), consistent with their strongly negative
cross-target scores in both directions. Full detail + caveats (null-
reference approximation, uneven per-cell significance at n=12) logged
in `RESULTS_RECAP_perfquant.md` §29.

Updated `REPORT.md`'s open item 3 ("drift vs. code") to reflect this new
pilot evidence -- partially addressed, not fully resolved (12-session
pilot, not full-pool).

## 2026-09-24 (continued): weight-vector comparison scaled to full pool

User: "For the weight vector, show other visualization and do on al
ldata" [sic]. Scoped to the weight-vector comparison specifically (not
89's null-controlled specificity matrix, left at pilot scale).

New scripts `091_perfquant_weightvector_fullpool.py` (compute, haas --
stripped down from 089: only the 3 single-output PLS+1SE fits + coef_
extraction needed for cosine similarity, no null shuffling at all, which
is what makes a full 88-session run cheap) and `092_perfquant_
weightvector_figures.py` (3 new figures, pure post-hoc, local).

Deployed/launched on haas the same way as 089 (repo-root launch, scp to
the source project's exploratory-analyses/). 88/89 sessions usable.
Pulled results back, built figures.

**Result: full-pool confirms and sharpens the 12-session pilot, now at
q<1e-9 for all 3 pairs.** whisker-performance +0.707 (strongest shared
axis), whisker-falsealarm +0.348, falsealarm-performance -0.301
(confirmed anti-correlated). Cohort-split distribution shows no R+/R-
separation (consistent with every other cohort check this project has
run). Neuron-level loadings scatter makes concrete what a given cosine
value looks like -- caveat noted: both "highest cosine" whisker-paired
examples land on the same low-unit-count session (MH015, n=19), likely
partly a small-n sampling-variance artifact rather than the most
trustworthy example; AB082 (n=302, falsealarm-performance highest) is
the more convincing near-perfect-alignment example.

Full detail in `RESULTS_RECAP_perfquant.md` §30.
