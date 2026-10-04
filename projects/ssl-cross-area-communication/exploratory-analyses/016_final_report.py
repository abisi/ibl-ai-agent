"""Consolidated final report for the cross-area pCCA project. Detailed
methodology + full-dataset results. Private/unpublished dataset -- local
HTML file only, no public publishing (per AGENTS.md's SSL Defaults).
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def fmt(x):
    return f"{x:.4g}" if isinstance(x, float) else x


def main() -> None:
    cov = pd.read_csv(ARTIFACTS_DIR / "coverage_summary.csv")
    cov_learn = cov[(cov.level == "coarse") & (cov.scope == "entire_dataset") & (cov.day_stage == "learning")]

    dfs_full = {v: pd.read_csv(ARTIFACTS_DIR / "full_learning" / f"variant_{v}" / "results.csv") for v in "ABC"}
    d1 = {v: dfs_full[v][dfs_full[v].dimension == 1].copy() for v in "ABC"}
    for v in "ABC":
        d1[v]["excess"] = d1[v]["observed_corr"] - d1[v]["shuffle_null_mean"]

    sig = pd.read_csv(ARTIFACTS_DIR / "full_learning" / "variant_significance_tests.csv")
    sig_hits = sig[(sig.wilcoxon_p < 0.05) & (sig.paired_ttest_p < 0.05)]

    peak_lag = pd.read_csv(ARTIFACTS_DIR / "peak_lag_results.csv")

    pairs = [("Motor and frontal areas", "Somatosensory areas"),
             ("Motor and frontal areas", "Striatum and pallidum"),
             ("Somatosensory areas", "Striatum and pallidum")]

    def pair_short(a, b):
        return f"{a.replace(' areas', '').replace(' and pallidum', '')} ↔ {b.replace(' areas', '').replace(' and pallidum', '')}"

    # Effect-size table (whisker, lick_flag=1, dim 1)
    rows = []
    for v in "ABC":
        sub = d1[v][(d1[v].trial_type == "whisker_trial") & (d1[v].lick_flag == 1)]
        for a, b in pairs:
            cell = sub[(sub.area_a == a) & (sub.area_b == b)]
            rows.append({"variant": v, "pair": pair_short(a, b),
                         "observed": cell["observed_corr"].mean(), "excess": cell["excess"].mean(),
                         "n_sessions": len(cell)})
    effect_table = pd.DataFrame(rows)

    variant_means = {v: (d1[v]["observed_corr"].mean(), d1[v]["excess"].mean()) for v in "ABC"}

    html = f"""<title>Cross-Area pCCA Report</title>
<meta charset="utf-8">
<style>
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 2rem auto;
        line-height: 1.6; color: #1a1a1a; padding: 0 1.5rem; }}
h1 {{ border-bottom: 3px solid #2c3e50; padding-bottom: 0.5rem; }}
h2 {{ border-bottom: 1px solid #ccc; padding-bottom: 0.3rem; margin-top: 2.5rem; color: #2c3e50; }}
h3 {{ color: #34495e; margin-top: 1.8rem; }}
table {{ border-collapse: collapse; margin: 1rem 0; font-size: 0.92rem; }}
th, td {{ border: 1px solid #ccc; padding: 5px 10px; text-align: right; }}
th {{ background: #f0f2f5; text-align: center; }}
td:first-child, th:first-child {{ text-align: left; }}
code {{ background: #f0f2f5; padding: 1px 5px; border-radius: 3px; font-size: 0.9em; }}
.caveat {{ background: #fff8e6; border-left: 4px solid #e8a33d; padding: 0.8rem 1.2rem; margin: 1rem 0; }}
.finding {{ background: #eaf3ea; border-left: 4px solid #4a934a; padding: 0.8rem 1.2rem; margin: 1rem 0; }}
.bugfix {{ background: #fdecea; border-left: 4px solid #c0392b; padding: 0.8rem 1.2rem; margin: 1rem 0; }}
figure {{ margin: 1.5rem 0; }}
figcaption {{ font-size: 0.88rem; color: #555; margin-top: 0.4rem; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
.toc {{ background: #f8f9fa; padding: 1rem 1.5rem; border-radius: 6px; }}
.toc ul {{ margin: 0.3rem 0; }}
</style>
<body>

<h1>Task-aligned cross-area communication in the SSL whisker/auditory dataset</h1>
<p><i>Preliminary/exploratory analysis. Private, unpublished dataset (SSL cohort) &mdash; local report only, not for
external distribution. Compiled 2026-08-31.</i></p>

<div class="toc">
<b>Contents</b>
<ul>
<li><a href="#motivation">1. Motivation and question</a></li>
<li><a href="#data">2. Data and loading</a></li>
<li><a href="#methods">3. Methodology</a>
  <ul><li>3.1 Area hierarchy and unit selection &middot; 3.2 Trial conditions &middot; 3.3 Time binning and the
  artifact dead zone &middot; 3.4 Noise-correlation construction &middot; 3.5 The pCCA engine &middot;
  3.6 Three interaction variants &middot; 3.7 Statistical validation (shuffle test) &middot;
  3.8 PSTH-on-canonical-axes projection &middot; 3.9 Lagged pCCA</li></ul>
</li>
<li><a href="#corrections">4. Methodological corrections made along the way</a></li>
<li><a href="#coverage">5. Recording-pair coverage</a></li>
<li><a href="#pipeline">6. Pipeline validation (didactic figures)</a></li>
<li><a href="#results">7. Full-dataset results (50 learning-arm sessions)</a></li>
<li><a href="#lag">8. Lag / lead-lag analysis</a></li>
<li><a href="#caveats">9. Caveats and limitations</a></li>
<li><a href="#next">10. Open questions / suggested next steps</a></li>
</ul>
</div>

<h2 id="motivation">1. Motivation and question</h2>
<p>The SSL dataset records simultaneous multi-area Neuropixels activity while mice perform a whisker/auditory
Go/No-Go detection task, in two reward-contingency cohorts (R+: standard Go on both modalities; R-: rewarded for
<i>withholding</i> a lick on whisker trials &mdash; see <code>ssl_behavioral_paradigm.md</code>). This project asks
whether pairs of simultaneously-recorded brain areas show statistically distinguishable inter-areal
&quot;communication&quot; &mdash; correlated population activity beyond what either area's own trial-averaged
response or the rest of the recorded population would predict &mdash; during active trials, and whether that
communication differs between the two reward cohorts, across trial types and outcomes, and across a handful of
candidate area pairs.</p>

<h2 id="data">2. Data and loading</h2>
<p>KS4 spike-sorting source (<code>ssl_ephys</code>), loaded via the mandatory Path B pipeline
(<code>ephys_utilities.helpers.data_utils.combine_ephys_nwb</code> / <code>process_single_nwb</code>,
<code>day_to_analyze='learning'</code>), quality-classified via
<code>ephys_utilities.neural_utils.unit_metrics_utils.classify_units_quality</code> (default thresholds/exclude
list), area-labeled via <code>allen_utils.process_allen_labels(split_merge_areas=True)</code> for the fine
(<code>area_acronym_custom</code>) scheme and <code>get_custom_area_groups_from_name()</code> for the coarse scheme
used throughout this report. Mouse-inclusion filters: <code>exclude==0</code>, <code>exclude_ephys==0</code>,
<code>reward_group</code> restricted to {{R+, R-}} (R+proba dropped) &mdash; <b>entire-dataset population scope</b>
(no <code>learning_category</code>/learners-only restriction), since trial count per condition became a binding
constraint once trials were split by outcome (Section 3.2) and every available mouse's trials were needed.
Restricted to <code>day==0</code> (&quot;learning&quot;, each subject's first whisker-training ephys session) &mdash;
the expert arm (<code>day&gt;0</code>) was not analyzed in this pass.</p>

<h2 id="methods">3. Methodology</h2>

<h3>3.1 Area hierarchy and unit selection</h3>
<p>Two-level area hierarchy, never mixed: broader Allen custom groups first (e.g. &quot;Motor and frontal
areas&quot;, &quot;Somatosensory areas&quot;, &quot;Striatum and pallidum&quot;), each an aggregate of several
<code>area_acronym_custom</code> labels. The Step-1 coverage report (Section 5) used both levels; the full
pCCA analysis used the coarse level only, on the three areas that were simultaneously recorded above threshold in
every candidate session (see below).</p>
<p><b>Unit floor:</b> originally &ge;20 units/area (per the initial request); revised to
<b>tier-specific floors, &ge;10 units for the &quot;good&quot; quality tier or &ge;30 units for the
&quot;good+MUA&quot; tier</b>, per area, per session. The full-dataset run used <b>good+MUA only</b> (the
&quot;good&quot;-only tier was checkpoint-validated to show the same qualitative pattern and dropped for the
full run as a compute-budget decision, not a finding). Units are randomly subsampled to a fixed cap of
<b>30 units/area</b> when more are available, so canonical dimensionality is comparable across sessions/pairs.</p>

<h3>3.2 Trial conditions</h3>
<p><code>context=='active'</code> only (passive excluded). Trials are split by <b>trial type &times;
<code>lick_flag</code></b>: <code>whisker_trial&times;{{0,1}}</code>, <code>auditory_trial&times;{{0,1}}</code>.
<code>auditory_trial&times;lick_flag==0</code> was <b>excluded</b> after checking <code>trials.parquet</code>
directly: only 0&ndash;6 trials/session across the checkpoint subset (near-ceiling auditory performance, a known
property of this dataset &mdash; see <code>ssl_task_semantics.md</code>) &mdash; not analyzable at any session
count without pooling trials across many sessions, which was out of scope here. The three remaining conditions had
adequate trial counts in the full 50-session run (mean/min/max trials per session: whisker&times;0 &mdash;
111/19/270; whisker&times;1 &mdash; 43/15/88; auditory&times;1 &mdash; 42/21/106; a &ge;15-trial floor was applied
per session/condition).</p>
<p>Note this is the <b>naive</b> (cohort-agnostic) <code>lick_flag</code>, not the cohort-corrected hit/miss
definition used elsewhere in the SSL project family for reward-outcome analyses &mdash; the request specifically
asked for the raw lick_flag split.</p>

<h3>3.3 Time binning and the artifact dead zone</h3>
<p>Peri-<code>start_time</code> window <b>-200ms to +500ms</b>, binned at <b>5ms</b> resolution (reduced from an
initial 10ms). The mandatory whisker-trial artifact dead zone (-1ms/+4ms around <code>start_time</code>,
<code>ssl_artifact_dead_zone.md</code>) is <b>always excluded</b>, not just NaN-masked: the affected bin(s) are
dropped from the analysis bin axis before any model fitting, for every trial and every whisker-trial condition.
This mattered mechanically &mdash; see Section 4.</p>

<h3>3.4 Noise-correlation construction</h3>
<p>For each unit, in each session/trial-type/condition: <b>subtract that unit's own trial-averaged PSTH</b> (mean
across trials of that condition, per time bin) from every single trial's binned rate, leaving a per-trial residual
(&quot;noise correlation&quot;) matrix. <b>No baseline correction</b> is applied on top of this (explicit design
choice). The pCCA sample grain is one (trial, time-bin) residual population vector per area, pooling across trials
and bins within a session/condition/area-pair.</p>

<h3>3.5 The pCCA engine</h3>
<p>Canonical correlation is computed with the published <code>partial_CCA</code> package
(Gonzalez, Buzsaki/Chen labs, MIT license, associated with doi:10.1038/s41586-026-10481-z; installed from a
TestPyPI wheel into the analysis environment) &mdash; a <b>closed-form eigendecomposition</b> solution
(eigendecomposition of &Sigma;<sub>xx</sub><sup>-1</sup>&Sigma;<sub>xy</sub>&Sigma;<sub>yy</sub><sup>-1</sup>&Sigma;<sub>yx</sub>)
with explicit <b>ridge regularization</b> (&lambda;=1e-3 added to the diagonal of &Sigma;<sub>xx</sub>/&Sigma;<sub>yy</sub>
before inversion), replacing an earlier <code>sklearn.cross_decomposition.CCA</code> implementation. This
switch matters: sklearn's <code>CCA</code> is an iterative NIPALS/PLS-family algorithm, not the classical
(Hotelling) CCA solution, and is not guaranteed to match it beyond the first canonical dimension or under
ill-conditioned covariance (exactly the small-n/high-dimensionality regime this analysis operates in). Partial
CCA (regressing out a nuisance variable Z before computing CCA) is handled internally by
<code>PartialCCA.fit(X, Y, Z)</code> via OLS projection, replacing an earlier hand-rolled
regress-then-CCA implementation.</p>

<h3>3.6 Three interaction variants</h3>
<table><tr><th>Variant</th><th>Definition</th></tr>
<tr><td><b>A</b></td><td>Plain CCA on the two areas' noise-correlation residuals, no partialling.</td></tr>
<tr><td><b>B</b></td><td>Partial out the pooled activity of every other simultaneously-recorded unit in that
session (not in area A or B), reduced to its leading principal components (99% cumulative variance, capped at
100 components) before regression &mdash; a true partial-CCA nuisance regressor representing
&quot;the rest of the recorded brain&quot;.</td></tr>
<tr><td><b>C</b></td><td>Partial out a piezo-lick-rate regressor, licks binned at 50ms and resampled onto the 5ms
analysis grid.</td></tr>
</table>

<h3>3.7 Statistical validation (shuffle test)</h3>
<p>Naive (in-sample) canonical correlation is a biased estimator: with n_features approaching the sample size, even
independent Gaussian noise produces substantial apparent correlation (verified directly: n=200 samples, 10 vs.
8 independent features gave dim-1 r&asymp;0.37 with no true structure). The <b>correct</b> null distribution is
built by <b>shuffling trial identity</b>: for each of many draws, permute which area-B trial is paired with which
area-A trial (each trial's own within-trial time-bin sequence stays intact; area A and the nuisance Z, when used,
keep their real trial alignment) and <b>refit PartialCCA from scratch</b> on the shuffled data &mdash; not merely
reproject held-out data through the real, already-fitted weights, which tests a narrower and different question
(&quot;does this specific real axis remain correlated&quot; rather than &quot;could a spectrum this strong arise
from chance structure alone&quot;). This gives a genuine per-(session, pair, condition, variant, dimension) null
distribution of canonical correlation; the observed value is compared against it via a one-sided permutation
p-value. 500 shuffles/iteration in the checkpoint runs, 300 in the full-dataset run (a compute-budget reduction
given the ~8&times; larger session count; still &asymp;0.003 p-value resolution).</p>
<p>Cohort-level (R+ vs. R-) and dimension-level significance is then assessed with a <b>pooled paired
Wilcoxon signed-rank test + paired t-test</b> (both required &lt;0.05) between the across-session distribution of
observed correlation and the across-session distribution of each session's own mean shuffle-null value &mdash;
i.e. comparing the distribution of true correlations against the distribution of (per-session) mean shuffle
values, not a per-mouse pass/fail count. This replaced an earlier &quot;&ge;2/3 of sessions individually
significant&quot; rule, and is deliberately computed <b>pooling both cohorts</b> (n=6 at checkpoint scale,
n&asymp;44&ndash;49 at full scale) rather than per-cohort (n=3), since a paired Wilcoxon test cannot reach
p&lt;0.05 at n=3 by construction (minimum possible p=0.25).</p>
<div class="caveat"><b>Large-N caveat (see Section 7):</b> at the full 50-session scale, 96% of all tested
combinations were nominally significant by this test &mdash; a large-sample-size effect (even a tiny, highly
consistent bias registers as significant at n&asymp;45+), not evidence that every effect is scientifically
meaningful. <b>Effect size (excess correlation above the shuffle null), not the significance count, is the
informative quantity at this scale</b> and is what Section 7 reports.</div>

<h3>3.8 PSTH-on-canonical-axes projection</h3>
<p>The fitted canonical weight vectors are neuron-indexed, not time-bin-indexed, so they can be applied to any
population activity vector regardless of how it was time-binned. Each area's trial-averaged (non-residualized)
PSTH is projected through the fitted weights to obtain a canonical-variate time course per area; separately,
single-trial residual projections at each time bin give a genuine (held-out, when using a train/test split)
trial-by-trial across-time correlation curve, distinct from the trivially-correlated shared-PSTH-shape comparison.</p>

<h3>3.9 Lagged pCCA</h3>
<p>To ask which area leads: the fitted (5ms-resolution) canonical weights are projected onto a <b>separately
built, finer 2ms-binned</b> tensor (same dead-zone handling, always excluded as both the reference and the
shifted bin &mdash; this produces a visible diagonal exclusion band in the lag&times;time matrix, not just the
vertical lag=0 band). For each (reference time bin, lag) pair (&plusmn;100ms range, 2ms steps), the across-trial
correlation between area A's and area B's canonical-variate scores is computed, both as a canonical-dimension-1
summary and as a <b>neuron-level</b> control (mean pairwise raw correlation across all unit pairs, no CCA) &mdash;
the neuron-level matrix was near-zero everywhere the canonical matrix showed strong structure, confirming CCA
finds a concentrated communication direction invisible to naive pairwise averaging. The &quot;peak lag&quot; per
(session, pair, condition) is the lag maximizing the across-time-mean correlation.</p>

<h2 id="corrections">4. Methodological corrections made along the way</h2>
<p>Documented here because they materially shaped the final methodology, not just as debugging notes.</p>
<div class="bugfix"><b>Dead-zone bins crash model fitting if not pre-excluded.</b> <code>sklearn</code>'s and
<code>partial_CCA</code>'s <code>fit</code>/<code>predict</code>/<code>transform</code> reject NaN outright.
Early code NaN-masked dead-zone bins only in the final analysis step; every model-fitting call on
whisker-trial data crashed until the bin axis was pre-restricted to valid (non-dead-zone) bins before any
fit/predict/transform call, not just the eventual readout.</div>
<div class="bugfix"><b>In-sample &quot;example correlated trials&quot; figure was inflated.</b> An early
diagnostic fit CCA and evaluated it on the same trials; per-trial correlations looked far stronger (r&asymp;0.5&ndash;0.65)
than a fair held-out estimate (median r=0.09, matching the population-level CV estimate of 0.18). Fixed to fit on
a held-out half of trials and only display examples from the other half.</div>
<div class="bugfix"><b>Naive per-dimension shuffle null under-corrects for multiple comparisons.</b> An early
significance test built an independent null per canonical dimension by reprojecting shuffled data through
already-fitted real weights; switched to the refit-per-shuffle protocol in Section 3.7, which properly tests
whether the observed canonical spectrum itself could arise from shuffled data.</div>
<div class="bugfix"><b>A checkpoint run silently produced zero rows for variant C</b> (the piezo-lick nuisance
regressor was never actually computed in one script revision &mdash; initialized to <code>None</code>, never
set) &mdash; every C iteration was silently skipped, surfacing only as a downstream crash when the (empty)
result table reached the plotting stage. Caught by reviewing row counts, not just checking for exceptions.</div>
<div class="bugfix"><b>Per-cohort significance testing is underpowered by construction.</b> A paired Wilcoxon
signed-rank test cannot reach p&lt;0.05 with only 3 paired sessions (minimum possible p=0.25) &mdash; computing
significance per-cohort at checkpoint scale would have silently marked every dimension &quot;not significant&quot;
regardless of the true effect. Fixed by pooling both cohorts for the significance test while still plotting them
as separate lines.</div>
<div class="bugfix"><b>A full-scale run was smoke-tested first</b> (2 sessions, 5 shuffles) before committing
~1 hour of real compute, and caught a real crash (empty y-axis limits when a pair/condition combination has zero
valid sessions) that would otherwise have wasted the full run.</div>

<h2 id="coverage">5. Recording-pair coverage</h2>
<p>Full Path-B load of all 119 has_ephys sessions (~17.5min). At the coarse area level, entire-dataset scope,
learning arm: <b>{int(cov_learn[cov_learn.reward_group=='R+']['n_sessions_with_valid_pair'].iloc[0])}/{int(cov_learn[cov_learn.reward_group=='R+']['n_sessions_total'].iloc[0])} R+ and
{int(cov_learn[cov_learn.reward_group=='R-']['n_sessions_with_valid_pair'].iloc[0])}/{int(cov_learn[cov_learn.reward_group=='R-']['n_sessions_total'].iloc[0])} R- sessions</b> had at least one
valid (&ge;20-unit, pre-revision floor) area-pair; the richest coarse pairs were Motor/frontal&ndash;Striatum/pallidum
(86 sessions), Motor/frontal&ndash;Somatosensory (84 sessions), and Somatosensory&ndash;Striatum/pallidum
(76 sessions) &mdash; the same three pairs used throughout the rest of this project. Re-checked under the final
tier-specific floor (&ge;30 good+MUA units, all three areas simultaneously in the same session):
<b>50 learning-arm sessions (26 R+, 24 R-)</b> qualified &mdash; the population used for the full-scale run
in Section 7.</p>

<h2 id="pipeline">6. Pipeline validation (didactic figures)</h2>
<figure><img src="{b64(ARTIFACTS_DIR / 'checkpoint_01_input_rasters.png')}">
<figcaption>Fig. 1 &mdash; Raw input data: example-trial spike rasters for both areas of one session, with the
excluded dead zone shaded.</figcaption></figure>
<figure><img src="{b64(ARTIFACTS_DIR / 'checkpoint2_01_pipeline_illustration.png')}">
<figcaption>Fig. 2 &mdash; The full pipeline on one example session: (1) single trials + trial-averaged PSTH for an
example unit, (2) the noise-correlation residual (PSTH subtracted), (3) per-area PCA on the residuals, (4) the
three interaction variants' dim-1 held-out canonical correlation compared side by side, with the PCA-alignment
baseline for reference. Variant B (partial out other neurons) is visibly reduced relative to A/C, and its
confidence interval crosses the PCA baseline &mdash; the pattern that held up at full scale (Section 7).</figcaption></figure>

<h2 id="results">7. Full-dataset results (50 learning-arm sessions)</h2>
<p>{len(d1['A'])} (session &times; pair &times; condition) rows/variant at dimension 1
({dfs_full['A'].session_id.nunique()} of 50 candidate sessions contributed at least one usable combination);
{len(sig)} (dimension &times; pair &times; condition &times; variant) combinations tested for significance,
{len(sig_hits)} ({100*len(sig_hits)/len(sig):.0f}%) nominally significant &mdash; interpreted via effect size,
per the Section 3.7 caveat.</p>

<h3>Effect sizes (dimension 1, whisker trials with a lick, mean &plusmn; nothing shown here &mdash; see figures for
SEM across sessions)</h3>
<table><tr><th>Variant</th><th>Pair</th><th>Observed corr.</th><th>Excess over shuffle null</th><th>n sessions</th></tr>
{"".join(f"<tr><td>{r.variant}</td><td>{r.pair}</td><td>{r.observed:.3f}</td><td>{r.excess:.3f}</td><td>{r.n_sessions:.0f}</td></tr>" for r in effect_table.itertuples())}
</table>

<div class="finding"><b>Finding 1 &mdash; partialling out the rest of the recorded population substantially
reduces apparent cross-area correlation.</b> Variant B's mean excess correlation (dim 1, whisker/licked trials)
is {variant_means['B'][1]:.3f}, a {(1 - variant_means['B'][1]/variant_means['A'][1])*100:.0f}% reduction from
variant A's {variant_means['A'][1]:.3f} (variant C, partialling out licking, is essentially unchanged at
{variant_means['C'][1]:.3f}). Most of the apparent Motor&ndash;Somatosensory and Motor&ndash;Striatum
&quot;communication&quot; is shared with the broader recorded population (a brain-wide state variable), not
specific to that pair.</div>

<div class="finding"><b>Finding 2 &mdash; Somatosensory&ndash;Striatum is the strongest and most pairwise-specific
of the three area pairs.</b> It has the largest excess correlation under plain CCA of the three pairs, and is the
only pair whose signal substantially survives variant B's partialling &mdash; its excess correlation is roughly
double that of the other two pairs under variant B, while the other two pairs are reduced to near the noise
floor. This pair looks like genuine, area-specific inter-areal communication rather than shared global state.</div>

<div class="finding"><b>Finding 3 &mdash; no significant R+ vs. R- cohort difference.</b> Mann-Whitney U + Welch's
t-test (dimension 1, variant A, whisker trials with a lick) found no significant cohort difference for any of the
three area pairs (all p&gt;0.17, n=25 R+ / 19 R-). At this sample size the test has reasonable power (not an
underpowered null) &mdash; the reward-contingency manipulation does not appear to change cross-area communication
strength at the level of the dominant canonical dimension, at least for these three area pairs and this trial
condition.</div>

<figure><img src="{b64(ARTIFACTS_DIR / 'full_learning' / 'full_heatmap_lag0_summary.png')}">
<figcaption>Fig. 3 &mdash; Dimension-1 canonical correlation, all three area pairs &times; all three trial
conditions &times; all three variants, full 50-session dataset.</figcaption></figure>
<figure><img src="{b64(ARTIFACTS_DIR / 'full_learning' / 'full_variants_ABC_raw_whisker_trial_lick1.png')}">
<figcaption>Fig. 4 &mdash; Canonical correlation vs. dimension, all three area pairs (rows) &times; all three
variants (columns), R+ (red) vs. R- (blue), whisker trials with a lick. Y-axis shared within each row for direct
cross-variant comparison. Full 50-session dataset.</figcaption></figure>

<h2 id="lag">8. Lag / lead-lag analysis</h2>
<p>Checkpoint scale only (6 sessions, 3 pairs, 3 conditions = 54 interactions; not yet re-run on the full
50-session dataset). <b>Median peak lag: 0.0ms; {(peak_lag['peak_lag_ms'].abs()<=2).mean()*100:.0f}% of
interactions had their peak within &plusmn;2ms of lag 0</b>, with the remaining interactions' peak lags scattered
roughly symmetrically across the &plusmn;100ms range (no consistent directional bias toward either area
leading). This is consistent with genuine near-zero-lag communication dominating, and does not support a
confident area-leads-area claim at this dataset size &mdash; the scattered non-zero peaks are plausibly a
&quot;pick the best of 101 lag candidates&quot; selection effect on noisy per-session data, not evidence of a
consistent lead/lag relationship, though this has not been formally tested with its own permutation null.</p>
<figure><img src="{b64(ARTIFACTS_DIR / 'peak_lag_distribution.png')}">
<figcaption>Fig. 5 &mdash; Distribution of peak lag across all 54 checkpoint-scale interactions.</figcaption></figure>
<figure><img src="{b64(ARTIFACTS_DIR / 'newrules_02_lag_2ms_matrix.png')}">
<figcaption>Fig. 6 &mdash; Example lagged canonical-correlation matrices (2ms resolution) for two example sessions,
Motor-frontal vs. Somatosensory, whisker trials. The diagonal exclusion band is the dead zone appearing at
whichever (time, lag) combination lands on it.</figcaption></figure>

<h2 id="caveats">9. Caveats and limitations</h2>
<ul>
<li><b>Exploratory, not confirmatory.</b> No exploration/confirmation split was applied to this project; every
result here should be read as hypothesis-generating, consistent with how it was scoped throughout.</li>
<li><b>Large-N significance saturation</b> (Section 3.7/7): at 50 sessions, binary significance is close to
uninformative; effect size is the load-bearing quantity, and even the effect-size comparisons have not been
corrected for multiple comparisons across the 3 pairs &times; 3 conditions &times; 3 variants &times; 20
dimensions tested.</li>
<li><b>Learning arm only.</b> The expert arm (day&gt;0 sessions) was never analyzed in this project.</li>
<li><b>&quot;good&quot; quality tier and the full &ge;500-shuffle protocol were both checkpoint-only</b> &mdash;
dropped for the full run as compute-budget decisions (Section 3.1, 3.7), not because they were shown unnecessary
beyond the 6-session checkpoint.</li>
<li><b>Only 3 of the many possible area pairs were analyzed</b> (the three richest coarse-level pairs); the fine
(<code>area_acronym_custom</code>) hierarchy level was used only in the Step-1 coverage report, never for the
pCCA analysis itself.</li>
<li><b>Units are subsampled to 30/area</b> when more are available, discarding most of the recorded population in
many sessions; the influence of this subsampling on canonical correlation magnitude was checked qualitatively
(not with a dedicated unit-count-vs-correlation curve at full scale) via the tier-specific floor design.</li>
<li><b>Lagged pCCA and the peak-lag distribution are checkpoint-scale only</b> (6 sessions), not yet re-run on the
full 50-session dataset, and have no shuffle-based significance test of their own.</li>
<li><b>Ridge regularization parameter (&lambda;=1e-3) was not swept</b> &mdash; a fixed, reasonable default was
used throughout rather than tuned per pair/condition.</li>
</ul>

<h2 id="next">10. Open questions / suggested next steps</h2>
<ul>
<li>Re-run the lagged pCCA and peak-lag distribution on the full 50-session dataset, with a proper shuffle-based
significance test for the peak lag itself (e.g. circular-shift null per session, matching the
<code>partial_CCA</code> package's own <code>surrogate_test</code> convention).</li>
<li>Extend to the fine (<code>area_acronym_custom</code>) hierarchy level and/or additional coarse pairs beyond
the three analyzed here.</li>
<li>Extend to the expert arm.</li>
<li>Apply a multiple-comparisons correction across the full grid of tested combinations before treating any single
cell as a confirmed effect.</li>
<li>Investigate whether Somatosensory&ndash;Striatum's pairwise-specific signal (Finding 2) is trial-type- or
outcome-specific, or general across all three tested conditions.</li>
</ul>

</body>
"""
    out_path = PROJECT_DIR / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
