"""Consolidated report for the post-first-hit re-analysis: statistics
battery (no BH-FDR), movement correlation, and temporal dynamics. See
question.md / change-log.md for the full decision trail.
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def b64_img(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ARTIFACTS_DIR / name).read_bytes()).decode("ascii")


def fmt(x):
    return f"{x:.4g}" if isinstance(x, float) else x


def main() -> None:
    stats = pd.read_csv(ARTIFACTS_DIR / "statistics_summary_posthit.csv")
    motion_corr = pd.read_csv(ARTIFACTS_DIR / "motion_control_posthit_correlations.csv")
    temporal = pd.read_csv(ARTIFACTS_DIR / "temporal_permanova_summary_posthit.csv")

    stats_cols = ["label", "n_units", "n_mice", "mannwhitney_p", "welch_p", "prop_mannwhitney_p", "prop_welch_p",
                  "ks_p_mouseblock", "permanova_p", "n_areas_tested", "n_areas_nominal_p_lt_05"]
    stats_cols = [c for c in stats_cols if c in stats.columns]
    stats_html = stats[stats_cols].to_html(index=False, float_format=fmt)
    motion_html = motion_corr.to_html(index=False, float_format=fmt)
    temporal_html = temporal.to_html(index=False, float_format=fmt)

    html = f"""<title>Post-First-Hit Re-Analysis Report</title>
<meta charset="utf-8">
<body style="font-family:sans-serif;max-width:1150px;margin:2rem auto;line-height:1.55">

<h1>Burstiness x cohort -- post-first-hit re-analysis</h1>
<p>Everything repeated with all data restricted to each session's period
at/after its first cohort-corrected whisker-trial hit (R+ hit = licked;
R- hit = withheld, per <code>ssl_task_semantics.md</code>), with <b>no
multiple-testing correction applied</b> to the per-area post-hoc (raw
p-values reported) per explicit request. The "only run per-area post-hoc
if the main PERMANOVA is significant" gate is kept regardless -- a
compute-cost/hierarchical-order choice, not a correction, stated
explicitly. Population: 127,020/129,127 learning units (1/89 sessions
excluded, no qualifying hit) + all 44,678 expert units (0/29 excluded).
Small-subset validated before the full run (4-20% of session excluded per
session, no dropped sessions there).</p>

<h2>Bugs found and fixed while building this thread</h2>
<ul>
<li>The compressed <code>ssl_ephys</code> dataset's <code>context</code> trial column is the
literal string "nan" (not a real null) for every subject below AB116 (never had
passive epochs recorded) -- silently zeroed out a naive <code>context=='active'</code>
filter, producing 12/88 spurious "no qualifying hit" sessions via that source vs.
1/89 via the mandatory Path B loader. Fixed by replicating Path B's own
nan-to-active normalization before filtering; reconciled cleanly afterward.</li>
<li><code>pandas.Series.map()</code> with <code>None</code>-valued dict entries produces an
object-dtype column that crashes newer scipy's <code>spearmanr</code> -- a parquet
round-trip silently "fixed" it on reload, which is why replaying saved data
didn't reproduce the crash. Fixed with explicit <code>pd.to_numeric(..., errors="coerce")</code>.</li>
</ul>

<h2>Statistics battery (raw p-values, no correction)</h2>
{stats_html}
<p><b>Both previously-found effects replicate cleanly</b>, same direction/pattern as the
unrestricted analysis: expert-arm <code>burst_index_whisker</code> significant on all four
test types in all four scope/tier combinations (<code>MO-wM1</code> nominally significant in
both scope variants of the post-hoc -- the most area-consistent lead in the project);
learning-arm <code>burst_index_auditory</code> significant on distribution/PERMANOVA but not
location/proportion in 3-4/4 combos (a shape-not-shift effect, as before).
<code>continuous_burstiness</code> remains null everywhere. <b>Caveat:</b> with ~22-45 areas
tested per combination and no correction, 1-2 nominal hits are expected by chance
alone under a true null -- these are leads, not confirmed localizations.</p>

<h2>Does bursting correlate with movement? (direct answer: no)</h2>
{motion_html}
<p>All 6 testable combinations null (p=0.09-0.99), matching the original
(pre-restriction) result. Expert-arm orofacial motion energy has zero usable
sessions (DLC tracking coverage gap, unchanged from before).</p>

<h2>Temporal dynamics (post-hit progression)</h2>
<img src="{b64_img('temporal_cohort_trajectories_posthit.png')}" style="max-width:100%">
{temporal_html}
<p>Mostly null, as before, with one new nominal hit
(<code>expert_learners_only_good_absolute_0-45min</code>, p=0.038) -- within chance
expectation across 16 uncorrected tests, not a confirmed effect. The learning-arm
step-like rise in R+ burstiness (visible in all 4 learning panels) is sharper here
than in the unrestricted version, consistent with removing the noisier
pre-engagement warm-up period, but was already shown NOT to survive a targeted
per-mouse test in the unrestricted analysis -- not re-derived here, flagged as an
open item if this restricted version should be re-checked the same way.</p>

<h2>Where this leaves the project</h2>
<p>The post-first-hit restriction does not change the overall picture: expert-arm
<code>burst_index_whisker</code> remains the one fully robust, multi-test-convergent
finding (now with a specific, replicated area lead: <code>MO-wM1</code>). Learning-arm
<code>burst_index_auditory</code> remains a distribution-only effect. Movement is not a
confound for burstiness. Temporal trajectory shape remains statistically null
despite a persistently visible learning-arm pattern that has not held up under
closer inspection.</p>

</body>
"""
    out_path = ARTIFACTS_DIR / "posthit_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
