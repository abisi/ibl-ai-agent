"""Intermediate report for the temporal-dynamics analysis thread (full
population): methodology, unit-level validation, cohort trajectories, the
omnibus PERMANOVA test, and the exploratory follow-up test -- consolidated
into one document per the established checkpoint-report pattern
(intermediate_report.html for burst detection).
"""
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def b64_img(name: str) -> str:
    data = (ARTIFACTS_DIR / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def main() -> None:
    permanova_df = pd.read_csv(ARTIFACTS_DIR / "temporal_permanova_summary.csv")
    late_early_df = pd.read_csv(ARTIFACTS_DIR / "temporal_late_minus_early_test.csv")

    def fmt(x):
        return f"{x:.4g}" if isinstance(x, float) else x

    permanova_html = permanova_df.to_html(index=False, float_format=fmt)
    late_early_html = late_early_df.to_html(index=False, float_format=fmt)

    html = f"""<title>Temporal Dynamics Intermediate Report</title>
<meta charset="utf-8">
<body style="font-family:sans-serif;max-width:1150px;margin:2rem auto;line-height:1.55">

<h1>Burstiness across session progression -- intermediate report</h1>
<p>Full-population run (173,805 units total across both arms), following the
small-subset preview and unit-level method validation. See
<code>question.md</code> / <code>change-log.md</code> for the full decision
trail. <b>This is exploratory, not confirmatory</b> -- the whole project
skipped the exploration/confirmation split by explicit agreement.</p>

<h2>Method</h2>
<ul>
<li>Per unit: continuous-burstiness spike train (whisker dead-zone excised, ISI-run burst labeling -- same validated pipeline as the rest of the project).</li>
<li><b>Normalized session progression</b>: 20 equal-width bins spanning [session start, session end] (task span: min trial start_time to max trial stop_time) -- bin <i>i</i> always represents the same fractional progress through the session regardless of its absolute duration.</li>
<li><b>Absolute time</b>: 20 fixed 3-minute bins from session start (covers 0-60min); sessions shorter than a given bin contribute no data to it -- coverage is 79-93% of mice through bin 14 (45min), dropping to 52-70% by bin 19 (60min).</li>
<li>Aggregation: unit -&gt; session (mean per bin) -&gt; mouse (mean across that mouse's sessions per bin) -- one row per mouse, matching this project's mouse-level statistical-unit convention.</li>
</ul>

<h2>Unit-level validation (small subset, before the full run)</h2>
<p>9 example units (low/median/high continuous burstiness x 3 sessions) confirmed the binning method produces sensible per-bin values matching the underlying raster directly -- including reproducing a genuine late-session decline in one specific unit (MH021), not an artifact.</p>
<img src="{b64_img('temporal_binning_unit_examples.png')}" style="max-width:100%">

<h2>Cohort trajectories (full population, mean &plusmn; SEM across mice)</h2>
<p>16 panels: 2 arms &times; 2 scopes (all mice / learners only) &times; 2 tiers (good / good+MUA), each shown both normalized and absolute.</p>
<img src="{b64_img('temporal_cohort_trajectories_full.png')}" style="max-width:100%">

<h2>Omnibus test: does the joint 20-bin trajectory shape differ by cohort?</h2>
<p>Mouse-block-permutation PERMANOVA (reusing this project's already-fixed <code>permanova.py</code>), treating the 20 bins as joint multivariate features -- the same logic already used for burst_index/continuous_burstiness elsewhere in this project, just higher-dimensional. This is the primary, pre-registered-style test.</p>
{permanova_html}
<p><b>Result: null everywhere</b> (p=0.09-0.92 across all 16 tests). No cohort difference in overall trajectory shape, either arm, scope, tier, or time representation.</p>

<h2>Exploratory follow-up: late-half minus early-half delta</h2>
<p>Motivated by inspecting the trajectory plot above (NOT pre-registered) -- the learning-arm panels showed a visually striking step-like rise in R+ burstiness starting ~60% through the session. Tested directly: per mouse,
<code>mean(bins 12-19, the last 40% of the session) - mean(bins 0-11, the first 60%)</code>, R+ vs R- (Mann-Whitney + Welch, the standard pair for this project).</p>
{late_early_html}
<p><b>Result: the learning-arm visual pattern did NOT hold up</b> (p&gt;0.2 in all 4 learning combinations, median deltas tiny relative to the burstiness scale) -- most likely driven by a subset of mice/units rather than a genuine cohort-wide shift, not something to build on.</p>
<p><b>An unanticipated signal turned up instead, in the expert arm</b>: good+MUA tier, both scopes -- both Mann-Whitney and Welch agree
(all-mice: p=0.041/0.018; learners-only: p=0.016/0.007). R+ burstiness stays flat or rises slightly across the session while R- declines.
<b>Caveats, stated plainly</b>: this is exploratory (found by looking at the plot, not planned in advance), small n (8-19 mice per test), and not corrected for the 8 post-hoc tests run here. It is a lead for a properly pre-registered follow-up, not a result to report as confirmed.</p>

<h2>Where this leaves the project</h2>
<p>Across everything run in this project so far, the one fully robust, multi-test-convergent finding remains <b>expert-arm <code>burst_index_whisker</code></b>
(significant on global location, PERMANOVA, proportion, and mouse-block-KS distribution tests, all four scope/tier combinations). Continuous burstiness and the session-trajectory shape are null on every formal test run. The expert-arm late-session decline in R- (this section) is a new, unconfirmed lead.</p>

</body>
"""
    out_path = ARTIFACTS_DIR / "temporal_dynamics_report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
