"""Render the ssl_ephys_utilities_qc_report.py outputs (CSVs, PNGs, run_metadata.json)
into a single self-contained HTML report (images embedded as base64 data URIs).
"""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

REPORT_DIR = Path(r"C:\Users\bisi\Github\int-brain-lab\ibl-ai-agent\reports\ssl_analysis\ephys_utilities_qc_report")
OUT_HTML = REPORT_DIR / "report.html"


def b64_img(name: str) -> str:
    data = (REPORT_DIR / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def fmt_int(n) -> str:
    return f"{int(n):,}"


METRIC_LABELS = {"n_mice": "Mice", "n_insertions": "Insertions", "n_good": "Good", "n_good_mua": "Good+MUA"}
REWARD_SUFFIX_LABELS = {"Rplus": "R+", "Rminus": "R-"}


def col_label(col: str) -> str:
    if col == "area_acronym_custom":
        return "Area"
    for suffix, rg_label in REWARD_SUFFIX_LABELS.items():
        if col.endswith(f"_{suffix}"):
            metric = col[: -len(f"_{suffix}")]
            return f"{METRIC_LABELS.get(metric, metric)} {rg_label}"
    return col


def table_html(csv_name: str, table_id: str) -> str:
    df = pd.read_csv(REPORT_DIR / csv_name)
    cols = list(df.columns)
    thead = "".join(
        f'<th data-col="{i}" data-type="{"text" if c == "area_acronym_custom" else "num"}">'
        f'{html.escape(col_label(c))}<span class="sort-ind"></span></th>'
        for i, c in enumerate(cols)
    )
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            if c == "area_acronym_custom":
                cells.append(f'<td class="area-cell">{html.escape(str(r[c]))}</td>')
            else:
                cells.append(f'<td class="num-cell">{fmt_int(r[c])}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    tbody = "".join(rows)
    return (
        f'<div class="table-wrap"><table class="data-table" id="{table_id}">'
        f"<thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table></div>"
    )


def stat_chip(label: str, value: str) -> str:
    return f'<span class="chip"><span class="chip-val">{html.escape(value)}</span><span class="chip-label">{html.escape(label)}</span></span>'


def figure_block(img_name: str, caption: str) -> str:
    src = b64_img(img_name)
    return (
        f'<figure class="fig-card"><img src="{src}" alt="{html.escape(caption)}" loading="lazy">'
        f'<figcaption>{caption}</figcaption></figure>'
    )


def main() -> None:
    meta = json.loads((REPORT_DIR / "run_metadata.json").read_text())
    learn = meta["learning"]
    exp = meta["expert"]

    excluded_mice = ", ".join(learn["mice_excluded_from_subset"])
    learn_rg = learn["n_mice_per_reward_group"]
    exp_rg = exp["n_mice_per_reward_group"]
    exp_inconsistent = exp.get("mice_with_inconsistent_reward_group", [])

    learning_all_table = table_html("learning_all_mice_area_summary.csv", "tbl-learning-all")
    learning_sub_table = table_html("learning_moderate_good_mice_area_summary.csv", "tbl-learning-sub")
    expert_all_table = table_html("expert_all_mice_area_summary.csv", "tbl-expert-all")

    html_doc = f"""<!doctype html>
<title>Ephys QC by Area</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,600&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #F4F6F8;
  --surface: #FFFFFF;
  --surface-2: #EAEEF2;
  --ink: #161B22;
  --ink-soft: #3C4550;
  --muted: #66717E;
  --border: #DAE1E7;
  --accent: #2F5D8A;
  --accent-soft: #E4EDF6;
  --warn: #A8402F;
  --warn-soft: #F5E6E2;
  --shadow: 0 1px 2px rgba(22,27,34,0.04), 0 6px 20px rgba(22,27,34,0.06);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #10151C;
    --surface: #1A222C;
    --surface-2: #212B37;
    --ink: #E8ECF1;
    --ink-soft: #C3CBD5;
    --muted: #8996A5;
    --border: #2B3642;
    --accent: #86AEDD;
    --accent-soft: #1E3140;
    --warn: #E2836E;
    --warn-soft: #3A231E;
    --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
  }}
}}
:root[data-theme="dark"] {{
  --bg: #10151C;
  --surface: #1A222C;
  --surface-2: #212B37;
  --ink: #E8ECF1;
  --ink-soft: #C3CBD5;
  --muted: #8996A5;
  --border: #2B3642;
  --accent: #86AEDD;
  --accent-soft: #1E3140;
  --warn: #E2836E;
  --warn-soft: #3A231E;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 24px rgba(0,0,0,0.35);
}}

* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.55;
}}
::selection {{ background: var(--accent-soft); }}

h1, h2, h3 {{
  font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
  font-weight: 600;
  color: var(--ink);
  text-wrap: balance;
  line-height: 1.2;
}}

a {{ color: var(--accent); }}
a:focus-visible, button:focus-visible, th:focus-visible {{
  outline: 2px solid var(--accent);
  outline-offset: 2px;
}}

code, .mono {{ font-family: "IBM Plex Mono", "Cascadia Code", ui-monospace, monospace; }}

.topbar {{
  position: sticky;
  top: 0;
  z-index: 20;
  display: flex;
  align-items: center;
  gap: 1.25rem;
  padding: 0.6rem 1.5rem;
  background: color-mix(in srgb, var(--surface) 88%, transparent);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--border);
  font-size: 0.85rem;
}}
.topbar .brand {{
  font-family: "Source Serif 4", serif;
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--ink);
  margin-right: auto;
  white-space: nowrap;
}}
.topbar nav {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
.topbar nav a {{
  color: var(--muted);
  text-decoration: none;
  padding: 0.3rem 0;
  border-bottom: 2px solid transparent;
}}
.topbar nav a:hover {{ color: var(--ink); border-bottom-color: var(--accent); }}

main {{
  max-width: 900px;
  margin: 0 auto;
  padding: 2.5rem 1.5rem 5rem;
}}

.report-header {{ margin-bottom: 2.25rem; }}
.eyebrow {{
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 0.6rem;
}}
h1.title {{ font-size: 2.1rem; margin: 0 0 0.5rem; }}
.subtitle {{ color: var(--ink-soft); font-size: 1.05rem; max-width: 62ch; margin: 0; }}

.trace-panel {{
  margin-top: 1.75rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.1rem 1.3rem;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.6rem 1.5rem;
  font-size: 0.85rem;
}}
.trace-item dt {{
  color: var(--muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 0.15rem;
}}
.trace-item dd {{
  margin: 0;
  font-family: "IBM Plex Mono", monospace;
  color: var(--ink-soft);
  word-break: break-word;
  font-size: 0.82rem;
}}

section.arm {{
  margin-top: 3.5rem;
  padding-top: 0.5rem;
  scroll-margin-top: 4rem;
}}
section.arm h2 {{ font-size: 1.5rem; margin-bottom: 0.35rem; }}
.arm-desc {{ color: var(--ink-soft); max-width: 68ch; margin: 0 0 1rem; }}

.chip-row {{ display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 1rem 0 1.5rem; }}
.chip {{
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
  background: var(--accent-soft);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 0.35rem 0.9rem;
}}
.chip-val {{
  font-family: "IBM Plex Mono", monospace;
  font-weight: 600;
  color: var(--accent);
  font-variant-numeric: tabular-nums;
}}
.chip-label {{ font-size: 0.78rem; color: var(--muted); }}

.fig-grid {{
  display: grid;
  grid-template-columns: 1fr;
  gap: 1.25rem;
  margin-bottom: 1.75rem;
}}
.fig-card {{
  margin: 0;
  background: #FFFFFF;
  border: 1px solid var(--border);
  border-radius: 10px;
  box-shadow: var(--shadow);
  padding: 0.75rem;
  overflow: hidden;
}}
.fig-card img {{ display: block; width: 100%; height: auto; border-radius: 4px; }}
.fig-card figcaption {{
  font-size: 0.82rem;
  color: #555;
  padding: 0.6rem 0.35rem 0.15rem;
  font-family: "Public Sans", sans-serif;
}}

h3.table-heading {{
  font-size: 1.05rem;
  margin: 1.75rem 0 0.6rem;
  display: flex;
  align-items: baseline;
  gap: 0.6rem;
}}
h3.table-heading .hint {{
  font-family: "Public Sans", sans-serif;
  font-weight: 400;
  font-size: 0.78rem;
  color: var(--muted);
}}

.table-wrap {{
  overflow: auto;
  max-height: 420px;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
}}
table.data-table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 0.88rem;
}}
table.data-table thead th {{
  position: sticky;
  top: 0;
  background: var(--surface-2);
  text-align: right;
  padding: 0.55rem 0.9rem;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  color: var(--muted);
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
table.data-table thead th:first-child {{ text-align: left; }}
table.data-table thead th:hover {{ color: var(--ink); }}
.sort-ind {{ display: inline-block; width: 0.9em; opacity: 0.6; }}
table.data-table tbody td {{
  padding: 0.42rem 0.9rem;
  border-bottom: 1px solid var(--border);
}}
table.data-table tbody tr:last-child td {{ border-bottom: none; }}
table.data-table tbody tr:hover {{ background: var(--accent-soft); }}
.area-cell {{ font-family: "IBM Plex Mono", monospace; font-weight: 500; }}
.num-cell {{ text-align: right; font-family: "IBM Plex Mono", monospace; font-variant-numeric: tabular-nums; color: var(--ink-soft); }}

.callout {{
  border: 1px solid var(--border);
  border-left: 4px solid var(--warn);
  background: var(--warn-soft);
  border-radius: 6px;
  padding: 0.9rem 1.1rem;
  margin: 1.5rem 0;
  font-size: 0.92rem;
}}
.callout strong {{ color: var(--warn); }}

.legend {{
  display: flex;
  gap: 1.25rem;
  font-size: 0.82rem;
  color: var(--muted);
  margin: -0.5rem 0 1.25rem;
}}
.legend span {{ display: inline-flex; align-items: center; gap: 0.4rem; }}
.swatch {{ width: 0.75rem; height: 0.75rem; border-radius: 2px; display: inline-block; }}
.swatch.good {{ background: #4C72B0; }}
.swatch.goodmua {{ background: #C44E52; }}

footer {{
  margin-top: 4rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.82rem;
}}
footer code {{ font-size: 0.8rem; }}
footer ul {{ padding-left: 1.2rem; }}

@media (max-width: 640px) {{
  .topbar {{ overflow-x: auto; }}
  h1.title {{ font-size: 1.6rem; }}
}}
</style>

<div class="topbar">
  <span class="brand">SSL Ephys QC</span>
  <nav>
    <a href="#trace">Method</a>
    <a href="#learning-all">Learning</a>
    <a href="#learning-sub">Learning (moderate+good)</a>
    <a href="#expert">Expert</a>
    <a href="#caveats">Caveats</a>
  </nav>
</div>

<main>
  <header class="report-header">
    <p class="eyebrow">SSL &middot; KS4 &middot; Path B pipeline</p>
    <h1 class="title">Ephys QC by brain area</h1>
    <p class="subtitle">Mice, insertions, and unit-quality counts per <code>area_acronym_custom</code>
      (<code>split_merge_areas=True</code>), split by reward group (R+/R-), built from raw NWB via
      <code>process_single_nwb</code> &rarr; <code>classify_units_quality</code> &rarr;
      <code>process_allen_labels</code> &mdash; for cross-checking against an independent count.</p>

    <dl class="trace-panel" id="trace">
      <div class="trace-item"><dt>NWB source</dt><dd>M:\\analysis\\Axel_Bisi\\NWB_ks4 (KS4)</dd></div>
      <div class="trace-item"><dt>ephys_utilities</dt><dd>M:\\analysis\\Axel_Bisi\\Github\\ephys_utilities</dd></div>
      <div class="trace-item"><dt>Loader</dt><dd>data_utils.combine_ephys_nwb &rarr; process_single_nwb</dd></div>
      <div class="trace-item"><dt>Quality fn</dt><dd>unit_metrics_utils.classify_units_quality</dd></div>
      <div class="trace-item"><dt>thresholds</dt><dd>DEFAULT_METRIC_THRESHOLDS</dd></div>
      <div class="trace-item"><dt>exclude</dt><dd>Lratio, isolationDistance, presenceRatio, maxDriftEstimate</dd></div>
      <div class="trace-item"><dt>label_col</dt><dd>quality_label</dd></div>
      <div class="trace-item"><dt>joint drift check</dt><dd>not evaluated (no DREDge merge)</dd></div>
      <div class="trace-item"><dt>non-soma</dt><dd>excluded from every count below</dd></div>
      <div class="trace-item"><dt>insertion key</dt><dd>session_id + electrode_group</dd></div>
      <div class="trace-item"><dt>area labeling</dt><dd>allen_utils.process_allen_labels(split_merge_areas=True)</dd></div>
      <div class="trace-item"><dt>reward_group</dt><dd>native unit_table['reward_group'] (wh_reward, int 0/1)</dd></div>
      <div class="trace-item"><dt>learner subset</dt><dd>joint_mouse_reference_weight.xlsx &rarr; learning_category</dd></div>
    </dl>
  </header>

  <section class="arm" id="learning-all">
    <h2>Learning &mdash; all mice</h2>
    <p class="arm-desc">Each subject's first whisker-training ephys session (<code>day_to_analyze='learning'</code>).
      {learn["n_nwb_files_scanned"]:,} NWB files scanned &rarr; {learn["n_nwb_files_with_ephys_after_day_filter"]} kept.</p>
    <div class="chip-row">
      {stat_chip("mice", str(learn["n_mice_all"]))}
      {stat_chip("mice R+", str(learn_rg.get("R+", 0)))}
      {stat_chip("mice R-", str(learn_rg.get("R-", 0)))}
      {stat_chip("sessions", str(learn["n_sessions_all"]))}
      {stat_chip("areas", str(learn["all_mice"]["n_areas"]))}
      {stat_chip("good units", fmt_int(learn["quality_label_counts"]["good"]))}
      {stat_chip("mua units", fmt_int(learn["quality_label_counts"]["mua"]))}
      {stat_chip("non-soma (excluded)", fmt_int(learn["quality_label_counts"]["non-soma"]))}
    </div>
    <div class="fig-grid">
      {figure_block("learning_all_mice_panels.png", "Mice / insertions / good / good+MUA per area, split R+ vs R-, sorted by combined good+MUA.")}
      {figure_block("learning_all_mice_good_vs_goodmua.png", "Good (front) inside Good+MUA (back), per area — R+ (left) vs R- (right).")}
    </div>
    <div class="legend"><span><span class="swatch good"></span>Good</span><span><span class="swatch goodmua"></span>Good+MUA (includes MUA)</span></div>
    <h3 class="table-heading">Full area table <span class="hint">click a column to sort</span></h3>
    {learning_all_table}
  </section>

  <section class="arm" id="learning-sub">
    <h2>Learning &mdash; moderate + good learners only</h2>
    <p class="arm-desc">Same session set, restricted to mice whose <code>learning_category</code> in the reference
      sheet is <code>moderate</code> or <code>good</code>.</p>
    <div class="callout">
      <strong>{len(learn["mice_excluded_from_subset"])} of {learn["n_mice_all"]} learning mice excluded</strong>
      (learning_category is <code>bad</code>, missing, or unmatched):
      <span class="mono">{html.escape(excluded_mice)}</span>
    </div>
    <div class="chip-row">
      {stat_chip("mice", str(learn["n_mice_moderate_good_subset"]))}
      {stat_chip("areas", str(learn["moderate_good_mice"]["n_areas"]))}
    </div>
    <div class="fig-grid">
      {figure_block("learning_moderate_good_mice_panels.png", "Mice / insertions / good / good+MUA per area, moderate+good learners only, split R+ vs R-.")}
      {figure_block("learning_moderate_good_mice_good_vs_goodmua.png", "Good vs Good+MUA per area, moderate+good learners only — R+ (left) vs R- (right).")}
    </div>
    <h3 class="table-heading">Full area table <span class="hint">click a column to sort</span></h3>
    {learning_sub_table}
  </section>

  <section class="arm" id="expert">
    <h2>Expert &mdash; all mice</h2>
    <p class="arm-desc">Later sessions per subject (<code>day_to_analyze='expert'</code>), unevenly distributed &mdash;
      some mice contribute several sessions. {exp["n_nwb_files_scanned"]:,} NWB files scanned &rarr;
      {exp["n_nwb_files_with_ephys_after_day_filter"]} kept.</p>
    <div class="chip-row">
      {stat_chip("mice", str(exp["n_mice_all"]))}
      {stat_chip("mice R+", str(exp_rg.get("R+", 0)))}
      {stat_chip("mice R-", str(exp_rg.get("R-", 0)))}
      {stat_chip("sessions", str(exp["n_sessions_all"]))}
      {stat_chip("areas", str(exp["all_mice"]["n_areas"]))}
      {stat_chip("good units", fmt_int(exp["quality_label_counts"]["good"]))}
      {stat_chip("mua units", fmt_int(exp["quality_label_counts"]["mua"]))}
      {stat_chip("non-soma (excluded)", fmt_int(exp["quality_label_counts"]["non-soma"]))}
    </div>
    <div class="fig-grid">
      {figure_block("expert_all_mice_panels.png", "Mice / insertions / good / good+MUA per area, split R+ vs R-, sorted by combined good+MUA.")}
      {figure_block("expert_all_mice_good_vs_goodmua.png", "Good vs Good+MUA per area — R+ (left) vs R- (right).")}
    </div>
    <h3 class="table-heading">Full area table <span class="hint">click a column to sort</span></h3>
    {expert_all_table}
  </section>

  <section class="arm" id="caveats">
    <h2>Caveats &amp; what to check against your own numbers</h2>
    <div class="callout">
      <strong>Joint drift criterion not evaluated.</strong> The DREDge motion-QC merge
      (<code>load_helpers.load_motion_dredge_shift_test_results</code>) was not run for this report, so
      <code>drift_abs_r</code>/<code>drift_shift_test_pval</code> were absent and that joint check was skipped
      entirely &mdash; not a pass, not a fail, just not applied. If your own pipeline includes drift QC, expect
      somewhat higher MUA counts here than yours.
    </div>
    <div class="callout">
      <strong>Insertion definition.</strong> This unit table has no separate <code>probe_name</code> column
      (unlike the compressed-dataset builder) &mdash; only <code>electrode_group</code> (e.g. <code>imec0</code> or
      <code>imec0_shank0</code>). <code>n_insertions</code> here is <code>nunique(session_id, electrode_group)</code>.
      If your own definition is per-probe-device rather than per-shank, expect small differences on multi-shank
      sessions.
    </div>
    <div class="callout">
      <strong>NWB directory listing drifted between the two runs.</strong> {learn["n_nwb_files_scanned"]:,} files were
      scanned for the learning arm vs. {exp["n_nwb_files_scanned"]:,} for the expert arm (the two runs were ~10
      minutes apart on a live, shared drive). Both arms' resulting ephys-session lists were inspected and look
      complete for their respective day filters.
    </div>
    <div class="callout">
      <strong>reward_group has two non-interchangeable sources in this project.</strong> This report splits on the
      <em>native</em> <code>unit_table['reward_group']</code> that <code>process_single_nwb</code> attaches directly
      from each session's <code>wh_reward</code> metadata field (int 0/1) &mdash; not the reference sheet's per-mouse
      <code>reward_group</code> column, which has three levels (<code>R+</code>/<code>R-</code>/<code>R+proba</code>).
      An <code>R+proba</code> mouse's sessions show up here as plain <code>R+</code>, since <code>wh_reward</code> is
      only binary. If your own split uses the reference-sheet column, expect any R+proba mice to land differently.
    </div>
    <div class="callout">
      <strong>{len(exp_inconsistent)} mouse with inconsistent reward_group across expert sessions.</strong>
      <span class="mono">{html.escape(", ".join(exp_inconsistent)) if exp_inconsistent else "none"}</span>
      had sessions disagreeing on <code>wh_reward</code> across its expert-arm recordings &mdash; each session's own
      value was still used for that session's units (not resolved to a single per-mouse label), so this mouse's units
      are genuinely split across both R+ and R- columns above.
    </div>
  </section>

  <footer>
    <p>Generated 2026-08-26 &middot; <code>scripts/ssl_ephys_utilities_qc_report.py</code>, run with the
      <code>bwa</code> conda env (only local env with <code>pynwb</code> + <code>ephys_utilities</code> +
      <code>allen_utils</code> all importable together).</p>
    <p>Source files alongside this report: <code>run_metadata.json</code>, three
      <code>*_area_summary.csv</code> tables, six <code>*.png</code> figures.</p>
  </footer>
</main>

<script>
(function () {{
  function cellValue(td, type) {{
    if (type === "num") return parseFloat(td.textContent.replace(/,/g, "")) || 0;
    return td.textContent.trim().toLowerCase();
  }}
  document.querySelectorAll("table.data-table").forEach(function (table) {{
    var state = {{ col: null, dir: 1 }};
    table.querySelectorAll("thead th").forEach(function (th) {{
      th.setAttribute("tabindex", "0");
      var activate = function () {{
        var col = parseInt(th.dataset.col, 10);
        var type = th.dataset.type;
        var dir = state.col === col ? -state.dir : (type === "num" ? -1 : 1);
        state = {{ col: col, dir: dir }};
        table.querySelectorAll("thead th .sort-ind").forEach(function (s) {{ s.textContent = ""; }});
        th.querySelector(".sort-ind").textContent = dir === 1 ? " \u2191" : " \u2193";
        var tbody = table.querySelector("tbody");
        var rows = Array.prototype.slice.call(tbody.querySelectorAll("tr"));
        rows.sort(function (a, b) {{
          var av = cellValue(a.children[col], type);
          var bv = cellValue(b.children[col], type);
          if (av < bv) return -1 * dir;
          if (av > bv) return 1 * dir;
          return 0;
        }});
        rows.forEach(function (r) {{ tbody.appendChild(r); }});
      }};
      th.addEventListener("click", activate);
      th.addEventListener("keydown", function (e) {{ if (e.key === "Enter" || e.key === " ") {{ e.preventDefault(); activate(); }} }});
    }});
  }});
}})();
</script>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
