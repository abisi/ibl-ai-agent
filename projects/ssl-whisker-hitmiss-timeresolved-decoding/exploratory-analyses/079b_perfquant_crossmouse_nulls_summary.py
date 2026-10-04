"""Summary + figure stage for `079`, run standalone from the already-
computed `079_perfquant_crossmouse_nulls_raw.csv` -- `079`'s own summary
code crashed post-compute (`row[metric][k]` on a pandas itertuples
namedtuple, which doesn't support string-key indexing; needed
`getattr(row, metric)[k]`) AFTER all 355 cells had already finished and
been saved to CSV, so this reruns only the cheap reporting step, not the
compute. The CSV also stringifies the list-valued r2/pearson/spearman
columns as `"[np.float64(...), ...]"` (not literal-eval-safe), so this
parses them with a float regex instead of assuming a clean list format.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
METRICS = ["r2", "pearson", "spearman"]
RECIPIENTS = [
    ("MH070", "MH070_20260121_140848"),
    ("MH069", "MH069_20260122_111455"),
    ("MH031", "MH031_20250507_104425"),
    ("AB162", "AB162_20250421_140550"),
    ("AB087", "AB087_20231017_141901"),
]

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_list_col(s: str) -> list[float]:
    """Parses both plain-float list reprs (the r2 column, e.g.
    "[0.86, 0.90, 0.71]") and numpy-scalar list reprs (pearson/spearman,
    e.g. "[np.float64(0.93), np.float64(0.95), np.float64(0.86)]") --
    stripping the `np.float64(...)` wrapper FIRST is essential: a naive
    float regex over the raw string also matches the "64" inside
    "float64" itself as a spurious extra number, silently corrupting 2 of
    every 3 parsed values (found and fixed 2026-09-23, after it produced
    impossible |r|>1 "correlations" in the first pass of this script)."""
    s = s.replace("np.float64(", "").replace(")", "")
    return [float(x) for x in _FLOAT_RE.findall(s)]


def main():
    df = pd.read_csv(OUT_DIR / "079_perfquant_crossmouse_nulls_raw.csv")
    for m in METRICS:
        df[m] = df[m].apply(parse_list_col)

    summary = []
    for rmouse, rsid in RECIPIENTS:
        real_row = df[(df.recipient_sid == rsid) & (df.null_type == "real")]
        if len(real_row) == 0:
            print(f"{rmouse}/{rsid}: no real fit, skipped")
            continue
        for null_type in ("rminus_null", "withinrplus_null"):
            donors = df[(df.recipient_sid == rsid) & (df.null_type == null_type)]
            for k, target in enumerate(TARGETS):
                for metric in METRICS:
                    real_val = real_row.iloc[0][metric][k]
                    donor_vals = np.array([v[k] for v in donors[metric]])
                    if len(donor_vals) == 0:
                        continue
                    corrected = real_val - np.median(donor_vals)
                    p = (1 + np.sum(donor_vals >= real_val)) / (1 + len(donor_vals))
                    summary.append(dict(recipient=rmouse, session_id=rsid, null_type=null_type, target=target,
                                         metric=metric, real=real_val, donor_median=float(np.median(donor_vals)),
                                         n_donors=len(donor_vals), corrected=corrected, p=p))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "079_perfquant_crossmouse_nulls_summary.csv", index=False)

    print("=== cross-mouse null results (pearson only shown; full table in CSV) ===")
    for rmouse, rsid in RECIPIENTS:
        sub = summary_df[(summary_df.recipient == rmouse) & (summary_df.metric == "pearson")]
        if len(sub) == 0:
            continue
        print(f"-- {rmouse} --")
        for _, row in sub.iterrows():
            print(f"    {row['null_type']:<18} {row['target']:<20} real={row['real']:+.3f}  "
                  f"donor_median={row['donor_median']:+.3f}  corrected={row['corrected']:+.3f}  "
                  f"p={row['p']:.3g}  (n_donors={int(row['n_donors'])})")

    fig, axes = plt.subplots(len(RECIPIENTS), 2, figsize=(9, 3.2 * len(RECIPIENTS)), constrained_layout=True)
    null_labels = {"rminus_null": "R- null\n(reward-dependence)", "withinrplus_null": "within-R+ null\n(specificity)"}
    for row_i, (rmouse, rsid) in enumerate(RECIPIENTS):
        for col_i, null_type in enumerate(("rminus_null", "withinrplus_null")):
            ax = axes[row_i][col_i]
            donors = df[(df.recipient_sid == rsid) & (df.null_type == null_type)]
            real_row = df[(df.recipient_sid == rsid) & (df.null_type == "real")]
            if len(donors) == 0 or len(real_row) == 0:
                continue
            positions, data = [], []
            for k, target in enumerate(TARGETS):
                data.append(np.array([v[k] for v in donors["pearson"]]))
                positions.append(k)
            parts = ax.violinplot(data, positions=positions, showmeans=False, showextrema=False)
            for pc in parts["bodies"]:
                pc.set_facecolor("#999999")
                pc.set_alpha(0.5)
            for k, target in enumerate(TARGETS):
                ax.scatter([k], [real_row.iloc[0]["pearson"][k]], color="#d62728", s=50, zorder=5, marker="D")
            ax.axhline(0, color="#888888", lw=0.6, linestyle=":")
            ax.set_xticks(range(len(TARGETS)))
            ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=7.5)
            ax.set_title(f"{rmouse} -- {null_labels[null_type]}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cross-mouse session-permutation nulls (red diamond = real, violin = donor distribution)", fontsize=12)
    fig_path = OUT_DIR / "079_perfquant_crossmouse_nulls.png"
    fig.savefig(fig_path, dpi=140)
    print(f"\nsaved {fig_path.name}")
    print("DONE_079b")


if __name__ == "__main__":
    main()
