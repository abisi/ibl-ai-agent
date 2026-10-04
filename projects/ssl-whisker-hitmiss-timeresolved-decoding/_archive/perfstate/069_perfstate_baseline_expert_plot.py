"""Expert-stage counterpart to `046_perfstate_baseline_decoding.py`'s own
figure, plotting from the ALREADY-COMPUTED
`046_perfstate_baseline_decoding_results_expert.csv` (pulled from haas
2026-09-23, computed there 2026-09-18) rather than recomputing -- this is
pure plotting, no new model fits, same pattern as `047`'s CSV-only
convention. `046`'s own `main()` always recomputes before plotting and has
no plot-only path, so this script lifts its plotting block verbatim
(the part starting at `ok = df[df["ok"] == True]`) rather than re-running
the expensive per-session decode a second time for a result that already
exists.

Closes the TODO.md (2026-09-23 status recap) item: "046 expert-stage
perf-state figure (raw + above-null + cohort-split, matching the
learning-stage one) -- still doesn't exist."

Usage: python 069_perfstate_baseline_expert_plot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
DAY_STAGE = "expert"
SUFFIX = f"_{DAY_STAGE}"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}  # this project's standing R+/R- color convention
AGGREGATE_COLOR = "#2c5f5b"  # matches 028/032's own "combined/aggregate" color


def savefig_retry(fig, out_path: Path, attempts: int = 5, delay: float = 1.0, **kwargs):
    import time
    last_err = None
    for attempt in range(attempts):
        try:
            fig.savefig(out_path, **kwargs)
            return
        except OSError as e:
            last_err = e
            print(f"  savefig attempt {attempt + 1}/{attempts} failed ({e}), retrying in {delay}s...")
            time.sleep(delay)
    raise last_err


def main():
    in_csv = OUT_DIR / f"046_perfstate_baseline_decoding_results{SUFFIX}.csv"
    if not in_csv.exists():
        print(f"missing {in_csv.name} -- pull it from haas first")
        sys.exit(1)
    df = pd.read_csv(in_csv)

    ok = df[df["ok"] == True]  # noqa: E712
    print(f"{len(ok)}/{len(df)} (session, window) cells decodable")
    for window_name in ("quiet_window", "iti"):
        sub = ok[ok.window == window_name]
        if len(sub):
            print(f"  {window_name}: n={len(sub)} sessions, real acc mean={sub['real'].mean():.3f}, "
                  f"null mean={sub['null_mean'].mean():.3f}")

    from ssl_bwm_trial_prep import REF_PATH
    ref = pd.read_parquet(Path(REF_PATH))
    reward_group_map = ref.set_index("subject_id")["reward_group"].to_dict()
    ok = ok.copy()
    ok["diff"] = ok["real"] - ok["null_mean"]
    ok["reward_group"] = ok["mouse"].map(reward_group_map)
    n_no_group = int(ok["reward_group"].isna().sum())
    if n_no_group:
        print(f"  NOTE: {n_no_group}/{len(ok)} rows have no R+/R- reward_group (excluded from cohort panels)")

    def _paired_panel(ax, sub: pd.DataFrame, values_col: str, ref_line: float, ylim: tuple[float, float], color: str):
        wide = sub.pivot_table(index="session_id", columns="window", values=values_col)
        wide = wide.dropna(subset=[c for c in ("quiet_window", "iti") if c in wide.columns])
        if {"quiet_window", "iti"}.issubset(wide.columns) and len(wide) >= 2:
            for _, r in wide.iterrows():
                ax.plot([0, 1], [r["quiet_window"], r["iti"]], color="#bbbbbb", lw=0.6, alpha=0.5, zorder=1)
            rng = np.random.default_rng(0)
            for x, col in enumerate(("quiet_window", "iti")):
                vals = wide[col].to_numpy()
                jitter = rng.uniform(-0.03, 0.03, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, zorder=2, color=color, alpha=0.85,
                           edgecolors="none" if x == 0 else color, facecolors=color if x == 0 else "none",
                           linewidths=1.1)
                ax.errorbar(x, vals.mean(), yerr=vals.std() / np.sqrt(len(vals)), fmt="D", color="black",
                            markersize=6, capsize=3, zorder=3)
            w_p = stats.wilcoxon(wide["quiet_window"], wide["iti"]).pvalue if len(wide) >= 3 else float("nan")
            t_p = stats.ttest_rel(wide["quiet_window"], wide["iti"]).pvalue
            note = f"paired: Wilcoxon p={w_p:.3g}, t p={t_p:.3g}"
            if ref_line == 0.0:
                qw_p = stats.wilcoxon(wide["quiet_window"]).pvalue if len(wide) >= 3 else float("nan")
                iti_p = stats.wilcoxon(wide["iti"]).pvalue if len(wide) >= 3 else float("nan")
                note += f"\nvs 0: qw p={qw_p:.3g}, ITI p={iti_p:.3g}"
            ax.set_title(f"n={len(wide)}\n{note}", fontsize=7.5, color="#444444")
        else:
            ax.set_title(f"n={len(wide)} (too few for stats)", fontsize=7.5, color="#444444")
        ax.axhline(ref_line, color="#888888", lw=1, linestyle=":")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["quiet_window", "ITI"], fontsize=8)
        ax.set_ylim(*ylim)
        ax.set_box_aspect(1)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)

    diff_ylim = (min(-0.02, ok["diff"].min() - 0.02), max(0.05, ok["diff"].max() + 0.02))
    cohorts = [("combined", ok, AGGREGATE_COLOR), ("R+", ok[ok.reward_group == "R+"], COHORT_COLOR["R+"]),
               ("R-", ok[ok.reward_group == "R-"], COHORT_COLOR["R-"])]
    metric_rows = [("real", "raw accuracy\nbalanced accuracy (perf-state)", 0.5, (0.3, 1.02)),
                   ("diff", "above shift-null\naccuracy - shift-null mean", 0.0, diff_ylim)]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 9), constrained_layout=True)
    for row_i, (values_col, ylabel, ref_line, ylim) in enumerate(metric_rows):
        for col_i, (cohort_label, sub, color) in enumerate(cohorts):
            ax = axes[row_i, col_i]
            _paired_panel(ax, sub, values_col, ref_line, ylim, color)
            if row_i == 0:
                ax.set_title(f"{cohort_label}\n{ax.get_title()}", fontsize=9, color=color if cohort_label != "combined" else "#444444")
            if col_i == 0:
                ax.set_ylabel(ylabel, fontsize=8.5)
    fig.suptitle(f"quiet_window vs ITI, perf-state decode, by cohort ({DAY_STAGE} stage)", fontsize=12)
    out_path = OUT_DIR / f"046_perfstate_baseline_decoding{SUFFIX}.png"
    savefig_retry(fig, out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path.name}")


if __name__ == "__main__":
    main()
