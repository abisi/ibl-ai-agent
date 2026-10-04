"""Single-trial convergent-validity check for the PLS+1SE perfquant
predictions (user request 2026-09-23: "get single trial estimates and
correlate that with some other metrics to gain confidence the model
makes sense"). Does the model's single-trial PREDICTED performance state
(`Y_pred`, not the residual -- the question is whether what the model
says tracks independent behavior, not whether its errors do) relate to
independent single-trial behavioral signals in the expected direction?

Reuses `065`'s cache (`065_perfquant_allmice_cache.pkl`, 98 good/moderate
sessions, no PLS recompute) for `Y_pred`/`trial_index`. Two independent
metrics, deliberately NOT trial-index/session-time (that would just
re-confirm the already-established slow-drift confound, not test against
it):
- **RT** (`lick_time - response_window_start_time`, corrected 2026-09-27; hit trials only -- undefined on a
  miss).
- **Previous-trial outcome** (`lick_flag` shifted by one position within
  the session's own full whisker-trial sequence, matched via
  `_active_trials_from_whisker_onset_for_curve` -- the SAME trial
  construction `prep_perfquant_curve_targets` uses internally, so
  `trial_index` (positions into that full sequence, pre-`valid`-mask)
  aligns exactly).

Statistical design (avoids the trial-autocorrelation p-value trap): ONE
Pearson r (and Spearman rho) per session per (target, metric) -- a
descriptive per-session point estimate, no within-session p-value is
ever trusted. Group-level significance comes from a Wilcoxon signed-rank
+ one-sample t-test (both reported, this project's standing convention)
on those per-session r values against 0, across sessions within each
(day_stage, cohort, target, metric) cell -- same logic as every
above-null test in this project, applied to a face-validity correlation
instead of a decoding score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon

SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS_DIR)

OUT_DIR = Path(__file__).resolve().parent
TARGETS = ["whisker_curve", "falsealarm_curve", "performance_curve"]
METRICS = ["RT", "prev_outcome"]
STAGES = ["learning", "expert"]
COHORTS = ["R+", "R-"]
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
MIN_TRIALS_RT = 10  # min hit trials in a session for its RT correlation to count
MIN_TRIALS_PREVOUT = 10


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q_sorted = ranked * n / (np.arange(n) + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


def _stars(q):
    if np.isnan(q):
        return "n/a"
    return "***" if q < 0.001 else ("**" if q < 0.01 else ("*" if q < 0.05 else "n.s."))


def session_rt_and_prevoutcome(session_id: str, trials_tbl: pd.DataFrame, n_full: int):
    """Full (pre-valid-mask) whisker-trial-sequence RT and lag-1 outcome,
    same row construction/order `prep_perfquant_curve_targets` uses
    internally, so positions match `trial_index` from the 065 cache."""
    from ssl_timeresolved_decoding import _active_trials_from_whisker_onset_for_curve

    all_trials = _active_trials_from_whisker_onset_for_curve(session_id, trials_tbl)
    whisker = all_trials[all_trials["trial_type"] == "whisker_trial"].sort_values("start_time").reset_index(drop=True)
    if len(whisker) < n_full:
        return None, None  # shouldn't happen -- same construction as the cache -- guarded, not silently truncated
    lick_flag = whisker["lick_flag"].to_numpy().astype(float)
    rt = (whisker["lick_time"] - whisker["response_window_start_time"]).to_numpy().copy()  # corrected RT from stimulus onset (skills/ssl-lick-alignment, 2026-09-27); per-session r unchanged (session-constant offset)
    rt[lick_flag == 0] = np.nan  # RT undefined on a miss
    prev_outcome = np.concatenate([[np.nan], lick_flag[:-1]])  # lag-1 within the FULL sequence, before subsetting
    return rt, prev_outcome


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir

    with open(OUT_DIR / "065_perfquant_allmice_cache.pkl", "rb") as f:
        cache = pickle.load(f)
    print(f"loaded {len(cache)} sessions from 065's cache", flush=True)

    dataset_root = resolve_dataset_dir("ssl_ephys")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")

    records = []
    for res in cache:
        session_id = res["session_id"]
        trial_index = res["trial_index"]
        n_full = int(trial_index.max()) + 1 if len(trial_index) else 0
        rt_full, prevout_full = session_rt_and_prevoutcome(session_id, trials_tbl, n_full)
        if rt_full is None:
            print(f"  {res['mouse']}/{session_id}: trial-count mismatch vs cache, skipped")
            continue
        rt = rt_full[trial_index]
        prev_outcome = prevout_full[trial_index]

        for target_idx, target in enumerate(TARGETS):
            y_pred = res["Y_pred"][:, target_idx]

            rt_mask = ~np.isnan(rt)
            if rt_mask.sum() >= MIN_TRIALS_RT:
                pr, _ = pearsonr(y_pred[rt_mask], rt[rt_mask])
                sr, _ = spearmanr(y_pred[rt_mask], rt[rt_mask])
                records.append(dict(mouse=res["mouse"], session_id=session_id, day_stage=res["day_stage"],
                                     reward_group=res["reward_group"], target=target, metric="RT",
                                     n_trials=int(rt_mask.sum()), pearson_r=pr, spearman_r=sr))

            po_mask = ~np.isnan(prev_outcome)
            if po_mask.sum() >= MIN_TRIALS_PREVOUT:
                pr, _ = pearsonr(y_pred[po_mask], prev_outcome[po_mask])
                sr, _ = spearmanr(y_pred[po_mask], prev_outcome[po_mask])
                records.append(dict(mouse=res["mouse"], session_id=session_id, day_stage=res["day_stage"],
                                     reward_group=res["reward_group"], target=target, metric="prev_outcome",
                                     n_trials=int(po_mask.sum()), pearson_r=pr, spearman_r=sr))

    per_session = pd.DataFrame(records)
    per_session.to_csv(OUT_DIR / "077_perfquant_singletrial_facevalidity_persession.csv", index=False)
    print(f"{len(per_session)} (session, target, metric) rows -- from {per_session.session_id.nunique()} sessions", flush=True)

    cells = []
    for stage in STAGES:
        for cohort in COHORTS:
            for target in TARGETS:
                for metric in METRICS:
                    g = per_session[(per_session.day_stage == stage) & (per_session.reward_group == cohort)
                                     & (per_session.target == target) & (per_session.metric == metric)]
                    vals = g["pearson_r"].dropna().values
                    rec = dict(day_stage=stage, cohort=cohort, target=target, metric=metric, n_sessions=len(vals))
                    if len(vals) >= 5:
                        try:
                            wp = wilcoxon(vals).pvalue
                        except ValueError:
                            wp = np.nan
                        tp = ttest_1samp(vals, 0.0).pvalue
                        rec.update(wilcoxon_p=wp, ttest_p=tp, median_r=float(np.median(vals)), mean_r=float(np.mean(vals)))
                    else:
                        rec.update(wilcoxon_p=np.nan, ttest_p=np.nan,
                                    median_r=float(np.median(vals)) if len(vals) else np.nan,
                                    mean_r=float(np.mean(vals)) if len(vals) else np.nan)
                    cells.append(rec)
    result_df = pd.DataFrame(cells)
    for col in ("wilcoxon_p", "ttest_p"):
        mask = result_df[col].notna()
        q = np.full(len(result_df), np.nan)
        q[mask.values] = bh_fdr(result_df.loc[mask, col].values)
        result_df[col.replace("_p", "_q")] = q
    result_df.to_csv(OUT_DIR / "077_perfquant_singletrial_facevalidity_summary.csv", index=False)

    print("\n=== single-trial face-validity: Y_pred vs RT / previous-trial outcome ===")
    for stage in STAGES:
        for cohort in COHORTS:
            sub = result_df[(result_df.day_stage == stage) & (result_df.cohort == cohort)]
            print(f"-- {stage} / {cohort} --")
            for _, row in sub.iterrows():
                print(f"    {row['target']:<20} {row['metric']:<13} n_sessions={int(row['n_sessions']):>3}  "
                      f"median_r={row['median_r']:+.3f}  wilcoxon p={row['wilcoxon_p']:.3g} q={row['wilcoxon_q']:.3g} "
                      f"{_stars(row['wilcoxon_q'])}  t-test p={row['ttest_p']:.3g} q={row['ttest_q']:.3g} "
                      f"{_stars(row['ttest_q'])}")

    # --- Figure: per-mouse strip, rows=metric, cols=(day_stage,cohort), x=target ---
    cell_order = [(s, c) for s in STAGES for c in COHORTS]
    fig, axes = plt.subplots(len(METRICS), len(cell_order), figsize=(3.6 * len(cell_order), 3.2 * len(METRICS)),
                              constrained_layout=True, squeeze=False)
    rng = np.random.default_rng(0)
    for row_i, metric in enumerate(METRICS):
        for col_i, (stage, cohort) in enumerate(cell_order):
            ax = axes[row_i][col_i]
            color = COHORT_COLOR[cohort]
            for x, target in enumerate(TARGETS):
                g = per_session[(per_session.day_stage == stage) & (per_session.reward_group == cohort)
                                 & (per_session.target == target) & (per_session.metric == metric)]
                vals = g["pearson_r"].dropna().values
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), x) + jitter, vals, s=14, color=color, alpha=0.6, edgecolors="none")
                mean, sem = np.mean(vals), (np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                ax.errorbar(x, mean, yerr=sem, fmt="D", color="black", markersize=5, capsize=3, zorder=3)
                row = result_df[(result_df.day_stage == stage) & (result_df.cohort == cohort)
                                 & (result_df.target == target) & (result_df.metric == metric)].iloc[0]
                ax.text(x, max(vals.max(), mean + sem) + 0.03, _stars(row["wilcoxon_q"]), ha="center", va="bottom", fontsize=7.5)
            ax.axhline(0, color="#888888", lw=0.7, linestyle=":")
            ax.set_xticks(range(len(TARGETS)))
            ax.set_xticklabels([t.replace("_curve", "") for t in TARGETS], fontsize=7.5, rotation=15, ha="right")
            if row_i == 0:
                ax.set_title(f"{stage} / {cohort}", fontsize=9.5, color=color)
            if col_i == 0:
                ax.set_ylabel(f"per-session Pearson r\nY_pred vs {metric}", fontsize=8.5)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Single-trial face validity: predicted performance state vs RT / previous-trial outcome\n"
                 "(* = BH-FDR q<0.05, Wilcoxon vs 0; points = sessions)", fontsize=11.5)
    fig_path = OUT_DIR / "077_perfquant_singletrial_facevalidity.png"
    fig.savefig(fig_path, dpi=150)
    print(f"\nsaved {fig_path.name}")
    print("DONE_077")


if __name__ == "__main__":
    main()
