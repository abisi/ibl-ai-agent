"""Comprehensive behavioral summary figure (user request 2026-09-12):
"Show a grid of sessions with performance state classification. Show
summary performance (whisker rate, false alarm rate, auditory rate)
statistics again for session halves and perf states. All in one figure
with subplots. Compare both within group and across groups, both with
mouse lines and no mouse lines. Add stat tests, paired or unpaired."

Layout (one figure, gridspec):
  - Top: session x block performance-state grid, one heatmap per cohort
    (R+, R-) -- block_id on x, session on y, color = high/low perf state.
  - Below: 3 metrics (whisker hit rate, FA rate, auditory hit rate) x 4
    columns (half w/ mouse lines, half w/o, perfstate w/ mouse lines,
    perfstate w/o) = 12 panels. Every panel overlays R+ (green) and R-
    (magenta): mean+-SEM line connecting the two condition levels, plus
    (in the "w/ mouse lines" columns) each session's own thin connecting
    line. Every panel is annotated with BOTH the within-cohort paired test
    (Wilcoxon + paired-t, R+ and R- separately) and the across-cohort
    unpaired test at each condition level (Mann-Whitney + Welch) -- same
    convention as `027`/`029`, just applied to behavior instead of decoding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from ssl_timeresolved_decoding import hitmiss_session_list, prep_perfstate_trials_generic  # noqa: E402
from ssl_bwm_trial_prep import prep_session  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
METRICS = [("whisker_hit_rate", "whisker hit rate"), ("fa_rate", "false alarm rate"), ("auditory_hit_rate", "auditory hit rate"),
           ("delta_lick_prob", "delta P(lick|whisker)-P(lick|FA)")]


def half_split_rates(trials: pd.DataFrame) -> dict | None:
    whisker = trials[trials["trial_type"] == "whisker_trial"].reset_index(drop=True)
    auditory = trials[trials["trial_type"] == "auditory_trial"]
    no_stim = trials[trials["trial_type"] == "no_stim_trial"]
    if len(whisker) == 0:
        return None
    median_idx = len(whisker) // 2
    whisker["half"] = ["first"] * median_idx + ["second"] * (len(whisker) - median_idx)
    out = {}
    for half in ("first", "second"):
        wh = whisker[whisker["half"] == half]
        if len(wh) == 0:
            continue
        t0, t1 = wh["start_time"].min(), wh["start_time"].max()
        ns = no_stim[(no_stim["start_time"] >= t0) & (no_stim["start_time"] <= t1)]
        au = auditory[(auditory["start_time"] >= t0) & (auditory["start_time"] <= t1)]
        whisker_hit_rate = wh["lick_flag"].mean()
        fa_rate = ns["lick_flag"].mean() if len(ns) else np.nan
        out[half] = dict(whisker_hit_rate=whisker_hit_rate, fa_rate=fa_rate,
                          auditory_hit_rate=au["lick_flag"].mean() if len(au) else np.nan,
                          delta_lick_prob=whisker_hit_rate - fa_rate)
    return out


def perfstate_split_rates(dataset_root, session_id, sessions_tbl, trials_tbl) -> tuple[dict | None, pd.DataFrame | None]:
    whisker = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["whisker_trial"])
    if whisker is None or len(whisker) == 0:
        return None, None
    auditory = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["auditory_trial"])
    no_stim = prep_perfstate_trials_generic(dataset_root, session_id, sessions_tbl, trials_tbl, decode_trial_types=["no_stim_trial"])
    out = {}
    for state in ("high", "low"):
        wh = whisker[whisker["perf_state"] == state]
        au = auditory[auditory["perf_state"] == state] if auditory is not None else pd.DataFrame()
        ns = no_stim[no_stim["perf_state"] == state] if no_stim is not None else pd.DataFrame()
        whisker_hit_rate = wh["lick_flag"].mean() if len(wh) else np.nan
        fa_rate = ns["lick_flag"].mean() if len(ns) else np.nan
        out[state] = dict(whisker_hit_rate=whisker_hit_rate, fa_rate=fa_rate,
                           auditory_hit_rate=au["lick_flag"].mean() if len(au) else np.nan,
                           delta_lick_prob=whisker_hit_rate - fa_rate)
    block_lick_prob = whisker.groupby("block_id")["lick_flag"].mean()
    block_fa = whisker.groupby("block_id")["block_fa_rate"].first()
    delta_lick_prob = (block_lick_prob - block_fa).rename("delta_lick_prob")
    blocks = whisker[["block_id", "perf_state"]].drop_duplicates().sort_values("block_id")
    blocks = blocks.merge(delta_lick_prob, on="block_id", how="left")
    return out, blocks


def build_tables():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    hm = hitmiss_session_list(sessions_tbl)

    long_rows = []
    block_rows = []
    for r in hm.itertuples():
        prepped = prep_session(dataset_root, r.session_id, sessions_tbl, trials_tbl)
        if prepped is None:
            continue
        trials = prepped["trials"]
        if len(trials[trials["trial_type"] == "whisker_trial"]) == 0:
            continue

        half_rates = half_split_rates(trials)
        if half_rates:
            for half, metrics in half_rates.items():
                for metric_col, val in metrics.items():
                    long_rows.append(dict(session_id=r.session_id, subject_id=r.subject_id, reward_group=r.reward_group,
                                           condition_type="half", condition_value=half, metric=metric_col, value=val))

        perf_rates, blocks = perfstate_split_rates(dataset_root, r.session_id, sessions_tbl, trials_tbl)
        if perf_rates:
            for state, metrics in perf_rates.items():
                for metric_col, val in metrics.items():
                    long_rows.append(dict(session_id=r.session_id, subject_id=r.subject_id, reward_group=r.reward_group,
                                           condition_type="perfstate", condition_value=state, metric=metric_col, value=val))
        if blocks is not None:
            for row in blocks.itertuples():
                block_rows.append(dict(session_id=r.session_id, reward_group=r.reward_group,
                                        block_id=row.block_id, perf_state=row.perf_state,
                                        delta_lick_prob=row.delta_lick_prob))

    return pd.DataFrame(long_rows), pd.DataFrame(block_rows)


def plot_session_block_grid(ax_state, ax_delta, block_df: pd.DataFrame, cohort: str):
    sub = block_df[block_df.reward_group == cohort]
    if len(sub) == 0:
        ax_state.axis("off")
        ax_delta.axis("off")
        return

    session_order = sorted(sub["session_id"].unique())

    piv_state = sub.pivot_table(index="session_id", columns="block_id", values="perf_state", aggfunc="first",
                                 observed=True).map(lambda v: 1.0 if v == "high" else (0.0 if v == "low" else np.nan))
    piv_state = piv_state.reindex(session_order)
    cmap_state = matplotlib.colors.ListedColormap(["#c8c8ff", COHORT_COLOR[cohort]])
    ax_state.imshow(piv_state.to_numpy(dtype=float), aspect="auto", cmap=cmap_state, vmin=0, vmax=1, interpolation="none")
    ax_state.set_title(f"{cohort}: perf-state per session x block (n={len(piv_state)} sessions)", fontsize=9.5)
    ax_state.set_xlabel("block index (5 whisker trials each)", fontsize=8.5)
    ax_state.set_ylabel("session", fontsize=8.5)
    ax_state.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color="#c8c8ff", label="low"), plt.Rectangle((0, 0), 1, 1, color=COHORT_COLOR[cohort], label="high")]
    ax_state.legend(handles=handles, fontsize=7, frameon=False, loc="upper right")

    piv_delta = sub.pivot_table(index="session_id", columns="block_id", values="delta_lick_prob", aggfunc="first", observed=True)
    piv_delta = piv_delta.reindex(session_order)
    vmax = np.nanmax(np.abs(piv_delta.to_numpy(dtype=float))) if piv_delta.size else 1.0
    im = ax_delta.imshow(piv_delta.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="none")
    ax_delta.set_title(f"{cohort}: delta P(lick|whisker)-P(lick|FA) per block", fontsize=9.5)
    ax_delta.set_xlabel("block index (5 whisker trials each)", fontsize=8.5)
    ax_delta.set_ylabel("session", fontsize=8.5)
    ax_delta.set_yticks([])
    plt.colorbar(im, ax=ax_delta, fraction=0.03, pad=0.02)


def metric_panel(ax, df: pd.DataFrame, metric: str, condition_type: str, values: tuple[str, str], show_lines: bool):
    sub = df[(df.condition_type == condition_type) & (df.metric == metric)]
    xs = [0, 1]
    p_lines = []
    group_ab = {}
    for cohort in ("R+", "R-"):
        csub = sub[sub.reward_group == cohort]
        piv = csub.pivot_table(index="session_id", columns="condition_value", values="value")
        if values[0] not in piv.columns or values[1] not in piv.columns:
            continue
        a, b = piv[values[0]].to_numpy(), piv[values[1]].to_numpy()
        valid = ~(np.isnan(a) | np.isnan(b))
        a, b = a[valid], b[valid]
        group_ab[cohort] = (a, b)
        n = len(a)
        color = COHORT_COLOR[cohort]
        if show_lines:
            for ai, bi in zip(a, b):
                ax.plot(xs, [ai, bi], color=color, alpha=0.15, lw=0.7, zorder=1)
        if n >= 2:
            mean_a, mean_b = np.mean(a), np.mean(b)
            sem_a, sem_b = np.std(a) / np.sqrt(n), np.std(b) / np.sqrt(n)
            ax.errorbar(xs, [mean_a, mean_b], yerr=[sem_a, sem_b], color=color, lw=2.2, marker="o",
                        markersize=5.5, zorder=3, capsize=3, label=cohort)
            try:
                w_p = stats.wilcoxon(a, b).pvalue if np.any(a != b) else np.nan
            except ValueError:
                w_p = np.nan
            t_p = stats.ttest_rel(a, b).pvalue
            p_lines.append(f"{cohort}: n={n}, W={w_p:.2g}, t={t_p:.2g}")
        else:
            p_lines.append(f"{cohort}: n={n}")

    if "R+" in group_ab and "R-" in group_ab:
        (a_rplus, b_rplus), (a_rminus, b_rminus) = group_ab["R+"], group_ab["R-"]
        for level_label, va, vb in ((values[0], a_rplus, a_rminus), (values[1], b_rplus, b_rminus)):
            vp, vm = va[~np.isnan(va)], vb[~np.isnan(vb)]
            if len(vp) >= 2 and len(vm) >= 2:
                mw_p = stats.mannwhitneyu(vp, vm, alternative="two-sided").pvalue
                p_lines.append(f"R+ vs R- @{level_label}: MW={mw_p:.2g}")

    ax.text(0.5, 0.02, "\n".join(p_lines), transform=ax.transAxes, fontsize=6.3, ha="center", va="bottom", linespacing=1.4)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(values, fontsize=8.5)
    ax.set_box_aspect(1)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    print("Building behavioral summary tables...")
    long_df, block_df = build_tables()
    print(f"  {long_df['session_id'].nunique()} sessions in long table, {block_df['session_id'].nunique()} sessions with blocks")

    n_rows = 2 + len(METRICS)
    fig = plt.figure(figsize=(17, 3.85 * n_rows))
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, height_ratios=[1.1, 1.1] + [1] * len(METRICS), hspace=0.55, wspace=0.5)

    ax_r1_state = fig.add_subplot(gs[0, 0:2])
    ax_r2_state = fig.add_subplot(gs[0, 2:4])
    ax_r1_delta = fig.add_subplot(gs[1, 0:2])
    ax_r2_delta = fig.add_subplot(gs[1, 2:4])
    plot_session_block_grid(ax_r1_state, ax_r1_delta, block_df, "R+")
    plot_session_block_grid(ax_r2_state, ax_r2_delta, block_df, "R-")

    col_specs = [("half", ("first", "second"), True), ("half", ("first", "second"), False),
                 ("perfstate", ("high", "low"), True), ("perfstate", ("high", "low"), False)]
    col_titles = ["halves (mouse lines)", "halves (no lines)", "perfstate (mouse lines)", "perfstate (no lines)"]

    for i, (metric, metric_label) in enumerate(METRICS):
        for j, (condition_type, values, show_lines) in enumerate(col_specs):
            ax = fig.add_subplot(gs[i + 2, j])
            metric_panel(ax, long_df, metric, condition_type, values, show_lines)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_xlabel("condition", fontsize=8.5)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=9.5)
            if i == len(METRICS) - 1 and j == 0:
                ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.suptitle("Behavioral summary: performance-state classification and session-half/perfstate comparisons", fontsize=14)
    out_path = OUT_DIR / "031_behavioral_summary_grid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
