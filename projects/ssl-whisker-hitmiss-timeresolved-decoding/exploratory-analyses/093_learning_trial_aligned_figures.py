"""Figures for `092_learning_trial_aligned_decoding.py`: hit/miss decoding
(sensory window 5-50ms) aligned to each session's learning_trial, with the
behavioral learning curve aligned the same way (user request 2026-09-24).

Per trial: `dec = p_correct - p_correct_null` (held-out P(true class) minus
that trial's own label-shuffle null). Trials binned by `rel_idx` (whisker
trials relative to learning_trial, 0 = learning trial) in BIN-trial bins.
Per session and bin: class-balanced decoding = mean of (hit-trial mean,
miss-trial mean), only when both classes are present in that bin; hits and
misses also shown separately. Curves = mean +- SEM across sessions.

"At-LT" test: per session, mean class-balanced `dec` over trials in
[-LOCAL, 0) vs [0, LOCAL) (both classes required in each) -> paired
Wilcoxon + paired t within cohort; R+ vs R- on the local delta with MWU +
Welch (mandatory test pair).

Every panel's session count per bin is shown (bottom row), split into
criterion-met vs flagged (floor10 / fallback10) learning_trials -- floor10
sessions contribute no trials before rel_idx=-10 (their learning_trial is
10), so the far-left bins are drawn only from late-learning sessions.

Usage (local, after pulling 092 parquets): python 093_learning_trial_aligned_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, ttest_rel, wilcoxon

OUT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("q034", OUT_DIR / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q034)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
FLAGGED = {"learner_floor10", "expert_floor10", "learner_fallback10"}
SCOPES = {"all": None, "learners": {"good", "moderate"}}
BIN = 10
REL_RANGE = (-40, 60)
LOCAL = 10
EDGES = np.arange(REL_RANGE[0], REL_RANGE[1] + BIN, BIN)
CENTERS = EDGES[:-1] + BIN / 2


def fig_dir(scheme: str) -> Path:
    d = OUT_DIR / "figures" / scheme / "learning"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(scheme: str) -> pd.DataFrame | None:
    path = OUT_DIR / f"092_lt_aligned_trials_{scheme}.parquet"
    if not path.exists():
        print(f"missing {path.name}")
        return None
    df = pd.read_parquet(path)
    print(f"{scheme}: {df.session_id.nunique()} sessions; decoded {df[df.skipped_reason.isna()].session_id.nunique()}")
    df = df[df.skipped_reason.isna() & df.rel_idx.notna()].copy()
    df["dec"] = df.p_correct - df.p_correct_null
    df["bin"] = pd.cut(df.rel_idx, EDGES, right=False, labels=False)
    df["flagged"] = df.lt_source.isin(FLAGGED)
    return df


def session_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, area, bin): class-balanced dec, per-class dec, behavior."""
    d = df.dropna(subset=["bin"])
    keys = ["session_id", "area_value", "bin"]
    per_class = d.groupby(keys + ["lick_flag"]).dec.mean().unstack("lick_flag")
    per_class.columns = ["dec_miss" if not c else "dec_hit" for c in per_class.columns]
    out = per_class.copy()
    out["dec_bal"] = out[["dec_hit", "dec_miss"]].mean(axis=1, skipna=False)
    beh = d.groupby(keys)[["behav_p_mean", "behav_p_chance"]].mean()
    meta = d.groupby(keys)[["reward_group", "flagged", "learning_category"]].first()
    return out.join(beh).join(meta).reset_index()


def mean_sem(tbl: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = tbl.dropna(subset=[col]).groupby("bin")[col]
    m = g.mean().reindex(range(len(CENTERS)))
    s = g.sem().reindex(range(len(CENTERS)))
    n = g.count().reindex(range(len(CENTERS))).fillna(0)
    return m.to_numpy(), s.to_numpy(), n.to_numpy()


def local_test(df: pd.DataFrame) -> pd.DataFrame:
    """Per (session, area): class-balanced dec in [-LOCAL,0) and [0,LOCAL)."""
    rows = []
    for (sid, area), g in df.groupby(["session_id", "area_value"]):
        vals = {}
        for name, lo, hi in (("before", -LOCAL, 0), ("after", 0, LOCAL)):
            w = g[(g.rel_idx >= lo) & (g.rel_idx < hi)]
            cm = w.groupby("lick_flag").dec.mean()
            vals[name] = cm.mean() if len(cm) == 2 else np.nan
        rows.append(dict(session_id=sid, area_value=area, reward_group=g.reward_group.iloc[0],
                         learning_category=g.learning_category.iloc[0], lt_source=g.lt_source.iloc[0], **vals))
    t = pd.DataFrame(rows)
    t["delta"] = t.after - t.before
    return t


def fmt_paired(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return f"n={ok.sum()}"
    return f"n={ok.sum()}, W p={wilcoxon(a[ok], b[ok]).pvalue:.2g}, t p={ttest_rel(a[ok], b[ok]).pvalue:.2g}"


def shade_local(ax):
    ax.axvspan(-LOCAL, 0, color="#dddddd", alpha=0.35, lw=0, zorder=0)
    ax.axvspan(0, LOCAL, color="#bbbbbb", alpha=0.35, lw=0, zorder=0)
    ax.axvline(0, color="#d62728", lw=1.3, zorder=1)


def whole_brain_figure(df: pd.DataFrame, scope: str):
    tbl = session_bin_table(df)
    loc = local_test(df)
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1, 1, 1, 0.55]))
    for c, cohorts in enumerate((["R+"], ["R-"], ["R+", "R-"])):
        for cohort in cohorts:
            col = COHORT_COLOR[cohort]
            t = tbl[tbl.reward_group == cohort]
            m, s, _ = mean_sem(t, "behav_p_mean")
            ax = axes[0, c]
            ax.plot(CENTERS, m, color=col, lw=2, label=f"{cohort} p(lick | whisker)")
            ax.fill_between(CENTERS, m - s, m + s, color=col, alpha=0.2, lw=0)
            mc, _, _ = mean_sem(t, "behav_p_chance")
            ax.plot(CENTERS, mc, color=col, lw=1.2, ls="--", label=f"{cohort} p_chance (FA)")

            ax = axes[1, c]
            m, s, _ = mean_sem(t, "dec_bal")
            ax.plot(CENTERS, m, color=col, lw=2, marker="o", ms=3, label=cohort)
            ax.fill_between(CENTERS, m - s, m + s, color=col, alpha=0.2, lw=0)

            ax = axes[2, c]
            for key, ls, lab in (("dec_hit", "-", "hit trials"), ("dec_miss", "--", "miss trials")):
                m, s, _ = mean_sem(t, key)
                ax.plot(CENTERS, m, color=col, lw=1.8, ls=ls, label=f"{cohort} {lab}")
                ax.fill_between(CENTERS, m - s, m + s, color=col, alpha=0.12, lw=0)

            ax = axes[3, c]
            n_crit = t[~t.flagged].dropna(subset=["dec_bal"]).groupby("bin").size().reindex(range(len(CENTERS))).fillna(0)
            n_flag = t[t.flagged].dropna(subset=["dec_bal"]).groupby("bin").size().reindex(range(len(CENTERS))).fillna(0)
            off = 0 if len(cohorts) == 1 else (-1 if cohort == "R+" else 1)
            w = BIN * (0.8 if len(cohorts) == 1 else 0.4)
            ax.bar(CENTERS + off * w / 2, n_crit, width=w, color=col, alpha=0.7, label=f"{cohort} criterion met")
            ax.bar(CENTERS + off * w / 2, n_flag, width=w, bottom=n_crit, color=col, alpha=0.3, hatch="///",
                   edgecolor=col, lw=0, label=f"{cohort} floor10/fallback10")

        title = " & ".join(cohorts)
        lines = []
        for cohort in cohorts:
            lt_ = loc[loc.reward_group == cohort]
            lines.append(f"{cohort} [-{LOCAL},0) vs [0,{LOCAL}): {fmt_paired(lt_.before.to_numpy(), lt_.after.to_numpy())}")
        if len(cohorts) == 2:
            a = loc[loc.reward_group == "R+"].delta.dropna().to_numpy()
            b = loc[loc.reward_group == "R-"].delta.dropna().to_numpy()
            if len(a) > 1 and len(b) > 1:
                lines.append(f"R+ vs R- local delta: MW p={mannwhitneyu(a, b).pvalue:.2g}, "
                             f"Welch p={ttest_ind(a, b, equal_var=False).pvalue:.2g}")
        axes[0, c].set_title(f"{title}: behavior aligned to learning_trial", fontsize=10)
        axes[1, c].set_title(f"{title}: decoding (class-balanced)\n" + "\n".join(lines), fontsize=8.5)
        axes[2, c].set_title(f"{title}: decoding, hit vs miss trials separately", fontsize=10)
        axes[3, c].set_title("sessions contributing per bin (both classes present)", fontsize=9)
        axes[0, c].set_ylim(-0.02, 1.02)
        for r in (1, 2):
            axes[r, c].axhline(0, color="#888888", lw=1, ls=":")
        for r in range(4):
            shade_local(axes[r, c])
            axes[r, c].spines[["top", "right"]].set_visible(False)
            axes[r, c].legend(frameon=False, fontsize=7, loc="upper left")
        axes[3, c].set_xlabel("whisker trial relative to learning_trial (0 = learning trial)")
    axes[0, 0].set_ylabel("P(lick)")
    axes[1, 0].set_ylabel("P(true class) - own null")
    axes[2, 0].set_ylabel("P(true class) - own null")
    axes[3, 0].set_ylabel("n sessions")
    ylims = [axes[r, c].get_ylim() for r in (1, 2) for c in range(3)]
    lo, hi = min(y[0] for y in ylims), max(y[1] for y in ylims)
    for r in (1, 2):
        for c in range(3):
            axes[r, c].set_ylim(lo, hi)
    fig.suptitle(f"Hit/miss decoding (whole brain, sensory 5-50ms) aligned to learning_trial, {BIN}-trial bins -- scope: {scope}\n"
                 f"one decoder per session fit on all trials (held-out outputs); shaded = +-{LOCAL}-trial windows used for the at-LT test",
                 fontsize=11)
    q034.savefig_retry(fig, fig_dir("whole_brain") / f"093_hitmiss_lt_aligned_whole_brain_sensory_{scope}.png", dpi=300,
                       bbox_inches="tight")
    plt.close(fig)
    loc.assign(scope=scope, scheme="whole_brain").to_csv(OUT_DIR / f"093_lt_local_test_whole_brain_{scope}.csv", index=False)
    print(f"  saved 093 whole_brain {scope}")


def area_figure(df: pd.DataFrame, scheme: str, scope: str):
    areas = q034.common_areas(df, scheme)
    colors = q034.get_area_color_map(areas)
    tbl = session_bin_table(df)
    loc = local_test(df)
    ncol = 5
    nrow = int(np.ceil(len(areas) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow), sharex=True, sharey=True, constrained_layout=True)
    for ax, area in zip(axes.flat, areas):
        lines = []
        for cohort in ("R+", "R-"):
            t = tbl[(tbl.reward_group == cohort) & (tbl.area_value == area)]
            m, s, n = mean_sem(t, "dec_bal")
            m[n < 3] = np.nan
            ax.plot(CENTERS, m, color=COHORT_COLOR[cohort], lw=1.6)
            ax.fill_between(CENTERS, m - s, m + s, color=COHORT_COLOR[cohort], alpha=0.18, lw=0)
            lt_ = loc[(loc.reward_group == cohort) & (loc.area_value == area)]
            lines.append(f"{cohort}: {fmt_paired(lt_.before.to_numpy(), lt_.after.to_numpy())}")
        shade_local(ax)
        ax.axhline(0, color="#888888", lw=1, ls=":")
        ax.set_title(f"{area}\n" + "\n".join(lines), fontsize=7.5, color=colors[area])
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(areas):]:
        ax.axis("off")
    fig.supxlabel("whisker trial relative to learning_trial")
    fig.supylabel("class-balanced P(true class) - own null")
    fig.suptitle(f"Hit/miss decoding aligned to learning_trial per area ({scheme}, sensory 5-50ms), R+ green / R- magenta; "
                 f"bins with <3 sessions hidden -- scope: {scope}", fontsize=10)
    q034.savefig_retry(fig, fig_dir(scheme) / f"093_hitmiss_lt_aligned_{scheme}_sensory_{scope}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    loc.assign(scope=scope, scheme=scheme).to_csv(OUT_DIR / f"093_lt_local_test_{scheme}_{scope}.csv", index=False)
    print(f"  saved 093 {scheme} {scope}")


def main():
    df = load("whole_brain")
    if df is not None:
        for scope, cats in SCOPES.items():
            whole_brain_figure(df if cats is None else df[df.learning_category.isin(cats)], scope)
    df = load("area_group")
    if df is not None:
        for scope, cats in SCOPES.items():
            area_figure(df if cats is None else df[df.learning_category.isin(cats)], "area_group", scope)


if __name__ == "__main__":
    main()
