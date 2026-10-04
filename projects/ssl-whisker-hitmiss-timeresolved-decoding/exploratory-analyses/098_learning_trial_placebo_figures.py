"""Figures + stats for `097_learning_trial_placebo_split.py`: is the
learning_trial special? (user request 2026-09-24).

Per (session, area, window):
  real delta     = matched acc_post - acc_pre at the stored learning_trial
  placebo deltas = same at every other valid split with |k - LT| >= EXCLUDE_NEAR
  pct            = fraction of placebo deltas below the real one (ties 1/2).
                   Under "LT is not special" pct ~ Uniform(0,1), mean 0.5;
                   pct < 0.5 = the real split gives a bigger DROP than
                   arbitrary splits, pct > 0.5 = a bigger RISE.
  pct_sizematched = same, placebos restricted to splits whose matched trial
                   count (n_hit* + n_miss*) is within x0.5-x2 of the real
                   split's (small pre epochs give noisier deltas).
  pct_abs        = fraction of placebos whose |delta - median placebo| is
                   below the real one's: "is the change at LT unusually
                   LARGE in either direction" (mean 0.5 under the null).
Group: per cohort x window x scope, Wilcoxon signed-rank + one-sample t of
pct - 0.5 (mandatory non-param + param pair); R+ vs R- on pct with MWU +
Welch. One session per mouse at the learning stage (checked in main).

Figures (figures/whole_brain/learning/):
  098_hitmiss_lt_placebo_whole_brain_summary_<scope>.png
     rows = window; cols = R+ delta profile, R- delta profile (delta vs split
     position relative to LT, mean +- SEM across sessions, real LT = star),
     per-session pct strip (both cohorts, lt_source markers) with tests.
  098_hitmiss_lt_placebo_whole_brain_sessions_<window>.png
     every session: delta vs split position, real LT starred.
Stats: 098_learning_trial_placebo_stats.csv, per-session: 098_learning_trial_placebo_persession.csv
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu, ttest_1samp, ttest_ind, wilcoxon

OUT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("q034", OUT_DIR / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q034)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
LT_MARKER = {"learner": "o", "expert": "s", "learner_floor10": "^", "expert_floor10": "^", "learner_fallback10": "X"}
LT_LABEL = {"o": "criterion met (learner)", "s": "criterion met (expert rule)", "^": "clamped to 10 (floor10)",
            "X": "R- fallback: hardcoded 10"}
SCOPES = {"all": None, "learners": {"good", "moderate"}}
WINDOW_LABELS = {"sensory": "sensory 5-50ms", "baseline": "baseline -200..-10ms", "sensory_minus_base": "sensory minus baseline"}
EXCLUDE_NEAR = 5
# Input variant (2026-09-25): e.g. "_lt-lt_lenient_clean_nodisengagedrop" for the re-run with the new
# learning trials and disengaged trials kept; "" = original stored-LT run.
VARIANT = sys.argv[1] if len(sys.argv) > 1 else ""
TAG = VARIANT.replace("_lt-", "_").replace("_nodisengagedrop", "_nodrop")
PROFILE_BIN = 9
PROFILE_RANGE = (-72, 108)


def per_session(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sid, area, win), g in df.groupby(["session_id", "area_value", "window"]):
        real = g[g.is_real]
        if real.empty:
            continue
        real = real.iloc[0]
        pl = g[(~g.is_real) & (g.rel_k.abs() >= EXCLUDE_NEAR)]
        if len(pl) < 3:
            continue
        d, rd = pl.delta.to_numpy(), real.delta
        n_real = real.matched_n_hit + real.matched_n_miss
        n_pl = (pl.matched_n_hit + pl.matched_n_miss).to_numpy()
        sm = d[(n_pl >= 0.5 * n_real) & (n_pl <= 2 * n_real)]
        med = np.median(d)
        rows.append(dict(session_id=sid, area_value=area, window=win, reward_group=real.reward_group,
                         learning_category=real.learning_category, lt_source=real.lt_source, mouse_id=real.mouse_id,
                         learning_trial=real.learning_trial, real_delta=rd, n_placebo=len(d),
                         placebo_mean=d.mean(), placebo_sd=d.std(ddof=1),
                         z=(rd - d.mean()) / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan,
                         pct=(np.sum(d < rd) + 0.5 * np.sum(d == rd)) / len(d),
                         pct_sizematched=((np.sum(sm < rd) + 0.5 * np.sum(sm == rd)) / len(sm)) if len(sm) >= 3 else np.nan,
                         n_placebo_sizematched=len(sm),
                         pct_abs=np.mean(np.abs(d - med) < abs(rd - med))))
    return pd.DataFrame(rows)


def vs_half(x: np.ndarray) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 3:
        return np.nan, np.nan
    return float(wilcoxon(x - 0.5).pvalue), float(ttest_1samp(x, 0.5).pvalue)


def group_stats(ps: pd.DataFrame, scope: str) -> list[dict]:
    out = []
    for win in WINDOW_LABELS:
        w = ps[ps.window == win]
        for col in ("pct", "pct_sizematched", "pct_abs"):
            for cohort in ("R+", "R-"):
                x = w[w.reward_group == cohort][col].to_numpy()
                pw, pt = vs_half(x)
                out.append(dict(scope=scope, window=win, stat=col, group=cohort, n=int(np.sum(~np.isnan(x))),
                                mean=np.nanmean(x), p_wilcoxon_vs_0p5=pw, p_t_vs_0p5=pt))
            a = w[w.reward_group == "R+"][col].dropna().to_numpy()
            b = w[w.reward_group == "R-"][col].dropna().to_numpy()
            if len(a) > 1 and len(b) > 1:
                out.append(dict(scope=scope, window=win, stat=col, group="R+ vs R-", n=len(a) + len(b),
                                p_mannwhitney=float(mannwhitneyu(a, b).pvalue),
                                p_welch=float(ttest_ind(a, b, equal_var=False).pvalue)))
    return out


def summary_figure(df: pd.DataFrame, ps: pd.DataFrame, scope: str):
    edges = np.arange(PROFILE_RANGE[0], PROFILE_RANGE[1] + PROFILE_BIN, PROFILE_BIN)
    centers = edges[:-1] + PROFILE_BIN / 2
    fig, axes = plt.subplots(len(WINDOW_LABELS), 3, figsize=(14, 4.2 * len(WINDOW_LABELS)), constrained_layout=True)
    for r, (win, wlabel) in enumerate(WINDOW_LABELS.items()):
        dw = df[df.window == win]
        for c, cohort in enumerate(("R+", "R-")):
            ax = axes[r, c]
            col = COHORT_COLOR[cohort]
            dc = dw[(dw.reward_group == cohort) & ~dw.is_real].copy()
            dc["bin"] = pd.cut(dc.rel_k, edges, labels=False)
            per = dc.dropna(subset=["bin"]).groupby(["session_id", "bin"]).delta.mean().reset_index()
            g = per.groupby("bin").delta
            m, s, n = (g.mean().reindex(range(len(centers))), g.sem().reindex(range(len(centers))),
                       g.count().reindex(range(len(centers))).fillna(0))
            m[n < 3] = np.nan
            ax.plot(centers, m, color=col, lw=2, marker="o", ms=3, label="placebo splits (binned by position)")
            ax.fill_between(centers, m - s, m + s, color=col, alpha=0.2, lw=0)
            real = dw[(dw.reward_group == cohort) & dw.is_real].delta
            ax.errorbar(0, real.mean(), yerr=real.sem(), color="k", marker="*", ms=14, capsize=3, zorder=5,
                        label=f"real learning_trial (n={len(real)})")
            ax.axvline(0, color="#d62728", lw=1)
            ax.axvspan(-EXCLUDE_NEAR, EXCLUDE_NEAR, color="#eeeeee", zorder=0)
            ax.axhline(0, color="#888888", lw=1, ls=":")
            ax.set_xlim(*PROFILE_RANGE)
            ax.set_title(f"{cohort} | {wlabel}: pre->post change vs split position", fontsize=9.5)
            ax.set_xlabel("split position relative to learning_trial (whisker trials)", fontsize=8.5)
            ax.set_ylabel("delta matched bal. acc. (post - pre)", fontsize=8.5)
            ax.legend(frameon=False, fontsize=7, loc="upper right")
            ax.spines[["top", "right"]].set_visible(False)
        ax = axes[r, 2]
        pw = ps[ps.window == win]
        rng = np.random.default_rng(0)
        lines = []
        for k, cohort in enumerate(("R+", "R-")):
            sub = pw[pw.reward_group == cohort]
            x = k + rng.uniform(-0.15, 0.15, len(sub))
            for mk in sub.lt_source.map(LT_MARKER).fillna("o").unique():
                mm = (sub.lt_source.map(LT_MARKER).fillna("o") == mk).to_numpy()
                ax.scatter(x[mm], sub.pct.to_numpy()[mm], marker=mk, s=20, color=COHORT_COLOR[cohort], alpha=0.65, lw=0)
            ax.errorbar(k + 0.32, sub.pct.mean(), yerr=sub.pct.sem(), color="k", marker="o", capsize=3)
            pwil, pt = vs_half(sub.pct.to_numpy())
            pwil_s, pt_s = vs_half(sub.pct_sizematched.to_numpy())
            lines.append(f"{cohort}: mean {sub.pct.mean():.2f}, vs 0.5 W p={pwil:.2g}, t p={pt:.2g} "
                         f"| size-matched {sub.pct_sizematched.mean():.2f} (W p={pwil_s:.2g})")
        a, b = pw[pw.reward_group == "R+"].pct.dropna(), pw[pw.reward_group == "R-"].pct.dropna()
        if len(a) > 1 and len(b) > 1:
            lines.append(f"R+ vs R-: MW p={mannwhitneyu(a, b).pvalue:.2g}, Welch p={ttest_ind(a, b, equal_var=False).pvalue:.2g}")
        ax.axhline(0.5, color="#888888", lw=1, ls=":")
        ax.set_xticks([0, 1], ["R+", "R-"])
        ax.set_xlim(-0.5, 1.6)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("percentile of real delta among placebo splits", fontsize=8.5)
        ax.set_title(f"{wlabel}: is the learning trial special?\n" + "\n".join(lines), fontsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], marker=m, ls="", color="#444444", markersize=6, label=l) for m, l in LT_LABEL.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Placebo-split test of the learning_trial{TAG} -- hit/miss decoding, whole brain, learning stage -- scope: {scope}\n"
                 f"percentile < 0.5: real split gives a larger DROP than arbitrary splits; placebos within +-{EXCLUDE_NEAR} "
                 f"trials of the real one excluded (grey band)", fontsize=11)
    q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"098_hitmiss_lt_placebo{TAG}_whole_brain_summary_{scope}.png",
                       dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved summary {scope}")


def sessions_figure(df: pd.DataFrame, ps: pd.DataFrame, win: str):
    dw = df[df.window == win]
    sids = (ps[ps.window == win].sort_values(["reward_group", "pct"]).session_id.tolist())
    ncol = 8
    nrow = int(np.ceil(len(sids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.0 * nrow), sharey=True, constrained_layout=True)
    for ax, sid in zip(axes.flat, sids):
        g = dw[dw.session_id == sid].sort_values("split_k")
        col = COHORT_COLOR[g.reward_group.iloc[0]]
        pl = g[~g.is_real]
        ax.plot(pl.split_k, pl.delta, color=col, lw=1, marker=".", ms=3)
        real = g[g.is_real].iloc[0]
        ax.plot(real.split_k, real.delta, marker="*", ms=11, color="k", zorder=5)
        ax.axvline(real.split_k, color="#d62728", lw=0.8)
        ax.axhline(0, color="#888888", lw=0.6, ls=":")
        p = ps[(ps.session_id == sid) & (ps.window == win)].iloc[0]
        ax.set_title(f"{sid[:5]} {real.reward_group} LT={int(real.learning_trial)} [{p.lt_source.replace('learner_', '')}]\n"
                     f"pct={p.pct:.2f} z={p.z:+.1f}", fontsize=6.5)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(sids):]:
        ax.axis("off")
    fig.suptitle(f"Every session: pre->post change in matched hit/miss decoding vs split position ({WINDOW_LABELS[win]}, "
                 f"whole brain); star = real learning_trial; sorted by cohort then percentile", fontsize=10)
    q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"098_hitmiss_lt_placebo{TAG}_whole_brain_sessions_{win}.png",
                       dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved sessions {win}")


def main():
    df = pd.read_parquet(OUT_DIR / f"097_lt_placebo{VARIANT}_whole_brain.parquet")
    print(f"{df.session_id.nunique()} sessions; skipped:\n{df[df.skipped_reason.notna()].skipped_reason.str.slice(0, 45).value_counts().to_string()}")
    df = df[df.skipped_reason.isna()].copy()
    df["is_real"] = df["is_real"].astype(bool)  # object dtype after concat with skip rows -> ~ gave ints
    ps = per_session(df)
    assert ps.groupby(["window", "mouse_id"]).session_id.nunique().max() == 1, "a mouse contributes >1 learning session"
    ps.to_csv(OUT_DIR / f"098_learning_trial_placebo{TAG}_persession.csv", index=False)
    stats = []
    for scope, cats in SCOPES.items():
        sel = ps if cats is None else ps[ps.learning_category.isin(cats)]
        dsel = df if cats is None else df[df.learning_category.isin(cats)]
        summary_figure(dsel, sel, scope)
        stats += group_stats(sel, scope)
    for win in WINDOW_LABELS:
        sessions_figure(df, ps, win)
    st = pd.DataFrame(stats)
    st.to_csv(OUT_DIR / f"098_learning_trial_placebo{TAG}_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(4).to_string())


if __name__ == "__main__":
    main()
