"""Exploration (user 2026-11-17: "with the new GLM, also regress reaction time (multi-head GLM)"): the two heads of glm_v7
(glm_v7_run.py) side by side. Same design matrix (pre-stimulus predictors of each whisker trial) and the same contiguous
CV blocks; lick head = L2 logistic on lick / no lick (glm_v6), RT head = ridge on the reaction time of the hits.
Unit = session (one whisker day 0 per learner mouse). Stats: per cohort Wilcoxon signed-rank vs 0; R+ vs R-
Mann-Whitney U and Welch t (both reported); uncorrected.
  a  held-out performance: lick head pseudo-R2 and RT head gain in mean absolute error over predicting the training-fold mean RT (ms;
     R2 and RMSE in the stats csv), each with its baseline
     (running mean of the previous whisker licks / previous hit RTs)
  b  example session (Figure 1 GLM example mouse): RT of each hit and its held-out prediction (mean over repeats)
  c  weights per predictor (z-scored predictors): lick head (log-odds per SD) and RT head (ms per SD)
  d  drop-and-refit importance per predictor group: lick head delta LL per trial, RT head increase in MAE (ms)
Output: Figures/_explore/glm_rt_<date>.{pdf,png} + _stats.csv.
Run on haas (repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_glm_rt.py
"""

import datetime
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import compute  # noqa: E402
from common import CACHE_DIR, FIGURES_ROOT  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402
from plot import GLM_LABELS  # noqa: E402

GLM = CACHE_DIR / "glm_v7"
OUT = FIGURES_ROOT / "_explore"
COH = ("R+", "R-")
GROUP_ORDER = compute.GLM_ORDER[::-1] + ["all_movement"]


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def pstr(p):
    return f"{p:.2g}" if p >= 0.001 else "<0.001"


def cohort_tests(df, col):
    a, b = (df[df.cohort == c][col].dropna() for c in COH)
    out = {f"wilcoxon_p_{c}": stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan for c, x in zip(COH, (a, b))}
    out.update(mw_p=stats.mannwhitneyu(a, b).pvalue, welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue,
               **{f"mean_{c}": x.mean() for c, x in zip(COH, (a, b))}, **{f"n_{c}": len(x) for c, x in zip(COH, (a, b))})
    return out


def read_all(kind):
    rows = []
    for f in GLM.glob(f"*/*_{kind}"):
        d = pd.read_csv(f)
        d["session_id"] = f.name[: -len(kind) - 1].split("_", 1)[1]
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def strip(ax, df, col, x0, color, label=None):
    for i, c in enumerate(COH):
        v = df[df.cohort == c][col].dropna().to_numpy()
        xs = x0 + i * 0.38 + np.random.default_rng(i).uniform(-0.08, 0.08, len(v))
        ax.scatter(xs, v, s=2, color=COLORS[c] if color is None else color, lw=0, alpha=0.6)
        ax.errorbar(x0 + i * 0.38, v.mean(), stats.sem(v), fmt="o", ms=2.5, color="k", lw=0.7, capsize=0, zorder=4)


def panel_perf(axes, lick, rt, rows):
    for ax, df, cols, ylab, title in (
            (axes[0], lick, ("test_pseudo_r2", "baseline_test_pseudo_r2"), "Held-out pseudo-R²", "Lick head"),
            (axes[1], rt, ("rt_mae_gain_ms", "baseline_rt_mae_gain_ms"), "MAE gain over training mean (ms)",
             "RT head")):
        for j, (col, name) in enumerate(zip(cols, ("GLM", "Baseline"))):
            strip(ax, df, col, j, None if j == 0 else "0.6")
            t = cohort_tests(df, col)
            rows.append({"panel": "a", "head": title, "metric": col, **t})
            for i, c in enumerate(COH):
                ax.text(j + i * 0.38, 1.0, stars(t[f"wilcoxon_p_{c}"]), transform=ax.get_xaxis_transform(),
                        ha="center", fontsize=4.5, color=COLORS[c])
        d = df.assign(delta=df[cols[0]] - df[cols[1]])
        t = cohort_tests(d, "delta")
        rows.append({"panel": "a", "head": title, "metric": "GLM minus baseline", **t})
        ax.axhline(0, color="0.5", lw=0.4, ls=":")
        ax.set_xticks([0.19, 1.19], ["GLM", "Baseline"])
        ax.set_ylabel(ylab)
        lo, hi = np.nanpercentile(df[list(cols)].to_numpy(), [1, 99])
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.1 * (hi - lo))
        ax.set_title(f"{title}\nGLM - baseline: R+ p {pstr(t['wilcoxon_p_R+'])}, R- p {pstr(t['wilcoxon_p_R-'])}\n"
                     f"R+ vs R- (GLM): MW p {pstr(cohort_tests(df, cols[0])['mw_p'])}, "
                     f"Welch p {pstr(cohort_tests(df, cols[0])['welch_p'])}", fontsize=4.8, pad=8)


def panel_example(ax, rt):
    mouse = compute.GLM_EXAMPLE_MOUSE
    r = rt[rt.mouse_id == mouse]
    if not len(r):
        r = rt.iloc[[(rt.rt_r2_vs_train - rt.rt_r2_vs_train.median()).abs().argmin()]]
        mouse = r.mouse_id.iloc[0]
    r = r.iloc[0]
    p = pd.read_parquet(GLM / mouse / f"{mouse}_{r.session_id}_rt_predictions.parquet")
    p = p[np.isfinite(p.rt_ms)]
    m = p.groupby("trial").agg(rt=("rt_ms", "first"), pred=("rt_pred_ms", "mean"), lo=("rt_pred_ms", "min"),
                               hi=("rt_pred_ms", "max"), base=("rt_baseline_ms", "mean")).reset_index()
    x = np.arange(1, len(m) + 1)
    ax.plot(x, m.rt, "o", ms=1.8, mew=0, color=COLORS[r.cohort], label="Observed RT")
    ax.fill_between(x, m.lo, m.hi, color="0.85", lw=0)
    ax.plot(x, m.pred, color="k", lw=0.7, label="GLM (held out)")
    ax.plot(x, m.base, color="0.55", lw=0.5, ls="--", label="Baseline")
    ax.set_xlabel("Whisker hit")
    ax.set_ylabel("Reaction time (ms)")
    ax.set_title(f"{mouse} ({r.cohort}): MAE {r.rt_mae_ms:.0f} ms (training mean {r.train_mean_mae_ms:.0f} ms), "
                 f"R² {r.rt_r2_vs_train:.2f}, r = {r.rt_corr:.2f}",
                 fontsize=4.8)
    ax.legend(frameon=False, fontsize=4, loc="upper right", handlelength=1.2)


def panel_bars(ax, df, value, order, xlabel, title, rows, panel):
    for i, c in enumerate(COH):
        g = df[df.cohort == c].groupby("feature")[value]
        mu, se = g.mean().reindex(order), g.sem().reindex(order)
        y = np.arange(len(order)) + (0.18 if c == "R+" else -0.18)
        ax.barh(y, mu, 0.34, xerr=se, color=COLORS[c], error_kw={"lw": 0.5}, label=c)
        for k, f in enumerate(order):
            v = df[(df.cohort == c) & (df.feature == f)][value].dropna()
            if len(v) > 5:
                pv = stats.wilcoxon(v).pvalue
                rows.append({"panel": panel, "head": title, "metric": f"{value} {f}", f"wilcoxon_p_{c}": pv,
                             f"mean_{c}": v.mean(), f"n_{c}": len(v)})
                if pv < 0.05:
                    xe = mu.iloc[k] + np.sign(mu.iloc[k]) * (se.iloc[k] + 0.0) if np.isfinite(mu.iloc[k]) else 0
                    ax.text(xe, y[k], " " + stars(pv) + " ", va="center", fontsize=3.8, color=COLORS[c],
                            ha="left" if mu.iloc[k] >= 0 else "right")
    for f in order:
        d = df[df.feature == f]
        a, b = (d[d.cohort == c][value].dropna() for c in COH)
        if len(a) > 5 and len(b) > 5:
            rows.append({"panel": panel, "head": title, "metric": f"{value} {f} R+ vs R-",
                         "mw_p": stats.mannwhitneyu(a, b).pvalue, "welch_p": stats.ttest_ind(a, b, equal_var=False).pvalue})
    ax.axvline(0, color="0.4", lw=0.4)
    ax.set_yticks(range(len(order)), [GLM_LABELS.get(f, f) for f in order])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=5.5)
    lo, hi = ax.get_xlim()
    ax.set_xlim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))


def main():
    set_nature_rc()
    rt = pd.read_csv(GLM / "rt_summary.csv")
    lick = pd.read_csv(GLM / "all_sessions_summary.csv")
    coh = pd.concat([lick[["session_id", "cohort"]], rt[["session_id", "cohort"]]]).drop_duplicates("session_id")
    cl = read_all("coefficients.csv")
    cl = cl[~cl.session_id.str.endswith("_rt")].merge(coh, on="session_id")
    cl["feature"] = cl.feature.map(compute._glm_feature)
    cr = read_all("rt_coefficients.csv").merge(coh, on="session_id")
    cr["feature"] = cr.feature.map(compute._glm_feature)
    nl = read_all("nested_model_comparison.csv").merge(coh, on="session_id")
    nl = nl[nl.model != "Full Model"].assign(feature=lambda d: d.model.map(compute._glm_feature))
    nr = read_all("rt_nested.csv").merge(coh, on="session_id")
    nr["feature"] = nr.model.map(compute._glm_feature)
    feats = [f for f in compute.GLM_ORDER[::-1] if f in set(cl.feature)]
    groups = [f for f in GROUP_ORDER if f in set(nr.feature)]

    rows = [{"panel": "a", "head": "RT head", "metric": m, **cohort_tests(rt, m)} for m in
            ("rt_r2_vs_train", "baseline_rt_r2_vs_train", "rt_mae_ms", "train_mean_mae_ms", "rt_rmse_gain_ms",
             "delta_rt_mae_ms_vs_baseline", "rt_corr")]
    fig = plt.figure(figsize=(183 * MM, 125 * MM))
    gs = fig.add_gridspec(2, 4, left=0.08, right=0.98, top=0.86, bottom=0.08, wspace=0.75, hspace=0.6,
                          width_ratios=[1, 1, 1.6, 0.01])
    ax_a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    panel_perf(ax_a, lick, rt, rows)
    panel_example(fig.add_subplot(gs[0, 2]), rt)
    gs2 = gs[1, :].subgridspec(1, 4, wspace=0.9)
    panel_bars(fig.add_subplot(gs2[0]), cl, "coefficient", feats, "Weight (log-odds / SD)", "Lick head weights", rows, "c")
    panel_bars(fig.add_subplot(gs2[1]), cr, "coefficient_ms_per_sd", feats, "Weight (ms / SD)", "RT head weights",
               rows, "c")
    panel_bars(fig.add_subplot(gs2[2]), nl, "delta_test_ll_per_trial", groups, "Δ LL per trial (nats)",
               "Lick head: drop predictor", rows, "d")
    ax = fig.add_subplot(gs2[3])
    panel_bars(ax, nr, "delta_rt_mae_ms", groups, "Δ MAE (ms)", "RT head: drop predictor", rows, "d")
    ax.legend(frameon=False, fontsize=4.5, loc="lower right")
    for a, l in zip([ax_a[0], fig.axes[2], fig.axes[3]], "abc"):
        a.text(-0.35, 1.25, l, transform=a.transAxes, fontsize=8, fontweight="bold")
    fig.axes[5].text(-0.6, 1.08, "d", transform=fig.axes[5].transAxes, fontsize=8, fontweight="bold")
    n = rt.groupby("cohort").size()
    fig.suptitle("Multi-head behaviour GLM (glm_v7): same pre-stimulus predictors and CV blocks; lick head = logistic on "
                 "lick / no lick (all whisker trials), RT head = ridge on reaction time (hits, RT 0-1 s)\n"
                 f"Whisker day 0, learners; RT head: {n.get('R+', 0)} R+ / {n.get('R-', 0)} R- sessions with >= 20 hits; "
                 "stars: Wilcoxon vs 0 per cohort (uncorrected); bars mean ± s.e.m. over sessions",
                 fontsize=5.5)
    stem = OUT / f"glm_rt_{datetime.date.today()}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    pd.DataFrame(rows).to_csv(f"{stem}_stats.csv", index=False)
    print(stem)
    print(pd.DataFrame(rows)[lambda d: d.panel == "a"].round(4).to_string())


if __name__ == "__main__":
    main()
