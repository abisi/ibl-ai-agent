"""143 -- Publication figures for the learning-trial split of hit vs miss decoding, three windows (user 2026-10-05: "Do the same
but with L5 change point: this removes some mice and keep gradual learners. Add a figure that compares the effect of gradual
learners vs non-gradual (median point like here). Duplicate analyses for 5-35, 5-50 and 5-100 windows. Show all in one figure").
Figure 1  143_lt_split_<variant>_windows_<scope>: one row per window (5-35, 5-50, 5-100 ms after the whisker stimulus), the panels
          of the 138 COSYNE figure for the split <variant> (default L5 whisker CP: Bayesian change point of the whisker hit
          sequence; sessions without a change point drop out):
            before vs after (accuracy - linear-shift null; 118) | real vs placebo change (122 every-whisker-trial placebo, 126) |
            excess over placebo, R+ vs R- (+ cohort-label permutation) | change vs split position relative to the split.
Figure 2  143_step_vs_gradual_<scope>: sessions split at their L5 change point ("step": L5 exists) vs learners without an L5 change
          point split at the session midpoint ("gradual": learning_category good / moderate, no L5) and non-learners without an
          L5 change point (midpoint, reference); per window: change above the shift null (after - before) and excess over
          placebo, per group x cohort, mean +- s.e.m. over sessions; R+ vs R- per group and step vs gradual per cohort
          (Mann-Whitney | Welch), each group vs 0 (Wilcoxon | t).
Inputs per window: 5-35 ms: 118 tag _A1v2_hm35, 122/126 tag _step1_hm35; 5-50 / 5-100 ms: 118 _A1v2, 122/126 _step1_pl100.
Units: sessions (one per mouse, learning stage). Scopes: all | learners (argument). Uncorrected. Style: skills/ssl-figure-style.
Outputs: figures/publication/143_*.{png,pdf,svg}, 143_stats_<variant>_<scope>.csv, 143_step_vs_gradual_stats_<scope>.csv
Run (haas, repo root): python .../143_lt_split_windows_figures.py [all|learners] [variant]
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

EA = Path(__file__).resolve().parent
REPO = EA.parents[2]
FIGDIR = EA / "figures" / "publication"
COL = {"R+": "#00B400", "R-": "#C800C8"}
COH = ("R+", "R-")
WINS = [("5-35ms", "_A1v2_hm35", "_step1_hm35", "5-35 ms"), ("5-50ms", "_A1v2", "_step1_pl100", "5-50 ms"),
        ("5-100ms", "_A1v2", "_step1_pl100", "5-100 ms")]
N_PERM, BIN, REL_RANGE = 20000, 10, (-40, 60)
GROUPS = [("step", "step\n(L5 CP split)"), ("gradual", "gradual learners\n(midpoint)"), ("nonlearner", "non-learners\n(midpoint)")]


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.titlepad": 4})


def pnum(p):
    return "n/a" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def sem(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def paired(a, b):
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan, np.nan, len(a)
    return (stats.wilcoxon(a, b).pvalue if len(a) > 5 else np.nan), stats.ttest_rel(a, b).pvalue, len(a)


def unpaired(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if min(len(x), len(y)) < 3:
        return np.nan, np.nan
    return stats.mannwhitneyu(x, y).pvalue, stats.ttest_ind(x, y, equal_var=False).pvalue


def one_sample(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, len(x)
    return (stats.wilcoxon(x).pvalue if len(x) > 5 else np.nan), stats.ttest_1samp(x, 0).pvalue, len(x)


def perm(x, y, rng):
    x, y = np.asarray(x, float), np.asarray(y, float); x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    v = np.r_[x, y]; lab = np.r_[np.ones(len(x), bool), np.zeros(len(y), bool)]
    obs = x.mean() - y.mean()
    null = np.array([(lambda p: v[p].mean() - v[~p].mean())(rng.permutation(lab)) for _ in range(N_PERM)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)


def bracket(ax, x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y - 0.012, y, y, y - 0.012], color=color, lw=0.6, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.01, text, color=color, ha="center", va="bottom", fontsize=4.8, transform=ax.get_xaxis_transform())


def mouse_of(d):
    return d.session_id.str.split("_").str[0]


def learners():
    ref = pd.read_parquet(REPO / "reports/ssl_analysis/derived/mouse_reference.parquet")
    return set(ref[ref.learning_category.isin(["good", "moderate"])].subject_id)


def load(win, t118, t122, scope):
    A = pd.read_parquet(EA / f"118_lt_definitions_split_whole_brain{t118}.parquet")
    A = A[(A.decoding == "hitmiss") & (A.window == win)].dropna(subset=["corr_pre_matched", "corr_post_matched"])
    S = pd.read_csv(EA / f"126_lt_variants_per_session{t122}.csv")
    S = S[(S.decoding == "hitmiss") & (S.window == win)].dropna(subset=["delta", "placebo_mean"])
    P = pd.read_parquet(EA / f"122_lt_placebo_whole_brain{t122}.parquet")
    P = P[(P.decoding == "hitmiss") & (P.window == win)].dropna(subset=["delta_matched"])
    if scope == "learners":
        L = learners()
        A, S, P = (d[mouse_of(d).isin(L)] for d in (A, S, P))
    return A, S, P


def mean_pair(ax, xp, a, b, label_open_filled=None):
    for c in COH:
        m = [np.nanmean(a[c]), np.nanmean(b[c])]; e = [sem(a[c]), sem(b[c])]
        ax.plot(xp[c], m, color=COL[c], lw=1.1)
        for x_, mm, ee, filled in zip(xp[c], m, e, (False, True)):
            ax.errorbar(x_, mm, ee, fmt="o", ms=3.6, color=COL[c], mfc=COL[c] if filled else "white", mew=0.9, lw=1.0, capsize=0)
    ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))


def figure_windows(scope, variant, rng):
    rows, fig = [], plt.figure(figsize=(7.4, 5.6))
    lefts, widths = [0.09, 0.32, 0.55, 0.76], [0.15, 0.15, 0.08, 0.21]
    tops = [0.71, 0.405, 0.10]
    all_axes = []
    for ri, (win, t118, t122, wl) in enumerate(WINS):
        A, S, P = load(win, t118, t122, scope)
        A, Sv = A[A.definition == variant], S[S.variant == variant]
        axs = [fig.add_axes([l, tops[ri], w, 0.17]) for l, w in zip(lefts, widths)]
        all_axes.append(axs)
        pre = {c: A[A.reward_group == c].corr_pre_matched.to_numpy() for c in COH}
        post = {c: A[A.reward_group == c].corr_post_matched.to_numpy() for c in COH}
        real = {c: Sv[Sv.reward_group == c].delta.to_numpy() for c in COH}
        plac = {c: Sv[Sv.reward_group == c].placebo_mean.to_numpy() for c in COH}
        exc = {c: real[c] - plac[c] for c in COH}
        xp = {"R+": (0, 1), "R-": (2.4, 3.4)}
        # before / after
        ax = axs[0]; mean_pair(ax, xp, pre, post)
        ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["before", "after", "before", "after"]); ax.set_xlim(-0.5, 3.9)
        ax.set_ylabel(f"{wl}\nhit vs miss decoding\n(accuracy - shift null)")
        for c in COH:
            pw, pt, n = paired(post[c], pre[c])
            rows.append(dict(window=win, panel="before/after", test="paired Wilcoxon | paired t", group=c, n=n,
                             mean_a=np.nanmean(pre[c]), mean_b=np.nanmean(post[c]), p_nonparam=pw, p_param=pt))
            bracket(ax, *xp[c], 1.03, f"{pnum(pw)} | {pnum(pt)}", COL[c])
        chg = {c: post[c] - pre[c] for c in COH}
        mw, we = unpaired(chg["R+"], chg["R-"])
        rows.append(dict(window=win, panel="before/after", test="change R+ vs R- (Mann-Whitney | Welch)", group="R+ vs R-",
                         n=len(chg["R+"]) + len(chg["R-"]), mean_a=np.nanmean(chg["R+"]), mean_b=np.nanmean(chg["R-"]), p_nonparam=mw, p_param=we))
        bracket(ax, 0.5, 2.9, 1.2, f"change: {pnum(mw)} | {pnum(we)}", "k")
        # real vs placebo
        ax = axs[1]; xc = {"R+": (0, 1.3), "R-": (2.8, 4.1)}; mean_pair(ax, xc, plac, real)
        ax.set_xticks([0.65, 3.45]); ax.set_xticklabels([f"R+ (n={len(real['R+'])})", f"R- (n={len(real['R-'])})"]); ax.set_xlim(-0.6, 4.7)
        ax.set_ylabel("change in decoding\n(after - before)")
        for c in COH:
            pw, pt, n = paired(real[c], plac[c])
            rows.append(dict(window=win, panel="real vs placebo", test="paired Wilcoxon | paired t", group=c, n=n,
                             mean_a=np.nanmean(plac[c]), mean_b=np.nanmean(real[c]), p_nonparam=pw, p_param=pt))
            bracket(ax, *xc[c], 1.03, f"{pnum(pw)} | {pnum(pt)}", COL[c])
        # excess
        ax = axs[2]
        for i, c in enumerate(COH):
            ax.bar(i, np.nanmean(exc[c]), 0.6, color=COL[c], alpha=0.35, lw=0, edgecolor="none")
            ax.errorbar(i, np.nanmean(exc[c]), sem(exc[c]), fmt="o", ms=3.6, color=COL[c], lw=1.0, capsize=0)
        ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R-"]); ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel("change beyond placebo")
        mw, we = unpaired(exc["R+"], exc["R-"]); obs, pp = perm(exc["R+"], exc["R-"], rng)
        for c in COH:
            pw, pt, n = one_sample(exc[c])
            rows.append(dict(window=win, panel="excess", test="vs 0 (Wilcoxon | one-sample t)", group=c, n=n, mean_a=np.nanmean(exc[c]),
                             mean_b=np.nan, p_nonparam=pw, p_param=pt))
        rows.append(dict(window=win, panel="excess", test="R+ vs R- (Mann-Whitney | Welch)", group="R+ vs R-", n=len(exc["R+"]) + len(exc["R-"]),
                         mean_a=np.nanmean(exc["R+"]), mean_b=np.nanmean(exc["R-"]), p_nonparam=mw, p_param=we))
        rows.append(dict(window=win, panel="excess", test=f"R+ - R- cohort-label permutation ({N_PERM})", group="R+ vs R-",
                         n=len(exc["R+"]) + len(exc["R-"]), mean_a=obs, mean_b=np.nan, p_nonparam=pp, p_param=np.nan))
        bracket(ax, 0, 1, 1.03, f"{pnum(mw)} | {pnum(we)}", "k"); bracket(ax, 0, 1, 1.2, f"perm. {pnum(pp)}", "k")
        # profile
        ax = axs[3]
        lt = Sv.set_index("session_id")["lt"]
        Q = P[P.session_id.isin(lt.index)].copy()
        Q["rel"] = Q.split_k - Q.session_id.map(lt)
        Q = Q[(Q.rel >= REL_RANGE[0]) & (Q.rel < REL_RANGE[1])]
        Q["bin"] = np.floor(Q.rel / BIN) * BIN + BIN / 2
        prof = Q.groupby(["reward_group", "session_id", "bin"]).delta_matched.mean().reset_index()
        for c in COH:
            g = prof[prof.reward_group == c].groupby("bin").delta_matched
            m, e, n = g.mean(), g.apply(sem), g.size()
            okb = n >= 5
            ax.fill_between(m.index[okb], (m - e)[okb], (m + e)[okb], color=COL[c], alpha=0.25, lw=0, edgecolor="none")
            ax.plot(m.index[okb], m[okb], color=COL[c], lw=1.0)
        ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
        ax.set_ylabel("change in decoding\n(after - before)")
        if ri == len(WINS) - 1:
            ax.set_xlabel("split position (whisker trials\nfrom the change point)")
    titles = ["At the change point", "Real vs placebo splits", "Beyond placebo", "Every split of the session"]
    for ax, t in zip(all_axes[0], titles):
        ax.set_title(t, y=1.42)
    W, Hh = fig.get_size_inches()
    for ri, axs in enumerate(all_axes):
        for ax, l in zip(axs, "abcd"):
            fig.text(ax.get_position().x0 - 0.42 / W, ax.get_position().y1 + 0.3 / Hh, f"{l}{ri + 1}", fontsize=8.5, weight="bold")
    return fig, pd.DataFrame(rows)


def figure_groups(scope, rng):
    rows = []
    fig, axes = plt.subplots(len(WINS), 2, figsize=(5.2, 5.6), squeeze=False)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.9, bottom=0.12, hspace=0.75, wspace=0.42)
    L = learners()
    for ri, (win, t118, t122, wl) in enumerate(WINS):
        A, S, P = load(win, t118, t122, "all")
        step_sids = set(S[S.variant == "L5 whisker CP"].session_id) | set(A[A.definition == "L5 whisker CP"].session_id)
        def grp(d, col):
            m = mouse_of(d)
            out = {}
            out["step"] = d[(d[col] == "L5 whisker CP")]
            nostep = d[(d[col] == "half") & ~d.session_id.isin(step_sids)]
            out["gradual"] = nostep[mouse_of(nostep).isin(L)]
            out["nonlearner"] = nostep[~mouse_of(nostep).isin(L)]
            if scope == "learners":
                out["step"] = out["step"][mouse_of(out["step"]).isin(L)]
            return out
        gA, gS = grp(A, "definition"), grp(S, "variant")
        for ci, (key, label) in enumerate((("chg", "change above shift null\n(after - before)"), ("exc", "change beyond placebo"))):
            ax = axes[ri, ci]
            vals = {}
            for gi, (g, gl) in enumerate(GROUPS):
                for k, c in enumerate(COH):
                    if key == "chg":
                        d = gA[g][gA[g].reward_group == c]; v = (d.corr_post_matched - d.corr_pre_matched).to_numpy()
                    else:
                        d = gS[g][gS[g].reward_group == c]; v = (d.delta - d.placebo_mean).to_numpy()
                    vals[(g, c)] = v
                    x = gi * 1.0 + (k - 0.5) * 0.36
                    if len(v):
                        ax.errorbar(x, np.nanmean(v), sem(v), fmt="o", ms=3.6, color=COL[c], lw=1.0, capsize=0,
                                    mfc=COL[c] if g == "step" else "white", mew=0.9)
                        ax.text(x, -0.02, f"{len(v)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4.6, color=COL[c])
                    pw, pt, n = one_sample(v)
                    rows.append(dict(window=win, measure=key, group=g, cohort=c, test="vs 0 (Wilcoxon | t)", n=n, mean=np.nanmean(v) if len(v) else np.nan,
                                     p_nonparam=pw, p_param=pt))
                mw, we = unpaired(vals[(g, "R+")], vals[(g, "R-")])
                rows.append(dict(window=win, measure=key, group=g, cohort="R+ vs R-", test="Mann-Whitney | Welch",
                                 n=len(vals[(g, "R+")]) + len(vals[(g, "R-")]), mean=np.nan, p_nonparam=mw, p_param=we))
                bracket(ax, gi - 0.18, gi + 0.18, 1.02, f"{pnum(mw)} | {pnum(we)}", "k")
            for k, c in enumerate(COH):
                mw, we = unpaired(vals[("step", c)], vals[("gradual", c)])
                rows.append(dict(window=win, measure=key, group="step vs gradual", cohort=c, test="Mann-Whitney | Welch",
                                 n=len(vals[("step", c)]) + len(vals[("gradual", c)]), mean=np.nan, p_nonparam=mw, p_param=we))
                bracket(ax, 0 + (k - 0.5) * 0.36, 1 + (k - 0.5) * 0.36, 1.17 + 0.15 * k, f"step vs gradual {c}: {pnum(mw)} | {pnum(we)}", COL[c])
            ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
            ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([gl for _, gl in GROUPS] if ri == len(WINS) - 1 else [""] * 3, fontsize=5)
            ax.set_xlim(-0.6, len(GROUPS) - 0.4)
            ax.set_ylabel(f"{wl}\n{label}" if ci == 0 else label)
    fig.suptitle("Step learners (split at the L5 change point) vs gradual learners and non-learners (split at the midpoint)",
                 fontsize=6.5, y=0.995)
    return fig, pd.DataFrame(rows)


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    variant = sys.argv[2] if len(sys.argv) > 2 else "L5 whisker CP"
    setup()
    rng = np.random.default_rng(0)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    tag = variant.split()[0]
    fig, R = figure_windows(scope, variant, rng)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"143_lt_split_{tag}_windows_{scope}.{ext}", dpi=300)
    R.insert(0, "scope", scope); R.insert(1, "variant", variant)
    R.to_csv(EA / f"143_stats_{tag}_{scope}.csv", index=False)
    fig2, R2 = figure_groups(scope, rng)
    for ext in ("png", "pdf", "svg"):
        fig2.savefig(FIGDIR / f"143_step_vs_gradual_{scope}.{ext}", dpi=300)
    R2.insert(0, "scope", scope)
    R2.to_csv(EA / f"143_step_vs_gradual_stats_{scope}.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.round(4).to_string(index=False)); print(R2.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
