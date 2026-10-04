"""138 -- COSYNE abstract figure: hit vs miss decoding (whole brain, 5-100 ms after the whisker stimulus) split at the stored
learning trial (L0), compared with the session's own linear-shift null and with placebo splits (user 2026-10-05: "Redo a
summary figure for the hit/miss 100ms decoding at L0 learning trial compared with placebo linear shift tests. Make a few
panels for a cosyne abstract, publication quality"). Style: skills/ssl-figure-style.
Panels:
  a  schematic: whisker hit-rate curve, the learning trial, pre / post epochs, placebo splits (>= 10 whisker trials away)
  b  accuracy minus linear-shift null before vs after the learning trial (118, tag _A1v2: separate size-matched decoders per
     epoch; null = labels shifted 10-50 % of the trials, non-wrapping), mean +- s.e.m. over sessions; paired Wilcoxon and
     paired t within cohort (cohort colour), Mann-Whitney and Welch on the change between cohorts (black)
  c  post - pre change at the learning trial vs the mean change at the session's placebo splits (122 step1_pl100 / 126);
     Wilcoxon and paired t, real vs placebo, per cohort
  d  excess = real change - mean placebo change; vs 0 per cohort (Wilcoxon and t); R+ vs R- (Mann-Whitney, Welch) and
     cohort-label permutation across mice (20000 shuffles, two-sided)
  e  change as a function of split position relative to the learning trial (whisker trials; 10-trial bins), mean +- s.e.m.
     over sessions per cohort; the real split at 0
Trials: prep_session (active, perf != 6, warm-up cut), rule A1 disengagement trim; learning stage, one session per mouse.
Scopes: all mice (main) and learners (learning_category good / moderate), argument `all` | `learners`. Uncorrected.
Outputs: figures/publication/138_cosyne_lt_placebo_<scope>.{png,pdf,svg}, 138_stats_<scope>.csv, 138_caption_<scope>.md
Run (haas, repo root): python .../138_cosyne_lt_placebo_figure.py [all|learners]
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
WHISKER = "#f7b519"
DEC, WIN, VAR = "hitmiss", "5-100ms", "L0 stored"
N_PERM, BIN, REL_RANGE = 20000, 10, (-10, 80)
COH = ("R+", "R-")


def setup():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 6, "axes.titlesize": 6.5, "axes.labelsize": 6, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5, "legend.fontsize": 5.5, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.titlepad": 4})


def letter_row(fig, axes, letters, dx_in=0.30, dy_in=0.16):
    W, H = fig.get_size_inches()
    top = max(ax.get_position().y1 for ax in axes)
    for ax, l in zip(axes, letters):
        fig.text(ax.get_position().x0 - dx_in / W, top + dy_in / H, l, fontsize=9, weight="bold", ha="left", va="bottom")


def pnum(p):
    return "n/a" if not np.isfinite(p) else ("<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}")


def fmt_p(p):
    return "n/a" if not np.isfinite(p) else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}")


def sem(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def paired(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    return (stats.wilcoxon(a, b).pvalue if len(a) > 5 else np.nan, stats.ttest_rel(a, b).pvalue if len(a) > 2 else np.nan, len(a))


def unpaired(x, y):
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    return stats.mannwhitneyu(x, y).pvalue, stats.ttest_ind(x, y, equal_var=False).pvalue


def one_sample(x):
    x = x[np.isfinite(x)]
    return stats.wilcoxon(x).pvalue, stats.ttest_1samp(x, 0).pvalue, len(x)


def bracket(ax, x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y - 0.012, y, y, y - 0.012], color=color, lw=0.6, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.01, text, color=color, ha="center", va="bottom", fontsize=5, transform=ax.get_xaxis_transform())


def load(scope):
    A = pd.read_parquet(EA / "118_lt_definitions_split_whole_brain_A1v2.parquet")
    A = A[(A.definition == VAR) & (A.decoding == DEC) & (A.window == WIN)].dropna(subset=["corr_pre_matched", "corr_post_matched"])
    S = pd.read_csv(EA / "126_lt_variants_per_session_step1_pl100.csv")
    S = S[(S.variant == VAR) & (S.decoding == DEC) & (S.window == WIN)].dropna(subset=["delta", "placebo_mean"])
    P = pd.read_parquet(EA / "122_lt_placebo_whole_brain_step1_pl100.parquet")
    P = P[(P.decoding == DEC) & (P.window == WIN)].dropna(subset=["delta_matched"])
    if scope == "learners":
        ref = pd.read_parquet(REPO / "reports/ssl_analysis/derived/mouse_reference.parquet")
        learners = set(ref[ref.learning_category.isin(["good", "moderate"])].subject_id)
        mouse = lambda d: d.session_id.str.split("_").str[0]
        A, S, P = (d[mouse(d).isin(learners)] for d in (A, S, P))
    return A, S, P


def main():
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    setup()
    A, S, P = load(scope)
    rng = np.random.default_rng(0)
    rows = []

    # ---- b: pre vs post above the linear-shift null
    pre = {c: A[A.reward_group == c].corr_pre_matched.to_numpy() for c in COH}
    post = {c: A[A.reward_group == c].corr_post_matched.to_numpy() for c in COH}
    chg = {c: post[c] - pre[c] for c in COH}
    for c in COH:
        pw, pt, n = paired(post[c], pre[c])
        rows.append(dict(panel="b", test="post vs pre (paired Wilcoxon | paired t)", cohort=c, n=n, mean_a=np.nanmean(pre[c]),
                         mean_b=np.nanmean(post[c]), p_nonparam=pw, p_param=pt))
    mw, we = unpaired(chg["R+"], chg["R-"])
    rows.append(dict(panel="b", test="change R+ vs R- (Mann-Whitney | Welch)", cohort="R+ vs R-", n=len(chg["R+"]) + len(chg["R-"]),
                     mean_a=np.nanmean(chg["R+"]), mean_b=np.nanmean(chg["R-"]), p_nonparam=mw, p_param=we))

    # ---- c, d: real vs placebo, excess
    real = {c: S[S.reward_group == c].delta.to_numpy() for c in COH}
    plac = {c: S[S.reward_group == c].placebo_mean.to_numpy() for c in COH}
    exc = {c: real[c] - plac[c] for c in COH}
    for c in COH:
        pw, pt, n = paired(real[c], plac[c])
        rows.append(dict(panel="c", test="real vs placebo change (paired Wilcoxon | paired t)", cohort=c, n=n, mean_a=np.nanmean(plac[c]),
                         mean_b=np.nanmean(real[c]), p_nonparam=pw, p_param=pt))
        pw, pt, n = one_sample(exc[c])
        rows.append(dict(panel="d", test="excess vs 0 (Wilcoxon | one-sample t)", cohort=c, n=n, mean_a=np.nanmean(exc[c]), mean_b=np.nan,
                         p_nonparam=pw, p_param=pt))
    mw, we = unpaired(exc["R+"], exc["R-"])
    x = np.r_[exc["R+"], exc["R-"]]; lab = np.r_[np.ones(len(exc["R+"]), bool), np.zeros(len(exc["R-"]), bool)]
    obs = x[lab].mean() - x[~lab].mean()
    null = np.array([(lambda p: x[p].mean() - x[~p].mean())(rng.permutation(lab)) for _ in range(N_PERM)])
    p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM + 1)
    rows.append(dict(panel="d", test="excess R+ vs R- (Mann-Whitney | Welch)", cohort="R+ vs R-", n=len(x), mean_a=np.nanmean(exc["R+"]),
                     mean_b=np.nanmean(exc["R-"]), p_nonparam=mw, p_param=we))
    rows.append(dict(panel="d", test=f"excess R+ - R-, cohort-label permutation ({N_PERM})", cohort="R+ vs R-", n=len(x), mean_a=obs,
                     mean_b=np.nan, p_nonparam=p_perm, p_param=np.nan))

    # ---- e: change vs split position relative to the learning trial
    lt = S.set_index("session_id")["lt"]
    Q = P[P.session_id.isin(lt.index)].copy()
    Q["rel"] = Q.split_k - Q.session_id.map(lt)
    Q = Q[(Q.rel >= REL_RANGE[0]) & (Q.rel < REL_RANGE[1])]
    Q["bin"] = (np.floor(Q.rel / BIN) * BIN + BIN / 2)
    prof = Q.groupby(["reward_group", "session_id", "bin"]).delta_matched.mean().reset_index()

    # ---- figure
    fig = plt.figure(figsize=(7.4, 2.35))
    xs = [0.045, 0.265, 0.475, 0.685, 0.83]
    ws = [0.15, 0.135, 0.135, 0.075, 0.16]
    axs = [fig.add_axes([x, 0.2, w, 0.5]) for x, w in zip(xs, ws)]

    # a schematic
    ax = axs[0]
    t = np.linspace(0, 100, 300); curve = 0.15 + 0.6 / (1 + np.exp(-(t - 40) / 6))
    ax.plot(t, curve, color=WHISKER, lw=1.4)
    ax.axvspan(0, 40, color="0.92", lw=0, edgecolor="none"); ax.axvspan(40, 100, color="0.85", lw=0, edgecolor="none")
    ax.axvline(40, color="k", lw=0.9)
    for k in range(4, 100, 6):
        if abs(k - 40) >= 10:
            ax.plot([k, k], [0.0, 0.06], color="0.45", lw=0.6)
    ax.text(20, 0.93, "before", ha="center", fontsize=5.5); ax.text(70, 0.93, "after", ha="center", fontsize=5.5)
    ax.text(42, 0.27, "learning\ntrial", fontsize=5, va="center")
    ax.text(73, 0.47, "placebo splits\n(>= 10 trials away)", ha="center", fontsize=4.8, color="0.35")
    ax.set_xlim(0, 100); ax.set_ylim(-0.02, 1.02); ax.set_yticks([0, 1]); ax.set_xticks([])
    ax.set_ylabel("Whisker hit rate"); ax.set_xlabel("Whisker trials in the session", labelpad=3)
    ax.set_title("Split each session", y=1.36)

    # b pre vs post
    ax = axs[1]
    xp = {"R+": (0, 1), "R-": (2.2, 3.2)}
    for c in COH:
        m = [np.nanmean(pre[c]), np.nanmean(post[c])]; e = [sem(pre[c]), sem(post[c])]
        ax.plot(xp[c], m, color=COL[c], lw=1.1)
        for x_, mm, ee, filled in zip(xp[c], m, e, (False, True)):
            ax.errorbar(x_, mm, ee, fmt="o", ms=4, color=COL[c], mfc=COL[c] if filled else "white", mew=1, lw=1.1, capsize=0)
    ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels(["before", "after", "before", "after"])
    ax.set_xlim(-0.5, 3.7)
    ax.set_ylabel("Hit vs miss decoding\n(accuracy - shift null)")
    rb = {r["cohort"]: r for r in rows if r["panel"] == "b"}
    for c, yb in zip(COH, (1.02, 1.02)):
        bracket(ax, *xp[c], yb, f"p {pnum(rb[c]['p_nonparam'])} | {pnum(rb[c]['p_param'])}", COL[c])
    bracket(ax, 0.5, 2.7, 1.17, f"change: p {pnum(rb['R+ vs R-']['p_nonparam'])} | {pnum(rb['R+ vs R-']['p_param'])}", "k")
    ax.set_title("At the learning trial", y=1.36)

    # c real vs placebo
    ax = axs[2]
    xc = {"R+": (0, 1.3), "R-": (2.8, 4.1)}
    for c in COH:
        m = [np.nanmean(plac[c]), np.nanmean(real[c])]; e = [sem(plac[c]), sem(real[c])]
        ax.plot(xc[c], m, color=COL[c], lw=1.1)
        for x_, mm, ee, filled in zip(xc[c], m, e, (False, True)):
            ax.errorbar(x_, mm, ee, fmt="o", ms=4, color=COL[c], mfc=COL[c] if filled else "white", mew=1, lw=1.1, capsize=0)
    ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0.65, 3.45]); ax.set_xticklabels(["R+", "R-"])
    ax.text(0.99, 0.99, "open: placebo splits\nfilled: learning trial", transform=ax.transAxes, ha="right", va="top",
            fontsize=4.8, color="0.3")
    ax.set_xlim(-0.6, 4.7); ax.set_ylabel("Change in decoding\n(after - before)")
    rc = {r["cohort"]: r for r in rows if r["panel"] == "c"}
    for c in COH:
        bracket(ax, *xc[c], 1.02, f"p {pnum(rc[c]['p_nonparam'])} | {pnum(rc[c]['p_param'])}", COL[c])
    ax.set_title("Real vs placebo splits", y=1.36)

    # d excess
    ax = axs[3]
    for i, c in enumerate(COH):
        ax.bar(i, np.nanmean(exc[c]), 0.6, color=COL[c], alpha=0.35, lw=0, edgecolor="none")
        ax.errorbar(i, np.nanmean(exc[c]), sem(exc[c]), fmt="o", ms=4, color=COL[c], lw=1.1, capsize=0)
    ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["R+", "R-"]); ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Change beyond placebo\n(real - placebo)")
    rd = [r for r in rows if r["panel"] == "d" and r["cohort"] == "R+ vs R-"]
    bracket(ax, 0, 1, 1.02, f"p {pnum(rd[0]['p_nonparam'])} | {pnum(rd[0]['p_param'])}", "k")
    bracket(ax, 0, 1, 1.17, f"permutation {fmt_p(rd[1]['p_nonparam'])}", "k")
    ax.set_title("Cohort difference", y=1.36)

    # e profile
    ax = axs[4]
    for c in COH:
        g = prof[prof.reward_group == c].groupby("bin").delta_matched
        m, e, n = g.mean(), g.apply(sem), g.size()
        ok = n >= 5
        ax.fill_between(m.index[ok], (m - e)[ok], (m + e)[ok], color=COL[c], alpha=0.25, lw=0, edgecolor="none")
        ax.plot(m.index[ok], m[ok], color=COL[c], lw=1.1, label=f"{c} (n = {S[S.reward_group == c].session_id.nunique()})")
    ax.axvline(0, color="k", lw=0.8); ax.axhline(0, color="0.5", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("Split position (whisker trials\nfrom the learning trial)"); ax.set_ylabel("Change in decoding\n(after - before)")
    ax.legend(frameon=False, loc="lower right", handlelength=1.2, borderaxespad=0.1)
    ax.set_title("Every split of the session", y=1.36)

    letter_row(fig, axs, list("abcde"), dy_in=0.40)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    name = f"138_cosyne_lt_placebo_{scope}"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGDIR / f"{name}.{ext}", dpi=300)
    R = pd.DataFrame(rows); R.insert(0, "scope", scope); R.insert(1, "decoding", f"{DEC} {WIN}, whole brain, split {VAR}")
    R.to_csv(EA / f"138_stats_{scope}.csv", index=False)
    n = {c: len(chg[c]) for c in COH}; nS = {c: len(exc[c]) for c in COH}
    cap = (f"**Hit vs miss decoding at the learning trial vs placebo splits** ({scope}; whole brain, 5-100 ms after the whisker "
           f"stimulus, learning stage, one session per mouse). **a** Each session is split at its learning trial; separate "
           f"size-matched decoders before and after. Placebo splits: every other whisker trial >= 10 trials away. **b** "
           f"Decoding accuracy minus each epoch's linear-shift null (labels shifted 10-50 % of the trials), mean +- s.e.m. over "
           f"sessions (R+ n = {n['R+']}, R- n = {n['R-']}); open = before, filled = after; paired Wilcoxon | paired t (cohort colour), "
           f"Mann-Whitney | Welch on the change (black); p values shown as non-parametric | parametric. **c** Change at the learning trial vs the mean change at the session's "
           f"placebo splits (R+ n = {nS['R+']}, R- n = {nS['R-']}). **d** Excess over placebo; Mann-Whitney | Welch and a cohort-label "
           f"permutation across mice ({N_PERM} shuffles). **e** Change vs split position relative to the learning trial "
           f"({BIN}-trial bins, bins with >= 5 sessions), mean +- s.e.m. over sessions. Trials: active context, invalid "
           f"(perf == 6) trials excluded, auditory warm-up block cut, end-of-session disengagement trimmed. Uncorrected.")
    (EA / f"138_caption_{scope}.md").write_text(cap + "\n")
    pd.set_option("display.width", 200)
    print(R.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
