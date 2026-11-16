"""Exact whisker and false-alarm learning curves of the sheet's 'bad' mice, mean curves per learning_category, and
candidate learner rules fitted to the sheet (user 2026-11-11: find a principled criterion that matches
learning_category).

Pages (Figures/_explore/learner_criterion_<date>/learner_curves.pdf, one PNG per page):
  1. every 'bad' mouse: exact whisker curve (cohort colour, 80% band), chance curve (black), lick ticks; d', hit, FA.
  2. sheet learners that the d' >= 0.3 rule calls non-discriminating (borderline mice), same layout.
  3. mean +- s.e.m. curves per cohort x category (good, moderate, bad): whisker, chance and whisker - chance, against
     normalised session time and against whisker trial (first 150).
  4. candidate splits: d' vs criterion c per cohort with the categories, and the rule table below.
Splits: per cohort, every single measure with its best threshold, every AND / OR pair of two thresholded measures, and
depth-1/2 decision trees; scored by balanced accuracy against the sheet (bad vs good/moderate) and, for the trees and the
best single / pair rules, by leave-one-mouse-out accuracy (threshold refitted without the mouse). Also rules that use the
same measure in both cohorts.

Run on haas after explore_learner_criterion.py (reads its per_mouse.csv):
  ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_learner_curves.py
"""

import importlib.util
import itertools
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from scipy import stats  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FIG_DIR.parents[1]))
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT  # noqa: E402

_spec = importlib.util.spec_from_file_location("fig1_compute", FIG_DIR / "compute.py")
fig1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fig1)

COLORS = {"R+": "#00B400", "R-": "#C800C8"}
CATS = ["good", "moderate", "bad"]
CAT_SHADE = {"good": 1.0, "moderate": 0.6, "bad": 0.3}
MEASURES = ["dprime", "criterion_c", "hit_whisker", "fa", "hit_minus_fa", "gap_max", "gap_mean", "gap_last20",
            "gap_first50", "run_above", "frac_above", "run_below", "dprime_curve_mean", "dprime_curve_max",
            "dprime_low_max", "first_whisker_hit", "n_whisker"]


def curves_of(st):
    st = st[st.early_lick == 0].sort_values("start_time")
    c, _, _ = fig1.session_curves(st)
    return c.assign(mouse_id=st.mouse_id.iloc[0], cohort=st.cohort.iloc[0], learning_category=st.learning_category.iloc[0],
                    t_norm=(c.whisker_trial - 1) / max(len(c) - 1, 1))


def shade(color, f):
    rgb = np.array(matplotlib.colors.to_rgb(color))
    return tuple(1 - f * (1 - rgb))


def mouse_grid(pdf, out, curves, per_mouse, mice, title, name):
    n = len(mice)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.3 * nrow), squeeze=False)
    for a, m in zip(axes.flat, mice):
        c = curves[curves.mouse_id == m]
        r = per_mouse.set_index("mouse_id").loc[m]
        col = COLORS[r.cohort]
        x = c.whisker_trial
        a.fill_between(x, c.p_low, c.p_high, color=shade(col, 0.3), lw=0)
        a.plot(x, c.p_mean, color=col, lw=1)
        a.plot(x, c.p_chance, color="k", lw=0.8)
        a.scatter(x[c.lick == 1], np.full((c.lick == 1).sum(), 1.04), marker="|", s=10, color=col, lw=0.5)
        a.scatter(x[c.lick == 0], np.full((c.lick == 0).sum(), -0.04), marker="|", s=10, color="0.6", lw=0.5)
        a.set_ylim(-0.08, 1.08)
        a.set_title(f"{m} {r.cohort} {r.learning_category}: d'={r.dprime:.2f}, c={r.criterion_c:.2f}\n"
                    f"hit {r.hit_whisker:.2f}, FA {r.fa:.2f}, run above {int(r.run_above)}", fontsize=7)
        a.tick_params(labelsize=6)
    for a in axes.flat[n:]:
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def mean_curves(pdf, out, curves):
    grid = np.linspace(0, 1, 51)
    rows = []
    for (m, coh, cat), c in curves.groupby(["mouse_id", "cohort", "learning_category"]):
        for v in ("p_mean", "p_chance"):
            rows.append(dict(mouse_id=m, cohort=coh, cat=cat, v=v, axis="t_norm", x=grid,
                             y=np.interp(grid, c.t_norm, c[v])))
            k = min(len(c), 150)
            y = np.full(150, np.nan)
            y[:k] = c[v].to_numpy()[:k]
            rows.append(dict(mouse_id=m, cohort=coh, cat=cat, v=v, axis="trial", x=np.arange(1, 151), y=y))
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 6, figsize=(16, 5.5))
    for i, ax_kind in enumerate(("t_norm", "trial")):
        for j, coh in enumerate(("R+", "R-")):
            for cat in CATS:
                d = df[(df.axis == ax_kind) & (df.cohort == coh) & (df.cat == cat)]
                if d.empty:
                    continue
                n = d.mouse_id.nunique()
                W = np.vstack(d[d.v == "p_mean"].sort_values("mouse_id").y)
                F = np.vstack(d[d.v == "p_chance"].sort_values("mouse_id").y)
                x = d.x.iloc[0]
                col = shade(COLORS[coh], CAT_SHADE[cat])
                for k, (Y, ls) in enumerate(((W, "-"), (F, "--"), (W - F, "-"))):
                    a = axes[i, j * 3 + k]
                    m, s = np.nanmean(Y, 0), stats.sem(Y, 0, nan_policy="omit")
                    a.fill_between(x, m - s, m + s, color=col, alpha=0.2, lw=0)
                    a.plot(x, m, color=col, ls=ls, lw=1.2, label=f"{cat} (n={n})")
            for k, t in enumerate(("whisker curve", "chance (FA) curve", "whisker - chance")):
                a = axes[i, j * 3 + k]
                a.set_title(f"{coh}: {t}", fontsize=8)
                a.set_xlabel("session time (norm.)" if ax_kind == "t_norm" else "whisker trial", fontsize=7)
                a.axhline(0, color="0.7", lw=0.5, ls=":")
                a.tick_params(labelsize=6)
            axes[i, j * 3].legend(fontsize=6, frameon=False)
    fig.suptitle("Mean +- s.e.m. exact curves per learning_category (whisker day 0, all mice)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "mean_curves_by_category.png", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def bacc(pred, bad):
    """Balanced accuracy of 'learner' predictions against the sheet (bad = non-learner)."""
    learner = ~bad
    sens = (pred & learner).sum() / max(learner.sum(), 1)
    spec = (~pred & bad).sum() / max(bad.sum(), 1)
    return (sens + spec) / 2


def best_threshold(x, bad):
    """Best threshold and direction on one measure (learner if x >= t, or x <= t)."""
    best = (-1, None, None)
    for t in np.unique(x[np.isfinite(x)]):
        for sign in (1, -1):
            pred = x >= t if sign == 1 else x <= t
            s = bacc(pred, bad)
            if s > best[0]:
                best = (s, t, sign)
    return best


def apply(x, t, sign):
    return x >= t if sign == 1 else x <= t


def loo_single(x, bad):
    hits = []
    for i in range(len(x)):
        keep = np.arange(len(x)) != i
        _, t, sign = best_threshold(x[keep], bad[keep])
        hits.append(apply(x[i:i + 1], t, sign)[0] == (not bad[i]))
    return float(np.mean(hits))


def splits(df):
    rows = []
    for coh in ("R+", "R-"):
        d = df[df.cohort == coh].reset_index(drop=True)
        bad = d.bad.to_numpy()
        singles = {}
        for m in MEASURES:
            x = d[m].to_numpy(float)
            s, t, sign = best_threshold(x, bad)
            singles[m] = (s, t, sign)
            pred = apply(x, t, sign)
            rows.append(dict(cohort=coh, kind="single", rule=f"{m} {'>=' if sign == 1 else '<='} {t:.3g}",
                             balanced_acc=s, accuracy=float((pred == ~bad).mean()), loo_accuracy=np.nan,
                             bad_called_learner=sorted(d.mouse_id[pred & bad]),
                             learner_called_bad=sorted(d.mouse_id[~pred & ~bad])))
        # pairs: each measure at its own best threshold, combined with AND / OR
        for m1, m2 in itertools.combinations(MEASURES, 2):
            p1 = apply(d[m1].to_numpy(float), *singles[m1][1:])
            p2 = apply(d[m2].to_numpy(float), *singles[m2][1:])
            for op, pred in (("AND", p1 & p2), ("OR", p1 | p2)):
                rows.append(dict(cohort=coh, kind="pair", rule=f"({m1} {'>=' if singles[m1][2] == 1 else '<='} "
                                 f"{singles[m1][1]:.3g}) {op} ({m2} {'>=' if singles[m2][2] == 1 else '<='} {singles[m2][1]:.3g})",
                                 balanced_acc=bacc(pred, bad), accuracy=float((pred == ~bad).mean()), loo_accuracy=np.nan,
                                 bad_called_learner=sorted(d.mouse_id[pred & bad]),
                                 learner_called_bad=sorted(d.mouse_id[~pred & ~bad])))
        for m in sorted(singles, key=lambda k: -singles[k][0])[:5]:
            r = next(r for r in rows if r["cohort"] == coh and r["kind"] == "single" and r["rule"].startswith(m + " "))
            r["loo_accuracy"] = loo_single(d[m].to_numpy(float), bad)
        from sklearn.model_selection import LeaveOneOut, cross_val_predict
        from sklearn.tree import DecisionTreeClassifier, export_text
        X = d[MEASURES].fillna(d[MEASURES].median()).to_numpy(float)
        for depth in (1, 2):
            clf = DecisionTreeClassifier(max_depth=depth, class_weight="balanced", random_state=0)
            loo = cross_val_predict(clf, X, ~bad, cv=LeaveOneOut())
            clf.fit(X, ~bad)
            pred = clf.predict(X).astype(bool)
            rows.append(dict(cohort=coh, kind=f"tree depth {depth}",
                             rule=export_text(clf, feature_names=MEASURES).replace("\n", " | "),
                             balanced_acc=bacc(pred, bad), accuracy=float((pred == ~bad).mean()),
                             loo_accuracy=float((loo.astype(bool) == ~bad).mean()),
                             bad_called_learner=sorted(d.mouse_id[pred & bad]),
                             learner_called_bad=sorted(d.mouse_id[~pred & ~bad])))
    # same measure and threshold in both cohorts (direction may differ per cohort)
    for m in MEASURES:
        best = None
        for t in np.unique(df[m].dropna()):
            for s_p, s_m in itertools.product((1, -1), (1, -1)):
                preds = {c: apply(df[df.cohort == c][m].to_numpy(float), t, s) for c, s in (("R+", s_p), ("R-", s_m))}
                score = np.mean([bacc(preds[c], df[df.cohort == c].bad.to_numpy()) for c in preds])
                if best is None or score > best[0]:
                    best = (score, t, s_p, s_m, preds)
        score, t, s_p, s_m, preds = best
        bc = sum((sorted(df[df.cohort == c].mouse_id[preds[c] & df[df.cohort == c].bad.to_numpy()]) for c in preds), [])
        lb = sum((sorted(df[df.cohort == c].mouse_id[~preds[c] & ~df[df.cohort == c].bad.to_numpy()]) for c in preds), [])
        rows.append(dict(cohort="both", kind="same measure",
                         rule=f"{m}: R+ {'>=' if s_p == 1 else '<='} {t:.3g}, R- {'>=' if s_m == 1 else '<='} {t:.3g}",
                         balanced_acc=score, accuracy=float(np.mean(np.concatenate(
                             [preds[c] == ~df[df.cohort == c].bad.to_numpy() for c in preds]))), loo_accuracy=np.nan,
                         bad_called_learner=bc, learner_called_bad=lb))
    return pd.DataFrame(rows)


def split_page(pdf, out, df, table):
    fig = plt.figure(figsize=(16, 11))
    for j, coh in enumerate(("R+", "R-")):
        a = fig.add_subplot(2, 3, j + 1)
        d = df[df.cohort == coh]
        for cat in CATS:
            dd = d[d.learning_category == cat]
            a.scatter(dd.dprime, dd.criterion_c, s=18, color=shade(COLORS[coh], CAT_SHADE[cat]),
                      edgecolor="k" if cat == "bad" else "none", lw=0.6, label=f"{cat} (n={len(dd)})")
            for r in dd[dd.learning_category != "good"].itertuples():
                a.annotate(r.mouse_id, (r.dprime, r.criterion_c), fontsize=5, xytext=(2, 2), textcoords="offset points")
        a.axvline(0.3, color="0.5", lw=0.6, ls="--")
        a.set_xlabel("d' (session)", fontsize=8)
        a.set_ylabel("criterion c (higher = conservative)", fontsize=8)
        a.set_title(f"{coh}: d' vs bias; dashed d' = 0.3", fontsize=9)
        a.legend(fontsize=6, frameon=False)
        a.tick_params(labelsize=7)
    a = fig.add_subplot(2, 3, 3)
    for coh in ("R+", "R-"):
        d = df[df.cohort == coh]
        for cat in CATS:
            dd = d[d.learning_category == cat]
            a.scatter(dd.hit_whisker, dd.fa, s=18, color=shade(COLORS[coh], CAT_SHADE[cat]),
                      edgecolor="k" if cat == "bad" else "none", lw=0.6, marker="o" if coh == "R+" else "s")
    a.plot([0, 1], [0, 1], color="0.6", lw=0.6, ls=":")
    a.set_xlabel("whisker hit rate", fontsize=8)
    a.set_ylabel("false-alarm rate", fontsize=8)
    a.set_title("Hit vs FA (circles R+, squares R-; black edge = bad)", fontsize=9)
    a.tick_params(labelsize=7)
    a = fig.add_subplot(2, 1, 2)
    a.axis("off")
    show = []
    for coh in ("R+", "R-", "both"):
        t = table[table.cohort == coh].sort_values(["balanced_acc", "accuracy"], ascending=False)
        show.append(t[t.kind == "single"].head(4) if coh != "both" else t.head(4))
        if coh != "both":
            show.append(t[t.kind == "pair"].head(3))
            show.append(t[t.kind.str.startswith("tree")])
    s = pd.concat(show)
    text = "\n".join(f"{r.cohort:4s} {r.kind:12s} bal.acc {r.balanced_acc:.2f} acc {r.accuracy:.2f} "
                     f"LOO {r.loo_accuracy:.2f}  {r.rule[:95]:95s}  bad->L {','.join(r.bad_called_learner)[:40]}  "
                     f"L->bad {','.join(r.learner_called_bad)[:60]}" for r in s.itertuples())
    a.text(0, 1, text, family="monospace", fontsize=5.6, va="top")
    fig.tight_layout()
    fig.savefig(out / "splits.png", dpi=150)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    out = sorted((FIGURES_ROOT / "_explore").glob("learner_criterion_*"))[-1]
    per_mouse = pd.read_csv(out / "per_mouse.csv")
    h = (per_mouse.hit_whisker * per_mouse.n_whisker + 0.5) / (per_mouse.n_whisker + 1)
    f = (per_mouse.fa * per_mouse.n_no_stim + 0.5) / (per_mouse.n_no_stim + 1)
    per_mouse["criterion_c"] = -(stats.norm.ppf(h) + stats.norm.ppf(f)) / 2  # log-linear corrected, as d'
    per_mouse["bad"] = per_mouse.learning_category == "bad"
    d0 = fig1.day0_table(shared_behaviour.load())
    with Pool(16) as pool:
        curves = pd.concat(pool.map(curves_of, [st for _, st in d0.groupby("session_id")]), ignore_index=True)
    curves.to_parquet(out / "curves_day0.parquet")
    table = splits(per_mouse)
    table.to_csv(out / "splits.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 120)
    for coh in ("R+", "R-", "both"):
        t = table[table.cohort == coh].sort_values(["balanced_acc", "accuracy"], ascending=False)
        print(f"===== {coh}")
        if coh == "both":
            print(t.head(8)[["rule", "balanced_acc", "accuracy", "bad_called_learner", "learner_called_bad"]].to_string())
            continue
        for kind in ("single", "pair"):
            print(t[t.kind == kind].head(6)[["rule", "balanced_acc", "accuracy", "loo_accuracy", "bad_called_learner",
                                             "learner_called_bad"]].to_string())
        print(t[t.kind.str.startswith("tree")][["kind", "rule", "balanced_acc", "accuracy", "loo_accuracy",
                                                 "bad_called_learner", "learner_called_bad"]].to_string())
    bad = per_mouse[per_mouse.bad].sort_values(["cohort", "dprime"]).mouse_id.tolist()
    border = per_mouse[(~per_mouse.bad) & (per_mouse.dprime < 0.3) & (per_mouse.cohort == "R+")].sort_values("dprime")
    with PdfPages(out / "learner_curves.pdf") as pdf:
        mouse_grid(pdf, out, curves, per_mouse, bad, "Sheet 'bad' mice: exact whisker (colour, 80% band) and chance "
                   "(black) curves, whisker day 0", "bad_mice_curves")
        mouse_grid(pdf, out, curves, per_mouse, border.mouse_id.tolist(), "R+ sheet learners with session d' < 0.3",
                   "borderline_rplus_curves")
        mean_curves(pdf, out, curves)
        split_page(pdf, out, per_mouse, table)
    per_mouse.to_csv(out / "per_mouse.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
