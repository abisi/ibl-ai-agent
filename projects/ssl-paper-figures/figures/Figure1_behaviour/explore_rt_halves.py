"""Exploration (user 2026-11-12): Figure 1 j left as first half vs second half of whisker hits instead of the change
curve and slope. Learners, whisker day 0, early licks dropped, reaction time 0-1 s (compute.day0_hits, cached behaviour
table; nothing recomputed upstream).

Per mouse, median reaction time of whisker hits in the first and second half of
  A. the 30 hits of the panel (hits 1-15 vs 16-30; mice with >= 20 hits, halves split at the middle of their hits <= 30)
  B. all whisker hits of the session (first vs second half of the mouse's hits; mice with >= 10 hits)
and the same for auditory hits (control, all hits). Paired dots per mouse (grey lines), cohort mean +- s.e.m.; stars,
paired Wilcoxon second vs first half within cohort; bracket, R+ vs R- Mann-Whitney on the per-mouse change (Welch in the
CSV).
Output: Figures/_explore/rt_halves_<date>.{pdf,png} and _stats.csv.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_rt_halves.py
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
import plot as fig1  # noqa: E402
import shared_behaviour  # noqa: E402
from common import FIGURES_ROOT, two_group_tests  # noqa: E402
from paperstyle import MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
VARIANTS = [("whisker_trial", 30, 20, "W hits 1-15 vs 16-30"), ("whisker_trial", None, 10, "All W hits, halves"),
            ("auditory_trial", None, 10, "All A hits, halves (control)")]


def halves(h, tt, max_hits, min_hits):
    d = h[(h.trial_type == tt)].dropna(subset=["rt_ms"])
    if max_hits:
        d = d[d.hit_number <= max_hits]
    rows = []
    for (m, s, c), g in d.groupby(["mouse_id", "session_id", "cohort"]):
        if len(g) < min_hits:
            continue
        g = g.sort_values("hit_number")
        k = len(g) // 2
        rows.append(dict(mouse_id=m, session_id=s, cohort=c, n_hits=len(g), first=g.rt_ms.iloc[:k].median(),
                         second=g.rt_ms.iloc[len(g) - k:].median()))
    out = pd.DataFrame(rows)
    out["change"] = out.second - out.first
    return out


def main():
    set_nature_rc()
    h = compute.day0_hits({"table": shared_behaviour.load()})
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(150 * MM, 52 * MM))
    stats_rows = []
    for ax, (tt, max_hits, min_hits, title) in zip(axes, VARIANTS):
        d = halves(h, tt, max_hits, min_hits)
        for k, c in enumerate(fig1.COHORTS):
            g = d[d.cohort == c]
            x = np.array([0, 1]) + 2.6 * k
            for r in g.itertuples():
                ax.plot(x, [r.first, r.second], color="0.82", lw=0.3, zorder=1)
            col = fig1.group_color(tt, c)
            ax.errorbar(x, [g["first"].mean(), g.second.mean()], [g["first"].sem(), g.second.sem()], color=col,
                        marker="o", ms=2.5, lw=0.9, elinewidth=0.7, capsize=0, zorder=3)
            p = stats.wilcoxon(g.change).pvalue if len(g) > 5 else np.nan
            top = max(g["first"].quantile(0.95), g.second.quantile(0.95))
            ax.text(x.mean(), top, f"{fig1.stars(p)}\n{g.change.mean():+.0f} ms", ha="center", va="bottom", fontsize=5,
                    color=col)
            stats_rows.append(dict(variant=title, cohort=c, n_mice=len(g), first_ms=g["first"].mean(),
                                   second_ms=g.second.mean(), change_ms=g.change.mean(), change_sem=g.change.sem(),
                                   wilcoxon_p=p))
        t = two_group_tests(d[d.cohort == "R+"].change, d[d.cohort == "R-"].change)
        stats_rows.append(dict(variant=title, cohort="R+ vs R-", **{k: v for k, v in t.items() if np.isscalar(v)}))
        mw = t.get("mannwhitney_p", t.get("mw_p", np.nan))
        ax.set_title(f"{title}\nchange R+ vs R−: {fig1.stars(mw) if np.isfinite(mw) else ''} (MW p = {mw:.2g})",
                     fontsize=6, pad=3)
        ax.set_xticks([0, 1, 2.6, 3.6], ["1st half\nR+", "2nd half\nR+", "1st half\nR−", "2nd half\nR−"], fontsize=5)
        ax.set_xlim(-0.5, 4.1)
        ax.set_ylabel(f"Median RT, {'whisker' if tt == 'whisker_trial' else 'auditory'} hits (ms)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"rt_halves_{datetime.date.today()}"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    pd.DataFrame(stats_rows).to_csv(f"{stem}_stats.csv", index=False)
    print(pd.DataFrame(stats_rows).to_string())
    print(stem)


if __name__ == "__main__":
    main()
