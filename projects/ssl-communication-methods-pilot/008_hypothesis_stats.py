"""Statistics and figures for the hypothesis test v2 (007; predictions pre-registered in question.md).

Per session: excess = value - trial-shuffle null mean, averaged over draws; window means for the pre-registered windows
(pre -0.3..-0.1, early 0.05..0.15, late 0.2..0.4 s centres). A pair enters a cohort with >= 3 sessions; cohort tests
need both cohorts. R+ vs R-: Mann-Whitney and Welch (both reported); within cohort vs 0: Wilcoxon signed-rank;
BH-FDR within each prediction x window x quantity family (across pairs).
  P1  W_lick, pairs from SSp-w (A = SSp-w), direction index: R+ vs R-
  P2  W_nolick, pairs from Frontal (A = Frontal), direction index: R- vs R+
  P3  (W_lick - A_hit) per session, P1 pairs: R+ vs R-
  PM  W_lick_pm150 (and A_hit_pm150), early window: R+ vs R-
  PC  VPM-PO -> SSp-w direction index vs 0 per cohort (positive control), every condition
Outputs: OUT/hypothesis_v2/{session_excess.parquet, tests.csv, summary.md, figures/*.png}
Run (haas): python projects/ssl-communication-methods-pilot/008_hypothesis_stats.py [--all-mice]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comm_lib as cl  # noqa: E402

OUT = cl.OUT / "hypothesis_v2"
COL = {"R+": "#00B400", "R-": "#C800C8"}
WINDOWS = {"pre": (-0.3, -0.1), "early": (0.05, 0.15), "late": (0.2, 0.4)}
PRIMARY = "lagRRR_dir"
DIRECTED = ["lagRRR_dir", "lagRRR|rest_dir", "Granger_dir", "Granger|rest_dir"]
UNDIRECTED = ["lagRRR_ab", "lagRRR_ba", "pCCA|rest", "CCA", "RRR"]
P1_PAIRS = [("SSp-w", "wMO"), ("SSp-w", "ALM-tjM1"), ("SSp-w", "Striatum")]
P2_PAIRS = [("Frontal", "SSp-w"), ("Frontal", "Striatum"), ("Frontal", "wMO"), ("Frontal", "ALM-tjM1")]
PC_PAIR = ("VPM-PO", "SSp-w")
MIN_SESSIONS = 3


def load(all_mice):
    files = [f for f in (OUT / "sessions").glob("*.parquet") if not f.stem.endswith("_quick")]
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if not all_mice:
        d = d[d.learner]
    d["excess"] = d.value - d.null_mean
    keys = ["session_id", "mouse_id", "cohort", "condition", "area_a", "area_b", "quantity", "window_center"]
    return d.groupby(keys, as_index=False).agg(excess=("excess", "mean"), value=("value", "mean"),
                                                 null_mean=("null_mean", "mean"), n_draws=("draw", "nunique"))


def window_means(s):
    out = []
    for w, (lo, hi) in WINDOWS.items():
        x = s[(s.window_center >= lo - 1e-9) & (s.window_center <= hi + 1e-9)]
        g = x.groupby(["session_id", "mouse_id", "cohort", "condition", "area_a", "area_b", "quantity"],
                      as_index=False).excess.mean()
        g["window"] = w
        out.append(g)
    return pd.concat(out, ignore_index=True)


def two_group(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_SESSIONS or len(b) < MIN_SESSIONS:
        return dict(n_Rp=len(a), n_Rm=len(b), p_mw=np.nan, p_welch=np.nan)
    return dict(n_Rp=len(a), n_Rm=len(b), mean_Rp=a.mean(), mean_Rm=b.mean(), diff=a.mean() - b.mean(),
                p_mw=stats.mannwhitneyu(a, b).pvalue, p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue)


def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        from scipy.stats import false_discovery_control
        out[ok] = false_discovery_control(p[ok])
    return out


def tests(wm):
    rows = []

    def cohort_test(pred, cond, pairs, quantity, window, frame=None):
        f = wm if frame is None else frame
        for a, b in pairs:
            x = f[(f.condition == cond) & (f.area_a == a) & (f.area_b == b) & (f.quantity == quantity) &
                  (f.window == window)]
            r = two_group(x[x.cohort == "R+"].excess, x[x.cohort == "R-"].excess)
            rows.append(dict(prediction=pred, condition=cond, pair=f"{a} -> {b}", quantity=quantity, window=window,
                             **r))

    for q in DIRECTED + UNDIRECTED:
        for w in WINDOWS:
            cohort_test("P1", "W_lick", P1_PAIRS, q, w)
            cohort_test("P2", "W_nolick", P2_PAIRS, q, w)
            # P3: per-session whisker lick minus auditory hit
            k = ["session_id", "mouse_id", "cohort", "area_a", "area_b", "quantity", "window"]
            wl = wm[wm.condition == "W_lick"].set_index(k).excess
            ah = wm[wm.condition == "A_hit"].set_index(k).excess
            dd = (wl - ah).dropna().reset_index()
            dd["condition"] = "W_lick-A_hit"
            cohort_test("P3", "W_lick-A_hit", P1_PAIRS, q, w, frame=dd)
            cohort_test("P3b", "A_hit", P1_PAIRS, q, w)
        cohort_test("PM", "W_lick_pm150", P1_PAIRS, q, "early")
        cohort_test("PM", "A_hit_pm150", P1_PAIRS, q, "early")
    t = pd.DataFrame(rows)
    for col in ("p_mw", "p_welch"):
        t[f"{col}_fdr"] = t.groupby(["prediction", "condition", "quantity", "window"])[col].transform(
            lambda p: pd.Series(bh(p.values), index=p.index))
    # positive control: VPM-PO -> SSp-w lead vs 0 per cohort
    pc = []
    for (cond, q, w, coh), x in wm[(wm.area_a == PC_PAIR[0]) & (wm.area_b == PC_PAIR[1]) &
                                   wm.quantity.isin(DIRECTED)].groupby(["condition", "quantity", "window", "cohort"]):
        v = x.excess.dropna().to_numpy()
        pc.append(dict(prediction="PC", condition=cond, pair="VPM-PO -> SSp-w", quantity=q, window=w, cohort=coh,
                       n=len(v), mean=v.mean() if len(v) else np.nan,
                       p_wilcoxon=stats.wilcoxon(v).pvalue if len(v) >= MIN_SESSIONS else np.nan))
    return t, pd.DataFrame(pc)


def figures(s, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    pairs = P1_PAIRS + P2_PAIRS + [PC_PAIR, ("wMO", "Striatum"), ("ALM-tjM1", "Striatum")]
    conds = ["W_lick", "W_nolick", "A_hit", "NS_cr"]
    for q in DIRECTED + UNDIRECTED:
        x = s[s.quantity == q]
        fig, axs = plt.subplots(len(pairs), len(conds), figsize=(12, 2.0 * len(pairs)), sharex=True, sharey="row")
        for i, (a, b) in enumerate(pairs):
            for j, c in enumerate(conds):
                ax = axs[i, j]
                y = x[(x.area_a == a) & (x.area_b == b) & (x.condition == c)]
                for coh in ("R+", "R-"):
                    g = y[y.cohort == coh].groupby("window_center").excess
                    m, se, n = g.mean(), g.sem(), y[y.cohort == coh].session_id.nunique()
                    if n == 0:
                        continue
                    ax.plot(m.index, m.values, color=COL[coh], lw=1.3, label=f"{coh} (n={n})")
                    ax.fill_between(m.index, m - se, m + se, color=COL[coh], alpha=0.2, lw=0)
                ax.axhline(0, color="0.5", lw=0.5, ls=":")
                ax.axvline(0, color="0.5", lw=0.5)
                if i == 0:
                    ax.set_title(c, fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"{a} -> {b}", fontsize=7)
                ax.legend(fontsize=5.5, frameon=False, loc="upper left")
                ax.tick_params(labelsize=6)
        for ax in axs[-1]:
            ax.set_xlabel("window centre (s from stimulus)", fontsize=7)
        sign = "A->B minus B->A (excess over shuffle)" if q in DIRECTED else "excess over shuffle"
        fig.suptitle(f"{q}: {sign}; mean +- s.e.m. over sessions", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"timecourse_{q.replace('|', '_')}.png", dpi=170)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-mice", action="store_true", help="sensitivity: learners and non-learners")
    args = ap.parse_args()
    tag = "all_mice" if args.all_mice else "learners"
    s = load(args.all_mice)
    s.to_parquet(OUT / f"session_excess_{tag}.parquet")
    wm = window_means(s)
    t, pc = tests(wm)
    t.to_csv(OUT / f"tests_{tag}.csv", index=False)
    pc.to_csv(OUT / f"positive_control_{tag}.csv", index=False)
    figures(s, OUT / "figures" / tag)
    lines = [f"# Hypothesis test v2 ({tag})", "",
             f"Sessions: {s.session_id.nunique()} (R+ {s[s.cohort == 'R+'].session_id.nunique()}, "
             f"R- {s[s.cohort == 'R-'].session_id.nunique()}).", "",
             f"## Primary quantity {PRIMARY}", ""]
    prim = t[t.quantity == PRIMARY].round(4)
    lines += [prim.to_csv(index=False), "", "## Positive control (VPM-PO -> SSp-w)", "",
              pc[pc.quantity == PRIMARY].round(4).to_csv(index=False)]
    (OUT / f"summary_{tag}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
