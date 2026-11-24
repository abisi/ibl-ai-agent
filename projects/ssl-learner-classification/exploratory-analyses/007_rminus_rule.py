"""007 -- Candidate learner rules for R- (user 2026-11-23: "two rules per cohort, because different tasks. What then could be
the rule for R-? Right now I have only 2 bad learners"; "for R-, we want to look at day 0 data"; session end cut as the
paper). R- task: the whisker stimulus is never rewarded; correct behaviour is to not lick it while still licking the rewarded
auditory tone. Mice that never licked the whisker are correct from the start (nothing to suppress).

Whisker day 0 only. Curves refitted on the session cut at the paper's session end (last 15 % of the time from the first to
the last whisker trial dropped; exact_cut.py, --keep=0.85 default; --keep=1.0 = whole session). k = 20 whisker trials (the
window of the R+ engagement rule). Metrics per mouse:
  early_w          raw whisker lick fraction, first k whisker trials (generalisation)
  late_sustained   max over k-trial windows in the second half of the raw whisker lick fraction (persistent licking)
  end_w            raw whisker lick fraction in the last k whisker trials of the cut session
  end_a            auditory curve P(lick) over those trials (still engaged?)
  late30_gap       raw whisker - FA lick rate, last 30 whisker trials and the no-stim trials of that period
  frac_above_h2    fraction of second-half whisker trials with the whisker curve credibly above chance (80 % band)
  supp_window      exists a k-trial window with whisker lick fraction <= 0.15 while the auditory curve is >= 0.7
  never_generalised  <= 3 whisker licks in the session
Candidate rules (non-learner = rule fails), with their non-learners, compared with the sheet (good / moderate / bad).
Outputs: tables/007_rminus_metrics<tag>.csv, tables/007_rminus_rules<tag>.csv, figures/007_rminus_rule<tag>.png|pdf.
Run (haas, repo root): ~/code/ibl-ai-agent/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/007_rminus_rule.py [--keep=0.85]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from axel_bisi_paths import axel_bisi_root  # noqa: E402
import exact_cut as EC  # noqa: E402

RES = axel_bisi_root() / "combined_results_ks4" / "ssl-learner-classification"
K = 20
KEEP = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--keep=")), 0.85)
TAG = f"_keep{int(round(KEEP * 100))}"


def run_mean(x, k):
    x = np.asarray(x, float)
    k = min(k, len(x))
    return np.convolve(x, np.ones(k) / k, mode="valid")


def metrics(mouse):
    c = EC.curves(mouse, KEEP)
    y, pl, ch, pa, st = c["outcomes"].astype(float), c["p_low"], c["p_chance"], c["p_auditory"], c["start"]
    n, h = len(y), len(y) // 2
    late = run_mean(y[h:], K) if n - h >= 5 else np.array([np.nan])
    rw, ra = run_mean(y, K), run_mean(pa, K)
    return dict(mouse_id=mouse, n_whisker=n, n_cut=c["n_cut"], n_w_licks=int(y.sum()), early_w=float(y[:K].mean()),
                late_sustained=float(np.nanmax(late)), end_w=float(y[-K:].mean()), end_a=float(np.nanmean(pa[-K:])),
                frac_above_h2=float((pl[h:] > ch[h:]).mean()),
                supp_window=bool(np.any((rw <= 0.15) & (ra >= 0.7))), never_generalised=bool(y.sum() <= 3))


def late30_gap(mouse):
    """raw whisker - FA rate, last 30 whisker trials of the cut session and the no-stim trials in that period"""
    c = EC.curves(mouse, KEEP)
    ns = EC._load(mouse, "no_stim_trial")
    t0 = c["start"][max(len(c["start"]) - 30, 0)]
    tn, yn = np.asarray(ns.start_time, float), np.asarray(ns.outcomes, float)
    m = (tn >= t0) & (tn <= c["start"][-1])
    return float(c["outcomes"][-30:].mean() - (yn[m].mean() if m.any() else np.nan))


def main():
    pm = pd.read_csv(RES / "tables" / "001_per_mouse.csv")
    r = pm[pm.cohort == "R-"][["mouse_id", "learning_category", "learner"]]
    M = pd.DataFrame([metrics(m) for m in r.mouse_id])
    M["late30_gap"] = [late30_gap(m) for m in M.mouse_id]
    d = r.merge(M, on="mouse_id")
    rules = {"end of session: last 20 W <= 25 % licked": d.end_w <= 0.25,
             "end of session: last 20 W <= 30 % licked": d.end_w <= 0.30,
             "suppression window (<= 15 % W, A >= 0.7)": d.supp_window,
             "no sustained late licking (< 9/20, 2nd half)": d.late_sustained < 0.45,
             "late W - FA <= 0.1 (last 30 W)": d.late30_gap <= 0.1,
             "credibly > chance <= 10 % of 2nd half": d.frac_above_h2 <= 0.1}
    for k, v in rules.items():
        d[f"L: {k}"] = v
    d.to_csv(RES / "tables" / f"007_rminus_metrics{TAG}.csv", index=False)
    rows = []
    for k, v in rules.items():
        nl = d[~v]
        rows.append(dict(rule=k, keep=KEEP, n_learner=int(v.sum()), n=len(d), non_learners=",".join(nl.mouse_id),
                         sheet_bad_caught=",".join(nl[nl.learning_category == "bad"].mouse_id),
                         sheet_bad_missed=",".join(d[v & (d.learning_category == "bad")].mouse_id),
                         never_generalised_non=int(nl.never_generalised.sum())))
    S = pd.DataFrame(rows)
    S.to_csv(RES / "tables" / f"007_rminus_rules{TAG}.csv", index=False)
    pd.set_option("display.width", 250)
    cols = ["mouse_id", "learning_category", "n_whisker", "n_cut", "n_w_licks", "early_w", "late_sustained", "end_w",
            "end_a", "late30_gap", "frac_above_h2", "supp_window"]
    print(f"keep = {KEEP}")
    print(d.sort_values("end_w", ascending=False)[cols].head(14).round(2).to_string(index=False))
    print(S.drop(columns=["keep"]).to_string(index=False))

    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3), gridspec_kw=dict(wspace=0.6))
    cat = {"good": 0, "moderate": 1, "bad": 2}
    col = {"good": "#C800C8", "moderate": "#e38ae3", "bad": "k"}
    for ax, (f, lab, th) in zip(axs, (("early_w", "First 20 W trials\n(generalisation)", None),
                                       ("late_sustained", "Max 20-trial W lick fraction,\nsecond half", 0.45),
                                       ("end_w", "Last 20 W trials\n(cut session)", 0.25),
                                       ("late30_gap", "W − FA, last 30 W trials", 0.1))):
        for c_, g in d.groupby("learning_category"):
            x = cat[c_] + np.random.default_rng(cat[c_]).uniform(-0.15, 0.15, len(g))
            ax.scatter(x, g[f], s=7, color=col[c_], lw=0)
            for xx, yy, m in zip(x, g[f], g.mouse_id):
                if c_ == "bad" or (th is not None and yy > th):
                    ax.annotate(m, (xx, yy), fontsize=4, xytext=(2, 0), textcoords="offset points")
        if th is not None:
            ax.axhline(th, color="0.4", lw=0.5, ls="--")
        ax.set_xticks(range(3), ["good", "mod.", "bad"], fontsize=5)
        ax.set_title(lab, fontsize=5.5)
        ax.tick_params(labelsize=5)
    fig.suptitle(f"R− whisker day 0 (last {100 - int(round(KEEP * 100))} % of the session cut, curves refitted): candidate "
                 "metrics by sheet category; above dashed = non-learner by that rule. Exploratory.", fontsize=6, y=1.04)
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"007_rminus_rule{TAG}.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
