"""006 -- Sustained-whisker-licking metrics, developed from 005 (user 2026-11-23: "develop metrics of 005"). In 005 the best
20-trial window of the exact whisker curve separated the user's known R+ non-learners from every good / moderate mouse.
Here the metric is made explicit and tested for robustness. All curves from the stored exact learning curves
(combined_results_ks4/<mouse>/whisker_0/learning_curve/*_learning_curve_exact.h5; grid HMM, sigma 1, 80 % band), day 0.

Metrics per mouse (window = k consecutive whisker trials, k in {10, 20, 30}):
  S_mean(k)  max over windows of the mean whisker-curve P(lick)                         (005 metric for k = 20)
  S_low(k)   max over windows of the mean lower 80 % bound of the whisker curve         (credible version)
  S_raw(k)   max over windows of the raw fraction of licked whisker trials              (model-free)
  S_rel(k)   max over windows of mean whisker P(lick) / mean auditory P(lick) in the same window (auditory curve
             interpolated at the whisker-trial times; controls for general engagement, no false alarms)
  T_cross    first whisker trial of the best S_mean(20) window (where sustained licking is first reached)
Evaluation, R+: known bad (7 fixed non-learners) vs good / moderate (47); flexible sheet-bad mice (5) held out.
  AUC; margin = (lowest good / moderate - highest known bad) / SD of all R+ (positive = perfect separation with a gap);
  best threshold and leave-one-out accuracy of re-choosing it; position of the flexible mice; R- distribution
  (no whisker reward: what generalisation alone reaches); agreement with the 9-criteria consensus (kappa); AUC of the
  learner label (metric > threshold) for day +1 whisker P(lick) and d'.
Outputs: tables/006_sustained_per_mouse.csv, tables/006_sustained_summary.csv, figures/006_sustained_metrics.png|pdf.
Session end (user 2026-11-23): the last 15 % of the session cut as Figure 1 (--keep=0.85, default; --keep=1.0 = whole session,
the first version of this script), curves refitted on the cut session (exact_cut.py); outputs carry the suffix _keep<pct>.
Run (haas, repo root): ~/code/ibl-ai-agent/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/006_sustained_metrics.py [--keep=0.85]
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score, roc_auc_score  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_cut as EC  # noqa: E402

KEEP = next((float(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--keep=")), 0.85)
TAG = f"_keep{int(round(KEEP * 100))}"

ROOT = axel_bisi_root() / "combined_results_ks4"
RES = ROOT / "ssl-learner-classification"
FIXED_BAD = ["MH020", "MH013", "AB142", "MH068", "AB151", "AB117", "AB107"]
FLEX = ["AB141", "MH007", "AB104", "MH026", "MH009"]
KS = (10, 20, 30)
GCOL = {"good": "#00B400", "moderate": "#7fd17f", "flexible": "tab:orange", "known bad": "k", "R-": "#C800C8"}


def load_exact(mouse, tt):
    f = ROOT / mouse / "whisker_0" / "learning_curve" / f"{mouse}_whisker_0_{tt}_learning_curve_exact.h5"
    return pd.read_hdf(f).iloc[0] if f.exists() else None


def win_max(x, k):
    x = np.asarray(x, float)
    k = min(k, len(x))
    v = np.convolve(x, np.ones(k) / k, mode="valid")
    return float(np.nanmax(v)), int(np.nanargmax(v))


def metrics(mouse):
    c = EC.curves(mouse, KEEP)   # session-end cut (paper: last 15 %), curves refitted on the cut session
    if c is None:
        return None
    pm, pl, y = c["p_mean"], c["p_low"], c["outcomes"].astype(float)
    r = dict(mouse_id=mouse, n_whisker=len(pm), n_cut=c["n_cut"])
    pa = c["p_auditory"] if np.isfinite(c["p_auditory"]).all() else None
    for k in KS:
        r[f"S_mean_{k}"], i = win_max(pm, k)
        if k == 20:
            r["T_cross"] = i
        r[f"S_low_{k}"], _ = win_max(pl, k)
        r[f"S_raw_{k}"], _ = win_max(y, k)
        if pa is not None:
            kk = min(k, len(pm))
            num = np.convolve(pm, np.ones(kk) / kk, mode="valid")
            den = np.convolve(pa, np.ones(kk) / kk, mode="valid")
            r[f"S_rel_{k}"] = float(np.max(num / np.maximum(den, 0.05)))
    return r


def best_threshold(x, y):
    best = (-1, None)
    for th in np.unique(x):
        pred = x <= th
        ba = 0.5 * (pred[y].mean() + (~pred[~y]).mean())
        if ba > best[0]:
            best = (ba, th)
    return best


def main():
    pm = pd.read_csv(RES / "tables" / "001_per_mouse.csv")
    M = pd.DataFrame([m for m in (metrics(x) for x in pm.mouse_id) if m is not None])
    d = pm[["mouse_id", "session_id", "cohort", "learning_category", "learner", "hit_w_day1", "dprime_day1"]].merge(M, on="mouse_id")
    d["group"] = np.where(d.cohort == "R-", "R-", np.where(d.mouse_id.isin(FIXED_BAD), "known bad",
                                                         np.where(d.mouse_id.isin(FLEX), "flexible", d.learning_category)))
    d.to_csv(RES / "tables" / f"006_sustained_per_mouse{TAG}.csv", index=False)
    metr = [c for c in M.columns if c.startswith("S_")]
    rp = d[d.cohort == "R+"]
    t = rp[rp.group.isin(["good", "moderate", "known bad"])]
    y = (t.group == "known bad").to_numpy()
    sd_all = rp[metr].std()
    rows = []
    for f in metr:
        x = t[f].to_numpy(float)
        auc = 1 - roc_auc_score(y, x)          # known bad low
        ba, th = best_threshold(x, y)
        loo = []
        for i in range(len(x)):
            m = np.ones(len(x), bool)
            m[i] = False
            _, th_i = best_threshold(x[m], y[m])
            loo.append((x[i] <= th_i) == y[i])
        margin = (x[~y].min() - x[y].max()) / sd_all[f]
        lab = rp[f] > th
        flex = rp[rp.group == "flexible"].set_index("mouse_id")[f]
        rm = d[d.cohort == "R-"][f]
        ok1 = rp.hit_w_day1.notna()
        ok2 = rp.dprime_day1.notna()
        rows.append(dict(metric=f, family=f.split("_")[1], k=int(f.split("_")[2]), auc=auc, threshold=th,
                         balanced_acc=ba, loo_acc=float(np.mean(loo)), margin_sd=margin,
                         confusers=",".join(t.mouse_id[(x <= th) & ~y]), missed=",".join(t.mouse_id[(x > th) & y]),
                         n_learner_rplus=int(lab.sum()), n_rplus=len(rp),
                         flexible_above=",".join(flex.index[flex > th]), flexible_below=",".join(flex.index[flex <= th]),
                         rminus_above=float((rm > th).mean()),
                         kappa_consensus=float(cohen_kappa_score(lab, rp.learner.astype(bool))),
                         auc_w_day1=roc_auc_score(lab[ok1], rp.hit_w_day1[ok1]) if lab[ok1].nunique() == 2 else np.nan,
                         auc_dprime_day1=roc_auc_score(lab[ok2], rp.dprime_day1[ok2]) if lab[ok2].nunique() == 2 else np.nan))
    S = pd.DataFrame(rows).sort_values(["margin_sd"], ascending=False)
    S.to_csv(RES / "tables" / f"006_sustained_summary{TAG}.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).to_string(index=False))
    print(d[d.group.isin(["known bad", "flexible"])][["mouse_id", "group", "S_mean_20", "S_low_20", "S_raw_20", "S_rel_20",
                                                      "T_cross", "n_whisker"]].round(2).to_string(index=False))

    fams = ["mean", "low", "raw", "rel"]
    flab = {"mean": "Whisker curve P(lick)", "low": "Lower 80 % bound", "raw": "Raw licked fraction",
            "rel": "Whisker / auditory P(lick)"}
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.6), gridspec_kw=dict(hspace=0.75, wspace=0.5))
    order = ["good", "moderate", "flexible", "known bad", "R-"]
    for j, fam in enumerate(fams):
        ax = axs[0, j]
        f = f"S_{fam}_20"
        for i, g in enumerate(order):
            v = d[d.group == g]
            x = i + np.random.default_rng(i).uniform(-0.15, 0.15, len(v))
            ax.scatter(x, v[f], s=5, color=GCOL[g], lw=0)
            if g in ("known bad", "flexible"):
                for xx, yy, m in zip(x, v[f], v.mouse_id):
                    ax.annotate(m, (xx, yy), fontsize=3.3, xytext=(2, 0), textcoords="offset points")
        r = S[S.metric == f].iloc[0]
        ax.axhline(r.threshold, color="tab:red", lw=0.6, ls="--")
        ax.set_xticks(range(len(order)), ["good", "mod.", "flex.", "bad", "R−"], fontsize=5)
        ax.set_title(f"{flab[fam]}, best 20-trial window\nAUC {r.auc:.2f}, margin {r.margin_sd:+.2f} SD, LOO {r.loo_acc:.2f}",
                     fontsize=5.3)
        ax.tick_params(labelsize=5)
    for j, (col, lab) in enumerate((("margin_sd", "Separation margin (SD)"), ("loo_acc", "Leave-one-out accuracy"),
                                    ("kappa_consensus", "κ with consensus label"), ("auc_dprime_day1", "AUC, d′ day +1"))):
        ax = axs[1, j]
        for i, fam in enumerate(fams):
            v = S[S.family == fam].set_index("k").reindex(KS)[col]
            ax.plot(KS, v, marker="o", ms=2.5, lw=0.8, label=flab[fam])
        ax.set_xticks(KS)
        ax.set_xlabel("Window (whisker trials)", fontsize=5)
        ax.set_title(lab, fontsize=5.5)
        ax.tick_params(labelsize=5)
        if col == "margin_sd":
            ax.axhline(0, color="0.6", lw=0.4)
        if j == 0:
            ax.legend(fontsize=4.2, frameon=False)
    fig.suptitle(f"Sustained whisker licking on day 0 (exact curves, last {100 - int(round(KEEP * 100))} % of the session cut,"
                 " refitted): known R+ non-learners (black) vs good / moderate; "
                 "flexible mice (orange) and R− (magenta) not used to set the threshold (dashed). Exploratory.",
                 fontsize=6, y=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"006_sustained_metrics{TAG}.{ext}", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
