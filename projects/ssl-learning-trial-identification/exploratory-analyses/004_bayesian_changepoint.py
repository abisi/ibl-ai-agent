"""004 -- Bayesian change-point learning trial (L5) + per-session review grid.

Left-to-right segment model on the whisker outcome sequence (the discrete
"HMM" view of learning: a few behavioral states visited in order), with
segment lick rates integrated out analytically (Beta(1,1) priors -> Beta-
function marginal likelihoods) and a uniform prior over change positions:
  R+ : naive (rate a) -> learned (rate b > a) -> optional end-of-session
       decline (rate c < b). LT = start of the learned segment.
       Null model: flat, or flat -> decline (no rise).
  R- : generalising (rate a) -> learned (rate b < a). LT = start of b.
       Null model: flat.
Rate ordering is imposed on the segment MLEs (configurations violating it get
zero prior weight). Segments >= MIN_SEG trials.
Outputs per session:
  L5_cp_median / L5_ci05 / L5_ci95 : posterior median and 90% credible
      interval of the LT (whisker trial index, as the stored LT);
  L5_map                           : posterior mode;
  L5_log10_bf                      : log10 Bayes factor, change model vs null;
  FA checks (Beta posteriors, FA = no-stim outcomes within the segment's
  time span): R+ P(b > fa); R- P(a > fa) (initial generalised licking) and
  P(b > fa + 0.1);
  L5_category: 'learner' (log10 BF > 0.5 and FA check passed),
      R+ 'high_from_start' (no rise but whisker > FA already in the first
      MIN_SEG*2 trials), R- 'never_licked' (initial segment not above FA),
      otherwise 'no_clear_transition' -> L5 LT set to NaN.
Evaluation metrics as in 003 (`evaluate`).
Outputs: artifacts/004_learning_trials.csv (all rules L0-L5 + metrics),
         exploratory-analyses/004_review_grid_<R+|R->.png
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import importlib.util  # noqa: E402

_s = importlib.util.spec_from_file_location("r003", HERE / "003_learning_trial_rules.py")
r003 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r003)

ART = HERE.parent / "artifacts"
MIN_SEG = 5
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def seg_ml(cs, a, b):
    s, m = cs[b] - cs[a], b - a
    return betaln(s + 1, m - s + 1)


def bayes_cp(o: np.ndarray, rg: int):
    n = len(o)
    cs = np.r_[0, np.cumsum(o)]
    rate = lambda a, b: (cs[b] - cs[a]) / (b - a)  # noqa: E731
    logs, ks = [], []
    if rg == 1:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            for j in list(range(k + MIN_SEG, n - MIN_SEG + 1)) + [n]:
                if rate(k, j) <= rate(0, k):
                    continue
                if j < n and rate(j, n) >= rate(k, j):
                    continue
                logs.append(seg_ml(cs, 0, k) + seg_ml(cs, k, j) + (seg_ml(cs, j, n) if j < n else 0.0))
                ks.append(k)
        null = [seg_ml(cs, 0, n)] + [seg_ml(cs, 0, j) + seg_ml(cs, j, n) for j in range(MIN_SEG, n - MIN_SEG + 1)
                                     if rate(j, n) < rate(0, j)]
    else:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            if rate(k, n) >= rate(0, k):
                continue
            logs.append(seg_ml(cs, 0, k) + seg_ml(cs, k, n))
            ks.append(k)
        null = [seg_ml(cs, 0, n)]
    if not logs:
        return None
    logs, ks = np.asarray(logs), np.asarray(ks)
    post = np.exp(logs - logsumexp(logs))
    pk = np.bincount(ks, weights=post, minlength=n)
    cdf = np.cumsum(pk)
    log_bf = (logsumexp(logs) - np.log(len(logs))) - (logsumexp(null) - np.log(len(null)))
    return dict(pk=pk, median=float(np.searchsorted(cdf, 0.5)), ci05=float(np.searchsorted(cdf, 0.05)),
                ci95=float(np.searchsorted(cdf, 0.95)), map=float(np.argmax(pk)), log10_bf=float(log_bf / np.log(10)))


def seg_vs_fa(o, w_t, n_out, n_t, a, b, margin=0.0, n_draw=4000, seed=0):
    """P(whisker rate in [a,b) > FA rate in the same time span + margin)."""
    rng = np.random.default_rng(seed)
    if b - a < 1:
        return np.nan
    fa = n_out[(n_t >= w_t[a]) & (n_t <= w_t[b - 1])]
    sw, mw = o[a:b].sum(), b - a
    wd = beta_dist(sw + 1, mw - sw + 1).rvs(n_draw, random_state=rng)
    fd = beta_dist(fa.sum() + 1, len(fa) - fa.sum() + 1).rvs(n_draw, random_state=rng)
    return float(np.mean(wd > fd + margin))


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    prev = pd.read_csv(ART / "003_learning_trials.csv").set_index("session_id")
    rows, cps = [], {}
    for sid, d in inputs.items():
        rg = 1 if d["reward_group"] == "R+" else 0
        o, wt, no, nt = d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"]
        r = bayes_cp(o, rg)
        row = prev.loc[sid].to_dict()
        row["session_id"] = sid
        cps[sid] = r
        if r is None:
            row.update(L5_category="no_clear_transition", L5=np.nan)
        else:
            k = int(r["median"])
            row.update(L5_cp_median=r["median"], L5_ci05=r["ci05"], L5_ci95=r["ci95"], L5_map=r["map"],
                       L5_log10_bf=r["log10_bf"])
            if rg == 1:
                end = min(len(o), k + 20)
                row["L5_p_learned_above_fa"] = seg_vs_fa(o, wt, no, nt, k, end)
                row["L5_p_start_above_fa"] = seg_vs_fa(o, wt, no, nt, 0, min(len(o), 2 * MIN_SEG))
                ok = r["log10_bf"] > 0.5 and row["L5_p_learned_above_fa"] > 0.95
                cat = "learner" if ok else ("high_from_start" if row["L5_p_start_above_fa"] > 0.95 else "no_clear_transition")
            else:
                row["L5_p_initial_above_fa"] = seg_vs_fa(o, wt, no, nt, 0, k)
                row["L5_p_learned_above_fa_by0.1"] = seg_vs_fa(o, wt, no, nt, k, min(len(o), k + 20), margin=0.1)
                ok = r["log10_bf"] > 0.5 and row["L5_p_initial_above_fa"] > 0.95 and row["L5_p_learned_above_fa_by0.1"] < 0.95
                cat = "learner" if ok else ("never_licked" if row["L5_p_initial_above_fa"] <= 0.95 else "no_clear_transition")
            row["L5_category"] = cat
            row["L5"] = r["median"] if cat == "learner" else np.nan
        for k_, v in r003.evaluate(row["L5"], o, wt, no, nt, rg).items():
            row[f"L5_{k_}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "004_learning_trials.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.crosstab(df.reward_group, df.L5_category))
    for rg, g in df.groupby("reward_group"):
        print(rg, "L5: n", int(g.L5.notna().sum()), "median LT", g.L5.median(), "median CI width",
              (g.L5_ci95 - g.L5_ci05)[g.L5.notna()].median(), "contrast20 %.3f held_post %.3f" % (g.L5_contrast20.mean(), g.L5_held_post.mean()),
              "| shift vs stored (median, where both):", (g.L5 - g.L0_stored).median())
    pickle.dump(cps, open(ART / "004_cp_posteriors.pkl", "wb"))

    for rg_name in ("R+", "R-"):
        g = df[df.reward_group == rg_name].sort_values("L5_category")
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), constrained_layout=True)
        for ax, row in zip(axes.flat, g.itertuples()):
            sid = row.session_id
            d, e = inputs[sid], curves[sid]["eb"]
            o = d["w_outcomes"]
            x = np.arange(len(o))
            col = COHORT_COLOR[rg_name]
            ax.fill_between(x, e["p_low80"], e["p_high80"], color=col, alpha=0.2, lw=0)
            ax.plot(x, e["p_mean"], color=col, lw=1.4)
            ax.plot(x, d["stored"]["p_mean"], color=col, lw=0.6, alpha=0.5)
            ax.plot(x, e["fa_time"], color="#444444", lw=1, ls="--")
            ax.scatter(x, np.where(o == 1, 1.07, -0.07), s=2, color="k", marker="|")
            r = cps[sid]
            if r is not None:
                ax.fill_between(x, 0, r["pk"] / r["pk"].max() * 0.25, color="#1f77b4", alpha=0.35, lw=0, step="mid")
            for val, c_, ls, lab in ((row.L0_stored, "#d62728", "-", "stored"), (row.L3_sustain, "#ff7f0e", "--", "L3"),
                                     (row.L5, "#1f77b4", "-", "L5")):
                if not pd.isna(val):
                    ax.axvline(val, color=c_, lw=1.3, ls=ls)
            if not pd.isna(row.L5):
                ax.axvspan(row.L5_ci05, row.L5_ci95, color="#1f77b4", alpha=0.1, lw=0)
            ax.set_ylim(-0.12, 1.12)
            ax.set_title(f"{sid[:5]} {row.learning_category} | {row.L5_category}\nstored={row.L0_stored:.0f} "
                         f"L3={row.L3_sustain:.0f} L5={row.L5:.0f} [{row.L5_ci05:.0f},{row.L5_ci95:.0f}] "
                         f"log10BF={row.L5_log10_bf:.1f}".replace("nan", "-"), fontsize=6.5)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(g):]:
            ax.axis("off")
        fig.suptitle(f"{rg_name}: whisker learning curves re-estimated. Thick = exact posterior with data-chosen smoothness "
                     f"('eb') + 80% CI; thin = stored curve; dashed = FA curve at actual no-stim times. Red = stored LT, "
                     f"orange dashed = L3 sustained rule, blue = L5 Bayesian change point (shade = 90% CI, bottom = posterior "
                     f"over LT)", fontsize=9)
        fig.savefig(HERE / f"004_review_grid_{rg_name.replace('+', 'plus').replace('-', 'minus')}.png", dpi=150)


if __name__ == "__main__":
    main()
