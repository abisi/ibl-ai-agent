"""005 -- Joint whisker + no-stim (FA) Bayesian change point (L6).

004's whisker-only change point misses sessions where whisker licking is
high from the start and learning shows up as the false-alarm rate FALLING
(e.g. MH028/MH030/MH070): discrimination (whisker - FA) is what learning
changes. Here segments are defined in TIME (boundaries at whisker trials),
and each segment has its own whisker rate b_s AND FA rate f_s (no-stim trials
assigned to segments by start time); both integrated out (Beta(1,1)).
  R+ : naive -> learned (discrimination b-f higher, segment MLEs) ->
       optional end-of-session segment (whisker rate lower than learned).
       Null: one segment, or one segment -> decline.
  R- : generalising -> learned (discrimination b-f lower). Null: one segment.
LT = posterior median of the learned segment's first whisker trial
(uniform prior over allowed boundaries, segments >= MIN_SEG whisker trials).
Categories (Beta-posterior checks within the relevant segment):
  R+ learner: log10 BF > 0.5 and P(b2 > f2) > 0.95
     high_from_start: not learner and P(whisker > FA) > 0.95 over the first 20
                      whisker trials (already discriminating, no transition)
  L6_lenient: same checks with log10 BF > 0 (change model merely favoured).
  R- learner: log10 BF > 0.5 and P(b1 > f1) > 0.95 (initial generalised
              licking) and discrimination after <= 0.5 x before (relaxed
              2026-09-25 from "P(b2 > f2 + 0.1) < 0.95", which rejected gradual
              learners)
     never_licked: P(b1 > f1) <= 0.95 (whisker never above FA -> nothing to unlearn)
  otherwise no_clear_transition (L6 = NaN).
Outputs: artifacts/005_learning_trials.csv (L0-L6, metrics, provenance),
         artifacts/005_cp_posteriors.pkl, exploratory-analyses/005_review_grid_<R+|R->.png,
         exploratory-analyses/005_rule_summary.csv
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import betaln, logsumexp
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
_s = importlib.util.spec_from_file_location("r003", HERE / "003_learning_trial_rules.py")
r003 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(r003)

ART = HERE.parent / "artifacts"
MIN_SEG = 5
BF_MAIN = 0.5     # log10 Bayes factor, change vs null: > 0.5 = "substantial" (Jeffreys)
BF_LENIENT = 0.0  # > 0 = change model merely favoured (reported as L6_lenient)
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


class Streams:
    def __init__(self, o, wt, no, nt):
        self.n = len(o)
        self.cw = np.r_[0, np.cumsum(o)]
        # no-stim trials before whisker trial k's start (k = 0..n; k = n -> all)
        self.nb = np.r_[np.searchsorted(nt, wt, side="left"), len(nt)]
        self.cf = np.r_[0, np.cumsum(no)]

    def counts(self, a, b):
        sw, mw = self.cw[b] - self.cw[a], b - a
        fa0, fa1 = self.nb[a] if a > 0 else 0, self.nb[b]
        sf, mf = self.cf[fa1] - self.cf[fa0], fa1 - fa0
        return sw, mw, sf, mf

    def ml(self, a, b):
        sw, mw, sf, mf = self.counts(a, b)
        return betaln(sw + 1, mw - sw + 1) + betaln(sf + 1, mf - sf + 1)

    def disc(self, a, b):
        sw, mw, sf, mf = self.counts(a, b)
        return sw / mw - (sf / mf if mf else 0.0)

    def wrate(self, a, b):
        return (self.cw[b] - self.cw[a]) / (b - a)

    def p_above(self, a, b, margin=0.0, n_draw=4000, seed=0):
        sw, mw, sf, mf = self.counts(a, b)
        rng = np.random.default_rng(seed)
        wd = beta_dist(sw + 1, mw - sw + 1).rvs(n_draw, random_state=rng)
        fd = beta_dist(sf + 1, mf - sf + 1).rvs(n_draw, random_state=rng)
        return float(np.mean(wd > fd + margin))


def joint_cp(S: Streams, rg: int):
    n = S.n
    logs, ks, js = [], [], []
    if rg == 1:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            for j in list(range(k + MIN_SEG, n - MIN_SEG + 1)) + [n]:
                if S.disc(k, j) <= S.disc(0, k):
                    continue
                if j < n and S.wrate(j, n) >= S.wrate(k, j):
                    continue
                logs.append(S.ml(0, k) + S.ml(k, j) + (S.ml(j, n) if j < n else 0.0))
                ks.append(k)
                js.append(j)
        null_cfg = [(None, S.ml(0, n))] + [(j, S.ml(0, j) + S.ml(j, n)) for j in range(MIN_SEG, n - MIN_SEG + 1)
                                           if S.wrate(j, n) < S.wrate(0, j)]
    else:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            if S.disc(k, n) >= S.disc(0, k):
                continue
            logs.append(S.ml(0, k) + S.ml(k, n))
            ks.append(k)
            js.append(n)
        null_cfg = [(None, S.ml(0, n))]
    if not logs:
        return None
    logs, ks, js = np.asarray(logs), np.asarray(ks), np.asarray(js)
    post = np.exp(logs - logsumexp(logs))
    pk = np.bincount(ks, weights=post, minlength=n)
    cdf = np.cumsum(pk)
    nl = np.asarray([v for _, v in null_cfg])
    best_null_j = null_cfg[int(np.argmax(nl))][0]
    i_map = int(np.argmax(post))
    return dict(pk=pk, median=float(np.searchsorted(cdf, 0.5)), ci05=float(np.searchsorted(cdf, 0.05)),
                ci95=float(np.searchsorted(cdf, 0.95)), map_k=float(ks[i_map]), map_j=float(js[i_map]),
                log10_bf=float(((logsumexp(logs) - np.log(len(logs))) - (logsumexp(nl) - np.log(len(nl)))) / np.log(10)),
                best_null_j=best_null_j)


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    prev = pd.read_csv(ART / "004_learning_trials.csv").set_index("session_id")
    rows, cps = [], {}
    for sid, d in inputs.items():
        rg = 1 if d["reward_group"] == "R+" else 0
        o, wt, no, nt = d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"]
        S = Streams(o, wt, no, nt)
        r = joint_cp(S, rg)
        cps[sid] = r
        row = prev.loc[sid].to_dict()
        row["session_id"] = sid
        if r is None:
            row.update(L6_category="no_clear_transition", L6=np.nan, L6_lenient=np.nan)
        else:
            k, j = int(r["median"]), int(r["map_j"])
            row.update(L6_cp_median=r["median"], L6_ci05=r["ci05"], L6_ci95=r["ci95"], L6_log10_bf=r["log10_bf"],
                       L6_learned_seg_end=j)
            if rg == 1:
                row["L6_p_learned_disc"] = S.p_above(k, max(j, k + MIN_SEG))
                # 'high_from_start' judged on the FIRST 20 whisker trials only (changed
                # 2026-09-25: the first version used the best null model's first
                # segment, i.e. often the whole session, and so labelled late learners
                # whose FA was initially as high as whisker licking (MH011, AB164) as
                # already discriminating).
                row["L6_p_start_disc"] = S.p_above(0, min(S.n, 20))
                if r["log10_bf"] > BF_MAIN and row["L6_p_learned_disc"] > 0.95:
                    cat = "learner"
                elif row["L6_p_start_disc"] > 0.95:
                    cat = "high_from_start"
                else:
                    cat = "no_clear_transition"
                lenient_ok = r["log10_bf"] > BF_LENIENT and row["L6_p_learned_disc"] > 0.95
            else:
                row["L6_p_initial_disc"] = S.p_above(0, k)
                row["L6_p_learned_disc_by0.1"] = S.p_above(k, S.n, margin=0.1)
                row["L6_disc_before"], row["L6_disc_after"] = S.disc(0, k), S.disc(k, S.n)
                # Relaxed 2026-09-25 after reviewing 005_review_grid_Rminus: the first
                # version required whisker to come within 0.1 of FA after LT, which
                # rejected clear GRADUAL R- learners (AB116, AB159, MH023: log10 BF
                # 1.4-5.5, discrimination halving but still > 0.1). Now: learner if
                # evidence for a change, initially above FA, and discrimination at
                # least halves.
                if row["L6_p_initial_disc"] <= 0.95:
                    cat = "never_licked"
                elif r["log10_bf"] > BF_MAIN and row["L6_disc_after"] <= 0.5 * row["L6_disc_before"]:
                    cat = "learner"
                else:
                    cat = "no_clear_transition"
                lenient_ok = (row["L6_p_initial_disc"] > 0.95 and r["log10_bf"] > BF_LENIENT
                              and row["L6_disc_after"] <= 0.5 * row["L6_disc_before"])
            row["L6_category"] = cat
            row["L6"] = r["median"] if cat == "learner" else np.nan
            row["L6_lenient"] = r["median"] if lenient_ok else np.nan
        for rule in ("L6", "L6_lenient"):
            for k_, v in r003.evaluate(row[rule], o, wt, no, nt, rg).items():
                row[f"{rule}_{k_}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "005_learning_trials.csv", index=False)
    pickle.dump(cps, open(ART / "005_cp_posteriors.pkl", "wb"))
    pd.set_option("display.width", 250)
    print(pd.crosstab(df.reward_group, df.L6_category))
    summ = []
    for rg, g in df.groupby("reward_group"):
        for r_ in ["L0_stored", "L1_exact", "L3_sustain", "L4_cp", "L5", "L6", "L6_lenient"]:
            summ.append(dict(reward_group=rg, rule=r_, n_defined=int(g[r_].notna().sum()), median_lt=g[r_].median(),
                             n_at_10=int((g[r_] == 10).sum()), median_shift_vs_stored=(g[r_] - g.L0_stored).median(),
                             contrast20=g[f"{r_}_contrast20"].mean(), held_post=g[f"{r_}_held_post"].mean(),
                             median_pre_n=g[f"{r_}_pre_n"].median()))
    summ = pd.DataFrame(summ)
    summ.to_csv(HERE / "005_rule_summary.csv", index=False)
    print(summ.round(3).to_string())
    for rg, g in df.groupby("reward_group"):
        both = g[g.L6.notna() & g.L0_stored.notna()]
        print(rg, f"L6 vs stored where both defined (n={len(both)}): later in {(both.L6 > both.L0_stored).sum()}, "
                  f"earlier {(both.L6 < both.L0_stored).sum()}, median shift {(both.L6 - both.L0_stored).median()}, "
                  f"median 90% CI width {(both.L6_ci95 - both.L6_ci05).median()}")

    for rg_name in ("R+", "R-"):
        g = df[df.reward_group == rg_name].sort_values(["L6_category", "session_id"])
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
            ax.plot(x, d["stored"]["p_mean"], color=col, lw=0.6, alpha=0.45)
            ax.plot(x, e["fa_time"], color="#444444", lw=1, ls="--")
            ax.scatter(x, np.where(o == 1, 1.07, -0.07), s=2, color="k", marker="|")
            r = cps[sid]
            if r is not None:
                ax.fill_between(x, 0, r["pk"] / r["pk"].max() * 0.25, color="#1f77b4", alpha=0.35, lw=0, step="mid")
            if not pd.isna(row.L0_stored):
                ax.axvline(row.L0_stored, color="#d62728", lw=1.3)
            if not pd.isna(row.L6):
                ax.axvline(row.L6, color="#1f77b4", lw=1.5)
                ax.axvspan(row.L6_ci05, row.L6_ci95, color="#1f77b4", alpha=0.12, lw=0)
            ax.set_ylim(-0.12, 1.12)
            ax.set_title(f"{sid[:5]} {row.learning_category} | {row.L6_category}\nstored={row.L0_stored:.0f} "
                         f"new={row.L6:.0f} [{row.L6_ci05:.0f},{row.L6_ci95:.0f}] log10BF={row.L6_log10_bf:.1f}".replace("nan", "-"),
                         fontsize=6.5)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(g):]:
            ax.axis("off")
        fig.suptitle(f"{rg_name}: re-estimated whisker learning curves (thick: exact posterior, data-chosen smoothness, 80% CI; "
                     f"thin: stored) with FA curve at actual no-stim times (dashed). Red = stored LT; blue = new LT "
                     f"(joint whisker+FA Bayesian change point, shade = 90% CI, bottom = posterior over LT)", fontsize=9)
        fig.savefig(HERE / f"005_review_grid_{rg_name.replace('+', 'plus').replace('-', 'minus')}.png", dpi=150)


if __name__ == "__main__":
    main()
