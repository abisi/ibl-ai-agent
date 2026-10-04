"""003 -- Alternative learning-trial (LT) rules on the exact curves (002),
evaluated against model-free behavioral criteria.

Rules (index = whisker trial in the curve-aligned set, as the stored LT):
  L0_stored   stored value (40-sample MCMC, sigma~1 prior, even-grid FA,
              5-run rule, floor 10, R- end-of-first-run / fallback 10).
  L1_exact    stored RULE on the exact orig-prior curve with FA at actual
              trial times (isolates MCMC noise + FA-interpolation fix).
  L2_eb_rule  stored RULE on the smoother eb-prior curve (+ FA fix).
  L3_sustain  sustained-probability rule on the eb curve, no floor:
                R+: first t >= first whisker hit with P(p_w > p_FA) >= 0.9 at t
                    and >= 0.9 on >= SUSTAIN_K of the SUSTAIN_W trials from t.
                R-: after the first trial with P(p_w > p_FA) >= 0.9 (initial
                    generalised licking), first t with P(p_w > p_FA) <= 0.5 on
                    >= SUSTAIN_K of the SUSTAIN_W trials from t (whisker
                    licking no longer above FA, and it stays so). If never
                    above FA -> NaN ('no initial whisker licking').
  L4_cp       model-free change point on raw outcomes, relative to FA:
                R+: 3-segment Bernoulli (low -> high -> free), LT = start of
                    the high segment (allows the end-of-session decline).
                R-: 2-segment (high -> low), LT = start of the low segment.
                Whisker outcomes only, segments >= MIN_SEG.
Model-free evaluation of each LT (raw outcomes, FA from no-stim outcomes in
the same time span):
  contrast20   (whisker - FA lick rate) in the 20 whisker trials after LT minus
               the 20 before (R+ should be > 0, R- < 0; larger |.| = sharper)
  held_post    fraction of post-LT 10-trial windows (up to the last 20% of the
               session, to avoid end-of-session decline) where the
               whisker - FA rate is on the learned side (R+ > 0.1, R- <= 0.1)
  pre_n        whisker trials before LT (matters for pre/post decoding power)
Outputs: artifacts/003_learning_trials.csv (one row per session, all rules +
metrics, full provenance), exploratory-analyses/003_rule_summary.png
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
from ssl_timeresolved_decoding import reconstruct_learning_trial  # noqa: E402

ART = HERE.parent / "artifacts"
SUSTAIN_W, SUSTAIN_K = 20, 16
MIN_SEG = 5
RULES = ["L0_stored", "L1_exact", "L2_eb_rule", "L3_sustain", "L4_cp"]


def stored_rule(outcomes, p_mean, p_low, p_high, p_chance, rg):
    cur = dict(outcomes=outcomes, p_mean=p_mean, p_low=p_low, p_high=p_high, p_chance=p_chance, reward_group=rg)
    _, lt, src = reconstruct_learning_trial(cur)
    return lt, src


def sustain_rule(p_above, outcomes, rg):
    n = len(p_above)
    fh = int(np.argmax(outcomes == 1)) if (outcomes == 1).any() else 0
    if rg == 1:
        for t in range(fh, n - SUSTAIN_W + 1):
            if p_above[t] >= 0.9 and (p_above[t:t + SUSTAIN_W] >= 0.9).sum() >= SUSTAIN_K:
                return float(t)
        return np.nan
    above = np.where(p_above >= 0.9)[0]
    if len(above) == 0:
        return np.nan
    for t in range(above[0], n - SUSTAIN_W + 1):
        if (p_above[t:t + SUSTAIN_W] <= 0.5).sum() >= SUSTAIN_K:
            return float(t)
    return np.nan


def _ll(s, m):
    if m == 0:
        return 0.0
    p = s / m
    return 0.0 if p in (0, 1) else s * np.log(p) + (m - s) * np.log(1 - p)


def changepoint(o, rg):
    n = len(o)
    cs = np.r_[0, np.cumsum(o)]
    seg = lambda a, b: _ll(cs[b] - cs[a], b - a)  # noqa: E731
    best, best_k = -np.inf, np.nan
    if rg == 1:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            r1 = (cs[k] - cs[0]) / k
            for j in range(k + MIN_SEG, n + 1):
                r2 = (cs[j] - cs[k]) / (j - k)
                if r2 <= r1:
                    continue
                v = seg(0, k) + seg(k, j) + (seg(j, n) if j < n else 0.0)
                if j < n and n - j < MIN_SEG:
                    continue
                if v > best:
                    best, best_k = v, k
    else:
        for k in range(MIN_SEG, n - MIN_SEG + 1):
            if (cs[n] - cs[k]) / (n - k) >= cs[k] / k:
                continue
            v = seg(0, k) + seg(k, n)
            if v > best:
                best, best_k = v, k
    return float(best_k)


def evaluate(lt, w_out, w_t, n_out, n_t, rg):
    if lt is None or np.isnan(lt):
        return dict(contrast20=np.nan, held_post=np.nan, pre_n=np.nan)
    lt = int(lt)
    n = len(w_out)

    def disc(a, b):
        if b - a < 3:
            return np.nan
        t0, t1 = w_t[a], w_t[min(b, n) - 1]
        fa = n_out[(n_t >= t0) & (n_t <= t1)]
        return w_out[a:b].mean() - (fa.mean() if len(fa) else np.nan)

    before = disc(max(0, lt - 20), lt)
    after = disc(lt, min(n, lt + 20))
    end = int(0.8 * n)
    wins = [disc(a, a + 10) for a in range(lt, max(lt + 1, end - 9), 5)]
    wins = np.array([x for x in wins if not np.isnan(x)])
    held = np.nan if len(wins) == 0 else float(np.mean(wins > 0.1) if rg == 1 else np.mean(wins <= 0.1))
    return dict(contrast20=after - before if not (np.isnan(after) or np.isnan(before)) else np.nan, held_post=held, pre_n=lt)


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "002_curves.pkl", "rb"))
    rows = []
    for sid, d in inputs.items():
        rg = 1 if d["reward_group"] == "R+" else 0
        c = curves[sid]
        wo, st = d["w_outcomes"], d["stored"]
        o, e = c["orig"], c["eb"]
        lts = {"L0_stored": (st["learning_trial"], None)}
        lts["L1_exact"] = stored_rule(wo, o["p_mean"], o["p_low80"], o["p_high80"], o["fa_time"], rg)
        lts["L2_eb_rule"] = stored_rule(wo, e["p_mean"], e["p_low80"], e["p_high80"], e["fa_time"], rg)
        lts["L3_sustain"] = (sustain_rule(e["p_above"], wo, rg), None)
        lts["L4_cp"] = (changepoint(wo, rg), None)
        row = dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=d["reward_group"],
                   learning_category=d["learning_category"], n_whisker=len(wo), stored_mouse_cat=st["mouse_cat"],
                   sigma_eb=e["sigma"])
        for r, (lt, src) in lts.items():
            lt = np.nan if lt is None or pd.isna(lt) else float(lt)
            row[r] = lt
            if src is not None:
                row[f"{r}_source"] = src
            for k, v in evaluate(lt, wo, d["w_start"], d["n_outcomes"], d["n_start"], rg).items():
                row[f"{r}_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ART / "003_learning_trials.csv", index=False)

    pd.set_option("display.width", 250)
    summ = []
    for rg, g in df.groupby("reward_group"):
        for r in RULES:
            summ.append(dict(reward_group=rg, rule=r, n_defined=int(g[r].notna().sum()), median_lt=g[r].median(),
                             n_at_10=int((g[r] == 10).sum()),
                             median_shift_vs_stored=(g[r] - g["L0_stored"]).median(),
                             contrast20_mean=g[f"{r}_contrast20"].mean(), held_post_mean=g[f"{r}_held_post"].mean()))
    summ = pd.DataFrame(summ)
    summ.to_csv(ART / "003_rule_summary.csv", index=False)
    print(summ.round(3).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for i, rg in enumerate(("R+", "R-")):
        g = df[df.reward_group == rg]
        s = summ[summ.reward_group == rg].set_index("rule")
        ax = axes[i, 0]
        for j, r in enumerate(RULES):
            v = g[r].dropna()
            ax.scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.2, 0.2, len(v)), v, s=10, alpha=0.6)
            ax.plot([j - 0.3, j + 0.3], [v.median()] * 2, color="k")
        ax.set_xticks(range(len(RULES)), RULES, rotation=20, fontsize=8)
        ax.set_ylabel("learning trial (whisker trial)")
        ax.set_title(f"{rg}: learning trial per rule (n defined: {', '.join(str(int(s.loc[r, 'n_defined'])) for r in RULES)})",
                     fontsize=9)
        for k, (metric, lab) in enumerate((("contrast20", "whisker-FA rate: 20 after - 20 before"),
                                           ("held_post", "fraction of post windows on learned side"))):
            ax = axes[i, k + 1]
            data = [g[f"{r}_{metric}"].dropna().to_numpy() for r in RULES]
            ax.boxplot(data, showfliers=False)
            for j, v in enumerate(data):
                ax.scatter(np.full(len(v), j + 1) + np.random.default_rng(j).uniform(-0.15, 0.15, len(v)), v, s=8, alpha=0.5)
            ax.set_xticks(range(1, len(RULES) + 1), RULES, rotation=20, fontsize=8)
            ax.axhline(0 if metric == "contrast20" else 0.5, color="#888888", ls=":")
            ax.set_title(f"{rg}: {lab}", fontsize=9)
    fig.suptitle("Learning-trial rules compared (model-free evaluation on raw outcomes). R+ wants contrast > 0 and "
                 "high held_post; R- wants contrast < 0 and high held_post", fontsize=10)
    fig.savefig(HERE / "003_rule_summary.png", dpi=160)


if __name__ == "__main__":
    main()
