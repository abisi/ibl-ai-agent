"""007 -- Less conservative learning trials + clean-separation gate + the
false-alarm question (user request 2026-09-25: "be a bit less conservative
and have more learning mice ... 1. should I use false alarm too, given that
R+ mice increase their false alarm compared to R-? 2. how to get more mice
considered learners? All the while keeping a cleanliness in the behavioural
separation pre vs post learning").

(A) FA behavior, R+ vs R- (answers Q1 with data): FA (no-stim lick) rate in
    the first 20% vs the middle 20-80% of each session's active trials, and
    whisker rate over the same spans; per cohort.

(B) Candidate LTs, from most to least conservative (all whisker-trial index):
    L6        joint whisker+FA change point, log10 BF > 0.5, strict checks (005).
    L6_len    same posterior, log10 BF > 0, relaxed checks:
              R+ P(learned whisker > learned FA) > 0.9;
              R- P(initial whisker > FA) > 0.8 and discrimination at least halves.
    L5w_len   R+ only: WHISKER-ONLY change point (004 posterior, rise in lick
              rate), log10 BF > 0, FA used only as a FLOOR (learned whisker
              rate > FA in the same span, P > 0.9) -- not as the thing that has
              to change. Rationale (Q1): R+ FA rises with task engagement, so a
              discrimination (whisker - FA) criterion under-detects R+ learners
              whose whisker AND FA licking both rise.
    L7        half-way rule on smooth discrimination curve (006).
    lt_lenient = first available of L6, L6_len, L5w_len (R+), L7.
(C) Clean-separation gate (the user's "clean learning consideration"),
    applied to every candidate, SEP_W whisker trials before vs after the LT
    (FA = no-stim trials in the same time spans), Beta posteriors:
      clean if max(P_hit, P_disc) > SEP_P, where
        P_hit  = P(hit_post > hit_pre + SEP_MIN)            (R+; reversed for R-)
        P_disc = P(disc_post > disc_pre + SEP_MIN), disc = whisker - FA (R+; reversed for R-)
    i.e. behavior must separate cleanly on EITHER the whisker lick rate or the
    whisker-vs-FA discrimination (the latter covers R+ mice that learn by
    dropping FA while whisker licking is already high). Defaults chosen from
    the trade-off table (2026-09-25): SEP_W=20, SEP_MIN=0.05, SEP_P=0.9
    (first version: whisker-only, SEP_W=30, SEP_MIN=0.1 -> kept only 14/34 R+).
    lt_lenient_clean = lt_lenient where clean, else NaN.
Outputs: artifacts/007_learning_trials_v2.csv (006 table + new columns),
         exploratory-analyses/007_fa_behavior.csv, 007_counts.csv
"""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
_s = importlib.util.spec_from_file_location("j005", HERE / "005_joint_changepoint.py")
j005 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(j005)
_s3 = importlib.util.spec_from_file_location("r003", HERE / "003_learning_trial_rules.py")
r003 = importlib.util.module_from_spec(_s3)
_s3.loader.exec_module(r003)

SEP_W, SEP_MIN, SEP_P = 20, 0.05, 0.9


def p_greater(s1, n1, s2, n2, margin, n_draw=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = beta_dist(s1 + 1, n1 - s1 + 1).rvs(n_draw, random_state=rng)
    b = beta_dist(s2 + 1, n2 - s2 + 1).rvs(n_draw, random_state=rng)
    return float(np.mean(a > b + margin))


def clean_sep(S, lt, rg, n_draw=4000, seed=0):
    """Returns (max(P_hit, P_disc), hit_pre, hit_post) -- see module docstring (C)."""
    if lt is None or np.isnan(lt):
        return np.nan, np.nan, np.nan
    lt = int(lt)
    a, b = max(0, lt - SEP_W), min(S.n, lt + SEP_W)
    if lt - a < 3 or b - lt < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    sw1, mw1, sf1, mf1 = S.counts(a, lt)
    sw2, mw2, sf2, mf2 = S.counts(lt, b)
    w1, w2 = (beta_dist(s_ + 1, m_ - s_ + 1).rvs(n_draw, random_state=rng) for s_, m_ in ((sw1, mw1), (sw2, mw2)))
    f1, f2 = (beta_dist(s_ + 1, m_ - s_ + 1).rvs(n_draw, random_state=rng) for s_, m_ in ((sf1, mf1), (sf2, mf2)))
    sign = 1 if rg == 1 else -1
    p_hit = np.mean(sign * (w2 - w1) > SEP_MIN)
    p_disc = np.mean(sign * ((w2 - f2) - (w1 - f1)) > SEP_MIN)
    return float(max(p_hit, p_disc)), sw1 / mw1, sw2 / mw2


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    cp_joint = pickle.load(open(ART / "005_cp_posteriors.pkl", "rb"))
    cp_w = pickle.load(open(ART / "004_cp_posteriors.pkl", "rb"))
    df = pd.read_csv(ART / "006_learning_trials_final.csv").set_index("session_id")

    fa_rows = []
    for sid, d in inputs.items():
        rg = 1 if d["reward_group"] == "R+" else 0
        o, wt, no, nt = d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"]
        S = j005.Streams(o, wt, no, nt)
        n = len(o)
        a, b = int(0.2 * n), int(0.8 * n)
        fa_rows.append(dict(session_id=sid, reward_group=d["reward_group"], fa_early=S.counts(0, max(a, 5))[2] / max(1, S.counts(0, max(a, 5))[3]),
                            fa_mid=S.counts(a, b)[2] / max(1, S.counts(a, b)[3]), wh_early=o[:max(a, 5)].mean(), wh_mid=o[a:b].mean()))

        # L6 lenient from the joint posterior
        r = cp_joint.get(sid)
        l6_len = np.nan
        if r is not None:
            k, j = int(r["median"]), int(r["map_j"])
            if rg == 1:
                ok = r["log10_bf"] > 0 and S.p_above(k, max(j, k + j005.MIN_SEG)) > 0.9
            else:
                ok = (r["log10_bf"] > 0 and S.p_above(0, k) > 0.8 and S.disc(k, n) <= 0.5 * S.disc(0, k))
            l6_len = r["median"] if ok else np.nan
        # whisker-only change point, FA as floor only (R+)
        l5w = np.nan
        rw = cp_w.get(sid)
        if rg == 1 and rw is not None:
            k = int(rw["median"])
            end = min(n, k + 20)
            if rw["log10_bf"] > 0 and S.p_above(k, end) > 0.9:
                l5w = rw["median"]
        df.loc[sid, "L6_len"] = l6_len
        df.loc[sid, "L5w_len"] = l5w
        row = df.loc[sid]
        cands = [("L6", row.L6), ("L6_len", l6_len), ("L5w_len", l5w), ("L7", row.L7)]
        src, val = next(((c, v) for c, v in cands if not pd.isna(v)), ("none", np.nan))
        df.loc[sid, "lt_lenient"] = val
        df.loc[sid, "lt_lenient_source"] = src
        for name in ("L0_stored", "L6", "L6_len", "L5w_len", "L7", "lt_lenient"):
            p, hpre, hpost = clean_sep(S, df.loc[sid, name], rg)
            df.loc[sid, f"{name}_sep_p"] = p
            df.loc[sid, f"{name}_hit_pre{SEP_W}"] = hpre
            df.loc[sid, f"{name}_hit_post{SEP_W}"] = hpost
        clean = df.loc[sid, "lt_lenient_sep_p"] > SEP_P
        df.loc[sid, "lt_lenient_clean"] = val if clean else np.nan
        for k_, v in r003.evaluate(df.loc[sid, "lt_lenient_clean"], o, wt, no, nt, rg).items():
            df.loc[sid, f"lt_lenient_clean_{k_}"] = v
    df = df.reset_index()
    df.to_csv(ART / "007_learning_trials_v2.csv", index=False)

    fa = pd.DataFrame(fa_rows)
    fa.to_csv(HERE / "007_fa_behavior.csv", index=False)
    pd.set_option("display.width", 250)
    print("FA / whisker lick rate, early (first 20% of whisker trials) vs middle (20-80%):")
    print(fa.groupby("reward_group")[["fa_early", "fa_mid", "wh_early", "wh_mid"]].agg(["mean", "median"]).round(3).to_string())

    rows = []
    for rg, g in df.groupby("reward_group"):
        for c in ("L0_stored", "L6", "L6_len", "L5w_len", "L7", "lt_lenient", "lt_lenient_clean"):
            sepcol = f"{c}_sep_p" if c != "lt_lenient_clean" else "lt_lenient_sep_p"
            v = g[c]
            rows.append(dict(reward_group=rg, rule=c, n_defined=int(v.notna().sum()),
                             n_clean=int((g.loc[v.notna(), sepcol] > SEP_P).sum()),
                             median_lt=v.median(), mean_hit_pre=g.loc[v.notna(), f"{c if c != 'lt_lenient_clean' else 'lt_lenient'}_hit_pre{SEP_W}"].mean(),
                             mean_hit_post=g.loc[v.notna(), f"{c if c != 'lt_lenient_clean' else 'lt_lenient'}_hit_post{SEP_W}"].mean()))
    cnt = pd.DataFrame(rows)
    cnt.to_csv(HERE / "007_counts.csv", index=False)
    print(cnt.round(3).to_string())
    print(pd.crosstab(df.reward_group, df.lt_lenient_source))


if __name__ == "__main__":
    main()
