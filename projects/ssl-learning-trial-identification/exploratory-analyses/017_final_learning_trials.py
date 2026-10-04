"""017 -- FINAL behavioural learning-trial definition, frozen before any neural analysis uses it
(user decision 2026-09-25: "do 1 and 2": freeze a behaviour-only definition, then run pre/post
decoding and the placebo test once with it). Defined from licks only; no neural data involved.

R+ (learning = whisker licking separates from FA licking):
  LT = half-way rule on the smoothed separation D = whisker - FA (L7, 006); if undefined, the
  lenient joint whisker+FA change point (L6_len, 007). Light gate: whisker-FA lick rate in the 20
  whisker trials after the LT minus the 20 before must be > 0 (right direction).
R- (learning = whisker licking stops; FA need not be lower than whisker -- fast R- learners often
  lick everything early and stop licking to both unrewarded stimuli):
  requires an initial licking phase: >= MIN_EARLY_HITS whisker hits in the first 20 whisker trials.
  LT = half-way point of the decline of the smoothed whisker curve (002, data-chosen smoothness):
  early level = max over the first 20 whisker trials; floor = 10th percentile after that peak;
  decline must be >= MIN_DROP; LT = first trial after the peak where the curve is <= half-way and
  stays so on 16/20 trials. Light gate: whisker lick rate 20 before minus 20 after > 0.
  R- with < MIN_EARLY_HITS early hits = 'immediate_learner' (no pre-learning epoch; no LT for
  decoding; lt_immediate = trial after the last early hit, behavioural bookkeeping only).
Categories: learner / immediate_learner (R-) / non_learner (no LT or gate failed).
Output: artifacts/017_learning_trials_final.csv (lt_final, category, source, gate values, and
all earlier method values for provenance); exploratory-analyses/017_final_review_<R+|R->.png,
017_final_aligned.png.
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
ART = HERE.parent / "artifacts"
sys.path.insert(0, str(HERE.parents[2] / "scripts"))

MIN_EARLY_HITS = 3
# Curve version (2026-09-25, user: per-session sigma too smooth with few licks -> ONE sigma for the
# dataset, 018): SSL_CURVE_KEY=shared reads artifacts/018_curves_shared.pkl["shared"] and recomputes the
# R+ half-way rule (L7) on those curves; default "eb" = per-session data-chosen sigma (first version).
import os  # noqa: E402
CURVE_KEY = os.environ.get("SSL_CURVE_KEY", "eb")
SUFFIX = "" if CURVE_KEY == "eb" else f"_{CURVE_KEY}"
MIN_DROP = 0.15
W, K, GATE_W = 20, 16, 20
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def rminus_halfway(w):
    i_max = int(np.argmax(w[:20]))
    hi = w[i_max]
    lo = np.percentile(w[i_max:], 10)
    if hi - lo < MIN_DROP:
        return np.nan
    thr = lo + 0.5 * (hi - lo)
    for t in range(i_max, len(w) - W + 1):
        if w[t] <= thr and (w[t:t + W] <= thr).sum() >= K:
            return float(t)
    return np.nan


def gate_values(lt, o, wt, no, nt):
    """(whisker change, whisker-FA change) = after minus before, GATE_W whisker trials each side."""
    if pd.isna(lt):
        return np.nan, np.nan
    lt = int(lt)
    a, b = max(0, lt - GATE_W), min(len(o), lt + GATE_W)
    if lt - a < 3 or b - lt < 3:
        return np.nan, np.nan

    def rates(i0, i1):
        m = (nt >= wt[i0]) & (nt <= wt[i1 - 1])
        return o[i0:i1].mean(), (no[m].mean() if m.any() else np.nan)

    w1, f1 = rates(a, lt)
    w2, f2 = rates(lt, b)
    return w2 - w1, (w2 - f2) - (w1 - f1)


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / ("002_curves.pkl" if CURVE_KEY == "eb" else f"018_curves_{CURVE_KEY}.pkl"), "rb"))
    import importlib.util
    _s = importlib.util.spec_from_file_location("m006", HERE / "006_halfway_rule_and_final_table.py")
    m006 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(m006)
    prev = pd.read_csv(ART / "007_learning_trials_v2.csv").set_index("session_id")
    rows = []
    for sid, d in inputs.items():
        rg = d["reward_group"]
        o, wt, no, nt = d["w_outcomes"], d["w_start"], d["n_outcomes"], d["n_start"]
        e = curves[sid][CURVE_KEY]
        p = prev.loc[sid].copy()
        if CURVE_KEY != "eb":
            p["L7"] = m006.halfway(e["p_mean"] - e["fa_time"], 1 if rg == "R+" else 0)
        row = dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=rg, learning_category=d["learning_category"],
                   L0_stored=p.L0_stored, L6=p.L6, L6_len=p.L6_len, L7=p.L7, L6_category=p.L6_category,
                   early_hits20=int(o[:20].sum()))
        if rg == "R+":
            lt, src = (p.L7, "L7_halfway_D") if not pd.isna(p.L7) else ((p.L6_len, "L6_lenient_cp") if not pd.isna(p.L6_len)
                                                                         else (np.nan, "none"))
            dw, dd = gate_values(lt, o, wt, no, nt)
            ok = not pd.isna(dd) and dd > 0
            cat = "learner" if ok else "non_learner"
            row.update(lt_candidate=lt, source=src, gate_whisker_change=dw, gate_disc_change=dd)
        else:
            if row["early_hits20"] < MIN_EARLY_HITS:
                hits = np.where(o[:20] == 1)[0]
                row.update(lt_candidate=np.nan, source="immediate", lt_immediate=float(hits[-1] + 1) if len(hits) else 0.0)
                cat, ok = "immediate_learner", False
            else:
                lt = rminus_halfway(e["p_mean"])
                dw, dd = gate_values(lt, o, wt, no, nt)
                ok = not pd.isna(dw) and dw < 0
                cat = "learner" if ok else "non_learner"
                row.update(lt_candidate=lt, source="halfway_whisker" if not pd.isna(lt) else "none",
                           gate_whisker_change=dw, gate_disc_change=dd)
        row["category"] = cat
        row["lt_final"] = row["lt_candidate"] if cat == "learner" else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ART / f"017_learning_trials_final{SUFFIX}.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.crosstab(df.reward_group, df.category))
    print(pd.crosstab(df.reward_group, df.source))
    for rg, g in df.groupby("reward_group"):
        L = g[g.category == "learner"]
        print(rg, "learners", len(L), "median LT", L.lt_final.median(), "gate whisker change mean %.3f, disc change mean %.3f"
              % (L.gate_whisker_change.mean(), L.gate_disc_change.mean()))

    # review grids
    for rg in ("R+", "R-"):
        g = df[df.reward_group == rg].sort_values(["category", "session_id"])
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), constrained_layout=True)
        for ax, row in zip(axes.flat, g.itertuples()):
            d, e = inputs[row.session_id], curves[row.session_id][CURVE_KEY]
            o = d["w_outcomes"]
            x = np.arange(len(o))
            col = COHORT_COLOR[rg]
            ax.fill_between(x, e["p_low80"], e["p_high80"], color=col, alpha=0.2, lw=0)
            ax.plot(x, e["p_mean"], color=col, lw=1.4)
            ax.plot(x, e["fa_time"], color="#444444", lw=1, ls="--")
            if rg == "R+":
                ax.plot(x, e["p_mean"] - e["fa_time"], color="#8c564b", lw=0.9, ls="-.")
            ax.scatter(x, np.where(o == 1, 1.07, -0.07), s=2, color="k", marker="|")
            if not pd.isna(row.L0_stored):
                ax.axvline(row.L0_stored, color="#d62728", lw=1)
            if not pd.isna(row.lt_final):
                ax.axvline(row.lt_final, color="#1f77b4", lw=2)
                ax.axvspan(max(0, row.lt_final - GATE_W), min(len(o), row.lt_final + GATE_W), color="#1f77b4", alpha=0.07)
            elif not pd.isna(row.lt_candidate):
                ax.axvline(row.lt_candidate, color="#1f77b4", lw=1.2, ls="--")
            if row.category == "immediate_learner":
                ax.axvspan(0, 20, color="#ffd27f", alpha=0.35, lw=0)
            ax.set_ylim(-0.12, 1.12)
            ax.set_title(f"{row.session_id[:5]} {row.learning_category} | {row.category}\nstored {row.L0_stored:.0f} -> final "
                         f"{row.lt_final:.0f} [{row.source}]; early hits {row.early_hits20}".replace("nan", "-"), fontsize=6.3)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(g):]:
            ax.axis("off")
        extra = ("dash-dot = D = whisker - FA; LT = half-way on D, else lenient change point" if rg == "R+" else
                 "LT = half-way decline of the whisker curve (>= 3 hits in first 20 trials); yellow = first 20 trials of "
                 "immediate learners")
        fig.suptitle(f"{rg} [curves: {CURVE_KEY} sigma]: FINAL learning trial (blue; dashed = candidate failing the direction gate), stored LT red; "
                     f"whisker curve (80% CI), FA dashed; {extra}", fontsize=9)
        fig.savefig(HERE / f"017_final_review_{rg.replace('+', 'plus').replace('-', 'minus')}{SUFFIX}.png", dpi=150)
        plt.close(fig)

    # aligned behaviour
    edges = np.arange(-40, 65, 5)
    cen = edges[:-1] + 2.5
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    for ax, rg in zip(axes, ("R+", "R-")):
        per = {"whisker": [], "FA": []}
        for row in df[(df.reward_group == rg) & (df.category == "learner")].itertuples():
            d = inputs[row.session_id]
            for k, (idx, out) in {"whisker": (np.arange(len(d["w_outcomes"])), d["w_outcomes"]),
                                  "FA": (np.searchsorted(d["w_start"], d["n_start"]), d["n_outcomes"])}.items():
                b = np.digitize(idx - row.lt_final, edges) - 1
                ok = (b >= 0) & (b < len(cen))
                per[k].append(pd.Series(out[ok].astype(float)).groupby(b[ok]).mean().reindex(range(len(cen))).to_numpy())
        for k, c in (("whisker", COHORT_COLOR[rg]), ("FA", "#555555")):
            M = np.array(per[k])
            m, se = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0))
            ax.plot(cen, m, color=c, lw=2.2, label=k)
            ax.fill_between(cen, m - se, m + se, color=c, alpha=0.2, lw=0)
        ax.axvline(0, color="#1f77b4")
        ax.set_ylim(0, 1)
        ax.set_xlabel("whisker trial relative to final learning trial")
        ax.set_ylabel("lick rate")
        ax.set_title(f"{rg} learners (n = {len(per['whisker'])}): behaviour aligned to the final LT", fontsize=10)
        ax.legend(frameon=False)
    fig.savefig(HERE / f"017_final_aligned{SUFFIX}.png", dpi=150)
    print("done")


if __name__ == "__main__":
    main()
