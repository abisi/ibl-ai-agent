"""020 -- Cohort-specific learning-trial rules on sigma = 1.0 curves, tuned per cohort (user request
2026-09-25: "two split methods for R+ and R- separately. For the direction, both cohorts should have
increasing and decreasing whisker respectively. The learning trial identification could also require
different parameters. Maximize the number of learners in both groups, while showing nice behavioural
difference on whisker. The learning trial should be not too early so that there are enough trial
types to do decoding, but if it is early, then it is early.").

Rule family (average-based half-way, robust to the spikes of sigma = 1 curves), signal s(t):
  R+  s = whisker curve w, or D = w - FA (parameter `signal`).
      start = mean of s over the first `start_w` whisker trials; plateau = max of the `hold`-trial
      running mean of s after the start window; need plateau - start >= `min_change`;
      LT = first t with s[t] >= half-way and running-mean(s)[t] >= half-way.
      Optional fallback to the lenient joint change point (L6 lenient) if no LT (`fallback`).
  R-  s = whisker curve w, or D. Need >= `min_hits` whisker hits in the first 20 trials (else
      'immediate learner'). early = mean of s over the first `start_w` trials; floor = min of the
      `hold`-trial running mean after the start window; need early - floor >= `min_change`;
      LT = first t with s[t] <= half-way and running-mean(s)[t] <= half-way.
Gate (same for every setting): whisker lick rate in the 20 whisker trials after the LT minus the 20
  before must be > 0 (R+) / < 0 (R-). Only raw whisker licks enter the gate.
Selection criterion per cohort (revised): every learner must pass the per-session gate (whisker change
  >= GATE_MIN in the cohort direction; R+ also whisker > FA after the LT); among settings, take the one
  with the MOST learners; ties -> larger median |delta whisker|.
LTs are never delayed. Decodability flag per learner: >= 2 whisker hits and >= 2 misses before the LT
  (and after), counted on the curve-aligned whisker trials.
Outputs: artifacts/020_grid.csv (every setting x cohort), artifacts/020_learning_trials_cohort_rules.csv
(chosen setting, per session), exploratory-analyses/020_grid.png, 020_review_Rplus.png,
020_review_Rminus.png, 020_aligned.png
"""

from __future__ import annotations

import itertools
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
GATE_W = 20
# Per-session gate (revised 2026-09-25 after review: the aggregate criterion (median >= 0.2, 80% > 0.1)
# admitted weak individual sessions, e.g. AB082 +0.05, AB080 +0.09, tiny late blips): every learner must
# change its whisker lick rate by >= GATE_MIN in the cohort's direction (20 trials either side), and for
# R+ the whisker rate after the LT must exceed the FA rate in the same span.
GATE_MIN = 0.15

GRID = {
    "R+": dict(signal=["whisker", "D"], start_w=[10, 20], min_change=[0.1, 0.15, 0.2, 0.3], hold=[10, 20],
               fallback=[False, True], min_hits=[0]),
    "R-": dict(signal=["whisker", "D"], start_w=[10, 20], min_change=[0.1, 0.15, 0.2, 0.3], hold=[10, 20],
               fallback=[False], min_hits=[1, 2, 3]),
}


def runmean(x, w):
    c = np.r_[0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    n = len(x) - w + 1
    if n > 0:
        out[:n] = (c[w:] - c[:-w]) / w
    return out


def halfway(s, up, start_w, min_change, hold):
    rm = runmean(s, hold)
    start = s[:start_w].mean()
    tail = rm[start_w:]
    if not np.isfinite(tail).any():
        return np.nan
    end = np.nanmax(tail) if up else np.nanmin(tail)
    if (end - start if up else start - end) < min_change:
        return np.nan
    thr = start + 0.5 * (end - start)
    for t in range(len(s) - hold + 1):
        if (s[t] >= thr and rm[t] >= thr) if up else (s[t] <= thr and rm[t] <= thr):
            return float(t)
    return np.nan


def whisker_change(lt, o):
    if pd.isna(lt):
        return np.nan
    lt = int(lt)
    a, b = max(0, lt - GATE_W), min(len(o), lt + GATE_W)
    if lt - a < 3 or b - lt < 3:
        return np.nan
    return o[lt:b].mean() - o[a:lt].mean()


def apply(setting, rg, d, c, l6len):
    o = d["w_outcomes"]
    s = c["p_mean"] if setting["signal"] == "whisker" else c["p_mean"] - c["fa_time"]
    up = rg == "R+"
    if not up and o[:20].sum() < setting["min_hits"]:
        return np.nan, "immediate"
    lt = halfway(s, up, setting["start_w"], setting["min_change"], setting["hold"])
    src = "halfway"
    if pd.isna(lt) and setting["fallback"] and not pd.isna(l6len):
        lt, src = l6len, "cp_fallback"
    if pd.isna(lt):
        return np.nan, "none"
    dw = whisker_change(lt, o)
    ok = not pd.isna(dw) and (dw >= GATE_MIN if up else dw <= -GATE_MIN)
    if ok and up:
        lt_i = int(lt)
        b = min(len(o), lt_i + GATE_W)
        nt, wt, no = d["n_start"], d["w_start"], d["n_outcomes"]
        m = (nt >= wt[lt_i]) & (nt <= wt[b - 1])
        ok = (not m.any()) or o[lt_i:b].mean() > no[m].mean()
    return (lt, src) if ok else (np.nan, "gate_fail")


def evaluate(setting, rg, sessions, inputs, curves, l6):
    lts, dws = [], []
    for sid in sessions:
        lt, _ = apply(setting, rg, inputs[sid], curves[sid]["sigma1"], l6.get(sid, np.nan))
        lts.append(lt)
        dws.append(whisker_change(lt, inputs[sid]["w_outcomes"]) if not pd.isna(lt) else np.nan)
    lts, dws = np.array(lts), np.array(dws)
    sign = 1 if rg == "R+" else -1
    v = ~np.isnan(lts)
    ch = sign * dws[v]
    return dict(n_learners=int(v.sum()), median_dw=float(np.median(ch)) if v.any() else np.nan,
                frac_dw_gt_0p1=float(np.mean(ch > 0.1)) if v.any() else np.nan, mean_dw=float(np.mean(ch)) if v.any() else np.nan,
                median_lt=float(np.median(lts[v])) if v.any() else np.nan)


def decodable(lt, o):
    if pd.isna(lt):
        return np.nan
    lt = int(lt)
    pre, post = o[:lt], o[lt:]
    return bool(min(pre.sum(), (1 - pre).sum(), post.sum(), (1 - post).sum()) >= 2)


def main():
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    curves = pickle.load(open(ART / "018_curves_sigma1.pkl", "rb"))
    l6 = pd.read_csv(ART / "007_learning_trials_v2.csv").set_index("session_id")["L6_len"].to_dict()
    stored = pd.read_csv(ART / "007_learning_trials_v2.csv").set_index("session_id")["L0_stored"].to_dict()
    grid_rows, chosen = [], {}
    for rg, spec in GRID.items():
        sessions = [s for s, d in inputs.items() if d["reward_group"] == rg]
        keys = list(spec)
        for vals in itertools.product(*[spec[k] for k in keys]):
            st = dict(zip(keys, vals))
            grid_rows.append(dict(reward_group=rg, **st, **evaluate(st, rg, sessions, inputs, curves, l6)))
        g = pd.DataFrame([r for r in grid_rows if r["reward_group"] == rg])
        best = g.sort_values(["n_learners", "median_dw"], ascending=False).iloc[0]
        chosen[rg] = {k: best[k] for k in keys}
        print(rg, "chosen:", chosen[rg], "->", best[["n_learners", "median_dw", "frac_dw_gt_0p1", "median_lt"]].to_dict())
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(ART / "020_grid.csv", index=False)

    rows = []
    for sid, d in inputs.items():
        rg = d["reward_group"]
        o = d["w_outcomes"]
        lt, src = apply(chosen[rg], rg, d, curves[sid]["sigma1"], l6.get(sid, np.nan))
        cat = "learner" if not pd.isna(lt) else ("immediate_learner" if src == "immediate" else "non_learner")
        rows.append(dict(session_id=sid, mouse_id=d["mouse_id"], reward_group=rg, learning_category=d["learning_category"],
                         lt_cohort=lt, source=src, category=cat, stored_lt=stored.get(sid, np.nan), early_hits20=int(o[:20].sum()),
                         whisker_change=whisker_change(lt, o), pre_hits=int(o[:int(lt)].sum()) if not pd.isna(lt) else np.nan,
                         pre_misses=int((1 - o[:int(lt)]).sum()) if not pd.isna(lt) else np.nan, decodable=decodable(lt, o),
                         setting=str(chosen[rg])))
    df = pd.DataFrame(rows)
    df.to_csv(ART / "020_learning_trials_cohort_rules.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.crosstab(df.reward_group, df.category))
    print(pd.crosstab(df.reward_group, df.source))
    L = df[df.category == "learner"]
    print(L.groupby("reward_group").agg(n=("lt_cohort", "size"), median_lt=("lt_cohort", "median"),
                                        min_lt=("lt_cohort", "min"), decodable=("decodable", "sum"),
                                        median_dw=("whisker_change", "median")).to_string())

    # grid figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    for ax, rg in zip(axes, ("R+", "R-")):
        g = grid[grid.reward_group == rg]
        ok = g.n_learners >= 0
        ax.scatter(g.n_learners[~ok], g.median_dw[~ok], s=18, color="#bbbbbb", label="fails whisker-difference criterion")
        ax.scatter(g.n_learners[ok], g.median_dw[ok], s=22, color=COHORT_COLOR[rg], label="passes")
        b = g[(g[list(chosen[rg])] == pd.Series(chosen[rg])).all(1)].iloc[0]
        ax.scatter(b.n_learners, b.median_dw, s=160, facecolor="none", edgecolor="k", lw=2, label="chosen")
        ax.axhline(0.2, color="#888888", ls="--", lw=0.8)
        ax.set_xlabel("number of learners")
        ax.set_ylabel("median whisker lick-rate change (20 after - 20 before, sign-correct)")
        ax.set_title(f"{rg}: every rule setting ({len(g)}); chosen = most learners with a clear whisker change\n"
                     + ", ".join(f"{k}={v}" for k, v in chosen[rg].items()), fontsize=9)
        ax.legend(fontsize=8, frameon=False)
    fig.savefig(HERE / "020_grid.png", dpi=150)
    plt.close(fig)

    # review grids
    for rg in ("R+", "R-"):
        g = df[df.reward_group == rg].sort_values(["category", "session_id"])
        ncol = 6
        nrow = int(np.ceil(len(g) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.3 * nrow), constrained_layout=True)
        for ax, row in zip(axes.flat, g.itertuples()):
            d, c = inputs[row.session_id], curves[row.session_id]["sigma1"]
            o = d["w_outcomes"]
            x = np.arange(len(o))
            col = COHORT_COLOR[rg]
            ax.fill_between(x, c["p_low80"], c["p_high80"], color=col, alpha=0.18, lw=0)
            ax.plot(x, c["p_mean"], color=col, lw=1.1)
            ax.plot(x, runmean(c["p_mean"], 20), color="k", lw=1, ls="-.")
            ax.plot(x, c["fa_time"], color="#555555", lw=0.9, ls="--")
            ax.scatter(x, np.where(o == 1, 1.07, -0.08), s=1.5, color="k", marker="|")
            if not pd.isna(row.stored_lt):
                ax.axvline(row.stored_lt, color="#d62728", lw=1)
            if not pd.isna(row.lt_cohort):
                ax.axvline(row.lt_cohort, color="#1f77b4", lw=2.2, ls="-" if row.decodable else ":")
                ax.axvspan(max(0, row.lt_cohort - GATE_W), row.lt_cohort + GATE_W, color="#1f77b4", alpha=0.07)
            ax.set_ylim(-0.12, 1.12)
            ax.set_title(f"{row.session_id[:5]} {row.learning_category} | {row.category}\nLT {row.lt_cohort:.0f} [{row.source}] "
                         f"stored {row.stored_lt:.0f}; dW {row.whisker_change:+.2f}; pre {row.pre_hits:.0f}h/{row.pre_misses:.0f}m"
                         .replace("nan", "-"), fontsize=6.2)
            ax.tick_params(labelsize=6)
        for ax in list(axes.flat)[len(g):]:
            ax.axis("off")
        fig.suptitle(f"{rg}: cohort-specific learning trial (blue; dotted = too few pre-LT hits/misses to decode), stored LT red; "
                     f"sigma = 1 whisker curve (80% band), dash-dot = 20-trial running mean, FA dashed; setting "
                     + ", ".join(f"{k}={v}" for k, v in chosen[rg].items()), fontsize=9)
        fig.savefig(HERE / f"020_review_{rg.replace('+', 'plus').replace('-', 'minus')}.png", dpi=150)
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
                b = np.digitize(idx - row.lt_cohort, edges) - 1
                ok = (b >= 0) & (b < len(cen))
                per[k].append(pd.Series(out[ok].astype(float)).groupby(b[ok]).mean().reindex(range(len(cen))).to_numpy())
        for k, cc in (("whisker", COHORT_COLOR[rg]), ("FA", "#555555")):
            M = np.array(per[k])
            m, se = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0))
            ax.plot(cen, m, color=cc, lw=2.2, label=k)
            ax.fill_between(cen, m - se, m + se, color=cc, alpha=0.2, lw=0)
        ax.axvline(0, color="#1f77b4")
        ax.set_ylim(0, 1)
        ax.set_xlabel("whisker trial relative to learning trial")
        ax.set_ylabel("lick rate")
        ax.set_title(f"{rg} learners (n = {len(per['whisker'])})", fontsize=10)
        ax.legend(frameon=False)
    fig.suptitle("Behaviour aligned to the cohort-specific learning trial (sigma = 1)", fontsize=11)
    fig.savefig(HERE / "020_aligned.png", dpi=150)


if __name__ == "__main__":
    main()
