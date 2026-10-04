"""009 -- Population-level comparison of stored vs re-estimated curves/LTs,
the false-alarm question, and behavior aligned to the new learning trials
(user request 2026-09-25).

Figure 1: 009_population_comparison.png
  a  random-walk step size: stored prior (~1) vs data-chosen ('eb'), per session
  b  log10 Bayes factor, data-chosen vs stored prior
  c  max |FA shift| between the stored even-grid FA and FA at real trial times
  d  stored LT vs new LT (lt_lenient_clean), colored by which rule produced it;
     sessions without a new LT shown in the bottom strip
  e  number of sessions with an LT per rule, and how many pass the clean-
     separation gate (007: 20 trials either side, change > 0.05 in whisker
     rate OR whisker-FA discrimination, P > 0.9)
  f  Q1: FA and whisker lick rate early (first 20% of whisker trials) vs
     middle (20-80%), R+ vs R-
Figure 2: 009_aligned_performance.png -- behavior aligned to the learning trial
  (stored vs lt_lenient_clean), R+ and R- rows: whisker hit rate (cohort
  color), FA rate (no-stim licks, grey) and auditory hit rate (blue), in
  BIN-trial bins of whisker-trial index relative to the LT (no-stim and
  auditory trials placed at the whisker index of their time); session means
  then mean +- SEM across sessions; thin lines = individual sessions (whisker).
  Bottom strip: sessions contributing per bin.
Run on haas (needs the trials table for auditory trials).
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
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_timeresolved_decoding import _active_trials_from_whisker_onset_for_curve  # noqa: E402

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
SRC_COLOR = {"L6": "#1f77b4", "L6_len": "#17becf", "L5w_len": "#ff7f0e", "L7": "#8c564b"}
BIN = 5
REL = (-40, 60)


def population(df, curves_sum, fa):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    rng = np.random.default_rng(0)
    ax = axes[0, 0]
    for k, rg in enumerate(("R+", "R-")):
        g = curves_sum[curves_sum.reward_group == rg]
        for j, (col_, lab) in enumerate((("sigma_orig", "stored prior"), ("sigma_eb", "data-chosen"))):
            xs = k * 2.5 + j + rng.uniform(-0.15, 0.15, len(g))
            ax.scatter(xs, g[col_], s=12, color=COHORT_COLOR[rg], alpha=0.35 if j == 0 else 0.8)
            ax.plot([k * 2.5 + j - 0.3, k * 2.5 + j + 0.3], [g[col_].median()] * 2, color="k")
            ax.text(k * 2.5 + j, g[col_].median() * 1.25, f"{g[col_].median():.2f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks([0, 1, 2.5, 3.5], ["R+ prior", "R+ data", "R- prior", "R- data"])
    ax.set_ylabel("random-walk step sigma (logit / trial)")
    ax.set_title("a. curve smoothness: stored prior forces sigma~1;\nthe data choose 2-7x smoother curves", fontsize=9)

    ax = axes[0, 1]
    for rg in ("R+", "R-"):
        v = curves_sum[curves_sum.reward_group == rg].log_bf_eb_vs_orig / np.log(10)
        ax.hist(v, bins=np.arange(-1, 6.5, 0.25), alpha=0.6, color=COHORT_COLOR[rg], label=f"{rg} (n={len(v)})")
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(np.log10(20), color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("log10 Bayes factor, data-chosen vs stored prior")
    ax.set_title(f"b. evidence for smoother curves: > 20:1 (dashed) in "
                 f"{(curves_sum.log_bf_eb_vs_orig > 3).sum()}/{len(curves_sum)} sessions, never < 1:20", fontsize=9)
    ax.legend(frameon=False)

    ax = axes[0, 2]
    for k, rg in enumerate(("R+", "R-")):
        g = curves_sum[curves_sum.reward_group == rg]
        ax.scatter(k + rng.uniform(-0.15, 0.15, len(g)), g.fa_even_vs_time_max, s=14, color=COHORT_COLOR[rg], alpha=0.7)
        ax.plot([k - 0.3, k + 0.3], [g.fa_even_vs_time_max.median()] * 2, color="k")
    ax.set_xticks([0, 1], ["R+", "R-"])
    ax.set_ylabel("max |FA even-grid - FA at real times|")
    ax.set_title("c. FA interpolation bug: stored 'chance' line\nplaced on an even time grid, not at no-stim trial times", fontsize=9)

    ax = axes[1, 0]
    new = df.lt_lenient_clean
    for src, c in SRC_COLOR.items():
        m = (df.lt_lenient_source == src) & new.notna()
        ax.scatter(df.loc[m, "L0_stored"], new[m], s=22, color=c, label=f"{src} (n={m.sum()})",
                   marker="o", edgecolor=[COHORT_COLOR[r] for r in df.loc[m, "reward_group"]], lw=1.2)
    none = df.L0_stored.notna() & new.isna()
    ax.scatter(df.loc[none, "L0_stored"], np.full(none.sum(), -12), s=10, color="#bbbbbb",
               label=f"stored LT, no clean new LT (n={none.sum()})")
    lim = np.nanmax([df.L0_stored.max(), new.max()]) + 5
    ax.plot([0, lim], [0, lim], color="#888888", ls="--", lw=0.8)
    ax.set_xlabel("stored learning trial")
    ax.set_ylabel("new learning trial (lenient, clean-gated)")
    ax.set_title("d. stored vs new LT (edge color = cohort); above the\ndiagonal = new LT later", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    cnt = pd.read_csv(HERE / "007_counts.csv")
    rules = ["L0_stored", "L6", "L6_len", "L5w_len", "L7", "lt_lenient"]
    for k, rg in enumerate(("R+", "R-")):
        c = cnt[cnt.reward_group == rg].set_index("rule").reindex(rules)
        xs = np.arange(len(rules)) + (k - 0.5) * 0.38
        ax.bar(xs, c.n_defined, width=0.36, color=COHORT_COLOR[rg], alpha=0.3, label=f"{rg} LT defined")
        ax.bar(xs, c.n_clean, width=0.36, color=COHORT_COLOR[rg], alpha=0.9, label=f"{rg} passes clean gate")
    ax.set_xticks(range(len(rules)), rules, rotation=25, fontsize=8)
    ax.set_ylabel("sessions")
    ax.set_title("e. sessions with an LT per rule (light) and passing the\nclean-separation gate (dark); stored LTs: "
                 "20/47 R+, 13/34 R- clean", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=2)

    ax = axes[1, 2]
    for k, rg in enumerate(("R+", "R-")):
        g = fa[fa.reward_group == rg]
        for j, (a_, b_, lab, c) in enumerate((("fa_early", "fa_mid", "FA", "#555555"), ("wh_early", "wh_mid", "whisker", COHORT_COLOR[rg]))):
            x0 = k * 2.6 + j * 1.2
            for r_ in g.itertuples():
                ax.plot([x0, x0 + 0.8], [getattr(r_, a_), getattr(r_, b_)], color=c, alpha=0.12, lw=0.7)
            ax.errorbar([x0, x0 + 0.8], [g[a_].mean(), g[b_].mean()], yerr=[g[a_].sem(), g[b_].sem()], color=c, lw=2.2,
                        marker="o", capsize=3)
            ax.text(x0 + 0.4, 0.97, f"{rg} {lab}", ha="center", fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_xticks([0, 0.8, 1.2, 2.0, 2.6, 3.4, 3.8, 4.6], ["early", "mid"] * 4, fontsize=7.5)
    ax.set_ylabel("lick rate")
    ax.set_title("f. Q1: FA stays high in R+ (0.30 -> 0.27) but falls in R-\n(0.21 -> 0.11): R+ discrimination "
                 "penalised by FA", fontsize=9)
    for a in axes.flat:
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Stored vs re-estimated learning curves and learning trials: population summary (88 ephys learning sessions)",
                 fontsize=12)
    fig.savefig(HERE / "009_population_comparison.png", dpi=170)
    fig.savefig(HERE / "009_population_comparison.pdf")


def aligned(df, inputs, trials_tbl):
    edges = np.arange(REL[0], REL[1] + BIN, BIN)
    centers = edges[:-1] + BIN / 2
    versions = [("L0_stored", "stored LT"), ("lt_lenient_clean", "new LT (lenient, clean-gated)")]
    fig, axes = plt.subplots(4, 2, figsize=(13, 13), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1, 0.35, 1, 0.35]))
    for c, (col, vlab) in enumerate(versions):
        for r, rg in enumerate(("R+", "R-")):
            ax, axn = axes[2 * r, c], axes[2 * r + 1, c]
            per = {"whisker": [], "FA": [], "auditory": []}
            for sid, d in inputs.items():
                if d["reward_group"] != rg:
                    continue
                lt = df.set_index("session_id")[col].get(sid, np.nan)
                if pd.isna(lt):
                    continue
                wt = d["w_start"]
                a_ = _active_trials_from_whisker_onset_for_curve(sid, trials_tbl)
                aud = a_[a_.trial_type == "auditory_trial"]
                streams = {"whisker": (np.arange(len(wt)), d["w_outcomes"]),
                           "FA": (np.searchsorted(wt, d["n_start"]), d["n_outcomes"]),
                           "auditory": (np.searchsorted(wt, aud.start_time.to_numpy()), aud.lick_flag.to_numpy())}
                for k, (idx, out) in streams.items():
                    b = np.digitize(idx - lt, edges) - 1
                    ok = (b >= 0) & (b < len(centers))
                    s = pd.Series(out[ok].astype(float)).groupby(b[ok]).mean().reindex(range(len(centers)))
                    per[k].append(s.to_numpy())
            n_sess = len(per["whisker"])
            for k, c_ in (("whisker", COHORT_COLOR[rg]), ("FA", "#555555"), ("auditory", "#1f77b4")):
                M = np.array(per[k]) if per[k] else np.full((1, len(centers)), np.nan)
                if k == "whisker":
                    for row_ in M:
                        ax.plot(centers, row_, color=c_, alpha=0.08, lw=0.7)
                m, se = np.nanmean(M, 0), np.nanstd(M, 0) / np.sqrt(np.sum(~np.isnan(M), 0))
                ax.plot(centers, m, color=c_, lw=2.2, label=f"{k}")
                ax.fill_between(centers, m - se, m + se, color=c_, alpha=0.2, lw=0)
            ax.axvline(0, color="#d62728", lw=1.2)
            ax.set_ylim(-0.02, 1.02)
            ax.set_ylabel("lick rate")
            ax.set_title(f"{rg} -- aligned to {vlab} (n = {n_sess} sessions)", fontsize=10)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            cnt = np.sum(~np.isnan(np.array(per["whisker"])), 0) if per["whisker"] else np.zeros(len(centers))
            axn.bar(centers, cnt, width=BIN * 0.8, color=COHORT_COLOR[rg], alpha=0.6)
            axn.set_ylabel("n sessions", fontsize=8)
            for a in (ax, axn):
                a.spines[["top", "right"]].set_visible(False)
    for a in axes[-1]:
        a.set_xlabel("whisker trial relative to learning trial (0 = learning trial)")
    fig.suptitle(f"Behavior aligned to the learning trial: stored vs new ({BIN}-trial bins; FA = no-stim licks; "
                 f"thin = individual sessions' whisker hit rate)", fontsize=11)
    fig.savefig(HERE / "009_aligned_performance.png", dpi=170)
    fig.savefig(HERE / "009_aligned_performance.pdf")


def main():
    df = pd.read_csv(ART / "007_learning_trials_v2.csv")
    inputs = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    cs = pd.read_csv(ART / "002_summary.csv")
    fa = pd.read_csv(HERE / "007_fa_behavior.csv")
    population(df, cs, fa)
    root = resolve_dataset_dir("ssl_ephys")
    aligned(df, inputs, pd.read_parquet(root / "metadata" / "trials.parquet"))
    print("done")


if __name__ == "__main__":
    main()
