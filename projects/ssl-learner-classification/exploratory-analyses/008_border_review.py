"""008 -- Mouse-by-mouse review of the R+ mice at the border of the sustained-whisker-licking rule (user 2026-11-23: "let's
focus on the whisker curve p(lick) sustained licking: let's review each mouse at the border"). Metric S_mean_20 of 006:
max over 20-consecutive-whisker-trial windows of the mean exact whisker curve P(lick), day 0, last 15 % of the session cut
and the curves refitted (exact_cut.py, keep = 0.85). Border = R+ mice with S_mean_20 in [0.25, 0.55] (threshold of 006 at
0.374, between the highest known-bad and the lowest good mouse).

One panel per mouse, ordered by S_mean_20: whisker curve (mean + 80 % band) refitted on the cut session, auditory and
no-stim (chance) curves, raw whisker outcomes (ticks; grey after the cut), best 20-trial window shaded, the cut marked.
Title: group (sheet / known bad / flexible), S_mean_20, S_raw_20 (best licked fraction), whisker hit rate and d' on day +1.
Outputs: tables/008_border_review.csv, figures/008_border_review.png|pdf.
Run (haas, repo root): ~/code/ibl-ai-agent/.venv/bin/python projects/ssl-learner-classification/exploratory-analyses/008_border_review.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exact_cut as EC  # noqa: E402

RES = EC.ROOT / "ssl-learner-classification"
KEEP, K, LO, HI = 0.85, 20, 0.25, 0.55
GCOL = {"good": "#00B400", "moderate": "#7fd17f", "flexible": "tab:orange", "known bad": "k"}


def main():
    d = pd.read_csv(RES / "tables" / "006_sustained_per_mouse_keep85.csv")
    s = pd.read_csv(RES / "tables" / "006_sustained_summary_keep85.csv").set_index("metric")
    th = float(s.loc["S_mean_20", "threshold"])
    b = d[(d.cohort == "R+") & d.S_mean_20.between(LO, HI)].sort_values("S_mean_20").reset_index(drop=True)
    rows = []
    n = len(b)
    nc = 4
    fig, axs = plt.subplots(int(np.ceil(n / nc)), nc, figsize=(7.4, 1.65 * np.ceil(n / nc)),
                            gridspec_kw=dict(hspace=0.9, wspace=0.3), squeeze=False)
    for ax, (_, r) in zip(axs.flat, b.iterrows()):
        c = EC.curves(r.mouse_id, KEEP)
        full = EC._load(r.mouse_id, "whisker_trial")
        yf = np.asarray(full.outcomes).astype(int)
        nk = len(c["p_mean"])
        x = np.arange(nk)
        v = np.convolve(c["p_mean"], np.ones(min(K, nk)) / min(K, nk), mode="valid")
        i0 = int(np.argmax(v))
        ax.axvspan(i0 - 0.5, i0 + min(K, nk) - 0.5, color="0.88", lw=0)
        ax.fill_between(x, c["p_low"], c["p_high"], color=GCOL[r.group], alpha=0.25, lw=0)
        ax.plot(x, c["p_mean"], color=GCOL[r.group], lw=0.9)
        ax.plot(x, c["p_auditory"], color="tab:blue", lw=0.6)
        ax.plot(x, c["p_chance"], color="0.45", lw=0.6)
        xf = np.arange(len(yf))
        cut = xf >= nk
        ax.scatter(xf[~cut], 1.04 + 0.04 * yf[~cut], marker="|", s=4, lw=0.4, color="k")
        ax.scatter(xf[cut], 1.04 + 0.04 * yf[cut], marker="|", s=4, lw=0.4, color="0.7")
        ax.axvline(nk - 0.5, color="tab:red", lw=0.5, ls=":")
        ax.axhline(th, color="0.3", lw=0.4, ls="--")
        ax.set(ylim=(0, 1.12), xlim=(-1, len(yf)))
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=4.5)
        lab = r.group if r.group in ("known bad", "flexible") else r.learning_category
        ax.set_title(f"{r.mouse_id}  ({lab})\nS={r.S_mean_20:.2f}, raw {r.S_raw_20 * K:.0f}/20; d+1 W {r.hit_w_day1:.2f},"
                     f" d′ {r.dprime_day1:.2f}", fontsize=5, color=GCOL[r.group])
        late = c["p_mean"][nk // 2:]
        rows.append(dict(mouse_id=r.mouse_id, group=r.group, sheet=r.learning_category, S_mean_20=r.S_mean_20,
                         S_raw_20=r.S_raw_20, best_window_start=i0, n_whisker_cut=nk, n_cut=c["n_cut"],
                         n_w_licks=int(c["outcomes"].sum()), w_end20=float(c["outcomes"][-K:].mean()),
                         a_end20=float(np.nanmean(c["p_auditory"][-K:])),
                         frac_above_chance=float((c["p_low"] > c["p_chance"]).mean()),
                         late_mean_w=float(late.mean()), hit_w_day1=r.hit_w_day1, dprime_day1=r.dprime_day1,
                         above_threshold=bool(r.S_mean_20 > th)))
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    for ax in axs[:, 0]:
        ax.set_ylabel("P(lick)", fontsize=5)
    for ax in axs[-1]:
        ax.set_xlabel("Whisker trial", fontsize=5)
    fig.suptitle(f"R+ mice at the border of the sustained-licking rule (best 20-trial window of the whisker curve; dashed = "
                 f"threshold {th:.2f}). Green / black / orange = whisker curve (good-moderate / known bad / flexible), blue = "
                 "auditory, grey = no-stim; shaded = best window; red dotted = session-end cut (last 15 %, grey ticks "
                 "dropped). Ticks = raw whisker outcomes. Exploratory.", fontsize=5.5, y=1.0, wrap=True)
    for ext in ("png", "pdf"):
        fig.savefig(RES / "figures" / f"008_border_review.{ext}", dpi=300, bbox_inches="tight")
    T = pd.DataFrame(rows)
    T.to_csv(RES / "tables" / "008_border_review.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"threshold {th:.3f}")
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
