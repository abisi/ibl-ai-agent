"""ITI signal screen, step 3 (haas): figures of the highlighted features, in raw units.

One row per feature, five panels:
  a  learning curve: per-session dr = r(feature, delta P(lick)) - mean of the linear-shift null (raw effect beyond
     shared slow drift, correlation units); day 0 vs expert x R+ / R-; median line; Mann-Whitney P day 0 vs expert
  b  upcoming choice: per-session dr of the partial r(feature, lick | time + history), same layout
  c  P(upcoming whisker lick) by within-session feature quintile (raw, not residualised), mean +- SEM over sessions
  d  day 0: feature z-scored within session, aligned to the exact-curve learning trial (5-trial bins), mean +- SEM
     over sessions that reach the criterion
  e  day 0: session mean of the feature vs learning trial (sessions reaching the criterion), per cohort; OLS line
     with 95 % CI band, solid only if P < 0.05; Spearman rho and P in the legend
Inputs: features/<session>.parquet (step 1), screen/per_session_scores.parquet (step 2).
Output: combined_results_ks4/ssl-iti-signal-screen/screen/highlighted_features.{png,pdf}
Run on haas: nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 003_highlight_figures.py
"""

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

BASE = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-iti-signal-screen")
OUT = BASE / "screen"
FEATURES = {  # feature: label
    "piezo_licks|iti|rate": "ITI lick rate (piezo, licks/s)",
    "piezo_licks|postprev|count": "Licks 0-2 s after previous trial",
    "jaw_velocity|pre0.2|mean": "Jaw speed, -0.2-0 s",
    "nose_distance|pre0.2|mean": "Nose position, -0.2-0 s",
    "whisker_angle|pre2|speed": "Whisker speed, -2-0 s",
    "tongue_dlc_licks|iti|count": "Tongue (DLC) licks in ITI",
    "iti|duration": "ITI duration (s)",
}
COLORS = {"R+": "#00B400", "R-": "#C800C8"}
GROUPS = [("day0", "R+"), ("day0", "R-"), ("expert", "R+"), ("expert", "R-")]
ALIGN_BINS = np.arange(-30, 35, 5)


def load():
    S = pd.read_parquet(OUT / "per_session_scores.parquet")
    S = S[S.cohort.isin(COLORS) & S.feature.isin(FEATURES)].copy()
    S["stage"] = np.where(S.day == 0, "day0", np.where(S.day > 0, "expert", "other"))
    cols = ["session_id", "cohort", "day", "lick", "learning_trial", "trial"] + list(FEATURES)
    T = []
    for p in glob.glob(str(BASE / "features" / "*.parquet")):
        d = pd.read_parquet(p)
        T.append(d[[c for c in cols if c in d.columns]])
    T = pd.concat(T, ignore_index=True)
    T = T[T.cohort.isin(COLORS)]
    T["stage"] = np.where(T.day == 0, "day0", np.where(T.day > 0, "expert", "other"))
    return S[S.stage != "other"], T[T.stage != "other"]


def strip(ax, S, f, col, title):
    rng = np.random.default_rng(0)
    for i, (st, coh) in enumerate(GROUPS):
        v = S[(S.feature == f) & (S.stage == st) & (S.cohort == coh)][col].dropna()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, alpha=0.45, linewidths=0.6,
                   facecolors=COLORS[coh] if st == "day0" else "none", edgecolors=COLORS[coh])
        ax.plot([i - 0.28, i + 0.28], [v.median()] * 2, color="k", lw=2)
    for j, coh in enumerate(("R+", "R-")):
        a = S[(S.feature == f) & (S.stage == "day0") & (S.cohort == coh)][col].dropna()
        b = S[(S.feature == f) & (S.stage == "expert") & (S.cohort == coh)][col].dropna()
        if len(a) > 3 and len(b) > 3:
            p = stats.mannwhitneyu(a, b).pvalue
            ax.text(j + 1.5, np.nanmax(np.r_[a, b]) + 0.02,
                    f"P={p:.2g}", ha="center", fontsize=6)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["d0 R+", "d0 R-", "exp R+", "exp R-"], fontsize=7)
    ax.set_ylabel(title, fontsize=7)


def quintiles(ax, T, f):
    for st, coh in GROUPS:
        g = T[(T.stage == st) & (T.cohort == coh)]
        rows = []
        for sid, h in g.groupby("session_id"):
            x = h[f].to_numpy(float)
            ok = np.isfinite(x)
            if ok.sum() < 25 or np.unique(x[ok]).size < 5:
                continue
            q = pd.qcut(pd.Series(x[ok]).rank(method="first"), 5, labels=False)
            rows.append(pd.Series(h.lick.to_numpy()[ok]).groupby(q.to_numpy()).mean().reindex(range(5)).to_numpy())
        if not rows:
            continue
        R = np.array(rows)
        m, se = np.nanmean(R, 0), np.nanstd(R, 0) / np.sqrt(np.sum(np.isfinite(R), 0))
        ls = "-" if st == "day0" else "--"
        ax.errorbar(np.arange(1, 6), m, se, color=COLORS[coh], ls=ls, lw=1.5, marker="o", ms=3,
                    mfc=COLORS[coh] if st == "day0" else "white", capsize=0, label=f"{st} {coh} (n={len(R)})")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("feature quintile (within session)", fontsize=7)
    ax.set_ylabel("P(upcoming whisker lick)", fontsize=7)


def aligned(ax, T, f):
    g = T[(T.stage == "day0") & np.isfinite(T.learning_trial)]
    for coh in ("R+", "R-"):
        rows = []
        for sid, h in g[g.cohort == coh].groupby("session_id"):
            x = h[f].to_numpy(float)
            if np.isfinite(x).sum() < 25 or np.nanstd(x) == 0:
                continue
            zx = (x - np.nanmean(x)) / np.nanstd(x)
            rel = h.trial.to_numpy() - h.learning_trial.iloc[0]
            b = np.digitize(rel, ALIGN_BINS) - 1
            rows.append([np.nanmean(zx[b == k]) if np.any(b == k) else np.nan for k in range(len(ALIGN_BINS) - 1)])
        if not rows:
            continue
        R = np.array(rows)
        n = np.sum(np.isfinite(R), 0)
        m, se = np.nanmean(R, 0), np.nanstd(R, 0) / np.sqrt(np.maximum(n, 1))
        x = (ALIGN_BINS[:-1] + ALIGN_BINS[1:]) / 2
        m[n < 5] = np.nan
        ax.plot(x, m, color=COLORS[coh], lw=1.5, label=f"{coh} (n={len(R)})")
        ax.fill_between(x, m - se, m + se, color=COLORS[coh], alpha=0.2, lw=0)
    ax.axvline(0, color="0.5", lw=0.8, ls=":")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("whisker trials from learning trial (day 0)", fontsize=7)
    ax.set_ylabel("feature (z within session)", fontsize=7)


def ols_band(ax, x, y, color, label):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6:
        return
    rho, p_rho = stats.spearmanr(x, y)
    lr = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 50)
    n = len(x)
    resid = y - (lr.intercept + lr.slope * x)
    s = np.sqrt(np.sum(resid ** 2) / (n - 2))
    se = s * np.sqrt(1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    t = stats.t.ppf(0.975, n - 2)
    yfit = lr.intercept + lr.slope * xs
    ax.scatter(x, y, s=10, color=color, alpha=0.6, linewidths=0)
    ax.plot(xs, yfit, color=color, lw=1.5, ls="-" if lr.pvalue < 0.05 else "--",
            label=f"{label}: rho={rho:.2f}, P={p_rho:.2g} (n={n})")
    ax.fill_between(xs, yfit - t * se, yfit + t * se, color=color, alpha=0.15, lw=0)


def session_scatter(ax, T, f):
    g = T[(T.stage == "day0") & np.isfinite(T.learning_trial)]
    sess = g.groupby("session_id").agg(x=(f, "mean"), ltrial=("learning_trial", "first"), cohort=("cohort", "first"))
    for coh in ("R+", "R-"):
        h = sess[sess.cohort == coh]
        ols_band(ax, h.x.to_numpy(float), h.ltrial.to_numpy(float), COLORS[coh], coh)
    ax.set_xlabel("session mean (day 0)", fontsize=7)
    ax.set_ylabel("learning trial", fontsize=7)


def main():
    S, T = load()
    n = len(FEATURES)
    fig, axs = plt.subplots(n, 5, figsize=(19, 3.0 * n))
    for i, (f, lab) in enumerate(FEATURES.items()):
        strip(axs[i, 0], S, f, "dr_pdelta", "dr, learning curve (delta P)")
        strip(axs[i, 1], S, f, "dr_lick_partial", "dr, upcoming choice | time + history")
        quintiles(axs[i, 2], T, f)
        aligned(axs[i, 3], T, f)
        session_scatter(axs[i, 4], T, f)
        axs[i, 0].set_title(lab, loc="left", fontsize=9, fontweight="bold")
        for ax in axs[i]:
            ax.tick_params(labelsize=6)
            ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            axs[i, 0].set_title(f"{lab}\n(a) learning curve, raw dr", loc="left", fontsize=9, fontweight="bold")
            axs[i, 1].set_title("(b) upcoming choice, raw dr", loc="left", fontsize=9)
            axs[i, 2].set_title("(c) P(lick) by feature quintile", loc="left", fontsize=9)
            axs[i, 3].set_title("(d) day 0, aligned to learning trial", loc="left", fontsize=9)
            axs[i, 4].set_title("(e) day 0, session mean vs learning trial", loc="left", fontsize=9)
        axs[i, 2].legend(fontsize=5, frameon=False)
        axs[i, 3].legend(fontsize=6, frameon=False)
        axs[i, 4].legend(fontsize=5.5, frameon=False)
    fig.text(0.01, 0.002, "dr = r - mean of the linear-shift null (200 shifts, |shift| > 10 trials). Filled markers / "
             "solid lines: day 0; open markers / dashed lines: expert (day > 0). Green R+, magenta R-. Exploratory, "
             "uncorrected; P = Mann-Whitney day 0 vs expert per cohort.", fontsize=7)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"highlighted_features.{ext}", dpi=140)
    summ = S.groupby(["feature", "stage", "cohort"])[["dr_pdelta", "dr_lick_partial", "r_pdelta", "r_lick_partial"]] \
        .median().round(3)
    summ.to_csv(OUT / "highlighted_features_dr_medians.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
