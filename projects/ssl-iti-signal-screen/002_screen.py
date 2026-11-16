"""ITI signal screen, step 2 (haas): score every candidate feature of step 1 on three criteria and rank them, day 0 and
expert (day > 0) sessions separately. Exploratory: effect sizes, uncorrected group P values, no multiple-comparison
correction (user 2026-11-04).

Per session and feature (trials = exact-curve whisker trials; features with > 30 % missing trials in a session are
skipped there, remaining gaps filled with the session median):
  C1 learning curve   r(feature, p_mean) and r(feature, p_mean - p_chance = delta P(lick)); dr = r - mean of the
                      shift null (raw effect beyond shared drift, correlation units); z = dr / SD of the null
                      (significance, comparable across sessions); both against a
                      linear-shift null (N_SHIFT shifts, |shift| > 10 trials, >= 10 trials overlap)
  C2 upcoming choice  partial r(feature, lick | time + history): feature and lick residualised on trial index,
                      index^2, previous lick, reward, whisker lick and log time since lick (OLS); z against a
                      linear-shift null of the residuals; raw r(feature, lick) also stored
Across sessions (one value per session):
  C3 performance      Spearman rho between the session mean of the feature, or its early->late change (last third
                      minus first third, in session-SD units), and performance: mean delta P(lick) over the session
                      (both stages), mean P(lick) of the last 20 trials and the exact-curve learning trial (day 0;
                      sessions that never reach the criterion get the session length). Computed within each cohort
                      (R+ / R- differ strongly in performance) and averaged
Ranking (per stage): E1 = |median z| of C1 (delta P(lick)), E2 = |median z| of C2, E3 = best mean within-cohort |rho|
of C3; combined rank = mean of the three ranks. Group P values: Wilcoxon signed-rank on the session z's.
Outputs: combined_results_ks4/ssl-iti-signal-screen/screen/ (per-session scores, ranking tables, figures).
Run on haas: nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python 002_screen.py --jobs 32
"""

import argparse
import glob
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")
BASE = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-iti-signal-screen")
OUT = BASE / "screen"
META = ["session_id", "mouse_id", "cohort", "day", "learning_trial", "trial", "start_time", "lick", "p_mean",
        "p_chance", "p_low", "p_high", "trial_idx", "prev_lick", "prev_reward", "prev_whisker_lick", "log_since_lick"]
HIST = ["trial_idx", "prev_lick", "prev_reward", "prev_whisker_lick", "log_since_lick"]
N_SHIFT, SHIFT_EXCLUDE, MIN_OVERLAP = 200, 10, 10
COLORS = {"R+": "#00B400", "R-": "#C800C8"}


def corr_cols(X, y):
    """Pearson r between each column of X and y (no NaN)."""
    Xc = X - X.mean(0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc * yc[:, None]).sum(0) / den


def shift_z(X, y, rng):
    """r of each column with y, the linear-shift null mean (y shifted against X), z and P against that null."""
    n = len(y)
    r = corr_cols(X, y)
    shifts = [s for s in range(-n + MIN_OVERLAP, n - MIN_OVERLAP + 1) if abs(s) > SHIFT_EXCLUDE]
    if not shifts:
        nan = np.full_like(r, np.nan)
        return r, nan, nan, nan
    null = []
    for s in rng.choice(shifts, min(N_SHIFT, len(shifts)), replace=False):
        null.append(corr_cols(X[s:], y[:-s]) if s > 0 else corr_cols(X[:s], y[-s:]))
    null = np.array(null)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (r - np.nanmean(null, 0)) / np.nanstd(null, 0)
    p = np.nanmean(np.abs(null) >= np.abs(r), 0)
    return r, np.nanmean(null, 0), z, p


def residual(A, B):
    """Residuals of the columns of A after OLS on B (with intercept)."""
    B1 = np.column_stack([B, np.ones(len(B))])
    beta = np.linalg.lstsq(B1, A, rcond=None)[0]
    return A - B1 @ beta


def session_scores(path):
    d = pd.read_parquet(path)
    rng = np.random.default_rng(abs(hash(d.session_id.iloc[0])) % 2 ** 32)
    feats = [c for c in d.columns if c not in META]
    X = d[feats].to_numpy(float)
    keep = (np.isfinite(X).mean(0) >= 0.7) & (np.nanstd(X, 0) > 0)
    feats, X = [f for f, k in zip(feats, keep) if k], X[:, keep]
    X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
    y = d.lick.to_numpy(float)
    p_mean = d.p_mean.to_numpy(float)
    p_delta = p_mean - d.p_chance.to_numpy(float)
    out = pd.DataFrame({"feature": feats})
    for k, v in d[["session_id", "mouse_id", "cohort", "day"]].iloc[0].items():
        out[k] = v
    out["n_trials"] = len(y)
    out["r_pmean"], out["rnull_pmean"], out["z_pmean"], out["p_pmean"] = shift_z(X, p_mean, rng)
    out["r_pdelta"], out["rnull_pdelta"], out["z_pdelta"], out["p_pdelta"] = shift_z(X, p_delta, rng)
    out["r_lick_raw"] = corr_cols(X, y)
    if 5 <= y.sum() <= len(y) - 5:
        H = d[HIST].to_numpy(float)
        H = np.column_stack([H, d.trial_idx.to_numpy(float) ** 2])
        Xr, yr = residual(X, H), residual(y[:, None], H)[:, 0]
        out["r_lick_partial"], out["rnull_lick_partial"], out["z_lick_partial"], out["p_lick_partial"] =             shift_z(Xr, yr, rng)
    else:
        out["r_lick_partial"] = out["rnull_lick_partial"] = out["z_lick_partial"] = out["p_lick_partial"] = np.nan
    for c in ("pmean", "pdelta", "lick_partial"):  # raw effect beyond drift, in correlation units
        out[f"dr_{c}"] = out[f"r_{c}"] - out[f"rnull_{c}"]
    # session-level summaries for C3
    third = max(len(y) // 3, 1)
    sd = X.std(0)
    out["sess_mean"] = X.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["sess_change"] = (X[-third:].mean(0) - X[:third].mean(0)) / sd
    out["perf_delta"] = p_delta.mean()
    out["perf_final"] = p_mean[-20:].mean()
    lt = d.learning_trial.iloc[0]
    out["learning_trial"] = lt if np.isfinite(lt) else len(y)
    return out


def rank_stage(S, stage):
    perf = ["perf_delta"] + (["perf_final", "learning_trial"] if stage == "day0" else [])
    rows = []
    for f, g in S.groupby("feature"):
        r = dict(feature=f, n_sessions=len(g))
        for c in ("pmean", "pdelta", "lick_partial"):
            z = g[f"z_{c}"].dropna()
            r[f"median_z_{c}"] = z.median()
            r[f"median_r_{c}"] = g[f"r_{c}"].median()
            r[f"median_dr_{c}"] = g[f"dr_{c}"].median()
            for coh in ("R+", "R-"):
                r[f"median_dr_{c}_{coh}"] = g[g.cohort == coh][f"dr_{c}"].median()
            r[f"p_wilcoxon_{c}"] = stats.wilcoxon(z).pvalue if len(z) >= 6 else np.nan
            r[f"frac_sig_{c}"] = (g[f"p_{c}"] < 0.05).mean()
            for coh in ("R+", "R-"):
                r[f"median_z_{c}_{coh}"] = g[g.cohort == coh][f"z_{c}"].median()
        r["median_r_lick_raw"] = g.r_lick_raw.median()
        best = (0.0, None, None)
        for summ in ("sess_mean", "sess_change"):
            for pf in perf:
                rhos = {}
                for coh in ("R+", "R-"):
                    h = g[g.cohort == coh][[summ, pf]].dropna()
                    rhos[coh] = stats.spearmanr(h[summ], h[pf])[0] if len(h) >= 6 else np.nan
                    r[f"rho_{summ}_{pf}_{coh}"] = rhos[coh]
                m = np.nanmean([abs(v) for v in rhos.values()]) if any(np.isfinite(list(rhos.values()))) else np.nan
                if np.isfinite(m) and m > best[0]:
                    best = (m, f"{summ} vs {pf}", rhos)
        r["E3_best_abs_rho"], r["E3_which"] = best[0], best[1]
        r["E3_rho_Rplus"] = best[2]["R+"] if best[2] else np.nan
        r["E3_rho_Rminus"] = best[2]["R-"] if best[2] else np.nan
        rows.append(r)
    R = pd.DataFrame(rows)
    R["E1"] = R.median_z_pdelta.abs()
    R["E2"] = R.median_z_lick_partial.abs()
    R["E3"] = R.E3_best_abs_rho
    for e in ("E1", "E2", "E3"):
        R[f"rank_{e}"] = R[e].rank(ascending=False)
    R["combined_rank"] = R[["rank_E1", "rank_E2", "rank_E3"]].mean(1)
    R["signal"] = R.feature.str.split("|").str[0]
    R["window"] = R.feature.str.split("|").str[1]
    R["stat"] = R.feature.str.split("|").str[2]
    return R.sort_values("combined_rank")


def figure(R, stage, n_top=25):
    top = R.head(n_top)
    fig, axs = plt.subplots(1, 4, figsize=(20, 0.32 * n_top + 2), sharey=False,
                            gridspec_kw=dict(width_ratios=[1, 1, 1, 1.3]))
    y = np.arange(len(top))[::-1]
    for ax, col, lab in ((axs[0], "median_z_pdelta", "C1: z, r(feature, delta P(lick))"),
                         (axs[1], "median_z_lick_partial", "C2: z, partial r(feature, upcoming lick)")):
        for k, coh in enumerate(("R+", "R-")):
            ax.scatter(top[f"{col}_{coh}"], y + (k - 0.5) * 0.25, color=COLORS[coh], s=14, label=coh)
        ax.barh(y, top[col], color="0.8", height=0.6, label="all sessions")
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel(f"median {lab}", fontsize=8)
    axs[0].legend(fontsize=7, frameon=False)
    ax = axs[2]
    ax.scatter(top.E3_rho_Rplus, y + 0.12, color=COLORS["R+"], s=14)
    ax.scatter(top.E3_rho_Rminus, y - 0.12, color=COLORS["R-"], s=14)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("C3: Spearman rho with performance (best summary)", fontsize=8)
    for ax in axs[:3]:
        ax.set_yticks(y)
        ax.set_yticklabels(top.feature if ax is axs[0] else [], fontsize=7)
    ax = axs[3]
    sigcol = {s: plt.cm.tab20(i % 20) for i, s in enumerate(sorted(R.signal.unique()))}
    ax.scatter(R.median_z_pdelta, R.median_z_lick_partial, s=8, c=[sigcol[s] for s in R.signal], alpha=0.7)
    for _, r in top.head(10).iterrows():
        ax.annotate(r.feature, (r.median_z_pdelta, r.median_z_lick_partial), fontsize=5)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("C1 median z (delta P(lick))", fontsize=8)
    ax.set_ylabel("C2 median z (upcoming lick | time + history)", fontsize=8)
    for s, c in sigcol.items():
        ax.scatter([], [], color=c, s=10, label=s)
    ax.legend(fontsize=5, frameon=False, ncol=2)
    fig.suptitle(f"ITI signal screen, {stage}: top {n_top} by combined rank (exploratory, uncorrected)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"screen_{stage}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(glob.glob(str(BASE / "features" / "*.parquet")))
    with ProcessPoolExecutor(a.jobs) as ex:
        S = pd.concat(list(ex.map(session_scores, paths)), ignore_index=True)
    S.to_parquet(OUT / "per_session_scores.parquet")
    for stage, sub in (("day0", S[S.day == 0]), ("expert", S[S.day > 0])):
        sub = sub[sub.cohort.isin(["R+", "R-"])]
        R = rank_stage(sub, stage)
        R.to_csv(OUT / f"ranking_{stage}.csv", index=False)
        figure(R, stage)
        print(f"\n=== {stage}: {sub.session_id.nunique()} sessions, {R.shape[0]} features")
        cols = ["feature", "n_sessions", "median_r_pdelta", "median_z_pdelta", "p_wilcoxon_pdelta",
                "median_r_lick_partial", "median_z_lick_partial", "p_wilcoxon_lick_partial", "E3", "E3_which",
                "E3_rho_Rplus", "E3_rho_Rminus", "combined_rank"]
        print(R[cols].head(25).round(3).to_string(index=False))
        for e, lab in (("E1", "learning curve (delta P)"), ("E2", "upcoming choice"), ("E3", "performance")):
            print(f"-- top 8 by {lab}:", R.sort_values(e, ascending=False).head(8)[["feature", e]].round(2).values.tolist())
    print("ALL DONE")


if __name__ == "__main__":
    main()
