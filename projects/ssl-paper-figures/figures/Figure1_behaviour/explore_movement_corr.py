"""Exploration (user 2026-11-09): pre-stimulus movement vs learning curves, trial by trial within each mouse, as in
behaviour_analysis/beh_correlates/beh_correlates_lick_decoding.py (Pearson, lag 0, linear-shift null), here with
Spearman next to Pearson.

Per whisker trial of whisker day 0 (all whisker trials, lick or not; video sessions as in shared_movements), the
beh_correlates measures through dlc_helpers (explore_movement_session.one_session): posture (window mean) and movement
(mean |frame-to-frame difference|) of jaw, nose, whisker angle and pupil area, pre-stimulus windows [-0.2, 0) s (pupil
[-2, 0) s); post-stimulus windows also computed and saved. Curves at each whisker trial: whisker (p_mean), false alarm
(p_chance) and their difference (exact curves of Figure 1).
Per mouse and pair: Pearson r and Spearman rho (= Pearson on ranks); linear-shift null as beh_correlates: the measure is
circularly shifted against the curve by every shift with |shift| > 10 trials (up to 500 shifts, evenly spaced); value
reported = observed minus the null median (user 2026-11-09). Mean +- s.e.m. over mice per cohort; Wilcoxon against 0.
Learners and all mice (figures for both). Output: Figures/_explore/movement_corr_<date>_{learners,all}.{pdf,png} + CSV.
Run on haas (repo root): .venv/bin/python projects/ssl-paper-figures/figures/Figure1_behaviour/explore_movement_corr.py
"""

import datetime
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import shared_behaviour  # noqa: E402
import shared_movements as sm  # noqa: E402
from common import FIGURES_ROOT, REFERENCE_SHEET  # noqa: E402
from explore_movement_session import BODYPARTS, one_session  # noqa: E402
from paperstyle import COLORS, MM, set_nature_rc  # noqa: E402

OUT = FIGURES_ROOT / "_explore"
EXCLUDE, N_SHIFT, MIN_TRIALS = 10, 500, 30
CURVES = ("whisker", "false_alarm", "difference")


def corr_with_null(x, y):
    """Pearson r of x, y and the median r over circular shifts of x with |shift| > EXCLUDE (user 2026-11-09: median)."""
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    n = len(x)
    r = float(np.mean(x * y))
    shifts = np.array([s for s in range(-n + 1, n) if abs(s) > EXCLUDE and abs(s) < n - EXCLUDE])
    if len(shifts) > N_SHIFT:
        shifts = shifts[np.linspace(0, len(shifts) - 1, N_SHIFT).astype(int)]
    null = np.array([np.mean(np.roll(x, s) * y) for s in shifts]) if len(shifts) else np.array([np.nan])
    return r, float(np.nanmedian(null))


def main():
    from compute import session_curves
    table = shared_behaviour.load()
    ref = pd.read_excel(REFERENCE_SHEET)
    dual = set(ref[ref.with_dual_view == 1].mouse_id)
    d0 = table[(table.behavior == "whisker") & (table.day == 0) & (table.early_lick == 0)
               & table.mouse_id.isin(dual)].sort_values("start_time")
    w = sm.data_utils().keep_video_data(d0[d0.trial_type == "whisker_trial"], views=("side", "top"))
    with Pool(16) as pool:
        parts = [p for p in pool.map(one_session, list(w.groupby("session_id"))) if p is not None]
    mv = pd.concat(parts, ignore_index=True)
    metrics = [c for c in mv.columns if c.endswith(("posture", "movement"))]
    rows, ctx = [], {}
    for sid, m in mv.groupby("session_id"):
        curves, _, _ = session_curves(d0[d0.session_id == sid], ctx)
        if len(curves) != len(m):
            continue
        cv = {"whisker": curves.p_mean.to_numpy(), "false_alarm": curves.p_chance.to_numpy(),
              "difference": (curves.p_mean - curves.p_chance).to_numpy()}
        for met in metrics:
            x = m[met].to_numpy(float)
            for cu, y in cv.items():
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < MIN_TRIALS or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
                    continue
                pr, pn = corr_with_null(x[ok], y[ok])
                sr, sn = corr_with_null(stats.rankdata(x[ok]).astype(float), stats.rankdata(y[ok]).astype(float))
                rows.append({"mouse_id": m.mouse_id.iloc[0], "session_id": sid, "cohort": m.cohort.iloc[0],
                             "metric": met, "curve": cu, "n_trials": int(ok.sum()), "pearson": pr, "pearson_null": pn,
                             "spearman": sr, "spearman_null": sn})
    df = pd.DataFrame(rows)
    df["pearson_vs_null"] = df.pearson - df.pearson_null
    df["spearman_vs_null"] = df.spearman - df.spearman_null
    df["learning_category"] = df.mouse_id.map(ref.set_index("mouse_id").learning_category)
    stem = OUT / f"movement_corr_{datetime.date.today()}"
    df.to_csv(f"{stem}.csv", index=False)
    for name, sel in (("learners", df.learning_category.isin(["good", "moderate"])), ("all", df.index == df.index)):
        plot(df[sel], f"{stem}_{name}", name)
    pre = df[df.metric.str.contains("_pre_") & df.learning_category.isin(["good", "moderate"])]
    print(pre.groupby(["curve", "metric", "cohort"])[["pearson_vs_null", "spearman_vs_null"]].mean().round(3).unstack("cohort").to_string())
    print("Pearson vs Spearman agreement (per mouse and pair):",
          round(stats.spearmanr(df.pearson_vs_null, df.spearman_vs_null)[0], 3))


def plot(df, stem, label):
    set_nature_rc()
    pre = [f"{bp}_pre_{k}" for bp in BODYPARTS for k in ("posture", "movement")]
    labels = [f"{bp.capitalize()} {('pos.' if k == 'posture' else 'mov.')}" for bp in BODYPARTS for k in ("posture", "movement")]
    fig, axs = plt.subplots(3, 1, figsize=(183 * MM, 150 * MM), sharex=True)
    for a, cu in zip(axs, CURVES):
        for i, met in enumerate(pre):
            for k, co in enumerate(("R+", "R-")):
                for j, (col, mk) in enumerate((("pearson_vs_null", "o"), ("spearman_vs_null", "s"))):
                    v = df[(df.metric == met) & (df.curve == cu) & (df.cohort == co)][col].dropna()
                    if not len(v):
                        continue
                    x = i + (k - 0.5) * 0.4 + (j - 0.5) * 0.16
                    p = stats.wilcoxon(v).pvalue if len(v) > 5 else np.nan
                    a.errorbar(x, v.mean(), v.sem(), color=COLORS[co], marker=mk, ms=2.6, lw=0, elinewidth=0.7, capsize=0,
                               mfc=COLORS[co] if p < 0.05 else "white", mew=0.6)
        a.axhline(0, color="0.5", lw=0.4)
        a.set_ylabel(f"vs {cu.replace('_', '-')} curve\ncorr. − shift null", labelpad=1)
        for i in range(1, len(pre)):
            a.axvline(i - 0.5, color="0.92", lw=0.4, zorder=0)
    axs[-1].set_xticks(range(len(pre)), labels, fontsize=5)
    n = df.groupby("cohort").mouse_id.nunique().to_dict()
    axs[0].set_title(f"Pre-stimulus movement vs learning curves, whisker trials of day 0, {label} (R+ n = {n.get('R+', 0)}, "
                     f"R- n = {n.get('R-', 0)} mice). Circles, Pearson; squares, Spearman; filled, Wilcoxon vs 0 P < 0.05; "
                     "mean ± s.e.m. over mice", fontsize=5.5, pad=4)
    fig.tight_layout(h_pad=0.6)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=400)


if __name__ == "__main__":
    main()
