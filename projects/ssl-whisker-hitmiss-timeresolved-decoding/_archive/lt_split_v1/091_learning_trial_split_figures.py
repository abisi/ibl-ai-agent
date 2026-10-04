"""Figures for `090_learning_trial_split_sensory_decode.py` (user request
2026-09-24: "produce similar figures as the session split halves figures").
Pre vs post `learning_trial` replaces half:first vs half:second; sensory
window (5-50ms) single-window decode, learning stage.

Three metrics, all per (session, area):
  - matched, above-null: acc_<ep>_matched - nullmean_<ep>_matched  (PRIMARY:
    both epochs subsampled to the same n_hit/n_miss, own label-shuffle null)
  - matched, raw: acc_<ep>_matched (chance 0.5)
  - unmatched, above-null: acc_<ep>_full - nullmean_<ep>_full (all trials of
    each epoch; post has far more training data -- biased toward post)

Figures (per population scope: `all` = entire dataset, `learners` =
learning_category in {good, moderate}; each R+/R- split):
  whole_brain
    091_..._paired_grid        027-style: rows=metric, cols=R+/R-/aggregated,
                               per-session lines, mean+-SEM, Wilcoxon + paired t
    091_..._overlaid           029-style: cohorts overlaid mean+-SEM per metric,
                               within-cohort W/t, R+ vs R- MWU/Welch at pre,
                               post; bottom row: per-session delta strip, R+ vs
                               R- MWU + Welch (mandatory test pair)
    091_..._delta_vs_ntrials   diagnostic: delta vs matched trial count
  area_group / area_acronym_custom
    091_..._condition_cohorts  034-style: per cohort side by side, x=area,
                               pre (desaturated) vs post (full) mean+-SEM;
                               title = epoch x area two-way ANOVA (034's
                               `factor_anova_and_posthoc`, same unpaired
                               simplification as the halves figure);
                               asterisk = PAIRED Wilcoxon per area, BH-FDR
                               q<0.05 (pre/post are within-session)

`lt_source` (how the stored learning_trial was reached -- see
`ssl_timeresolved_decoding.reconstruct_learning_trial`) is marked in every
per-session plot by marker shape (legend in figure), and counted in every
area-figure title (user request 2026-09-24: keep all sessions, but
identify the hardcoded-10 ones everywhere).

Usage (local, after pulling 090 parquets): python 091_learning_trial_split_figures.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import linregress, mannwhitneyu, ttest_ind, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

OUT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("q034", OUT_DIR / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q034)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
AGGREGATE_COLOR = "#2c5f5b"
LT_MARKER = {"learner": "o", "expert": "s", "learner_floor10": "^", "expert_floor10": "^", "learner_fallback10": "X"}
LT_LABEL = {"o": "criterion met (learner)", "s": "criterion met (expert rule)", "^": "clamped to 10 (floor10)",
            "X": "R- fallback: no criterion, hardcoded 10"}
FLAGGED = {"learner_floor10", "expert_floor10", "learner_fallback10"}
SCOPES = {"all": None, "learners": {"good", "moderate"}}
METRICS = [
    ("matched_abovenull", "matched: acc - own null", lambda d, ep: d[f"acc_{ep}_matched"] - d[f"nullmean_{ep}_matched"], 0.0),
    ("matched_raw", "matched: balanced accuracy", lambda d, ep: d[f"acc_{ep}_matched"], 0.5),
    ("full_abovenull", "unmatched: acc - own null", lambda d, ep: d[f"acc_{ep}_full"] - d[f"nullmean_{ep}_full"], 0.0),
]
EPOCHS = ("pre", "post")
# Overridable by callers (096 reuses this module for 095's shift-null /
# multi-window results): output-name prefix, window tag/label, input path.
FILE_PREFIX = "091_hitmiss_ltsplit"
WIN_TAG = "sensory"
WIN_LABEL = "sensory 5-50ms"
NULL_LABEL = "own null"
SRC_TEMPLATE = "090_lt_split_results_{scheme}.parquet"
STATS_CSV = "091_learning_trial_split_stats.csv"
WINDOW_FILTER = None  # 095 results carry a `window` column (one row per window)


def fig_dir(scheme: str) -> Path:
    d = OUT_DIR / "figures" / scheme / "learning"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(scheme: str) -> pd.DataFrame | None:
    path = OUT_DIR / SRC_TEMPLATE.format(scheme=scheme)
    if not path.exists():
        print(f"missing {path.name}")
        return None
    df = pd.read_parquet(path)
    if WINDOW_FILTER is not None:
        df = df[(df.window == WINDOW_FILTER) | df.skipped_reason.notna()]
    print(f"{scheme}: {df.session_id.nunique()} sessions, {df.skipped_reason.isna().sum()} decoded rows; skips:\n"
          f"{df[df.skipped_reason.notna()].drop_duplicates('session_id').skipped_reason.str.slice(0, 40).value_counts().to_string()}")
    df = df[df.skipped_reason.isna()].copy()
    for key, _, fn, _ in METRICS:
        for ep in EPOCHS:
            df[f"{key}_{ep}"] = fn(df, ep)
        df[f"{key}_delta"] = df[f"{key}_post"] - df[f"{key}_pre"]
    df["lt_marker"] = df.lt_source.map(LT_MARKER).fillna("o")
    df["matched_n"] = df.matched_n_hit + df.matched_n_miss
    return df


def scope_filter(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    cats = SCOPES[scope]
    return df if cats is None else df[df.learning_category.isin(cats)]


def paired_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.allclose(a, b):
        return np.nan, np.nan
    return float(wilcoxon(a, b).pvalue), float(ttest_rel(a, b).pvalue)


def unpaired_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return float(mannwhitneyu(a, b).pvalue), float(ttest_ind(a, b, equal_var=False).pvalue)


def flag_note(sub: pd.DataFrame) -> str:
    s = sub.drop_duplicates("session_id")
    n_floor = s.lt_source.isin({"learner_floor10", "expert_floor10"}).sum()
    n_fb = (s.lt_source == "learner_fallback10").sum()
    return f"incl. {n_floor} floor10, {n_fb} fallback10"


def lt_legend(fig):
    handles = [Line2D([], [], marker=m, ls="", color="#444444", markersize=6, label=l) for m, l in LT_LABEL.items()]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0))


def savefig(fig, path: Path):
    q034.savefig_retry(fig, path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(OUT_DIR)}")


# ---------------------------------------------------------------------------
# whole-brain figures
# ---------------------------------------------------------------------------

def paired_grid(df: pd.DataFrame, scope: str):
    groups = [("R+", df[df.reward_group == "R+"]), ("R-", df[df.reward_group == "R-"]), ("aggregated", df)]
    fig, axes = plt.subplots(len(METRICS), 3, figsize=(10, 3.9 * len(METRICS)), constrained_layout=True)
    for r, (key, label, _, ref) in enumerate(METRICS):
        vals = np.concatenate([df[f"{key}_pre"], df[f"{key}_post"]])
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        pad = 0.08 * (hi - lo)
        for c, (gname, sub) in enumerate(groups):
            ax = axes[r, c]
            for row in sub.itertuples():
                col = COHORT_COLOR[row.reward_group]
                a, b = getattr(row, f"{key}_pre"), getattr(row, f"{key}_post")
                flagged = row.lt_source in FLAGGED
                ax.plot([0, 1], [a, b], color=col, alpha=0.45 if flagged else 0.3, lw=0.9, ls="--" if flagged else "-")
                ax.scatter([0, 1], [a, b], marker=row.lt_marker, s=14 if flagged else 8, color=col,
                           alpha=0.8 if flagged else 0.4, lw=0, zorder=3)
            pre, post = sub[f"{key}_pre"].to_numpy(), sub[f"{key}_post"].to_numpy()
            gcol = COHORT_COLOR.get(gname, AGGREGATE_COLOR)
            m = [np.nanmean(pre), np.nanmean(post)]
            se = [np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x))) for x in (pre, post)]
            ax.errorbar([0, 1], m, yerr=se, color=gcol, lw=3, marker="o", markersize=7, capsize=3, zorder=5)
            pw, pt = paired_p(pre, post)
            ax.axhline(ref, color="#888888", lw=1, ls=":", zorder=0)
            ax.text(0.5, 0.02, f"n={len(sub)} sessions ({flag_note(sub)})\nWilcoxon p={pw:.3g}\npaired-t p={pt:.3g}",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5)
            ax.set_xticks([0, 1], ["pre", "post"])
            ax.set_xlim(-0.3, 1.3)
            ax.set_ylim(lo - 3 * pad, hi + pad)
            ax.set_box_aspect(1)
            ax.set_title(f"{label}\n{gname}", fontsize=9.5)
            ax.spines[["top", "right"]].set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle(f"Hit vs miss, {WIN_LABEL}, pre vs post learning_trial, null = {NULL_LABEL} (whole brain) -- scope: {scope}", fontsize=11)
    lt_legend(fig)
    savefig(fig, fig_dir("whole_brain") / f"{FILE_PREFIX}_whole_brain_{WIN_TAG}_paired_grid_{scope}.png")


def overlaid(df: pd.DataFrame, scope: str):
    fig, axes = plt.subplots(2, len(METRICS), figsize=(4.4 * len(METRICS), 9), constrained_layout=True)
    for c, (key, label, _, ref) in enumerate(METRICS):
        ax = axes[0, c]
        lines = []
        for k, cohort in enumerate(("R+", "R-")):
            sub = df[df.reward_group == cohort]
            pre, post = sub[f"{key}_pre"].to_numpy(), sub[f"{key}_post"].to_numpy()
            m = [np.nanmean(pre), np.nanmean(post)]
            se = [np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x))) for x in (pre, post)]
            ax.errorbar(np.array([0, 1]) + (k - 0.5) * 0.04, m, yerr=se, color=COHORT_COLOR[cohort], lw=2.5, marker="o",
                        capsize=3, label=cohort)
            pw, pt = paired_p(pre, post)
            lines.append(f"{cohort}: n={len(sub)}, W p={pw:.2g}, t p={pt:.2g}")
        for ep in EPOCHS:
            pm, pw_ = unpaired_p(df[df.reward_group == "R+"][f"{key}_{ep}"].to_numpy(),
                                 df[df.reward_group == "R-"][f"{key}_{ep}"].to_numpy())
            lines.append(f"R+ vs R- @{ep}: MW p={pm:.2g}, Welch p={pw_:.2g}")
        ax.axhline(ref, color="#888888", lw=1, ls=":", zorder=0)
        ax.text(0.5, 0.02, "\n".join(lines), transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo - 0.45 * (yhi - ylo), yhi)
        ax.set_xticks([0, 1], ["pre", "post"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_box_aspect(1)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1, c]
        rng = np.random.default_rng(0)
        for k, cohort in enumerate(("R+", "R-")):
            sub = df[df.reward_group == cohort]
            x = k + rng.uniform(-0.12, 0.12, len(sub))
            for marker in sub.lt_marker.unique():
                mm = (sub.lt_marker == marker).to_numpy()
                ax.scatter(x[mm], sub[f"{key}_delta"].to_numpy()[mm], marker=marker, s=18, color=COHORT_COLOR[cohort],
                           alpha=0.6, lw=0)
            d = sub[f"{key}_delta"].to_numpy()
            ax.errorbar(k + 0.28, np.nanmean(d), yerr=np.nanstd(d) / np.sqrt(np.sum(~np.isnan(d))), color="k", marker="o",
                        capsize=3)
        pm, pw_ = unpaired_p(df[df.reward_group == "R+"][f"{key}_delta"].to_numpy(),
                             df[df.reward_group == "R-"][f"{key}_delta"].to_numpy())
        ax.axhline(0, color="#888888", lw=1, ls=":", zorder=0)
        ax.set_xticks([0, 1], ["R+", "R-"])
        ax.set_xlim(-0.5, 1.6)
        ax.set_box_aspect(1)
        ax.set_title(f"delta (post - pre), {label}\nR+ vs R-: MW p={pm:.3g}, Welch p={pw_:.3g}", fontsize=9)
        ax.set_ylabel("post - pre", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Hit vs miss, {WIN_LABEL}, pre vs post learning_trial, null = {NULL_LABEL}, cohorts overlaid (whole brain) -- scope: {scope}",
                 fontsize=11)
    lt_legend(fig)
    savefig(fig, fig_dir("whole_brain") / f"{FILE_PREFIX}_whole_brain_{WIN_TAG}_overlaid_{scope}.png")


def delta_vs_ntrials(df: pd.DataFrame, scope: str):
    """Diagnostic: is the pre->post change driven by how small the matched
    set is (floor10 sessions have ~9 pre trials)? Correlation-figure
    convention: scatter + OLS + 95% CI band, solid line only if p<0.05."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), constrained_layout=True)
    for ax, (xcol, xlabel) in zip(axes, [("matched_n", "matched trials per epoch (n_hit* + n_miss*)"),
                                         ("learning_trial", "learning_trial (whisker-trial index)")]):
        for cohort in ("R+", "R-"):
            sub = df[df.reward_group == cohort].dropna(subset=[xcol, "matched_abovenull_delta"])
            for marker in sub.lt_marker.unique():
                mm = sub.lt_marker == marker
                ax.scatter(sub[mm][xcol], sub[mm]["matched_abovenull_delta"], marker=marker, s=22,
                           color=COHORT_COLOR[cohort], alpha=0.65, lw=0)
            if len(sub) >= 4:
                x, y = sub[xcol].to_numpy(float), sub["matched_abovenull_delta"].to_numpy()
                res = linregress(x, y)
                xs = np.linspace(x.min(), x.max(), 100)
                n = len(x)
                se_fit = np.sqrt(np.sum((y - (res.intercept + res.slope * x)) ** 2) / (n - 2)) * np.sqrt(
                    1 / n + (xs - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
                from scipy.stats import t as tdist
                yfit = res.intercept + res.slope * xs
                ax.fill_between(xs, yfit - tdist.ppf(0.975, n - 2) * se_fit, yfit + tdist.ppf(0.975, n - 2) * se_fit,
                                color=COHORT_COLOR[cohort], alpha=0.12, lw=0)
                ax.plot(xs, yfit, color=COHORT_COLOR[cohort], ls="-" if res.pvalue < 0.05 else "--", lw=1.5,
                        label=f"{cohort}: r={res.rvalue:.2f}, p={res.pvalue:.2g}, n={n}")
        ax.axhline(0, color="#888888", lw=1, ls=":")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("delta matched acc - null (post - pre)", fontsize=9)
        ax.legend(frameon=False, fontsize=7.5)
        ax.set_box_aspect(1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Diagnostic ({WIN_LABEL}, null = {NULL_LABEL}): pre->post change vs epoch size / learning_trial (whole brain) -- scope: {scope}", fontsize=10)
    lt_legend(fig)
    savefig(fig, fig_dir("whole_brain") / f"{FILE_PREFIX}_whole_brain_{WIN_TAG}_delta_vs_ntrials_{scope}.png")


# ---------------------------------------------------------------------------
# area-level figure (034 condition_cohorts style)
# ---------------------------------------------------------------------------

def area_condition_cohorts(df: pd.DataFrame, scheme: str, scope: str):
    areas = q034.common_areas(df, scheme)
    colors = q034.get_area_color_map(areas)
    metrics = [m for m in METRICS if m[0] in ("matched_abovenull", "full_abovenull")]
    stats_rows = []
    ylo, yhi = [], []
    for key, _, _, _ in metrics:
        for cohort in ("R+", "R-"):
            sub = df[(df.reward_group == cohort) & df.area_value.isin(areas)]
            g = sub.groupby("area_value")
            for ep in EPOCHS:
                m, s = g[f"{key}_{ep}"].mean(), g[f"{key}_{ep}"].sem()
                ylo.append((m - s).min()); yhi.append((m + s).max())
    pad = 0.12 * (max(yhi) - min(ylo))
    ylim = (min(min(ylo) - pad, -0.01), max(yhi) + 2 * pad)

    fig, axes = plt.subplots(len(metrics), 2, figsize=(q034.area_fig_width_mult(areas) * 2, 4.4 * len(metrics)),
                             squeeze=False, constrained_layout=True)
    for r, (key, label, _, _) in enumerate(metrics):
        for c, cohort in enumerate(("R+", "R-")):
            ax = axes[r][c]
            sub = df[(df.reward_group == cohort) & df.area_value.isin(areas)]
            top = {}
            for ep, off in (("pre", -0.12), ("post", 0.12)):
                g = sub.groupby("area_value")[f"{key}_{ep}"]
                mean, sem = g.mean().reindex(areas), g.sem().reindex(areas)
                for i, area in enumerate(areas):
                    col = colors[area] if ep == "post" else q034.lighten(colors[area])
                    ax.errorbar(i + off, mean[area], yerr=sem[area], fmt="o", color=col, markersize=4, capsize=2,
                                markeredgecolor="black", markeredgewidth=0.5)
                    if not np.isnan(mean[area]):
                        top[area] = max(top.get(area, -np.inf), mean[area] + (0 if np.isnan(sem[area]) else sem[area]))
            long = pd.concat([sub[["session_id", "area_value"]].assign(lt_epoch=ep, v=sub[f"{key}_{ep}"]) for ep in EPOCHS])
            long["m"] = long.v + 0.5  # factor_anova_and_posthoc subtracts 0.5
            f_val, p_val, _ = q034.factor_anova_and_posthoc(long, areas, "m", "lt_epoch", "pre", "post")
            raw = {}
            for area in areas:
                a = sub[sub.area_value == area]
                raw[area] = paired_p(a[f"{key}_pre"].to_numpy(), a[f"{key}_post"].to_numpy())
            testable = [a for a in areas if not np.isnan(raw[a][0])]
            q = dict(zip(testable, multipletests([raw[a][0] for a in testable], method="fdr_bh")[1])) if testable else {}
            span = ylim[1] - ylim[0]
            for i, area in enumerate(areas):
                if q.get(area, 1) < 0.05 and area in top:
                    ax.text(i, min(top[area] + 0.03 * span, ylim[1] - 0.03 * span), "*", ha="center", fontsize=11)
                a = sub[sub.area_value == area]
                stats_rows.append(dict(scheme=scheme, scope=scope, metric=key, reward_group=cohort, area=area,
                                       n_sessions=a.session_id.nunique(), n_flagged=int(a.lt_source.isin(FLAGGED).sum()),
                                       mean_pre=a[f"{key}_pre"].mean(), mean_post=a[f"{key}_post"].mean(),
                                       p_wilcoxon=raw[area][0], p_paired_t=raw[area][1], q_wilcoxon_fdr=q.get(area, np.nan),
                                       anova_epoch_F=f_val, anova_epoch_p=p_val))
            n_s = sub.session_id.nunique()
            anova = f"ANOVA epoch: F={f_val:.2f}, p={p_val:.3g}" if not np.isnan(p_val) else "ANOVA epoch: n/a"
            q034.style_ax(ax, areas, colors, f"{cohort} | {label}\n"
                                             f"(n={n_s} sessions, {flag_note(sub)})\n{anova}\n* = paired Wilcoxon, FDR q<0.05",
                          ylim, ylabel=f"balanced accuracy - {NULL_LABEL}")
    fig.suptitle(f"hitmiss learning_trial split, {WIN_LABEL}, null = {NULL_LABEL} -- {scheme} -- pre (desaturated) vs post (full color) "
                 f"-- scope: {scope}", fontsize=11)
    savefig(fig, fig_dir(scheme) / f"{FILE_PREFIX}_{scheme}_{WIN_TAG}_grid_condition_cohorts_{scope}.png")
    return stats_rows


def main():
    all_stats = []
    df = load("whole_brain")
    if df is not None:
        for scope in SCOPES:
            d = scope_filter(df, scope)
            paired_grid(d, scope)
            overlaid(d, scope)
            delta_vs_ntrials(d, scope)
            for key, _, _, _ in METRICS:
                for gname, sub in (("R+", d[d.reward_group == "R+"]), ("R-", d[d.reward_group == "R-"]), ("aggregated", d)):
                    pw, pt = paired_p(sub[f"{key}_pre"].to_numpy(), sub[f"{key}_post"].to_numpy())
                    all_stats.append(dict(scheme="whole_brain", scope=scope, metric=key, reward_group=gname, area="All units",
                                          n_sessions=len(sub), n_flagged=int(sub.lt_source.isin(FLAGGED).sum()),
                                          mean_pre=sub[f"{key}_pre"].mean(), mean_post=sub[f"{key}_post"].mean(),
                                          p_wilcoxon=pw, p_paired_t=pt))
                pm, pwe = unpaired_p(d[d.reward_group == "R+"][f"{key}_delta"].to_numpy(),
                                     d[d.reward_group == "R-"][f"{key}_delta"].to_numpy())
                all_stats.append(dict(scheme="whole_brain", scope=scope, metric=f"{key}_delta", reward_group="R+ vs R-",
                                      area="All units", p_mannwhitney=pm, p_welch=pwe))
    for scheme in ("area_group", "area_acronym_custom"):
        df = load(scheme)
        if df is not None:
            for scope in SCOPES:
                all_stats += area_condition_cohorts(scope_filter(df, scope), scheme, scope)
    pd.DataFrame(all_stats).to_csv(OUT_DIR / STATS_CSV, index=False)


if __name__ == "__main__":
    main()
