"""Example sessions: per-trial held-out hit/miss decoder predictions around
the learning_trial (user request 2026-09-24: "plot example sessions with
test trial predictions, highlighting the change before vs after learning
trial"). Whole brain, sensory window 5-50ms, learning stage.

Data: `092_lt_aligned_trials_whole_brain.parquet` (per-trial held-out
P(true class) from one class-balanced decoder per session fit on all
trials, + per-trial label-shuffle null) and the session's learning-curve H5
(behavior). Session-level pre/post numbers in each title come from 090's
size-matched, separately-fit decoders (`090_lt_split_results_whole_brain.parquet`).

Per session, two stacked panels sharing the x-axis (whisker trial index in
the curve-aligned trial set, the index learning_trial counts in):
  top    -- behavior: p_mean with 80% CI, p_chance (FA) dashed, lick
            outcomes as ticks, learning_trial red.
  bottom -- decoder P(hit) per held-out trial: filled = hit, open = miss
            (cohort color). Running separation = mean P(hit | hit) -
            mean P(hit | miss) in a centered RUN-trial window (both classes
            required), black line, right axis. Horizontal bars = pre and
            post class-balanced mean of (p_correct - null).

Selection rule (stated in the figure): per cohort, from sessions with
>=MIN_PRE_PER_CLASS pre trials of each class, ranked by 090's
`delta_matched_nullcorr` (post - pre, size-matched, above own null):
the 2 strongest in the cohort's group-mean direction (R+: increase, R-:
decrease), the median session, and the strongest counter-direction one.

Output: figures/whole_brain/learning/094_hitmiss_lt_example_sessions.png/.pdf
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ssl_timeresolved_decoding import load_whisker_curve_row  # noqa: E402

_spec = importlib.util.spec_from_file_location("q034", OUT_DIR / "034_area_window_quant_grid.py")
q034 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(q034)

COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}
MIN_PRE_PER_CLASS = 4
RUN = 15


def pick_examples(res: pd.DataFrame) -> list[tuple[str, str]]:
    picks = []
    for cohort, direction in (("R+", 1), ("R-", -1)):
        c = res[(res.reward_group == cohort) & (res.pre_hit >= MIN_PRE_PER_CLASS) & (res.pre_miss >= MIN_PRE_PER_CLASS)]
        c = c.assign(score=direction * c.delta_matched_nullcorr).sort_values("score", ascending=False)
        chosen = [(r.session_id, f"strong {'increase' if direction > 0 else 'decrease'} #{i + 1}")
                  for i, r in enumerate(c.head(2).itertuples())]
        med = c.iloc[(c.score - c.score.median()).abs().argsort().iloc[0]]
        chosen.append((med.session_id, "median session"))
        chosen.append((c.iloc[-1].session_id, "counter-direction"))
        seen = set()
        picks += [(s, lab) for s, lab in chosen if not (s in seen or seen.add(s))]
    return picks


def running_separation(ph: np.ndarray, y: np.ndarray, run: int) -> np.ndarray:
    out = np.full(len(ph), np.nan)
    h = run // 2
    for i in range(len(ph)):
        sl = slice(max(0, i - h), i + h + 1)
        yy, pp = y[sl], ph[sl]
        if yy.any() and (~yy).any():
            out[i] = pp[yy].mean() - pp[~yy].mean()
    return out


def draw_session(fig_sub, sid: str, label: str, trials: pd.DataFrame, res_row: pd.Series, curve_root: Path):
    t = trials[trials.session_id == sid].sort_values("curve_idx")
    cohort = t.reward_group.iloc[0]
    col = COHORT_COLOR[cohort]
    lt = int(t.learning_trial.iloc[0])
    curve = load_whisker_curve_row(curve_root, t.mouse_id.iloc[0], 0)
    ax_b, ax_d = fig_sub.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[1, 1.5]))

    x = np.arange(len(curve["p_mean"]))
    ax_b.fill_between(x, curve["p_low"], curve["p_high"], color=col, alpha=0.2, lw=0)
    ax_b.plot(x, curve["p_mean"], color=col, lw=1.5)
    ax_b.plot(x, curve["p_chance"], color="#555555", lw=1, ls="--")
    o = np.asarray(curve["outcomes"])
    ax_b.scatter(x, np.where(o == 1, 1.06, -0.06), s=4, color="k", marker="|", clip_on=False)
    ax_b.set_ylim(-0.1, 1.1)
    ax_b.set_ylabel("P(lick)", fontsize=8)

    ci = t.curve_idx.to_numpy()
    y = t.lick_flag.to_numpy().astype(bool)
    ph = np.where(y, t.p_correct, 1 - t.p_correct)
    ax_d.scatter(ci[y], ph[y], s=16, color=col, edgecolor=col, lw=0.6, label="hit trial", zorder=3)
    ax_d.scatter(ci[~y], ph[~y], s=16, facecolor="white", edgecolor=col, lw=0.8, label="miss trial", zorder=3)
    ax_d.axhline(0.5, color="#888888", lw=0.8, ls=":")
    ax_d.set_ylim(-0.03, 1.03)
    ax_d.set_ylabel("decoder P(hit)\n(held-out)", fontsize=8)
    ax_d.set_xlabel("whisker trial (learning_trial = red)", fontsize=8)

    ax_s = ax_d.twinx()
    ax_s.plot(ci, running_separation(ph, y, RUN), color="k", lw=1.4, label=f"separation ({RUN}-trial window)")
    ax_s.axhline(0, color="k", lw=0.6, ls=":")
    ax_s.set_ylim(-0.6, 1.0)
    ax_s.set_ylabel("P(hit|hit) - P(hit|miss)", fontsize=7)
    ax_s.tick_params(labelsize=6.5)

    dec = (t.p_correct - t.p_correct_null).to_numpy()
    xmax = len(x)
    for ep, lo, hi in (("pre", 0, lt), ("post", lt, xmax)):
        m = (ci >= lo) & (ci < hi)
        cm = [dec[m & y].mean() if (m & y).any() else np.nan, dec[m & ~y].mean() if (m & ~y).any() else np.nan]
        val = np.nanmean(cm)
        ax_s.hlines(val, lo, hi, color="#d62728" if ep == "pre" else "#1f77b4", lw=3, alpha=0.7)
        ax_s.text((lo + hi) / 2, val + 0.05, f"{ep}: {val:+.2f}", ha="center", fontsize=7,
                  color="#d62728" if ep == "pre" else "#1f77b4")
    for ax in (ax_b, ax_d):
        ax.axvline(lt, color="#d62728", lw=1.4)
        ax.axvspan(0, lt, color="#f2f2f2", zorder=0)
        ax.tick_params(labelsize=7)
        ax.spines[["top"]].set_visible(False)
    ax_d.legend(frameon=False, fontsize=6.5, loc="lower right", ncol=2)

    lt_src = t.lt_source.iloc[0]
    ttl = (f"{sid} ({cohort}, {t.learning_category.iloc[0]}) -- {label}\n"
           f"learning_trial={lt} [{lt_src}]; pre {res_row.pre_hit}h/{res_row.pre_miss}m, post {res_row.post_hit}h/{res_row.post_miss}m\n"
           f"090 size-matched acc-null: pre {res_row.acc_pre_matched - res_row.nullmean_pre_matched:+.2f} -> "
           f"post {res_row.acc_post_matched - res_row.nullmean_post_matched:+.2f}")
    ax_b.set_title(ttl, fontsize=8)


def main():
    trials = pd.read_parquet(OUT_DIR / "092_lt_aligned_trials_whole_brain.parquet")
    trials = trials[trials.skipped_reason.isna() & trials.curve_idx.notna()]
    res = pd.read_parquet(OUT_DIR / "090_lt_split_results_whole_brain.parquet")
    res = res[res.skipped_reason.isna() & res.session_id.isin(trials.session_id.unique())]
    picks = pick_examples(res)
    print("examples:", picks)
    curve_root = axel_bisi_path("combined_results_ks4")

    ncol = 2
    nrow = int(np.ceil(len(picks) / ncol))
    fig = plt.figure(figsize=(8.5 * ncol, 5.2 * nrow), constrained_layout=True)
    subs = fig.subfigures(nrow, ncol).flat
    by_cohort = {"R+": [p for p in picks if res.set_index("session_id").reward_group[p[0]] == "R+"],
                 "R-": [p for p in picks if res.set_index("session_id").reward_group[p[0]] == "R-"]}
    order = [p for pair in zip(by_cohort["R+"], by_cohort["R-"]) for p in pair]
    for sub, (sid, label) in zip(subs, order):
        draw_session(sub, sid, label, trials, res.set_index("session_id").loc[sid], curve_root)
    fig.suptitle("Example sessions: held-out hit/miss decoder predictions (whole brain, sensory 5-50ms) around the learning_trial "
                 "(left R+, right R-)\nshaded = pre; bars = class-balanced mean P(true class) - own shuffle null, pre (red) vs post (blue); "
                 f"selection: per cohort top-2 in group direction, median, and strongest counter-direction by 090 size-matched delta "
                 f"(>= {MIN_PRE_PER_CLASS} pre trials per class)", fontsize=10)
    out = q034.fig_dir("whole_brain") / "094_hitmiss_lt_example_sessions.png"
    q034.savefig_retry(fig, out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
