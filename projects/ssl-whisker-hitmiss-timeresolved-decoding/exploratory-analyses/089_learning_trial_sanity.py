"""Sanity check for the new pre/post-learning-trial split (user request
2026-09-24: replace the session-half split of hit/miss decoding with a
split at each session's `learning_trial`, read from Axel Bisi's per-session
learning-curve H5, `<mouse>/whisker_<day>/learning_curve/<mouse>_whisker_
<day>_whisker_trial_learning_curve_interp.h5`).

Checks, per learning-stage (day 0) hit/miss session:
  1. curve file exists; `learning_trial` value (and `mouse_cat`).
  2. alignment: len(outcomes) == n whisker trials in the curve-aligned
     trial set (`_active_trials_from_whisker_onset_for_curve`, the same
     alignment `prep_perfstate_trials_curve` validated 2026-09-15/16), and
     outcomes == lick_flag trial-by-trial.
  3. rule reconstruction: `ssl_timeresolved_decoding.reconstruct_learning_trial` ports
     `behaviour_analysis/learning_utils.py`'s `identify_learning_trial_
     rewarded` (R+) / `identify_learning_trial_nonrewarded(...,
     interp_flag=True)` (R-), params from `beh_plotting_functions.py`
     (n_trials_for_expert=20, min/max_perf_for_expert=0.8/0.2,
     n_consec=5), only counting trials >= first whisker hit. First tried
     "first run of >=K above-chance trials" for K=1..10 -- max 40% match,
     rejected. The port matches the stored value in 89/89 sessions, and
     additionally labels HOW each value was reached (`lt_source`):
       - R+ 'expert': first 20-consecutive above-chance window within
         trials <=20 with mean p_mean>=0.8 -> window start
       - R+ 'learner': first run of 5 consecutive above-chance -> run start
       - R- 'expert': first 20-consecutive below-chance window with mean
         p_mean<=0.2 -> window start
       - R- 'learner': LAST trial of the first 5-run of above-chance trials
         followed by 5 not-above-chance trials (end of the "licking to
         whisker like to the rewarded auditory" phase)
       - R- 'fallback10': no criterion met -> hardcoded ('learner', 10)
       - any 'floor10': criterion met at trial <=10 -> clamped to 10
       - 'non-learner': NaN
  4. mapping to the DECODED trial set (`prep_hitmiss_trials`, which is
     built by `prep_session` and so differs by one dropped trial from the
     curve-aligned set): split by `start_time` of the learning trial, not by
     positional index. The learning trial itself goes to 'post'.
  5. hit/miss counts per epoch.

Outputs: `089_learning_trial_sanity.csv`, `089_learning_trial_curves_grid.png`,
`089_learning_trial_summary.png`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from axel_bisi_paths import axel_bisi_path  # noqa: E402
from ibl_ai_agent.data_locations import resolve_dataset_dir  # noqa: E402
from ssl_bwm_trial_prep import session_training_day  # noqa: E402
from ssl_timeresolved_decoding import (  # noqa: E402
    _active_trials_from_whisker_onset_for_curve,
    hitmiss_session_list,
    load_whisker_curve_row,
    prep_hitmiss_trials,
    reconstruct_learning_trial,
)

OUT_DIR = Path(__file__).resolve().parent
COHORT_COLOR = {"R+": "#00B400", "R-": "#C800C8"}


def first_run_start(idx: np.ndarray, k: int) -> int | None:
    idx = np.sort(np.asarray(idx, dtype=int))
    if len(idx) == 0:
        return None
    run_start, run_len = idx[0], 1
    if run_len >= k:
        return int(run_start)
    for prev, cur in zip(idx[:-1], idx[1:]):
        if cur == prev + 1:
            run_len += 1
        else:
            run_start, run_len = cur, 1
        if run_len >= k:
            return int(run_start)
    return None


def main():
    dataset_root = resolve_dataset_dir("ssl_ephys")
    sessions_tbl = pd.read_parquet(dataset_root / "metadata" / "sessions.parquet")
    trials_tbl = pd.read_parquet(dataset_root / "metadata" / "trials.parquet")
    curve_root = axel_bisi_path("combined_results_ks4")
    sess = hitmiss_session_list(sessions_tbl)
    sess = sess[sess.day_stage == "learning"].reset_index(drop=True)
    print(f"learning-stage hit/miss sessions: {len(sess)}", flush=True)

    rows, curves = [], {}
    for s in sess.itertuples():
        desc = sessions_tbl.loc[sessions_tbl.session_id == s.session_id, "session_description"].iloc[0]
        day = session_training_day(desc)
        row = dict(session_id=s.session_id, subject_id=s.subject_id, reward_group=s.reward_group,
                   learning_category=s.learning_category, day=day)
        curve = load_whisker_curve_row(curve_root, s.subject_id, day) if day is not None else None
        if curve is None:
            rows.append(dict(row, status="no_curve_file"))
            continue
        lt = curve["learning_trial"]
        lt = None if pd.isna(lt) or lt < 0 else int(lt)
        outcomes = np.asarray(curve["outcomes"]).astype(int)
        above = np.asarray(curve["trials_above_chance"], dtype=int)
        row.update(learning_trial=lt, mouse_cat=curve["mouse_cat"], curve_reward_group=curve["reward_group"],
                   n_curve_trials=len(outcomes), n_above=len(above))
        for k in range(1, 11):
            row[f"rule_k{k}"] = first_run_start(above, k)
        cat_r, lt_r, src = reconstruct_learning_trial(curve)
        row.update(lt_repro=lt_r, mouse_cat_repro=cat_r, lt_source=src,
                   lt_repro_match=(lt is None and pd.isna(lt_r)) or (lt == lt_r))

        cw = _active_trials_from_whisker_onset_for_curve(s.session_id, trials_tbl)
        cw = cw[cw.trial_type == "whisker_trial"].reset_index(drop=True)
        row["n_curve_aligned_whisker"] = len(cw)
        aligned = len(cw) == len(outcomes)
        row["len_match"] = aligned
        row["outcome_match_frac"] = float(np.mean(cw.lick_flag.to_numpy().astype(int) == outcomes)) if aligned else np.nan

        dec = prep_hitmiss_trials(dataset_root, s.session_id, sessions_tbl, trials_tbl)
        row["n_decoded_whisker"] = 0 if dec is None else len(dec)
        if aligned and lt is not None and lt < len(cw) and dec is not None:
            t_learn = cw.start_time.iloc[lt]
            post = dec.start_time.to_numpy() >= t_learn
            y = dec.lick_flag.to_numpy().astype(bool)
            row.update(t_learn=t_learn, n_pre=int((~post).sum()), n_post=int(post.sum()),
                       pre_hit=int((y & ~post).sum()), pre_miss=int((~y & ~post).sum()),
                       post_hit=int((y & post).sum()), post_miss=int((~y & post).sum()),
                       lt_frac=lt / len(cw))
        row["status"] = "ok" if aligned else "len_mismatch"
        rows.append(row)
        curves[s.session_id] = dict(curve=curve, row=row)
        print(f"{s.session_id} {s.reward_group} lt={lt} n={len(outcomes)} aligned={aligned} "
              f"match={row['outcome_match_frac']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "089_learning_trial_sanity.csv", index=False)

    # --- console summary ---
    ok = df[df.status == "ok"]
    print("\nstatus counts:\n", df.status.value_counts().to_string())
    print("\noutcome_match_frac (aligned sessions):", ok.outcome_match_frac.describe().round(3).to_dict())
    has_lt = ok[ok.learning_trial.notna()]
    print(f"sessions with a learning_trial: {len(has_lt)}/{len(ok)}; no learning_trial: {ok.learning_trial.isna().sum()}")
    for k in range(1, 11):
        m = (has_lt[f"rule_k{k}"] == has_lt.learning_trial).mean()
        print(f"  rule 'first run of >= {k} consecutive above-chance trials' matches: {m:.2%}")
    print("reconstruction match:", ok.lt_repro_match.mean())
    print(pd.crosstab(ok.lt_source, ok.reward_group).to_string())
    for cls in ["hit", "miss"]:
        for ep in ["pre", "post"]:
            print(f"  {ep}_{cls}: min={has_lt[f'{ep}_{cls}'].min()}, median={has_lt[f'{ep}_{cls}'].median()}")

    # --- figure 1: per-session curve grid ---
    sids = [sid for sid in df.session_id if sid in curves]
    ncol = 8
    nrow = int(np.ceil(len(sids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.2 * nrow), sharey=True)
    for ax, sid in zip(axes.flat, sids):
        c, r = curves[sid]["curve"], curves[sid]["row"]
        x = np.arange(len(c["p_mean"]))
        col = COHORT_COLOR[r["reward_group"]]
        ax.fill_between(x, c["p_low"], c["p_high"], color=col, alpha=0.2, lw=0)
        ax.plot(x, c["p_mean"], color=col, lw=1.3)
        ax.plot(x, c["p_chance"], color="#555555", lw=1, ls="--")
        o = np.asarray(c["outcomes"])
        ax.scatter(x, np.where(o == 1, 1.04, -0.04), s=3, color="k", clip_on=False)
        if r.get("learning_trial") is not None and not pd.isna(r.get("learning_trial")):
            ax.axvline(r["learning_trial"], color="#d62728", lw=1.4)
        pre, post = r.get("n_pre", np.nan), r.get("n_post", np.nan)
        ax.set_title(f"{r['subject_id']} {r['reward_group']} lt={r.get('learning_trial')}\n"
                     f"pre {r.get('pre_hit','-')}h/{r.get('pre_miss','-')}m  post {r.get('post_hit','-')}h/{r.get('post_miss','-')}m",
                     fontsize=7)
        ax.set_ylim(-0.08, 1.08)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(sids):]:
        ax.axis("off")
    fig.suptitle("Whisker learning curves (p_mean, 80% CI, dashed=p_chance, dots=lick outcomes); red = learning_trial", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUT_DIR / "089_learning_trial_curves_grid.png", dpi=150)

    # --- figure 2: summary ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for rg, sub in has_lt.groupby("reward_group"):
        axes[0].hist(sub.learning_trial, bins=np.arange(0, has_lt.learning_trial.max() + 10, 5), alpha=0.6,
                     color=COHORT_COLOR[rg], label=f"{rg} (n={len(sub)})")
        axes[1].hist(sub.lt_frac, bins=np.linspace(0, 1, 21), alpha=0.6, color=COHORT_COLOR[rg])
        axes[2].scatter(sub[["pre_hit", "pre_miss"]].min(axis=1), sub[["post_hit", "post_miss"]].min(axis=1),
                        color=COHORT_COLOR[rg], s=18, edgecolor="k", lw=0.4)
    axes[0].set_xlabel("learning_trial (whisker-trial index)")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("learning_trial / n whisker trials")
    lim = max(has_lt[["post_hit", "post_miss"]].min(axis=1).max(), 10) + 2
    axes[2].axvline(5, color="#888", ls=":"); axes[2].axhline(5, color="#888", ls=":")
    axes[2].set_xlabel("pre: min(hit, miss) count"); axes[2].set_ylabel("post: min(hit, miss) count")
    axes[2].set_xlim(-1, lim); axes[2].set_ylim(-1, lim)
    for ax in axes[:2]:
        ax.set_ylabel("sessions")
    fig.savefig(OUT_DIR / "089_learning_trial_summary.png", dpi=150)


if __name__ == "__main__":
    main()
