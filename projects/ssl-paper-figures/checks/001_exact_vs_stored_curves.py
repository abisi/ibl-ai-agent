"""Exact re-fit (learning_curve_exact, sigma 1, 80% CI) vs the stored ks4 PyMC curves
(combined_results_ks4/<mouse>/whisker_0/learning_curve/<mouse>_whisker_0_whisker_trial_learning_curve_interp.h5).

Per whisker day-0 session: learning category, learning trial, and per-trial performance state
(learning_utils.assign_expertise_blocks, n_consecutive 5). Two exact fits:
  same_trials  -- the stored pipeline's trial set (active, early_lick == 0, no warm-up cut): method change only;
  figure_trials -- the figure's shared table (warm-up block cut): what the figure will show.
Output: Figures/_cache/checks/001_exact_vs_stored_curves.csv and a printed summary.

Run on haas (repo root):
    nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/checks/001_exact_vs_stored_curves.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import learning_curve_exact as lce  # noqa: E402
import shared_behaviour  # noqa: E402
from common import CACHE_DIR, axel_bisi_root, import_external_repos  # noqa: E402

import_external_repos()
import learning_utils  # noqa: E402

STORED = axel_bisi_root() / "combined_results_ks4"
SIGMA, CONF_INT = 1.0, 80


def states(mouse_id, reward_group, p_mean, p_low, p_high, p_chance):
    """Per whisker trial 'low'/'high' with the repo rule (learning_utils.assign_expertise_blocks)."""
    n = len(p_low)
    df = pd.DataFrame({"mouse_id": mouse_id, "reward_group": reward_group, "trial_id": np.arange(n),
                       "whisker_trial_id": np.arange(n), "trial_type": "whisker_trial",
                       "p_mean": p_mean, "p_low": p_low, "p_high": p_high, "p_chance": p_chance})
    return learning_utils.assign_expertise_blocks(df, n_consecutive=5).block_perf_type.to_numpy()


def exact(session_table):
    out = lce.session_learning_curves_exact(session_table, sigma=SIGMA, conf_int=CONF_INT)
    return out.get("whisker_trial") if "whisker_trial" in out and "p_chance" in out["whisker_trial"] else None


def main():
    table = shared_behaviour.load()
    raw = shared_behaviour.load_untrimmed_day0()
    rows = []
    for sid, st in table[(table.behavior == "whisker") & (table.day == 0)].groupby("session_id"):
        mouse, rg = st.mouse_id.iloc[0], int(st.reward_group.iloc[0])
        f = STORED / mouse / "whisker_0" / "learning_curve" / f"{mouse}_whisker_0_whisker_trial_learning_curve_interp.h5"
        if not f.exists():
            rows.append(dict(mouse_id=mouse, session_id=sid, cohort=st.cohort.iloc[0], stored="missing"))
            continue
        s = pd.read_hdf(f).iloc[0]
        row = dict(mouse_id=mouse, session_id=sid, cohort=st.cohort.iloc[0], stored="ok", n_stored=len(s.outcomes),
                   cat_stored=s.mouse_cat, lt_stored=s.learning_trial)
        s_states = states(mouse, rg, s.p_mean, s.p_low, s.p_high, s.p_chance)
        row["frac_high_stored"] = float(np.mean(s_states == "high"))
        for label, src in (("same", raw[raw.session_id == sid]), ("fig", st)):
            src = src[src.early_lick == 0].sort_values("start_time")
            w = exact(src)
            if w is None:
                row[f"cat_{label}"] = "no fit"
                continue
            row.update({f"n_{label}": len(w["outcomes"]), f"cat_{label}": w["mouse_cat"], f"lt_{label}": w["learning_trial"]})
            e_states = states(mouse, rg, w["p_mean"], w["p_low"], w["p_high"], w["p_chance"])
            row[f"frac_high_{label}"] = float(np.mean(e_states == "high"))
            if len(e_states) == len(s_states):
                row[f"state_agree_{label}"] = float(np.mean(e_states == s_states))
            if label == "same":
                row["outcomes_identical"] = bool(len(w["outcomes"]) == len(s.outcomes)
                                                 and np.array_equal(w["outcomes"], s.outcomes))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = CACHE_DIR / "checks"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "001_exact_vs_stored_curves.csv", index=False)

    ok = df[df.stored == "ok"]
    print(f"sessions {len(df)}, stored curve found {len(ok)}, missing {int((df.stored == 'missing').sum())}")
    print("identical whisker outcomes (same trial set):", int(ok.outcomes_identical.sum()), "/", len(ok))
    for label in ("same", "fig"):
        cat_agree = (ok.cat_stored == ok[f"cat_{label}"])
        both = ok[(ok.cat_stored == "learner") & (ok[f"cat_{label}"] == "learner")]
        dlt = (both[f"lt_{label}"] - both.lt_stored).abs()
        print(f"\n[{label}] category agrees {int(cat_agree.sum())}/{len(ok)}")
        print(pd.crosstab(ok.cat_stored, ok[f"cat_{label}"], margins=True).to_string())
        print(f"[{label}] learning trial, learners in both (n={len(both)}): identical {int((dlt == 0).sum())}, "
              f"|diff| <= 3 {int((dlt <= 3).sum())}, median |diff| {dlt.median():.1f}, max {dlt.max():.0f}")
        print(f"[{label}] per-trial state agreement: median {ok[f'state_agree_{label}'].median():.3f}, "
              f"min {ok[f'state_agree_{label}'].min():.3f}, sessions < 0.95: {int((ok[f'state_agree_{label}'] < 0.95).sum())}")
        for c in ("R+", "R-"):
            o = ok[ok.cohort == c]
            print(f"[{label}] {c}: learners stored {int((o.cat_stored == 'learner').sum())}, exact "
                  f"{int((o[f'cat_{label}'] == 'learner').sum())} of {len(o)}")


if __name__ == "__main__":
    main()
