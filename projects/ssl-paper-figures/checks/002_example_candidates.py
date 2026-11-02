"""Example-session candidates for Figure 1 row 1, by reference-sheet learning_category (good/moderate = learner,
bad = non-learner): per whisker day-0 session, n whisker trials, whisker hit rate, FA rate, exact-curve learning trial
and the fraction of whisker trials in the high-performance state (learning_utils.assign_expertise_blocks).

Run on haas (repo root):
    nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/checks/002_example_candidates.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import learning_curve_exact as lce  # noqa: E402
import shared_behaviour  # noqa: E402
from common import CACHE_DIR, import_external_repos  # noqa: E402

import_external_repos()
import learning_utils  # noqa: E402


def main():
    table = shared_behaviour.load()
    rows = []
    for sid, st in table[(table.behavior == "whisker") & (table.day == 0)].groupby("session_id"):
        st = st[st.early_lick == 0].sort_values("start_time")
        out = lce.session_learning_curves_exact(st, sigma=1.0, conf_int=80)
        w = out.get("whisker_trial", {})
        if "p_chance" not in w:
            continue
        n = len(w["p_low"])
        blocks = learning_utils.assign_expertise_blocks(pd.DataFrame({
            "mouse_id": st.mouse_id.iloc[0], "reward_group": int(st.reward_group.iloc[0]), "trial_id": np.arange(n),
            "trial_type": "whisker_trial", "p_mean": w["p_mean"], "p_low": w["p_low"], "p_high": w["p_high"],
            "p_chance": w["p_chance"]}), n_consecutive=5)
        rows.append(dict(mouse_id=st.mouse_id.iloc[0], session_id=sid, cohort=st.cohort.iloc[0],
                         sheet=st.learning_category.iloc[0], n_whisker=n,
                         hit_rate=float(np.mean(w["outcomes"])),
                         fa_rate=float(st[st.trial_type == "no_stim_trial"].lick_flag.mean()),
                         auditory_hit_rate=float(st[st.trial_type == "auditory_trial"].lick_flag.mean()),
                         learning_trial=w["learning_trial"], rule_category=w["mouse_cat"],
                         frac_high=float((blocks.block_perf_type == "high").mean())))
    df = pd.DataFrame(rows).sort_values(["cohort", "sheet", "mouse_id"])
    out = CACHE_DIR / "checks"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "002_example_candidates.csv", index=False)
    pd.set_option("display.width", 200)
    for (cohort, sheet), g in df.groupby(["cohort", "sheet"]):
        print(f"\n== {cohort} {sheet} (n={len(g)}), median learning trial {g.learning_trial.median():.0f}")
        print(g.drop(columns=["session_id", "cohort", "sheet"]).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
