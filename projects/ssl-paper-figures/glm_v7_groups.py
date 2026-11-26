"""Group drop-and-refit of the glm_v7 lick head (user 2026-11-23, Figure 1J: "groups of predictors removed").

Three groups of predictors:
  - reward:   cumulative_reward_fraction, time_since_reward
  - movement: jaw, nose, whisker speed and pupil area before the stimulus (= glm_v7 'all_movement')
  - history:  licks on whisker trials n-1, n-2, n-3 and the spontaneous lick rate
Same model as glm_v7 (= glm_v6 lick head): cached design matrices (glm_v7/_design), the session's leave-one-session-out
C* (glm_v7/c_choice.csv), glm_v5 splits (5 contiguous blocks, gap 3, 10 repeats with the glm_v5 offsets), predictors
standardised on the training blocks. Per repeat: full model and model without the group, refitted at C*; delta = full
minus reduced held-out log-likelihood per trial (nats) and AUC (glm_v5 scores), mean over repeats.
Group shuffle (user 2026-11-23, Supp. Fig. 1 shuffle controls): as glm_v5 shuffle importance, the group's columns
permuted in the training and test trials (N_PERM = 10 permutations, glm_v5 seeds) and the model refitted at C*;
delta = full minus shuffled, mean over permutations and repeats.
Output: Figures/_cache/glm_v7/_groups/<mouse>_<session>_group_{drop,shuffle}.csv, group_drop_all.csv, group_shuffle_all.csv.
Run on haas (repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-paper-figures/glm_v7_groups.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_v5_run as v5  # noqa: E402
import shared_behaviour  # noqa: E402
from common import CACHE_DIR  # noqa: E402
from glm_cv_compare import block_splits  # noqa: E402

SRC = CACHE_DIR / ("glm_v7" + shared_behaviour.CUT_TAG)  # SSL_DAY0_KEEP (user 2026-11-25)
OUT = SRC / "_groups"
GROUPS = {"reward": ["cumulative_reward_fraction", "time_since_reward"],
          "movement": ["jaw_speed_pre", "nose_speed_pre", "whisker_speed_pre", "pupil_area_pre"],
          "history": ["lick_whisker_trial_n-1", "lick_whisker_trial_n-2", "lick_whisker_trial_n-3",
                      "spontaneous_lick_rate"]}


def run(f, c_star, mouse, cohort):
    z = np.load(f, allow_pickle=True)
    X, y, names = z["X"], z["y"].astype(int), list(z["names"])
    if len(np.unique(y)) < 2:
        return None
    v5.GROUPS = GROUPS
    cols = v5.group_columns(names)
    n = len(y)
    rng = np.random.default_rng(7)  # glm_v5 offsets
    offsets = [0] + [int(rng.integers(n)) for _ in range(v5.N_REPEATS - 1)]
    rows, srows = [], []
    for rep, off in enumerate(offsets):
        P = np.full(n, np.nan)
        thr = np.full(n, np.nan)
        Pd = {g: np.full(n, np.nan) for g in GROUPS}
        Ps = {g: np.full((v5.N_PERM, n), np.nan) for g in GROUPS}
        for k, (tr, te) in enumerate(block_splits(n, v5.N_FOLDS, offset=off, gap=v5.GAP)):
            if len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            P[te] = v5._fit(Xtr, y[tr], c_star).predict_proba(Xte)[:, 1]
            thr[te] = y[tr].mean()
            for g, idx in cols.items():
                keep = [i for i in range(X.shape[1]) if i not in set(idx)]
                Pd[g][te] = v5._fit(Xtr[:, keep], y[tr], c_star).predict_proba(Xte[:, keep])[:, 1]
            perm_rng = np.random.default_rng(1000 * rep + k)  # glm_v5 seeds
            for g, idx in cols.items():
                for j in range(v5.N_PERM):
                    Xtr_s, Xte_s = Xtr.copy(), Xte.copy()
                    for i in idx:
                        Xtr_s[:, i] = perm_rng.permutation(Xtr_s[:, i])
                        Xte_s[:, i] = perm_rng.permutation(Xte_s[:, i])
                    Ps[g][j, te] = v5._fit(Xtr_s, y[tr], c_star).predict_proba(Xte_s)[:, 1]
        ok = np.isfinite(P)
        full = v5.scores(y[ok], P[ok], thr[ok], thr[ok])
        for g in GROUPS:
            d = v5.scores(y[ok], Pd[g][ok], thr[ok], thr[ok])
            rows.append({"repeat": rep, "group": g, "n_columns": len(cols[g]),
                         "delta_test_ll_per_trial": full["test_ll_per_trial"] - d["test_ll_per_trial"],
                         "delta_test_auc": full["test_auc"] - d["test_auc"]})
            for j in range(v5.N_PERM):
                sh = v5.scores(y[ok], Ps[g][j, ok], thr[ok], thr[ok])
                srows.append({"repeat": rep, "perm": j, "group": g,
                              "delta_test_ll_per_trial": full["test_ll_per_trial"] - sh["test_ll_per_trial"],
                              "delta_test_auc": full["test_auc"] - sh["test_auc"],
                              "delta_test_balanced_accuracy": full["test_balanced_accuracy"] - sh["test_balanced_accuracy"]})
    sh = (pd.DataFrame(srows).groupby("group", sort=False)[["delta_test_ll_per_trial", "delta_test_auc",
                                                            "delta_test_balanced_accuracy"]].mean().reset_index())
    for c, v in (("cohort", cohort), ("session_id", f.stem), ("mouse_id", mouse)):
        sh.insert(0, c, v)
    sh.to_csv(OUT / f"{mouse}_{f.stem}_group_shuffle.csv", index=False)
    out = (pd.DataFrame(rows).groupby("group", sort=False)[["n_columns", "delta_test_ll_per_trial", "delta_test_auc"]]
           .mean().reset_index())
    out.insert(0, "cohort", cohort)
    out.insert(0, "session_id", f.stem)
    out.insert(0, "mouse_id", mouse)
    out["C_star"] = c_star
    out.to_csv(OUT / f"{mouse}_{f.stem}_group_drop.csv", index=False)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv(SRC / "c_choice.csv").set_index("session_id").C_star
    ref = shared_behaviour.mouse_reference().set_index("mouse_id").cohort
    jobs = []
    for f in sorted((SRC / "_design").glob("*.npz")):
        mouse = f.stem.split("_")[0]
        if f.stem in c.index and mouse in ref.index:
            jobs.append((f, float(c[f.stem]), mouse, ref[mouse]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = Parallel(n_jobs=100, verbose=1)(delayed(run)(*j) for j in jobs)
    df = pd.concat([r for r in res if r is not None], ignore_index=True)
    df.to_csv(OUT / "group_drop_all.csv", index=False)
    sh = pd.concat([pd.read_csv(OUT / f"{m}_{f.stem}_group_shuffle.csv") for f, _, m, _ in jobs
                    if (OUT / f"{m}_{f.stem}_group_shuffle.csv").exists()], ignore_index=True)
    sh.to_csv(OUT / "group_shuffle_all.csv", index=False)
    print(sh.groupby(["group", "cohort"]).delta_test_ll_per_trial.agg(["mean", "sem", "size"]).round(4).to_string())
    print(df.groupby(["group", "cohort"])[["delta_test_ll_per_trial", "delta_test_auc"]].agg(["mean", "sem", "size"])
          .round(4).to_string())


if __name__ == "__main__":
    main()
