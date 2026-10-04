"""108 -- Neural-behaviour coupling model, example sessions (user 2026-09-26: "Example coupling model?").

Question: does the neural signal predict licking BETTER after learning than before?
Per session (whole brain, all learning-stage whisker trials, disengaged kept, 8 example sessions of 106/107):
 1. Neural projection z_i: cross-validated decision value of the class-balanced L2 hit/miss decoder
    (StandardScaler + logistic, C from select_fixed_c_pooled), N_REP x stratified 5-fold, averaged over
    repeats; every z_i comes from a decoder that never saw trial i. z is z-scored over the session.
 2. Learning state s_i:
      "lt"  : 1 if trial i is at/after the cohort-specific learning trial (020, learners);
              non-learners use the session midpoint instead (flagged split_ref="midpoint").
      "pos" : trial position scaled to [0, 1] (agnostic to any LT).
 3. Coupling GLM:  logit P(lick_i) = b0 + b1 z_i + b2 s_i + b3 z_i s_i
      b1       = coupling before learning (s = 0): how steeply lick probability rises with the neural signal
      b1 + b3  = coupling after learning (s = 1)
      b3       = CHANGE in coupling (the test); b2 absorbs the change in overall lick rate.
    Fit with a mild L2 penalty (C = 100 on standardised z) so partial separation does not diverge;
    inference is by the null, not by Wald SEs.
 4. Linear shift null: N_SHIFT distinct shifts (10-50% of the session, both directions) of the behaviour
    (labels and s) relative to the neural data; the decoder is retrained on the shifted pairs and the GLM
    refitted -> null distribution of b3 that keeps slow drift in both. Percentile of the real b3.
Windows: sensory (5-50 ms) and sensory minus baseline.
Outputs: 108_coupling_examples.parquet (one row per session x window x state; full provenance),
         108_coupling_examples.pkl (per-trial z, y, s), figures/whole_brain/learning/108_coupling_examples_<window>.png/.pdf
Run: python 108_coupling_model_examples.py [plot]
"""

from __future__ import annotations

import os
import pickle
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS = str(Path(__file__).resolve().parents[3] / "scripts")
sys.path.insert(0, SCRIPTS)
OUT = Path(__file__).resolve().parent
LTP = OUT.parents[1] / "ssl-learning-trial-identification"
EXAMPLES = ["AB119", "AB125", "MH029", "AB085", "MH018", "AB120", "MH030", "AB159"]
WINDOWS = {"sensory": (0.005, 0.050), "baseline": (-0.200, -0.010)}
WIN_USE = ["sensory", "sensory_minus_base"]
N_REP, N_SHIFT, GLM_C = 5, 50, 100.0
DZ = (-0.010, 0.005)


def _init():
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"


def cv_projection(X, y, C, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    nf = min(5, int(y.sum()), int((~y).sum()))
    z = np.zeros(len(y))
    for _ in range(N_REP):
        for tr, te in StratifiedKFold(nf, shuffle=True, random_state=int(rng.integers(1 << 31))).split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(penalty="l2", solver="liblinear", C=C,
                                                                   class_weight="balanced", max_iter=1000))
            z[te] += m.fit(X[tr], y[tr]).decision_function(X[te])
    z /= N_REP
    return (z - z.mean()) / (z.std() + 1e-12)


def glm(z, s, y):
    from sklearn.linear_model import LogisticRegression
    D = np.column_stack([z, s, z * s])
    m = LogisticRegression(penalty="l2", C=GLM_C, max_iter=5000).fit(D, y)
    return np.r_[m.intercept_, m.coef_[0]]  # b0, b1, b2, b3


def process(args):
    key, X, y, states, C, seed = args
    rng = np.random.default_rng(seed)
    n = len(y)
    z = cv_projection(X, y, C, rng)
    real = {st: glm(z, s, y) for st, s in states.items()}
    ks = [(d, k) for k in range(int(0.1 * n), int(0.5 * n) + 1) for d in (0, 1)]
    ks = [ks[i] for i in rng.choice(len(ks), min(N_SHIFT, len(ks)), replace=False)]
    null = {st: [] for st in states}
    for d, k in ks:
        Xs, sl = (X[:n - k], slice(k, n)) if d == 0 else (X[k:], slice(0, n - k))
        ys = y[sl]
        if min(ys.sum(), (~ys).sum()) < 3:
            continue
        zs = cv_projection(Xs, ys, C, rng)
        for st, s in states.items():
            if np.ptp(s[sl]) > 0:
                null[st].append(glm(zs, s[sl], ys))
    return key, dict(z=z, real=real, null={st: np.array(v) for st, v in null.items()})


def main():
    from ibl_ai_agent.data_locations import resolve_dataset_dir
    from ssl_timeresolved_decoding import (
        AREA_LABELS_PATH, _active_trials_from_whisker_onset_for_curve, add_whole_brain_column, area_units,
        load_session_unit_spikes, prep_hitmiss_trials, select_fixed_c_pooled, sliding_bin_population_matrices,
    )
    root = resolve_dataset_dir("ssl_ephys")
    st, tt = pd.read_parquet(root / "metadata" / "sessions.parquet"), pd.read_parquet(root / "metadata" / "trials.parquet")
    labels = add_whole_brain_column(pd.read_parquet(AREA_LABELS_PATH))
    tab = pd.read_csv(LTP / "artifacts" / "020_lt_eval.csv").set_index("session_id")
    sids = [next(s for s in tab.index if s.startswith(p)) for p in EXAMPLES]
    jobs, meta = [], {}
    for sid in sids:
        tr = prep_hitmiss_trials(root, sid, st, tt)
        y = tr["lick_flag"].to_numpy().astype(bool)
        t = tr["start_time"].to_numpy()
        n = len(y)
        cw = _active_trials_from_whisker_onset_for_curve(sid, tt)
        cw_t = cw.loc[cw["trial_type"] == "whisker_trial", "start_time"].to_numpy()
        lt = tab.loc[sid, "lt_cohort"]
        learner = tab.loc[sid, "group"] == "learner" and not pd.isna(lt)
        split = int(np.sum(t < cw_t[int(lt)])) if learner else n // 2
        states = {"lt": (np.arange(n) >= split).astype(float), "pos": np.arange(n) / (n - 1)}
        units = area_units(sid, "whole_brain", "All units", labels)
        mats = sliding_bin_population_matrices(load_session_unit_spikes(root, sid), units, t, np.ones(n, bool),
                                               [WINDOWS["sensory"], WINDOWS["baseline"]], dead_zone=DZ)
        feats = {"sensory": mats[0], "sensory_minus_base": mats[0] - mats[1]}
        meta[sid] = dict(mouse_id=sid.split("_")[0], reward_group=tab.loc[sid, "reward_group"], group=tab.loc[sid, "group"],
                         split=split, split_ref="learning_trial" if learner else "midpoint", lt_curve=lt, y=y,
                         states=states, n_units=len(units))
        rng = np.random.default_rng(zlib.crc32(sid.encode()))
        for w in WIN_USE:
            C = select_fixed_c_pooled(feats[w], y, rng, n_folds=5)
            meta[sid][f"C_{w}"] = C
            jobs.append(((sid, w), feats[w], y, states, C, int(rng.integers(1 << 31))))
    with ProcessPoolExecutor(max_workers=len(jobs), initializer=_init) as ex:
        res = dict(ex.map(process, jobs))
    rows = []
    for (sid, w), r in res.items():
        m = meta[sid]
        for stn, b in r["real"].items():
            nb = r["null"][stn]
            rows.append(dict(session_id=sid, mouse_id=m["mouse_id"], reward_group=m["reward_group"], group=m["group"],
                             window=w, state=stn, split_ref=m["split_ref"] if stn == "lt" else "position",
                             split_trial=m["split"] if stn == "lt" else np.nan, n_trials=len(m["y"]), n_hits=int(m["y"].sum()),
                             n_units=m["n_units"], C_decoder=m[f"C_{w}"], glm_C=GLM_C, n_rep=N_REP,
                             b0=b[0], b1=b[1], b2=b[2], b3=b[3], coupling_pre=b[1], coupling_post=b[1] + b[3],
                             n_null=len(nb), null_b3_mean=nb[:, 3].mean() if len(nb) else np.nan,
                             b3_pct=(nb[:, 3] < b[3]).mean() if len(nb) else np.nan,
                             b3_p_two=min(1.0, 2 * min(((nb[:, 3] >= b[3]).sum() + 1) / (len(nb) + 1),
                                                       ((nb[:, 3] <= b[3]).sum() + 1) / (len(nb) + 1))) if len(nb) else np.nan))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "108_coupling_examples.parquet", index=False)
    pickle.dump(dict(res=res, meta=meta), open(OUT / "108_coupling_examples.pkl", "wb"))
    pd.set_option("display.width", 250)
    print(df[["session_id", "reward_group", "group", "window", "state", "coupling_pre", "coupling_post", "b3", "null_b3_mean",
              "b3_pct", "b3_p_two", "n_null"]].round(3).to_string())
    plot(res, meta, df)


def plot(res, meta, df):
    import importlib.util

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _s = importlib.util.spec_from_file_location("q034", OUT / "034_area_window_quant_grid.py")
    q034 = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(q034)
    col = {"R+": "#00B400", "R-": "#C800C8"}
    zz = np.linspace(-3, 3, 200)
    for w in WIN_USE:
        sids = list(meta)
        fig, axes = plt.subplots(3, len(sids), figsize=(3.9 * len(sids), 10), constrained_layout=True)
        for c, sid in enumerate(sids):
            m, r = meta[sid], res[(sid, w)]
            rg, y, z = m["reward_group"], m["y"], r["z"]
            n, k = len(y), m["split"]
            ref = "LT" if m["split_ref"] == "learning_trial" else "midpoint"
            # row 1: neural projection over trials
            ax = axes[0, c]
            ax.scatter(np.where(y)[0], z[y], s=9, color=col[rg], label="hit (lick)", lw=0)
            ax.scatter(np.where(~y)[0], z[~y], s=9, color="#999999", label="miss", lw=0)
            ax.axvline(k, color="#1f77b4", lw=1.6, label=ref)
            ax.set_title(f"{sid[:5]} {rg} {m['group']}\n{n} trials, {int(y.sum())} hits, {m['n_units']} units", fontsize=8.5)
            ax.set_xlabel("whisker trial")
            ax.legend(fontsize=6.5, frameon=False, loc="upper right")
            # row 2: P(lick | z) before vs after, data + fitted GLM
            ax = axes[1, c]
            b = r["real"]["lt"]
            pre = np.arange(n) < k
            rng = np.random.default_rng(0)
            for msk, s, lab, cc in ((pre, 0, f"before {ref}", "#9ecae1"), (~pre, 1, f"after {ref}", "#08519c")):
                ax.scatter(z[msk], y[msk] + rng.uniform(-0.04, 0.04, msk.sum()) + (0.06 if s else -0.06), s=7, color=cc, lw=0, alpha=0.8)
                ax.plot(zz, 1 / (1 + np.exp(-(b[0] + b[1] * zz + b[2] * s + b[3] * zz * s))), color=cc, lw=2,
                        label=f"{lab}: slope {b[1] + b[3] * s:.2f} (n={msk.sum()}, {int(y[msk].sum())} hits)")
            ax.set_xlim(-3.2, 3.2)
            ax.set_ylim(-0.15, 1.15)
            ax.set_xlabel("neural projection z (cross-validated, z-scored)")
            ax.set_ylabel("P(lick)" if c == 0 else "")
            ax.legend(fontsize=6.3, frameon=False, loc="upper left")
            # row 3: shift null of b3
            ax = axes[2, c]
            for stn, cc in (("lt", "#08519c"), ("pos", "#ff7f0e")):
                nb = r["null"][stn][:, 3] if len(r["null"][stn]) else np.array([])
                row = df[(df.session_id == sid) & (df.window == w) & (df.state == stn)].iloc[0]
                if len(nb):
                    ax.hist(nb, bins=15, color=cc, alpha=0.35)
                ax.axvline(row.b3, color=cc, lw=2, label=f"{'pre/post ' + ref if stn == 'lt' else 'trial position'}: "
                                                         f"b3={row.b3:.2f}, pct={row.b3_pct:.2f}, p={row.b3_p_two:.2g}")
            ax.axvline(0, color="#888888", ls=":")
            ax.set_xlabel("b3 = change in coupling (real line vs shift null)")
            ax.legend(fontsize=6.3, frameon=False, loc="upper left")
            ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
        fig.suptitle(f"Coupling model ({w}, whole brain): logit P(lick) = b0 + b1 z + b2 s + b3 z s. Row 1: cross-validated "
                     f"decoder projection z per trial. Row 2: P(lick) vs z before/after the split (dots = trials, lines = GLM); "
                     f"slope = coupling. Row 3: b3 (after minus before coupling) vs {N_SHIFT}-shift linear null.", fontsize=10)
        q034.savefig_retry(fig, q034.fig_dir("whole_brain") / f"108_coupling_examples_{w}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    print("plotted")


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        d = pickle.load(open(OUT / "108_coupling_examples.pkl", "rb"))
        plot(d["res"], d["meta"], pd.read_parquet(OUT / "108_coupling_examples.parquet"))
    else:
        main()
