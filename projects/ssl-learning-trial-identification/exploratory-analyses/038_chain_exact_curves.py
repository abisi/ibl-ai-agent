"""038 -- Learning-trial chain rerun on the CURRENT exact learning curves (user 2026-11-23: "rerun that project with exact
curves"). Same chain as 028 (003 rules -> 004 whisker CP -> 005 joint CP -> 006 half-way + final table -> 007 lenient LTs
-> 013 all-methods table), with two changes:
  inputs  whisker and no-stim outcomes / start times from the stored exact curve files
          (combined_results_ks4/<mouse>/whisker_0/learning_curve/<mouse>_whisker_0_<type>_learning_curve_exact.h5,
          2026-11-03; trials: not passive, perf != 6, early_lick == 0), i.e. the trial set of the paper curves;
  curves  002 replaced: whisker and no-stim curves at FIXED sigma = 1 (the paper's exact model, grid HMM; lt_lib forward-
          backward), written into BOTH slots 'orig' and 'eb' of 002_curves.pkl, so every curve-based rule (L3, L7, L8,
          and the 'eb'-based quantities of 004-007) runs on the paper curves. Checked against the stored p_mean.
The stored (old PyMC) curve stays as L0 where its trials match; otherwise L0 is undefined.
Workspace: artifacts/038_chain_exact/ (inputs, every chain artifact), figures in exploratory-analyses/038_chain_exact_figs/.
Final table: artifacts/038_learning_trials_all_methods_exact.csv (+ column per method of the old 028 table for comparison
in 037).
Run (haas, repo root): ~/code/unit_spikes_analysis/.venv/bin/python projects/ssl-learning-trial-identification/exploratory-analyses/038_chain_exact_curves.py
"""
from __future__ import annotations

import importlib.util
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
WS = ART / "038_chain_exact"
FIGS = HERE / "038_chain_exact_figs"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
import lt_lib as L  # noqa: E402

ROOT = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
CHAIN = ["003_learning_trial_rules.py", "004_bayesian_changepoint.py", "005_joint_changepoint.py",
         "006_halfway_rule_and_final_table.py", "007_lenient_learning_trials.py", "013_methods_schematic_and_examples.py"]
SIGMA = 1.0


def exact(mouse, tt):
    f = ROOT / mouse / "whisker_0" / "learning_curve" / f"{mouse}_whisker_0_{tt}_learning_curve_exact.h5"
    return pd.read_hdf(f).iloc[0] if f.exists() else None


def fit_sigma1(y):
    g, lz = L.forward_backward(np.asarray(y).astype(int), SIGMA)
    return dict(marg=g, p_mean=g @ L.P_GRID, sigma_post_mean=SIGMA, log_evidence=float(lz))


def curves_for(item):
    sid, d = item
    out = {"meta": {k: d[k] for k in ("session_id", "mouse_id", "reward_group", "learning_category")}}
    ws, ns = d["w_start"], d["n_start"]
    w, n = fit_sigma1(d["w_outcomes"]), fit_sigma1(d["n_outcomes"])
    grid = np.linspace(min(ws.min(), ns.min()), max(d["w_stop"].max(), ns.max()), len(n["p_mean"]))
    fa_even = CubicSpline(grid, n["p_mean"], extrapolate=False)(ws)
    fa_marg = L.interp_marginals(n["marg"], ns, ws)
    lo80, hi80, lo90, hi90 = L.quantiles(w["marg"], [0.1, 0.9, 0.05, 0.95])
    c = dict(p_mean=w["p_mean"], p_low80=lo80, p_high80=hi80, p_low90=lo90, p_high90=hi90, fa_evengrid=fa_even,
             fa_time=fa_marg @ L.P_GRID, p_above=L.prob_greater(w["marg"], fa_marg), sigma=SIGMA, sigma_fa=SIGMA,
             logev=w["log_evidence"], logev_fa=n["log_evidence"], marg_w=w["marg"].astype(np.float32),
             marg_fa=fa_marg.astype(np.float32))
    out["orig"], out["eb"] = c, c
    return sid, out


def load(fname):
    spec = importlib.util.spec_from_file_location("m" + fname[:3], HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    m.ART, m.HERE = WS, FIGS
    return m


def main():
    WS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    old = pickle.load(open(ART / "028_chain_all" / "001_inputs.pkl", "rb"))
    inputs, log = {}, []
    for sid, o in old.items():
        m = o["mouse_id"]
        w, n = exact(m, "whisker_trial"), exact(m, "no_stim_trial")
        if w is None or n is None or w.session_id != sid:
            log.append(dict(session_id=sid, status="no exact curve for this session"))
            continue
        wo, wst = np.asarray(w.outcomes).astype(int), np.asarray(w.start_time, float)
        st = o["stored"]
        if len(st["p_mean"]) != len(wo):     # stored (old PyMC) curve on another trial set: L0 undefined
            nan = np.full(len(wo), np.nan)
            st = dict(p_mean=nan, p_low=nan, p_high=nan, p_chance=nan, learning_trial=np.nan, mouse_cat="", fa_p_mean=nan)
        inputs[sid] = dict(session_id=sid, mouse_id=m, reward_group=o["reward_group"],
                           learning_category=o["learning_category"], w_outcomes=wo, w_start=wst, w_stop=wst + 1.0,
                           n_outcomes=np.asarray(n.outcomes).astype(int), n_start=np.asarray(n.start_time, float), stored=st)
        log.append(dict(session_id=sid, status="ok", n_w_old=len(o["w_outcomes"]), n_w_exact=len(wo),
                        same_w=bool(len(wo) == len(o["w_outcomes"]) and np.array_equal(wo, o["w_outcomes"])),
                        n_ns_old=len(o["n_outcomes"]), n_ns_exact=len(n.outcomes)))
    pickle.dump(inputs, open(WS / "001_inputs.pkl", "wb"))
    pd.DataFrame(log).to_csv(WS / "038_inputs_log.csv", index=False)
    print(pd.DataFrame(log).status.value_counts().to_dict(), flush=True)
    with ProcessPoolExecutor(max_workers=24) as ex:
        curves = dict(ex.map(curves_for, inputs.items()))
    pickle.dump(curves, open(WS / "002_curves.pkl", "wb"))
    dev = []
    for sid, c in curves.items():
        w = exact(inputs[sid]["mouse_id"], "whisker_trial")
        dev.append(np.abs(np.asarray(w.p_mean, float) - c["eb"]["p_mean"]).max())
    print(f"sigma-1 refit vs stored exact p_mean: median max|diff| {np.median(dev):.4f}, max {np.max(dev):.4f}", flush=True)
    for f in CHAIN:
        print(f"--- {f}", flush=True)
        load(f).main()
    t = pd.read_csv(WS / "013_learning_trials_all_methods.csv")
    t.to_csv(ART / "038_learning_trials_all_methods_exact.csv", index=False)
    ref = pd.read_csv(ART / "028_learning_trials_all_methods_all_mice.csv").set_index("session_id")
    defs = ["L0 stored", "L3 sustained prob.", "L5 whisker CP", "L7 half-way", "L8 fixed margin", "L6 joint CP",
            "L6 lenient", "lenient cascade"]
    rows = []
    for dname in defs:
        for coh in ("R+", "R-"):
            a = t[t.reward_group == coh].set_index("session_id")[dname]
            b = ref.loc[a.index, dname]
            both = a.notna() & b.notna()
            rows.append(dict(method=dname, cohort=coh, n_old=int(b.notna().sum()), n_exact=int(a.notna().sum()),
                             both=int(both.sum()), identical=int((both & (a == b)).sum()),
                             median_abs_diff=float(np.median(np.abs(a[both] - b[both]))) if both.any() else np.nan))
    cmp = pd.DataFrame(rows)
    cmp.to_csv(ART / "038_exact_vs_028.csv", index=False)
    pd.set_option("display.width", 200)
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
