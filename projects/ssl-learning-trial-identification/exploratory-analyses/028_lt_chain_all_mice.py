"""028 -- Learning-trial definitions for the behaviour-only mice with the SAME method as the ephys mice (user 2026-10-01:
"Run the learning curves with the same method as the others for behaviour only mice").
The LT chain (002 exact curves -> 003 rules L0-L4 -> 004 whisker CP L5 -> 005 joint CP L6 -> 006 half-way L7 + final
table -> 007 lenient LTs + clean gate -> 013 all-methods table incl. L8) is run UNCHANGED (modules imported, their ART /
HERE redirected to a separate workspace) on the union of the 88 ephys sessions (001) and the 12 behaviour-only sessions
(022, NWB_ks4). Running all 100 lets the 88 act as a check: their rows must reproduce artifacts/013 exactly.
Behaviour-only inputs get a `stored` entry from their stored curve file (combined_results_ks4/<mouse>/whisker_0/
learning_curve, PyMC, as 001a) when it exists and matches; otherwise (no file / length mismatch) stored arrays are NaN and
L0 is undefined for that session (flagged in stored_source).
Workspace: artifacts/028_chain_all/ (inputs + every chain artifact), figures in exploratory-analyses/028_chain_all_figs/.
Final table: artifacts/028_learning_trials_all_methods_all_mice.csv (+ column source, stored_source, check vs 013).
Run (haas, repo root): python projects/ssl-learning-trial-identification/exploratory-analyses/028_lt_chain_all_mice.py
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ART = HERE.parent / "artifacts"
WS = ART / "028_chain_all"
FIGS = HERE / "028_chain_all_figs"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "scripts"))
CHAIN = ["002_exact_curves.py", "003_learning_trial_rules.py", "004_bayesian_changepoint.py", "005_joint_changepoint.py",
         "006_halfway_rule_and_final_table.py", "007_lenient_learning_trials.py", "013_methods_schematic_and_examples.py"]


def stored_for(inp):
    from axel_bisi_paths import axel_bisi_path
    from ssl_timeresolved_decoding import load_whisker_curve_row
    n = len(inp["w_outcomes"])
    nan = np.full(n, np.nan)
    empty = dict(p_mean=nan, p_low=nan, p_high=nan, p_chance=nan, learning_trial=np.nan, mouse_cat="", fa_p_mean=nan)
    root = axel_bisi_path("combined_results_ks4")
    wc = load_whisker_curve_row(root, inp["mouse_id"], 0, "whisker_trial", interp=True)
    nc = load_whisker_curve_row(root, inp["mouse_id"], 0, "no_stim_trial", interp=False)
    if wc is None or nc is None:
        return empty, "no stored curve file"
    if len(wc["outcomes"]) != n or not np.array_equal(np.asarray(wc["outcomes"]).astype(int), inp["w_outcomes"]):
        return empty, f"stored whisker curve differs ({len(wc['outcomes'])} vs {n} trials)"
    return dict(p_mean=np.asarray(wc["p_mean"]), p_low=np.asarray(wc["p_low"]), p_high=np.asarray(wc["p_high"]),
                p_chance=np.asarray(wc["p_chance"]), learning_trial=wc["learning_trial"], mouse_cat=wc["mouse_cat"],
                fa_p_mean=np.asarray(nc["p_mean"])), "stored curve file"


def load(fname):
    spec = importlib.util.spec_from_file_location("m" + fname[:3], HERE / fname)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m          # pool workers must be able to pickle the module's functions by name
    spec.loader.exec_module(m)
    m.ART, m.HERE = WS, FIGS            # redirect every artifact / figure written by main()
    return m


def main():
    WS.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    a = pickle.load(open(ART / "001_inputs.pkl", "rb"))
    b = pickle.load(open(ART / "022_inputs_behaviour_only.pkl", "rb"))
    src = {s: "001 ephys" for s in a}
    sto = {s: "stored curve file" for s in a}
    for sid, inp in b.items():
        st, how = stored_for(inp)
        a[sid] = dict(session_id=sid, mouse_id=inp["mouse_id"], reward_group=inp["reward_group"],
                      learning_category=inp["learning_category"], w_outcomes=inp["w_outcomes"], w_start=inp["w_start"],
                      w_stop=inp["w_stop"], n_outcomes=inp["n_outcomes"], n_start=inp["n_start"], stored=st)
        src[sid], sto[sid] = "022 behaviour-only (NWB_ks4)", how
        print(sid, how, flush=True)
    pickle.dump(a, open(WS / "001_inputs.pkl", "wb"))
    for f in CHAIN:
        print(f"--- {f}", flush=True)
        load(f).main()
    t = pd.read_csv(WS / "013_learning_trials_all_methods.csv")
    t.insert(1, "source", t.session_id.map(src))
    t.insert(2, "stored_source", t.session_id.map(sto))
    ref = pd.read_csv(ART / "013_learning_trials_all_methods.csv").set_index("session_id")
    defs = ["L0 stored", "L1 stored rule, exact", "L2 stored rule, smooth", "L3 sustained prob.", "L5 whisker CP",
            "L7 half-way", "L8 fixed margin", "L6 joint CP", "L6 lenient", "L5w lenient (R+)", "lenient cascade",
            "lenient cascade + clean gate"]
    chk = []
    for _, r in t[t.session_id.isin(ref.index)].iterrows():
        for d in defs:
            x, y = r[d], ref.loc[r.session_id, d]
            chk.append((pd.isna(x) and pd.isna(y)) or (not pd.isna(x) and not pd.isna(y) and abs(x - y) < 1e-9))
    print(f"check vs artifacts/013 on the {int(t.session_id.isin(ref.index).sum())} ephys sessions: "
          f"{sum(chk)}/{len(chk)} LT values identical", flush=True)
    t["matches_013"] = t.session_id.map(lambda s: None if s not in ref.index else all(
        (pd.isna(t.loc[t.session_id == s, d].iloc[0]) and pd.isna(ref.loc[s, d])) or
        abs(t.loc[t.session_id == s, d].iloc[0] - ref.loc[s, d]) < 1e-9 for d in defs))
    t.to_csv(ART / "028_learning_trials_all_methods_all_mice.csv", index=False)
    new = t[t.source.str.startswith("022")]
    pd.set_option("display.width", 250)
    print(new[["session_id", "reward_group", "learning_category", "stored_source"] + defs].to_string(index=False))


if __name__ == "__main__":
    main()
