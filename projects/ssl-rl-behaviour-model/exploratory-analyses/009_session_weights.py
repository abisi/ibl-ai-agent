"""009 -- Session-start body weight per NWB file (user 2026-10-18: "flag NWB files without start bodyweight"), for an optional thirst
state in the model (Matteucci et al. poster: session-start thirst T0 = scale * (1 - W / W_stable)).
Per NWB in NWB_combined: subject.weight (session-start weight, g) and experiment_description['reference_weight'] (g; candidate W_stable);
relative weight = weight / reference_weight. Flags: no_weight (missing, empty, non-numeric or <= 0), no_reference, implausible (relative
weight outside [0.6, 1.2]). Only sessions of the model's trial table (001) are counted in the summary; all files are listed.
Output: tables/009_session_weights.csv
Run (haas): python 009_session_weights.py [--n-jobs 16]
"""
from __future__ import annotations

import argparse
import ast
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

import common as C


def num(x):
    try:
        v = float(str(x).strip().lower().removesuffix("g").strip().replace(",", "."))      # e.g. "17.5 g"
        return v if np.isfinite(v) and v > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def read(path):
    from pynwb import NWBHDF5IO
    try:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            n = io.read()
            w_raw = getattr(n.subject, "weight", None) if n.subject is not None else None
            try:
                ed = ast.literal_eval(n.experiment_description) if n.experiment_description else {}
            except (ValueError, SyntaxError):
                ed = {}
            return dict(session_id=Path(path).stem, mouse_id=n.subject.subject_id if n.subject is not None else None,
                        session_description=n.session_description, weight_raw=w_raw, weight=num(w_raw),
                        reference_weight=num(ed.get("reference_weight")) if isinstance(ed, dict) else np.nan, error=None)
    except Exception as e:  # noqa: BLE001
        return dict(session_id=Path(path).stem, error=repr(e)[:200])


def main(a):
    files = sorted(C.NWB_DIR.glob("*.nwb"))
    with mp.get_context("spawn").Pool(a.n_jobs) as pool:
        W = pd.DataFrame(pool.map(read, files))
    W["rel_weight"] = W.weight / W.reference_weight
    W["no_weight"] = W.weight.isna()
    W["no_reference"] = W.reference_weight.isna()
    W["implausible"] = W.rel_weight.notna() & ~W.rel_weight.between(0.6, 1.2)
    used = set(pd.read_csv(C.TRIALS, usecols=["session_id", "bhv_type"]).query("bhv_type != 'free_licking'").session_id)
    W["in_model"] = W.session_id.isin(used)
    W.to_csv(C.TAB / "009_session_weights.csv", index=False)
    m = W[W.in_model]
    print(f"NWB files {len(W)} (read errors {W.error.notna().sum()}); sessions used by the model {len(m)}")
    print(f"  without start weight: {m.no_weight.sum()}  | without reference weight: {m.no_reference.sum()}  | implausible ratio: {m.implausible.sum()}")
    print("  mice with any session lacking a start weight:", m[m.no_weight].groupby("mouse_id").size().to_dict())
    print("  raw values flagged:", m[m.no_weight].weight_raw.astype(str).value_counts().to_dict())
    print("  implausible:", m[m.implausible][["session_id", "weight", "reference_weight", "rel_weight"]].round(3).to_string(index=False))
    print(m.rel_weight.describe().round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=16)
    main(ap.parse_args())
