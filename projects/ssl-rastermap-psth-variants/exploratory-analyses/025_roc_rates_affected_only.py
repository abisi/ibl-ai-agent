"""Rate-based ROC, recomputing ONLY the analysis types whose two classes use windows of different lengths, then merging
them into the existing count-based results (identical AUCs for the other types, so they are copied).

Affected (count -> rate changes the AUC):
  pre (35 ms) vs post (30 ms):      whisker/auditory_passive_pre, _passive_post, _active
  post (30 ms) vs spont. post (200 ms): whisker_hit_vs_spontaneous, auditory_hit_vs_spontaneous, spontaneous_licks_vs_cr
Unchanged (same window on both sides): choice types, *_sensory (miss vs CR), *_hit_vs_cr, wh_vs_aud_*,
*_pre_vs_post_learning, baseline_*, spontaneous_licks (200 vs 200 ms).
Per session: roc_utils_new.extract_spike_data -> process_unit for the affected types with rates (wrapper from
024_roc_rates.py, skipped automatically if roc_utils_new already computes rates) -> replace those rows in
<mouse>/whisker_0/roc_analysis/<mouse>_roc_results_new.csv -> recompute the per-neuron labels
(compute_neuron_labels) on the merged table -> <mouse>/whisker_0/roc_analysis_rates/. Sessions already fully rerun by
024 (a results file exists there) are skipped.
"""
import argparse
import importlib
import json
import multiprocessing
import pathlib
import sys
import time

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv, _argv = sys.argv[:1], sys.argv                      # 024 has no CLI on import
d024 = importlib.import_module("024_roc_rates")               # installs the rate wrapper if needed
sys.argv = _argv
ru = d024.ru

AFFECTED = ["whisker_passive_pre", "whisker_passive_post", "whisker_active",
            "auditory_passive_pre", "auditory_passive_post", "auditory_active",
            "whisker_hit_vs_spontaneous", "auditory_hit_vs_spontaneous", "spontaneous_licks_vs_cr"]


def run_session(s, n_workers):
    m = s.split("_")[0]
    src = d024.RES / m / "whisker_0" / "roc_analysis" / f"{m}_roc_results_new.csv"
    out = d024.RES / m / "whisker_0" / d024.SUBDIR
    if (out / f"{m}_roc_results_new.csv").exists():
        return "exists, skip"
    old = pd.read_csv(src)
    todo = [t for t in AFFECTED if t in set(old.analysis_type)]
    put = ru.extract_spike_data(str(d024.NWB / f"{s}.nwb"), n_workers=n_workers)
    nids = put["neuron_id"].unique()
    slices = {n: put[put["neuron_id"] == n] for n in nids}
    tasks = [(n, slices[n], t, str(out)) for t in todo for n in nids]
    with multiprocessing.Pool(processes=n_workers, maxtasksperchild=ru.N_WORKER_MAX_TASKS) as pool:
        new = pd.DataFrame(pool.map(ru._process_unit_task, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))))
    base_cols = list(new.columns)                              # per-analysis columns (labels are recomputed below)
    merged = pd.concat([old[~old.analysis_type.isin(todo)][base_cols], new[base_cols]], ignore_index=True)
    merged = merged.merge(ru.compute_neuron_labels(merged), on="neuron_id", how="left")
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / f"{m}_roc_results_new.csv", index=False)
    json.dump(dict(session_id=s, measure="rate (spikes/s) = baseline-corrected count / window length",
                   recomputed_types=todo, copied_types=sorted(set(old.analysis_type) - set(todo)),
                   source=str(src), note="copied types compare equal-length windows: identical AUC as counts"),
              open(out / f"{m}_roc_rates_note.json", "w"), indent=2)
    return f"{len(nids)} neurons, {len(todo)} types recomputed"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=32)
    a = ap.parse_args()
    ss = d024.sessions()
    print(f"[025] {len(ss)} sessions; recomputing {AFFECTED}", flush=True)
    for i, s in enumerate(ss):
        t0 = time.time()
        try:
            msg = run_session(s, a.n_workers)
        except Exception as e:
            msg = f"FAILED {type(e).__name__}: {e}"
        print(f"[025] {i + 1}/{len(ss)} {s}: {msg} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("ALL DONE", flush=True)
