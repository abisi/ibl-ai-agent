"""Convert every single-mouse ROC result to firing RATES (roc_utils_new now computes rates natively, ROC_USES_RATES).

For every <mouse>/whisker_<day>/roc_analysis/<mouse>_roc_results_new.csv (day 0 first, expert days last):
  1. recompute the analysis types whose two classes use windows of DIFFERENT lengths (count vs rate changes their AUC;
     025_roc_rates_affected_only.AFFECTED) with the rate-based roc_utils_new;
  2. copy all other types (same-window comparisons: identical AUC with rates);
  3. recompute the per-neuron labels (compute_neuron_labels) on the merged table;
  4. back up the count-based file to roc_analysis/counts_backup_20260928/ and write the rate-based table in its place.
Sessions already produced by 024/025 in roc_analysis_rates/ are installed directly.
A per-session roc_rates_note.json records what was recomputed vs copied.
"""
import argparse
import glob
import json
import multiprocessing
import pathlib
import shutil
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path.home() / "code/unit_spikes_analysis"))
from roc_analysis import roc_utils_new as ru                             # noqa: E402

assert getattr(ru, "ROC_USES_RATES", False), "roc_utils_new must compute rates (ROC_USES_RATES)"
RES = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4")
NWB = pathlib.Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
AFFECTED = ["whisker_passive_pre", "whisker_passive_post", "whisker_active",
            "auditory_passive_pre", "auditory_passive_post", "auditory_active",
            "whisker_hit_vs_spontaneous", "auditory_hit_vs_spontaneous", "spontaneous_licks_vs_cr"]
BACKUP = "counts_backup_20260928"


def all_sessions():
    rows = []
    for f in glob.glob(str(RES / "*" / "whisker_*" / "roc_analysis" / "*_roc_results_new.csv")):
        f = pathlib.Path(f)
        day = int(f.parent.parent.name.split("_")[1]) if f.parent.parent.name.split("_")[1].lstrip("-+").isdigit() else 99
        sid = pd.read_csv(f, usecols=["session_id"], nrows=1).session_id.iloc[0]
        rows.append(dict(file=f, session_id=sid, mouse=f.parent.parent.parent.name, day=day))
    df = pd.DataFrame(rows)
    return df.sort_values(["day", "mouse"], key=lambda s: s.abs() if s.name == "day" else s)   # day 0 first


def recompute(src, sid, n_workers, fig_dir):
    old = pd.read_csv(src)
    todo = [t for t in AFFECTED if t in set(old.analysis_type)]
    put = ru.extract_spike_data(str(NWB / f"{sid}.nwb"), n_workers=n_workers)
    nids = put["neuron_id"].unique()
    slices = {n: put[put["neuron_id"] == n] for n in nids}
    tasks = [(n, slices[n], t, str(fig_dir)) for t in todo for n in nids]
    with multiprocessing.Pool(processes=n_workers, maxtasksperchild=ru.N_WORKER_MAX_TASKS) as pool:
        new = pd.DataFrame(pool.map(ru._process_unit_task, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))))
    cols = list(new.columns)
    merged = pd.concat([old[~old.analysis_type.isin(todo)][cols], new[cols]], ignore_index=True)
    merged = merged.merge(ru.compute_neuron_labels(merged), on="neuron_id", how="left")
    return merged, todo, sorted(set(old.analysis_type) - set(todo))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, default=32)
    a = ap.parse_args()
    ss = all_sessions()
    print(f"[026] {len(ss)} ROC result files (days: {ss.day.value_counts().sort_index().to_dict()})", flush=True)
    for i, r in enumerate(ss.itertuples()):
        t0 = time.time()
        src = r.file
        folder = src.parent
        bdir = folder / BACKUP
        note_f = folder / f"{r.mouse}_roc_rates_note.json"
        if note_f.exists():
            print(f"[026] {i + 1}/{len(ss)} {r.session_id}: already rate-based, skip", flush=True)
            continue
        try:
            staged = folder.parent / "roc_analysis_rates" / src.name
            if r.day == 0 and staged.exists():
                merged, todo, copied = pd.read_csv(staged), None, None
                how = "installed from roc_analysis_rates/"
            else:
                merged, todo, copied = recompute(src, r.session_id, a.n_workers, folder)
                how = f"{len(todo)} types recomputed"
            bdir.mkdir(exist_ok=True)
            shutil.copyfile(src, bdir / src.name)
            cfg = folder / f"{r.mouse}_roc_config.json"
            if cfg.exists():
                shutil.copyfile(cfg, bdir / cfg.name)
            tmp = src.with_suffix(".tmp.csv")
            merged.to_csv(tmp, index=False)
            tmp.replace(src)
            json.dump(dict(session_id=r.session_id, measure="rate (spikes/s) = baseline-corrected count / window length",
                           roc_utils_new="ROC_USES_RATES=True (2026-09-28)", how=how, recomputed_types=todo,
                           copied_types=copied, count_based_backup=str(bdir / src.name)),
                      open(note_f, "w"), indent=2)
            msg = how
        except Exception as e:
            msg = f"FAILED {type(e).__name__}: {e}"
        print(f"[026] {i + 1}/{len(ss)} {r.session_id} (day {r.day}): {msg} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print("ALL DONE", flush=True)
