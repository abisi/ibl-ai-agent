"""02 -- Run one sorting (one row of the run table) on the shared drift-corrected input.

Usage: python 02_sort.py --env <container|npw-host|npw-ibl|npw-dartsort> --task <k> [--slice]
  <k> = 1-based index among the run-table rows of that env (SLURM_ARRAY_TASK_ID). --slice: 5-min test slice.
Outputs: sortings[_slice]/<rec_id>/<run_id>/{sorter_output/, sorting/ (NumpyFolder), run_info.json}
Every sorter gets the same input; nothing re-applies phase shift, CAR or motion correction (see config.yaml).
"""
import argparse
import shutil
import time
from pathlib import Path

import numpy as np

import npw


def sorter_preprocessing(rec, sc):
    """Only what the sorter needs and its own preprocessing would otherwise do with extra steps (logged)."""
    import spikeinterface.preprocessing as spre
    steps = []
    for name, kw in (sc.get("si_preprocessing") or {}).items():
        rec = getattr(spre, name)(rec, **kw)
        steps.append({name: kw})
    return rec, steps


def run_si_sorter(cfg, row, rec, folder, image=None):
    import spikeinterface.sorters as ss
    params = dict(row["params"])
    if row["sorter"] == "spykingcircus2":
        params["job_kwargs"] = {"n_jobs": cfg["job_kwargs"]["n_jobs"]}
    elif row["sorter"] in ("kilosort4", "kilosort2_5", "mountainsort5"):
        params.update(n_jobs=cfg["job_kwargs"]["n_jobs"], chunk_duration=cfg["job_kwargs"]["chunk_duration"])
    if image:   # SpikeInterface image; SI installed in it from PyPI at the host version (not GitHub main)
        import spikeinterface
        import spikeinterface.extractors as se
        from spikeinterface.sorters.runsorter import run_sorter_container
        ks_out = folder / "sorter_output" / "sorter_output"
        if not (ks_out / "spike_times.npy").exists():
            # with_output=False: the in-container sorting export fails in 0.104.3; the Kilosort output is read below
            run_sorter_container(row["sorter"], rec, mode="singularity", container_image=image,
                                 folder=str(folder / "sorter_output"), remove_existing_folder=True, verbose=True,
                                 with_output=False, installation_mode="pypi",
                                 spikeinterface_version=spikeinterface.__version__, **params)
        return se.read_kilosort(ks_out, keep_good_only=False)
    return ss.run_sorter(row["sorter"], rec, folder=str(folder / "sorter_output"), remove_existing_folder=True,
                         verbose=True, **params)


def run_iblsorter(cfg, row, rec, folder, scratch):
    """IBL sorter reads a SpikeGLX .bin: write the shared input as int16 + a zero sync channel, with the original
    CatGT .meta (only the file size / duration updated). ADC sample shift set to 0 (CatGT already applied tshift),
    destripe spatial filter off (CatGT CAR), drift registration off; its 300 Hz high-pass and whitening kept."""
    import spikeinterface.extractors as se
    from iblsorter.ibl import ibl_pykilosort_params, run_spike_sorting_ibl
    meta_src = npw.raw_bin(cfg, row["rec_id"]).with_suffix(".meta")
    work = Path(scratch) / "bin"
    work.mkdir(parents=True, exist_ok=True)
    stem = meta_src.name.replace(".ap.meta", "")
    bin_file = work / f"{stem}.ap.bin"
    write_spikeglx_int16(rec, bin_file, meta_src, cfg["job_kwargs"])
    params = ibl_pykilosort_params(bin_file)
    p = row["params"]
    params.perform_drift_registration = p["perform_drift_registration"]
    params.destripe_parameters.k_filter = p["destripe_parameters"]["k_filter"]
    # ibldsp destripe phase-shifts with h["sample_shift"] (fourier.fshift): zero it, CatGT applied tshift already
    params.probe["sample_shift"] = np.zeros_like(params.probe["sample_shift"])
    params.probe["h"]["sample_shift"] = np.zeros_like(params.probe["h"]["sample_shift"])
    run_spike_sorting_ibl(bin_file, scratch_dir=Path(scratch) / "iblsorter", params=params,
                          ks_output_dir=folder / "sorter_output", alf_path=None, extract_waveforms=False)
    sorting = se.read_kilosort(folder / "sorter_output", keep_good_only=False)
    shutil.rmtree(work, ignore_errors=True)
    return sorting


def write_spikeglx_int16(rec, bin_file, meta_src, job_kwargs):
    """Data channels rounded to int16 plus a zero sync channel, written block by block (spre.astype(round=True)
    fails on the read-only memmap of the input and spre.zero_channel_pad ignores channel_mapping in 0.104.3)."""
    nc, n = rec.get_num_channels(), rec.get_num_samples()
    dst = np.memmap(bin_file, dtype="int16", mode="w+", shape=(n, nc + 1))
    step = int(rec.sampling_frequency * 30)
    for i in range(0, n, step):
        x = np.asarray(rec.get_traces(start_frame=i, end_frame=min(i + step, n)), dtype=np.float32)
        dst[i:i + len(x), :nc] = np.clip(np.rint(x), -32768, 32767).astype(np.int16)
        dst[i:i + len(x), nc] = 0
    dst.flush()
    del dst
    lines = []
    for ln in open(meta_src).read().splitlines():
        if ln.startswith("fileSizeBytes="):
            ln = f"fileSizeBytes={n * (nc + 1) * 2}"
        elif ln.startswith("fileTimeSecs="):
            ln = f"fileTimeSecs={n / rec.sampling_frequency}"
        lines.append(ln)
    bin_file.with_suffix(".meta").write_text("\n".join(lines) + "\n")


def run_dartsort(cfg, row, rec, folder, scratch):
    import dartsort
    p = row["params"]
    (folder / "sorter_output").mkdir(parents=True, exist_ok=True)
    res = dartsort.dartsort(rec, folder / "sorter_output",
                            cfg=dartsort.DARTsortUserConfig(do_motion_estimation=p["do_motion_estimation"],
                                                            already_preprocessed=p["already_preprocessed"],
                                                            preprocessing=p["preprocessing"],
                                                            work_in_tmpdir=True, tmpdir_parent=str(scratch)))
    return res["sorting"].to_numpy_sorting()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--slice", action="store_true")
    a = ap.parse_args()
    cfg = npw.load_config()
    rows = [r for r in npw.run_table(cfg, a.slice) if r["env"] == a.env]
    row = rows[a.task - 1]
    sc = cfg["sorters"][row["sorter"]]
    folder = npw.sorting_dir(cfg, row["rec_id"], row["run_id"], a.slice)
    if (folder / "sorting").exists():
        print("done already:", folder)
        return
    scratch = Path(cfg["paths"]["scratch"]) / f"{row['rec_id']}_{row['run_id']}{'_slice' if a.slice else ''}"
    scratch.mkdir(parents=True, exist_ok=True)
    print("run", row, flush=True)
    rec = npw.load_input(cfg, row["rec_id"], a.slice)
    rec, steps = sorter_preprocessing(rec, sc)
    t0 = time.time()
    if row["sorter"] == "iblsorter":
        sorting = run_iblsorter(cfg, row, rec, folder, scratch)
    elif row["sorter"] == "dartsort":
        sorting = run_dartsort(cfg, row, rec, folder, scratch)
    else:
        sorting = run_si_sorter(cfg, row, rec, folder, image=sc.get("image"))
    runtime = time.time() - t0
    sorting.save(folder=folder / "sorting", format="numpy_folder", overwrite=True)
    npw.write_json(folder / "run_info.json", npw.versions({
        "row": row, "slice": a.slice, "runtime_s": runtime, "n_units": int(sorting.get_num_units()),
        "input": str(npw.input_dir(cfg, row["rec_id"]) / cfg["input"]["preprocess_subdir"]),
        "motion_handling": sc["motion"], "extra_si_preprocessing": steps, "config": cfg["_path"]}))
    shutil.rmtree(scratch, ignore_errors=True)
    print(f"{row['rec_id']} {row['run_id']}: {sorting.get_num_units()} units in {runtime / 60:.1f} min")


if __name__ == "__main__":
    main()
