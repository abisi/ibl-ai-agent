"""03 -- Per sorting: SortingAnalyzer + SpikeInterface quality metrics, Phy export, the user's Bombcell, and the inputs
of the user's own criteria (ephys_utilities presence / coverage ratio, DREDge drift shift test). Pipeline container.

Usage: python 03_postprocess.py --task <k> [--slice]     (k = 1-based index into npw.all_runs)
The analyzer reads the shared drift-corrected input high-passed at 300 Hz (same for every sorter; waveforms / SNR only).
Bombcell: same call and parameter updates as the pipeline's run_py_bombcell.py, on the Phy export of every new
sorting, raw waveforms from the CatGT .bin the pipeline used; the as-run KS4 keeps its existing pipeline Bombcell.
Outputs: postprocess[_slice]/<rec_id>/<run_id>/{analyzer/, phy/, bombcell/, units.csv, post_info.json}
"""
import argparse
import shutil
import sys
import time

import numpy as np
import pandas as pd

import npw

sys.path.insert(0, str(npw.HERE / "vendor"))


def analyzer_metrics(cfg, sorting, rec, folder):
    import spikeinterface.core as si
    import spikeinterface.preprocessing as spre
    a = cfg["analyzer"]
    rec_hp = spre.highpass_filter(rec, freq_min=300.0, dtype="float32")
    from spikeinterface.curation import remove_excess_spikes
    sorting = remove_excess_spikes(sorting, rec_hp)
    an = si.create_sorting_analyzer(sorting, rec_hp, format="binary_folder", folder=folder / "analyzer",
                                    overwrite=True, sparse=True, method=a["sparsity"]["method"],
                                    radius_um=a["sparsity"]["radius_um"], **cfg["job_kwargs"])
    an.compute("random_spikes", **a["random_spikes"])
    an.compute(["noise_levels"])
    an.compute("waveforms", **a["waveforms"], **cfg["job_kwargs"])
    an.compute("templates")
    an.compute("spike_amplitudes", **cfg["job_kwargs"])
    an.compute("spike_locations", **cfg["job_kwargs"])
    an.compute("unit_locations")
    an.compute("quality_metrics", metric_names=a["quality_metrics"], **cfg["job_kwargs"])
    qm = an.get_extension("quality_metrics").get_data()
    peak = si.get_template_extremum_channel(an, outputs="index")
    loc = an.get_extension("unit_locations").get_data()
    units = qm.copy()
    units.index.name = "unit_id"
    units["peak_channel"] = [peak[u] for u in units.index]
    units["depth_um"] = loc[:, 1]
    return an, units.reset_index()


def run_bombcell(cfg, an, rec_id, folder, slice_test):
    """Phy export + Bombcell exactly as the pipeline calls it; returns per-unit Bombcell metrics and unitType."""
    import bombcell as bc
    from spikeinterface.exporters import export_to_phy
    phy = folder / "phy"
    export_to_phy(an, phy, compute_pc_features=True, compute_amplitudes=True, copy_binary=False,
                  remove_if_exists=True, verbose=False, **cfg["job_kwargs"])
    # Bombcell unwhitens the templates with whitening_mat_inv.npy; SI-exported templates are already unwhitened
    eye = np.eye(an.get_num_channels(), dtype=np.float32)
    np.save(phy / "whitening_mat.npy", eye)
    np.save(phy / "whitening_mat_inv.npy", eye)
    # Bombcell reads templates.npy as dense (units x samples x channels), as Kilosort writes it; SI writes sparse
    # templates plus template_ind.npy (-1 padded), so densify (pc_features stay sparse, as in Kilosort)
    tpl, ind = np.load(phy / "templates.npy"), np.load(phy / "template_ind.npy")
    dense = np.zeros(tpl.shape[:2] + (an.get_num_channels(),), tpl.dtype)
    for u in range(tpl.shape[0]):
        ok = ind[u] >= 0
        dense[u][:, ind[u][ok]] = tpl[u][:, ok]
    np.save(phy / "templates.npy", dense)
    (phy / "template_ind.npy").unlink()   # dense templates: Phy must not apply the sparse index
    b = cfg["bombcell"]
    raw = npw.raw_bin(cfg, rec_id)
    # slice test: spike times are relative to the slice, so no raw waveforms from the full-length .bin
    kw = {} if slice_test else dict(raw_file=str(raw), meta_file=str(raw.with_suffix(".meta")))
    param = bc.get_default_parameters(str(phy), kilosort_version=b["kilosort_version"], gain_to_uV=b["gain_to_uV"],
                                      **kw)
    param.update(b["param_updates"])
    param["probeType"] = 1 if "imDatPrb_type=0" in raw.with_suffix(".meta").read_text() else 2
    if slice_test:
        param["extractRaw"] = False
    qm, param, unit_type, unit_type_string = bc.run_bombcell(str(phy), str(folder / "bombcell"), param)
    qm = pd.DataFrame(qm)
    out = qm.add_prefix("bc_")
    out["bc_unitType"] = np.asarray(unit_type_string)
    # phy cluster ids follow the analyzer's unit order
    out["unit_id"] = an.unit_ids[np.asarray(qm["phy_clusterID"], int)] if "phy_clusterID" in qm else an.unit_ids
    return out


def asrun_bombcell(cfg, rec_id, sorting):
    """Existing pipeline Bombcell of the as-run KS4 output (cluster_bc_unitType.tsv + qMetrics)."""
    d = npw.probe_dir(cfg, rec_id) / cfg["asrun"]["folder"]
    lab = pd.read_csv(d / "cluster_bc_unitType.tsv", sep="\t")
    qm = pd.read_parquet(d / "bombcell" / "templates._bc_qMetrics.parquet") \
        if (d / "bombcell" / "templates._bc_qMetrics.parquet").exists() \
        else pd.read_csv(d / "bombcell" / "_bc_parameters._bc_qMetrics.csv")
    out = qm.add_prefix("bc_")
    out["unit_id"] = qm["phy_clusterID"].to_numpy() if "phy_clusterID" in qm else qm.index.to_numpy()
    out = out.merge(lab.rename(columns={"cluster_id": "unit_id", "bc_unitType": "bc_unitType"}), on="unit_id",
                    how="left")
    return out


def own_criteria_inputs(cfg, rec_id, sorting, units, slice_test):
    """presence_ratio / coverage_ratio (ephys_utilities, 60-s bins, per session) and the DREDge drift shift test
    (Harris 2021: 1-s rate vs displacement at the unit depth, shifts -N..N, first-last whisker stimulus)."""
    import unit_metrics_utils as umu
    import shift_test as stt
    from joblib import Parallel, delayed
    fs = sorting.sampling_frequency
    t_off = cfg["input"]["slice_test"]["start_s"] if slice_test else 0.0
    spikes = {u: sorting.get_unit_spike_train(u) / fs + t_off for u in units.unit_id}
    df = pd.DataFrame({"unit_id": list(spikes), "spike_times": list(spikes.values()), "session_id": rec_id})
    df["presence_ratio_eu"] = umu.compute_presence_ratio(df).to_numpy()
    df["coverage_ratio"] = umu.compute_coverage_ratio(df).to_numpy()
    r = cfg["recordings"][rec_id]
    stim = np.loadtxt(npw.probe_dir(cfg, rec_id) / f"whisker_stim_times_to_imec{r['imec']}.txt")
    t0, t1 = float(np.min(stim)), float(np.max(stim))
    if slice_test:
        t0, t1 = t_off, t_off + cfg["input"]["slice_test"]["duration_s"]
    motion = stt.load_motion(npw.input_dir(cfg, rec_id) / "dredge_motion")
    dt = cfg["curation"]["axel"]["drift_test"]
    n = dt["N"] if not slice_test else 50   # 5-min slice: N = 500 needs > 1000 one-second bins
    depth = dict(zip(units.unit_id, units.depth_um))
    res = Parallel(n_jobs=cfg["job_kwargs"]["n_jobs"])(
        delayed(stt.time_binned_drift_test)(spikes[u], motion, depth[u], t0, t1, dt["bin_s"], n,
                                            {"drift_test_corr_method": dt["method"]}) for u in df.unit_id)
    df["drift_r"] = [x["r"] for x in res]
    df["drift_abs_r"] = df.drift_r.abs()
    df["drift_shift_test_pval"] = [x["p_conservative"] for x in res]
    df["drift_test_error"] = [x["error"] for x in res]
    return df.drop(columns=["spike_times", "session_id"]), (t0, t1, n)


def main():
    import spikeinterface.core as si
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--slice", action="store_true")
    a = ap.parse_args()
    cfg = npw.load_config()
    si.set_global_job_kwargs(**cfg["job_kwargs"])
    rec_id, run_id, sorter = npw.all_runs(cfg, a.slice)[a.task - 1]
    if a.slice and run_id == cfg["asrun"]["run_id"]:
        print("as-run sorting has no slice version")
        return
    folder = npw.post_dir(cfg, rec_id, run_id, a.slice)
    if (folder / "units.csv").exists():
        print("done already:", folder)
        return
    t = time.time()
    sorting = npw.load_sorting(cfg, rec_id, run_id, a.slice)
    rec = npw.load_input(cfg, rec_id, a.slice)
    an, units = analyzer_metrics(cfg, sorting, rec, folder)
    if run_id == cfg["asrun"]["run_id"]:
        bc_units = asrun_bombcell(cfg, rec_id, sorting)
    else:
        bc_units = run_bombcell(cfg, an, rec_id, folder, a.slice)
    units = units.merge(bc_units, on="unit_id", how="left")
    own, window = own_criteria_inputs(cfg, rec_id, sorting, units, a.slice)
    units = units.merge(own, on="unit_id", how="left")
    fine, coarse = npw.region_of_channels(cfg, rec_id)
    units["region"] = units.peak_channel.map(fine)
    units["area"] = units.peak_channel.map(coarse)
    units.insert(0, "run_id", run_id)
    units.insert(0, "sorter", sorter)
    units.insert(0, "rec_id", rec_id)
    units.to_csv(folder / "units.csv", index=False)
    shutil.rmtree(folder / "phy" / "recording.dat", ignore_errors=True)
    npw.write_json(folder / "post_info.json", npw.versions({
        "rec_id": rec_id, "run_id": run_id, "slice": a.slice, "n_units": len(units), "runtime_s": time.time() - t,
        "analyzer_input": "shared input, highpass 300 Hz", "drift_test_window_s": window, "config": cfg["_path"]}))
    print(f"{rec_id} {run_id}: {len(units)} units, {(units.bc_unitType == 'GOOD').sum()} Bombcell GOOD")


if __name__ == "__main__":
    main()
