"""Shared helpers for the workshop pipeline: config, paths, the shared drift-corrected input, run table, provenance."""
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def load_config(path=None):
    path = Path(path or os.environ.get("NPW_CONFIG", HERE / "config.yaml"))
    cfg = yaml.safe_load(open(path))
    cfg["_path"] = str(path)
    return cfg


def out(cfg, *parts):
    p = Path(cfg["paths"]["out_root"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def probe_dir(cfg, rec_id):
    r = cfg["recordings"][rec_id]
    return Path(cfg["paths"]["data_root"]) / r["mouse"] / r["session"] / "Ephys" / r["catgt"] / r["probe"]


def input_dir(cfg, rec_id):
    """Staged copy of the pipeline's saved drift-corrected recording (01_stage_inputs.sh)."""
    return Path(cfg["paths"]["out_root"]) / "inputs" / rec_id


def raw_bin(cfg, rec_id):
    """CatGT output the pipeline sorted (and Bombcell reads): *_tcat_corrected.ap.bin, else *_tcat.ap.bin."""
    d = probe_dir(cfg, rec_id)
    for pat in ("*_tcat_corrected.imec*.ap.bin", "*_tcat.imec*.ap.bin"):
        f = sorted(d.glob(pat))
        if f:
            return f[0]
    raise FileNotFoundError(d)


def load_input(cfg, rec_id, slice_test=False):
    """Shared input of every sorter: the pipeline's DREDge-interpolated float32 recording (no further correction)."""
    import spikeinterface.core as si
    rec = si.load(input_dir(cfg, rec_id) / cfg["input"]["preprocess_subdir"])
    if slice_test:
        s = cfg["input"]["slice_test"]
        fs = rec.sampling_frequency
        rec = rec.frame_slice(int(s["start_s"] * fs), int((s["start_s"] + s["duration_s"]) * fs))
    return rec


def run_table(cfg, slice_test=False):
    """One row per sorting run: (rec_id, run_id, sorter, env, gpu, params). Slice test: first recording, one run per
    sorter (no KS4 variants)."""
    rows = []
    for rec_id in list(cfg["recordings"])[:1] if slice_test else cfg["recordings"]:
        for sorter, sc in cfg["sorters"].items():
            rows.append(dict(rec_id=rec_id, run_id=sorter, sorter=sorter, env=sc["env"], gpu=sc["gpu"],
                             params=dict(sc["params"])))
        if slice_test:
            continue
        ks = cfg["sorters"]["kilosort4"]
        v = cfg["ks4_variability"]
        variants = []
        if v["repeat_default"]:
            variants.append(("kilosort4_repeat", {}))
        variants += [(f"kilosort4_seed{s}", {"cluster_init_seed": s}) for s in v["seeds"]]
        variants += [(f"kilosort4_Th{t}", {"Th_universal": t}) for t in v["th_universal"]]
        for run_id, over in variants:
            rows.append(dict(rec_id=rec_id, run_id=run_id, sorter="kilosort4", env=ks["env"], gpu=True,
                             params={**ks["params"], **over}))
    return rows


def sorting_dir(cfg, rec_id, run_id, slice_test=False):
    return out(cfg, "sortings_slice" if slice_test else "sortings", rec_id, run_id)


def post_dir(cfg, rec_id, run_id, slice_test=False):
    return out(cfg, "postprocess_slice" if slice_test else "postprocess", rec_id, run_id)


def all_runs(cfg, slice_test=False):
    """Every sorting incl. the existing pipeline output (as run; not in the slice test)."""
    runs = [(r["rec_id"], r["run_id"], r["sorter"]) for r in run_table(cfg, slice_test)]
    if not slice_test:
        runs += [(rec_id, cfg["asrun"]["run_id"], "kilosort4") for rec_id in cfg["recordings"]]
    return runs


def load_sorting(cfg, rec_id, run_id, slice_test=False):
    """Final sorting of a run as a SpikeInterface sorting (02_sort.py saves every sorter as a NumpyFolder).
    Spike frames are input samples, so the sampling rate is set to the input's calibrated rate (iblsorter and the
    Kilosort params.py write the nominal 30 kHz)."""
    import spikeinterface.core as si
    if run_id == cfg["asrun"]["run_id"]:
        import spikeinterface.extractors as se
        s = se.read_kilosort(probe_dir(cfg, rec_id) / cfg["asrun"]["folder"], keep_good_only=False)
    else:
        s = si.load(sorting_dir(cfg, rec_id, run_id, slice_test) / "sorting")
    fs = load_input(cfg, rec_id, slice_test).sampling_frequency
    if s.sampling_frequency != fs:
        s = si.NumpySorting(s.to_spike_vector(), fs, s.unit_ids)
    return s


def versions(extra=None):
    v = {"python": sys.version.split()[0], "host": platform.node(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
         "slurm_job": os.environ.get("SLURM_JOB_ID"), "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID")}
    for mod in ("spikeinterface", "probeinterface", "numpy", "kilosort", "torch", "cupy", "mountainsort5",
                "iblsorter", "dartsort", "bombcell"):
        if mod in sys.modules or _importable(mod):
            try:
                m = __import__(mod)
                v[mod] = getattr(m, "__version__", "?")
            except Exception as e:  # noqa: BLE001
                v[mod] = f"import failed: {type(e).__name__}"
    try:
        import torch
        v["cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    v["code_sha256"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:12] for f in sorted(HERE.glob("*.py"))}
    v.update(extra or {})
    return v


def _importable(mod):
    import importlib.util
    return importlib.util.find_spec(mod) is not None


def write_json(path, d):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2, default=str))


def region_of_channels(cfg, rec_id):
    """Recording channel index -> aligned Allen acronym (IBL alignment, channel_locations.json, whose channel_<i> is
    the i-th channel of the IBL export, raw index channels.rawInd[i]) and its coarse area (layer suffix removed).
    Channels absent from the export (dropped by the original sort) get no region."""
    import re

    import numpy as np
    d = json.load(open(input_dir(cfg, rec_id) / "channel_locations.json"))
    raw = np.load(input_dir(cfg, rec_id) / "channels.rawInd.npy")
    fine = {int(raw[int(k.split("_")[1])]): v["brain_region"] for k, v in d.items() if k.startswith("channel_")}
    coarse = {k: re.sub(r"(?<=[A-Za-z])(1|2/3|4|5|6a|6b)$", "", v) for k, v in fine.items()}
    return fine, coarse


def style(cfg):
    import matplotlib as mpl
    f = cfg["figures"]
    mpl.rcParams.update({"font.size": f["font_size"], "axes.titlesize": f["font_size"] + 2,
                         "axes.labelsize": f["font_size"], "xtick.labelsize": f["font_size"] - 2,
                         "ytick.labelsize": f["font_size"] - 2, "legend.fontsize": f["font_size"] - 4,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})


def save_fig(cfg, fig, name):
    for ext in ("pdf", "png"):
        p = out(cfg, "figures", f"{name}.{ext}")
        fig.savefig(p, dpi=cfg["figures"]["dpi"])
    print("saved", out(cfg, "figures", name))


SORTER_LABEL = {"kilosort4": "Kilosort 4", "kilosort2_5": "Kilosort 2.5", "iblsorter": "IBL sorter",
                "spykingcircus2": "SpyKING Circus 2", "mountainsort5": "MountainSort 5", "dartsort": "DARTsort",
                "kilosort4_asrun": "KS4 as run (2x drift corr.)"}
SCHEME_LABEL = {"ibl": "IBL", "bombcell": "Bombcell", "allen": "Allen", "axel": "Own (ephys_utilities)"}
