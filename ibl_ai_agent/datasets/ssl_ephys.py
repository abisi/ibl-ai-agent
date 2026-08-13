"""Build the ssl_ephys dataset: a compressed local dataset from Axel Bisi's
single-session-learning (SSL) whisker/auditory NWB files.

Source is a local directory of NWB files (not ONE/Alyx), one file per session,
named ``<subject>_<YYYYMMDD>_<HHMMSS>.nwb``. Roughly a quarter of sessions
carry ephys (Neuropixels units + electrodes); the rest are behavior/training
sessions with trials only. Mirrors bwm_ephys's on-disk contract (schema.yaml,
provenance.yaml, parquet metadata tables, delta-encoded spike shards) so it can
be read the same way via resolve_dataset_dir/load_spike_shard, but everything
is derived from NWB structures instead of ALF/Alyx.

Epoch structure (passive_pre/active/passive_post) is session-specific: some
subjects (roughly AB116 onward) have all three epochs recorded in
processing/behavior/BehavioralEpochs/interval_series; earlier subjects have
only "active". This is read per session, never assumed fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from tempfile import mkdtemp
import shutil
import traceback
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ibl_ai_agent.datasets import bwm_shared
from ibl_ai_agent.datasets.bwm_ephys import (
    DEFAULT_SPIKE_TIME_ENCODING,
    DEFAULT_SPIKE_TIME_QUANTIZATION_US,
    SpikeShardWriter,
    _encode_spike_times_dataset,
)


DATASET_NAME = "ssl_ephys"
DATASET_VERSION = "1.0.0"
SCHEMA_VERSION = 1
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "zstd"
SIGNAL_CONTAINER_FORMAT = "blosc_file_shards"
SIGNAL_COMPRESSION_VARIANT = "blosc_zstd_shuffle"

SESSION_FILENAME_RE = re.compile(r"^(?P<subject>[A-Za-z]+\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.nwb$")

# Columns dropped from the units metadata table: too large / not needed for
# scalar per-unit analysis. spike_times moves into the spike store; waveform
# mean can be added back as a companion array store later if needed.
UNIT_METADATA_DROP_COLUMNS = ("spike_times", "waveform_mean")

# Source NWB stores almost every units column (QC metrics, CCF coordinates,
# firing_rate, cluster_id, ...) as an HDF5 string dataset, not native numeric
# types (verified against the raw HDF5 dtypes; only id/spike_times/
# spike_times_index/waveform_mean are natively typed). Every other column is
# coerced to numeric except these genuinely categorical/string ones.
UNIT_STRING_COLUMNS = {
    "bc_label", "ks_label", "ccf_acronym", "ccf_name", "ccf_atlas_acronym", "ccf_atlas_name",
    "ccf_parent_acronym", "ccf_parent_name", "ccf_atlas_parent_acronym", "ccf_atlas_parent_name",
    "electrode_group", "group", "session_id", "probe_name",
}


class BuildError(RuntimeError):
    """Raised when the dataset build cannot complete successfully."""


@dataclass(frozen=True)
class BuildConfig:
    nwb_root: Path
    output_root: Path
    spike_time_quantization_us: int = DEFAULT_SPIKE_TIME_QUANTIZATION_US
    limit_sessions: int | None = None
    include_filenames: frozenset[str] | None = None
    verbose: bool = True


@dataclass(frozen=True)
class BuildOutputs:
    dataset_dir: Path
    sessions_path: Path
    subjects_path: Path
    probes_path: Path
    electrodes_path: Path
    units_path: Path
    trials_path: Path
    events_path: Path
    epochs_path: Path
    spikes_store_path: Path
    manifest_path: Path
    schema_path: Path
    provenance_path: Path
    build_report_path: Path


def _list_session_files(
    nwb_root: Path, *, limit_sessions: int | None, include_filenames: frozenset[str] | None = None
) -> list[tuple[str, str, str, Path]]:
    """Return (subject, date, time, path) tuples for files matching the naming convention, sorted."""
    items = []
    for path in sorted(nwb_root.glob("*.nwb")):
        if include_filenames is not None and path.name not in include_filenames:
            continue
        match = SESSION_FILENAME_RE.match(path.name)
        if match is None:
            continue
        items.append((match.group("subject"), match.group("date"), match.group("time"), path))
    items.sort(key=lambda item: (item[0], item[1], item[2]))
    if limit_sessions is not None:
        items = items[:limit_sessions]
    return items


def _extract_epoch_rows(session_id: str, behavioral_epochs: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, series in behavioral_epochs.interval_series.items():
        data = np.asarray(series.data[:])
        timestamps = np.asarray(series.timestamps[:])
        starts = timestamps[data == 1]
        stops = timestamps[data == -1]
        for start, stop in zip(starts, stops):
            rows.append({
                "session_id": session_id,
                "epoch_name": str(name),
                "start_time": float(start),
                "stop_time": float(stop),
            })
    return rows


def _extract_event_rows(session_id: str, behavioral_events: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, ts in behavioral_events.time_series.items():
        if getattr(ts, "timestamps", None) is not None and len(ts.timestamps) > 0:
            times = np.asarray(ts.timestamps[:])
        else:
            times = np.asarray(ts.data[:])
        for time in times:
            rows.append({"session_id": session_id, "event_type": str(name), "time": float(time)})
    return rows


def _read_one_session(path: Path, *, subject: str, date: str, time_str: str) -> dict[str, Any]:
    from pynwb import NWBHDF5IO

    session_id = path.stem
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()

        subject_row = None
        if nwb.subject is not None:
            subject_row = {
                "subject_id": str(nwb.subject.subject_id or subject),
                "sex": nwb.subject.sex,
                "genotype": nwb.subject.genotype,
                "species": nwb.subject.species,
                "strain": nwb.subject.strain,
                "date_of_birth": str(nwb.subject.date_of_birth) if nwb.subject.date_of_birth else None,
                "description": nwb.subject.description,
            }
        subject_id = subject_row["subject_id"] if subject_row is not None else subject

        trials_df = pd.DataFrame()
        if nwb.trials is not None and len(nwb.trials) > 0:
            trials_df = nwb.trials.to_dataframe().reset_index(drop=True)
            trials_df.insert(0, "session_id", session_id)

        epoch_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        beh = nwb.processing.get("behavior") if nwb.processing else None
        if beh is not None:
            if "BehavioralEpochs" in beh.data_interfaces:
                epoch_rows = _extract_epoch_rows(session_id, beh.data_interfaces["BehavioralEpochs"])
            if "BehavioralEvents" in beh.data_interfaces:
                event_rows = _extract_event_rows(session_id, beh.data_interfaces["BehavioralEvents"])

        has_ephys = nwb.units is not None and len(nwb.units) > 0
        probes_df = pd.DataFrame()
        electrodes_df = pd.DataFrame()
        units_df = pd.DataFrame()
        spike_times_list: list[np.ndarray] = []
        cluster_ids: np.ndarray = np.asarray([], dtype=np.int64)

        if has_ephys:
            electrodes_raw = nwb.electrodes.to_dataframe().reset_index()  # index is 'id' -> becomes a column
            electrodes_raw = electrodes_raw.rename(columns={"id": "electrode_id"})
            # group_name is already the clean device/probe name (e.g. 'imec0'); the
            # 'group' column holds the raw ElectrodeGroup object (e.g. name
            # 'imec0_shank0') and can't be parquet-serialized, so it's dropped.
            electrodes_raw["probe_name"] = electrodes_raw["group_name"]
            electrodes_raw = electrodes_raw.drop(columns=["group"], errors="ignore")
            electrodes_raw.insert(0, "session_id", session_id)
            electrodes_df = electrodes_raw

            probes_df = (
                electrodes_df.groupby("probe_name", as_index=False)
                .agg(n_electrodes=("probe_name", "size"))
            )
            probes_df.insert(0, "session_id", session_id)
            probes_df["device_name"] = probes_df["probe_name"]

            units_raw = nwb.units.to_dataframe().reset_index(drop=True)
            spike_times_list = [np.asarray(arr, dtype=np.float64) for arr in units_raw["spike_times"]]
            # source NWB stores cluster_id (and most other units columns) as a string
            # dataset; cast to int64. Kilosort cluster ids are only unique PER PROBE,
            # not per session, so a multi-probe session has real duplicate cluster_id
            # values across probes (verified: e.g. cluster_id 4 exists on both imec0
            # and imec1). Keep the original as raw_cluster_id, and make the
            # session-unique join key (cluster_id) by offsetting per probe.
            units_raw["raw_cluster_id"] = units_raw["cluster_id"].astype(np.int64)
            if "electrode_group" in units_raw.columns:
                # electrode_group is a raw ElectrodeGroup object per row (e.g. name
                # 'imec0_shank0', device.name 'imec0'); extract probe_name from the
                # device, then replace the column with the group's own name string
                # so it stays parquet-serializable instead of being dropped.
                units_raw["probe_name"] = units_raw["electrode_group"].map(lambda g: g.device.name)
                units_raw["electrode_group"] = units_raw["electrode_group"].map(lambda g: g.name)
            else:
                units_raw["probe_name"] = "unknown"
            probe_order = {name: i for i, name in enumerate(sorted(units_raw["probe_name"].unique()))}
            probe_offset = units_raw["probe_name"].map(probe_order) * 1_000_000
            units_raw["cluster_id"] = units_raw["raw_cluster_id"] + probe_offset
            assert units_raw["cluster_id"].is_unique, "cluster_id must be unique within a session after probe offset"
            cluster_ids = units_raw["cluster_id"].to_numpy(dtype=np.int64)
            drop_cols = [c for c in UNIT_METADATA_DROP_COLUMNS if c in units_raw.columns]
            units_raw = units_raw.drop(columns=drop_cols, errors="ignore")
            for col in units_raw.columns:
                if col not in UNIT_STRING_COLUMNS and col not in ("cluster_id", "raw_cluster_id"):
                    units_raw[col] = pd.to_numeric(units_raw[col], errors="coerce")
            units_raw.insert(0, "session_id", session_id)
            units_df = units_raw

        n_trials = len(trials_df)
        n_units = len(units_df)
        duration_candidates = [0.0]
        if n_trials:
            duration_candidates.append(float(trials_df["stop_time"].max()))
        if epoch_rows:
            duration_candidates.append(max(row["stop_time"] for row in epoch_rows))
        if spike_times_list:
            duration_candidates.append(max(float(arr.max()) for arr in spike_times_list if arr.size))
        duration_s = max(duration_candidates)

        session_row = {
            "session_id": session_id,
            "subject_id": subject_id,
            "date": date,
            "session_start_time": str(nwb.session_start_time),
            "session_description": nwb.session_description,
            "has_ephys": bool(has_ephys),
            "has_passive_epochs": any(row["epoch_name"] != "active" for row in epoch_rows),
            "n_units": int(n_units),
            "n_trials": int(n_trials),
            "n_probes": int(len(probes_df)),
            "duration_s": float(duration_s),
            "source_filename": path.name,
        }

    return {
        "session_row": session_row,
        "subject_row": subject_row,
        "trials_df": trials_df,
        "epoch_rows": epoch_rows,
        "event_rows": event_rows,
        "probes_df": probes_df,
        "electrodes_df": electrodes_df,
        "units_df": units_df,
        "spike_times_list": spike_times_list,
        "cluster_ids": cluster_ids,
    }


def _write_session_spike_shard(
    writer: SpikeShardWriter,
    *,
    session_id: str,
    spike_times_list: list[np.ndarray],
    cluster_ids: np.ndarray,
    quantization_us: int,
) -> dict[str, Any]:
    if not spike_times_list or cluster_ids.size == 0:
        return {"status": "empty", "n_spikes": 0}

    dense_local = np.concatenate([
        np.full(arr.shape[0], i, dtype=np.int32) for i, arr in enumerate(spike_times_list)
    ])
    all_times = np.concatenate(spike_times_list)
    order = np.argsort(all_times, kind="stable")
    all_times = all_times[order]
    dense_local = dense_local[order]
    counts = np.bincount(dense_local, minlength=cluster_ids.size).astype(np.int32)

    encoded_times, attrs = _encode_spike_times_dataset(all_times, quantization_us=quantization_us)
    metadata = {
        "format": "ibl_ai_agent_spike_shard_v2",
        "dataset_name": DATASET_NAME,
        "dataset_version": DATASET_VERSION,
        "session_id": session_id,
        "n_spikes": int(all_times.size),
        "n_units": int(cluster_ids.size),
        "time_encoding": DEFAULT_SPIKE_TIME_ENCODING,
        "time_quantization_us": int(quantization_us),
        "cluster_encoding": "dense_local_indices",
        "compression": {"name": SIGNAL_COMPRESSION_VARIANT},
        **attrs,
    }
    arrays = {
        "spike_times_delta_ticks": encoded_times,
        "spike_clusters": dense_local,
        "cluster_ids": cluster_ids.astype(np.int32, copy=False),
        "cluster_spike_counts": counts,
    }
    shard_bytes = writer.write(session_id, metadata=metadata, arrays=arrays)
    return {"status": "ok", "n_spikes": int(all_times.size), "shard_bytes": int(shard_bytes)}


def load_spike_shard(path: Path) -> dict[str, Any]:
    """Decode a ssl_ephys spike shard, same contract as bwm_ephys.load_spike_shard."""
    shard = bwm_shared.read_array_directory(path)
    arrays = shard["arrays"]
    meta = shard["meta"]
    ticks = np.cumsum(arrays["spike_times_delta_ticks"].astype(np.int64), dtype=np.int64)
    ticks = ticks + int(meta.get("time_origin_ticks", 0))
    arrays["spike_times_seconds"] = ticks * int(meta["time_quantization_us"]) / 1_000_000.0
    return {"meta": meta, **arrays}


def build_ssl_ephys_dataset(config: BuildConfig) -> BuildOutputs:
    target_dir = config.output_root / DATASET_NAME / DATASET_VERSION
    if target_dir.exists():
        raise BuildError(f"Output directory already exists: {target_dir}. Remove it or use a different output root.")

    sessions = _list_session_files(
        config.nwb_root, limit_sessions=config.limit_sessions, include_filenames=config.include_filenames
    )
    if not sessions:
        raise BuildError(f"No NWB files matching the naming convention found under {config.nwb_root}")

    tmp_parent = target_dir.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(mkdtemp(prefix=f".{DATASET_NAME}-{DATASET_VERSION}-", dir=tmp_parent))
    spikes_dir = tmp_dir / "spikes"
    writer = SpikeShardWriter(spikes_dir)

    session_rows: list[dict[str, Any]] = []
    subject_rows: dict[str, dict[str, Any]] = {}
    trials_frames: list[pd.DataFrame] = []
    epoch_rows_all: list[dict[str, Any]] = []
    event_rows_all: list[dict[str, Any]] = []
    probes_frames: list[pd.DataFrame] = []
    electrodes_frames: list[pd.DataFrame] = []
    units_frames: list[pd.DataFrame] = []
    spike_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    try:
        for index, (subject, date, time_str, path) in enumerate(sessions, start=1):
            if config.verbose:
                print(f"[{index}/{len(sessions)}] {path.name}")
            try:
                result = _read_one_session(path, subject=subject, date=date, time_str=time_str)
            except Exception as exc:
                failures.append({"file": path.name, "error": str(exc), "traceback": traceback.format_exc()})
                if config.verbose:
                    print(f"  FAILED: {exc}")
                continue

            session_rows.append(result["session_row"])
            if result["subject_row"] is not None:
                subject_rows.setdefault(result["subject_row"]["subject_id"], result["subject_row"])
            if len(result["trials_df"]):
                trials_frames.append(result["trials_df"])
            epoch_rows_all.extend(result["epoch_rows"])
            event_rows_all.extend(result["event_rows"])
            if len(result["probes_df"]):
                probes_frames.append(result["probes_df"])
            if len(result["electrodes_df"]):
                electrodes_frames.append(result["electrodes_df"])
            if len(result["units_df"]):
                units_frames.append(result["units_df"])

            if result["session_row"]["has_ephys"]:
                spike_result = _write_session_spike_shard(
                    writer,
                    session_id=result["session_row"]["session_id"],
                    spike_times_list=result["spike_times_list"],
                    cluster_ids=result["cluster_ids"],
                    quantization_us=config.spike_time_quantization_us,
                )
                spike_result["session_id"] = result["session_row"]["session_id"]
                spike_results.append(spike_result)

        sessions_df = pd.DataFrame(session_rows)
        subjects_df = pd.DataFrame(list(subject_rows.values()))
        trials_df = pd.concat(trials_frames, ignore_index=True, sort=False) if trials_frames else pd.DataFrame()
        epochs_df = pd.DataFrame(epoch_rows_all)
        events_df = pd.DataFrame(event_rows_all)
        probes_df = pd.concat(probes_frames, ignore_index=True, sort=False) if probes_frames else pd.DataFrame()
        electrodes_df = pd.concat(electrodes_frames, ignore_index=True, sort=False) if electrodes_frames else pd.DataFrame()
        units_df = pd.concat(units_frames, ignore_index=True, sort=False) if units_frames else pd.DataFrame()

        metadata_dir = tmp_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        sessions_path = metadata_dir / "sessions.parquet"
        subjects_path = metadata_dir / "subjects.parquet"
        probes_path = metadata_dir / "probes.parquet"
        electrodes_path = metadata_dir / "electrodes.parquet"
        units_path = metadata_dir / "units.parquet"
        trials_path = metadata_dir / "trials.parquet"
        events_path = metadata_dir / "events.parquet"
        epochs_path = metadata_dir / "epochs.parquet"

        sessions_df.to_parquet(sessions_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        subjects_df.to_parquet(subjects_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        probes_df.to_parquet(probes_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        electrodes_df.to_parquet(electrodes_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        units_df.to_parquet(units_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        trials_df.to_parquet(trials_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        events_df.to_parquet(events_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        epochs_df.to_parquet(epochs_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)

        schema = {
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "tables": {
                "sessions": {"path": "metadata/sessions.parquet", "primary_key": ["session_id"]},
                "subjects": {"path": "metadata/subjects.parquet", "primary_key": ["subject_id"]},
                "probes": {"path": "metadata/probes.parquet", "primary_key": ["session_id", "probe_name"]},
                "electrodes": {"path": "metadata/electrodes.parquet", "primary_key": ["session_id", "electrode_id"]},
                "units": {"path": "metadata/units.parquet", "primary_key": ["session_id", "cluster_id"]},
                "trials": {"path": "metadata/trials.parquet", "primary_key": ["session_id", "trial_id"]},
                "events": {"path": "metadata/events.parquet", "primary_key": []},
                "epochs": {"path": "metadata/epochs.parquet", "primary_key": ["session_id", "epoch_name"]},
            },
            "stores": {
                "spikes": {
                    "path": "spikes",
                    "shard_key": "session_id",
                    "container_format": SIGNAL_CONTAINER_FORMAT,
                    "shard_layout": "<session_id>/meta.json + <session_id>/*.blosc",
                    "arrays": ["spike_times_delta_ticks", "spike_clusters", "cluster_ids", "cluster_spike_counts"],
                },
            },
        }
        schema_path = tmp_dir / "schema.yaml"
        schema_path.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")

        provenance = {
            "created_at": bwm_shared.now_iso(),
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "source": {
                "kind": "local_nwb_files",
                "nwb_root": str(config.nwb_root),
                "n_session_files_found": len(sessions),
                "n_sessions_built": len(session_rows),
                "n_sessions_failed": len(failures),
            },
            "spike_encoding": {
                "cluster_encoding": "dense_local_indices",
                "delta_encoding_explicit": True,
                "overflow_policy": "fallback_to_uint32_if_uint16_limit_exceeded",
                "time_encoding": DEFAULT_SPIKE_TIME_ENCODING,
                "time_quantization_us": config.spike_time_quantization_us,
                "time_storage_dtype": "adaptive_uint16_or_uint32",
            },
            "storage": {
                "included_signal_stores": ["spikes"],
                "metadata_compression": PARQUET_COMPRESSION,
                "metadata_format": "parquet",
                "signal_compression": SIGNAL_COMPRESSION_VARIANT,
                "signal_format": SIGNAL_CONTAINER_FORMAT,
            },
        }
        provenance_path = tmp_dir / "provenance.yaml"
        provenance_path.write_text(yaml.safe_dump(provenance, sort_keys=False), encoding="utf-8")

        build_report = {
            "n_session_files_found": len(sessions),
            "n_sessions_built": len(session_rows),
            "n_sessions_with_ephys": int(sessions_df["has_ephys"].sum()) if len(sessions_df) else 0,
            "n_sessions_with_passive_epochs": int(sessions_df["has_passive_epochs"].sum()) if len(sessions_df) else 0,
            "n_subjects": len(subjects_df),
            "n_units_total": len(units_df),
            "n_trials_total": len(trials_df),
            "spike_shard_results": spike_results,
            "failures": failures,
        }
        build_report_path = tmp_dir / "build_report.yaml"
        build_report_path.write_text(yaml.safe_dump(build_report, sort_keys=False), encoding="utf-8")

        manifest = bwm_shared.build_manifest(dataset_name=DATASET_NAME, dataset_version=DATASET_VERSION, dataset_dir=tmp_dir)
        manifest_path = tmp_dir / "manifest.json"
        import json
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        tmp_dir.rename(target_dir)
        if config.verbose:
            print(f"Done: dataset built at {target_dir} ({len(failures)} session(s) failed)")
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    return BuildOutputs(
        dataset_dir=target_dir,
        sessions_path=target_dir / "metadata" / "sessions.parquet",
        subjects_path=target_dir / "metadata" / "subjects.parquet",
        probes_path=target_dir / "metadata" / "probes.parquet",
        electrodes_path=target_dir / "metadata" / "electrodes.parquet",
        units_path=target_dir / "metadata" / "units.parquet",
        trials_path=target_dir / "metadata" / "trials.parquet",
        events_path=target_dir / "metadata" / "events.parquet",
        epochs_path=target_dir / "metadata" / "epochs.parquet",
        spikes_store_path=target_dir / "spikes",
        manifest_path=target_dir / "manifest.json",
        schema_path=target_dir / "schema.yaml",
        provenance_path=target_dir / "provenance.yaml",
        build_report_path=target_dir / "build_report.yaml",
    )
