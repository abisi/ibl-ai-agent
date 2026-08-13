"""Build the ssl_ks2_behavior dataset: compressed continuous keypoint tracking
(DLC-style face/whisker/top-camera tracking) from Axel Bisi's single-session-
learning (SSL) NWB files, KS2/"combined" source (M:\\analysis\\Axel_Bisi\\NWB_combined).

Identical in mechanism to ssl_behavior.py -- this module only reads
processing/behavior/BehavioralTimeSeries, which was verified to have the same
structure and per-keypoint data/timestamps-length-validity conventions in this
source as in ssl_behavior's NWB_ks4 source. See ssl_ks2_ephys.py for the
session/trial/epoch/spike tables and the schema differences from ssl_ephys
that motivated a separate ephys module; no equivalent differences were found
for behavior tracking.

Timestamp integrity is session-specific and NOT assumed: some sessions have no
BehavioralTimeSeries at all, some have per-keypoint `data`/`timestamps` arrays
of equal length (valid), and some have mismatched or empty `timestamps`
(invalid). Only keypoints with data_len == timestamps_len > 0 are compressed;
everything else is recorded as unavailable in keypoint_availability.parquet.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import traceback
from tempfile import mkdtemp
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ibl_ai_agent.datasets import bwm_shared
from ibl_ai_agent.datasets.bwm_ephys import _encode_spike_times_dataset
from ibl_ai_agent.datasets.ssl_ks2_ephys import SESSION_FILENAME_RE, _list_session_files


DATASET_NAME = "ssl_ks2_behavior"
DATASET_VERSION = "1.0.0"
SCHEMA_VERSION = 1
PARQUET_ENGINE = "pyarrow"
PARQUET_COMPRESSION = "zstd"
SIGNAL_CONTAINER_FORMAT = "blosc_file_shards"
SIGNAL_COMPRESSION_VARIANT = "blosc_zstd_shuffle"

DEFAULT_TIMESTAMP_QUANTIZATION_US = 100
DEFAULT_CONTINUOUS_PRECISION = 0.05  # native units (pixels, degrees, etc.)
LIKELIHOOD_SUFFIX = "_likelihood"


class BuildError(RuntimeError):
    """Raised when the dataset build cannot complete successfully."""


@dataclass(frozen=True)
class BuildConfig:
    nwb_root: Path
    output_root: Path
    timestamp_quantization_us: int = DEFAULT_TIMESTAMP_QUANTIZATION_US
    continuous_precision: float = DEFAULT_CONTINUOUS_PRECISION
    limit_sessions: int | None = None
    include_filenames: frozenset[str] | None = None
    verbose: bool = True


@dataclass(frozen=True)
class BuildOutputs:
    dataset_dir: Path
    sessions_path: Path
    keypoint_availability_path: Path
    tracking_store_path: Path
    manifest_path: Path
    schema_path: Path
    provenance_path: Path
    build_report_path: Path


def _quantize_continuous(values: np.ndarray, *, precision: float) -> tuple[np.ndarray, dict[str, Any]]:
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    quantized = np.rint(safe / precision).astype(np.int32)
    return quantized, {"kind": "continuous", "precision": float(precision), "dtype": "int32", "has_nan_mask": bool((~finite).any())}


def _dequantize_continuous(quantized: np.ndarray, *, precision: float) -> np.ndarray:
    return quantized.astype(np.float64) * precision


def _quantize_likelihood(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    clipped = np.clip(np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0), 0.0, 1.0)
    quantized = np.rint(clipped * 255.0).astype(np.uint8)
    return quantized, {"kind": "likelihood", "dtype": "uint8"}


def _dequantize_likelihood(quantized: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float64) / 255.0


def _group_keypoints_by_timing(time_series: dict[str, Any]) -> dict[tuple[int, int], list[str]]:
    groups: dict[tuple[int, int], list[str]] = {}
    for name, ts in time_series.items():
        key = (int(ts.data.shape[0]), int(ts.timestamps.shape[0]))
        groups.setdefault(key, []).append(name)
    return groups


def _read_one_session(path: Path, *, quantization_us: int, precision: float) -> dict[str, Any]:
    from pynwb import NWBHDF5IO

    session_id = path.stem
    availability_rows: list[dict[str, Any]] = []
    shard_arrays: dict[str, np.ndarray] = {}
    shard_meta: dict[str, Any] = {"timing_groups": {}}

    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        beh = nwb.processing.get("behavior") if nwb.processing else None
        bts = beh.data_interfaces.get("BehavioralTimeSeries") if beh is not None else None
        if bts is None or len(bts.time_series) == 0:
            return {"session_id": session_id, "availability_rows": [], "shard_arrays": {}, "shard_meta": shard_meta, "has_tracking": False}

        groups = _group_keypoints_by_timing(bts.time_series)
        group_index = 0
        for (data_len, ts_len), names in groups.items():
            valid = data_len == ts_len and data_len > 0
            if not valid:
                for name in names:
                    availability_rows.append({
                        "session_id": session_id, "keypoint_name": name, "available": False,
                        "n_frames": data_len, "n_timestamps": ts_len, "timing_group": None,
                    })
                continue

            group_key = f"timing_group_{group_index}"
            timestamps = np.asarray(bts.time_series[names[0]].timestamps[:], dtype=np.float64)
            encoded_ts, ts_attrs = _encode_spike_times_dataset(timestamps, quantization_us=quantization_us)
            shard_arrays[f"{group_key}.timestamps_delta_ticks"] = encoded_ts
            shard_meta["timing_groups"][group_key] = {
                "n_frames": data_len, "keypoints": names, "time_quantization_us": int(quantization_us), **ts_attrs,
            }

            for name in names:
                ts = bts.time_series[name]
                values = np.asarray(ts.data[:])
                if name.endswith(LIKELIHOOD_SUFFIX):
                    quantized, attrs = _quantize_likelihood(values)
                else:
                    quantized, attrs = _quantize_continuous(values, precision=precision)
                shard_arrays[f"{group_key}.{name}"] = quantized
                shard_meta["timing_groups"][group_key].setdefault("array_attrs", {})[name] = attrs
                availability_rows.append({
                    "session_id": session_id, "keypoint_name": name, "available": True,
                    "n_frames": data_len, "n_timestamps": ts_len, "timing_group": group_key,
                })
            group_index += 1

    return {
        "session_id": session_id,
        "availability_rows": availability_rows,
        "shard_arrays": shard_arrays,
        "shard_meta": shard_meta,
        "has_tracking": len(shard_arrays) > 0,
    }


def load_tracking_shard(path: Path) -> dict[str, Any]:
    """Decode a ssl_ks2_behavior tracking shard: reconstruct real timestamps and values."""
    shard = bwm_shared.read_array_directory(path)
    meta = shard["meta"]
    arrays = shard["arrays"]
    decoded: dict[str, Any] = {}
    for group_key, group_meta in meta.get("timing_groups", {}).items():
        ticks = np.cumsum(arrays[f"{group_key}.timestamps_delta_ticks"].astype(np.int64), dtype=np.int64)
        ticks = ticks + int(group_meta.get("time_origin_ticks", 0))
        decoded[f"{group_key}.timestamps_seconds"] = ticks * int(group_meta["time_quantization_us"]) / 1_000_000.0
        for name, attrs in group_meta.get("array_attrs", {}).items():
            raw = arrays[f"{group_key}.{name}"]
            if attrs["kind"] == "likelihood":
                decoded[f"{group_key}.{name}"] = _dequantize_likelihood(raw)
            else:
                decoded[f"{group_key}.{name}"] = _dequantize_continuous(raw, precision=attrs["precision"])
    return {"meta": meta, **decoded}


def build_ssl_ks2_behavior_dataset(config: BuildConfig) -> BuildOutputs:
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
    tracking_dir = tmp_dir / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    session_rows: list[dict[str, Any]] = []
    availability_rows_all: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    try:
        for index, (subject, date, time_str, path) in enumerate(sessions, start=1):
            if config.verbose:
                print(f"[{index}/{len(sessions)}] {path.name}")
            try:
                result = _read_one_session(
                    path, quantization_us=config.timestamp_quantization_us, precision=config.continuous_precision
                )
            except Exception as exc:
                failures.append({"file": path.name, "error": str(exc), "traceback": traceback.format_exc()})
                if config.verbose:
                    print(f"  FAILED: {exc}")
                continue

            session_rows.append({
                "session_id": result["session_id"],
                "subject_id": subject,
                "date": date,
                "has_tracking": result["has_tracking"],
                "n_keypoints_available": sum(1 for r in result["availability_rows"] if r["available"]),
                "n_keypoints_unavailable": sum(1 for r in result["availability_rows"] if not r["available"]),
            })
            availability_rows_all.extend(result["availability_rows"])

            if result["has_tracking"]:
                shard_path = tracking_dir / result["session_id"]
                bwm_shared.write_array_directory(
                    shard_path, metadata=result["shard_meta"], arrays=result["shard_arrays"]
                )

        sessions_df = pd.DataFrame(session_rows)
        availability_df = pd.DataFrame(availability_rows_all)

        metadata_dir = tmp_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        sessions_path = metadata_dir / "sessions.parquet"
        keypoint_availability_path = metadata_dir / "keypoint_availability.parquet"
        sessions_df.to_parquet(sessions_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)
        availability_df.to_parquet(keypoint_availability_path, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION, index=False)

        schema = {
            "dataset_name": DATASET_NAME,
            "dataset_version": DATASET_VERSION,
            "schema_version": SCHEMA_VERSION,
            "tables": {
                "sessions": {"path": "metadata/sessions.parquet", "primary_key": ["session_id"]},
                "keypoint_availability": {
                    "path": "metadata/keypoint_availability.parquet",
                    "primary_key": ["session_id", "keypoint_name"],
                },
            },
            "stores": {
                "tracking": {
                    "path": "tracking",
                    "shard_key": "session_id",
                    "container_format": SIGNAL_CONTAINER_FORMAT,
                    "shard_layout": "<session_id>/meta.json + <session_id>/*.blosc",
                    "notes": "Arrays are namespaced <timing_group>.<name>; see meta.json timing_groups for per-group keypoints and quantization.",
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
            "quantization": {
                "timestamps": {
                    "time_encoding": "delta_int_ticks",
                    "time_quantization_us": config.timestamp_quantization_us,
                },
                "continuous_keypoints": {"kind": "fixed_precision_scaled_int32", "precision": config.continuous_precision},
                "likelihood_keypoints": {"kind": "uint8_linear_0_1"},
            },
            "storage": {
                "included_signal_stores": ["tracking"],
                "metadata_compression": PARQUET_COMPRESSION,
                "metadata_format": "parquet",
                "signal_compression": SIGNAL_COMPRESSION_VARIANT,
                "signal_format": SIGNAL_CONTAINER_FORMAT,
            },
            "timestamp_integrity_note": (
                "Only keypoints whose NWB data/timestamps array lengths matched exactly were "
                "compressed. Keypoints with empty or mismatched-length timestamps are recorded "
                "as available=False in keypoint_availability.parquet and were not compressed, "
                "rather than reconstructed from an assumed frame rate/start time."
            ),
        }
        provenance_path = tmp_dir / "provenance.yaml"
        provenance_path.write_text(yaml.safe_dump(provenance, sort_keys=False), encoding="utf-8")

        build_report = {
            "n_session_files_found": len(sessions),
            "n_sessions_built": len(session_rows),
            "n_sessions_with_tracking": int(sessions_df["has_tracking"].sum()) if len(sessions_df) else 0,
            "n_keypoint_rows_available": int(availability_df["available"].sum()) if len(availability_df) else 0,
            "n_keypoint_rows_unavailable": int((~availability_df["available"]).sum()) if len(availability_df) else 0,
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
        keypoint_availability_path=target_dir / "metadata" / "keypoint_availability.parquet",
        tracking_store_path=target_dir / "tracking",
        manifest_path=target_dir / "manifest.json",
        schema_path=target_dir / "schema.yaml",
        provenance_path=target_dir / "provenance.yaml",
        build_report_path=target_dir / "build_report.yaml",
    )
