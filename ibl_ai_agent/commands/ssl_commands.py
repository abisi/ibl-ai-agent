from __future__ import annotations

from pathlib import Path

import typer

from ibl_ai_agent.commands.common import fail


DEFAULT_NWB_ROOT = Path(r"M:\analysis\Axel_Bisi\NWB_ks4")
DEFAULT_NWB_KS2_ROOT = Path(r"M:\analysis\Axel_Bisi\NWB_combined")


def register(app: typer.Typer) -> None:
    @app.command("build-ssl-ephys-dataset")
    def build_ssl_ephys_dataset_command(
        nwb_root: Path = typer.Option(DEFAULT_NWB_ROOT, help="Directory of source NWB files."),
        output_root: Path = typer.Option(Path("reports/datasets"), help="Root directory for the versioned dataset."),
        spike_time_quantization_us: int = typer.Option(100, min=1, help="Spike time quantization in microseconds."),
        limit_sessions: int | None = typer.Option(None, min=1, help="Build only the first N sessions, for a smoke run."),
    ) -> None:
        """Build the ssl_ephys dataset (units, trials, epochs, spike shards) from local NWB files."""
        try:
            from ibl_ai_agent.datasets.ssl_ephys import BuildConfig, build_ssl_ephys_dataset

            outputs = build_ssl_ephys_dataset(
                BuildConfig(
                    nwb_root=nwb_root,
                    output_root=output_root,
                    spike_time_quantization_us=spike_time_quantization_us,
                    limit_sessions=limit_sessions,
                    verbose=True,
                )
            )
        except Exception as exc:
            fail(str(exc))

        typer.echo(f"Dataset directory: {outputs.dataset_dir}")
        typer.echo(f"Sessions table: {outputs.sessions_path}")
        typer.echo(f"Subjects table: {outputs.subjects_path}")
        typer.echo(f"Probes table: {outputs.probes_path}")
        typer.echo(f"Electrodes table: {outputs.electrodes_path}")
        typer.echo(f"Units table: {outputs.units_path}")
        typer.echo(f"Trials table: {outputs.trials_path}")
        typer.echo(f"Events table: {outputs.events_path}")
        typer.echo(f"Epochs table: {outputs.epochs_path}")
        typer.echo(f"Spikes shard directory: {outputs.spikes_store_path}")
        typer.echo(f"Manifest: {outputs.manifest_path}")
        typer.echo(f"Schema: {outputs.schema_path}")
        typer.echo(f"Provenance: {outputs.provenance_path}")
        typer.echo(f"Build report: {outputs.build_report_path}")

    @app.command("build-ssl-ks2-ephys-dataset")
    def build_ssl_ks2_ephys_dataset_command(
        nwb_root: Path = typer.Option(DEFAULT_NWB_KS2_ROOT, help="Directory of source NWB files."),
        output_root: Path = typer.Option(Path("reports/datasets"), help="Root directory for the versioned dataset."),
        spike_time_quantization_us: int = typer.Option(100, min=1, help="Spike time quantization in microseconds."),
        limit_sessions: int | None = typer.Option(None, min=1, help="Build only the first N sessions, for a smoke run."),
    ) -> None:
        """Build the ssl_ks2_ephys dataset (units, trials, epochs, spike shards) from the KS2/combined NWB source."""
        try:
            from ibl_ai_agent.datasets.ssl_ks2_ephys import BuildConfig, build_ssl_ks2_ephys_dataset

            outputs = build_ssl_ks2_ephys_dataset(
                BuildConfig(
                    nwb_root=nwb_root,
                    output_root=output_root,
                    spike_time_quantization_us=spike_time_quantization_us,
                    limit_sessions=limit_sessions,
                    verbose=True,
                )
            )
        except Exception as exc:
            fail(str(exc))

        typer.echo(f"Dataset directory: {outputs.dataset_dir}")
        typer.echo(f"Sessions table: {outputs.sessions_path}")
        typer.echo(f"Subjects table: {outputs.subjects_path}")
        typer.echo(f"Probes table: {outputs.probes_path}")
        typer.echo(f"Electrodes table: {outputs.electrodes_path}")
        typer.echo(f"Units table: {outputs.units_path}")
        typer.echo(f"Trials table: {outputs.trials_path}")
        typer.echo(f"Events table: {outputs.events_path}")
        typer.echo(f"Epochs table: {outputs.epochs_path}")
        typer.echo(f"Spikes shard directory: {outputs.spikes_store_path}")
        typer.echo(f"Manifest: {outputs.manifest_path}")
        typer.echo(f"Schema: {outputs.schema_path}")
        typer.echo(f"Provenance: {outputs.provenance_path}")
        typer.echo(f"Build report: {outputs.build_report_path}")

    @app.command("build-ssl-ks2-behavior-dataset")
    def build_ssl_ks2_behavior_dataset_command(
        nwb_root: Path = typer.Option(DEFAULT_NWB_KS2_ROOT, help="Directory of source NWB files."),
        output_root: Path = typer.Option(Path("reports/datasets"), help="Root directory for the versioned dataset."),
        timestamp_quantization_us: int = typer.Option(100, min=1, help="Keypoint timestamp quantization in microseconds."),
        continuous_precision: float = typer.Option(0.05, help="Fixed-precision quantization step for continuous keypoint signals."),
        limit_sessions: int | None = typer.Option(None, min=1, help="Build only the first N sessions, for a smoke run."),
    ) -> None:
        """Build the ssl_ks2_behavior dataset (compressed keypoint tracking) from the KS2/combined NWB source."""
        try:
            from ibl_ai_agent.datasets.ssl_ks2_behavior import BuildConfig, build_ssl_ks2_behavior_dataset

            outputs = build_ssl_ks2_behavior_dataset(
                BuildConfig(
                    nwb_root=nwb_root,
                    output_root=output_root,
                    timestamp_quantization_us=timestamp_quantization_us,
                    continuous_precision=continuous_precision,
                    limit_sessions=limit_sessions,
                    verbose=True,
                )
            )
        except Exception as exc:
            fail(str(exc))

        typer.echo(f"Dataset directory: {outputs.dataset_dir}")
        typer.echo(f"Sessions table: {outputs.sessions_path}")
        typer.echo(f"Keypoint availability table: {outputs.keypoint_availability_path}")
        typer.echo(f"Tracking shard directory: {outputs.tracking_store_path}")
        typer.echo(f"Manifest: {outputs.manifest_path}")
        typer.echo(f"Schema: {outputs.schema_path}")
        typer.echo(f"Provenance: {outputs.provenance_path}")
        typer.echo(f"Build report: {outputs.build_report_path}")

    @app.command("build-ssl-behavior-dataset")
    def build_ssl_behavior_dataset_command(
        nwb_root: Path = typer.Option(DEFAULT_NWB_ROOT, help="Directory of source NWB files."),
        output_root: Path = typer.Option(Path("reports/datasets"), help="Root directory for the versioned dataset."),
        timestamp_quantization_us: int = typer.Option(100, min=1, help="Keypoint timestamp quantization in microseconds."),
        continuous_precision: float = typer.Option(0.05, help="Fixed-precision quantization step for continuous keypoint signals."),
        limit_sessions: int | None = typer.Option(None, min=1, help="Build only the first N sessions, for a smoke run."),
    ) -> None:
        """Build the ssl_behavior dataset (compressed keypoint tracking) from local NWB files."""
        try:
            from ibl_ai_agent.datasets.ssl_behavior import BuildConfig, build_ssl_behavior_dataset

            outputs = build_ssl_behavior_dataset(
                BuildConfig(
                    nwb_root=nwb_root,
                    output_root=output_root,
                    timestamp_quantization_us=timestamp_quantization_us,
                    continuous_precision=continuous_precision,
                    limit_sessions=limit_sessions,
                    verbose=True,
                )
            )
        except Exception as exc:
            fail(str(exc))

        typer.echo(f"Dataset directory: {outputs.dataset_dir}")
        typer.echo(f"Sessions table: {outputs.sessions_path}")
        typer.echo(f"Keypoint availability table: {outputs.keypoint_availability_path}")
        typer.echo(f"Tracking shard directory: {outputs.tracking_store_path}")
        typer.echo(f"Manifest: {outputs.manifest_path}")
        typer.echo(f"Schema: {outputs.schema_path}")
        typer.echo(f"Provenance: {outputs.provenance_path}")
        typer.echo(f"Build report: {outputs.build_report_path}")
