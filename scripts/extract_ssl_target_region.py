"""Path-A metadata extraction for the SSL KS4 analysis: per (session_id,
probe_name) target_region, read directly from raw NWB electrode_group.location
per ssl-load/references/ssl_loading_policy.md. Metadata-only (electrode_groups
dict on the NWB file object) -- does not touch spike or behavior data.

electrode_group.location is a stringified Python dict (e.g. "{'hemisphere':
'left', 'area': 'wM1', ...}"), parsed with ast.literal_eval, not json.loads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
from pynwb import NWBHDF5IO

NWB_ROOT = Path(r"M:\analysis\Axel_Bisi\NWB_ks4")
OUT_DIR = Path("reports/ssl_analysis/derived")
OUT_PATH = OUT_DIR / "electrode_group_target_region.parquet"
ERRORS_PATH = OUT_DIR / "target_region_extraction_errors.parquet"


def extract_one(path: Path) -> tuple[list[dict], str | None]:
    session_id = path.stem
    rows: list[dict] = []
    try:
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()
            for group_name, group in nwb.electrode_groups.items():
                loc_raw = group.location
                parsed: dict = {}
                if loc_raw:
                    try:
                        maybe = ast.literal_eval(loc_raw)
                        if isinstance(maybe, dict):
                            parsed = maybe
                    except (ValueError, SyntaxError):
                        parsed = {}
                def _to_float(value: object) -> float | None:
                    try:
                        return float(value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        return None

                rows.append(
                    {
                        "session_id": session_id,
                        "probe_name": group.device.name,
                        "group_name": group_name,
                        "target_region": parsed.get("area"),
                        "hemisphere": parsed.get("hemisphere"),
                        "ap": _to_float(parsed.get("ap")),
                        "ml": _to_float(parsed.get("ml")),
                        "depth_um": _to_float(parsed.get("depth")),
                        "location_raw": loc_raw,
                    }
                )
        return rows, None
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop the other 856
        return [], f"{type(exc).__name__}: {exc}"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(NWB_ROOT.glob("*.nwb"))
    all_rows: list[dict] = []
    errors: list[dict] = []
    n = len(paths)
    for i, path in enumerate(paths, 1):
        rows, error = extract_one(path)
        all_rows.extend(rows)
        if error:
            errors.append({"file": path.name, "error": error})
        if i % 25 == 0 or i == n:
            print(f"[{i}/{n}] {path.name} (rows so far: {len(all_rows)}, errors: {len(errors)})", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Wrote {len(df)} electrode-group rows to {OUT_PATH}")

    if errors:
        errors_df = pd.DataFrame(errors)
        errors_df.to_parquet(ERRORS_PATH, index=False)
        print(f"Wrote {len(errors_df)} extraction errors to {ERRORS_PATH}")
    else:
        print("No extraction errors.")


if __name__ == "__main__":
    main()
