"""Is an SSL intermediate (cache, dataset, derived table) older than the NWB_ks4 session files it was built from?

NWB_ks4 is the single source of truth (user 2026-11-11); every intermediate built from it should record the NWB files it
saw (`fingerprint()` into its json sidecar) and check them on load (`warn_if_stale()`). Intermediates without a
recorded fingerprint fall back to their own modification time: any session NWB modified after it counts as newer.

Library:
    from ssl_nwb_freshness import fingerprint, warn_if_stale
    meta["nwb_fingerprint"] = fingerprint()                  # at build time, before reading the files
    warn_if_stale(cache_file, label="behaviour table")       # at load time (prints a warning, never raises)

Report over every known intermediate layer (run on haas, or locally with the NAS mounted):
    python scripts/ssl_nwb_freshness.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SESSION_NWB = re.compile(r"^[A-Z]+\d+_\d{8}_\d{6}\.nwb$")  # skips *_backup_<date>.nwb left by reconversions


def session_nwbs():
    return sorted(f for f in (axel_bisi_root() / "NWB_ks4").glob("*_*.nwb") if SESSION_NWB.match(f.name))


def fingerprint(paths=None):
    """{session_id: [size_bytes, mtime_s]} of the session NWB files (default: all of NWB_ks4)."""
    out = {}
    for p in paths if paths is not None else session_nwbs():
        st = Path(p).stat()
        out[Path(p).stem] = [st.st_size, round(st.st_mtime, 1)]
    return out


def compare(stored, current, sessions=None):
    """Changed / new / removed sessions between two fingerprints, optionally restricted to `sessions`."""
    keep = (lambda s: s in sessions) if sessions is not None else (lambda s: True)
    return {"changed": sorted(s for s in stored.keys() & current.keys() if keep(s) and stored[s] != current[s]),
            "new": sorted(s for s in current.keys() - stored.keys() if keep(s)),
            "removed": sorted(s for s in stored.keys() - current.keys() if keep(s))}


def stale_sessions(target, sessions=None, current=None):
    """Sessions whose NWB differs from what `target` was built from. With a json sidecar holding `nwb_fingerprint`:
    changed / new / removed files; otherwise every session NWB modified after `target` (file, or the oldest file in a
    folder) counts as changed."""
    target = Path(target)
    current = fingerprint() if current is None else current
    sidecar = target.with_suffix(".json")
    if target.is_file() and sidecar.exists():
        stored = json.load(open(sidecar)).get("nwb_fingerprint")
        if stored:
            return compare(stored, current, sessions), "fingerprint"
    files = [target] if target.is_file() else [f for f in target.rglob("*") if f.is_file()]
    built = min(f.stat().st_mtime for f in files)
    newer = sorted(s for s, (_, mtime) in current.items() if mtime > built and (sessions is None or s in sessions))
    return {"changed": newer, "new": [], "removed": []}, "mtime"


def warn_if_stale(target, label=None, sessions=None):
    """Print a warning when NWB files changed since `target` was built; returns the stale-session dict."""
    diff, how = stale_sessions(target, sessions)
    n = sum(len(v) for v in diff.values())
    if n:
        parts = [f"{k} {len(v)} ({', '.join(v[:5])}{', ...' if len(v) > 5 else ''})" for k, v in diff.items() if v]
        print(f"WARNING: {label or Path(target).name} is older than NWB_ks4 ({how}): {'; '.join(parts)}. "
              f"Rebuild it before trusting results for these sessions.", file=sys.stderr, flush=True)
    return diff


def known_layers():
    """Intermediate layers built from NWB_ks4 (skills/ssl-load; rebuild template projects/ssl-table-rebuild-20261106)."""
    root = axel_bisi_root()
    res = root / "combined_results_ks4"
    cache = root / "manuscripts" / "Bisi_Hamon" / "Figures" / "_cache"

    def newest(folder, pattern):
        files = sorted(folder.glob(pattern), key=lambda f: f.stat().st_mtime)
        return files[-1] if files else folder / pattern

    return {
        "dataset ssl_ephys": REPO / "reports" / "datasets" / "ssl_ephys" / "1.0.0" / "metadata",
        "dataset ssl_behavior": REPO / "reports" / "datasets" / "ssl_behavior" / "1.0.0" / "metadata",
        "derived tables": REPO / "reports" / "ssl_analysis" / "derived",
        "137 stable units": res / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet",
        "all-days unit cache": res / "rastermap_variants" / "_cache" / "tables_all_days_v2.pkl",
        "ROC stage units": res / "_roc_stage_analysis" / "units.parquet",
        "figure behaviour table": newest(cache, "behaviour_table_v*[0-9].parquet"),
        "figure movement PETH": newest(cache, "movement_peth_day0_v*.parquet"),
        "figure movements": newest(cache, "movements_day0_v*.parquet"),
    }


def main():
    current = fingerprint()
    print(f"NWB_ks4: {len(current)} session files")
    for label, path in known_layers().items():
        if not path.exists():
            print(f"  {label:24s} missing ({path})")
            continue
        diff, how = stale_sessions(path, current=current)
        n = sum(len(v) for v in diff.values())
        status = "up to date" if not n else "STALE: " + "; ".join(
            f"{k} {len(v)} ({', '.join(v[:4])}{', ...' if len(v) > 4 else ''})" for k, v in diff.items() if v)
        print(f"  {label:24s} [{how}] {status}")


if __name__ == "__main__":
    main()
