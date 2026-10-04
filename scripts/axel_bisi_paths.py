"""Resolve Axel Bisi's external analysis-share root (his `Github`/lab-code
checkouts and raw NWB trees under `.../analysis/Axel_Bisi/...`) across
machines, instead of hardcoding one machine's path (user request
2026-09-15: "it should be a skill to correctly locate the server repos
(analysis/Axel_Bisi) whether on local machine (M:) or Haas
(/mnt/lsens-analysis)").

The same NAS share shows up at a different mount point per machine:
  - This local Windows machine: `M:\\analysis\\Axel_Bisi\\...`
  - haas056.rcp.epfl.ch (and presumably any other machine in the same lab):
    `/mnt/lsens-analysis/Axel_Bisi/...`

Any script that needs something under there (allen_utils, ephys_utilities,
raw NWB files, the mouse-reference xlsx, ...) should resolve the root via
`axel_bisi_root()` below rather than hardcoding `M:\\...`. Override with the
SSL_AXEL_BISI_ROOT env var for a machine this doesn't already know about.
"""

from __future__ import annotations

import os
from pathlib import Path

_KNOWN_ROOTS = [
    r"M:\analysis\Axel_Bisi",
    "/mnt/lsens-analysis/Axel_Bisi",
]


def axel_bisi_root() -> Path:
    """Returns the first existing root, checked in order: SSL_AXEL_BISI_ROOT
    env var, then this machine's `M:` drive, then haas' `/mnt/lsens-analysis`
    mount. Raises FileNotFoundError (not a silent grey/fallback path) if
    none exist -- callers that already have a defined fallback for a
    missing share (e.g. grey area colors when allen_utils isn't reachable)
    should catch this, not `ModuleNotFoundError`, since the failure now
    happens here rather than at the `sys.path.insert` + import step."""
    override = os.environ.get("SSL_AXEL_BISI_ROOT")
    candidates = ([override] if override else []) + _KNOWN_ROOTS
    for candidate in candidates:
        p = Path(candidate)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"None of these Axel_Bisi share roots exist on this machine: {candidates}. "
        "Set SSL_AXEL_BISI_ROOT if this is a new machine not yet in _KNOWN_ROOTS."
    )


def axel_bisi_path(*parts: str) -> Path | None:
    """Convenience: `axel_bisi_root()` joined with `parts`, or None (not a
    raise) if the share isn't mounted at all on this machine -- for the
    common case of a caller that already has a grey/fallback path and just
    wants "give me the path if it's there, else None"."""
    try:
        return axel_bisi_root().joinpath(*parts)
    except FileNotFoundError:
        return None
