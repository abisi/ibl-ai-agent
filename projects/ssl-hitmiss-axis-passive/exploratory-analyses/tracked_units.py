"""Re-export of tracked_units.py (shared module; the file lives in ssl-whisker-hitmiss-timeresolved-decoding, kept in one place so the two
projects cannot drift). Created by the 2026-10-15 split of ssl-hitmiss-axis-passive from ssl-whisker-hitmiss-timeresolved-decoding."""
import importlib.util as _u
from pathlib import Path as _P

_SRC = _P(__file__).resolve().parents[2] / "ssl-whisker-hitmiss-timeresolved-decoding" / "exploratory-analyses" / "tracked_units.py"
_spec = _u.spec_from_file_location("_shared_tracked_units", _SRC)
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({k: v for k, v in vars(_mod).items() if not k.startswith("__")})
