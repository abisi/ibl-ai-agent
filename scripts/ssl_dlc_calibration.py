"""Pixel -> mm conversion for SSL DLC series (user 2026-10-11: "convert pixels to mm ... setup 1 / setup 2 ... add that as a skill").
Canonical rules: skills/ssl-dlc-units/SKILL.md.

Calibration (px/mm, from grid images), per behaviour setup and camera:
  reference_setup1.xlsx / reference_setup2.xlsx in <Axel_Bisi>/mice_info (side and top columns)
Setup of a session: joint_probe_insertion_info.xlsx (column `setup`, keyed by mouse_name + date) in
  <share_internal>/Axel_Bisi_Share/dataset_info; the NWB files carry no setup field. Setup can change between sessions of one mouse
  (MH025, MH027), so look it up per session, never per mouse.
Camera of a series (from the DLC series names; not verified against the videos): top camera = whisker_* and top_* series; side
camera = everything else (jaw, tongue, nose, pupil, spout, particle).

    from ssl_dlc_calibration import px_per_mm, to_mm
    k = px_per_mm("MH015_20250221_105527", "jaw_distance")      # px per mm for that session's setup and camera
    jaw_mm = to_mm(jaw_px, "MH015_20250221_105527", "jaw_distance")
    pupil_mm2 = to_mm(pupil_px2, sid, "pupil_area", power=2)     # areas scale with the square
"""
from __future__ import annotations

import functools
import re
from pathlib import Path

import pandas as pd

from axel_bisi_paths import axel_bisi_path

_SHARE_INTERNAL = [r"M:\share_internal", "/mnt/share_internal"]
# angles (deg) and likelihoods: no conversion. Stored *_velocity series are per frame (scale with the frame rate): recompute
# d(value)/dt from the data and its timestamps instead of converting them.
UNITLESS = re.compile(r"(_angle|_likelihood)$")


def _first(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"none of {paths} exists")


@functools.lru_cache(maxsize=None)
def calibration() -> dict:
    """{1: {"side": px/mm, "top": px/mm}, 2: {...}}"""
    out = {}
    for s in (1, 2):
        f = axel_bisi_path("mice_info", f"reference_setup{s}.xlsx")
        r = pd.read_excel(f).iloc[0]
        out[s] = {"side": float(r["side (px/mm)"]), "top": float(r["top (px/mm)"])}
    return out


@functools.lru_cache(maxsize=None)
def _setup_table() -> pd.DataFrame:
    f = _first(_SHARE_INTERNAL) / "Axel_Bisi_Share" / "dataset_info" / "joint_probe_insertion_info.xlsx"
    t = pd.read_excel(f, usecols=["mouse_name", "date", "setup"]).dropna(subset=["mouse_name", "setup"])
    # dates are mixed: Excel dates and "dd.mm.yyyy" text, which must be parsed day first (pandas' default month-first parse swaps
    # day and month and loses 18 AB sessions). Checked 2026-10-11: all 107 sessions of
    # ssl-prelick-convergence 113 match their exact date (99 setup 1, 8 setup 2: MH001-MH010, MH025 and MH027 on their recording day)
    t["date"] = [pd.Timestamp(d).strftime("%Y%m%d") if not isinstance(d, str) else pd.to_datetime(d, dayfirst=True).strftime("%Y%m%d")
                 for d in t["date"]]
    t["setup_n"] = t["setup"].astype(str).str.extract(r"setup\s*(\d)")[0].astype(int)
    return t.drop_duplicates(["mouse_name", "date", "setup_n"])


def setup_of(session_id: str) -> int:
    """behaviour setup (1 or 2) of a session id "<mouse>_<YYYYMMDD>_<HHMMSS>"; raises if missing or ambiguous"""
    mouse, date = session_id.split("_")[:2]
    t = _setup_table(); tm = t[t.mouse_name == mouse]
    swapped = date[:4] + date[6:8] + date[4:6]
    for dates in ([date], [swapped]):                        # exact date, then day / month swapped (safety net for mis-entered dates)
        r = tm[tm.date.isin(dates)].setup_n.unique()
        if len(r) == 1:
            return int(r[0])
    r = tm.setup_n.unique()                                  # date not found: the mouse's setup if it never changed
    if len(r) == 1:
        return int(r[0])
    raise KeyError(f"setup of {session_id}: mouse setups {list(r)}, no row for the date in joint_probe_insertion_info.xlsx")


def camera_of(series: str) -> str:
    return "top" if series.startswith(("whisker", "top_")) else "side"


def px_per_mm(session_id: str, series: str) -> float:
    return calibration()[setup_of(session_id)][camera_of(series)]


def to_mm(values, session_id: str, series: str, power: int = 1):
    """pixel values -> mm (power=2 for areas, mm^2); unitless series (angles, likelihoods) are returned unchanged"""
    if UNITLESS.search(series):
        return values
    if series.endswith("_velocity"):
        raise ValueError(f"{series} is per frame: recompute the derivative from the data and timestamps, then convert")
    return values / px_per_mm(session_id, series) ** power
