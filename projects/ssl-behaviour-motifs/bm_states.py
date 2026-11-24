"""Loaders for the fitted state sequences (v2, 200 Hz; results synced from Kuma into combined_results_ks4/
ssl-behaviour-motifs/kuma/), mapped onto the session's side 200 Hz grid (cache/inputs/<sid>_session.npz) by nearest
time. The chosen model per method x view (normalisation, kappa, number of states, seeds, held-out fit) is in
tables/selected_models.json (written after the sweeps; see change-log)."""
from __future__ import annotations

import json
import os

import numpy as np

import bm_lib as L

SET = os.environ.get("BM_SET", "pilot")  # pilot (8 sessions) or full (tables/full_sessions.csv)
KR = L.RES / "kuma" / ("" if SET == "pilot" else "full")
INP = L.CACHE / ("inputs" if SET == "pilot" else "full/inputs")
METHODS = ("kpms", "arhmm", "lightning_action")
METHOD_LABEL = {"kpms": "keypoint-MoSeq", "arhmm": "AR-HMM (features)", "lightning_action": "lightning-action"}
LA_CLASSES = ["background", "still", "whisking", "nose", "jaw", "lick"] if SET == "pilot" else ["background", "still", "whisking", "jaw", "lick"]
TOL = 0.0075  # s: max distance to the nearest frame of the other camera (1.5 frames at 200 Hz)


def selected():
    return json.load(open(L.TABLES / ("selected_models.json" if SET == "pilot" else "full_selected_models.json")))


def session_grid(sid):
    return np.load(INP / f"{sid}_session.npz", allow_pickle=True)


def view_t(sid, view):
    return np.load(INP / f"{sid}_{view}_keypoints.npz")["t"]


def to_grid(z, t_src, t_dst, fill=-1):
    j = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[j - 1] - t_dst) <= np.abs(t_src[j] - t_dst), j - 1, j)
    out = np.asarray(z)[j].copy()
    out[np.abs(t_src[j] - t_dst) > TOL] = fill
    return out


def state_file(sid, method, view, seed=0, heldout=False):
    if method == "lightning_action":
        return KR / "lightning_action" / view / f"{sid}_probs.npz"
    m = selected()[method][view]
    tag = m["heldout"] if heldout else m["tags"][str(seed)]
    base = KR / "kpms" / view / m["norm"] if method == "kpms" else KR / "arhmm" / view
    return base / tag / f"{sid}.npz"


def kpms_seeds(view):
    return sorted(int(s) for s in selected()["kpms"][view]["tags"])


def load_states_native(sid, method, view, seed=0, heldout=False):
    f = state_file(sid, method, view, seed, heldout)
    if not f.exists():
        return None
    d = np.load(f)
    return np.asarray(d["z"] if "z" in d else d["pred"]).astype(int)


def load_states(sid, method, view, seed=0, heldout=False):
    """state per side-grid frame (-1 = missing). lightning_action: argmax class of the held-out prediction."""
    z = load_states_native(sid, method, view, seed, heldout)
    if z is None:
        return None
    t = session_grid(sid)["t"]
    tv = view_t(sid, view)
    n = min(len(z), len(tv))
    return to_grid(z[:n], tv[:n], t)


def state_names(method, n):
    if method == "lightning_action":
        return LA_CLASSES
    return [f"{'S' if method == 'kpms' else 'A'}{i}" for i in range(n)]


def kuma_meta(method, view):
    try:
        m = selected()[method][view]
        base = KR / "kpms" / view / m["norm"] if method == "kpms" else KR / "arhmm" / view
        return json.load(open(base / m["tags"]["0"] / "done.json"))
    except (KeyError, FileNotFoundError):
        return {}
