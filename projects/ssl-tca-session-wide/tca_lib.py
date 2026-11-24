"""Shared paths, configuration and loaders for ssl-tca-session-wide.

Session-wide TCA: one tensor per session (trials x units x time), all trials of the session in time order.
Units: stable good + mua (unit_info_all_days_v2 + 137_stable_units.stable). Cohort / learner from the reference sheet.
Trials (ssl-trial-exclusion): per-trial context rule; active trials perf != 6 and early_lick == 0, auditory warm-up cut
(keep one trial before the first whisker trial), rule A1 end-of-session disengagement trim; passive trials kept as
passive_pre / passive_post (not filtered on perf). Artefact: spikes in -10..+5 ms of every trial start removed (all
trial types; window from ssl-s1m2-communication 006 artefact check), no Poisson fill.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

ROOT = axel_bisi_root()
NWB_DIR = ROOT / "NWB_ks4"
OUT = ROOT / "combined_results_ks4" / "ssl-tca-session-wide"
TENSORS = OUT / "artifacts" / "tensors"
UNIT_TABLE = ROOT / "combined_results_ks4" / "ssl-roc-learning-stages" / "unit_info_all_days_v2.parquet"
STABLE_TABLE = ROOT / "combined_results_ks4" / "ssl-whisker-hitmiss-timeresolved-decoding" / "tables" / "137_stable_units.parquet"
_SHARE = next((p for p in (Path(r"M:\share_internal"), Path("/mnt/share_internal")) if p.exists()), None)
REFERENCE_SHEET = _SHARE / "Axel_Bisi_Share" / "dataset_info" / "joint_mouse_reference_weight.xlsx" if _SHARE else None
for p in (ROOT / "NWB_reader", ROOT / "Github" / "ephys_utilities", ROOT / "Github" / "allen_utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CONFIG = {
    "version": 1,
    "unit_quality": ["good", "mua"],
    "unit_stability": "137_stable_units.stable (coverage >= 0.9, presence >= 0.5, drift joint test passed)",
    "min_rate_hz": 0.0,
    "bin_s": 0.005,
    "window_s": [-1.0, 1.0],          # around trial start_time (stimulus onset; no-stim trials: nominal onset)
    "artifact_window_s": [-0.010, 0.005],
    "trials": "context per-trial rule; active perf != 6, early_lick == 0, warm-up cut (1 kept), A1 trim; "
              "passive kept (passive_pre / passive_post)",
    "cohort": "reference sheet reward_group (R+ / R-); exclude == 0, exclude_ephys == 0; R+proba dropped",
    "learning_curve": "exact curves h5 (combined_results_ks4/<mouse>/whisker_<day>/learning_curve/*_exact.h5), joined "
                      "on start_time",
}


def reference_sheet():
    ref = pd.read_excel(REFERENCE_SHEET)
    return ref.rename(columns={"mouse_id": "subject_id", "mouse_name": "subject_id"})


def stable_units():
    u = pd.read_parquet(UNIT_TABLE)
    u["cluster_id"] = u.cluster_id.astype(int)
    s = pd.read_parquet(STABLE_TABLE, columns=["session_id", "electrode_group", "nwb_cluster_id", "stable"])
    s = s.rename(columns={"nwb_cluster_id": "cluster_id"})
    u = u.merge(s, on=["session_id", "electrode_group", "cluster_id"], how="left")
    return u[u.quality_label.isin(CONFIG["unit_quality"]) & u.stable.fillna(False).astype(bool)
             & (u.firing_rate >= CONFIG["min_rate_hz"])].copy()


def resolve_context(t):
    from ssl_bwm_trial_prep import resolve_context as rc
    return rc(t)


def select_trials(trials):
    """All trials of the session in time order with an `epoch` column: passive_pre / active / passive_post.
    Active: perf != 6, early_lick == 0, warm-up cut (one trial kept before the first whisker trial), A1 trim."""
    t = trials.sort_values("start_time").reset_index(drop=True).copy()
    ctx = resolve_context(t).to_numpy()
    t["ctx"] = ctx
    act = np.flatnonzero(ctx == "active")
    if len(act) == 0:
        raise RuntimeError("no active trials")
    t["epoch"] = np.where(ctx == "active", "active",
                          np.where(np.arange(len(t)) < act[0], "passive_pre", "passive_post"))
    a = t[t.epoch == "active"]
    a = a[(a.perf != 6) & (a.early_lick == 0)]
    w = np.flatnonzero(a.trial_type.to_numpy() == "whisker_trial")
    n_warm = 0
    if len(w):
        n_warm = max(w[0] - 1, 0)
        a = a.iloc[n_warm:]
    licked = np.flatnonzero(a.lick_flag.to_numpy() == 1)
    n_a1 = 0
    if len(licked):
        trail = a.iloc[licked[-1] + 1:]
        if (trail.trial_type == "whisker_trial").sum() >= 5 and (trail.trial_type == "auditory_trial").sum() >= 1:
            n_a1 = len(trail)
            a = a.iloc[:licked[-1] + 1]
    out = pd.concat([t[t.epoch == "passive_pre"], a, t[t.epoch == "passive_post"]]).sort_values("start_time")
    out = out.reset_index(drop=True)
    out.attrs.update(n_warmup_cut=int(n_warm), n_a1_trimmed=int(n_a1))
    return out


def load_session(session_id, unit_keys):
    """Units (spike times, artefact window removed) for unit_keys (electrode_group, cluster_id, ...) + all raw trials."""
    from ephys_utilities.helpers import data_utils
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = data_utils.process_single_nwb(str(NWB_DIR / f"{session_id}.nwb"), day_to_analyze="all",
                                             apply_artifact_correction=False)
    if res is None:
        raise RuntimeError(f"{session_id}: no ephys")
    units, trials = res["unit_table"], res["trial_table"]
    on = ["electrode_group", "cluster_id"]
    units = units.drop(columns=[x for x in unit_keys.columns if x in units.columns and x not in on])
    units["cluster_id"] = units["cluster_id"].astype(int)
    units["electrode_group"] = units["electrode_group"].apply(lambda x: getattr(x, "name", x)).astype(str)
    uk = unit_keys.assign(cluster_id=unit_keys.cluster_id.astype(int), electrode_group=unit_keys.electrode_group.astype(str))
    units = units.merge(uk, on=on, how="inner")
    if len(units) != len(uk):
        raise RuntimeError(f"{session_id}: {len(units)} of {len(uk)} units matched in the NWB")
    lo, hi = CONFIG["artifact_window_s"]
    st_all = np.sort(trials.start_time.to_numpy(float))

    def blank(sp):
        sp = np.asarray(sp, float)
        i = np.searchsorted(st_all, sp, side="right") - 1
        prev = np.where(i >= 0, st_all[np.clip(i, 0, None)], -np.inf)
        nxt = np.where(i + 1 < len(st_all), st_all[np.clip(i + 1, 0, len(st_all) - 1)], np.inf)
        return sp[~(((sp - prev) < hi) | ((nxt - sp) <= -lo))]

    units["spike_times"] = units.spike_times.apply(blank)
    return units, trials


def bin_tensor(units, t):
    """Spike counts (trials x units x bins), uint8, CONFIG bin / window around start_time."""
    b, (w0, w1) = CONFIG["bin_s"], CONFIG["window_s"]
    nb = int(round((w1 - w0) / b))
    st = t.start_time.to_numpy(float)
    X = np.zeros((len(t), len(units), nb), np.uint16)
    for j, sp in enumerate(units.spike_times):
        sp = np.asarray(sp, float)
        lo = np.searchsorted(sp, st + w0)
        hi = np.searchsorted(sp, st + w1)
        for i in range(len(st)):
            k = np.floor((sp[lo[i]:hi[i]] - st[i] - w0) / b).astype(int)
            k = k[(k >= 0) & (k < nb)]
            if len(k):
                X[i, j] += np.bincount(k, minlength=nb).astype(np.uint16)
    assert X.max() < 256
    return X.astype(np.uint8)


def bin_centres():
    b, (w0, w1) = CONFIG["bin_s"], CONFIG["window_s"]
    return w0 + b * (np.arange(int(round((w1 - w0) / b))) + 0.5)


def learning_curves(mouse, day, t):
    """Join exact learning curves (whisker / auditory / no-stim p_mean, p_low, p_high) to trials on start_time.
    Adds learning_trial (whisker, in curve-trial index) and p_chance columns."""
    out = t.copy()
    meta = {}
    for tt, tag in (("whisker_trial", "wh"), ("auditory_trial", "au"), ("no_stim_trial", "ns")):
        f = ROOT / "combined_results_ks4" / mouse / f"whisker_{day}" / "learning_curve" / \
            f"{mouse}_whisker_{day}_{tt}_learning_curve_exact.h5"
        if not f.exists():
            continue
        c = pd.read_hdf(f).iloc[0]
        cur = pd.DataFrame({"start_time": np.asarray(c.start_time, float), f"p_{tag}": np.asarray(c.p_mean, float),
                            f"p_{tag}_low": np.asarray(c.p_low, float), f"p_{tag}_high": np.asarray(c.p_high, float)})
        out = pd.merge_asof(out.sort_values("start_time"), cur.sort_values("start_time"), on="start_time",
                            direction="nearest", tolerance=1e-3)
        # curve at every trial time (linear interpolation between the curve's own trials; NaN outside its range)
        out[f"p_{tag}_i"] = np.interp(out.start_time, cur.start_time, cur[f"p_{tag}"], left=np.nan, right=np.nan)
        if tt == "whisker_trial":
            meta.update(learning_trial=c.get("learning_trial", np.nan), p_chance=c.get("p_chance", np.nan))
    return out, meta
