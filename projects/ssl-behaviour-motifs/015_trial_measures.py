"""015 -- per-trial behaviour measures of the full run (haas): action and state usage, transitions and latencies per
trial and epoch, with and without lick-related frames.

Inputs: cache/full/inputs (014), state sequences synced from Kuma into kuma/full/ (tables/full_selected_models.json):
lightning-action actions (side view, out-of-mouse predictions: each session predicted by the fold model that did not
see it; classes still / whisking / jaw / lick), kpms syllables (side and top) and their meta-states (009 full,
cache/full/009_meta.npz).
Trials: kept active trials (active context, perf != 6, warm-up cut; 002 conventions), all trial types.
Epochs per trial (bm_lib.trial_windows; user 2026-11-20): pre = [start - 2, start) s; resp = [start,
response_window_stop]; post = (response_window_stop, +5 s] (reward consumption; capped at the next quiet window);
iti = (response_window_stop + 5 s, start of the next trial's quiet window) -- absent when shorter than 0.1 s.
(Before 2026-11-20 the ITI was (response_window_stop, min(next start - 2, +5 s)], i.e. today's post epoch.)
Measures per trial x epoch (NaN where the epoch has < 20 frames):
- act_<class>: fraction of frames with lightning-action class <class>;
- meta_<view>_M<k>: fraction of frames in kpms meta-state k (shared states only; flagged states excluded);
- trans_<view>: kpms syllable transitions per second;
- every usage measure also in a no-lick version (suffix _nl): frames > 0.5 s from any piezo lick;
- resp only: lat_<class> (s from stimulus to the first frame of whisking / jaw / lick; NaN if absent).
Sessions: full_sessions.csv at 200 Hz (the 3 sessions with 100 Hz video are left out: kpms / AR-HMM dynamics and
lightning-action velocities were fitted at 200 Hz).
Output: tables/full_trial_measures.parquet (one row per session x trial x epoch), provenance/015.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402

INP = L.CACHE / "full" / "inputs"
KF = L.RES / "kuma" / "full"
LA = ["background", "still", "whisking", "jaw", "lick"]
NOLICK_S = 0.5
MIN_FR = 20


def to_grid(z, t_src, t_dst, tol=0.0075):
    j = np.clip(np.searchsorted(t_src, t_dst), 1, len(t_src) - 1)
    j = np.where(np.abs(t_src[j - 1] - t_dst) <= np.abs(t_src[j] - t_dst), j - 1, j)
    out = np.asarray(z)[j].copy()
    out[np.abs(t_src[j] - t_dst) > tol] = -1
    return out


def one(sid):
    sel = json.load(open(L.TABLES / "full_selected_models.json"))
    sg = np.load(INP / f"{sid}_session.npz", allow_pickle=True)
    t = sg["t"]
    licks = sg["licks"]
    j = np.clip(np.searchsorted(licks, t), 1, max(len(licks) - 1, 1))
    dl = np.minimum(np.abs(t - licks[j - 1]), np.abs(licks[j] - t)) if len(licks) > 1 else np.full(len(t), np.inf)
    nolick = dl > NOLICK_S
    seq = {}
    f = KF / "lightning_action" / "side" / f"{sid}_probs.npz"
    if f.exists():
        d = np.load(f)
        tv = np.load(INP / f"{sid}_side_keypoints.npz")["t"]
        seq["act"] = to_grid(d["pred"][:len(tv)], tv[:len(d["pred"])], t)
    meta = np.load(L.CACHE / "full" / "009_meta.npz") if (L.CACHE / "full" / "009_meta.npz").exists() else None
    for view in L.VIEWS:
        m = sel["kpms"][view]
        f = KF / "kpms" / view / m["norm"] / m["tags"]["0"] / f"{sid}.npz"
        if f.exists():
            tv = np.load(INP / f"{sid}_{view}_keypoints.npz")["t"]
            z = np.load(f)["z"].astype(int)
            seq[f"syl_{view}"] = to_grid(z, tv[:len(z)], t)
        if meta is not None and f"{sid}|kpms|{view}" in meta:
            seq[f"meta_{view}"] = meta[f"{sid}|kpms|{view}"]
    K = int(max(seq[k].max() for k in seq if k.startswith("meta_")) + 1) if any(k.startswith("meta_") for k in seq) else 0
    tr = pd.read_parquet(INP / f"{sid}_trials.parquet")
    k = tr[tr.keep].reset_index(drop=True)
    W = L.trial_windows(sid, k)
    rows = []
    for i, r in k.iterrows():
        w = W.iloc[i]
        win = {ep: (w[f"{ep}_start"], w[f"{ep}_stop"]) for ep in ("pre", "resp", "post", "iti")}
        for ep, (a, b) in win.items():
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
            if i1 - i0 < MIN_FR:
                continue
            sl = slice(i0, i1)
            row = dict(session_id=sid, trial=i, start_time=r.start_time, stim=r.stim, lick_flag=int(r.lick_flag),
                       whisker_trial_index=int(r.whisker_trial_index), epoch=ep, n_frames=i1 - i0)
            nl = nolick[sl]
            for suf, msk in (("", np.ones(i1 - i0, bool)), ("_nl", nl)):
                if "act" in seq:
                    a_ = seq["act"][sl][msk]
                    a_ = a_[a_ >= 0]
                    for c in (1, 2, 3, 4):
                        row[f"act_{LA[c]}{suf}"] = np.mean(a_ == c) if len(a_) >= MIN_FR else np.nan
                for view in L.VIEWS:
                    if f"meta_{view}" in seq:
                        mm = seq[f"meta_{view}"][sl][msk]
                        mm = mm[mm >= 0]
                        for q in range(K):
                            row[f"meta_{view}_M{q}{suf}"] = np.mean(mm == q) if len(mm) >= MIN_FR else np.nan
                    if f"syl_{view}" in seq:
                        zz = seq[f"syl_{view}"][sl]
                        ch = (np.diff(zz) != 0) & (zz[1:] >= 0) & (zz[:-1] >= 0) & msk[1:]
                        row[f"trans_{view}{suf}"] = ch.sum() / (msk.sum() / L.FS_DS) if msk.sum() >= MIN_FR else np.nan
            if ep == "resp" and "act" in seq:
                a_ = seq["act"][sl]
                for c in (2, 3, 4):
                    w = np.flatnonzero(a_ == c)
                    row[f"lat_{LA[c]}"] = (t[i0 + w[0]] - r.start_time) if len(w) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    S = pd.read_csv(L.TABLES / "full_sessions.csv")
    sids = list(S[S.rate_side > 150].session_id)  # 100 Hz videos (3 sessions) left out: models / velocities are 200 Hz
    with ProcessPoolExecutor(12) as ex:
        out = pd.concat([d for d in ex.map(one, sids) if len(d)], ignore_index=True)
    out.to_parquet(L.TABLES / "full_trial_measures.parquet")
    json.dump(dict(script="015_trial_measures.py", n_sessions=int(out.session_id.nunique()), nolick_s=NOLICK_S,
                   min_frames=MIN_FR, measures=[c for c in out.columns if c.split("_")[0] in ("act", "meta", "trans", "lat")]),
              open(L.RES / "provenance" / "015.json", "w"), indent=1)
    print(out.shape, out.session_id.nunique())


if __name__ == "__main__":
    main()
