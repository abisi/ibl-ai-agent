"""Before / after re-inference on the 8 pilot videos (023, 024): what changes in the NWB-derived traces (user 2026-11-10).

"Before" = the series in the session's NWB_ks4 file. "After" = the same series computed from the merged-model h5
(CSPNeXt-M snapshot-330, filtered) exactly as the converter would: dlc2kinematics smoothing (Savitzky-Golay, window 3,
order 1, x / y only), top orientation check (DLC_to_nwb), NWB_converter utils/dlc_utils.compute_kinematics_alt,
velocities x frame rate, written as a minimal NWB-like h5 (BehavioralTimeSeries/<name>/data, timestamps, conversion)
with the timestamps and mm-per-px conversion of the real NWB series. Both are then read with ephys_utilities
dlc_helpers (load_dlc + clean_series, pupil without the pupil QC on both since the QC was built on the old tracks), so
the only difference is the DLC model. Check: the same rebuild from the OLD h5 must reproduce the NWB (printed).
Figure 1 (before_after_frames): per video, frames where before and after differ most (99th percentile of the
difference, or a frame the old model could not track), old keypoints (left) vs new (right), with the derived geometry
(pupil polygon, jaw reference -> jaw, nose vector and whisker_base -> particle).
Figure 2 (before_after_trial_aligned): mean traces aligned to trial start (W hit, A hit, CR), before (dashed) vs
after (solid). Side: jaw distance, jaw angle, pupil area; top: whisker angle, whisker speed, % frames whose whisker_base and
particle pass the dlc_helpers likelihood cutoffs (0.1; clean_series itself does not gate by likelihood).
Figure 3 (before_after_heatmaps): single-trial heatmaps (whisker trials) before vs after, 4 videos.
Figure 4 (jaw_angle_scale): old vs new jaw angle and jaw distance per side video, and the converter references
(jaw 5th percentile, median jaw_ref x) of each model drawn on a frame.
Outputs: DLCProjects/merged_training_2026-11-09/pilot_inference/before_after/
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 025_pilot_before_after.py
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "/mnt/lsens-analysis/Axel_Bisi/Github/ephys_utilities")
from ephys_utilities.helpers import dlc_helpers as dh  # noqa: E402

sys.modules.setdefault("dlc2kinematics", types.ModuleType("dlc2kinematics"))  # only used to load h5s, done here
sys.path.insert(0, str(Path.home() / "code/NWB_converter"))
from utils.dlc_utils import compute_kinematics_alt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl-pupil-qc"))
import pupil_qc as pq  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pv = _load("pv", "024_pilot_vs_old.py")
ts_, ps = pv.ts, pv.ps
OUT = pv.OUT / "before_after"
NWB = Path("/mnt/lsens-analysis/Axel_Bisi/NWB_ks4")
RAW = Path("/mnt/lsens-data")
COLORS = {"old": "#d6604d", "new": ts_.MODEL_COLORS[6]}
OUTCOMES = {"W hit": "#f7b519", "A hit": "#2c2cdb", "CR": "0.4"}
WINDOW = (-1.0, 2.0)
SIDE_Q = {"jaw_distance": "Jaw distance (mm)", "jaw_angle": "Jaw angle (deg)", "pupil_area": "Pupil area (mm$^2$)"}
TOP_Q = {"whisker_angle": "Whisker angle (deg)", "whisker_speed": "Whisker speed (deg/s)"}


# ----------------------------------------------------------------------------------------------- converter rebuild
def converter_frame(h5, view):
    """DLC h5 -> converter dataframe (bodypart_coord columns), dlc2kinematics smoothing, top orientation check."""
    d = pv.wide(h5).astype(float)
    xy = d.columns.get_level_values("coords") != "likelihood"
    d.loc[:, xy] = savgol_filter(d.loc[:, xy].to_numpy(), 3, 1, axis=0)
    d.columns = ["_".join(c) for c in d.columns]
    rotated = False
    if view == "top":
        conf = (d.nose_tip_likelihood >= 0.6) & (d.nose_base_likelihood >= 0.6)
        dy = float(np.nanmedian((d.nose_tip_y - d.nose_base_y)[conf])) if conf.mean() >= 0.01 else np.nan
        if np.isfinite(dy) and dy < -20:
            for c in d.columns:
                if c.endswith("_x"):
                    d[c] = 639 - d[c]
                elif c.endswith("_y"):
                    d[c] = 479 - d[c]
            rotated = True
    return d, rotated


def build_nwb_like(h5, session, view, dest):
    """Converter series from a DLC h5, written with the real NWB timestamps / conversion. Returns the references."""
    d, rotated = converter_frame(h5, view)
    k = compute_kinematics_alt(d, "sideview" if view == "side" else "topview")
    with h5py.File(NWB / f"{session}.nwb", "r") as f:
        g = f[dh.TS_PATH]
        marker = "jaw_angle" if view == "side" else "whisker_angle"
        ts = g[marker]["timestamps"][:]
        conv_default = float(g[marker]["data"].attrs.get("conversion", 1.0))
        conv = {n: float(g[n]["data"].attrs.get("conversion", conv_default)) for n in g}
    n = min(len(k), len(ts))
    rate = np.round(1 / np.median(np.diff(ts[0:200])), 2)
    with h5py.File(dest, "w") as f:
        grp = f.create_group(dh.TS_PATH)
        for name in k.columns:
            out = f"top_{name}" if view == "top" and ("nose" in name or "particle" in name) else name
            x = k[name].to_numpy()[:n].astype(float)
            if "velocity" in name:
                x = x * rate
            s = grp.create_group(out)
            s.create_dataset("data", data=x).attrs["conversion"] = conv.get(out, conv_default)
            s.create_dataset("timestamps", data=ts[:n])
    refs = {"rotated": rotated, "n_frames": int(len(k)), "n_timestamps": int(len(ts))}
    if view == "side":
        cut = 0.5
        refs.update(jaw_ref_x_median=float(d.jaw_ref_x.median()),
                    jaw_ref_y_median=float(d.jaw_ref_y.median()),
                    jaw_ref5_x=float(np.percentile(np.where(d.jaw_likelihood > cut, d.jaw_x, 0), 5)),
                    jaw_ref5_y=float(np.percentile(np.where(d.jaw_likelihood > cut, d.jaw_y, 0), 5)),
                    pct_jaw_below_cut=float(100 * (d.jaw_likelihood <= cut).mean()),
                    pct_jaw_ref_below_cut=float(100 * (d.jaw_ref_likelihood <= cut).mean()))
    return refs


def load_view(path, view, session):
    v = dh.load_dlc(str(path), views=(view,))[view]
    v.session_id = session
    return v


def series(v, name):
    if name == "whisker_speed":
        x, _ = dh.clean_series(v, "whisker_velocity")
        return np.abs(x)
    x, _ = dh.clean_series(v, name, pupil_qc=False) if name == "pupil_area" else dh.clean_series(v, name)
    return x


def trials_of(session):
    with h5py.File(NWB / f"{session}.nwb", "r") as f:
        tr = f["intervals/trials"]
        cols = ["start_time", "trial_type", "lick_flag", "perf", "context", "early_lick", "lick_time",
                "response_window_start_time", "reward_available"]
        t = pd.DataFrame({c: [pq._str(x) if isinstance(x, bytes) else x for x in tr[c][:]] for c in cols if c in tr})
    return pq._trial_table(t)


def video_path(session, stem):
    mouse = session.split("_")[0]
    for r in ("Recording", "Training"):
        p = RAW / mouse / r / session / "Video" / f"{stem}.avi"
        if p.exists():
            return p
    return None


def read_frame(video, idx):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, img = cap.read()
    cap.release()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None


# ----------------------------------------------------------------------------------------------- drawing
def draw_keypoints(ax, h, i, view, color, refs=None):
    """Keypoints of frame i (raw h5 positions, likelihood >= 0.5 filled, below: open) and the derived geometry."""
    get = lambda kp: (h[(kp, "x")].iloc[i], h[(kp, "y")].iloc[i], h[(kp, "likelihood")].iloc[i])  # noqa: E731
    for kp in ts_.VIEW_KEYPOINTS[view]:
        if (kp, "x") not in h.columns:
            continue
        x, y, lk = get(kp)
        ax.scatter(x, y, s=9, facecolors=color if lk >= 0.5 else "none", edgecolors=color, lw=0.6, zorder=3)
    if view == "side":
        q = [get(f"pupil_{p}") for p in ("top", "right", "bottom", "left")]
        ax.add_patch(plt.Polygon([(a, b) for a, b, _ in q], fill=False, ec=color, lw=0.6))
        if refs:
            ax.plot(refs["jaw_ref5_x"], refs["jaw_ref5_y"], "+", color=color, ms=5, mew=0.8)
            jx, jy, _ = get("jaw")
            ax.plot([refs["jaw_ref5_x"], jx], [refs["jaw_ref5_y"], jy], color=color, lw=0.6)
            ax.plot(refs["jaw_ref_x_median"], refs["jaw_ref_y_median"], "x", color=color, ms=5, mew=0.8)
    else:
        for a, b in (("nose_base", "nose_tip"), ("whisker_base", "particle")):
            (x0, y0, _), (x1, y1, _) = get(a), get(b)
            ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=color, lw=0.7))


def pick_frame(old, new, rng):
    """Frame index: 99th percentile of |old - new| among frames defined in both; if < 1% are, a random frame defined
    in new but not in old."""
    both = np.isfinite(old) & np.isfinite(new)
    if both.mean() >= 0.01:
        diff = np.abs(old - new)
        thr = np.nanpercentile(diff[both], 99)
        cand = np.where(both & (diff >= thr))[0]
        return int(cand[np.argmin(np.abs(diff[cand] - thr))]), "99th pct |before - after|"
    cand = np.where(np.isfinite(new) & ~np.isfinite(old))[0]
    return int(rng.choice(cand)), "untracked before"


def mean_sem(m):
    n = np.sum(np.isfinite(m), 0)
    return np.nanmean(m, 0), np.nanstd(m, 0) / np.sqrt(np.maximum(n, 1))


# ----------------------------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sessions = []
    checks, refs_all = [], {}
    for sdir in sorted(d for d in pv.PILOT.iterdir() if d.name in pv.PILOT8):
        session = sdir.name
        new_h5 = next(sdir.glob("*_filtered.h5"))
        view = "side" if "_side" in new_h5.name else "top"
        old_h5 = pv.old_file(session, view)
        refs = {}
        for name, h5 in (("old", old_h5), ("new", new_h5)):
            refs[name] = build_nwb_like(h5, session, view, OUT / f"{session}_{view}_{name}.h5")
        refs_all[f"{session}_{view}"] = refs
        real = load_view(NWB / f"{session}.nwb", view, session)
        rebuilt = load_view(OUT / f"{session}_{view}_old.h5", view, session)
        new = load_view(OUT / f"{session}_{view}_new.h5", view, session)
        names = list(SIDE_Q if view == "side" else TOP_Q)
        for q in names:
            a, b = series(real, q), series(rebuilt, q)
            n = min(len(a), len(b))
            ok = np.isfinite(a[:n]) & np.isfinite(b[:n])
            checks.append({"session": session, "view": view, "series": q,
                           "median_abs_diff_rebuilt_old_vs_nwb": float(np.median(np.abs(a[:n] - b[:n])[ok]))
                           if ok.any() else np.nan, "pct_defined_nwb": 100 * np.isfinite(a).mean(),
                           "pct_defined_rebuilt_old": 100 * np.isfinite(b).mean()})
        tr = {name: {q: series(v, q) for q in names} for name, v in (("old", real), ("new", new))}
        if view == "top":  # whisker angle frames passing the dlc_helpers likelihood cutoffs (clean_series does not gate)
            for name, v in (("old", real), ("new", new)):
                ok = np.all([v.likelihood[l].to_numpy() >= dh.cutoff_for(l)
                             for l in dh.likelihood_series("whisker_angle")], 0)
                tr[name]["valid"] = 100 * ok.astype(float)
        sessions.append(dict(session=session, view=view, names=names, old_h5=old_h5, new_h5=new_h5,
                             ts={"old": real.timestamps, "new": new.timestamps}, tr=tr,
                             trials=trials_of(session), refs=refs))
        print(session, view, refs, flush=True)
    chk = pd.DataFrame(checks)
    chk.to_csv(OUT / "rebuild_check.csv", index=False)
    print(chk.round(4).to_string())
    (OUT / "references.json").write_text(json.dumps(refs_all, indent=1))
    ps.set_nature_rc()

    # ---- figure 1: frames
    fig, axs = plt.subplots(len(sessions), 4, figsize=(183 * ps.MM, 32 * ps.MM * len(sessions)))
    for r, s in enumerate(sessions):
        h = {"old": pv.wide(s["old_h5"]), "new": pv.wide(s["new_h5"])}
        stem = s["new_h5"].name.split("DLC_")[0]
        video = video_path(s["session"], stem)
        qs = ["pupil_area", "jaw_distance"] if s["view"] == "side" else ["whisker_angle", "whisker_angle"]
        for j, q in enumerate(qs):
            o, nw = s["tr"]["old"][q], s["tr"]["new"][q]
            n = min(len(o), len(nw))
            if j == 1 and s["view"] == "top":  # second top example: a random whisking frame tracked by the new model
                cand = np.where(np.isfinite(nw[:n]) & (s["tr"]["new"]["whisker_speed"][:n] > 500))[0]
                i, why = (int(rng.choice(cand)), "whisking (speed > 500 deg/s)") if len(cand) else \
                    pick_frame(o[:n], nw[:n], rng)
            else:
                i, why = pick_frame(o[:n], nw[:n], rng)
            img = read_frame(video, i) if video else None
            for k, name in enumerate(("old", "new")):
                ax = axs[r, 2 * j + k]
                if img is not None:
                    ax.imshow(img)
                draw_keypoints(ax, h[name], i, s["view"], COLORS[name],
                               s["refs"][name] if q == "jaw_distance" else None)
                val = s["tr"][name][q][i]
                unit = SIDE_Q.get(q, TOP_Q.get(q, "")).split("(")[-1].rstrip(")")
                ax.set_title(f"{'Before' if name == 'old' else 'After'}: {q.replace('_', ' ')} "
                             f"{'NaN' if not np.isfinite(val) else f'{val:.2f}'} {unit}", fontsize=5, pad=1,
                             color=COLORS[name])
                if q == "pupil_area":  # crop around the eye
                    cx, cy = np.nanmedian([h["new"][(f"pupil_{p}", "x")].iloc[i] for p in ("top", "bottom")]), \
                        np.nanmedian([h["new"][(f"pupil_{p}", "y")].iloc[i] for p in ("left", "right")])
                    ax.set_xlim(cx - 60, cx + 60); ax.set_ylim(cy + 45, cy - 45)
                ax.set_xticks([]); ax.set_yticks([])
            axs[r, 2 * j].set_ylabel(f"{s['session'][:5]} {s['session'][6:14]}\n{s['view']}, frame {i}\n{why}",
                                     fontsize=5)
    fig.tight_layout(pad=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"before_after_frames.{ext}", dpi=250)
    plt.close(fig)

    # ---- figure 2: trial-aligned means
    fig, axs = plt.subplots(len(sessions), 3, figsize=(183 * ps.MM, 30 * ps.MM * len(sessions)))
    summ = []
    for r, s in enumerate(sessions):
        cols = s["names"] + ([] if s["view"] == "side" else ["valid"])
        for c, q in enumerate(cols):
            ax = axs[r, c]
            for oc, color in OUTCOMES.items():
                ev = s["trials"].loc[s["trials"].outcome == oc, "start_time"].to_numpy()
                for name, ls in (("old", "--"), ("new", "-")):
                    x = s["tr"][name][q]
                    t, m = dh.peth(x, s["ts"][name][:len(x)], ev, window=WINDOW)
                    mu, se = mean_sem(m)
                    ax.plot(t, mu, color=color, ls=ls, lw=0.7 if name == "new" else 0.6)
                    if name == "new":
                        ax.fill_between(t, mu - se, mu + se, color=color, alpha=0.2, lw=0)
                    summ.append({"session": s["session"], "view": s["view"], "series": q, "outcome": oc,
                                 "model": name, "n_trials": len(ev),
                                 "pct_bins_defined": float(100 * np.isfinite(m).mean())})
            ax.axvline(0, color="k", lw=0.4)
            ax.set_ylabel({**SIDE_Q, **TOP_Q, "valid": "Whisker pts >= cutoff (%)"}[q], fontsize=5)
            if c == 0:
                ax.set_title(f"{s['session']} {s['view']}", fontsize=6, loc="left", pad=2)
        if r == len(sessions) - 1:
            for ax in axs[r]:
                ax.set_xlabel("Time from trial start (s)")
    handles = [plt.Line2D([], [], color=c, label=o) for o, c in OUTCOMES.items()] + \
              [plt.Line2D([], [], color="k", ls="--", label="before (NWB)"),
               plt.Line2D([], [], color="k", ls="-", label="after (merged model)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.985), pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"before_after_trial_aligned.{ext}", dpi=200)
    plt.close(fig)
    pd.DataFrame(summ).to_csv(OUT / "trial_aligned_defined.csv", index=False)

    # ---- figure 3: single-trial heatmaps, whisker trials
    picks = [("MH062", "jaw_distance"), ("MH070", "jaw_distance"), ("MH025", "whisker_angle"),
             ("MH064", "whisker_angle")]
    fig, axs = plt.subplots(2, 4, figsize=(183 * ps.MM, 90 * ps.MM))
    for c, (mouse, q) in enumerate(picks):
        s = next(x for x in sessions if x["session"].startswith(mouse))
        ev = s["trials"].loc[s["trials"].trial_type == "whisker_trial", "start_time"].to_numpy()
        mats = {name: dh.peth(s["tr"][name][q], s["ts"][name][:len(s["tr"][name][q])], ev, window=WINDOW)
                for name in ("old", "new")}
        lo, hi = np.nanpercentile(mats["new"][1], [2, 98])
        for r, name in enumerate(("old", "new")):
            t, m = mats[name]
            ax = axs[r, c]
            cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
            im = ax.imshow(m, aspect="auto", cmap=cmap, vmin=lo, vmax=hi, extent=(t[0], t[-1], len(ev), 0),
                           interpolation="none")
            ax.axvline(0, color="w", lw=0.4)
            ax.set_title(f"{'Before' if name == 'old' else 'After'}: {s['session'][:14]} {q.replace('_', ' ')}",
                         fontsize=5.5, pad=2, color=COLORS[name])
            ax.set_ylabel("Whisker trial" if c == 0 else "")
            if r == 1:
                ax.set_xlabel("Time from trial start (s)")
            fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).ax.tick_params(labelsize=4.5)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"before_after_heatmaps.{ext}", dpi=200)
    plt.close(fig)

    # ---- figure 4: jaw angle scale
    side = [s for s in sessions if s["view"] == "side"]
    fig, axs = plt.subplots(3, len(side), figsize=(183 * ps.MM, 150 * ps.MM))
    for c, s in enumerate(side):
        for r, q in enumerate(("jaw_angle", "jaw_distance")):
            o, nw = s["tr"]["old"][q], s["tr"]["new"][q]
            n = min(len(o), len(nw))
            ok = np.isfinite(o[:n]) & np.isfinite(nw[:n])
            ax = axs[r, c]
            ax.hexbin(o[:n][ok], nw[:n][ok], gridsize=60, bins="log", cmap="Greys", mincnt=1, linewidths=0)
            lim = np.nanpercentile(np.r_[o[:n][ok], nw[:n][ok]], [0.1, 99.9])
            ax.plot(lim, lim, color=COLORS["old"], lw=0.5, ls=":")
            slope = np.polyfit(o[:n][ok], nw[:n][ok], 1)[0]
            ax.set_title(f"{s['session'][:14]}: slope {slope:.2f}, r = {np.corrcoef(o[:n][ok], nw[:n][ok])[0, 1]:.2f}",
                         fontsize=5.5, pad=2)
            ax.set_xlabel(f"Before: {SIDE_Q[q]}"); ax.set_ylabel(f"After: {SIDE_Q[q]}")
        ax = axs[2, c]
        h = pv.wide(s["new_h5"])
        stem = s["new_h5"].name.split("DLC_")[0]
        video = video_path(s["session"], stem)
        i = int(np.nanargmin(np.abs(s["tr"]["new"]["jaw_angle"] - np.nanmedian(s["tr"]["new"]["jaw_angle"]))))
        img = read_frame(video, i) if video else None
        if img is not None:
            ax.imshow(img)
        for name in ("old", "new"):
            rf = s["refs"][name]
            ax.plot(rf["jaw_ref_x_median"], rf["jaw_ref_y_median"], "x", color=COLORS[name], ms=6, mew=1)
            ax.plot(rf["jaw_ref5_x"], rf["jaw_ref5_y"], "+", color=COLORS[name], ms=7, mew=1)
            ax.axvline(rf["jaw_ref_x_median"], color=COLORS[name], lw=0.4, ls=":")
            ax.axhline(rf["jaw_ref5_y"], color=COLORS[name], lw=0.4, ls=":")
        jx = h[("jaw", "x")].iloc[i]; jy = h[("jaw", "y")].iloc[i]
        ax.set_xlim(jx - 120, jx + 120); ax.set_ylim(jy + 90, jy - 90)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("x: median jaw_ref (angle x ref)  +: jaw 5th pct (angle y ref)", fontsize=4.5, pad=2)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"jaw_angle_scale.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
