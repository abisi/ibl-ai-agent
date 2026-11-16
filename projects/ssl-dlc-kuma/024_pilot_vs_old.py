"""Pilot inference of the merged model (023) vs the filtered h5s in use, on full videos (user 2026-11-09).

Videos: the 8 sessions/views of the worst 022 example frames (4 side, 4 top, incl. the failed top views MH025 0421 and
MH027 0422). New = CSPNeXt-M (shuffle 6) snapshot-330, DLC median filter; old = the *_filtered.h5 in the session's
Video folder (s3 for AB, s5 best-540 for MH015/025/027, s6 for MH062-070). No labels on these frames: the comparison is
of tracking behaviour. Keypoints of the video's view only (as 019), likelihood cutoff 0.5 (019), mm per px of the
mouse and view (mm_per_px.csv, view median as fallback).
Per keypoint and model: % frames confident, frame-to-frame step (median, 99th percentile) and % steps > 1 mm (5 ms at
200 Hz) between consecutive confident frames, and the old <-> new distance on frames confident in both. Per video:
whisker angle (compute_kinematics_alt, top) or jaw angle (side): % frames defined, frame-to-frame step.
Figure: a % confident new vs old (per keypoint), b jump rate, c old <-> new median distance, d inference speed per
video, e-l 10-s traces (window of largest activity of the new trace), old vs new.
Outputs: DLCProjects/merged_training_2026-11-09/pilot_inference/ (pilot_metrics.csv, pilot_derived.csv,
pilot_timing.csv, pilot_vs_old.png/pdf, pilot_summary.json)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 024_pilot_vs_old.py
"""

import importlib.util
import json
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ts = _load("ts", "019_training_summary.py")
ps = ts.ps
CUT = ts.CUTOFF
PILOT = ts.KUMA / "pilot_inference"
OUT = ts.SIDE / "pilot_inference"
DATA = {"AB": ts.AB / "data", "MH": Path("/mnt/lsens-analysis/Myriam_Hamon/data")}
INVENTORY = ts.AB / "combined_results_ks4/ssl-dlc-kuma/inventory.csv"
COLORS = {"old": "#d6604d", "new": ts.MODEL_COLORS[6]}
PILOT8 = {"AB155_20250208_175103", "MH062_20260109_114135", "AB153_20250207_110145", "MH070_20260121_140848",
          "MH025_20250421_115741", "MH027_20250422_111013", "MH064_20260114_102210", "MH015_20250221_105527"}  # 023 list;
# the other failed top views re-run later (030) share the folder
JUMP_MM = 1.0
WINDOW_S = 10
FPS = 200


def wide(path):
    df = pd.read_hdf(path)
    df.columns = df.columns.droplevel([n for n in df.columns.names if n not in ("bodyparts", "coords")])
    return df


def old_file(session, view):
    mouse = session.split("_")[0]
    folder = DATA[mouse[:2]] / mouse / session / "Video"
    files = [f for f in folder.glob(f"{mouse}_{view}*filtered.h5") if "backup" not in f.name]
    assert len(files) == 1, files
    return files[0]


def mm_per_px(mouse, view):
    mm = pd.read_csv(ts.SIDE / "mm_per_px.csv")
    m = mm[(mm.mouse == mouse) & (mm.view == view)].mm_per_px
    return float(m.median() if len(m) else mm[mm.view == view].mm_per_px.median())


def keypoint_metrics(d, kp, mmpx):
    x, y, lik = (d[(kp, c)].to_numpy() for c in ("x", "y", "likelihood"))
    conf = lik >= CUT
    step = np.hypot(np.diff(x), np.diff(y)) * mmpx
    step = step[conf[1:] & conf[:-1]]
    return conf, {"pct_confident": 100 * conf.mean(),
                  "step_median_mm": np.median(step) if len(step) else np.nan,
                  "step_p99_mm": np.percentile(step, 99) if len(step) else np.nan,
                  "pct_jumps": 100 * (step > JUMP_MM).mean() if len(step) else np.nan}


def derived(d, view):
    """Whisker angle (top, compute_kinematics_alt, nose reference = session median of confident frames) or jaw angle
    (side, reference y = 5th percentile of confident jaw_y, jaw_ref x = median); NaN where a keypoint is < cutoff."""
    ok = lambda kp: d[(kp, "likelihood")].to_numpy() >= CUT  # noqa: E731
    g = lambda kp, c: d[(kp, c)].to_numpy()  # noqa: E731
    if view == "top":
        nose = ok("nose_tip") & ok("nose_base")
        ref = (np.median((g("nose_tip", "x") - g("nose_base", "x"))[nose]),
               np.median((g("nose_tip", "y") - g("nose_base", "y"))[nose]))
        w = pd.DataFrame({f"{k}_{c}": g(k, c) for k in ("particle", "whisker_base") for c in "xy"})
        v = ts.whisker_angle_alt(w, ref).to_numpy().copy()
        v[~(ok("particle") & ok("whisker_base"))] = np.nan
        return "whisker angle (deg)", v
    jaw, jr = ok("jaw"), ok("jaw_ref")
    w = pd.DataFrame({"jaw_x": g("jaw", "x"), "jaw_y": g("jaw", "y")})
    v = ts.jaw_angle_alt(w, np.percentile(g("jaw", "y")[jaw], 5), np.median(g("jaw_ref", "x")[jr])).to_numpy().copy()
    v[~jaw] = np.nan
    return "jaw angle (deg)", v


def active_window(v, n):
    """Start of the n-frame window with the largest variance of v (NaN-aware, 1-s grid)."""
    best, start = -1, 0
    for s in range(0, len(v) - n, FPS):
        seg = v[s:s + n]
        if np.isfinite(seg).mean() > 0.8 and np.nanvar(seg) > best:
            best, start = np.nanvar(seg), s
    return start


def dataset_frames():
    inv = pd.read_csv(INVENTORY)
    n = inv.video_n_frames.dropna().astype(str).map(lambda s: sum(int(i) for i in re.findall(r"\d+", s)))
    return int(n.sum()), int((inv.n_videos > 0).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, drows, trows, traces = [], [], [], []
    for sdir in sorted(d for d in PILOT.iterdir() if d.name in PILOT8):
        session = sdir.name
        mouse = session.split("_")[0]
        new_f = next(sdir.glob("*_filtered.h5"))
        view = "side" if "_side" in new_f.name else "top"
        timing = json.loads(next(sdir.glob("*_timing.json")).read_text())
        old_f = old_file(session, view)
        old_shuffle = re.search(r"shuffle(\d)", old_f.name)
        old_name = f"s{old_shuffle.group(1)}" if old_shuffle else "s6"  # MH062-070 h5s were renamed (shuffle 6)
        new, old = wide(new_f), wide(old_f)
        n = min(len(new), len(old))
        trows.append({"session": session, "view": view, "n_frames": timing["n_frames"], "n_new": len(new),
                      "n_old": len(old), "inference_s": timing["inference_s"], "filter_s": timing["filter_s"],
                      "frames_per_s": timing["inference_frames_per_s"]})
        new, old = new.iloc[:n], old.iloc[:n]
        mmpx = mm_per_px(mouse, view)
        for kp in ts.VIEW_KEYPOINTS[view]:
            res = {}
            for name, d in (("new", new), ("old", old)):
                if (kp, "x") not in d.columns:
                    continue
                res[name] = keypoint_metrics(d, kp, mmpx)
                rows.append({"session": session, "view": view, "keypoint": kp, "model": name,
                             "old_shuffle": old_name, **res[name][1]})
            if len(res) == 2:
                both = res["new"][0] & res["old"][0]
                dist = np.hypot(new[(kp, "x")].to_numpy() - old[(kp, "x")].to_numpy(),
                                new[(kp, "y")].to_numpy() - old[(kp, "y")].to_numpy())[both] * mmpx
                rows.append({"session": session, "view": view, "keypoint": kp, "model": "old_vs_new",
                             "old_shuffle": old_name, "pct_confident": 100 * both.mean(),
                             "dist_median_mm": np.median(dist) if len(dist) else np.nan,
                             "pct_dist_gt_0p5mm": 100 * (dist > 0.5).mean() if len(dist) else np.nan})
        tr = {}
        for name, d in (("new", new), ("old", old)):
            label, v = derived(d, view)
            step = np.abs(np.diff(v))
            drows.append({"session": session, "view": view, "quantity": label, "model": name,
                          "pct_defined": 100 * np.isfinite(v).mean(), "step_median_deg": np.nanmedian(step),
                          "step_p99_deg": np.nanpercentile(step, 99)})
            tr[name] = v
        traces.append((session, view, old_name, label, tr))
    m, dq, tt = pd.DataFrame(rows), pd.DataFrame(drows), pd.DataFrame(trows)
    m.to_csv(OUT / "pilot_metrics.csv", index=False)
    dq.to_csv(OUT / "pilot_derived.csv", index=False)
    tt.to_csv(OUT / "pilot_timing.csv", index=False)

    # full-dataset estimate from the aggregate speed (all pilot frames / all pilot inference time)
    fps_all = tt.n_frames.sum() / tt.inference_s.sum()
    n_total, n_videos = dataset_frames()
    gpu_h = n_total / fps_all / 3600
    summary = {"pilot_frames": int(tt.n_frames.sum()), "pilot_inference_h": float(tt.inference_s.sum() / 3600),
               "aggregate_frames_per_s": float(fps_all), "per_video_frames_per_s": tt.frames_per_s.round(1).tolist(),
               "filter_s_per_million_frames": float(tt.filter_s.sum() / tt.n_frames.sum() * 1e6),
               "dataset_frames": n_total, "dataset_videos": n_videos, "dataset_gpu_hours": float(gpu_h),
               "days_on_8_gpus": float(gpu_h / 8 / 24),
               "frame_count_mismatch": tt[tt.n_new != tt.n_old][["session", "n_new", "n_old"]].to_dict("records")}
    (OUT / "pilot_summary.json").write_text(json.dumps(summary, indent=1))

    # figure
    ps.set_nature_rc()
    fig = plt.figure(figsize=(183 * ps.MM, 200 * ps.MM))
    gs = fig.add_gridspec(6, 4, left=0.08, right=0.97, top=0.96, bottom=0.05, wspace=0.95, hspace=0.9)
    kp_tab = m[m.model.isin(["new", "old"])].pivot_table(index=["session", "view", "keypoint"], columns="model",
                                                        values=["pct_confident", "pct_jumps"]).reset_index()
    for j, (col, title, lab) in enumerate((("pct_confident", "Frames confident (%)", "a"),
                                           ("pct_jumps", f"Steps > {JUMP_MM:g} mm (%)", "b"))):
        ax = fig.add_subplot(gs[:2, j]); ts.panel_label(ax, lab)
        for v in ts.VIEWS:
            k = kp_tab[kp_tab.view == v]
            ax.scatter(k[(col, "old")], k[(col, "new")], s=6, color=ts.VIEW_COLORS[v], edgecolors="none", label=v)
        lim = [0, max(np.nanmax(kp_tab[(col, "old")]), np.nanmax(kp_tab[(col, "new")])) * 1.05]
        if col == "pct_jumps":
            ax.set_xscale("symlog", linthresh=0.1); ax.set_yscale("symlog", linthresh=0.1)
        ax.plot(lim, lim, color="0.5", lw=0.5, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f"Old in use: {title.lower()}"); ax.set_ylabel(f"Merged: {title.lower()}")
        if j == 0:
            ax.legend(frameon=False, loc="lower right")
    ax = fig.add_subplot(gs[:2, 2]); ts.panel_label(ax, "c")
    a = m[m.model == "old_vs_new"].copy()
    a["label"] = a.session.str[:5] + " " + a.view
    piv = a.pivot_table(index="keypoint", columns="label", values="dist_median_mm")
    im = ax.imshow(piv.to_numpy(), cmap="magma_r", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(piv)), piv.index, fontsize=5); ax.set_xticks(range(piv.shape[1]), piv.columns,
                                                                          rotation=90, fontsize=5)
    ax.set_title("Old <-> merged median distance (mm)", pad=2)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
    ax = fig.add_subplot(gs[:2, 3]); ts.panel_label(ax, "d")
    ax.barh(range(len(tt)), tt.frames_per_s, color=[ts.VIEW_COLORS[v] for v in tt.view])
    ax.axvline(fps_all, color="k", lw=0.6, ls="--")
    ax.axvline(FPS, color="0.5", lw=0.6, ls=":")
    ax.set_yticks(range(len(tt)), tt.session.str[:5] + " " + tt.view, fontsize=5)
    ax.set_xlabel("Inference speed (frames / s, 1 L40S)")
    ax.set_title(f"{fps_all:.0f} fr/s: {gpu_h:.0f} GPU-h for\n{n_total / 1e6:.0f} M frames", pad=2, fontsize=6)
    for i, (session, view, old_name, label, tr) in enumerate(traces):
        ax = fig.add_subplot(gs[2 + i // 2, (i % 2) * 2:(i % 2) * 2 + 2]); ts.panel_label(ax, "efghijkl"[i])
        n = WINDOW_S * FPS
        s0 = active_window(tr["new"], n)
        t = np.arange(n) / FPS
        for name in ("old", "new"):
            ax.plot(t, tr[name][s0:s0 + n], color=COLORS[name], lw=0.5,
                    label=f"{'old ' + old_name if name == 'old' else 'merged'}")
        ax.set_title(f"{session} {view}, from {s0 / FPS / 60:.1f} min", pad=2, fontsize=6)
        ax.set_ylabel(label, fontsize=5)
        if i >= 6:
            ax.set_xlabel("Time (s)")
        ax.legend(frameon=False, fontsize=5, ncol=2, loc="upper left", borderaxespad=0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pilot_vs_old.{ext}", dpi=200)
    print(json.dumps(summary, indent=1))
    print(m[m.model.isin(["new", "old"])].groupby(["session", "view", "model"])[
        ["pct_confident", "pct_jumps", "step_median_mm"]].mean().round(2).to_string())
    print(a.groupby(["session", "view"]).dist_median_mm.median().round(3).to_string())
    print(dq.round(2).to_string())


if __name__ == "__main__":
    main()
