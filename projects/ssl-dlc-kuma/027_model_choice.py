"""Model choice for re-inference: ResNet-50 / ResNet-101 / CSPNeXt-M on full videos (026), vs the old filtered h5s.

Videos: 6 well-tracked by the old models (held-out labelled frames, old error 0.34-0.51 mm; 3 side + 3 top) with s1,
s2, s6; the 8 worst videos of 023 with s2 (026) and s6 (023). No labels on these frames (labelled accuracy: 019 / 022):
the comparison is of tracking behaviour, per keypoint of the video's view (019), likelihood cutoff 0.5 (019).
Per video x model x keypoint: % frames confident, % frame-to-frame steps > 1 mm (5 ms), median step; distance to the
old model and to CSPNeXt-M on frames confident in both (median, mm). Per video x model: whisker angle (top) / jaw angle
and pupil area (side) as in 024 (% defined, correlation with CSPNeXt-M and with the old model). Speed: frames / s
(analyze_videos, batch 64, 1 L40S) and mean GPU utilisation (nvidia-smi every 5 s).
Figure (model_choice): a speed per model (dots = videos), b GPU utilisation, c-d % confident and jump rate per
keypoint (well-tracked videos, model - old), e distance to the old model per keypoint (well-tracked videos), f ResNet-101
vs CSPNeXt-M distance per keypoint (all videos), g worst videos: % frames confident per view keypoint, s2 vs s6 vs old,
h derived quantity % defined per video and model.
Outputs: DLCProjects/merged_training_2026-11-09/model_choice/ (keypoints.csv, derived.csv, speed.csv, model_choice.png)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 027_model_choice.py
"""

import importlib.util
import json
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


pv = _load("pv", "024_pilot_vs_old.py")
ts, ps = pv.ts, pv.ps
CUT = ts.CUTOFF
OUT = ts.SIDE / "model_choice"
RUNS = {1: ts.KUMA / "pilot_models/s1", 2: ts.KUMA / "pilot_models/s2", 6: ts.KUMA / "pilot_models/s6"}
WORST_S6 = ts.KUMA / "pilot_inference"
WELL = ["MH068_20260122_151614 side", "MH015_20250221_105527 side", "AB151_20250128_123409 side",
        "AB153_20250207_110145 top", "MH062_20260109_114135 top", "AB118_20240728_122249 top"]
NAMES = {"old": "Old (in use)", 1: ts.MODELS[1], 2: ts.MODELS[2], 6: ts.MODELS[6]}
COLORS = {"old": "#d6604d", 1: ts.MODEL_COLORS[1], 2: ts.MODEL_COLORS[2], 6: ts.MODEL_COLORS[6]}


def collect():
    """{(session, view): {model: (h5, timing dict or None, gpu csv or None)}}."""
    runs = {}
    for s, root in RUNS.items():
        for h5 in sorted(root.glob("*/*_filtered.h5")) if root.exists() else []:
            view = "side" if "_side" in h5.name else "top"
            stem = h5.name.split("DLC_")[0]
            tj, gpu = h5.parent / f"{stem}_timing.json", h5.parent / f"{stem}_gpu.csv"
            runs.setdefault((h5.parent.name, view), {})[s] = (h5, json.loads(tj.read_text()) if tj.exists() else None,
                                                              gpu if gpu.exists() else None)
    for h5 in sorted(WORST_S6.glob("*/*_filtered.h5")):
        view = "side" if "_side" in h5.name else "top"
        key = (h5.parent.name, view)
        stem = h5.name.split("DLC_")[0]
        if 6 not in runs.get(key, {}):
            tj = h5.parent / f"{stem}_timing.json"
            runs.setdefault(key, {})[6] = (h5, json.loads(tj.read_text()) if tj.exists() else None, None)
    return runs


def pupil_area(d):
    w = pd.DataFrame({f"pupil_{p}_{c}": d[(f"pupil_{p}", c)].to_numpy() for p in ("top", "right", "bottom", "left")
                      for c in "xy"})
    v = ts.pupil_area4(w).to_numpy().astype(float)
    ok = np.all([d[(f"pupil_{p}", "likelihood")].to_numpy() >= CUT for p in ("top", "right", "bottom", "left")], 0)
    v[~ok] = np.nan
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = collect()
    krows, drows, srows = [], [], []
    for (session, view), models in sorted(runs.items()):
        group = "well" if f"{session} {view}" in WELL else "worst"
        mouse = session.split("_")[0]
        mmpx = pv.mm_per_px(mouse, view)
        try:
            data = {"old": pv.wide(pv.old_file(session, view))}
        except AssertionError:  # no old filtered h5 (e.g. AB118_20240728: DLC never run): new models only
            data = {}
        for s, (h5, timing, gpu) in models.items():
            data[s] = pv.wide(h5)
            if timing:
                g = pd.read_csv(gpu, header=None, names=["util", "mem"]) if gpu else None
                srows.append({"session": session, "view": view, "group": group, "model": NAMES[s],
                              "n_frames": timing["n_frames"], "inference_s": timing["inference_s"],
                              "frames_per_s": timing["inference_frames_per_s"],
                              "gpu_util_mean": float(g.util.mean()) if g is not None else np.nan,
                              "gpu_mem_mb": float(g.mem.max()) if g is not None else np.nan})
        n = min(len(d) for d in data.values())
        data = {k: d.iloc[:n] for k, d in data.items()}
        conf = {}
        for kp in ts.VIEW_KEYPOINTS[view]:
            for m, d in data.items():
                if (kp, "x") not in d.columns:
                    continue
                c, met = pv.keypoint_metrics(d, kp, mmpx)
                conf[(m, kp)] = c
                row = {"session": session, "view": view, "group": group, "keypoint": kp, "model": NAMES[m], **met}
                for ref in ("old", 6):
                    if ref == m or ref not in data:
                        continue
                    r = data[ref]
                    if (kp, "x") not in r.columns:
                        continue
                    cr = r[(kp, "likelihood")].to_numpy() >= CUT
                    both = c & cr
                    dist = np.hypot(d[(kp, "x")].to_numpy() - r[(kp, "x")].to_numpy(),
                                    d[(kp, "y")].to_numpy() - r[(kp, "y")].to_numpy())[both] * mmpx
                    row[f"dist_to_{'old' if ref == 'old' else 's6'}_mm"] = float(np.median(dist)) if len(dist) else np.nan
                krows.append(row)
        qs = {"top": {"whisker angle": lambda d: pv.derived(d, "top")[1]},
              "side": {"jaw angle": lambda d: pv.derived(d, "side")[1], "pupil area": pupil_area}}[view]
        for q, fn in qs.items():
            vals = {m: fn(d) for m, d in data.items()}
            for m, v in vals.items():
                row = {"session": session, "view": view, "group": group, "quantity": q, "model": NAMES[m],
                       "pct_defined": 100 * np.isfinite(v).mean()}
                for ref in ("old", 6):
                    if ref in vals and ref != m:
                        ok = np.isfinite(v) & np.isfinite(vals[ref])
                        row[f"r_with_{'old' if ref == 'old' else 's6'}"] = \
                            float(np.corrcoef(v[ok], vals[ref][ok])[0, 1]) if ok.sum() > 100 else np.nan
                drows.append(row)
        print(session, view, group, sorted(map(str, data)), flush=True)
    k, dq, sp = pd.DataFrame(krows), pd.DataFrame(drows), pd.DataFrame(srows)
    k.to_csv(OUT / "keypoints.csv", index=False)
    dq.to_csv(OUT / "derived.csv", index=False)
    sp.to_csv(OUT / "speed.csv", index=False)

    ps.set_nature_rc()
    order = [NAMES[m] for m in ("old", 1, 2, 6)]
    col = {NAMES[m]: COLORS[m] for m in COLORS}
    fig = plt.figure(figsize=(183 * ps.MM, 210 * ps.MM))
    gs = fig.add_gridspec(4, 4, left=0.08, right=0.97, top=0.96, bottom=0.08, wspace=0.6, hspace=1.0)
    for j, (c, lab, title) in enumerate((("frames_per_s", "a", "Inference speed (frames / s)"),
                                         ("gpu_util_mean", "b", "Mean GPU utilisation (%)"))):
        ax = fig.add_subplot(gs[0, j]); ts.panel_label(ax, lab)
        for i, m in enumerate(order[1:]):
            d = sp[sp.model == m]
            ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d[c], s=6,
                       color=col[m], edgecolors="none")
            if len(d):
                ax.hlines(d[c].median(), i - 0.3, i + 0.3, color="k", lw=0.8)
        ax.set_xticks(range(3), [m.replace("Merged ", "") for m in order[1:]], rotation=30, ha="right")
        ax.set_ylabel(title)
        if c == "frames_per_s":
            ax.axhline(200, color="0.5", lw=0.5, ls=":")
    kw = k[k.group == "well"]
    for j, (c, lab, title) in enumerate((("pct_confident", "c", "Frames confident (%)"),
                                         ("pct_jumps", "d", "Steps > 1 mm (%)"))):
        ax = fig.add_subplot(gs[0, 2 + j]); ts.panel_label(ax, lab)
        piv = kw.pivot_table(index=["view", "keypoint"], columns="model", values=c)
        for i, m in enumerate(order):
            if m in piv:
                ax.scatter(np.full(len(piv), i) + np.random.default_rng(1).uniform(-0.15, 0.15, len(piv)), piv[m],
                           s=5, color=col[m], edgecolors="none")
                ax.hlines(piv[m].median(), i - 0.3, i + 0.3, color="k", lw=0.8)
        ax.set_xticks(range(4), [m.replace("Merged ", "") for m in order], rotation=30, ha="right")
        ax.set_ylabel(title); ax.set_title("well-tracked videos, per keypoint", fontsize=5, pad=2)
        if c == "pct_jumps":
            ax.set_yscale("symlog", linthresh=0.01)
    for j, (c, lab, title, grp) in enumerate((("dist_to_old_mm", "e", "Distance to old model (mm)", "well"),
                                              ("dist_to_s6_mm", "f", "Distance to CSPNeXt-M (mm)", None))):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); ts.panel_label(ax, lab)
        d = k if grp is None else k[k.group == grp]
        piv = d.pivot_table(index=["view", "keypoint"], columns="model", values=c)
        x = np.arange(len(piv))
        ms = [m for m in order[1:] if m in piv]
        for i, m in enumerate(ms):
            ax.bar(x + (i - (len(ms) - 1) / 2) * 0.8 / len(ms), piv[m], width=0.8 / len(ms), color=col[m], label=m)
        ax.set_xticks(x, [f"{v[0]} {kp.replace('_', ' ')}" for v, kp in zip(piv.index.get_level_values(0),
                                                                           piv.index.get_level_values(1))],
                      rotation=60, ha="right", fontsize=4.5)
        ax.set_ylabel(title); ax.legend(frameon=False, fontsize=5)
        ax.set_title("well-tracked videos" if grp else "all videos (median over videos)", fontsize=5, pad=2)
    ax = fig.add_subplot(gs[2, :2]); ts.panel_label(ax, "g")
    kwst = k[k.group == "worst"]
    piv = kwst.pivot_table(index=["view", "keypoint"], columns="model", values="pct_confident")
    x = np.arange(len(piv))
    ms = [m for m in order if m in piv and m != NAMES[1]]
    for i, m in enumerate(ms):
        ax.bar(x + (i - (len(ms) - 1) / 2) * 0.8 / len(ms), piv[m], width=0.8 / len(ms), color=col[m], label=m)
    ax.set_xticks(x, [f"{v} {kp.replace('_', ' ')}" for v, kp in piv.index], rotation=60, ha="right", fontsize=4.5)
    ax.set_ylabel("Frames confident (%)"); ax.legend(frameon=False, fontsize=5)
    ax.set_title("8 worst videos (mean over videos)", fontsize=5, pad=2)
    ax = fig.add_subplot(gs[2, 2:]); ts.panel_label(ax, "h")
    piv = dq.pivot_table(index=["session", "view", "quantity"], columns="model", values="pct_defined")
    x = np.arange(len(piv))
    ms = [m for m in order if m in piv]
    for i, m in enumerate(ms):
        ax.bar(x + (i - (len(ms) - 1) / 2) * 0.8 / len(ms), piv[m], width=0.8 / len(ms), color=col[m], label=m)
    ax.set_xticks(x, [f"{s[:5]} {s[6:14]} {q}" for s, v, q in piv.index], rotation=60, ha="right", fontsize=4)
    ax.set_ylabel("Frames defined (%)"); ax.set_title("derived quantity, all keypoints >= 0.5", fontsize=5, pad=2)
    ax = fig.add_subplot(gs[3, :]); ts.panel_label(ax, "i")
    piv = dq.pivot_table(index=["session", "view", "quantity"], columns="model", values="r_with_s6")
    for i, m in enumerate([m for m in order if m in piv]):
        ax.scatter(np.arange(len(piv)) + (i - 1) * 0.15, piv[m], s=8, color=col[m], edgecolors="none", label=m)
    ax.set_xticks(np.arange(len(piv)), [f"{s[:5]} {s[6:14]} {q}" for s, v, q in piv.index], rotation=45, ha="right",
                  fontsize=4.5)
    ax.set_ylabel("r with CSPNeXt-M"); ax.set_ylim(-0.05, 1.02); ax.legend(frameon=False, fontsize=5, ncol=3)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model_choice.{ext}", dpi=200)
    print(sp.groupby("model")[["frames_per_s", "gpu_util_mean", "gpu_mem_mb"]].median().round(1).to_string())
    print(k.groupby(["group", "model"])[["pct_confident", "pct_jumps", "dist_to_old_mm", "dist_to_s6_mm"]]
          .median().round(3).to_string())
    print(dq.groupby(["group", "quantity", "model"])[["pct_defined", "r_with_old", "r_with_s6"]].median().round(3)
          .to_string())


if __name__ == "__main__":
    main()
