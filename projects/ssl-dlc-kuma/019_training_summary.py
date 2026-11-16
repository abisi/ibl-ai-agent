"""Merged DLC model training summary (training_plan.md section 7): figure, per-snapshot error tables, source data.

Inputs (DLCProjects/merged_training_2026-11-09/, copied from Kuma by 019_sync_from_kuma.sh):
  kuma/split_ground_truth.h5             ground truth + train/test split (018_export_split.py)
  kuma/learning_stats_shuffle<s>.csv     DLC training log per architecture (train loss, DLC test metrics / 10 epochs)
  kuma/eval_shuffle<s>/*snapshot_<e>.h5  predictions of every labelled frame per snapshot (017_evaluate.sbatch)
  human_variability_mm.csv, mm_per_px.csv (015, 016)
Architectures without evaluation files (still training) appear only in the training-loss and DLC-metric panels.

Keypoints per view (VIEW_KEYPOINTS): only the keypoints of a frame's view are quantified (the model predicts all 13 on
every frame). Mapping = the keypoints labelled in each view and used in the NWB (NWB_converter DLC_to_nwb: top series
are nose, particle and whisker): side = pupil x4, nose tip / base, jaw, jaw_ref, tongue, particle; top = nose tip /
base, particle, whisker base / tip. Top-view tongue (22 labels, not used in the NWB) and the spout (user 2026-11-09)
are excluded.

Error = Euclidean distance between prediction and label over the view's keypoints labelled in the frame, all
predictions whatever their likelihood (DLC's test error pooling), on the TEST frames unless stated. mm: NWB camera
conversion of the frame's folder (mm_per_px.csv); mm axes on px panels use the view's median mm per px. Likelihood
cutoff 0.5 throughout (user 2026-11-09; project config pcutoff 0.4). Best snapshot = lowest mean test error (chosen
on the test frames: there is no separate validation set; the final snapshot is reported too).

Panels: a example predictions; b dataset; c training loss; d DLC test metric; e model comparison (best-snapshot test
error per view, mm); f train / test error during training, all models; g the same, best model, side / top; h per-frame
test error; i PCK; j error vs likelihood; k per keypoint; l presence; m-p derived quantities; q per mouse.

Outputs: DLCProjects/merged_training_2026-11-09/training_summary/
  training_summary.{pdf,svg,eps,ai,png,jpg}, errors_per_snapshot.csv (model x epoch x split x view),
  errors_best_per_keypoint.csv, errors_best_per_mouse.csv, presence.csv, summary.json
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 019_training_summary.py
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl-paper-figures"))
import paperstyle as ps  # noqa: E402

AB = Path("/mnt/lsens-analysis/Axel_Bisi")
PROJ = AB / "DLCProjects/Tracking2D-Axel-2026-11-04"
SIDE = AB / "DLCProjects/merged_training_2026-11-09"
KUMA = SIDE / "kuma"
OUT = SIDE / "training_summary"
MODELS = {1: "ResNet-50", 2: "ResNet-101", 3: "HRNet-W18", 4: "HRNet-W32", 5: "HRNet-W48", 6: "CSPNeXt-M"}
MODEL_COLORS = {1: "#1f77b4", 2: "#08306b", 3: "#fdae6b", 4: "#e6550d", 5: "#7f2704", 6: "#2ca02c"}
VIEW_COLORS = {"side": "#4d4d4d", "top": "#9e9ac8"}
VIEWS = ("side", "top")
CUTOFF = 0.5  # likelihood cutoff for the whole figure (user 2026-11-09)
VIEW_KEYPOINTS = {
    "side": ["pupil_top", "pupil_left", "pupil_right", "pupil_bottom", "nose_tip", "nose_base", "jaw", "jaw_ref",
             "tongue", "particle"],
    "top": ["nose_tip", "nose_base", "particle", "whisker_base", "whisker_tip"],
}
GT_INDEX = []  # ground-truth image index (labeled-data, folder, image), set by load_ground_truth


def frame_table(df):
    """Long table (frame, bodypart, x, y[, likelihood]) from a DLC dataframe (scorer[, individuals], bodyparts, coords)."""
    d = df.copy()
    d.columns = d.columns.droplevel([n for n in d.columns.names if n not in ("bodyparts", "coords")])
    long = d.stack(level=0, future_stack=True).reset_index()
    long.columns = ["frame", "bodypart"] + list(long.columns[2:])
    return long


def in_view(df):
    """Rows whose bodypart belongs to the frame's view."""
    keep = pd.Series(False, index=df.index)
    for v, kps in VIEW_KEYPOINTS.items():
        keep |= (df.view == v) & df.bodypart.isin(kps)
    return df[keep]


def load_ground_truth():
    gt = pd.read_hdf(KUMA / "split_ground_truth.h5", "df")
    split = gt[("split", "", "")]
    gt = gt.drop(columns=[("split", "", "")])
    frames = pd.DataFrame({"frame": range(len(gt)), "split": split.to_numpy()})
    frames["folder"] = [i[1] for i in gt.index]
    frames["image"] = ["/".join(i) for i in gt.index]
    frames["mouse"] = frames.folder.str.extract(r"^([A-Z]+\d+)")[0]
    frames["view"] = np.where(frames.folder.str.contains("_side|side_", case=False), "side",
                              np.where(frames.folder.str.contains("_top|top_", case=False), "top", "unknown"))
    mm = pd.read_csv(SIDE / "mm_per_px.csv")[["folder", "mm_per_px"]]
    frames = frames.merge(mm, on="folder", how="left")
    frames["mm_per_px"] = frames.mm_per_px.fillna(frames.groupby("view").mm_per_px.transform("median"))
    GT_INDEX.extend(gt.index)
    gt.index = range(len(gt))
    long = frame_table(gt).rename(columns={"x": "x_gt", "y": "y_gt"})
    return frames, long


def snapshot_files(shuffle):
    files = {}
    for f in (KUMA / f"eval_shuffle{shuffle}").glob("*snapshot_*.h5"):
        m = re.search(r"snapshot_(\d+)\.h5$", f.name)
        if m:
            files[int(m.group(1))] = f
    return dict(sorted(files.items()))


def errors(pred_file, frames, gt_long):
    """Per frame x keypoint of the frame's view: prediction, label, error (px, mm), likelihood, labelled flag."""
    p = pd.read_hdf(pred_file).reindex(pd.MultiIndex.from_tuples(GT_INDEX))  # align on the image path
    p.index = range(len(p))
    e = gt_long.merge(frame_table(p), on=["frame", "bodypart"])
    e["labelled"] = e[["x_gt", "y_gt"]].notna().all(axis=1)
    e["err_px"] = np.hypot(e.x - e.x_gt, e.y - e.y_gt)
    e = e.merge(frames[["frame", "split", "view", "mouse", "mm_per_px"]], on="frame")
    e["err_mm"] = e.err_px * e.mm_per_px
    return in_view(e)


def human_reference():
    h = pd.read_csv(SIDE / "human_variability_mm.csv")
    return in_view(h[h.state == "both"])


def panel_label(ax, letter, dx=-24):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 8), textcoords="offset points",
                fontsize=8, fontweight="bold", va="bottom", ha="left")


def human_line(ax, value, label=True):
    ax.axhline(value, color="0.35", lw=0.6, ls="--", zorder=0)
    if label:
        ax.text(1.0, value, " human", transform=ax.get_yaxis_transform(), fontsize=5, color="0.35", va="center")


def mm_axis(ax, mm_per_px, which="y"):
    """Secondary axis in mm for a px axis (view's median mm per px)."""
    f = (lambda p: p * mm_per_px, lambda m: m / mm_per_px)
    sec = ax.secondary_yaxis("right", functions=f) if which == "y" else ax.secondary_xaxis("top", functions=f)
    sec.set_ylabel("mm") if which == "y" else sec.set_xlabel("mm")
    return sec


def pick_side_lick(t):
    """Side test frame with a clear lick: tongue labelled and predicted (likelihood >= CUTOFF, error < 5 px), the
    frame's mean error below the view median, tongue furthest from the jaw."""
    lab = t[t.labelled]
    fe = lab.groupby("frame").err_px.mean()
    tg = t[(t.bodypart == "tongue") & t.labelled & (t.likelihood >= CUTOFF) & (t.err_px < 5)].set_index("frame")
    jw = t[t.bodypart == "jaw"].set_index("frame")
    ok = [f for f in tg.index if f in jw.index and fe.get(f, np.inf) <= fe.median()]
    ext = pd.Series({f: np.hypot(tg.at[f, "x"] - jw.at[f, "x"], tg.at[f, "y"] - jw.at[f, "y"]) for f in ok})
    return int(ext.idxmax())


def pick_median(t):
    """Test frame with (nearly) every view keypoint labelled, mean error closest to the median."""
    full = t.groupby("frame").labelled.sum()
    full = full[full >= full.max() - 1].index
    fe = t[t.frame.isin(full) & t.labelled].groupby("frame").err_px.mean()
    return int((fe - fe.median()).abs().idxmin())


def whisker_angle_overlay(ax, p, dy):
    """Top view: nose vector (nose base -> nose tip), whisker vector (whisker base -> particle) and the angle
    between them (arc at the whisker base, value in degrees)."""
    need = {"nose_tip", "nose_base", "whisker_base", "particle"}
    if not need <= set(p.index):
        return None
    xy = {b: np.array([p.at[b, "x"], p.at[b, "y"] + dy]) for b in need}
    nose, whisk = xy["nose_tip"] - xy["nose_base"], xy["particle"] - xy["whisker_base"]
    ax.annotate("", xy=xy["nose_tip"], xytext=xy["nose_base"], zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0))
    # the particle sits close to the whisker base: the whisker vector is drawn extended along its direction
    w_end = xy["whisker_base"] + whisk / np.linalg.norm(whisk) * max(np.linalg.norm(whisk), 90)
    ax.annotate("", xy=w_end, xytext=xy["whisker_base"], zorder=2,
                arrowprops=dict(arrowstyle="-|>", color="white", lw=0.8, mutation_scale=6, shrinkA=0, shrinkB=0))
    ang = np.degrees(np.arccos(np.dot(nose, whisk) / (np.linalg.norm(nose) * np.linalg.norm(whisk))))
    # arc at the whisker base between the whisker vector and a copy of the nose vector
    a0, a1 = np.arctan2(nose[1], nose[0]), np.arctan2(whisk[1], whisk[0])
    d = (a1 - a0 + np.pi) % (2 * np.pi) - np.pi
    r = 40.0
    th = np.linspace(a0, a0 + d, 50)
    c = xy["whisker_base"]
    ax.plot(c[0] + r * np.cos(th), c[1] + r * np.sin(th), color="#ffd92f", lw=0.8, zorder=2)
    ax.plot([c[0], c[0] + 1.3 * r * np.cos(a0)], [c[1], c[1] + 1.3 * r * np.sin(a0)], color="white", lw=0.5,
            ls=(0, (2, 2)), zorder=2)
    mid = a0 + d / 2
    ax.text(c[0] + 1.6 * r * np.cos(mid), c[1] + 1.6 * r * np.sin(mid), f"{ang:.0f}°", color="#ffd92f",
            fontsize=6, ha="center", va="center")
    return ang


def example_panel(ax, frames, eb, colors, skeleton):
    """Side frame (a lick) above top frame (median-error frame; nose and whisker vectors, whisker angle); test
    frames, best model predictions of the view's keypoints with likelihood >= CUTOFF, no marker edge."""
    import matplotlib.image as mpimg
    imgs, offs, y0 = [], {}, 0
    for v in VIEWS:
        t = eb[(eb.view == v) & (eb.split == "test")]
        fr = pick_side_lick(t) if v == "side" else pick_median(t)
        img = mpimg.imread(PROJ / frames.loc[frames.frame == fr, "image"].iloc[0])
        img = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        imgs.append(img)
        offs[v] = (fr, y0)
        y0 += img.shape[0]
    ax.imshow(np.vstack(imgs), cmap="gray")
    angle = None
    for v, (fr, dy) in offs.items():
        p = eb[(eb.frame == fr) & (eb.likelihood >= CUTOFF)].set_index("bodypart")
        for a, b in skeleton:
            if a in p.index and b in p.index and v == "side":
                ax.plot([p.at[a, "x"], p.at[b, "x"]], [p.at[a, "y"] + dy, p.at[b, "y"] + dy], color="white", lw=0.6,
                         zorder=2)
        if v == "top":
            angle = whisker_angle_overlay(ax, p, dy)
        for bp, r in p.iterrows():
            ax.scatter(r.x, r.y + dy, s=14, color=colors[bp], edgecolors="none", zorder=3)
        ax.text(12, dy + imgs[0].shape[0] - 12, f"{v.capitalize()} view", color="white", fontsize=6, va="bottom",
                ha="left")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    out = {v: frames.loc[frames.frame == fr, "image"].iloc[0] for v, (fr, _) in offs.items()}
    out["top_whisker_angle_deg"] = None if angle is None else float(angle)
    return out


def keypoint_bars(ax, k, view, colors, human_k, mm_per_px):
    sub = k[k.view == view].set_index("bodypart").reindex(VIEW_KEYPOINTS[view])
    x = np.arange(len(sub))
    ax.bar(x, sub["mean"], color=[colors[b] for b in sub.index], width=0.75)
    for i, b in enumerate(sub.index):
        if (view, b) in human_k.index:
            ax.plot([i - 0.4, i + 0.4], [human_k[(view, b)]] * 2, color="0.2", lw=0.6, ls="--")
    ax.set_xticks(x, [b.replace("_", " ") for b in sub.index], rotation=60, ha="right")
    ax.set_title(f"{view.capitalize()} view", pad=2)
    mm_axis(ax, mm_per_px)


# --- derived quantities (NWB_converter utils/dlc_utils.compute_kinematics_alt, used for AB / MH sessions), per frame.
# Session references (median nose vector, closed-jaw position) are taken per labelled-data folder from the LABELS, the
# same for model, label and human relabel, so the errors are those of the frame's keypoints only.
def whisker_angle_alt(w, nose_ref):
    """compute_kinematics_alt top view: angle between (particle - whisker_base) and the nose vector (nose tip - nose
    base, folder median of the labels), with the converter's sign convention (whisker y difference base - particle)."""
    dot = (w.particle_x - w.whisker_base_x) * nose_ref[0] + (w.whisker_base_y - w.particle_y) * nose_ref[1]
    m1 = np.hypot(w.particle_x - w.whisker_base_x, w.particle_y - w.whisker_base_y)
    return np.degrees(np.arccos(np.clip(dot / (m1 * np.hypot(*nose_ref)), -1, 1)))


def jaw_angle_alt(w, ref_y, jaw_ref_x):
    """compute_kinematics_alt side view: arcsin((jaw_y - ref_y) / |(jaw_x - jaw_ref_x, jaw_y - ref_y)|), ref_y = 5th
    percentile of jaw_y (folder labels), jaw_ref_x = folder median of the jaw_ref labels."""
    dy = w.jaw_y - ref_y
    return np.degrees(np.arcsin(dy / np.hypot(dy, w.jaw_x - jaw_ref_x)))


def pupil_area4(w):
    """Shoelace area of the polygon top -> right -> bottom -> left (compute_kinematics_alt pupil_area), px^2."""
    q = ["top", "right", "bottom", "left"]
    a = 0
    for i in range(4):
        p1, p2 = q[i], q[(i + 1) % 4]
        a = a + w[f"pupil_{p1}_x"] * w[f"pupil_{p2}_y"] - w[f"pupil_{p1}_y"] * w[f"pupil_{p2}_x"]
    return 0.5 * np.abs(a)


def wide(long, xcol, ycol, keep=None):
    d = long if keep is None else long[keep]
    w = d.pivot_table(index="frame", columns="bodypart", values=[xcol, ycol], aggfunc="first")
    w.columns = [f"{b}_{'x' if c == xcol else 'y'}" for c, b in w.columns]
    return w


DERIVED = {  # name: (keypoints needed, view, reference key)
    "whisker_angle": (["particle", "whisker_base"], "top", "nose"),
    "jaw_angle": (["jaw"], "side", "jaw"),
    "pupil_area": (["pupil_top", "pupil_right", "pupil_bottom", "pupil_left"], "side", None),
}


def derived_value(name, w, r):
    if name == "whisker_angle":
        return whisker_angle_alt(w, r["nose"])
    if name == "jaw_angle":
        return jaw_angle_alt(w, *r["jaw"])
    return pupil_area4(w)


def derived_errors(eb, frames, gt_long, human):
    """Per test frame: derived quantity from the model (keypoints with likelihood >= CUTOFF), from the labels and
    (relabelled frames) from the human relabel; frames where every keypoint needed is labelled (and predicted)."""
    g = gt_long.merge(frames[["frame", "folder", "view", "split", "image"]], on="frame")
    refs = {}
    for folder, d in g.groupby("folder"):
        w = wide(d, "x_gt", "y_gt")
        r = {}
        if {"nose_tip_x", "nose_base_x"} <= set(w.columns):
            r["nose"] = (np.nanmedian(w.nose_tip_x - w.nose_base_x), np.nanmedian(w.nose_tip_y - w.nose_base_y))
        if {"jaw_y", "jaw_ref_x"} <= set(w.columns) and w.jaw_y.notna().any():
            r["jaw"] = (np.nanpercentile(w.jaw_y.dropna(), 5), np.nanmedian(w.jaw_ref_x))
        refs[folder] = r
    test = eb[eb.split == "test"]
    sources = {"label": wide(g[g.split == "test"], "x_gt", "y_gt"),
               "model": wide(test, "x", "y", keep=test.likelihood >= CUTOFF)}
    h = human.assign(image=human.source_image).merge(frames[["frame", "image"]], on="image")
    sources["human"] = wide(h, "x_relabel", "y_relabel")
    folder = frames.set_index("frame").folder
    rows = []
    for name, (kps, view, refkey) in DERIVED.items():
        cols = [f"{k}_{c}" for k in kps for c in "xy"]
        fr_view = frames[(frames.view == view) & (frames.split == "test")].frame
        for src, w in sources.items():
            if not set(cols) <= set(w.columns):
                continue
            w = w.reindex(fr_view)[cols].dropna()
            ok = [fr for fr in w.index if refkey is None or refkey in refs.get(folder[fr], {})]
            for fr in ok:
                v = derived_value(name, w.loc[[fr]], refs.get(folder[fr], {}))
                rows.append({"quantity": name, "frame": fr, "source": src, "value": float(np.asarray(v)[0])})
    return pd.DataFrame(rows).pivot_table(index=["quantity", "frame"], columns="source", values="value").reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps.set_nature_rc()
    cfg = yaml.safe_load((PROJ / "config.yaml").read_text())
    cmap = plt.get_cmap(cfg["colormap"], len(cfg["bodyparts"]))
    colors = {b: cmap(i) for i, b in enumerate(cfg["bodyparts"])}  # DLC colours follow the config order
    frames, gt_long = load_ground_truth()
    mm_view = frames.groupby("view").mm_per_px.median()
    human = human_reference()
    human_all, human_view = human.dist_px.mean(), human.groupby("view").dist_px.mean()
    human_view_mm = human.groupby("view").dist_mm.mean()

    # --- per-snapshot errors for every evaluated model (view keypoints only)
    rows, best = [], {}
    for s in MODELS:
        files = snapshot_files(s)
        if not files:
            continue
        per = {}
        for ep, f in files.items():
            e = errors(f, frames, gt_long)
            lab = e[e.labelled]
            for (split, view), g in pd.concat([lab.assign(view="all"), lab]).groupby(["split", "view"]):
                rows.append({"shuffle": s, "model": MODELS[s], "epoch": ep, "split": split, "view": view,
                             "mean_err_px": g.err_px.mean(), "median_err_px": g.err_px.median(),
                             "mean_err_mm": g.err_mm.mean(), "n_keypoints": len(g)})
            per[ep] = e
        t = pd.DataFrame([r for r in rows if r["shuffle"] == s and r["split"] == "test" and r["view"] == "all"])
        ep_best = int(t.loc[t.mean_err_px.idxmin(), "epoch"])
        best[s] = {"epoch_best": ep_best, "epoch_final": max(files), "e_best": per[ep_best]}
    if not best:
        raise SystemExit("no evaluated model yet")
    snap = pd.DataFrame(rows)
    snap.to_csv(OUT / "errors_per_snapshot.csv", index=False)
    test_all = snap[(snap.split == "test") & (snap.view == "all")]
    s_best = int(test_all.loc[test_all.mean_err_px.idxmin(), "shuffle"])
    eb = best[s_best]["e_best"]
    test_lab = eb[(eb.split == "test") & eb.labelled]

    fig = plt.figure(figsize=(183 * ps.MM, 280 * ps.MM))
    gs = fig.add_gridspec(6, 4, left=0.07, right=0.93, top=0.975, bottom=0.04, wspace=0.75, hspace=0.85,
                          height_ratios=[1.35, 1.35, 1, 1, 1, 1])

    # a: example predictions, side above top; larger, against the left page margin (user 2026-11-10)
    pos = gs[0:2, 0:2].get_position(fig)
    ax = fig.add_axes([0.015, pos.y0, pos.x1 - 0.015 - 0.07, pos.height])
    ax.set_anchor("NW")
    examples = example_panel(ax, frames, eb, colors, cfg.get("skeleton", []))
    panel_label(ax, "a", dx=-2)
    shown = [b for b in cfg["bodyparts"] if any(b in VIEW_KEYPOINTS[v] for v in VIEWS)]
    ax.apply_aspect()  # legend right of the drawn image (the axes box is wider than the image)
    img_pos = ax.get_position()
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=3, mec="none", color=colors[b],
                                  label=b.replace("_", " ")) for b in shown],
              loc="upper left", bbox_to_anchor=(img_pos.x1 + 0.005, img_pos.y1), bbox_transform=fig.transFigure,
              handletextpad=0.1, labelspacing=0.3)

    # right of a: 3 x 2 panels (user 2026-11-10: model comparison, e for all models and for the best model)
    right = gs[0:2, 2:4].subgridspec(3, 2, wspace=0.75, hspace=0.9)

    # b: dataset size
    ax = fig.add_subplot(right[0, 0]); panel_label(ax, "b")
    cnt = frames.groupby(["view", "split"]).size().unstack()[["train", "test"]]
    x = np.arange(len(cnt))
    ax.bar(x, cnt.train, color=[VIEW_COLORS[v] for v in cnt.index], width=0.6)
    ax.bar(x, cnt.test, bottom=cnt.train, color=[ps.light(VIEW_COLORS[v], 0.6) for v in cnt.index], width=0.6)
    for i, v in enumerate(cnt.index):
        ax.text(i, cnt.loc[v].sum() + 40, f"{cnt.loc[v].sum():,}\n{frames[frames.view == v].mouse.nunique()} mice",
                ha="center", va="bottom", fontsize=5)
    ax.set_xticks(x, [v.capitalize() for v in cnt.index]); ax.set_ylabel("Labelled frames")
    ax.set_ylim(0, cnt.sum(axis=1).max() * 1.3)
    ax.legend(handles=[Patch(color="0.3", label="train"), Patch(color="0.8", label="test")], loc="upper right",
              bbox_to_anchor=(1.3, 1.0), handlelength=0.8)

    # c: training loss; d: DLC test metric during training (all architectures)
    ax_c = fig.add_subplot(right[0, 1]); panel_label(ax_c, "c")
    ax_d = fig.add_subplot(right[1, 0]); panel_label(ax_d, "d")
    for s, name in MODELS.items():
        f = KUMA / f"learning_stats_shuffle{s}.csv"
        if f.exists():
            ls = pd.read_csv(f)
            ax_c.plot(ls.step, ls["losses/train.total_loss"].rolling(5, min_periods=1).mean(), color=MODEL_COLORS[s],
                      label=name)
            m = ls.dropna(subset=["metrics/test.rmse"])
            ax_d.plot(m.step, m["metrics/test.rmse"], color=MODEL_COLORS[s])
    ax_c.set_yscale("log"); ax_c.set_xlabel("Epoch"); ax_c.set_ylabel("Training loss")
    ax_c.legend(loc="upper right", handlelength=1, labelspacing=0.2)
    ax_d.set_xlabel("Epoch"); ax_d.set_ylabel("DLC test error (px)\n(all labelled keypoints)")
    ax_d.set_ylim(0, 20)

    # e: model comparison: test error of each evaluated model at its best snapshot, per view (mm), human per view
    ax = fig.add_subplot(right[1, 1]); panel_label(ax, "e")
    order = sorted(best, key=lambda s: float(test_all[(test_all.shuffle == s)
                                                      & (test_all.epoch == best[s]["epoch_best"])].mean_err_mm.iloc[0]))
    w = 0.38
    for j, v in enumerate(VIEWS):
        vals = [float(snap[(snap.shuffle == s) & (snap.split == "test") & (snap.view == v)
                           & (snap.epoch == best[s]["epoch_best"])].mean_err_mm.iloc[0]) for s in order]
        ax.bar(np.arange(len(order)) + (j - 0.5) * w, vals, width=w, color=[MODEL_COLORS[s] for s in order],
               edgecolor=VIEW_COLORS[v], linewidth=0.6, hatch="////" if v == "top" else None)
        ax.axhline(human_view_mm[v], color=VIEW_COLORS[v], lw=0.5, ls="--")
    ax.set_xticks(np.arange(len(order)), [MODELS[s] for s in order], rotation=50, ha="right", fontsize=5)
    ax.set_ylabel("Test error (mm)\n(best snapshot)"); ax.set_ylim(0, max(human_view_mm) * 1.5)
    ax.legend(handles=[Patch(facecolor="w", edgecolor=VIEW_COLORS["side"], label="side"),
                       Patch(facecolor="w", edgecolor=VIEW_COLORS["top"], hatch="////", label="top"),
                       plt.Line2D([], [], color="0.4", ls="--", lw=0.5, label="human")],
              loc="upper left", fontsize=4.5, handlelength=1.2, ncol=3, columnspacing=0.6)

    # f: train vs test error during training, every evaluated model (view keypoints), human, best snapshots
    ax = fig.add_subplot(right[2, 0]); panel_label(ax, "f")
    for s in best:
        for split, ls_ in (("train", ":"), ("test", "-")):
            t = snap[(snap.shuffle == s) & (snap.split == split) & (snap.view == "all")]
            ax.plot(t.epoch, t.mean_err_px, color=MODEL_COLORS[s], ls=ls_)
        tb = test_all[(test_all.shuffle == s) & (test_all.epoch == best[s]["epoch_best"])]
        ax.scatter(tb.epoch, tb.mean_err_px, s=10, marker="v", color=MODEL_COLORS[s], edgecolors="none", zorder=3)
    human_line(ax, human_all)
    ax.plot([], [], color="0.3", ls="-", label="test"); ax.plot([], [], color="0.3", ls=":", label="train")
    ax.legend(loc="center right", handlelength=1.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Mean error (px)\n(view keypoints)"); ax.set_ylim(0, human_all * 1.25)
    ax.set_title("All models", pad=2, fontsize=6)

    # g: the same for the best model, with its side and top test errors
    ax = fig.add_subplot(right[2, 1], sharey=ax); panel_label(ax, "g")
    for split, ls_ in (("train", ":"), ("test", "-")):
        t = snap[(snap.shuffle == s_best) & (snap.split == split) & (snap.view == "all")]
        ax.plot(t.epoch, t.mean_err_px, color=MODEL_COLORS[s_best], ls=ls_, label=f"{split}, all")
    for v in VIEWS:
        t = snap[(snap.shuffle == s_best) & (snap.split == "test") & (snap.view == v)]
        ax.plot(t.epoch, t.mean_err_px, color=VIEW_COLORS[v], lw=0.7, label=f"test, {v}")
    tb = test_all[(test_all.shuffle == s_best) & (test_all.epoch == best[s_best]["epoch_best"])]
    ax.scatter(tb.epoch, tb.mean_err_px, s=12, marker="v", color=MODEL_COLORS[s_best], edgecolors="none", zorder=3)
    human_line(ax, human_all)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.78), fontsize=4.5, handlelength=1.2)
    ax.set_xlabel("Epoch"); ax.set_title(f"Best: {MODELS[s_best]} (epoch {best[s_best]['epoch_best']})", pad=2,
                                         fontsize=6)

    # h: per-frame test error histograms per view (mm, px on top), model mean and human mean
    for j, v in enumerate(VIEWS):
        ax = fig.add_subplot(gs[2, j]); panel_label(ax, "h" if j == 0 else "")
        fe = test_lab[test_lab.view == v].groupby("frame").err_mm.mean()
        bins = np.linspace(0, 1.5, 31)
        ax.hist(fe.clip(upper=bins[-1]), bins=bins, color=VIEW_COLORS[v], edgecolor="none")
        ax.axvline(fe.mean(), color="k", lw=0.8)
        ax.axvline(human_view_mm[v], color="0.35", lw=0.6, ls="--")
        ax.set_xlabel("Test error per frame (mm)"); ax.set_ylabel("Frames" if j == 0 else "")
        sec = ax.secondary_xaxis("top", functions=(lambda m, k=mm_view[v]: m / k, lambda p, k=mm_view[v]: p * k))
        sec.set_xlabel(f"{v.capitalize()} view, n = {len(fe)} frames (px)", labelpad=2)
        ax.text(1.0, 0.55, f"> {bins[-1]} mm\nin last bin", transform=ax.transAxes, ha="right", fontsize=4.5,
                color="0.4")
    ax.plot([], [], color="k", lw=0.8, label="model mean"); ax.plot([], [], color="0.35", lw=0.6, ls="--",
                                                                     label="human mean")
    ax.legend(loc="upper right")

    # i: PCK (fraction of test keypoints within a distance of the label), model vs human, per view
    ax = fig.add_subplot(gs[2, 2]); panel_label(ax, "i")
    thr = np.linspace(0, 1.0, 201)
    for v in VIEWS:
        d = test_lab[test_lab.view == v].err_mm.to_numpy()
        ax.plot(thr, [(d <= t).mean() for t in thr], color=VIEW_COLORS[v], label=f"{v}, model")
        hd = human[human.view == v].dist_mm.to_numpy()
        ax.plot(thr, [(hd <= t).mean() for t in thr], color=VIEW_COLORS[v], ls="--", label=f"{v}, human")
    ax.set_xlabel("Distance to label (mm)"); ax.set_ylabel("Fraction of keypoints within")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower right", handlelength=1.2)

    # j: error vs likelihood (test, labelled)
    ax = fig.add_subplot(gs[2, 3]); panel_label(ax, "j")
    bins = np.linspace(0, 1, 11)
    for v in VIEWS:
        d = test_lab[test_lab.view == v]
        m = d.groupby(pd.cut(d.likelihood, bins, include_lowest=True), observed=False).err_px.median()
        ax.plot((bins[:-1] + bins[1:]) / 2, m.to_numpy(), color=VIEW_COLORS[v], marker="o", ms=1.5, mec="none",
                label=v)
    ax.axvline(CUTOFF, color="0.5", lw=0.5, ls=":")
    ax.text(CUTOFF, 1.0, " cutoff", transform=ax.get_xaxis_transform(), fontsize=5, color="0.5", va="top")
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_xlim(0, 1)
    ax.set_yscale("log"); ax.set_xlabel("Likelihood"); ax.set_ylabel("Median test error (px)")
    ax.legend(loc="upper right")

    # k: per keypoint test error per view (px, mm on the right), human per keypoint
    human_k = human.groupby(["view", "bodypart"]).dist_px.mean()
    k = test_lab.groupby(["view", "bodypart"]).agg(mean=("err_px", "mean"), median=("err_px", "median"),
                                                   mean_mm=("err_mm", "mean"), n=("err_px", "size")).reset_index()
    k.to_csv(OUT / "errors_best_per_keypoint.csv", index=False)
    sub = gs[3, 0:2].subgridspec(1, 2, width_ratios=[len(VIEW_KEYPOINTS["side"]), len(VIEW_KEYPOINTS["top"])],
                                 wspace=0.45)
    for j, v in enumerate(VIEWS):
        ax = fig.add_subplot(sub[0, j])
        keypoint_bars(ax, k, v, colors, human_k, mm_view[v])
        if j == 0:
            ax.set_ylabel("Mean test error (px)"); panel_label(ax, "k")

    # l: presence: detected (labelled, likelihood >= cutoff) and confident when not labelled (absent / occluded)
    ax = fig.add_subplot(gs[3, 2:]); panel_label(ax, "l")
    tst = eb[eb.split == "test"]
    pres = tst.groupby(["view", "bodypart"]).apply(lambda g: pd.Series({
        "n_labelled": int(g.labelled.sum()), "detected": (g[g.labelled].likelihood >= CUTOFF).mean(),
        "n_not_labelled": int((~g.labelled).sum()),
        "confident_not_labelled": (g[~g.labelled].likelihood >= CUTOFF).mean()}), include_groups=False).reset_index()
    pres.to_csv(OUT / "presence.csv", index=False)
    xt, xl, x0 = [], [], 0
    for v in VIEWS:
        p = pres[pres.view == v].set_index("bodypart").reindex(VIEW_KEYPOINTS[v])
        for i, (b, r) in enumerate(p.iterrows()):
            xx = x0 + i
            ax.bar(xx - 0.2, r.detected, width=0.4, color=colors[b])
            if r.n_not_labelled >= 10:
                ax.bar(xx + 0.2, r.confident_not_labelled, width=0.4, color=ps.light(colors[b], 0.55))
                ax.text(xx + 0.2, r.confident_not_labelled + 0.02, f"{int(r.n_not_labelled)}", ha="center",
                        fontsize=4, rotation=90, va="bottom")
            xt.append(xx); xl.append(b.replace("_", " "))
        ax.text(x0 + (len(p) - 1) / 2, 1.08, f"{v.capitalize()} view", ha="center", fontsize=6,
                transform=ax.get_xaxis_transform())
        x0 += len(p) + 1
    ax.set_xticks(xt, xl, rotation=60, ha="right"); ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"Fraction with likelihood >= {CUTOFF}")
    ax.legend(handles=[Patch(color="0.3", label="labelled (visible)"),
                       Patch(color="0.8", label="not labelled (absent / occluded); n on top")],
              loc="upper left", bbox_to_anchor=(0.0, 1.32), ncol=2)

    # m-p: derived quantities (as in the NWB, compute_kinematics_alt): model vs label, model and human errors
    der = derived_errors(eb, frames, gt_long, human)
    der.to_csv(OUT / "derived_quantities_test.csv", index=False)
    wa = der[der.quantity == "whisker_angle"].dropna(subset=["label", "model"])
    ax = fig.add_subplot(gs[4, 0]); panel_label(ax, "m")
    ax.scatter(wa.label, wa.model, s=2, color=VIEW_COLORS["top"], edgecolors="none", rasterized=True)
    lim = [np.nanmin(wa[["label", "model"]].to_numpy()) - 5, np.nanmax(wa[["label", "model"]].to_numpy()) + 5]
    ax.plot(lim, lim, color="0.5", lw=0.5, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Whisker angle, labels (deg)"); ax.set_ylabel("Whisker angle, model (deg)")
    ax.set_title(f"r = {np.corrcoef(wa.label, wa.model)[0, 1]:.3f}, n = {len(wa)} test frames", pad=2)
    panels = {"whisker_angle": ("Whisker angle error (deg)", 15), "jaw_angle": ("Jaw angle error (deg)", 15),
              "pupil_area": ("Pupil area error (% of label)", 50)}
    derived_summary = {}
    for j, (q, (xlabel, xmax)) in enumerate(panels.items()):
        ax = fig.add_subplot(gs[4, j + 1]); panel_label(ax, "nop"[j])
        d = der[der.quantity == q]
        out = {}
        for src, ls_ in (("model", "-"), ("human", "--")):
            if src not in d:
                continue
            dd = d.dropna(subset=["label", src])
            diff = dd[src] - dd.label
            err = (100 * diff / dd.label).abs() if q == "pupil_area" else diff.abs()
            xs = np.linspace(0, xmax, 200)
            ax.plot(xs, [(err <= x).mean() for x in xs], color="k" if src == "model" else "0.5", ls=ls_,
                    label=f"{src}: {err.median():.1f}")
            out[src] = {"n": int(len(err)), "median": float(err.median()), "mean": float(err.mean()),
                        "p90": float(err.quantile(0.9))}
        derived_summary[q] = out
        ax.set_title(f"Median error; n = {out.get('model', {}).get('n', 0)} / {out.get('human', {}).get('n', 0)} "
                     "frames", pad=2, fontsize=5)
        ax.set_xlabel(xlabel); ax.set_ylabel("Fraction of test frames within" if j == 0 else ""); ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=4.5, handlelength=1.2)

    # q: per-mouse test error (best model)
    ax = fig.add_subplot(gs[5, :]); panel_label(ax, "q")
    pm = test_lab.groupby(["mouse", "view"]).agg(err_px=("err_px", "mean"), err_mm=("err_mm", "mean"),
                                                 n=("err_px", "size")).reset_index()
    pm.to_csv(OUT / "errors_best_per_mouse.csv", index=False)
    order = pm.groupby("mouse").err_mm.mean().sort_values().index
    xi = {m: i for i, m in enumerate(order)}
    for v in VIEWS:
        d = pm[pm.view == v]
        ax.scatter(d.mouse.map(xi), d.err_mm, s=4, color=VIEW_COLORS[v], edgecolors="none", label=v)
    ax.axhline(human.dist_mm.mean(), color="0.35", lw=0.6, ls="--", zorder=0)
    ax.text(1.0, human.dist_mm.mean(), " human", transform=ax.get_yaxis_transform(), fontsize=5, color="0.35",
            va="center")
    high = pm[pm.err_mm > human.dist_mm.mean()].sort_values("err_mm")
    gap = 0.07 * high.err_mm.max()  # labels spread vertically (min gap) so that close points do not overlap
    y_lab = []
    for e_ in high.err_mm:
        y_lab.append(max(e_, y_lab[-1] + gap) if y_lab else e_)
    x_lab = min(xi[m] for m in high.mouse) - 6
    for (_, r), y in zip(high.iterrows(), y_lab):
        ax.annotate(r.mouse, (xi[r.mouse], r.err_mm), xytext=(x_lab, y), textcoords="data", fontsize=4, ha="right",
                    va="center", color=VIEW_COLORS[r.view], arrowprops=dict(arrowstyle="-", lw=0.3, color="0.6"))
    ax.set_xticks([]); ax.set_xlabel(f"Mouse (n = {len(order)}, sorted by mean test error)")
    ax.set_ylabel("Mean test error (mm)"); ax.set_ylim(bottom=0); ax.legend(loc="upper left")

    ps.save_figure(fig, OUT, "training_summary", raster_dpi=600)

    summary = {
        "best_model": MODELS[s_best], "best_shuffle": s_best, "cutoff": CUTOFF, "view_keypoints": VIEW_KEYPOINTS,
        "evaluated_models": {MODELS[s]: {"epoch_best": b["epoch_best"], "epoch_final": b["epoch_final"]}
                             for s, b in best.items()},
        "test_error_best_px": {v: float(test_lab[test_lab.view == v].err_px.mean()) for v in VIEWS}
        | {"all": float(test_lab.err_px.mean())},
        "test_error_best_mm": {v: float(test_lab[test_lab.view == v].err_mm.mean()) for v in VIEWS}
        | {"all": float(test_lab.err_mm.mean())},
        "human_px": {"all": float(human_all)} | {v: float(x) for v, x in human_view.items()},
        "human_mm": {"all": float(human.dist_mm.mean())} | {v: float(x) for v, x in human_view_mm.items()},
        "n_frames": frames.groupby(["view", "split"]).size().unstack().to_dict(),
        "n_mice": int(frames.mouse.nunique()), "example_frames": examples, "derived_errors": derived_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
