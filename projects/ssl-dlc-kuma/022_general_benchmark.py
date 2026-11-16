"""General benchmark: old DLC shuffles 3 / 5 / 6 vs the merged model(s), plus example frames (user 2026-11-09).

Models: old s3, s5, s6 (one fine-tuning chain, 007 predictions), "Old (in use)" = the shuffle whose filtered h5s are
in the NWBs for that mouse (s3 for AB mice, s5 for MH mice, s6 for MH062-070; ssl-dlc-kuma inventory), and every
evaluated merged architecture (best snapshot).
Frame sets: "held-out" = merged TEST frames that no old shuffle trained on (108 frames, 29 mice; gt.csv train_s*),
"unseen mice" = the subset from the 13 mice never seen by the old chain (58 frames; 021). Same rules as 019 / 021:
merged labels for all models, keypoints of the frame's view, spout and top tongue excluded, likelihood cutoff 0.5.
Figure 1 (general_benchmark): a mean error per view (held-out, mouse-bootstrap 95% CI; unseen-mice subset as open
markers), b PCK (mm) per view, c gross errors (> 0.5 mm), d detection of labelled keypoints (likelihood >= 0.5),
e confident predictions of unlabelled (occluded / absent) keypoints, f median error vs likelihood, g error by cohort,
h derived quantities (median error), i mean error per keypoint (models x keypoints).
Figure 2 (example_frames): held-out frames with the largest gain of the merged model over the old model in use;
labels (white crosses), predictions (dots, DLC colours) and error lines, old in use vs merged.
Outputs: DLCProjects/merged_training_2026-11-09/general_benchmark/
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 022_general_benchmark.py
"""

import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bm = _load("bm", "021_unseen_mice_benchmark.py")
ts, ps = bm.ts, bm.ps
OUT = ts.SIDE / "general_benchmark"
CUT = ts.CUTOFF
IN_USE_COLOR = "#d6604d"


INVENTORY = Path("/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/ssl-dlc-kuma/inventory.csv")


def in_use_shuffle(mouse):
    """Fallback per mouse (frames from sessions without a filtered h5)."""
    if mouse.startswith("AB"):
        return 3
    return 6 if mouse in ("MH062", "MH064", "MH065", "MH067", "MH068", "MH069", "MH070") else 5


def in_use_per_frame(frames):
    """Shuffle of the filtered h5 in use for the frame's session and view (inventory: mouse + recording date + view;
    the folder name holds the date as yy-mm-dd), else the mouse fallback. Shuffle-6 h5s were made with best-720 (gone):
    s6 here is best-820."""
    inv = pd.read_csv(INVENTORY).dropna(subset=["dlc_shuffles"])
    inv["date"] = inv.session_id.str.split("_").str[1]
    lut = inv.groupby(["mouse_id", "date", "view"]).dlc_shuffles.first().astype(int).to_dict()
    date = "20" + frames.folder.str.extract(r"(\d\d)-(\d\d)-(\d\d)").fillna("").agg("".join, axis=1)
    out = [lut.get((m, d, v), in_use_shuffle(m)) for m, d, v in zip(frames.mouse, date, frames.view)]
    return pd.Series(out, index=frames.frame.to_numpy())


def build_models(frames, gt_long):
    old = {s: bm.model_errors(bm.old_long(s, frames), frames, gt_long) for s in bm.OLD}
    models = {bm.OLD[s]: (e, bm.OLD_COLORS[bm.OLD[s]]) for s, e in old.items()}
    use_s = in_use_per_frame(frames)
    use = pd.concat([old[s][old[s].frame.map(use_s) == s] for s in old])
    use["in_use_shuffle"] = use.frame.map(use_s)
    models["Old (in use)"] = (use, IN_USE_COLOR)
    for name, (e, color, _) in bm.merged_models(frames, gt_long).items():
        models[name] = (e, color)
    return models


def frame_sets(frames):
    gt = pd.read_csv(bm.BENCH / "gt.csv")
    f = frames.assign(img=frames.image.str.split("/").str[-1]).merge(
        gt[["folder", "image", "train_s3", "train_s5", "train_s6", "cohort"]].rename(columns={"image": "img"}),
        on=["folder", "img"], how="left")
    trained = f[["train_s3", "train_s5", "train_s6"]].fillna(True).astype(bool).any(axis=1)
    held = f[(f.split == "test") & ~trained]
    frames = frames.merge(f[["frame", "cohort"]], on="frame", how="left")
    return frames, {"held-out": set(held.frame), "unseen mice": set(held[held.mouse.isin(bm.UNSEEN)].frame)}


def grouped_bars(ax, values, names, colors, groups, ylabel, log=False):
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        ax.bar(np.arange(len(groups)) + (i - (len(names) - 1) / 2) * w, [values.get((g, n), np.nan) for g in groups],
               width=w, color=colors[n], label=n)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")


def general_figure(models, frames, gt_long, sets):
    names = list(models)
    colors = {n: c for n, (_, c) in models.items()}
    held = sets["held-out"]
    lab = {n: e[e.labelled & e.frame.isin(held)] for n, (e, _) in models.items()}
    allk = {n: e[e.frame.isin(held)] for n, (e, _) in models.items()}
    rows = []
    fig = plt.figure(figsize=(183 * ps.MM, 200 * ps.MM))
    gs = fig.add_gridspec(3, 4, left=0.07, right=0.97, top=0.93, bottom=0.07, wspace=0.65, hspace=0.75)

    # a: mean error per view, held-out (bars, CI by mice) + unseen-mice subset (open circles)
    ax = fig.add_subplot(gs[0, 0]); ts.panel_label(ax, "a")
    w = 0.8 / len(names)
    for j, view in enumerate(ts.VIEWS):
        for i, n in enumerate(names):
            d = lab[n][lab[n].view == view]
            x = j + (i - (len(names) - 1) / 2) * w
            lo, hi = bm.boot_ci(d.err_mm.to_numpy(), d.mouse.to_numpy())
            ax.bar(x, d.err_mm.mean(), width=w, color=colors[n])
            ax.errorbar(x, d.err_mm.mean(), yerr=[[d.err_mm.mean() - lo], [hi - d.err_mm.mean()]], fmt="none",
                        ecolor="k", elinewidth=0.5, capsize=1)
            u = d[d.frame.isin(sets["unseen mice"])]
            ax.scatter(x, u.err_mm.mean(), s=6, facecolors="none", edgecolors="k", linewidths=0.5, zorder=3)
            rows.append({"panel": "a", "model": n, "view": view, "mean_err_mm": d.err_mm.mean(), "ci_lo": lo,
                         "ci_hi": hi, "mean_err_px": d.err_px.mean(), "unseen_mice_mean_err_mm": u.err_mm.mean(),
                         "n_frames": d.frame.nunique(), "n_mice": d.mouse.nunique()})
    ax.set_xticks([0, 1], ["Side", "Top"]); ax.set_ylabel("Mean error (mm)")
    ax.set_title(f"{len(held)} held-out frames; o = unseen mice", pad=2, fontsize=5)
    ax.legend(handles=[plt.Line2D([], [], color=colors[n], lw=3, label=n.replace("Merged ", "")) for n in names],
              loc="upper left", fontsize=4.5, handlelength=1)

    # b: PCK in mm per view
    thr = np.linspace(0, 1.0, 101)
    for j, view in enumerate(ts.VIEWS):
        ax = fig.add_subplot(gs[0, 1 + j]); ts.panel_label(ax, "b" if j == 0 else "")
        for n in names:
            d = lab[n][lab[n].view == view].err_mm.to_numpy()
            ax.plot(thr, [(d <= t).mean() for t in thr], color=colors[n], lw=0.8)
        ax.set_xlabel("Distance to label (mm)"); ax.set_ylabel("Fraction of keypoints within" if j == 0 else "")
        ax.set_title(f"{view.capitalize()} view", pad=2); ax.set_ylim(0, 1.02)

    # c: gross errors (> 0.5 mm, about the human disagreement)
    ax = fig.add_subplot(gs[0, 3]); ts.panel_label(ax, "c")
    vals = {(v.capitalize(), n): 100 * (lab[n][lab[n].view == v].err_mm > 0.5).mean() for n in names for v in ts.VIEWS}
    grouped_bars(ax, vals, names, colors, ["Side", "Top"], "% keypoints with error > 0.5 mm")

    # d: detection of labelled keypoints; e: confident when not labelled
    ax = fig.add_subplot(gs[1, 0]); ts.panel_label(ax, "d")
    vals = {(v.capitalize(), n): 100 * (lab[n][lab[n].view == v].likelihood >= CUT).mean() for n in names
            for v in ts.VIEWS}
    grouped_bars(ax, vals, names, colors, ["Side", "Top"], f"% labelled keypoints\nwith likelihood >= {CUT}")
    ax.set_ylim(50, 101)
    ax = fig.add_subplot(gs[1, 1]); ts.panel_label(ax, "e")
    groups = {"Pupil": ["pupil_top", "pupil_left", "pupil_right", "pupil_bottom"], "Tongue": ["tongue"]}
    vals = {}
    for n in names:
        d = allk[n][~allk[n].labelled & (allk[n].view == "side")]
        for g, kps in groups.items():
            dd = d[d.bodypart.isin(kps)]
            vals[(f"{g}\n(n = {len(dd)})", n)] = 100 * (dd.likelihood >= CUT).mean()
    grouped_bars(ax, vals, names, colors, sorted({k[0] for k in vals}), f"% unlabelled (hidden)\nwith likelihood >= {CUT}")
    ax.set_title("Side view, occluded keypoints", pad=2, fontsize=5)

    # f: median error vs likelihood (both views)
    ax = fig.add_subplot(gs[1, 2]); ts.panel_label(ax, "f")
    bins = np.linspace(0, 1, 11)
    for n in names:
        d = lab[n]
        m = d.groupby(pd.cut(d.likelihood, bins, include_lowest=True), observed=False).err_mm.median()
        cnt = d.groupby(pd.cut(d.likelihood, bins, include_lowest=True), observed=False).size()
        ax.plot(((bins[:-1] + bins[1:]) / 2)[cnt >= 5], m[cnt >= 5], color=colors[n], marker="o", ms=1.2, lw=0.7)
    ax.axvline(CUT, color="0.5", lw=0.5, ls=":")
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yscale("log")
    ax.set_xlabel("Likelihood"); ax.set_ylabel("Median error (mm)")

    # g: error by cohort
    ax = fig.add_subplot(gs[1, 3]); ts.panel_label(ax, "g")
    vals = {}
    for n in names:
        d = lab[n].merge(frames[["frame", "cohort"]], on="frame")
        for c, dc in d.groupby("cohort"):
            vals[(f"{c} (n = {dc.frame.nunique()})", n)] = dc.err_mm.mean()
    grouped_bars(ax, vals, names, colors, sorted({k[0] for k in vals}), "Mean error (mm)", log=True)

    # h: derived quantities on the held-out frames
    ax = fig.add_subplot(gs[2, 0]); ts.panel_label(ax, "h")
    no_human = ts.human_reference().iloc[0:0]
    fr_h = frames.assign(split=np.where(frames.frame.isin(held), "test", "other"))
    vals, labels = {}, {"whisker_angle": "Whisker\n(deg)", "jaw_angle": "Jaw\n(deg)", "pupil_area": "Pupil\n(%)"}
    for n, (e, _) in models.items():
        d = ts.derived_errors(e.assign(split=np.where(e.frame.isin(held), "test", "other")), fr_h, gt_long, no_human)
        d = d.dropna(subset=["label", "model"])
        diff = d["model"] - d["label"]
        d["abs_err"] = np.where(d.quantity == "pupil_area", (100 * diff / d["label"]).abs(), diff.abs())
        for q, dq in d.groupby("quantity"):
            vals[(labels[q], n)] = dq.abs_err.median()
            rows.append({"panel": "h", "model": n, "quantity": q, "median_abs_err": dq.abs_err.median(),
                         "n_frames": len(dq)})
    grouped_bars(ax, vals, names, colors, list(labels.values()), "Median error", log=True)

    # i: models x keypoints mean error (mm)
    ax = fig.add_subplot(gs[2, 1:]); ts.panel_label(ax, "i")
    cols = [(v, b) for v in ts.VIEWS for b in ts.VIEW_KEYPOINTS[v]]
    mat = np.array([[lab[n][(lab[n].view == v) & (lab[n].bodypart == b)].err_mm.mean() for v, b in cols]
                    for n in names])
    im = ax.imshow(np.log10(mat), cmap="magma_r", aspect="auto", vmin=np.log10(0.05), vmax=np.log10(3))
    for r in range(len(names)):
        for c in range(len(cols)):
            ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", fontsize=3.5,
                    color="white" if mat[r, c] > 0.6 else "k")
    ax.set_yticks(range(len(names)), [n.replace("Merged ", "Merged\n") for n in names], fontsize=5)
    ax.set_xticks(range(len(cols)), [f"{b.replace('_', ' ')}" for v, b in cols], rotation=60, ha="right")
    ax.axvline(len(ts.VIEW_KEYPOINTS["side"]) - 0.5, color="w", lw=1.5)
    ax.text((len(ts.VIEW_KEYPOINTS["side"]) - 1) / 2, -0.9, "Side view", ha="center", fontsize=6)
    ax.text(len(ts.VIEW_KEYPOINTS["side"]) + (len(ts.VIEW_KEYPOINTS["top"]) - 1) / 2, -0.9, "Top view", ha="center",
            fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2]
    cb.set_ticks(np.log10(ticks)); cb.set_ticklabels([str(t) for t in ticks]); cb.set_label("Mean error (mm)")
    pd.DataFrame(mat, index=names, columns=[f"{v}:{b}" for v, b in cols]).to_csv(OUT / "per_keypoint_mm.csv")
    ps.save_figure(fig, OUT, "general_benchmark", raster_dpi=600)
    return pd.DataFrame(rows)


def example_frames(models, frames, gt_long, sets, colors, n_per_view=4):
    """Held-out frames with the largest gain of the best merged model over the old model in use (one per mouse)."""
    merged = [n for n in models if n.startswith("Merged")]
    errs = {n: models[n][0] for n in ["Old (in use)"] + merged}
    best = min(merged, key=lambda n: errs[n][errs[n].labelled & errs[n].frame.isin(sets["held-out"])].err_px.mean())
    fe = {n: errs[n][errs[n].labelled & errs[n].frame.isin(sets["held-out"])].groupby("frame").err_px.mean()
          for n in ("Old (in use)", best)}
    gain = (fe["Old (in use)"] - fe[best]).dropna().rename("gain").to_frame().join(frames.set_index("frame")[
        ["view", "mouse", "image"]])
    pick = []
    for view in ts.VIEWS:
        g = gain[gain.view == view].sort_values("gain", ascending=False)
        pick += list(g.drop_duplicates("mouse").head(n_per_view).index)
    fig, axes = plt.subplots(len(pick), 2, figsize=(120 * ps.MM, 46 * ps.MM * len(pick)))
    for r, fr in enumerate(pick):
        info = gain.loc[fr]
        img = mpimg.imread(ts.PROJ / info.image)
        img = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        kps = ts.VIEW_KEYPOINTS[info.view]
        gl = gt_long[(gt_long.frame == fr) & gt_long.bodypart.isin(kps) & gt_long.x_gt.notna()]
        for c, n in enumerate(("Old (in use)", best)):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray")
            p = errs[n][(errs[n].frame == fr) & errs[n].bodypart.isin(kps)].set_index("bodypart")
            for _, g in gl.iterrows():
                ax.scatter(g.x_gt, g.y_gt, marker="x", s=12, lw=0.6, color="white", zorder=3)
                if g.bodypart in p.index:
                    q = p.loc[g.bodypart]
                    ax.plot([g.x_gt, q.x], [g.y_gt, q.y], color="#ffd92f", lw=0.6, zorder=2)
            for b, q in p.iterrows():
                ax.scatter(q.x, q.y, s=10, color=colors[b], edgecolors="none", zorder=4,
                           alpha=1.0 if q.likelihood >= CUT else 0.35)
            ax.set_xticks([]); ax.set_yticks([])
            err = fe[n].get(fr, np.nan)
            s_use = int(errs["Old (in use)"].loc[errs["Old (in use)"].frame == fr, "in_use_shuffle"].iloc[0])
            title = f"Old in use (s{s_use})" if c == 0 else best
            ax.set_title(f"{title}: {err:.1f} px", fontsize=5.5, pad=1.5)
            if c == 0:
                ax.set_ylabel(f"{info.mouse} {info.view}\n{'unseen mouse' if fr in sets['unseen mice'] else ''}",
                              fontsize=5.5)
    fig.text(0.5, 0.002, "White x = label; dot = prediction (faint if likelihood < 0.5); yellow line = error",
             ha="center", fontsize=5.5)
    fig.tight_layout(rect=(0, 0.01, 1, 1))
    ps.save_figure(fig, OUT, "example_frames", raster_dpi=300)
    gain.loc[pick].to_csv(OUT / "example_frames.csv")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ps.set_nature_rc()
    frames, gt_long = ts.load_ground_truth()
    frames, sets = frame_sets(frames)
    models = build_models(frames, gt_long)
    tab = general_figure(models, frames, gt_long, sets)
    tab.to_csv(OUT / "summary.csv", index=False)
    import yaml
    cfg = yaml.safe_load((ts.PROJ / "config.yaml").read_text())
    cmap = plt.get_cmap(cfg["colormap"], len(cfg["bodyparts"]))
    example_frames(models, frames, gt_long, sets, {b: cmap(i) for i, b in enumerate(cfg["bodyparts"])})
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
