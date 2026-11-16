"""Our merged DLC model vs other users' models (user 2026-11-11: "compare the performance of my best model with that of
models trained by other users"; labelling conventions are the same across users).

Models: ours = CSPNeXt-M shuffle 6 snapshot-330 (one model, both views); Pol Bech's ResNet-50 context models (side /
top, pytorch, shuffle 1, best snapshot; trained on PB177-PB201 and Robin Dard's RD039-RD057). Predictions of 038 (Kuma).
Datasets: our labelled frames (split of 018: train / test) and Pol's (his shuffle-1 split, 95% train). Each model is
scored on the frames of the other dataset too: all of Pol's frames are unseen by our model and vice versa; the fair
cells are "test" frames for the model that trained on that dataset, all frames for the other one (both reported).
Keypoints: the view's keypoints present in both projects (Pol's side 'nose' = our 'nose_tip'; spout excluded).
Error = Euclidean distance prediction - label, all predictions whatever their likelihood (as in 019); mm with our
view median mm per px (same rig and camera; image sizes are checked and printed).

Outputs: DLCProjects/merged_training_2026-11-09/other_users/ (pred / gt h5 copied there from Kuma other_users/ with rclone):
  errors_per_keypoint.csv, errors_summary.csv, other_users_comparison.{png,pdf}
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 039_other_users_comparison.py
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ssl-paper-figures"))
import paperstyle as ps  # noqa: E402

SIDE = Path("/mnt/lsens-analysis/Axel_Bisi/DLCProjects/merged_training_2026-11-09")
OUT = SIDE / "other_users"
KEYPOINTS = {"side": ["pupil_top", "pupil_left", "pupil_right", "pupil_bottom", "nose_tip", "nose_base", "jaw",
                      "jaw_ref", "tongue", "particle"],
             "top": ["nose_tip", "nose_base", "particle", "whisker_base", "whisker_tip"]}
POL_RENAME = {"nose": "nose_tip"}
MODEL_LABEL = {"ours": "Ours (CSPNeXt-M)", "pol": "Pol (ResNet-50)"}
MODEL_COLOR = {"ours": "#2ca02c", "pol": "#7f7f7f"}
DATA_LABEL = {"ours": "Our frames", "pol": "Pol's frames"}


def long(df, scorer_level=True):
    """Wide DLC frame (index = image path parts) -> long table frame, bodypart, x, y[, likelihood]."""
    if scorer_level and df.columns.nlevels == 3:
        df = df.droplevel(0, axis=1)
    df = df.rename(columns=POL_RENAME, level=0)
    df.index = pd.Index(["/".join(map(str, i)) if isinstance(i, tuple) else str(i) for i in df.index], name="frame")
    df.columns = df.columns.set_names(["bodypart", "coords"])
    return df.stack(level="bodypart", future_stack=True).reset_index()


def ground_truth():
    gt = {}
    ours = pd.read_hdf(SIDE / "kuma/split_ground_truth.h5")
    split = ours.pop(("split", "", "")) if ("split", "", "") in ours.columns else ours.pop("split")
    for v in ("side", "top"):
        sel = np.array([f"_{v}" in i[1] for i in ours.index])
        g = long(ours[sel])
        g["split"] = g.frame.map(dict(zip(["/".join(i) for i in ours.index[sel]], split[sel])))
        gt[("ours", v)] = g
        p = pd.read_hdf(OUT / f"gt_pol_{v}.h5")
        sp = p.pop("split") if "split" in p.columns.get_level_values(0) else None
        if isinstance(sp, pd.DataFrame):
            sp = sp.iloc[:, 0]
        g = long(p)
        g["split"] = g.frame.map(dict(zip(["/".join(i) for i in p.index], sp.values)))
        gt[("pol", v)] = g
    return gt


def errors(gt):
    rows = []
    for v in ("side", "top"):
        for d in ("ours", "pol"):
            g = gt[(d, v)].dropna(subset=["x", "y"])
            g = g[g.bodypart.isin(KEYPOINTS[v])]
            for m in ("ours", "pol"):
                f = OUT / (f"pred_ours_on_{d}_{v}.h5" if m == "ours" else f"pred_pol_{v}_on_{d}_{v}.h5")
                p = long(pd.read_hdf(f), scorer_level=False).rename(columns={"x": "px", "y": "py",
                                                                            "likelihood": "lik"})
                e = g.merge(p[["frame", "bodypart", "px", "py", "lik"]], on=["frame", "bodypart"], how="left")
                assert e.px.notna().all(), f"missing predictions {m} on {d} {v}"
                e["err_px"] = np.hypot(e.px - e.x, e.py - e.y)
                rows.append(e.assign(model=m, data=d, view=v))
    return pd.concat(rows, ignore_index=True)


def main():
    gt = ground_truth()
    e = errors(gt)
    mm_view = pd.read_csv(SIDE / "mm_per_px.csv").groupby("view").mm_per_px.median()
    e["err_mm"] = e.err_px * e.view.map(mm_view)
    hum = pd.read_csv(SIDE / "human_variability_mm.csv").groupby(["view", "bodypart"]).dist_mm.mean()
    # frames: "test" = the held-out frames of that dataset's own split; "all" = every labelled frame
    frame_err = e.groupby(["model", "data", "view", "split", "frame"]).err_mm.mean().reset_index()
    summ = []
    for sel, fr in (("test", frame_err[frame_err.split == "test"]), ("all", frame_err)):
        s = fr.groupby(["model", "data", "view"]).err_mm.agg(["mean", "median", "size"]).reset_index()
        summ.append(s.assign(frames=sel))
    summ = pd.concat(summ).rename(columns={"size": "n_frames"})
    summ.to_csv(OUT / "errors_summary.csv", index=False)
    kp = (e[e.split == "test"].groupby(["model", "data", "view", "bodypart"]).err_mm.mean().reset_index())
    kp.to_csv(OUT / "errors_per_keypoint.csv", index=False)
    print(summ.round(3).to_string(index=False))

    ps.apply() if hasattr(ps, "apply") else None
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1.1, 1.1], hspace=0.9, wspace=0.55)
    # a: mean frame error on each dataset's test frames, per view and model
    for j, v in enumerate(("side", "top")):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        s = summ[(summ.frames == "test") & (summ.view == v)]
        for k, d in enumerate(("ours", "pol")):
            for i, m in enumerate(("ours", "pol")):
                r = s[(s.data == d) & (s.model == m)]
                ax.bar(k + (i - 0.5) * 0.38, r["mean"].iloc[0], width=0.36, color=MODEL_COLOR[m],
                       label=MODEL_LABEL[m] if k == 0 else None)
                ax.text(k + (i - 0.5) * 0.38, r["mean"].iloc[0], f"{r['mean'].iloc[0]:.2f}", ha="center",
                        va="bottom", fontsize=5)
        ax.axhline(hum.loc[v].mean(), color="k", ls="--", lw=0.6, label="human (ours)")
        n = s.groupby("data").n_frames.first()
        ax.set_xticks([0, 1], [f"{DATA_LABEL[d]}\n(test, n = {n[d]})" for d in ("ours", "pol")])
        ax.set_ylabel("Mean test error (mm)")
        ax.set_title(f"{v.capitalize()} view", pad=2)
        if j == 0:
            ax.legend(frameon=False, fontsize=5)
            ax.text(-0.25, 1.1, "a", transform=ax.transAxes, fontweight="bold")
    # b: per keypoint, test frames of each dataset
    for i, d in enumerate(("ours", "pol")):
        for j, v in enumerate(("side", "top")):
            ax = fig.add_subplot(gs[1 + i, 2 * j:2 * j + 2])
            x = np.arange(len(KEYPOINTS[v]))
            for k, m in enumerate(("ours", "pol")):
                s = kp[(kp.data == d) & (kp.view == v) & (kp.model == m)].set_index("bodypart").reindex(KEYPOINTS[v])
                ax.bar(x + (k - 0.5) * 0.38, s.err_mm, width=0.36, color=MODEL_COLOR[m])
            h = hum.reindex(pd.MultiIndex.from_product([[v], KEYPOINTS[v]])).values
            ax.scatter(x, h, marker="_", s=40, color="k", lw=0.8, zorder=3)
            ax.set_xticks(x, [b.replace("_", " ") for b in KEYPOINTS[v]], rotation=50, ha="right", fontsize=5)
            ax.set_ylabel("Mean test error (mm)")
            ax.set_title(f"{DATA_LABEL[d]}, {v} (black: human, ours)", pad=2, fontsize=6)
            if j == 0:
                ax.text(-0.25, 1.1, "bc"[i], transform=ax.transAxes, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"other_users_comparison.{ext}", dpi=300, bbox_inches="tight")
    print("saved", OUT / "other_users_comparison.png")


if __name__ == "__main__":
    main()
