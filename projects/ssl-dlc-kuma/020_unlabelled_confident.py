"""Keypoints the model predicts confidently (likelihood >= 0.5) where the label is missing (absent / occluded, or
missed by the labeller): pupil x4 and tongue (side view), best snapshot of the best evaluated model (019). One crop
(120 x 120 px) per case, test frames first, then train frames; prediction as a filled dot, labelled keypoints as
crosses. Counts per keypoint and split.

Output: DLCProjects/merged_training_2026-11-09/training_summary/020_unlabelled_confident.png (+ .csv)
Run on haas: ~/code/ibl-ai-agent/.venv/bin/python 020_unlabelled_confident.py
"""

import importlib.util
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

spec = importlib.util.spec_from_file_location("ts", Path(__file__).with_name("019_training_summary.py"))
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)
KPS = ["pupil_top", "pupil_left", "pupil_right", "pupil_bottom", "tongue"]
HALF = 40


def main():
    frames, gt_long = ts.load_ground_truth()
    snap = pd.read_csv(ts.OUT / "errors_per_snapshot.csv")
    t = snap[(snap.split == "test") & (snap.view == "all")]
    s, ep = t.loc[t.mean_err_px.idxmin(), ["shuffle", "epoch"]].astype(int)
    e = ts.errors(ts.snapshot_files(s)[ep], frames, gt_long)
    c = e[e.bodypart.isin(KPS) & ~e.labelled & (e.likelihood >= ts.CUTOFF)].copy()
    n_unlab = e[e.bodypart.isin(KPS) & ~e.labelled].groupby(["bodypart", "split"]).size().rename("n_not_labelled")
    counts = c.groupby(["bodypart", "split"]).size().rename("n_confident").to_frame().join(n_unlab, how="right")
    counts = counts.fillna(0).astype(int)
    counts["fraction"] = (counts.n_confident / counts.n_not_labelled).round(3)
    print(f"model shuffle {s}, epoch {ep}\n{counts.to_string()}")
    c = c.merge(frames[["frame", "image"]], on="frame").sort_values(["split", "bodypart", "likelihood"],
                                                                     ascending=[True, True, False])
    c[["split", "bodypart", "image", "likelihood", "x", "y"]].to_csv(ts.OUT / "020_unlabelled_confident.csv",
                                                                     index=False)
    c = pd.concat([c[c.split == "test"], c[c.split == "train"].groupby("bodypart").head(4)])
    n = len(c)
    ncol = 8
    fig, axes = plt.subplots(int(np.ceil(n / ncol)), ncol, figsize=(ncol * 1.3, 1.45 * np.ceil(n / ncol)),
                             squeeze=False)
    cmap = plt.get_cmap("tab20b_r", 13)
    for ax, (_, r) in zip(axes.flat, c.iterrows()):
        img = mpimg.imread(ts.PROJ / r.image)
        img = img[..., :3].mean(axis=2) if img.ndim == 3 else img
        x0, y0 = int(np.clip(r.x - HALF, 0, img.shape[1] - 2 * HALF)), int(np.clip(r.y - HALF, 0, img.shape[0] - 2 * HALF))
        ax.imshow(img[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF], cmap="gray")
        lab = gt_long[(gt_long.frame == frames.loc[frames.image == r.image, "frame"].iloc[0]) & gt_long.x_gt.notna()]
        ax.scatter(lab.x_gt - x0, lab.y_gt - y0, marker="x", s=8, lw=0.6, color="cyan")
        ax.scatter(r.x - x0, r.y - y0, s=14, color="red", edgecolors="none")
        ax.set_xlim(0, 2 * HALF); ax.set_ylim(2 * HALF, 0)
        ax.set_title(f"{r.split} {r.bodypart}\n{Path(r.image).parent.name[:18]} p={r.likelihood:.2f}", fontsize=4.5)
        ax.axis("off")
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Confident predictions (red) where the keypoint is not labelled; cyan = labelled keypoints",
                 fontsize=7)
    fig.tight_layout()
    fig.savefig(ts.OUT / "020_unlabelled_confident.png", dpi=200)


if __name__ == "__main__":
    main()
