"""Contact sheet of eye-ROI candidates (eye_roi_candidates.csv from eye_roi_inset.py): the 20 best-tracked sessions'
chosen frames, cropped around the pupil with the four-point contour, to pick the example by eye (user 2026-11-25: "find a
better example where we see the pupil well"). Output: Figures/_explore/Figure4_passive_cd_<date>/eye_roi_contact_sheet.png.
Run (haas, repo root; cv2): .venv/bin/python projects/ssl-paper-figures/figures/Figure4_passive_cd/_explore/eye_roi_contact_sheet.py"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("eye", HERE / "eye_roi_inset.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)

c = pd.read_csv(E.OUT / "eye_roi_candidates.csv").sort_values("quality", ascending=False).head(20)
fig, axs = plt.subplots(4, 5, figsize=(12, 10))
for ax, r in zip(axs.ravel(), c.itertuples()):
    xy, _ = E.keypoints(r.h5)
    im = E.read_frame(r.video, r.frame)
    poly = xy[r.frame]
    cc = poly.mean(0)
    half = int(round(2.2 * r.diameter_px))
    x0, y0 = max(int(cc[0]) - half, 0), max(int(cc[1]) - half, 0)
    crop = im[y0: y0 + 2 * half, x0: x0 + 2 * half]
    lo, hi = np.percentile(crop, [1, 99.5])
    ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi)
    P = poly - [x0, y0]
    P = np.vstack([P, P[:1]])
    ax.plot(P[:, 0], P[:, 1], color="white", lw=0.8)
    ax.set_title(f"{r.session_id}\nq {r.quality:.2f}  c {r.contrast:.2f}  d {r.diameter_px * E.MM_PER_PX:.2f} mm", fontsize=7)
    ax.set_axis_off()
fig.tight_layout()
fig.savefig(E.OUT / "eye_roi_contact_sheet.png", dpi=150)
print("saved", E.OUT / "eye_roi_contact_sheet.png")
