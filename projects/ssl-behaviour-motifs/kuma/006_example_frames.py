"""006 (Kuma, CPU) -- raw-video snapshots of the most used states, per session x method x view.

For each of the 6 most used states of the session (lightning-action: classes still / whisking / jaw / lick), 3 bouts
(longest-first among bouts >= 100 ms, spread over the session): the raw video frame at the bout middle (50 Hz index b ->
200 Hz video frame 4 b + 2, DLC frame = video frame), cropped around the modelled keypoints, with the keypoint
trajectories of the bout overlaid (aligned coordinates mapped back to pixels with the stored similarity transform).
Output: results/frames/<sid>_<method>_<view>.png
Usage: python 006_example_frames.py
"""
import glob
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path("/work/lsens/bisi/behaviour_motifs")
INP, RES = BASE / "data" / "inputs", BASE / "results"
OUT = RES / "frames"
LA = ["background", "still", "whisking", "jaw", "lick"]
COLS = plt.cm.tab10.colors


def states(sid, method, view):
    if method == "kpms":
        f = RES / "kpms" / view / "seed0" / f"{sid}.npz"
    elif method == "arhmm":
        f = RES / "arhmm" / view / f"{sid}.npz"
    else:
        f = RES / "lightning_action" / view / f"{sid}_probs.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return d["z"] if "z" in d else d["pred"]


def bouts(z, s, min_len=5):
    ch = np.flatnonzero(np.diff(np.r_[-99, z, -99]) != 0)
    out = [(a, b) for a, b in zip(ch[:-1], ch[1:]) if z[a] == s and b - a >= min_len]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = pd.read_csv(BASE / "data" / "001_selected_sessions.csv")
    for sid in sel.session_id:
        mouse = sid.split("_")[0]
        for view in ("side", "top"):
            vids = glob.glob(str(BASE.parent / "data" / mouse / sid / "Video" / f"*_{view} *.avi"))
            if len(vids) != 1:
                print("no video", sid, view, vids)
                continue
            k = np.load(INP / f"{sid}_{view}_keypoints.npz")
            xy = k["xy"].astype(float)
            raw = np.einsum("ij,tkj->tki", k["R"].T, (xy - k["tvec"]) / float(k["scale"]))
            cap = cv2.VideoCapture(vids[0])
            nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for method in ("kpms", "arhmm", "lightning_action"):
                z = states(sid, method, view)
                if z is None:
                    continue
                z = np.asarray(z)[: len(raw)]
                if method == "lightning_action":
                    top = [1, 2, 3, 4]
                else:
                    top = list(np.argsort(-np.bincount(z))[:6])
                fig, axs = plt.subplots(len(top), 3, figsize=(6, 1.9 * len(top)))
                for r, s in enumerate(top):
                    bs = sorted(bouts(z, s), key=lambda ab: -(ab[1] - ab[0]))[:30]
                    bs = sorted(bs)[:: max(1, len(bs) // 3)][:3] if bs else []
                    for c in range(3):
                        ax = axs[r, c]
                        ax.axis("off")
                        if c >= len(bs):
                            continue
                        a, b = bs[c]
                        fr = 4 * ((a + b) // 2) + 2
                        if fr >= nvid:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
                        ok, img = cap.read()
                        if not ok:
                            continue
                        ax.imshow(img[..., ::-1] if img.ndim == 3 else img, cmap="gray")
                        tr = raw[a:b]
                        for j in range(tr.shape[1]):
                            ax.plot(tr[:, j, 0], tr[:, j, 1], color=COLS[j % 10], lw=0.8)
                        cx, cy = np.nanmedian(raw[:, :, 0]), np.nanmedian(raw[:, :, 1])
                        half = 1.3 * max(np.nanpercentile(raw[:, :, 0], 99) - np.nanpercentile(raw[:, :, 0], 1),
                                         np.nanpercentile(raw[:, :, 1], 99) - np.nanpercentile(raw[:, :, 1], 1), 80) / 2
                        ax.set_xlim(cx - half, cx + half)
                        ax.set_ylim(cy + half, cy - half)
                        ax.set_title(f"{(b - a) * 20} ms @ {k['t'][a]:.0f} s", fontsize=6)
                    name = LA[s] if method == "lightning_action" else f"{'S' if method == 'kpms' else 'A'}{s}"
                    axs[r, 0].text(-0.05, 0.5, name, transform=axs[r, 0].transAxes, ha="right", va="center", fontsize=8)
                fig.suptitle(f"{sid} {method} {view} (lines: keypoint paths over the bout)", fontsize=7)
                fig.tight_layout()
                fig.savefig(OUT / f"{sid}_{method}_{view}.png", dpi=110)
                plt.close(fig)
            cap.release()
            print("done", sid, view, flush=True)


if __name__ == "__main__":
    main()
