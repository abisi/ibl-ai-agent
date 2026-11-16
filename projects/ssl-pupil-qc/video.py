"""Raw side-video access for pupil QC, cached per session in <pupil_qc>/video_cache/ so pages re-render without the
video (user, 2026-11-03): eye-ROI box (roi.json), eye-ROI mean intensity of every frame (eye_intensity.npy, ffmpeg crop
pipe, ~12 min per 200 GB session), example frames (frames.npz, whole frames, OpenCV random access)."""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

def eye_roi(df, valid, scale=1.6, frame_shape=(480, 640)):
    """Square box around the median pupil centre, half-size = scale x median pupil width (covers the lid)."""
    cx = np.nanmedian(np.mean([df[f"{p}_x"][valid] for p in ("top", "left", "right", "bottom")], 0))
    cy = np.nanmedian(np.mean([df[f"{p}_y"][valid] for p in ("top", "left", "right", "bottom")], 0))
    d = np.nanmedian(np.hypot(df.left_x[valid] - df.right_x[valid], df.left_y[valid] - df.right_y[valid]))
    h = int(max(scale * d, 12))
    x0, y0 = int(max(cx - h, 0)), int(max(cy - h, 0))
    x1, y1 = int(min(cx + h, frame_shape[1])), int(min(cy + h, frame_shape[0]))
    return [x0, y0, x1 - x0, y1 - y0]


class VideoCache:
    def __init__(self, out_dir, video_path):
        self.dir = Path(out_dir) / "video_cache"
        self.dir.mkdir(exist_ok=True)
        self.video = video_path

    # -- ROI
    def roi(self, df, valid):
        f = self.dir / "roi.json"
        if f.exists():
            return json.loads(f.read_text())["roi"]
        roi = eye_roi(df, valid)
        f.write_text(json.dumps({"roi": roi, "video": self.video}))
        return roi

    # -- intensity
    def intensity(self, roi, compute=True):
        f = self.dir / "eye_intensity.npy"
        if f.exists():
            return np.load(f)
        if not compute:
            return None
        x = eye_intensity(self.video, roi)
        np.save(f, x)
        return x

    # -- example frames (whole frames, so crops can be re-cut without the video)
    def frames(self, frames):
        """{frame: image}; frames not yet cached are read from the video and added to the cache."""
        f = self.dir / "frames.npz"
        cache = {}
        if f.exists():
            with np.load(f) as z:  # read each array once (indexing z["images"] repeatedly decompresses it every time)
                images, idx = z["images"], z["idx"]
            cache = {int(k): images[i] for i, k in enumerate(idx)}
        need = [int(x) for x in frames if int(x) not in cache and x >= 0]
        if need:
            cache.update(read_frames(self.video, need))
            keys = sorted(cache)
            np.savez_compressed(f, idx=np.array(keys), images=np.stack([cache[k] for k in keys]))
        return {int(x): cache.get(int(x)) for x in frames}


def pupil_crop_box(df, valid, frame_shape=(480, 640), width_factor=2.0, min_half=20):
    """Square crop centred on the median pupil centre, half-size = width_factor x median pupil width."""
    cx = np.nanmedian(np.mean([df[f"{p}_x"][valid] for p in ("top", "left", "right", "bottom")], 0))
    cy = np.nanmedian(np.mean([df[f"{p}_y"][valid] for p in ("top", "left", "right", "bottom")], 0))
    w = np.nanmedian(np.hypot(df.left_x[valid] - df.right_x[valid], df.left_y[valid] - df.right_y[valid]))
    h = int(max(width_factor * w, min_half))
    return int(cx) - h, int(cy) - h, 2 * h


def crop(img, box):
    """Square crop (x0, y0, size), padded with black outside the frame."""
    x0, y0, n = box
    out = np.zeros((n, n), img.dtype)
    xa, ya = max(x0, 0), max(y0, 0)
    xb, yb = min(x0 + n, img.shape[1]), min(y0 + n, img.shape[0])
    out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return out


def read_frames(path, frames):
    cap = cv2.VideoCapture(path)
    out = {}
    for f in sorted(set(int(x) for x in frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            out[f] = img[:, :, 0]
    cap.release()
    return out


def eye_intensity(path, roi, chunk=4096):
    """Mean grey level inside roi=(x, y, w, h) for every frame of the video."""
    x, y, w, h = roi
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"crop={w}:{h}:{x}:{y}", "-f", "rawvideo", "-pix_fmt", "gray",
           "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    size = w * h
    means = []
    while True:
        buf = proc.stdout.read(size * chunk)
        if not buf:
            break
        a = np.frombuffer(buf, np.uint8)
        a = a[: len(a) // size * size].reshape(-1, size)
        means.append(a.mean(1, dtype=np.float32))
    proc.wait()
    return np.concatenate(means) if means else np.array([], np.float32)
