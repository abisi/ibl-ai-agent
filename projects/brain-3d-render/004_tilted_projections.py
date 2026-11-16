"""004 -- Projection views of the cleared brain from the top and tilted from the top, autofluorescence in cyan (summed
along the view ray) and DiI in magenta (maximum along the ray), on black and on white backgrounds.

DiI is kept only within `TUBE_UM` of the probe tracks clicked in sample space (brainglobe segmentation
sample_space/tracks/*.points, voxel coordinates of the 10 um downsampled volume): blood vessels and other DiI-bright
lines away from the probes are removed. Track = polyline through the clicked points (ordered along the track's
principal axis), extended by EXTEND_UM at both ends.
Views are orthographic: every voxel is splatted onto the image plane of the view (pixel = voxel size), so no
interpolation of the volume is needed. Axes: x = ML (right +), y = AP (anterior +), z = DV (dorsal +), from the
"asr" volume (origin anterior, superior, right).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 004_tilted_projections.py [brain]
Output: combined_results_ks4/brain-3d-render/<brain>/projections/
"""
import glob
import sys
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import tifffile
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BRAIN = sys.argv[1] if len(sys.argv) > 1 else "AB164"
REG = Path(f"/raid0/lmsmith/AB_brains/{BRAIN}/fused/registered")   # read only
OUT = Path(f"/mnt/lsens-analysis/Axel_Bisi/combined_results_ks4/brain-3d-render/{BRAIN}/projections")
VOX = 10.0              # um, downsampled volume
TUBE_UM = 80.0          # DiI kept within this distance of a clicked track
EXTEND_UM = 150.0       # track extension beyond the first / last clicked point
VIEWS = {               # name: (azimuth, elevation); camera tilted from the vertical toward azimuth (0 = anterior,
    "top": (0, 90),     # 90 = left side, 135 = back-left); anterior is up in every image
    "tilt_left_20": (90, 70),
    "tilt_left_35": (90, 55),
    "tilt_left_50": (90, 40),
    "tilt_front_left_35": (45, 55),
    "tilt_back_left_35": (135, 55),
}
CYAN_K, MAGENTA_K = np.array([0.0, 0.85, 0.95]), np.array([1.0, 0.0, 0.85])     # on black
CYAN_W, MAGENTA_W = np.array([0.0, 0.50, 0.65]), np.array([0.80, 0.0, 0.60])     # on white


def load():
    af = tifffile.imread(REG / f"downsampled_{BRAIN}.czi_fused_tp_0_ch_0.tiff")
    di = tifffile.imread(REG / f"downsampled_{BRAIN}.czi_fused_tp_0_ch_1.tiff")
    return af, di


def tracks():
    out = {}
    for f in sorted(glob.glob(str(REG / "segmentation" / "sample_space" / "tracks" / "*.points"))):
        p = h5py.File(f, "r")["df/block0_values"][:].astype(float)
        c = p.mean(0)
        axis = np.linalg.svd(p - c)[2][0]
        p = p[np.argsort((p - c) @ axis)]
        ext = EXTEND_UM / VOX
        p = np.vstack([p[0] - axis * ext, p, p[-1] + axis * ext])
        out[Path(f).stem] = p
    return out


def track_mask(shape, trk):
    """Voxels within TUBE_UM of any track polyline (distance transform of the rasterised, densely sampled lines)."""
    lines = np.zeros(shape, bool)
    for p in trk.values():
        for a, b in zip(p[:-1], p[1:]):
            n = int(np.ceil(np.linalg.norm(b - a) * 2)) + 1
            q = np.rint(a + (b - a) * np.linspace(0, 1, n)[:, None]).astype(int)
            q = q[(q >= 0).all(1) & (q < shape).all(1)]
            lines[tuple(q.T)] = True
    lo = np.maximum(np.argwhere(lines).min(0) - int(TUBE_UM / VOX) - 2, 0)
    hi = np.minimum(np.argwhere(lines).max(0) + int(TUBE_UM / VOX) + 3, shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    mask = np.zeros(shape, bool)
    mask[sl] = ndimage.distance_transform_edt(~lines[sl]) * VOX <= TUBE_UM
    return mask


def basis(az, el):
    az, el = np.deg2rad(az), np.deg2rad(el)
    d = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])   # center -> camera
    f = -d
    up = np.array([0.0, 1.0, 0.0])          # anterior up in every view (tilted top views)
    right = np.cross(f, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    return right, up


def world(idx, center):
    """Voxel indices (AP, DV, ML) -> world um (x ML right +, y AP anterior +, z DV dorsal +)."""
    i0, i1, i2 = np.broadcast_arrays(*idx)
    return np.stack([-(i2 - center[2]), -(i0 - center[0]), -(i1 - center[1])], -1) * VOX


def project(af, di_masked, az, el, size):
    right, up = basis(az, el)
    center = (np.array(af.shape) - 1) / 2
    n = size
    sum_img = np.zeros(n * n)
    max_img = np.zeros(n * n)
    i1, i2 = np.meshgrid(np.arange(af.shape[1]), np.arange(af.shape[2]), indexing="ij")
    for i0 in range(0, af.shape[0], 40):
        sl = slice(i0, min(i0 + 40, af.shape[0]))
        k = np.arange(sl.start, sl.stop)[:, None, None]
        w = world((np.broadcast_to(k, (len(k),) + i1.shape), i1[None], i2[None]), center)
        uf = w @ right / VOX + n / 2
        vf = -(w @ up) / VOX + n / 2
        u0, v0 = np.floor(uf).astype(np.int64), np.floor(vf).astype(np.int64)
        fu, fv = uf - u0, vf - v0
        val = af[sl].astype(np.float64)
        # bilinear splatting (each voxel spread over its 4 nearest pixels): no aliasing stripes in tilted views
        for du, dv_, wt in ((0, 0, (1 - fu) * (1 - fv)), (1, 0, fu * (1 - fv)), (0, 1, (1 - fu) * fv), (1, 1, fu * fv)):
            uu, vv = u0 + du, v0 + dv_
            ok4 = (uu >= 0) & (uu < n) & (vv >= 0) & (vv < n) & (val > 0)
            sum_img += np.bincount((vv * n + uu)[ok4], weights=(val * wt)[ok4], minlength=n * n)
        u, v = np.rint(uf).astype(np.int64), np.rint(vf).astype(np.int64)
        ok = (u >= 0) & (u < n) & (v >= 0) & (v < n)
        pix = (v * n + u)[ok]
        dv = di_masked[sl][ok]
        sel = dv > 0
        if sel.any():
            o = np.argsort(dv[sel])
            np.maximum.at(max_img, pix[sel][o], dv[sel][o])
    return sum_img.reshape(n, n), max_img.reshape(n, n)


def compose(a, m, white):
    if white:
        rgb = np.ones(a.shape + (3,))
        alpha_a = 0.85 * a[..., None]
        rgb = rgb * (1 - alpha_a) + CYAN_W * alpha_a
        alpha_m = m[..., None]
        return rgb * (1 - alpha_m) + MAGENTA_W * alpha_m
    return np.clip(a[..., None] * CYAN_K * 0.9 + m[..., None] * MAGENTA_K, 0, 1)


def crop_box(imgs, pad=20):
    occ = np.any([i > 0 for i in imgs], axis=0)
    r, c = np.nonzero(occ)
    return slice(max(r.min() - pad, 0), r.max() + pad), slice(max(c.min() - pad, 0), c.max() + pad)


def save(rgb, path, white, title=None):
    h, w = rgb.shape[:2]
    fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="lanczos")
    col = "k" if white else "w"
    bar = 1000 / VOX
    ax.plot([w - 40 - bar, w - 40], [h - 40, h - 40], color=col, lw=3)
    ax.text(w - 40 - bar / 2, h - 50, "1 mm", color=col, ha="center", va="bottom", fontsize=9)
    if title:
        ax.text(20, 30, title, color=col, fontsize=10, va="top")
    ax.axis("off")
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, facecolor="w" if white else "k")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    af, di = load()
    print(BRAIN, af.shape, flush=True)
    trk = tracks()
    mask = track_mask(af.shape, trk)
    bg = np.percentile(af[::4, ::4, ::4], 5)
    tissue = af > bg + 0.15 * (np.percentile(af[::4, ::4, ::4], 99) - bg)
    inside = di[mask & tissue].astype(np.float32)
    lo, hi = np.percentile(inside, 75), np.percentile(inside, 99.9)
    di_m = np.where(mask & tissue, np.clip((di.astype(np.float32) - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    di_all = np.where(tissue, np.clip((di.astype(np.float32) - lo) / (hi - lo), 0, 1), 0).astype(np.float32)
    af_t = np.where(tissue, af, 0).astype(np.float32)
    size = int(np.ceil(np.linalg.norm(af.shape))) + 40
    # check figure: DiI from the top before / after the track mask, clicked points on top
    s_top, all_top = project(af_t, di_all, *VIEWS["top"], size)
    _, kept_top = project(af_t, di_m, *VIEWS["top"], size)
    rr, cc = crop_box([s_top])
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), facecolor="k")
    for ax, img, t in zip(axs, (all_top, kept_top), ("DiI, all", f"DiI within {TUBE_UM:.0f} um of the clicked tracks")):
        ax.imshow(img[rr, cc], cmap="magma", vmin=0, vmax=1)
        ax.set_title(t, color="w")
        ax.axis("off")
    right, up = basis(*VIEWS["top"])
    center = (np.array(af.shape) - 1) / 2
    for name, p in trk.items():
        w = world(p.T, center)
        u = w @ right / VOX + size / 2 - cc.start
        v = -(w @ up) / VOX + size / 2 - rr.start
        axs[1].plot(u, v, "-", color="c", lw=0.8, alpha=0.7)
        axs[1].text(u[-1], v[-1], name, color="c", fontsize=8)
    fig.savefig(OUT / f"{BRAIN}_check_track_mask.png", dpi=150, facecolor="k", bbox_inches="tight")
    plt.close(fig)
    print("check figure saved", flush=True)
    panels = {}
    for name, (az, el) in VIEWS.items():
        s, m = (s_top, kept_top) if name == "top" else project(af_t, di_m, az, el, size)
        rr, cc = crop_box([s])
        s, m = s[rr, cc], ndimage.gaussian_filter(m[rr, cc], 0.6)
        brain = s[s > 0]
        a = np.clip((s - np.percentile(brain, 2)) / (np.percentile(brain, 99.7) - np.percentile(brain, 2)), 0, 1)
        a = ndimage.gaussian_filter(a, 0.6)
        m = np.clip(m / max(np.percentile(m[m > 0], 99.5), 1e-6), 0, 1) if (m > 0).any() else m
        for white in (False, True):
            rgb = compose(a, m, white)
            tag = "white" if white else "black"
            save(rgb, OUT / f"{BRAIN}_{name}_{tag}", white)
            panels[(name, tag)] = rgb
        print("view", name, flush=True)
    for tag in ("black", "white"):
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), facecolor="w" if tag == "white" else "k")
        for ax, name in zip(axs.ravel(), VIEWS):
            ax.imshow(panels[(name, tag)], interpolation="lanczos")
            az, el = VIEWS[name]
            ax.set_title("top" if el == 90 else name.replace("_", " ").replace("tilt", "tilted") + " deg from vertical",
                         color="k" if tag == "white" else "w")
            ax.axis("off")
        fig.suptitle(f"{BRAIN}: autofluorescence (cyan), DiI probe tracks (magenta)",
                     color="k" if tag == "white" else "w", fontsize=16)
        fig.savefig(OUT / f"{BRAIN}_overview_{tag}.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
    print("done", OUT)


if __name__ == "__main__":
    main()
