"""003 -- Imaris .ims file (HDF5, Imaris 5.5 layout) of both full-resolution channels, with a 2x block-mean pyramid,
per-level histograms, voxel size and channel colours / display ranges, for rendering in Imaris.
Axes: Z = AP, Y = DV, X = ML (fused volume order; rotate freely in Imaris). gzip level 2, chunks (16, 128, 128).
Output: <work>/<brain>/<brain>_fused_2ch.ims
"""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import tifffile
import yaml

cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))
work = Path(cfg["paths"]["work"]) / cfg["brain"]
CHUNK = (16, 128, 128)
MIN_SIZE = 256


def s1(x):
    return np.frombuffer(str(x).encode(), dtype="S1")


def attrs(obj, **kw):
    for k, v in kw.items():
        obj.attrs[k] = s1(v)


def histogram(ds):
    h = np.zeros(256, np.uint64)
    lo, hi = 0, 65535
    for i in range(0, ds.shape[0], 64):
        h += np.histogram(ds[i:i + 64], bins=256, range=(lo, hi))[0].astype(np.uint64)
    return h, lo, hi


def main():
    t0 = time.time()
    vox = cfg["voxel_um"]
    src = [tifffile.memmap(work / "fused" / ch["file"], mode="r") for ch in cfg["channels"]]
    shape = src[0].shape
    levels = [shape]
    while min(levels[-1]) > MIN_SIZE:
        levels.append(tuple(max(1, s // 2) for s in levels[-1]))
    info = json.load(open(work / "render" / "render_info.json")) if (work / "render" / "render_info.json").exists() \
        else {}
    out = work / f"{cfg['brain']}_fused_2ch.ims"
    tmp = out.with_suffix(".ims.tmp")
    with h5py.File(tmp, "w") as f:
        f.attrs["ImarisDataSet"] = s1("ImarisDataSet")
        f.attrs["ImarisVersion"] = s1("5.5.0")
        f.attrs["DataSetInfoDirectoryName"] = s1("DataSetInfo")
        f.attrs["ThumbnailDirectoryName"] = s1("Thumbnail")
        f.attrs["DataSetDirectoryName"] = s1("DataSet")
        f.attrs["NumberOfDataSets"] = np.array([1], np.uint32)
        dsi = f.create_group("DataSetInfo")
        z, y, x = shape
        attrs(dsi.create_group("Image"), X=x, Y=y, Z=z, Unit="um", Noc=len(src), Description=cfg["brain"],
              ExtMin0=0, ExtMin1=0, ExtMin2=0, ExtMax0=f"{x * vox:.3f}", ExtMax1=f"{y * vox:.3f}",
              ExtMax2=f"{z * vox:.3f}", Name=f"{cfg['brain']} fused")
        attrs(dsi.create_group("ImarisDataSet"), Creator="ibl-ai-agent brain-3d-render", NumberOfImages=1,
              Version="5.5")
        attrs(dsi.create_group("Imaris"), Version="7.0")
        attrs(dsi.create_group("TimeInfo"), DatasetTimePoints=1, FileTimePoints=1,
              TimePoint1="2025-06-26 00:00:00.000")
        for c, ch in enumerate(cfg["channels"]):
            rng = info.get("display_ranges_raw", {}).get(ch["name"], (0, 4000))
            attrs(dsi.create_group(f"Channel {c}"), Name=ch["name"], Color=" ".join(f"{v:.3f}" for v in ch["color"]),
                  ColorMode="BaseColor", ColorOpacity=1, ColorRange=f"{rng[0]:.1f} {rng[1]:.1f}", GammaCorrection=1)
        root = f.create_group("DataSet")
        for r, lshape in enumerate(levels):
            for c in range(len(src)):
                g = root.require_group(f"ResolutionLevel {r}/TimePoint 0/Channel {c}")
                attrs(g, ImageSizeX=lshape[2], ImageSizeY=lshape[1], ImageSizeZ=lshape[0])
                chunk = tuple(min(a, b) for a, b in zip(CHUNK, lshape))
                ds = g.create_dataset("Data", shape=lshape, dtype=np.uint16, chunks=chunk, compression="gzip",
                                      compression_opts=2)
                if r == 0:
                    for i in range(0, lshape[0], 64):
                        ds[i:i + 64] = src[c][i:i + 64]
                else:
                    prev = root[f"ResolutionLevel {r - 1}/TimePoint 0/Channel {c}/Data"]
                    for i in range(lshape[0]):
                        a = prev[2 * i:2 * i + 2, :2 * lshape[1], :2 * lshape[2]].astype(np.float32)
                        a = a.reshape(a.shape[0], lshape[1], 2, lshape[2], 2).mean(axis=(0, 2, 4))
                        ds[i] = a.round().astype(np.uint16)
                h, lo, hi = histogram(ds)
                g.create_dataset("Histogram", data=h)
                attrs(g, HistogramMin=f"{lo:.3f}", HistogramMax=f"{hi:.3f}")
                print(f"level {r} {lshape} channel {c} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    tmp.rename(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
