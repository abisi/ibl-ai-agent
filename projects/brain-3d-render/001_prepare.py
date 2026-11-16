"""001 -- Block-mean downsampling of the fused full-resolution channels (3.64 um isotropic) for rendering.
Input: <work>/<brain>/fused/<channel file> (staged from the NAS by sbatch/prepare.sbatch).
Output: <work>/<brain>/ds<f>_<channel name>.npy (uint16, axes AP, DV, ML) for every factor in config `downsample`.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
import yaml

cfg = yaml.safe_load(open(Path(__file__).with_name("config.yaml")))
work = Path(cfg["paths"]["work"]) / cfg["brain"]


def block_mean(vol, f, out):
    n0, n1, n2 = (s // f for s in vol.shape)
    for i in range(n0):
        slab = np.asarray(vol[i * f:(i + 1) * f, :n1 * f, :n2 * f], dtype=np.float32)
        out[i] = slab.reshape(f, n1, f, n2, f).mean(axis=(0, 2, 4)).round().astype(np.uint16)
        if i % 100 == 0:
            print(f"  plane {i}/{n0}", flush=True)


for ch in cfg["channels"]:
    src = work / "fused" / ch["file"]
    vol = tifffile.memmap(src, mode="r")
    print(ch["name"], vol.shape, vol.dtype, flush=True)
    for f in cfg["downsample"]:
        dst = work / f"ds{f}_{ch['name']}.npy"
        if dst.exists():
            continue
        out = np.lib.format.open_memmap(dst.with_suffix(".tmp.npy"), mode="w+", dtype=np.uint16,
                                        shape=tuple(s // f for s in vol.shape))
        block_mean(vol, f, out)
        out.flush()
        del out
        dst.with_suffix(".tmp.npy").rename(dst)
        print("wrote", dst, flush=True)
sys.exit(0)
