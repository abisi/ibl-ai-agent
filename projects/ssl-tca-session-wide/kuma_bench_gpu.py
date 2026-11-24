"""Kuma benchmark on the same tensors (AB123, baseline-subtracted, soft-normalised, as 006): tensortools cp_als (CPU,
1 thread), tensorly parafac ALS on CPU (numpy, 1 thread) and on GPU (pytorch, float32 / float64). Same tolerance and
iteration cap. Reports seconds per fit, best relative error, iterations, and agreement of the best tensorly solutions
with the best tensortools solution (kruskal_align similarity, permutation-matched)."""
import json
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
import tensorly as tl
import tensortools as tt
import torch
from tensorly.decomposition import parafac

D = "/work/lsens/bisi/tca_gpu/data/AB123_20240806_110231"
RANKS = [1, 2, 5, 10, 15, 20]
TOL, MAX_ITER = 1e-6, 500


def prepare(align):
    d = np.load(f"{D}/{align}.npz")
    R, base = d["rate"], d["baseline"]
    keep = (np.nanmean(R, (0, 2)) + base.mean(0)) / 2 >= 0.5
    X = R[:, keep] - base[:, keep, None]
    X = X / (np.nanmax(np.abs(np.nanmean(X, 0, keepdims=True)), 2, keepdims=True) + 5.0)
    return np.nan_to_num(X).astype(np.float64)


def tl_fit(X, r, seed):
    cp, err = parafac(X, r, n_iter_max=MAX_ITER, tol=TOL, init="random", random_state=seed, return_errors=True)
    return cp, float(err[-1]), len(err)


def to_kt(cp):
    w, fs = cp
    fs = [tl.to_numpy(f).astype(np.float64) for f in fs]
    fs[0] = fs[0] * tl.to_numpy(w)[None].astype(np.float64)
    return tt.KTensor(fs)


out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "tensorly": tl.__version__}
print(out, flush=True)
for align in ("stim_long", "jaw"):
    X = prepare(align)
    out[align] = {"shape": list(X.shape)}
    best = {}
    for r in RANKS:
        tl.set_backend("numpy")
        t0 = time.time()
        res = [tt.cp_als(X, r, random_state=s, verbose=False, max_iter=MAX_ITER, tol=TOL) for s in range(3)]
        sec = (time.time() - t0) / 3
        bt = min(res, key=lambda f: f.obj)
        out[align][f"tensortools_cpu_r{r}"] = dict(sec_per_fit=round(sec, 3), best_err=round(bt.obj, 5),
                                                   mean_iters=float(np.mean([len(f.obj_hist) for f in res])))
        t0 = time.time()
        res = [tl_fit(X, r, s) for s in range(3)]
        sec = (time.time() - t0) / 3
        bc = min(res, key=lambda x: x[1])
        out[align][f"tensorly_cpu_r{r}"] = dict(sec_per_fit=round(sec, 3), best_err=round(bc[1], 5),
                                                mean_iters=float(np.mean([n for _, _, n in res])),
                                                sim_to_tensortools=round(float(tt.kruskal_align(bt.factors, to_kt(bc[0]))), 3))
        tl.set_backend("pytorch")
        for dtype in (torch.float32, torch.float64):
            Xg = torch.tensor(X, dtype=dtype, device="cuda")
            tl_fit(Xg, 2, 0)   # warm-up
            torch.cuda.synchronize()
            t0 = time.time()
            res = [tl_fit(Xg, r, s) for s in range(10)]
            torch.cuda.synchronize()
            sec = (time.time() - t0) / 10
            bg = min(res, key=lambda x: x[1])
            k = f"tensorly_gpu_{str(dtype)[-7:]}_r{r}"
            out[align][k] = dict(sec_per_fit=round(sec, 3), best_err=round(bg[1], 5),
                                 mean_iters=float(np.mean([n for _, _, n in res])),
                                 sim_to_tensortools=round(float(tt.kruskal_align(bt.factors, to_kt(bg[0]))), 3))
        for k, v in out[align].items():
            if k.endswith(f"_r{r}"):
                print(align, k, v, flush=True)
json.dump(out, open("/work/lsens/bisi/tca_gpu/bench_result.json", "w"), indent=1)
print("done", flush=True)
