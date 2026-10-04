"""007 -- Mouse-level cohort-label permutation test, R+ vs R-, learning day (user 2026-09-29: "do the permutation
overnight but without the shifts"). Same pseudo-population decoding as 002 (imported), per target x area:
  - pooled eligible sessions (one per mouse) of both cohorts; permutation p = 0 keeps the true cohort labels, p >= 1
    shuffles the labels across mice (group sizes kept);
  - per permutation, N_IT iterations per pseudo-group: N_MICE mice with replacement from the group, N_NEURONS units
    per session with replacement, balanced-reuse pseudo-trials, 3-fold outer CV, z-scoring in training -> real-label
    balanced-accuracy curve (NO linear-shift null, user);
  - C fixed per bin (not re-selected): the C index chosen most often in the 002 real decodes of this target x area,
    pooled over both cohorts, iterations and folds -- identical for every permutation and group (makes the job
    ~15x cheaper than re-running the inner CV);
  - stored: mean accuracy curve over the N_IT iterations of each pseudo-group.
Test (in the figure scripts): statistic = mean_a - mean_b per bin (or window); two-sided p = (1 + #{p >= 1:
|stat_p| >= |stat_0|}) / (1 + N_PERM).
Output: ../artifacts/007_cohort_perm_<target>.parquet (appendable; one row per area x permutation).
Run (haas, repo root): OMP_NUM_THREADS=1 ... python .../007_cohort_permutation.py [n_perm=200] [n_it=20] [areas|all]
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("SSL_PSEUDO_N_NEURONS", "10")          # as the 002 full run (user: 20 mice x 10 neurons)
os.environ.setdefault("SSL_PSEUDO_N_MICE", "20")
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT_REPO = HERE.parents[2]
sys.path.insert(0, str(ROOT_REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("p002", HERE / "002_pseudopop_decoding.py")
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)
ART = HERE.parent / "artifacts"
COHORTS = ("R+", "R-")
N_WORKERS = int(os.environ.get("SSL_PERM_N_WORKERS", "60"))
CHUNK = int(os.environ.get("SSL_PERM_CHUNK", "20"))


def fixed_c(target, area):
    """Modal 002 C index per bin over both cohorts, iterations and folds -> (N_OUTER x bins)."""
    cs = []
    for c in COHORTS:
        d = pd.read_parquet(ART / f"002_pseudo_{target}_{c}.parquet", columns=["area", "c_index", "skipped_reason"])
        d = d[(d.area == area) & d.skipped_reason.isna()]
        cs += [np.stack([np.asarray(f) for f in ci]) for ci in d.c_index]
    A = np.concatenate(cs)                                    # (iterations*folds) x bins
    mode = np.array([Counter(A[:, j].tolist()).most_common(1)[0][0] for j in range(A.shape[1])])
    return np.tile(mode, (P.N_OUTER, 1))


def group_curve(views_list, n_it, rng, C_grid, Cfix):
    curves = []
    for _ in range(n_it):
        mice = rng.choice(len(views_list), P.N_MICE, replace=True)
        draw = [(views_list[m], rng.choice(views_list[m].units, P.N_NEURONS, replace=True)) for m in mice]
        parts = P.make_parts(draw, rng, shift=False)
        if parts is None:
            continue
        curves.append(P.decode(parts, rng, C_grid, fixed=Cfix)[0])
    return (np.mean(curves, 0) if curves else None), len(curves)


def task(args):
    area_col, area, todo, cfix, n_it = args
    sys.path.insert(0, str(ROOT_REPO / "scripts"))
    os.chdir(ROOT_REPO)
    from ssl_timeresolved_decoding import C_GRID
    C_grid = np.asarray(C_GRID, float)
    V = {c: P.load_cohort_area(c, area_col, area) for c in COHORTS}
    out = {t: [] for t in todo}
    for t, perms in todo.items():
        va, vb = [V["R+"][t][s] for s in sorted(V["R+"][t])], [V["R-"][t][s] for s in sorted(V["R-"][t])]
        pooled, na = va + vb, len(va)
        for p in perms:
            rl = np.random.default_rng(zlib.crc32(f"{t}|{area}|perm|{p}".encode()))
            idx = np.arange(len(pooled)) if p == 0 else rl.permutation(len(pooled))
            ga, gb = [pooled[i] for i in idx[:na]], [pooled[i] for i in idx[na:]]
            rng = np.random.default_rng(zlib.crc32(f"{t}|{area}|perm|{p}|draws".encode()))
            ca, ka = group_curve(ga, n_it, rng, C_grid, cfix[t])
            cb, kb = group_curve(gb, n_it, rng, C_grid, cfix[t])
            out[t].append(dict(target=t, area_col=area_col, area=area, perm=p, n_a=na, n_b=len(gb), n_it=n_it,
                               n_it_ok_a=ka, n_it_ok_b=kb, curve_a=None if ca is None else ca.tolist(),
                               curve_b=None if cb is None else cb.tolist(), c_fixed=cfix[t][0].tolist(),
                               n_mice=P.N_MICE, n_neurons=P.N_NEURONS, shifts="none (user)",
                               labels="true (R+ = a)" if p == 0 else "permuted at mouse level"))
    return out


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_it = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    areas_arg = sys.argv[3] if len(sys.argv) > 3 else "all"
    os.chdir(ROOT_REPO)
    # areas decoded (not skipped) in both cohorts, per target
    ok = {}
    for t in P.TARGETS:
        s = [set(pd.read_parquet(ART / f"002_pseudo_{t}_{c}.parquet", columns=["area", "skipped_reason"])
                 .query("skipped_reason.isna()", engine="python").area) for c in COHORTS]
        ok[t] = s[0] & s[1]
    areas = sorted(set().union(*ok.values()), key=lambda a: (a != "All units", a))
    if areas_arg != "all":
        areas = [a for a in areas if a in areas_arg.split(",") or (a == "All units" and "whole_brain" in areas_arg)]
    paths = {t: ART / f"007_cohort_perm_{t}.parquet" for t in P.TARGETS}
    done = {t: set() for t in P.TARGETS}
    for t, pth in paths.items():
        if pth.exists():
            d = pd.read_parquet(pth, columns=["area", "perm"])
            done[t] = set(zip(d.area, d.perm))
    tasks = []
    for a in areas:
        cfix = {t: fixed_c(t, a) for t in P.TARGETS if a in ok[t]}
        for c0 in range(0, n_perm + 1, CHUNK):
            todo = {t: [p for p in range(c0, min(c0 + CHUNK, n_perm + 1)) if (a, p) not in done[t]] for t in cfix}
            todo = {t: v for t, v in todo.items() if v}
            if todo:
                tasks.append(("whole_brain" if a == "All units" else "area_group", a, todo, cfix, n_it))
    print(f"[007] {len(tasks)} tasks, {len(areas)} areas, {n_perm} permutations (+ observed) x {n_it} iterations per group, "
          f"{P.N_MICE} mice x {P.N_NEURONS} neurons, no shifts, fixed C, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=P._init) as ex:
        futs = {ex.submit(task, a): a for a in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            a = futs[f]
            try:
                res = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[007] ERROR {a[1]}: {e!r}", flush=True)
                continue
            for t, rows in res.items():
                if rows:
                    new = pd.DataFrame(rows)
                    out = pd.concat([pd.read_parquet(paths[t]), new], ignore_index=True) if paths[t].exists() else new
                    out.to_parquet(paths[t], index=False)
            print(f"[007] [{i}/{len(tasks)}] {a[1]}: " + ", ".join(f"{t} {len(r)}" for t, r in res.items())
                  + f" -- {time.time() - t0:.0f}s", flush=True)
    print("[007] DONE", flush=True)


if __name__ == "__main__":
    main()
