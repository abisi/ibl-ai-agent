"""009 -- Mouse-level cohort-label permutation test, R+ vs R-, on WINDOWS only, WITH the paired linear-shift null
(user 2026-09-29: "only test specific windows with linear shifts (not each time bin): sensory 5-50 ms post stim, and
pre-lick 100 ms pre-lick"). Replaces the per-bin, no-shift 007 run (aborted; partial files *_aborted_perbin_noshift).

Per target x area (same pseudo-population decoding as 002, imported):
  - windows: hitmiss 5..50 ms and 5..100 ms (the user's two hit/miss windows; 5..100 contains 5..50, so both come from
    the same decodes), modality_stim 5..50 ms, modality_lick -100..0 ms; only the bins whose end lies in the target's
    widest window are decoded (bins labelled at their end, as everywhere);
  - pooled eligible sessions (one per mouse) of both cohorts; permutation 0 = true labels (a = R+, b = R-),
    permutations >= 1 = cohort labels shuffled across mice (group sizes kept);
  - per permutation and pseudo-group, N_IT iterations: N_MICE mice with replacement, N_NEURONS units per session with
    replacement, balanced-reuse pseudo-trials, 3-fold outer CV -> real curve; N_SHIFTS linear-shift decodes of the SAME
    draw -> null; d = real - mean(null) per bin; window value = mean d over the window's bins;
  - C fixed per bin at the 002 modal choice (pooled over cohorts, iterations, folds), identical for real and shift
    decodes, every permutation and both groups;
  - stored per permutation: mean over iterations of the window d (and of real and null) for each group.
Test: statistic = d_a - d_b per window; two-sided p = (1 + #{perm >= 1: |stat_p| >= |stat_0|}) / (1 + n_perm).
Output: ../artifacts/009_cohort_perm_windows.parquet (appendable; one row per target x area x window x permutation).
Run (haas, repo root): OMP_NUM_THREADS=1 ... python .../009_cohort_perm_windows.py [n_perm=200] [n_it=20] [areas|all]
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

os.environ.setdefault("SSL_PSEUDO_N_NEURONS", "10")
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
# SSL_PERM_OUT_TAG: separate output per process (whole brain runs in its own low-worker process: each whole-brain task
# holds both cohorts, ~27 GB; 11 concurrent ones ran haas out of memory on 2026-09-30); readers glob 009_cohort_perm_windows*.parquet
OUT = ART / f"009_cohort_perm_windows{os.environ.get('SSL_PERM_OUT_TAG', '')}.parquet"
COHORTS = ("R+", "R-")
WINDOWS = {"hitmiss": {"5-50ms": (0.005, 0.05), "5-100ms": (0.005, 0.1)},
           "modality_stim": {"5-50ms": (0.005, 0.05)},
           "modality_lick": {"-100-0ms": (-0.10, 0.0)}}
# SSL_PERM_TARGETS (comma list) restricts the targets (user 2026-09-30: whole brain + the abstract's windows only --
# hitmiss 5-100 ms (5-50 comes from the same bins) and modality_lick -100..0 ms)
if os.environ.get("SSL_PERM_TARGETS"):
    WINDOWS = {t: WINDOWS[t] for t in os.environ["SSL_PERM_TARGETS"].split(",")}
AREAS = ["All units", "Motor areas", "Frontal areas", "Somatosensory-orofacial", "Somatosensory-whisker", "Auditory areas",
         "Retrosplenial areas", "Posterior parietal areas", "Insular areas", "Hippocampus", "Striatum", "Pallidum",
         "Lateral septal complex", "Thalamus", "Midbrain"]      # areas kept in the comparisons (user exclusions)
N_WORKERS = int(os.environ.get("SSL_PERM_N_WORKERS", "60"))
CHUNK = {"All units": int(os.environ.get("SSL_PERM_CHUNK_WB", "50"))}   # whole brain: loading both cohorts ~35 min per task
CHUNK_DEFAULT = int(os.environ.get("SSL_PERM_CHUNK", "20"))


def bins_of(target):
    import ssl_timeresolved_decoding as T
    align = "lick" if target == "modality_lick" else "stim"
    t = np.array([e[1] for e in T.causal_bin_edges(P.WIN[align], bin_width=P.BIN_W, stride=P.STRIDE)])
    lo = min(w[0] for w in WINDOWS[target].values()) - 1e-9
    hi = max(w[1] for w in WINDOWS[target].values()) + 1e-9
    sel = np.where((t >= lo) & (t <= hi))[0]
    return sel, t[sel]


def fixed_c(target, area, sel):
    cs = []
    for c in COHORTS:
        d = pd.read_parquet(ART / f"002_pseudo_{target}_{c}.parquet", columns=["area", "c_index", "skipped_reason"])
        d = d[(d.area == area) & d.skipped_reason.isna()]
        cs += [np.stack([np.asarray(f) for f in ci]) for ci in d.c_index]
    A = np.concatenate(cs)[:, sel]
    mode = np.array([Counter(A[:, j].tolist()).most_common(1)[0][0] for j in range(A.shape[1])])
    return np.tile(mode, (P.N_OUTER, 1))


def sliced_parts(draw, rng, shift, sel):
    parts = P.make_parts(draw, rng, shift=shift)
    if parts is not None:
        for p in parts:
            p["R"] = p["R"][sel]
    return parts


def group_value(views_list, n_it, rng, C_grid, Cfix, sel):
    """Mean over iterations of the per-bin real, null and d curves (window bins only)."""
    R, N = [], []
    for _ in range(n_it):
        mice = rng.choice(len(views_list), P.N_MICE, replace=True)
        draw = [(views_list[m], rng.choice(views_list[m].units, P.N_NEURONS, replace=True)) for m in mice]
        parts = sliced_parts(draw, rng, False, sel)
        if parts is None:
            continue
        real = P.decode(parts, rng, C_grid, fixed=Cfix)[0]
        nulls = []
        for _ in range(P.N_SHIFTS):
            sp = sliced_parts(draw, rng, True, sel)
            if sp is not None:
                nulls.append(P.decode(sp, rng, C_grid, fixed=Cfix)[0])
        if nulls:
            R.append(real)
            N.append(np.mean(nulls, 0))
    if not R:
        return None
    return np.mean(R, 0), np.mean(N, 0), len(R)


def task(args):
    area_col, area, todo, n_it = args
    sys.path.insert(0, str(ROOT_REPO / "scripts"))
    os.chdir(ROOT_REPO)
    from ssl_timeresolved_decoding import C_GRID
    C_grid = np.asarray(C_GRID, float)
    V = {c: P.load_cohort_area(c, area_col, area) for c in COHORTS}
    rows = []
    for t, perms in todo.items():
        sel, tb = bins_of(t)
        Cfix = fixed_c(t, area, sel)
        va, vb = [V["R+"][t][s] for s in sorted(V["R+"][t])], [V["R-"][t][s] for s in sorted(V["R-"][t])]
        if len(va) < 2 or len(vb) < 2:
            continue
        pooled, na = va + vb, len(va)
        for p in perms:
            rl = np.random.default_rng(zlib.crc32(f"{t}|{area}|wperm|{p}".encode()))
            idx = np.arange(len(pooled)) if p == 0 else rl.permutation(len(pooled))
            ga, gb = [pooled[i] for i in idx[:na]], [pooled[i] for i in idx[na:]]
            rng = np.random.default_rng(zlib.crc32(f"{t}|{area}|wperm|{p}|draws".encode()))
            A = group_value(ga, n_it, rng, C_grid, Cfix, sel)
            B = group_value(gb, n_it, rng, C_grid, Cfix, sel)
            for wname, (w0, w1) in WINDOWS[t].items():
                m = (tb >= w0 - 1e-9) & (tb <= w1 + 1e-9)
                row = dict(target=t, area=area, window=wname, perm=p, n_a=na, n_b=len(gb), n_it=n_it,
                           n_shifts=P.N_SHIFTS, n_mice=P.N_MICE, n_neurons=P.N_NEURONS, n_bins=int(m.sum()),
                           labels="true (a = R+)" if p == 0 else "permuted at mouse level")
                for g, X in (("a", A), ("b", B)):
                    if X is None:
                        row.update({f"d_{g}": np.nan, f"real_{g}": np.nan, f"null_{g}": np.nan, f"n_it_ok_{g}": 0})
                    else:
                        row.update({f"d_{g}": float(np.mean(X[0][m] - X[1][m])), f"real_{g}": float(np.mean(X[0][m])),
                                    f"null_{g}": float(np.mean(X[1][m])), f"n_it_ok_{g}": X[2]})
                rows.append(row)
    return rows


def main():
    n_perm = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    n_it = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    areas = AREAS if len(sys.argv) <= 3 or sys.argv[3] == "all" else sys.argv[3].split(",")
    os.chdir(ROOT_REPO)
    done = set()
    if OUT.exists():
        d = pd.read_parquet(OUT, columns=["target", "area", "perm"])
        done = set(zip(d.target, d.area, d.perm))
    tasks = []
    for a in areas:
        ch = CHUNK.get(a, CHUNK_DEFAULT)
        for c0 in range(0, n_perm + 1, ch):
            todo = {t: [p for p in range(c0, min(c0 + ch, n_perm + 1)) if (t, a, p) not in done] for t in WINDOWS}
            todo = {t: v for t, v in todo.items() if v}
            if todo:
                tasks.append(("whole_brain" if a == "All units" else "area_group", a, todo, n_it))
    print(f"[009] {len(tasks)} tasks, {len(areas)} areas, {n_perm} permutations (+ observed) x {n_it} iterations per group, "
          f"{P.N_MICE} mice x {P.N_NEURONS} neurons, {P.N_SHIFTS} shifts, windows {WINDOWS}, {N_WORKERS} workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(N_WORKERS, initializer=P._init) as ex:
        futs = {ex.submit(task, a): a for a in tasks}
        for i, f in enumerate(as_completed(futs), 1):
            a = futs[f]
            try:
                rows = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[009] ERROR {a[1]}: {e!r}", flush=True)
                continue
            if rows:
                new = pd.DataFrame(rows)
                out = pd.concat([pd.read_parquet(OUT), new], ignore_index=True) if OUT.exists() else new
                out.to_parquet(OUT, index=False)
            print(f"[009] [{i}/{len(tasks)}] {a[1]}: {len(rows)} rows -- {time.time() - t0:.0f}s", flush=True)
    print("[009] DONE", flush=True)


if __name__ == "__main__":
    main()
