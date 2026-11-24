"""v3 baseline: within- and across-area spectral coupling in quiet (no-lick) time, all stable units (user 2026-11-18).

Quiet time: quiet_lib stretches (realised quiet window, 0.5 s after the last lick dropped; correct rejections add their
response window), cut into 1-s segments (never concatenated). Segments matched across sessions: N_SEG per condition,
drawn evenly over 5 session-phase bins (R+ quiet time sits late in the session, quiescence 008).
Conditions: pooled (all segments), prestim (segments of non-CR trials), cr (segments of CR trials).
Signal: 5-ms counts of stable good + mua units (rate >= 1 Hz), each unit divided by (SD + soft); each segment's mean
removed in the tapered FFT (multitaper NW = 3, K = 5, 0.78-Hz grid, <= 100 Hz).
Per area (area group and area acronym), from the unit FFTs:
  sums over all units, 10 random half splits, and N_MATCH-unit subsets (5 draws, split in halves for within).
Measures (each with a segment-shuffle null, 10 permutations that never pair a segment with one of its own trial):
  across pair A-B   msc (all-unit sums), msc_matched (N_MATCH-unit sums), pcoh (all-unit sums given the sums of every
                    other area), popcoh = |mean pairwise unit cross-spectrum| / mean unit auto-spectrum (independent
                    of the number of units), imc
  within area A     msc_within / pcoh_within / popcoh_within between two disjoint random halves (all units), and
                    msc_within_matched (halves of N_MATCH units)
  per area          power of the all-unit sum / unit-power sum (synchrony spectrum), rate
No multiple-comparison correction here; statistics in 010.
Outputs: OUT/v3_baseline/sessions/<session>_{bands,spectra}.parquet + _meta.json
Run (haas, repo root): OMP_NUM_THREADS=1 nice -n 10 ~/code/unit_spikes_analysis/.venv/bin/python \
    projects/ssl-spectral-coupling/009_baseline_within.py [--n-proc 45] [--sessions S ...] [--quick]
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quiet_lib as ql  # noqa: E402
import spec_lib as sl  # noqa: E402

OUT = sl.OUT / "v3_baseline"
SEG, NW, SKIP = 1.0, 3.0, 0.5
NB = int(round(SEG / sl.BIN))
NFFT = 256
N_SEG = 250
N_PHASE = 5
N_SPLIT, N_SUB, N_SH = 10, 5, 10
N_MATCH = {"group": 30, "acronym": 12}
MIN_UNITS = {"group": 10, "acronym": 6}
MIN_RATE = 1.0
CONFIG = dict(version=3, seg_s=SEG, NW=NW, skip_after_lick_s=SKIP, nfft=NFFT, n_seg=N_SEG, n_phase=N_PHASE,
              n_split=N_SPLIT, n_sub=N_SUB, n_shuffle=N_SH, n_match=N_MATCH, min_units=MIN_UNITS, min_rate_hz=MIN_RATE,
              bands=sl.BANDS, soft_sd=sl.SOFT_SD, conditions=["pooled", "prestim", "cr"])


def session_table():
    spec = importlib.util.spec_from_file_location("run002", HERE / "002_run_spikes.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.session_table()[0]


def pick_segments(seg, st, rng, mask):
    """N_SEG segments of `mask`, evenly over N_PHASE session-phase bins (fewer where a bin runs short)."""
    s = seg[mask].copy()
    t0, t1 = st.start_time.min(), st.start_time.max()
    s["phase"] = np.minimum(((s.t0 - t0) / (t1 - t0 + 1e-9) * N_PHASE).astype(int).clip(0, N_PHASE - 1), N_PHASE - 1)
    per = N_SEG // N_PHASE
    out = [g.sample(min(len(g), per), random_state=int(rng.integers(1 << 31))) for _, g in s.groupby("phase")]
    out = pd.concat(out) if out else s.iloc[:0]
    short = N_SEG - len(out)
    if short > 0:  # top up from the remaining segments
        rest = s.drop(out.index)
        out = pd.concat([out, rest.sample(min(short, len(rest)), random_state=int(rng.integers(1 << 31)))])
    return out.sort_values("t0")


def seg_counts(spike_times, t0):
    """[segment, bin, unit] counts in NB 5-ms bins from each t0."""
    edges = t0[:, None] + np.arange(NB + 1)[None, :] * sl.BIN
    X = np.zeros((len(t0), NB, len(spike_times)), np.float32)
    for u, st in enumerate(spike_times):
        X[:, :, u] = np.diff(np.searchsorted(st, edges), axis=1)
    return X


def unit_fft(X):
    """[segment, taper, freq, unit] tapered FFT of each unit (segment mean removed), f in (0, 100]."""
    Y = X - X.mean(1, keepdims=True)
    tp = sl.tapers(NB, NW)
    F = np.fft.rfft(Y[:, None, :, :] * tp[None, :, :, None], n=NFFT, axis=2)
    f = np.fft.rfftfreq(NFFT, sl.BIN)
    keep = (f > 0) & (f <= sl.FMAX + 1e-9)
    return F[:, :, keep, :].astype(np.complex64), f[keep]


def xspec(A, B):
    """<A conj(B)> over segments and tapers [freq]; A, B [segment, taper, freq]."""
    return (A * np.conj(B)).mean((0, 1))


def coh2(A, B):
    return np.abs(xspec(A, B)) ** 2 / (xspec(A, A).real * xspec(B, B).real + 1e-30)


def pcoh(A, B, Z):
    """squared partial coherence of A, B given channels Z [segment, taper, freq, k] (k may be 0)."""
    if Z is None or Z.shape[3] == 0:
        return coh2(A, B)
    V = np.concatenate([A[..., None], B[..., None], Z], 3)
    W = np.ascontiguousarray(np.transpose(V, (2, 0, 1, 3))).reshape(V.shape[2], -1, V.shape[3])
    S = np.matmul(np.transpose(W, (0, 2, 1)), W.conj()) / W.shape[1]
    G = np.linalg.inv(S + np.eye(S.shape[1])[None] * 1e-9 * np.abs(S).max())
    return np.abs(G[:, 0, 1]) ** 2 / (G[:, 0, 0].real * G[:, 1, 1].real)


def popcoh(SA, SB, PA, PB, nA, nB):
    """|mean pairwise unit cross-spectrum| / sqrt(mean unit auto-spectra): SA, SB sums [seg, taper, f], PA, PB
    unit-power sums [f]."""
    return np.abs(xspec(SA, SB)) / (nA * nB) / np.sqrt((PA / nA) * (PB / nB) + 1e-30)


def derange(trial, rng):
    """permutation of segments that never pairs a segment with one of the same trial."""
    n = len(trial)
    p = rng.permutation(n)
    for _ in range(1000):
        bad = np.flatnonzero(trial[p] == trial)
        if not len(bad):
            return p
        i, j = bad[0], rng.integers(n)  # swap a clash with a random position
        p[[i, j]] = p[[j, i]]
    raise RuntimeError("no derangement across trials")


def run_session(args):
    sid, cohort, learner, quick, levels = args
    t0 = time.time()
    rng = np.random.default_rng(zlib.crc32(sid.encode()) + 3)
    units, _, rule = sl.load_session(sid)
    tb, licks = ql.read_behaviour(sid)
    st = ql.stretches(tb, licks, skip=SKIP)
    seg = ql.segments(st, SEG)
    seg["trial_start"] = st.start_time.to_numpy()[seg.trial.to_numpy()]
    spikes = [np.sort(np.asarray(s, float)) for s in units.spike_times]
    span = st.start_time.max() - st.start_time.min()
    rate = np.array([((s >= st.start_time.min()) & (s <= st.start_time.max())).sum() / span for s in spikes])
    keep = rate >= MIN_RATE
    units = units[keep].reset_index(drop=True)
    spikes = [s for s, k in zip(spikes, keep) if k]
    rate = rate[keep]
    meta = dict(session_id=sid, mouse_id=sid.split("_")[0], cohort=cohort, learner=learner, unit_rule=rule,
                n_units=len(units), n_seg_available=dict(pooled=len(seg), prestim=int((~seg.cr).sum()),
                                                           cr=int(seg.cr.sum())), config=CONFIG)
    bands = sl.bands_for(SEG)
    rows, srows = [], []
    conds = {"pooled": np.ones(len(seg), bool), "prestim": ~seg.cr.to_numpy(), "cr": seg.cr.to_numpy()}
    if quick:
        conds = {"pooled": conds["pooled"]}
    for cond, mask in conds.items():
        S = pick_segments(seg, st, rng, mask)
        meta[f"n_seg_{cond}"] = len(S)
        if len(S) < 50:
            continue
        trial = S.trial.to_numpy()
        perms = [np.arange(len(S))] + [derange(trial, rng) for _ in range(2 if quick else N_SH)]
        X = seg_counts(spikes, S.t0.to_numpy())
        X /= X.std((0, 1), keepdims=True) + sl.SOFT_SD
        F, f = unit_fft(X)
        del X
        Pu = (np.abs(F) ** 2).mean((0, 1))  # [f, unit]
        for level in levels:
            lab = (units.area_group if level == "group" else
                   units.area_acronym_custom.where(units.area_group.notna())).to_numpy(object)
            members = {g: np.flatnonzero(lab == g) for g in pd.unique(lab[pd.notna(lab)])}
            areas = sorted(g for g, m in members.items() if len(m) >= MIN_UNITS[level])
            meta[f"n_units_{level}"] = {g: int(len(m)) for g, m in members.items()}
            A = {}
            for g in areas:
                m = members[g]
                d = dict(n=len(m), sum=F[..., m].sum(3), P=Pu[:, m].sum(1), rate=float(rate[m].mean()))
                d["halves"] = []
                for _ in range(N_SPLIT):
                    p = rng.permutation(m)
                    h1, h2 = p[: len(m) // 2], p[len(m) // 2: 2 * (len(m) // 2)]
                    d["halves"].append((F[..., h1].sum(3), F[..., h2].sum(3), Pu[:, h1].sum(1), Pu[:, h2].sum(1),
                                        len(h1), len(h2)))
                nm = N_MATCH[level]
                d["subs"] = []
                if len(m) >= nm:
                    for _ in range(N_SUB):
                        p = rng.permutation(m)[:nm]
                        d["subs"].append((F[..., p].sum(3), F[..., p[: nm // 2]].sum(3), F[..., p[nm // 2:]].sum(3)))
                A[g] = d

            def add(kind, a, b, measure, vals):
                """vals [perm, freq] -> band rows (observed, null mean, null sd) and a 2-Hz grid spectrum."""
                vals = np.asarray(vals)
                bm = np.array([[sl.band_means(v, f, bands)[k] for k in bands] for v in vals])
                for j, band in enumerate(bands):
                    rows.append(dict(level=level, condition=cond, kind=kind, source=a, target=b, measure=measure,
                                     band=band, value=bm[0, j], null_mean=np.nanmean(bm[1:, j]),
                                     null_sd=np.nanstd(bm[1:, j]), n_units_a=A[a]["n"], n_units_b=A[b]["n"],
                                     n_seg=len(S)))
                if level == "group":
                    obs, nul = sl.to_grid(vals[0], f), sl.to_grid(np.nanmean(vals[1:], 0), f)
                    for fq, o, n_ in zip(sl.SPEC_GRID, obs, nul):
                        if np.isfinite(o):
                            srows.append(dict(level=level, condition=cond, kind=kind, source=a, target=b,
                                              measure=measure, freq=float(fq), value=float(o), null_mean=float(n_)))

            for g in areas:  # per-area descriptors (no null)
                d = A[g]
                sync = (np.abs(d["sum"]) ** 2).mean((0, 1)) / (d["P"] + 1e-30)
                sync = (sync - 1) / max(d["n"] - 1, 1)  # mean pairwise cross / mean auto (real), N-independent
                for k, v in sl.band_means(sync, f, bands).items():
                    rows.append(dict(level=level, condition=cond, kind="area", source=g, target=g, measure="sync",
                                     band=k, value=v, null_mean=np.nan, null_sd=np.nan, n_units_a=d["n"],
                                     n_units_b=d["n"], n_seg=len(S)))
                rows.append(dict(level=level, condition=cond, kind="area", source=g, target=g, measure="rate",
                                 band="all", value=d["rate"], null_mean=np.nan, null_sd=np.nan, n_units_a=d["n"],
                                 n_units_b=d["n"], n_seg=len(S)))
                # within area: two disjoint halves
                others = [h for h in areas if h != g]
                Z = np.stack([A[h]["sum"] for h in others], 3) if others else None
                res = {k: [] for k in ("msc_within", "pcoh_within", "popcoh_within", "msc_within_matched")}
                for p in perms:
                    v = {k: [] for k in res}
                    for h1, h2, p1, p2, n1, n2 in d["halves"]:
                        v["msc_within"].append(coh2(h1[p], h2))
                        v["popcoh_within"].append(popcoh(h1[p], h2, p1, p2, n1, n2))
                    for h1, h2, p1, p2, n1, n2 in d["halves"][:3]:
                        v["pcoh_within"].append(pcoh(h1[p], h2, Z))
                    for _, s1, s2 in d["subs"]:
                        v["msc_within_matched"].append(coh2(s1[p], s2))
                    for k in res:
                        res[k].append(np.mean(v[k], 0) if v[k] else np.full(len(f), np.nan))
                for k, vals in res.items():
                    add("within", g, g, k, vals)
            for i, a in enumerate(areas):  # across pairs
                for b in areas[i + 1:]:
                    others = [h for h in areas if h not in (a, b)]
                    Z = np.stack([A[h]["sum"] for h in others], 3) if others else None
                    res = {k: [] for k in ("msc", "imc", "pcoh", "popcoh", "msc_matched")}
                    for p in perms:
                        Sa, Sb = A[a]["sum"][p], A[b]["sum"]
                        c = xspec(Sa, Sb) / np.sqrt(xspec(Sa, Sa).real * xspec(Sb, Sb).real + 1e-30)
                        res["msc"].append(np.abs(c) ** 2)
                        res["imc"].append(c.imag)
                        res["pcoh"].append(pcoh(Sa, Sb, Z))
                        res["popcoh"].append(popcoh(Sa, Sb, A[a]["P"], A[b]["P"], A[a]["n"], A[b]["n"]))
                        mm = [coh2(sa[p], sb) for (sa, _, _), (sb, _, _) in zip(A[a]["subs"], A[b]["subs"])]
                        res["msc_matched"].append(np.mean(mm, 0) if mm else np.full(len(f), np.nan))
                    for k, vals in res.items():
                        add("across", a, b, k, vals)
            print(sid, cond, level, len(areas), "areas", f"{time.time() - t0:.0f} s", flush=True)
        del F
    od = OUT / "sessions"
    od.mkdir(parents=True, exist_ok=True)
    sfx = "_quick" if quick else ""
    for name, rr in (("bands", rows), ("spectra", srows)):
        d = pd.DataFrame(rr)
        d.insert(0, "session_id", sid)
        d.insert(1, "mouse_id", meta["mouse_id"])
        d.insert(2, "cohort", cohort)
        d.insert(3, "learner", learner)
        tmp = od / f"{sid}{sfx}_{name}.tmp.parquet"
        d.to_parquet(tmp)
        os.replace(tmp, od / f"{sid}{sfx}_{name}.parquet")
    meta["runtime_s"] = round(time.time() - t0)
    json.dump(meta, open(od / f"{sid}{sfx}_meta.json", "w"), indent=1, default=str)
    return sid, len(rows), meta["runtime_s"]


def run_safe(job):
    try:
        return run_session(job)
    except Exception as err:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return job[0], f"ERROR {type(err).__name__}: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--levels", nargs="*", default=["group", "acronym"])
    ap.add_argument("--n-proc", type=int, default=45)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    s = session_table()
    if args.sessions:
        s = s[s.session_id.isin(args.sessions)]
    done = {p.name.replace("_meta.json", "") for p in (OUT / "sessions").glob("*_meta.json")} if not args.quick else set()
    jobs = [(r.session_id, r.cohort, bool(r.learner), args.quick, args.levels) for r in s.itertuples()
            if r.session_id not in done]
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_csv(OUT / "sessions_included.csv", index=False)
    json.dump(CONFIG, open(OUT / "config.json", "w"), indent=1, default=str)
    print(f"{len(jobs)} sessions to run ({len(done)} done)", flush=True)
    with Pool(min(args.n_proc, max(1, len(jobs)))) as pool:
        for res in pool.imap_unordered(run_safe, jobs):
            print("done", res, flush=True)


if __name__ == "__main__":
    main()
