"""016 -- What soft normalisation does to the correlation readout: one example session (user 2026-11-02).

The 009 correlation readout z-scores every unit, z_u = (x_u - mu_u) / sd_u, so a sparse unit (tiny sd in a 30-ms window)
jumps to z ~ 10-20 on a single spike and can dominate the Pearson r across ~1000 units. Soft normalisation divides by
sd_u + c instead (c = SOFT_FRACS x median sd over units): units with sd >> c are unchanged, sparse units are down-weighted in
proportion to sd_u / c. The template is rescaled the same way (template_u = delta mean count_u / (sd_u + c)), so only the unit
weighting changes. Also shown: a movement-partial correlation r(z, template | movement pattern), the movement pattern =
mean soft-z count of whisking minus still windows (valid, outside the evoked tail), partialled out across units per window.

Data: 009 v7 files of the session (stable units, whole brain), spikes from 006 prepare (as 009 / 013). Shuffles: N_SHUF
template permutations within the 009 rate strata; threshold = median over the shuffles of their 99.9th percentile (009).
Panels:
  a  unit s.d. of the 30-ms count vs mean count (log-log), with each c
  b  relative weight of a unit in r (1 / (sd + c), normalised to the median unit) vs mean count, per c
  c  concentration at event peaks: share of the positive contribution to r (or to the decision value) from the top 5 units
  d  30-s stretch: decoder, correlation (c = 0), soft correlations, movement-partial soft correlation, each with the 2.5-97.5 %
     band of the shuffles and the threshold; whisker speed, licks
  e  autocorrelation of each readout over valid time (timescale of the fluctuations)
  f  events / min vs threshold (in shuffle s.d. above the shuffle median), real vs shuffles
  g  event rate at the threshold in still / whisking / all valid time, real vs median shuffle
  h  Spearman correlation of each readout with whisker speed (robust z), valid time outside the evoked tail
Output: figures/full_stable/analysis_016_softnorm/<session>_<target>.{png,pdf,svg} + _summary.csv
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m6 = importlib.import_module("006_fdr_threshold_heldout_controls")
m7 = importlib.import_module("007_content_threshold_participation")
m9 = importlib.import_module("009_full_dataset_events")

SOFT_FRACS = (0.0, 0.5, 1.0, 2.0)                # c = frac x median unit s.d. (0 = the 009 v7 correlation)
N_SHUF = 20
CHUNK = 40000
STRETCH_S = 30.0
TOPK = 5
COLS = {"dec": "k", "c0": "#8c510a", "c0.5": "#d95f02", "c1": "#1b9e77", "c2": "#7570b3", "c1|mv": "#e7298a"}


def name_of(f):
    return f"c{f:g}"


def readouts(spikes, t, G, Gr, consts, mv_pat, rng_seed):
    """decoder + correlation readouts (real and N_SHUF shuffles) for every c, chunked over the session"""
    w, mu, sd, b = G["w"], G["mu"], G["sd"], float(G["b"])
    mur, sdr = Gr["mu"], Gr.get("sd_raw", Gr["sd"])
    delta = Gr["template"] * Gr["sd"]                                       # delta mean count (saved template = delta / saved sd)
    st = m9.rate_strata(mu, m9.SHUF_STRATA)
    rng = np.random.default_rng(rng_seed)
    W = np.c_[w, np.stack([m7.perm_within(w, st, rng) for _ in range(N_SHUF)], 1)]
    TCs, sds = {}, {}
    pidx = [np.arange(len(w))] + [m7.perm_within(np.arange(len(w)), st, rng) for _ in range(N_SHUF)]
    for nm, c in consts.items():
        s_ = sdr + c
        tc = delta / s_                                                   # delta mean count / (sd + c)
        tc = tc - tc.mean()
        TCs[nm] = np.stack([tc[p] for p in pidx], 1)
        sds[nm] = s_
    out = {"dec": np.empty((len(t), N_SHUF + 1), np.float32)}
    for nm in consts:
        out[nm] = np.empty((len(t), N_SHUF + 1), np.float32)
    out["c1|mv"] = np.empty((len(t), N_SHUF + 1), np.float32)
    for c0 in range(0, len(t), CHUNK):
        ts = t[c0:c0 + CHUNK]
        X = np.stack([m1.counts(s, ts) for s in spikes], 1).astype(np.float32)
        out["dec"][c0:c0 + len(ts)] = X @ (W / sd[:, None]).astype(np.float32) + (b - (mu / sd) @ W).astype(np.float32)
        for nm in consts:
            s1, s2 = m9.unit_moments(X, mur, sds[nm])
            out[nm][c0:c0 + len(ts)] = m9.corr_readout(X, TCs[nm], mur, sds[nm], s1, s2)
            if nm == "c1":                                                # partial r(z, tc | movement pattern)
                rzm = m9.corr_readout(X, mv_pat[:, None], mur, sds[nm], s1, s2)[:, 0]
                tcs = TCs[nm]
                rtm = (tcs - tcs.mean(0)).T @ (mv_pat - mv_pat.mean()) / (np.linalg.norm(tcs - tcs.mean(0), axis=0) * np.linalg.norm(mv_pat - mv_pat.mean()))
                r = out[nm][c0:c0 + len(ts)]
                out["c1|mv"][c0:c0 + len(ts)] = (r - rzm[:, None] * rtm[None, :]) / np.sqrt(np.maximum((1 - rzm[:, None] ** 2) * (1 - rtm[None, :] ** 2), 1e-12))
        print(f"  chunk {c0 // CHUNK + 1} / {int(np.ceil(len(t) / CHUNK))}", flush=True)
    return out, TCs, sds


def movement_pattern(spikes, t, S, mur, sdr_c):
    """mean count in whisking minus still windows (valid, outside the evoked tail) / (sd + c)"""
    base = S["valid"] & ~S["evoked"]
    mw, ms = base & S["whisk"], base & S["still"]
    acc_w, acc_s = np.zeros(len(spikes)), np.zeros(len(spikes))
    for c0 in range(0, len(t), CHUNK):
        sl = slice(c0, c0 + CHUNK)
        X = np.stack([m1.counts(s, t[sl]) for s in spikes], 1).astype(np.float32)
        acc_w += X[mw[sl]].sum(0)
        acc_s += X[ms[sl]].sum(0)
    return (acc_w / max(mw.sum(), 1) - acc_s / max(ms.sum(), 1)) / sdr_c, int(mw.sum()), int(ms.sum())


def share_top(X, mu, s_, tc):
    """share of the positive per-unit contributions z_u tc_u held by the TOPK largest, per row of X"""
    c = ((X - mu) / s_) * tc
    c = np.clip(c, 0, None)
    srt = -np.sort(-c, 1)
    return srt[:, :TOPK].sum(1) / np.maximum(c.sum(1), 1e-12)


def main(sid, target, unit_set):
    plt = m2.setup()
    out = m9.ROOT / unit_set / sid
    G = dict(np.load(out / f"g_All_{target}.npz", allow_pickle=True))
    Gr = dict(np.load(out / f"g_All_{target}_corr.npz", allow_pickle=True))
    S = dict(np.load(out / "session.npz"))
    B = dict(np.load(out / "behaviour.npz", allow_pickle=True))
    t, valid = S["t"], S["valid"]
    D = m6.prepare(sid)
    keep = m9.unit_set_mask(sid, D["units"], unit_set)
    sp_all = [s for s, k in zip(D["spikes"], keep) if k]
    spikes = [np.asarray(sp_all[i]) for i in G["unit_index"]]
    mur, sdr = Gr["mu"], Gr.get("sd_raw", Gr["sd"])                          # raw unit s.d. (009 >= soft: sd_raw)
    med_sd = float(np.median(sdr))
    consts = {name_of(f): f * med_sd for f in SOFT_FRACS}
    mv_pat, n_w, n_s = movement_pattern(spikes, t, S, mur, sdr + consts["c1"])
    R, TCs, sds = readouts(spikes, t, G, Gr, consts, mv_pat, zlib.crc32(f"{sid}|{target}|016".encode()))
    # sanity: the c of the saved 009 correlation readout reproduces it, the decoder the saved decision value
    c_saved = name_of(Gr["soft_c"] / med_sd) if "soft_c" in Gr else "c0"
    err_c = float(np.nanmax(np.abs(R[c_saved][:, 0] - Gr["z"]))) if c_saved in R else np.nan
    err_d = float(np.nanmax(np.abs(R["dec"][:, 0] - G["z"])))
    print(f"{sid} {target}: reconstruction max |err| decoder {err_d:.2e}, correlation {c_saved} {err_c:.2e}", flush=True)
    base = valid & ~S["evoked"]
    rows, ev_idx = [], {}
    for nm, Z in R.items():
        thr = float(np.median([np.percentile(Z[valid, j], 99.9) for j in range(1, N_SHUF + 1)]))
        E = m1.detect(Z[:, 0], valid, thr)
        ev_idx[nm] = E[:, 2]
        d = dict(readout=nm, c=consts.get(nm, consts["c1"] if nm == "c1|mv" else np.nan), thr=thr)
        for mk, m in (("all", valid), ("still", valid & S["still"]), ("whisk", valid & S["whisk"])):
            sec = m.sum() * m1.STEP
            d[f"rate_{mk}"] = np.isin(E[:, 2], np.flatnonzero(m)).sum() / sec if sec else np.nan
            d[f"rate_shuf_{mk}"] = float(np.median([np.isin(m1.detect(Z[:, j], valid, thr)[:, 2], np.flatnonzero(m)).sum() / sec
                                                    for j in range(1, N_SHUF + 1)])) if sec else np.nan
        ok = base & np.isfinite(S["wz"])
        ok[np.arange(len(ok)) % 10 != 0] = False
        from scipy import stats
        d["rho_whisk"] = stats.spearmanr(Z[ok, 0], S["wz"][ok])[0]
        x = np.where(valid, Z[:, 0] - np.median(Z[valid, 0]), 0.0)
        lags = np.arange(1, 401, 2)
        d["acf"] = np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in lags])
        rows.append(d)
    # concentration at event peaks (first 300 events of each readout), recomputed from the spikes
    rng = np.random.default_rng(0)
    conc = {}
    for nm in ["dec"] + list(consts):
        pk = ev_idx[nm]
        pk = pk if len(pk) <= 300 else np.sort(rng.choice(pk, 300, replace=False))
        if not len(pk):
            conc[nm] = np.array([])
            continue
        X = np.stack([m1.counts(s, t[pk]) for s in spikes], 1).astype(np.float64)
        if nm == "dec":
            conc[nm] = share_top(X, G["mu"], G["sd"], G["w"])
        else:
            conc[nm] = share_top(X, mur, sds[nm], TCs[nm][:, 0])
    # ---- figure
    fig = plt.figure(figsize=(7.4, 9.6), layout="constrained")
    sf = fig.subfigures(3, 1, height_ratios=[1.0, 2.2, 1.1])
    ax = sf[0].subplots(1, 3)
    a = ax[0]
    a.loglog(np.maximum(mur, 1e-5), np.maximum(sdr, 1e-5), ".", ms=1.5, color="0.4")
    for f in SOFT_FRACS[1:]:
        a.axhline(f * med_sd, color=COLS[name_of(f)], lw=0.8, label=f"c = {f:g} x median s.d.")
    a.set_xlabel("mean count / 30 ms")
    a.set_ylabel("s.d. of count / 30 ms")
    a.legend(fontsize=4.5, frameon=False)
    a.set_title("a  unit s.d. vs rate", fontsize=6, loc="left")
    a = ax[1]
    o = np.argsort(mur)
    for nm, c in consts.items():
        g = 1 / (sdr + c)
        a.loglog(np.maximum(mur[o], 1e-5), (g / np.median(g))[o], color=COLS[nm], lw=0.8, label=nm.replace("c", "c = ") + " x med")
    a.set_xlabel("mean count / 30 ms")
    a.set_ylabel("weight of a spike in r\n(rel. to median unit)")
    a.legend(fontsize=4.5, frameon=False)
    a.set_title("b  weight of one spike, by unit rate", fontsize=6, loc="left")
    a = ax[2]
    ks = [k for k in ["dec"] + list(consts) if len(conc[k])]
    a.boxplot([conc[k] for k in ks], showfliers=False, widths=0.6)
    for i, k in enumerate(ks):
        a.plot(i + 1 + rng.uniform(-0.15, 0.15, len(conc[k])), conc[k], ".", ms=1, color=COLS[k], alpha=0.5)
    a.set_xticks(range(1, len(ks) + 1), [{"dec": "decoder"}.get(k, k) for k in ks], fontsize=5)
    a.set_ylabel(f"share of positive contribution\nfrom the top {TOPK} units")
    a.set_ylim(0, 1.02)
    a.set_title("c  concentration at event peaks", fontsize=6, loc="left")
    # stretch: still + whisking, choose the 30-s valid task stretch with the most corr (c=0) events and some whisking
    win = int(STRETCH_S / m1.STEP)
    task = S["epoch"] == 2
    score = np.convolve(np.isin(np.arange(len(t)), ev_idx["c0"]).astype(float), np.ones(win), "same") \
        * (np.convolve(S["whisk"].astype(float), np.ones(win), "same") > win * 0.1) * (np.convolve(task.astype(float), np.ones(win), "same") > win * 0.95)
    ic = int(np.argmax(score))
    i0, i1 = max(ic - win // 2, 0), min(ic + win // 2, len(t))
    ts = t[i0:i1] - t[i0]
    order = ["dec", "c0", "c1", "c2", "c1|mv"]
    ax = sf[1].subplots(len(order) + 1, 1, sharex=True, height_ratios=[1] * len(order) + [0.7])
    for a, nm in zip(ax, order):
        Z = R[nm][i0:i1]
        d = next(r for r in rows if r["readout"] == nm)
        lo, hi = np.percentile(Z[:, 1:], [2.5, 97.5], axis=1)
        a.fill_between(ts, lo, hi, color="0.85", lw=0)
        a.plot(ts, Z[:, 0], color=COLS[nm], lw=0.5)
        a.axhline(d["thr"], color=COLS[nm], ls="--", lw=0.6)
        pk = ev_idx[nm][(ev_idx[nm] >= i0) & (ev_idx[nm] < i1)]
        a.plot(t[pk] - t[i0], np.full(len(pk), np.nanmax(Z[:, 0]) * 1.05), "v", ms=2.5, color=COLS[nm])
        inv = ~valid[i0:i1]
        a.fill_between(ts, 0, 1, where=S["still"][i0:i1], transform=a.get_xaxis_transform(), color="#fee391", alpha=0.35, lw=0)
        a.fill_between(ts, 0, 1, where=inv, transform=a.get_xaxis_transform(), color="0.6", alpha=0.35, lw=0)
        lab = {"dec": "decoder\n(decision value)", "c0": "corr. c = 0\n(009 v7, r)", "c1": "soft corr.\nc = 1 x med (r)",
               "c2": "soft corr.\nc = 2 x med (r)", "c1|mv": "soft corr. c = 1,\npartial | movement"}[nm]
        a.set_ylabel(lab, fontsize=5)
        a.text(1.0, 1.0, f"{d['rate_all']:.3f}/s (shuf {d['rate_shuf_all']:.3f}); still {d['rate_still']:.3f} (shuf {d['rate_shuf_still']:.3f})",
               transform=a.transAxes, ha="right", va="bottom", fontsize=4.5)
    a = ax[-1]
    wt, wv = B["whisker_speed_t"], B["whisker_speed_v"]
    mm = (wt >= t[i0]) & (wt < t[i1 - 1])
    a.plot(wt[mm] - t[i0], wv[mm], color="0.3", lw=0.4)
    lk = B["licks"][(B["licks"] >= t[i0]) & (B["licks"] < t[i1 - 1])] - t[i0]
    a.plot(lk, np.full(len(lk), np.nanmax(wv[mm]) if mm.any() else 1), "|", color="#ff7f00", ms=4)
    a.set_ylabel("whisker speed\n(deg/s), licks", fontsize=5)
    a.set_xlabel(f"time (s) from {t[i0] / 60:.2f} min")
    sf[1].suptitle("d  30-s task stretch: readouts (grey band = 2.5-97.5 % of 20 template shuffles, dashed = threshold, triangles = events; "
                   "gold = still, grey = invalid)", fontsize=5.5, x=0.01, ha="left")
    ax = sf[2].subplots(1, 3)
    a = ax[0]
    lags = np.arange(1, 401, 2) * m1.STEP
    for r in rows:
        a.plot(lags, r["acf"], color=COLS[r["readout"]], lw=0.8, label={"dec": "decoder"}.get(r["readout"], r["readout"]))
    a.axhline(1 / np.e, color="0.6", ls=":", lw=0.6)
    a.set_xscale("log")
    a.set_xlabel("lag (s)")
    a.set_ylabel("autocorrelation")
    a.legend(fontsize=4.5, frameon=False)
    a.set_title("e  timescale of the readout", fontsize=6, loc="left")
    a = ax[1]
    names = [r["readout"] for r in rows]
    x = np.arange(len(rows))
    for j, (mk, off) in enumerate((("all", -0.25), ("still", 0.0), ("whisk", 0.25))):
        real = [r[f"rate_{mk}"] for r in rows]
        sh = [r[f"rate_shuf_{mk}"] for r in rows]
        a.bar(x + off, real, 0.22, color=["0.2", "#d4b106", "#3182bd"][j], label=f"{mk} real")
        a.plot(x + off, sh, "_", color="r", ms=6, mew=1.2, label="median shuffle" if j == 0 else None)
    a.set_yscale("log")
    a.set_xticks(x, [{"dec": "decoder"}.get(n, n) for n in names], fontsize=5)
    a.set_ylabel("events / s at threshold")
    a.legend(fontsize=4.5, frameon=False)
    a.set_title("f  real vs shuffle, by state", fontsize=6, loc="left")
    a = ax[2]
    a.bar(x, [r["rho_whisk"] for r in rows], color=[COLS[n] for n in names])
    a.set_xticks(x, [{"dec": "decoder"}.get(n, n) for n in names], fontsize=5)
    a.set_ylabel("Spearman rho with\nwhisker speed")
    a.set_title("g  coupling to whisking", fontsize=6, loc="left")
    fig.suptitle(f"016  soft normalisation of the correlation readout -- {sid}, {target} target, {len(spikes)} {unit_set} units. Correlation "
                 f"z_u = (x_u - mu_u) / (sd_u + c), c in multiples of the median unit s.d. ({med_sd:.3f} counts / 30 ms); template rescaled the same "
                 f"way; partial | movement: whisking-minus-still pattern ({n_w * m1.STEP:.0f} s whisking, {n_s * m1.STEP:.0f} s still) partialled out "
                 f"across units; thresholds = median over {N_SHUF} template shuffles of the 99.9th percentile", fontsize=5.5, x=0.01, ha="left",
                 wrap=True)
    fdir = m1.HOME / "figures" / f"full_{unit_set}" / "analysis_016_softnorm"
    fdir.mkdir(parents=True, exist_ok=True)
    m2.save(fig, fdir, f"{sid}_{target}")
    plt.close(fig)
    pd.DataFrame([{k: v for k, v in r.items() if k != "acf"} | dict(session_id=sid, target=target, med_sd=med_sd,
                                                                      conc_top5_median=float(np.median(conc[r["readout"]])) if r["readout"] in conc and len(conc[r["readout"]]) else np.nan,
                                                                      tau_s=float(lags[np.argmax(r["acf"] < 1 / np.e)]) if (r["acf"] < 1 / np.e).any() else np.nan)
                  for r in rows]).to_csv(fdir / f"{sid}_{target}_summary.csv", index=False)
    print("saved", fdir / f"{sid}_{target}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="AB147_20241212_151234")
    ap.add_argument("--target", default="whisker")
    ap.add_argument("--unit_set", default="stable")
    a = ap.parse_args()
    main(a.session, a.target, a.unit_set)
