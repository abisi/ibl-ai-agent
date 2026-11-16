"""008 -- Threshold per decoder set by its sensitivity to held-out stimuli; excess over the rate-matched shuffled template;
continuous-detection figures with behaviour; event rates across cohorts (user 2026-10-29, after 007: the shuffled-template
rule put thresholds at 7-27 MAD, where held-out passive stimuli no longer cross, so the absence of events was uninformative).

Decoder: clean template (as 006), trained on quiet task trials only (passive pre and post held out, 006 "ho_both"), whisker
and auditory; one decoder per area group plus a whole-brain decoder ("All": every unit of the session).
Continuous readout as 006 (30-ms windows, 5-ms step; valid time outside -100..600 ms of trial onsets and shams and after the
first 30 s of the recording; quiet = > 1 s from licks and whisker speed < 200 deg/s), in median / MAD units of the readout.
Threshold: the value crossed by a fraction SENS of the held-out passive target stimuli (pre + post; peak window ending
10-100 ms after onset), for SENS in SENS_LEVELS (primary SENS_MAIN). Sessions with < MIN_HELD held-out stimuli: the same
quantile of the cross-validated (nested CV, out-of-fold) task target values (thr_source "cv").
At each threshold: event rate (all valid time, quiet, per epoch); rate-matched shuffled template (N_SHUF permutations of the
weights within N_STRATA firing-rate strata, each readout in its own median / MAD units): mean rate, excess = real - shuffled,
ratio, p = fraction of shuffled rates >= real; time-shift null (N_NULL per-unit circular shifts of the binned counts, 30-300 s);
identity p per event (007, rate-matched permuted weights), reported, not used as a gate. Held-out sensitivity at the
threshold for pre, post and pre + post separately, and the false-positive rate on shams and other-modality stimuli.
Behaviour: piezo licks (NWB piezo_lick_times), whisker speed and jaw distance (DLC). Lick-jaw sync check per session: the
cross-correlation of the lick rate and jaw speed (50-ms bins) must peak within +-JAW_LAG_MAX with r >= JAW_R_MIN, otherwise
the jaw trace is not shown.
Writes <HOME>/pilot/<session>/sensitivity/: decoders.csv (one row per decoder x SENS), heldout.csv, curves.parquet,
events_<target>_<group>.parquet (SENS_MAIN), continuous_<target>_<group>.npz, behaviour.npz, params.json; figures
fig14_continuous_<session>_<target> (60-s segments, every group, behaviour below) and fig15_examples_<session>_<target>
(event-centred). --summary: <HOME>/sens_decoders.csv, sens_heldout.csv, figures fig13_sens_real_vs_shuffled,
fig16_cohort_rates_<target>.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m5 = importlib.import_module("005_stim_threshold_examples")
m6 = importlib.import_module("006_fdr_threshold_heldout_controls")
m7 = importlib.import_module("007_content_threshold_participation")
SENS_LEVELS = (0.25, 0.5, 0.8)
SENS_MAIN = 0.5
MIN_HELD = 10
N_SHUF = 20
N_NULL = 10
P_GATE = 0.05
GOOD_PAUC = 0.8
JAW_LAG_MAX = 0.1                                # s
JAW_R_MIN = 0.05
SEG = 60.0                                       # s, continuous-detection segments
HALF = 1.0                                       # s around an example event
OUT = "sensitivity"
ALL = "All"
GROUPS = [ALL] + m1.GROUPS
EPOCHS = ("passive_pre", "task", "passive_post")
THR_C = np.r_[np.arange(1, 10, 0.25), np.arange(10, 30.5, 1.0)]
COH = {"R+": "#00B400", "R-": "#C800C8"}
STAGES = ["day 0 R+", "day 0 R-", "expert R+", "expert R-"]


def lick_jaw_sync(licks, dlc):
    if "jaw" not in dlc or len(licks) < 50:
        return np.nan, np.nan
    tj, jd = dlc["jaw"]
    sp = np.nan_to_num(np.abs(np.gradient(jd, tj)))
    b = np.arange(max(tj[0], licks[0]), min(tj[-1], licks[-1]), 0.05)
    lr = np.histogram(licks, b)[0].astype(float)
    js = np.interp(b[:-1], tj, sp)
    lr, js = lr - lr.mean(), js - js.mean()
    L = np.arange(-10, 11)
    cc = np.array([np.corrcoef(np.roll(js, k), lr)[0, 1] for k in L])
    return float(L[np.argmax(cc)] * 0.05), float(cc.max())


def epoch_grid(grid, epochs):
    lab = np.full(len(grid), "outside", dtype=object)
    for k, name in (("passive_pre", "passive_pre"), ("active", "task"), ("passive_post", "passive_post")):
        for a, b in epochs.get(k, []):
            lab[(grid >= a) & (grid <= b)] = name
    return lab


def rate(trace, valid, thr):
    return len(m1.detect(trace, valid, thr)) / (valid.sum() * m1.STEP / 60)


def run(sid):
    t0 = time.time()
    D = m6.prepare(sid)
    ev, grid, valid, qg, spikes, units = D["ev"], D["grid"], D["valid"], D["quiet"], D["spikes"], D["units"]
    vq = valid & qg
    vmin = valid.sum() * m1.STEP / 60
    eg = epoch_grid(grid, D["epochs"])
    jaw_lag, jaw_r = lick_jaw_sync(D["licks"], D["dlc"])
    jaw_ok = bool(np.isfinite(jaw_r) and abs(jaw_lag) <= JAW_LAG_MAX and jaw_r >= JAW_R_MIN)
    out = m1.HOME / "pilot" / sid / OUT
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(zlib.crc32(f"{sid}|sens".encode()))
    ends_peri = ev.t.to_numpy()[:, None] + m1.PERI[None, :]
    i_tr = int(np.argmin(np.abs(m1.PERI - m1.TRAIN_WIN[1])))
    after = np.flatnonzero((m1.PERI >= m1.DET_WIN[0]) & (m1.PERI <= m1.DET_WIN[1]))
    passive = (ev.context == "passive").to_numpy()
    dec_rows, held_rows, curve_rows = [], [], []
    for g in GROUPS:
        sel = np.arange(len(units)) if g == ALL else np.flatnonzero(units.area_group.to_numpy() == g)
        if len(sel) < 10:
            continue
        tg = time.time()
        S = [spikes[i] for i in sel]
        P = np.stack([m1.counts(s, ends_peri.ravel()).reshape(ends_peri.shape) for s in S], 2).astype(np.float32)
        Xc = np.stack([m1.counts(s, grid) for s in S], 1).astype(np.float32)
        dec = []
        for target in m6.TARGETS:
            tm = m6.train_mask(ev, target, "ho_both")
            y = ev.cls.eq(m1.TARGET_CLS[target][0]).to_numpy()[tm].astype(int)
            ctx = ev.context.to_numpy()[tm].astype(str)
            _, oof, _, final = m1.nested(P[tm, i_tr].astype(float), y, m1.model_weights(y, ctx), np.char.add(ctx, y.astype(str)), rng)
            a, c = m1.lin(final)
            dec.append(dict(target=target, a=a, c=c, model=final, oof=oof, y=y, n_train=int(tm.sum())))
        A = np.stack([d["a"] for d in dec], 1).astype(np.float32)
        Cv = np.array([d["c"] for d in dec], np.float32)
        raw = Xc @ A + Cv
        R_ev = np.einsum("etu,ud->etd", P, A) + Cv
        # time-shift null: per-unit circular shifts of the binned counts
        T = np.zeros((N_NULL, len(grid), len(dec)), np.float32)
        for j in range(N_NULL):
            lag = (rng.uniform(*m1.SHIFT_S, len(S)) / m1.STEP).astype(int) * rng.choice([-1, 1], len(S))
            for i in range(len(S)):
                T[j] += np.roll(Xc[:, i], lag[i])[:, None] * A[i][None, :]
            T[j] += Cv
        t_fit = time.time() - tg
        for k, d in enumerate(dec):
            fin, target = d["model"], d["target"]
            med, sc = m1.robust(raw[valid, k])
            z = ((raw[:, k] - med) / sc).astype(np.float32)
            zn = (T[:, :, k] - med) / sc
            st = m7.strata_of(fin["mu"])
            perm = np.stack([m7.perm_within(fin["w"], st, rng) for _ in range(N_SHUF)], 1)
            TW = Xc @ (perm / fin["sd"][:, None]).astype(np.float32)
            TW = np.stack([m1.norm(TW[:, i], valid) for i in range(N_SHUF)], 1).astype(np.float32)
            Wp = np.stack([m7.perm_within(fin["w"], st, rng) for _ in range(m7.N_PERM)], 1)
            # held-out passive peaks (normalised)
            jb = after[np.argmax(R_ev[:, after, k], 1)]
            pk = (R_ev[np.arange(len(ev)), jb, k] - med) / sc
            stim, sham = m1.TARGET_CLS[target]
            other = "A" if target == "whisker" else "W"
            held = passive & ev.epoch.isin(["passive_pre", "passive_post"]).to_numpy()
            zt = pk[held & ev.cls.eq(stim).to_numpy()]
            src = "passive"
            if len(zt) < MIN_HELD:
                zt, src = (d["oof"][d["y"] == 1] - med) / sc, "cv"
            pauc = m1.pauc(d["y"], d["oof"])
            curve_rows.append(pd.DataFrame(dict(session_id=sid, group=g, target=target, thr_norm=THR_C,
                                                real=[rate(z, valid, x) for x in THR_C],
                                                shuf_mean=[np.mean([rate(TW[:, i], valid, x) for i in range(0, N_SHUF, 4)]) for x in THR_C])))
            for s in SENS_LEVELS:
                thr = float(np.quantile(zt, 1 - s)) if len(zt) else np.nan
                E = m1.detect(z, valid, thr) if np.isfinite(thr) else np.zeros((0, 3), int)
                shuf = np.array([rate(TW[:, i], valid, thr) for i in range(N_SHUF)]) if np.isfinite(thr) else np.full(N_SHUF, np.nan)
                nul = np.array([rate(zn[j], valid, thr) for j in range(N_NULL)]) if np.isfinite(thr) else np.full(N_NULL, np.nan)
                r_real = len(E) / vmin
                ep_rates = {}
                for e_ in EPOCHS:
                    mins = (valid & (eg == e_)).sum() * m1.STEP / 60
                    ep_rates[f"rate_{e_}"] = (eg[E[:, 2]] == e_).sum() / mins if mins > 0.5 else np.nan
                    ep_rates[f"min_{e_}"] = mins
                p_id = m7.identity_p((Xc[E[:, 2]] - fin["mu"]) / fin["sd"], fin["w"], Wp)
                dec_rows.append(dict(
                    mouse_id=sid[:5], session_id=sid, stage=m5.STAGE[sid[:5]], group=g, target=target, n_units=len(S),
                    n_train=d["n_train"], C=fin["C"], pauc=pauc, sens=s, thr_source=src, n_held=int(len(zt)), thr_norm=thr,
                    n_events=len(E), rate=r_real, rate_quiet=(qg[E[:, 2]]).sum() / (vq.sum() * m1.STEP / 60), **ep_rates,
                    rate_shuf=float(shuf.mean()), rate_shuf_sd=float(shuf.std()), excess=r_real - float(shuf.mean()),
                    ratio=r_real / shuf.mean() if shuf.mean() > 0 else np.nan, p_shuf=float((1 + (shuf >= r_real).sum()) / (1 + N_SHUF)),
                    null_shift=float(nul.mean()), frac_identity=float((p_id < P_GATE).mean()) if len(E) else np.nan,
                    valid_min=vmin, jaw_ok=jaw_ok))
                # held-out sensitivity / false positives at this threshold
                for es, eps in (("pre", ["passive_pre"]), ("post", ["passive_post"]), ("pre+post", ["passive_pre", "passive_post"])):
                    ins = passive & ev.epoch.isin(eps).to_numpy()
                    row = dict(session_id=sid, stage=m5.STAGE[sid[:5]], group=g, target=target, sens=s, eval_set=es, thr_norm=thr,
                               thr_source=src)
                    for nm, cl in (("target", stim), ("sham", sham), ("other", other)):
                        m = ins & ev.cls.eq(cl).to_numpy()
                        row[f"n_{nm}"] = int(m.sum())
                        row[f"cross_{nm}"] = float((pk[m] > thr).mean()) if m.any() else np.nan
                    held_rows.append(row)
                if s == SENS_MAIN:
                    C = pd.DataFrame(dict(session_id=sid, group=g, target=target, t_peak=grid[E[:, 2]], peak=z[E[:, 2]],
                                          t_start=grid[E[:, 0]] - m1.WIDTH, t_end=grid[E[:, 1]],
                                          dur_ms=(E[:, 1] - E[:, 0] + 1) * m1.STEP * 1000, quiet=qg[E[:, 2]],
                                          epoch=eg[E[:, 2]].astype(str), p_identity=p_id))
                    C.to_parquet(out / f"events_{target}_{g.replace(' ', '_')}.parquet")
                    np.savez_compressed(out / f"continuous_{target}_{g.replace(' ', '_')}.npz", t=grid, z=z, valid=valid, quiet=qg,
                                        thr=thr, sens=s, pauc=pauc, w=fin["w"], mu=fin["mu"], sd=fin["sd"], unit_index=sel)
            del TW
        del Xc, P, raw, R_ev, T
        print(f"{sid} {g}: {len(S)} units | fit+null {t_fit:.0f} s, total {time.time() - tg:.0f} s", flush=True)
    Dd = pd.DataFrame(dec_rows)
    Dd.to_csv(out / "decoders.csv", index=False)
    pd.DataFrame(held_rows).to_csv(out / "heldout.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_parquet(out / "curves.parquet")
    np.savez_compressed(out / "behaviour.npz", licks=D["licks"], ev_t=ev.t.to_numpy(), ev_cls=ev.cls.to_numpy(str),
                        ev_context=ev.context.to_numpy(str), **{f"{k}_t": v[0] for k, v in D["dlc"].items()},
                        **{f"{k}_v": v[1] for k, v in D["dlc"].items()})
    (out / "params.json").write_text(json.dumps(dict(
        session_id=sid, stage=m5.STAGE[sid[:5]], sens_levels=SENS_LEVELS, sens_main=SENS_MAIN, min_held=MIN_HELD, n_shuf=N_SHUF,
        n_null=N_NULL, n_strata=m7.N_STRATA, n_perm=m7.N_PERM, p_gate=P_GATE, train="ho_both (task only)", groups=GROUPS,
        train_win=m1.TRAIN_WIN, step=m1.STEP, excl=m1.EXCL, start_excl=m6.START_EXCL, shift_s=m1.SHIFT_S, merge=m1.MERGE,
        det_win=m1.DET_WIN, quiet_gap=m1.QUIET_GAP, whisk_max=m1.WHISK_MAX, sd_floor=m1.SD_FLOOR, valid_min=float(vmin),
        jaw_lag=jaw_lag, jaw_r=jaw_r, jaw_ok=jaw_ok, epochs=D["epochs"], date=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=str))
    pd.set_option("display.width", 250)
    print(Dd[Dd.sens == SENS_MAIN][["group", "target", "pauc", "thr_source", "thr_norm", "rate", "rate_shuf", "excess", "ratio", "p_shuf",
                                    "null_shift", "frac_identity"]].round(3).to_string(index=False))
    print(f"jaw sync: lag {jaw_lag} s, r {jaw_r:.2f} -> {'ok' if jaw_ok else 'not shown'}")
    print("DONE", sid, f"{time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ figures
def behaviour_rows(axs, B, jaw_ok, lo, hi, t0=0.0, lick_bin=0.1):
    """lick rate (piezo), whisker speed, jaw distance in [lo, hi); an empty axis when the data are missing"""
    a_l, a_w, a_j = axs
    lk = B["licks"][(B["licks"] >= lo) & (B["licks"] < hi)]
    b = np.arange(lo, hi + lick_bin, lick_bin)
    a_l.step(b[:-1] - t0, np.histogram(lk, b)[0] / lick_bin, where="post", color="k", lw=0.5)
    a_l.set_ylim(0, 14)
    for ax, key, ok, scale, cl in ((a_w, "whisker_speed", True, (0, 2000), "0.35"), (a_j, "jaw", jaw_ok, None, "0.35")):
        if not ok or f"{key}_t" not in B:
            ax.axis("off")
            continue
        m = (B[f"{key}_t"] >= lo) & (B[f"{key}_t"] < hi)
        v = B[f"{key}_v"][m]
        if not len(v) or not np.isfinite(v).any():
            ax.axis("off")
            continue
        ax.plot(B[f"{key}_t"][m] - t0, np.clip(v, *scale) if scale else v, color=cl, lw=0.4)


def fig_continuous(sid, target, out=None, fig_dir=None):
    plt = m2.setup()
    col = m2.colors()
    out = out or m1.HOME / "pilot" / sid / OUT
    Pm = json.loads((out / "params.json").read_text())
    Dd = pd.read_csv(out / "decoders.csv")
    Dd = Dd[(Dd.target == target) & (Dd.sens == SENS_MAIN)].set_index("group")
    B = dict(np.load(out / "behaviour.npz", allow_pickle=True))
    G = [g for g in GROUPS + sorted(set(Dd.index) - set(GROUPS)) if g in Dd.index]
    C = {g: dict(np.load(out / f"continuous_{target}_{g.replace(' ', '_')}.npz")) for g in G}
    E = {g: pd.read_parquet(out / f"events_{target}_{g.replace(' ', '_')}.parquet") for g in G}
    t, valid = C[G[0]]["t"], C[G[0]]["valid"]
    eg = epoch_grid(t, Pm["epochs"])
    allp = np.concatenate([E[g].t_peak.to_numpy() for g in G if g != ALL] or [np.zeros(0)])
    rng = np.random.default_rng(zlib.crc32(f"{sid}|{target}|seg".encode()))
    segs = []
    for e_ in EPOCHS:
        tt = t[(eg == e_) & valid]
        if len(tt) < SEG / m1.STEP:
            continue
        starts = np.arange(tt[0], tt[-1] - SEG, 5.0)
        if not len(starts):
            continue
        n = np.searchsorted(np.sort(allp), starts + SEG) - np.searchsorted(np.sort(allp), starts)
        segs.append((f"{e_}: densest {SEG:.0f} s", float(starts[np.argmax(n)])))
    tv = t[valid]
    segs.append(("random", float(rng.uniform(tv[0], tv[-1] - SEG))))
    hr = [1.0] * len(G) + [0.45, 0.45, 0.45]
    fig, axs = plt.subplots(len(hr), len(segs), figsize=(7.4, 0.62 * sum(hr) + 0.6), squeeze=False, sharex="col",
                            gridspec_kw=dict(height_ratios=hr, hspace=0.12, wspace=0.08))
    evt, evc, evx = B["ev_t"], B["ev_cls"], B["ev_context"]
    for j, (lab, lo) in enumerate(segs):
        hi = lo + SEG
        m = (t >= lo) & (t < hi)
        for i, g in enumerate(G):
            ax = axs[i, j]
            cc = col.get(g, "k") if g != ALL else "k"
            zz = np.clip(C[g]["z"][m], -4, 40)
            ax.plot(t[m] - lo, np.where(C[g]["valid"][m], zz, np.nan), color=cc, lw=0.35)
            ax.plot(t[m] - lo, np.where(C[g]["valid"][m], np.nan, zz), color="0.85", lw=0.35)
            thr = float(C[g]["thr"])
            ax.axhline(thr, color="crimson", lw=0.4, ls="--")
            e = E[g][(E[g].t_peak >= lo) & (E[g].t_peak < hi)]
            top = max(thr * 1.6, np.nanmax(zz) if len(zz) else thr) * 1.08
            ax.scatter(e.t_peak - lo, np.full(len(e), top * 0.97), marker="v", s=7, lw=0.4, edgecolors="k",
                       facecolors=["k" if p < P_GATE else "w" for p in e.p_identity], zorder=3)
            for x_, c_, cx in zip(evt, evc, evx):
                if lo <= x_ < hi and c_ in ("W", "A"):
                    ax.axvline(x_ - lo, color="#1f77b4" if c_ == "W" else "#d62728", lw=0.35, alpha=0.6)
            ax.set_ylim(-4, top)
            if j == 0:
                d = Dd.loc[g]
                ax.set_ylabel(f"{m2.short(g)}\n{d.pauc:.2f}", fontsize=4.6, color=cc)
            else:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(lab, loc="left", fontsize=5)
        behaviour_rows(axs[len(G):, j], B, Pm["jaw_ok"], lo, hi, t0=lo)
        if j == 0:
            for ax, l_ in zip(axs[len(G):, 0], ("licks/s", "whisk\n(deg/s)", "jaw\n(px)")):
                if ax.axison:
                    ax.set_ylabel(l_, fontsize=4.6)
        axs[-1, j].set_xlabel("time in segment (s)")
        for ax in axs[len(G):, j]:
            ax.set_xlim(0, SEG)
            if j > 0:
                ax.set_yticklabels([])
    jaw = "jaw shown" if Pm["jaw_ok"] else f"jaw hidden (lick-jaw sync check failed: lag {Pm['jaw_lag']} s, r {Pm['jaw_r']:.2f})"
    fig.suptitle(f"{sid} ({Pm['stage']}): {target} decoders (task-trained), readout in MAD units; threshold (red) = "
                 f"{SENS_MAIN:.0%} of held-out passive {target} stimuli cross\ntriangles: events (filled: identity p < {P_GATE:g}); "
                 f"grey: excluded time; vertical lines: whisker (blue) / auditory (red) stimuli\nleft labels: group, pAUC; {jaw}",
                 x=0.01, ha="left", fontsize=5.6, weight="bold")
    name = f"fig14_continuous_{sid}_{target}"
    m2.save(fig, out / "figures", name)
    m2.save(fig, fig_dir or m1.HOME / "figures" / "sensitivity_continuous", name)
    plt.close(fig)


def fig_examples(sid, target, out=None, fig_dir=None, units=None):
    plt = m2.setup()
    col = m2.colors()
    out = out or m1.HOME / "pilot" / sid / OUT
    Pm = json.loads((out / "params.json").read_text())
    Dd = pd.read_csv(out / "decoders.csv")
    Dd = Dd[(Dd.target == target) & (Dd.sens == SENS_MAIN)].set_index("group")
    B = dict(np.load(out / "behaviour.npz", allow_pickle=True))
    if units is None:
        units, *_ = m1.load_session(sid)
    G = [g for g in GROUPS + sorted(set(Dd.index) - set(GROUPS)) if g in Dd.index and g != ALL]
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7.4, 2.9 * len(G)), layout="constrained")
    sub = np.atleast_1d(fig.subfigures(len(G), 1))
    for r, g in enumerate(G):
        E = pd.read_parquet(out / f"events_{target}_{g.replace(' ', '_')}.parquet")
        c = dict(np.load(out / f"continuous_{target}_{g.replace(' ', '_')}.npz"))
        d = Dd.loc[g]
        cc = col.get(g, "k")
        sub[r].suptitle(f"{m2.short(g)}: {int(d.n_units)} units, pAUC {d.pauc:.2f}, threshold {d.thr_norm:.1f} MAD "
                        f"({d.thr_source}); {d.rate:.2f}/min vs shuffled template {d.rate_shuf:.2f} (excess {d.excess:+.2f}, "
                        f"p {d.p_shuf:.2f}); time-shift null {d.null_shift:.2f}", x=0.01, ha="left", fontsize=5.4, color=cc,
                        weight="bold")
        axs = sub[r].subplots(5, m5.N_EX, sharex="col", gridspec_kw=dict(height_ratios=[2.2, 1, 0.45, 0.45, 0.45], hspace=0.06))
        Pk = m5.pick(E, rng) if len(E) else E
        order = np.argsort(c["w"])[::-1]
        for j in range(m5.N_EX):
            col_axes = axs[:, j]
            if j >= len(Pk):
                for a in col_axes:
                    a.axis("off")
                continue
            e = Pk.iloc[j]
            t0 = float(e.t_peak)
            a0, a1 = col_axes[:2]
            for k, u in enumerate(order):
                st = np.asarray(units.spike_times.iloc[int(c["unit_index"][u])])
                st = st[(st >= t0 - HALF) & (st < t0 + HALF)] - t0
                a0.vlines(st, k, k + 0.9, color=cc if c["w"][u] > 0 else "0.6", lw=0.3)
            a0.set_ylim(0, len(order)); a0.set_yticks([])
            a0.axvspan(-m1.WIDTH, 0, color="crimson", alpha=0.08, lw=0)
            a0.set_title(f"{e.epoch}, t = {t0:.0f} s, peak {e.peak:.1f} MAD\nidentity p = {e.p_identity:.3f}"
                         f"{', quiet' if e.quiet else ''}", fontsize=4.5)
            m = (c["t"] >= t0 - HALF) & (c["t"] <= t0 + HALF)
            a1.plot(c["t"][m] - t0, np.where(c["valid"][m], c["z"][m], np.nan), color=cc, lw=0.6)
            a1.plot(c["t"][m] - t0, np.where(c["valid"][m], np.nan, c["z"][m]), color="0.8", lw=0.6)
            a1.axhline(float(c["thr"]), color="crimson", lw=0.5, ls="--")
            behaviour_rows(col_axes[2:], B, Pm["jaw_ok"], t0 - HALF, t0 + HALF, t0=t0, lick_bin=0.05)
            for a in col_axes:
                if a.axison:
                    a.set_xlim(-HALF, HALF)
            if j == 0:
                a0.set_ylabel("units\n(by weight)", fontsize=5); a1.set_ylabel("readout\n(MAD)", fontsize=5)
                for a, l_ in zip(col_axes[2:], ("licks/s", "whisk", "jaw")):
                    if a.axison:
                        a.set_ylabel(l_, fontsize=5)
            if r == len(G) - 1:
                col_axes[-1].set_xlabel("time from event peak (s)")
    fig.suptitle(f"{sid} ({Pm['stage']}): {target} decoders, threshold = {SENS_MAIN:.0%} held-out sensitivity; events: 2 "
                 f"strongest, 2 median, 2 near threshold; raster: units sorted by weight, colour = positive weight, grey = weight <= 0"
                 f"{'' if Pm['jaw_ok'] else '; jaw hidden (sync check failed)'}", x=0.01, ha="left", fontsize=6, weight="bold")
    name = f"fig15_examples_{sid}_{target}"
    m2.save(fig, out / "figures", name)
    m2.save(fig, fig_dir or m1.HOME / "figures" / "sensitivity_examples", name)
    plt.close(fig)


def two_tests(a, b):
    from scipy.stats import mannwhitneyu, ttest_ind
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    return float(mannwhitneyu(a, b).pvalue), float(ttest_ind(a, b, equal_var=False).pvalue)


def summary():
    plt = m2.setup()
    col = m2.colors()
    load = lambda name: pd.concat([pd.read_csv(f) for f in sorted((m1.HOME / "pilot").glob(f"*/{OUT}/{name}"))], ignore_index=True)
    Dd, Hd = load("decoders.csv"), load("heldout.csv")
    Cu = pd.concat([pd.read_parquet(f) for f in sorted((m1.HOME / "pilot").glob(f"*/{OUT}/curves.parquet"))], ignore_index=True)
    Dd["cohort"] = Dd.stage.str[-2:]
    Dd.to_csv(m1.HOME / "sens_decoders.csv", index=False)
    Hd.to_csv(m1.HOME / "sens_heldout.csv", index=False)
    G = [g for g in GROUPS if g in set(Dd.group)]
    # fig13: real vs shuffled at the sensitivity thresholds
    fig, axs = plt.subplots(len(G), 4, figsize=(7.4, 1.5 * len(G)), squeeze=False)
    for i, g in enumerate(G):
        cc = col.get(g, "k") if g != ALL else "k"
        for j, tgt in enumerate(m6.TARGETS):
            q = Dd[(Dd.group == g) & (Dd.target == tgt)]
            a0, a1 = axs[i, 2 * j], axs[i, 2 * j + 1]
            m = q[q.sens == SENS_MAIN]
            for coh, cl in COH.items():
                mm = m[m.cohort == coh]
                a0.scatter(mm.rate_shuf.clip(lower=1e-2), mm.rate.clip(lower=1e-2), s=9, lw=0.5, edgecolors=cl,
                           facecolors=[cl if p >= GOOD_PAUC else "w" for p in mm.pauc], marker="o", zorder=3,
                           label=coh if i == 0 and j == 0 else None)
            lim = (1e-2, 300)
            a0.plot(lim, lim, color="0.6", lw=0.4, ls=":"); a0.set_xscale("log"); a0.set_yscale("log")
            a0.set_xlim(lim); a0.set_ylim(lim)
            for sid_, qq in q.groupby("session_id"):
                qq = qq.sort_values("sens")
                good = qq.pauc.iloc[0] >= GOOD_PAUC
                a1.plot(qq.sens, qq.ratio, color=COH[qq.cohort.iloc[0]] if "cohort" in qq else cc, lw=0.7 if good else 0.3,
                        alpha=0.9 if good else 0.4, marker="o", ms=1.5)
            a1.axhline(1, color="0.6", lw=0.4, ls=":"); a1.set_yscale("log"); a1.set_ylim(0.3, 30)
            a1.set_xticks(SENS_LEVELS)
            if i == 0:
                a0.set_title(f"{tgt}: events / min at {SENS_MAIN:.0%} sensitivity\nreal (y) vs shuffled template (x)", loc="left", fontsize=5)
                a1.set_title(f"{tgt}: real / shuffled vs\nheld-out sensitivity of the threshold", loc="left", fontsize=5)
                if j == 0:
                    a0.legend(frameon=False, fontsize=4.5)
            if i == len(G) - 1:
                a0.set_xlabel("shuffled template (/min)"); a1.set_xlabel("held-out sensitivity")
        axs[i, 0].set_ylabel(f"{m2.short(g)}\nreal (/min)", color=cc)
    fig.suptitle("Threshold set by held-out passive sensitivity: real decoder vs rate-matched shuffled template "
                 f"(filled / thick: pAUC >= {GOOD_PAUC}; one point / line per session)", x=0.01, ha="left", fontsize=6, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    m2.save(fig, m1.HOME / "figures", "fig13_sens_real_vs_shuffled")
    plt.close(fig)
    # fig16: cohorts, per target; rows = metric, columns = whole brain + groups
    stats = []
    metrics = [("rate", "events / min"), ("excess", "excess over shuffled\n(/ min)"), ("ratio", "real / shuffled"),
               ("rate_quiet", "events / min, quiet")]
    for tgt in m6.TARGETS:
        q0 = Dd[(Dd.target == tgt) & (Dd.sens == SENS_MAIN)]
        fig, axs = plt.subplots(len(metrics), len(G), figsize=(7.4, 1.25 * len(metrics) + 0.5), squeeze=False)
        for j, g in enumerate(G):
            cc = col.get(g, "k") if g != ALL else "k"
            q = q0[(q0.group == g) & (q0.pauc >= GOOD_PAUC)]
            for i, (mt, lab) in enumerate(metrics):
                ax = axs[i, j]
                for x_, stg in enumerate(STAGES):
                    v = q[q.stage == stg][mt].to_numpy(float)
                    cl = COH[stg[-2:]]
                    ax.scatter(np.full(len(v), x_) + np.linspace(-0.12, 0.12, max(len(v), 1))[:len(v)], v, s=8, color=cl, lw=0, zorder=3)
                    if len(v):
                        ax.plot([x_ - 0.25, x_ + 0.25], [np.nanmedian(v)] * 2, color=cl, lw=0.9)
                for stg in ("day 0", "expert"):
                    a_ = q[q.stage == f"{stg} R+"][mt]
                    b_ = q[q.stage == f"{stg} R-"][mt]
                    p_mw, p_t = two_tests(a_, b_)
                    stats.append(dict(target=tgt, group=g, metric=mt, stage=stg, n_rplus=len(a_), n_rminus=len(b_),
                                      median_rplus=a_.median(), median_rminus=b_.median(), p_mannwhitney=p_mw, p_welch=p_t))
                if mt == "ratio":
                    ax.axhline(1, color="0.6", lw=0.4, ls=":")
                if mt == "excess":
                    ax.axhline(0, color="0.6", lw=0.4, ls=":")
                ax.set_xticks(range(len(STAGES)), [s.replace(" ", "\n", 2) for s in STAGES] if i == len(metrics) - 1 else [],
                              fontsize=4.2)
                ax.set_xlim(-0.6, len(STAGES) - 0.4)
                if j == 0:
                    ax.set_ylabel(lab, fontsize=5)
                if i == 0:
                    ax.set_title(f"{'whole brain' if g == ALL else m2.short(g)}\n(n = {q.session_id.nunique()})", fontsize=5.2, color=cc)
        fig.suptitle(f"{tgt} decoders: event rates at {SENS_MAIN:.0%} held-out sensitivity, per session (pAUC >= {GOOD_PAUC}); "
                     "R+ green, R- magenta; whole brain = one decoder on all units", x=0.01, ha="left", fontsize=6, weight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        m2.save(fig, m1.HOME / "figures", f"fig16_cohort_rates_{tgt}")
        plt.close(fig)
    St = pd.DataFrame(stats)
    St.to_csv(m1.HOME / "sens_cohort_stats.csv", index=False)
    pd.set_option("display.width", 260)
    m = Dd[Dd.sens == SENS_MAIN]
    print(m.groupby(["target", "group"]).apply(lambda q: pd.Series(dict(
        n=len(q), good=(q.pauc >= GOOD_PAUC).sum(), thr_med=q[q.pauc >= GOOD_PAUC].thr_norm.median(),
        rate_med=q[q.pauc >= GOOD_PAUC].rate.median(), shuf_med=q[q.pauc >= GOOD_PAUC].rate_shuf.median(),
        ratio_med=q[q.pauc >= GOOD_PAUC].ratio.median(), p_shuf_lt05=(q[q.pauc >= GOOD_PAUC].p_shuf < 0.05).sum()))).round(3).to_string())
    print(Hd[Hd.sens == SENS_MAIN].groupby(["target", "eval_set"])[["cross_target", "cross_sham", "cross_other"]].median().round(2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="*", default=[])
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    for s in a.session:
        if not a.figures_only:
            run(s)
        for tgt in m6.TARGETS:
            fig_continuous(s, tgt)
            fig_examples(s, tgt)
    if a.summary:
        summary()
