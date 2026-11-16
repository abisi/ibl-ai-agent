"""006 -- Threshold per decoder from the continuous data (time-shift null <= FDR of the data rate), held-out passive stimuli,
content controls (user 2026-10-29).

Decoders: stimulus data only (001 clean template: quiet target stimuli vs correct rejections + passive shams), every unit of
each area group, one per session x area group x target (whisker; auditory = specificity control) x training set:
  all      every training event
  ho_pre   passive-pre trials held out (task + passive post)
  ho_post  passive-post trials held out (task + passive pre)
  ho_both  both passive blocks held out (task only)
Continuous readout as 001 (30-ms windows, 5-ms step, valid time outside -100..600 ms of trial onsets and shams and
after the first START_EXCL s of the recording; every rate = events in valid time / valid minutes; quiet = > 1 s
from licks and whisker speed < 200 deg/s), in the decoder's own decision-value units (log-odds), not normalised.
Threshold (Sugden et al. 2020): the smallest value at which the mean time-shift null rate (N_NULL independent per-unit shifts
of 30-300 s, all valid time) is <= FDR x the data rate, at that value and every higher one. Not reached: no threshold at which
the data exceed the null by 1 / FDR; the value is still reported, with fdr_reached False.
Held-out stimuli (point 3), per decoder trained without them, for passive pre, post, and pre + post:
  direct     a window ending 10-100 ms after the onset exceeds the threshold (target stimuli; shams; other modality)
  injected   the trial's evoked increment (counts 5-35 ms after onset minus -40..-10 ms, per unit) added to N_INJ random quiet
             valid continuous windows: fraction above threshold (detection inside spontaneous activity)
Content controls (decoder "all"): shuffled template (N_NULL permutations of the weights over units, rate at the decoder's
threshold in median / MAD units of each readout); cell-identity test per event (the event's z-scored population vector scored with N_PERM permuted weight vectors;
p = fraction >= the real score); whisker-auditory coincidence (whisker event peak within COINC of an auditory event of the same
group, vs auditory times circularly shifted); cross-area coincidence (other groups with a whisker event within COINC).
Writes <HOME>/pilot/<session>/fdr/: decoders.csv, heldout.csv, curves.parquet, events_<target>_<group>.parquet,
continuous_<target>_<group>.npz, params.json; figures fig8_examples_<session>_<target>; with --summary: <HOME>/fdr_decoders.csv,
fdr_heldout.csv, figures/fig9_fdr_summary.
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
FDR = 0.05
N_NULL = 20
N_PERM = 200
N_INJ = 50                                       # random quiet windows per injected trial
BASE_END = -0.010                                # baseline window for the evoked increment ends here (-40..-10 ms)
COINC = 0.025                                    # s
START_EXCL = 30.0                                # s after the first spike excluded from valid time (events and rates)
TARGETS = ("whisker", "auditory")
VARIANTS = ("all", "ho_pre", "ho_post", "ho_both")
HELD = {"all": (), "ho_pre": ("passive_pre",), "ho_post": ("passive_post",), "ho_both": ("passive_pre", "passive_post")}
THR_N = np.r_[np.linspace(1, 10, 91), np.linspace(10.5, 40, 60)]   # grid: median + THR_N x (1.4826 MAD) of the data readout
OUT = "fdr"


def n_events(trace, valid, thr):
    above = valid & (trace > thr)
    i = np.flatnonzero(above)
    return 0 if not len(i) else 1 + int((np.diff(i) > max(1, int(round(m1.MERGE / m1.STEP)))).sum())


def curve(trace, valid, grid):
    mins = valid.sum() * m1.STEP / 60
    return np.array([n_events(trace, valid, t) for t in grid]) / mins


def fdr_threshold(grid, real, null):
    ok = null <= FDR * real
    bad = np.flatnonzero(~ok)
    k = 0 if not len(bad) else bad[-1] + 1
    if k >= len(grid):
        return float(grid[-1]), False
    return float(grid[k]), bool(real[k] > 0)


def prepare(sid):
    """as 001 run(): events, artefact-corrected spikes, continuous grid, valid and quiet time"""
    units, trials, licks, dlc, epochs, ru = m1.load_session(sid)
    ev = pd.concat([m1.task_events(trials), m1.passive_events(trials)], ignore_index=True).sort_values("t").reset_index(drop=True)
    ev["quiet"] = m1.quiet(ev.t.to_numpy(), licks)
    ev["epoch"] = [m5.epoch_of(t, epochs, None) for t in ev.t]
    ev.loc[ev.context == "task", "epoch"] = "task"
    wh = trials.loc[trials.whisker_stim == 1, "start_time"].to_numpy()
    onsets = np.unique(np.r_[wh, ev.t.to_numpy()])
    rng0 = np.random.default_rng(zlib.crc32(f"{sid}|react".encode()))
    spikes = [ru.correct_neuron_spike_train(np.asarray(st), onsets, rng0) for st in units.spike_times]
    t_lo = min(float(np.min(s)) for s in spikes if len(s)); t_hi = max(float(np.max(s)) for s in spikes if len(s))
    grid = np.round(np.arange(t_lo + m1.WIDTH, t_hi, m1.STEP), 6)
    excl_t = np.r_[trials.start_time.to_numpy(), ev.loc[ev.cls.isin(["SW", "SA"]), "t"].to_numpy()]
    valid = grid >= t_lo + START_EXCL
    for a in np.sort(excl_t):
        valid[np.searchsorted(grid, a + m1.EXCL[0]):np.searchsorted(grid, a + m1.EXCL[1])] = False
    qg = m1.quiet_grid(grid, licks, dlc)
    return dict(units=units, ev=ev, spikes=spikes, grid=grid, valid=valid, quiet=qg, t_lo=t_lo, t_hi=t_hi, licks=licks,
                dlc=dlc, epochs=epochs, trials=trials)


def train_mask(ev, target, variant):
    stim, sham = m1.TARGET_CLS[target]
    m = ev.quiet & (((ev.context == "task") & ev.cls.isin([stim, "N"])) | ((ev.context == "passive") & ev.cls.isin([stim, sham])))
    return (m & ~ev.epoch.isin(HELD[variant])).to_numpy()


def eval_sets(variant):
    h = HELD[variant]
    if not h:
        return {}
    s = {"pre": ("passive_pre",), "post": ("passive_post",)}
    out = {k: v for k, v in s.items() if v[0] in h}
    if len(h) == 2:
        out["pre+post"] = h
    return out


def fit_all(ev, P, i_tr, rng):
    """one clean-template decoder per target x training set; P: ev x PERI x unit counts"""
    dec = []
    for target in TARGETS:
        for v in VARIANTS:
            tm = train_mask(ev, target, v)
            y = ev.cls.eq(m1.TARGET_CLS[target][0]).to_numpy()[tm].astype(int)
            ctx = ev.context.to_numpy()[tm].astype(str)
            X = P[tm, i_tr].astype(float)
            fo, oof, _, final = m1.nested(X, y, m1.model_weights(y, ctx), np.char.add(ctx, y.astype(str)), rng)
            a, c = m1.lin(final)
            dec.append(dict(target=target, variant=v, a=a, c=c, model=final, oof=oof, y=y, n_train=int(tm.sum())))
    return dec


def run(sid):
    t0 = time.time()
    D = prepare(sid)
    ev, grid, valid, qg, spikes, units = D["ev"], D["grid"], D["valid"], D["quiet"], D["spikes"], D["units"]
    vq = valid & qg
    out = m1.HOME / "pilot" / sid / OUT
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(zlib.crc32(f"{sid}|fdr".encode()))
    ends_peri = ev.t.to_numpy()[:, None] + m1.PERI[None, :]
    i_tr = int(np.argmin(np.abs(m1.PERI - m1.TRAIN_WIN[1])))
    i_bs = int(np.argmin(np.abs(m1.PERI - BASE_END)))
    after = (m1.PERI >= m1.DET_WIN[0]) & (m1.PERI <= m1.DET_WIN[1])
    inj_pos = rng.choice(np.flatnonzero(vq), N_INJ, replace=False)
    dec_rows, held_rows, curve_rows, events = [], [], [], {}
    for g in m1.GROUPS:
        sel = np.flatnonzero(units.area_group.to_numpy() == g)
        if len(sel) < 10:
            continue
        S = [spikes[i] for i in sel]
        P = np.stack([m1.counts(s, ends_peri.ravel()).reshape(ends_peri.shape) for s in S], 2).astype(np.float32)
        Xc = np.stack([m1.counts(s, grid) for s in S], 1).astype(np.float32)
        delta = P[:, i_tr] - P[:, i_bs]                                   # evoked increment, ev x unit
        dec = fit_all(ev, P, i_tr, rng)
        A = np.stack([d["a"] for d in dec], 1)
        Cv = np.array([d["c"] for d in dec], np.float32)
        raw = Xc @ A + Cv                                                 # grid x decoder
        stats = [m1.robust(raw[valid, k]) for k in range(len(dec))]
        tg = [med + sc * THR_N for med, sc in stats]
        real = [curve(raw[:, k], valid, tg[k]) for k in range(len(dec))]
        nulls, null_curves = [], [[] for _ in dec]
        for j in range(N_NULL):
            lag = rng.uniform(*m1.SHIFT_S, len(S)) * rng.choice([-1, 1], len(S))
            lo_ok, hi_ok = D["t_lo"] + np.clip(lag, 0, None).max(), D["t_hi"] + np.clip(lag, None, 0).min()
            T = np.stack([m1.counts(s + l, grid) for s, l in zip(S, lag)], 1).astype(np.float32) @ A + Cv
            vk = valid & (grid > lo_ok + m1.WIDTH) & (grid < hi_ok)
            for k in range(len(dec)):
                null_curves[k].append(curve(T[:, k], vk, tg[k]))
            nulls.append((T, vk))
        R_ev = np.einsum("etu,ud->etd", P, A) + Cv                        # ev x PERI x decoder
        pk = R_ev[:, after].max(1)
        inj_base = raw[inj_pos]                                           # N_INJ x decoder
        inj_add = delta @ A                                               # ev x decoder
        for k, d in enumerate(dec):
            nc = np.mean(null_curves[k], 0)
            thr, reached = fdr_threshold(tg[k], real[k], nc)
            d["thr"], d["reached"] = thr, reached
            n_real = n_events(raw[:, k], valid, thr)
            nq = np.mean([n_events(T[:, k], vk & qg, thr) / max((vk & qg).sum() * m1.STEP / 60, 1e-9) for T, vk in nulls])
            nall = np.mean([n_events(T[:, k], vk, thr) / (vk.sum() * m1.STEP / 60) for T, vk in nulls])
            fin = d["model"]
            perm = np.stack([rng.permutation(fin["w"]) for _ in range(N_NULL)], 1)
            TW = Xc @ (perm / fin["sd"][:, None]).astype(np.float32) + (fin["b"] - (fin["mu"] / fin["sd"]) @ perm).astype(np.float32)
            # each shuffled readout at the same threshold in its own median / MAD units (raw scales differ: permuted weights
            # land on units of other variance)
            thn = (thr - stats[k][0]) / stats[k][1]
            shuf = np.array([n_events(m1.norm(TW[:, i], valid), valid, thn) for i in range(N_NULL)]) / (valid.sum() * m1.STEP / 60)
            del TW
            neg, pos = d["oof"][d["y"] == 0], d["oof"][d["y"] == 1]
            dec_rows.append(dict(session_id=sid, mouse_id=sid[:5], stage=m5.STAGE[sid[:5]], group=g, target=d["target"],
                                 variant=d["variant"], n_units=len(S), n_train=d["n_train"], C=fin["C"],
                                 pauc=m1.pauc(d["y"], d["oof"]), thr=thr, thr_norm=(thr - stats[k][0]) / stats[k][1],
                                 fdr_reached=reached, cv_sens=float(np.mean(pos > thr)), cv_fpr=float(np.mean(neg > thr)),
                                 rate=n_real / (valid.sum() * m1.STEP / 60),
                                 rate_quiet=n_events(raw[:, k], vq, thr) / (vq.sum() * m1.STEP / 60),
                                 null_shift=nall, null_shift_quiet=nq, shuffled_template=float(shuf.mean()),
                                 shuffled_template_hi=float(np.quantile(shuf, 0.95))))
            curve_rows.append(pd.DataFrame(dict(session_id=sid, group=g, target=d["target"], variant=d["variant"], thr=tg[k],
                                                thr_norm=THR_N, real=real[k], null_shift=nc)))
            stim, sham = m1.TARGET_CLS[d["target"]]
            other = "A" if d["target"] == "whisker" else "W"
            for es, eps in eval_sets(d["variant"]).items():
                inset = (ev.context == "passive").to_numpy() & ev.epoch.isin(eps).to_numpy()
                row = dict(session_id=sid, mouse_id=sid[:5], stage=m5.STAGE[sid[:5]], group=g, target=d["target"],
                           variant=d["variant"], eval_set=es, thr=thr, fdr_reached=reached)
                for name, cl in (("target", stim), ("sham", sham), ("other", other)):
                    m = inset & ev.cls.eq(cl).to_numpy()
                    row[f"n_{name}"] = int(m.sum())
                    row[f"direct_{name}"] = float(np.mean(pk[m, k] > thr)) if m.any() else np.nan
                    row[f"inj_{name}"] = float(np.mean(inj_base[:, k][None, :] + inj_add[m, k][:, None] > thr)) if m.any() else np.nan
                held_rows.append(row)
        # content controls and saved readouts: decoders "all"
        for k, d in enumerate(dec):
            if d["variant"] != "all":
                continue
            fin = d["model"]
            E = m1.detect(raw[:, k], valid, d["thr"])
            Z = (Xc[E[:, 2]] - fin["mu"]) / fin["sd"] if len(E) else np.zeros((0, len(S)))
            sc = Z @ fin["w"]
            Wp = np.stack([rng.permutation(fin["w"]) for _ in range(N_PERM)], 1)
            p_id = ((Z @ Wp) >= sc[:, None]).mean(1) if len(E) else np.zeros(0)
            tp = grid[E[:, 2]] if len(E) else np.zeros(0)
            events[(d["target"], g)] = pd.DataFrame(dict(
                session_id=sid, group=g, target=d["target"], t_start=grid[E[:, 0]] - m1.WIDTH if len(E) else [], t_end=grid[E[:, 1]] if len(E) else [],
                t_peak=tp, peak=raw[E[:, 2], k] if len(E) else [], dur_ms=(E[:, 1] - E[:, 0] + 1) * m1.STEP * 1000 if len(E) else [],
                quiet=qg[E[:, 2]] if len(E) else [], epoch=[m5.epoch_of(x, D["epochs"], None) for x in tp], p_identity=p_id))
            np.savez_compressed(out / f"continuous_{d['target']}_{g.replace(' ', '_')}.npz", t=grid, raw=raw[:, k].astype(np.float32),
                                valid=valid, quiet=qg, thr=d["thr"], fdr_reached=d["reached"], w=fin["w"], b=fin["b"], mu=fin["mu"],
                                sd=fin["sd"], unit_index=sel, cluster_id=units.cluster_id.to_numpy(str)[sel],
                                electrode_group=units.electrode_group.to_numpy(str)[sel], oof=d["oof"], y=d["y"])
        del Xc, P, raw, nulls
        print(f"{sid} {g}: {len(S)} units | {time.time() - t0:.0f} s", flush=True)
    chance = coincidence(events)
    for (target, g), E in events.items():
        E.to_parquet(out / f"events_{target}_{g.replace(' ', '_')}.parquet")
    Dd = pd.DataFrame(dec_rows)
    Dd = Dd.merge(pd.DataFrame([dict(target=t, group=g, n_events=len(E), frac_identity_p05=float((E.p_identity < 0.05).mean()) if len(E) else np.nan,
                                     frac_coinc_other_target=float(E.coinc_other.mean()) if len(E) else np.nan,
                                     frac_coinc_other_target_chance=chance.get((t, g), np.nan),
                                     frac_cross_area=float((E.n_areas > 0).mean()) if len(E) else np.nan, variant="all")
                                for (t, g), E in events.items()]), on=["target", "group", "variant"], how="left")
    Dd.to_csv(out / "decoders.csv", index=False)
    pd.DataFrame(held_rows).to_csv(out / "heldout.csv", index=False)
    pd.concat(curve_rows, ignore_index=True).to_parquet(out / "curves.parquet")
    (out / "params.json").write_text(json.dumps(dict(
        session_id=sid, stage=m5.STAGE[sid[:5]], fdr=FDR, n_null=N_NULL, n_perm=N_PERM, n_inj=N_INJ, base_end=BASE_END, coinc=COINC,
        targets=TARGETS, variants=VARIANTS, held=HELD, train_win=m1.TRAIN_WIN, step=m1.STEP, excl=m1.EXCL, shift_s=m1.SHIFT_S,
        merge=m1.MERGE, det_win=m1.DET_WIN, quiet_gap=m1.QUIET_GAP, whisk_max=m1.WHISK_MAX, sd_floor=m1.SD_FLOOR,
        valid_min=float(valid.sum() * m1.STEP / 60), quiet_min=float(vq.sum() * m1.STEP / 60), epochs=D["epochs"],
        n_events_by_class={f"{c}|{e}": int(n) for (c, e), n in ev.groupby(["cls", "epoch"]).size().items()},
        date=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=str))
    np.savez_compressed(out / "behaviour.npz", licks=D["licks"], **{f"{k}_t": v[0] for k, v in D["dlc"].items()},
                        **{f"{k}_v": v[1] for k, v in D["dlc"].items()})
    print(Dd[Dd.variant == "all"][["group", "target", "pauc", "thr_norm", "fdr_reached", "rate", "null_shift", "shuffled_template",
                                    "frac_identity_p05", "frac_coinc_other_target", "frac_cross_area"]].round(3).to_string(index=False))
    print("DONE", sid, f"{time.time() - t0:.0f} s", flush=True)


def coincidence(events):
    """per event: an event of the other target in the same group within COINC (chance: the other target's times circularly
    shifted by 10-100 s, 20 times); number of other groups with an event of the same target within COINC"""
    rng = np.random.default_rng(1)
    chance = {}
    for (target, g), E in events.items():
        other = events.get(("auditory" if target == "whisker" else "whisker", g))
        tp = E.t_peak.to_numpy()
        o = np.sort(other.t_peak.to_numpy()) if other is not None else np.zeros(0)
        E["coinc_other"] = near(tp, o)
        if len(o) and len(tp):
            lo, hi = min(tp.min(), o.min()), max(tp.max(), o.max()) + 1e-6
            chance[(target, g)] = float(np.mean([near(tp, np.sort(lo + (o - lo + rng.uniform(10, 100)) % (hi - lo))).mean()
                                                 for _ in range(20)]))
        E["n_areas"] = np.sum([near(tp, np.sort(F.t_peak.to_numpy())) for (t2, g2), F in events.items()
                               if t2 == target and g2 != g], 0).astype(int) if len(events) > 1 else 0
    return chance


def near(t, o):
    if not len(o):
        return np.zeros(len(t), bool)
    i = np.clip(np.searchsorted(o, t), 1, len(o) - 1)
    return np.minimum(np.abs(t - o[i - 1]), np.abs(o[i] - t)) <= COINC


# ------------------------------------------------------------------------------------------------ figures
RULE = f"time-shift null <= {FDR:.0%} of the data rate"


def fig_examples(sid, target, sub_dir=OUT, rule=RULE, fig_name="fig8_examples"):
    plt = m2.setup()
    col = m2.colors()
    out = m1.HOME / "pilot" / sid / sub_dir
    Dd = pd.read_csv(out / "decoders.csv")
    Hd = pd.read_csv(out / "heldout.csv")
    B = dict(np.load(out / "behaviour.npz"))
    units, *_ = m1.load_session(sid)
    G = [g for g in m1.GROUPS if (out / f"events_{target}_{g.replace(' ', '_')}.parquet").exists()]
    rng = np.random.default_rng(0)
    fig = plt.figure(figsize=(7.4, 2.2 * len(G)), layout="constrained")
    sub = np.atleast_1d(fig.subfigures(len(G), 1))
    for r, g in enumerate(G):
        E = pd.read_parquet(out / f"events_{target}_{g.replace(' ', '_')}.parquet")
        c = dict(np.load(out / f"continuous_{target}_{g.replace(' ', '_')}.npz"))
        d = Dd[(Dd.group == g) & (Dd.target == target) & (Dd.variant == "all")].iloc[0]
        h = Hd[(Hd.group == g) & (Hd.target == target) & (Hd.variant == "ho_both") & (Hd.eval_set == "pre+post")]
        hs = (f"held-out passive pre+post: direct {h.direct_target.iloc[0]:.2f}, injected {h.inj_target.iloc[0]:.2f}"
              if len(h) else "no held-out passive")
        cc = col.get(g, "k")
        sub[r].suptitle(f"{m2.short(g)}: {int(d.n_units)} units, pAUC {d.pauc:.2f}, threshold {d.thr:.2f} ({d.thr_norm:.1f} MAD"
                        f"{'' if d.fdr_reached else ', FDR not reached'}); {d.rate:.2f}/min vs time-shift null {d.null_shift:.2f}, "
                        f"shuffled template {d.shuffled_template:.2f}; {hs}", x=0.01, ha="left", fontsize=5.4, color=cc, weight="bold")
        axs = sub[r].subplots(3, m5.N_EX, sharex="col", gridspec_kw=dict(height_ratios=[2.2, 1, 0.5], hspace=0.05))
        Pk = m5.pick(E, rng) if len(E) else E
        order = np.argsort(c["w"])[::-1]
        for j in range(m5.N_EX):
            a0, a1, a2 = axs[:, j]
            if j >= len(Pk):
                for a in (a0, a1, a2):
                    a.axis("off")
                continue
            e = Pk.iloc[j]
            t0 = float(e.t_peak)
            for k, u in enumerate(order):
                st = np.asarray(units.spike_times.iloc[int(c["unit_index"][u])])
                st = st[(st >= t0 - m5.HALF) & (st < t0 + m5.HALF)] - t0
                a0.vlines(st, k, k + 0.9, color=cc if c["w"][u] > 0 else "0.6", lw=0.3)
            a0.set_ylim(0, len(order)); a0.set_yticks([])
            a0.axvspan(-m1.WIDTH, 0, color="crimson", alpha=0.08, lw=0)
            flags = [f for f, ok in (("quiet", e.quiet), (f"{'aud' if target == 'whisker' else 'whisk'}-coinc.", e.coinc_other),
                                     (f"+{int(e.n_areas)} areas", e.n_areas > 0)) if ok]
            a0.set_title(f"{e.epoch}, t = {t0:.0f} s, peak {e.peak:.1f}\nidentity p = {e.p_identity:.3f}"
                         f"{'; ' + ', '.join(flags) if flags else ''}", fontsize=4.5)
            m = (c["t"] >= t0 - m5.HALF) & (c["t"] <= t0 + m5.HALF)
            a1.plot(c["t"][m] - t0, np.where(c["valid"][m], c["raw"][m], np.nan), color=cc, lw=0.6)
            a1.plot(c["t"][m] - t0, np.where(c["valid"][m], np.nan, c["raw"][m]), color="0.8", lw=0.6)
            a1.axhline(float(c["thr"]), color="crimson", lw=0.5, ls="--")
            lk = B["licks"][(B["licks"] >= t0 - m5.HALF) & (B["licks"] < t0 + m5.HALF)] - t0
            a2.vlines(lk, 0, 1, color="k", lw=0.5)
            if "whisker_speed_t" in B:
                mm = (B["whisker_speed_t"] >= t0 - m5.HALF) & (B["whisker_speed_t"] < t0 + m5.HALF)
                v = B["whisker_speed_v"][mm]
                if len(v) and np.isfinite(v).any():
                    a2.plot(B["whisker_speed_t"][mm] - t0, 1 + np.clip(v, 0, 2000) / 2000, color="0.4", lw=0.4)
            a2.set_ylim(0, 2.05); a2.set_yticks([]); a2.set_xlim(-m5.HALF, m5.HALF)
            if j == 0:
                a0.set_ylabel("units\n(by weight)", fontsize=5); a1.set_ylabel("readout\n(log-odds)", fontsize=5)
                a2.set_ylabel("licks /\nwhisk", fontsize=5)
            if r == len(G) - 1:
                a2.set_xlabel("time from event peak (s)")
    fig.suptitle(f"{sid} ({m5.STAGE[sid[:5]]}): {target} decoders trained on stimulus data, threshold per decoder: {rule}\n"
                 f"events: 2 strongest, 2 median, 2 near threshold; raster colour: positive weight; "
                 "identity p: event scored with permuted weights", x=0.01, ha="left", fontsize=6.3, weight="bold")
    name = f"{fig_name}_{sid}_{target}"
    m2.save(fig, out / "figures", name)
    m2.save(fig, m1.HOME / "figures" / f"{sub_dir}_examples", name)
    plt.close(fig)


def summary():
    plt = m2.setup()
    col = m2.colors()
    Dd = pd.concat([pd.read_csv(f) for f in sorted((m1.HOME / "pilot").glob(f"*/{OUT}/decoders.csv"))], ignore_index=True)
    Hd = pd.concat([pd.read_csv(f) for f in sorted((m1.HOME / "pilot").glob(f"*/{OUT}/heldout.csv"))], ignore_index=True)
    Dd.to_csv(m1.HOME / "fdr_decoders.csv", index=False)
    Hd.to_csv(m1.HOME / "fdr_heldout.csv", index=False)
    G = [g for g in m1.GROUPS if g in set(Dd.group)]
    sids = sorted(Dd.session_id.unique(), key=lambda s: (list(m5.STAGE).index(s[:5]), s))
    x = np.arange(len(sids))
    fig, axs = plt.subplots(len(G), 4, figsize=(7.4, 1.45 * len(G)), squeeze=False, gridspec_kw=dict(width_ratios=[1.5, 1.2, 1.2, 1]))
    for i, g in enumerate(G):
        cc = col.get(g, "k")
        for tgt, off, mk in (("whisker", -0.15, "o"), ("auditory", 0.15, "s")):
            q = Dd[(Dd.group == g) & (Dd.target == tgt) & (Dd.variant == "all")].set_index("session_id").reindex(sids)
            fc = [cc if r else "w" for r in q.fdr_reached.fillna(False)]
            ax = axs[i, 0]
            ax.scatter(x + off, q.rate.clip(lower=1e-3), s=9, marker=mk, facecolors=fc, edgecolors=cc, lw=0.5, zorder=3,
                       label=f"{tgt} data" if i == 0 else None)
            ax.scatter(x + off, q.null_shift.clip(lower=1e-3), s=5, marker="_", color="k", lw=0.7, zorder=3,
                       label="time-shift null" if i == 0 and tgt == "whisker" else None)
            ax.scatter(x + off, q.shuffled_template.clip(lower=1e-3), s=5, marker="x", color="0.5", lw=0.5, zorder=3,
                       label="shuffled template" if i == 0 and tgt == "whisker" else None)
            ax = axs[i, 3]
            ax.scatter(x + off, q.frac_identity_p05, s=8, marker=mk, color=cc if tgt == "whisker" else "0.5", lw=0)
            ax.scatter(x + off, q.frac_coinc_other_target, s=8, marker=mk, facecolors="w", edgecolors=cc if tgt == "whisker" else "0.5", lw=0.5)
        ax = axs[i, 0]
        ax.set_yscale("log"); ax.set_xticks(x, [s[:5] for s in sids], rotation=90, fontsize=4.5); ax.set_ylabel(f"{m2.short(g)}\nevents / min", color=cc)
        if i == 0:
            ax.set_title("data (filled: FDR reached) vs nulls at the decoder's threshold", loc="left", fontsize=5.2)
            ax.legend(frameon=False, fontsize=3.8, ncol=2)
        for j, (k, lab) in enumerate((("direct", "held-out passive whisker, direct"), ("inj", "injected into quiet spontaneous"))):
            ax = axs[i, 1 + j]
            for es, mk in (("pre", "^"), ("post", "v"), ("pre+post", "o")):
                vv = "ho_both" if es == "pre+post" else ("ho_pre" if es == "pre" else "ho_post")
                for nm, fcol in (("target", cc), ("other", "0.45"), ("sham", "0.8")):
                    q = Hd[(Hd.group == g) & (Hd.target == "whisker") & (Hd.variant == vv) & (Hd.eval_set == es)].set_index("session_id").reindex(sids)
                    ax.scatter(x + {"pre": -0.2, "post": 0.2, "pre+post": 0}[es], q[f"{k}_{nm}"], s=6, marker=mk, color=fcol, lw=0,
                               label=f"{nm} {es}" if i == 0 and j == 0 else None)
            ax.set_ylim(-0.03, 1.03); ax.set_xticks(x, [s[:5] for s in sids], rotation=90, fontsize=4.5)
            if i == 0:
                ax.set_title(lab, loc="left", fontsize=5.2)
                if j == 0:
                    ax.legend(frameon=False, fontsize=3.4, ncol=3)
        ax = axs[i, 3]
        ax.set_ylim(-0.03, 1.03); ax.set_xticks(x, [s[:5] for s in sids], rotation=90, fontsize=4.5)
        ax.axhline(0.05, color="0.6", lw=0.4, ls=":")
        if i == 0:
            ax.set_title("filled: identity p < 0.05\nopen: coincident with other-modality event", loc="left", fontsize=5.0)
    fig.suptitle(f"Per-decoder thresholds (time-shift null <= {FDR:.0%} of data), held-out passive stimuli and content controls; "
                 "circles whisker, squares auditory decoder; sessions: day 0 R+, day 0 R-, expert R+, expert R-",
                 x=0.01, ha="left", fontsize=6, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    m2.save(fig, m1.HOME / "figures", "fig9_fdr_summary")
    plt.close(fig)
    pd.set_option("display.width", 260)
    print(Dd[Dd.variant == "all"][["session_id", "group", "target", "pauc", "thr_norm", "fdr_reached", "rate", "rate_quiet", "null_shift",
                                    "shuffled_template", "frac_identity_p05", "frac_coinc_other_target", "frac_coinc_other_target_chance",
                                    "frac_cross_area"]].round(3).to_string(index=False))
    print(Hd[Hd.target == "whisker"][["session_id", "group", "variant", "eval_set", "n_target", "direct_target", "inj_target",
                                       "direct_sham", "inj_sham", "direct_other", "inj_other"]].round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", nargs="*", default=[])
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    for s in a.session:
        if not a.figures_only:
            run(s)
        for tgt in TARGETS:
            fig_examples(s, tgt)
    if a.summary:
        summary()
