"""007 -- Threshold per decoder against the shuffled template, cell-identity gate per event, leave-one-unit-out participation
(user 2026-10-29, after 006: at the time-shift rule shuffled templates gave as many events as the real ones and events were
not template-specific).

Decoders, training sets and continuous readout as 006 (clean template; whisker and auditory; all / ho_pre / ho_post /
ho_both; 30-ms windows, 5-ms step; valid and quiet time as 006, first 30 s of the recording excluded; rates per valid minute).
Shuffled template: the decoder's weights permuted over its units (N_SHUF permutations); the same units, spikes and time, so
the shuffled readout sees every population burst the real one sees, but not the stimulus pattern. Permutations within
N_STRATA firing-rate strata (deciles of the training mean count): a free permutation puts large weights on rarely-firing units,
whose z-scored count jumps on a single spike, and gives the shuffled readout heavy tails (pilot check 2026-10-29: more
crossings and more gated events than the real template). Each readout is compared in its own median / MAD units.
Threshold: the smallest value (median / MAD units of the decoder's readout) at which the mean shuffled-template event rate is
<= FDR x the decoder's event rate, there and at every higher value. Not reached: the value is still reported, fdr_reached False.
Identity gate: a candidate (readout above threshold) is kept when its z-scored population vector scores higher with the real
weights than with >= 95 % of N_PERM permuted weight vectors (rate-matched as above; p_identity < P_GATE).
Reported per decoder: candidate and kept event rates (all valid and quiet time); time-shift null (006 shifts, 30-300 s per
unit) with the same threshold and gate; shuffled-template rate at the threshold. Held-out passive stimuli (pre, post, pre+post):
direct (peak window 10-100 ms after onset) and injected (evoked increment added to N_INJ random quiet windows), crossing only
and crossing + identity gate.
Leave-one-unit-out participation (decoders "all"): for each unit, the readout without it (its count replaced by its training
mean), candidates at the decoder's threshold, identity gate without the unit; participation = the unit's mean count in those
events vs the same events circularly shifted within the valid time (N_PSHIFT shifts of 10-100 s): index (z) and p. The unit
never contributes to finding or gating the events it is scored on. Unit attributes: weight, evoked response to passive whisker
and auditory stimuli (mean increment, Wilcoxon p), waveform type, sensory label (units.parquet).
Writes <HOME>/pilot/<session>/content/: decoders.csv, heldout.csv, units.parquet, events_<target>_<group>.parquet (kept),
candidates_<target>_<group>.parquet, continuous_*.npz, curves.parquet (real vs shuffled rate per threshold), params.json; figures fig10_examples_<session>_<target>; --summary:
<HOME>/content_decoders.csv, content_heldout.csv, content_units.parquet, figures/fig11_content_summary,
fig12_real_vs_shuffled.
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
FDR = 0.05
N_SHUF = 20
N_NULL = 20
N_PERM = 200
N_STRATA = 10
P_GATE = 0.05
N_INJ = 50
N_PSHIFT = 200
PSHIFT_S = (10.0, 100.0)
OUT = "content"
RULE = f"rate-matched shuffled template <= {FDR:.0%} of the data rate, identity p < {P_GATE:g}"


def strata_of(mu):
    return np.minimum(np.argsort(np.argsort(mu)) * N_STRATA // len(mu), N_STRATA - 1)


def perm_within(w, strata, rng):
    out = w.copy()
    for s_ in np.unique(strata):
        i = np.flatnonzero(strata == s_)
        out[i] = w[rng.permutation(i)]
    return out


def identity_p(Z, w, Wp):
    """fraction of permuted weight vectors scoring the population vectors Z at least as high as the real weights"""
    if not len(Z):
        return np.zeros(0)
    return ((Z @ Wp) >= (Z @ w)[:, None]).mean(1)


def unit_meta(sid, units):
    U = pd.read_parquet(m1.UNITS, columns=["session_id", "electrode_group", "cluster_id", "mouse_id", "waveform_type", "sensory_label",
                                           "reward_group"])
    U = U[U.session_id == sid].drop_duplicates(["electrode_group", "cluster_id"])
    U["cluster_id"] = U.cluster_id.astype(str)
    return units[["electrode_group", "cluster_id"]].merge(U, on=["electrode_group", "cluster_id"], how="left", validate="one_to_one")


def evoked(P, ev, i_tr, i_bs, cls):
    from scipy.stats import wilcoxon
    m = ((ev.context == "passive") & ev.cls.eq(cls)).to_numpy()
    d = P[m, i_tr] - P[m, i_bs]
    p = np.array([wilcoxon(x, zero_method="zsplit").pvalue if np.any(x != 0) else 1.0 for x in d.T]) if m.sum() > 5 else np.full(d.shape[1], np.nan)
    return d.mean(0), p


def participation(raw_k, Xc, valid, thr_raw, fin, a, Wp, rng):
    vidx = np.flatnonzero(valid)
    offs = (rng.uniform(*PSHIFT_S, N_PSHIFT) / m1.STEP).astype(int)
    rows = []
    for i in range(Xc.shape[1]):
        r = raw_k - a[i] * (Xc[:, i] - fin["mu"][i])
        pk = m1.detect(r, valid, thr_raw)[:, 2]
        Z = (Xc[pk] - fin["mu"]) / fin["sd"]
        Z[:, i] = 0
        pk = pk[identity_p(Z, fin["w"], Wp) < P_GATE]
        if not len(pk):
            rows.append(dict(n_events_loo=0, count_events=np.nan, count_null=np.nan, part_index=np.nan, p_part=np.nan))
            continue
        x = float(Xc[pk, i].mean())
        pos = np.searchsorted(vidx, pk)
        nul = Xc[vidx[(pos[None, :] + offs[:, None]) % len(vidx)], i].mean(1)
        sdn = nul.std()
        rows.append(dict(n_events_loo=len(pk), count_events=x, count_null=float(nul.mean()),
                         part_index=float((x - nul.mean()) / sdn) if sdn > 0 else np.nan,
                         p_part=float((1 + (nul >= x).sum()) / (1 + N_PSHIFT))))
    return pd.DataFrame(rows)


def run(sid):
    t0 = time.time()
    D = m6.prepare(sid)
    ev, grid, valid, qg, spikes, units = D["ev"], D["grid"], D["valid"], D["quiet"], D["spikes"], D["units"]
    meta = unit_meta(sid, units)
    vq = valid & qg
    vmin, qmin = valid.sum() * m1.STEP / 60, vq.sum() * m1.STEP / 60
    out = m1.HOME / "pilot" / sid / OUT
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(zlib.crc32(f"{sid}|content".encode()))
    ends_peri = ev.t.to_numpy()[:, None] + m1.PERI[None, :]
    i_tr = int(np.argmin(np.abs(m1.PERI - m1.TRAIN_WIN[1])))
    i_bs = int(np.argmin(np.abs(m1.PERI - m6.BASE_END)))
    after = np.flatnonzero((m1.PERI >= m1.DET_WIN[0]) & (m1.PERI <= m1.DET_WIN[1]))
    inj_pos = rng.choice(np.flatnonzero(vq), N_INJ, replace=False)
    dec_rows, held_rows, unit_rows, events, curve_rows = [], [], [], {}, []
    for g in m1.GROUPS:
        sel = np.flatnonzero(units.area_group.to_numpy() == g)
        if len(sel) < 10:
            continue
        S = [spikes[i] for i in sel]
        P = np.stack([m1.counts(s, ends_peri.ravel()).reshape(ends_peri.shape) for s in S], 2).astype(np.float32)
        Xc = np.stack([m1.counts(s, grid) for s in S], 1).astype(np.float32)
        delta = P[:, i_tr] - P[:, i_bs]
        dec = m6.fit_all(ev, P, i_tr, rng)
        A = np.stack([d["a"] for d in dec], 1)
        Cv = np.array([d["c"] for d in dec], np.float32)
        raw = Xc @ A + Cv
        # threshold against the shuffled template, in median / MAD units of each readout
        for k, d in enumerate(dec):
            fin = d["model"]
            med, sc = m1.robust(raw[valid, k])
            real = m6.curve((raw[:, k] - med) / sc, valid, m6.THR_N)
            st = strata_of(fin["mu"])
            perm = np.stack([perm_within(fin["w"], st, rng) for _ in range(N_SHUF)], 1)
            TW = Xc @ (perm / fin["sd"][:, None]).astype(np.float32) + (fin["b"] - (fin["mu"] / fin["sd"]) @ perm).astype(np.float32)
            shuf = np.stack([m6.curve(m1.norm(TW[:, i], valid), valid, m6.THR_N) for i in range(N_SHUF)])
            del TW
            thn, reached = m6.fdr_threshold(m6.THR_N, real, shuf.mean(0))
            curve_rows.append(pd.DataFrame(dict(session_id=sid, group=g, target=d["target"], variant=d["variant"],
                                                thr_norm=m6.THR_N, real=real, shuf_mean=shuf.mean(0), shuf_max=shuf.max(0))))
            d.update(med=med, sc=sc, thr_norm=thn, thr=med + sc * thn, reached=reached,
                     shuf_rate=float(np.interp(thn, m6.THR_N, shuf.mean(0))),
                     Wp=np.stack([perm_within(fin["w"], st, rng) for _ in range(N_PERM)], 1))
        # data: candidates and identity gate
        for k, d in enumerate(dec):
            fin = d["model"]
            E = m1.detect(raw[:, k], valid, d["thr"])
            Z = (Xc[E[:, 2]] - fin["mu"]) / fin["sd"]
            d["E"], d["p_id"] = E, identity_p(Z, fin["w"], d["Wp"])
        # time-shift null with the same threshold and gate
        ts = np.zeros((len(dec), N_NULL, 4))                              # cand all, kept all, cand quiet, kept quiet
        for j in range(N_NULL):
            lag = rng.uniform(*m1.SHIFT_S, len(S)) * rng.choice([-1, 1], len(S))
            lo_ok, hi_ok = D["t_lo"] + np.clip(lag, 0, None).max(), D["t_hi"] + np.clip(lag, None, 0).min()
            Xs = np.stack([m1.counts(s + l, grid) for s, l in zip(S, lag)], 1).astype(np.float32)
            T = Xs @ A + Cv
            vk = valid & (grid > lo_ok + m1.WIDTH) & (grid < hi_ok)
            mk, mq = vk.sum() * m1.STEP / 60, (vk & qg).sum() * m1.STEP / 60
            for k, d in enumerate(dec):
                pk = m1.detect(T[:, k], vk, d["thr"])[:, 2]
                keep = identity_p((Xs[pk] - d["model"]["mu"]) / d["model"]["sd"], d["model"]["w"], d["Wp"]) < P_GATE
                ts[k, j] = len(pk) / mk, keep.sum() / mk, qg[pk].sum() / mq, (keep & qg[pk]).sum() / mq
            del Xs, T
        R_ev = np.einsum("etu,ud->etd", P, A) + Cv
        for k, d in enumerate(dec):
            fin, thr, E, p_id = d["model"], d["thr"], d["E"], d["p_id"]
            kept = p_id < P_GATE
            neg, pos = d["oof"][d["y"] == 0], d["oof"][d["y"] == 1]
            dec_rows.append(dict(session_id=sid, mouse_id=sid[:5], stage=m5.STAGE[sid[:5]], group=g, target=d["target"],
                                 variant=d["variant"], n_units=len(S), n_train=d["n_train"], C=fin["C"],
                                 pauc=m1.pauc(d["y"], d["oof"]), thr=thr, thr_norm=d["thr_norm"], fdr_reached=d["reached"],
                                 cv_sens=float(np.mean(pos > thr)), cv_fpr=float(np.mean(neg > thr)),
                                 rate_cand=len(E) / vmin, rate=kept.sum() / vmin,
                                 rate_quiet=(kept & qg[E[:, 2]]).sum() / qmin, frac_kept=float(kept.mean()) if len(E) else np.nan,
                                 null_shift_cand=ts[k, :, 0].mean(), null_shift=ts[k, :, 1].mean(),
                                 null_shift_quiet=ts[k, :, 3].mean(), shuffled_template=d["shuf_rate"]))
            # held-out passive stimuli: crossing, and crossing + identity gate
            stim, sham = m1.TARGET_CLS[d["target"]]
            other = "A" if d["target"] == "whisker" else "W"
            jb = after[np.argmax(R_ev[:, after, k], 1)]                   # peak window per event
            pk_ev = R_ev[np.arange(len(ev)), jb, k]
            p_ev = identity_p((P[np.arange(len(ev)), jb] - fin["mu"]) / fin["sd"], fin["w"], d["Wp"])
            for es, eps in m6.eval_sets(d["variant"]).items():
                inset = (ev.context == "passive").to_numpy() & ev.epoch.isin(eps).to_numpy()
                row = dict(session_id=sid, mouse_id=sid[:5], stage=m5.STAGE[sid[:5]], group=g, target=d["target"],
                           variant=d["variant"], eval_set=es, thr=thr, fdr_reached=d["reached"])
                for name, cl in (("target", stim), ("sham", sham), ("other", other)):
                    m = np.flatnonzero(inset & ev.cls.eq(cl).to_numpy())
                    row[f"n_{name}"] = len(m)
                    if not len(m):
                        continue
                    cross = pk_ev[m] > thr
                    row[f"direct_cross_{name}"] = float(cross.mean())
                    row[f"direct_{name}"] = float((cross & (p_ev[m] < P_GATE)).mean())
                    Xi = Xc[inj_pos][None, :, :] + delta[m][:, None, :]                       # trial x window x unit
                    ri = Xi @ d["a"] + d["c"]
                    pi = identity_p(((Xi - fin["mu"]) / fin["sd"]).reshape(-1, len(S)), fin["w"], d["Wp"]).reshape(ri.shape)
                    row[f"inj_cross_{name}"] = float((ri > thr).mean())
                    row[f"inj_{name}"] = float(((ri > thr) & (pi < P_GATE)).mean())
                held_rows.append(row)
            if d["variant"] != "all":
                continue
            tp = grid[E[:, 2]]
            C = pd.DataFrame(dict(session_id=sid, group=g, target=d["target"], t_start=grid[E[:, 0]] - m1.WIDTH, t_end=grid[E[:, 1]],
                                  t_peak=tp, peak=raw[E[:, 2], k], dur_ms=(E[:, 1] - E[:, 0] + 1) * m1.STEP * 1000,
                                  quiet=qg[E[:, 2]], epoch=[m5.epoch_of(x, D["epochs"], None) for x in tp], p_identity=p_id, kept=kept))
            C.to_parquet(out / f"candidates_{d['target']}_{g.replace(' ', '_')}.parquet")
            events[(d["target"], g)] = C[C.kept].reset_index(drop=True)
            np.savez_compressed(out / f"continuous_{d['target']}_{g.replace(' ', '_')}.npz", t=grid, raw=raw[:, k].astype(np.float32),
                                valid=valid, quiet=qg, thr=thr, fdr_reached=d["reached"], w=fin["w"], b=fin["b"], mu=fin["mu"],
                                sd=fin["sd"], unit_index=sel, cluster_id=units.cluster_id.to_numpy(str)[sel],
                                electrode_group=units.electrode_group.to_numpy(str)[sel], oof=d["oof"], y=d["y"])
            # leave-one-unit-out participation
            U = participation(raw[:, k], Xc, valid, thr, fin, d["a"], d["Wp"], rng)
            ew, pw = evoked(P, ev, i_tr, i_bs, "W")
            ea, pa = evoked(P, ev, i_tr, i_bs, "A")
            mm = meta.iloc[sel].reset_index(drop=True)
            unit_rows.append(pd.concat([pd.DataFrame(dict(
                mouse_id=sid[:5], session_id=sid, electrode_group=units.electrode_group.to_numpy(str)[sel],
                cluster_id=units.cluster_id.to_numpy(str)[sel], area_group=g, area_acronym_custom=units.area_acronym_custom.to_numpy()[sel],
                stage=m5.STAGE[sid[:5]], target=d["target"], weight=fin["w"], evoked_whisker=ew, p_evoked_whisker=pw,
                evoked_auditory=ea, p_evoked_auditory=pa, waveform_type=mm.waveform_type.to_numpy(),
                sensory_label=mm.sensory_label.to_numpy(), reward_group=mm.reward_group.to_numpy(), thr=thr,
                fdr_reached=d["reached"])), U], axis=1))
        del Xc, P, raw, R_ev
        print(f"{sid} {g}: {len(S)} units | {time.time() - t0:.0f} s", flush=True)
    chance = m6.coincidence(events)
    for (target, g), E in events.items():
        E.to_parquet(out / f"events_{target}_{g.replace(' ', '_')}.parquet")
    Dd = pd.DataFrame(dec_rows).merge(pd.DataFrame([dict(
        target=t, group=g, variant="all", n_events=len(E), frac_coinc_other_target=float(E.coinc_other.mean()) if len(E) else np.nan,
        frac_coinc_other_target_chance=chance.get((t, g), np.nan), frac_cross_area=float((E.n_areas > 0).mean()) if len(E) else np.nan)
        for (t, g), E in events.items()]), on=["target", "group", "variant"], how="left")
    Dd.to_csv(out / "decoders.csv", index=False)
    pd.DataFrame(held_rows).to_csv(out / "heldout.csv", index=False)
    pd.concat(unit_rows, ignore_index=True).to_parquet(out / "units.parquet")
    pd.concat(curve_rows, ignore_index=True).to_parquet(out / "curves.parquet")
    (out / "params.json").write_text(json.dumps(dict(
        session_id=sid, stage=m5.STAGE[sid[:5]], rule=RULE, fdr=FDR, n_shuf=N_SHUF, n_null=N_NULL, n_perm=N_PERM, n_strata=N_STRATA,
        p_gate=P_GATE,
        n_inj=N_INJ, n_pshift=N_PSHIFT, pshift_s=PSHIFT_S, base_end=m6.BASE_END, coinc=m6.COINC, targets=m6.TARGETS,
        variants=m6.VARIANTS, held=m6.HELD, train_win=m1.TRAIN_WIN, step=m1.STEP, excl=m1.EXCL, shift_s=m1.SHIFT_S, merge=m1.MERGE,
        det_win=m1.DET_WIN, quiet_gap=m1.QUIET_GAP, whisk_max=m1.WHISK_MAX, sd_floor=m1.SD_FLOOR, valid_min=float(vmin),
        quiet_min=float(qmin), start_excl=m6.START_EXCL, epochs=D["epochs"], date=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=str))
    np.savez_compressed(out / "behaviour.npz", licks=D["licks"], **{f"{k}_t": v[0] for k, v in D["dlc"].items()},
                        **{f"{k}_v": v[1] for k, v in D["dlc"].items()})
    pd.set_option("display.width", 250)
    print(Dd[Dd.variant == "all"][["group", "target", "pauc", "thr_norm", "fdr_reached", "rate_cand", "rate", "frac_kept", "null_shift",
                                    "shuffled_template", "frac_coinc_other_target", "frac_cross_area"]].round(3).to_string(index=False))
    print("DONE", sid, f"{time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------------ summary
def summary():
    plt = m2.setup()
    col = m2.colors()
    load = lambda name, rd: pd.concat([rd(f) for f in sorted((m1.HOME / "pilot").glob(f"*/{OUT}/{name}"))], ignore_index=True)
    Dd, Hd, Uu = load("decoders.csv", pd.read_csv), load("heldout.csv", pd.read_csv), load("units.parquet", pd.read_parquet)
    Dd.to_csv(m1.HOME / "content_decoders.csv", index=False)
    Hd.to_csv(m1.HOME / "content_heldout.csv", index=False)
    Uu.to_parquet(m1.HOME / "content_units.parquet")
    G = [g for g in m1.GROUPS if g in set(Dd.group)]
    sids = sorted(Dd.session_id.unique(), key=lambda s: (list(m5.STAGE).index(s[:5]), s))
    x = np.arange(len(sids))
    # unit categories for participation (whisker decoders): evoked whisker response, weight sign, waveform
    Uw = Uu[(Uu.target == "whisker") & Uu.p_part.notna()].copy()
    Uw["resp"] = np.where((Uw.p_evoked_whisker < 0.05) & (Uw.evoked_whisker > 0), "whisker-resp.", "not resp.")
    Uw["sign"] = np.where(Uw.weight > 0, "w > 0", "w <= 0")
    Uw["wf"] = Uw.waveform_type.fillna("n/a")
    cats = [("resp", ["whisker-resp.", "not resp."]), ("sign", ["w > 0", "w <= 0"]), ("wf", ["rsu", "fsu"])]
    fig, axs = plt.subplots(len(G), 4, figsize=(7.4, 1.45 * len(G)), squeeze=False, gridspec_kw=dict(width_ratios=[1.5, 1.2, 1.2, 1.3]))
    for i, g in enumerate(G):
        cc = col.get(g, "k")
        for tgt, off, mk in (("whisker", -0.15, "o"), ("auditory", 0.15, "s")):
            q = Dd[(Dd.group == g) & (Dd.target == tgt) & (Dd.variant == "all")].set_index("session_id").reindex(sids)
            fc = [cc if r else "w" for r in q.fdr_reached.fillna(False)]
            ax = axs[i, 0]
            ax.scatter(x + off, q.rate.clip(lower=1e-3), s=9, marker=mk, facecolors=fc, edgecolors=cc, lw=0.5, zorder=3,
                       label=f"{tgt}: kept" if i == 0 else None)
            ax.scatter(x + off, q.rate_cand.clip(lower=1e-3), s=4, marker=mk, color="0.75", lw=0, zorder=2,
                       label="candidates" if i == 0 and tgt == "whisker" else None)
            ax.scatter(x + off, q.null_shift.clip(lower=1e-3), s=5, marker="_", color="k", lw=0.7, zorder=3,
                       label="time-shift null, kept" if i == 0 and tgt == "whisker" else None)
            ax.scatter(x + off, q.shuffled_template.clip(lower=1e-3), s=5, marker="x", color="0.5", lw=0.5, zorder=3,
                       label="shuffled template" if i == 0 and tgt == "whisker" else None)
        ax = axs[i, 0]
        ax.set_yscale("log"); ax.set_xticks(x, [s[:5] for s in sids], rotation=90, fontsize=4.5)
        ax.set_ylabel(f"{m2.short(g)}\nevents / min", color=cc)
        if i == 0:
            ax.set_title("filled: FDR reached; circles whisker, squares auditory", loc="left", fontsize=5.2)
            ax.legend(frameon=False, fontsize=3.6, ncol=2)
        for j, (k, lab) in enumerate((("direct", "held-out passive, direct (gated)"), ("inj", "injected into quiet (gated)"))):
            ax = axs[i, 1 + j]
            for es, mk in (("pre", "^"), ("post", "v"), ("pre+post", "o")):
                vv = {"pre": "ho_pre", "post": "ho_post", "pre+post": "ho_both"}[es]
                q = Hd[(Hd.group == g) & (Hd.target == "whisker") & (Hd.variant == vv) & (Hd.eval_set == es)].set_index("session_id").reindex(sids)
                for nm, fcol in (("target", cc), ("other", "0.45"), ("sham", "0.8")):
                    ax.scatter(x + {"pre": -0.2, "post": 0.2, "pre+post": 0}[es], q.get(f"{k}_{nm}"), s=6, marker=mk, color=fcol, lw=0,
                               label=f"{nm} {es}" if i == 0 and j == 0 else None)
            ax.set_ylim(-0.03, 1.03); ax.set_xticks(x, [s[:5] for s in sids], rotation=90, fontsize=4.5)
            if i == 0:
                ax.set_title(lab, loc="left", fontsize=5.2)
                if j == 0:
                    ax.legend(frameon=False, fontsize=3.4, ncol=3)
        ax = axs[i, 3]
        q = Uw[Uw.area_group == g]
        xx = 0
        ticks = []
        for cname, levels in cats:
            for lv in levels:
                fr = q[q[cname] == lv].groupby("session_id").p_part.apply(lambda p: (p < 0.05).mean())
                if len(fr):
                    ax.scatter(np.full(len(fr), xx) + np.linspace(-0.2, 0.2, len(fr)), fr, s=4, color=cc, lw=0)
                    ax.plot([xx - 0.3, xx + 0.3], [fr.median()] * 2, color="k", lw=0.8)
                ticks.append((xx, lv))
                xx += 1
            xx += 0.5
        ax.axhline(0.05, color="0.6", lw=0.4, ls=":")
        ax.set_xticks([t for t, _ in ticks], [l for _, l in ticks], rotation=90, fontsize=4.5); ax.set_ylim(-0.02, 1.02)
        if i == 0:
            ax.set_title("leave-one-out participation (whisker):\nfraction of units p < 0.05 per session", loc="left", fontsize=5.0)
    fig.suptitle(f"Per-decoder threshold: {RULE}; held-out passive stimuli; leave-one-unit-out participation; sessions: day 0 R+, "
                 "day 0 R-, expert R+, expert R-", x=0.01, ha="left", fontsize=6, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    m2.save(fig, m1.HOME / "figures", "fig11_content_summary")
    plt.close(fig)
    # real vs shuffled template: crossing rate per threshold, and their ratio (FDR rule = ratio >= 1 / FDR)
    Cu = load("curves.parquet", pd.read_parquet)
    Cu = Cu[Cu.variant == "all"].merge(Dd[Dd.variant == "all"][["session_id", "group", "target", "pauc", "thr_norm", "fdr_reached"]]
                                       .rename(columns=dict(thr_norm="thr_sel")), on=["session_id", "group", "target"])
    Cu.to_parquet(m1.HOME / "content_curves.parquet")
    fig, axs = plt.subplots(len(G), 4, figsize=(7.4, 1.45 * len(G)), squeeze=False)
    for i, g in enumerate(G):
        cc = col.get(g, "k")
        for j, tgt in enumerate(m6.TARGETS):
            a0, a1 = axs[i, 2 * j], axs[i, 2 * j + 1]
            for (s, good), q in Cu[(Cu.group == g) & (Cu.target == tgt)].groupby(["session_id", Cu.pauc >= 0.8]):
                lw, al = (0.7, 0.9) if good else (0.4, 0.35)
                a0.plot(q.thr_norm, q.real.clip(lower=1e-3), color=cc, lw=lw, alpha=al)
                a0.plot(q.thr_norm, q.shuf_mean.clip(lower=1e-3), color="0.4", lw=lw, alpha=al, ls="--")
                ratio = np.where(q.shuf_mean > 0, q.real / q.shuf_mean.where(q.shuf_mean > 0), np.nan)
                a1.plot(q.thr_norm, ratio, color=cc, lw=lw, alpha=al)
                if q.fdr_reached.iloc[0]:
                    a1.scatter(q.thr_sel.iloc[0], 1 / FDR, s=8, color=cc, edgecolors="k", lw=0.3, zorder=3)
            a0.set_yscale("log"); a0.set_ylim(1e-3, 300); a0.set_xlim(1, 30)
            a1.set_yscale("log"); a1.set_ylim(0.1, 300); a1.set_xlim(1, 30)
            a1.axhline(1, color="0.6", lw=0.4, ls=":"); a1.axhline(1 / FDR, color="k", lw=0.5, ls="--")
            if i == 0:
                a0.set_title(f"{tgt} decoder: crossings / min\nsolid real, dashed shuffled template", loc="left", fontsize=5.0)
                a1.set_title(f"{tgt}: real / shuffled\n1 = no template-specific excess; dashed = FDR rule", loc="left", fontsize=5.0)
            if i == len(G) - 1:
                a0.set_xlabel("threshold (MAD)"); a1.set_xlabel("threshold (MAD)")
        axs[i, 0].set_ylabel(m2.short(g), color=cc)
    fig.suptitle("Real decoder vs rate-matched shuffled template (variant all; thick: pAUC >= 0.8; dot: threshold reached)",
                 x=0.01, ha="left", fontsize=6, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    m2.save(fig, m1.HOME / "figures", "fig12_real_vs_shuffled")
    plt.close(fig)
    pd.set_option("display.width", 260)
    print(Dd[Dd.variant == "all"][["session_id", "group", "target", "pauc", "thr_norm", "fdr_reached", "rate_cand", "rate", "rate_quiet",
                                    "null_shift", "shuffled_template", "frac_coinc_other_target", "frac_coinc_other_target_chance",
                                    "frac_cross_area"]].round(3).to_string(index=False))


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
            m6.fig_examples(s, tgt, sub_dir=OUT, rule=RULE, fig_name="fig10_examples")
    if a.summary:
        summary()
