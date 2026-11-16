"""012 -- Browse reactivation events at millisecond resolution; movements aligned to event peaks (user 2026-10-30).

Data: 009 saved files (stable units), whole-brain decoder ("All"), threshold = 009 THR_MAIN (v6: median over 100 rate-matched weight shuffles of the 99.9th percentile). Events (009 v4): runs
of the readout above threshold anywhere in valid time (outside trials onset - 10 ms .. + 5 s, passive blocks, borders), each
labelled with its movement (none / whisk / jaw / lick, combinations; 009 event table); panel titles give the label.
--browse: one multi-page PDF per session x target, figures/full_<unit_set>/event_browser/<session>_<target>.pdf.
  page 1, first panel: reference = held-out passive target stimuli (readout mean +- s.d. aligned to onset, raster of the
  stimulus closest to the median peak), same layout and scale as the events.
  one panel per event (EV_PER_PAGE per page, up to EV_MAX events spread evenly over the session), +-HALF around the peak:
  raster of the N_TOP units with the largest |weight| (sorted by weight; red positive, blue negative weights; tick = spike),
  readout (MAD units; window end time, 30-ms window, 5-ms stride; threshold red; event extent shaded), whisker speed, jaw speed
  (only when lick-jaw sync passes, 008), licks (piezo, ticks).
  Sessions (--sessions): default = 010 example sessions (25 / 50 / 75th percentile of each cohort's still-event rate, day 0,
  balanced accuracy >= GOOD_BACC).
--eta: fig23_event_movements_<target>: whisker speed, jaw speed and lick rate aligned to event peaks (+-ETA_HALF), mean over
  events per session, then mean +- s.e.m. over sessions per cohort (R+ / R-), day 0 and expert; dashed: the same for random
  valid times (N_CTRL x the events' number); table event_movements_<unit_set>.csv per session: events,
  fraction of events with whisker speed > WHISK_MAX (deg/s) at the peak, the same at random valid times, fraction still.
"""
from __future__ import annotations

import argparse
import importlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
m1 = importlib.import_module("001_session_reactivation")
m2 = importlib.import_module("002_session_figures")
m8 = importlib.import_module("008_sensitivity_threshold_cohorts")
m9 = importlib.import_module("009_full_dataset_events")
m10 = importlib.import_module("010_quiet_window_examples")
HALF = 0.3                                       # s around the event peak (browser)
CTX = 2.0                                        # s: context strip (readout only) around the peak
N_TOP = 60
EV_PER_PAGE = 8
EV_MAX = 64
ETA_HALF = 2.0
ETA_STEP = 0.01
N_CTRL = 10
WPOS, WNEG = "#b2182b", "#2166ac"
COH = m10.COH
GROUP = m9.m8.ALL


def load(sid, target, unit_set):
    out = m9.ROOT / unit_set / sid
    S = np.load(out / "session.npz")
    C = dict(np.load(out / f"continuous_{target}_{GROUP}.npz"))
    B = dict(np.load(out / "behaviour.npz", allow_pickle=True))
    Pm = json.loads((out / "params.json").read_text())
    E = pd.read_parquet(out / f"events_{target}_{GROUP}.parquet")
    t = C["t"]
    i0 = np.clip(np.searchsorted(t, E.t_start.to_numpy() + m1.WIDTH - 1e-6), 0, len(t) - 1)  # first window of the event
    E = E[C["valid"][i0]].reset_index(drop=True)
    jaw = None
    if Pm.get("jaw_ok") and "jaw_t" in B:
        jt, jv = B["jaw_t"], B["jaw_v"]
        jaw = (jt, np.abs(np.gradient(jv, jt)))
    whisk = (B["whisker_speed_t"], B["whisker_speed_v"]) if "whisker_speed_t" in B else None
    return dict(C=C, S=S, B=B, E=E, jaw=jaw, whisk=whisk, licks=B["licks"], P=Pm)


def trace(ax, tv, t0, half, color, ylab, scale=1.0):
    if tv is None:
        ax.axis("off")
        return
    m = (tv[0] >= t0 - half) & (tv[0] <= t0 + half)
    ax.plot((tv[0][m] - t0) * 1000 * scale, tv[1][m], color=color, lw=0.5)
    ax.set_ylabel(ylab, fontsize=4.5, rotation=0, ha="right", va="center")


def panel(axs, Q, spikes, order, t0, title, ref=None):
    """one event (or the stimulus reference): raster, readout, whisker, jaw, licks around t0"""
    C = Q["C"]
    a_r, a_z, a_w, a_j, a_c = axs
    w = C["w"][order]
    for k, (u, wk) in enumerate(zip(order, w)):
        st = spikes[u]
        st = st[(st >= t0 - HALF) & (st < t0 + HALF)] - t0
        a_r.vlines(st * 1000, k, k + 0.85, color=WPOS if wk > 0 else WNEG, lw=0.45)
    a_r.set_ylim(len(order), 0)
    a_r.set_yticks([])
    a_r.set_title(title, fontsize=4.6, loc="left")
    t = C["t"]
    m = (t >= t0 - HALF) & (t <= t0 + HALF)
    if ref is not None:                                               # stimulus reference: mean +- s.d. over stimuli
        x, mu, sd = ref
        a_z.fill_between(x * 1000, mu - sd, mu + sd, color="0.8", lw=0)
        a_z.plot(x * 1000, mu, color="k", lw=0.6)
        a_r.axvspan(m1.TRAIN_WIN[0] * 1000, m1.TRAIN_WIN[1] * 1000, color="crimson", alpha=0.08, lw=0)
    else:
        a_z.plot((t[m] - t0) * 1000, C["z"][m], color="k", lw=0.6)
        e = Q["E"]
        r = e[(e.t_peak == t0)]
        if len(r):
            for a in (a_r, a_z):
                a.axvspan((r.t_start.iloc[0] - t0) * 1000, (r.t_end.iloc[0] - t0) * 1000, color="crimson", alpha=0.08, lw=0)
    a_z.axhline(float(C["thr"]), color="r", lw=0.7, ls="--")
    a_z.axvline(0, color="0.7", lw=0.3, ls=":")
    a_z.set_ylabel("readout\n(MAD)", fontsize=4.5)
    trace(a_w, Q["whisk"], t0, HALF, "0.3", "whisk\n(deg/s)")
    if Q["jaw"] is not None:
        trace(a_j, Q["jaw"], t0, HALF, "0.3", "jaw\nspeed")
    else:
        a_j.set_yticks([])
    lk = Q["licks"][(Q["licks"] >= t0 - HALF) & (Q["licks"] <= t0 + HALF)] - t0
    if a_j.axison:
        a_j.vlines(lk * 1000, 0, 1, transform=a_j.get_xaxis_transform(), color="tab:orange", lw=0.6)
    for a in axs[:4]:
        a.set_xlim(-HALF * 1000, HALF * 1000)
        a.tick_params(labelsize=4.2)
    for a in axs[:3]:
        a.tick_params(labelbottom=False)
    a_j.set_xlabel("time from peak (ms)" if ref is None else "time from stimulus onset (ms)", fontsize=4.5)
    # context strip: readout +-CTX s (valid time black, invalid grey), the +-HALF window shaded
    mc = (t >= t0 - CTX) & (t <= t0 + CTX)
    x, zc, vc = t[mc] - t0, C["z"][mc], C["valid"][mc]
    a_c.plot(x, np.where(vc, zc, np.nan), color="k", lw=0.4)
    a_c.plot(x, np.where(vc, np.nan, zc), color="0.75", lw=0.4)
    a_c.axhline(float(C["thr"]), color="r", lw=0.5, ls="--")
    a_c.axvspan(-HALF, HALF, color="gold", alpha=0.15, lw=0)
    a_c.set_xlim(-CTX, CTX)
    a_c.tick_params(labelsize=4.2)
    a_c.set_ylabel("readout +-2 s", fontsize=4.5)
    a_c.set_xlabel("s (grey: invalid, e.g. trial onset)", fontsize=4.2)


def browse(sid, target, unit_set):
    from matplotlib.backends.backend_pdf import PdfPages
    plt = m2.setup()
    Q = load(sid, target, unit_set)
    C, E, B = Q["C"], Q["E"], Q["B"]
    units, *_ = m1.load_session(sid)
    units = units[m9.unit_set_mask(sid, units, unit_set)].reset_index(drop=True)
    spikes = {u: np.asarray(units.spike_times.iloc[int(C["unit_index"][u])]) for u in range(len(C["w"]))}
    top = np.argsort(np.abs(C["w"]))[::-1][:N_TOP]
    order = top[np.argsort(C["w"][top])[::-1]]
    Dd = m9.decoders_table(sid, unit_set)
    d = Dd[(Dd.group == GROUP) & (Dd.target == target) & (Dd.thr_source == m9.THR_MAIN)].iloc[0]
    # stimulus reference: held-out passive target stimuli
    stim = m1.TARGET_CLS[target][0]
    ts = B["ev_t"][(B["ev_context"] == "passive") & (B["ev_cls"] == stim)]
    lags = np.round(np.arange(-HALF, HALF + 1e-9, m1.STEP), 6)
    idx = np.clip(np.searchsorted(C["t"], ts[:, None] + lags[None, :]), 0, len(C["t"]) - 1)
    Z = C["z"][idx]
    pk = Z[:, (lags >= m1.DET_WIN[0]) & (lags <= m1.DET_WIN[1])].max(1)
    t_ref = ts[np.argmin(np.abs(pk - np.median(pk)))] if len(ts) else None
    ev = E.iloc[np.unique(np.linspace(0, len(E) - 1, min(EV_MAX, len(E))).round().astype(int))] if len(E) else E
    items = ([("ref", t_ref)] if t_ref is not None else []) + [("ev", r) for r in ev.itertuples()]
    fdir = m1.HOME / "figures" / f"full_{unit_set}" / "event_browser"
    fdir.mkdir(parents=True, exist_ok=True)
    coh = m9.sessions_table().set_index("session_id").cohort.get(sid, "?")
    head = (f"{sid} ({coh}, {m9.stage_of(sid)}), whole-brain {target} decoder, {len(C['w'])} {unit_set} units, pAUC {d.pauc:.2f}, "
            f"threshold {d.thr_norm:.1f} MAD ({m9.THR_MAIN}); passive AUC (5-35 ms, passive in training) vs sham pre "
            f"{d.auc_sham_pre:.2f} / post {d.auc_sham_post:.2f}, vs other modality {d.auc_other_pre:.2f} / {d.auc_other_post:.2f}; "
            f"events {len(E)} ({d.rate:.4f}/s), still {int(E.still.sum())} ({d.rate_still:.4f}/s; shuffled template "
            f"{d.rate_shuf_still:.4f}/s, p {d.p_shuf_still:.2f}); "
            f"shown {len(ev)}\nraster: top {len(order)} units by |weight|, sorted by weight (red +, blue -); readout = 30-ms window "
            f"ending at t, 5-ms stride; shaded: event extent (first window start to last window end); orange: licks")
    with PdfPages(fdir / f"{sid}_{target}.pdf") as pdf:
        for p0 in range(0, max(len(items), 1), EV_PER_PAGE):
            fig = plt.figure(figsize=(11, 8.2), layout="constrained")
            fig.suptitle(head, x=0.01, ha="left", fontsize=6, weight="bold")
            sub = fig.subfigures(2, EV_PER_PAGE // 2).ravel()
            for k, (kind, it) in enumerate(items[p0:p0 + EV_PER_PAGE]):
                axs = sub[k].subplots(5, 1, gridspec_kw=dict(height_ratios=[3, 1.3, 0.6, 0.6, 0.8]))
                if kind == "ref":
                    panel(axs, Q, spikes, order, t_ref, f"reference: passive {target} stimulus (n {len(ts)})\nraster: median-peak "
                          "stimulus; readout mean +- s.d.", ref=(lags, Z.mean(0), Z.std(0)))
                else:
                    panel(axs, Q, spikes, order, float(it.t_peak), f"t {it.t_peak:.3f} s, peak {it.peak:.1f} MAD, "
                          f"{it.dur_ms:.0f} ms, id. p {it.p_identity:.2f}\nmovement: {it.movement}")
            for k in range(len(items[p0:p0 + EV_PER_PAGE]), EV_PER_PAGE):
                sub[k].subplots().axis("off")
            pdf.savefig(fig, dpi=200)
            if p0 == 0:                                                   # preview of the first page
                fig.savefig(fdir / f"{sid}_{target}_page1.png", dpi=130)
            plt.close(fig)
    print("browser", sid, target, len(E), "events ->", fdir / f"{sid}_{target}.pdf", flush=True)


def eta_session(sid, target, unit_set, rng):
    Q = load(sid, target, unit_set)
    C, E = Q["C"], Q["E"]
    lag = np.round(np.arange(-ETA_HALF, ETA_HALF + 1e-9, ETA_STEP), 6)
    q = np.flatnonzero(C["valid"])
    tp = E.t_peak.to_numpy()
    tc = C["t"][rng.choice(q, N_CTRL * max(len(tp), 1))] if len(q) else np.zeros(0)
    row = dict(session_id=sid, target=target, n_events=len(tp), valid_s=len(q) * m1.STEP, frac_still=float(E.still.mean()) if len(E) else np.nan)
    curves = {}
    for nm, tv in (("whisker", Q["whisk"]), ("jaw", Q["jaw"])):
        if tv is None or not np.isfinite(tv[1]).any():
            continue
        f = lambda x: np.interp(x, tv[0], np.nan_to_num(tv[1], nan=np.nanmedian(tv[1])), left=np.nan, right=np.nan)
        if len(tp):
            curves[nm] = np.nanmean(f(tp[:, None] + lag[None, :]), 0)
            curves[f"{nm}_ctrl"] = np.nanmean(f(tc[:, None] + lag[None, :]), 0)
        if nm == "whisker":
            row["frac_whisking_event"] = float(np.mean(f(tp) > m1.WHISK_MAX)) if len(tp) else np.nan
            row["frac_whisking_ctrl"] = float(np.mean(f(tc) > m1.WHISK_MAX)) if len(tc) else np.nan
    lk = Q["licks"]
    edges = np.r_[lag - ETA_STEP / 2, lag[-1] + ETA_STEP / 2]
    if len(tp):
        h = lambda x: np.mean([np.histogram(lk - t, edges)[0] for t in x], 0) / ETA_STEP
        curves["lick"], curves["lick_ctrl"] = h(tp), h(tc)
    return row, curves, lag


def eta(target, unit_set):
    plt = m2.setup()
    Dd = pd.read_csv(m1.HOME / f"full_decoders_{unit_set}.csv")
    D = Dd[(Dd.group == GROUP) & (Dd.target == target) & (Dd.thr_source == m9.THR_MAIN) & (Dd.bacc >= m9.GOOD_BACC)]
    rng = np.random.default_rng(0)
    rows, cur = [], {}
    for r in D.itertuples():
        row, c, lag = eta_session(r.session_id, target, unit_set, rng)
        rows.append(dict(row, cohort=r.cohort, stage=r.stage))
        cur[r.session_id] = c
    R = pd.DataFrame(rows)
    stages = [("day 0", "day 0"), ("expert", "expert")]
    meas = [("whisker", "whisker speed (deg/s)"), ("jaw", "jaw speed"), ("lick", "lick rate (/s)")]
    fig, axs = plt.subplots(len(stages), len(meas), figsize=(7.4, 4.2), squeeze=False)
    for i, (stg, lab_s) in enumerate(stages):
        for j, (nm, lab) in enumerate(meas):
            ax = axs[i, j]
            for coh, cl in COH.items():
                ids = R[(R.stage == f"{stg} {coh}") & (R.n_events > 0)].session_id
                for sfx, ls in (("", "-"), ("_ctrl", "--")):
                    X = np.array([cur[s][nm + sfx] for s in ids if nm + sfx in cur[s]])
                    if not len(X):
                        continue
                    mu, se = np.nanmean(X, 0), np.nanstd(X, 0) / np.sqrt(len(X))
                    ax.plot(lag, mu, color=cl, ls=ls, lw=0.8 if sfx == "" else 0.6,
                            label=f"{coh} {'events' if sfx == '' else 'random valid times'} (n {len(X)})")
                    if sfx == "":
                        ax.fill_between(lag, mu - se, mu + se, color=cl, alpha=0.2, lw=0)
            ax.axvline(0, color="0.6", lw=0.4, ls=":")
            ax.set_title(f"{lab_s}: {lab}", fontsize=5.5)
            ax.tick_params(labelsize=4.5)
            if i == len(stages) - 1:
                ax.set_xlabel("time from event peak (s)", fontsize=5)
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=4)
    fig.suptitle(f"movements aligned to event peaks (all valid time), whole-brain {target} decoders ({unit_set} units, balanced accuracy >= "
                 f"{m9.GOOD_BACC}), {m9.THR_TEXT}; mean over events per "
                 f"session, mean +- s.e.m. over sessions; dashed: random valid times; jaw only where lick-jaw "
                 f"sync passes", x=0.01, ha="left", fontsize=5.4, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    m2.save(fig, m1.HOME / "figures" / f"full_{unit_set}", f"fig23_event_movements_{target}")
    plt.close(fig)
    R.to_csv(m1.HOME / f"event_movements_{target}_{unit_set}.csv", index=False)
    pd.set_option("display.width", 220)
    print(R.groupby("stage")[["n_events", "frac_still", "frac_whisking_event", "frac_whisking_ctrl"]].median().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs="*", default=["whisker", "auditory"])
    ap.add_argument("--unit-set", default="stable")
    ap.add_argument("--sessions", nargs="*", default=None, help="browser sessions (default: 010 example sessions)")
    ap.add_argument("--browse", action="store_true")
    ap.add_argument("--eta", action="store_true")
    a = ap.parse_args()
    for tg in a.target:
        if a.browse:
            ss = a.sessions
            if ss is None:
                Dd = pd.read_csv(m1.HOME / f"full_decoders_{a.unit_set}.csv")
                D = Dd[(Dd.group == GROUP) & (Dd.target == tg) & (Dd.thr_source == m9.THR_MAIN)
                       & (Dd.bacc >= m9.GOOD_BACC) & Dd.stage.str.startswith("day 0")].dropna(subset=["rate_still"])
                ss = [r.session_id for c in COH for r in m10.pick(D, c)]
            for s in ss:
                browse(s, tg, a.unit_set)
        if a.eta:
            eta(tg, a.unit_set)
