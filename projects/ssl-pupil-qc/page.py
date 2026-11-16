"""Pupil-QC pages per session (user specification, 2026-11-03, refined the same day):
page 1, QC summary (header, session timeline, zoom, state diagnostic, event-triggered averages, trial rasters, mean
response, per-trial QC, loss durations, spectrum, likelihoods, pupil x motion); page 2, example frames (valid frames,
6 blinks and 6 failed events as 5-frame sequences)."""

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from scipy import signal

import video
from pupil_qc import PARAMS, POINTS, STATE, VERDICT, _runs, event_onsets, failed_events

MM = 1 / 25.4
STATE_COLORS = ["#a5d6a7", "#42a5f5", "#ff9800", "#c62828"]  # valid, blink, squint, failed
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NAMES = {0: "valid", 1: "blink", 2: "squint", 3: "failed"}
OUTCOME_COLORS = {"W hit": "#f7b519", "W miss": "#fbe3a6", "A hit": "#2c2cdb", "A miss": "#aab0f0", "FA": "#262626",
                  "CR": "#b0b0b0"}
LABEL_COLORS = {"valid": "#a5d6a7", "blink": "#42a5f5", "squint": "#ff9800", "failed": "#c62828"}
VERDICT_COLORS = {"pass": "#2e7d32", "warn": "#f9a825", "fail": "#c62828"}
Z_THRESHOLD = 3.0  # eye-ROI intensity excursion counted as a lid closure (robust z)


def _rc():
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 5,
                         "axes.titlesize": 5.5, "axes.labelsize": 5, "xtick.labelsize": 4.5, "ytick.labelsize": 4.5,
                         "legend.fontsize": 4.5, "axes.linewidth": 0.4, "lines.linewidth": 0.5,
                         "xtick.major.width": 0.4, "ytick.major.width": 0.4, "xtick.major.size": 1.5,
                         "ytick.major.size": 1.5, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42, "svg.fonttype": "none"})


def _bin(x, n_per_bin, how="mean"):
    n = len(x) // n_per_bin * n_per_bin
    a = np.asarray(x[:n], float).reshape(-1, n_per_bin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {"mean": np.nanmean, "max": np.nanmax, "min": np.nanmin}[how](a, 1)


def _state_bins(state, n_per_bin, min_frac=0.1):
    n = len(state) // n_per_bin * n_per_bin
    a = state[:n].reshape(-1, n_per_bin)
    frac = np.stack([(a == s).mean(1) for s in range(4)], 1)
    worst = frac[:, 1:].argmax(1) + 1
    return np.where(frac[:, 1:].sum(1) >= min_frac, worst, 0)


def eyelid_proxy(r):
    """Eye-ROI intensity robust z (video) or number of tracked pupil points."""
    if r.get("intensity_z") is not None:
        return r["intensity_z"], "Eye-ROI intensity (z)", Z_THRESHOLD
    return r["c"]["n_good"].astype(float), "Tracked pupil points", 1.5


def _letter(ax, s, x=-0.12, y=1.08):
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", fontsize=6.5, va="bottom")


# ----------------------------------------------------------------------------------------------- page 1 panels

def _timeline(axs, r, a, b, bin_s, title=None):
    c, fs, t = r["c"], r["meta"]["fs"], r["df"].t.to_numpy()
    k = max(int(round(bin_s * fs)), 1)
    sl = slice(a, b)
    tt = _bin(t[sl], k)
    t0 = t[a]
    unit, scale = ("min", 60) if (b - a) / fs > 300 else ("s", 1)
    x = (tt - t0) / scale
    axs[0].plot(x, _bin(c["area"][sl], k), color="0.7", lw=0.3)
    axs[0].plot(x, _bin(r["clean"][sl], k), color="k", lw=0.4)
    lo, hi = np.nanpercentile(r["clean"], [0.5, 99.5])
    axs[0].set_ylim(max(lo - 0.3 * (hi - lo), 0), hi + 0.3 * (hi - lo))
    axs[0].set_ylabel("Area\n(px²)")
    axs[1].plot(x, _bin(c["ratio_norm"][sl], k, "min" if k > 1 else "mean"), color="0.25", lw=0.3)
    axs[1].axhline(PARAMS["squint_ratio"], color="#ff9800", lw=0.4, ls="--")
    axs[1].axhline(1, color="0.7", lw=0.3)
    axs[1].set_ylim(0.2, 1.6)
    axs[1].set_ylabel("Height/width\n(norm.)", fontsize=4.5)
    L = np.stack([_bin(c["L"][sl, i], k) for i in range(4)])
    ext = [x[0], x[-1] if len(x) > 1 else x[0] + 1]
    axs[2].imshow(L, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest", extent=ext + [3.5, -0.5])
    axs[2].set_yticks(range(4))
    axs[2].set_yticklabels(POINTS, fontsize=4)
    axs[2].set_ylabel("Likelihood")
    axs[3].imshow(_state_bins(c["state"][sl], k)[None], aspect="auto", cmap=STATE_CMAP, vmin=-0.5, vmax=3.5,
                  interpolation="nearest", extent=ext + [0, 1])
    axs[3].set_yticks([])
    axs[3].set_ylabel("State", rotation=0, ha="right", va="center")
    mot = r["df"].get("whisker_velocity", r["df"].get("jaw_velocity"))
    if mot is not None:
        mv = _bin(mot.to_numpy()[sl], k)
        axs[4].plot(x, np.clip(mv / (np.nanpercentile(mv, 99) + 1e-9), 0, 1.2), color="0.5", lw=0.3)
    tr = r["trials"]
    inwin = tr[(tr.start_time >= t[a]) & (tr.start_time <= t[b - 1])]
    axs[4].vlines((inwin.start_time - t0) / scale, 1.3, 1.8, colors=[OUTCOME_COLORS.get(o, "k") for o in inwin.outcome],
                  lw=0.5 if scale == 1 else 0.25)
    axs[4].set_ylim(0, 1.9)
    axs[4].set_yticks([])
    axs[4].set_ylabel("Trials\nmotion", fontsize=4.5)
    axs[4].set_xlabel(f"Time ({unit})")
    for ax in axs:
        ax.set_xlim(x[0], x[-1])
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    if title:
        axs[0].set_title(title, loc="left")


def _zoom_window(r, win_s=60):
    fs = r["meta"]["fs"]
    bl, fl = event_onsets(r["c"], fs)
    n = len(r["c"]["state"])
    w = int(win_s * fs)
    best, best_a = (-1, -1), 0
    for a in range(0, max(n - w, 1), int(10 * fs)):
        nb, nf = np.sum((bl >= a) & (bl < a + w)), np.sum((fl >= a) & (fl < a + w))
        if (min(nb, nf), nb + nf) > best:
            best, best_a = (min(nb, nf), nb + nf), a
    return best_a, min(best_a + w, n)


def _scatter(ax, r):
    c = r["c"]
    proxy, plabel, thr = eyelid_proxy(r)
    y = c["L"].mean(1)
    x = proxy.copy()
    if r.get("intensity_z") is None:
        x = x + np.random.default_rng(0).uniform(-0.3, 0.3, len(x))
    ok = np.isfinite(x)
    lo, hi = np.nanpercentile(x[ok], [0.1, 99.9])
    ax.hist2d(x[ok], y[ok], bins=[np.linspace(lo, hi, 80), np.linspace(0, 1, 60)], cmap="Greys",
              norm=mpl.colors.LogNorm())
    sub = np.random.default_rng(1).choice(np.flatnonzero(ok), min(ok.sum(), 15000), replace=False)
    ax.scatter(x[sub], y[sub], c=[STATE_COLORS[s] for s in c["state"][sub]], s=0.4, lw=0, alpha=0.7, rasterized=True)
    ax.axhline(PARAMS["p_cut"], color="k", lw=0.4, ls="--")
    ax.axvline(thr, color="#42a5f5", lw=0.4, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(plabel)
    ax.set_ylabel("Mean pupil likelihood")
    ax.set_title("State diagnostic (green valid, blue blink, orange squint, red failed)", loc="left", fontsize=4.8)


def _triggered(axs, r):
    fs = r["meta"]["fs"]
    proxy, plabel, _ = eyelid_proxy(r)
    meanL = r["c"]["L"].mean(1)
    bl, fl = event_onsets(r["c"], fs)
    w = int(fs)
    lag = np.arange(-w, w) / fs
    for ax, ons, name in ((axs[0], bl, "Blink onsets"), (axs[1], fl, "Low-likelihood failed onsets")):
        ons = ons[(ons > w) & (ons < len(meanL) - w)]
        ax2 = ax.twinx()
        if len(ons):
            idx = ons[:, None] + np.arange(-w, w)[None]
            ax.plot(lag, np.nanmean(proxy[idx], 0), color="0.2")
            ax2.plot(lag, np.nanmean(meanL[idx], 0), color="#7b1fa2")
        ax.set_title(f"{name} (n = {len(ons)})", loc="left", fontsize=4.8)
        ax.set_ylabel(plabel, fontsize=4.2)
        ax2.set_ylabel("Mean likelihood", color="#7b1fa2", fontsize=4.2)
        ax2.set_ylim(0, 1.05)
        ax2.tick_params(colors="#7b1fa2", labelsize=4)
        ax2.spines["right"].set_visible(True)
        ax.axvline(0, color="0.6", lw=0.3)
    axs[1].set_xlabel("Time from onset (s)")
    axs[0].tick_params(labelbottom=False)


def _trial_matrix(r, bin_s=0.05):
    t = r["df"].t.to_numpy()
    w0, w1 = PARAMS["trial_window_s"]
    edges = np.arange(w0, w1 + 1e-9, bin_s)
    tq = r["trials_qc"]
    D = np.full((len(tq), len(edges) - 1), np.nan)
    S = np.zeros_like(D, dtype=int)
    st, clean = r["c"]["state"], r["clean"]
    for j, row in enumerate(tq.itertuples()):
        idx = np.searchsorted(t, row.start_time + edges)
        for k in range(len(edges) - 1):
            a, b = idx[k], idx[k + 1]
            if b <= a:
                continue
            seg = st[a:b]
            if (seg == STATE["blink"]).any():
                S[j, k] = 1
            elif (seg == STATE["squint"]).any():
                S[j, k] = 2
            elif np.isnan(clean[a:b]).any():
                S[j, k] = 3
            if np.isfinite(clean[a:b]).any():
                D[j, k] = np.nanmean(clean[a:b])
        base = (edges[:-1] >= PARAMS["baseline_s"][0]) & (edges[:-1] < PARAMS["baseline_s"][1])
        D[j] -= np.nanmean(D[j, base]) if np.isfinite(D[j, base]).any() else np.nan
    return edges, D, S


def _rasters(axF, axG, axS, r, edges, D, S):
    tq, tr = r["trials_qc"], r["trials"]
    vmax = np.nanpercentile(np.abs(D), 98) if np.isfinite(D).any() else 1
    rgb = mpl.colormaps["RdBu_r"](np.clip((D + vmax) / (2 * vmax), 0, 1))
    rgb[S == 1] = (1, 1, 1, 1)
    rgb[S == 2] = mpl.colors.to_rgba("#ffcc80")
    rgb[(S == 3) | (np.isnan(D) & (S == 0))] = (0.6, 0.6, 0.6, 1)
    ext = [edges[0], edges[-1], len(tq) - 0.5, -0.5]
    axF.imshow(rgb, aspect="auto", extent=ext, interpolation="nearest")
    trs = tr.loc[tq.trial]
    for ax in (axF, axG):
        ax.scatter(trs.lick - trs.start_time, np.arange(len(tq)), s=0.6, c="k", lw=0)
        ax.scatter(trs.reward - trs.start_time, np.arange(len(tq)), s=1.2, c="#00b0ff", marker="|", lw=0.4)
        ax.axvline(0, color="k", lw=0.3)
        ax.set_xlabel("Time from trial start (s)")
    axF.set_ylabel("Trial")
    axF.set_title(f"Area, baseline-subtracted (±{vmax:.0f} px²)", loc="left")
    axG.imshow(S, aspect="auto", extent=ext, cmap=ListedColormap(["#a5d6a7", "#ffffff", "#ffcc80", "#9e9e9e"]),
               vmin=-0.5, vmax=3.5, interpolation="nearest")
    axG.set_title("Frame state (white blink, orange squint, grey failed)", loc="left")
    axG.tick_params(labelleft=False)
    side = np.array([[mpl.colors.to_rgb(OUTCOME_COLORS.get(o, "k")), mpl.colors.to_rgb(LABEL_COLORS[l])]
                     for o, l in zip(tq.outcome, tq.label)])
    axS.imshow(side, aspect="auto", extent=[0, 2, len(tq) - 0.5, -0.5], interpolation="nearest")
    axS.set_xticks([0.5, 1.5])
    axS.set_xticklabels(["out.", "QC"], fontsize=3.8, rotation=90)
    axS.set_yticks([])
    for s in axS.spines.values():
        s.set_visible(False)


def _mean_by_outcome(ax, r, edges, D):
    tq = r["trials_qc"]
    x = (edges[:-1] + edges[1:]) / 2
    usable = (tq.label == "valid").to_numpy()
    for o, col in OUTCOME_COLORS.items():
        m = (tq.outcome == o).to_numpy() & usable
        if m.sum() < 3:
            continue
        mu = np.nanmean(D[m], 0)
        se = np.nanstd(D[m], 0) / np.sqrt(np.isfinite(D[m]).sum(0).clip(1))
        ax.fill_between(x, mu - se, mu + se, color=col, alpha=0.25, lw=0)
        ax.plot(x, mu, color=col if o != "CR" else "0.45", label=f"{o} (n = {m.sum()})")
    ax.axvline(0, color="k", lw=0.3)
    ax.axhline(0, color="0.7", lw=0.3)
    ax.set_xlabel("Time from trial start (s)")
    ax.set_ylabel("Δ area (px²)")
    ax.set_title("Mean response, valid trials (non-valid frames masked)", loc="left")
    ax.legend(loc="upper left", ncol=2, handlelength=0.8, columnspacing=0.6)


def _trial_qc_rows(axs, r):
    tq = r["trials_qc"]
    n = np.arange(len(tq))
    axs[0].scatter(n, tq.valid_frac, s=1.2, c=[LABEL_COLORS[l] for l in tq.label], lw=0)
    axs[0].axhline(PARAMS["trial_min_valid"], color="#c62828", lw=0.4, ls="--")
    axs[0].set_ylabel("Valid frac.")
    axs[0].set_ylim(-0.05, 1.05)
    axs[1].plot(n, tq.baseline_area, color="k", lw=0.4)
    axs[1].set_ylabel("Baseline\narea (px²)")
    axs[2].plot(n, tq.mean_likelihood, color="#7b1fa2", lw=0.4)
    axs[2].set_ylabel("Mean lik.")
    axs[2].set_xlabel("Trial (chronological)")
    vc = tq.label.value_counts()
    axs[0].set_title(f"Per-trial QC ({PARAMS['trial_window_s'][0]:g} to {PARAMS['trial_window_s'][1]:g} s): "
                     + ", ".join(f"{k} {vc.get(k, 0)}" for k in LABEL_COLORS), loc="left")
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    for ax in axs:
        ax.set_xlim(-1, len(tq))


def _loss_durations(ax, r):
    fs = r["meta"]["fs"]
    bs, be = _runs(r["c"]["state"] == STATE["blink"])
    fs_, fe_, kinds = failed_events(r["c"], fs)
    loss = (fe_ - fs_)[np.isin(kinds, ["short loss", "long loss"])] / fs
    bins = np.logspace(np.log10(0.005), np.log10(60), 40)
    ax.hist((be - bs) / fs, bins=bins, color="#42a5f5", label=f"Blinks ({len(bs)})")
    ax.hist(loss, bins=bins, histtype="step", color="#c62828", lw=0.6, label=f"Failed losses ({len(loss)})")
    ax.axvline(PARAMS["blink_max_s"], color="k", lw=0.4, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right", handlelength=0.8)
    m = r["metrics"]
    extra = f", video-confirmed {100 * m['video_blink_confirmed']:.0f}%" if "video_blink_confirmed" in m else ""
    ax.set_title(f"Loss durations{extra}", loc="left")


def _psd(ax, r):
    fs, clean = r["meta"]["fs"], r["clean"]
    starts, stops = _runs(np.isfinite(clean))
    ps = [signal.welch(clean[s:e] - clean[s:e].mean(), fs=fs, nperseg=int(10 * fs))[1]
          for s, e in zip(starts, stops) if e - s >= int(10 * fs)]
    if ps:
        fv = signal.welch(np.zeros(int(10 * fs)), fs=fs, nperseg=int(10 * fs))[0]
        ax.loglog(fv[1:], np.mean(ps, 0)[1:], color="k", lw=0.5, label=f"Valid segments ({len(ps)})")
    ax.axvline(2, color="#42a5f5", lw=0.4, ls="--")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (px⁴/Hz)")
    ax.legend(loc="lower left", handlelength=0.8)
    ax.set_title(f"Area spectrum, SNR = {r['metrics']['snr']:.1f}", loc="left")


def _lik_hist(fig, gs, r):
    sub = gs.subgridspec(2, 2, hspace=0.6, wspace=0.4)
    axs = []
    for i, p in enumerate(POINTS):
        ax = fig.add_subplot(sub[i // 2, i % 2])
        ax.hist(r["c"]["L"][:, i], bins=50, range=(0, 1), color="#7b1fa2", log=True)
        ax.axvline(PARAMS["p_cut"], color="k", lw=0.4, ls="--")
        ax.set_title(p, fontsize=4.5, pad=1)
        ax.tick_params(labelsize=3.8)
        if i == 2:
            ax.set_xlabel("Likelihood", fontsize=4.2)
        axs.append(ax)
    return axs


def _xcorr(ax, r):
    fs = r["meta"]["fs"]
    mot = r["df"].get("whisker_velocity", r["df"].get("jaw_velocity"))
    if mot is None:
        ax.text(0.5, 0.5, "No motion trace", ha="center", transform=ax.transAxes)
        return
    k = max(int(fs / 20), 1)
    x = r["clean"].copy()
    ok = np.isfinite(x)
    x[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), x[ok])
    p, m = _bin(x, k), _bin(mot.to_numpy(), k)
    good = np.isfinite(p) & np.isfinite(m)
    p, m = (p - np.nanmean(p[good])) / np.nanstd(p[good]), (m - np.nanmean(m[good])) / np.nanstd(m[good])
    p[~good], m[~good] = 0, 0
    lags = np.arange(-100, 101)
    cc = np.array([np.mean(p[max(l, 0):len(p) + min(l, 0)] * m[max(-l, 0):len(m) + min(-l, 0)]) for l in lags])
    ax.plot(lags / 20, cc, color="k")
    i = np.argmax(cc)
    ax.axvline(lags[i] / 20, color="#42a5f5", lw=0.4, ls="--")
    ax.axvline(0, color="0.7", lw=0.3)
    ax.set_xlabel("Lag (s; > 0 pupil after motion)")
    ax.set_ylabel("Correlation")
    ax.set_title(f"Pupil × whisker motion, peak {cc[i]:.2f} at {lags[i] / 20:+.2f} s", loc="left")


def _header(ax, r):
    ax.axis("off")
    m, meta = r["metrics"], r["meta"]
    vid = r.get("video") or {}
    ax.text(0, 1, f"{meta['mouse_id']} · {meta['session_id']} · {meta['description']} · {meta['group']} · "
                  f"{m['n_trials']} active trials · {meta['fs']:.0f} Hz", fontsize=7.5, fontweight="bold", va="top")
    rows = [("% valid", f"{m['pct_valid']:.1f}"), ("% blink", f"{m['pct_blink']:.2f}"),
            ("% squint", f"{m['pct_squint']:.2f}"), ("% failed", f"{m['pct_failed']:.2f}"),
            ("of which jumps", f"{m['pct_jump']:.2f}%"),
            ("Long losses", f"{m['long_loss_min']:.1f} min"), ("Blinks / min", f"{m['blink_rate_per_min']:.1f}"),
            ("Median blink", f"{m['blink_median_ms']:.0f} ms"), ("SNR (info)", f"{m['snr']:.1f}"),
            ("% interpolated", f"{m['pct_interpolated']:.2f}"), ("Longest gap", f"{m['longest_gap_s']:.1f} s"),
            ("% dropped frames", f"{m['pct_dropped']:.3f}"),
            ("Trials valid/blink/squint/failed",
             f"{m['n_trials_valid']} / {m['n_trials_blink']} / {m['n_trials_squint']} / {m['n_trials_failed']}"),
            ("% trials valid", f"{m['pct_trials_valid']:.1f}"), ("Open-eye h/w ref.", f"{m['squint_ratio_ref']:.2f}")]
    if "video_blink_confirmed" in m:
        rows += [("Blinks confirmed (video)", f"{100 * m['video_blink_confirmed']:.0f}%"),
                 ("Lost, not blink, lid signal", f"{100 * m['video_failure_lid']:.0f}%")]
    ncol = 7
    for j, (k, v) in enumerate(rows):
        x, y = (j % ncol) / ncol * 0.86, 0.74 - (j // ncol) * 0.3
        ax.text(x, y, k, fontsize=4.4, color="0.35", va="top")
        ax.text(x, y - 0.11, v, fontsize=6.2, va="top")
    ax.text(0.995, 0.55, m["verdict"].upper(), fontsize=11, fontweight="bold", color="w", ha="right", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc=VERDICT_COLORS[m["verdict"]], ec="none"))
    thr = "; ".join(f"{k} ≥ {v[0]:g} pass, ≥ {v[1]:g} warn" for k, v in VERDICT.items())
    ax.text(0, -0.2, f"Verdict: {thr}. Trial label in {PARAMS['trial_window_s']} s around start_time: valid if "
                      f">= {PARAMS['trial_min_valid']:.0%} valid frames, else the dominant non-valid state. Squint: "
                      f"height/width < {PARAMS['squint_ratio']} x open-eye value for >= {PARAMS['squint_min_s']} s. "
                      f"Video: "
                      f"{str(vid.get('video', 'none')).replace('/mnt/lsens-data/', '')}", fontsize=3.8, color="0.4",
            va="top")


def page_qc(r):
    fig = plt.figure(figsize=(210 * MM, 297 * MM))
    outer = fig.add_gridspec(6, 1, height_ratios=[0.95, 2.45, 2.05, 1.8, 1.55, 1.35], hspace=0.42,
                             left=0.07, right=0.95, top=0.975, bottom=0.035)
    _header(fig.add_subplot(outer[0]), r)
    gB = outer[1].subgridspec(5, 1, height_ratios=[1.3, 0.8, 0.8, 0.25, 0.7], hspace=0.12)
    axB = [fig.add_subplot(gB[i]) for i in range(5)]
    _timeline(axB, r, 0, len(r["c"]["state"]), 1.0, title="Full session")
    _letter(axB[0], "A", -0.06, 1.05)
    rowC = outer[2].subgridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.28)
    gC = rowC[0].subgridspec(5, 1, height_ratios=[1.3, 0.8, 0.8, 0.25, 0.7], hspace=0.12)
    a, b = _zoom_window(r)
    axC = [fig.add_subplot(gC[i]) for i in range(5)]
    _timeline(axC, r, a, b, 1 / r["meta"]["fs"],
              title=f"Zoom, {r['df'].t.iloc[a]:.0f}-{r['df'].t.iloc[b - 1]:.0f} s")
    _letter(axC[0], "B", -0.09, 1.05)
    axD = fig.add_subplot(rowC[1])
    _scatter(axD, r)
    _letter(axD, "C", -0.2, 1.03)
    rowE = outer[3].subgridspec(1, 3, width_ratios=[1, 1.3, 1.3], wspace=0.35)
    gE = rowE[0].subgridspec(2, 1, hspace=0.35)
    axE = [fig.add_subplot(gE[i]) for i in range(2)]
    _triggered(axE, r)
    _letter(axE[0], "D", -0.3, 1.2)
    gFG = rowE[1:].subgridspec(1, 3, width_ratios=[1, 0.06, 1], wspace=0.08)
    edges, D, S = _trial_matrix(r)
    axF, axS, axG = fig.add_subplot(gFG[0]), fig.add_subplot(gFG[1]), fig.add_subplot(gFG[2])
    _rasters(axF, axG, axS, r, edges, D, S)
    _letter(axF, "E")
    _letter(axG, "F", -0.05)
    rowH = outer[4].subgridspec(1, 2, width_ratios=[1, 2], wspace=0.25)
    axH = fig.add_subplot(rowH[0])
    _mean_by_outcome(axH, r, edges, D)
    _letter(axH, "G", -0.2)
    gI = rowH[1].subgridspec(3, 1, hspace=0.15)
    axI = [fig.add_subplot(gI[i]) for i in range(3)]
    _trial_qc_rows(axI, r)
    _letter(axI[0], "H", -0.08, 1.2)
    rowJ = outer[5].subgridspec(1, 4, wspace=0.45)
    axJ, axK = fig.add_subplot(rowJ[0]), fig.add_subplot(rowJ[1])
    _loss_durations(axJ, r)
    _psd(axK, r)
    axL = _lik_hist(fig, rowJ[2], r)
    axM = fig.add_subplot(rowJ[3])
    _xcorr(axM, r)
    for ax, l in ((axJ, "I"), (axK, "J"), (axM, "L")):
        _letter(ax, l, -0.3, 1.1)
    _letter(axL[0], "K", -0.6, 1.3)
    return fig


# ----------------------------------------------------------------------------------------------- page 2: examples

def example_events(r, n_blink=6, n_failed=6, seed=0):
    """Example frames: valid area percentiles; blinks spread over the session; failed events (2 longest losses, then
    short losses, partial losses, a jump). Returns [(section, label, [(tag, NWB index)])]."""
    c, fs, t = r["c"], r["meta"]["fs"], r["df"].t.to_numpy()
    st = c["state"]
    rng = np.random.default_rng(seed)
    out = []
    valid = np.flatnonzero(st == STATE["valid"])
    if len(valid):
        a = c["area"][valid]
        frames = [(f"P{q}", int(valid[np.argmin(np.abs(a - np.percentile(a, q)))])) for q in (2, 10, 25, 50, 75, 90, 98)]
        out.append(("Valid", "area percentiles", frames))

    def seq(s, e):
        d = int(0.1 * fs)
        return [("-100 ms", max(s - d, 0)), ("onset", s), ("mid", (s + e) // 2), ("end", max(e - 1, s)),
                ("+100 ms", min(e - 1 + d, len(st) - 1))]

    bs, be = _runs(st == STATE["blink"])
    if len(bs):
        for i in np.unique(np.linspace(0, len(bs) - 1, min(n_blink, len(bs))).round().astype(int)):
            out.append(("Blink", f"t = {t[bs[i]]:.1f} s, {1000 * (be[i] - bs[i]) / fs:.0f} ms", seq(bs[i], be[i])))
    ss, se = _runs(st == STATE["squint"])
    if len(ss):
        order = np.argsort(se - ss)[::-1][:2]
        rest = np.setdiff1d(np.arange(len(ss)), order)
        extra = rng.choice(rest, min(len(rest), 2), replace=False) if len(rest) else []
        for i in list(order) + list(extra):
            out.append(("Squint", f"t = {t[ss[i]]:.1f} s, {1000 * (se[i] - ss[i]) / fs:.0f} ms, "
                                  f"h/w {np.nanmedian(c['ratio_norm'][ss[i]:se[i]]):.2f}", seq(ss[i], se[i])))
    fs_, fe_, kinds = failed_events(c, fs)
    chosen = []
    li = np.flatnonzero(kinds == "long loss")
    chosen += list(li[np.argsort(fe_[li] - fs_[li])[::-1][:2]])
    for kind, n in (("short loss", 2), ("partial loss", 1), ("jump", 1)):
        idx = np.flatnonzero(kinds == kind)
        if kind != "jump":
            idx = idx[(fe_[idx] - fs_[idx]) >= 2]
        if len(idx):
            chosen += list(rng.choice(idx, min(len(idx), n), replace=False))
    for i in chosen[:n_failed]:
        out.append(("Failed", f"{kinds[i]}, t = {t[fs_[i]]:.1f} s, {1000 * (fe_[i] - fs_[i]) / fs:.0f} ms",
                    seq(fs_[i], fe_[i])))
    return out


def page_examples(r, images):
    c, df = r["c"], r["df"]
    box = video.pupil_crop_box(df, c["state"] == STATE["valid"])
    events = example_events(r)
    valid = [ev for ev in events if ev[0] == "Valid"]
    rows = [ev for ev in events if ev[0] != "Valid"]
    n_rows = 1 + (len(rows) + 1) // 2
    fig = plt.figure(figsize=(210 * MM, 297 * MM))
    fig.text(0.04, 0.988, f"{r['meta']['session_id']} · {r['meta']['description']} · example frames (keypoints coloured "
                          f"by likelihood; pink outline = 4-point polygon when all tracked; frame border = state)",
             fontsize=7, fontweight="bold", va="top")
    gs = fig.add_gridspec(n_rows, 1, left=0.03, right=0.98, top=0.95, bottom=0.02, hspace=0.5)
    off = r["video"]["offset"] if r.get("video") else 0

    def draw(ax, tag, i):
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(STATE_COLORS[c["state"][i]]), s.set_linewidth(1.2)
        frame = images.get(int(i) + off) if images else None
        if frame is None:
            ax.text(0.5, 0.5, "no frame", ha="center", va="center", transform=ax.transAxes, fontsize=4)
            return
        X0, Y0, _ = box
        img = video.crop(frame, box)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        xs = np.array([df[f"{p}_x"].iloc[i] for p in POINTS]) - X0
        ys = np.array([df[f"{p}_y"].iloc[i] for p in POINTS]) - Y0
        inside = (xs >= 0) & (xs < img.shape[1]) & (ys >= 0) & (ys < img.shape[0])
        ax.scatter(xs[inside], ys[inside], c=c["L"][i][inside], cmap="viridis", vmin=0, vmax=1, s=5, edgecolors="w",
                   linewidths=0.2)
        if c["n_good"][i] == 4:
            order = [0, 2, 3, 1, 0]  # top, right, bottom, left
            ax.plot(xs[order], ys[order], color="#ff4081", lw=0.4)
        ax.set_xlim(-0.5, img.shape[1] - 0.5)
        ax.set_ylim(img.shape[0] - 0.5, -0.5)
        z = r.get("intensity_z")
        zt = f", z {z[i]:.0f}" if z is not None and np.isfinite(z[i]) else ""
        rn = c["ratio_norm"][i]
        rt = f", h/w {rn:.2f}" if np.isfinite(rn) else ""
        ax.set_title(f"{tag} · {STATE_NAMES[c['state'][i]]}{rt}{zt}", fontsize=3.6, pad=1.5)

    if valid:
        _, label, frames = valid[0]
        sub = gs[0].subgridspec(1, len(frames), wspace=0.05)
        for j, (tag, i) in enumerate(frames):
            ax = fig.add_subplot(sub[0, j])
            draw(ax, tag, i)
            if j == 0:
                ax.text(0, 1.25, f"Valid frames ({label})", transform=ax.transAxes, fontsize=5.5, fontweight="bold")
    for k in range(0, len(rows), 2):
        sub = gs[1 + k // 2].subgridspec(1, 11, width_ratios=[1] * 5 + [0.25] + [1] * 5, wspace=0.05)
        for h, (sec, label, frames) in enumerate(rows[k:k + 2]):
            for j, (tag, i) in enumerate(frames):
                ax = fig.add_subplot(sub[0, j + 6 * h])
                draw(ax, tag, i)
                if j == 0:
                    ax.text(0, 1.22, f"{sec}: {label}", transform=ax.transAxes, fontsize=5, fontweight="bold",
                            color=STATE_COLORS[STATE[sec.lower()]])
    return fig


def example_frame_indices(r):
    return sorted({i for _, _, frames in example_events(r) for _, i in frames})


def _writable(path):
    """A file open in a viewer on Windows cannot be overwritten on the NAS: use a timestamped name instead."""
    import datetime
    from pathlib import Path
    path = Path(path)
    try:
        if path.exists():
            with open(path, "ab"):
                pass
        return path
    except OSError:
        alt = path.with_name(f"{path.stem}_{datetime.datetime.now():%H%M%S}{path.suffix}")
        print(f"warning: {path.name} is locked; writing {alt.name}")
        return alt


def make_pages(r, images, path):
    _rc()
    figs = [page_qc(r), page_examples(r, images)]
    with PdfPages(_writable(str(path) + ".pdf")) as pdf:
        for f in figs:
            pdf.savefig(f, dpi=200)
    figs[0].savefig(_writable(str(path) + ".png"), dpi=150)
    figs[1].savefig(_writable(str(path) + "_examples.png"), dpi=150)
    for f in figs:
        plt.close(f)
