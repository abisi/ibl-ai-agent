"""AP-band multi-unit envelope on the peak channels of stable units (ssl-spectral-coupling, AP-band source).

Per probe (electrode_group imecP):
  raw      Axel_Bisi/data/<mouse>/<session>/Ephys/catgt_*/<run>_imecP/*_tcat_corrected.imecP.ap.bin (whisker-artifact
           corrected CatGT output, int16, nSavedChans channels, last = sync)
  clock    probe seconds = a * t_NWB + b, fitted on whisker stimulus times (NWB whisker trial start_time vs
           whisker_stim_times_to_imecP.txt, same count, residuals < 0.1 ms); sample = probe seconds * imSampRate
  channels peak_channel of the session's stable units on that probe (rate >= 1 Hz not required); each channel takes
           the area group / acronym of the majority of its stable units
  signal   per trial chunk [start - 1.25, start + 1.05] s: common average reference (median over the 384 channels per
           sample), band-pass 300-3000 Hz (Butterworth order 3, zero-phase), rectified, mean per 5-ms bin of EDGES
Envelopes are cached per session (npz on the NAS) for a trial subset: up to 30 trials per condition.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import spec_lib as sl

DATA = [sl.ROOT / "data", sl.ROOT.parent / "Myriam_Hamon" / "data"]
PAD = 0.05
BAND = (300.0, 3000.0)
MAX_PER_COND = 30


def meta(path):
    out = {}
    for line in open(path):
        if "=" in line:
            k, v = line.rstrip("\n").split("=", 1)
            out[k.lstrip("~")] = v
    return out


def probe_dirs(session_id):
    """{probe index: run folder} of catgt imec folders holding a corrected AP binary."""
    mouse = session_id.split("_")[0]
    for root in DATA:
        eph = root / mouse / session_id / "Ephys"
        if eph.exists():
            out = {}
            for d in sorted(eph.glob("catgt_*/*_imec*")):
                m = re.search(r"_imec(\d+)$", d.name)
                if m and list(d.glob("*_tcat_corrected.imec*.ap.bin")):
                    out[int(m.group(1))] = d
            if out:
                return out
    return {}


def clock_fit(probe_dir, p, whisker_starts_nwb):
    s = np.loadtxt(next(probe_dir.glob(f"whisker_stim_times_to_imec{p}.txt")))
    w = np.sort(whisker_starts_nwb)
    s = np.sort(s)
    # offset = mode of all pairwise differences (1-ms bins, within +-60 s), then nearest-neighbour matching around it;
    # handles unequal counts (extra / missing stimuli) and clocks offset by seconds
    dif = (s[None, :] - w[:, None]).ravel()
    dif = dif[np.abs(dif) < 60]
    h, e = np.histogram(dif, bins=np.arange(-60, 60.0005, 0.001))
    off = e[np.argmax(h)] + 0.0005
    # nearest-neighbour matching, refined twice with the linear fit (clock drift can exceed 10 ms over a session)
    c = np.array([1.0, off])
    for _ in range(3):
        pred = c[0] * w + c[1]
        j = np.clip(np.searchsorted(s, pred), 1, len(s) - 1)
        j = np.where(np.abs(s[j - 1] - pred) < np.abs(s[j] - pred), j - 1, j)
        ok = np.abs(s[j] - pred) < 0.01
        if ok.sum() < 10:
            break
        c = np.linalg.lstsq(np.column_stack([w[ok], np.ones(ok.sum())]), s[j[ok]], rcond=None)[0]
    if ok.sum() < 0.9 * min(len(s), len(w)):
        raise RuntimeError(f"imec{p}: only {ok.sum()} of {len(w)} NWB whisker trials matched a stimulus time")
    w, s = w[ok], s[j[ok]]
    A = np.column_stack([w, np.ones(len(w))])
    c = np.linalg.lstsq(A, s, rcond=None)[0]
    resid = float(np.abs(s - A @ c).max())
    if resid > 1e-3:
        raise RuntimeError(f"imec{p}: clock fit residual {resid * 1e3:.2f} ms")
    return c, resid


def channel_table(units, level_cols=("area_group", "area_acronym_custom")):
    """One row per (electrode_group, peak_channel) with the majority label of its stable units."""
    u = units.assign(peak_channel=units.peak_channel.astype(int))
    rows = []
    for (eg, ch), g in u.groupby(["electrode_group", "peak_channel"]):
        r = dict(electrode_group=eg, channel=int(ch), n_units=len(g))
        for c in level_cols:
            vc = g[c].dropna().value_counts()
            r[c] = vc.index[0] if len(vc) else None
        rows.append(r)
    return pd.DataFrame(rows)


def envelope_chunks(bin_path, n_chan, fs, chans, sample_starts, n_samp):
    """[trial, bin, channel] envelope for chunks starting at sample_starts (each n_samp long, PAD included)."""
    mm = np.memmap(bin_path, dtype=np.int16, mode="r")
    mm = mm[: (len(mm) // n_chan) * n_chan].reshape(-1, n_chan)
    sos = butter(3, BAND, btype="bandpass", fs=fs, output="sos")
    n_bins = len(sl.EDGES) - 1
    out = np.full((len(sample_starts), n_bins, len(chans)), np.nan, dtype=np.float32)
    pad = int(round(PAD * fs))
    bin_samp = sl.BIN * fs
    for i, s0 in enumerate(sample_starts):
        if s0 < 0 or s0 + n_samp > mm.shape[0]:
            continue
        x = np.asarray(mm[s0:s0 + n_samp, :384], dtype=np.float32)
        x -= np.median(x, axis=1, keepdims=True)
        y = np.abs(sosfiltfilt(sos, x[:, chans], axis=0))[pad:]
        idx = (np.arange(len(y)) / bin_samp).astype(int)
        ok = idx < n_bins
        cnt = np.bincount(idx[ok], minlength=n_bins)[:n_bins]
        for j in range(len(chans)):
            out[i, :, j] = np.bincount(idx[ok], weights=y[ok, j], minlength=n_bins)[:n_bins] / np.maximum(cnt, 1)
    return out


def select_trials(trials, rng):
    """Positions (not index labels) of up to MAX_PER_COND random trials per condition."""
    keep = []
    cond = trials.condition.to_numpy()
    for c in np.unique(cond):
        idx = np.flatnonzero(cond == c)
        keep.extend(rng.choice(idx, min(len(idx), MAX_PER_COND), replace=False))
    return np.sort(np.array(keep, int))


def build_session(session_id, units, trials, whisker_starts_nwb, rng):
    """Envelope tensor [trial subset, bin, channel], channel table and provenance for one session."""
    dirs = probe_dirs(session_id)
    tsel = select_trials(trials, rng)
    starts = trials.start_time.to_numpy()[tsel]
    chan = channel_table(units)
    blocks, prov, ctabs = [], {}, []
    for eg, ct in chan.groupby("electrode_group"):
        p = int(re.search(r"imec(\d+)", eg).group(1))
        if p not in dirs:
            prov[eg] = "no AP binary"
            continue
        d = dirs[p]
        bin_path = next(d.glob("*_tcat_corrected.imec*.ap.bin"))
        m = meta(bin_path.with_suffix(".meta"))
        fs, n_chan = float(m["imSampRate"]), int(m["nSavedChans"])
        (a, b), resid = clock_fit(d, p, whisker_starts_nwb)
        s0 = np.round(((a * (starts + sl.EDGES[0]) + b) - PAD) * fs).astype(np.int64)
        n_samp = int(round((sl.EDGES[-1] - sl.EDGES[0] + 2 * PAD) * fs))
        E = envelope_chunks(bin_path, n_chan, fs, ct.channel.to_numpy(), s0, n_samp)
        blocks.append(E)
        ctabs.append(ct)
        prov[eg] = dict(file=str(bin_path), fs=fs, clock_a=float(a), clock_b=float(b), clock_resid_ms=resid * 1e3,
                        n_channels=int(len(ct)))
    if not blocks:
        raise RuntimeError("no probe with an AP binary")
    return np.concatenate(blocks, axis=2), pd.concat(ctabs, ignore_index=True), tsel, prov
