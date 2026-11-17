"""AP-band feasibility and validity check on one session (default AB127 day 0).

1. Clock: stimulus artefact in the UNcorrected AP binary at the predicted sample (whisker trials): latency of the
   peak |CAR signal| relative to the predicted stimulus time, per probe (should be within ~1 ms).
2. Envelope vs spikes: per channel, trial-wise correlation of the envelope (quiet window mean) with the spike count of
   the stable units on that channel; and the mean evoked envelope (whisker trials) per area group.
3. Read time per trial.
Output: OUT/ap/check/004_<session>.{txt,png}
"""

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ap_lib as al  # noqa: E402
import spec_lib as sl  # noqa: E402

SID = sys.argv[1] if len(sys.argv) > 1 else "AB127_20240821_103757"


def main():
    from ephys_utilities.helpers import data_utils
    od = sl.OUT / "ap" / "check"
    od.mkdir(parents=True, exist_ok=True)
    lines = []
    units, trials, rule = sl.load_session(SID)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = data_utils.process_single_nwb(str(sl.cl.NWB_DIR / f"{SID}.nwb"), day_to_analyze="learning")["trial_table"]
    wst = full[full.trial_type == "whisker_trial"].start_time.to_numpy()
    dirs = al.probe_dirs(SID)
    lines.append(f"{SID}: probes {sorted(dirs)}; stable units {len(units)} ({rule})")
    # 1. artefact latency in the uncorrected binary
    wtr = trials[trials.trial_type == "whisker_trial"].start_time.to_numpy()[:10]
    for p, d in dirs.items():
        raw = [f for f in d.glob(f"*_tcat.imec{p}.ap.bin")][0]
        m = al.meta(raw.with_suffix(".meta"))
        fs, nc = float(m["imSampRate"]), int(m["nSavedChans"])
        (a, b), resid = al.clock_fit(d, p, wst)
        mm = np.memmap(raw, dtype=np.int16, mode="r").reshape(-1, nc)
        lat = []
        for t in wtr:
            s = int(round((a * t + b) * fs))
            x = np.asarray(mm[s - 300:s + 300, :384], np.float32)
            x -= np.median(x, 0, keepdims=True)
            env = np.abs(x).mean(1)
            lat.append((np.argmax(env) - 300) / fs * 1e3)
        lines.append(f"imec{p}: clock a={a:.8f} b={b * 1e3:.3f} ms resid={resid * 1e3:.3f} ms; artefact peak latency "
                     f"(ms) median {np.median(lat):.2f} [{min(lat):.2f}, {max(lat):.2f}]")
    # 2. envelopes for a trial subset
    rng = np.random.default_rng(0)
    t0 = time.time()
    al.MAX_PER_COND = 8
    E, chans, tsel, prov = al.build_session(SID, units, trials, wst, rng)
    dt = time.time() - t0
    lines.append(f"envelopes: {E.shape} in {dt:.0f} s ({dt / len(tsel) / max(1, len(dirs)):.2f} s per trial per probe)")
    starts = trials.start_time.to_numpy()[tsel]
    X = sl.counts(units.spike_times.tolist(), starts)
    q = sl.window_idx(-1.0, 0.0)
    rs = []
    for j, r in chans.iterrows():
        u = np.flatnonzero((units.electrode_group == r.electrode_group).to_numpy()
                           & (units.peak_channel.astype(int) == r.channel).to_numpy())
        e, s = E[:, q, j].mean(1), X[:, q][:, :, u].sum((1, 2))
        if np.std(e) > 0 and np.std(s) > 0:
            rs.append(np.corrcoef(e, s)[0, 1])
    rs = np.array(rs)
    lines.append(f"trial-wise corr(envelope, spikes on channel), quiet window: median {np.median(rs):.2f}, "
                 f"frac > 0: {np.mean(rs > 0):.2f}, n = {len(rs)}")
    # fine-time correlation (5-ms bins, residuals) per channel
    rf = []
    for j, r in chans.iterrows():
        u = np.flatnonzero((units.electrode_group == r.electrode_group).to_numpy()
                           & (units.peak_channel.astype(int) == r.channel).to_numpy())
        e, s = E[:, q, j].ravel(), X[:, q][:, :, u].sum(2).ravel()
        if np.std(e) > 0 and np.std(s) > 0:
            rf.append(np.corrcoef(e, s)[0, 1])
    lines.append(f"5-ms-bin corr(envelope, spikes on channel), quiet window: median {np.median(rf):.2f}")
    open(od / f"004_{SID}.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = trials.trial_type.to_numpy()[tsel] == "whisker_trial"
    tt = (sl.EDGES[:-1] + sl.EDGES[1:]) / 2
    fig, axs = plt.subplots(1, 2, figsize=(8, 3))
    for g in chans.area_group.dropna().unique():
        js = np.flatnonzero((chans.area_group == g).to_numpy())
        ev = E[w][:, :, js].mean((0, 2))
        axs[0].plot(tt, (ev - ev[:100].mean()) / ev[:100].std(), lw=0.8, label=f"{g} ({len(js)} ch)")
        u = np.flatnonzero((units.area_group == g).to_numpy())
        sp = X[w][:, :, u].mean((0, 2))
        axs[1].plot(tt, (sp - sp[:100].mean()) / (sp[:100].std() + 1e-9), lw=0.8)
    axs[0].set(title="AP envelope, whisker trials (z vs baseline)", xlabel="time from stimulus (s)", xlim=(-0.3, 0.6))
    axs[1].set(title="spikes, stable units (z)", xlabel="time from stimulus (s)", xlim=(-0.3, 0.6))
    axs[0].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(od / f"004_{SID}.png", dpi=180)


if __name__ == "__main__":
    main()
