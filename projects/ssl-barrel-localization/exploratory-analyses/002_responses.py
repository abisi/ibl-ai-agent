"""002 -- Whisker and auditory responses of the linked units (001), from the NWB spikes.

Trials: active whisker / auditory trials (context != passive, perf != 6, early_lick == 0; skills/ssl-trial-exclusion),
aligned at whisker_stim_time / auditory_stim_time; windows end before licking (reaction times > 100 ms). Passive trials
(context == passive) only for the mice kept by ephys_utilities keep_passive_data (passive pre and post; rule), prefix `pas_`.
Per unit and modality (whisker window 5-50 ms, auditory 5-100 ms; baseline the 100 ms before the stimulus):
  evoked - baseline rate (Hz), AUC of evoked vs baseline spike counts (trial by trial, 0.5 = no change), Wilcoxon
  signed-rank p (`responsive` = p < 0.01 and rate increase), latency = first 1-ms bin of the trial-averaged PSTH (no
  smoothing; user 2026-11-11) above baseline mean + 3 SD for 2 consecutive bins, and peak time.
Output: combined_results_ks4/ssl-barrel-localization/tables/002_responses_<mouse>.parquet (keys mouse_id, session_id,
electrode_group, cluster_id).
Run (haas): ~/code/unit_spikes_analysis/.venv/bin/python 002_responses.py AB156 AB164 MH064   [no args: every 001 table]
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from axel_bisi_paths import axel_bisi_root  # noqa: E402

sys.path.insert(0, str(axel_bisi_root() / "Github" / "ephys_utilities"))
from ephys_utilities.helpers import data_utils  # noqa: E402

NWB = axel_bisi_root() / "NWB_ks4"
TABLES = axel_bisi_root() / "combined_results_ks4" / "ssl-barrel-localization" / "tables"
WINDOWS = {"whisker": (0.005, 0.050), "auditory": (0.005, 0.100)}
BASELINE = (-0.100, 0.0)


def _s(x):
    return np.array([v.decode() if isinstance(v, bytes) else v for v in x], dtype=object)


def trial_times(f):
    t = f["intervals/trials"]
    tr = pd.DataFrame({k: (_s(t[k][:]) if t[k].dtype.kind in "SO" else t[k][:]) for k in
                       ["trial_type", "context", "perf", "early_lick", "whisker_stim_time", "auditory_stim_time"]})
    active = (tr.context != "passive") & (tr.perf != 6) & (tr.early_lick == 0)
    passive = tr.context == "passive"
    out = {}
    for mod, col, tt in [("whisker", "whisker_stim_time", "whisker_trial"), ("auditory", "auditory_stim_time", "auditory_trial")]:
        out[mod] = tr[active & (tr.trial_type == tt)][col].dropna().to_numpy()
        out[f"pas_{mod}"] = tr[passive & (tr.trial_type == tt)][col].dropna().to_numpy()
    return out


def response(spikes, onsets, window):
    """Evoked vs baseline statistics of one unit for one set of stimulus onsets."""
    if len(onsets) < 5:
        return {}
    w0, w1 = window
    b0, b1 = BASELINE
    ev = np.array([np.searchsorted(spikes, o + w1) - np.searchsorted(spikes, o + w0) for o in onsets]) / (w1 - w0)
    bl = np.array([np.searchsorted(spikes, o + b1) - np.searchsorted(spikes, o + b0) for o in onsets]) / (b1 - b0)
    diff = ev - bl
    p = stats.wilcoxon(ev, bl).pvalue if np.any(diff != 0) else 1.0
    auc = (stats.mannwhitneyu(ev, bl).statistic / (len(ev) * len(bl)))
    edges = np.arange(b0, w1 + 1e-9, 0.001)
    psth = np.zeros(len(edges) - 1)
    for o in onsets:
        lo, hi = np.searchsorted(spikes, [o + b0, o + w1])
        psth += np.histogram(spikes[lo:hi] - o, edges)[0]
    psth = psth / len(onsets) / 0.001  # Hz, 1-ms bins, unsmoothed
    t = edges[:-1]
    base, post = psth[t < 0], psth[t >= w0]
    above = post > base.mean() + 3 * base.std()
    run = np.flatnonzero(above[:-1] & above[1:])
    return dict(n_trials=len(onsets), rate_base_hz=bl.mean(), rate_evoked_hz=ev.mean(), delta_hz=diff.mean(), auc=auc,
                p=p, responsive=bool(p < 0.01 and diff.mean() > 0),
                latency_ms=float(t[t >= w0][run[0]] * 1000) if len(run) else np.nan,
                peak_ms=float(t[t >= w0][np.argmax(post)] * 1000))


def session_responses(session_id, units, passive_ok):
    with h5py.File(NWB / f"{session_id}.nwb", "r") as f:
        on = trial_times(f)
        u = f["units"]
        grp = [f[r].name.split("/")[-1] for r in u["electrode_group"][:]]
        cid = u["cluster_id"][:].astype(str)
        row_of = {(g, c): i for i, (g, c) in enumerate(zip(grp, cid))}
        st, idx = u["spike_times"], u["spike_times_index"][:]
        rows = []
        for r in units.itertuples():
            i = row_of.get((r.electrode_group, str(int(float(r.cluster_id)))), row_of.get((r.electrode_group, str(r.cluster_id))))
            if i is None:
                continue
            spikes = np.sort(st[(idx[i - 1] if i else 0):idx[i]])
            out = dict(mouse_id=r.mouse_id, session_id=session_id, electrode_group=r.electrode_group, cluster_id=r.cluster_id)
            for mod, win in WINDOWS.items():
                out.update({f"{mod}_{k}": v for k, v in response(spikes, on[mod], win).items()})
                if passive_ok:
                    out.update({f"pas_{mod}_{k}": v for k, v in response(spikes, on[f"pas_{mod}"], win).items()})
            rows.append(out)
    return pd.DataFrame(rows)


def main(mice):
    mice = mice or sorted(p.stem.split("_")[-1] for p in TABLES.glob("001_units_*.parquet"))
    for m in mice:
        if (TABLES / f"002_responses_{m}.parquet").exists() and "--redo" not in sys.argv:
            continue
        units = pd.read_parquet(TABLES / f"001_units_{m}.parquet")
        passive_ok = len(data_utils.keep_passive_data(units[["mouse_id", "session_id", "electrode_group"]])) == len(units)
        df = pd.concat([session_responses(s, g, passive_ok) for s, g in units.groupby("session_id")], ignore_index=True)
        df.to_parquet(TABLES / f"002_responses_{m}.parquet")
        print(m, len(df), "units; whisker responsive", int(df.whisker_responsive.fillna(False).sum()),
              "auditory responsive", int(df.auditory_responsive.fillna(False).sum()), "passive", passive_ok, flush=True)


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")])
