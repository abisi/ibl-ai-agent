"""011 -- Prototype: top-camera frame -> pulse offset estimated per whisker trial from the stimulus-locked whisker
response (010 showed lick-based top fits are unreliable, while the whisker-speed response to the whisker stimulus is
sharp and constant in in-register sessions).

Per session: whisker speed (|d whisker_angle|, z) in DLC frame order, raw top pulses and whisker-stimulus times (010a,
NWB). For a whisker trial with stimulus at pulse p and an offset D (pulse index - frame), the stimulus falls on frame
p - D; the trial's offset estimate is the D in [-60, pulses - frames + 60] maximising the correlation of the frames
around p - D with a response template (-0.1..+0.4 s). The template starts as the mean response under the current NWB
pairing and is re-estimated twice from the trials re-aligned at their own best D. Confidence per trial = (peak -
median) / MAD over D. Figure: per-trial D over video time with the current NWB pairing, the 010 side re-fit and the
total pulse surplus; and the stimulus-aligned heatmap re-aligned at the per-trial D.
Outputs: combined_results_ks4/ssl-dlc-alignment-debug/drift_review/top_stimulus_fit.png, tables/top_stimulus_fit.csv
Run (haas, ibl-ai-agent .venv, after 010a): python 011_top_stimulus_fit.py
"""
import importlib.util
import pathlib

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m010', HERE / '010_drift_review_pdf.py')
m010 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m010)
m006 = m010.m006
OUT = m010.OUT
SESSIONS = ['MH017_20250302_120918', 'MH013_20250219_170437', 'AB092_20231205_140109',
            'AB095_20231212_141728', 'AB085_20231005_152636', 'AB116_20240726_094206']
A, B = -20, 80  # template frames around the stimulus (-0.1 .. +0.4 s at 200 Hz)


def trial_lags(x, pulses, wstim, template, d_lo, d_hi):
    t = (template - template.mean()) / (template.std() or 1)
    rows = []
    for s in wstim:
        p = int(np.searchsorted(pulses, s))
        f0, f1 = p - d_hi + A, p - d_lo + B  # frames covering every candidate stimulus frame p - D
        if f0 < 0 or f1 > len(x):
            rows.append((np.nan, np.nan))
            continue
        c = np.correlate(x[f0:f1], t, 'valid')  # c[k]: stimulus on frame f0 - A + k -> D = p - (f0 - A + k)
        k = int(np.argmax(c))
        mad = np.median(np.abs(c - np.median(c))) or 1
        rows.append((p - (f0 - A + k), (c[k] - np.median(c)) / mad))
    return np.array(rows)


def session(s, ax_d, ax_h):
    raw = np.load(m010.CACHE / f'raw_pulses_{s}.npz')
    pulses = raw['top']
    d = m010.nwb_data(s)
    x = np.nan_to_num(m006.zs(d['top']))
    n = len(x)
    wstim = d['wstim']
    extra = len(pulses) - n
    g_cur, _ = m010.current_index(d['top_ts'], pulses, n)
    # initial template: mean response under the current NWB pairing
    t_cur = m010.times(g_cur, pulses)
    m = m006.triggered(x, t_cur, wstim, np.arange(A, B) / 200)
    template = np.nanmean(m, 0)
    d_lo, d_hi = -60, max(extra, 0) + 60
    for _ in range(3):
        est = trial_lags(x, pulses, wstim, template, d_lo, d_hi)
        good = est[:, 1] >= 6
        if good.sum() < 5:
            break
        stack = [x[int(np.searchsorted(pulses, s_) - D) + A:int(np.searchsorted(pulses, s_) - D) + B]
                 for s_, (D, z) in zip(wstim, est) if z >= 6]
        template = np.mean([w for w in stack if len(w) == B - A], 0)
    p_idx = np.searchsorted(pulses, wstim)
    frame = p_idx - est[:, 0]
    sc = ax_d.scatter(frame / 200, est[:, 0], c=np.clip(est[:, 1], 0, 20), cmap='viridis', s=10, zorder=3)
    f = np.arange(n)
    ax_d.plot(f[::200] / 200, np.where(g_cur >= 0, g_cur - f, np.nan)[::200], color=m010.COL['current'], lw=1.5,
              label='current NWB')
    side_csv = OUT / 'tables' / f'refit_{s}_side.csv'
    if side_csv.exists():
        sg = pd.read_csv(side_csv)
        ax_d.step(np.r_[sg.frame_start, sg.frame_end.iloc[-1]] / 200, np.r_[sg.D, sg.D.iloc[-1]], where='post',
                  color='0.5', lw=1, ls='--', label='side re-fit (010)')
    ax_d.axhline(extra, color='r', lw=0.6, ls=':', label=f'pulses - frames ({extra:+,})')
    ax_d.set_title(f'{s} top: per-whisker-trial offset (colour = confidence z)', fontsize=8)
    ax_d.set_xlabel('video time (s)', fontsize=7)
    ax_d.set_ylabel('offset D (frames)', fontsize=7)
    ax_d.legend(fontsize=6)
    ax_d.tick_params(labelsize=6)
    plt.colorbar(sc, ax=ax_d, fraction=0.03)
    # heatmap re-aligned at each trial's own estimate
    mm = np.full((len(wstim), 300), np.nan)
    for i, (s_, (D, z)) in enumerate(zip(wstim, est)):
        if np.isfinite(D):
            c = int(np.searchsorted(pulses, s_) - D)
            if c - 100 >= 0 and c + 200 <= n:
                mm[i] = x[c - 100:c + 200]
    lo, hi = np.nanpercentile(mm, [2, 98]) if np.isfinite(mm).any() else (0, 1)
    ax_h.imshow(mm, aspect='auto', cmap='magma', vmin=lo, vmax=hi, extent=[-0.5, 1.0, len(wstim), 0],
                interpolation='nearest')
    ax_h.axvline(0, color='c', lw=0.5)
    ax_h.set_title('whisker speed re-aligned at per-trial D', fontsize=8)
    ax_h.set_xlabel('time from stimulus (s)', fontsize=7)
    ax_h.tick_params(labelsize=6)
    return pd.DataFrame(dict(session_id=s, stim_time=wstim, frame=frame, D=est[:, 0], z=est[:, 1]))


def main():
    fig, axs = plt.subplots(len(SESSIONS), 2, figsize=(16, 3.6 * len(SESSIONS)),
                            gridspec_kw=dict(width_ratios=[2.6, 1]))
    out = []
    for r, s in enumerate(SESSIONS):
        try:
            out.append(session(s, axs[r, 0], axs[r, 1]))
        except Exception as ex:
            print(s, 'ERROR', type(ex).__name__, ex)
    fig.tight_layout()
    fig.savefig(OUT / 'top_stimulus_fit.png', dpi=80)
    pd.concat(out).to_csv(OUT / 'tables' / 'top_stimulus_fit.csv', index=False)
    print(OUT / 'top_stimulus_fit.png')


if __name__ == '__main__':
    main()
