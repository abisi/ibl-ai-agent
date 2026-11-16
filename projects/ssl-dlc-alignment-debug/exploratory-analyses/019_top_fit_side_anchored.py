"""019 -- Top-camera clause with the step times anchored on the side steps (user 2026-11-16, MH017_20250302: the free 017
fit merged two small side steps; on MH013 the top and side steps coincide in time and size).

For each side step (010 re-fit, side frame -> session time on the corrected side timestamps), the top frame at that time
under the running top offset; segment offsets pooled over each whole segment (search +-HALF frames around the previous
offset + the side step size); each step then re-placed within +-PLACE frames by the 008 two-segment score. Same
reference signal as 017 (side nose speed sampled at every raw top pulse). Checks as 017: final D vs top pulses - frames,
top vs side lag in 10 s windows.
Outputs: drift_review/tables/top_fit_<session>.csv (overwrites the 017 candidate), figures/019_<session>.png
Run (haas, ibl-ai-agent .venv): python 019_top_fit_side_anchored.py SESSION_ID
"""
import importlib.util
import pathlib
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('m017', HERE / '017_top_fit_from_side.py')
m017 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m017)
m008, m006, OUT, FR = m017.m008, m017.m006, m017.OUT, m017.FR
HALF, PLACE = 30, 3000


def main(sid):
    r = m017.fit(sid)  # signals, pulses and the free fit (for comparison)
    x, S, pulses, n = r['x'], r['S'], r['pulses'], r['n']
    sg = pd.read_csv(OUT / 'tables' / f'refit_{sid}_side.csv')
    with h5py.File(m006.NWB_DIR / f'{sid}.nwb', 'r') as f:
        t_side = f['processing/behavior/BehavioralTimeSeries/jaw_angle/timestamps'][:]
    D = [m008.pooled_lag(x, S, 0, int(sg.frame_start.iloc[1]) - 2000, int(sg.D.iloc[0]), half=HALF)]
    bounds = [0]
    for k in range(1, len(sg)):
        t_step = t_side[int(sg.frame_start.iloc[k])]
        fb = int(np.searchsorted(pulses, t_step)) - D[-1]  # top frame at the side step time
        nxt = int(sg.frame_start.iloc[k + 1]) - 2000 if k + 1 < len(sg) else n
        guess = D[-1] + int(sg.D.iloc[k] - sg.D.iloc[k - 1])
        Dk = m008.pooled_lag(x, S, fb + 2000, max(nxt, fb + 4000), guess, half=HALF)
        b, _, _ = m008.place_step(x, S, D[-1], Dk, max(fb - PLACE, bounds[-1] + 1), min(fb + PLACE, n - 1))
        bounds.append(b)
        D.append(Dk)
    bounds.append(n)
    segs = pd.DataFrame(dict(frame_start=bounds[:-1], frame_end=bounds[1:], D=np.array(D, int)))
    segs = segs[segs.D.diff().fillna(1) != 0].reset_index(drop=True)  # merge equal neighbours
    segs['frame_end'] = list(segs.frame_start.iloc[1:]) + [n]
    D_new = np.repeat(segs.D.to_numpy(), (segs.frame_end - segs.frame_start).to_numpy())
    fig, ax = plt.subplots(figsize=(12, 4))
    out = {}
    for Dv, lab, col in ((r['D_cur'], 'current NWB top', '#d62728'), (r['D_new'], 'free fit (017)', '0.6'),
                         (D_new, 'side-anchored (019)', '#1f77b4')):
        tt, lag = m017.lag_after(r, Dv)
        out[lab] = np.mean(np.abs(lag) <= 2)
        ax.plot(tt, lag, '.', ms=3, color=col, label=f'{lab}: |lag| <= 2 in {out[lab] * 100:.0f}% of windows')
    ax.set_ylim(-30, 30)
    ax.axhline(0, color='k', lw=0.4)
    ax.set_xlabel('session time (s)')
    ax.set_ylabel('top vs side lag (frames)')
    ax.set_title(f'{sid} top: side-anchored steps; final D {segs.D.iloc[-1]} vs pulses - frames {r["surplus"]}', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / f'019_{sid}.png', dpi=75)
    segs.to_csv(OUT / 'tables' / f'top_fit_{sid}.csv', index=False)
    print(sid, segs[['frame_start', 'D']].values.tolist(), 'surplus', r['surplus'],
          {k: round(v, 3) for k, v in out.items()})


if __name__ == '__main__':
    main(sys.argv[1])
