"""000 -- Physical units and rates of the DLC series loaded by dlc_helpers.load_dlc (see the project question.md).

Per session and camera: mm per pixel (data / raw pixel values), frame rate, and the 1st / 50th / 99th percentiles of
the main series in physical units (likelihood-masked). Rate check: stored *_velocity vs the frame-to-frame difference
of its base series x nominal rate (slope 1 = stored per second and the loader adds no second rate factor), and vs
the derivative from the true timestamps (dlc_helpers.derivative, forward difference); velocity vs speed: whisker
speed = |whisker_velocity|, jaw keypoint 2-D speed vs |jaw_velocity|.
Outputs: figures/000_units.png, 000_rates.png; tables/000_units.csv, 000_rates.csv
Run (haas, ibl-ai-agent .venv): python 000_units_and_rates.py
"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import pathlib  # noqa: E402
import sys  # noqa: E402

_EA = pathlib.Path(__file__).resolve().parents[1]  # exploratory-analyses/: shared common.py
if str(_EA) not in sys.path:
    sys.path.append(str(_EA))  # appended: the script folder and sibling folders below take precedence
from common import OUT, SESSIONS, dh, nwb  # noqa: E402

SERIES = {'side': ['jaw_distance', 'jaw_angle', 'tongue_distance', 'nose_distance', 'pupil_area', 'jaw_velocity',
                   'pupil_area_velocity'],
          'top': ['whisker_angle', 'whisker_velocity', 'top_nose_distance', 'top_nose_velocity']}
PAIRS = {'side': [('jaw_distance', 'jaw_velocity'), ('nose_distance', 'nose_velocity'),
                  ('tongue_distance', 'tongue_velocity'), ('pupil_area', 'pupil_area_velocity')],
         'top': [('whisker_angle', 'whisker_velocity'), ('top_nose_distance', 'top_nose_velocity')]}


def slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.sum(x[ok] * y[ok]) / np.sum(x[ok] ** 2)) if ok.sum() > 100 else np.nan


def main():
    unit_rows, rate_rows = [], []
    for s in SESSIONS:
        phys = dh.load_dlc(nwb(s), fix_reference=False)
        pix = dh.load_dlc(nwb(s), physical_units=False, mask_likelihood=False, fix_reference=False)
        unmasked = dh.load_dlc(nwb(s), mask_likelihood=False, fix_reference=False)
        for view, v in phys.items():
            mm_per_px = float(np.nanmedian(unmasked[view].data[SERIES[view][0]] / pix[view].data[SERIES[view][0]]))
            for name in SERIES[view]:
                if name not in v.data:
                    continue
                x = v.data[name].to_numpy()
                p1, p50, p99 = np.nanpercentile(x, [1, 50, 99])
                unit_rows.append(dict(session_id=s, view=view, series=name, unit=v.units[name], p1=p1, p50=p50, p99=p99,
                                      mm_per_px=mm_per_px, rate=v.rate, frac_masked=v.masked.get(name, 0)))
            if view == 'side' and 'pupil_area' in v.data:
                d = np.sqrt(4 * v.data['pupil_area'].to_numpy() / np.pi)
                unit_rows.append(dict(session_id=s, view=view, series='pupil_diameter_equiv', unit='mm',
                                      p1=np.nanpercentile(d, 1), p50=np.nanmedian(d), p99=np.nanpercentile(d, 99),
                                      mm_per_px=mm_per_px, rate=v.rate))
            u = unmasked[view]
            ts = u.timestamps
            nominal = np.round(1 / np.median(np.diff(ts[0:200])), 2)
            for base, vel in PAIRS[view]:
                if base not in u.data or vel not in u.data:
                    continue
                b, sv = u.data[base].to_numpy(), u.data[vel].to_numpy()
                d_nom = np.r_[np.nan, np.diff(b) * nominal]
                d_true = np.r_[np.nan, np.diff(b) / np.diff(ts)]
                rate_rows.append(dict(session_id=s, view=view, base=base, velocity=vel, unit=u.units[vel],
                                      slope_vs_diff_x_nominal_rate=slope(d_nom, sv),
                                      slope_vs_diff_over_true_dt=slope(d_true, sv),
                                      max_abs_dev_nominal=float(np.nanmax(np.abs(sv[1:] - d_nom[1:]))),
                                      nominal_rate=nominal, true_rate=1 / np.median(np.diff(ts))))
            if view == 'side' and {'jaw_x', 'jaw_y', 'jaw_velocity'} <= set(u.data):
                sp2d = dh.keypoint_speed(u.data['jaw_x'], u.data['jaw_y'], ts)
                jv = np.abs(u.data['jaw_velocity'].to_numpy())
                unit_rows.append(dict(session_id=s, view=view, series='jaw_keypoint_speed_2d', unit='mm/s',
                                      p1=np.nanpercentile(sp2d, 1), p50=np.nanmedian(sp2d), p99=np.nanpercentile(sp2d, 99),
                                      mm_per_px=mm_per_px, rate=u.rate))
                unit_rows.append(dict(session_id=s, view=view, series='jaw_speed_abs_velocity', unit='mm/s',
                                      p1=np.nanpercentile(jv, 1), p50=np.nanmedian(jv), p99=np.nanpercentile(jv, 99),
                                      mm_per_px=mm_per_px, rate=u.rate))
        print(s, 'done', flush=True)
    units = pd.DataFrame(unit_rows)
    rates = pd.DataFrame(rate_rows)
    units.to_csv(OUT / 'tables' / '000_units.csv', index=False)
    rates.to_csv(OUT / 'tables' / '000_rates.csv', index=False)
    print(units.groupby(['view', 'series', 'unit'])[['p1', 'p50', 'p99']].median().round(3).to_string())
    print(rates.groupby(['view', 'velocity'])[['slope_vs_diff_x_nominal_rate', 'slope_vs_diff_over_true_dt',
                                               'max_abs_dev_nominal']].agg(['min', 'max']).round(4).to_string())
    print(units.drop_duplicates(['session_id', 'view'])[['session_id', 'view', 'mm_per_px', 'rate']].to_string(index=False))

    names = list(units.series.unique())
    nc = 4
    nr = int(np.ceil(len(names) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(4.2 * nc, 2.8 * nr), squeeze=False)
    for ax, name in zip(axs.flat, names):
        d = units[units.series == name].reset_index(drop=True)
        y = np.arange(len(d))
        ax.hlines(y, d.p1, d.p99, color='0.6', lw=2)
        ax.plot(d.p50, y, 'o', color='k', ms=4)
        ax.set_yticks(y)
        ax.set_yticklabels(d.session_id.str[:14], fontsize=6)
        ax.set_title(f'{name} ({d.unit.iloc[0]})', fontsize=8)
        ax.tick_params(axis='x', labelsize=7)
    for ax in axs.flat[len(names):]:
        ax.axis('off')
    fig.suptitle('DLC series in physical units (dlc_helpers.load_dlc, likelihood-masked): 1st-99th percentile, median',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '000_units.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, col in enumerate(['slope_vs_diff_x_nominal_rate', 'slope_vs_diff_over_true_dt']):
        for j, (vel, d) in enumerate(rates.groupby('velocity')):
            ax[k].plot(d[col], np.full(len(d), j) + np.random.uniform(-0.15, 0.15, len(d)), 'o', ms=4)
        ax[k].set_yticks(range(rates.velocity.nunique()))
        ax[k].set_yticklabels(sorted(rates.velocity.unique()), fontsize=7)
        ax[k].axvline(1, color='k', lw=0.6, ls='--')
        ax[k].set_xlabel(col.replace('_', ' '), fontsize=8)
    ax[0].set_title('stored velocity / (diff of base series x nominal rate)', fontsize=8)
    ax[1].set_title('stored velocity / (diff of base series / true frame interval)', fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'figures' / '000_rates.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    main()
