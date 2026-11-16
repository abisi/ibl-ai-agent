"""017 -- Generic unit stability between passive pre and passive post from Kilosort 4's own per-spike outputs (no raw
data needed, so it can run for every session): per unit and epoch, spike count, median template-scaled amplitude
(amplitudes.npy) and median spike depth (spike_positions.npy, y). A neuron can change its rate between the passive
blocks, but not its amplitude or its position on the probe.
Applied here to the joined AB155 session (run A), every probe; groups from 014_units_lost_post.csv (quality_label good):
  collapsing  passive post rate < 20% of passive pre
  stable      passive post within x2 of passive pre
Output: concat_figure/017_epoch_waveform_stability.csv (+ printed summary)
Run (haas): python 017_epoch_waveform_stability.py
"""
import numpy as np
import pandas as pd

J = '/mnt/lsens-analysis/Axel_Bisi/Misc/SessionToConcat/AB155_20250208_175103_joined'
RUN = f'{J}/Ephys/catgt_AB155_g0'
LOST = f'{J}/concat_figure/014_units_lost_post.csv'
OUT = f'{J}/concat_figure/017_epoch_waveform_stability.csv'
PRE = (0.0, 246.214)
OFF = {l.split()[0].rstrip(':'): l.split()[1:] for l in open(f'{RUN}/AB155_g0_sc_offsets.txt')}


def main():
    rows = []
    for p in range(5):
        so = f'{RUN}/AB155_g0_imec{p}/kilosort4/sorter_output'
        t = np.load(f'{so}/spike_times_sec.npy').ravel()
        c = np.load(f'{so}/spike_clusters.npy').ravel()
        a = np.load(f'{so}/amplitudes.npy').ravel()
        y = np.load(f'{so}/spike_positions.npy')[:, 1]
        tj = float(OFF[f'sec_imap{p}'][1])
        ep = np.full(len(t), '', dtype=object)
        ep[(t >= PRE[0]) & (t < PRE[1])] = 'pre'
        ep[t >= tj] = 'post'
        d = pd.DataFrame({'cluster_id': c, 'ep': ep, 'amp': a, 'y': y})
        d = d[d.ep != '']
        g = d.groupby(['cluster_id', 'ep']).agg(n=('amp', 'size'), amp=('amp', 'median'), y=('y', 'median')).unstack('ep')
        g.columns = [f'{a}_{b}' for a, b in g.columns]
        g = g.reset_index()
        g['probe'] = f'imec{p}'
        rows.append(g)
    s = pd.concat(rows, ignore_index=True)
    s['amp_ratio'] = s.amp_post / s.amp_pre
    s['dy_um'] = s.y_post - s.y_pre
    lost = pd.read_csv(LOST)
    s = s.merge(lost[['probe', 'cluster_id', 'depth', 'quality_label', 'rate_pre', 'rate_post']], on=['probe', 'cluster_id'],
                how='inner', validate='one_to_one')
    s['post_pre'] = (s.rate_post + 0.05) / (s.rate_pre + 0.05)
    s['group'] = np.select([s.post_pre < 0.2, (s.post_pre >= 0.5) & (s.post_pre <= 2)], ['collapsing', 'stable'], 'other')
    s.to_csv(OUT, index=False)
    g = s[(s.quality_label == 'good') & (s.n_pre >= 50)]
    pd.set_option('display.width', 250)
    print('good units: |log2 amplitude ratio| and |depth shift| between passive pre and post (median [IQR])')
    for (probe, grp), x in g.groupby(['probe', 'group']):
        la = np.abs(np.log2(x.amp_ratio))
        print(f'  {probe} {grp:10s} n={len(x):3d}  |log2 amp ratio| {la.median():.2f} [{la.quantile(.25):.2f}-{la.quantile(.75):.2f}]'
              f'  |dy| {x.dy_um.abs().median():5.1f} um [{x.dy_um.abs().quantile(.25):.1f}-{x.dy_um.abs().quantile(.75):.1f}]')
    for thr_a, thr_y in ((0.5, 20), (0.585, 30)):
        f = (np.abs(np.log2(g.amp_ratio)) > thr_a) | (g.dy_um.abs() > thr_y) | (g.n_post < 50)
        print(f'\nflag = amplitude ratio outside x{2 ** thr_a:.2f} OR |dy| > {thr_y} um OR < 50 post spikes:')
        print(pd.crosstab([g.probe, g.group], f.rename('flagged')).to_string())
    x = g[(g.probe == 'imec4') & (g.group == 'collapsing')].sort_values('depth')
    print('\nimec4 collapsing good units:')
    print(x[['cluster_id', 'depth', 'rate_pre', 'rate_post', 'n_post', 'amp_pre', 'amp_post', 'amp_ratio', 'y_pre', 'y_post',
             'dy_um']].round(2).to_string(index=False))
    x = g[(g.probe == 'imec4') & (g.group == 'stable')].sort_values('depth')
    print('\nimec4 stable good units:')
    print(x[['cluster_id', 'depth', 'rate_pre', 'rate_post', 'n_post', 'amp_ratio', 'dy_um']].round(2).to_string(index=False))


if __name__ == '__main__':
    main()
