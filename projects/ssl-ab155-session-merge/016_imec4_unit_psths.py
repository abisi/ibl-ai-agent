"""016 -- Activity of the imec4 good units that collapse in passive post, per context, with stable imec4 good units
for comparison.
Units: quality_label good on imec4 (005 labels, joined run A); collapsing = passive post rate < 20% of passive pre
(014_units_lost_post.csv); stable = post within x2 of pre (up to 8, highest rate).
Per unit (one row):
  firing rate over the session (30-s bins, joined clock; junction and context epochs marked)
  PSTH to whisker stim and to auditory stim per context: passive pre, active, passive post (20-ms bins, no smoothing,
    mean over trials)
  whisker-trial raster, trials in session order, coloured by context
Source: full joined-clock test NWB (010). Times in part 2 are joined-clock (no gap).
Outputs: concat_figure/016_imec4_collapsing_units.png/pdf, 016_imec4_stable_units.png/pdf
Run (local, conda env nwb): python 016_imec4_unit_psths.py
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

J = r'M:\analysis\Axel_Bisi\Misc\SessionToConcat\AB155_20250208_175103_joined'
NWB = rf'{J}\nwb_test\AB155_20250208_175103_joined_full_test.nwb'
TAB = rf'{J}\concat_figure\014_units_lost_post.csv'
OUT = rf'{J}\concat_figure\016_imec4'
TJ = 2997.087361
CTX = {'passive pre': '#4C72B0', 'active': '0.15', 'passive post': '#DD4444'}
EDGES = np.arange(-0.2, 0.5 + 1e-9, 0.02)
plt.rcParams.update({'font.size': 6, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42,
                     'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5})


def load():
    f = h5py.File(NWB, 'r')
    tr = f['intervals/trials']
    st = tr['start_time'][()]
    ctx = np.array([c.decode() for c in tr['context'][()]])
    ctx = np.where(ctx == 'passive', np.where(st < TJ, 'passive pre', 'passive post'), 'active')
    trials = pd.DataFrame({'start': st, 'ctx': ctx, 'wh': tr['whisker_stim'][()] == 1,
                           'aud': tr['auditory_stim'][()] == 1, 'wt': tr['whisker_stim_time'][()],
                           'at': tr['auditory_stim_time'][()]})
    eg = [f[r].name.split('/')[-1].split('_')[0] for r in f['units/electrode_group'][()]]
    cid = [int(x.decode() if isinstance(x, bytes) else x) for x in f['units/cluster_id'][()]]
    idx = f['units/spike_times_index'][()]
    starts = np.r_[0, idx[:-1]]
    spikes = {(p, c): f['units/spike_times'][a:b] for p, c, a, b in zip(eg, cid, starts, idx) if p == 'imec4'}
    end = float(f['units/spike_times'][()].max())
    return trials, spikes, end


def psth(s, ev):
    if len(ev) == 0:
        return None
    h = np.zeros(len(EDGES) - 1)
    for e in ev:
        a, b = np.searchsorted(s, [e + EDGES[0], e + EDGES[-1]])
        h += np.histogram(s[a:b] - e, EDGES)[0]
    return h / len(ev) / np.diff(EDGES)


def figure(units, trials, spikes, end, name, title):
    n = len(units)
    fig, axs = plt.subplots(n, 4, figsize=(7.2, 1.0 * n + 0.6),
                            gridspec_kw={'width_ratios': [2.2, 1, 1, 1.2]}, squeeze=False)
    c = (EDGES[:-1] + EDGES[1:]) / 2
    for i, (_, u) in enumerate(units.iterrows()):
        s = spikes[('imec4', int(u.cluster_id))]
        ax = axs[i, 0]
        e30 = np.arange(0, end + 30, 30)
        ax.plot(e30[:-1] + 15, np.histogram(s, e30)[0] / 30, color='0.2', lw=0.6)
        for k, col in CTX.items():
            t = trials.start[trials.ctx == k]
            if k == 'active':
                t = t[t < TJ]
            ax.axvspan(t.min(), t.max(), color=col, alpha=0.08, lw=0)
        ax.axvline(TJ, color='k', ls='--', lw=0.5)
        ax.set_ylabel(f'cl {int(u.cluster_id)}\n{int(u.depth)} um\nHz', fontsize=6)
        ax.set_xlim(0, end)
        for j, (sel, tcol, lab) in enumerate((('wh', 'wt', 'whisker'), ('aud', 'at', 'auditory'))):
            ax = axs[i, 1 + j]
            for k, col in CTX.items():
                ev = trials.loc[trials[sel] & (trials.ctx == k), tcol].dropna().to_numpy()
                h = psth(s, ev)
                if h is not None:
                    ax.plot(c * 1e3, h, color=col, lw=0.7, label=f'{k} ({len(ev)})')
            ax.axvline(0, color='k', lw=0.4, ls=':')
            ax.set_xlim(-200, 500)
            if i == 0:
                ax.set_title(f'{lab} stim', fontsize=7)
                ax.legend(fontsize=4.5, frameon=False, loc='upper right')
        ax = axs[i, 3]
        w = trials[trials.wh].sort_values('start').reset_index(drop=True)
        for r, row in w.iterrows():
            a, b = np.searchsorted(s, [row.wt - 0.2, row.wt + 0.5])
            ax.vlines((s[a:b] - row.wt) * 1e3, r, r + 0.9, color=CTX[row.ctx], lw=0.3)
        ax.axvline(0, color='k', lw=0.4, ls=':')
        ax.set_xlim(-200, 500)
        ax.set_ylim(len(w), 0)
        ax.set_ylabel('whisker trial', fontsize=5)
        if i == 0:
            ax.set_title('whisker raster', fontsize=7)
        if i == n - 1:
            axs[i, 0].set_xlabel('time in session (s, joined clock)')
            for k in (1, 2, 3):
                axs[i, k].set_xlabel('time from stim (ms)')
    axs[0, 0].set_title('firing rate (30-s bins); dashed = junction', fontsize=7)
    fig.suptitle(title, fontsize=8)
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}_{name}.{ext}', dpi=300)
    plt.close(fig)
    print(f'saved {OUT}_{name}.png ({n} units)')


def main():
    trials, spikes, end = load()
    t = pd.read_csv(TAB)
    g = t[(t.probe == 'imec4') & (t.quality_label == 'good')].copy()
    g['post_pre'] = (g.rate_post + 0.05) / (g.rate_pre + 0.05)
    coll = g[g.post_pre < 0.2].sort_values('depth')
    stab = g[(g.post_pre >= 0.5) & (g.post_pre <= 2) & (g.rate_pre >= 1)].nlargest(8, 'rate_pre').sort_values('depth')
    print(trials.groupby(['ctx']).agg(whisker=('wh', 'sum'), auditory=('aud', 'sum')).to_string())
    figure(coll, trials, spikes, end, 'collapsing_units', f'imec4 good units, passive post < 20% of passive pre')
    figure(stab, trials, spikes, end, 'stable_units', 'imec4 good units, passive post within x2 of passive pre')


if __name__ == '__main__':
    main()
