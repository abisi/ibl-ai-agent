"""TENTATIVE: recovered spikes, rasters and stimulus-triggered PSTHs on the two NP1.0 probes (read-only, 2026-11-23).

Same machinery as 10_validate_recovered (pilot 08, high-frequency gain): units with clean F1 >= 0.6; whisker trials
gain-corrected (x / max(g_hf, G_MIN)), masked before the artefact end and where g < G_MIN; auditory trials uncorrected.
Detection over the whole -60..+80 ms window, real and sign-inverted (noise null). KS4 spikes of the same units for comparison.
Rates per 1-ms bin = events / usable trial-time in that bin (coverage). Null-corrected rate = real - (null(t) - null at
-55..-5 ms): removes only the extra noise crossings after the stimulus (baseline false positives are part of the calibrated
detector, as for KS4). No recall correction.
Recovery time per trial = first t >= 5 ms with g_hf >= 0.8 for 3 consecutive 1-ms bins. MH037 imec3: ~25-50 ms = amplifier
release period (false KS4 bursts there) -> shaded, not masked.
Outputs: <ROOT>/gain_recovery_pilot/<tag>_recovered_{fig.png,fig.pdf,psth.npz,events.csv.gz}
Usage: python 11_recovered_psth.py <0|1>
"""
import importlib.util
import os
import sys

ci = int(sys.argv[1])
sys.argv = [sys.argv[0], str(ci), 'hf']
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p08', os.path.join(HERE, '08_gain_recovery_pilot.py'))
p08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p08)
OUT, G_MIN = p08.OUT, p08.G_MIN
CW, CA, CR, CN = '#f7b519', '#2c2cdb', '#d95f02', '0.55'
RELEASE = {'imec3': (25, 50)}


def rec_time(P, g):
    gg = np.array([np.nanmedian(g[P.bi == k]) if (P.bi == k).any() else np.nan for k in range(len(P.ctr))])
    ok = (gg >= 0.8) & (P.ctr >= 5)
    run = np.convolve(ok.astype(int), np.ones(3, int), 'valid') == 3
    j = np.flatnonzero(run)
    return float(P.ctr[j[0]]) if len(j) else float(P.ctr[-1])


def smooth(p, k=3):
    return np.apply_along_axis(lambda z: np.convolve(z, np.ones(k) / k, 'same'), -1, p)


def main():
    sid, probe = p08.CASES[ci]
    tag = f'{sid}_{probe}'
    r, x, fs, uv, maxint, tw, ta, st, sc, cch = p08.load(sid, probe)
    P = p08.Prep(x, fs, uv, maxint)
    chs = np.load(f'{OUT}/{tag}_hf_psth.npz')['channels']
    U = pd.read_csv(f'{OUT}/{tag}_hf_units.csv')
    U = U[U.f1_clean >= 0.6]
    th = dict(zip(U.cluster_id.astype(int), U.theta))

    def prep(t):
        w, s = P.raw(t)
        e = P.art_end(w, chs)
        return P.run(w, chs, e), e, P.gain_hf(w, chs, e), s
    R = Parallel(n_jobs=24, prefer='threads')(delayed(prep)(t) for t in tw)
    A = Parallel(n_jobs=24, prefer='threads')(delayed(lambda t: (P.run(P.raw(t)[0], chs), P.raw(t)[1]))(t) for t in ta)
    T, L0 = p08.templates(P, [q[0] for q in R] + [q[0] for q in A], st, sc, cch, chs, np.r_[tw, ta])
    units = [c for c in th if c in T]
    nS, nb, nu = len(P.tms), len(P.ctr), len(units)
    idx = np.arange(nS)
    ui = {c: i for i, c in enumerate(units)}
    print(tag, 'units', nu, flush=True)

    def ks4(s):
        i0, i1 = np.searchsorted(st, [s - P.n0, s + P.n1])
        sp, cl = st[i0:i1] - s + P.n0, sc[i0:i1]
        m = np.isin(cl, units) & (sp >= 0) & (sp < nS)
        return sp[m], cl[m]

    def per_trial(k, whisker):
        if whisker:
            h, e, g, s = R[k]
            gg = np.where(np.isnan(g), 1.0, g)
            v = (idx < P.n0 - int(0.5e-3 * fs)) | ((idx >= e) & (gg >= G_MIN))
            hc = (h / np.maximum(gg, G_MIN)[:, None]).astype(np.float32)
        else:
            hc, s = A[k]; gg = np.ones(nS, np.float32); v = None
        ev = []
        for cl in units:
            tmpl, cs = T[cl]
            for inv, sig in ((0, hc), (1, -hc)):
                d, amp = p08.detect(sig, tmpl, cs, th[cl], L0, v, fs)
                for sp, a in zip(d, amp):
                    if sp < nS:
                        ev.append((int(whisker), k, cl, inv, float(P.tms[sp]), float(a), float(gg[sp])))
        sp, cl = ks4(s)
        ks = [(int(whisker), k, int(c), 2, float(P.tms[q]), np.nan, float(gg[q])) for q, c in zip(sp, cl)]
        cov = np.ones(nb) if v is None else np.bincount(P.bi, weights=v.astype(float), minlength=nb) / np.bincount(P.bi, minlength=nb)
        return ev + ks, cov
    resw = Parallel(n_jobs=48, prefer='threads')(delayed(per_trial)(k, True) for k in range(len(tw)))
    resa = Parallel(n_jobs=48, prefer='threads')(delayed(per_trial)(k, False) for k in range(len(ta)))
    E = pd.DataFrame([e for q in resw + resa for e in q[0]],
                     columns=['whisker', 'trial', 'cluster_id', 'kind', 't_ms', 'amp', 'g'])   # kind 0 real, 1 null, 2 KS4
    E.to_csv(f'{OUT}/{tag}_recovered_events.csv.gz', index=False)
    covw = np.array([q[1] for q in resw])                                   # trials x bins
    rt = np.array([rec_time(P, q[2]) for q in R])
    E['bin'] = np.clip(np.searchsorted(P.ed, E.t_ms, side='right') - 1, 0, nb - 1)
    E['u'] = E.cluster_id.map(ui)

    def counts(df, trials=None):
        if trials is not None:
            df = df[df.trial.isin(trials)]
        c = np.zeros((nu, nb))
        np.add.at(c, (df.u.to_numpy(), df.bin.to_numpy()), 1)
        return c
    bm = (P.ctr > -55) & (P.ctr < -5)

    def psths(trials):
        w = E[E.whisker == 1]
        cov = covw[trials].sum(0)
        real = counts(w[w.kind == 0], trials) / np.maximum(cov, 1) / 1e-3
        null = counts(w[w.kind == 1], trials) / np.maximum(cov, 1) / 1e-3
        corr = real - (null - null[:, bm].mean(1, keepdims=True))
        ksw = counts(w[w.kind == 2], trials) / len(trials) / 1e-3
        return dict(real=real, null=null, corr=corr, ksw=ksw, cov=cov / len(trials))
    allw = np.arange(len(tw))
    fast, slow = allw[rt <= np.median(rt)], allw[rt > np.median(rt)]
    Pall, Pf, Ps = psths(allw), psths(fast), psths(slow)
    a_ = E[E.whisker == 0]
    aud = dict(real=counts(a_[a_.kind == 0]) / len(ta) / 1e-3, null=counts(a_[a_.kind == 1]) / len(ta) / 1e-3,
               ks=counts(a_[a_.kind == 2]) / len(ta) / 1e-3)
    np.savez_compressed(f'{OUT}/{tag}_recovered_psth.npz', t_ms=P.ctr, units=np.array(units), recovery_ms=rt,
                        **{f'whisker_all_{k}': v for k, v in Pall.items()}, **{f'whisker_fast_{k}': v for k, v in Pf.items()},
                        **{f'whisker_slow_{k}': v for k, v in Ps.items()}, **{f'auditory_{k}': v for k, v in aud.items()},
                        session_id=sid, probe=probe)
    d5 = (P.ctr > 5) & (P.ctr < 30)
    for nm, p in (('all', Pall), ('fast', Pf), ('slow', Ps)):
        print(nm, 'base %.1f  5-30: KS4 %.1f  recovered %.1f  null %.1f  null-corrected %.1f  coverage %.2f' % (
            p['corr'][:, bm].mean(), p['ksw'][:, d5].mean(), p['real'][:, d5].mean(), p['null'][:, d5].mean(),
            p['corr'][:, d5].mean(), p['cov'][d5].mean()), flush=True)
    print('auditory 5-30: KS4 %.1f detector %.1f' % (aud['ks'][:, d5].mean(), aud['real'][:, d5].mean()), flush=True)

    # ---- example units: largest null-corrected dead-window excess with enough events -------------------------------
    ex_score = (Pall['corr'][:, d5].mean(1) - Pall['corr'][:, bm].mean(1)) * (Pall['real'][:, bm].mean(1) > 2)
    ex = [units[i] for i in np.argsort(-np.abs(ex_score))[:4]]
    order = np.argsort(rt)

    fig = plt.figure(figsize=(23, 21))
    gs = fig.add_gridspec(4, 4, hspace=0.42, wspace=0.28)
    rel = RELEASE.get(probe) if sid.startswith('MH037') else None

    def deco(ax, xl=(-20, 70)):
        ax.axvspan(0, 5, color='0.88', zorder=0)
        if rel:
            ax.axvspan(*rel, color='#fde0dd', zorder=0)
        ax.set_xlim(*xl)

    ax = fig.add_subplot(gs[0, 0])
    G = np.array([q[2] for q in R])[order]
    im = ax.imshow(G, aspect='auto', cmap='magma', vmin=0, vmax=1.2, extent=[P.tms[0], P.tms[-1], len(tw), 0], interpolation='none')
    ax.plot(rt[order], np.arange(len(tw)) + 0.5, color='c', lw=1, label='recovery (g >= 0.8)')
    ax.axhline(len(fast), color='w', lw=0.6, ls='--')
    ax.set_xlim(-10, 70); ax.set_xlabel('ms from whisker stimulus'); ax.set_ylabel('whisker trials, sorted by recovery')
    ax.legend(fontsize=7, loc='lower right'); plt.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title('A  estimated gain (black/NaN = masked)', loc='left', fontsize=10)

    # example trace: slow trial with most dead-window events of example unit 0
    cl0 = ex[0]; tmpl, cs = T[cl0]
    pc = cs[int(np.argmax(np.abs(tmpl).max(0)))]
    w0 = E[(E.whisker == 1) & (E.cluster_id == cl0) & (E.kind == 0) & (E.t_ms > 5) & (E.t_ms < 30) & (E.g < 0.6)]
    k0 = int(w0.trial.value_counts().index[0]) if len(w0) else int(order[-1])
    h, e, g, s = R[k0]
    gg = np.where(np.isnan(g), np.nan, np.maximum(g, G_MIN))
    ax = fig.add_subplot(gs[0, 1:3])
    m = (P.tms > -20) & (P.tms < 70)
    off = 6 * np.std(h[P.bm, pc])
    ax.plot(P.tms[m], h[m, pc] + off, color='0.4', lw=0.6, label='high-passed, uncorrected')
    hc = h[:, pc] / gg
    ax.plot(P.tms[m], hc[m] - off, color='k', lw=0.6, label='gain-corrected')
    ax2 = ax.twinx(); ax2.plot(P.tms[m], g[m], color='c', lw=1); ax2.set_ylim(0, 3); ax2.set_ylabel('g_hf', color='c')
    ev = E[(E.whisker == 1) & (E.trial == k0) & (E.cluster_id == cl0)]
    for kind, col, y, lab in ((0, CR, -off, 'recovered (this unit)'), (1, CN, -off, 'null detection'), (2, 'k', off, 'KS4 spike')):
        tt = ev[ev.kind == kind].t_ms
        ax.plot(tt, np.full(len(tt), y + (2.2 if kind != 1 else -2.2) * off / 2), 'v' if kind != 1 else '^', color=col, ms=6, label=lab)
    deco(ax); ax.set_xlabel('ms'); ax.set_ylabel('uV (offset)'); ax.legend(fontsize=7, ncol=4, loc='lower right', frameon=False)
    ax.set_title(f'B  example trial {k0} (recovery {rt[k0]:.0f} ms), unit {cl0} peak channel {pc}', loc='left', fontsize=10)

    ax = fig.add_subplot(gs[0, 3])
    for key, col, ls, lab in ((Pall['ksw'], '0.3', '-', 'KS4 whisker'), (Pall['corr'], CR, '-', 'recovered whisker (null-corr.)'),
                              (Pall['null'], CN, ':', 'inverted null'), (aud['ks'], CA, '-', 'KS4 auditory')):
        ax.plot(P.ctr, smooth(key.mean(0)), color=col, ls=ls, label=lab)
    deco(ax); ax.set_xlabel('ms'); ax.set_ylabel('mean rate per unit (Hz)'); ax.legend(fontsize=7, frameon=False)
    ax2 = ax.twinx(); ax2.plot(P.ctr, Pall['cov'], color='k', lw=0.6, ls='--'); ax2.set_ylim(0, 1.05); ax2.set_ylabel('usable trial fraction', fontsize=8)
    ax.set_title(f'C  population PSTH, {nu} units', loc='left', fontsize=10)

    # rasters + per-unit PSTHs for the example units
    for j, cl in enumerate(ex):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow((covw[order] < 0.5).astype(float), aspect='auto', cmap='Greys', vmin=0, vmax=4,
                  extent=[P.ed[0], P.ed[-1], len(tw), 0], interpolation='none')
        pos = {k: i for i, k in enumerate(order)}
        ww = E[(E.whisker == 1) & (E.cluster_id == cl)]
        for kind, col, dy, sz in ((2, 'k', -0.2, 2), (0, CR, 0.2, 2)):
            q = ww[ww.kind == kind]
            ax.scatter(q.t_ms, q.trial.map(pos) + 0.5 + dy, s=sz, color=col, lw=0, rasterized=True)
        ax.plot(rt[order], np.arange(len(tw)) + 0.5, color='c', lw=0.8)
        deco(ax); ax.set_ylim(len(tw), 0); ax.set_xlabel('ms')
        if j == 0:
            ax.set_ylabel('whisker trials (sorted by recovery)')
        ax.set_title(f'D{j + 1}  unit {cl}: KS4 (black) vs recovered (orange)', loc='left', fontsize=9)
        ax = fig.add_subplot(gs[2, j])
        u = ui[cl]
        ax.plot(P.ctr, smooth(Pall['ksw'][u], 5), color='0.3', label='KS4 whisker')
        ax.plot(P.ctr, smooth(Pf['corr'][u], 5), color=CR, label='recovered, fast trials')
        ax.plot(P.ctr, smooth(Ps['corr'][u], 5), color=CR, ls='--', label='recovered, slow trials')
        ax.plot(P.ctr, smooth(Pall['null'][u], 5), color=CN, ls=':', label='null')
        ax.plot(P.ctr, smooth(aud['ks'][u], 5), color=CA, label='KS4 auditory')
        deco(ax); ax.set_xlabel('ms'); ax.set_ylabel('Hz' if j == 0 else '')
        if j == 0:
            ax.legend(fontsize=7, frameon=False)
        ax.set_title(f'E{j + 1}  unit {cl} PSTH (5-ms smoothing)', loc='left', fontsize=9)

    # heatmaps: baseline-subtracted, all units, same scale
    srt = np.argsort(-(Pall['corr'][:, d5].mean(1) - Pall['corr'][:, bm].mean(1)))
    mats = [(Pall['ksw'], 'F1  KS4 whisker'), (Pall['corr'], 'F2  recovered whisker (null-corr.)'), (aud['ks'], 'F3  KS4 auditory')]
    vm = np.nanpercentile(np.abs(smooth(aud['ks']) - aud['ks'][:, bm].mean(1, keepdims=True)), 98)
    for j, (M, ttl) in enumerate(mats):
        ax = fig.add_subplot(gs[3, j])
        Z = smooth(M)[srt] - M[srt][:, bm].mean(1, keepdims=True)
        im = ax.imshow(Z, aspect='auto', cmap='RdBu_r', vmin=-vm, vmax=vm, extent=[P.ed[0], P.ed[-1], nu, 0], interpolation='none')
        ax.axvline(0, color='k', lw=0.5); ax.axvline(5, color='k', lw=0.5, ls=':')
        if rel:
            for v_ in rel:
                ax.axvline(v_, color='m', lw=0.5, ls=':')
        ax.set_xlim(-20, 70); ax.set_xlabel('ms'); ax.set_ylabel('units (sorted by recovered 5-30 ms)' if j == 0 else '')
        ax.set_title(ttl + ' - baseline', loc='left', fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.04, label='Hz')
    ax = fig.add_subplot(gs[3, 3])
    for p_, ls, lab in ((Pf, '-', f'fast recovery ({len(fast)} trials)'), (Ps, '--', f'slow recovery ({len(slow)} trials)')):
        ax.plot(P.ctr, smooth(p_['corr'].mean(0)), color=CR, ls=ls, label='recovered, ' + lab)
        ax.plot(P.ctr, smooth(p_['ksw'].mean(0)), color='0.3', ls=ls, lw=0.8, label='KS4, ' + lab)
    ax.plot(P.ctr, smooth(aud['ks'].mean(0)), color=CA, label='KS4 auditory')
    deco(ax); ax.set_xlabel('ms'); ax.set_ylabel('mean rate per unit (Hz)'); ax.legend(fontsize=7, frameon=False)
    ax.set_title('G  population PSTH by trial recovery', loc='left', fontsize=10)
    fig.suptitle(f'{tag} (NP1.0): recovered spikes in the whisker dead time - TENTATIVE (HF gain, {nu} units clean F1 >= 0.6, '
                 f'{len(tw)} whisker / {len(ta)} auditory trials). Grey 0-5 ms = artefact'
                 + ('; pink = amplifier release period' if rel else ''), fontsize=11, y=0.905)
    for ext in ('png', 'pdf'):
        fig.savefig(f'{OUT}/{tag}_recovered_fig.{ext}', dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    main()
