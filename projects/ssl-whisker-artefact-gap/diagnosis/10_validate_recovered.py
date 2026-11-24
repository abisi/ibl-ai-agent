"""TENTATIVE: are the events recovered inside the dead window real spikes? (read-only, 2026-11-23)

Uses the pilot-08 machinery with the high-frequency gain estimate (08 ... hf). Per NP1.0 probe, units with clean F1 >= 0.6:
whisker trials are gain-corrected (x / max(g_hf, G_MIN); masked before the artefact end and where g < G_MIN) and searched
with each unit's template and threshold. Events are grouped by window:
  base   -55..-5 ms (g = 1 by construction)
  dead   +5..+30 ms where the trial's g_hf < 0.6
  late   +45..+80 ms where g_hf > 0.9
Checks (if the recovered events are real spikes of that unit, dead ~ base):
  1. amplitude distribution (matched-filter amplitude in corrected units, normalised per unit to its base median):
     real spikes keep the base distribution; noise crossings pile up just above threshold;
  2. amplitude vs estimated gain (5 bins): flat if spikes and noise shrink by the same factor (SNR preserved), falling
     with lower g if spikes shrink more than the noise;
  3. mean waveform per unit (corrected) vs its template: shape correlation and amplitude;
  4. noise control: the same detector on the sign-inverted corrected signal (template-shaped events of opposite
     polarity = noise / artefact): event rate per group; real recovery needs the dead-window rate to exceed this null.
Outputs: <ROOT>/gain_recovery_pilot/<tag>_validate.{png,json} + <tag>_validate_events.csv.gz
Usage: python 10_validate_recovered.py <0|1>
"""
import importlib.util
import json
import os
import sys

ci = int(sys.argv[1])
sys.argv = [sys.argv[0], str(ci), 'hf']                       # 08 reads its gain mode from argv[2] at import
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('p08', os.path.join(HERE, '08_gain_recovery_pilot.py'))
p08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p08)
OUT = p08.OUT
G_MIN = p08.G_MIN
GROUPS = dict(base=(-55, -5, 0.0, 9.0), dead=(5, 30, 0.0, 0.6), late=(45, 80, 0.9, 9.0))


def main():
    sid, probe = p08.CASES[ci]
    tag = f'{sid}_{probe}'
    r, x, fs, uv, maxint, tw, ta, st, sc, cch = p08.load(sid, probe)
    P = p08.Prep(x, fs, uv, maxint)
    old = np.load(f'{OUT}/{tag}_hf_psth.npz')
    chs = old['channels']
    U = pd.read_csv(f'{OUT}/{tag}_hf_units.csv')
    U = U[U.f1_clean >= 0.6]
    th = dict(zip(U.cluster_id.astype(int), U.theta))

    def prep(t):
        w, s = P.raw(t)
        e = P.art_end(w, chs)
        return P.run(w, chs, e), e, P.gain_hf(w, chs, e)
    R = Parallel(n_jobs=24, prefer='threads')(delayed(prep)(t) for t in tw)
    Ha = Parallel(n_jobs=24, prefer='threads')(delayed(lambda t: P.run(P.raw(t)[0], chs))(t) for t in ta)
    T, L0 = p08.templates(P, [q[0] for q in R] + Ha, st, sc, cch, chs, np.r_[tw, ta])
    units = [c for c in th if c in T]
    L = len(next(iter(T.values()))[0])
    nS = len(P.tms)
    idx = np.arange(nS)
    print(tag, 'units', len(units), flush=True)

    def per_trial(k):
        h, e, g = R[k]
        gg = np.where(np.isnan(g), 1.0, g)
        v = (idx < P.n0 - int(0.5e-3 * fs)) | ((idx >= e) & (gg >= G_MIN))
        hc = (h / np.maximum(gg, G_MIN)[:, None]).astype(np.float32)
        ev, wf = [], {}
        for cl in units:
            tmpl, cs = T[cl]
            for inv, sig in ((0, hc), (1, -hc)):
                d, amp = p08.detect(sig, tmpl, cs, th[cl], L0, v, fs)
                for sp, a in zip(d, amp):
                    if sp >= nS:
                        continue
                    tm, gv = P.tms[sp], gg[sp]
                    grp = next((nm for nm, (lo, hi, gl, gh) in GROUPS.items() if lo <= tm < hi and gl <= gv <= gh), None)
                    if grp is None:
                        continue
                    ev.append((k, cl, inv, grp, float(tm), float(a), float(gv)))
                    if not inv and sp - L0 >= 0 and sp - L0 + L <= nS:
                        key = (cl, grp)
                        seg = hc[sp - L0:sp - L0 + L][:, cs]
                        if key not in wf:
                            wf[key] = [np.zeros_like(seg, np.float64), 0]
                        wf[key][0] += seg; wf[key][1] += 1
        return ev, wf
    res = Parallel(n_jobs=48, prefer='threads')(delayed(per_trial)(k) for k in range(len(tw)))
    E = pd.DataFrame([e for q in res for e in q[0]], columns=['trial', 'cluster_id', 'inverted', 'group', 't_ms', 'amp', 'g'])
    E.to_csv(f'{OUT}/{tag}_validate_events.csv.gz', index=False)
    W = {}
    for q in res:
        for key, (s_, n_) in q[1].items():
            if key not in W:
                W[key] = [np.zeros_like(s_), 0]
            W[key][0] += s_; W[key][1] += n_

    # exposure (trial-ms) per group, for rates
    expo = {nm: 0.0 for nm in GROUPS}
    for h, e, g in R:
        gg = np.where(np.isnan(g), 1.0, g)
        ok = (idx < P.n0 - int(0.5e-3 * fs)) | ((idx >= e) & (gg >= G_MIN))
        for nm, (lo, hi, gl, gh) in GROUPS.items():
            m = ok & (P.tms >= lo) & (P.tms < hi) & (gg >= gl) & (gg <= gh)
            expo[nm] += m.sum() / fs                                             # seconds
    real = E[E.inverted == 0]
    basemed = real[real.group == 'base'].groupby('cluster_id').amp.median()
    E['amp_n'] = E.amp / E.cluster_id.map(basemed)
    out = dict(session_id=sid, probe=probe, n_units=len(units), exposure_s=expo)
    rows = []
    for nm in GROUPS:
        for inv in (0, 1):
            d = E[(E.group == nm) & (E.inverted == inv)]
            rows.append(dict(group=nm, inverted=inv, n=len(d), rate_hz_per_unit=len(d) / max(expo[nm], 1e-9) / len(units),
                             amp_n_median=float(d.amp_n.median()) if len(d) else np.nan,
                             amp_n_q25=float(d.amp_n.quantile(.25)) if len(d) else np.nan,
                             amp_n_q75=float(d.amp_n.quantile(.75)) if len(d) else np.nan))
    S = pd.DataFrame(rows)
    out['groups'] = S.round(4).to_dict('records')
    # shape: per unit, corr of mean waveform with template (flattened), and amplitude of mean waveform (<w,T>/<T,T>)
    shp = []
    for cl in units:
        tmpl = T[cl][0]
        for nm in GROUPS:
            if (cl, nm) in W and W[(cl, nm)][1] >= 10:
                m = W[(cl, nm)][0] / W[(cl, nm)][1]
                shp.append(dict(cluster_id=cl, group=nm, n=W[(cl, nm)][1], corr=float(np.corrcoef(m.ravel(), tmpl.ravel())[0, 1]),
                                amp=float((m * tmpl).sum() / (tmpl ** 2).sum())))
    Sh = pd.DataFrame(shp)
    out['shape_median'] = Sh.groupby('group')[['corr', 'amp']].median().round(3).to_dict('index') if len(Sh) else {}
    out['shape_n_units'] = Sh.groupby('group').size().to_dict() if len(Sh) else {}
    # amplitude vs gain (dead window events, real)
    dd = E[(E.inverted == 0) & (E.t_ms >= 5) & (E.t_ms < 30)].copy()
    dd['gbin'] = pd.cut(dd.g, [0, 0.2, 0.3, 0.4, 0.6, 0.9, 2])
    ag = dd.groupby('gbin', observed=True).agg(n=('amp_n', 'size'), amp_n=('amp_n', 'median')).reset_index()
    out['amp_vs_gain'] = [dict(gbin=str(a), n=int(b), amp_n=float(c)) for a, b, c in ag.itertuples(index=False)]
    json.dump(out, open(f'{OUT}/{tag}_validate.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 200)
    print(S.round(3).to_string(index=False)); print(out['shape_median'], out['shape_n_units']); print(ag.round(3).to_string(index=False), flush=True)

    fig, axs = plt.subplots(1, 4, figsize=(22, 4.8))
    ax = axs[0]
    bins = np.linspace(0, 3, 61)
    for nm, col in (('base', '0.3'), ('dead', '#d95f02'), ('late', '#1b9e77')):
        d = E[(E.group == nm) & (E.inverted == 0)]
        ax.hist(d.amp_n.dropna(), bins, density=True, histtype='step', color=col, lw=1.5, label=f'{nm} (n={len(d)})')
    d = E[(E.group == 'dead') & (E.inverted == 1)]
    ax.hist(d.amp_n.dropna(), bins, density=True, histtype='step', color='#d95f02', ls='--', label=f'dead, inverted null (n={len(d)})')
    ax.set_xlabel('event amplitude / unit baseline median'); ax.legend(fontsize=8, frameon=False)
    ax.set_title('A  amplitude distributions (corrected)', loc='left', fontsize=10)
    ax = axs[1]
    ax.plot(range(len(ag)), ag.amp_n, 'o-', color='#d95f02'); ax.axhline(1, color='0.5', lw=0.5)
    ax.set_xticks(range(len(ag))); ax.set_xticklabels([str(b) for b in ag.gbin], rotation=30, fontsize=8)
    ax.set_xlabel('estimated gain at event'); ax.set_ylabel('median amplitude / baseline'); ax.set_ylim(0, 2)
    ax.set_title('B  dead-window amplitude vs gain (flat = SNR kept)', loc='left', fontsize=10)
    ax = axs[2]
    x_ = np.arange(len(GROUPS))
    rr = S.pivot(index='group', columns='inverted', values='rate_hz_per_unit').reindex(list(GROUPS))
    ax.bar(x_ - 0.2, rr[0], 0.4, color='#d95f02', label='events'); ax.bar(x_ + 0.2, rr[1], 0.4, color='0.6', label='inverted null')
    ax.set_xticks(x_); ax.set_xticklabels(list(GROUPS)); ax.set_ylabel('Hz per unit'); ax.legend(fontsize=8, frameon=False)
    ax.set_title('C  event rate vs noise null', loc='left', fontsize=10)
    ax = axs[3]
    if len(Sh):
        piv = Sh.pivot(index='cluster_id', columns='group', values='corr')
        for nm, col in (('dead', '#d95f02'), ('late', '#1b9e77')):
            if nm in piv and 'base' in piv:
                ax.scatter(piv['base'], piv[nm], s=14, color=col, label=nm)
        ax.plot([0, 1], [0, 1], 'k', lw=0.5); ax.set_xlim(0, 1.02); ax.set_ylim(-0.2, 1.02)
        ax.set_xlabel('base: corr(mean waveform, template)'); ax.set_ylabel('window: corr'); ax.legend(fontsize=8, frameon=False)
    ax.set_title('D  waveform shape per unit', loc='left', fontsize=10)
    fig.suptitle(f'{tag} (NP1.0, high-frequency gain): validation of recovered dead-window events - TENTATIVE ({len(units)} units)', fontsize=11)
    fig.savefig(f'{OUT}/{tag}_validate.png', dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    main()
