"""036 -- figure suite for the two reward-related effects on the whisker-stimulus-evoked movement found in 026
(haas; reads the 024 cache only, no model fits).

Effects examined:
 (C) after a whisker hit (rewarded in R+, not in R-) the next whisker trial's evoked whisker movement is more often
     present in R+ only (026: whisker angle present +0.10, cohort p 0.02);
 (B) after the learning trial (017 table, lt_final in whisker-trial index) the evoked movement drops in R- only.

Definitions as 026: evoked = mean over 25-150 ms after the stimulus (angles / positions minus the -200..0 ms baseline);
hits kept only if the first jaw onset / lick / tongue event is > 150 ms after the stimulus; trials with a lick in -1..0 s
dropped; 'present' = above the session's 95th percentile on correct rejections. Previous trial = previous whisker trial.
Traces are truncated at the trial's first jaw / lick / tongue event (pre-lick).

Controls added here:
 - next-trial outcome: hits have larger evoked movement (024 AUC ~0.6), and a hit is more likely after a hit
   -> 'next: misses' and a per-session OLS  present ~ prev_hit + next_hit + position (whisker trials);
 - time in session: the evoked movement habituates over the session (026 A) and the LT split is early vs late
   -> per-session detrending on whisker-trial position (linear) before the before / after comparison.
Figures: figures/036_{1_examples,2_timecourses,3_after_hit,4_learning_trial}.png|pdf|svg; table tables/036_*.csv.
Exploratory, uncorrected; R+ vs R- Mann-Whitney + Welch, within cohort sign-flip (session = mouse).
"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("BM_SET", "full")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bm_lib as L  # noqa: E402

_sp = importlib.util.spec_from_file_location("m24", HERE / "024_stimulus_micromovements.py")
M24 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(M24)

plt.rcParams.update({"font.size": 6, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
WIN = (0.025, 0.150)
MEAS = ["whisker_angle", "top_particle_y", "whisker_speed", "nose_speed"]
LAB = {"whisker_angle": "whisker angle (deg)", "top_particle_y": "whisker particle y (mm)",
       "whisker_speed": "whisker speed (deg/s)", "nose_speed": "nose speed side (mm/s)"}
SI = {m: M24.SIGN.index(m) for m in MEAS}
TG = M24.TGRID
TPLOT = (TG >= -0.1) & (TG <= 0.3)
COL = L.COH_COL
rng = np.random.default_rng(0)


def signflip(x, n=10000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return np.nan
    s = rng.choice([-1, 1], size=(n, len(x)))
    return float((np.sum(np.abs((s * x).mean(1)) >= abs(x.mean())) + 1) / (n + 1))


def two(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan
    return float(stats.mannwhitneyu(a, b).pvalue), float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def ptxt(p):
    return "n.s." if not np.isfinite(p) else (f"p={p:.2g}" if p >= 0.001 else "p<0.001")


# ------------------------------------------------------------------------------------------------ data
def load():
    S = M24.M18.sessions()
    coh = dict(zip(S.session_id, S.reward_group))
    lt = pd.read_csv(M24.LT_TABLE)
    ltd = dict(zip(lt.session_id, lt.lt_final))
    w = (TG >= WIN[0]) & (TG < WIN[1])
    D = {}
    for f in sorted(M24.OUT.glob("*.npz")):
        sid = f.stem
        if sid not in coh:
            continue
        z = np.load(f)
        X, stim, lick, first, lb, wti = z["X"].astype(float), z["stim"], z["lick"], z["first_event"], z["lick_before"], z["whisker_trial_index"]
        valid = ~lb & ((lick == 0) | (first > WIN[1]))
        cr = valid & (stim == "no_stim") & (lick == 0)
        if cr.sum() < 10 or (stim == "whisker").sum() < 30:
            continue
        Tr = {}
        for m in MEAS:
            x = X[:, SI[m]].copy()
            x[TG[None, :] >= first[:, None]] = np.nan  # pre-lick traces
            Tr[m] = x
        with np.errstate(all="ignore"):
            V = {m: np.nanmean(X[:, SI[m]][:, w], 1) for m in MEAS}
        thr = {m: np.nanpercentile(V[m][cr], 95) for m in MEAS}
        P = {m: np.where(np.isfinite(V[m]), (V[m] > thr[m]).astype(float), np.nan) for m in MEAS}
        wt = np.flatnonzero(stim == "whisker")
        prev = np.full(len(stim), -1)
        prev[wt[1:]] = wt[:-1]
        pos = np.full(len(stim), np.nan)
        pos[wt] = np.arange(len(wt)) / (len(wt) - 1)
        D[sid] = dict(cohort=coh[sid], lt=ltd.get(sid, np.nan), stim=stim, lick=lick, valid=valid, cr=cr, wti=wti, prev=prev,
                      pos=pos, V=V, P=P, thr=thr, Tr=Tr, n_w=len(wt), hit_rate=float(lick[wt].mean()))
    return D


# ------------------------------------------------------------------------------------------------ per-session effects
def after_hit_table(D):
    rows = []
    for sid, d in D.items():
        st, lk, va, pv = d["stim"], d["lick"], d["valid"], d["prev"]
        nxt = np.flatnonzero((st == "whisker") & (pv >= 0) & va)
        ph = lk[pv[nxt]] == 1
        for m in MEAS:
            for kind, Y in (("present", d["P"][m]), ("magnitude", d["V"][m])):
                y = Y[nxt]
                ok = np.isfinite(y)
                r = dict(session_id=sid, cohort=d["cohort"], measure=m, kind=kind, n_after_hit=int((ok & ph).sum()),
                         n_after_miss=int((ok & ~ph).sum()), hit_rate=d["hit_rate"],
                         p_hit_after_hit=float(lk[nxt][ph].mean()) if ph.sum() else np.nan,
                         p_hit_after_miss=float(lk[nxt][~ph].mean()) if (~ph).sum() else np.nan)
                if (ok & ph).sum() >= 5 and (ok & ~ph).sum() >= 5:
                    r["raw"] = np.nanmean(y[ph]) - np.nanmean(y[~ph])
                    mm = ok & (lk[nxt] == 0)
                    if (mm & ph).sum() >= 5 and (mm & ~ph).sum() >= 5:
                        r["next_misses"] = np.nanmean(y[mm & ph]) - np.nanmean(y[mm & ~ph])
                    Xd = np.column_stack([np.ones(ok.sum()), ph[ok], lk[nxt][ok], d["pos"][nxt][ok]])
                    beta = np.linalg.lstsq(Xd, y[ok], rcond=None)[0]
                    r["ols_prev_hit"] = beta[1]
                    r["ols_next_hit"] = beta[2]
                    r["ols_position"] = beta[3]
                rows.append(r)
    return pd.DataFrame(rows)


def lt_table(D, n_null=200):
    """before / after the LT on whisker trials: raw, detrended on position, and a position-matched pseudo-LT null"""
    rows = []
    for sid, d in D.items():
        st, va, wti = d["stim"], d["valid"], d["wti"]
        if not np.isfinite(d["lt"]):
            continue
        for subset in ("all whisker", "whisker misses"):
          wt = np.flatnonzero((st == "whisker") & va & ((d["lick"] == 0) if subset == "whisker misses" else True))
          for m in MEAS:
            for kind, Y in (("present", d["P"][m]), ("magnitude", d["V"][m])):
                y, x, k = Y[wt], d["pos"][wt], wti[wt]
                ok = np.isfinite(y)
                y, x, k = y[ok], x[ok], k[ok]
                bef, aft = k < d["lt"], k >= d["lt"]
                if bef.sum() < 5 or aft.sum() < 5:
                    continue
                res = y - np.polyval(np.polyfit(x, y, 1), x)
                rows.append(dict(session_id=sid, cohort=d["cohort"], subset=subset, measure=m, kind=kind, lt=d["lt"], n_before=int(bef.sum()),
                                 n_after=int(aft.sum()), raw=y[aft].mean() - y[bef].mean(), detrended=res[aft].mean() - res[bef].mean()))
    return pd.DataFrame(rows)


def group(df, cols, keys):
    out = []
    for k, g in df.groupby(keys):
        for c in cols:
            a, b = g[g.cohort == "R+"][c].dropna(), g[g.cohort == "R-"][c].dropna()
            pm, pw = two(a, b)
            out.append(dict(**dict(zip(keys, k)), estimate=c, n_Rplus=len(a), n_Rminus=len(b), mean_Rplus=a.mean(),
                            p_Rplus=signflip(a), mean_Rminus=b.mean(), p_Rminus=signflip(b), p_cohort_mw=pm, p_cohort_welch=pw))
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ figures
def save(fig, name):
    t = fig._suptitle
    if t is not None:
        import textwrap
        t.set_text("\n".join(textwrap.fill(par, 165) for par in t.get_text().split("\n")))
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"036_{name}.{ext}", dpi=300)
    plt.close(fig)


def pick_examples(D, AH):
    """per cohort: session with most after-hit whisker trials whose raw whisker-angle effect is near the cohort median"""
    q = AH[(AH.measure == "whisker_angle") & (AH.kind == "present")].dropna(subset=["raw"])
    ex = {}
    for cg in ("R+", "R-"):
        g = q[(q.cohort == cg) & (q.n_after_hit >= 15)].copy()
        if not len(g):
            g = q[q.cohort == cg].copy()
        med = g.raw.median()
        g["score"] = -abs(g.raw - med) / (g.raw.std() + 1e-9) + 0.02 * g.n_after_hit
        ex[cg] = g.sort_values("score").session_id.iloc[-1]
    return ex


def fig_examples(D, ex):
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 5.0), gridspec_kw=dict(width_ratios=[1.3, 0.12, 1, 1], hspace=0.55, wspace=0.45,
                                                                     left=0.07, right=0.98, top=0.88, bottom=0.08))
    t = TG[TPLOT]
    for r, cg in enumerate(("R+", "R-")):
        sid = ex[cg]
        d = D[sid]
        st, lk, va = d["stim"], d["lick"], d["valid"]
        wt = np.flatnonzero(st == "whisker")
        Y = d["Tr"]["whisker_angle"][wt][:, TPLOT]
        ax = axs[r, 0]
        vm = np.nanpercentile(np.abs(Y), 97)
        ax.imshow(Y, aspect="auto", cmap="RdBu_r", vmin=-vm, vmax=vm, extent=[t[0], t[-1], len(wt), 0], interpolation="none")
        ax.axvspan(*WIN, color="k", alpha=0.06, lw=0)
        ax.axvline(0, color="k", lw=0.4)
        ax.set_ylabel("whisker trial (session order)")
        ax.set_xlabel("time from whisker stimulus (s)")
        ax.set_title(f"{cg} example {sid}\nwhisker angle (deg, baseline-subtracted; blank = after first jaw / lick event)", fontsize=5.5)
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vm, vm))
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
        cb.ax.tick_params(labelsize=4.5)
        ax = axs[r, 1]  # rug: outcome of this trial and evoked present
        ax.imshow(np.c_[lk[wt], np.nan_to_num(d["P"]["whisker_angle"][wt], nan=-1)], aspect="auto",
                  cmap=matplotlib.colors.ListedColormap(["#dddddd", "white", COL[cg]]), vmin=-1, vmax=1, interpolation="none",
                  extent=[0, 2, len(wt), 0])
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["hit", "evoked"], rotation=90, fontsize=4.5)
        ax.set_yticks([])
        # mean traces: next whisker trial after hit vs after miss (all next; next misses), CR reference
        pv = d["prev"]
        nxt = np.flatnonzero((st == "whisker") & (pv >= 0) & va)
        ph = lk[pv[nxt]] == 1
        for j, (sub, msk) in enumerate((("next whisker trial: all", np.ones(len(nxt), bool)), ("next whisker trial: misses only", lk[nxt] == 0))):
            ax = axs[r, 2 + j]
            for lab_, sel, c, ls in (("after hit", ph & msk, COL[cg], "-"), ("after miss", ~ph & msk, "0.45", "-")):
                Z = d["Tr"]["whisker_angle"][nxt[sel]][:, TPLOT]
                mu, se = np.nanmean(Z, 0), np.nanstd(Z, 0) / np.sqrt(np.sum(np.isfinite(Z), 0))
                ax.plot(t, mu, color=c, lw=0.8, ls=ls, label=f"{lab_} (n={sel.sum()})")
                ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.2, lw=0)
            Z = d["Tr"]["whisker_angle"][d["cr"]][:, TPLOT]
            ax.plot(t, np.nanmean(Z, 0), color="k", lw=0.5, ls=":", label=f"correct rejections (n={d['cr'].sum()})")
            ax.axvspan(*WIN, color="k", alpha=0.06, lw=0)
            ax.axvline(0, color="k", lw=0.4)
            ax.set_title(sub, fontsize=5.5)
            ax.set_xlabel("time from whisker stimulus (s)")
            if j == 0:
                ax.set_ylabel("whisker angle (deg)")
            ax.legend(frameon=False, fontsize=4.3, loc="upper left")
    fig.suptitle("036.1  Examples: whisker-stimulus-evoked whisker movement on every whisker trial (left) and on the whisker trial after a "
                 "hit vs after a miss (right). Grey band = 25-150 ms evoked window. Session chosen near its cohort's median effect.",
                 fontsize=6, x=0.01, ha="left")
    save(fig, "1_examples")


def fig_timecourses(D):
    fig, axs = plt.subplots(len(MEAS), 4, figsize=(7.2, 7.0), sharex=True,
                            gridspec_kw=dict(hspace=0.45, wspace=0.45, left=0.08, right=0.98, top=0.9, bottom=0.06))
    t = TG[TPLOT]
    for i, m in enumerate(MEAS):
        for c_, cg in enumerate(("R+", "R-")):
            for j, sub in enumerate(("all", "misses")):
                ax = axs[i, c_ * 2 + j]
                M = {"after hit": [], "after miss": [], "CR": []}
                for sid, d in D.items():
                    if d["cohort"] != cg:
                        continue
                    st, lk, va, pv = d["stim"], d["lick"], d["valid"], d["prev"]
                    nxt = np.flatnonzero((st == "whisker") & (pv >= 0) & va)
                    msk = np.ones(len(nxt), bool) if sub == "all" else lk[nxt] == 0
                    ph = lk[pv[nxt]] == 1
                    if (ph & msk).sum() < 5 or (~ph & msk).sum() < 5:
                        continue
                    with np.errstate(all="ignore"):
                        M["after hit"].append(np.nanmean(d["Tr"][m][nxt[ph & msk]][:, TPLOT], 0))
                        M["after miss"].append(np.nanmean(d["Tr"][m][nxt[~ph & msk]][:, TPLOT], 0))
                        M["CR"].append(np.nanmean(d["Tr"][m][d["cr"]][:, TPLOT], 0))
                for k, c, ls in (("after hit", COL[cg], "-"), ("after miss", "0.45", "-"), ("CR", "k", ":")):
                    Z = np.array(M[k])
                    if not len(Z):
                        continue
                    mu, se = np.nanmean(Z, 0), np.nanstd(Z, 0) / np.sqrt(len(Z))
                    ax.plot(t, mu, color=c, lw=0.8, ls=ls, label=f"{k}" + (f" ({len(Z)} sessions)" if k == "after hit" else ""))
                    if k != "CR":
                        ax.fill_between(t, mu - se, mu + se, color=c, alpha=0.2, lw=0)
                ax.axvspan(*WIN, color="k", alpha=0.06, lw=0)
                ax.axvline(0, color="k", lw=0.4)
                ax.tick_params(labelsize=4.8, length=1.5, pad=1)
                if i == 0:
                    ax.set_title(f"{cg}, next whisker trial: {sub}", fontsize=5.8, color=COL[cg])
                    ax.legend(frameon=False, fontsize=4.3)
                if c_ * 2 + j == 0:
                    ax.set_ylabel(LAB[m], fontsize=5.5)
                if i == len(MEAS) - 1:
                    ax.set_xlabel("time from whisker stimulus (s)")
    fig.suptitle("036.2  Group time courses (mean ± SEM over sessions): evoked movement on the whisker trial after a whisker hit vs after a "
                 "whisker miss.\nLeft pair: all next trials; right pair: next trial is a miss (removes the 'next trial is also a hit' "
                 "confound). Traces end at the first jaw / lick event.", fontsize=6, x=0.01, ha="left")
    save(fig, "2_timecourses")


def dots(ax, df, cols, labels, ylabel, G, keyfilter):
    for h, c in enumerate(cols):
        for c_, cg in enumerate(("R+", "R-")):
            v = df[df.cohort == cg][c].dropna()
            x = h * 2.6 + c_
            ax.scatter(x + np.random.default_rng(h * 7 + c_).uniform(-0.17, 0.17, len(v)), v, s=3, color=COL[cg], lw=0, alpha=0.6)
            ax.errorbar(x + 0.32, v.mean(), v.sem(), color="k", lw=0.7, marker="o", ms=1.8, capsize=0)
            r = G[keyfilter(G) & (G.estimate == c)]
            if len(r):
                p = r[f"p_{'Rplus' if cg == 'R+' else 'Rminus'}"].iloc[0]
                ax.text(h * 2.6 + 0.5, 1.0 + 0.045 * c_, ptxt(p), transform=ax.get_xaxis_transform(), ha="center", fontsize=4.2, color=COL[cg], va="bottom")
        r = G[keyfilter(G) & (G.estimate == c)]
        if len(r):
            ax.text(h * 2.6 + 0.5, -0.17, f"{ptxt(r.p_cohort_mw.iloc[0])}\n{ptxt(r.p_cohort_welch.iloc[0])}".replace("p=", ""),
                    transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4.2)
    ax.text(-0.03, -0.17, "R+ vs R−  MW\nWelch", transform=ax.transAxes, ha="right", va="top", fontsize=4.2)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks([h * 2.6 + 0.5 for h in range(len(cols))])
    ax.set_xticklabels(labels, fontsize=4.8)
    ax.set_ylabel(ylabel, fontsize=5.3)
    ax.tick_params(labelsize=4.8, length=1.5, pad=1)


def fig_after_hit(AH, GA):
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 4, hspace=0.95, wspace=0.55, left=0.08, right=0.98, top=0.86, bottom=0.1)
    # A: why the raw contrast is confounded: P(next whisker trial = hit | prev hit / miss)
    ax = fig.add_subplot(gs[0, 0])
    q = AH[(AH.measure == "whisker_angle") & (AH.kind == "present")]
    for c_, cg in enumerate(("R+", "R-")):
        g = q[q.cohort == cg]
        for _, r in g.iterrows():
            ax.plot([c_ * 2.4, c_ * 2.4 + 1], [r.p_hit_after_miss, r.p_hit_after_hit], color=COL[cg], lw=0.3, alpha=0.4)
        ax.errorbar([c_ * 2.4, c_ * 2.4 + 1], [g.p_hit_after_miss.mean(), g.p_hit_after_hit.mean()],
                    [g.p_hit_after_miss.sem(), g.p_hit_after_hit.sem()], color="k", lw=0.8, marker="o", ms=2)
    ax.set_xticks([0, 1, 2.4, 3.4])
    ax.set_xticklabels(["after\nmiss", "after\nhit", "after\nmiss", "after\nhit"], fontsize=4.6)
    ax.set_ylabel("P(next whisker trial is a hit)", fontsize=5.3)
    ax.set_title("A  hits follow hits\n(R+ green, R− magenta)", fontsize=5.6, loc="left")
    ax.tick_params(labelsize=4.8, length=1.5, pad=1)
    # B-D: estimates per measure (present)
    for j, m in enumerate(MEAS[:3]):
        ax = fig.add_subplot(gs[0, 1 + j])
        df = AH[(AH.measure == m) & (AH.kind == "present")]
        dots(ax, df, ["raw", "next_misses", "ols_prev_hit"], ["raw", "next = miss", "OLS: + next hit\n+ position"],
             "Δ P(evoked) after hit − after miss" if j == 0 else "", GA, lambda G, m=m: (G.measure == m) & (G.kind == "present"))
        ax.set_title(f"{'BCD'[j]}  {LAB[m]}", fontsize=5.6, loc="left", pad=16)
    # E-H: magnitude
    for j, m in enumerate(MEAS):
        ax = fig.add_subplot(gs[1, j])
        df = AH[(AH.measure == m) & (AH.kind == "magnitude")]
        dots(ax, df, ["raw", "next_misses", "ols_prev_hit"], ["raw", "next = miss", "OLS"], f"Δ evoked {LAB[m]}", GA,
             lambda G, m=m: (G.measure == m) & (G.kind == "magnitude"))
        ax.set_title(f"{'EFGH'[j]}  magnitude: {m.replace('_', ' ')}", fontsize=5.6, loc="left", pad=16)
    fig.suptitle("036.3  Is the evoked movement larger after a whisker hit? Per session (dots), mean ± SEM; p above = sign-flip within cohort.\n"
                 "raw = 026 contrast; next = miss = only next trials without a lick; OLS = per-session regression of the next trial's evoked "
                 "measure on previous hit, next-trial hit and position in session (coefficient of previous hit).", fontsize=6, x=0.01, ha="left")
    save(fig, "3_after_hit")


def fig_learning_trial(D, LT, GL):
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 4, hspace=0.95, wspace=0.55, left=0.08, right=0.98, top=0.86, bottom=0.1)
    edges = np.arange(-40, 41, 10)
    cen = edges[:-1] + 5
    panels = [("hit", None)] + [(m, "present") for m in MEAS[:3]]
    for j, (m, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        for cg, sub in [(c, s_) for c in ("R+", "R-") for s_ in (("all", "misses") if m != "hit" else ("all",))]:
            Z = []
            for sid, d in D.items():
                if d["cohort"] != cg or not np.isfinite(d["lt"]):
                    continue
                wt = np.flatnonzero((d["stim"] == "whisker") & d["valid"] & ((d["lick"] == 0) if sub == "misses" else True))
                rel = d["wti"][wt] - d["lt"]
                y = d["lick"][wt].astype(float) if m == "hit" else d["P"][m][wt]
                b = np.digitize(rel, edges) - 1
                Z.append([np.nanmean(y[b == k]) if np.sum((b == k) & np.isfinite(y)) >= 3 else np.nan for k in range(len(cen))])
            Z = np.array(Z)
            n = np.sum(np.isfinite(Z), 0)
            mu, se = np.nanmean(Z, 0), np.nanstd(Z, 0) / np.sqrt(np.maximum(n, 1))
            ax.errorbar(cen + (1 if cg == "R-" else 0), mu, se, color=COL[cg], lw=0.8, marker="o", ms=1.8, capsize=0,
                        ls="-" if sub == "all" else "--", alpha=1 if sub == "all" else 0.6,
                        label=f"{cg} learners (n={len(Z)})" + ("" if sub == "all" else ", misses only"))
        ax.axvline(0, color="k", lw=0.4, ls="--")
        if m != "hit":
            ax.axhline(0.05, color="k", lw=0.4, ls=":")
        ax.set_xlabel("whisker trials from learning trial")
        ax.set_title(f"{'ABCD'[j]}  " + ("P(whisker hit)" if m == "hit" else f"P(evoked) {m.replace('_', ' ')}"), fontsize=5.6, loc="left")
        ax.tick_params(labelsize=4.8, length=1.5, pad=1)
        if j in (0, 1):
            ax.legend(frameon=False, fontsize=4.2)
    for j, m in enumerate(MEAS):
        ax = fig.add_subplot(gs[1, j])
        df = LT[(LT.measure == m) & (LT.kind == "present")]
        a = df[df.subset == "all whisker"].set_index(["session_id", "cohort"])[["raw", "detrended"]]
        b = df[df.subset == "whisker misses"].set_index(["session_id", "cohort"])[["raw", "detrended"]].add_suffix("_m")
        df2 = a.join(b, how="outer").reset_index()
        G2 = pd.concat([GL[GL.subset == "all whisker"], GL[GL.subset == "whisker misses"].assign(estimate=lambda x: x.estimate + "_m")])
        dots(ax, df2, ["raw", "detrended", "raw_m", "detrended_m"], ["all\nraw", "all\ndetr.", "misses\nraw", "misses\ndetr."],
             "Δ P(evoked) after − before LT" if j == 0 else "", G2, lambda G, m=m: (G.measure == m) & (G.kind == "present"))
        ax.set_title(f"{'EFGH'[j]}  {m.replace('_', ' ')}", fontsize=5.6, loc="left", pad=16)
    fig.suptitle("036.4  Evoked movement around the learning trial (learners with an LT; 10-whisker-trial bins; dotted = CR false-positive rate).\n"
                 "Bottom: after − before the LT per session, raw and after removing each session's linear trend over whisker-trial position "
                 "(the evoked movement habituates over the session); all whisker trials and misses only (hits carry larger evoked movement). Top: dashed = whisker misses only.", fontsize=6, x=0.01, ha="left")
    save(fig, "4_learning_trial")


def main():
    D = load()
    print(len(D), "sessions", flush=True)
    AH = after_hit_table(D)
    LT = lt_table(D)
    GA = group(AH, ["raw", "next_misses", "ols_prev_hit", "ols_next_hit", "ols_position"], ["measure", "kind"])
    GL = group(LT, ["raw", "detrended"], ["subset", "measure", "kind"])
    for df, name in ((AH, "after_hit_per_session"), (LT, "learning_trial_per_session"), (GA, "after_hit_group"), (GL, "learning_trial_group")):
        df.to_csv(L.TABLES / f"036_{name}.csv", index=False)
    ex = pick_examples(D, AH)
    print("examples", ex, flush=True)
    fig_examples(D, ex)
    fig_timecourses(D)
    fig_after_hit(AH, GA)
    fig_learning_trial(D, LT, GL)
    pd.set_option("display.width", 250)
    print(GA.round(4).to_string(index=False))
    print(GL.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
