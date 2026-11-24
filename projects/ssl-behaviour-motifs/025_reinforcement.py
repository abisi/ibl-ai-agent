"""025 -- does reward reinforce the behaviour that precedes it (Markowitz et al. 2023 logic, head-fixed task)?
(haas, full run; BM_SET=full)

Usage: BM_SET=full python 025_reinforcement.py compute   (per session -> cache/full/025/<sid>.pkl)
       BM_SET=full python 025_reinforcement.py stats     (tables/025_*.csv, figures/025_*.png|pdf|svg)

Events: reward = corrected first lick of every active rewarded trial (reward_available = 1 and lick; R+ whisker hits,
auditory hits of both cohorts); unrewarded lick = corrected first lick of active lick trials without reward (R- whisker
hits, false alarms of both cohorts).
Per session and state set (lightning-action still / whisking / jaw, kpms meta-states side / top, shared kpms syllables
side / top, shared AR-HMM states side / top):
- pre-event usage: fraction of frames in each state in [lick - 0.5 s, lick) and in [stimulus, lick), averaged over
  rewards and over unrewarded licks;
- baseline usage: non-reward frames (task period, > 10 s after the last reward, > 0.5 s from any piezo lick / tongue-out
  frame), whole session, first and last third of the kept trials (session time);
- enrichment = pre-reward usage / baseline usage (log2, states with baseline >= 0.5 % only);
- growth = late - early non-reward usage, relative to the baseline (log2 ratio late / early);
- local reinforcement: 2-min bins of non-reward usage u_s(b); predictors: number of rewards in the previous 5 min that
  were preceded by state s (s present in >= 50 % of the 0.5-s pre-lick window), total rewards in the previous 5 min,
  bin time; per session OLS on z-scored predictors, coefficient of the s-specific count (states with >= 5 such rewards).
Tests (unit = session = mouse): per session Spearman(enrichment, growth) across states -> sign-flip vs 0 per cohort,
R+ vs R- MW + Welch; the same for the unrewarded-lick enrichment (control); whisking specifically (lightning-action
whisking; whisking meta-states): pre-reward enrichment, growth per cohort, R+ vs R-; local coefficients pooled over
states (per-session mean) -> sign-flip.
"""
import importlib.util
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
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

_sp = importlib.util.spec_from_file_location("m18", HERE / "018_report_examples.py")
M18 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(M18)

plt.rcParams.update({"font.size": 5.5, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5})
T = L.TABLES
OUT = L.CACHE / "full" / "025"
SETS = ["la", "meta_side", "meta_top", "kpms_side", "kpms_top", "arhmm_side", "arhmm_top"]
PRE = 0.5
NONREW_S = 10.0
NOLICK_S = 0.5
BIN_S = 120.0
HIST_S = 300.0
rng = np.random.default_rng(0)


def nstates(name, z):
    return M18.nstates(name, z)


def one(sid):
    S = M18.load_session(sid)
    t, Fm, Z, k = S["t"], S["feats"], S["Z"], S["trials"]
    tr = pd.read_parquet(M18.INP / f"{sid}_trials.parquet")
    act = tr[(tr.context != "passive") & (tr.lick_flag == 1) & np.isfinite(tr.first_lick_time)]
    rew = act[act.reward_available == 1]
    unr = act[act.reward_available == 0]
    rt = np.sort(rew.first_lick_time.to_numpy())
    j = np.searchsorted(rt, t, side="right") - 1
    since = np.where(j >= 0, t - rt[np.clip(j, 0, None)], np.inf)
    tong = np.isfinite(Fm[:, 3]) & (Fm[:, 3] > 0.5)
    ev = np.sort(np.r_[S["licks"], t[tong]])
    jj = np.clip(np.searchsorted(ev, t), 1, len(ev) - 1)
    nl = np.minimum(np.abs(t - ev[jj - 1]), np.abs(ev[jj] - t)) > NOLICK_S
    starts = k.start_time.to_numpy()
    task = (t >= starts[0] - 2) & (t <= starts[-1] + 6.5)
    base = task & (since > NONREW_S) & nl
    n = len(starts)
    early = t < starts[n // 3]
    late = t >= starts[(2 * n) // 3]
    out = dict(session_id=sid, n_rewards=len(rew), n_unrewarded_licks=len(unr), sets={})
    edges = np.arange(starts[0] - 2, starts[-1] + 6.5 + BIN_S, BIN_S)
    bidx = np.digitize(t, edges) - 1
    for name in SETS:
        if name not in Z:
            continue
        z = Z[name]
        ns = nstates(name, z)

        def usage(m):
            zz = z[m]
            zz = zz[zz >= 0]
            return np.bincount(zz, minlength=ns)[:ns] / len(zz) if len(zz) >= 100 else np.full(ns, np.nan)

        def pre_usage(df, start_col=None):
            U = []
            for _, r in df.iterrows():
                a = r.first_lick_time - PRE if start_col is None else r[start_col]
                b = r.first_lick_time
                i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
                zz = z[i0:i1]
                zz = zz[zz >= 0]
                if len(zz) >= 5:
                    U.append(np.bincount(zz, minlength=ns)[:ns] / len(zz))
            return (np.mean(U, 0) if U else np.full(ns, np.nan)), np.array(U) if U else np.zeros((0, ns))
        pre_r, Ur = pre_usage(rew)
        pre_rs, _ = pre_usage(rew, "start_time")
        pre_u, _ = pre_usage(unr)
        # local: 2-min bins of non-reward usage; per reward, which states dominated its pre-lick window
        nb = len(edges) - 1
        ub = np.full((nb, ns), np.nan)
        for b in range(nb):
            m = base & (bidx == b)
            zz = z[m]
            zz = zz[zz >= 0]
            if len(zz) >= 2000:  # >= 10 s of usable frames
                ub[b] = np.bincount(zz, minlength=ns)[:ns] / len(zz)
        dom = Ur >= 0.5 if len(Ur) else np.zeros((0, ns), bool)
        rtimes = []
        for _, r in rew.iterrows():
            i0, i1 = np.searchsorted(t, r.first_lick_time - PRE), np.searchsorted(t, r.first_lick_time)
            zz = z[i0:i1]
            if (zz >= 0).sum() >= 5:
                rtimes.append(r.first_lick_time)
        rtimes = np.array(rtimes)
        out["sets"][name] = dict(base=usage(base), early=usage(base & early), late=usage(base & late), pre_reward=pre_r,
                                 pre_reward_from_stim=pre_rs, pre_unrewarded=pre_u, bin_usage=ub, bin_edges=edges,
                                 reward_times=rtimes, reward_dominant=dom, all_reward_times=rt)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{sid}.pkl", "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    return sid


def compute():
    sids = list(M18.sessions().session_id)
    with ProcessPoolExecutor(int(os.environ.get("BM_WORKERS", 12))) as ex:
        for s in ex.map(one, sids):
            print("done", s, flush=True)


# ------------------------------------------------------------------------------------------------ stats
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


def run_stats():
    Sx = M18.sessions()
    coh = dict(zip(Sx.session_id, Sx.reward_group))
    Pp = pd.read_csv(T / "full_009_fingerprints_pooled.csv")
    shared = {f"{r.method}_{r.view}": set(int(s) for s in Pp[(Pp.method == r.method) & (Pp.view == r.view) & Pp.shared].state)
              for r in Pp.drop_duplicates(["method", "view"]).itertuples()}
    mn = list(pd.read_csv(T / "full_009_metastates.csv").name)
    meta = {f"{r.method}_{r.view}:{r.state}": r.meta_name for r in Pp.itertuples()}

    def valid(name, s):
        if name == "la":
            return s in (1, 2, 3)
        if name.startswith("meta"):
            return s < len(mn)
        return s in shared.get(name, set())

    def label(name, s):
        if name == "la":
            return f"action {M18.LA[s]}"
        if name.startswith("meta"):
            return f"{name.replace('_', ' ')} {mn[s]}"
        return f"{name.replace('_', ' ')} {'S' if name.startswith('kpms') else 'A'}{s} ({meta.get(f'{name}:{s}', '?')})"
    rows, corr, local = [], [], []
    for f in sorted(OUT.glob("*.pkl")):
        d = pickle.load(open(f, "rb"))
        sid = d["session_id"]
        if sid not in coh:
            continue
        for name, r in d["sets"].items():
            ns = len(r["base"])
            for s in range(ns):
                if not valid(name, s):
                    continue
                b, e, l_, pr, prs, pu = (r[k][s] for k in ("base", "early", "late", "pre_reward", "pre_reward_from_stim", "pre_unrewarded"))
                ok = np.isfinite(b) and b >= 0.005
                rows.append(dict(session_id=sid, cohort=coh[sid], set=name, state=s, label=label(name, s), base=b, early=e, late=l_,
                                 pre_reward=pr, pre_reward_from_stim=prs, pre_unrewarded=pu,
                                 enrich_reward=np.log2((pr + 1e-3) / (b + 1e-3)) if ok and np.isfinite(pr) else np.nan,
                                 enrich_reward_stim=np.log2((prs + 1e-3) / (b + 1e-3)) if ok and np.isfinite(prs) else np.nan,
                                 enrich_unrewarded=np.log2((pu + 1e-3) / (b + 1e-3)) if ok and np.isfinite(pu) else np.nan,
                                 growth=np.log2((l_ + 1e-3) / (e + 1e-3)) if ok and np.isfinite(e) and np.isfinite(l_) else np.nan,
                                 n_rewards=d["n_rewards"], n_unrewarded=d["n_unrewarded_licks"]))
            # local reinforcement
            ub, edges, rtm, dom, allr = r["bin_usage"], r["bin_edges"], r["reward_times"], r["reward_dominant"], r["all_reward_times"]
            ctr = (edges[:-1] + edges[1:]) / 2
            ok_b = np.isfinite(ub).all(1)
            if ok_b.sum() < 8 or len(rtm) < 5:
                continue
            tot = np.array([np.sum((allr >= c - BIN_S / 2 - HIST_S) & (allr < c - BIN_S / 2)) for c in ctr])
            for s in range(ns):
                if not valid(name, s) or len(dom) == 0 or dom[:, s].sum() < 5:
                    continue
                cnt = np.array([np.sum(dom[:, s] & (rtm >= c - BIN_S / 2 - HIST_S) & (rtm < c - BIN_S / 2)) for c in ctr])
                y = ub[ok_b, s]
                X = np.c_[cnt[ok_b], tot[ok_b], ctr[ok_b]]
                if np.std(y) == 0 or np.any(X.std(0) == 0):
                    continue
                Xz = (X - X.mean(0)) / X.std(0)
                beta = np.linalg.lstsq(np.c_[np.ones(len(y)), Xz], (y - y.mean()) / y.std(), rcond=None)[0]
                local.append(dict(session_id=sid, cohort=coh[sid], set=name, state=s, label=label(name, s), beta_specific=beta[1],
                                  beta_total=beta[2], beta_time=beta[3], n_bins=int(ok_b.sum()), n_dom_rewards=int(dom[:, s].sum())))
    E = pd.DataFrame(rows)
    E.to_csv(T / "025_enrichment_growth_per_session.csv", index=False)
    Lc = pd.DataFrame(local)
    Lc.to_csv(T / "025_local_reinforcement_per_session.csv", index=False)
    # per-session correlation across states (all sets pooled, and per set)
    for (sid, cg), g in E.groupby(["session_id", "cohort"]):
        for scope, gg in [("all states", g)] + [(s_, g[g.set == s_]) for s_ in SETS]:
            for ek in ("enrich_reward", "enrich_reward_stim", "enrich_unrewarded"):
                q = gg[[ek, "growth"]].dropna()
                if len(q) >= 4 and q[ek].std() > 0 and q.growth.std() > 0:
                    corr.append(dict(session_id=sid, cohort=cg, scope=scope, enrichment=ek, rho=stats.spearmanr(q[ek], q.growth)[0],
                                     n_states=len(q)))
    C = pd.DataFrame(corr)
    C.to_csv(T / "025_enrichment_growth_corr_per_session.csv", index=False)
    grp = []
    for (scope, ek), g in C.groupby(["scope", "enrichment"]):
        a, b = g[g.cohort == "R+"].rho, g[g.cohort == "R-"].rho
        pm, pw = two(a, b)
        grp.append(dict(scope=scope, enrichment=ek, rho_Rplus=a.mean(), p_Rplus=signflip(a), n_Rplus=a.notna().sum(),
                        rho_Rminus=b.mean(), p_Rminus=signflip(b), n_Rminus=b.notna().sum(), p_cohort_mw=pm, p_cohort_welch=pw))
    G = pd.DataFrame(grp)
    G.to_csv(T / "025_enrichment_growth_corr_group.csv", index=False)
    # per state: enrichment and growth by cohort
    st = []
    for (setn, s, lab), g in E.groupby(["set", "state", "label"]):
        a, b = g[g.cohort == "R+"], g[g.cohort == "R-"]
        pm, pw = two(a.growth, b.growth)
        st.append(dict(set=setn, state=s, label=lab, enrich_reward_Rplus=a.enrich_reward.mean(), enrich_reward_Rminus=b.enrich_reward.mean(),
                       p_enrich_Rplus=signflip(a.enrich_reward), enrich_unrewarded_Rminus=b.enrich_unrewarded.mean(),
                       growth_Rplus=a.growth.mean(), p_growth_Rplus=signflip(a.growth), growth_Rminus=b.growth.mean(),
                       p_growth_Rminus=signflip(b.growth), p_growth_cohort_mw=pm, p_growth_cohort_welch=pw,
                       base_Rplus=a.base.mean(), base_Rminus=b.base.mean()))
    ST = pd.DataFrame(st)
    ST.to_csv(T / "025_state_enrichment_growth.csv", index=False)
    lg = []
    for (scope_set,), g in [((s_,), Lc[Lc.set == s_]) for s_ in SETS] + [(("all states",), Lc)]:
        pm_ = g.groupby(["session_id", "cohort"]).beta_specific.mean().reset_index()
        a, b = pm_[pm_.cohort == "R+"].beta_specific, pm_[pm_.cohort == "R-"].beta_specific
        pm, pw = two(a, b)
        lg.append(dict(scope=scope_set, beta_Rplus=a.mean(), p_Rplus=signflip(a), n_Rplus=len(a), beta_Rminus=b.mean(),
                       p_Rminus=signflip(b), n_Rminus=len(b), p_cohort_mw=pm, p_cohort_welch=pw))
    LG = pd.DataFrame(lg)
    LG.to_csv(T / "025_local_reinforcement_group.csv", index=False)
    figure(E, C, G, ST, Lc)
    json.dump(dict(script="025_reinforcement.py", pre_s=PRE, nonreward_s=NONREW_S, nolick_s=NOLICK_S, bin_s=BIN_S, hist_s=HIST_S,
                   versions=L.versions()), open(L.RES / "provenance" / "025.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(G.round(4).to_string())
    print(LG.round(4).to_string())
    print(ST.sort_values("enrich_reward_Rplus", ascending=False).head(15)[
        ["label", "enrich_reward_Rplus", "enrich_reward_Rminus", "growth_Rplus", "p_growth_Rplus", "growth_Rminus", "p_growth_Rminus",
         "p_growth_cohort_mw", "p_growth_cohort_welch"]].round(4).to_string())
    print(ST[ST.set == "la"][["label", "enrich_reward_Rplus", "enrich_reward_Rminus", "enrich_unrewarded_Rminus", "growth_Rplus",
                              "p_growth_Rplus", "growth_Rminus", "p_growth_Rminus", "p_growth_cohort_mw", "p_growth_cohort_welch"]].round(4).to_string())


def figure(E, C, G, ST, Lc):
    fig = plt.figure(figsize=(7.4, 7.8))
    gs = fig.add_gridspec(3, 3, hspace=0.75, wspace=0.55, left=0.08, right=0.98, top=0.93, bottom=0.08)
    # A: what precedes reward (top states by R+ enrichment), both cohorts
    ax = fig.add_subplot(gs[0, :2])
    top = ST.dropna(subset=["enrich_reward_Rplus"]).sort_values("enrich_reward_Rplus", ascending=False).head(14)
    y = np.arange(len(top))
    ax.barh(y - 0.2, top.enrich_reward_Rplus, 0.38, color=L.COH_COL["R+"], lw=0, label="R+ before rewards")
    ax.barh(y + 0.2, top.enrich_reward_Rminus, 0.38, color=L.COH_COL["R-"], lw=0, label="R− before rewards (auditory)")
    ax.scatter(top.enrich_unrewarded_Rminus, y + 0.2, s=8, color="k", marker="|", label="R− before unrewarded licks", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(top.label, fontsize=4.2)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("log2 enrichment in the 0.5 s before the reward-triggering lick vs non-reward baseline", fontsize=5)
    ax.set_title("A  What precedes reward (states most enriched in R+)", fontsize=6, loc="left")
    ax.legend(frameon=False, fontsize=4.3, loc="lower right")
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    # B: whisking growth by cohort
    ax = fig.add_subplot(gs[0, 2])
    items = [("la", 2, "action whisking"), ("la", 1, "action still"), ("meta_side", 1, "meta side M1"), ("meta_top", 1, "meta top M1"),
             ("meta_top", 3, "meta top M3")]
    for i, (sn, s, lab) in enumerate(items):
        for h, cg in enumerate(("R+", "R-")):
            v = E[(E.set == sn) & (E.state == s) & (E.cohort == cg)].growth.dropna()
            x = i + (h - 0.5) * 0.38
            ax.scatter(x + np.random.default_rng(1).uniform(-0.08, 0.08, len(v)), v, s=2, color=L.COH_COL[cg], lw=0, alpha=0.6)
            ax.plot([x - 0.14, x + 0.14], [v.mean()] * 2, color="k", lw=0.9)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([lab for _, _, lab in items], rotation=45, ha="right", fontsize=4.5)
    ax.set_ylabel("growth, log2 late / early\n(non-reward, no-lick frames)", fontsize=5)
    ax.set_title("B  Do they become more frequent?", fontsize=6, loc="left")
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    # C: enrichment vs growth correlation per session
    ax = fig.add_subplot(gs[1, :2])
    scopes = ["all states"] + SETS
    for k_, ek in enumerate(("enrich_reward", "enrich_unrewarded")):
        for h, cg in enumerate(("R+", "R-")):
            for i, sc in enumerate(scopes):
                v = C[(C.scope == sc) & (C.enrichment == ek) & (C.cohort == cg)].rho.dropna()
                x = i * 5 + k_ * 2 + h
                ax.bar(x, v.mean(), 0.9, yerr=v.sem(), color=L.COH_COL[cg], alpha=1.0 if ek == "enrich_reward" else 0.4, lw=0,
                       error_kw=dict(lw=0.4))
                g = G[(G.scope == sc) & (G.enrichment == ek)]
                p = g[f"p_{'Rplus' if cg == 'R+' else 'Rminus'}"].iloc[0] if len(g) else np.nan
                if p < 0.05:
                    ax.text(x, (v.mean() + v.sem()) * 1.05 if v.mean() > 0 else (v.mean() - v.sem()) * 1.05 - 0.02, "*", ha="center", fontsize=6)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xticks([i * 5 + 1.5 for i in range(len(scopes))])
    ax.set_xticklabels([s.replace("_", " ") for s in scopes], fontsize=4.8)
    ax.set_ylabel("Spearman ρ across states\n(enrichment vs growth), mean ± s.e.m.", fontsize=5)
    ax.set_title("C  Do the states that precede reward grow? (solid: before rewards; faded: before unrewarded licks; * sign-flip p < 0.05)",
                 fontsize=5.5, loc="left")
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    # D: local reinforcement coefficients
    ax = fig.add_subplot(gs[1, 2])
    pm_ = Lc.groupby(["session_id", "cohort"]).beta_specific.mean().reset_index()
    for h, cg in enumerate(("R+", "R-")):
        v = pm_[pm_.cohort == cg].beta_specific
        ax.scatter(h + np.random.default_rng(2).uniform(-0.15, 0.15, len(v)), v, s=4, color=L.COH_COL[cg], lw=0)
        ax.plot([h - 0.25, h + 0.25], [v.mean()] * 2, color="k", lw=1)
    ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R+", "R−"])
    ax.set_ylabel("β: rewards preceded by the state\nin the last 5 min → its usage (per mouse)", fontsize=5)
    ax.set_title("D  Local reinforcement", fontsize=6, loc="left")
    ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    # E: example scatter enrichment vs growth, pooled per state means
    for j, cg in enumerate(("R+", "R-")):
        ax = fig.add_subplot(gs[2, j])
        g = ST.dropna(subset=[f"enrich_reward_{'Rplus' if cg == 'R+' else 'Rminus'}"])
        xk, yk = (f"enrich_reward_{'Rplus' if cg == 'R+' else 'Rminus'}", f"growth_{'Rplus' if cg == 'R+' else 'Rminus'}")
        ax.scatter(g[xk], g[yk], s=6, color=L.COH_COL[cg], lw=0)
        for r in g.sort_values(xk, ascending=False).head(4).itertuples():
            ax.text(getattr(r, xk), getattr(r, yk), " " + r.label.split(" (")[0], fontsize=3.8)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_xlabel("pre-reward enrichment (log2, mean over mice)", fontsize=5)
        ax.set_ylabel("growth (log2 late / early)", fontsize=5)
        ax.set_title(f"E  {cg}: every state (mean over mice)", fontsize=6, loc="left", color=L.COH_COL[cg])
        ax.tick_params(labelsize=4.5, length=1.5, pad=1)
    fig.suptitle("Reinforcement of behaviour by reward (Markowitz et al. 2023 logic): states before the reward-triggering lick, and "
                 "their usage away from reward over the session", fontsize=6, x=0.01, ha="left")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(L.FIGS / f"025_reinforcement.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    {"compute": compute, "stats": run_stats}[sys.argv[1]]()
