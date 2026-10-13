"""001 -- Reliability of the delay between jaw-opening onset and the first lick (user 2026-10-13: "estimate the reliability of the timing
between jaw onset and lick time, across sessions and individuals ... to infer a jaw opening time for sessions without video data").
Events and onsets: ssl-prelick-convergence 118 (events.csv; see ../question.md): WH, AH (corrected first lick) and SL (piezo bout onset);
jaw onset = first frame with jaw distance > 7 x baseline SD (WH / AH: baseline 1 s before trial start, search from stimulus + 50 ms to
the lick; SL: baseline [-1, -0.5] s before the lick, search 0.5 s before the lick); events without a crossing were dropped there.
lead = lick - jaw onset (ms). Censoring: for WH / AH the onset cannot precede stimulus + 50 ms, so lead <= RT - 50 ms (flagged: onset
within one frame of that bound); SL leads are bounded by 500 ms.
Population: sessions of 105 (learners rule for day 0, all expert sessions); mouse = mouse_id; setup from 118 sessions.csv.
Analyses
  1  distributions per class, session and mouse (median, IQR)
  2  variance components per class: lead ~ 1 + (1 | mouse) + (1 | session within mouse) (statsmodels MixedLM, REML; SL subsampled to
     <= 60 events per session); shares of mouse, session and trial (residual) variance; ICCs
  3  dependence: lead vs reaction time (WH / AH; pooled Spearman and per-session slopes), vs time in session, vs stage and setup
  4  inference without video, leave-one-session-out per class: global median lead; same-mouse median from the mouse's other sessions
     (global if none); global linear model lead ~ RT (WH / AH); within-session oracle (leave-one-event-out median of the session, the
     best a constant per-session offset can do). Error = predicted - actual jaw onset (ms); per event |error| and per session bias
     (median error)
Outputs: combined_results_ks4/ssl-jaw-lick-latency/{tables/*.csv, figures/jaw_lick_latency.{png,pdf,svg}}
SSL_JAW_POP=all (2026-10-13): every session with usable jaw video, no learners rule, outputs in all_sessions/; per-session estimates
(tables/session_estimates.csv) and a within-session leave-one-out lead ~ RT model (each session on its own).
Run (haas): [SSL_JAW_POP=all] PRELICK_REF=sl PRELICK_UNITS=stable python .../001_jaw_lick_latency.py
"""
import importlib
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
PRE = pathlib.Path(__file__).resolve().parents[2] / "ssl-prelick-convergence" / "exploratory-analyses"
sys.path.insert(0, str(PRE))
m113 = importlib.import_module("113_dlc_passive_pre")
m51, m105 = m113.m51, m113.m105
m61 = importlib.import_module("061_roc_prelick_learners")
SRC = m51.HOME / "across_days" / f"{m51.REF}{m51.USUF}_jaw" / "jaw_onsets"
OUT0 = m51.RES / "ssl-jaw-lick-latency"
import os  # noqa: E402
POP = os.environ.get("SSL_JAW_POP", "learners")          # 2026-10-13: "all" = every session with usable jaw video, no learners rule
OUT = OUT0 if POP == "learners" else OUT0 / "all_sessions"
CLS = ("WH", "AH", "SL")
COLC = {"WH": "#f7b519", "AH": "#2c2cdb", "SL": "#262626"}
COH = {"R+": "#00B400", "R-": "#C800C8"}
FRAME = 0.005


def load():
    E = pd.read_csv(SRC / "events.csv")
    E["cls"] = E.cls.replace({"FA": "SL"})
    E = E.dropna(subset=["jaw_onset"]).copy()
    E["lead"] = (E.lick_time_used - E.jaw_onset) * 1e3
    E["rt"] = np.where(E.cls != "SL", (E.lick_time_used - E.start_time) * 1e3, np.nan)
    E["censored"] = (E.cls != "SL") & ((E.jaw_onset - E.start_time) <= 0.05 + FRAME)
    meta = pd.read_csv(m105.OUT / "sessions.csv")[["session_id", "mouse_id", "cohort", "stage"]]
    S = pd.read_csv(SRC / "sessions.csv")[["session_id", "setup"]]
    E = E.merge(meta, on="session_id").merge(S, on="session_id", how="left")
    if POP == "learners":
        E = m61.learner_filter(E)
    t0 = E.groupby("session_id").start_time.transform("min"); t1 = E.groupby("session_id").start_time.transform("max")
    E["tau"] = (E.start_time - t0) / (t1 - t0).clip(lower=1e-9)
    return E


def var_components(E, rng):
    import statsmodels.formula.api as smf
    rows = []
    for c in CLS:
        d = E[E.cls == c]
        if c == "SL":
            d = pd.concat([g.sample(min(len(g), 60), random_state=0) for _, g in d.groupby("session_id")])
        try:
            md = smf.mixedlm("lead ~ 1", d, groups=d["mouse_id"], re_formula="1", vc_formula={"session": "0 + C(session_id)"}).fit(reml=True)
            vm, vs, ve = float(md.cov_re.iloc[0, 0]), float(md.vcomp[0]), float(md.scale)
        except Exception as e:  # noqa: BLE001
            print("mixedlm failed", c, e); continue
        tot = vm + vs + ve
        rows.append(dict(cls=c, n_events=len(d), n_sessions=d.session_id.nunique(), n_mice=d.mouse_id.nunique(), mean_ms=md.params["Intercept"],
                         sd_mouse=np.sqrt(vm), sd_session=np.sqrt(vs), sd_trial=np.sqrt(ve), share_mouse=vm / tot, share_session=vs / tot,
                         share_trial=ve / tot, icc_mouse=vm / tot, icc_session_total=(vm + vs) / tot))
    return pd.DataFrame(rows)


def loso(E):
    """leave-one-session-out predictions of lead (ms); error = predicted onset - actual onset = lead - predicted lead"""
    out = []
    for c in CLS:
        d = E[E.cls == c]
        for sid, g in d.groupby("session_id"):
            o = d[d.session_id != sid]; om = o[o.mouse_id == g.mouse_id.iloc[0]]
            pred = {"global": np.full(len(g), o.lead.median()),
                    "same mouse": np.full(len(g), om.lead.median() if len(om) >= 10 else o.lead.median())}
            if c != "SL":
                ok = o.rt.notna(); b, a = np.polyfit(o.rt[ok], o.lead[ok], 1)
                pred["RT model"] = a + b * g.rt.to_numpy()
                pred["stimulus-locked"] = g.rt.to_numpy() - (o.jaw_onset - o.start_time).median() * 1e3   # onset = stimulus + typical latency
            x = g.lead.to_numpy(); n = len(x)
            pred["session oracle"] = np.array([np.median(np.delete(x, i)) for i in range(n)]) if n > 1 else np.full(n, np.nan)
            if c != "SL" and n >= 10:                                   # each session on its own: leave-one-out lead ~ RT within the session
                rt = g.rt.to_numpy(); pr = np.full(n, np.nan)
                for i in range(n):
                    m_ = np.arange(n) != i; b_, a_ = np.polyfit(rt[m_], x[m_], 1); pr[i] = a_ + b_ * rt[i]
                pred["within-session RT model"] = pr
            for k, p in pred.items():
                out.append(pd.DataFrame(dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cls=c, scheme=k, err=x - p)))
    return pd.concat(out, ignore_index=True)


def example_traces(sid, E):
    """jaw opening (mm, minus the session median) around each lick of one session"""
    info, _, series = m113.load_series(sid)
    t, v = series["jaw opening"]
    v = v - np.nanmedian(v)
    grid = np.round(np.arange(-0.4, 0.15 + 1e-9, 0.005), 4)
    d = E[(E.session_id == sid) & E.cls.isin(["WH", "AH"])].sort_values("lead")
    G = m113.to_grid(t, v, d.lick_time_used.to_numpy(), None, grid=grid)
    return grid * 1e3, G, d


def figure(E, VC, P, Ssum, ex_sid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 5.4, "axes.labelsize": 5.4, "xtick.labelsize": 4.8, "ytick.labelsize": 4.8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none", "pdf.fonttype": 42, "font.family": "Arial"})
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(2, 4, wspace=0.55, hspace=0.75, left=0.06, right=0.985, top=0.9, bottom=0.1)
    L = []
    # A example: jaw traces aligned to the lick, sorted by lead, onset dots
    tg, G, d = example_traces(ex_sid, E)
    ax = fig.add_subplot(gs[0, 0]); L.append(ax)
    im = ax.imshow(G, aspect="auto", cmap="magma", extent=[tg[0], tg[-1], len(G), 0], vmin=0, vmax=np.nanpercentile(G, 98), interpolation="none")
    ax.plot(-d.lead.to_numpy(), np.arange(len(d)) + 0.5, ".", ms=1.2, color="#7FDBFF")
    ax.axvline(0, color="w", lw=0.6, ls=":")
    ax.set_xlabel("time from lick (ms)"); ax.set_ylabel("WH / AH trials (sorted by lead)")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.ax.tick_params(labelsize=4.2)
    ax.set_title(f"Example ({ex_sid.split('_')[0]}): jaw opening (mm)\ncyan: detected jaw onset", fontsize=5.0)
    # B example: mean trace + lead distribution per class
    ax = fig.add_subplot(gs[0, 1]); L.append(ax)
    for c in CLS:
        x = E[E.cls == c].lead
        ax.hist(x, bins=np.arange(0, 505, 10), histtype="step", density=True, color=COLC[c], lw=0.9, label=f"{c} (median {x.median():.0f} ms)")
    ax.set_xlabel("lead: lick − jaw onset (ms)"); ax.set_ylabel("density"); ax.legend(frameon=False, fontsize=4.2)
    ax.set_title("All events, all sessions", fontsize=5.4)
    # C per-session medians grouped by mouse
    ax = fig.add_subplot(gs[0, 2:4]); L.append(ax)
    S = E[E.cls != "SL"].groupby(["mouse_id", "session_id", "cohort", "stage"]).lead.agg(med="median", q1=lambda x: x.quantile(0.25),
                                                                                    q3=lambda x: x.quantile(0.75)).reset_index()
    order = S.groupby("mouse_id").med.median().sort_values().index
    xi, xt, xl = 0, [], []
    for k, m in enumerate(order):
        g = S[S.mouse_id == m].sort_values("stage", ascending=False)
        if k % 2:
            ax.axvspan(xi - 0.5, xi + len(g) - 0.5, color="0.95", lw=0, zorder=0)
        for _, r in g.iterrows():
            ax.plot([xi, xi], [r.q1, r.q3], color=COH[r.cohort], lw=0.7)
            ax.plot(xi, r.med, "o", ms=2.2, color=COH[r.cohort], mfc=COH[r.cohort] if r.stage == "expert" else "white", mew=0.6)
            xi += 1
    ax.axhline(E[E.cls != "SL"].lead.median(), color="0.4", lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(-1, xi); ax.set_xticks([]); ax.set_xlabel(f"sessions, grouped by mouse (n = {S.session_id.nunique()} sessions, {S.mouse_id.nunique()} mice)")
    ax.set_ylabel("lead WH + AH (ms; median, IQR)")
    ax.set_title("Per-session lead (open: day 0, filled: expert; green R+, magenta R−)", fontsize=5.4)
    # D variance components
    ax = fig.add_subplot(gs[1, 0]); L.append(ax)
    bottom = np.zeros(len(VC))
    for k, col in (("share_mouse", "#4C72B0"), ("share_session", "#8FB0D8"), ("share_trial", "0.8")):
        ax.bar(range(len(VC)), VC[k], bottom=bottom, color=col, width=0.6, label=k.replace("share_", ""))
        bottom += VC[k].to_numpy()
    ax.set_xticks(range(len(VC)), VC.cls); ax.set_ylim(0, 1); ax.set_ylabel("share of lead variance")
    ax.legend(frameon=False, fontsize=4.2, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, handlelength=1.0, columnspacing=0.6)
    for i, r in VC.reset_index().iterrows():
        ax.text(i, 1.01, f"SD trial\n{r.sd_trial:.0f} ms", ha="center", va="bottom", fontsize=3.8)
    ax.set_title("Variance components", fontsize=5.4, pad=16)
    # E lead vs RT
    ax = fig.add_subplot(gs[1, 1]); L.append(ax)
    d = E[(E.cls != "SL") & ~E.censored & E.rt.between(0, 1000)]
    ax.hexbin(d.rt, d.lead, gridsize=35, cmap="Greys", mincnt=1, linewidths=0, bins="log")
    rs, _ = stats.spearmanr(d.rt, d.lead)
    xx = np.linspace(d.rt.quantile(0.01), d.rt.quantile(0.99), 50); b, a = np.polyfit(d.rt, d.lead, 1)
    ax.plot(xx, a + b * xx, color="#d94801", lw=0.9)
    ax.plot(xx, xx - 50, color="0.4", lw=0.6, ls=":"); ax.text(xx[-1], xx[-1] - 50, " RT − 50", fontsize=3.8, va="center", color="0.4")
    ax.set_xlabel("reaction time (ms)"); ax.set_ylabel("lead (ms)")
    ax.set_title(f"Lead grows with RT (Spearman {rs:.2f})", fontsize=5.4)
    # F within-mouse stability: day 0 vs expert session medians
    ax = fig.add_subplot(gs[1, 2]); L.append(ax)
    M = S.groupby(["mouse_id", "stage", "cohort"]).med.median().unstack("stage").dropna().reset_index()
    for c in COH:
        q = M[M.cohort == c]; ax.plot(q.learning, q.expert, "o", ms=2.6, color=COH[c], label=c.replace("-", "−"))
    lo, hi = np.nanmin(M[["learning", "expert"]].to_numpy()) - 10, np.nanmax(M[["learning", "expert"]].to_numpy()) + 10
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.5, ls=(0, (2, 2))); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_box_aspect(1)
    r_, p_ = stats.spearmanr(M.learning, M.expert) if len(M) > 3 else (np.nan, np.nan)
    ax.set_xlabel("day 0 median lead (ms)"); ax.set_ylabel("expert median lead (ms)"); ax.legend(frameon=False, fontsize=4.2)
    ax.set_title(f"Same mouse across days (n = {len(M)})\nSpearman {r_:.2f}", fontsize=5.4)
    # G inference error
    ax = fig.add_subplot(gs[1, 3]); L.append(ax)
    sch = ["global", "same mouse", "stimulus-locked", "RT model", "session oracle", "within-session RT model"]
    colS = {"global": "#999999", "same mouse": "#4C72B0", "stimulus-locked": "#2CA02C", "RT model": "#d94801", "session oracle": "k",
            "within-session RT model": "#9467BD"}
    q = P[P.cls != "SL"]
    for k in sch:
        e = np.sort(np.abs(q[q.scheme == k].err.dropna().to_numpy()))
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=colS[k], lw=0.9, label=f"{k} (median {np.median(e):.0f} ms)")
    for v in (25, 50):
        ax.axvline(v, color="0.75", lw=0.5, ls=":")
    ax.set_xlabel("|predicted − actual jaw onset| (ms)"); ax.set_ylabel("cumulative fraction (WH + AH)")
    ax.legend(frameon=False, fontsize=3.6, loc="lower right", bbox_to_anchor=(1.02, 0.0), handlelength=1.0, labelspacing=0.2)
    ax.set_xlim(0, 250)
    ax.set_title("Inferring jaw onset from the lick\n(leave one session out)", fontsize=5.4)
    for ax, l in zip(L, "abcdefg"):
        ax.text(-0.3 if l != "c" else -0.12, 1.12, l, transform=ax.transAxes, fontsize=8, weight="bold")
    fig.suptitle(f"How stable is the delay between jaw-opening onset and the first lick? ({'all sessions' if POP == 'all' else 'learners'})", x=0.01, y=0.995, ha="left", fontsize=7, weight="bold")
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / "figures" / f"jaw_lick_latency.{ext}", dpi=300)


def main():
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    E = load()
    E.to_csv(OUT / "tables" / "events.csv", index=False)
    pd.set_option("display.width", 220)
    print("events per class:", E.cls.value_counts().to_dict(), "| sessions", E.session_id.nunique(), "| mice", E.mouse_id.nunique())
    print("censored (WH/AH onset at stimulus + 50 ms):", round(E[E.cls != "SL"].censored.mean(), 3))
    D = E.groupby("cls").lead.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1); print(D); D.to_csv(OUT / "tables" / "lead_by_class.csv")
    VC = var_components(E, rng); VC = VC.set_index("cls").loc[[c for c in CLS if c in VC.cls.values]].reset_index()
    print(VC.round(3).to_string(index=False)); VC.to_csv(OUT / "tables" / "variance_components.csv", index=False)
    # dependence
    dep = []
    for c in ("WH", "AH"):
        d = E[(E.cls == c) & ~E.censored]
        dep.append(dict(test=f"{c} lead vs RT (pooled Spearman)", r=stats.spearmanr(d.rt, d.lead)[0], p=stats.spearmanr(d.rt, d.lead)[1]))
        sl = d.groupby("session_id").apply(lambda g: np.polyfit(g.rt, g.lead, 1)[0] if len(g) >= 10 else np.nan).dropna()
        dep.append(dict(test=f"{c} per-session slope lead/RT (median, Wilcoxon vs 0)", r=sl.median(), p=stats.wilcoxon(sl).pvalue))
    for c in CLS:
        d = E[E.cls == c]
        tt = d.groupby("session_id").apply(lambda g: np.polyfit(g.tau, g.lead, 1)[0] if len(g) >= 10 else np.nan).dropna()
        dep.append(dict(test=f"{c} per-session slope lead vs time in session (ms / session; median, Wilcoxon)", r=tt.median(), p=stats.wilcoxon(tt).pvalue))
        sm = d.groupby(["session_id", "stage", "setup", "cohort"]).lead.median().reset_index()
        a, b = sm[sm.stage == "learning"].lead, sm[sm.stage == "expert"].lead
        dep.append(dict(test=f"{c} session median, day 0 vs expert (MWU)", r=b.median() - a.median(), p=stats.mannwhitneyu(a, b).pvalue))
        a, b = sm[sm.setup == 1].lead, sm[sm.setup == 2].lead
        if len(b) >= 3:
            dep.append(dict(test=f"{c} session median, setup 1 vs 2 (MWU)", r=b.median() - a.median(), p=stats.mannwhitneyu(a, b).pvalue))
    for c in ("WH", "AH"):
        d = E[(E.cls == c) & ~E.censored].assign(lat=lambda q: (q.jaw_onset - q.start_time) * 1e3)
        sd = d.groupby("session_id").agg(sd_lead=("lead", "std"), sd_lat=("lat", "std"), sd_rt=("rt", "std")).dropna()
        dep.append(dict(test=f"{c} within-session SD: jaw onset from stimulus (ms, median)", r=sd.sd_lat.median(), p=np.nan))
        dep.append(dict(test=f"{c} within-session SD: lead = lick - jaw onset (ms, median)", r=sd.sd_lead.median(), p=np.nan))
        dep.append(dict(test=f"{c} within-session SD: RT (ms, median)", r=sd.sd_rt.median(), p=np.nan))
        dep.append(dict(test=f"{c} fraction of sessions SD(onset from stimulus) < SD(lead); Wilcoxon", r=(sd.sd_lat < sd.sd_lead).mean(),
                        p=stats.wilcoxon(sd.sd_lat, sd.sd_lead).pvalue))
    DEP = pd.DataFrame(dep); print(DEP.round(4).to_string(index=False)); DEP.to_csv(OUT / "tables" / "dependence.csv", index=False)
    # inference
    P = loso(E); P.to_csv(OUT / "tables" / "loso_errors.csv", index=False)
    summ = P.groupby(["cls", "scheme"]).err.agg(median_abs=lambda x: np.nanmedian(np.abs(x)), within25=lambda x: np.nanmean(np.abs(x) <= 25),
                                                 within50=lambda x: np.nanmean(np.abs(x) <= 50), bias=lambda x: np.nanmedian(x)).reset_index()
    sb = P.groupby(["cls", "scheme", "session_id"]).err.median().groupby(["cls", "scheme"]).agg(session_bias_sd="std",
                                                                                              session_bias_abs_median=lambda x: np.nanmedian(np.abs(x)))
    summ = summ.merge(sb.reset_index(), on=["cls", "scheme"]); print(summ.round(3).to_string(index=False))
    summ.to_csv(OUT / "tables" / "inference_summary.csv", index=False)
    # example: an R+ expert session with many cued licks, near the median lead
    S = E[E.cls != "SL"].groupby(["session_id", "cohort", "stage"]).lead.agg(["median", "size"]).reset_index()
    c = S[(S.cohort == "R+") & (S.stage == "expert") & (S["size"] >= 60)]
    ex = c.iloc[(c["median"] - E[E.cls != "SL"].lead.median()).abs().argsort().iloc[0]].session_id
    # each session on its own (2026-10-13): per-session estimates and within-session predictability
    rows = []
    for (sid, c), g in E.groupby(["session_id", "cls"]):
        r = dict(session_id=sid, mouse_id=g.mouse_id.iloc[0], cohort=g.cohort.iloc[0], stage=g.stage.iloc[0], setup=g.setup.iloc[0], cls=c,
                 n=len(g), median=g.lead.median(), iqr=g.lead.quantile(0.75) - g.lead.quantile(0.25), sd=g.lead.std(),
                 mad=np.median(np.abs(g.lead - g.lead.median())), frac_censored=g.censored.mean())
        if c != "SL" and len(g) >= 10:
            r["rt_slope"], r["rt_intercept"] = np.polyfit(g.rt, g.lead, 1); r["rt_spearman"] = stats.spearmanr(g.rt, g.lead)[0]
            lat = (g.jaw_onset - g.start_time) * 1e3; r["sd_onset_from_stim"] = lat.std()
        for k in ("session oracle", "within-session RT model"):
            e = P[(P.session_id == sid) & (P.cls == c) & (P.scheme == k)].err.dropna()
            if len(e):
                r[f"abs_err_median {k}"], r[f"within50 {k}"] = np.median(np.abs(e)), np.mean(np.abs(e) <= 50)
        rows.append(r)
    SE = pd.DataFrame(rows); SE.to_csv(OUT / "tables" / "session_estimates.csv", index=False)
    q = SE.groupby("cls")[[k for k in SE.columns if k not in ("session_id", "mouse_id", "cohort", "stage", "setup", "cls")]]
    print("per-session estimates (median [IQR] across sessions):")
    print(q.median().round(2).T.to_string()); print(q.quantile(0.25).round(2).T.add_suffix(" q25").to_string()); print(q.quantile(0.75).round(2).T.add_suffix(" q75").to_string())
    figure(E, VC, P, summ, ex)
    print("ALL DONE", OUT / "figures" / "jaw_lick_latency.png")


if __name__ == "__main__":
    main()
