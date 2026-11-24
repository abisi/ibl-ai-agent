"""Pilot checks before scaling (v2; pages of the summary 008), each saved as a table:
1. DLC dropout per bodypart and view: fraction of frames below the keypoint's likelihood cutoff (dlc_helpers values;
   jaw at 0.5 for reporting although it is never gated), over the session (per minute) and around piezo licks.
2. Geometry removal (v3). kpms AR stage (kappa 1e8, 50 states) on spout-aligned mm keypoints with no per-session
   normalisation (raw), median centring (centre) and resting-pose subtraction (rest): share of frames in session-specific
   syllables. Full fits rest vs centre: share of syllables used (>= 0.5 %) by >= 6 of 8 mice. Session-identity decoding
   (chance 1/8; multinomial logistic regression, 5-fold CV over contiguous time blocks within each session):
   (a) from the pose (modelled keypoints of both views) in whisking-labelled frames: aligned (raw) vs rest-subtracted;
   (b) from syllable / state usage in 30 s windows, counted only in frames of one heuristic behaviour (whisking or still,
   i.e. matched behaviour): kpms rest vs kpms centre vs AR-HMM.
3. Held-out stability: model fitted without fold 0 (one R+ and one R- session) and applied to them: share of held-out
   frames in syllables used by the training sessions (>= 0.5 %), correlation of the held-out usage profile with the
   training-mean profile (vs training sessions among themselves, leave-one-out), and agreement (adjusted mutual
   information) of held-out labels between the fit without and with those sessions after matching syllables
   (Hungarian on the contingency).
4. Drift with time in session: meta-state usage (009) in 10 equal-time bins of the active block; slope per session.
5. Rough power: within-session change of meta-state usage (last third minus first third of the active block);
   effect size from the pilot (learners vs non-learners, R+ vs R-: Cohen's d with pooled SD; within-session change:
   mean / SD) -> sessions per group for 80 % power, two-sided alpha 0.05 (t-test, noncentral t).
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score as ami

import bm_lib as L
import bm_states as BS


def n_for_power(d, power=0.8, alpha=0.05, two_group=True):
    d = abs(d)
    if not np.isfinite(d) or d < 1e-6:
        return np.inf
    for n in range(2, 100000):
        df = 2 * n - 2 if two_group else n - 1
        nc = d * np.sqrt(n / 2) if two_group else d * np.sqrt(n)
        tc = stats.t.ppf(1 - alpha / 2, df)
        if 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc) >= power:
            return n
    return np.inf


def dropout_page(pdf, sids):
    D = pd.read_csv(L.TABLES / "002_dropout.csv")
    fig = plt.figure(figsize=(7.4, 9.8))
    for c, view in enumerate(L.VIEWS):
        d = D[D.view == view]
        P = d.pivot(index="bodypart", columns="session_id", values="frac_low")[sids]
        ax = fig.add_axes([0.1 + 0.48 * c, 0.72, 0.36, 0.22])
        im = ax.imshow(P.to_numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
        ax.set_yticks(range(len(P)))
        ax.set_yticklabels([f"{b} (<{d[d.bodypart == b].cutoff.iloc[0]:g})" for b in P.index], fontsize=5)
        ax.set_xticks(range(len(sids)))
        ax.set_xticklabels([s[:5] for s in sids], rotation=90, fontsize=5)
        ax.set_title(f"{view}: fraction of frames below cutoff", fontsize=6, loc="left")
        plt.colorbar(im, ax=ax, fraction=0.04)
        ax1 = fig.add_axes([0.1 + 0.48 * c, 0.42, 0.36, 0.2])
        ax2 = fig.add_axes([0.1 + 0.48 * c, 0.12, 0.36, 0.2])
        for i, b in enumerate(P.index):
            pm, pl = [], []
            for sid in sids:
                z = np.load(L.CACHE / "inputs" / f"{sid}_dropout.npz")
                pm.append(z[f"{view}|{b}|per_minute"])
                pl.append(z[f"{view}|{b}|peri_lick"])
                lag = z["lag_s"]
            n = min(len(x) for x in pm)
            col = plt.cm.tab10(i)
            ax1.plot(np.arange(n), np.mean([x[:n] for x in pm], 0), color=col, lw=0.8, label=b)
            ax2.plot(lag, np.nanmean(pl, 0), color=col, lw=0.8, label=b)
        ax1.set_xlabel("time in session (min)")
        ax1.set_ylabel("fraction below cutoff (mean of 8)")
        ax1.set_ylim(-0.02, 1.02)
        ax1.legend(frameon=False, fontsize=4.5, ncol=2)
        ax2.axvline(0, color="k", lw=0.5, ls="--")
        ax2.set_xlabel("time from piezo lick (s)")
        ax2.set_ylabel("fraction below cutoff (mean of 8)")
        ax2.set_ylim(-0.02, 1.02)
    fig.suptitle("P1  DLC dropout per bodypart and view (NWB likelihoods; tongue < 0.6 = tongue in the mouth, expected)", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def norm_rows():
    rows = []
    for view in L.VIEWS:
        for norm in ("raw", "centre", "rest"):
            f = BS.KR / "kpms" / view / norm / "sweep_ns50_k1e+08.json"
            if f.exists():
                rows.append(dict(view=view, **json.load(open(f))))
    return pd.DataFrame(rows)


def match_ami(a, b):
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    ua, ub = np.unique(a), np.unique(b)
    J = np.array([[np.sum((a == i) & (b == j)) for j in ub] for i in ua], float)
    r, c = linear_sum_assignment(-J)
    return float(ami(a[::20], b[::20])), float(J[r, c].sum() / J.sum())


def heldout_rows(sids):
    sel = pd.read_csv(L.TABLES / "001_selected_sessions.csv")
    rp, rm = list(sel[sel.reward_group == "R+"].session_id), list(sel[sel.reward_group == "R-"].session_id)
    ho = [rp[0], rm[0]]
    rows = []
    for m in ("kpms", "arhmm"):
        for view in L.VIEWS:
            try:
                Zh = {s: BS.load_states_native(s, m, view, heldout=True) for s in sids}
                Za = {s: BS.load_states_native(s, m, view) for s in sids}
            except KeyError:
                continue
            if any(v is None for v in Zh.values()):
                continue
            tr = [s for s in sids if s not in ho]
            K = max(int(z.max()) for z in Zh.values()) + 1
            U = {s: np.bincount(Zh[s], minlength=K) / len(Zh[s]) for s in sids}
            Utr = np.mean([U[s] for s in tr], 0)
            used = Utr >= 0.005
            loo = [np.corrcoef(U[s], np.mean([U[q] for q in tr if q != s], 0))[0, 1] for s in tr]
            for s in ho:
                a_, f_ = match_ami(Zh[s], Za[s])
                rows.append(dict(method=m, view=view, session_id=s, frac_frames_in_training_states=float(U[s][used].sum()),
                                 usage_corr_with_training=float(np.corrcoef(U[s], Utr)[0, 1]),
                                 usage_corr_training_loo_mean=float(np.mean(loo)), ami_heldout_vs_full_fit=a_,
                                 frac_frames_matched_heldout_vs_full=f_))
    return pd.DataFrame(rows)


def drift_power(sids, coh, learner):
    M = pd.read_csv(L.TABLES / "009_metastates.csv")
    meta = np.load(L.CACHE / "009_meta.npz")
    rows, prof = [], {}
    for sid in sids:
        sg = BS.session_grid(sid)
        act = sg["epoch"] != "other"
        t = sg["t"]
        ta, tb = t[act].min(), t[act].max()
        b = np.clip(((t - ta) / (tb - ta) * 10).astype(int), 0, 9)
        third = np.clip(((t - ta) / (tb - ta) * 3).astype(int), 0, 2)
        for m, v in [(m, v) for m in BS.METHODS for v in L.VIEWS]:
            z = meta[f"{sid}|{m}|{v}"]
            ok = act & (z >= 0)
            for k in M.meta:
                u = np.array([np.mean(z[ok & (b == i)] == k) for i in range(10)])
                prof[(sid, m, v, k)] = u
                d3 = np.mean(z[ok & (third == 2)] == k) - np.mean(z[ok & (third == 0)] == k)
                rows.append(dict(session_id=sid, cohort=coh[sid], learner=learner[sid], method=m, view=v, meta=k,
                                 slope_per_session=float(np.polyfit(np.arange(10) / 9, u, 1)[0]), delta_last_first_third=float(d3)))
    R = pd.DataFrame(rows)
    pw = []
    for (m, v, k), g in R.groupby(["method", "view", "meta"]):
        x = g.delta_last_first_third
        for name, a, b in (("learner vs non-learner", g[g.learner == "learner"].delta_last_first_third,
                            g[g.learner == "non-learner"].delta_last_first_third),
                           ("R+ vs R-", g[g.cohort == "R+"].delta_last_first_third, g[g.cohort == "R-"].delta_last_first_third)):
            sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
            d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
            pw.append(dict(method=m, view=v, meta=k, contrast=name, n_a=len(a), n_b=len(b), cohen_d=d,
                           n_per_group_80pct=n_for_power(d)))
        d1 = x.mean() / x.std()
        pw.append(dict(method=m, view=v, meta=k, contrast="within-session change (all 8)", n_a=len(x), n_b=0, cohen_d=d1,
                       n_per_group_80pct=n_for_power(d1, two_group=False)))
    return R, pd.DataFrame(pw), prof, M


def block_cv_logreg(X, y, blk, n_splits=5):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    acc = []
    for f in range(n_splits):
        tr, te = blk != f, blk == f
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        acc.append(np.mean(m.predict(sc.transform(X[te])) == y[te]))
    return float(np.mean(acc))


def identity_decoding(sids):
    rows = []
    rng = np.random.default_rng(0)
    # (a) pose in whisking frames
    for kind in ("raw", "rest"):
        X, y, blk = [], [], []
        for i, sid in enumerate(sids):
            sg = BS.session_grid(sid)
            lab = sg["labels"]
            parts = []
            for view in L.VIEWS:
                k = np.load(L.CACHE / "inputs" / f"{sid}_{view}_keypoints.npz")
                xy = k["xy"].astype(float) - (k["rest"] if kind == "rest" else 0)
                parts.append(BS.to_grid(np.arange(len(xy)), k["t"], sg["t"]))
                parts[-1] = (parts[-1], xy.reshape(len(xy), -1))
            ok = (lab == 2) & np.all([p[0] >= 0 for p in parts], 0)
            idx = np.flatnonzero(ok)
            idx = np.sort(rng.choice(idx, min(3000, len(idx)), replace=False))
            X.append(np.hstack([p[1][p[0][idx]] for p in parts]))
            y += [i] * len(idx)
            blk.append(np.minimum((np.searchsorted(np.flatnonzero(ok), idx) * 5) // ok.sum(), 4))
        rows.append(dict(test="pose in whisking frames", model=f"keypoints ({kind})",
                         accuracy=block_cv_logreg(np.vstack(X), np.array(y), np.concatenate(blk))))
    # (b) usage in matched behaviour
    sel = BS.selected()
    models = [("kpms", "rest", False), ("kpms", "centre", True), ("arhmm", "", False)]
    for method, norm, alt in models:
        for view in L.VIEWS:
            for beh, code in (("whisking", 2), ("still", 1)):
                X, y, blk = [], [], []
                K = 0
                Zs = {}
                for sid in sids:
                    if method == "kpms" and alt:
                        m = sel.get("kpms_centre", {}).get(view)
                        if not m:
                            break
                        f = BS.KR / "kpms" / view / "centre" / m["tags"]["0"] / f"{sid}.npz"
                        z = np.load(f)["z"].astype(int) if f.exists() else None
                        z = None if z is None else BS.to_grid(z, BS.view_t(sid, view)[:len(z)], BS.session_grid(sid)["t"])
                    else:
                        z = BS.load_states(sid, method, view)
                    if z is None:
                        break
                    Zs[sid] = z
                    K = max(K, int(z.max()) + 1)
                if len(Zs) < len(sids):
                    continue
                for i, sid in enumerate(sids):
                    sg = BS.session_grid(sid)
                    z, lab, t = Zs[sid], sg["labels"], sg["t"]
                    w = ((t - t[0]) // 30).astype(int)
                    for wi in np.unique(w):
                        q = (w == wi) & (lab == code) & (z >= 0)
                        if q.sum() < 200:
                            continue
                        X.append(np.bincount(z[q], minlength=K) / q.sum())
                        y.append(i)
                    nwin = sum(1 for yy in y if yy == i)
                    blk += list(np.minimum(np.arange(nwin) * 5 // max(nwin, 1), 4))
                if len(X) > 40:
                    rows.append(dict(test=f"usage in {beh} frames (30 s windows)", model=f"{method} {norm or ''} {view}".strip(),
                                     accuracy=block_cv_logreg(np.array(X), np.array(y), np.array(blk))))
    return pd.DataFrame(rows)


def shared_rows(sids):
    rows = []
    sel = BS.selected()
    for method, norm in (("kpms", "rest"), ("kpms", "centre"), ("arhmm", "")):
        for view in L.VIEWS:
            m = sel.get("kpms_centre" if norm == "centre" else method, {}).get(view)
            if not m:
                continue
            base = BS.KR / "kpms" / view / (norm or m.get("norm", "")) if method == "kpms" else BS.KR / "arhmm" / view
            U = []
            for sid in sids:
                f = base / m["tags"]["0"] / f"{sid}.npz"
                if not f.exists():
                    break
                z = np.load(f)["z"].astype(int)
                U.append(np.bincount(z, minlength=200)[:200] / len(z))
            if len(U) < len(sids):
                continue
            U = np.array(U)
            used = U.max(0) >= 0.005
            nm = (U[:, used] >= 0.005).sum(0)
            tot = U[:, used].sum(0)
            rows.append(dict(method=method, norm=norm, view=view, n_syllables=int(used.sum()),
                             frac_syllables_ge6_mice=float(np.mean(nm >= 6)), frac_syllables_lt3_mice=float(np.mean(nm < 3)),
                             frac_frames_in_ge6_mice_syllables=float(tot[nm >= 6].sum() / tot.sum())))
    return pd.DataFrame(rows)


def checks_pages(pdf, sids, coh):
    sel = pd.read_csv(L.TABLES / "001_selected_sessions.csv")
    learner = dict(zip(sel.session_id, sel.learner))
    dropout_page(pdf, sids)
    N = norm_rows()
    H = heldout_rows(sids)
    SH = shared_rows(sids)
    ID = identity_decoding(sids)
    SH.to_csv(L.TABLES / "010_shared_syllables.csv", index=False)
    ID.to_csv(L.TABLES / "010_identity_decoding.csv", index=False)
    R, PW, prof, M = drift_power(sids, coh, learner)
    N.to_csv(L.TABLES / "010_normalisation_comparison.csv", index=False)
    H.to_csv(L.TABLES / "010_heldout_stability.csv", index=False)
    R.to_csv(L.TABLES / "010_drift.csv", index=False)
    PW.to_csv(L.TABLES / "010_power.csv", index=False)
    fig = plt.figure(figsize=(7.4, 9.8))
    ax = fig.add_axes([0.08, 0.76, 0.38, 0.18])
    if len(N):
        for i, view in enumerate(L.VIEWS):
            d = N[N.view == view].set_index("norm").reindex(["raw", "centre", "rest"])
            ax.bar(np.arange(3) + (i - 0.5) * 0.38, d.frac_frames_session_specific, 0.38, color=["#444444", "#1f77b4"][i], lw=0,
                   label=f"{view} view")
        ax.set_xticks(range(3))
        ax.set_xticklabels(["aligned only", "median centring", "rest subtraction"])
        ax.set_ylabel("frames in session-specific syllables")
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=5)
    ax.set_title("P2  Geometry removal (kpms AR stage, kappa 1e8, 50 states)", fontsize=6, loc="left")
    ax = fig.add_axes([0.58, 0.76, 0.38, 0.18])
    for i, (m, v) in enumerate([(m, v) for m in BS.METHODS for v in L.VIEWS]):
        for k in M.meta:
            for sid in sids:
                u = prof[(sid, m, v, k)]
                if m == "kpms" and v == "side":
                    ax.plot(np.arange(10) / 9, u, color=plt.cm.tab10(k), lw=0.6,
                            ls="-" if learner[sid] == "learner" else "--", alpha=0.8)
    ax.set_xlabel("time in active block (0 = start, 1 = end)")
    ax.set_ylabel("meta-state usage (kpms side)")
    ax.set_title("P4  Drift (solid learners, dashed non-learners;\ncolour = meta-state M0 blue, M1 orange, M2 green, M3 red)",
                 fontsize=6, loc="left")
    lines = ["P2b Syllables / states shared across mice (usage >= 0.5 % per session):"]
    for q in SH.itertuples():
        lines.append(f"  {q.method:<5} {q.norm:<6} {q.view:<5} {q.n_syllables:>3} used; >= 6 of 8 mice: {100 * q.frac_syllables_ge6_mice:.0f}% of syllables "
                     f"({100 * q.frac_frames_in_ge6_mice_syllables:.0f}% of frames); < 3 mice (flagged): {100 * q.frac_syllables_lt3_mice:.0f}%")
    lines += ["", "P2c Session-identity decoding (chance 0.125; block CV):"]
    for q in ID.itertuples():
        lines.append(f"  {q.test:<38} {q.model:<20} accuracy {q.accuracy:.2f}")
    lines += [""]
    lines += ["P3  Leave-mouse-out recurrence (fit without " + ", ".join(s[:5] for s in H.session_id.unique()) + "; applied to them):"]
    for r in H.itertuples():
        lines.append(f"  {r.method:<6} {r.view:<5} {r.session_id[:5]}  frames in training syllables {r.frac_frames_in_training_states:.2f}  "
                     f"usage r with training {r.usage_corr_with_training:.2f} (training LOO {r.usage_corr_training_loo_mean:.2f})  "
                     f"AMI vs full fit {r.ami_heldout_vs_full_fit:.2f}, matched frames {r.frac_frames_matched_heldout_vs_full:.2f}")
    g = R.groupby(["method", "view", "meta", "learner"]).slope_per_session.mean().unstack()
    lines += ["", "P4  Drift: mean slope of meta-state usage over the active block (usage change start -> end), by learner group:"]
    for (m, v, k), r in g.iterrows():
        lines.append(f"  {m:<16} {v:<5} {M.name[k]:<38} learners {r.get('learner', np.nan):+.3f}   non-learners {r.get('non-learner', np.nan):+.3f}")
    best = PW[PW.contrast != "within-session change (all 8)"].sort_values("n_per_group_80pct").head(6)
    w = PW[PW.contrast == "within-session change (all 8)"].sort_values("n_per_group_80pct").head(4)
    lines += ["", "P5  Rough power (pilot effect sizes are noisy and biased upward with n = 4 vs 4; for planning only):",
              "  smallest required n per group (80 % power, alpha 0.05, two-sided) for change of usage last - first third:"]
    for r in pd.concat([best, w]).itertuples():
        lines.append(f"  {r.contrast:<30} {r.method:<16} {r.view:<5} {M.name[r.meta]:<34} d = {r.cohen_d:+.2f}  n = {r.n_per_group_80pct}")
    med = PW.groupby("contrast").cohen_d.apply(lambda x: np.nanmedian(np.abs(x)))
    lines.append("  median |d| over all metric x method x view: " + "; ".join(f"{k} {v:.2f} (n {n_for_power(v, two_group='within' not in k)})"
                                                                         for k, v in med.items()))
    ax = fig.add_axes([0.05, 0.02, 0.92, 0.66])
    ax.axis("off")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=4.4)
    fig.suptitle("Pilot checks before scaling", fontsize=8)
    pdf.savefig(fig)
    plt.close(fig)
