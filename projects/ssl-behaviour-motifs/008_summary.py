"""008 -- cross-session summary page (haas): method / view agreement, kpms seed stability, kappa sweeps, lightning-action
held-out F1, exploratory R+ vs R- usage (all frames and no-lick frames).

- Agreement: adjusted mutual information (sklearn) between every pair of the 6 sequences (method x view) on the side
  50 Hz grid, frames in the active block (epoch != other) where both are defined; mean over sessions.
- Seed stability: AMI between kpms seeds (same view), mean over sessions and seed pairs; n syllables > 1 % usage.
- R+ vs R-: usage of each state (kpms seed 0 and AR-HMM: the 8 most used states over all sessions; lightning-action:
  4 classes), per session (= mouse, one day-0 session each), active block; all frames vs no-lick frames (no piezo lick
  within +-1 s). Mann-Whitney U + Welch t, uncorrected, EXPLORATORY, n = 4 vs 4 (underpowered).
Output: reports/summary_behaviour_motifs.pdf, tables/008_*.csv, provenance/008.json
"""
import itertools
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.metrics import adjusted_mutual_info_score as ami

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bm_lib as L  # noqa: E402
import bm_states as BS  # noqa: E402
import bm_atlas as BA  # noqa: E402
import bm_checks as BC  # noqa: E402

plt.rcParams.update({"font.size": 6, "font.family": "Arial", "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})
ROWS = [(m, v) for m in BS.METHODS for v in L.VIEWS]
LABEL = {(m, v): f"{ {'kpms': 'kpms', 'arhmm': 'AR-HMM', 'lightning_action': 'l-action'}[m]} {v}" for m, v in ROWS}


def subsample_ami(a, b, step=20):
    ok = (a >= 0) & (b >= 0)
    return ami(a[ok][::step], b[ok][::step]) if ok.sum() > 100 else np.nan


def main():
    sel = pd.read_csv(L.TABLES / "001_selected_sessions.csv")
    sids = list(sel.session_id)
    coh = dict(zip(sel.session_id, sel.reward_group))
    Z, act, nolick = {}, {}, {}
    for sid in sids:
        sg = BS.session_grid(sid)
        t, lk = sg["t"], sg["licks"]
        j = np.clip(np.searchsorted(lk, t), 1, len(lk) - 1)
        nolick[sid] = np.minimum(np.abs(t - lk[j - 1]), np.abs(lk[j] - t)) > 1.0
        act[sid] = sg["epoch"] != "other"
        for m, v in ROWS:
            z = BS.load_states(sid, m, v)
            if z is not None:
                Z[(sid, m, v)] = np.where(act[sid], z, -1)
    # agreement
    A = np.full((len(ROWS), len(ROWS), len(sids)), np.nan)
    for k, sid in enumerate(sids):
        for i, j in itertools.combinations(range(len(ROWS)), 2):
            a, b = Z.get((sid, *ROWS[i])), Z.get((sid, *ROWS[j]))
            if a is not None and b is not None:
                A[i, j, k] = A[j, i, k] = subsample_ami(a, b)
    Am = np.nanmean(A, 2)
    pd.DataFrame(Am, index=[LABEL[r] for r in ROWS], columns=[LABEL[r] for r in ROWS]).to_csv(L.TABLES / "008_ami_methods_views.csv")
    # seed stability
    seed_rows = []
    for v in L.VIEWS:
        seeds = BS.kpms_seeds(v)
        for s1, s2 in itertools.combinations(seeds, 2):
            for sid in sids:
                a, b = BS.load_states(sid, "kpms", v, s1), BS.load_states(sid, "kpms", v, s2)
                seed_rows.append(dict(view=v, seed_a=s1, seed_b=s2, session_id=sid,
                                      ami=subsample_ami(np.where(act[sid], a, -1), np.where(act[sid], b, -1))))
    seed_df = pd.DataFrame(seed_rows)
    seed_df.to_csv(L.TABLES / "008_kpms_seed_ami.csv", index=False)
    seed_meta = []
    for v in L.VIEWS:
        for s in BS.kpms_seeds(v):
            d = json.load(open(BS.state_file(sids[0], "kpms", v, s).parent / "done.json"))
            seed_meta.append(dict(view=v, seed=s, median_ms=d["median_ms"], n_syll_1pct=d["n_syll_1pct"],
                                  kappa_ar=d["kappa_ar"], kappa_full=d["kappa_full"]))
    seed_meta = pd.DataFrame(seed_meta)
    # R+ vs R- usage
    urows = []
    for m, v in ROWS:
        allz = np.concatenate([Z[(s, m, v)] for s in sids if (s, m, v) in Z] or [np.array([-1])])
        allz = allz[allz >= 0]
        if not len(allz):
            continue
        states = [1, 2, 3, 4, 5] if m == "lightning_action" else list(np.argsort(-np.bincount(allz))[:8])
        for sid in sids:
            z = Z.get((sid, m, v))
            if z is None:
                continue
            for scope, msk in (("all frames", z >= 0), ("no-lick frames", (z >= 0) & nolick[sid])):
                n = msk.sum()
                for s in states:
                    urows.append(dict(method=m, view=v, state=(BS.LA_CLASSES[s] if m == "lightning_action" else f"{'S' if m == 'kpms' else 'A'}{s}"), session_id=sid, cohort=coh[sid],
                                      scope=scope, usage=float(np.sum(z[msk] == s) / max(n, 1)), n_frames=int(n)))
    U = pd.DataFrame(urows)
    U.to_csv(L.TABLES / "008_usage_per_session.csv", index=False)
    trows = []
    for (m, v, st, sc), g in U.groupby(["method", "view", "state", "scope"], sort=False):
        a, b = g[g.cohort == "R+"].usage, g[g.cohort == "R-"].usage
        if len(a) < 2 or len(b) < 2:
            continue
        trows.append(dict(method=m, view=v, state=st, scope=sc, mean_Rplus=a.mean(), mean_Rminus=b.mean(),
                          p_mannwhitney=stats.mannwhitneyu(a, b).pvalue, p_welch=stats.ttest_ind(a, b, equal_var=False).pvalue,
                          n_Rplus=len(a), n_Rminus=len(b)))
    T = pd.DataFrame(trows)
    # session specificity: share of each state's frames that come from its dominant session (8 sessions: 1/8 = even)
    spec = []
    for m, v in ROWS:
        C = {sid: np.bincount(Z[(sid, m, v)][Z[(sid, m, v)] >= 0], minlength=120)[:120] for sid in sids if (sid, m, v) in Z}
        if not C:
            continue
        M = np.array(list(C.values()), float)
        Mn = M / M.sum(1, keepdims=True)  # per-session usage
        tot = M.sum(0)
        used = np.flatnonzero(tot / tot.sum() > 0.005)
        share = Mn[:, used].max(0) / Mn[:, used].sum(0)  # share of the usage-normalised state frequency
        spec.append(dict(method=m, view=v, n_states=len(used), median_max_share=float(np.median(share)),
                         frac_states_one_session=float(np.mean(share > 0.8)),
                         frac_frames_in_one_session_states=float(tot[used][share > 0.8].sum() / tot.sum()),
                         n_sessions_using_median=float(np.median((Mn[:, used] > 0.005).sum(0)))))
    spec = pd.DataFrame(spec)
    spec.to_csv(L.TABLES / "008_state_session_specificity.csv", index=False)
    T.to_csv(L.TABLES / "008_rplus_rminus_exploratory.csv", index=False)
    f1f = list((BS.KR / "lightning_action").glob("*/f1_heldout_fold*.csv"))
    f1 = pd.concat([pd.read_csv(f) for f in f1f]) if f1f else pd.DataFrame()
    if len(f1):
        f1.to_csv(L.TABLES / "008_lightning_action_f1_per_mouse.csv", index=False)

    final, tmp = L.REPORTS / "summary_behaviour_motifs.pdf", L.REPORTS / ".summary_behaviour_motifs.tmp.pdf"
    with PdfPages(tmp) as pdf:
        fig = plt.figure(figsize=(7.4, 9.8))
        ax = fig.add_axes([0.08, 0.68, 0.36, 0.26])
        im = ax.imshow(Am, cmap="magma", vmin=0, vmax=max(0.05, np.nanmax(Am)))
        ax.set_xticks(range(len(ROWS)))
        ax.set_xticklabels([LABEL[r] for r in ROWS], rotation=45, ha="right")
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([LABEL[r] for r in ROWS])
        for i in range(len(ROWS)):
            for j in range(len(ROWS)):
                if i != j and np.isfinite(Am[i, j]):
                    ax.text(j, i, f"{Am[i, j]:.2f}", ha="center", va="center", color="w" if Am[i, j] < 0.6 * np.nanmax(Am) else "k", fontsize=5)
        plt.colorbar(im, ax=ax, fraction=0.04).set_label("adjusted mutual information")
        ax.set_title(f"A  method / view agreement (mean over {len(sids)} sessions, active block)", loc="left")
        ax = fig.add_axes([0.58, 0.72, 0.38, 0.22])
        for i, v in enumerate(L.VIEWS):
            d = seed_df[seed_df.view == v]
            ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.15, 0.15, len(d)), d.ami, s=6, color="#555555")
            if len(d):
                ax.errorbar(i + 0.3, d.ami.mean(), d.ami.std(), fmt="o", color="k", ms=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{v} view\n" + "\n".join(f"s{r.seed}: {r.n_syll_1pct} syll / {r.median_ms:.0f} ms"
                                                       for r in seed_meta[seed_meta.view == v].itertuples()) for v in L.VIEWS], fontsize=5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("AMI between kpms seeds (per session x seed pair)")
        ax.set_title("B  keypoint-MoSeq seed stability", loc="left")
        # sweeps
        ax = fig.add_axes([0.08, 0.42, 0.36, 0.18])
        for v, c in zip(L.VIEWS, ("#444444", "#1f77b4")):
            ks = sorted((BS.KR / "kpms" / v / BS.selected()["kpms"][v]["norm"]).glob("sweep_ns*_k*.json"))
            if ks:
                d = pd.DataFrame([json.load(open(f)) for f in ks])
                for ns, g in d.groupby("ns"):
                    g = g.sort_values("kappa")
                    ax.plot(g.kappa, g.median_ms, "o-", color=c, alpha=0.5 + 0.5 * (ns == d.ns.max()), label=f"kpms AR stage, {v}, {ns} states")
            ks = sorted((BS.KR / "arhmm" / v).glob("sweep_ns*_k*.json"))
            if ks:
                d = pd.DataFrame([json.load(open(f)) for f in ks])
                for ns, g in d.groupby("ns"):
                    g = g.sort_values("kappa")
                    ax.plot(g.kappa, g.median_ms, "s--", color=c, alpha=0.5 + 0.5 * (ns == d.ns.max()), label=f"AR-HMM, {v}, {ns} states")
        ax.axhspan(150, 500, color="#dddddd", lw=0, edgecolor="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("kappa (stickiness)")
        ax.set_ylabel("median state duration (ms)")
        ax.legend(frameon=False, fontsize=5)
        ax.set_title("C  kappa x number-of-states sweeps at 200 Hz (grey = target 150-500 ms)", loc="left")
        ax = fig.add_axes([0.58, 0.42, 0.38, 0.18])
        if len(f1):
            cls = ["still", "whisking", "nose", "jaw", "lick"]
            for i, v in enumerate(L.VIEWS):
                for j, c in enumerate(cls):
                    q = f1[(f1.view == v) & (f1.cls == c)]
                    x = j + (i - 0.5) * 0.38
                    ax.bar(x, q.f1.mean(), 0.36, color=["#444444", "#1f77b4"][i], lw=0, alpha=0.5, label=f"{v} view" if j == 0 else None)
                    ax.scatter(np.full(len(q), x), q.f1, s=4, color=[L.COH_COL[coh[sid]] for sid in q.session_id], lw=0, zorder=3)
            ax.set_xticks(range(len(cls)))
            ax.set_xticklabels(cls)
            ax.set_ylim(0, 1.02)
            ax.legend(frameon=False)
        ax.set_ylabel("F1 vs heuristic labels (held-out mouse)")
        ax.set_title("D  lightning-action, leave-one-mouse-out (dots = mice, R+ green / R- magenta);\n"
                     "F1 vs heuristic labels is circular (labels from the same signals)", loc="left", fontsize=6)
        # R+ vs R-: smallest p per method x view
        ax = fig.add_axes([0.08, 0.02, 0.88, 0.33])
        ax.axis("off")
        if len(T):
            T["p_min"] = T[["p_mannwhitney", "p_welch"]].min(1)
            top = T.sort_values("p_min").groupby(["method", "view", "scope"]).head(1).sort_values(["method", "view", "scope"])
            txt = ["E  EXPLORATORY R+ vs R- state usage (session = mouse, n = 4 vs 4, underpowered; Mann-Whitney + Welch,",
                   "   uncorrected). State with the smallest p per method x view x scope (cf. F: kpms / AR-HMM states are partly session-specific):"]
            for r in top.itertuples():
                txt.append(f"{LABEL[(r.method, r.view)]:<16} {r.scope:<15} {r.state:<10} R+ {r.mean_Rplus:.3f}  R- {r.mean_Rminus:.3f}"
                           f"   p_MW {r.p_mannwhitney:.3f}  p_Welch {r.p_welch:.3f}")
            txt.append(f"(min possible two-sided Mann-Whitney p with 4 vs 4 = {stats.mannwhitneyu([1, 2, 3, 4], [5, 6, 7, 8]).pvalue:.3f}; "
                       f"{len(T)} comparisons;")
            txt.append(" full table tables/008_rplus_rminus_exploratory.csv)")
            txt.append("")
            txt.append("F  Session specificity (states > 0.5 % usage): share of a state's usage-normalised frequency in its dominant")
            txt.append("   session (1/8 = evenly shared); kpms / l-action keypoints centred and AR-HMM features z-scored per session")
            for r in spec.itertuples():
                txt.append(f"{LABEL[(r.method, r.view)]:<16} states {r.n_states:>3}  median dominant share {r.median_max_share:.2f}  "
                           f">80 % in one session: {100 * r.frac_states_one_session:.0f}% of states, "
                           f"{100 * r.frac_frames_in_one_session_states:.0f}% of frames; sessions/state {r.n_sessions_using_median:.1f}/8")
            ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=5.2)
        fig.suptitle("ssl-behaviour-motifs pilot: cross-session summary (8 day-0 whisker sessions)", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)
        # page 2: usage dots per cohort
        fig, axs = plt.subplots(6, 2, figsize=(7.4, 9.8))
        for r, (m, v) in enumerate(ROWS):
            for c, sc in enumerate(("all frames", "no-lick frames")):
                ax = axs[r, c]
                g = U[(U.method == m) & (U.view == v) & (U.scope == sc)]
                sts = list(dict.fromkeys(g.state))
                for k, (cname, off) in enumerate((("R+", -0.15), ("R-", 0.15))):
                    gg = g[g.cohort == cname]
                    x = np.array([sts.index(s) for s in gg.state]) + off
                    ax.scatter(x, gg.usage, s=5, color=L.COH_COL[cname], label=cname if (r == 0 and c == 0) else None, lw=0)
                ax.set_xticks(range(len(sts)))
                ax.set_xticklabels(sts, fontsize=5)
                ax.set_ylabel("usage (fraction)")
                ax.set_title(f"{LABEL[(m, v)]}: {sc}", fontsize=6)
        axs[0, 0].legend(frameon=False)
        fig.suptitle("Exploratory: per-session state usage by cohort (active block); right = frames > 1 s from any lick", fontsize=7)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        if (L.TABLES / "009_metastates.csv").exists():
            BA.summary_pages(pdf, sids, coh)  # meta-states, contingency, session mapping, movie index
            BC.checks_pages(pdf, sids, coh)  # dropout, normalisation, held-out stability, drift, power
    try:  # a PDF open in a viewer on Windows can lock the NAS file
        os.replace(tmp, final)
    except OSError:
        os.replace(tmp, final.with_name(final.stem + "_new.pdf"))
        print("locked, written as", final.stem + "_new.pdf")
    json.dump(dict(script="008_summary.py", sessions=sids, ami_subsample_step=20, nolick_s=1.0,
                   tests="Mann-Whitney U + Welch t, uncorrected, exploratory", versions=L.versions()),
              open(L.RES / "provenance" / "008.json", "w"), indent=1)
    print(pd.DataFrame(Am, index=[LABEL[r] for r in ROWS], columns=[LABEL[r] for r in ROWS]).round(2))
    print(seed_df.groupby("view").ami.describe())
    print(T.sort_values("p_mannwhitney").head(10).to_string() if len(T) else "no tests")
    print(spec.to_string())


if __name__ == "__main__":
    main()
